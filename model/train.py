import torch
import torch.nn as nn
import torch.optim as optim
from dataset import create_dataloaders
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import zarr
from models import AttentionUNet, RandomForestBaseline, ToyModel
#from dataset import global_stats 

# Config

class Config:
    zarr_dir = "/discover/nobackup/cmbreen/gap-filling-data/zarr_chunks"
    batch_size = 16
    patch_size = 256
    stride = 128
    num_workers = 4
    normalize = True
    random_crop_train = False

    epochs = 5
    lr = 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = "./checkpoints"
    unet_name = "attention_unet_final.pth"
    rf_name = "random_forest_baseline.joblib"

# Metrics

## should we add the masks here?

# def compute_metrics(pred, target, masks):
#     mae = torch.mean(torch.abs(pred - target)).item()
#     rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
#     return mae, rmse

def compute_metrics(pred, target, mask):
    """
    Compute MAE and RMSE only over valid pixels indicated by mask.
    """
    masked_diff = (pred - target) * mask  # zero out invalid pixels
    num_valid = mask.sum() + 1e-8  # avoid division by zero

    mae = masked_diff.abs().sum() / num_valid
    rmse = torch.sqrt((masked_diff ** 2).sum() / num_valid)

    return mae.item(), rmse.item()

# Training

def masked_loss(predictions, targets, mask, global_stats=None): #nn.L1Loss(reduction='none')):
    ## we are only computing the loss on the valid Y pixels 
    """Compute loss only on Y valid pixels."""

    ### new code to weight the larger swe values ###
    squared_error = (predictions - targets) ** 2 
    # This would be the max for each batch
    #max_target = torch.max(torch.abs(targets[mask > 0])) + 1e-8 ## weighs the larger pixel in the patch 

    ### this is the global Mean/Max
    Y_mean = global_stats['Y_mean']
    Y_std = global_stats['Y_std']
    Y_max_meters = global_stats['Y_max']
    Y_max_normalized = (Y_max_meters - Y_mean) / (Y_std + 1e-8)
    # targets_denorm = targets * Y_std + Y_mean  # Back to meters

    # ## because the global i
    # targets_denorm = targets * Y_std + Y_mean  # log-space
    # targets_denorm = torch.expm1(targets_denorm)  # meters
    # Use GLOBAL max (converted to tensor)
    global_max = torch.tensor(Y_max_normalized, device=targets.device, dtype=targets.dtype)
    
    # Linear weighting: weight = 1 + (target / global_max)
    # 0m SWE → weight = 1.0
    # max SWE (e.g., 2m) → weight = 2.0
    #weights = 1.0 + (torch.abs(targets) / (global_max + 1e-8))

    # element_loss = loss_fn(predictions, targets)
    # masked_loss_vals = element_loss * mask
    # num_valid = mask.sum() + 1e-8
    # return masked_loss_vals.sum() / num_valid

    weights = 1.0 + (torch.abs(targets) / (torch.abs(global_max) + 1e-8))
    #weights = torch.exp(targets / global_max)
    # weights = 1.0 + torch.clamp(targets / global_max, 0, 10)
    # weighted_error = (predictions - targets)**2 * weights * mask
    # loss = weighted_error.sum() / (mask.sum() + 1e-8)
    # # # Apply weights and mask
    weighted_error = squared_error * weights * mask
    
    num_valid = mask.sum() + 1e-8
    return weighted_error.sum() / num_valid

def train():

    cfg = Config()
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)


    # Data

    dataloaders = create_dataloaders(
        zarr_dir=cfg.zarr_dir,
        batch_size=cfg.batch_size,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        num_workers=cfg.num_workers,
        normalize=cfg.normalize,
        random_crop_train=cfg.random_crop_train
    )

    # 1. Train Attention U-Net
    train_dataset = dataloaders['train'].dataset
    global_stats = train_dataset.global_stats

    print("Training Attention U-Net")


    model = ToyModel(in_channels=17).to(cfg.device)
    # Train this

    #AttentionUNet(in_channels=17, out_channels=1).to(cfg.device)
    criterion = nn.L1Loss()   # Better for SWE / regression problems
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float("inf")
    
    # Track losses for plotting
    train_losses = []
    val_losses = []

    for epoch in range(1, cfg.epochs + 1):

        print(f"\nEpoch {epoch}/{cfg.epochs}")

        # Train

        model.train()
        train_loss = 0.0

        for X, Y, metadata in tqdm(dataloaders['train']):

            X = X.to(cfg.device)
            Y = Y.to(cfg.device)

            optimizer.zero_grad()
            outputs = model(X)
            #loss = criterion(outputs, Y)
            Y_mask = metadata['Y_mask'].to(cfg.device)
            loss = masked_loss(outputs, Y, Y_mask, global_stats)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            valid_pixels = Y_mask.sum().item()
            total_pixels = Y_mask.numel()
            print("Valid pixel fraction:", valid_pixels / total_pixels)
            print("Outputs std:", outputs.std().item())
            print("Targets std (valid):", Y[Y_mask > 0].std().item())
            print("Mask fraction:", Y_mask.mean().item())

        train_loss /= len(dataloaders['train'])
        train_losses.append(train_loss)
        print(f"Train L1: {train_loss:.6f}")



        # Validation

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_rmse = 0.0

        with torch.no_grad():
            for X, Y, metadata in dataloaders['val']:
                X = X.to(cfg.device)
                Y = Y.to(cfg.device)
                Y_mask = metadata['Y_mask'].to(cfg.device)
                #global_stats = dataloaders['val'].global_stats

                outputs = model(X)
                #loss = criterion(outputs, Y)
                loss = masked_loss(outputs, Y, Y_mask, global_stats)  # ← Use masked loss!
                mae, rmse = compute_metrics(outputs, Y, Y_mask)

                val_loss += loss.item()
                val_mae += mae
                val_rmse += rmse

        val_loss /= len(dataloaders['val'])
        val_mae /= len(dataloaders['val'])
        val_rmse /= len(dataloaders['val'])
        
        val_losses.append(val_loss)

        print(f"Val L1: {val_loss:.6f}")
        print(f"Val MAE: {val_mae:.6f} | RMSE: {val_rmse:.6f}")

        scheduler.step(val_loss)
    
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6e}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(cfg.save_dir, cfg.unet_name)
            torch.save(model.state_dict(), save_path)
            print(f"Saved best U-Net to {save_path}")

    # Plot Training and Validation Loss

    print("Generating Loss Plot")
    
    epochs_range = range(1, cfg.epochs + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    plt.plot(epochs_range, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('L1 Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(cfg.save_dir, 'loss_curve.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved loss plot to {plot_path}")
    
    # Also save loss values to a text file
    loss_txt_path = os.path.join(cfg.save_dir, 'loss_values.txt')
    with open(loss_txt_path, 'w') as f:
        f.write("Epoch,Train_Loss,Val_Loss\n")
        for i, (train_l, val_l) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f"{i},{train_l:.6f},{val_l:.6f}\n")
    print(f"Saved loss values to {loss_txt_path}")

    # 2. Train Random Forest Baseline

    print("Training Random Forest Baseline")

    rf = RandomForestBaseline(n_estimators=100)

    # Subsample pixels to avoid RAM explosion
    rf.fit(dataloaders['train'], subsample=20)

    rf_path = os.path.join(cfg.save_dir, cfg.rf_name)
    rf.save(rf_path)

    print(f"Saved Random Forest to {rf_path}")

    # Feature importance
    importance = rf.feature_importance()
    print("\nFeature Importance:")
    for i, score in enumerate(importance):
        print(f"Channel {i}: {score:.4f}")

    # debug_results = debug_model_output(
    #     model=model,
    #     dataloader=dataloaders['val'],  # Check on validation set
    #     device=cfg.device,
    #     global_stats=global_stats
    # )


def check_zarr_validity(zarr_dir: str):
    """Check all zarr files for invalid target values."""
    print("\n" + "="*80)
    print("CHECKING ZARR FILES FOR DATA QUALITY ISSUES")
    print("="*80)
    
    zarr_files = list(Path(zarr_dir).glob("*.zarr"))
    
    for zarr_path in zarr_files[:10]:  # Check first 10
        z = zarr.open(str(zarr_path), mode='r')
        Y = np.array(z['Y'], dtype=np.float32)
        
        # Check for issues
        has_nan = np.isnan(Y).any()
        has_neg = (Y < 0).any()
        has_extreme = (Y > 10).any()  # >10m is suspicious
        
        if has_nan or has_neg or has_extreme:
            print(f"\n  {zarr_path.name}:")
            print(f"   NaN values: {np.isnan(Y).sum()}")
            print(f"   Negative values: {(Y < 0).sum()}")
            print(f"   Values > 10m: {(Y > 10).sum()}")
            print(f"   Min: {np.nanmin(Y):.2f}m, Max: {np.nanmax(Y):.2f}m")
            print(f"   Mean: {np.nanmean(Y[~np.isnan(Y)]):.2f}m")


def debug_model_output(model, dataloader, device, global_stats):
    """Comprehensive debugging of model behavior."""
    model.eval()
    
    all_preds_norm = []
    all_targets_norm = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 5:  # Just check first 5 batches
                break
                
            X, Y, metadata = batch
            X = X.to(device)
            mask = metadata['Y_mask']
            
            pred = model(X).cpu()
            
            all_preds_norm.append(pred[mask > 0])
            all_targets_norm.append(Y[mask > 0])
    
    all_preds_norm = torch.cat(all_preds_norm)
    all_targets_norm = torch.cat(all_targets_norm)
    
    # Denormalize
    pred_denorm = all_preds_norm.numpy() * global_stats['Y_std'] + global_stats['Y_mean']
    target_denorm = all_targets_norm.numpy() * global_stats['Y_std'] + global_stats['Y_mean']
    
    print("\n" + "="*60)
    print("MODEL OUTPUT DIAGNOSTIC")
    print("="*60)
    
    print("\nNORMALIZED (what model sees):")
    print(f"  Predictions: mean={all_preds_norm.mean():.4f}, std={all_preds_norm.std():.4f}")
    print(f"               min={all_preds_norm.min():.4f}, max={all_preds_norm.max():.4f}")
    print(f"  Targets:     mean={all_targets_norm.mean():.4f}, std={all_targets_norm.std():.4f}")
    print(f"               min={all_targets_norm.min():.4f}, max={all_targets_norm.max():.4f}")
    print(f"  Variance Ratio: {all_preds_norm.std() / (all_targets_norm.std() + 1e-8):.4f}")
    
    print("\nDENORMALIZED (real SWE in meters):")
    print(f"  Predictions: mean={pred_denorm.mean():.4f}m, std={pred_denorm.std():.4f}m")
    print(f"               min={pred_denorm.min():.4f}m, max={pred_denorm.max():.4f}m")
    print(f"  Targets:     mean={target_denorm.mean():.4f}m, std={target_denorm.std():.4f}m")
    print(f"               min={target_denorm.min():.4f}m, max={target_denorm.max():.4f}m")
    print(f"  Variance Ratio: {pred_denorm.std() / (target_denorm.std() + 1e-8):.4f}")
    
    print("\nGLOBAL STATS USED:")
    print(f"  Y_mean: {global_stats['Y_mean']:.4f}")
    print(f"  Y_std:  {global_stats['Y_std']:.4f}")
    
    print("="*60 + "\n")
    
    return {
        'pred_norm': all_preds_norm,
        'target_norm': all_targets_norm,
        'pred_denorm': pred_denorm,
        'target_denorm': target_denorm
    }


if __name__ == "__main__":
    
    # Run this before training
        zarr_dir = "/discover/nobackup/cmbreen/gap-filling-data/zarr_chunks"
        check_zarr_validity(zarr_dir)
        train()
   