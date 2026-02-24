import torch
import torch.nn as nn
import torch.optim as optim
from dataset import create_dataloaders
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from models import AttentionUNet, RandomForestBaseline


# ============================================================
# Config
# ============================================================

class Config:
    zarr_dir = "/discover/nobackup/cmbreen/gap-filling-data/zarr_chunks"
    batch_size = 16
    patch_size = 256
    stride = 128
    num_workers = 4
    normalize = True
    random_crop_train = False

    epochs = 20
    lr = 1e-3

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = "./checkpoints"
    unet_name = "attention_unet_final.pth"
    rf_name = "random_forest_baseline.joblib"


# ============================================================
# Metrics
# ============================================================

def compute_metrics(pred, target):
    mae = torch.mean(torch.abs(pred - target)).item()
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    return mae, rmse


# ============================================================
# Training
# ============================================================

def train():

    cfg = Config()
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Data
    # -------------------------
    dataloaders = create_dataloaders(
        zarr_dir=cfg.zarr_dir,
        batch_size=cfg.batch_size,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        num_workers=cfg.num_workers,
        normalize=cfg.normalize,
        random_crop_train=cfg.random_crop_train
    )

    # ============================================================
    # 1. Train Attention U-Net
    # ============================================================

    print("\n==============================")
    print("Training Attention U-Net")
    print("==============================")

    model = AttentionUNet(in_channels=17, out_channels=1).to(cfg.device)
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

        # -------------------------
        # Train
        # -------------------------
        model.train()
        train_loss = 0.0

        for X, Y, _ in tqdm(dataloaders['train']):
            X = X.to(cfg.device)
            Y = Y.to(cfg.device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(dataloaders['train'])
        train_losses.append(train_loss)
        print(f"Train L1: {train_loss:.6f}")

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_rmse = 0.0

        with torch.no_grad():
            for X, Y, _ in dataloaders['val']:
                X = X.to(cfg.device)
                Y = Y.to(cfg.device)

                outputs = model(X)
                loss = criterion(outputs, Y)

                mae, rmse = compute_metrics(outputs, Y)

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

    # ============================================================
    # Plot Training and Validation Loss
    # ============================================================
    
    print("\n==============================")
    print("Generating Loss Plot")
    print("==============================")
    
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

    # ============================================================
    # 2. Train Random Forest Baseline
    # ============================================================

    print("\n==============================")
    print("Training Random Forest Baseline")
    print("==============================")

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


if __name__ == "__main__":
        train()
