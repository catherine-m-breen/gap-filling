# eval.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from dataset import create_dataloaders, NUM_SNOW_CLASSES, SNOW_CLASSES
from models import AttentionUNet, RandomForestBaseline, ToyModel

# ============================================================
# Config
# ============================================================

class EvalConfig:
    zarr_dir = "/discover/nobackup/cmbreen/gap-filling-data/zarr_chunks"
    batch_size = 16
    patch_size = 256
    stride = 128
    num_workers = 4
    normalize = True
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint_dir = "./checkpoints"
    unet_name = "attention_unet_final.pth"
    rf_name = "random_forest_baseline.joblib"
    
    results_dir = "./results"
    
    # SWE bins for analysis (in mm)
    swe_bins = [0, 100, 300, 600, 1000]
    swe_labels = ['Low (0-100mm)', 'Medium (100-300mm)', 'High (300-600mm)', 'Very High (>600mm)']
    
    # Tree canopy bins (%)
    canopy_bins = [0, 25, 50, 75, 100]
    canopy_labels = ['0-25%', '25-50%', '50-75%', '75-100%']


# ============================================================
# Metrics Functions
# ============================================================

def compute_metrics_masked(pred, target, mask):
    """Compute metrics only on valid pixels."""
    pred_valid = pred[mask]
    target_valid = target[mask]
    
    if len(pred_valid) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'n_pixels': 0}
    
    mae = np.mean(np.abs(pred_valid - target_valid))
    rmse = np.sqrt(np.mean((pred_valid - target_valid) ** 2))
    ss_res = np.sum((target_valid - pred_valid) ** 2)
    ss_tot = np.sum((target_valid - np.mean(target_valid)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'n_pixels': len(pred_valid)}


def compute_total_swe(swe_array, mask):
    """Compute total SWE (sum of valid pixels)."""
    return np.sum(swe_array[mask])


def plot_pred_vs_target(predictions, targets, masks, save_path):
    """Scatter plot: predicted vs actual SWE."""
    valid_idx = masks
    pred_valid = predictions[valid_idx]
    target_valid = targets[valid_idx]
    
    if len(pred_valid) > 50000:
        idx = np.random.choice(len(pred_valid), 50000, replace=False)
        pred_valid = pred_valid[idx]
        target_valid = target_valid[idx]
    
    plt.figure(figsize=(10, 10))
    plt.hexbin(target_valid, pred_valid, gridsize=50, cmap='viridis', mincnt=1)
    plt.colorbar(label='Pixel Count')
    min_val = min(target_valid.min(), pred_valid.min())
    max_val = max(target_valid.max(), pred_valid.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')
    plt.xlabel('Actual SWE (mm)')
    plt.ylabel('Predicted SWE (mm)')
    plt.title('Predicted vs Actual SWE')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved pred vs target plot to {save_path}")


# ============================================================
# Data Collection WITH DENORMALIZATION
# ============================================================

def collect_predictions(model, dataloader, device, model_type='unet', global_stats=None, undo_log=False):
    """
    Collect predictions, targets, masks, and features from model.
    Denormalizes outputs consistently.
    """
    if model_type == 'unet':
        model.eval()
    
    all_preds = []
    all_targets = []
    all_masks = []
    all_features = []
    
    with torch.no_grad():
        for X, Y, metadata in tqdm(dataloader, desc="Batches"):
            X = X.to(device)
            Y = Y.to(device)
            
            # -------------------------
            # Model Prediction
            # -------------------------
            if model_type == 'unet':
                pred = model(X)
            else:
                pred_np = model.predict(X.cpu().numpy())
                pred = torch.from_numpy(pred_np).float().to(device)
            
            # -------------------------
            # Extract masks
            # -------------------------
            Y_mask = metadata['Y_mask'].cpu().numpy()  # (B,1,H,W)
            X_mask = metadata['X_mask'].cpu().numpy()  # (B,17,H,W)
            
            pred_np = pred.cpu().numpy()
            target_np = Y.cpu().numpy()
            X_np = X.cpu().numpy()
            
            # -------------------------
            # Denormalize target & predictions
            # -------------------------
            if global_stats is not None:
                Y_mean = global_stats['Y_mean']
                Y_std = global_stats['Y_std']
                pred_np = pred_np * Y_std + Y_mean
                target_np = target_np * Y_std + Y_mean
                
                if undo_log:
                    pred_np = np.expm1(pred_np)
                    target_np = np.expm1(target_np)
                
                # Denormalize continuous X channels
                X_mean = global_stats['X_mean'][:, None, None, None]
                X_std = global_stats['X_std'][:, None, None, None]
                continuous_channels = [8, 9, 10, 11, 12, 13, 14]
                for c in continuous_channels:
                    X_np[:, c] = X_np[:, c] * X_std[c] + X_mean[c]
            
            # Convert SWE to mm
            pred_np *= 1000
            target_np *= 1000
            
            # Flatten
            B, C, H, W = pred_np.shape
            all_preds.append(pred_np.reshape(B, -1))
            all_targets.append(target_np.reshape(B, -1))
            all_masks.append(Y_mask.reshape(B, -1).astype(bool))
            all_features.append(X_np.reshape(B, X_np.shape[1], -1))
    
    # Concatenate
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    all_masks = np.concatenate(all_masks, axis=0).flatten()
    
    all_features = np.concatenate(all_features, axis=0)
    all_features = all_features.reshape(all_features.shape[1], -1)  # (17, N_pixels)
    
    print(f"\nCollected {len(all_preds):,} pixels, {all_masks.sum():,} valid")
    
    # Debug prints
    if all_masks.sum() > 0:
        valid_preds = all_preds[all_masks]
        valid_targets = all_targets[all_masks]
        print(f"Predictions (mm): min={valid_preds.min():.2f}, max={valid_preds.max():.2f}, mean={valid_preds.mean():.2f}")
        print(f"Targets (mm): min={valid_targets.min():.2f}, max={valid_targets.max():.2f}, mean={valid_targets.mean():.2f}")
    
    return {'predictions': all_preds, 'targets': all_targets, 'masks': all_masks, 'features': all_features}


# ============================================================
# Analysis Functions
# ============================================================

def analyze_by_swe_bins(predictions, targets, masks, bins, labels):
    results = []
    for i in range(len(bins)-1):
        in_bin = (targets >= bins[i]) & (targets < bins[i+1]) & masks
        metrics = compute_metrics_masked(predictions, targets, in_bin)
        metrics.update({'swe_range': labels[i], 'bin_min': bins[i], 'bin_max': bins[i+1]})
        results.append(metrics)
    return pd.DataFrame(results)


def analyze_by_snow_class(predictions, targets, masks, snow_onehot_channels):
    results = []
    snow_class_idx = np.argmax(snow_onehot_channels, axis=0)
    snow_class_values = np.array([SNOW_CLASSES[i] for i in snow_class_idx])
    
    for cls_idx, cls_value in enumerate(SNOW_CLASSES):
        in_class = (snow_class_values == cls_value) & masks
        if in_class.sum() < 10: continue
        metrics = compute_metrics_masked(predictions, targets, in_class)
        metrics.update({'snow_class': int(cls_value), 'snow_class_name': f"Class_{cls_value}"})
        results.append(metrics)
    return pd.DataFrame(results)


def analyze_by_canopy_cover(predictions, targets, masks, canopy_channel, bins, labels):
    results = []
    for i in range(len(bins)-1):
        in_bin = (canopy_channel >= bins[i]) & (canopy_channel < bins[i+1]) & masks
        metrics = compute_metrics_masked(predictions, targets, in_bin)
        metrics.update({'canopy_range': labels[i], 'bin_min': bins[i], 'bin_max': bins[i+1]})
        results.append(metrics)
    return pd.DataFrame(results)


def plot_canopy_vs_error(predictions, targets, masks, canopy_channel, save_path):
    valid_idx = masks
    canopy_valid = canopy_channel[valid_idx]
    error = np.abs(predictions[valid_idx] - targets[valid_idx])
    
    if len(canopy_valid) > 50000:
        idx = np.random.choice(len(canopy_valid), 50000, replace=False)
        canopy_valid = canopy_valid[idx]
        error = error[idx]
    
    plt.figure(figsize=(10,6))
    plt.hexbin(canopy_valid, error, gridsize=50, cmap='viridis', mincnt=1)
    plt.colorbar(label='Pixel Count')
    plt.xlabel('Tree Canopy Cover (%)')
    plt.ylabel('Absolute Error (mm SWE)')
    plt.title('Prediction Error vs Tree Canopy Cover')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved canopy vs error plot to {save_path}")


# ============================================================
# Main Evaluation
# ============================================================

def evaluate():
    cfg = EvalConfig()
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    
    dataloaders = create_dataloaders(
        zarr_dir=cfg.zarr_dir,
        batch_size=cfg.batch_size,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        num_workers=cfg.num_workers,
        normalize=cfg.normalize,
        random_crop_train=False
    )
    
    test_dataset = dataloaders['val'].dataset
    global_stats = test_dataset.global_stats
    
    # ---------------------------------------------
    # Load U-Net
    # ---------------------------------------------
    unet_path = Path(cfg.checkpoint_dir) / cfg.unet_name
    if not unet_path.exists():
        print(f"ERROR: U-Net checkpoint not found at {unet_path}")
        return
    
    unet = ToyModel(in_channels=17).to(cfg.device)
    unet.load_state_dict(torch.load(unet_path, map_location=cfg.device))
    print(f"Loaded U-Net from {unet_path}")
    
    # Collect predictions
    unet_data = collect_predictions(
        unet,
        dataloaders['val'],
        cfg.device,
        model_type='unet',
        global_stats=global_stats,
        undo_log=False  # change to True if you used log1p
    )
    
    plot_pred_vs_target(
        unet_data['predictions'],
        unet_data['targets'],
        unet_data['masks'],
        Path(cfg.results_dir) / 'unet_pred_vs_target.png'
    )
    
    # Overall metrics
    overall = compute_metrics_masked(unet_data['predictions'], unet_data['targets'], unet_data['masks'])
    print(f"U-Net MAE={overall['mae']:.2f}, RMSE={overall['rmse']:.2f}, R²={overall['r2']:.4f}, valid pixels={overall['n_pixels']:,}")
    
    # Feature indices
    CANOPY_CHANNEL = 8
    SNOW_ONEHOT_CHANNELS = list(range(NUM_SNOW_CLASSES))
    canopy = unet_data['features'][CANOPY_CHANNEL, :]
    snow_onehot = unet_data['features'][SNOW_ONEHOT_CHANNELS, :]
    
    # Analyses
    swe_results = analyze_by_swe_bins(unet_data['predictions'], unet_data['targets'], unet_data['masks'], cfg.swe_bins, cfg.swe_labels)
    swe_results.to_csv(Path(cfg.results_dir) / 'unet_swe_bins.csv', index=False)
    
    snow_results = analyze_by_snow_class(unet_data['predictions'], unet_data['targets'], unet_data['masks'], snow_onehot)
    snow_results.to_csv(Path(cfg.results_dir) / 'unet_snow_classes.csv', index=False)
    
    canopy_results = analyze_by_canopy_cover(unet_data['predictions'], unet_data['targets'], unet_data['masks'], canopy, cfg.canopy_bins, cfg.canopy_labels)
    canopy_results.to_csv(Path(cfg.results_dir) / 'unet_canopy_bins.csv', index=False)
    
    plot_canopy_vs_error(unet_data['predictions'], unet_data['targets'], unet_data['masks'], canopy, Path(cfg.results_dir) / 'unet_canopy_vs_error.png')
    
    # ---------------------------------------------
    # Random Forest (if available)
    # ---------------------------------------------
    rf_path = Path(cfg.checkpoint_dir) / cfg.rf_name
    if rf_path.exists():
        rf = RandomForestBaseline()
        rf.load(str(rf_path))
        print(f"Loaded RF from {rf_path}")
        
        rf_data = collect_predictions(rf, dataloaders['val'], cfg.device, model_type='rf', global_stats=global_stats)
        
        rf_overall = compute_metrics_masked(rf_data['predictions'], rf_data['targets'], rf_data['masks'])
        print(f"RF MAE={rf_overall['mae']:.2f}, RMSE={rf_overall['rmse']:.2f}, R²={rf_overall['r2']:.4f}")
        
        rf_canopy = rf_data['features'][CANOPY_CHANNEL, :]
        rf_snow_onehot = rf_data['features'][SNOW_ONEHOT_CHANNELS, :]
        rf_snow_results = analyze_by_snow_class(rf_data['predictions'], rf_data['targets'], rf_data['masks'], rf_snow_onehot)
        rf_snow_results.to_csv(Path(cfg.results_dir) / 'rf_snow_classes.csv', index=False)
    
    print(f"\nEvaluation complete. Results saved to {cfg.results_dir}/")


if __name__ == "__main__":
    evaluate()