# eval.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json

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
    
    # SWE bins for analysis (in mm, AFTER denormalization)
    swe_bins = [0, 100, 300, 600, 10000]
    swe_labels = ['Low (0-100mm)', 'Medium (100-300mm)', 'High (300-600mm)', 'Very High (>600mm)']
    
    # Tree canopy bins (%)
    canopy_bins = [0, 25, 50, 75, 100]
    canopy_labels = ['0-25%', '25-50%', '50-75%', '75-100%']


# ============================================================
# Metrics Functions
# ============================================================

def compute_metrics_masked(pred, target, mask):
    """
    Compute metrics only on valid (non-masked) pixels.
    Where are the masks coming from? 
    """
    pred_valid = pred[mask]
    target_valid = target[mask]
    
    if len(pred_valid) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'n_pixels': 0}
    
    # MAE
    mae = np.mean(np.abs(pred_valid - target_valid))
    
    # RMSE
    rmse = np.sqrt(np.mean((pred_valid - target_valid) ** 2))
    
    # R²
    ss_res = np.sum((target_valid - pred_valid) ** 2)
    ss_tot = np.sum((target_valid - np.mean(target_valid)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'n_pixels': len(pred_valid)
    }


def compute_total_swe(swe_array, mask):
    """Compute total SWE volume (sum of all valid pixels)."""
    return np.sum(swe_array[mask])


def plot_pred_vs_target(predictions, targets, masks, save_path):
    """Scatter plot: predicted vs actual SWE."""
    valid_idx = masks > 0
    
    pred_valid = predictions[valid_idx]
    target_valid = targets[valid_idx]
    
    # Subsample if too many points
    if len(pred_valid) > 50000:
        idx = np.random.choice(len(pred_valid), 50000, replace=False)
        pred_valid = pred_valid[idx]
        target_valid = target_valid[idx]
    
    plt.figure(figsize=(10, 10))
    plt.hexbin(target_valid, pred_valid, gridsize=50, cmap='viridis', mincnt=1)
    plt.colorbar(label='Pixel Count')
    
    # Add 1:1 line
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

def collect_predictions(model, dataloader, device, model_type='unet', global_stats=None):
    """
    Run model on entire dataloader and collect all predictions, targets, and metadata.
    Denormalizes outputs to original scale.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the dataset
        device: torch device
        model_type: 'unet' or 'rf'
        global_stats: Dict with 'X_mean', 'X_std', 'Y_mean', 'Y_std' for denormalization
    
    Returns:
        dict with arrays for predictions, targets, masks, and feature channels (all denormalized)
    """
    if model_type == 'unet':
        model.eval()
    # No eval() for Random Forest
    
    all_preds = []
    all_targets = []
    all_masks = []
    all_features = []
    
    print(f"Collecting predictions from {len(dataloader)} batches...")
    
    with torch.no_grad():
        for X, Y, metadata in tqdm(dataloader):
            X = X.to(device)
            Y = Y.to(device)
            
            # Predict
            if model_type == 'unet':
                pred = model(X)
            else:  # Random Forest
                pred = model.predict(X.cpu())
                pred = pred.to(device)
            
            # Get masks from metadata (TRUE validity masks!)
            Y_mask = metadata['Y_mask'].cpu().numpy()  # (B, 1, H, W)
            X_mask = metadata['X_mask'].cpu().numpy()  # (B, 17, H, W) ← NOW 17 CHANNELS!
            
            # Move to CPU
            pred_np = pred.cpu().numpy()      # (B, 1, H, W)
            target_np = Y.cpu().numpy()       # (B, 1, H, W)
            X_np = X.cpu().numpy()            # (B, 17, H, W) ← NOW 17 CHANNELS!
            
            # ========================================
            # DENORMALIZE predictions and targets
            # ========================================
            if global_stats is not None:
                Y_mean = global_stats['Y_mean']
                Y_std = global_stats['Y_std']
                
                # Denormalize Y: y_original = y_normalized * std + mean
                pred_np = pred_np * Y_std + Y_mean
                target_np = target_np * Y_std + Y_mean
                
                # Denormalize continuous X channels
                X_mean = global_stats['X_mean'][:, None, None, None]  # (17, 1, 1, 1)
                X_std = global_stats['X_std'][:, None, None, None]
                
                # Only denormalize continuous channels (AFTER one-hot encoding)
                # Channels 8, 9, 10-13, 14 are continuous (canopy, elevation, 4 TBs, NDSI)
                continuous_channels = [8, 9, 10, 11, 12, 13, 14]
                for c in continuous_channels:
                    X_np[:, c] = X_np[:, c] * X_std[c] + X_mean[c]
                
                # Categorical channels (0-6: snow one-hot, 7: land, 15-16: masks) 
                # were "normalized" with mean=0, std=1, so they stay as-is
            
            # Convert SWE to mm for analysis (already in meters)
            pred_np = pred_np * 1000
            target_np = target_np * 1000
            
            # Flatten spatial dimensions
            B, C, H, W = pred_np.shape
            pred_flat = pred_np.reshape(B, -1)      # (B, H*W)
            target_flat = target_np.reshape(B, -1)  # (B, H*W)
            mask_flat = Y_mask.reshape(B, -1)       # (B, H*W) - boolean
            X_flat = X_np.reshape(B, X_np.shape[1], -1)  # (B, 17, H*W)
            
            all_preds.append(pred_flat)
            all_targets.append(target_flat)
            all_masks.append(mask_flat)
            all_features.append(X_flat)
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    all_masks = np.concatenate(all_masks, axis=0).flatten().astype(bool)
    all_features = np.concatenate(all_features, axis=0)
    all_features = all_features.reshape(all_features.shape[1], -1)  # (17, N_pixels)
    
    print(f"\nCollected {len(all_preds)} total pixels, {all_masks.sum():,} valid")
    
    # Print sample values for debugging
    if all_masks.sum() > 0:
        valid_preds = all_preds[all_masks]
        valid_targets = all_targets[all_masks]
        print(f"\nDenormalized sample values:")
        print(f"  Predictions (mm): min={valid_preds.min():.2f}, max={valid_preds.max():.2f}, mean={valid_preds.mean():.2f}")
        print(f"  Targets (mm): min={valid_targets.min():.2f}, max={valid_targets.max():.2f}, mean={valid_targets.mean():.2f}")
        
        # Check snow one-hot encoding (channels 0-6)
        print(f"\nSnow class distribution (one-hot encoded):")
        for i in range(NUM_SNOW_CLASSES):
            snow_channel = all_features[i, :]
            valid_snow = snow_channel[all_masks]
            active_pixels = (valid_snow > 0.5).sum()  # Count where this class is active
            pct = 100 * active_pixels / all_masks.sum()
            print(f"    Class {SNOW_CLASSES[i]}: {active_pixels:,} pixels ({pct:.2f}%)")
    
    return {
        'predictions': all_preds,
        'targets': all_targets,
        'masks': all_masks,
        'features': all_features
    }


# ============================================================
# Analysis Functions
# ============================================================

def analyze_by_swe_bins(predictions, targets, masks, bins, labels):
    """Compute metrics for different SWE value ranges."""
    results = []
    
    for i in range(len(bins) - 1):
        bin_min = bins[i]
        bin_max = bins[i + 1]
        
        # Find pixels in this SWE range
        in_bin = (targets >= bin_min) & (targets < bin_max) & masks
        
        metrics = compute_metrics_masked(predictions, targets, in_bin)
        metrics['swe_range'] = labels[i]
        metrics['bin_min'] = bin_min
        metrics['bin_max'] = bin_max
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def analyze_by_snow_class(predictions, targets, masks, snow_onehot_channels):
    """
    Compute metrics for each snow map class.
    
    Args:
        predictions: Model predictions
        targets: Ground truth
        masks: Valid pixel mask
        snow_onehot_channels: Array of shape (NUM_SNOW_CLASSES, N_pixels) with one-hot encoding
    """
    results = []
    
    # Convert one-hot to class indices
    # For each pixel, find which channel has the max value (argmax)
    snow_class_idx = np.argmax(snow_onehot_channels, axis=0)  # (N_pixels,)
    
    # Convert indices back to original class values
    snow_class_values = np.array([SNOW_CLASSES[i] for i in snow_class_idx])
    
    print(f"\nAnalyzing by snow class...")
    print(f"Unique snow classes found: {np.unique(snow_class_values[masks])}")
    
    for cls_idx, cls_value in enumerate(SNOW_CLASSES):
        # Find pixels belonging to this class
        in_class = (snow_class_values == cls_value) & masks
        
        if in_class.sum() < 10:  # Skip classes with too few samples
            print(f"  Skipping class {cls_value}: only {in_class.sum()} pixels")
            continue
        
        metrics = compute_metrics_masked(predictions, targets, in_class)
        metrics['snow_class'] = int(cls_value)
        metrics['snow_class_name'] = f"Class_{cls_value}"
        
        print(f"  Class {cls_value}: {in_class.sum():,} pixels, MAE={metrics['mae']:.2f} mm")
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def analyze_by_canopy_cover(predictions, targets, masks, canopy_channel, bins, labels):
    """Compute metrics for different tree canopy cover ranges."""
    results = []
    
    for i in range(len(bins) - 1):
        bin_min = bins[i]
        bin_max = bins[i + 1]
        
        in_bin = (canopy_channel >= bin_min) & (canopy_channel < bin_max) & masks
        
        metrics = compute_metrics_masked(predictions, targets, in_bin)
        metrics['canopy_range'] = labels[i]
        metrics['bin_min'] = bin_min
        metrics['bin_max'] = bin_max
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def plot_canopy_vs_error(predictions, targets, masks, canopy_channel, save_path):
    """Scatter plot: canopy cover vs. prediction error for all valid pixels."""
    valid_idx = masks > 0
    
    canopy_valid = canopy_channel[valid_idx]
    error = np.abs(predictions[valid_idx] - targets[valid_idx])
    
    # Subsample if too many points
    if len(canopy_valid) > 50000:
        idx = np.random.choice(len(canopy_valid), 50000, replace=False)
        canopy_valid = canopy_valid[idx]
        error = error[idx]
    
    plt.figure(figsize=(10, 6))
    plt.hexbin(canopy_valid, error, gridsize=50, cmap='viridis', mincnt=1)
    plt.colorbar(label='Pixel Count')
    plt.xlabel('Tree Canopy Cover (%)')
    plt.ylabel('Absolute Error (mm SWE)')
    plt.title('Prediction Error vs. Tree Canopy Cover')
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
    
    # Load data
    dataloaders = create_dataloaders(
        zarr_dir=cfg.zarr_dir,
        batch_size=cfg.batch_size,
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        num_workers=cfg.num_workers,
        normalize=cfg.normalize,
        random_crop_train=False
    )
    
    # ========================================
    # GET GLOBAL STATS from the dataset
    # ========================================
    test_dataset = dataloaders['test'].dataset
    global_stats = test_dataset.global_stats
    
    print("\n" + "="*60)
    print("GLOBAL STATS DEBUG")
    print("="*60)
    print(f"\nX_mean for all channels (after one-hot encoding):")
    
    # Updated channel names for 17 channels
    channel_names = [
        'snow_class_0', 'snow_class_1', 'snow_class_2', 'snow_class_4', 
        'snow_class_5', 'snow_class_6', 'snow_class_7',  # 0-6: snow one-hot
        'landcover',     # 7
        'canopy',        # 8
        'elevation',     # 9
        'TB_37H',        # 10
        'TB_37V',        # 11
        'TB_19H',        # 12
        'TB_19V',        # 13
        'NDSI',          # 14
        'canopy_mask',   # 15
        'snow_mask'      # 16
    ]
    
    for i in range(17):
        print(f"  Ch{i:2d} ({channel_names[i]:15s}): mean={global_stats['X_mean'][i]:10.4f}, std={global_stats['X_std'][i]:10.4f}")

    print(f"\nY (SWE):")
    print(f"  mean={global_stats['Y_mean']:.6f} m  ({global_stats['Y_mean']*1000:.2f} mm)")
    print(f"  std={global_stats['Y_std']:.6f} m  ({global_stats['Y_std']*1000:.2f} mm)")

    print("\nCategorical channels (should have mean=0, std=1):")
    categorical = list(range(0, 7)) + [7, 15, 16]  # Snow one-hot + land + masks
    for c in categorical:
        print(f"  Ch{c:2d} ({channel_names[c]:15s}): mean={global_stats['X_mean'][c]}, std={global_stats['X_std'][c]}")

    print("\nContinuous channels (should have real stats):")
    continuous = [8, 9, 10, 11, 12, 13, 14]
    for c in continuous:
        print(f"  Ch{c:2d} ({channel_names[c]:15s}): mean={global_stats['X_mean'][c]:.2f}, std={global_stats['X_std'][c]:.2f}")
    
    test_loader = dataloaders['test']
    
    # ============================================================
    # 1. Evaluate Attention U-Net
    # ============================================================
    
    print("\n" + "="*60)
    print("Evaluating Attention U-Net")
    print("="*60)
    
    #unet = AttentionUNet(in_channels=17, out_channels=1).to(cfg.device)  # ← CHANGED from 11 to 17
    unet  = ToyModel(in_channels=17).to(cfg.device)
    unet_path = Path(cfg.checkpoint_dir) / cfg.unet_name
    
    if not unet_path.exists():
        print(f"ERROR: U-Net checkpoint not found at {unet_path}")
        return
    
    unet.load_state_dict(torch.load(unet_path, map_location=cfg.device))
    print(f"Loaded U-Net from {unet_path}")
    
    # Collect predictions WITH denormalization
    unet_data = collect_predictions(
        unet, 
        test_loader, 
        cfg.device, 
        model_type='unet',
        global_stats=global_stats
    )
    
    plot_pred_vs_target(
    unet_data['predictions'],
    unet_data['targets'],
    unet_data['masks'],
    Path(cfg.results_dir) / 'unet_pred_vs_target.png')

    # Overall metrics
    print("\n--- Overall Metrics (U-Net) ---")
    overall_metrics = compute_metrics_masked(
        unet_data['predictions'],
        unet_data['targets'],
        unet_data['masks']
    )
    
    print(f"R²: {overall_metrics['r2']:.4f}")
    print(f"RMSE: {overall_metrics['rmse']:.2f} mm")
    print(f"MAE: {overall_metrics['mae']:.2f} mm")
    print(f"Valid pixels: {overall_metrics['n_pixels']:,}")
    
    # Total SWE comparison
    total_pred = compute_total_swe(unet_data['predictions'], unet_data['masks'])
    total_target = compute_total_swe(unet_data['targets'], unet_data['masks'])
    
    print(f"\nTotal Predicted SWE: {total_pred:.2f} mm")
    print(f"Total Target SWE: {total_target:.2f} mm")
    print(f"Difference: {total_pred - total_target:.2f} mm ({100*(total_pred - total_target)/total_target:.2f}%)")
    
    # Extract feature channels (UPDATED INDICES!)
    CANOPY_CHANNEL = 8   # Tree canopy cover (was 2, now 8 after one-hot)
    SNOW_ONEHOT_CHANNELS = list(range(0, NUM_SNOW_CLASSES))  # Channels 0-6
    
    canopy = unet_data['features'][CANOPY_CHANNEL, :]
    snow_onehot = unet_data['features'][SNOW_ONEHOT_CHANNELS, :]  # (7, N_pixels)
    
    # --- Analysis by SWE bins ---
    print("\n--- Metrics by SWE Range (U-Net) ---")
    swe_results = analyze_by_swe_bins(
        unet_data['predictions'],
        unet_data['targets'],
        unet_data['masks'],
        cfg.swe_bins,
        cfg.swe_labels
    )
    print(swe_results.to_string(index=False))
    swe_results.to_csv(Path(cfg.results_dir) / 'unet_swe_bins.csv', index=False)
    
    # --- Analysis by snow class (UPDATED!) ---
    print("\n--- Metrics by Snow Map Class (U-Net) ---")
    snow_results = analyze_by_snow_class(
        unet_data['predictions'],
        unet_data['targets'],
        unet_data['masks'],
        snow_onehot  # Pass one-hot encoded channels
    )
    print(snow_results.to_string(index=False))
    snow_results.to_csv(Path(cfg.results_dir) / 'unet_snow_classes.csv', index=False)
    
    # --- Analysis by canopy cover ---
    print("\n--- Metrics by Tree Canopy Cover (U-Net) ---")
    canopy_results = analyze_by_canopy_cover(
        unet_data['predictions'],
        unet_data['targets'],
        unet_data['masks'],
        canopy,
        cfg.canopy_bins,
        cfg.canopy_labels
    )
    print(canopy_results.to_string(index=False))
    canopy_results.to_csv(Path(cfg.results_dir) / 'unet_canopy_bins.csv', index=False)
    
    # --- Plot canopy vs error ---
    plot_canopy_vs_error(
        unet_data['predictions'],
        unet_data['targets'],
        unet_data['masks'],
        canopy,
        Path(cfg.results_dir) / 'unet_canopy_vs_error.png'
    )
    
    # ============================================================
    # 2. Evaluate Random Forest (if available)
    # ============================================================
    
    rf_path = Path(cfg.checkpoint_dir) / cfg.rf_name
    
    if rf_path.exists():
        print("\n" + "="*60)
        print("Evaluating Random Forest Baseline")
        print("="*60)
        
        rf = RandomForestBaseline()
        rf.load(str(rf_path))
        print(f"Loaded Random Forest from {rf_path}")
        
        rf_data = collect_predictions(
            rf, 
            test_loader, 
            cfg.device, 
            model_type='rf',
            global_stats=global_stats
        )
        
        print("\n--- Overall Metrics (Random Forest) ---")
        rf_overall = compute_metrics_masked(
            rf_data['predictions'],
            rf_data['targets'],
            rf_data['masks']
        )
        
        print(f"R²: {rf_overall['r2']:.4f}")
        print(f"RMSE: {rf_overall['rmse']:.2f} mm")
        print(f"MAE: {rf_overall['mae']:.2f} mm")
        
        # Save RF results
        rf_canopy = rf_data['features'][CANOPY_CHANNEL, :]
        rf_snow_onehot = rf_data['features'][SNOW_ONEHOT_CHANNELS, :]
        
        # Analysis by snow class
        rf_snow_results = analyze_by_snow_class(
            rf_data['predictions'],
            rf_data['targets'],
            rf_data['masks'],
            rf_snow_onehot
        )
        rf_snow_results.to_csv(Path(cfg.results_dir) / 'rf_snow_classes.csv', index=False)
    
    print(f"\n Evaluation complete. Results saved to {cfg.results_dir}/")


if __name__ == "__main__":
    evaluate()