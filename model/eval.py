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

from dataset import create_dataloaders
from models import AttentionUNet, RandomForestBaseline


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
            X_mask = metadata['X_mask'].cpu().numpy()  # (B, 11, H, W) ## i think this is the bug:
            
            # Move to CPU
            pred_np = pred.cpu().numpy()      # (B, 1, H, W)
            target_np = Y.cpu().numpy()       # (B, 1, H, W)
            X_np = X.cpu().numpy()            # (B, 11, H, W)
            
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
                X_mean = global_stats['X_mean'][:, None, None, None]  # (11, 1, 1, 1)
                X_std = global_stats['X_std'][:, None, None, None]
                
                # Only denormalize continuous channels
                continuous_channels = [2, 3, 4, 5, 6, 7, 8]  # Based on your config
                for c in continuous_channels:
                    X_np[:, c] = X_np[:, c] * X_std[c] + X_mean[c]
                
                # Categorical channels (0, 1, 9, 10) were "normalized" with mean=0, std=1
                # So they stay as-is (already original values)
            
            # Convert to mm for analysis (SWE in meters → mm)
            # it already is in meters!
            pred_np = pred_np * 1000
            target_np = target_np * 1000
            
            # Flatten spatial dimensions
            B, C, H, W = pred_np.shape
            pred_flat = pred_np.reshape(B, -1)      # (B, H*W)
            target_flat = target_np.reshape(B, -1)  # (B, H*W)
            mask_flat = Y_mask.reshape(B, -1)       # (B, H*W) - boolean
            X_flat = X_np.reshape(B, X_np.shape[1], -1)  # (B, 11, H*W)
            
            all_preds.append(pred_flat)
            all_targets.append(target_flat)
            all_masks.append(mask_flat)
            all_features.append(X_flat)
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    all_masks = np.concatenate(all_masks, axis=0).flatten().astype(bool)
    all_features = np.concatenate(all_features, axis=0)
    all_features = all_features.reshape(all_features.shape[1], -1)  # (11, N_pixels)
    
    print(f"\nCollected {len(all_preds)} total pixels, {all_masks.sum():,} valid")
    
    # Print sample values for debugging
    if all_masks.sum() > 0:
        valid_preds = all_preds[all_masks]
        valid_targets = all_targets[all_masks]
        print(f"\nDenormalized sample values:")
        print(f"  Predictions (mm): min={valid_preds.min():.2f}, max={valid_preds.max():.2f}, mean={valid_preds.mean():.2f}")
        print(f"  Targets (mm): min={valid_targets.min():.2f}, max={valid_targets.max():.2f}, mean={valid_targets.mean():.2f}")
        
        # Check snow class values (channel 0)
        snow_map = all_features[0, :]
        valid_snow = snow_map[all_masks]
        unique_snow = np.unique(valid_snow)
        print(f"  Snow classes (first 20): {unique_snow[:20]}")
    
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


def analyze_by_snow_class(predictions, targets, masks, snow_map_channel):
    """Compute metrics for each snow map class."""
    # Round to nearest integer for categorical analysis
    snow_map_int = np.round(snow_map_channel).astype(int)
    
    unique_classes = np.unique(snow_map_int[masks])
    
    # Filter out obviously invalid classes
    valid_classes = unique_classes[(unique_classes >= 0) & (unique_classes < 100)]
    
    results = []
    
    for cls in valid_classes:
        in_class = (snow_map_int == cls) & masks
        
        if in_class.sum() < 10:  # Skip classes with too few samples
            continue
        
        metrics = compute_metrics_masked(predictions, targets, in_class)
        metrics['snow_class'] = int(cls)
        
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
    
    print("\nGlobal stats being used for denormalization:")
    print(f"  X_mean: {global_stats['X_mean']}")
    print(f"  X_std: {global_stats['X_std']}")
    print(f"  Y_mean: {global_stats['Y_mean']:.4f}")
    print(f"  Y_std: {global_stats['Y_std']:.4f}")

    # ========================================
    # GET GLOBAL STATS from the dataset
    # ========================================

    print("\n" + "="*60)
    print("GLOBAL STATS DEBUG")
    print("="*60)
    print(f"\nX_mean for all channels:")
    channel_names = ['snow_class', 'landcover', 'canopy', 'elevation', 
                    'TB_37H', 'TB_37V', 'TB_19H', 'TB_19V', 
                    'NDSI', 'forested', 'unforested']
    for i in range(11):
        print(f"  Ch{i} ({channel_names[i]:12s}): mean={global_stats['X_mean'][i]:10.4f}, std={global_stats['X_std'][i]:10.4f}")

    print(f"\nY (SWE):")
    print(f"  mean={global_stats['Y_mean']:.6f} m  ({global_stats['Y_mean']*1000:.2f} mm)")
    print(f"  std={global_stats['Y_std']:.6f} m  ({global_stats['Y_std']*1000:.2f} mm)")

    print("\nCategorical channels (should have mean=0, std=1):")
    categorical = [0, 1, 9, 10]
    for c in categorical:
        print(f"  Ch{c} ({channel_names[c]}): mean={global_stats['X_mean'][c]}, std={global_stats['X_std'][c]}")

    print("\nContinuous channels (should have real stats):")
    continuous = [2, 3, 4, 5, 6, 7, 8]
    for c in continuous:
        print(f"  Ch{c} ({channel_names[c]}): mean={global_stats['X_mean'][c]:.2f}, std={global_stats['X_std'][c]:.2f}")

    print("\nGlobal stats being used for denormalization:")
    print(f"  X_mean: {global_stats['X_mean']}")
    print(f"  X_std: {global_stats['X_std']}")
    print(f"  Y_mean: {global_stats['Y_mean']:.4f}")
    print(f"  Y_std: {global_stats['Y_std']:.4f}")
    
    test_loader = dataloaders['test']
    
    # ============================================================
    # 1. Evaluate Attention U-Net
    # ============================================================
    
    print("\n" + "="*60)
    print("Evaluating Attention U-Net")
    print("="*60)
    
    unet = AttentionUNet(in_channels=11, out_channels=1).to(cfg.device)
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
        global_stats=global_stats  # ← CRITICAL: Pass this!
    )
    
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
    
    # Extract feature channels
    CANOPY_CHANNEL = 2   # Tree canopy cover
    SNOW_MAP_CHANNEL = 0  # Snow map classification
    
    canopy = unet_data['features'][CANOPY_CHANNEL, :]
    snow_map = unet_data['features'][SNOW_MAP_CHANNEL, :]
    
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
    
    # --- Analysis by snow class ---
    print("\n--- Metrics by Snow Map Class (U-Net) ---")
    snow_results = analyze_by_snow_class(
        unet_data['predictions'],
        unet_data['targets'],
        unet_data['masks'],
        snow_map
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
        
        # NO model.eval() for Random Forest!
        rf_data = collect_predictions(
            rf, 
            test_loader, 
            cfg.device, 
            model_type='rf',
            global_stats=global_stats  # ← Pass this!
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
        
        # Save RF results...
        # (add similar analysis as U-Net if needed)
    
    print(f"\n✅ Evaluation complete. Results saved to {cfg.results_dir}/")


if __name__ == "__main__":
    evaluate()