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
    
    # SWE bins for analysis (in mm, before normalization)
    # Adjust these based on your data distribution
    swe_bins = [0, 0.3, 0.6, 1.2]  # Low: 0-100, Medium: 100-500, High: 500+
    swe_labels = ['Low', 'Medium', 'High']
    
    # Tree canopy bins (%)
    canopy_bins = [0, 25, 50, 75, 100]
    canopy_labels = ['0-25%', '25-50%', '50-75%', '75-100%']


# ============================================================
# Metrics Functions
# ============================================================

def compute_metrics_masked(pred, target, mask):
    """
    Compute metrics only on valid (non-masked) pixels.
    
    Args:
        pred: (N,) array of predictions
        target: (N,) array of targets
        mask: (N,) boolean array (True = valid)
    
    Returns:
        dict with MAE, RMSE, R2
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
    """
    Compute total SWE volume (sum of all valid pixels).
    
    Args:
        swe_array: (N,) array of SWE values
        mask: (N,) boolean array
    
    Returns:
        float: total SWE
    """
    return np.sum(swe_array[mask])


# ============================================================
# Data Collection
# ============================================================

def collect_predictions(model, dataloader, device, model_type='unet'):
    """
    Run model on entire dataloader and collect all predictions, targets, and metadata.
    
    Returns:
        dict with arrays for predictions, targets, masks, and feature channels
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_masks = []
    all_features = []  # Store X for feature extraction
    all_metadata = []
    
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
            
            # Move to CPU and flatten
            pred_np = pred.cpu().numpy()  # (B, 1, H, W)
            target_np = Y.cpu().numpy()   # (B, 1, H, W)
            X_np = X.cpu().numpy()        # (B, 11, H, W)
            
            # Flatten spatial dimensions
            B, C, H, W = pred_np.shape
            pred_flat = pred_np.reshape(B, -1)      # (B, H*W)
            target_flat = target_np.reshape(B, -1)  # (B, H*W)
            X_flat = X_np.reshape(B, X_np.shape[1], -1)  # (B, 11, H*W)
            
            # Create mask: valid where target is not 0 (our NoData replacement)
            # AND not -1 (normalized NoData)
            # Better: check if target was affected by normalization
            # Since we normalized, invalid pixels should still be around 0 or negative
            mask = (target_flat > 1e-6)  # Valid pixels have positive SWE after normalization
            
            all_preds.append(pred_flat)
            all_targets.append(target_flat)
            all_masks.append(mask)
            all_features.append(X_flat)
            all_metadata.extend([metadata] * B)
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)      # (N_total, H*W)
    all_targets = np.concatenate(all_targets, axis=0)  # (N_total, H*W)
    all_masks = np.concatenate(all_masks, axis=0)      # (N_total, H*W)
    all_features = np.concatenate(all_features, axis=0)  # (N_total, 11, H*W)
    
    # Flatten completely
    all_preds = all_preds.flatten()
    all_targets = all_targets.flatten()
    all_masks = all_masks.flatten()
    all_features = all_features.reshape(all_features.shape[1], -1)  # (11, N_pixels)
    
    print(f"Collected {len(all_preds)} total pixels, {all_masks.sum()} valid")
    
    return {
        'predictions': all_preds,
        'targets': all_targets,
        'masks': all_masks,
        'features': all_features,
        'metadata': all_metadata
    }


# ============================================================
# Analysis Functions
# ============================================================

def analyze_by_swe_bins(predictions, targets, masks, bins, labels):
    """
    Compute metrics for different SWE value ranges.
    """
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
    """
    Compute metrics for each snow map class.
    
    Assumes snow map is in channel index (need to check your data).
    Snow map classes typically: 1=snow, 2=ice, 3=water, etc.
    """
    unique_classes = np.unique(snow_map_channel[masks])
    print(f'unique snow map classed {unique_classes}')
    
    results = []
    
    for cls in unique_classes:
        if cls == 0:  # Skip background/NoData
            continue
        
        in_class = (snow_map_channel == cls) & masks
        
        metrics = compute_metrics_masked(predictions, targets, in_class)
        metrics['snow_class'] = int(cls)
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def analyze_by_canopy_cover(predictions, targets, masks, canopy_channel, bins, labels):
    """
    Compute metrics for different tree canopy cover ranges.
    """
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
    """
    Scatter plot: canopy cover vs. prediction error for all valid pixels.
    """
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
    
    # Collect predictions
    unet_data = collect_predictions(unet, test_loader, cfg.device, model_type='unet')
    
    # Overall metrics
    print("\n--- Overall Metrics (U-Net) ---")
    overall_metrics = compute_metrics_masked(
        unet_data['predictions'],
        unet_data['targets'],
        unet_data['masks']
    )
    
    print(f"R²: {overall_metrics['r2']:.4f}")
    print(f"RMSE: {overall_metrics['rmse']:.4f}")
    print(f"MAE: {overall_metrics['mae']:.4f}")
    print(f"Valid pixels: {overall_metrics['n_pixels']:,}")
    
    # Total SWE comparison
    total_pred = compute_total_swe(unet_data['predictions'], unet_data['masks'])
    total_target = compute_total_swe(unet_data['targets'], unet_data['masks'])
    
    print(f"\nTotal Predicted SWE: {total_pred:.2f}")
    print(f"Total Target SWE: {total_target:.2f}")
    print(f"Difference: {total_pred - total_target:.2f} ({100*(total_pred - total_target)/total_target:.2f}%)")
    
    # Extract feature channels
    # Assuming channel order: [elevation, slope, aspect, northness, eastness, 
    #                          tree_canopy, snow_map, dem, ?, ?, ?]
    # ADJUST THESE INDICES BASED ON YOUR ACTUAL DATA!
    CANOPY_CHANNEL = 2   # Tree canopy cover
    SNOW_MAP_CHANNEL = 0  # Snow map classification
    
    canopy = unet_data['features'][CANOPY_CHANNEL, :]
    snow_map = unet_data['features'][SNOW_MAP_CHANNEL, :]
    
    # Denormalize if needed (features were normalized per-channel)
    # For now, assume they're in reasonable ranges
    
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
    # 2. Evaluate Random Forest 
    # ============================================================
    
    rf_path = Path(cfg.checkpoint_dir) / cfg.rf_name
    
    if rf_path.exists():
        print("\n" + "="*60)
        print("Evaluating Random Forest Baseline")
        print("="*60)
        
        rf = RandomForestBaseline()
        rf.load(str(rf_path))
        print(f"Loaded Random Forest from {rf_path}")
        
        rf_data = collect_predictions(rf, test_loader, cfg.device, model_type='rf')
        
        print("\n--- Overall Metrics (Random Forest) ---")
        rf_overall = compute_metrics_masked(
            rf_data['predictions'],
            rf_data['targets'],
            rf_data['masks']
        )
        
        print(f"R²: {rf_overall['r2']:.4f}")
        print(f"RMSE: {rf_overall['rmse']:.4f}")
        print(f"MAE: {rf_overall['mae']:.4f}")
        
        rf_total_pred = compute_total_swe(rf_data['predictions'], rf_data['masks'])
        rf_total_target = compute_total_swe(rf_data['targets'], rf_data['masks'])
        
        print(f"\nTotal Predicted SWE: {rf_total_pred:.2f}")
        print(f"Total Target SWE: {rf_total_target:.2f}")
        print(f"Difference: {rf_total_pred - rf_total_target:.2f}")
        
        # Same analyses for RF
        rf_canopy = rf_data['features'][CANOPY_CHANNEL, :]
        rf_snow_map = rf_data['features'][SNOW_MAP_CHANNEL, :]
        
        rf_swe_results = analyze_by_swe_bins(
            rf_data['predictions'], rf_data['targets'], rf_data['masks'],
            cfg.swe_bins, cfg.swe_labels
        )
        rf_swe_results.to_csv(Path(cfg.results_dir) / 'rf_swe_bins.csv', index=False)
        
        rf_snow_results = analyze_by_snow_class(
            rf_data['predictions'], rf_data['targets'], rf_data['masks'], rf_snow_map
        )
        rf_snow_results.to_csv(Path(cfg.results_dir) / 'rf_snow_classes.csv', index=False)
        
        rf_canopy_results = analyze_by_canopy_cover(
            rf_data['predictions'], rf_data['targets'], rf_data['masks'],
            rf_canopy, cfg.canopy_bins, cfg.canopy_labels
        )
        rf_canopy_results.to_csv(Path(cfg.results_dir) / 'rf_canopy_bins.csv', index=False)
        
        plot_canopy_vs_error(
            rf_data['predictions'], rf_data['targets'], rf_data['masks'],
            rf_canopy, Path(cfg.results_dir) / 'rf_canopy_vs_error.png'
        )
    
    # ============================================================
    # 3. Model Comparison
    # ============================================================
    
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    
    comparison = pd.DataFrame({
        'Model': ['U-Net', 'Random Forest'] if rf_path.exists() else ['U-Net'],
        'R²': [overall_metrics['r2'], rf_overall['r2']] if rf_path.exists() else [overall_metrics['r2']],
        'RMSE': [overall_metrics['rmse'], rf_overall['rmse']] if rf_path.exists() else [overall_metrics['rmse']],
        'MAE': [overall_metrics['mae'], rf_overall['mae']] if rf_path.exists() else [overall_metrics['mae']],
        'Total_Pred': [total_pred, rf_total_pred] if rf_path.exists() else [total_pred],
        'Total_Target': [total_target, rf_total_target] if rf_path.exists() else [total_target]
    })
    
    print(comparison.to_string(index=False))
    comparison.to_csv(Path(cfg.results_dir) / 'model_comparison.csv', index=False)
    
    print(f"\n Evaluation complete. Results saved to {cfg.results_dir}/")


if __name__ == "__main__":
    evaluate()