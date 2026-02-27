# feature_importance.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict

from dataset import create_dataloaders, NUM_SNOW_CLASSES, SNOW_CLASSES
from models import AttentionUNet


# ============================================================
# Config
# ============================================================

class FeatureImportanceConfig:
    zarr_dir = "/discover/nobackup/cmbreen/gap-filling-data/zarr_chunks"
    batch_size = 16
    patch_size = 256
    stride = 128
    num_workers = 4
    normalize = True
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint_dir = "./checkpoints"
    unet_name = "attention_unet_final.pth"
    
    results_dir = "./feature_importance"
    
    # Feature groups for ablation
    # Channel indices after one-hot encoding (17 total channels)
    feature_groups = {
        'snow_class': list(range(0, 7)),      # Channels 0-6: snow one-hot
        'landcover': [7],                      # Channel 7: land cover
        'canopy_cover': [8],                   # Channel 8: tree canopy
        'elevation': [9],                      # Channel 9: elevation
        'passive_microwave': [10, 11, 12, 13], # Channels 10-13: TB bands
        'viirs_ndsi': [14],                    # Channel 14: NDSI
        'masks': [15, 16]                      # Channels 15-16: validity masks
    }
    
    # Stratification bins for interaction analysis
    canopy_bins = [0, 25, 50, 75, 100]
    canopy_labels = ['0-25%', '25-50%', '50-75%', '75-100%']
    
    elevation_bins = [0, 2500, 3000, 3500, 10000]  # meters
    elevation_labels = ['<2500m', '2500-3000m', '3000-3500m', '>3500m']
    
    swe_bins = [0, 100, 300, 600, 10000]
    swe_labels = ['Low (0-100mm)', 'Medium (100-300mm)', 'High (300-600mm)', 'Very High (>600mm)']


# ============================================================
# Metrics
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


# ============================================================
# Baseline Evaluation (No Ablation)
# ============================================================

def evaluate_baseline(model, dataloader, device, global_stats):
    """Evaluate model with all features."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_masks = []
    all_features = []
    
    print("Evaluating baseline (all features)...")
    
    with torch.no_grad():
        for X, Y, metadata in tqdm(dataloader):
            X = X.to(device)
            Y = Y.to(device)
            
            pred = model(X)
            
            Y_mask = metadata['Y_mask'].cpu().numpy()
            
            pred_np = pred.cpu().numpy()
            target_np = Y.cpu().numpy()
            X_np = X.cpu().numpy()
            
            # Denormalize
            if global_stats is not None:
                Y_mean = global_stats['Y_mean']
                Y_std = global_stats['Y_std']
                pred_np = pred_np * Y_std + Y_mean
                target_np = target_np * Y_std + Y_mean
                
                X_mean = global_stats['X_mean'][:, None, None, None]
                X_std = global_stats['X_std'][:, None, None, None]
                continuous_channels = [8, 9, 10, 11, 12, 13, 14]
                for c in continuous_channels:
                    X_np[:, c] = X_np[:, c] * X_std[c] + X_mean[c]
            
            # Convert to mm
            pred_np = pred_np * 1000
            target_np = target_np * 1000
            
            # Flatten
            B = pred_np.shape[0]
            pred_flat = pred_np.reshape(B, -1)
            target_flat = target_np.reshape(B, -1)
            mask_flat = Y_mask.reshape(B, -1)
            X_flat = X_np.reshape(B, X_np.shape[1], -1)
            
            all_preds.append(pred_flat)
            all_targets.append(target_flat)
            all_masks.append(mask_flat)
            all_features.append(X_flat)
    
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    all_masks = np.concatenate(all_masks, axis=0).flatten().astype(bool)
    all_features = np.concatenate(all_features, axis=0)
    all_features = all_features.reshape(all_features.shape[1], -1)
    
    metrics = compute_metrics_masked(all_preds, all_targets, all_masks)
    
    print(f"Baseline - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.2f} mm, MAE: {metrics['mae']:.2f} mm")
    
    return {
        'predictions': all_preds,
        'targets': all_targets,
        'masks': all_masks,
        'features': all_features,
        'metrics': metrics
    }


# ============================================================
# Ablation Study
# ============================================================

def ablation_study(model, dataloader, device, global_stats, feature_groups):
    """
    Remove each feature group and measure performance drop.
    Larger drop = more important feature.
    """
    print("\n" + "="*60)
    print("ABLATION STUDY: Removing Feature Groups")
    print("="*60)
    
    results = []
    baseline = None
    
    for group_name, channels in feature_groups.items():
        print(f"\nAblating: {group_name} (channels {channels})")
        
        model.eval()
        all_preds = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for X, Y, metadata in tqdm(dataloader, desc=f"Ablating {group_name}"):
                X = X.to(device)
                Y = Y.to(device)
                
                # ABLATE: Zero out the channels
                X_ablated = X.clone()
                for c in channels:
                    X_ablated[:, c, :, :] = 0.0
                
                pred = model(X_ablated)
                
                Y_mask = metadata['Y_mask'].cpu().numpy()
                pred_np = pred.cpu().numpy()
                target_np = Y.cpu().numpy()
                
                # Denormalize
                if global_stats is not None:
                    Y_mean = global_stats['Y_mean']
                    Y_std = global_stats['Y_std']
                    pred_np = pred_np * Y_std + Y_mean
                    target_np = target_np * Y_std + Y_mean
                
                pred_np = pred_np * 1000
                target_np = target_np * 1000
                
                B = pred_np.shape[0]
                pred_flat = pred_np.reshape(B, -1)
                target_flat = target_np.reshape(B, -1)
                mask_flat = Y_mask.reshape(B, -1)
                
                all_preds.append(pred_flat)
                all_targets.append(target_flat)
                all_masks.append(mask_flat)
        
        all_preds = np.concatenate(all_preds, axis=0).flatten()
        all_targets = np.concatenate(all_targets, axis=0).flatten()
        all_masks = np.concatenate(all_masks, axis=0).flatten().astype(bool)
        
        metrics = compute_metrics_masked(all_preds, all_targets, all_masks)
        
        # Store baseline
        if group_name == 'masks':  # Use masks as baseline (should have minimal effect)
            baseline = metrics
        
        results.append({
            'feature_group': group_name,
            'channels': str(channels),
            'r2': metrics['r2'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'n_pixels': metrics['n_pixels']
        })
        
        print(f"  R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.2f} mm, MAE: {metrics['mae']:.2f} mm")
    
    df = pd.DataFrame(results)
    return df


# ============================================================
# Permutation Importance
# ============================================================

def permutation_importance(model, dataloader, device, global_stats, feature_groups, n_repeats=5):
    """
    Randomly shuffle each feature group and measure performance drop.
    More robust than ablation for correlated features.
    """
    print("\n" + "="*60)
    print("PERMUTATION IMPORTANCE")
    print("="*60)
    
    results = defaultdict(list)
    
    for group_name, channels in feature_groups.items():
        print(f"\nPermuting: {group_name} (channels {channels})")
        
        for repeat in range(n_repeats):
            model.eval()
            all_preds = []
            all_targets = []
            all_masks = []
            
            with torch.no_grad():
                for X, Y, metadata in tqdm(dataloader, desc=f"Repeat {repeat+1}/{n_repeats}"):
                    X = X.to(device)
                    Y = Y.to(device)
                    
                    # PERMUTE: Shuffle the channels along batch dimension
                    X_permuted = X.clone()
                    for c in channels:
                        # Shuffle each channel independently
                        perm_idx = torch.randperm(X.shape[0])
                        X_permuted[:, c, :, :] = X[perm_idx, c, :, :]
                    
                    pred = model(X_permuted)
                    
                    Y_mask = metadata['Y_mask'].cpu().numpy()
                    pred_np = pred.cpu().numpy()
                    target_np = Y.cpu().numpy()
                    
                    # Denormalize
                    if global_stats is not None:
                        Y_mean = global_stats['Y_mean']
                        Y_std = global_stats['Y_std']
                        pred_np = pred_np * Y_std + Y_mean
                        target_np = target_np * Y_std + Y_mean
                    
                    pred_np = pred_np * 1000
                    target_np = target_np * 1000
                    
                    B = pred_np.shape[0]
                    pred_flat = pred_np.reshape(B, -1)
                    target_flat = target_np.reshape(B, -1)
                    mask_flat = Y_mask.reshape(B, -1)
                    
                    all_preds.append(pred_flat)
                    all_targets.append(target_flat)
                    all_masks.append(mask_flat)
            
            all_preds = np.concatenate(all_preds, axis=0).flatten()
            all_targets = np.concatenate(all_targets, axis=0).flatten()
            all_masks = np.concatenate(all_masks, axis=0).flatten().astype(bool)
            
            metrics = compute_metrics_masked(all_preds, all_targets, all_masks)
            
            results['feature_group'].append(group_name)
            results['repeat'].append(repeat)
            results['r2'].append(metrics['r2'])
            results['rmse'].append(metrics['rmse'])
            results['mae'].append(metrics['mae'])
        
        mean_r2 = np.mean([r for g, r in zip(results['feature_group'], results['r2']) if g == group_name])
        mean_rmse = np.mean([r for g, r in zip(results['feature_group'], results['rmse']) if g == group_name])
        print(f"  Mean R²: {mean_r2:.4f}, Mean RMSE: {mean_rmse:.2f} mm")
    
    df = pd.DataFrame(results)
    return df


# ============================================================
# Interaction Analysis
# ============================================================

def interaction_analysis(baseline_data, cfg):
    """
    Analyze how feature importance varies across different conditions.
    E.g., is passive microwave more important in high canopy areas?
    """
    print("\n" + "="*60)
    print("INTERACTION ANALYSIS")
    print("="*60)
    
    predictions = baseline_data['predictions']
    targets = baseline_data['targets']
    masks = baseline_data['masks']
    features = baseline_data['features']
    
    # Extract relevant features
    canopy = features[8, :]  # Canopy cover
    elevation = features[9, :]  # Elevation
    ndsi = features[14, :]  # NDSI
    
    # Passive microwave (average of TB channels)
    tb_mean = np.mean(features[[10, 11, 12, 13], :], axis=0)
    
    # Snow class (convert one-hot to class index)
    snow_onehot = features[0:7, :]
    snow_class_idx = np.argmax(snow_onehot, axis=0)
    snow_class_values = np.array([SNOW_CLASSES[i] for i in snow_class_idx])
    
    # Compute errors
    errors = np.abs(predictions - targets)
    
    results = []
    
    # -----------------------------------------------
    # 1. Passive Microwave vs Canopy Cover
    # -----------------------------------------------
    print("\n--- Passive Microwave Importance vs Canopy Cover ---")
    
    for i in range(len(cfg.canopy_bins) - 1):
        bin_min = cfg.canopy_bins[i]
        bin_max = cfg.canopy_bins[i + 1]
        
        in_bin = (canopy >= bin_min) & (canopy < bin_max) & masks
        
        if in_bin.sum() < 100:
            continue
        
        # Correlation between TB and error in this bin
        tb_valid = tb_mean[in_bin]
        error_valid = errors[in_bin]
        
        if len(tb_valid) > 0 and np.std(tb_valid) > 0:
            correlation = np.corrcoef(tb_valid, error_valid)[0, 1]
        else:
            correlation = np.nan
        
        mean_error = np.mean(error_valid)
        
        results.append({
            'analysis': 'PM_vs_canopy',
            'stratification': cfg.canopy_labels[i],
            'bin_min': bin_min,
            'bin_max': bin_max,
            'mean_error': mean_error,
            'correlation': correlation,
            'n_pixels': in_bin.sum()
        })
        
        print(f"  {cfg.canopy_labels[i]}: Mean Error={mean_error:.2f} mm, "
              f"TB-Error Corr={correlation:.3f}, N={in_bin.sum():,}")
    
    # -----------------------------------------------
    # 2. NDSI vs Canopy Cover
    # -----------------------------------------------
    print("\n--- VIIRS NDSI Importance vs Canopy Cover ---")
    
    for i in range(len(cfg.canopy_bins) - 1):
        bin_min = cfg.canopy_bins[i]
        bin_max = cfg.canopy_bins[i + 1]
        
        in_bin = (canopy >= bin_min) & (canopy < bin_max) & masks
        
        if in_bin.sum() < 100:
            continue
        
        ndsi_valid = ndsi[in_bin]
        error_valid = errors[in_bin]
        
        if len(ndsi_valid) > 0 and np.std(ndsi_valid) > 0:
            correlation = np.corrcoef(ndsi_valid, error_valid)[0, 1]
        else:
            correlation = np.nan
        
        mean_error = np.mean(error_valid)
        
        results.append({
            'analysis': 'NDSI_vs_canopy',
            'stratification': cfg.canopy_labels[i],
            'bin_min': bin_min,
            'bin_max': bin_max,
            'mean_error': mean_error,
            'correlation': correlation,
            'n_pixels': in_bin.sum()
        })
        
        print(f"  {cfg.canopy_labels[i]}: Mean Error={mean_error:.2f} mm, "
              f"NDSI-Error Corr={correlation:.3f}, N={in_bin.sum():,}")
    
    # -----------------------------------------------
    # 3. Passive Microwave vs Snow Class
    # -----------------------------------------------
    print("\n--- Passive Microwave Importance vs Snow Class ---")
    
    for cls_value in SNOW_CLASSES:
        in_class = (snow_class_values == cls_value) & masks
        
        if in_class.sum() < 100:
            continue
        
        tb_valid = tb_mean[in_class]
        error_valid = errors[in_class]
        
        if len(tb_valid) > 0 and np.std(tb_valid) > 0:
            correlation = np.corrcoef(tb_valid, error_valid)[0, 1]
        else:
            correlation = np.nan
        
        mean_error = np.mean(error_valid)
        
        results.append({
            'analysis': 'PM_vs_snowclass',
            'stratification': f'Class_{cls_value}',
            'bin_min': cls_value,
            'bin_max': cls_value,
            'mean_error': mean_error,
            'correlation': correlation,
            'n_pixels': in_class.sum()
        })
        
        print(f"  Class {cls_value}: Mean Error={mean_error:.2f} mm, "
              f"TB-Error Corr={correlation:.3f}, N={in_class.sum():,}")
    
    # -----------------------------------------------
    # 4. NDSI vs Snow Class
    # -----------------------------------------------
    print("\n--- VIIRS NDSI Importance vs Snow Class ---")
    
    for cls_value in SNOW_CLASSES:
        in_class = (snow_class_values == cls_value) & masks
        
        if in_class.sum() < 100:
            continue
        
        ndsi_valid = ndsi[in_class]
        error_valid = errors[in_class]
        
        if len(ndsi_valid) > 0 and np.std(ndsi_valid) > 0:
            correlation = np.corrcoef(ndsi_valid, error_valid)[0, 1]
        else:
            correlation = np.nan
        
        mean_error = np.mean(error_valid)
        
        results.append({
            'analysis': 'NDSI_vs_snowclass',
            'stratification': f'Class_{cls_value}',
            'bin_min': cls_value,
            'bin_max': cls_value,
            'mean_error': mean_error,
            'correlation': correlation,
            'n_pixels': in_class.sum()
        })
        
        print(f"  Class {cls_value}: Mean Error={mean_error:.2f} mm, "
              f"NDSI-Error Corr={correlation:.3f}, N={in_class.sum():,}")
    
    # -----------------------------------------------
    # 5. Passive Microwave vs Elevation
    # -----------------------------------------------
    print("\n--- Passive Microwave Importance vs Elevation ---")
    
    for i in range(len(cfg.elevation_bins) - 1):
        bin_min = cfg.elevation_bins[i]
        bin_max = cfg.elevation_bins[i + 1]
        
        in_bin = (elevation >= bin_min) & (elevation < bin_max) & masks
        
        if in_bin.sum() < 100:
            continue
        
        tb_valid = tb_mean[in_bin]
        error_valid = errors[in_bin]
        
        if len(tb_valid) > 0 and np.std(tb_valid) > 0:
            correlation = np.corrcoef(tb_valid, error_valid)[0, 1]
        else:
            correlation = np.nan
        
        mean_error = np.mean(error_valid)
        
        results.append({
            'analysis': 'PM_vs_elevation',
            'stratification': cfg.elevation_labels[i],
            'bin_min': bin_min,
            'bin_max': bin_max,
            'mean_error': mean_error,
            'correlation': correlation,
            'n_pixels': in_bin.sum()
        })
        
        print(f"  {cfg.elevation_labels[i]}: Mean Error={mean_error:.2f} mm, "
              f"TB-Error Corr={correlation:.3f}, N={in_bin.sum():,}")
    
    df = pd.DataFrame(results)
    return df


# ============================================================
# Visualization
# ============================================================

def plot_ablation_results(ablation_df, baseline_metrics, save_path):
    """Bar plot showing performance drop for each ablated feature."""
    
    # Calculate delta from baseline (assuming baseline is no ablation)
    baseline_mae = baseline_metrics['mae']
    baseline_rmse = baseline_metrics['rmse']
    
    ablation_df['mae_increase'] = ablation_df['mae'] - baseline_mae
    ablation_df['rmse_increase'] = ablation_df['rmse'] - baseline_rmse
    
    # Sort by importance
    ablation_df = ablation_df.sort_values('mae_increase', ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MAE increase
    axes[0].barh(ablation_df['feature_group'], ablation_df['mae_increase'], color='steelblue')
    axes[0].set_xlabel('MAE Increase (mm) when feature removed')
    axes[0].set_title('Feature Importance: MAE')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[0].grid(axis='x', alpha=0.3)
    
    # RMSE increase
    axes[1].barh(ablation_df['feature_group'], ablation_df['rmse_increase'], color='coral')
    axes[1].set_xlabel('RMSE Increase (mm) when feature removed')
    axes[1].set_title('Feature Importance: RMSE')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ablation plot to {save_path}")


def plot_permutation_results(permutation_df, baseline_metrics, save_path):
    """Box plot showing performance drop distribution for permuted features."""
    
    baseline_mae = baseline_metrics['mae']
    baseline_rmse = baseline_metrics['rmse']
    
    permutation_df['mae_increase'] = permutation_df['mae'] - baseline_mae
    permutation_df['rmse_increase'] = permutation_df['rmse'] - baseline_rmse
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # MAE
    permutation_df.boxplot(column='mae_increase', by='feature_group', ax=axes[0])
    axes[0].set_xlabel('Feature Group')
    axes[0].set_ylabel('MAE Increase (mm)')
    axes[0].set_title('Permutation Importance: MAE')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0].grid(axis='y', alpha=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # RMSE
    permutation_df.boxplot(column='rmse_increase', by='feature_group', ax=axes[1])
    axes[1].set_xlabel('Feature Group')
    axes[1].set_ylabel('RMSE Increase (mm)')
    axes[1].set_title('Permutation Importance: RMSE')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[1].grid(axis='y', alpha=0.3)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('')  # Remove default title
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved permutation plot to {save_path}")


def plot_interaction_results(interaction_df, save_dir):
    """Create plots for each interaction analysis."""
    
    # PM vs Canopy
    df_pm_canopy = interaction_df[interaction_df['analysis'] == 'PM_vs_canopy']
    if len(df_pm_canopy) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].bar(df_pm_canopy['stratification'], df_pm_canopy['mean_error'], color='steelblue')
        axes[0].set_ylabel('Mean Absolute Error (mm)')
        axes[0].set_xlabel('Canopy Cover')
        axes[0].set_title('Error vs Canopy Cover')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].bar(df_pm_canopy['stratification'], df_pm_canopy['correlation'], color='coral')
        axes[1].set_ylabel('TB-Error Correlation')
        axes[1].set_xlabel('Canopy Cover')
        axes[1].set_title('Passive Microwave Relevance by Canopy')
        axes[1].axhline(0, color='black', linestyle='-', linewidth=1)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'interaction_PM_vs_canopy.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # NDSI vs Canopy
    df_ndsi_canopy = interaction_df[interaction_df['analysis'] == 'NDSI_vs_canopy']
    if len(df_ndsi_canopy) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].bar(df_ndsi_canopy['stratification'], df_ndsi_canopy['mean_error'], color='steelblue')
        axes[0].set_ylabel('Mean Absolute Error (mm)')
        axes[0].set_xlabel('Canopy Cover')
        axes[0].set_title('Error vs Canopy Cover')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].bar(df_ndsi_canopy['stratification'], df_ndsi_canopy['correlation'], color='forestgreen')
        axes[1].set_ylabel('NDSI-Error Correlation')
        axes[1].set_xlabel('Canopy Cover')
        axes[1].set_title('VIIRS NDSI Relevance by Canopy')
        axes[1].axhline(0, color='black', linestyle='-', linewidth=1)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'interaction_NDSI_vs_canopy.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # PM vs Snow Class
    df_pm_snow = interaction_df[interaction_df['analysis'] == 'PM_vs_snowclass']
    if len(df_pm_snow) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].bar(df_pm_snow['stratification'], df_pm_snow['mean_error'], color='steelblue')
        axes[0].set_ylabel('Mean Absolute Error (mm)')
        axes[0].set_xlabel('Snow Class')
        axes[0].set_title('Error vs Snow Class')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].bar(df_pm_snow['stratification'], df_pm_snow['correlation'], color='coral')
        axes[1].set_ylabel('TB-Error Correlation')
        axes[1].set_xlabel('Snow Class')
        axes[1].set_title('Passive Microwave Relevance by Snow Class')
        axes[1].axhline(0, color='black', linestyle='-', linewidth=1)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'interaction_PM_vs_snowclass.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # NDSI vs Snow Class
    df_ndsi_snow = interaction_df[interaction_df['analysis'] == 'NDSI_vs_snowclass']
    if len(df_ndsi_snow) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].bar(df_ndsi_snow['stratification'], df_ndsi_snow['mean_error'], color='steelblue')
        axes[0].set_ylabel('Mean Absolute Error (mm)')
        axes[0].set_xlabel('Snow Class')
        axes[0].set_title('Error vs Snow Class')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        axes[1].bar(df_ndsi_snow['stratification'], df_ndsi_snow['correlation'], color='forestgreen')
        axes[1].set_ylabel('NDSI-Error Correlation')
        axes[1].set_xlabel('Snow Class')
        axes[1].set_title('VIIRS NDSI Relevance by Snow Class')
        axes[1].axhline(0, color='black', linestyle='-', linewidth=1)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'interaction_NDSI_vs_snowclass.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved interaction plots to {save_dir}/")


# ============================================================
# Main
# ============================================================

def main():
    cfg = FeatureImportanceConfig()
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
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
    global_stats = test_loader.dataset.global_stats
    
    # Load model
    print("Loading model...")
    model = AttentionUNet(in_channels=17, out_channels=1).to(cfg.device)
    model_path = Path(cfg.checkpoint_dir) / cfg.unet_name
    
    if not model_path.exists():
        print(f"ERROR: Model checkpoint not found at {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    print(f"Loaded model from {model_path}")
    
    # ============================================================
    # 1. Baseline Evaluation
    # ============================================================
    baseline_data = evaluate_baseline(model, test_loader, cfg.device, global_stats)
    baseline_metrics = baseline_data['metrics']
    
    # ============================================================
    # 2. Ablation Study
    # ============================================================
    ablation_df = ablation_study(
        model, 
        test_loader, 
        cfg.device, 
        global_stats, 
        cfg.feature_groups
    )
    
    ablation_df.to_csv(Path(cfg.results_dir) / 'ablation_results.csv', index=False)
    print(f"\nSaved ablation results to {cfg.results_dir}/ablation_results.csv")
    
    plot_ablation_results(
        ablation_df, 
        baseline_metrics, 
        Path(cfg.results_dir) / 'ablation_importance.png'
    )
    
    # ============================================================
    # 3. Permutation Importance
    # ============================================================
    permutation_df = permutation_importance(
        model,
        test_loader,
        cfg.device,
        global_stats,
        cfg.feature_groups,
        n_repeats=5
    )
    
    permutation_df.to_csv(Path(cfg.results_dir) / 'permutation_results.csv', index=False)
    print(f"\nSaved permutation results to {cfg.results_dir}/permutation_results.csv")
    
    plot_permutation_results(
        permutation_df,
        baseline_metrics,
        Path(cfg.results_dir) / 'permutation_importance.png'
    )
    
    # ============================================================
    # 4. Interaction Analysis
    # ============================================================
    interaction_df = interaction_analysis(baseline_data, cfg)
    
    interaction_df.to_csv(Path(cfg.results_dir) / 'interaction_results.csv', index=False)
    print(f"\nSaved interaction results to {cfg.results_dir}/interaction_results.csv")
    
    plot_interaction_results(interaction_df, Path(cfg.results_dir))
    
    # ============================================================
    # 5. Summary Report
    # ============================================================
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE SUMMARY")
    print("="*60)
    
    print(f"\nBaseline Performance:")
    print(f"  R²: {baseline_metrics['r2']:.4f}")
    print(f"  RMSE: {baseline_metrics['rmse']:.2f} mm")
    print(f"  MAE: {baseline_metrics['mae']:.2f} mm")
    
    print(f"\nFeature Importance (Ablation - MAE increase):")
    for _, row in ablation_df.sort_values('mae', ascending=False).iterrows():
        mae_inc = row['mae'] - baseline_metrics['mae']
        print(f"  {row['feature_group']:20s}: +{mae_inc:6.2f} mm ({100*mae_inc/baseline_metrics['mae']:5.1f}% increase)")
    
    print(f"\n Feature importance analysis complete!")
    print(f"   Results saved to {cfg.results_dir}/")


if __name__ == "__main__":
    main()