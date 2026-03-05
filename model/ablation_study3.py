# module load anaconda/py3.11.7
# conda activate gapfill2

'''
Ablation study to understand feature importance for SWE prediction model.
Tests individual and grouped feature importance using:
1. Feature ablation (zeroing out features)
2. Permutation importance

Features:
- Channel 0: Forest Cover Fraction
- Channel 1: Elevation
- Channels 2-5: Passive Microwave (4 channels)
- Channel 6: VIIRS NDSI
- Channel 7: VIIRS Mask
- Channel 8: Noisy SWE (Unforested)  ← NEW
- Channel 9: Noisy SWE Mask  ← NEW

python ablation_study3.py --folder 'exp3_elevPM_NDSI_CC_1e-6_ps256_SmoothL1Loss'

'''

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os
from pathlib import Path
import zarr
from dictionaries import split_basin_dict, flight_to_basin
import json
from tqdm import tqdm
import pandas as pd

print('Starting ablation study...')

# ============================================================
# Model Definition (must match training script)
# ============================================================

class Model(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = self._make_layer(input_channels, 16)
        self.conv2 = self._make_layer(16, 32)
        self.conv3 = self._make_layer(32, 64)
        self.conv4 = self._make_layer(64, 128)
        self.conv5 = self._make_layer(128, 64)
        self.conv6 = self._make_layer(64, 32)
        self.conv7 = self._make_layer(32, 16)
        self.conv8 = self._make_layer(16, 8)
        self.conv100 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_uniform_(self.conv100.weight, nonlinearity='relu')

    def _make_layer(self, in_channels, out_channels, dropout_prob=0.2):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        nn.init.kaiming_uniform_(layer[0].weight, nonlinearity='relu')
        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv100(x)
        x = x.squeeze(1)
        return x

# ============================================================
# Evaluation Functions
# ============================================================

def compute_metrics(predictions, targets, masks):
    """Compute MAE, RMSE, R2 on valid (masked) pixels."""
    mask_flat = masks.reshape(-1)
    pred_flat = predictions.reshape(-1)[mask_flat]
    target_flat = targets.reshape(-1)[mask_flat]
    
    if len(pred_flat) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'n_pixels': 0}
    
    mae = mean_absolute_error(target_flat, pred_flat)
    rmse = np.sqrt(mean_squared_error(target_flat, pred_flat))
    r2 = r2_score(target_flat, pred_flat)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'n_pixels': len(pred_flat)
    }

def evaluate_model(model, test_x, test_y, test_masks, device):
    """Run model inference on test set and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_patch, y_patch, mask_patch in zip(test_x, test_y, test_masks):
            x_tensor = torch.from_numpy(x_patch).to(device, dtype=torch.float32)
            y_tensor = torch.from_numpy(y_patch).to(device, dtype=torch.float32)
            mask_tensor = torch.from_numpy(mask_patch).to(device, dtype=torch.bool)
            
            output = model(x_tensor)
            
            if len(y_tensor.shape) == 4 and y_tensor.shape[1] == 1:
                y_tensor = y_tensor.squeeze(1)
            if len(mask_tensor.shape) == 4 and mask_tensor.shape[1] == 1:
                mask_tensor = mask_tensor.squeeze(1)
            
            mask = mask_tensor
            
            valid_preds = output[mask].cpu().numpy()
            valid_labels = y_tensor[mask].cpu().numpy()
            
            all_preds.extend(valid_preds)
            all_labels.extend(valid_labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return all_preds, all_labels

# ============================================================
# Ablation Study Functions
# ============================================================

def ablate_features(test_x, channel_indices, ablation_type='zero'):
    """
    Create ablated version of test data.
    
    Args:
        test_x: List of test patches (each is (1, C, H, W))
        channel_indices: List of channel indices to ablate
        ablation_type: 'zero', 'noise', or 'mean'
    
    Returns:
        List of ablated test patches
    """
    ablated_x = []
    
    for x_patch in test_x:
        x_ablated = x_patch.copy()
        
        for ch_idx in channel_indices:
            if ablation_type == 'zero':
                x_ablated[:, ch_idx, :, :] = 0.0
            elif ablation_type == 'noise':
                # Replace with Gaussian noise (normalized, mean=0, std=1)
                x_ablated[:, ch_idx, :, :] = np.random.randn(*x_ablated[:, ch_idx, :, :].shape)
            elif ablation_type == 'mean':
                # Replace with channel mean
                ch_mean = np.nanmean(x_ablated[:, ch_idx, :, :])
                x_ablated[:, ch_idx, :, :] = ch_mean
        
        ablated_x.append(x_ablated)
    
    return ablated_x

def permutation_importance(model, test_x, test_y, test_masks, channel_indices, 
                          y_mean, y_std, device, n_repeats=5):
    """
    Compute permutation importance for specified channels.
    
    Args:
        n_repeats: Number of times to permute each feature
    """
    # Baseline performance
    baseline_preds, baseline_labels = evaluate_model(model, test_x, test_y, test_masks, device)
    
    # Denormalize
    baseline_preds_m = baseline_preds * y_std + y_mean
    baseline_labels_m = baseline_labels * y_std + y_mean
    baseline_preds_m = np.expm1(baseline_preds_m)
    baseline_labels_m = np.expm1(baseline_labels_m)
    
    baseline_rmse = np.sqrt(mean_squared_error(baseline_labels_m, baseline_preds_m))
    
    print(f"  Baseline RMSE: {baseline_rmse:.4f} m")
    
    importance_scores = []
    
    for repeat in range(n_repeats):
        # Permute specified channels
        permuted_x = []
        for x_patch in test_x:
            x_perm = x_patch.copy()
            
            # Randomly permute each channel spatially
            for ch_idx in channel_indices:
                original_shape = x_perm[:, ch_idx, :, :].shape
                flat = x_perm[:, ch_idx, :, :].reshape(-1)
                np.random.shuffle(flat)
                x_perm[:, ch_idx, :, :] = flat.reshape(original_shape)
            
            permuted_x.append(x_perm)
        
        # Evaluate with permuted features
        perm_preds, perm_labels = evaluate_model(model, permuted_x, test_y, test_masks, device)
        
        perm_preds_m = perm_preds * y_std + y_mean
        perm_labels_m = perm_labels * y_std + y_mean
        perm_preds_m = np.expm1(perm_preds_m)
        perm_labels_m = np.expm1(perm_labels_m)
        
        perm_rmse = np.sqrt(mean_squared_error(perm_labels_m, perm_preds_m))
        
        # Importance = increase in error
        importance = perm_rmse - baseline_rmse
        importance_scores.append(importance)
        
        print(f"    Repeat {repeat+1}: RMSE = {perm_rmse:.4f} m (Δ = {importance:+.4f} m)")
    
    mean_importance = np.mean(importance_scores)
    std_importance = np.std(importance_scores)
    
    return mean_importance, std_importance

# ============================================================
# Feature Group Definitions - UPDATED FOR 10 CHANNELS
# ============================================================

FEATURE_GROUPS = {
    'Forest Cover': [0],
    'Elevation': [1],
    'Microwave (all)': [2, 3, 4, 5],
    'Microwave Ch1': [2],
    'Microwave Ch2': [3],
    'Microwave Ch3': [4],
    'Microwave Ch4': [5],
    'VIIRS NDSI': [6],
    'VIIRS Mask': [7],
    'Noisy SWE (Unforested)': [8],  # NEW
    'Noisy SWE Mask': [9],  # NEW
    'VIIRS (both)': [6, 7],
    'Noisy SWE (both)': [8, 9],  # NEW - both noisy SWE channels
    'Topography (FC+Elev)': [0, 1],
    'All Remote Sensing': [2, 3, 4, 5, 6],
    'All SWE Features': [8, 9]  # NEW - same as 'Noisy SWE (both)'
}

# ============================================================
# Main Ablation Study
# ============================================================

def run_ablation_study(checkpoint_dir, test_x, test_y, test_masks, 
                       y_mean, y_std, device, save_dir):
    """
    Run complete ablation study with multiple methods.
    """
    print("\n" + "="*80)
    print("ABLATION STUDY: FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Load model
    print("\nLoading model from checkpoint...")
    model_path = os.path.join(checkpoint_dir, 'best_model_cnn.pth')
    
    # Detect number of input channels from first test patch
    input_channels = test_x[0].shape[1]
    print(f"Detected {input_channels} input channels")
    
    model = Model(input_channels=input_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Loaded model from: {model_path}")
    
    # Baseline performance
    print("\n" + "-"*80)
    print("BASELINE PERFORMANCE (all features)")
    print("-"*80)
    
    baseline_preds, baseline_labels = evaluate_model(model, test_x, test_y, test_masks, device)
    
    # Denormalize
    baseline_preds_m = baseline_preds * y_std + y_mean
    baseline_labels_m = baseline_labels * y_std + y_mean
    baseline_preds_m = np.expm1(baseline_preds_m)
    baseline_labels_m = np.expm1(baseline_labels_m)
    
    baseline_metrics = compute_metrics(baseline_preds_m, baseline_labels_m, 
                                       np.ones_like(baseline_preds_m, dtype=bool))
    
    print(f"MAE:  {baseline_metrics['mae']:.4f} m")
    print(f"RMSE: {baseline_metrics['rmse']:.4f} m")
    print(f"R²:   {baseline_metrics['r2']:.4f}")
    print(f"N pixels: {baseline_metrics['n_pixels']:,}")
    
    # ========================================
    # 1. FEATURE ABLATION (zeroing out)
    # ========================================
    print("\n" + "-"*80)
    print("METHOD 1: FEATURE ABLATION (Zero Replacement)")
    print("-"*80)
    
    ablation_results = {}
    
    for feature_name, channel_indices in tqdm(FEATURE_GROUPS.items(), desc="Ablating features"):
        print(f"\nAblating: {feature_name} (channels {channel_indices})")
        
        # Create ablated data
        ablated_x = ablate_features(test_x, channel_indices, ablation_type='zero')
        
        # Evaluate
        ablated_preds, ablated_labels = evaluate_model(model, ablated_x, test_y, test_masks, device)
        
        # Denormalize
        ablated_preds_m = ablated_preds * y_std + y_mean
        ablated_labels_m = ablated_labels * y_std + y_mean
        ablated_preds_m = np.expm1(ablated_preds_m)
        ablated_labels_m = np.expm1(ablated_labels_m)
        
        ablated_metrics = compute_metrics(ablated_preds_m, ablated_labels_m,
                                         np.ones_like(ablated_preds_m, dtype=bool))
        
        # Compute importance as performance drop
        mae_drop = ablated_metrics['mae'] - baseline_metrics['mae']
        rmse_drop = ablated_metrics['rmse'] - baseline_metrics['rmse']
        r2_drop = baseline_metrics['r2'] - ablated_metrics['r2']  # Decrease in R²
        
        ablation_results[feature_name] = {
            'channels': channel_indices,
            'mae': ablated_metrics['mae'],
            'rmse': ablated_metrics['rmse'],
            'r2': ablated_metrics['r2'],
            'mae_drop': mae_drop,
            'rmse_drop': rmse_drop,
            'r2_drop': r2_drop
        }
        
        print(f"  MAE:  {ablated_metrics['mae']:.4f} m (Δ = {mae_drop:+.4f} m)")
        print(f"  RMSE: {ablated_metrics['rmse']:.4f} m (Δ = {rmse_drop:+.4f} m)")
        print(f"  R²:   {ablated_metrics['r2']:.4f} (Δ = {r2_drop:+.4f})")
    
    # ========================================
    # 2. PERMUTATION IMPORTANCE
    # ========================================
    print("\n" + "-"*80)
    print("METHOD 2: PERMUTATION IMPORTANCE")
    print("-"*80)
    
    permutation_results = {}
    
    for feature_name, channel_indices in tqdm(FEATURE_GROUPS.items(), desc="Permuting features"):
        print(f"\nPermuting: {feature_name} (channels {channel_indices})")
        
        mean_imp, std_imp = permutation_importance(
            model, test_x, test_y, test_masks, channel_indices,
            y_mean, y_std, device, n_repeats=5
        )
        
        permutation_results[feature_name] = {
            'channels': channel_indices,
            'importance_mean': mean_imp,
            'importance_std': std_imp
        }
        
        print(f"  Importance: {mean_imp:.4f} ± {std_imp:.4f} m (RMSE increase)")
    
    # ========================================
    # Save results
    # ========================================
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as JSON
    results_dict = {
        'baseline_metrics': baseline_metrics,
        'ablation_results': ablation_results,
        'permutation_results': permutation_results
    }
    
    with open(os.path.join(save_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nSaved results to {save_dir}/ablation_results.json")
    
    return ablation_results, permutation_results, baseline_metrics

# ============================================================
# Visualization Functions - UPDATED FOR NEW CHANNELS
# ============================================================

def plot_feature_importance(ablation_results, permutation_results, baseline_metrics, save_dir):
    """
    Create comprehensive visualization of feature importance.
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Focus on key feature groups for main plot - UPDATED
    key_features = ['Forest Cover', 'Elevation', 'Microwave (all)', 'VIIRS NDSI', 
                    'VIIRS Mask', 'Noisy SWE (Unforested)', 'Noisy SWE Mask']
    
    # ========================================
    # PLOT 1: Side-by-side comparison
    # ========================================
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    features = [f for f in key_features if f in ablation_results]
    
    # Ablation - MAE drop
    mae_drops = [ablation_results[f]['mae_drop'] for f in features]
    axes[0].barh(features, mae_drops, color='steelblue', alpha=0.8, edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('MAE Increase (m)', fontsize=12, fontweight='bold')
    axes[0].set_title('Feature Ablation\n(Higher = More Important)', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Ablation - RMSE drop
    rmse_drops = [ablation_results[f]['rmse_drop'] for f in features]
    axes[1].barh(features, rmse_drops, color='coral', alpha=0.8, edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('RMSE Increase (m)', fontsize=12, fontweight='bold')
    axes[1].set_title('Feature Ablation\n(Higher = More Important)', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Permutation importance
    perm_means = [permutation_results[f]['importance_mean'] for f in features]
    perm_stds = [permutation_results[f]['importance_std'] for f in features]
    axes[2].barh(features, perm_means, xerr=perm_stds, color='seagreen', 
                 alpha=0.8, edgecolor='black', capsize=5)
    axes[2].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[2].set_xlabel('RMSE Increase (m)', fontsize=12, fontweight='bold')
    axes[2].set_title('Permutation Importance\n(Higher = More Important)', fontsize=14, fontweight='bold')
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/feature_importance_comparison.png")
    plt.close()
    
    # ========================================
    # PLOT 2: Detailed microwave channels
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    microwave_features = ['Microwave Ch1', 'Microwave Ch2', 'Microwave Ch3', 'Microwave Ch4']
    mw_features = [f for f in microwave_features if f in ablation_results]
    
    # Ablation
    mw_rmse_drops = [ablation_results[f]['rmse_drop'] for f in mw_features]
    axes[0].bar(range(len(mw_features)), mw_rmse_drops, color='skyblue', 
                alpha=0.8, edgecolor='black', width=0.6)
    axes[0].set_xticks(range(len(mw_features)))
    axes[0].set_xticklabels([f.replace('Microwave ', '') for f in mw_features], fontsize=11)
    axes[0].set_ylabel('RMSE Increase (m)', fontsize=12, fontweight='bold')
    axes[0].set_title('Microwave Channel Importance\n(Ablation)', fontsize=14, fontweight='bold')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Permutation
    mw_perm_means = [permutation_results[f]['importance_mean'] for f in mw_features]
    mw_perm_stds = [permutation_results[f]['importance_std'] for f in mw_features]
    axes[1].bar(range(len(mw_features)), mw_perm_means, yerr=mw_perm_stds,
                color='lightcoral', alpha=0.8, edgecolor='black', width=0.6, capsize=5)
    axes[1].set_xticks(range(len(mw_features)))
    axes[1].set_xticklabels([f.replace('Microwave ', '') for f in mw_features], fontsize=11)
    axes[1].set_ylabel('RMSE Increase (m)', fontsize=12, fontweight='bold')
    axes[1].set_title('Microwave Channel Importance\n(Permutation)', fontsize=14, fontweight='bold')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'microwave_channel_importance.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/microwave_channel_importance.png")
    plt.close()
    
    # ========================================
    # PLOT 3: VIIRS vs Static Features vs Noisy SWE - UPDATED
    # ========================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    comparison_features = ['Forest Cover', 'Elevation', 'VIIRS NDSI', 'VIIRS Mask',
                          'Noisy SWE (Unforested)', 'Noisy SWE Mask']
    comp_features = [f for f in comparison_features if f in ablation_results and f in permutation_results]
    
    x = np.arange(len(comp_features))
    width = 0.35
    
    ablation_vals = [ablation_results[f]['rmse_drop'] for f in comp_features]
    permutation_vals = [permutation_results[f]['importance_mean'] for f in comp_features]
    permutation_errs = [permutation_results[f]['importance_std'] for f in comp_features]
    
    bars1 = ax.bar(x - width/2, ablation_vals, width, label='Ablation (Zero)', 
                   color='steelblue', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, permutation_vals, width, yerr=permutation_errs,
                   label='Permutation', color='coral', alpha=0.8, edgecolor='black', capsize=5)
    
    ax.set_ylabel('RMSE Increase (m)', fontsize=14, fontweight='bold')
    ax.set_title('Feature Importance: VIIRS vs Static Features vs Noisy SWE\n(Higher = More Important)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comp_features, fontsize=11, rotation=25, ha='right')
    ax.legend(fontsize=12, loc='upper left')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom' if height >= 0 else 'top', 
                       fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'viirs_vs_static_vs_noisy_swe.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/viirs_vs_static_vs_noisy_swe.png")
    plt.close()
    
    # ========================================
    # PLOT 4: Summary table - UPDATED
    # ========================================
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table with updated features
    table_data = []
    table_data.append(['Feature', 'Channels', 'Ablation\nRMSE Δ (m)', 'Ablation\nR² Drop', 
                      'Permutation\nRMSE Δ (m)', 'Rank'])
    
    # Sort by ablation RMSE drop (descending)
    all_key_features = ['Forest Cover', 'Elevation', 'Microwave (all)', 'VIIRS NDSI', 
                        'VIIRS Mask', 'Noisy SWE (Unforested)', 'Noisy SWE Mask', 
                        'Noisy SWE (both)']
    
    sorted_features = sorted([f for f in all_key_features if f in ablation_results], 
                            key=lambda f: ablation_results.get(f, {}).get('rmse_drop', 0),
                            reverse=True)
    
    for rank, feature in enumerate(sorted_features, 1):
        if feature in ablation_results and feature in permutation_results:
            abl = ablation_results[feature]
            perm = permutation_results[feature]
            
            table_data.append([
                feature,
                str(abl['channels']),
                f"{abl['rmse_drop']:.4f}",
                f"{abl['r2_drop']:.4f}",
                f"{perm['importance_mean']:.4f} ± {perm['importance_std']:.4f}",
                str(rank)
            ])
    
    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.2, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Color code rows by importance
    for i in range(1, len(table_data)):
        rmse_drop = float(table_data[i][2])
        if rmse_drop > 0.1:
            color = '#FFE6E6'  # Light red - very important
        elif rmse_drop > 0.05:
            color = '#FFF4E6'  # Light orange - moderately important
        else:
            color = '#E6F7FF'  # Light blue - less important
        
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Feature Importance Summary\n(Sorted by Ablation RMSE Impact)', 
             fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(save_dir, 'feature_importance_table.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/feature_importance_table.png")
    plt.close()
    
    # ========================================
    # PLOT 5: Heatmap of all results - UPDATED
    # ========================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for heatmap
    all_features = list(FEATURE_GROUPS.keys())
    metrics = ['RMSE Δ', 'R² Drop']
    
    heatmap_data = []
    for feature in all_features:
        if feature in ablation_results and feature in permutation_results:
            row = [
                ablation_results[feature]['rmse_drop'],
                ablation_results[feature]['r2_drop'],
            ]
            heatmap_data.append(row)
        else:
            heatmap_data.append([0, 0])
    
    heatmap_array = np.array(heatmap_data).T  # Transpose for better layout
    
    # Normalize each metric to [0, 1] for better color comparison
    heatmap_norm = np.zeros_like(heatmap_array)
    for i in range(heatmap_array.shape[0]):
        row = heatmap_array[i, :]
        if row.max() > row.min():
            heatmap_norm[i, :] = (row - row.min()) / (row.max() - row.min())
    
    sns.heatmap(heatmap_norm, annot=heatmap_array, fmt='.3f', cmap='YlOrRd',
                xticklabels=all_features, yticklabels=metrics,
                cbar_kws={'label': 'Normalized Importance'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_title('Feature Importance Heatmap\n(All Metrics Normalized)', 
                fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'feature_importance_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/feature_importance_heatmap.png")
    plt.close()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)

# ============================================================
# Data Loading (reuse from training script)
# ============================================================

def load_full_zarr_files(zarr_dir, split_dict, flight_to_basin_dict, skip_tb_channels=True):
    """Load FULL zarr files - UPDATED to include noisy SWE channels."""
    zarr_dir = Path(zarr_dir)
    zarr_files = sorted(zarr_dir.glob("*.zarr"))
    
    test_x, test_y = [], []
    test_y_mask = []
    filenames_test = []
    
    print(f"Loading {len(zarr_files)} FULL zarr files...")
    
    for zarr_path in zarr_files:
        flight_id = zarr_path.stem
        tif_name = flight_id + '.tif'
        
        if tif_name not in flight_to_basin_dict:
            continue
        
        basin = flight_to_basin_dict[tif_name]
        
        # Only load TEST split
        split = None
        for split_name, basins in split_dict.items():
            if basin in basins:
                split = split_name
                break
        
        if split != 'test':
            continue
        
        # Load FULL zarr file
        z = zarr.open(str(zarr_path), mode='r')
        X = np.array(z['X'], dtype=np.float32)
        Y = np.array(z['Y'], dtype=np.float32)
        
        # Process channels - UPDATED for 10 channels
        if skip_tb_channels:
            # VIIRS NDSI (channel 8)
            viirs_raw = X[8, :, :]
            viirs_mask = (~np.isnan(viirs_raw)).astype(np.float32)
            viirs_filled = np.nan_to_num(viirs_raw, nan=0.0)
            
            # Get channels: FC, Elev, 4xMicrowave (channels 2-7)
            channels_except_viirs = [2, 3, 4, 5, 6, 7]
            all_but_viirs = X[channels_except_viirs, :, :]
            all_but_viirs[all_but_viirs == -9999] = 0
            all_but_viirs_filled = np.nan_to_num(all_but_viirs, nan=0.0)
            
            # Get noisy SWE channels (14 and 15)
            noisy_swe = X[14, :, :]  # Noisy SWE (Unforested)
            noisy_swe_mask = X[15, :, :]  # Noisy SWE Mask
            
            # Fill noisy SWE
            noisy_swe[noisy_swe == -9999] = 0
            noisy_swe = np.nan_to_num(noisy_swe, nan=0.0)
            noisy_swe_mask = np.nan_to_num(noisy_swe_mask, nan=0.0)
            
            # Concatenate all channels: FC, Elev, 4xMW, VIIRS NDSI, VIIRS Mask, Noisy SWE, Noisy SWE Mask
            X = np.concatenate([
                all_but_viirs_filled,  # 6 channels: FC, Elev, 4xMW
                viirs_filled[np.newaxis, :, :],  # 1 channel: VIIRS NDSI
                viirs_mask[np.newaxis, :, :],  # 1 channel: VIIRS Mask
                noisy_swe[np.newaxis, :, :],  # 1 channel: Noisy SWE
                noisy_swe_mask[np.newaxis, :, :]  # 1 channel: Noisy SWE Mask
            ], axis=0)
        
        Y[Y < 0] = np.nan
        Y[Y > 10.0] = np.nan
        Y_mask = ~np.isnan(Y)
        
        X = X[None, :, :, :]
        Y = Y[None, :, :, :]
        Y_mask = Y_mask[None, :, :, :]
        
        test_x.append(X)
        test_y.append(Y)
        test_y_mask.append(Y_mask)
        filenames_test.append(tif_name)
    
    print(f"\nLoaded: {len(test_x)} test FULL images")
    print(f"Input shape: {test_x[0].shape} (should have 10 channels)")
    
    return test_x, test_y, test_y_mask, filenames_test

def convert_to_patches(test_x, test_y, test_y_mask, filenames_test,
                      patch_size=128, stride=64, min_valid_fraction=0.3):
    """Convert full images to patches - simplified for test only."""
    print("\nConverting test images to patches...")
    
    patched_data = []
    patched_labels = []
    patched_masks = []
    total_patches = 0
    
    for img_idx, (data, label, mask, filename) in enumerate(zip(test_x, test_y, test_y_mask, filenames_test)):
        _, C, H, W = data.shape
        
        for row in range(0, H - patch_size + 1, stride):
            for col in range(0, W - patch_size + 1, stride):
                data_patch = data[:, :, row:row+patch_size, col:col+patch_size]
                label_patch = label[:, :, row:row+patch_size, col:col+patch_size]
                mask_patch = mask[:, :, row:row+patch_size, col:col+patch_size]
                
                valid_fraction = mask_patch.sum() / mask_patch.size
                
                if valid_fraction < min_valid_fraction:
                    continue
                
                patched_data.append(data_patch)
                patched_labels.append(label_patch)
                patched_masks.append(mask_patch)
                total_patches += 1
    
    print(f"Created {total_patches} test patches")
    
    return patched_data, patched_labels, patched_masks

def normalize_dataset_per_channel(test_data, norm_mean, norm_std, skip_channels=None):
    """Normalize using provided statistics."""
    if skip_channels is None:
        skip_channels = []
    
    normalized_test = []
    for data in test_data:
        normalized = (data - norm_mean) / (norm_std + 1e-7)
        normalized_test.append(normalized)
    
    return normalized_test

# ============================================================
# Forest Cover Analysis Function
# ============================================================

def analyze_importance_by_forest_cover(model, test_x, test_y, test_masks, 
                                       y_mean, y_std, device, save_dir, norm_mean, norm_std,
                                       n_bins=20):
    """
    Analyze how feature importance varies with forest cover fraction.
    UPDATED to include noisy SWE channels.
    """
    print("\n" + "="*80)
    print("ANALYZING FEATURE IMPORTANCE BY FOREST COVER FRACTION")
    print("="*80)
    
    # Define features to analyze - UPDATED
    features_to_test = {
        'Microwave Ch1': [2],
        'Microwave Ch2': [3],
        'Microwave Ch3': [4],
        'Microwave Ch4': [5],
        'VIIRS NDSI': [6],
        'Noisy SWE (Unforested)': [8],  # NEW
    }
    
    # Forest cover is channel 0
    FOREST_COVER_CHANNEL = 0
    
    # Collect all forest cover values and predictions for binning
    print("\nCollecting forest cover values from all patches...")
    all_forest_cover = []
    all_predictions = {}
    all_labels = []
    all_valid_masks = []
    
    # Get baseline predictions (all features)
    print("Computing baseline predictions...")
    model.eval()
    
    baseline_preds_list = []
    labels_list = []
    masks_list = []
    forest_cover_list = []
    
    with torch.no_grad():
        for x_patch, y_patch, mask_patch in tqdm(zip(test_x, test_y, test_masks), 
                                                   total=len(test_x), 
                                                   desc="Baseline predictions"):
            x_tensor = torch.from_numpy(x_patch).to(device, dtype=torch.float32)
            y_tensor = torch.from_numpy(y_patch).to(device, dtype=torch.float32)
            mask_tensor = torch.from_numpy(mask_patch).to(device, dtype=torch.bool)
            
            output = model(x_tensor)
            
            if len(y_tensor.shape) == 4 and y_tensor.shape[1] == 1:
                y_tensor = y_tensor.squeeze(1)
            if len(mask_tensor.shape) == 4 and mask_tensor.shape[1] == 1:
                mask_tensor = mask_tensor.squeeze(1)
            
            # Extract forest cover (normalized) from original input
            forest_cover = x_patch[0, FOREST_COVER_CHANNEL, :, :]  # (H, W)
            
            # Store pixel-level data
            baseline_preds_list.append(output.cpu().numpy())
            labels_list.append(y_tensor.cpu().numpy())
            masks_list.append(mask_tensor.cpu().numpy())
            forest_cover_list.append(forest_cover)
    
    # Flatten all arrays
    baseline_preds = np.concatenate([p.flatten() for p in baseline_preds_list])
    all_labels = np.concatenate([l.flatten() for l in labels_list])
    all_valid_masks = np.concatenate([m.flatten() for m in masks_list])
    all_forest_cover = np.concatenate([f.flatten() for f in forest_cover_list])
    
    # Filter to valid pixels only
    baseline_preds = baseline_preds[all_valid_masks]
    all_labels = all_labels[all_valid_masks]
    all_forest_cover = all_forest_cover[all_valid_masks]
    
    print(f"Total valid pixels: {len(all_forest_cover):,}")
    
    # Denormalize predictions and labels
    baseline_preds_m = baseline_preds * y_std + y_mean
    all_labels_m = all_labels * y_std + y_mean
    baseline_preds_m = np.expm1(baseline_preds_m)
    all_labels_m = np.expm1(all_labels_m)
    
    # Compute baseline error for each pixel
    baseline_errors = np.abs(baseline_preds_m - all_labels_m)
    
    print(f"Forest cover range (normalized): [{all_forest_cover.min():.3f}, {all_forest_cover.max():.3f}]")
    print(f"Baseline MAE: {baseline_errors.mean():.4f} m")
    
    # Now ablate each feature and compute predictions
    print("\nComputing ablated predictions for each feature...")
    
    ablated_predictions = {}
    
    for feature_name, channel_indices in features_to_test.items():
        print(f"\n  Ablating {feature_name} (channels {channel_indices})...")
        
        # Create ablated version of test data
        ablated_x = []
        for x_patch in test_x:
            x_ablated = x_patch.copy()
            for ch_idx in channel_indices:
                x_ablated[:, ch_idx, :, :] = 0.0
            ablated_x.append(x_ablated)
        
        # Get predictions with ablated feature
        ablated_preds_list = []
        
        with torch.no_grad():
            for x_patch in tqdm(ablated_x, desc=f"  {feature_name} predictions"):
                x_tensor = torch.from_numpy(x_patch).to(device, dtype=torch.float32)
                output = model(x_tensor)
                ablated_preds_list.append(output.cpu().numpy())
        
        # Flatten and filter
        ablated_preds = np.concatenate([p.flatten() for p in ablated_preds_list])
        ablated_preds = ablated_preds[all_valid_masks]
        
        # Denormalize
        ablated_preds_m = ablated_preds * y_std + y_mean
        ablated_preds_m = np.expm1(ablated_preds_m)
        
        ablated_predictions[feature_name] = ablated_preds_m
    
    # Bin by forest cover and compute importance
    print("\n" + "-"*80)
    print(f"Binning pixels by forest cover ({n_bins} bins)...")
    print("-"*80)
    
    # Denormalize forest cover for reporting
    forest_cover_mean = norm_mean[0, 0, 0, 0]  # Channel 0 mean
    forest_cover_std = norm_std[0, 0, 0, 0]    # Channel 0 std
    all_forest_cover_denorm = all_forest_cover * forest_cover_std + forest_cover_mean
    
    forest_cover_bins = np.linspace(all_forest_cover.min(), all_forest_cover.max(), n_bins + 1)
    bin_centers = (forest_cover_bins[:-1] + forest_cover_bins[1:]) / 2
    
    print(f"Bin edges: {forest_cover_bins}")
    
    # Assign each pixel to a bin
    bin_indices = np.digitize(all_forest_cover, forest_cover_bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Compute importance for each bin
    importance_by_bin = {name: [] for name in features_to_test.keys()}
    bin_counts = []
    bin_forest_cover_means = []
    
    for bin_idx in range(n_bins):
        mask = (bin_indices == bin_idx)
        n_pixels_in_bin = mask.sum()
        
        if n_pixels_in_bin < 10:
            for name in features_to_test.keys():
                importance_by_bin[name].append(np.nan)
            bin_counts.append(n_pixels_in_bin)
            bin_forest_cover_means.append(bin_centers[bin_idx])
            continue
        
        baseline_errors_bin = baseline_errors[mask]
        labels_bin = all_labels_m[mask]
        
        for feature_name in features_to_test.keys():
            ablated_preds_bin = ablated_predictions[feature_name][mask]
            ablated_errors_bin = np.abs(ablated_preds_bin - labels_bin)
            
            importance = ablated_errors_bin.mean() - baseline_errors_bin.mean()
            importance_by_bin[feature_name].append(importance)
        
        bin_counts.append(n_pixels_in_bin)
        bin_forest_cover_means.append(all_forest_cover[mask].mean())
        
        print(f"  Bin {bin_idx+1}: Forest cover {bin_forest_cover_means[-1]:.3f}, "
              f"N={n_pixels_in_bin:,} pixels")
    
    # Plot relative importance vs forest cover
    print("\nCreating plots...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Define colors for each feature - UPDATED
    colors = {
        'Microwave Ch1': '#1f77b4',
        'Microwave Ch2': '#ff7f0e',
        'Microwave Ch3': '#2ca02c',
        'Microwave Ch4': '#d62728',
        'VIIRS NDSI': '#9467bd',
        'Noisy SWE (Unforested)': '#8c564b'  # NEW - brown color
    }
    
    linestyles = {
        'Microwave Ch1': '-',
        'Microwave Ch2': '-',
        'Microwave Ch3': '-',
        'Microwave Ch4': '-',
        'VIIRS NDSI': '--',
        'Noisy SWE (Unforested)': '-.'  # NEW
    }
    
    markers = {
        'Microwave Ch1': 'o',
        'Microwave Ch2': 's',
        'Microwave Ch3': '^',
        'Microwave Ch4': 'D',
        'VIIRS NDSI': '*',
        'Noisy SWE (Unforested)': 'p'  # NEW - pentagon marker
    }
    
    # Plot 1: Absolute importance
    ax1 = axes[0]
    
    for feature_name in features_to_test.keys():
        importance_values = np.array(importance_by_bin[feature_name])
        
        ax1.plot(bin_forest_cover_means, importance_values,
                label=feature_name,
                color=colors[feature_name],
                linestyle=linestyles[feature_name],
                marker=markers[feature_name],
                markersize=8,
                linewidth=2.5,
                alpha=0.8)
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.set_xlabel('Forest Cover Fraction (normalized)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Feature Importance\n(MAE Increase when removed, m)', 
                   fontsize=13, fontweight='bold')
    ax1.set_title('Feature Importance vs Forest Cover Fraction\n(Including Noisy SWE)', 
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=bin_forest_cover_means[0], right=bin_forest_cover_means[-1])
    
    # Plot 2: Relative importance
    ax2 = axes[1]
    
    relative_importance = {name: [] for name in features_to_test.keys()}
    
    for bin_idx in range(n_bins):
        total_importance = 0
        for feature_name in features_to_test.keys():
            val = importance_by_bin[feature_name][bin_idx]
            if not np.isnan(val) and val > 0:
                total_importance += val
        
        for feature_name in features_to_test.keys():
            val = importance_by_bin[feature_name][bin_idx]
            if np.isnan(val) or total_importance == 0:
                relative_importance[feature_name].append(np.nan)
            else:
                relative_importance[feature_name].append(
                    max(0, val) / total_importance * 100 if total_importance > 0 else 0
                )
    
    for feature_name in features_to_test.keys():
        rel_values = np.array(relative_importance[feature_name])
        
        ax2.plot(bin_forest_cover_means, rel_values,
                label=feature_name,
                color=colors[feature_name],
                linestyle=linestyles[feature_name],
                marker=markers[feature_name],
                markersize=8,
                linewidth=2.5,
                alpha=0.8)
    
    ax2.set_xlabel('Forest Cover Fraction (normalized)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Relative Importance (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Relative Feature Importance vs Forest Cover Fraction', 
                  fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=bin_forest_cover_means[0], right=bin_forest_cover_means[-1])
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'importance_vs_forest_cover.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/importance_vs_forest_cover.png")
    plt.close()
    
    # Distribution plot
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.bar(bin_forest_cover_means, bin_counts, 
           width=(forest_cover_bins[1] - forest_cover_bins[0]) * 0.8,
           color='steelblue', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Forest Cover Fraction (normalized)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Pixels', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Pixels by Forest Cover Fraction', 
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.text(0.98, 0.97, f'Total pixels: {len(all_forest_cover):,}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'forest_cover_distribution.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/forest_cover_distribution.png")
    plt.close()
    
    # Save data as CSV
    results_df = pd.DataFrame({
        'forest_cover_bin_center': bin_forest_cover_means,
        'n_pixels': bin_counts,
        **{f'{name}_importance': importance_by_bin[name] for name in features_to_test.keys()},
        **{f'{name}_relative_importance': relative_importance[name] for name in features_to_test.keys()}
    })
    
    csv_path = os.path.join(save_dir, 'importance_vs_forest_cover.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY: Feature Importance Trends")
    print("="*80)
    
    for feature_name in features_to_test.keys():
        importance_values = np.array(importance_by_bin[feature_name])
        valid_values = importance_values[~np.isnan(importance_values)]
        
        if len(valid_values) > 0:
            print(f"\n{feature_name}:")
            print(f"  Mean importance: {valid_values.mean():.4f} m")
            print(f"  Std importance:  {valid_values.std():.4f} m")
            print(f"  Min importance:  {valid_values.min():.4f} m")
            print(f"  Max importance:  {valid_values.max():.4f} m")
            
            if len(valid_values) >= 3:
                from scipy.stats import pearsonr
                valid_forest = np.array(bin_forest_cover_means)[~np.isnan(importance_values)]
                if len(valid_forest) > 2:
                    corr, p_value = pearsonr(valid_forest, valid_values)
                    trend = "increases" if corr > 0 else "decreases"
                    print(f"  Trend: {trend} with forest cover (r={corr:.3f}, p={p_value:.4f})")
    
    print("\n" + "="*80)
    
    return results_df

# [Continue with analyze_per_flight_swe function - same as before but ensure it loads 10 channels]
# [Continue with run() function - updated to call new functions]

# ============================================================
# Main execution - UPDATED
# ============================================================

def run(folder):
    checkpoint_dir = f"/discover/nobackup/cmbreen/gap-filling/{folder}"
    zarr_dir = "/discover/nobackup/cmbreen/gap-filling-data/zarr_chunks"
    save_dir = os.path.join(checkpoint_dir, "ablation_study")
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    patch_size = 256
    stride = 128
    min_valid_fraction = 0.3
    
    print("\nLoading normalization statistics...")
    with open(os.path.join(checkpoint_dir, 'normalization_stats.json'), 'r') as f:
        norm_stats = json.load(f)
    
    norm_mean = np.array(norm_stats['X_mean']).reshape(1, -1, 1, 1).astype(np.float32)
    norm_std = np.array(norm_stats['X_std']).reshape(1, -1, 1, 1).astype(np.float32)
    y_mean = norm_stats['Y_mean']
    y_std = norm_stats['Y_std']
    
    print(f"Loaded normalization stats:")
    print(f"  X channels: {len(norm_stats['X_mean'])} (should be 10)")
    print(f"  Y mean: {y_mean:.4f}, Y std: {y_std:.4f}")
    
    # Load FULL images
    print("\n" + "="*80)
    print("LOADING FULL TEST IMAGES")
    print("="*80)
    
    test_x_full, test_y_full, test_y_mask_full, filenames_test = load_full_zarr_files(
        zarr_dir, split_basin_dict, flight_to_basin, skip_tb_channels=True
    )
    
    # Create patches
    print("\n" + "="*80)
    print("CREATING PATCHES FOR ABLATION ANALYSIS")
    print("="*80)
    
    test_x_patches, test_y_patches, test_y_mask_patches = convert_to_patches(
        test_x_full, test_y_full, test_y_mask_full, filenames_test,
        patch_size=patch_size, stride=stride, min_valid_fraction=min_valid_fraction
    )
    
    test_y_patches_log = [np.log1p(y) for y in test_y_patches]
    
    # Normalize - skip channels 7-9 are now wrong, should skip fewer if any
    test_x_patches_norm = normalize_dataset_per_channel(
        test_x_patches, norm_mean, norm_std, 
        skip_channels=[]  # Normalize all channels now
    )
    test_y_patches_norm = [(y - y_mean) / (y_std + 1e-7) for y in test_y_patches_log]
    
    print(f"\nPatched test data ready:")
    print(f"  {len(test_x_patches_norm)} patches")
    print(f"  Shape: {test_x_patches_norm[0].shape} (should be (1, 10, 256, 256))")
    
    # Normalize full images
    print("\nPreparing full images for per-flight analysis...")
    
    test_y_full_log = [np.log1p(y) for y in test_y_full]
    
    test_x_full_norm = normalize_dataset_per_channel(
        test_x_full, norm_mean, norm_std,
        skip_channels=[]
    )
    test_y_full_norm = [(y - y_mean) / (y_std + 1e-7) for y in test_y_full_log]
    
    print(f"Full images ready: {len(test_x_full_norm)} flights")
    
    # Load model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    
    model_path = os.path.join(checkpoint_dir, 'best_model_cnn.pth')
    input_channels = test_x_patches_norm[0].shape[1]
    print(f"Input channels: {input_channels} (should be 10)")
    
    model = Model(input_channels=input_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {model_path}")
    
    # Run ablation study
    print("\n" + "="*80)
    print("RUNNING FEATURE ABLATION STUDY")
    print("="*80)
    
    ablation_results, permutation_results, baseline_metrics = run_ablation_study(
        checkpoint_dir, test_x_patches_norm, test_y_patches_norm, test_y_mask_patches,
        y_mean, y_std, device, save_dir
    )
    
    plot_feature_importance(ablation_results, permutation_results, 
                           baseline_metrics, save_dir)
    
    # Forest cover analysis
    print("\n" + "="*80)
    print("RUNNING FOREST COVER ANALYSIS")
    print("="*80)
    
    forest_cover_results = analyze_importance_by_forest_cover(
        model, test_x_patches_norm, test_y_patches_norm, test_y_mask_patches,
        y_mean, y_std, device, save_dir, norm_mean, norm_std,
        n_bins=20
    )
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {save_dir}/")
    
    # Summary
    print("\n" + "-"*80)
    print("KEY FINDINGS")
    print("-"*80)
    
    sorted_by_ablation = sorted(ablation_results.items(),
                               key=lambda x: x[1]['rmse_drop'],
                               reverse=True)
    
    print("\nTop 5 most important features (by RMSE drop):")
    for i, (feature, results) in enumerate(sorted_by_ablation[:5], 1):
        print(f"  {i}. {feature}: RMSE Δ = {results['rmse_drop']:+.4f} m, "
              f"R² Drop = {results['r2_drop']:.4f}")
    
    print("\n" + "="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='model folder')
    parser.add_argument('--folder', type=str,
                       default='exp3_elevPM_NDSI_CC_1e-6_ps256_SmoothL1Loss')
    args = parser.parse_args()
    run(args.folder)

if __name__ == '__main__':
    main()