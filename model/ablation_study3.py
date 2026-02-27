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
                x_ablated[:, ch_idx, :, :] = np.random.randn(*x_ablated[:, ch_idx, :, :].shape)
            elif ablation_type == 'mean':
                ch_mean = np.nanmean(x_ablated[:, ch_idx, :, :])
                x_ablated[:, ch_idx, :, :] = ch_mean
        
        ablated_x.append(x_ablated)
    
    return ablated_x

def permutation_importance(model, test_x, test_y, test_masks, channel_indices, 
                          y_mean, y_std, device, n_repeats=5):
    """Compute permutation importance for specified channels."""
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
        permuted_x = []
        for x_patch in test_x:
            x_perm = x_patch.copy()
            
            for ch_idx in channel_indices:
                original_shape = x_perm[:, ch_idx, :, :].shape
                flat = x_perm[:, ch_idx, :, :].reshape(-1)
                np.random.shuffle(flat)
                x_perm[:, ch_idx, :, :] = flat.reshape(original_shape)
            
            permuted_x.append(x_perm)
        
        perm_preds, perm_labels = evaluate_model(model, permuted_x, test_y, test_masks, device)
        
        perm_preds_m = perm_preds * y_std + y_mean
        perm_labels_m = perm_labels * y_std + y_mean
        perm_preds_m = np.expm1(perm_preds_m)
        perm_labels_m = np.expm1(perm_labels_m)
        
        perm_rmse = np.sqrt(mean_squared_error(perm_labels_m, perm_preds_m))
        
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
    'Topography (FC+Elev)': [0, 1],
    'All Remote Sensing': [2, 3, 4, 5, 6],
    'All SWE Features': [8, 9]  # NEW - both noisy SWE channels
}

# ============================================================
# Main Ablation Study
# ============================================================

def run_ablation_study(checkpoint_dir, test_x, test_y, test_masks, 
                       y_mean, y_std, device, save_dir):
    """Run complete ablation study with multiple methods."""
    print("\n" + "="*80)
    print("ABLATION STUDY: FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Load model
    print("\nLoading model from checkpoint...")
    model_path = os.path.join(checkpoint_dir, 'best_model_cnn.pth')
    
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
    
    # Feature Ablation
    print("\n" + "-"*80)
    print("METHOD 1: FEATURE ABLATION (Zero Replacement)")
    print("-"*80)
    
    ablation_results = {}
    
    for feature_name, channel_indices in tqdm(FEATURE_GROUPS.items(), desc="Ablating features"):
        print(f"\nAblating: {feature_name} (channels {channel_indices})")
        
        ablated_x = ablate_features(test_x, channel_indices, ablation_type='zero')
        ablated_preds, ablated_labels = evaluate_model(model, ablated_x, test_y, test_masks, device)
        
        ablated_preds_m = ablated_preds * y_std + y_mean
        ablated_labels_m = ablated_labels * y_std + y_mean
        ablated_preds_m = np.expm1(ablated_preds_m)
        ablated_labels_m = np.expm1(ablated_labels_m)
        
        ablated_metrics = compute_metrics(ablated_preds_m, ablated_labels_m,
                                         np.ones_like(ablated_preds_m, dtype=bool))
        
        mae_drop = ablated_metrics['mae'] - baseline_metrics['mae']
        rmse_drop = ablated_metrics['rmse'] - baseline_metrics['rmse']
        r2_drop = baseline_metrics['r2'] - ablated_metrics['r2']
        
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
    
    # Permutation Importance
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
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    
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
# Visualization Functions (KEEPING ALL YOUR PLOTS)
# ============================================================

def plot_feature_importance(ablation_results, permutation_results, baseline_metrics, save_dir):
    """Create comprehensive visualization of feature importance."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Updated key features to include new channels
    key_features = ['Forest Cover', 'Elevation', 'Microwave (all)', 'VIIRS NDSI', 
                    'VIIRS Mask', 'Noisy SWE (Unforested)', 'Noisy SWE Mask']
    
    # PLOT 1: Side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    features = [f for f in key_features if f in ablation_results]
    
    mae_drops = [ablation_results[f]['mae_drop'] for f in features]
    axes[0].barh(features, mae_drops, color='steelblue', alpha=0.8, edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('MAE Increase (m)', fontsize=12, fontweight='bold')
    axes[0].set_title('Feature Ablation\n(Higher = More Important)', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    rmse_drops = [ablation_results[f]['rmse_drop'] for f in features]
    axes[1].barh(features, rmse_drops, color='coral', alpha=0.8, edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('RMSE Increase (m)', fontsize=12, fontweight='bold')
    axes[1].set_title('Feature Ablation\n(Higher = More Important)', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
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
    
    # PLOT 2: Microwave channels (unchanged)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    microwave_features = ['Microwave Ch1', 'Microwave Ch2', 'Microwave Ch3', 'Microwave Ch4']
    mw_features = [f for f in microwave_features if f in ablation_results]
    
    mw_rmse_drops = [ablation_results[f]['rmse_drop'] for f in mw_features]
    axes[0].bar(range(len(mw_features)), mw_rmse_drops, color='skyblue', 
                alpha=0.8, edgecolor='black', width=0.6)
    axes[0].set_xticks(range(len(mw_features)))
    axes[0].set_xticklabels([f.replace('Microwave ', '') for f in mw_features], fontsize=11)
    axes[0].set_ylabel('RMSE Increase (m)', fontsize=12, fontweight='bold')
    axes[0].set_title('Microwave Channel Importance\n(Ablation)', fontsize=14, fontweight='bold')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].grid(axis='y', alpha=0.3)
    
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
    
    # PLOT 3: VIIRS vs Static Features (updated to include noisy SWE)
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
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom' if height >= 0 else 'top', 
                       fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'viirs_vs_static_features.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/viirs_vs_static_features.png")
    plt.close()
    
    # PLOT 4: Summary table
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_features = ['Forest Cover', 'Elevation', 'Microwave (all)', 
                     'VIIRS NDSI', 'VIIRS Mask', 'Noisy SWE (Unforested)', 
                     'Noisy SWE Mask', 'All SWE Features']
    
    table_data = []
    for feat in table_features:
        if feat in ablation_results and feat in permutation_results:
            row = [
                feat,
                f"{ablation_results[feat]['rmse_drop']:+.4f}",
                f"{ablation_results[feat]['r2_drop']:+.4f}",
                f"{permutation_results[feat]['importance_mean']:.4f} ± {permutation_results[feat]['importance_std']:.4f}"
            ]
            table_data.append(row)
    
    # Add baseline row
    table_data.insert(0, [
        'BASELINE (all features)',
        f"{baseline_metrics['rmse']:.4f}",
        f"{baseline_metrics['r2']:.4f}",
        '-'
    ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Feature', 'RMSE Change (m)', 'R² Change', 'Permutation RMSE Δ (m)'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.3, 0.2, 0.2, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style baseline row
    for i in range(4):
        table[(1, i)].set_facecolor('#FFC000')
        table[(1, i)].set_text_props(weight='bold')
    
    # Alternate row colors
    for i in range(2, len(table_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    plt.title('Feature Importance Summary\n(Positive values = performance degradation when feature removed)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(save_dir, 'feature_importance_table.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/feature_importance_table.png")
    plt.close()
    
    # PLOT 5: Heatmap of feature importance
    fig, ax = plt.subplots(figsize=(10, 8))
    
    heatmap_features = ['Forest Cover', 'Elevation', 'Microwave (all)', 
                       'VIIRS NDSI', 'VIIRS Mask', 'Noisy SWE (Unforested)', 
                       'Noisy SWE Mask']
    heatmap_features = [f for f in heatmap_features if f in ablation_results]
    
    heatmap_data = []
    for feat in heatmap_features:
        heatmap_data.append([
            ablation_results[feat]['mae_drop'],
            ablation_results[feat]['rmse_drop'],
            ablation_results[feat]['r2_drop'],
            permutation_results[feat]['importance_mean']
        ])
    
    heatmap_df = pd.DataFrame(heatmap_data, 
                             index=heatmap_features,
                             columns=['MAE Δ', 'RMSE Δ', 'R² Δ', 'Perm. RMSE Δ'])
    
    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                center=0, linewidths=0.5, cbar_kws={'label': 'Impact (higher = more important)'},
                ax=ax)
    ax.set_title('Feature Importance Heatmap\n(Higher values = more important)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/feature_importance_heatmap.png")
    plt.close()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)

# ============================================================
# Data Loading Function
# ============================================================

def load_test_data(data_dir, test_basin):
    """Load test data from zarr files."""
    print(f"\nLoading test data for basin: {test_basin}")
    
    zarr_path = os.path.join(data_dir, f'{test_basin}_data.zarr')
    
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr file not found: {zarr_path}")
    
    root = zarr.open(zarr_path, mode='r')
    
    test_x = root['test_x'][:]
    test_y = root['test_y'][:]
    test_masks = root['test_masks'][:]
    y_mean = root.attrs['y_mean']
    y_std = root.attrs['y_std']
    
    print(f"Loaded test data:")
    print(f"  test_x shape: {test_x.shape}")
    print(f"  test_y shape: {test_y.shape}")
    print(f"  test_masks shape: {test_masks.shape}")
    print(f"  y_mean: {y_mean:.4f}")
    print(f"  y_std: {y_std:.4f}")
    
    # Convert to list of patches if needed
    if len(test_x.shape) == 4:
        test_x = [test_x]
        test_y = [test_y]
        test_masks = [test_masks]
    
    return test_x, test_y, test_masks, y_mean, y_std

# ============================================================
# Main Execution
# ============================================================

if __name__ == '__main__':
    # Configuration
    checkpoint_dir = '/path/to/your/checkpoint/directory'  # UPDATE THIS
    data_dir = '/path/to/your/data/directory'  # UPDATE THIS
    test_basin = 'your_test_basin'  # UPDATE THIS (e.g., 'tuolumne', 'american', etc.)
    save_dir = os.path.join(checkpoint_dir, 'ablation_study')
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load test data
    test_x, test_y, test_masks, y_mean, y_std = load_test_data(data_dir, test_basin)
    
    # Run ablation study
    ablation_results, permutation_results, baseline_metrics = run_ablation_study(
        checkpoint_dir=checkpoint_dir,
        test_x=test_x,
        test_y=test_y,
        test_masks=test_masks,
        y_mean=y_mean,
        y_std=y_std,
        device=device,
        save_dir=save_dir
    )
    
    # Create visualizations
    plot_feature_importance(ablation_results, permutation_results, baseline_metrics, save_dir)
    
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE!")
    print(f"Results saved to: {save_dir}")
    print("="*80)