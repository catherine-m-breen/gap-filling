# module load anaconda/py3.11.7
# conda activate gapfill2

'''
Ablation study to understand feature importance for SWE prediction model.
Tests individual and grouped feature importance using:
1. Feature ablation (zeroing out features)
2. Permutation importance
3. Dropout analysis

Features:
- Channel 0: Forest Cover Fraction
- Channel 1: Elevation
- Channels 2-5: Passive Microwave (4 channels)
- Channel 6: VIIRS NDSI
- Channel 7: VIIRS Mask


to run: 
Exp 1:
python ablation_study.py --folder "exp1_elevPM_NDSI_CC_1e-6_ps256_W2_smoothL1loss_full"

Exp 2:
python ablation_study2.py --folder "exp2_elevPM_NDSI_CC_1e-6_ps256_W2_smoothL1loss_full"

python ablation_study3_v2.py --folder "exp3_elevPM_NDSI_CC_1e-6_ps256_SmoothL1Loss"

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


##### EVAL FUNCTIONS ######

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


##############################################

def plot_sample_basin_patches(zarr_dir, sample_flight_id, patch_size=256, stride=128, 
                              min_valid_fraction=0.3, save_path=None):
    """
    Visualize a sample flight showing:
    1. Full image with patch grid overlay
    2. Sample of individual patches
    3. Distribution of valid pixels per patch
    
    Args:
        zarr_dir: Directory containing zarr files
        sample_flight_id: Flight ID to visualize (e.g., 'ASO_Dolores_2023Apr06_swe_50m')
        patch_size: Size of patches (default 256)
        stride: Stride for patch extraction (default 128)
        min_valid_fraction: Minimum valid pixel fraction to keep patch
        save_path: Path to save figure (optional)
    """
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    
    print(f"\nLoading sample flight: {sample_flight_id}")
    
    # Load the zarr file
    zarr_path = Path(zarr_dir) / f"{sample_flight_id}.zarr"
    if not zarr_path.exists():
        print(f"ERROR: {zarr_path} not found")
        return
    
    z = zarr.open(str(zarr_path), mode='r')
    X = np.array(z['X'], dtype=np.float32)
    Y = np.array(z['Y'], dtype=np.float32)
    
    print(f"Loaded data shape: X={X.shape}, Y={Y.shape}")
    
    # Get dimensions
    C, H, W = X.shape
    
    # Extract key channels for visualization
    forest_cover = X[2, :, :]  # Channel 2: Forest cover
    elevation = X[3, :, :]     # Channel 3: Elevation
    ndsi = X[8, :, :]          # Channel 8: VIIRS NDSI
    swe = Y[0, :, :]           # SWE target
    
    # Create mask
    swe_mask = ~np.isnan(swe) & (swe >= 0) & (swe <= 10.0)
    
    # Generate patch locations and check validity
    patch_info = []
    
    for row in range(0, H - patch_size + 1, stride):
        for col in range(0, W - patch_size + 1, stride):
            mask_patch = swe_mask[row:row+patch_size, col:col+patch_size]
            valid_fraction = mask_patch.sum() / mask_patch.size
            
            patch_info.append({
                'row': row,
                'col': col,
                'valid_fraction': valid_fraction,
                'is_valid': valid_fraction >= min_valid_fraction
            })
    
    print(f"\nPatch statistics:")
    print(f"  Patch size: {patch_size}x{patch_size}, Stride: {stride}")
    print(f"  Total possible patches: {len(patch_info)}")
    print(f"  Valid patches (>{min_valid_fraction*100:.0f}% valid): "
          f"{sum(p['is_valid'] for p in patch_info)}")
    
    # ========================================
    # CREATE FIGURE
    # ========================================
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # ========================================
    # ROW 1: Full images with patch grid
    # ========================================
    
    # Plot 1: SWE with patch grid
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(swe, cmap='viridis', vmin=0, vmax=np.nanpercentile(swe, 95))
    
    # Overlay patch grid
    for p in patch_info:
        color = 'lime' if p['is_valid'] else 'red'
        alpha = 0.3 if p['is_valid'] else 0.15
        rect = Rectangle((p['col'], p['row']), patch_size, patch_size,
                        linewidth=1, edgecolor=color, facecolor='none', alpha=alpha)
        ax1.add_patch(rect)
    
    ax1.set_title(f'SWE (m)\nGreen=Valid patches, Red=Rejected', 
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Plot 2: Forest cover
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(forest_cover, cmap='Greens', vmin=0, vmax=100)
    ax2.set_title('Forest Cover (%)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Plot 3: Elevation
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(elevation, cmap='terrain')
    ax3.set_title('Elevation (m)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Plot 4: NDSI
    ax4 = fig.add_subplot(gs[0, 3])
    ndsi_valid = np.where(~np.isnan(ndsi), ndsi, 0)
    im4 = ax4.imshow(ndsi_valid, cmap='Blues', vmin=0, vmax=1)
    ax4.set_title('VIIRS NDSI', fontsize=12, fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # ========================================
    # ROW 2: Sample valid patches
    # ========================================
    valid_patches = [p for p in patch_info if p['is_valid']]
    
    if len(valid_patches) >= 4:
        # Select 4 patches: corners and center
        indices = [
            0,  # Top-left
            len(valid_patches) // 3,  # Left-center
            2 * len(valid_patches) // 3,  # Right-center
            -1  # Bottom-right
        ]
        
        for i, idx in enumerate(indices):
            p = valid_patches[idx]
            row, col = p['row'], p['col']
            
            ax = fig.add_subplot(gs[1, i])
            
            # Extract patch
            swe_patch = swe[row:row+patch_size, col:col+patch_size]
            mask_patch = swe_mask[row:row+patch_size, col:col+patch_size]
            
            # Plot with mask overlay
            im = ax.imshow(swe_patch, cmap='viridis', vmin=0, 
                          vmax=np.nanpercentile(swe, 95))
            
            # Overlay invalid pixels in red
            invalid_overlay = np.zeros((patch_size, patch_size, 4))
            invalid_overlay[~mask_patch] = [1, 0, 0, 0.3]  # Red with transparency
            ax.imshow(invalid_overlay)
            
            ax.set_title(f'Patch {i+1}\nValid: {p["valid_fraction"]*100:.1f}%\n'
                        f'Position: ({row}, {col})',
                        fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # ========================================
    # ROW 3: Statistics
    # ========================================
    
    # Plot 1: Valid fraction histogram
    ax_hist = fig.add_subplot(gs[2, :2])
    valid_fractions = [p['valid_fraction'] for p in patch_info]
    
    ax_hist.hist(valid_fractions, bins=50, color='steelblue', 
                alpha=0.7, edgecolor='black')
    ax_hist.axvline(x=min_valid_fraction, color='red', linestyle='--', 
                   linewidth=2, label=f'Threshold ({min_valid_fraction*100:.0f}%)')
    ax_hist.set_xlabel('Valid Pixel Fraction', fontsize=12, fontweight='bold')
    ax_hist.set_ylabel('Number of Patches', fontsize=12, fontweight='bold')
    ax_hist.set_title('Distribution of Valid Pixels per Patch', 
                     fontsize=13, fontweight='bold')
    ax_hist.legend(fontsize=11)
    ax_hist.grid(alpha=0.3)
    
    # Plot 2: Patch coverage map
    ax_coverage = fig.add_subplot(gs[2, 2:])
    
    # Create coverage map showing number of times each pixel is covered
    coverage_map = np.zeros((H, W), dtype=int)
    for p in patch_info:
        if p['is_valid']:
            coverage_map[p['row']:p['row']+patch_size, 
                        p['col']:p['col']+patch_size] += 1
    
    im_cov = ax_coverage.imshow(coverage_map, cmap='hot', interpolation='nearest')
    ax_coverage.set_title('Patch Coverage Map\n(Times each pixel is included)', 
                         fontsize=13, fontweight='bold')
    ax_coverage.axis('off')
    cbar = plt.colorbar(im_cov, ax=ax_coverage, fraction=0.046, pad=0.04)
    cbar.set_label('Coverage count', fontsize=11)
    
    # ========================================
    # Add overall title with metadata
    # ========================================
    basin = flight_to_basin.get(f"{sample_flight_id}.tif", "Unknown")
    fig.suptitle(f'Sample Flight Patch Visualization\n'
                f'{sample_flight_id} ({basin})\n'
                f'Image: {H}x{W} | Patches: {sum(p["is_valid"] for p in patch_info)}/{len(patch_info)} valid',
                fontsize=16, fontweight='bold', y=0.98)
    
    # ========================================
    # Save or show
    # ========================================
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved figure to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return patch_info
#############################################




#############
################### ABLATION STUDY ######################
#############

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

## FEATURE GROUPS FOR ABLATION STUDY
## check what happens if missing channels
## This is the same for Exp1 and Exp2 because all we are changing is whether we predict forest or unforest pixels 

FEATURE_GROUPS = {
    'Forest Cover': [0],
    'Elevation': [1],
    'PM (all)': [2, 3, 4, 5],
    'Tb 37H': [2],
    'Tb 37V': [3],
    'Tb 19H': [4],
    'Tb 19V': [5],
    'VIIRS NDSI': [6],
    'VIIRS Mask': [7],
    'VIIRS (both)': [6, 7],
    'Topography (FC+Elev)': [0, 1],
    'All Remote Sensing': [2, 3, 4, 5, 6]
}


########## Run the ablation study 

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
# Visualization Functions
# ============================================================

def plot_feature_importance(ablation_results, permutation_results, baseline_metrics, save_dir):
    """
    Create comprehensive visualization of feature importance.
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Focus on key feature groups for main plot
    key_features = ['Forest Cover', 'Elevation', 'Microwave (all)', 'VIIRS NDSI', 'VIIRS Mask']
    
    # ========================================
    # PLOT 1: Side-by-side comparison
    # ========================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
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
    
    # # Permutation importance
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
    
    microwave_features = ['Tb 37H', 'Tb 37V', 'Tb 19H', 'Tb 19V']
    mw_features = [f for f in microwave_features if f in ablation_results]
    
    # Ablation
    mw_rmse_drops = [ablation_results[f]['rmse_drop'] for f in mw_features]
    axes[0].bar(range(len(mw_features)), mw_rmse_drops, color='skyblue', 
                alpha=0.8, edgecolor='black', width=0.6)
    axes[0].set_xticks(range(len(mw_features)))
    axes[0].set_xticklabels([f.replace('Microwave ', '') for f in mw_features], fontsize=11)
    axes[0].set_ylabel('RMSE Increase (m)', fontsize=12, fontweight='bold')
    axes[0].set_title('PM Channel Importance\n(Ablation)', fontsize=14, fontweight='bold')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Permutation
    mw_perm_means = [permutation_results[f]['importance_mean'] for f in mw_features]
    mw_perm_stds = [permutation_results[f]['importance_std'] for f in mw_features]
    axes[1].bar(range(len(mw_features)), mw_perm_means, yerr=mw_perm_stds,
                color='lightcoral', alpha=0.8, edgecolor='black', width=0.6, capsize=5)
    axes[1].set_xticks(range(len(mw_features)))
    axes[1].set_xticklabels([f.replace('PM ', '') for f in mw_features], fontsize=11)
    axes[1].set_ylabel('RMSE Increase (m)', fontsize=12, fontweight='bold')
    axes[1].set_title('PM Importance\n(Permutation)', fontsize=14, fontweight='bold')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'microwave_channel_importance.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/microwave_channel_importance.png")
    plt.close()
    
    # ========================================
    # PLOT 3: VIIRS vs Forest Cover vs Elevation
    # ========================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    comparison_features = ['Forest Cover', 'Elevation', 'VIIRS NDSI', 'VIIRS Mask', 'VIIRS (both)']
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
    ax.set_title('Feature Importance: VIIRS vs Static Features\n(Higher = More Important)', 
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comp_features, fontsize=12, rotation=15, ha='right')
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
    plt.savefig(os.path.join(save_dir, 'viirs_vs_static_features.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/viirs_vs_static_features.png")
    plt.close()
    
    # ========================================
    # PLOT 4: Summary table
    # ========================================
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table
    table_data = []
    table_data.append(['Feature', 'Channels', 'Ablation\nRMSE Δ (m)', 'Ablation\nR² Drop', 
                      'Permutation\nRMSE Δ (m)', 'Rank'])
    
    # Sort by ablation RMSE drop (descending)
    sorted_features = sorted(key_features, 
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
    # PLOT 5: Heatmap of all results
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare data for heatmap
    all_features = list(FEATURE_GROUPS.keys())
    metrics = ['MAE Δ', 'RMSE Δ', 'R² Drop', 'Perm. Imp.']
    metrics = ['RMSE Δ', 'R² Drop']
    
    heatmap_data = []
    for feature in all_features:
        if feature in ablation_results and feature in permutation_results:
            row = [
             #   ablation_results[feature]['mae_drop'],
                ablation_results[feature]['rmse_drop'],
                ablation_results[feature]['r2_drop'],
              #  permutation_results[feature]['importance_mean']
            ]
            heatmap_data.append(row)
        else:
            heatmap_data.append([0, 0, 0, 0])
    
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
    """Load FULL zarr files - copied from training script."""
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
        
        # Process channels (same as training)
        if skip_tb_channels:
            viirs_raw = X[8, :, :]
            viirs_mask = (~np.isnan(viirs_raw)).astype(np.float32)
            viirs_filled = np.nan_to_num(viirs_raw, nan=0.0)
            
            channels_except_viirs = [2, 3, 4, 5, 6, 7]
            all_but_viirs = X[channels_except_viirs, :, :]
            all_but_viirs[all_but_viirs == -9999] = 0
            all_but_viirs_filled = np.nan_to_num(all_but_viirs, nan=0.0)
            
            X = np.concatenate([
                all_but_viirs_filled,
                viirs_filled[np.newaxis, :, :],
                viirs_mask[np.newaxis, :, :]
            ], axis=0)
        
        # import IPython
        # IPython.embed()
        Y[Y < 0] = np.nan
        Y[Y > 10.0] = np.nan
        Y_mask = ~np.isnan(Y)

        canopy_cover = X[2, :, :]  # Original X, channel 2 = tree cover
        # Mask out forested pixels (tree cover > 40%) for exp2
        Y[0, canopy_cover <= 40] = np.nan  # Note: > 40, not <= 40 for exp2!

        # Create final mask
        Y_mask = ~np.isnan(Y[0])  # Shape: (H, W)
        
        
        X = X[None, :, :, :]
        Y = Y[None, :, :, :]
      #  import IPython
       # IPython.embed()
        Y_mask = Y_mask if len(Y_mask) == 2 else Y_mask.squeeze()


        ####### now do the X part -- 2 new channels #### 
        Y_unforested = np.zeros_like(Y[0])  # (H, W) - initialize with zeros from tree channel
        unforested_mask = (X[:,2, :, :] <= 40)
        Y_unforested[unforested_mask] = Y[0, unforested_mask] # make an unforested mask of Y where pixels are >= 40

        # Add Gaussian noise to unforested areas only
        noise = np.random.normal(loc=0, scale=0.25, size=Y_unforested.shape)  # 25cm noise
        Y_unforested[unforested_mask] += noise[unforested_mask]
        Y_unforested = np.maximum(Y_unforested, 0) ## clip because we can't have negative values 

        Y_unforested = np.nan_to_num(Y_unforested, nan=0.0)
        Y_unforested_mask = (Y_unforested > 0).astype(np.float32)

        # ### now do the Y part 
        # Y[0, X[0, :, :] <= 40] = np.nan
        # Y_mask = ~np.isnan(Y)  # Boolean mask: True where valid, False where NaN ## pass this through so we can only look where we have data! 

        ## then you need to concantenate this at the end!!! 
        X = np.concatenate([X,Y_unforested[np.newaxis, :, :], Y_unforested_mask[np.newaxis, :, :]], axis=1)

        ############################
        
        test_x.append(X)
        test_y.append(Y)
        test_y_mask.append(Y_mask)
        filenames_test.append(tif_name)
    
    print(f"\nLoaded: {len(test_x)} test FULL images")
    
    return test_x, test_y, test_y_mask, filenames_test

def convert_to_patches(test_x, test_y, test_y_mask, filenames_test,
                      patch_size=128, stride=64, min_valid_fraction=0.3):
    """Convert full images to patches - simplified for test only."""
    print("\nConverting test images to patches...")
    
    patched_data = []
    patched_labels = []
    patched_masks = []
    total_patches = 0

    # import IPython 
    # IPython.embed()
    # Ensure mask is 2D (H, W)
    ## basically add some dummy dimensions 

    for img_idx, (data, label, mask, filename) in enumerate(zip(test_x, test_y, test_y_mask, filenames_test)):
        _, C, H, W = data.shape
        #import IPython
        #IPython.embed()
        # if mask.ndim == 3:
        #     mask = mask.squeeze()  # Remove extra dimensions
        # elif mask.ndim == 4:
        mask = mask[None, None, :, :]  # Extract (H, W) from (1, 1, H, W)
        

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


# Add this new function to your ablation study script

def analyze_importance_by_forest_cover(model, test_x, test_y, test_masks, 
                                       y_mean, y_std, device, save_dir, norm_mean, norm_std,
                                       n_bins=20):
    """
    Analyze how feature importance varies with forest cover fraction.
    
    Creates a plot showing relative importance of VIIRS and 4 microwave channels
    as a function of forest cover fraction.
    
    Args:
        model: Trained model
        test_x: List of test patches (normalized, (1, C, H, W))
        test_y: List of test labels (normalized, (1, 1, H, W))
        test_masks: List of masks (1, 1, H, W)
        y_mean, y_std: Normalization parameters for Y
        device: torch device
        save_dir: Directory to save plots
        n_bins: Number of forest cover bins (default 10 = 0-10%, 10-20%, etc.)
    """
    print("\n" + "="*80)
    print("ANALYZING FEATURE IMPORTANCE BY FOREST COVER FRACTION")
    print("="*80)
    
    # Define features to analyze
    features_to_test = {
        'Tb 37H': [2],
        'Tb 37V': [3],
        'Tb 19H': [4],
        'Tb 19V': [5],
        'VIIRS NDSI': [6],
        'Noisy SWE': [8],
    }
    
    # Forest cover is channel 0
    FOREST_COVER_CHANNEL = 0
    
    # Create bins for forest cover (0-100%)
    # Note: forest cover is normalized, so we need to work with normalized values
    # but report in original units
    
    # Collect all forest cover values and predictions for binning
    print("\nCollecting forest cover values from all patches...")
    all_forest_cover = []
    all_predictions = {}  # Will store predictions for each feature ablation
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
    
    # ========================================
    # Bin by forest cover and compute importance
    # ========================================
    print("\n" + "-"*80)
    print(f"Binning pixels by forest cover ({n_bins} bins)...")
    print("-"*80)
    
    # Create bins based on percentiles for equal sample sizes
    # Or use linear bins from min to max

    forest_cover_mean = norm_mean[0, 0, 0, 0]  # Channel 0 mean
    forest_cover_std = norm_std[0, 0, 0, 0]    # Channel 0 std

    # Denormalize to original scale
    all_forest_cover_denorm = all_forest_cover * forest_cover_std + forest_cover_mean
    all_forest_cover = all_forest_cover_denorm
    #forest_cover_bins = np.linspace(all_forest_cover.min(), all_forest_cover.max(), n_bins + 1)
    
    forest_cover_bins = np.array([40, 60, 80, 100])
    n_bins = len(forest_cover_bins) - 1  # 5 bins
    bin_centers = (forest_cover_bins[:-1] + forest_cover_bins[1:]) / 2
    
    # Alternative: use percentile-based bins for equal samples
    # forest_cover_bins = np.percentile(all_forest_cover, np.linspace(0, 100, n_bins + 1))
    
    print(f"Bin edges: {forest_cover_bins}")
    
    # Assign each pixel to a bin
    bin_indices = np.digitize(all_forest_cover, forest_cover_bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Handle edge cases
    
    # Compute importance for each bin
    importance_by_bin = {name: [] for name in features_to_test.keys()}
    bin_counts = []
    bin_forest_cover_means = []
    
    for bin_idx in range(n_bins):
        mask = (bin_indices == bin_idx)
        n_pixels_in_bin = mask.sum()
        
        if n_pixels_in_bin < 10:  # Skip bins with too few pixels
            for name in features_to_test.keys():
                importance_by_bin[name].append(np.nan)
            bin_counts.append(n_pixels_in_bin)
            bin_forest_cover_means.append(bin_centers[bin_idx])
            continue
        
        # Get baseline errors for this bin
        baseline_errors_bin = baseline_errors[mask]
        labels_bin = all_labels_m[mask]
        
        # Compute importance for each feature
        for feature_name in features_to_test.keys():
            ablated_preds_bin = ablated_predictions[feature_name][mask]
            ablated_errors_bin = np.abs(ablated_preds_bin - labels_bin)
            #ablated_errors_bin = (ablated_preds_bin - labels_bin) ** 2  # Squared errors
            
            # Importance = increase in MAE when feature is removed
            importance = ablated_errors_bin.mean() - baseline_errors_bin.mean()
            importance_by_bin[feature_name].append(importance)
        
        bin_counts.append(n_pixels_in_bin)
        bin_forest_cover_means.append(all_forest_cover[mask].mean())
        
        print(f"  Bin {bin_idx+1}: Forest cover {bin_forest_cover_means[-1]:.3f}, "
              f"N={n_pixels_in_bin:,} pixels")
    
    # ========================================
    # Plot relative importance vs forest cover
    # ========================================
    print("\nCreating plots...")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Define colors for each feature
    colors = {
        'Tb 37H': '#1f77b4',  # Blue
        'Tb 37V': '#ff7f0e',  # Orange
        'Tb 19H': '#2ca02c',  # Green
        'Tb 19V': '#d62728',  # Red
        'VIIRS NDSI': '#9467bd',      # Purple
        'Noisy SWE': 'black'
    }
    
    linestyles = {
        'Tb 37H': '-',
        'Tb 37V': '-',
        'Tb 19H': '-',
        'Tb 19V': '-',
        'VIIRS NDSI': '--',
        'Noisy SWE': '-',

    }
    
    markers = {
        'Tb 37H': 'o',
        'Tb 37V': 's',
        'Tb 19H': '^',
        'Tb 19V': 'D',
        'VIIRS NDSI': '*',
        'Noisy SWE': '-',
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
    ax1.set_xlabel('Forest Cover Fraction', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Feature Importance\n(MAE Increase when removed, m)', 
                   fontsize=13, fontweight='bold')
    ax1.set_title('Feature Importance vs Forest Cover Fraction', 
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0, right=100) #right=bin_forest_cover_means[-1]) bin_forest_cover_means[0]
    ax1.tick_params(axis='both', labelsize=16)  # Add this to increase tick label size
    
    # Plot 2: Relative importance (normalized to sum to 1 in each bin)
    ax2 = axes[1]
    
    # Compute relative importance (as fraction of total importance)
    relative_importance = {name: [] for name in features_to_test.keys()}
    
    for bin_idx in range(n_bins):
        total_importance = 0
        for feature_name in features_to_test.keys():
            val = importance_by_bin[feature_name][bin_idx]
            if not np.isnan(val) and val > 0:  # Only positive contributions
                total_importance += val
        
        for feature_name in features_to_test.keys():
            val = importance_by_bin[feature_name][bin_idx]
            if np.isnan(val) or total_importance == 0:
                relative_importance[feature_name].append(np.nan)
            else:
                # Compute percentage of total importance
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
    
    ax2.set_xlabel('Forest Cover Fraction', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Relative Importance (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Relative Feature Importance vs Forest Cover Fraction', 
                  fontsize=15, fontweight='bold')
    ax2.legend(fontsize=11, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(left=bin_forest_cover_means[0], right=100) # right=bin_forest_cover_means[-1])
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'importance_vs_forest_cover.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/importance_vs_forest_cover.png")
    plt.close()
    
    # ========================================
    # Additional plot: Sample distribution
    # ========================================
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.bar(bin_forest_cover_means, bin_counts, 
           width=(forest_cover_bins[1] - forest_cover_bins[0]) * 0.8,
           color='steelblue', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Forest Cover Fraction (normalized)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Pixels', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Pixels by Forest Cover Fraction', 
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text with total pixels
    ax.text(0.98, 0.97, f'Total pixels: {len(all_forest_cover):,}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'forest_cover_distribution.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/forest_cover_distribution.png")
    plt.close()
    
    # ========================================
    # Save data as CSV
    # ========================================
    results_df = pd.DataFrame({
        'forest_cover_bin_center': bin_forest_cover_means,
        'n_pixels': bin_counts,
        **{f'{name}_importance': importance_by_bin[name] for name in features_to_test.keys()},
        **{f'{name}_relative_importance': relative_importance[name] for name in features_to_test.keys()}
    })
    
    csv_path = os.path.join(save_dir, 'importance_vs_forest_cover.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # ========================================
    # Print summary statistics
    # ========================================
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
            
            # Check for trends
            if len(valid_values) >= 3:
                # Simple linear trend check
                from scipy.stats import pearsonr
                valid_forest = np.array(bin_forest_cover_means)[~np.isnan(importance_values)]
                if len(valid_forest) > 2:
                    corr, p_value = pearsonr(valid_forest, valid_values)
                    trend = "increases" if corr > 0 else "decreases"
                    print(f"  Trend: {trend} with forest cover (r={corr:.3f}, p={p_value:.4f})")
    
    print("\n" + "="*80)
    
    return results_df



def analyze_per_flight_swe(model, test_x_full, test_y_full, test_masks_full, 
                           filenames_test, y_mean, y_std, device, save_dir,
                           pixel_area_m2=50*50):
    """
    Analyze per-flight SWE predictions and compute total missing SWE.
    
    Args:
        model: Trained model
        test_x_full: List of FULL test images (not patches), normalized (1, C, H, W)
        test_y_full: List of FULL test labels (not patches), normalized (1, 1, H, W)
        test_masks_full: List of FULL test masks (1, 1, H, W)
        filenames_test: List of flight filenames
        y_mean, y_std: Normalization parameters
        device: torch device
        save_dir: Directory to save results
        pixel_area_m2: Area of each pixel in square meters (default 50m x 50m = 2500 m²)
    
    Returns:
        DataFrame with per-flight statistics
    """
    print("\n" + "="*80)
    print("PER-FLIGHT SWE ANALYSIS")
    print("="*80)
    
    model.eval()
    
    flight_results = []
    
    print(f"\nProcessing {len(test_x_full)} flights...")
    print(f"Pixel area: {pixel_area_m2} m² ({np.sqrt(pixel_area_m2):.1f}m x {np.sqrt(pixel_area_m2):.1f}m)")
    
    with torch.no_grad():
        for idx, (x_full, y_full, mask_full, filename) in enumerate(
            zip(test_x_full, test_y_full, test_masks_full, filenames_test)):
            
            print(f"\n[{idx+1}/{len(test_x_full)}] Processing {filename}...")
            
            # Move to device
            x_tensor = torch.from_numpy(x_full).to(device, dtype=torch.float32)
            y_tensor = torch.from_numpy(y_full).to(device, dtype=torch.float32)
            mask_tensor = torch.from_numpy(mask_full).to(device, dtype=torch.bool)
            
            # Get prediction for full image
            output = model(x_tensor)
            
            # Ensure proper dimensions
            if len(y_tensor.shape) == 4 and y_tensor.shape[1] == 1:
                y_tensor = y_tensor.squeeze(1)
            if len(mask_tensor.shape) == 4 and mask_tensor.shape[1] == 1:
                mask_tensor = mask_tensor.squeeze(1)
            
            # Convert to numpy
            pred_norm = output.cpu().numpy()  # (batch, H, W)
            true_norm = y_tensor.cpu().numpy()  # (batch, H, W)
            mask = mask_tensor.cpu().numpy()  # (batch, H, W)
            
            # Remove batch dimension
            pred_norm = pred_norm[0]  # (H, W)
            true_norm = true_norm[0]  # (H, W)
            #mask = mask[0]  # (H, W)
            
            # Denormalize
            pred_m = pred_norm * y_std + y_mean
            true_m = true_norm * y_std + y_mean
            
            # Undo log transform
            pred_m = np.expm1(pred_m)
            true_m = np.expm1(true_m)
            
            # Apply mask to get valid pixels only
            # import IPython 
            # IPython.embed()
            pred_valid = pred_m[mask]
            true_valid = true_m[mask]
            
            n_valid_pixels = len(pred_valid)
            
            if n_valid_pixels == 0:
                print(f"  WARNING: No valid pixels in {filename}, skipping")
                continue
            
            # ========================================
            # Compute pixel-level metrics
            # ========================================
            errors = pred_valid - true_valid
            abs_errors = np.abs(errors)
            
            mae = abs_errors.mean()
            rmse = np.sqrt((errors ** 2).mean())
            r2 = r2_score(true_valid, pred_valid)
            bias = errors.mean()  # Mean bias (positive = overprediction)
            
            # ========================================
            # Compute total SWE volumes
            # ========================================
            # SWE in meters × pixel area in m² = volume in m³
            # 1 m³ of water = 1000 liters = 1 metric ton
            
            true_volume_m3 = (true_valid * pixel_area_m2).sum()
            pred_volume_m3 = (pred_valid * pixel_area_m2).sum()
            
            # Missing volume (positive = underprediction, negative = overprediction)
            missing_volume_m3 = true_volume_m3 - pred_volume_m3
            
            # # Convert to more intuitive units
            # true_volume_acre_ft = true_volume_m3 / 1233.48  # 1 acre-foot = 1233.48 m³
            # pred_volume_acre_ft = pred_volume_m3 / 1233.48
            # missing_volume_acre_ft = missing_volume_m3 / 1233.48
            
            # # Metric tons (1 m³ water ≈ 1 metric ton)
            # true_volume_metric_tons = true_volume_m3
            # pred_volume_metric_tons = pred_volume_m3
            # missing_volume_metric_tons = missing_volume_m3
            
            # Percent error
            volume_percent_error = (missing_volume_m3 / true_volume_m3 * 100) if true_volume_m3 > 0 else 0
            
            # ========================================
            # Spatial statistics
            # ========================================
            mean_true_swe = true_valid.mean()
            mean_pred_swe = pred_valid.mean()
            std_true_swe = true_valid.std()
            std_pred_swe = pred_valid.std()
            
            max_true_swe = true_valid.max()
            max_pred_swe = pred_valid.max()
            
            # ========================================
            # Store results
            # ========================================
            flight_results.append({
                'flight': filename,
                'basin': flight_to_basin.get(filename, 'Unknown'),
                'n_pixels': n_valid_pixels,
                'area_km2': n_valid_pixels * pixel_area_m2 / 1e6,
                
                # Pixel-level metrics
                'mae_m': mae,
                'rmse_m': rmse,
                'r2': r2,
                'bias_m': bias,
                
                # Mean SWE
                'mean_true_swe_m': mean_true_swe,
                'mean_pred_swe_m': mean_pred_swe,
                'std_true_swe_m': std_true_swe,
                'std_pred_swe_m': std_pred_swe,
                'max_true_swe_m': max_true_swe,
                'max_pred_swe_m': max_pred_swe,
                
                # Total volumes
                'true_volume_m3': true_volume_m3,
                'pred_volume_m3': pred_volume_m3,
                'missing_volume_m3': missing_volume_m3,
                
                # 'true_volume_acre_ft': true_volume_acre_ft,
                # 'pred_volume_acre_ft': pred_volume_acre_ft,
                # 'missing_volume_acre_ft': missing_volume_acre_ft,
                
                # 'true_volume_metric_tons': true_volume_metric_tons,
                # 'pred_volume_metric_tons': pred_volume_metric_tons,
                # 'missing_volume_metric_tons': missing_volume_metric_tons,
                
                'volume_percent_error': volume_percent_error
            })
            
            print(f"  Valid pixels: {n_valid_pixels:,} ({n_valid_pixels * pixel_area_m2 / 1e6:.2f} km²)")
            print(f"  MAE: {mae:.3f} m, RMSE: {rmse:.3f} m, R²: {r2:.3f}")
            print(f"  Mean SWE - True: {mean_true_swe:.3f} m, Pred: {mean_pred_swe:.3f} m")
            print(f"  Total volume - True: {true_volume_m3:,.0f} m³, "
                f"Pred: {pred_volume_m3:,.0f} m³")
            print(f"  Missing: {missing_volume_m3:+,.0f} m³ ({volume_percent_error:+.1f}%)")
                
       
    # Create DataFrame and compute totals

    results_df = pd.DataFrame(flight_results)
    
    # Sort by missing volume (descending - most underpredicted first)
    results_df = results_df.sort_values('missing_volume_m3', ascending=False)
    #results_df = results_df.sort_values('true_volume_m3', ascending=False)
    
    # Compute totals
    total_true_volume = results_df['true_volume_m3'].sum()
    total_pred_volume = results_df['pred_volume_m3'].sum()
    total_missing_volume = results_df['missing_volume_m3'].sum()
    total_missing_percent = (total_missing_volume / total_true_volume * 100) if total_true_volume > 0 else 0
    
    total_area_km2 = results_df['area_km2'].sum()
    mean_mae = results_df['mae_m'].mean()
    mean_rmse = results_df['rmse_m'].mean()
    mean_r2 = results_df['r2'].mean()
    
    # ========================================
    # Print summary
    # ========================================
    print("\n" + "="*80)
    print("SUMMARY: TOTAL SWE ACROSS ALL FLIGHTS")
    print("="*80)
    print(f"\nTotal area: {total_area_km2:.1f} km²")
    print(f"Total flights: {len(results_df)}")
    print(f"Total valid pixels: {results_df['n_pixels'].sum():,}")
    
    print(f"\nOverall Performance:")
    print(f"  Mean MAE:  {mean_mae:.3f} m")
    print(f"  Mean RMSE: {mean_rmse:.3f} m")
    print(f"  Mean R²:   {mean_r2:.3f}")
    
    # Breakdown by over/under prediction
    overpredicted = results_df[results_df['missing_volume_m3'] < 0]
    underpredicted = results_df[results_df['missing_volume_m3'] > 0]
    
    print(f"\nBreakdown:")
    print(f"  Underpredicted flights: {len(underpredicted)} "
        f"(missing {underpredicted['missing_volume_m3'].sum():,.0f} m³)")
    print(f"  Overpredicted flights: {len(overpredicted)} "
        f"(excess {-overpredicted['missing_volume_m3'].sum():,.0f} m³)")
    
    # ========================================
    # Save results
    # ========================================
    csv_path = os.path.join(save_dir, 'per_flight_swe_analysis.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed results to: {csv_path}")
    
    # ========================================
    # VISUALIZATION 1: Bar plot of missing SWE per flight
    # ========================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color by sign (red = missing, blue = excess)
    colors = ['red' if x > 0 else 'blue' for x in results_df['missing_volume_m3']]
    
    bars = ax.barh(range(len(results_df)), results_df['missing_volume_m3'],
                   color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels([f.replace('.tif', '') for f in results_df['flight']], fontsize=8)
    ax.set_xlabel('Missing SWE (m)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Flight Missing SWE Volume\n(Red = Underprediction, Blue = Overprediction)', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax.grid(axis='x', alpha=0.3)
    
    # Add text box with totals
    textstr = (f'Total Missing: {total_missing_volume:+,.0f} m³\n'
            f'({total_missing_percent:+.1f}%)\n'
            f'Flights: {len(results_df)}')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_flight_missing_swe.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/per_flight_missing_swe.png")
    plt.close()
    
    # ========================================
    # VISUALIZATION 2: Scatter plot - true vs predicted volume
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(results_df['true_volume_m3'], results_df['pred_volume_m3'],
               c=results_df['volume_percent_error'], cmap='RdBu_r', 
               s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
               vmin=-50, vmax=50)
    
    # 1:1 line
    max_vol = max(results_df['true_volume_m3'].max(), 
                  results_df['pred_volume_m3'].max())
    ax.plot([0, max_vol], [0, max_vol], 'k--', linewidth=2, alpha=0.5, label='1:1 Line')
    
    ax.set_xlabel('True SWE Volume (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted SWE Volume (m)', fontsize=13, fontweight='bold')
    ax.set_title('Per-Flight Total SWE Volume: Predicted vs True', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(ax.collections[0], ax=ax, label='Volume Error (%)')
    cbar.set_label('Volume Error (%)', fontsize=11, fontweight='bold')
    
    # Add text with R²
    volume_r2 = r2_score(results_df['true_volume_m3'], 
                         results_df['pred_volume_m3'])
    textstr = f'R² = {volume_r2:.3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_flight_volume_scatter.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/per_flight_volume_scatter.png")
    plt.close()
    
    # ========================================
    # VISUALIZATION 3: Summary table (top 10 most missing + top 10 most excess)
    # ========================================
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Top 10 most underpredicted
    top_under = results_df.nlargest(10, 'missing_volume_m3')
    
    # Top 10 most overpredicted
    top_over = results_df.nsmallest(10, 'missing_volume_m3')
    
    # Create table
    table_data = []
    table_data.append(['Flight', 'Basin', 'Area (km²)', 'Mean True\nSWE (m)', 
                  'Mean Pred\nSWE (m)', 'RMSE (m)', 'Missing\nVolume (m³)', 
                  'Error (%)'])
    
    # Add underpredicted
    for _, row in top_under.iterrows():
        table_data.append([
            row['flight'].replace('.tif', '')[:30],
            row['basin'][:15],
            f"{row['area_km2']:.1f}",
            f"{row['mean_true_swe_m']:.2f}",
            f"{row['mean_pred_swe_m']:.2f}",
            f"{row['rmse_m']:.3f}",
            f"{row['missing_volume_m3']:+,.0f}",
            f"{row['volume_percent_error']:+.1f}%"
        ])
    # Separator
    table_data.append(['─'*30] * 8)
    
    # Add overpredicted
    for _, row in top_over.iterrows():
        table_data.append([
            row['flight'].replace('.tif', '')[:30],
            row['basin'][:15],
            f"{row['area_km2']:.1f}",
            f"{row['mean_true_swe_m']:.2f}",
            f"{row['mean_pred_swe_m']:.2f}",
            f"{row['rmse_m']:.3f}",
            f"{row['missing_volume_m3']:+,.0f}",
            f"{row['volume_percent_error']:+.1f}%"
        ])

    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                    colWidths=[0.20, 0.12, 0.08, 0.10, 0.10, 0.08, 0.15, 0.08])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=10)
    
    # Color code rows
    for i in range(1, len(table_data)):
        if '─' in table_data[i][0]:
            continue
        
        # Color by missing volume
        if i <= 10:  # Underpredicted
            color = '#FFE6E6'  # Light red
        else:  # Overpredicted
            color = '#E6F0FF'  # Light blue
        
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Per-Flight SWE Analysis\nTop 10 Most Underpredicted and Top 10 Most Overpredicted', 
             fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(save_dir, 'per_flight_table.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/per_flight_table.png")
    plt.close()
    
    # ========================================
    # VISUALIZATION 4: Basin-level summary
    # ========================================
    basin_summary = results_df.groupby('basin').agg({
        'n_pixels': 'sum',
        'area_km2': 'sum',
        'true_volume_m3': 'sum',
        'pred_volume_m3': 'sum',
        'missing_volume_m3': 'sum',
        'mae_m': 'mean',
        'rmse_m': 'mean',
        'r2': 'mean'
    }).reset_index()

    basin_summary['volume_percent_error'] = (
        basin_summary['missing_volume_m3'] / 
        basin_summary['true_volume_m3'] * 100
    )
        
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Basin missing volume
    colors_basin = ['red' if x > 0 else 'blue' 
                    for x in basin_summary['missing_volume_m3']]
    axes[0].barh(basin_summary['basin'], basin_summary['missing_volume_m3'],
                color=colors_basin, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Missing SWE Volume (m³)', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='both', labelsize=16)  # Add this to increase tick label size
    axes[0].grid(axis='x', alpha=0.3)
    
    # Basin missing volume as percent 

    axes[1].barh(basin_summary['basin'], basin_summary['missing_volume_m3']/ basin_summary['true_volume_m3'] * 100 ,
                color=colors_basin, alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Missing SWE Volume (%)', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=16)  # Add this to increase tick label size
    axes[1].grid(axis='x', alpha=0.3)
    # Basin RMSE
    # axes[1].barh(basin_summary['basin'], basin_summary['rmse_m'],
    #             color='steelblue', alpha=0.7, edgecolor='black')
    # axes[1].set_xlabel('RMSE (m)', fontsize=12, fontweight='bold')
    # axes[1].set_title('Basin-Level Prediction Error', fontsize=14, fontweight='bold')
    # axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'basin_summary.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/basin_summary.png")
    plt.close()
    
    # Save basin summary
    basin_csv_path = os.path.join(save_dir, 'basin_summary.csv')
    basin_summary.to_csv(basin_csv_path, index=False)
    print(f"Saved: {basin_csv_path}")
    
    print("\n" + "="*80)
    print("PER-FLIGHT SWE ANALYSIS COMPLETE")
    print("="*80)
    
    return results_df, basin_summary

def reconstruct_and_plot_flight(model, zarr_dir, sample_flight_id, 
                                norm_mean, norm_std, y_mean, y_std,
                                patch_size=256, stride=128, min_valid_fraction=0.3,
                                device='cuda', save_path=None):
    """
    Reconstruct a full flight prediction from patches and plot side-by-side with actual.
    """
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    
    print(f"\nReconstructing flight: {sample_flight_id}")
    
    # Load the zarr file
    zarr_path = Path(zarr_dir) / f"{sample_flight_id}.zarr"
    if not zarr_path.exists():
        print(f"ERROR: {zarr_path} not found")
        return
    
    z = zarr.open(str(zarr_path), mode='r')
    X = np.array(z['X'], dtype=np.float32)
    Y = np.array(z['Y'], dtype=np.float32)
    
    print(f"Loaded data shape: X={X.shape}, Y={Y.shape}")
    
    # Process channels (same as training)
    # viirs_raw = X[8, :, :]
    # viirs_mask = (~np.isnan(viirs_raw)).astype(np.float32)
    # viirs_filled = np.nan_to_num(viirs_raw, nan=0.0)
    
    # channels_except_viirs = [2, 3, 4, 5, 6, 7]
    # all_but_viirs = X[channels_except_viirs, :, :]
    # all_but_viirs[all_but_viirs == -9999] = 0
    # all_but_viirs_filled = np.nan_to_num(all_but_viirs, nan=0.0)
    
    # X_processed = np.concatenate([
    #     all_but_viirs_filled,
    #     viirs_filled[np.newaxis, :, :],
    #     viirs_mask[np.newaxis, :, :]
    # ], axis=0)
    
    # Get dimensions
    # C, H, W = X_processed.shape

    #########################################
         # Load FULL zarr file
    # z = zarr.open(str(zarr_path), mode='r')
    # X = np.array(z['X'], dtype=np.float32)
    # Y = np.array(z['Y'], dtype=np.float32)
        
        # Process channels (same as training)
    # if skip_tb_channels:
    viirs_raw = X[8, :, :]
    viirs_mask = (~np.isnan(viirs_raw)).astype(np.float32)
    viirs_filled = np.nan_to_num(viirs_raw, nan=0.0)
        
    channels_except_viirs = [2, 3, 4, 5, 6, 7]
    all_but_viirs = X[channels_except_viirs, :, :]
    all_but_viirs[all_but_viirs == -9999] = 0
    all_but_viirs_filled = np.nan_to_num(all_but_viirs, nan=0.0)
        
    X = np.concatenate([
        all_but_viirs_filled,
        viirs_filled[np.newaxis, :, :],
        viirs_mask[np.newaxis, :, :]
    ], axis=0)
        
        # import IPython
        # IPython.embed()
    Y[Y < 0] = np.nan
    Y[Y > 10.0] = np.nan
    Y_mask = ~np.isnan(Y)

    canopy_cover = X[2, :, :]  # Original X, channel 2 = tree cover
    # Mask out forested pixels (tree cover > 40%) for exp2
    Y[0, canopy_cover <= 40] = np.nan  # Note: > 40, not <= 40 for exp2!

    # Create final mask
    Y_mask = ~np.isnan(Y[0])  # Shape: (H, W)
    
    
    X = X[None, :, :, :]
    Y = Y[None, :, :, :]
    #  import IPython
    # IPython.embed()
    Y_mask = Y_mask if len(Y_mask) == 2 else Y_mask.squeeze()


    ####### now do the X part -- 2 new channels #### 
    Y_unforested = np.zeros_like(Y[0])  # (H, W) - initialize with zeros from tree channel
    unforested_mask = (X[:,2, :, :] <= 40)
    Y_unforested[unforested_mask] = Y[0, unforested_mask] # make an unforested mask of Y where pixels are >= 40

    # Add Gaussian noise to unforested areas only
    noise = np.random.normal(loc=0, scale=0.25, size=Y_unforested.shape)  # 25cm noise
    Y_unforested[unforested_mask] += noise[unforested_mask]
    Y_unforested = np.maximum(Y_unforested, 0) ## clip because we can't have negative values 

    Y_unforested = np.nan_to_num(Y_unforested, nan=0.0)
    Y_unforested_mask = (Y_unforested > 0).astype(np.float32)

    # ### now do the Y part 
    # Y[0, X[0, :, :] <= 40] = np.nan
    # Y_mask = ~np.isnan(Y)  # Boolean mask: True where valid, False where NaN ## pass this through so we can only look where we have data! 

    ## then you need to concantenate this at the end!!! 
    X_processed = np.concatenate([X,Y_unforested[np.newaxis, :, :], Y_unforested_mask[np.newaxis, :, :]], axis=1)

    #########################################

    _, C, H, W = X_processed.shape
    
    print(f"Processed X shape: {X_processed.shape}")
    print(f"Norm mean shape: {norm_mean.shape}")
    print(f"Norm std shape: {norm_std.shape}")
    
    # ========================================
    # FIX: Extract normalization parameters correctly
    # ========================================
    # norm_mean and norm_std have shape (1, C, 1, 1)
    # We need shape (C, 1, 1) for broadcasting with (C, H, W)
    
    import IPython 
    IPython.embed()
    if norm_mean.shape == (1, C, 1, 1):
        # Squeeze out the batch dimension
        norm_mean_2d = norm_mean[0, :, 0, 0]  # Shape: (C,)
        norm_std_2d = norm_std[0, :, 0, 0]    # Shape: (C,)
        
        # Reshape for broadcasting with (C, H, W)
        norm_mean_2d = norm_mean_2d[:, np.newaxis, np.newaxis]  # Shape: (C, 1, 1)
        norm_std_2d = norm_std_2d[:, np.newaxis, np.newaxis]    # Shape: (C, 1, 1)
    else:
        raise ValueError(f"Unexpected norm_mean shape: {norm_mean.shape}")
    
    print(f"Reshaped norm_mean: {norm_mean_2d.shape}")
    print(f"Reshaped norm_std: {norm_std_2d.shape}")
    
        #### I think we did this earlier ! ####
    # Create mask for Y 
    # Y[Y < 0] = np.nan
    # Y[Y > 10.0] = np.nan
    # Y_mask = ~np.isnan(Y[0])

    # canopy_cover = X[2, :, :]  # Original X, channel 2 = tree cover
    # # Mask out forested pixels (tree cover > 40%) for exp2
    # Y[0, canopy_cover <= 40] = np.nan  # Note: > 40, not <= 40 for exp2!

    # # Create final mask
    # Y_mask = ~np.isnan(Y[0])  # Shape: (H, W)

    # # Add batch dimension for consistency: (1, C, H, W)
    # X = X[None, :, :, :]
    # Y = Y[None, :, :, :]
    # #Y_mask = Y_mask[None, :, :].squeeze()
    # Y_mask = Y_mask if len(Y_mask) == 2 else Y_mask.squeeze()

    # Initialize reconstruction arrays
    reconstruction = np.zeros((H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)
    
    # Store patch info for visualization
    valid_patch_locations = []
    
    print("\nProcessing patches...")
    model.eval()
    
    patch_count = 0
    valid_patch_count = 0
    
    X_processed = X_processed[0]  # Remove batch dim: (10, 854, 952)

    with torch.no_grad():
        for row in tqdm(range(0, H - patch_size + 1, stride), desc="Rows"):
            for col in range(0, W - patch_size + 1, stride):
                patch_count += 1
                
                # Extract patch
                x_patch = X_processed[:, row:row+patch_size, col:col+patch_size]
                y_patch = Y[0, row:row+patch_size, col:col+patch_size]
                mask_patch = Y_mask[row:row+patch_size, col:col+patch_size]
                
                # Check validity
                valid_fraction = mask_patch.sum() / mask_patch.size
                
                if valid_fraction < min_valid_fraction:
                    continue
                
                valid_patch_count += 1
                valid_patch_locations.append((row, col))
                
                # Prepare patch for model
                # Log transform
                y_patch_log = np.log1p(y_patch)
                
                # ========================================
                # FIX: Normalize X using corrected shapes
                # ========================================
                # x_patch: (C, H, W) = (8, 256, 256)
                # norm_mean_2d: (C, 1, 1) = (8, 1, 1)
                # This will broadcast correctly!
                x_patch_norm = (x_patch - norm_mean_2d) / (norm_std_2d + 1e-7)
                
                # Verify shape
                assert x_patch_norm.shape == x_patch.shape, \
                    f"Normalization changed shape: {x_patch.shape} -> {x_patch_norm.shape}"
                
                # Normalize Y
                y_patch_norm = (y_patch_log - y_mean) / (y_std + 1e-7)
                
                # Add batch dimension
                x_tensor = torch.from_numpy(x_patch_norm[np.newaxis, :, :, :]).to(device, dtype=torch.float32)
                
                # Predict
                output = model(x_tensor)
                pred_patch_norm = output.cpu().numpy()[0]  # Remove batch dim
                
                # Denormalize prediction
                pred_patch_log = pred_patch_norm * y_std + y_mean
                pred_patch = np.expm1(pred_patch_log)
                
                # Add to reconstruction with weights
                # Use Gaussian weights to reduce edge artifacts
                center = patch_size // 2
                y_grid, x_grid = np.ogrid[:patch_size, :patch_size]
                dist_from_center = np.sqrt((x_grid - center)**2 + (y_grid - center)**2)
                weights = np.exp(-(dist_from_center**2) / (2 * (patch_size/4)**2))
                
                # Accumulate weighted predictions
                reconstruction[row:row+patch_size, col:col+patch_size] += pred_patch * weights
                weight_map[row:row+patch_size, col:col+patch_size] += weights
    
    print(f"\nProcessed {valid_patch_count}/{patch_count} valid patches")
    
    # Normalize by weights (average overlapping predictions)
    reconstruction = np.divide(reconstruction, weight_map, 
                              out=np.zeros_like(reconstruction), 
                              where=weight_map > 0)
    
    # Apply original mask (only show predictions where we have ground truth)
    reconstruction[~Y_mask] = np.nan
    
    
    # Get actual SWE
    # import IPython
    # IPython.embed()
    actual_swe = Y[0]
    
    # Compute error map
    error_map = reconstruction - actual_swe
    abs_error_map = np.abs(error_map)
    
    # Compute metrics
    valid_mask = Y_mask
    # pred_valid = reconstruction[valid_mask]
    
    pred_valid = reconstruction.copy()
    pred_valid[~valid_mask] = np.nan
    actual_valid = actual_swe.squeeze().copy() 
    actual_valid[~valid_mask] = np.nan
    
   # Use NaN-aware functions
    mae = np.nanmean(np.abs(pred_valid - actual_valid))
    rmse = np.sqrt(np.nanmean((pred_valid - actual_valid)**2))
    bias = np.nanmean(pred_valid - actual_valid)

    # For R², sklearn doesn't handle NaNs, so extract valid pixels as 1D arrays
    pred_valid_1d = pred_valid[valid_mask]  # Flatten to 1D, only valid pixels
    actual_valid_1d = actual_valid[valid_mask]  # Flatten to 1D, only valid pixels
    r2 = r2_score(actual_valid_1d, pred_valid_1d)

    print(f"\nReconstruction Metrics:")
    print(f"  MAE:  {mae:.4f} m")
    print(f"  RMSE: {rmse:.4f} m")
    print(f"  R²:   {r2:.4f}")
    print(f"  Bias: {bias:+.4f} m")

    # import IPython
    # IPython.embed
    # ========================================
    # CREATE VISUALIZATION
    # ========================================
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.25)
    
    # Determine common color scale for SWE
    vmin_swe = 0
    vmax_swe = np.nanpercentile(actual_swe, 98)
    
    # ========================================
    # ROW 1: Main comparison
    # ========================================
    
    # Plot 1: Actual SWE
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(actual_valid, cmap='viridis', vmin=vmin_swe, vmax=vmax_swe) #reconstruction
    ax1.set_title('Actual SWE (m)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=10)
    
    # Plot 2: Predicted SWE
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(pred_valid, cmap='viridis', vmin=vmin_swe, vmax=vmax_swe)
    ax2.set_title('Predicted SWE (m)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=10)
    
    # Plot 3: Error map (prediction - actual)
    ax3 = fig.add_subplot(gs[0, 2])
    error_limit = np.nanpercentile(np.abs(error_map), 95)
    im3 = ax3.imshow(error_map.squeeze(), cmap='RdBu_r', vmin=-error_limit, vmax=error_limit)
    ax3.set_title('Error: Pred - Actual (m)\n(Red=Over, Blue=Under)', 
                  fontsize=14, fontweight='bold')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(labelsize=10)
    
    # ========================================
    # ROW 2: Additional context
    # ========================================
    
    # Plot 4: Absolute error
    ax4 = fig.add_subplot(gs[1, 0])
    im4 = ax4.imshow(abs_error_map.squeeze(), cmap='hot_r', vmin=0, vmax=error_limit)
    ax4.set_title('Absolute Error (m)', fontsize=14, fontweight='bold')
    ax4.axis('off')
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.ax.tick_params(labelsize=10)
    
    # Plot 5: Coverage map (how many patches covered each pixel)
    ax5 = fig.add_subplot(gs[1, 1])
    coverage_map = (weight_map > 0).astype(float)
    coverage_map[~Y_mask] = np.nan
    im5 = ax5.imshow(coverage_map, cmap='Greys', vmin=0, vmax=1)
    ax5.set_title('Prediction Coverage\n(1=Predicted, 0=No coverage)', 
                  fontsize=14, fontweight='bold')
    ax5.axis('off')
    cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    cbar5.ax.tick_params(labelsize=10)
    
    # Plot 6: Patch weight map (shows overlap intensity)
    ax6 = fig.add_subplot(gs[1, 2])
    im6 = ax6.imshow(weight_map, cmap='plasma')
    
    # Overlay patch boundaries
    for row, col in valid_patch_locations[::10]:  # Show every 10th patch to avoid clutter
        rect = Rectangle((col, row), patch_size, patch_size,
                        linewidth=0.5, edgecolor='white', facecolor='none', alpha=0.3)
        ax6.add_patch(rect)
    
    ax6.set_title(f'Weight Map & Patch Grid\n({len(valid_patch_locations)} patches)', 
                  fontsize=14, fontweight='bold')
    ax6.axis('off')
    cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    cbar6.ax.tick_params(labelsize=10)
    
    # ========================================
    # ROW 3: Statistics and scatter
    # ========================================
    
    # Plot 7: Scatter plot
    ax7 = fig.add_subplot(gs[2, 0])

    # Extract valid pixels as 1D arrays (no NaNs)
    actual_valid_1d = actual_swe.squeeze()[valid_mask]
    pred_valid_1d = reconstruction[valid_mask]

    # Subsample for plotting if too many points
    n_points = len(actual_valid_1d)
    if n_points > 50000:
        indices = np.random.choice(n_points, 50000, replace=False)
        plot_actual = actual_valid_1d[indices]
        plot_pred = pred_valid_1d[indices]
    else:
        plot_actual = actual_valid_1d
        plot_pred = pred_valid_1d

    # Additional safety check: remove any remaining NaNs or infs
    valid_plot_mask = np.isfinite(plot_actual) & np.isfinite(plot_pred)
    plot_actual = plot_actual[valid_plot_mask]
    plot_pred = plot_pred[valid_plot_mask]

    # Now hexbin will work without warnings
    h = ax7.hexbin(plot_actual, plot_pred, gridsize=50, cmap='viridis', 
                mincnt=1, linewidths=0.2, edgecolors='face')

    # 1:1 line
    max_val = max(plot_actual.max(), plot_pred.max())
    ax7.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1 Line')

    ax7.set_xlabel('Actual SWE (m)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Predicted SWE (m)', fontsize=12, fontweight='bold')
    ax7.set_title('Pixel-Level Comparison', fontsize=14, fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(alpha=0.3)
    ax7.set_aspect('equal', adjustable='box')


    # Plot 8: Error histogram
    ax8 = fig.add_subplot(gs[2, 1])
    
    # errors_valid = error_map.copy() 
    # errors_valid[~valid_mask] = np.nan

    errors_valid = error_map.squeeze()[valid_mask]

    # Use nanmedian for the median calculation (in case any NaNs slipped through)
    ax8.hist(errors_valid, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax8.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax8.axvline(x=np.median(errors_valid), color='orange', linestyle='--', 
            linewidth=2, label=f'Median = {np.median(errors_valid):.3f} m')
    ax8.set_xlabel('Error (m)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Pixel Count', fontsize=12, fontweight='bold')
    ax8.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(alpha=0.3, axis='y')
    
    # Plot 9: Statistics table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Compute additional statistics
    stats_data = [
        ['Metric', 'Value'],
        ['─'*20, '─'*20],
        ['Total Pixels', f'{n_points:,}'],
        ['Valid Area', f'{n_points * 2500 / 1e6:.2f} km²'],
        ['', ''],
        ['MAE', f'{mae:.4f} m'],
        ['RMSE', f'{rmse:.4f} m'],
        ['R²', f'{r2:.4f}'],
        ['Bias', f'{bias:+.4f} m'],
        ['', ''],
        ['Mean Actual', f'{actual_valid.mean():.4f} m'],
        ['Mean Predicted', f'{pred_valid.mean():.4f} m'],
        ['', ''],
        ['Max Error', f'{np.abs(errors_valid).max():.4f} m'],
        ['Median Error', f'{np.median(np.abs(errors_valid)):.4f} m'],
        ['', ''],
        ['Patches Used', f'{len(valid_patch_locations)}'],
        ['Patch Size', f'{patch_size}x{patch_size}'],
        ['Stride', f'{stride}'],
    ]
    
    table = ax9.table(cellText=stats_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')
    
    # Color code metric rows
    for i in range(5, 9):  # Metric rows
        for j in range(2):
            table[(i, j)].set_facecolor('#E6F0FF')
    
    ax9.set_title('Reconstruction Statistics', fontsize=14, fontweight='bold', pad=20)
    
    # ========================================
    # Overall title
    # ========================================
    basin = flight_to_basin.get(f"{sample_flight_id}.tif", "Unknown")
    fig.suptitle(f'Flight Reconstruction: {sample_flight_id}\n'
                f'Basin: {basin} | Resolution: {H}x{W} pixels',
                fontsize=18, fontweight='bold', y=0.995)
    
    # ========================================
    # Save
    # ========================================
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved reconstruction to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Return results
    return {
        'reconstruction': reconstruction,
        'actual': actual_swe,
        'error': error_map,
        'mask': Y_mask,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'bias': bias
        },
        'n_patches': len(valid_patch_locations)
    }

# ============================================================
# Update main() to call this function
# ============================================================
def run(folder):
    # CONFIGURATION - MODIFY THESE PATHS
    checkpoint_dir = f"/discover/nobackup/cmbreen/gap-filling/{folder}"  # MODIFY THIS
    zarr_dir = "/discover/nobackup/cmbreen/gap-filling-data/zarr_chunks"
    save_dir = os.path.join(checkpoint_dir, "ablation_study")
    
    os.makedirs(save_dir, exist_ok=True)  #
    print(f"Results will be saved to: {save_dir}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Patching config (should match training)
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
    print(f"  X channels: {len(norm_stats['X_mean'])}")
    print(f"  Y mean: {y_mean:.4f}, Y std: {y_std:.4f}")
    
    # ========================================
    # STEP 1: Load FULL images (keep them for per-flight analysis)
    # ========================================
    print("\n" + "="*80)
    print("LOADING FULL TEST IMAGES")
    print("="*80)
    
    test_x_full, test_y_full, test_y_mask_full, filenames_test = load_full_zarr_files(
        zarr_dir, split_basin_dict, flight_to_basin, skip_tb_channels=True
    )
    
    # ========================================
    # STEP 2: Create patches for ablation study
    # ========================================
    print("\n" + "="*80)
    print("CREATING PATCHES FOR ABLATION ANALYSIS")
    print("="*80)
    
    test_x_patches, test_y_patches, test_y_mask_patches = convert_to_patches(
        test_x_full, test_y_full, test_y_mask_full, filenames_test,
        patch_size=patch_size, stride=stride, min_valid_fraction=min_valid_fraction
    )
    
    # Apply log transform to patches
    test_y_patches_log = [np.log1p(y) for y in test_y_patches]
    
    # Normalize patches
    test_x_patches_norm = normalize_dataset_per_channel(
        test_x_patches, norm_mean, norm_std, 
        skip_channels=[7, 8, 9, 10, 11, 12, 13]
    )
    test_y_patches_norm = [(y - y_mean) / (y_std + 1e-7) for y in test_y_patches_log]
    
    print(f"\nPatched test data ready:")
    print(f"  {len(test_x_patches_norm)} patches")
    print(f"  Shape: {test_x_patches_norm[0].shape}")
    
    # ========================================
    # STEP 3: Also normalize FULL images for per-flight analysis
    # ========================================
    print("\nPreparing full images for per-flight analysis...")
    
    # Apply log transform to full images
    test_y_full_log = [np.log1p(y) for y in test_y_full]
    
    # Normalize full images
    test_x_full_norm = normalize_dataset_per_channel(
        test_x_full, norm_mean, norm_std,
        skip_channels=[7, 8, 9, 10, 11, 12, 13]
    )
    test_y_full_norm = [(y - y_mean) / (y_std + 1e-7) for y in test_y_full_log]
    
    print(f"Full images ready: {len(test_x_full_norm)} flights")
    
    # ========================================
    # STEP 4: Load model
    # ========================================
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    
    model_path = os.path.join(checkpoint_dir, 'best_model_cnn.pth')
    input_channels = test_x_patches_norm[0].shape[1]
    model = Model(input_channels=input_channels).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {model_path}")
    
    # ========================================
    # STEP 5: Per-flight SWE analysis (on FULL images)
    # ========================================
    print("\n" + "="*80)
    print("RUNNING PER-FLIGHT SWE ANALYSIS")
    print("="*80)
    
    flight_results, basin_summary = analyze_per_flight_swe(
        model, test_x_full_norm, test_y_full_norm, test_y_mask_full,
        filenames_test, y_mean, y_std, device, save_dir,
        pixel_area_m2=50*50  # Adjust if your pixels are different size
    )
    
    # ========================================
    # STEP 6: Standard ablation study (on patches)
    # ========================================
    print("\n" + "="*80)
    print("RUNNING FEATURE ABLATION STUDY")
    print("="*80)
    
    # ablation_results, permutation_results, baseline_metrics = run_ablation_study(
    #     checkpoint_dir, test_x_patches_norm, test_y_patches_norm, test_y_mask_patches,
    #     y_mean, y_std, device, save_dir
    # )
    
    # # Create visualizations
    # plot_feature_importance(ablation_results, permutation_results, 
    #                        baseline_metrics, save_dir)
    
    # ========================================
    # STEP 7: Forest cover analysis (on patches)
    # ========================================
    print("\n" + "="*80)
    print("RUNNING FOREST COVER ANALYSIS")
    print("="*80)
    
    forest_cover_results = analyze_importance_by_forest_cover(
        model, test_x_patches_norm, test_y_patches_norm, test_y_mask_patches,
        y_mean, y_std, device, save_dir, norm_mean, norm_std, ## add in this forest cover denorm
        n_bins=20  # Adjust number of bins as needed
    )

    print("\n" + "="*80)
    print("CREATING SAMPLE PATCH VISUALIZATION")
    print("="*80)
    
    # Pick a sample flight (you can change this)
    sample_flight = 'ASO_Dolores_2023Apr06_swe_50m'  # Modify as needed
    
    # Or pick the first test flight automatically
    if len(filenames_test) > 0:
        sample_flight = filenames_test[0].replace('.tif', '')
    
    plot_sample_basin_patches(
        zarr_dir=zarr_dir,
        sample_flight_id=sample_flight,
        patch_size=patch_size,
        stride=stride,
        min_valid_fraction=min_valid_fraction,
        save_path=os.path.join(save_dir, f'sample_patches_{sample_flight}.png')
    )
    
    # ========================================
    # STEP 7.5: Reconstruct and visualize sample flights
    # ========================================
    print("\n" + "="*80)
    print("RECONSTRUCTING SAMPLE FLIGHTS FROM PATCHES")
    print("="*80)
    
    # Visualize first 3 test flights (or specify particular ones)
    flights_to_visualize = filenames_test[:3]  # Modify as needed
    
    # Or specify particular flights:
    # flights_to_visualize = [
    #     'ASO_Dolores_2023Apr06_swe_50m.tif',
    #     'ASO_Conejos_2023May05_swe_50m.tif'
    # ]
    
    reconstruction_results = []
    
    for flight_file in flights_to_visualize:
        flight_id = flight_file.replace('.tif', '')
        print(f"\n{'='*80}")
        print(f"Processing: {flight_id}")
        print('='*80)
        
        result = reconstruct_and_plot_flight(
            model=model,
            zarr_dir=zarr_dir,
            sample_flight_id=flight_id,
            norm_mean=norm_mean,
            norm_std=norm_std,
            y_mean=y_mean,
            y_std=y_std,
            patch_size=patch_size,
            stride=stride,
            min_valid_fraction=min_valid_fraction,
            device=device,
            save_path=os.path.join(save_dir, f'reconstruction_{flight_id}.png')
        )
        
        reconstruction_results.append({
            'flight': flight_id,
            'metrics': result['metrics'],
            'n_patches': result['n_patches']
        })
    
    # Print summary
    print("\n" + "="*80)
    print("RECONSTRUCTION SUMMARY")
    print("="*80)
    for res in reconstruction_results:
        print(f"\n{res['flight']}:")
        print(f"  Patches: {res['n_patches']}")
        print(f"  MAE:  {res['metrics']['mae']:.4f} m")
        print(f"  RMSE: {res['metrics']['rmse']:.4f} m")
        print(f"  R²:   {res['metrics']['r2']:.4f}")
        print(f"  Bias: {res['metrics']['bias']:+.4f} m")

    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {save_dir}/")
    
    print("\n" + "-"*80)
    print("KEY FINDINGS")
    print("-"*80)
    
    # Ablation study summary
    # sorted_by_ablation = sorted(ablation_results.items(),
    #                            key=lambda x: x[1]['rmse_drop'],
    #                            reverse=True)
    
    # print("\nTop 5 most important features (by RMSE drop):")
    # for i, (feature, results) in enumerate(sorted_by_ablation[:5], 1):
    #     print(f"  {i}. {feature}: RMSE Δ = {results['rmse_drop']:+.4f} m, "
    #           f"R² Drop = {results['r2_drop']:.4f}")
    
    # Per-flight summary
    print("\nPer-flight SWE totals:")
    print(f"  Total flights analyzed: {len(flight_results)}")
    print(f"  Total missing SWE: {flight_results['missing_volume_m3'].sum():+,.0f} m³")
    print(f"  Mean per-flight RMSE: {flight_results['rmse_m'].mean():.3f} m")

    print("\n" + "="*80)
    print("GENERATED FILES:")
    print("  - ablation_results.json")
    print("  - per_flight_swe_analysis.csv")
    print("  - basin_summary.csv")
    print("  - importance_vs_forest_cover.csv")
    print("  - Multiple visualization PNGs")
    print("="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='model folder')
    parser.add_argument('--folder', type=str,
    default='checkpoints_elevPM_NDSI_CC_1e-5_ps256_W_smoothL1loss')
    args = parser.parse_args()
    run(args.folder)

if __name__ == '__main__':
    main()