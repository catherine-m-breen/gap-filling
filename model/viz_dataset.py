#!/usr/bin/env python3
"""
Dataset Visualization Script
Creates histograms showing distributions of all variables across train/val/test splits.

Usage:
    module load anaconda/py3.11.7
    conda activate gapfill2
    python viz_dataset.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import zarr
from tqdm import tqdm
import sys

# Import your dataset
from dataset import create_dataloaders, split_basin_dict

# Feature channel indices (based on your zarr structure)
FEATURE_CHANNELS = {
    'elevation': 0,
    'snowmap_class': 1,
    'tree_canopy': 2,
    'land_cover': 3,
    'viirs_ndsi': 4,
    'pm_37h': 5,  # Passive Microwave 37 GHz H-pol
    'pm_37v': 6,  # Passive Microwave 37 GHz V-pol
    'pm_19h': 7,  # Passive Microwave 19 GHz H-pol
    'pm_19v': 8,  # Passive Microwave 19 GHz V-pol
}

# Categorical variables
CATEGORICAL_VARS = ['snowmap_class', 'land_cover']


def collect_data_statistics(zarr_dir: str, num_samples: int = 1000):
    """
    Collect data from all splits for visualization.
    
    Args:
        zarr_dir: Path to zarr directory
        num_samples: Number of random samples to collect per split (to avoid memory issues)
    
    Returns:
        Dictionary with data for each variable and split
    """
    print("=" * 80)
    print("COLLECTING DATA STATISTICS")
    print("=" * 80)
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        zarr_dir=zarr_dir,
        batch_size=16,
        patch_size=256,
        stride=256,  # Non-overlapping for faster collection
        num_workers=4,
        normalize=False,  # Don't normalize - we want raw values
        random_crop_train=False
    )
    
    # Initialize storage
    data_dict = {split: {} for split in ['train', 'val', 'test']}
    
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*80}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*80}")
        
        dataloader = dataloaders[split]
        
        # Initialize lists for each variable
        for var_name in FEATURE_CHANNELS.keys():
            data_dict[split][var_name] = []
        data_dict[split]['swe'] = []
        
        # Collect samples
        samples_collected = 0
        max_samples = num_samples
        
        print(f"Collecting up to {max_samples} samples...")
        
        for batch_idx, (batch_X, batch_Y, batch_metadata) in enumerate(tqdm(dataloader)):
            # batch_X: (batch, 11, 256, 256)
            # batch_Y: (batch, 1, 256, 256)
            
            # Extract each feature channel
            for var_name, channel_idx in FEATURE_CHANNELS.items():
                # Get channel data and flatten
                channel_data = batch_X[:, channel_idx, :, :].flatten().numpy()
                # Remove zeros and NaNs (likely masked/invalid values)
                channel_data = channel_data[(channel_data != 0) & (~np.isnan(channel_data))]
                data_dict[split][var_name].extend(channel_data)
            
            # Extract SWE (target)
            swe_data = batch_Y[:, 0, :, :].flatten().numpy()
            swe_data = swe_data[(swe_data != 0) & (~np.isnan(swe_data))]
            data_dict[split]['swe'].extend(swe_data)
            
            samples_collected += batch_X.size(0)
            
            if samples_collected >= max_samples:
                break
        
        # Convert to numpy arrays
        for var_name in list(data_dict[split].keys()):
            arr = np.array(data_dict[split][var_name])
            data_dict[split][var_name] = arr
            print(f"  {var_name}: {len(arr):,} valid pixels, "
                  f"range=[{arr.min():.2f}, {arr.max():.2f}], "
                  f"mean={arr.mean():.2f}")
    
    return data_dict


def plot_continuous_variable(data_dict, var_name, title, bins=50, figsize=(14, 10)):
    """
    Create histogram for continuous variables with train/val/test in separate rows.
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    splits = ['train', 'val', 'test']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Get global min/max for consistent x-axis
    all_data = np.concatenate([data_dict[split][var_name] for split in splits])
    vmin, vmax = np.percentile(all_data, [1, 99])  # Use percentiles to avoid outliers
    
    for idx, split in enumerate(splits):
        data = data_dict[split][var_name]
        
        # Plot histogram
        axes[idx].hist(data, bins=bins, color=colors[idx], alpha=0.7, 
                       edgecolor='black', range=(vmin, vmax))
        
        # Add statistics text
        stats_text = (f"n={len(data):,}\n"
                     f"mean={data.mean():.2f}\n"
                     f"std={data.std():.2f}\n"
                     f"median={np.median(data):.2f}")
        
        axes[idx].text(0.98, 0.97, stats_text, 
                      transform=axes[idx].transAxes,
                      verticalalignment='top',
                      horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                      fontsize=10)
        
        # Formatting
        axes[idx].set_ylabel(f'{split.upper()}\nFrequency', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Set title and x-label
    axes[0].set_title(title, fontsize=14, fontweight='bold', pad=20)
    axes[2].set_xlabel('Value', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_categorical_variable(data_dict, var_name, title, figsize=(14, 10)):
    """
    Create bar chart for categorical variables with train/val/test in separate rows.
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    splits = ['train', 'val', 'test']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Get all unique classes
    all_data = np.concatenate([data_dict[split][var_name] for split in splits])
    unique_classes = np.unique(all_data)
    unique_classes = unique_classes[~np.isnan(unique_classes)]  # Remove NaN
    
    for idx, split in enumerate(splits):
        data = data_dict[split][var_name]
        
        # Count occurrences of each class
        class_counts = []
        for cls in unique_classes:
            count = np.sum(data == cls)
            class_counts.append(count)
        
        # Plot bar chart
        bars = axes[idx].bar(unique_classes, class_counts, color=colors[idx], 
                            alpha=0.7, edgecolor='black')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                              f'{int(height):,}',
                              ha='center', va='bottom', fontsize=8)
        
        # Add statistics text
        stats_text = f"n={len(data):,}\n{len(unique_classes)} classes"
        axes[idx].text(0.98, 0.97, stats_text, 
                      transform=axes[idx].transAxes,
                      verticalalignment='top',
                      horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                      fontsize=10)
        
        # Formatting
        axes[idx].set_ylabel(f'{split.upper()}\nCount', fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Set title and x-label
    axes[0].set_title(title, fontsize=14, fontweight='bold', pad=20)
    axes[2].set_xlabel('Class', fontsize=12)
    
    # Ensure integer ticks for classes
    axes[2].set_xticks(unique_classes)
    
    plt.tight_layout()
    return fig


def create_all_plots(data_dict, output_dir='./dataset_distributions'):
    """
    Create all 10 distribution plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "=" * 80)
    print("CREATING DISTRIBUTION PLOTS")
    print("=" * 80)
    
    # Plot configurations
    plots = [
        # Continuous variables
        ('elevation', 'Elevation Distribution (m)', False),
        ('tree_canopy', 'Tree Canopy Cover Distribution (%)', False),
        ('viirs_ndsi', 'VIIRS NDSI Distribution', False),
        ('swe', 'SWE Distribution (mm)', False),
        ('pm_37h', 'Passive Microwave 37 GHz H-pol Distribution', False),
        ('pm_37v', 'Passive Microwave 37 GHz V-pol Distribution', False),
        ('pm_19h', 'Passive Microwave 19 GHz H-pol Distribution', False),
        ('pm_19v', 'Passive Microwave 19 GHz V-pol Distribution', False),
        
        # Categorical variables
        ('snowmap_class', 'Snowmap Class Distribution', True),
        ('land_cover', 'Land Cover Distribution', True),
    ]
    
    for var_name, title, is_categorical in plots:
        print(f"\nCreating plot: {title}")
        
        if is_categorical:
            fig = plot_categorical_variable(data_dict, var_name, title)
        else:
            fig = plot_continuous_variable(data_dict, var_name, title)
        
        # Save figure
        filename = f"{var_name}_distribution.png"
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath}")
        
        plt.close(fig)
    
    print("\n" + "=" * 80)
    print(f"All plots saved to: {output_dir}")
    print("=" * 80)


def main():
    """Main execution function."""
    
    # Configuration
    zarr_dir = "/discover/nobackup/cmbreen/gap-filling-data/zarr_chunks"
    output_dir = "./dataset_distributions"
    num_samples = 2000  # Number of patches to sample per split
    
    print("\n" + "=" * 80)
    print("ASO DATASET DISTRIBUTION VISUALIZATION")
    print("=" * 80)
    print(f"Zarr directory: {zarr_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Samples per split: {num_samples}")
    print(f"Train basins: {split_basin_dict['train']}")
    print(f"Val basins: {split_basin_dict['val']}")
    print(f"Test basins: {split_basin_dict['test']}")
    print("=" * 80)
    
    # Step 1: Collect data statistics
    data_dict = collect_data_statistics(zarr_dir, num_samples=num_samples)
    
    # Step 2: Create plots
    create_all_plots(data_dict, output_dir=output_dir)
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()