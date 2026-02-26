# module load anaconda/py3.11.7
# conda activate gapfill2

'''
same thing as new_try2 but works with full zarr data
'''

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import os
from pathlib import Path
import zarr
from dictionaries import split_basin_dict, flight_to_basin
import IPython
print('Starting training script...')

# ============================================================
# Random Crop Function
# ============================================================

def random_crop(data, crop_size):
    """Random crop for data augmentation."""
    _, _, width, height = data.shape
    
    pad_width = max(crop_size[0] - width, 0)
    pad_height = max(crop_size[1] - height, 0)

    if pad_width > 0 or pad_height > 0:
        data = np.pad(data, ((0, 0), (0, 0), (0, pad_width), (0, pad_height)), mode='constant')

    _, _, width, height = data.shape
    dw = np.random.randint(0, width - crop_size[0] + 1)
    dh = np.random.randint(0, height - crop_size[1] + 1)
    
    return data[:, :, dw:dw+crop_size[0], dh:dh+crop_size[1]]


def random_flip(data, labels, masks):
    """Random horizontal and vertical flips."""
    flip_w = np.random.choice([True, False])
    flip_h = np.random.choice([True, False])
    
    if flip_w:
        data = np.flip(data, axis=2).copy()
        labels = np.flip(labels, axis=1).copy()  # Match label dims
        masks = np.flip(masks, axis=1).copy()  # Match label dims
    if flip_h:
        data = np.flip(data, axis=3).copy()
        labels = np.flip(labels, axis=2).copy()
        masks = np.flip(masks, axis=2).copy()
    
    return data, labels, masks


def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to image."""
    _, ch, row, col = image.shape
    channel = 0
    #noisy = np.zeros_like(image)
    noisy = image.copy()  ## WE NEED IT TO COPY THE IMAGE NOT MAKE ZEROS OR ELSE WE LOSE ALL THE DATA!!
    gauss = np.random.normal(mean, sigma, (1, 1, row, col))
    noisy[:, channel:channel+1, :, :] = image[:, channel:channel+1, :, :] + gauss

    # gauss = np.random.normal(mean, sigma, (1, ch, row, col))
    # noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.float32)


# ============================================================
# Dataset Class WITH PATCHING
# ============================================================

class ASODataset(Dataset):
    def __init__(self, data, labels, crop_size=(128, 128), patch_size=128, stride=64, augment=False):
        """
        Dataset that creates patches from full zarr images on-the-fly.
        
        Args:
            data: List of full images (each is (1, C, H, W))
            labels: List of full labels (each is (1, 1, H, W))
            crop_size: Size for random cropping during augmentation
            patch_size: Size of patches to extract from full images
            stride: Stride for patch extraction
            augment: Whether to apply random cropping/flipping
        """
        self.data = data
        self.labels = labels
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        
        # Create patch index: (file_idx, row, col)
        self.patch_index = []
        for file_idx in range(len(data)):
            _, _, H, W = data[file_idx].shape
            for row in range(0, H - patch_size + 1, stride):
                for col in range(0, W - patch_size + 1, stride):
                    self.patch_index.append((file_idx, row, col))
        
        print(f"  Created {len(self.patch_index)} patches from {len(data)} images")

    def __len__(self):
        return len(self.patch_index) 

    def __getitem__(self, idx):
        file_idx, row, col = self.patch_index[idx]
        
        # Extract patch from full image
        data_full = self.data[file_idx]
        label_full = self.labels[file_idx]
        
        data_patch = data_full[:, :, row:row+self.patch_size, col:col+self.patch_size]
        label_patch = label_full[:, :, row:row+self.patch_size, col:col+self.patch_size]
        
        # Remove batch dimension (1, C, H, W) -> (C, H, W)
        data_patch = data_patch[0]
        label_patch = label_patch[0]

        label_mask = ((label_patch > 0) & (~np.isnan(label_patch))).astype(np.float32)
        
        if self.augment:
            # Apply same crop/flip to data, label, AND mask
            # Stack all three along channel dimension
            combined = np.concatenate([data_patch, label_patch, label_mask], axis=0)
            combined = random_crop(combined[None, :, :, :], self.crop_size)[0]
            
            num_data_channels = data_patch.shape[0]
            num_label_channels = label_patch.shape[0]
            
            # Split back out
            data_patch = combined[:num_data_channels, :, :]
            label_patch = combined[num_data_channels:num_data_channels+num_label_channels, :, :]
            label_mask = combined[num_data_channels+num_label_channels:, :, :]
            
        return data_patch, label_patch #, label_mask

# ============================================================
# Model (2D CNN like original script)
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
        # Input: (batch, channels, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv100(x)
        
        # Output: (batch, 1, H, W)
        x = x.squeeze(1)  # Remove channel dimension → (batch, H, W)
        
        return x


# ============================================================
# Training and Validation Functions
# ============================================================

def save_first_batch_viz(features, labels, predictions, masks, epoch, save_dir):
    """
    Save visualizations of first batch item: input channels, target, prediction, mask
    
    Args:
        features: (batch, channels, H, W) - input patches
        labels: (batch, H, W) - target SWE
        predictions: (batch, H, W) - model predictions
        masks: (batch, H, W) - boolean mask
        epoch: current epoch number
        save_dir: directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    #  save_first_batch_viz(features, labels, output, masks, epoch)
    # Move to CPU and get first item in batch
    features_np = features[0].detach().cpu().numpy()  # (channels, H, W)
    labels_np = labels[0].detach().cpu().numpy()  # (H, W)
    preds_np = predictions[0].detach().cpu().numpy()  # (H, W)
    masks_np = masks[0].cpu().numpy()  # (H, W)
    
    # Get number of input channels
    n_channels = features_np.shape[0]
    
    # Create figure: input channels + target + prediction + mask
    n_plots = n_channels + 3
    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 4))
    
    # Plot each input channel
    for ch in range(n_channels):
        im = axes[ch].imshow(features_np[ch], cmap='viridis')
        axes[ch].set_title(f'Input Channel {ch}')
        axes[ch].axis('off')
        plt.colorbar(im, ax=axes[ch], fraction=0.046)
    
    # Plot target (NOT masked)
    target = labels_np.copy()
    #  target_masked[~masks_np] = np.nan
    #target_masked[~masks_np.squeeze()] = np.nan
    im = axes[n_channels].imshow(target, cmap='Blues', vmin=0)
    axes[n_channels].set_title('Target SWE')
    axes[n_channels].axis('off')
    plt.colorbar(im, ax=axes[n_channels], fraction=0.046)
    
    # Plot prediction (NOT masked)
    pred = preds_np.copy()
    #pred_masked[~masks_np] = np.nan
    im = axes[n_channels+1].imshow(pred, cmap='Blues', vmin=0)
    axes[n_channels+1].set_title('Predicted SWE')
    axes[n_channels+1].axis('off')
    plt.colorbar(im, ax=axes[n_channels+1], fraction=0.046)
    
    # Plot mask
    im = axes[n_channels+2].imshow(masks_np.squeeze(), cmap='RdYlGn', vmin=0, vmax=1)
    axes[n_channels+2].set_title(f'Mask ({masks_np.sum()}/{masks_np.size} valid)')
    axes[n_channels+2].axis('off')
    plt.colorbar(im, ax=axes[n_channels+2], fraction=0.046)
    
    plt.suptitle(f'Epoch {epoch} - First Batch Sample', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_first_batch.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved first batch visualization to {save_path}")

def train_model(model, dataloader, optimizer, criterion, device, epoch, batch_size=8, save_dir= 'checkpoints'):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    first_batch = True 
    for i, (features, labels, masks) in enumerate(dataloader):
        ## labels aren't normalized here??
        features = features.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.bool) ## keep as boolean
        ## this is redundant
        #labels_masks = labels_masks.to(device, dtype=torch.float32)

        # Skip tiny patches
        if features.shape[2] < 2 or features.shape[3] < 2:
            continue

        output = model(features)
        
        # FIX: Ensure labels have same shape as output
        # labels comes as (batch, 1, H, W), squeeze to (batch, H, W)
        if len(labels.shape) == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)  # (batch, 1, H, W) -> (batch, H, W)
        
        # Mask: only compute loss on valid pixels (non-zero)
        #mask = (labels != 0) & (~torch.isnan(labels)) & (labels >= 0) ## no negative values either 
        
        # Flatten for loss computation
        ## these are all the same shape##
        labels_flat = labels.reshape(-1)
        output_flat = output.reshape(-1)
        mask_flat = masks.reshape(-1) ## use the mask that was create at the beginning
        
        labels_masked = labels_flat[mask_flat]
        output_masked = output_flat[mask_flat]

        # Save first batch visualization
        #IPython.embed()
        "Saving Image...."
        if (epoch % 5) & first_batch:
            ## labels aren't normalized weirdly??
            save_first_batch_viz(features, labels, output, masks, epoch, save_dir)
            first_batch = False
        # Skip if no valid pixels
        # if len(labels_masked) == 0:
        #     continue

        # L1 loss + L1 regularizationcheckp
        #IPython.embed()
        l1_lambda = 0.000001
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        #IPython.embed()
        loss = criterion(labels_masked, output_masked) + l1_lambda * l1_norm

        loss.backward()
        total_loss += loss.item()

        ## all predictions or just masked ones ? This is just masked ones
        ## we are just passing the valid preds and valid labels to each new step
        all_preds.extend(output_masked.detach().cpu().numpy())
        all_labels.extend(labels_masked.cpu().numpy())

        # Gradient accumulation
        if (i + 1) % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()

    # Final step if not divisible by batch_size
    if (i + 1) % batch_size != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / max(len(dataloader), 1)
    print(f"Epoch {epoch} - Train Loss: {avg_loss:.6f}, Valid pixels: {len(all_labels):,}")
    
    #### this is what will get passed through to the next round ##
    return avg_loss, (all_labels, all_preds)


def validate_model(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (features, labels, masks) in enumerate(dataloader):
            features = features.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.bool) ## keep as boolean

            if features.shape[2] < 2 or features.shape[3] < 2:
                continue

            output = model(features)
            
            # FIX: Ensure labels have same shape as output
            if len(labels.shape) == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)  # (batch, 1, H, W) -> (batch, H, W)
            
            # Mask: only compute loss on valid pixels
            #mask = (labels != 0) & (~torch.isnan(labels)) & (labels >= 0) ## no negative values either 
            
            # Flatten for loss computation
            labels_flat = labels.reshape(-1)
            output_flat = output.reshape(-1)
            mask_flat = masks.reshape(-1)
            
            labels_masked = labels_flat[mask_flat]
            output_masked = output_flat[mask_flat]

            if len(labels_masked) == 0:
                continue

            loss = criterion(labels_masked, output_masked)
            total_loss += loss.item()

            all_preds.extend(output_masked.detach().cpu().numpy())
            all_labels.extend(labels_masked.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    print(f"Epoch {epoch} - Val Loss: {avg_loss:.6f}, Valid pixels: {len(all_labels):,}")
    
    return avg_loss, (all_labels, all_preds)

# ============================================================
# Data Loading Functions - LOAD FULL ZARR FILES
# ============================================================
def load_full_zarr_files(zarr_dir, split_dict, flight_to_basin_dict, skip_tb_channels=True):
    """
    Load FULL zarr files (not patches) and organize by train/val/test split.
    
    Args:
        skip_tb_channels: If True, remove brightness temperature channels (4-7)
    """
    zarr_dir = Path(zarr_dir)
    zarr_files = sorted(zarr_dir.glob("*.zarr"))
    
    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []
    
    train_y_mask = []
    val_y_mask = []
    test_y_mask = []
    
    # ADD THIS: Track extreme values
    extreme_value_files = []
    
    print(f"Loading {len(zarr_files)} FULL zarr files...")
    
    for zarr_path in zarr_files:
        # Get flight_id from filename (remove .zarr extension)
        flight_id = zarr_path.stem
        tif_name = flight_id + '.tif'
        
        if tif_name not in flight_to_basin_dict:
            print(f"  Warning: {tif_name} not in flight_to_basin dict, skipping")
            continue
        
        basin = flight_to_basin_dict[tif_name]
        
        # Determine split
        split = None
        for split_name, basins in split_dict.items():
            if basin in basins:
                split = split_name
                break
        
        if split is None:
            print(f"  Warning: Basin {basin} not in split_dict, skipping")
            continue
        
        # Load FULL zarr file
        z = zarr.open(str(zarr_path), mode='r')
        X = np.array(z['X'], dtype=np.float32)  # (C, H, W) - full image
        Y = np.array(z['Y'], dtype=np.float32)  # (1, H, W) - full label
        
        # ========================================
        # SKIP BRIGHTNESS TEMPERATURE CHANNELS
        # ========================================
        if skip_tb_channels:
            channels_to_keep = [3]
            X = X[channels_to_keep, :, :]

            # ========================================
            # CREATE NaN MASK CHANNEL
            # ========================================
            # Create binary mask: 1 = valid data, 0 = NaN
            nan_mask = (~np.isnan(X)).astype(np.float32)
            
            # Fill NaN in original data with 0 (or mean)
            X_filled = np.nan_to_num(X, nan=0.0)
            
            # Stack data and mask as separate channels
            # Result: (2, H, W) - channel 0 = data, channel 1 = mask
            X = np.concatenate([X_filled, nan_mask], axis=0)

            if len(train_x) == 0:  # Only print once
                print(f"  Removed TB channels, X shape: {X.shape[0]} channels")
        
        # Handle invalid values
        X[X == -9999] = 0.0
        #X[X == 9999] = 0.0
        #Y[Y == -9999] = 0.0
        #Y[Y == 9999] = 0.0
        Y[Y < 0] = np.nan  # No negative SWE
        Y_mask = ~np.isnan(Y)  # Boolean mask: True where valid, False where NaN ## pass this through so we can only look where we have data! 
        #Y[Y > 10] = np.nan # not values greater than 10? 

        # ========================================
        # CHECK FOR EXTREME Y VALUES (BEFORE CAPPING)
        # ========================================
        extreme_mask = Y > 10.0
        num_extreme = extreme_mask.sum()
        
        if num_extreme > 0:
            max_value = Y[extreme_mask].max()
            mean_extreme = Y[extreme_mask].mean()
            extreme_value_files.append({
                'tif': tif_name,
                'basin': basin,
                'split': split,
                'count': num_extreme,
                'max': max_value,
                'mean_extreme': mean_extreme,
                'total_pixels': Y.size,
                'percent': 100 * num_extreme / Y.size
            })
        
        # Now cap the values
        Y[Y > 10.0] = 0.0  # No SWE over 10m
        
        # Add batch dimension for consistency: (1, C, H, W)
        X = X[None, :, :, :]
        Y = Y[None, :, :, :]
        Y_mask = Y_mask[None, :, :, :]
        
        if len(train_x) == 0:  # Only print once
            print(f"  Loaded {flight_id} ({basin}, {split}): X={X.shape}, Y={Y.shape}")
        
        # Assign to split
        if split == 'train':
            train_x.append(X)
            train_y.append(Y)
            train_y_mask.append(Y_mask)
        elif split == 'val':
            val_x.append(X)
            val_y.append(Y)
            val_y_mask.append(Y_mask)
        elif split == 'test':
            test_x.append(X)
            test_y.append(Y)
            test_y_mask.append(Y_mask)
    
    print(f"\nLoaded: {len(train_x)} train, {len(val_x)} val, {len(test_x)} test FULL images")
    
    # ========================================
    # REPORT EXTREME VALUES
    # ========================================
    if len(extreme_value_files) > 0:
        print("\n" + "="*80)
        print(f"FOUND {len(extreme_value_files)} FILES WITH Y > 10m")
        print("="*80)
        
        # Sort by max value descending
        extreme_value_files.sort(key=lambda x: x['max'], reverse=True)
        
        print(f"\n{'TIF Name':<50} {'Basin':<15} {'Split':<8} {'Count':>8} {'Max (m)':>10} {'Mean (m)':>10} {'% of pixels':>12}")
        print("-" * 120)
        
        for info in extreme_value_files:
            print(f"{info['tif']:<50} {info['basin']:<15} {info['split']:<8} "
                  f"{info['count']:>8,} {info['max']:>10.2f} {info['mean_extreme']:>10.2f} "
                  f"{info['percent']:>11.2f}%")
        
        total_extreme = sum(f['count'] for f in extreme_value_files)
        total_pixels = sum(f['total_pixels'] for f in extreme_value_files)
        
        print("-" * 120)
        print(f"{'TOTAL':<50} {'':<15} {'':<8} {total_extreme:>8,} "
              f"{'':>10} {'':>10} {100*total_extreme/total_pixels:>11.2f}%")
        print("\n" + "="*80 + "\n")
    else:
        print("\n✓ No files with Y > 10m found\n")
    
    return train_x, train_y, val_x, val_y, test_x, test_y, train_y_mask, val_y_mask, test_y_mask ## also pass all the masks!! this is going to be all the stuff where we don't have data and we don't care about

# ============================================================
# Patch Conversion Function
# ============================================================

def convert_to_patches(train_x, train_y, val_x, val_y, test_x, test_y, train_y_mask, val_y_mask, test_y_mask,
                      patch_size=128, stride=64, min_valid_fraction=0.3):
    """
    Convert full images to patches for all splits.
    
    Args:
        train_x, train_y, val_x, val_y, test_x, test_y: Lists of full images
        patch_size: Size of patches to extract
        stride: Stride for patch extraction
        min_valid_fraction: Skip patches with <% valid (non-zero) pixels in target
    
    Returns:
        Patched versions of all inputs as lists
    """
    print("\n" + "="*60)
    print("CONVERTING TO PATCHES")
    print("="*60)
    
    def extract_patches_from_list(data_list, label_list, mask_list, split_name):
        """Extract patches from a list of images."""
        patched_data = []
        patched_labels = []
        patched_masks = []
        total_patches = 0
        skipped_patches = 0
        
        print(f"\n{split_name} split:")
        for img_idx, (data, label, mask) in enumerate(zip(data_list, label_list, mask_list)):
            _, C, H, W = data.shape
            _, _, H_y, W_y = label.shape
            
            img_patches = 0
            
            for row in range(0, H - patch_size + 1, stride):
                for col in range(0, W - patch_size + 1, stride):
                    # Extract patch
                    data_patch = data[:, :, row:row+patch_size, col:col+patch_size]
                    label_patch = label[:, :, row:row+patch_size, col:col+patch_size]
                    mask_patch = mask[:, :, row:row+patch_size, col:col+patch_size]
                    # Quality filter: skip patches with too many invalid pixels
                    valid_pixels = (label_patch != 0) & (~np.isnan(label_patch))
                    valid_fraction = valid_pixels.sum() / label_patch.size
                    
                    if valid_fraction < min_valid_fraction:
                        skipped_patches += 1
                        continue
                    
                    patched_data.append(data_patch)
                    patched_labels.append(label_patch)
                    patched_masks.append(mask_patch)
                    img_patches += 1
                    total_patches += 1
            
            if (img_idx + 1) % 10 == 0 or img_idx == len(data_list) - 1:
                print(f"  Processed {img_idx + 1}/{len(data_list)} images, "
                      f"{total_patches} patches created, {skipped_patches} skipped")
        
        print(f"  Total {split_name}: {total_patches} patches from {len(data_list)} images and {len(mask_list)} masks")
        return patched_data, patched_labels, patched_masks
    
    # Convert each split
    train_x_patched, train_y_patched, train_y_mask_patched = extract_patches_from_list(train_x, train_y, train_y_mask, "TRAIN")
    val_x_patched, val_y_patched, val_y_mask_patched = extract_patches_from_list(val_x, val_y, val_y_mask, "VAL")
    test_x_patched, test_y_patched, test_y_mask_patched = extract_patches_from_list(test_x, test_y, test_y_mask, "TEST")
    
    print(f"\n{'='*60}")
    print(f"PATCHING COMPLETE")
    print(f"  Train: {len(train_x_patched)} patches")
    print(f"  Val:   {len(val_x_patched)} patches")
    print(f"  Test:  {len(test_x_patched)} patches")
    print(f"{'='*60}\n")
    
    return train_x_patched, train_y_patched, val_x_patched, val_y_patched, test_x_patched, test_y_patched, train_y_mask_patched, val_y_mask_patched, test_y_mask_patched

# def normalize_dataset_per_channel(train_data, val_data):
#     """
#     Normalize train and val together using combined statistics.
#     Works with full images and patches
    
#     Returns normalized train and val lists.
#     """
#     print("Computing normalization statistics from train+val combined...")
    
#     # Combine train and val for computing stats
#     combined_data = train_data + val_data
    
#     # Stack all images
#     all_data = np.concatenate(combined_data, axis=0)  # (N, C, H, W)
    
#     # Compute mean and std per channel
#     # mean = np.mean(all_data, axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
#     # std = np.std(all_data, axis=(0, 2, 3), keepdims=True)    # (1, C, 1, 1)
#     mean = np.nanmean(all_data, axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
#     std = np.nanstd(all_data, axis=(0, 2, 3), keepdims=True)    # (1, C, 1, 1)
    
    
#     print(f"Channel means: {mean[0, :, 0, 0]}")
#     print(f"Channel stds: {std[0, :, 0, 0]}")
    
#     # Normalize train
#     normalized_train = []
#     for data in train_data:
#         normalized = (data - mean) / (std + 1e-7)
#         normalized_train.append(normalized)
    
#     # Normalize val
#     normalized_val = []
#     for data in val_data:
#         normalized = (data - mean) / (std + 1e-7)
#         normalized_val.append(normalized)
    
#     return normalized_train, normalized_val, mean, std

def normalize_dataset_per_channel(train_data, val_data, skip_channels=None):
    """
    Normalize train and val together using combined statistics.
    Works with full images and patches.
    
    Args:
        train_data: List of training data arrays
        val_data: List of validation data arrays
        skip_channels: List of channel indices to skip normalization (e.g., [2] for mask)
    
    Returns:
        normalized train and val lists, mean, std
    """
    print("Computing normalization statistics from train+val combined...")
    
    if skip_channels is None:
        skip_channels = []
    
    # Combine train and val for computing stats
    combined_data = train_data + val_data
    
    # Stack all images
    all_data = np.concatenate(combined_data, axis=0)  # (N, C, H, W)
    
    num_channels = all_data.shape[1]
    
    # Compute mean and std per channel
    mean = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    std = np.ones((1, num_channels, 1, 1), dtype=np.float32)  # Default std=1 for skipped channels
    
    for ch in range(num_channels):
        if ch in skip_channels:
            # Don't normalize this channel (keep mean=0, std=1)
            print(f"  Channel {ch}: SKIPPED (mask or binary channel)")
            continue
        
        ch_data = all_data[:, ch, :, :]
        mean[0, ch, 0, 0] = np.nanmean(ch_data)
        std[0, ch, 0, 0] = np.nanstd(ch_data)
        
        # Handle edge cases
        if np.isnan(mean[0, ch, 0, 0]):
            print(f"  Channel {ch}: WARNING - all NaN, setting mean=0, std=1")
            mean[0, ch, 0, 0] = 0.0
            std[0, ch, 0, 0] = 1.0
        elif std[0, ch, 0, 0] < 1e-7:
            print(f"  Channel {ch}: WARNING - zero variance, setting std=1")
            std[0, ch, 0, 0] = 1.0
    
    print(f"Channel means: {mean[0, :, 0, 0]}")
    print(f"Channel stds: {std[0, :, 0, 0]}")
    
    # Normalize train
    normalized_train = []
    for data in train_data:
        normalized = (data - mean) / (std + 1e-7)
        normalized_train.append(normalized)
    
    # Normalize val
    normalized_val = []
    for data in val_data:
        normalized = (data - mean) / (std + 1e-7)
        normalized_val.append(normalized)
    
    return normalized_train, normalized_val, mean, std


# +++++++++++
# Add this function after validate_model():

def evaluate_test_set(model, test_x, test_y, y_mean, y_std, device, checkpoint_dir):
    """
    Evaluate model on test set and create predicted vs actual plot.

      test_results = evaluate_test_set(
        model, 
        test_x_norm, 
        test_y_norm, 
        y_mean, 
        y_std, 
        device, 
        checkpoint_dir
    )
    
    """
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, (x_patch, y_patch) in enumerate(zip(test_x, test_y)):
            # Convert to torch and add batch dimension
            x_tensor = torch.from_numpy(x_patch).to(device, dtype=torch.float32)
            y_tensor = torch.from_numpy(y_patch).to(device, dtype=torch.float32)
            
            # Model expects (batch, C, H, W), data is (1, C, H, W)
            if x_tensor.shape[0] == 1:
                x_tensor = x_tensor  # Already has batch dim
            
            # Get prediction
            output = model(x_tensor)
            
            # squeeze prediction to match label shape
            if len(y_tensor.shape) == 4 and y_tensor.shape[1] == 1:
                y_tensor = y_tensor.squeeze(1)
            
            # if len(output.shape) == 3 and output.shape[0] == 1:
            #     output = output.squeeze(0)
            
            # create mask for valid pixels
            mask = (y_tensor > 0) & (~torch.isnan(y_tensor))
            
            # Extract valid pixels
            #IPython.embed() 
            valid_preds = output[mask].cpu().numpy()
            valid_labels = y_tensor[mask].cpu().numpy()
            
            all_preds.extend(valid_preds)
            all_labels.extend(valid_labels)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(test_x)} test patches...")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Denormalize to get back to meters
    preds_meters = all_preds * y_std + y_mean
    labels_meters = all_labels * y_std + y_mean
    
    # Compute metrics
    mae = mean_absolute_error(labels_meters, preds_meters)
    rmse = np.sqrt(mean_squared_error(labels_meters, preds_meters))
    r2 = r2_score(labels_meters, preds_meters)
    
    print(f"\ntest set metrics:")
    print(f"  mae:  {mae:.4f} m")
    print(f"  rmse: {rmse:.4f} m")
    print(f"  r²:   {r2:.4f}")
    print(f"  valid pixels: {len(all_labels):,}")
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Subsample if too many points for plotting
    if len(preds_meters) > 50000:
        indices = np.random.choice(len(preds_meters), 50000, replace=False)
        plot_preds = preds_meters[indices]
        plot_labels = labels_meters[indices]
    else:
        plot_preds = preds_meters
        plot_labels = labels_meters
    
    # Hexbin plot for density
    hexbin = ax.hexbin(plot_labels, plot_preds, gridsize=50, cmap='viridis', 
                       mincnt=1, bins='log')
    
    # Add colorbar
    cb = plt.colorbar(hexbin, ax=ax, label='Log10(Count)')
    
    # Add 1:1 line
    max_val = max(plot_labels.max(), plot_preds.max())
    min_val = min(plot_labels.min(), plot_preds.min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
            label='1:1 Line', alpha=0.8)
    
    # Labels and title
    ax.set_xlabel('Actual SWE (m)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted SWE (m)', fontsize=14, fontweight='bold')
    ax.set_title(f'Test Set: Predicted vs Actual SWE\n'
                 f'R² = {r2:.3f}, RMSE = {rmse:.3f} m, MAE = {mae:.3f} m',
                 fontsize=16, fontweight='bold')
    
    # Add text box with stats
    textstr = f'N = {len(all_labels):,}\nR² = {r2:.3f}\nRMSE = {rmse:.3f} m\nMAE = {mae:.3f} m'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim(0,5)
    ax.set_xlim(0,5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(checkpoint_dir, 'test_predicted_vs_actual.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved test evaluation plot to {plot_path}")
    
    # Also create a residual plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Residuals vs actual
    residuals = preds_meters - labels_meters
    axes[0].hexbin(plot_labels, residuals[indices] if len(preds_meters) > 50000 else residuals,
                   gridsize=50, cmap='RdBu_r', mincnt=1)
    axes[0].axhline(y=0, color='k', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Actual SWE (m)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Residual (Predicted - Actual) (m)', fontsize=12, fontweight='bold')
    axes[0].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of residuals
    axes[1].hist(residuals, bins=100, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero')
    axes[1].axvline(x=np.mean(residuals), color='g', linestyle='--', linewidth=2, 
                    label=f'Mean = {np.mean(residuals):.3f} m')
    axes[1].set_xlabel('Residual (m)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    residual_path = os.path.join(checkpoint_dir, 'test_residuals.png')
    plt.savefig(residual_path, dpi=150, bbox_inches='tight')
    print(f"Saved residual plots to {residual_path}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'n_pixels': len(all_labels),
        'preds': preds_meters,
        'actuals': labels_meters
    }
#+++++++++++++++

# ============================================================
# Main Training Script
# ============================================================
def main():
    import IPython
    # Config
    zarr_dir = "/discover/nobackup/cmbreen/gap-filling-data/zarr_chunks"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #IPython.embed()
    num_epochs = 30 #10 #1000
    batch_size = 16
    learning_rate = 1e-5 #0.01 ### learning rate start it really small? it will take longer to learn though 
    patience = 5 #400
    
    # Patching config
    patch_size = 128
    stride = 64  # 50% overlap
    min_valid_fraction = 0.3  # Skip patches with <30% valid pixels
    
    checkpoint_dir = "./checkpoints_Elevation"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load FULL zarr files
    print("\n" + "="*60)
    print("LOADING FULL ZARR FILES")
    print("="*60)
    
    train_x, train_y, val_x, val_y, test_x, test_y, train_y_mask, val_y_mask, test_y_mask = load_full_zarr_files(
        zarr_dir, split_basin_dict, flight_to_basin,
        skip_tb_channels=True  # ← NEW PARAMETER
    )
    # ========================================
    # CONVERT TO PATCHES
    # ========================================
    train_x, train_y, val_x, val_y, test_x, test_y, train_y_mask_patched, val_y_mask_patched, test_y_mask_patched = convert_to_patches(
        train_x, train_y, val_x, val_y, test_x, test_y,
        train_y_mask, val_y_mask, test_y_mask,
        patch_size=patch_size,
        stride=stride,
        min_valid_fraction=min_valid_fraction
    )
    
    # Normalize train and val together
    print("\n" + "="*60)
    print("NORMALIZING DATA")
    print("="*60)
    
    train_x_norm, val_x_norm, norm_mean, norm_std = normalize_dataset_per_channel(train_x, val_x, skip_channels=[1])
    
    # Normalize labels (Y) - same approach
    #IPython.embed()
    train_y_all = np.concatenate(train_y + val_y, axis=0)
    train_y_all_masks = np.concatenate(train_y_mask_patched + val_y_mask_patched, axis=0)
    #y_mean = np.nanmean(train_y_all)
    #y_std = np.nanstd(train_y_all)
    #train_y_all = np.concatenate(train_y + val_y, axis=0)

    # Only compute stats on VALID (positive, non-NaN) values
    #valid_y = train_y_all[(train_y_all > 0) & (~np.isnan(train_y_all))]
    valid_y = train_y_all[train_y_all_masks] ## does this work?? #[(train_y_all > 0) & (~np.isnan(train_y_all))]

    if len(valid_y) == 0:
        print("  ERROR: No valid Y values found! All labels are 0 or NaN")
        print(f"  Y range: [{train_y_all.min():.3f}, {train_y_all.max():.3f}]")
        print(f"  Y unique values: {np.unique(train_y_all[:100])}")  # Sample
        y_mean = 0.0
        y_std = 1.0
    else:
        y_mean = np.mean(valid_y)
        y_std = np.std(valid_y)
        print(f"  Valid Y pixels: {len(valid_y):,} / {train_y_all.size:,} ({100*len(valid_y)/train_y_all.size:.2f}%)")
        print(f"  Valid Y range: [{valid_y.min():.3f}, {valid_y.max():.3f}] m")
        
    print(f"\nTarget (SWE) normalization:")
    print(f"  Mean: {y_mean:.4f} m")
    print(f"  Std: {y_std:.4f} m")
    
    ########### I think something is breaking here??? because the mean and std are soooo small ########
    train_y_norm = [(y - y_mean) / (y_std + 1e-7) for y in train_y]
    val_y_norm = [(y - y_mean) / (y_std + 1e-7) for y in val_y]
    
    ## these are the masks for each dataset ##
    train_y_mask_patched
    val_y_mask_patched
    #####################
    
    # Data augmentation for training set
    print("\n" + "="*60)
    print("AUGMENTING TRAINING DATA")
    print("="*60)
    
    augmented_x = []
    augmented_y = []
    augmented_y_masks = []
    
    ############# this is all data, even where we have the masks !! #############
    for x, y, z in zip(train_x_norm, train_y_norm, train_y_mask_patched):
        # Apply flip
        x_flip, y_flip, y_mask_flip = random_flip(x, y, z)
        
        # Apply noise to X only (not Y)
        x_noisy = add_gaussian_noise(x_flip, mean=0, sigma=0.1) ## just do the first channel
        
        augmented_x.append(x_noisy)
        augmented_y.append(y_flip)
        augmented_y_masks.append(y_mask_flip)

    # Combine original + augmented
    combined_train_x = train_x_norm + augmented_x
    ## somehow combined trained y aren't normalized??
    combined_train_y = train_y_norm + augmented_y
    combined_train_y_mask = train_y_mask_patched + augmented_y_masks
    
    print(f"Training patches: {len(train_x_norm)} original + {len(augmented_x)} augmented = {len(combined_train_x)} total")
    
    # ========================================
    # SIMPLIFIED DATASET (NO MORE PATCHING NEEDED!)
    # ========================================
    class SimpleDataset(Dataset):
        def __init__(self, data, labels, labels_masks):
            self.data = data
            self.labels = labels
            self.masks = labels_masks
        
        def __len__(self):
            return len(self.data) #// 150
        
        def __getitem__(self, idx):
            # Data already patched, just return
            # Remove batch dimension: (1, C, H, W) -> (C, H, W)
            return self.data[idx][0], self.labels[idx][0], self.masks[idx][0]
    
    train_dataset = SimpleDataset(combined_train_x, combined_train_y, combined_train_y_mask)
    val_dataset = SimpleDataset(val_x_norm, val_y_norm, val_y_mask_patched)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} patches")
    print(f"  Val:   {len(val_dataset)} patches")
    
    ### fyi that labels in the dataset AREN'T NORMALIZED ####
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Model
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    
    # Auto-detect number of channels from first patch
    first_patch = train_x_norm[0]
    input_channels = first_patch.shape[1]  # (1, C, H, W) -> C
    
    print(f"Detected {input_channels} input channels")
    model = Model(input_channels=input_channels).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.000001)
    #criterion = nn.MSELoss() #nn.L1Loss()
    class ValueWeightedMSELoss(nn.Module):
        def __init__(self, alpha=1.0):
            super().__init__()
            self.alpha = alpha
        
        def forward(self, predictions, targets):
            #IPython.embed()
            # Weight proportional to target value (higher SWE = higher weight)
            weights = 1.0 + self.alpha * (targets / (targets.max() + 1e-8))
            loss = (predictions - targets) ** 2
            weighted_loss = loss * weights
            return weighted_loss.mean()

    criterion = ValueWeightedMSELoss(alpha=2.0)   

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    train_exp = True
    if train_exp == True:
        # Training loop
        print("\n" + "="*60)
        print("TRAINING")
        print("="*60)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")
            
            train_loss, _ = train_model(
                model, train_loader, optimizer, criterion, device, epoch, batch_size= 2, #batch_size
            save_dir=checkpoint_dir)
            
            val_loss, _ = validate_model(
                model, val_loader, criterion, device, epoch
            )
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(checkpoint_dir, 'best_model_cnn.pth')
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ Saved best model (val_loss={val_loss:.6f})")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
            # Plot loss curves
            print("\n" + "="*60)
            print("SAVING RESULTS")
            print("="*60)
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-o', label='Train Loss', linewidth=2)
            plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-s', label='Val Loss', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('L1 Loss', fontsize=12)
            plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = os.path.join(checkpoint_dir, 'loss_curve.png')
            plt.savefig(plot_path, dpi=150)
            print(f"Saved loss plot to {plot_path}")
            
            # Save loss values
            loss_txt_path = os.path.join(checkpoint_dir, 'loss_values.txt')
            with open(loss_txt_path, 'w') as f:
                f.write("Epoch,Train_Loss,Val_Loss\n")
                for i, (train_l, val_l) in enumerate(zip(train_losses, val_losses), 1):
                    f.write(f"{i},{train_l:.6f},{val_l:.6f}\n")
            print(f"Saved loss values to {loss_txt_path}")
            
            # Save normalization stats for later use
            norm_stats = {
                'X_mean': norm_mean[0, :, 0, 0].tolist(),
                'X_std': norm_std[0, :, 0, 0].tolist(),
                'Y_mean': float(y_mean),
                'Y_std': float(y_std)
            }
            
            import json
            stats_path = os.path.join(checkpoint_dir, 'normalization_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(norm_stats, f, indent=2)
            print(f"Saved normalization stats to {stats_path}")
            
            print("\n" + "="*60)
            print("TRAINING COMPLETE")
            print("="*60)
            print(f"Best validation loss: {best_val_loss:.6f}")
            print(f"Model saved to: {checkpoint_dir}/best_model_cnn.pth")

##############
## if just want the eval
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model_cnn.pth')))
    
    # Normalize test data using same stats as train/val
    print("\nNormalizing test data...")
    test_x_norm = []
    for data in train_x:
        normalized = (data - norm_mean) / (norm_std + 1e-7)
        test_x_norm.append(normalized)
    
    test_y_norm = [(y - y_mean) / (y_std + 1e-7) for y in test_y]
    
    # Evaluate on test set
    test_results = evaluate_test_set(
        model, 
        test_x_norm, 
        test_y_norm, 
        y_mean, 
        y_std, 
        device, 
        checkpoint_dir
    )
    
    # Save test metrics
    test_metrics = {
        'mae_m': float(test_results['mae']),
        'rmse_m': float(test_results['rmse']),
        'r2': float(test_results['r2']),
        'n_pixels': int(test_results['n_pixels'])
    }
    
    with open(os.path.join(checkpoint_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print("\n" + "="*60)
    print("ALL COMPLETE!")
    print("="*60)










##################


if __name__ == '__main__':
    main()