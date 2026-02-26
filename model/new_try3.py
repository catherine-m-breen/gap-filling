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


def random_flip(data, labels):
    """Random horizontal and vertical flips."""
    flip_w = np.random.choice([True, False])
    flip_h = np.random.choice([True, False])
    
    if flip_w:
        data = np.flip(data, axis=2).copy()
        labels = np.flip(labels, axis=1).copy()  # Match label dims
    if flip_h:
        data = np.flip(data, axis=3).copy()
        labels = np.flip(labels, axis=2).copy()
    
    return data, labels


def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to image."""
    _, ch, row, col = image.shape
    noisy = np.zeros_like(image)
    gauss = np.random.normal(mean, sigma, (1, ch, row, col))
    noisy = image + gauss
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
        
        if self.augment:
            # Apply same crop to both data and label
            combined = np.concatenate([data_patch, label_patch], axis=0)  # Stack along channel
            combined = random_crop(combined[None, :, :, :], self.crop_size)[0]
            data_patch = combined[:-1, :, :]  # All channels except last
            label_patch = combined[-1:, :, :]  # Last channel
            
        return data_patch, label_patch


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

def train_model(model, dataloader, optimizer, criterion, device, epoch, batch_size=8):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for i, (features, labels) in enumerate(dataloader):
        features = features.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)

        # Skip tiny patches
        if features.shape[2] < 2 or features.shape[3] < 2:
            continue

        output = model(features)
        
        # Mask: only compute loss on valid pixels (non-zero)
        mask = (labels != 0)
        labels_masked = labels[mask]
        output_masked = output[mask]

        # L1 loss + L1 regularization
        l1_lambda = 0.000001
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = criterion(output_masked, labels_masked) + l1_lambda * l1_norm

        loss.backward()
        total_loss += loss.item()

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

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} - Train Loss: {avg_loss:.6f}")
    
    return avg_loss, (all_labels, all_preds)


def validate_model(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (features, labels) in enumerate(dataloader):
            features = features.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            if features.shape[2] < 2 or features.shape[3] < 2:
                continue

            output = model(features)
            mask = (labels != 0)
            labels_masked = labels[mask]
            output_masked = output[mask]

            loss = criterion(output_masked, labels_masked)
            total_loss += loss.item()

            all_preds.extend(output_masked.detach().cpu().numpy())
            all_labels.extend(labels_masked.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch} - Val Loss: {avg_loss:.6f}, Valid pixels: {len(all_labels):,}")
    
    return avg_loss, (all_labels, all_preds)


# ============================================================
# Data Loading Functions - LOAD FULL ZARR FILES
# ============================================================

def load_full_zarr_files(zarr_dir, split_dict, flight_to_basin_dict):
    """
    Load FULL zarr files (not patches) and organize by train/val/test split.
    Patching will happen in the Dataset class on-the-fly.
    
    Returns: train_data, train_labels, val_data, val_labels, test_data, test_labels
    """
    zarr_dir = Path(zarr_dir)
    zarr_files = sorted(zarr_dir.glob("*.zarr"))
    
    train_x, train_y = [], []
    val_x, val_y = [], []
    test_x, test_y = [], []
    
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
        
        # Handle invalid values
        X[X == -9999] = 0.0
        X[X == 9999] = 0.0
        Y[Y == -9999] = 0.0
        Y[Y == 9999] = 0.0
        Y[Y < 0] = 0.0  # No negative SWE
        
        # Add batch dimension for consistency: (1, C, H, W)
        X = X[None, :, :, :]
        Y = Y[None, :, :, :]
        
        print(f"  Loaded {flight_id} ({basin}, {split}): X={X.shape}, Y={Y.shape}")
        
        # Assign to split
        if split == 'train':
            train_x.append(X)
            train_y.append(Y)
        elif split == 'val':
            val_x.append(X)
            val_y.append(Y)
        elif split == 'test':
            test_x.append(X)
            test_y.append(Y)
    
    print(f"\nLoaded: {len(train_x)} train, {len(val_x)} val, {len(test_x)} test FULL images")
    
    return train_x, train_y, val_x, val_y, test_x, test_y


# ============================================================
# Patch Conversion Function
# ============================================================

def convert_to_patches(train_x, train_y, val_x, val_y, test_x, test_y, 
                      patch_size=128, stride=64, min_valid_fraction=0.3):
    """
    Convert full images to patches for all splits.
    
    Args:
        train_x, train_y, val_x, val_y, test_x, test_y: Lists of full images
        patch_size: Size of patches to extract
        stride: Stride for patch extraction
        min_valid_fraction: Skip patches with <X% valid (non-zero) pixels in target
    
    Returns:
        Patched versions of all inputs as lists
    """
    print("\n" + "="*60)
    print("CONVERTING TO PATCHES")
    print("="*60)
    
    def extract_patches_from_list(data_list, label_list, split_name):
        """Extract patches from a list of images."""
        patched_data = []
        patched_labels = []
        total_patches = 0
        skipped_patches = 0
        
        print(f"\n{split_name} split:")
        for img_idx, (data, label) in enumerate(zip(data_list, label_list)):
            _, C, H, W = data.shape
            _, _, H_y, W_y = label.shape
            
            img_patches = 0
            
            for row in range(0, H - patch_size + 1, stride):
                for col in range(0, W - patch_size + 1, stride):
                    # Extract patch
                    data_patch = data[:, :, row:row+patch_size, col:col+patch_size]
                    label_patch = label[:, :, row:row+patch_size, col:col+patch_size]
                    
                    # Quality filter: skip patches with too many invalid pixels
                    valid_pixels = (label_patch != 0) & (~np.isnan(label_patch))
                    valid_fraction = valid_pixels.sum() / label_patch.size
                    
                    if valid_fraction < min_valid_fraction:
                        skipped_patches += 1
                        continue
                    
                    patched_data.append(data_patch)
                    patched_labels.append(label_patch)
                    img_patches += 1
                    total_patches += 1
            
            if (img_idx + 1) % 10 == 0 or img_idx == len(data_list) - 1:
                print(f"  Processed {img_idx + 1}/{len(data_list)} images, "
                      f"{total_patches} patches created, {skipped_patches} skipped")
        
        print(f"  Total {split_name}: {total_patches} patches from {len(data_list)} images")
        return patched_data, patched_labels
    
    # Convert each split
    train_x_patched, train_y_patched = extract_patches_from_list(train_x, train_y, "TRAIN")
    val_x_patched, val_y_patched = extract_patches_from_list(val_x, val_y, "VAL")
    test_x_patched, test_y_patched = extract_patches_from_list(test_x, test_y, "TEST")
    
    print(f"\n{'='*60}")
    print(f"PATCHING COMPLETE")
    print(f"  Train: {len(train_x_patched)} patches")
    print(f"  Val:   {len(val_x_patched)} patches")
    print(f"  Test:  {len(test_x_patched)} patches")
    print(f"{'='*60}\n")
    
    return train_x_patched, train_y_patched, val_x_patched, val_y_patched, test_x_patched, test_y_patched

def normalize_dataset_per_channel(train_data, val_data):
    """
    Normalize train and val together using combined statistics.
    Works with full images, not patches.
    
    Returns normalized train and val lists.
    """
    print("Computing normalization statistics from train+val combined...")
    
    # Combine train and val for computing stats
    combined_data = train_data + val_data
    
    # Stack all images
    all_data = np.concatenate(combined_data, axis=0)  # (N, C, H, W)
    
    # Compute mean and std per channel
    mean = np.mean(all_data, axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
    std = np.std(all_data, axis=(0, 2, 3), keepdims=True)    # (1, C, 1, 1)
    
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


# ============================================================
# Main Training Script
# ============================================================
def main():
    # Config
    zarr_dir = "/discover/nobackup/cmbreen/gap-filling-data/zarr_chunks"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    num_epochs = 1000
    batch_size = 16
    learning_rate = 0.01
    patience = 400
    
    # Patching config
    patch_size = 128
    stride = 64  # 50% overlap
    min_valid_fraction = 0.3  # Skip patches with <30% valid pixels
    
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load FULL zarr files
    print("\n" + "="*60)
    print("LOADING FULL ZARR FILES")
    print("="*60)
    
    train_x, train_y, val_x, val_y, test_x, test_y = load_full_zarr_files(
        zarr_dir, split_basin_dict, flight_to_basin
    )
    
    # ========================================
    # CONVERT TO PATCHES
    # ========================================
    train_x, train_y, val_x, val_y, test_x, test_y = convert_to_patches(
        train_x, train_y, val_x, val_y, test_x, test_y,
        patch_size=patch_size,
        stride=stride,
        min_valid_fraction=min_valid_fraction
    )
    
    # Normalize train and val together
    print("\n" + "="*60)
    print("NORMALIZING DATA")
    print("="*60)
    
    train_x_norm, val_x_norm, norm_mean, norm_std = normalize_dataset_per_channel(train_x, val_x)
    
    # Normalize labels (Y) - same approach
    train_y_all = np.concatenate(train_y + val_y, axis=0)
    y_mean = np.mean(train_y_all)
    y_std = np.std(train_y_all)
    
    print(f"\nTarget (SWE) normalization:")
    print(f"  Mean: {y_mean:.4f} m")
    print(f"  Std: {y_std:.4f} m")
    
    train_y_norm = [(y - y_mean) / (y_std + 1e-7) for y in train_y]
    val_y_norm = [(y - y_mean) / (y_std + 1e-7) for y in val_y]
    
    # Data augmentation for training set
    print("\n" + "="*60)
    print("AUGMENTING TRAINING DATA")
    print("="*60)
    
    augmented_x = []
    augmented_y = []
    
    for x, y in zip(train_x_norm, train_y_norm):
        # Apply flip
        x_flip, y_flip = random_flip(x, y)
        
        # Apply noise to X only (not Y)
        x_noisy = add_gaussian_noise(x_flip, mean=0, sigma=0.1)
        
        augmented_x.append(x_noisy)
        augmented_y.append(y_flip)
    
    # Combine original + augmented
    combined_train_x = train_x_norm + augmented_x
    combined_train_y = train_y_norm + augmented_y
    
    print(f"Training patches: {len(train_x_norm)} original + {len(augmented_x)} augmented = {len(combined_train_x)} total")
    
    # ========================================
    # SIMPLIFIED DATASET (NO MORE PATCHING NEEDED!)
    # ========================================
    class SimpleDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            # Data already patched, just return
            # Remove batch dimension: (1, C, H, W) -> (C, H, W)
            return self.data[idx][0], self.labels[idx][0]
    
    train_dataset = SimpleDataset(combined_train_x, combined_train_y)
    val_dataset = SimpleDataset(val_x_norm, val_y_norm)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} patches")
    print(f"  Val:   {len(val_dataset)} patches")
    
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
    criterion = nn.L1Loss()
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)
    
    
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
            model, train_loader, optimizer, criterion, device, epoch, batch_size=batch_size
        )
        
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


if __name__ == '__main__':
    main()