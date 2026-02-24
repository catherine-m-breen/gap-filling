## module load anaconda/py3.11.7
## conda activate gapfill2

#######

'''
Need to deal with the -9999: 

# Handle NaN values and -9999
# Create validity mask BEFORE replacing
X_valid_mask = ~(np.isnan(X_patch) | (X_patch == -9999))
Y_valid_mask = ~(np.isnan(Y_patch) | (Y_patch == -9999))

# Replace with 0
X_patch = np.nan_to_num(X_patch, nan=0.0)
Y_patch = np.nan_to_num(Y_patch, nan=0.0)
X_patch[X_patch == -9999] = 0.0
Y_patch[Y_patch == -9999] = 0.0

# ... rest of normalization code ...

# Add mask to metadata (no model changes needed)
metadata['X_mask'] = X_valid_mask  # Can convert to tensor later if needed
metadata['Y_mask'] = Y_valid_mask

'''



#########
import torch
from torch.utils.data import Dataset, DataLoader
import zarr
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import random
import IPython


land_class_dict = {11: "water", 12: 'perennial ice snow', 21: "Developed, open space", \
                   22: "Developed, Low Intensity", 23: "Developed: Medium Intensity", \
                    24: "Developed, High Intensity", 31: "Bare Rock/Sand/Clay", \
                    41: "Deciduous Forest", 42: "Evergreen Forest", \
                    43: "Mixed Forest", 52: "Shrub/ Scrub", 71: "Grasslands", \
                    81: "Pasture/Hay", 82: "Cultivated Crops", 90: "Woody Wetlands", \
                        95: "Emergent Wetlands"}

snowclass_dict = {1: "ice", 2: "tundra", 3: "boreal forest", 4: "montane forest", \
                  5: "prairie", 6: "maritime", 7: "ephemeral"}

split_basin_dict = {'train': ["Poudre River", "Big and Little Thompson", "Windy Gap",\
                             "St Vrain and Lefthand","Boulder Creek", "Clear Creek", \
                             "Blue River", "Upper South Platte", "Yampa River"], \
                   'val': ["Roaring Fork", "North Fork Gunnison", "East River", "Taylor"], \
                   'test': ["Dolores","Animas","Upper Rio Grande","Conejos", "Uncompahgre River"]
}
flight_to_basin = {
    # Animas
    'ASO_Animas_Mosaic_2021Apr19_swe_50m.tif': 'Animas',
    'ASO_Animas_Mosaic_2021May15-16_swe_50m.tif': 'Animas',
    
    # Big and Little Thompson
    'ASO_BigThompson_2024Apr21_swe_50m.tif': 'Big and Little Thompson',
    'ASO_BigThompson_2025Apr11_swe_50m.tif': 'Big and Little Thompson',
    'ASO_BigThompsonLittleThompson_2023May21_swe_50m.tif': 'Big and Little Thompson',
    
    # Blue River
    'ASO_50M_SWE_USCOBR_20190419.tif': 'Blue River',
    'ASO_50M_SWE_USCOBR_20190624.tif': 'Blue River',
    'ASO_Blue_Mosaic_2022Apr19_swe_50m.tif': 'Blue River',
    'ASO_Blue_Mosaic_2022May26_swe_50m.tif': 'Blue River',
    'ASO_BlueRiver_2023Apr16_swe_50m.tif': 'Blue River',
    'ASO_BlueRiver_2023May29_swe_50m.tif': 'Blue River',
    'ASO_BlueRiver_2024Apr25_swe_50m.tif': 'Blue River',
    'ASO_BlueRiver_2024Jun05_swe_50m.tif': 'Blue River',
    'ASO_BlueRiver_2025Apr11_swe_50m.tif': 'Blue River',
    'ASO_BlueRiver_2025May24_swe_50m.tif': 'Blue River',
    'ASO_BlueRiver_Mosaic_2019Apr19_swe_50m.tif': 'Blue River',
    'ASO_BlueRiver_Mosaic_2019June24-28_swe_50m.tif': 'Blue River',
    'ASO_BlueRiver_Mosaic_2021Apr18_swe_50m.tif': 'Blue River',
    'ASO_BlueRiver_Mosaic_2021May24_swe_50m.tif': 'Blue River',
    'ASO_TenMileCk_2019June13-25_swe_50m.tif': 'Blue River',
    
    # Boulder Creek
    'ASO_BoulderCreek_2023May09_swe_50m.tif': 'Boulder Creek',
    'ASO_BoulderCreek_2024May02_swe_50m.tif': 'Boulder Creek',
    'ASO_BoulderCreek_2025Apr09-10_swe_50m.tif': 'Boulder Creek',
    
    # Clear Creek
    'ASO_ClearCreek_2023May09_swe_50m.tif': 'Clear Creek',
    'ASO_ClearCreek_2024May02_swe_50m.tif': 'Clear Creek',
    'ASO_ClearCreek_2025Apr09-10_swe_50m.tif': 'Clear Creek',
    
    # Conejos
    'ASO_50M_SWE_USCOCJ_20150406.tif': 'Conejos',
    'ASO_50M_SWE_USCOCJ_20150602.tif': 'Conejos',
    'ASO_50M_SWE_USCOCJ_20160403.tif': 'Conejos',
    'ASO_Conejos_2023May05_swe_50m.tif': 'Conejos',
    'ASO_Conejos_2024Apr02-03_swe_50m.tif': 'Conejos',
    'ASO_Conejos_2024Apr02-03_swe_50m.tif.aux.xml': 'Conejos',
    'ASO_Conejos_2024May08_swe_50m.tif': 'Conejos',
    'ASO_Conejos_2025Apr28_swe_50m.tif': 'Conejos',
    'ASO_Conejos_2025Mar21_swe_50m.tif': 'Conejos',
    'ASO_Conejos_Mosaic_2021Apr20-21_swe_50m.tif': 'Conejos',
    'ASO_Conejos_Mosaic_2021May16_swe_50m.tif': 'Conejos',
    'ASO_Conejos_Mosaic_2022Apr15_swe_50m.tif': 'Conejos',
    'ASO_Conejos_Mosaic_2022May10_swe_50m.tif': 'Conejos',
    
    # Dolores
    'ASO_Dolores_2023Apr06_swe_50m.tif': 'Dolores',
    'ASO_Dolores_2023May25_swe_50m.tif': 'Dolores',
    'ASO_Dolores_2024Apr04_swe_50m.tif': 'Dolores',
    'ASO_Dolores_2024Apr30_swe_50m.tif': 'Dolores',
    'ASO_Dolores_2025Apr06_swe_50m.tif': 'Dolores',
    'ASO_Dolores_2025Apr27_swe_50m.tif': 'Dolores',
    'ASO_Dolores_Mosaic_2021Apr20-21_swe_50m.tif': 'Dolores',
    'ASO_Dolores_Mosaic_2021May14_swe_50m.tif': 'Dolores',
    'ASO_Dolores_Mosaic_2022Apr15_swe_50m.tif': 'Dolores',
    'ASO_Dolores_Mosaic_2022May10_swe_50m.tif': 'Dolores',
    
    # East River
    'ASO_50M_SWE_USCOCB_20160404.tif': 'East River',
    'ASO_50M_SWE_USCOCB_20180330.tif': 'East River',
    'ASO_50M_SWE_USCOGE_20180331.tif': 'East River',
    'ASO_50M_SWE_USCOGE_20180524.tif': 'East River',
    'ASO_50M_SWE_USCOGE_20190407.tif': 'East River',
    'ASO_50M_SWE_USCOGE_20190610.tif': 'East River',
    'ASO_EastRiver_2023Apr01_swe_50m.tif': 'East River',
    'ASO_EastRiver_2023May23_swe_50m.tif': 'East River',
    'ASO_EastRiver_2024Apr03_swe_50m.tif': 'East River',
    'ASO_EastRiver_2024May20_swe_50m.tif': 'East River',
    'ASO_EastRiver_Mosaic_2022May18_swe_50m.tif' : 'East River',
    'ASO_EastRiver_2025Apr07_swe_50m.tif' : 'East River',
    'ASO_EastRiver_2025May20_swe_50m.tif' : 'East River',
    'ASO_Gunnison_EastRiver_2022Apr21_swe_50m.tif' : 'East River',

    # North Fork Gunnison
    'ASO_GunnisonNorth_2025Apr27_swe_50m.tif': 'North Fork Gunnison',
    'ASO_GunnisonNorth_2025Mar27_swe_50m.tif': 'North Fork Gunnison',
    
    # Poudre River
    'ASO_Poudre_2023May22_swe_50m.tif': 'Poudre River',
    'ASO_Poudre_2024Apr15_swe_50m.tif': 'Poudre River',
    'ASO_Poudre_2025Apr07_swe_50m.tif': 'Poudre River',
    
    # Roaring Fork
    'ASO_50M_SWE_USCOCM_20190407.tif': 'Roaring Fork',
    'ASO_50M_SWE_USCOCM_20190610.tif': 'Roaring Fork',
    'ASO_RoaringFork_2023Apr11-12_swe_50m.tif': 'Roaring Fork',
    'ASO_RoaringFork_2023May28_swe_50m.tif': 'Roaring Fork',
    'ASO_RoaringFork_2024Apr09_swe_50m.tif': 'Roaring Fork',
    'ASO_RoaringFork_2024May22_swe_50m.tif': 'Roaring Fork',
    'ASO_RoaringFork_2025Apr12_swe_50m.tif': 'Roaring Fork',
    'ASO_RoaringFork_2025May22-23_swe_50m.tif': 'Roaring Fork',
    
    # St Vrain and Lefthand
    'ASO_StVrainLefthand_2023May21_swe_50m.tif': 'St Vrain and Lefthand',
    'ASO_StVrainLefthand_2024Apr21_swe_50m.tif': 'St Vrain and Lefthand',
    'ASO_StVrainLefthand_2025Apr11_swe_50m.tif': 'St Vrain and Lefthand',
    
    # Taylor
    'ASO_50M_SWE_USCOGT_20180330.tif': 'Taylor',
    'ASO_50M_SWE_USCOGT_20190408.tif': 'Taylor',
    'ASO_50M_SWE_USCOGT_20190609.tif': 'Taylor',
    'ASO_Gunnison_Lottis_2022May25_swe_50m.tif': 'Taylor',
    'ASO_Gunnison_Mosaic_2022Apr21_swe_50m.tif': 'Taylor',
    'ASO_Gunnison_Taylor_2022Apr21_swe_50m.tif': 'Taylor',
    'ASO_Gunnison_Taylor_2022May25_swe_50m.tif': 'Taylor',
    'ASO_Taylor_2023Apr01_swe_50m.tif': 'Taylor',
    'ASO_Taylor_2024Apr04_swe_50m.tif': 'Taylor',
    'ASO_Taylor_2024May20_swe_50m.tif': 'Taylor',
    'ASO_Taylor_2025Apr07_swe_50m.tif': 'Taylor',
    'ASO_Taylor_2025May20-21_swe_50m.tif': 'Taylor',
    'ASO_TaylorAndLottis_2023May23_swe_50m.tif': 'Taylor',
    
    # Uncompahgre River
    'ASO_50M_SWE_USCOUB_20140320.tif': 'Uncompahgre River',
    
    # Upper Rio Grande
    'ASO_50M_SWE_USCORG_20150407.tif': 'Upper Rio Grande',
    'ASO_50M_SWE_USCORG_20150602.tif': 'Upper Rio Grande',
    'ASO_50M_SWE_USCORG_20160403.tif': 'Upper Rio Grande',
    'ASO_RioGrande_2025Mar23-24_swe_50m.tif': 'Upper Rio Grande',
    'ASO_RioGrande_2025May13-15_swe_50m.tif': 'Upper Rio Grande',
    
    # Upper South Platte
    'ASO_SouthPlatte_2023Apr16_swe_50m.tif': 'Upper South Platte',
    'ASO_SouthPlatte_2023May26_swe_50m.tif': 'Upper South Platte',
    'ASO_SouthPlatte_2024Apr24-25_swe_50m.tif': 'Upper South Platte',
    'ASO_SouthPlatte_2024Jun05_swe_50m.tif': 'Upper South Platte',
    'ASO_SouthPlatte_2025Apr10_swe_50m.tif': 'Upper South Platte',
    'ASO_SouthPlatte_2025May27-30_swe_50m.tif': 'Upper South Platte',
    
    # Windy Gap
    'ASO_WindyGap_2022May26_swe_50m.tif': 'Windy Gap',
    'ASO_WindyGap_2023Apr16_swe_50m.tif': 'Windy Gap',
    'ASO_WindyGap_2023May27_swe_50m.tif': 'Windy Gap',
    'ASO_WindyGap_2024Apr14_swe_50m.tif': 'Windy Gap',
    'ASO_WindyGap_2024Mar21-22_swe_50m.tif': 'Windy Gap',
    'ASO_WindyGap_2024May30_swe_50m.tif': 'Windy Gap',
    'ASO_WindyGap_2025Apr07_swe_50m.tif': 'Windy Gap',
    'ASO_WindyGap_2025Apr29_swe_50m.tif': 'Windy Gap',
    'ASO_WindyGap_2025May31_swe_50m.tif': 'Windy Gap',
    'ASO_WindyGap_Mosaic_2022Apr18_swe_50m.tif': 'Windy Gap',
    
    # Yampa River
    'ASO_YampaRiver_2024Apr11_swe_50m.tif': 'Yampa River',
    'ASO_YampaRiver_2024May27-28_swe_50m.tif': 'Yampa River',
    'ASO_YampaRiver_2025Apr11_swe_50m.tif': 'Yampa River',
    'ASO_YampaRiver_2025May22-24_swe_50m.tif': 'Yampa River'
}


# -------------------------------
# Snow class mapping
# -------------------------------
SNOW_CLASSES = [0, 1, 2, 4, 5, 6, 7]  # Original snow class values
SNOW_CLASS_TO_IDX = {v: i for i, v in enumerate(SNOW_CLASSES)}  # Map to 0..4
NUM_SNOW_CLASSES = len(SNOW_CLASSES)  # 5

# -------------------------------
# Dataset
# -------------------------------
class ASOPatchDataset(Dataset):
    def __init__(
        self,
        zarr_dir: str,
        split: str = 'train',
        patch_size: int = 256,
        stride: int = 128,
        normalize: bool = True,
        random_crop: bool = False,
        seed: int = 42,
        global_stats: Dict = None
    ):
        self.zarr_dir = Path(zarr_dir)
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.normalize = normalize
        self.random_crop = random_crop
        self.global_stats = global_stats
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.basins = split_basin_dict[split]
        self.zarr_files = self._get_zarr_files()
        self.patches = self._create_patch_index()
        
        print(f"{split.upper()} split: {len(self.zarr_files)} files, {len(self.patches)} patches")
        print(f"Basins: {self.basins}")

    def _get_zarr_files(self) -> List[Path]:
        zarr_files = []
        for zarr_path in sorted(self.zarr_dir.glob("*.zarr")):
            tif_name = zarr_path.stem + '.tif'
            if tif_name in flight_to_basin:
                basin = flight_to_basin[tif_name]
                if basin in self.basins:
                    zarr_files.append(zarr_path)
        return zarr_files

    def _create_patch_index(self) -> List[Tuple[int, int, int]]:
        patches = []
        for file_idx, zarr_path in enumerate(self.zarr_files):
            z = zarr.open(str(zarr_path), mode='r')
            X = z['X']
            _, height, width = X.shape

            if self.random_crop:
                patches.append((file_idx, -1, -1))
            else:
                for row in range(0, height - self.patch_size + 1, self.stride):
                    for col in range(0, width - self.patch_size + 1, self.stride):
                        patches.append((file_idx, row, col))
        return patches

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        file_idx, row, col = self.patches[idx]
        zarr_path = self.zarr_files[file_idx]
        
        # Load zarr
        z = zarr.open(str(zarr_path), mode='r')
        X = np.array(z['X'], dtype=np.float32)
        Y = np.array(z['Y'], dtype=np.float32)
        _, height, width = X.shape
        
        # Random crop
        if self.random_crop or (row == -1 and col == -1):
            row = np.random.randint(0, max(1, height - self.patch_size))
            col = np.random.randint(0, max(1, width - self.patch_size))

        X_patch = X[:, row:row+self.patch_size, col:col+self.patch_size]
        Y_patch = Y[:, row:row+self.patch_size, col:col+self.patch_size]

        # Padding for edge cases
        if X_patch.shape[1] < self.patch_size or X_patch.shape[2] < self.patch_size:
            X_padded = np.zeros((X_patch.shape[0], self.patch_size, self.patch_size), dtype=np.float32)
            Y_padded = np.zeros((Y_patch.shape[0], self.patch_size, self.patch_size), dtype=np.float32)
            X_padded[:, :X_patch.shape[1], :X_patch.shape[2]] = X_patch
            Y_padded[:, :Y_patch.shape[1], :Y_patch.shape[2]] = Y_patch
            X_patch, Y_patch = X_padded, Y_padded

        # -----------------------
        # Handle invalid values
        # -----------------------
        X_valid_mask = ~(np.isnan(X_patch) | (X_patch == -9999) | (X_patch[0] == 255) | (X_patch[1] == 250))
        Y_valid_mask = ~(np.isnan(Y_patch) | (Y_patch == -9999))
        X_patch[~X_valid_mask] = 0.0
        Y_patch[~Y_valid_mask] = 0.0

        # -----------------------
        # One-hot encode snow_map (channel 0)
        # -----------------------
        snow_orig = X_patch[0]  # Original snow class
        # Map original classes 0,1,2,5,6 → 0..4
        snow_mapped = np.vectorize(SNOW_CLASS_TO_IDX.get)(snow_orig)
        snow_onehot = np.zeros((NUM_SNOW_CLASSES, self.patch_size, self.patch_size), dtype=np.float32)
        for i in range(NUM_SNOW_CLASSES):
            snow_onehot[i] = (snow_mapped == i).astype(np.float32)

        # Replace original snow channel with one-hot channels
        X_patch = np.concatenate([snow_onehot, X_patch[1:]], axis=0)

        # -----------------------
        # Normalize continuous channels
        # -----------------------
        if self.normalize and self.global_stats is not None:
            # Continuous: [canopy_cover(2), elevation(3), brightness 4-7, ndsi(8)]
            # NOTE: after adding snow one-hot, channel indices shift by NUM_SNOW_CLASSES-1
            shift = NUM_SNOW_CLASSES - 1
            continuous_channels = np.array([2, 3, 4, 5, 6, 7, 8]) + shift
            X_mean = self.global_stats['X_mean'][:, None, None]
            X_std = self.global_stats['X_std'][:, None, None]
            for c in continuous_channels:
                X_patch[c] = (X_patch[c] - X_mean[c]) / (X_std[c] + 1e-8)
            
            # Normalize target SWE
            Y_patch = (Y_patch - self.global_stats['Y_mean']) / (self.global_stats['Y_std'] + 1e-8)
            
            # Zero invalid locations
            X_patch *= X_valid_mask
            Y_patch *= Y_valid_mask

        # Convert to tensors
        X_tensor = torch.from_numpy(X_patch).float()
        Y_tensor = torch.from_numpy(Y_patch).float()

        tif_name = zarr_path.stem + '.tif'
        basin = flight_to_basin.get(tif_name, 'Unknown')
        metadata = {
            'file': zarr_path.name,
            'basin': basin,
            'row': row,
            'col': col,
            'height': height,
            'width': width,
            'X_mask': torch.from_numpy(X_valid_mask).float(),
            'Y_mask': torch.from_numpy(Y_valid_mask).float()
        }

        return X_tensor, Y_tensor, metadata

# ========================================
# FIX 3: COMPUTE GLOBAL STATISTICS
# ========================================
def compute_global_statistics(zarr_dir: str, split: str = 'train') -> Dict:
    """
    Compute global mean/std from the training set for proper normalization.
    Only computes stats for continuous features.
    """
    print(f"\nComputing global statistics from {split} split...")
    
    # Create temporary dataset without normalization
    temp_dataset = ASOPatchDataset(
        zarr_dir=zarr_dir,
        split=split,
        normalize=False,
        random_crop=False
    )
    
    # Define which channels are continuous
    continuous_channels = [2, 3, 4, 5, 6, 7, 8]  # Elevation, TBs, NDSI
    categorical_channels = [0, 1, 9, 10]   # Snow, land, canopy, masks
    
    # Accumulate statistics only for continuous channels
    X_sum = np.zeros(11, dtype=np.float64)
    X_sq_sum = np.zeros(11, dtype=np.float64)
    X_count = np.zeros(11, dtype=np.int64)
    
    Y_sum = 0.0
    Y_sq_sum = 0.0
    Y_count = 0
    
    print("Scanning all patches...")
    for i in range(len(temp_dataset)):
        X, Y, metadata = temp_dataset[i]
        X = X.numpy()
        Y = Y.numpy()

        X_mask = metadata['X_mask'].numpy().astype(bool)
        Y_mask = metadata['Y_mask'].numpy().astype(bool)
        
        # Only count valid values for continuous channels
        # for c in continuous_channels:
        #     valid_mask = (X[c] != -9999) & (X[c] != 255) & ~np.isnan(X[c]) #(X[c] != 0) &
        #     X_sum[c] += X[c][valid_mask].sum()
        #     X_sq_sum[c] += (X[c][valid_mask] ** 2).sum()
        #     X_count[c] += valid_mask.sum()

        for c in continuous_channels:
            valid_mask = X_mask[c]  # ← USE THE TRUE MASK
            valid_values = X[c][valid_mask]

            X_sum[c] += valid_values.sum()
            X_sq_sum[c] += (valid_values ** 2).sum()
            X_count[c] += valid_values.size
                
        # Target (SWE)
        # Y_valid_mask = (Y != -9999) & ~np.isnan(Y) #(Y != 0) &
        # Y_mask = metadata['Y_mask'].numpy().astype(bool)
        # Y_valid_mask = Y[Y_mask]
        # Y_sum += Y[Y_valid_mask].sum()
        # Y_sq_sum += (Y[Y_valid_mask] ** 2).sum()
        # Y_count += Y_valid_mask.sum()
        Y_valid_values = Y[Y_mask]
        Y_sum += Y_valid_values.sum()
        Y_sq_sum += (Y_valid_values ** 2).sum()
        Y_count += Y_valid_values.size
                    
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(temp_dataset)} patches")
    
    # Initialize arrays
    X_mean = np.zeros(11, dtype=np.float64)
    X_std = np.ones(11, dtype=np.float64)  # Default std=1
    
    # Compute mean and std only for continuous channels
    for c in continuous_channels:
        if X_count[c] > 0:
            X_mean[c] = X_sum[c] / X_count[c]
            X_var = (X_sq_sum[c] / X_count[c]) - (X_mean[c] ** 2)
            X_std[c] = np.sqrt(np.maximum(X_var, 0))
    
    # Categorical channels: mean=0, std=1 (no normalization effect)
    for c in categorical_channels:
        X_mean[c] = 0.0
        X_std[c] = 1.0
    
    # Target (SWE)
    Y_mean = Y_sum / (Y_count + 1e-8)
    Y_var = (Y_sq_sum / (Y_count + 1e-8)) - (Y_mean ** 2)
    Y_std = np.sqrt(np.maximum(Y_var, 0))
    
    print("\nGlobal Statistics:")
    print(f"Continuous channels {continuous_channels}:")
    print(f"  X mean: {X_mean[continuous_channels]}")
    print(f"  X std: {X_std[continuous_channels]}")
    print(f"Categorical channels {categorical_channels}:")
    print(f"  X mean: {X_mean[categorical_channels]} (should be all 0)")
    print(f"  X std: {X_std[categorical_channels]} (should be all 1)")
    print(f"Y mean: {Y_mean:.2f}")
    print(f"Y std: {Y_std:.2f}")
    print(f"X counts: {X_count}")
    print(f"Y count: {Y_count}")
    
    return {
        'X_mean': X_mean,
        'X_std': X_std,
        'Y_mean': Y_mean,
        'Y_std': Y_std
    }

# ========================================
# FIX 4: UPDATED DATALOADER CREATION
# ========================================
def create_dataloaders(
    zarr_dir: str,
    batch_size: int = 32,
    patch_size: int = 256,
    stride: int = 128,
    num_workers: int = 4,
    normalize: bool = True,
    random_crop_train: bool = False
) -> Dict[str, DataLoader]:
    """Create train, val, and test dataloaders with proper normalization."""
    
    # Compute global statistics from training set
    global_stats = None
    if normalize:
        global_stats = compute_global_statistics(zarr_dir, split='train')
    
    datasets = {}
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        use_random = random_crop_train if split == 'train' else False
        
        datasets[split] = ASOPatchDataset(
            zarr_dir=zarr_dir,
            split=split,
            patch_size=patch_size,
            stride=stride,
            normalize=normalize,
            random_crop=use_random,
            global_stats=global_stats  # PASS GLOBAL STATS
        )
        
        shuffle = (split == 'train')
        
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train')
        )
    
    return dataloaders
