# module load anaconda/py3.11.7
# run from conda activate gapfill2

import zarr
import rasterio
import numpy as np
import pandas as pd
import os
import glob
import IPython


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



## each of these is 117 long
## they all have already been reprojected and aligned to match the aso data at 50m resolution
asopath = glob.glob("/discover/nobackup/cmbreen/aso_data_wbasinname/swe_tifs/colorado/**/*.tif")
snowmappath = glob.glob("/discover/nobackup/cmbreen/gap-filling-data/snowclassmap/processed_50m/*.tif")
landcoverpath = glob.glob("/discover/nobackup/cmbreen/gap-filling-data/landcover/landcover_50m/*.tif")
treecoverpath = glob.glob("/discover/nobackup/cmbreen/gap-filling-data/treecanopycover/treecover_50m/*.tif")
elevation = glob.glob("/discover/nobackup/cmbreen/gap-filling-data/elevation/elevation_50m/*.tif")
viirs = glob.glob("/discover/nobackup/cmbreen/gap-filling-data/viirs/viirs_50m/*.tif")
## this one has 117*4 or 468
passivemicrowavepath = glob.glob("/discover/nobackup/cmbreen/gap-filling-data/passive_microwave/pm_50m/*.tif")

output_dir = "/discover/nobackup/cmbreen/gap-filling-data/zarr_patches_128x128"
os.makedirs(output_dir, exist_ok=True)

# Configuration
PATCH_SIZE = 128
STRIDE = 64  # 50% overlap for more training samples
MIN_VALID_FRACTION = 0.3  # Skip patches with >70% NaN/zeros in target

## helper function ##
def build_lookup(file_list):
    lookup = {}
    for path in file_list:
        fname = os.path.basename(path)
        lookup[fname] = path
    return lookup

def get_single_file(lookup, flight_id, layer_name):
    """Return a single matching file for a flight; raise errors if none or multiple."""
    matches = [f for fname, f in lookup.items() if flight_id in fname]
    if len(matches) == 0:
        raise ValueError(f"No match found for {layer_name} and flight {flight_id}")
    if len(matches) > 1:
        raise ValueError(f"Multiple matches found for {layer_name} and flight {flight_id}: {matches}")
    return matches[0]

def get_split_for_basin(basin, split_dict):
    """Determine train/val/test split for a basin."""
    for split_name, basin_list in split_dict.items():
        if basin in basin_list:
            return split_name
    return "unknown"

# Build lookups once to avoid repeated glob searches
snow_lookup = build_lookup(snowmappath)
land_lookup = build_lookup(landcoverpath)
tree_lookup = build_lookup(treecoverpath)
elev_lookup = build_lookup(elevation)
viirs_lookup = build_lookup(viirs)
pm_lookup = build_lookup(passivemicrowavepath)

total_patches = 0
skipped_patches = 0

for aso_flight in asopath:

    flight_id = aso_flight.split('/')[-1].split(".")[0]
    aso_filename = os.path.basename(aso_flight)
    basin = flight_to_basin.get(aso_filename, "unknown")
    split = get_split_for_basin(basin, split_basin_dict)
    
    print(f"\nProcessing {flight_id} (Basin: {basin}, Split: {split})...")
    
    # Open ASO to get shape and metadata
    with rasterio.open(aso_flight) as src:
        H, W = src.height, src.width
        crs = src.crs.to_string()
        transform = src.transform
        resolution = src.res
        aso = src.read(1).astype("float32")
    
    # Validate ASO data
    if np.isnan(aso).all() or (aso == 0).all():
        print(f"  WARNING: {flight_id} has all-NaN or all-zero ASO data, skipping...")
        continue
    
    # --- Match single-layer datasets ---
    try:
        snow_path = get_single_file(snow_lookup, flight_id, "snowmap")
        land_path = get_single_file(land_lookup, flight_id, "landcover")
        tree_path = get_single_file(tree_lookup, flight_id, "treecover")
        elev_path = get_single_file(elev_lookup, flight_id, "elevation")
        viirs_path = get_single_file(viirs_lookup, flight_id, "viirs")
    except ValueError as e:
        print(f"  ERROR: {e}, skipping {flight_id}")
        continue

    # --- Passive microwave matching (4 bands) ---
    pm_matches = [f for fname, f in pm_lookup.items() if flight_id in fname]
    required_bands = ["37H", "37V", "19H", "19V"]
    pm_arrays = []
    
    if len(pm_matches) == 0:
        print(f"  Warning: No PM data for {flight_id}, filling with NaN")
        pm_arrays = [np.full((H, W), np.nan, dtype='float32') for _ in range(4)]
    else:
        for band in required_bands:
            band_matches = [f for f in pm_matches if band in f]
            
            if len(band_matches) == 0:
                print(f"  Warning: No {band} for {flight_id}, filling with NaN")
                pm_arrays.append(np.full((H, W), np.nan, dtype='float32'))
            elif len(band_matches) > 1:
                print(f"  Using first match for {band}")
                with rasterio.open(band_matches[0]) as src:
                    pm_arrays.append(src.read(1).astype('float32'))
            else:
                with rasterio.open(band_matches[0]) as src:
                    pm_arrays.append(src.read(1).astype('float32'))

    # Read single-layer datasets
    predictors = []
    for path in [snow_path, land_path, tree_path, elev_path]:
        with rasterio.open(path) as src:
            data = src.read(1).astype('float32')
            if data.shape != (H, W):
                raise ValueError(f"Shape mismatch for {path}: expected ({H}, {W}), got {data.shape}")
            predictors.append(data)
    
    # Add PM arrays
    predictors.extend(pm_arrays)
    
    # Add VIIRS
    with rasterio.open(viirs_path) as src:
        viirs_data = src.read(1).astype('float32')
        if viirs_data.shape != (H, W):
            raise ValueError(f"Shape mismatch for VIIRS: expected ({H}, {W}), got {viirs_data.shape}")
        predictors.append(viirs_data)

    predictors = np.stack(predictors).astype("float32")  # Shape: (10, H, W)

    # Create forest mask
    canopy = predictors[2]
    forested = (canopy > 40).astype("float32")

    # One-hot encode snow class 
    snow = predictors[0]  # Shape: (H, W)
    unique_classes = np.unique(snow[~np.isnan(snow)])
    print(f"  Snow classes found: {unique_classes}")
    
    snow_classes = [1, 2, 3, 4, 5, 6, 7]  # Adjust based on your data
    snow_one_hot = []
    for class_id in snow_classes:
        snow_class_mask = (snow == class_id).astype("float32")
        snow_one_hot.append(snow_class_mask)
    
    snow_one_hot = np.stack(snow_one_hot)  # Shape: (6, H, W)

    # Stack all channels: 10 predictors + 1 forested + 6 one-hot = 17 channels
    all_channels = np.concatenate([
        predictors,           # 10 channels
        forested[None, :, :], # 1 channel
        snow_one_hot          # 6 channels
    ], axis=0)  # Shape: (17, H, W)
    
    n_channels = all_channels.shape[0]
    print(f"  Total channels: {n_channels}")
    
    # --- Extract patches ---
    flight_patches = 0
    for row in range(0, H - PATCH_SIZE + 1, STRIDE):
        for col in range(0, W - PATCH_SIZE + 1, STRIDE):
            # Extract patch
            x_patch = all_channels[:, row:row+PATCH_SIZE, col:col+PATCH_SIZE]
            y_patch = aso[row:row+PATCH_SIZE, col:col+PATCH_SIZE]
            
            # Quality filter: skip if too many invalid pixels in target
            valid_pixels = (~np.isnan(y_patch)) & (y_patch > 0)
            valid_fraction = valid_pixels.sum() / y_patch.size
            
            if valid_fraction < MIN_VALID_FRACTION:
                skipped_patches += 1
                continue
            
            # Save patch as individual zarr file
            patch_name = f"{flight_id}_r{row}_c{col}.zarr"
            patch_path = os.path.join(output_dir, patch_name)
            
            store = zarr.open(patch_path, mode="w")
            
            # Create datasets
            store.create_dataset(
                "X",
                data=x_patch,
                chunks=(n_channels, PATCH_SIZE, PATCH_SIZE),
                dtype="float32"
            )
            
            store.create_dataset(
                "Y",
                data=y_patch[None, :, :],  # Add channel dimension
                chunks=(1, PATCH_SIZE, PATCH_SIZE),
                dtype="float32"
            )
            
            # Metadata
            store.attrs["flight_id"] = flight_id
            store.attrs["basin"] = basin
            store.attrs["split"] = split
            store.attrs["row"] = int(row)
            store.attrs["col"] = int(col)
            store.attrs["crs"] = crs
            store.attrs["transform"] = tuple(transform)
            store.attrs["resolution"] = resolution
            store.attrs["valid_fraction"] = float(valid_fraction)
            store.attrs["channel_names"] = [
                "snow_class",
                "landcover",
                "canopy_cover",
                "elevation",
                "tb_37H",
                "tb_37V",
                "tb_19H",
                "tb_19V",
                "ndsi",
                "forested_mask",
                "snow_class_1_tundra",
                "snow_class_2_boreal_forest",
                "snow_class_3_maritime",
                "snow_class_4_ephemeral",
                "snow_class_5_prairie",
                "snow_class_6_montane",
                "snow_class_7_ice",
            ]
            
            flight_patches += 1
            total_patches += 1
    
    print(f"  Created {flight_patches} patches from {flight_id}")

print(f"\n{'='*60}")
print(f"Finished processing!")
print(f"Total patches created: {total_patches}")
print(f"Patches skipped (low valid fraction): {skipped_patches}")
print(f"Output directory: {output_dir}")