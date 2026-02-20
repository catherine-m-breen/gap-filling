import zarr
import rasterio
import numpy as np
import pandas as pd
import os
import glob
import IPython

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

# don't worry about split by year for now, it's stronger if we can predict in new areas, because it suggests
# model has learned general relationships between sensors and trees and neighboring relationships
# split_year_dict = {}

output_dir = "/discover/nobackup/cmbreen/gap-filling-data/zarr_chunks"

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

# Build lookups once to avoid repeated glob searches
snow_lookup = build_lookup(snowmappath)
land_lookup = build_lookup(landcoverpath)
tree_lookup = build_lookup(treecoverpath)
elev_lookup = build_lookup(elevation)
viirs_lookup = build_lookup(viirs)
pm_lookup = build_lookup(passivemicrowavepath)

for aso_flight in asopath:

    flight_id = aso_flight.split('/')[-1].split(".")[0]
    out_path = os.path.join(output_dir, f"{flight_id}.zarr")
    
    print(f"Processing {flight_id}...")
    
    # Open ASO to get shape
    with rasterio.open(aso_flight) as src:
        H, W = src.height, src.width
        crs = src.crs.to_string()
        transform = src.transform
        resolution = src.res
        aso = src.read(1)

    # Create Zarr store
    store = zarr.open(out_path, mode="w")
    
    X = store.create_dataset(
        "X",
        shape=(11, H, W),
        chunks=(11, 256, 256),
        dtype="float32"
    )
    
    Y = store.create_dataset(
        "Y",
        shape=(1, H, W),
        chunks=(1, 256, 256),
        dtype="float32"
    )
    
    # --- Match single-layer datasets ---
    snow_path = get_single_file(snow_lookup, flight_id, "snowmap")
    land_path = get_single_file(land_lookup, flight_id, "landcover")
    tree_path = get_single_file(tree_lookup, flight_id, "treecover")
    elev_path = get_single_file(elev_lookup, flight_id, "elevation")
    viirs_path = get_single_file(viirs_lookup, flight_id, "viirs")

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
                print(f"  Using second match for {band}")
                with rasterio.open(band_matches[1]) as src:
                    pm_arrays.append(src.read(1))
            else:
                with rasterio.open(band_matches[0]) as src:
                    pm_arrays.append(src.read(1))

    # Read single-layer datasets
    predictors = []
    for path in [snow_path, land_path, tree_path, elev_path]:
        with rasterio.open(path) as src:
            predictors.append(src.read(1))
    
    # Add PM arrays (either real or NaN-filled)
    predictors.extend(pm_arrays)
    
    # Add VIIRS
    with rasterio.open(viirs_path) as src:
        predictors.append(src.read(1))

    predictors = np.stack(predictors).astype("float32")

    # Create forest masks
    canopy = predictors[2]
    forested = (canopy > 40).astype("float32")
    unforested = (canopy <= 40).astype("float32")

    # Stack everything
    X[:] = np.concatenate([
        predictors,
        forested[None, :, :],
        unforested[None, :, :]
    ], axis=0)
    Y[0] = aso.astype("float32")
    
    # Metadata
    store.attrs["flight_id"] = flight_id
    aso_filename = os.path.basename(aso_flight)
    store.attrs["basin"] = flight_to_basin.get(aso_filename, "unknown")
    store.attrs["crs"] = crs
    store.attrs["transform"] = tuple(transform)
    store.attrs["resolution"] = resolution
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
        "unforested_mask"
    ]
    
    print(f"Completed {flight_id}")

print("\nFinished.")