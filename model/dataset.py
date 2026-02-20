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

split_basin_dict = {'train': ["Poudre River", "Big and Little Thompson", "Windy Gap",\
                             "St Vrain and Lefthand","Boulder Creek", "Clear Creek", \
                             "Blue River", "Upper South Platte"], \
                   'val': ["Yampa River", "Roaring Fork", "North Fork Gunnison", "East River", "Taylor"], \
                   'test': ["Dolores","Animas","Upper Rio Grande","Conejos", "Uncompahgre River"]
}
flight_to_basin = {
    # Animas
    'ASO_Animas_Mosaic_2021Apr19_swe_50m.tif': 'Animas',
    'ASO_Animas_Mosaic_2021Apr19_swe_50m.tif.aux.xml': 'Animas',
    'ASO_Animas_Mosaic_2021May15-16_swe_50m.tif': 'Animas',
    'ASO_Animas_Mosaic_2021May15-16_swe_50m.tif.aux.xml': 'Animas',
    
    # Big and Little Thompson
    'ASO_BigThompson_2024Apr21_swe_50m.tif': 'Big and Little Thompson',
    'ASO_BigThompson_2025Apr11_swe_50m.tif': 'Big and Little Thompson',
    'ASO_BigThompsonLittleThompson_2023May21_swe_50m.tif': 'Big and Little Thompson',
    
    # Blue River
    'ASO_50M_SWE_USCOBR_20190419.tif': 'Blue River',
    'ASO_50M_SWE_USCOBR_20190624.tif': 'Blue River',
    'ASO_50M_SWE_USCOBR_20190624.tif.xml': 'Blue River',
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
    'ASO_Dolores_Mosaic_2021Apr20-21_swe_50m.tif.aux.xml': 'Dolores',
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
    
    # Uncompahgre River
    'ASO_50M_SWE_USCOUB_20140320.tif': 'Uncompahgre River',
    
    # Upper Rio Grande
    'ASO_50M_SWE_USCORG_20150407.tif': 'Upper Rio Grande',
    'ASO_50M_SWE_USCORG_20150407.tif.xml': 'Upper Rio Grande',
    'ASO_50M_SWE_USCORG_20150602.tif': 'Upper Rio Grande',
    'ASO_50M_SWE_USCORG_20160403.tif': 'Upper Rio Grande',
    'ASO_RioGrande_2025Mar23-24_swe_50m.tif': 'Upper Rio Grande',
    'ASO_RioGrande_2025Mar23-24_swe_50m.tif.aux.xml': 'Upper Rio Grande',
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
    
    # Read and stack predictors
    predictors = []
    
    def find_single_match(file_list, flight_id, layer_name):
        matches = [f for f in file_list if flight_id in os.path.basename(f)]
        
        if len(matches) == 0:
            raise ValueError(f"No match found for {layer_name} and {flight_id}")
        if len(matches) > 1:
            raise ValueError(f"Multiple matches found for {layer_name} and {flight_id}: {matches}")
        
        return matches[0]

   
    # --- Match single-layer datasets ---
    snow_path = get_single_file(snow_lookup, flight_id, "snowmap")
    land_path = get_single_file(land_lookup, flight_id, "landcover")
    tree_path = get_single_file(tree_lookup, flight_id, "treecover")
    elev_path = get_single_file(elev_lookup, flight_id, "elevation")
    viirs_path = get_single_file(viirs_lookup, flight_id, "viirs")

    # --- Passive microwave matching (4 bands) ---
    pm_matches = [f for fname, f in pm_lookup.items() if flight_id in fname]

    required_bands = ["37H", "37V", "19H", "19V"]
    pm_paths = {}
    for band in required_bands:
        band_matches = [f for f in pm_matches if band in f]
        if len(band_matches) > 1:
            selected = band_matches[1]  # take the second one
        else:
            try:
                selected = band_matches[0]
            except:
                IPython.embed()
        
        pm_paths[band] = selected

    tb37H_path = pm_paths["37H"]
    tb37V_path = pm_paths["37V"]
    tb19H_path = pm_paths["19H"]
    tb19V_path = pm_paths["19V"]

    layer_paths = [
        snow_path,
        land_path,
        tree_path,
        elev_path,
        tb37H_path,
        tb37V_path,
        tb19H_path,
        tb19V_path,
        viirs_path,
    ]
        
    # for path in layer_paths:
    #     with rasterio.open(path) as src:
    #         predictors.append(src.read(1))
    
    # X[:] = np.stack(predictors).astype("float32")
    # Y[0] = aso.astype("float32")
    
    # Metadata
    store.attrs["flight_id"] = flight_id
    #store.attrs["year"] = int(row["year"])
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

    predictors = []

    for path in layer_paths:
        with rasterio.open(path) as src:
            predictors.append(src.read(1))

    predictors = np.stack(predictors).astype("float32")

    canopy = predictors[2]

    forested = (canopy > 40).astype("float32")
    unforested = (canopy <= 40).astype("float32")

    X[:] = np.concatenate([
        predictors,
        forested[None, :, :],
        unforested[None, :, :]
    ], axis=0)
    Y[0] = aso.astype("float32")