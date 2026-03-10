
land_class_dict = {11: "water", 12: 'perennial ice snow', 21: "Developed, open space", \
                   22: "Developed, Low Intensity", 23: "Developed: Medium Intensity", \
                    24: "Developed, High Intensity", 31: "Bare Rock/Sand/Clay", \
                    41: "Deciduous Forest", 42: "Evergreen Forest", \
                    43: "Mixed Forest", 52: "Shrub/ Scrub", 71: "Grasslands", \
                    81: "Pasture/Hay", 82: "Cultivated Crops", 90: "Woody Wetlands", \
                        95: "Emergent Wetlands"}

snowclass_dict = {1: "tundra", 2: "boreal forest", 3: "maritime", 4: "ephemeral", \
                  5: "praire", 6: "montane", 7: "ice"}

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


### year split: 

flight_split_dict = {
    # Animas (2021 → train)
    'ASO_Animas_Mosaic_2021Apr19_swe_50m.tif': 'train',
    'ASO_Animas_Mosaic_2021May15-16_swe_50m.tif': 'train',

    # Big and Little Thompson
    'ASO_BigThompsonLittleThompson_2023May21_swe_50m.tif': 'train',
    'ASO_BigThompson_2024Apr21_swe_50m.tif': 'val',
    'ASO_BigThompson_2025Apr11_swe_50m.tif': 'test',

    # Blue River
    'ASO_50M_SWE_USCOBR_20190419.tif': 'train',
    'ASO_50M_SWE_USCOBR_20190624.tif': 'train',
    'ASO_BlueRiver_Mosaic_2019Apr19_swe_50m.tif': 'train',
    'ASO_BlueRiver_Mosaic_2019June24-28_swe_50m.tif': 'train',
    'ASO_TenMileCk_2019June13-25_swe_50m.tif': 'train',
    'ASO_BlueRiver_Mosaic_2021Apr18_swe_50m.tif': 'train',
    'ASO_BlueRiver_Mosaic_2021May24_swe_50m.tif': 'train',
    'ASO_Blue_Mosaic_2022Apr19_swe_50m.tif': 'train',
    'ASO_Blue_Mosaic_2022May26_swe_50m.tif': 'train',
    'ASO_BlueRiver_2023Apr16_swe_50m.tif': 'train',
    'ASO_BlueRiver_2023May29_swe_50m.tif': 'train',
    'ASO_BlueRiver_2024Apr25_swe_50m.tif': 'val',
    'ASO_BlueRiver_2024Jun05_swe_50m.tif': 'val',
    'ASO_BlueRiver_2025Apr11_swe_50m.tif': 'test',
    'ASO_BlueRiver_2025May24_swe_50m.tif': 'test',

    # Boulder Creek
    'ASO_BoulderCreek_2023May09_swe_50m.tif': 'train',
    'ASO_BoulderCreek_2024May02_swe_50m.tif': 'val',
    'ASO_BoulderCreek_2025Apr09-10_swe_50m.tif': 'test',

    # Clear Creek
    'ASO_ClearCreek_2023May09_swe_50m.tif': 'train',
    'ASO_ClearCreek_2024May02_swe_50m.tif': 'val',
    'ASO_ClearCreek_2025Apr09-10_swe_50m.tif': 'test',

    # Conejos
    'ASO_50M_SWE_USCOCJ_20150406.tif': 'train',
    'ASO_50M_SWE_USCOCJ_20150602.tif': 'train',
    'ASO_50M_SWE_USCOCJ_20160403.tif': 'train',
    'ASO_Conejos_Mosaic_2021Apr20-21_swe_50m.tif': 'train',
    'ASO_Conejos_Mosaic_2021May16_swe_50m.tif': 'train',
    'ASO_Conejos_Mosaic_2022Apr15_swe_50m.tif': 'train',
    'ASO_Conejos_Mosaic_2022May10_swe_50m.tif': 'train',
    'ASO_Conejos_2023May05_swe_50m.tif': 'train',
    'ASO_Conejos_2024Apr02-03_swe_50m.tif': 'val',
    'ASO_Conejos_2024Apr02-03_swe_50m.tif.aux.xml': 'val',
    'ASO_Conejos_2024May08_swe_50m.tif': 'val',
    'ASO_Conejos_2025Mar21_swe_50m.tif': 'test',
    'ASO_Conejos_2025Apr28_swe_50m.tif': 'test',

    # Dolores
    'ASO_Dolores_Mosaic_2021Apr20-21_swe_50m.tif': 'train',
    'ASO_Dolores_Mosaic_2021May14_swe_50m.tif': 'train',
    'ASO_Dolores_Mosaic_2022Apr15_swe_50m.tif': 'train',
    'ASO_Dolores_Mosaic_2022May10_swe_50m.tif': 'train',
    'ASO_Dolores_2023Apr06_swe_50m.tif': 'train',
    'ASO_Dolores_2023May25_swe_50m.tif': 'train',
    'ASO_Dolores_2024Apr04_swe_50m.tif': 'val',
    'ASO_Dolores_2024Apr30_swe_50m.tif': 'val',
    'ASO_Dolores_2025Apr06_swe_50m.tif': 'test',
    'ASO_Dolores_2025Apr27_swe_50m.tif': 'test',

    # East River
    'ASO_50M_SWE_USCOCB_20160404.tif': 'train',
    'ASO_50M_SWE_USCOCB_20180330.tif': 'train',
    'ASO_50M_SWE_USCOGE_20180331.tif': 'train',
    'ASO_50M_SWE_USCOGE_20180524.tif': 'train',
    'ASO_50M_SWE_USCOGE_20190407.tif': 'train',
    'ASO_50M_SWE_USCOGE_20190610.tif': 'train',
    'ASO_Gunnison_EastRiver_2022Apr21_swe_50m.tif': 'train',
    'ASO_EastRiver_Mosaic_2022May18_swe_50m.tif': 'train',
    'ASO_EastRiver_2023Apr01_swe_50m.tif': 'train',
    'ASO_EastRiver_2023May23_swe_50m.tif': 'train',
    'ASO_EastRiver_2024Apr03_swe_50m.tif': 'val',
    'ASO_EastRiver_2024May20_swe_50m.tif': 'val',
    'ASO_EastRiver_2025Apr07_swe_50m.tif': 'test',
    'ASO_EastRiver_2025May20_swe_50m.tif': 'test',

    # North Fork Gunnison (2025 only → test)
    'ASO_GunnisonNorth_2025Mar27_swe_50m.tif': 'test',
    'ASO_GunnisonNorth_2025Apr27_swe_50m.tif': 'test',

    # Poudre River
    'ASO_Poudre_2023May22_swe_50m.tif': 'train',
    'ASO_Poudre_2024Apr15_swe_50m.tif': 'val',
    'ASO_Poudre_2025Apr07_swe_50m.tif': 'test',

    # Roaring Fork
    'ASO_50M_SWE_USCOCM_20190407.tif': 'train',
    'ASO_50M_SWE_USCOCM_20190610.tif': 'train',
    'ASO_RoaringFork_2023Apr11-12_swe_50m.tif': 'train',
    'ASO_RoaringFork_2023May28_swe_50m.tif': 'train',
    'ASO_RoaringFork_2024Apr09_swe_50m.tif': 'val',
    'ASO_RoaringFork_2024May22_swe_50m.tif': 'val',
    'ASO_RoaringFork_2025Apr12_swe_50m.tif': 'test',
    'ASO_RoaringFork_2025May22-23_swe_50m.tif': 'test',

    # St Vrain and Lefthand
    'ASO_StVrainLefthand_2023May21_swe_50m.tif': 'train',
    'ASO_StVrainLefthand_2024Apr21_swe_50m.tif': 'val',
    'ASO_StVrainLefthand_2025Apr11_swe_50m.tif': 'test',

    # Taylor
    'ASO_50M_SWE_USCOGT_20180330.tif': 'train',
    'ASO_50M_SWE_USCOGT_20190408.tif': 'train',
    'ASO_50M_SWE_USCOGT_20190609.tif': 'train',
    'ASO_Gunnison_Mosaic_2022Apr21_swe_50m.tif': 'train',
    'ASO_Gunnison_Lottis_2022May25_swe_50m.tif': 'train',
    'ASO_Gunnison_Taylor_2022Apr21_swe_50m.tif': 'train',
    'ASO_Gunnison_Taylor_2022May25_swe_50m.tif': 'train',
    'ASO_Taylor_2023Apr01_swe_50m.tif': 'train',
    'ASO_TaylorAndLottis_2023May23_swe_50m.tif': 'train',
    'ASO_Taylor_2024Apr04_swe_50m.tif': 'val',
    'ASO_Taylor_2024May20_swe_50m.tif': 'val',
    'ASO_Taylor_2025Apr07_swe_50m.tif': 'test',
    'ASO_Taylor_2025May20-21_swe_50m.tif': 'test',

    # Uncompahgre River (2014 only → train)
    'ASO_50M_SWE_USCOUB_20140320.tif': 'train',

    # Upper Rio Grande
    'ASO_50M_SWE_USCORG_20150407.tif': 'train',
    'ASO_50M_SWE_USCORG_20150602.tif': 'train',
    'ASO_50M_SWE_USCORG_20160403.tif': 'train',
    'ASO_RioGrande_2025Mar23-24_swe_50m.tif': 'test',
    'ASO_RioGrande_2025May13-15_swe_50m.tif': 'test',

    # Upper South Platte
    'ASO_SouthPlatte_2023Apr16_swe_50m.tif': 'train',
    'ASO_SouthPlatte_2023May26_swe_50m.tif': 'train',
    'ASO_SouthPlatte_2024Apr24-25_swe_50m.tif': 'val',
    'ASO_SouthPlatte_2024Jun05_swe_50m.tif': 'val',
    'ASO_SouthPlatte_2025Apr10_swe_50m.tif': 'test',
    'ASO_SouthPlatte_2025May27-30_swe_50m.tif': 'test',

    # Windy Gap
    'ASO_WindyGap_Mosaic_2022Apr18_swe_50m.tif': 'train',
    'ASO_WindyGap_2022May26_swe_50m.tif': 'train',
    'ASO_WindyGap_2023Apr16_swe_50m.tif': 'train',
    'ASO_WindyGap_2023May27_swe_50m.tif': 'train',
    'ASO_WindyGap_2024Mar21-22_swe_50m.tif': 'val',
    'ASO_WindyGap_2024Apr14_swe_50m.tif': 'val',
    'ASO_WindyGap_2024May30_swe_50m.tif': 'val',
    'ASO_WindyGap_2025Apr07_swe_50m.tif': 'test',
    'ASO_WindyGap_2025Apr29_swe_50m.tif': 'test',
    'ASO_WindyGap_2025May31_swe_50m.tif': 'test',

    # Yampa River
    'ASO_YampaRiver_2024Apr11_swe_50m.tif': 'val',
    'ASO_YampaRiver_2024May27-28_swe_50m.tif': 'val',
    'ASO_YampaRiver_2025Apr11_swe_50m.tif': 'test',
    'ASO_YampaRiver_2025May22-24_swe_50m.tif': 'test',
}