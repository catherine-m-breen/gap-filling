#!/bin/bash
## There is a .netrc file in the discover home directory and in the local home directory so you don't have to put in the password a bunch of times 

# chmod +x bash_script.sh
# ./bash_script.sh

# Base directory containing the TIF files
TIF_DIR="/Volumes/MyBook/aso_data/swe_tifs/colorado"

## The ones that are missing
TIFS=(
    "ASO_BlueRiver_Mosaic_2019June24-28_swe_50m.tif"
    "ASO_TenMileCk_2019June13-25_swe_50m.tif"
)

    # "ASO_WindyGap_2023Apr16_swe_50m.tif"
    # "ASO_WindyGap_2023May27_swe_50m.tif"
    # "ASO_WindyGap_2024Apr14_swe_50m.tif"
    # "ASO_WindyGap_2024Mar21-22_swe_50m.tif"
    # "ASO_WindyGap_2024May30_swe_50m.tif"
    # "ASO_WindyGap_2025Apr07_swe_50m.tif"
    # "ASO_WindyGap_2025Apr29_swe_50m.tif"
    # "ASO_WindyGap_2025May31_swe_50m.tif"
    # "ASO_WindyGap_Mosaic_2022Apr18_swe_50m.tif"
    # "ASO_YampaRiver_2024Apr11_swe_50m.tif"
    # "ASO_YampaRiver_2024May27-28_swe_50m.tif"
    # "ASO_YampaRiver_2025Apr11_swe_50m.tif"
    # "ASO_YampaRiver_2025May22-24_swe_50m.tif"

# Base output directory for downloaded data
#BASE_OUTPUT_DIR="/discover/nobackup/cmbreen/gap-filling-data/passive_microwave/nsidc_pm_data"
BASE_OUTPUT_DIR="/Volumes/MyBook/passive_microwave"


# Filter for specific channels (adjust as needed)
FILTER="*_N3.125km_F18_SSMIS_E_37H_*,*_N3.125km_F18_SSMIS_E_37V_*,*_N6.25km_F18_SSMIS_E_19H_*,*_N6.25km_F18_SSMIS_E_19V_*"

# Function to extract date from filename
# Function to extract date from filename
# Function to extract date from filename
extract_date() {
    local filename="$1"
    
    # Try pattern 1: YYYYMMDD (e.g., ASO_50M_SWE_USCOBR_20190419.tif)
    if [[ $filename =~ _([0-9]{8})\.tif$ ]]; then
        local date_str="${BASH_REMATCH[1]}"
        echo "${date_str:0:4}-${date_str:4:2}-${date_str:6:2}"
        return 0
    fi
    
    # Try pattern 2: YYYYMonthDD (with 3-letter OR full month name)
    # e.g., ASO_BlueRiver_2023Apr16_swe_50m.tif OR ASO_BlueRiver_2019June24_swe_50m.tif
    if [[ $filename =~ ([0-9]{4})(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)([0-9]{2})_swe ]]; then
        local year="${BASH_REMATCH[1]}"
        local month="${BASH_REMATCH[2]}"
        local day="${BASH_REMATCH[3]}"
        
        # Convert month name/abbreviation to number
        case $month in
            Jan|January) month="01";;
            Feb|February) month="02";;
            Mar|March) month="03";;
            Apr|April) month="04";;
            May) month="05";;
            Jun|June) month="06";;
            Jul|July) month="07";;
            Aug|August) month="08";;
            Sep|September) month="09";;
            Oct|October) month="10";;
            Nov|November) month="11";;
            Dec|December) month="12";;
        esac
        
        echo "${year}-${month}-${day}"
        return 0
    fi
    
    # Try pattern 3: YYYYMonthDD-DD (date range with 3-letter OR full month name)
    # e.g., ASO_BlueRiver_2019June24-28_swe_50m.tif
    if [[ $filename =~ ([0-9]{4})(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)([0-9]{2})-([0-9]{2})_swe ]]; then
        local year="${BASH_REMATCH[1]}"
        local month="${BASH_REMATCH[2]}"
        local day="${BASH_REMATCH[3]}"
        # Note: day_end="${BASH_REMATCH[4]}" if you need the end date
        
        # Convert month name/abbreviation to number
        case $month in
            Jan|January) month="01";;
            Feb|February) month="02";;
            Mar|March) month="03";;
            Apr|April) month="04";;
            May) month="05";;
            Jun|June) month="06";;
            Jul|July) month="07";;
            Aug|August) month="08";;
            Sep|September) month="09";;
            Oct|October) month="10";;
            Nov|November) month="11";;
            Dec|December) month="12";;
        esac
        
        echo "${year}-${month}-${day}"
        return 0
    fi
    
    return 1
}

# Counter for processed files
count=0
total=${#TIFS[@]}

echo "========================================================================"
echo "ASO TIF Passive Microwave Download Script (Missing Files Only)"
echo "========================================================================"
echo "TIF directory: $TIF_DIR (searching recursively)"
echo "Base output directory: $BASE_OUTPUT_DIR"
echo "Filter: $FILTER"
echo "Total TIF files to process: $total"
echo "========================================================================"
echo ""

# Loop through the specific missing TIF files
for filename in "${TIFS[@]}"; do
    # Search recursively for the file
    echo "Searching for: $filename"
    tif_file=$(find "$TIF_DIR" -name "$filename" -type f 2>/dev/null | head -n 1)
    
    # Check if file was found
    if [ -z "$tif_file" ]; then
        echo "WARNING: File not found: $filename - SKIPPING"
        echo ""
        continue
    fi
    
    echo "Found at: $tif_file"
    
    # Extract filename without extension
    filename_no_ext="${filename%.tif}"
    
    # Create output directory: nsidc_pm_data/[filename_without_tif]
    output_dir="${BASE_OUTPUT_DIR}/${filename_no_ext}"
    
    # Extract date from filename
    date=$(extract_date "$filename")
    
    if [ $? -ne 0 ] || [ -z "$date" ]; then
        echo "WARNING: Could not extract date from $filename - SKIPPING"
        echo ""
        continue
    fi
    
    # Increment counter
    ((count++))
    
    echo "------------------------------------------------------------------------"
    echo "Processing $count/$total: $filename"
    echo "ASO date: $date"
    echo "Output directory: $output_dir"
    echo "------------------------------------------------------------------------"
    
    # Run the download script - only for the exact ASO date
    python pm_download.py \
        --path "$tif_file" \
        --time_start "$date" \
        --filter "$FILTER" \
        --output "$output_dir"
    
    # Check if download was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed $filename"
    else
        echo "✗ ERROR processing $filename"
    fi
    
    echo ""
done

echo "========================================================================"
echo "Processing complete!"
echo "Processed $count out of $total TIF files"
echo "Data downloaded to: $BASE_OUTPUT_DIR/"
echo "========================================================================"