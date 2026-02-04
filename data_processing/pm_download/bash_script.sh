#python pm_download.py --path /discover/nobackup/cmbreen/aso_data/swe_tifs/colorado/ASO_50M_SWE_USCOBR_20190419.tif --time_start 2019-04-19 --filter "*_N3.125km_F18_SSMIS_E_37H_*,*_N3.125km_F18_SSMIS_E_37V_*,*_N6.25km_F18_SSMIS_E_19H_*,*_N6.25km_F18_SSMIS_E_19V_*" --output nsidc_pm_data/ASO_50M_SWE_USCOBR_20190419

#!/bin/bash
## There is a .netrc file in the discover home directory and in the local home directory so you don't have to put in the password a bunch of times 

# Directory containing the TIF files
TIF_DIR="/discover/nobackup/cmbreen/aso_data/swe_tifs"

# Output directory for downloaded data
OUTPUT_DIR="/discover/nobackup/cmbreen/passive_microwave/"

# Filter for specific channels (adjust as needed)
FILTER="*_N3.125km_F18_SSMIS_E_37H_*,*_N3.125km_F18_SSMIS_E_37V_*,*_N6.25km_F18_SSMIS_E_19H_*,*_N6.25km_F18_SSMIS_E_19V_*"

# Function to extract date from filename
extract_date() {
    local filename="$1"
    
    # Try pattern 1: YYYYMMDD (e.g., ASO_50M_SWE_USCOBR_20190419.tif)
    if [[ $filename =~ _([0-9]{8})\.tif$ ]]; then
        local date_str="${BASH_REMATCH[1]}"
        echo "${date_str:0:4}-${date_str:4:2}-${date_str:6:2}"
        return 0
    fi
    
    # Try pattern 2: YYYYMmmDD (e.g., ASO_BlueRiver_2023Apr16_swe_50m.tif)
    if [[ $filename =~ _([0-9]{4})(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)([0-9]{2})_ ]]; then
        local year="${BASH_REMATCH[1]}"
        local month="${BASH_REMATCH[2]}"
        local day="${BASH_REMATCH[3]}"
        
        # Convert month abbreviation to number
        case $month in
            Jan) month="01";;
            Feb) month="02";;
            Mar) month="03";;
            Apr) month="04";;
            May) month="05";;
            Jun) month="06";;
            Jul) month="07";;
            Aug) month="08";;
            Sep) month="09";;
            Oct) month="10";;
            Nov) month="11";;
            Dec) month="12";;
        esac
        
        echo "${year}-${month}-${day}"
        return 0
    fi
    
    # Try pattern 3: YYYYMmmDD-DD (date range, e.g., ASO_BlueRiver_2019June24-28_swe_50m.tif)
    if [[ $filename =~ _([0-9]{4})(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)([0-9]{2})-([0-9]{2})_ ]]; then
        local year="${BASH_REMATCH[1]}"
        local month="${BASH_REMATCH[2]}"
        local day="${BASH_REMATCH[3]}"
        
        # Convert month abbreviation to number
        case $month in
            Jan) month="01";;
            Feb) month="02";;
            Mar) month="03";;
            Apr) month="04";;
            May) month="05";;
            Jun) month="06";;
            Jul) month="07";;
            Aug) month="08";;
            Sep) month="09";;
            Oct) month="10";;
            Nov) month="11";;
            Dec) month="12";;
        esac
        
        echo "${year}-${month}-${day}"
        return 0
    fi
    
    return 1
}

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Counter for processed files
count=0
total=$(find "$TIF_DIR" -name "*.tif" ! -name "*.xml" | wc -l)

echo "========================================================================"
echo "ASO TIF Passive Microwave Download Script"
echo "========================================================================"
echo "TIF directory: $TIF_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Filter: $FILTER"
echo "Total TIF files found: $total"
echo "========================================================================"
echo ""

# Loop through all .tif files (excluding .xml files)
for tif_file in "$TIF_DIR"/*.tif; do
    # Skip if no files found
    [ -e "$tif_file" ] || continue
    
    # Skip .xml files
    [[ "$tif_file" == *.xml ]] && continue
    
    # Get just the filename
    filename=$(basename "$tif_file")
    
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
    echo "Extracted date: $date"
    echo "------------------------------------------------------------------------"
    
    # Run the download script
    python download.py \
        --path "$tif_file" \
        --time_start "$date" \
        --filter "$FILTER" \
        --output "$OUTPUT_DIR"
    
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
echo "Data downloaded to: $OUTPUT_DIR"
echo "========================================================================"