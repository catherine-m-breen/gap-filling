#!/usr/bin/env python3
"""
VIIRS Snow Cover Batch Download Script
Downloads VNP10A1F data from NASA NSIDC DAAC
"""

import os
import sys
from datetime import datetime, timedelta
import requests
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

EARTHDATA_USER = "your_username_here"
EARTHDATA_PASS = "your_password_here"

PRODUCT = "VNP10A1F"  # VNP10A1F (375m) or VNP10A1 (1km)
START_DATE = "2023-01-01"
END_DATE = "2023-01-31"
TILES = []  # Example: ["h09v04", "h10v04"] or leave empty for all
OUTPUT_DIR = "./viirs_snow_data"

# ============================================================================
# FUNCTIONS
# ============================================================================

def setup_session():
    """Create authenticated session for NASA Earthdata"""
    session = requests.Session()
    session.auth = (EARTHDATA_USER, EARTHDATA_PASS)
    return session

def search_granules(product, start_date, end_date, tiles=None):
    """Search for granules using CMR API"""
    
    cmr_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
    
    params = {
        'short_name': product,
        'temporal': f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
        'page_size': 2000
    }
    
    print(f"Searching for {product} granules...")
    print(f"Date range: {start_date} to {end_date}")
    
    response = requests.get(cmr_url, params=params)
    response.raise_for_status()
    
    data = response.json()
    granules = data.get('feed', {}).get('entry', [])
    
    # Extract download URLs
    urls = []
    for granule in granules:
        for link in granule.get('links', []):
            if link.get('rel') == 'http://esipfed.org/ns/fedsearch/1.1/data#':
                url = link.get('href')
                
                # Filter by tiles if specified
                if tiles:
                    if any(tile in url for tile in tiles):
                        urls.append(url)
                else:
                    urls.append(url)
    
    print(f"Found {len(urls)} files")
    return urls

def download_file(session, url, output_dir):
    """Download a single file with progress"""
    
    filename = url.split('/')[-1]
    output_path = Path(output_dir) / filename
    
    # Skip if already downloaded
    if output_path.exists():
        print(f"Skipping (already exists): {filename}")
        return True
    
    print(f"Downloading: {filename}")
    
    try:
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Simple progress indicator
                    percent = (downloaded / total_size) * 100
                    print(f"  Progress: {percent:.1f}%", end='\r')
        
        print(f"  Complete: {filename}")
        return True
        
    except Exception as e:
        print(f"  ERROR downloading {filename}: {e}")
        if output_path.exists():
            output_path.unlink()  # Remove partial file
        return False

def main():
    """Main execution"""
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Setup authentication
    print("Setting up authentication...")
    session = setup_session()
    
    # Search for files
    urls = search_granules(PRODUCT, START_DATE, END_DATE, TILES if TILES else None)
    
    if not urls:
        print("No files found! Check your search criteria.")
        sys.exit(1)
    
    # Download files
    print(f"\nStarting download of {len(urls)} files...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    successful = 0
    failed = 0
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}]")
        if download_file(session, url, OUTPUT_DIR):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary:")
    print(f"  Total files: {len(urls)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()