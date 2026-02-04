#!/usr/bin/env python
# ----------------------------------------------------------------------------
# NSIDC Data Download Script - Command-line Version
# ----------------------------------------------------------------------------


###### example download: 
###### python download.py --path /Users/cmbreen/Documents/snow/gap_filling/processed/ndsi_regridded_to_aso.tif --time_start 2019-03-17 --filter "*N3.125km*_37H_*,*N3.125km*_37V_*,*N6.25km*_19H_*,*N6.25km*_19V_*"
###### python download.py --path /discover/nobackup/cmbreen/aso_data/swe_tifs/colorado --time_start 2019-03-17 --filter "*N3.125km*_37H_*,*N3.125km*_37V_*,*N6.25km*_19H_*,*N6.25km*_19V_*"

#python pm_download.py --path /discover/nobackup/cmbreen/aso_data/swe_tifs/colorado/ASO_50M_SWE_USCOBR_20190419.tif --time_start 2019-04-19 --filter "*N3.125km*_37H_*,*N3.125km*_37V_*,*N6.25km*_19H_*,*N6.25km*_19V_*" --output nsidc_pm_data//ASO_50M_SWE_USCOBR_20190419

#python pm_download.py --path /discover/nobackup/cmbreen/aso_data/swe_tifs/colorado/ASO_50M_SWE_USCOBR_20190419.tif --time_start 2019-04-19 --filter "*_N3.125km_F18_SSMIS_E_37H_*,*_N3.125km_F18_SSMIS_E_37V_*,*_N6.25km_F18_SSMIS_E_19H_*,*_N6.25km_F18_SSMIS_E_19V_*" --output nsidc_pm_data/ASO_50M_SWE_USCOBR_20190419

from __future__ import print_function

import argparse
import base64
import itertools
import json
import math
import netrc
import os
import os.path
import ssl
import sys
import time
from datetime import datetime, timedelta
from getpass import getpass

try:
    from urllib.parse import urlparse
    from urllib.request import urlopen, Request, build_opener, HTTPCookieProcessor
    from urllib.error import HTTPError, URLError
except ImportError:
    from urlparse import urlparse
    from urllib2 import (
        urlopen,
        Request,
        HTTPError,
        URLError,
        build_opener,
        HTTPCookieProcessor,
    )

# Constants
CMR_URL = "https://cmr.earthdata.nasa.gov"
URS_URL = "https://urs.earthdata.nasa.gov"
CMR_PAGE_SIZE = 2000
CMR_FILE_URL = (
    "{0}/search/granules.json?"
    "&sort_key[]=start_date&sort_key[]=producer_granule_id"
    "&page_size={1}".format(CMR_URL, CMR_PAGE_SIZE)
)
CMR_COLLECTIONS_URL = "{0}/search/collections.json?".format(CMR_URL)
FILE_DOWNLOAD_MAX_RETRIES = 3


def get_bounding_box_from_raster(raster_path):
    """
    Extract bounding box from raster file (GeoTIFF, NetCDF, etc.).
    
    Parameters
    ----------
    raster_path : str
        Path to raster file (.tif, .tiff, .nc, etc.)
    
    Returns
    -------
    str
        Bounding box as "lon_min,lat_min,lon_max,lat_max"
    """
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError:
        print("Error: rasterio is required to read raster files.")
        print("Install with: pip install rasterio")
        sys.exit(1)
    
    try:
        with rasterio.open(raster_path) as src:
            # Get bounds in the file's CRS
            bounds = src.bounds  # (left, bottom, right, top)
            src_crs = src.crs
            
            print("Original CRS: {}".format(src_crs))
            print("Original bounds: {}".format(bounds))
            
            # Transform bounds to WGS84 (EPSG:4326)
            if src_crs is None:
                print("Warning: Raster has no CRS. Assuming WGS84 (EPSG:4326)")
                bounds_wgs84 = bounds
            elif src_crs.to_epsg() == 4326:
                print("Raster is already in WGS84")
                bounds_wgs84 = bounds
            else:
                print("Reprojecting bounds to WGS84 (EPSG:4326)")
                bounds_wgs84 = transform_bounds(
                    src_crs, 
                    'EPSG:4326', 
                    bounds.left, 
                    bounds.bottom, 
                    bounds.right, 
                    bounds.top
                )
            
            # Format as "lon_min,lat_min,lon_max,lat_max"
            bounding_box = "{:.6f},{:.6f},{:.6f},{:.6f}".format(
                bounds_wgs84[0],  # left (lon_min)
                bounds_wgs84[1],  # bottom (lat_min)
                bounds_wgs84[2],  # right (lon_max)
                bounds_wgs84[3]   # top (lat_max)
            )
            
            print("Extracted bounding box from raster: {}".format(bounding_box))
            return bounding_box
            
    except Exception as e:
        print("Error reading raster file: {}".format(str(e)))
        sys.exit(1)


def get_bounding_box_from_shapefile(shapefile_path):
    """
    Extract bounding box from shapefile.
    
    Parameters
    ----------
    shapefile_path : str
        Path to shapefile (.shp)
    
    Returns
    -------
    str
        Bounding box as "lon_min,lat_min,lon_max,lat_max"
    """
    try:
        import geopandas as gpd
    except ImportError:
        print("Error: geopandas is required to read shapefiles.")
        print("Install with: pip install geopandas")
        sys.exit(1)
    
    try:
        gdf = gpd.read_file(shapefile_path)
        
        # Ensure CRS is WGS84 (EPSG:4326)
        if gdf.crs is None:
            print("Warning: Shapefile has no CRS. Assuming WGS84 (EPSG:4326)")
        elif gdf.crs.to_epsg() != 4326:
            print("Reprojecting shapefile from {} to WGS84 (EPSG:4326)".format(gdf.crs))
            gdf = gdf.to_crs(epsg=4326)
        
        # Get bounds
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        bounding_box = "{:.6f},{:.6f},{:.6f},{:.6f}".format(
            bounds[0], bounds[1], bounds[2], bounds[3]
        )
        
        print("Extracted bounding box from shapefile: {}".format(bounding_box))
        return bounding_box
        
    except Exception as e:
        print("Error reading shapefile: {}".format(str(e)))
        sys.exit(1)


def get_bounding_box_from_file(file_path):
    """
    Extract bounding box from either raster or vector file.
    
    Parameters
    ----------
    file_path : str
        Path to spatial file (.tif, .shp, .nc, etc.)
    
    Returns
    -------
    str
        Bounding box as "lon_min,lat_min,lon_max,lat_max"
    """
    # Determine file type from extension
    _, ext = os.path.splitext(file_path.lower())
    
    raster_extensions = ['.tif', '.tiff', '.nc', '.hdf', '.h5', '.img', '.vrt']
    vector_extensions = ['.shp', '.geojson', '.gpkg', '.kml', '.json']
    
    if ext in raster_extensions:
        return get_bounding_box_from_raster(file_path)
    elif ext in vector_extensions:
        return get_bounding_box_from_shapefile(file_path)
    else:
        # Try raster first, then vector
        print("Unknown file extension '{}'. Trying raster format...".format(ext))
        try:
            return get_bounding_box_from_raster(file_path)
        except:
            print("Failed as raster. Trying vector format...")
            return get_bounding_box_from_shapefile(file_path)


def parse_date(date_str, end_of_day=False):
    """
    Parse date string to ISO format required by CMR.
    
    Parameters
    ----------
    date_str : str
        Date in format YYYY-MM-DD or YYYY-MM-DD HH:MM:SS
    end_of_day : bool
        If True and no time specified, set to 23:59:59. Otherwise 00:00:00
    
    Returns
    -------
    str
        ISO formatted datetime string (e.g., '2019-03-18T00:00:00Z')
    """
    try:
        # Try parsing with time
        if 'T' in date_str or ' ' in date_str:
            # Already has time component
            date_str = date_str.replace(' ', 'T')
            if not date_str.endswith('Z'):
                date_str += 'Z'
            return date_str
        else:
            # Only date provided
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if end_of_day:
                dt = dt.replace(hour=23, minute=59, second=59)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError as e:
        print("Error parsing date '{}': {}".format(date_str, str(e)))
        print("Expected format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ")
        sys.exit(1)


def get_username():
    username = ""
    try:
        do_input = raw_input  # noqa
    except NameError:
        do_input = input
    username = do_input("Earthdata username (or press Return to use a bearer token): ")
    return username


def get_password():
    password = ""
    while not password:
        password = getpass("password: ")
    return password


def get_token():
    token = ""
    while not token:
        token = getpass("bearer token: ")
    return token


def get_login_credentials():
    """Get user credentials from .netrc or prompt for input."""
    credentials = None
    token = None

    try:
        info = netrc.netrc()
        username, _account, password = info.authenticators(urlparse(URS_URL).hostname)
        if username == "token":
            token = password
        else:
            credentials = "{0}:{1}".format(username, password)
            credentials = base64.b64encode(credentials.encode("ascii")).decode("ascii")
    except Exception:
        username = None
        password = None

    if not username:
        username = get_username()
        if len(username):
            password = get_password()
            credentials = "{0}:{1}".format(username, password)
            credentials = base64.b64encode(credentials.encode("ascii")).decode("ascii")
        else:
            token = get_token()

    return credentials, token


def build_version_query_params(version):
    desired_pad_length = 3
    if len(version) > desired_pad_length:
        print('Version string too long: "{0}"'.format(version))
        quit()

    version = str(int(version))
    query_params = ""

    while len(version) <= desired_pad_length:
        padded_version = version.zfill(desired_pad_length)
        query_params += "&version={0}".format(padded_version)
        desired_pad_length -= 1
    return query_params


def filter_add_wildcards(filter):
    if not filter.startswith("*"):
        filter = "*" + filter
    if not filter.endswith("*"):
        filter = filter + "*"
    return filter


def build_filename_filter(filename_filter):
    filters = filename_filter.split(",")
    result = "&options[producer_granule_id][pattern]=true"
    for filter in filters:
        result += "&producer_granule_id[]=" + filter_add_wildcards(filter)
    return result


def build_query_params_str(
    short_name,
    version,
    time_start="",
    time_end="",
    bounding_box=None,
    polygon=None,
    filename_filter=None,
    provider=None,
):
    """Create the query params string for the given inputs."""
    params = "&short_name={0}".format(short_name)
    params += build_version_query_params(version)
    if time_start or time_end:
        params += "&temporal[]={0},{1}".format(time_start, time_end)
    if polygon:
        params += "&polygon={0}".format(polygon)
    elif bounding_box:
        params += "&bounding_box={0}".format(bounding_box)
    if filename_filter:
        params += build_filename_filter(filename_filter)
    if provider:
        params += "&provider={0}".format(provider)

    return params


def build_cmr_query_url(
    short_name,
    version,
    time_start,
    time_end,
    bounding_box=None,
    polygon=None,
    filename_filter=None,
    provider=None,
):
    params = build_query_params_str(
        short_name=short_name,
        version=version,
        time_start=time_start,
        time_end=time_end,
        bounding_box=bounding_box,
        polygon=polygon,
        filename_filter=filename_filter,
        provider=provider,
    )

    return CMR_FILE_URL + params


def get_speed(time_elapsed, chunk_size):
    if time_elapsed <= 0:
        return ""
    speed = chunk_size / time_elapsed
    if speed <= 0:
        speed = 1
    size_name = ("", "k", "M", "G", "T", "P", "E", "Z", "Y")
    i = int(math.floor(math.log(speed, 1000)))
    p = math.pow(1000, i)
    return "{0:.1f}{1}B/s".format(speed / p, size_name[i])


def output_progress(count, total, status="", bar_len=60):
    if total <= 0:
        return
    fraction = min(max(count / float(total), 0), 1)
    filled_len = int(round(bar_len * fraction))
    percents = int(round(100.0 * fraction))
    bar = "=" * filled_len + " " * (bar_len - filled_len)
    fmt = "  [{0}] {1:3d}%  {2}   ".format(bar, percents, status)
    print("\b" * (len(fmt) + 4), end="")
    sys.stdout.write(fmt)
    sys.stdout.flush()


def cmr_read_in_chunks(file_object, chunk_size=1024 * 1024):
    """Read a file in chunks using a generator. Default chunk size: 1Mb."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data


def get_login_response(url, credentials, token):
    opener = build_opener(HTTPCookieProcessor())

    req = Request(url)
    if token:
        req.add_header("Authorization", "Bearer {0}".format(token))
    elif credentials:
        try:
            response = opener.open(req)
            url = response.url
        except HTTPError:
            pass
        except Exception as e:
            print("Error{0}: {1}".format(type(e), str(e)))
            sys.exit(1)

        req = Request(url)
        req.add_header("Authorization", "Basic {0}".format(credentials))

    try:
        response = opener.open(req)
    except HTTPError as e:
        err = "HTTP error {0}, {1}".format(e.code, e.reason)
        if "Unauthorized" in e.reason:
            if token:
                err += ": Check your bearer token"
            else:
                err += ": Check your username and password"
            print(err)
            sys.exit(1)
        raise
    except Exception as e:
        print("Error{0}: {1}".format(type(e), str(e)))
        sys.exit(1)

    return response

def cmr_download(urls, download_dir, force=False, quiet=False):
    """Download files from list of urls."""
    if not urls:
        return

    # Convert to absolute path
    download_dir = os.path.abspath(download_dir)
    
    # Create download directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        if not quiet:
            print("Created directory: {0}".format(download_dir))

    if not quiet:
        print("Downloading to: {0}".format(download_dir))

    url_count = len(urls)
    if not quiet:
        print("Downloading {0} files...".format(url_count))
    credentials = None
    token = None

    for index, url in enumerate(urls, start=1):
        if not credentials and not token:
            p = urlparse(url)
            if p.scheme == "https":
                credentials, token = get_login_credentials()

        filename = url.split("/")[-1]
        filepath = os.path.join(download_dir, filename)
        
        if not quiet:
            print(
                "{0}/{1}: {2}".format(
                    str(index).zfill(len(str(url_count))), url_count, filename
                )
            )

        for download_attempt_number in range(1, FILE_DOWNLOAD_MAX_RETRIES + 1):
            if not quiet and download_attempt_number > 1:
                print("Retrying download of {0}".format(url))
            try:
                response = get_login_response(url, credentials, token)
                length = int(response.headers["content-length"])
                try:
                    if not force and os.path.exists(filepath) and length == os.path.getsize(filepath):
                        if not quiet:
                            print("  File exists, skipping")
                        break
                except OSError:
                    pass
                count = 0
                chunk_size = min(max(length, 1), 1024 * 1024)
                max_chunks = int(math.ceil(length / chunk_size))
                time_initial = time.time()
                with open(filepath, "wb") as out_file:
                    for data in cmr_read_in_chunks(response, chunk_size=chunk_size):
                        out_file.write(data)
                        if not quiet:
                            count = count + 1
                            time_elapsed = time.time() - time_initial
                            download_speed = get_speed(time_elapsed, count * chunk_size)
                            output_progress(count, max_chunks, status=download_speed)
                if not quiet:
                    print()
                break
            except HTTPError as e:
                print("HTTP error {0}, {1}".format(e.code, e.reason))
            except URLError as e:
                print("URL error: {0}".format(e.reason))
            except IOError as e:
                print("IO error: {0}".format(str(e)))
                raise

            if download_attempt_number == FILE_DOWNLOAD_MAX_RETRIES:
                print("failed to download file {0}.".format(filename))
                sys.exit(1)
def cmr_filter_urls(search_results):
    """Select only the desired data files from CMR response."""
    if "feed" not in search_results or "entry" not in search_results["feed"]:
        return []

    entries = [e["links"] for e in search_results["feed"]["entry"] if "links" in e]
    links = list(itertools.chain(*entries))

    urls = []
    unique_filenames = set()
    for link in links:
        if "href" not in link:
            continue
        if "inherited" in link and link["inherited"] is True:
            continue
        if "rel" in link and "data#" not in link["rel"]:
            continue

        if "title" in link and "opendap" in link["title"].lower():
            continue

        filename = link["href"].split("/")[-1]

        if "metadata#" in link["rel"] and filename.endswith(".dmrpp"):
            continue

        if "metadata#" in link["rel"] and filename == "s3credentials":
            continue

        if filename in unique_filenames:
            continue
        unique_filenames.add(filename)

        urls.append(link["href"])

    return urls


def check_provider_for_collection(short_name, version, provider):
    """Return `True` if the collection is available for the given provider, otherwise `False`."""
    query_params = build_query_params_str(
        short_name=short_name, version=version, provider=provider
    )
    cmr_query_url = CMR_COLLECTIONS_URL + query_params

    req = Request(cmr_query_url)
    try:
        response = urlopen(req)
    except Exception as e:
        print("Error: " + str(e))
        sys.exit(1)

    search_page = response.read()
    search_page = json.loads(search_page.decode("utf-8"))

    if "feed" not in search_page or "entry" not in search_page["feed"]:
        return False

    if len(search_page["feed"]["entry"]) > 0:
        return True
    else:
        return False


def get_provider_for_collection(short_name, version):
    """Return the provider for the collection associated with the given short_name and version."""
    cloud_provider = "NSIDC_CPRD"
    in_earthdata_cloud = check_provider_for_collection(
        short_name, version, cloud_provider
    )
    if in_earthdata_cloud:
        return cloud_provider

    ecs_provider = "NSIDC_ECS"
    in_ecs = check_provider_for_collection(short_name, version, ecs_provider)
    if in_ecs:
        return ecs_provider

    raise RuntimeError(
        "Found no collection matching the given short_name ({0}) and version ({1})".format(
            short_name, version
        )
    )


def cmr_search(
    short_name,
    version,
    time_start,
    time_end,
    bounding_box="",
    polygon="",
    filename_filter="",
    quiet=False,
):
    """Perform a scrolling CMR query for files matching input criteria."""
    provider = get_provider_for_collection(short_name=short_name, version=version)
    cmr_query_url = build_cmr_query_url(
        provider=provider,
        short_name=short_name,
        version=version,
        time_start=time_start,
        time_end=time_end,
        bounding_box=bounding_box,
        polygon=polygon,
        filename_filter=filename_filter,
    )
    if not quiet:
        print("Querying for data:\n\t{0}\n".format(cmr_query_url))

    cmr_paging_header = "cmr-search-after"
    cmr_page_id = None
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    urls = []
    hits = 0
    while True:
        req = Request(cmr_query_url)
        if cmr_page_id:
            req.add_header(cmr_paging_header, cmr_page_id)
        try:
            response = urlopen(req, context=ctx)
        except Exception as e:
            print("Error: " + str(e))
            sys.exit(1)

        headers = {k.lower(): v for k, v in dict(response.info()).items()}
        if not cmr_page_id:
            hits = int(headers["cmr-hits"])
            if not quiet:
                if hits > 0:
                    print("Found {0} matches.".format(hits))
                else:
                    print("Found no matches.")

        cmr_page_id = headers.get(cmr_paging_header)

        search_page = response.read()
        search_page = json.loads(search_page.decode("utf-8"))
        url_scroll_results = cmr_filter_urls(search_page)
        if not url_scroll_results:
            break
        if not quiet and hits > CMR_PAGE_SIZE:
            print(".", end="")
            sys.stdout.flush()
        urls += url_scroll_results

    if not quiet and hits > CMR_PAGE_SIZE:
        print()
    return urls


def main():
    parser = argparse.ArgumentParser(
        description='Download NSIDC data from NASA Earthdata',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download using GeoTIFF extent
  %(prog)s --path /path/to/raster.tif --time_start 2019-03-17 --time_end 2019-03-18
  
  # Download using shapefile for bounding box
  %(prog)s --path /path/to/region.shp --time_start 2019-03-17 --time_end 2019-03-18
  
  # Download using manual bounding box
  %(prog)s --bbox -122.77,35.08,-116.81,40.43 --time_start 2019-03-17 --time_end 2019-03-18
  
  # Download single day (automatically adds 23:59:59 to end time)
  %(prog)s --path region.tif --time_start 2019-03-17
  
  # Download with specific output directory
  %(prog)s --path region.tif --time_start 2019-03-17 --output /data/microwave
  
  # Download with filename filter
  %(prog)s --path region.tif --time_start 2019-03-17 --filter "*3.125km*"
        """
    )
    
    # Spatial extent (either path to file or bbox)
    spatial_group = parser.add_mutually_exclusive_group(required=True)
    spatial_group.add_argument(
        '--path',
        type=str,
        help='Path to raster (.tif, .nc) or vector (.shp, .geojson) file defining the region of interest'
    )
    spatial_group.add_argument(
        '--bbox',
        type=str,
        help='Bounding box as "lon_min,lat_min,lon_max,lat_max" (e.g., "-122.77,35.08,-116.81,40.43")'
    )
    
    # Time range
    parser.add_argument(
        '--time_start',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)'
    )
    parser.add_argument(
        '--time_end',
        type=str,
        default=None,
        help='End date (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ). If not provided, uses time_start + 23:59:59'
    )
    
    # Dataset parameters
    parser.add_argument(
        '--short_name',
        type=str,
        default='NSIDC-0630',
        help='NSIDC dataset short name (default: NSIDC-0630)'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='2',
        help='Dataset version (default: 2)'
    )
    
    # File filtering
    parser.add_argument(
        '--filter',
        type=str,
        default='',
        help='Comma-separated filename filter patterns (e.g., "*3.125km*37*,*3.125km*19*")'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default='./nsidc_pm_data',
        help='Output directory for downloaded files (default: ./nsidc_pm_data)'
    )
    
    # Download options
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download of existing files'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Get bounding box
    if args.path:
        bounding_box = get_bounding_box_from_file(args.path)
    else:
        bounding_box = args.bbox
    
    # Parse dates
    time_start = parse_date(args.time_start, end_of_day=False)
    
    if args.time_end:
        time_end = parse_date(args.time_end, end_of_day=True)
    else:
        # If no end time provided, use same day but 23:59:59
        time_end = parse_date(args.time_start, end_of_day=True)
    
    print("\n" + "="*70)
    print("NSIDC Data Download")
    print("="*70)
    print("Dataset: {} (version {})".format(args.short_name, args.version))
    print("Time range: {} to {}".format(time_start, time_end))
    print("Bounding box: {}".format(bounding_box))
    if args.filter:
        print("Filename filter: {}".format(args.filter))
    print("Output directory: {}".format(args.output))
    print("="*70 + "\n")
    
    try:
        # Search for matching files
        url_list = cmr_search(
            short_name=args.short_name,
            version=args.version,
            time_start=time_start,
            time_end=time_end,
            bounding_box=bounding_box,
            filename_filter=args.filter,
            quiet=args.quiet,
        )
        
        if not url_list:
            print("\nNo files found matching the search criteria.")
            sys.exit(0)
        
        # Download files
        cmr_download(
            url_list,
            download_dir=args.output,
            force=args.force,
            quiet=args.quiet
        )
        
        print("\n" + "="*70)
        print("Download complete! {} files downloaded to {}".format(
            len(url_list), args.output
        ))
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()



# # Single day download
# python download.py --path /Users/cmbreen/Documents/snow/gap_filling/processed/ndsi_regridded_to_aso.tif --time_start 2019-03-17

# # Date range
# python download.py --path /Users/cmbreen/Documents/snow/gap_filling/processed/ndsi_regridded_to_aso.tif --time_start 2019-03-17 --time_end 2019-03-20

# # With output directory
# python download.py --path /Users/cmbreen/Documents/snow/gap_filling/processed/ndsi_regridded_to_aso.tif --time_start 2019-03-17 --output /Volumes/MyBook/pm_data

# # With filename filter
# python download.py --path /Users/cmbreen/Documents/snow/gap_filling/processed/ndsi_regridded_to_aso.tif --time_start 2019-03-17 --filter "*3.125km*37*,*3.125km*19*"