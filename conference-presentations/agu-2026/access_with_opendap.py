"""
Stream data from Cloud opendap with pydap

Date: 8/31/2026
Author: Jacob Schaperow
Last Update: JRS, Oct. 15, 2026
Environment: nasa-gesdisc-opendap
"""
from tqdm.auto import tqdm # for progress bar
import xarray as xr
import earthaccess
import concurrent.futures # for multithreading with xarray.open_dataset
import time # in order to add time delay between requests

# from pydap.net import create_session

# Function for streaming data with cloud opendap, using dask and multithreading
def get_opendap(session, shortname, version, BeginDate, EndDate, bbox, subvar, nworkers = 8):
    """
    Stream data with Cloud OPENDAP. 
    Uses earthaccess to construct a list of opendap URLs with variable subsetting
    Uses lazy loading to stream all data and then subset to bounding box
    Uses multithreading to speed up data access
    Sensitive to number of workers. 8 is OK. 50 is too many and will get you blocked.
    Args:
        inputs:
        session = opendap session (authentication cookies)
        nworkers = number of workers for multi-threaded opendap access
    """

    # Create search query for Cloud OPeNDAP URLs
    results = earthaccess.search_data(
        short_name = str(shortname),
        version=str(version),
        temporal=(BeginDate, EndDate),
        bounding_box=tuple(bbox)
    )

    # Parse out URL from request, add to OPeNDAP URLs list for querying multiple granules with constraint expressions
    opendap_urls = []
    for item in results:
        for urls in item['umm']['RelatedUrls']:  # Iterate over RelatedUrls in each request step
            if 'OPENDAP' in urls.get('Description', '').upper():  # Check if 'OPENDAP' is in the Description
                # Extract OPeNDAP URL, replace 
                url = urls['URL'].replace('https', 'dap4')

                # Subset desired variables, along with lat, lon, and time
                # To view all variables, comment out these two lines
                ce = "?dap4.ce=/{}%3B/{}%3B/{}%3B/{}".format(subvar, "lat", "lon", "time")
                url = url + ce

                # Can add lat/lon slicing here
                # need indices for each variable (lat/lon)

                # Add URL to list
                opendap_urls.append(url)

    print("created list of Opendap URLs")

    # We assume that opendap_urls are in chronological order, which they should be, since they come from earthaccess.search
    # Helper function to open the dataset (needed to pass arguments to executor.map)
    def fetch_dataset(url, max_retries=3):
        # Try to open the URL up to 'max_retries' times
        for attempt in range(max_retries):
            try:
                ds = xr.open_dataset(url, engine="pydap", session=session, chunks={})
#                 ds.coords.load() # forcing the background thread to finish downloading coordinates to make sure we trigger an exception on 404
                return ds
            except Exception as e:
                # Using tqdm.write instead of print prevents the progress bar from glitching/breaking
                tqdm.write(f"Attempt {attempt + 1} failed for {url} | Error: {e}")
                time.sleep(2) # wait before trying again
        # If it fails max_retries times, print a final message and return None
        tqdm.write(f"All {max_retries} attempts failed for {url}. File skipped.")
        return None

    ds_lst = [] # initialize ds_lst
    with concurrent.futures.ThreadPoolExecutor(max_workers=nworkers) as executor:

        # executor.map ensures the result stays in the order of opendap_urls
        results = tqdm(executor.map(fetch_dataset, opendap_urls), total = len(opendap_urls), desc = "Fetching Metadata")
        ds_lst = [ds for ds in results if ds is not None] # parse results and filter out failures

    # At this point, ds_lst is a list of lazily-loaded xarray Datasets backed by Dask arrays

    # If we have problems with out-of-order data, we can use this to sort them:
    # Sort the list of datasets chronologically based on their ACTUAL internal time
    # ds_lst.sort(key=lambda ds: ds.time.isel(time=0).values)

    # concatenate to make datacube
    if len(ds_lst) > 0:
        print("Concatenating files...")
        ds_concat = xr.concat(
            ds_lst, 
            dim='time',
            coords='minimal', 
            compat='override' 
        )

    # Spatial subsetting
    ds_sub = ds_concat.sel(
        lon=slice(bbox[0], bbox[2]),
        lat=slice(bbox[1], bbox[3])
    ) # this happens client-side (we never sent this to Hyrax)

    # Ensure datetime global attributes are set correctly
    last_dataset = ds_lst[-1] # Grabs the last file chronologically
    ds_sub.attrs["EndDate"] = last_dataset.attrs.get("EndDate")
    ds_sub.attrs["EndTime"] = last_dataset.attrs.get("EndTime")    

    return ds_sub

# Function for streaming data with cloud opendap, trying to optimize for speed
def get_opendap_2(session, shortname, version, BeginDate, EndDate, bbox, subvar):
    """
    Stream data with Cloud OPENDAP. 
    Uses earthaccess to construct a list of opendap URLs with variable subsetting
    Uses lazy loading to stream all data and then subset to bounding box
    Args:
        inputs:
        session = opendap session (authentication cookies)
    """

    # Create search query for Cloud OPeNDAP URLs
    results = earthaccess.search_data(
        short_name = str(shortname),
        version=str(version),
        temporal=(BeginDate, EndDate),
        bounding_box=tuple(bbox)
    )

    # Parse out URL from request, add to OPeNDAP URLs list for querying multiple granules with constraint expressions
    opendap_urls = []
    for item in results:
        for urls in item['umm']['RelatedUrls']:  # Iterate over RelatedUrls in each request step
            if 'OPENDAP' in urls.get('Description', '').upper():  # Check if 'OPENDAP' is in the Description
                # Extract OPeNDAP URL, replace 
                url = urls['URL'].replace('https', 'dap4')

                # Subset desired variables, along with lat, lon, and time
                # To view all variables, comment out these two lines
                ce = "?dap4.ce=/{}%3B/{}%3B/{}%3B/{}".format(subvar, "lat", "lon", "time")
                url = url + ce

                # Add URL to list
                opendap_urls.append(url)

    print("created list of Opendap URLs")

    # We assume that opendap_urls are in chronological order, which they should be, since they come from earthaccess.search
    # Helper function to open the dataset (needed to pass arguments to executor.map)
    def fetch_dataset(url, max_retries=3):
        # Try to open the URL up to 'max_retries' times
        for attempt in range(max_retries):
            try:
                return xr.open_dataset(url, engine="pydap", session=session, chunks={})
            except Exception as e:
                # USE tqdm.write() SO IT DOESN'T BREAK THE PROGRESS BAR
                tqdm.write(f"Attempt {attempt + 1} failed for {url} | Error: {e}")
                time.sleep(2) # wait before trying again
        # If it fails max_retries times, print a final message and return None
        tqdm.write(f"All {max_retries} attempts failed for {url}. File skipped.")
        return None

    # Use a simple list comprehension instead of ThreadPoolExecutor for metadata fetching
    # Because of chunks={}, this is just reading metadata, which should be relatively fast,
    # and avoids Dask/Threadpool lock fighting.
    ds_lst = []
    for url in tqdm(opendap_urls, desc="Fetching Metadata", unit="file"):
        ds = fetch_dataset(url, max_retries=3)
        if ds is not None:
            ds_lst.append(ds)

    # At this point, ds_lst is a list of lazily-loaded xarray Datasets backed by Dask arrays

    # concatenate to make datacube
    if len(ds_lst) > 0:
        print("Concatenating files...")
        ds_concat = xr.concat(
            ds_lst, 
            dim='time',
            coords='minimal', 
            compat='override',
            combine_attrs='override'
        )

    # Spatial subsetting
    ds_sub = ds_concat.sel(
        lon=slice(bbox[0], bbox[2]),
        lat=slice(bbox[1], bbox[3])
    )

    # Ensure datetime global attributes are set correctly
    last_dataset = ds_lst[-1] # Grabs the last file chronologically
    ds_sub.attrs["EndDate"] = last_dataset.attrs.get("EndDate")
    ds_sub.attrs["EndTime"] = last_dataset.attrs.get("EndTime")    

    return ds_sub
