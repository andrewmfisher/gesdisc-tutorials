"""
Stream data from Cloud opendap with pydap

Date: 8/31/2026
Author: Jacob Schaperow
Last Update: JRS, Oct. 12, 2026
Environment: nasa-gesdisc-opendap
"""
import xarray as xr
import earthaccess
import concurrent.futures # for multithreading with xarray.open_dataset

# from pydap.net import create_session

# Create virtual dataset loader function (based on Chris B's How To)
def get_opendap(session, shortname, version, BeginDate, EndDate, bbox, subvar, nworkers = 8):
    """
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

                # Add URL to list
                opendap_urls.append(url)

    print("created list of Opendap URLs")

    #authentication using earthaccess
    # auth = earthaccess.login()
    # token = earthaccess.get_edl_token()['access_token']
    # my_session = create_session(session_kwargs={"token": token})

    # May want to use open_dataset instead of open_mfdataset to avoid overloading opendap servers?
    # Another idea is to use multiple threads like Andrey does to get data faster?
    
    ds_lst = [] # initialize ds_lst
    with concurrent.futures.ThreadPoolExecutor(max_workers=nworkers) as executor:
        
        # Submit all URLs to the background workers
        futures = {
            executor.submit(xr.open_dataset, url, engine="pydap", session = session, chunks = {}): url 
            for url in opendap_urls
        }
        
        # Grab them as soon as they finish downloading
        for future in concurrent.futures.as_completed(futures):
            failed_url = futures[future]
            try:
                ds = future.result()
                ds_lst.append(ds)
            except Exception as e:
                print(f"Failed to open {failed_url} | Error: {e}") # lists failed URLs in case of access errors

    # concatenate to make datacube
    if len(ds_lst) > 0:
        print("Concatenating files...")
        ds_concat = xr.concat(
            ds_lst, 
            dim='time',
            coords='minimal', 
            compat='override' 
        )
        
        # Because the ThreadPool finishes out of order, sort it back chronologically
        ds_concat = ds_concat.sortby('time')

        # Spatial subsetting
        ds_sub = ds_concat.sel(
            lon=slice(bbox[0], bbox[2]),
            lat=slice(bbox[1], bbox[3])
        )

        # Assign correct begindate and enddate attributes (they get lost in the open_dataset logic above)
        ds_sub.attrs["BeginDate"] = str(ds_sub.time.min().dt.date.values)
        ds_sub.attrs["EndDate"] = str(ds_sub.time.max().dt.date.values)

        return ds_sub
