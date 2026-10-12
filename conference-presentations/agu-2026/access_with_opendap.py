"""
Stream data from Cloud opendap with pydap

Date: 8/31/2026
Author: Jacob Schaperow
Last Update: JRS, Oct. 12, 2026
Environment: nasa-gesdisc-opendap
"""
import xarray as xr
import earthaccess
# from pydap.net import create_session

# Create virtual dataset loader function (based on Chris B's How To)
def get_opendap(session, shortname, version, BeginDate, EndDate, bbox, subvar):
    """
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

    #authentication using earthaccess
    # auth = earthaccess.login()
    # token = earthaccess.get_edl_token()['access_token']
    # my_session = create_session(session_kwargs={"token": token})

    # May want to use open_dataset instead of open_mfdataset to avoid overloading opendap servers?
    # Another idea is to use multiple threads like Andrey does to get data faster?

    try:
        # Load dataset object and metadata, but don't open the values yet
        ds = xr.open_mfdataset(opendap_urls, engine="netcdf4", coords = "minimal", compat = "override", combine = "nested", concat_dim = "time")
        # ds = xr.open_mfdataset(opendap_urls, engine="pydap", session=session)
    except OSError as e:
        print('Error', e)
        print('Please check that your .edl_token file exists and is valid, or that your username/password were entered correctly.')
        raise

    # Subset the dataset based on lat/lon bounds, which is performed server-side
    ds_sub = ds.sel(
        lon = slice(bbox[0], bbox[2]),
        lat = slice(bbox[1], bbox[3])
    )

    return ds_sub
