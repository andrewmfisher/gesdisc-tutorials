"""
Stream data from Cloud opendap with pydap

Date: 8/31/2026
Author: Jacob Schaperow
Last Update: ALM, Sept 13, 2026 @ 11:20AM
Environment: nasa-gesdisc-opendap
"""
import xarray as xr
import earthaccess
from pydap.net import create_session


# Create virtual dataset loader function (from Chris B's How To)
def get_opendap(ShortName, Version, BeginDate, EndDate, LatMin, LatMax, LonMin, LonMax):
    """
    Args:
        inputs:

    """

    # Create search query for 1980-01-01 Cloud OPeNDAP URL
    results = earthaccess.search_data(
        short_name = str(ShortName),
        version=str(Version),
        temporal=(BeginDate, EndDate), # This will stream one granule, but can be edited for a longer temporal extent
        bounding_box=(LonMin, LatMin, LonMax, LatMax)
    )

    # Parse out URL from request, add to OPeNDAP URLs list for querying multiple granules with constraint expressions
    opendap_urls = []
    for item in results:
        for urls in item['umm']['RelatedUrls']:  # Iterate over RelatedUrls in each request step
            if 'OPENDAP' in urls.get('Description', '').upper():  # Check if 'OPENDAP' is in the Description
                # Extract OPeNDAP URL, replace 
                url = urls['URL'].replace('https', 'dap4')

                # Subset T2M, lat, lon, and time
                # To view all variables, comment out these two lines
                ce = "?dap4.ce=/{}%3B/{}%3B/{}%3B/{}".format("precipitation", "lat", "lon", "time")
                url = url + ce

                # Add URL to list
                opendap_urls.append(url)


    #authentication using earthaccess
    auth = earthaccess.login()
    token = earthaccess.get_edl_token()['access_token']
    my_session = create_session(session_kwargs={"token": token})

    try:
        # Load dataset object and metadata, but don't open the values yet
        # NOTE: When opening HDF files, the group to be accessed must be specified with the "group=" parameter. 
        #       E.g., for GPM IMERG, group="Grid" must be entered or an error will occur
        # Remove the session parameter if you are just using a .netrc file to authenticate
        ds = xr.open_mfdataset(opendap_urls, engine="pydap", session=my_session)
    except OSError as e:
        print('Error', e)
        print('Please check that your .edl_token file exists and is valid, or that your username/password were entered correctly.')
        raise

    # Define latitude and longitude bounds for CONUS
    #LatMin, LatMax = 29.78140, 30.29064  # Latitude bounds
    #LonMin, LonMax = -99.85886, -98.91769  # Longitude bounds

    # Subset the dataset based on lat/lon bounds, which is performed server-side
    ds_conus = ds.sel(lat=slice(LatMin, LatMax), lon=slice(LonMin, LonMax))

    return ds_conus

    print("starting get_opendap")
    print("this is a placeholder function pending development")

    data = 1
