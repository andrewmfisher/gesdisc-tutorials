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

def opendap_input_params():
    #Prompts for user input for data access parameters with openday in the get_opendap function (called by main.py)
    short_name = input("Enter the short name of the dataset (e.g., GPM_3IMERGDF): ")
    version = input("Enter the version of the dataset (e.g., 07): ")
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    lat_min = float(input("Enter the minimum latitude in DD: "))
    lat_max = float(input("Enter the maximum latitude in DD: "))
    lon_min = float(input("Enter the minimum longitude in DD: "))
    lon_max = float(input("Enter the maximum longitude in DD: "))

    return short_name, version, start_date, end_date, lat_min, lat_max, lon_min, lon_max


# Create virtual dataset loader function (from Chris B's How To)
def get_opendap(short_name, version, start_date, end_date, lat_min, lat_max, lon_min, lon_max):
    """
    Args:
        inputs:

    """

    # Create search query for 1980-01-01 Cloud OPeNDAP URL
    results = earthaccess.search_data(
        short_name = str(short_name),
        version=str(version),
        temporal=(start_date, end_date), # This will stream one granule, but can be edited for a longer temporal extent
        bounding_box=(lon_min, lat_min, lon_max, lat_max)
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
    lat_min, lat_max = 29.78140, 30.29064  # Latitude bounds
    lon_min, lon_max = -99.85886, -98.91769  # Longitude bounds

    # Subset the dataset based on lat/lon bounds, which is performed server-side
    ds_conus = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    return ds_conus

    print("starting get_opendap")
    print("this is a placeholder function pending development")

    data = 1
