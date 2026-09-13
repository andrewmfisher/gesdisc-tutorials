# Main script for benchmarking data access methods
#
# 8/31/2026 JRS
# Methods are defined in separate files

import earthaccess
import xarray as xr
from access_with_kerchunk import get_vds
from access_with_opendap import get_opendap
from access_with_GiC import get_GiC
from pydap.net import create_session
import opendap_input_params()

def main(short_name, version, start_date, end_date, lat_min, lat_max, lon_min, lon_max):

    #passing user input opendap data variables into main from the opendap_input_params function
    print("Starting main script for benchmarking different data access methods for {short_name} version {version} from" \
    "{start_date} to {end_date} for a bounding box of ({lat_min}, {lat_max}, {lon_min}, {lon_max})"
    .format(short_name=short_name, version=version, start_date=start_date, end_date=end_date, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max))

if __name__ == "__main__":
    
    # Authenticate with earthaccess

    auth = earthaccess.login()

    #setting and calling variables for opendap data access (from user input) passed from the opendap_input_params function
    short_name, version, start_date, end_date, lat_min, lat_max, lon_min, lon_max = opendap_input_params() #set variables
    main(short_name, version, start_date, end_date, lat_min, lat_max, lon_min, lon_max) #calling variables from def(main)
    # Load functions from separate function files

    # Call each to get data for the domain of interest

    # kerchunk ----------------------------------------------------------

    # Read in kerchunk files of IMERG data (one year each) and concatenate them
    # Refine this so it can load multiple years with one function...
    vds2010 = get_vds("https://data.gesdisc.earthdata.nasa.gov/browse/kerchunk/GPM_L3/GPM_3IMERGDF.07/2010.parq", auth = auth)
    vds2011 = get_vds("https://data.gesdisc.earthdata.nasa.gov/browse/kerchunk/GPM_L3/GPM_3IMERGDF.07/2011.parq", auth = auth)

    vds_concat = xr.concat([vds2010, vds2011], dim="time")
    vds_concat.attrs["BeginDate"] = str(vds_concat.time.min().dt.date.values)
    vds_concat.attrs["EndDate"] = str(vds_concat.time.max().dt.date.values)
    print(vds_concat)

    # opendap -----------------------------------------------------------

    # Cloud Giovanni time series ----------------------------------------

    data = get_GiC()

    # Track computational demands for each method

    # Plot method vs. computational requirements


