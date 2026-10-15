# Main script for benchmarking data access methods
#
# 8/31/2026 JRS
# 10/9/2026 ALM
# Methods are defined in separate files

import access_with_kerchunk as vds
import access_with_opendap as opendap
import earthaccess
from datetime import datetime
from pydap.net import create_session

# from datetime import datetime

#from access_with_kerchunk import get_vds
#from access_with_opendap import get_opendap

# import earthaccess
# import xarray as xr
#from access_with_GiC import get_GiC
# from pydap.net import create_session

# inputs: subsetting parameters, spatiotemporal domain
# outputs: data cube with the data we desired

def main():

    # input parameters (will stay fixed)
    shortname = "GPM_3IMERGDF"
    Version = "07"
    
    LatMin = 24
    LatMax = 34
    LonMin = -103
    LonMax = -93
    bbox = [LonMin, LatMin, LonMax, LatMax]

    begin_dt = datetime.strptime("2024-01-01", "%Y-%m-%d") # note: with longer time periods, opendap hangs in current configuration
    end_dt = datetime.strptime("2024-12-31", "%Y-%m-%d")
    beginyear = begin_dt.year
    endyear = end_dt.year

    #function start statement
    print(
        "Starting main script for benchmarking different data access methods for {} version {} from" "{} to {} for a bounding box of ({}, {}, {}, {})".format(
        shortname, Version, begin_dt, end_dt, LatMin, LatMax, LonMin, LonMax)
        )

    # get bearer token for EDL authentication (once)
    auth = earthaccess.login(strategy="netrc")
    bearer_token = auth.token["access_token"]
    # cloud opendap uses session cookies, rather than a bearer token
    opendap_session = create_session(session_kwargs = {"token":bearer_token})
    
    precip_imerg_vds = vds.get_vds_multiyear(bearer_token, shortname, Version, beginyear, endyear, bbox, subvar = "precipitation")
    print(precip_imerg_vds) # confirm that the data have been (lazily) loaded

    precip_imerg_opendap = opendap.get_opendap(opendap_session, shortname, Version, begin_dt, end_dt, bbox, subvar = "precipitation", nworkers = 8)
    print(precip_imerg_opendap) # confirm that the data have been (lazily) loaded

    # opendap -----------------------------------------------------------
    # opendap2025July = get_opendap(shortname, Version, BeginDate, EndDate, LatMin, LatMax, LonMin, LonMax)
    # print(opendap2025July)

    # precip_imerg_opendap = subset_opendap(begindate, enddate, bbox)

    # Cloud Giovanni time series ----------------------------------------

    #data = get_GiC()

    # Track computational demands for each method

    # Plot method vs. computational requirements

if __name__ == "__main__":
    main()

# 1. add time profiling
# 2. push changes
# 3. 

# Time profiling: snakeviz my_profile.prof
# python -m cProfile -o ~/Documents/Codes/gesdisc-tutorials/conference-presentations/agu-2026/main_profile.prof ~/Documents/Codes/gesdisc-tutorials/conference-presentations/agu-2026/main.py