# Main script for benchmarking data access methods
#
# 8/31/2026 JRS
# 10/9/2026 ALM
# Methods are defined in separate files

import access_with_kerchunk as vds
# import access_with_opendap as opendap
import earthaccess

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
    ShortName = "GPM_3IMERGDF"
    Version = "07"
    
    LatMin = 24
    LatMax = 34
    LonMin = -103
    LonMax = -93
    bbox = [LonMin, LatMin, LonMax, LatMax]

    beginyear = 2024
    endyear = 2025

    #function start statement
    print(
        "Starting main script for benchmarking different data access methods for {} version {} from" "{} to {} for a bounding box of ({}, {}, {}, {})".format(
        ShortName, Version, beginyear, endyear, LatMin, LatMax, LonMin, LonMax)
        )

    # get bearer token for EDL authentication (once)
    auth = earthaccess.login(strategy="netrc")
    bearer_token = auth.token["access_token"]
    
    precip_imerg_vds = vds.get_vds_multiyear(bearer_token, ShortName, Version, beginyear, endyear, bbox, subvar = "precipitation")

    print(precip_imerg_vds) # confirm that the data have been (lazily) loaded




    # opendap -----------------------------------------------------------
    # opendap2025July = get_opendap(ShortName, Version, BeginDate, EndDate, LatMin, LatMax, LonMin, LonMax)
    # print(opendap2025July)

    # precip_imerg_opendap = subset_opendap(begindate, enddate, bbox)

    # Cloud Giovanni time series ----------------------------------------

    #data = get_GiC()

    # Track computational demands for each method

    # Plot method vs. computational requirements

if __name__ == "__main__":
    main()