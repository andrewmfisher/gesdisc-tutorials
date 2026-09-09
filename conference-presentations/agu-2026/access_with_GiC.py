"""
Access data from the Giovanni in the Cloud time series service

Date: 8/31/2026
Author: 
Environment: nasa-gesdisc-placeholder
"""

import warnings
import earthaccess
import xarray as xr

# Create virtual dataset loader function (from Chris B's How To)
def get_GiC():
    """
    Args:
        inputs:

    """

    print("starting get_GiC")
    print("this is a placeholder function pending development")
    data = 1


#     Dataset: NLDAS_FORA0125_H, version 2.0

# Location:
# Latitude: 38.9517
# Longitude: -92.3341

# Date range:
# 2024-01-01T00:00:00/2024-01-10T23:00:00

# Air temperature request parameters:
# data=NLDAS_FORA0125_H_2_0_Tair
# location=[38.9517,-92.3341]
# time=2024-01-01T00:00:00/2024-01-10T23:00:00
# version=2.0

# Precipitation request parameters:
# data=NLDAS_FORA0125_H_2_0_Rainf
# location=[38.9517,-92.3341]
# time=2024-01-01T00:00:00/2024-01-10T23:00:00
# version=2.0

# Works: https://api.giovanni.earthdata.nasa.gov/timeseries?data=NLDAS_FORA0125_H_2_0_Rainf&location=[38.99,-76.85]&time=2000-06-01T00:00:00/2000-06-01T07:30:00&version=2.0
# Works: https://api.giovanni.earthdata.nasa.gov/timeseries?data=NLDAS_FORA0125_H_2_0_Rainf&location=[38.9517,-92.3341]&time=2000-06-01T00:00:00/2000-06-01T07:30:00&version=2.0

# https://api.giovanni.earthdata.nasa.gov/timeseries?data=NLDAS_FORA0125_H_2_0_Rainf&location=[38.9517,-92.3341]&time=2024-01-01T00:00:00/2024-01-10T23:00:00&version=2.0

curl -H "authorization:Bearer ${bearer_token}" -X --url "https://api.giovanni.earthdata.nasa.gov/timeseries?data=NLDAS_FORA0125_H_2_0_Rainf&location=[38.9517,-92.3341]&time=2024-01-01T00:00:00/2024-01-10T23:00:00&version=2.0" -o "test.csv"


