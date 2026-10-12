"""
Stream IMERG data from Kerchunk virtual data stores

Date: 8/27/2026
Author: JRS
Environment: nasa-gesdisc-kerchunk
"""

import warnings
import xarray as xr

# Create virtual dataset loader function (based on Chris B's How To)
def get_vds(parq: str, bearer_token: str, bbox, subvar, chunks: dict={}, **kwargs): #also pass in: ShortName, Version, BeginDate, EndDate, LatMin, LatMax, LonMin, LonMax
    """
    Args:
        parq (str): The URL or path to the Kerchunk .parq file.
        bearer_token
        auth =  earthaccess.login() # output of earthaccess.login()
        chunks (dict, optional): Chunking scheme for the dataset. Defaults to {}.
        **kwargs: Additional keyword arguments to pass directly to xarray.open_dataset()

    """
    print("starting get_vds_sub")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=xr.SerializationWarning)
        ds = xr.open_dataset(
            "reference://",
            engine="zarr",
            chunks=chunks,
            backend_kwargs={
                "storage_options": {
                    "fo": str(parq),
                    "remote_protocol": "https",
                    "asynchronous": True,
                    "remote_options": {
                        "headers": {"Authorization": f'Bearer {bearer_token}'},
                        "asynchronous": True,
                    }
                },
                "consolidated": True
            },
            **kwargs
        )

    # subset to just the variable and spatial area of interest
    ds_sub = ds[[subvar]]
    ds_sub = ds_sub.sel(
        lon = slice(bbox[0], bbox[2]),
        lat = slice(bbox[1], bbox[3])
    )

    return ds_sub

def get_vds_multiyear(bearer_token, shortname, version, beginyear, endyear, bbox, subvar):
    
    vds_url = f"https://data.gesdisc.earthdata.nasa.gov/browse/kerchunk/GPM_L3/{shortname}.{version}"

    # Lazy load GPM IMERG data from virtual data store, one year at a time
    vds_lst = [] # initialize empty vds_lst
    for yr in range(beginyear, endyear+1):
        yr_string = str(yr)
        current_parq = f"{vds_url}/{yr}.parq"
        ds_subset = get_vds(current_parq, bearer_token, bbox, subvar)
        vds_lst.append(ds_subset)
    
    # Concatenate into one virtual data cube
    # xarray concatenate takes a list of Kerchunk files and concatenates them together along a specified dimension
    vds_cube = xr.concat(vds_lst, dim="time")
    vds_cube.attrs["BeginDate"] = str(vds_cube.time.min().dt.date.values)
    vds_cube.attrs["EndDate"] = str(vds_cube.time.max().dt.date.values)

    return vds_cube