"""
Stream IMERG data from Kerchunk virtual data stores

Date: 8/27/2026
Author: JRS
Last Update: JRS, Oct. 12, 2026
Environment: nasa-gesdisc-kerchunk
"""

import warnings
from datetime import datetime

import xarray as xr

# Create virtual dataset loader function (based on Chris B's How To)
def get_vds(
    parq: str,
    bearer_token: str,
    bbox,
    subvar,
    begin_date=None,
    end_date=None,
    chunks=None,
    **kwargs,
):
    """
    Args:
        parq (str): The URL or path to the Kerchunk .parq file.
        bearer_token
        auth =  earthaccess.login() # output of earthaccess.login()
        bbox = bounding box for subset, bbox = [LonMin, LatMin, LonMax, LatMax]
        subvar = variable name for subset
        begin_date: Optional first datetime to retain.
        end_date: Optional last datetime to retain.
        chunks (dict, optional): Chunking scheme for the dataset. Defaults to {}.
        **kwargs: Additional keyword arguments to pass directly to xarray.open_dataset()

    """
    print("starting get_vds_sub")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=xr.SerializationWarning)
        # Changed: avoid a mutable dictionary as a function default while keeping
        # the previous behavior of asking xarray for automatic Dask chunks.
        if chunks is None:
            chunks = {}

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

    # Changed: optionally subset each yearly store before concatenation. This
    # preserves exact date ranges and avoids carrying unrequested dates forward.
    if begin_date is not None or end_date is not None:
        ds_sub = ds_sub.sel(time=slice(begin_date, end_date))

    return ds_sub

def get_vds_multiyear(bearer_token, shortname, version, begin_date, end_date, bbox, subvar):
    """
    Args:
        bearer_token
        shortname
        version
        begin_date: First datetime to retain (inclusive).
        end_date: Last datetime to retain (inclusive).
        bbox
        subvar
    """

    # Changed: accept datetime inputs, validate their order, and derive years only
    # for locating the annual Kerchunk reference stores.
    if not isinstance(begin_date, datetime) or not isinstance(end_date, datetime):
        raise TypeError("begin_date and end_date must be datetime objects")
    if end_date < begin_date:
        raise ValueError("end_date must be on or after begin_date")

    vds_url = f"https://data.gesdisc.earthdata.nasa.gov/browse/kerchunk/GPM_L3/{shortname}.{version}"

    # Lazy load GPM IMERG data from virtual data store, one year at a time
    vds_lst = [] # initialize empty vds_lst
    # Changed: open every annual store touched by the requested date interval.
    for yr in range(begin_date.year, end_date.year + 1):
        current_parq = f"{vds_url}/{yr}.parq"
        # Changed: pass the exact bounds to each store; xarray returns only dates
        # that overlap this interval, while out-of-year bounds are harmless.
        ds_subset = get_vds(
            current_parq,
            bearer_token,
            bbox,
            subvar,
            begin_date=begin_date,
            end_date=end_date,
        )
        vds_lst.append(ds_subset)
    
    # Concatenate into one virtual data cube
    # xarray concatenate takes a list of Kerchunk files and concatenates them together along a specified dimension
    vds_cube = xr.concat(vds_lst, dim="time")
    # Changed: record the requested inclusive bounds without forcing lazy time
    # coordinates to compute during the metadata portion of the benchmark.
    vds_cube.attrs["BeginDate"] = begin_date.date().isoformat()
    vds_cube.attrs["EndDate"] = end_date.date().isoformat()

    return vds_cube
