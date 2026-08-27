# Stream IMERG data from Kerchunk virtual data stores
#
# 8/27/2026 JRS
# Use nasa-gesdisc-kerchunk environment

import warnings
import earthaccess
import xarray as xr
import dask
from dask.distributed import Client

auth = earthaccess.login(strategy="interactive")

# Create virtual dataset loader function (from Chris B's How To)
def get_vds(parq: str, chunks: dict={}, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=xr.SerializationWarning)
        return xr.open_dataset(
            "reference://",
            engine="zarr",
            chunks=chunks,
            backend_kwargs={
                "storage_options": {
                    "fo": str(parq),
                    "remote_protocol": "https",
                    "asynchronous": True,
                    "remote_options": {
                        "headers": {"Authorization": f'Bearer {auth.bearerToken}'},
                        "asynchronous": True,
                    }
                },
                "consolidated": True
            },
            **kwargs
        )

print("getting data")

vds = get_vds("https://data.gesdisc.earthdata.nasa.gov/browse/kerchunk/GPM_L3/GPM_3IMERGDF.07/2011.parq")

print("got data")

print(vds)