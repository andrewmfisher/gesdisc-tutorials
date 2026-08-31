"""
Stream IMERG data from Kerchunk virtual data stores

Date: 8/27/2026
Author: JRS
Environment: nasa-gesdisc-kerchunk
"""

import warnings
import earthaccess
import xarray as xr

# Create virtual dataset loader function (from Chris B's How To)
def get_vds(parq: str, auth, chunks: dict={}, **kwargs):
    """
    Args:
        parq (str): The URL or path to the Kerchunk .parq file.
        bearer_token 
        auth =  earthaccess.login() # output of earthaccess.login()
        chunks (dict, optional): Chunking scheme for the dataset. Defaults to {}.
        **kwargs: Additional keyword arguments to pass directly to xarray.open_dataset()

    """

    print("starting get_vds")
    bearer_token = auth.token["access_token"]


    #auth = earthaccess.login(strategy="netrc")
    #bearer_token = auth.token["access_token"]

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
                        "headers": {"Authorization": f'Bearer {bearer_token}'},
                        "asynchronous": True,
                    }
                },
                "consolidated": True
            },
            **kwargs
        )
