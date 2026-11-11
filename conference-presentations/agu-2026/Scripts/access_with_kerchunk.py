"""
Stream IMERG data from Kerchunk virtual data stores

Date: 8/27/2026
Author: JRS
Last Update: JRS, Oct. 12, 2026
Environment: nasa-gesdisc-kerchunk
"""

import asyncio
import threading
import warnings
from datetime import datetime

import aiohttp
import earthaccess
import xarray as xr
from fsspec.implementations.http import HTTPFileSystem


class KerchunkRetryStats:
    """Thread-safe counters for remote Kerchunk requests."""

    def __init__(self):
        self._lock = threading.Lock()
        self.http_retries = 0
        self.token_refreshes = 0

    def increment_retries(self):
        with self._lock:
            self.http_retries += 1

    def increment_refreshes(self):
        with self._lock:
            self.token_refreshes += 1

    def snapshot(self):
        with self._lock:
            return self.http_retries, self.token_refreshes


class EarthdataTokenManager:
    """Hold the current bearer token and replace it after auth failures."""

    def __init__(self, bearer_token, stats):
        self._bearer_token = bearer_token
        self._stats = stats
        self._lock = threading.Lock()

    @property
    def bearer_token(self):
        with self._lock:
            return self._bearer_token

    def refresh_if_current(self, failed_token):
        """Refresh once when several requests fail with the same token."""
        with self._lock:
            if self._bearer_token != failed_token:
                return
            refreshed_auth = earthaccess.Auth().login(strategy="netrc")
            self._bearer_token = refreshed_auth.token["access_token"]
            self._stats.increment_refreshes()


class RetryingHTTPFileSystem(HTTPFileSystem):
    """HTTP filesystem with range-request retries and token renewal."""

    def __init__(
        self,
        token_manager,
        stats,
        max_attempts=5,
        backoff_seconds=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds cannot be negative")
        self.token_manager = token_manager
        self.stats = stats
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds

    async def _cat_file(self, url, start=None, end=None, **kwargs):
        retry_statuses = {408, 429, 500, 502, 503, 504}
        last_error = None

        for attempt in range(self.max_attempts):
            request_kwargs = self.kwargs.copy()
            request_kwargs.update(kwargs)
            headers = request_kwargs.pop("headers", {}).copy()
            token = self.token_manager.bearer_token
            headers["Authorization"] = f"Bearer {token}"
            if start is not None or end is not None:
                if start == end:
                    return b""
                headers["Range"] = await self._process_limits(url, start, end)
            request_kwargs["headers"] = headers

            try:
                session = await self.set_session()
                async with session.get(
                    self.encode_url(url), **request_kwargs
                ) as response:
                    if response.status in {401, 403}:
                        await response.read()
                        if attempt + 1 == self.max_attempts:
                            self._raise_not_found_for_status(response, url)
                        self.stats.increment_retries()
                        await asyncio.to_thread(
                            self.token_manager.refresh_if_current, token
                        )
                        await self._retry_delay(attempt)
                        continue
                    if response.status in retry_statuses:
                        await response.read()
                        if attempt + 1 == self.max_attempts:
                            self._raise_not_found_for_status(response, url)
                        self.stats.increment_retries()
                        await self._retry_delay(attempt)
                        continue

                    result = await response.read()
                    self._raise_not_found_for_status(response, url)
                    return result
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                last_error = error
                if attempt + 1 == self.max_attempts:
                    raise
                self.stats.increment_retries()
                await self._retry_delay(attempt)

        raise last_error  # pragma: no cover - loop always returns or raises

    async def _retry_delay(self, attempt):
        delay = min(self.backoff_seconds * (2 ** attempt), 30.0)
        if delay:
            await asyncio.sleep(delay)

# Create virtual dataset loader function (based on Chris B's How To)
def get_vds(
    parq: str,
    bearer_token: str,
    bbox,
    subvar,
    begin_date=None,
    end_date=None,
    chunks=None,
    remote_fs=None,
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

    if remote_fs is None:
        retry_stats = KerchunkRetryStats()
        remote_fs = RetryingHTTPFileSystem(
            token_manager=EarthdataTokenManager(bearer_token, retry_stats),
            stats=retry_stats,
            asynchronous=True,
        )

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
                    "asynchronous": True,
                    "fs": {"https": remote_fs, "http": remote_fs},
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

def get_vds_multiyear(
    bearer_token,
    shortname,
    version,
    begin_date,
    end_date,
    bbox,
    subvar,
    http_max_attempts=5,
    http_backoff_seconds=1.0,
):
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
    retry_stats = KerchunkRetryStats()
    token_manager = EarthdataTokenManager(bearer_token, retry_stats)
    remote_fs = RetryingHTTPFileSystem(
        token_manager=token_manager,
        stats=retry_stats,
        max_attempts=http_max_attempts,
        backoff_seconds=http_backoff_seconds,
        asynchronous=True,
    )

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
            remote_fs=remote_fs,
        )
        vds_lst.append(ds_subset)
    
    # Concatenate into one virtual data cube
    # xarray concatenate takes a list of Kerchunk files and concatenates them together along a specified dimension
    # vds_cube = xr.concat(vds_lst, dim="time")
    print("Concatenating files...")
    vds_cube = xr.concat(
        vds_lst, 
        dim='time',
        data_vars='minimal',
        coords='minimal', 
        compat='override',
        join = 'override',
        combine_attrs='override'
    )

    # Changed: record the requested inclusive bounds without forcing lazy time
    # coordinates to compute during the metadata portion of the benchmark.
    vds_cube.attrs["BeginDate"] = begin_date.date().isoformat()
    vds_cube.attrs["EndDate"] = end_date.date().isoformat()
    # Keep the mutable counters outside dataset attributes so they are not
    # serialized into the NetCDF output.
    vds_cube.encoding["kerchunk_retry_stats"] = retry_stats

    return vds_cube
