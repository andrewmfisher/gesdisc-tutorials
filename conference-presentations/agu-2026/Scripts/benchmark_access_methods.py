# Main script for benchmarking data access methods
#
# Access methods are defined in separate files

import access_with_kerchunk as vds
import access_with_opendap as opendap
import earthaccess
from datetime import datetime
from pydap.net import create_session
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time
import csv
import os

import dask

# =========================================================================================================
# Configuration
# =========================================================================================================

@dataclass
class BenchmarkConfig:
    shortname: str = "GPM_3IMERGDF"
    version: str = "07"
    bbox: list = None
    subvar: str = "precipitation"
    start_dates: tuple = ("1998-01-01","1998-01-01","1998-01-01","1998-01-01","1998-01-01","1998-01-01","1998-01-01","1998-01-01","1998-01-01")
    end_dates: tuple = ("1998-01-02","1998-01-04","1998-01-08","1998-01-16","1998-02-01", "1998-03-05", "1998-05-08","1998-09-13","1999-05-27")
    #start_dates: tuple = ("1998-01-01",)
    #end_dates: tuple = ("1998-01-30",)
    num_trials: int = 5 # number of trials for benchmarking (time may differ from one run to another)
    nworkers_opendap: int = 8 # number of workers for making cloud opendap requests
    nworkers_compute: int = 9 # n workers for local dask compute (loading data into memory with .compute)
    kerchunk_http_max_attempts: int = 5 # attempts per failed Kerchunk HTTP range request
    kerchunk_http_backoff_seconds: float = 1.0 # initial exponential-backoff delay
    max_compute_days: Optional[int] = 365 # threshold for actually calling .compute(). use None to compute ranges of any length
    save_long_kerchunk_once: bool = False # normally, no saving of netcdf file subsets. this overrides
    save_long_opendap_once: bool = False # force compute/save of the longest OPeNDAP range on its final trial
    benchmark_results_saveloc: str = "./conference-presentations/agu-2026/Data/access_benchmark_results.csv"
    subset_output_dir: str = "./conference-presentations/agu-2026/Data" # location to save downloaded data

    def __post_init__(self): # bbox is handled this way bc lists are mutable objects
        if self.bbox is None:
            self.bbox = [-101, 29.5, -97, 32.5] # [LonMin, LatMin, LonMax, LatMax]
        if len(self.start_dates) != len(self.end_dates):
            raise ValueError("start_dates and end_dates must contain the same number of dates")
        if not self.start_dates:
            raise ValueError("At least one start and end date must be provided")

# =========================================================================================================
# Set up "adapters" so we can call the kerchunk and opendap access functions with the same inputs
# =========================================================================================================

def run_kerchunk(config, auth, begin_dt, end_dt):
    """Adapter for Kerchunk streaming"""
    return vds.get_vds_multiyear(
        auth['bearer_token'],
        config.shortname,
        config.version,
        begin_dt,
        end_dt,
        config.bbox,
        subvar=config.subvar,
        http_max_attempts=config.kerchunk_http_max_attempts,
        http_backoff_seconds=config.kerchunk_http_backoff_seconds,
    )

def run_opendap(config, auth, begin_dt, end_dt):
    """Adapter for OPeNDAP streaming"""
    return opendap.get_opendap(
        auth['opendap_session'],
        config.shortname,
        config.version,
        begin_dt,
        end_dt,
        config.bbox,
        subvar=config.subvar,
        nworkers=config.nworkers_opendap
    )

# Dictionary mapping names to our standardized adapter functions
ACCESS_METHODS = {
    "Kerchunk": run_kerchunk,
    "OPeNDAP": run_opendap
}

RESULT_FIELDNAMES = (
    "Method",
    "Days",
    "Trial",
    "Lazy_Load_Time_s",
    "Compute_Time_s",
    "Estimated_Data_Size_MiB",
    "HTTP_Retries",
    "Token_Refreshes",
    "NetCDF_File",
    "Status",
)

# =========================================================================================================
# Benchmarking engine
# =========================================================================================================

class BenchmarkExecutionError(RuntimeError):
    """Preserve Kerchunk request counters when materialization fails."""

    def __init__(self, message, http_retries=0, token_refreshes=0):
        super().__init__(message)
        self.http_retries = http_retries
        self.token_refreshes = token_refreshes


def measure_execution(func, *args, compute=True, output_path=None, num_workers=5):
    """Measure lazy loading and compute time, optionally saving the result."""
    lazy_ds = None
    computed_ds = None
    http_retries = 0
    token_refreshes = 0

    try:
        # 1. Lazy load and subset metadata.
        t0 = time.perf_counter()
        lazy_ds = func(*args)
        lazy_load_time = time.perf_counter() - t0
        # Analytical, uncompressed array payload implied by shapes and dtypes.
        estimated_data_size_mib = lazy_ds.nbytes / (1024 ** 2)

        compute_time = None
        if compute:
            # 2. Materialize the lazy data using local threads because
            # Pydap-backed graphs contain an authenticated session that cannot
            # be serialized.
            t0 = time.perf_counter()
            print("calling .compute")
            with dask.config.set(scheduler="threads", num_workers=num_workers):
                computed_ds = lazy_ds.compute()
            compute_time = time.perf_counter() - t0
            print("data loaded into memory")

        # Saving is outside the compute timer.
        if output_path is not None and computed_ds is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            computed_ds.to_netcdf(output_path)
        retry_stats = lazy_ds.encoding.get("kerchunk_retry_stats")
        if retry_stats is not None:
            http_retries, token_refreshes = retry_stats.snapshot()
    except Exception as error:
        if lazy_ds is not None:
            retry_stats = lazy_ds.encoding.get("kerchunk_retry_stats")
            if retry_stats is not None:
                http_retries, token_refreshes = retry_stats.snapshot()
        raise BenchmarkExecutionError(
            str(error), http_retries, token_refreshes
        ) from error
    finally:
        if computed_ds is not None:
            computed_ds.close()
        if lazy_ds is not None:
            lazy_ds.close()

    return (
        lazy_load_time,
        compute_time,
        estimated_data_size_mib,
        http_retries,
        token_refreshes,
    )

# =========================================================================================================
# Run benchmarking over suite of access methods
# =========================================================================================================

def run_benchmark_suite(config, auth, methods, results_filename):
    """Run all methods and trials for each configured date range."""
    results = []
    initialize_results_file(results_filename)
    date_ranges = []
    for start_date, end_date in zip(config.start_dates, config.end_dates):
        begin_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        if end_dt < begin_dt:
            raise ValueError(f"End date {end_date} is before start date {start_date}")
        days = (end_dt - begin_dt).days + 1
        date_ranges.append((begin_dt, end_dt, days))

    compute_range = [
        config.max_compute_days is None or days <= config.max_compute_days
        for _, _, days in date_ranges
    ]
    eligible_range_indexes = [
        index for index, should_compute in enumerate(compute_range)
        if should_compute
    ]
    longest_range_index = max(
        range(len(date_ranges)), key=lambda index: date_ranges[index][2]
    )
    save_range_index = (
        longest_range_index
        if config.save_long_kerchunk_once
        else (
            max(eligible_range_indexes, key=lambda index: date_ranges[index][2])
            if eligible_range_indexes else None
        )
    )

    for range_index, (begin_dt, end_dt, days) in enumerate(date_ranges):

        print(f"\n=== Testing {days} Day(s): {begin_dt.date()} to {end_dt.date()} ===")

        for trial in range(1, config.num_trials + 1):
            print(f"  --- Trial {trial}/{config.num_trials} ---")

            # Iterate through our dictionary of methods dynamically!
            for method_name, method_func in methods.items():
                print(f"      Running {method_name}...")

                output_path = None
                try:
                    force_long_compute = (
                        (
                            method_name == "Kerchunk"
                            and config.save_long_kerchunk_once
                        )
                        or (
                            method_name == "OPeNDAP"
                            and config.save_long_opendap_once
                        )
                    ) and (
                        range_index == longest_range_index
                        and trial == config.num_trials
                    )
                    should_compute = (
                        compute_range[range_index]
                        or force_long_compute
                    )
                    save_opendap_subset = (
                        method_name == "OPeNDAP"
                        and config.save_long_opendap_once
                        and range_index == longest_range_index
                        and trial == config.num_trials
                    )
                    save_kerchunk_subset = (
                        method_name == "Kerchunk"
                        and range_index == save_range_index
                        and trial == config.num_trials
                    )
                    save_subset = save_kerchunk_subset or save_opendap_subset
                    if save_subset:
                        output_path = (
                            Path(config.subset_output_dir)
                            / f"subset_{method_name.lower()}.nc"
                        )
                    # Changed: use the configured number of local Dask compute threads.
                    (
                        lazy_time,
                        comp_time,
                        estimated_size_mib,
                        http_retries,
                        token_refreshes,
                    ) = measure_execution(
                        method_func,
                        config,
                        auth,
                        begin_dt,
                        end_dt,
                        compute=should_compute,
                        output_path=output_path,
                        num_workers=config.nworkers_compute,
                    )
                    if should_compute:
                        print(
                            f"      -> Lazy load: {lazy_time:.2f}s | "
                            f"Compute: {comp_time:.2f}s | "
                            f"Estimated data size: {estimated_size_mib:.2f} MiB"
                        )
                    else:
                        print(
                            f"      -> Lazy load: {lazy_time:.2f}s | "
                            "Compute: skipped | "
                            f"Estimated data size: {estimated_size_mib:.2f} MiB"
                        )
                    if output_path is not None:
                        print(f"      -> Saved subset to {output_path}")
                    if http_retries or token_refreshes:
                        print(
                            f"      -> HTTP retries: {http_retries} | "
                            f"Token refreshes: {token_refreshes}"
                        )

                    result = {
                        "Method": method_name,
                        "Days": days, "Trial": trial,
                        "Lazy_Load_Time_s": lazy_time,
                        "Compute_Time_s": comp_time,
                        "Estimated_Data_Size_MiB": estimated_size_mib,
                        "HTTP_Retries": http_retries,
                        "Token_Refreshes": token_refreshes,
                        "NetCDF_File": str(output_path) if output_path else None,
                        "Status": "Success" if should_compute else "Success (compute skipped)"
                    }
                    results.append(result)
                    append_result(result, results_filename)
                except Exception as e:
                    print(f"      -> {method_name} Failed: {e}")
                    http_retries = getattr(e, "http_retries", 0)
                    token_refreshes = getattr(e, "token_refreshes", 0)
                    result = {
                        "Method": method_name,
                        "Days": days, "Trial": trial,
                        "Lazy_Load_Time_s": None,
                        "Compute_Time_s": None,
                        "Estimated_Data_Size_MiB": None,
                        "HTTP_Retries": http_retries,
                        "Token_Refreshes": token_refreshes,
                        "NetCDF_File": str(output_path) if output_path else None,
                        "Status": f"Failed: {e}"
                    }
                    results.append(result)
                    append_result(result, results_filename)
    return results


# =========================================================================================================
# Output handler
# =========================================================================================================

def initialize_results_file(filename):
    """Create a fresh results CSV and persist its header immediately."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())


def append_result(result, filename):
    """Append and durably flush one completed benchmark result."""
    with Path(filename).open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writerow(result)
        f.flush()
        os.fsync(f.fileno())


def format_elapsed_duration(elapsed_seconds):
    """Format elapsed seconds as a filename-safe duration."""
    total_seconds = round(elapsed_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"


def make_results_paths(config, start_datetime):
    """Build the in-progress results path and the final path prefix."""
    configured_path = Path(config.benchmark_results_saveloc)
    start_label = start_datetime.strftime("%Y%m%dT%H%M%S")
    prefix = configured_path.parent / f"{configured_path.stem}_{start_label}"
    return prefix.with_name(f"{prefix.name}_running.csv"), prefix

# =========================================================================================================
# Main orchestrator
# =========================================================================================================

def main():

    # 1. Initialize Configuration
    config = BenchmarkConfig()

    # 2. Setup Auth (Bundle into a dictionary to pass to adapters easily)
    # get bearer token for EDL authentication (once)
    print("Authenticating...")
    auth = earthaccess.login(strategy="netrc")
    bearer_token = auth.token["access_token"]
    # cloud opendap uses session cookies, rather than a bearer token
    opendap_session = create_session(session_kwargs = {"token":bearer_token})
    auth_bundle = {
        'bearer_token': bearer_token,
        'opendap_session': opendap_session
    }

    # 3. Run Benchmark Suite. Keep a recoverable, incrementally written file
    # until the suite completes normally.
    run_start_datetime = datetime.now().astimezone()
    run_start_counter = time.perf_counter()
    temporary_results_path, final_results_prefix = make_results_paths(
        config, run_start_datetime
    )
    run_benchmark_suite(
        config,
        auth_bundle,
        ACCESS_METHODS,
        results_filename=temporary_results_path,
    )

    # 4. Add active elapsed duration to the name after successful completion.
    elapsed_label = format_elapsed_duration(
        time.perf_counter() - run_start_counter
    )
    final_results_path = final_results_prefix.with_name(
        f"{final_results_prefix.name}_{elapsed_label}.csv"
    )
    temporary_results_path.rename(final_results_path)
    print(
        f"\nBenchmarking complete! Results saved to '{final_results_path}'."
    )

if __name__ == "__main__":
    # Changed: do not start a distributed Client. The explicit threaded compute
    # above avoids Pydap graph deserialization and Bokeh dashboard shutdown errors.
    main()
