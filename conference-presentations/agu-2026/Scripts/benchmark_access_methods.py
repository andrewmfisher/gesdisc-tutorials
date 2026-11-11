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
# Changed: this benchmark uses Dask's local threaded scheduler because Pydap's
# authenticated backend objects cannot reliably be serialized to worker processes.
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import time
import csv

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
    start_dates: tuple = ("2024-01-01", "2024-01-01")
    end_dates: tuple = ("2024-01-01", "2025-09-30")
    num_trials: int = 1 # number of trials for benchmarking (time may differ from one run to another)
    nworkers_opendap: int = 8 # number of workers for making cloud opendap requests
    nworkers_compute: int = 9 # n workers for local dask compute (loading data into memory with .compute)
    max_compute_days: Optional[int] = 365 # threshold for actually calling .compute(). use None to compute ranges of any length
    save_long_kerchunk_once: bool = True # normally, no saving of netcdf file subsets. this overrides
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
        subvar=config.subvar
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

# =========================================================================================================
# Benchmarking engine
# =========================================================================================================

def measure_execution(func, *args, compute=True, output_path=None, num_workers=5):
    """Measure lazy loading and compute time, optionally saving the result."""
    lazy_ds = None
    computed_ds = None

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
            with dask.config.set(scheduler="threads", num_workers=num_workers):
                computed_ds = lazy_ds.compute()
            compute_time = time.perf_counter() - t0

        # Saving is outside the compute timer.
        if output_path is not None and computed_ds is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            computed_ds.to_netcdf(output_path)
    finally:
        if computed_ds is not None:
            computed_ds.close()
        if lazy_ds is not None:
            lazy_ds.close()

    return lazy_load_time, compute_time, estimated_data_size_mib

# =========================================================================================================
# Run benchmarking over suite of access methods
# =========================================================================================================

def run_benchmark_suite(config, auth, methods):
    """Run all methods and trials for each configured date range."""
    results = []
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

                try:
                    should_compute = (
                        compute_range[range_index]
                        or (
                            config.save_long_kerchunk_once
                            and method_name == "Kerchunk"
                            and range_index == longest_range_index
                            and trial == config.num_trials
                        )
                    )
                    save_subset = (
                        method_name == "Kerchunk"
                        and range_index == save_range_index
                        and trial == config.num_trials
                    )
                    output_path = None
                    if save_subset:
                        output_path = (
                            Path(config.subset_output_dir)
                            / f"subset_{method_name.lower()}.nc"
                        )
                    # Changed: use the configured number of local Dask compute threads.
                    lazy_time, comp_time, estimated_size_mib = measure_execution(
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

                    results.append({
                        "Method": method_name,
                        "Days": days, "Trial": trial,
                        "Lazy_Load_Time_s": lazy_time,
                        "Compute_Time_s": comp_time,
                        "Estimated_Data_Size_MiB": estimated_size_mib,
                        "NetCDF_File": str(output_path) if output_path else None,
                        "Status": "Success" if should_compute else "Success (compute skipped)"
                    })
                except Exception as e:
                    print(f"      -> {method_name} Failed: {e}")
                    results.append({
                        "Method": method_name,
                        "Days": days, "Trial": trial,
                        "Lazy_Load_Time_s": None,
                        "Compute_Time_s": None,
                        "Estimated_Data_Size_MiB": None,
                        "NetCDF_File": str(output_path) if output_path else None,
                        "Status": f"Failed: {e}"
                    })
    return results


# =========================================================================================================
# Output handler
# =========================================================================================================

def export_results(results, filename):
    """Writes the dictionary of results to a CSV."""
    if not results:
        return
    keys = results[0].keys()
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nBenchmarking complete! Results saved to '{filename}'.")

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

    # 3. Run Benchmark Suite
    results = run_benchmark_suite(config, auth_bundle, ACCESS_METHODS)

    # 4. Export
    export_results(results, filename = config.benchmark_results_saveloc)

if __name__ == "__main__":
    # Changed: do not start a distributed Client. The explicit threaded compute
    # above avoids Pydap graph deserialization and Bokeh dashboard shutdown errors.
    main()
