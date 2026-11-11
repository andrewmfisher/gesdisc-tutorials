# Main script for benchmarking data access methods
#
# 8/31/2026 JRS
# 10/9/2026 ALM
# Methods are defined in separate files

from Scripts import access_with_kerchunk as vds
from Scripts import access_with_opendap as opendap
import earthaccess
from datetime import datetime, timedelta
from pydap.net import create_session
# Changed: this benchmark uses Dask's local threaded scheduler because Pydap's
# authenticated backend objects cannot reliably be serialized to worker processes.
from dataclasses import dataclass
from pathlib import Path
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
    base_date: str = "2024-01-01"
    durations_days: tuple = (1, 7, 14) # length of time to run benchmarking trials
    num_trials: int = 1 # number of trials for benchmarking (time may differ from one run to another)
    nworkers_opendap: int = 8 # number of workers for making cloud opendap requests
    nworkers_compute: int = 5 # n workers for local dask compute (loading data into memory with .compute)
    benchmark_results_saveloc: str = "./access_benchmark_results.csv"
    subset_output_dir: str = "./benchmark_subsets" # location to save downloaded data

    def __post_init__(self): # bbox is handled this way bc lists are mutable objects
        if self.bbox is None:
            self.bbox = [-101, 29.5, -97, 32.5] # [LonMin, LatMin, LonMax, LatMax]

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

def measure_execution(func, *args, output_path=None, num_workers=5):
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

        # 2. Materialize the lazy data using local threads because Pydap-backed
        # graphs contain an authenticated session that cannot be serialized.
        t0 = time.perf_counter()
        with dask.config.set(scheduler="threads", num_workers=num_workers):
            computed_ds = lazy_ds.compute()
        compute_time = time.perf_counter() - t0

        # Saving is outside the compute timer and is requested only for the last
        # trial of the longest benchmark period.
        if output_path is not None:
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
    """The generic engine that runs all methods over all durations and trials."""
    results = []
    base_dt = datetime.strptime(config.base_date, "%Y-%m-%d")
    longest_duration = max(config.durations_days)

    for days in config.durations_days:
        begin_dt = base_dt
        end_dt = base_dt + timedelta(days=days - 1)
        
        print(f"\n=== Testing {days} Day(s): {begin_dt.date()} to {end_dt.date()} ===")

        for trial in range(1, config.num_trials + 1):
            print(f"  --- Trial {trial}/{config.num_trials} ---")

            # Iterate through our dictionary of methods dynamically!
            for method_name, method_func in methods.items():
                print(f"      Running {method_name}...")
                
                try:
                    save_subset = (
                        days == longest_duration and trial == config.num_trials
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
                        output_path=output_path,
                        num_workers=config.nworkers_compute,
                    )
                    print(
                        f"      -> Lazy load: {lazy_time:.2f}s | "
                        f"Compute: {comp_time:.2f}s | "
                        f"Estimated data size: {estimated_size_mib:.2f} MiB"
                    )
                    if output_path is not None:
                        print(f"      -> Saved subset to {output_path}")
                    
                    results.append({
                        "Method": method_name, "Days": days, "Trial": trial,
                        "Lazy_Load_Time_s": lazy_time,
                        "Compute_Time_s": comp_time,
                        "Estimated_Data_Size_MiB": estimated_size_mib,
                        "NetCDF_File": str(output_path) if output_path else None,
                        "Status": "Success"
                    })
                except Exception as e:
                    print(f"      -> {method_name} Failed: {e}")
                    results.append({
                        "Method": method_name, "Days": days, "Trial": trial,
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
