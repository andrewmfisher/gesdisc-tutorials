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
import time
import csv

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
    # Changed: control local Dask compute concurrency without starting distributed processes.
    nworkers_compute: int = 5
    benchmark_results_saveloc: str = "./access_benchmark_results.csv"

    def __post_init__(self): # bbox is handled this way bc lists are mutable objects
        if self.bbox is None:
            self.bbox = [-103, 24, -93, 34] # [LonMin, LatMin, LonMax, LatMax]

# =========================================================================================================
# Set up "adapters" so we can call the kerchunk and opendap access functions with the same inputs
# =========================================================================================================

def run_kerchunk(config, auth, begin_dt, end_dt):
    """Adapter for Kerchunk streaming"""
    # Changed: pass the full dates so the Kerchunk loader can return the exact
    # requested interval, including intervals that cross a year boundary.
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

def measure_execution(func, *args, num_workers=5):
    """Executes a lazy loading function and measures metadata vs compute time."""
    # 1. Metadata
    t0 = time.perf_counter()
    lazy_ds = func(*args)
    t1 = time.perf_counter()
    
    # 2. Compute
    t2 = time.perf_counter()
    # Changed: Pydap-backed Dask graphs contain an authenticated HTTP session,
    # so compute them in local threads instead of serializing them to processes.
    _ = lazy_ds.compute(scheduler="threads", num_workers=num_workers)
    t3 = time.perf_counter()
    
    return (t1 - t0), (t3 - t2)

# =========================================================================================================
# Run benchmarking over suite of access methods
# =========================================================================================================

def run_benchmark_suite(config, auth, methods):
    """The generic engine that runs all methods over all durations and trials."""
    results = []
    base_dt = datetime.strptime(config.base_date, "%Y-%m-%d")

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
                    # Changed: use the configured number of local Dask compute threads.
                    meta_time, comp_time = measure_execution(
                        method_func,
                        config,
                        auth,
                        begin_dt,
                        end_dt,
                        num_workers=config.nworkers_compute,
                    )
                    print(f"      -> Meta: {meta_time:.2f}s | Compute: {comp_time:.2f}s")
                    
                    results.append({
                        "Method": method_name, "Days": days, "Trial": trial,
                        "Metadata_Time_s": meta_time, "Compute_Time_s": comp_time, "Status": "Success"
                    })
                except Exception as e:
                    print(f"      -> {method_name} Failed: {e}")
                    results.append({
                        "Method": method_name, "Days": days, "Trial": trial,
                        "Metadata_Time_s": None, "Compute_Time_s": None, "Status": f"Failed: {e}"
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
