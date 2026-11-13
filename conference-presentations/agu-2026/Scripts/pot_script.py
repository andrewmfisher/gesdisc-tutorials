# POT script
#
# Written with assistance from NASA Chat
# Uses pyextremes to estimate recurrence intervals with peak-over-threshold method

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pyextremes import EVA
from scipy.stats import genpareto
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION SECTION - Adjust these parameters as needed
# ============================================================================

# File paths
NETCDF_FILE = '/Users/jschaper/Documents/Codes/gesdisc-tutorials/conference-presentations/agu-2026/IMERG_subsets/subset_kerchunk.nc'  # Update with your file path
OUTPUT_PLOT = '/Users/jschaper/Documents/Codes/gesdisc-tutorials/conference-presentations/agu-2026/Figures/recurrence_intervals_texas.png'

# Spatial domain (Central Texas region - adjust as needed)
LAT_MIN = 29.5 
LAT_MAX = 32.5 
LON_MIN = -101 
LON_MAX = -97  

# POT parameters
THRESHOLD_PERCENTILE = 90  # 90th percentile
RADIUS_HOURS = '48h'  # Decluster radius

# Return periods to calculate and plot
RETURN_PERIODS = [2, 5, 10, 100]  # years

# Variable name in NetCDF (adjust if different)
PRECIP_VAR = 'precipitation'  # or 'precipitationCal', etc.

# Plotting parameters
FIGURE_SIZE = (14, 12)
DPI = 300

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_subset_data(filepath, lat_min, lat_max, lon_min, lon_max):
    """
    Load NetCDF file and subset to region of interest.
    """
    print("Loading NetCDF data...")
    ds = xr.open_dataset(filepath)
    
    # Subset to Texas region
    ds_subset = ds.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max)
    )
    
    print(f"Data shape: {ds_subset[PRECIP_VAR].shape}")
    print(f"Time range: {ds_subset.time.values[0]} to {ds_subset.time.values[-1]}")
    
    return ds_subset


def calculate_threshold(precip_series, percentile):
    """
    Calculate threshold as the nth percentile of non-zero precipitation.
    """
    precip_nonzero = precip_series[precip_series > 0]
    if len(precip_nonzero) == 0:
        return np.nan
    threshold = np.percentile(precip_nonzero, percentile)
    return threshold


def fit_pot_model(precip_series, threshold, radius):
    """
    Fit POT model using pyextremes and return return levels.
    
    Returns:
        dict: Return levels for specified return periods, or None if fitting fails
    """
    try:
        # Remove zeros and create time series
        precip_nonzero = precip_series[precip_series > 0]
        
        if len(precip_nonzero) < 50:  # Minimum data requirement
            return None
        
        # Initialize EVA model
        model = EVA(precip_nonzero)
        
        # Extract extremes using POT method
        model.get_extremes(
            method="POT",
            threshold=threshold,
            r=radius
        )
        
        # Check if we have enough extremes
        if len(model.extremes) < 10:
            return None
        
        # Fit the model (Generalized Pareto Distribution)
        model.fit_model()
        
        # Calculate return levels
        return_levels = {}
        for rp in RETURN_PERIODS:
            try:
                # Get return level for this return period
                rl = model.get_return_value(
                    exceedance_probability=1/rp,
                    return_period_size='365.25D',  # Annual return periods
                    alpha=0.95  # Confidence interval
                )
                return_levels[rp] = rl
            except:
                return_levels[rp] = np.nan
        
        return return_levels
    
    except Exception as e:
        print(f"Error in POT fitting: {e}")
        return None


def process_grid_cell(ds, i, j, threshold_percentile, radius):
    """
    Process a single grid cell and return return levels.
    """
    # Extract time series for this grid cell
    df = ds.isel(lat=i, lon=j)[PRECIP_VAR].to_dataframe().reset_index()
    
    # Create time series
    precip_series = df.set_index('time')[PRECIP_VAR]
    
    # Calculate threshold
    threshold = calculate_threshold(precip_series, threshold_percentile)
    
    if np.isnan(threshold) or threshold <= 0:
        return None
    
    # Fit POT model
    return_levels = fit_pot_model(precip_series, threshold, radius)
    
    return return_levels


def process_all_grid_cells(ds, threshold_percentile, radius):
    """
    Process all grid cells and return 2D arrays of return levels.
    """
    n_lat = len(ds.lat)
    n_lon = len(ds.lon)
    
    # Initialize arrays to store return levels
    return_level_maps = {rp: np.full((n_lat, n_lon), np.nan) for rp in RETURN_PERIODS}
    
    print(f"\nProcessing {n_lat} x {n_lon} = {n_lat * n_lon} grid cells...")
    
    # Process each grid cell
    for i in range(n_lat):
        if i % 5 == 0:
            print(f"Processing latitude {i+1}/{n_lat}...")
        
        for j in range(n_lon):
            return_levels = process_grid_cell(ds, i, j, threshold_percentile, radius)
            
            if return_levels is not None:
                for rp in RETURN_PERIODS:
                    return_level_maps[rp][i, j] = return_levels[rp]
    
    return return_level_maps


def plot_return_levels(ds, return_level_maps, output_file):
    """
    Create 2x2 plot of return levels for different return periods.
    """
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE, subplot_kw={'projection': None})
    axes = axes.flatten()
    
    lats = ds.lat.values
    lons = ds.lon.values
    
    for idx, rp in enumerate(RETURN_PERIODS):
        ax = axes[idx]
        
        # Create meshgrid for plotting
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Plot data
        im = ax.pcolormesh(
            lon_grid, 
            lat_grid, 
            return_level_maps[rp],
            cmap='YlGnBu',
            shading='auto'
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Precipitation (mm)', fontsize=10)
        
        # Set title and labels
        ax.set_title(f'{rp}-Year Return Level', fontsize=12, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        
        # Add gridlines
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Format ticks
        ax.tick_params(labelsize=9)
    
    plt.suptitle(
        f'Precipitation Return Levels - POT Method ({THRESHOLD_PERCENTILE}th Percentile, {RADIUS_HOURS} Radius)',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    print("="*70)
    print("POT RECURRENCE INTERVAL ANALYSIS - IMERG DATA")
    print("="*70)
    
    # Load and subset data
    ds = load_and_subset_data(NETCDF_FILE, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    
    # Process all grid cells
    return_level_maps = process_all_grid_cells(ds, THRESHOLD_PERCENTILE, RADIUS_HOURS)
    
    # Create plots
    plot_return_levels(ds, return_level_maps, OUTPUT_PLOT)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    for rp in RETURN_PERIODS:
        valid_data = return_level_maps[rp][~np.isnan(return_level_maps[rp])]
        if len(valid_data) > 0:
            print(f"\n{rp}-Year Return Level:")
            print(f"  Mean: {np.mean(valid_data):.2f} mm")
            print(f"  Median: {np.median(valid_data):.2f} mm")
            print(f"  Min: {np.min(valid_data):.2f} mm")
            print(f"  Max: {np.max(valid_data):.2f} mm")
            print(f"  Valid cells: {len(valid_data)}")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
    
    # Close dataset
    ds.close()


if __name__ == "__main__":
    main()