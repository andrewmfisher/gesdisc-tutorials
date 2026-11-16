# POT script
#
# Written with assistance from NASA Chat
# Uses pyextremes to estimate recurrence intervals with peak-over-threshold method

# Import packages
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pyextremes import EVA
import pickle
import openpyxl
from tqdm import tqdm

# Read precipitation data from subset_kerchunk and fit EVA models for each grid cell ---------------------------------------------------------------------
# RUNS OVER ALL GRID CELLS
# --------------------------------------------------------------------------------------------------------------------------------------------------------

USE_POT = True # False uses AMM
method_name = 'POT' if USE_POT else 'AMM'

# Read data in netcdf format with xarray and convert to a dataframe
ds = xr.open_dataset("/Users/jschaper/Documents/Codes/gesdisc-tutorials/conference-presentations/agu-2026/IMERG_subsets/subset_kerchunk.nc", engine="netcdf4")

# Initialize dictionary to store all models (indexed by lat_index, lon_index)
models_dict = {}

# Initialize list to store summary data for Excel
summary_data = []

# Calculate total number of grid cells for progress bar
total_cells = len(ds.lat) * len(ds.lon)
print(f"Processing {total_cells} grid cells ({len(ds.lat)} latitudes × {len(ds.lon)} longitudes)")

# Create progress bar
pbar = tqdm(total=total_cells, desc="Processing grid cells", unit="cell")

# Loop through latitude and longitude indices
for i in range(len(ds.lat)):  # Loop through all latitudes
    for j in range(len(ds.lon)):  # Loop through all longitudes
        
        # Extract data for specific lat/lon and create dataframe
        df = ds.isel(lat=i, lon=j)['precipitation'].to_dataframe().reset_index()[['time', 'lat', 'lon', 'precipitation']]
        df.columns = ['date', 'lat', 'lon', 'precip_mm']
        
        # Get actual lat/lon values
        lat_val = df['lat'].iloc[0]
        lon_val = df['lon'].iloc[0]
        
        # Filter out zero precipitation values and keep datetime index
        df_noz = df[df['precip_mm'] > 0].copy()
        df_noz = df_noz.set_index('date')['precip_mm']
        
        # Update progress bar description with current location
        pbar.set_postfix({
            'lat_idx': i, 
            'lon_idx': j, 
            'lat': f'{lat_val:.2f}', 
            'lon': f'{lon_val:.2f}'
        })
        
        # Perform extreme value analysis
        model = EVA(data=df_noz)
        
        # Initialize threshold variable
        thres = None
        
        if USE_POT:
            thres = np.percentile(df_noz, 95)  # calculate threshold as a percentile
            model.get_extremes("POT", threshold=thres, r='48hr')
            model.fit_model(distribution="genpareto")  # for POT
        else:
            model.get_extremes("BM", block_size="365.2425D")  # Annual maxima
            model.fit_model(distribution="genextreme")  # for BM/AMM
            thres = np.nan  # No threshold for AMM

        # Store model and metadata in dictionary with (lat_index, lon_index) as key
        grid_key = (i, j)
        models_dict[grid_key] = {
            'model': model,
            'lat_index': i,
            'lon_index': j,
            'latitude': lat_val,
            'longitude': lon_val,
            'threshold': thres,
            'n_extremes': len(model.extremes),
            'n_nonzero': len(df_noz),
            'date_range': (df_noz.index.min(), df_noz.index.max()),
            'fit_parameters': model.model.fit_parameters
        }
        
        # Add summary data for Excel
        summary_row = {
            'lat_index': i,
            'lon_index': j,
            'latitude': lat_val,
            'longitude': lon_val,
            'threshold_mm': thres,
            'n_extreme_events': len(model.extremes),
            'n_nonzero_precip': len(df_noz),
            'date_start': df_noz.index.min(),
            'date_end': df_noz.index.max(),
        }
        
        # Add fit parameters to the row
        for param_name, param_value in model.model.fit_parameters.items():
            summary_row[f'param_{param_name}'] = param_value
        
        summary_data.append(summary_row)
        
        # Update progress bar
        pbar.update(1)

# Close progress bar
pbar.close()

# Save all models to a pickle file
print("\nSaving models to pickle file...")
# Update filename based on USE_POT setting
output_file = f'/Users/jschaper/Documents/Codes/gesdisc-tutorials/conference-presentations/agu-2026/Data/eva_models_all_gridcells_{method_name}.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(models_dict, f)
print(f"All models saved to {output_file}")

# Create DataFrame and save to Excel
print("Saving summary to Excel file...")
summary_df = pd.DataFrame(summary_data)
excel_file = f'/Users/jschaper/Documents/Codes/gesdisc-tutorials/conference-presentations/agu-2026/Data/eva_models_summary_{method_name}.xlsx'
summary_df.to_excel(excel_file, index=False, sheet_name='Model Summary')
print(f"Model summary saved to {excel_file}")

# Print summary statistics
print(f"\nTotal number of grid cells processed: {len(models_dict)}")
print(f"\nSummary DataFrame:")
print(summary_df.head())

# Calculate return periods and make plots ------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------------------------------

# Plot maps of precipitation return levels for multiple return periods
#
# Will take a long time if you want to estimate confidence intervals with nonparametric bootstrapping

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
import warnings

# Suppress runtime warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============= CONFIGURATION =============
USE_CONFIDENCE_INTERVALS = False  # Set to True to calculate CIs (much slower)
N_BOOTSTRAP_SAMPLES = 100  # Number of bootstrap samples (if using CIs)
RETURN_PERIODS = [2, 5, 10, 100]  # Return periods to plot (in years)

# Hunt, Texas location
HUNT_TX_LAT = 30.07
HUNT_TX_LON = -99.3
# =========================================

# Load the saved models
print("Loading fitted models...")
with open('/Users/jschaper/Documents/Codes/gesdisc-tutorials/conference-presentations/agu-2026/Data/eva_models_all_gridcells_POT.pkl', 'rb') as f:
    models_dict = pickle.load(f)

# Load the original dataset to get coordinate information
ds = xr.open_dataset("/Users/jschaper/Documents/Codes/gesdisc-tutorials/conference-presentations/agu-2026/IMERG_subsets/subset_kerchunk.nc", engine="netcdf4")

# Initialize arrays to store return levels
n_lat = len(ds.lat)
n_lon = len(ds.lon)
n_periods = len(RETURN_PERIODS)

return_levels = {rp: np.full((n_lat, n_lon), np.nan) for rp in RETURN_PERIODS}

if USE_CONFIDENCE_INTERVALS:
    return_levels_lower = {rp: np.full((n_lat, n_lon), np.nan) for rp in RETURN_PERIODS}
    return_levels_upper = {rp: np.full((n_lat, n_lon), np.nan) for rp in RETURN_PERIODS}

# Calculate return levels for each grid cell and each return period
print(f"Calculating return levels for periods: {RETURN_PERIODS}")
if USE_CONFIDENCE_INTERVALS:
    print(f"Using bootstrap with {N_BOOTSTRAP_SAMPLES} samples (this will take longer)...")
else:
    print("Not calculating confidence intervals (faster)...")

for (i, j), model_info in models_dict.items():
    model = model_info['model']
    
    for rp in RETURN_PERIODS:
        try:
            if USE_CONFIDENCE_INTERVALS:
                # Get return level with confidence intervals using bootstrap
                rl, rl_lower, rl_upper = model.get_return_value(
                    return_period=rp, 
                    alpha=0.95, 
                    n_samples=N_BOOTSTRAP_SAMPLES
                )
                return_levels[rp][i, j] = rl
                return_levels_lower[rp][i, j] = rl_lower
                return_levels_upper[rp][i, j] = rl_upper
            else:
                # Get return level without confidence intervals (faster)
                result = model.get_return_value(return_period=rp)
                # Extract just the point estimate
                if isinstance(result, tuple):
                    rl = result[0]
                else:
                    rl = result
                return_levels[rp][i, j] = rl
                
        except Exception as e:
            print(f"Error calculating {rp}-year return level for grid cell ({i}, {j}): {e}")
            return_levels[rp][i, j] = np.nan
            if USE_CONFIDENCE_INTERVALS:
                return_levels_lower[rp][i, j] = np.nan
                return_levels_upper[rp][i, j] = np.nan

# Get extent for plots
lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())
lat_min, lat_max = float(ds.lat.min()), float(ds.lat.max())

# Create four-panel figure
fig = plt.figure(figsize=(20, 16))

for idx, rp in enumerate(RETURN_PERIODS):
    ax = plt.subplot(2, 2, idx+1, projection=ccrs.PlateCarree())
    
    # Add geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle='-', edgecolor='black')
    ax.add_feature(cfeature.STATES, linewidth=0.6, edgecolor='black')
    
    # Add counties
    try:
        counties = NaturalEarthFeature(
            category='cultural',
            name='admin_2_counties',
            scale='10m',
            facecolor='none',
            edgecolor='gray'
        )
        ax.add_feature(counties, linewidth=0.4, edgecolor='gray', alpha=0.6)
    except Exception as e:
        print(f"Could not load county boundaries: {e}")
    
    # Add lakes and rivers for geographic context
    ax.add_feature(cfeature.LAKES, alpha=0.3, facecolor='lightblue', edgecolor='blue', linewidth=0.5)
    ax.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.4, alpha=0.5)
    
    # Add cities and towns
    try:
        import cartopy.io.shapereader as shpreader
        cities_shp = shpreader.natural_earth(resolution='10m', category='cultural', 
                                             name='populated_places')
        
        # Plot cities within the extent
        for city in shpreader.Reader(cities_shp).records():
            lon, lat = city.geometry.x, city.geometry.y
            if lon_min-1 <= lon <= lon_max+1 and lat_min-1 <= lat <= lat_max+1:
                pop = city.attributes.get('POP_MAX', 0)
                city_name = city.attributes.get('NAME', '')
                
                # Plot all cities/towns with population > 10,000
                if pop > 10000:
                    marker_size = 3 if pop < 100000 else 5 if pop < 500000 else 7
                    ax.plot(lon, lat, 'ko', markersize=marker_size, 
                           transform=ccrs.PlateCarree(), zorder=5)
                    
                    # Label cities > 50k or important places like Kerrville
                    if pop > 50000 or 'Kerr' in city_name.lower():
                        fontsize = 7 if pop < 200000 else 9
                        ax.text(lon+0.05, lat, city_name, transform=ccrs.PlateCarree(), 
                               fontsize=fontsize, ha='left', va='center',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                       alpha=0.8, edgecolor='none'), zorder=6)
    except Exception as e:
        print(f"Could not load city data: {e}")
    
    # Plot the precipitation data (filter out NaN to avoid warnings)
    data = return_levels[rp]
    # Only plot where we have valid data
    valid_mask = ~np.isnan(data)
    
    if np.any(valid_mask):
        im = ax.pcolormesh(ds.lon, ds.lat, data, 
                           transform=ccrs.PlateCarree(),
                           cmap='YlGnBu', 
                           shading='auto')
        
        # Add colorbar with individual scale for each panel
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, shrink=0.8)
        cbar.set_label('Precipitation (mm)', fontsize=10)
    
    # Add Hunt, Texas marker (special location)
    ax.plot(HUNT_TX_LON, HUNT_TX_LAT, marker='*', color='red', markersize=15, 
           transform=ccrs.PlateCarree(), zorder=10, markeredgecolor='black', markeredgewidth=1)
    ax.text(HUNT_TX_LON+0.08, HUNT_TX_LAT, 'Hunt, TX', transform=ccrs.PlateCarree(), 
           fontsize=10, fontweight='bold', ha='left', va='center', color='red',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                   alpha=0.9, edgecolor='red', linewidth=1.5), zorder=11)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    if idx % 2 == 1:  # Right panels
        gl.left_labels = False
    if idx < 2:  # Top panels
        gl.bottom_labels = False
    
    # Set extent
    ax.set_extent([lon_min-0.5, lon_max+0.5, lat_min-0.5, lat_max+0.5], crs=ccrs.PlateCarree())
    
    # Set title
    ax.set_title(f'{rp}-Year Return Level', fontsize=14, fontweight='bold', pad=10)

plt.suptitle('Precipitation Return Levels', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()

# Save figure
filename = '/Users/jschaper/Documents/Codes/gesdisc-tutorials/conference-presentations/agu-2026/Data/return_levels_multipanel_POT.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
plt.show()

# Print statistics for each return period
for rp in RETURN_PERIODS:
    print(f"\n{rp}-Year Return Level Statistics:")
    print(f"  Minimum: {np.nanmin(return_levels[rp]):.2f} mm")
    print(f"  Maximum: {np.nanmax(return_levels[rp]):.2f} mm")
    print(f"  Mean: {np.nanmean(return_levels[rp]):.2f} mm")
    print(f"  Median: {np.nanmedian(return_levels[rp]):.2f} mm")

# Print return level at Hunt, Texas location
print(f"\nReturn Levels at Hunt, Texas ({HUNT_TX_LAT}°N, {HUNT_TX_LON}°W):")
for rp in RETURN_PERIODS:
    # Find nearest grid cell to Hunt, TX
    lat_idx = np.argmin(np.abs(ds.lat.values - HUNT_TX_LAT))
    lon_idx = np.argmin(np.abs(ds.lon.values - HUNT_TX_LON))
    hunt_value = return_levels[rp][lat_idx, lon_idx]
    print(f"  {rp}-Year: {hunt_value:.2f} mm")