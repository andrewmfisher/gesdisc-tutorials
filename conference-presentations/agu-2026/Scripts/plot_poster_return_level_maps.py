"""Create poster-ready 2- and 10-year IMERG and NOAA return-level maps."""

from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "Data" / "pot_threshold_sensitivity"
FIGURE_DIR = PROJECT_DIR / "Figures"
IMERG_RESULTS = DATA_DIR / "pot_threshold_sensitivity_summary.csv"
NOAA_RESULTS = DATA_DIR / "noaa_atlas_domain_comparison.csv"
RETURN_PERIODS = (2.0, 10.0)
REFERENCE_PERCENTILE = 90.0
HUNT_LATITUDE = 30.07
HUNT_LONGITUDE = -99.3
CMAP = "YlGnBu"


def grid_from_table(
    table: pd.DataFrame, value_column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot a long-form grid table into sorted coordinates and values."""
    grid = table.pivot(index="latitude", columns="longitude", values=value_column)
    grid = grid.sort_index().sort_index(axis=1)
    return (
        grid.index.to_numpy(dtype=float),
        grid.columns.to_numpy(dtype=float),
        grid.to_numpy(dtype=float),
    )


def load_maps() -> tuple[dict[float, tuple], dict[float, tuple]]:
    """Load the saved reference-threshold IMERG and area-adjusted NOAA maps."""
    imerg_table = pd.read_csv(IMERG_RESULTS)
    imerg_table = imerg_table[
        (imerg_table["threshold_percentile"] == REFERENCE_PERCENTILE)
        & (imerg_table["status"] == "ok")
    ]
    noaa_table = pd.read_csv(NOAA_RESULTS)

    imerg_maps = {
        period: grid_from_table(
            imerg_table, f"return_level_{period:g}yr_mm"
        )
        for period in RETURN_PERIODS
    }
    noaa_maps = {
        period: grid_from_table(
            noaa_table[noaa_table["return_period_years"] == period],
            "noaa_atlas_depth_mm",
        )
        for period in RETURN_PERIODS
    }
    return imerg_maps, noaa_maps


def add_map_context(ax: plt.Axes, latitudes: np.ndarray, longitudes: np.ndarray) -> None:
    """Add boundaries, coordinates, and the Hunt reference location."""
    ax.add_feature(cfeature.STATES, linewidth=0.8, edgecolor="0.25")
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="0.35")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="0.35")
    ax.plot(
        HUNT_LONGITUDE,
        HUNT_LATITUDE,
        marker="*",
        color="#D73027",
        markeredgecolor="white",
        markeredgewidth=0.7,
        markersize=12,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    ax.text(
        HUNT_LONGITUDE + 0.08,
        HUNT_LATITUDE + 0.03,
        "Hunt",
        fontsize=9,
        fontweight="bold",
        color="#A50026",
        transform=ccrs.PlateCarree(),
        zorder=6,
    )
    ax.set_extent(
        [
            float(longitudes.min()) - 0.08,
            float(longitudes.max()) + 0.08,
            float(latitudes.min()) - 0.08,
            float(latitudes.max()) + 0.08,
        ],
        crs=ccrs.PlateCarree(),
    )
    gridlines = ax.gridlines(
        draw_labels=True, linewidth=0.4, color="0.3", alpha=0.4, linestyle="--"
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 9}
    gridlines.ylabel_style = {"size": 9}


def create_map_figure(
    maps: dict[float, tuple],
    title: str,
    output_stem: str,
    vmin: float,
    vmax: float,
) -> None:
    """Create one 1x2 figure and save raster and vector versions."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.4),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    image = None
    for panel_index, (ax, period) in enumerate(zip(axes, RETURN_PERIODS)):
        latitudes, longitudes, values = maps[period]
        image = ax.pcolormesh(
            longitudes,
            latitudes,
            values,
            cmap=CMAP,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            transform=ccrs.PlateCarree(),
        )
        add_map_context(ax, latitudes, longitudes)
        ax.text(
            0.025,
            0.975,
            f"({chr(97 + panel_index)}) {period:g}-year return level",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "0.35", "alpha": 0.88},
            zorder=10,
        )
    colorbar = fig.colorbar(
        image,
        ax=axes,
        orientation="horizontal",
        shrink=0.72,
        pad=0.08,
        aspect=35,
    )
    colorbar.set_label("Precipitation depth (mm)", fontsize=11)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for extension, dpi in (("png", 300), ("pdf", 300)):
        fig.savefig(
            FIGURE_DIR / f"{output_stem}.{extension}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    imerg_maps, noaa_maps = load_maps()
    all_values = np.concatenate(
        [
            values.ravel()
            for collection in (imerg_maps, noaa_maps)
            for _, _, values in collection.values()
        ]
    )
    finite = all_values[np.isfinite(all_values)]
    # Identical full-range limits make the two poster figures directly comparable.
    vmin, vmax = float(finite.min()), float(finite.max())
    create_map_figure(
        imerg_maps,
        "IMERG POT return levels",
        "poster_imerg_return_levels_2yr_10yr",
        vmin,
        vmax,
    )
    create_map_figure(
        noaa_maps,
        "Area-adjusted NOAA Atlas 14 return levels",
        "poster_noaa_return_levels_2yr_10yr",
        vmin,
        vmax,
    )
    print(f"Shared color scale: {vmin:.1f}–{vmax:.1f} mm")
    print(f"Figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
