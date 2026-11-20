"""Retrieve NOAA Atlas precipitation frequencies and compare them with POT fits.

The NOAA HDSC/PFDS endpoint returns the currently operational atlas for a
coordinate. Responses are cached locally so subsequent plotting is offline and
reproducible. For daily IMERG data, use the 24-hour duration and PDS series.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PFDS_ENDPOINT = "https://hdsc.nws.noaa.gov/cgi-bin/hdsc/new/cgi_readH5.py"
ATLAS_GIS_BASE = "https://hdsc.nws.noaa.gov/pub/hdsc/data/tx"
ATLAS_RETURN_PERIODS = (1, 2, 5, 10, 25, 50, 100, 200, 500, 1000)
ATLAS_DURATIONS = (
    "5-min", "10-min", "15-min", "30-min", "60-min", "2-hr", "3-hr",
    "6-hr", "12-hr", "24-hr", "2-day", "3-day", "4-day", "7-day",
    "10-day", "20-day", "30-day", "45-day", "60-day",
)
TP29_ARF_SOURCE = (
    "U.S. Weather Bureau (1957), Technical Paper No. 29; recommended by "
    "NOAA Atlas 14 Volume 11, Section 4.6.1"
)


def _parse_assignment(response: str, name: str):
    match = re.search(rf"^{name}\s*=\s*(.*?);\s*$", response, re.MULTILINE)
    if not match:
        raise ValueError(f"NOAA PFDS response did not contain {name!r}")
    return ast.literal_eval(match.group(1))


def _cache_path(cache_dir: Path, latitude: float, longitude: float, series: str) -> Path:
    coordinate = f"{latitude:.4f}_{longitude:.4f}".replace("-", "m").replace(".", "p")
    return cache_dir / f"noaa_atlas_{coordinate}_{series}.json"


def fetch_noaa_atlas(
    latitude: float,
    longitude: float,
    cache_dir: Path,
    series: str = "pds",
) -> dict:
    """Fetch and cache metric precipitation-frequency tables from NOAA PFDS."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(cache_dir, latitude, longitude, series)
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    parameters = {
        "lat": f"{latitude:.4f}",
        "lon": f"{longitude:.4f}",
        "type": "pf",
        "data": "depth",
        "units": "metric",
        "series": series,
    }
    request = Request(
        f"{PFDS_ENDPOINT}?{urlencode(parameters)}",
        headers={"User-Agent": "gesdisc-tutorials POT/NOAA comparison"},
    )
    with urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8")

    payload = {
        "quantiles": _parse_assignment(text, "quantiles"),
        "lower": _parse_assignment(text, "lower"),
        "upper": _parse_assignment(text, "upper"),
        "latitude": _parse_assignment(text, "lat"),
        "longitude": _parse_assignment(text, "lon"),
        "region": _parse_assignment(text, "region"),
        "volume": _parse_assignment(text, "volume"),
        "version": _parse_assignment(text, "version"),
        "authors": _parse_assignment(text, "authors"),
        "series": series,
        "units": "mm",
        "source_url": request.full_url,
    }
    cache_path.write_text(json.dumps(payload, indent=2))
    return payload


def atlas_frequency_table(
    payload: dict, duration: str, return_periods: tuple[float, ...]
) -> pd.DataFrame:
    """Extract one duration and selected return periods from a PFDS response."""
    if duration not in ATLAS_DURATIONS:
        raise ValueError(f"Unsupported NOAA duration {duration!r}")
    duration_index = ATLAS_DURATIONS.index(duration)
    rows = []
    for period in return_periods:
        if period not in ATLAS_RETURN_PERIODS:
            raise ValueError(
                f"NOAA Atlas does not provide a {period:g}-year column; "
                f"available values are {ATLAS_RETURN_PERIODS}"
            )
        period_index = ATLAS_RETURN_PERIODS.index(period)
        rows.append(
            {
                "return_period_years": period,
                "noaa_atlas_depth_mm": float(payload["quantiles"][duration_index][period_index]),
                "noaa_atlas_lower_mm": float(payload["lower"][duration_index][period_index]),
                "noaa_atlas_upper_mm": float(payload["upper"][duration_index][period_index]),
            }
        )
    return pd.DataFrame(rows)


def _gis_archive_name(return_period: float, duration: str = "24-hr") -> str:
    if duration != "24-hr":
        raise ValueError("Domain comparison currently supports the matched 24-hr grids")
    if return_period not in ATLAS_RETURN_PERIODS:
        raise ValueError(f"Unsupported NOAA return period: {return_period:g}")
    return f"tx{return_period:g}yr24ha.zip"


def load_noaa_atlas_grid(
    return_period: float, cache_dir: Path, duration: str = "24-hr"
) -> tuple[np.ndarray, dict[str, float]]:
    """Download/cache and read one official NOAA Atlas 14 Texas ASCII grid."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_name = _gis_archive_name(return_period, duration)
    archive_path = cache_dir / archive_name
    if not archive_path.exists():
        request = Request(
            f"{ATLAS_GIS_BASE}/{archive_name}",
            headers={"User-Agent": "gesdisc-tutorials POT/NOAA comparison"},
        )
        with urlopen(request, timeout=60) as response, archive_path.open("wb") as file:
            shutil.copyfileobj(response, file)

    with ZipFile(archive_path) as archive:
        ascii_name = next(name for name in archive.namelist() if name.endswith(".asc"))
        with archive.open(ascii_name) as file:
            header = {}
            for _ in range(6):
                key, value = file.readline().decode().split()
                header[key.lower()] = float(value)
            values = np.loadtxt(file)

    values[values == header["nodata_value"]] = np.nan
    # NOAA GIS rasters store precipitation depth in thousandths of an inch.
    values = values / 1000 * 25.4
    return values, header


def sample_ascii_grid(
    values: np.ndarray,
    header: dict[str, float],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor sample an ESRI ASCII grid at target cell centers."""
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
    cellsize = header["cellsize"]
    north_edge = header["yllcorner"] + header["nrows"] * cellsize
    rows = np.rint((north_edge - latitude_grid) / cellsize - 0.5).astype(int)
    columns = np.rint(
        (longitude_grid - header["xllcorner"]) / cellsize - 0.5
    ).astype(int)
    valid = (
        (rows >= 0)
        & (rows < values.shape[0])
        & (columns >= 0)
        & (columns < values.shape[1])
    )
    sampled = np.full(latitude_grid.shape, np.nan)
    sampled[valid] = values[rows[valid], columns[valid]]
    return sampled


def average_ascii_grid_over_cells(
    values: np.ndarray,
    header: dict[str, float],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    """Average fine NOAA point-frequency pixels within each target grid cell."""
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    if len(latitudes) < 2 or len(longitudes) < 2:
        raise ValueError("At least two coordinates per dimension are required")
    lat_step = float(np.median(np.diff(np.sort(latitudes))))
    lon_step = float(np.median(np.diff(np.sort(longitudes))))
    cellsize = header["cellsize"]
    source_lons = header["xllcorner"] + (
        np.arange(values.shape[1]) + 0.5
    ) * cellsize
    north_edge = header["yllcorner"] + header["nrows"] * cellsize
    source_lats = north_edge - (np.arange(values.shape[0]) + 0.5) * cellsize
    averaged = np.full((len(latitudes), len(longitudes)), np.nan)
    for lat_index, latitude in enumerate(latitudes):
        rows = np.flatnonzero(
            (source_lats >= latitude - lat_step / 2)
            & (source_lats < latitude + lat_step / 2)
        )
        for lon_index, longitude in enumerate(longitudes):
            columns = np.flatnonzero(
                (source_lons >= longitude - lon_step / 2)
                & (source_lons < longitude + lon_step / 2)
            )
            if rows.size and columns.size:
                averaged[lat_index, lon_index] = np.nanmean(
                    values[np.ix_(rows, columns)]
                )
    return averaged


def latitude_longitude_cell_areas_km2(
    latitudes: np.ndarray, longitudes: np.ndarray
) -> np.ndarray:
    """Calculate spherical areas for a regular latitude-longitude grid."""
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)
    lat_step = float(np.median(np.diff(np.sort(latitudes))))
    lon_step = float(np.median(np.diff(np.sort(longitudes))))
    south = np.deg2rad(latitudes - lat_step / 2)
    north = np.deg2rad(latitudes + lat_step / 2)
    longitude_width = np.deg2rad(lon_step)
    row_areas = 6371.0088**2 * longitude_width * (np.sin(north) - np.sin(south))
    return np.repeat(row_areas[:, np.newaxis], len(longitudes), axis=1)


def compare_domain_with_noaa_atlas(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    imerg_return_levels: dict[float, np.ndarray],
    return_periods: tuple[float, ...],
    output_dir: Path,
    duration: str = "24-hr",
    areal_reduction_factor: float = 0.95,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
    """Compare IMERG POT maps with area-adjusted NOAA Atlas 14 grids."""
    if not 0 < areal_reduction_factor <= 1:
        raise ValueError("areal_reduction_factor must be greater than 0 and at most 1")
    cache_dir = output_dir / "noaa_cache" / "gis"
    noaa_maps: dict[float, np.ndarray] = {}
    difference_maps: dict[float, np.ndarray] = {}
    cell_areas = latitude_longitude_cell_areas_km2(latitudes, longitudes)
    table_rows = []
    for period in return_periods:
        values, header = load_noaa_atlas_grid(period, cache_dir, duration)
        noaa_center = sample_ascii_grid(values, header, latitudes, longitudes)
        noaa_cell_mean = average_ascii_grid_over_cells(
            values, header, latitudes, longitudes
        )
        noaa = noaa_cell_mean * areal_reduction_factor
        difference = imerg_return_levels[period] - noaa
        noaa_maps[period] = noaa
        difference_maps[period] = difference
        for lat_index, latitude in enumerate(latitudes):
            for lon_index, longitude in enumerate(longitudes):
                table_rows.append(
                    {
                        "return_period_years": period,
                        "lat_index": lat_index,
                        "lon_index": lon_index,
                        "latitude": latitude,
                        "longitude": longitude,
                        "imerg_pot_depth_mm": imerg_return_levels[period][lat_index, lon_index],
                        "noaa_atlas_center_point_depth_mm": noaa_center[lat_index, lon_index],
                        "noaa_atlas_cell_mean_point_depth_mm": noaa_cell_mean[lat_index, lon_index],
                        "noaa_areal_reduction_factor": areal_reduction_factor,
                        "noaa_atlas_depth_mm": noaa[lat_index, lon_index],
                        "noaa_atlas_areal_depth_mm": noaa[lat_index, lon_index],
                        "imerg_cell_area_km2": cell_areas[lat_index, lon_index],
                        "difference_imerg_minus_noaa_mm": difference[lat_index, lon_index],
                        "ratio_imerg_to_noaa": (
                            imerg_return_levels[period][lat_index, lon_index]
                            / noaa[lat_index, lon_index]
                        ),
                        "noaa_atlas": "NOAA Atlas 14 Volume 11",
                        "noaa_duration": duration,
                        "noaa_series": "PDS",
                        "noaa_arf_source": TP29_ARF_SOURCE,
                    }
                )
    pd.DataFrame(table_rows).to_csv(
        output_dir / "noaa_atlas_domain_comparison.csv", index=False
    )
    return noaa_maps, difference_maps


def summarize_domain_confidence_comparison(
    summary: pd.DataFrame,
    domain_comparison: pd.DataFrame,
    reference_percentile: float,
    return_periods: tuple[float, ...],
    output_dir: Path,
    confidence_level: float = 0.95,
    figure_dpi: int = 300,
    figure_format: str = "png",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare NOAA point estimates with saved POT CIs at every grid cell."""
    reference = summary[
        (summary["threshold_percentile"] == reference_percentile)
        & (summary["status"] == "ok")
    ].copy()
    key_columns = ["lat_index", "lon_index", "latitude", "longitude"]
    rows = []
    for period in return_periods:
        period_label = f"{period:g}"
        columns = key_columns + [
            f"return_level_{period_label}yr_mm",
            f"return_level_{period_label}yr_lower_mm",
            f"return_level_{period_label}yr_upper_mm",
        ]
        renamed = reference[columns].rename(
            columns={
                f"return_level_{period_label}yr_mm": "imerg_pot_depth_mm",
                f"return_level_{period_label}yr_lower_mm": "imerg_ci_lower_mm",
                f"return_level_{period_label}yr_upper_mm": "imerg_ci_upper_mm",
            }
        )
        noaa_columns = [
            "lat_index", "lon_index", "noaa_atlas_depth_mm",
            "noaa_atlas_center_point_depth_mm",
            "noaa_atlas_cell_mean_point_depth_mm",
            "noaa_atlas_areal_depth_mm", "noaa_areal_reduction_factor",
            "imerg_cell_area_km2", "noaa_arf_source",
        ]
        noaa_columns = [
            column for column in noaa_columns if column in domain_comparison.columns
        ]
        noaa = domain_comparison[
            domain_comparison["return_period_years"] == period
        ][noaa_columns]
        merged = renamed.merge(
            noaa, on=["lat_index", "lon_index"], how="left", validate="one_to_one"
        )
        merged.insert(0, "return_period_years", period)
        rows.append(merged)

    comparison = pd.concat(rows, ignore_index=True)
    valid = comparison[
        ["noaa_atlas_depth_mm", "imerg_ci_lower_mm", "imerg_ci_upper_mm"]
    ].notna().all(axis=1)
    comparison["noaa_position_relative_to_imerg_ci"] = np.select(
        [
            ~valid,
            comparison["noaa_atlas_depth_mm"] < comparison["imerg_ci_lower_mm"],
            comparison["noaa_atlas_depth_mm"] > comparison["imerg_ci_upper_mm"],
        ],
        ["CI unavailable", "below", "above"],
        default="within",
    )
    comparison["noaa_estimate_within_imerg_ci"] = pd.array(
        np.where(
            valid,
            comparison["noaa_position_relative_to_imerg_ci"] == "within",
            pd.NA,
        ),
        dtype="boolean",
    )
    comparison["noaa_distance_to_imerg_ci_mm"] = np.select(
        [
            ~valid,
            comparison["noaa_position_relative_to_imerg_ci"] == "below",
            comparison["noaa_position_relative_to_imerg_ci"] == "above",
        ],
        [
            np.nan,
            comparison["noaa_atlas_depth_mm"] - comparison["imerg_ci_lower_mm"],
            comparison["noaa_atlas_depth_mm"] - comparison["imerg_ci_upper_mm"],
        ],
        default=0.0,
    )
    comparison["imerg_confidence_level"] = confidence_level
    comparison["pot_threshold_percentile"] = reference_percentile
    comparison.to_csv(
        output_dir / "noaa_atlas_domain_confidence_comparison.csv", index=False
    )

    counts = (
        comparison.groupby(
            ["return_period_years", "noaa_position_relative_to_imerg_ci"],
            observed=False,
        )
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["below", "within", "above", "CI unavailable"], fill_value=0)
    )
    domain_summary = counts.reset_index()
    domain_summary["valid_grid_cells"] = counts[["below", "within", "above"]].sum(axis=1).to_numpy()
    for position in ("below", "within", "above"):
        domain_summary[f"percent_{position}"] = (
            100 * domain_summary[position] / domain_summary["valid_grid_cells"]
        )
    domain_summary["imerg_confidence_level"] = confidence_level
    domain_summary["pot_threshold_percentile"] = reference_percentile
    domain_summary.to_csv(
        output_dir / "noaa_atlas_domain_confidence_summary.csv", index=False
    )

    category_codes = {"below": -1, "within": 0, "above": 1}
    cmap = ListedColormap(["#3b82f6", "#22c55e", "#ef4444"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, period in zip(axes.flat, return_periods):
        period_rows = comparison[comparison["return_period_years"] == period]
        latitudes = np.sort(period_rows["latitude"].unique())
        longitudes = np.sort(period_rows["longitude"].unique())
        codes = period_rows.assign(
            category=period_rows["noaa_position_relative_to_imerg_ci"].map(category_codes)
        ).pivot(index="latitude", columns="longitude", values="category")
        codes = codes.reindex(index=latitudes, columns=longitudes)
        ax.pcolormesh(longitudes, latitudes, codes, cmap=cmap, norm=norm, shading="nearest")
        period_summary = domain_summary[
            domain_summary["return_period_years"] == period
        ].iloc[0]
        ax.set_title(
            f"{period:g}-year: {period_summary['percent_within']:.1f}% within CI\n"
            f"{period_summary['percent_below']:.1f}% below, "
            f"{period_summary['percent_above']:.1f}% above"
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
    fig.legend(
        handles=[
            Patch(color=cmap(0), label="NOAA below IMERG CI"),
            Patch(color=cmap(1), label="NOAA within IMERG CI"),
            Patch(color=cmap(2), label="NOAA above IMERG CI"),
        ],
        loc="outside lower center", ncol=3,
    )
    fig.suptitle(
        f"Area-adjusted NOAA Atlas estimates relative to IMERG POT "
        f"{confidence_level * 100:g}% confidence intervals\n"
        f"{reference_percentile:g}th-percentile POT threshold",
        fontsize=15, fontweight="bold",
    )
    fig.savefig(
        output_dir / f"noaa_atlas_domain_confidence_summary.{figure_format}",
        dpi=figure_dpi, bbox_inches="tight",
    )
    plt.close(fig)
    return comparison, domain_summary


def compare_with_noaa_atlas(
    summary: pd.DataFrame,
    reference_percentile: float,
    return_periods: tuple[float, ...],
    latitude: float,
    longitude: float,
    location_name: str,
    output_dir: Path,
    duration: str = "24-hr",
    series: str = "pds",
    confidence_level: float = 0.95,
    figure_dpi: int = 300,
    figure_format: str = "png",
) -> pd.DataFrame:
    """Save a point comparison table and uncertainty plot for NOAA versus POT."""
    atlas = fetch_noaa_atlas(latitude, longitude, output_dir / "noaa_cache", series)
    comparison = atlas_frequency_table(atlas, duration, return_periods)

    candidates = summary[
        (summary["threshold_percentile"] == reference_percentile)
        & (summary["status"] == "ok")
    ].copy()
    if candidates.empty:
        raise ValueError("No successful reference-threshold fit is available for NOAA comparison")
    distance_squared = (
        (candidates["latitude"] - latitude) ** 2
        + (candidates["longitude"] - longitude) ** 2
    )
    imerg = candidates.loc[distance_squared.idxmin()]

    comparison["imerg_pot_depth_mm"] = [
        imerg[f"return_level_{period:g}yr_mm"] for period in return_periods
    ]
    comparison["imerg_ci_lower_mm"] = [
        imerg.get(f"return_level_{period:g}yr_lower_mm", np.nan)
        for period in return_periods
    ]
    comparison["imerg_ci_upper_mm"] = [
        imerg.get(f"return_level_{period:g}yr_upper_mm", np.nan)
        for period in return_periods
    ]
    comparison["difference_imerg_minus_noaa_mm"] = (
        comparison["imerg_pot_depth_mm"] - comparison["noaa_atlas_depth_mm"]
    )
    comparison["ratio_imerg_to_noaa"] = (
        comparison["imerg_pot_depth_mm"] / comparison["noaa_atlas_depth_mm"]
    )
    comparison["difference_noaa_minus_imerg_mm"] = (
        comparison["noaa_atlas_depth_mm"] - comparison["imerg_pot_depth_mm"]
    )
    comparison["percent_difference_noaa_from_imerg"] = (
        100
        * comparison["difference_noaa_minus_imerg_mm"]
        / comparison["imerg_pot_depth_mm"]
    )

    finite_bounds = (
        np.isfinite(comparison["imerg_ci_lower_mm"])
        & np.isfinite(comparison["imerg_ci_upper_mm"])
    )
    noaa_within_imerg = (
        (comparison["noaa_atlas_depth_mm"] >= comparison["imerg_ci_lower_mm"])
        & (comparison["noaa_atlas_depth_mm"] <= comparison["imerg_ci_upper_mm"])
    )
    comparison["noaa_estimate_within_imerg_ci"] = pd.array(
        np.where(finite_bounds, noaa_within_imerg, pd.NA), dtype="boolean"
    )
    comparison["noaa_position_relative_to_imerg_ci"] = np.select(
        [
            ~finite_bounds,
            comparison["noaa_atlas_depth_mm"] < comparison["imerg_ci_lower_mm"],
            comparison["noaa_atlas_depth_mm"] > comparison["imerg_ci_upper_mm"],
        ],
        ["CI unavailable", "below", "above"],
        default="within",
    )
    comparison["noaa_distance_to_imerg_ci_mm"] = np.select(
        [
            ~finite_bounds,
            comparison["noaa_atlas_depth_mm"] < comparison["imerg_ci_lower_mm"],
            comparison["noaa_atlas_depth_mm"] > comparison["imerg_ci_upper_mm"],
        ],
        [
            np.nan,
            comparison["noaa_atlas_depth_mm"] - comparison["imerg_ci_lower_mm"],
            comparison["noaa_atlas_depth_mm"] - comparison["imerg_ci_upper_mm"],
        ],
        default=0.0,
    )
    overlap_width = (
        np.minimum(comparison["noaa_atlas_upper_mm"], comparison["imerg_ci_upper_mm"])
        - np.maximum(comparison["noaa_atlas_lower_mm"], comparison["imerg_ci_lower_mm"])
    )
    comparison["noaa_and_imerg_intervals_overlap"] = pd.array(
        np.where(finite_bounds, overlap_width >= 0, pd.NA), dtype="boolean"
    )
    comparison["interval_overlap_width_mm"] = np.where(
        finite_bounds, np.maximum(overlap_width, 0), np.nan
    )
    comparison["imerg_grid_latitude"] = float(imerg["latitude"])
    comparison["imerg_grid_longitude"] = float(imerg["longitude"])
    comparison["pot_threshold_percentile"] = reference_percentile
    comparison["noaa_duration"] = duration
    comparison["noaa_series"] = series.upper()
    comparison["noaa_atlas"] = f"NOAA Atlas 14 Volume {atlas['volume']}"
    comparison["noaa_atlas_version"] = atlas["version"]
    comparison["noaa_region"] = atlas["region"]
    comparison["noaa_source_url"] = atlas["source_url"]
    comparison["imerg_confidence_level"] = confidence_level
    comparison["comparison_caveat"] = (
        "IMERG is a 0.1-degree calendar-day grid estimate; NOAA Atlas is a "
        "point-frequency estimate for a sliding 24-hour duration."
    )
    comparison.to_csv(output_dir / "noaa_atlas_comparison.csv", index=False)
    comparison.to_csv(output_dir / "hunt_tx_noaa_imerg_comparison.csv", index=False)

    periods = comparison["return_period_years"].to_numpy()
    fig, (ax, coverage_ax) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    noaa = comparison["noaa_atlas_depth_mm"].to_numpy()
    noaa_error = np.vstack(
        [noaa - comparison["noaa_atlas_lower_mm"], comparison["noaa_atlas_upper_mm"] - noaa]
    )
    imerg_values = comparison["imerg_pot_depth_mm"].to_numpy()
    imerg_error = np.vstack(
        [
            imerg_values - comparison["imerg_ci_lower_mm"],
            comparison["imerg_ci_upper_mm"] - imerg_values,
        ]
    )
    ax.errorbar(
        periods, noaa, yerr=noaa_error, marker="s", capsize=5,
        linewidth=2, label=f"NOAA Atlas 14 Vol. {atlas['volume']} (published bounds)",
    )
    ax.errorbar(
        periods, imerg_values, yerr=imerg_error, marker="o", capsize=5,
        linewidth=2,
        label=(
            f"IMERG POT {reference_percentile:g}th percentile "
            f"({confidence_level * 100:g}% bootstrap CI)"
        ),
    )
    ax.set_xscale("log")
    ax.set_xticks(periods, [f"{period:g}" for period in periods])
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel(f"{duration} precipitation depth (mm)")
    ax.set_title(
        f"IMERG POT versus NOAA Atlas at {location_name}\n"
        f"NOAA coordinate: {float(atlas['latitude']):.3f}, {float(atlas['longitude']):.3f}"
    )
    ax.grid(alpha=0.3)
    ax.legend()

    positions = np.arange(len(comparison))
    imerg_lower = comparison["imerg_ci_lower_mm"].to_numpy(dtype=float)
    imerg_upper = comparison["imerg_ci_upper_mm"].to_numpy(dtype=float)
    status = comparison["noaa_position_relative_to_imerg_ci"].to_numpy()
    colors = [
        "tab:green" if value == "within" else "tab:red"
        if value in {"below", "above"} else "0.5"
        for value in status
    ]
    coverage_ax.hlines(
        positions, imerg_lower, imerg_upper, color="tab:blue", linewidth=8,
        alpha=0.35, label=f"IMERG {confidence_level * 100:g}% bootstrap CI",
    )
    coverage_ax.plot(
        imerg_values, positions, "o", color="tab:blue", label="IMERG POT estimate"
    )
    coverage_ax.errorbar(
        noaa, positions, xerr=noaa_error, fmt="none", ecolor="0.5",
        capsize=4, alpha=0.65, label="NOAA published bounds",
    )
    coverage_ax.scatter(noaa, positions, marker="s", c=colors, zorder=3)
    for x_value, y_value, label in zip(noaa, positions, status):
        coverage_ax.annotate(
            label, (x_value, y_value), xytext=(7, 0),
            textcoords="offset points", va="center", fontsize=9,
        )
    coverage_ax.set_yticks(positions, [f"{period:g}-year" for period in periods])
    coverage_ax.invert_yaxis()
    coverage_ax.set_xlabel(f"{duration} precipitation depth (mm)")
    coverage_ax.set_title("Does the NOAA estimate fall within the IMERG CI?")
    coverage_ax.grid(axis="x", alpha=0.3)
    coverage_ax.legend(fontsize=9)
    fig.tight_layout()
    for filename in (
        f"noaa_atlas_comparison.{figure_format}",
        f"hunt_tx_noaa_imerg_comparison.{figure_format}",
    ):
        fig.savefig(output_dir / filename, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return comparison
