"""Compare a local NOAA AMS fit, NOAA Atlas 14, and the saved IMERG POT fit.

NOAA Atlas 14 Volume 11 does not publish partial-duration event series. This
script follows its AMS-to-PDS frequency conversion: fit a distribution to a
station's annual maxima and evaluate it at ``F(x) = exp(-1 / T_pds)``. This is
Langbein's Poisson conversion between AMS annual exceedance probability and
PDS average recurrence interval.

The local fit is intentionally separate from ``pot_script.py``. It is not a
reproduction of the operational Atlas curve, which uses regional L-moments,
duration smoothing, and spatial interpolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from noaa_atlas_comparison import atlas_frequency_table, fetch_noaa_atlas


PROJECT_DIR = Path(__file__).resolve().parent.parent
AMS_URL = "https://hdsc.nws.noaa.gov/pub/hdsc/data/tx/dur01d_ams_na14v11.txt"


@dataclass(frozen=True)
class Config:
    output_dir: Path = PROJECT_DIR / "Data" / "noaa_ams_pds_comparison"
    imerg_summary_file: Path = (
        PROJECT_DIR
        / "Data"
        / "pot_threshold_sensitivity"
        / "pot_threshold_sensitivity_summary.csv"
    )
    station_identifier: str = "41-4374"
    station_name: str = "HUNT 3 SW"
    location_name: str = "Hunt, TX"
    location_latitude: float = 30.07
    location_longitude: float = -99.3
    reference_percentile: float = 90.0
    return_periods_years: tuple[float, ...] = (2.0, 5.0, 10.0, 100.0)
    confidence_level: float = 0.95
    bootstrap_samples: int = 300
    random_seed: int = 42
    noaa_duration: str = "24-hr"
    noaa_series: str = "pds"
    # NOAA Atlas 14 Vol. 11, Table 4.5.3: constrained 1-day to
    # unconstrained 24-hour annual maxima.
    constrained_to_unconstrained_factor: float = 1.11
    figure_dpi: int = 150


CONFIG = Config()


def download_ams_file(config: Config) -> Path:
    """Download and cache NOAA Atlas 14 Volume 11's official 1-day AMS file."""
    cache_dir = config.output_dir / "noaa_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / Path(AMS_URL).name
    if not path.exists():
        request = Request(AMS_URL, headers={"User-Agent": "gesdisc-tutorials AMS/PDS comparison"})
        with urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    return path


def read_station_ams(path: Path, station_identifier: str) -> tuple[dict, pd.Series]:
    """Extract one station block and convert NOAA's inches to millimetres."""
    header_pattern = re.compile(
        r"([^,]+),\s*(.*?),\s*([A-Z]{2}),\s*([+-]?\d+\.\d+),"
        r"\s*([+-]?\d+\.\d+),\s*(\d+)"
    )
    value_pattern = re.compile(r"(\d{2}/\d{2}/\d{4})\s+([\d.]+)")
    metadata = None
    records: list[tuple[pd.Timestamp, float]] = []
    active = False
    for raw_line in path.read_text().splitlines():
        header = header_pattern.fullmatch(raw_line.strip())
        if header:
            active = header.group(1).strip() == station_identifier
            if active:
                metadata = {
                    "station_identifier": header.group(1).strip(),
                    "station_name": header.group(2).strip(),
                    "state": header.group(3),
                    "latitude": float(header.group(4)),
                    "longitude": float(header.group(5)),
                    "elevation_ft": float(header.group(6)),
                }
            continue
        if active:
            if not raw_line.strip():
                break
            value = value_pattern.fullmatch(raw_line.strip())
            if value:
                records.append(
                    (pd.to_datetime(value.group(1)), float(value.group(2)) * 25.4)
                )
    if metadata is None or not records:
        raise ValueError(f"Station {station_identifier!r} was not found in {path}")
    series = pd.Series(
        [value for _, value in records],
        index=pd.DatetimeIndex([date for date, _ in records], name="date"),
        name="annual_maximum_mm",
    ).sort_index()
    return metadata, series


def pds_nonexceedance_probability(return_period_years: float) -> float:
    """Convert PDS ARI to the equivalent AMS nonexceedance probability."""
    return float(np.exp(-1.0 / return_period_years))


def fit_ams_and_convert_to_pds(
    annual_maxima: pd.Series, config: Config
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit a local GEV AMS model and bootstrap Langbein-converted PDS levels."""
    values = annual_maxima.to_numpy(dtype=float)
    shape, location, scale = stats.genextreme.fit(values)
    probabilities = np.array(
        [pds_nonexceedance_probability(period) for period in config.return_periods_years]
    )
    estimates = stats.genextreme.ppf(probabilities, shape, loc=location, scale=scale)

    rng = np.random.default_rng(config.random_seed)
    bootstrap = np.full((config.bootstrap_samples, len(probabilities)), np.nan)
    successful = 0
    for sample_index in range(config.bootstrap_samples):
        sample = rng.choice(values, size=len(values), replace=True)
        try:
            boot_shape, boot_location, boot_scale = stats.genextreme.fit(sample)
            levels = stats.genextreme.ppf(
                probabilities, boot_shape, loc=boot_location, scale=boot_scale
            )
            if np.all(np.isfinite(levels)):
                bootstrap[sample_index] = levels
                successful += 1
        except (ValueError, FloatingPointError):
            continue
    alpha = (1 - config.confidence_level) / 2
    lower = np.nanquantile(bootstrap, alpha, axis=0)
    upper = np.nanquantile(bootstrap, 1 - alpha, axis=0)
    table = pd.DataFrame(
        {
            "return_period_years": config.return_periods_years,
            "equivalent_ams_return_period_years": 1 / (1 - probabilities),
            "noaa_station_ams_to_pds_depth_mm": estimates,
            "noaa_station_ci_lower_mm": lower,
            "noaa_station_ci_upper_mm": upper,
        }
    )
    diagnostics = {
        "gev_shape_scipy": float(shape),
        "gev_location_mm": float(location),
        "gev_scale_mm": float(scale),
        "bootstrap_samples_requested": config.bootstrap_samples,
        "bootstrap_samples_successful": successful,
    }
    return table, diagnostics


def get_saved_imerg_comparison(config: Config) -> pd.DataFrame:
    """Read the nearest successful saved IMERG reference-threshold fit."""
    summary = pd.read_csv(config.imerg_summary_file)
    candidates = summary[
        (summary["threshold_percentile"] == config.reference_percentile)
        & (summary["status"] == "ok")
    ].copy()
    distance = (
        (candidates["latitude"] - config.location_latitude) ** 2
        + (candidates["longitude"] - config.location_longitude) ** 2
    )
    row = candidates.loc[distance.idxmin()]
    return pd.DataFrame(
        {
            "return_period_years": config.return_periods_years,
            "imerg_pot_depth_mm": [
                row[f"return_level_{period:g}yr_mm"]
                for period in config.return_periods_years
            ],
            "imerg_ci_lower_mm": [
                row[f"return_level_{period:g}yr_lower_mm"]
                for period in config.return_periods_years
            ],
            "imerg_ci_upper_mm": [
                row[f"return_level_{period:g}yr_upper_mm"]
                for period in config.return_periods_years
            ],
            "imerg_grid_latitude": float(row["latitude"]),
            "imerg_grid_longitude": float(row["longitude"]),
        }
    )


def plot_comparison(table: pd.DataFrame, metadata: dict, config: Config) -> None:
    """Plot all three frequency estimates and their uncertainty intervals."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    periods = table["return_period_years"].to_numpy()
    plot_ceiling = 1.1 * np.nanmax(
        table[
            [
                "noaa_atlas_upper_mm", "imerg_ci_upper_mm",
                "noaa_station_ams_to_pds_depth_mm",
            ]
        ].to_numpy(dtype=float)
    )
    series = (
        (
            "NOAA Atlas 14 published PDS",
            "noaa_atlas_depth_mm", "noaa_atlas_lower_mm", "noaa_atlas_upper_mm", "s",
        ),
        (
            f"{metadata['station_name']} corrected AMS→PDS",
            "noaa_station_ams_to_pds_depth_mm",
            "noaa_station_ci_lower_mm", "noaa_station_ci_upper_mm", "^",
        ),
        (
            f"IMERG POT {config.reference_percentile:g}th percentile",
            "imerg_pot_depth_mm", "imerg_ci_lower_mm", "imerg_ci_upper_mm", "o",
        ),
    )
    for label, value_column, lower_column, upper_column, marker in series:
        values = table[value_column].to_numpy(dtype=float)
        raw_upper = table[upper_column].to_numpy(dtype=float)
        displayed_upper = np.minimum(raw_upper, plot_ceiling)
        errors = np.vstack(
            [values - table[lower_column], displayed_upper - values]
        )
        ax.errorbar(
            periods, values, yerr=errors, marker=marker, capsize=5,
            linewidth=2, label=label,
        )
        if value_column == "noaa_station_ams_to_pds_depth_mm":
            for period, upper in zip(periods[raw_upper > plot_ceiling], raw_upper[raw_upper > plot_ceiling]):
                ax.scatter(period, plot_ceiling, marker="^", color="tab:orange", zorder=4)
                ax.annotate(
                    f"CI upper {upper:,.0f} mm", (period, plot_ceiling),
                    xytext=(0, -16), textcoords="offset points", ha="center",
                    va="top", fontsize=8, rotation=30,
                )
    ax.set_xscale("log")
    ax.set_xticks(periods, [f"{period:g}" for period in periods])
    ax.set_xlabel("PDS average recurrence interval (years)")
    ax.set_ylabel("1-day / 24-hour precipitation depth (mm)")
    ax.set_ylim(bottom=0, top=plot_ceiling * 1.03)
    ax.set_title(
        f"Local NOAA AMS→PDS, NOAA Atlas, and IMERG POT near {config.location_name}\n"
        f"Station: {metadata['station_name']} "
        f"({metadata['latitude']:.4f}, {metadata['longitude']:.4f})"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        config.output_dir / "noaa_ams_pds_imerg_comparison.png",
        dpi=config.figure_dpi, bbox_inches="tight",
    )
    plt.close(fig)


def main(config: Config = CONFIG) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    ams_path = download_ams_file(config)
    metadata, constrained_ams = read_station_ams(ams_path, config.station_identifier)
    annual_maxima = constrained_ams * config.constrained_to_unconstrained_factor
    annual_maxima.name = "unconstrained_24_hour_annual_maximum_mm"
    local_fit, diagnostics = fit_ams_and_convert_to_pds(annual_maxima, config)
    atlas_payload = fetch_noaa_atlas(
        config.location_latitude,
        config.location_longitude,
        config.output_dir / "noaa_cache",
        config.noaa_series,
    )
    atlas = atlas_frequency_table(
        atlas_payload, config.noaa_duration, config.return_periods_years
    )
    imerg = get_saved_imerg_comparison(config)
    comparison = local_fit.merge(atlas, on="return_period_years").merge(
        imerg, on="return_period_years"
    )
    comparison["station_identifier"] = metadata["station_identifier"]
    comparison["station_name"] = metadata["station_name"]
    comparison["station_latitude"] = metadata["latitude"]
    comparison["station_longitude"] = metadata["longitude"]
    comparison["station_ams_start"] = annual_maxima.index.min().date().isoformat()
    comparison["station_ams_end"] = annual_maxima.index.max().date().isoformat()
    comparison["station_ams_years"] = len(annual_maxima)
    comparison["constrained_to_unconstrained_factor"] = (
        config.constrained_to_unconstrained_factor
    )
    comparison["confidence_level"] = config.confidence_level
    comparison["method_note"] = (
        "NOAA constrained 1-day AMS multiplied by the Volume 11 Table 4.5.3 "
        "factor; local GEV MLE fit; PDS ARIs converted with F(x)=exp(-1/T). "
        "Operational Atlas estimates use regional L-moments."
    )
    for key, value in diagnostics.items():
        comparison[key] = value
    pd.DataFrame(
        {
            "constrained_1_day_annual_maximum_mm": constrained_ams,
            "unconstrained_24_hour_annual_maximum_mm": annual_maxima,
        }
    ).to_csv(config.output_dir / "hunt_3_sw_annual_maxima_mm.csv")
    comparison.to_csv(config.output_dir / "noaa_ams_pds_imerg_comparison.csv", index=False)
    plot_comparison(comparison, metadata, config)
    print(comparison.to_string(index=False))
    print(f"\nOutputs saved to {config.output_dir}")


if __name__ == "__main__":
    main()
