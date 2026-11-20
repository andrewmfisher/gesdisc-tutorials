"""Modular peak-over-threshold analysis with threshold sensitivity plots.

For a normal run, edit only the USER CONFIGURATION section below. Threshold
percentiles are calculated independently at each grid cell from strictly
positive precipitation observations, matching the original script.
"""

from __future__ import annotations

import math
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pyextremes import EVA
from scipy import stats
from tqdm import tqdm

from noaa_atlas_comparison import (
    compare_domain_with_noaa_atlas,
    compare_with_noaa_atlas,
    summarize_domain_confidence_comparison,
)


# =============================================================================
# USER CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    """User-adjustable analysis settings."""

    input_file: Path = PROJECT_DIR / "IMERG_subsets" / "subset_kerchunk.nc"
    output_dir: Path = PROJECT_DIR / "Data" / "pot_threshold_sensitivity"
    precipitation_variable: str = "precipitation"
    # None lets xarray choose an installed backend; set "netcdf4" if required.
    netcdf_engine: str | None = None

    threshold_percentiles: tuple[float, ...] = (90,)
    # threshold_percentiles: tuple[float, ...] = (50, 60, 70, 80, 90, 95, 98)
    reference_percentile: float = 90.0
    declustering_window: str = "48hr"
    return_periods_years: tuple[float, ...] = (2.0, 5.0, 10.0, 100.0)
    return_period_size: str = "365.2425D"
    minimum_nonzero_observations: int = 50
    minimum_extreme_events: int = 15

    # AD is calculated for every fit. Set bootstrap samples to at least 199 for
    # a calibrated p-value; zero avoids a large runtime increase on full grids.
    fit_significance_level: float = 0.05
    fit_qq_r_squared_minimum: float = 0.95
    fit_ad_bootstrap_samples: int = 0
    fit_random_seed: int = 42

    calculate_confidence_intervals: bool = True
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000 # try 200
    # Bootstrap only the default/reference threshold to keep sensitivity runs tractable.
    bootstrap_reference_only: bool = True

    # Use small index slices, such as slice(0, 3), for a quick test run.
    latitude_slice: slice | None = None
    longitude_slice: slice | None = None

    location_name: str = "Hunt, TX"
    location_latitude: float = 30.07
    location_longitude: float = -99.3

    save_models: bool = True
    save_excel_summary: bool = True
    make_reference_maps: bool = True
    make_case_maps: bool = True
    make_difference_maps: bool = True
    make_sensitivity_curves: bool = True
    compare_noaa_atlas: bool = True
    compare_noaa_maps: bool = True
    noaa_duration: str = "24-hr"
    noaa_series: str = "pds"
    # Approximate TP-29 factor for a ~40 mi2 (105 km2), 24-hour area.
    noaa_areal_reduction_factor: float = 0.95
    show_plots: bool = False
    figure_dpi: int = 72
    map_cmap: str = "YlGnBu"
    difference_cmap: str = "RdBu_r"
    # Robust limits prevent a few extreme grid cells from flattening the maps.
    # Values outside these percentiles remain present and use colorbar end colors.
    color_scale_percentiles: tuple[float, float] = (2.0, 98.0)
    figure_format: str = "png"


CONFIG = Config()

# =============================================================================
# END USER CONFIGURATION
# =============================================================================


@dataclass
class ThresholdResult:
    """Models and gridded results for one percentile."""

    models: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    return_levels: dict[float, np.ndarray] = field(default_factory=dict)
    return_levels_lower: dict[float, np.ndarray] = field(default_factory=dict)
    return_levels_upper: dict[float, np.ndarray] = field(default_factory=dict)
    n_extremes: np.ndarray | None = None
    thresholds_mm: np.ndarray | None = None


def validate_config(config: Config) -> None:
    """Validate settings before starting a potentially long run."""
    percentiles = config.threshold_percentiles
    if not percentiles or len(set(percentiles)) != len(percentiles):
        raise ValueError("threshold_percentiles must be nonempty and unique")
    if any(not 0 < value < 100 for value in percentiles):
        raise ValueError("threshold percentiles must be between 0 and 100")
    if config.reference_percentile not in percentiles:
        raise ValueError("reference_percentile must be in threshold_percentiles")
    if any(value <= 0 for value in config.return_periods_years):
        raise ValueError("return periods must be positive")
    if config.minimum_nonzero_observations < 1 or config.minimum_extreme_events < 1:
        raise ValueError("minimum observation and event counts must be positive")
    if not 0 < config.confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if not 0 < config.fit_significance_level < 1:
        raise ValueError("fit_significance_level must be between 0 and 1")
    if not 0 <= config.fit_qq_r_squared_minimum <= 1:
        raise ValueError("fit_qq_r_squared_minimum must be between 0 and 1")
    if config.fit_ad_bootstrap_samples < 0:
        raise ValueError("fit_ad_bootstrap_samples cannot be negative")
    if not 0 < config.noaa_areal_reduction_factor <= 1:
        raise ValueError("noaa_areal_reduction_factor must be greater than 0 and at most 1")
    color_low, color_high = config.color_scale_percentiles
    if not 0 <= color_low < color_high <= 100:
        raise ValueError("color_scale_percentiles must be ordered within 0 to 100")


def percentile_label(percentile: float) -> str:
    """Create a filename-safe label, for example p92p5."""
    return f"p{percentile:g}".replace(".", "p")


def load_dataset(config: Config) -> xr.Dataset:
    """Open the input data and apply optional index subsets."""
    if not config.input_file.exists():
        raise FileNotFoundError(config.input_file)
    kwargs = {"engine": config.netcdf_engine} if config.netcdf_engine else {}
    dataset = xr.open_dataset(config.input_file, **kwargs)
    if config.precipitation_variable not in dataset:
        dataset.close()
        raise KeyError(f"Missing variable {config.precipitation_variable!r}")
    if config.latitude_slice is not None:
        dataset = dataset.isel(lat=config.latitude_slice)
    if config.longitude_slice is not None:
        dataset = dataset.isel(lon=config.longitude_slice)
    if len(dataset.lat) == 0 or len(dataset.lon) == 0:
        dataset.close()
        raise ValueError("The spatial subset contains no grid cells")
    return dataset


def extract_wet_series(
    dataset: xr.Dataset, lat_index: int, lon_index: int, variable: str
) -> pd.Series:
    """Return finite, positive precipitation at one grid cell."""
    series = dataset[variable].isel(lat=lat_index, lon=lon_index).to_series()
    series = series[np.isfinite(series) & (series > 0)].astype(float)
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def allocate_result(shape: tuple[int, int], config: Config) -> ThresholdResult:
    """Allocate output arrays for one threshold case."""
    result = ThresholdResult(
        return_levels={
            period: np.full(shape, np.nan) for period in config.return_periods_years
        },
        n_extremes=np.full(shape, np.nan),
        thresholds_mm=np.full(shape, np.nan),
    )
    if config.calculate_confidence_intervals:
        result.return_levels_lower = {
            period: np.full(shape, np.nan) for period in config.return_periods_years
        }
        result.return_levels_upper = {
            period: np.full(shape, np.nan) for period in config.return_periods_years
        }
    return result


def as_scalar(value: Any) -> float:
    """Convert a pyextremes result to a scalar float."""
    if isinstance(value, tuple):
        value = value[0]
    return float(np.asarray(value).reshape(-1)[0])


def calculate_return_levels(
    model: EVA, config: Config, include_uncertainty: bool
) -> tuple[dict[float, float], dict[float, float], dict[float, float]]:
    """Calculate point estimates and optional confidence bounds."""
    point: dict[float, float] = {}
    lower: dict[float, float] = {}
    upper: dict[float, float] = {}
    for period in config.return_periods_years:
        kwargs: dict[str, Any] = {
            "return_period": period,
            "return_period_size": config.return_period_size,
        }
        if include_uncertainty:
            kwargs.update(alpha=config.confidence_level, n_samples=config.bootstrap_samples)
            estimate, low, high = model.get_return_value(**kwargs)
            point[period], lower[period], upper[period] = (
                as_scalar(estimate),
                as_scalar(low),
                as_scalar(high),
            )
        else:
            point[period] = as_scalar(model.get_return_value(**kwargs))
    return point, lower, upper


def anderson_darling_statistic(cdf_values: np.ndarray) -> float:
    """Calculate the two-sided AD statistic from probability integral values."""
    values = np.sort(np.asarray(cdf_values, dtype=float))
    values = np.clip(values, np.finfo(float).eps, 1 - np.finfo(float).eps)
    indices = np.arange(1, len(values) + 1)
    terms = (2 * indices - 1) * (
        np.log(values) + np.log1p(-values[::-1])
    )
    return float(-len(values) - np.sum(terms) / len(values))


def bootstrap_ad_pvalue(
    model: EVA, threshold: float, observed_statistic: float, config: Config
) -> float:
    """Calibrate the GPD AD statistic by parametric bootstrap and refitting."""
    samples = config.fit_ad_bootstrap_samples
    if samples == 0:
        return np.nan

    parameters = model.model.fit_parameters
    shape = float(parameters["c"])
    scale = float(parameters["scale"])
    sample_size = len(model.extremes)
    rng = np.random.default_rng(config.fit_random_seed)
    bootstrap_statistics = np.empty(samples)
    for index in range(samples):
        simulated = stats.genpareto.rvs(
            shape, loc=threshold, scale=scale, size=sample_size, random_state=rng
        )
        fitted_shape, _, fitted_scale = stats.genpareto.fit(
            simulated, floc=threshold
        )
        fitted_cdf = stats.genpareto.cdf(
            simulated, fitted_shape, loc=threshold, scale=fitted_scale
        )
        bootstrap_statistics[index] = anderson_darling_statistic(fitted_cdf)
    return float(
        (1 + np.count_nonzero(bootstrap_statistics >= observed_statistic))
        / (samples + 1)
    )


def calculate_fit_diagnostics(
    model: EVA, threshold: float, config: Config
) -> dict[str, Any]:
    """Calculate complementary diagnostics for a fitted extreme-value model.

    AD emphasizes tail discrepancies. Its p-value is only populated when the
    configured parametric bootstrap refits the GPD for each simulated sample.
    The KS p-value remains an explicitly approximate secondary diagnostic.
    """
    observed = np.sort(model.extremes.to_numpy(dtype=float))
    probabilities = (np.arange(len(observed), dtype=float) + 0.5) / len(observed)
    fitted_quantiles = np.asarray(model.model.ppf(probabilities), dtype=float)
    probability_integral_transform = np.asarray(
        model.model.cdf(observed), dtype=float
    )

    ks_result = stats.kstest(probability_integral_transform, "uniform")
    ad_statistic = anderson_darling_statistic(probability_integral_transform)
    ad_pvalue = bootstrap_ad_pvalue(model, threshold, ad_statistic, config)
    residuals = observed - fitted_quantiles
    qq_rmse = float(np.sqrt(np.mean(residuals**2)))
    if np.std(observed) > 0 and np.std(fitted_quantiles) > 0:
        qq_r_squared = float(np.corrcoef(observed, fitted_quantiles)[0, 1] ** 2)
    else:
        qq_r_squared = np.nan

    test_rejects = (
        np.isfinite(ad_pvalue) and ad_pvalue < config.fit_significance_level
    )
    qq_needs_review = (
        not np.isfinite(qq_r_squared)
        or qq_r_squared < config.fit_qq_r_squared_minimum
    )
    if test_rejects:
        fit_flag = "possible_lack_of_fit"
    elif qq_needs_review:
        fit_flag = "review_qq_fit"
    elif not np.isfinite(ad_pvalue):
        fit_flag = "no_qq_issue_ad_not_bootstrapped"
    else:
        fit_flag = "no_issue_detected"

    return {
        "fit_aic": float(model.model.AIC),
        "fit_loglikelihood": float(model.model.loglikelihood),
        "fit_ks_statistic": float(ks_result.statistic),
        "fit_ks_pvalue_approx": float(ks_result.pvalue),
        "fit_ad_statistic": ad_statistic,
        "fit_ad_pvalue_bootstrap": ad_pvalue,
        "fit_ad_bootstrap_samples": config.fit_ad_bootstrap_samples,
        "fit_qq_rmse_mm": qq_rmse,
        "fit_qq_r_squared": qq_r_squared,
        "fit_flag": fit_flag,
    }


def fit_pot_case(
    wet_series: pd.Series,
    percentile: float,
    config: Config,
    include_uncertainty: bool,
) -> tuple[EVA, float, dict[float, float], dict[float, float], dict[float, float]]:
    """Fit one POT model and calculate its return levels."""
    threshold = float(np.percentile(wet_series.to_numpy(), percentile))
    model = EVA(data=wet_series)
    model.get_extremes(
        method="POT", threshold=threshold, r=config.declustering_window
    )
    if len(model.extremes) < config.minimum_extreme_events:
        raise ValueError(
            f"only {len(model.extremes)} extremes; minimum is "
            f"{config.minimum_extreme_events}"
        )
    model.fit_model(distribution="genpareto")
    point, lower, upper = calculate_return_levels(model, config, include_uncertainty)
    return model, threshold, point, lower, upper


def analyse_thresholds(
    dataset: xr.Dataset, config: Config
) -> tuple[dict[float, ThresholdResult], pd.DataFrame]:
    """Fit every percentile at every cell in one spatial pass."""
    shape = (len(dataset.lat), len(dataset.lon))
    results = {
        percentile: allocate_result(shape, config)
        for percentile in config.threshold_percentiles
    }
    rows: list[dict[str, Any]] = []
    total = shape[0] * shape[1] * len(config.threshold_percentiles)

    with tqdm(total=total, desc="Fitting POT cases", unit="fit") as progress:
        for lat_index in range(shape[0]):
            for lon_index in range(shape[1]):
                latitude = float(dataset.lat.values[lat_index])
                longitude = float(dataset.lon.values[lon_index])
                wet = extract_wet_series(
                    dataset, lat_index, lon_index, config.precipitation_variable
                )
                for percentile in config.threshold_percentiles:
                    row: dict[str, Any] = {
                        "threshold_percentile": percentile,
                        "lat_index": lat_index,
                        "lon_index": lon_index,
                        "latitude": latitude,
                        "longitude": longitude,
                        "n_nonzero_precip": len(wet),
                        "date_start": wet.index.min() if len(wet) else pd.NaT,
                        "date_end": wet.index.max() if len(wet) else pd.NaT,
                        "status": "failed",
                        "error": "",
                    }
                    progress.set_postfix(
                        lat=f"{latitude:.2f}", lon=f"{longitude:.2f}",
                        percentile=f"{percentile:g}", refresh=False,
                    )
                    try:
                        if len(wet) < config.minimum_nonzero_observations:
                            raise ValueError(
                                f"only {len(wet)} nonzero observations; minimum is "
                                f"{config.minimum_nonzero_observations}"
                            )
                        include_uncertainty = config.calculate_confidence_intervals and (
                            not config.bootstrap_reference_only
                            or percentile == config.reference_percentile
                        )
                        model, threshold, point, lower, upper = fit_pot_case(
                            wet, percentile, config, include_uncertainty
                        )
                        fit_diagnostics = calculate_fit_diagnostics(
                            model, threshold, config
                        )
                        result = results[percentile]
                        result.thresholds_mm[lat_index, lon_index] = threshold
                        result.n_extremes[lat_index, lon_index] = len(model.extremes)
                        for period, value in point.items():
                            result.return_levels[period][lat_index, lon_index] = value
                        for period, value in lower.items():
                            result.return_levels_lower[period][lat_index, lon_index] = value
                        for period, value in upper.items():
                            result.return_levels_upper[period][lat_index, lon_index] = value

                        model_info = {
                            "model": model,
                            "lat_index": lat_index,
                            "lon_index": lon_index,
                            "latitude": latitude,
                            "longitude": longitude,
                            "threshold_percentile": percentile,
                            "threshold_mm": threshold,
                            "n_extremes": len(model.extremes),
                            "n_nonzero": len(wet),
                            "date_range": (wet.index.min(), wet.index.max()),
                            "fit_parameters": dict(model.model.fit_parameters),
                        }
                        result.models[(lat_index, lon_index)] = model_info
                        row.update(
                            status="ok", threshold_mm=threshold,
                            n_extreme_events=len(model.extremes),
                        )
                        for name, value in model.model.fit_parameters.items():
                            row[f"param_{name}"] = value
                        row.update(fit_diagnostics)
                        for period, value in point.items():
                            row[f"return_level_{period:g}yr_mm"] = value
                        for period, value in lower.items():
                            row[f"return_level_{period:g}yr_lower_mm"] = value
                        for period, value in upper.items():
                            row[f"return_level_{period:g}yr_upper_mm"] = value
                        if include_uncertainty:
                            row["uncertainty_interval_coverage"] = config.confidence_level
                            row["uncertainty_bootstrap_samples"] = config.bootstrap_samples
                    except Exception as error:  # keep other cases running
                        row["error"] = f"{type(error).__name__}: {error}"
                    rows.append(row)
                    progress.update()
    return results, pd.DataFrame(rows)


def save_results(
    results: dict[float, ThresholdResult], summary: pd.DataFrame, config: Config
) -> None:
    """Save combined tables, gridded arrays, and optional fitted models."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.output_dir / "pot_threshold_sensitivity_summary.csv", index=False)
    if config.save_excel_summary:
        excel_path = config.output_dir / "pot_threshold_sensitivity_summary.xlsx"
        identifier_columns = [
            "threshold_percentile", "lat_index", "lon_index", "latitude",
            "longitude", "threshold_mm", "n_extreme_events", "status", "error",
        ]
        diagnostic_columns = [
            column for column in summary.columns if column.startswith("fit_")
        ]
        fit_table = summary[
            [column for column in identifier_columns if column in summary]
            + diagnostic_columns
        ]
        try:
            with pd.ExcelWriter(excel_path) as writer:
                summary.to_excel(writer, index=False, sheet_name="POT Sensitivity")
                fit_table.to_excel(writer, index=False, sheet_name="Fit Diagnostics")
        except ImportError as error:
            warnings.warn(
                f"Excel summary skipped ({error}). The complete CSV was saved.",
                stacklevel=2,
            )
    for percentile, result in results.items():
        label = percentile_label(percentile)
        arrays: dict[str, np.ndarray] = {
            "threshold_mm": result.thresholds_mm,
            "n_extremes": result.n_extremes,
        }
        for period, values in result.return_levels.items():
            arrays[f"return_level_{period:g}yr_mm"] = values
        for period, values in result.return_levels_lower.items():
            arrays[f"return_level_{period:g}yr_lower_mm"] = values
        for period, values in result.return_levels_upper.items():
            arrays[f"return_level_{period:g}yr_upper_mm"] = values
        np.savez_compressed(config.output_dir / f"pot_results_{label}.npz", **arrays)
        if config.save_models:
            path = config.output_dir / f"eva_models_all_gridcells_{label}.pkl"
            with path.open("wb") as file:
                pickle.dump(result.models, file)


def add_map_context(ax: Any, dataset: xr.Dataset, config: Config) -> None:
    """Add geographic features and the configured point to a map."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    ax.add_feature(cfeature.STATES, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.25)
    ax.add_feature(cfeature.RIVERS, linewidth=0.35, alpha=0.5)
    ax.plot(
        config.location_longitude, config.location_latitude, marker="*", color="red",
        markeredgecolor="black", markersize=12, transform=ccrs.PlateCarree(), zorder=10,
    )
    ax.text(
        config.location_longitude + 0.08, config.location_latitude,
        config.location_name, color="red", fontsize=8,
        transform=ccrs.PlateCarree(),
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "red"}, zorder=11,
    )
    ax.set_extent(
        [float(dataset.lon.min()) - 0.5, float(dataset.lon.max()) + 0.5,
         float(dataset.lat.min()) - 0.5, float(dataset.lat.max()) + 0.5],
        crs=ccrs.PlateCarree(),
    )
    grid = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.5, linestyle="--")
    grid.top_labels = grid.right_labels = False


def save_figure(fig: plt.Figure, path: Path, config: Config) -> None:
    """Save, optionally display, then close a figure."""
    fig.savefig(path, dpi=config.figure_dpi, bbox_inches="tight")
    if config.show_plots:
        plt.show()
    plt.close(fig)


def plot_map_panels(
    dataset: xr.Dataset,
    panels: list[tuple[str, np.ndarray]],
    title: str,
    colorbar_label: str,
    output_path: Path,
    config: Config,
    cmap: str,
    symmetric: bool = False,
    shared_scale: bool = True,
    columns: int | None = None,
) -> None:
    """Plot maps using robust shared or panel-specific color scales."""
    import cartopy.crs as ccrs

    columns = min(columns or 3, len(panels))
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, 5 * rows), squeeze=False,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    def color_limits(values: np.ndarray) -> tuple[float | None, float | None]:
        finite = values[np.isfinite(values)]
        if not finite.size:
            return None, None
        low, high = config.color_scale_percentiles
        if symmetric:
            limit = float(np.percentile(np.abs(finite), high))
            return (-limit, limit) if limit else (-1.0, 1.0)
        vmin, vmax = np.percentile(finite, [low, high])
        if np.isclose(vmin, vmax):
            padding = abs(float(vmin)) * 0.01 or 1.0
            return float(vmin - padding), float(vmax + padding)
        return float(vmin), float(vmax)

    shared_limits = (None, None)
    if shared_scale:
        finite = [values[np.isfinite(values)] for _, values in panels]
        finite = [values for values in finite if values.size]
        if finite:
            shared_limits = color_limits(np.concatenate(finite))

    images = []
    for ax, (panel_title, values) in zip(axes.flat, panels):
        vmin, vmax = shared_limits if shared_scale else color_limits(values)
        image = None
        if np.any(np.isfinite(values)):
            image = ax.pcolormesh(
                dataset.lon, dataset.lat, values, transform=ccrs.PlateCarree(),
                cmap=cmap, shading="auto", vmin=vmin, vmax=vmax,
            )
            images.append(image)
            if not shared_scale:
                colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.03, extend="both")
                colorbar.set_label(colorbar_label)
        add_map_context(ax, dataset, config)
        ax.set_title(panel_title)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    if shared_scale and images:
        colorbar = fig.colorbar(
            images[-1], ax=axes.ravel().tolist(), shrink=0.8, pad=0.03, extend="both"
        )
        colorbar.set_label(colorbar_label)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    save_figure(fig, output_path, config)


def plot_reference_maps(
    dataset: xr.Dataset, results: dict[float, ThresholdResult], config: Config
) -> None:
    """Plot all return periods for the reference percentile."""
    result = results[config.reference_percentile]
    panels = [
        (f"{period:g}-year return level", result.return_levels[period])
        for period in config.return_periods_years
    ]
    plot_map_panels(
        dataset, panels,
        f"POT return levels: {config.reference_percentile:g}th-percentile threshold",
        "Precipitation (mm)",
        config.output_dir / (
            f"return_levels_reference_{percentile_label(config.reference_percentile)}."
            f"{config.figure_format}"
        ),
        config, config.map_cmap, shared_scale=False, columns=2,
    )


def plot_reference_confidence_maps(
    dataset: xr.Dataset, results: dict[float, ThresholdResult], config: Config
) -> None:
    """Plot lower bound, upper bound, and width of bootstrap confidence intervals."""
    result = results[config.reference_percentile]
    coverage = config.confidence_level * 100
    products = {
        "lower": (
            result.return_levels_lower,
            f"{coverage:g}% CI lower bound",
            "Lower confidence bound (mm)",
        ),
        "upper": (
            result.return_levels_upper,
            f"{coverage:g}% CI upper bound",
            "Upper confidence bound (mm)",
        ),
        "width": (
            {
                period: result.return_levels_upper[period]
                - result.return_levels_lower[period]
                for period in config.return_periods_years
            },
            f"{coverage:g}% CI width",
            "Confidence interval width (mm)",
        ),
    }
    label = percentile_label(config.reference_percentile)
    for product_name, (values_by_period, title_text, colorbar_label) in products.items():
        panels = [
            (f"{period:g}-year {title_text}", values_by_period[period])
            for period in config.return_periods_years
        ]
        plot_map_panels(
            dataset,
            panels,
            f"Bootstrap {title_text}: {config.reference_percentile:g}th-percentile threshold",
            colorbar_label,
            config.output_dir
            / f"return_level_ci_{product_name}_reference_{label}.{config.figure_format}",
            config,
            config.map_cmap,
            shared_scale=False,
            columns=2,
        )


def plot_case_and_difference_maps(
    dataset: xr.Dataset, results: dict[float, ThresholdResult], config: Config
) -> None:
    """Plot absolute cases and changes relative to the reference threshold."""
    reference = results[config.reference_percentile]
    for period in config.return_periods_years:
        cases = [
            (f"Threshold: {p:g}th percentile", results[p].return_levels[period])
            for p in config.threshold_percentiles
        ]
        if config.make_case_maps:
            plot_map_panels(
                dataset, cases, f"{period:g}-year POT return level by threshold",
                "Precipitation (mm)",
                config.output_dir / f"return_level_cases_{period:g}yr.{config.figure_format}",
                config, config.map_cmap,
            )
        if config.make_difference_maps:
            differences = [
                (f"{p:g}th minus {config.reference_percentile:g}th",
                 results[p].return_levels[period] - reference.return_levels[period])
                for p in config.threshold_percentiles
            ]
            plot_map_panels(
                dataset, differences,
                f"Change in {period:g}-year return level by threshold",
                "Difference from reference (mm)",
                config.output_dir / f"return_level_differences_{period:g}yr.{config.figure_format}",
                config, config.difference_cmap, symmetric=True,
            )


def plot_sensitivity_curves(
    dataset: xr.Dataset, results: dict[float, ThresholdResult], config: Config
) -> None:
    """Plot location-specific and domain-wide threshold responses."""
    percentiles = np.asarray(config.threshold_percentiles)
    lat_index = int(np.argmin(np.abs(dataset.lat.values - config.location_latitude)))
    lon_index = int(np.argmin(np.abs(dataset.lon.values - config.location_longitude)))
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    for period in config.return_periods_years:
        location_values = [
            results[p].return_levels[period][lat_index, lon_index]
            for p in config.threshold_percentiles
        ]
        axes[0].plot(percentiles, location_values, marker="o", label=f"{period:g}-year")
        quartiles = []
        for p in config.threshold_percentiles:
            values = results[p].return_levels[period]
            values = values[np.isfinite(values)]
            quartiles.append(np.percentile(values, [25, 50, 75]) if values.size else [np.nan] * 3)
        quartiles = np.asarray(quartiles)
        axes[2].plot(percentiles, quartiles[:, 1], marker="o", label=f"{period:g}-year")
        axes[2].fill_between(percentiles, quartiles[:, 0], quartiles[:, 2], alpha=0.12)

    event_counts = [
        results[p].n_extremes[lat_index, lon_index]
        for p in config.threshold_percentiles
    ]
    axes[1].plot(percentiles, event_counts, marker="o", color="tab:purple")
    actual_lat = float(dataset.lat.values[lat_index])
    actual_lon = float(dataset.lon.values[lon_index])
    axes[0].set_title(
        f"Return levels near {config.location_name}\nGrid cell: {actual_lat:.3f}, {actual_lon:.3f}"
    )
    axes[0].set_ylabel("Return level (mm)")
    axes[0].legend()
    axes[1].set_title(f"Independent extremes near {config.location_name}")
    axes[1].set_ylabel("Number of events")
    axes[2].set_title("Domain median and interquartile range")
    axes[2].set_ylabel("Return level (mm)")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("POT threshold percentile (positive precipitation)")
        ax.axvline(config.reference_percentile, color="black", linestyle="--", alpha=0.6)
        ax.grid(alpha=0.3)
    fig.suptitle("POT threshold sensitivity", fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_figure(
        fig, config.output_dir / f"threshold_sensitivity_curves.{config.figure_format}", config
    )


def plot_noaa_domain_comparison(
    dataset: xr.Dataset,
    results: dict[float, ThresholdResult],
    summary: pd.DataFrame,
    config: Config,
) -> None:
    """Create 2x2 NOAA Atlas and IMERG-minus-NOAA domain maps."""
    imerg = results[config.reference_percentile].return_levels
    noaa_maps, difference_maps = compare_domain_with_noaa_atlas(
        latitudes=dataset.lat.values,
        longitudes=dataset.lon.values,
        imerg_return_levels=imerg,
        return_periods=config.return_periods_years,
        output_dir=config.output_dir,
        duration=config.noaa_duration,
        areal_reduction_factor=config.noaa_areal_reduction_factor,
    )
    noaa_panels = [
        (f"NOAA {period:g}-year, {config.noaa_duration}", noaa_maps[period])
        for period in config.return_periods_years
    ]
    plot_map_panels(
        dataset,
        noaa_panels,
        "Area-adjusted NOAA Atlas 14 Volume 11 precipitation-frequency depths",
        "NOAA areal precipitation depth (mm)",
        config.output_dir / f"noaa_atlas_domain_maps.{config.figure_format}",
        config,
        config.map_cmap,
        shared_scale=False,
        columns=2,
    )
    summarize_domain_confidence_comparison(
        summary=summary,
        domain_comparison=pd.read_csv(
            config.output_dir / "noaa_atlas_domain_comparison.csv"
        ),
        reference_percentile=config.reference_percentile,
        return_periods=config.return_periods_years,
        output_dir=config.output_dir,
        confidence_level=config.confidence_level,
        figure_dpi=config.figure_dpi,
        figure_format=config.figure_format,
    )
    difference_panels = [
        (f"{period:g}-year IMERG POT minus NOAA", difference_maps[period])
        for period in config.return_periods_years
    ]
    plot_map_panels(
        dataset,
        difference_panels,
        "IMERG POT difference from area-adjusted NOAA Atlas 14",
        "IMERG minus areal NOAA (mm)",
        config.output_dir / f"noaa_atlas_domain_differences.{config.figure_format}",
        config,
        config.difference_cmap,
        symmetric=True,
        shared_scale=False,
        columns=2,
    )


def print_run_summary(summary: pd.DataFrame, config: Config) -> None:
    """Print fitting success and domain median return levels."""
    successful = summary[summary["status"] == "ok"]
    print(f"\nSuccessful fits: {len(successful)}/{len(summary)}")
    failures = summary.loc[summary["status"] != "ok", "error"].value_counts()
    if not failures.empty:
        print("Most common fitting failures:")
        for message, count in failures.head(5).items():
            print(f"  {count:>5}  {message}")
    if "fit_flag" in successful:
        print("\nFit diagnostic flags:")
        for flag, count in successful["fit_flag"].value_counts().items():
            print(f"  {count:>5}  {flag}")
    print("\nDomain median return levels (mm):")
    for percentile in config.threshold_percentiles:
        case = successful[successful["threshold_percentile"] == percentile]
        values = []
        for period in config.return_periods_years:
            column = f"return_level_{period:g}yr_mm"
            median = case[column].median() if column in case else np.nan
            values.append(f"{period:g}yr={median:.2f}")
        print(f"  {percentile:g}th percentile: " + ", ".join(values))
    print(f"\nOutputs saved to: {config.output_dir}")


def main(config: Config = CONFIG) -> None:
    """Run all configured POT sensitivity cases and outputs."""
    validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {config.input_file}")
    dataset = load_dataset(config)
    try:
        cells = len(dataset.lat) * len(dataset.lon)
        print(
            f"Analysing {cells} grid cells at percentiles "
            f"{list(config.threshold_percentiles)}"
        )
        results, summary = analyse_thresholds(dataset, config)
        save_results(results, summary, config)
        if config.make_reference_maps:
            plot_reference_maps(dataset, results, config)
            if config.calculate_confidence_intervals:
                plot_reference_confidence_maps(dataset, results, config)
        if config.make_case_maps or config.make_difference_maps:
            plot_case_and_difference_maps(dataset, results, config)
        if config.make_sensitivity_curves:
            plot_sensitivity_curves(dataset, results, config)
        if config.compare_noaa_atlas:
            try:
                compare_with_noaa_atlas(
                    summary=summary,
                    reference_percentile=config.reference_percentile,
                    return_periods=config.return_periods_years,
                    latitude=config.location_latitude,
                    longitude=config.location_longitude,
                    location_name=config.location_name,
                    output_dir=config.output_dir,
                    duration=config.noaa_duration,
                    series=config.noaa_series,
                    confidence_level=config.confidence_level,
                    figure_dpi=config.figure_dpi,
                    figure_format=config.figure_format,
                )
            except Exception as error:
                warnings.warn(
                    f"NOAA Atlas comparison skipped: {type(error).__name__}: {error}",
                    stacklevel=2,
                )
        if config.compare_noaa_maps:
            try:
                plot_noaa_domain_comparison(dataset, results, summary, config)
            except Exception as error:
                warnings.warn(
                    f"NOAA Atlas domain comparison skipped: "
                    f"{type(error).__name__}: {error}",
                    stacklevel=2,
                )
        print_run_summary(summary, config)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
