"""Plot every lazy-load and compute benchmark trial by time-series length."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    PROJECT_DIR
    / "Data"
    / "access_benchmark_results_20261113T170903_01h29m07s.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "Figures"
    / "access_benchmark_times_20261113T170903_01h29m07s.png"
)

METHOD_STYLES = {
    "Kerchunk": {
        "label": "Virtual Zarr (Kerchunk)", "color": "#0072B2", "marker": "o"
    },
    "OPeNDAP": {"label": "OPeNDAP", "color": "#D55E00", "marker": "s"},
}
REQUIRED_COLUMNS = {
    "Method", "Days", "Trial", "Lazy_Load_Time_s", "Compute_Time_s"
}


def load_results(path: Path) -> pd.DataFrame:
    """Load and validate one benchmark result table."""
    results = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for column in ("Days", "Trial", "Lazy_Load_Time_s", "Compute_Time_s"):
        results[column] = pd.to_numeric(results[column], errors="coerce")
    results = results.dropna(subset=["Method", "Days", "Trial", "Lazy_Load_Time_s"])
    unexpected = sorted(set(results["Method"]) - set(METHOD_STYLES))
    if unexpected:
        raise ValueError(f"Unexpected access methods: {unexpected}")
    if (results["Days"] <= 0).any():
        raise ValueError("Days must be positive")
    if results.duplicated(["Method", "Days", "Trial"]).any():
        raise ValueError("Duplicate Method/Days/Trial rows are present")
    return results.sort_values(["Method", "Trial", "Days"])


def plot_trials(
    ax: plt.Axes,
    results: pd.DataFrame,
    value_column: str,
    title: str,
) -> None:
    """Draw one connected series for every benchmark trial."""
    for method, style in METHOD_STYLES.items():
        method_rows = results[results["Method"] == method]
        for _, trial_rows in method_rows.groupby("Trial"):
            available = trial_rows.dropna(subset=[value_column]).sort_values("Days")
            ax.plot(
                available["Days"],
                available[value_column],
                color=style["color"],
                marker=style["marker"],
                markersize=5,
                linewidth=1.35,
                alpha=0.62,
            )
    ax.set_xscale("log", base=2)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time-series length (days)")
    ax.set_ylabel("Elapsed time (seconds)")
    ax.grid(True, alpha=0.25)


def create_figure(results: pd.DataFrame, output_path: Path) -> None:
    """Create lazy-load and compute panels containing all individual trials."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    plot_trials(axes[0], results, "Lazy_Load_Time_s", "Lazy-load time")
    plot_trials(axes[1], results, "Compute_Time_s", "Compute time")

    days = np.sort(results["Days"].unique())
    legend_handles = [
        Line2D(
            [0], [0], color=style["color"], marker=style["marker"],
            linewidth=2, label=style["label"],
        )
        for style in METHOD_STYLES.values()
    ]
    for ax in axes:
        ax.set_xticks(days, [f"{day:g}" for day in days])
        ax.legend(handles=legend_handles)

    skipped_days = np.sort(
        results.loc[results["Compute_Time_s"].isna(), "Days"].unique()
    )
    if skipped_days.size:
        axes[1].text(
            0.98,
            0.03,
            "Compute skipped at "
            + ", ".join(f"{day:g}" for day in skipped_days)
            + " days",
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="0.35",
        )

    fig.suptitle(
        "IMERG data-access performance: OPeNDAP versus virtual Zarr",
        fontsize=15,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = load_results(args.input)
    create_figure(results, args.output)
    print(
        results.groupby(["Method", "Days"])
        .agg(
            trials=("Trial", "count"),
            lazy_trials=("Lazy_Load_Time_s", "count"),
            compute_trials=("Compute_Time_s", "count"),
        )
        .to_string()
    )
    print(f"\nFigure saved to {args.output}")


if __name__ == "__main__":
    main()
