"""Render Fig. 2a and Fig. 2b from saved correlation tables, without rerunning statistics."""

import argparse
import os
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from importlib import import_module
_correlation_config = import_module("03_correlation.config.correlation_config")
BANDS = _correlation_config.BANDS
CONDITIONS = _correlation_config.CONDITIONS
GROUPS = _correlation_config.GROUPS
REGIONS = _correlation_config.REGIONS
project_root = _correlation_config.project_root
relative_path = import_module("common.paths").relative_path
resolve_path = _correlation_config.resolve_path


def load_correlations(result_dir):
    """Require one valid row for each of the 108 planned tests."""
    path = Path(result_dir) / "group_mean_correlations.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run the analysis step first.")
    table = pd.read_csv(path)
    keys = ["condition", "region", "band", "age_group"]
    required = set(keys + ["r", "p_pearson", "q_fdr", "fdr_family"])
    if not required <= set(table.columns):
        raise ValueError(f"Missing correlation columns: {sorted(required - set(table.columns))}")
    expected = pd.MultiIndex.from_product([CONDITIONS, REGIONS, BANDS, GROUPS], names=keys)
    actual = pd.MultiIndex.from_frame(table[keys])
    if not actual.is_unique or len(table) != len(expected) or len(expected.difference(actual)):
        raise ValueError(
            "Expected exactly one row for each of the 108 region/movie/band/age-group tests."
        )
    if not np.isfinite(table[["r", "p_pearson", "q_fdr"]].to_numpy()).all():
        raise ValueError("Saved correlations contain non-finite statistics.")
    if (
        not table.r.between(-1, 1).all()
        or not table.q_fdr.between(0, 1).all()
        or not table.p_pearson.between(0, 1).all()
    ):
        raise ValueError("Saved correlations contain out-of-range statistics.")
    expected_family = np.where(
        table.region == "Frontal", "primary_frontal_36", "supplementary_regions_72"
    )
    if not np.array_equal(table.fdr_family.to_numpy(), expected_family):
        raise ValueError(
            "Saved FDR families do not match the 36 frontal / 72 central-posterior design."
        )
    return table


def plot_paths(results=None, figures=None, *, section="Main"):
    root = project_root()
    output = resolve_path(os.environ.get("EEG_IBI_OUTPUT_ROOT") or "03_correlation", root)
    result_dir = resolve_path(
        results or os.environ.get("EEG_IBI_PLOT_RESULTS") or output / "cache", root
    )
    publication = (root if output == root / "03_correlation" else output) / "results"
    figure_dir = resolve_path(
        figures or os.environ.get("EEG_IBI_PLOT_OUTPUT") or publication / section, root
    )
    return result_dir, figure_dir


def finish_display(figure, show):
    if show:
        from IPython.display import display

        display(figure)
    plt.close(figure)


def plot_figure_2a(result_dir, figure_dir, *, show=False):
    """Draw signed frontal correlations; filled markers use saved q_fdr < 0.05."""
    DOT_RESULT_DIR = Path(result_dir)
    DOT_FIGURE_DIR = Path(figure_dir)
    DOT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    dot_table = load_correlations(DOT_RESULT_DIR)
    dot_table = dot_table[dot_table.region == "Frontal"].copy()
    assert len(dot_table) == 36
    assert dot_table.fdr_family.eq("primary_frontal_36").all()
    dot_lookup = dot_table.set_index(["condition", "age_group", "band"])
    DOT_CONDITIONS = ["neutral", "fear", "happy"]
    DOT_GROUPS = ["5-6", "7-8", "9-10"]
    DOT_BANDS = ["delta", "theta", "alpha", "beta"]
    DOT_GROUP_COLORS = {"5-6": "red", "7-8": "blue", "9-10": "black"}
    DOT_CONDITION_COLORS = {"neutral": "0.40", "fear": "tab:red", "happy": "tab:blue"}
    DOT_CONDITION_LABELS = {"neutral": "Neutral", "fear": "Fearful", "happy": "Happy"}
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "dejavusans",
        }
    )

    dot_rows = np.arange(len(DOT_BANDS))[::-1]
    dot_offsets = {"5-6": 0.20, "7-8": 0.0, "9-10": -0.20}
    dot_limit = float(np.ceil(dot_table.r.abs().max() / 0.1) * 0.1)
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 6.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.13, top=0.73, wspace=0.14)
    dot_drawn_values = []
    dot_filled_count = 0
    for ax, condition in zip(axes, DOT_CONDITIONS):
        for row in range(len(DOT_BANDS)):
            if row % 2 == 0:
                ax.axhspan(
                    dot_rows[row] - 0.48, dot_rows[row] + 0.48, color="0.965", lw=0, zorder=0
                )
        ax.axvline(0, color="0.25", lw=1.4, zorder=1)
        for age_group in DOT_GROUPS:
            values = np.array(
                [dot_lookup.loc[(condition, age_group, band), "r"] for band in DOT_BANDS]
            )
            corrected = np.array(
                [dot_lookup.loc[(condition, age_group, band), "q_fdr"] for band in DOT_BANDS]
            )
            positions = dot_rows + dot_offsets[age_group]
            color = DOT_GROUP_COLORS[age_group]
            significant = corrected < 0.05
            # Stems connect estimates to zero; they are not confidence intervals.
            ax.hlines(positions, 0, values, color=color, lw=1.6, alpha=0.45, zorder=2)
            ax.plot(
                values[significant],
                positions[significant],
                "o",
                ms=8.5,
                color=color,
                mec="white",
                mew=1.0,
                zorder=4,
            )
            ax.plot(
                values[~significant],
                positions[~significant],
                "o",
                ms=8.5,
                mfc="white",
                mec=color,
                mew=1.9,
                zorder=4,
            )
            dot_drawn_values.extend(values.tolist())
            dot_filled_count += int(significant.sum())
        ax.set_xlim(-dot_limit, dot_limit)
        ax.set_xticks(np.linspace(-dot_limit, dot_limit, 5))
        ax.set_ylim(-0.55, 3.55)
        ax.set_xlabel("Pearson r", fontsize=17, labelpad=8)
        ax.set_title(
            DOT_CONDITION_LABELS[condition],
            color=DOT_CONDITION_COLORS[condition],
            fontsize=23,
            pad=13,
        )
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", length=0, labelsize=15)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(dot_rows, [f"{band.capitalize()}-IBI" for band in DOT_BANDS])
    # The shared legend and significance note sit above all three panels.
    age_handles = [
        Line2D([], [], color=DOT_GROUP_COLORS[group], lw=2.8, label=f"{group} yrs")
        for group in DOT_GROUPS
    ]
    fig.legend(
        handles=age_handles,
        frameon=False,
        ncol=1,
        fontsize=13,
        loc="upper left",
        bbox_to_anchor=(0.40, 0.99),
        borderaxespad=0,
        handlelength=1.3,
        handletextpad=0.6,
        labelspacing=0.6,
    )
    fig.text(
        0.55,
        0.925,
        "Filled circles:" + "\n" + r"($p_{\mathrm{FDR}} < 0.05$)",
        ha="left",
        va="center",
        fontsize=11.5,
        linespacing=1.4,
        color="black",
    )
    assert np.allclose(np.sort(dot_drawn_values), np.sort(dot_table.r.to_numpy()))
    assert dot_filled_count == int((dot_table.q_fdr < 0.05).sum())
    dot_output = DOT_FIGURE_DIR / "Figure_2a_eeg_ibi_correlations.png"
    fig.savefig(dot_output, dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved:", relative_path(dot_output, project_root()))
    finish_display(fig, show)
    return dot_output


def plot_figure_2b(result_dir, figure_dir, *, show=False):
    """Draw centered alpha–IBI regression lines and ordinary OLS 95% mean-response CIs."""
    PLOT_RESULT_DIR = Path(result_dir)
    PLOT_FIGURE_DIR = Path(figure_dir)
    PLOT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_CONDITIONS = ["neutral", "fear", "happy"]
    PLOT_GROUPS = ["5-6", "7-8", "9-10"]
    PLOT_BANDS = ["delta", "theta", "alpha", "beta"]
    PLOT_CONDITION_LABELS = {"neutral": "Neutral", "fear": "Fearful", "happy": "Happy"}
    PLOT_CONDITION_COLORS = {"neutral": "0.40", "fear": "tab:red", "happy": "tab:blue"}
    PLOT_GROUP_COLORS = {"5-6": "red", "7-8": "blue", "9-10": "black"}
    PLOT_GROUP_LABELS = {group: f"{group} yrs" for group in PLOT_GROUPS}
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.weight": "bold",
            "font.size": 12,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "dejavusans",
        }
    )

    plot_correlations = load_correlations(PLOT_RESULT_DIR)
    plot_primary = plot_correlations[plot_correlations.region == "Frontal"].copy()
    assert len(plot_primary) == 36
    plot_lookup = plot_primary.set_index(["condition", "age_group", "band"])
    plot_curves = pd.read_csv(PLOT_RESULT_DIR / "group_mean_timeseries.csv")

    def format_plot_p_fdr(value):
        label = r"$p_{\mathrm{FDR}}$"
        return label + (" < 0.001" if value < 0.001 else f" = {value:.3f}")

    def finish_figure(figure, name):
        figure.savefig(PLOT_FIGURE_DIR / name, dpi=300, bbox_inches="tight", facecolor="white")
        print("Saved:", relative_path(PLOT_FIGURE_DIR / name, project_root()))
        finish_display(figure, show)

    # Alpha-IBI scatter (Fig. 2b): independent panel scales with p_FDR annotations inside each panel.
    alpha_plot_data = {}
    for condition in PLOT_CONDITIONS:
        for age_group in PLOT_GROUPS:
            part = plot_curves[
                (plot_curves.region == "Frontal")
                & (plot_curves.band == "alpha")
                & (plot_curves.condition == condition)
                & (plot_curves.age_group == age_group)
                & plot_curves.used_in_test
            ].sort_values("ibi_bin_index")
            x = part.ibi_ms.to_numpy()
            y = part.eeg_log_power.to_numpy()
            x, y = x - x.mean(), y - y.mean()
            assert np.isfinite(x).all() and np.isfinite(y).all()
            sxx = float(x @ x)
            assert len(x) > 2 and sxx > 0
            slope = float(x @ y / sxx)
            grid = np.linspace(x.min(), x.max(), 120)
            fitted = slope * grid
            residual_variance = float(np.sum((y - slope * x) ** 2) / (len(x) - 2))
            half_width = student_t.ppf(0.975, len(x) - 2) * np.sqrt(
                residual_variance * (1 / len(x) + grid**2 / sxx)
            )
            alpha_plot_data[(condition, age_group)] = (grid, fitted, half_width)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.16, top=0.86, wspace=0.32)
    for ax, condition in zip(axes, PLOT_CONDITIONS):
        handles = []
        lower, upper = [], []
        for age_group in PLOT_GROUPS:
            grid, fitted, half_width = alpha_plot_data[(condition, age_group)]
            color = PLOT_GROUP_COLORS[age_group]
            ax.fill_between(
                grid,
                fitted - half_width,
                fitted + half_width,
                color=color,
                alpha=0.10,
                lw=0,
                zorder=2,
            )
            ax.plot(grid, fitted, color=color, lw=2.7, zorder=3, solid_capstyle="round")
            lower.append(float((fitted - half_width).min()))
            upper.append(float((fitted + half_width).max()))
            row = plot_lookup.loc[(condition, age_group, "alpha")]
            handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    ls="none",
                    color=color,
                    markersize=6.5,
                    label=f"{PLOT_GROUP_LABELS[age_group]}  (r = {row.r:.2f}, "
                    f"{format_plot_p_fdr(row.q_fdr)})",
                )
            )
        span = max(upper) - min(lower)
        # Reserve a clear strip below the curves for the in-panel annotations.
        ax.set_ylim(min(lower) - 0.39 * span, max(upper) + 0.08 * span)
        ax.margins(x=0.06)
        ax.axhline(0, color="0.86", lw=0.8, zorder=1)
        ax.axvline(0, color="0.86", lw=0.8, zorder=1)
        ax.set_title(
            PLOT_CONDITION_LABELS[condition],
            fontsize=23,
            color=PLOT_CONDITION_COLORS[condition],
            pad=13,
        )
        ax.set_xlabel(r"$\Delta$ IBI (ms)", fontsize=17, labelpad=7)
        ax.set_ylabel(r"$\Delta$ EEG alpha power", fontsize=17, labelpad=8)
        ax.tick_params(labelsize=13, width=1.0, length=4)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.legend(
            handles=handles,
            frameon=False,
            fontsize=12.0,
            loc="lower right",
            bbox_to_anchor=(0.995, 0.01),
            borderaxespad=0,
            handlelength=0.9,
            handletextpad=0.45,
            labelspacing=0.45,
        )
    finish_figure(fig, "Figure_2b_alpha_ibi_regression.png")

    return PLOT_FIGURE_DIR / "Figure_2b_alpha_ibi_regression.png"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", help="Directory containing saved analysis CSVs.")
    parser.add_argument("--figures", help="Figure directory (default: results/Main/ at the repository root).")
    args = parser.parse_args(argv)
    result_dir, figure_dir = plot_paths(args.results, args.figures)
    plot_figure_2a(result_dir, figure_dir)
    plot_figure_2b(result_dir, figure_dir)


if __name__ == "__main__":
    main()
