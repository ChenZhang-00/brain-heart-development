"""Draw the Figure 3 regional waveforms from saved ROI means and cluster tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

from importlib import import_module
_hep_config = import_module("04_HEP.config.hep_config")
COLORS = _hep_config.COLORS
CONDITIONS = _hep_config.CONDITIONS
GROUPS = _hep_config.GROUPS
PAIR_COLORS = _hep_config.PAIR_COLORS
REGIONS = _hep_config.REGIONS
make_paths = _hep_config.make_paths
relative_path = import_module("common.paths").relative_path
parse_args = _hep_config.parse_args
_inputs = import_module("04_HEP.analysis.inputs")
load_roi = _inputs.load_roi
load_clusters = _inputs.load_clusters


def build_waveform_figure(paths=None):
    """Create the nine waveform axes for standalone or combined Figure 3."""
    paths = paths or make_paths()
    _, times, _, groups, waves = load_roi(paths)
    significant = load_clusters(paths).query("survives").copy()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 11,
            "font.weight": "bold",
            "text.color": "black",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.4,
            "xtick.color": "black",
            "ytick.color": "black",
            "xtick.major.width": 1.3,
            "ytick.major.width": 1.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 11.4), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.07, top=0.85, hspace=0.24, wspace=0.24)
    for row, group in enumerate(GROUPS):
        for col, region in enumerate(REGIONS):
            ax = axes[row, col]
            for condition in CONDITIONS:
                values = waves[region, condition][groups == group]
                mean, sem = values.mean(axis=0), stats.sem(values, axis=0)
                ax.fill_between(
                    times, mean - sem, mean + sem, color=COLORS[condition], alpha=0.20, lw=0
                )
                ax.plot(times, mean, color=COLORS[condition], lw=1.8)
            ax.axhline(0, color="0.5", lw=1)
            ax.axvline(0, color="0.5", lw=1)
            selected = significant.loc[
                (significant.group == group) & (significant.region == region)
            ]
            selected = selected.sort_values(["pair", "t_start", "window"])
            low, high = ax.get_ylim()
            reserve = max(0.28, 0.10 * len(selected) + 0.06)
            ax.set_ylim(low - (high - low) * reserve / (1 - reserve), high)
            for lane, cluster in enumerate(selected.itertuples()):
                y = 0.045 + 0.10 * (len(selected) - lane - 1)
                color = PAIR_COLORS[cluster.pair]
                ax.plot(
                    [cluster.t_start, cluster.t_end],
                    [y, y],
                    transform=ax.get_xaxis_transform(),
                    color=color,
                    lw=5,
                    solid_capstyle="butt",
                )
                p_label = "$p$ < 0.001" if cluster.p < 0.001 else f"$p$ = {cluster.p:.3f}"
                label = f"{cluster.t_start:.0f}–{cluster.t_end:.0f} ms, {p_label}"
                ax.text(
                    (cluster.t_start + cluster.t_end) / 2,
                    y + 0.026,
                    label,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
                )
            ax.set_xlim(-200, 600)
            ax.set_xticks([-200, 0, 200, 400, 600])
            if row == 0:
                ax.set_title(region, fontsize=16, pad=10)
            if col == 0:
                ax.text(
                    -0.25,
                    0.5,
                    group.replace("-", "–") + " years",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=15,
                )
            if row == 2:
                ax.set_xlabel("Time from R peak (ms)", fontsize=12)

    condition_labels = {"fear": "Fearful", "neutral": "Neutral", "happy": "Happy"}
    condition_handles = [
        Line2D([], [], color=COLORS[c], lw=2.3, label=condition_labels[c]) for c in CONDITIONS
    ]
    cluster_handles = [
        Patch(color=color, label=" vs ".join(condition_labels[part] for part in pair.split(" - ")))
        for pair, color in PAIR_COLORS.items()
    ]
    fig.legend(
        handles=condition_handles,
        title="Conditions",
        loc="upper right",
        bbox_to_anchor=(0.50, 0.995),
        frameon=False,
        fontsize=11,
        title_fontsize=12,
    )
    fig.legend(
        handles=cluster_handles,
        title="Cluster contrasts",
        loc="upper left",
        bbox_to_anchor=(0.54, 0.995),
        frameon=False,
        fontsize=11,
        title_fontsize=12,
    )
    return fig, axes


def plot_waveforms(paths=None, *, check_only=False):
    paths = paths or make_paths()
    if check_only:
        load_roi(paths)
        load_clusters(paths)
        print("Waveform plot inputs validated.")
        return
    paths.figures.mkdir(parents=True, exist_ok=True)
    FIGURE = paths.figures / "Figure_3_waveforms.png"
    fig, _ = build_waveform_figure(paths)
    fig.savefig(FIGURE, dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved:", relative_path(FIGURE, paths.root))
    plt.close(fig)
    return FIGURE


def main(argv=None):
    paths, check_only = parse_args(__doc__, argv)
    plot_waveforms(paths, check_only=check_only)


if __name__ == "__main__":
    main()
