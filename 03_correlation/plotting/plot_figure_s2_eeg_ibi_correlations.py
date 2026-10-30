"""Figure S2: group-mean EEG–IBI covariation across all three regions.

Uses the saved frontal 36-test and central/posterior 72-test FDR families.
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from importlib import import_module
_plot_figure_2_eeg_ibi_correlations = import_module("03_correlation.plotting.plot_figure_2_eeg_ibi_correlations")
load_correlations = _plot_figure_2_eeg_ibi_correlations.load_correlations
plot_paths = _plot_figure_2_eeg_ibi_correlations.plot_paths
project_root = import_module("03_correlation.config.correlation_config").project_root
relative_path = import_module("common.paths").relative_path


def plot_figure_s2(result_dir, figure_dir):
    FIG = Path(figure_dir)
    FIG.mkdir(parents=True, exist_ok=True)
    gc = load_correlations(result_dir)
    assert len(gc) == 108
    look = gc.set_index(["condition", "region", "band", "age_group"])
    CONDS = ["neutral", "fear", "happy"]
    REGIONS = ["Frontal", "Central", "Posterior"]
    BANDS = ["delta", "theta", "alpha", "beta"]
    GROUPS = ["5-6", "7-8", "9-10"]
    GCOL = {"5-6": "red", "7-8": "blue", "9-10": "black"}
    CCOL = {"neutral": "0.40", "fear": "tab:red", "happy": "tab:blue"}
    CLAB = {"neutral": "Neutral", "fear": "Fearful", "happy": "Happy"}
    OFF = {"5-6": 0.22, "7-8": 0.0, "9-10": -0.22}

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "dejavusans",
        }
    )
    rows = [(r, b) for r in REGIONS for b in BANDS]
    ypos = np.arange(len(rows))[::-1]
    lim = float(np.ceil(gc.r.abs().max() / 0.1) * 0.1)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 7.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.09, top=0.86, wspace=0.12)
    for ax, c in zip(axes, CONDS):
        for k in range(len(REGIONS)):
            if k % 2 == 0:
                ax.axhspan(ypos[k * 4 + 3] - 0.5, ypos[k * 4] + 0.5, color="0.955", lw=0, zorder=0)
        ax.axvline(0, color="0.25", lw=1.2, zorder=1)
        for g in GROUPS:
            v = np.array([look.loc[(c, r, b, g), "r"] for r, b in rows])
            q = np.array([look.loc[(c, r, b, g), "q_fdr"] for r, b in rows])
            y = ypos + OFF[g]
            sig = q < 0.05
            ax.hlines(y, 0, v, color=GCOL[g], lw=1.3, alpha=0.45, zorder=2)
            ax.plot(v[sig], y[sig], "o", ms=6.5, color=GCOL[g], mec="white", mew=0.8, zorder=4)
            ax.plot(v[~sig], y[~sig], "o", ms=6.5, mfc="white", mec=GCOL[g], mew=1.5, zorder=4)
        ax.set_xlim(-lim, lim)
        ax.set_xticks(np.linspace(-lim, lim, 5))
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlabel("Pearson r", fontsize=12)
        ax.set_title(CLAB[c], color=CCOL[c], fontsize=16, pad=10)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([f"{r} {b}" for r, b in rows])
    fig.legend(
        handles=[Line2D([], [], color=GCOL[g], lw=2.5, label=f"{g} yrs") for g in GROUPS],
        frameon=False,
        ncol=3,
        fontsize=11,
        loc="upper left",
        bbox_to_anchor=(0.13, 0.975),
        handlelength=1.3,
        columnspacing=1.8,
    )
    fig.text(
        0.99,
        0.955,
        r"Filled circles: $p_{\mathrm{FDR}}$ < 0.05",
        ha="right",
        va="center",
        fontsize=10,
    )
    fig.savefig(FIG / "Figure_S2_eeg_ibi_correlations.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("Saved:", relative_path(FIG / "Figure_S2_eeg_ibi_correlations.png", project_root()))
    plt.close(fig)
    return FIG / "Figure_S2_eeg_ibi_correlations.png"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", help="Directory containing group_mean_correlations.csv.")
    parser.add_argument("--figures", help="Figure directory (default: results/SI/ at the repository root).")
    args = parser.parse_args(argv)
    plot_figure_s2(*plot_paths(args.results, args.figures, section="SI"))


if __name__ == "__main__":
    main()
