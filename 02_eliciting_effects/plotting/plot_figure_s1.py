"""Render Figure S1 from saved analysis outputs; no raw EEG/ECG inputs are needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats

from importlib import import_module
_eliciting_config = import_module("02_eliciting_effects.config.eliciting_config")
BAND_COLOR = _eliciting_config.BAND_COLOR
BANDS = _eliciting_config.BANDS
CONDITIONS = _eliciting_config.CONDITIONS
EMO_COLOR = _eliciting_config.EMO_COLOR
EMO_TITLE = _eliciting_config.EMO_TITLE
EMOTIONS = _eliciting_config.EMOTIONS
EXPECTED_SUBJECTS = _eliciting_config.EXPECTED_SUBJECTS
LABEL = _eliciting_config.LABEL
T_EEG = _eliciting_config.T_EEG
make_paths = _eliciting_config.make_paths
project_root = _eliciting_config.project_root
relative_path = import_module("common.paths").relative_path
resolve_path = _eliciting_config.resolve_path


def plot_figure_s1(result_dir, figure_dir):
    result_dir, FIG = Path(result_dir), Path(figure_dir)
    required = [
        "figure_s1_inputs.npz",
        "eliciting_effects_posthoc.csv",
        "neuro_dynamics_clusters.csv",
    ]
    missing = [name for name in required if not (result_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing saved outputs: {', '.join(missing)}. Run analysis steps 3–5 first."
        )
    with np.load(result_dir / "figure_s1_inputs.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    if int(data["schema_version"]) != 1 or int(data["n_subjects"]) != EXPECTED_SUBJECTS:
        raise ValueError(
            "Figure inputs need schema version 1 and the complete 148-participant cohort."
        )
    for name, expected in [("conditions", CONDITIONS), ("emotions", EMOTIONS), ("bands", BANDS)]:
        if data[name].tolist() != expected:
            raise ValueError(f"Unexpected {name} in figure inputs.")
    value = {m: data[f"value_{m}"] for m in ["IBI", "HF", "LF"]}
    if any(x.shape != (EXPECTED_SUBJECTS, len(CONDITIONS)) for x in value.values()):
        raise ValueError("Figure cardiac arrays must have shape (148, 3).")
    intensity = {e: data[f"intensity_{e}"] for e in EMOTIONS}
    ibi_mean = {e: data[f"ibi_mean_{e}"] for e in EMOTIONS}
    eeg_mean = {"Frontal": {b: {e: data[f"eeg_Frontal_{b}_{e}"] for e in EMOTIONS} for b in BANDS}}
    t_ibi = data["t_ibi"]
    post = pd.read_csv(result_dir / "eliciting_effects_posthoc.csv")
    clust = pd.read_csv(result_dir / "neuro_dynamics_clusters.csv")
    if not {"measure", "i", "j", "stars"} <= set(post.columns) or len(post) != 15:
        raise ValueError("Expected the 15 paired-test rows from the response analysis.")
    if not {"signal", "band", "emotion", "t_start_s", "t_end_s", "tsum", "p"} <= set(clust.columns):
        raise ValueError("Cluster table is missing required columns.")
    FIG.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
        }
    )
    fig = plt.figure(figsize=(7.4, 8.6))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 1.55], hspace=0.32, left=0.08, right=0.98, top=0.96, bottom=0.05
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.35)
    # a: self-report
    ax = fig.add_subplot(top[0])
    rng = np.random.default_rng(42)
    jit = lambda a: a + rng.uniform(-0.4, 0.4, size=len(a))
    for e, lab in [("fear", "Fearful − Neutral"), ("happy", "Happy − Neutral")]:
        ax.scatter(
            jit(data[f"arousal_{e}"]),
            jit(data[f"valence_{e}"]),
            s=11,
            color=EMO_COLOR[e],
            alpha=0.8,
            lw=0,
            label=lab,
            zorder=3,
        )
    ax.axhline(0, color="0.5", lw=0.7, ls=":")
    ax.axvline(0, color="0.5", lw=0.7, ls=":")
    ax.set_xlim(-9, 9)
    ax.set_ylim(-9, 9)
    ax.set_xticks([-8, -4, 0, 4, 8])
    ax.set_yticks([-8, -4, 0, 4, 8])
    ax.set_xlabel("Δ Arousal")
    ax.set_ylabel("Δ Valence")
    ax.set_title("Self-reported feelings", fontsize=9, fontweight="bold", loc="left")
    ax.legend(frameon=False, loc="lower left", handletextpad=0.2, borderpad=0.1)
    ax.text(-0.26, 1.06, "a", transform=ax.transAxes, fontsize=12, fontweight="bold")

    # b: cardiac
    def wsci(X):
        n, k = X.shape
        Xn = X - X.mean(1, keepdims=True) + X.mean()
        sem = Xn.std(0, ddof=1) / np.sqrt(n) * np.sqrt(k / (k - 1))
        return sem * stats.t.ppf(0.975, n - 1)

    bgrid = top[1].subgridspec(1, 3, wspace=0.75)
    for k, (m, title, ylab) in enumerate(
        [
            ("IBI", "IBI", "IBI (ms)"),
            ("HF", "HF-HRV", "ln power (a.u.)"),
            ("LF", "LF-HRV", "ln power (a.u.)"),
        ]
    ):
        ax = fig.add_subplot(bgrid[k])
        X = value[m]
        mean = X.mean(0)
        ci = wsci(X)
        lo, hi = float(np.min(mean - ci)), float(np.max(mean + ci))
        span = hi - lo
        bottom = lo - 1.2 * span
        for x, c in enumerate(CONDITIONS):
            ax.bar(x, mean[x] - bottom, 0.62, bottom=bottom, color=EMO_COLOR[c], zorder=2)
            ax.errorbar(
                x,
                mean[x],
                yerr=ci[x],
                fmt="none",
                ecolor="k",
                elinewidth=0.8,
                capsize=2,
                capthick=0.8,
                zorder=3,
            )
        sub = post[(post.measure == m) & (post.stars != "n.s.")].copy()
        sub["w"] = sub.j - sub.i
        sub = sub.sort_values(["w", "i"])
        y = hi + 0.2 * span
        for _, r in sub.iterrows():
            ax.plot([r.i, r.i, r.j, r.j], [y, y + 0.07 * span, y + 0.07 * span, y], lw=0.8, c="k")
            ax.text(
                (r.i + r.j) / 2,
                y + 0.08 * span,
                r.stars,
                ha="center",
                va="bottom",
                fontsize=8,
                color="#C00000",
                fontweight="bold",
            )
            y += 0.36 * span
        ax.set_ylim(bottom, max(y, hi + 0.3 * span))
        ax.set_xlim(-0.7, 2.7)
        ax.set_xticks(range(3))
        ax.set_xticklabels([LABEL[c] for c in CONDITIONS], fontsize=7, rotation=30, ha="right")
        ax.tick_params(axis="x", length=0)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4, prune="lower"))
        ax.set_ylabel(ylab, labelpad=2)
        ax.set_title(title, fontsize=9, fontweight="bold", loc="center")
        for dy in (0, 0.022):
            ax.plot(
                (-0.016, 0.016),
                (0.015 + dy - 0.014, 0.015 + dy + 0.014),
                transform=ax.transAxes,
                color="k",
                clip_on=False,
                lw=0.8,
            )
        if k == 0:
            ax.text(-0.75, 1.06, "b", transform=ax.transAxes, fontsize=12, fontweight="bold")
    # c: dynamics
    cgrid = outer[1].subgridspec(1, 2, wspace=0.22)
    for col, e in enumerate(EMOTIONS):
        inner = cgrid[col].subgridspec(3, 1, height_ratios=[0.07, 1, 1.1], hspace=0.35)
        ax_s = fig.add_subplot(inner[0])
        ax_i = fig.add_subplot(inner[1])
        ax_e = fig.add_subplot(inner[2])
        ax_s.imshow(
            intensity[e][None, :],
            aspect="auto",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            extent=[0, T_EEG, 0, 1],
        )
        ax_s.set_xticks([])
        ax_s.set_yticks([])
        ax_s.set_title(EMO_TITLE[e], fontsize=9, fontweight="bold", color=EMO_COLOR[e], pad=3)
        if col == 0:
            ax_s.text(-0.17, 1.9, "c", transform=ax_s.transAxes, fontsize=12, fontweight="bold")
        ax_i.axhline(0, color="0.6", lw=0.7, ls="--")
        for _, r in clust[
            (clust.signal == "IBI") & (clust.emotion == e) & (clust.p < 0.05)
        ].iterrows():
            ax_i.axvspan(r.t_start_s, r.t_end_s, color=EMO_COLOR[e], alpha=0.18, lw=0)
        ax_i.plot(t_ibi, ibi_mean[e], color=EMO_COLOR[e], lw=1.1)
        ax_i.set_xlim(0, T_EEG)
        ax_i.set_ylabel("ΔIBI (ms)")
        ax_i.set_xticklabels([])
        ax_e.axhline(0, color="0.6", lw=0.7, ls="--")
        t = np.arange(T_EEG)
        ytop = 0.56
        for bi, b in enumerate(BANDS):
            ax_e.plot(
                t, eeg_mean["Frontal"][b][e], color=BAND_COLOR[b], lw=0.9, label=b.capitalize()
            )
            yb = ytop - bi * 0.05
            for _, r in clust[
                (clust.signal == "EEG")
                & (clust.band == b)
                & (clust.emotion == e)
                & (clust.p < 0.05)
            ].iterrows():
                ax_e.plot(
                    [r.t_start_s, r.t_end_s],
                    [yb, yb],
                    color=BAND_COLOR[b],
                    lw=3,
                    solid_capstyle="butt",
                )
        ax_e.set_xlim(0, T_EEG)
        ax_e.set_ylim(-0.8, 0.6)
        ax_e.set_ylabel("Δ frontal log power")
        ax_e.set_xlabel("Time (s)")
        if col == 0:
            ax_e.legend(
                frameon=False,
                loc="lower left",
                ncol=4,
                columnspacing=0.8,
                handlelength=1.2,
                handletextpad=0.3,
                fontsize=7,
            )
    fig.savefig(FIG / "Figure_S1_movie_responses.png", dpi=300, facecolor="white")
    plt.close(fig)
    print("Saved:", relative_path(FIG / "Figure_S1_movie_responses.png", project_root()))
    return FIG / "Figure_S1_movie_responses.png"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", help="Directory containing the saved analysis tables and figure cache."
    )
    parser.add_argument("--figures", help="Figure directory (default: results/SI/ at the repository root).")
    args = parser.parse_args(argv)
    paths = make_paths()
    result_dir = resolve_path(args.results or paths.results, paths.root)
    figure_dir = resolve_path(args.figures or paths.figures, paths.root)
    plot_figure_s1(result_dir, figure_dir)


if __name__ == "__main__":
    main()
