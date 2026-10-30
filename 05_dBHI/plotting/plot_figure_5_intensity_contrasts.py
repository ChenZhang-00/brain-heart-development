"""Figure 5: delta-band intensity contrasts with significant spatial clusters."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.dbhi_config import parse_paths, shown


def run(paths):
    import json
    from pathlib import Path

    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    from mne.viz.topomap import _get_pos_outlines

    BASE = paths.spatial_results
    FIG = paths.figures
    FIG.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(BASE / "topo_channel_tests.csv")
    res = res[res.band == "delta"]
    labels = json.load(open(BASE / "analysis_parameters.json"))["channel_order"]
    minfo = mne.create_info(labels, sfreq=1.0, ch_types="eeg")
    minfo.set_montage("standard_1020")
    positions, _ = _get_pos_outlines(minfo, picks=None, sphere=None)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "figure.titleweight": "bold",
        }
    )
    from config.dbhi_config import GROUPS as GROUP_AGES, HRV

    GROUPS = list(GROUP_AGES)
    EMOS = [("fear", "Fearful", "#1172BF"), ("happy", "Happy", "#C60700")]

    fig = plt.figure(figsize=(7.09, 4.9))
    # rows: fear bth, fear htb, gap, happy bth, happy htb ; cols: 3 maps + cbar + gap + 3 maps + cbar
    grid = fig.add_gridspec(
        5,
        9,
        height_ratios=[1, 1, 0.2, 1, 1],
        width_ratios=[1, 1, 1, 0.07, 0.35, 1, 1, 1, 0.07],
        left=0.11,
        right=0.93,
        bottom=0.05,
        top=0.88,
        hspace=0.45,
        wspace=0.25,
    )
    for ei, (emo, etitle, ecol) in enumerate(EMOS):
        r0 = ei * 3
        for di, direction in enumerate(["bth", "htb"]):
            row = r0 + di
            for hi, hrv in enumerate(HRV):
                c0 = hi * 5
                panel = res[(res.emotion == emo) & (res.direction == direction) & (res.hrv == hrv)]
                limit = float(panel["mean"].abs().max())
                for gi, group in enumerate(GROUPS):
                    v = panel[panel.group == group].set_index("channel").loc[labels]
                    ax = fig.add_subplot(grid[row, c0 + gi])
                    im, _ = mne.viz.plot_topomap(
                        v["mean"].to_numpy(),
                        minfo,
                        axes=ax,
                        show=False,
                        cmap="RdBu_r",
                        vlim=(-limit, limit),
                        contours=4,
                        sensors=False,
                    )
                    sig = v.cluster_significant.to_numpy()
                    ax.scatter(
                        *positions[sig].T, s=16, c="k", edgecolors="white", linewidths=0.5, zorder=6
                    )
                    if di == 0:
                        ax.set_title(group, fontsize=9, pad=3)
                    if gi == 1:
                        lab = (
                            f"EEG delta → {hrv}-HRV"
                            if direction == "bth"
                            else f"{hrv}-HRV → EEG delta"
                        )
                        ax.text(
                            0.5,
                            -0.04,
                            lab,
                            transform=ax.transAxes,
                            ha="center",
                            va="top",
                            fontsize=8.5,
                        )
                    if hi == 0 and gi == 0:
                        p = ax.get_position()
                        fig.text(
                            0.085,
                            p.y0 + p.height / 2,
                            "Brain-to-heart" if direction == "bth" else "Heart-to-brain",
                            rotation=90,
                            ha="center",
                            va="center",
                            fontsize=9.5,
                        )
                cax = fig.add_subplot(grid[row, c0 + 3])
                box = cax.get_position()
                cax.set_position([box.x0, box.y0 + 0.18 * box.height, box.width, 0.64 * box.height])
                bar = fig.colorbar(
                    im, cax=cax, orientation="vertical", ticks=[-limit, 0, limit], format="%.2g"
                )
                bar.ax.tick_params(labelsize=7, length=1.5, pad=1)
        # emotion title centered over the block
        axs = [
            a
            for a in fig.axes
            if a.get_subplotspec() is not None
            and a.get_subplotspec().rowspan.start == r0
            and a.get_subplotspec().colspan.start in (0, 1, 2, 5, 6, 7)
        ]
        top = max(a.get_position().y1 for a in axs)
        xc = np.mean([a.get_position().x0 + a.get_position().width / 2 for a in axs])
        fig.text(
            xc,
            top + 0.06,
            f"{etitle} (top 30% − bottom 30%)",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=ecol,
        )
        fig.add_artist(
            plt.Line2D([0.11, 0.93], [top + 0.035] * 2, color=ecol, lw=1, transform=fig.transFigure)
        )
    out = FIG / "Figure_5_intensity_contrasts"
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.06)
    print("Saved:", shown(out.with_suffix(".png")))


if __name__ == "__main__":
    run(parse_paths(__doc__, ()))
