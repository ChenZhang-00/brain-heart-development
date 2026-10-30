"""Figures S6–S7: all-band intensity contrasts with significant cluster members."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.dbhi_config import parse_paths, shown


def run(paths):
    import json
    from pathlib import Path

    import matplotlib
    import mne
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mne.viz.topomap import _get_pos_outlines

    OUT = paths.spatial_results
    FIG = paths.supplementary
    FIG.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(OUT / "topo_channel_tests.csv")
    P = json.load(open(OUT / "analysis_parameters.json"))
    labels = P["channel_order"]
    from config.dbhi_config import BANDS, GROUPS as GROUP_AGES

    GROUPS = list(GROUP_AGES)
    EMOTIONS = ["fear", "happy"]
    minfo = mne.create_info(labels, sfreq=1.0, ch_types="eeg")
    minfo.set_montage("standard_1020")
    positions, _ = _get_pos_outlines(minfo, picks=None, sphere=None)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "figure.titleweight": "bold",
        }
    )
    cols = [("LF", b) for b in BANDS] + [None] + [("HF", b) for b in BANDS]
    titles = {"happy": "Happy (top 30% − bottom 30%)", "fear": "Fearful (top 30% − bottom 30%)"}
    tcol = {"happy": "#C60700", "fear": "#1172BF"}
    for emotion in EMOTIONS:
        fig = plt.figure(figsize=(7.09, 5.1))
        grid = fig.add_gridspec(
            9,
            len(cols),
            height_ratios=[1, 1, 1, 0.08, 0.45, 1, 1, 1, 0.08],
            width_ratios=[1 if c else 0.3 for c in cols],
            left=0.105,
            right=0.985,
            bottom=0.03,
            top=0.87,
            hspace=0.25,
            wspace=0.15,
        )
        for block, direction in enumerate(["bth", "htb"]):
            start = block * 5
            subset = res[(res.emotion == emotion) & (res.direction == direction)]
            row_centers = []
            for col, c in enumerate(cols):
                if c is None:
                    continue
                hrv, band = c
                panel = subset[(subset.band == band) & (subset.hrv == hrv)]
                limit = float(panel["mean"].abs().max())
                for row, group in enumerate(GROUPS):
                    values = panel[panel.group == group].set_index("channel").loc[labels]
                    ax = fig.add_subplot(grid[start + row, col])
                    im, _ = mne.viz.plot_topomap(
                        values["mean"].to_numpy(),
                        minfo,
                        axes=ax,
                        show=False,
                        cmap="RdBu_r",
                        vlim=(-limit, limit),
                        contours=4,
                        sensors=False,
                    )
                    corrected = values.cluster_significant.to_numpy().astype(bool)
                    ax.scatter(
                        *positions[corrected].T,
                        s=10,
                        c="k",
                        edgecolors="white",
                        linewidths=0.5,
                        zorder=6,
                    )
                    if row == 0:
                        ax.set_title(f"EEG {band}", fontsize=8.5, pad=4)
                    if col == 0:
                        center = ax.get_position().y0 + ax.get_position().height / 2
                        row_centers.append(center)
                        fig.text(0.09, center, group, ha="right", va="center", fontsize=8.5)
                box = ax.get_position()
                cax = fig.add_axes(
                    [box.x0 + 0.15 * box.width, box.y0 - 0.04, 0.7 * box.width, 0.012]
                )
                bar = fig.colorbar(
                    im, cax=cax, orientation="horizontal", ticks=[-limit, limit], format="%.2g"
                )
                bar.ax.tick_params(labelsize=6.5, length=1.5, pad=1.5)
            fig.text(
                0.022,
                np.mean(row_centers),
                (
                    "Brain-to-heart (EEG → HRV)"
                    if direction == "bth"
                    else "Heart-to-brain (HRV → EEG)"
                ),
                rotation=90,
                ha="center",
                va="center",
                fontsize=9.5,
            )
        # block headers over the LF and HF column groups
        for (j0, j1), txt in [((0, 3), "LF-HRV"), ((5, 8), "HF-HRV")]:
            a0 = fig.add_subplot(grid[0, j0])
            a1 = fig.add_subplot(grid[0, j1])
            x = (a0.get_position().x0 + a1.get_position().x1) / 2
            a0.remove()
            a1.remove()
            fig.text(x, 0.915, txt, ha="center", va="bottom", fontsize=10)
        fig.suptitle(titles[emotion], fontsize=12, y=0.995, color=tcol[emotion])
        number = 6 if emotion == "fear" else 7
        fig.savefig(
            FIG / f"Figure_S{number}_intensity_contrasts_{emotion}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.06,
        )
        plt.close(fig)
        print("Saved:", shown(FIG / f"Figure_S{number}_intensity_contrasts_{emotion}.png"))


if __name__ == "__main__":
    run(parse_paths(__doc__, ()))
