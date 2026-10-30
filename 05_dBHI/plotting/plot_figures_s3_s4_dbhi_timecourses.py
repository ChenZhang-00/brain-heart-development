"""Figures S3–S4: frontal delta time courses and temporal-cluster rasters."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.dbhi_config import parse_paths, shown


def run(paths):
    import sys
    from pathlib import Path

    import matplotlib as mpl
    import numpy as np
    import pandas as pd

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from config.transform_dbhi import transform
    from config.dbhi_config import BANDS, REGIONS, GROUPS, HRV
    from analysis.inputs import validate_model_cache
    from sdg.pipeline import fingerprint
    import json

    RES, CACHE, FIG = paths.time_results, paths.cache, paths.supplementary
    FIG.mkdir(parents=True, exist_ok=True)
    manifest, _, expected_ids, truncation_info = validate_model_cache(paths, temporal=True)
    saved = json.loads((RES / "analysis_parameters.json").read_text())
    if saved["cache_fingerprint"] != fingerprint(manifest):
        raise ValueError("Temporal results and model caches have different fingerprints.")
    for sid, current in truncation_info.items():
        if saved["neutral_truncation"][sid]["sha256"] != current["sha256"]:
            raise ValueError(f"{sid}: model array changed since the temporal analysis.")
    REG_ORDER = ["Frontal", "Central", "Posterior"]
    COLORS = {"5-6 yrs": "#E4001B", "7-8 yrs": "#0033E4", "9-10 yrs": "#111111"}
    OFF = {"5-6 yrs": 0.27, "7-8 yrs": 0.0, "9-10 yrs": -0.27}
    FS, TMAX = 4, 180
    COLS = [
        ("htb", 0, "LF-HRV → EEG"),
        ("htb", 1, "HF-HRV → EEG"),
        ("bth", 0, "EEG → LF-HRV"),
        ("bth", 1, "EEG → HF-HRV"),
    ]
    EMO = {
        "fear": ("Fearful (The Lion King)", "Fear_Intensity", "Figure_S3_dbhi_fear.png"),
        "happy": ("Happy (The Jungle Book)", "Happy_Intensity", "Figure_S4_dbhi_happy.png"),
    }

    ids = pd.read_csv(RES / "included_subjects.csv", dtype={"ID": str})
    assert len(ids) == 148 and sorted(ids.ID) == expected_ids
    grp = dict(zip(ids.ID, ids.analysis_group))
    cl = pd.concat(
        [pd.read_csv(RES / f"dBHI_{b}_clusters.csv").assign(band=b) for b in BANDS],
        ignore_index=True,
    )
    sig = cl.loc[cl.significant.eq(True)]
    assert (sig[sig.direction == "htb"].t_mass > 0).all() and (
        sig[sig.direction == "bth"].t_mass < 0
    ).all()
    idx = np.array(REGIONS["Frontal"]) - 1
    series = {}
    for sid in ids.ID:
        with np.load(CACHE / f"{sid}_dbhi.npz", allow_pickle=False) as z:
            for d in ("htb", "bth"):
                neu = transform(z[f"neutral__{d}"][0][:, idx], d).mean(1)
                base = np.nanmean(neu, axis=-1)
                assert np.isfinite(base).all()
                for e in EMO:
                    y = transform(z[f"{e}__{d}"][0][:, idx], d).mean(1) - base[:, None]
                    ticks = np.rint(z[f"{e}__t_{d}"] * FS).astype(int)
                    series.setdefault((e, d), []).append((sid, ticks, y))
    data = {}
    for key, recs in series.items():
        lo = max(t[0] for _, t, _ in recs)
        hi = min(t[-1] for _, t, _ in recs)
        data[key] = (
            np.arange(lo, hi + 1) / FS,
            [sid for sid, _, _ in recs],
            np.stack([y[:, lo - t[0] : hi - t[0] + 1] for _, t, y in recs]),
        )
    itab = pd.read_csv(paths.ratings)

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    rows = [(b, r) for b in BANDS for r in REG_ORDER]
    ypos = np.arange(len(rows))[::-1]

    def sigrows(e, d, hi, b, r, g):
        return sig[
            (sig.emotion == e)
            & (sig.direction == d)
            & (sig.hrv == HRV[hi])
            & (sig.band == b)
            & (sig.region == r)
            & (sig.group == g)
        ]

    for e, (title, icol, fname) in EMO.items():
        v = itab[icol].to_numpy(float)
        inten = (v - v.min()) / np.ptp(v)
        fig = plt.figure(figsize=(10.2, 5.5))
        gs = fig.add_gridspec(
            5,
            4,
            height_ratios=[0.05, 0.16, 1.0, 0.16, 2.9],
            hspace=0.10,
            wspace=0.22,
            left=0.085,
            right=0.995,
            top=0.90,
            bottom=0.075,
        )
        for j, (d, hi, lab) in enumerate(COLS):
            axs = fig.add_subplot(gs[0, j])
            axs.imshow(
                inten[None, :],
                aspect="auto",
                cmap="YlOrRd",
                vmin=0,
                vmax=1,
                extent=[0, len(inten), 0, 1],
            )
            axs.set_xlim(0, TMAX)
            axs.set_xticks([])
            axs.set_yticks([])
            axs.set_title(lab, fontsize=9, fontweight="bold", pad=3)
            axb = fig.add_subplot(gs[1, j])
            axb.set_xlim(0, TMAX)
            axb.set_ylim(0, 1)
            axb.axis("off")
            ax = fig.add_subplot(gs[2, j])
            t, sids, vals = data[(e, d)]
            for gi, g in enumerate(GROUPS):
                x = vals[[grp[s] == g for s in sids], hi]
                x = x[np.isfinite(x).all(1)]
                ax.plot(t, np.median(x, 0), color=COLORS[g], lw=1.0)
                for _, r in sigrows(e, d, hi, "delta", "Frontal", g).iterrows():
                    axb.plot(
                        [r.start_s, r.stop_s],
                        [0.8 - 0.3 * gi] * 2,
                        color=COLORS[g],
                        lw=2.0,
                        solid_capstyle="butt",
                    )
            ax.axhline(0, color="0.6", lw=0.6, ls="--")
            ax.set_xlim(0, TMAX)
            ax.set_xticklabels([])
            ax.margins(y=0.15)
            ax.tick_params(axis="y", labelsize=6.5)
            if j == 0:
                ax.set_ylabel("Frontal delta" + chr(10) + "log10 index − neutral", fontsize=7)
                axs.text(
                    -0.3,
                    1.0,
                    "a",
                    transform=axs.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    va="bottom",
                )
            # raster
            ax = fig.add_subplot(gs[4, j])
            for k in range(len(BANDS)):
                if k % 2 == 0:
                    ax.axhspan(
                        ypos[k * 3 + 2] - 0.5, ypos[k * 3] + 0.5, color="0.955", lw=0, zorder=0
                    )
            for (b, r), y0 in zip(rows, ypos):
                for g in GROUPS:
                    for _, rr in sigrows(e, d, hi, b, r, g).iterrows():
                        ax.plot(
                            [rr.start_s, rr.stop_s],
                            [y0 + OFF[g]] * 2,
                            color=COLORS[g],
                            lw=2.2,
                            solid_capstyle="butt",
                            zorder=3,
                        )
            ax.set_xlim(0, TMAX)
            ax.set_ylim(-0.6, len(rows) - 0.4)
            ax.set_xlabel("Time (s)", fontsize=7.5)
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.set_yticks(ypos)
            if j == 0:
                ax.set_yticklabels([f"{b.capitalize()} {r.lower()}" for b, r in rows], fontsize=7)
                ax.text(
                    -0.3,
                    1.01,
                    "b",
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    va="bottom",
                )
            else:
                ax.set_yticklabels([])
        fig.legend(
            handles=[Line2D([], [], color=COLORS[g], lw=2.5, label=g) for g in GROUPS],
            frameon=False,
            ncol=3,
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
            fontsize=8,
            handlelength=1.4,
        )
        fig.text(
            0.085,
            0.975,
            f"{title}: emotion − neutral, n = 148",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
        )
        fig.savefig(FIG / fname, dpi=300, facecolor="white")
        plt.close(fig)
        print("Saved:", shown(FIG / fname))


if __name__ == "__main__":
    run(parse_paths(__doc__, ("participants", "ratings", "cache", "neutral", "reports_dir")))
