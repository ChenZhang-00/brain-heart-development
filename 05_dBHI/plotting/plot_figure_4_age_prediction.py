"""Figure 4: Haufe topographies and descriptive age trajectories."""

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
    from scipy import stats

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from mne.channels.layout import _find_topomap_coords

    OUT = paths.age_results
    FIG = paths.figures
    FIG.mkdir(parents=True, exist_ok=True)
    P = json.load(open(OUT / "analysis_parameters.json"))
    BANDS = P["bands"]
    LABELS = P["channel_order"]
    F = np.load(OUT / "features_electrode_top30.npz", allow_pickle=False)
    age = F["age"] / 12.0
    feat = list(F["feature_names"])
    subj = pd.read_csv(OUT / "subjects_n148.csv")
    assert (subj.ID.astype(str).values == F["ids"]).all()
    grp = subj.group.values
    X = {
        m: F[m] - F[m].mean(1, keepdims=True)
        for m in ["bth_fear", "bth_happy", "htb_fear", "htb_happy"]
    }  # child-centred
    H = {m: pd.read_csv(OUT / f"haufe_patterns_{m}.csv").set_index("feature") for m in X}
    info = mne.create_info(LABELS, sfreq=4.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"), match_case=False)
    pos = _find_topomap_coords(info, picks="eeg")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "mathtext.fontset": "custom",
            "mathtext.it": "Arial:italic",
            "mathtext.rm": "Arial",
        }
    )
    EC = {"fear": "#1172BF", "happy": "#C60700"}
    EN = {"fear": "Fearful", "happy": "Happy"}
    GREEK = {"delta": "δ", "theta": "θ", "alpha": "α", "beta": "β"}
    # ---- selected clusters: (direction, hrv, band_fear, band_happy, title)
    SEL = {
        "bth": [
            ("LF", "beta", "beta", "EEG β → LF-HRV"),
            ("HF", "theta", "theta", "EEG θ → HF-HRV"),
        ],
        "htb": [
            ("LF", "beta", "beta", "LF-HRV → EEG β"),
            ("HF", "theta", "alpha", "HF-HRV → EEG θ (fearful) / α (happy)"),
        ],
    }

    def cluster(m, hrv, band):
        names = [f"{band}|{hrv}|{ch}" for ch in LABELS]
        sig = H[m].loc[names, "ci_excl0"].to_numpy().astype(bool)
        idx = [feat.index(n) for n, s in zip(names, sig) if s]
        if not idx:
            raise ValueError(
                f"No CI-selected electrodes for {m}/{hrv}/{band}; cannot draw the published age panel."
            )
        return X[m][:, idx].mean(1), int(sig.sum())

    # ---- layout
    W = 7.09
    fig = plt.figure(figsize=(W, 9.8))
    L, R = 0.085, 0.955
    gapc = 0.32
    ncol = 8
    wcell = (R - L) / (ncol + gapc)
    cols = [L + wcell * (j + (gapc if j >= 4 else 0)) for j in range(ncol)]
    topo_h = 0.076
    age_h = 0.155

    def block(y_top, direction, letter_topo, letters_age):
        # band labels
        for j in range(ncol):
            b = BANDS[j % 4]
            fig.text(
                cols[j] + wcell / 2, y_top - 0.004, f"EEG {b}", ha="center", va="bottom", fontsize=8
            )
        for k, hrv in enumerate(["LF", "HF"]):
            x0 = cols[4 * k]
            x1 = cols[4 * k + 3] + wcell
            fig.text(
                (x0 + x1) / 2,
                y_top + 0.016,
                f"{hrv}-HRV",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
            fig.add_artist(
                plt.Line2D(
                    [x0 + 0.004, x1 - 0.004],
                    [y_top + 0.014, y_top + 0.014],
                    transform=fig.transFigure,
                    color="k",
                    lw=0.6,
                )
            )
        fig.text(
            cols[0] + 0.03 - 0.062,
            y_top + 0.026,
            letter_topo,
            fontsize=15,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        fig.text(
            (cols[0] + cols[-1] + wcell) / 2,
            y_top + 0.034,
            "Brain→Heart" if direction == "bth" else "Heart→Brain",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
        ys = [y_top - 0.012 - topo_h, y_top - 0.012 - 2 * topo_h - 0.006]
        for r, emo in enumerate(["fear", "happy"]):
            m = f"{direction}_{emo}"
            d = H[m]
            lim = np.abs(d.pattern).max()
            y0 = ys[r]
            fig.text(
                L - 0.012,
                y0 + topo_h / 2,
                EN[emo],
                ha="right",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=EC[emo],
            )
            selected = {(hrv, bf if emo == "fear" else bh) for hrv, bf, bh, _ in SEL[direction]}
            for j in range(ncol):
                hrv = "LF" if j < 4 else "HF"
                b = BANDS[j % 4]
                if (hrv, b) in selected:
                    fig.add_artist(
                        FancyBboxPatch(
                            (cols[j] + 0.003, y0 + 0.002),
                            wcell - 0.006,
                            topo_h - 0.004,
                            boxstyle="round,pad=0.002,rounding_size=0.01",
                            transform=fig.transFigure,
                            fc="#ECECEC",
                            ec="none",
                            zorder=0,
                        )
                    )
                ax = fig.add_axes([cols[j], y0, wcell, topo_h])
                ax.set_zorder(1)
                names = [f"{b}|{hrv}|{ch}" for ch in LABELS]
                v = d.loc[names, "pattern"].to_numpy()
                sig = d.loc[names, "ci_excl0"].to_numpy().astype(bool)
                im, _ = mne.viz.plot_topomap(
                    v,
                    pos,
                    axes=ax,
                    show=False,
                    cmap="RdBu_r",
                    vlim=(-lim, lim),
                    contours=4,
                    sensors=False,
                    outlines="head",
                )
                if sig.any():
                    ax.scatter(
                        *pos[sig].T, s=16, c="k", edgecolors="white", linewidths=0.5, zorder=5
                    )
            cax = fig.add_axes([cols[-1] + wcell + 0.006, y0 + 0.12 * topo_h, 0.007, 0.76 * topo_h])
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=7, length=2, pad=1)
            cb.set_label("a.u.", fontsize=7.5, labelpad=1)
        # age panels
        y_age = ys[1] - 0.056 - age_h
        for k, (hrv, bf, bh, title) in enumerate(SEL[direction]):
            x0 = cols[4 * k] + 0.03
            x1 = cols[4 * k + 3] + wcell - 0.006
            ax = fig.add_axes([x0, y_age, x1 - x0, age_h])
            fig.text(
                x0 - 0.062,
                y_age + age_h + 0.006,
                letters_age[k],
                fontsize=15,
                fontweight="bold",
                ha="left",
                va="bottom",
            )
            for emo, band in [("fear", bf), ("happy", bh)]:
                m = f"{direction}_{emo}"
                y, n = cluster(m, hrv, band)
                c = EC[emo]
                ax.scatter(age, y, s=7, c=c, alpha=0.45, linewidths=0, zorder=2)
                sl, ic, r, p, se = stats.linregress(age, y)
                xx = np.linspace(age.min(), age.max(), 100)
                yy = ic + sl * xx
                res = y - (ic + sl * age)
                s2 = (res**2).sum() / (len(y) - 2)
                sxx = ((age - age.mean()) ** 2).sum()
                ci = stats.t.ppf(0.975, len(y) - 2) * np.sqrt(
                    s2 * (1 / len(y) + (xx - age.mean()) ** 2 / sxx)
                )
                ax.fill_between(xx, yy - ci, yy + ci, color=c, alpha=0.15, linewidths=0, zorder=1)
                ax.plot(xx, yy, color=c, lw=1.4, zorder=3)
                lab = f"{EN[emo]} ({n} electrodes), $r$ = {r:.2f}"
                ax.plot([], [], color=c, lw=1.4, label=lab)
            ax.set_xlim(4.8, 11.2)
            ax.set_xticks(range(5, 12))
            ax.set_xlabel("Age (years)", fontsize=9, labelpad=2)
            ax.set_ylabel("Relative modulation", fontsize=9, labelpad=2)
            ax.tick_params(labelsize=8, length=2.5, pad=2)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
            ax.set_title(title, fontsize=9.5, fontweight="bold", pad=3)
            ax.legend(frameon=False, fontsize=7.5, loc="best", handlelength=1.2)
        return y_age

    yb = block(0.945, "bth", "a", ["b", "c"])
    block(yb - 0.10, "htb", "d", ["e", "f"])
    fig.savefig(
        FIG / "Figure_4_age_prediction.png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    print("Saved:", shown(FIG / "Figure_4_age_prediction.png"))

    plt.close(fig)


if __name__ == "__main__":
    run(parse_paths(__doc__, ()))
