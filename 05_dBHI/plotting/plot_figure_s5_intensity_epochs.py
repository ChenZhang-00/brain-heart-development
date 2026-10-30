"""Figure S5: selected high- and low-intensity epochs."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.dbhi_config import parse_paths, shown


def run(paths):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    OUT = paths.intensity_results
    FIG = paths.supplementary
    FIG.mkdir(parents=True, exist_ok=True)
    ra = pd.read_csv(OUT / "smoothed_intensity.csv")
    selection = pd.read_csv(OUT / "ra_selected_epochs.csv")
    EMOTIONS, SHIFT = ["fear", "happy"], 7.5
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    for ax, emotion in zip(axes, EMOTIONS):
        ax.plot(ra.Time + SHIFT, ra[emotion.capitalize() + "_Intensity"], color="0.35", lw=1)
        for level, color in [("top30", "#C44E52"), ("bottom30", "#4C72B0")]:
            chosen = selection[(selection.emotion == emotion) & (selection.level == level)]
            ax.scatter(
                chosen.dbhi_epoch_start_s, chosen.smoothed_ra_score, s=16, color=color, label=level
            )
        ax.set_ylabel(f"{emotion.capitalize()} intensity")
        ax.legend(frameon=False, loc="upper left")
    axes[-1].set_xlabel("Movie time (s), window-centre alignment")
    fig.tight_layout()
    fig.savefig(FIG / "Figure_S5_intensity_epochs.png", dpi=220)
    plt.close(fig)
    print("Saved:", shown(FIG / "Figure_S5_intensity_epochs.png"))


if __name__ == "__main__":
    run(parse_paths(__doc__, ()))
