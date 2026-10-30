"""Figure 3: regional HEP waveforms with four scalp insets and a shared colour bar."""

from importlib import import_module
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
settings = import_module("04_HEP.config.hep_config")
waveforms = import_module("04_HEP.plotting.plot_hep_waveforms")
topographies = import_module("04_HEP.plotting.plot_hep_cluster_topographies")
relative_path = import_module("common.paths").relative_path

INSET_BOUNDS = {
    ("5-6", "Frontal"): (0.66, 0.58, 0.34, 0.44),
    ("5-6", "Central"): (0.00, 0.58, 0.34, 0.44),
    ("5-6", "Posterior"): (0.00, 0.02, 0.34, 0.44),
    ("7-8", "Central"): (0.66, 0.03, 0.34, 0.44),
}


def build_figure(paths):
    pos, maps, roi_masks, limit, _ = topographies.load_maps(paths)
    expected = set(INSET_BOUNDS)
    actual = {(item["group"], item["region"]) for item in maps}
    if actual != expected or len(maps) != len(expected):
        raise ValueError("Figure 3 requires the four reported age-group/region scalp maps.")

    fig, axes = waveforms.build_waveform_figure(paths)
    fig.set_size_inches(15.5, 12.0)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.065, top=0.875,
                        hspace=0.23, wspace=0.30)
    for row, group in enumerate(settings.GROUPS):
        for text in list(axes[row, 0].texts):
            if text.get_text() == group.replace("-", "–") + " years":
                text.remove()
        axes[row, 0].set_ylabel(
            group.replace("-", "–") + " years\nAmplitude (µV)", fontsize=12, labelpad=12,
        )

    # Leave room above the youngest group's waveforms for the two upper insets.
    for region in ("Frontal", "Central"):
        ax = axes[0, settings.REGIONS.index(region)]
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + 0.25 * (high - low))

    for item in maps:
        key = item["group"], item["region"]
        ax = axes[settings.GROUPS.index(key[0]), settings.REGIONS.index(key[1])]
        inset = ax.inset_axes(INSET_BOUNDS[key], zorder=8)
        inset.patch.set_alpha(0)
        topographies.draw_map(
            inset, item, pos, roi_masks[item["region"]], limit,
            sensor_size=5, line_width=1.5,
        )

    for legend in list(fig.legends):
        legend.remove()
    condition_handles = [
        Line2D([], [], color=settings.COLORS[c], lw=2.3, label=settings.CONDITION_LABELS[c])
        for c in settings.CONDITIONS
    ]
    contrast_handles = [
        Patch(color=color, label=" vs ".join(settings.CONDITION_LABELS[c] for c in pair.split(" - ")))
        for pair, color in settings.PAIR_COLORS.items()
    ]
    for handles, title, left in [
        (condition_handles, "Conditions", 0.095),
        (contrast_handles, "Cluster contrasts", 0.385),
    ]:
        fig.legend(handles=handles, title=title, ncol=3, loc="upper left",
                   bbox_to_anchor=(left, 0.985), frameon=False, fontsize=10.5,
                   title_fontsize=12, columnspacing=1.2, handlelength=1.7,
                   handletextpad=0.5, borderaxespad=0)

    color_axis = fig.add_axes([0.865, 0.935, 0.11, 0.016])
    scale = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-limit, limit), cmap="RdBu_r")
    colorbar = fig.colorbar(scale, cax=color_axis, orientation="horizontal", ticks=[-limit, 0, limit])
    colorbar.ax.set_title("Δ HEP (µV)", fontsize=11, pad=6)
    colorbar.ax.tick_params(labelsize=10, length=3, pad=2)
    return fig


def plot_figure(paths=None, *, check_only=False):
    paths = paths or settings.make_paths()
    if check_only:
        waveforms.plot_waveforms(paths, check_only=True)
        topographies.plot_topographies(paths, check_only=True)
        return
    paths.figures.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(mpl.rcParamsDefault):
        fig = build_figure(paths)
        output = paths.figures / "Figure_3_hep.png"
        fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
        print("Saved:", relative_path(output, paths.root))
        plt.close(fig)
    return output


def main(argv=None):
    paths, check_only = settings.parse_args(__doc__, argv)
    plot_figure(paths, check_only=check_only)


if __name__ == "__main__":
    main()
