"""Render the four Figure 3 scalp panels from prepared map inputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import hashlib
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mne
import numpy as np

from importlib import import_module
_hep_config = import_module("04_HEP.config.hep_config")
CONDITION_LABELS = _hep_config.CONDITION_LABELS
ROI = _hep_config.ROI
make_paths = _hep_config.make_paths
relative_path = import_module("common.paths").relative_path
parse_args = _hep_config.parse_args
require = import_module("04_HEP.analysis.inputs").require


def figure_name(item):
    a, b = item["pair"].split(" - ")
    contrast = f"{CONDITION_LABELS[a].lower()}-{CONDITION_LABELS[b].lower()}"
    return (
        f"Figure_3_topography_{item['group']}_{item['region'].lower()}_{contrast}_"
        f"{item['t_start']:.0f}-{item['t_end']:.0f}ms.png"
    )


def figure_paths(paths):
    with np.load(paths.topographies / "temporal_cluster_map_inputs.npz", allow_pickle=False) as data:
        maps = json.loads(data["metadata_json"].item())["maps"]
    require(len(maps) == 4, "Expected four prepared scalp maps.")
    return [paths.figures / figure_name(item) for item in maps]


def load_maps(paths):
    """Read the four prepared maps and their shared colour range."""
    source = paths.topographies / "temporal_cluster_map_inputs.npz"
    if not source.is_file():
        raise FileNotFoundError(f"Missing {source}. Run the topography preparation step first.")
    with np.load(source, allow_pickle=False) as data:
        pos, labels, values = data["pos"], data["labels"].astype(str), data["values"]
        details = json.loads(data["metadata_json"].item())
    maps = [dict(**item, values=value) for item, value in zip(details["maps"], values)]
    require(
        values.shape == (4, 63) and pos.shape == (63, 2) and len(labels) == 63,
        "Expected four maps with all 63 channels and 2-D scalp positions.",
    )
    require(
        len(details["maps"]) == 4 and np.isfinite(values).all() and np.isfinite(pos).all(),
        "Invalid prepared map inputs.",
    )
    roi_masks = {region: np.isin(labels, channels) for region, channels in ROI.items()}
    limit = max(0.1, np.ceil(max(np.abs(item["values"]).max() for item in maps) * 10) / 10)
    return pos, maps, roi_masks, limit, details


def draw_map(ax, item, pos, roi, limit, *, sensor_size=14, line_width=1.8):
    """Draw one map on a standalone axis or a waveform inset."""
    image, _ = mne.viz.plot_topomap(
        item["values"], pos, axes=ax, show=False, cmap="RdBu_r",
        vlim=(-limit, limit), contours=4, sphere=(0.0, 0.0, 0.0, 0.5),
        extrapolate="head", sensors=False,
    )
    sensors = ax.scatter(
        pos[:, 0], pos[:, 1], s=sensor_size, c="black", linewidths=0,
        zorder=5, clip_on=False,
    )
    lower, upper = pos[roi].min(axis=0) - 0.025, pos[roi].max(axis=0) + 0.025
    assert np.array_equal(((pos >= lower) & (pos <= upper)).all(axis=1), roi)
    ax.add_patch(Rectangle(
        lower, *(upper - lower), fill=False, edgecolor="black",
        linewidth=line_width, zorder=6, clip_on=False,
    ))
    ax.set_xlim(-0.65, 0.65)
    ax.set_ylim(-0.59, 0.64)
    return image, sensors


def plot_topographies(paths=None, *, check_only=False):
    paths = paths or make_paths()
    OUT, FIG = paths.topographies, paths.figures
    pos, maps, roi_masks, limit, details = load_maps(paths)
    if check_only:
        print("Prepared topography plot inputs validated.")
        return
    FIG.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 12,
            "font.weight": "bold",
            "text.color": "black",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )
    figure_files = []
    for item in maps:
        fig = plt.figure(figsize=(4.6, 4.8))
        ax = fig.add_axes([0.10, 0.16, 0.80, 0.72])
        _, sensors = draw_map(ax, item, pos, roi_masks[item["region"]], limit)
        a, b = item["pair"].split(" - ")
        ax.set_title(f"{CONDITION_LABELS[a]} − {CONDITION_LABELS[b]}", fontsize=16, pad=12)
        interval = f"{item['t_start']:.0f}–{item['t_end']:.0f} ms"
        ax.text(
            0.5,
            -0.075,
            interval,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=15,
            clip_on=False,
        )
        name = figure_name(item)
        assert len(fig.axes) == 1 and len(sensors.get_offsets()) == 63
        fig.savefig(FIG / name, dpi=300, facecolor="white")
        figure_files.append(name)
        print("Saved:", relative_path(FIG / name, paths.root))
        plt.close(fig)

    assert len(figure_files) == 4
    manifest = {
        "maps": [{k: v for k, v in item.items() if k != "values"} for item in maps],
        "layout": "four_individual_panels",
        "figure_files": figure_files,
        "frontal_union_samples": 108,
        "shared_300ms_sample_counted_once": True,
        "new_statistical_tests": False,
        "p_values_displayed": False,
        "colorbars": 0,
        "shared_color_limits_uV": [-float(limit), float(limit)],
        "channels_per_map": 63,
        "source_sha256": details["source_sha256"],
        "figure_sha256": {
            name: hashlib.sha256((FIG / name).read_bytes()).hexdigest() for name in figure_files
        },
    }
    (OUT / "temporal_cluster_map_manifest_final.json").write_text(
        json.dumps(manifest, indent=2).replace(str(paths.root) + "/", "") + "\n"
    )

    return [FIG / name for name in figure_files]


def main(argv=None):
    paths, check_only = parse_args(__doc__, argv)
    plot_topographies(paths, check_only=check_only)


if __name__ == "__main__":
    main()
