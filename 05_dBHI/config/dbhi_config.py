"""Shared paths and fixed settings for the submitted directional analyses."""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import sys

DBHI = Path(__file__).resolve().parents[1]
ROOT = DBHI.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from importlib import import_module

_paths = import_module("common.paths")
BANDS = ["delta", "theta", "alpha", "beta"]
HRV = ["LF", "HF"]
FS = 4
GROUPS = {"5-6 yrs": (5, 6), "7-8 yrs": (7, 8), "9-10 yrs": (9, 10)}
REGIONS = {
    "Frontal": [1, 32, 33, 34, 35, 62, 63, 5, 4, 37, 3, 36, 2, 61, 31, 60, 30, 29],
    "Central": [
        38,
        6,
        39,
        7,
        27,
        58,
        28,
        59,
        9,
        41,
        8,
        40,
        26,
        57,
        25,
        56,
        24,
        42,
        10,
        43,
        11,
        53,
        22,
        54,
        23,
        55,
    ],
    "Posterior": [15, 14, 45, 13, 44, 12, 52, 21, 51, 20, 19, 46, 47, 48, 49, 50, 16, 17, 18],
}


def shown(path):
    return _paths.relative_path(path, ROOT)


def resolve_path(value):
    return _paths.resolve_path(value, ROOT)


@dataclass(frozen=True)
class Paths:
    output_root: Path = DBHI
    participants: Path = ROOT / "00_data/participants.csv"
    ratings: Path = ROOT / "00_data/events/Movies_rating_by_RA.csv"
    cache: Path = DBHI / "outputs/manuscript/model"
    neutral_subject: str = "sub-047"  # neutral clip truncated to the R-peak-supported segment
    reports_dir: Path | None = None

    @property
    def reports(self):
        return self.reports_dir or self.output_root / "outputs/manuscript/model_reports"

    @property
    def time_results(self):
        return self.output_root / "cache/time_resolved"

    @property
    def intensity_results(self):
        return self.output_root / "cache/intensity_contrast"

    @property
    def spatial_results(self):
        return self.intensity_results / "topo_cluster"

    @property
    def age_results(self):
        return self.output_root / "cache/age_prediction"

    @property
    def publication(self):
        base = ROOT if self.output_root == DBHI else self.output_root
        return base / "results"

    @property
    def figures(self):
        return self.publication / "Main"

    @property
    def supplementary(self):
        return self.publication / "SI"

    @property
    def neutral_file(self):
        return self.cache / f"{self.neutral_subject}_dbhi.npz"


def make_parser(description, inputs=()):
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--output-root",
        type=resolve_path,
        default=DBHI,
        help="Working folder for cache/ and logs/ (default: 05_dBHI/).",
    )
    options = {
        "participants": ("--participants", ROOT / "00_data/participants.csv"),
        "ratings": ("--ratings", ROOT / "00_data/events/Movies_rating_by_RA.csv"),
        "cache": ("--cache-dir", DBHI / "outputs/manuscript/model"),
        "reports_dir": ("--reports-dir", None),
    }
    for name in inputs:
        if name == "neutral":
            continue
        option, default = options[name]
        parser.add_argument(option, dest=name, type=resolve_path, default=default)
    if "neutral" in inputs:
        parser.add_argument(
            "--neutral-subject",
            default="sub-047",
            help="ID whose neutral clip is truncated to its R-peak-supported segment.",
        )
    return parser


def paths_from_args(args):
    return Paths(**{k: v for k, v in vars(args).items() if k in Paths.__dataclass_fields__})


def parse_paths(description, inputs=(), argv=None):
    return paths_from_args(make_parser(description, inputs).parse_args(argv))
