"""Fixed HEP study settings and repository-relative paths."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from importlib import import_module

_paths = import_module("common.paths")

EXPECTED_SUBJECTS = 148
EXPECTED_CHANNELS = 63
TIME_STEP_MS = 2
GROUPS = ["5-6", "7-8", "9-10"]
REGIONS = ["Frontal", "Central", "Posterior"]
CONDITIONS = ["fear", "neutral", "happy"]
PAIRS = [("fear", "neutral"), ("happy", "neutral"), ("fear", "happy")]
WINDOWS = [(100, 300), (300, 500)]
ALPHA, MIN_MS, N_PERM, SEED = 0.05, 20, 10_000, 0
COLORS = {"fear": "#1172BF", "neutral": "#7F7F7F", "happy": "#C60700"}
PAIR_COLORS = {
    "fear - neutral": COLORS["fear"],
    "happy - neutral": COLORS["happy"],
    "fear - happy": "#7B3FA0",
}
CONDITION_LABELS = {"fear": "Fearful", "neutral": "Neutral", "happy": "Happy"}
ROI = {
    "Frontal": "Fp1 Fz F3 F7 F9 F10 F8 F4 Fp2 AF7 AF3 AFz F1 F5 F6 F2 AF4 AF8".split(),
    "Central": "FC5 FC1 C3 T7 CP5 CP1 CP2 CP6 T8 C4 Cz FC2 FC6 FT7 FC3 C1 C5 TP7 CP3 CPz CP4 TP8 C6 C2 FC4 FT8".split(),
    "Posterior": "Pz P3 P7 P9 O1 Oz O2 P10 P8 P4 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2".split(),
}
MIN_SAMPLES = round(MIN_MS / TIME_STEP_MS)


def project_root(value=None):
    """Use an explicit root, the HEP_PROJECT_ROOT override, or this repository."""
    return _paths.project_root(value, env="HEP_PROJECT_ROOT")


def resolve_path(value, root):
    """Resolve user-supplied relative paths from the repository, not the shell cwd."""
    return _paths.resolve_path(value, root)


@dataclass(frozen=True)
class AnalysisPaths:
    root: Path
    participants: Path
    epochs: Path
    roi: Path
    channel_dir: Path
    output: Path

    @property
    def results(self):
        return self.output / "cache"

    @property
    def topographies(self):
        return self.results / "topographies"

    @property
    def publication(self):
        base = self.root if self.output == self.root / "04_HEP" else self.output
        return base / "results"

    @property
    def figures(self):
        return self.publication / "Main"

    @property
    def tables(self):
        return self.publication / "SI"

    @property
    def clusters(self):
        return self.results / "condition_within_group_ecgreg_final.csv"


def make_paths(
    *,
    root=None,
    participants=None,
    epochs_dir=None,
    roi_file=None,
    channel_dir=None,
    output_root=None,
):
    root = project_root(root)
    output = resolve_path(output_root or os.environ.get("HEP_OUTPUT_ROOT") or "04_HEP", root)
    return AnalysisPaths(
        root=root,
        participants=resolve_path(participants or "00_data/participants.csv", root),
        epochs=resolve_path(epochs_dir or output / "outputs/batch/hep_epochs_ecgreg", root),
        roi=resolve_path(roi_file or "04_HEP/outputs/manuscript/averages/roi_waves_ecgreg.mat", root),
        channel_dir=resolve_path(channel_dir or "04_HEP/outputs/manuscript/averages", root),
        output=output,
    )


def parse_args(description, argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--project-root", help="Repository root (default: locate from this script)."
    )
    parser.add_argument("--participants", help="Participant CSV (default: 00_data/participants.csv).")
    parser.add_argument(
        "--epochs-dir", help="ECG-regressed epoch directory (default: <output>/outputs/batch/hep_epochs_ecgreg)."
    )
    parser.add_argument(
        "--roi-file", help="ROI MAT file (default: 04_HEP/outputs/manuscript/averages/roi_waves_ecgreg.mat)."
    )
    parser.add_argument(
        "--channel-dir",
        help="Channel averages and manifest (default: 04_HEP/outputs/manuscript/averages).",
    )
    parser.add_argument("--output-root", help="Working folder for outputs/, cache/ and logs/ (default: 04_HEP/).")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs without writing outputs or running tests.",
    )
    args = parser.parse_args(argv)
    return (
        make_paths(
            root=args.project_root,
            participants=args.participants,
            epochs_dir=args.epochs_dir,
            roi_file=args.roi_file,
            channel_dir=args.channel_dir,
            output_root=args.output_root,
        ),
        args.check_only,
    )
