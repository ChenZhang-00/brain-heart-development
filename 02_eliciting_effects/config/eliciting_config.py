"""Fixed study settings and repository-relative paths for movie-evoked responses."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from importlib import import_module

_paths = import_module("common.paths")

EXPECTED_SUBJECTS = 148
MIN_LEN = 5  # seconds; multiply by FS_IBI for the IBI cluster test
CL_ALPHA = 0.05
N_PERM = 10000
BAND_LIMITS = [(1, 4), (4, 7), (7, 12), (12, 30)]
CONDITIONS = ["fear", "happy", "neutral"]
EMOTIONS = ["fear", "happy"]
BANDS = ["delta", "theta", "alpha", "beta"]
REGIONS = {
    "Frontal": [
        "Fp1",
        "Fp2",
        "AF7",
        "AF3",
        "AFz",
        "AF4",
        "AF8",
        "F9",
        "F7",
        "F5",
        "F3",
        "F1",
        "Fz",
        "F2",
        "F4",
        "F6",
        "F8",
        "F10",
    ],
    "Central": [
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "FC2",
        "FC4",
        "FC6",
        "FT8",
        "T7",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "T8",
        "TP7",
        "CP5",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "CP6",
        "TP8",
    ],
    "Posterior": [
        "P9",
        "P7",
        "P5",
        "P3",
        "P1",
        "Pz",
        "P2",
        "P4",
        "P6",
        "P8",
        "P10",
        "PO7",
        "PO3",
        "POz",
        "PO4",
        "PO8",
        "O1",
        "Oz",
        "O2",
    ],
}
T_EEG, FS_IBI, N_SURR, SEED = 176, 4, 10000, 0
EMO_COLOR = {"fear": "#1172BF", "happy": "#C60700", "neutral": "#A6A6A6"}
LABEL = {"fear": "Fearful", "happy": "Happy", "neutral": "Neutral"}
BAND_COLOR = {"delta": "#0000FF", "theta": "#FF0000", "alpha": "#000000", "beta": "#00A000"}
EMO_TITLE = {"fear": "The Lion King (fearful)", "happy": "The Jungle Book (happy)"}


def project_root(value=None):
    """Use an explicit root, the ELICITING_PROJECT_ROOT override, or this repository."""
    return _paths.project_root(value, env="ELICITING_PROJECT_ROOT")


def resolve_path(value, root):
    """Resolve user-supplied relative paths from the repository, not the shell cwd."""
    return _paths.resolve_path(value, root)


@dataclass(frozen=True)
class AnalysisPaths:
    root: Path
    participants: Path
    ecg: Path
    eeg_dir: Path
    ratings: Path
    output: Path

    @property
    def results(self):
        return self.output / "cache"

    @property
    def publication(self):
        base = self.root if self.output == self.root / "02_eliciting_effects" else self.output
        return base / "results"

    @property
    def figures(self):
        return self.publication / "SI"

    @property
    def tables(self):
        return self.publication / "SI"


def make_paths(
    *, root=None, participants=None, ecg_file=None, eeg_dir=None, ratings=None, output_root=None
):
    root = project_root(root)
    return AnalysisPaths(
        root=root,
        participants=resolve_path(participants or "00_data/participants.csv", root),
        ecg=resolve_path(ecg_file or "01_preprocessing/outputs/manuscript/ecg/hrv/ecg_ts.npz", root),
        eeg_dir=resolve_path(eeg_dir or "01_preprocessing/outputs/manuscript/eeg/band_power", root),
        ratings=resolve_path(ratings or "00_data/events/Movies_rating_by_RA.csv", root),
        output=resolve_path(
            output_root or os.environ.get("ELICITING_OUTPUT_ROOT") or "02_eliciting_effects", root
        ),
    )


def parse_analysis_args(description, argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--project-root", help="Repository root (default: locate from this script)."
    )
    parser.add_argument("--participants", help="Participant CSV (default: 00_data/participants.csv).")
    parser.add_argument("--ecg-file", help="Batch cardiac cache (default: 01_preprocessing/outputs/manuscript/ecg/hrv/ecg_ts.npz).")
    parser.add_argument("--eeg-dir", help="EEG power folder (default: 01_preprocessing/outputs/manuscript/eeg/band_power).")
    parser.add_argument(
        "--ratings", help="Intensity ratings (default: 00_data/events/Movies_rating_by_RA.csv)."
    )
    parser.add_argument(
        "--output-root", help="Working folder for cache/ and logs/ (default: 02_eliciting_effects/)."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate this stage's inputs without running statistics or writing outputs.",
    )
    args = parser.parse_args(argv)
    return (
        make_paths(
            root=args.project_root,
            participants=args.participants,
            ecg_file=args.ecg_file,
            eeg_dir=args.eeg_dir,
            ratings=args.ratings,
            output_root=args.output_root,
        ),
        args.check_only,
    )
