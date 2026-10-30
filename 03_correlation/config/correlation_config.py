"""Fixed settings and repository-relative paths for EEG–IBI covariation."""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from importlib import import_module

_paths = import_module("common.paths")

import numpy as np


def project_root(value=None):
    """Use an explicit root, the EEG_IBI_PROJECT_ROOT override, or this repository."""
    return _paths.project_root(value, env="EEG_IBI_PROJECT_ROOT")


def resolve_path(value, root):
    """Resolve user-supplied relative paths from the repository, not the shell cwd."""
    return _paths.resolve_path(value, root)


EXPECTED_SUBJECTS = 148
CONDITIONS = ["neutral", "fear", "happy"]
BANDS = ["delta", "theta", "alpha", "beta"]
BAND_LIMITS = [(1, 4), (4, 7), (7, 12), (12, 30)]
GROUPS = ["5-6", "7-8", "9-10"]
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
PRIMARY_REGION = "Frontal"
T = 176
FS_IBI = 4.0
IBI_SAMPLES_PER_BIN = 4
EEG_DROP = 1
TIME = np.arange(2.0, 178.0)
MIN_PAIRED_SAMPLES = 60
ALPHA = 0.05
