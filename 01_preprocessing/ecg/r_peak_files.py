"""Find saved R peaks that match the extracted ECG."""

from pathlib import Path
from zipfile import BadZipFile

import numpy as np


def has_corrected_rpeaks(subject_id, outputs_dir):
    """Return False for missing/empty corrections; reject mismatched files."""
    outputs_dir = Path(outputs_dir)
    raw_path = outputs_dir / "ecg/extracted" / f"{subject_id}_movie_ecg.npz"
    peak_path = outputs_dir / "ecg/r_peaks" / f"{subject_id}_movie_rpeaks.npz"
    if not peak_path.is_file() or peak_path.stat().st_size == 0:
        return False
    if not raw_path.is_file():
        raise FileNotFoundError(f"Extract ECG first: {raw_path}")

    try:
        with np.load(raw_path, allow_pickle=False) as raw, np.load(peak_path, allow_pickle=False) as peaks:
            r = peaks["r_peaks"]
            matches = (
                str(peaks["subject_id"].item()) == subject_id
                and int(peaks["fs"]) == int(raw["fs"]) == 500
                and int(peaks["n_samples"]) == int(raw["n_samples"])
                and np.array_equal(peaks["event_sample"], raw["event_sample"])
                and np.array_equal(peaks["event_desc"], raw["event_desc"])
            )
            if matches and r.ndim == 1 and r.size < 2:
                return False
            valid = (
                matches and r.ndim == 1 and r.size >= 2
                and np.issubdtype(r.dtype, np.integer)
                and np.all(np.diff(r) > 0)
                and np.all((r >= 0) & (r < int(raw["n_samples"])))
            )
    except (OSError, EOFError, BadZipFile, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{subject_id}: cannot read ECG/R-peak files: {error}") from error
    if not valid:
        raise ValueError(f"{peak_path.name}: R peaks do not match the extracted ECG.")
    return True


def split_corrected_rpeaks(subjects, outputs_dir):
    """Return completed and pending IDs in the selected order."""
    completed, pending = [], []
    for subject_id in subjects:
        if has_corrected_rpeaks(subject_id, outputs_dir):
            completed.append(subject_id)
        else:
            pending.append(subject_id)
    return completed, pending
