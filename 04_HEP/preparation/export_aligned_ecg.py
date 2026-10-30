"""
Export the ECG on the EEG's trimmed movie timeline, for regress_ecg_from_eeg.m.

For every subject with a continuous_clean.mat:
  - read the full-recording ECG saved by the preprocessing step
    ({id}_movie_ecg.npz from extract_ecg.py: BrainVision channel '65'), flip so R is up
  - 0.1-45 Hz Butterworth order 4, zero-phase   (= preprocess_eeg P04 main band)
  - cut to the three movie segments with seg_info.onsets_orig/offsets_orig
    (1-based MATLAB latencies stored in continuous_clean.mat) and concatenate,
    which is exactly preprocess_eeg P06 -> same length as EEG.pnts
Output: 04_HEP/outputs/demo/ecg_trimmed/{id}_ecg_trimmed.npz  {ecg, fs, n_samples}

Usage: python 04_HEP/preparation/export_aligned_ecg.py --ids sub-001
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
PRE_DIR = ROOT / "01_preprocessing/outputs/demo/eeg/cleaned"
ECG_DIR = ROOT / "01_preprocessing/outputs/demo/ecg/extracted"
OUT_DIR = ROOT / "04_HEP/outputs/demo" / "ecg_trimmed"
FLIP = -1.0
BP, ORDER = (0.1, 45.0), 4


def main(ecg_dir: Path, overwrite: bool, ids=None, *, pre_dir=PRE_DIR, out_dir=OUT_DIR):
    ecg_dir, pre_dir, out_dir = map(Path, (ecg_dir, pre_dir, out_dir))
    files = (
        [pre_dir / f"{sid}_continuous_clean.mat" for sid in ids]
        if ids is not None else sorted(pre_dir.glob("*_continuous_clean.mat"))
    )
    if not files:
        raise FileNotFoundError(f"No preprocessed EEG in {pre_dir}; run preprocess_eeg first.")
    if not ecg_dir.is_dir():
        raise FileNotFoundError(f"Extracted ECG directory does not exist: {ecg_dir}")
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Preprocessed EEG not found: {path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, fp in enumerate(files):
        sid = fp.name.removesuffix("_continuous_clean.mat")
        out = out_dir / f"{sid}_ecg_trimmed.npz"
        if out.exists() and not overwrite:
            continue
        with h5py.File(fp, "r") as f:
            on1 = np.asarray(f["seg_info"]["onsets_orig"][()]).ravel().astype(int)
            off1 = np.asarray(f["seg_info"]["offsets_orig"][()]).ravel().astype(int)
            n_eeg = f["EEG"]["data"].shape[0]
            fs_eeg = float(np.asarray(f["EEG"]["srate"][()]).ravel()[0])
        ecg_file = ecg_dir / f"{sid}_movie_ecg.npz"
        if not ecg_file.is_file():
            raise FileNotFoundError(f"Extracted ECG not found (run preprocessing step 5): {ecg_file}")
        with np.load(ecg_file, allow_pickle=False) as source:
            fs = float(source["fs"])
            ecg = FLIP * source["ecg"].astype(np.float64)
        assert abs(fs - fs_eeg) < 1e-6, f"{sid}: fs {fs} vs EEG {fs_eeg}"
        assert len(ecg) >= off1.max(), f"{sid}: extracted ECG shorter than the EEG segments"
        sos = signal.butter(ORDER, BP, btype="bandpass", fs=fs, output="sos")
        ecg = signal.sosfiltfilt(sos, ecg)
        trimmed = np.concatenate([ecg[a - 1 : b] for a, b in zip(on1, off1)])
        assert len(trimmed) == n_eeg, f"{sid}: trimmed {len(trimmed)} vs EEG {n_eeg}"
        np.savez(
            out, ecg=trimmed.astype(np.float64), fs=np.int32(fs), n_samples=np.int64(len(trimmed))
        )
        print(f"Aligned ECG {k + 1}/{len(files)}: {sid}, {len(trimmed)} samples")
    try:
        shown = out_dir.resolve().relative_to(ROOT)
    except ValueError:
        shown = out_dir
    print("Saved:", shown)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecg-dir", type=Path, default=ECG_DIR, help="Folder of {ID}_movie_ecg.npz from preprocessing step 5")
    ap.add_argument("--preprocessed", type=Path, default=PRE_DIR)
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--ids", nargs="+", help="Process only these subject IDs")
    a = ap.parse_args()
    main(a.ecg_dir, a.overwrite, a.ids, pre_dir=a.preprocessed, out_dir=a.output_dir)
