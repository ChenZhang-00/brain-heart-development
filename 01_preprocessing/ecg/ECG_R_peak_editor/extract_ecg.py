#!/usr/bin/env python3
"""Extract lead-II ECG and movie markers from BrainVision recordings."""

import argparse
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ECG_CHANNEL = "65"  # BrainVision label; R peaks point downward.
TASK = "movie"


def select_movie_markers(descriptions, samples, fs):
    """Select the last complete 150–260 s take of each clip, as in preprocess_eeg.m."""
    selected = []
    for start_code in (1, 3, 5):
        onsets = np.flatnonzero(descriptions == f"S{start_code}")
        offsets = np.flatnonzero(descriptions == f"S{start_code + 1}")
        pairs = [
            (i, j) for i in onsets for j in offsets if 150 <= (samples[j] - samples[i]) / fs <= 260
        ]
        if not pairs:
            raise ValueError(f"No complete S{start_code}/S{start_code+1} movie segment.")
        selected.extend(max(pairs, key=lambda pair: samples[pair[0]]))
    return sorted(selected, key=lambda i: samples[i])


def clean_desc(desc: str) -> str:
    """`'Stimulus/S  1'` -> `'S1'`; `'New Segment/'` -> `'NewSeg'`; else passthrough."""
    if desc.startswith("Stimulus/S"):
        m = re.search(r"\d+", desc)
        if m:
            return f"S{int(m.group())}"
    if "New Segment" in desc:
        return "NewSeg"
    return desc.strip()


def extract_one(vhdr_path: Path, out_path: Path) -> tuple[int, int, float]:
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose="ERROR")
    if ECG_CHANNEL not in raw.ch_names:
        raise ValueError(f"channel {ECG_CHANNEL!r} not found; channels are {raw.ch_names}")
    ecg = raw.get_data(picks=[ECG_CHANNEL])[0].astype(np.float64)
    fs = float(raw.info["sfreq"])
    time = np.arange(len(ecg), dtype=np.float64) / fs

    onsets = raw.annotations.onset
    raw_descs = raw.annotations.description
    cleaned = [clean_desc(d) for d in raw_descs]
    keep = [c.startswith("S") and c != "NewSeg" for c in cleaned]
    ev_time = np.array([o for o, k in zip(onsets, keep) if k], dtype=np.float64)
    ev_desc = np.array([c for c, k in zip(cleaned, keep) if k])
    ev_sample = np.round(ev_time * fs).astype(np.int64)

    selected = select_movie_markers(ev_desc, ev_sample, fs)
    ev_desc, ev_sample = ev_desc[selected], ev_sample[selected]
    ev_time = ev_sample / fs
    sid = vhdr_path.stem.replace(f"_{TASK}", "")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Finish writing before replacing an existing output in a synced folder.
    with TemporaryDirectory(dir=out_path.parent, prefix=".ecg-") as temp_dir:
        temp_path = Path(temp_dir) / out_path.name
        np.savez(
            temp_path,
            ecg=ecg,
            time=time,
            fs=np.int32(round(fs)),
            ch_name=np.array(ECG_CHANNEL),
            event_sample=ev_sample,
            event_time=ev_time,
            event_desc=ev_desc,
            duration_sec=np.float64(time[-1] if len(time) else 0.0),
            n_samples=np.int64(len(ecg)),
            subject_id=np.array(sid),
            task=np.array(TASK),
        )
        temp_path.replace(out_path)
    return len(ecg), len(ev_desc), float(time[-1] if len(time) else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "00_data/raw_data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "01_preprocessing/outputs/demo/ecg/extracted")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    files = sorted(args.input_root.rglob("*_movie.vhdr"))
    if not files:
        raise FileNotFoundError(f"No BrainVision movie recordings in {args.input_root}")
    if len({p.name for p in files}) != len(files):
        raise ValueError("Duplicate recording filenames in the input directory.")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for path in files:
        output = args.output_root / f"{path.stem}_ecg.npz"
        if output.exists() and not args.overwrite:
            print("Already extracted:", output.name)
            continue
        samples, events, duration = extract_one(path, output)
        print(f"{output.name}: {samples} samples, {events} markers, {duration:.1f} s")


if __name__ == "__main__":
    main()
