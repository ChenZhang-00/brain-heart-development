"""Export all-channel condition means from ECG-regressed MATLAB v7.3 epochs.

Only retained beats are averaged. No ROI averaging, filtering, baseline change,
artifact rejection, resampling, or amplitude scaling is applied to the cache.
Requires numpy, scipy and h5py. A complete metadata-matched cohort is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

CONDITIONS = ("fear", "neutral", "happy")


def matlab_string(f, ref):
    return "".join(chr(int(c)) for c in np.asarray(f[ref][()]).ravel())


def numeric_refs(f, dataset):
    return np.array([float(np.asarray(f[r][()]).ravel()[0]) for r in dataset[()].ravel()])


def read_subject(path, chunk_size=64):
    with h5py.File(path, "r") as f:
        eeg = f["EEG"]
        times = np.asarray(eeg["times"][()], dtype=np.float64).ravel()
        srate = float(eeg["srate"][()].ravel()[0])
        loc = eeg["chanlocs"]
        labels = np.array([matlab_string(f, r) for r in loc["labels"][()].ravel()])
        theta = np.deg2rad(numeric_refs(f, loc["theta"]))
        radius = numeric_refs(f, loc["radius"])
        pos = np.column_stack((radius * np.sin(theta), radius * np.cos(theta)))
        conds = np.array([matlab_string(f, r).strip().lower() for r in f["conditions"][()].ravel()])
        data = eeg["data"]  # HDF5 reverses MATLAB channel x time x beat axes.
        if len(labels) != 63 or len(set(labels)) != 63:
            raise ValueError("Expected 63 unique channel labels")
        if data.shape != (len(conds), len(times), len(labels)):
            raise ValueError(f"Unexpected EEG.data shape {data.shape}")
        if not np.isfinite(times).all() or not np.isfinite(pos).all():
            raise ValueError("Nonfinite time axis or channel positions")
        if len(times) < 2 or srate <= 0 or not np.isfinite(srate):
            raise ValueError("Invalid time axis or sampling rate")
        if not np.allclose(np.diff(times), 1000.0 / srate, rtol=0, atol=1e-6):
            raise ValueError("Time spacing differs from EEG.srate")
        if set(conds) != set(CONDITIONS):
            raise ValueError(f"Missing or unexpected conditions: {set(conds)}")
        means, counts = {}, []
        for cond in CONDITIONS:
            selected = np.flatnonzero(conds == cond)
            counts.append(len(selected))
            acc = np.zeros((len(times), len(labels)), dtype=np.float64)
            for start in range(0, len(selected), chunk_size):
                block = data[selected[start : start + chunk_size], :, :]
                if not np.isfinite(block).all():
                    raise ValueError(f"NaN/Inf in {cond} source epochs")
                acc += block.sum(axis=0, dtype=np.float64)
            means[cond] = (acc / len(selected)).astype(np.float32)
            if not np.isfinite(means[cond]).all():
                raise ValueError(f"NaN/Inf in {cond} mean")
    return times, labels, pos, srate, means, np.array(counts, dtype=np.int64)


def check_roi(reference_path, config_path, ids, times, means, counts):
    ref = loadmat(reference_path, simplify_cells=True)
    ref_ids = [str(s) for s in np.atleast_1d(ref["ids"])]
    if len(set(ref_ids)) != len(ref_ids):
        raise ValueError("Duplicate ROI reference IDs")
    missing = sorted(set(ids) - set(ref_ids))
    if missing:
        raise ValueError(f"ROI reference is missing IDs: {missing}")
    order = [ref_ids.index(s) for s in ids]
    np.testing.assert_allclose(times, np.asarray(ref["times"]).ravel(), rtol=0, atol=1e-6)
    # simplify_cells squeezes the subject axis for a one-participant reference.
    reference_counts = np.asarray(ref["n_trials"]).reshape(len(ref_ids), len(CONDITIONS))
    np.testing.assert_array_equal(counts, reference_counts[order])
    config = config_path.read_text(encoding="utf-8-sig")
    rois = re.findall(r"CFG\.roi\.(\w+)\s*=\s*\[([^\]]+)\]", config)
    if not rois:
        raise ValueError("No ROI index definitions found in preprocessing_config.m")
    comparisons = {}
    for region, expression in rois:
        if not re.fullmatch(r"[\d,\s]+", expression):
            raise ValueError(f"Unsupported ROI expression for {region}")
        channels = np.array([int(n) - 1 for n in re.findall(r"\d+", expression)])
        if (channels < 0).any() or (channels >= 63).any():
            raise ValueError(f"Invalid ROI indices for {region}")
        for cond in CONDITIONS:
            key = f"{region}__{cond}"
            actual = means[cond][:, :, channels].mean(axis=2, dtype=np.float64)
            expected = np.asarray(ref[key]).reshape(len(ref_ids), len(times))[order]
            if actual.shape != expected.shape or not np.isfinite(expected).all():
                raise ValueError(f"Invalid ROI reference array: {key}")
            diff = actual - expected
            comparisons[key] = {
                "max_abs_error": float(np.abs(diff).max()),
                "rmse": float(np.sqrt(np.mean(diff**2))),
                "passed": bool(np.allclose(actual, expected, rtol=1e-6, atol=1e-6)),
            }
    return {
        "status": "passed" if all(v["passed"] for v in comparisons.values()) else "failed",
        "reference": str(reference_path),
        "config": str(config_path),
        "n_subjects": len(ids),
        "n_trials_match": True,
        "atol": 1e-6,
        "rtol": 1e-6,
        "comparisons": comparisons,
    }


def main(argv=None):
    here = Path(__file__).resolve().parents[1]
    root = here.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "04_HEP/outputs/batch/hep_epochs_ecgreg")
    parser.add_argument("--output-dir", type=Path, default=here / "outputs/batch/averages")
    parser.add_argument("--info-csv", type=Path, default=root / "00_data/participants.csv")
    parser.add_argument(
        "--config", type=Path, default=root / "01_preprocessing/config/preprocessing_config.m"
    )
    parser.add_argument("--roi-reference", type=Path)
    parser.add_argument("--expected-subjects", type=int, default=148)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args(argv)
    # Resolve CLI paths from the repository, as in the entry notebook.
    for name in ["input_dir", "output_dir", "info_csv", "config", "roi_reference"]:
        value = getattr(args, name)
        if value is not None:
            value = value.expanduser()
            setattr(args, name, value if value.is_absolute() else root / value)
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    inp, out = args.input_dir.resolve(), args.output_dir.resolve()
    if (out / "chan_means.npz").exists():
        parser.error("Cache already exists; use a new output directory to avoid overwriting it")
    with args.info_csv.open(encoding="utf-8-sig", newline="") as stream:
        ids = [row["ID"].strip() for row in csv.DictReader(stream)]
    if len(ids) != args.expected_subjects or len(set(ids)) != len(ids) or "" in ids:
        parser.error("Metadata IDs must be unique, nonempty, and match expected cohort size")
    if not inp.is_dir():
        parser.error(f"Input directory does not exist: {inp}")
    out.mkdir(parents=True, exist_ok=True)
    extra = sorted(p.name for p in inp.glob("*_hep_epochs.mat") if p.name.split("_")[0] not in ids)
    rows, subjects, source_files = [], {}, []
    template = None
    for index, sid in enumerate(ids, 1):
        path = inp / f"{sid}_hep_epochs.mat"
        row = {
            "ID": sid,
            "fear": "",
            "neutral": "",
            "happy": "",
            "status": "missing",
            "error": "File not found",
        }
        if path.exists():
            try:
                result = read_subject(path, args.chunk_size)
                times, labels, pos, srate, means, counts = result
                row.update(dict(zip(CONDITIONS, counts.tolist())))
                if template is not None:
                    np.testing.assert_allclose(times, template[0], rtol=0, atol=1e-6)
                    np.testing.assert_array_equal(labels, template[1])
                    np.testing.assert_allclose(pos, template[2], rtol=0, atol=1e-10)
                    if srate != template[3]:
                        raise ValueError("Inconsistent sampling rate")
                else:
                    template = result[:4]
                subjects[sid] = (means, counts)
                stat = path.stat()
                source_files.append(
                    {"ID": sid, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                )
                row.update(status="passed", error="")
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        if index % 10 == 0 or row["status"] != "passed" or index == len(ids):
            print(f"{index}/{len(ids)} {sid}: {row['status']}", flush=True)
    manifest = {
        "data_version": "ECG-regressed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(inp),
        "output_dir": str(out),
        "info_csv": str(args.info_csv.resolve()),
        "condition_order": list(CONDITIONS),
        "axis_order": ["subject", "time", "channel"],
        "expected_subjects": len(ids),
        "successful_subjects": len(subjects),
        "missing_ids": [r["ID"] for r in rows if r["status"] == "missing"],
        "failed_ids": [r["ID"] for r in rows if r["status"] == "failed"],
        "extra_files_excluded": extra,
        "source_files": source_files,
        "amplitude_units": "unchanged from input EEG.data (pipeline: microvolts)",
        "reduction": "mean over all retained beats per condition only; no channel or time averaging",
        "accumulator_dtype": "float64",
        "output_dtype": "float32",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status": "incomplete",
        "roi_check": {"status": "not_run"},
    }
    exit_code = 1
    try:
        if len(subjects) != len(ids):
            raise ValueError("Incomplete cohort; no channel cache will be written")
        times, labels, pos, srate = template
        means = {c: np.stack([subjects[s][0][c] for s in ids]) for c in CONDITIONS}
        counts = np.stack([subjects[s][1] for s in ids])
        manifest.update(
            array_shapes={c: list(v.shape) for c, v in means.items()},
            time_range_ms=[float(times[0]), float(times[-1])],
            sampling_rate_hz=srate,
            n_channels=len(labels),
            n_times=len(times),
        )
        reference = args.roi_reference
        if reference is None:
            candidate = out / "roi_waves_ecgreg.mat"
            reference = candidate if candidate.is_file() else None
        if reference is not None:
            manifest["roi_check"] = check_roi(
                reference.resolve(), args.config.resolve(), ids, times, means, counts
            )
            if manifest["roi_check"]["status"] != "passed":
                raise ValueError("ROI comparison failed; see manifest")
        else:
            manifest["roi_check"] = {"status": "unavailable"}
        cache = out / "chan_means.npz"
        pending = out / "chan_means.pending.npz"
        np.savez_compressed(
            pending,
            times=times,
            labels=labels,
            pos=pos,
            ids=np.array(ids),
            n_trials=counts,
            **means,
        )
        with np.load(pending, allow_pickle=False) as z:
            for key, expected in dict(
                times=times, labels=labels, pos=pos, ids=np.array(ids), n_trials=counts, **means
            ).items():
                np.testing.assert_array_equal(z[key], expected)
        pending.replace(cache)
        manifest.update(
            status="complete",
            reload_verified=True,
            cache_bytes=cache.stat().st_size,
            cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        )
        exit_code = 0
    except Exception as exc:
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        print(manifest["error"], file=sys.stderr)
    (out / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in manifest.items() if k != "source_files"}, indent=2), flush=True
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
