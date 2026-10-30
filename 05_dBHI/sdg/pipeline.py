"""Validated, resumable per-subject computation used by compute_dbhi.ipynb."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
import uuid
import warnings
from pathlib import Path

import h5py
import numpy as np
from joblib import Parallel, delayed, parallel_config

from .model import PySDG

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
_paths = importlib.import_module("common.paths")
# The guided notebook points these at outputs/demo for the single-subject example.
CONT_DIR = Path(os.environ.get("DBHI_EEG_DIR", PROJECT / "01_preprocessing/outputs/manuscript/eeg/cleaned"))
RPEAK_DIR = Path(os.environ.get("DBHI_RPEAK_DIR", PROJECT / "01_preprocessing/outputs/manuscript/ecg/r_peaks"))
CONDITIONS = ["fear", "neutral", "happy"]
DEFAULT_CONFIG = {
    "flims_eeg": {"delta": [1, 4], "theta": [4, 7], "alpha": [7, 12], "beta": [12, 30]},
    "window": 15,
    "fs_bhi": 4,
    "tfr_eeg_win": 2,
    "bth_alignment": "window_center",
    "rr_gap_warning_s": 2.0,
    "edge_gap_warning_s": 5.0,  # no longer used for flagging; kept so the manifest matches the supplied cache
}


def shown(path):
    return _paths.relative_path(path, PROJECT)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    return {name: file_hash(HERE / name) for name in ("pipeline.py", "model.py", "utils.py")}


_IMPORTED_SOURCES = source_hashes()


def make_manifest(config):
    if source_hashes() != _IMPORTED_SOURCES:
        raise RuntimeError(
            "Source changed after import. Restart the notebook kernel before running."
        )
    if config["bth_alignment"] != "window_center":
        raise ValueError("This model implements window_center alignment only.")
    samples = config["window"] * config["fs_bhi"]
    if samples < 15 or samples != int(samples) or int(samples) % 2:
        raise ValueError("window * fs_bhi must be an even integer >= 16.")
    if config["fs_bhi"] < 2 or config["tfr_eeg_win"] <= 0:
        raise ValueError("Require fs_bhi >= 2 for HF 0.80 Hz and positive EEG window.")
    for bounds in config["flims_eeg"].values():
        if not 0 < bounds[0] < bounds[1]:
            raise ValueError("Invalid EEG frequency band.")
    # Explicit band order is necessary: sorting dictionary keys in JSON must
    # not make two different array-axis orders look like the same computation.
    return {
        "schema": "dbhi_v2_centered_1",
        "config": config,
        "band_order": list(config["flims_eeg"]),
        "conditions": CONDITIONS,
        "hrv_bands_hz": {"LF": [0.04, 0.15], "HF": [0.15, 0.80]},
        "ipfm_centers_hz": {"LF": 0.083, "HF": 0.32},
        "sources_sha256": _IMPORTED_SOURCES,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "numba", "h5py", "joblib")
        },
        "python": platform.python_version(),
        "normalization": "per_clip_minmax_as_in_original_model",
    }


def fingerprint(manifest):
    return hashlib.sha256(canonical(manifest).encode("utf-8")).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def prepare_cache(cache_dir, manifest):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "_manifest.json"
    if path.exists():
        if canonical(json.loads(path.read_text(encoding="utf-8"))) != canonical(manifest):
            raise RuntimeError(
                "Cache fingerprint mismatch. Choose a NEW CACHE_DIR; existing results were not changed."
            )
    else:
        if list(cache_dir.glob("*_dbhi.npz")):
            raise RuntimeError(
                "Cache contains data without a manifest; refusing to adopt these files."
            )
        # This function is called while holding the run lock, before any tasks.
        atomic_json(path, manifest)
    return fingerprint(manifest)


def subject_paths(sid):
    return (CONT_DIR / f"{sid}_continuous_clean.mat", RPEAK_DIR / f"{sid}_movie_rpeaks.npz")


def _strings(f, dataset):
    return ["".join(chr(int(c)) for c in f[ref][()].ravel()) for ref in dataset[()].ravel()]


def read_metadata(cont_path, rpeak_path, config):
    with h5py.File(cont_path, "r") as f:
        e, s = f["EEG"], f["seg_info"]
        shape = e["data"].shape
        fs = float(e["srate"][()].ravel()[0])
        labels = _strings(f, e["chanlocs"]["labels"])
        conditions = _strings(f, s["cond_order"])
        seg = {
            key: np.asarray(s[key][()]).ravel()
            for key in ("onsets_orig", "offsets_orig", "onsets_new", "offsets_new")
        }
    if not np.isfinite(fs) or fs <= 0 or len(shape) != 2 or shape[1] != len(labels):
        raise ValueError("Invalid EEG sample rate, data orientation or channel labels.")
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate channel labels.")
    if sorted(conditions) != sorted(CONDITIONS):
        raise ValueError(f"Expected exactly three distinct conditions, got {conditions}")
    if not np.isclose(fs / config["fs_bhi"], round(fs / config["fs_bhi"])):
        raise ValueError("EEG sample rate must be divisible by fs_bhi.")
    if max(b[1] for b in config["flims_eeg"].values()) >= fs / 2:
        raise ValueError("EEG band exceeds the available Nyquist frequency.")
    for key, values in seg.items():
        if (
            len(values) != 3
            or not np.isfinite(values).all()
            or not np.equal(values, np.round(values)).all()
        ):
            raise ValueError(f"Invalid segment indices: {key}")
        seg[key] = values.astype(np.int64)
    orig_len = seg["offsets_orig"] - seg["onsets_orig"] + 1
    new_len = seg["offsets_new"] - seg["onsets_new"] + 1
    if (
        np.any(orig_len <= 0)
        or not np.array_equal(orig_len, new_len)
        or seg["onsets_new"][0] != 1
        or seg["offsets_new"][-1] != shape[0]
        or not np.array_equal(seg["onsets_new"][1:], seg["offsets_new"][:-1] + 1)
    ):
        raise ValueError("Original/new EEG segment lengths or concatenation boundaries disagree.")
    with np.load(rpeak_path, allow_pickle=False) as z:
        peaks = np.asarray(z["r_peaks"], dtype=float)
        fs_rp = float(z["fs"])
        if not np.isfinite(fs_rp) or fs_rp <= 0:
            raise ValueError("Invalid R-peak sampling rate.")
        if (
            peaks.ndim != 1
            or len(peaks) < 3
            or not np.isfinite(peaks).all()
            or np.any(np.diff(peaks) <= 0)
            or np.any(peaks < 0)
        ):
            raise ValueError("R peaks must be finite, nonnegative, strictly increasing and unique.")
        peaks = peaks * fs / fs_rp
        # Both boundaries are now 0-based, matching the stored R peaks.
        onsets = seg["onsets_orig"] - 1
        offsets = seg["offsets_orig"] - 1
        if np.any(onsets < 0):
            raise ValueError("Invalid original segment onset.")
        if "event_sample" in z and "event_desc" in z:
            for i, cond in enumerate(conditions):
                start_code = {"fear": 1, "neutral": 3, "happy": 5}[cond]
                for code, boundary in ((start_code, onsets[i]), (start_code + 1, offsets[i])):
                    candidates = z["event_sample"][z["event_desc"] == f"S{code}"] * fs / fs_rp
                    if not len(candidates) or np.min(np.abs(candidates - boundary)) > 0.5:
                        raise ValueError(f"{cond}: EEG boundary disagrees with ECG marker S{code}.")
    clips = []
    for i, cond in enumerate(conditions):
        relative = (peaks[(peaks >= onsets[i]) & (peaks <= offsets[i])] - onsets[i]) / fs
        duration = float((orig_len[i] - 1) / fs)
        if len(relative) < 4 or relative[-1] - relative[0] < max(20, config["window"] + 5):
            raise ValueError(f"{cond}: insufficient R-peak support for SDG.")
        gaps = np.diff(relative)
        clips.append(
            {
                "condition": cond,
                "start": int(seg["onsets_new"][i] - 1),
                "stop": int(seg["offsets_new"][i]),
                "r_seconds": relative,
                "duration_s": duration,
                "rpeak_count": len(relative),
                "max_rr_s": float(gaps.max()),
                "rr_over_2s": int((gaps > 2).sum()),
                "rr_gaps": [
                    [float(relative[j]), float(relative[j + 1])]
                    for j in np.where(gaps > config["rr_gap_warning_s"])[0]
                ],
            }
        )
    return fs, labels, conditions, clips


def preflight(subject_ids, config):
    rows = []
    reference_labels = None
    for sid in subject_ids:
        try:
            fs, labels, conditions, clips = read_metadata(*subject_paths(sid), config)
            if reference_labels is None:
                reference_labels = labels
            if labels != reference_labels:
                raise ValueError("Channel labels/order differ from the first subject.")
            for clip in clips:
                row = {k: v for k, v in clip.items() if k != "r_seconds"}
                flagged = clip["max_rr_s"] > config["rr_gap_warning_s"]
                rows.append(
                    {
                        "subject": sid,
                        "status": "warning" if flagged else "ok",
                        "detail": "",
                        "srate": fs,
                        "channels": len(labels),
                        **row,
                    }
                )
        except Exception as ex:
            rows.append(
                {
                    "subject": sid,
                    "condition": "",
                    "status": "failed",
                    "detail": f"{type(ex).__name__}: {ex}",
                }
            )
    return rows


def input_identity(paths):
    result = {}
    for path in paths:
        before = path.stat()
        digest = file_hash(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Input changed during hashing: {path}")
        result[path.name] = {"sha256": digest, "size": after.st_size, "mtime_ns": after.st_mtime_ns}
    return result


def validate_arrays(z, config, expected_labels=None):
    conditions = z["cond_order"].tolist()
    labels = z["labels"].tolist()
    if sorted(conditions) != sorted(CONDITIONS) or not labels:
        raise ValueError("Invalid cache conditions or labels.")
    if expected_labels is not None and labels != expected_labels:
        raise ValueError("Cached channel labels differ from the input.")
    if z["bands"].tolist() != list(config["flims_eeg"]) or z["hrv_bands"].tolist() != ["LF", "HF"]:
        raise ValueError("Cached frequency-band order mismatch.")
    summary = []
    for cond in CONDITIONS:
        for direction in ("bth", "htb"):
            values = z[f"{cond}__{direction}"]
            times = z[f"{cond}__t_{direction}"]
            if (
                times.ndim != 1
                or len(times) < 2
                or not np.isfinite(times).all()
                or times[0] < 0
                or not np.allclose(np.diff(times), 1 / config["fs_bhi"], atol=1e-9, rtol=0)
            ):
                raise ValueError(f"{cond}/{direction}: invalid time axis.")
            expected = (len(config["flims_eeg"]), 2, len(labels), len(times))
            if values.shape != expected or np.isinf(values).any():
                raise ValueError(f"{cond}/{direction}: shape mismatch or infinite values.")
            if not np.isfinite(values).any(axis=-1).all():
                raise ValueError(f"{cond}/{direction}: at least one series has no finite samples.")
            summary.append(
                {
                    "condition": cond,
                    "direction": direction,
                    "n_time": len(times),
                    "start_s": float(times[0]),
                    "end_s": float(times[-1]),
                    "nan_fraction": float(np.isnan(values).mean()),
                }
            )
    return summary


def validate_cache(path, config, run_fingerprint, inputs=None, labels=None):
    with np.load(path, allow_pickle=False) as z:
        if str(z["_fingerprint"].item()) != run_fingerprint:
            raise ValueError("Per-subject fingerprint mismatch.")
        if inputs is not None and str(z["_inputs_json"].item()) != canonical(inputs):
            raise ValueError(
                "Input files changed since this subject was computed; use a new cache directory."
            )
        summary = validate_arrays(z, config, labels)
        qc = json.loads(str(z["_qc_json"].item()))
    return summary, qc


def compute_subject(sid, cache_dir, config, run_fingerprint):
    start = time.monotonic()
    try:
        paths = subject_paths(sid)
        inputs = input_identity(paths)
        fs, labels, conditions, clips = read_metadata(*paths, config)
        destination = Path(cache_dir) / f"{sid}_dbhi.npz"
        if destination.exists():
            summary, qc = validate_cache(destination, config, run_fingerprint, inputs, labels)
            return {
                "subject": sid,
                "status": "cached",
                "seconds": round(time.monotonic() - start, 1),
                "detail": "",
                "arrays": summary,
                "qc": qc,
            }
        output = {
            "labels": np.array(labels),
            "cond_order": np.array(conditions),
            "srate": np.array(fs),
            "bands": np.array(list(config["flims_eeg"])),
            "hrv_bands": np.array(["LF", "HF"]),
            "_fingerprint": np.array(run_fingerprint),
            "_inputs_json": np.array(canonical(inputs)),
        }
        qc = {"clips": [], "warnings": []}
        with warnings.catch_warnings(record=True) as caught, h5py.File(paths[0], "r") as f:
            warnings.simplefilter("always")
            for clip in clips:
                cond = clip["condition"]
                eeg = np.asarray(
                    f["EEG"]["data"][clip["start"] : clip["stop"], :], dtype=np.float64
                ).T
                if not np.isfinite(eeg).all():
                    raise ValueError(f"{cond}: nonfinite EEG values.")
                obj = PySDG(
                    eeg=eeg,
                    eeg_fs=fs,
                    r_peaks=clip["r_seconds"],
                    flims_eeg=config["flims_eeg"],
                    fs_bhi=config["fs_bhi"],
                    window=config["window"],
                    parallelize=False,
                    progressbar=False,
                )
                obj.compute_tfr_eeg(window=config["tfr_eeg_win"])
                obj.compute_tfr_hrv()
                obj.compute_sdg()
                for direction, time_key in (("bth", "time_BToH"), ("htb", "time_HToB")):
                    output[f"{cond}__{direction}"] = obj.bhi[direction]
                    output[f"{cond}__t_{direction}"] = obj.bhi["t"][time_key]
                    if obj.bhi["t"][time_key][-1] > clip["duration_s"]:
                        raise ValueError(f"{cond}/{direction}: output extends beyond clip.")
                qc["clips"].append({k: v for k, v in clip.items() if k != "r_seconds"})
                del obj, eeg
            qc["warnings"] = sorted(set(str(w.message) for w in caught))
        summary = validate_arrays(output, config, labels)
        for path in paths:
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (
                inputs[path.name]["size"],
                inputs[path.name]["mtime_ns"],
            ):
                raise RuntimeError(f"Input changed during computation: {path}")
        output["_qc_json"] = np.array(canonical(qc))
        temp = destination.with_name(destination.stem + "." + uuid.uuid4().hex + ".part.npz")
        try:
            np.savez(temp, **output)
            validate_cache(temp, config, run_fingerprint, inputs, labels)
            os.replace(temp, destination)
        finally:
            temp.unlink(missing_ok=True)
        return {
            "subject": sid,
            "status": "ok",
            "seconds": round(time.monotonic() - start, 1),
            "detail": "",
            "arrays": summary,
            "qc": qc,
        }
    except Exception as ex:
        return {
            "subject": sid,
            "status": "failed",
            "seconds": round(time.monotonic() - start, 1),
            "detail": f"{type(ex).__name__}: {ex}",
            "traceback": traceback.format_exc(),
        }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: canonical(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def run_all(subject_ids, cache_dir, report_dir, config, n_jobs=4):
    if not subject_ids or len(set(subject_ids)) != len(subject_ids):
        raise ValueError("Provide a nonempty list of unique subject IDs.")
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer.")
    config = json.loads(json.dumps(config))
    cache_dir, report_dir = Path(cache_dir).resolve(), Path(report_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    lock = cache_dir / "_run.lock"
    # Exclusive creation prevents simultaneous notebook/CLI runs in this cache.
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise RuntimeError(
            f"Another run or interrupted run owns {lock}. Check its PID before removing the lock."
        ) from None
    try:
        with handle:
            json.dump({"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}, handle)
        manifest = make_manifest(config)
        run_fingerprint = prepare_cache(cache_dir, manifest)
        atomic_json(
            report_dir / "run_status.json",
            {"state": "preflight", "expected": len(subject_ids), "completed": 0},
        )
        check = preflight(subject_ids, config)
        write_csv(report_dir / "input_qc.csv", check)
        failed = [r for r in check if r["status"] == "failed"]
        if failed:
            raise RuntimeError(
                f"Input preflight failed for {len(failed)} subjects; see input_qc.csv."
            )
        flagged = sum(r["status"] == "warning" for r in check)
        if flagged:
            print(f"{flagged} clip(s) flagged in input_qc.csv (RR gap above 2 s).", flush=True)
        results = []

        def collect(iterator):
            for result in iterator:
                results.append(result)
                atomic_json(
                    report_dir / "run_status.json",
                    {
                        "state": "running",
                        "expected": len(subject_ids),
                        "completed": len(results),
                        "failed": sum(r["status"] == "failed" for r in results),
                        "last_subject": result["subject"],
                        "last_status": result["status"],
                    },
                )
                print(
                    f'[{len(results)}/{len(subject_ids)}] {result["subject"]}: '
                    f'{result["status"]} {result["seconds"]:.1f}s {result["detail"]}',
                    flush=True,
                )

        if n_jobs == 1:
            collect(compute_subject(sid, cache_dir, config, run_fingerprint) for sid in subject_ids)
        else:
            # Limit BLAS threading in each worker, and report completions as
            # they occur instead of waiting for every subject to finish.
            with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
                collect(
                    Parallel(return_as="generator_unordered")(
                        delayed(compute_subject)(sid, cache_dir, config, run_fingerprint)
                        for sid in subject_ids
                    )
                )
        results.sort(key=lambda r: r["subject"])
        # Reopen every expected cache, including cached subjects, before claiming
        # completion. Existing unrelated subjects cannot substitute for missing IDs.
        audit = verify_all(subject_ids, cache_dir, config)
        bad = [r for r in results if r["status"] == "failed"]
        invalid = [r for r in audit if r["status"] != "ok"]
        state = "failed" if bad or invalid else "complete"
        atomic_json(
            report_dir / "run_status.json",
            {
                "state": state,
                "expected": len(subject_ids),
                "completed": len(results),
                "estimated": sum(r["status"] == "ok" for r in results),
                "resumed": sum(r["status"] == "cached" for r in results),
                "failed": len(bad),
                "failed_subjects": {r["subject"]: r["detail"] for r in bad},
                "invalid_cache": len(invalid),
                "invalid_subjects": {r["subject"]: r["detail"] for r in invalid},
                "fingerprint": run_fingerprint,
                "flagged_input_clips": sum(r["status"] == "warning" for r in check),
                "seconds_per_subject": {r["subject"]: round(r["seconds"], 1) for r in results},
            },
        )
        if bad or invalid:
            raise RuntimeError("Run incomplete; see run_status.json. Valid subjects can be resumed.")
        print(f"Complete: {len(subject_ids)} valid model files "
              f"({sum(r['status'] == 'ok' for r in results)} estimated now, "
              f"{sum(r['status'] == 'cached' for r in results)} resumed, in {shown(cache_dir)}).", flush=True)
        return results
    except BaseException as ex:
        # A failed/preflight/interrupted run must never appear complete.
        status_path = report_dir / "run_status.json"
        state = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        state.update(state="failed", error=f"{type(ex).__name__}: {ex}")
        atomic_json(status_path, state)
        raise
    finally:
        lock.unlink(missing_ok=True)


def verify_all(subject_ids, cache_dir, config, *, check_environment=True):
    """Check stored arrays; model resumption also requires the current environment."""
    saved = json.loads((Path(cache_dir) / "_manifest.json").read_text(encoding="utf-8"))
    manifest = make_manifest(config) if check_environment else saved
    if canonical(saved) != canonical(manifest) or canonical(saved["config"]) != canonical(config):
        raise RuntimeError("Manifest mismatch during final verification.")
    run_fingerprint = fingerprint(manifest)
    rows = []
    reference_labels = None
    for sid in subject_ids:
        try:
            path = Path(cache_dir) / f"{sid}_dbhi.npz"
            summary, qc = validate_cache(path, config, run_fingerprint)
            with np.load(path, allow_pickle=False) as z:
                labels = z["labels"].tolist()
            if reference_labels is None:
                reference_labels = labels
            if labels != reference_labels:
                raise ValueError("Channel order differs between caches.")
            rows.append(
                {
                    "subject": sid,
                    "status": "ok",
                    "detail": "",
                    "max_nan_fraction": max(r["nan_fraction"] for r in summary),
                }
            )
        except Exception as ex:
            rows.append(
                {"subject": sid, "status": "failed", "detail": f"{type(ex).__name__}: {ex}"}
            )
    return rows
