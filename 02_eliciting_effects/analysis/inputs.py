"""Shared input checks and EEG extraction for the eliciting-effects analyses."""

from datetime import datetime, timezone
import hashlib
import warnings
import json
import platform

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy
import statsmodels

from importlib import import_module
_eliciting_config = import_module("02_eliciting_effects.config.eliciting_config")
BAND_LIMITS = _eliciting_config.BAND_LIMITS
BANDS = _eliciting_config.BANDS
CONDITIONS = _eliciting_config.CONDITIONS
EMOTIONS = _eliciting_config.EMOTIONS
EXPECTED_SUBJECTS = _eliciting_config.EXPECTED_SUBJECTS
FS_IBI = _eliciting_config.FS_IBI
REGIONS = _eliciting_config.REGIONS
T_EEG = _eliciting_config.T_EEG


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_cohort(paths):
    """Keep CSV row order; return cardiac arrays in the fixed movie order."""
    missing = [str(p) for p in [paths.participants, paths.ecg] if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing cohort inputs:\n"
            + "\n".join(missing)
            + "\nComplete preprocessing for all 148 participants first. "
            "The included sub-001 example alone is insufficient."
        )
    info = pd.read_csv(paths.participants, encoding="utf-8-sig", dtype={"ID": str})
    sam_columns = [f"{c}_{suffix}" for suffix in ["A", "V"] for c in CONDITIONS]
    require(
        {"ID", *sam_columns} <= set(info.columns),
        "Participant CSV needs ID and fear_A/V, happy_A/V, neutral_A/V columns.",
    )
    require(info.ID.notna().all() and info.ID.str.strip().ne("").all(), "Blank participant IDs.")
    require(info.ID.is_unique, "Duplicate participant IDs in metadata.")
    require(
        len(info) == EXPECTED_SUBJECTS,
        f"Expected {EXPECTED_SUBJECTS} participant rows; found {len(info)}.",
    )
    for column in sam_columns:
        info[column] = pd.to_numeric(info[column], errors="raise")
    require(np.isfinite(info[sam_columns].to_numpy(float)).all(), "SAM ratings must be finite.")

    with np.load(paths.ecg, allow_pickle=False) as source:
        required = {"ids", "conditions", "ibi", "hf", "lf", "time_4hz"}
        require(
            required <= set(source.files),
            f"Cardiac cache is missing arrays: {sorted(required - set(source.files))}",
        )
        cache = {key: source[key] for key in required}
        if "time_1hz" in source.files:
            cache["time_1hz"] = source["time_1hz"]
    ids = cache["ids"].astype(str)
    require(ids.ndim == 1, "Cardiac participant IDs must be one-dimensional.")
    require(
        len(ids) == EXPECTED_SUBJECTS and len(set(ids)) == len(ids),
        "Expected 148 unique participants in the batch cardiac cache; single-subject caches are insufficient.",
    )
    require(set(ids) == set(info.ID), "Metadata and cardiac IDs differ; use matching recoded IDs.")
    require(cache["conditions"].ndim == 1, "Cardiac conditions must be one-dimensional.")
    conditions = cache["conditions"].astype(str).tolist()
    require(
        len(conditions) == 3 and set(conditions) == set(CONDITIONS),
        "Cardiac conditions must contain fear, happy and neutral exactly once.",
    )
    for name, samples in [("ibi", T_EEG * FS_IBI), ("lf", T_EEG), ("hf", T_EEG)]:
        require(
            cache[name].shape == (EXPECTED_SUBJECTS, 3, samples),
            f"Incorrect {name} shape: {cache[name].shape}; expected (148, 3, {samples}).",
        )
    require(
        cache["time_4hz"].shape == (T_EEG * FS_IBI,)
        and np.allclose(cache["time_4hz"], np.arange(T_EEG * FS_IBI) / FS_IBI),
        "Expected 704 IBI time labels from 0 to 175.75 s at 4 Hz.",
    )
    if "time_1hz" in cache:
        require(
            cache["time_1hz"].shape == (T_EEG,)
            and np.allclose(cache["time_1hz"], np.arange(T_EEG)),
            "Expected 176 HRV time labels from 0 to 175 s at 1 Hz.",
        )
    order = [conditions.index(c) for c in CONDITIONS]
    # Avoid copying arrays when already ordered; retain their source dtypes.
    if order != list(range(len(CONDITIONS))):
        for name in ["ibi", "lf", "hf"]:
            cache[name] = cache[name][:, order]
    cache["ids"] = ids
    cache["conditions"] = np.array(CONDITIONS)
    return info, cache


def get_eeg_files(paths, ids):
    if not paths.eeg_dir.is_dir():
        raise FileNotFoundError(
            f"Missing EEG power folder: {paths.eeg_dir}. Complete EEG preprocessing first."
        )
    files = {
        p.name.removesuffix("_movie_EEG_TS.mat"): p
        for p in paths.eeg_dir.glob("*_movie_EEG_TS.mat")
        if p.is_file()
    }
    missing = sorted(set(ids) - set(files))
    require(
        not missing, f"Missing EEG files for {len(missing)} participants: {', '.join(missing[:10])}"
    )
    # Preserve the original sorted-filename order for EEG means and permutation draws.
    return sorted(files[sid] for sid in ids)


def validate_eeg_file(handle, path, regions):
    group = handle["movie_EEG_TS"]
    labels = [
        "".join(chr(int(c)) for c in handle[ref[0]][()].ravel())
        for ref in group["chanlocs"]["labels"][()]
    ]
    require(len(labels) == len(set(labels)), f"{path.name}: duplicate EEG channel labels.")
    missing = sorted(
        {channel for channels in regions.values() for channel in channels} - set(labels)
    )
    require(not missing, f"{path.name}: missing EEG channels: {missing}")
    freq = group["freq"][()].ravel()
    bins = [int(((freq >= low) & (freq <= high)).sum()) for low, high in BAND_LIMITS]
    require(bins == [7, 7, 11, 37], f"{path.name}: incorrect frequency bins: {bins}")
    for condition in CONDITIONS:
        times = group["time"][condition][()].ravel()
        require(
            np.array_equal(times, np.arange(1.0, 178.0)),
            f"{path.name}, {condition}: expected 177 stored EEG time labels from 1 to 177 s.",
        )
        for band in BANDS:
            require(
                group[condition][band].shape == (177, len(labels)),
                f"{path.name}, {condition}, {band}: incorrect EEG power shape.",
            )
    indices = {region: [labels.index(c) for c in channels] for region, channels in regions.items()}
    return group, indices


def check_eeg_inputs(files, regions):
    for path in files:
        with h5py.File(path, "r") as handle:
            validate_eeg_file(handle, path, regions)
    print(f"Validated {len(files)} EEG files: channel labels, frequencies, times and band shapes.")


def band_series(group, condition, band, indices):
    """Original log-channel mean, interpolate nonfinite bins, retain first 176 samples."""
    x = np.log(group[condition][band][()][:, indices])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        series = np.nanmean(x, axis=1)
    time = np.arange(series.size)
    bad = ~np.isfinite(series)
    if bad.any() and (~bad).sum() >= 2:
        series[bad] = np.interp(time[bad], time[~bad], series[~bad])
    series = series[:T_EEG]
    return (
        np.pad(series, (0, T_EEG - series.size), constant_values=np.nan)
        if series.size < T_EEG
        else series
    )


def load_eeg_differences(files, regions, *, verbose=False):
    differences = {r: {b: {e: [] for e in EMOTIONS} for b in BANDS} for r in regions}
    for file_index, path in enumerate(files):
        with h5py.File(path, "r") as handle:
            group, indices = validate_eeg_file(handle, path, regions)
            for region in regions:
                for band in BANDS:
                    neutral_mean = np.nanmean(band_series(group, "neutral", band, indices[region]))
                    for emotion in EMOTIONS:
                        series = band_series(group, emotion, band, indices[region])
                        differences[region][band][emotion].append(series - neutral_mean)
        if verbose and ((file_index + 1) % 40 == 0 or file_index + 1 == len(files)):
            print(f"Loaded EEG: {file_index + 1}/{len(files)} participants", flush=True)
    return differences


def load_intensity(paths):
    if not paths.ratings.is_file():
        raise FileNotFoundError(
            f"Missing movie-intensity ratings: {paths.ratings}. "
            "Supply Movies_rating_by_RA.csv with Time, Fear_Intensity and Happy_Intensity."
        )
    table = pd.read_csv(paths.ratings)
    require(
        {"Time", "Fear_Intensity", "Happy_Intensity"} <= set(table.columns),
        "Intensity CSV needs Time, Fear_Intensity and Happy_Intensity columns.",
    )
    require(len(table) >= T_EEG, f"Intensity CSV needs at least {T_EEG} rows.")
    require(
        np.array_equal(table["Time"].to_numpy()[:T_EEG], np.arange(T_EEG)),
        "The first 176 intensity rows must have Time = 0, 1, ..., 175 s in order.",
    )
    result = {}
    for emotion in EMOTIONS:
        values = np.asarray(table[f"{emotion.capitalize()}_Intensity"], float)[:T_EEG]
        require(
            np.isfinite(values).all() and np.ptp(values) > 0,
            f"{emotion}: selected intensity ratings must be finite and nonconstant.",
        )
        result[emotion] = (values - values.min()) / (np.ptp(values) + 1e-12)
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_run(paths, stage, inputs, parameters):
    records = []
    for path in inputs:
        try:
            name = str(path.relative_to(paths.root))
        except ValueError:
            name = str(path)
        records.append({"file": name, "sha256": sha256(path)})
    manifest = {
        "stage": stage,
        "notebook": "run_eliciting_effects.ipynb",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_subjects": EXPECTED_SUBJECTS,
        "conditions": CONDITIONS,
        "parameters": parameters,
        "inputs": records,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "h5py": h5py.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / f"{stage}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
