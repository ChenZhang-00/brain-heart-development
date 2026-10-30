"""Validated HEP inputs, cohort matching and run provenance."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import platform

import numpy as np
import pandas as pd
from scipy.io import loadmat

from importlib import import_module
_hep_config = import_module("04_HEP.config.hep_config")
ALPHA = _hep_config.ALPHA
CONDITIONS = _hep_config.CONDITIONS
EXPECTED_CHANNELS = _hep_config.EXPECTED_CHANNELS
EXPECTED_SUBJECTS = _hep_config.EXPECTED_SUBJECTS
GROUPS = _hep_config.GROUPS
PAIR_COLORS = _hep_config.PAIR_COLORS
REGIONS = _hep_config.REGIONS
ROI = _hep_config.ROI
TIME_STEP_MS = _hep_config.TIME_STEP_MS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(paths):
    if not paths.participants.is_file():
        raise FileNotFoundError(f"Missing participant table: {paths.participants}")
    table = pd.read_csv(paths.participants, encoding="utf-8-sig", dtype={"ID": str})
    require(
        {"ID", "Age_month", "Age_group"} <= set(table.columns),
        "Participant CSV needs ID, Age_month and Age_group columns.",
    )
    require(table.ID.notna().all() and table.ID.str.strip().ne("").all(), "Blank participant IDs.")
    require(
        table.ID.is_unique and len(table) == EXPECTED_SUBJECTS,
        "Expected 148 unique participants; an example-only cohort is insufficient.",
    )
    for name in ["Age_month", "Age_group"]:
        table[name] = pd.to_numeric(table[name], errors="raise")
    require(
        np.isfinite(table[["Age_month", "Age_group"]].to_numpy()).all(),
        "Nonfinite participant ages.",
    )
    require(
        table.Age_month.ge(60).all() and table.Age_month.lt(132).all(),
        "Age_month must be in [60, 132).",
    )
    require(
        table.Age_group.isin(range(5, 11)).all(), "Age_group must contain completed years 5–10."
    )
    require(
        np.array_equal(np.floor(table.Age_month.to_numpy() / 12), table.Age_group.to_numpy()),
        "Age_group must agree with completed years derived from Age_month.",
    )
    return table


def age_groups(table, ids):
    require(
        len(ids) == EXPECTED_SUBJECTS and len(set(ids)) == len(ids),
        "Expected 148 unique IDs in the HEP data.",
    )
    require(
        set(ids) == set(table.ID),
        "HEP and participant-table IDs differ; use consistent recoded IDs.",
    )
    ages = table.set_index("ID").loc[ids, "Age_group"].to_numpy()
    groups = np.where(ages <= 6, "5-6", np.where(ages <= 8, "7-8", "9-10"))
    require(set(groups) == set(GROUPS), "All three study age groups must be represented.")
    return groups


def load_roi(paths):
    table = load_metadata(paths)
    if not paths.roi.is_file():
        raise FileNotFoundError(
            f"Missing ROI waveforms: {paths.roi}. Run preparation/compute_hep_roi_waveforms.m first."
        )
    data = loadmat(paths.roi, simplify_cells=True)
    required = {"times", "ids", "n_trials"} | {f"{r}__{c}" for r in REGIONS for c in CONDITIONS}
    require(required <= set(data), f"ROI MAT file is missing arrays: {sorted(required-set(data))}")
    times = np.asarray(data["times"], dtype=float)
    ids = np.asarray(data["ids"]).astype(str)
    require(ids.ndim == 1 and times.ndim == 1, "ROI IDs and times must be one-dimensional.")
    groups = age_groups(table, ids)
    require(
        np.array_equal(times, np.arange(-200, 600, TIME_STEP_MS)),
        "Expected 400 ECG-regressed HEP samples from -200 to 598 ms at 500 Hz.",
    )
    counts = np.asarray(data["n_trials"])
    require(
        counts.shape == (EXPECTED_SUBJECTS, len(CONDITIONS))
        and np.isfinite(counts).all()
        and (counts > 0).all()
        and np.equal(counts, np.floor(counts)).all(),
        "ROI n_trials must contain positive retained-trial counts for all 148 children and three movies.",
    )
    waves = {(r, c): np.asarray(data[f"{r}__{c}"]) for r in REGIONS for c in CONDITIONS}
    require(
        all(x.shape == (len(ids), len(times)) and np.isfinite(x).all() for x in waves.values()),
        "ROI waveforms must be finite arrays of shape (148, 400).",
    )
    return data, times, ids, groups, waves


def load_clusters(paths):
    if not paths.clusters.is_file():
        raise FileNotFoundError(
            f"Missing {paths.clusters}. Run the waveform cluster analysis first."
        )
    table = pd.read_csv(paths.clusters)
    required = {
        "group",
        "region",
        "test",
        "pair",
        "window",
        "t_start",
        "t_end",
        "stat",
        "p",
        "dz",
        "survives",
    }
    require(required <= set(table.columns), "Cluster table is missing required columns.")
    require(
        table.group.isin(GROUPS).all()
        and table.region.isin(REGIONS).all()
        and table.pair.isin(PAIR_COLORS).all()
        and table.test.eq("pairwise").all(),
        "Unexpected group, region, contrast or test in cluster table.",
    )
    require(
        table.p.between(0, 1).all()
        and np.isfinite(table[["t_start", "t_end", "stat", "dz"]].to_numpy()).all(),
        "Invalid cluster statistics.",
    )
    require(
        table.survives.dtype == bool and np.array_equal(table.survives, table.p < ALPHA),
        "Cluster survives flags must agree with p < 0.05.",
    )
    return table


def load_channels(paths):
    reference, ref_times, ref_ids, _, _ = load_roi(paths)
    cache_path = paths.channel_dir / "chan_means.npz"
    manifest_path = paths.channel_dir / "export_manifest.json"
    missing = [str(p) for p in [cache_path, manifest_path] if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing validated channel cache:\n" + "\n".join(missing))
    export = json.loads(manifest_path.read_text())
    require(
        export.get("status") == "complete" and export.get("data_version") == "ECG-regressed",
        "Channel export must be complete and labelled ECG-regressed.",
    )
    require(
        sha256(cache_path) == export.get("cache_sha256"),
        "Channel-cache SHA-256 differs from its export manifest.",
    )
    with np.load(cache_path, allow_pickle=False) as cache:
        required = {"times", "pos", "ids", "labels", "n_trials", *CONDITIONS}
        require(required <= set(cache.files), "Channel cache is missing required arrays.")
        times, pos = cache["times"], cache["pos"]
        ids, labels = cache["ids"].astype(str), cache["labels"].astype(str)
        signals = {c: cache[c] for c in CONDITIONS}
        counts = cache["n_trials"]
    require(
        ids.ndim == 1
        and len(labels) == EXPECTED_CHANNELS
        and len(set(labels)) == EXPECTED_CHANNELS,
        "Expected unique participant IDs and 63 unique channel labels.",
    )
    require(
        pos.shape == (63, 2) and np.isfinite(pos).all(), "Expected 63 finite 2-D scalp positions."
    )
    require(
        all(x.shape == (148, len(times), 63) and np.isfinite(x).all() for x in signals.values()),
        "Channel means must have shape (148, time, 63) and contain finite values.",
    )
    groups = age_groups(load_metadata(paths), ids)
    roi_masks = {r: np.isin(labels, channels) for r, channels in ROI.items()}
    require(
        np.all(np.stack(list(roi_masks.values())).sum(axis=0) == 1)
        and [int(roi_masks[r].sum()) for r in REGIONS] == [18, 26, 19],
        "Channel labels do not reproduce the 18/26/19-channel ROI partition.",
    )
    order = pd.Index(ref_ids).get_indexer(ids)
    require(
        np.all(order >= 0) and np.array_equal(times, ref_times),
        "Channel and ROI IDs or times differ.",
    )
    np.testing.assert_array_equal(counts, reference["n_trials"][order])
    for region, mask in roi_masks.items():
        for condition in CONDITIONS:
            np.testing.assert_allclose(
                signals[condition][:, :, mask].mean(axis=2, dtype=float),
                reference[f"{region}__{condition}"][order],
                atol=1e-6,
                rtol=1e-6,
            )
    return times, pos, ids, labels, signals, groups, roi_masks, export


def check_epoch_files(paths, ids):
    if not paths.epochs.is_dir():
        raise FileNotFoundError(
            f"Missing ECG-regressed epoch directory: {paths.epochs}. "
            "Complete preprocessing for all 148 children, or supply validated ROI and channel caches."
        )
    missing = [sid for sid in ids if not (paths.epochs / f"{sid}_hep_epochs.mat").is_file()]
    require(
        not missing,
        f"Missing ECG-regressed epochs for {len(missing)} participants. "
        "Complete preprocessing for all 148 children, or supply existing validated ROI and channel caches.",
    )
    return [paths.epochs / f"{sid}_hep_epochs.mat" for sid in ids]


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
        "notebook": "run_hep.ipynb",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_subjects": EXPECTED_SUBJECTS,
        "data_version": "ECG-regressed",
        "parameters": parameters,
        "inputs": records,
        "software": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ["numpy", "pandas", "scipy", "statsmodels", "h5py", "matplotlib", "mne"]
            },
        },
    }
    paths.results.mkdir(parents=True, exist_ok=True)
    (paths.results / f"{stage}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
