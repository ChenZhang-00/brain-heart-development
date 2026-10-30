"""Reproduce the 148-participant group-mean EEG–IBI analysis.

Run from the repository root:
    python 03_correlation/analysis/analyze_eeg_ibi_correlations.py --check-only
    python 03_correlation/analysis/analyze_eeg_ibi_correlations.py

The reported index pairing, separate finite means and two BH-FDR families are
preserved. This module does no work when imported.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from importlib import import_module
_correlation_config = import_module("03_correlation.config.correlation_config")
ALPHA = _correlation_config.ALPHA
BAND_LIMITS = _correlation_config.BAND_LIMITS
BANDS = _correlation_config.BANDS
CONDITIONS = _correlation_config.CONDITIONS
EEG_DROP = _correlation_config.EEG_DROP
EXPECTED_SUBJECTS = _correlation_config.EXPECTED_SUBJECTS
FS_IBI = _correlation_config.FS_IBI
GROUPS = _correlation_config.GROUPS
IBI_SAMPLES_PER_BIN = _correlation_config.IBI_SAMPLES_PER_BIN
MIN_PAIRED_SAMPLES = _correlation_config.MIN_PAIRED_SAMPLES
PRIMARY_REGION = _correlation_config.PRIMARY_REGION
REGIONS = _correlation_config.REGIONS
T = _correlation_config.T
TIME = _correlation_config.TIME
project_root = _correlation_config.project_root
relative_path = import_module("common.paths").relative_path
resolve_path = _correlation_config.resolve_path


def require(condition, message):
    """Keep input validation active even when Python is run with -O."""
    if not condition:
        raise ValueError(message)


def finite_mean(values, axis):
    values = np.asarray(values, dtype=float)
    count = np.isfinite(values).sum(axis=axis)
    total = np.where(np.isfinite(values), values, 0.0).sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_name(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def adjust_fdr(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    result = np.full(values.shape, np.nan)
    if valid.any():
        result[valid] = multipletests(values[valid], method="fdr_bh")[1]
    return result


def run_analysis(
    *, root=None, participants=None, ecg_file=None, eeg_dir=None, output_root=None,
    check_only=False, print_tables=True,
):
    """Return checked inputs or compute the correlation table and saved outputs."""
    ROOT = project_root(root)
    INFO_FILE = resolve_path(participants or "00_data/participants.csv", ROOT)
    ECG_FILE = resolve_path(ecg_file or "01_preprocessing/outputs/manuscript/ecg/hrv/ecg_ts.npz", ROOT)
    EEG_DIR = resolve_path(eeg_dir or "01_preprocessing/outputs/manuscript/eeg/band_power", ROOT)
    OUTPUT_ROOT = resolve_path(
        output_root or os.environ.get("EEG_IBI_OUTPUT_ROOT") or "03_correlation", ROOT
    )
    RESULT_DIR = OUTPUT_ROOT / "cache"
    missing = [str(path) for path in (INFO_FILE, ECG_FILE) if not path.is_file()]
    if not EEG_DIR.is_dir():
        missing.append(str(EEG_DIR) + "/")
    if missing:
        raise FileNotFoundError(
            "Missing analysis inputs:\n"
            + "\n".join(missing)
            + "\nComplete preprocessing for all 148 participants first. "
            "The included sub-001 example alone cannot reproduce this cohort analysis."
        )
    if print_tables:
        print("Project:", ROOT)
        print("Participants:", relative_path(INFO_FILE, ROOT))
        print("IBI:", relative_path(ECG_FILE, ROOT))
        print("EEG:", relative_path(EEG_DIR, ROOT))
        print("Results:", relative_path(RESULT_DIR, ROOT))

    info = pd.read_csv(INFO_FILE, dtype={"ID": str})
    require(
        {"ID", "Age_month"} <= set(info.columns), "Participant CSV needs ID and Age_month columns."
    )
    require(
        info.ID.notna().all() and info.ID.str.strip().ne("").all(),
        "Blank participant IDs in metadata.",
    )
    require(info.ID.is_unique, "Duplicate participant IDs in metadata.")
    info["Age_month"] = pd.to_numeric(info["Age_month"], errors="raise")
    info["age_years"] = info["Age_month"] / 12.0
    info["age_group"] = pd.cut(
        info["Age_month"], bins=[60, 84, 108, 132], right=False, labels=GROUPS
    ).astype("string")

    with np.load(ECG_FILE, allow_pickle=False) as source:
        require(
            {"ids", "conditions", "ibi", "time_4hz"} <= set(source.files),
            "ECG cache needs ids, conditions, ibi and time_4hz arrays.",
        )
        ids = source["ids"].astype(str)
        cache_conditions = source["conditions"].astype(str).tolist()
        ibi_native = source["ibi"].astype(float)
        native_time = source["time_4hz"].astype(float)
    require(ids.ndim == 1, "ECG participant IDs must be a one-dimensional array.")
    require(len(set(ids)) == len(ids), "Duplicate participant IDs in ECG cache.")
    require(
        len(ids) == EXPECTED_SUBJECTS,
        f"Expected {EXPECTED_SUBJECTS} participants; ECG cache contains {len(ids)}. "
        "Generate the full-cohort ecg_ts.npz; single-subject outputs are insufficient.",
    )
    require(
        len(cache_conditions) == len(CONDITIONS) and set(cache_conditions) == set(CONDITIONS),
        "ECG conditions must contain neutral, fear and happy exactly once.",
    )
    require(
        native_time.shape == (T * IBI_SAMPLES_PER_BIN,)
        and np.allclose(native_time, np.arange(T * IBI_SAMPLES_PER_BIN) / FS_IBI),
        "ECG time_4hz must contain 704 samples from 0 to 175.75 s at 4 Hz.",
    )
    require(
        ibi_native.shape == (len(ids), len(CONDITIONS), T * IBI_SAMPLES_PER_BIN),
        f"Incorrect IBI shape: {ibi_native.shape}.",
    )

    files = {
        p.name.removesuffix("_movie_EEG_TS.mat"): p for p in EEG_DIR.glob("*_movie_EEG_TS.mat")
    }
    require(
        set(ids) == set(info.ID),
        "ECG and metadata participant sets differ; use matching recoded IDs.",
    )
    missing_eeg = sorted(set(ids) - set(files))
    require(
        not missing_eeg,
        f"Missing EEG files for {len(missing_eeg)} participants: {', '.join(missing_eeg[:10])}",
    )
    info = info.set_index("ID").loc[ids].reset_index()
    require(
        info["age_group"].notna().all(),
        "Age_month must be in [60, 132) for the three study age groups.",
    )
    require(info["age_years"].notna().all(), "Missing participant ages.")
    require(set(info["age_group"]) == set(GROUPS), "All three age groups must be represented.")
    AGE_CENTER = float(info.age_years.mean())

    condition_order = [cache_conditions.index(condition) for condition in CONDITIONS]
    blocks = ibi_native[:, condition_order].reshape(
        len(ids), len(CONDITIONS), T, IBI_SAMPLES_PER_BIN
    )
    ibi = finite_mean(blocks, axis=3)
    shape = (len(ids), len(CONDITIONS), T)
    require(ibi.shape == shape, f"Incorrect reduced IBI shape: {ibi.shape}.")
    alignment_rows = []
    for subject_index, sid in enumerate(ids):
        for condition_index, condition in enumerate(CONDITIONS):
            counts = np.isfinite(blocks[subject_index, condition_index]).sum(axis=1)
            alignment_rows.append(
                {
                    "ID": sid,
                    "condition": condition,
                    "pairing_method": "legacy_index_pairing",
                    "native_valid_samples": int(counts.sum()),
                    "ibi_valid_bins": int((counts > 0).sum()),
                    "ibi_full_bins": int((counts == IBI_SAMPLES_PER_BIN).sum()),
                    "ibi_partial_bins": int(((counts > 0) & (counts < IBI_SAMPLES_PER_BIN)).sum()),
                    "ibi_empty_bins": int((counts == 0).sum()),
                }
            )
    alignment = pd.DataFrame(alignment_rows)
    if check_only or print_tables:
        print(
            "Participants:",
            len(ids),
            "| Age groups:",
            info.groupby("age_group", observed=True).size().to_dict(),
        )
        print(
            "Valid IBI bins per child and movie:",
            alignment.ibi_valid_bins.min(),
            "to",
            alignment.ibi_valid_bins.max(),
        )

    if check_only:
        print("Input check passed. Full EEG content checks run during analysis.")
        return {
            "participants": info,
            "ibi": ibi,
            "eeg_files": {sid: files[sid] for sid in ids},
        }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    input_rows = [
        {"file": input_name(ECG_FILE, ROOT), "sha256": sha256(ECG_FILE)},
        {"file": input_name(INFO_FILE, ROOT), "sha256": sha256(INFO_FILE)},
    ]

    eeg = {region: {band: np.full(shape, np.nan) for band in BANDS} for region in REGIONS}
    eeg_qc = []

    for subject_index, sid in enumerate(ids):
        path = files[sid]
        input_rows.append({"file": input_name(path, ROOT), "sha256": sha256(path)})
        with h5py.File(path, "r") as source:
            group = source["movie_EEG_TS"]
            labels = [
                "".join(chr(int(x)) for x in source[reference[0]][()].ravel())
                for reference in group["chanlocs"]["labels"][()]
            ]
            require(len(set(labels)) == len(labels), f"Duplicate channel labels: {sid}")
            missing_labels = sorted(
                set(label for wanted in REGIONS.values() for label in wanted) - set(labels)
            )
            require(not missing_labels, f"Missing EEG channel labels for {sid}: {missing_labels}")
            channel_index = {
                region: [labels.index(label) for label in wanted]
                for region, wanted in REGIONS.items()
            }
            frequencies = group["freq"][()].ravel()
            bins = [
                int(((frequencies >= low) & (frequencies <= high)).sum())
                for low, high in BAND_LIMITS
            ]
            require(bins == [7, 7, 11, 37], f"{sid}: incorrect frequency-bin definitions: {bins}")
            for condition_index, condition in enumerate(CONDITIONS):
                stored_time = group["time"][condition][()].ravel()
                require(
                    np.array_equal(stored_time, np.arange(1.0, 178.0)),
                    f"{sid}, {condition}: incorrect EEG time axis; expected 1–177 s.",
                )
                selected_time = np.arange(EEG_DROP, EEG_DROP + T)
                require(
                    np.array_equal(stored_time[selected_time], TIME),
                    f"{sid}: incorrect selected EEG times.",
                )
                for band in BANDS:
                    values = group[condition][band][()].astype(float)
                    require(
                        values.shape == (177, len(labels)),
                        f"{sid}, {condition}, {band}: incorrect power shape: {values.shape}",
                    )
                    log_power = np.full_like(values, np.nan)
                    valid = np.isfinite(values) & (values > 0)
                    np.log(values, out=log_power, where=valid)
                    for region in REGIONS:
                        series = finite_mean(log_power[:, channel_index[region]], axis=1)[
                            selected_time
                        ]
                        eeg[region][band][subject_index, condition_index] = series
                        eeg_qc.append(
                            {
                                "ID": sid,
                                "condition": condition,
                                "region": region,
                                "band": band,
                                "n_valid_eeg": int(np.isfinite(series).sum()),
                            }
                        )
        if print_tables and ((subject_index + 1) % 40 == 0 or subject_index + 1 == len(ids)):
            print(f"Loaded EEG: {subject_index + 1}/{len(ids)} participants")

    print("EEG frequency bins, time axes, channel labels, and participant IDs validated.")

    curve_rows, group_rows = [], []
    for region in REGIONS:
        for condition_index, condition in enumerate(CONDITIONS):
            for age_group in GROUPS:
                members = np.flatnonzero(info.age_group.to_numpy() == age_group)
                for band in BANDS:
                    x = eeg[region][band][members, condition_index]
                    y = ibi[members, condition_index]
                    count_x = np.isfinite(x).sum(axis=0)
                    count_y = np.isfinite(y).sum(axis=0)
                    count_paired = (np.isfinite(x) & np.isfinite(y)).sum(axis=0)
                    mean_x = finite_mean(x, axis=0)
                    mean_y = finite_mean(y, axis=0)
                    valid = np.isfinite(mean_x) & np.isfinite(mean_y)
                    require(
                        valid.sum() >= MIN_PAIRED_SAMPLES,
                        f"{region}, {condition}, {age_group}, {band}: fewer than {MIN_PAIRED_SAMPLES} valid time pairs.",
                    )
                    require(
                        mean_x[valid].std() > 0 and mean_y[valid].std() > 0,
                        f"{region}, {condition}, {age_group}, {band}: constant group-mean signal.",
                    )
                    result = stats.pearsonr(mean_x[valid], mean_y[valid])
                    group_rows.append(
                        {
                            "region": region,
                            "condition": condition,
                            "age_group": age_group,
                            "band": band,
                            "n_subjects": len(members),
                            "n_time": int(valid.sum()),
                            "r": float(result.statistic),
                            "p_pearson": float(result.pvalue),
                        }
                    )
                    for j, time in enumerate(TIME):
                        curve_rows.append(
                            {
                                "region": region,
                                "condition": condition,
                                "age_group": age_group,
                                "band": band,
                                "ibi_bin_index": j,
                                "eeg_time_s": time,
                                "eeg_log_power": mean_x[j],
                                "ibi_ms": mean_y[j],
                                "n_eeg_children": int(count_x[j]),
                                "n_ibi_children": int(count_y[j]),
                                "n_paired_children": int(count_paired[j]),
                                "used_in_test": bool(valid[j]),
                            }
                        )

    group_correlations = pd.DataFrame(group_rows)
    group_correlations["fdr_family"] = np.where(
        group_correlations.region == PRIMARY_REGION,
        "primary_frontal_36",
        "supplementary_regions_72",
    )
    group_correlations["q_fdr"] = group_correlations.groupby("fdr_family")["p_pearson"].transform(
        adjust_fdr
    )
    alignment.to_csv(RESULT_DIR / "ibi_alignment_qc.csv", index=False)
    pd.DataFrame(eeg_qc).to_csv(RESULT_DIR / "eeg_coverage_qc.csv", index=False)
    pd.DataFrame(input_rows).to_csv(RESULT_DIR / "input_manifest.csv", index=False)
    group_correlations.to_csv(RESULT_DIR / "group_mean_correlations.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(RESULT_DIR / "group_mean_timeseries.csv", index=False)
    primary = group_correlations[group_correlations.region == PRIMARY_REGION].copy()
    if print_tables:
        print("Primary frontal group-mean correlations:")
        print(
            primary[["condition", "age_group", "band", "r", "p_pearson", "q_fdr"]]
            .round(4)
            .to_string(index=False)
        )
    print(
        "Primary correlations surviving FDR:", int((primary.q_fdr < ALPHA).sum()), "/", len(primary)
    )

    manifest = {
        "analysis": "Nondirectional EEG-IBI covariation",
        "notebook": "run_correlation.ipynb",
        "script": "analysis/analyze_eeg_ibi_correlations.py",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cardiac_measure": "IBI",
        "conditions": CONDITIONS,
        "bands": BANDS,
        "regions": REGIONS,
        "primary_region": PRIMARY_REGION,
        "n_subjects": len(ids),
        "expected_subjects": EXPECTED_SUBJECTS,
        "age_group_counts": info.groupby("age_group", observed=True).size().to_dict(),
        "age_center_years": AGE_CENTER,
        "time_start_s": float(TIME[0]),
        "time_end_s": float(TIME[-1]),
        "time_reference": "EEG movie-onset labels; IBI is paired by legacy array position",
        "alignment_method": "legacy_index_pairing",
        "eeg_drop_samples": EEG_DROP,
        "ibi_samples_per_block": IBI_SAMPLES_PER_BIN,
        "ibi_minimum_finite_samples_per_block": 1,
        "minimum_paired_time_samples": MIN_PAIRED_SAMPLES,
        "eeg_temporal_interpolation": False,
        "group_mean_method": "separate_finite_means",
        "group_minimum_subject_fraction": None,
        "group_correlation_test": "ordinary_two_sided_pearson",
        "temporal_autocorrelation_adjustment": False,
        "group_correlation_fdr_families": {"primary_frontal": 36, "supplementary_regions": 72},
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
    (RESULT_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("Saved tables and analysis_manifest.json to:", relative_path(RESULT_DIR, ROOT))
    return group_correlations


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--project-root", help="Repository root (default: locate from this script)."
    )
    parser.add_argument("--participants", help="Participant CSV (default: 00_data/participants.csv).")
    parser.add_argument("--ecg-file", help="Full-cohort IBI cache (default: 01_preprocessing/outputs/manuscript/ecg/hrv/ecg_ts.npz).")
    parser.add_argument("--eeg-dir", help="MATLAB EEG power folder (default: 01_preprocessing/outputs/manuscript/eeg/band_power).")
    parser.add_argument("--output-root", help="Working folder for cache/ and logs/ (default: 03_correlation/).")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check cohort metadata, IBI and EEG file availability without writing tables.",
    )
    args = parser.parse_args(argv)
    run_analysis(
        root=args.project_root,
        participants=args.participants,
        ecg_file=args.ecg_file,
        eeg_dir=args.eeg_dir,
        output_root=args.output_root,
        check_only=args.check_only,
    )


if __name__ == "__main__":
    main()
