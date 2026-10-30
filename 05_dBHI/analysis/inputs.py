"""Cohort and cache checks shared by all analyses; no computation on import."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.dbhi_config import BANDS, FS, GROUPS, shown


def read_participants(path, *, full_cohort=True):
    info = pd.read_csv(path, dtype={"ID": str})
    required = {"ID", "Age_group", "Age_month"}
    if not required.issubset(info):
        raise ValueError(f"Participant table needs {sorted(required)}: {path}")
    ids = info.ID
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("Participant IDs must be nonblank and unique.")
    if not ids.eq(ids.str.strip()).all() or ids.str.contains(r"[/\\]").any():
        raise ValueError("IDs must not contain path separators or surrounding whitespace.")
    for column in ("Age_group", "Age_month"):
        info[column] = pd.to_numeric(info[column], errors="raise")
        if not np.isfinite(info[column]).all():
            raise ValueError(f"Nonfinite {column} in {path}")
    if not info.Age_group.isin([a for ages in GROUPS.values() for a in ages]).all():
        raise ValueError("Age_group must contain integer ages 5 through 10.")
    if not info.Age_month.between(60, 132, inclusive="left").all():
        raise ValueError("Expected ages from 60 to less than 132 months.")
    if full_cohort:
        counts = [int(info.Age_group.isin(ages).sum()) for ages in GROUPS.values()]
        if len(info) != 148 or counts != [50, 49, 49]:
            raise ValueError(f"Expected 148 children with group sizes 50/49/49; got {counts}.")
    return info.set_index("ID")


def validate_model_cache(paths, *, temporal=False):
    from sdg.pipeline import (
        DEFAULT_CONFIG,
        canonical,
        file_hash,
        fingerprint,
        make_manifest,
        validate_arrays,
        verify_all,
    )

    info = read_participants(paths.participants)
    ids = sorted(info.index)
    cache_ids = sorted(p.name.removesuffix("_dbhi.npz") for p in paths.cache.glob("*_dbhi.npz"))
    if cache_ids != ids:
        raise ValueError("Expected matching metadata and model caches for all 148 children.")
    manifest = json.loads((paths.cache / "_manifest.json").read_text())
    expected = make_manifest(DEFAULT_CONFIG)
    settings = expected.keys() - {"sources_sha256", "versions", "python"}
    if canonical({key: manifest.get(key) for key in settings}) != canonical(
        {key: expected[key] for key in settings}
    ):
        raise RuntimeError("Saved model settings differ from this analysis specification.")
    if manifest["band_order"] != BANDS or manifest["config"]["fs_bhi"] != FS:
        raise ValueError("Analysis requires delta/theta/alpha/beta in that order and 4-Hz output.")
    audit = pd.DataFrame(verify_all(ids, paths.cache, manifest["config"], check_environment=False))
    if not audit.status.eq("ok").all():
        raise ValueError(
            "Invalid model caches:\n" + audit.loc[audit.status != "ok"].to_string(index=False)
        )
    with np.load(paths.cache / f"{ids[0]}_dbhi.npz", allow_pickle=False) as first:
        labels = first["labels"].tolist()
    if len(labels) != 63 or len(set(labels)) != 63:
        raise ValueError("Expected 63 unique EEG channels.")
    truncation_info = {}
    if temporal:
        status = json.loads((paths.reports / "run_status.json").read_text())
        if status["state"] != "complete" or status.get("expected") != len(ids):
            raise RuntimeError("Model report does not describe a complete 148-child run.")
        if status.get("fingerprint") != fingerprint(manifest):
            raise RuntimeError("Model report and cache fingerprints differ.")
        truncation_info = neutral_truncation(paths, manifest, labels)
    return manifest, info, ids, truncation_info


def neutral_truncation(paths, manifest, labels):
    """Describe the truncated neutral clip of the documented participant; raise if absent."""
    from sdg.pipeline import file_hash, fingerprint, validate_arrays

    sid, path = paths.neutral_subject, paths.neutral_file
    if not path.is_file():
        raise ValueError(f"{sid}: model array not found: {path}")
    with np.load(path, allow_pickle=False) as z:
        if str(z["_fingerprint"].item()) != fingerprint(manifest):
            raise ValueError(f"{sid}: cache fingerprint mismatch.")
        clip = next(c for c in json.loads(str(z["_qc_json"].item()))["clips"] if c["condition"] == "neutral")
        if "segment_used_s" not in clip:
            raise ValueError(
                f"{sid}: the neutral clip has not been truncated to its R-peak-supported segment. "
                "Run preparation/recompute_neutral_dbhi.py first."
            )
        summary = validate_arrays(z, manifest["config"], labels)
        note = str(z["_note"].item())
    return {sid: dict(path=shown(path), sha256=file_hash(path), segment_s=clip["segment_used_s"],
                      rpeaks_used=clip["rpeak_count_used"], note=note,
                      neutral=[r for r in summary if r["condition"] == "neutral"])}


def epoch_mask(times, selection, emotion, level, *, fs=None):
    """Boolean mask of the 4-Hz samples inside the saved intensity epochs of one movie and tail."""
    rows = selection[(selection.emotion == emotion) & (selection.level == level)]
    mask = np.zeros(len(times), dtype=bool)
    for start, stop in zip(rows.dbhi_epoch_start_s, rows.dbhi_epoch_stop_s):
        mask |= (times >= start - 1e-9) & (times < stop - 1e-9)
    if fs is not None and (len(rows) != 48 or mask.sum() != 48 * fs):
        raise ValueError(f"Incomplete {emotion}/{level} coverage.")
    return mask


def main():
    from config.dbhi_config import make_parser, paths_from_args

    parser = make_parser(__doc__, ("participants", "cache", "neutral", "reports_dir"))
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Also check the model completion report and the truncated neutral clip.",
    )
    args = parser.parse_args()
    manifest, _, ids, _ = validate_model_cache(paths_from_args(args), temporal=args.temporal)
    print(f"Validated {len(ids)} caches; bands: {manifest['band_order']}.")


if __name__ == "__main__":
    main()
