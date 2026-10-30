"""Truncate the documented neutral clip to its R-peak-supported segment (0–159.342 s), in place.

The participant's model array in the cache is replaced after validation; fear/happy arrays
are kept unchanged. Running again on an already truncated array does nothing."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.dbhi_config import parse_paths, shown


def run(settings):
    import json
    import sys
    import time
    import warnings
    from pathlib import Path

    import h5py
    import numpy as np

    from sdg.model import PySDG
    from sdg.pipeline import (
        DEFAULT_CONFIG,
        canonical,
        read_metadata,
        subject_paths,
        validate_arrays,
        input_identity,
        validate_cache,
        make_manifest,
        fingerprint,
    )

    SID = settings.neutral_subject
    CONFIG = json.loads(json.dumps(DEFAULT_CONFIG))
    ORIG = settings.cache / f"{SID}_dbhi.npz"
    OUT = ORIG
    with np.load(ORIG, allow_pickle=False) as existing:
        clip = next(c for c in json.loads(str(existing["_qc_json"].item()))["clips"] if c["condition"] == "neutral")
    if "segment_used_s" in clip:
        print(f"{SID}: neutral clip already truncated to {clip['segment_used_s']} s; nothing to do.")
        return
    manifest = json.loads((settings.cache / "_manifest.json").read_text())
    if canonical(manifest) != canonical(make_manifest(CONFIG)):
        raise RuntimeError("Model/configuration/environment differs from the main cache.")
    paths = subject_paths(SID)
    fs, labels, conditions, clips = read_metadata(*paths, CONFIG)
    validate_cache(ORIG, CONFIG, fingerprint(manifest), input_identity(paths), labels)
    clip = next(c for c in clips if c["condition"] == "neutral")
    r = clip["r_seconds"]
    gaps = np.diff(r)
    large_gaps = np.flatnonzero(gaps > CONFIG["rr_gap_warning_s"])
    if not len(large_gaps):
        raise ValueError("Expected the documented neutral R-peak gap for sub-047.")
    first_gap = int(large_gaps[0])
    cut_s = float(r[first_gap])  # last R peak before the first gap
    if not np.isclose(cut_s, 159.342, atol=0.01, rtol=0):
        raise ValueError(f"Expected the documented neutral cut at 159.342 s; found {cut_s:.3f} s.")
    keep = r[r <= cut_s]
    n_samples = int(round(cut_s * fs)) + 1  # EEG samples 0 .. cut_s inclusive
    print(
        f'neutral clip {clip["duration_s"]:.3f} s, {clip["rpeak_count"]} R peaks; first gap at {cut_s:.3f}-{r[first_gap+1]:.3f} s'
    )
    print(
        f"using segment 0-{cut_s:.3f} s: {len(keep)} R peaks, {n_samples} EEG samples, max RR {np.diff(keep).max():.3f} s"
    )

    with np.load(ORIG, allow_pickle=False) as z:
        output = {k: z[k] for k in z.files}
        orig_qc = json.loads(str(z["_qc_json"].item()))
    if output["labels"].tolist() != labels:
        raise RuntimeError("channel labels differ from the cache")

    t0 = time.monotonic()
    with warnings.catch_warnings(record=True) as caught, h5py.File(paths[0], "r") as f:
        warnings.simplefilter("always")
        eeg = np.asarray(
            f["EEG"]["data"][clip["start"] : clip["start"] + n_samples, :], dtype=np.float64
        ).T
        if not np.isfinite(eeg).all():
            raise ValueError("nonfinite EEG values")
        obj = PySDG(
            eeg=eeg,
            eeg_fs=fs,
            r_peaks=keep,
            flims_eeg=CONFIG["flims_eeg"],
            fs_bhi=CONFIG["fs_bhi"],
            window=CONFIG["window"],
            parallelize=False,
            progressbar=False,
        )
        obj.compute_tfr_eeg(window=CONFIG["tfr_eeg_win"])
        obj.compute_tfr_hrv()
        obj.compute_sdg()
        for direction, time_key in (("bth", "time_BToH"), ("htb", "time_HToB")):
            output[f"neutral__{direction}"] = obj.bhi[direction]
            output[f"neutral__t_{direction}"] = obj.bhi["t"][time_key]
            print(
                direction,
                obj.bhi[direction].shape,
                "t",
                obj.bhi["t"][time_key][0],
                obj.bhi["t"][time_key][-1],
                "nan%",
                round(100 * np.isnan(obj.bhi[direction]).mean(), 2),
            )
    print(
        f"SDG done in {time.monotonic() - t0:.1f} s; warnings: {sorted(set(str(w.message) for w in caught))}"
    )

    summary = validate_arrays(output, CONFIG, labels)
    qc = dict(orig_qc)
    for c in qc["clips"]:
        if c["condition"] == "neutral":
            c.update(
                {
                    "segment_used_s": [0.0, cut_s],
                    "rpeak_count_used": int(len(keep)),
                    "note": "neutral recomputed on the R-peak-supported segment before the first gap (recompute_neutral_dbhi.py)",
                }
            )
    output["_qc_json"] = np.array(canonical(qc))
    output["_note"] = np.array(
        "neutral arrays recomputed on 0-%.3f s; fear/happy unchanged" % cut_s
    )
    # Replace the cached array only after a complete, validated write.
    import os
    import uuid

    temp = OUT.with_name(OUT.stem + "." + uuid.uuid4().hex + ".part.npz")
    try:
        np.savez(temp, **output)
        validate_cache(temp, CONFIG, fingerprint(manifest), input_identity(paths), labels)
        os.replace(temp, OUT)
    finally:
        temp.unlink(missing_ok=True)
    settings.reports.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "subject": SID,
            "neutral_segment_s": [0.0, cut_s],
            "rpeaks_used": int(len(keep)),
            "eeg_samples": n_samples,
            "config": CONFIG,
            "source_cache": shown(ORIG),
            "arrays": summary,
        },
        open(settings.reports / "neutral_truncation.json", "w", encoding="utf-8"),
        indent=2,
    )
    print("wrote", OUT)


if __name__ == "__main__":
    run(parse_paths(__doc__, ("cache", "neutral", "reports_dir")))
