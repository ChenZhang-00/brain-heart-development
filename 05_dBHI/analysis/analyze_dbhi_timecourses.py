"""Temporal sign-flip clusters: emotional viewing minus each child’s neutral mean (Tables S4, Figs. S3–S4)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.dbhi_config import parse_paths, shown


def run(paths):
    import json
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from scipy import stats

    from sdg.pipeline import atomic_json, file_hash, fingerprint
    from config.dbhi_config import BANDS, FS, GROUPS, REGIONS, HRV
    from config.transform_dbhi import params as transform_params, transform
    from analysis.inputs import validate_model_cache

    CACHE, OUT = paths.cache, paths.time_results
    OUT.mkdir(parents=True, exist_ok=True)
    INTENSITY = paths.ratings
    N_PERM, ALPHA, MIN_LEN, SEED = 10000, 0.05, 20, 0
    EMOTIONS = {"fear": "Fearful – The Lion King", "happy": "Happy – The Jungle Book"}
    EXCLUDE = set()
    RUN_BANDS = BANDS
    FORCE = True
    cache_manifest, info, ANALYSIS_IDS, truncation_info = validate_model_cache(paths, temporal=True)
    ages = info.loc[ANALYSIS_IDS, "Age_group"].to_numpy()
    cohort = pd.DataFrame({"ID": ANALYSIS_IDS, "Age_group": ages})
    cohort["analysis_group"] = [
        " / ".join(g for g, a in GROUPS.items() if age in a) for age in ages
    ]
    cohort.to_csv(OUT / "included_subjects.csv", index=False)
    print(cohort.groupby("analysis_group").size().rename("n").to_frame())
    atomic_json(
        OUT / "analysis_parameters.json",
        {
            "cache": shown(CACHE),
            "cache_fingerprint": fingerprint(cache_manifest),
            "bands": RUN_BANDS,
            **transform_params(),
            "fs": FS,
            "n_perm": N_PERM,
            "alpha": ALPHA,
            "min_len_samples": MIN_LEN,
            "seed": SEED,
            "exclude": sorted(EXCLUDE),
            "neutral_truncation": truncation_info,
            "groups": GROUPS,
            "regions_1based": REGIONS,
            "across_series_correction": "none; 72 series per band, 288 total",
            "age_csv_sha256": file_hash(paths.participants),
            "intensity_csv_sha256": file_hash(INTENSITY),
        },
    )

    # transform(values, direction) from transform_dbhi: per-electrode log10 (htb floored), applied before the ROI mean

    def load(BAND):
        """Return aligned values[subject, hrv, time] and age-group membership."""
        info = pd.read_csv(paths.participants, dtype={"ID": str}).set_index("ID")
        bi = BANDS.index(BAND)
        rows = {}
        for sid in ANALYSIS_IDS:
            with np.load(CACHE / f"{sid}_dbhi.npz", allow_pickle=False) as z:
                for direction in ("htb", "bth"):
                    # Read each large array once per band, rather than once per ROI.
                    arrays = {
                        cond: z[f"{cond}__{direction}"][bi] for cond in ["neutral", *EMOTIONS]
                    }
                    times = {emotion: z[f"{emotion}__t_{direction}"] for emotion in EMOTIONS}
                    for region, channels in REGIONS.items():
                        idx = np.array(channels) - 1
                        neutral = transform(arrays["neutral"][:, idx], direction).mean(1)
                        base = np.nanmean(neutral, axis=-1)
                        if not np.isfinite(base).all():
                            raise ValueError(
                                f"{sid}/{direction}/{region}: invalid neutral baseline."
                            )
                        for emotion in EMOTIONS:
                            y = (
                                transform(arrays[emotion][:, idx], direction).mean(1)
                                - base[:, None]
                            )
                            seconds = times[emotion]
                            ticks = np.rint(seconds * FS).astype(int)
                            if not np.allclose(seconds * FS, ticks) or not np.all(
                                np.diff(ticks) == 1
                            ):
                                raise ValueError(
                                    f"{sid}/{emotion}/{direction}: irregular time grid."
                                )
                            rows.setdefault((emotion, direction, region), []).append((ticks, y))
        data = {}
        for key, records in rows.items():
            lo = max(t[0] for t, _ in records)
            hi = min(t[-1] for t, _ in records)
            if hi < lo:
                raise ValueError(f"No common physical-time interval for {key}.")
            values = np.stack([y[:, lo - t[0] : hi - t[0] + 1] for t, y in records])
            data[key] = (np.arange(lo, hi + 1) / FS, values)
        ages = info.loc[ANALYSIS_IDS, "Age_group"].to_numpy()
        return data, {g: np.isin(ages, a) for g, a in GROUPS.items()}

    def clusters(t, thr):
        out = []
        for s in (1, -1):
            m = np.r_[False, s * t > thr, False].astype(int)
            for i, j in zip(*(np.flatnonzero(np.diff(m) == k) for k in (1, -1))):
                if j - i >= MIN_LEN:
                    out.append((i, j, t[i:j].sum()))
        return out

    def cluster_test(x):
        """x[subject, time] -> list of (start, stop, t_mass, p). Rows with NaN dropped."""
        x = x[np.isfinite(x).all(1)]
        n = len(x)
        if n < 2:
            raise ValueError("Fewer than two finite subjects in a cluster test.")
        thr = stats.t.ppf(1 - ALPHA / 2, n - 1)
        obs = clusters(stats.ttest_1samp(x, 0).statistic, thr)
        if not obs:
            return [], n
        rng, ss = np.random.default_rng(SEED), np.square(x).sum(0)
        null = np.empty(N_PERM)
        for k in range(N_PERM):
            m = (rng.choice([-1.0, 1.0], n) @ x) / n
            tt = m / np.sqrt(np.maximum(ss - n * m**2, 0) / (n - 1) / n)
            null[k] = max((abs(c[2]) for c in clusters(tt, thr)), default=0)
        return [(i, j, s, (1 + (null >= abs(s)).sum()) / (N_PERM + 1)) for i, j, s in obs], n

    def main(band):
        data, members = load(band)
        rows, counts = [], []
        for region in REGIONS:
            for emotion in EMOTIONS:
                for hi, hrv in enumerate(HRV):
                    for direction in ("htb", "bth"):
                        times, values = data[emotion, direction, region]
                        for group, selected in members.items():
                            available = values[selected, hi]
                            x = available[np.isfinite(available).all(axis=1)]
                            found, n = cluster_test(x)
                            meta = dict(
                                region=region,
                                emotion=emotion,
                                direction=direction,
                                hrv=hrv,
                                group=group,
                            )
                            counts.append(
                                dict(
                                    meta,
                                    n_available=len(available),
                                    n_used=n,
                                    n_dropped=len(available) - n,
                                    start_s=times[0],
                                    end_s=times[-1],
                                )
                            )
                            for i, j, mass, p in found:
                                rows.append(
                                    dict(
                                        meta,
                                        n=n,
                                        start_s=times[i],
                                        stop_s=times[j - 1] + 1 / FS,
                                        t_mass=mass,
                                        p=p,
                                        significant=p < ALPHA,
                                    )
                                )
        columns = [
            "region",
            "emotion",
            "direction",
            "hrv",
            "group",
            "n",
            "start_s",
            "stop_s",
            "t_mass",
            "p",
            "significant",
        ]
        table = pd.DataFrame(rows, columns=columns)
        table.to_csv(OUT / f"dBHI_{band}_clusters.csv", index=False)
        pd.DataFrame(counts).to_csv(OUT / f"dBHI_{band}_sample_counts.csv", index=False)
        return dict(
            band=band,
            candidate_clusters=len(table),
            significant_clusters=int(table.significant.sum()),
            series_tested=len(counts),
        )

    summaries = []
    started = time.monotonic()
    try:
        for BAND in RUN_BANDS:
            if not FORCE and (OUT / f"dBHI_{BAND}_clusters.csv").exists():
                print("skip", BAND, "(results exist; set FORCE=True to redo)")
                continue
            print("=== band:", BAND, flush=True)
            atomic_json(
                OUT / "analysis_status.json",
                {"state": "running", "band": BAND, "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
            )
            summaries.append(main(BAND))
            pd.DataFrame(summaries).to_csv(OUT / "analysis_summary.csv", index=False)
        atomic_json(
            OUT / "analysis_status.json",
            {
                "state": "complete",
                "bands": RUN_BANDS,
                "subjects": len(ANALYSIS_IDS),
                "seconds": round(time.monotonic() - started, 1),
                "summary": summaries,
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
    except BaseException as error:
        atomic_json(
            OUT / "analysis_status.json",
            {
                "state": "failed",
                "band": BAND,
                "error": f"{type(error).__name__}: {error}",
                "completed_bands": summaries,
            },
        )
        raise
    print(pd.DataFrame(summaries))
    print("Results:", shown(OUT))


if __name__ == "__main__":
    run(parse_paths(__doc__, ("participants", "ratings", "cache", "neutral", "reports_dir")))
