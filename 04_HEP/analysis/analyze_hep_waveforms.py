"""The 54 paired temporal cluster tests for ECG-regressed regional HEP waveforms."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy import stats

from importlib import import_module
_hep_config = import_module("04_HEP.config.hep_config")
ALPHA = _hep_config.ALPHA
GROUPS = _hep_config.GROUPS
MIN_MS = _hep_config.MIN_MS
MIN_SAMPLES = _hep_config.MIN_SAMPLES
N_PERM = _hep_config.N_PERM
PAIRS = _hep_config.PAIRS
PAIR_COLORS = _hep_config.PAIR_COLORS
REGIONS = _hep_config.REGIONS
SEED = _hep_config.SEED
WINDOWS = _hep_config.WINDOWS
make_paths = _hep_config.make_paths
relative_path = import_module("common.paths").relative_path
parse_args = _hep_config.parse_args
_inputs = import_module("04_HEP.analysis.inputs")
load_roi = _inputs.load_roi
record_run = _inputs.record_run


def find_clusters(values, threshold):
    """Return same-sign runs above the two-sided threshold, lasting >= MIN_SAMPLES."""
    clusters, i = [], 0
    while i < len(values):
        if abs(values[i]) <= threshold:
            i += 1
            continue
        j = i + 1
        while (
            j < len(values)
            and abs(values[j]) > threshold
            and np.sign(values[j]) == np.sign(values[i])
        ):
            j += 1
        if j - i >= MIN_SAMPLES:
            clusters.append((i, j, float(values[i:j].sum())))
        i = j
    return clusters


def paired_t(difference):
    sd = difference.std(axis=0, ddof=1)
    return difference.mean(axis=0) / np.where(sd == 0, np.inf, sd / np.sqrt(len(difference)))


def permutation_clusters(difference):
    threshold = stats.t.ppf(1 - ALPHA / 2, len(difference) - 1)
    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        signs = rng.choice([-1.0, 1.0], (len(difference), 1))
        candidates = find_clusters(paired_t(difference * signs), threshold)
        null[i] = max((abs(mass) for _, _, mass in candidates), default=0.0)
    return [
        (start, end, mass, (1 + (null >= abs(mass)).sum()) / (N_PERM + 1))
        for start, end, mass in find_clusters(paired_t(difference), threshold)
    ]


def compute_statistics(times, waves, groups):
    rows = []
    for group in GROUPS:
        for region in REGIONS:
            print(f"{group}, {region}: six tests with {N_PERM:,} sign flips each", flush=True)
            for low, high in WINDOWS:
                mask = (times >= low) & (times <= high)
                window_times = times[mask]
                for a, b in PAIRS:
                    difference = (waves[region, a] - waves[region, b])[groups == group][:, mask]
                    for start, end, mass, p in permutation_clusters(difference):
                        average = difference[:, start:end].mean(axis=1)
                        rows.append(
                            dict(
                                group=group,
                                region=region,
                                test="pairwise",
                                pair=f"{a} - {b}",
                                window=f"{low}-{high}",
                                t_start=window_times[start],
                                t_end=window_times[end - 1],
                                stat=mass,
                                p=p,
                                dz=average.mean() / average.std(ddof=1),
                            )
                        )
    columns = ["group", "region", "test", "pair", "window", "t_start", "t_end", "stat", "p", "dz"]
    return pd.DataFrame(rows, columns=columns)


def run_analysis(paths=None, *, check_only=False, print_tables=True):
    paths = paths or make_paths()
    data, times, ids, groups, waves = load_roi(paths)
    if check_only:
        print("ROI waveform inputs validated for 148 participants.")
        return
    clusters = compute_statistics(times, waves, groups)
    clusters["survives"] = clusters.p < ALPHA
    assert clusters.p.between(0, 1).all() and clusters.pair.isin(PAIR_COLORS).all()
    paths.results.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(paths.clusters, index=False)
    significant = clusters.loc[clusters.survives].copy()
    if print_tables:
        print(
            significant[["group", "region", "pair", "window", "t_start", "t_end", "p"]].to_string(
                index=False
            )
        )
    record_run(
        paths,
        "waveforms",
        [paths.participants, paths.roi],
        {
            "n_tests": 54,
            "n_permutations_per_test": N_PERM,
            "seed_reset_per_test": SEED,
            "alpha_two_sided": ALPHA,
            "minimum_cluster_samples": MIN_SAMPLES,
            "minimum_cluster_ms": MIN_MS,
            "windows_ms_inclusive": WINDOWS,
            "window_boundary_300ms_in_both_tests": True,
            "cluster_correction": "maximum absolute cluster mass within each test",
            "correction_across_tests": False,
            "n_candidate_clusters": len(clusters),
            "n_significant_clusters": len(significant),
        },
    )
    print("Saved:", relative_path(paths.clusters, paths.root))
    return clusters


def main(argv=None):
    paths, check_only = parse_args(__doc__, argv)
    run_analysis(paths, check_only=check_only)


if __name__ == "__main__":
    main()
