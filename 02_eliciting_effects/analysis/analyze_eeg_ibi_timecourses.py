"""Temporal sign-flip cluster tests for emotion-minus-neutral IBI and frontal EEG.

The reported convention uses the first 176 EEG samples on a 0–175 s index axis,
and zero-fills missing differences for the cluster tests.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
import numpy as np
import pandas as pd
from scipy import stats

from importlib import import_module
_eliciting_config = import_module("02_eliciting_effects.config.eliciting_config")
BANDS = _eliciting_config.BANDS
CL_ALPHA = _eliciting_config.CL_ALPHA
CONDITIONS = _eliciting_config.CONDITIONS
EMOTIONS = _eliciting_config.EMOTIONS
FS_IBI = _eliciting_config.FS_IBI
MIN_LEN = _eliciting_config.MIN_LEN
N_PERM = _eliciting_config.N_PERM
REGIONS = _eliciting_config.REGIONS
SEED = _eliciting_config.SEED
T_EEG = _eliciting_config.T_EEG
make_paths = _eliciting_config.make_paths
relative_path = import_module("common.paths").relative_path
parse_analysis_args = _eliciting_config.parse_analysis_args
_inputs = import_module("02_eliciting_effects.analysis.inputs")
check_eeg_inputs = _inputs.check_eeg_inputs
get_eeg_files = _inputs.get_eeg_files
load_cohort = _inputs.load_cohort
load_eeg_differences = _inputs.load_eeg_differences
record_run = _inputs.record_run


def cluster_perm_1samp(D, min_len=MIN_LEN, alpha=CL_ALPHA, n_perm=N_PERM, seed=SEED):
    # One-sample temporal cluster test of D (n_subj x T) against 0, sign-flip null.
    # Returns: boolean sig mask (T,), list of (start, end, tsum, p), permutation maxdist.
    D = np.asarray(D, float)
    n, T = D.shape
    if n < 3:
        # not enough subjects to permute (e.g. pilot with one file loaded)
        return np.zeros(T, bool), [], np.array([])

    def tstat(X):
        m = X.mean(0)
        s = X.std(0, ddof=1)
        s[s == 0] = np.inf
        return m / (s / np.sqrt(n))

    tcrit = stats.t.ppf(1 - alpha / 2, n - 1)

    def find_clusters(t):
        mask = np.abs(t) > tcrit
        out, i = [], 0
        while i < T:
            if mask[i]:
                j = i
                while j < T and mask[j] and np.sign(t[j]) == np.sign(t[i]):
                    j += 1
                if j - i >= min_len:
                    out.append((i, j, float(t[i:j].sum())))
                i = j
            else:
                i += 1
        return out

    obs = find_clusters(tstat(D))
    rng = np.random.default_rng(seed)
    maxdist = np.empty(n_perm)
    for p in range(n_perm):
        flip = rng.choice([-1.0, 1.0], size=(n, 1))
        cl = find_clusters(tstat(D * flip))
        maxdist[p] = max((abs(c[2]) for c in cl), default=0.0)

    sig = np.zeros(T, bool)
    kept = []
    for i, j, s in obs:
        pval = (1 + np.sum(maxdist >= abs(s))) / (n_perm + 1)
        if pval < alpha:
            sig[i:j] = True
        kept.append((i, j, s, pval))
    return sig, kept, maxdist


def run_analysis(paths=None, *, check_only=False, print_tables=True):
    paths = paths or make_paths()
    _, cache = load_cohort(paths)
    eeg_files = get_eeg_files(paths, cache["ids"])
    if check_only:
        check_eeg_inputs(eeg_files, {"Frontal": REGIONS["Frontal"]})
        print("Time-course input check passed for 148 participants.")
        return
    # Suppress only the expected empty-bin warning; retain the reported NaN handling.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        return _compute(paths, cache, eeg_files, print_tables=print_tables)


def _compute(paths, cache, eeg_files, *, print_tables=True):
    CLUST_CSV = paths.results / "neuro_dynamics_clusters.csv"
    ibi = cache["ibi"]
    t_ibi = np.arange(ibi.shape[2]) / FS_IBI  # 0..176 s
    ne_base = np.nanmean(ibi[:, CONDITIONS.index("neutral"), :], axis=1, keepdims=True)

    dIBI = {}  # emotion -> (N, 704) per-subject ΔIBI
    for e in EMOTIONS:
        dIBI[e] = ibi[:, CONDITIONS.index(e), :] - ne_base

    ibi_mean, ibi_sig, ibi_clusters = {}, {}, {}
    for e in EMOTIONS:
        ibi_mean[e] = np.nanmean(dIBI[e], axis=0)
        if print_tables:
            print(f"IBI {e}: {N_PERM:,} sign-flip permutations", flush=True)
        sig, kept, _ = cluster_perm_1samp(np.nan_to_num(dIBI[e]), min_len=MIN_LEN * FS_IBI)
        ibi_sig[e] = sig
        ibi_clusters[e] = kept

    if print_tables:
        print(f"IBI: N = {ibi.shape[0]} subjects")
        for e in EMOTIONS:
            print(
                f"  {e:5s}  range {ibi_mean[e].min():6.1f}..{ibi_mean[e].max():5.1f} ms"
                f"   sig samples: {int(ibi_sig[e].sum())}/{len(ibi_sig[e])}"
            )

    dPow = load_eeg_differences(eeg_files, {"Frontal": REGIONS["Frontal"]}, verbose=print_tables)["Frontal"]
    eeg_mean = {b: {} for b in BANDS}
    eeg_sig = {b: {} for b in BANDS}
    eeg_clusters = {b: {} for b in BANDS}
    N_eeg = len(eeg_files)
    for b in BANDS:
        for e in EMOTIONS:
            M = np.vstack(dPow[b][e]) if dPow[b][e] else np.empty((0, T_EEG))
            eeg_mean[b][e] = np.nanmean(M, axis=0) if M.size else np.full(T_EEG, np.nan)
            if print_tables:
                print(f"EEG {b} {e}: {N_PERM:,} sign-flip permutations", flush=True)
            sig, kept, _ = (
                cluster_perm_1samp(np.nan_to_num(M))
                if M.shape[0] >= 3
                else (np.zeros(T_EEG, bool), [], np.array([]))
            )
            eeg_sig[b][e] = sig
            eeg_clusters[b][e] = kept

    if print_tables:
        print(f"EEG group N = {N_eeg}")

    rows = []
    for e in EMOTIONS:
        for i, j, s, p in ibi_clusters[e]:
            rows.append(
                {
                    "signal": "IBI",
                    "band": "",
                    "emotion": e,
                    "t_start_s": i / FS_IBI,
                    "t_end_s": j / FS_IBI,
                    "tsum": s,
                    "p": p,
                }
            )
        for b in BANDS:
            for i, j, s, p in eeg_clusters[b][e]:
                rows.append(
                    {
                        "signal": "EEG",
                        "band": b,
                        "emotion": e,
                        "t_start_s": float(i),
                        "t_end_s": float(j),
                        "tsum": s,
                        "p": p,
                    }
                )

    clust = pd.DataFrame(
        rows, columns=["signal", "band", "emotion", "t_start_s", "t_end_s", "tsum", "p"]
    )
    CLUST_CSV.parent.mkdir(parents=True, exist_ok=True)
    clust.to_csv(CLUST_CSV, index=False)
    print(f"{len(clust)} clusters -> {relative_path(CLUST_CSV, paths.root)}")
    if print_tables:
        print(clust.to_string(index=False) if len(clust) else "(no candidate clusters)")

    record_run(
        paths,
        "timecourses",
        [paths.participants, paths.ecg, *eeg_files],
        {
            "n_permutations": N_PERM,
            "seed_reset_per_test": SEED,
            "cluster_forming_alpha_two_sided": CL_ALPHA,
            "cluster_p_threshold": CL_ALPHA,
            "minimum_cluster_seconds": MIN_LEN,
            "missing_differences": "numpy.nan_to_num before cluster tests",
            "baseline": "each child's neutral temporal mean",
            "eeg_samples": "first 176 of 177, after channel log-mean and temporal interpolation",
            "eeg_display_time_s": [0, 175],
            "ibi_display_time_s": [0, 175.75],
            "cross_signal_or_emotion_correction": False,
            "exported_clusters": "all observed candidates, including p >= 0.05",
        },
    )
    return clust


def main(argv=None):
    paths, check_only = parse_analysis_args(__doc__, argv)
    run_analysis(paths, check_only=check_only)


if __name__ == "__main__":
    main()
