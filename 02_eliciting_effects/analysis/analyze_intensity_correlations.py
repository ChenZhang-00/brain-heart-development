"""Table S2: 15 intensity correlations per movie and 10,000 phase surrogates.

Also save figure_s1_inputs.npz so Figure S1 can be redrawn without loading raw inputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from importlib import import_module
_eliciting_config = import_module("02_eliciting_effects.config.eliciting_config")
BANDS = _eliciting_config.BANDS
CONDITIONS = _eliciting_config.CONDITIONS
EMOTIONS = _eliciting_config.EMOTIONS
FS_IBI = _eliciting_config.FS_IBI
N_SURR = _eliciting_config.N_SURR
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
load_intensity = _inputs.load_intensity
record_run = _inputs.record_run


def _block_means(values, block):
    """Group-mean of per-child block means; all-NaN blocks stay NaN without a warning."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        return np.nanmean(values.reshape(values.shape[0], -1, block), axis=2).mean(0)


def phase_surrogates(x, n, rng):
    X = np.fft.rfft(x - x.mean())
    k = X.size
    ph = rng.uniform(0, 2 * np.pi, size=(n, k))
    ph[:, 0] = 0
    if x.size % 2 == 0:
        ph[:, -1] = 0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), n=x.size, axis=1)


def run_analysis(paths=None, *, check_only=False, print_tables=True):
    paths = paths or make_paths()
    info, cache = load_cohort(paths)
    intensity = load_intensity(paths)
    files = get_eeg_files(paths, cache["ids"])
    if check_only:
        check_eeg_inputs(files, REGIONS)
        print("Full input check passed: 148 participants, cardiac data, EEG and intensity ratings.")
        return
    ibi, lf, hf = cache["ibi"], cache["lf"], cache["hf"]
    value = {
        "IBI": np.nanmean(ibi, axis=2),
        "HF": np.nanmean(np.log(hf), axis=2),
        "LF": np.nanmean(np.log(lf), axis=2),
    }
    if any(not np.isfinite(x).all() for x in value.values()):
        raise ValueError("Cardiac participant-level means must be finite.")
    ni = CONDITIONS.index("neutral")
    dIBI = {
        e: ibi[:, CONDITIONS.index(e), :] - np.nanmean(ibi[:, ni, :], axis=1, keepdims=True)
        for e in EMOTIONS
    }
    t_ibi = np.arange(ibi.shape[2]) / FS_IBI
    dPow = load_eeg_differences(files, REGIONS, verbose=print_tables)
    eeg_mean = {
        r: {b: {e: np.nanmean(np.vstack(dPow[r][b][e]), axis=0) for e in EMOTIONS} for b in BANDS}
        for r in REGIONS
    }
    ibi_mean = {e: np.nanmean(dIBI[e], axis=0) for e in EMOTIONS}
    TAB = paths.results
    TAB.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows = []
    for e in EMOTIONS:
        if print_tables:
            print(f"{e}: {N_SURR:,} phase-randomised surrogates", flush=True)
        ei = CONDITIONS.index(e)
        y = intensity[e]
        surr = phase_surrogates(y, N_SURR, rng)
        series = {
            "IBI": _block_means(dIBI[e], FS_IBI)[:T_EEG]
        }
        for nm, arr in (("LF-HRV", lf), ("HF-HRV", hf)):
            la = np.log(arr)
            series[nm] = np.nanmean(
                la[:, ei, :T_EEG] - np.nanmean(la[:, ni, :], axis=1, keepdims=True), axis=0
            )
        for r in REGIONS:
            for b in BANDS:
                series[f"{r} {b}"] = eeg_mean[r][b][e][:T_EEG]
        for nm, x in series.items():
            ok = np.isfinite(x) & np.isfinite(y)
            r_, p_ = stats.pearsonr(x[ok], y[ok])
            xs = (x[ok] - x[ok].mean()) / x[ok].std()
            S = surr[:, ok]
            S = (S - S.mean(1, keepdims=True)) / S.std(1, keepdims=True)
            r_surr = S @ xs / ok.sum()
            p_surr = (np.sum(np.abs(r_surr) >= abs(r_)) + 1) / (N_SURR + 1)
            rows.append(
                {
                    "emotion": e,
                    "measure": nm,
                    "r": r_,
                    "p_raw": p_,
                    "p_surrogate": p_surr,
                    "n_time": int(ok.sum()),
                }
            )
    corr = pd.DataFrame(rows)
    corr["p_fdr"] = np.nan
    for e in EMOTIONS:
        m = corr.emotion == e
        corr.loc[m, "p_fdr"] = multipletests(corr.loc[m, "p_raw"], method="fdr_bh")[1]
    corr.to_csv(TAB / "intensity_correlations_long.csv", index=False)
    fmtp = lambda x: "<.001" if x < 0.001 else f"{x:.3f}"
    t2 = []
    for meas in list(corr.measure.unique()):
        r = {"Measure": meas}
        for e, lab in [("fear", "Fearful"), ("happy", "Happy")]:
            x = corr[(corr.measure == meas) & (corr.emotion == e)].iloc[0]
            r[f"{lab} r"] = f"{x.r:.2f}"
            r[f"{lab} pFDR"] = fmtp(x.p_fdr)
            r[f"{lab} p surrogate"] = fmtp(x.p_surrogate)
        t2.append(r)
    T2 = pd.DataFrame(t2)
    paths.tables.mkdir(parents=True, exist_ok=True)
    T2.to_csv(paths.tables / "Table_S2_intensity_correlations.csv", index=False)
    if print_tables:
        print(T2.to_string(index=False))
        print("n_time:", corr.n_time.unique())

    figure_inputs = {
        "schema_version": np.array(1),
        "n_subjects": np.array(len(info)),
        "conditions": np.array(CONDITIONS),
        "emotions": np.array(EMOTIONS),
        "bands": np.array(BANDS),
        "t_ibi": t_ibi,
    }
    for measure, values in value.items():
        figure_inputs[f"value_{measure}"] = values
    for e in EMOTIONS:
        # Retain CSV row order for the original seed-42 scatter jitter.
        figure_inputs[f"arousal_{e}"] = (info[f"{e}_A"] - info["neutral_A"]).to_numpy()
        figure_inputs[f"valence_{e}"] = (info[f"{e}_V"] - info["neutral_V"]).to_numpy()
        figure_inputs[f"intensity_{e}"] = intensity[e]
        figure_inputs[f"ibi_mean_{e}"] = ibi_mean[e]
        for b in BANDS:
            figure_inputs[f"eeg_Frontal_{b}_{e}"] = eeg_mean["Frontal"][b][e]
    np.savez_compressed(TAB / "figure_s1_inputs.npz", **figure_inputs)
    record_run(
        paths,
        "intensity",
        [paths.participants, paths.ecg, paths.ratings, *files],
        {
            "n_surrogates": N_SURR,
            "seed": SEED,
            "fdr_method": "BH on 15 ordinary Pearson p values separately within each movie",
            "surrogate_p_adjustment": None,
            "surrogate_method": "randomise Fourier phases, fixing DC and even-length Nyquist phase to zero",
            "intensity_rows": [0, T_EEG - 1],
            "intensity_scaling": "min-max; no temporal smoothing",
            "baseline": "each child's neutral temporal mean",
            "eeg_display_time_s": [0, T_EEG - 1],
            "ibi_reduction": "nanmean over consecutive four-sample blocks, then ordinary mean over children",
        },
    )
    print("Saved Table S2 and figure inputs to:", relative_path(TAB, paths.root))
    return corr


def main(argv=None):
    paths, check_only = parse_analysis_args(__doc__, argv)
    run_analysis(paths, check_only=check_only)


if __name__ == "__main__":
    main()
