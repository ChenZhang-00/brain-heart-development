"""Movie-evoked SAM ratings and cardiac responses (Fig. S1a–b)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from importlib import import_module
_eliciting_config = import_module("02_eliciting_effects.config.eliciting_config")
CONDITIONS = _eliciting_config.CONDITIONS
LABELS = _eliciting_config.LABEL
make_paths = _eliciting_config.make_paths
relative_path = import_module("common.paths").relative_path
parse_analysis_args = _eliciting_config.parse_analysis_args
_inputs = import_module("02_eliciting_effects.analysis.inputs")
load_cohort = _inputs.load_cohort
record_run = _inputs.record_run


def omnibus(values):
    """Hotelling repeated-measures F for the three movie conditions."""
    n, k = values.shape
    differences = np.diff(values, axis=1)
    mean = differences.mean(axis=0)
    covariance = np.cov(differences, rowvar=False)
    t_squared = n * mean @ np.linalg.solve(covariance, mean)
    f = (n - k + 1) * t_squared / ((n - 1) * (k - 1))
    return dict(n=n, F_mv=f, df1_mv=k - 1, df2_mv=n - k + 1, p_mv=stats.f.sf(f, k - 1, n - k + 1))


def posthoc(values):
    """Paired t tests with Holm correction across three contrasts per measure."""
    rows = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        difference = values[:, i] - values[:, j]
        test = stats.ttest_rel(values[:, i], values[:, j])
        half_width = stats.t.ppf(0.975, len(values) - 1) * stats.sem(difference)
        rows.append(
            dict(
                i=i,
                j=j,
                a=LABELS[CONDITIONS[i]],
                b=LABELS[CONDITIONS[j]],
                mean_diff=difference.mean(),
                ci_lo=difference.mean() - half_width,
                ci_hi=difference.mean() + half_width,
                dz=difference.mean() / difference.std(ddof=1),
                t=test.statistic,
                p_raw=test.pvalue,
            )
        )
    result = pd.DataFrame(rows)
    result["p_corr"] = multipletests(result.p_raw, method="holm")[1]
    result["stars"] = result.p_corr.map(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    )
    return result


def run_analysis(paths=None, *, check_only=False, print_tables=True):
    paths = paths or make_paths()
    info, cache = load_cohort(paths)
    # Retain the original condition-indexed array layout for exact reproducibility.
    order = list(range(len(CONDITIONS)))
    values = {
        "IBI": np.nanmean(cache["ibi"][:, order], axis=2),
        "HF": np.nanmean(np.log(cache["hf"][:, order]), axis=2),
        "LF": np.nanmean(np.log(cache["lf"][:, order]), axis=2),
    }
    info = info.set_index("ID").loc[cache["ids"]]
    if check_only:
        for name, x in values.items():
            if not np.isfinite(x).all():
                raise ValueError(f"{name}: missing participant-level means.")
        print("Movie-response input check passed for 148 participants.")
        return
    values.update(
        {
            name: info[[f"{c}_{suffix}" for c in CONDITIONS]].to_numpy(float)
            for name, suffix in [("Arousal", "A"), ("Valence", "V")]
        }
    )
    omnibus_rows, pairwise, means = [], [], []
    for name, x in values.items():
        if not np.isfinite(x).all():
            raise ValueError(f"{name}: missing participant-level means.")
        omnibus_rows.append(dict(measure=name, **omnibus(x)))
        pairwise.append(posthoc(x).assign(measure=name))
        for i, condition in enumerate(CONDITIONS):
            means.append(
                dict(
                    measure=name,
                    condition=condition,
                    n=len(x),
                    mean=x[:, i].mean(),
                    sd=x[:, i].std(ddof=1),
                )
            )
    out = paths.results
    out.mkdir(parents=True, exist_ok=True)
    omnibus_table = pd.DataFrame(omnibus_rows)
    omnibus_table.to_csv(out / "eliciting_effects_omnibus.csv", index=False)
    pd.concat(pairwise, ignore_index=True).to_csv(
        out / "eliciting_effects_posthoc.csv", index=False
    )
    pd.DataFrame(means).to_csv(out / "eliciting_effects_means.csv", index=False)
    if print_tables:
        print(omnibus_table.to_string(index=False))

    record_run(
        paths,
        "movie_responses",
        [paths.participants, paths.ecg],
        {
            "omnibus_test": "Hotelling repeated-measures F",
            "posthoc_test": "two-sided paired t",
            "posthoc_adjustment": "Holm across three contrasts within each measure",
            "cardiac_summary": "temporal mean of IBI or natural-log HF/LF power",
        },
    )
    print("Saved response tables to:", relative_path(out, paths.root))
    return omnibus_table


def main(argv=None):
    paths, check_only = parse_analysis_args(__doc__, argv)
    run_analysis(paths, check_only=check_only)


if __name__ == "__main__":
    main()
