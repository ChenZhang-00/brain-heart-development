"""Table S3: five HEP clusters and continuous-age regression.

BH-FDR covers the five age regressions; within-group correlations are descriptive.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from statsmodels.stats.multitest import multipletests

from importlib import import_module
_hep_config = import_module("04_HEP.config.hep_config")
make_paths = _hep_config.make_paths
parse_args = _hep_config.parse_args
_inputs = import_module("04_HEP.analysis.inputs")
load_roi = _inputs.load_roi
load_clusters = _inputs.load_clusters
record_run = _inputs.record_run
require = _inputs.require


def run_analysis(paths=None, *, check_only=False, print_tables=True):
    paths = paths or make_paths()
    load_roi(paths)
    checked = load_clusters(paths)
    require(
        int(checked.survives.sum()) == 5,
        "Table S3 reproduces the five reported clusters; review the analysis if this count changes.",
    )
    if check_only:
        print("Table S3 inputs validated.")
        return
    OUT = paths.results
    OUT.mkdir(parents=True, exist_ok=True)
    paths.tables.mkdir(parents=True, exist_ok=True)
    mat = sio.loadmat(paths.roi)
    times = mat["times"].ravel().astype(int)
    ids = np.array([str(x[0]) if isinstance(x, np.ndarray) else str(x) for x in mat["ids"].ravel()])
    ids = np.array([s.strip() for s in ids])
    info = pd.read_csv(paths.participants, encoding="utf-8-sig", dtype={"ID": str}).set_index("ID")
    age = info.loc[ids, "Age_month"].to_numpy(float)
    grp = (
        pd.cut(info.loc[ids, "Age_group"], [4, 6, 8, 10], labels=["5-6", "7-8", "9-10"])
        .astype(str)
        .to_numpy()
    )
    assert len(ids) == 148 and not np.isnan(age).any()

    cl = pd.read_csv(paths.clusters)
    sig = cl[cl.survives].copy()

    # (a) cluster table
    ta = sig.assign(
        **{
            "Age group": sig.group,
            "Region": sig.region,
            "Contrast": sig.pair.str.replace(" - ", " − ")
            .str.capitalize()
            .str.replace("neutral", "Neutral")
            .str.replace("happy", "Happy")
            .str.replace("fear", "Fear"),
            "Window (ms)": sig.window,
            "Cluster (ms)": sig.apply(lambda r: f"{int(r.t_start)}–{int(r.t_end)}", axis=1),
            "Cluster mass (sum t)": sig.stat.round(1),
            "Cluster p": sig.p.round(3),
            "dz": sig.dz.round(2),
        }
    )[
        [
            "Age group",
            "Region",
            "Contrast",
            "Window (ms)",
            "Cluster (ms)",
            "Cluster mass (sum t)",
            "Cluster p",
            "dz",
        ]
    ]
    ta.to_csv(paths.tables / "Table_S3a_hep_clusters.csv", index=False)
    if print_tables:
        print(ta.to_string())

    # (b) continuous-age linear regression (OLS only)
    rows, subj = [], []
    for _, c in sig.iterrows():
        a, b = c.pair.split(" - ")
        diff = mat[f"{c.region}__{a}"] - mat[f"{c.region}__{b}"]
        sel = (times >= c.t_start) & (times <= c.t_end)
        y = diff[:, sel].mean(axis=1)
        lr = stats.linregress(age, y)
        gmask = grp == c.group
        lr_g = stats.linregress(
            age[gmask], y[gmask]
        )  # within the defining age group only (descriptive; selection-biased)
        rows.append(
            {
                "Age group (cluster)": c.group,
                "Region": c.region,
                "Contrast": ta.loc[c.name, "Contrast"],
                "Cluster (ms)": ta.loc[c.name, "Cluster (ms)"],
                "n": len(y),
                "β (µV/year)": lr.slope * 12,
                "SE": lr.stderr * 12,
                "r": lr.rvalue,
                "p": lr.pvalue,
                "r within group": lr_g.rvalue,
                "n within group": int(gmask.sum()),
            }
        )
        subj.append(
            pd.DataFrame(
                {
                    "ID": ids,
                    "age_month": age,
                    "group": grp,
                    "cluster": f"{c.group} {c.region} {c.pair} {int(c.t_start)}-{int(c.t_end)}",
                    "mean_diff_uV": y,
                }
            )
        )
    tb = pd.DataFrame(rows)
    tb["pFDR"] = multipletests(tb.p, method="fdr_bh")[1]
    pd.concat(subj).to_csv(OUT / "hep_subject_cluster_means.csv", index=False)
    tb.to_csv(paths.tables / "Table_S3b_hep_age_regression.csv", index=False)
    if print_tables:
        with pd.option_context("display.width", 250):
            print(tb.round(3).to_string())

    record_run(
        paths,
        "age_effects",
        [paths.participants, paths.roi, paths.clusters],
        {
            "n_clusters": 5,
            "regression": "ordinary least squares against Age_month",
            "coefficient_units": "microvolts per year",
            "age_p_adjustment": "BH-FDR across five clusters",
            "within_group_correlations": "descriptive; selection-biased",
            "cluster_endpoints": "inclusive",
        },
    )
    return ta, tb


def main(argv=None):
    paths, check_only = parse_args(__doc__, argv)
    run_analysis(paths, check_only=check_only)


if __name__ == "__main__":
    main()
