"""Child-centred ridge age models and fixed-penalty bootstrap Haufe patterns (Fig. 4, Table S5)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.dbhi_config import parse_paths


def run(paths):
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge, RidgeCV
    from sklearn.preprocessing import StandardScaler

    from config.transform_dbhi import params as transform_params
    from analysis.age_models import ALPHAS, N_PERM, OBSERVED_REPEATS, PERMUTATION_REPEATS, SEED, oos

    OUT = paths.age_results
    N_BOOT = 1000
    main_results = pd.read_csv(OUT / "model_results.csv").set_index("model")
    with np.load(OUT / "features_electrode_top30.npz", allow_pickle=False) as archive:
        F = {key: archive[key] for key in archive.files}
    if F["age"].shape != (148,) or not np.isfinite(F["age"]).all():
        raise ValueError("Expected 148 finite ages.")
    if any(
        F[key].shape != (148, 504) or not np.isfinite(F[key]).all()
        for key in ("bth_fear", "bth_happy", "htb_fear", "htb_happy")
    ):
        raise ValueError("Expected four finite 148-by-504 feature matrices.")
    age = F["age"]
    feat = list(F["feature_names"])
    models = {k.replace("_", " "): F[k] for k in ["bth_fear", "bth_happy", "htb_fear", "htb_happy"]}

    def haufe_full(Xin, y):
        """Fit the full-sample ridge model and return its Haufe pattern."""
        Xs = StandardScaler().fit_transform(Xin)
        m = RidgeCV(alphas=ALPHAS).fit(Xs, y)
        yh = Xs @ m.coef_
        return (Xs - Xs.mean(0)).T @ (yh - yh.mean()) / len(y) / yh.var(), m.coef_, m.alpha_

    def pattern_fixed(Xin, y, l):
        """Bootstrap pattern with the full-sample penalty held fixed."""
        Xs = StandardScaler().fit_transform(Xin)
        wb = Ridge(alpha=l).fit(Xs, y).coef_
        yh = Xs @ wb
        return (Xs - Xs.mean(0)).T @ (yh - yh.mean()) / len(y) / yh.var()

    summary = {}
    for name, X in models.items():
        rowmean = X.mean(1)
        Xc = X - rowmean[:, None]
        rng = np.random.default_rng(SEED)
        r_raw = main_results.loc[name, "r_oos"]
        r_c, pred_c = oos(Xc, age)
        null = np.array(
            [
                oos(Xc, rng.permutation(age), seed=k, rep=PERMUTATION_REPEATS)[0]
                for k in range(N_PERM)
            ]
        )
        p_c = (1 + (null >= r_c).sum()) / (N_PERM + 1)
        A, w, lam = haufe_full(Xc, age)
        brng = np.random.default_rng(SEED + 1)
        B = np.zeros((N_BOOT, X.shape[1]))
        for b in range(N_BOOT):
            idx = brng.integers(0, len(age), len(age))
            B[b] = pattern_fixed(Xc[idx], age[idx], lam)
        lo, hi = np.percentile(B, [2.5, 97.5], 0)
        res = pd.DataFrame(
            dict(feature=feat, pattern=A, ci_lo=lo, ci_hi=hi, ci_excl0=(lo > 0) | (hi < 0))
        )
        tag = name.replace(" ", "_")
        res.to_csv(OUT / f"haufe_patterns_{tag}.csv", index=False)
        np.savez(OUT / f"haufe_bootstrap_{tag}.npz", B_fixed=B, null_centred=null, A=A, w=w)
        summary[name] = dict(
            r_cv_raw=float(r_raw),
            r_cv_centred=float(r_c),
            perm_p_centred=float(p_c),
            mae_centred_yr=float(np.abs(pred_c - age).mean() / 12),
            lambda_full=float(lam),
            n_ci_excl0=int(res.ci_excl0.sum()),
        )
        print(name, json.dumps(summary[name]), flush=True)
    json.dump(
        dict(
            features="electrode-level top-30% means, 504 per model",
            n=len(age),
            n_perm=N_PERM,
            n_boot=N_BOOT,
            seed=SEED,
            observed_cv_repeats=OBSERVED_REPEATS,
            permutation_cv_repeats=PERMUTATION_REPEATS,
            bootstrap="lambda fixed at full-sample value",
            **{
                **transform_params(),
                "transform_order": "per electrode: (htb floor) -> log10 -> temporal aggregation; no spatial averaging",
            },
            models=summary,
        ),
        open(OUT / "haufe_parameters.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    run(parse_paths(__doc__, ()))
