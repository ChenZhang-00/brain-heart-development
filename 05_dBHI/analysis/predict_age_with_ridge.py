"""Four ridge age models from 504 top-intensity features; observed CV repeats=5, permutation repeats=1 (see README)."""

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

    from config.transform_dbhi import params as transform_params, transform
    from analysis.age_models import N_PERM, OBSERVED_REPEATS, PERMUTATION_REPEATS, SEED, oos
    from analysis.inputs import epoch_mask, validate_model_cache
    from sdg.pipeline import fingerprint, file_hash

    OUT, CACHE = paths.age_results, paths.cache
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, info, cache_ids, _ = validate_model_cache(paths)
    BANDS = manifest["band_order"]
    GMAP = {5: "5-6 yrs", 6: "5-6 yrs", 7: "7-8 yrs", 8: "7-8 yrs", 9: "9-10 yrs", 10: "9-10 yrs"}
    with np.load(CACHE / f"{cache_ids[0]}_dbhi.npz", allow_pickle=False) as first:
        LABELS = first["labels"].tolist()
    subj = pd.DataFrame(
        {
            "ID": cache_ids,
            "group": [GMAP[int(info.loc[s, "Age_group"])] for s in cache_ids],
            "age_month": info.loc[cache_ids, "Age_month"].to_numpy(),
        }
    )
    subj.to_csv(OUT / "subjects_n148.csv", index=False)
    ids = subj.ID.tolist()
    age = subj.age_month.to_numpy(float)
    grp = subj.group.to_numpy()
    ra = pd.read_csv(paths.intensity_results / "ra_selected_epochs.csv")
    feat = [
        f"{b}|{h}|{ch}" for b in BANDS for h in ["LF", "HF"] for ch in LABELS
    ]  # order matches array.ravel() of (band, hrv, channel)

    models = {}
    for d in ["bth", "htb"]:
        for emo in ["fear", "happy"]:
            rows = []
            for sid in ids:
                with np.load(f"{CACHE}/{sid}_dbhi.npz", allow_pickle=False) as z:
                    assert list(z["labels"]) == LABELS and list(z["bands"]) == BANDS
                    t = z[f"{emo}__t_{d}"]
                    top = epoch_mask(t, ra, emo, "top30", fs=manifest["config"]["fs_bhi"])
                    x = transform(z[f"{emo}__{d}"][..., top], d).mean(
                        -1
                    )  # (4,2,63)
                    rows.append(x.ravel())
            models[f"{d} {emo}"] = np.array(rows)
            assert (
                models[f"{d} {emo}"].shape == (148, 504) and np.isfinite(models[f"{d} {emo}"]).all()
            )
    np.savez(
        OUT / "features_electrode_top30.npz",
        ids=np.array(ids),
        age=age,
        feature_names=np.array(feat),
        **{k.replace(" ", "_"): v for k, v in models.items()},
    )

    rng = np.random.default_rng(SEED)
    rows = []
    nulls = {}
    preds = {}
    for name, X in models.items():
        r, p = oos(X, age)
        null = np.array(
            [
                oos(X, rng.permutation(age), seed=k, rep=PERMUTATION_REPEATS)[0]
                for k in range(N_PERM)
            ]
        )
        rows.append(
            dict(
                model=name,
                n_features=X.shape[1],
                r_oos=r,
                perm_p=(1 + (null >= r).sum()) / (N_PERM + 1),
                mae_yr=np.abs(p - age).mean() / 12,
            )
        )
        nulls[name] = null
        preds[name] = p
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "model_results.csv", index=False)
    pd.DataFrame(
        {
            "ID": ids,
            "group": grp,
            "age_month": age,
            **{f"pred_{k.replace(' ','_')}": v for k, v in preds.items()},
        }
    ).to_csv(OUT / "oos_predictions.csv", index=False)
    np.savez(OUT / "null_distributions.npz", **{k.replace(" ", "_"): v for k, v in nulls.items()})
    json.dump(
        dict(
            n_perm=N_PERM,
            cache_fingerprint=fingerprint(manifest),
            epochs_sha256=file_hash(paths.intensity_results / "ra_selected_epochs.csv"),
            participants_sha256=file_hash(paths.participants),
            seed=SEED,
            observed_cv_repeats=OBSERVED_REPEATS,
            permutation_cv_repeats=PERMUTATION_REPEATS,
            cv="10-fold x 5 repeats, RidgeCV alphas logspace(-2,4,25) inner LOO",
            n_features=len(feat),
            feature_order="band | hrv | channel (array.ravel of band x hrv x channel)",
            feature_definition="mean over top-30% intensity epochs (ra_selected_epochs.csv) at each electrode; per-electrode log10 (htb floored); no spatial averaging",
            **{
                **transform_params(),
                "transform_order": "per electrode: (htb floor) -> log10 -> temporal aggregation; no spatial averaging",
            },
            bands=BANDS,
            channel_order=LABELS,
            n=len(ids),
        ),
        open(OUT / "analysis_parameters.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    run(parse_paths(__doc__, ("participants", "cache")))
