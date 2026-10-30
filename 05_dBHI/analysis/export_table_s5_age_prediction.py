"""Table S5: main and centred age-prediction models."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.dbhi_config import parse_paths


def run(paths):
    paths.supplementary.mkdir(parents=True, exist_ok=True)
    import json
    from pathlib import Path

    import pandas as pd

    pd.set_option("display.width", 250)
    fmtp = lambda x: "<.001" if x < 0.001 else f"{x:.3f}"

    # Table S5: age prediction, n = 148
    AP = paths.age_results
    mr = pd.read_csv(AP / "model_results.csv").set_index("model")
    hp = json.load(open(AP / "haufe_parameters.json"))["models"]
    assert json.load(open(AP / "haufe_parameters.json"))["n"] == 148
    names = {
        "bth fear": ("Brain→Heart", "Fearful"),
        "bth happy": ("Brain→Heart", "Happy"),
        "htb fear": ("Heart→Brain", "Fearful"),
        "htb happy": ("Heart→Brain", "Happy"),
    }
    t5 = []
    for m, (d, e) in names.items():
        assert abs(hp[m]["r_cv_raw"] - mr.loc[m, "r_oos"]) < 1e-9
        t5.append(
            {
                "Direction": d,
                "Movie": e,
                "Features": int(mr.loc[m, "n_features"]),
                "r": f"{mr.loc[m, 'r_oos']:.2f}",
                "p": fmtp(mr.loc[m, "perm_p"]),
                "MAE (years)": f"{mr.loc[m, 'mae_yr']:.2f}",
                "r (centred)": f"{hp[m]['r_cv_centred']:.2f}",
                "p (centred)": fmtp(hp[m]["perm_p_centred"]),
                "MAE (centred, years)": f"{hp[m]['mae_centred_yr']:.2f}",
                "Features with CI excluding 0": int(hp[m]["n_ci_excl0"]),
            }
        )
    T5 = pd.DataFrame(t5)
    T5.to_csv(paths.supplementary / "Table_S5_age_prediction.csv", index=False)
    print(T5.to_string())


if __name__ == "__main__":
    run(parse_paths(__doc__, ()))
