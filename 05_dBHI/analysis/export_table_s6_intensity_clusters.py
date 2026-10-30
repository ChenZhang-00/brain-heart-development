"""Table S6: significant spatial intensity-contrast clusters."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.dbhi_config import parse_paths


def run(paths):
    paths.supplementary.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    # Table S6
    tc = pd.read_csv(paths.intensity_results / "topo_cluster" / "topo_clusters.csv")
    s = tc.loc[tc.significant.eq(True)].copy()
    s["Direction"] = s.direction.map({"bth": "Brain→Heart", "htb": "Heart→Brain"})
    s["Movie"] = s.emotion.map({"fear": "Fearful", "happy": "Happy"})
    s["Age group"] = s.group.str.replace(" yrs", "")
    s["Effect"] = s.sign.map({"positive": "high > low", "negative": "high < low"})
    s["Cluster p"] = s.p_cluster.map(lambda x: "<.001" if x < 0.001 else f"{x:.3f}")
    t6 = s[
        [
            "Movie",
            "Direction",
            "Age group",
            "n",
            "band",
            "hrv",
            "n_channels",
            "Effect",
            "t_mass",
            "Cluster p",
        ]
    ].rename(
        columns={
            "band": "EEG band",
            "hrv": "HRV band",
            "n_channels": "Electrodes",
            "t_mass": "Cluster mass (sum t)",
        }
    )
    t6["Cluster mass (sum t)"] = pd.to_numeric(t6["Cluster mass (sum t)"]).round(1)
    t6 = t6.sort_values(["Movie", "Direction", "Age group", "EEG band", "HRV band"])
    t6.to_csv(paths.supplementary / "Table_S6_intensity_clusters.csv", index=False)
    pd.set_option("display.width", 250)
    print(t6.to_string())
    print("maps tested:", tc.map_id.nunique(), " significant clusters:", len(s))


if __name__ == "__main__":
    run(parse_paths(__doc__, ()))
