"""Table S4: significant temporal clusters and duration summaries."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.dbhi_config import parse_paths


def run(paths):
    paths.supplementary.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    RES = paths.time_results
    # Table S4
    inc = pd.read_csv(RES / "included_subjects.csv", dtype={"ID": str})
    assert len(inc) == 148
    print("n per group:", inc.groupby("analysis_group").size().to_dict())
    frames = []
    for band in ["delta", "theta", "alpha", "beta"]:
        d = pd.read_csv(RES / f"dBHI_{band}_clusters.csv")
        d["band"] = band
        frames.append(d)
    cl = pd.concat(frames)
    cl["dur"] = cl.stop_s - cl.start_s
    sig = cl.loc[cl.significant.eq(True)].copy()
    print(
        "n significant clusters:",
        len(sig),
        " sign by direction:",
        sig.groupby("direction").t_mass.agg(lambda s: f"{(s>0).sum()}+/{(s<0).sum()}-").to_dict(),
    )
    print("n per group:", sig.groupby("group").n.unique().to_dict())
    sig["group"] = sig.group.str.replace(" yrs", "")
    summ = (
        sig.groupby(["direction", "hrv", "band", "region", "emotion", "group"])
        .agg(k=("dur", "size"), total_s=("dur", "sum"))
        .reset_index()
    )
    summ.to_csv(RES / "temporal_clusters_long.csv", index=False)
    cells = {}
    for _, r in summ.iterrows():
        cells[(r.direction, r.hrv, r.band, r.region, r.emotion, r.group)] = (
            f"{int(r.k)} ({r.total_s:.0f} s)"
        )
    rows = []
    for direction, dlab in [("htb", "Heart→Brain"), ("bth", "Brain→Heart")]:
        for hrv in ["LF", "HF"]:
            for band in ["delta", "theta", "alpha", "beta"]:
                for region in ["Frontal", "Central", "Posterior"]:
                    r = {"Direction": dlab, "HRV band": hrv, "EEG band": band, "Region": region}
                    for emo, el in [("fear", "Fearful"), ("happy", "Happy")]:
                        for g in ["5-6", "7-8", "9-10"]:
                            r[f"{el} {g}"] = cells.get((direction, hrv, band, region, emo, g), "—")
                    rows.append(r)
    wide = pd.DataFrame(rows)
    wide.to_csv(paths.supplementary / "Table_S4_temporal_clusters.csv", index=False)
    tot = (
        sig.groupby(["direction", "group"])
        .agg(k=("dur", "size"), total_s=("dur", "sum"))
        .reset_index()
    )
    print(tot.to_string())
    tot.to_csv(RES / "temporal_cluster_totals.csv", index=False)
    sig_out = sig[
        [
            "emotion",
            "direction",
            "hrv",
            "band",
            "region",
            "group",
            "n",
            "start_s",
            "stop_s",
            "dur",
            "t_mass",
            "p",
        ]
    ].sort_values(["direction", "hrv", "band", "region", "emotion", "group", "start_s"])
    sig_out.to_csv(RES / "temporal_clusters_significant.csv", index=False)


if __name__ == "__main__":
    run(parse_paths(__doc__, ()))
