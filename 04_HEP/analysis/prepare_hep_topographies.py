"""Prepare four descriptive scalp maps from the five reported HEP temporal clusters."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import numpy as np
import pandas as pd

from importlib import import_module
_hep_config = import_module("04_HEP.config.hep_config")
GROUPS = _hep_config.GROUPS
REGIONS = _hep_config.REGIONS
make_paths = _hep_config.make_paths
parse_args = _hep_config.parse_args
_inputs = import_module("04_HEP.analysis.inputs")
load_channels = _inputs.load_channels
load_clusters = _inputs.load_clusters
record_run = _inputs.record_run
sha256 = _inputs.sha256


def run_analysis(paths=None, *, check_only=False, print_tables=True):
    paths = paths or make_paths()
    times, pos, ids, labels, signals, groups, roi_masks, export = load_channels(paths)
    clusters = load_clusters(paths).query("p < 0.05").copy()
    assert clusters["test"].eq("pairwise").all()
    clusters["group"] = pd.Categorical(clusters["group"], GROUPS, ordered=True)
    clusters["region"] = pd.Categorical(clusters["region"], REGIONS, ordered=True)
    clusters = clusters.sort_values(["group", "region", "pair", "t_start"]).reset_index(drop=True)
    if check_only:
        print("Topography source inputs validated.")
        return
    OUT = paths.topographies
    OUT.mkdir(parents=True, exist_ok=True)
    clusters = clusters.copy()
    clusters["source_cluster_id"] = np.arange(1, len(clusters) + 1)
    frontal = clusters.loc[
        (clusters.group == "5-6")
        & (clusters.region == "Frontal")
        & (clusters.pair == "happy - neutral")
    ].sort_values("t_start")
    assert frontal[["t_start", "t_end"]].to_numpy().tolist() == [[198.0, 300.0], [300.0, 412.0]]
    bundles = [frontal] + [clusters.loc[[i]] for i in clusters.index if i not in frontal.index]
    maps, channel_tables, subject_tables = [], [], []
    for map_id, members in enumerate(bundles, 1):
        first = members.iloc[0]
        group, region, pair = str(first.group), str(first.region), str(first.pair)
        assert (
            members.group.eq(group).all()
            and members.region.eq(region).all()
            and members.pair.eq(pair).all()
        )
        a, b = pair.split(" - ")
        subjects = groups == group
        window = np.zeros(len(times), dtype=bool)
        for row in members.itertuples():
            assert row.t_start in times and row.t_end in times
            window |= (times >= row.t_start) & (times <= row.t_end)
        start, end = float(members.t_start.min()), float(members.t_end.max())
        np.testing.assert_array_equal(window, (times >= start) & (times <= end))
        mean_a = signals[a][subjects][:, window].mean(axis=1, dtype=float)
        mean_b = signals[b][subjects][:, window].mean(axis=1, dtype=float)
        difference = mean_a - mean_b
        if len(members) == 2:
            assert int(window.sum()) == 108
            full = signals[a][subjects].astype(float) - signals[b][subjects].astype(float)
            early, late = (times >= 198) & (times <= 300), (times >= 300) & (times <= 412)
            independent = (
                full[:, early].sum(axis=1)
                + full[:, late].sum(axis=1)
                - full[:, times == 300].sum(axis=1)
            ) / window.sum()
            np.testing.assert_allclose(difference, independent, atol=1e-12, rtol=1e-12)
        values = difference.mean(axis=0)
        details = dict(
            cluster_id=map_id,
            group=group,
            region=region,
            pair=pair,
            source_cluster_ids=";".join(map(str, members.source_cluster_id.tolist())),
            source_cluster_count=len(members),
            source_window=";".join(members.window.astype(str)),
            source_time_ranges=";".join(f"{r.t_start:g}-{r.t_end:g}" for r in members.itertuples()),
            source_cluster_p_values=";".join(f"{p:.17g}" for p in members.p),
            t_start=start,
            t_end=end,
            cluster_p=float(first.p) if len(members) == 1 else None,
            n_subjects=int(subjects.sum()),
            n_time_samples=int(window.sum()),
        )
        maps.append(dict(**details, values=values))
        channel_tables.append(
            pd.DataFrame(
                {
                    **details,
                    "channel": labels,
                    "in_roi": roi_masks[region],
                    "condition_a": a,
                    "condition_b": b,
                    "mean_a_uV": mean_a.mean(axis=0),
                    "mean_b_uV": mean_b.mean(axis=0),
                    "difference_uV": values,
                }
            )
        )
        subject_tables.append(
            pd.DataFrame(
                {
                    **details,
                    "ID": np.repeat(ids[subjects], len(labels)),
                    "channel": np.tile(labels, subjects.sum()),
                    "in_roi": np.tile(roi_masks[region], subjects.sum()),
                    "condition_a": a,
                    "condition_b": b,
                    "mean_a_uV": mean_a.ravel(),
                    "mean_b_uV": mean_b.ravel(),
                    "difference_uV": difference.ravel(),
                }
            )
        )
    channel_means = pd.concat(channel_tables, ignore_index=True)
    subject_means = pd.concat(subject_tables, ignore_index=True)
    assert len(maps) == 4 and len(channel_means) == 252 and len(subject_means) == 12537
    channel_means.to_csv(OUT / "temporal_cluster_channel_means_final.csv", index=False)
    subject_means.to_csv(OUT / "temporal_cluster_subject_channel_means_final.csv", index=False)
    details = {
        "maps": [{k: v for k, v in item.items() if k != "values"} for item in maps],
        "source_sha256": {
            str(paths.channel_dir / "chan_means.npz"): export["cache_sha256"],
            str(paths.clusters): sha256(paths.clusters),
        },
    }
    np.savez_compressed(
        OUT / "temporal_cluster_map_inputs.npz",
        pos=pos,
        labels=labels,
        values=np.stack([item["values"] for item in maps]),
        metadata_json=np.array(json.dumps(details)),
    )
    record_run(
        paths,
        "topographies",
        [paths.participants, paths.roi, paths.clusters, paths.channel_dir / "chan_means.npz"],
        {
            "new_statistical_tests": False,
            "n_display_maps": 4,
            "frontal_union_ms": [198, 412],
            "frontal_union_unique_samples": 108,
            "shared_300ms_sample_counted_once": True,
        },
    )
    if print_tables:
        print(
            channel_means.drop_duplicates("cluster_id")[
                ["group", "region", "pair", "t_start", "t_end", "source_cluster_p_values"]
            ].to_string(index=False)
        )
    return channel_means, subject_means


def main(argv=None):
    paths, check_only = parse_args(__doc__, argv)
    run_analysis(paths, check_only=check_only)


if __name__ == "__main__":
    main()
