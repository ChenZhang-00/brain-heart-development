"""Two-sided spatial cluster tests over 96 intensity-contrast maps."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.dbhi_config import parse_paths, shown


def run(paths):
    import hashlib
    import json
    import platform
    from pathlib import Path

    import matplotlib
    import numpy as np
    import pandas as pd
    import scipy
    from scipy import sparse, stats

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    from mne.viz.topomap import _get_pos_outlines

    from config.dbhi_config import DBHI as HERE, GROUPS as GROUP_AGES

    BASE, OUT = paths.intensity_results, paths.spatial_results
    OUT.mkdir(parents=True, exist_ok=True)
    FIG = paths.spatial_results
    FIG.mkdir(parents=True, exist_ok=True)
    SOURCE = BASE / "topo_subject_channel_diff.npz"
    ALPHA, N_PERM, SEED = 0.05, 10000, 20260928
    EMOTIONS, DIRS = ["fear", "happy"], ["bth", "htb"]
    GROUPS = list(GROUP_AGES)
    with np.load(SOURCE, allow_pickle=False) as z:
        ids, groups, labels = z["ids"], z["groups"], z["labels"].tolist()
        BANDS, HRV = z["bands"].tolist(), z["hrv"].tolist()
        diff = {(e, d): z[f"{e}__{d}"] for e in EMOTIONS for d in DIRS}
    assert len(ids) == len(set(ids)) == 148 and len(labels) == 63
    assert sorted((groups == g).sum() for g in GROUPS) == [49, 49, 50]
    assert all(x.shape == (148, 4, 2, 63) and np.isfinite(x).all() for x in diff.values())
    minfo = mne.create_info(labels, sfreq=1.0, ch_types="eeg")
    minfo.set_montage("standard_1020")
    adjacency, names = mne.channels.find_ch_adjacency(minfo, ch_type="eeg")
    assert names == labels
    removed_edges = [("F9", "P9"), ("F10", "P10")]
    adjacency = adjacency.tolil()
    for a, b in removed_edges:
        i, j = labels.index(a), labels.index(b)
        adjacency[i, j] = adjacency[j, i] = 0
    adjacency = adjacency.tocsr()
    adjacency.eliminate_zeros()
    assert (adjacency != adjacency.T).nnz == 0
    assert sparse.csgraph.connected_components(adjacency, directed=False)[0] == 1
    sparse.save_npz(OUT / "channel_adjacency.npz", adjacency)
    positions, _ = _get_pos_outlines(minfo, picks=None, sphere=None)
    r, c = sparse.triu(adjacency, k=1).nonzero()
    pd.DataFrame({"channel_1": np.array(labels)[r], "channel_2": np.array(labels)[c]}).to_csv(
        OUT / "channel_adjacency_edges.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(9, 9))
    for i, j in zip(r, c):
        ax.plot(positions[[i, j], 0], positions[[i, j], 1], color="0.75", lw=0.7)
    ax.scatter(*positions.T, s=14)
    for label, xy in zip(labels, positions):
        ax.annotate(label, xy, xytext=(2, 3), textcoords="offset points", fontsize=9)
    ax.set(aspect="equal", title="Channel adjacency: standard_1020 / Delaunay")
    ax.axis("off")
    fig.savefig(FIG / "channel_adjacency.png", dpi=160)
    plt.close(fig)
    print("Loaded 148 participants; 50/49/49 per group; 63 channels;", len(r), "adjacency edges.")

    channel_rows, cluster_rows, map_rows, nulls = [], [], [], {}
    for (emotion, direction), arr in diff.items():
        for group in GROUPS:
            for bi, band in enumerate(BANDS):
                for hi, hrv in enumerate(HRV):
                    x = arr[groups == group, bi, hi, :]
                    threshold = float(stats.t.ppf(1 - ALPHA / 2, len(x) - 1))
                    t, p = stats.ttest_1samp(x, 0, axis=0)
                    clusters, cluster_p, h0 = [], np.array([]), np.array([])
                    if np.any(np.abs(t) > threshold):
                        observed, clusters, cluster_p, h0 = (
                            mne.stats.permutation_cluster_1samp_test(
                                x,
                                threshold=threshold,
                                tail=0,
                                adjacency=adjacency,
                                n_permutations=N_PERM,
                                seed=SEED,
                                n_jobs=1,
                                t_power=1,
                                out_type="indices",
                                verbose="ERROR",
                            )
                        )
                        np.testing.assert_allclose(observed, t, rtol=1e-12, atol=1e-12)
                        t = observed
                    map_id = f"map_{len(map_rows):03d}"
                    meta = dict(
                        map_id=map_id,
                        emotion=emotion,
                        direction=direction,
                        group=group,
                        band=band,
                        hrv=hrv,
                        n=len(x),
                    )
                    membership = np.zeros(len(labels), dtype=int)
                    channel_p = np.ones(len(labels))
                    for ci, (cluster, probability) in enumerate(zip(clusters, cluster_p), 1):
                        idx = cluster[0]
                        mass = float(t[idx].sum())
                        assert np.all(np.sign(t[idx]) == np.sign(mass))
                        assert np.all(np.abs(t[idx]) > threshold)
                        assert (
                            sparse.csgraph.connected_components(
                                adjacency[idx][:, idx], directed=False
                            )[0]
                            == 1
                        )
                        tolerance = 1e-12 * max(1.0, abs(mass))
                        lower = np.mean(np.abs(h0) >= abs(mass) + tolerance)
                        upper = np.mean(np.abs(h0) >= abs(mass) - tolerance)
                        assert lower - 1e-12 <= probability <= upper + 1e-12
                        membership[idx], channel_p[idx] = ci, probability
                        cluster_rows.append(
                            dict(
                                meta,
                                cluster_id=ci,
                                sign="positive" if mass > 0 else "negative",
                                n_channels=len(idx),
                                t_mass=mass,
                                p_cluster=float(probability),
                                significant=bool(probability < ALPHA),
                                channels=";".join(np.array(labels)[idx]),
                            )
                        )
                    for ci, channel in enumerate(labels):
                        channel_rows.append(
                            dict(
                                meta,
                                channel=channel,
                                mean=float(x[:, ci].mean()),
                                t=float(t[ci]),
                                p_uncorrected=float(p[ci]),
                                cluster_id=int(membership[ci]),
                                p_cluster=float(channel_p[ci]),
                                cluster_significant=bool(channel_p[ci] < ALPHA),
                            )
                        )
                    map_rows.append(
                        dict(
                            meta,
                            threshold_t=threshold,
                            n_clusters=len(clusters),
                            n_significant_clusters=int((cluster_p < ALPHA).sum()),
                            n_significant_channels=int((channel_p < ALPHA).sum()),
                            min_cluster_p=float(cluster_p.min()) if len(cluster_p) else 1.0,
                            n_null_configurations=len(h0),
                        )
                    )
                    nulls[map_id] = h0
                    print(
                        f"{len(map_rows):02d}/96 {emotion} {direction} {group} {band}-{hrv}: "
                        f"{(cluster_p < ALPHA).sum()} significant clusters",
                        flush=True,
                    )

    res = pd.DataFrame(channel_rows)
    cluster_columns = [
        "map_id",
        "emotion",
        "direction",
        "group",
        "band",
        "hrv",
        "n",
        "cluster_id",
        "sign",
        "n_channels",
        "t_mass",
        "p_cluster",
        "significant",
        "channels",
    ]
    clusters_df, summary = pd.DataFrame(cluster_rows, columns=cluster_columns), pd.DataFrame(
        map_rows
    )
    res.to_csv(OUT / "topo_channel_tests.csv", index=False)
    clusters_df.to_csv(OUT / "topo_clusters.csv", index=False)
    summary.to_csv(OUT / "topo_map_summary.csv", index=False)
    np.savez_compressed(OUT / "cluster_null_distributions.npz", **nulls)
    parameters = dict(
        source=shown(SOURCE),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        test="MNE two-sided one-sample spatial cluster permutation on participant differences",
        contrast="top30 minus bottom30",
        log10_bth=True,
        log10_htb=True,
        htb_floor=json.load(open(HERE / "sdg" / "htb_floor.json"))["htb_floor"],
        transform_order="per electrode before spatial/temporal aggregation",
        n_permutations=N_PERM,
        seed=SEED,
        cluster_forming_p=ALPHA,
        cluster_alpha=ALPHA,
        statistic="sum of t values; maximum absolute cluster mass over both signs",
        correction="cluster-level FWER within each map; no correction across 96 maps",
        minimum_cluster_size=1,
        adjacency="standard_1020 Delaunay with two hull shortcuts removed",
        removed_edges=removed_edges,
        channel_order=labels,
        groups={g: int((groups == g).sum()) for g in GROUPS},
        versions=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            mne=mne.__version__,
        ),
        null_note="MNE includes the observed configuration; maps without clusters skip permutations",
    )
    (OUT / "analysis_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    print(summary[summary.n_significant_clusters > 0].to_string(index=False))


if __name__ == "__main__":
    run(parse_paths(__doc__, ()))
