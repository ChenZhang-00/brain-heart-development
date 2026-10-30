"""Per-electrode top-30% minus bottom-30% intensity contrasts for 148 children."""

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

    from config.transform_dbhi import transform
    from config.dbhi_config import GROUPS, HRV
    from analysis.inputs import epoch_mask, validate_model_cache

    CACHE, OUT = paths.cache, paths.intensity_results
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, info, ids, _ = validate_model_cache(paths)
    BANDS, FS = manifest["band_order"], manifest["config"]["fs_bhi"]
    EMOTIONS, DIRECTIONS = ["fear", "happy"], ["htb", "bth"]
    selection = pd.read_csv(OUT / "ra_selected_epochs.csv")

    groups, labels = [], None
    differences = {(emotion, direction): [] for emotion in EMOTIONS for direction in DIRECTIONS}
    for sid in ids:
        age = int(info.loc[sid, "Age_group"])
        groups.append(next(group for group, ages in GROUPS.items() if age in ages))
        with np.load(CACHE / f"{sid}_dbhi.npz", allow_pickle=False) as cache:
            if labels is None:
                labels = cache["labels"].tolist()
            assert cache["labels"].tolist() == labels
            for emotion in EMOTIONS:
                for direction in DIRECTIONS:
                    values = transform(cache[f"{emotion}__{direction}"], direction)
                    times = cache[f"{emotion}__t_{direction}"]
                    high = epoch_mask(times, selection, emotion, "top30", fs=FS)
                    low = epoch_mask(times, selection, emotion, "bottom30", fs=FS)
                    if np.any(high & low):
                        raise ValueError("High and low epochs overlap.")
                    contrast = np.nanmean(values[..., high], axis=-1) - np.nanmean(
                        values[..., low], axis=-1
                    )
                    differences[emotion, direction].append(contrast)
    differences = {key: np.stack(values) for key, values in differences.items()}
    assert all(x.shape == (148, 4, 2, 63) and np.isfinite(x).all() for x in differences.values())
    np.savez_compressed(
        OUT / "topo_subject_channel_diff.npz",
        ids=np.array(ids),
        groups=np.array(groups),
        labels=np.array(labels),
        bands=np.array(BANDS),
        hrv=np.array(HRV),
        **{f"{emotion}__{direction}": x for (emotion, direction), x in differences.items()},
    )
    print(pd.Series(groups).value_counts().to_dict())

    from sdg.pipeline import atomic_json, file_hash, fingerprint
    from config.transform_dbhi import params as transform_params

    atomic_json(
        OUT / "contrast_parameters.json",
        {
            "n": len(ids),
            "cache_fingerprint": fingerprint(manifest),
            "participants_sha256": file_hash(paths.participants),
            "epochs_sha256": file_hash(OUT / "ra_selected_epochs.csv"),
            "contrast": "top30 minus bottom30; fearful/happy clips only",
            **transform_params(),
        },
    )


if __name__ == "__main__":
    run(parse_paths(__doc__, ("participants", "cache")))
