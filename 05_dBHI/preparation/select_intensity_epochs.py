"""Select 48 high- and 48 low-intensity one-second epochs per emotional movie."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.dbhi_config import parse_paths, shown


def run(paths):
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    OUT = paths.intensity_results
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(paths.ratings)
    WINDOW, SHIFT, N_SELECT = 15, 7.5, 48
    EMOTIONS = ["fear", "happy"]
    starts = np.arange(3, 164)
    if raw.Time.duplicated().any():
        raise ValueError("Duplicate intensity timestamps.")
    ra = pd.DataFrame({"Time": starts})
    for emotion in EMOTIONS:
        column = emotion.capitalize() + "_Intensity"
        if not np.isfinite(raw[column]).all():
            raise ValueError(f"{emotion}: missing intensity values.")
        values = []
        for start in starts:
            window = raw.loc[(raw.Time >= start) & (raw.Time < start + WINDOW)]
            if not np.array_equal(window.Time, np.arange(start, start + WINDOW)):
                raise ValueError(f"Incomplete rating window at {start} s.")
            values.append(window[column].mean())
        ra[column] = values
    ra.to_csv(OUT / "smoothed_intensity.csv", index=False)

    rows = []
    for emotion in EMOTIONS:
        column = emotion.capitalize() + "_Intensity"
        for level, ascending in [("top30", False), ("bottom30", True)]:
            selected = ra.sort_values([column, "Time"], ascending=[ascending, True]).head(N_SELECT)
            for row in selected.sort_values("Time").to_dict("records"):
                rows.append(
                    dict(
                        emotion=emotion,
                        level=level,
                        ra_window_start=int(row["Time"]),
                        dbhi_epoch_start_s=row["Time"] + SHIFT,
                        dbhi_epoch_stop_s=row["Time"] + SHIFT + 1,
                        smoothed_ra_score=row[column],
                    )
                )
    selection = pd.DataFrame(rows)
    selection.to_csv(OUT / "ra_selected_epochs.csv", index=False)
    parameters = dict(
        window_seconds=WINDOW,
        start_to_centre_shift_seconds=SHIFT,
        n_seconds_per_tail=N_SELECT,
        tie_break="earlier time",
        minimum_run_filter_applied=False,
    )
    (OUT / "analysis_parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")

    print("Selection:", shown(OUT))


if __name__ == "__main__":
    run(parse_paths(__doc__, ("ratings",)))
