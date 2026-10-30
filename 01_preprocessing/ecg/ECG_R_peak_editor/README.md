# ECG R-peak correction

The preprocessing notebook offers **1 = skip existing corrections (default)** or
**2 = review again**. With option 1, `launch_editor.py` opens only for missing or
empty files; with option 2, it also opens for subjects with saved corrections.
Both folders and the first ID to review are filled in. The launcher opens
`edit_r_peaks.py`; existing files change only when you save. For multiple subjects,
enter the next ID printed by the notebook after saving and returning to the launcher.
The notebook checks that saved R peaks match the ECG before computing HRV.

An Apple Silicon Mac application is available in [distribution/](distribution/README.md).
It includes Python and the editor dependencies. Extract ECG first using the command
below, then open the app to select input/output folders and enter `sub-001`.

Run from the repository root with the `dbh` environment active:

```bash
python 01_preprocessing/ecg/ECG_R_peak_editor/extract_ecg.py
python 01_preprocessing/ecg/ECG_R_peak_editor/edit_r_peaks.py sub-001
```

The extractor reads `00_data/raw_data/` recursively and writes `01_preprocessing/outputs/demo/ecg/extracted/`.
Override these with `--input-root` and `--output-root`. It extracts the BrainVision
channel labelled `65` and selects the last complete 150–260 s take of each movie,
matching the EEG marker rule. The editor inverts this channel for detection/display,
then writes `01_preprocessing/outputs/demo/ecg/r_peaks/{ID}_movie_rpeaks.npz`. Use `--raw-dir` and
`--out-dir` to override those locations. A desktop display and Tk are required.

| Control | Action |
|---|---|
| Left click / `a` | Add a peak, snapping to the local maximum |
| Right click / `d` | Delete a peak |
| `e` | Redetect within the visible window |
| `Shift+C` | Clear peaks in the visible window |
| `u` | Undo |
| `s` / `q` | Save / save and quit |
| Arrow keys | Move through the recording |
| `+` / `-` | Time zoom |
| `w` / `z` | Amplitude zoom |

Saved `r_peaks` and `event_sample` are zero-based sample indices; `fs` is Hz.
The files also retain `event_desc` (`S1`–`S6`), recording duration and manual edit
counts. Review every beat before using these files downstream.
