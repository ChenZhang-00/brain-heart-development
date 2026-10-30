# Preprocessing

After [environment setup](../README.md#setup), open
[run_preprocessing.ipynb](run_preprocessing.ipynb) with **Python (dbh)**.

- Steps 1–2: choose one subject or a CSV batch. Press Enter for defaults; missing recordings are listed and skipped.
- Steps 3–4: clean EEG and compute band power (MATLAB).
- Steps 5–7: extract ECG, review R peaks and compute IBI/HRV (Python).

EEG and ECG can run independently after setup. Existing outputs: **1 = keep**,
**2 = replace**. Existing R-peak corrections skip the editor with option 1;
otherwise review them, press `q` to save, then close the editor.

Outputs are grouped by run type, then by signal:

```text
outputs/
  manuscript/      # supplied, read only: 148-child manuscript version
  demo/            # single-subject runs (default sub-001)
  batch/           # CSV batch runs
    eeg/
      cleaned/     # continuous EEG
      band_power/  # EEG band power
    ecg/
      extracted/   # extracted ECG
      r_peaks/     # reviewed R peaks
      hrv/         # IBI and LF/HF time series
```

`manuscript/` holds cleaned EEG, band power, reviewed R peaks and `ecg/hrv/ecg_ts.npz`
for all 148 children; the analysis notebooks read it by default and this notebook
never writes to it. HRV files: `ecg_ts.npz` for a batch; `ecg_ts_{ID}.npz` for one
subject. Each notebook run writes one log in `logs/`.

Analysis entries: [eliciting](../02_eliciting_effects/run_eliciting_effects.ipynb) →
[correlation](../03_correlation/run_correlation.ipynb) →
[HEP](../04_HEP/run_hep.ipynb) → [dBHI](../05_dBHI/run_dbhi.ipynb).
