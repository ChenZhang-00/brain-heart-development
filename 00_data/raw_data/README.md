# Real recording example

`sub-001_movie` is a complete recording from one study participant, not a simulated
signal. The three BrainVision files must remain together:

| File | Content |
|---|---|
| `sub-001_movie.vhdr` | Channel information and relative companion-file references |
| `sub-001_movie.vmrk` | Movie markers and their original sample positions |
| `sub-001_movie.eeg` | Original signal samples, unchanged |

The recording contains 65 channels at 500 Hz, 444,780 samples per channel and
889.56 seconds of data. EEG preprocessing retains 63 EEG channels; the ECG channel
has the BrainVision label `65`. All three complete movie segments are present.

The example uses the recoded ID `sub-001`. The absolute recording date, local
workstation paths and equipment serial numbers have been removed from the text
metadata. Signal samples, channel information, electrode coordinates and event
sample positions are retained. Recoded participant information is included in
[participants.csv](../participants.csv); the original ID mapping is not included.

## Read and preprocess

Start the signal-processing sequence with
[run_preprocessing.ipynb](../../01_preprocessing/run_preprocessing.ipynb) with the
**Python (dbh)** kernel. In step 1, choose **1** (single subject), then press Enter
to accept `sub-001` and the default raw recording folder. After setup, run the EEG
section (cleaning and band power), the ECG section (R-peak review and HRV), or both.
Analysis entries: [eliciting](../../02_eliciting_effects/run_eliciting_effects.ipynb) →
[correlation](../../03_correlation/run_correlation.ipynb) →
[HEP](../../04_HEP/run_hep.ipynb) → [dBHI](../../05_dBHI/run_dbhi.ipynb).

For a batch, choose **2** and use the included `00_data/participants.csv` or another CSV
with an `ID` column, then confirm the matching raw recording folder. The included
table lists 148 participants. The notebook prints `Missing` and skips IDs without
all three recording files, then processes the available subjects in table order.
With the included files, batch mode processes `sub-001` and skips the other 147.
Single-subject HRV is saved as `01_preprocessing/outputs/demo/ecg/hrv/ecg_ts_sub-001.npz`; batch HRV is saved
as `01_preprocessing/outputs/batch/ecg/hrv/ecg_ts.npz`. The supplied 148-child version is
`01_preprocessing/outputs/manuscript/ecg/hrv/ecg_ts.npz`.

From the repository root, with the `dbh` Python environment active:

```bash
python 01_preprocessing/ecg/ECG_R_peak_editor/extract_ecg.py
python 01_preprocessing/ecg/ECG_R_peak_editor/edit_r_peaks.py sub-001
```

The extractor creates `01_preprocessing/outputs/demo/ecg/extracted/sub-001_movie_ecg.npz`. The editor requires a
desktop display; verify and save R peaks before using them downstream.

To inspect the recording directly in Python:

```python
import mne

raw = mne.io.read_raw_brainvision(
    "00_data/raw_data/sub-001_movie.vhdr", preload=False
)
print(raw)
```

With MATLAB, EEGLAB and FieldTrip configured as described in the root README,
run from the repository root:

```matlab
addpath('01_preprocessing/config', '01_preprocessing/eeg')
preprocess_eeg('sub-001')
compute_eeg_band_power('sub-001')
```

The guided single-participant workflow does not require `participants.csv`.
The included table supports demographic summaries. Signal-based group analyses
require recordings from the appropriate participant set. This example alone does
not reproduce the paper's group results.

## Download and verify the recording

The binary file is 115,642,800 bytes (110.29 MiB) and uses Git LFS. After cloning,
install Git LFS and run `git lfs pull` from the repository root to download the
signal file before preprocessing. Keep the accompanying `.vhdr` and `.vmrk` files
in the same folder.

SHA-256 of `sub-001_movie.eeg`:

```text
aad6490e5e5942d0ce5c74e6161ac404847b25852fd6a011d6e0d047c82b56a4
```
