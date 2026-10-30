# Heartbeat-evoked potentials

Open [run_hep.ipynb](run_hep.ipynb) with **Python (dbh)** after the
[environment setup](../README.md#setup). Download supplied arrays with `git lfs pull`.

Choose the workflow in step 1:

- **1 — single-subject demonstration:** use the current preprocessing outputs
  for `sub-001` (`../01_preprocessing/outputs/demo/`: cleaned EEG, extracted ECG,
  reviewed R peaks). Save aligned ECG, ECG-regressed EEG and clean epochs to
  `outputs/demo/`. Step 3 then offers the group analyses (default: finish).
  No raw recording is read.
- **2 — manuscript batch:** read IDs from the participant CSV. Start from the
  manuscript cleaned EEG and reviewed R peaks in `../01_preprocessing/outputs/manuscript/`
  and the manuscript aligned ECG in `outputs/manuscript/`. Save new ECG-regressed
  EEG and epochs to `outputs/batch/` or a chosen external folder. ICA and R-peak
  review are already complete in these manuscript inputs.

**HEP preprocessing:** step 2 prepares signals; step 3 inspects the demonstration
and builds/selects the 148-child averages. **Group analysis:** steps 4–7 compute/load
temporal clusters, Table S3, scalp-map contrasts and Figure 3 (which draws the maps). Each run uses one concise log.

Step 2 asks to skip, fill missing outputs, or replace outputs. Step 3 lets both
workflows select rebuilt averages or the manuscript averages explicitly.
Missing inputs are reported and skipped. Manuscript inputs remain unchanged.
Rebuilt averages must agree with the manuscript reference before group analyses run.
A full rebuild needs MATLAB with EEGLAB/ERPLAB and tens of GB of free space;
the notebook checks space before running. Manuscript averages need Python only.

| Location | Contents |
|---|---|
| `../01_preprocessing/outputs/manuscript/eeg/cleaned/` | Manuscript cleaned EEG; available count shown in step 1 |
| `../01_preprocessing/outputs/manuscript/ecg/r_peaks/` | Manuscript manually corrected R peaks |
| `outputs/manuscript/ecg_trimmed/` | Manuscript aligned ECG |
| `outputs/manuscript/averages/` | Manuscript 148-child ROI/channel averages |
| `outputs/demo/` | Current single-subject demonstration outputs |
| `outputs/batch/` | Rebuilt ECG-regressed EEG, epochs and averages |
| `cache/` | Supplied intermediate results of the submitted analysis: temporal clusters, age summaries and map inputs; replaced when recomputed |
| `../results/Main/`, `../results/SI/` | Figure 3 PNG and Table S3a–b |
| `logs/` | One log per run: input version, choices, parameters and step summaries |

Manuscript EEG samples are unchanged; redundant FieldTrip copies and ICA activation
caches were omitted. `outputs/manuscript/input_manifest.json` records the saved
input files and hashes of all three input kinds. Demonstration inputs are produced by
[preprocessing](../01_preprocessing/run_preprocessing.ipynb).
