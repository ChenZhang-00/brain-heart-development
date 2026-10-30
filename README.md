# Developmental Reorganization of Brain–Heart Interplay during Affective Experience

Analysis code for the manuscript: 148 children aged 5–10 years viewed fearful,
happy and neutral movies with simultaneous EEG and ECG recording.

## Analyses

| Folder | Analysis | Outputs |
|---|---|---|
| [`00_data/`](00_data/README.md) | Source data and participant demographics by age group | Table S1 |
| `01_preprocessing/` | EEG cleaning, verified R peaks and HRV | Methods |
| `02_eliciting_effects/` | SAM ratings, cardiac responses, temporal dynamics and intensity correlations | Fig. S1; Table S2 |
| `03_correlation/` | Group-mean EEG–IBI covariation | Fig. 2; Fig. S2 |
| `04_HEP/` | ECG-regressed HEPs, temporal clusters and continuous-age regressions | Fig. 3; Table S3 |
| `05_dBHI/` | Directional modulation, age prediction and intensity contrasts | Figs. 4–5; Figs. S3–S7; Tables S4–S6 |

The numeric prefixes set the browsing order. Preprocessing supplies the inputs
for the subsequent analysis branches.

Fig. 1 is a design/pipeline illustration. Each folder's README gives the run order.
The repository contains source code, recoded participant information, analysis
data for all 148 children and one example recording. Download the signal arrays
with `git lfs pull` after cloning. File versions and hashes are recorded in
`00_data/analysis_data_manifest.json`.

## Setup

The workflow uses MATLAB for EEG preprocessing and Python for HRV, dBHI,
statistics and figures. Create the Python environment using `environment.yml`.

In particular, use NumPy 2.2.6, Numba 0.61.2 and llvmlite 0.44.0 together.
The HRV code calls `numpy.trapezoid` (NumPy 2.0+), and Numba 0.60 cannot import
with NumPy 2.2. MNE is pinned because the scalp plots use an internal plotting
function. The modified SDG implementation is included in `05_dBHI/sdg/`.

```bash
conda env create -f environment.yml
conda activate dbh
python -m ipykernel install --user --name dbh --display-name "Python (dbh)"
python check_setup.py --python-only
```

Start Jupyter with `python -m jupyterlab` from this environment and select
**Python (dbh)** in every notebook. Selecting the default `base` kernel can load
incompatible packages even when the `dbh` environment has been installed.

Install MATLAB with Signal Processing Toolbox and
EEGLAB with BVA-io (`pop_loadbv`), DIPFIT (`eeglab2fieldtrip`), Zapline-plus,
CleanLine, ICLabel and ERPLAB Classic, plus FieldTrip (`ft_freqanalysis`).
See the [BVA-io source](https://github.com/sccn/bva-io/blob/master/pop_loadbv.m),
[DIPFIT documentation](https://eeglab.org/plugins/dipfit/) and
[ERPLAB requirements](https://github.com/ucdavis/erplab/wiki/Installation).
Check the linked installation requirements for your MATLAB version and platform.
Place EEGLAB and FieldTrip in `toolboxes/eeglab/` and `toolboxes/fieldtrip/`
at the repository root (see `toolboxes/README.md`). Only that README is tracked
by Git; the toolboxes themselves are not. Existing installations can be used without copying: create symbolic
links at these two paths, or set the environment variables `DBH_EEGLAB_PATH` and
`DBH_FIELDTRIP_PATH` before starting the preprocessing notebook
(see `01_preprocessing/config/preprocessing_config.m`).

The code uses CPU computation. The interactive R-peak editor also needs Tk and a
desktop display; its GUI cannot run in a headless session.

Run Python commands from the repository root unless a folder README says otherwise.
Run notebooks from their own folder. The supplied analysis data support the four
cohort workflows directly.

Run `python check_setup.py` to check
Python imports, input columns and BrainVision filenames. This check does not run
MATLAB or validate the full analyses. Follow the analysis entries in order:
eliciting → correlation → HEP → dBHI. Preprocessing is available for new recordings.

## Example recording

Start with [01_preprocessing/run_preprocessing.ipynb](01_preprocessing/run_preprocessing.ipynb)
for the guided raw-data workflow. After shared setup, EEG and ECG run in separate
sections. MATLAB is used only in the EEG section. Each processing step offers
**1 = skip existing results** or **2 = recompute and replace**. For R peaks, option 2
reopens the editor; existing corrections change only when saved. Select the
**Python (dbh)** kernel.

A complete real recording for `sub-001` is included in `00_data/raw_data/`, with
65 channels at 500 Hz and all three movie segments. The signal samples are
unchanged; filenames are recoded, and the recording date, workstation paths and
equipment serial numbers are removed.
See the [example instructions](00_data/raw_data/README.md) for reading the files,
running ECG extraction, and starting single-participant EEG preprocessing.

The preprocessing notebook produces cleaned EEG, verified R peaks, band power and
HRV. Analysis entries are [eliciting](02_eliciting_effects/run_eliciting_effects.ipynb) →
[correlation](03_correlation/run_correlation.ipynb) → [HEP](04_HEP/run_hep.ipynb) →
[dBHI](05_dBHI/run_dbhi.ipynb). Single-subject mode runs the included example. Batch mode
looks for the 148 IDs in `00_data/participants.csv`, reports missing recordings and
processes only complete recordings in CSV order. Cohort analyses use the supplied
148-child features and caches below.

The 110.29 MiB `.eeg` example uses Git LFS. After cloning, install Git LFS and run
`git lfs pull` if the recording has not been downloaded.

## Generated tables and figures

Final exports are collected at the repository root and named by manuscript number:

```text
results/
  Main/    # Figure_2a..., Figure_2b..., Figure_3..., Figure_4..., Figure_5...
  SI/      # Figure_S1... through Figure_S7...; Table_S1... through Table_S6...
```

| Workflow | Main text | Supplementary information |
|---|---|---|
| `00_data` | — | Table S1 and its sex-ratio test note |
| `02_eliciting_effects` | — | Figure S1, Table S2 |
| `03_correlation` | Figure 2a–b | Figure S2 |
| `04_HEP` | Complete Figure 3 (PNG), with four scalp insets | Table S3a–b |
| `05_dBHI` | Figures 4–5 | Figures S3–S7, Tables S4–S6 |

Each analysis keeps intermediate statistics and plotting inputs in its own
`cache/`, and execution logs in its own `logs/`. The supplied `cache/` folders hold
the intermediate results of the submitted analysis, so a notebook that reuses
saved results reproduces the manuscript tables and figures without recomputing;
recomputing replaces them in place. `results/Main/` and `results/SI/` ship the
final figures and tables generated by the notebooks; a rerun replaces them in
place. Logs and other generated files are ignored by Git. Figure 1 is an experimental
schematic and is not generated by these analysis scripts.

## Data

Source metadata and movie ratings are in `00_data/`; signal features remain with
their processing modules. The supplied analysis arrays use Git LFS. Open
[run_sample_characteristics.ipynb](00_data/run_sample_characteristics.ipynb) to
check participants and generate Table S1 under `results/SI/`. The direct
command is `python 00_data/export_table_s1.py`; see the [data instructions](00_data/README.md).

```text
common/                            # shared by every notebook: paths, run log, reuse policy, participant IDs, notebook style

00_data/
  README.md
  run_sample_characteristics.ipynb # guided Table S1 workflow
  export_table_s1.py              # sample characteristics (Table S1)
  participants.csv                # included: 148 recoded participant records
  events/Movies_rating_by_RA.csv
  raw_data/{ID}_movie.vhdr          # matching .eeg and .vmrk files

01_preprocessing/outputs/
  manuscript/                     # supplied, read only: the 148-child manuscript version
    eeg/cleaned/                  # cleaned EEG; exact samples, redundant caches omitted
    eeg/band_power/               # EEG band power
    ecg/r_peaks/                  # manually reviewed R peaks
    ecg/hrv/ecg_ts.npz            # IBI and LF/HF time series
  demo/                           # single-subject runs of run_preprocessing.ipynb (sub-001)
    eeg/cleaned/, eeg/band_power/, ecg/extracted/, ecg/r_peaks/, ecg/hrv/
  batch/                          # CSV batch runs of run_preprocessing.ipynb (same layout)

04_HEP/outputs/
  manuscript/                     # supplied, read only: HEP inputs of the manuscript version
    ecg_trimmed/                  # aligned ECG
    averages/                     # 148-child ROI/channel means
    input_manifest.json           # files and hashes of all manuscript HEP inputs
  demo/                           # current sub-001 demonstration outputs
  batch/                          # rebuilt ECG-regressed EEG, epochs and averages

02_eliciting_effects/cache/, 03_correlation/cache/, 04_HEP/cache/, 05_dBHI/cache/
                                  # supplied: intermediate results of the submitted analysis;
                                  # recomputing replaces them in place

05_dBHI/outputs/
  manuscript/model/               # supplied: 148 model arrays and original manifest
  manuscript/model_reports/       # supplied: completion report of that model run
  batch/, demo/                   # new model estimates: full cohort / selected IDs
```

The included [participants.csv](00_data/participants.csv) has columns `ID`, `Gender` (`M`/`F`), `Age_month`, `Age_group`
(completed years, 5–10), and `fear_A`, `fear_V`, `happy_A`, `happy_V`, `neutral_A`,
`neutral_V` (SAM scores, 1–9). The study cohort has 50, 49 and 49 children in
60–83, 84–107 and 108–131 month groups. Use consistent recoded IDs (`sub-001` etc.)
in metadata, filenames and array contents. Birth dates, experiment dates and the
original ID mapping are not included.

`Movies_rating_by_RA.csv` has `Time` (integer seconds 0–189), `Fear_Intensity` and
`Happy_Intensity` (six-rater means at 1 Hz). It is an analysis input derived from
5-s ratings; the original rater sheets are not included. The intensity notebook
computes the 15-s smoothing and selected epochs from this input.

The `manuscript/` folders hold the preprocessing outputs and HEP inputs of the
submitted manuscript for all 148 children. The eliciting, correlation and dBHI
analyses read them by default. HEP offers a single-subject demonstration and a
manuscript CSV batch. The batch starts from the manuscript cleaned EEG, R peaks
and aligned ECG; it rebuilds ECG-regressed EEG and clean HEP epochs in a separate
output folder. A full rebuild needs tens of GB of free space. Alternatively,
select the supplied manuscript averages for group analyses. dBHI analyses the
supplied model arrays by default; its step 1 can instead estimate the SDG model
from the manuscript cleaned EEG and R peaks, or for the included `sub-001`.

## License

Code: MIT (`LICENSE`). The modified PySDG implementation in `05_dBHI/sdg/` retains
its upstream Clear BSD license (`05_dBHI/sdg/LICENSE.pysdg`). See `CITATION.cff`.
