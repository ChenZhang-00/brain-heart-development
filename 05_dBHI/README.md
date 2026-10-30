# Directional brain–heart interplay

## Start here

Open [run_dbhi.ipynb](run_dbhi.ipynb) for the guided workflow producing
**Figs. 4–5, Figs. S3–S7 and Tables S4–S6**. After the
[environment setup](../README.md#setup), run from the repository root:

```bash
python -m jupyterlab 05_dBHI/run_dbhi.ipynb
```

Select **Python (dbh)** and run the cells in order. Step 1 asks two questions:
**1** analyse the supplied 148-child model arrays (default) or **2** estimate the SDG
model from cleaned EEG and reviewed R peaks (all participant-table IDs, or the
included `sub-001` example). Step 4 asks which model arrays the group analyses
use (the manuscript arrays, or your own estimates when they cover all 148
children) and whether to reuse complete saved results or recompute them; saved
results are also refreshed when they are missing, incomplete or older than their
inputs. Each analysis branch (temporal, age, intensity) asks whether to run before
its first step. Every step explains its method and parameters, lists its
intermediate files, and displays the resulting tables and figures. Each run writes
one log to `logs/`.

Group analyses require **all 148 children**, with 50/49/49 in the three age groups;
after estimating the model for a subset (for example `sub-001`) the group analyses
can still run on the manuscript arrays. The supplied model arrays in `outputs/manuscript/model/` are never
replaced: new estimates go to `outputs/batch/model/` (full cohort) or `outputs/demo/model/`
(selected IDs). Estimates in those folders are resumed only when their fingerprint
(model source, settings, package versions) matches; otherwise the prompt offers to
replace them. Missing children stop execution; they are not silently skipped.

## Layout

| Folder | Contents |
|---|---|
| `config/` | Shared paths and constants, notebook prompts and reuse policy, per-electrode transforms, display helpers, notebook style |
| `preparation/` | Model runner, supported neutral-segment recomputation and intensity-epoch selection |
| `analysis/` | Cohort/cache checks and epoch masks, shared ridge model, temporal and spatial tests, Haufe analysis and table exports |
| `plotting/` | Scripts for Figs. 4–5 and Figs. S3–S7 |
| `sdg/` | Modified PySDG, regression tests, fixed heart-to-brain floor and upstream license |
| `cache/` | Supplied intermediate results of the submitted analysis: model reports, statistics, feature arrays and parameters; replaced when recomputed |
| `logs/` | One log per notebook run |
| `../results/Main/`, `../results/SI/` | Numbered manuscript figures and tables |

The top level contains this README and `run_dbhi.ipynb`. `config/cache_policy.py`
holds the notebook prompts; the saved-result reuse check is shared with the other
notebooks in `common/cache_policy.py`. Model estimation reads
cleaned EEG and reviewed R peaks from `01_preprocessing/outputs/manuscript/`
(or `outputs/demo/` for the included example; `DBHI_EEG_DIR` and `DBHI_RPEAK_DIR`
override these folders).
`outputs/manuscript/model/` holds the supplied model arrays; new estimates default to `outputs/batch/model/` (full cohort) or `outputs/demo/model/` (selected IDs).
The supplied arrays use Git LFS. Analysis validates their settings, arrays and
original fingerprints. Recomputing the model requires cleaned EEG/R peaks and a
new cache directory when the computation environment differs.

## Inputs and paths

| Input | Required content |
|---|---|
| `00_data/participants.csv` | Unique, nonblank `ID`, numeric `Age_group` (integer ages 5–10) and `Age_month` (60 to less than 132 months) |
| `01_preprocessing/outputs/manuscript/eeg/cleaned/{ID}_continuous_clean.mat` | Preprocessed continuous EEG and segmentation metadata, MATLAB v7.3/HDF5 |
| `01_preprocessing/outputs/manuscript/ecg/r_peaks/{ID}_movie_rpeaks.npz` | Reviewed R peaks and sampling rate, with matching recording/condition boundaries |
| `00_data/events/Movies_rating_by_RA.csv` | 1-Hz six-rater mean ratings: `Time`, `Fear_Intensity`, `Happy_Intensity` |
| `outputs/manuscript/model/{ID}_dbhi.npz` and `_manifest.json` | Model arrays, timestamps, labels, input identities and model fingerprint |

Use matching IDs throughout. Groups use the existing `Age_group` column
(5–6, 7–8, 9–10); continuous age prediction uses `Age_month`. Analyses require
exactly the matching 148-ID cache set and consistent 63-channel order. Temporal
ROI lists use one-based positions in the study montage; retain that channel order.

Defaults are anchored to the repository root, so commands also work from other
working directories. Notebook step 1 exposes `INFO_FILE`, `RATINGS_FILE`,
`OUTPUT_ROOT`, `N_JOBS` and the branch switches; the model cache folder is chosen
at the prompt. Relative paths are resolved from the repository root. Command-line equivalents are `--participants`,
`--ratings`, `--cache-dir`, `--reports-dir`, `--neutral-subject` and `--output-root`
where relevant; use each script's `--help`.

Model EEG and R-peak inputs use the `01_preprocessing/outputs/manuscript/` locations defined in
`sdg/pipeline.py`. Analysis of completed caches does not reopen those inputs.
Model reports sit next to the model arrays (`outputs/manuscript/model_reports/`,
or `outputs/batch/model_reports/` for new estimates). Temporal analysis requires a complete
148-child model report with a matching fingerprint. The neutral clip of `sub-047`
is truncated in place to its R-peak-supported segment (0–159.342 s) by
`preparation/recompute_neutral_dbhi.py`; `--neutral-subject` supports its
equivalent ID in a locally unrelabelled dataset, not arbitrary trimming.

Cache fingerprints record the original sources, settings and environment.
Reading completed arrays preserves that fingerprint and checks the analysis
settings. Model resumption additionally requires matching current sources,
environment and EEG/R-peak identities. Use a new cache for a different computation
environment; do not replace the original manifest.

## Run order

The notebook steps and their scripts, runnable from the repository root with the
`dbh` environment active:

```bash
# Step 2: validate the supplied model arrays (including the truncated neutral clip).
python 05_dBHI/analysis/inputs.py --temporal

# Step 3 (estimation workflow): model preflight and estimation; then truncate the
# neutral clip of sub-047 (0–159.342 s) in place.
python 05_dBHI/preparation/compute_dbhi.py --check-only
python 05_dBHI/preparation/compute_dbhi.py --n-jobs 4
python 05_dBHI/preparation/recompute_neutral_dbhi.py

# Steps 5–6, temporal branch: Figs. S3–S4 and Table S4.
python 05_dBHI/analysis/analyze_dbhi_timecourses.py
python 05_dBHI/plotting/plot_figures_s3_s4_dbhi_timecourses.py
python 05_dBHI/analysis/export_table_s4_temporal_clusters.py

# Step 7: intensity epochs shared by the age and intensity branches; also Fig. S5.
python 05_dBHI/preparation/select_intensity_epochs.py
python 05_dBHI/plotting/plot_figure_s5_intensity_epochs.py

# Steps 8–10, age branch: Fig. 4 and Table S5; independent of the intensity tests.
python 05_dBHI/analysis/predict_age_with_ridge.py
python 05_dBHI/analysis/compute_haufe_patterns.py
python 05_dBHI/plotting/plot_figure_4_age_prediction.py
python 05_dBHI/analysis/export_table_s5_age_prediction.py

# Steps 11–13, intensity branch: Fig. 5, Figs. S6–S7 and Table S6.
python 05_dBHI/analysis/compute_intensity_contrasts.py
python 05_dBHI/analysis/analyze_intensity_spatial_clusters.py
python 05_dBHI/plotting/plot_figure_5_intensity_contrasts.py
python 05_dBHI/plotting/plot_figures_s6_s7_intensity_contrasts.py
python 05_dBHI/analysis/export_table_s6_intensity_clusters.py
```

Run from the command line, every script recomputes and replaces its named outputs;
the notebook adds the reuse check. Use a separate `--output-root` (or
`DBHI_OUTPUT_ROOT`) to retain earlier analyses. Model estimation resumes valid
per-child caches and prevents concurrent writes with a run lock. Plotting and
table exports read saved statistics; Figs. S3–S4 additionally read the model
arrays. Changing model inputs requires rerunning the dependent
analyses, not just redrawing figures.

## Outputs

`cache/` and `logs/` are inside this analysis folder. `results/Main/` and
`results/SI/` are at the repository root.

| Location | Principal files |
|---|---|
| `outputs/manuscript/model/`, `model_reports/` | Supplied 148-child model arrays and the completion report of their run |
| `outputs/batch/model/` or `outputs/demo/model/`, with `model_reports/` | New model estimates; `input_qc.csv` and `run_status.json` |
| `cache/time_resolved/` | Included subjects; four `dBHI_{band}_clusters.csv` and sample-count tables; parameters/status/summary; detailed temporal-cluster summaries |
| `cache/intensity_contrast/` | `smoothed_intensity.csv`, `ra_selected_epochs.csv`, selection parameters, `topo_subject_channel_diff.npz`, `contrast_parameters.json` |
| `cache/intensity_contrast/topo_cluster/` | Channel tests, clusters, map summaries, null distributions, adjacency arrays/edges and parameters |
| `cache/age_prediction/` | Subjects, 504-feature matrices, model results, predictions, nulls, Haufe patterns, bootstraps and parameters |
| `results/Main/` | `Figure_4_age_prediction.png`, `Figure_5_intensity_contrasts.png` |
| `results/SI/` | Figures S3–S7 and Tables S4–S6, each prefixed by its manuscript number |
| `cache/intensity_contrast/topo_cluster/channel_adjacency.png` | Channel-adjacency diagnostic |
| `logs/` | One log per notebook run: workflow, choices, parameters and step summaries |

Figure filenames follow manuscript numbering. Layouts are unchanged. PNG resolution remains 300 dpi for
Figs. 4–5 and S3–S4/S6–S7, 220 dpi for S5 and 160 dpi for adjacency QC. The
notebook has cleared outputs and execution counts; clear them again before sharing.

## Analysis settings and interpretation

- **Model:** 15-s windows, 4-Hz output, 2-s EEG periodograms, 7.5-s smoothing and
  within-clip min–max normalization. EEG bands: 1–4, 4–7, 7–12, 12–30 Hz;
  LF/HF: 0.04–0.15/0.15–0.80 Hz. Oscillator centres: 0.083/0.32 Hz.
- **Transforms:** per-electrode `log10` precedes averaging in both directions.
  Heart-to-brain values first use floor `0.00034448521910235286`. Its fixed
  original 147-child estimation is recorded in `sdg/htb_floor.json`; it is not
  re-estimated for 148 children. NaNs remain; infinite and nonpositive
  brain-to-heart values are rejected.
- **Temporal branch:** subtract each child's neutral ROI mean. For `sub-047`,
  the supplied array holds neutral recomputed before the first R-peak gap, on
  0–159.342 s; fearful/happy arrays are unchanged. Align emotion arrays on the common physical 4-Hz
  grid. Per-series exclusions of nonfinite children are saved in sample-count
  tables. Two-sided clusters use forming p < .05, at least 20 samples (5 s),
  summed t and 10,000 sign flips, seed 0, with plus-one p correction. Correction
  is within each series, not across the 288 series (72 per EEG band).
- **Intensity selection:** 15-s forward rating windows start at integer seconds
  3–163. Select 48 highest and 48 lowest starts per movie, ties preferring earlier
  time. Add 7.5 s for model-centre alignment. Each epoch is one second, requiring
  192 samples per tail. No minimum-run filter is applied.
- **Spatial branch:** each child's transformed high-minus-low mean per electrode,
  (fearful/happy clips only). There are 96 two-sided one-sample spatial cluster
  tests, forming p < .05, 10,000 MNE configurations, seed 20260928, summed t and
  maximum absolute cluster mass. MNE includes the observed configuration; maps
  without observed clusters skip permutations. Adjacency is standard-1020
  Delaunay with F9–P9 and F10–P10 hull shortcuts removed. Minimum cluster size is
  one electrode. Correction is within each map, not across maps.
- **Age models:** four direction × movie models use 504 top-intensity features,
  ordered EEG band → HRV band → channel. Scaling is fitted within training folds;
  RidgeCV selects 25 log-spaced penalties from 0.01 to 10,000 using its internal
  leave-one-out criterion. Predict age in months; export MAE in years. Seed 0 and
  1,000 label permutations are retained. Centred models subtract each child's
  mean over 504 features. Haufe patterns use 1,000 participant bootstraps, seed 1,
  with the full-sample penalty fixed.

**Retained age-model limitation:** observed predictions average five repeats of
10-fold CV, whereas each label permutation uses one repeat. The null and observed
statistics therefore use different repetition schemes. This organization preserves
that procedure; it does not resolve the mismatch. Both repeat counts are saved
in the parameter files. Pointwise 95% Haufe bootstrap CIs are not corrected across
features. Fig. 4's age trajectories use the same data that selected the electrodes
and are descriptive rather than an independent validation.

Run the existing model alignment and cache-safety tests from `05_dBHI/`:

```bash
python -m unittest sdg.test_pipeline -v
```

With a custom `--output-root`, intermediate files go to its `cache/` folder and
final exports to its `results/Main/` or `results/SI/`. Explicit figure-directory
overrides take precedence where supported.
