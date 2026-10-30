# Eliciting effects

Open [run_eliciting_effects.ipynb](run_eliciting_effects.ipynb) with **Python (dbh)**
after the [environment setup](../README.md#setup). The 148-child analysis data are
supplied; download the arrays with `git lfs pull` after cloning.

Run the six steps in order:

1. Set paths; choose **1** to use the saved results (the supplied submitted-analysis results, or your own rerun) or **2** to recompute and replace.
2. Check and inspect participant, HRV, EEG and movie-intensity inputs.
3. Compare SAM ratings and cardiac responses across the three movies.
4. Test emotion-minus-neutral IBI and frontal EEG time courses.
5. Correlate physiological dynamics with movie intensity; display Table S2.
6. Draw and display Figure S1.

Each analysis returns a table available in the notebook. The numerical functions
are in `analysis/`; figure code is in `plotting/`. Fixed settings are in
`config/eliciting_config.py`. Mode 1 reads complete saved steps and recomputes
incomplete steps. Mode 2 replaces the results of each step you run; use it after
changing inputs or settings. Figure S1 is refreshed when an earlier step changes.

| Location | Contents |
|---|---|
| `cache/` | Supplied intermediate results of the submitted analysis: test tables, figure inputs and calculation manifests; replaced when recomputed |
| `../results/SI/` | `Table_S2_intensity_correlations.csv`, `Figure_S1_movie_responses.png` |
| `logs/` | One concise log per notebook run: settings, step times, summaries and errors |

Rerunning step 1 starts a new log; later steps append to it. To redraw saved
results, run steps 1 and 6 and choose mode 2. Subsequent analysis entries are
correlation → HEP → dBHI, linked at the end of the notebook.
