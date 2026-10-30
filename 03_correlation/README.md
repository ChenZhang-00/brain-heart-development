# Nondirectional EEG–IBI covariation

Open [run_correlation.ipynb](run_correlation.ipynb) with **Python (dbh)** after the
[environment setup](../README.md#setup). The 148-child participant information,
IBI and EEG band power are supplied; download the arrays with `git lfs pull`.

Run the five steps in order:

1. Set paths; choose **1** to use the saved results (the supplied submitted-analysis results, or your own rerun) or **2** to recompute and replace.
2. Check and inspect participant IDs, age groups, IBI and EEG file availability.
3. Compute or load 108 group-mean EEG–IBI correlations; display the frontal tests.
4. Draw or display Figure 2a (frontal correlations) and Figure 2b (alpha–IBI regression).
5. Draw or display Figure S2 (all three regions).

The result tables remain available in the notebook. Numerical functions are in
`analysis/`; figure code is in `plotting/`. Fixed settings and electrode groups
are in `config/correlation_config.py`. Mode 1 reads complete saved steps and
recomputes incomplete steps. Mode 2 replaces the results of each step you run;
use it after changing inputs or settings. Figures refresh when their inputs change.

| Location | Contents |
|---|---|
| `cache/` | Supplied intermediate results of the submitted analysis: correlation tables, mean time courses, input summaries and settings; replaced when recomputed |
| `../results/Main/` | `Figure_2a_eeg_ibi_correlations.png`, `Figure_2b_alpha_ibi_regression.png` |
| `../results/SI/` | `Figure_S2_eeg_ibi_correlations.png` |
| `logs/` | One concise log per notebook run: settings, step times, summaries and errors |

Rerunning step 1 starts a new log; later steps append to it. To redraw saved
results, run steps 1 and 4–5 and choose mode 2. Subsequent analysis entries are
HEP → dBHI, linked at the end of the notebook.
