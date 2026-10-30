# MATLAB toolboxes

This folder holds the MATLAB toolboxes used by `01_preprocessing/` (EEG cleaning
and band power) and by the full HEP rebuild in `04_HEP/`. It is not tracked by
Git apart from this file.

| Path | Toolbox |
|---|---|
| `toolboxes/eeglab/` | EEGLAB with the BVA-io, DIPFIT, Zapline-plus, CleanLine, ICLabel and ERPLAB plugins |
| `toolboxes/fieldtrip/` | FieldTrip (used by `compute_eeg_band_power.m` for `ft_freqanalysis`) |

Copy or unpack each toolbox into its folder, or create a symbolic link to an
existing installation:

```bash
ln -s /path/to/eeglab    toolboxes/eeglab
ln -s /path/to/fieldtrip toolboxes/fieldtrip
```

Alternatively set `DBH_EEGLAB_PATH` and `DBH_FIELDTRIP_PATH` before starting
Jupyter; `01_preprocessing/config/preprocessing_config.m` reads them. The paths
are resolved from the repository location, not from the MATLAB working directory.
