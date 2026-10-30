# Modified PySDG

`model.py` and `utils.py` implement the SDG model; `pipeline.py` runs each child
and movie and validates cached arrays. The upstream Clear BSD license is retained
in `LICENSE.pysdg`.

Settings: 15-s windows, 4-Hz output, 2-s EEG periodograms, 7.5-s smoothing and
within-clip min–max normalisation. LF/HF oscillator centres are 0.083/0.32 Hz.

Run steps 1–4 of [run_dbhi.ipynb](../run_dbhi.ipynb) to estimate dBHI with this module.
See the [dBHI instructions](../README.md) for inputs and subsequent analyses.

Run the alignment and cache tests from `05_dBHI/`:

```bash
python -m unittest sdg.test_pipeline -v
```

The supplied arrays in `../outputs/manuscript/model/` use Git LFS and retain the
original computation manifest. Analysis validates their settings and stored
fingerprints. Model resumption also requires matching source and environment.
The guided runner writes reports next to the model folder (`model_reports/`); an existing
compatible `reports/` folder can be selected with `--reports-dir`.
`htb_floor.json` supplies the fixed floor for heart-to-brain coefficients.
