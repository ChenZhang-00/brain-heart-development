# Mac application

`RpeakEditor-macOS-AppleSilicon.zip` contains `RpeakEditor.app`, built from
`../launch_editor.py` and `../edit_r_peaks.py` on 2026-10-25.

- Apple Silicon (M-series), macOS 14 or later; tested on macOS 26.6.2.
- Python, Tk, NumPy, SciPy and Matplotlib are included.
- Input: `{ID}_movie_ecg.npz`, created by `extract_ecg.py`.
- Output: `{ID}_movie_rpeaks.npz`, with zero-based sample indices.

## Use the example

1. Run ECG extraction from the repository root in the `dbh` environment:

   ```bash
   python 01_preprocessing/ecg/ECG_R_peak_editor/extract_ecg.py
   ```

2. Unzip the application and double-click `RpeakEditor.app`.
3. Set **INPUT folder** to the repository's `01_preprocessing/outputs/demo/ecg/extracted` folder.
4. Set **OUTPUT folder** to the repository's `01_preprocessing/outputs/demo/ecg/r_peaks` folder.
5. Enter **Subject ID** `sub-001` and click **Open Editor**.
6. Review the peaks. Press `s` to save or `q` to save and close the editor.
   The launcher stays open for the next recording.

When an output already exists, **Resume previous** reloads it. **Start fresh**
detects peaks again and replaces the old output when saved.
See the [controls](../README.md) for adding, deleting and navigating peaks.

The application remembers folder choices in `~/.rpeak_editor_config.json` and
writes a diagnostic log to `~/RpeakEditor.log`. These local files and recordings
are not included in the program package.

This build has an ad-hoc signature and is not notarized by Apple. If macOS blocks
the first launch, use **System Settings → Privacy & Security → Open Anyway**
after confirming that you are opening this package. See
[Apple's instructions](https://support.apple.com/en-us/102445).

## Rebuild

Use an Apple Silicon Mac with Python 3.12 and working Tk. The supplied build used
Python 3.12.7 and Tk 8.6.14. From the repository root:

```bash
python -m venv /tmp/dbh-editor-build
/tmp/dbh-editor-build/bin/python -m pip install -r 01_preprocessing/ecg/ECG_R_peak_editor/distribution/requirements-macos-build.txt
/tmp/dbh-editor-build/bin/python -m PyInstaller --clean --noconfirm --workpath /tmp/dbh-editor-build/work --distpath /tmp/dbh-editor-build/dist 01_preprocessing/ecg/ECG_R_peak_editor/distribution/RpeakEditor.spec
```

The resulting application is `/tmp/dbh-editor-build/dist/RpeakEditor.app`.
