"""Prepare HEP signals from the manuscript batch inputs or the current demonstration."""

from dataclasses import replace
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import shutil
import sys

import h5py
import numpy as np

shown = import_module("common.paths").relative_path
_notebook_helpers = import_module("04_HEP.config.notebook_helpers")
matlab_quote = _notebook_helpers.matlab_quote
run_logged = _notebook_helpers.run_logged
run_matlab = _notebook_helpers.run_matlab


def report(message, log=None):
    print(message)
    if log is not None:
        log.write(message)


def show_average_files(paths, log=None):
    """Show the exact files selected for the cohort analyses."""
    files = {"ROI averages": paths.roi, "Channel averages": paths.channel_dir / "chan_means.npz"}
    for label, path in files.items():
        state = f"found, {path.stat().st_size / 1024**2:.1f} MB" if path.is_file() else "missing"
        report(f"  {label:<18}{shown(path, paths.root)}  [{state}]", log)


def choose_workflow(root, output_root, *, prompt=None, log=None):
    """Keep the manuscript CSV batch separate from the current single-subject demo."""
    selection = import_module("common.subjects")
    paths_module = import_module("common.paths")
    ask = input if prompt is None else prompt
    root, output_root = Path(root), Path(output_root)
    manuscript_pre = root / "01_preprocessing/outputs/manuscript"
    manuscript_hep = root / "04_HEP/outputs/manuscript"
    print("1 = single-subject demonstration using current preprocessing outputs (default)")
    print("2 = batch from the manuscript cleaned EEG, reviewed R peaks and aligned ECG")
    while True:
        answer = ask("Type 1 or 2, then press Enter (Enter alone = 1): ").strip() or "1"
        if answer in {"1", "2"}:
            break
        print("Enter 1 for the demonstration or 2 for the manuscript batch.")
    mode = "demo" if answer == "1" else "batch"
    if mode == "demo":
        while True:
            sid = ask("Subject ID (Enter alone = sub-001): ").strip() or "sub-001"
            try:
                ids = selection.validate_ids([sid])
                break
            except ValueError as error:
                print(error)
        source = root / "01_preprocessing/outputs/demo"
        destination = output_root / "outputs/demo"
        paths = {"ecgExtractedPath": source / "ecg/extracted",
                 "preprocessedOut": source / "eeg/cleaned",
                 "correctedRpeakPath": source / "ecg/r_peaks",
                 "ecgTrimmedPath": destination / "ecg_trimmed"}
    else:
        default_csv = root / "00_data/participants.csv"
        print(f"Batch participant CSV: {shown(default_csv, root)}")
        while True:
            value = ask("Press Enter to use this CSV, or type another CSV path: ").strip()
            csv = paths_module.resolve_path(value or str(default_csv), root)
            try:
                ids = selection.read_subject_ids(csv)
                break
            except (OSError, ValueError) as error:
                print(f"Cannot read this CSV: {error}")
        default_output = output_root / "outputs/batch"
        print(f"Batch output folder: {shown(default_output, root)}")
        print("A complete rebuild needs tens of GB; an external output folder is supported.")
        while True:
            value = ask("Press Enter to use this folder, or type another output folder: ").strip()
            destination = paths_module.resolve_path(value or str(default_output), root)
            protected = [manuscript_pre, manuscript_hep, root / "01_preprocessing", output_root / "outputs/demo"]
            if any(destination == p or p in destination.parents for p in protected):
                print("Choose a separate output folder for the manuscript batch.")
                continue
            break
        paths = {"preprocessedOut": manuscript_pre / "eeg/cleaned",
                 "correctedRpeakPath": manuscript_pre / "ecg/r_peaks",
                 "ecgTrimmedPath": manuscript_hep / "ecg_trimmed"}
        available = sum((paths["preprocessedOut"] / f"{sid}_continuous_clean.mat").is_file()
                        for sid in ids)
        notes = [f"Manuscript cleaned EEG available: {available}/{len(ids)} selected participants."]
        manifest = manuscript_hep / "input_manifest.json"
        if manifest.is_file():
            try:
                archive = json.loads(manifest.read_text())
            except (OSError, ValueError):
                archive = {}
            if archive.get("status") != "complete":
                notes.append("Manuscript EEG archive is incomplete; missing subjects will be skipped.")
            mismatched = archive.get("reference_checks", {}).get("mismatched_ids", [])
            selected_mismatches = sorted(set(ids) & set(mismatched))
            if selected_mismatches:
                notes.append("Previous rebuild differs from the manuscript averages for: "
                             + ", ".join(selected_mismatches) + ". Step 3 checks rebuilt averages "
                             "before group analysis; choose the manuscript averages there to "
                             "reproduce the supplied group results.")
    paths.update(preprocessedEcgRegOut=destination / "preprocessed_ecgreg",
                 hepEcgRegOut=destination / "hep_epochs_ecgreg")
    if mode == "demo":
        report(f"\nWorkflow: single-subject demonstration ({ids[0]})", log)
        inputs = [("Cleaned EEG", paths["preprocessedOut"]),
                  ("Extracted ECG", paths["ecgExtractedPath"]),
                  ("Reviewed R peaks", paths["correctedRpeakPath"])]
        outputs = [("Aligned ECG", paths["ecgTrimmedPath"])]
    else:
        report(f"\nWorkflow: manuscript batch; {len(ids)} subjects from {shown(csv, root)}", log)
        for note in notes:
            report(note, log)
        inputs = [("Cleaned EEG", paths["preprocessedOut"]),
                  ("Reviewed R peaks", paths["correctedRpeakPath"]),
                  ("Aligned ECG", paths["ecgTrimmedPath"])]
        outputs = []
    outputs += [("ECG-regressed EEG", paths["preprocessedEcgRegOut"]),
                ("Clean HEP epochs", paths["hepEcgRegOut"])]
    for heading, items in (("Inputs (read only)", inputs), ("Outputs", outputs)):
        report(heading, log)
        for label, path in items:
            report(f"  {label:<18}{shown(path, root)}", log)
    return mode, ids, paths, destination


def choose_epoch_mode(*, prompt=None, log=None):
    """Ask explicitly whether to skip, resume or replace signal preparation."""
    print("\nStep 2: prepare clean HEP epochs")
    print("1 = skip signal preparation; use existing epochs or averages")
    print("2 = prepare missing epochs; keep existing signal outputs (default)")
    print("3 = recompute and replace signal outputs for selected subjects")
    ask = input if prompt is None else prompt
    modes = {"1": "skip", "2": "reuse", "3": "recompute"}
    while True:
        answer = ask("Type 1, 2 or 3, then press Enter (Enter alone = 2): ").strip() or "2"
        if answer in modes:
            report(f"Signal preparation mode: {modes[answer]}", log)
            return modes[answer]
        print("Enter 1 to skip, 2 to prepare missing files, or 3 to recompute.")


def missing_epochs(paths):
    read_ids = import_module("common.subjects").read_subject_ids
    ids = read_ids(paths.participants)
    missing = []
    for sid in ids:
        path = paths.epochs / f"{sid}_hep_epochs.mat"
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(sid)
    return ids, missing


def choose_average_source(manuscript, batch, *, updated_epochs=(), prompt=None, log=None, demo=False):
    """Select rebuilt batch averages or the unchanged manuscript reference."""
    default = "4" if demo else "1"
    if demo:
        print("\nGroup analyses (steps 4-7) are optional after the demonstration.")
    print("\nManuscript averages (read only)")
    show_average_files(manuscript, log)
    print("Rebuilt batch averages")
    show_average_files(batch, log)
    ids, missing = missing_epochs(batch)
    report(f"Batch epochs: {len(ids) - len(missing)}/{len(ids)} in {shown(batch.epochs, batch.root)}", log)
    averages = [batch.roi, batch.channel_dir / "chan_means.npz"]
    existing = all(path.is_file() for path in averages)
    oldest = min(path.stat().st_mtime_ns for path in averages) if existing else 0
    newer_epochs = existing and any(
        (batch.epochs / f"{sid}_hep_epochs.mat").stat().st_mtime_ns > oldest
        for sid in ids if sid not in missing
    )
    labels = {"1": "use rebuilt batch averages; create them if missing",
              "2": "recompute and replace batch averages from all 148 epoch files",
              "3": "use the supplied manuscript averages for group analyses",
              "4": "finish without group analyses"}
    for key, label in labels.items():
        print(f"{key} = {label}" + (" (default)" if key == default else ""))
    ask = input if prompt is None else prompt
    while True:
        answer = ask(f"Type 1–4, then press Enter (Enter alone = {default}): ").strip() or default
        if answer == "4":
            report("Finished without group analyses.", log)
            return None
        if answer == "3":
            if not manuscript.roi.is_file() or not (manuscript.channel_dir / "chan_means.npz").is_file():
                report("Missing: supplied manuscript averages. Choose another option.", log)
                continue
            report("Group inputs: supplied manuscript averages (unchanged).", log)
            return manuscript, False
        if answer not in {"1", "2"}:
            print("Enter 1, 2, 3 or 4.")
            continue
        need_epochs = answer == "2" or bool(updated_epochs) or newer_epochs or not existing
        if need_epochs and missing:
            report(f"Missing: {len(missing)} epoch files; cannot generate full-cohort averages. "
                   "Existing files are unchanged.", log)
            print("First missing IDs: " + ", ".join(missing[:5]))
            print("Choose 3 for the manuscript averages or 4 to finish.")
            continue
        if (updated_epochs or newer_epochs) and answer == "1":
            report("Epochs were updated; their batch averages will be refreshed.", log)
        report("Group inputs: rebuilt batch averages.", log)
        return batch, need_epochs


def ready_file(path):
    """Treat only complete, readable stage outputs as reusable."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.suffix == ".mat":
        required = {"hep_epochs": ["EEG/data", "conditions", "qc_hep/n_kept_total"],
                    "ecg_regressed": ["EEG/data", "qc_ecgreg"],
                    "continuous_clean": ["EEG/data", "EEG/srate"]}
        keys = next((v for k, v in required.items() if path.stem.endswith(k)), ["EEG/data"])
        try:
            with h5py.File(path, "r") as source:
                return all(key in source for key in keys)
        except OSError:
            return False
    if path.suffix == ".npz":
        try:
            with np.load(path, allow_pickle=False) as data:
                return {"ecg", "fs", "n_samples"}.issubset(data.files)
        except (OSError, ValueError):
            return False
    return True


def validate_rpeaks(sid, path):
    """Check the identity and indices of an already reviewed R-peak file."""
    with np.load(path, allow_pickle=False) as peaks:
        r = peaks["r_peaks"]
        valid = (
            str(peaks["subject_id"].item()) == sid
            and int(peaks["fs"]) == 500
            and r.ndim == 1 and r.size >= 2
            and np.issubdtype(r.dtype, np.integer)
            and np.all(np.diff(r) > 0)
            and np.all((r >= 0) & (r < int(peaks["n_samples"])))
        )
        if not valid:
            raise ValueError(f"{sid}: corrected R-peak identity, sampling rate or indices are invalid.")


def plan_epoch_preparation(ids, paths, *, rebuild=False, align_missing=True, log=None):
    """Use CSV order, reuse saved stages, and report missing source files."""
    jobs = {"align": [], "regress": [], "epochs": []}
    reused = missing_count = 0
    for sid in ids:
        epoch = paths["hepEcgRegOut"] / f"{sid}_hep_epochs.mat"
        if ready_file(epoch) and not rebuild:
            reused += 1
            continue
        regressed = paths["preprocessedEcgRegOut"] / f"{sid}_continuous_clean_ecg_regressed.mat"
        aligned = paths["ecgTrimmedPath"] / f"{sid}_ecg_trimmed.npz"
        need_regression = rebuild or not ready_file(regressed)
        need_alignment = align_missing and need_regression and (rebuild or not ready_file(aligned))
        required = [paths["correctedRpeakPath"] / f"{sid}_movie_rpeaks.npz"]
        if need_regression:
            required.append(paths["preprocessedOut"] / f"{sid}_continuous_clean.mat")
            if not align_missing:
                required.append(aligned)
        if need_alignment:
            required.append(paths["ecgExtractedPath"] / f"{sid}_movie_ecg.npz")
        missing = [p.name for p in required if not p.is_file() or p.stat().st_size == 0]
        if missing:
            print(f"Missing: {sid} ({', '.join(missing)}); skipped")
            missing_count += 1
            continue
        validate_rpeaks(sid, paths["correctedRpeakPath"] / f"{sid}_movie_rpeaks.npz")
        if need_alignment:
            jobs["align"].append(sid)
        if need_regression:
            jobs["regress"].append(sid)
        jobs["epochs"].append(sid)
    summary = f"Existing epochs: {reused}; to prepare: {len(jobs['epochs'])}; Missing: {missing_count}"
    print(summary)
    if log is not None:
        log.write(summary)
    return jobs


def enough_output_space(jobs, paths, log=None, root=None):
    """Check conservative output sizes before a potentially large batch."""
    required = {"preprocessedEcgRegOut": 0, "hepEcgRegOut": 0}
    for sid in jobs["regress"]:
        with h5py.File(paths["preprocessedOut"] / f"{sid}_continuous_clean.mat", "r") as source:
            data = source["EEG/data"]
            required["preprocessedEcgRegOut"] += data.size * data.dtype.itemsize * 1.3
    for sid in jobs["epochs"]:
        with np.load(paths["correctedRpeakPath"] / f"{sid}_movie_rpeaks.npz", allow_pickle=False) as peaks:
            required["hepEcgRegOut"] += peaks["r_peaks"].size * 400 * 63 * 4 * 2 * 1.2
    volumes = {}
    for key, size in required.items():
        if not size:
            continue
        directory = paths[key]
        while not directory.exists():
            directory = directory.parent
        device = directory.stat().st_dev
        volumes.setdefault(device, [directory, 0])[1] += size
    sufficient = True
    for directory, size in volumes.values():
        free = shutil.disk_usage(directory).free
        location = shown(directory, root) if root else str(directory)
        if log is not None:
            log.write(f"Storage estimate: up to {size / 1024**3:.1f} GiB for new outputs; "
                      f"{free / 1024**3:.1f} GiB available ({location})")
        if free < size + 1024**3:
            sufficient = False
            report(f"Signal preparation skipped: new outputs need up to {size / 1024**3:.1f} GiB "
                   f"but {free / 1024**3:.1f} GiB is free for {location}. Rerun step 1 and "
                   "choose a larger batch output folder. Existing files are retained.", log)
    return sufficient


def epoch_summary(ids, paths, log=None, root=None):
    """Show the retained beats and saved files for a single-subject demonstration."""
    rows = []
    for sid in ids:
        epoch = paths["hepEcgRegOut"] / f"{sid}_hep_epochs.mat"
        if not ready_file(epoch):
            report(f"Missing: {epoch}; demonstration summary skipped.", log)
            continue
        with h5py.File(epoch, "r") as source:
            counts = {condition: int(source[f"qc_hep/n_kept_{condition}"][()].item())
                      for condition in ("fear", "neutral", "happy")}
        location = shown(epoch, root) if root else str(epoch)
        rows.append({"ID": sid, **counts, "Clean epochs": location})
        report(f"{sid}: retained beats {counts}; epochs saved to {location}", log)
    return rows


def compare_manuscript_averages(batch, manuscript, log=None):
    """Report numerical agreement without replacing the manuscript reference."""
    with np.load(batch.channel_dir / "chan_means.npz", allow_pickle=False) as rebuilt, \
            np.load(manuscript.channel_dir / "chan_means.npz", allow_pickle=False) as reference:
        if not np.array_equal(rebuilt["ids"], reference["ids"]):
            report("Manuscript comparison: participant order differs.", log)
            return False
        difference = max(float(np.max(np.abs(rebuilt[c] - reference[c])))
                         for c in ("fear", "neutral", "happy"))
        counts_match = np.array_equal(rebuilt["n_trials"], reference["n_trials"])
    report(f"Manuscript comparison: maximum channel-mean difference {difference:.6g} µV; "
           f"retained-trial counts {'match' if counts_match else 'differ'}.", log)
    return counts_match and difference < 1e-5


def run_alignment(ids, paths, *, root, output_root, log=None):
    if not ids:
        print("ECG alignment: no files to generate.")
        return
    run_logged(
        [sys.executable, "-u", Path(root) / "04_HEP/preparation/export_aligned_ecg.py",
         "--ecg-dir", paths["ecgExtractedPath"], "--preprocessed", paths["preprocessedOut"],
         "--output-dir", paths["ecgTrimmedPath"], "--overwrite", "--ids", *ids],
        "align_ecg", root=root, output_root=output_root, log=log,
    )
    for sid in ids:
        output = paths["ecgTrimmedPath"] / f"{sid}_ecg_trimmed.npz"
        if not ready_file(output):
            raise RuntimeError(f"ECG alignment did not create a valid output: {output}")
    if log is not None:
        log.write(f"ECG alignment complete: {len(ids)} subjects")


def run_matlab_preparation(function, ids, paths, *, root, output_root, executable=None, log=None):
    output_key, suffix = {
        "regress_ecg_from_eeg": ("preprocessedEcgRegOut", "continuous_clean_ecg_regressed"),
        "extract_hep_epochs": ("hepEcgRegOut", "hep_epochs"),
    }[function]
    if not ids:
        print(f"{function}: no files to generate.")
        return
    config = "CFG=preprocessing_config(); " + " ".join(
        f"CFG.{key}={matlab_quote(value)};" for key, value in paths.items()
    )
    subjects = ",".join(matlab_quote(sid) for sid in ids)
    command = config + f" ids={{{subjects}}}; for k=1:numel(ids), {function}(ids{{k}}, CFG); end"
    run_matlab(command, function, root=root, output_root=output_root, executable=executable, log=log)
    for sid in ids:
        output = paths[output_key] / f"{sid}_{suffix}.mat"
        if not ready_file(output):
            raise RuntimeError(f"{function} did not create a valid output: {output}")
    if log is not None:
        log.write(f"{function} complete: {len(ids)} subjects")


def prepare_aggregates(paths, *, recompute=False, executable=None, log=None):
    """Read or build averages; validate replacements before publishing them."""
    check_inputs = import_module("04_HEP.preparation.check_hep_inputs").check_inputs
    if recompute:
        _, missing = missing_epochs(paths)
        if missing:
            raise ValueError(f"Cannot replace averages: missing epochs for {len(missing)} participants.")
        paths.channel_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".hep_averages_", dir=paths.channel_dir.parent) as temporary:
            staging = Path(temporary)
            staged_paths = replace(paths, roi=staging / paths.roi.name, channel_dir=staging)
            prepare_aggregates(staged_paths, executable=executable, log=log)
            manifest_file = staging / "export_manifest.json"
            manifest = json.loads(manifest_file.read_text())
            manifest["output_dir"] = str(paths.channel_dir)
            manifest["roi_check"]["reference"] = str(paths.roi)
            manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
            replacements = [(staged_paths.roi, paths.roi)] + [
                (staging / name, paths.channel_dir / name)
                for name in ("chan_means.npz", "export_manifest.json")
            ]
            for source, target in replacements:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))
        report("Replaced ROI and channel averages after validation.", log)
        return check_inputs(paths)
    checked = check_inputs(paths)
    generated = False
    if not paths.roi.is_file():
        command = (
            f"compute_hep_roi_waveforms({matlab_quote(paths.epochs)}, "
            f"{matlab_quote(paths.roi)}, {matlab_quote(paths.participants)})"
        )
        run_matlab(command, "ROI waveforms", root=paths.root, output_root=paths.output,
                   executable=executable, log=log)
        if not paths.roi.is_file():
            raise FileNotFoundError(paths.roi)
        generated = True
    if not (paths.channel_dir / "chan_means.npz").is_file():
        run_logged(
            [sys.executable, "-u", paths.root / "04_HEP/preparation/export_hep_channel_means.py",
             "--input-dir", paths.epochs, "--output-dir", paths.channel_dir,
             "--info-csv", paths.participants,
             "--config", paths.root / "01_preprocessing/config/preprocessing_config.m",
             "--roi-reference", paths.roi],
            "Channel means", root=paths.root, output_root=paths.output, log=log,
        )
        generated = True
    return check_inputs(paths) if generated else checked
