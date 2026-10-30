"""Workflow prompts and saved-result reuse for the guided dBHI notebook."""

from importlib import import_module
from pathlib import Path
import json

_shared = import_module("common.cache_policy")


def choose_workflow(root, default_cache, *, prompt=None, log=None):
    """Analyse the supplied model arrays, or estimate the model for selected children."""
    selection = __import__("importlib").import_module("common.subjects")
    paths_module = __import__("importlib").import_module("common.paths")
    print("1 = analyse the supplied 148-child model arrays (default)")
    print("2 = estimate the SDG model from cleaned EEG and reviewed R peaks, then analyse")
    ask = input if prompt is None else prompt
    while True:
        choice = ask("Type 1 or 2, then press Enter (Enter alone = 1): ").strip() or "1"
        if choice in {"1", "2"}:
            break
        print("Enter 1 to analyse the supplied arrays or 2 to estimate the model.")
    if choice == "1":
        return "analyse", None, Path(default_cache)
    print("Participants: Enter alone = every ID in the participant table (full cohort);")
    print("or type one ID for the included example (sub-001) or several IDs separated by commas.")
    while True:
        value = ask("Participant IDs (Enter alone = all): ").strip()
        if not value:
            ids = None
            break
        try:
            ids = selection.validate_ids([v.strip() for v in value.split(",") if v.strip()])
            break
        except ValueError as error:
            print(error)
    default_new = Path(root) / ("05_dBHI/outputs/batch/model" if ids is None else "05_dBHI/outputs/demo/model")
    print(f"New estimates are written to {_shown(default_new, root)} "
          "(the supplied arrays in 05_dBHI/outputs/manuscript/model/ are never replaced).")
    while True:
        value = ask("Press Enter to use this folder, or type another folder: ").strip()
        cache = paths_module.resolve_path(value or str(default_new), root)
        supplied = Path(default_cache).resolve()
        if cache.resolve() == supplied or supplied in cache.resolve().parents:
            print("Choose a folder outside the supplied model cache.")
            continue
        if not _same_fingerprint(cache):
            print(f"{_shown(cache, root)} holds estimates made with different model code or package versions;")
            print("they cannot be resumed.")
            print("1 = replace them with new estimates (default)")
            print("2 = type another folder")
            answer = ask("Type 1 or 2, then press Enter (Enter alone = 1): ").strip() or "1"
            if answer != "1":
                continue
            _clear_estimates(cache)
            if log is not None:
                log.write(f"Replaced estimates with a different fingerprint in {cache}")
        else:
            existing = sorted(p.name.removesuffix("_dbhi.npz") for p in Path(cache).glob("*_dbhi.npz"))
            if existing:
                print(f"{_shown(cache, root)} already holds {len(existing)} valid estimate(s): "
                      + ", ".join(existing[:5]) + (" …" if len(existing) > 5 else ""))
                print("1 = keep them and estimate only missing children (default)")
                print("2 = re-estimate and replace them")
                answer = ask("Type 1 or 2, then press Enter (Enter alone = 1): ").strip() or "1"
                if answer == "2":
                    _clear_estimates(cache)
                    if log is not None:
                        log.write(f"Replaced {len(existing)} existing estimates in {cache}")
                elif log is not None:
                    log.write(f"Resuming {len(existing)} existing estimates in {cache}")
        break
    return "estimate", ids, cache


def _same_fingerprint(cache):
    """True when the folder is empty or its manifest matches the current model and environment."""
    from importlib import import_module

    pipeline = import_module("05_dBHI.sdg.pipeline")
    manifest_file = Path(cache) / "_manifest.json"
    if not manifest_file.is_file():
        return not list(Path(cache).glob("*_dbhi.npz"))
    saved = json.loads(manifest_file.read_text(encoding="utf-8"))
    return pipeline.canonical(saved) == pipeline.canonical(pipeline.make_manifest(pipeline.DEFAULT_CONFIG))


def _clear_estimates(cache):
    """Delete a user-owned model folder and its reports; never touches the supplied arrays."""
    import shutil

    cache = Path(cache)
    for folder in (cache, cache.with_name(cache.name + "_reports")):
        if folder.is_dir():
            shutil.rmtree(folder)


def choose_model_source(manuscript, batch, expected, *, root, prompt=None, log=None):
    """Pick the model arrays for the group analyses; incomplete batch estimates cannot be used."""
    counts = {}
    for name, folder in (("manuscript", manuscript), ("batch", batch)):
        folder = Path(folder)
        counts[name] = len(list(folder.glob("sub-*_dbhi.npz"))) if folder.is_dir() else 0
    print("Model arrays for the group analyses (steps 5-13)")
    print(f"1 = manuscript arrays, {_shown(manuscript, root)} ({counts['manuscript']}/{expected} children) (default)")
    usable = counts["batch"] == expected
    note = "" if usable else f" — incomplete, cannot be used; choose 1"
    print(f"2 = your own estimates, {_shown(batch, root)} ({counts['batch']}/{expected} children){note}")
    ask = input if prompt is None else prompt
    while True:
        answer = ask("Type 1 or 2, then press Enter (Enter alone = 1): ").strip() or "1"
        if answer == "1":
            choice = "manuscript"
            break
        if answer == "2" and usable:
            choice = "batch"
            break
        if answer == "2":
            print(f"Only {counts['batch']} of {expected} children are estimated; the group analyses need all "
                  f"{expected}. Choose 1 for the manuscript arrays.")
            continue
        print("Enter 1 or 2.")
    message = f"Group analyses use the {choice} model arrays: {_shown(manuscript if choice == 'manuscript' else batch, root)}"
    print(message)
    if log is not None:
        log.write(message)
    return choice


def choose_branch(name, steps, *, prompt=None, log=None):
    """Ask whether to run one analysis branch."""
    print(f"Run the {name} branch (steps {steps})?")
    print("1 = yes (default)")
    print("2 = skip")
    ask = input if prompt is None else prompt
    while True:
        answer = ask("Type 1 or 2, then press Enter (Enter alone = 1): ").strip() or "1"
        if answer in {"1", "2"}:
            run = answer == "1"
            if log is not None:
                log.write(f"{name.capitalize()} branch: {'run' if run else 'skipped'}")
            return run
        print("Enter 1 to run or 2 to skip.")


def choose_recompute(prompt=None):
    return _shared.choose_recompute(
        prompt,
        heading="Saved results of steps 5-13 (statistics, tables and figures)",
        note="Choose 2 after changing the model arrays, the participant table or the ratings.",
    )


use_cached_results = _shared.use_cached_results


def _shown(path, root):
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)
