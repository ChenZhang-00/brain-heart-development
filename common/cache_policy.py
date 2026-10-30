"""Reuse-or-recompute policy shared by the analysis notebooks."""

from pathlib import Path
import hashlib
import json


def choose_recompute(prompt=None, *, heading=None, note=None):
    """Ask once whether to reuse complete saved results or recompute them."""
    if heading:
        print(heading)
    print("1 = use complete saved results; compute missing or outdated steps (default)")
    print("2 = recompute and replace saved results and figures")
    print(note or "Choose 2 after changing the input data or analysis settings.")
    ask = input if prompt is None else prompt
    while True:
        answer = ask("Type 1 or 2, then press Enter (Enter alone = 1): ").strip() or "1"
        if answer in {"1", "2"}:
            return answer == "2"
        print("Enter 1 to use saved results or 2 to recompute.")


def use_cached_results(files, *, recompute, label, log, inputs=(), manifest=None, status=None):
    """Refresh an output group if incomplete, failed, outdated, or built from other inputs."""
    files = [Path(path) for path in files]
    missing = [p.name for p in files if not p.is_file() or p.stat().st_size == 0]
    oldest = min((p.stat().st_mtime_ns for p in files), default=0) if not missing else 0
    newer_inputs = any(Path(p).is_file() and Path(p).stat().st_mtime_ns > oldest for p in inputs)
    changed_source = manifest is not None and not inputs_match_manifest(manifest, inputs, log.root)
    incomplete = False
    if status is not None and Path(status).is_file():
        try:
            incomplete = json.loads(Path(status).read_text()).get("state") != "complete"
        except ValueError:
            incomplete = True
    reuse = not (recompute or missing or newer_inputs or changed_source or incomplete)
    if reuse:
        message = f"{label}: using saved results."
    elif recompute:
        message = f"{label}: recomputing and replacing saved results."
    elif missing:
        message = f"{label}: incomplete saved results ({', '.join(missing[:4])}); computing this step."
    elif incomplete:
        message = f"{label}: the previous run did not finish; computing this step."
    else:
        message = f"{label}: upstream inputs changed; recomputing this step."
    print(message)
    log.write(message)
    return reuse


def figure_needs_update(figure, inputs):
    """Redraw when the figure is missing or any of its inputs is newer."""
    figure = Path(figure)
    if not figure.is_file() or figure.stat().st_size == 0:
        return True
    return any(Path(p).is_file() and Path(p).stat().st_mtime_ns > figure.stat().st_mtime_ns
               for p in inputs)


def inputs_match_manifest(manifest, inputs, root):
    """Prevent reusing statistics computed from a different input set."""
    try:
        records = json.loads(Path(manifest).read_text())["inputs"]
        expected = {(Path(root) / item["file"]).resolve(): item["sha256"] for item in records}
        for path in inputs:
            path = Path(path).resolve()
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if expected.get(path) != digest:
                return False
    except (OSError, KeyError, ValueError):
        return False
    return True
