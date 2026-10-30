"""Choose whether a processing step should replace its existing results."""

from pathlib import Path


def has_output(path):
    """Check for a nonempty result file without reading its contents."""
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


def choose_overwrite(outputs, label, *, replace_label="recompute and replace", prompt=None):
    """Ask once per step; missing and empty outputs do not need a choice."""
    existing = [Path(p) for p in outputs if has_output(p)]
    if not existing:
        return False
    print(f"\n{label}: {len(existing)} existing result file(s)", flush=True)
    for path in existing[:5]:
        print(" ", path.name)
    if len(existing) > 5:
        print(f"  ... and {len(existing) - 5} more")
    print("1 = skip existing results; create only missing results (default)")
    print(f"2 = {replace_label}")
    ask = input if prompt is None else prompt
    while True:
        answer = ask("Type 1 or 2, then press Enter (Enter alone = 1): ").strip() or "1"
        if answer in {"1", "2"}:
            return answer == "2"
        print("Enter 1 to skip or 2 to replace.")
