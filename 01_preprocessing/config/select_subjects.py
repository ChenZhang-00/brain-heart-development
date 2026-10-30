"""Interactive subject and raw-folder selection for the preprocessing notebook."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.paths import relative_path, resolve_path
from common.subjects import read_subject_ids, validate_ids


def select_subjects(root, prompt=None):
    """Ask for mode, ID or information table, and the raw recording directory; retry on invalid input."""
    root = Path(root).resolve()
    ask = input if prompt is None else prompt
    print("Choose a processing mode:\n1 = single subject\n2 = batch from participant CSV")
    while True:
        choice = ask("Type 1 or 2, then press Enter (Enter alone = 1): ").strip().lower() or "1"
        if choice in ("1", "single", "2", "batch"):
            break
        print("Enter 1 for a single subject or 2 for a batch from CSV.")
    raw_default = root / "00_data/raw_data"

    if choice in ("1", "single"):
        mode = "single"
        while True:
            sid = ask("Subject ID (press Enter to use sub-001): ").strip() or "sub-001"
            try:
                ids = validate_ids([sid])
                break
            except ValueError as error:
                print(f"{error} Try again.")
    else:
        mode = "batch"
        info_default = root / "00_data/participants.csv"
        print(f"\nParticipant information table\nDefault CSV: {relative_path(info_default, root)}")
        while True:
            value = ask("Press Enter to use this CSV, or type another CSV path: ").strip() or str(info_default)
            info_file = resolve_path(value, root)
            if not info_file.is_file():
                print(f"File not found: {relative_path(info_file, root)}. Try again.")
                continue
            try:
                ids = read_subject_ids(info_file)
                break
            except (OSError, ValueError) as error:
                print(f"Cannot use this CSV: {error} Try again.")
        print(f"Participant table: {relative_path(info_file, root)}")
        if (info_file.parent / "raw_data").is_dir():
            raw_default = info_file.parent / "raw_data"

    print(f"\nRaw recording folder\nDefault folder: {relative_path(raw_default, root)}")
    while True:
        value = ask("Press Enter to use this folder, or type another folder path: ").strip() or str(raw_default)
        raw_dir = resolve_path(value, root)
        if raw_dir.is_dir():
            break
        print(f"Folder not found: {relative_path(raw_dir, root)}. Try again.")
    print(f"Mode: {mode}; selected subjects: {len(ids)}")
    print(f"Raw recording folder: {relative_path(raw_dir, root)}")
    return mode, ids, raw_dir
