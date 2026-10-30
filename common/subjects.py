"""Participant ID validation shared by preprocessing and the analyses."""

import csv
import re
from pathlib import Path


def validate_ids(ids):
    if not ids or any(not sid for sid in ids):
        raise ValueError("Provide at least one ID; blank IDs are not allowed.")
    if len(ids) != len(set(ids)):
        raise ValueError("Participant IDs must be unique.")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", sid) for sid in ids):
        raise ValueError("IDs may contain letters, numbers, hyphens and underscores.")
    return ids


def read_subject_ids(info_file):
    """Read the ID column in table order, preserving leading zeros."""
    with Path(info_file).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "ID" not in reader.fieldnames:
            raise ValueError("The participant CSV must contain an ID column.")
        ids = [(row.get("ID") or "").strip() for row in reader]
    return validate_ids(ids)
