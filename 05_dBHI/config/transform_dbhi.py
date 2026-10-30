"""Log10 dBHI indices before any averaging; floor heart-to-brain values first.

The fixed floor and its original 147-child, subsampled estimation are recorded
in sdg/htb_floor.json."""

import json
from pathlib import Path

import numpy as np

_FLOOR_FILE = Path(__file__).resolve().parents[1] / "sdg" / "htb_floor.json"
HTB_FLOOR = float(json.loads(_FLOOR_FILE.read_text(encoding="utf-8"))["htb_floor"])
LOG_BTH = True
LOG_HTB = True
TRANSFORM_ORDER = "per electrode: (htb floor) -> log10 -> ROI spatial mean -> temporal aggregation"


def transform(values, direction):
    """log10 of bth; floor+log10 of htb. NaN is preserved. Raises on infinite or nonpositive bth."""
    x = np.asarray(values, dtype=float)
    if np.isinf(x).any():
        raise ValueError("Infinite coupling values.")
    if direction == "bth":
        if np.any(x <= 0):
            raise ValueError("Nonpositive brain-to-heart values cannot be log10-transformed.")
        return np.log10(x)
    if direction == "htb":
        return np.log10(np.where(np.isnan(x), np.nan, np.maximum(x, HTB_FLOOR)))
    raise ValueError(f"unknown direction {direction!r}")


def params():
    return {
        "log10_bth": LOG_BTH,
        "log10_htb": LOG_HTB,
        "htb_floor": HTB_FLOOR,
        "transform_order": TRANSFORM_ORDER,
    }
