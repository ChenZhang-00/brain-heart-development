"""Check the active Python environment and source inputs without running analyses."""

import argparse
import csv
import importlib
import importlib.metadata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "statsmodels": "statsmodels",
    "h5py": "h5py",
    "joblib": "joblib",
    "numba": "numba",
    "llvmlite": "llvmlite",
    "tqdm": "tqdm",
    "mne": "mne",
    "neurokit2": "neurokit2",
    "IPython": "ipython",
    "ipykernel": "ipykernel",
    "jupyterlab": "jupyterlab",
    "nbformat": "nbformat",
    "tkinter": None,
}


def read_table(relative_path, required):
    path = ROOT / relative_path
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{relative_path}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{relative_path}: table is empty")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-only", action="store_true", help="Skip private input checks")
    parser.add_argument("--quiet", action="store_true", help="One line of package versions; failures only otherwise")
    args = parser.parse_args()
    failures = 0
    print(f"Python {sys.version.split()[0]}: {sys.executable}")
    if sys.version_info[:2] != (3, 12):
        print("[FAIL] Use Python 3.12 from environment.yml.")
        failures += 1
    available = []
    for module, distribution in PACKAGES.items():
        try:
            importlib.import_module(module)
            version = importlib.metadata.version(distribution) if distribution else "available"
            available.append(f"{module} {version}")
            if not args.quiet:
                print(f"[OK] {module}: {version}")
        except Exception as exc:
            print(f"[FAIL] {module}: {type(exc).__name__}: {exc}")
            failures += 1
    if args.quiet:
        print("[OK] " + "; ".join(available))

    if not args.python_only:
        try:
            required = {"ID", "Gender", "Age_month", "Age_group"}
            required.update(f"{c}_{s}" for c in ("fear", "happy", "neutral") for s in ("A", "V"))
            rows = read_table("00_data/participants.csv", required)
            ids = [row["ID"].strip() for row in rows]
            if "" in ids or len(ids) != len(set(ids)):
                raise ValueError("Participant IDs must be unique and nonempty.")
            missing = sum(
                not (ROOT / "00_data/raw_data" / f"{sid}_movie.{ext}").is_file()
                for sid in ids
                for ext in ("vhdr", "eeg", "vmrk")
            )
            if missing:
                raise ValueError(f"00_data/raw_data/: {missing} BrainVision companion files missing")
            print(f"[OK] Metadata and BrainVision filenames: {len(ids)} participants")
        except (OSError, ValueError) as exc:
            print(f"[FAIL] Source inputs: {exc}")
            failures += 1
        try:
            read_table(
                "00_data/events/Movies_rating_by_RA.csv",
                {"Time", "Fear_Intensity", "Happy_Intensity"},
            )
            print("[OK] Movie intensity input")
        except (OSError, ValueError) as exc:
            print(f"[FAIL] Movie intensity input: {exc}")
            failures += 1

    if failures:
        print(f"{failures} setup check(s) failed. See README.md before running the notebooks.")
    elif not args.quiet:
        print("MATLAB, its plugins, recording contents and downstream analyses were not tested.")
        print("Requested setup checks passed. Follow the analysis order in the READMEs.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
