"""Check existing HEP aggregates or the epoch files needed to generate them."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from importlib import import_module
parse_args = import_module("04_HEP.config.hep_config").parse_args
_inputs = import_module("04_HEP.analysis.inputs")
check_epoch_files = _inputs.check_epoch_files
load_channels = _inputs.load_channels
load_metadata = _inputs.load_metadata
load_roi = _inputs.load_roi
shown = import_module("common.paths").relative_path


def check_inputs(paths):
    metadata = load_metadata(paths)
    roi_exists = paths.roi.is_file()
    cache_exists = (paths.channel_dir / "chan_means.npz").is_file()
    roi = channels = None
    if roi_exists:
        roi = load_roi(paths)
        print(f"Loaded ROI averages: {shown(paths.roi, paths.root)}")
    if cache_exists and roi_exists:
        channels = load_channels(paths)
        print(f"Loaded channel averages: {shown(paths.channel_dir / 'chan_means.npz', paths.root)}")
    if not roi_exists or not cache_exists:
        check_epoch_files(paths, metadata.ID.tolist())
        print("All 148 ECG-regressed epoch files are available for the missing preparation steps.")
    if not roi_exists:
        print(
            "ROI generation requires MATLAB. Configure MATLAB_EXE in the entry notebook if needed."
        )
    print("Input validation passed.")
    return {"participants": metadata, "roi": roi, "channels": channels}


def main(argv=None):
    paths, _ = parse_args(__doc__, argv)
    check_inputs(paths)


if __name__ == "__main__":
    main()
