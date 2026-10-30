"""Run external preparation commands in the notebook's shared log."""

from importlib import import_module
from pathlib import Path


def run_logged(args, label, *, root, output_root, log=None, completion_marker=None):
    if log is None:
        RunLog = import_module("common.run_log").RunLog
        log = RunLog(Path(output_root) / "logs", root, title="HEP")
        with log.step(label):
            log.command(args, completion_marker=completion_marker)
    else:
        log.command(args, completion_marker=completion_marker)
    return log.path


def matlab_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def find_matlab(executable=None):
    import os
    import shutil
    import sys

    if executable:
        return str(Path(executable).expanduser())
    found = shutil.which("matlab")
    if found:
        return found
    if sys.platform == "darwin":
        candidates = sorted(Path("/Applications").glob("MATLAB_R*.app/bin/matlab"))
    elif os.name == "nt":
        candidates = sorted(
            Path(os.environ.get("ProgramFiles", "C:/Program Files")).glob(
                "MATLAB/R*/bin/matlab.exe"
            )
        )
    else:
        candidates = []
    if candidates:
        return str(candidates[-1])
    raise FileNotFoundError("MATLAB was not found. Set MATLAB_EXE in notebook step 1.")


def run_matlab(command, label, *, root, output_root, executable=None, log=None):
    import os

    root = Path(root)
    prefix = (
        f"addpath({matlab_quote(root / '01_preprocessing/config')}); "
        f"addpath({matlab_quote(root / '04_HEP/preparation')}); "
    )
    args = [find_matlab(executable)]
    if os.name == "nt":
        args.append("-wait")
    args += ["-sd", str(root), "-batch", prefix + command + "; disp('DBH_HEP_STEP_COMPLETE');"]
    return run_logged(
        args, label, root=root, output_root=output_root, log=log,
        completion_marker="DBH_HEP_STEP_COMPLETE",
    )
