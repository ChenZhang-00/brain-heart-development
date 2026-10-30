"""One concise log for a preprocessing notebook run."""

from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from common.paths import relative_path
from time import monotonic
import os
import re
import subprocess
import sys
import traceback
import warnings


class RunLog:
    def __init__(self, directory, root, *, title="Preprocessing", announce=True):
        self.root = Path(root).resolve()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        number = 1
        while True:
            suffix = "" if number == 1 else f"_{number}"
            self.path = directory / f"{stamp}{suffix}.log"
            try:
                with self.path.open("x", encoding="utf-8") as stream:
                    stream.write(f"{title} | {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                break
            except FileExistsError:
                number += 1
        self._warnings = Counter()
        self._parameters = set()
        self.write(f"Python {sys.version.split()[0]}: {sys.executable}")
        if announce:
            print("Run log:", self.display(self.path))

    def display(self, path):
        """Show paths inside the repository relative to its root."""
        return relative_path(path, self.root)

    def write(self, message):
        text = str(message).replace(str(self.root) + os.sep, "")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(text.rstrip() + "\n")

    def warning(self, message):
        self._warnings[message] += 1
        if self._warnings[message] == 1:
            self.write(message)

    @contextmanager
    def step(self, label):
        """Record status, warnings and errors without hiding notebook output."""
        self.write(f"\n[{datetime.now():%H:%M:%S}] {label}")
        self._warnings.clear()
        self._parameters.clear()
        started = monotonic()
        with warnings.catch_warnings():
            show_warning = warnings.showwarning

            def record_warning(message, category, *args, **kwargs):
                self.warning(f"{category.__name__}: {message}")
                show_warning(message, category, *args, **kwargs)

            warnings.showwarning = record_warning
            try:
                yield
            except BaseException as error:
                state = "Interrupted" if isinstance(error, KeyboardInterrupt) else "Failed"
                self.write(f"{state} after {monotonic() - started:.1f} s")
                self.write("".join(traceback.format_exception(error, limit=5)))
                raise
            else:
                self.write(f"Step finished ({monotonic() - started:.1f} s)")
            finally:
                repeats = sum(self._warnings.values()) - len(self._warnings)
                if repeats:
                    self.write(f"Repeated warnings omitted: {repeats}")

    def command(self, args, *, env=None, completion_marker=None):
        """Stream full output to the notebook; retain summaries and failures."""
        tail = deque(maxlen=30)
        packages = []
        warning_lines = []
        complete = completion_marker is None

        def save_warning():
            self.warning(" ".join(warning_lines).removeprefix("[").removesuffix("]"))
            warning_lines.clear()

        with subprocess.Popen(
            [str(arg) for arg in args], cwd=self.root, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        ) as process:
            try:
                for line in process.stdout:
                    print(line.replace(str(self.root) + os.sep, ""), end="", flush=True)
                    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line)
                    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text).strip()
                    if not text:
                        continue
                    tail.append(text[-3000:])
                    if text == completion_marker:
                        complete = True
                    elif warning_lines or text.lower().startswith("[warning:"):
                        warning_lines.append(text)
                        if text.endswith("]"):
                            save_warning()
                    elif "[summary] " in text:
                        message = text.split("[summary] ", 1)[1]
                        if not message.startswith("Parameters:") or message not in self._parameters:
                            self.write(message)
                            self._parameters.add(message)
                    elif text.startswith("[OK] "):
                        packages.append(text[5:])
                    elif re.search(r"\b\w*warning:", text, re.IGNORECASE):
                        self.warning(text)
                    elif re.search(
                        r"\b(error|exception|failed|failure)\b|\w+(Error|Exception):|\[FAIL\]",
                        text, re.IGNORECASE,
                    ):
                        self.write(text)
                status = process.wait()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                self.write("Last console output:\n" + "\n".join(tail))
                raise
        if warning_lines:
            save_warning()
        if packages:
            self.write("Packages: " + "; ".join(packages))
        if status != 0 or not complete:
            self.write("Last console output:\n" + "\n".join(tail))
            reason = f"exit status {status}" if status else "completion marker missing"
            raise RuntimeError(f"External command failed ({reason}). See {self.path}")
