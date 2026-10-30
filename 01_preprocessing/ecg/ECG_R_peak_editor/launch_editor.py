#!/usr/bin/env python3
"""Desktop launcher for the shared R-peak editor."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import traceback
from pathlib import Path

if os.environ.get("DBH_EDITOR_STDOUT") != "1":
    _LOG_PATH = Path(os.path.expanduser("~")) / "RpeakEditor.log"
    _log_fh = open(_LOG_PATH, "a", buffering=1, encoding="utf-8")
    sys.stdout = _log_fh
    sys.stderr = _log_fh
print("\n=========================================")
print(f"RpeakEditor launched at {datetime.datetime.now()}")
print("=========================================")

CONFIG_PATH = Path(os.path.expanduser("~")) / ".rpeak_editor_config.json"

def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[config] save failed: {e}")

import tkinter as tk
from tkinter import ttk, filedialog

COLOR_ACCENT = "#E63946"   # primary accent / Open button
COLOR_ERR    = "#8B0000"   # errors  (= COLOR_R_EDGE in editor)
COLOR_OK     = "#16A34A"   # success state
COLOR_INFO   = "#6A4C93"   # resume prompt heading (= COLOR_S in editor)
COLOR_TEXT   = "#2C3E50"   # primary text (= COLOR_ECG in editor)
COLOR_MUTE   = "#495057"   # secondary text (= COLOR_TEXT_MUTE in editor)
COLOR_BORDER = "#E9ECEF"   # card border (= COLOR_GRID in editor)
COLOR_BG     = "#FAFAFA"   # page background (= COLOR_BG in editor)
COLOR_PANEL  = "#FFFFFF"   # card surface (= COLOR_AX_BG in editor)

FONT_SANS = "Helvetica Neue"
FONT_MONO = "Menlo"

class LauncherApp:
    def __init__(self, raw_dir=None, out_dir=None, subject_id="") -> None:
        self.cfg = load_config()
        if raw_dir is not None:
            self.cfg["raw_dir"] = str(Path(raw_dir).expanduser().resolve())
        if out_dir is not None:
            self.cfg["out_dir"] = str(Path(out_dir).expanduser().resolve())
        self.subject_id = subject_id
        self.root = tk.Tk()
        self.root.title("ECG R-peak Editor")
        self.root.configure(bg=COLOR_BG)
        self.root.minsize(820, 580)

        w, h = 880, 620
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = max(0, (sw - w) // 2)
        y = max(0, sh // 3 - h // 3)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        self._busy = False
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_clicked)

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("aqua")
        except tk.TclError:
            pass
        style.configure("TButton", padding=(10, 4))
        style.configure("Open.TButton", padding=(14, 6))

        header = tk.Frame(self.root, bg=COLOR_BG)
        header.pack(fill="x", padx=20, pady=(18, 0))
        tk.Label(
            header, text="ECG R-peak Editor",
            font=(FONT_SANS, 22, "bold"),
            fg=COLOR_TEXT, bg=COLOR_BG,
        ).pack(anchor="w")
        tk.Label(
            header, text="人工矫正心电 R-peak 位置",
            font=(FONT_SANS, 11),
            fg=COLOR_MUTE, bg=COLOR_BG,
        ).pack(anchor="w", pady=(2, 0))
        tk.Frame(self.root, bg=COLOR_ACCENT, height=2).pack(
            fill="x", padx=20, pady=(10, 14)
        )

        form = tk.Frame(self.root, bg=COLOR_BG)
        form.pack(fill="x", padx=20)
        form.grid_columnconfigure(1, weight=1)

        def label(text, row):
            tk.Label(
                form, text=text, bg=COLOR_BG, fg=COLOR_TEXT,
                font=(FONT_SANS, 11), anchor="w",
            ).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=6)

        label("INPUT folder", 0)
        self.input_var = tk.StringVar(value=self.cfg.get("raw_dir", ""))
        self.input_entry = ttk.Entry(form, textvariable=self.input_var)
        self.input_entry.grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Button(
            form, text="Browse…",
            command=lambda: self._browse_into(self.input_var),
        ).grid(row=0, column=2, sticky="ew", padx=(8, 0), pady=6)

        label("OUTPUT folder", 1)
        self.output_var = tk.StringVar(value=self.cfg.get("out_dir", ""))
        self.output_entry = ttk.Entry(form, textvariable=self.output_var)
        self.output_entry.grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Button(
            form, text="Browse…",
            command=lambda: self._browse_into(self.output_var),
        ).grid(row=1, column=2, sticky="ew", padx=(8, 0), pady=6)

        label("Subject ID", 2)
        self.sid_var = tk.StringVar(value=self.subject_id)
        self.sid_entry = ttk.Entry(form, textvariable=self.sid_var)
        self.sid_entry.grid(row=2, column=1, sticky="ew", pady=(14, 6))
        self.sid_entry.bind("<Return>", lambda e: self._on_open_clicked())
        self.open_btn = ttk.Button(
            form, text="Open Editor", style="Open.TButton",
            command=self._on_open_clicked,
        )
        self.open_btn.grid(row=2, column=2, sticky="ew",
                           padx=(8, 0), pady=(14, 6))

        wrap = tk.Frame(self.root, bg=COLOR_BG)
        wrap.pack(fill="both", expand=True, padx=20, pady=(14, 18))
        self.status_frame = tk.Frame(
            wrap, bg=COLOR_PANEL,
            highlightbackground=COLOR_BORDER, highlightthickness=1,
        )
        self.status_frame.pack(fill="both", expand=True)
        self._set_status("Ready. Enter a subject ID and click Open.", "neutral")

    def _clear_status(self) -> None:
        for widget in self.status_frame.winfo_children():
            widget.destroy()

    def _set_status(self, text: str, kind: str = "neutral") -> None:
        self._clear_status()
        color = {
            "neutral": COLOR_TEXT,
            "muted": COLOR_MUTE,
            "error": COLOR_ERR,
            "success": COLOR_OK,
        }[kind]
        tk.Label(
            self.status_frame, text=text, fg=color, bg=COLOR_PANEL,
            font=(FONT_SANS, 12), wraplength=800, justify="left",
            anchor="nw",
        ).pack(fill="both", expand=True, padx=16, pady=14)

    def _show_resume_prompt(self, sid, info, on_choice) -> None:
        self._clear_status()
        tk.Label(
            self.status_frame,
            text=f"Found previous corrections for subject {sid}",
            fg=COLOR_INFO, bg=COLOR_PANEL,
            font=(FONT_SANS, 13, "bold"),
            anchor="w",
        ).pack(fill="x", padx=16, pady=(14, 6), anchor="w")

        info_text = (
            f"   File      {info['file']}\n"
            f"   Last      {info['date']}\n"
            f"   R-peaks   {info['n_rpeaks']}\n"
            f"   Edits     +{info['n_added']} / -{info['n_deleted']}"
        )
        tk.Label(
            self.status_frame, text=info_text, fg=COLOR_TEXT, bg=COLOR_PANEL,
            font=(FONT_MONO, 11), justify="left", anchor="nw",
        ).pack(fill="x", padx=16, pady=(0, 8), anchor="w")

        tk.Label(
            self.status_frame,
            text="Resume previous = continue editing.   "
                 "Start fresh = re-detect; previous file overwritten on save.",
            fg=COLOR_MUTE, bg=COLOR_PANEL,
            font=(FONT_SANS, 10), wraplength=780,
            justify="left", anchor="nw",
        ).pack(fill="x", padx=16, pady=(0, 12), anchor="w")

        btn_row = tk.Frame(self.status_frame, bg=COLOR_PANEL)
        btn_row.pack(fill="x", padx=16, pady=(0, 14), anchor="w")
        ttk.Button(
            btn_row, text="Resume previous",
            command=lambda: on_choice(True),
        ).pack(side="left", padx=(0, 10))
        ttk.Button(
            btn_row, text="Start fresh",
            command=lambda: on_choice(False),
        ).pack(side="left")

    def _browse_into(self, var: tk.StringVar) -> None:
        cur = var.get().strip()
        initdir = cur if cur and Path(cur).exists() else str(Path.home())
        path = filedialog.askdirectory(
            parent=self.root, mustexist=False, initialdir=initdir,
            title="Choose folder",
        )
        if path:
            var.set(path)

    def _validate(self) -> tuple[Path, Path, str] | None:
        in_str = self.input_var.get().strip()
        out_str = self.output_var.get().strip()
        sid = self.sid_var.get().strip()

        if not in_str:
            self._set_status("INPUT folder is empty.", "error")
            return None
        in_path = Path(os.path.expanduser(in_str)).resolve()
        if not in_path.is_dir():
            self._set_status(
                f"INPUT folder does not exist:\n{in_path}", "error")
            return None
        if not list(in_path.glob("*_movie_ecg.npz")):
            self._set_status(
                f"No *_movie_ecg.npz files found in INPUT:\n{in_path}",
                "error")
            return None

        if not out_str:
            self._set_status("OUTPUT folder is empty.", "error")
            return None
        out_path = Path(os.path.expanduser(out_str)).resolve()
        try:
            out_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self._set_status(
                f"Cannot create OUTPUT folder:\n{out_path}\n{e}", "error")
            return None

        if not sid:
            self._set_status("Please enter a subject ID.", "error")
            return None
        return in_path, out_path, sid

    def _on_open_clicked(self) -> None:
        if self._busy:
            return
        v = self._validate()
        if v is None:
            return
        in_path, out_path, sid = v

        self.cfg["raw_dir"] = str(in_path)
        self.cfg["out_dir"] = str(out_path)
        save_config(self.cfg)

        from edit_r_peaks import resolve_input
        try:
            npz = resolve_input(sid, in_path)
        except FileNotFoundError:
            self._set_status(
                f"Subject not found: '{sid}'\n"
                f"Expected: {sid}_movie_ecg.npz in:\n{in_path}",
                "error")
            return

        self._open_editor(sid, npz, out_path)

    def _open_editor(self, sid, npz: Path, out_path: Path) -> None:
        self._busy = True
        self.open_btn.config(state="disabled")
        self._set_status(f"Opening editor for {sid}…", "muted")
        self.root.update()

        def ask_resume_inline(info: dict) -> bool:
            choice = tk.IntVar(self.root, value=-1)
            self._show_resume_prompt(
                sid, info,
                on_choice=lambda r: choice.set(1 if r else 0),
            )
            self.root.wait_variable(choice)
            return choice.get() == 1

        try:
            print(f"[launcher] opening editor for {sid}")
            from edit_r_peaks import RpeakEditor
            editor = RpeakEditor(
                npz, out_dir=out_path,
                ask_resume=ask_resume_inline,
            )
            self._set_status(
                f"Saved subject {sid}: {len(editor.r_peaks)} R-peaks  "
                f"(+{editor.n_added} / -{editor.n_deleted})",
                "success",
            )
            print(f"[launcher] editor closed for {sid}")
        except Exception as e:
            traceback.print_exc()
            self._set_status(
                f"Editor error opening {sid}:\n{type(e).__name__}: {e}",
                "error",
            )
        finally:
            self._busy = False
            self.open_btn.config(state="normal")
            self.sid_entry.focus_set()
            self.sid_entry.select_range(0, "end")

    def _on_close_clicked(self) -> None:
        if self._busy:
            self._set_status(
                "Editor is open. Press 'q' in the editor to save & close, "
                "then close this window.",
                "muted",
            )
            return
        self.root.destroy()

    def run(self) -> None:
        self.sid_entry.focus_set()
        self.root.mainloop()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, help="Prefill the INPUT folder")
    parser.add_argument("--out-dir", type=Path, help="Prefill the OUTPUT folder")
    parser.add_argument("--subject", default="", help="Prefill the subject ID")
    args = parser.parse_args()
    print("[launcher] starting (single-window mode)")
    LauncherApp(args.raw_dir, args.out_dir, args.subject).run()
    print("[launcher] done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
