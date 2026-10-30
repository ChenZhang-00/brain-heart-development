#!/usr/bin/env python3
"""Interactive R-peak correction; see README for keyboard controls."""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass
# Disable matplotlib's default keymaps that conflict with our editor.
for _k in ("save", "back", "forward", "quit", "quit_all"):
    matplotlib.rcParams[f"keymap.{_k}"] = []
# Fonts fall back to DejaVu on other platforms.
FONT_SANS = "Helvetica Neue"
FONT_MONO = "Menlo"
matplotlib.rcParams["font.family"] = FONT_SANS
matplotlib.rcParams["font.sans-serif"] = [FONT_SANS, "Helvetica", "Arial", "DejaVu Sans"]
matplotlib.rcParams["font.monospace"] = [FONT_MONO, "Monaco", "DejaVu Sans Mono", "Courier New"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.labelcolor"] = "#343A40"
matplotlib.rcParams["xtick.color"] = "#495057"
matplotlib.rcParams["ytick.color"] = "#495057"
matplotlib.rcParams["lines.solid_capstyle"] = "round"
matplotlib.rcParams["lines.antialiased"] = True
matplotlib.rcParams["text.antialiased"] = True
matplotlib.rcParams["figure.dpi"] = 110
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, find_peaks

HERE = Path(__file__).resolve().parent
# Defaults if caller does not supply explicit dirs (useful for dev / CLI).
DEFAULT_RAW_DIR = HERE.parents[2] / "01_preprocessing/outputs/demo/ecg/extracted"
DEFAULT_OUT_DIR = HERE.parents[2] / "01_preprocessing/outputs/demo/ecg/r_peaks"

# visual design
COLOR_BG = "#FAFAFA"  # figure background
COLOR_AX_BG = "#FFFFFF"  # axis background
COLOR_ECG = "#2C3E50"  # ECG trace
COLOR_R = "#E63946"  # R-peak fill
COLOR_R_EDGE = "#8B0000"  # R-peak outline
COLOR_S = "#6A4C93"  # S-marker vertical line & label
COLOR_WIN_HL = "#A3E635"  # current window highlight in RR panel
COLOR_RR_LINE = "#E63946"  # RR timeline line
COLOR_RR_ABN = "#E63946"  # abnormal RR point
COLOR_RR_NORMAL = "#495057"  # normal RR text annotation
COLOR_BORDER = "#343A40"  # box / spine edge
COLOR_TEXT_MUTE = "#495057"
COLOR_GRID = "#E9ECEF"

# (S_start, S_end) -> (clip background color, name)
CLIP_SPEC = [
    (("S1", "S2"), "#FFE3E3", "Clip 1"),
    (("S3", "S4"), "#D6E9FF", "Clip 2"),
    (("S5", "S6"), "#FFF5D1", "Clip 3"),
]


def bandpass(
    sig: np.ndarray, fs: float, lo: float = 0.5, hi: float = 40.0, order: int = 4
) -> np.ndarray:
    nyq = fs / 2
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def qrs_emphasis(ecg_upward: np.ndarray, fs: int) -> np.ndarray:
    """5-15 Hz band-pass + derivative + square + 150 ms integration.

    Output is a non-negative envelope that bumps where QRS is and stays
    near zero on T waves, P waves and slow baseline drift. Used both by
    the auto-detector and by the manual 'a' (add) action so the two
    snapping behaviours are consistent.
    """
    fs = int(fs)
    nyq = fs / 2
    b, a = butter(4, [5.0 / nyq, 15.0 / nyq], btype="band")
    filtered = filtfilt(b, a, ecg_upward)
    deriv = np.gradient(filtered)
    sq = deriv * deriv
    win = max(3, int(0.15 * fs))
    return np.convolve(sq, np.ones(win) / win, mode="same")


def detect_rpeaks(
    ecg_upward: np.ndarray,
    fs: int,
    method: str = "pan-tompkins",
    snap_signal: np.ndarray | None = None,
) -> np.ndarray:
    """
    Pan-Tompkins-style R-peak detector (pure numpy + scipy.signal).

    Pipeline:
      1-4) QRS emphasis envelope (see qrs_emphasis)
      5) point-wise adaptive threshold (rolling-median baseline)
      6) QRS segments = contiguous regions where emphasis > threshold
      7) within each segment, pick the most prominent local maximum
         in `snap_signal` (T waves can't enter a QRS segment because
         their |gradient| is too low to clear the threshold)
      8) min-spacing 200 ms cleanup

    `snap_signal` defaults to `ecg_upward`. Pass the bandpassed display
    signal to make the dot land exactly on the visible peak instead of
    the raw-signal peak (which can differ by 1-3 samples).
    """
    fs = int(fs)
    integ = qrs_emphasis(ecg_upward, fs)
    snap = snap_signal if snap_signal is not None else ecg_upward
    # 5) point-wise adaptive threshold:
    #    rolling MEDIAN of the integrator (robust to QRS energy in the window
    #    so it tracks the noise floor, not the beat envelope) times a
    #    multiplier, with a global p30 floor so quiet segments don't trip on
    #    noise. Median (not mean) avoids the self-suppression failure mode
    #    where the baseline rises with the QRS bursts it's meant to detect.
    win_b = int(1.5 * fs) | 1  # odd
    baseline = median_filter(integ, size=win_b, mode="nearest")
    floor_val = float(np.percentile(integ, 30))
    thr_arr = np.maximum(floor_val, baseline * 3.0)
    # 6) QRS segments: contiguous samples where the emphasis exceeds
    #    threshold. T waves don't survive the 5-15 Hz band-pass + d^2 path,
    #    so their emphasis stays low and they never form a segment - which
    #    means later steps have no way to mistake a T for an R.
    above = integ > thr_arr
    if not above.any():
        return np.array([], dtype=np.int64)
    transitions = np.diff(above.view(np.int8))
    beg = np.where(transitions == 1)[0] + 1
    end = np.where(transitions == -1)[0] + 1
    if above[0]:
        beg = np.r_[0, beg]
    if above[-1]:
        end = np.r_[end, len(above)]
    n_qrs = min(len(beg), len(end))
    if n_qrs == 0:
        return np.array([], dtype=np.int64)
    beg, end = beg[:n_qrs], end[:n_qrs]
    # drop segments that are too short to be a real QRS (likely noise)
    qrs_lens = end - beg
    min_len = max(int(0.03 * fs), int(0.4 * np.median(qrs_lens)))

    # 7) Within each segment (padded by ±50 ms to catch peaks at edges),
    #    pick the most prominent local maximum in `snap` - that's the R.
    #    Q and S are also inside the segment but they are smaller and rounder,
    #    so prominence picks R reliably without needing magic windows.
    refractory = max(1, int(0.20 * fs))
    pad = int(0.05 * fs)
    n_snap = len(snap)
    r_peaks: list[int] = []
    for b, e in zip(beg, end):
        if e - b < min_len:
            continue
        lo = max(0, b - pad)
        hi = min(n_snap, e + pad)
        seg = snap[lo:hi]
        if len(seg) < 3:
            continue
        locmax, props = find_peaks(seg, prominence=0)
        if len(locmax) > 0:
            loc = lo + int(locmax[int(np.argmax(props["prominences"]))])
        else:
            loc = lo + int(np.argmax(seg))
        if r_peaks and loc - r_peaks[-1] < refractory:
            if snap[loc] > snap[r_peaks[-1]]:
                r_peaks[-1] = loc
            continue
        r_peaks.append(loc)
    return np.array(r_peaks, dtype=np.int64)


def resolve_input(arg: str, raw_dir: Path) -> Path:
    p = Path(arg)
    if p.suffix == ".npz" and p.exists():
        return p
    candidate = raw_dir / f"{arg}_movie_ecg.npz"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Cannot find raw ECG for {arg!r}. Tried {p} and {candidate}.")


def extract_clips(event_desc: list[str], event_time: np.ndarray):
    """Return list of (t_start, t_end, color, name) for the 3 movie clips."""
    desc = list(event_desc)
    t = list(event_time)
    clips = []
    for (a, b), color, name in CLIP_SPEC:
        if a in desc and b in desc:
            ta, tb = t[desc.index(a)], t[desc.index(b)]
            if ta < tb:
                clips.append((ta, tb, color, name))
    return clips


def marker_role(desc: str) -> str:
    """S1 -> 'Clip 1 start', S2 -> 'Clip 1 end', etc."""
    mapping = {
        "S1": "Clip 1 start",
        "S2": "Clip 1 end",
        "S3": "Clip 2 start",
        "S4": "Clip 2 end",
        "S5": "Clip 3 start",
        "S6": "Clip 3 end",
    }
    return mapping.get(desc, desc)


class RpeakEditor:
    def __init__(
        self,
        npz_path: Path,
        out_dir: Path,
        method: str = "pan-tompkins",
        window_sec: int = 30,
        invert: bool = True,
        ask_resume=None,
    ):
        """
        ask_resume : callable(info_dict) -> bool, or None.
            When an existing corrected file is found, called to ask
            whether to load previous edits. If None, falls back to
            terminal input(). For GUI launchers, pass a tkinter-based
            yes/no dialog callable.
        """
        try:
            data = np.load(npz_path, allow_pickle=False)
            _ = data["ecg"]  # force-read to surface any corruption early
        except Exception as e:
            raise RuntimeError(
                f"Cannot load {npz_path.name}: {e}. "
                f"The file may be truncated (incomplete download). "
                f"Re-run rsync to restore it."
            ) from e
        self.npz_path = npz_path
        self.out_dir = Path(out_dir)
        self.ecg_raw = data["ecg"].astype(np.float64)
        self.fs = int(data["fs"])
        self.subject_id = str(data["subject_id"].item())
        self.event_desc = data["event_desc"].tolist()
        self.event_time = data["event_time"].astype(np.float64)
        self.event_sample = data["event_sample"].astype(np.int64)
        self.clips = extract_clips(self.event_desc, self.event_time)

        self.total_samples = len(self.ecg_raw)
        self.total_sec = self.total_samples / self.fs

        sign = -1.0 if invert else 1.0
        self.ecg = sign * self.ecg_raw
        self.ecg_display = bandpass(self.ecg, self.fs, 0.5, 40.0)
        # QRS-emphasis envelope used by manual `a` to ignore T/P/drift.
        self.qrs_env = qrs_emphasis(self.ecg, self.fs)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_path = self.out_dir / f"{self.subject_id}_movie_rpeaks.npz"
        self.method = method

        load_prev = False
        if self.out_path.is_file() and self.out_path.stat().st_size > 0:
            prev = np.load(self.out_path, allow_pickle=False)
            prev_n = len(prev["r_peaks"])
            prev_date = str(prev["edit_date"].item())
            prev_added = int(prev["n_added"])
            prev_deleted = int(prev["n_deleted"])
            info = {
                "file": self.out_path.name,
                "date": prev_date,
                "n_rpeaks": prev_n,
                "n_added": prev_added,
                "n_deleted": prev_deleted,
            }
            if ask_resume is not None:
                load_prev = bool(ask_resume(info))
            else:
                print("\n" + "!" * 60)
                print(f"  FOUND EXISTING CORRECTED FILE: {self.out_path.name}")
                print(f"    last edit : {prev_date}")
                print(f"    R-peaks   : {prev_n}")
                print(f"    edits     : +{prev_added} / -{prev_deleted}")
                print("!" * 60)
                ans = (
                    input(
                        "  Load previous corrections to continue editing? "
                        "[Y/n, n=detect again]: "
                    )
                    .strip()
                    .lower()
                )
                load_prev = ans in ("", "y", "yes")

        if load_prev:
            prev = np.load(self.out_path, allow_pickle=False)
            self.r_peaks: list[int] = prev["r_peaks"].astype(np.int64).tolist()
            self.n_initial = int(prev["n_initial"])
            self.n_added = int(prev["n_added"])
            self.n_deleted = int(prev["n_deleted"])
            print(f"  -> loaded {len(self.r_peaks)} R-peaks")
        else:
            print(f"[initial detect] method='{method}' ...")
            rp = detect_rpeaks(self.ecg, self.fs, method=method, snap_signal=self.ecg_display)
            self.r_peaks = rp.tolist()
            self.n_initial = len(rp)
            self.n_added = 0
            self.n_deleted = 0
            print(f"  got {self.n_initial} R-peaks")

        self.current_pos = 0
        self.window_sec = window_sec
        self.amp_scale = 1.5  # larger = wider y-range = smaller ECG peaks
        self.history: list[list[int]] = []
        self.mouse_x: float | None = None  # last mouse x (seconds) over ECG axis
        # Snap radius override (samples). None → auto-scale with zoom.
        self.snap_radius_override: int | None = None
        self._bg = None  # cached blit background of ECG axis
        self._snap_patch: Rectangle | None = None

        # ================ design system ================
        # SCALE        units are figure fractions (0..1 vertically)
        #   header     [0.94, 1.00]   suptitle + subtitle
        #   quick-ops  [0.89, 0.92]   single compact line
        #   ECG        [0.19, 0.88]   main panel  (69%)
        #   RR         [0.10, 0.18]   compact panel (8%)
        #   cheat      [0.02, 0.07]   bottom reference
        #
        # TYPOGRAPHY   all sizes in pt (coherent scale 9→12)
        #   suptitle   12  bold
        #   subtitle    9  regular
        #   quick-ops   9  mono
        #   cheat       9  mono
        #   axis-lbl    9
        #   S-label     8  bold
        #   RR-ann      7
        #
        # SPACING      inter-item in cheat lines uses '·' as separator
        self.fig, self.axes = plt.subplots(
            2,
            1,
            figsize=(18, 9),
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.20},
        )
        self.fig.patch.set_facecolor(COLOR_BG)
        for ax in self.axes:
            ax.set_facecolor(COLOR_AX_BG)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_edgecolor(COLOR_BORDER)
            ax.tick_params(colors=COLOR_TEXT_MUTE, width=1.0, labelsize=9)

        self.fig.subplots_adjust(top=0.85, bottom=0.15, left=0.06, right=0.97)
        mgr = self.fig.canvas.manager
        if mgr is not None:
            mgr.set_window_title(f"R-peak Editor - {self.subject_id}")

        # layout plan (figsize 18x9)
        # y=0.965   suptitle        (12 bold)
        # y=0.935   subtitle        ( 9 regular)
        # y=0.890   quick-ops line  ( 9 mono)  — above S-marker labels
        # y=0.85    ECG axes TOP  (top=0.85)
        # y=0.15    RR  axes BOTTOM (bottom=0.15)
        # y=0.050   cheat-sheet     ( 9 mono)  — below RR x-label

        self._suptitle = self.fig.suptitle(
            "",
            fontsize=12,
            fontweight="bold",
            y=0.965,
            color=COLOR_BORDER,
            family=FONT_SANS,
        )
        self._subtitle = self.fig.text(
            0.5,
            0.935,
            "",
            ha="center",
            va="center",
            fontsize=9,
            color=COLOR_TEXT_MUTE,
            family=FONT_SANS,
        )
        self.fig.text(
            0.5,
            0.89,
            "[ a ] / 1-finger = ADD"
            "       |       "
            "[ d ] / 2-finger = DELETE"
            "       |       "
            "[ Arrow keys ] = PREV / NEXT"
            "       |       "
            "[ q ] = SAVE & QUIT",
            ha="center",
            va="center",
            fontsize=9,
            color=COLOR_BORDER,
            family=FONT_MONO,
        )

        self.fig.text(
            0.5,
            0.05,
            "[e] re-detect"
            " · "
            "[Shift+C] clear"
            " · "
            "[u] undo"
            " · "
            "[s] save"
            " · "
            "[+/-] time"
            " · "
            "[w/z] amp"
            " · "
            "[ [ / ] ] snap"
            " · "
            "[0] auto"
            " · "
            "close=auto-save",
            ha="center",
            va="center",
            fontsize=9,
            color=COLOR_TEXT_MUTE,
            family=FONT_MONO,
        )

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self._closed = False

        self._update_plot()
        plt.show()

    # edit ops
    def _push(self) -> None:
        self.history.append(list(self.r_peaks))

    def _snap_window_samples(self) -> tuple[int, int, int]:
        """(snap, refine, max_climb) in samples. Manual override → all three
        equal to override (so the user gets the tight band they see).
        Auto → scaled by zoom, floor 2 ms (≈1 sample at 500 Hz)."""
        if self.snap_radius_override is not None:
            r = self.snap_radius_override
            return r, r, r
        zf = self.window_sec / 30
        floor = max(1, int(0.002 * self.fs))
        snap = int(np.clip(zf * 0.10 * self.fs, floor, int(0.20 * self.fs)))
        refine = int(np.clip(zf * 0.05 * self.fs, floor, int(0.05 * self.fs)))
        max_climb = int(np.clip(zf * 0.05 * self.fs, floor, int(0.05 * self.fs)))
        return snap, refine, max_climb

    def _add_rpeak(self, x_sec: float) -> None:
        self._push()
        cs = int(x_sec * self.fs)
        snap, refine, max_climb = self._snap_window_samples()
        lo = max(0, cs - snap)
        hi = min(self.total_samples, cs + snap)
        if lo >= hi:
            return
        # Step 1: QRS centre via emphasis envelope (T-/P-/baseline-proof).
        qrs_center = lo + int(np.argmax(self.qrs_env[lo:hi]))
        # Step 2: refine to actual R-peak top in raw display signal.
        r_lo = max(0, qrs_center - refine)
        r_hi = min(self.total_samples, qrs_center + refine)
        loc = r_lo + int(np.argmax(self.ecg_display[r_lo:r_hi]))
        # Hill-climb a few samples for sample-perfect placement.
        bound_lo = max(0, r_lo - max_climb)
        bound_hi = min(self.total_samples - 1, r_hi - 1 + max_climb)
        while bound_lo < loc < bound_hi:
            if self.ecg_display[loc - 1] > self.ecg_display[loc]:
                loc -= 1
            elif self.ecg_display[loc + 1] > self.ecg_display[loc]:
                loc += 1
            else:
                break
        if loc not in self.r_peaks:
            self.r_peaks.append(loc)
            self.r_peaks.sort()
            self.n_added += 1

    def _delete_rpeak(self, x_sec: float) -> None:
        if not self.r_peaks:
            return
        cs = int(x_sec * self.fs)
        ws = self.current_pos
        we = self.current_pos + int(self.window_sec * self.fs)
        cands = [(abs(p - cs), i) for i, p in enumerate(self.r_peaks) if ws <= p <= we]
        if not cands:
            return
        self._push()
        cands.sort()
        self.r_peaks.pop(cands[0][1])
        self.n_deleted += 1

    def _redetect_window(self) -> None:
        ws = self.current_pos
        we = min(self.current_pos + int(self.window_sec * self.fs), self.total_samples)
        ctx_lo = max(0, ws - self.fs)
        ctx_hi = min(self.total_samples, we + self.fs)
        seg = self.ecg[ctx_lo:ctx_hi]
        seg_display = self.ecg_display[ctx_lo:ctx_hi]
        self._push()
        self.r_peaks = [p for p in self.r_peaks if not (ws <= p < we)]
        try:
            new_local = detect_rpeaks(seg, self.fs, method=self.method, snap_signal=seg_display)
            new_rp = new_local + ctx_lo
            kept = [int(p) for p in new_rp if ws <= p < we]
            self.r_peaks.extend(kept)
            self.r_peaks.sort()
            print(f"  re-detected {len(kept)} R-peaks in window")
        except Exception as e:
            print(f"  re-detection failed: {e}")

    def _clear_window(self) -> None:
        ws = self.current_pos
        we = min(self.current_pos + int(self.window_sec * self.fs), self.total_samples)
        in_win = sum(1 for p in self.r_peaks if ws <= p < we)
        if in_win == 0:
            print("  no R-peaks in current window")
            return
        self._push()
        self.r_peaks = [p for p in self.r_peaks if not (ws <= p < we)]
        self.n_deleted += in_win
        print(f"  cleared {in_win} R-peaks from current window")

    def _undo(self) -> None:
        if self.history:
            self.r_peaks = self.history.pop()

    def _save(self) -> None:
        r_arr = np.array(self.r_peaks, dtype=np.int64)
        np.savez(
            self.out_path,
            r_peaks=r_arr,
            r_peaks_time=(r_arr / self.fs).astype(np.float64),
            fs=np.int32(self.fs),
            subject_id=np.array(self.subject_id),
            n_initial=np.int32(self.n_initial),
            n_added=np.int32(self.n_added),
            n_deleted=np.int32(self.n_deleted),
            edit_date=np.array(datetime.date.today().isoformat()),
            source_npz=np.array(self.npz_path.name),
            method=np.array(self.method),
            # Event markers are carried through so downstream analyses can
            # segment by condition without re-opening the large raw ECG file.
            event_sample=self.event_sample,
            event_time=self.event_time,
            event_desc=np.array(self.event_desc, dtype="<U4"),
            n_samples=np.int64(self.total_samples),
            duration_sec=np.float64(self.total_sec),
        )
        print(
            f"  saved {len(r_arr)} R-peaks -> {self.out_path.name}  "
            f"(+{self.n_added}/-{self.n_deleted})"
        )

    # plotting
    def _update_plot(self) -> None:
        ax_ecg, ax_rr = self.axes
        win_samp = int(self.window_sec * self.fs)
        ws = self.current_pos
        we = min(ws + win_samp, self.total_samples)
        ts, te = ws / self.fs, we / self.fs

        # ===== top: ECG =====
        ax_ecg.clear()
        # Recreate the snap-radius indicator band (ax.clear() removed it).
        # x in data coords, y in axes fraction so it always spans full height.
        self._snap_patch = Rectangle(
            (0, 0),
            0,
            1,
            transform=ax_ecg.get_xaxis_transform(),
            facecolor="#FB923C",
            edgecolor="none",
            alpha=0.20,
            zorder=4,
            visible=False,
        )
        ax_ecg.add_patch(self._snap_patch)
        ax_ecg.set_facecolor(COLOR_AX_BG)
        for spine in ax_ecg.spines.values():
            spine.set_linewidth(1.4)
            spine.set_edgecolor(COLOR_BORDER)
        ax_ecg.grid(
            True, axis="y", linestyle=":", linewidth=0.6, color=COLOR_GRID, alpha=0.8, zorder=0
        )

        # clip background shading (visible portion only)
        for t0, t1, color, _ in self.clips:
            if t1 < ts or t0 > te:
                continue
            ax_ecg.axvspan(max(t0, ts), min(t1, te), color=color, alpha=0.5, zorder=0)

        # ECG trace
        idx = np.arange(ws, we)
        ax_ecg.plot(
            idx / self.fs,
            self.ecg_display[ws:we],
            color=COLOR_ECG,
            linewidth=1.35,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2,
        )

        # R-peaks (red dots)
        rp_in = [p for p in self.r_peaks if ws <= p < we]
        if rp_in:
            ax_ecg.plot(
                np.array(rp_in) / self.fs,
                self.ecg_display[rp_in],
                "o",
                color=COLOR_R,
                markersize=5.5,
                markeredgecolor=COLOR_R_EDGE,
                markeredgewidth=0.6,
                linestyle="None",
                zorder=5,
                label=f"R-peak ({len(rp_in)})",
            )
            # RR ms annotations above each pair
            for i in range(1, len(rp_in)):
                rr_ms = (rp_in[i] - rp_in[i - 1]) / self.fs * 1000
                mid_t = (rp_in[i] + rp_in[i - 1]) / 2 / self.fs
                mid_v = max(self.ecg_display[rp_in[i]], self.ecg_display[rp_in[i - 1]])
                c = COLOR_R if (rr_ms < 400 or rr_ms > 1500) else COLOR_RR_NORMAL
                ax_ecg.annotate(
                    f"{rr_ms:.0f}",
                    xy=(mid_t, mid_v),
                    fontsize=7.5,
                    color=c,
                    ha="center",
                    xytext=(0, 10),
                    textcoords="offset points",
                    family=FONT_MONO,
                    zorder=6,
                )

        # S marker vertical lines
        for desc_i, t_i in zip(self.event_desc, self.event_time):
            if ts <= t_i <= te:
                ax_ecg.axvline(
                    t_i, color=COLOR_S, linestyle="--", linewidth=1.3, alpha=0.8, zorder=3
                )

        ax_ecg.set_xlim(ts, te)
        seg_data = self.ecg_display[ws:we]
        data_center = float(np.mean(seg_data))
        data_range = float(np.ptp(seg_data)) or 1.0
        half = data_range * self.amp_scale
        ax_ecg.set_ylim(data_center - half, data_center + half)
        ax_ecg.set_ylabel("ECG  (inverted, μV)", fontsize=9.5, color=COLOR_BORDER, family=FONT_SANS)

        # S-marker labels ABOVE the axis (no overlap with plot content)
        for desc_i, t_i in zip(self.event_desc, self.event_time):
            if ts <= t_i <= te:
                label = f"{desc_i}  |  {marker_role(desc_i)}"
                ax_ecg.annotate(
                    label,
                    xy=(t_i, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=COLOR_S,
                    fontweight="bold",
                    family=FONT_SANS,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        edgecolor=COLOR_S,
                        linewidth=1.2,
                    ),
                    zorder=10,
                )

        # R-peak legend (inside axes, top right)
        if rp_in:
            leg = ax_ecg.legend(
                loc="upper right",
                fontsize=8.5,
                framealpha=0.95,
                edgecolor=COLOR_BORDER,
                prop={"family": FONT_SANS},
            )
            leg.get_frame().set_linewidth(1.0)

        # ===== bottom: all-time RR =====
        ax_rr.clear()
        ax_rr.set_facecolor(COLOR_AX_BG)
        for spine in ax_rr.spines.values():
            spine.set_linewidth(1.4)
            spine.set_edgecolor(COLOR_BORDER)
        ax_rr.grid(
            True, axis="y", linestyle=":", linewidth=0.6, color=COLOR_GRID, alpha=0.8, zorder=0
        )

        for t0, t1, color, name in self.clips:
            ax_rr.axvspan(t0, t1, color=color, alpha=0.5, zorder=0)
            ax_rr.annotate(
                name,
                xy=((t0 + t1) / 2, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=COLOR_BORDER,
                family=FONT_SANS,
            )

        if len(self.r_peaks) >= 2:
            rp_arr = np.array(self.r_peaks)
            rr = np.diff(rp_arr) / self.fs * 1000
            rr_t = rp_arr[:-1] / self.fs
            ax_rr.plot(
                rr_t,
                rr,
                "-",
                color=COLOR_RR_LINE,
                linewidth=1.1,
                alpha=0.75,
                solid_capstyle="round",
                zorder=2,
            )
            abn = (rr < 400) | (rr > 1500)
            if abn.any():
                ax_rr.plot(rr_t[abn], rr[abn], "o", color=COLOR_RR_ABN, markersize=4, zorder=3)
            ax_rr.axhline(400, color=COLOR_TEXT_MUTE, linestyle="--", linewidth=0.6, alpha=0.6)
            ax_rr.axhline(1500, color=COLOR_TEXT_MUTE, linestyle="--", linewidth=0.6, alpha=0.6)

        # current window highlight
        ax_rr.axvspan(ts, te, color=COLOR_WIN_HL, alpha=0.35, zorder=1)
        ax_rr.set_xlim(0, self.total_sec)
        ax_rr.set_ylabel("RR (ms)", fontsize=9.5, color=COLOR_BORDER, family=FONT_SANS)
        ax_rr.set_xlabel(
            "Time (s)    click to jump window", fontsize=9, color=COLOR_TEXT_MUTE, family=FONT_SANS
        )

        # ===== update suptitle + subtitle (dynamic text) =====
        mid = (ts + te) / 2
        cur_clip = next((name for t0, t1, _, name in self.clips if t0 <= mid <= t1), None)
        loc_str = f"  |  in {cur_clip}" if cur_clip else "  |  between clips"
        self._suptitle.set_text(
            f"Subject  {self.subject_id}"
            f"       R-peaks: {len(self.r_peaks)}  (window {len(rp_in)})"
            f"       edits  +{self.n_added} / -{self.n_deleted}"
        )
        snap_samp = self._snap_window_samples()[0]
        snap_ms = snap_samp / self.fs * 1000
        snap_mode = "manual" if self.snap_radius_override is not None else "auto"
        self._subtitle.set_text(
            f"Window  {ts:.1f} s  to  {te:.1f} s"
            f"  ({self.window_sec:.0f}s){loc_str}"
            f"   ·   snap ±{snap_ms:.0f} ms ({snap_mode})"
        )

        self.fig.canvas.draw_idle()

    # handlers
    def _on_motion(self, event) -> None:
        if event.inaxes is self.axes[0] and event.xdata is not None:
            self.mouse_x = float(event.xdata)
        else:
            self.mouse_x = None
        self._refresh_snap_band()

    def _on_draw(self, event) -> None:
        # Cache a clean background (without the band) for blitting on motion.
        if self._snap_patch is not None:
            self._snap_patch.set_visible(False)
        self._bg = self.fig.canvas.copy_from_bbox(self.axes[0].bbox)

    def _refresh_snap_band(self) -> None:
        if self._bg is None or self._snap_patch is None:
            return
        self.fig.canvas.restore_region(self._bg)
        if self.mouse_x is not None:
            radius_sec = self._snap_window_samples()[0] / self.fs
            self._snap_patch.set_x(self.mouse_x - radius_sec)
            self._snap_patch.set_width(2 * radius_sec)
            self._snap_patch.set_visible(True)
            self.axes[0].draw_artist(self._snap_patch)
        self.fig.canvas.blit(self.axes[0].bbox)

    def _on_close(self, event) -> None:
        # Auto-save when user closes the window (red X, Cmd+W, etc.)
        if self._closed:
            return
        self._closed = True
        print("  [window closed — auto-saving]")
        self._save()

    def _on_click(self, event) -> None:
        if event.inaxes is self.axes[0]:
            if event.xdata is None:
                return
            if event.button == 1:
                self._add_rpeak(event.xdata)
            elif event.button == 3:
                self._delete_rpeak(event.xdata)
            self._update_plot()
        elif event.inaxes is self.axes[1]:
            if event.xdata is None:
                return
            cs = int(event.xdata * self.fs)
            self.current_pos = max(0, cs - int(self.window_sec * self.fs / 2))
            self._update_plot()

    def _on_key(self, event) -> None:
        win_samp = int(self.window_sec * self.fs)
        if event.key == "right":
            self.current_pos = min(
                self.current_pos + win_samp,
                max(0, self.total_samples - win_samp),
            )
        elif event.key == "left":
            self.current_pos = max(0, self.current_pos - win_samp)
        elif event.key in ("+", "="):
            self.window_sec = max(3, self.window_sec - 2)
        elif event.key == "-":
            self.window_sec = min(120, self.window_sec + 2)
        elif event.key == "w":
            self.amp_scale = max(0.1, self.amp_scale * 0.7)
        elif event.key == "z":
            self.amp_scale = min(5.0, self.amp_scale * 1.4)
        elif event.key == "a":
            if self.mouse_x is not None:
                self._add_rpeak(self.mouse_x)
            else:
                print("  (hover the ECG panel, then press 'a')")
        elif event.key == "d":
            if self.mouse_x is not None:
                self._delete_rpeak(self.mouse_x)
            else:
                print("  (hover the ECG panel, then press 'd')")
        elif event.key == "e":
            self._redetect_window()
        elif event.key == "C":
            self._clear_window()
        elif event.key == "u":
            self._undo()
        elif event.key in ("[", "]"):
            cur = self.snap_radius_override
            if cur is None:
                cur = self._snap_window_samples()[0]
            step = max(1, int(0.005 * self.fs))  # 5 ms per press
            new = cur - step if event.key == "[" else cur + step
            new = int(np.clip(new, max(1, int(0.002 * self.fs)), int(0.20 * self.fs)))
            self.snap_radius_override = new
            self._update_plot()
            return
        elif event.key == "0":
            self.snap_radius_override = None
            self._update_plot()
            return
        elif event.key == "s":
            self._save()
            return
        elif event.key == "q":
            self._closed = True  # prevent double-save from close_event
            self._save()
            plt.close(self.fig)
            return
        self._update_plot()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Interactive R-peak editor for movie ECG npz files.")
    ap.add_argument("target", help="Subject ID (e.g. sub-001) or .npz path")
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Folder of *_movie_ecg.npz " f"(default: {DEFAULT_RAW_DIR})",
    )
    ap.add_argument(
        "--out-dir",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Folder to save *_movie_rpeaks.npz " f"(default: {DEFAULT_OUT_DIR})",
    )
    ap.add_argument(
        "--method", default="pan-tompkins", help="R-peak detection method (default: pan-tompkins)"
    )
    ap.add_argument(
        "--window-sec", type=int, default=30, help="Initial window width in seconds (default: 30)"
    )
    ap.add_argument(
        "--no-invert",
        action="store_true",
        help="Do NOT invert ECG (default: invert, since " "channel '65' has R-peaks pointing down)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        path = resolve_input(args.target, args.raw_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"\n[ECG R-peak Editor]  subject: {path.name}")

    RpeakEditor(
        path,
        out_dir=args.out_dir,
        method=args.method,
        window_sec=args.window_sec,
        invert=not args.no_invert,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
