"""Convert verified R peaks to IBI and LF/HF time series."""

from pathlib import Path
from tempfile import TemporaryDirectory

import neurokit2 as nk
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RPEAK_DIR = ROOT / "01_preprocessing/outputs/batch/ecg/r_peaks"
CHILD_INFO = ROOT / "00_data/participants.csv"
OUT_NPZ = ROOT / "01_preprocessing/outputs/batch/ecg/hrv/ecg_ts.npz"

FS = 500  # Hz, ECG sampling rate
DROP_SEC = 2  # s dropped after clip onset (onset transient)
ANALYSIS_SEC = 176  # s analysed -> 704 @4 Hz / 176 @1 Hz
RESAMPLE_HZ = 4  # Hz, IBI resampling rate

LF_BAND = (0.04, 0.15)  # Hz
HF_BAND = (0.15, 0.80)  # Hz, upper bound raised from the adult 0.40 for children


CONDITIONS = ["fear", "happy", "neutral"]
ONSET_CODE = {"fear": "S1", "neutral": "S3", "happy": "S5"}  # S2/S4/S6 are offsets

N_4HZ, N_1HZ = ANALYSIS_SEC * RESAMPLE_HZ, ANALYSIS_SEC


def load_rpeaks(sid, rpeak_dir=RPEAK_DIR):
    with np.load(Path(rpeak_dir) / f"{sid}_movie_rpeaks.npz", allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def segment(rp, condition):
    """R peaks in one condition's window, re-zeroed to its start.

    Clip order is counterbalanced, so onsets are found by marker label.
    """
    desc = rp["event_desc"].astype(str)
    hit = rp["event_sample"][desc == ONSET_CODE[condition]]
    if hit.size != 1:
        raise ValueError(f"{condition}: expected 1 onset marker, found {hit.size}")
    start = int(hit[0]) + DROP_SEC * FS
    peaks = rp["r_peaks"].astype(np.int64)
    return peaks[(peaks >= start) & (peaks < start + ANALYSIS_SEC * FS)] - start


def clip_order(rp):
    """Position (1/2/3) of each condition for this subject."""
    desc = rp["event_desc"].astype(str)
    onset = {c: int(rp["event_sample"][desc == ONSET_CODE[c]][0]) for c in CONDITIONS}
    ranked = sorted(onset, key=onset.get)
    return {c: ranked.index(c) + 1 for c in CONDITIONS}, "-".join(ranked)


def fit_length(x, n):
    """Trim or NaN-pad to exactly n samples."""
    x = np.asarray(x, float).ravel()
    if x.size >= n:
        return x[:n]
    out = np.full(n, np.nan)
    out[: x.size] = x
    return out


def condition_series(peaks_rel):
    """IBI @4 Hz and LF / HF power @1 Hz for one condition of one subject."""
    ibi_beat = (
        nk.signal_period(
            peaks=peaks_rel, sampling_rate=FS, interpolation_method="signal_interpolate"
        )
        * 1000.0
    )

    proc = nk.intervals_process(
        ibi_beat, intervals_time=peaks_rel / FS, interpolate=True, interpolation_rate=RESAMPLE_HZ
    )
    # Spans first to last beat, so 701-705 samples rather than exactly 704. The
    # spectral step needs this native series; only its outputs are grid-fitted.
    native = np.asarray(proc[0], float).ravel()[:N_4HZ]

    # min/max_frequency only crop the output, so they are tied to the bands.
    f, _, tf = nk.signal_timefrequency(
        native,
        sampling_rate=RESAMPLE_HZ,
        min_frequency=LF_BAND[0],
        max_frequency=HF_BAND[1],
        method="pwvd",
        show=False,
    )
    # Clamp negative Wigner-Ville cross terms before integrating band power.
    tf = np.where(tf < 0, 0.0, tf)

    band = {}
    for name, (lo, hi) in (("lf", LF_BAND), ("hf", HF_BAND)):
        p4 = tf[(f >= lo) & (f <= hi), :].sum(axis=0)
        p1 = np.array(
            [
                np.trapezoid(p4[i * RESAMPLE_HZ : (i + 1) * RESAMPLE_HZ], dx=1 / RESAMPLE_HZ)
                for i in range(p4.size // RESAMPLE_HZ)
            ]
        )
        # A band left at exactly zero by clamping is unusable, not zero power.
        band[name] = fit_length(np.where(p1 > 0, p1, np.nan), N_1HZ)

    return fit_length(native, N_4HZ), band["lf"], band["hf"], int(ibi_beat.size)


def compute_hrv(ids, rpeak_dir=RPEAK_DIR, out_npz=OUT_NPZ, *, overwrite=False):
    """Save IBI and LF/HF time series for the reviewed subjects."""
    ids = [str(sid) for sid in ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Provide a nonempty list of unique subject IDs.")
    if Path(out_npz).is_file() and Path(out_npz).stat().st_size > 0 and not overwrite:
        with np.load(out_npz, allow_pickle=False) as previous:
            if previous["ids"].astype(str).tolist() != ids:
                raise ValueError(
                    "The existing HRV file contains different subjects. "
                    "Use the original subject list or a different output path."
                )
    n = len(ids)

    ibi = np.full((n, 3, N_4HZ), np.nan)
    lf = np.full((n, 3, N_1HZ), np.nan)
    hf = np.full((n, 3, N_1HZ), np.nan)
    position = np.zeros((n, 3), int)
    n_beats = np.zeros((n, 3), dtype=int)

    for i, sid in enumerate(ids):
        rp = load_rpeaks(sid, rpeak_dir)
        if int(rp["fs"]) != FS:
            raise ValueError(f"{sid}: expected {FS} Hz ECG, got {rp['fs']}")
        pos, _ = clip_order(rp)
        for j, cond in enumerate(CONDITIONS):
            ibi[i, j], lf[i, j], hf[i, j], n_beats[i, j] = condition_series(segment(rp, cond))
            position[i, j] = pos[cond]
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"{i+1:3d}/{n}", flush=True)

    arrays = dict(
        ids=np.array(ids),
        conditions=np.array(CONDITIONS),
        ibi=ibi, lf=lf, hf=hf, position=position,
        time_4hz=np.arange(N_4HZ) / RESAMPLE_HZ,
        time_1hz=np.arange(N_1HZ) * 1.0,
        lf_band=np.array(LF_BAND), hf_band=np.array(HF_BAND),
        n_beats=n_beats,
    )
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".hrv_", dir=out_npz.parent) as temporary:
        staged = Path(temporary) / "hrv.npz"
        np.savez_compressed(staged, **arrays)
        staged.replace(out_npz)
    print(f"Saved {out_npz.name}")
    return arrays


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", help="Reviewed subject IDs; default: participants.csv")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing a different subject list")
    args = parser.parse_args()
    ids = args.ids or pd.read_csv(CHILD_INFO, dtype={"ID": str})["ID"].tolist()
    compute_hrv(ids, overwrite=args.overwrite)
