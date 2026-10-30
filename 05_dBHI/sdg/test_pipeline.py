"""Regression checks for scientific alignment and cache safety (no cohort writes)."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from . import pipeline as p
from .model import PySDG


class PipelineTests(unittest.TestCase):
    def test_archived_arrays_can_be_read_without_allowing_model_resumption(self):
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(p.DEFAULT_CONFIG)
            manifest = p.make_manifest(config)
            manifest["python"] = "original-computation-environment"
            p.atomic_json(Path(directory) / "_manifest.json", manifest)
            arrays = {
                "cond_order": np.array(p.CONDITIONS), "labels": np.array(["Cz"]),
                "bands": np.array(list(config["flims_eeg"])), "hrv_bands": np.array(["LF", "HF"]),
                "_fingerprint": np.array(p.fingerprint(manifest)), "_qc_json": np.array("{}"),
            }
            for condition in p.CONDITIONS:
                for direction in ("bth", "htb"):
                    arrays[f"{condition}__{direction}"] = np.ones((4, 2, 1, 3))
                    arrays[f"{condition}__t_{direction}"] = np.arange(3) / config["fs_bhi"]
            path = Path(directory) / "sub-001_dbhi.npz"
            np.savez(path, **arrays)
            result = p.verify_all(["sub-001"], directory, config, check_environment=False)
            self.assertEqual(result[0]["status"], "ok")
            with self.assertRaisesRegex(RuntimeError, "Manifest mismatch"):
                p.verify_all(["sub-001"], directory, config)
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                p.prepare_cache(directory, p.make_manifest(config))
            changed = copy.deepcopy(config)
            changed["window"] = 20
            with self.assertRaisesRegex(RuntimeError, "Manifest mismatch"):
                p.verify_all(["sub-001"], directory, changed, check_environment=False)
            arrays["_fingerprint"] = np.array("wrong-generation")
            np.savez(path, **arrays)
            result = p.verify_all(["sub-001"], directory, config, check_environment=False)
            self.assertEqual(result[0]["status"], "failed")
            self.assertIn("fingerprint mismatch", result[0]["detail"])

    def test_stale_manifest_is_rejected_without_relabeling(self):
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(p.DEFAULT_CONFIG)
            manifest = p.make_manifest(config)
            p.prepare_cache(directory, manifest)
            path = Path(directory) / "_manifest.json"
            before = path.read_bytes()
            config["window"] = 20
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                p.prepare_cache(directory, p.make_manifest(config))
            self.assertEqual(path.read_bytes(), before)

    def test_orphan_cache_not_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory) / "test_dbhi.npz", old=np.array(1))
            with self.assertRaisesRegex(RuntimeError, "without a manifest"):
                p.prepare_cache(directory, p.make_manifest(p.DEFAULT_CONFIG))

    def test_band_axis_order_changes_fingerprint(self):
        config = copy.deepcopy(p.DEFAULT_CONFIG)
        original = p.fingerprint(p.make_manifest(config))
        config["flims_eeg"] = dict(reversed(list(config["flims_eeg"].items())))
        self.assertNotEqual(original, p.fingerprint(p.make_manifest(config)))

    def test_bth_uses_centered_cardiac_values(self):
        obj = PySDG(
            r_peaks=np.arange(0, 100, 0.6),
            fs_bhi=4,
            window=15,
            parallelize=False,
            progressbar=False,
        )
        t = np.arange(0, 80, 0.25)
        ones = np.ones_like(t)
        # Disable smoothing only for this diagnostic, so it cannot hide a lag.
        with patch("sdg.model.convolve_same", side_effect=lambda a, b: a):
            output = obj._BHI_InsideModel(ones, ones, t.copy(), t.copy(), t, True)
        np.testing.assert_allclose(output[1], output[-1]["time_BToH"])
        np.testing.assert_allclose(output[2], output[-1]["time_BToH"])

    def test_cardiac_origin_follows_first_beat(self):
        rr = 0.6 + 0.03 * np.sin(np.arange(120))
        for first in (0.0, 0.4):
            peaks = first + np.r_[0, np.cumsum(rr)]
            obj = PySDG(r_peaks=peaks, fs_bhi=4, window=15, progressbar=False)
            obj.tfr["t"] = np.arange(10, 60, 0.25)
            axes = []

            def capture(x, y, xq):
                axes.append(x.copy())
                return np.linspace(0, 1, len(xq))

            obj._compute_cpr_and_csr(15, interpmethod=capture)
            if first == 0:
                baseline = axes[0]
            else:
                np.testing.assert_allclose(axes[0] - baseline, first)

    def test_matlab_boundaries_include_correct_edge_beats(self):
        with tempfile.TemporaryDirectory() as directory:
            mat = Path(directory) / "input.mat"
            rp = Path(directory) / "peaks.npz"
            with h5py.File(mat, "w") as f:
                eeg = f.create_group("EEG")
                eeg.create_dataset("data", shape=(6000, 1), dtype="f4")
                eeg.create_dataset("srate", data=[[100.0]])
                channels = eeg.create_group("chanlocs")

                def refs(parent, key, strings):
                    ds = parent.create_dataset(key, shape=(len(strings), 1), dtype=h5py.ref_dtype)
                    for i, string in enumerate(strings):
                        ref = f.create_dataset(
                            key + str(i), data=np.array([ord(c) for c in string], dtype="u2")
                        )
                        ds[i, 0] = ref.ref

                refs(channels, "labels", ["Cz"])
                seg = f.create_group("seg_info")
                refs(seg, "cond_order", p.CONDITIONS)
                for key, values in {
                    "onsets_orig": [101, 2101, 4101],
                    "offsets_orig": [2100, 4100, 6100],
                    "onsets_new": [1, 2001, 4001],
                    "offsets_new": [2000, 4000, 6000],
                }.items():
                    seg.create_dataset(key, data=values)
            # Lengthen recording to meet the SDG support requirement.
            with h5py.File(mat, "a") as f:
                del f["EEG/data"]
                f["EEG"].create_dataset("data", shape=(12000, 1), dtype="f4")
                for key, values in {
                    "onsets_orig": [101, 4101, 8101],
                    "offsets_orig": [4100, 8100, 12100],
                    "onsets_new": [1, 4001, 8001],
                    "offsets_new": [4000, 8000, 12000],
                }.items():
                    f["seg_info"][key][...] = values
            peaks = np.unique(np.r_[np.arange(100, 12100, 60), 4099, 8099, 12099])
            np.savez(rp, r_peaks=peaks, fs=100)
            _, _, _, clips = p.read_metadata(mat, rp, p.DEFAULT_CONFIG)
            self.assertEqual(clips[0]["r_seconds"][0], 0.0)
            self.assertEqual(clips[0]["r_seconds"][-1], 39.99)

    def test_missing_direction_and_infinity_rejected(self):
        data = {
            "labels": np.array(["Cz"]),
            "cond_order": np.array(p.CONDITIONS),
            "bands": np.array(list(p.DEFAULT_CONFIG["flims_eeg"])),
            "hrv_bands": np.array(["LF", "HF"]),
        }
        for cond in p.CONDITIONS:
            for direction in ("bth", "htb"):
                data[f"{cond}__{direction}"] = np.ones((4, 2, 1, 3))
                data[f"{cond}__t_{direction}"] = np.array([10, 10.25, 10.5])
        p.validate_arrays(data, p.DEFAULT_CONFIG)
        data["fear__htb"][0, 0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            p.validate_arrays(data, p.DEFAULT_CONFIG)
        del data["fear__htb"]
        with self.assertRaises(KeyError):
            p.validate_arrays(data, p.DEFAULT_CONFIG)


if __name__ == "__main__":
    unittest.main()
