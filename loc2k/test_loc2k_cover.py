#!/usr/bin/env python3
"""Regression tests for exact disjoint tiling and header checks in loc2k/src/bk16_loc2verify.py.

Usage (from repository root):
    python3 -B loc2k/test_loc2k_cover.py
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_verifier():
    mod_path = REPO_ROOT / "loc2k/src/bk16_loc2verify.py"
    spec = importlib.util.spec_from_file_location("bk16_loc2verify", str(mod_path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestLoc2KCoverVerification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_verifier()
        cls.ra, cls.rl, cls.T = 0.0300001, 0.005, (-0.027, 0.029)
        loc_path = REPO_ROOT / "loc2k/run/loc2_P002/loc2K_boxes.jsonl"
        cls.lines = [json.loads(l) for l in loc_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        cls.origin = cls.lines[0]
        cls.boxes = cls.lines[1:]

    def test_stored_certificate_passes_strict_tiling(self):
        e = self.origin["eps0"]
        psi_boxes = [(r["lo"], r["hi"]) for r in self.boxes if r["kind"] == "psi"] + [
            ([0, 0, -e, -e], [e, min(self.rl, e), e, e])
        ]
        single_boxes = [(r["lo"], r["hi"]) for r in self.boxes if r["kind"] == "single"]

        psi_ok, psi_st = self.mod.check_exact_tiling(
            [0, 0, self.T[0], self.T[0]], [self.ra, self.rl, self.T[1], self.T[1]], psi_boxes
        )
        single_ok, single_st = self.mod.check_exact_tiling(
            [-self.ra, 0, self.T[0]], [self.ra, self.rl, self.T[1]], single_boxes
        )
        self.assertTrue(psi_ok, f"Stored psi tiling failed: {psi_st}")
        self.assertTrue(single_ok, f"Stored single tiling failed: {single_st}")
        self.assertEqual(psi_st["overlaps"], 0)
        self.assertEqual(single_st["overlaps"], 0)

    def test_rejects_duplicate_and_delete_equal_volume_swap(self):
        single_boxes = [(r["lo"], r["hi"]) for r in self.boxes if r["kind"] == "single"]
        tampered = list(single_boxes)
        tampered[1] = tampered[0]
        ok, st = self.mod.check_exact_tiling([-self.ra, 0, self.T[0]], [self.ra, self.rl, self.T[1]], tampered)
        self.assertTrue(st["vol_ok"])
        self.assertGreater(st["overlaps"], 0)
        self.assertFalse(ok)

    def test_rejects_minimal_2box_half_domain_overlap(self):
        half = ([0.0, 0.0, self.T[0]], [self.ra, self.rl, self.T[1]])
        ok, st = self.mod.check_exact_tiling([-self.ra, 0, self.T[0]], [self.ra, self.rl, self.T[1]], [half, half])
        self.assertTrue(st["vol_ok"])
        self.assertEqual(st["overlaps"], 1)
        self.assertFalse(ok)

    def test_rejects_disjoint_equal_volume_partially_outside_domain(self):
        b1 = ([-self.ra, 0.0, self.T[0]], [0.0, self.rl, self.T[1]])
        b2 = ([0.0, 0.0, self.T[0]], [2.0 * self.ra, 0.5 * self.rl, self.T[1]])
        ok, st = self.mod.check_exact_tiling([-self.ra, 0, self.T[0]], [self.ra, self.rl, self.T[1]], [b1, b2])
        self.assertTrue(st["vol_ok"])
        self.assertEqual(st["overlaps"], 0)
        self.assertEqual(st["out_of_bounds"], 1)
        self.assertEqual(st["malformed"], 0)
        self.assertFalse(ok)

    def test_rejects_completely_out_of_bounds_box(self):
        shifted = ([self.ra, 0.0, self.T[0]], [3 * self.ra, self.rl, self.T[1]])
        ok, st = self.mod.check_exact_tiling([-self.ra, 0, self.T[0]], [self.ra, self.rl, self.T[1]], [shifted])
        self.assertEqual(st["out_of_bounds"], 1)
        self.assertFalse(ok)

    def test_rejects_inverted_and_nan_boxes(self):
        inverted = ([self.ra, 0.0, self.T[0]], [-self.ra, self.rl, self.T[1]])
        ok, st = self.mod.check_exact_tiling([-self.ra, 0, self.T[0]], [self.ra, self.rl, self.T[1]], [inverted])
        self.assertEqual(st["malformed"], 1)
        self.assertFalse(ok)

        nan_box = ([float("nan"), 0.0, self.T[0]], [self.ra, self.rl, self.T[1]])
        ok2, st2 = self.mod.check_exact_tiling([-self.ra, 0, self.T[0]], [self.ra, self.rl, self.T[1]], [nan_box])
        self.assertEqual(st2["malformed"], 1)
        self.assertFalse(ok2)

    def test_cli_exits_nonzero_on_overlap_and_invalid_origin_header(self):
        script = REPO_ROOT / "loc2k/src/bk16_loc2verify.py"
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "overlap.jsonl"
            p1.write_text(
                "\n".join(
                    json.dumps(x)
                    for x in [
                        self.origin,
                        {"kind": "single", "lo": [0.0, 0.0, self.T[0]], "hi": [self.ra, self.rl, self.T[1]], "lb": 0.0},
                        {"kind": "single", "lo": [0.0, 0.0, self.T[0]], "hi": [self.ra, self.rl, self.T[1]], "lb": 0.0},
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            pr = subprocess.run([sys.executable, "-B", str(script), str(p1)], capture_output=True, text=True)
            self.assertNotEqual(pr.returncode, 0)
            self.assertIn("single tiling valid: False", pr.stdout)

            bad_origin = dict(self.origin, Smin=2.0)
            p2 = Path(td) / "bad_header.jsonl"
            p2.write_text(json.dumps(bad_origin) + "\n", encoding="utf-8")
            pr2 = subprocess.run([sys.executable, "-B", str(script), str(p2)], capture_output=True, text=True)
            self.assertNotEqual(pr2.returncode, 0)
            self.assertIn("origin False", pr2.stdout)


if __name__ == "__main__":
    unittest.main()
