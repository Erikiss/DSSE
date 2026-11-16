"""Exercise selection, provenance and rejection of mixed H+ checkpoints without loading a model."""
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import r1b_core as K
import r1b_htq_variant as V
import r1b_k_s1 as S
import r1b_run_session as R
import r1b_sessions_K as SK


class PolicyTests(unittest.TestCase):
    def test_only_instability_fallback_changes(self):
        tq, ar = {"H": [[0, 1]], "Hplus": [[0, 1], [0, 2]]}, {"H": [[1, 3]]}
        for small in (False, True):
            self.assertEqual(S.h_choice("ok", True, tq, ar, small), ("H_tq", tq["H"]))
            for stable in (False, True):
                self.assertEqual(S.h_choice("H_ar primary (TriviaQA yield)", stable, tq, ar, small), ("H_ar", ar["H"]))
        self.assertEqual(S.h_choice("ok", False, tq, ar, False), ("H+", tq["Hplus"]))
        self.assertEqual(S.h_choice("ok", False, tq, ar, True), ("H_tq", tq["H"]))

    def test_variant_provenance_labels_k_l_only(self):
        with tempfile.TemporaryDirectory() as d:
            for session in ("r1b_K_S1", "r1b_L_A", "r1b_D_C", "r1b_R_S7"):
                ctx = K.Ctx(session, d, device="cpu", mock=False, manifest={"h_policy": V.VARIANT})
                p = ctx.provenance(labels=["existing label"])
                self.assertEqual(p["h_policy"], V.VARIANT)
                self.assertIn("existing label", p["labels"])
                self.assertEqual(V.LABEL in p["labels"], ctx.part in ("K", "L"))

    def test_original_provenance_unchanged_by_default(self):
        with patch.dict(os.environ, {}, clear=True), tempfile.TemporaryDirectory() as d:
            p = K.Ctx("r1b_K_S1", d, device="cpu", mock=False).provenance(labels=["original"])
            self.assertNotIn("h_policy", p)
            self.assertNotIn("post_hoc", p)
            self.assertEqual(p["labels"], ["original"])

    def test_foreign_hn_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ctx = K.Ctx("r1b_L_A", d, device="cpu", mock=False, manifest={"h_policy": V.VARIANT})
            p = Path(ctx.inp) / "r1b_K_S1/hn/hn_set.json"
            p.parent.mkdir(parents=True)
            for value in ({"H_op_name": "H+"}, {"h_policy": V.VARIANT, "H_op_name": "H+"}):
                p.write_text(json.dumps(value))
                with self.assertRaises(RuntimeError):
                    ctx.input("r1b_K_S1", "hn/hn_set.json")
            for name in ("H_tq", "H_ar"):
                p.write_text(json.dumps({"h_policy": V.VARIANT, "H_op_name": name}))
                self.assertEqual(ctx.input("r1b_K_S1", "hn/hn_set.json"), str(p))

    def test_old_completed_derived_sessions_are_rejected(self):
        with patch.dict(os.environ, {"R1B_H_POLICY": V.VARIANT}), patch.object(R, "MOCK", False), tempfile.TemporaryDirectory() as d:
            p = Path(d) / "session_done.json"
            p.write_text("{}")
            for session in ("r1b_K_S1", "r1b_K_S4", "r1b_L_A", "r1b_ana"):
                with self.assertRaises(ValueError):
                    R.check_session_policy(d, session)
            for session in ("r1b_K_S3", "r1b_R_S7", "r1b_D_C"):
                R.check_session_policy(d, session)
            p.write_text(json.dumps({"h_policy": V.VARIANT}))
            R.check_session_policy(d, "r1b_K_S1")

    def test_new_runtime_rebuilds_har_cache_without_repeating_screening(self):
        try:
            with patch.dict(os.environ, {"R1B_H_POLICY": V.VARIANT}):
                importlib.reload(SK)
                steps = {s.name: s for s in SK.STEPS["r1b_K_S1"]}
                self.assertIn("hn/htq_har_ready.json", steps["k2_har"].out)
                self.assertIn("hn/htq_policy.json", steps["k2_sets"].out)
                self.assertEqual(list(steps["k2_screen_00"].out), ["k/k2/screen0.pt"])
                with tempfile.TemporaryDirectory() as d:
                    ctx = K.Ctx("r1b_K_S1", d, root=str(Path(d) / "local-root"), device="cpu", mock=False,
                                manifest={"h_policy": V.VARIANT})
                    marker = Path(d) / "steps/k2_har.json"
                    marker.parent.mkdir(parents=True)
                    marker.write_text(json.dumps({"status": "ok"}))
                    out = Path(d) / "k/k2/h_ar.pt"
                    out.parent.mkdir(parents=True)
                    out.write_bytes(b"original output")
                    self.assertFalse(R.step_done(ctx, steps["k2_har"]))
                    with patch.object(S, "k2_har") as rebuild:
                        S.k2_har_htq(ctx)
                    rebuild.assert_called_once_with(ctx)
                    self.assertTrue(R.step_done(ctx, steps["k2_har"]))
                    ready = Path(d) / "hn/htq_har_ready.json"
                    ready.write_text(json.dumps({"h_policy": V.VARIANT, "fit_rows_required": True}))
                    self.assertFalse(R.step_done(ctx, steps["k2_har"]), "fresh ROOT must rebuild local fit rows")
                    cache = Path(ctx.root) / "k_local" / ctx.session / "k2_har_rows.pt"
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_bytes(b"local feature fixture")
                    try:
                        self.assertTrue(R.step_done(ctx, steps["k2_har"]))
                    finally:
                        cache.unlink()
        finally:
            importlib.reload(SK)


if __name__ == "__main__":
    unittest.main()
