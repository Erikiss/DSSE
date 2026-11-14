"""Standalone G orchestration tests: no GPUs, model weights or Qwen runs."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import r1b_g_run as R
from r1b_g_data import mock_data, normalize


class RuntimeTests(unittest.TestCase):
    def args(self, td, **kw):
        base = dict(model="gemma4-e2b", stage="pilot", mode="mock", root=str(Path(td)/"models"),
                    runs=str(Path(td)/"runs"), experiment="test", inputs=None, freeze=None,
                    guard_seconds=120, rerun=False)
        return argparse.Namespace(**(base | kw))

    def test_isolated_run_resume_and_tamper_detection(self):
        calls = []

        def pilot(ctx):
            calls.append(ctx.model_name)
            ctx.write_json("pilot.json", {"sample": 1, "mock": ctx.mock})
            return {"status": "done"}

        with tempfile.TemporaryDirectory() as td, patch.object(R.importlib, "import_module",
                return_value=types.SimpleNamespace(pilot=pilot)), contextlib.redirect_stdout(io.StringIO()):
            a = self.args(td)
            self.assertEqual(R.run(a), 0)
            self.assertEqual(R.run(a), 0)
            self.assertEqual(calls, ["gemma4-e2b"])
            run = R.model_directory(a.runs, a.experiment, a.model)
            summary = R.read_json(run/"summary.json")
            self.assertFalse(summary["scientific_complete"])
            self.assertEqual(summary["state"], "mock_complete")
            output = run/"stages/pilot/pilot.json"
            d = json.loads(output.read_text()); d["sample"] = 2
            R.write_json(output, d)
            self.assertEqual(R.run(a), 0)
            self.assertEqual(len(calls), 2)
            self.assertEqual(R.read_json(output)["sample"], 1)
            self.assertFalse((Path(a.runs)/"dsse_runs").exists())

    def test_missing_implementation_is_not_success(self):
        with tempfile.TemporaryDirectory() as td, patch.object(R.importlib, "import_module",
                side_effect=ModuleNotFoundError("planted missing G module")), contextlib.redirect_stdout(io.StringIO()):
            a = self.args(td)
            self.assertEqual(R.run(a), 4)
            result = R.read_json(R.model_directory(a.runs,a.experiment,a.model)/"summary.json")
            self.assertEqual(result["stages"]["pilot"]["state"], "blocked")
            self.assertFalse(result["scientific_complete"])

    def test_missing_provenance_or_no_output_is_failure(self):
        with tempfile.TemporaryDirectory() as td, patch.object(R.importlib, "import_module",
                return_value=types.SimpleNamespace(pilot=lambda c: {"status": "done"})), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(R.run(self.args(td)), 1)

    def test_mock_cannot_freeze_real_experiment(self):
        with tempfile.TemporaryDirectory() as td:
            a = self.args(td)
            d = R.model_directory(a.runs,a.experiment,a.model)
            R.write_json(d/"state/pilot.json", {"mock": True, "state": "done"})
            a.output, a.inputs = str(Path(td)/"freeze.json"), str(Path(td)/"inputs.json")
            R.write_json(a.inputs, {"mock": False})
            with self.assertRaisesRegex(ValueError, "successful real G pilot"):
                R.create_freeze(a)

    def test_paths_cannot_cross_model_or_experiment_root(self):
        with self.assertRaises(ValueError):
            R.safe_name("../qwen")
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                R.checked_relative(td, "../different-model")

    def test_quantized_stage_graph_is_independent(self):
        from r1b_sessions_G import STAGES, check, ordered
        self.assertTrue(check())
        self.assertEqual(ordered("pilot"), ["pilot"])
        self.assertTrue(all(not name.startswith("r1b_K") for s in STAGES.values() for name in s["after"]))
        self.assertNotIn("gemma4-31b-qat", STAGES["lens_fit"]["models"])

    def test_input_schema_rejects_qwen_token_arrays(self):
        with self.assertRaisesRegex(ValueError, "token arrays"):
            normalize({"wt": {"refw": [[1,2,3]]}})
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"data.json"
            R.write_json(p, {"mock": True, "pilot_texts": ["x"]})
            with self.assertRaisesRegex(ValueError, "mock input"):
                R.load_data(p, False)

    def test_pilot_signature_survives_freeze_and_scientific_bank_change(self):
        a = mock_data()
        b = {**a, "tqa": [], "lens_train": []}
        with tempfile.TemporaryDirectory() as td:
            f = Path(td)/"f.json"; R.write_json(f, {"example": "freeze"})
            sa = R.stage_signature("pilot", "gemma4-e2b", False, a, "a")
            sb = R.stage_signature("pilot", "gemma4-e2b", False, b, "b", f)
            self.assertEqual(sa, sb)
            self.assertNotEqual(sa, R.stage_signature("pilot", "gemma4-e2b", False,
                                                      {"pilot_texts": ["different"]}, "b", f))

    def test_interrupted_cache_is_bound_before_first_stage_finishes(self):
        def crash(ctx):
            Path(ctx.path("unfinished-cache.bin")).write_bytes(b"partial")
            raise KeyboardInterrupt("simulated process interruption")
        with tempfile.TemporaryDirectory() as td, patch.object(R.importlib, "import_module",
                return_value=types.SimpleNamespace(pilot=crash)), contextlib.redirect_stdout(io.StringIO()):
            a = self.args(td)
            with self.assertRaises(KeyboardInterrupt):
                R.run(a)
            d = R.model_directory(a.runs, a.experiment, a.model)
            self.assertEqual(R.read_json(d/"state/pilot.json")["state"], "running")
            with patch.object(R, "source_hashes", return_value={"changed": "code"}):
                with self.assertRaisesRegex(ValueError, "changed code/plans/mode"):
                    R.run(a)

    def test_soft_observation_does_not_block_another_branch(self):
        def blocked(ctx):
            raise R.core.NotRun("closed data gate")
        def observer(ctx):
            ctx.write_json("observed.json", {"available": False})
            return {"status": "done"}
        from r1b_sessions_G import stage
        graph = {"pilot": stage("test:blocked"), "observer": stage("test:observer", observes=("pilot",))}
        with tempfile.TemporaryDirectory() as td, patch.dict(R.STAGES, graph, clear=True), \
                patch.object(R.importlib, "import_module", return_value=types.SimpleNamespace(blocked=blocked, observer=observer)), \
                contextlib.redirect_stdout(io.StringIO()):
            a = self.args(td, stage="observer")
            self.assertEqual(R.run(a), 4)
            d = R.model_directory(a.runs, a.experiment, a.model)
            self.assertEqual(R.read_json(d/"state/observer.json")["state"], "done")

    def test_cpu_calibration_needs_no_gpu_freeze_or_model(self):
        def calibrate(ctx):
            self.assertEqual(str(ctx.device), "cpu")
            ctx.write_json("l_cal.json", {"example": "known answer"})
            return {"status": "done"}
        with tempfile.TemporaryDirectory() as td, patch.object(R, "hardware_check", side_effect=AssertionError("GPU requested")), \
                patch.object(R, "freeze_check", side_effect=AssertionError("freeze requested")), \
                patch.object(R.importlib, "import_module", return_value=types.SimpleNamespace(calibrate=calibrate)), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(R.run(self.args(td, mode="real", stage="l_calibrate")), 0)

    def test_tensor_sidecar_is_part_of_verified_resume(self):
        calls = []
        def pilot(ctx):
            calls.append(1)
            ctx.save_pt("sample.pt", {"value": 1})
        with tempfile.TemporaryDirectory() as td, patch.object(R.importlib, "import_module",
                return_value=types.SimpleNamespace(pilot=pilot)), contextlib.redirect_stdout(io.StringIO()):
            a = self.args(td)
            self.assertEqual(R.run(a), 0)
            run = R.model_directory(a.runs, a.experiment, a.model)
            rec = R.read_json(run/"state/pilot.json")
            self.assertIn("sample.pt.prov.json", rec["outputs"])
            sidecar = run/"stages/pilot/sample.pt.prov.json"
            d = R.read_json(sidecar); d["provenance"]["g_input_sha256"] = "tampered"
            R.write_json(sidecar, d)
            self.assertFalse(R.resumable(rec, rec["signature"], run/"stages/pilot"))
            self.assertEqual(R.run(a), 0)
            self.assertEqual(len(calls), 2)

    def test_report_keeps_other_stages_after_pilot_rerun(self):
        def pilot(ctx): ctx.write_json("pilot.json", {"ok": True})
        def observer(ctx): ctx.write_json("observed.json", {"ok": True})
        from r1b_sessions_G import stage
        graph = {"pilot": stage("test:pilot"), "observer": stage("test:observer", after=("pilot",))}
        with tempfile.TemporaryDirectory() as td, patch.dict(R.STAGES, graph, clear=True), \
                patch.object(R.importlib, "import_module", return_value=types.SimpleNamespace(pilot=pilot, observer=observer)), \
                contextlib.redirect_stdout(io.StringIO()):
            a = self.args(td, stage="all")
            self.assertEqual(R.run(a), 0)
            a.stage = "pilot"
            self.assertEqual(R.run(a), 0)
            a.command = "report"
            R.report(a)
            report = R.read_json(Path(a.runs)/a.experiment/"g_report.json")
            self.assertEqual(report["models"][a.model]["stages"]["observer"]["state"], "done")

    def test_local_catalogue_content_is_bound_into_stage_identity(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"catalogue.json"
            R.write_json(p, {"mock": True, "rows": []})
            data = {"neuron_catalogue": {"path": str(p), "mock": True}}
            first = R.stage_signature("bridge", "gemma4-12b-qat", True, data, "unchanged-json")
            R.write_json(p, {"mock": True, "rows": [{"changed": True}]})
            self.assertNotEqual(first, R.stage_signature("bridge", "gemma4-12b-qat", True, data, "unchanged-json"))

    def test_upstream_output_changes_invalidate_downstream_signature(self):
        args = ("k_measure", "gemma4-e2b", True, {}, "same-raw-input")
        self.assertNotEqual(R.stage_signature(*args, dependencies={"k_prepare": {"outputs": {"hn.json": "a"}}}),
                            R.stage_signature(*args, dependencies={"k_prepare": {"outputs": {"hn.json": "b"}}}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
