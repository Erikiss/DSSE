"""Tests for standalone G notebooks, including execution of their real mock CLI.

The integration test snapshots the current G source into a temporary local git
repository. Only the GitHub transport is redirected to that local snapshot;
generated Python cells and the G runtime itself execute unchanged. No Colab,
model downloads, GPU, or existing Qwen run directory is required.

Fast structural checks run by default. Set RUN_G_NOTEBOOK_EXEC=1 to execute all
six notebooks, their actual native mock CLI and resume checks (several minutes).
"""
from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import nbformat


REPO = Path(__file__).resolve().parents[3]
BUILDER = REPO / "notebooks" / "build_runde1b_gemma_notebooks.py"
_spec = importlib.util.spec_from_file_location("g_notebook_builder", BUILDER)
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


def execute_cell(nb, section, ns):
    cell = next(c for c in nb.cells if c.metadata.get("g_section") == section)
    exec(compile(cell.source, f"notebook:{section}", "exec"), ns)


class NotebookStructureTests(unittest.TestCase):
    def test_six_complete_valid_notebooks_and_python_cells(self):
        self.assertEqual(set(B.NOTEBOOKS), {"PILOT", "E2B", "E4B", "31B_QAT", "12B_QAT", "ANA"})
        for slot in B.NOTEBOOKS:
            with self.subTest(slot=slot):
                nb = B.notebook(slot)
                nbformat.validate(nb)
                self.assertEqual(nb.metadata.dsse_g.default_mode, "mock")
                self.assertTrue(nb.metadata.dsse_g.independent)
                for cell in nb.cells:
                    if cell.cell_type == "code":
                        compile(cell.source, slot + ":" + cell.id, "exec")
                        self.assertIsNone(cell.execution_count)
                        self.assertEqual(cell.outputs, [])

    def test_repeated_builds_are_byte_identical_and_match_deliverables(self):
        with tempfile.TemporaryDirectory() as td:
            for slot in B.NOTEBOOKS:
                a = B.build(slot, Path(td) / "first")
                b = B.build(slot, Path(td) / "second")
                self.assertEqual(a.read_bytes(), b.read_bytes())
                self.assertEqual(a.read_bytes(), (REPO / "notebooks" / a.name).read_bytes())

    def test_no_qwen_runtime_or_upstream_integration_dependency(self):
        for slot in B.NOTEBOOKS:
            code = "\n".join(c.source for c in B.notebook(slot).cells if c.cell_type == "code")
            for forbidden in ["r1b_run_slot.py", "r1b_run_session.py", "runde1b_freeze.json", "dsse_runs_B",
                              "r1b_xfer_A", "Integrationstest bestanden", "Qwen/Qwen", "shutil.rmtree"]:
                self.assertNotIn(forbidden, code)
            self.assertIn("r1b_g_run.py", code)
            self.assertIn("venv.EnvBuilder", code)

    def test_mock_and_real_run_storage_are_distinct(self):
        with tempfile.TemporaryDirectory() as td:
            roots = []
            for mode in ["mock", "pilot"]:
                ns = {}
                nb = B.notebook("PILOT")
                execute_cell(nb, "settings", ns)
                ns.update(MODE=mode, USE_DRIVE=False, LOCAL_BASE=td)
                execute_cell(nb, "paths", ns)
                roots.append(ns["RUNS"])
            self.assertNotEqual(*roots)

    def test_closed_or_failed_run_cannot_print_completion(self):
        ns = dict(subprocess=subprocess, g_command=lambda *args: ["unused"], MODEL="gemma4-e2b",
                  REPO=REPO, ENV={}, USE_DRIVE=False, EXPERIMENT_DIR="test", FINAL_RC=4, MODE="run")
        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            with self.assertRaisesRegex(RuntimeError, "Rückgabewert 4"):
                exec(B.SUMMARY, ns)

    def test_comparison_shows_decisions_without_expanding_atlas_arrays(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            report = {"scientific_complete": False, "measurements": {"gemma4-e2b": {
                "K atlas": {"state": "done", "mock": True, "artifact": "k_extra/extra_analysis.json",
                            "branches": {"K7a": {"status": "measured", "channels": {"payload": "large matrix omitted"}}}},
                "L": {"state": "partial", "mock": True, "artifact": "l_analyze/l_analysis.json",
                      "verdicts": {"L4": {"pair": "undecided", "entity": "mixed"}}},
                "G families": {"state": "partial", "mock": True, "artifact": "g_analyze/g_analysis.json",
                               "families": {"CF-HERR": {"method": "Holm", "members": {"K8c": {"label": "exploratory"}}}}}}}}
            (root / "g_report.json").write_text(json.dumps(report))
            ns = {"EXPERIMENT_DIR": root, "json": json}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(B.COMPARISON, ns)
            self.assertIn("measured", ns["comparison"])
            self.assertIn("pair: undecided", ns["comparison"])
            self.assertIn("k_extra/extra_analysis.json", ns["comparison"])
            self.assertIn("CF-HERR", ns["comparison"])
            self.assertIn("exploratory", ns["comparison"])
            self.assertNotIn("large matrix omitted", ns["comparison"])


@unittest.skipUnless(os.environ.get("RUN_G_NOTEBOOK_EXEC") == "1", "set RUN_G_NOTEBOOK_EXEC=1 for full native notebook acceptance")
class NotebookExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (REPO / "analysis/ju_space/round1b/r1b_g_run.py").is_file():
            raise RuntimeError("Standalone G runtime must exist before notebook acceptance.")
        cls.temp = tempfile.TemporaryDirectory(prefix="g-notebook-acceptance-")
        cls.base = Path(cls.temp.name)
        cls.snapshot = cls.base / "snapshot"
        cls.snapshot.mkdir()
        for rel in ["analysis/ju_space/round1b", "analysis/ju_space/round2", "results/ju_space/plans/runde1b"]:
            shutil.copytree(REPO / rel, cls.snapshot / rel,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        cls.git = staticmethod(subprocess.run)
        cls.git(["git", "init", "-q", "-b", B.BRANCH, str(cls.snapshot)], check=True)
        cls.git(["git", "-C", str(cls.snapshot), "add", "."], check=True)
        cls.git(["git", "-C", str(cls.snapshot), "-c", "user.name=G notebook test", "-c", "user.email=test@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-q", "-m", "Temporary notebook acceptance snapshot"], check=True)
        cls.commit = cls.git(["git", "-C", str(cls.snapshot), "rev-parse", "HEAD"], check=True,
                             capture_output=True, text=True).stdout.strip()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def local_transport(self, args, **kwargs):
        """Redirect only git transport; all G Python subprocesses are real."""
        args = list(args)
        if args and args[0] == "git":
            args = [str(self.snapshot) if x == "https://github.com/Erikiss/DSSE.git" else x for x in args]
        return self.git(args, **kwargs)

    def execute_notebook(self, slot, *, run_selftests=False):
        nb = B.notebook(slot)
        ns = {}
        execute_cell(nb, "settings", ns)
        ns.update(USE_DRIVE=False, LOCAL_BASE=str(self.base / "workspace"),
                  INSTALL_DEPENDENCIES=False, RUN_SELFTESTS=run_selftests)
        with patch.object(subprocess, "run", side_effect=self.local_transport):
            for cell in nb.cells:
                if cell.cell_type == "code" and cell.metadata.g_section != "settings":
                    exec(compile(cell.source, slot + ":" + cell.metadata.g_section, "exec"), ns)
        self.assertEqual(ns["COMMIT"], self.commit)
        self.assertIn(ns["FINAL_RC"], (0, 4))
        self.assertEqual(ns["ENV"]["MOCK"], "1")
        self.assertEqual(ns["ENV"]["HF_HUB_OFFLINE"], "1")
        if slot != "ANA":
            summary = json.loads((ns["EXPERIMENT_DIR"] / ns["MODEL"] / "summary.json").read_text())
            self.assertFalse(summary["scientific_complete"])
            for stage, record in summary["stages"].items():
                with self.subTest(stage=stage):
                    self.assertNotEqual(record["state"], "failed")
                    reason = str(record.get("reason") or "").lower()
                    self.assertNotIn("implementation unavailable", reason)
                    self.assertNotIn("no module named", reason)
        else:
            self.assertIn("| Modell | Messung |", ns["comparison"])
            self.assertEqual(set(ns["report"]["measurements"]), set(B.TARGETS))
            self.assertFalse(ns["report"]["scientific_complete"])
        return ns

    def test_all_six_default_notebooks_execute_real_mock_entrypoint(self):
        # PILOT and the same model notebook share one experiment deliberately:
        # this also checks that pilot outputs can be resumed by the full run.
        for slot in B.NOTEBOOKS:
            with self.subTest(slot=slot):
                # Same self-test code in every model notebook; exercise it once
                # as part of the pilot before the six real mock CLI invocations.
                self.execute_notebook(slot, run_selftests=slot == "PILOT")

    def test_rerun_keeps_the_original_code_commit(self):
        first = self.execute_notebook("PILOT")
        source = json.loads(first["SOURCE"].read_text())
        self.assertEqual(source["commit"], self.commit)
        second = self.execute_notebook("PILOT")
        self.assertEqual(first["REPO"], second["REPO"])
        self.assertEqual(source, json.loads(second["SOURCE"].read_text()))

    def test_interrupted_checkout_resumes_and_preserves_local_edits(self):
        nb, ns = B.notebook("PILOT"), {}
        execute_cell(nb, "settings", ns)
        ns.update(USE_DRIVE=False, LOCAL_BASE=str(self.base / "interrupted_workspace"))
        execute_cell(nb, "paths", ns)
        checkout = ns["BASE"] / "code" / self.commit
        checkout.mkdir(parents=True)
        self.git(["git", "init", "-q", str(checkout)], check=True)
        with patch.object(subprocess, "run", side_effect=self.local_transport):
            execute_cell(nb, "checkout", ns)
            self.assertEqual(ns["COMMIT"], self.commit)
            self.assertTrue(ns["RUNNER"].is_file())
            edited = ns["RUNNER"].read_text() + "\n# preserved operator edit\n"
            ns["RUNNER"].write_text(edited)
            with self.assertRaisesRegex(RuntimeError, "Lokale Änderungen"):
                execute_cell(nb, "checkout", ns)
            self.assertEqual(ns["RUNNER"].read_text(), edited)


if __name__ == "__main__":
    unittest.main(verbosity=2)
