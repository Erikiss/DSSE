"""Model-free execution checks of notebook setup failure and release cleanup."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("auto_builder", Path(__file__).with_name("build_runde1b_auto_notebook.py"))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


class NotebookTests(unittest.TestCase):
    def test_generated_defaults_syntax_and_single_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            notebook = json.loads(B.build(tmp).read_text())
        codes = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        for code in codes:
            ast.parse(code)
        self.assertIn("STUFE2_AUTOMATISCH = True", codes[0])
        self.assertIn("MAX_GPU_STUNDEN = 20", codes[0])
        self.assertIn('ZoneInfo("Europe/Berlin")', codes[0])
        self.assertIn("keine garantierte Stoppzeit", codes[0])
        self.assertNotIn("r1b_run_slot", codes[1])
        self.assertNotIn("R1B_CAPTURE_HIDDEN=", codes[1])
        self.assertEqual(sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "unassign"
                             for n in ast.walk(ast.parse(codes[1]))), 1)

    def test_mount_failure_releases_empty_gpu_but_preserves_existing_local_work(self):
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                if existing:
                    work = base / "work_htq_v2" / "r1b_K_S2a"
                    work.mkdir(parents=True)
                    (work / "only_local_copy.bin").write_bytes(b"keep")
                torch = ModuleType("torch")
                torch.cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda n: "A100",
                                              get_device_capability=lambda n: (8, 0))
                google = ModuleType("google")
                colab = ModuleType("google.colab")
                colab.drive = SimpleNamespace(mount=Mock(side_effect=OSError("auth failed")), flush_and_unmount=Mock())
                colab.runtime = SimpleNamespace(unassign=Mock())
                google.colab = colab
                scope = dict(BRANCH=B.BRANCH, MAX_GPU_STUNDEN=20, STUFE2_AUTOMATISCH=True,
                             RUNTIME_NACH_STOP_FREIGEBEN=True, T0=0)
                with patch.dict(sys.modules, {"torch": torch, "google": google, "google.colab": colab}), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "AUTO-Fehler"):
                        exec(compile(B.source().replace("/content", str(base)), "AUTO", "exec"), scope)
                self.assertEqual(colab.runtime.unassign.call_count, 0 if existing else 1)
                colab.drive.flush_and_unmount.assert_not_called()
                if existing:
                    self.assertEqual((work / "only_local_copy.bin").read_bytes(), b"keep")

    def test_notebook_stop_waits_for_controller_before_trusting_final_status(self):
        tree = ast.parse(B.source())
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_controller")
        import signal
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status.json"
            path.write_text(json.dumps(dict(run_id="test", safe_to_release=True)))
            process = SimpleNamespace(wait=Mock(side_effect=[KeyboardInterrupt(), 2]), poll=lambda: None,
                                      send_signal=Mock())
            child_api = SimpleNamespace(Popen=Mock(return_value=process), STDOUT=-2)
            scope = dict(subprocess=child_api, env={}, _notebook_log=None, _status_path=path, signal=signal, json=json)
            exec(compile(ast.Module(body=[helper], type_ignores=[]), "helper", "exec"), scope)
            code, status, interruption = scope["_controller"](["controller"])
            self.assertEqual(code, 2)
            self.assertTrue(status["safe_to_release"])
            self.assertIsInstance(interruption, KeyboardInterrupt)
            process.send_signal.assert_called_once_with(signal.SIGTERM)
            self.assertEqual(process.wait.call_count, 2)
            self.assertTrue(child_api.Popen.call_args.kwargs["start_new_session"])


if __name__ == "__main__":
    unittest.main()
