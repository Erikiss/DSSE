"""Model-free checks of calibration gating and the non-polling CPU finaliser."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "analysis/ju_space/round1b"))
spec = importlib.util.spec_from_file_location("htq_auto_cpu_builder", REPO / "notebooks/build_runde1b_htq_auto_cpu.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
import r1b_core as K
import r1b_l_ana as LA
import r1b_run_session as R
import r1b_htq_variant as V
import r1b_auto_colab as AC


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        self.module_dir = self.repo / "analysis/ju_space/round1b"
        self.module_dir.mkdir(parents=True)
        for key, rel in B.CAL_FILES.items():
            (self.repo / rel).write_text("# frozen test file: " + key + "\n")
        self.git("add", ".")
        self.git("-c", "commit.gpgsign=false", "commit", "-qm", "freeze")
        self.freeze = self.git("rev-parse", "HEAD").strip()
        self.path = self.repo / B.CAL_REL
        self.path.parent.mkdir(parents=True)
        self.path.with_name("runde1b_freeze.json").write_text(json.dumps({"freeze_commit": self.freeze}))
        self.record = dict(quick=False, items={k: {"pass": False, "measurement": {"n": 5}} for k in B.CAL_ITEMS},
                           gates={"G_cal_L" + str(i): False for i in range(1, 6)},
                           provenance={key: hashlib.sha256((self.repo / rel).read_bytes()).hexdigest() for key, rel in B.CAL_FILES.items()})
        self.record["provenance"]["written"] = "2026-11-16T00:00:00Z"
        self.commit_record()

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True).strip()

    def commit_record(self):
        self.path.write_text(json.dumps(self.record))
        self.git("add", ".")
        self.git("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "calibration")

    def validate(self):
        with patch.object(K, "REPO", str(self.repo)), patch.object(LA, "HERE", str(self.module_dir)):
            return B.validate_committed_lcal(self.repo)

    def test_full_committed_failed_gates_are_valid_scientific_results(self):
        result = self.validate()
        self.assertEqual(set(result["item_pass"]), set(B.CAL_ITEMS))
        self.assertFalse(any(result["gates"].values()))
        self.assertEqual(result["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_missing_uncommitted_quick_partial_and_wrong_provenance_stop(self):
        good = json.loads(json.dumps(self.record))
        for kind in ("quick", "partial", "provenance", "gates", "stub"):
            with self.subTest(kind=kind):
                self.record = json.loads(json.dumps(good))
                if kind == "quick":
                    self.record["quick"] = True
                elif kind == "partial":
                    del self.record["items"]["n"]
                elif kind == "provenance":
                    self.record["provenance"]["rules_sha256"] = "0" * 64
                elif kind == "gates":
                    self.record["gates"]["G_cal_L1"] = True
                else:
                    self.record["items"]["a"] = {"pass": False}
                self.commit_record()
                with self.assertRaises(RuntimeError):
                    self.validate()
        self.record = good
        self.commit_record()
        self.path.write_text(self.path.read_text() + "\n")
        with self.assertRaisesRegex(RuntimeError, "HEAD"):
            self.validate()
        self.path.unlink()
        with self.assertRaisesRegex(RuntimeError, "fehlt"):
            self.validate()

    def test_changed_calibration_code_cannot_claim_frozen_provenance(self):
        changed = self.repo / B.CAL_FILES["code_sha256"]
        changed.write_text("# changed code\n")
        self.record["provenance"]["code_sha256"] = hashlib.sha256(changed.read_bytes()).hexdigest()
        self.commit_record()
        with self.assertRaisesRegex(RuntimeError, "eingefrorenen Code"):
            self.validate()

    def test_existing_l_ana_provenance_rejection_is_obeyed(self):
        with patch.object(LA, "l_cal_record", return_value=({}, ["invalid source"])):
            with self.assertRaisesRegex(RuntimeError, "L-ANA"):
                self.validate()

    def test_cached_report_must_use_same_committed_calibration(self):
        folder = self.repo / "report"
        folder.mkdir()
        producer = self.git("rev-parse", "HEAD")
        (folder / "session_done.json").write_text(json.dumps({"code_commit": producer}))
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(B.verify_calibration_lineage(self.repo, folder, digest)["producer_commit"], producer)
        with self.assertRaisesRegex(RuntimeError, "anderen l_cal"):
            B.verify_calibration_lineage(self.repo, folder, "0" * 64)


class CpuQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.runs, self.runs_b, self.work = (self.base / n for n in ("A", "B", "work"))
        for path in (self.runs, self.runs_b):
            path.mkdir()
            (path / "_htq_pair_ready.json").write_text(json.dumps({"variant": V.VARIANT, "accounts": ["A", "B"]}))
        self.states = {}
        self.calls, self.syncs = [], []
        self.pending_gpu = False
        self.guard_once = False
        self.failed_exit = None
        def resolve(paths, session):
            if session.startswith("gpu_"):
                return ("waiting", "still running") if self.pending_gpu else ("done", "producer")
            return self.states.get(session, ("waiting", "not started"))
        def run(command, env=None, log=None, **kwargs):
            session = command[3]
            self.calls.append(session)
            path = self.work / session
            path.mkdir(parents=True, exist_ok=True)
            if self.failed_exit is not None:
                return SimpleNamespace(returncode=self.failed_exit)
            if self.guard_once and self.calls.count(session) == 1:
                return SimpleNamespace(returncode=R.EXIT_GUARD)
            self.states[session] = ("done", str(self.runs / session))
            return SimpleNamespace(returncode=R.EXIT_DONE)
        def write_verified(path, obj):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(obj))
        def durable_sync(work, drive, session=None):
            self.syncs.append(session)
            return {"verified": True}
        helpers = SimpleNamespace(write_verified=write_verified, durable_sync=durable_sync, files=AC.files, digest=AC.digest)
        patches = [patch.dict(sys.modules, {"r1b_auto_colab": helpers}), patch.object(V, "check", return_value={}),
                   patch.object(R, "Paths", return_value=object()), patch.object(R, "resolve", side_effect=resolve),
                   patch.object(R, "check_start", return_value={"ok": True, "not_run": None, "waiting": []}),
                   patch.object(K, "freeze_record", return_value=({}, None)),
                   patch.object(B, "_gpu_ancestors", return_value=["gpu_prerequisite"]),
                   patch.object(B, "validate_committed_lcal", return_value={"sha256": "calibration"}),
                   patch.object(B, "verify_calibration_lineage", return_value={"verified": True}),
                   patch.object(B, "_cpu_process", side_effect=run),
                   patch.object(B.time, "sleep", side_effect=AssertionError("CPU finaliser must not poll"))]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    def run_final(self, **kwargs):
        return B.run_cpu_final(REPO, self.base / "root", self.work, self.runs, self.runs_b, {}, **kwargs)

    def test_runs_two_cpu_sessions_in_order_without_polling(self):
        result = self.run_final()
        self.assertEqual(self.calls, ["r1b_ana", "r1b_ana2"])
        self.assertEqual(self.syncs, self.calls)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(json.loads((self.runs / "_auto_cpu_status.json").read_text())["status"], "complete")

    def test_missing_calibration_or_gpu_prerequisite_does_not_start_formal_analysis(self):
        with patch.object(B, "validate_committed_lcal", side_effect=RuntimeError("L-CAL fehlt")):
            with self.assertRaisesRegex(RuntimeError, "L-CAL"):
                self.run_final()
        self.assertEqual(self.calls, [])
        self.pending_gpu = True
        with self.assertRaisesRegex(RuntimeError, "GPU-Voraussetzung"):
            self.run_final()
        self.assertEqual(self.calls, [])
        self.assertFalse((self.runs / "r1b_ana").exists())

    def test_checkpoint_continues_immediately_and_errors_preserve_runtime_files(self):
        self.guard_once = True
        result = self.run_final(max_continuations=2)
        self.assertEqual(self.calls, ["r1b_ana", "r1b_ana", "r1b_ana2", "r1b_ana2"])
        self.assertEqual(result["status"], "complete")
        self.states.clear()
        self.failed_exit = R.EXIT_FAILED
        with self.assertRaisesRegex(RuntimeError, "scheiterte"):
            self.run_final()
        self.assertTrue((self.work / "r1b_ana").is_dir())
        self.assertEqual(json.loads((self.runs / "_auto_cpu_status.json").read_text())["status"], "failed")

    def test_sync_failure_cannot_claim_completion(self):
        import r1b_auto_colab
        original_write = r1b_auto_colab.write_verified
        def observe_write(path, value):
            if Path(path).name == "_auto_cpu_status.json" and value.get("status") == "failed":
                owner = self.runs / ".auto_cpu_writer.lock/owner.json"
                self.assertTrue(owner.is_file(), "failure status must be written before releasing ownership")
                self.assertEqual(json.loads(owner.read_text())["run_id"], value["run_id"])
            original_write(path, value)
        with patch.object(r1b_auto_colab, "durable_sync", side_effect=OSError("Drive unavailable")), \
             patch.object(r1b_auto_colab, "write_verified", side_effect=observe_write):
            with self.assertRaisesRegex(OSError, "Drive unavailable"):
                self.run_final()
        self.assertEqual(self.calls, ["r1b_ana"])
        self.assertEqual(json.loads((self.runs / "_auto_cpu_status.json").read_text())["status"], "failed")

    def test_retry_repairs_trusted_unique_local_output_before_accepting_old_drive_done(self):
        import r1b_auto_colab
        with patch.object(r1b_auto_colab, "durable_sync", side_effect=OSError("Drive unavailable")):
            with self.assertRaises(OSError):
                self.run_final()
        local = self.work / "r1b_ana"
        new = local / "result.json"
        new.write_text("new unique result")
        (local / "session_done.json").write_text(json.dumps({
            "session": "r1b_ana", "mock": False,
            "files": {"result.json": hashlib.sha256(new.read_bytes()).hexdigest()},
        }))
        drive = self.runs / "r1b_ana"
        drive.mkdir()
        (drive / "result.json").write_text("old result")
        (drive / "session_done.json").write_text(json.dumps({"session": "r1b_ana", "mock": False, "files": {}}))
        self.calls.clear()
        with patch.object(r1b_auto_colab, "durable_sync", side_effect=AC.durable_sync):
            self.assertEqual(self.run_final()["status"], "complete")
        self.assertEqual((drive / "result.json").read_bytes(), new.read_bytes())
        self.assertEqual(self.calls, ["r1b_ana2"], "the recovered completed ANA must not be recomputed")

    def test_untrusted_local_work_and_other_writer_are_never_overwritten(self):
        local = self.work / "r1b_ana"
        local.mkdir(parents=True)
        (local / "unique.txt").write_text("preserve me")
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "Ungeprüfte lokale"):
                self.run_final()
        self.assertEqual((local / "unique.txt").read_text(), "preserve me")
        self.assertFalse((self.runs / "r1b_ana").exists())
        status = self.runs / "_auto_cpu_status.json"
        status.write_text(json.dumps({"run_id": "other writer", "owner_nonce": "other"}))
        lock = self.runs / ".auto_cpu_writer.lock"
        lock.mkdir()
        (lock / "owner.json").write_text(json.dumps({"run_id": "other writer"}))
        before = status.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "Schreib|Sperre|andere CPU"):
            self.run_final()
        self.assertEqual(before, status.read_bytes(), "a competing writer must not clobber the active writer's status")
        self.assertEqual(self.calls, [])


class ProcessTests(unittest.TestCase):
    def test_interruption_stops_child_process_group_before_returning(self):
        from unittest.mock import Mock
        child = Mock(pid=12345)
        child.wait.side_effect = [KeyboardInterrupt(), 0]
        with patch.object(B.subprocess, "Popen", return_value=child), patch.object(B.os, "killpg") as kill:
            with self.assertRaises(KeyboardInterrupt):
                B._cpu_process(["python", "analysis"], {}, None)
        kill.assert_called_once_with(12345, B.signal.SIGTERM)
        self.assertEqual(child.wait.call_count, 2)


class NotebookTests(unittest.TestCase):
    def test_generated_cpu_notebook_has_no_waiting_slot_or_unconditional_shutdown(self):
        with tempfile.TemporaryDirectory() as output:
            path = B.build(output)
            doc = json.loads(path.read_text())
        code = ["".join(c["source"]) for c in doc["cells"] if c["cell_type"] == "code"]
        for source in code:
            ast.parse(source)
        joined = "\n".join(code)
        self.assertIn("validate_committed_lcal", joined)
        self.assertIn("run_cpu_final(REPO", joined)
        self.assertNotIn('"--wait-h"', joined)
        self.assertNotIn('"--poll-s"', joined)
        self.assertNotIn("time.sleep(", joined)
        self.assertIn('AUTO_CPU["status"] == "complete"', joined)
        self.assertLess(joined.index("drive.flush_and_unmount()"), joined.index("runtime.unassign()"))
        self.assertNotIn("gpuType", doc["metadata"].get("colab", {}))


if __name__ == "__main__":
    unittest.main()
