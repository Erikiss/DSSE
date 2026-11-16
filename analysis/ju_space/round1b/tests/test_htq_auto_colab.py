"""Model-free checks of the one-GPU continuation and durable release gate."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import r1b_auto_colab as A


class FakeEngine:
    def __init__(self, a, b):
        self.roots = {"A": a, "B": b}
        self.xfer = str(a / "r1b_xfer")
        self.waiting = []
        self.bootstrap_error = None
        self.checks = []

    def bootstrap(self):
        if self.bootstrap_error:
            raise ValueError(self.bootstrap_error)

    def state(self, session):
        folder = self.roots[A.C.SESSIONS[session]["account"]] / session
        if (folder / "session_done.json").is_file():
            for rel, expected in A.current_hashes(folder, session).items():
                if A.digest(folder / rel) != expected:
                    raise ValueError("corrupt producer")
            return "done"
        if (folder / "session_not_run.json").is_file():
            return "not_run"
        return "pending"

    def ready(self, session):
        self.checks.append(session)
        return dict(ok=not self.waiting, waiting=self.waiting)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.a, self.b = self.base / "A", self.base / "B"
        self.a.mkdir(); self.b.mkdir()
        self.work = self.base / "work_htq_v2"
        self.engine = FakeEngine(self.a, self.b)
        self.session = "r1b_K_S2a"
        self.calls = []

    def finish(self, session, partial=False):
        p = self.work / session
        p.mkdir(parents=True, exist_ok=True)
        (p / "result.bin").write_bytes(b"frozen scientific result")
        marker = "session_partial.json" if partial else "session_done.json"
        A.write_verified(p / marker, dict(session=session, mock=False,
                         files={"result.bin": A.digest(p / "result.bin")}))
        if not partial:
            (p / "session_partial.json").unlink(missing_ok=True)

    def executor(self, stage, session, env, log):
        self.calls.append((stage, session, env))
        self.finish(session)
        return 0

    def controller(self, **kwargs):
        c = A.Controller(self.a, self.b, self.work, self.base / "root", engine=self.engine,
                         executor=kwargs.pop("executor", self.executor), **kwargs)
        c.queue = [("B2", self.session)]
        return c

    def test_exact_queue_and_no_start_recapture(self):
        queue = A.sequence()
        stages = list(dict.fromkeys(s for s, _ in queue))
        self.assertEqual(stages, ["B2", "B1", "A1_DC_zuerst", "A2", "A_T2", "B_T2"])
        self.assertEqual(next(s for stage, s in queue if stage == "A1_DC_zuerst"), "r1b_D_C")
        # START's finished K1 may be listed by frozen B1, but is skipped by state.
        self.assertEqual(len(queue), len(set(s for _, s in queue)))
        self.assertNotIn("A_T2", [s for s, _ in A.sequence(False)])

    def test_completed_sessions_skipped_and_status_durable(self):
        self.finish(self.session)
        A.durable_sync(self.work / self.session, self.b / self.session, self.session)
        c = self.controller()
        self.assertEqual(c.run(), 0)
        self.assertFalse(self.calls)
        status = A.load(c.status_file)
        self.assertTrue(status["safe_to_release"])
        self.assertEqual(status["state"], "complete")
        self.assertEqual(status, A.load(c.directory / "status.json"))

    def test_missing_dependency_fails_once_without_wait(self):
        self.engine.waiting = ["r1b_K_S1"]
        c = self.controller()
        with patch.object(A.time, "sleep", side_effect=AssertionError("dependency polling")):
            self.assertEqual(c.run(), 4)
        self.assertEqual(self.engine.checks, [self.session])
        self.assertFalse(self.calls)
        self.assertEqual(A.load(c.status_file)["state"], "blocked")

    def test_only_guard_return_continues_and_global_budget_is_fixed(self):
        count = 0
        def execute(stage, session, env, log):
            nonlocal count
            count += 1
            self.calls.append(env)
            self.finish(session, partial=count == 1)
            return 3 if count == 1 else 0
        start = time.time() - 30
        c = self.controller(executor=execute, runtime_start=start, max_hours=1)
        self.assertEqual(c.run(), 0)
        self.assertEqual(count, 2)
        self.assertEqual([float(env["R1B_RUNTIME_T0"]) for env in self.calls], [start, start])
        self.assertTrue(all(0 < float(env["R1B_GUARD_S"]) <= 3570 for env in self.calls))
        self.assertNotEqual(self.calls[0]["R1B_T0"], self.calls[1]["R1B_T0"])

    def test_failed_science_has_no_outer_retry_but_saved_checkpoint(self):
        def execute(*args):
            self.calls.append(1)
            self.finish(self.session, partial=True)
            return 1
        c = self.controller(executor=execute)
        self.assertEqual(c.run(), 2)
        self.assertEqual(self.calls, [1])
        self.assertTrue(A.load(c.status_file)["safe_to_release"])
        self.assertTrue((self.b / self.session / "session_partial.json").is_file())

    def test_failed_sync_preserves_runtime_and_local_copy(self):
        c = self.controller()
        with patch.object(A, "durable_sync", side_effect=OSError("Drive disconnected")):
            self.assertEqual(c.run(), 5)
        self.assertFalse(A.load(c.status_file)["safe_to_release"])
        self.assertEqual(A.load(c.status_file)["state"], "sync_failed")
        self.assertTrue((self.work / self.session / "result.bin").is_file())

    def test_early_bootstrap_and_lock_errors_never_release_unverified_work(self):
        for failure in ("bootstrap", "lock"):
            with self.subTest(failure=failure):
                self.finish(self.session)
                c = self.controller(run_id=failure)
                if failure == "bootstrap":
                    self.engine.bootstrap_error = "missing START"
                else:
                    self.engine.bootstrap_error = None
                    (c.directory / ".writer.lock").mkdir()
                self.assertEqual(c.run(), 5)
                self.assertFalse(A.load(c.status_file)["safe_to_release"])
        self.assertTrue((c.directory / ".writer.lock").exists())

    def test_environment_isolated_and_soft_budget_pause(self):
        c = self.controller()
        with patch.dict(os.environ, {"R1B_CAPTURE_HIDDEN": "1", "R1B_MOCK": "1", "MOCK": "1", "QUICK": "1"}):
            env = c.environment("B2", self.session, 42)
        self.assertNotIn("R1B_CAPTURE_HIDDEN", env)
        self.assertNotIn("R1B_MOCK", env)
        self.assertEqual((env["MOCK"], env["QUICK"], env["R1B_ACCOUNT"]), ("0", "0", "B"))
        self.assertEqual(float(env["R1B_GUARD_S"]), 42)
        c.pause_file.parent.mkdir(parents=True)
        c.pause_file.touch()
        self.assertEqual(c.run(), 3)
        self.assertFalse(self.calls)

    def test_pending_script_sessions_fail_closed(self):
        c = self.controller()
        c.queue = [("A1_DC_zuerst", "r1b_R_S1")]
        self.assertEqual(c.run(), 2)
        self.assertFalse(self.calls)
        self.assertIn("legacy script", A.load(c.status_file)["error"])


class SyncTests(unittest.TestCase):
    def test_repair_same_size_corruption_and_marker_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, drive = Path(tmp) / "local", Path(tmp) / "Drive"
            work.mkdir(); drive.mkdir()
            (work / "result.bin").write_bytes(b"right")
            (drive / "result.bin").write_bytes(b"wrong")
            A.write_verified(work / "session_done.json", dict(session="r1b_K_S2a", mock=False,
                             files={"result.bin": A.digest(work / "result.bin")}))
            (drive / "session_partial.json").write_text("obsolete")
            result = A.durable_sync(work, drive, "r1b_K_S2a")
            self.assertEqual((drive / "result.bin").read_bytes(), b"right")
            self.assertIn("session_done.json", result["files"])
            self.assertFalse((drive / "session_partial.json").exists())
            self.assertTrue(A.durable_sync(drive, drive, "r1b_K_S2a")["verified"])

    def test_corrupt_current_producer_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            work, drive = Path(tmp) / "local", Path(tmp) / "Drive"
            work.mkdir()
            (work / "x").write_bytes(b"broken")
            A.write_verified(work / "session_done.json", dict(session="r1b_K_S2a", files={"x": "0" * 64}))
            with self.assertRaisesRegex(ValueError, "producer"):
                A.durable_sync(work, drive, "r1b_K_S2a")
            self.assertFalse(drive.exists())

    @unittest.skipUnless(os.name == "posix", "process groups require POSIX")
    def test_interrupt_stops_real_session_group_including_term_ignoring_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pidfile = base / "grandchild.pid"
            child = ("import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                     f"open({str(pidfile)!r},'w').write(str(os.getpid()));time.sleep(120)")
            leader = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c'," + repr(child) + "]);time.sleep(120)"
            real_popen = subprocess.Popen
            def popen(command, **kwargs):
                return real_popen([sys.executable, "-c", leader], **kwargs)
            def interrupt(seconds):
                deadline = time.monotonic() + 5
                while not pidfile.is_file() and time.monotonic() < deadline:
                    time.sleep(.01)
                if not pidfile.is_file():
                    raise AssertionError("test child did not start")
                raise KeyboardInterrupt()
            c = A.Controller(base / "A", base / "B", base / "work_htq_v2", base / "root",
                             engine=FakeEngine(base / "A", base / "B"))
            # Patch only the module's time proxy; the test's bounded polling
            # keeps the real clock and sleep.
            clock = SimpleNamespace(monotonic=time.monotonic, sleep=interrupt)
            with open(base / "log", "w") as log, patch.object(A.subprocess, "Popen", side_effect=popen), patch.object(A, "time", clock):
                with self.assertRaises(KeyboardInterrupt):
                    c.execute("B2", "r1b_K_S2a", dict(os.environ), log)
            pid = int(pidfile.read_text())
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                stat = Path(f"/proc/{pid}/stat")
                if not stat.exists() or stat.read_text().split()[2] == "Z":
                    break
                time.sleep(.01)
            else:
                os.kill(pid, signal.SIGKILL)
                self.fail("detached science grandchild survived controller cleanup")


if __name__ == "__main__":
    unittest.main()
