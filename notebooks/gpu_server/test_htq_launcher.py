"""Model-free orchestration tests; never execute a real model or provision a host."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("htq_launcher", HERE / "htq_launcher.py")
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


class LauncherTests(unittest.TestCase):
    def source(self, root, account):
        source = root / ("source_" + account) / "variants/htq_v2"
        source.mkdir(parents=True)
        L.write(source / L.MARKER, dict(schema=1, account=account, variant=L.VARIANT, h_policy=L.VARIANT,
                                       source="/original/colab/dsse_runs", destination="/original/colab/dsse_runs/variants/htq_v2", files={}))
        (source / "important.pt").write_bytes(b"all .pt files must survive")
        if account == "A":
            L.write(source / "r1b_R_S0/session_done.json", dict(session="r1b_R_S0", account="A", files={}))
            exported = source / "r1b_xfer/fixture.pt"
            exported.parent.mkdir()
            exported.write_bytes(b"unchanged R0 export")
            L.write(source / "r1b_xfer/xfer_manifest.json", {"fixture.pt": dict(sha256=L.sha(exported))})
        return source

    def initial(self, directory):
        directory = Path(directory)
        a, b = self.source(directory, "A"), self.source(directory, "B")
        state = directory / "server_state"
        L.initialise(state, a, b)
        return state, a, b

    def test_import_preserves_tensor_bytes_and_records_relocation(self):
        with tempfile.TemporaryDirectory() as d:
            state, a, b = self.initial(d)
            import r1b_htq_variant as H
            for account, original in (("A", a), ("B", b)):
                target = L.runs(state, account)
                self.assertEqual((target / "important.pt").read_bytes(), (original / "important.pt").read_bytes())
                evidence = L.read(state / f"import_{account}.json")
                self.assertEqual(evidence["original_seed_marker_utf8"], (original / L.MARKER).read_text())
                self.assertEqual(evidence["original_seed_sha256"], L.sha(original / L.MARKER))
                self.assertEqual(H.check(target, account)["destination"], str(target))
            with self.assertRaisesRegex(ValueError, "already initialised"):
                L.initialise(state, a, b)

    def test_shortcuts_and_concurrent_coordinators_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = self.source(root, "A")
            (source / "linked.pt").symlink_to(source / "important.pt")
            with self.assertRaisesRegex(ValueError, "shortcuts"):
                L.copy_verified(source, root / "copy")
            with L.exclusive(root / "state"):
                with self.assertRaisesRegex(ValueError, "another coordinator"):
                    with L.exclusive(root / "state"):
                        self.fail("must not get here")

    def test_blackwell_and_hopper_fail_preflight_without_gate_override(self):
        for name, cc in (("NVIDIA B200", [10, 0]), ("NVIDIA H100", [9, 0])):
            result = SimpleNamespace(returncode=0, stdout=json.dumps(dict(name=name, cc=cc, memory=180e9,
                                     transformers="5.22.2")), stderr="")
            with patch.object(L.subprocess, "run", return_value=result):
                with self.assertRaisesRegex(ValueError, "not admitted by frozen SG2"):
                    L.preflight([0])

    def test_parallel_workers_are_private_and_publish_verified_tensors(self):
        with tempfile.TemporaryDirectory() as d:
            state, _, _ = self.initial(d)
            calls = []
            sessions = ["r1b_L_A", "r1b_K_S2a", "r1b_K_S4"]

            def plan(_state, pending):
                return {s: dict(state="done" if (L.runs(state, "B") / s / "session_done.json").exists() else "ready") for s in pending}

            def popen(command, *, env, stdout, stderr, start_new_session):
                self.assertTrue(start_new_session)
                session = command[command.index("--session") + 1]
                work = Path(command[command.index("--work") + 1])
                self.assertEqual(work.parent.name, "work_htq_v2")  # actual runner's isolation guard
                work.mkdir(parents=True)
                (work / "states.pt").write_bytes(session.encode())
                L.write(work / "session_done.json", dict(session=session, account="B", h_policy=L.VARIANT,
                                                          files={"states.pt": L.sha(work / "states.pt")}))
                calls.append((command, env, work))
                return SimpleNamespace(poll=lambda: 0)

            with patch.object(L, "preflight", return_value=[dict(name="A100")]), patch.object(L, "plan", side_effect=plan), \
                    patch.object(L.subprocess, "Popen", side_effect=popen), patch.object(L, "stop_worker") as stopped:
                L.run(state, sessions, [0, 1])
            self.assertEqual(stopped.call_count, 3)
            self.assertEqual(len(calls), 3)
            self.assertEqual([c[0][c[0].index("--session") + 1] for c in calls[:2]], ["r1b_L_A", "r1b_K_S2a"])
            self.assertEqual({c[1]["CUDA_VISIBLE_DEVICES"] for c in calls[:2]}, {"0", "1"})
            self.assertEqual(len({str(c[2]) for c in calls}), 3)
            self.assertNotEqual(calls[0][1]["R1B_RUNS_B"], calls[1][1]["R1B_RUNS_B"])
            self.assertTrue(all(call[1]["R1B_ACCOUNT"] == "B" for call in calls))
            for session in sessions:
                self.assertEqual((L.runs(state, "B") / session / "states.pt").read_bytes(), session.encode())

    def test_failed_publication_does_not_replace_imported_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            state, _, _ = self.initial(d)
            original = L.runs(state, "B") / "r1b_K_S1"
            original.mkdir()
            (original / "prefix.pt").write_bytes(b"keep")
            worker = state / "workers/gpu0"
            worker.mkdir(parents=True)
            job = L.stage_job(state, worker, "r1b_K_S1")
            work = job / "work_htq_v2/r1b_K_S1"
            work.mkdir(parents=True)
            (work / "states.pt").write_bytes(b"corrupt")
            L.write(work / "session_done.json", dict(session="r1b_K_S1", account="B", h_policy=L.VARIANT,
                                                      files={"states.pt": "0" * 64}))
            with self.assertRaisesRegex(ValueError, "verification"):
                L.publish_job(state, job, "r1b_K_S1")
            self.assertEqual((original / "prefix.pt").read_bytes(), b"keep")

    def test_import_rejects_corrupt_current_producer_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, b = self.source(root, "A"), self.source(root, "B")
            L.write(b / "r1b_K_S1/session_partial.json", dict(session="r1b_K_S1", files={"state.pt": "0" * 64}))
            (b / "r1b_K_S1/state.pt").write_bytes(b"corrupt current artifact")
            with self.assertRaisesRegex(ValueError, "current producer hash"):
                L.initialise(root / "state", a, b)
            self.assertFalse((root / "state/config.json").exists())

    def test_prefix_uses_latest_step_owner_not_overwritten_historical_hash(self):
        with tempfile.TemporaryDirectory() as d:
            source = self.source(Path(d), "B")
            session = source / "r1b_K_S1"
            state = session / "same.pt"
            state.parent.mkdir()
            state.write_bytes(b"latest")
            for step, digest in (("k1_check", "0" * 64), ("k4j", L.sha(state))):
                L.write(session / f"steps/{step}.json", dict(status="ok", outputs={"same.pt": dict(sha256=digest)}))
            self.assertEqual(L.verify_producer_hashes(source), 1)

    def test_process_cleanup_targets_whole_group(self):
        process = SimpleNamespace(pid=999999, wait=lambda **kw: 0)
        with patch.object(L.os, "killpg") as signals:
            L.stop_worker(process)
        self.assertEqual([c.args for c in signals.call_args_list], [(999999, L.signal.SIGTERM), (999999, L.signal.SIGKILL)])

    @unittest.skipUnless(sys.platform.startswith("linux") and Path("/proc").is_dir(), "POSIX process-group test")
    def test_real_worker_group_stops_sigterm_ignoring_grandchild_before_unlock(self):
        child_code = "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(120)"
        worker_code = ("import subprocess,sys,time;"
                       f"p=subprocess.Popen([sys.executable,'-u','-c',{child_code!r}],stdout=subprocess.PIPE,text=True);"
                       "p.stdout.readline();print(p.pid,flush=True);time.sleep(120)")
        proc = subprocess.Popen([sys.executable, "-u", "-c", worker_code], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            child = int(proc.stdout.readline().strip())
            with tempfile.TemporaryDirectory() as d:
                with L.exclusive(d):
                    L.stop_worker(proc)
                    running = True
                    for _ in range(50):
                        try:
                            status = Path(f"/proc/{child}/stat").read_text().split()[2]
                            running = status not in ("Z", "X")
                        except FileNotFoundError:
                            running = False
                        if not running:
                            break
                        time.sleep(0.02)
                    self.assertFalse(running, "grandchild remains active after worker termination")
                    self.assertTrue((Path(d) / ".coordinator.lock").is_dir())
                self.assertFalse((Path(d) / ".coordinator.lock").exists())
        finally:
            L.stop_worker(proc)
            proc.stdout.close()

    def test_private_work_path_passes_actual_runner_isolation_guard(self):
        import r1b_run_session as R
        with tempfile.TemporaryDirectory() as d:
            state, _, _ = self.initial(d)
            worker = state / "workers/gpu0"
            worker.mkdir(parents=True)
            job = L.stage_job(state, worker, "r1b_K_S1")
            work = job / "work_htq_v2/r1b_K_S1"
            L.write(work / "session_done.json", dict(session="r1b_K_S1", h_policy=L.VARIANT))
            a, b = (job / account / "dsse_runs" / L.SUBDIR for account in ("A", "B"))
            old_logs = len(R.LOGF)
            try:
                env = L.execution_env(state, a, b, worker / "root", 0)
                env["R1B_ACCOUNT"] = "B"
                with L.installed_env(env):
                    code = R.run_session("r1b_K_S1", str(worker / "root"), str(work), str(b))
                self.assertEqual(code, R.EXIT_DONE)
            finally:
                for stream in R.LOGF[old_logs:]:
                    stream.close()
                del R.LOGF[old_logs:]

    def test_pending_escape_group_scripts_are_refused_but_completed_allowed(self):
        sessions = ["r1b_R_S1", "r1b_R_S2_14"]
        with self.assertRaisesRegex(ValueError, "completed script sessions"):
            L.reject_pending_scripts(sessions, {s: dict(state="ready") for s in sessions})
        L.reject_pending_scripts(sessions, {s: dict(state="done") for s in sessions})

    def test_explicit_mock_and_wrong_account_producers_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            source = self.source(Path(d), "B")
            record = source / "r1b_K_S1/session_done.json"
            for metadata in (dict(mock=True), dict(account="A")):
                L.write(record, dict(session="r1b_K_S1", files={}, **metadata))
                with self.assertRaises(ValueError):
                    L.verify_producer_hashes(source)

    def test_space_guard_refuses_snapshot_before_copy(self):
        with tempfile.TemporaryDirectory() as d:
            state, _, _ = self.initial(d)
            worker = state / "workers/gpu0"
            worker.mkdir(parents=True)
            with patch.object(L.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(ValueError, "insufficient space"):
                    L.stage_job(state, worker, "r1b_K_S1")
            self.assertEqual(list(worker.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
