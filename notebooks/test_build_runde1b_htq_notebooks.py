"""Model-free checks for the notebook hand-off and process isolation contract."""
import ast
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("htq_builder", HERE / "build_runde1b_htq_notebooks.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class NotebookTests(unittest.TestCase):
    def test_generated_notebooks_parse_and_leave_sources_unchanged(self):
        originals = {p: p.read_bytes() for p in HERE.glob("runde1b_*.ipynb") if not p.name.startswith("runde1b_HTQ_")}
        with tempfile.TemporaryDirectory() as d:
            first = {p.name: p.read_bytes() for p in builder.build(Path(d))}
            second = {p.name: p.read_bytes() for p in builder.build(Path(d))}
            self.assertEqual(first, second)
            self.assertEqual(len(first), 10)
            for filename, raw in first.items():
                doc = json.loads(raw)
                self.assertEqual(doc["metadata"]["htq_variant"]["branch"], builder.BRANCH)
                for c in doc["cells"]:
                    if c["cell_type"] == "code":
                        ast.parse(builder.text(c), filename=filename)
                        self.assertEqual(c["outputs"], [])
        self.assertTrue(all(p.read_bytes() == raw for p, raw in originals.items()))

    def test_real_drive_policy_never_exported_into_host_environment(self):
        for name, spec in builder.SPECS.items():
            doc = builder.transform(name, spec)
            writes = []
            for c in doc["cells"]:
                if c["cell_type"] != "code":
                    continue
                for n in ast.walk(ast.parse(builder.text(c))):
                    if isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store):
                        if isinstance(n.value, ast.Attribute) and isinstance(n.value.value, ast.Name):
                            if n.value.value.id == "os" and n.value.attr == "environ":
                                writes.append(ast.unparse(n))
            self.assertEqual(writes, [], name)
            selftest = next(builder.text(c) for c in doc["cells"] if "# Selbsttest (nur Code)" in builder.text(c))
            self.assertIn("env=dict(TEST_ENV, RUN_DEVICE=", selftest)
            self.assertNotIn("VARIANT_ENV", selftest)

    def test_start_runs_only_k_s1_and_continues_after_guard(self):
        with tempfile.TemporaryDirectory() as d:
            runs = Path(d)
            commands = []

            def run(cmd, *, env):
                commands.append((cmd, env))
                if len(commands) == 1:
                    return SimpleNamespace(returncode=3)
                p = runs / "r1b_K_S1"
                (p / "hn").mkdir(parents=True)
                (p / "session_done.json").write_text("{}")
                (p / "hn/hn_set.json").write_text(json.dumps({"H_op_name": "H_tq", "H_op": [[1, 2]]}))
                return SimpleNamespace(returncode=0)

            env = dict(NUR_SELBSTTEST=False, RUNS=d, ROOT="/models", WORK="/isolated-work", R1B="/code",
                       T0=0, LAUFZEIT_MAX_H=23, VARIANT_ENV={"R1B_H_POLICY": "htq_v2"},
                       check_variant=lambda path, account: None, json=json,
                       os=SimpleNamespace(path=SimpleNamespace(isfile=lambda path: Path(path).is_file())),
                       sys=SimpleNamespace(executable="python"), time=SimpleNamespace(time=lambda: 1),
                       subprocess=SimpleNamespace(run=run))
            exec(compile(builder.start_main(), "START", "exec"), env)
            self.assertEqual(env["RC"], 0)
            self.assertEqual(len(commands), 2)
            for cmd, proc_env in commands:
                self.assertIn("r1b_K_S1", cmd)
                self.assertIn("/code/r1b_run_session.py", cmd)
                self.assertNotIn("/code/r1b_run_slot.py", cmd)
                self.assertEqual(proc_env["R1B_H_POLICY"], "htq_v2")

    def test_readiness_is_checked_before_copy_and_main(self):
        for name, spec in builder.SPECS.items():
            doc = builder.transform(name, spec)
            sources = [builder.text(c) for c in doc["cells"]]
            guard = next(i for i, s in enumerate(sources) if "# Varianten-Bootstrap" in s)
            copies = [i for i, s in enumerate(sources) if '"--copy-only"' in s]
            main = next(i for i, s in enumerate(sources) if "# Hauptlauf:" in s or "# Nur das neue K_S1" in s or "# Zuerst r1b_D_C" in s)
            self.assertTrue(all(guard < i for i in copies), name)
            self.assertLess(guard, main)
            if name != "HTQ_START":
                self.assertIn("_htq_pair_ready.json", sources[guard])
                self.assertIn("session_done.json", sources[guard])

    def test_failed_b_seed_streams_wrapped_cause_and_reuses_a(self):
        calls = []

        def popen(command, **options):
            calls.append((command, options))
            action, account = command[-3], command[-1]
            failure = account == "B" and action == "seed"
            output = ("PermissionError: symlink outside source root: /drive/" + "long-path/" * 30 + "\n"
                      if failure else "Konto A: vollständige Vorbereitung wird wiederverwendet.\n")
            return SimpleNamespace(stdout=io.StringIO(output), wait=lambda: int(failure), poll=lambda: int(failure))

        with tempfile.TemporaryDirectory() as d:
            source_a, source_b = Path(d) / "A", Path(d) / "B"
            source_a.mkdir(); source_b.mkdir()
            output = io.StringIO()
            environment = dict(R1B="/current/code", TEST_ENV={"SAFE": "yes"}, NUR_SELBSTTEST=False,
                               SOURCE_A=str(source_a), SOURCE_B=str(source_b), B_RUNS=str(source_b / builder.SUBDIR),
                               COMMIT="current", os=os, sys=sys, json=json,
                               subprocess=SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT))
            with contextlib.redirect_stdout(output):
                with self.assertRaisesRegex(RuntimeError, "Konto B: seed fehlgeschlagen \\(Exit 1\\)"):
                    exec(compile(builder.prepare_code("start"), "prepare", "exec"), environment)
            visible = output.getvalue()
            self.assertIn("[Konto A] seed: Exit 0", visible)
            self.assertIn("Vorbereitung wird wiederverwendet", visible)
            self.assertIn("[Konto B] seed:", visible)
            self.assertIn("PermissionError: symlink outside source root", visible)
            self.assertIn("[Konto B] seed: Exit 1", visible)
            self.assertTrue(all(len(line) <= 72 for line in visible.splitlines()))
            self.assertEqual([(c[-3], c[-1]) for c, _ in calls], [("seed", "A"), ("check", "A"), ("seed", "B")])
            self.assertTrue(all(o["stderr"] == subprocess.STDOUT and o["stdout"] == subprocess.PIPE for _, o in calls))
            self.assertTrue(all(o["text"] and o["bufsize"] == 1 and o["env"] == {"SAFE": "yes"} for _, o in calls))
            self.assertFalse((source_a / builder.SUBDIR / builder.PAIR_MARKER).exists())

    def test_interrupted_stream_terminates_child(self):
        calls = []

        class InterruptedOutput(io.StringIO):
            def __next__(self):
                raise KeyboardInterrupt()

        process = SimpleNamespace(stdout=InterruptedOutput(""), poll=lambda: None,
                                  terminate=lambda: calls.append("terminate"), wait=lambda **kw: calls.append("wait"))
        environment = dict(R1B="/current/code", TEST_ENV={}, sys=sys,
                           subprocess=SimpleNamespace(Popen=lambda *args, **kw: process,
                                                      PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT,
                                                      TimeoutExpired=subprocess.TimeoutExpired))
        exec(compile(builder.variant_helpers(), "helpers", "exec"), environment)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                environment["run_variant"]("seed", "/unchanged/source", "B")
        self.assertEqual(calls, ["terminate", "wait"])
        self.assertTrue(process.stdout.closed)

    def test_prep_check_is_cpu_only_and_uses_same_bootstrap(self):
        doc = builder.prep_check()
        sources = [builder.text(c) for c in doc["cells"] if c["cell_type"] == "code"]
        all_code = "\n".join(sources)
        for forbidden in ("torch", "cuda", "fetch-models", "--selftest", "r1b_run_session.py", "r1b_run_slot.py", "rmtree", "os.remove", "unlink"):
            self.assertNotIn(forbidden, all_code)
        self.assertIn('tempfile.mkdtemp(prefix="dsse_htq_prep_", dir="/content")', all_code)
        self.assertIn('"analysis/ju_space"', all_code)
        self.assertIn('"results/ju_space/plans"', all_code)
        self.assertEqual(sources[-1], builder.logged_prep_code())
        self.assertNotIn("accelerator", doc["metadata"])

    def test_prep_log_persists_success_and_running_state_without_flooding_output(self):
        with tempfile.TemporaryDirectory() as d:
            status = Path(d) / "htq_prep_check_status.json"
            status.write_text(json.dumps({"status": "success", "run_id": "old"}))
            program = '''import sys
with open(HTQ_PREP_STATUS) as f:
    assert json.load(f)["status"] == "running"
for i in range(100):
    print("full child output", i)
print("stderr preserved", file=sys.stderr)
'''
            env = dict(SOURCE_A=d, COMMIT="abc123")
            visible = io.StringIO()
            with contextlib.redirect_stdout(visible):
                exec(compile(builder.logged_prep_code(program), "logged-prep", "exec"), env)
            saved = json.loads(status.read_text())
            log = (Path(d) / "htq_prep_check.log").read_text()
            self.assertEqual(saved["status"], "success")
            self.assertEqual(saved["commit"], "abc123")
            self.assertEqual(saved["stage"], "complete")
            self.assertNotEqual(saved["run_id"], "old")
            self.assertTrue(saved["started_utc"].endswith("+00:00"))
            self.assertTrue(saved["finished_utc"].endswith("+00:00"))
            self.assertIn("full child output 99", log)
            self.assertIn("stderr preserved", log)
            self.assertNotIn("full child output", visible.getvalue())
            self.assertLess(len(visible.getvalue().splitlines()), 10)

    def test_prep_log_persists_child_failure_and_appends_prior_log(self):
        with tempfile.TemporaryDirectory() as d:
            logfile = Path(d) / "htq_prep_check.log"
            logfile.write_text("earlier diagnostic\n", encoding="utf-8")
            process = SimpleNamespace(stdout=io.StringIO("PermissionError: Konto B cannot write\n"),
                                      wait=lambda: 1, poll=lambda: 1)
            env = dict(SOURCE_A=d, R1B="/current/code", TEST_ENV={}, sys=sys,
                       subprocess=SimpleNamespace(Popen=lambda *a, **kw: process,
                                                  PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT))
            program = builder.variant_helpers() + '\nrun_variant("seed", "/source/B", "B")\n'
            visible = io.StringIO()
            with contextlib.redirect_stdout(visible):
                with self.assertRaisesRegex(RuntimeError, "Ursache in htq_prep_check.log"):
                    exec(compile(builder.logged_prep_code(program), "logged-prep", "exec"), env)
            log = logfile.read_text()
            saved = json.loads((Path(d) / "htq_prep_check_status.json").read_text())
            self.assertTrue(log.startswith("earlier diagnostic\n"))
            self.assertIn("PermissionError: Konto B cannot write", log)
            self.assertIn("[Konto B] seed: Exit 1", log)
            self.assertIn("Traceback", log)
            self.assertEqual(saved["status"], "failed")
            self.assertIn("Konto B", saved["error"])
            self.assertNotIn("PermissionError", visible.getvalue())

    def test_prep_log_catches_missing_global_before_child_starts(self):
        with tempfile.TemporaryDirectory() as d:
            env = dict(SOURCE_A=d)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "Ursache in htq_prep_check.log"):
                    exec(compile(builder.logged_prep_code(), "logged-prep", "exec"), env)
            saved = json.loads((Path(d) / "htq_prep_check_status.json").read_text())
            log = (Path(d) / "htq_prep_check.log").read_text()
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["error_type"], "NameError")
            self.assertIn("R1B", saved["error"])
            self.assertIn("NameError", log)
            self.assertIsNone(saved["commit"])

    def test_prep_log_open_failure_never_claims_saved_or_creates_mount(self):
        with tempfile.TemporaryDirectory() as d:
            absent = Path(d) / "unmounted" / "dsse_runs"
            visible = io.StringIO()
            with contextlib.redirect_stdout(visible):
                with self.assertRaisesRegex(RuntimeError, "Drive-Protokoll nicht gespeichert"):
                    exec(compile(builder.logged_prep_code("raise AssertionError('must not run')"), "logged-prep", "exec"),
                         dict(SOURCE_A=str(absent)))
            self.assertFalse(absent.exists())
            self.assertIn("NICHT", visible.getvalue())
            self.assertNotIn("Vollständiges Protokoll:", visible.getvalue())


if __name__ == "__main__":
    unittest.main()
