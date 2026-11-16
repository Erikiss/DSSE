"""Model-free checks for the notebook hand-off and process isolation contract."""
import ast
import importlib.util
import json
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
            self.assertEqual(len(first), 9)
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


if __name__ == "__main__":
    unittest.main()
