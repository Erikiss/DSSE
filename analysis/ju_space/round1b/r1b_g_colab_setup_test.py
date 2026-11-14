"""Exercise the G installation cell without ensurepip or package downloads.

The virtual environments are real. Only pip/network operations and the heavy
dependency metadata are simulated, so interrupted setup can be tested cheaply.
"""
from __future__ import annotations

import contextlib
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import venv


REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "g_colab_notebook_builder", REPO / "notebooks/build_runde1b_gemma_notebooks.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)


class PackageInstaller:
    """Replace downloads while retaining the installation cell's control flow."""

    def __init__(self):
        self.calls = []
        self.pip_targets = set()
        self.versions = {}
        self.fail_bootstrap = False
        self.fail_install = False
        self.bad_version = False

    def run(self, args, **kwargs):
        args = list(map(str, args))
        self.calls.append(args)
        code, output = 0, ""
        if args[1:4] == ["-m", "pip", "--python"]:
            # The notebook must seed the target, never alter the kernel env.
            assert args[0] == sys.executable
            target = args[4]
            assert target != sys.executable
            assert Path(target).is_file()
            if self.fail_bootstrap:
                code = 1
            else:
                self.pip_targets.add(target)
        elif args[1:4] == ["-m", "pip", "--version"]:
            code = 0 if args[0] in self.pip_targets else 1
        elif args[1:4] == ["-m", "pip", "install"]:
            assert args[0] in self.pip_targets
            assert args[0] != sys.executable
            if self.fail_install:
                code = 1
            else:
                versions = self.versions.setdefault(args[0], {})
                for value in args[4:]:
                    if "==" in value:
                        name, version = value.split("==", 1)
                        versions[name] = version
        elif args[1] == "-c" and "importlib.metadata" in args[2]:
            versions = dict(self.versions.get(args[0], {}))
            if self.bad_version:
                versions["transformers"] = "0.0.0"
            output = json.dumps(versions)
        else:
            raise AssertionError("Unexpected subprocess in the dependency cell: " + repr(args))
        if code and kwargs.get("check"):
            raise subprocess.CalledProcessError(code, args, output, "simulated interruption")
        return subprocess.CompletedProcess(args, code, output, "")

    @property
    def install_calls(self):
        return [args for args in self.calls if args[1:4] == ["-m", "pip", "install"]]


class ColabBootstrapTests(unittest.TestCase):
    def namespace(self, root, *, minor=None, mode="mock"):
        interpreter = sys if minor is None else SimpleNamespace(
            executable=sys.executable, version_info=SimpleNamespace(major=3, minor=minor))
        return dict(BASE=Path(root), MODE=mode, SLOT="PILOT", COMMIT="a" * 40,
                    INSTALL_DEPENDENCIES=True, sys=interpreter, os=os,
                    subprocess=subprocess, fcntl=fcntl, json=json, Path=Path)

    def execute(self, ns, installer):
        with patch.object(venv.EnvBuilder, "_setup_pip", side_effect=AssertionError("ensurepip unavailable")) as ensurepip:
            with patch.object(subprocess, "run", side_effect=installer.run), contextlib.redirect_stdout(io.StringIO()):
                exec(B.DEPENDENCIES, ns)
            ensurepip.assert_not_called()

    def test_fresh_environment_bootstraps_without_ensurepip(self):
        with tempfile.TemporaryDirectory() as td:
            ns, installer = self.namespace(td), PackageInstaller()
            self.execute(ns, installer)
            self.assertTrue(ns["PYTHON"].is_file())
            self.assertEqual(json.loads((ns["ENV_PATH"] / "g_dependencies.json").read_text()), ns["PINS"])
            self.assertTrue(any(args[1:4] == ["-m", "pip", "--python"] for args in installer.calls))
            self.assertIn("https://download.pytorch.org/whl/cpu", installer.install_calls[0])

    def test_python_left_by_failed_bootstrap_is_repaired(self):
        with tempfile.TemporaryDirectory() as td:
            ns, installer = self.namespace(td), PackageInstaller()
            installer.fail_bootstrap = True
            with self.assertRaises(subprocess.CalledProcessError):
                self.execute(ns, installer)
            self.assertTrue(ns["PYTHON"].is_file())
            self.assertFalse((ns["ENV_PATH"] / "g_dependencies.json").exists())
            installer.fail_bootstrap = False
            self.execute(ns, installer)
            self.assertEqual(sum(args[1:4] == ["-m", "pip", "--python"] for args in installer.calls), 2)

    def test_interrupted_package_install_does_not_mark_environment_ready(self):
        with tempfile.TemporaryDirectory() as td:
            ns, installer = self.namespace(td), PackageInstaller()
            installer.fail_install = True
            with self.assertRaises(subprocess.CalledProcessError):
                self.execute(ns, installer)
            self.assertFalse((ns["ENV_PATH"] / "g_dependencies.json").exists())
            installer.fail_install = False
            self.execute(ns, installer)
            self.assertEqual(sum(args[1:4] == ["-m", "pip", "--python"] for args in installer.calls), 1)

    def test_corrupted_marker_and_missing_package_are_repaired(self):
        with tempfile.TemporaryDirectory() as td:
            ns, installer = self.namespace(td), PackageInstaller()
            self.execute(ns, installer)
            marker = ns["ENV_PATH"] / "g_dependencies.json"
            marker.write_text("{interrupted")
            self.execute(ns, installer)
            self.assertEqual(json.loads(marker.read_text()), ns["PINS"])
            del installer.versions[str(ns["PYTHON"])]["accelerate"]
            self.execute(ns, installer)
            self.assertEqual(len(installer.install_calls), 6)
            self.assertEqual(installer.versions[str(ns["PYTHON"])]["accelerate"], ns["PINS"]["accelerate"])

    def test_marker_requires_verified_post_install_versions(self):
        with tempfile.TemporaryDirectory() as td:
            ns, installer = self.namespace(td), PackageInstaller()
            installer.bad_version = True
            with self.assertRaisesRegex(RuntimeError, "unvollständig"):
                self.execute(ns, installer)
            self.assertFalse((ns["ENV_PATH"] / "g_dependencies.json").exists())
            installer.bad_version = False
            self.execute(ns, installer)

    def test_healthy_environment_is_reused_without_reinstallation(self):
        with tempfile.TemporaryDirectory() as td:
            ns, installer = self.namespace(td), PackageInstaller()
            self.execute(ns, installer)
            installs = len(installer.install_calls)
            self.execute(ns, installer)
            self.assertEqual(len(installer.install_calls), installs)

    def test_python_versions_and_cpu_cuda_use_separate_environments(self):
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for minor, mode in [(12, "mock"), (13, "mock"), (13, "pilot")]:
                ns, installer = self.namespace(td, minor=minor, mode=mode), PackageInstaller()
                self.execute(ns, installer)
                paths.append(ns["ENV_PATH"])
                self.assertIn(f"py3{minor}", ns["ENV_PATH"].name)
                expected = "cpu" if mode == "mock" else "cu128"
                self.assertIn("https://download.pytorch.org/whl/" + expected, installer.install_calls[0])
            self.assertEqual(len(set(paths)), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
