"""Model-free tests of variant isolation and verified checkpoint reuse."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import r1b_htq_variant as V


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(root, rel, value="data"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value)
    return p


def hashes(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*")
            if p.is_file() and "variants" not in p.relative_to(root).parts}


class SeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "dsse_runs"
        self.source.mkdir()

    def prefix(self):
        root = self.source / "r1b_K_S1"
        for st in V._prefix_steps():
            outputs = {}
            for rel in st.out:
                self.assertFalse(any(c in rel for c in "*?["), "fixture requires explicit prefix outputs")
                p = write(root, rel, f"{st.name}:{rel}")
                side = write(root, rel + ".prov.json", json.dumps({"step": st.name}))
                for q in (p, side):
                    outputs[q.relative_to(root).as_posix()] = {"sha256": digest(q), "size": q.stat().st_size}
            write(root, f"steps/{st.name}.json", json.dumps({"name": st.name, "status": "ok", "outputs": outputs}))
        for rel in ("session_done.json", "session_partial.json", "cache/fit_data.pt", "in/old_h.json",
                    "hn/hn_set.json", "steps/k2_sets.json", "k/k3/k3a_data.pt", "steps/k3a.json"):
            write(root, rel, "H+ result")
        return root

    def independent(self):
        for rel in ("r1b_R_S0/session_done.json", "r1b_R_S3/cache/keep.pt", "r1b_D_C/session_partial.json",
                    "r1b_D_C/cache/continue.pt", "r1b_K_S3/session_done.json", "gpu1/rk/b14/state.pt",
                    "r2_q2/theta/b14_q0.pt", "r1b_inputs/jlens.pt", "r1b_xfer/xfer_manifest.json"):
            write(self.source, rel)
        for rel in ("r1b_K_S4/session_done.json", "r1b_K_S4/k/k8b/part0.pt", "r1b_K_S2a/session_done.json",
                    "r1b_K_S7/session_done.json", "r1b_L_A/session_done.json", "r1b_ana/session_done.json",
                    "r1b_ana2/session_done.json", "r1b_declared_not_run.json"):
            write(self.source, rel, "must not reuse")

    def test_seed_b_preserves_original_and_excludes_h_dependents(self):
        self.prefix()
        self.independent()
        before = hashes(self.source)
        dest = Path(V.seed(self.source, "B"))
        self.assertEqual(before, hashes(self.source))
        self.assertEqual(dest, self.source / V.SUBDIR)
        rec = V.check(dest, "B")
        self.assertEqual(rec["h_policy"], V.VARIANT)
        for rel in ("r1b_D_C/session_partial.json", "r1b_D_C/cache/continue.pt", "r1b_K_S3/session_done.json",
                    "gpu1/rk/b14/state.pt", "r2_q2/theta/b14_q0.pt", "r1b_inputs/jlens.pt"):
            self.assertEqual(digest(dest / rel), digest(self.source / rel))
        self.assertFalse((dest / "r1b_K_S4").exists())
        self.assertFalse((dest / "r1b_K_S2a").exists())
        self.assertFalse((dest / "r1b_K_S7").exists())
        self.assertFalse((dest / "r1b_L_A").exists())
        self.assertFalse((dest / "r1b_ana").exists())
        self.assertFalse((dest / "r1b_declared_not_run.json").exists())
        k1 = dest / "r1b_K_S1"
        for rel in ("session_done.json", "session_partial.json", "cache", "in", "hn", "steps/k2_sets.json", "steps/k3a.json"):
            self.assertFalse((k1 / rel).exists(), rel)
        for st in V._prefix_steps():
            # The real runner can skip every copied independent prefix step.
            from r1b_run_session import step_done
            self.assertTrue(step_done(SimpleNamespace(work=str(k1), mock=False), st), st.name)
            marker = json.loads((k1 / f"steps/{st.name}.json").read_text())
            for rel, info in marker["outputs"].items():
                self.assertEqual(digest(k1 / rel), info["sha256"])

    def test_seed_a_does_not_reuse_stale_k1_mirror(self):
        self.prefix()
        self.independent()
        dest = Path(V.seed(self.source, "A"))
        self.assertFalse((dest / "r1b_K_S1").exists())
        self.assertTrue((dest / "r1b_R_S0/session_done.json").exists())

    def test_active_policy_seeds_original_har_and_requires_fresh_variant_cache(self):
        # Fixture contains only the original outputs, never the new HTQ marker.
        self.prefix()
        import r1b_sessions_K as registry
        try:
            with patch.dict(os.environ, {"R1B_H_POLICY": V.VARIANT}):
                importlib.reload(registry)
                dest = Path(V.seed(self.source, "B"))
                har = next(s for s in registry.STEPS["r1b_K_S1"] if s.name == "k2_har")
                self.assertIn("hn/htq_har_ready.json", har.out)
                self.assertTrue((dest / "r1b_K_S1/k/k2/h_ar.pt").is_file())
                self.assertFalse((dest / "r1b_K_S1/hn/htq_har_ready.json").exists())
                from r1b_run_session import step_done
                self.assertFalse(step_done(SimpleNamespace(work=str(dest / "r1b_K_S1"), mock=False,
                                                         session="r1b_K_S1", root=str(self.source / "fresh-root")), har))
        finally:
            importlib.reload(registry)

    def test_rerun_keeps_in_progress_variant_files(self):
        self.prefix()
        dest = Path(V.seed(self.source, "B"))
        progress = write(dest, "r1b_K_S4/k/k8b/part0.pt", "new H_tq result")
        changed = write(dest, "r1b_K_S1/k4j.json", "legitimately replaced")
        marker_hash = digest(dest / V.MARKER)
        self.assertEqual(V.seed(self.source, "B"), str(dest))
        self.assertEqual(progress.read_text(), "new H_tq result")
        self.assertEqual(changed.read_text(), "legitimately replaced")
        self.assertEqual(marker_hash, digest(dest / V.MARKER))

    def test_tampered_prefix_never_publishes_completed_destination(self):
        root = self.prefix()
        write(root, "k4j.json", "tampered")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            V.seed(self.source, "B")
        self.assertFalse((self.source / V.SUBDIR).exists())
        self.assertFalse(list((self.source / "variants").glob(".htq_v2.seed*")))

    def test_source_marker_change_during_copy_is_rejected(self):
        root = self.prefix()
        original_copy = V.shutil.copy2
        changed = False

        def copy_then_change(src, dst):
            nonlocal changed
            result = original_copy(src, dst)
            if not changed:
                changed = True
                p = root / "steps/k1_check.json"
                rec = json.loads(p.read_text())
                rec["note"] = "source is still running"
                p.write_text(json.dumps(rec))
            return result

        with patch.object(V.shutil, "copy2", side_effect=copy_then_change):
            with self.assertRaisesRegex(ValueError, "SHA256|changed"):
                V.seed(self.source, "B")
        self.assertFalse((self.source / V.SUBDIR / V.MARKER).exists())

    def test_copied_prefix_output_change_before_publish_is_rejected(self):
        root = self.prefix()
        original_copy = V.shutil.copy2
        changed = False

        def copy_then_change(src, dst):
            nonlocal changed
            result = original_copy(src, dst)
            # Change an earlier output after its copy and its immediate checks.
            if not changed and str(src).endswith("steps/k2_har.json"):
                changed = True
                write(root, "k4j.json", "modified after copy")
            return result

        with patch.object(V.shutil, "copy2", side_effect=copy_then_change):
            with self.assertRaisesRegex(ValueError, "source changed"):
                V.seed(self.source, "B")
        self.assertFalse((self.source / V.SUBDIR).exists())

    def test_copy_failure_does_not_publish_seed(self):
        self.prefix()
        with patch.object(V.shutil, "copy2", side_effect=OSError("Drive disconnected")):
            with self.assertRaisesRegex(OSError, "Drive disconnected"):
                V.seed(self.source, "B")
        self.assertFalse((self.source / V.SUBDIR / V.MARKER).exists())
        self.assertFalse(list((self.source / "variants").glob(".htq_v2.seed*")))

    def test_failed_or_missing_prefix_and_missing_roots_fail(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            V.seed(self.source / "absent", "B")
        with self.assertRaisesRegex(ValueError, "prefix"):
            V.seed(self.source, "B")
        with self.assertRaisesRegex(ValueError, "r1b_R_S0"):
            V.seed(self.source, "A")
        root = self.prefix()
        write(root, "steps/k1_check.json", json.dumps({"name": "k1_check", "status": "failed", "outputs": {}}))
        with self.assertRaisesRegex(ValueError, "not successful"):
            V.seed(self.source, "B")
        self.assertFalse((self.source / V.SUBDIR).exists())

    def test_conflicting_destination_and_concurrent_lock_fail(self):
        self.prefix()
        dest = self.source / V.SUBDIR
        dest.mkdir(parents=True)
        write(dest, "existing.txt", "preserve")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            V.seed(self.source, "B")
        self.assertEqual((dest / "existing.txt").read_text(), "preserve")
        lock = dest.parent / ".htq_v2.seed.lock"
        lock.mkdir()
        with self.assertRaisesRegex(ValueError, "already running"):
            V.seed(self.source, "B")

    def test_path_traversal_symlinks_and_variant_source_fail(self):
        self.prefix()
        with self.assertRaisesRegex(ValueError, "traversal"):
            V.seed(str(self.source / ".." / "dsse_runs"), "B")
        target = Path(self.temp.name) / "outside"
        target.mkdir()
        (self.source / "r1b_inputs").symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            V.seed(self.source, "B")
        (self.source / "r1b_inputs").unlink()
        dest = Path(V.seed(self.source, "B"))
        with self.assertRaisesRegex(ValueError, "original source"):
            V.seed(dest, "B")

    def test_recorded_output_traversal_fails(self):
        root = self.prefix()
        p = root / "steps/k1_check.json"
        rec = json.loads(p.read_text())
        rec["outputs"]["../escape"] = {"sha256": "a" * 64}
        p.write_text(json.dumps(rec))
        with self.assertRaisesRegex(ValueError, "invalid recorded"):
            V.seed(self.source, "B")
        self.assertFalse((self.source / V.SUBDIR).exists())

    def test_check_rejects_wrong_account_and_bad_identity(self):
        self.prefix()
        dest = Path(V.seed(self.source, "B"))
        with self.assertRaisesRegex(ValueError, "identity"):
            V.check(dest, "A")
        p = dest / V.MARKER
        rec = json.loads(p.read_text())
        rec["h_policy"] = "H+"
        p.write_text(json.dumps(rec))
        with self.assertRaisesRegex(ValueError, "identity"):
            V.check(dest, "B")
        p.write_text("[]")
        with self.assertRaisesRegex(ValueError, "identity"):
            V.check(dest, "B")

    def test_b_drive_mount_aliases_are_accepted_in_both_directions(self):
        self.prefix()
        dest = Path(V.seed(self.source, "B"))
        rec = json.loads((dest / V.MARKER).read_text())
        alias_root = self.source.parent / "dsse_runs_B"
        alias_dest = alias_root / V.SUBDIR
        # These two physical test folders emulate the two virtual Drive mounts.
        write(alias_dest, V.MARKER, json.dumps(rec))
        self.assertEqual(V.check(alias_dest, "B")["source"], str(self.source))
        rec["source"] = str(alias_root)
        rec["destination"] = str(alias_dest)
        write(dest, V.MARKER, json.dumps(rec))
        self.assertEqual(V.check(dest, "B")["source"], str(alias_root))

    def test_alias_exception_does_not_allow_other_roots_or_account_a(self):
        self.prefix()
        dest = Path(V.seed(self.source, "B"))
        raw = (dest / V.MARKER).read_text()
        for root in (self.source.parent / "unrelated", self.source.parent / "elsewhere/dsse_runs_B"):
            alias = root / V.SUBDIR
            write(alias, V.MARKER, raw)
            with self.assertRaisesRegex(ValueError, "identity"):
                V.check(alias, "B")
        rec = json.loads(raw)
        rec["account"] = "A"
        alias = self.source.parent / "dsse_runs_B" / V.SUBDIR
        write(alias, V.MARKER, json.dumps(rec))
        with self.assertRaisesRegex(ValueError, "identity"):
            V.check(alias, "A")
        # Even the permitted B alias must retain consistent source/destination.
        rec["account"] = "B"
        rec["destination"] = str(dest.parent / "other")
        write(alias, V.MARKER, json.dumps(rec))
        with self.assertRaisesRegex(ValueError, "identity"):
            V.check(alias, "B")


class ActiveTests(unittest.TestCase):
    def test_explicit_manifest_policy_precedes_environment(self):
        with patch.dict(os.environ, {"R1B_H_POLICY": V.VARIANT}):
            self.assertTrue(V.active())
            self.assertTrue(V.active(SimpleNamespace(manifest={})))
            self.assertFalse(V.active(SimpleNamespace(manifest={"h_policy": "frozen"})))
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(V.active())
            self.assertTrue(V.active(SimpleNamespace(manifest={"h_policy": V.VARIANT})))


if __name__ == "__main__":
    unittest.main()
