"""CPU-only tests of hidden-state audit persistence and opt-in isolation.

The model has random tiny weights; no pretrained model or dataset is downloaded.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn

import r1b_core as K
import r1b_hidden_audit as A
from r1b_hidden_capture import COUNTS, FLAG


class TinyBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, width)

    def forward(self, hidden):
        return hidden + self.proj(hidden).tanh()


class TinyLM(nn.Module):
    """The decoder-block layout used by the existing model-free hook helpers."""
    def __init__(self, width=6, layers=3, vocab=37):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, width)
        self.model.layers = nn.ModuleList([TinyBlock(width) for _ in range(layers)])
        self.lm_head = nn.Linear(width, vocab, bias=False)
        self.config = SimpleNamespace(hidden_size=width, num_hidden_layers=layers, vocab_size=vocab)

    def forward(self, input_ids, attention_mask=None, position_ids=None, logits_to_keep=None, **kwargs):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        logits = self.lm_head(hidden)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)


class TinyTokenizer:
    pad_token_id = 0

    @staticmethod
    def decode(ids):
        return " ".join(str(int(i)) for i in ids)


class RecordingCtx(K.Ctx):
    def __init__(self, work, policy="htq_v2", session="r1b_K_S1"):
        super().__init__(session=session, work=work, root=str(Path(work) / "root"), device="cpu",
                         mock=False, manifest={"h_policy": policy, "code_commit": "tiny-model-test"}, step="k3a")
        self.pt_writes = []
        self.json_writes = []
        self.messages = []

    def save_pt(self, rel, obj, **kwargs):
        path = super().save_pt(rel, obj, **kwargs)
        self.pt_writes.append((rel, obj))
        return path

    def write_json(self, rel, obj, **kwargs):
        path = super().write_json(rel, obj, **kwargs)
        self.json_writes.append((rel, obj))
        return path

    def log(self, message):
        self.messages.append(message)


class HiddenAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        torch.manual_seed(2026)
        self.model = TinyLM().eval()
        self.tok = TinyTokenizer()
        self.ctx = RecordingCtx(self.temp.name)
        # Extra unselected rows ensure selection is bounded at the fixed limit.
        self.ids = {family: [f"{family}-question-{i}" for i in range(n + 3)]
                    for family, n in COUNTS.items()}

    def start(self, ctx=None):
        with patch.dict(os.environ, {FLAG: "1"}):
            return A.start_audit(ctx or self.ctx, self.tok, self.model, self.ids)

    def manifest(self):
        return json.loads(Path(self.ctx.path(f"{A.DIRECTORY}/manifest.json")).read_text())

    def family_inputs(self, family):
        n = len(self.ids[family])
        prompts = [list(range(1, 3 + i % 3)) for i in range(n)]
        keys = [70000 + i for i in range(n)]
        conts = [[10 + i % 4] if i % 2 else [4, 5, 6] for i in range(n)]
        # The later positive examples must not displace the selected prefix.
        labels = ["wrong"] * COUNTS[family] + ["correct"] * (n - COUNTS[family])
        return prompts, keys, conts, labels

    def observe(self, capture, prompts):
        """Drive the real callback API on variable left-padded batch shapes."""
        reference_logits = []
        captured_logits = []
        with capture, torch.no_grad():
            for b0 in range(0, len(prompts), 17):
                rows = prompts[b0:b0 + 17]
                ids, mask, _ = K.pad_left(rows, self.tok.pad_token_id)
                pos = (mask.cumsum(-1) - 1).clamp(min=0)
                plain = self.model(input_ids=ids, attention_mask=mask, position_ids=pos).logits.clone()
                with capture.setup(b0, len(rows)):
                    observed = self.model(input_ids=ids, attention_mask=mask, position_ids=pos).logits.clone()
                    capture.on_fwd(b0, 0)
                    # A decode forward must not be substituted for the prefill.
                    self.model(input_ids=ids[:, -1:], attention_mask=mask)
                    capture.on_fwd(b0, 1)
                reference_logits.append(plain)
                captured_logits.append(observed)
        for plain, observed in zip(reference_logits, captured_logits):
            self.assertTrue(torch.equal(plain, observed))
        self.assertTrue(all(not layer._forward_hooks for layer in self.model.model.layers))

    def save(self, audit, family):
        prompts, keys, conts, labels = self.family_inputs(family)
        capture = audit.prepare(family, prompts, keys, max_new=3)
        self.observe(capture, prompts)
        audit.save_family(family, capture, prompts, keys, conts, labels, max_new=3)
        return prompts, keys, conts, labels

    def test_default_off_has_no_writes_or_hooks(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(A.start_audit(self.ctx, self.tok, self.model, self.ids))
        self.assertEqual(self.ctx.json_writes, [])
        self.assertEqual(self.ctx.pt_writes, [])
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        self.assertTrue(all(not layer._forward_hooks for layer in self.model.model.layers))

    def test_enabled_capture_rejects_frozen_policy_and_other_sessions(self):
        for policy, session in (("frozen", "r1b_K_S1"), ("htq_v2", "r1b_K_S7"), ("htq_v2", "r1b_L_A")):
            with self.subTest(policy=policy, session=session):
                ctx = RecordingCtx(self.temp.name, policy=policy, session=session)
                # An explicit frozen manifest must win over a stale process flag.
                with patch.dict(os.environ, {FLAG: "1", "R1B_H_POLICY": "htq_v2"}):
                    with self.assertRaisesRegex(ValueError, "htq_v2"):
                        A.start_audit(ctx, self.tok, self.model, self.ids)
                self.assertEqual(ctx.json_writes, [])
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_three_families_complete_with_exact_metadata_and_real_sidecars(self):
        initial_weights = {name: value.clone() for name, value in self.model.state_dict().items()}
        rng_before = torch.random.get_rng_state().clone()
        audit = self.start()
        initial = self.manifest()
        self.assertEqual(initial["status"], "running")
        self.assertEqual(initial["planned_count"], 256)
        self.assertEqual(initial["request_count"], 0)
        for family, limit in COUNTS.items():
            with self.subTest(family=family):
                prompts, keys, conts, labels = self.save(audit, family)
                rel, shard = self.ctx.pt_writes[-1]
                hidden = shard["hidden"]
                self.assertEqual(tuple(hidden.shape), (limit, 3, 6))
                self.assertEqual(hidden.dtype, torch.float32)
                self.assertEqual(hidden.device.type, "cpu")
                self.assertFalse(hidden.requires_grad)
                self.assertTrue(bool(torch.isfinite(hidden).all()))
                self.assertEqual(shard["layer_ids"], [0, 1, 2])
                self.assertEqual(shard["config"]["selection"], "fixed source order, before labels")
                self.assertEqual([r["question_id"] for r in shard["records"]], self.ids[family][:limit])
                self.assertEqual({r["behavior_label"] for r in shard["records"]}, {"wrong"})
                for i, row in enumerate(shard["records"]):
                    self.assertEqual(row["request_id"], f"{family}:{self.ids[family][i]}")
                    self.assertEqual(row["prompt_token_ids"], prompts[i])
                    self.assertEqual(row["sampling_key"], keys[i])
                    self.assertEqual(row["generated_token_ids"], conts[i])
                    self.assertEqual(row["answer_text"], self.tok.decode(conts[i]))
                    self.assertEqual(row["token_position"], len(prompts[i]) - 1)
                    self.assertEqual(row["phase"], "prefill")
                    self.assertEqual(row["forward_step"], 0)
                self.assertEqual(shard["records"][0]["stop_reason"], "length_limit")
                self.assertEqual(shard["records"][1]["stop_reason"], "stop_token")
                path = self.ctx.path(rel)
                persisted = torch.load(path, map_location="cpu", weights_only=False)
                self.assertTrue(torch.equal(persisted["hidden"], hidden))
                self.assertIsNone(K.check_output(path))
                sidecar = json.loads(Path(path + ".prov.json").read_text())
                self.assertIn("exploratory hidden-state observation", sidecar["provenance"]["labels"])
                self.assertEqual(self.manifest()["status"], "running")
        audit.finish()
        complete = self.manifest()
        self.assertEqual(complete["status"], "complete")
        self.assertEqual(complete["request_count"], 256)
        self.assertEqual(complete["tensor_bytes"], 256 * 3 * 6 * 4)
        self.assertEqual(set(complete["families"]), set(COUNTS))
        for family, saved in complete["families"].items():
            self.assertEqual(saved["sha256"], K.sha256_file(self.ctx.path(saved["file"])))
            self.assertEqual(saved["count"], COUNTS[family])
        benchmarks = [rel for rel, _ in self.ctx.json_writes if rel.endswith("benchmark.json")]
        self.assertEqual(len(benchmarks), 1)
        benchmark = json.loads(Path(self.ctx.path(complete["benchmark"])).read_text())
        self.assertTrue(benchmark["logits_exact"])
        self.assertEqual(benchmark["repetitions"], 3)
        self.assertIn("uncached prefill", benchmark["method"])
        self.assertIn("not an estimate of whole-run overhead", benchmark["note"])
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(initial_weights[name], value), name)

    def test_incomplete_or_tampered_artifacts_cannot_publish_completion(self):
        audit = self.start()
        self.save(audit, "s3")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            audit.finish()
        self.assertEqual(self.manifest()["status"], "running")
        self.save(audit, "adet")
        self.save(audit, "ne")
        damaged = Path(self.ctx.path(f"{A.DIRECTORY}/s3.pt"))
        with damaged.open("ab") as f:
            f.write(b"modified after capture")
        with self.assertRaisesRegex(ValueError, "integrity"):
            audit.finish()
        self.assertEqual(self.manifest()["status"], "running")

    def test_failed_benchmark_stops_before_any_family_write(self):
        audit = self.start()
        prompts, keys, _, _ = self.family_inputs("s3")
        with patch.object(A, "benchmark_prefill", return_value={"status": "failed", "logits_exact": False}):
            with self.assertRaisesRegex(RuntimeError, "changed prefill logits"):
                audit.prepare("s3", prompts, keys, max_new=3)
        self.assertEqual(self.ctx.pt_writes, [])
        self.assertEqual(self.manifest()["status"], "running")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            audit.finish()

    def test_family_save_failure_remains_incomplete_and_existing_complete_is_reset(self):
        self.ctx.write_json(f"{A.DIRECTORY}/manifest.json", {"status": "complete"})
        audit = self.start()
        self.assertEqual(self.manifest()["status"], "running")
        prompts, keys, conts, labels = self.family_inputs("s3")
        capture = audit.prepare("s3", prompts, keys, max_new=3)
        self.observe(capture, prompts)
        with patch.object(self.ctx, "save_pt", side_effect=OSError("Drive disconnected")):
            with self.assertRaisesRegex(OSError, "Drive disconnected"):
                audit.save_family("s3", capture, prompts, keys, conts, labels, max_new=3)
        self.assertEqual(self.manifest()["families"], {})
        self.assertEqual(self.manifest()["status"], "running")

    def test_changed_request_identity_is_rejected_before_family_persistence(self):
        audit = self.start()
        prompts, keys, conts, labels = self.family_inputs("s3")
        capture = audit.prepare("s3", prompts, keys, max_new=3)
        self.observe(capture, prompts)
        changed = list(keys)
        changed[0] += 1
        with self.assertRaisesRegex(ValueError, "identity changed"):
            audit.save_family("s3", capture, prompts, changed, conts, labels, max_new=3)
        with self.assertRaisesRegex(ValueError, "repeated capture family"):
            audit.prepare("s3", prompts, keys, max_new=3)
        self.assertEqual(self.ctx.pt_writes, [])
        self.assertEqual(self.manifest()["status"], "running")

    def test_selection_ids_are_checked_before_writing_any_manifest(self):
        for invalid in ([], ["duplicate"] * COUNTS["s3"]):
            with self.subTest(invalid_length=len(invalid)):
                ids = dict(self.ids, s3=invalid)
                with patch.dict(os.environ, {FLAG: "1"}):
                    with self.assertRaisesRegex(ValueError, "Missing or duplicate"):
                        A.start_audit(self.ctx, self.tok, self.model, ids)
        too_few = dict(self.ids, s3=self.ids["s3"][:127])
        with patch.dict(os.environ, {FLAG: "1"}):
            with self.assertRaisesRegex(ValueError, "requires 128"):
                A.start_audit(self.ctx, self.tok, self.model, too_few)
        self.assertEqual(self.ctx.json_writes, [])
        self.assertEqual(self.ctx.pt_writes, [])

    def test_training_mode_is_rejected_before_capture(self):
        self.model.train()
        with self.assertRaisesRegex(ValueError, "eval-mode"):
            A.benchmark_prefill(self.model, self.tok, [[1, 2], [1]])
        self.assertTrue(all(not layer._forward_hooks for layer in self.model.model.layers))


if __name__ == "__main__":
    unittest.main()
