"""CPU-only G adapter tests. Run: MOCK=1 python r1b_g_selftest.py.

All weights are random or hand-constructed. Nothing calls from_pretrained on a
remote repository. Passing these tests is not a real-model/GPU validation.
"""
from __future__ import annotations

import os
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["MOCK"] = "1"

import torch

import r1b_core as core
from r1b_g_mock import make_assistant, make_mock

torch.set_num_threads(1)


class NativeGemmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("socket.socket.connect", side_effect=AssertionError("mock accessed network")):
            cls.fixtures = {k: make_mock(k) for k in ("e2b", "e4b", "12b", "31b")}

    def ids(self, n=16):
        return torch.arange(20, 20 + n)[None]

    def test_native_architectures_and_ragged_widths(self):
        for k, (_, m) in self.fixtures.items():
            with self.subTest(kind=k):
                widths = [b.mlp.down_proj.in_features for b in core.layers_of(m)]
                self.assertEqual(widths, [48, 48, 96, 96] if k == "e2b" else [48] * 4)
                self.assertEqual(m.config.layer_types, ["sliding_attention", "full_attention"] * 2)
                with torch.no_grad():
                    out = m(self.ids(), logits_to_keep=0)
                self.assertEqual(tuple(out.logits.shape), (1, 16, len(self.fixtures[k][0])))

    def test_random_initialization_reproducible_without_rng_side_effect(self):
        before = torch.random.get_rng_state()
        _, m = make_mock("e2b")
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for p, q in zip(m.parameters(), self.fixtures["e2b"][1].parameters()):
            self.assertTrue(torch.equal(p, q))

    def test_hook_identity_and_intervention(self):
        for k, (_, m) in self.fixtures.items():
            with self.subTest(kind=k), torch.no_grad():
                clean = m(self.ids(), logits_to_keep=0).logits
                with core.Hooks(m).neuron(2, [0, 3], "scale", 1):
                    neutral = m(self.ids(), logits_to_keep=0).logits
                with core.Hooks(m).neuron(2, [0, 3], "zero"):
                    zero = m(self.ids(), logits_to_keep=0).logits
                with core.Hooks(m).neuron(2, [0, 3], "clamp", 0):
                    clamp = m(self.ids(), logits_to_keep=0).logits
                self.assertTrue(torch.equal(clean, neutral))
                self.assertTrue(torch.equal(zero, clamp))
                self.assertGreater((clean - zero).abs().max().item(), 0)
                self.assertTrue(torch.equal(clean, m(self.ids(), logits_to_keep=0).logits))

    def test_neuron_read_is_activated_down_projection_input(self):
        _, m = self.fixtures["e2b"]
        mlp = core.layers_of(m)[2].mlp
        captures = {}
        handles = [mlp.gate_proj.register_forward_hook(lambda mod, inp, out: captures.update(gate=out)),
                   mlp.up_proj.register_forward_hook(lambda mod, inp, out: captures.update(up=out))]
        try:
            with torch.no_grad(), core.Hooks(m).neuron(2, [0, 70, 95], "read", name="z") as h:
                m(self.ids(), logits_to_keep=0)
            expected = (mlp.act_fn(captures["gate"]) * captures["up"])[..., [0, 70, 95]]
            torch.testing.assert_close(h.reads["z"][0]["x"], expected, atol=0, rtol=0)
        finally:
            for h in handles:
                h.remove()

    def test_capture_matches_native_hidden_states_and_cleans_up(self):
        _, m = self.fixtures["e2b"]
        with torch.no_grad():
            o = m(self.ids(), output_hidden_states=True, logits_to_keep=0)
            with core.Hooks(m) as h:
                h.capture([0, 1, 2])
                m(self.ids(), logits_to_keep=0)
            for b in [0, 1, 2]:
                torch.testing.assert_close(h.states[b][0], o.hidden_states[b + 1], rtol=0, atol=0)
        self.assertEqual(h.h, [])
        with self.assertRaisesRegex(RuntimeError, "planted"):
            with core.Hooks(m).neuron(0, [0], "zero") as h:
                raise RuntimeError("planted")
        self.assertEqual(h.h, [])

    def test_cache_parity_across_sliding_window(self):
        for k, (_, m) in self.fixtures.items():
            with self.subTest(kind=k), torch.no_grad():
                ids = self.ids(22)
                full = m(ids, use_cache=False, logits_to_keep=0).logits
                cache = None
                pieces = []
                for a, b in [(0, 6), (6, 13), (13, 21), (21, 22)]:
                    o = m(ids[:, a:b], past_key_values=cache, use_cache=True,
                          attention_mask=torch.ones(1, b, dtype=torch.long),
                          position_ids=torch.arange(a, b)[None], logits_to_keep=0)
                    pieces.append(o.logits)
                    cache = o.past_key_values
                torch.testing.assert_close(torch.cat(pieces, dim=1), full, atol=2e-6, rtol=2e-5)

    def test_left_padding_preserves_nonpadding_logits(self):
        tok, m = self.fixtures["e2b"]
        ids = self.ids(12)
        batch, mask, pads = core.pad_left([ids[0].tolist(), ids[0, 4:].tolist()], tok.pad_token_id)
        with torch.no_grad():
            full = m(batch, attention_mask=mask, position_ids=(mask.cumsum(-1)-1).clamp(min=0),
                     logits_to_keep=0).logits
            alone = m(ids[:, 4:], logits_to_keep=0).logits
        torch.testing.assert_close(full[1:, pads[1]:], alone, atol=2e-6, rtol=2e-5)

    def test_assistant_uses_shared_target_states(self):
        _, m = self.fixtures["e2b"]
        assistant = make_assistant(m)
        with torch.no_grad():
            out = m(self.ids(), return_shared_kv_states=True, output_hidden_states=True, logits_to_keep=0)
            inp = torch.cat([m.get_input_embeddings()(self.ids()[:, -1:]), out.hidden_states[-1][:, -1:]], -1)
            o = assistant(inputs_embeds=inp, shared_kv_states=out.shared_kv_states,
                          position_ids=torch.tensor([[15]]))
        self.assertEqual(tuple(o.logits.shape), (1, 1, m.config.vocab_size))
        self.assertEqual(tuple(o.last_hidden_state.shape), (1, 1, m.config.hidden_size))
        with self.assertRaises(ValueError):
            assistant(inputs_embeds=inp)


class QuantizedReferenceTests(unittest.TestCase):
    def test_compressed_tensors_roundtrip_and_hooks(self):
        from compressed_tensors import ModelCompressor
        from compressed_tensors.quantization import (QuantizationArgs, QuantizationConfig,
            QuantizationScheme, QuantizationStatus, apply_quantization_config)

        q = QuantizationConfig(config_groups={"group_0": QuantizationScheme(targets=["Linear"],
                weights=QuantizationArgs(num_bits=4, type="int", symmetric=True,
                                         strategy="group", group_size=16))},
                format="pack-quantized", quantization_status=QuantizationStatus.FROZEN)
        m = torch.nn.Sequential(torch.nn.Linear(32, 8, bias=False)).eval()
        apply_quantization_config(m, q)
        scale = torch.tensor([[0.125, 0.25]]).expand(8, -1).clone()
        ints = (torch.arange(256).reshape(8, 32) % 15 - 7).float()
        weights = ints * scale.repeat_interleave(16, dim=1)
        with torch.no_grad():
            m[0].weight.copy_(weights)
            m[0].weight_scale.copy_(scale)
            m[0].weight_zero_point.zero_()
        compressor = ModelCompressor(quantization_config=q)
        compressor.compress_model(m)
        self.assertIn("0.weight_packed", m.state_dict())
        self.assertNotIn("0.weight", m.state_dict())
        compressor.decompress_model(m)
        torch.testing.assert_close(m[0].weight, weights, atol=0, rtol=0)
        x = torch.arange(96).reshape(3, 32).float() / 32
        seen = []
        handle = m[0].register_forward_pre_hook(lambda mod, args: seen.append(args[0].clone()))
        try:
            result = m(x)
        finally:
            handle.remove()
        torch.testing.assert_close(result, torch.nn.functional.linear(x, weights), atol=0, rtol=0)
        self.assertEqual(len(seen), 1)


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import r1b_g_adapter as adapter
        cls.adapter = adapter

    def test_manifest_covers_all_eight_reviewed_models(self):
        from r1b_g_models import check, get_spec, MODEL_SPECS
        self.assertTrue(check()["ok"])
        self.assertEqual(len(MODEL_SPECS), 8)
        self.assertEqual(get_spec("gemma4-12b-qat")["family"], "gemma4_unified")
        s = get_spec("gemma4-e2b")
        s["sha256"].clear()
        self.assertTrue(get_spec("gemma4-e2b")["sha256"])

    def test_layout_roundtrip_and_bounds(self):
        _, model = self.adapter.load_model("gemma4-e2b", mock=True)
        layout = self.adapter.NeuronLayout.from_model(model)
        self.assertEqual(layout.widths, (48, 48, 96, 96))
        self.assertEqual(layout.offsets, (0, 48, 96, 192, 288))
        for i in range(layout.total):
            self.assertEqual(layout.encode(*layout.decode(i)), i)
        for layer, neuron in [(0, -1), (0, 48), (-1, 0), (4, 0)]:
            with self.assertRaises(IndexError):
                layout.encode(layer, neuron)
        for i in [-1, layout.total]:
            with self.assertRaises(IndexError):
                layout.decode(i)

    def test_gemma_chat_boundary_and_stops(self):
        tok, model = self.adapter.load_model("gemma4-e2b", mock=True)
        for user in ["Name a tree.", "Äpfel und Bäume.", "first\nsecond"]:
            rendered = self.adapter.chat_text(tok, user)
            ids = self.adapter.chat_ids(tok, user)
            self.assertEqual(ids.count(tok.bos_token_id), 1)
            start = self.adapter.user_start(tok, user)
            prefix = tok.decode(ids[:start], skip_special_tokens=False)
            self.assertEqual(prefix, rendered[:rendered.index(user)])
        eos, stop = self.adapter.stop_tokens(tok, model)
        self.assertIn(tok.convert_tokens_to_ids("<turn|>"), eos)
        self.assertNotIn(tok.unk_token_id, stop)
        self.assertIn(tok.pad_token_id, stop)

    def test_unembedding_matches_full_forward_with_softcap(self):
        for name in ["gemma4-e2b", "gemma4-12b-qat"]:
            with self.subTest(name=name):
                _, m = self.adapter.load_model(name, mock=True)
                if name.endswith("-qat"):
                    self.assertEqual(next(m.parameters()).dtype, torch.float16)
                with core.Hooks(m) as h, torch.no_grad():
                    h.capture([len(core.layers_of(m))-1], cpu=False)
                    out = m(torch.tensor([[20, 21, 22]]), logits_to_keep=0)
                    actual = self.adapter.unembed(m, h.states[len(core.layers_of(m))-1][0])
                torch.testing.assert_close(actual, out.logits, atol=0, rtol=0)

    def test_direct_loader_applies_round_numerical_policy(self):
        precision = torch.get_float32_matmul_precision()
        matmul = torch.backends.cuda.matmul.allow_tf32
        cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.set_float32_matmul_precision("medium")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.adapter.load_model("gemma4-e2b", mock=True)
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        finally:
            torch.set_float32_matmul_precision(precision)
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn

    def test_quantized_mock_refuses_gradient_measurements(self):
        _, m = self.adapter.load_model("gemma4-31b-qat", mock=True)
        with self.assertRaises(ValueError):
            self.adapter.require_gradients(m, "routing VJP")
        with self.assertRaises(RuntimeError):
            m(torch.tensor([[10, 11]]))
        with self.assertRaises(RuntimeError):
            m.model(torch.tensor([[10, 11]]))
        with torch.no_grad():
            self.assertEqual(m(torch.tensor([[10, 11]]), logits_to_keep=0).logits.shape[1], 2)

    def test_quantized_block_push_keeps_execution_dtype(self):
        _, m = self.adapter.load_model("gemma4-12b-qat", mock=True)
        ids = torch.tensor([[10, 11, 12, 13]])
        direction = torch.arange(m.config.hidden_size).float()
        with torch.no_grad():
            clean = m(ids, logits_to_keep=0).logits
            with core.Hooks(m).push(1, direction, 0):
                neutral = m(ids, logits_to_keep=0).logits
            with core.Hooks(m).push(1, direction, 0.2):
                changed = m(ids, logits_to_keep=0).logits
        self.assertEqual(changed.dtype, clean.dtype)
        self.assertTrue(torch.equal(clean, neutral))
        self.assertGreater((clean - changed).abs().max().item(), 0)

    def test_real_cpu_load_is_rejected_before_network(self):
        with patch("socket.socket.connect", side_effect=AssertionError("accessed network")):
            with self.assertRaisesRegex(RuntimeError, "CUDA-only"):
                self.adapter.load_model("gemma4-e2b", root="/not-a-model", device="cpu", mock=False)

    def test_freeze_cannot_change_precision_or_drop_required_pins(self):
        from r1b_g_models import get_spec
        with self.assertRaises(ValueError):
            self.adapter.load_model("gemma4-e2b", mock=True,
                                    freeze={"models": {"gemma4-e2b": {"dtype": "bfloat16"}}})
        hashes = get_spec("gemma4-e2b")["sha256"]
        del hashes["tokenizer.json"]
        with self.assertRaises(ValueError):
            self.adapter.load_model("gemma4-e2b", mock=True,
                                    freeze={"models": {"gemma4-e2b": {"sha256": hashes}}})

    def test_registration_preserves_existing_models_and_functions(self):
        models, loaders = dict(core.MODELS), dict(core.LOADERS)
        funcs = (core.load_model, core.chat_ids, core.stop_tokens, core.Hooks)
        try:
            names = self.adapter.register_models()
            self.assertEqual(len(names), 8)
            for n, value in models.items():
                self.assertEqual(core.MODELS[n], value)
            self.assertEqual(funcs, (core.load_model, core.chat_ids, core.stop_tokens, core.Hooks))
        finally:
            core.MODELS.clear()
            core.MODELS.update(models)
            core.LOADERS.clear()
            core.LOADERS.update(loaders)

    def test_assistant_pairing_and_native_recursion(self):
        _, target = self.adapter.load_model("gemma4-e2b", mock=True)
        _, assistant = self.adapter.load_model("gemma4-e2b-assistant", mock=True)
        bridge = self.adapter.AssistantBridge(target, assistant)
        ids = torch.tensor([[10, 11, 12, 13]])
        state, seed_logits = bridge.prefill(ids)
        seed = seed_logits[:, -1].argmax(-1)
        logits, state2 = bridge.step(seed, state)
        self.assertIs(state.shared_kv_states, state2.shared_kv_states)
        torch.testing.assert_close(state.position_ids, state2.position_ids)
        with torch.no_grad():
            expected = assistant(inputs_embeds=torch.cat([target.get_input_embeddings()(seed[:, None]), state.hidden], -1),
                                 shared_kv_states=state.shared_kv_states, position_ids=state.position_ids,
                                 attention_mask=state.attention_mask, use_cache=False)
        torch.testing.assert_close(logits, expected.logits, atol=0, rtol=0)
        torch.testing.assert_close(state2.hidden, expected.last_hidden_state, atol=0, rtol=0)
        _, wrong = self.adapter.load_model("gemma4-e4b-assistant", mock=True)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.adapter.AssistantBridge(target, wrong)

    def test_snapshot_rechecks_hashes_and_rejects_unpinned_shards(self):
        with tempfile.TemporaryDirectory() as td:
            config = Path(td) / "config.json"
            config.write_text("{}")
            pins = {"revision": "a" * 40, "sha256": {"config.json": hashlib.sha256(b"{}").hexdigest()}}
            self.assertTrue(self.adapter.validate_snapshot(td, pins, require_weights=False)["ok"])
            config.write_text('{"changed":true}')
            with self.assertRaises(RuntimeError):
                self.adapter.validate_snapshot(td, pins, require_weights=False)
            config.write_text("{}")
            index = Path(td) / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"model.x": "unpinned.safetensors"}}))
            pins["sha256"][index.name] = hashlib.sha256(index.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "unpinned weight shards"):
                self.adapter.validate_snapshot(td, pins, require_weights=False)

    def test_snapshot_rejects_path_traversal_and_mutable_revision(self):
        pins = {"revision": "main", "sha256": {"config.json": "b" * 64}}
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "immutable"):
                self.adapter.validate_snapshot(td, pins, require_weights=False)
            pins["revision"] = "a" * 40
            config = Path(td) / "config.json"
            config.write_text("{}")
            pins["sha256"] = {"config.json": hashlib.sha256(b"{}").hexdigest(), "../escape": "b" * 64}
            with self.assertRaisesRegex(ValueError, "unsafe manifest path"):
                self.adapter.validate_snapshot(td, pins, require_weights=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
