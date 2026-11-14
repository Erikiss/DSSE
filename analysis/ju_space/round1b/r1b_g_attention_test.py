"""Offline grouped-query SDPA regression checks; optional tiny CUDA preflight."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import r1b_g_adapter as A
import r1b_g_attention as G
from r1b_g_mock import make_mock, make_assistant


class GemmaAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_real_head_width_grouping_preserves_outputs_and_gradients(self):
        generator = torch.Generator().manual_seed(20261114)
        for kv_heads, nq, nk, mask_kind in ((1, 17, 17, None), (2, 5, 17, "bool"),
                                            (1, 1, 17, "additive")):
            with self.subTest(kv_heads=kv_heads, query=nq, mask=mask_kind):
                raw = [torch.randn((1, h, n, 512), generator=generator) / 8
                       for h, n in ((8, nq), (kv_heads, nk), (kv_heads, nk))]
                q, k, v = [x.clone().requires_grad_() for x in raw]
                mask = None
                if mask_kind:
                    mask = (torch.arange(nk)[None] <= (torch.arange(nq) + nk - nq)[:, None])[None, None]
                    if mask_kind == "additive":
                        mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, -torch.inf)
                ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0,
                                                     is_causal=mask is None and nq > 1, enable_gqa=True)
                ref_grad = torch.autograd.grad(ref.square().sum(), (q, k, v))
                q, k, v = [x.clone().requires_grad_() for x in raw]
                out, weights = G.gemma_sdpa(SimpleNamespace(is_causal=True), q, k, v, mask, scaling=1.0)
                got_grad = torch.autograd.grad(out.square().sum(), (q, k, v))
                self.assertIsNone(weights)
                self.assertEqual(out.dtype, torch.float32)
                torch.testing.assert_close(out.transpose(1, 2), ref, rtol=1e-6, atol=1e-7)
                for got, expected in zip(got_grad, ref_grad):
                    torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-6)
                for got, expected in zip((q, k, v), raw):
                    torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_kernel_boundary_has_equal_heads_without_gqa_flag_or_dtype_change(self):
        actual_sdpa = F.scaled_dot_product_attention
        for dtype in (torch.float32, torch.float16):
            q, k, v = (torch.zeros((1, h, 17, 512), dtype=dtype) for h in (8, 1, 1))
            with patch.object(F, "scaled_dot_product_attention", wraps=actual_sdpa) as call:
                G.gemma_sdpa(SimpleNamespace(is_causal=True), q, k, v, None, scaling=1.0)
            args, kwargs = call.call_args
            self.assertEqual([x.shape[1] for x in args[:3]], [8, 8, 8])
            self.assertEqual([x.dtype for x in args[:3]], [dtype] * 3)
            self.assertNotIn("enable_gqa", kwargs)
            self.assertTrue(kwargs["is_causal"])
            self.assertEqual(kwargs["scale"], 1.0)

    def test_registered_backend_preserves_native_models_and_input_gradients(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        native = ALL_ATTENTION_FUNCTIONS["sdpa"]
        for kind in ("e2b", "e4b", "12b", "31b"):
            with self.subTest(kind=kind):
                _, reference = make_mock(kind)
                reference.set_attn_implementation("sdpa")
                actual = copy.deepcopy(reference)
                G.configure_attention(actual)
                self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], native)
                self.assertEqual(A.text_config(actual)._attn_implementation, G.BACKEND)
                ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
                outputs, gradients = [], []
                for model in (reference, actual):
                    embedding = model.get_input_embeddings()
                    leaf = embedding(ids).detach().requires_grad_()
                    handle = embedding.register_forward_hook(lambda module, args, value: leaf)
                    try:
                        result = model(input_ids=ids, use_cache=False, logits_to_keep=0).logits
                        gradients.append(torch.autograd.grad(result.square().sum(), leaf)[0])
                        outputs.append(result.detach())
                    finally:
                        handle.remove()
                torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(gradients[0], gradients[1], rtol=3e-5, atol=2e-6)

    def test_native_cache_and_assistant_keep_shared_kv_layout(self):
        from transformers import DynamicCache
        _, reference = make_mock("e2b")
        reference.set_attn_implementation("sdpa")
        actual = copy.deepcopy(reference)
        G.configure_attention(actual)
        assistants = [make_assistant(model) for model in (reference, actual)]
        assistants[1].load_state_dict(assistants[0].state_dict())
        assistants[0].set_attn_implementation("sdpa")
        G.configure_attention(assistants[1])
        ids = torch.tensor([[10 + i % 20 for i in range(27)]])
        with torch.no_grad():
            runs = []
            for model, assistant in zip((reference, actual), assistants):
                cache, pieces = DynamicCache(config=model.config), []
                for begin in range(0, ids.shape[1], 5):
                    end = min(begin + 5, ids.shape[1])
                    output = model(input_ids=ids[:, begin:end], past_key_values=cache, use_cache=True,
                                   attention_mask=torch.ones((1, end), dtype=torch.long),
                                   position_ids=torch.arange(begin, end)[None],
                                   output_hidden_states=True, return_shared_kv_states=True, logits_to_keep=0)
                    cache = output.past_key_values
                    pieces.append(output.logits)
                seed = model.get_input_embeddings()(ids[:, -1:])
                h = output.hidden_states[-1][:, -1:]
                saved = {name: tuple(t.clone() for t in pair) for name, pair in output.shared_kv_states.items()}
                draft = assistant(inputs_embeds=torch.cat([seed, h], -1),
                                  shared_kv_states=output.shared_kv_states,
                                  position_ids=torch.tensor([[ids.shape[1]]]),
                                  attention_mask=torch.ones_like(ids), use_cache=False)
                for name, pair in saved.items():
                    for expected, got in zip(pair, output.shared_kv_states[name]):
                        torch.testing.assert_close(got, expected, rtol=0, atol=0)
                runs.append((torch.cat(pieces, 1), draft.logits, saved))
            torch.testing.assert_close(runs[0][0], runs[1][0], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(runs[0][1], runs[1][1], rtol=1e-5, atol=1e-6)
            for name in runs[0][2]:
                for expected, got in zip(runs[0][2][name], runs[1][2][name]):
                    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)

    def test_native_local_global_shape_selection(self):
        config = {"layer_types": ["sliding_attention", "full_attention"], "num_attention_heads": 8,
                  "num_key_value_heads": 2, "num_global_key_value_heads": 1,
                  "head_dim": 256, "global_head_dim": 512, "attention_k_eq_v": True}
        self.assertEqual(G.attention_shapes(config), [(8, 1, 512), (8, 2, 256)])
        config["attention_k_eq_v"] = False
        self.assertEqual(G.attention_shapes(config), [(8, 2, 256), (8, 2, 512)])

    def test_fp16_cache_gate_agrees_with_native_sdpa_without_relaxing_it(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        import r1b_g_k as K
        import r1b_g_k_persistence as P
        from r1b_g_k_test import MiniCtx

        reports = []
        for backend in ("sdpa", G.BACKEND):
            with tempfile.TemporaryDirectory() as td:
                tok, model = A.load_model("gemma4-12b-qat", mock=True)
                G.configure_attention(model)
                model.set_attn_implementation(backend)
                engine = K._engine(MiniCtx(td, "gemma4-12b-qat", {}, (tok, model)))
                seq = engine.encode("The river flows through town. The trees grow.", chat=False)
                args = (engine, seq, [[0, 2], [2, 1]], 3.0, [5, 6, 7])
                reports.append(P.cache_equivalence(*args))
                with sdpa_kernel(SDPBackend.MATH):
                    self.assertTrue(P.cache_equivalence(*args)["pass_"])
        # CPU SDPA may have fp16 full/chunk rounding large enough to close the
        # gate. The wrapper must preserve the native result, including failure.
        self.assertEqual(reports[0]["pass_"], reports[1]["pass_"])
        for key in ("relative_max_logits", "hidden_max_abs", "gumbel_flip_rate"):
            self.assertAlmostEqual(reports[0][key], reports[1][key], delta=1e-7)

    def test_cuda_preflight_precedes_any_weight_snapshot_or_model_load(self):
        from r1b_g_models import get_spec
        _, model = make_mock("e2b")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "models" / "gemma4-e2b"
            path.mkdir(parents=True)
            raw = {"model_type": "gemma4", "text_config": model.config.to_dict()}
            data = json.dumps(raw).encode()
            (path / "config.json").write_bytes(data)
            spec = get_spec("gemma4-e2b")
            spec["sha256"]["config.json"] = hashlib.sha256(data).hexdigest()
            with patch.object(A, "_spec", return_value=spec), \
                    patch.object(torch.cuda, "is_available", return_value=True), \
                    patch.object(G, "preflight_cuda_attention", side_effect=RuntimeError("kernel unavailable")) as preflight, \
                    patch.object(A, "ensure_snapshot") as snapshot:
                with self.assertRaisesRegex(RuntimeError, "kernel unavailable"):
                    A._load_real("gemma4-e2b", td, "cuda")
            preflight.assert_called_once_with(json.loads(data)["text_config"], torch.float32, "cuda")
            snapshot.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the real efficient kernel")
    def test_cuda_fp32_width512_efficient_preflight(self):
        config = {"layer_types": ["full_attention"], "num_attention_heads": 8,
                  "num_key_value_heads": 1, "head_dim": 512}
        report = G.preflight_cuda_attention(config, torch.float32, "cuda")
        self.assertEqual(report["cuda_kernel"], "EFFICIENT_ATTENTION")
        self.assertEqual(len(report["cases"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
