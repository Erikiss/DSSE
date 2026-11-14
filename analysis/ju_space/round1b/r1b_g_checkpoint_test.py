"""Repeatable offline checkpoint-conversion tests for the actual G loader.

Every checkpoint is a tiny random model created in TemporaryDirectory. Only the
CUDA availability check and requested placement are patched; from_pretrained,
manifest verification, key mapping and CT decompression are the production path.
Run with MOCK=1 python r1b_g_checkpoint_test.py. No real weights are downloaded.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("MOCK") != "1":
            raise RuntimeError("checkpoint tests require MOCK=1")
        import torch
        torch.set_num_threads(1)

    def _roundtrip(self, kind, quantized):
        import torch
        import r1b_g_adapter as A
        from r1b_g_mock import make_mock
        from r1b_g_models import get_spec
        from transformers import (Gemma4Config, Gemma4ForConditionalGeneration, Gemma4ForCausalLM,
                                  Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration,
                                  Gemma4UnifiedForCausalLM)

        tok, tiny = make_mock(kind)
        if quantized:
            # Real CT files use groups of 32; keep every tiny Linear input width
            # divisible by 32, including the E2B per-layer input projection.
            tiny_cfg = copy.deepcopy(tiny.config)
            tiny_cfg.intermediate_size = 64
            if hasattr(tiny_cfg, "hidden_size_per_layer_input"):
                tiny_cfg.hidden_size_per_layer_input = 32
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1701)
                tiny = tiny.__class__(tiny_cfg).eval()
        if kind == "12b":
            Cfg, Full, Text = Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration, Gemma4UnifiedForCausalLM
        else:
            Cfg, Full, Text = Gemma4Config, Gemma4ForConditionalGeneration, Gemma4ForCausalLM
        full = Full(Cfg(text_config=tiny.config.to_dict())).eval()
        full.model.language_model.load_state_dict(tiny.model.state_dict())
        full.set_attn_implementation("sdpa")
        if quantized:
            from compressed_tensors import ModelCompressor
            from compressed_tensors.quantization import QuantizationConfig, apply_quantization_config
            q = {"config_groups": {"group_0": {"targets": ["Linear"], "weights": {
                "num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": 32,
                "zp_dtype": None, "scale_dtype": None}}},
                "ignore": ["lm_head"], "quantization_status": "frozen", "format": "pack-quantized"}
            qc = QuantizationConfig.model_validate(q)
            apply_quantization_config(full, qc)
            for module in full.modules():
                if hasattr(module, "weight_scale"):
                    module.weight_scale.data.fill_(0.01)
                    if hasattr(module, "weight_zero_point"):
                        module.weight_zero_point.data.zero_()
            compressor = ModelCompressor(qc)
            compressor.compress_model(full)
            q = qc.model_dump(mode="json")
            q["quant_method"] = "compressed-tensors"
            full.config.quantization_config = q
        alias = "gemma4-" + kind + ("-qat" if quantized else "")
        # 12B has only a QAT target in G; its uncompressed conversion still uses
        # that identity's shape with an explicitly synthetic test specification.
        spec = get_spec(alias if kind != "12b" or quantized else "gemma4-12b-qat")
        spec.update(revision="1" * 40, layers=4, hidden=32,
                    widths=list(A.NeuronLayout.from_model(tiny).widths), quantized=quantized,
                    forward_only=quantized, dtype="float16" if quantized else "float32")
        with tempfile.TemporaryDirectory(prefix="r1b-g-checkpoint-") as td:
            checkpoint = Path(td) / "models" / alias
            full.save_pretrained(checkpoint)
            tok.save_pretrained(checkpoint)
            spec["sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in checkpoint.iterdir() if p.is_file()}
            from_pretrained = Text.from_pretrained
            seen = []

            def on_cpu(*args, **kwargs):
                self.assertEqual(kwargs["device_map"], {"": "cuda"})
                self.assertTrue(kwargs["local_files_only"])
                self.assertFalse(kwargs["trust_remote_code"])
                if quantized:
                    self.assertFalse(kwargs["quantization_config"].run_compressed)
                seen.append(True)
                kwargs["device_map"] = {"": "cpu"}
                return from_pretrained(*args, **kwargs)

            with patch.object(A, "_spec", return_value=copy.deepcopy(spec)), \
                    patch.object(torch.cuda, "is_available", return_value=True), \
                    patch.object(Text, "from_pretrained", side_effect=on_cpu):
                _, actual = A.load_model(alias, root=td, device="cuda", mock=False)
            self.assertEqual(seen, [True])
            self.assertEqual(A.NeuronLayout.from_model(actual).widths, A.NeuronLayout.from_model(tiny).widths)
            self.assertEqual(next(actual.parameters()).dtype, torch.float16 if quantized else torch.float32)
            ids = torch.tensor([[10, 11, 12, 13]])
            with torch.no_grad():
                if quantized:
                    # Decompress the independently retained reference, then cast
                    # to execution precision. Never compare with original weights.
                    compressor.decompress_model(full)
                    for parameter in full.parameters():
                        if parameter.is_floating_point():
                            parameter.data = parameter.data.to(torch.float16)
                # The execution backend and RoPE buffers match. Parameter equality
                # checks the conversion independently of the output comparison.
                source = dict(full.named_parameters())
                for key, value in actual.named_parameters():
                    if key.endswith(("weight_scale", "weight_zero_point")):
                        continue
                    mapped = key.replace("model.", "model.language_model.", 1) if key.startswith("model.") else key
                    torch.testing.assert_close(value, source[mapped], rtol=0, atol=0)
                expected = full(ids, logits_to_keep=0).logits
                got = actual(ids, logits_to_keep=0).logits
                self.assertTrue(torch.isfinite(got).all())
                torch.testing.assert_close(got, expected, rtol=3e-3 if quantized else 1e-5,
                                           atol=8e-4 if quantized else 3e-7)
                import r1b_core as core
                with core.Hooks(actual).neuron(0, [0], "read", name="z") as hk:
                    actual(ids, logits_to_keep=0)
                    self.assertEqual(hk.reads["z"][0]["x"].shape, (1, 4, 1))
            if quantized:
                with self.assertRaises(RuntimeError):
                    actual.model(ids)

    def test_gemma4_multimodal_to_text(self):
        self._roundtrip("e2b", False)

    def test_unified_multimodal_to_text(self):
        self._roundtrip("12b", False)

    def test_gemma4_packed_checkpoint_to_decompressed_text(self):
        self._roundtrip("e2b", True)

    def test_unified_packed_checkpoint_to_decompressed_text(self):
        self._roundtrip("12b", True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
