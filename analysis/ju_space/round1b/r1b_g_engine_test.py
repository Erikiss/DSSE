"""Offline numerical tests for native random Gemma G instruments.

These exercise equations and interventions rather than scientific outcomes.
Run: OMP_NUM_THREADS=1 python r1b_g_engine_test.py
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

import r1b_g_adapter as A
from r1b_g_engine import ModelEngine
from r1b_g_lens import (jacobian_blocks, _residual_sum, rollout, frame_spec,
                        native_rollout, calibration_stats)


class Context:
    mock = True
    model_name = "gemma4-e2b"
    device = "cpu"
    freeze = {}

    def __init__(self, name="gemma4-e2b"):
        self.model_name = name
        self._models = {}

    def model(self, name=None):
        name = name or self.model_name
        if name not in self._models:
            self._models[name] = A.load_model(name, mock=True)
        return self._models[name]

    def key(self, *parts):
        import r1b_core as core
        return core.keyseed("engine-test", *parts)


class NativeEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ctx = Context()
        cls.e = ModelEngine(cls.ctx)
        cls.seq = cls.e.encode("The river passes a quiet village beside the trees. " * 4, chat=False)[:32]

    def test_ragged_cett_matches_direct_down_projection(self):
        E = self.e
        flat = [0, E.layout.offsets[2] + 3, E.layout.total - 1]
        positions = [3, 8, 17, 29]
        row = E.teacher_force([self.seq], populations={"P": flat}, positions=positions)["rows"][0]
        direct, handles = {}, []
        for f in flat:
            layer, neuron = E.layout.decode(f)
            def capture(module, args, output, f=f, neuron=neuron):
                z = args[0][0, positions, neuron].float()
                c = z.abs() * module.weight[:, neuron].float().norm() / output[0, positions].float().norm(dim=-1).clamp(min=1e-9)
                direct[f] = (z, c)
            handles.append(E.layers[layer].mlp.down_proj.register_forward_hook(capture))
        try:
            with torch.no_grad():
                E.model(torch.tensor([self.seq]))
        finally:
            for h in handles:
                h.remove()
        for i, f in enumerate(sorted(flat)):
            torch.testing.assert_close(row["z"][:, i], direct[f][0])
            torch.testing.assert_close(row["cett"][:, i], direct[f][1])
        mean = E.features([self.seq], [positions], {"P": flat})
        torch.testing.assert_close(mean["cett"][0], row["cett"].mean(0))
        self.assertEqual(E.layout.widths, (48, 48, 96, 96))

    def test_teacher_scores_native_softcap_and_chunking(self):
        E = self.e
        with torch.no_grad():
            out = E.model(torch.tensor([self.seq]), use_cache=False).logits[0, :-1].float()
        expected = out.log_softmax(-1).gather(-1, torch.tensor(self.seq[1:])[:, None])[:, 0]
        E.logit_chunk = 7
        try:
            actual = E.teacher_force([self.seq])["rows"][0]["next_logprobs"]
        finally:
            E.logit_chunk = 32
        torch.testing.assert_close(actual, expected)

    def test_interventions_are_positioned_and_cleaned_up(self):
        E = self.e
        f = E.layout.offsets[1] + 2
        base = E.teacher_force([self.seq], {"P": [f]})["rows"][0]
        edit = E.teacher_force([self.seq], {"P": [f]}, setup=lambda: E.intervene([f], "zero", positions=[7]))["rows"][0]
        self.assertEqual(float(edit["z"][7, 0]), 0.0)
        torch.testing.assert_close(edit["z"][:7], base["z"][:7])
        after = E.teacher_force([self.seq], {"P": [f]})["rows"][0]
        torch.testing.assert_close(after["z"], base["z"], rtol=0, atol=0)
        before = len(E.layers[1]._forward_hooks)
        with self.assertRaises(RuntimeError):
            with E.patch_state(1, torch.zeros(32), 3):
                raise RuntimeError("probe")
        self.assertEqual(len(E.layers[1]._forward_hooks), before)

    def test_keyed_generation_and_truncation(self):
        E = self.e
        kwargs = dict(keys=[381], max_new=8, temperature=.7, topk=50, topp=.9)
        a = E.decode([self.seq[:8]], **kwargs)
        torch.manual_seed(932); torch.randn(100)
        b = E.decode([self.seq[:8]], **kwargs)
        self.assertEqual(a, b)
        logits = torch.tensor([0., 3., 1., 2.])
        for i in range(20):
            self.assertEqual(E.choose(logits, torch.Generator().manual_seed(i), 3., topk=1), 1)

    def test_forward_jacobian_matches_finite_difference(self):
        E = self.e
        blocks = list(jacobian_blocks(E, self.seq, row_block=8))
        maps = torch.cat([x for _, x in blocks], dim=1)
        torch.testing.assert_close(maps[-1], torch.eye(32), atol=1e-6, rtol=1e-6)
        v = torch.randn(32, generator=torch.Generator().manual_seed(9)); v /= v.norm()
        eps = .003
        for layer in (0, 2):
            direct = (_residual_sum(E, self.seq, layer, eps*v) - _residual_sum(E, self.seq, layer, -eps*v)) / (2*eps)
            torch.testing.assert_close(maps[layer] @ v, direct, atol=1e-3, rtol=2e-3)

    def test_fixed_placeholder_rollout_forced_replay(self):
        E = self.e
        state = E.hidden([self.seq], [0])[0][0][-1]
        frame = frame_spec(E)["frame_ids"]
        seed = E.encode("river", chat=False)[0]
        calls = []
        out = rollout(E, 0, state, [seed], steps=3, prior_state=state*.5, frame_ids=frame,
                      keep_distributions=True, keep_first=True, observer=lambda *x: calls.append(x))
        forced = rollout(E, 0, state, [seed], steps=3, prior_state=state*.5, frame_ids=frame, forced=out["tok"].tolist())
        torch.testing.assert_close(out["e"], forced["e"])
        self.assertEqual(out["alt"].dtype, torch.long)
        self.assertEqual(len(calls), 3)
        with E.patch_state(0, state, len(frame)-1):
            expected = E.next_logits(frame+[seed]).log_softmax(-1)
        torch.testing.assert_close(out["log_probs"][0,0], expected)
        torch.testing.assert_close(out["first"]["lr"][0], expected)

    def test_native_assistant_recursive_first_step_and_kl(self):
        E = self.e
        _, assistant = self.ctx.model("gemma4-e2b-assistant")
        bridge = A.AssistantBridge(E.model, assistant)
        initial, _ = bridge.prefill(torch.tensor([self.seq]))
        seed = E.encode("tree", chat=False)[0]
        result = native_rollout(E, bridge, self.seq, initial.hidden, initial.hidden*.5, [seed], steps=3, self_kl=True)
        logits, _ = bridge.step(torch.tensor([[seed]]), initial)
        lp = logits[0,0].float().log_softmax(-1)
        lq = E.next_logits(self.seq+[seed]).log_softmax(-1)
        self.assertAlmostEqual(float(result["kl"][0,0]), float((lq.exp()*(lq-lp)).sum()), places=6)
        token = result["tok"][0,0]
        self.assertAlmostEqual(float(result["lpr"][0,0]), float(lp[token]), places=6)
        self.assertTrue(torch.isfinite(result["kl"]).all())

    def test_native_fov_includes_local_masks_and_shared_kv(self):
        from r1b_g_k_extra import _FOV
        E = self.e
        captured, handles = [], []
        before = E.cfg._attn_implementation
        E.cfg._attn_implementation = "eager"
        try:
            for layer in E.layers[1:]:
                handles.append(layer.self_attn.register_forward_hook(
                    lambda m,a,o: captured.append(o[1][0,:,-1,2].detach().float().mean())))
            with _FOV(E,0,2) as fov:
                E.next_logits(self.seq)
            torch.testing.assert_close(fov.steps[0][0], torch.stack(captured).mean(), atol=1e-7, rtol=1e-6)
        finally:
            E.cfg._attn_implementation = before
            for h in handles:
                h.remove()

    def test_native_cached_placeholder_matches_full_replay(self):
        from r1b_g_k_extra import _cached_logp
        E = self.e
        frame = self.seq[:24]
        state = E.hidden([self.seq],[0])[0][0][-1]
        seed = self.seq[24]
        out = rollout(E,0,state,[seed],steps=4,prior_state=state*.5,frame_ids=frame)
        cached = _cached_logp(E,frame[:-1],frame[-1],state,0,seed,out["tok"][0].tolist())
        torch.testing.assert_close(cached,out["lpr"][0],atol=1e-5,rtol=1e-5)

    def test_qat_fp16_native_engine_scores(self):
        E = ModelEngine(Context("gemma4-12b-qat"))
        seq = E.encode("The river is quiet.", chat=False)
        row = E.teacher_force([seq], {"P": [E.layout.total-1]}, blocks=[3])["rows"][0]
        self.assertTrue(torch.isfinite(row["next_logprobs"]).all())
        with torch.no_grad():
            expected = E.model(torch.tensor([seq])).logits[0,-1].float()
        torch.testing.assert_close(E.next_logits(seq), expected, atol=.002, rtol=.002)
        with self.assertRaises(ValueError):
            list(jacobian_blocks(E, seq))


class CalibrationTests(unittest.TestCase):
    def test_full_comparison_branch_on_explicit_engineering_gate(self):
        from unittest.mock import patch
        from r1b_g_lens import _mtp_chains
        from r1b_g_data import mock_data
        ctx = Context()
        data = mock_data()
        ctx.data = lambda: data
        ctx.cached = lambda name, fn: fn()
        E = ModelEngine(ctx)
        _, assistant = ctx.model("gemma4-e2b-assistant")
        lens = {"maps": torch.eye(32).repeat(4, 1, 1)}
        def fixture_gate(context, engine, items):
            # Only this explicit unit-test fixture replaces correctness labels;
            # the production stage always uses native generated answers.
            return {it["id"]: {"stratum": "correct", "own": [], "W": []} for it in items}
        with patch("r1b_g_k_chain.gate_arithmetic", fixture_gate):
            rows, result = _mtp_chains(ctx, E, A.AssistantBridge(E.model, assistant), lens)
        self.assertTrue(rows["rows"])
        self.assertEqual({x["lens"] for x in rows["rows"]}, {"CL35", "CL35_b1", "E", "L"})
        self.assertFalse(set(rows["calibration_ids"]) & set(rows["analysis_ids"]))
        self.assertTrue(rows["selfkl"])
        self.assertEqual(result["G_K5"], "not decidable")
        self.assertIn("random-model engineering fixture, no scientific verdict", result["labels"])

    def test_disjoint_selection_is_not_reselected_on_analysis(self):
        cal, held = ["c0", "c1", "c2"], ["a0", "a1", "a2", "a3"]
        rows = []
        for item in cal+held:
            for instrument in ("CL35", "E", "L"):
                for variant in ("V1", "V2"):
                    # E is identical to native on calibration; heldout L is
                    # identical instead. Choice must remain E on heldout.
                    claims = [(1, "7", "TRUE-INT")] if variant == "V1" else []
                    if (instrument == "L" and item in cal) or (instrument == "E" and item in held):
                        claims.append((2, "99", "CONFAB"))
                    rows.append(dict(item=item, lens=instrument, variant=variant, claims=claims, censor=None))
        result = calibration_stats(rows, cal, held, B=30, steps=4)
        self.assertEqual(result["variant"], "E")
        self.assertEqual(result["G_K5"], "self-rollout differs from the native chain lens")
        self.assertEqual(result["dC16"]["est"], 1.)
        with self.assertRaises(ValueError):
            calibration_stats(rows, cal, cal, B=10)


if __name__ == "__main__":
    unittest.main()
