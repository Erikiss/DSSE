"""Native Gemma L dynamics parity and independent protocol tests (offline)."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import r1b_g_adapter as A
import r1b_g_l_native as E
import r1b_g_l_dynamics as D


class Context:
    def __init__(self, root, model):
        self.model_name, self.mock, self.freeze = "gemma4-e2b", True, {}
        self.device, self.root = "cpu", str(root)
        self.work = str(Path(root) / "stages/l_dynamics")
        self.outputs, self._model, self._cache = [], model, {}
        self._data = {"wt": {"test": ["A river flows through a green valley. " * 30]},
                      "ref_k16": ["A quiet view of a river and bridge." for _ in range(4)]}

    def model(self):
        return self._model

    def data(self):
        return self._data

    def path(self, rel):
        p = Path(self.work) / rel; p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)

    def artifact(self, stage, rel):
        return Path(self.root) / "stages" / stage / rel

    def write_json(self, rel, value, **kwargs):
        Path(self.path(rel)).write_text(json.dumps(value, allow_nan=False))
        self.outputs.append(rel)

    def save_pt(self, rel, value, **kwargs):
        torch.save(value, self.path(rel)); self.outputs.append(rel)

    def cached(self, name, fn, **kwargs):
        if name not in self._cache:
            self._cache[name] = fn()
        return self._cache[name]

    def time_left(self):
        return 100000

    def key(self, *parts):
        import r1b_core as core
        return core.keyseed("G-dynamics-test", *parts)


def references(model):
    n = A.NeuronLayout.from_model(model).total
    return {k: torch.zeros(n) if k.startswith("mu") else torch.ones(n)
            for k in ("mu_z", "sd_z", "mu_cett", "sd_cett")}


def fixture(root, pair):
    """Explicit synthetic populations/labels, independent of scientific K runs."""
    tok, model = pair
    root = Path(root)
    ctx = Context(root, pair)
    for stage in ("l_prepare", "k_prepare"):
        (root / "stages" / stage).mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(16):
        split = "S_L" if i < 12 else "NE"
        cls = ("CC" if i < 4 else "CI") if split == "S_L" else "NE"
        items.append({"id": f"q{i:02}", "question": f"Code {i}?", "split": split, "cls": cls,
                      "ans0": tok("red", add_special_tokens=False)["input_ids"], "aliases": ["red"]})
    layout = A.NeuronLayout.from_model(model)
    pop = {"status": "measured", "P": list(range(12)), "SEL": list(range(12, 24)), "UNC": list(range(24, 36)),
           "CTRL": [list(range(36 + i * 12, 48 + i * 12)) for i in range(9)], "boot": [], "CTRL_reserve": [],
           "sg8_ok": False, "labels": ["synthetic fixture"], "P_name": "fixture", "layout": layout.as_dict()}
    (root / "stages/l_prepare/items.json").write_text(json.dumps({"items": items}))
    (root / "stages/l_prepare/l_pop.json").write_text(json.dumps(pop))
    (root / "stages/k_prepare/hn.json").write_text(json.dumps({"C_star": .05, "mock": True}))
    torch.save(references(model), root / "stages/l_prepare/refw.pt")
    prompt = A.chat_ids(tok, "River?")
    sequence = prompt + tok("blue", add_special_tokens=False)["input_ids"]
    rows = [{"ids": sequence, "positions": list(range(len(prompt) - 1, len(sequence) - 1))}] * 2
    torch.save({"ref_r16": rows}, root / "stages/k_prepare/references.pt")
    return ctx


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.pair = A.load_model("gemma4-e2b", mock=True)

    def test_native_ragged_probe_cache_tf_and_crn(self):
        tok, model = self.pair
        layout = A.NeuronLayout.from_model(model)
        units = [1, layout.total - 1]
        prompt = A.chat_ids(tok, "hello")
        with E.Probe(model, references(model), chans=[("P", units)], spins=units) as probe:
            rows = E.decode(model, tok, [prompt, prompt], [2, 2], max_new=8, stop=False, probe=probe)
            self.assertEqual(rows[0]["tokens"], rows[1]["tokens"])
            np.testing.assert_array_equal(rows[0]["a"], rows[1]["a"])
            tf = E.tf_pass(model, tok, [prompt + rows[0]["tokens"]], [len(prompt)], probe=probe)[0]
            np.testing.assert_allclose(rows[0]["a"], tf["a"], atol=2e-5)
            seq = prompt + rows[0]["tokens"]
            lp = E.full_logprobs(model, seq)
            cache = E.lce_logprobs(model, seq, chunk=4)
            np.testing.assert_allclose(lp[1:], cache[1:], atol=2e-5)
        self.assertTrue(all(not l.mlp.down_proj._forward_hooks and not l.mlp.down_proj._forward_pre_hooks
                            for l in A.layers_of(model)))

    def test_live_push_retains_native_kv_and_sham_is_identical(self):
        tok, model = self.pair
        prompt = A.chat_ids(tok, "blue river")
        # Change a subset in an early layer: its cached representation must
        # survive after the edit is switched off, even for an unchanged token.
        layout = A.NeuronLayout.from_model(model)
        units = [0, 1]
        with E.Probe(model, references(model), chans=[("P", units), ("late", [layout.offsets[1] + 1])]) as probe:
            probe.add_op(units, "scale", val=1., rows=[False, True], on=False)
            def switch(step, p):
                p.ops[0]["on"] = step <= 2
            rows = E.decode(model, tok, [prompt, prompt], [3, 3], max_new=8, stop=False, probe=probe, on_step=switch)
            self.assertEqual(rows[0]["tokens"], rows[1]["tokens"])
            np.testing.assert_allclose(rows[0]["a"], rows[1]["a"], atol=1e-6)
            probe.ops[0]["val"] = 2.
            pushed = E.decode(model, tok, [prompt, prompt], [3, 3], max_new=8, stop=False, probe=probe, on_step=switch)
            self.assertGreater(float(np.max(np.abs(pushed[0]["a"] - pushed[1]["a"]))), 1e-6)

    def test_anchor_information_and_gemma_turn_suffix(self):
        tok, model = self.pair
        prompt = A.chat_ids(tok, "hello")
        same = E.ctx_logprobs(model, tok, [([prompt] * 3, [70, 71, 72])])[0]
        np.testing.assert_array_equal(same["kl"], np.zeros((2, 3)))
        suffix = tok.decode(E.turn_suffix_ids(tok, "Continue."), skip_special_tokens=False)
        self.assertIn("<|turn>user", suffix)
        self.assertNotIn("<|im_start|>", suffix)

    def test_missing_g_inputs_blocks_every_branch(self):
        with tempfile.TemporaryDirectory() as td:
            result = D.measure(Context(td, self.pair))
            self.assertEqual(result["status"], "partial")
            self.assertTrue(all(r["status"] == "blocked" for r in result["coverage"].values()))

    def test_reserved_reference_questions_never_enter_lr_or_twins(self):
        items = {f"q{i:02}": {"id": f"q{i:02}", "question": "Q?", "split": "S_L", "cls": "CC"} for i in range(12)}
        cf = {"ref_heldout": 4, "n_fam": 12, "l3b_prompts": 12}
        excluded = set(list(items)[-4:])
        self.assertFalse(excluded & {r["id"] for r in D.lr_items(items, cf)["F-known"]})
        self.assertFalse(excluded & {r["id"] for _, r in D.l3b_prompts(items, cf)})

    def test_inherited_dynamics_statistics_match(self):
        import r1b_l_b2 as old
        rng = np.random.default_rng(41)
        B = rng.integers(0, 2, (4, 32), dtype=np.int8)
        lengths = np.array([32, 31, 25, 28]); bins = np.tile(np.arange(32) // 8, (4, 1))
        entropy = rng.normal(size=(4, 32))
        np.testing.assert_array_equal(D.design(B, lengths, bins, entropy), old.design(B, lengths, bins, entropy))
        self.assertEqual(D.hd_vec(B, lengths, bins), old.hd_vec(B, lengths, bins))
        cf = {"pos_scale": 1 / 16}
        self.assertEqual(D.entries(B[0], 32, cf), old.entries(B[0], 32, cf))
        self.assertEqual(D.exits(B[0], 32, cf), old.exits(B[0], 32, cf))

    def test_complete_protocol_fixture_and_manifest_resume(self):
        tok, model = self.pair
        cfg = A.text_config(model)
        original = cfg.max_position_embeddings
        # A larger native random fixture accommodates all three mock turns;
        # this changes no real model and is never scientific evidence.
        cfg.max_position_embeddings = 1024
        try:
            with tempfile.TemporaryDirectory() as td:
                ctx = fixture(td, self.pair)
                result = D.measure(ctx)
                self.assertEqual(result["status"], "completed", result)
                self.assertTrue(all(r["status"] == "measured" for r in result["coverage"].values()))
                ising = json.loads(Path(ctx.path("l3c.json")).read_text())
                self.assertIn("h_basin", ising)
                self.assertEqual(ising["glauber_dwell"]["n_simulated_chains"], 16)
                expected = {f"l{part}.json" for part in ("2b", "2c", "3a", "3b", "3c", "3d")}
                ctx.outputs.clear()
                resumed = D.measure(ctx)
                self.assertEqual(resumed["status"], "completed")
                self.assertTrue(expected.issubset(ctx.outputs))
                self.assertTrue(resumed["coverage"]["L2b"]["resumed"])
        finally:
            cfg.max_position_embeddings = original

    def test_dwell_keeps_chain_boundaries_and_censoring(self):
        paths = [{"labels": np.array([0, 0, 1, 1, 1, 2, 2])}, {"labels": np.array([2, 2, 2])}]
        result = D._dwell_summary(paths)
        self.assertEqual(result["n_runs"], 2)
        self.assertEqual(result["n_right_censored"], 1)
        self.assertEqual(result["uncensored_mean"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
