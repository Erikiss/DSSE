"""Offline native-Gemma checks for G L; no scientific claims or downloads."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import r1b_core as K
from r1b_g_adapter import NeuronLayout
from r1b_g_engine import ModelEngine
from r1b_g_mock import make_mock
from r1b_g_l_long import logprobs, own_continuation, long_context
from r1b_g_l_erasure import partial_mean, conditions, Damage, erasure
from r1b_g_l import prepare, _answer, config, _sl, GuardedContext, analyze

torch.set_num_threads(1)


class Context:
    mock = True
    model_name = "gemma4-e2b"
    device = "cpu"
    freeze = {}

    def __init__(self, root, stage, model, data):
        self.root, self.stage = Path(root), stage
        self.work = self.root / stage
        self.work.mkdir(parents=True, exist_ok=True)
        self._model, self._data = model, data

    def model(self):
        return self._model

    def data(self):
        return self._data

    def artifact(self, stage, relative):
        return self.root / stage / relative

    def write_json(self, relative, value, **kwargs):
        (self.work / relative).write_text(json.dumps(value, allow_nan=False))

    def save_pt(self, relative, value, **kwargs):
        torch.save(value, self.work / relative)

    def cached(self, name, fn):
        return fn()

    def key(self, *parts):
        return K.keyseed(*parts)


class GemmaLTests(unittest.TestCase):
    def test_cached_logprobs_match_full_native_across_local_windows(self):
        with patch("socket.socket.connect", side_effect=AssertionError("network used")):
            for kind in ("e2b", "12b"):
                tok, model = make_mock(kind)
                ids = tok("A river flows through the valley. Trees grow beside it.", add_special_tokens=False)["input_ids"]
                cached = logprobs(model, ids, chunk=7, logit_chunk=5)
                full = logprobs(model, ids, cached=False, logit_chunk=5)
                self.assertTrue(np.isnan(cached[0]))
                np.testing.assert_allclose(cached[1:], full[1:], rtol=1e-6, atol=1e-6)
                with torch.no_grad():
                    logits = model(torch.tensor([ids]), use_cache=False).logits[0, :-1].log_softmax(-1)
                    expected = logits.gather(1, torch.tensor(ids[1:])[:, None])[:, 0].numpy()
                np.testing.assert_allclose(cached[1:], expected, rtol=1e-6, atol=1e-6)

    def test_own_generation_is_keyed_and_logs_unmasked_eos_probability(self):
        with tempfile.TemporaryDirectory() as td:
            pair = make_mock()
            E = ModelEngine(Context(td, "l_measure", pair, {}))
            seed = E.encode("A river.", chat=False)
            first = own_continuation(E, seed, 123, 12)
            again = own_continuation(E, seed, 123, 12)
            self.assertEqual(first, again)
            self.assertEqual(len(first["tokens"]), 12)
            self.assertTrue(all(0 < p < 1 for p in first["peos"]))
            self.assertFalse(set(first["tokens"]) & {0, 1, 5})
            records = {}
            def checkpoint(offset, fn):
                if offset not in records:
                    records[offset] = fn()
                return records[offset]
            chunked = own_continuation(E, seed, 123, 12, checkpoint=checkpoint, chunk_size=4)
            self.assertEqual(first["tokens"], chunked["tokens"])
            del records[4], records[8]
            resumed = own_continuation(E, seed, 123, 12, checkpoint=checkpoint, chunk_size=4)
            self.assertEqual(first["tokens"], resumed["tokens"])
            np.testing.assert_allclose(first["peos"], resumed["peos"], rtol=1e-6, atol=1e-8)

    def test_guard_yields_before_a_new_compute_piece(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = Context(td, "l_measure", None, {})
            ctx.time_left = lambda: 0
            with self.assertRaises(K.StepYield):
                GuardedContext(ctx).cached("test", lambda: self.fail("computation ran after guard"))

    def test_analysis_keeps_independent_branches_and_blocks_unvalidated_H(self):
        with tempfile.TemporaryDirectory() as td:
            def stat(value):
                return {"pt": value, "lo": value - .01, "hi": value + .01}
            calibration = Context(td, "l_calibrate", None, {})
            calibration.write_json("l_cal.json", {"gates": {"L4": True, "L2a": True}, "quick": False})
            Context(td, "l_measure", None, {}).write_json("l2a.json", {"gmi_ok": False})
            Context(td, "l_duet", None, {}).write_json("l4.json", {
                "g_duet": True, "J_ex": stat(.4), "ll_gain": stat(.4), "J_ex_minus_doc": stat(.3),
                "consensus": stat(.7), "agree_excess": stat(.4), "gap_ratio": stat(.7), "g_manip": True,
                "dmg_ok": True, "DD": stat(.3), "rup_minus_half": stat(.2), "hup_minus_half": stat(.3),
                "catfish": stat(.4), "dpsi": stat(-.3)})
            result = analyze(Context(td, "l_analyze", None, {}))
            self.assertEqual(result["verdicts"]["L4"]["pair"], "agents Ising-modellable (coupled)")
            self.assertEqual(result["verdicts"]["L4"]["h_line"], "not testable (no H-set)")
            self.assertIn("estimator check failed", result["verdicts"]["L2a"]["growth"])
            self.assertFalse(result["scientific_complete"])

    def test_equivalence_power_guard_uses_planned_se_and_preserves_directional_findings(self):
        from r1b_g_l_power import assessment, guard
        self.assertTrue(assessment({"se": .06}, .04, "declared")["passed"])
        self.assertFalse(assessment({"se": .060001}, .04, "declared")["passed"])
        for bad in (-1., float("nan"), True, {"code": "NaN:undefined"}):
            self.assertIsNone(assessment({"se": bad}, .04, "declared")["passed"])
        verdicts = {"L4": {"pair": "no detectable inter-agent coupling",
                           "collaboration": "collaboration (coupling beyond lexical priming)",
                           "macro": "no pair macro beyond uncoupled agents"},
                    "L2b": {"verdict": "H rises beyond information loss", "labels": []},
                    "L1_erasure": {"curve": "threshold-like within f <= 1/16", "privilege": "H causally privileged"}}
        statistics = {"L4": {"J_ex": {"se": .07}, "dpsi": {"se": .001}}}
        guarded, checks = guard(verdicts, statistics)
        self.assertEqual(guarded["L4"]["pair"], "not decidable (underpowered)")
        self.assertEqual(guarded["L4"]["macro"], "not decidable (planning SE unavailable)")
        self.assertEqual(guarded["L4"]["collaboration"], verdicts["L4"]["collaboration"])
        self.assertEqual(guarded["L2b"], verdicts["L2b"])
        self.assertEqual(guarded["L1_erasure"]["curve"], "not decidable (planning SE unavailable)")
        self.assertEqual(guarded["L1_erasure"]["privilege"], "H causally privileged")
        self.assertEqual(verdicts["L4"]["pair"], "no detectable inter-agent coupling")
        self.assertAlmostEqual(checks["L4.pair"]["J_ex"]["planning_se"], .04)

    def test_analysis_recomputes_combined_owner_after_power_guard(self):
        with tempfile.TemporaryDirectory() as td:
            owner = Context(td, "l_owner", None, {})
            owner.write_json("l5_verdicts.json", {"verdicts": {
                "L5a": {"verdict": "break-off owned by the model"},
                "L5b": {"verdict": "not testable", "labels": []},
                "L5c": {"verdict": "not testable", "labels": []}},
                "combined": {"V-OWN": {"verdict": "belong to the model"}}, "coverage": {}})
            owner.write_json("l5_stats.json", {"L5a": {"GF_known": {"se": .1}}})
            result = analyze(Context(td, "l_analyze", None, {}))
            self.assertEqual(result["verdicts"]["L5a"]["verdict"], "not decidable (underpowered)")
            self.assertEqual(result["verdicts"]["V-OWN"]["verdict"], "not decidable")
            self.assertEqual(result["verdicts_before_power_guard"]["L5a"]["verdict"], "break-off owned by the model")
            self.assertFalse(result["equivalence_power_checks"]["L5a.verdict"]["GF_known"]["passed"])

    def test_bounded_locus_algebra_matches_inherited_definitions(self):
        import r1b_g_l_locus as own
        import r1b_l_a as inherited
        generator = torch.Generator().manual_seed(19)
        X = torch.randn(12, 5000, generator=generator)
        X[:, 4490] = X[:, 2] * .9 + torch.randn(12, generator=generator) * .1
        self.assertEqual(own.hull(X, [2, 4999]), inherited.hull(X, [2, 4999]))
        s = np.where(np.arange(12) % 2, 1., -1.)
        for profile in ("dense", "thr16", "thr64"):
            a, ya = own.plant(X, profile, s, .8, np.random.default_rng(9), 2)
            b, yb = inherited.plant(X, profile, s, .8, np.random.default_rng(9), 2)
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
            self.assertEqual(ya, yb)

    def test_partial_mean_identity_and_hook_cleanup_with_ragged_neurons(self):
        with tempfile.TemporaryDirectory() as td:
            E = ModelEngine(Context(td, "l_measure", make_mock(), {}))
            ids = E.encode("A tree.", chat=False)
            base = E.next_logits(ids)
            neurons = [E.layout.encode(0, 2), E.layout.encode(3, 80)]
            means = torch.ones(E.layout.total) * 4
            with partial_mean(E, neurons, means, 1.0):
                torch.testing.assert_close(E.next_logits(ids), base)
            with partial_mean(E, neurons, means, 0.0):
                self.assertGreater(float((E.next_logits(ids) - base).abs().max()), 1e-4)
            torch.testing.assert_close(E.next_logits(ids), base)
            with self.assertRaisesRegex(RuntimeError, "test cleanup"):
                with partial_mean(E, neurons, means, 0):
                    raise RuntimeError("test cleanup")
            torch.testing.assert_close(E.next_logits(ids), base)

    def test_insufficient_controls_are_not_duplicated(self):
        cf = config(type("C", (), {"mock": True})())
        population = {"P": [1, 3], "H_tq": [1, 3], "CTRL": [[2, 4]]}
        conds = conditions(None, population, 288, cf)
        self.assertEqual(len([c for c in conds if "ctrl" in c]), 1)

    def test_bounded_native_damage_matches_shared_kl_definition(self):
        with tempfile.TemporaryDirectory() as td:
            E = ModelEngine(Context(td, "l_measure", make_mock(), {}))
            rows = [E.encode(t, chat=False) for t in ("A river flows.", "A tree grows beside it.")]
            refs = K.RefSet([(s, list(range(len(s) - 1))) for s in rows], "test")
            bounded, original = Damage(E.model, refs, chunk=3), K.Damage(E.model, refs)
            factory = lambda: E.intervene([E.layout.encode(3, 80), E.layout.encode(1, 12)], op="zero")
            a, b = bounded(factory), original(factory)
            np.testing.assert_allclose(a["per_text"], b["per_text"], atol=2e-12, rtol=1e-5)
            self.assertEqual(a["per_text_top1"], b["per_text_top1"])
            self.assertEqual(a["refs_sha256"], b["refs_sha256"])
            self.assertEqual(bounded()["damage"], 0.0)

    def test_native_erasure_uses_disjoint_heldout_damage_items(self):
        with tempfile.TemporaryDirectory() as td:
            pair = make_mock()
            E = ModelEngine(Context(td, "l_measure", pair, {}))
            data = {"ref_k16": ["A tree grows.", "A river flows.", "A bird sings.", "A boat floats."]}
            kp = Context(td, "k_prepare", pair, data)
            refs = []
            for question in ("Name a tree.", "Name a bird."):
                p, c = E.encode(question), E.encode("Oak.", chat=False)
                refs.append({"ids": p + c, "positions": list(range(len(p) - 1, len(p) + len(c) - 1))})
            kp.save_pt("references.pt", {"ref_r16": refs})
            lp = Context(td, "l_prepare", pair, data)
            rows = []
            for i in range(12):
                rows.append({"id": f"question-{i:02d}", "question": f"Who has {i} trees?", "aliases": [str(i)],
                             "cls": "CC" if i % 2 == 0 else "CI", "split": "S_L", "ans0": E.encode(str(i), chat=False), "span0": [0, 1]})
            rows += [{"id": f"NE{i}", "question": f"Who founded imaginary city {i}?", "aliases": [], "cls": "NE", "split": "NE",
                      "ans0": E.encode("Nobody", chat=False), "span0": [0, 6]} for i in range(2)]
            lp.write_json("items.json", {"items": rows})
            lp.write_json("l_pop.json", {"P": [2], "H_tq": [2], "CTRL": [[3], [4]]})
            lp.save_pt("refw.pt", {"mu_z": torch.zeros(E.layout.total)})
            ctx = Context(td, "l_measure", pair, data)
            result = erasure(ctx)
            self.assertEqual(result["status"], "measured")
            measured = torch.load(ctx.work / "l1_era.pt", weights_only=False)
            self.assertEqual(len(measured["heldout_ids"]), 4)
            self.assertFalse(set(measured["ids"]) & set(measured["heldout_ids"]))
            self.assertEqual(set(measured["cls"]), {"CC", "CI", "NE"})

    def test_k_answer_span_and_question_fold_contract(self):
        self.assertEqual(_answer({"answers": [{"ids": [11, 12, 13]}], "span": 2}), ([11, 12, 13], [0, 2]))
        with tempfile.TemporaryDirectory() as td:
            ctx = Context(td, "l_prepare", None, {})
            ctx.save_pt("cett_rows.pt", {"ids": ["a", "a", "b", "c"], "split": ["S_L"] * 4,
                                        "cls": ["CC", "CI", "CC", "CI"], "cmean": torch.zeros(4, 2)})
            with self.assertRaisesRegex(ValueError, "unique question"):
                _sl(ctx)

    def test_native_prepare_preserves_ragged_widths_and_actual_features(self):
        with tempfile.TemporaryDirectory() as td:
            pair = make_mock()
            tok, model = pair
            layout = NeuronLayout.from_model(model)
            prep = Context(td, "k_prepare", pair, {})
            lab = []
            for split, count in (("S_fit", 12), ("S_L", 12)):
                for i in range(count):
                    lab.append({"qid": f"{split}:{i}", "question": f"Who has {i} trees?", "aliases": [str(i)],
                                "split": split, "cls": "CC" if i % 2 == 0 else "CI",
                                "answers": [{"ids": tok(str(i), add_special_tokens=False)["input_ids"]}], "span": 1})
            prep.write_json("lab.json", {"items": lab})
            prep.write_json("hn.json", {"H_op": [[0, 2], [3, 80]], "H_op_name": "H+", "H_tq": [[0, 2], [3, 80]],
                                       "H_plus": [[0, 2], [3, 80]],
                                       "C_star": 1., "boot": []})
            prep.save_pt("references.pt", {"z_mu": torch.zeros(layout.total), "z_sd": torch.ones(layout.total),
                                           "c_mu": torch.zeros(layout.total), "c_sd": torch.ones(layout.total)})
            ctx = Context(td, "l_prepare", pair, {})
            result = prepare(ctx)
            self.assertEqual(result["status"], "completed")
            data = torch.load(ctx.work / "cett_rows.pt", weights_only=False)
            self.assertEqual(data["cmean"].shape, (24, 288))
            self.assertTrue(torch.isfinite(data["cmean"]).all())
            self.assertGreater(float(data["cmean"].var()), 0)
            pop = json.loads((ctx.work / "l_pop.json").read_text())
            self.assertEqual(pop["P"], [2, 272])
            self.assertEqual(pop["layer_counts"], [1, 0, 0, 1])
            self.assertFalse(pop["sg8_ok"])
            # Exercise actual held-out probe fits and early-stop QR captures.
            # Reduced test counts never become scientific protocol defaults.
            import r1b_g_l as gl
            import r1b_g_l_locus as locus
            cf = config(ctx)
            cf.update(pip_frag=1, pip_perm=1, pip_grid=[1, 2], syn_grid=[1], n_resel=2,
                      l1c_frag=1, l1c_perm=1, l1c_seeds=1, nboot=20)
            measured = Context(td, "l_measure", pair, {})
            with patch.object(gl, "config", return_value=cf):
                locus.l1_pip(measured)
                gl.rotated_features(measured)
                locus.l1_resel(measured)
                locus.l1_cal(measured)
                locus.l1_stats(measured)
            pip = torch.load(measured.work / "l1_pip.pt", weights_only=False)
            self.assertEqual(pip["M"].shape[1], 12)
            self.assertTrue(torch.isfinite(pip["M"]).all())
            self.assertEqual(set(pip["ids"]), {f"S_L:{i}" for i in range(12)})
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
            stats = json.loads((measured.work / "l1.json").read_text())
            self.assertEqual(set(stats["l1c"]["verdicts"]), set(locus.PROFILES))
            self.assertIn("tree_input", stats)

    def test_l2a_full_mock_controls_use_native_gemma(self):
        with tempfile.TemporaryDirectory() as td:
            paragraphs = [f"Book {i}. " + "A river flows near a town. Trees grow beside a bridge. " * 20 for i in range(3)]
            data = {"longdocs": [{"id": str(i), "text": p} for i, p in enumerate(paragraphs)]}
            ctx = Context(td, "l_measure", make_mock(), data)
            result = long_context(ctx)
            self.assertEqual(result["status"], "measured")
            self.assertTrue(result["lce_ok"])
            main = torch.load(ctx.work / "l2a_main.pt", weights_only=False)
            self.assertEqual(main["I"].shape, (3, 4))
            self.assertTrue(np.isfinite(main["I"]).all())
            self.assertTrue(all(i != j for i, j in enumerate(main["perm"])))
            stats = json.loads((ctx.work / "l2a.json").read_text())
            self.assertEqual(stats["bootstrap_unit"], "document")
            # Random native models are not expected to pass G-MI.
            self.assertIsInstance(stats["gmi_ok"], bool)


if __name__ == "__main__":
    unittest.main()
