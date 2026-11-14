"""Offline L5 checks: native generation, token alignment and honest gates."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import r1b_core as K
import r1b_g_l_owner as O
from r1b_g_engine import ModelEngine
from r1b_g_mock import make_mock

torch.set_num_threads(1)


class Context:
    mock = True
    model_name = "gemma4-e2b"
    freeze = {}

    def __init__(self, root, pair, data):
        self.root, self.pair, self.data_ = Path(root), pair, data
        self.work = self.root / "l_owner"
        self.work.mkdir(exist_ok=True)
        self.cache = {}

    def model(self):
        return self.pair

    def data(self):
        return self.data_

    def artifact(self, stage, name):
        return self.root / stage / name

    def key(self, *parts):
        return K.keyseed(*parts)

    def time_left(self):
        return 10000

    def cached(self, name, function):
        if name not in self.cache:
            self.cache[name] = function()
        return self.cache[name]

    def save_pt(self, name, value, **kwargs):
        torch.save(value, self.work / name)

    def write_json(self, name, value, **kwargs):
        (self.work / name).write_text(json.dumps(value, allow_nan=False))


def raw_item(ident="ar0"):
    return {"id": ident, "prompt": "Compute (2 + 3) * 4. Reply with only the final number.",
            "P": [2, 3, 4], "I": [5], "A": 20, "Iplus": [], "ops": ["+", "*"],
            "nops": 2, "split": "A-lens", "cal": False}


class OwnerTests(unittest.TestCase):
    def test_standardization_precedes_population_mean_on_ragged_coordinates(self):
        row = {"neuron_ids": [2, 9], "cett": torch.tensor([[5., 14.], [7., 18.]])}
        refs = {"mu_cett": np.array([0, 0, 3, 0, 0, 0, 0, 0, 0, 10]),
                "sd_cett": np.array([1, 1, 2, 1, 1, 1, 1, 1, 1, 4])}
        result = O.standardized_channels(row, {"P": [2, 9], "C0": [9]}, refs)
        np.testing.assert_allclose(result, [[1, 1], [2, 2]])

    def test_completed_multidigit_runs_keep_spaces_and_ignore_trailing_partial(self):
        gate, generated, records = {}, {}, []
        for i in range(4):
            ident = f"i{i}"
            gate[ident] = {"answers": ["99"], "V": [12], "stratum": "known" if i < 2 else "prone", "ops": 2}
            generated[ident] = {"gate": [], "cot": [], "sub": []}
            records.append({"item": ident, "tokens_text": ["1", "2", " ", "9", "9", " ", "3"],
                            "variant": "V1", "ptype": "t0", "block": 1, "seed_rank": 0,
                            "k_star_conf": 4, "share_H": None})
        with np.errstate(invalid="ignore"):
            result, rows = O.ownership(gate, records, generated, 20)
        self.assertEqual(rows[0]["classes"], [[1, 12, "TRUE"], [4, 99, "MODEL"]])
        self.assertEqual(rows[0]["BM"], 4)
        self.assertIsNone(rows[0]["BF"])
        self.assertEqual(result["kconf_model_share"], 1.)
        self.assertTrue(all(row["foreign_item"] != row["item"] for row in rows))

    def test_first_step_uses_prompt_state_and_later_steps_exclude_their_operands(self):
        text = "2 + 3 = 5\n5 * 4 = 21\n"
        activity = np.arange(len(text) + 1, dtype=float)[:, None]
        cot = {"text": text, "offsets": [-1] + list(range(len(text))), "alignment_ok": True,
               "a": activity, "e": np.zeros(len(activity)), "sample": 0,
               "absolute_positions": list(range(20, 20 + len(activity)))}
        gate = {"i": {"V": [2, 3, 4, 5, 20], "steps": [[2, "+", 3, 5], [5, "*", 4, 20]], "I_plus": []}}
        observations, report = O.step_observations(gate, {"i": {"cot": [cot]}})
        self.assertEqual(len(observations), 2)
        self.assertEqual(observations[0]["read_char_offset"], -1)
        self.assertEqual(observations[0]["a"][0], 0)
        second = observations[1]
        self.assertEqual(second["read_char_offset"], text.index("\n"))
        self.assertLess(second["read_char_offset"], second["line_start_char"])
        self.assertEqual(second["kind"], "result")
        self.assertEqual(report["n_events"], 1)
        changed = activity.copy()
        changed[second["read_token_index"] + 1:] = 1e9
        after, _ = O.step_observations(gate, {"i": {"cot": [{**cot, "a": changed}]}})
        np.testing.assert_array_equal(after[1]["a"], second["a"])

    def test_unaligned_or_unparseable_text_stays_in_parse_denominator(self):
        cot = {"text": "2 + 3 = 5", "offsets": [-1], "alignment_ok": False}
        observations, report = O.step_observations({"i": {}}, {"i": {"cot": [cot]}})
        self.assertEqual(observations, [])
        self.assertEqual(report["parse_share"], 0)
        self.assertEqual(report["n_misaligned"], 1)
        self.assertFalse(report["parse_ok"])

    def test_multichar_token_crossing_line_boundary_is_excluded_from_pre_step(self):
        pieces = ["2 + 3 = 5", "\n5", " * 4 = 21", "\n"]
        text = "".join(pieces)
        starts = np.cumsum([0] + list(map(len, pieces))).tolist()
        cot = {"text": text, "offsets": [-1] + starts[:-1], "token_ends": [0] + starts[1:],
               "alignment_ok": True, "a": np.arange(5, dtype=float)[:, None], "e": np.zeros(5),
               "sample": 0, "absolute_positions": list(range(20, 25))}
        gate = {"i": {"V": [2, 3, 4, 5, 20], "steps": [[2, "+", 3, 5], [5, "*", 4, 20]], "I_plus": []}}
        observations, _ = O.step_observations(gate, {"i": {"cot": [cot]}})
        # '\n5' contains the second line's operand, so the pre-step read is
        # the preceding whole token, even though its end precedes the newline.
        self.assertEqual(observations[1]["read_token_index"], 1)
        self.assertLessEqual(observations[1]["read_char_end"], observations[1]["line_start_char"])
        self.assertTrue(observations[1]["equal_token_contains_result"])

    def test_a_closed_lens_does_not_make_native_own_error_gate_pass(self):
        gates = {"G_CL": False, "target_lens": False, "SG8": False, "cal_L5a": False,
                 "cal_L5c": False, "controls_99": False, "share_H": False, "own_sample_size": False}
        result = O.decide({"L5a": None, "L5b": None, "L5c": {"parse_ok": False}}, gates)
        self.assertTrue(all(row["verdict"].startswith("not testable") for row in result["verdicts"].values()))
        self.assertEqual(result["CF_HERR_member"], 1)
        self.assertEqual(result["combined"]["V-OWN"]["verdict"], "not decidable")

    def test_pattern_uses_genuine_share_and_does_not_fake_99_controls(self):
        generated = {f"i{i}": {"tstar": np.array([i, -i], float)} for i in range(4)}
        rows = [{"item": f"i{i}", "block": 1, "BF": 4 if seed < i else None,
                 "share_H": [float(seed < i)] * 17} for i in range(4) for seed in range(4)]
        rows += [{**row, "variant": "V2", "share_H": [-value for value in row["share_H"]]} for row in rows]
        result = O.pattern(type("C", (), {"mock": True})(), rows, generated, ["P", "C0"], 12)
        self.assertGreater(result["rho_b"]["pt"], .9)
        self.assertGreater(result["rho_w"]["pt"], .9)
        self.assertEqual(result["rank_p"], 1.)
        self.assertEqual(result["n_share_rows"], 16)
        for row in rows:
            row["share_H"] = None
            row["aX"] = {"H": [1.] * 17}
        self.assertIsNone(O.pattern(type("C", (), {"mock": True})(), rows, generated, ["P", "C0"], 12))

    def test_hazard_fits_item_cluster_controls_without_target_gradients(self):
        gate, generated = {}, {}
        random = np.random.default_rng(10)
        for i in range(8):
            ident = f"i{i}"
            gate[ident] = {"V": [2, 3, 4, 5, 20], "I_plus": [], "stratum": "known" if i < 4 else "prone",
                           "steps": [[2, "+", 3, 5], [5, "*", 4, 20]]}
            generated[ident] = {"cot": []}
            for sample in range(4):
                error = sample % 2 == 1
                text = "2 + 3 = 5\n5 * 4 = " + ("21" if error else "20") + "\n"
                activity = random.normal(size=(len(text) + 1, 3))
                activity[0, 0] = -2 + random.normal(scale=.2)
                activity[text.index("\n") + 1, 0] = (2 if error else -2) + random.normal(scale=.2)
                generated[ident]["cot"].append({"text": text, "offsets": [-1] + list(range(len(text))),
                    "alignment_ok": True, "a": activity, "e": random.normal(size=len(activity)), "sample": sample,
                    "absolute_positions": list(range(30, 30 + len(activity)))})
        result = O.hazard(type("C", (), {"mock": True})(), gate, generated, ["P", "UNC", "C0"], 8)
        self.assertTrue(result["fit_available"])
        self.assertEqual(result["n_events"], 16)
        self.assertGreater(result["beta_H"]["pt"], 0)
        self.assertEqual(result["rank_p"], 1)
        self.assertEqual(result["p_cf_herr"], 1)
        self.assertIn("minus_E", result)
        self.assertIn("minus_unc", result)
        self.assertIn("beta_E_without_duplicate_entropy", result)

    def test_native_item_generation_and_resume_need_no_lens_or_h_artifact(self):
        with tempfile.TemporaryDirectory() as td, patch("socket.socket.connect", side_effect=AssertionError("network used")):
            pair = make_mock("e2b")
            context = Context(td, pair, {"arithmetic": [raw_item()]})
            result = O.measure(context)
            self.assertEqual(result["status"], "partial")
            output = torch.load(context.work / "l5_gen.pt", weights_only=False)
            generated = output["gen"]["ar0"]
            self.assertEqual(len(generated["gate"]), 6)
            self.assertEqual(len(generated["cot"]), 5)
            self.assertEqual(len(generated["sub"]), 10)
            self.assertEqual(output["channels"], [])
            for cot in generated["cot"]:
                self.assertEqual(len(cot["a"]), len(cot["tokens"]) + 1)
                self.assertEqual(cot["offsets"][0], -1)
                self.assertTrue(np.isfinite(cot["e"]).all())
            verdicts = json.loads((context.work / "l5_verdicts.json").read_text())
            self.assertFalse(verdicts["gates"]["G_CL"])
            self.assertFalse(verdicts["gates"]["SG8"])
            self.assertFalse(verdicts["gates"]["own_sample_size"])
            with patch.object(ModelEngine, "decode", side_effect=AssertionError("resume regenerated samples")):
                again = O.measure(context)
            self.assertEqual(again["gates"], result["gates"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
