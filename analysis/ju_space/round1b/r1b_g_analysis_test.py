"""G family composition: fixed multiplicities, gates and verified inputs."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import r1b_core as K
from r1b_g_analysis import summarize, _verified_source, analyze, LINK_MEMBERS, ATTRACTOR_MEMBERS


def evidence():
    p = .001
    association = {"string": "H-activity predicts lens confabulation beyond chain length", "p_iut": .025, "p_iut_pi": p}
    causal = {"string": "(i) H clamp in the rollout reduces lens confabulation specifically", "p_iut": .025, "p_iut_pi": p}
    attr_strings = {name: "H-neuron attractor in the chain lens (bistable within items)" if "bistability" in name and name.startswith("K6e") else
                         "within-item bistability (sessions)" if "bistability" in name else
                         "H-neuron attractor in the chain lens (self-sustaining carry, TOK)" if name.startswith("K6e") else
                         "self-sustaining H (TQA)" if "TQA" in name else "self-sustaining H (ACH)"
                    for name in ATTRACTOR_MEMBERS}
    return {
        "detection": {"SG8": True, "K3a": {"arith": {"string": "H-neurons replicated (in-domain detection) transfers to arithmetic errors"}}},
        "chain": {"gates": {key: {"pass": True} for key in ("G_K_J", "G_K4", "G_K7")},
                  "statistics": {"K6b_i_provisional": copy.deepcopy(association), "K6b_ii_provisional": copy.deepcopy(association),
                                 "K6c_provisional": {"(i)": copy.deepcopy(causal), "(ii)": copy.deepcopy(causal)}}},
        "persistence": {"Q9": {name: {"p": p} for name in ATTRACTOR_MEMBERS},
                        "strings": {**attr_strings, "K6e PUSH_H": "H push raises lens confabulation beyond random sets",
                                    "K6e CLAMP_H": "H clamp lowers lens confabulation beyond random clamps"},
                        "K6e": {"causal": {name: {"p_iut": p} for name in ("PUSH_H", "CLAMP_H")}}},
        "persistence_status": {"missing": {}, "status": "done"},
        "extra": {"branches": {"K6h": {"G_K_J": True, "statistics": {"G_K23": "lens reads the gold answer (QA)",
                                                       "Q12": {"p_iut": .025, "p_iut_pi": p}}}}},
        "long": {"string": "H-activity predicts the next error beyond length, position, difficulty and uncertainty",
                 "rule": {"p_iut": .025}, "G_K13": None, "parse_fail": .1},
        "owner": {"verdicts": {"L5c": {"verdict": "H predicts the model's own step errors (within item, beyond step and uncertainty)"}},
                  "CF_HERR_member": .01},
        "l_analysis": {"verdicts": {"attractor": {"verdict": "H-attractor (self-sustaining or bistable, H-specific, seen in natural dynamics)"},
                                    "L2b": {"verdict": "H rises beyond information loss"}}},
        "l3a": {"bi_sur_p": .02, "bi_rank_p": .02, "bi_minus_sel": {"p_gt0": .02},
                "gamma_sur_p": .01, "dgamma_rank_p": .01, "dgamma_minus_sel": {"p_gt0": .01},
                "hd_rank_p": .02, "hd_minus_sel": {"p_gt0": .02}},
        "l3b": {"K_pool": {"p_gt1": .01}, "dK": {"p_gt0": .01}},
        "l2b": {"bie": {"p_gt0": .01}, "dbie": {"p_gt0": .02}},
    }


class FamilyTests(unittest.TestCase):
    def report(self, sources=None, mock=False):
        return summarize(evidence() if sources is None else sources, mock=mock, model="gemma4-e2b")

    def test_pi_is_primary_rank_is_only_descriptive(self):
        result = self.report()
        primary = result["families"]["CF-K-link"]
        self.assertEqual(set(primary["members"]), set(LINK_MEMBERS))
        self.assertEqual(primary["size"], 7)
        self.assertTrue(primary["members"]["K6b(i)"]["confirmed"])
        self.assertAlmostEqual(primary["members"]["K6b(i)"]["p_holm"], .007)
        rank = result["rank_sensitivity"]
        self.assertFalse(rank["confirmatory"])
        self.assertFalse(rank["any_confirmatory"])
        self.assertTrue(all(not value["confirmed"] for value in rank["members"].values()))

    def test_missing_members_keep_full_family_denominator(self):
        source = evidence()
        source["chain"]["statistics"].pop("K6b_ii_provisional")
        result = self.report(source)["families"]["CF-K-link"]
        self.assertEqual(result["size"], 7)
        self.assertEqual(result["members"]["K6b(ii)"]["p"], 1.)
        self.assertAlmostEqual(result["members"]["K6b(i)"]["p_holm"], .007)

    def test_unreached_part_string_cannot_be_confirmatory(self):
        source = evidence()
        source["chain"]["statistics"]["K6b_i_provisional"]["string"] = "undecided"
        member = self.report(source)["families"]["CF-K-link"]["members"]["K6b(i)"]
        self.assertEqual(member["raw_p"], .001)
        self.assertEqual(member["p"], 1.)
        self.assertFalse(member["confirmed"])

    def test_target_lens_gate_does_not_disable_model_side_attractor(self):
        source = evidence()
        source["chain"]["gates"]["G_K_J"]["pass"] = False
        source["extra"]["branches"]["K6h"]["G_K_J"] = False
        result = self.report(source)["families"]
        self.assertFalse(result["CF-K-link"]["any_confirmatory"])
        self.assertFalse(result["CF-K-attr"]["members"]["K6e TOK carry"]["confirmed"])
        self.assertTrue(result["CF-K-attr"]["members"]["K8b ACH carry"]["confirmed"])

    def test_qa_positive_control_is_independent_of_arithmetic_and_seed_label(self):
        source = evidence()
        source["chain"]["gates"]["G_K4"]["pass"] = False
        source["chain"]["gates"]["G_K7"]["pass"] = False
        result = self.report(source)
        members = result["families"]["CF-K-link"]["members"]
        self.assertTrue(members["K6h"]["confirmed"])
        self.assertFalse(members["K6b(i)"]["confirmed"])
        self.assertEqual(result["qualifiers"]["G_K7"]["effect"], "label only (K.6)")
        source["chain"]["gates"]["G_K4"]["pass"] = True
        self.assertTrue(self.report(source)["families"]["CF-K-link"]["members"]["K6b(i)"]["confirmed"])

    def test_format_gate_does_not_disable_independent_owner(self):
        source = evidence()
        source["long"]["G_K13"] = "format not followed"
        members = self.report(source)["families"]["CF-HERR"]["members"]
        self.assertEqual(members["K8c"]["p"], 1.)
        self.assertTrue(members["L5c"]["confirmed"])
        self.assertAlmostEqual(members["L5c"]["p_holm"], .02)

    def test_missing_sampling_status_cannot_confirm_persistence(self):
        source = evidence()
        source.pop("persistence_status")
        result = self.report(source)["families"]
        self.assertFalse(result["CF-K-attr"]["any_confirmatory"])
        for name in ("K6e-c push", "K6e-c clamp"):
            self.assertEqual(result["CF-K-link"]["members"][name]["p"], 1.)

    def test_cf_l_formula_is_exact_and_missing_attractor_is_independent(self):
        result = self.report()["families"]["CF-L"]
        self.assertEqual(result["members"]["L3 attractor"]["raw_p"], .02)
        self.assertEqual(result["members"]["L2b"]["raw_p"], .02)
        source = evidence()
        source.pop("l3a")
        source["l_analysis"]["verdicts"].pop("attractor")
        result = self.report(source)["families"]["CF-L"]
        self.assertEqual(result["members"]["L3 attractor"]["p"], 1.)
        self.assertTrue(result["members"]["L2b"]["confirmed"])
        source = evidence()
        source.pop("l2b")
        source["l_analysis"]["verdicts"].pop("L2b")
        result = self.report(source)["families"]["CF-L"]
        self.assertTrue(result["members"]["L3 attractor"]["confirmed"])
        self.assertEqual(result["members"]["L2b"]["p"], 1.)
        source = evidence()
        source.pop("l3b")
        result = self.report(source)["families"]["CF-L"]
        self.assertTrue(result["members"]["L3 attractor"]["confirmed"])
        self.assertEqual(result["members"]["L3 attractor"]["raw_p"], .04)

    def test_mock_and_empty_evidence_never_confirm(self):
        for result in (self.report(mock=True), self.report({})):
            self.assertEqual([f["size"] for f in result["families"].values()], [6, 7, 2, 2])
            self.assertTrue(all(not family["any_confirmatory"] for family in result["families"].values()))
        self.assertEqual(self.report({})["status"], "partial")

    def test_coded_or_invalid_p_values_are_not_positive_evidence(self):
        for value in (None, float("nan"), -1., 2., {"code": "NA"}):
            source = evidence()
            source["chain"]["statistics"]["K6b_i_provisional"]["p_iut_pi"] = value
            self.assertEqual(self.report(source)["families"]["CF-K-link"]["members"]["K6b(i)"]["p"], 1.)

    def test_artifacts_need_matching_target_and_stage_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root/"stages/k_analyze/k3_verdicts.json"
            path.parent.mkdir(parents=True)
            (root/"state").mkdir()
            data = {"mock": False, "provenance": {**{k: None for k in K.PROV_REQUIRED}, "g_model": "gemma4-e2b", "mock": False}}
            path.write_text(json.dumps(data))
            record = {"state": "done", "outputs": {path.name: K.sha256_file(path)}}
            state = root/"state/k_analyze.json"
            state.write_text(json.dumps(record))
            ctx = SimpleNamespace(model_run=root, model_name="gemma4-e2b", mock=False,
                                  artifact=lambda stage, name: root/"stages"/stage/name)
            self.assertEqual(_verified_source(ctx,"k_analyze",path.name)[1]["state"], "verified")
            data["SG8"] = True
            path.write_text(json.dumps(data))
            self.assertEqual(_verified_source(ctx,"k_analyze",path.name)[1]["state"], "invalid")
            record["outputs"][path.name] = K.sha256_file(path)
            state.write_text(json.dumps(record))
            ctx.model_name = "gemma4-e4b"
            self.assertIn("another target", _verified_source(ctx,"k_analyze",path.name)[1]["reason"])

    def test_analysis_entrypoint_never_needs_a_model(self):
        with tempfile.TemporaryDirectory() as td:
            saved = {}
            ctx = SimpleNamespace(model_name="gemma4-e2b", mock=True,
                                  artifact=lambda stage, name: Path(td)/stage/name,
                                  write_json=lambda name, obj, **kw: saved.update({name: obj}))
            ctx.model = lambda: self.fail("analysis loaded a target model")
            result = analyze(ctx)
            self.assertEqual(result["status"], "partial")
            self.assertIn("g_analysis.json", saved)
            self.assertFalse(saved["g_analysis.json"]["families"]["CF-HERR"]["any_confirmatory"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
