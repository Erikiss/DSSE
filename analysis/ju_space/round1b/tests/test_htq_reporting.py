"""Model-free regression checks for amendment 2's post-hoc inference boundary.

Run: python -m unittest discover -s analysis/ju_space/round1b/tests -p test_htq_reporting.py
"""
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import r1b_report as RP
import r1b_k_ana as KA
import r1b_l_ana as LA
from r1b_htq_variant import LABEL, VARIANT


class Context(SimpleNamespace):
    def __init__(self, variant=False):
        super().__init__(manifest={"h_policy": VARIANT} if variant else {}, mock=False, freeze={}, outputs={})

    def path(self, rel):
        return rel

    def write_json(self, rel, obj, **kw):
        self.outputs[rel] = {"data": copy.deepcopy(obj), **kw}


def planted_inputs():
    """Tiny p-values would confirm every reached family unless post-hoc exclusion applies."""
    p = 1e-8
    return {
        "R": {"F_plus": [{"member": "R-positive", "reached": True, "p": p, "label": "confirmatory (Holm)"}],
              "F_minus": [{"member": "R-negative", "reached": True, "p": p, "label": "confirmatory (Holm)"}]},
        "D": {"parts": {"a": {"support_condition": True, "p": p, "holm_reject": True, "verdict": "D-positive"}}},
        "K": {"families": {"Q9": {"K-positive": {"string": "H-neuron attractor", "p": p}},
                             "A_plus": {"K-link": {"reached": True, "p_pi_form": p, "p_rank_form": p}}},
              "round_level": {"CF_HERR_K8c": p},
              "strings": {"K8c": "H-activity predicts the next error beyond length"},
              "composite": {"string": "no H association with lens break-off (dissipative)", "suffixes": []}},
        "L": {"verdicts": {"attractor": RP.ATTR_L, "L2b": "H rises beyond information loss",
                            "L5c": "H predicts the model's own step errors beyond uncertainty"},
              "combined": {"CF-L": {"p": {"attractor": p, "l2b": p},
                                     "confirmed": {"attractor": True, "l2b": True}},
                           "CF-HERR_member_L5c": {"p": p},
                           "L-ALL": "no length trend and no H-attractor", "answer": "descriptive L answer"}},
    }


def compute_families(ctx, inputs):
    with patch.object(RP, "_wait_l_ana", return_value="ok"), patch.object(RP, "_inputs", return_value=inputs):
        RP.families(ctx)
    return ctx.outputs["ana/families.json"]["data"]


class PosthocFamiliesTests(unittest.TestCase):
    def test_tiny_pvalues_are_retained_but_cannot_confirm_htq_families(self):
        normal = compute_families(Context(), planted_inputs())["families"]
        ctx = Context(True)
        post = compute_families(ctx, planted_inputs())["families"]
        for name in ("CF-R+", "CF-R-", "CF-D"):
            self.assertEqual(normal[name], post[name])
        for name in RP.HTQ_EXCLUDED:
            self.assertGreater(normal[name]["confirmed"], 0)
            self.assertEqual(post[name]["confirmed"], 0)
            self.assertFalse(post[name]["confirmatory_eligible"])
            for before, after in zip(normal[name]["members"], post[name]["members"]):
                self.assertEqual(before["p"], after["p"])
                self.assertEqual(before["p_holm"], after["p_holm"])
                self.assertEqual(before["reached"], after["reached"])
                self.assertFalse(after["confirmatory_eligible"])
                self.assertTrue(after["excluded"])
                self.assertTrue(after["label"].startswith("exploratory (post hoc"))
        self.assertIn(LABEL, ctx.outputs["ana/families.json"]["labels"])

    def test_labelled_inputs_fail_closed_even_without_runtime_policy(self):
        inputs = planted_inputs()
        inputs["K"]["h_policy"] = VARIANT
        fams = compute_families(Context(), inputs)["families"]
        self.assertEqual(fams["CF-K-attr"]["confirmed"], 0)
        self.assertEqual(fams["CF-R+"]["confirmed"], 1)

    def test_summary_excludes_posthoc_for_and_against_claims(self):
        ctx, inputs = Context(True), planted_inputs()
        families = compute_families(ctx, inputs)
        with patch.object(RP, "load_json", return_value=families), patch.object(RP, "_inputs", return_value=inputs), \
             patch.object(RP, "_ld", return_value=None), patch.object(RP, "manifest", return_value=({}, {}, {})):
            RP.report(ctx)
        summary = ctx.outputs[RP.C.FILES["summary"]["rel"]]["data"]
        self.assertEqual(summary["for_families"]["with_confirmed_member"], ["CF-R+", "CF-D"])
        self.assertEqual(summary["for_families"]["n"], 2)
        self.assertEqual(summary["for_families"]["of"], 2)
        self.assertIn("0.10", summary["for_families"]["null_bound"])
        self.assertFalse(summary["confirmatory_against"]["K7"])
        self.assertFalse(summary["confirmatory_against"]["L_ALL"])
        self.assertEqual(summary["confirmatory_against"]["CF_R_minus"], ["R-negative"])
        self.assertEqual(summary["confirmatory_against"]["descriptive_posthoc"], {"K7": True, "L_ALL": True})
        self.assertEqual(summary["h_policy"], VARIANT)
        for part in ("K", "L"):
            self.assertIn(LABEL, summary["answers"][part]["labels"])

    def test_frozen_summary_retains_six_family_bound(self):
        ctx, inputs = Context(), planted_inputs()
        families = compute_families(ctx, inputs)
        with patch.object(RP, "load_json", return_value=families), patch.object(RP, "_inputs", return_value=inputs), \
             patch.object(RP, "_ld", return_value=None), patch.object(RP, "manifest", return_value=({}, {}, {})):
            RP.report(ctx)
        summary = ctx.outputs[RP.C.FILES["summary"]["rel"]]["data"]
        self.assertEqual(summary["for_families"]["of"], 6)
        self.assertIn("0.30", summary["for_families"]["null_bound"])
        self.assertNotIn("h_policy", summary)
        self.assertTrue(summary["confirmatory_against"]["K7"])


class PartVerdictTests(unittest.TestCase):
    def test_k_strings_and_pvalues_survive_without_confirmatory_suffix(self):
        original = {"labels": [], "composite": {"string": "original rule string",
                    "suffixes": ["model side: M+", "round level: A+ confirmatory; B+ confirmatory"]},
                    "round_level": {"CF_K_attr": {"member": {"p": 1e-8, "reject": True}}},
                    "families": {"Q9": {"member": {"p": 1e-8, "reject": True}}}}
        self.assertEqual(KA._htq_verdict(Context(), copy.deepcopy(original)), original)
        result = KA._htq_verdict(Context(True), copy.deepcopy(original))
        self.assertEqual(result["composite"]["string"], original["composite"]["string"])
        self.assertIn(LABEL, result["composite"]["labels"])
        self.assertFalse(any("A+ confirmatory" in x for x in result["composite"]["suffixes"]))
        self.assertEqual(result["round_level"]["CF_K_attr"]["member"]["p"], 1e-8)
        self.assertFalse(result["round_level"]["CF_K_attr"]["member"]["confirmatory_eligible"])

    def test_l_preserves_descriptive_holm_result_and_rule_strings(self):
        original = {"verdicts": {"attractor": {"verdict": "original rule string", "labels": []}},
                    "combined": {"CF-L": {"p": {"attractor": 1e-8}, "confirmed": {"attractor": True}}}}
        labels = []
        self.assertEqual(LA._htq_verdict(Context(), copy.deepcopy(original), labels), original)
        self.assertEqual(labels, [])
        result = LA._htq_verdict(Context(True), copy.deepcopy(original), labels)
        self.assertEqual(result["verdicts"]["attractor"]["verdict"], "original rule string")
        self.assertFalse(result["combined"]["CF-L"]["confirmed"]["attractor"])
        self.assertTrue(result["combined"]["CF-L"]["descriptive_holm_confirmed"]["attractor"])
        self.assertEqual(result["combined"]["CF-L"]["p"]["attractor"], 1e-8)
        self.assertIn(LABEL, labels)

    def test_tier2_labels_k_outputs_without_changing_r_or_d(self):
        ctx = Context(True)
        with patch.object(RP, "_ld", return_value={"numeric": 2.0}):
            RP.tier2(ctx)
        out = ctx.outputs[RP.C.FILES["summary_t2"]["rel"]]
        self.assertEqual(out["data"]["h_policy"], VARIANT)
        for part in ("R", "D"):
            self.assertNotIn(LABEL, out["data"]["next_to"][part]["tier2"]["labels"])
        for key in ("RK1", "RK2"):
            self.assertIn(LABEL, out["data"]["next_to"]["K"][key]["labels"])
        self.assertIn(LABEL, out["labels"])


if __name__ == "__main__":
    unittest.main()
