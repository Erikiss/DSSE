"""Offline native-model tests for G's independent D1 and 12B bridge stages."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

import r1b_g_bridge as bridge
import r1b_g_d1 as d1
from r1b_g_adapter import load_model


class Context:
    def __init__(self, root, data, stage, model_name="gemma4-12b-qat", model=None):
        self.root = Path(root)
        self.work = self.root / "stages" / stage
        self.work.mkdir(parents=True, exist_ok=True)
        self.mock, self.freeze, self.model_name = True, {}, model_name
        self._data, self._model, self._cache = data, model, {}

    def model(self):
        if self._model is None:
            self._model = load_model(self.model_name, mock=True)
        return self._model

    def data(self):
        return self._data

    def artifact(self, stage, rel):
        return self.root / "stages" / stage / rel

    def write_json(self, rel, obj):
        path = self.work / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, allow_nan=False))

    def cached(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def time_left(self):
        return 9999


def item(i=0, gold=0):
    return {"id": f"item-{i}", "context": "The code is red.",
            "control_context": "The code is blue.", "question": "Code?",
            "choices": ["red", "blue"], "gold": gold}


def probes():
    return [{"id": f"probe-{pole}-{active}", "layer": 0, "neuron": 1,
             "pole": pole, "split": "heldout", "expected_active": active,
             "text": "red" if active else "blue", "expected_token": "a"}
            for pole in (-1, 1) for active in (False, True)]


def bridge_fixture(root, include_probes=True):
    root = Path(root)
    path = root / "catalogue.json"
    path.write_text(json.dumps({"mock": True, "rows": [
        {"layer": 0, "neuron": n, "pole": p, "label": "synthetic fixture"}
        for n in (1, 2) for p in (-1, 1)]}))
    hn = root / "stages/k_prepare/hn.json"
    hn.parent.mkdir(parents=True, exist_ok=True)
    hn.write_text(json.dumps({"mock": True, "H_op": [[0, 1]], "RS39": [[[0, 2]]]}))
    data = {"neuron_catalogue": {"mock": True, "path": str(path)}}
    if include_probes:
        data["bridge_probes"] = probes()
    return data


class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.unified = load_model("gemma4-12b-qat", mock=True)

    def test_d1_exact_gap_and_unexecutable_window(self):
        tok, _ = self.unified
        row = item()
        ids, meta = d1.prepare_ids(tok, row, row["context"], 160, "neutral ")
        self.assertEqual(meta["actual_gap"], 160)
        self.assertEqual(meta["read_position"] - meta["cue_end"], 160)
        self.assertEqual(meta["n_tokens"], len(ids))
        ids, meta = d1.prepare_ids(tok, row, row["context"], 4, "neutral ")
        self.assertIsNone(ids)
        self.assertGreater(meta["minimum_gap"], 4)

    def test_d1_rejects_duplicate_or_identical_arms(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            d1.validate_items([item(), item()])
        row = item(); row["control_context"] = row["context"]
        with self.assertRaisesRegex(ValueError, "differ"):
            d1.validate_items([row])

    def test_d1_bootstraps_items_and_orients_letter_swaps(self):
        records = []
        for i, gold in enumerate((1, 0)):
            sign = 2 * gold - 1
            records.append({"id": str(i), "cluster_id": "shared", "gold": gold,
                            "cue": {"layer_logodds": [sign, 3 * sign]},
                            "control": {"layer_logodds": [0, 0]}})
        result = d1.summarize(records, ["sliding_attention", "full_attention"])
        self.assertEqual(result["n_items"], 1)
        self.assertEqual(result["mean_global_minus_local_cue_effect"], 2)
        self.assertIn("code", result["interval90"])

    def test_d1_raw_package_preserves_questions_and_swaps(self):
        repo = Path(__file__).resolve().parents[3]
        packet = json.loads((repo / "results/ju_space/plans/runde1b/packages/d1_stimuli.json").read_text())
        packet = {**packet, "fsym": packet["fsym"][:1], "fasym": packet["fasym"][:1]}
        rows = d1.make_inputs(packet)
        self.assertEqual(len(rows), 2 * (len(packet["fsym"][0]["slots"]) + len(packet["fasym"][0]["slots"])))
        self.assertEqual(rows[0]["cluster_id"], rows[1]["cluster_id"])
        self.assertEqual(rows[0]["choices"], rows[1]["choices"][::-1])
        self.assertEqual((rows[0]["gold"], rows[1]["gold"]), (0, 1))

    def test_d1_native_cache_and_resumable_records(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = Context(td, {"d1": [item(), item(1)]}, "d1", model=self.unified)
            result = d1.run(ctx)
            self.assertEqual(result["status"], "completed", result.get("reason"))
            self.assertTrue(result["mock"])
            self.assertTrue(result["gates"]["G-CACHE"]["passed"])
            self.assertEqual(len(result["records"]), 2)
            self.assertEqual(result["summary"]["n_items"], 2)
            self.assertTrue(all(r["mock_extra_gap"] for r in result["records"]))
            self.assertEqual(d1.run(ctx), result)
            self.assertTrue((ctx.work / "d1.json").is_file())

    def test_bridge_controls_are_layer_matched_and_exclude_h(self):
        h, controls = bridge.matched_coordinates({"H_op": [[1, 0], [0, 2]],
                                                  "RS39": [[[0, 1], [1, 3]]]}, [4, 8], True)
        self.assertEqual(h, [(0, 2), (1, 0)])
        self.assertEqual(controls[0][(1, 0)], (1, 3))
        with self.assertRaisesRegex(ValueError, "exclude H"):
            bridge.matched_coordinates({"H_op": [[0, 2]], "RS39": [[[0, 2]]]}, [4], True)
        with self.assertRaisesRegex(ValueError, "layer counts"):
            bridge.matched_coordinates({"H_op": [[0, 2]], "RS39": [[[1, 2]]]}, [4, 4], True)
        with self.assertRaisesRegex(ValueError, "39"):
            bridge.matched_coordinates({"H_op": [[0, 2]], "RS39": [[[0, 1]]]}, [4])

    def test_bridge_catalogue_is_explicitly_mock_and_reports_missing_rows(self):
        with tempfile.TemporaryDirectory() as td:
            data = bridge_fixture(td)
            rows, meta = bridge.load_catalogue(data["neuron_catalogue"], {(0, 1), (0, 3)}, mock=True)
            self.assertEqual(len(rows), 2)
            self.assertEqual(meta["missing"], [[0, 3, -1], [0, 3, 1]])
            with self.assertRaisesRegex(ValueError, "mock/real"):
                bridge.load_catalogue(data["neuron_catalogue"], {(0, 1)}, mock=False)

    def test_bridge_probe_split_and_full_pole_coverage(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            bridge.validate_probes(probes(), [(0, 1)], [probes()[0]["id"]])
        with self.assertRaisesRegex(ValueError, "both poles"):
            bridge.validate_probes(probes()[:2], [(0, 1)])
        with self.assertRaisesRegex(ValueError, "positive and negative"):
            bridge.validate_probes(probes()[1:], [(0, 1)])

    def test_bridge_real_probe_provenance_and_text_overlap(self):
        with self.assertRaisesRegex(ValueError, "bridge_probe_manifest"):
            bridge.probe_provenance({}, probes())
        manifest = {"source": "independent frozen test corpus", "frozen_before_evaluation": True,
                    "label_generation_contexts_excluded": True, "discovery_text_sha256": []}
        data = {"bridge_probe_manifest": manifest}
        self.assertTrue(bridge.probe_provenance(data, probes())["passed"])
        manifest["discovery_text_sha256"] = [bridge.hashlib.sha256(b"red").hexdigest()]
        with self.assertRaisesRegex(ValueError, "overlaps"):
            bridge.probe_provenance(data, probes())

    def test_bridge_no_probes_is_blocked_after_real_join(self):
        with tempfile.TemporaryDirectory() as td:
            data = bridge_fixture(td, include_probes=False)
            result = bridge.run(Context(td, data, "bridge", model=self.unified))
            self.assertEqual(result["status"], "blocked")
            self.assertIn("bridge_probes", result["reason"])
            self.assertEqual(len(result["joined"]), 4)
            self.assertEqual(result["annotation_label"], bridge.ANNOTATION_LABEL)
            self.assertNotIn("summary", result)

    def test_bridge_native_quantized_forward_checks(self):
        with tempfile.TemporaryDirectory() as td:
            data = bridge_fixture(td)
            ctx = Context(td, data, "bridge", model=self.unified)
            result = bridge.run(ctx)
            self.assertEqual(result["status"], "completed", result.get("reason"))
            self.assertTrue(result["mock"])
            self.assertFalse(result["semantic_equivalence_claim"])
            self.assertEqual(len(result["records"]), 4)
            self.assertEqual(len(result["summary"]), 2)
            self.assertTrue(all(len(r["controls"]) == 1 for r in result["records"]))
            self.assertTrue(all(len(r["target"]["user_positions"]) == len("red" if r["expected_active"] else "blue")
                                for r in result["records"]))
            self.assertEqual(result["checks"]["activation"]["status"], "measured")
            self.assertEqual(d1._hash(bridge.run(ctx)), d1._hash(result))
            self.assertTrue((ctx.work / "bridge.json").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
