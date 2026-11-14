"""Regression checks for the G DUET descriptive kinetic readouts."""
import unittest

import numpy as np
import torch

from r1b_g_l_duet import (_design, descriptive_kinetics, j_ex,
                          kinetic_descriptor, kinetic_series)

torch.set_num_threads(1)


class DuetStatisticsTests(unittest.TestCase):
    def test_partner_alignment_uses_latest_message_in_each_direction(self):
        # Distinct markers make a one-exchange lag observable independently of
        # the statistical fit: A3 hears B2, while B4 hears A3.
        messages = np.array([[11, 12, 13, 14, 15, 16]])
        for direction, own_previous, partner_latest, prime in (
                ("A_from_B", [11, 13], [12, 14], 0),
                ("B_from_A", [12, 14], [13, 15], 1)):
            series = kinetic_series(messages, "C-half", np.array([1]), np.array([0]), direction)
            _, _, design, rows, turns = _design(series)
            np.testing.assert_array_equal(design[:, 1], own_previous)
            np.testing.assert_array_equal(series[1][rows, turns - 1], partner_latest)
            np.testing.assert_array_equal(design[:, 0], prime)
        document = kinetic_series(np.array([[21, 23, 25]]), "C-doc", np.array([1]), np.array([0]),
                                  "A_from_replay", np.array([[12, 14, 16]]))
        _, _, _, rows, turns = _design(document)
        np.testing.assert_array_equal(document[1][rows, turns - 1], [14, 16])

    def test_seed_null_removes_config_determined_partner_even_when_raw_coupling_positive(self):
        rng = np.random.default_rng(490)
        item = np.repeat(np.arange(8), 8)
        cfgs = np.tile(np.repeat([1, 2], 4), 8)
        partner = np.repeat(np.where(cfgs == 1, 1., -1.)[:, None], 6, axis=1)
        listener = np.where(rng.random(partner.shape) < .9, partner, -partner)
        series = (listener, partner, np.zeros(len(item)), item)
        groups = np.unique(np.column_stack([item, cfgs]), axis=0, return_inverse=True)[1]
        seed_excess, raw = j_ex(series, groups, 20, np.random.default_rng(10))
        primary_excess, _ = j_ex(series, item, 20, np.random.default_rng(10))
        self.assertGreater(raw, .5)
        self.assertAlmostEqual(seed_excess, 0., places=8)
        self.assertGreater(primary_excess, .4)

    def test_descriptive_table_has_both_live_directions_and_only_document_listener(self):
        rng = np.random.default_rng(148)
        item = np.repeat(np.arange(4), 8)
        cfgs = np.tile(np.repeat([1, 2], 4), 4)
        messages = {c: rng.choice([-1., 1.], (len(item), 12))
                    for c in ("C-opp", "C-half", "C-Hup", "C-Rup")}
        messages["C-doc"] = rng.choice([-1., 1.], (len(item), 6))
        fits, seed = descriptive_kinetics(messages, cfgs, item, 6)
        for cond in messages:
            expected = {"A_from_replay"} if cond == "C-doc" else {"A_from_B", "B_from_A"}
            self.assertEqual(set(fits[cond]), expected)
            self.assertEqual(set(seed[cond]), expected)
            for direction in expected:
                self.assertEqual(fits[cond][direction]["n_updates"], len(item) * 5)
                self.assertEqual(seed[cond][direction]["n_repair"], 6)
                self.assertAlmostEqual(seed[cond][direction]["J_ex"],
                                       seed[cond][direction]["J_c"] - seed[cond][direction]["mean_repaired_J_c"])

    def test_reported_ising_parameters_reconstruct_fitted_log_likelihood(self):
        rng = np.random.default_rng(153)
        item = np.repeat(np.arange(4), 16)
        cfgs = np.tile(np.repeat([1, 2], 8), 4)
        messages = rng.choice([-1., 1.], (len(item), 12))
        series = kinetic_series(messages, "C-opp", cfgs, item, "B_from_A")
        fit = kinetic_descriptor(series)
        y, fe, x, rows, turns = _design(series)
        logits = 2 * (fit["h0"] + fit["h_item"][fe] + fit["h_prime"] * x[:, 0]
                      + fit["J_s"] * x[:, 1] + fit["J_c"] * series[1][rows, turns - 1])
        ll = (y * logits - np.logaddexp(0, logits)).mean()
        self.assertAlmostEqual(ll, fit["mean_log_likelihood"], places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
