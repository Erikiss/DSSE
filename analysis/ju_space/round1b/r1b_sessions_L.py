"""Part L: sessions and steps (main §4.2): r1b_L_A and r1b_L_B (account B), L-ANA as a step of r1b_ana (account A, CPU), the P0-L
pilot items in r1b_P0. Tier-2 items X1-X6 are steps with item='X..' (run only if funded). Modules: r1b_l_base (inputs, populations),
r1b_l_engine (probe, decoding, reads), r1b_l_stats (estimators), r1b_l_rules (verdict rules), r1b_l_a / r1b_l_duet (r1b_L_A),
r1b_l_b / r1b_l_b2 (r1b_L_B), r1b_l_ana (L-ANA), r1b_l_cal (L-CAL, CPU before the freeze), r1b_l_pilot (L0). Inputs are read in the
contract formats; under MOCK=1 a missing or placeholder input is replaced by a labelled stand-in (code tests only)."""
from r1b_contracts import Step

A, D, B, B2, AN, PI = "r1b_l_a:", "r1b_l_duet:", "r1b_l_b:", "r1b_l_b2:", "r1b_l_ana:", "r1b_l_pilot:"


def _s(name, mod, out, after="", minutes=4.0, item=None, drop_if=()):
    return Step(name, fn=mod + name, out=out, after=after, minutes=minutes, item=item, drop_if=drop_if)


STEPS = {
    "r1b_P0": [_s("l0_a", PI, "pilot/l0_a.json", minutes=5), _s("l0_b", PI, "pilot/l0_b.json", minutes=5)],
    "r1b_L_A": [
        # step 1: L.POP
        _s("pop_refw", A, "l/refw.pt", minutes=1),
        _s("pop_cett", A, "l/cett_rows.pt", "pop_refw", 3),
        _s("pop_select", A, "l/l_pop.json", "pop_cett", 2),
        # L1 LOCUS
        _s("l1_pip", A, "l/l1_pip.pt", "pop_select", 5),
        _s("l1_rot", A, "l/l1_rot.pt", "pop_select", 4),
        _s("l1_resel", A, "l/l1_resel.pt", "pop_select", 3),
        _s("l1_cal", A, "l/l1_cal.pt", "l1_pip", 5),
        _s("l1_era_cal", A, "l/l1_era_cal.json", "pop_select", 2),
        _s("l1_era", A, "l/l1_era.pt", "l1_era_cal", 5),
        _s("l1_stats", A, "l/l1.json", "l1_pip,l1_rot,l1_resel,l1_cal,l1_era", 2),
        # L3b causal twins
        _s("l3b_cal", A, "l/l3b_cal.json", "pop_select", 2),
        _s("l3b_gen", A, "l/l3b_gen.pt", "l3b_cal", 5),
        _s("l3b_reads", A, "l/l3b_reads.pt", "l3b_gen", 5),
        _s("l3b_stats", A, "l/l3b.json l/l3b_twin0.json l/l3b_twin0.pt", "l3b_reads", 2),
        # L4 DUET (X2, X6: tier 2)
        _s("l4_cal", D, "l/l4_cal.json", "pop_select", 1),
        _s("l4_opp", D, "l/l4_C-opp.pt", "l4_cal", 5),
        _s("l4_half", D, "l/l4_C-half.pt", "l4_cal", 5),
        _s("l4_doc", D, "l/l4_C-doc.pt", "l4_half", 2),
        _s("l4_hup", D, "l/l4_C-Hup.pt", "l4_cal", 5),
        _s("l4_rup", D, "l/l4_C-Rup.pt", "l4_cal", 5),
        _s("x2_hdown", D, "l/l4_C-Hdown.pt", "l4_cal", 5, item="X2"),
        _s("x2_rdown", D, "l/l4_C-Rdown.pt", "l4_cal", 5, item="X2"),
        _s("x6_solo", D, "l/l4_C-solo.pt", "l4_cal", 2, item="X6"),
        _s("l4_reads", D, "l/l4_reads.pt", "l4_opp,l4_half,l4_doc,l4_hup,l4_rup", 5),
        _s("l4_stats", D, "l/l4.json", "l4_reads", 5),
        _s("session_A", A, "l_session_A.json", "l1_stats,l3b_stats,l4_stats", 0.2),
    ],
    "r1b_L_B": [
        # corpus L.R
        _s("lr_fneu", B, "l/lr_thr.pt l/lr/fneu.pt", minutes=5),
        _s("lr_gen", B, "l/lr/gen.pt", "lr_fneu", 5),
        _s("lr_tf", B, "l/lr_tf.pt", "lr_gen", 5),
        # L2
        _s("l2a_lce", B, "l/l2a_lce.json", minutes=2),
        _s("l2a_main", B, "l/l2a_main.pt", "l2a_lce", 5),
        _s("l2a_own", B, "l/l2a_own.pt", "l2a_lce", 5),
        _s("l2a_stats", B, "l/l2a.json", "l2a_main,l2a_own", 1),
        _s("l2b", B, "l/l2b.json", "lr_tf", 3),
        _s("l2c", B, "l/l2c.json", "lr_gen", 3),
        # L3
        _s("l3a", B2, "l/l3a.json", "lr_gen", 5),
        _s("l3c", B2, "l/l3c.json", "lr_gen", 5),
        _s("l3d", B2, "l/l3d.json", "lr_gen", 3),
        # L5 generation
        _s("l5_gen", B, "l/l5_gen.pt", "lr_fneu", 4),
        # tier 2
        _s("x1", B2, "l/x1.json", "lr_gen", 5, item="X1"),
        _s("x3", B2, "l/x3.json", minutes=5, item="X3"),
        _s("x4", B2, "l/x4.json", "l3c", 3, item="X4"),
        _s("x5", B2, "l/x5.json", "l3c", 3, item="X5"),
        _s("session_B", B2, "l_session_B.json", "l2a_stats,l2b,l2c,l3a,l3c,l3d,l5_gen", 0.2),
    ],
    "r1b_ana": [_s("l_ana", AN, "l_verdicts.json", minutes=5)],
}
