"""Part K: sessions and steps (main §4.2; r1b_contracts.SESSIONS): r1b_K_S1, _S3, _S2a, _S2b, _S4 (account B), r1b_K_S5, r1b_K_ana
(account A); tier 2: r1b_K_S6 (A: RK-1, two heads), r1b_K_S7 (B: RK-2); K0 pilot items in r1b_P0; K3h (atlas-compatible H-neuron
metrics) is a tier-2 item in r1b_K_S2b. Long work is split into part steps (<= 5 min each, SG5); every step writes its outputs with
provenance (r1b_core.Ctx.write_json / save_pt). Code: r1b_k_base, r1b_k_labels (K1), r1b_k_hn (SO-HN), r1b_k_lens (SO-CL),
r1b_k_gvra (SO-GVRA), r1b_k_te (SO-TE), r1b_k_s1 ... r1b_k_s5, r1b_k_rk, r1b_k_rules, r1b_k_ana."""
from r1b_contracts import Step


def L(fn, **kw):
    from r1b_k_base import lazy
    return lazy(fn, **kw)


def parts(name, fn, n, minutes, out_dir, after="", item=None, drop_if=()):
    return [Step(f"{name}_{i}", fn=L(fn, i=i, n=n), out=f"k/{out_dir}/part{i}.pt", after=after, minutes=minutes, item=item, drop_if=drop_if)
            for i in range(n)]


S1, S2, S2B, S3, S4, S5 = "r1b_k_s1:", "r1b_k_s2:", "r1b_k_s2b:", "r1b_k_s3:", "r1b_k_s4:", "r1b_k_s5:"
ANA, RK = "r1b_k_ana:", "r1b_k_rk:"
STEPS = {
    "r1b_P0": [Step("k0_pilot", fn=L(S1 + "k0_pilot"), out="k0_pilot.json", minutes=10)],
    "r1b_K_S1": [
        Step("k1_check", fn=L(S1 + "k1_check"), out="k/k1_check.json", minutes=0.5),
        Step("k4j", fn=L(S1 + "k4j"), out="k4j.json", after="k1_check", minutes=3.1),
        Step("k2_refs", fn=L(S1 + "k2_refs"), out="k/refs.pt", after="k1_check", minutes=2.6),
        *[Step(f"k2_screen_{i:02d}", fn=L(S1 + "k2_screen", i=i), out=f"k/k2/screen{i}.pt", after="k1_check", minutes=1.3) for i in range(15)],
        Step("k2_alloc", fn=L(S1 + "k2_alloc"), out="lab/so_lab.json", after=",".join(f"k2_screen_{i:02d}" for i in range(15)), minutes=0.2),
        Step("k2_path", fn=L(S1 + "k2_path"), out="k/k2/path.pt", after="k2_alloc", minutes=4.5),
        Step("k2_supp", fn=L(S1 + "k2_supp"), out="k/k2/cstar.pt", after="k2_path", minutes=4.0),
        Step("k2_diag", fn=L(S1 + "k2_diag"), out="k/k2/h_tq.pt", after="k2_supp", minutes=3.0),
        Step("k2_har", fn=L(S1 + "k2_har"), out="k/k2/h_ar.pt", after="k1_check", minutes=4.0),
        Step("k2_sets", fn=L(S1 + "k2_sets"), out="hn/hn_set.json hn/rs_sets.json hn/hn_detectors.pt hn/hn_meta.json",
             after="k2_refs,k2_diag,k2_har", minutes=4.7),
        Step("k3a", fn=L(S1 + "k3a"), out="k/k3/k3a_data.pt sg8_preview.json", after="k2_sets", minutes=1.9),
        Step("k3g2", fn=L(S1 + "k3g2"), out="k/k3/k3g2.pt", after="k3a", minutes=0.5),
        *parts("k3d", S1 + "k3d_part", 8, 3.4, "k3d", after="k2_sets"),
        Step("k3e", fn=L(S1 + "k3e"), out="k/k3/k3e.json", after="k2_sets,k4j", minutes=5.0),
    ],
    "r1b_K_S3": [
        Step("k5a_load", fn=L(S3 + "k5a_load"), out="k/k5a/load.json", minutes=1.0),
        Step("k5a_wiring", fn=L(S3 + "k5a_wiring"), out="k/k5a/wiring.json", after="k5a_load", minutes=1.5),
        Step("k5a_gate", fn=L(S3 + "k5a_gate"), out="k/k5a/states.pt", after="k5a_load", minutes=3.0),
        Step("k5a_run", fn=L(S3 + "k5a_run"), out="k/k5a/chains.pt k5a_calib.json", after="k5a_wiring,k5a_gate", minutes=5.0),
    ],
    "r1b_K_S2a": [
        Step("k4_gate", fn=L(S2 + "k4_gate"), out="lab/k4_gating.json k/k4/items.pt", minutes=1.0),
        Step("k4_states", fn=L(S2 + "k4_states"), out="k/k4/states.pt", after="k4_gate", minutes=2.0),
        *parts("k4_run", S2 + "k4_run", 6, 4.0, "k4", after="k4_states"),
        Step("k4_merge", fn=L(S2 + "k4_merge", n=6), out="k/k4/summary.pt k/k4/b_star.json lensout/so_lensout.jsonl",
             after=",".join(f"k4_run_{i}" for i in range(6)), minutes=4.5),
        Step("k4p", fn=L(S2 + "k4p"), out="k/k4p.pt", after="k4_states", minutes=1.2),
        *parts("k6c", S2 + "k6c_part", 4, 4.2, "k6c", after="k4_states"),
        *parts("k6f", S2 + "k6f_part", 2, 3.1, "k6f", after="k4_merge"),
        Step("k6g", fn=L(S2 + "k6g"), out="k/k6g.pt", after="k4_merge", minutes=1.1),
        *parts("k6h", S2 + "k6h_part", 2, 1.9, "k6h", after="k4_states"),
        Step("k10c", fn=L(S2 + "k10c"), out="k/k10c.pt", after="k4_merge", minutes=2.2),
        Step("kref_u32", fn=L(S2 + "kref_u32"), out="k/ref_u32_S1.json", after="k4_states", minutes=3.0),
    ],
    "r1b_K_S2b": [
        Step("k6e_cal", fn=L(S2B + "k6e_cal"), out="k/k6e/cal.json", minutes=2.0),
        Step("k6e_twin", fn=L(S2B + "k6e_twin"), out="k/k6e/twin.pt", after="k6e_cal", minutes=3.5),
        *parts("k6e", S2B + "k6e_part", 8, 4.0, "k6e", after="k6e_twin"),
        Step("k6e_offset", fn=L(S2B + "k6e_offset"), out="k/k6e/offset.pt", after="k6e_cal", minutes=3.4),
        *parts("k3h", "r1b_k_k3h:k3h_part", 4, 4.5, "k3h", item="K3h"),
        Step("k3h_ana", fn=L("r1b_k_k3h:k3h_ana", n=4), out="k/k3h.json", after=",".join(f"k3h_{i}" for i in range(4)), minutes=1.0, item="K3h"),
    ],
    "r1b_K_S4": [
        Step("k8a_bnd", fn=L(S4 + "k8a_bnd"), out="k/k8/bnd.pt", minutes=1.6),
        *parts("k8a", S4 + "k8a_run", 4, 3.0, "k8a", after="k8a_bnd"),
        Step("k8b_cal", fn=L(S4 + "k8b_cal"), out="k/k8b/cal.json", after="k8a_bnd", minutes=0.8),
        *parts("k8b", S4 + "k8b_part", 12, 4.6, "k8b", after="k8b_cal", drop_if=()),
        *parts("k8c", S4 + "k8c_part", 6, 1.0, "k8c", after="k8a_bnd"),
        Step("k8c_reach", fn=L(S4 + "k8c_reach", n=6), out="k/k8c/reach.pt", after=",".join(f"k8c_{i}" for i in range(6)), minutes=2.4),
        Step("k7a", fn=L(S4 + "k7a"), out="k/k7a.json", after=",".join(f"k8c_{i}" for i in range(6)), minutes=3.9),
    ],
    "r1b_K_S5": [
        Step("ks_axes", fn=L(S5 + "ks_axes"), out="k/ax.pt", minutes=1.3),
        Step("k3f", fn=L(S5 + "k3f"), out="k/k3f.json", after="ks_axes", minutes=1.5),
        Step("k7b", fn=L(S5 + "k7b"), out="k/k7b.json", after="ks_axes", minutes=4.6, drop_if=("K-drop1",)),
        *parts("k9", S5 + "k9_part", 6, 2.9, "k9", after="ks_axes"),
        Step("k10a_texts", fn=L(S5 + "k10a_texts"), out="k/k10a/texts.pt", after="ks_axes", minutes=0.8),
        Step("k10a_cal", fn=L(S5 + "k10a_cal"), out="k/k10a/cal.pt", after="k10a_texts", minutes=1.5),
        *parts("k10a", S5 + "k10a_part", 3, 3.4, "k10a", after="k10a_cal"),
        Step("k10b", fn=L(S5 + "k10b"), out="k/k10b.json", after="k10a_texts", minutes=3.7),
    ],
    "r1b_K_ana": [
        Step("k_ana_1", fn=L(ANA + "ana_part1"), out="k/ana1.json", minutes=4.0),
        Step("k_ana_2", fn=L(ANA + "ana_part2"), out="k/ana2.json", minutes=4.5),
        Step("k_ana_3", fn=L(ANA + "ana_part3"), out="k/ana3.json", minutes=4.0),
        Step("k_verdict", fn=L(ANA + "verdict"), out="k_verdict.json", after="k_ana_1,k_ana_2,k_ana_3", minutes=1.0),
    ],
    "r1b_K_S6": [
        *[Step(f"rk1_train_{i:02d}", fn=L(RK + "rk1_train", i=i, n=12), out=f"k/rk1/ckpt{i}.pt", minutes=4.5) for i in range(12)],
        Step("rk1_eval", fn=L(RK + "rk1_eval", n=12), out="k/rk1.json", after="rk1_train_11", minutes=4.0),
    ],
    "r1b_K_S7": [
        Step("rk2_model", fn=L(RK + "rk2_model"), out="k/rk2/model.json", minutes=0.5),
        Step("k2_refs", fn=L(S1 + "k2_refs"), out="k/refs.pt", after="rk2_model", minutes=2.0),
        *[Step(f"k2_screen_{i:02d}", fn=L(S1 + "k2_screen", i=i), out=f"k/k2/screen{i}.pt", after="rk2_model", minutes=2.0) for i in range(6)],
        Step("k2_alloc", fn=L(S1 + "k2_alloc"), out="lab/so_lab.json", after=",".join(f"k2_screen_{i:02d}" for i in range(6)), minutes=0.2),
        Step("k2_path", fn=L(S1 + "k2_path"), out="k/k2/path.pt", after="k2_alloc", minutes=4.0),
        Step("k2_supp", fn=L(S1 + "k2_supp"), out="k/k2/cstar.pt", after="k2_path", minutes=4.5),
        Step("k2_diag", fn=L(S1 + "k2_diag"), out="k/k2/h_tq.pt", after="k2_supp", minutes=3.0),
        Step("k2_har", fn=L(RK + "rk2_no_har"), out="k/k2/h_ar.pt", after="rk2_model", minutes=0.1),
        Step("k2_sets", fn=L(S1 + "k2_sets"), out="hn/hn_set.json hn/rs_sets.json hn/hn_detectors.pt hn/hn_meta.json",
             after="k2_refs,k2_diag,k2_har", minutes=4.5),
        Step("k3a", fn=L(S1 + "k3a"), out="k/k3/k3a_data.pt sg8_preview.json", after="k2_sets", minutes=2.0),
        Step("rk2_run", fn=L("r1b_k_s3:rk2_run"), out="k/rk2/rows.pt k/rk2/rk2.json", after="k3a", minutes=4.5),
    ],
}
