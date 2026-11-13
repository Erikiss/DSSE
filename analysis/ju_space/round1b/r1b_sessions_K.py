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


S1 = "r1b_k_s1:"
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
}
