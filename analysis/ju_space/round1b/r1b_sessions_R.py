"""Part R: sessions and steps (main §4.2; r1b_contracts.SESSIONS: r1b_R_S0, _S1, _S2_14, _S2_18, _S2_06, _S3, _S5, _S6, _S7, _ana;
tier 2 _S5L, _S8, _ana2). Steps run in the session process (fn) or as subprocesses (script: rk_loop paths, one block per process).
Every step <= 5 min at the planned rates and resumable (evaluations, cells, turns and chunks are cached files). Modules r1b_r_*.py.
Light at import (heavy imports inside the step functions); registering R's loop names (SO-KEYS) is the only side effect."""
import json, os
from r1b_contracts import Step
import r1b_core as K

LOOPS_1B = ("RSI_a", "RSI_b", "SEA_t", "SPO_t", "SEA_raw", "RND_1", "RND_2")
K.register_r_loops(LOOPS_1B)
RL1 = ("RL1", "RL2")                 # an RL1 item is also dropped under RL2 (R.5)
DIRS3 = {14: ("ustar", "T_v", "sea_v", "cov4"), 18: ("ustar", "T_v"), 6: ("ustar", "T_v")}


def _s2(b):
    s = "r1b_R_S2_%02d" % b; f = "r1b_r_s2:"; st = []
    if b == 14:
        st.append(Step("r21_texts", fn=f + "axis_texts", out="ax1b/axis_texts.pt", minutes=1.4))
    st.append(Step("r21_axes", fn=f + "axes", out=f"ax1b/ax1b_b{b}.pt ax1b/ax1b_b{b}.json", after="r21_texts" if b == 14 else "", minutes=1.0))
    st.append(Step("r22_checks", fn=f + "checks", out=f"r2/b{b}/checks.json", after="r21_axes", minutes=0.3))
    if b == 14:
        st.append(Step("r22_cc4", script="r1b_r_s2.py", out="r2/b14/cc4.json", minutes=1.0))
    for arm, loops in (("DN", LOOPS_1B), ("FX", ("RSI_a", "SEA_raw", "RND_1")), ("NP", ("RSI_a", "SEA_t", "RND_1"))):
        for L in loops:
            drop = () if b == 14 else (RL1 if (arm == "FX" and L != "RSI_a") else (("RL2",) if arm == "NP" else ()))
            st.append(Step(f"r22_{arm}_{L}", fn=f + "loop", out=f"r2/b{b}/dirs_{arm}_{L}.pt", after="r22_checks", minutes=3.5, drop_if=drop))
    for L in ("RSI_a", "RSI_b"):
        st.append(Step(f"r22_DN_{L}_r56", fn=f + "loop", out=f"r2/b{b}/dirs_DN_{L}_r56.pt", after=f"r22_DN_{L}", minutes=2.0, drop_if=RL1))
    st.append(Step("r22_summary", fn=f + "summary", out=f"r2/b{b}/r31_calibration.json", after="r22_DN_RSI_a", minutes=0.2))
    return s, st


def _s3():
    f = "r1b_r_s3:"; st = [Step("s3_unpushed_b%d" % b, fn=f + "unpushed", out=f"r3/b{b}/unpushed.pt", minutes=1.0) for b in (14, 18, 6)]
    for b, ds in DIRS3.items():
        for u in ds:
            st.append(Step(f"s3_dose_b{b}_{u}", fn=f + "dose", out=f"r3/b{b}/dose_{u}.json", after=f"s3_unpushed_b{b}", minutes=4.5))
    st += [Step("s3_txt0_a", fn=f + "txt0", out="txt0/txt0_a.pt", minutes=2.5), Step("s3_txt0_b", fn=f + "txt0", out="txt0/txt0_b.pt", minutes=2.5),
           Step("s3_words", fn=f + "words", out="txt0/wordsets.json", after="s3_txt0_a", minutes=0.5)]
    for b in (14, 18, 6):
        st += [Step(f"s3_txt0_read_b{b}", fn=f + "txt0_read", out=f"txt0/reads_b{b}.pt", after="s3_txt0_a, s3_txt0_b", minutes=1.0),
               Step(f"s3_fdt_b{b}", fn=f + "fdt", out=f"r4/fdt_b{b}.pt", after=f"s3_txt0_read_b{b}", minutes=4.0),
               Step(f"s3_r42_b{b}", fn=f + "r42", out=f"r4/jloop_b{b}.pt", after=f"s3_fdt_b{b}, s3_words", minutes=2.5),
               Step(f"s3_r43_b{b}", fn=f + "r43", out=f"r4/r43_b{b}.json", after=f"s3_r42_b{b}", minutes=2.4)]
    return "r1b_R_S3", st


def _ana():
    f = "r1b_r_ana:"
    st = [Step("a_load", fn=f + "load", out="ana/index.json", minutes=2.0)]
    for b in (14, 18, 6):
        st.append(Step(f"a_r2_b{b}", fn=f + "r2_block", out=f"ana/r2_b{b}.json", after="a_load", minutes=4.0))
        st.append(Step(f"a_sens_b{b}", fn=f + "sens", out=f"ana/sens_b{b}.json", after=f"a_r2_b{b}", minutes=4.5))
        st.append(Step(f"a_r3_b{b}", fn=f + "r3_block", out=f"ana/r3_b{b}.json", after="a_load", minutes=3.0))
        st.append(Step(f"a_r4_b{b}", fn=f + "r4_block", out=f"ana/r4_b{b}.json", after="a_load", minutes=4.5))
    st += [Step("a_r6", fn=f + "r6", out="ana/r6.json", after="a_load", minutes=2.0),
           Step("a_verdicts", fn=f + "verdicts", out="r1b_R_verdict.json rg15.json",
                after="a_r2_b14, a_r2_b18, a_r2_b6, a_sens_b14, a_sens_b18, a_sens_b6, a_r3_b14, a_r3_b18, a_r3_b6, a_r4_b14, a_r4_b18, a_r4_b6, a_r6",
                minutes=1.5)]
    return "r1b_R_ana", st


STEPS = dict([_s2(14), _s2(18), _s2(6), _s3(), _ana()])
STEPS["r1b_R_S0"] = [Step("s0_soax", fn="r1b_r_s0:soax", out="so_ax/so_ax_b14.pt so_ax/so_ax_b18.pt so_ax/so_ax_b6.pt so_ax/so_ax_check.json "
                          "xfer/so_ax_ustar.pt", minutes=1.0),
                     Step("s0_ref16", fn="r1b_r_s0:ref16", out="xfer/dmg_ref_R16.json", minutes=0.3),
                     *[Step(f"s0_r0_b{b}", fn="r1b_r_s0:r0_block", out=f"r0/r0_b{b}.json", after="s0_soax", minutes=2.5) for b in (14, 18, 6)],
                     Step("s0_r07", fn="r1b_r_s0:r07", out="r0/r07.json", after="s0_r0_b14, s0_r0_b18, s0_r0_b6", minutes=4.0),
                     Step("s0_r0_table", fn="r1b_r_s0:r0_table", out="r0_table.json", after="s0_r07", minutes=0.2),
                     Step("s0_r81", fn="r1b_r_s0:r81", out="r8_1.json", minutes=0.2),
                     Step("s0_r95", fn="r1b_r_s0:r95", out="r95/r95.json", minutes=0.2)]
STEPS["r1b_R_S1"] = [*[Step(f"s1_closed_b{b}_{stg}", script="r1b_r_s1.py", args=f"closed {b} {stg}", out=f"r1/closed_b{b}_{stg}.json",
                            minutes=3.5, timeout_s=3 * 3600) for b, stgs in ((14, ("pilot1", "pilot1.5", "loops1.5")), (18, ("pilot1", "loops1")),
                                                                            (6, ("pilot1", "pilot1.5", "loops1"))) for stg in stgs],
                     *[Step(f"s1_open_b{b}", script="r1b_r_s1.py", args=f"open {b}", out=f"r1/open_b{b}.json", minutes=4.0, timeout_s=3 * 3600)
                       for b in (14, 18, 6)],
                     Step("s1_compare", fn="r1b_r_s1:compare", out="r1/r1_replay.json",
                          after=", ".join([f"s1_open_b{b}" for b in (14, 18, 6)] + ["s1_closed_b14_loops1.5", "s1_closed_b18_loops1", "s1_closed_b6_loops1"]),
                          minutes=2.0)]
STEPS["r1b_R_S5"] = [Step("s5_lens", fn="r1b_r_s5:lens", out="r5/lens.json", minutes=1.0),
                     Step("s5_axes_lens", fn="r1b_r_s5:axes_lens", out="ax1b/ax1b_lens.json", minutes=2.0),
                     *[Step(f"s5_r51_b{b}", fn="r1b_r_s5:r51_block", out=f"r5/r51_b{b}.json", after="s5_lens, s5_axes_lens", minutes=1.5)
                       for b in (2, 6, 10, 14, 18, 22, 24, 26)],
                     Step("s5_rg10", fn="r1b_r_s5:rg10", out="jlens_validity.json rg11.json",
                          after=", ".join(f"s5_r51_b{b}" for b in (2, 6, 10, 14, 18, 22, 24, 26)), minutes=0.5),
                     Step("s5_r52", fn="r1b_r_s5:r52", out="r5/r52.json", after="s5_rg10", minutes=1.7),
                     Step("s5_r53", fn="r1b_r_s5:r53", out="r5/r53.json", after="s5_rg10", minutes=2.4),
                     Step("s5_r55", fn="r1b_r_s5:r55", out="r5/r55.json", after="s5_lens", minutes=2.2)]
STEPS["r1b_R_S6"] = [*[Step(f"s6_r61_b{b}", fn="r1b_r_s6:r61", out=f"r6/r61_b{b}.json", minutes=4.0) for b in (14, 18, 6)],
                     *[Step(f"s6_r61it_b{b}", fn="r1b_r_s6:r61_iter", out=f"r6/r61it_b{b}.json", minutes=3.0, drop_if=(RL1 if b != 14 else ("RL2",)))
                       for b in (14, 18, 6)],
                     Step("s6_r62_T0.7", fn="r1b_r_s6:r62", out="r6/r62_T0.7.pt", minutes=4.5),
                     Step("s6_r62_T1.0", fn="r1b_r_s6:r62", out="r6/r62_T1.0.pt", minutes=4.5),
                     Step("s6_r62_rnd", fn="r1b_r_s6:r62_rnd", out="r6/r62_rnd.pt", minutes=2.0),
                     Step("s6_r63", fn="r1b_r_s6:r63", out="r6/r63.pt", minutes=1.5)]
STEPS["r1b_R_S7"] = [Step("s7_r71_select", fn="r1b_r_s7:select", out="r7/r71_select.json", minutes=2.0),
                     Step("s7_r71_texts", fn="r1b_r_s7:texts", out="r7/r71.json", after="s7_r71_select", minutes=4.0),
                     Step("s7_r72", fn="r1b_r_s7:cue", out="r7/r72.json", after="s7_r71_texts", minutes=0.6),
                     Step("s7_r73", fn="r1b_r_s7:q4b", out="r7/r73.json", after="s7_r71_texts", minutes=3.0)]
STEPS["r1b_R_S5L"] = [Step("s5l_texts", fn="r1b_r_s5:refit_texts", out="r5l/texts.pt", minutes=2.0),
                      *[Step(f"s5l_fit_{k}", fn="r1b_r_s5:refit_fit", out=f"r5l/fit_{k}.pt", after="s5l_texts", minutes=4.5) for k in range(16)],
                      Step("s5l_lens", fn="r1b_r_s5:refit_lens", out="r5l/lens_j1b.pt", after=", ".join(f"s5l_fit_{k}" for k in range(16)), minutes=0.5),
                      *[Step(f"s5l_r51_b{b}", fn="r1b_r_s5:r51_block", out=f"r5l/r51_b{b}.json", after="s5l_lens", minutes=1.5)
                        for b in (2, 6, 10, 14, 18, 22, 24, 26)],
                      Step("s5l_rg10", fn="r1b_r_s5:rg10", out="r5l/jlens_validity_j1b.json",
                           after=", ".join(f"s5l_r51_b{b}" for b in (2, 6, 10, 14, 18, 22, 24, 26)), minutes=0.3),
                      Step("s5l_r52", fn="r1b_r_s5:r52", out="r5l/r52.json", after="s5l_rg10", minutes=1.7),
                      Step("s5l_r53", fn="r1b_r_s5:r53", out="r5l/r53.json", after="s5l_rg10", minutes=2.4)]
STEPS["r1b_R_S8"] = [Step("s8_setup", fn="r1b_r_s8:setup", out="r8/setup.pt r8/j3.json", minutes=3.0),
                     *[Step(f"s8_loop_{L}", fn="r1b_r_s8:loop", out=f"r8/dirs_{L}.pt", after="s8_setup", minutes=3.0)
                       for L in ("RSI_a", "RSI_b", "SEA_t", "RND_1")],
                     Step("s8_summary", fn="r1b_r_s8:summary", out="r8/r8_2.json", after="s8_loop_RSI_a, s8_loop_RSI_b, s8_loop_SEA_t, s8_loop_RND_1",
                          minutes=1.0)]
STEPS["r1b_R_ana2"] = [Step("a2_verdicts", fn="r1b_r_ana:pass2", out="r1b_R_verdict2.json", minutes=2.0)]


def mock_inputs(runs_A, runs_B, root):
    """MOCK round-2 inputs consistent with the rk_loop MOCK round-1 states (R1B_MOCK_WITH_R1=1): b<b>_q0.pt u_star rebuilt from the mock
    dir_<L><SUF>_4 with TV's mode sets (r2RSIb: RSI_a's directions plus seeded noise), b<b>_q0.json, a TV copy whose
    image_cos_d1_u_star equals the mock values, and a small geom_stats_qwen3-0.6b.json for R8.1."""
    import glob, torch
    tvp = os.path.join(K.REPO, "results/ju_space/runde2_ergebnisse/theta_verdict.json")
    if not os.path.exists(tvp):
        return
    tv = json.load(open(tvp)); shard = {14: "gpu1", 18: "gpu2", 6: "gpu3"}; th = os.path.join(runs_A, "r2_q2", "theta"); os.makedirs(th, exist_ok=True)
    for b in (14, 18, 6):
        sp = os.path.join(runs_A, shard[b], "rk", f"b{b}", "state.pt")
        if not os.path.exists(sp):
            return
        st = torch.load(sp, map_location="cpu", weights_only=False)
        if "axes" not in st:
            return                                   # placeholder states (no R1B_MOCK_WITH_R1): nothing consistent to build
        suf = json.load(open(os.path.join(runs_A, shard[b], "rk", f"b{b}", "loops.json")))["_meta"].get("suffix", "")
        ms = tv["blocks"][str(b)]["B_Q0"]["mode_set"]; g = torch.Generator().manual_seed(b)
        dirs = {L: {n: st[f"dir_{L}{suf}_{n}"].double() for n in range(1, 5)} for L in ("RSI_a", "RSI_b", "SEA", "SPO", "RND_1", "RND_2")
                if f"dir_{L}{suf}_4" in st}
        if b in (18, 6):
            dirs["r2RSIb"] = {n: dirs["RSI_a"][n] + 0.3 * dirs["RSI_a"][n].norm() * torch.randn(dirs["RSI_a"][n].shape[0], generator=g,
                                                                                                 dtype=torch.float64) / dirs["RSI_a"][n].shape[0] ** 0.5 for n in range(1, 5)}
            torch.save({"entries": {f"dir_r2RSIb_{n}": dirs["r2RSIb"][n].float() for n in range(1, 5)}}, os.path.join(th, f"b{b}_r2RSIb.pt"))
        u = sum(ms["signs"][L] * dirs[L][4] for L in ms["members"]); u = u / u.norm()
        torch.save({"u_star": u, "uA": u, "uB": u, "d4": {L: dirs[L][4] for L in ms["members"]}, "cfg": {"mock": True}}, os.path.join(th, f"b{b}_q0.pt"))
        json.dump(dict(mode_set=ms, m_star2=tv["blocks"][str(b)]["B_Q0"]["m_star2"], mock=True), open(os.path.join(th, f"b{b}_q0.json"), "w"))
        tv["blocks"][str(b)]["B_Q0"]["image_cos_d1_u_star"] = {L: float(dirs[L][1] @ u / dirs[L][1].norm()) for L in dirs}
    os.makedirs(os.path.join(runs_A, "r2_ana"), exist_ok=True)
    json.dump(tv, open(os.path.join(runs_A, "r2_ana", "theta_verdict.json"), "w"))
    NL = 28; parts = {"P_cell_lo>0|IMP-DES": [l >= 4 for l in range(NL)], "q>=0.2|IMP-DES": [True] * NL,
                      "P_cell_lo>0|CHG-DES": [l >= 2 for l in range(NL)], "q>=0.2|CHG-DES": [True] * NL}
    fg = [0.7 + 0.01 * (l % 5) if l >= 12 else 0.85 for l in range(NL)]
    pooled = {k: dict(point=fg, lo=[x - 0.05 for x in fg], hi=[x + 0.05 for x in fg], n_nan_draws=[0] * NL)
              for k in ("FG|CHG-DES", "FG|WOR-DES", "P_cell|IMP-DES", "P_cell|CHG-DES", "q|IMP-DES", "q|CHG-DES")}
    gate = dict(C0=[l >= 4 for l in range(NL)], C0_parts=parts, uncomposed=[x > 0.8 for x in fg], passing=[], P=[], n_P=0, runs=[])
    gp = os.path.join(runs_A, "r2_q3", "rgi"); os.makedirs(gp, exist_ok=True)
    json.dump(dict(mock=True, gate=gate, per_block=dict(pooled=pooled), blocks=list(range(NL))),
              open(os.path.join(gp, "geom_stats_qwen3-0.6b.json"), "w"))
