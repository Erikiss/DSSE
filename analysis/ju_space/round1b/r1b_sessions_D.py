"""Part D: sessions and steps (main §4.2: r1b_D_A (account B), r1b_D_B, r1b_D_C, r1b_D_ana; tier 2 r1b_D_R). r1b_D_A writes the SG9
records (contract file 'glce': lce/g_lce_1_2.json) in its first step, since r1b_L_B starts on them. Code: r1b_lce.py (SO-LCE),
r1b_d_stim.py (stimuli, streams, helpers), r1b_d_d1.py (D0 / D1 arms), r1b_d_d2.py (D2), r1b_d_d3.py (D3, D4), r1b_d_d5.py (D5, D5-x),
r1b_d_ana.py (verdicts, D.17), r1b_d_res.py (reserves R1-R3). Self-test: python r1b_d_selftest.py <root> (MOCK, CPU)."""
from r1b_contracts import Step

D1, D2, D3, D5, ANA, RES = "r1b_d_d1", "r1b_d_d2", "r1b_d_d3", "r1b_d_d5", "r1b_d_ana", "r1b_d_res"

STEPS = {
    "r1b_P0": [Step("d0_pilot", fn=f"{D1}:p0_d", out="p0/p0_d.json", minutes=4.0)],
    "r1b_D_A": [
        Step("d_glce", fn=f"{D1}:d_glce", out="lce/g_lce_1_2.json", minutes=7.0),
        Step("d1_ends", fn=f"{D1}:d1_ends", out="d1/raw_ends.pt d1/ends.json", after="d_glce", minutes=19.2),
        Step("d1_lstar", fn=f"{D1}:d1_lstar", out="d1/lstar.json", after="d1_ends", minutes=0.1),
        Step("d1_L0", fn=f"{D1}:d1_L0", out="d1/raw_L0.pt d1/L0.json", after="d1_lstar", minutes=30.0),
        Step("d1_Ls", fn=f"{D1}:d1_Ls", out="d1/raw_Ls.pt d1/Ls.json", after="d1_lstar", minutes=54.0),
        Step("d1_Lf", fn=f"{D1}:d1_Lf", out="d1/raw_Lf.pt d1/Lf.json", after="d1_lstar", minutes=8.1),
        Step("d1_v", fn=f"{D1}:d1_v", out="d1/d1v.json d1/d1v_states.pt", minutes=1.0),
        Step("d1_cov", fn=f"{D1}:d1_cov", out="d1/sg8_cov.json", minutes=0.5),
        Step("d1_s", fn=f"{D1}:d1_s", out="d1/raw_s.pt d1/s.json d1/d1s_glce.json", minutes=10.8),
    ],
    "r1b_D_B": [
        Step("d2_glce3", fn=f"{D2}:d2_glce3", out="d2/glce3.json", minutes=1.3),
        Step("d2_gen_long", fn=f"{D2}:d2_gen_long", out="d2/gen_long.pt d2/gen_long.json", opt="d2/fallback.json", after="d2_glce3",
             minutes=62.2),
        Step("d2_gen_short", fn=f"{D2}:d2_gen_short", out="d2/gen_short.pt d2/gen_short.json", minutes=3.3),
        Step("d2_read", fn=f"{D2}:d2_read", out="d2/read.pt d2/d2_read.json", after="d2_gen_long,d2_gen_short", minutes=11.3),
    ],
    "r1b_D_C": [
        Step("d3_starts", fn=f"{D3}:d3_starts", out="d3/starts.pt d3/starts.json", minutes=3.0),
        Step("d3_gen", fn=f"{D3}:d3_gen", out="d3/traj.pt d3/d3_gen.json", after="d3_starts", minutes=44.7),
        Step("d3_name", fn=f"{D3}:d3_name", out="d3/d3_name.json", after="d3_gen", item="D3-name", minutes=5.0),
        Step("d4_starts", fn=f"{D3}:d4_starts", out="d4/starts.pt d4/starts.json", minutes=1.5),
        Step("d4_gen", fn=f"{D3}:d4_gen", out="d4/traj.pt d4/d4_gen.json", after="d4_starts", minutes=12.9),
        Step("d4_uaa", fn=f"{D3}:d4_uaa", out="d4/u_aa.pt d4/u_aa.json", minutes=1.5),
        Step("d5_pool", fn=f"{D5}:d5_pool", out="d5/pool.json", minutes=2.0),
        Step("d5_g6", fn=f"{D5}:d5_g6", out="d5/g6_detector.json", minutes=0.2),
        Step("d5_gen", fn=f"{D5}:d5_gen", out="d5/gen.pt d5/d5_gen.json", after="d5_pool", minutes=30.8),
        Step("d5x", fn=f"{D5}:d5x", out="d5/d5x.json", after="d5_gen", item="D5-x", minutes=20.0),
    ],
    "r1b_D_ana": [
        Step("ana_d1", fn=f"{ANA}:ana_d1", out="d_ana/d1.json", minutes=6.0),
        Step("ana_d2", fn=f"{ANA}:ana_d2", out="d_ana/d2.json", minutes=3.0),
        Step("ana_d3", fn=f"{ANA}:ana_d3", out="d_ana/d3d4.json", minutes=3.0),
        Step("ana_d5", fn=f"{ANA}:ana_d5", out="d_ana/d5.json", minutes=1.0),
        Step("ana_comb", fn=f"{ANA}:ana_comb", out="d_verdict.json", after="ana_d1,ana_d2,ana_d3,ana_d5", minutes=0.5),
    ],
    "r1b_D_R": [
        Step("res_r1", fn=f"{RES}:res_r1", out="res/r1.json", minutes=21.0),
        Step("res_r2", fn=f"{RES}:res_r2", out="res/r2.json", minutes=36.0),
        Step("res_r3", fn=f"{RES}:res_r3", out="res/r3.json res/r3_raw.pt", minutes=19.0),
        Step("res_summary", fn=f"{RES}:res_summary", out="d_reserves.json", minutes=0.1),
    ],
}
