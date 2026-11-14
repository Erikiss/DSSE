"""Part L work (L.8 and the 'Runs in' breakdowns of L1-L5, X1-X6) re-costed at r1b_P0's measured rates (arithmetic only).
Rate classes and factors (measured / planned):
  RD   teacher-forced reads: planned 5,600 (R-all) / 5,000 (R-full) tok/s; measured 4,545 tok/s hooked (L0_a (ii)) -> x1.23 / x1.10
  DEC  E.decode at batch 32: measured 46.5 ms / step at < 100 tokens of context (L0_b duet: 2.13 s per 32 x 40 incl. ~0.27 s prefill),
       + 15 us per context token (bounded by L0_a: 33-70 ms at 1.1k), hooks x 1.10; planned 25-35 ms x 1.15
  PF   batched prefill: measured ~0.17-0.2 ms / token (L0_a totals, D's B = 1 fit) = planned 5,000 tok/s (x 1.0)
  LONG B = 1 32k pass: measured 9.1 s (31.3k tokens, P0-D) + 1.3 s lm_head over all positions, against 12-21 s planned
  GPU  fits / statistics on the GPU: unmeasured, x 1.0;  CPU: 0
DUET=reprefill (as coded in r1b_l_duet.py) or carried (the pre-registered -40 % option; needs a code path)."""
import os, json
DUET = os.environ.get("DUET", "reprefill")


def dec_step(ctx_mean, hooked):
    return (0.0465 + 15e-6 * ctx_mean) * (1.10 if hooked else 1.0)


W = {}   # measurement -> (planned work min, measured work min)
# ---- r1b_L_A
l1 = dict(mu_j=(0.3, 0.3 * 1.10), cett=(2.0, 2.0 * 1.10), sel=(1.0, 1.0), reread=(1.0, 1.10), pip=(10.0, 10.0), resel=(2.0, 2.0),
          l1c=(15.0, 15.0), erasure=(10.0, 1.5 + (10.0 - 1.5) * (0.0175 + 128 * 0.24e-3) / (0.020 + 128 * 0.218e-3)))
W["L1"] = (sum(v[0] for v in l1.values()), sum(v[1] for v in l1.values()))
l3b_dec_p, l3b_dec_m = 624 * 64 * 0.027 * 1.15 / 60, 624 * 64 * dec_step(250, True) / 60
W["L3b"] = (34.5, l3b_dec_m + 624 * 0.4 / 60 + 7.4 * 1.23 + 0.6 * 1.23 + 1.5)
msg_plan = 32 * 420 / 5000 + 40 * 0.030
msg_meas = (32 * 420 * 0.17e-3 + 40 * dec_step(440, False)) if DUET == "reprefill" else (32 * 60 * 0.19e-3 + 0.038 + 40 * dec_step(440, False))
W["L4"] = (34.4, 4 * 96 * msg_meas / 60 + 2 + 5.8 * 1.23 + 0.8 + 1)
# ---- r1b_L_B
l2a_p = 21 + 1 + 2 + 3.4 + 2
l2a_m = 24 * ((9.1 + 1.3) + 2 * (9.1 + 1.3)) / 60 + 1 + 2 + 4096 * (max(0.038, 0.0134 + 2500 * (3.45e-6 * 4 + 0.486e-6 * 24)) + 24 * 0.218e-3) / 60 + 2   # one batch of 24 rows, 4,096 steps, mean context ~2.5k
W["L2a"] = (29.4, l2a_m)
W["L2b"] = (23.1, 16 * 1024 * dec_step(612, True) / 60 + 3.8 * 1.10 + 0.8 * 1.10 + 24 * 512 * dec_step(356, True) / 60)
W["L2c"] = (3.0, 3.0)
W["L3a"] = (6.0, 6.0)
W["L3c"] = (8.0, 8.0)
W["L3d"] = (2.5, 0.8 * 1.10 + 1.6 + 0.1)
W["L5"] = (3.4, 3.4 * dec_step(300, False) / 0.035)
EXT = {"X1": (8.5, 88 * (80 * dec_step(400, True) + 0.5) / 60 + 0.5), "X2": (15.3, 15.3 * W["L4"][1] / W["L4"][0]), "X3": (5.0, 5.0),
       "X4": (2.5, 2.5 * 1.2), "X5": (3.0, 3.0 * 1.2), "X6": (2.6, 2.6 * W["L4"][1] / W["L4"][0])}
A_core = ("L1", "L3b", "L4"); B_core = ("L2a", "L2b", "L2c", "L3a", "L3c", "L3d", "L5")
iota_p, iota_m = 2.9, 2.9 * 1.10            # L.14 amendment (planned minutes, x1.5 included)
def sess(keys, ext, w):
    core = sum(W[k][w] for k in keys)
    return core, 4 + 1.5 * core, 1.5 * sum(EXT[x][w] for x in ext)
out = {}
for w, nm in ((0, "plan"), (1, "meas")):
    ca, pa, ea = sess(A_core, ("X2", "X6"), w)
    cb, pb, eb = sess(B_core, ("X1", "X3", "X4", "X5"), w)
    pb += iota_p if w == 0 else iota_m
    inK = 10.6 * (1.0 if w == 0 else 1.15)    # S_L screening and 22 extra wrong items run in K's sessions (scaled like K_S1 / K_S2a)
    out[nm] = dict(core_work=dict(L_A=round(ca, 1), L_B=round(cb, 1)), L_A_min=round(pa, 1), L_B_min=round(pb, 1),
                   tier1_h=round((pa + pb) / 60, 2), tier1_charged_h=round((pa + pb + inK) / 60, 2), tier2_h=round((ea + eb) / 60, 2),
                   items={k: round(v[w], 1) for k, v in {**W, **EXT}.items()})
print(json.dumps(dict(duet=DUET, **out), indent=1))
