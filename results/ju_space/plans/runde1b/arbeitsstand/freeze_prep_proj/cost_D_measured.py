"""Part D cost table re-run at r1b_P0's measured rates (arithmetic only; structure copied from merge_D/cost_final.py, D.19).
MODE=plan reproduces D.19 (attention 10 TFLOP/s, KV factor 3); MODE=meas uses the P0-D fits:
  prefill(T) = 0.115 s + 0.1546 ms * T + 4.243e-9 s * T^2           (B = 1, 15.7K / 28.0K / 31.3K exact; 3.4K point is warm-up)
  decode step(L, B) = max(HOST, 13.4 ms + L * (LAM * waves(1, B) + KAP * B * padf)) + extra linear rows
     LAM = 3.45 us / context token (fp32 mem-efficient kernel, q_len 1, one wave of <= 108 CTAs = B * 16 heads),
     KAP = 0.486 us / context token / row (DynamicCache rewrite), HOST = 38 ms (decode floor at 4K, B <= 3)
  chunk(c, P, B) = HOST + B * c * 0.19 ms + P * (LAMC * waves(c, B) + KAP * B)    (LAMC 3.83 us: fits the 700-token chunk at 4-32K)
  short batched decode (B 32, < 1K context): 46.5 ms / step (L0_b duet: 2.13 s per 32 x 40 incl. a ~0.27 s prefill)
  teacher-forced reads: 4,545 tok/s hooked (L0_a (ii)) instead of 5,000 (x 1.10); 9,000 tok/s stopped reads x 1.10
Trims: TRIMS env, e.g. 'T2,T3'; FUND_D5X=1 counts D5-x (tier 2) with T1 if listed. B_LONG env (default 3) for D2 / D3 long rows."""
import math, os, json
MODE = os.environ.get("MODE", "meas"); TR = set(x for x in os.environ.get("TRIMS", "").split(",") if x)
B_LONG = int(os.environ.get("B_LONG", "3")); B_MID = int(os.environ.get("B_MID", "7"))
LIN, LINR, AC, OVH, KVB, HBM = 2.819e9, 15.8e12, 229376.0, 0.03, 229376.0, 1.555e12
ATT, KVF = 10e12, 3.0
HOST, A0, LAM, LAMC, KAP, LINC, LIND = 0.038, 0.0134, 3.45e-6, 3.83e-6, 0.486e-6, 0.19e-3, 0.218e-3
SHORT_STEP, READ = 0.0465, 1.10
if os.environ.get("KAP0") == "1":
    KAP = 0.0          # scenario: in-place (non-rewriting) KV buffer, same kernel and values
if os.environ.get("HOST") :
    HOST = float(os.environ["HOST"])


def waves(c, B):
    return math.ceil(math.ceil(c / 64) * 16 * B / 108)


def chunk(c, P, B=1, lin=LIN):
    if MODE == "plan":
        mem = 6 * KVB * (P + c) * B / HBM if P > 0 else 0.0
        return B * c * lin / LINR + AC * c * (P + c / 2) * B / ATT + mem + OVH
    return HOST + B * c * LINC * (lin / LIN) + P * (LAMC * waves(c, B) + KAP * B) + (AC * c * c / 2 * B / 27e12)


def prefill(T, lin=LIN):
    if MODE == "plan":
        return T * lin / LINR + AC * T * T / 2 / ATT + OVH
    return 0.115 + 0.1546e-3 * T * (lin / LIN) + 4.243e-9 * T * T


def step(L, B, pad=1.0):
    if MODE == "plan":
        gpu = (KVF + (2.0 if pad > 1 else 0.0)) * KVB * L * B / HBM + AC * L * B / ATT + B * LIN / LINR
        return max(0.018, gpu) + 0.002
    padf = 3.0 if pad > 1 else 1.0
    return max(HOST, A0 + L * (LAM * waves(1, B) + KAP * B * padf) + max(0, B - 3) * LIND)


def gen(L, B, ntok, pad=1.0):
    return ntok * step(L, B, pad)


BLK, Q = 680, 55


def ctx(L, nq, lin=LIN):
    P = 200 if L < 2000 else L - BLK - Q
    return chunk(BLK, P, 1, lin) + nq * chunk(Q, P + BLK, 1, lin)


L0, L4, L16, L32 = 1100, 4096, 16384, 32000


def pl(L, items, n6, n12, lin=LIN):
    return items * (n6 * ctx(L, 6, lin) + n12 * ctx(L, 12, lin)) / 60


def it(n, trim, to=16):
    return to if trim in TR else n


A, Bd, C = {}, {}, {}
A['D0a G-LCE-1/2'] = (16 * (prefill(L16) + prefill(L32)) + 32 * (ctx(L16, 2) + ctx(L32, 2))) / 60
A['stream prefills'] = (16 * (prefill(L4) + prefill(L16) + prefill(L32)) + 8 * prefill(L32)) / 60
A['D1 main (int 48, rel 24 items; T6 rel 16)'] = sum(pl(L, 48, 20, 2) + pl(L, it(24, 'T6'), 0, 10) for L in (L0, L32))
A['D1 neutral / pure / 4K subset'] = (sum(pl(L, 48, 0, 1) for L in (L0, L4, L16, L32)) + sum(pl(L, 48, 0, 2) for L in (L4, L16)) + pl(L4, 16, 20, 2))
A['D1 non-competing'] = sum(pl(L, 48, 4, 0) for L in (L0, L32))


def costread(L):
    P = 200 if L < 2000 else L
    return chunk(256, P) + 4 * chunk(40, P)


A['D1-d cost readouts'] = sum(24 * 11 * costread(L) / 60 for L in (L0, L32))
A['D1-c (T2: L0 only)'] = sum(pl(L, 24, 12, 0) for L in ((L0,) if 'T2' in TR else (L0, L32)))
A['D1-h'] = sum(pl(L, 32, 8, 0) for L in (L0, L32))
A['D1-f (T5: 16 items)'] = pl(L32, it(24, 'T5'), 8, 2)
A['D1-e'] = sum(pl(L, 24, 12, 2) for L in (L0, L32)) + pl(L0, 24, 0, 1)
A['D1-r'] = (24 * 32 * (chunk(BLK + 240, 200) + 12 * chunk(Q, 1200))) / 60
A['D1-v'] = 1.0
lin06 = 0.88e9
A['D1-s (T3: 16 items)'] = (sum(pl(L, it(24, 'T3'), 8, 3, lin06) for L in (L0, L32)) + 8 * prefill(L32, lin06) / 60 + 1.5)
A['impl. amendment D1-f / D1-s own streams (raw)'] = (8 * prefill(L32) + 8 * prefill(L32, lin06)) / 60 if MODE == 'meas' else 3.93
n_sec = 48 if 'T4' in TR else 64
rows = 160 * 4 + n_sec * 6
Lb = 31800
Bd['D0b G-LCE-3'] = (4 * prefill(Lb) + 3 * 4 * gen(Lb, 1, 40)) / 60
Bd['D2 long decode'] = rows / B_LONG * gen(Lb + 300, B_LONG, 192) / 60
Bd['D2 long suffix chunks + stream prefills'] = (rows / B_LONG * 2 * chunk(150, Lb, B_LONG) + 16 * prefill(31500)) / 60
Bd['D2 short arm'] = rows * 192 / (1000 if MODE == 'plan' else 32 / SHORT_STEP) / 60
Bd['D2 probes, reads, NLL'] = (6144 * 350 / 5000 + 6144 * 130 / 9000 + 6144 * 130 / 5000) / 60 * (1.0 if MODE == 'plan' else READ)


def split(L, B, items=16, seeds=16):
    nb = items * seeds / B
    return (nb * gen(L + 440, B, 8 * 56) + nb * 4 * 6 * chunk(40, L + 500, B)) / 60


L3 = 16384 if 'T7' in TR else 28672
C['D3 split L0 (B 16)'] = split(200, 16)
C['D3 split long (B_LONG)'] = split(L3, B_LONG)
C['D3 n = 0 probes; A2/B2 n = 1 control'] = (2 * 16 * 3 * 6 * chunk(40, L3 + 300) + 16 * 16 / B_LONG * (gen(L3 + 300, B_LONG, 56) + 6 * chunk(40, L3 + 300, B_LONG))) / 60 + 0.5
C['INJ starts'] = 6.0 * (1.0 if MODE == 'plan' else 1.5)


def ident(L, B, roles=16, seeds=8):
    nb = roles * seeds / B
    return (nb * gen(L + 400, B, 8 * 56) + nb * 2 * 8 * chunk(40, L + 850, B)) / 60


C['D4 identity L0 (B 16) + 16,384 (B_MID)'] = ident(200, 16) + ident(16384, B_MID) + 2.0
starts = 24 if 'T8' in TR else 32
C['D5 lifetime (T8: 24 starts)'] = starts * 5 / 20 * sum(gen(1000 + 256 * t + 128, 20, 272, pad=2.0) for t in range(16)) / 60
C['D5 checkpoint probes + RSP pool'] = starts * 5 / 20 * 8 * 4 * chunk(40, 3000, 20) / 60 + 1.5
ana = 15.0
d5x = (10.0 if 'T1' in TR else 20.0) if os.environ.get("FUND_D5X", "1") == "1" else 0.0
out = dict(mode=MODE, trims=sorted(TR), B_long=B_LONG, items={k: round(v, 1) for d in (A, Bd, C) for k, v in d.items()})
sA, sB, sC = sum(A.values()), sum(Bd.values()), sum(C.values())
hA, hB, hC = (5 + 1.5 * sA) / 60, (5 + 1.5 * sB) / 60, (5 + 1.5 * sC) / 60
hana = (5 + 1.5 * ana + 5) / 60      # 5-min I/O allowance (main 4.1)
out.update(raw=dict(D_A=round(sA, 1), D_B=round(sB, 1), D_C=round(sC, 1), ana=ana), sessions_h=dict(D_A=round(hA, 2), D_B=round(hB, 2), D_C=round(hC, 2), D_ana=round(hana, 2)))
t1 = hA + hB + hC + hana
out.update(tier1_h=round(t1, 2), d5x_h=round(1.5 * d5x / 60, 2), tier1_plus_d5x_h=round(t1 + 1.5 * d5x / 60, 2))
print(json.dumps(out, indent=1))
