"""Final part-D grids. Rates as in cost.py (fp32, TF32 off, memory-efficient SDPA; attention rate A unmeasured: 8/12/20 TFLOP/s)."""
import json
LIN = 2.819e9 / 15.8e12; HOST = 0.03
def prefill(T, A): return T * LIN + 114688 * T * T / A + HOST
def chunk(E, P, A): return E * LIN + (229376 * E * P + 114688 * E * E) / A + HOST
def probe(P, A, n=1, size=40): return n * chunk(size, P, A)
def dec_rate(L):
    for lim, r in [(2500, 800), (5000, 450), (9000, 250), (17000, 120), (40000, 60)]:
        if L <= lim: return r
REC = 96; FL = [250, 4000, 12000, 28000]; PRE = 256
def run(A):
    o = {}
    o['D0'] = 15 * 60
    o['P (pools)'] = (512 + 128) * 96 / 1000 + 60
    # D1a kernel (A-background, 32k; B-only normalisation at d 1,16,256; run-length)
    t = 0
    for it in range(4):
        t += 3 * prefill(31500, A)
        for v in range(2):
            for d in [1, 4, 16, 64, 256]:
                P = (31500 // REC - d) * REC; E = d * REC + 128; t += chunk(E, P, A) + probe(P + E, A, 4)
            for d in [1, 16, 256]:
                P = (31500 // REC - d) * REC; E = d * REC + 128; t += chunk(E, P, A) + probe(P + E, A, 4)
            for r in [4, 16]:
                for span in [r, 4 * r]:
                    P = 31500 - span * REC; E = span * REC + 128; t += 2 * (chunk(E, P, A) + probe(P + E, A, 4))
    o['D1a'] = t
    t = 0
    for it in range(12):
        for v in range(2):
            for c in range(2):
                for F in [250, 28000]:
                    P = PRE + F; E = 3100; t += chunk(E, P, A) + probe(P + E, A, 4)
    o['D1b'] = t
    t = 0
    for it in range(24):
        for v in range(2):
            for order in range(2):
                for F in [250, 28000]:
                    P = PRE + F; E = 12 * REC + 188; t += chunk(E, P, A) + probe(P + E, A, 4)
    for it in range(24):
        for F in [250, 28000]:
            P = PRE + F; E = 12 * REC + 188; t += chunk(E, P, A) + probe(P + E, A, 4)
    o['D1c'] = t
    # D3 M1 (32 items)
    t = 6 * prefill(28500, A)
    for it in range(32):
        for nrec in [30, 36]:
            E = nrec * REC + 188
            for F in FL:
                P = PRE + F; t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
    for it in range(8):
        E = 30 * REC + 188; P = PRE + 28000; t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
    for it in range(6):
        E = 30 * REC + 548
        for conv in range(2):
            for F in [250, 28000]:
                P = PRE + F; t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
    for it in range(8):
        for F in [250, 28000]:
            E = 30 * REC + 188; P = PRE + F; t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
    o['D3a (M1 Lambda)'] = t
    t = 0
    for it in range(24):
        for (na, nb) in [(12, 3), (48, 12)]:
            E = (na + nb) * REC + 188; P = PRE + 12000; t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
        for (na, nb) in [(12, 6), (48, 6)]:
            E = (na + nb) * REC + 188; P = PRE + 12000; t += 2 * (chunk(E, P, A) + probe(P + E, A, 4))
    o['D3c (count/share)'] = t
    t = 0
    for s in range(24):
        for cell in range(3):
            for F in FL:
                P = 300 + F; E = 1200; t += chunk(E, P, A) + probe(P + E, A, 30 if F in (250, 28000) else 12, 50)
        for m in [2, 24]:
            for pole in range(2):
                for F in [250, 28000]:
                    E = 150 * m + 100; P = 300 + F; t += chunk(E, P, A) + probe(P + E, A, 30, 50)
    for s_ in range(24):
        for chk in range(2):
            for cell in range(2):
                P = 300 + 27500 + 210 * 20; E = 1200; t += chunk(E, P, A) + probe(P + E, A, 30, 50)
    o['D3b (M2 Lambda)'] = t
    # D2-S: 12 items; 9 contexts nested to 32k, 4 to 16k; reads (spacer chunk + 4 probes) at 1k..16k (+32k)
    t = 0
    for it in range(8):
        t += 9 * prefill(32000, A) + 4 * prefill(16000, A)
        t += 9 * sum(probe(L, A, 4) + chunk(128, L, A) for L in [1000, 2000, 4000, 8000, 16000, 32000])
        t += 4 * sum(probe(L, A, 4) + chunk(128, L, A) for L in [1000, 2000, 4000, 8000, 16000])
    t += 12 * 4 * 2 * (16 * 4 / 100 + probe(16000, A, 1))
    o['D2S'] = t
    t = 0
    for it in range(16):
        t += 3 * prefill(0.9 * 32000, A)
        for a in [0.2, 0.33]:
            for L in [2000, 8000, 32000]:
                E = a * L + 128; P = (1 - a) * L; t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
    o['D2R'] = t
    t = 0
    for it in range(24):
        for cond, npr in [('B', 12), ('C', 8), ('F', 8)]:
            for v in range(2):
                for arr in range(2):
                    for F in [250, 4000, 28000]:
                        E = 16 * REC + 188; P = PRE + F; t += chunk(E, P, A) + probe(P + E, A, npr)
                for F in [250, 28000]:
                    E = 32 * REC + 188; P = PRE + F; t += chunk(E, P, A) + probe(P + E, A, npr)
    o['D4'] = t
    t = 0
    for it in range(32):
        for F in [250, 28000]:
            P = PRE + F + 3100; t += 7 * (2 * chunk(128, P, A) + probe(P, A, 4))
    for it in range(24):
        for F in [250, 28000]:
            P = PRE + F + 1700; t += 8 * (2 * chunk(128, P, A) + probe(P, A, 4))
    o['D7'] = t
    t = 0
    for f in [1.0, 4.35]:
        t += 3 * prefill(f * 3756, A)
        for it in range(10):
            E = f * 3000 + 188; P = f * PRE + 250 * f; t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
            E = f * 3000 + 188; P = f * (PRE + 3500); t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
            E2 = f * 1600 + 188; P2 = f * (PRE + 5000); t += 4 * (chunk(E2, P2, A) + probe(P2 + E2, A, 12))
    o['D8'] = t
    def d5(L0, nseed=16, nturn=3, ntok=64):
        g = nseed * nturn * ntok
        return prefill(L0, A) + g / dec_rate(L0 + nturn * (ntok + 16)) + nseed * 2 * probe(L0 + 200, A, 6, 50)
    t = 0
    for pole in range(2):
        for s in range(8):
            t += d5(1000) + d5(8000) + d5(16000) + d5(8000) + d5(8000) + d5(8000) + d5(1000) + 2 * prefill(1000, A)
            if s < 4: t += d5(16000) + d5(1000)
    o['D5'] = t
    n = 4 * 24
    t = n * 4096 / 450 + n * 8 * probe(3000, A, 3, 40) + n * prefill(1100, A)
    o['D6'] = t
    return o
res = {}
for A in [8e12, 12e12, 20e12]:
    o = run(A); res[A] = o; tot = sum(o.values())
    tot_m = (tot - 900) * 1.3 + 900 + 6 * 240 + 600
    print(f"A {A/1e12:.0f} TFLOP/s | work {tot/60:.0f} min ({tot/3600:.2f} h) | with x1.3 margin (pilot fixed), 6 setups x 4 min, analysis 10 min: {tot_m/60:.0f} min = {tot_m/3600:.2f} h")
print('per measurement, minutes at A = 8 / 12 / 20 TFLOP/s:')
for k in res[12e12]:
    print(f"   {k:20s} {res[8e12][k]/60:6.1f} {res[12e12][k]/60:6.1f} {res[20e12][k]/60:6.1f}")
json.dump({str(int(A/1e12)): {k: round(v/60, 2) for k, v in o.items()} for A, o in res.items()}, open('/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/scratch_D_confounds/cost_trimmed.json', 'w'), indent=1)
