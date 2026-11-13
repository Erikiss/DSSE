"""Trimmed part-D grids (see cost.py for the rate assumptions)."""
LIN = 2.819e9 / 15.8e12; HOST = 0.03
def prefill(T, A): return T * LIN + 114688 * T * T / A + HOST
def chunk(E, P, A): return E * LIN + (229376 * E * P + 114688 * E * E) / A + HOST
def probe(P, A, n=1, size=40): return n * chunk(size, P, A)
def dec_rate(L):
    for lim, r in [(2500, 800), (5000, 450), (9000, 250), (17000, 120), (40000, 60)]:
        if L <= lim: return r
REC = 96
FL = [250, 4000, 12000, 28000]           # shared filler lengths (L-levels 1-4)
def run(A):
    o = {}
    o['D0 pilot'] = 15 * 60
    o['pools'] = (768 + 256) * 96 / 1000 + 60
    # D1a kernel: 12 items; 32k background (1 prefill/item), d in {1,4,16,64,256}; 8k background, d in {1,4,16,64}; run-length r {4,16} contig/spread at 32k
    t = 0
    for it in range(12):
        t += prefill(31500, A) + prefill(7500, A)
        for v in range(2):
            for d in [1, 4, 16, 64, 256]:
                P = (31500 // REC - d) * REC; E = d * REC + 128; t += chunk(E, P, A) + probe(P + E, A, 4)
            for d in [1, 4, 16, 64]:
                P = (7500 // REC - d) * REC; E = d * REC + 128; t += chunk(E, P, A) + probe(P + E, A, 4)
            for r in [4, 16]:
                for span in [r, 4 * r]:
                    P = 31500 - span * REC; E = span * REC + 128; t += chunk(E, P, A) + probe(P + E, A, 4)
    o['D1a kernel'] = t
    # D1b copy-bait: 12 items x 2 v x {AB, Bonly} x F {250, 28000}
    t = 0
    for it in range(12):
        for v in range(2):
            for c in range(2):
                for F in [250, 28000]:
                    P = 256 + F; E = 3100; t += chunk(E, P, A) + probe(P + E, A, 4)
    o['D1b copy'] = t
    # D3 M1: filler prefixes: 3 coherent + 3 shuffled nested to 28k
    t = 6 * prefill(28500, A)
    for it in range(24):
        for nrec in [30, 36]:
            E = nrec * REC + 188
            for F in FL:
                P = 256 + F; t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
        # shuffled filler at F 28000, alpha 0.2 only
        E = 30 * REC + 188; P = 256 + 28000; t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
    for it in range(12):   # format check: 30 turns, 2 template conventions, F {250, 28000}, alpha 0.2
        E = 30 * REC + 548
        for conv in range(2):
            for F in [250, 28000]:
                P = 256 + F; t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
    o['D3 M1 Lambda'] = t
    t = 0
    for s in range(16):
        for cell in range(3):
            for F in FL:
                P = 300 + F; E = 1200; t += chunk(E, P, A) + probe(P + E, A, 6, 50)
    o['D3 M2 Lambda'] = t
    t = 0
    for it in range(24):
        for v in range(2):
            for nrec in [18, 30, 54, 15, 60]:
                E = nrec * REC + 188; P = 256 + 12000; t += chunk(E, P, A) + probe(P + E, A, 4)
    o['D3 count/share'] = t
    # D2-S: 12 items; AB alpha {0.2,0.5} x 2v nested to 32k (4); A-only/B-only alpha {0.2,0.5} x 2v nested to 16k (8); neither nested to 32k (1)
    t = 0
    for it in range(12):
        t += 5 * prefill(32000, A) + 8 * prefill(16000, A)
        t += 5 * sum(probe(L, A, 4) + chunk(128, L, A) for L in [1000, 2000, 4000, 8000, 16000, 32000])
        t += 8 * sum(probe(L, A, 4) + chunk(128, L, A) for L in [1000, 2000, 4000, 8000, 16000])
    o['D2 S'] = t
    # D2-R: 16 items; per item 3 nested prefixes (A_v+, A_v-, neutral) to 0.9*32k; alpha {0.1,0.2,0.33} x L {2k,8k,32k} x 7 run chunks
    t = 0
    for it in range(16):
        t += 3 * prefill(0.9 * 32000, A)
        for a in [0.1, 0.2, 0.33]:
            for L in [2000, 8000, 32000]:
                E = a * L + 128; P = (1 - a) * L; t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
    o['D2 R'] = t
    # D4: 24 items; parity: 3 conds x 2 v x 2 arr x F {250, 4000, 28000}; dominant 8:24 interleaved x F {250, 28000}
    t = 0
    for it in range(24):
        for cond, npr in [('B', 12), ('C', 8), ('F', 8)]:
            for v in range(2):
                for arr in range(2):
                    for F in [250, 4000, 28000]:
                        E = 16 * REC + 188; P = 256 + F; t += chunk(E, P, A) + probe(P + E, A, npr)
                for F in [250, 28000]:
                    E = 32 * REC + 188; P = 256 + F; t += chunk(E, P, A) + probe(P + E, A, npr)
    o['D4 catfish'] = t
    # D7: 2 reference chunks + 4 factual probes on D3 alpha 0.2 F {250, 28000} 7 cells, and D4 parity interleaved F {250, 28000}
    t = 0
    for it in range(24):
        for F in [250, 28000]:
            P = 256 + F + 3100; t += 7 * (2 * chunk(128, P, A) + probe(P, A, 4))
            P = 256 + F + 1700; t += 6 * (2 * chunk(128, P, A) + probe(P, A, 4))
    o['D7 cost'] = t
    # D8: 12 items; encodings x{1, 2.0, 4.35}; levels canonical {short: 12 records alone, long: 7k}; D3 2x2 alpha 0.2 (7 cells); D4 parity competing + non-competing interleaved (4) at long only
    t = 0
    for f in [1.0, 2.0, 4.35]:
        t += 3 * prefill(f * 3700, A)
        for it in range(12):
            E = f * 1200 + 188; P = f * 256; t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
            E = f * 3000 + 188; P = f * (256 + 3500); t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
            E2 = f * 1600 + 188; P2 = f * (256 + 5000); t += 4 * (chunk(E2, P2, A) + probe(P2 + E2, A, 12))
    o['D8 retok'] = t
    # D5: probes after turns 1 and 3 only
    def d5(L0, nseed=16, nturn=3, ntok=64):
        g = nseed * nturn * ntok
        return prefill(L0, A) + g / dec_rate(L0 + nturn * (ntok + 16)) + nseed * 2 * probe(L0 + 200, A, 6, 50)
    t = 0
    for pole in range(2):
        for s in range(8):
            t += d5(1000) + d5(8000) + d5(16000)     # L-arm T 0.7
            t += d5(16000) + d5(1000)                # L-arm T 1.0 endpoints
            t += d5(8000) + d5(8000)                 # T-arm, E-arm
            t += d5(8000)                            # official-template convention
            t += d5(1000) + prefill(1000, A)         # split-detector positive control; reference readouts
    o['D5 split'] = t
    # D6: 3 arms x 16 starts x 2 seeds at T 0.7 + 3 arms x 8 starts x 2 seeds at T 1.0; 4096 tokens; probes every 512 (3 x 40 tok)
    n = 3 * 16 * 2 + 3 * 8 * 2
    t = n * 4096 / 450 + n * 8 * probe(3000, A, 3, 40) + n * prefill(1100, A)
    o['D6 lifetime'] = t
    return o
for A in [8e12, 12e12, 20e12]:
    o = run(A); tot = sum(o.values())
    print(f"A {A/1e12:.0f} TFLOP/s | work {tot/60:.0f} min ({tot/3600:.2f} h)")
    if A == 12e12:
        for k, v in o.items(): print(f"   {k:16s} {v/60:6.1f} min")
