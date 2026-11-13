"""Cost model for part D (A100-40GB, float32 weights and activations, TF32 off, SDPA memory-efficient kernel).
Rates (int_machinery.md, frozen round-2 plan section 7.1): linear 2.819 GFLOP/token at 15.8 TFLOP/s; causal attention
114,688*T^2 FLOP for a prefill, 229,376*E*P FLOP for an E-token chunk over a P-token cached prefix (+ the chunk's own causal
part); attention rate A in {8, 12, 20} TFLOP/s (unmeasured for the 3xTF32 kernel); host overhead 0.03 s per forward call.
Decode aggregate (own caches, HF DynamicCache rewrite): <=2k 800 tok/s, 4k 450, 8k 250, 16k 120, 32k 60."""
import sys
LIN = 2.819e9 / 15.8e12; HOST = 0.03
def prefill(T, A): return T * LIN + 114688 * T * T / A + HOST
def chunk(E, P, A): return E * LIN + (229376 * E * P + 114688 * E * E) / A + HOST
def probe(P, A, n=1, size=40): return n * chunk(size, P, A)
def dec_rate(L):
    for lim, r in [(2500, 800), (5000, 450), (9000, 250), (17000, 120), (40000, 60)]:
        if L <= lim: return r
REC = 96          # tokens per record
OV = 256 + 128 + 60   # preamble + spacer + inventory
def run(A):
    out = {}
    # ---------------- D0 pilot (engineering): fixed 15 min
    out['D0'] = 15 * 60
    # ---------------- D1a position kernel: per (item, version): neutral background to ~32k (item-specific, version-free: 1 prefill per item),
    # chunks B-record + (d-1) neutral records + spacer for d in {1,4,16,64,256}; and at 8k d in {1,4,16,64} (separate 8k background per item)
    t = 0; NI = 12
    for it in range(NI):
        t += prefill(31500, A) + prefill(7500, A)
        for v in range(2):
            for d in [1, 4, 16, 64, 256]:
                N = 31500 // REC; P = (N - d) * REC; E = d * REC + 128
                t += chunk(E, P, A) + probe(P + E, A, 4)
            for d in [1, 4, 16, 64]:
                N = 7500 // REC; P = (N - d) * REC; E = d * REC + 128
                t += chunk(E, P, A) + probe(P + E, A, 4)
            # run-length: r in {4,16}, contiguous vs spread (spacing 4), ending near d=1, at 32k
            for r in [4, 16]:
                for span in [r, 4 * r]:
                    P = 31500 - span * REC; E = span * REC + 128
                    t += chunk(E, P, A) + probe(P + E, A, 4)
    out['D1a'] = t
    # D1b copy-bait: 12 items x 2 versions x {AB, Bonly} x L {4k, 32k}: material chunk 3k over filler prefix
    t = 0
    for it in range(12):
        for v in range(2):
            for cell in range(2):
                for L in [4250, 32000]:
                    P = L - 3500; t += chunk(3500, P, A) + probe(L, A, 4)
    out['D1b'] = t
    # ---------------- D3 Lambda (M1): 24 items x alpha {0.2 (30 rec), 0.33 (36 rec)} x 7 cells x L-levels F {250, 4000, 12000, 28000}
    # shared filler prefixes: 3 filler samples nested to 28k (+ shuffled copies at 28k)
    t = 3 * prefill(28500, A) + 3 * prefill(28500, A)   # coherent + shuffled filler prefixes
    for it in range(24):
        for nrec in [30, 36]:
            E = nrec * REC + 60 + 128
            for F in [250, 4000, 12000, 28000]:
                P = 256 + F
                t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
            # shuffled filler check at F 28000
            P = 256 + 28000; t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
    # format check: 30-turn rendering (+360 tokens) x 2 template conventions, alpha 0.2 only, F {250, 28000}
    for it in range(24):
        E = 30 * REC + 360 + 60 + 128
        for conv in range(2):
            for F in [250, 28000]:
                P = 256 + F; t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
    out['D3_M1'] = t
    # D3 M2 (own stance): 16 starts x {pro persona, anti persona, neutral} x F-levels; chunk 1.2k; 6 statement probes (50 tok)
    t = 0
    for s in range(16):
        for cell in range(3):
            for F in [250, 4000, 12000, 28000]:
                P = 300 + F; E = 1200; t += chunk(E, P, A) + probe(P + E, A, 6, 50)
    out['D3_M2'] = t
    # D3 count-vs-share (AB only, 2 versions, at F 12000): 6 conditions, material 1.5-6k
    t = 0
    for it in range(24):
        for v in range(2):
            for nrec in [18, 30, 54, 15, 30, 60]:
                E = nrec * REC + 188; P = 256 + 12000; t += chunk(E, P, A) + probe(P + E, A, 4)
    out['D3_cs'] = t
    # ---------------- D2-S: 16 items; AB nested to 32k for alpha {0.2,0.5} x 2 versions (4); A/B-only to 32k alpha 0.2 (4); A/B-only to 16k alpha 0.5 (4);
    # neither to 32k (1). Reads at 1k,2k,4k,8k,16k(,32k): probes 4 per read
    t = 0
    for it in range(16):
        t += 9 * prefill(32000, A) + 4 * prefill(16000, A)
        t += 9 * sum(probe(L, A, 4) + chunk(128, L, A) for L in [1000, 2000, 4000, 8000, 16000, 32000])
        t += 4 * sum(probe(L, A, 4) + chunk(128, L, A) for L in [1000, 2000, 4000, 8000, 16000])
    out['D2_S'] = t
    # ---------------- D2-R: 16 items x alpha {0.1, 0.2, 0.33}; per (item, alpha): 3 prefixes to (1-alpha)*32k; 7 run chunks of alpha*L over (1-alpha)*L at L {2k, 8k, 32k}
    t = 0
    for it in range(16):
        for a in [0.1, 0.2, 0.33]:
            t += 3 * prefill((1 - a) * 32000, A)
            for L in [2000, 8000, 32000]:
                E = a * L + 128; P = (1 - a) * L
                t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
    out['D2_R'] = t
    # ---------------- D4 catfish/binding: 24 items; parity (16 records) 3 conds x 2 versions x 2 arrangements x F {250, 4000, 28000}; probes 12/8/8
    t = 0
    for it in range(24):
        for cond, npr in [('B', 12), ('C', 8), ('F', 8)]:
            for v in range(2):
                for arr in range(2):
                    for F in [250, 4000, 28000]:
                        E = 16 * REC + 188; P = 256 + F; t += chunk(E, P, A) + probe(P + E, A, npr)
        # dominant 8:24, interleaved only, F {250, 28000}
        for cond, npr in [('B', 12), ('C', 8), ('F', 8)]:
            for v in range(2):
                for F in [250, 28000]:
                    E = 32 * REC + 188; P = 256 + F; t += chunk(E, P, A) + probe(P + E, A, npr)
    out['D4'] = t
    # ---------------- D7 losing game extras: 2 reference chunks (128) + 8 factual probes on D3 (alpha 0.2, F 250 & 28000, 7 cells, 24 items) and D4 parity (all)
    t = 0
    for it in range(24):
        for F in [250, 28000]:
            P = 256 + F + 3000; t += 7 * (2 * chunk(128, P, A) + probe(P, A, 8))
        for F in [250, 4000, 28000]:
            P = 256 + F + 1700; t += 12 * (2 * chunk(128, P, A) + probe(P, A, 8))
    out['D7'] = t
    # ---------------- D8 retokenisation: 16 items; encodings canon (1x), randseg (2.0x), char (4.35x); canonical L {1.5k, 7k}
    # D3-type 2x2 at alpha 0.2 (30 records, 3k canonical) -> 7 cells x 2 versions-free? (7 cells include versions); D4 parity competing + non-competing interleaved (2 conds x 2 v)
    t = 0
    for f in [1.0, 2.0, 4.35]:
        t += 3 * prefill(f * 4000, A)        # filler prefixes per encoding (3 samples), nested to canonical ~4k filler
        for it in range(16):
            for Lc in [1500, 7000]:
                Fc = max(Lc - 3500, 0); P = f * (256 + Fc); E = f * 3000 + 188
                if Lc == 1500:   # material alone at 1.5k is too big: use 12 records (1.2k) at the short level
                    E = f * 1200 + 188; P = f * 256
                t += 7 * (chunk(E, P, A) + probe(P + E, A, 4))
                E2 = f * 1600 + 188; P2 = f * (256 + max(Lc - 2000, 0))
                t += 4 * (chunk(E2, P2, A) + probe(P2 + E2, A, 12))
    out['D8'] = t
    # ---------------- D5 split (generation): per start: prefill L0 + seeds x 3 turns x 64 tokens + probes 6 per seed-turn (50 tok)
    def d5(L0, nseed=16, nturn=3, ntok=64):
        g = nseed * nturn * ntok
        return prefill(L0, A) + g / dec_rate(L0 + nturn * (ntok + 16)) + nseed * nturn * probe(L0 + 200, A, 6, 50) / 1.0
    t = 0
    for pole in range(2):
        for s in range(8):
            t += d5(1000) + d5(8000) + d5(16000)                 # L-arm T 0.7
            t += d5(1000) + d5(16000)                            # L-arm T 1.0 (endpoints)
            t += d5(8000) + d5(8000)                             # T-arm, E-arm (T 0.7)
            t += d5(8000)                                        # template convention (official) at 8k
            t += 2 * d5(1000)                                    # split-detector positive control + reference contexts
    out['D5'] = t
    # ---------------- D6 lifetime: 3 arms x 16 starts x 2 seeds x 2 T x 4096 tokens (context up to ~5k), probes every 512 tokens (3 per checkpoint)
    gen = 3 * 16 * 2 * 2 * 4096; t = gen / 450 + 3 * 16 * 2 * 2 * 8 * probe(3000, A, 3, 40) + 3 * 16 * 2 * 2 * prefill(1100, A)
    out['D6'] = t
    # ---------------- pools: F2 stance pool (768 x 96 tokens at batch 32, ~1000 tok/s) + neutral answers (256 x 96)
    out['pool'] = (768 + 256) * 96 / 1000 + 60
    return out
for A in [8e12, 12e12, 20e12]:
    o = run(A); tot = sum(o.values())
    print(f"attention rate {A/1e12:.0f} TFLOP/s: " + ", ".join(f"{k} {v/60:.1f}" for k, v in o.items()) + f" | work {tot/60:.1f} min = {tot/3600:.2f} h")
print('prefill s at 4k/8k/16k/32k (A=12):', [round(prefill(T, 12e12), 1) for T in [4096, 8192, 16384, 32000]])
