"""A100 cost estimate for part D (lean design). No model is run; arithmetic only.

Rates (fp32, TF32 off, SDPA memory-efficient kernel; from the int:machinery report and the round-2 plan 7.1):
  linear 2.819 GFLOP/token at 15.8 TFLOP/s (plan-implied, optimistic);
  attention 229,376 FLOP per (query token x key) at 8-20 TFLOP/s (unmeasured for the 3xTF32 kernel) -> use 10 TFLOP/s;
  per-call overhead 0.03 s; decode aggregate at 32k with HF DynamicCache and B = 3 rows: 60 tok/s (int:machinery 3.6);
  short-context generation 1,000 new tok/s at batch 32 (plan R-gen 1,280; 985 with 80-token prompts);
  short teacher-forced reads ~5,000 tok/s batched (plan R-full).
"""
LIN, ATT, ATT_C, OVH = 2.819e9, 15.8e12, 229376, 0.03
ATT_RATE = 10e12


def chunk_s(c, P):
    """chunk of c new tokens over a cached prefix of P tokens"""
    return c * LIN / ATT + ATT_C * c * (P + c / 2) / ATT_RATE + OVH


def context_s(L, block=1050, q=45, nq=6):
    P = max(L - block - nq * 0, 0)
    s = chunk_s(block, P)
    s += nq * chunk_s(q, P + block)
    return s


def prefill_s(L):
    return chunk_s(L, 0)


if __name__ == '__main__':
    print('per-context time (s): entity block chunk + 6 question chunks over the shared filler prefix')
    lens = {'L0~1.3K': 150, '4K': 4096, '16K': 16384, '32K': 32000}
    per = {}
    for name, L in lens.items():
        # at L0 the 'prefix' is ~150 tokens of filler
        per[name] = context_s(L if L > 1300 else 1200)
        print(f'  {name:8s}: {per[name]:.2f} s  (prefix prefill once per stream: {prefill_s(L):.1f} s)')
    # D1 design
    I_s, I_a, I_r = 48, 24, 16
    main = I_s * (2 * (1 + 5 * 2) + 2 * 5)          # interleaved 22 + release 10 per item per L
    count = I_s * (2 * 2 * 2)                        # 2 half-density conditions x 2 swaps x 2 orders
    asym = I_a * (2 * (1 + 5 * 2))                   # interleaved only
    retok = I_r * (2 * 6) * 2                        # 1 order, k 0..5, 2 swaps, 2 encodings
    print(f'contexts per L: main {main}, count {count}, asym {asym}, retok {retok}')
    t = {}
    t['D1 main'] = sum(main * per[n] for n in lens) / 60
    t['D1c count'] = sum(count * per[n] for n in ('L0~1.3K', '32K')) / 60
    t['D1e asym'] = sum(asym * per[n] for n in ('L0~1.3K', '32K')) / 60
    retok_32 = retok * context_s(32000, block=1350) / 60
    retok_0 = retok * context_s(1500, block=1350) / 60
    t['D1r retok'] = retok_32 + retok_0
    t['D1 prefix prefills'] = 4 * sum(prefill_s(L) for L in (4096, 16384, 32000)) / 60
    # D0a parity: 8 contexts x full recompute at 16K and 32K (+ cached versions already in D1)
    t['D0a parity'] = 8 * (prefill_s(16384) + prefill_s(32000)) * 1.2 / 60 + 3
    # D2
    P, cond, turns, new = 192, 6, 3, 80
    texts = P * cond
    tok_turn = texts * new
    short = turns * tok_turn / 1000 / 60
    long_dec = turns * tok_turn / 60 / 60
    # per batch of 3 rows at 32K: copy prefix cache (~0.05 s) + suffix chunk of <= 400 tokens over 31.5K
    long_pref = turns * (texts / 3) * (0.05 + chunk_s(300, 31500)) / 60
    probes = (texts * turns * 2) * (130 + 8 * 35) / 5000 / 60
    reads = (texts * turns * 2) * 130 / 9000 / 60
    nll = (texts * turns * 2) * 130 / 5000 / 60
    t['D2 short gen'] = short
    t['D2 long gen (decode)'] = long_dec
    t['D2 long gen (prefill/copies)'] = long_pref
    t['D2 probes+reads+NLL'] = probes + reads + nll
    t['D0b decode parity'] = 6
    tot = 0
    for k, v in t.items():
        tot += v
        print(f'{k:30s} {v:6.1f} min')
    print(f'{"raw total":30s} {tot:6.1f} min = {tot / 60:.2f} h; x1.5 margin = {1.5 * tot / 60:.2f} h; + setup 4 x 8 min -> {(1.5 * tot + 32) / 60:.2f} h')
