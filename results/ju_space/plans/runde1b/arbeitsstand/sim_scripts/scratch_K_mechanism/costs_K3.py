# Revised cost model for part K (same rates as costs_K.py). Estimates, not measurements. fp32, TF32 off unless stated.
F17, LOG17 = 2.819e9, 0.622e9
F35, LOG35, MTP35 = 2.6e9, 1.02e9, 0.12e9
R = 15.8e12; HOST = 0.020; HOOK = 1.25   # +25 % for H-activation capture and DLA in rollouts
def fwd(tok, logits=False, f=F17, lg=LOG17): return tok * (f + (lg if logits else 0)) / R
def dec(rows, steps, batch, f=F17, lg=LOG17, host=HOST):
    nb = -(-rows // batch); return nb * steps * (host + batch * (f + lg) / R)
def attn(T): return 114688.0 * T * T
M = {}
def add(k, sec, s, note): M[k] = (sec / 60, s, note)
# K_s1
add('K1.screen', fwd(8000*45) + dec(80000, 24, 160) + 60, 'K_s1', 'TriviaQA screening 8,000 q x 10 samples')
add('KS.labels', dec(1200*11, 6, 128) + dec(300*11, 32, 128) + 30, 'K_s1', 'G_AR and G_FAB labels')
add('KS.arcot', dec(1200, 128, 64) + fwd(1200*200, logits=True)*1.3, 'K_s1', 'G_ARcot 1,200 CoT texts + CETT/entropy pass (K8c)')
add('K1.cett', fwd(5100*60, logits=True)*1.3 + 30, 'K_s1', 'CETT extraction ~5,100 responses')
add('K1.l1', 12*60, 'K_s1', 'GPU L1 path, stability, permutations')
add('K1.csel', dec(15*600, 24, 128) + fwd(15*600*45) + 30, 'K_s1', 'C selection with 15 suppression runs')
add('K1.alphaH', dec(6*1400, 28, 128) + fwd(6*1400*60), 'K_s1', 'alpha sweep on H')
add('K1.dmgcand', 200*fwd(80*100) + 200*80*50*LOG17/R, 'K_s1', 'damage of 100 candidate random sets x 2 alphas')
add('K1.alphaR', dec(19*2*700, 28, 128) + fwd(19*2*700*60), 'K_s1', '19 random sets x 2 alphas x 700 rows')
add('K1.pct', fwd(300000)*1.3, 'K_s1', 'WikiText: mu_l, covariance, neuron percentiles')
add('K4a.gen', dec(2400, 128, 64) + fwd(2400*40), 'K_s1', 'probe corpus generation')
add('K4a.acc', fwd(600000)*1.2 + 10*17*2*2048**2*600000/9.0e12, 'K_s1', 'probe accumulators')
add('K4a.eval', 140*8*8000*151936*2048*2/R + 120, 'K_s1', 'held-out probe scoring')
add('KS.dump', 150, 'K_s1', 'front dump write (ids, logprobs, H activations)')
# K_s2
cells = 5920
real = cells*240; foreign = cells*240; cov = 1440*240; var = 2*960*240
add('K6.main', HOOK*dec(real+foreign+cov+var, 1, 128) + fwd((cells*2+1440+1920)*10*34), 'K_s2', 'CL_17 real + foreign + cov-null + 2 variants')
add('K6.rt', 3840*1.0e12/R, 'K_s2', 'round-trip verifier')
add('K6.vjp', 2400*35*F17*3/R + 60, 'K_s2', 'gradient chain G(s)')
add('K6.fr', fwd(240*6*2*120) + 60, 'K_s2', 'Folge vs Reihe answer digits')
add('K6.event', HOOK*dec(2400*240, 1, 128), 'K_s2', 'event-locked CoT reads (2,400 cells, 5 seeds, + foreign)')
add('K7.clamp', HOOK*dec(20*240*2*5*12*2, 1, 128), 'K_s2', 'H + 19 random mean-clamps')
add('K7.boost', HOOK*dec(20*240*2*5*12*2, 1, 128), 'K_s2', 'H + 19 random boosts')
add('K9.A1', dec(3*320*160, 1, 128) + fwd(3*3200*30) + 30, 'K_s2', 'neuron lens on H and 100 random neurons')
add('K9.A2', fwd(3*480*15*70), 'K_s2', 'earlier-intermediate reconstruction')
# K_s3
add('K6c.labels', 2*dec(1200*11, 8, 128, f=F35, lg=LOG35), 'K_s3', 'G_AR labels on Qwen3.5-2B')
add('K6c.wiring', fwd(4000, True, F35, LOG35) + 30, 'K_s3', 'GK7 MTP wiring')
add('K6c.mu', fwd(300000, f=F35)*1.2, 'K_s3', 'means on WikiText')
c35 = 1800
add('K6c.mtp', dec(c35*480, 1, 128, f=MTP35, lg=LOG35, host=0.004), 'K_s3', 'CL_MTP35')
add('K6c.sr', 2*dec(c35*480, 1, 128, f=F35, lg=LOG35), 'K_s3', 'CL_SR35 (x2 risk factor)')
add('K6c.var', 0.25*(dec(c35*480, 1, 128, f=MTP35, lg=LOG35, host=0.004) + 2*dec(c35*480, 1, 128, f=F35, lg=LOG35)), 'K_s3', 're-injection variants')
# K_s4
add('KS.axes', fwd(64*2*4*60)*3 + 60, 'K_s4', 'axes at 10/20/22, u*_b')
add('K2.kv', fwd(96*4*19*2*100), 'K_s4', 'KV-only route, 19 sources')
add('K2.tok', dec(96*4*19*2, 48, 32) + fwd(96*4*19*2*100), 'K_s4', 'token route regenerations + re-reads')
add('K3.ljac', fwd(96*8*4*3*2*100) + 120, 'K_s4', 'L-jac FD (state, foreign, frozen-QK)')
add('K3.route', 1536*100*F17*3/R + 60, 'K_s4', 'routing-map VJPs')
add('K3.bridge', fwd(100*4*8*2*128), 'K_s4', 'J-lens bridge')
cond = fwd(26*4*32*2*128)
add('K5.cond', 9*cond + cond + 3*60 + 9*dec(32, 128, 32) + 3*32*(fwd(4096) + attn(4096)/R), 'K_s4', '9 conditions (QA doubled) + long prefixes')
add('K4a.R1', fwd(600000)*1.1 + 60, 'K_s4', 'R1-direction channel probes')
add('K1.style', dec(21*128, 28, 128) + fwd(21*128*60) + 30, 'K_s4', 'style->H behavioural link')
# K_s5
pre = 300*5*((fwd(400) + attn(400)/R) + (fwd(2000) + attn(2000)/R))
add('K8a', pre + dec(3000, 24, 32) + 60, 'K_s5', 'matched positions and contents')
sess = 640 + 1216 + 256 + 608 + 768
add('K8b', fwd(sess*650) + dec(sess, 12*24, 32)*1.15 + 120, 'K_s5', f'{sess} sessions')
# K_ana
add('Kana', 20*60, 'K_ana', 'statistics and verdicts')
setup = {'K_p': 4, 'K_s1': 9, 'K_s2': 5, 'K_s3': 12, 'K_s4': 8, 'K_s5': 4, 'K_ana': 5, 'K_r': 4}
pilot_work = 10.0; reserve_work = 12.0
tot = {}
for k, (m, s, n) in M.items(): tot[s] = tot.get(s, 0) + m
tot['K_p'] = pilot_work; tot['K_r'] = reserve_work
grand = 0
for s in ['K_p', 'K_s1', 'K_s2', 'K_s3', 'K_s4', 'K_s5', 'K_ana', 'K_r']:
    w = tot[s]; t = setup[s] + 1.5*w; grand += t
    print(f'{s}: work {w:5.1f} min -> {t:5.1f} min with setup+margin ({t/60:.2f} h)')
print(f'GRAND TOTAL {grand/60:.2f} A100-h (incl. pilot and reserve arm)')
print('--- per measurement (work x 1.5) ---')
agg = {}
for k, (m, s, n) in M.items():
    root = k.split('.')[0]
    agg.setdefault((root, s), 0); agg[(root, s)] += m*1.5
for (r, s), v in agg.items(): print(f'{r:6s} {s:6s} {v:6.1f}')
for k, (m, s, n) in M.items(): print(f'  {k:12s} {s:6s} work {m:6.2f}  {n}')
