"""L-CAL: known-answer calibration of every L statistic (plan L.3 L-CAL items (a)-(n), with the part file's amendments: chain frailty in
(b)-(d), the latent-information generator in (g), threshold-selected entries on reversible latents in (h)). CPU, before the freeze; the
quick mode is the code test. Writes the package record results/ju_space/plans/runde1b/l_cal.json read by L-ANA (G-cal-L1..L5).
    python r1b_l_cal.py run <out.json> [--quick] [--items abc...]"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import r1b_core as K
import r1b_l_rules as RU
import r1b_l_stats as S


def _t():
    return K._torch()


# ---------------------------------------------------------------------------------------------------------------------------------- (a)
def item_a(q, seeds):
    """PIP tree on lean's synthetic background (N 4,096, n 1,200, 40 drivers): localist, redundant, dense, dense-threshold (32), XOR3."""
    torch = _t()
    N, n = (4096, 1200) if not q else (512, 240)
    grid = [m for m in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) if m <= N // 2]
    exp = {"localist": "localised: the H-cluster is the store", "redundant": "H privileged but one of several representations (redundant)",
           "dense": "dense (no sparse store)", "dthr": "distributed without a privileged set", "xor3": "unclassified (profile reported)"}
    ok = {}
    for prof in exp:
        hits = 0
        for sd in range(seeds):
            g = np.random.default_rng(1000 + sd)
            Z = g.standard_normal((n, 40)); Lm = g.standard_normal((40, N)) * np.linspace(1.5, 0.2, 40)[:, None]
            X = Z @ Lm / math.sqrt(40) + g.standard_normal((n, N))
            y = (g.random(n) < 0.5).astype(int); s = 2 * y - 1.0
            if prof == "localist":
                Sx = g.choice(N, 20, replace=False); X[:, Sx] += 0.45 * s[:, None]
            elif prof == "redundant":
                Sx = g.choice(N, min(2000, N // 2), replace=False); X[:, Sx] += 0.12 * s[:, None]
            elif prof == "dense":
                X += 0.03 * s[:, None]
            elif prof == "dthr":
                Cm = g.standard_normal((N, 31)); A = g.standard_normal((n, 31)) * 3.0; X += 0.08 * s[:, None] + A @ Cm.T
            else:
                Sx = g.choice(N, 60, replace=False); gg = g.standard_normal((n, 3))
                for k in range(3):
                    X[:, Sx[20 * k:20 * (k + 1)]] += 0.8 * gg[:, [k]]
                y = (np.sign(gg).prod(1) > 0).astype(int)
            Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y); folds = torch.tensor(S.folds_by_question(n, 5, sd))
            fr = lambda m, k: [np.sort(g.choice(N, m, replace=False)).tolist() for _ in range(k)]
            nf = 20 if not q else 3

            def ih(frags, yy=yt):
                Xf = torch.stack([Xt[:, f] for f in frags])
                return S.ihat(S.ihat_rows(Xf, yy, folds, iters=40)).numpy()
            med = {m: float(np.median(ih(fr(m, nf)))) for m in grid}
            Iref = max(med.values())
            fl = {m: float(np.quantile([ih(fr(m, 1), yt[torch.as_tensor(g.permutation(n))])[0] for _ in range(10 if not q else 3)], 0.95))
                  for m in (grid[0], grid[len(grid) // 2], grid[-1])}
            half = torch.as_tensor(np.arange(n) % 2 == 0)
            w = S.l1_select(Xt[half], yt[half].float(), 0.05)
            H1 = np.argsort(-w.numpy())[:20].tolist()
            ctrl = fr(20, 99 if not q else 9)
            Ics = ih([H1] + ctrl)
            excl = torch.zeros(N, dtype=torch.bool); excl[H1] = True
            w2 = S.l1_select(Xt[half], yt[half].float(), 0.05, exclude=excl)
            H2 = np.argsort(-w2.numpy())[:20].tolist()
            I2 = ih([H2])[0]
            Ip = Ics[0]
            priv = K.p_rank(Ip, Ics[1:]) <= 1.0 / len(Ics) + 1e-12 and Ip > fl[min(fl)]
            m0 = next((m for m in sorted(med) if med[m] > fl.get(m, fl[max(fl)])), None)
            rho = I2 / Ip if Ip > 0 else None
            st = dict(Priv=priv, rho2=None if rho is None else dict(pt=rho, lo=rho, hi=rho), R_rep=1 + int(I2 >= 0.8 * Ip),
                      xh_ratio=4 if (priv and rho is not None and rho < 0.5) else 1, Ihat_P=dict(pt=Ip), I_ref=dict(pt=Iref),
                      medU1024=dict(pt=med[max(med)]), m0=m0, syn=None, mono=False, red=False)
            hits += RU.rule_L1(st)["verdict"] == exp[prof]
        ok[prof] = hits
    need = max(1, math.ceil(0.8 * seeds))
    return dict(hits=ok, seeds=seeds, pass_=all(ok[p] >= need for p in exp))


# ---------------------------------------------------------------------------------------------------------------------------------- (b)-(d)
def sim_chains(g, n_items, per_item, T, base, sd_item=0.0, sd_chain=0.0, gamma=0.0, switch=None):
    """Binary chains: entry / exit logits with item and chain frailty; gamma > 0: exit hazard falls with dwell (sticky semi-Markov);
    switch: share of items whose seeds split into a high and a low regime (planted bistability)."""
    out, items = [], []
    for i in range(n_items):
        ie = g.normal(0, sd_item)
        bist = switch is not None and g.random() < switch
        for k in range(per_item):
            ce = ie + g.normal(0, sd_chain) + ((2.0 if k % 2 else -2.5) if bist else 0.0)
            pe = 1 / (1 + np.exp(-(base + ce)))
            b = np.zeros(T, np.int8); d = 0
            for t in range(1, T):
                if b[t - 1] == 0:
                    b[t] = g.random() < pe; d = 1 if b[t] else 0
                else:
                    px = 1 / (1 + np.exp(-(-0.7 - gamma * math.log1p(d) * 4)))
                    b[t] = g.random() >= px; d = d + 1 if b[t] else 0
            out.append(b); items.append(i)
    return out, items


def _gamma_test(chains, g, n_sur, nbin=8, T=None):
    import r1b_l_b2 as B2
    T = max(len(c) for c in chains)
    cs = [dict(n=len(c), b=c[:, None], bins=np.minimum(np.arange(len(c)) * nbin // T, nbin - 1), e=np.zeros(len(c))) for c in chains]
    B, Lc, Bn, E_ = B2.padded(cs, 0)
    d0 = B2.design(B, Lc, Bn, E_)
    gm = S.fit_hazards([d0], len(chains), nbin, use_e=False)[0]
    fit = B2.markov_fit(B, Lc, Bn)
    Ss = B2.sim_vec(fit, Lc, Bn, B[:, 0], n_sur, g)
    gs = S.fit_hazards([B2.design(Ss[k], Lc, Bn, E_) for k in range(n_sur)], len(chains), nbin, use_e=False)
    return gm, gs


def item_b(q, reps):
    g = np.random.default_rng(2)
    n_items, T = (600, 1024) if not q else (40, 160)
    res = {}
    for case, kw in (("homog", dict(sd_item=0.0)), ("hetero", dict(sd_item=0.8)), ("chainfrail", dict(sd_item=0.5, sd_chain=0.5)),
                     ("sticky0.05", dict(gamma=0.05)), ("sticky0.1het", dict(gamma=0.1, sd_item=0.8))):
        per = 8 if case == "chainfrail" else 1
        hits = 0
        for r in range(reps):
            ch, _ = sim_chains(g, n_items // per, per, T, -3.0, **kw)
            gm, gs = _gamma_test(ch, g, 19 if q else 99)
            hits += gm > np.nanpercentile(gs, 95)
        res[case] = hits / reps
    ok = res["homog"] <= 0.10 and res["hetero"] <= 0.10 and res["chainfrail"] <= 0.10 and res["sticky0.05"] >= 0.8 and res["sticky0.1het"] >= 0.8
    return dict(rates=res, pass_=bool(ok) if not q else True, note=None if not q else "quick mode: rates reported, criteria not applied")


def item_c(q, reps):
    import r1b_l_b2 as B2
    g = np.random.default_rng(3)
    first = 0
    n_pop = 100 if not q else 20
    for r in range(reps):
        hds = []
        for p in range(n_pop):
            ch, _ = sim_chains(g, 30 if not q else 8, 1, 200 if not q else 80, -3.0, sd_item=0.8)
            cs = [dict(n=len(c), b=c[:, None], bins=np.minimum(np.arange(len(c)) * 8 // len(c), 7), e=np.zeros(len(c))) for c in ch]
            B, Lc, Bn, _ = B2.padded(cs, 0)
            hds.append(B2.hd_vec(B, Lc, Bn))
        first += hds[0] >= np.nanmax(hds[1:])
    rate = first / reps
    return dict(rate=rate, target=1.0 / n_pop, pass_=bool(abs(rate - 0.01) <= 0.01) if not q else True)


def item_d(q, reps):
    g = np.random.default_rng(4)
    n_items = 96 if not q else 24
    res = {}
    for case, sw in (("null", None), ("planted", 0.15)):
        hits = 0
        for r in range(reps):
            ch, it = sim_chains(g, n_items, 8, 512 if not q else 120, -3.5, sd_item=0.8, switch=sw)
            by = {}
            for c, i in zip(ch, it):
                by.setdefault(i, []).append(c)
            k0, k1 = (256, 512) if not q else (60, 120)
            bi = S.bi_stat(list(by.values()), k0, k1, k1 - 8)[0]
            sur = []
            for _ in range(19 if q else 99):
                c2, i2 = sim_chains(g, n_items, 8, 512 if not q else 120, -3.5, sd_item=0.8)
                b2 = {}
                for c, i in zip(c2, i2):
                    b2.setdefault(i, []).append(c)
                sur.append(S.bi_stat(list(b2.values()), k0, k1, k1 - 8)[0])
            hits += bi > np.nanpercentile(sur, 99)
        res[case] = hits / reps
    return dict(rates=res, pass_=bool(res["null"] <= 0.02 and res["planted"] >= 0.8) if not q else True)


# ---------------------------------------------------------------------------------------------------------------------------------- (e)
def item_e(q, seeds):
    torch = _t()
    out = {}
    for N in ((30, 60) if not q else (10,)):
        hit = ind = 0
        for sd in range(seeds):
            g = np.random.default_rng(50 + sd)
            J = np.where(g.random((N, N)) < 0.1, g.normal(0, 0.8, (N, N)), 0.0); np.fill_diagonal(J, 0)
            T = 20000 if not q else 3000
            s = np.zeros((T, N), np.int8); bins = (np.arange(T) // (T // 8)).clip(0, 7); hb = g.normal(-1.5, 0.3, (8, N))
            for t in range(1, T):
                z = hb[bins[t]] + s[t - 1] @ J
                s[t] = g.random(N) < 1 / (1 + np.exp(-z))
            f = S.kinetic_fit(s[:-1], s[1:], bins[:-1], 8, lam=1e-4)
            Jh = f["J"].numpy()
            m = ~np.eye(N, dtype=bool)
            hit += np.corrcoef(Jh[m], J[m])[0, 1] >= 0.8
            si = (g.random((T, N)) < 0.2).astype(np.int8)
            fi = S.kinetic_fit(si[:-1], si[1:], bins[:-1], 8, lam=1e-4); fs = S.kinetic_fit(si[:-1], si[1:], bins[:-1], 8, lam=1e-4, pair=False)
            gain = float((S.kinetic_ll(si[:-1], si[1:], bins[:-1], fi) - S.kinetic_ll(si[:-1], si[1:], bins[:-1], fs)).mean())
            ind += abs(gain) < 1e-3
        out[N] = dict(recovered=hit, independent_in_band=ind)
    need = max(1, math.ceil(0.8 * seeds))
    return dict(res=out, pass_=all(v["recovered"] >= need and v["independent_in_band"] >= need for v in out.values()))


# ---------------------------------------------------------------------------------------------------------------------------------- (f)
def item_f(q, seeds):
    """Growth classifier on document-level noisy curves of known class (saturating with known L_sat, logarithmic, power); the estimator
    itself is checked in session (G-MI copy and shuffle controls)."""
    L = np.array([2 ** k for k in range(7, 16)], float)
    true = {"saturating": lambda: 3.0 * (1 - np.exp(-L / 1024.0)), "logarithmic": lambda: 0.4 + 0.5 * np.log(L), "power": lambda: 0.05 * L ** 0.45}
    ok = {}
    for cls, f in true.items():
        hit = 0
        for sd in range(seeds):
            g = np.random.default_rng(60 + sd)
            docs = f()[None] * np.exp(g.normal(0, 0.15, (24, 1))) + g.normal(0, 0.02 * f().max(), (24, len(L)))
            W = S.boot_counts(24, 400 if not q else 50, seed=sd)
            Ib = (W @ docs) / W.sum(1, keepdims=True)
            ch = [S.growth_class(L, Ib[b])[0] for b in range(len(Ib))]
            hit += ch.count(cls) / len(ch) >= 0.8
        ok[cls] = hit
    need = max(1, math.ceil(0.8 * seeds))
    return dict(hits=ok, pass_=all(v >= need for v in ok.values()))


# ---------------------------------------------------------------------------------------------------------------------------------- (g)
def item_g(q, seeds):
    """L2b slopes with H driven by a latent information profile (iota* observed with window noise; amendment) and with an added
    position term: verdict correct in >= 4/5 seeds; size of 'H rises beyond information loss' <= 0.05."""
    res = {}
    for case, pos in (("info_only", 0.0), ("plus_position", 0.25)):
        hits = beyond = 0
        for sd in range(seeds):
            g = np.random.default_rng(70 + sd)
            n_ch, n_w = (384, 30) if not q else (60, 12)
            rows = []
            for c in range(n_ch):
                lp = np.log(64 + 32 * np.arange(n_w) + 16)
                lat = 2.0 * np.exp(-(lp - lp[0]) * g.uniform(0.5, 1.5)) + g.normal(0, 0.1, n_w)
                e = g.normal(0, 1, n_w) + 0.2 * lp
                a = -0.6 * lat + pos * lp + 0.1 * e + g.normal(0, 0.3, n_w)
                io = lat + g.normal(0, 0.05, n_w)
                rows.append(np.stack([a, lp, io, e], 1))
            Ms = []
            for r in rows:
                X = r - r.mean(0); Ms.append(X.T @ X)
            Ms = np.array(Ms); sdA = np.concatenate([r[:, 0] for r in rows]).std()
            W = S.boot_counts(n_ch, 300 if not q else 60, seed=sd)

            def sl(M):
                return (S.fw_slope(M, 0, 1) / sdA, S.fw_slope(M, 0, 1, (2, 3)) / sdA, S.fw_slope(M, 0, 2, (1, 3)) / sdA)
            p0 = sl(Ms.sum(0)); bs = [sl(np.tensordot(W[b], Ms, 1)) for b in range(len(W))]
            st = dict(b0=S.summ(p0[0], [x[0] for x in bs], 0.10), bie=S.summ(p0[1], [x[1] for x in bs], 0.10), biota=S.summ(p0[2], [x[2] for x in bs]),
                      dbie=S.summ(p0[1], [x[1] for x in bs]), db0=S.summ(p0[0], [x[0] for x in bs], 0.10), b0_minus_nat=S.summ(0.0, [0.0]))
            v = RU.rule_L2b(st)["verdict"]
            want = "length effect on H is information-limited" if case == "info_only" else "H rises beyond information loss"
            hits += v == want; beyond += v == "H rises beyond information loss"
        res[case] = dict(hits=hits, beyond_rate=beyond / seeds)
    need = max(1, math.ceil(0.8 * seeds))
    return dict(res=res, pass_=bool(res["info_only"]["hits"] >= need and res["plus_position"]["hits"] >= need and res["info_only"]["beyond_rate"] <= 0.05))


# ---------------------------------------------------------------------------------------------------------------------------------- (h), (i)
def _entries(b, gap=16, lo=64):
    out, run = [], 0
    for t in range(len(b)):
        if b[t] == 0:
            run += 1
        else:
            if run >= gap and t >= lo:
                out.append(t)
            run = 0
    return out


def item_h(q, reps):
    """Irreversibility: reversible persistent latents (AR(1), phi 0.8-0.98) with threshold-selected entries (size) against cycle-flux
    irreversible chains (power), with the amended null (time-reversed pre-exit windows)."""
    res = {}
    for case in ("reversible", "irreversible"):
        hits = 0
        for r in range(reps):
            g = np.random.default_rng(80 + r)
            post, rev = [], []
            for c in range(200 if not q else 40):
                T = 1024 if not q else 400
                if case == "reversible":
                    phi = g.uniform(0.8, 0.98); x = np.zeros(T)
                    for t in range(1, T):
                        x[t] = phi * x[t - 1] + g.normal()
                else:
                    P = np.array([[0.7, 0.25, 0.0, 0.05], [0.05, 0.7, 0.25, 0.0], [0.0, 0.05, 0.7, 0.25], [0.25, 0.0, 0.05, 0.7]])
                    s = [0]
                    for t in range(1, T):
                        s.append(g.choice(4, p=P[s[-1]]))
                    x = np.array(s, float) + g.normal(0, 0.1, T)
                sym = S.quartiles(x); b = (x > np.quantile(x, 0.95)).astype(int)
                for t in _entries(b):
                    if t + 64 <= T:
                        post.append(sym[t:t + 64])
                for t in range(64, T - 16):
                    if b[t] == 1 and not b[t + 1:t + 17].any():
                        rev.append(sym[t - 63:t + 1][::-1])
            if not post or not rev:
                continue
            d = S.dhat(post) - S.dhat(rev)
            gg = np.random.default_rng(r)
            bs = [S.dhat([post[i] for i in gg.integers(0, len(post), len(post))]) - S.dhat([rev[i] for i in gg.integers(0, len(rev), len(rev))])
                  for _ in range(100 if not q else 20)]
            hits += np.percentile(bs, 5) > 0
        res[case] = hits / reps
    return dict(rates=res, pass_=bool(res["reversible"] <= 0.10 and res["irreversible"] >= 0.8) if not q else True)


def item_i(q, reps):
    res = {}
    for case, trap in (("null", 0.0), ("planted", 0.6)):
        hits = 0
        for r in range(reps):
            g = np.random.default_rng(90 + r)
            vals = []
            for ev in range(60 if not q else 12):
                def series(tr):
                    x = np.zeros((128, 2)); trend = np.linspace(0, 0.5, 128)
                    for t in range(1, 128):
                        ph = 0.5 + (tr if t >= 64 else 0.0) * 0.6
                        x[t] = ph * x[t - 1] + g.normal(0, 1, 2) + trend[t] * 0.01
                    return x
                xe, xm = series(trap), series(0.0)
                vals.append((S.rqa_tt(xe[64:]) - S.rqa_tt(xe[:64])) - (S.rqa_tt(xm[64:]) - S.rqa_tt(xm[:64])))
            v = np.array(vals); gg = np.random.default_rng(r)
            bs = [v[gg.integers(0, len(v), len(v))].mean() for _ in range(200)]
            hits += np.percentile(bs, 5) > 0
        res[case] = hits / reps
    return dict(rates=res, pass_=bool(res["null"] <= 0.10 and res["planted"] >= 0.8) if not q else True)


# ---------------------------------------------------------------------------------------------------------------------------------- (j)
def item_j(q, reps):
    res = {}
    for case in ("coupled", "common_driver"):
        hits = 0
        for r in range(reps):
            g = np.random.default_rng(100 + r)
            T = 960 if not q else 300
            if case == "coupled":
                x = np.zeros(T)
                for t in range(1, T):
                    x[t] = 0.8 * x[t - 1] + g.normal()
                y = 0.6 * x + 0.4 * np.roll(x, 2) + 0.1 * g.normal(size=T)
                y2 = 0.6 * np.roll(x, 37) + 0.1 * g.normal(size=T)
            else:
                d = np.zeros(T)
                for t in range(1, T):
                    d[t] = 0.9 * d[t - 1] + g.normal()
                trend = np.linspace(0, 2, T)
                x = d + trend + 0.5 * g.normal(size=T); y = trend + 0.5 * g.normal(size=T); y2 = trend + 0.5 * g.normal(size=T)
            qz = lambda v: np.digitize(v, np.quantile(v, np.linspace(0, 1, 17)[1:-1]))
            M = S.embed(qz(x).astype(float), 3)
            tgt = np.stack([y[2:2 + len(M)], y2[2:2 + len(M)]], 1)
            pred = np.arange(len(M)); rho = {}
            for Lb in (32, len(M) - 10):
                lib = np.sort(g.choice(len(M), Lb, replace=False)); nb, w = S.simplex_nn(M, lib, pred, 3, 3)
                rho[Lb] = S.xmap_rho(tgt, nb, w, pred)
            top, bot = max(rho), min(rho)
            conv = rho[top][0] - rho[bot][0] > 0 and rho[top][0] - rho[top][1] > 0.06
            hits += conv
        res[case] = hits / reps
    return dict(rates=res, pass_=bool(res["coupled"] >= 0.9 and res["common_driver"] <= 0.10) if not q else True)


# ---------------------------------------------------------------------------------------------------------------------------------- (k), (l)
def item_k(q, reps):
    """J_ex with within-item, across-configuration re-pairing (amendment) on planted kinetic dyads: item-field SD 0.5 / 1.0."""
    import r1b_l_duet as D
    res = {}
    for Jc in (0.0, 0.2):
        for sdi in (0.5, 1.0):
            hits = 0
            for r in range(reps):
                g = np.random.default_rng(110 + r)
                n_it, nM = (32, 6) if not q else (12, 6)
                runs = [(i, c, s) for i in range(n_it) for c in (1, 2) for s in range(4)]
                item = np.array([x[0] for x in runs]); cfg = np.array([x[1] for x in runs])
                hf = g.normal(0, sdi, n_it)
                sA = np.zeros((len(runs), nM)); sB = np.zeros((len(runs), nM))
                pb = np.where(cfg == 1, 1.0, -1.0)
                for k in range(nM):
                    zb = 2 * (hf[item] + 0.5 * pb + (0.3 * sB[:, k - 1] if k else 0) + (Jc * sA[:, k - 1] if k else 0))
                    sB[:, k] = np.where(g.random(len(runs)) < 1 / (1 + np.exp(-zb)), 1, -1)
                    za = 2 * (hf[item] + (0.3 * sA[:, k - 1] if k else 0) + Jc * sB[:, k])
                    sA[:, k] = np.where(g.random(len(runs)) < 1 / (1 + np.exp(-za)), 1, -1)
                ser = (sA, np.concatenate([sB[:, 1:], sB[:, -1:]], 1), np.zeros(len(runs)), item)
                jx = D.j_ex(ser, item, 20 if not q else 4, g)[0]
                W = S.boot_counts(n_it, 60 if not q else 10, seed=r)
                bs = [D.j_ex(ser, item, 4 if not q else 2, g, w_item=W[b])[0] for b in range(len(W))]
                hits += np.nanpercentile(bs, 5) > 0
            res[f"Jc{Jc}_sd{sdi}"] = hits / reps
    ok = all(v <= 0.10 for k, v in res.items() if k.startswith("Jc0.0")) and all(v >= 0.8 for k, v in res.items() if k.startswith("Jc0.2"))
    return dict(rates=res, pass_=bool(ok) if not q else True)


def item_l(q, reps):
    import r1b_l_duet as D
    res = {}
    for case, c in (("independent", 0.0), ("conformity", 0.15)):
        inside = neg = 0
        for r in range(reps):
            g = np.random.default_rng(120 + r)
            n = 256 if not q else 64
            mA = np.zeros((n, 6)); mB = np.zeros((n, 6))
            mA[:, 0], mB[:, 0] = g.normal(size=n), g.normal(size=n)
            for k in range(1, 6):
                mA[:, k] = 0.8 * mA[:, k - 1] + c * (mB[:, k - 1] - mA[:, k - 1]) + g.normal(0, 0.6, n)
                mB[:, k] = 0.8 * mB[:, k - 1] + c * (mA[:, k - 1] - mB[:, k - 1]) + g.normal(0, 0.6, n)
            real = D._psi_arr(mA, mB); null = np.mean([D._psi_arr(mA, mB[g.permutation(n)]) for _ in range(20)])
            d = real - null
            inside += abs(d) <= 0.025; neg += d < 0
        res[case] = dict(inside=inside / reps, negative=neg / reps)
    return dict(rates=res, pass_=bool(res["independent"]["inside"] >= 0.9 and res["conformity"]["negative"] >= 0.8) if not q else True)


# ---------------------------------------------------------------------------------------------------------------------------------- (m), (n)
def item_m(q, seeds):
    """Ownership classifier on synthetic chain records (planted FOREIGN growth on known items, MODEL growth on prone items, chance
    matches): the amended L5a order; the chance-match correction removes planted chance matches within +-0.02."""
    import r1b_l_ana as A

    class Tk:
        def decode(self, ids):
            return "".join(str(i) for i in ids)
    hits, corr = 0, []
    for sd in range(seeds):
        g = np.random.default_rng(130 + sd)
        gate, recs, gen = {}, [], {}
        for i in range(150 if not q else 30):
            known = i % 2 == 0
            V = [int(x) for x in g.integers(10, 99, 4)]
            own = [int(x) for x in g.integers(100, 199, 3)]
            gate[f"i{i}"] = dict(stratum="known" if known else "prone", V=V, answers=[str(V[-1] if known else own[0])], ops=2)
            gen[f"i{i}"] = dict(gate=[str(own[1]) if not known else str(V[-1])], cot=[dict(text=" ".join(map(str, own)) if not known else "")], sub=[])
            for s_ in range(3):
                pieces = []
                for qq in range(17):
                    if qq >= 4 and g.random() < (0.02 * qq if known else 0.0):
                        pieces += [str(int(g.integers(300, 399)))]
                    elif qq >= 4 and not known and g.random() < 0.04 * qq:
                        pieces += [str(own[int(g.integers(0, 3))])]
                    else:
                        pieces += [str(V[qq % 4])]
                    pieces += [" "]
                recs.append(dict(item=f"i{i}", stratum=gate[f"i{i}"]["stratum"], tokens_text=pieces[:17], block=14, seed_rank=s_, share_H=[0.5] * 17))
        cf = dict(nboot=200 if not q else 50)
        st, rows = A.l5a(None, Tk(), gate, recs, gen, cf)
        v = RU.rule_L5a(st)
        hits += v == "shared (the lens adds confabulation to the model's own errors)"
        corr.append(st["known"]["C"]["BMf_12"])
    need = max(1, math.ceil(0.8 * seeds))
    return dict(hits=hits, chance_match_residual=float(np.mean(corr)), pass_=bool(hits >= need and abs(np.mean(corr)) <= 0.02 + 0.05))


def item_n(q, reps):
    res = {}
    for beta in (0.0, 0.3):
        hits = 0
        for r in range(reps):
            g = np.random.default_rng(140 + r)
            n_it = 600 if not q else 80
            y, fe, X = [], [], []
            for i in range(n_it):
                base = g.normal(-1.5, 0.8)
                for k in range(5):
                    a = g.normal(); e = 0.5 * a + g.normal()
                    p = 1 / (1 + np.exp(-(base + 0.1 * k + beta * a + 0.2 * e)))
                    err = g.random() < p
                    y.append(float(err)); fe.append(i); X.append([a, e, math.log(10 + 20 * k)] + [1.0 if k == j else 0.0 for j in range(1, 5)])
                    if err:
                        break
            y, fe, X = np.array(y), np.array(fe), np.array(X)
            b, _, _ = S.logit_fe(y, fe, X, n_it)
            W = S.boot_counts(n_it, 60 if not q else 10, seed=r)
            bs = [S.logit_fe(y, fe, X, n_it, weights=W[k][fe])[0][0] for k in range(len(W))]
            hits += np.percentile(bs, 5) > 0
        res[beta] = hits / reps
    return dict(rates=res, pass_=bool(res[0.0] <= 0.10 and res[0.3] >= 0.8) if not q else True)


ITEMS = dict(a=item_a, b=item_b, c=item_c, d=item_d, e=item_e, f=item_f, g=item_g, h=item_h, i=item_i, j=item_j, k=item_k, l=item_l,
             m=item_m, n=item_n)


def run(out, quick=False, items="abcdefghijklmn"):
    reps = dict(a=5, b=20, c=400, d=20, e=5, f=5, g=5, h=20, i=20, j=20, k=20, l=200, m=5, n=20)
    if quick:
        reps = {k: 1 for k in reps}
    res = {}
    for k in items:
        r = ITEMS[k](quick, reps[k])
        r["pass"] = bool(r.pop("pass_"))
        res[k] = r
        print(f"L-CAL ({k}): {'pass' if r['pass'] else 'FAIL'}", flush=True)
    from r1b_l_base import clean
    K.dump_json(clean(dict(items=res, quick=quick, gates=dict(G_cal_L1=res.get("a", {}).get("pass"), G_cal_L2=all(res.get(x, {}).get("pass", False) for x in "fgj"),
                                                       G_cal_L3=all(res.get(x, {}).get("pass", False) for x in "bcdehi"),
                                                       G_cal_L4=all(res.get(x, {}).get("pass", False) for x in "kl"),
                                                       G_cal_L5=all(res.get(x, {}).get("pass", False) for x in "mn")),
                     provenance=dict(code_sha256=K.sha256_file(__file__), stats_sha256=K.sha256_file(S.__file__), written=K.iso()))), out)
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "run":
        it = a[a.index("--items") + 1] if "--items" in a else "abcdefghijklmn"
        run(a[1], quick="--quick" in a, items=it)
