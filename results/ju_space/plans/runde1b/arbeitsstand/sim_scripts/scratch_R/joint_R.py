"""Part R (revision): joint power of the conjunctive verdict functions under a stated normal noise model (no model run).
Model per block: A_n^L (n = 1..4, L in {RSI_a, RSI_b}) ~ N(mu_n, SD_A^2 C), C = C_loop (x) C_row, C_loop = [[1, rb], [rb, 1]],
C_row exchangeable with rw. p_s <= 0.01 <-> A_n / SD_A >= 2.326; a 90 % bound = estimate -/+ 1.645 x the statistic's sampling SD
(stand-in for the joint bootstrap). PC loop SEA_t (512 pairs) like one RSI loop; RND loops 256 pairs (SD x 1.17), correlation 0.3
with SEA_t (same prompts, other keys). Conditions that are not tests (rel, R_4^cf, D4, SPO_t, RND, damage band) are assumed met.
Run: taskset -c 3 python3 -I joint_R.py > joint_R.json"""
import json, math
import numpy as np

rng = np.random.default_rng(20261113)
Z90, Z01, ZP = 1.645, 2.326, 0.842
SD = {"14": 0.045, "18": 0.033, "6": 0.034}                    # SD of A_n at 512 pairs (mde_R: 0.042-0.051 / 0.030-0.034 / 0.033-0.035)
NS = 40000


def draw(mu, sd, rw, rb, n=NS):
    """mu: [2, 4] means (loops x rows). Returns [n, 2, 4]."""
    Cr = np.full((4, 4), rw); np.fill_diagonal(Cr, 1.0); Cl = np.array([[1.0, rb], [rb, 1.0]])
    C = np.kron(Cl, Cr) * sd ** 2
    X = rng.multivariate_normal(np.asarray(mu, float).ravel(), C, size=n)
    return X.reshape(n, 2, 4)


def stat_sd(fun, mu, sd, rw, rb):
    """Sampling SD of a statistic under the model (the bootstrap's stand-in), from an independent draw."""
    return float(np.nanstd(fun(draw(mu, sd, rw, rb, 20000))))


def abar(X): return X[:, :, 2:4].mean((1, 2))
def abar_abs(X): return np.abs(X[:, :, 2:4]).mean((1, 2))
def E_of(X): return (X[:, 0, 3] + X[:, 1, 3]) / (X[:, 0, 1] + X[:, 1, 1])


def rows_pass(X, rows, lo_bar, sd):
    z = X[:, :, rows] / sd; lo = X[:, :, rows] - Z90 * sd
    return ((z >= Z01) & (lo >= lo_bar)).all(axis=(1, 2))


def manip(X, sd): return rows_pass(X, [0], 0.10, sd)


def likely(mu, sd, rw, rb, tier, zA=Z90, zE=Z90):
    X = draw(mu, sd, rw, rb)
    sA = stat_sd(abar, mu, sd, rw, rb); sden = sd * math.sqrt(2 * (1 + rb)); den = X[:, 0, 1] + X[:, 1, 1]
    sE = stat_sd(E_of, mu, sd, rw, rb)
    den_ok = np.abs(den) - Z90 * sden > 0
    ok = manip(X, sd) & rows_pass(X, [1, 2, 3], -9.0, sd) & (abar(X) - zA * sA >= tier) & den_ok & (E_of(X) - zE * sE >= 0.5)
    return float(ok.mean()), sA, sE


def pc(A, sd, rw, n=NS):
    """P1 (rows 2-4: z >= 2.326 and A_lo >= 0.10) and P5 (A_4(SEA_t) - max RND A_4 lower bound > 0) on one PC loop."""
    Cr = np.full((4, 4), rw); np.fill_diagonal(Cr, 1.0)
    S = rng.multivariate_normal([A] * 4, Cr * sd ** 2, size=n)
    sr = sd * 1.17; R = rng.multivariate_normal([0, 0], np.array([[1, 0.3], [0.3, 1]]) * sr ** 2, size=n)
    rho_sr = 0.3
    # SEA_t row 4 correlated 0.3 with each RND A_4 (shared prompts)
    R = R + rho_sr * (S[:, 3:4] - A) * sr / sd
    diff = S[:, 3] - R.max(1)
    sdiff = float(np.std(diff))
    p1 = ((S[:, 1:4] / sd >= Z01) & (S[:, 1:4] - Z90 * sd >= 0.10)).all(1)
    p5 = diff - Z90 * sdiff > 0
    return float(p1.mean()), float((p1 & p5).mean())


def mde(f, grid, target=0.8):
    for a in grid:
        if f(a) >= target: return round(float(a), 3)
    return None


out = {"model": dict(SD_A=SD, NS=NS, note="normal model; non-test conditions assumed met; bounds use the statistic's sampling SD")}
for b, sd in SD.items():
    r = {}
    for rw in (0.5, 0.8):
        rb = 0.5
        g = np.arange(0.10, 0.70, 0.005)
        # PC: P1 alone and P1 & P5
        r[f"PC_P1_MDE80 rw{rw}"] = mde(lambda a: pc(a, sd, rw)[0], g)
        r[f"PC_P1P5_MDE80 rw{rw}"] = mde(lambda a: pc(a, sd, rw)[1], g)
        # SENS: planted A = 0.25 at rows 2-4 of both RSI loops (P1-type for each loop)
        Xs = draw(np.full((2, 4), 0.25), sd, rw, rb); r[f"SENS_pass_rate_at_0.25 rw{rw}"] = round(float(rows_pass(Xs, [1, 2, 3], 0.10, sd).mean()), 3)
        r[f"SENS_planted_A_for_0.8 rw{rw}"] = mde(lambda a: float(rows_pass(draw(np.full((2, 4), a), sd, rw, rb), [1, 2, 3], 0.10, sd).mean()), g)
        # MANIP_1b: both loops A_1 lo >= 0.10 and z >= 2.326
        r[f"MANIP_MDE80 rw{rw}"] = mde(lambda a: float(manip(draw(np.full((2, 4), a), sd, rw, rb), sd).mean()), g)
        # 'likely' tiers, flat profile (E_true = 1) and decaying profile (E_true = 0.8: A_2 = A, A_3 = 0.9 A, A_4 = 0.8 A, A_1 = A)
        for tier, nm in ((0.25, "substantial"), (0.50, "dominated")):
            for prof, mk in (("flat", lambda a: np.full((2, 4), a)), ("decay0.8", lambda a: np.tile([a, a, 0.9 * a, 0.8 * a], (2, 1)))):
                gg = np.arange(tier, tier + 0.40, 0.005)
                r[f"likely_{nm}_{prof}_MDE80_in_Abar rw{rw}"] = (lambda m: None if m is None else round(float(np.mean(mk(m)[:, 2:4])), 3))(
                    mde(lambda a: likely(mk(a), sd, rw, rb, tier)[0], gg))
                # worst case under Holm (the only true effect among k = 10 'for' hypotheses): bounds at one-sided 0.005
                r[f"likely_{nm}_{prof}_MDE80_Holm_worst rw{rw}"] = (lambda m: None if m is None else round(float(np.mean(mk(m)[:, 2:4])), 3))(
                    mde(lambda a: likely(mk(a), sd, rw, rb, tier, zA=2.576, zE=2.576)[0], gg))
        # 'unlikely (does not persist)': A_1 = 0.30 (first link present), A_2..A_4 = a; P(MANIP and A_bar_abs upper < 0.25)
        def unl(a, zU=Z90):
            mu = np.tile([0.30, a, a, a], (2, 1)); X = draw(mu, sd, rw, rb); s = stat_sd(abar_abs, mu, sd, rw, rb)
            return float((manip(X, sd) & (abar_abs(X) + zU * s < 0.25)).mean())
        gd = np.arange(0.25, -0.05, -0.005)
        r[f"unlikely_persist_largest_true_A_late_for_0.8 rw{rw}"] = mde(unl, gd)
        r[f"unlikely_persist_Holm_worst rw{rw}"] = mde(lambda a: unl(a, 2.576), gd)
        # 'unlikely (no first link)': A_1 hi < 0.10 for both loops
        def nfl(a):
            X = draw(np.full((2, 4), a), sd, rw, rb); return float((X[:, :, 0] + Z90 * sd < 0.10).all(1).mean())
        r[f"no_first_link_P_at_true_A1_0 rw{rw}"] = round(nfl(0.0), 3)
        r[f"no_first_link_largest_true_A1_for_0.8 rw{rw}"] = mde(nfl, np.arange(0.10, -0.10, -0.0025))
        # SD(A_bar), SD(E) at A levels (flat profile)
        for A in (0.25, 0.33, 0.50):
            mu = np.full((2, 4), A)
            r[f"SD_Abar A{A} rw{rw}"] = round(stat_sd(abar, mu, sd, rw, rb), 4)
            r[f"SD_E A{A} rw{rw}"] = round(stat_sd(E_of, mu, sd, rw, rb), 3)
    out[b] = r
# delta rule: SD of A_bar_abs difference between two blocks (prompts shared; correlation across blocks 0.3 assumed)
dl = {}
for (b1, b2) in (("14", "18"), ("14", "6"), ("18", "6")):
    s1 = out[b1]["SD_Abar A0.33 rw0.8"]; s2 = out[b2]["SD_Abar A0.33 rw0.8"]; s = math.sqrt(s1 ** 2 + s2 ** 2 - 2 * 0.3 * s1 * s2)
    dl[f"{b1}-{b2}"] = dict(sd_diff=round(s, 4), MDE80=round((Z90 + ZP) * s, 3))
out["delta_rule"] = dl
print(json.dumps(out, indent=1))
