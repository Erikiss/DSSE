"""STAT-1b (part R, R.1 / R.2): model-free statistics. Joint prompt bootstrap of axes and pairs (weights W [B, P]: every text of a
prompt carries its prompt's weight; axes re-estimated per replicate with u*_b and u_neg fixed), cross-fitted fold axes and
disattenuated A_n = c_n^cf / sqrt(rel_n kappa_X), K1-s (paired sign-flip test over prompts, 10,000 flips, seed 2001; bootstrap-t
fallback seed 2002, RG6), K1-c centred pseudo-axes (seeds 311-314), cross-fitted S_4 / R_4 and loop / mode cosines over the 20 K2
halvings, the SENS plant, a normal-model joint-MDE simulator, interval / p helpers. Never imports rk_loop."""
import math
import numpy as np
import r1b_core as K
import rk_verdict

FLOOR_REL, FLOOR_KAPPA = 0.05, 0.5
SEED_FLIP, SEED_BT, B_FLIP = 2001, 2002, 10000
OTHERS = {"rsi": ("sports", "tone", "sea"), "sports": ("rsi", "tone", "sea"), "tone": ("rsi", "sports", "sea"), "sea": ("rsi", "sports", "tone")}


def T():
    return K._torch()


def unit(v):
    return v / (v.norm(dim=-1, keepdim=True) + 1e-12)


def cos(a, b):
    a, b = T().as_tensor(a).double(), T().as_tensor(b).double()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


def halvings(P):
    """The 20 K2 halvings of rk_verdict.halvings(P) (seed 3; RSI-topic prompts are the first half)."""
    return [T().as_tensor(h) for h in rk_verdict.halvings(P)]


def rel_of(r):
    if not math.isfinite(r):
        return float("nan")
    return 0.05 if r <= -1 else max(2 * r / (1 + r), FLOOR_REL)


def iv(draws):
    """90 % percentile interval (SO-PVAL; coded when > 5 % of the replicates are undefined)."""
    return K.interval(np.asarray(draws, dtype=float))


def num(x):
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


# ------------------------------------------------------------------------------------------------------------------ axes (AX-1b)
class AxisSet:
    """Per topic X the per-prompt differences D_X [P, D] of the axis texts (NaN-free rows; keep marks prompts with both texts), fold
    [P], u*_b and u_neg fixed. Axis names '<X>_<variant>' with X in T, sports, tone, sea and variant raw | 1b (minus span(u_neg, other
    three)) | ps (perp*: minus span(u*_b, u_neg, other three)) | v (verdict: ps at 14 / 18, 1b at 6); extra: {name: (base name, [fixed
    vectors also projected out])}. get(W, names) -> {name: dict(u, aA, aB, kappa)} with [B, D] / [B] tensors (W [B, P] prompt weights)."""
    XT = {"T": "rsi", "sports": "sports", "tone": "tone", "sea": "sea"}

    def __init__(self, diff, keep, fold, ustar, uneg, block, dev="cpu", extra=None):
        t = T(); self.dev = dev
        self.Dm = {x: t.nan_to_num(t.as_tensor(diff[x]).double()).to(dev) for x in diff}
        self.keep = {x: t.as_tensor(keep[x], dtype=t.bool).to(dev) for x in keep}
        self.fold = t.as_tensor(fold, dtype=t.bool).to(dev); self.us = t.as_tensor(ustar).double().to(dev)
        self.un = t.as_tensor(uneg).double().to(dev); self.block = int(block); self.extra = extra or {}

    def _mean(self, W, x, mask=None):
        k = self.keep[x].double() * (1.0 if mask is None else mask.double())
        Wk = W * k[None]; return (Wk @ self.Dm[x]) / Wk.sum(1, keepdim=True).clamp_min(1e-12)

    def get(self, W, names):
        t = T(); W = t.as_tensor(W, dtype=t.float64).to(self.dev); B = W.shape[0]; out = {}
        raw = {x: self._mean(W, self.XT[x]) for x in self.XT}
        fA = {x: self._mean(W, self.XT[x], self.fold) for x in self.XT}; fB = {x: self._mean(W, self.XT[x], ~self.fold) for x in self.XT}
        for nm in names:
            base, vecs = self.extra.get(nm, (nm, []))
            X, var = base.rsplit("_", 1)
            if var == "v":
                var = "1b" if self.block == 6 else "ps"
            if var == "raw" and not vecs:
                u, a, b = raw[X], unit(fA[X]), unit(fB[X])
            else:
                cols = ([] if var == "raw" else [raw[o] for o in self.XT if o != X] + [self.un.expand(B, -1)]) + \
                       ([self.us.expand(B, -1)] if var == "ps" else []) + [t.as_tensor(v).double().to(self.dev).expand(B, -1) for v in vecs]
                Fm = t.stack([unit(c) for c in cols], -1); Q, _ = t.linalg.qr(Fm)
                pr = lambda v: v - (Q @ (Q.transpose(1, 2) @ v[..., None]))[..., 0]
                u, a, b = pr(raw[X]), unit(pr(fA[X])), unit(pr(fB[X]))
            out[nm] = dict(u=u, aA=a, aB=b, kappa=(a * b).sum(-1))
        return out


def centred_nulls(diff, keep, seed, Q=None, n=2000):
    """K1-c: n centred pseudo-axes E_k (D - mean D) / N over the kept prompts, E +-1 from torch.Generator(seed) (R.1), projected off
    span(Q) like the axis."""
    t = T(); Dk = t.as_tensor(diff).double()[t.as_tensor(keep, dtype=t.bool)]; N = len(Dk)
    E = (t.randint(0, 2, (n, N), generator=t.Generator().manual_seed(int(seed))) * 2 - 1).double()
    nl = E @ (Dk - Dk.mean(0)) / N
    if Q is not None:
        Q = t.as_tensor(Q).double(); nl = nl - (nl @ Q) @ Q.T
    return nl


def k1c(d, axis, null):
    """p_c = (1 + #{|cos(d, null_k)| >= |c|}) / (n + 1), q99_c, Q_c = |c| / q99_c."""
    t = T(); d = t.as_tensor(d).double(); c = cos(d, axis); nc = (null @ d).abs() / (null.norm(dim=1) * d.norm() + 1e-12)
    q99 = float(t.quantile(nc, 0.99))
    return dict(c=c, p_c=float((1 + (nc >= abs(c)).sum()) / (1 + len(nc))), q99_c=q99, Q_c=abs(c) / q99 if q99 > 0 else K.coded("NaN:zero_null"))


# ------------------------------------------------------------------------------------------------------------------ rows (loop evaluations)
class Row:
    """One evaluation's ok pairs reduced to per-prompt sums S [P, D] and counts n [P] (delta_j = (x_j+ - x_j-) / 2)."""

    def __init__(self, delta, ok, prompt_of, P, dev="cpu"):
        t = T(); dl = t.as_tensor(delta).double(); ok = t.as_tensor(ok, dtype=t.bool); po = t.as_tensor(prompt_of).long()
        dl = t.nan_to_num(dl); self.P, self.D = P, dl.shape[1]
        self.S = t.zeros(P, self.D, dtype=t.float64).index_add_(0, po[ok], dl[ok]).to(dev)
        self.n = t.bincount(po[ok], minlength=P).double().to(dev); self.dev = dev; self.delta, self.ok, self.po = dl, ok, po

    def d(self, W=None):
        t = T()
        if W is None:
            return self.S.sum(0) / self.n.sum().clamp_min(1e-12)
        W = t.as_tensor(W, dtype=t.float64).to(self.dev); return (W @ self.S) / (W @ self.n).clamp_min(1e-12)[:, None]

    def halves(self, H, W=None):
        """[(mA, mB)] per halving: the ok-pair means of each half ([D] or [B, D])."""
        t = T(); out = []
        for h in H:
            h = h.to(self.dev).double()
            if W is None:
                out.append(((h @ self.S) / (h @ self.n).clamp_min(1e-12), ((1 - h) @ self.S) / ((1 - h) @ self.n).clamp_min(1e-12)))
            else:
                W = t.as_tensor(W, dtype=t.float64).to(self.dev); Wa, Wb = W * h[None], W * (1 - h)[None]
                out.append(((Wa @ self.S) / (Wa @ self.n).clamp_min(1e-12)[:, None], (Wb @ self.S) / (Wb @ self.n).clamp_min(1e-12)[:, None]))
        return out

    def rel(self, H, W=None):
        """rel = max(2r / (1 + r), 0.05), r = mean over the halvings of cos(half A, half B) (rk.stats; recomputed per replicate through the
        per-prompt Gram matrix G = S S', so a replicate costs O(P^2), not O(P D))."""
        t = T()
        if W is None:
            hs = self.halves(H); r = float(np.mean([cos(a, b) for a, b in hs])); return rel_of(r), r
        W = t.as_tensor(W, dtype=t.float64).to(self.dev); G = self.gram(); rs = []
        for h in H:
            h = h.to(self.dev).double(); wa, wb = W * h[None], W * (1 - h)[None]; ga, gb = wa @ G, wb @ G
            rs.append((ga * wb).sum(1) / t.sqrt(((ga * wa).sum(1) * (gb * wb).sum(1)).clamp_min(1e-300)))
        r = t.stack(rs).mean(0)
        rl = t.where(r <= -1, t.full_like(r, 0.05), (2 * r / (1 + r)).clamp_min(FLOOR_REL)); return rl, r

    def gram(self):
        if getattr(self, "_G", None) is None:
            self._G = self.S @ self.S.T
        return self._G


def readout(row, ax, fold, H, W=None, rel=None):
    """c_n^cf (cross-fitted: fold-True prompts' pairs on aB, fold-False on aA), in-sample c, rel, kappa and A_n = c^cf / sqrt(rel x
    max(kappa, 0.5)) for the full sample (W None) or replicates W [B, P] (axes per replicate: ax tensors [B, D]); Gram-matrix form."""
    t = T(); f = t.as_tensor(fold, dtype=t.float64).to(row.dev)
    W1 = t.ones(1, row.P, dtype=t.float64, device=row.dev) if W is None else t.as_tensor(W, dtype=t.float64).to(row.dev)
    aA, aB, u, kap = [ax[k] if ax[k].dim() == (2 if k != "kappa" else 1) else ax[k][None] for k in ("aA", "aB", "u", "kappa")]
    n = (W1 @ row.n).clamp_min(1e-12); dn = t.sqrt(((W1 @ row.gram()) * W1).sum(1).clamp_min(1e-300)) / n
    SA, SB, SU = (row.S @ aA.T).T, (row.S @ aB.T).T, (row.S @ u.T).T
    if SA.shape[0] != W1.shape[0]:
        SA, SB, SU = SA.expand(W1.shape[0], -1), SB.expand(W1.shape[0], -1), SU.expand(W1.shape[0], -1)
    ccf = ((W1 * f[None] * SB).sum(1) + (W1 * (1 - f)[None] * SA).sum(1)) / n / dn
    c = (W1 * SU).sum(1) / n / (dn * u.norm(dim=-1).clamp_min(1e-12))
    if rel is None:
        rel = row.rel(H, None if W is None else W1)[0]
    rel_t = t.as_tensor(rel, dtype=t.float64, device=row.dev)
    A = ccf / t.sqrt(rel_t * kap.clamp_min(FLOOR_KAPPA))
    if W is None:
        return dict(c_cf=float(ccf[0]), c=float(c[0]), A=float(A[0]), rel=float(rel_t.reshape(-1)[0]), kappa=float(kap[0]),
                    kappa_floored=bool(kap[0] < FLOOR_KAPPA), d_norm=float(dn[0]))
    return dict(c_cf=ccf, c=c, A=A, rel=rel_t, kappa=kap)


def bvals(row, ax, fold):
    """b_p = mean over prompt p's ok pairs of a_{other fold}(p) . delta_j (full-sample fold axes); NaN-free vector over prompts with pairs."""
    t = T(); f = t.as_tensor(fold, dtype=t.bool).to(row.dev); aA, aB = (ax[k].reshape(-1, row.D)[0] for k in ("aA", "aB"))
    b = t.where(f, row.S @ aB, row.S @ aA) / row.n.clamp_min(1e-12)
    return b[row.n > 0].cpu().numpy()


def k1s(b, two_sided=False, boot_t=False, flips=B_FLIP):
    """K1-s on the per-prompt b_p: p_s = (1 + #{k: mean eps_k b >= T}) / (flips + 1), eps Rademacher from default_rng(2001) (two-sided:
    |.|); Wilcoxon and one-sample t beside it. boot_t (RG6 fallback): studentised bootstrap-t over prompts (10,000, seed 2002)."""
    from scipy import stats as ss
    b = np.asarray(b, float); n = len(b); Tm = float(b.mean()) if n else float("nan")
    if n < 2 or not np.isfinite(Tm):
        c = K.coded("NaN:fewer_than_2_prompts"); return dict(p_s=c, T=c, p_wilcoxon=c, p_t=c, n=n, stat="none")
    if boot_t:
        g = np.random.default_rng(SEED_BT); sd = b.std(ddof=1); tobs = Tm / (sd / math.sqrt(n) + 1e-300)
        idx = g.integers(0, n, (flips, n)); bb = b[idx]; ts = (bb.mean(1) - Tm) / (bb.std(1, ddof=1) / math.sqrt(n) + 1e-300)
        p = (1 + int((np.abs(ts) >= abs(tobs)).sum() if two_sided else (ts >= tobs).sum())) / (flips + 1); stat = "bootstrap-t (RG6)"
    else:
        eps = np.random.default_rng(SEED_FLIP).integers(0, 2, (flips, n)) * 2 - 1; nl = eps @ b / n
        p = (1 + int((np.abs(nl) >= abs(Tm)).sum() if two_sided else (nl >= Tm).sum())) / (flips + 1); stat = "sign-flip"
    alt = "two-sided" if two_sided else "greater"
    try:
        pw = float(ss.wilcoxon(b, alternative=alt).pvalue)
    except ValueError:
        pw = K.coded("NaN:wilcoxon_undefined")
    return dict(p_s=float(p), T=Tm, p_wilcoxon=K.val(pw) if not isinstance(pw, dict) else pw,
                p_t=K.val(float(ss.ttest_1samp(b, 0.0, alternative=alt).pvalue), "NaN:t_undefined"), n=n, stat=stat,
                share_pos=float((b > 0).mean()))


# ------------------------------------------------------------------------------------------------------------------ cross-fitted cosines
def cf_cos(rowx, rowsy, H, signs=None, W=None):
    """Cross-fitted, disattenuated cosine between row x's d and the sign-aligned sum of rows y (one row: R_4^cf; several: a mode):
    mean over halvings of (cos(x_A, y_B) + cos(x_B, y_A)) / 2, divided by sqrt(r_x r_y) (raw split-half cosines); the uncorrected
    full-sample cosine beside it."""
    signs = signs or [1.0] * len(rowsy)
    hx = rowx.halves(H, W); hy = [r.halves(H, W) for r in rowsy]
    ys = [(sum(s * h[i][0] for s, h in zip(signs, hy)), sum(s * h[i][1] for s, h in zip(signs, hy))) for i in range(len(H))]
    cr = float(np.mean([(cos(xa, yb) + cos(xb, ya)) / 2 for (xa, xb), (ya, yb) in zip(hx, ys)]))
    rx = float(np.mean([cos(a, b) for a, b in hx])); ry = float(np.mean([cos(a, b) for a, b in ys]))
    yd = sum(s * r.d() for s, r in zip(signs, rowsy)); raw = cos(rowx.d(), yd)
    den = rx * ry
    return dict(cf=cr / math.sqrt(den) if den > 0 else K.coded("NaN:split_half_cos_nonpositive"), cross=cr, r_x=rx, r_y=ry, raw=raw)


def s4cf(row3, row4, H):
    """S_4^cf = mean_h (cos(d_4^A, d_3^B) + cos(d_4^B, d_3^A)) / 2 / sqrt(r_3 r_4); uncorrected beside it."""
    return cf_cos(row4, [row3], H)


def deb_fixed(row, u, rel, m2=1.0):
    """Debiased |cos(d_4, u)| = |cos| / sqrt(rel_4 m2) for a fixed vector u (m2: its attenuation, round 2's m*^2 for u*_b)."""
    c = abs(cos(row.d(), u)); den = rel * m2
    return c / math.sqrt(den) if den > 0 else float("nan")


# ------------------------------------------------------------------------------------------------------------------ SENS plant
def plant(row, u, rel, c_star_A=0.25):
    """R2.3b: delta_j <- delta_j - (u . d_n) u + eta_n u with eta_n = c* |d_n - (u . d_n) u| / sqrt(1 - c*^2), c* = 0.25 sqrt(rel_n),
    so that cos(d_n', u) = c*. Returns a new Row (same prompts)."""
    t = T(); u = unit(t.as_tensor(u).double()); d = row.d().cpu(); c = c_star_A * math.sqrt(max(rel, 1e-12))
    perp = d - (u @ d) * u; eta = c * float(perp.norm()) / math.sqrt(1 - c * c)
    dl = row.delta + (eta - float(u @ d)) * u[None]
    return Row(dl, row.ok, row.po, row.P, row.dev)


# ------------------------------------------------------------------------------------------------------------------ joint MDE (normal model)
def joint_mde(sd_a, rho_w, rho_b, rule, grid=np.arange(0.0, 0.81, 0.005), n_sim=4000, seed=7):
    """Normal-model simulation (R.1 joint-MDE convention, joint_R.py's model): rows A_1..A_4 of two loops with SD sd_a, within-loop
    correlation rho_w, between-loop rho_b. rule 'substantial' / 'dominated': smallest flat true A with P(A_bar lower bound >= 0.25 /
    0.50) >= 0.8 (A_bar = mean of A_3, A_4 over both loops); 'unlikely': largest true late A (A_1 = 0.30) with P(A_bar_abs upper
    bound < 0.25) >= 0.8. Bounds from the simulated SD of A_bar (one-sided 5 %)."""
    g = np.random.default_rng(seed); idx = [(l, n) for l in range(2) for n in range(4)]
    Cm = np.array([[1.0 if i == j else (rho_w if a[0] == b[0] else rho_b) for j, b in enumerate(idx)] for i, a in enumerate(idx)]) * sd_a ** 2
    Z = g.multivariate_normal(np.zeros(8), Cm, n_sim); late = [2, 3, 6, 7]; sdb = Z[:, late].mean(1).std()
    out = None
    for a in (grid if rule != "unlikely" else grid[::-1]):
        mu = np.full(8, a); mu[[0, 4]] = 0.30 if rule == "unlikely" else a
        Ab = (Z + mu)[:, late]
        if rule == "unlikely":
            ok = (np.abs(Ab).mean(1) + 1.645 * sdb < 0.25).mean() >= 0.8
        else:
            ok = (Ab.mean(1) - 1.645 * sdb >= (0.25 if rule == "substantial" else 0.50)).mean() >= 0.8
        if ok:
            out = float(a); break
    return dict(rule=rule, sd_a=sd_a, rho_w=rho_w, rho_b=rho_b, mde=K.val(out, "CENSORED:above_grid" if rule != "unlikely" else "CENSORED:below_grid"),
                sd_abar=float(sdb))


# ------------------------------------------------------------------------------------------------------------------ R4 (FDT Jacobian, products only)
def fdt_centre(X, Sg, prompt_of, w=None):
    """R4.1: x and S centred within prompt (weighted by w per text), times n_s / (n_s - 1); returns (Xc, Sc, wt) for J = Xc' diag(wt) Sc."""
    t = T(); X, Sg = t.as_tensor(X).double(), t.as_tensor(Sg).double(); po = t.as_tensor(prompt_of).long(); P = int(po.max()) + 1
    w = t.ones(len(X), dtype=t.float64) if w is None else t.as_tensor(w, dtype=t.float64)
    cnt = t.bincount(po, minlength=P).double(); wp = t.zeros(P, dtype=t.float64).index_add_(0, po, w)
    mx = t.zeros(P, X.shape[1], dtype=t.float64).index_add_(0, po, w[:, None] * X) / wp.clamp_min(1e-12)[:, None]
    ms = t.zeros(P, X.shape[1], dtype=t.float64).index_add_(0, po, w[:, None] * Sg) / wp.clamp_min(1e-12)[:, None]
    ns = cnt[po]; fac = ns / (ns - 1).clamp_min(1)
    return X - mx[po], Sg - ms[po], w * fac / w.sum()


def restricted(Xc, Sc, wt, cbar, Q):
    """M = Q' J Q with J = Xc' diag(wt) Sc / cbar (k x k, forward products only)."""
    return ((Xc @ Q) * wt[:, None]).T @ (Sc @ Q) / cbar


def rightmost(M):
    """Rightmost eigenvalue (largest real part) of M with right (eig M) and left (eig M') eigenvectors (forward eigenproblem)."""
    t = T(); ev, V = t.linalg.eig(M); i = int(t.argmax(ev.real)); lam = ev[i]
    el, U = t.linalg.eig(M.T); j = int(t.argmin((el - lam).abs()))
    return complex(lam), V[:, i], U[:, j], complex(ev[int(t.argmax(ev.abs()))])


def heldout(MB, y, z):
    """Held-out value z' M_B y / (z' y) (real part); Rayleigh read y' M_B y / y'y if |cos(y, z)| < 0.2 (label 'non-normal: Rayleigh read')."""
    t = T(); yr, zr = y.real.double(), z.real.double()
    yr = yr if float(yr.norm()) > 1e-9 else y.imag.double(); zr = zr if float(zr.norm()) > 1e-9 else z.imag.double()
    c = abs(float(yr @ zr) / (float(yr.norm() * zr.norm()) + 1e-300))
    if c < 0.2:
        return float(yr @ MB @ yr / (yr @ yr)), True
    return float(zr @ MB @ yr / (zr @ yr)), False


def arnoldi(matvec, v0, k=20):
    """Arnoldi with full re-orthogonalisation (forward or backward products): Ritz values and vectors of the k x k Hessenberg matrix."""
    t = T(); n = v0.shape[0]; Vb = t.zeros(n, k + 1, dtype=t.float64); H = t.zeros(k + 1, k, dtype=t.float64)
    Vb[:, 0] = v0 / v0.norm(); m = k
    for j in range(k):
        w = matvec(Vb[:, j])
        for _ in range(2):
            h = Vb[:, :j + 1].T @ w; w = w - Vb[:, :j + 1] @ h; H[:j + 1, j] += h
        H[j + 1, j] = w.norm()
        if float(H[j + 1, j]) < 1e-12:
            m = j + 1; break
        Vb[:, j + 1] = w / H[j + 1, j]
    ev, Y = t.linalg.eig(H[:m, :m]); return ev, Vb[:, :m].to(t.complex128) @ Y


def full_ritz(Xc, Sc, wt, cbar, v0, k=20):
    """Full-space top mode (R4.2, descriptive): forward Arnoldi (J y) for the rightmost Ritz value and right vector, backward Arnoldi
    (J' z) from the same start for the left vector of the closest Ritz value."""
    t = T(); f = lambda v: Xc.T @ (wt * (Sc @ v)) / cbar; g = lambda v: Sc.T @ (wt * (Xc @ v)) / cbar
    ev, Y = arnoldi(f, v0, k); i = int(t.argmax(ev.real)); el, Z = arnoldi(g, v0, k); j = int(t.argmin((el - ev[i]).abs()))
    return complex(ev[i]), Y[:, i], Z[:, j]


def axes_for(ax1b, names, W, dev="cpu", extra=None):
    """AX-1b of a block (r1b_R_S2 file) as an AxisSet: full-sample axes and the joint-bootstrap replicates W for the names."""
    t = T(); AS = AxisSet(ax1b["diff"], ax1b["keep"], ax1b["fold"], ax1b["u_star"], ax1b["u_neg"], ax1b["block"], dev, extra)
    P = len(ax1b["fold"]); return AS, AS.get(t.ones(1, P, dtype=t.float64), names), (AS.get(W, names) if W is not None else None)


def stat_row(row, a1, aW, fold, H, W, two_sided=False, boot_t=False, keep_draws=False):
    """STAT-1b of one evaluation on one axis: A_n (cross-fitted, disattenuated) with its 90 % joint-bootstrap interval, c^cf, in-sample c,
    rel, kappa, K1-s p_s (sign-flip or bootstrap-t (RG6); one-sided toward the pro pole, two-sided on request) with Wilcoxon and t."""
    out = readout(row, a1, fold, H)
    k = k1s(bvals(row, a1, fold), two_sided=two_sided, boot_t=boot_t); out.update(p_s=k["p_s"], p_wilcoxon=k["p_wilcoxon"], p_t=k["p_t"], stat=k["stat"],
                                                                              share_pos=k.get("share_pos"))
    if aW is not None:
        rb = readout(row, aW, fold, H, W=W); A = rb["A"].cpu().numpy(); lo, hi = iv(A); out.update(A_lo=lo, A_hi=hi)
        alo, ahi = iv(np.abs(A)); out.update(absA_lo=alo, absA_hi=ahi)
        if keep_draws:
            out["A_draws"] = A; out["rel_draws"] = rb["rel"].cpu().numpy()
    return out
