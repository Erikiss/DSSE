"""Round 2, session r2_ana: C-Q1 (frozen plan results/ju_space/plans/abschnitt11_12_runde2_plan.md, sha256 5e476340...edc2d,
section 5.2 C-Q1 with OD-C5, OD-C17, gates G41 and G42, 2.7 C28, section 13 Checks) -> unlearn/q1.json.

    python r2_linearity.py <r1_dir> <r2_root> <out.json>          (the runner: <out.json> = <work>/unlearn/q1.json)
    python r2_linearity.py selftest <workdir>                       synthetic code test (planted signals, every branch)
    env DEVICE (cuda | cpu; default cuda when available), MOCK (code tests only: R2_Q1_NCAL / R2_Q1_NPERM / R2_Q1_NFLIP may lower the
    replicate counts; a real run always uses the plan's 50 / 200 / 1,000)

Inputs: <r2_root>/r2_q1/axes/tok14.pt (O2 (iii): per-token block-14 states, float16, of the round-1 RSI and sea pro / anti texts and of
this session's RGI_PRO / RGI_ANTI texts, continuation tokens only, and the round-1 prompt fold); <r2_root>/r2_q1/unlearn/q3.json (C-Q3's
fixed-axis gaps: the causal linear rung); R1.state[14]['axes'] (u_T for the calibration's d', sea for the plant directions);
R1.summary[14] (G-desc); <r2_root>/r2_t3/unlearn_toy/t3.json (G41).
Per concept (RSI, RGI, sea), with the two round-1 prompt folds A (fold True) and B (fold False); every probe is a closed-form ridge
least-squares classifier (labels +-1, ridge 1e-2 x tr(Sigma) / D on its training tokens, Cholesky in float64), trained on tokens; a
text's score is the mean of its tokens' scores; AUC over texts (OD-C5):
  (1) mean-difference direction AUC, cross-fitted (direction = the training fold's token class-mean difference, AUC on the other fold's
      texts, both directions averaged), against 1,000 within-prompt sign flips (descriptive);
  (2) AUC_cf: fold A's LEACE eraser applied to fold B; a fresh ridge probe trained on one half of fold B's prompts (fixed random halves,
      seed 25) and tested on the other half, both assignments; folds swapped; the four AUCs averaged (descriptive);
  (3) classic INLP r*: up to 8 times a ridge probe on the current training-fold states, its unit weight direction removed from both
      folds by orthogonal projection; r* = removals until the held-out AUC <= 0.60 ('> 8' if never; 0 if <= 0.60 before any removal);
      both directions reported (descriptive);
  (4) nonlinear residue: the training fold's own LEACE eraser applied to both folds; a degree-2 probe on the top-32 PCs of the erased
      training-fold states (32 linear + 528 product features, standardised, ridge 1), trained on the training fold, tested on the other,
      both directions averaged; AUC against 200 text-label permutations (each prompt's pro and anti labels swapped at random), all 201
      fits in one multi-right-hand-side solve.
LEACE eraser of a fold: x' = x - delta w^T (x - mu), delta = class-mean difference, mu = mean, w = (Sigma + r I)^-1 delta / (delta^T
(Sigma + r I)^-1 delta) (ridge-regularised covariance of the fold); it leaves zero label cross-covariance in the fold.
Calibration first (G42; 50 replicates each) on the round-1 sea token states with fresh random text labels (balanced per prompt): e1, e2
random orthonormal directions in the span of the top-32 PCs of those states, orthogonal to u_sea; each planted component scaled to u_T's
per-token d' on the RSI tokens (OD-C5); (i) '1-dim': + y a1 e1 on every token of a text with label y (also the AUC_cf reference of (2));
(ii) 'XOR': + a1 z e1 + a2 y z e2 with a second label z, balanced within each class of y. G42 = XOR 'pattern' in >= 80 % and 1-dim no 'pattern' in >= 80 %.
Rules: nonlinear rung valid only under G42 (else 'not valid (calibration failed)'); 'pattern (within the top-32 PCs)' = the degree-2
AUC >= 0.70 and above its permutation p99, otherwise 'no nonlinear residue found'; no pattern claim if the circle gate G41 failed.
Linear rung, causal (RSI, RGI; from C-Q3's g_abl / g_0 on the fixed axes O1): 'one direction' = the 1-dim ablation leaves g_abl / g_0
<= 0.5 (RSI: u_RSI on the RSI-instructed texts; RGI: unit(u_RGI_perp) on the RGI-instructed texts); 'two-dimensional span' = it does not
but the span(u_RSI, u_RGI_perp) ablation does; otherwise 'not removed within the tested 2-dim span'. Sea: no linear rung.
Both rungs are reported together; (1)-(3) decide nothing."""
import math, os, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); JU = os.path.dirname(HERE)
for _p in (HERE, JU):
    if _p not in sys.path: sys.path.insert(0, _p)
import r2_report as RP

MOCK = os.environ.get("MOCK") == "1"
SEEDS = dict(halves=25, flips=4101, perm=4102, calib=4103)          # OD-G6 (halves: the plan's seed 25)
N_FLIP = 1000 if not MOCK else int(os.environ.get("R2_Q1_NFLIP", "1000"))
N_PERM = 200 if not MOCK else int(os.environ.get("R2_Q1_NPERM", "200"))
N_CAL = 50 if not MOCK else int(os.environ.get("R2_Q1_NCAL", "50"))
N_TOP, RIDGE, RIDGE_Q = 32, 1e-2, 1.0
INLP_MAX, INLP_AUC = 8, 0.60
PATTERN_AUC, PATTERN_Q, G42_SHARE, GAP_POINT = 0.70, 99, 0.80, 0.5
CONCEPTS = dict(RSI=("R1_RSI_PRO", "R1_RSI_ANTI"), RGI=("RGI_PRO", "RGI_ANTI"), sea=("R1_SEA_PRO", "R1_SEA_ANTI"))
RUNG_INPUTS = dict(RSI=("RSI_1dim", "RSI_span", "u_RSI", "RSI"), RGI=("RGI_1dim", "RGI_span", "u_RGI_perp", "RGI"))
NOT_RUN_R1 = RP.NOT_RUN_R1


def log(msg): print(f"[r2_linearity {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def device():
    want = os.environ.get("DEVICE")
    if want: return torch.device(want if (want == "cpu" or torch.cuda.is_available()) else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================================================================================
# Token data of one concept
# ======================================================================================================================================
class Concept:
    """Token states of one concept. X [N, D] float64; per token tt [N] (text index); per text: prompt, y (+1 / -1), fold (bool), the
    text-mean states Xbar [T, D]. Empty texts are dropped (the other text of the prompt stays)."""

    def __init__(self, pro, anti, fold, dev, X=None, meta=None):
        if X is not None:                                                         # a view with other states / labels (calibration)
            self.__dict__.update(meta); self.X = X; self._means(); return
        fold = torch.as_tensor(fold, dtype=torch.bool).cpu()
        prompt, y, n_tok, chunks = [], [], [], []
        for p in range(len(fold)):
            for lab, t in ((1, pro[p] if p < len(pro) else None), (-1, anti[p] if p < len(anti) else None)):
                if t is None or int(t.shape[0]) == 0: continue
                prompt.append(p); y.append(lab); n_tok.append(int(t.shape[0])); chunks.append(torch.as_tensor(t).to(torch.float64))
        self.dev = dev; self.P = len(fold); self.fold_p = fold
        self.prompt = torch.tensor(prompt, dtype=torch.long); self.y = torch.tensor(y, dtype=torch.float64); self.n_tok = torch.tensor(n_tok, dtype=torch.long)
        self.tfold = fold[self.prompt] if len(prompt) else torch.zeros(0, dtype=torch.bool)
        self.tt = torch.repeat_interleave(torch.arange(len(prompt)), self.n_tok)
        self.X = torch.cat(chunks).to(dev) if chunks else torch.zeros(0, 1, dtype=torch.float64, device=dev)
        self._means()

    def _means(self):
        T, D = len(self.prompt), self.X.shape[1]; tt = self.tt.to(self.X.device)
        S = torch.zeros(T, D, dtype=torch.float64, device=self.X.device).index_add_(0, tt, self.X)
        self.sums = S; self.Xbar = S / self.n_tok.to(self.X.device, torch.float64)[:, None]

    def meta(self):
        return {k: v for k, v in self.__dict__.items() if k not in ("X", "sums", "Xbar")}

    def view(self, X=None, y=None):
        m = self.meta()
        if y is not None: m["y"] = torch.as_tensor(y, dtype=torch.float64)
        return Concept(None, None, None, None, X=self.X if X is None else X, meta=m)

    def texts(self, fold, prompts=None):
        """Text indices of a fold (optionally only of the given prompts)."""
        m = self.tfold == bool(fold)
        if prompts is not None: m &= torch.isin(self.prompt, torch.as_tensor(sorted(prompts), dtype=torch.long))
        return torch.nonzero(m).flatten()

    def tokens(self, texts):
        """Token mask [N] of the given texts."""
        keep = torch.zeros(len(self.prompt), dtype=torch.bool); keep[texts] = True; return keep[self.tt].to(self.X.device)

    def ytok(self, y=None):
        y = self.y if y is None else y; return y[self.tt].to(self.X.device)

    def summary(self):
        return dict(n_texts=int(len(self.prompt)), n_tokens=int(self.X.shape[0]), D=int(self.X.shape[1]),
                    n_texts_fold=dict(A=int(self.tfold.sum()), B=int((~self.tfold).sum())), n_pro=int((self.y > 0).sum()), n_anti=int((self.y < 0).sum()))


# ======================================================================================================================================
# Probes, eraser, AUC
# ======================================================================================================================================
def moments(X, y):
    """Mean, covariance (1/n) and class-mean difference (pro minus anti) of token states X [n, D] with token labels y [n] (+-1)."""
    mu = X.mean(0); Xc = X - mu; S = Xc.T @ Xc / max(len(X), 1)
    pos, neg = y > 0, y < 0
    delta = (X[pos].mean(0) if pos.any() else torch.full_like(mu, float("nan"))) - (X[neg].mean(0) if neg.any() else torch.full_like(mu, float("nan")))
    return mu, S, delta


def ridge_lam(S, ridge):
    return ridge * float(torch.diagonal(S).sum()) / S.shape[0]


def solve_spd(S, B, lam):
    """(S + lam I)^-1 B by Cholesky (float64)."""
    A = S + lam * torch.eye(S.shape[0], dtype=S.dtype, device=S.device)
    L, info = torch.linalg.cholesky_ex(A)
    if int(info) != 0: return torch.linalg.solve(A, B)
    return torch.cholesky_solve(B, L)


def probe_dir(S, delta, ridge=RIDGE):
    """Direction of the closed-form ridge least-squares classifier (labels +-1): w = (Sigma + r I)^-1 Cov(x, y) with r = ridge tr(Sigma) / D;
    Cov(x, y) = 2 p (1 - p) delta, so w is proportional to (Sigma + r I)^-1 delta (AUC is scale-free)."""
    return solve_spd(S, delta[:, None], ridge_lam(S, ridge))[:, 0]


def leace(X, y, ridge=RIDGE):
    """LEACE eraser of one fold: (mu, delta, w) with w = (Sigma + r I)^-1 delta / (delta^T (Sigma + r I)^-1 delta)."""
    mu, S, delta = moments(X, y); z = probe_dir(S, delta, ridge); return mu, delta, z / (delta @ z)


def erase(X, E):
    mu, delta, w = E; return X - ((X - mu) @ w)[:, None] * delta[None]


def auc_cols(scores, labels):
    """AUC per column (Mann-Whitney with average ranks; ties count 1/2): scores [n] or [n, K], labels [n] or [n, K] bool (pro = True).
    NaN where a column has no pro or no anti text."""
    from scipy.stats import rankdata
    s = np.asarray(scores, dtype=np.float64); l = np.asarray(labels, dtype=bool)
    one = s.ndim == 1
    if one: s = s[:, None]
    if l.ndim == 1: l = np.repeat(l[:, None], s.shape[1], 1)
    if s.shape[0] == 0: out = np.full(s.shape[1], np.nan)
    else:
        r = rankdata(s, axis=0); n1 = l.sum(0).astype(float); n0 = l.shape[0] - n1
        with np.errstate(invalid="ignore", divide="ignore"):
            out = ((np.where(l, r, 0.0).sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0))
        out = np.where((n1 > 0) & (n0 > 0), out, np.nan)
    return float(out[0]) if one else out


def flip_labels(C, texts, F):
    """Labels [len(texts), K] (pro = True) of the given texts under sign rows F [K, P] (+-1 per prompt; row of ones = the true labels)."""
    Fp = torch.as_tensor(F, dtype=torch.float64)[:, C.prompt[texts]]                # [K, n]
    return ((Fp * C.y[texts][None]) > 0).T.numpy()


# ======================================================================================================================================
# (1) mean-difference direction, (2) AUC_cf, (3) INLP, (4) degree-2 probe after LEACE
# ======================================================================================================================================
DIRECTIONS = ((True, False), (False, True))                                      # (training fold, held-out fold): A -> B, B -> A


def md_auc(C, F):
    """(1): cross-fitted AUC of the token class-mean difference of the training fold on the held-out fold's texts, both directions
    averaged, for every sign row of F [K, P] (row 0 = the true labels): returns (auc [K], per-direction [2, K])."""
    per = []
    Fd = torch.as_tensor(F, dtype=torch.float64, device=C.X.device)
    for f, g in DIRECTIONS:
        tf, tg = C.texts(f), C.texts(g)
        s = Fd[:, C.prompt[tf].to(C.X.device)] * C.y[tf].to(C.X.device)[None]          # [K, nf] signs under each flip
        cnt = C.n_tok[tf].to(C.X.device, torch.float64); Sm = C.sums[tf.to(C.X.device)]
        P_, N_ = (s > 0).double(), (s < 0).double()
        delta = (P_ @ Sm) / (P_ @ cnt)[:, None] - (N_ @ Sm) / (N_ @ cnt)[:, None]       # [K, D]
        sc = (C.Xbar[tg.to(C.X.device)] @ delta.T).cpu().numpy()
        per.append(auc_cols(sc, flip_labels(C, tg, F)))
    per = np.stack(per); return np.nanmean(per, 0), per


def halves(fold_p, seed=SEEDS["halves"]):
    """Fixed random halves of each fold's prompts (np.random.default_rng(seed): fold True's permutation first, then fold False's)."""
    rng = np.random.default_rng(seed); out = {}
    for f in (True, False):
        ps = np.nonzero(np.asarray(fold_p, dtype=bool) == f)[0]; perm = rng.permutation(ps); k = len(perm) // 2
        out[f] = (sorted(int(x) for x in perm[:k]), sorted(int(x) for x in perm[k:]))
    return out


def auc_cf(C, H):
    """(2) AUC_cf: the eraser of fold f applied to fold g; a fresh probe trained on one half of fold g's prompts, tested on the other half,
    both assignments; f and g swapped; the four AUCs averaged. Returns dict(auc, parts)."""
    parts = []
    for f, g in DIRECTIONS:
        tf = C.texts(f); E = leace(C.X[C.tokens(tf)], C.ytok()[C.tokens(tf)])
        tg = C.texts(g); mg = C.tokens(tg); Xg = erase(C.X[mg], E); yg = C.ytok()[mg]; ttg = C.tt.to(C.X.device)[mg]
        Xbar = erase(C.Xbar, E)
        for tr, te in (H[g], H[g][::-1]):
            ttr, tte = C.texts(g, tr), C.texts(g, te)
            if len(ttr) == 0 or len(tte) == 0: parts.append(float("nan")); continue
            m = torch.isin(ttg, ttr.to(C.X.device))
            _, S, delta = moments(Xg[m], yg[m]); w = probe_dir(S, delta)
            parts.append(auc_cols((Xbar[tte.to(C.X.device)] @ w).cpu().numpy(), (C.y[tte] > 0).numpy()))
    return dict(auc=float(np.nanmean(parts)) if np.isfinite(parts).any() else float("nan"), parts=parts,
                order="eraser A -> fold B (train half 1, test half 2; swapped), eraser B -> fold A (same)")


def _proj_cov(S, Q):
    if Q.shape[1] == 0: return S
    SQ = S @ Q; return S - Q @ SQ.T - SQ @ Q.T + Q @ (Q.T @ SQ) @ Q.T


def inlp(C, f, g, max_r=INLP_MAX, thr=INLP_AUC):
    """(3) classic INLP with training fold f and held-out fold g: AUC of a fresh ridge probe after r = 0, 1, ... removals; r* = the first r
    with held-out AUC <= thr ('> 8' if none up to 8 removals)."""
    tf, tg = C.texts(f), C.texts(g); m = C.tokens(tf); _, S, delta = moments(C.X[m], C.ytok()[m])
    Q = torch.zeros(S.shape[0], 0, dtype=torch.float64, device=S.device); aucs = []; lab = (C.y[tg] > 0).numpy(); Xg = C.Xbar[tg.to(S.device)]
    for r in range(max_r + 1):
        Sp = _proj_cov(S, Q); dp = delta - Q @ (Q.T @ delta); w = probe_dir(Sp, dp)
        a = auc_cols((Xg @ w).cpu().numpy(), lab); aucs.append(a)
        if fin(a) and a <= thr: return r, aucs
        if r == max_r: return f"> {max_r}", aucs
        v = w - Q @ (Q.T @ w); Q = torch.cat([Q, (v / v.norm())[:, None]], 1)
    return f"> {max_r}", aucs


def fin(x): return RP.fin(x)


_IU = {}


def quad(Z):
    """[n, k] -> [n, k + k(k+1)/2]: the k linear and the products z_i z_j (i <= j) features."""
    k = Z.shape[1]
    if k not in _IU: _IU[k] = torch.triu_indices(k, k)
    iu = _IU[k].to(Z.device); return torch.cat([Z, Z[:, iu[0]] * Z[:, iu[1]]], 1)


def nonlinear(C, F, n_top=N_TOP, ridge=RIDGE_Q):
    """(4): per direction the training fold's LEACE eraser on both folds, top-n_top PCs of the erased training fold, degree-2 features
    standardised on the training tokens, ridge least squares (Sigma_F + ridge I)^-1 Cov(F, y) for every label row of F [K, P] (row 0 =
    the true labels) in one solve, text scores = mean token scores, AUC on the held-out fold. Returns dict(auc (row 0), null [K - 1],
    p99, per_direction, pattern_statistic, zero_cov (max |label cross-covariance| of the erased training fold, a LEACE check))."""
    per, zc = [], []
    Fd = torch.as_tensor(F, dtype=torch.float64)
    for f, g in DIRECTIONS:
        tf, tg = C.texts(f), C.texts(g); mf, mg = C.tokens(tf), C.tokens(tg); yf = C.ytok()[mf]
        E = leace(C.X[mf], yf); Xf, Xg = erase(C.X[mf], E), erase(C.X[mg], E)
        mu = Xf.mean(0); Xc = Xf - mu; zc.append(float((Xc.T @ (yf - yf.mean())).abs().max() / max(len(Xf), 1)))
        Cf = Xc.T @ Xc / max(len(Xf), 1); ev, V = torch.linalg.eigh(Cf); V = V[:, -min(n_top, V.shape[1]):]
        Ff = quad(Xc @ V); Fg = quad((Xg - mu) @ V); m = Ff.mean(0); s = Ff.std(0, unbiased=False).clamp_min(1e-12)
        Ff = (Ff - m) / s; Fg = (Fg - m) / s
        Y = (Fd[:, C.prompt[C.tt[mf.cpu()]]].T * C.y[C.tt[mf.cpu()]][:, None]).to(Ff.device)          # [nf, K] token labels under each row
        Yc = Y - Y.mean(0); n = len(Ff)
        W = solve_spd(Ff.T @ Ff / n, Ff.T @ Yc / n, ridge)
        ttg = C.tt.to(C.X.device)[mg]; idx = torch.searchsorted(tg.to(C.X.device), ttg)
        Fbar = torch.zeros(len(tg), Fg.shape[1], dtype=torch.float64, device=Fg.device).index_add_(0, idx, Fg) / C.n_tok[tg].to(Fg.device, torch.float64)[:, None]
        per.append(auc_cols((Fbar @ W).cpu().numpy(), flip_labels(C, tg, F)))
    per = np.stack(per); a = np.nanmean(per, 0); obs, null = float(a[0]), a[1:]
    p99 = RP.pct(null, PATTERN_Q)
    return dict(auc=obs, per_direction=per[:, 0].tolist(), null_mean=RP.pct(null, 50) if len(null) else float("nan"), p99=p99,
                n_perm=int(len(null)), pattern_statistic=bool(fin(obs) and obs >= PATTERN_AUC and fin(p99) and obs > p99), leace_max_abs_cov=max(zc))


def sign_rows(K, P, seed):
    """[K + 1, P] +-1: row 0 = ones (the true labels), K rows of independent per-prompt swaps (np.random.default_rng(seed))."""
    rng = np.random.default_rng(seed); return np.concatenate([np.ones((1, P)), rng.choice([-1.0, 1.0], size=(K, P))])


def concept_stats(C, H, flips, perm):
    """(1)-(4) of one concept."""
    a1, per1 = md_auc(C, flips); obs, null = float(a1[0]), a1[1:]
    md = dict(auc=obs, per_direction=per1[:, 0].tolist(), n_flip=int(len(null)), p=float((1 + np.sum(null >= obs)) / (1 + len(null))) if fin(obs) else float("nan"),
              null_p95=RP.pct(null, 95))
    cf = auc_cf(C, H)
    rAB, aAB = inlp(C, True, False); rBA, aBA = inlp(C, False, True)
    rmax = max((r for r in (rAB, rBA)), key=lambda r: INLP_MAX + 1 if isinstance(r, str) else r)
    nl = nonlinear(C, perm)
    return dict(data=C.summary(), md_auc=md, auc_cf=cf, inlp=dict(r_star_A_to_B=rAB, r_star_B_to_A=rBA, r_star_max=rmax, auc_A_to_B=aAB, auc_B_to_A=aBA,
                                                                    rule=f"removals until the held-out AUC <= {INLP_AUC} ('> {INLP_MAX}' if never)"),
                nonlinear=nl)


# ======================================================================================================================================
# Calibration (G42)
# ======================================================================================================================================
def dprime(proj, y):
    """Per-token d' = (mean pro - mean anti) / pooled within-class SD (OD-C5)."""
    a, b = proj[y > 0], proj[y < 0]
    if len(a) < 2 or len(b) < 2: return float("nan")
    sp = math.sqrt(((len(a) - 1) * float(a.var()) + (len(b) - 1) * float(b.var())) / (len(a) + len(b) - 2))
    return (float(a.mean()) - float(b.mean())) / sp if sp > 0 else float("nan")


def plant_basis(X, u_sea, n_top=N_TOP):
    """Orthonormal basis [D, n_top - 1] of span(top-n_top PCs of X) intersected with the complement of u_sea."""
    Xc = X - X.mean(0); ev, V = torch.linalg.eigh(Xc.T @ Xc / len(X)); V = V[:, -min(n_top, V.shape[1]):]
    u = torch.as_tensor(u_sea, dtype=torch.float64, device=X.device); c = V.T @ (u / u.norm())
    if float(c.norm()) < 1e-12: return V
    c = c / c.norm(); Pm = torch.eye(V.shape[1], dtype=torch.float64, device=X.device) - c[:, None] * c[None]
    U, s, _ = torch.linalg.svd(V @ Pm, full_matrices=False); return U[:, : V.shape[1] - 1]


def balanced_labels(C, rng):
    """y: per prompt one text +1 and the other -1 at random (a prompt with one non-empty text: a random sign)."""
    s = rng.choice([-1.0, 1.0], size=C.P); return torch.as_tensor(s[C.prompt.numpy()] * C.y.numpy(), dtype=torch.float64)


def balanced_within(y, rng):
    """z: a second balanced label per text, balanced within each class of y (half of the y = +1 texts get z = +1, half -1; the same for
    y = -1; an odd extra text a random sign), independent of the prompt pairing, so that the plant has exactly equal class means at the
    text level and z is not one of the permutation rows y s_p."""
    y = np.asarray(y); z = np.zeros(len(y))
    for c in (1.0, -1.0):
        idx = rng.permutation(np.nonzero(y == c)[0]); k = len(idx) // 2
        z[idx[:k]] = 1.0; z[idx[k:2 * k]] = -1.0
        if len(idx) % 2: z[idx[-1]] = rng.choice([-1.0, 1.0])
    return torch.as_tensor(z, dtype=torch.float64)


def calibration(sea, d_T, u_sea, H, n_rep=N_CAL, seed=SEEDS["calib"]):
    """G42: n_rep replicates of the 1-dim and the XOR plant in the sea token states (fresh labels, directions and permutations per
    replicate from np.random.default_rng([seed, kind, replicate])); each replicate runs (4) with the replicate's labels (and (2) for the
    1-dim plant). Returns the per-replicate AUC, p99 and 'pattern' and the G42 decision."""
    B = plant_basis(sea.X, u_sea); out = dict(d_T=d_T, n_rep=n_rep, plants={}, seed=seed, basis_dim=int(B.shape[1]))
    for ki, kind in enumerate(("onedim", "xor")):
        rows = []
        for rep in range(n_rep):
            rng = np.random.default_rng([seed, ki, rep])
            q, _ = np.linalg.qr(rng.standard_normal((B.shape[1], 2))); e = B @ torch.as_tensor(q, dtype=torch.float64, device=B.device)
            e1, e2 = e[:, 0], e[:, 1]
            y = balanced_labels(sea, rng); z = balanced_within(y.numpy(), rng)
            yt, zt = y[sea.tt].to(sea.X.device), z[sea.tt].to(sea.X.device)
            sd1, sd2 = float((sea.X @ e1).std()), float((sea.X @ e2).std())
            a1, a2 = abs(d_T) * sd1 / 2, abs(d_T) * sd2 / 2
            if kind == "onedim": Xp = sea.X + (yt * a1)[:, None] * e1[None]
            else: Xp = sea.X + (zt * a1)[:, None] * e1[None] + (yt * zt * a2)[:, None] * e2[None]
            V = sea.view(X=Xp, y=y); perm = sign_rows(N_PERM, sea.P, int(rng.integers(0, 2 ** 31)))
            nl = nonlinear(V, perm)
            r = dict(auc=nl["auc"], p99=nl["p99"], pattern=nl["pattern_statistic"], a1=a1, a2=a2 if kind == "xor" else None,
                     dprime_planted=dprime((Xp @ e1).cpu() if kind == "onedim" else (Xp @ e2).cpu(), (yt if kind == "onedim" else yt * zt).cpu()))
            if kind == "onedim": r["auc_cf"] = auc_cf(V, H)["auc"]
            rows.append(r)
        pat = [r["pattern"] for r in rows]
        out["plants"][kind] = dict(replicates=rows, n_pattern=int(sum(pat)), share_pattern=float(np.mean(pat)) if pat else float("nan"),
                                   auc_median=RP.pct([r["auc"] for r in rows], 50))
        if kind == "onedim": out["plants"][kind]["auc_cf_median"] = RP.pct([r["auc_cf"] for r in rows], 50)
    xs, os_ = out["plants"]["xor"]["share_pattern"], 1 - out["plants"]["onedim"]["share_pattern"]
    out["G42"] = dict(pass_=bool(fin(xs) and xs >= G42_SHARE and fin(os_) and os_ >= G42_SHARE), xor_pattern_share=xs, onedim_no_pattern_share=os_,
                      threshold=G42_SHARE)
    return out


# ======================================================================================================================================
# Linear rung (causal, from C-Q3) and the statements
# ======================================================================================================================================
def linear_rung(q3, concept):
    """'one direction' if the 1-dim ablation leaves g_abl / g_0 <= 0.5 on the concept's instructed texts; 'two-dimensional span' if it does
    not but the span ablation does; otherwise 'not removed within the tested 2-dim span'. g_0 <= 0: 'not decidable (no baseline gap)'."""
    k1, k2, a1, cc = RUNG_INPUTS[concept]
    if q3 is None: return dict(verdict="not run (round-2 output missing: unlearn/q3.json)")
    if not RP.ran(q3): return dict(verdict=f"not run (C-Q3: {RP.status_of(q3)})")
    one = RP.get(q3, "linear_rung", "inputs", k1) or {}; two = RP.get(q3, "linear_rung", "inputs", k2) or {}
    if not one:                                                                   # fall back to the ablation entries
        one = dict(RP.get(q3, "ablations", a1, "texts", cc, default={}) or {}, ablation=a1)
        two = dict(RP.get(q3, "ablations", "span", "texts", cc, default={}) or {}, ablation="span")
    g0 = RP.num(one.get("g0")); r1, r2 = RP.num(one.get("ratio")), RP.num(two.get("ratio"))
    res = dict(inputs={"1dim": one, "span": two}, g0=g0, ratio_1dim=r1, ratio_span=r2, threshold=GAP_POINT,
               in_session_view=RP.get(q3, "linear_rung", concept))
    if not (fin(g0) and g0 > 0): res["verdict"] = "not decidable (no baseline gap: g_0 <= 0)"
    elif fin(r1) and r1 <= GAP_POINT: res["verdict"] = "one direction"
    elif fin(r2) and r2 <= GAP_POINT: res["verdict"] = "two-dimensional span"
    else: res["verdict"] = "not removed within the tested 2-dim span"
    res["agrees_with_in_session"] = (res["in_session_view"] is None) or (res["in_session_view"] == res["verdict"])
    return res


def nonlinear_verdict(stat, g42, g41):
    if stat is None: return "not run"
    if not g42: return "not valid (calibration failed)" if g42 is False else "not valid (calibration not run)"
    if stat:
        return "pattern (within the top-32 PCs)" if g41 is not False else "pattern claim withheld (G41: circle gate failed)"
    return "no nonlinear residue found"


def statement(lin, nlv):
    """Both rungs together (C-Q1: e.g. 'one direction, with a nonlinear residue')."""
    if nlv.startswith("pattern (within"):
        return f"{lin}, with a nonlinear residue (pattern within the top-32 PCs)" if lin else "nonlinear residue: pattern within the top-32 PCs"
    return f"{lin}; {nlv}" if lin else nlv


# ======================================================================================================================================
# Driver
# ======================================================================================================================================
def run(r1_dir, r2_root, out_path):
    t0 = time.time(); dev = device(); inp = RP.Inputs(r1_dir, r2_root, out_path)
    info = RP.r1_info(r1_dir); g41 = RP.g41(inp); labels = RP.desc_labels(info, 14)
    out = dict(script="r2_linearity.py", plan=RP.PLAN_REL, plan_sha256=RP.PLAN_SHA256, part="C-Q1", block=14, device=str(dev), mock=MOCK, seeds=SEEDS,
               settings=dict(n_flip=N_FLIP, n_perm=N_PERM, n_calibration=N_CAL, top_pcs=N_TOP, ridge=f"{RIDGE} x tr(Sigma) / D", ridge_quadratic=RIDGE_Q,
                             inlp_max=INLP_MAX, inlp_auc=INLP_AUC, pattern_auc=PATTERN_AUC, pattern_quantile=PATTERN_Q, g42_share=G42_SHARE, gap_point=GAP_POINT),
               labels=labels, G41=g41, G_desc=info.get("G_desc"))
    tok = inp.pt("axes/tok14.pt"); q3 = inp.json("unlearn/q3.json"); q3_ok = RP.ran(q3)
    final14 = bool(info["final"].get("14"))
    ST = None
    if final14:
        import r2_rk as R
        try: ST = R.load_r1_state(r1_dir, 14)
        except Exception as e:                                                    # noqa: BLE001
            out.setdefault("errors", []).append(f"R1.state[14]: {e!r}")
    concepts = {}; statements = {}
    if tok is None:
        out["status"] = "not run (round-2 output missing: axes/tok14.pt)"
        for c in CONCEPTS:
            concepts[c] = dict(status=out["status"], linear_rung=linear_rung(q3, c) if c != "sea" else None)
    else:
        fold = tok["fold"]; sets = tok.get("sets", {}); H = halves(fold); P = len(fold)
        flips = sign_rows(N_FLIP, P, SEEDS["flips"]); perm = sign_rows(N_PERM, P, SEEDS["perm"])
        data = {}
        for c, (a, b) in CONCEPTS.items():
            if a in sets and b in sets: data[c] = Concept([t for t in sets[a]["tok"]], [t for t in sets[b]["tok"]], fold, dev)
        # calibration (sea tokens, u_T's d' on the RSI tokens)
        cal = None
        if "sea" in data and "RSI" in data and ST is not None and "u_T" in ST.get("axes", {}):
            uT = ST["axes"]["u_T"].double().to(dev); d_T = dprime((data["RSI"].X @ (uT / uT.norm())).cpu(), data["RSI"].ytok().cpu())
            if fin(d_T):
                cal = calibration(data["sea"], d_T, ST["axes"]["sea"].double(), H); log(f"calibration: G42 {cal['G42']}")
            else: cal = dict(status="not computable (u_T's d' not finite)", G42=dict(pass_=None))
        else:
            why = ("round-1 output missing (R1.state[14] or the round-1 token sets)" if (ST is None or "sea" not in data or "RSI" not in data)
                   else "R1.state[14]['axes'] has no u_T")
            cal = dict(status=f"not run ({why})", G42=dict(pass_=None))
        out["calibration"] = cal; g42 = RP.get(cal, "G42", "pass_")
        out["G42"] = dict(cal.get("G42", {}), status=cal.get("status", "ran"),
                          consequence=None if g42 else "C-Q1's nonlinear rung 'not valid (calibration failed)'; the causal linear rung is unaffected")
        for c in CONCEPTS:
            if c not in data:
                concepts[c] = dict(status=NOT_RUN_R1 if c != "RGI" else "not run (round-2 output missing: RGI token sets)"); continue
            st = concept_stats(data[c], H, flips, perm); nlv = nonlinear_verdict(st["nonlinear"]["pattern_statistic"], g42, RP.get(g41, "passes"))
            concepts[c] = dict(status="ran", **st, nonlinear_rung=nlv); log(f"{c}: nonlinear AUC {st['nonlinear']['auc']:.3f} p99 {st['nonlinear']['p99']:.3f} -> {nlv}")
        out["status"] = "ran"
    for c in CONCEPTS:
        e = concepts[c]
        if c != "sea": e["linear_rung"] = linear_rung(q3, c)
        lin = e["linear_rung"]["verdict"] if c != "sea" else None
        nlv = e.get("nonlinear_rung", e.get("status", "not run"))
        lab = list(labels)
        if c == "RGI" and q3_ok and RP.get(q3, "GA14") is False: lab.append(RP.LBL_RGI)
        e["labels"] = lab; e["statement"] = statement(lin, nlv); statements[c] = e["statement"]
    out.update(concepts=concepts, statements=statements, inputs=inp.record(["axes/tok14.pt", "unlearn/q3.json", "unlearn_toy/t3.json"]),
               finder=inp.log, r1_finder=info.get("finder"), seconds=time.time() - t0)
    RP.write_json(out_path, out); log(f"C-Q1 -> {out_path} ({time.time() - t0:.0f}s): {statements}")
    return out


# ======================================================================================================================================
# Self-test (synthetic token states with planted structure; every branch)
# ======================================================================================================================================
def _synth_tokens(P=24, L=12, D=40, seed=0, kind="linear", strength=1.2, n_lead=36):
    """Synthetic per-token states: power-law spectrum with n_lead leading directions, a text-level offset, and a planted concept:
    'linear' (+ y s e1), 'xor' (+ s z e1 + s y z e2, equal class means), 'none'. Returns (pro, anti, fold, e1, e2)."""
    g = torch.Generator().manual_seed(seed); sd = torch.cat([torch.linspace(3.0, 1.0, n_lead), torch.full((D - n_lead,), 0.15)])
    Q, _ = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float64)); e1, e2 = Q[:, 0], Q[:, 1]
    fold = torch.zeros(P, dtype=torch.bool); fold[torch.randperm(P, generator=g)[: P // 2]] = True
    zz = {lab: torch.cat([torch.ones(P // 2), -torch.ones(P - P // 2)])[torch.randperm(P, generator=g)] for lab in (1.0, -1.0)}   # z balanced within each class
    pro, anti = [], []
    for p in range(P):
        for y, lst in ((1.0, pro), (-1.0, anti)):
            z = float(zz[y][p])
            base = (torch.randn(D, generator=g, dtype=torch.float64) * sd * 0.3) @ Q.T
            X = (torch.randn(L, D, generator=g, dtype=torch.float64) * sd) @ Q.T + base
            if kind == "linear": X = X + strength * y * e1 * sd[0]
            elif kind == "xor": X = X + strength * z * e1 * sd[0] + strength * y * z * e2 * sd[1]
            lst.append(X.half())
    return pro, anti, fold, e1, e2


def selftest(work):
    os.makedirs(work, exist_ok=True); res = []; t0 = time.time(); dev = torch.device("cpu")

    def check(name, cond, info=""):
        res.append(dict(test=name, ok=bool(cond), info=str(info))); print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)
    global N_PERM, N_CAL, N_FLIP
    N_PERM, N_FLIP = 100, 200
    # LEACE: zero label cross-covariance in the fitted fold; probe = (S + rI)^-1 delta
    pro, anti, fold, e1, e2 = _synth_tokens(kind="linear", seed=1)
    C = Concept(pro, anti, fold, dev); m = C.tokens(C.texts(True)); X, y = C.X[m], C.ytok()[m]; E = leace(X, y); Xe = erase(X, E)
    cov = float(((Xe - Xe.mean(0)).T @ (y - y.mean())).abs().max()); check("LEACE leaves zero label cross-covariance in its fold", cov < 1e-9, f"{cov:.1e}")
    _, S, delta = moments(Xe, y); check("class means equal after LEACE (in-sample)", float(delta.abs().max()) < 1e-9, f"{float(delta.abs().max()):.1e}")
    w = probe_dir(*moments(X, y)[1:]); Xc = X - X.mean(0); lam = ridge_lam(Xc.T @ Xc / len(X), RIDGE)
    wls = torch.linalg.solve(Xc.T @ Xc / len(X) + lam * torch.eye(X.shape[1], dtype=torch.float64), Xc.T @ (y - y.mean()) / len(X))
    check("ridge probe direction = closed-form ridge least squares", abs(float(R_cos(w, wls))) > 1 - 1e-9)
    check("AUC: Mann-Whitney with ties", abs(auc_cols(np.array([1, 2, 2, 3.]), np.array([0, 0, 1, 1], bool)) - 0.875) < 1e-12 and
          np.isnan(auc_cols(np.array([1., 2.]), np.array([1, 1], bool))))
    H = halves(fold); flips = sign_rows(N_FLIP, len(fold), 1); perm = sign_rows(N_PERM, len(fold), 2)
    st = concept_stats(C, H, flips, perm)
    check("linear plant: md AUC high and significant, INLP r* >= 1, no 'pattern'", st["md_auc"]["auc"] > 0.9 and st["md_auc"]["p"] < 0.05 and
          st["inlp"]["r_star_A_to_B"] not in (0,) and not st["nonlinear"]["pattern_statistic"], (st["md_auc"]["auc"], st["inlp"]["r_star_A_to_B"], st["nonlinear"]["auc"]))
    check("LEACE inside (4): erased training fold has zero label cross-covariance", st["nonlinear"]["leace_max_abs_cov"] < 1e-8, st["nonlinear"]["leace_max_abs_cov"])
    pro, anti, fold, e1, e2 = _synth_tokens(kind="xor", seed=3, strength=1.5)
    Cx = Concept(pro, anti, fold, dev); sx = concept_stats(Cx, halves(fold), flips, perm)
    check("XOR plant (equal class means): md AUC near chance, degree-2 probe finds 'pattern'", sx["nonlinear"]["pattern_statistic"] and sx["md_auc"]["auc"] < 0.8,
          (sx["md_auc"]["auc"], sx["nonlinear"]["auc"], sx["nonlinear"]["p99"]))
    pro, anti, fold, e1, e2 = _synth_tokens(kind="none", seed=4)
    Cn = Concept(pro, anti, fold, dev); sn = concept_stats(Cn, halves(fold), flips, perm)
    check("no signal: no 'pattern', INLP r* = 0, md AUC not significant", not sn["nonlinear"]["pattern_statistic"] and sn["inlp"]["r_star_A_to_B"] == 0, (sn["nonlinear"]["auc"], sn["md_auc"]))
    # empty texts are dropped, the other text of the prompt stays
    pro2 = list(pro); pro2[0] = pro2[0][:0]; Ce = Concept(pro2, anti, fold, dev)
    check("empty text dropped (other text kept)", Ce.summary()["n_texts"] == 2 * len(fold) - 1 and Ce.summary()["n_anti"] == len(fold))
    # calibration: strong d' -> G42 passes; tiny d' -> the XOR plant is not found -> G42 fails
    pro, anti, fold, _, _ = _synth_tokens(kind="none", seed=5, P=32, L=16)
    Cs = Concept(pro, anti, fold, dev); u_sea = torch.randn(Cs.X.shape[1], generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    N_CAL = 10
    cal = calibration(Cs, 2.0, u_sea, halves(fold), n_rep=10)
    B = plant_basis(Cs.X, u_sea)
    check("plant basis: orthonormal, in the top-32 PC span, orthogonal to u_sea", float((B.T @ B - torch.eye(B.shape[1], dtype=torch.float64)).abs().max()) < 1e-9
          and float((B.T @ (u_sea / u_sea.norm())).abs().max()) < 1e-9 and B.shape[1] == N_TOP - 1)
    pl = cal["plants"]["onedim"]["replicates"][0]
    check("plants scaled to the target d' (1-dim plant's d' close to 2)", abs(pl["dprime_planted"] - 2.0) < 0.6, pl["dprime_planted"])
    check("calibration at d' 2: G42 passes (XOR found, 1-dim not)", cal["G42"]["pass_"], cal["G42"])
    cal0 = calibration(Cs, 0.05, u_sea, halves(fold), n_rep=6)
    check("calibration at d' 0.05: G42 fails (XOR not found)", cal0["G42"]["pass_"] is False, cal0["G42"])
    # linear rung branches
    mk = lambda r1, r2, g0=1.0: dict(status="ran", GA14=True, linear_rung=dict(inputs=dict(RSI_1dim=dict(ratio=r1, g0=g0), RSI_span=dict(ratio=r2, g0=g0),
                                                                                     RGI_1dim=dict(ratio=r1, g0=g0), RGI_span=dict(ratio=r2, g0=g0))))
    check("linear rung: one direction / span / not removed / no baseline / not run",
          linear_rung(mk(0.4, 0.3), "RSI")["verdict"] == "one direction" and linear_rung(mk(0.7, 0.5), "RSI")["verdict"] == "two-dimensional span"
          and linear_rung(mk(0.7, 0.6), "RGI")["verdict"] == "not removed within the tested 2-dim span"
          and linear_rung(mk(0.1, 0.1, g0=-0.2), "RSI")["verdict"].startswith("not decidable")
          and linear_rung(None, "RSI")["verdict"].startswith("not run") and linear_rung(dict(status="not run (round-1 output missing)"), "RSI")["verdict"].startswith("not run"))
    check("nonlinear verdict branches (G42 failed / not run, G41 failed, pattern, none)",
          nonlinear_verdict(True, False, True) == "not valid (calibration failed)" and nonlinear_verdict(True, None, True) == "not valid (calibration not run)"
          and nonlinear_verdict(True, True, False).startswith("pattern claim withheld") and nonlinear_verdict(True, True, None) == "pattern (within the top-32 PCs)"
          and nonlinear_verdict(False, True, True) == "no nonlinear residue found")
    check("statements combine both rungs", statement("one direction", "pattern (within the top-32 PCs)") == "one direction, with a nonlinear residue (pattern within the top-32 PCs)"
          and statement("one direction", "no nonlinear residue found") == "one direction; no nonlinear residue found" and statement(None, "no nonlinear residue found") == "no nonlinear residue found")
    # end to end on a synthetic r2_root without round-1 files: concepts that need R1 'not run', RGI computed, calibration 'not run'
    r2 = os.path.join(work, "r2"); r1 = os.path.join(work, "r1"); os.makedirs(os.path.join(r2, "r2_q1", "axes"), exist_ok=True); os.makedirs(r1, exist_ok=True)
    pro, anti, fold, _, _ = _synth_tokens(kind="linear", seed=7)
    torch.save(dict(block=14, fold=fold, sets=dict(RGI_PRO=dict(tok=pro), RGI_ANTI=dict(tok=anti)), r1_sets="not run (round-1 output missing)"),
               os.path.join(r2, "r2_q1", "axes", "tok14.pt"))
    o = run(r1, r2, os.path.join(work, "out", "unlearn", "q1.json"))
    check("end to end without round-1 files: RSI / sea 'not run', RGI ran, calibration not run, linear rung not run (q3 missing)",
          o["concepts"]["RSI"]["status"] == NOT_RUN_R1 and o["concepts"]["RGI"]["status"] == "ran" and o["calibration"]["status"].startswith("not run")
          and o["concepts"]["RGI"]["nonlinear_rung"] == "not valid (calibration not run)" and o["concepts"]["RGI"]["linear_rung"]["verdict"].startswith("not run"), o["statements"])
    o2 = run(r1, os.path.join(work, "empty_r2"), os.path.join(work, "out2", "q1.json"))
    check("end to end without tok14.pt: every concept 'not run'", o2["status"].startswith("not run") and all(v["status"].startswith("not run") for v in o2["concepts"].values()))
    ok = all(r["ok"] for r in res); RP.write_json(os.path.join(work, "r2_linearity_selftest.json"), dict(ok=ok, tests=res, seconds=time.time() - t0))
    print(f"r2_linearity selftest: {sum(r['ok'] for r in res)}/{len(res)} passed ({time.time() - t0:.0f}s)", flush=True)
    return ok


def R_cos(a, b): return float(a @ b / (a.norm() * b.norm() + 1e-300))


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if len(a) >= 2 and a[0] == "selftest": return 0 if selftest(a[1]) else 1
    if len(a) != 3: print(__doc__); return 2
    run(*a); return 0


if __name__ == "__main__":
    sys.exit(main())
