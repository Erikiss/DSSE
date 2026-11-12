"""Round 2 of README sections 11/12, Part A statistics (frozen plan results/ju_space/plans/abschnitt11_12_runde2_plan.md, sections
2.6, 3.1 A2-A5 and A7's in-session decisions, 6 (G30-G32), 9.2): geometry statistics of the A1 states, the causal (A3) statistics with
the g_SELF spillover, delta per block (A4) and the over-block relation to delta (A5), r_DiD when both models exist. Runs on the GPU
(torch, env DEVICE=cuda) in session r2_q3; the same code runs on the CPU for the tests.

    python r2_rgi_stats.py <root> <r1_dir> <r2_dir> [all | selftest] [--model qwen3-1.7b | qwen3-0.6b] [--force]

Reads r2_rgi_geom.py's outputs of the model (local/geom_<m>.npz, local/delta_states_<m>.npz, local/pool_<m>.pt, rgi/causal_<m>.pt,
rgi/geom_meta_<m>.json); writes rgi/geom_stats_<m>.json and rgi/geom_draws_<m>.npz (per-draw values), and a resumable per-block cache
local/geom_stats_cache_<m>.pt. <root> and <r1_dir> are not read. Env: DEVICE, NT, MOCK (tests only: R2_NDRAW / R2_NDBOOT may lower the
numbers of draws, default 64 and 20 under MOCK; a real run always uses the plan's 1,000 and 200).

Estimator (A2, the RGI design's sim_tg.py extended to the owner x substrate factorial; frames pooled = mean of the F1 and F2 states):
  units x(F, l, o) per family F, lexeme l, object o; cells c = (owner, substrate) for the 48 model objects (12 cells x 4 paraphrases)
  and the 3 non-model categories (4 items each). For a contrast A-B, a lexeme split of A and of B and a paraphrase split (the same split
  pattern of the paraphrase index for every substrate and category): d(c, h, i) = weighted mean of A over lexeme half h and paraphrase
  half i of cell c minus the same for B. Pp(c, d) = mean over h != h', i != i' of <d(c, h, i), d(d, h', i')>; P_rel(c) = mean over
  h = h', i != i' within cell c. P_cell, P_own (same substrate, other owner), P_sub (same owner, other substrate), P_gen (other owner and
  substrate) are means of Pp over the 12 model cells / ordered cell pairs of each class; FG = P_gen / P_cell, OG = P_own / P_cell,
  SG = P_sub / P_cell, q = P_cell / P_rel, FG_all over all 15 cells (non-model cells differ from every cell in owner and substrate),
  OG_LR = P_own / P_cell within the LR column. rho(SELF, X, s) = Pp((SELF, s), (X, s)) / sqrt(Pp of both cells) (NaN unless both
  P_cell > 0); rho_S = nanmean over X in {SMALL, LARGE, FAMILY} and the substrates; rho_OO = nanmean over the pairs of non-self owners;
  sigma_self = rho_OO - rho_S; the same over the LR column alone. Cross-validated, disattenuated direction cosine of the general parts:
  cos(A, B) = P_gen(A, B) / sqrt(P_gen(A) P_gen(B)) (cross-contrast P_gen with disjoint halves; NaN unless both P_gen > 0).
  Point estimates average the P's over all split combinations (the 81 joint combinations of the IMP, CHG and DES lexeme splits and the
  paraphrase split = each contrast's 27, or 9 with WOR / PRA, equally often; OD-A1), then form the ratios. Bayesian bootstrap: 1,000
  draws with Dirichlet(1) weights on the lexemes of every family, on the 4 paraphrases of every substrate (shared by the owners: the
  paraphrase is the substrate's wording) and on the 4 items of every non-model category, one random split per draw; the same draws for
  every block, reading, metric, frame variant and contrast, and for both models (seed 1101).
Variants: '.' and object readings x raw and whitened metric ((Sigma_b + mean eigenvalue I)^-1/2 from O8) x frames pooled / F1 / F2,
and the lexical reference; draws for the '.' reading, raw metric, all three frame variants (the rule quantities); point estimates for the
rest (robustness), and leave-one-owner-out / leave-one-substrate-out point estimates of the primary variant.
Order (2.6, A7): C0 and 'uncomposed' at all 28 blocks with pointwise 95 % percentile intervals -> passing blocks -> P = the longest
contiguous run (OD-A4: block 27 included; ties: higher mean delta_MMLU, then the earlier run) -> sup-t simultaneous 95 % bands over P
for every per-block quantity with draws: point +- c sd_b, c = 95 % quantile over draws of max_{b in P} |draw_b - mean_b| / sd_b.
A3: W from rgi/causal_<m>.pt; units = the IMP-DES and CHG-DES gradients of lexeme half h (fixed halves) per object; the same estimator
over paraphrases (point: the 3 paraphrase splits; draws: the A2 draws' paraphrase weights and split). Composed (G32): FG_causal(CHG-DES)
< 0.8. g_SELF spillover per block: s_o = w_o . unit(g_SELF), w_o = the IMP-DES gradient (mean of both halves) of model object o, g_SELF
= the F2 SELF contrast of IMP-DES (rgi/dirs_<m>.pt); z = (mean - null mean) / null sd against 200 directions drawn with the O8 pool
covariance of the block (unit((z * sqrt(ev)) @ U^T), torch seed 1105 + block); FR = other / SELF mean only where z_self >= 3.
A4: delta_MMLU = O7's dn_delta values (geom_meta); 200-draw within-topic text bootstrap of F - floor (floor = mean over 50 label
permutations per draw; draws and permutations shared by blocks and readings; seed 1103); delta_OBJ = F - floor (50 permutations from
np.random.default_rng([1104, block]), F1 then F2) of the 15 object categories on the neutral sentences ('.'), per frame, series = mean of the frames; tau = cross-validated share of
the general IMP-DES direction's energy in the span of the centred O7 topic means (last token for '.', question mean for the object
reading) against tau of CHG-DES and the mean over the 200 pool-covariance directions of |P v|^2 / |v|^2 (also tr(P Sigma) / tr(Sigma));
anisotropy = mean pairwise cosine of the frame-pooled DES states. Separability (G31) over P: SD (ddof 0) of delta's cubic-in-block-index
residual >= 2 x the mean bootstrap SD of delta over P.
A5 over P (k = floor(|P| / 3); top / bottom third by delta_MMLU 'last' (object reading: 'mean')): D = mean(top) - mean(bottom) with a 95 %
percentile interval over the draws; Spearman with delta_MMLU and delta_OBJ; r3 = corr of the cubic residuals with one-sided surrogate
p-values from 2,000 phase-randomised, mirror-padded surrogates of delta's residual, each residualised again (seed 1102; the RGI design's
sim_depth.py), r3 per draw; nuisance r3 after partialling the anisotropy's cubic residual, with its surrogate p; D in the whitened metric.
Series: dFG, FG(IMP-DES), rho_S, sigma_self (pooled, F1, F2), FG_causal, rho_causal (also over the composed blocks of P).
r_DiD: corr over P_1.7B & P_0.6B of dFG_1.7B - dFG_0.6B with delta_1.7B - delta_0.6B, surrogate p (phase-randomised delta difference).
The verdicts of A7 are made by r2_rgi_verdict.py (r2_ana) from these numbers; the u_T_perp spillover (A3) uses spillover() below."""
import argparse, datetime, hashlib, itertools, json, math, os, shutil, sys, time
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); JU = os.path.dirname(HERE)
for _p in (HERE, JU):
    if _p not in sys.path: sys.path.insert(0, _p)
import r2_rgi_geom as GEOM
import r2_stimuli as STIM
from r2_rgi_geom import FAMS, NLEX, VF_OFF, NVF, NOBJ, NCELL, CELL_NAMES, CELL_OF, PARA_OF, SELF_OBJ, OTHER_OBJ, sentence_layout

MOCK = os.environ.get("MOCK") == "1"
VERSION = "r2_rgi_stats 1.0 (plan sha256 5e476340dfdb1d82)"
PLAN_SHA256 = GEOM.PLAN_SHA256
CONTRASTS = (("IMP", "DES"), ("CHG", "DES"), ("IMP", "CHG"), ("WOR", "CHG"), ("WOR", "DES"), ("PRA", "DES"))
CN = tuple(f"{a}-{b}" for a, b in CONTRASTS); NX = len(CN)
CROSS = (("cos_quality", 2, 3), ("cos_valence", 2, 5))                                     # cos(g_IMP-CHG, g_WOR-CHG), cos(g_IMP-CHG, g_PRA-DES)
CAUSAL_CN = ("IMP-DES", "CHG-DES")
LEX_SPLITS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))                        # the 3 ways to halve 4 items (sim_tg.py)
SPLIT_FAMS = ("IMP", "CHG", "DES")                                                          # families with 4 lexemes (WOR, PRA: halves (0,), (1,))
SEEDS = dict(draws=1101, surrogates=1102, delta_boot=1103, delta_obj=1104, randdirs=1105, did=1106)
LEVEL = 0.95
N_DRAWS = 1000 if not MOCK else int(os.environ.get("R2_NDRAW", "64"))        # MOCK (code tests): 64 draws unless R2_NDRAW is set
N_DBOOT = 200 if not MOCK else int(os.environ.get("R2_NDBOOT", "20"))
N_SURR, N_PERM, N_RAND = 2000, 50, 200
OWN = np.array([c // 3 if c < 12 else 4 + c - 12 for c in range(NCELL)]); SUB = np.array([c % 3 if c < 12 else 3 + c - 12 for c in range(NCELL)])
MODEL_CELL = np.arange(NCELL) < 12
OWNERS, SUBS = GEOM.OWNERS, GEOM.SUBS


def log(msg):
    print(f"[r2_rgi_stats {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def device():
    return torch.device(os.environ.get("DEVICE", "cpu"))


# ======================================================================================================================================
# Draw design: Dirichlet weights and splits (A2 bootstrap); the 81 split combinations (point estimates)
# ======================================================================================================================================
def draw_design(R, seed=SEEDS["draws"]):
    """R Bayesian-bootstrap draws (np.random.default_rng(seed), in this order per draw: Dirichlet(1) weights of the IMP, CHG, DES, WOR,
    PRA lexemes; the lexeme split (0..2) of IMP, CHG, DES; Dirichlet(1) weights of the 4 paraphrases of W, KV, LR and the 4 items of
    TOOL, PROCESS, SKILL; the paraphrase split (0..2))."""
    rng = np.random.default_rng(seed); lw = np.zeros((R, NVF)); ls = np.zeros((R, 3), np.int64); pw = np.zeros((R, 6, 4)); ps = np.zeros(R, np.int64)
    for r in range(R):
        for F in FAMS: lw[r, VF_OFF[F]:VF_OFF[F] + NLEX[F]] = rng.dirichlet(np.ones(NLEX[F]))
        ls[r] = rng.integers(0, 3, 3)
        for g in range(6): pw[r, g] = rng.dirichlet(np.ones(4))
        ps[r] = rng.integers(0, 3)
    return dict(lex_w=lw, lex_split=ls, para_w=pw, para_split=ps, seed=seed)


def point_design():
    """The 81 joint split combinations (IMP x CHG x DES lexeme splits x paraphrase split) with uniform weights."""
    combos = list(itertools.product(range(3), range(3), range(3), range(3))); R = len(combos)
    return dict(lex_w=np.ones((R, NVF)), lex_split=np.array([c[:3] for c in combos]), para_w=np.ones((R, 6, 4)), para_split=np.array([c[3] for c in combos]))


def weights(des):
    """Lam [R, 2 NX, NVF]: row 2x + h = the lexeme-half-h mean of A minus that of B for contrast x = (A, B) (weights normalised within the
    half); Om [R, 10, NVF] the family-half rows (2 f + h); Psi [R, NCELL, 2, 4]: paraphrase / item weights of half i in cell c."""
    lw, ls, pw, ps = des["lex_w"], des["lex_split"], des["para_w"], des["para_split"]; R = len(ps)
    Om = np.zeros((R, 2 * len(FAMS), NVF))
    for f, F in enumerate(FAMS):
        n, o = NLEX[F], VF_OFF[F]
        for h in range(2):
            m = np.zeros((R, n))
            if n == 4:
                for r in range(R): m[r, list(LEX_SPLITS[ls[r, SPLIT_FAMS.index(F)]][h])] = 1.0
            else: m[:, h] = 1.0
            w = lw[:, o:o + n] * m; Om[:, 2 * f + h, o:o + n] = w / w.sum(1, keepdims=True)
    Lam = np.zeros((R, 2 * NX, NVF))
    for x, (A, B) in enumerate(CONTRASTS):
        for h in range(2): Lam[:, 2 * x + h] = Om[:, 2 * FAMS.index(A) + h] - Om[:, 2 * FAMS.index(B) + h]
    Psi = np.zeros((R, NCELL, 2, 4))
    for c in range(NCELL):
        g = SUB[c] if c < 12 else 3 + (c - 12)
        for i in range(2):
            m = np.zeros((R, 4))
            for r in range(R): m[r, list(LEX_SPLITS[ps[r]][i])] = 1.0
            w = pw[:, g] * m; Psi[:, c, i] = w / w.sum(1, keepdims=True)
    return torch.tensor(Lam), torch.tensor(Psi), torch.tensor(Om)


# ======================================================================================================================================
# Core: cross-validated cell-pair products
# ======================================================================================================================================
def gram_units(U):
    """U [NVF, NOBJ, D] -> K [NVF, NOBJ, NVF, NOBJ] (inner products of all verb units)."""
    F = U.reshape(-1, U.shape[-1]); return (F @ F.T).reshape(U.shape[0], U.shape[1], U.shape[0], U.shape[1])


def para_step(Ge, Psi):
    """Ge [R, X, 2, NCELL, 4, X, 2, NCELL, 4] (products of the per-object half contrasts e(x, h, o)), Psi [R, NCELL, 2, 4] ->
    Pp [R, X, X, NCELL, NCELL] (mean over h != h', i != i') and Prel [R, X, NCELL] (h = h', i != i', same cell)."""
    T2 = torch.einsum("rcik,rxhckygdm->rxhciygdm", Psi, Ge); G = torch.einsum("rxhciygdm,rdjm->rxhciygdj", T2, Psi)
    Pp = (G[:, :, 0, :, 0, :, 1, :, 1] + G[:, :, 0, :, 1, :, 1, :, 0] + G[:, :, 1, :, 0, :, 0, :, 1] + G[:, :, 1, :, 1, :, 0, :, 0]) / 4   # [R, X, c, Y, d]
    Dg = torch.diagonal(torch.diagonal(G, dim1=1, dim2=5), dim1=2, dim2=5)                                                     # [R, h, i, g, j, X, c]
    Prel = (Dg[:, 0, 0, 0, 1] + Dg[:, 0, 1, 0, 0] + Dg[:, 1, 0, 1, 1] + Dg[:, 1, 1, 1, 0]) / 4
    return Pp.permute(0, 1, 3, 2, 4), Prel


def pp_draws(K, Lam, Psi, chunk):
    """Per-draw Pp [R, NX, NX, NCELL, NCELL], Prel [R, NX, NCELL] of the A2 contrasts from the unit Gram K (device float64)."""
    dev = K.device; out_p, out_r = [], []
    for i in range(0, Lam.shape[0], chunk):
        L = Lam[i:i + chunk].to(dev); Ps = Psi[i:i + chunk].to(dev); R = L.shape[0]
        T = torch.einsum("rav,vowp->raowp", L, K); Ge = torch.einsum("raowp,rbw->raobp", T, L).reshape(R, NX, 2, NCELL, 4, NX, 2, NCELL, 4); del T
        Pp, Prel = para_step(Ge, Ps); out_p.append(Pp.cpu()); out_r.append(Prel.cpu())
    return torch.cat(out_p), torch.cat(out_r)


def point_operators(des):
    """Expected weight outer products over the point design (exact averages over the 81 combinations): LL [2NX, NVF, 2NX, NVF] and
    PP [NCELL, 2, 4, NCELL, 2, 4]. The lexeme and paraphrase splits vary independently over the 81 combinations, so the average of the
    bilinear products factorises."""
    Lam, Psi, _ = weights(des); LL = torch.einsum("rav,rbw->avbw", Lam, Lam) / Lam.shape[0]; PP = torch.einsum("rcik,rdjm->cikdjm", Psi, Psi) / Psi.shape[0]
    return LL, PP


def pp_point(K, LL, PP):
    """Pp [1, NX, NX, NCELL, NCELL], Prel [1, NX, NCELL] averaged over the 81 split combinations (exact)."""
    Ge = torch.einsum("avbw,vowp->aobp", LL.to(K.device), K).reshape(NX, 2, NCELL, 4, NX, 2, NCELL, 4)
    G = torch.einsum("cikdjm,xhckygdm->xhciygdj", PP.to(K.device), Ge)
    Pp = (G[:, 0, :, 0, :, 1, :, 1] + G[:, 0, :, 1, :, 1, :, 0] + G[:, 1, :, 0, :, 0, :, 1] + G[:, 1, :, 1, :, 0, :, 0]) / 4
    Dg = torch.diagonal(torch.diagonal(G, dim1=0, dim2=4), dim1=1, dim2=4)                                                     # [h, i, g, j, X, c]
    Prel = (Dg[0, 0, 0, 1] + Dg[0, 1, 0, 0] + Dg[1, 0, 1, 1] + Dg[1, 1, 1, 0]) / 4
    return Pp.permute(0, 2, 1, 3)[None].cpu(), Prel[None].cpu()


def causal_ge(Wb):
    """A3 units: Wb [NOBJ, 4, D] (rows IMP-DES h1, h2, CHG-DES h1, h2) -> Ge [2, 2, NCELL, 4, 2, 2, NCELL, 4] (x, h, cell, para)."""
    E = Wb.permute(1, 0, 2).reshape(4 * NOBJ, -1); G = (E @ E.T).reshape(2, 2, NCELL, 4, 2, 2, NCELL, 4); return G


# ======================================================================================================================================
# Statistics from the products
# ======================================================================================================================================
def cell_masks(keep=None):
    """Masks [NCELL, NCELL] (float64) over ordered cell pairs; keep [NCELL] bool restricts the model cells (leave-one-out)."""
    k = torch.ones(NCELL, dtype=torch.bool) if keep is None else torch.as_tensor(keep, dtype=torch.bool)
    model = torch.as_tensor(MODEL_CELL); km = k & model; o = torch.as_tensor(OWN); s = torch.as_tensor(SUB)
    so, ss = o[:, None] == o[None], s[:, None] == s[None]; pm = km[:, None] & km[None]; pa = k[:, None] & k[None]; lr = (s == 2)
    M = dict(cell=torch.diag(km), own=pm & ss & ~so, sub=pm & so & ~ss, gen=pm & ~so & ~ss, cell_all=torch.diag(k), gen_all=pa & ~so & ~ss,
             cell_LR=torch.diag(km & lr), own_LR=pm & ss & ~so & lr[:, None])
    return {n: v.double() for n, v in M.items()}


def _mm(P, M):
    return (P * M).sum((-1, -2)) / M.sum()


def _ratio(a, b):
    return torch.where(b != 0, a / torch.where(b != 0, b, torch.ones_like(b)), torch.full_like(a, float("nan")))


def self_rho(Pd, subs):
    """rho_S, rho_OO [R, X] over the given substrates (disattenuated cross-owner products within a substrate; NaN unless both P_cell > 0)."""
    dg = Pd.diagonal(dim1=-2, dim2=-1)

    def rho(c1, c2):
        a, b = dg[..., c1], dg[..., c2]; ok = (a > 0) & (b > 0)
        return torch.where(ok, Pd[..., c1, c2] / torch.sqrt(torch.where(ok, a * b, torch.ones_like(a))), torch.full_like(a, float("nan")))
    rs = torch.stack([rho(s, 3 * x + s) for x in (1, 2, 3) for s in subs], -1); ro = torch.stack([rho(3 * a + s, 3 * b + s) for a, b in ((1, 2), (1, 3), (2, 3)) for s in subs], -1)
    return torch.nanmean(rs, -1), torch.nanmean(ro, -1)


def geo_stats(Pp, Prel, names=CN, cross=CROSS, keep=None, self_stats=True):
    """Statistics [R] from Pp [R, X, X, NCELL, NCELL], Prel [R, X, NCELL]. Keys 'stat|contrast' and the derived ones."""
    X = Pp.shape[1]; Pd = Pp[:, range(X), range(X)]; M = cell_masks(keep); out = {}
    km = torch.diag(M["cell"]); P_rel = (Prel * km).sum(-1) / km.sum()
    P = {n: _mm(Pd, M[n]) for n in ("cell", "own", "sub", "gen", "cell_all", "gen_all", "cell_LR", "own_LR")}
    st = dict(P_cell=P["cell"], P_own=P["own"], P_sub=P["sub"], P_gen=P["gen"], P_rel=P_rel, FG=_ratio(P["gen"], P["cell"]), OG=_ratio(P["own"], P["cell"]),
              SG=_ratio(P["sub"], P["cell"]), q=_ratio(P["cell"], P_rel), FG_all=_ratio(P["gen_all"], P["cell_all"]), OG_LR=_ratio(P["own_LR"], P["cell_LR"]))
    if self_stats:
        for tag, subs in (("", (0, 1, 2)), ("_LR", (2,))):
            rs, ro = self_rho(Pd, subs); st[f"rho_S{tag}"], st[f"rho_OO{tag}"], st[f"sigma_self{tag}"] = rs, ro, ro - rs
    for k, v in st.items():
        for x in range(X): out[f"{k}|{names[x]}"] = v[:, x]
    i0, i1 = names.index("IMP-DES"), names.index("CHG-DES")
    out["dFG"] = st["FG"][:, i0] - st["FG"][:, i1]
    if "PRA-DES" in names: out["dFG_val"] = st["FG"][:, i0] - st["FG"][:, names.index("PRA-DES")]
    if self_stats: out["dsigma"] = st["sigma_self"][:, i0] - st["sigma_self"][:, i1]; out["dsigma_LR"] = st["sigma_self_LR"][:, i0] - st["sigma_self_LR"][:, i1]
    for nm, x, y in cross:
        pg = _mm(Pp[:, x, y], M["gen"]); a, b = st["P_gen"][:, x], st["P_gen"][:, y]; ok = (a > 0) & (b > 0)
        out[nm] = torch.where(ok, pg / torch.sqrt(torch.where(ok, a * b, torch.ones_like(a))), torch.full_like(pg, float("nan")))
    return out


def naive_stats(K):
    """Continuity only: same-lexeme cell contrasts (all lexemes and paraphrases) of the 12 model cells: mean pairwise cosine and PC1 share
    (uncentred) per contrast."""
    out = {}
    for x, (A, B) in enumerate(CONTRASTS):
        C = torch.zeros(12, NVF, NOBJ, dtype=K.dtype, device=K.device)
        for c in range(12):
            objs = np.nonzero(CELL_OF == c)[0]
            for l in range(NLEX[A]): C[c, VF_OFF[A] + l, objs] += 1.0 / (NLEX[A] * len(objs))
            for l in range(NLEX[B]): C[c, VF_OFF[B] + l, objs] -= 1.0 / (NLEX[B] * len(objs))
        Gc = torch.einsum("cvo,vowp,dwp->cd", C, K, C); n = Gc.diagonal().clamp_min(1e-300).sqrt(); cs = Gc / n[:, None] / n[None]
        off = ~torch.eye(12, dtype=torch.bool, device=K.device); ev = torch.linalg.eigvalsh(Gc)
        out[f"naive_cos|{CN[x]}"] = float(cs[off].mean()); out[f"pc1_share|{CN[x]}"] = float(ev[-1] / ev.sum())
    return out


# ======================================================================================================================================
# Intervals, bands, passing set
# ======================================================================================================================================
def pointwise(draws, level=LEVEL):
    """Percentile interval per column of draws [R, n] (NaN-aware)."""
    a = (1 - level) / 2 * 100; d = np.asarray(draws, dtype=np.float64)
    with np.errstate(all="ignore"):
        lo = np.array([np.nanpercentile(d[:, j], a) if np.isfinite(d[:, j]).any() else np.nan for j in range(d.shape[1])])
        hi = np.array([np.nanpercentile(d[:, j], 100 - a) if np.isfinite(d[:, j]).any() else np.nan for j in range(d.shape[1])])
    return lo, hi


def supt_band(point, draws, sel, level=LEVEL):
    """Sup-t simultaneous band over the columns sel (bool [n]): point_b +- c sd_b, c = the level quantile over draws of
    max_{b in sel} |draw_b - mean_b| / sd_b (NaN-aware; columns with sd 0 or no finite draw are skipped). Returns lo, hi [n] (NaN outside
    sel) and c."""
    point = np.asarray(point, dtype=np.float64); d = np.asarray(draws, dtype=np.float64); n = len(point); lo = np.full(n, np.nan); hi = np.full(n, np.nan)
    idx = [j for j in np.nonzero(sel)[0] if np.isfinite(d[:, j]).sum() >= 2]
    if not idx: return lo, hi, float("nan")
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(d[:, idx], 0); s = np.nanstd(d[:, idx], 0, ddof=1); use = s > 0
        if not use.any():
            lo[idx], hi[idx] = point[idx], point[idx]; return lo, hi, 0.0
        T = np.nanmax(np.abs(d[:, idx][:, use] - m[use]) / s[use], 1); c = float(np.nanquantile(T[np.isfinite(T)], level)) if np.isfinite(T).any() else float("nan")
    for k, j in enumerate(idx): lo[j], hi[j] = point[j] - c * s[k], point[j] + c * s[k]
    return lo, hi, c


def passing_set(passing, delta, min_len=12):
    """P (A5, OD-A4): the longest contiguous run of passing blocks; ties: the higher mean delta_MMLU, then the earlier run."""
    runs, cur = [], []
    for b, ok in enumerate(passing):
        if ok: cur.append(b)
        elif cur: runs.append(cur); cur = []
    if cur: runs.append(cur)
    if not runs: return [], dict(runs=[], tie=False, ok=False)
    L = max(len(r) for r in runs); best = [r for r in runs if len(r) == L]
    pick = max(best, key=lambda r: (float(np.mean([delta[b] for b in r])), -r[0]))
    return pick, dict(runs=runs, tie=len(best) > 1, ok=len(pick) >= min_len, min_len=min_len)


# ======================================================================================================================================
# Over-block statistics (A5)
# ======================================================================================================================================
def cubic_resid(Y, blocks):
    """Residuals of Y [..., n] on a cubic in the block index (standardised; OLS), as sim_depth.py's resid(v, 3)."""
    b = np.asarray(blocks, dtype=np.float64); z = (b - b.mean()) / (b.std() if b.std() > 0 else 1.0); V = np.vander(z, 4)
    H = V @ np.linalg.pinv(V); Y = np.asarray(Y, dtype=np.float64); return Y - Y @ H.T


def partial_out(Y, a):
    """Y [..., n] minus its OLS fit on [1, a]."""
    V = np.stack([np.ones_like(a), a], 1); H = V @ np.linalg.pinv(V); Y = np.asarray(Y, dtype=np.float64); return Y - Y @ H.T


def corr_rows(Y, x):
    """Pearson correlation of each row of Y [..., n] with x [n] (NaN where an SD is 0 or a value is not finite)."""
    Y = np.asarray(Y, dtype=np.float64); x = np.asarray(x, dtype=np.float64); yc = Y - Y.mean(-1, keepdims=True); xc = x - x.mean()
    with np.errstate(all="ignore"):
        r = (yc * xc).sum(-1) / np.sqrt((yc ** 2).sum(-1) * (xc ** 2).sum())
    return np.where(np.isfinite(r), r, np.nan)


def ranks(Y):
    """Average ranks along the last axis (ties averaged)."""
    Y = np.asarray(Y, dtype=np.float64); gt = (Y[..., :, None] > Y[..., None, :]).sum(-1); eq = (Y[..., :, None] == Y[..., None, :]).sum(-1)
    return gt + (eq - 1) / 2 + 1


def spearman_rows(Y, x):
    return corr_rows(ranks(Y), ranks(x))


def surrogates(x, n, seed, zero_phase=False):
    """Phase-randomised, mirror-padded surrogates of x [m] (sim_depth.py: rfft of [x, x reversed], uniform phases, DC and Nyquist
    phases 0, irfft, first m values); torch on the CPU generator, so every device gives the same numbers. Returns [n, m] float64.
    zero_phase (test only): the original phases are kept, which must return x itself."""
    x = torch.as_tensor(np.asarray(x, dtype=np.float64)); m = torch.cat([x, x.flip(0)]); F = torch.fft.rfft(m); amp = F.abs()
    g = torch.Generator().manual_seed(int(seed)); ph = torch.rand(n, F.shape[0], generator=g, dtype=torch.float64) * 2 * math.pi; ph[:, 0] = 0; ph[:, -1] = 0
    if zero_phase: ph = torch.angle(F)[None].expand(n, -1)
    return torch.fft.irfft(amp * torch.exp(1j * ph), n=len(m))[:, :len(x)].numpy()


def surrogate_pack(delta, blocks, aniso, seed, n=N_SURR):
    """Surrogates of delta's cubic residual over 'blocks', each residualised again, and their anisotropy-partialled versions."""
    xr = cubic_resid(delta, blocks); S = cubic_resid(surrogates(xr, n, seed), blocks); ar = cubic_resid(aniso, blocks)
    return dict(xr=xr, S=S, ar=ar, xp=partial_out(xr, ar), Sp=partial_out(S, ar), seed=seed, n=n)


def pvals(r, rs):
    rs = rs[np.isfinite(rs)]
    if not np.isfinite(r) or not len(rs): return float("nan"), float("nan")
    return float((1 + (rs >= r).sum()) / (1 + len(rs))), float((1 + (rs <= r).sum()) / (1 + len(rs)))


def over_blocks(y, y_draws, blocks, d_mmlu, d_obj, pack, y_white=None, level=LEVEL):
    """The A5 battery for one series over a block set (all inputs indexed by the blocks of the set, in block order)."""
    y = np.asarray(y, dtype=np.float64); n = len(blocks); k = n // 3; out = dict(blocks=list(map(int, blocks)), n=n, k=k)
    a = (1 - level) / 2 * 100
    pc = lambda v: [float(np.nanpercentile(v, a)), float(np.nanpercentile(v, 100 - a))] if np.isfinite(v).any() else [float("nan")] * 2
    if n == 0 or not np.isfinite(y).all(): out["status"] = "not computable (empty block set or non-finite series)"; return out
    order = np.argsort(np.asarray(d_mmlu), kind="stable"); bot, top = order[:k], order[n - k:]
    out.update(top=[int(blocks[i]) for i in top], bottom=[int(blocks[i]) for i in bot])
    D = float(y[top].mean() - y[bot].mean()) if k else float("nan"); out["D"] = D
    out["spearman_mmlu"] = float(spearman_rows(y, d_mmlu)); out["spearman_obj"] = float(spearman_rows(y, d_obj))
    if y_white is not None and k: out["D_whitened"] = float(np.asarray(y_white)[top].mean() - np.asarray(y_white)[bot].mean())
    if y_draws is not None:
        Yd = np.asarray(y_draws, dtype=np.float64); ok = np.isfinite(Yd).all(1); Yd = Yd[ok]; out["n_draws_finite"] = int(ok.sum())
        if k and len(Yd):
            Dd = Yd[:, top].mean(1) - Yd[:, bot].mean(1); out["D_ci"] = pc(Dd)
        if len(Yd):
            out["spearman_mmlu_ci"] = pc(spearman_rows(Yd, d_mmlu)); out["spearman_obj_ci"] = pc(spearman_rows(Yd, d_obj))
    if n >= 6 and pack is not None:
        yr = cubic_resid(y, blocks); r3 = float(corr_rows(yr, pack["xr"])); rs = corr_rows(pack["S"], yr)
        out["r3"] = r3; out["r3_p_pos"], out["r3_p_neg"] = pvals(r3, rs)
        yp = partial_out(yr, pack["ar"]); r3n = float(corr_rows(yp, pack["xp"])); rsn = corr_rows(pack["Sp"], yp)
        out["r3_nuis"] = r3n; out["r3_nuis_p_pos"], out["r3_nuis_p_neg"] = pvals(r3n, rsn)
        if y_draws is not None and len(Yd): out["r3_ci"] = pc(corr_rows(cubic_resid(Yd, blocks), pack["xr"]))
    else: out["r3_status"] = "fewer than 6 blocks: r3 not computed"
    return out


def separability(delta, boot_sd, blocks):
    """G31 over the block set: SD (ddof 0) of delta's cubic residual >= 2 x the mean bootstrap SD of delta."""
    if len(blocks) < 6: return dict(ok=False, status="fewer than 6 blocks", blocks=list(map(int, blocks)))
    r = cubic_resid(np.asarray(delta)[blocks], blocks); sd = float(np.std(r)); bs = float(np.mean(np.asarray(boot_sd)[blocks]))
    return dict(ok=bool(sd >= 2 * bs), resid_sd=sd, mean_boot_sd=bs, ratio=sd / bs if bs > 0 else float("inf"), blocks=list(map(int, blocks)))


def r_did(dfg_a, delta_a, P_a, dfg_b, delta_b, P_b, draws_a=None, draws_b=None, seed=SEEDS["did"], n=N_SURR, level=LEVEL):
    """r_DiD (A5): corr over P_a & P_b of (dFG_a - dFG_b) with (delta_a - delta_b); one-sided surrogate p (phase-randomised,
    mirror-padded surrogates of the delta difference); per-draw values when both models' draws are given (the same draw design)."""
    blocks = sorted(set(P_a) & set(P_b)); out = dict(blocks=blocks, n=len(blocks))
    if len(blocks) < 4: out["status"] = "fewer than 4 common passing blocks"; out["r_DiD"] = float("nan"); return out
    dd = np.asarray(dfg_a)[blocks] - np.asarray(dfg_b)[blocks]; dx = np.asarray(delta_a)[blocks] - np.asarray(delta_b)[blocks]
    r = float(corr_rows(dd, dx)); S = surrogates(dx - dx.mean(), n, seed); rs = corr_rows(S, dd)
    out.update(r_DiD=r, p_pos=pvals(r, rs)[0], p_neg=pvals(r, rs)[1], contiguous=bool(blocks == list(range(blocks[0], blocks[-1] + 1))))
    if draws_a is not None and draws_b is not None:
        Dd = np.asarray(draws_a)[:, blocks] - np.asarray(draws_b)[:, blocks]; v = corr_rows(Dd, dx); a = (1 - level) / 2 * 100
        out["ci"] = [float(np.nanpercentile(v, a)), float(np.nanpercentile(v, 100 - a))]
    return out


# ======================================================================================================================================
# Delta (A4)
# ======================================================================================================================================
def fisher_batch(X, Y):
    """Fisher ratio (dn_delta's definition) for X [..., N, D] and one-hot labels Y [..., N, K] (broadcast)."""
    return GEOM.fisher_t(X, Y)


def fisher_gram(G, Y):
    """The same from the Gram matrix G [..., N, N] of the (centred) states: B = sum_k Y_k' G Y_k / n_k - 1'G1 / N, total = tr G - 1'G1 / N."""
    N = G.shape[-1]; tot = G.sum((-1, -2)) / N; n = Y.sum(-2); S = (Y * (G @ Y)).sum(-2)
    B = (S / n.clamp_min(1)).sum(-1) - tot; T = torch.diagonal(G, dim1=-2, dim2=-1).sum(-1) - tot; return B / (T - B)


def boot_index(labels, nb, seed):
    """nb within-topic bootstrap draws of the text indices (slot j keeps the topic of text j) and, per draw, N_PERM label permutations."""
    rng = np.random.default_rng(seed); labels = np.asarray(labels); groups = [np.nonzero(labels == k)[0] for k in np.unique(labels)]
    idx = np.zeros((nb, len(labels)), np.int64); perm = np.zeros((nb, N_PERM, len(labels)), np.int64)
    for r in range(nb):
        for g in groups: idx[r, g] = rng.choice(g, size=len(g), replace=True)
        for p in range(N_PERM): perm[r, p] = rng.permutation(labels)
    return idx, perm


def delta_boot(X, labels, idx, perm, chunk=8):
    """delta = F - floor per bootstrap draw for one block and reading: X [N, D] (float64, device). Returns [nb] numpy."""
    dev = X.device; X = X - X.mean(0); K = int(labels.max()) + 1; N, D = X.shape; I = torch.eye(K, dtype=torch.float64, device=dev)
    Y0 = I[torch.as_tensor(labels, device=dev)]; use_gram = D > N; G = X @ X.T if use_gram else None; out = []
    for i in range(0, len(idx), chunk):
        ix = torch.as_tensor(idx[i:i + chunk], device=dev); pm = torch.as_tensor(perm[i:i + chunk], device=dev); Yp = I[pm]           # [c, P, N, K]
        if use_gram:
            Gd = G[ix[:, :, None], ix[:, None, :]]; F = fisher_gram(Gd, Y0); Fp = fisher_gram(Gd[:, None], Yp)
        else:
            Xd = X[ix]; F = fisher_batch(Xd, Y0); Fp = fisher_batch(Xd[:, None], Yp)
        out.append((F - Fp.mean(1)).cpu())
    return torch.cat(out).numpy()


def delta_obj(Xn, seed_rng):
    """F - floor (50 permutations from seed_rng) of the 15 object categories for the neutral states Xn [60, D] of one frame and block."""
    labels = CELL_OF; I = np.eye(NCELL); dev = Xn.device
    Y0 = torch.tensor(I[labels], dtype=torch.float64, device=dev); F = float(fisher_batch(Xn, Y0))
    Yp = torch.tensor(np.stack([I[seed_rng.permutation(labels)] for _ in range(N_PERM)]), dtype=torch.float64, device=dev)
    fl = float(fisher_batch(Xn[None], Yp).mean()); return F, fl


# ======================================================================================================================================
# Spillover (A3) and tau (A4)
# ======================================================================================================================================
def spillover(w, is_self, u, nulls):
    """A3 spillover. w [n, D] gradients of the model objects (IMP-DES), is_self [n] bool, u [D] direction, nulls [J, D] null directions
    (normalised here). s_o = w_o . unit(u); S_self / S_other = means over the SELF / other-owner objects; z = (S - mean of the same over
    the nulls) / their SD (ddof 1); FR = S_other / S_self only where z_self >= 3 (critique 22f), else NaN. Used for g_SELF here and for
    u_T_perp (with R1 null_u_T_perp) by r2_rgi_verdict.py."""
    w = torch.as_tensor(w, dtype=torch.float64); u = torch.as_tensor(u, dtype=torch.float64).to(w.device); nl = torch.as_tensor(nulls, dtype=torch.float64).to(w.device)
    sm = torch.as_tensor(np.asarray(is_self, dtype=bool), device=w.device); u = u / u.norm(); nl = nl / nl.norm(dim=1, keepdim=True)
    s = w @ u; sn = w @ nl.T                                                                                # [n], [n, J]
    S_self, S_oth = float(s[sm].mean()), float(s[~sm].mean()); N_self, N_oth = sn[sm].mean(0), sn[~sm].mean(0)
    z = lambda v, nv: float((v - nv.mean()) / nv.std()) if float(nv.std()) > 0 else float("nan")
    zs, zo = z(S_self, N_self), z(S_oth, N_oth)
    return dict(S_self=S_self, S_other=S_oth, z_self=zs, z_other=zo, FR=S_oth / S_self if (np.isfinite(zs) and zs >= 3 and S_self != 0) else float("nan"),
                null_self_mean=float(N_self.mean()), null_self_sd=float(N_self.std()), null_other_mean=float(N_oth.mean()), null_other_sd=float(N_oth.std()),
                n_null=int(nl.shape[0]), s_per_object=s.cpu().tolist())


def spill_inputs(W, b):
    """The spillover inputs of block b from rgi/causal_<m>.pt's W [60, 4, NL, D]: w [48, D] = the IMP-DES gradient of each model object
    (mean of the two lexeme halves) and is_self [48] (owner SELF). For r2_ana's u_T_perp spillover: spillover(*spill_inputs(W, b),
    u_T_perp(b), R1 null_u_T_perp(b))."""
    W = torch.as_tensor(W); return 0.5 * (W[:48, 0, b].double() + W[:48, 1, b].double()), np.array([o in SELF_OBJ for o in range(48)])


def sign_flip_nulls(c_o, n=2000, seed=1107):
    """A6 null axes for g_b and s_b (rgi/dirs_<m>.pt): sign flips of the 48 model-object contrasts c_o [48, D] (torch.Generator seed),
    each null built like its axis: g' = mean(eps c_o), s' = mean over the SELF objects of eps c_o minus its projection on unit(g').
    Returns dict(g [n, D], s [n, D]) float64."""
    c = torch.as_tensor(c_o).double(); g = torch.Generator().manual_seed(int(seed)); E = (torch.randint(0, 2, (n, c.shape[0]), generator=g) * 2 - 1).double()
    gn = E @ c / c.shape[0]; sm = torch.zeros(c.shape[0], dtype=torch.float64); sm[SELF_OBJ] = 1.0 / len(SELF_OBJ); ds = (E * sm) @ c
    u = gn / gn.norm(dim=1, keepdim=True).clamp_min(1e-300); return dict(g=gn, s=ds - (ds * u).sum(1, keepdim=True) * u)


def pool_dirs(pool_b, n, seed, dev):
    """n unit directions with the pool covariance of the block: unit((z * sqrt(ev)) @ U^T), z ~ N(0, I) from torch's CPU generator."""
    g = torch.Generator().manual_seed(int(seed)); D = pool_b["ev"].shape[0]; z = torch.randn(n, D, generator=g, dtype=torch.float64)
    v = (z * pool_b["ev"].double().clamp_min(0).sqrt()) @ pool_b["U"].double().T; return (v / v.norm(dim=1, keepdim=True)).to(dev)


def topic_basis(Xs, labels):
    """Orthonormal basis [D, r] of the span of the centred topic means of Xs [N, D] (r = rank <= K - 1)."""
    K = int(labels.max()) + 1; lab = torch.as_tensor(labels, device=Xs.device)
    Mu = torch.stack([Xs[lab == k].mean(0) for k in range(K)]); Mu = Mu - Mu.mean(0); U, s, _ = torch.linalg.svd(Mu.T, full_matrices=False)
    r = int((s > s[0] * 1e-9).sum()) if float(s[0]) > 0 else 0; return U[:, :r]


def general_dirs(Upool, Lam, Psi, xs):
    """g(x, h, i) [R, len(xs), 2, 2, D] = mean over the 12 model cells of d(c, h, i) for contrasts xs (frames-pooled units Upool [NVF, NOBJ, D])."""
    Lx = Lam.reshape(Lam.shape[0], NX, 2, NVF)[:, list(xs)].to(Upool.device)                                # [R, x, h, v]
    beta = torch.zeros(Psi.shape[0], 2, NOBJ, dtype=torch.float64, device=Upool.device)
    for o in range(48): beta[:, :, o] = Psi[:, CELL_OF[o], :, PARA_OF[o]].to(Upool.device) / 12.0
    return torch.einsum("rxhv,rio,vod->rxhid", Lx, beta, Upool)


def tau_stats(Upool, Q, des_pt, rand_dirs, pool_b):
    """tau of IMP-DES and CHG-DES (point: averaged numerator and denominator over the 81 combinations and both disjoint pairings) and the
    random expectations for the pool-covariance directions."""
    Lam, Psi, _ = weights(des_pt); g = general_dirs(Upool, Lam, Psi, (0, 1)); out = {}
    for j, nm in enumerate(("IMP-DES", "CHG-DES")):
        a, b = g[:, j], g[:, j]; num = den = 0.0
        for (h, i), (h2, i2) in (((0, 0), (1, 1)), ((0, 1), (1, 0))):
            num = num + ((a[:, h, i] @ Q) * (b[:, h2, i2] @ Q)).sum(-1); den = den + (a[:, h, i] * b[:, h2, i2]).sum(-1)
        out[f"tau|{nm}"] = float(num.mean() / den.mean()) if float(den.mean()) != 0 else float("nan")
    pv = rand_dirs @ Q; out["tau_rand_mc"] = float(((pv ** 2).sum(1) / (rand_dirs ** 2).sum(1)).mean())
    UQ = pool_b["U"].double().to(Q.device).T @ Q; ev = pool_b["ev"].double().to(Q.device); out["tau_rand_trace"] = float((ev[:, None] * UQ ** 2).sum() / ev.sum())
    out["topic_like"] = bool(out["tau|IMP-DES"] > 3 * out["tau_rand_mc"] and out["tau|IMP-DES"] > out["tau|CHG-DES"])
    out["topic_dims"] = int(Q.shape[1]); return out


def anisotropy(X):
    """Mean pairwise cosine of the rows of X [n, D] (uncentred)."""
    u = X / X.norm(dim=1, keepdim=True).clamp_min(1e-300); s = u.sum(0); n = X.shape[0]; return float(((s @ s) - n) / (n * (n - 1)))


# ======================================================================================================================================
# The run
# ======================================================================================================================================
def paths(r2_dir, m):
    P = GEOM.out_paths(r2_dir, m)
    P.update(stats=os.path.join(r2_dir, "rgi", f"geom_stats_{m}.json"), draws=os.path.join(r2_dir, "rgi", f"geom_draws_{m}.npz"),
             cache=os.path.join(r2_dir, "local", f"geom_stats_cache_{m}"))
    return P


DRAW_VARIANTS = ("pooled", "F1", "F2")


def block_stats(b, A, ctx):
    """Every per-block quantity of block b. A: the arrays (dot, obj, lex, W, last, mean), ctx: layout, designs, pools, labels."""
    dev = ctx["dev"]; unit, neu = ctx["unit"], ctx["neu"]; out = dict(point={}, draws={}, extra={})
    pool_b = ctx["pool"][b]; ev = pool_b["ev"].double().to(dev); Ub = pool_b["U"].double().to(dev); scale = (ev.clamp_min(0) + ev.clamp_min(0).mean()).rsqrt()
    rd = pool_dirs(pool_b, N_RAND, SEEDS["randdirs"] + b, dev)
    for reading in ("dot", "obj"):
        X = torch.as_tensor(np.asarray(A[reading][:, b, :]), dtype=torch.float64, device=dev); U = X[torch.as_tensor(unit, device=dev)]   # [2, NVF, NOBJ, D]
        des = [VF_OFF["DES"] + l for l in range(NLEX["DES"])]; out["extra"][f"anisotropy_{reading}"] = anisotropy(((U[0, des] + U[1, des]) / 2).reshape(-1, X.shape[1]))
        if reading == "dot":
            rng = np.random.default_rng([SEEDS["delta_obj"], b]); dob = {}
            for f in (0, 1): dob[f"F{f + 1}"] = delta_obj(X[torch.as_tensor(neu[f], device=dev)], rng)
            out["extra"]["delta_obj"] = dob
        U = U - U.reshape(2, -1, U.shape[-1]).mean(1)[:, None, None, :]
        for metric in ("raw", "white"):
            Um = U if metric == "raw" else (U @ Ub) * scale
            for frame in DRAW_VARIANTS:
                Uf = (Um[0] + Um[1]) / 2 if frame == "pooled" else Um[0 if frame == "F1" else 1]; K = gram_units(Uf)
                Pp, Prel = pp_point(K, ctx["LL"], ctx["PP"]); key = f"{reading}/{metric}/{frame}"
                out["point"][key] = {k: float(v[0]) for k, v in geo_stats(Pp, Prel).items()}
                if reading == "dot" and metric == "raw":
                    if frame == "pooled":
                        out["extra"]["per_cell_P_cell"] = Pp[0, range(NX), range(NX)].diagonal(dim1=-2, dim2=-1).tolist()
                        out["extra"]["naive"] = naive_stats(K); out["extra"]["loo"] = {}
                        for kind, idx in (("owner", OWN), ("substrate", SUB)):
                            for j, nm in enumerate(OWNERS if kind == "owner" else SUBS):
                                keep = ~(MODEL_CELL & (idx == j)); st = geo_stats(Pp, Prel, keep=keep, self_stats=False)
                                out["extra"]["loo"][f"{kind}:{nm}"] = {k: float(v[0]) for k, v in st.items()}
                    Pd, Rd = pp_draws(K, ctx["Lam"], ctx["Psi"], ctx["chunk"])
                    out["draws"][frame] = {k: v.numpy().astype(np.float64) for k, v in geo_stats(Pd, Rd).items()}
                if metric == "raw" and frame == "pooled":
                    o7 = torch.as_tensor(np.asarray(A["last" if reading == "dot" else "mean"][:, b, :]), dtype=torch.float64, device=dev)
                    out["extra"][f"tau_{reading}"] = tau_stats(Uf, topic_basis(o7, ctx["labels"]), ctx["des_pt"], rd, pool_b)
    Wb = torch.as_tensor(np.asarray(A["W"][:, :, b, :]), dtype=torch.float64, device=dev); Ge = causal_ge(Wb); Pp, Prel = causal_point(Ge, ctx["PP"])
    out["point"]["causal"] = {k: float(v[0]) for k, v in geo_stats(Pp, Prel, names=CAUSAL_CN, cross=()).items()}
    Pd, Rd = causal_draws(Ge, ctx["Psi"], ctx["chunk"])
    out["draws"]["causal"] = {k: v.numpy().astype(np.float64) for k, v in geo_stats(Pd, Rd, names=CAUSAL_CN, cross=()).items()}
    w = 0.5 * (Wb[:48, 0] + Wb[:48, 1]); is_self = np.array([o in SELF_OBJ for o in range(48)])
    out["extra"]["spill_g_self"] = spillover(w, is_self, torch.as_tensor(ctx["g_self"][b], dtype=torch.float64, device=dev), rd)
    return out


def causal_point(Ge, PP):
    """Point Pp, Prel of the causal units: average over the 3 paraphrase splits (exact, via the paraphrase operator PP)."""
    G = torch.einsum("cikdjm,xhckygdm->xhciygdj", PP.to(Ge.device), Ge)
    Pp = (G[:, 0, :, 0, :, 1, :, 1] + G[:, 0, :, 1, :, 1, :, 0] + G[:, 1, :, 0, :, 0, :, 1] + G[:, 1, :, 1, :, 0, :, 0]) / 4
    Dg = torch.diagonal(torch.diagonal(G, dim1=0, dim2=4), dim1=1, dim2=4); Prel = (Dg[0, 0, 0, 1] + Dg[0, 1, 0, 0] + Dg[1, 0, 1, 1] + Dg[1, 1, 1, 0]) / 4
    return Pp.permute(0, 2, 1, 3)[None].cpu(), Prel[None].cpu()


def causal_draws(Ge, Psi, chunk):
    out_p, out_r = [], []
    for i in range(0, Psi.shape[0], chunk):
        Ps = Psi[i:i + chunk].to(Ge.device); Pp, Prel = para_step(Ge[None].expand(Ps.shape[0], *Ge.shape), Ps); out_p.append(Pp.cpu()); out_r.append(Prel.cpu())
    return torch.cat(out_p), torch.cat(out_r)


def inputs_record(P, gm):
    rec = {}
    for k in ("geom", "delta", "pool", "causal", "dirs", "meta"):
        rec[k] = dict(path=os.path.basename(P[k]), bytes=os.path.getsize(P[k]))
    for k in ("geom", "delta", "pool", "causal", "dirs"):
        want = (gm.get("outputs") or {}).get(k, {}).get("sha256"); got = GEOM.sha256_file(P[k]); rec[k]["sha256"] = got
        if want is not None and want != got: raise RuntimeError(f"{P[k]} differs from the file recorded in geom_meta (sha256): stale local output")
    return rec


def run(r2_dir, m="qwen3-1.7b", force=False):
    t0 = time.time(); P = paths(r2_dir, m); dev = device()
    if os.path.exists(P["stats"]) and not force:
        log(f"{P['stats']} exists; skipped (use --force to recompute)"); return json.load(open(P["stats"]))
    for k in ("geom", "delta", "pool", "causal", "dirs", "meta"):
        if not os.path.exists(P[k]): raise FileNotFoundError(f"{P[k]} missing: run r2_rgi_geom.py first")
    gm = json.load(open(P["meta"])); inp = inputs_record(P, gm)
    z = np.load(P["geom"], allow_pickle=False); dz = np.load(P["delta"], allow_pickle=False); pool = GEOM.load_pt(P["pool"])["blocks"]
    cz = GEOM.load_pt(P["causal"]); dr = GEOM.load_pt(P["dirs"])
    unit, neu = sentence_layout(z["s_obj"], z["s_frame"], z["s_family"], z["s_lexeme"])
    A = dict(dot=z["dot"], obj=z["obj"], lex=z["lex"], last=dz["last"], mean=dz["mean"], W=cz["W"].numpy())   # each array read once (NpzFile is lazy)
    NL, D = A["dot"].shape[1], A["dot"].shape[2]
    labels = dz["labels"].astype(np.int64); des_d = draw_design(N_DRAWS); des_pt = point_design(); Lam, Psi, _ = weights(des_d); LL, PP = point_operators(des_pt)
    chunk = 64 if dev.type == "cuda" else 8
    ctx = dict(dev=dev, unit=unit, neu=neu, pool=pool, Lam=Lam, Psi=Psi, LL=LL, PP=PP, des_pt=des_pt, chunk=chunk, labels=labels, g_self=dr["g_self_f2"].double())
    fp = dict(version=VERSION, model=m, mock=MOCK, n_draws=N_DRAWS, n_dboot=N_DBOOT, seeds=SEEDS, inputs={k: v.get("sha256") for k, v in inp.items()})
    BS = []; os.makedirs(P["cache"], exist_ok=True); n_cached = 0
    for b in range(NL):                                                    # resumable: one cache file per block, keyed by the fingerprint
        cf = os.path.join(P["cache"], f"b{b}.pt"); c = torch.load(cf, map_location="cpu", weights_only=False) if os.path.exists(cf) else None
        if c is not None and c.get("fingerprint") == fp: BS.append(c["stats"]); n_cached += 1; continue
        tb = time.time(); st = block_stats(b, A, ctx); GEOM.torch_save(cf, dict(fingerprint=fp, stats=st)); BS.append(st)
        log(f"block {b}: done ({time.time() - tb:.1f}s)")
    if n_cached: log(f"{n_cached} blocks taken from the cache {P['cache']}")
    # lexical reference (block-free; point estimates)
    lex = {}
    X = torch.as_tensor(A["lex"], dtype=torch.float64, device=dev); U = X[torch.as_tensor(unit, device=dev)]; U = U - U.reshape(2, -1, D).mean(1)[:, None, None, :]
    for frame in DRAW_VARIANTS:
        Uf = (U[0] + U[1]) / 2 if frame == "pooled" else U[0 if frame == "F1" else 1]; Pp, Prel = pp_point(gram_units(Uf), LL, PP)
        lex[frame] = {k: float(v[0]) for k, v in geo_stats(Pp, Prel).items()}
    # delta (A4)
    o7 = gm["stages"]["o7"]["delta"]; d_mmlu = {nm: np.array(o7[nm]["delta"]) for nm in ("last", "mean")}
    idx, perm = boot_index(labels, N_DBOOT, SEEDS["delta_boot"]); boot = {nm: np.zeros((N_DBOOT, NL)) for nm in ("last", "mean")}
    for b in range(NL):
        for nm in ("last", "mean"):
            boot[nm][:, b] = delta_boot(torch.as_tensor(np.asarray(A[nm][:, b, :]), dtype=torch.float64, device=dev), labels, idx, perm)
    boot_sd = {nm: boot[nm].std(0, ddof=1) for nm in boot}
    dobj = {f: np.array([BS[b]["extra"]["delta_obj"][f][0] - BS[b]["extra"]["delta_obj"][f][1] for b in range(NL)]) for f in ("F1", "F2")}
    d_obj = (dobj["F1"] + dobj["F2"]) / 2; aniso = {r: np.array([BS[b]["extra"][f"anisotropy_{r}"] for b in range(NL)]) for r in ("dot", "obj")}
    # series, intervals, gates
    series = {v: {k: np.array([BS[b]["point"][f"dot/raw/{v}"][k] for b in range(NL)]) for k in BS[0]["point"]["dot/raw/pooled"]} for v in DRAW_VARIANTS}
    series["causal"] = {k: np.array([BS[b]["point"]["causal"][k] for b in range(NL)]) for k in BS[0]["point"]["causal"]}
    draws = {v: {k: np.stack([BS[b]["draws"][v][k] for b in range(NL)], 1) for k in BS[0]["draws"][v]} for v in DRAW_VARIANTS + ("causal",)}
    pw = {v: {k: pointwise(draws[v][k]) for k in draws[v]} for v in draws}
    C0_parts = {}
    for x in ("IMP-DES", "CHG-DES"):
        C0_parts[f"P_cell_lo>0|{x}"] = pw["pooled"][f"P_cell|{x}"][0] > 0; C0_parts[f"q>=0.2|{x}"] = series["pooled"][f"q|{x}"] >= 0.2
    C0 = np.logical_and.reduce(list(C0_parts.values())); uncomp = (series["pooled"]["FG|CHG-DES"] > 0.8) & (series["pooled"]["FG|WOR-DES"] > 0.8)
    passing = C0 & ~uncomp; Pset, pinfo = passing_set(passing, d_mmlu["last"]); sel = np.zeros(NL, bool); sel[Pset] = True
    composed = series["causal"]["FG|CHG-DES"] < 0.8
    per_block = {}
    for v in draws:
        per_block[v] = {}
        for k in draws[v]:
            lo, hi = pw[v][k]; slo, shi, c = supt_band(series[v][k], draws[v][k], sel)
            fin = np.isfinite(draws[v][k]).sum(0)
            with np.errstate(all="ignore"): sd = np.array([np.nanstd(draws[v][k][:, j], ddof=1) if fin[j] >= 2 else np.nan for j in range(NL)])
            per_block[v][k] = dict(point=series[v][k], lo=lo, hi=hi, sd=sd, sim_lo=slo, sim_hi=shi, sim_crit=c, n_nan_draws=np.isnan(draws[v][k]).sum(0))
    point_only = {}
    for key in BS[0]["point"]:
        if key.startswith("dot/raw/") or key == "causal": continue
        point_only[key] = {k: np.array([BS[b]["point"][key][k] for b in range(NL)]) for k in BS[0]["point"][key]}
    loo = {nm: {k: np.array([BS[b]["extra"]["loo"][nm][k] for b in range(NL)]) for k in BS[0]["extra"]["loo"][nm]} for nm in BS[0]["extra"]["loo"]}
    naive = {k: np.array([BS[b]["extra"]["naive"][k] for b in range(NL)]) for k in BS[0]["extra"]["naive"]}
    per_cell = {CN[x]: [BS[b]["extra"]["per_cell_P_cell"][x] for b in range(NL)] for x in range(NX)}
    spill = {k: [BS[b]["extra"]["spill_g_self"][k] for b in range(NL)] for k in BS[0]["extra"]["spill_g_self"] if k != "s_per_object"}
    spill["s_per_object"] = [BS[b]["extra"]["spill_g_self"]["s_per_object"] for b in range(NL)]
    tau = {r: {k: [BS[b]["extra"][f"tau_{r}"][k] for b in range(NL)] for k in BS[0]["extra"][f"tau_{r}"]} for r in ("dot", "obj")}
    # A5 over P, over the composed blocks of P, object reading
    Pb = np.array(Pset, dtype=np.int64); Pc = np.array([b for b in Pset if composed[b]], dtype=np.int64); over = {}
    packs = {}
    if len(Pb) >= 6:
        packs["P"] = surrogate_pack(d_mmlu["last"][Pb], Pb, aniso["dot"][Pb], SEEDS["surrogates"])
        packs["P_mean"] = surrogate_pack(d_mmlu["mean"][Pb], Pb, aniso["obj"][Pb], SEEDS["surrogates"] + 1)
    if len(Pc) >= 6: packs["Pc"] = surrogate_pack(d_mmlu["last"][Pc], Pc, aniso["dot"][Pc], SEEDS["surrogates"] + 2)
    white = point_only.get("dot/white/pooled", {})
    spec = [("dFG", "pooled", "dFG"), ("FG|IMP-DES", "pooled", "FG|IMP-DES")] + [(f"{s}|IMP-DES@{v}", v, f"{s}|IMP-DES") for v in DRAW_VARIANTS for s in ("rho_S", "sigma_self")] \
        + [("FG_causal|IMP-DES", "causal", "FG|IMP-DES"), ("rho_causal|IMP-DES", "causal", "rho_S|IMP-DES"), ("dFG_causal", "causal", "dFG")]
    for setname, blocks in (("P", Pb), ("P_composed", Pc)):
        over[setname] = {}
        for nm, v, k in spec:
            if setname == "P_composed" and v != "causal": continue
            if len(blocks) == 0: over[setname][nm] = dict(status="empty block set"); continue
            yw = white[k][blocks] if (v == "pooled" and k in white) else None
            over[setname][nm] = over_blocks(series[v][k][blocks], draws[v][k][:, blocks], blocks, d_mmlu["last"][blocks], d_obj[blocks],
                                            packs.get("P" if setname == "P" else "Pc"), y_white=yw)
    if len(Pb):
        over["object_reading"] = {"dFG": over_blocks(point_only["obj/raw/pooled"]["dFG"][Pb], None, Pb, d_mmlu["mean"][Pb], d_obj[Pb], packs.get("P_mean"),
                                                     y_white=point_only["obj/white/pooled"]["dFG"][Pb])}
        if "P" in packs:
            r3o = corr_rows(cubic_resid(series["pooled"]["dFG"][Pb], Pb), cubic_resid(d_obj[Pb], Pb)); S = cubic_resid(surrogates(cubic_resid(d_obj[Pb], Pb), N_SURR, SEEDS["surrogates"] + 3), Pb)
            over["P"]["r3_delta_obj_dFG"] = dict(r3=float(r3o), p_pos=pvals(float(r3o), corr_rows(S, cubic_resid(series["pooled"]["dFG"][Pb], Pb)))[0])
    sep = dict(last=separability(d_mmlu["last"], boot_sd["last"], Pb), mean=separability(d_mmlu["mean"], boot_sd["mean"], Pb),
               last_all_blocks=separability(d_mmlu["last"], boot_sd["last"], np.arange(NL)))
    # r_DiD
    did = dict(status="the other model's statistics are not available")
    other = "qwen3-0.6b" if m == "qwen3-1.7b" else "qwen3-1.7b"; Po = paths(r2_dir, other)
    if os.path.exists(Po["stats"]):
        so = json.load(open(Po["stats"])); dro = np.load(Po["draws"], allow_pickle=False) if os.path.exists(Po["draws"]) else None
        mine = dict(dfg=series["pooled"]["dFG"], delta=d_mmlu["last"], P=Pset, draws=draws["pooled"]["dFG"])
        theirs = dict(dfg=np.array(so["per_block"]["pooled"]["dFG"]["point"], dtype=np.float64), delta=np.array(so["delta"]["mmlu"]["last"]["delta"]),
                      P=so["gate"]["P"], draws=dro["pooled__dFG"] if dro is not None and "pooled__dFG" in dro.files else None)
        a, bb = (mine, theirs) if m == "qwen3-1.7b" else (theirs, mine)
        did = r_did(a["dfg"], a["delta"], a["P"], bb["dfg"], bb["delta"], bb["P"], a["draws"], bb["draws"]) if (a["draws"] is not None and bb["draws"] is not None and
              a["draws"].shape == bb["draws"].shape) else r_did(a["dfg"], a["delta"], a["P"], bb["dfg"], bb["delta"], bb["P"])
        did["definition"] = "corr over P_1.7B & P_0.6B of dFG_1.7B - dFG_0.6B with delta_MMLU(last)_1.7B - delta_MMLU(last)_0.6B"
    flags = []
    if not pinfo["ok"]: flags.append("G30: fewer than 12 passing blocks in P ('undecided (fewer than 12 passing blocks)')")
    if not sep["last"].get("ok"): flags.append("G31: separability fails (V1 impossible, V2 at best)")
    flags += list(gm.get("flags") or [])
    res = dict(version=VERSION, model=m, mock=MOCK, plan_sha256=PLAN_SHA256, written=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               device=str(dev), versions=GEOM.versions(), inputs=inp, flags=flags,
               settings=dict(n_draws=N_DRAWS, n_delta_boot=N_DBOOT, n_surrogates=N_SURR, n_perm=N_PERM, n_rand_dirs=N_RAND, level=LEVEL, seeds=SEEDS,
                             point="average of the P's over the 81 joint split combinations (each contrast's 27, or 9 with WOR / PRA; OD-A1), then the ratios",
                             pointwise="percentile intervals (2.5 / 97.5) of the draws", simultaneous="sup-t over P: point +- c sd, c = 95 % quantile of max |draw - mean| / sd",
                             chunk=chunk),
               design=dict(contrasts=CN, cells=CELL_NAMES, families=FAMS, lexemes={F: list(STIM.VERBS[F][0]) for F in FAMS}, lex_splits=LEX_SPLITS,
                           causal_contrasts=CAUSAL_CN, causal_halves=GEOM.A3_HALVES, objects=[o["text"]["F1"] for o in GEOM.OBJECTS]),
               blocks=list(range(NL)),
               gate=dict(C0=C0, C0_parts=C0_parts, uncomposed=uncomp, passing=passing, P=Pset, n_P=len(Pset), G30_ok=pinfo["ok"], runs=pinfo["runs"], tie=pinfo["tie"],
                         composed=composed, P_composed=[int(b) for b in Pc],
                         label=None if pinfo["ok"] else "undecided (fewer than 12 passing blocks)"),
               per_block=per_block, point_only=point_only, lexical=lex, loo=loo, naive=naive, per_cell_P_cell=per_cell, spillover_g_self=spill,
               delta=dict(mmlu={nm: dict(**{k: o7[nm][k] for k in ("F", "floor", "delta", "delta_rel")}, boot_sd=boot_sd[nm], boot_lo=np.percentile(boot[nm], 2.5, 0),
                                         boot_hi=np.percentile(boot[nm], 97.5, 0)) for nm in ("last", "mean")},
                          obj=dict(F1=dict(F=[BS[b]["extra"]["delta_obj"]["F1"][0] for b in range(NL)], floor=[BS[b]["extra"]["delta_obj"]["F1"][1] for b in range(NL)], delta=dobj["F1"]),
                                   F2=dict(F=[BS[b]["extra"]["delta_obj"]["F2"][0] for b in range(NL)], floor=[BS[b]["extra"]["delta_obj"]["F2"][1] for b in range(NL)], delta=dobj["F2"]),
                                   mean=d_obj),
                          separability=sep, tau=tau),
               anisotropy=aniso, over_blocks=over, r_DiD=did, seconds=time.time() - t0)
    arrays = {f"{v}__{k}": draws[v][k].astype(np.float32) for v in draws for k in draws[v]}
    arrays.update(design_lex_w=des_d["lex_w"].astype(np.float32), design_lex_split=des_d["lex_split"].astype(np.int8), design_para_w=des_d["para_w"].astype(np.float32),
                  design_para_split=des_d["para_split"].astype(np.int8), delta_boot_last=boot["last"].astype(np.float32), delta_boot_mean=boot["mean"].astype(np.float32))
    GEOM.npz_save(P["draws"], dict(version=VERSION, model=m, n_draws=N_DRAWS, seeds=SEEDS, note="per-draw values [n_draws, n_blocks] '<variant>__<stat>'"), **arrays)
    GEOM.write_json(P["stats"], res); log(f"statistics -> {P['stats']} ({time.time() - t0:.1f}s); P = {Pset} ({len(Pset)} blocks)")
    return res


# ======================================================================================================================================
# Self-test (CPU, MOCK=1): reference implementation, planted structure, calibration, end-to-end runs
# ======================================================================================================================================
def _ref_stats(U, des, r_list):
    """Explicit reference (the RGI design's sim_tg.tg, extended to owner x substrate): for each draw / combination r, the cell
    contrasts d(c, h, i) are built one by one and every P is the mean of the Gram entries over the row pairs of its class; the P's are
    averaged over r_list, then the ratios. U [NVF, NOBJ, D] numpy (frames pooled)."""
    Lam, Psi, _ = (t.numpy() for t in weights(des)); acc = {}
    rows = [(x, c, h, i) for x in range(NX) for c in range(NCELL) for h in range(2) for i in range(2)]
    for r in r_list:
        V = np.zeros((len(rows), U.shape[-1]))
        for j, (x, c, h, i) in enumerate(rows):
            e = np.einsum("v,vod->od", Lam[r, 2 * x + h], U); V[j] = sum(Psi[r, c, i, k] * e[c * 4 + k] for k in range(4))
        G = V @ V.T; X_, C_, H_, I_ = (np.array([rw[q] for rw in rows]) for q in range(4))
        sameX = X_[:, None] == X_[None]; o, s_ = OWN[C_], SUB[C_]; mod = MODEL_CELL[C_]; hd = H_[:, None] != H_[None]; idf = I_[:, None] != I_[None]
        so, ss = o[:, None] == o[None], s_[:, None] == s_[None]; mm = mod[:, None] & mod[None]; sc = C_[:, None] == C_[None]
        def P(mask, x=None, y=None):
            mk = mask & (X_[:, None] == x) & (X_[None] == (x if y is None else y)); return G[mk].mean()
        for x in range(NX):
            vals = dict(cell=P(sc & mm & hd & idf, x), own=P(mm & ss & ~so & hd & idf, x), sub=P(mm & so & ~ss & hd & idf, x), gen=P(mm & ~so & ~ss & hd & idf, x),
                        rel=P(sc & mm & ~hd & idf, x), cell_all=P(sc & hd & idf, x), gen_all=P(~so & ~ss & hd & idf, x),
                        cell_LR=P(sc & mm & hd & idf & (s_[:, None] == 2), x), own_LR=P(mm & ss & ~so & hd & idf & (s_[:, None] == 2), x))
            for a in range(4):
                for b in range(4):
                    for sb in range(3):
                        c1, c2 = 3 * a + sb, 3 * b + sb; vals[f"pair{c1}_{c2}"] = P((C_[:, None] == c1) & (C_[None] == c2) & hd & idf, x)
            for k, v in vals.items(): acc.setdefault((x, k), []).append(v)
        for nm, x, y in CROSS: acc.setdefault((nm, "gen"), []).append(P(mm & ~so & ~ss & hd & idf, x, y))
    A = {k: float(np.mean(v)) for k, v in acc.items()}; out = {}
    for x in range(NX):
        g = lambda k: A[(x, k)]; out[f"FG|{CN[x]}"] = g("gen") / g("cell"); out[f"OG|{CN[x]}"] = g("own") / g("cell"); out[f"SG|{CN[x]}"] = g("sub") / g("cell")
        out[f"q|{CN[x]}"] = g("cell") / g("rel"); out[f"FG_all|{CN[x]}"] = g("gen_all") / g("cell_all"); out[f"OG_LR|{CN[x]}"] = g("own_LR") / g("cell_LR")
        out[f"P_cell|{CN[x]}"] = g("cell")
        for tag, subs in (("", (0, 1, 2)), ("_LR", (2,))):
            rho = lambda c1, c2: g(f"pair{c1}_{c2}") / math.sqrt(g(f"pair{c1}_{c1}") * g(f"pair{c2}_{c2}")) if g(f"pair{c1}_{c1}") > 0 and g(f"pair{c2}_{c2}") > 0 else float("nan")
            with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore", RuntimeWarning)
                rs = np.nanmean([rho(s_, 3 * b + s_) for b in (1, 2, 3) for s_ in subs]); ro = np.nanmean([rho(3 * a + s_, 3 * b + s_) for a, b in ((1, 2), (1, 3), (2, 3)) for s_ in subs])
            out[f"rho_S{tag}|{CN[x]}"], out[f"sigma_self{tag}|{CN[x]}"] = rs, ro - rs
    for nm, x, y in CROSS:
        out[nm] = A[(nm, "gen")] / math.sqrt(A[(x, "gen")] * A[(y, "gen")]) if A[(x, "gen")] > 0 and A[(y, "gen")] > 0 else float("nan")
    return out


def _synth_units(rng, D, fg=dict(IMP=0.6, CHG=0.3, WOR=0.5, PRA=0.4), self_extra=0.0, signal=1.0, lex=0.5, item=0.5, noise=1.0):
    """Planted units [2, NVF, NOBJ, D] and the contrasts' population cell vectors delta[F] [NCELL, D] (DES = 0): x = base(o) +
    delta_F(cell) + lexeme effect + lexeme x cell + family x object (item) effect + sentence noise (frame-specific)."""
    r = lambda e, *sh: rng.standard_normal(sh + (D,)) * math.sqrt(e / D)
    delta = {"DES": np.zeros((NCELL, D))}
    for F, f in fg.items():
        g = r(f * signal); a = r((1 - f) / 3 * signal, 4); b = r((1 - f) / 3 * signal, 3); cc = r((1 - f) / 3 * signal, 4, 3); nm = r((1 - f) * signal, 3); dF = np.zeros((NCELL, D))
        for c in range(12): dF[c] = g + a[c // 3] + b[c % 3] + cc[c // 3, c % 3]
        for c in range(12, NCELL): dF[c] = g + nm[c - 12]
        if F == "IMP" and self_extra:
            for s_ in range(3): dF[s_] += r(self_extra * signal)
        delta[F] = dF
    base = r(4.0, NOBJ); U = np.zeros((2, NVF, NOBJ, D))
    for F in FAMS:
        lam = r(lex, NLEX[F]); eta = r(0.2 * lex, NLEX[F], NCELL); zeta = r(item, NOBJ)
        for l in range(NLEX[F]):
            for f in range(2): U[f, VF_OFF[F] + l] = base + delta[F][CELL_OF] + lam[l] + eta[l][CELL_OF] + zeta + r(noise, NOBJ)
    return U, delta


def _truth(delta, A, B):
    """Population (infinite lexemes and paraphrases) values of the contrast A-B from the planted cell vectors."""
    d = delta[A] - delta[B]; G = d @ d.T; M = cell_masks(); Gt = torch.tensor(G)
    P = {n: float(_mm(Gt, M[n])) for n in ("cell", "own", "sub", "gen")}
    rho = lambda c1, c2: G[c1, c2] / math.sqrt(G[c1, c1] * G[c2, c2])
    rs = np.mean([rho(s_, 3 * b + s_) for b in (1, 2, 3) for s_ in range(3)]); ro = np.mean([rho(3 * a + s_, 3 * b + s_) for a, b in ((1, 2), (1, 3), (2, 3)) for s_ in range(3)])
    return dict(FG=P["gen"] / P["cell"], OG=P["own"] / P["cell"], SG=P["sub"] / P["cell"], rho_S=rs, sigma_self=ro - rs, d=d)


def _synth_topics(D, NL, seed, n_topic=16):
    """O7-like topic states with a planted, non-cubic separation per block, and their dn_delta values."""
    rng = np.random.default_rng(seed); labels = np.repeat(np.arange(8), n_topic); sep = 0.15 + 0.25 * (0.5 + 0.5 * np.sin(np.arange(NL) / 2.5 + seed)) + 0.004 * np.arange(NL)
    LAST = np.zeros((len(labels), NL, D)); MEAN = np.zeros_like(LAST)
    for b in range(NL):
        mu = rng.standard_normal((8, D)) * math.sqrt(sep[b] / D) * 3; LAST[:, b] = mu[labels] + rng.standard_normal((len(labels), D)) / math.sqrt(D) * 3
        MEAN[:, b] = LAST[:, b] + rng.standard_normal((len(labels), D)) / math.sqrt(D)
    return dict(LAST=LAST, MEAN=MEAN, labels=labels, dl=GEOM.o7_delta(LAST, MEAN, labels, torch.device("cpu")))


def _synth_world(path_r2, m, D, NL, seed, fg_of_delta, topics):
    """A complete set of r2_rgi_geom.py outputs with planted structure: blocks 0-2 carry no contrast (C0 must fail), blocks 3.. carry
    FG(IMP-DES) = fg_of_delta(delta_b); topic states (_synth_topics); random pool covariances; planted causal gradients."""
    rng = np.random.default_rng(seed); P = paths(path_r2, m); S = STIM.sentences()
    rows = [(s["obj"], 0 if s["frame"] == "F1" else 1, GEOM.FAM_CODE[s["family"]], -1 if s["lexeme"] is None else s["lexeme"]) for s in S]
    s_obj, s_frame, s_family, s_lex = (np.array([r[q] for r in rows]) for q in range(4)); unit, neu = sentence_layout(s_obj, s_frame, s_family, s_lex)
    LAST, MEAN, labels, dl = topics["LAST"], topics["MEAN"], topics["labels"], topics["dl"]; dlast = dl["last"]["delta"]
    dot = np.zeros((len(rows), NL, D), np.float32); obj = np.zeros_like(dot); W = np.zeros((NOBJ, 4, NL, D), np.float32); truth = []
    for b in range(NL):
        sig = 0.0 if b < 3 else 1.0; f_imp = float(fg_of_delta(dlast[b]))
        U, delta = _synth_units(rng, D, fg=dict(IMP=f_imp, CHG=0.3, WOR=0.5, PRA=0.4), signal=sig, self_extra=0.3)
        truth.append(dict(FG_IMP=_truth(delta, "IMP", "DES")["FG"] if sig else float("nan"), FG_CHG=_truth(delta, "CHG", "DES")["FG"] if sig else float("nan")))
        for f in range(2):
            for v in range(NVF): dot[unit[f, v], b] = U[f, v]
            dot[neu[f], b] = rng.standard_normal((NCELL, D))[CELL_OF] + 0.5 * rng.standard_normal((NOBJ, D))
        obj[:, b] = dot[:, b] + 0.3 * rng.standard_normal((len(rows), D)) / math.sqrt(D)
        for x, F in ((0, "IMP"), (1, "CHG")):
            for h in range(2): W[:, 2 * x + h, b] = delta[F][CELL_OF] + 0.3 * rng.standard_normal((NOBJ, D)) / math.sqrt(D)
    lexr = rng.standard_normal((len(rows), D)).astype(np.float32)
    os.makedirs(os.path.dirname(P["geom"]), exist_ok=True); os.makedirs(os.path.dirname(P["causal"]), exist_ok=True)
    fp = dict(version=GEOM.VERSION, model=m, mock=True, n_blocks=NL, hidden_size=D, path="synthetic")
    GEOM.npz_save(P["geom"], dict(fingerprint=fp, summary={}), dot=dot, obj=obj, lex=lexr, s_obj=s_obj.astype(np.int16), s_frame=s_frame.astype(np.int8),
                  s_family=s_family.astype(np.int8), s_lexeme=s_lex.astype(np.int8))
    GEOM.npz_save(P["delta"], dict(fingerprint=fp, summary={}), last=LAST.astype(np.float16), mean=MEAN.astype(np.float16), labels=labels.astype(np.int16))
    blocks = {}
    for b in range(NL):
        Q, _ = np.linalg.qr(rng.standard_normal((D, D))); ev = np.sort(rng.gamma(2.0, 1.0, D))[::-1] * np.exp(-np.arange(D) / 20)
        blocks[b] = dict(mu=torch.zeros(D, dtype=torch.float64), ev=torch.tensor(ev.copy()), U=torch.tensor(Q, dtype=torch.float32))
    GEOM.torch_save(P["pool"], dict(blocks=blocks, fingerprint=fp)); GEOM.torch_save(P["causal"], dict(W=torch.tensor(W), fingerprint=fp))
    dr = GEOM.directions(dot, unit); GEOM.torch_save(P["dirs"], dict(**{k: v.float() for k, v in dr.items()}, fingerprint=fp))
    outs = {k: dict(sha256=GEOM.sha256_file(P[k])) for k in ("geom", "delta", "pool", "causal", "dirs")}
    GEOM.write_json(P["meta"], dict(fingerprint=fp, stages=dict(o7=dict(delta={nm: {k: v.tolist() for k, v in r.items()} for nm, r in dl.items()})), outputs=outs, flags=[]))
    return dict(truth=truth, delta=dlast, P=P)


def selftest(root, r2_dir):
    assert MOCK, "selftest needs MOCK=1"
    global N_DRAWS, N_DBOOT
    torch.set_num_threads(int(os.environ.get("NT", "1"))); res = []; t0 = time.time(); work = os.path.join(r2_dir, "stats_selftest")
    shutil.rmtree(work, ignore_errors=True); os.makedirs(work)                             # the test's own folder, from scratch

    def check(name, ok, info=""):
        res.append((name, bool(ok))); print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    rng = np.random.default_rng(0)
    # 1. vectorised pipeline = explicit reference (point: 81 combinations; draws: three random draws)
    U2 = rng.standard_normal((2, NVF, NOBJ, 24)); Up = (U2[0] + U2[1]) / 2; K = gram_units(torch.tensor(Up)); dpt = point_design(); LL, PP = point_operators(dpt)
    st = geo_stats(*pp_point(K, LL, PP)); ref = _ref_stats(Up, dpt, range(81)); err = max(abs(st[k][0].item() - v) for k, v in ref.items())
    check("point estimate = explicit reference over the 81 split combinations", err < 1e-9, f"max abs diff {err:.1e} over {len(ref)} statistics")
    Lp, Ps, _ = weights(dpt); st81 = geo_stats(*(lambda a, b: (a.mean(0, keepdim=True), b.mean(0, keepdim=True)))(*pp_draws(K, Lp, Ps, 27)))
    err = max(abs(st81[k][0].item() - st[k][0].item()) for k in ref)
    check("factorised point operators = mean over the 81 explicit combinations", err < 1e-10, f"{err:.1e}")
    dd = draw_design(3, seed=5); Ld, Pd_, _ = weights(dd); sd = geo_stats(*pp_draws(K, Ld, Pd_, 2)); err = 0.0
    for r in range(3):
        rf = _ref_stats(Up, dd, [r]); err = max(err, max((abs(sd[k][r].item() - v) if np.isfinite(v) else (0.0 if not np.isfinite(sd[k][r].item()) else 1.0)) for k, v in rf.items()))
    check("per-draw values (Dirichlet weights, random splits) = explicit reference", err < 1e-9, f"{err:.1e}")
    dA, dB = draw_design(50), draw_design(50)
    check("draw design deterministic and Dirichlet weights normalised", all(np.array_equal(dA[k], dB[k]) for k in ("lex_w", "lex_split", "para_w", "para_split"))
          and np.allclose(dA["lex_w"][:, :4].sum(1), 1) and set(np.unique(dA["lex_split"])) <= {0, 1, 2})
    # 2. planted structure recovered (FG, OG, SG, rho_S, sigma_self, cosines)
    U2, delta = _synth_units(np.random.default_rng(1), 1536, self_extra=0.6); K = gram_units(torch.tensor((U2[0] + U2[1]) / 2)); st = geo_stats(*pp_point(K, LL, PP)); worst = []
    for A, B in (("IMP", "DES"), ("CHG", "DES"), ("IMP", "CHG")):
        t = _truth(delta, A, B)
        for k in ("FG", "OG", "SG", "rho_S", "sigma_self"): worst.append((abs(st[f"{k}|{A}-{B}"][0].item() - t[k]), f"{k}|{A}-{B}", t[k]))
    tq = (lambda a, b: _mm(torch.tensor(a @ b.T), cell_masks()["gen"]).item())
    dq, dw = delta["IMP"] - delta["CHG"], delta["WOR"] - delta["CHG"]; cq = tq(dq, dw) / math.sqrt(tq(dq, dq) * tq(dw, dw)); worst.append((abs(st["cos_quality"][0].item() - cq), "cos_quality", cq))
    w = max(worst); check("planted FG, OG, SG, rho_S, sigma_self, cos recovered (D 1536)", w[0] < 0.06, f"worst {w[1]}: |est - truth| {w[0]:.3f} (truth {w[2]:.3f}); FG(IMP-DES) "
          f"{st['FG|IMP-DES'][0].item():.3f} vs {_truth(delta, 'IMP', 'DES')['FG']:.3f}, sigma_self {st['sigma_self|IMP-DES'][0].item():.3f} vs {_truth(delta, 'IMP', 'DES')['sigma_self']:.3f}")
    # a known TG / FG value recovered on average over replications (bias check)
    fgs = []
    for rep in range(6):
        U2, delta = _synth_units(np.random.default_rng(100 + rep), 512, fg=dict(IMP=0.6, CHG=0.3, WOR=0.5, PRA=0.4))
        fgs.append(geo_stats(*pp_point(gram_units(torch.tensor((U2[0] + U2[1]) / 2)), LL, PP))["FG|IMP-DES"][0].item() - _truth(delta, "IMP", "DES")["FG"])
    check("FG unbiased over replications (planted 0.6)", abs(np.mean(fgs)) < 0.03, f"mean error {np.mean(fgs):+.3f}, sd {np.std(fgs):.3f}")
    # 3. sup-t critical values
    z = np.random.default_rng(2).standard_normal((4000, 1)); _, _, c1 = supt_band(np.zeros(20), np.repeat(z, 20, 1), np.ones(20, bool))
    _, _, c2 = supt_band(np.zeros(20), np.random.default_rng(3).standard_normal((4000, 20)), np.ones(20, bool))
    check("sup-t: c = 1.96 for identical columns, about 3.0 for 20 independent ones", abs(c1 - 1.96) < 0.1 and 2.8 < c2 < 3.25, f"c {c1:.2f}, {c2:.2f}")
    # 4. surrogates: spectrum kept; calibration under a smooth null; power under a planted relation
    x = np.cumsum(np.random.default_rng(4).standard_normal(20)); Sg = surrogates(x, 400, 9)
    check("surrogates: the original phases return the series; random phases keep its energy and change it", np.allclose(surrogates(x, 2, 9, zero_phase=True), x[None], atol=1e-12)
          and abs(np.mean((Sg ** 2).sum(1)) / (x ** 2).sum() - 1) < 0.25 and np.abs(Sg - x).max() > 1e-3, f"energy ratio {np.mean((Sg ** 2).sum(1)) / (x ** 2).sum():.2f}")
    rej, rejH1, B = 0, 0, np.arange(4, 24); g = np.random.default_rng(6)
    def smooth(n): return np.convolve(g.standard_normal(n + 8), np.ones(5) / 5, "valid")[:n]
    for rep in range(200):
        dlt = 1 - np.exp(-B / 7) + 0.05 * smooth(20); y = 0.2 * (1 - np.exp(-B / 9)) + 0.05 * smooth(20)
        pk = surrogate_pack(dlt, B, smooth(20), 1000 + rep, n=500); r3 = float(corr_rows(cubic_resid(y, B), pk["xr"])); rej += pvals(r3, corr_rows(pk["S"], cubic_resid(y, B)))[0] <= 0.05
        y1 = 2.0 * dlt + 0.05 * smooth(20); r3 = float(corr_rows(cubic_resid(y1, B), pk["xr"])); rejH1 += pvals(r3, corr_rows(pk["S"], cubic_resid(y1, B)))[0] <= 0.05
    check("surrogate r3 test: size about 5 % under a smooth depth-only null, power under a planted relation", rej / 200 <= 0.10 and rejH1 / 200 >= 0.8, f"size {rej / 200:.3f}, power {rejH1 / 200:.2f}")
    # 5. Fisher ratio and delta bootstrap
    Xf = np.random.default_rng(7).standard_normal((40, 12)); yl = np.repeat(np.arange(4), 10); Y = torch.tensor(np.eye(4)[yl])
    fr = GEOM.fisher_np(Xf, yl); f1 = float(fisher_batch(torch.tensor(Xf), Y)); Xc = torch.tensor(Xf - Xf.mean(0)); f2 = float(fisher_gram(Xc @ Xc.T, Y))
    ix = np.random.default_rng(8).integers(0, 40, 40); f3 = float(fisher_gram((Xc @ Xc.T)[ix][:, ix], Y)); f3r = GEOM.fisher_np(Xf[ix], yl)
    check("Fisher ratio: batched and Gram forms = dn_delta's fisher (incl. duplicated rows)", max(abs(f1 - fr), abs(f2 - fr), abs(f3 - f3r)) < 1e-12, f"{fr:.6f}")
    Xb = np.random.default_rng(9).standard_normal((48, 10)); lb = np.repeat(np.arange(4), 12); bi, bp = boot_index(lb, 3, 11); db = delta_boot(torch.tensor(Xb), lb, bi, bp, chunk=2)
    dref = [GEOM.fisher_np(Xb[bi[r]], lb) - np.mean([GEOM.fisher_np(Xb[bi[r]], bp[r, p]) for p in range(N_PERM)]) for r in range(3)]
    Xg = np.random.default_rng(10).standard_normal((48, 80)); dg = delta_boot(torch.tensor(Xg), lb, bi, bp, chunk=2)
    dgr = [GEOM.fisher_np(Xg[bi[r]], lb) - np.mean([GEOM.fisher_np(Xg[bi[r]], bp[r, p]) for p in range(N_PERM)]) for r in range(3)]
    check("delta bootstrap draws (feature and Gram paths) = direct recomputation", np.allclose(db, dref, atol=1e-10) and np.allclose(dg, dgr, atol=1e-10))
    # 6. tau: a direction inside the topic span gives 1, a random one about r / D
    D6 = 64; labs = np.repeat(np.arange(8), 6); Xs = torch.tensor(np.random.default_rng(12).standard_normal((48, D6))); Q = topic_basis(Xs, labs)
    v_in = (Q @ torch.randn(Q.shape[1], generator=torch.Generator().manual_seed(1), dtype=torch.float64)); pool_b = dict(ev=torch.ones(D6, dtype=torch.float64), U=torch.eye(D6))
    rdirs = pool_dirs(pool_b, 2000, 3, torch.device("cpu"))
    check("tau: topic basis rank 7; inside-span share 1; isotropic expectation 7 / D", Q.shape[1] == 7 and abs(float((v_in @ Q) @ (v_in @ Q) / (v_in @ v_in)) - 1) < 1e-10
          and abs(float(((rdirs @ Q) ** 2).sum(1).mean()) - 7 / D6) < 0.01)
    # 7. spillover
    g7 = np.random.default_rng(13); u = g7.standard_normal(64); u /= np.linalg.norm(u); nulls = g7.standard_normal((200, 64)); is_self = np.arange(48) < 12
    w = 0.05 * g7.standard_normal((48, 64)); w[~is_self] += 0.3 * u; sp = spillover(w, is_self, u, nulls)
    w2 = 0.05 * g7.standard_normal((48, 64)) + 0.3 * u[None]; w2[~is_self] -= 0.15 * u; sp2 = spillover(w2, is_self, u, nulls)
    check("spillover: other-owner push detected, FR only with a SELF response", sp["z_other"] > 5 and abs(sp["z_self"]) < 3 and np.isnan(sp["FR"])
          and sp2["z_self"] > 5 and abs(sp2["FR"] - sp2["S_other"] / sp2["S_self"]) < 1e-12 and 0.3 < sp2["FR"] < 0.7, f"z_other {sp['z_other']:.1f}, FR {sp2['FR']:.2f}")
    Wt = torch.tensor(g7.standard_normal((60, 4, 3, 64))); wi, si = spill_inputs(Wt, 1)
    co = torch.tensor(g7.standard_normal((48, 64))); nl = sign_flip_nulls(co, n=5, seed=3)
    e0 = (torch.randint(0, 2, (5, 48), generator=torch.Generator().manual_seed(3)) * 2 - 1).double()[0]; c0 = e0[:, None] * co
    gA = c0.mean(0); dS = c0[SELF_OBJ].mean(0); sA = dS - (dS @ gA / (gA @ gA)) * gA
    check("helpers: spill_inputs (IMP-DES half mean, SELF flags); sign-flip nulls built like g_b and s_b", torch.allclose(wi, 0.5 * (Wt[:48, 0, 1] + Wt[:48, 1, 1]))
          and si.sum() == 12 and si[:12].all() and nl["g"].shape == (5, 64) and torch.allclose(nl["g"][0], gA) and torch.allclose(nl["s"][0], sA))
    # 8. r_DiD
    b8 = np.arange(28); da = 0.1 + 0.05 * np.sin(b8 / 3); db_ = 0.1 + 0.05 * np.cos(b8 / 4); fa = 0.2 + 2 * da + 0.002 * g7.standard_normal(28); fb = 0.2 + 2 * db_ + 0.002 * g7.standard_normal(28)
    rd = r_did(fa, da, list(range(2, 26)), fb, db_, list(range(4, 28)))
    check("r_DiD: planted relation, common passing blocks", rd["blocks"] == list(range(4, 26)) and rd["r_DiD"] > 0.95 and rd["p_pos"] < 0.01, f"r {rd['r_DiD']:.3f}, p {rd['p_pos']:.4f}")
    # 9. passing set
    pz = np.zeros(28, bool); pz[2:9] = True; pz[12:19] = True; dl9 = np.arange(28) / 10.0; Pq, inf = passing_set(pz, dl9)
    pz2 = pz.copy(); pz2[12:28] = True; Pq2, inf2 = passing_set(pz2, dl9)
    check("passing set: longest run, tie by mean delta, >= 12", Pq == list(range(12, 19)) and inf["tie"] and not inf["ok"] and Pq2 == list(range(12, 28)) and inf2["ok"])
    # 10. end-to-end on planted synthetic worlds for both models (r_DiD in the second run)
    N_DRAWS, N_DBOOT = 48, 12; syn = os.path.join(work, "synth"); T17, T06 = _synth_topics(256, 28, 21), _synth_topics(256, 28, 22)
    lo_, hi_ = (f(np.concatenate([T["dl"]["last"]["delta"][3:] for T in (T17, T06)])) for f in (np.min, np.max))
    fgfun = lambda d: 0.35 + 0.5 * float(np.clip((d - lo_) / (hi_ - lo_), 0, 1))                  # one linear map for both models
    W17 = _synth_world(syn, "qwen3-1.7b", 256, 28, 21, fgfun, T17); W06 = _synth_world(syn, "qwen3-0.6b", 256, 28, 22, fgfun, T06)
    t1 = time.time(); r17 = run(syn, "qwen3-1.7b", force=True); te = time.time() - t1; r06 = run(syn, "qwen3-0.6b", force=True)
    fg = np.array(r17["per_block"]["pooled"]["FG|IMP-DES"]["point"]); tr = np.array([t["FG_IMP"] for t in W17["truth"]])
    check("synthetic world: P = blocks 3-27 (no contrast in 0-2)", r17["gate"]["P"] == list(range(3, 28)) and r17["gate"]["G30_ok"], str(r17["gate"]["P"]))
    check("synthetic world: FG(IMP-DES) per block recovered", np.nanmax(np.abs(fg[3:] - tr[3:])) < 0.12, f"max abs err {np.nanmax(np.abs(fg[3:] - tr[3:])):.3f}")
    ob = r17["over_blocks"]["P"]["dFG"]
    check("synthetic world: dFG grows with delta (D > 0, Spearman > 0.5, CI)", ob["D"] > 0.05 and ob["spearman_mmlu"] > 0.5 and ob["D_ci"][0] > 0 and "r3" in ob and "r3_p_pos" in ob,
          f"D {ob['D']:.3f} {ob['D_ci']}, rho {ob['spearman_mmlu']:.2f}, r3 {ob.get('r3', float('nan')):.2f} p {ob.get('r3_p_pos', float('nan')):.3f}")
    bnd = r17["per_block"]["pooled"]["FG|IMP-DES"]
    check("bands: simultaneous band wider than pointwise inside P, absent outside", all(bnd["sim_hi"][b] - bnd["sim_lo"][b] >= bnd["hi"][b] - bnd["lo"][b] - 1e-12 for b in range(3, 28))
          and all(bnd["sim_lo"][b] is None or not np.isfinite(bnd["sim_lo"][b]) for b in range(3)))
    check("synthetic world: causal statistics, composed gate, spillover and tau present", len(r17["gate"]["composed"]) == 28 and all(np.isfinite(r17["per_block"]["causal"]["FG|IMP-DES"]["point"][3:]))
          and len(r17["spillover_g_self"]["z_self"]) == 28 and len(r17["delta"]["tau"]["dot"]["tau|IMP-DES"]) == 28)
    check("r_DiD computed in the second model's run", np.isfinite(r06["r_DiD"].get("r_DiD", float("nan"))) and r06["r_DiD"]["n"] >= 12,
          f"r_DiD {r06['r_DiD'].get('r_DiD')} over {r06['r_DiD'].get('n')} blocks, p {r06['r_DiD'].get('p_pos')}")
    dz = np.load(paths(syn, "qwen3-1.7b")["draws"], allow_pickle=False)
    check("draws file: per-draw arrays [n_draws, 28] and the design", dz["pooled__dFG"].shape == (48, 28) and "design_para_split" in dz.files)
    t1 = time.time(); run(syn, "qwen3-1.7b", force=True); tr2 = time.time() - t1
    check("resume: the per-block cache is reused", tr2 < te, f"{te:.0f}s first, {tr2:.0f}s with the cache")
    GEOM.npz_save(paths(syn, "qwen3-1.7b")["geom"], dict(fingerprint={}), a=np.zeros(1))
    try: run(syn, "qwen3-1.7b", force=True); stale = False
    except RuntimeError: stale = True
    check("a local file that differs from geom_meta's sha256 is refused", stale)
    # 11. end-to-end on the MOCK model's outputs (r2_rgi_geom.py stages on rk_loop's tiny random model)
    mw = os.path.join(work, "mock"); GEOM.run(root, mw, "qwen3-1.7b"); rm = run(mw, "qwen3-1.7b", force=True)
    check("MOCK model: statistics written, finite delta, gates evaluated", os.path.exists(paths(mw, "qwen3-1.7b")["stats"]) and len(rm["gate"]["C0"]) == 28
          and np.isfinite(rm["delta"]["mmlu"]["last"]["boot_sd"]).all(), f"P = {rm['gate']['P']}, flags {rm['flags']}")
    n_ok = sum(ok for _, ok in res); print(f"selftest r2_rgi_stats: {n_ok}/{len(res)} passed ({time.time() - t0:.0f}s)", flush=True)
    return n_ok == len(res)


# ======================================================================================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="r2_rgi_stats.py (plan 9.2)", usage=__doc__)
    ap.add_argument("root"); ap.add_argument("r1_dir"); ap.add_argument("r2_dir"); ap.add_argument("stage", nargs="?", default="all")
    ap.add_argument("--model", default="qwen3-1.7b", choices=GEOM.MODELS); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv); torch.set_num_threads(int(os.environ.get("NT", "3")))
    if a.stage == "selftest": return 0 if selftest(a.root, a.r2_dir) else 1
    if a.stage != "all": raise SystemExit(f"unknown stage {a.stage!r} (all | selftest)")
    run(a.r2_dir, a.model, force=a.force); return 0


if __name__ == "__main__":
    sys.exit(main())
