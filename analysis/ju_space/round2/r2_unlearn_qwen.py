"""Round 2, Part C on Qwen3-1.7B at block 14 (final plan results/ju_space/plans/abschnitt11_12_runde2_plan.md, sha256 5e476340...edc2d):
section 5.2 C-Q2 to C-Q6 with the shared objects of 2.4 (O1 fixed gap axes, O2 axes, O4 statement sets, O5 behavioural read-outs, O6
random-direction controls), the gates of section 6 that are decided here (G13-G16, G43, G44; labels of G28) and the CLI of 9.2.

Stages (a stage is skipped when unlearn/<stage>.json exists; its intermediate units are cached in <r2_dir>/local/unlearn_<stage>/ (local
only, keyed by input signature and code fingerprint) so that a crashed stage resumes; 'all' runs the seven in order in one process, see
main(): a failing stage does not stop the independent later ones; q3, q4a and q2 need r2_axes.py's outputs in <r2_dir>, q4a also q3's):
  q3    C-Q3 unlearning by exact write-orthogonalisation (r2_rk.WriteOrth). 29 ablations: u_RSI, unit(u_RGI_perp), span(u_RSI,
        u_RGI_perp), u_diff, the O6 controls ('rand', 14, 2) = null 2, ('cov', 14, 0) = null 8 and [('cov', 14, 1), ('cov', 14, 2)], the 16
        null directions and the 8 two-dim null pairs [('rand', 14, i), ('cov', 14, i)]. Per ablation: damage (teacher-forced KL on the 8
        references), H30 and RGI30 discrimination D (+ per statement) and z_D against the null ablations (a control without itself), the
        statement bootstrap of D ('inverted'). 1,216 ablated texts (keys of the O2 texts; unprompted: ('ref', p)) and the 64 session
        references, read by the unablated model: g_abl on O1's fixed axes, g_abl / g_0 with the joint prompt bootstrap and the random
        reference, the own-axis gap, the foreign-axis moves (OD-C6), the unprompted shift. In session: the 'unlearned' rule for u_RSI and
        the span, which selects C-Q4a's ablation (G43, OD-G8); the causal linear-rung inputs of C-Q1.
  q4a   C-Q4a in-context re-search (OD-C3 cue) under the ablation that passed C-Q3 (u_RSI, else the span; u_RSI if none passed): 64 cued
        texts ablated and 64 unablated (keys of the O2 RSI_PRO texts); restored share (g_cue - g_abl) / (g_0 - g_abl) with the joint prompt
        bootstrap; the unablated ceiling; the rule.
  q2    C-Q2 counter-push at matched damage. (a) K6 at 0.03 nats (r2_rk.k6_direction 'r2_<dir>') for unit(u_RGI_perp), u_com, u_diff on H30
        and for unit(u_RGI_perp), u_com on RGI30, z_fin from R1.k6's regression (H30) or from the 16 R1 null directions evaluated on RGI30
        at their R1 push sizes; (b) alpha_u = the 0.05-nat push of each of u_RSI, u_diff, unit(u_RGI_perp), ('cov', 14, 3) (rk.set_alpha's
        rule, start alpha_14, calibrated on the counter-push -alpha_u unit(u)); 640 counter-pushed instructed texts read unpushed; (b')
        behaviour at alpha_u against the 16 null directions recalibrated to 0.05 nats on H30 and RGI30 (state entries 'r2_b05_<dir>',
        'r2_b05_null<i>'); (c) +- alpha_u content pairs on the RSI-topic prompts (keys ('r2CNT', direction, p)) and the perplexity of the
        scene prompts' reference continuations; the inputs of the 'suppressed' rule.
  q4b   C-Q4b loop under ablation. Gate G44 = (G13 LIKELY_wh and G15 twin start check) or G14 (attractor); ablated loop (writer with
        WriteOrth(u_RSI)) and unablated twin, same start and keys ('r2Q4B', n, p, 0, 14), 64 pairs, 3 rounds, both pushing unit(d_n
        orthogonalised against u_RSI); read-outs on u_T_perp with R1's nulls and the loop's own rel; Ue_n with its 90 % prompt bootstrap.
  armA  C-Q4 arm A: round 1 with the loop push along u_T_perp plus a constant -beta u_RSI on both sides (beta alone 0.03 nats, rk's cap
        rule), keys ('r2ARMA', n, p, 0, 14); G16 against the 90 % prompt-bootstrap interval of c(d_1) of the seed-0 half of R1 RSI_a,
        shifted so that the bootstrap median equals c (Amendment 3 (a)); rounds 2-3 only if c_1 falls below it.
  q5    C-Q5 pushes: superposition (alpha_14 (unit(u_T_perp) + unit(sea)) / sqrt 2, keys ('r2SUP', p)) and dose (2 alpha_14 along u_T_perp,
        keys ('r2DOSE', p)), 64 pairs each, against the round-1 read-backs of evaluation 0 (RSI_a seed-0 half, SEA).
  q6    C-Q6 nesting primitives and depth cap: rsi_direction.py's Q2' stimuli (288) with every state at blocks 8, 14, 18, 22; object-centred
        depth modes (12 training objects), helix test, angular against linear readout; caps of the top-r depth modes (r = 1, 2, 3) at block
        14 from the second 'the way it' on (OD-C4) against equal-norm random r-dim patches; next-token KL fractions with the same-depth
        floor; validated depth readouts at blocks 18 and 22 and self-repair.
The verdicts are decided in r2_ana (r2_unlearn_verdict.py, r2_linearity.py) with the rule functions of this module (rule_unlearned,
rule_suppressed, rule_cue, q4b_gate, rule_loop, arma_continues; OD-G8); the JSON files carry the rule inputs with their intervals and the
in-session values of the decisions that select what runs.

Outputs (in <r2_dir>): unlearn/<stage>.json (rule inputs with intervals, gates, labels, seeds, finder log), unlearn/<stage>.pt (per-text or
per-pair arrays, texts as token ids, bootstrap draws). A stage whose round-1 inputs are missing or not final writes only its JSON with
status 'not run (round-1 output missing)' (G4, G6); C-Q4b with a closed gate 'not testable (gate ...)'.
Usage: python r2_unlearn_qwen.py <root> <r1_dir> <r2_dir> [q3 | q4a | q2 | q4b | armA | q5 | q6 | all | selftest]
       env DEVICE, NT, MOCK; BLOCK (must be 14 if set). Exit code 0 if every requested stage is done ('ran', 'not run (...)', 'not testable
       (...)' or done earlier), 1 if one failed or was skipped because a prerequisite failed (a per-stage summary is printed)."""
import contextlib, gc, hashlib, json, math, os, shutil, subprocess, sys, time, traceback
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); JU = os.path.dirname(HERE)
if HERE not in sys.path: sys.path.insert(0, HERE)
import r2_rk as R
import r2_stimuli as S

PLAN = "results/ju_space/plans/abschnitt11_12_runde2_plan.md"
PLAN_SHA256 = "5e476340dfdb1d827582c869f2dfc08e32ed8e9a9dbebf7642acceb5bb5edc2d"
BLOCK = 14
STAGES = ("q3", "q4a", "q2", "q4b", "armA", "q5", "q6")
SEEDS = dict(boot=31, stmt=32, obj=33)          # OD-G6: fixed per script, written into every output JSON (C-Q6 random patches: keyseed('r2Q6rand', ...))
N_BOOT, LEVEL = 2000, 0.90
NOT_RUN_R1 = "not run (round-1 output missing)"
CONCEPT_SETS = dict(RSI=("RSI_PRO", "RSI_ANTI"), RGI=("RGI_PRO", "RGI_ANTI"))
# frozen thresholds (section 5.2 rules, OD-C6)
RULE = dict(gap_point=0.5, gap_hi=0.75, D_frac=0.5, z_D=-2.33, D0_min=0.2, damage_max=0.05, foreign=0.5, ue_frac=0.5, z_fin=2.33,
            M0_min=0.1, band05=(0.045, 0.055), share_shallow=0.5, share_deep=0.2, share_deep_hi=0.35, p_unsearched=0.10, A_small=0.25,
            p_sig=0.01, A_big=0.25, rel_rerouted=0.5, ue_rerouted=0.25, sup_cos=0.8, sup_coef=(0.5 / math.sqrt(2), 1.5 / math.sqrt(2)),
            q6_frac=0.5, q6_rand=0.2, q6_valid=0.75, q6_repair_depth=3.5, q6_repair_share=0.75)
CAL05 = dict(target=0.05, lo=0.045, hi=0.055, max_eval=6, start_factor=1.0)      # rk.set_alpha's rule, start alpha_14 (O5, C-Q2)
CAL03 = dict(target=0.03, lo=0.025, hi=0.035, max_eval=3, start_factor=0.75)     # rk.k6_direction's rule (O5): arm A's beta
Q6_BLOCKS = (8, 14, 18, 22)
Q6_READ_BLOCKS = (18, 22)
Q6_RANKS = (1, 2, 3)


def log(msg): print(f"[r2_unlearn {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def out_paths(r2_dir, stage):
    d = os.path.join(r2_dir, "unlearn"); return os.path.join(d, f"{stage}.json"), os.path.join(d, f"{stage}.pt")


def fin(x): return x is not None and isinstance(x, (int, float, np.floating)) and math.isfinite(float(x))


def _f(x):
    """float, NaN for None / non-finite input (for JSON rule inputs)."""
    try: return float(x)
    except (TypeError, ValueError): return float("nan")


def sig_of(*parts):
    """Signature of a cached unit: sha256 over the given numbers, strings and tensors (exact bytes)."""
    h = hashlib.sha256()
    for p in parts:
        if torch.is_tensor(p): h.update(p.detach().cpu().double().contiguous().numpy().tobytes())
        elif isinstance(p, np.ndarray): h.update(np.ascontiguousarray(p, dtype=np.float64).tobytes())
        else: h.update(repr(p).encode())
        h.update(b"|")
    return h.hexdigest()


CODE_FILES = (os.path.abspath(__file__), os.path.join(HERE, "r2_rk.py"), os.path.join(HERE, "r2_stimuli.py"), os.path.join(JU, "rk_loop.py"))
_CODE = {}


def code_sha256():
    """Code fingerprint (review unlearn_a-1): sha256 over the bytes of this script, r2_rk.py, r2_stimuli.py and rk_loop.py. Every cached
    unit and q2's round-2 state entries are valid only for it, so a resume after a code change never mixes two code versions."""
    if "sha" not in _CODE:
        h = hashlib.sha256()
        for p in CODE_FILES:
            h.update(os.path.basename(p).encode()); h.update(b"|"); h.update(open(p, "rb").read()); h.update(b"|")
        _CODE["sha"] = h.hexdigest()
    return _CODE["sha"]


class Cache:
    """Units of one stage in <r2_dir>/local/unlearn_<stage>/ (local only, never synced): a unit is computed once and reused when the
    stage is rerun after a crash; each unit carries the signature of its inputs and the code fingerprint (code_sha256) and is recomputed
    if either differs."""
    def __init__(self, r2_dir, stage, code=None): self.dir = os.path.join(r2_dir, "local", f"unlearn_{stage}"); self.code = code or code_sha256()

    def get(self, name, sig, fn):
        p = os.path.join(self.dir, name + ".pt")
        if os.path.exists(p):
            obj = torch.load(p, map_location="cpu", weights_only=False)
            if obj.get("sig") == sig and obj.get("code") == self.code: return obj["val"]
        val = fn(); R.torch_save(p, dict(sig=sig, code=self.code, val=val)); return val


# ======================================================================================================================================
# Statistics and rules (no model; importable by r2_ana, OD-G8)
# ======================================================================================================================================
def boot_g(per_prompt, W):
    """Bootstrap replicates of g (O1): sum_p W[b, p] k_p s_p / sum_p W[b, p] k_p with s_p = per-prompt values (NaN = dropped prompt, k_p = 0);
    W [B, P] prompt weights of the joint prompt bootstrap. Returns float64 [B] (NaN where a replicate keeps no prompt)."""
    s = torch.as_tensor(per_prompt).double(); k = torch.isfinite(s); W = torch.as_tensor(np.asarray(W), dtype=torch.float64) * k.double()[None]
    den = W.sum(1); out = (W @ torch.nan_to_num(s)) / den; out[den == 0] = float("nan"); return out


def pct(draws, q):
    x = np.asarray(draws, dtype=float); x = x[np.isfinite(x)]; return float(np.percentile(x, q)) if len(x) else float("nan")


def proj_mean(X, v):
    """Mean projection of the non-empty texts' states X [N, D] on unit(v)."""
    X = torch.as_tensor(X).double(); ok = ~torch.isnan(X).any(1); return float((X[ok] @ R.unit(torch.as_tensor(v).double())).mean()) if ok.any() else float("nan")


def foreign_moves(X_new, X_ref, axes, gaps):
    """OD-C6: m_X = |change of the mean projection of all instructed texts on unit(u_X)| / gap_cross_X (NaN if gap_cross_X <= 0)."""
    out = {}
    for name, v in axes.items():
        g = _f(gaps[name]); ch = abs(proj_mean(X_new, v) - proj_mean(X_ref, v))
        out[name] = ch / g if (fin(g) and g > 0) else float("nan")
    return out


def gap_entry(Xp, Xa, ref, W, foreign_axes, foreign_gaps):
    """One instructed text set (pro Xp, anti Xa [P, D], read under the plain prompt by the unmodified model) against the unpushed,
    unablated references ref = dict(Xp, Xa, fold, a_A, a_B, g0, g0_draws, Q): g on the fixed axes (O1), its bootstrap draws and the
    ratio g / g_0 per replicate (the same W for both), the 90 % interval of the ratio, m_concept = |g - g_0| / g_0 (OD-C6), the foreign
    moves and the own-axis gap (re-routing diagnostic). Returns (numbers for the JSON, arrays for the .pt)."""
    G = R.gap_on_fixed_axes(Xp, Xa, ref["fold"], ref["a_A"], ref["a_B"]); gd = boot_g(G["per_prompt"], W); rd = gd / ref["g0_draws"]
    g0 = ref["g0"]; lo, hi = R.interval(rd.numpy(), LEVEL)
    try:
        own = R.axis_from_texts(Xp, Xa, ref["fold"], basis=ref["Q"], n_null=0); own_gap, own_auc = own["gap_perp_cross"], own["auc"]
    except Exception as e:                                                       # e.g. a fold without a kept prompt (MOCK)
        own_gap = own_auc = float("nan"); log(f"own-axis gap not computable: {e}")
    Xn = torch.cat([torch.as_tensor(Xp).double(), torch.as_tensor(Xa).double()]); Xr = torch.cat([torch.as_tensor(ref["Xp"]).double(), torch.as_tensor(ref["Xa"]).double()])
    num = dict(g=G["g"], g0=g0, kept=G["kept"], ratio=G["g"] / g0 if (fin(g0) and g0 != 0) else float("nan"), ratio_ci90=[lo, hi],
               g_ci90=list(R.interval(gd.numpy(), LEVEL)), m_concept=abs(G["g"] - g0) / g0 if (fin(g0) and g0 > 0) else float("nan"),
               m_foreign=foreign_moves(Xn, Xr, foreign_axes, foreign_gaps), own_axis_gap=own_gap, own_axis_auc=own_auc,
               n_empty=int(torch.isnan(Xn).any(1).sum()))
    arr = dict(per_prompt=G["per_prompt"].float(), g_draws=gd.float(), ratio_draws=rd.float())
    return num, arr


def stmt_D(y, items):
    """D = mean y(pro statements) - mean y(anti statements) (O5), pro = answer matching the behaviour ' Yes'."""
    y = np.asarray(y, dtype=float); pro = np.array([x["answer_matching_behavior"].strip() == "Yes" for x in items])
    return float(y[pro].mean() - y[~pro].mean())


def stmt_boot_index(items, B=N_BOOT, seed=SEEDS["stmt"]):
    """Index draws of the statement bootstrap (pro and anti stratified, with replacement), the same draws for every ablation."""
    pro = np.nonzero([x["answer_matching_behavior"].strip() == "Yes" for x in items])[0]; anti = np.nonzero([x["answer_matching_behavior"].strip() != "Yes" for x in items])[0]
    g = np.random.default_rng(seed); return pro[g.integers(0, len(pro), (B, len(pro)))], anti[g.integers(0, len(anti), (B, len(anti)))]


def stmt_D_boot(y, idx):
    y = np.asarray(y, dtype=float); ip, ia = idx; return y[ip].mean(1) - y[ia].mean(1)


def z_against(x, null):
    """(x - mean(null)) / SD(null) with ddof 1 (O5, C-Q3 z_D; NaN if fewer than 2 nulls or SD 0)."""
    a = np.asarray(null, dtype=float); a = a[np.isfinite(a)]
    if len(a) < 2: return float("nan")
    sd = float(a.std(ddof=1)); return float((x - a.mean()) / sd) if sd > 0 else float("nan")


def ols_nulls(nulls):
    """O5 null regression: OLS of Ue_fin on A_fin over the null directions (rk.k6's formula): beta, se (df n - 2), mean a, S_aa, n."""
    a = np.array([_f(x["A_fin"]) for x in nulls]); u = np.array([_f(x["Ue_fin"]) for x in nulls]); X = np.stack([np.ones_like(a), a], 1)
    beta, *_ = np.linalg.lstsq(X, u, rcond=None); resid = u - X @ beta; se = math.sqrt(float((resid ** 2).sum()) / max(len(u) - 2, 1))
    return dict(beta=[float(beta[0]), float(beta[1])], se=se, a_mean=float(a.mean()), Saa=float(((a - a.mean()) ** 2).sum()), n=len(u))


def reg_from_r1_k6(k6):
    """The regression of R1.k6: beta and se as stored, a = the R1 nulls' A_fin (O5)."""
    a = np.array([_f(x["A_fin"]) for x in k6["nulls"]])
    return dict(beta=[float(k6["beta"][0]), float(k6["beta"][1])], se=float(k6["se"]), a_mean=float(a.mean()), Saa=float(((a - a.mean()) ** 2).sum()), n=len(a))


def z_fin(Ue, A, reg):
    """O5: z_fin = e / (se sqrt(1 + 1/n + (A_fin - mean a)^2 / S_aa)), e = Ue_fin - beta_0 - beta_1 A_fin (rk.k6's formula)."""
    e = _f(Ue) - reg["beta"][0] - reg["beta"][1] * _f(A)
    den = reg["se"] * math.sqrt(1 + 1 / reg["n"] + (_f(A) - reg["a_mean"]) ** 2 / max(reg["Saa"], 1e-12))
    return float(e / den) if den > 0 else float("nan")


def calibrate(dmg_fn, start, rule):
    """rk.set_alpha / rk.k6_direction's calibration: f <- f sqrt(target / damage) until the damage lies in [lo, hi] or max_eval damage
    evaluations were made; the closest kept. Returns dict(f, damage, tried, in_band, start)."""
    f, tried = float(start), []
    for _ in range(rule["max_eval"]):
        dm = float(dmg_fn(f)); assert math.isfinite(dm), "damage is not finite"; tried.append((f, dm))
        if rule["lo"] <= dm <= rule["hi"]: break
        f = f * math.sqrt(rule["target"] / max(dm, 1e-6))
    best = min(tried, key=lambda t: abs(t[1] - rule["target"]))
    return dict(f=best[0], damage=best[1], tried=tried, in_band=bool(rule["lo"] <= best[1] <= rule["hi"]), start=float(start), target=rule["target"])


def _gap_conditions(g, g0, ratio_hi, ctrl_p5):
    return dict(g0_positive=bool(fin(g0) and g0 > 0), gap_half=bool(fin(g) and fin(g0) and g0 > 0 and g <= RULE["gap_point"] * g0),
                gap_hi=bool(fin(ratio_hi) and ratio_hi <= RULE["gap_hi"]), gap_below_random=bool(fin(ratio_hi) and fin(ctrl_p5) and ratio_hi < ctrl_p5))


def _foreign_ok(m_foreign, m_concept):
    return bool(fin(m_concept) and all(fin(m) and m < RULE["foreign"] * m_concept for m in m_foreign.values()))


def rule_unlearned(g, g0, ratio_hi, ctrl_p5, D, D0, zD, damage, m_foreign, m_concept, D_hi=None):
    """C-Q3 'unlearned' (u_RSI or the span): g_abl <= 0.5 g_0 on the fixed axes, the 90 % upper end of g_abl / g_0 <= 0.75 and below the
    5th percentile of the random ablations' bootstrap distribution of g_abl / g_0 (ctrl_p5); D_abl <= 0.5 D_0 with z_D <= -2.33 (D_0 > 0.2
    required, else 'no baseline discrimination'); damage <= 0.05 nats; every foreign axis moves less than half as much as RSI (OD-C6).
    'inverted' if D_abl < 0 with the 90 % upper end of its statement bootstrap below 0 (D_hi)."""
    c = _gap_conditions(g, g0, ratio_hi, ctrl_p5)
    c.update(D_baseline=bool(fin(D0) and D0 > RULE["D0_min"]), D_half=bool(fin(D) and fin(D0) and D <= RULE["D_frac"] * D0),
             z_D=bool(fin(zD) and zD <= RULE["z_D"]), damage=bool(fin(damage) and damage <= RULE["damage_max"]), foreign=_foreign_ok(m_foreign, m_concept))
    ok = all(c.values()); failed = [k for k, v in c.items() if not v]
    note = "no baseline discrimination" if not c["D_baseline"] else ("no baseline gap" if not c["g0_positive"] else "")
    return dict(conditions=c, failed=failed, unlearned=ok, inverted=bool(fin(D) and D < 0 and D_hi is not None and fin(D_hi) and D_hi < 0),
                verdict="unlearned" if ok else "not unlearned", note=note)


def rule_suppressed(g, g0, ratio_hi, ctrl_p5, Ue, z, M0, damage, m_foreign, m_concept):
    """C-Q2 'suppressed' at the one push -alpha_u: (i) g_push <= 0.5 g_0, 90 % upper end of g_push / g_0 <= 0.75 and below the 5th
    percentile of the bootstrap distribution under ('cov', 14, 3) (same concept); (ii) Ue_fin >= 0.5 M0 and z_fin >= 2.33 at alpha_u on the
    concept's statements (M0 > 0.1 required, else 'no baseline discrimination'); (iii) the push's damage in 0.045-0.055 nats; (iv) every
    foreign axis moves less than half as much as the concept. Otherwise 'not suppressed' (never 'unlearned')."""
    c = _gap_conditions(g, g0, ratio_hi, ctrl_p5)
    c.update(M0_baseline=bool(fin(M0) and M0 > RULE["M0_min"]), Ue=bool(fin(Ue) and fin(M0) and Ue >= RULE["ue_frac"] * M0),
             z_fin=bool(fin(z) and z >= RULE["z_fin"]), damage_band=bool(fin(damage) and RULE["band05"][0] <= damage <= RULE["band05"][1]),
             foreign=_foreign_ok(m_foreign, m_concept))
    ok = all(c.values())
    return dict(conditions=c, failed=[k for k, v in c.items() if not v], suppressed=ok, verdict="suppressed" if ok else "not suppressed",
                note="no baseline discrimination" if not c["M0_baseline"] else "")


def restored_share(per_cue, per_abl, per_0, W):
    """C-Q4a: restored share (g_cue - g_abl) / (g_0 - g_abl) from the per-prompt values on the fixed axes of the cued (ablated) texts against
    C-Q3's ablated anti texts, of C-Q3's ablated instructed texts and of the unablated instructed texts, with the joint prompt bootstrap
    (the same W for all three). Undefined (NaN) if |g_0 - g_abl| <= 1e-9 |g_0| (the ablation left the instructed gap unchanged).
    Returns dict(share, ci90, defined, g_cue, g_abl, g0) and the draws."""
    nanmean = lambda v: float(np.nanmean(np.asarray(v, dtype=float))) if np.isfinite(np.asarray(v, dtype=float)).any() else float("nan")
    gc, ga, g0 = nanmean(per_cue), nanmean(per_abl), nanmean(per_0); den = g0 - ga; defined = bool(fin(den) and abs(den) > 1e-9 * abs(g0))
    dc, da, d0 = boot_g(per_cue, W), boot_g(per_abl, W), boot_g(per_0, W); draws = (dc - da) / (d0 - da)
    share = (gc - ga) / den if defined else float("nan")
    return dict(share=share, ci90=list(R.interval(draws.numpy(), LEVEL)), defined=defined, g_cue=gc, g_abl=ga, g0=g0), draws


def rule_cue(share, share_hi, passed):
    """C-Q4a: only if an ablation passed C-Q3: 'shallow' if the restored share >= 0.5; 'deep' if < 0.2 with its 90 % upper end < 0.35;
    otherwise 'partial'. If none passed: 'not applicable (nothing removed)' (the share is descriptive)."""
    if not passed: return "not applicable (nothing removed)"
    if fin(share) and share >= RULE["share_shallow"]: return "shallow"
    if fin(share) and share < RULE["share_deep"] and fin(share_hi) and share_hi < RULE["share_deep_hi"]: return "deep"
    return "partial"


def likely_wh(V):
    """LIKELY_wh (2.1): V_RSI starts with 'likely', or starts with 'not decidable (pilot gate' and the text after 'would have read: '
    starts with 'likely'. Returns (bool, from_would_have_read)."""
    V = str(V or "")
    if V.startswith("likely"): return True, False
    if V.startswith("not decidable (pilot gate") and "would have read: " in V and V.split("would have read: ", 1)[1].startswith("likely"): return True, True
    return False, False


def attracted_loop(label_attractor, rnd_rows):
    """G14: the attracted RND loop (p <= 0.01 and |A| >= 0.25 at its last row); RND_1 if both are attracted (OD-C12); None if the label is
    false or no RND loop qualifies. rnd_rows: {L: loops[14][L][3]}."""
    if not label_attractor: return None
    hits = [L for L in ("RND_1", "RND_2") if L in rnd_rows and fin(rnd_rows[L].get("p")) and fin(rnd_rows[L].get("A"))
            and rnd_rows[L]["p"] <= RULE["p_sig"] and abs(rnd_rows[L]["A"]) >= RULE["A_big"]]
    return hits[0] if hits else None


def q4b_gate(G13, G15, attracted):
    """G44: open if (G13 and G15) or G14 (an attracted RND loop). Branch 'likely' (start from d_1^RSI_a without its u_RSI part) has
    priority; 'attractor' (start from the attracted loop's start without its u_RSI part)."""
    if G13 and G15: return dict(open=True, branch="likely")
    if attracted is not None: return dict(open=True, branch="attractor", loop=attracted)
    why = []
    if not G13: why.append("G13 (LIKELY_wh) false")
    elif not G15: why.append("G15 (twin start: c > 0 and p <= 0.01) failed")
    why.append("G14 (attractor) false")
    return dict(open=False, branch=None, verdict=f"not testable (gate: {'; '.join(why)})")


def rule_loop(abl, twin=None, comparator=None, ref_name=None):
    """C-Q4b rule on the ablated loop at d_2 and d_3 (abl / twin: {n: dict(c, p, A, rel, Ue, Ue_ci90)} for n = 2, 3), first match in the
    order of the plan text (review unlearn_b-3; the same order as r2_unlearn_verdict.loop_rule, OD-G8): 'un-searched' if p > 0.10 and
    |A| < 0.25 while the twin (or, for arm A, the R1 seed-0 comparator) has p <= 0.01 and A >= 0.25; 'survives' if p <= 0.01, c > 0 and
    A >= 0.25; 're-routed' if rel >= 0.5, Ue >= 0.25 with its 90 % interval above 0 and |A| < 0.25; 'undecided (<reference> does not
    re-find RSI)' if the reference fails p <= 0.01 and A >= 0.25; otherwise 'undecided'. Every condition at both d_2 and d_3."""
    ref = twin if twin is not None else comparator; name = ref_name or ("the R1 seed-0 comparator" if (twin is None and comparator is not None) else "twin")
    ns = (2, 3); g = lambda r, k: r.get(k) if isinstance(r, dict) else None
    if abl is None or any(n not in abl for n in ns): return "undecided (rounds d_2, d_3 missing)"
    allv = lambda f, rows: all(f(rows[n]) for n in ns)
    gone = allv(lambda r: fin(g(r, "p")) and r["p"] > RULE["p_unsearched"] and fin(g(r, "A")) and abs(r["A"]) < RULE["A_small"], abl)
    refind = ref is not None and all(n in ref for n in ns) and allv(lambda r: fin(g(r, "p")) and r["p"] <= RULE["p_sig"] and fin(g(r, "A")) and r["A"] >= RULE["A_big"], ref)
    surv = allv(lambda r: fin(g(r, "p")) and r["p"] <= RULE["p_sig"] and fin(g(r, "c")) and r["c"] > 0 and fin(g(r, "A")) and r["A"] >= RULE["A_big"], abl)
    rer = allv(lambda r: fin(g(r, "rel")) and r["rel"] >= RULE["rel_rerouted"] and fin(g(r, "Ue")) and r["Ue"] >= RULE["ue_rerouted"]
               and isinstance(g(r, "Ue_ci90"), (list, tuple)) and fin(r["Ue_ci90"][0]) and r["Ue_ci90"][0] > 0 and fin(g(r, "A")) and abs(r["A"]) < RULE["A_small"], abl)
    if gone and refind: return "un-searched"
    if surv: return "survives"
    if rer: return "re-routed"
    if not refind: return f"undecided ({name} does not re-find RSI)"
    return "undecided"


def centred_interval(c, draws, level=LEVEL):
    """Amendment 3 (a) (G16): the percentile interval [q_lo, q_hi] of the bootstrap draws shifted so that their median m coincides with the
    point value c: [q_lo + c - m, q_hi + c - m]. Returns dict(ci90 (shifted), ci90_percentile (unshifted), boot_median)."""
    lo, hi = R.interval(draws, level); m = pct(draws, 50); s = c - m if (fin(c) and fin(m)) else float("nan")
    return dict(ci90=[lo + s, hi + s], ci90_percentile=[lo, hi], boot_median=m)


def arma_continues(c1, ref_lo):
    """G16: arm A continues with rounds 2-3 only if c_1 falls below the 90 % interval of c(d_1) of the seed-0 half of R1 RSI_a."""
    return bool(fin(c1) and fin(ref_lo) and c1 < ref_lo)


def rel_split(fn, n_prompts):
    """Split-half reliability over the 20 K2 halvings (rk.stats's formula and floor): r = mean cos(fn(half A), fn(half B)), fn(mask) -> vector."""
    rs = []
    for m in R.halvings(n_prompts):
        m = torch.as_tensor(m); rs.append(R.cosf(fn(m), fn(~m)))
    r = float(np.mean(rs))
    if not math.isfinite(r): return dict(rel=float("nan"), r=r)
    return dict(rel=max(2 * r / (1 + r), 0.05) if r > -1 else 0.05, r=r)


def pair_mean(ev, mask_prompts=None, rows=None):
    """Mean of delta over ok pairs (optionally only the pairs in 'rows' and of prompts in mask_prompts [P] bool)."""
    d, ok, po = ev["delta"].double(), torch.as_tensor(ev["ok"], dtype=torch.bool), torch.as_tensor(ev["prompt_of"])
    m = ok.clone()
    if rows is not None: r = torch.zeros_like(m); r[rows] = True; m &= r
    if mask_prompts is not None: m &= torch.as_tensor(mask_prompts)[po]
    return d[m].mean(0) if m.any() else torch.full((d.shape[1],), float("nan"), dtype=torch.float64)


def seed0_rows(ev, n_prompts):
    """Indices of the seed-0 pairs of an rk evaluation (prompt-major, nseed seeds per prompt: pair j = p * nseed + s)."""
    n = len(ev["prompt_of"]); nseed = n // n_prompts; assert nseed * n_prompts == n, "evaluation is not prompt-major"
    return torch.arange(0, n, nseed)


def u_norm2(delta, ok, w=None):
    """U-statistic estimate of |E delta|^2 over the ok pairs: the mean of delta_j . delta_k over distinct pairs j != k,
    (|sum w delta|^2 - sum w^2 |delta|^2) / ((sum w)^2 - sum w^2); unit weights give (|sum delta|^2 - sum |delta|^2) / (n (n - 1)); with
    bootstrap counts w, products of copies of the same pair are excluded."""
    d = delta.double()[ok]; w = torch.ones(len(d), dtype=torch.float64) if w is None else torch.as_tensor(w, dtype=torch.float64)[ok]
    s = (w[:, None] * d).sum(0); den = float(w.sum() ** 2 - (w * w).sum())
    return float((s @ s - (w * w * (d * d).sum(1)).sum()) / den) if den > 0 else float("nan")


def superposition_stats(F_mix, F_rsi, F_sea, u_t, u_sea):
    """C-Q5: coefficients of the least-squares solution of F(mix) on [F(RSI), F(SEA)] within span(unit(u_T_perp), unit(sea)) and the raw
    cos(F(mix), (F(RSI) + F(SEA)) / sqrt 2)."""
    E, _ = torch.linalg.qr(torch.stack([R.unit(u_t.double()), R.unit(u_sea.double())], 1))
    A = torch.stack([E.T @ F_rsi, E.T @ F_sea], 1); b = E.T @ F_mix
    coef = torch.linalg.lstsq(A, b[:, None]).solution[:, 0]
    F_sum = (F_rsi + F_sea) / math.sqrt(2)
    return dict(coef_RSI=float(coef[0]), coef_SEA=float(coef[1]), cos=R.cosf(F_mix, F_sum)), F_sum


# ======================================================================================================================================
# Model side (block 14 open in this process)
# ======================================================================================================================================
class MissingInput(Exception):
    """A round-2 input of an earlier step is missing (a session-order problem, not a round-1 file)."""


class RefKL:
    """KL(unmodified || modified) over the continuation positions of the first n round-1 reference texts, teacher-forced: rk.damage's texts,
    positions and formula, with the unmodified log-probabilities computed once. push(d, alpha, sign) with sign +1 equals rk.damage(d, alpha)
    (the push sign * alpha * min(|h_t|, cap_b) * unit(d) from the first user token on); hooks(...) is C-Q3's damage of an ablation."""
    def __init__(self, rk, n=8):
        self.rk = rk; Rf = rk.ST["ref"]; self.items = []
        for p in range(n):
            seq = rk.PIDS[p] + Rf["conts"][p]; P0 = len(rk.PIDS[p]); pos = torch.arange(P0 - 1, len(seq) - 1).to(rk.DEV)
            if len(pos) == 0: continue
            ids = torch.tensor([seq]).to(rk.DEV)
            with torch.no_grad(): a = torch.log_softmax(rk.model(ids).logits[0][pos].double(), -1)
            self.items.append((ids, pos, a))

    def _kl(self, setup):
        rk = self.rk; kl = []
        for ids, pos, a in self.items:
            with torch.no_grad():
                setup()
                try: b = torch.log_softmax(rk.model(ids).logits[0][pos].double(), -1)
                finally: rk.steer_off()
            kl.append(float((a.exp() * (a - b)).sum(-1).mean()))
        return float(np.mean(kl))

    def push(self, d, alpha, sign=1.0):
        rk = self.rk
        return self._kl(lambda: rk.STEER.update(v=R.unit(torch.as_tensor(d).float()), V=None, alpha=alpha, cap=rk.ST["ref"]["cap"],
                                                sign=torch.tensor([float(sign)]), rowstart=torch.tensor([rk.START])))

    def hooks(self, hooks):
        with contextlib.ExitStack() as es:
            for h in hooks: es.enter_context(h)
            return self._kl(lambda: None)


class ConstPush:
    """Arm A's constant push (C-Q4): adds sign * beta * min(|h_t|, cap_b) * unit(u) to the output of block 14 on every row (both sides of
    a pair), from the first user-content token on (rk's rowstart during the prefill, every position during decoding), with |h_t| the block
    output before any push (recorded by a prepend hook) - i.e. a second term of rk's steer hook with the same cap rule, registered after it.
    The product is formed in rk's order, so ConstPush(u, beta, -1) with a zero loop push equals rk's push of u at beta with sign -1."""
    def __init__(self, rk, u, beta, sign=-1.0):
        self.rk, self.v, self.beta, self.sign, self.cap = rk, R.unit(torch.as_tensor(u).float()), float(beta), float(sign), float(rk.ST["ref"]["cap"])   # unit in float32 as rk
        self.h = []; self.raw = None

    def _pre(self, mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out; self.raw = x.norm(dim=-1, keepdim=True).detach(); return None

    def _post(self, mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out; B, T = x.shape[:2]; rs = self.rk.STEER["rowstart"]
        m = (torch.arange(T, device=x.device)[None, :] >= rs.to(x.device)[:, None]).to(x.dtype)[..., None] if (T > 1 and rs is not None) else torch.ones(B, T, 1, dtype=x.dtype, device=x.device)
        nrm = self.raw.clamp(max=self.cap)
        add = self.beta * nrm * torch.full((B,), self.sign, dtype=x.dtype, device=x.device)[:, None, None] * self.v.to(x.device, x.dtype) * m
        x = x + add; return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x

    def __enter__(self):
        L = self.rk.LAY[self.rk.BL]; self.h = [L.register_forward_hook(self._pre, prepend=True), L.register_forward_hook(self._post)]; return self

    def __exit__(self, *a):
        for h in self.h: h.remove()
        self.h = []; self.raw = None


class Ctx:
    """Session context of a model stage at block 14: the open rk_loop module, R1 axes, the prompt bootstrap, the statement sets, O2."""
    def __init__(self, root, r1_dir, r2_dir, stage, rk):
        self.root, self.r1, self.r2, self.stage, self.rk = root, r1_dir, r2_dir, stage, rk
        ax = rk.ST["axes"]; self.ax = ax; self.uT = ax["u_T_perp"].double(); self.u_rsi = R.unit(self.uT); self.fold = ax["fold"].clone()
        self.null_uT = ax["null_u_T_perp"]; self.Delta_T = float(ax["gap_perp_cross"]); self.NP, self.NRSI = rk.NP, rk.NRSI
        self.alpha, self.cap = float(R.BLK["alpha"]), float(R.BLK["cap"]); self.W = R.prompt_bootstrap(self.NP, B=N_BOOT, seed=SEEDS["boot"])
        self.cache = Cache(r2_dir, stage); self._refkl = None; self._ids = {}; self._o2 = None
        assert rk.TEMP == 0.7 and rk.STEER["v"] is None, "block 14 must be unsteered at T 0.7"

    # ---- statement sets (O4)
    def H30(self): return list(self.rk.persona_sets()["H30"])

    def RGI30(self): return S.rgi30_items()

    def stmt_ids(self, x):
        q = x["question"]
        if q not in self._ids: self._ids[q] = torch.tensor([self.rk.enc(self.rk.chat(q))]).to(self.rk.DEV)
        return self._ids[q]

    def scores(self, H, d=None, f=0.0, sign=1.0, hooks=()):
        """Per statement: lo = log-odds of the matching answer, y = log p(Yes) - log p(No) at the answer position (rk.logodds_matching's
        computation, statement by statement; push sign * f * min(|h_t|, cap_b) * unit(d) from the first user token on; hooks during the
        forward). Returns (lo, y) float64 arrays."""
        rk = self.rk; lo, ys = [], []
        with contextlib.ExitStack() as es:
            for h in hooks: es.enter_context(h)
            for x in H:
                if d is not None: rk.STEER.update(v=R.unit(torch.as_tensor(d).float()), V=None, alpha=f, cap=rk.ST["ref"]["cap"], sign=torch.tensor([float(sign)]), rowstart=torch.tensor([rk.START]))
                try:
                    with torch.no_grad(): lp = torch.log_softmax(rk.model(self.stmt_ids(x)).logits[0, -1].double(), -1)
                finally: rk.steer_off()
                m = rk.YES if x["answer_matching_behavior"].strip() == "Yes" else rk.NO; o = rk.NO if m == rk.YES else rk.YES
                lo.append(float(lp[m] - lp[o])); ys.append(float(lp[rk.YES] - lp[rk.NO]))
        return np.array(lo), np.array(ys)

    def behaviour(self, H, d, f):
        """O5 at the push f: M and Y at +f and -f; Ue_fin = (M(+f) - M(-f)) / 2, A_fin = (Y(+f) - Y(-f)) / 2 (per-statement values kept)."""
        lp, yp = self.scores(H, d, f, 1.0); lm, ym = self.scores(H, d, f, -1.0)
        Mp, Mm, Yp, Ym = float(lp.mean()), float(lm.mean()), float(yp.mean()), float(ym.mean())
        return dict(f=float(f), Mp=Mp, Mm=Mm, Yp=Yp, Ym=Ym, Ue_fin=(Mp - Mm) / 2, A_fin=(Yp - Ym) / 2, lo_p=lp.tolist(), lo_m=lm.tolist(), y_p=yp.tolist(), y_m=ym.tolist())

    # ---- damage
    def refkl(self):
        if self._refkl is None: self._refkl = RefKL(self.rk)
        return self._refkl

    # ---- generation and reading
    def generate(self, rows, d=None, alpha=0.0, hooks=()):
        rk = self.rk
        with contextlib.ExitStack() as es:
            for h in hooks: es.enter_context(h)
            rk.capped_reset(); conts = rk.generate(rows, d, alpha); cs = rk.capped_share() if d is not None else float("nan")
        return conts, cs

    def read(self, conts):
        """Block-14 mean over the continuation tokens of each text, read as the reply to its plain prompt by the unmodified model (rk.read)."""
        return self.rk.read(self.rk.PIDS[: len(conts)], conts)

    # ---- O2 (axes/r2_axes.pt of this r2_dir)
    def o2(self):
        """O2 / O1 at block 14 from r2_axes.py's outputs; dict(B=None, BJ) if block 14 has no O2 statistics there (round-1 output missing)."""
        if self._o2 is not None: return self._o2
        p_pt, p_js = os.path.join(self.r2, "axes", "r2_axes.pt"), os.path.join(self.r2, "axes", "r2_axes.json")
        if not os.path.exists(p_js): raise MissingInput(f"{p_js} missing: run r2_axes.py first (same session)")
        AJ = json.load(open(p_js)); BJ = AJ.get("blocks", {}).get(str(BLOCK), {})
        if BJ.get("status") != "ok": self._o2 = dict(B=None, BJ=dict(BJ, status=BJ.get("status") or AJ.get("status") or "missing")); return self._o2
        if not os.path.exists(p_pt): raise MissingInput(f"{p_pt} missing: run r2_axes.py first (same session)")
        AX = torch.load(p_pt, map_location="cpu", weights_only=False); B = AX["blocks"][BLOCK]; rk = self.rk
        assert dict(AX["instr"]) == dict(S.O2_INSTR), "O2 instructions differ from r2_stimuli.O2_INSTR"
        assert list(AX["prompts"]) == list(rk.PROMPTS), "O2 prompts differ from rk.PROMPTS"
        if not rk.MOCK:
            for nm in S.O2_SETS: assert list(AX["keys"][nm]) == [rk.keyseed("axis", "rsi", p) for p in range(rk.NP)], f"O2 keys of {nm} are not keyseed('axis', 'rsi', p)"
        refs = {}
        for c, (pro, anti) in CONCEPT_SETS.items():
            F = B["fixed_axes"][c]; Q = B["basis_R1"] if c == "RSI" else B["basis_RGI"]
            G = R.gap_on_fixed_axes(B["X"][pro], B["X"][anti], self.fold, F["a_A"], F["a_B"])
            assert abs(G["g"] - F["g0"]) <= 1e-6 * max(1.0, abs(F["g0"])), f"g_0 of {c} does not reproduce O1's g0"
            refs[c] = dict(Xp=B["X"][pro].double(), Xa=B["X"][anti].double(), fold=self.fold, a_A=F["a_A"].double(), a_B=F["a_B"].double(),
                           g0=float(F["g0"]), Q=Q.double(), per_prompt=G["per_prompt"], g0_draws=boot_g(G["per_prompt"], self.W))
        foreign_axes = dict(sports=self.ax["sports"], tone=self.ax["tone"], sea=self.ax["sea"], TOPIC=B["u_TOPIC"])
        foreign_gaps = dict(sports=self.ax["gap_cross"]["sports"], tone=self.ax["gap_cross"]["tone"], sea=self.ax["gap_cross"]["sea"],
                            TOPIC=BJ["axes"]["TOPIC"]["gap_cross"])
        GA = bool(B["GA"])
        self._o2 = dict(B=B, BJ=BJ, texts=AX["texts"], keys=AX["keys"], refs=refs, foreign_axes=foreign_axes, foreign_gaps=foreign_gaps, GA=GA,
                        labels_rgi=[] if GA else ["RGI axis not resolved"], u_RGI_perp=R.unit(B["u_RGI_perp"].double()),
                        u_diff=R.unit(B["u_diff"].double()), u_com=R.unit(B["u_com"].double()), Delta_RGI=float(BJ["Delta_RGI"]),
                        g0_RSI=float(BJ["g0_RSI"]))
        return self._o2

    def instructed_rows(self, concept, sign=1.0):
        """Rows (prompt ids, key, sign) of the 64 pro and 64 anti instructed texts of a concept, interleaved per prompt in rk_loop.text_axis's
        order, with the keys of this session's O2 texts (keyseed('axis', 'rsi', p); MOCK: r2_axes' MOCK rule)."""
        rk = self.rk; O = self.o2(); pro, anti = CONCEPT_SETS[concept]; rows = []
        for p in range(rk.NP):
            for nm in (pro, anti): rows.append((rk.enc(rk.chat(S.instructed(rk.PROMPTS[p], S.O2_INSTR[nm]))), int(O["keys"][nm][p]), float(sign)))
        return rows

    def instructed_texts(self, tag, concept, d=None, alpha=0.0, sign=1.0, hooks=(), hook_sig=""):
        """64 pro + 64 anti instructed texts of a concept generated under an operation (push sign * alpha * unit(d) and / or hooks), read by
        the unmodified model under the plain prompt (cached unit). Returns dict(pro, anti, keys, Xp, Xa, capped_share)."""
        rows = self.instructed_rows(concept, sign)
        sg = sig_of("instructed", concept, [r[1] for r in rows], d if d is not None else "none", alpha, sign, hook_sig)

        def make():
            conts, cs = self.generate(rows, d, alpha, hooks); pro, anti = conts[0::2], conts[1::2]
            return dict(pro=pro, anti=anti, keys=[r[1] for r in rows[0::2]], keys_anti=[r[1] for r in rows[1::2]], Xp=self.read(pro).float(),
                        Xa=self.read(anti).float(), capped_share=cs)
        return self.cache.get(f"texts_{tag}_{concept}", sg, make)


def base_info(stage, extra=None):
    d = dict(script="r2_unlearn_qwen.py", plan=PLAN, plan_sha256=PLAN_SHA256, amendments="Amendments 1-3 (G16: Amendment 3 (a))", stage=stage,
             block=BLOCK, mock=R.MOCK, seeds=dict(SEEDS), n_boot=N_BOOT, level=LEVEL, rule_thresholds=RULE, code_sha256=code_sha256())
    if R.BLK: d.update(alpha_14=R.BLK.get("alpha"), alpha_scale=R.BLK.get("alpha_scale"), suffix=R.BLK.get("suffix"), cap=R.BLK.get("cap"))
    d.update(extra or {}); return d


def versions():
    import transformers
    return dict(torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def r1_meta_gpu1(r1_dir):
    """R1.meta of gpu1 (G47; metadata only)."""
    p = os.path.join(r1_dir, "gpu1", "shard_done.json")
    if not os.path.exists(p): return None
    m = json.load(open(p)); return {k: m.get(k) for k in ("gpu", "torch", "transformers", "code_commit")}


_LOG = dict(stage=None, start=0, open_records=[], open_stage=None)


def stage_finder_log():
    """The round-1 file choices of the running stage: open_block's (made in this process, perhaps by an earlier stage) and the stage's own,
    so that a stage's JSON does not depend on how many stages share the process."""
    pre = list(_LOG["open_records"]) if _LOG["open_stage"] not in (None, _LOG["stage"]) else []
    return pre + list(R.FINDER_LOG[_LOG["start"]:])


def write_outputs(r2_dir, stage, js, pt=None):
    """.pt first, then the JSON (the JSON's existence marks the stage done)."""
    js_p, pt_p = out_paths(r2_dir, stage)
    if pt is not None: R.torch_save(pt_p, pt)
    js = dict(js, finder=stage_finder_log(), versions=versions()); R.write_json(js_p, js); return js


def not_run(r2_dir, stage, status, **extra):
    log(f"{stage}: {status} {extra.get('error', '')}"); return write_outputs(r2_dir, stage, base_info(stage, dict(status=status, **extra)))


def null_dirs16(D):
    """O6: the 16 null directions in rk.k6's order: ('rand', 14, i) = null i, ('cov', 14, i) = null 8 + i (i = 0..7), as Pool directions
    (MOCK: r2_rk's seeded MOCK Pool directions, so that every MOCK null set has 16 members)."""
    return [(f"{k}_14_{i}", R.pool_dir(k, BLOCK, i, D=D)) for k in ("rand", "cov") for i in range(8)]


# ======================================================================================================================================
# C-Q3
# ======================================================================================================================================
Q3_TEXTS = (("u_RSI", "RSI"), ("u_RSI", "unprompted"), ("span", "RSI"), ("span", "RGI"), ("u_diff", "RSI"), ("u_RGI_perp", "RGI"),
            ("rand_14_2", "RSI"), ("cov_14_0", "RSI"), ("ctrl2", "RSI"), ("ctrl2", "RGI"))


def q3_ablations(ctx):
    """The 29 ablations of C-Q3 (O6): targets, the three controls, 16 one-dim nulls, 8 two-dim null pairs. The 1-dim controls are nulls 2
    and 8; 'rand_14_2' / 'cov_14_0' name them."""
    O = ctx.o2(); D = ctx.rk.D; u_rsi, u_rgi = ctx.u_rsi, O["u_RGI_perp"]; pd = lambda k, i: R.pool_dir(k, BLOCK, i, D=D)
    A = {}
    A["u_RSI"] = dict(U=u_rsi[None], dim=1, role="target", concept="RSI")
    A["u_RGI_perp"] = dict(U=u_rgi[None], dim=1, role="target", concept="RGI")
    A["span"] = dict(U=torch.stack([u_rsi, u_rgi]), dim=2, role="target", concept="RSI")
    A["u_diff"] = dict(U=O["u_diff"][None], dim=1, role="target", concept="RSI")
    A["ctrl2"] = dict(U=torch.stack([pd("cov", 1), pd("cov", 2)]), dim=2, role="control", pool=["cov_14_1", "cov_14_2"])
    for j, (nm, v) in enumerate(null_dirs16(D)): A[f"null{j}"] = dict(U=v[None], dim=1, role="null", pool=[nm])
    for i in range(8): A[f"pair{i}"] = dict(U=torch.stack([pd("rand", i), pd("cov", i)]), dim=2, role="null", pool=[f"rand_14_{i}", f"cov_14_{i}"])
    alias = dict(rand_14_2="null2", cov_14_0="null8")
    return A, alias


def stage_q3(ctx):
    rk = ctx.rk; O = ctx.o2()
    if O["B"] is None: return not_run(ctx.r2, "q3", NOT_RUN_R1, error=f"O2 at block 14: {O['BJ'].get('status', 'missing')}")
    H30, RGI = ctx.H30(), ctx.RGI30(); A, alias = q3_ablations(ctx); name = lambda n: alias.get(n, n)
    idx_H, idx_G = stmt_boot_index(H30), stmt_boot_index(RGI); KL = ctx.refkl()
    # unablated baselines
    base = ctx.cache.get("base", sig_of("base", [x["question"] for x in H30 + RGI]), lambda: dict(H30=ctx.scores(H30), RGI30=ctx.scores(RGI)))
    D0 = dict(H30=stmt_D(base["H30"][1], H30), RGI30=stmt_D(base["RGI30"][1], RGI))
    # every ablation: damage, statement scores
    ops = {}
    for nm, a in A.items():
        def make(a=a):
            W_ = R.WriteOrth(a["U"]); return dict(damage=KL.hooks([W_]), H30=ctx.scores(H30, hooks=[W_]), RGI30=ctx.scores(RGI, hooks=[W_]))
        ops[nm] = ctx.cache.get(f"op_{nm}", sig_of("op", a["U"]), make)
    log(f"q3: {len(ops)} ablations: damage and statements done")
    # texts generated under the ablation, read by the unablated model
    refs_unprompted = ctx.cache.get("ref_unprompted", sig_of("ref", [rk.keyseed("ref", p) for p in range(rk.NP)]), lambda: _gen_unprompted(ctx, ()))
    texts = {}
    for nm, c in Q3_TEXTS:
        a = A[name(nm)]; W_ = R.WriteOrth(a["U"])
        if c == "unprompted": texts[(nm, c)] = ctx.cache.get(f"texts_{nm}_unprompted", sig_of("unprompted", a["U"]), lambda W_=W_: _gen_unprompted(ctx, (W_,)))
        else: texts[(nm, c)] = ctx.instructed_texts(nm, c, hooks=(W_,), hook_sig=sig_of("WriteOrth", a["U"]))
    log("q3: ablated texts done")
    # numbers
    J = {}; P = dict(ablations={}, texts={}, refs_unprompted=refs_unprompted)
    for nm, a in A.items():
        o = ops[nm]; e = dict(dim=a["dim"], role=a["role"], pool=a.get("pool"), damage=o["damage"])
        for sset, H, idx in (("H30", H30, idx_H), ("RGI30", RGI, idx_G)):
            y = o[sset][1]; Dv = stmt_D(y, H); bD = stmt_D_boot(y, idx); lo, hi = R.interval(bD, LEVEL)
            e[sset] = dict(D=Dv, D0=D0[sset], dD=Dv - D0[sset], D_ci90_statements=[lo, hi], M=float(np.mean(o[sset][0])), inverted=bool(Dv < 0 and hi < 0))
        J[nm] = e; P["ablations"][nm] = dict(U=a["U"].float(), H30=o["H30"], RGI30=o["RGI30"], damage=o["damage"])
    for nm in ("u_RSI", "u_RGI_perp", "u_diff", "span", "ctrl2", "null2", "null8"):
        e = J[nm]
        if a_dim(A, nm) == 1: pool_ = [f"null{j}" for j in range(16) if f"null{j}" != nm]
        else: pool_ = [f"pair{i}" for i in range(8) if not (nm == "ctrl2" and i in (1, 2))]
        for sset in ("H30", "RGI30"):
            e[sset]["z_D"] = z_against(e[sset]["dD"], [J[q][sset]["dD"] for q in pool_]); e[sset]["z_D_nulls"] = pool_
    for (nm, c), T in texts.items():
        if c == "unprompted": continue
        num, arr = gap_entry(T["Xp"], T["Xa"], O["refs"][c], ctx.W, O["foreign_axes"], O["foreign_gaps"])
        J[name(nm)].setdefault("texts", {})[c] = num; P["texts"][f"{nm}|{c}"] = dict(T, **arr)
    # random references (bootstrap distributions of g / g_0 under the random ablations of the same dimension)
    rd = lambda nm, c: P["texts"][f"{nm}|{c}"]["ratio_draws"].numpy()
    ctrl_p5 = dict(u_RSI=pct(np.concatenate([rd("rand_14_2", "RSI"), rd("cov_14_0", "RSI")]), 5), span=pct(rd("ctrl2", "RSI"), 5),
                   span_RGI=pct(rd("ctrl2", "RGI"), 5))
    # unprompted shift (u_RSI ablation against the session references, same keys)
    Tu = texts[("u_RSI", "unprompted")]; unpr = _unprompted_shift(ctx, Tu, refs_unprompted, O)
    J["u_RSI"]["unprompted"] = unpr["num"]; P["texts"]["u_RSI|unprompted"] = dict(Tu, **unpr["arr"])
    # rules: 'unlearned' (u_RSI, span; RSI concept), descriptive numbers for the others
    rules = {}
    for nm in ("u_RSI", "span"):
        t = J[nm]["texts"]["RSI"]; h = J[nm]["H30"]
        rules[nm] = rule_unlearned(t["g"], t["g0"], t["ratio_ci90"][1], ctrl_p5[nm], h["D"], h["D0"], h["z_D"], J[nm]["damage"], t["m_foreign"],
                                   t["m_concept"], D_hi=h["D_ci90_statements"][1])
        rules[nm]["ctrl_p5"] = ctrl_p5[nm]; rules[nm]["controls"] = ["rand_14_2", "cov_14_0"] if nm == "u_RSI" else ["ctrl2 (cov_14_1, cov_14_2)"]
    chosen = "u_RSI" if rules["u_RSI"]["unlearned"] else ("span" if rules["span"]["unlearned"] else None)
    rung = dict(RSI=_rung(J["u_RSI"]["texts"]["RSI"]["ratio"], J["span"]["texts"]["RSI"]["ratio"], J["u_RSI"]["texts"]["RSI"]["g0"]),
                RGI=_rung(J["u_RGI_perp"]["texts"]["RGI"]["ratio"], J["span"]["texts"]["RGI"]["ratio"], J["u_RGI_perp"]["texts"]["RGI"]["g0"]),
                inputs=dict(RSI_1dim=dict(ablation="u_RSI", **_ratio_view(J["u_RSI"]["texts"]["RSI"])), RSI_span=dict(ablation="span", **_ratio_view(J["span"]["texts"]["RSI"])),
                            RGI_1dim=dict(ablation="u_RGI_perp", **_ratio_view(J["u_RGI_perp"]["texts"]["RGI"])), RGI_span=dict(ablation="span", **_ratio_view(J["span"]["texts"]["RGI"]))),
                note="C-Q1 causal linear rung inputs (g_abl / g_0 on the fixed axes); decided in r2_linearity.py")
    labels = {nm: list(O["labels_rgi"]) for nm in ("u_RGI_perp", "span", "u_diff")}
    match = _match_rate(refs_unprompted["conts"], rk.ST["ref"]["conts"])
    js = base_info("q3", dict(status="ran", D0=D0, Delta_T=ctx.Delta_T, g0=dict(RSI=O["refs"]["RSI"]["g0"], RGI=O["refs"]["RGI"]["g0"]),
                              Delta_RGI=O["Delta_RGI"], GA14=O["GA"], labels=labels, ablations=J, controls=dict(rand_14_2="null2", cov_14_0="null8", ctrl2="cov_14_1 + cov_14_2"),
                              ctrl_p5=ctrl_p5, rules=rules, c_q4a=dict(ablation=chosen, used=chosen or "u_RSI", rule="G43: u_RSI if it passes, else the span; u_RSI (descriptive) if none passed"),
                              linear_rung=rung, references=dict(unprompted_match_rate_R1=match, n_unprompted=len(refs_unprompted["conts"])),
                              r1_meta_gpu1=r1_meta_gpu1(ctx.r1), statement_counts=dict(H30=len(H30), RGI30=len(RGI)),
                              notes=["g on O1's fixed axes (r2_axes.pt fixed_axes, block 14); ratio draws: the same prompt weights for g and g_0",
                                     "z_D against the 16 one-dim nulls (1-dim) or the 8 pairs (2-dim); a control without itself (O6)"]))
    return write_outputs(ctx.r2, "q3", js, P)


def a_dim(A, nm): return A[dict(rand_14_2="null2", cov_14_0="null8").get(nm, nm)]["dim"]


def _ratio_view(t): return dict(ratio=t["ratio"], ratio_ci90=t["ratio_ci90"], g=t["g"], g0=t["g0"])


RUNG_NO_BASELINE = "not decidable (no baseline gap: g_0 <= 0)"                    # r2_linearity.linear_rung's string


def _rung(r1, r2, g0):
    """C-Q1's causal linear rung (in-session view; r2_linearity.py decides): not decidable unless g_0 > 0 (as everywhere in this script,
    review unlearn_a-2); 'one direction' if the 1-dim ablation leaves g_abl / g_0 <= 0.5; 'two-dimensional span' if it does not but the span
    ablation does; otherwise 'not removed within the tested 2-dim span'."""
    if not (fin(g0) and g0 > 0): return RUNG_NO_BASELINE
    if fin(r1) and r1 <= RULE["gap_point"]: return "one direction"
    if fin(r2) and r2 <= RULE["gap_point"]: return "two-dimensional span"
    return "not removed within the tested 2-dim span"


def _gen_unprompted(ctx, hooks):
    """64 unprompted texts with keys keyseed('ref', p) (rk.references' rows), generated under hooks, read by the unmodified model."""
    rk = ctx.rk; rows = [(rk.PIDS[p], rk.keyseed("ref", p), 1) for p in range(rk.NP)]
    conts, _ = ctx.generate(rows, None, 0.0, hooks); return dict(conts=conts, keys=[r[1] for r in rows], X=ctx.read(conts).float())


def _unprompted_shift(ctx, Tu, ref, O):
    """C-Q3 unprompted shift (reported, floor caveat): mean over prompts of (x_abl,p - x_ref,p) . unit(u_T_perp) in units of g_0 (and of
    Delta_T) with its 90 % prompt-bootstrap interval; the room above the anti level = (mean ref projection - mean RSI_ANTI projection) / g_0;
    the foreign moves of the unprompted texts (descriptive)."""
    Xa, Xr = Tu["X"].double(), ref["X"].double(); g0 = O["refs"]["RSI"]["g0"]; e = ctx.u_rsi
    per = (Xa - Xr) @ e; per[torch.isnan(per)] = float("nan"); draws = boot_g(per, ctx.W)
    room = (proj_mean(Xr, e) - proj_mean(O["refs"]["RSI"]["Xa"], e)) / g0 if g0 else float("nan")
    num = dict(shift_g0=float(np.nanmean(per.numpy())) / g0, shift_ci90_g0=list(R.interval(draws.numpy() / g0, LEVEL)),
               shift_Delta_T=float(np.nanmean(per.numpy())) / ctx.Delta_T, room_above_anti_g0=room, kept=int(torch.isfinite(per).sum()),
               m_foreign=foreign_moves(Xa, Xr, O["foreign_axes"], O["foreign_gaps"]), identical_to_ref=sum(a == b for a, b in zip(Tu["conts"], ref["conts"])))
    return dict(num=num, arr=dict(per_prompt=per.float(), draws=draws.float()))


def _match_rate(a_list, b_list):
    """Token-level match of regenerated texts with round-1 texts of the same keys (22k): identical share and common-prefix token share."""
    ident = [a == b for a, b in zip(a_list, b_list)]; num = den = 0
    for a, b in zip(a_list, b_list):
        k = R.first_divergence(a, b); num += max(len(a), len(b)) if k < 0 else k; den += max(len(a), len(b))
    return dict(n=len(ident), identical_share=float(np.mean(ident)) if ident else float("nan"), prefix_token_share=num / den if den else float("nan"))


# ======================================================================================================================================
# C-Q4a
# ======================================================================================================================================
def stage_q4a(ctx):
    rk = ctx.rk; q3_js, q3_pt = out_paths(ctx.r2, "q3")
    if not os.path.exists(q3_js): raise MissingInput(f"{q3_js} missing: run stage q3 first (same session)")
    J3 = json.load(open(q3_js))
    if J3.get("status") != "ran": return not_run(ctx.r2, "q4a", J3.get("status", NOT_RUN_R1), error="C-Q3 did not run")
    P3 = torch.load(q3_pt, map_location="cpu", weights_only=False); O = ctx.o2()
    passed = J3["c_q4a"]["ablation"]; use = passed or "u_RSI"; U = P3["ablations"][use]["U"].double()
    # cue messages (OD-C3): this session's RSI_PRO texts of prompts p+1, p+2, p+3 (mod NP), decoded
    pro_txt = [rk.tok.decode(c) for c in O["texts"]["RSI_PRO"]]; keys = [int(k) for k in O["keys"]["RSI_PRO"]]
    msgs = [S.cue_message(p, list(rk.PROMPTS), pro_txt) for p in range(rk.NP)]
    rows = [(rk.enc(rk.chat(m)), keys[p], 1.0) for p, m in enumerate(msgs)]
    def gen(hooks):
        def make():
            c = ctx.generate(rows, None, 0.0, hooks)[0]; return dict(conts=c, X=ctx.read(c).float())
        return make
    abl = ctx.cache.get("cued_ablated", sig_of("cue", keys, msgs, U), gen((R.WriteOrth(U),)))
    una = ctx.cache.get("cued_unablated", sig_of("cue", keys, msgs), gen(()))
    ref = O["refs"]["RSI"]; Xa_abl = P3["texts"][f"{use}|RSI"]["Xa"].double()
    G_cue = R.gap_on_fixed_axes(abl["X"], Xa_abl, ctx.fold, ref["a_A"], ref["a_B"])
    G_abl = R.gap_on_fixed_axes(P3["texts"][f"{use}|RSI"]["Xp"], Xa_abl, ctx.fold, ref["a_A"], ref["a_B"])
    G_ceil = R.gap_on_fixed_axes(una["X"], ref["Xa"], ctx.fold, ref["a_A"], ref["a_B"])
    W = ctx.W; d_cue, d_ceil = boot_g(G_cue["per_prompt"], W), boot_g(G_ceil["per_prompt"], W); g0 = ref["g0"]
    RS, share_d = restored_share(G_cue["per_prompt"], G_abl["per_prompt"], ref["per_prompt"], W); share, (lo, hi), defined = RS["share"], RS["ci90"], RS["defined"]
    den = g0 - G_abl["g"]; ceil_share = (G_ceil["g"] - G_abl["g"]) / den if defined else float("nan")
    try: own_cue = R.axis_from_texts(abl["X"], Xa_abl, ctx.fold, basis=ref["Q"], n_null=0)["gap_perp_cross"]
    except Exception: own_cue = float("nan")
    try: own_ceil = R.axis_from_texts(una["X"], ref["Xa"], ctx.fold, basis=ref["Q"], n_null=0)["gap_perp_cross"]   # review unlearn_a-3
    except Exception: own_ceil = float("nan")
    verdict = rule_cue(share, hi, passed is not None)
    lens = [len(r[0]) for r in rows]
    js = base_info("q4a", dict(status="ran", ablation=use, passed_C_Q3=passed is not None, g_cue=G_cue["g"], g_abl=G_abl["g"], g0=g0,
                               g_abl_q3=J3["ablations"][dict(rand_14_2="null2", cov_14_0="null8").get(use, use)]["texts"]["RSI"]["g"],
                               restored_share=share, restored_share_ci90=[lo, hi], share_defined=defined, own_axis_gap_cue=own_cue, ceiling=dict(g=G_ceil["g"], share=ceil_share, ratio_g0=G_ceil["g"] / g0 if g0 else float("nan"),
                               ci90_g=list(R.interval(d_ceil.numpy(), LEVEL)), own_axis_gap=own_ceil), kept=dict(cue=G_cue["kept"], abl=G_abl["kept"], ceiling=G_ceil["kept"]),
                               verdict_in_session=verdict, rule="shallow if share >= 0.5; deep if share < 0.2 with 90 % upper end < 0.35; otherwise partial; "
                               "'not applicable (nothing removed)' if no ablation passed C-Q3 (G43)", cue_prompt_tokens=dict(min=min(lens), max=max(lens), mean=float(np.mean(lens))),
                               keys="the O2 RSI_PRO keys (keyseed('axis', 'rsi', p))", empty=dict(ablated=sum(len(c) == 0 for c in abl["conts"]), unablated=sum(len(c) == 0 for c in una["conts"]))))
    P = dict(messages=msgs, keys=keys, ablated=dict(abl, per_prompt=G_cue["per_prompt"].float(), draws=d_cue.float()),
             unablated=dict(una, per_prompt=G_ceil["per_prompt"].float(), draws=d_ceil.float()), U=U.float(), share_draws=share_d.float())
    return write_outputs(ctx.r2, "q4a", js, P)


# ======================================================================================================================================
# C-Q2
# ======================================================================================================================================
Q2_PUSH = (("u_RSI", "RSI"), ("u_diff", "RSI"), ("u_RGI_perp", "RGI"), ("cov_14_3", "RSI"), ("cov_14_3", "RGI"))
Q2_RULE = (("u_RSI", "RSI", "H30"), ("u_diff", "RSI", "H30"), ("u_RGI_perp", "RGI", "RGI30"))


def load_r1_k6(ctx):
    """R1.k6 (k6.json, found by name; G6). If the file is missing, R1.state[14]['k6'] (the same dict, written by rk.k6 before k6.json) is
    used and flagged; None if neither exists."""
    try: return R.load_r1_json(ctx.r1, f"rk/b{BLOCK}/k6.json", BLOCK), "k6.json"
    except R.R1Missing:
        k6 = ctx.rk.ST.get("k6"); return (k6, "R1.state[14]['k6'] (k6.json missing)") if k6 is not None else (None, "missing")


Q2_STATE_KEY, Q2_STATE_PREFIXES = "r2_q2_valid", ("k6_r2_", "r2_b05_")


def q2_state_purge(ST, sig):
    """Review unlearn_a-1: q2's round-2 state entries (k6_r2_*, r2_b05_*; only q2 writes them, and they survive a new VM with the synced
    state) count only for the code fingerprint and inputs recorded in ST['r2_q2_valid']; if that entry is missing or differs, every such
    entry is deleted so that it is recomputed (the guard protects round-1 keys only). Returns the deleted keys."""
    if ST.get(Q2_STATE_KEY) == sig: return []
    gone = sorted(k for k in ST if isinstance(k, str) and k.startswith(Q2_STATE_PREFIXES))
    for k in gone: del ST[k]
    return gone


def stage_q2(ctx):
    rk = ctx.rk; O = ctx.o2(); ST = rk.ST
    if O["B"] is None: return not_run(ctx.r2, "q2", NOT_RUN_R1, error=f"O2 at block 14: {O['BJ'].get('status', 'missing')}")
    H30, RGI = ctx.H30(), ctx.RGI30(); KL = ctx.refkl(); D = rk.D
    dirs = dict(u_RSI=ctx.u_rsi, u_diff=O["u_diff"], u_RGI_perp=O["u_RGI_perp"], u_com=O["u_com"], cov_14_3=R.pool_dir("cov", BLOCK, 3, D=D))
    k6, k6_src = load_r1_k6(ctx); flags = [] if k6_src == "k6.json" else [f"R1.k6 from {k6_src}"]
    nulls = R.k6_null_dirs(BLOCK, D=D); nn = len(nulls)
    vsig = sig_of("q2-state", code_sha256(), *[dirs[k] for k in sorted(dirs)], *nulls, [x["question"] for x in H30 + RGI], ctx.alpha)
    purged = q2_state_purge(ST, vsig)                                            # before rk.k6_direction can return a stale entry
    if ST.get(Q2_STATE_KEY) != vsig: R.st_put(Q2_STATE_KEY, vsig)
    J = dict(K6={}, alpha_u={}, texts={}, behaviour={}, content={}, perplexity={}, rules={}); P = dict(texts={}, content={}, perplexity={})
    # ---- (a) K6 at 0.03 nats (descriptive)
    if k6 is not None:
        reg1 = reg_from_r1_k6(k6); M0 = float(k6["M0"])
        for nm in ("u_RGI_perp", "u_com", "u_diff"):
            e = dict(R.k6_direction(f"r2_{nm}", dirs[nm], H30, M0)); e["z_fin"] = z_fin(e["Ue_fin"], e["A_fin"], reg1); J["K6"][f"{nm}|H30"] = e
        J["K6"]["u_RSI|H30 (R1.k6 tests['u_T_perp'])"] = k6["tests"].get("u_T_perp")
        J["K6"]["regression_H30"] = reg1; J["K6"]["M0_H30"] = M0
    else: J["K6"]["status"] = NOT_RUN_R1 + " (R1.k6)"
    lo0, _ = ctx.cache.get("rgi30_base", sig_of("rgi30", [x["question"] for x in RGI]), lambda: ctx.scores(RGI)); M0_RGI = float(np.mean(lo0)); J["K6"]["M0_RGI30"] = M0_RGI
    fR1 = [ST.get(f"k6_null{i}", {}).get("f") for i in range(nn)]
    if all(fin(f) for f in fR1):
        rn = [ctx.cache.get(f"rgi30_null{i}", sig_of("rgi30null", nulls[i], fR1[i]), lambda i=i: ctx.behaviour(RGI, nulls[i], fR1[i])) for i in range(nn)]
        regG = ols_nulls(rn); J["K6"]["regression_RGI30"] = regG; J["K6"]["nulls_RGI30"] = [{k: v for k, v in x.items() if k in ("f", "Ue_fin", "A_fin")} for x in rn]
        for nm in ("u_RGI_perp", "u_com"):
            e = dict(R.k6_direction(f"r2_{nm}_RGI30", dirs[nm], RGI, M0_RGI)); e["z_fin"] = z_fin(e["Ue_fin"], e["A_fin"], regG); J["K6"][f"{nm}|RGI30"] = e
    else: J["K6"]["RGI30_status"] = NOT_RUN_R1 + " (R1 'k6_null<i>' push sizes)"
    log("q2: K6 at 0.03 nats done")
    # ---- (b) alpha_u at 0.05 nats (calibrated on the counter-push -alpha_u unit(u)); counter-pushed instructed texts
    for nm in ("u_RSI", "u_diff", "u_RGI_perp", "cov_14_3"):
        cal = ctx.cache.get(f"alpha_{nm}", sig_of("cal05-", dirs[nm], ctx.alpha), lambda nm=nm: calibrate(lambda f: KL.push(dirs[nm], f, -1.0), ctx.alpha * CAL05["start_factor"], CAL05))
        cal = dict(cal, damage_plus=ctx.cache.get(f"dplus_{nm}", sig_of("dplus", dirs[nm], cal["f"]), lambda nm=nm, f=cal["f"]: KL.push(dirs[nm], f, 1.0)))
        J["alpha_u"][nm] = cal
    for nm, c in Q2_PUSH:
        T = ctx.instructed_texts(f"push_{nm}", c, d=dirs[nm], alpha=J["alpha_u"][nm]["f"], sign=-1.0)
        num, arr = gap_entry(T["Xp"], T["Xa"], O["refs"][c], ctx.W, O["foreign_axes"], O["foreign_gaps"])
        J["texts"][f"{nm}|{c}"] = dict(num, capped_share=T["capped_share"]); P["texts"][f"{nm}|{c}"] = dict(T, **arr)
    log("q2: counter-pushed instructed texts done")
    # ---- (b') behaviour at alpha_u against the null directions recalibrated to 0.05 nats (state entries r2_b05_*)
    b05 = {}
    for i in range(nn):
        key = f"r2_b05_null{i}"
        if key not in ST:
            cal = calibrate(lambda f, i=i: KL.push(nulls[i], f, 1.0), ctx.alpha * CAL05["start_factor"], CAL05)
            R.st_put(key, dict(cal, H30=ctx.behaviour(H30, nulls[i], cal["f"]), RGI30=ctx.behaviour(RGI, nulls[i], cal["f"])), do_save=False)
        b05[key] = ST[key]
    R.save(); reg05 = {s: ols_nulls([b05[f"r2_b05_null{i}"][s] for i in range(nn)]) for s in ("H30", "RGI30")}
    for nm, c, s in Q2_RULE:
        key = f"r2_b05_{nm}"; f = J["alpha_u"][nm]["f"]
        if key not in ST or abs(ST[key]["f"] - f) > 1e-12 * max(1.0, abs(f)):
            beh = ctx.behaviour(H30 if s == "H30" else RGI, dirs[nm], f); R.st_put(key, dict(f=f, statements=s, **{s: beh}), do_save=False)
        beh = ST[key][s]; J["behaviour"][nm] = dict(statements=s, f=f, Ue_fin=beh["Ue_fin"], A_fin=beh["A_fin"], Mp=beh["Mp"], Mm=beh["Mm"],
                                                     z_fin=z_fin(beh["Ue_fin"], beh["A_fin"], reg05[s]), M0=(float(k6["M0"]) if (s == "H30" and k6 is not None) else (M0_RGI if s == "RGI30" else float("nan"))))
    R.save(); J["behaviour"]["regression"] = reg05; J["behaviour"]["nulls"] = {k: dict(f=v["f"], damage=v["damage"], in_band=v["in_band"], H30=dict(Ue_fin=v["H30"]["Ue_fin"], A_fin=v["H30"]["A_fin"]),
                                                                                       RGI30=dict(Ue_fin=v["RGI30"]["Ue_fin"], A_fin=v["RGI30"]["A_fin"])) for k, v in b05.items()}
    log("q2: behaviour at alpha_u done")
    # ---- (c) unprompted content: +- alpha_u pairs on the RSI-topic prompts; perplexity of the scene references under +- the push
    rsi_p, scene_p = list(range(ctx.NRSI)), list(range(ctx.NRSI, ctx.NP)); Wr = ctx.W[:, : ctx.NRSI]; refX = ST["ref"]["X"].double()
    base_nll = ctx.cache.get("nll_base", sig_of("nll", scene_p), lambda: _ref_nll(ctx, scene_p, None))
    for nm in ("u_RSI", "u_diff", "u_RGI_perp", "cov_14_3"):
        f = J["alpha_u"][nm]["f"]
        ev = ctx.cache.get(f"content_{nm}", sig_of("cnt", nm, dirs[nm], f),
                           lambda nm=nm, f=f: _ev_keep(R.evaluate_variant(dirs[nm], f, lambda p, s, nm=nm: rk.keyseed("r2CNT", nm, p), nseed=1, prompts=rsi_p, cbar=False)))
        J["content"][nm] = _content(ctx, ev, refX, O, Wr); P["content"][nm] = ev
        pp = {sg: ctx.cache.get(f"nll_{nm}_{sg}", sig_of("nll", nm, dirs[nm], f, sg, scene_p), lambda nm=nm, f=f, sg=sg: _ref_nll(ctx, scene_p, (dirs[nm], f, 1.0 if sg == "plus" else -1.0))) for sg in ("plus", "minus")}
        J["perplexity"][nm] = {sg: dict(ppl=v["ppl"], mean_nll=v["mean_nll"]) for sg, v in pp.items()}; P["perplexity"][nm] = pp
    J["perplexity"]["unpushed"] = dict(ppl=base_nll["ppl"], mean_nll=base_nll["mean_nll"]); P["perplexity"]["unpushed"] = base_nll
    # ---- rule inputs ('suppressed'; decided in r2_ana)
    for nm, c, s in Q2_RULE:
        t = J["texts"][f"{nm}|{c}"]; ctrl = pct(P["texts"][f"cov_14_3|{c}"]["ratio_draws"].numpy(), 5); b = J["behaviour"][nm]
        M0 = b["M0"]; dmg = J["alpha_u"][nm]["damage"]
        J["rules"][nm] = dict(concept=c, statements=s, ctrl_p5=ctrl, **rule_suppressed(t["g"], t["g0"], t["ratio_ci90"][1], ctrl, b["Ue_fin"], b["z_fin"], M0, dmg,
                                                                                     t["m_foreign"], t["m_concept"]))
    labels = {nm: list(O["labels_rgi"]) for nm in ("u_RGI_perp", "u_diff", "u_com")}
    js = base_info("q2", dict(status="ran", flags=flags, state_entries_purged=purged, GA14=O["GA"], labels=labels, g0=dict(RSI=O["refs"]["RSI"]["g0"], RGI=O["refs"]["RGI"]["g0"]),
                              Delta_RGI=O["Delta_RGI"], Delta_T=ctx.Delta_T, n_nulls=nn, statement_counts=dict(H30=len(H30), RGI30=len(RGI)), **J,
                              notes=["alpha_u calibrated on the counter-push -alpha_u unit(u) (the push of rule (iii)); damage_plus = damage of +alpha_u unit(u)",
                                     "null directions recalibrated to 0.05 nats on +d (rk.set_alpha's rule, start alpha_14), evaluated at +-f on H30 and RGI30",
                                     "content pairs: RSI-topic prompts, keys keyseed('r2CNT', direction, p); counter side alone against the R1 reference texts"]))
    return write_outputs(ctx.r2, "q2", js, P)


def _ev_keep(ev):
    keep = ("delta", "ok", "xp", "xm", "prompt_of", "first_div", "conts", "alpha", "keys", "signs", "capped_share", "dropped", "n_pairs")
    return {k: ev[k] for k in keep if k in ev}


def _content(ctx, ev, refX, O, Wr):
    """C-Q2 (c), descriptive: RSI / RGI content of the +- pairs (delta projected on unit(u_T_perp) / g_0 and on unit(u_RGI_perp) / Delta_RGI,
    90 % bootstrap over the RSI-topic prompts), the counter side alone (s0 - s-) against the R1 reference texts of the same prompts, sigma
    (rk.stats's formula on u_T and on u_RGI_perp), foreign axes (delta on unit(u_X) / gap_cross_X)."""
    d, ok, po = ev["delta"].double(), torch.as_tensor(ev["ok"], dtype=torch.bool), torch.as_tensor(ev["prompt_of"])
    xp, xm = ev["xp"].double(), ev["xm"].double(); X0 = refX[po.unique()]; X0 = X0[~torch.isnan(X0).any(1)]
    out = {}
    for nm, v, unit_ in (("RSI", ctx.uT, O["refs"]["RSI"]["g0"]), ("RGI", O["u_RGI_perp"], O["Delta_RGI"])):
        e = R.unit(v.double()); per = d @ e; dr = R.boot_mean(per, po, Wr, ok).numpy(); s0 = float((X0 @ e).mean())
        sp, sm = float((xp[ok] @ e).mean()), float((xm[ok] @ e).mean())
        out[nm] = dict(content=float(per[ok].mean()) / unit_, content_ci90=list(R.interval(dr / unit_, LEVEL)), counter_side=(s0 - sm) / unit_,
                       sigma=(sp - s0) / (s0 - sm) if abs(s0 - sm) > 1e-9 else float("nan"), unit=unit_)
    uTr = ctx.ax["u_T"].double(); s0 = float((X0 @ uTr).mean()); sp, sm = float((xp[ok] @ uTr).mean()), float((xm[ok] @ uTr).mean())
    out["sigma_u_T"] = (sp - s0) / (s0 - sm) if abs(s0 - sm) > 1e-9 else float("nan")
    out["foreign"] = {nm: float((d[ok] @ R.unit(torch.as_tensor(v).double())).mean()) / _f(O["foreign_gaps"][nm]) if _f(O["foreign_gaps"][nm]) > 0 else float("nan")
                      for nm, v in O["foreign_axes"].items()}
    out.update(n_pairs=int(len(d)), dropped=int((~ok).sum()), capped_share=ev.get("capped_share"))
    return out


def _ref_nll(ctx, prompts, push):
    """Teacher-forced NLL per token of the R1 reference continuations of the given prompts, unpushed or under the push (d, f, sign) from
    the first user token on; ppl = exp(total NLL / total tokens)."""
    rk = ctx.rk; per = []
    for p in prompts:
        c = rk.ST["ref"]["conts"][p]
        if not len(c): per.append(None); continue
        seq = rk.PIDS[p] + c; P0 = len(rk.PIDS[p]); ids = torch.tensor([seq]).to(rk.DEV); pos = torch.arange(P0 - 1, len(seq) - 1).to(rk.DEV)
        if push is not None:
            d, f, sg = push; rk.STEER.update(v=R.unit(torch.as_tensor(d).float()), V=None, alpha=f, cap=rk.ST["ref"]["cap"], sign=torch.tensor([float(sg)]), rowstart=torch.tensor([rk.START]))
        try:
            with torch.no_grad(): lp = torch.log_softmax(rk.model(ids).logits[0][pos].double(), -1)
        finally: rk.steer_off()
        per.append((-lp[torch.arange(len(c), device=lp.device), torch.tensor(c, device=lp.device)]).cpu())
    tot = sum(float(x.sum()) for x in per if x is not None); n = sum(len(x) for x in per if x is not None)
    return dict(ppl=math.exp(tot / n) if n else float("nan"), mean_nll=tot / n if n else float("nan"), per_text=[float(x.mean()) if x is not None else None for x in per], n_tokens=n)


# ======================================================================================================================================
# Loops: C-Q4b, arm A (64 pairs per round, 1 seed per prompt, keys (namespace, n, p, 0, 14))
# ======================================================================================================================================
def loop_round(ctx, tag, ns, n, push, alpha, hooks=(), hook_sig=""):
    """One loop round: 64 pairs pushed +- alpha min(|h_t|, cap_b) unit(push) from the first user token on (rk's steer hook), keys
    keyseed(ns, n, p, 0, 14) shared by the + and - text of a pair (and by the twin and the ablated loop), hooks during generation only,
    read by the unmodified model (r2_rk.evaluate_variant; cached unit)."""
    rk = ctx.rk; sg = sig_of(tag, ns, n, push, alpha, hook_sig)
    make = lambda: _ev_keep(R.evaluate_variant(push, alpha, lambda p, s: rk.keyseed(ns, n, p, 0, BLOCK), nseed=1, hooks=hooks, cbar=False))
    return ctx.cache.get(f"{tag}_{n}_{sg[:12]}", sg, make)


def round_stats(ctx, ev, push, damage):
    """Read-outs of the derived direction d_(n+1) on u_T_perp with R1's 2,000 null axes and the loop's own rel (rk.stats's formulas),
    Ue = unit(u_T_perp) . d / Delta_T with its 90 % prompt-bootstrap interval, cos(d, u_RSI), cos(push, u_RSI), the push's damage."""
    s, d = R.stats_on_axis(ev, ctx.uT, ctx.null_uT, n_prompts=ctx.NP); e = ctx.u_rsi
    dr = (R.boot_mean(ev["delta"], ev["prompt_of"], ctx.W, ev["ok"]) @ e).numpy() / ctx.Delta_T
    fd = [k for k in ev["first_div"] if k >= 0]
    s.update(Ue=float(e @ d) / ctx.Delta_T, Ue_ci90=list(R.interval(dr, LEVEL)), cos_u_RSI=R.cosf(d, e), cos_push_u_RSI=R.cosf(push.double(), e),
             damage_of_push=damage, dropped=int(ev["dropped"]), capped_share=ev.get("capped_share"), identical_pairs=int(sum(k == -1 for k in ev["first_div"])),
             first_div_median=float(np.median(fd)) if fd else None)
    return s, d


def run_loop_variant(ctx, tag, ns, start, rounds, alpha, hooks=(), hook_sig="", orth=None, first=None):
    """d_0 = start; round n pushes unit(d_n) (orthogonalised against unit(orth) if given, 2.7 C14); d_(n+1) = mean over ok pairs of delta_j.
    first: an already computed round-0 evaluation (the twin start check) reused if its push equals round 0's. Returns (rows, evs, d)."""
    KL = ctx.refkl(); d = start.double(); rows, evs, dirs, pushes = [], [], {}, {}
    for n in range(rounds):
        push = d if orth is None else d - (d @ R.unit(orth)) * R.unit(orth); push = R.unit(push)
        ev = first if (n == 0 and first is not None) else loop_round(ctx, tag, ns, n, push, alpha, hooks, hook_sig)
        dmg = ctx.cache.get(f"{tag}_dmg_{n}", sig_of("dmg", push, alpha), lambda push=push: KL.push(push, alpha, 1.0))
        s, dn = round_stats(ctx, ev, push, dmg); s.update(n=n, d=n + 1, norm_d=float(dn.norm())); rows.append(s); evs.append(ev)
        dirs[n + 1] = dn.float(); pushes[n] = push.float(); d = dn
    return rows, evs, dict(d=dirs, push=pushes)


def _rows_by_d(rows): return {r["d"]: dict(c=r["c"], p=r["p"], A=r["A"], rel=r["rel"], Ue=r["Ue"], Ue_ci90=r["Ue_ci90"]) for r in rows}


def stage_q4b(ctx):
    rk = ctx.rk; ST = rk.ST; suf = rk.SUF; u = ctx.u_rsi; alpha = ctx.alpha
    try:
        summ = R.load_r1_json(ctx.r1, f"summary_b{BLOCK}.json", BLOCK); loops = R.load_r1_json(ctx.r1, f"rk/b{BLOCK}/loops.json", BLOCK)
    except R.R1Missing as e: return not_run(ctx.r2, "q4b", NOT_RUN_R1, error=str(e))
    key_d1 = f"dir_RSI_a{suf}_1"
    if key_d1 not in ST: return not_run(ctx.r2, "q4b", NOT_RUN_R1, error=f"R1.state[14] has no {key_d1}")
    v = summ.get("verdict", {}); V = v.get("verdict"); label = bool(v.get("label_attractor")); failed = list(v.get("gates", {}).get("failed", []) or [])
    G13, wh = likely_wh(V); rnd = {L: loops[L][3] for L in ("RND_1", "RND_2") if L in loops and len(loops[L]) >= 4}; att = attracted_loop(label, rnd)
    d1 = ST[key_d1].double(); s_likely = R.unit(d1 - (d1 @ u) * u)
    r1info = dict(V_RSI=V, LIKELY_wh=G13, from_would_have_read=wh, label_attractor=label, attracted_loop=att, gates_failed=failed,
                  rnd_last_rows={L: dict(p=r.get("p"), A=r.get("A")) for L, r in rnd.items()}, d1_RSI_a_key=key_d1, cos_d1_u_RSI=R.cosf(d1, u))
    G15, twin0, t0s = None, None, None
    if G13:                                                      # G15: the twin's first round from the orthogonal start s0
        twin0 = loop_round(ctx, "twin", "r2Q4B", 0, s_likely, alpha)
        t0s, _ = round_stats(ctx, twin0, s_likely, ctx.cache.get("twin_dmg_start", sig_of("dmg", s_likely, alpha), lambda: ctx.refkl().push(s_likely, alpha, 1.0)))
        G15 = bool(fin(t0s["c"]) and t0s["c"] > 0 and fin(t0s["p"]) and t0s["p"] <= RULE["p_sig"])
    gate = q4b_gate(G13, bool(G15), att); gate.update(G13=G13, G15=G15, G14=att is not None)
    labels = ["descriptive (round-1 gate failed)"] if (gate.get("branch") == "likely" and wh) else []
    if not gate["open"]:
        js = base_info("q4b", dict(status=gate["verdict"], gate=gate, r1=r1info, twin_start_check=t0s, labels=labels))
        return write_outputs(ctx.r2, "q4b", js, dict(twin_start_check=dict(ev=twin0, start=s_likely.float())) if twin0 is not None else None)
    if gate["branch"] == "likely": s0, first = s_likely, twin0
    else:
        st = R.pool_dir("rand", BLOCK, 0 if att == "RND_1" else 1, D=rk.D); s0, first = R.unit(st - (st @ u) * u), None
    Wo = R.WriteOrth(u); hs = sig_of("WriteOrth", u)
    twin_rows, twin_evs, twin_d = run_loop_variant(ctx, "twin", "r2Q4B", s0, 3, alpha, orth=u, first=first)
    abl_rows, abl_evs, abl_d = run_loop_variant(ctx, "ablated", "r2Q4B", s0, 3, alpha, hooks=(Wo,), hook_sig=hs, orth=u)
    verdict = rule_loop(_rows_by_d(abl_rows), twin=_rows_by_d(twin_rows))
    js = base_info("q4b", dict(status="ran", gate=gate, r1=r1info, twin_start_check=t0s, labels=labels, start=dict(branch=gate["branch"], cos_s0_u_RSI=R.cosf(s0, u)),
                               ablated=abl_rows, twin=twin_rows, verdict_in_session=verdict, Delta_T=ctx.Delta_T,
                               note="rows n = 0..2 hold the read-outs of d_1..d_3 (d = n + 1); the rule reads d_2 and d_3; the matched-size reference (seed-0 half of R1 RSI_a) is r2_ana's (R1-6)"))
    P = dict(start=s0.float(), twin=dict(evs=twin_evs, **twin_d), ablated=dict(evs=abl_evs, **abl_d), twin_start_check=dict(ev=twin0, start=s_likely.float()) if twin0 is not None else None)
    return write_outputs(ctx.r2, "q4b", js, P)


def ref_seed0_c(ctx, ev):
    """G16's reference (Amendment 3 (a)): c(d_1) of the seed-0 half of R1 RSI_a's evaluation 0; the 90 % prompt-bootstrap percentile interval
    of its replicates (the session's draws) shifted by c - (bootstrap median) is 'ci90'; the unshifted interval ('ci90_percentile') and the
    median ('boot_median') are kept next to it."""
    r0 = seed0_rows(ev, ctx.NP); dl, ok, po = ev["delta"][r0].double(), torch.as_tensor(ev["ok"])[r0], torch.as_tensor(ev["prompt_of"])[r0]
    d = dl[ok].mean(0); Db = R.boot_mean(dl, po, ctx.W, ok); e = ctx.u_rsi
    cb = ((Db @ e) / Db.norm(dim=1)).numpy(); c = R.cosf(d, e)
    return dict(c=c, n_pairs=int(ok.sum()), **centred_interval(c, cb)), cb


def stage_armA(ctx):
    rk = ctx.rk; ST = rk.ST; key = f"ev_RSI_a{rk.SUF}_0"; u = ctx.u_rsi; alpha = ctx.alpha
    if key not in ST: return not_run(ctx.r2, "armA", NOT_RUN_R1, error=f"R1.state[14] has no {key}")
    ref, cb = ref_seed0_c(ctx, ST[key]); KL = ctx.refkl()
    beta = ctx.cache.get("beta", sig_of("cal03-", u, ctx.alpha), lambda: calibrate(lambda f: KL.push(u, f, -1.0), ctx.alpha * CAL03["start_factor"], CAL03))
    hook = ConstPush(rk, u, beta["f"], -1.0); hs = sig_of("ConstPush", u, beta["f"], -1.0)
    rows, evs, dd = run_loop_variant(ctx, "armA", "r2ARMA", ctx.uT, 1, alpha, hooks=(hook,), hook_sig=hs)
    c1 = rows[0]["c"]; cont = arma_continues(c1, ref["ci90"][0])
    ref = dict(ref, continues_with_percentile_interval=arma_continues(c1, ref["ci90_percentile"][0]),
               amendment="Amendment 3 (a): ci90 = the percentile interval shifted by c - boot_median; G16 uses ci90[0]")
    if cont: rows, evs, dd = run_loop_variant(ctx, "armA", "r2ARMA", ctx.uT, 3, alpha, hooks=(hook,), hook_sig=hs, first=evs[0])
    verdict = "translation (as predicted), not un-searched" if not cont else "continued (rounds 2-3 ran; the C-Q4b loop rule with the R1 seed-0 comparator is applied in r2_ana)"
    js = base_info("armA", dict(status="ran", beta=beta, reference_seed0_RSI_a=dict(ref, key=key), c1=c1, continues=cont, rounds=len(rows), rows=rows,
                                verdict_in_session=verdict, Delta_T=ctx.Delta_T,
                                note="push +- alpha_14 unit(d_n) (d_0 = u_T_perp) plus the constant -beta min(|h_t|, cap_b) unit(u_RSI) on both rows (ConstPush)"))
    return write_outputs(ctx.r2, "armA", js, dict(evs=evs, ref_c_draws=torch.as_tensor(cb).float(), **dd))


# ======================================================================================================================================
# C-Q5 (pushes and the in-session view of the superposition / dose numbers; SVD and +- symmetry are r2_ana's, R1-5)
# ======================================================================================================================================
def subset_ev(ev, rows):
    rows = torch.as_tensor(rows); return dict(delta=ev["delta"][rows], ok=torch.as_tensor(ev["ok"])[rows], prompt_of=torch.as_tensor(ev["prompt_of"])[rows])


def stage_q5(ctx):
    rk = ctx.rk; ST = rk.ST; suf = rk.SUF; k_rsi, k_sea = f"ev_RSI_a{suf}_0", f"ev_SEA{suf}_0"
    miss = [k for k in (k_rsi, k_sea) if k not in ST]
    if miss: return not_run(ctx.r2, "q5", NOT_RUN_R1, error=f"R1.state[14] has no {miss}")
    sea = ST["axes"]["sea"].double(); mix = R.unit(R.unit(ctx.uT) + R.unit(sea)); alpha = ctx.alpha; NP = ctx.NP
    ev_mix = ctx.cache.get("mix", sig_of("mix", mix, alpha), lambda: _ev_keep(R.evaluate_variant(mix, alpha, lambda p, s: rk.keyseed("r2SUP", p), nseed=1, cbar=False)))
    ev_dose = ctx.cache.get("dose", sig_of("dose", ctx.u_rsi, 2 * alpha), lambda: _ev_keep(R.evaluate_variant(ctx.u_rsi, 2 * alpha, lambda p, s: rk.keyseed("r2DOSE", p), nseed=1, cbar=False)))
    rsi0 = subset_ev(ST[k_rsi], seed0_rows(ST[k_rsi], NP)); evs = ST[k_sea]; sea1 = subset_ev(evs, torch.arange(len(evs["prompt_of"])))
    F_mix, F_rsi, F_sea = pair_mean(ev_mix), pair_mean(rsi0), pair_mean(sea1)
    sup, F_sum = superposition_stats(F_mix, F_rsi, F_sea, ctx.uT, sea)
    rel_mix = rel_split(lambda m: pair_mean(ev_mix, m), NP); rel_sum = rel_split(lambda m: (pair_mean(rsi0, m) + pair_mean(sea1, m)) / math.sqrt(2), NP)
    dis = sup["cos"] / math.sqrt(rel_mix["rel"] * rel_sum["rel"]) if (fin(rel_mix["rel"]) and fin(rel_sum["rel"])) else float("nan")
    W = ctx.W; Bm, Br, Bs = (R.boot_mean(e["delta"], e["prompt_of"], W, e["ok"]) for e in (ev_mix, rsi0, sea1))
    E, _ = torch.linalg.qr(torch.stack([R.unit(ctx.uT), R.unit(sea)], 1)); cm, cr, cs = Bm @ E, Br @ E, Bs @ E
    A = torch.stack([cr, cs], 2); coef = torch.linalg.lstsq(A, cm[..., None]).solution[..., 0]
    Bsum = (Br + Bs) / math.sqrt(2); cosb = ((Bm * Bsum).sum(1) / (Bm.norm(dim=1) * Bsum.norm(dim=1))).numpy()
    lo_c, hi_c = R.interval(cosb, LEVEL); med_c = pct(cosb, 50)                    # Amendment 3 (a): the bootstrap median is reported too
    sup_view = bool(fin(dis) and dis >= RULE["sup_cos"] and all(RULE["sup_coef"][0] <= sup[k] <= RULE["sup_coef"][1] for k in ("coef_RSI", "coef_SEA")))
    # dose: |F(2 alpha)| / |F(alpha)| (U-statistic norms) and debiased cos
    n1, n2 = u_norm2(rsi0["delta"], rsi0["ok"]), u_norm2(ev_dose["delta"], torch.as_tensor(ev_dose["ok"]))
    F2 = pair_mean(ev_dose); cross = float(F2 @ F_rsi)
    ratio = math.sqrt(n2 / n1) if (fin(n1) and fin(n2) and n1 > 0 and n2 >= 0) else float("nan")
    cosd = cross / math.sqrt(n1 * n2) if (fin(n1) and fin(n2) and n1 > 0 and n2 > 0) else float("nan")
    rb, cb = _dose_boot(rsi0, ev_dose, W)
    js = base_info("q5", dict(status="ran", mix=dict(cos_mix_u_T_perp=R.cosf(mix, ctx.uT), cos_mix_sea=R.cosf(mix, sea), cos_u_T_perp_sea=R.cosf(ctx.uT, sea)),
                              superposition=dict(sup, cos_disatt=dis, rel_mix=rel_mix, rel_sum=rel_sum, cos_ci90=[lo_c, hi_c],
                                                 coef_RSI_ci90=list(R.interval(coef[:, 0].numpy(), LEVEL)), coef_SEA_ci90=list(R.interval(coef[:, 1].numpy(), LEVEL)),
                                                 cos_disatt_ci90=[x / math.sqrt(rel_mix["rel"] * rel_sum["rel"]) for x in (lo_c, hi_c)] if fin(dis) else [float("nan")] * 2,
                                                 cos_boot_median=med_c, cos_disatt_boot_median=med_c / math.sqrt(rel_mix["rel"] * rel_sum["rel"]) if fin(dis) else float("nan"),
                                                 holds_in_session=sup_view, coef_band=list(RULE["sup_coef"])),
                              dose=dict(norm_ratio=ratio, cos_debiased=cosd, norm2_alpha=n1, norm2_2alpha=n2, ratio_ci90=list(R.interval(rb, LEVEL)), cos_ci90=list(R.interval(cb, LEVEL)),
                                        cos_boot_median=pct(cb, 50)),
                              references=dict(RSI=f"{k_rsi} seed-0 half ({int(rsi0['ok'].sum())} ok pairs)", SEA=f"{k_sea} ({int(sea1['ok'].sum())} ok pairs)"),
                              read_outs=dict(mix=round_stats(ctx, ev_mix, mix, float("nan"))[0], dose=round_stats(ctx, ev_dose, ctx.u_rsi, float("nan"))[0]),
                              note="rel_mix, rel_sum from the K2 halvings (rk.stats's formula); disattenuated cos interval with the rels fixed at their point estimates; "
                                   "the SVD of the stacked d_n and the +- symmetry (R1 sigma) are computed in r2_ana (R1-5)"))
    P = dict(mix=ev_mix, dose=ev_dose, mix_dir=mix.float(), F_mix=F_mix.float(), F_RSI=F_rsi.float(), F_SEA=F_sea.float(), F_sum=F_sum.float(), F_2alpha=F2.float(),
             boot=dict(cos=torch.as_tensor(cosb).float(), coef=coef.float(), dose_ratio=torch.as_tensor(rb).float(), dose_cos=torch.as_tensor(cb).float()))
    return write_outputs(ctx.r2, "q5", js, P)


def _dose_boot(ev1, ev2, W):
    """Prompt-bootstrap replicates of the dose ratio sqrt(U2 / U1) and of the debiased cos F2 . F1 / sqrt(U1 U2) (U = u_norm2 on the
    replicate's weighted pairs, distinct pairs only; the same prompt weights for both evaluations)."""
    out = []
    for ev in (ev1, ev2):
        d = ev["delta"].double(); ok = torch.as_tensor(ev["ok"], dtype=torch.bool); w = torch.as_tensor(R.pair_weights(W, ev["prompt_of"], ok))
        d = torch.where(ok[:, None], d, torch.zeros_like(d)); s = w @ d; n = w.sum(1)
        U = ((s * s).sum(1) - (w * w) @ (d * d).sum(1)) / (n * n - (w * w).sum(1)); out.append((s / n[:, None], U))
    (m1, U1), (m2, U2) = out
    ratio = torch.sqrt(U2 / U1); ratio[(U1 <= 0) | (U2 < 0)] = float("nan")
    cosd = (m1 * m2).sum(1) / torch.sqrt(U1 * U2); cosd[(U1 <= 0) | (U2 <= 0)] = float("nan")
    return ratio.numpy(), cosd.numpy()


# ======================================================================================================================================
# C-Q6 nesting primitives and depth cap
# ======================================================================================================================================
def way_it_starts(ids, enc):
    """Start indices of every ' the way it' run in ids, and of ' The way it' (the repetition texts write it after a full stop)."""
    pats = [enc(" the way it"), enc(" The way it")]
    return sorted(s for w in pats for s in range(len(ids) - len(w) + 1) if ids[s:s + len(w)] == w)


def q6_items(enc):
    """rsi_direction.py's Q2' stimuli (r2_stimuli copies): relations improves / describes / repetition x depths 1-4 x 24 objects (288),
    plain text without a chat template, token ids as rsi_direction.py's enc; the patch start of depth >= 3 = the first token of the second
    'the way it' (OD-C4)."""
    items = []
    for rel in S.NEST_RELS:
        for k in S.NEST_DEPTHS:
            for o, obj in enumerate(S.NEST_OBJ):
                t = S.nesting_text(rel, k, obj); ids = enc(t); ws = way_it_starts(ids, enc)
                assert len(ws) == k - 1, f"{t!r}: {len(ws)} 'the way it' runs, expected {k - 1}"
                items.append(dict(rel=rel, k=k, obj=o, text=t, ids=ids, way_it=ws, start=ws[1] if k >= 3 else None))
    return items


class DepthPatch:
    """C-Q6 cap at block 14 (OD-C4): at every position from 'starts' (per row) to the last token, the projection of the block output on the
    top-r depth modes V [D, r] is replaced by that of m2: x <- x + V V^T (m2 - x). Control (Rr [B, D, r], one random orthonormal r-dim
    subspace per row): x <- x + P_R (m2 - x) rescaled per position to the norm of the mode patch ('random r-dim patches of equal norm').
    Registered after rk's hooks; capture hooks registered later see the patched output."""
    def __init__(self, rk, starts, V, m2, Rr=None):
        self.rk = rk; dev = rk.DEV; dt = next(rk.model.parameters()).dtype
        self.starts = torch.as_tensor(starts).to(dev); self.V = V.to(dev, dt); self.m2 = m2.to(dev, dt); self.Rr = None if Rr is None else Rr.to(dev, dt); self.h = None

    def _hook(self, mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out; B, T = x.shape[:2]
        mask = (torch.arange(T, device=x.device)[None, :] >= self.starts[:, None]).to(x.dtype)[..., None]
        dm = self.m2[None, None, :] - x; delta = (dm @ self.V) @ self.V.T
        if self.Rr is not None:
            dr = torch.einsum("btr,bdr->btd", torch.einsum("btd,bdr->btr", dm, self.Rr), self.Rr)
            delta = dr * (delta.norm(dim=-1, keepdim=True) / dr.norm(dim=-1, keepdim=True).clamp_min(1e-12))
        x = x + delta * mask; return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x

    def __enter__(self): self.h = self.rk.LAY[BLOCK].register_forward_hook(self._hook); return self

    def __exit__(self, *a):
        if self.h is not None: self.h.remove(); self.h = None


def q6_forward(rk, seqs, blocks, patch=None, last_only=False, batch=32):
    """Forward passes without padding (texts grouped by token length; rsi_direction.py read each text alone): outputs of the given blocks
    (all positions, or the last one) and the log-probabilities of the next token at the last position (float32). patch(rows) -> a context
    manager (DepthPatch) for the rows of a batch, or None."""
    st = {b: [None] * len(seqs) for b in blocks}; lps = [None] * len(seqs); groups = {}
    for i, s in enumerate(seqs): groups.setdefault(len(s), []).append(i)
    for L, idx in sorted(groups.items()):
        for c in range(0, len(idx), batch):
            ii = idx[c:c + batch]; ids = torch.tensor([seqs[i] for i in ii]).to(rk.DEV); cur = {}

            def mk(b):
                def h(mod, inp, out): cur[b] = (out[0] if isinstance(out, tuple) else out).detach()
                return h
            with contextlib.ExitStack() as es:
                if patch is not None: es.enter_context(patch(ii))
                hs = [rk.LAY[b].register_forward_hook(mk(b)) for b in blocks]
                try:
                    with torch.no_grad():
                        hid = rk.model.model(input_ids=ids).last_hidden_state; lp = torch.log_softmax(rk.model.lm_head(hid[:, -1]).double(), -1)
                finally:
                    for h in hs: h.remove()
            for r, i in enumerate(ii):
                lps[i] = lp[r].float().cpu()
                for b in blocks: st[b][i] = (cur[b][r, -1] if last_only else cur[b][r]).float().cpu()
    return st, lps


def kl_lp(p, q):
    """KL(p || q) of two log-probability vectors (float64)."""
    p, q = p.double(), q.double(); return float((p.exp() * (p - q)).sum())


def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi


def ridge_dual(A, y, lam=1e-2):
    """rsi_direction.py's ridge readout: mu = mean(A); w = Z^T (Z Z^T + lam tr(Z Z^T) / n I)^-1 (y - mean y). Returns (mu, w, mean y)."""
    A = np.asarray(A, dtype=np.float64); y = np.asarray(y, dtype=np.float64); mu = A.mean(0); Z = A - mu; G = Z @ Z.T
    w = Z.T @ np.linalg.solve(G + lam * np.trace(G) / len(G) * np.eye(len(G)), y - y.mean()); return mu, w, float(y.mean())


def obj_boot(values, B=N_BOOT, seed=SEEDS["obj"], stat=np.mean):
    """Object bootstrap: replicates of stat over objects resampled with replacement (values [n] or [n, m] columns resampled together)."""
    v = np.asarray(values, dtype=float); g = np.random.default_rng(seed); idx = g.integers(0, len(v), (B, len(v)))
    return np.array([stat(v[i], axis=0) for i in idx])


def helix_stats(X):
    """Helix test on the mean last-token states over objects at depths 1-4 (X [n_obj, 4, D]): |d12|, |d23|, |d34|, cos(d12, d23),
    cos(d23, d34), cos(d12, d34) and the rotation prediction cos(2 theta) = 2 cbar^2 - 1 (cbar = mean of the two consecutive cosines)."""
    m = X.mean(0); d = [m[1] - m[0], m[2] - m[1], m[3] - m[2]]; c = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    c12, c23, c13 = c(d[0], d[1]), c(d[1], d[2]), c(d[0], d[2]); cb = (c12 + c23) / 2
    return dict(norm_d12=float(np.linalg.norm(d[0])), norm_d23=float(np.linalg.norm(d[1])), norm_d34=float(np.linalg.norm(d[2])), cos_d12_d23=c12,
                cos_d23_d34=c23, cos_d12_d34=c13, rotation_prediction_cos_d12_d34=2 * cb ** 2 - 1)


def depth_modes(X, tr):
    """Object-centred depth modes (12 training objects): SVD of the last-token states at depths 1-4 minus each object's mean over its
    depths (48 rows). Returns (V [D, 3], singular values)."""
    Xc = X[tr] - X[tr].mean(1, keepdims=True); U, Sv, Vt = np.linalg.svd(Xc.reshape(-1, X.shape[2]), full_matrices=False)
    return Vt[:3].T, Sv


def angular_readout(X, tr, te):
    """Angular readout in the top-2 non-step plane: object-centred states; the step direction s = unit(mean over training objects of
    (x_4 - x_1)) is projected out; the plane = the top-2 right singular vectors of the training objects' centred, step-free states; the
    rotation sense omega = sign of the summed wrapped angle step 1 -> 2 of the training objects (learned from depths 1-2, as the linear
    readout is); a held-out object ranks depth 4 above depth 3 if omega * wrap(phi_4 - phi_3) > 0."""
    Xc = X - X.mean(1, keepdims=True); s = Xc[tr][:, 3].mean(0) - Xc[tr][:, 0].mean(0); s = s / np.linalg.norm(s)
    Y = Xc - (Xc @ s)[..., None] * s; U, Sv, Vt = np.linalg.svd(Y[tr].reshape(-1, X.shape[2]), full_matrices=False); e1, e2 = Vt[0], Vt[1]
    phi = np.arctan2(Y @ e2, Y @ e1); omega = np.sign(wrap(phi[tr][:, 1] - phi[tr][:, 0]).sum()) or 1.0
    ranks = (omega * wrap(phi[te][:, 3] - phi[te][:, 2]) > 0).astype(float)
    return dict(share=float(ranks.mean()), omega=float(omega), plane_share=float((Sv[:2] ** 2).sum() / (Sv ** 2).sum())), ranks


def linear_readout(X, tr, te):
    """rsi_direction.py's Q2' linear readout: ridge on the training objects' depth-1 and depth-2 states (k-major), held-out objects rank
    depth 4 above depth 3 if (x_4 - mu) . w > (x_3 - mu) . w."""
    A = np.concatenate([X[tr][:, 0], X[tr][:, 1]]); y = np.array([1.0] * len(tr) + [2.0] * len(tr)); mu, w, _ = ridge_dual(A, y)
    ranks = (((X[te][:, 3] - mu) @ w) > ((X[te][:, 2] - mu) @ w)).astype(float); return dict(share=float(ranks.mean())), ranks


def depth_readout(X, tr):
    """Ridge depth readout (rsi_direction.py's ridge) trained on the training objects' unpatched last-token states at depths 1-4 (targets
    1-4, k-major); value(x) = (x - mu) . w + mean depth."""
    A = np.concatenate([X[tr][:, k] for k in range(4)]); y = np.repeat(np.arange(1.0, 5.0), len(tr)); mu, w, yb = ridge_dual(A, y)
    return lambda x: (np.asarray(x, dtype=np.float64) - mu) @ w + yb


def _share_ci(v):
    v = np.asarray(v, dtype=float); return dict(share=float(v.mean()), ci90=list(R.interval(obj_boot(v), LEVEL)), n=len(v))


def _median_ci(v):
    v = np.asarray(v, dtype=float); return dict(median=float(np.median(v)), ci90=list(R.interval(obj_boot(v, stat=np.median), LEVEL)), n=len(v))


def q6_relation(rk, rel, items, tr, te, cache):
    """Unpatched states of one relation, its helix / readout statistics, the caps of depth-3 and depth-4 sentences and their KL fractions,
    readouts and self-repair."""
    it = [x for x in items if x["rel"] == rel]; pos = {(x["k"], x["obj"]): j for j, x in enumerate(it)}; nO = len(S.NEST_OBJ)
    st, lp = q6_forward(rk, [x["ids"] for x in it], Q6_BLOCKS)                 # unpatched (recomputed on a rerun: 96 short forwards)
    X = {b: np.stack([np.stack([st[b][pos[(k, o)]][-1].double().numpy() for k in S.NEST_DEPTHS]) for o in range(nO)]) for b in Q6_BLOCKS}
    out = dict(helix={}, angular={}, linear={}, readout={}, cap={}, kl={}); arr = dict(ranks={})
    for b in Q6_BLOCKS:
        out["helix"][str(b)] = helix_stats(X[b]); a, ra = angular_readout(X[b], tr, te); l, rl = linear_readout(X[b], tr, te)
        out["angular"][str(b)] = dict(a, **_share_ci(ra)); out["linear"][str(b)] = dict(l, **_share_ci(rl)); arr["ranks"][b] = dict(angular=ra, linear=rl)
    V, Sv = depth_modes(X[BLOCK], tr); m2 = X[BLOCK][tr][:, 1].mean(0)
    out["modes"] = dict(singular_values=Sv[:6].tolist(), share_top3=float((Sv[:3] ** 2).sum() / (Sv ** 2).sum()), block=BLOCK)
    RD = {b: depth_readout(X[b], tr) for b in Q6_READ_BLOCKS}
    for b in Q6_READ_BLOCKS:
        val = np.array([[RD[b](X[b][o, k]) for k in range(4)] for o in range(nO)]); r43 = (val[te, 3] > val[te, 2]).astype(float)
        out["readout"][str(b)] = dict(validation=_share_ci(r43), validated=bool(r43.mean() >= RULE["q6_valid"]), unpatched_depth4_median=float(np.median(val[:, 3])),
                                      unpatched_depth3_median=float(np.median(val[:, 2])), unpatched_depth4_ge_3_5_all=float((val[:, 3] >= RULE["q6_repair_depth"]).mean()))
        arr.setdefault("readout_values", {})[b] = val
    LP = lambda k, o: lp[pos[(k, o)]]
    floor = [kl_lp(LP(4, o), LP(4, q)) for o in range(nO) for q in range(nO) if q != o]; k24 = np.array([kl_lp(LP(2, o), LP(4, o)) for o in range(nO)])
    k23 = np.array([kl_lp(LP(2, o), LP(3, o)) for o in range(nO)])
    out["kl"] = dict(same_depth_floor_median=float(np.median(floor)), kl_2_4_median=float(np.median(k24)), kl_2_3_median=float(np.median(k23)),
                     usable=bool(np.median(k24) > np.median(floor)), rule="KL used only if the median KL(p2 || p4) exceeds the median KL between different objects at depth 4")
    arr["kl"] = dict(kl_2_4=k24, kl_2_3=k23, floor=np.array(floor))
    arr["patch"] = {}; checks = []
    for k in (3, 4):
        seqs = [it[pos[(k, o)]]["ids"] for o in range(nO)]; starts = torch.tensor([it[pos[(k, o)]]["start"] for o in range(nO)]); base = k23 if k == 3 else k24
        for r in Q6_RANKS:
            Vr = torch.as_tensor(V[:, :r]); m2t = torch.as_tensor(m2)
            for kind in ("mode", "rand"):
                Rr = None
                if kind == "rand":
                    Rr = torch.stack([torch.linalg.qr(torch.randn(rk.D, r, generator=torch.Generator().manual_seed(int(rk.keyseed("r2Q6rand", rel, k, r, o) % (2 ** 31))), dtype=torch.float64))[0]
                                      for o in range(nO)])
                mk = lambda ii, Rr=Rr, Vr=Vr: DepthPatch(rk, starts[ii], Vr, m2t, None if Rr is None else Rr[ii])

                def run_cap(seqs=seqs, mk=mk):
                    s_, l_ = q6_forward(rk, seqs, (BLOCK,) + Q6_READ_BLOCKS, patch=mk, last_only=True)
                    return dict(st=s_, kc=np.array([kl_lp(LP(2, o), l_[o]) for o in range(nO)]))
                res = cache.get(f"{rel}_cap_{k}_{r}_{kind}", sig_of("cap", rel, k, r, kind, Vr, m2t, Rr if Rr is not None else "none"), run_cap)
                kc = res["kc"]; frac = 1 - kc / base
                rv = {b: np.array([RD[b](res["st"][b][o].double().numpy()) for o in range(nO)]) for b in Q6_READ_BLOCKS}
                if kind == "mode":                                     # the patched block-14 last state carries m2's projection on the modes
                    x14 = torch.stack([res["st"][BLOCK][o].double() for o in range(nO)]); checks.append(float(((x14 - m2t) @ Vr).abs().max() / (x14 @ Vr).abs().max().clamp_min(1e-12)))
                e = dict(frac_all=_median_ci(frac), frac_heldout=_median_ci(frac[te]), kl_cap_median=float(np.median(kc)))
                for b in Q6_READ_BLOCKS:
                    e[f"readout_{b}"] = dict(median=float(np.median(rv[b])), ge_3_5_all=float((rv[b] >= RULE["q6_repair_depth"]).mean()),
                                             ge_3_5_heldout=float((rv[b][te] >= RULE["q6_repair_depth"]).mean()))
                out["cap"][f"k{k}_r{r}_{kind}"] = e; arr["patch"][f"k{k}_r{r}_{kind}"] = dict(frac=frac, kl=kc, readout=rv, last18=torch.stack(res["st"][18]), last22=torch.stack(res["st"][22]))
    out["patch_check_rel_max"] = max(checks) if checks else float("nan")
    # in-session view of r* (r2_ana decides): fewest r whose mode cap of depth-4 sentences moves >= 50 % (median) while random patches < 20 %
    rs = [r for r in Q6_RANKS if out["cap"][f"k4_r{r}_mode"]["frac_all"]["median"] >= RULE["q6_frac"] and out["cap"][f"k4_r{r}_rand"]["frac_all"]["median"] < RULE["q6_rand"]]
    out["r_star_in_session"] = rs[0] if rs else None
    states16 = [torch.stack([st[b][j] for b in Q6_BLOCKS], 1).half() for j in range(len(it))]          # [L, 4 blocks, D] per text
    arr.update(V=torch.as_tensor(V).float(), m2=torch.as_tensor(m2).float(), states=states16, ids=[x["ids"] for x in it], keys=[(x["k"], x["obj"]) for x in it],
               starts=[x["start"] for x in it])
    return out, arr


def stage_q6(root, r1_dir, r2_dir, rk, how):
    """C-Q6 (R1 inputs: none). rk: block 14 opened through r2_rk when its round-1 outputs are final, else the model loaded without a round-1
    state (open_model_only; run_stage decides)."""
    cache = Cache(r2_dir, "q6"); items = q6_items(rk.enc); tr, te = S.nesting_split()
    rels, arrs = {}, {}
    for rel in S.NEST_RELS:
        rels[rel], arrs[rel] = q6_relation(rk, rel, items, tr, te, cache); log(f"q6: {rel} done")
    spec = {}
    for b in Q6_BLOCKS:
        a_i, a_d = arrs["improves"]["ranks"][b]["angular"], arrs["describes"]["ranks"][b]["angular"]; diff = a_i - a_d
        spec[str(b)] = dict(diff=float(diff.mean()), ci90=list(R.interval(obj_boot(diff), LEVEL)), rule="RSI-specific only if improves beats describes by >= 0.2 with the interval excluding 0")
    js = base_info("q6", dict(status="ran", model=how, split=dict(train=tr, heldout=te, source="rsi_direction.py's split (default_rng(13))"),
                              n_items=len(items), blocks=list(Q6_BLOCKS), read_blocks=list(Q6_READ_BLOCKS), ranks=list(Q6_RANKS), relations=rels, improves_minus_describes_angular=spec,
                              notes=["texts as plain text without chat template, read alone (rsi_direction.py's Q2'); last-token states for modes and readouts",
                                     "cap: block 14, every position from the second 'the way it' (' The way it' in the repetition texts) to the last token; m2 = mean depth-2 last-token state of the 12 training objects",
                                     "random control: one random orthonormal r-dim subspace per (relation, depth, r, object), seed keyseed('r2Q6rand', rel, k, r, obj), patch rescaled per position to the mode patch's norm",
                                     "frac = 1 - KL(p2 || p_cap) / KL(p2 || p_k) at the last position; medians over all 24 objects (frac_all) and over the 12 held-out objects",
                                     "self-repair: share of capped sentences that the depth readouts at blocks 18 / 22 still read as depth >= 3.5"]))
    return write_outputs(r2_dir, "q6", js, dict(items=[{k: v for k, v in x.items() if k != "text"} for x in items], relations=arrs))


def open_model_only(root, r2_dir):
    """C-Q6 without final round-1 outputs at block 14: imports rk_loop with an empty state of its own (<r2_dir>/local/q6_rk; no round-1
    file read), only for the model, tokenizer and hooks."""
    if "rk_loop" in sys.modules: raise RuntimeError("rk_loop already imported")
    os.environ["BLOCK"] = str(BLOCK); os.environ["ASCALE"] = "1"; os.environ.setdefault("POOLDIRS", R.pooldirs_path(root)); os.environ.setdefault("PERSONA", f"{root}/rsi_dl/persona")
    argv0 = list(sys.argv); sys.argv = [os.path.join(JU, "rk_loop.py"), root, os.path.join(r2_dir, "local", "q6_rk"), "lib"]
    try:
        import rk_loop as rk
    finally: sys.argv = argv0
    return rk


# ======================================================================================================================================
# Driver
# ======================================================================================================================================
STAGE_FN = dict(q3=stage_q3, q4a=stage_q4a, q2=stage_q2, q4b=stage_q4b, armA=stage_armA, q5=stage_q5)
# prerequisites inside <r2_dir>: 'axes' = r2_axes.py's outputs (axes/r2_axes.json, axes/r2_axes.pt); q4a also needs q3's outputs
PREREQ = dict(q3=("axes",), q4a=("axes", "q3"), q2=("axes",), q4b=(), armA=(), q5=(), q6=())
_HOOK_DICTS = ("_forward_hooks", "_forward_pre_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called", "_forward_pre_hooks_with_kwargs",
               "_backward_hooks", "_backward_pre_hooks")


def _hook_snapshot(model):
    return {id(m): {n: set(getattr(m, n).keys()) for n in _HOOK_DICTS if isinstance(getattr(m, n, None), dict)} for m in model.modules()}


def _hook_restore(model, snap):
    """Removes every hook registered after the snapshot (all hook dicts of every module). Returns the number of forward / pre-hooks removed."""
    n = 0
    for m in model.modules():
        s = snap.get(id(m), {})
        for nm in _HOOK_DICTS:
            d = getattr(m, nm, None)
            if not isinstance(d, dict): continue
            for k in [k for k in d if k not in s.get(nm, set())]:
                d.pop(k); n += nm in ("_forward_hooks", "_forward_pre_hooks")
    return n


@contextlib.contextmanager
def stage_scope(rk, stage):
    """Isolation of one stage in a process that runs several ('all'): on exit, also after an exception, every hook registered on the model
    during the stage is removed, rk.TEMP and rk.Gumbel are restored, steering and the early-stop switch are off; after a failure rk.ST is
    reloaded from its saved state, so no unsaved entry survives; GPU memory is released (gc, torch.cuda.empty_cache)."""
    snap = _hook_snapshot(rk.model); temp, gum = rk.TEMP, rk.Gumbel; ok = False
    try:
        yield
        ok = True
    finally:
        n = _hook_restore(rk.model, snap); rk.TEMP, rk.Gumbel = temp, gum; rk.steer_off(); rk.capped_reset(); rk.CAP["on"] = False; rk.CAP.pop("h", None)
        if not ok and R._RK is rk and os.path.exists(R.BLK.get("state", "")):
            rk.ST = torch.load(R.BLK["state"], map_location="cpu", weights_only=False)
        if n: log(f"{stage}: {n} leftover hook(s) removed")
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


def _selftest_injection(rk, stage):
    """Code test only (MOCK and env R2_UNLEARN_FAIL=<stage>): the stage fails after leaving a WriteOrth hook, a temperature, a steering
    switch and an unsaved state entry behind, so that the 'all' test sees that stage_scope contains a failure."""
    if not (R.MOCK and os.environ.get("R2_UNLEARN_FAIL") == stage): return
    if R._RK is rk:
        R.WriteOrth(R.unit(rk.ST["axes"]["u_T_perp"].double())[None]).__enter__(); rk.ST["r2_selftest_unsaved"] = 1
        rk.STEER.update(v=R.unit(rk.ST["axes"]["u_T_perp"].float()), V=None, alpha=1.0, cap=rk.ST["ref"]["cap"], sign=torch.tensor([1.0]), rowstart=torch.tensor([rk.START]))
    rk.TEMP = 0.3
    raise RuntimeError(f"injected failure in stage {stage} (selftest, R2_UNLEARN_FAIL)")


def run_stage(root, r1_dir, r2_dir, stage):
    """Runs one stage (skipped when unlearn/<stage>.json exists). Block 14 is opened through r2_rk.open_block; a block whose round-1 outputs
    are missing or not final (G4) gives 'not run (round-1 output missing)' (C-Q6, which reads no round-1 file, runs anyway: the model is then
    loaded without a round-1 state). The stage runs inside stage_scope."""
    if stage not in STAGES: raise SystemExit(f"unknown stage {stage!r}; stages: {', '.join(STAGES)}")
    if os.environ.get("BLOCK", str(BLOCK)) not in ("", str(BLOCK)): raise SystemExit(f"r2_unlearn_qwen.py works at block {BLOCK}; env BLOCK={os.environ['BLOCK']}")
    js_p, _ = out_paths(r2_dir, stage)
    if os.path.exists(js_p): log(f"{stage}: {js_p} exists, skipped"); return dict(json.load(open(js_p)), _done_earlier=True)
    t0 = time.time(); _LOG.update(stage=stage, start=len(R.FINDER_LOG)); was_open = R._RK is not None
    try: rk = R.open_block(root, r1_dir, r2_dir, BLOCK); how = "open_block"
    except R.R1Missing as e:
        if stage != "q6": return not_run(r2_dir, stage, NOT_RUN_R1, error=str(e))
        rk = open_model_only(root, r2_dir); how = f"model only ({e})"
    if not was_open and R._RK is not None: _LOG.update(open_records=list(R.FINDER_LOG[_LOG["start"]:]), open_stage=stage)
    with stage_scope(rk, stage):
        _selftest_injection(rk, stage)
        js = stage_q6(root, r1_dir, r2_dir, rk, how) if stage == "q6" else STAGE_FN[stage](Ctx(root, r1_dir, r2_dir, stage, rk))
    log(f"{stage}: {js.get('status')} ({time.time() - t0:.0f} s)"); return js


def main(root, r1_dir, r2_dir, stage):
    """One stage, or 'all' = the seven stages in order in this process (one model load). Every stage whose prerequisites exist runs (PREREQ:
    q3, q4a and q2 need r2_axes.py's outputs, q4a also needs q3); a failing stage does not stop the independent later ones, and the stages
    that depend on it are skipped. A stage whose output exists is skipped, so a rerun repeats only what failed. Prints a per-stage summary
    and returns 0 if every stage is done ('ran', 'not run (...)', 'not testable (...)', or done earlier), else 1."""
    stages = list(STAGES) if stage == "all" else [stage]; summary, bad = {}, set()
    for st in stages:
        dead = [p for p in PREREQ[st] if p in bad]
        if dead: summary[st] = f"skipped (prerequisite {', '.join(dead)} failed)"; bad.add(st); continue
        try:
            js = run_stage(root, r1_dir, r2_dir, st); summary[st] = ("done earlier: " if js.get("_done_earlier") else "") + str(js.get("status"))
        except MissingInput as e:
            summary[st] = f"failed (missing round-2 input: {e})"; bad.add(st); log(f"{st}: {summary[st]}")
        except Exception as e:                                                   # noqa: BLE001 - one stage must not stop the others
            traceback.print_exc(); summary[st] = f"failed ({type(e).__name__}: {e})"; bad.add(st)
        finally:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("[r2_unlearn] summary: " + " | ".join(f"{st}: {v}" for st, v in summary.items()), flush=True)
    for st, v in summary.items(): print(f"[r2_unlearn]   {st:5s} {v}", flush=True)
    return 1 if bad else 0


# ======================================================================================================================================
# MOCK fixture and code test
# ======================================================================================================================================
def mock_fixture(root, r1_dir):
    """MOCK round-1 layout (section 9): rk_loop.py MOCK=1 at block 14 'all' into <r1_dir>/gpu1/rk (skipped when present; DEVICE=cpu, NT=1),
    then r2_rk.mock_r1's fake dn_delta.json, blocks_chosen.json (b_lo 8, b_hi 20, whose states are absent), summary_b14.json (a 'would
    have read: likely' verdict with GO1 failed) and shard_done.json. Test use only."""
    if not os.path.exists(os.path.join(r1_dir, "gpu1", "rk", f"b{BLOCK}", "loops.json")):
        os.makedirs(os.path.join(r1_dir, "gpu1", "rk"), exist_ok=True); env = dict(os.environ, MOCK="1", BLOCK=str(BLOCK), DEVICE="cpu", NT="1", OMP_NUM_THREADS="1"); env.pop("ASCALE", None)
        subprocess.run([sys.executable, os.path.join(JU, "rk_loop.py"), root, os.path.join(r1_dir, "gpu1", "rk"), "all"], env=env, check=True)
    if not os.path.exists(os.path.join(r1_dir, "gpu1", f"summary_b{BLOCK}.json")): R.mock_r1(root, r1_dir, spec=((BLOCK, "gpu1", ("all",)),))


def _cli(root, r1, r2, stage, script=None, extra_env=None, want_out=False, quiet=False):
    env = dict(os.environ); env.pop("ASCALE", None); env.pop("BLOCK", None); env.pop("R2_UNLEARN_FAIL", None)
    env.update(MOCK="1", DEVICE=os.environ.get("DEVICE", "cpu"), NT=os.environ.get("NT", "1"), OMP_NUM_THREADS="1", **(extra_env or {}))
    cmd = [sys.executable, script or os.path.abspath(__file__), root, r1, r2] + ([stage] if stage else [])
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if p.returncode != 0 and not quiet: print(p.stdout[-3000:], p.stderr[-3000:], flush=True)
    return (p.returncode, p.stdout + p.stderr) if want_out else p.returncode


def selftest(root, r1_dir, r2_dir):
    """MOCK code test of every stage: fixture, O2 (r2_axes.py), every stage through the CLI (one process each, as the runner calls them),
    resume, then the outputs checked against their definitions and against rk_loop / r2_rk in this process; gate branches and 'not run'
    paths on modified copies of the fixture; the rule functions on synthetic inputs."""
    assert R.MOCK, "selftest needs MOCK=1"; res = []; t0 = time.time()

    def check(name, cond, info=""):
        res.append(dict(test=name, ok=bool(cond), info=str(info)[:600])); print(f"[{'PASS' if cond else 'FAIL'}] {name} {str(info)[:300]}", flush=True)
    mock_fixture(root, r1_dir)
    if not os.path.exists(os.path.join(r2_dir, "axes", "r2_axes.json")):
        assert _cli(root, r1_dir, r2_dir, "all", script=os.path.join(HERE, "r2_axes.py")) == 0, "r2_axes.py failed"
    selftest_rules(check); selftest_cache(check, r2_dir)
    rc = {st: _cli(root, r1_dir, r2_dir, st) for st in STAGES}
    check("CLI: every stage exits 0 and writes unlearn/<stage>.json (+ .pt)", all(v == 0 for v in rc.values()) and all(os.path.exists(out_paths(r2_dir, st)[0]) for st in STAGES), rc)
    mt = {st: os.path.getmtime(out_paths(r2_dir, st)[0]) for st in STAGES}
    rc2 = _cli(root, r1_dir, r2_dir, "all")
    check("resume: a second run ('all') skips every stage", rc2 == 0 and mt == {st: os.path.getmtime(out_paths(r2_dir, st)[0]) for st in STAGES})
    J = {st: json.load(open(out_paths(r2_dir, st)[0])) for st in STAGES}
    P = {st: torch.load(out_paths(r2_dir, st)[1], map_location="cpu", weights_only=False) if os.path.exists(out_paths(r2_dir, st)[1]) else None for st in STAGES}
    rk = R.open_block(root, r1_dir, r2_dir, BLOCK); ctx = Ctx(root, r1_dir, r2_dir, "selftest", rk)
    selftest_q3(check, ctx, J["q3"], P["q3"])
    selftest_q4a(check, ctx, J["q4a"], P["q4a"], J["q3"], P["q3"])
    selftest_q2(check, ctx, J["q2"], P["q2"])
    selftest_loops(check, ctx, J["q4b"], P["q4b"], J["armA"], P["armA"])
    selftest_q5(check, ctx, J["q5"], P["q5"])
    selftest_q6(check, ctx, J["q6"], P["q6"]); selftest_scope(check, ctx)
    ctx_q5_before[0] = J["q5"]; selftest_variants(check, root, r1_dir, r2_dir, ctx, J)
    ok = all(r["ok"] for r in res); R.write_json(os.path.join(r2_dir, "unlearn", "selftest.json"), dict(ok=ok, n=len(res), passed=sum(r["ok"] for r in res), seconds=time.time() - t0, tests=res))
    print(f"r2_unlearn_qwen selftest: {sum(r['ok'] for r in res)}/{len(res)} passed ({time.time() - t0:.0f} s)", flush=True); return ok


def selftest_rules(check):
    """The rule functions on synthetic inputs (no model)."""
    mf = dict(sports=0.01, tone=0.02, sea=0.0, TOPIC=0.03)
    u = rule_unlearned(0.3, 1.0, 0.6, 0.8, 0.5, 2.0, -3.0, 0.02, mf, 0.7, D_hi=0.9)
    f1 = rule_unlearned(0.3, 1.0, 0.6, 0.55, 0.5, 2.0, -3.0, 0.02, mf, 0.7)                      # upper end not below the random p5
    f2 = rule_unlearned(0.3, 1.0, 0.6, 0.8, 0.05, 0.1, -3.0, 0.02, mf, 0.7)                      # D_0 <= 0.2
    f3 = rule_unlearned(0.3, 1.0, 0.6, 0.8, 0.5, 2.0, -3.0, 0.02, dict(mf, sea=0.4), 0.7)        # a foreign axis moves too much
    f4 = rule_unlearned(float("nan"), 1.0, 0.6, 0.8, 0.5, 2.0, -3.0, 0.02, mf, 0.7)
    inv = rule_unlearned(0.3, 1.0, 0.6, 0.8, -0.5, 2.0, -3.0, 0.02, mf, 0.7, D_hi=-0.1)
    check("rule_unlearned: pass, random reference, baseline D, foreign axes, NaN, inverted",
          u["unlearned"] and not f1["unlearned"] and f1["failed"] == ["gap_below_random"] and not f2["unlearned"] and f2["note"] == "no baseline discrimination"
          and f3["failed"] == ["foreign"] and not f4["unlearned"] and inv["inverted"] and not u["inverted"], (u["failed"], f1["failed"], f2["failed"], f3["failed"]))
    s = rule_suppressed(0.4, 1.0, 0.7, 0.9, 2.0, 3.0, 3.5, 0.05, mf, 0.6); s2 = rule_suppressed(0.4, 1.0, 0.7, 0.9, 2.0, 3.0, 0.05, 0.05, mf, 0.6)
    s3 = rule_suppressed(0.4, 1.0, 0.7, 0.9, 2.0, 3.0, 3.5, 0.06, mf, 0.6)
    check("rule_suppressed: pass; M0 <= 0.1 fails as 'no baseline discrimination'; damage outside the band",
          s["suppressed"] and not s2["suppressed"] and s2["note"] == "no baseline discrimination" and s3["failed"] == ["damage_band"], (s["failed"], s2["failed"], s3["failed"]))
    check("rule_cue: shallow / deep / partial / not applicable", rule_cue(0.6, 0.8, True) == "shallow" and rule_cue(0.1, 0.3, True) == "deep" and rule_cue(0.1, 0.4, True) == "partial"
          and rule_cue(0.3, 0.4, True) == "partial" and rule_cue(0.9, 1.0, False) == "not applicable (nothing removed)")
    w = "not decidable (pilot gate GO1 failed; descriptive only; would have read: likely, substantial RSI component)"
    check("LIKELY_wh: 'likely', would-have-read, other verdicts", likely_wh("likely (provisional)") == (True, False) and likely_wh(w) == (True, True)
          and likely_wh("not decidable (pilot gate GO1 failed; would have read: unlikely)") == (False, False) and likely_wh("unlikely") == (False, False))
    rows = dict(RND_1=dict(p=0.001, A=0.3), RND_2=dict(p=0.001, A=-0.5))
    check("G14 attracted loop (RND_1 first; |A|; label needed)", attracted_loop(True, rows) == "RND_1" and attracted_loop(True, dict(RND_1=dict(p=0.05, A=0.3), RND_2=rows["RND_2"])) == "RND_2"
          and attracted_loop(False, rows) is None and attracted_loop(True, dict(RND_1=dict(p=0.001, A=0.1))) is None)
    check("G44 gate: likely branch first, attractor, closed", q4b_gate(True, True, "RND_1")["branch"] == "likely" and q4b_gate(True, False, "RND_2")["branch"] == "attractor"
          and not q4b_gate(True, False, None)["open"] and "G15" in q4b_gate(True, False, None)["verdict"] and "G13" in q4b_gate(False, False, None)["verdict"])
    mk = lambda c, p, A, rel=0.6, Ue=0.3, lo=0.1: dict(c=c, p=p, A=A, rel=rel, Ue=Ue, Ue_ci90=[lo, 0.5])
    gone = {2: mk(0.01, 0.5, 0.1), 3: mk(0.02, 0.4, -0.1)}; good = {2: mk(0.5, 0.001, 0.6), 3: mk(0.5, 0.001, 0.5)}; rer = {2: mk(0.1, 0.05, 0.2), 3: mk(0.1, 0.05, 0.2)}
    gone_low = {2: mk(0.01, 0.5, 0.1, rel=0.3), 3: mk(0.02, 0.4, -0.1, rel=0.3)}
    check("C-Q4b loop rule in the plan's order (review unlearn_b-3): un-searched; survives; re-routed before 'twin does not re-find RSI' (also when gone); undecided",
          rule_loop(gone, good) == "un-searched" and rule_loop(gone, gone) == "re-routed" and rule_loop(gone_low, gone_low) == "undecided (twin does not re-find RSI)"
          and rule_loop(good, gone) == "survives" and rule_loop(rer, good) == "re-routed" and rule_loop({2: mk(0.1, 0.05, 0.3), 3: mk(0.1, 0.05, 0.3)}, good) == "undecided"
          and rule_loop({2: mk(0.1, 0.05, 0.2, rel=0.3), 3: mk(0.1, 0.05, 0.2)}, good) == "undecided"
          and rule_loop(gone_low, comparator=gone_low) == "undecided (the R1 seed-0 comparator does not re-find RSI)" and rule_loop(None) == "undecided (rounds d_2, d_3 missing)")
    try:
        import r2_unlearn_verdict as UV
        rows = [mk(c, p, A, rel, Ue, lo) for c in (-0.1, 0.2) for p in (0.005, 0.05, 0.5) for A in (-0.5, 0.1, 0.5) for rel in (0.3, 0.7) for Ue in (0.1, 0.4) for lo in (-0.1, 0.1)]
        refs = [None] + [{2: mk(0.5, p, A), 3: mk(0.5, p, A)} for p in (0.005, 0.5) for A in (0.1, 0.5)]
        nm = "the R1 seed-0 comparator"
        dis = [(a, t) for a in rows for t in refs if rule_loop({2: a, 3: a}, twin=t) != UV.loop_rule({2: a, 3: a}, t, "twin")
               or rule_loop({2: a, 3: a}, comparator=t, ref_name=nm) != UV.loop_rule({2: a, 3: a}, t, nm)]
        check(f"rule_loop = r2_unlearn_verdict.loop_rule (OD-G8) on {len(rows) * len(refs)} row combinations, twin and comparator", not dis, dis[:2])
    except ImportError as e: print(f"[INFO] r2_unlearn_verdict not importable, OD-G8 cross-check of rule_loop skipped: {e}", flush=True)
    check("_rung: 'not decidable (no baseline gap: g_0 <= 0)' unless g_0 > 0 (review unlearn_a-2); one direction / span / not removed",
          _rung(-0.25, 0.1, -0.04) == RUNG_NO_BASELINE and _rung(0.4, 0.9, float("nan")) == RUNG_NO_BASELINE and _rung(2.0, 2.0, 0.0) == RUNG_NO_BASELINE
          and _rung(0.4, 0.9, 1.0) == "one direction" and _rung(0.7, 0.4, 1.0) == "two-dimensional span" and _rung(0.7, 0.6, 1.0) == "not removed within the tested 2-dim span")
    try:
        import r2_linearity as LIN
        mk3 = lambda r1, r2, g0: dict(status="ran", linear_rung=dict(inputs=dict(RSI_1dim=dict(ratio=r1, g0=g0), RSI_span=dict(ratio=r2, g0=g0)), RSI=_rung(r1, r2, g0)))
        cases = [(r1, r2, g0) for r1 in (-0.25, 0.4, 0.7, 2.0) for r2 in (0.3, 0.6) for g0 in (-0.04, 0.0, 1.0)]
        check("_rung agrees with r2_linearity.linear_rung (OD-G8), also for g_0 <= 0", all(LIN.linear_rung(mk3(*c), "RSI")["agrees_with_in_session"] for c in cases))
    except ImportError as e: print(f"[INFO] r2_linearity not importable, OD-G8 cross-check of _rung skipped: {e}", flush=True)
    dr = np.random.default_rng(3).normal(0.4, 0.05, 2000); ci = centred_interval(0.5, dr); q = np.percentile(dr, [5, 95])
    check("Amendment 3 (a): G16 interval = percentile interval shifted by c - median; unshifted interval and median kept",
          abs(ci["boot_median"] - np.median(dr)) < 1e-12 and np.allclose(ci["ci90_percentile"], q) and np.allclose(ci["ci90"], q + 0.5 - np.median(dr))
          and ci["ci90"][0] < 0.5 < ci["ci90"][1] and not ci["ci90_percentile"][0] < 0.5 < ci["ci90_percentile"][1], ci)
    STx = {"ref": 1, "k6_null0": 2, "k6_r2_u_com": 3, "r2_b05_null0": 4, "r2_b05_u_RSI": 5, Q2_STATE_KEY: "old", "r2_other": 6}
    gone_keys = q2_state_purge(STx, "new"); same = q2_state_purge(dict(STx, **{Q2_STATE_KEY: "new", "k6_r2_x": 1}), "new")
    check("q2 state validity (review unlearn_a-1): k6_r2_* and r2_b05_* purged when code or inputs differ; kept when they match",
          gone_keys == ["k6_r2_u_com", "r2_b05_null0", "r2_b05_u_RSI"] and set(STx) == {"ref", "k6_null0", Q2_STATE_KEY, "r2_other"} and same == [], gone_keys)
    check("G16 arm A continuation", arma_continues(0.1, 0.2) and not arma_continues(0.3, 0.2) and not arma_continues(float("nan"), 0.2))
    a = np.array([0.1, -0.2, 0.3, 0.05]); u = 0.5 + 2 * a; nl = [dict(A_fin=x, Ue_fin=y) for x, y in zip(a, u)]; rg = ols_nulls(nl)
    check("O5 null regression and z_fin (exact line: se 0, beta (0.5, 2))", abs(rg["beta"][0] - 0.5) < 1e-12 and abs(rg["beta"][1] - 2) < 1e-12 and rg["se"] < 1e-12)
    rg2 = dict(beta=[0.0, 1.0], se=0.5, a_mean=0.0, Saa=1.0, n=16); z = z_fin(1.0, 0.0, rg2)
    check("z_fin formula", abs(z - 1.0 / (0.5 * math.sqrt(1 + 1 / 16))) < 1e-12, z)
    cal = calibrate(lambda f: 0.05 * (f / 2.0) ** 2, 1.0, CAL05); cal3 = calibrate(lambda f: 1e-3 * f, 1.0, CAL03)
    check("calibration rule: converges into the band; else the closest of at most max_eval", cal["in_band"] and abs(cal["f"] - 2.0) < 1e-9 and len(cal3["tried"]) == 3 and not cal3["in_band"], (cal, cal3["tried"]))
    g = torch.Generator().manual_seed(0); d = torch.randn(400, 8, generator=g, dtype=torch.float64) + 0.5; ok = torch.ones(400, dtype=torch.bool)
    check("U-statistic |E delta|^2 (unbiased; equals |mean|^2 - tr(cov)/n)", abs(u_norm2(d, ok) - (float(d.mean(0) @ d.mean(0)) - float(d.var(0).sum()) / 400)) < 1e-10)
    s = torch.tensor([1.0, float("nan"), 3.0, 5.0], dtype=torch.float64); Wt = np.array([[1, 1, 1, 1], [2, 0, 0, 2], [0, 4, 0, 0]], dtype=float)
    bg = boot_g(s, Wt).numpy()
    check("boot_g: weighted mean over kept prompts, NaN if none kept", abs(bg[0] - 3.0) < 1e-12 and abs(bg[1] - 3.0) < 1e-12 and np.isnan(bg[2]))
    check("z_against: ddof 1", abs(z_against(3.0, [1.0, 2.0, 3.0]) - 1.0) < 1e-12 and np.isnan(z_against(1.0, [1.0])))
    g = torch.Generator().manual_seed(1); p0 = 1.0 + 0.2 * torch.randn(64, generator=g, dtype=torch.float64); pa = 0.1 * p0 + 0.05 * torch.randn(64, generator=g, dtype=torch.float64)
    pc = pa + 0.6 * (p0 - pa); pc[3] = float("nan"); Wb = R.prompt_bootstrap(64, B=500, seed=1); rs, dr = restored_share(pc, pa, p0, Wb)
    rs0, _ = restored_share(pa, p0, p0, Wb)
    check("restored share: (g_cue - g_abl) / (g_0 - g_abl) on per-prompt values (NaN prompts drop out), joint bootstrap; undefined if g_abl = g_0",
          rs["defined"] and abs(rs["share"] - (np.nanmean(pc.numpy()) - pa.mean().item()) / (p0.mean().item() - pa.mean().item())) < 1e-12 and rs["ci90"][0] < 0.6 < rs["ci90"][1]
          and not rs0["defined"] and math.isnan(rs0["share"]), (rs, rs0["defined"]))


def _gap_recompute(ref, Xp, Xa, W):
    G = R.gap_on_fixed_axes(Xp, Xa, ref["fold"], ref["a_A"], ref["a_B"]); return G["g"], (boot_g(G["per_prompt"], W) / ref["g0_draws"]).numpy()


def selftest_q3(check, ctx, J, P):
    rk = ctx.rk; O = ctx.o2(); A, alias = q3_ablations(ctx)
    check("q3: status ran, 29 ablations, 10 text sets, GA label", J["status"] == "ran" and len(J["ablations"]) == 29 and len(P["texts"]) == 10
          and (J["GA14"] or J["labels"]["u_RGI_perp"] == ["RGI axis not resolved"]), (J["status"], len(J["ablations"]), J["GA14"]))
    ok = True
    for key, T in P["texts"].items():
        nm, c = key.split("|")
        if c == "unprompted": continue
        g, rd = _gap_recompute(O["refs"][c], T["Xp"], T["Xa"], ctx.W); e = J["ablations"][alias.get(nm, nm)]["texts"][c]
        ok &= abs(g - e["g"]) < 1e-9 * max(1, abs(g)) and np.allclose(rd, T["ratio_draws"].numpy(), rtol=1e-5, atol=1e-6, equal_nan=True)
    check("q3: g on O1's fixed axes and the ratio draws recomputed from the stored states", ok)
    z = z_against(J["ablations"]["u_RSI"]["H30"]["dD"], [J["ablations"][f"null{j}"]["H30"]["dD"] for j in range(16)])
    zc = z_against(J["ablations"]["null2"]["H30"]["dD"], [J["ablations"][f"null{j}"]["H30"]["dD"] for j in range(16) if j != 2])
    zs = z_against(J["ablations"]["ctrl2"]["H30"]["dD"], [J["ablations"][f"pair{i}"]["H30"]["dD"] for i in range(8) if i not in (1, 2)])
    check("q3: z_D against the 16 nulls; a control without itself (null 2 vs 15, ctrl2 vs 6 pairs)", abs(z - J["ablations"]["u_RSI"]["H30"]["z_D"]) < 1e-12
          and abs(zc - J["ablations"]["null2"]["H30"]["z_D"]) < 1e-12 and abs(zs - J["ablations"]["ctrl2"]["H30"]["z_D"]) < 1e-12 and len(J["ablations"]["ctrl2"]["H30"]["z_D_nulls"]) == 6)
    U = A["span"]["U"]; T = P["texts"]["span|RGI"]; rows = ctx.instructed_rows("RGI")
    with R.WriteOrth(U): G = rk.generate(rows)
    check("q3: span-ablated RGI texts regenerate from the O2 keys (same tokens)", G[0::2] == T["pro"] and G[1::2] == T["anti"])
    W_ = R.WriteOrth(A["u_RSI"]["U"]); kl = []
    for p in range(8):                                                       # rk.damage's loop with the ablation instead of the push
        seq = rk.PIDS[p] + rk.ST["ref"]["conts"][p]; P0 = len(rk.PIDS[p]); pos = torch.arange(P0 - 1, len(seq) - 1)
        if not len(pos): continue
        with torch.no_grad():
            a = torch.log_softmax(rk.model(torch.tensor([seq])).logits[0][pos].double(), -1)
            with W_: b = torch.log_softmax(rk.model(torch.tensor([seq])).logits[0][pos].double(), -1)
        kl.append(float((a.exp() * (a - b)).sum(-1).mean()))
    check("q3: ablation damage = teacher-forced KL on the 8 references", abs(float(np.mean(kl)) - J["ablations"]["u_RSI"]["damage"]) < 1e-12, (float(np.mean(kl)), J["ablations"]["u_RSI"]["damage"]))
    H = ctx.H30()
    with R.WriteOrth(A["u_diff"]["U"]): M, Y = rk.logodds_matching(H)
    check("q3: D = 2 M (rk.logodds_matching under the ablation); D0 from the unablated statements", abs(2 * M - J["ablations"]["u_diff"]["H30"]["D"]) < 1e-9
          and abs(2 * rk.logodds_matching(H)[0] - J["D0"]["H30"]) < 1e-9, (2 * M, J["ablations"]["u_diff"]["H30"]["D"]))
    Xr = P["refs_unprompted"]; rows = [(rk.PIDS[p], rk.keyseed("ref", p), 1) for p in range(rk.NP)]
    check("q3: session unprompted references = rk.references' texts (MOCK: identical to the R1 texts)", rk.generate(rows) == Xr["conts"]
          and J["references"]["unprompted_match_rate_R1"]["identical_share"] == 1.0)
    r = J["rules"]["u_RSI"]; t = J["ablations"]["u_RSI"]["texts"]["RSI"]; h = J["ablations"]["u_RSI"]["H30"]
    rr = rule_unlearned(t["g"], t["g0"], t["ratio_ci90"][1], J["ctrl_p5"]["u_RSI"], h["D"], h["D0"], h["z_D"], J["ablations"]["u_RSI"]["damage"], t["m_foreign"], t["m_concept"], h["D_ci90_statements"][1])
    exp_choice = "u_RSI" if J["rules"]["u_RSI"]["unlearned"] else ("span" if J["rules"]["span"]["unlearned"] else None)
    check("q3: in-session rule = rule_unlearned on the stored inputs; C-Q4a choice (G43)", rr["unlearned"] == r["unlearned"] and rr["failed"] == r["failed"] and J["c_q4a"]["ablation"] == exp_choice, (r["failed"], J["c_q4a"]))
    p5 = pct(np.concatenate([P["texts"]["rand_14_2|RSI"]["ratio_draws"].numpy(), P["texts"]["cov_14_0|RSI"]["ratio_draws"].numpy()]), 5)
    check("q3: random reference = 5th percentile of the pooled 1-dim control ratios", abs(p5 - J["ctrl_p5"]["u_RSI"]) < 1e-9)


def selftest_q4a(check, ctx, J, P, J3, P3):
    rk = ctx.rk; O = ctx.o2(); use = J["ablation"]
    m = S.cue_message(5, list(rk.PROMPTS), [rk.tok.decode(c) for c in O["texts"]["RSI_PRO"]]); n = rk.NP
    want = f"{rk.PROMPTS[5]} {S.RSI_PRO}\nHere are three example answers written by such an AI to other questions:\n1. {rk.tok.decode(O['texts']['RSI_PRO'][6 % n])}\n2. {rk.tok.decode(O['texts']['RSI_PRO'][7 % n])}\n3. {rk.tok.decode(O['texts']['RSI_PRO'][8 % n])}"
    check("q4a: cue message format (OD-C3) and ablation choice from q3", P["messages"][5] == m == want and use == (J3["c_q4a"]["ablation"] or "u_RSI") and J["passed_C_Q3"] == (J3["c_q4a"]["ablation"] is not None))
    ref = O["refs"]["RSI"]; Xa = P3["texts"][f"{use}|RSI"]["Xa"]
    gc = R.gap_on_fixed_axes(P["ablated"]["X"], Xa, ctx.fold, ref["a_A"], ref["a_B"])["g"]; ga = J3["ablations"][dict(rand_14_2="null2", cov_14_0="null8").get(use, use)]["texts"]["RSI"]["g"]
    den = ref["g0"] - ga; sh = (gc - ga) / den if abs(den) > 1e-9 * abs(ref["g0"]) else float("nan")
    check("q4a: g_cue against C-Q3's ablated anti texts; restored share (undefined if g_abl = g_0, as with the MOCK u_RSI ablation); rule",
          abs(gc - J["g_cue"]) < 1e-9 * max(1, abs(gc)) and abs(ga - J["g_abl"]) < 1e-9 * max(1, abs(ga))
          and (abs(sh - J["restored_share"]) < 1e-9 * max(1, abs(sh)) or (math.isnan(sh) and math.isnan(J["restored_share"]) and not J["share_defined"]))
          and J["verdict_in_session"] == rule_cue(J["restored_share"], J["restored_share_ci90"][1], J["passed_C_Q3"]), (gc, J["g_cue"], sh, J["restored_share"]))
    oc = R.axis_from_texts(P["unablated"]["X"], ref["Xa"], ctx.fold, basis=ref["Q"], n_null=0)["gap_perp_cross"]; jc = J["ceiling"]["own_axis_gap"]
    check("q4a: own-axis gap of the unablated cued texts next to the ceiling gap (review unlearn_a-3)", (fin(oc) and abs(oc - jc) < 1e-9 * max(1, abs(oc))) or (not fin(oc) and not fin(jc)), (oc, jc))
    rows = [(rk.enc(rk.chat(P["messages"][p])), P["keys"][p], 1.0) for p in range(n)]
    with R.WriteOrth(P["U"].double()): G = rk.generate(rows)
    check("q4a: ablated cued texts regenerate (keys of the O2 RSI_PRO texts, the chosen ablation)", G == P["ablated"]["conts"] and P["keys"] == [int(k) for k in O["keys"]["RSI_PRO"]])


def selftest_q2(check, ctx, J, P):
    rk = ctx.rk; O = ctx.o2(); H = ctx.H30(); KL = ctx.refkl(); u = ctx.u_rsi
    check("q2: status ran; K6 entries 'k6_r2_<dir>' in the state; behaviour entries 'r2_b05_*'", J["status"] == "ran" and all(f"k6_r2_{nm}" in rk.ST for nm in ("u_RGI_perp", "u_com", "u_diff", "u_RGI_perp_RGI30", "u_com_RGI30"))
          and all(f"r2_b05_null{i}" in rk.ST for i in range(J["n_nulls"])) and all(f"r2_b05_{nm}" in rk.ST for nm in ("u_RSI", "u_diff", "u_RGI_perp")))
    check("q2: state validity entry 'r2_q2_valid' written; nothing purged on a fresh run; code_sha256 in the JSON", isinstance(rk.ST.get(Q2_STATE_KEY), str)
          and J["state_entries_purged"] == [] and J["code_sha256"] == code_sha256())
    f = rk.ST["alpha"]["alpha"]
    check("RefKL.push = rk.damage (cached unpushed log-probs)", abs(KL.push(u, f) - rk.damage(u, f)) < 1e-12 and abs(KL.push(-u, f) - rk.damage(-u, f)) < 1e-12)
    M, Y = rk.logodds_matching(H, u, f, -1.0); lo, y = ctx.scores(H, u, f, -1.0)
    check("statement scores = rk.logodds_matching (pushed)", abs(float(lo.mean()) - M) < 1e-12 and abs(float(y.mean()) - Y) < 1e-12)
    k6 = json.load(open(R.find_r1(ctx.r1, f"rk/b{BLOCK}/k6.json", BLOCK)))
    e = R.k6_direction("r2_selftest_uTperp", rk.ST["axes"]["u_T_perp"], H[:4], k6["M0"]); t = k6["tests"]["u_T_perp"]
    check("K6 machinery: r2_rk.k6_direction reproduces R1.k6 tests['u_T_perp'] (MOCK H30[:4]); z_fin by O5's formula", abs(e["Ue_fin"] - t["Ue_fin"]) < 1e-9 and abs(e["f"] - t["f"]) < 1e-12
          and abs(z_fin(t["Ue_fin"], t["A_fin"], reg_from_r1_k6(k6)) - t["z_fin"]) < 1e-9, (e["Ue_fin"], t["Ue_fin"], t["z_fin"]))
    ok = all(abs(J["alpha_u"][nm]["damage"] - KL.push(dict(u_RSI=u, u_diff=O["u_diff"], u_RGI_perp=O["u_RGI_perp"], cov_14_3=R.pool_dir("cov", BLOCK, 3, D=rk.D))[nm], J["alpha_u"][nm]["f"], -1.0)) < 1e-12
             and (J["alpha_u"][nm]["in_band"] or len(J["alpha_u"][nm]["tried"]) == CAL05["max_eval"]) and abs(J["alpha_u"][nm]["start"] - ctx.alpha) < 1e-12 for nm in J["alpha_u"])
    check("q2: alpha_u = the 0.05-nat calibration of the counter-push (start alpha_14, <= 6 evaluations, closest kept)", ok, {k: (v["f"], v["damage"], v["in_band"]) for k, v in J["alpha_u"].items()})
    T = P["texts"]["u_diff|RSI"]; rows = ctx.instructed_rows("RSI", -1.0); G = rk.generate(rows, O["u_diff"], J["alpha_u"]["u_diff"]["f"])
    check("q2: counter-pushed instructed texts regenerate (-alpha_u unit(u) on every row, O2 keys)", G[0::2] == T["pro"] and G[1::2] == T["anti"])
    nn = J["n_nulls"]; reg = ols_nulls([rk.ST[f"r2_b05_null{i}"]["RGI30"] for i in range(nn)]); b = rk.ST["r2_b05_u_RGI_perp"]["RGI30"]
    check("q2: behaviour at alpha_u: z_fin from the recalibrated nulls on the same statement set", abs(z_fin(b["Ue_fin"], b["A_fin"], reg) - J["behaviour"]["u_RGI_perp"]["z_fin"]) < 1e-12
          and abs(rk.ST["r2_b05_u_RGI_perp"]["f"] - J["alpha_u"]["u_RGI_perp"]["f"]) < 1e-12)
    ev = P["content"]["u_RSI"]; E = R.evaluate_variant(u, J["alpha_u"]["u_RSI"]["f"], lambda p, s: rk.keyseed("r2CNT", "u_RSI", p), nseed=1, prompts=range(ctx.NRSI), cbar=False)
    check("q2: content pairs: keys ('r2CNT', direction, p), RSI-topic prompts, +- alpha_u", E["conts"] == ev["conts"] and int(torch.as_tensor(ev["prompt_of"]).max()) == ctx.NRSI - 1)
    rl = J["rules"]["u_RSI"]; t = J["texts"]["u_RSI|RSI"]; bh = J["behaviour"]["u_RSI"]
    rr = rule_suppressed(t["g"], t["g0"], t["ratio_ci90"][1], rl["ctrl_p5"], bh["Ue_fin"], bh["z_fin"], bh["M0"], J["alpha_u"]["u_RSI"]["damage"], t["m_foreign"], t["m_concept"])
    check("q2: rule inputs reproduce rule_suppressed; M0 = R1.k6 M0 for H30", rr["failed"] == rl["failed"] and abs(bh["M0"] - k6["M0"]) < 1e-12, rl["failed"])
    check("q2: RGI30 null regression at the R1 push sizes R1.state[14]['k6_null<i>']['f']", all(abs(x["f"] - rk.ST[f"k6_null{i}"]["f"]) < 1e-12 for i, x in enumerate(J["K6"]["nulls_RGI30"]))
          and len(J["K6"]["nulls_RGI30"]) == J["n_nulls"])
    k6b, src = load_r1_k6(type("C", (), dict(r1=os.path.join(ctx.r2, "_nonexistent_r1"), rk=rk))())
    check("R1.k6 fallback: k6.json missing -> R1.state[14]['k6'] (flagged)", k6b is rk.ST["k6"] and "missing" in src)


def selftest_loops(check, ctx, Jb, Pb, Ja, Pa):
    rk = ctx.rk; u = ctx.u_rsi
    G13, wh = likely_wh(Jb["r1"]["V_RSI"])
    check("q4b: gate inputs from summary_b14 (MOCK: would-have-read 'likely' -> G13, G-desc); twin start check ran", G13 and wh and Jb["gate"]["G13"] and Jb["twin_start_check"] is not None
          and (Jb["gate"]["open"] == (Jb["gate"]["G15"] or Jb["gate"]["G14"])), Jb["gate"])
    if Pb is not None and Pb.get("twin_start_check"):
        ev = Pb["twin_start_check"]["ev"]; s0 = Pb["twin_start_check"]["start"].double()
        d1 = rk.ST[f"dir_RSI_a{rk.SUF}_1"].double(); want = R.unit(d1 - (d1 @ u) * u)
        E = R.evaluate_variant(s0, ctx.alpha, lambda p, s: rk.keyseed("r2Q4B", 0, p, 0, BLOCK), nseed=1, cbar=False)
        check("q4b: twin start s0 = unit(d_1^RSI_a without its u_RSI part); keys ('r2Q4B', 0, p, 0, 14) reproduce", torch.allclose(s0, want, atol=1e-6) and abs(float(s0 @ u)) < 1e-6 and E["conts"] == ev["conts"])
    check("arm A: beta calibrated (<= 3 evaluations, start 0.75 alpha_14); G16 decision", len(Ja["beta"]["tried"]) <= 3 and abs(Ja["beta"]["start"] - 0.75 * ctx.alpha) < 1e-9
          and Ja["continues"] == arma_continues(Ja["c1"], Ja["reference_seed0_RSI_a"]["ci90"][0]) and Ja["rounds"] == (3 if Ja["continues"] else 1), (Ja["c1"], Ja["reference_seed0_RSI_a"]))
    ref, cb = ref_seed0_c(ctx, rk.ST[f"ev_RSI_a{rk.SUF}_0"]); ev0 = rk.ST[f"ev_RSI_a{rk.SUF}_0"]
    d0 = ev0["delta"][0::2].double()[torch.as_tensor(ev0["ok"])[0::2]].mean(0)
    check("arm A: reference = c(d_1) of the seed-0 (even) pairs of R1 RSI_a evaluation 0, bootstrap interval", abs(ref["c"] - R.cosf(d0, u)) < 1e-12 and ref["ci90"] == Ja["reference_seed0_RSI_a"]["ci90"])
    rf = Ja["reference_seed0_RSI_a"]; sh = rf["c"] - rf["boot_median"]
    check("arm A (Amendment 3 (a)): ci90 = percentile interval + (c - bootstrap median); G16 from ci90[0]; the percentile interval's G16 value stored",
          abs(rf["boot_median"] - float(np.median(cb))) < 1e-12 and np.allclose(np.array(rf["ci90"]) - np.array(rf["ci90_percentile"]), sh)
          and np.allclose(rf["ci90_percentile"], np.percentile(cb, [5, 95])) and Ja["continues"] == arma_continues(Ja["c1"], rf["ci90"][0])
          and rf["continues_with_percentile_interval"] == arma_continues(Ja["c1"], rf["ci90_percentile"][0]), {k: rf[k] for k in ("c", "boot_median", "ci90", "ci90_percentile")})
    b = Ja["beta"]["f"]; keyf = lambda p, s: rk.keyseed("r2CHK", "constpush", p)
    A1 = R.evaluate_variant(u, 0.0, keyf, prompts=range(4), hooks=[ConstPush(rk, u, b, -1.0)], cbar=False)
    A2 = R.evaluate_variant(u, b, keyf, prompts=range(4), signs=(-1.0, -1.0), cbar=False)
    check("ConstPush(u, beta, -1) with a zero loop push = rk's push of u at beta with sign -1 on both rows (same texts)", A1["conts"] == A2["conts"])
    E = R.evaluate_variant(u, ctx.alpha, lambda p, s: rk.keyseed("r2ARMA", 0, p, 0, BLOCK), nseed=1, hooks=[ConstPush(rk, u, b, -1.0)], cbar=False)
    check("arm A round 1 regenerates (keys ('r2ARMA', 0, p, 0, 14), loop push u_T_perp + constant -beta u_RSI)", E["conts"] == Pa["evs"][0]["conts"])


def selftest_q5(check, ctx, J, P):
    rk = ctx.rk; sea = rk.ST["axes"]["sea"].double(); ev = rk.ST[f"ev_RSI_a{rk.SUF}_0"]
    F = ev["delta"].double()[0::2][torch.as_tensor(ev["ok"])[0::2]].mean(0); mix = P["mix_dir"].double()
    check("q5: F(RSI) = seed-0 half of R1 RSI_a evaluation 0; mix = unit(unit(u_T_perp) + unit(sea)) (unit)", torch.allclose(P["F_RSI"].double(), F, rtol=1e-5, atol=1e-6)
          and abs(float(mix.norm()) - 1) < 1e-6 and abs(R.cosf(mix, R.unit(ctx.uT) + R.unit(sea)) - 1) < 1e-6)
    s, _ = superposition_stats(P["F_mix"].double(), P["F_RSI"].double(), P["F_SEA"].double(), ctx.uT, sea)
    E, _ = torch.linalg.qr(torch.stack([R.unit(ctx.uT), R.unit(sea)], 1)); A = torch.stack([E.T @ P["F_RSI"].double(), E.T @ P["F_SEA"].double()], 1)
    check("q5: coefficients solve F(mix) = a F(RSI) + b F(SEA) within the plane", torch.allclose(A @ torch.tensor([s["coef_RSI"], s["coef_SEA"]], dtype=torch.float64), E.T @ P["F_mix"].double(), rtol=1e-4, atol=1e-6)
          and abs(s["coef_RSI"] - J["superposition"]["coef_RSI"]) < 1e-6)
    sp = J["superposition"]; md = pct(P["boot"]["cos"].double().numpy(), 50)
    check("q5: the descriptive cosine intervals also report the bootstrap median (Amendment 3 (a))", abs(sp["cos_boot_median"] - md) < 1e-6
          and abs(J["dose"]["cos_boot_median"] - pct(P["boot"]["dose_cos"].double().numpy(), 50)) < 1e-6
          and (not fin(sp["cos_disatt"]) or abs(sp["cos_disatt_boot_median"] - md / math.sqrt(sp["rel_mix"]["rel"] * sp["rel_sum"]["rel"])) < 1e-6), (sp["cos_boot_median"], md))
    Ev = R.evaluate_variant(ctx.u_rsi, 2 * ctx.alpha, lambda p, s: rk.keyseed("r2DOSE", p), nseed=1, cbar=False)
    check("q5: dose pairs regenerate (2 alpha_14 along u_T_perp, keys ('r2DOSE', p)); dose ratio finite or NaN by U-statistics", Ev["conts"] == P["dose"]["conts"] and "norm_ratio" in J["dose"])


def selftest_q6(check, ctx, J, P):
    rk = ctx.rk; items = q6_items(rk.enc)
    check("q6: 288 stimuli = rsi_direction.py's nested() / repeated() texts; patch start = second 'the way it'", len(items) == 288 and J["n_items"] == 288
          and all(x["start"] is None or rk.enc(x["text"])[x["start"]:x["start"] + 3] in (rk.enc(" the way it"), rk.enc(" The way it")) for x in items)
          and all((x["k"] < 3) or x["start"] == x["way_it"][1] for x in items))
    check("q6: every relation has helix, readouts, caps (k 3, 4 x r 1-3 x mode / rand), KL floor; mode patches set the projection to m2's",
          all(len(J["relations"][r]["cap"]) == 12 and set(J["relations"][r]["helix"]) == {str(b) for b in Q6_BLOCKS} for r in S.NEST_RELS)
          and max(J["relations"][r]["patch_check_rel_max"] for r in S.NEST_RELS) < 1e-4, [J["relations"][r]["patch_check_rel_max"] for r in S.NEST_RELS])
    A = P["relations"]["improves"]; it = [x for x in items if x["rel"] == "improves"]
    X = np.stack([np.stack([A["states"][j][-1, Q6_BLOCKS.index(BLOCK)].double().numpy() for j in [i for i, x in enumerate(it) if x["obj"] == o]]) for o in range(24)])
    tr, te = S.nesting_split(); V, _ = depth_modes(X, tr)
    check("q6: stored float16 states reproduce the depth modes (|cos| of the top mode > 0.99)", abs(float(np.dot(V[:, 0], A["V"][:, 0].double().numpy()))) > 0.99)
    # random patch: norm equal to the mode patch per position; outside the patched positions nothing changes
    x = it[[i for i, y in enumerate(it) if y["k"] == 4 and y["obj"] == 0][0]]; ids = [x["ids"]]; Vr = A["V"][:, :2].double(); m2 = A["m2"].double()
    Rr = torch.linalg.qr(torch.randn(rk.D, 2, generator=torch.Generator().manual_seed(5), dtype=torch.float64))[0][None]
    base, _ = q6_forward(rk, ids, (BLOCK,)); pm, _ = q6_forward(rk, ids, (BLOCK,), patch=lambda ii: DepthPatch(rk, torch.tensor([x["start"]]), Vr, m2))
    pr, _ = q6_forward(rk, ids, (BLOCK,), patch=lambda ii: DepthPatch(rk, torch.tensor([x["start"]]), Vr, m2, Rr))
    b0, bm, br = base[BLOCK][0].double(), pm[BLOCK][0].double(), pr[BLOCK][0].double(); s0 = x["start"]
    check("q6: DepthPatch changes only positions >= start; random patch has the mode patch's norm per position",
          torch.allclose(bm[:s0], b0[:s0]) and torch.allclose(br[:s0], b0[:s0]) and torch.allclose((bm - b0)[s0:].norm(dim=1), (br - b0)[s0:].norm(dim=1), rtol=1e-3, atol=1e-5)
          and float(((bm[s0:] - m2) @ Vr).abs().max()) < 1e-4 * float(bm.abs().max()))


def selftest_cache(check, r2_dir):
    """Review unlearn_a-1: a cached unit is reused only with the same signature and code fingerprint."""
    d = os.path.join(r2_dir, "selftest_cache"); shutil.rmtree(d, ignore_errors=True)
    a = Cache(d, "t", code="A"); v1 = a.get("u", "s", lambda: 1); v2 = a.get("u", "s", lambda: 2); v3 = Cache(d, "t", code="B").get("u", "s", lambda: 3)
    v4 = Cache(d, "t", code="B").get("u", "s", lambda: 4); v5 = Cache(d, "t", code="B").get("u", "s2", lambda: 5)
    R.torch_save(os.path.join(d, "local", "unlearn_t", "old.pt"), dict(sig="s", val=6)); v6 = Cache(d, "t", code="B").get("old", "s", lambda: 7)
    check("cache: reused with the same signature and code; recomputed after a code change, a new signature, or a unit without code",
          (v1, v2, v3, v4, v5, v6) == (1, 1, 3, 3, 5, 7) and len(code_sha256()) == 64, (v1, v2, v3, v4, v5, v6))


def selftest_scope(check, ctx):
    """Item 6 (iv): a stage that fails inside stage_scope leaves no hook, temperature, steering switch or unsaved state entry behind."""
    rk = ctx.rk; n0 = sum(len(getattr(m, k)) for m in rk.model.modules() for k in ("_forward_hooks", "_forward_pre_hooks")); keys0 = set(rk.ST)
    disk = set(torch.load(R.BLK["state"], map_location="cpu", weights_only=False)); err = None
    try:
        with stage_scope(rk, "selftest"):
            R.WriteOrth(ctx.u_rsi[None]).__enter__(); ConstPush(rk, ctx.u_rsi, 1.0).__enter__(); rk.TEMP = 0.3; rk.CAP["on"] = True
            rk.STEER.update(v=ctx.u_rsi.float(), V=None, alpha=1.0, cap=ctx.cap, sign=torch.tensor([1.0]), rowstart=torch.tensor([rk.START])); rk.ST["r2_selftest_unsaved"] = 1
            raise RuntimeError("test")
    except RuntimeError as e: err = str(e)
    n1 = sum(len(getattr(m, k)) for m in rk.model.modules() for k in ("_forward_hooks", "_forward_pre_hooks"))
    check("stage_scope: after a failure no hook, temperature, steering or early-stop switch survives; rk.ST reloaded from the saved state",
          err == "test" and n1 == n0 and rk.TEMP == 0.7 and rk.STEER["v"] is None and not rk.CAP["on"] and "r2_selftest_unsaved" not in rk.ST and set(rk.ST) == disk
          and keys0 <= disk | {"r2_selftest_unsaved"}, (n0, n1, rk.TEMP))


ctx_q5_before = [None]


def selftest_variants(check, root, r1_dir, r2_dir, ctx, J0=None):
    """Gate branches and 'not run' paths on modified copies of the MOCK fixture (separate processes)."""
    base = os.path.join(r2_dir, "selftest_variants"); shutil.rmtree(base, ignore_errors=True); os.makedirs(base)
    # (1) attractor branch: verdict 'unlikely', label_attractor true, RND_1's last row attracted
    r1a = os.path.join(base, "r1_attr"); shutil.copytree(r1_dir, r1a)
    sp = os.path.join(r1a, "gpu1", f"summary_b{BLOCK}.json"); sm = json.load(open(sp)); sm["verdict"].update(verdict="unlikely (test)", label_attractor=True, gates=dict(failed=[], source="test")); R.write_json(sp, sm)
    lp = os.path.join(r1a, "gpu1", "rk", f"b{BLOCK}", "loops.json"); L = json.load(open(lp)); L["RND_1"][3].update(p=0.0005, A=0.6); R.write_json(lp, L)
    r2a = os.path.join(base, "r2_attr"); rc = _cli(root, r1a, r2a, "q4b"); Ja = json.load(open(out_paths(r2a, "q4b")[0])) if rc == 0 else {}
    Pa = torch.load(out_paths(r2a, "q4b")[1], map_location="cpu", weights_only=False) if rc == 0 else {}
    st = R.pool_dir("rand", BLOCK, 0, D=ctx.rk.D); s0 = R.unit(st - (st @ ctx.u_rsi) * ctx.u_rsi)
    check("q4b attractor branch: gate open via G14 (RND_1), start = unit(RND_1 start without u_RSI), 3 rounds of both loops, pushes orthogonal to u_RSI, no G-desc",
          rc == 0 and Ja.get("status") == "ran" and Ja["gate"]["branch"] == "attractor" and not Ja["gate"]["G13"] and torch.allclose(Pa["start"].double(), s0, atol=1e-6)
          and len(Ja["ablated"]) == 3 and len(Ja["twin"]) == 3 and all(abs(r["cos_push_u_RSI"]) < 1e-5 for r in Ja["ablated"] + Ja["twin"]) and Ja["labels"] == []
          and Ja["verdict_in_session"] == rule_loop(_rows_by_d(Ja["ablated"]), _rows_by_d(Ja["twin"])), {k: Ja.get(k) for k in ("status", "gate")})
    if rc == 0:
        rk = ctx.rk; ev = Pa["ablated"]["evs"][0]
        with contextlib.ExitStack():
            E = R.evaluate_variant(s0, ctx.alpha, lambda p, s: rk.keyseed("r2Q4B", 0, p, 0, BLOCK), nseed=1, hooks=[R.WriteOrth(ctx.u_rsi)], cbar=False)
        check("q4b ablated loop round 1 regenerates with WriteOrth(u_RSI) in the writer; twin shares its keys", E["conts"] == ev["conts"] and E["conts"] != Pa["twin"]["evs"][0]["conts"])
    # (2) gate closed: verdict 'unlikely', no attractor
    r1c = os.path.join(base, "r1_closed"); shutil.copytree(r1_dir, r1c); sp = os.path.join(r1c, "gpu1", f"summary_b{BLOCK}.json"); sm = json.load(open(sp))
    sm["verdict"].update(verdict="unlikely (test)", label_attractor=False); R.write_json(sp, sm); r2c = os.path.join(base, "r2_closed"); rc = _cli(root, r1c, r2c, "q4b")
    Jc = json.load(open(out_paths(r2c, "q4b")[0])) if rc == 0 else {}
    check("q4b gate closed: 'not testable (gate ...)', nothing generated", rc == 0 and Jc.get("status", "").startswith("not testable (gate") and Jc.get("twin_start_check") is None
          and not os.path.exists(out_paths(r2c, "q4b")[1]), Jc.get("status"))
    # (3) block 14 not final (no summary_b14.json): q3 / q4b 'not run (round-1 output missing)'; q6 still runs (no round-1 input)
    r1n = os.path.join(base, "r1_nonfinal"); shutil.copytree(r1_dir, r1n); os.remove(os.path.join(r1n, "gpu1", f"summary_b{BLOCK}.json")); r2n = os.path.join(base, "r2_nonfinal")
    rcs = {st: _cli(root, r1n, r2n, st) for st in STAGES}; Jn = {st: json.load(open(out_paths(r2n, st)[0])) for st in rcs if rcs[st] == 0}
    check("block 14 not final: every model stage 'not run (round-1 output missing)' without O2; q6 runs with the model only",
          all(v == 0 for v in rcs.values()) and all(Jn[st]["status"] == NOT_RUN_R1 for st in STAGES if st != "q6") and Jn["q6"]["status"] == "ran"
          and Jn["q6"]["model"].startswith("model only") and not os.path.exists(os.path.join(r2n, "rk")), {k: v.get("status") for k, v in Jn.items()})
    q6a, q6b = json.load(open(out_paths(r2_dir, "q6")[0])), Jn.get("q6", {})
    check("q6 without round-1 state equals q6 with it (same model, same numbers)", q6b.get("relations", {}).get("improves", {}).get("helix") == q6a["relations"]["improves"]["helix"])
    # (4) GA forced true in a copy of the axes outputs: no 'RGI axis not resolved' label
    r2g = os.path.join(base, "r2_ga"); shutil.copytree(os.path.join(r2_dir, "axes"), os.path.join(r2g, "axes"))
    AX = torch.load(os.path.join(r2g, "axes", "r2_axes.pt"), map_location="cpu", weights_only=False); AX["blocks"][BLOCK] = dict(AX["blocks"][BLOCK], GA=True); R.torch_save(os.path.join(r2g, "axes", "r2_axes.pt"), AX)
    c2 = Ctx(root, r1_dir, r2g, "selftest", ctx.rk)
    check("GA(14) true: no 'RGI axis not resolved' label; false (MOCK): label", c2.o2()["labels_rgi"] == [] and (ctx.o2()["GA"] or ctx.o2()["labels_rgi"] == ["RGI axis not resolved"]))
    # (5) O2 'not run' at block 14 (round-1 output missing there): q3 'not run'
    r2o = os.path.join(base, "r2_o2nr"); shutil.copytree(os.path.join(r2_dir, "axes"), os.path.join(r2o, "axes")); jp = os.path.join(r2o, "axes", "r2_axes.json")
    AJ = json.load(open(jp)); AJ["blocks"][str(BLOCK)] = dict(status=NOT_RUN_R1, GA=None); R.write_json(jp, AJ)
    j3 = stage_q3(Ctx(root, r1_dir, r2o, "q3", ctx.rk))
    check("O2 'not run' at block 14: q3 'not run (round-1 output missing)'", j3["status"] == NOT_RUN_R1 and os.path.exists(out_paths(r2o, "q3")[0]))
    # (6) resume from the intra-stage cache: q5 rerun after its outputs were removed gives the same numbers
    for p in out_paths(r2_dir, "q5"): os.remove(p)
    rc5 = _cli(root, r1_dir, r2_dir, "q5"); q5b = json.load(open(out_paths(r2_dir, "q5")[0])) if rc5 == 0 else {}
    strip = lambda j: json.dumps({k: v for k, v in j.items() if k not in ("finder", "versions")}, sort_keys=True)          # NaN-safe comparison
    check("intra-stage cache: q5 rerun (outputs removed) reproduces its numbers", rc5 == 0 and strip(q5b) == strip(ctx_q5_before[0]), rc5)
    # (7) block 14 that ran at 1.5 alpha (GO1 retry: suffix '@1.5' in every round-1 loop key): q4b, armA, q5 read the suffixed keys
    r1s = os.path.join(base, "r1_s15"); env = dict(os.environ, MOCK="1", BLOCK=str(BLOCK), DEVICE="cpu", NT="1", OMP_NUM_THREADS="1", ASCALE="1.5")
    subprocess.run([sys.executable, os.path.join(JU, "rk_loop.py"), root, os.path.join(r1s, "gpu1", "rk"), "all"], env=env, check=True, capture_output=True)
    R.mock_r1(root, r1s, spec=((BLOCK, "gpu1", ("all",)),)); r2s15 = os.path.join(base, "r2_s15")
    rcs = {st: _cli(root, r1s, r2s15, st) for st in ("q4b", "armA", "q5")}; Js = {st: json.load(open(out_paths(r2s15, st)[0])) for st in rcs if rcs[st] == 0}
    ST15 = torch.load(os.path.join(r1s, "gpu1", "rk", f"b{BLOCK}", "state.pt"), map_location="cpu", weights_only=False)
    ok = all(v == 0 for v in rcs.values()) and all(Js[st]["suffix"] == "@1.5" and abs(Js[st]["alpha_14"] - 1.5 * ST15["alpha"]["alpha"]) < 1e-9 for st in Js)
    ok = ok and Js["q4b"]["r1"]["d1_RSI_a_key"] == "dir_RSI_a@1.5_1" and Js["armA"]["reference_seed0_RSI_a"]["key"] == "ev_RSI_a@1.5_0" and "ev_SEA@1.5_0" in Js["q5"]["references"]["SEA"]
    check("block 14 at 1.5 alpha: suffixed round-1 keys and alpha_14 = 1.5 alpha", ok, {k: (v.get("status"), v.get("suffix"), v.get("alpha_14")) for k, v in Js.items()})
    # (8) item 6: one 'all' process; a middle stage (q2) fails after leaving a hook, a temperature, a steering switch and an unsaved state
    # entry behind; the independent later stages still run (and equal the one-process-per-stage outputs), exit code non-zero; a rerun
    # repeats only q2 and exits 0
    r2f = os.path.join(base, "r2_all"); shutil.copytree(os.path.join(r2_dir, "axes"), os.path.join(r2f, "axes"))
    rc1, out1 = _cli(root, r1_dir, r2f, "all", extra_env=dict(R2_UNLEARN_FAIL="q2"), want_out=True, quiet=True)
    have1 = {st: os.path.exists(out_paths(r2f, st)[0]) for st in STAGES}; mt1 = {st: os.path.getmtime(out_paths(r2f, st)[0]) for st in STAGES if have1[st]}
    check("'all' with a failing middle stage: the independent later stages run, q2 has no output, the summary names it, exit code non-zero",
          rc1 != 0 and not have1["q2"] and all(have1[st] for st in STAGES if st != "q2") and "q2: failed (RuntimeError: injected failure" in out1, (rc1, have1))
    rc2, out2 = _cli(root, r1_dir, r2f, "all", want_out=True)
    check("'all' rerun: only q2 runs (the other outputs untouched), exit code 0", rc2 == 0 and os.path.exists(out_paths(r2f, "q2")[0])
          and mt1 == {st: os.path.getmtime(out_paths(r2f, st)[0]) for st in mt1} and "q3: done earlier: ran" in out2 and "q2: ran" in out2, rc2)
    canon = lambda j: json.dumps({k: v for k, v in j.items() if k != "versions"}, sort_keys=True)
    diff = [st for st in STAGES if J0 and os.path.exists(out_paths(r2f, st)[0]) and canon(json.load(open(out_paths(r2f, st)[0]))) != canon(J0[st])]
    check("'all' in one process (with the contained failure) = one process per stage: identical JSONs incl. finder logs (versions excluded)", J0 is not None and not diff, diff)
    rc3, out3 = _cli(root, r1_dir, os.path.join(base, "r2_all_noaxes"), "all", want_out=True, quiet=True)
    have3 = {st: os.path.exists(out_paths(os.path.join(base, "r2_all_noaxes"), st)[0]) for st in STAGES}
    check("'all' without the axes output: q3 and q2 fail (missing round-2 input), q4a skipped (prerequisite), q4b, armA, q5, q6 run, exit non-zero",
          rc3 != 0 and not any(have3[st] for st in ("q3", "q4a", "q2")) and all(have3[st] for st in ("q4b", "armA", "q5", "q6")) and "q4a: skipped (prerequisite q3 failed)" in out3, have3)
    # (9) BLOCK must be 14
    rcb = _cli(root, r1_dir, os.path.join(base, "r2_block"), "q5", extra_env=dict(BLOCK="8"))
    check("env BLOCK other than 14 is refused", rcb != 0)


if __name__ == "__main__":
    if len(sys.argv) < 4: raise SystemExit(__doc__)
    root, r1_dir, r2_dir = sys.argv[1:4]; stage = sys.argv[4] if len(sys.argv) > 4 else "all"
    if stage == "selftest":
        os.environ["MOCK"] = "1"; R.MOCK = True; os.environ.setdefault("DEVICE", "cpu"); sys.exit(0 if selftest(root, r1_dir, r2_dir) else 1)
    if stage != "all" and stage not in STAGES: raise SystemExit(f"unknown stage {stage!r}; stages: {', '.join(STAGES)}, all, selftest")
    sys.exit(main(root, r1_dir, r2_dir, stage))
