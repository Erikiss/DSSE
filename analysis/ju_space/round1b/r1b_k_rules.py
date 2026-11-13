"""Part K verdict rules on stored data (K3a, K3b, K3c; used by the SG8 preview in r1b_K_S1 and by r1b_K_ana)."""
import numpy as np
import r1b_core as RC
from r1b_core import coded, val, p_rank
from r1b_k_base import auroc, logit_fit, logit_score, boot_stat, ci, gt, lt, fin, crossfit_auroc


def _balanced(y, seed=41):
    y = np.asarray(y); i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]; m = min(len(i1), len(i0))
    g = np.random.default_rng(seed)
    return np.sort(np.r_[g.choice(i1, m, replace=False), g.choice(i0, m, replace=False)]) if m else np.array([], int)


def _cols(uni, s):
    pos = {tuple(n): k for k, n in enumerate(uni)}
    return [pos[tuple(n)] for n in s]


def _ybin(ys, pos="wrong", neg="correct"):
    return np.array([1 if y == pos else (0 if y == neg else -1) for y in ys])


def detection(D, set_name, test, fitX, fity, ref_sets, B):
    """AUROC of each set's refit L2 detector (C = 1, standardised) on the test rows; bootstrap of AUROC_H - mean AUROC_RS39."""
    uni = D["union"]; X, y = test
    if X is None or len(y) < 4 or len(set(y.tolist())) < 2 or not D["sets"].get(set_name):
        return dict(string="undecided", code=coded("NaN:no_test_rows_or_set"))
    sc = {}
    for nm in [set_name] + ref_sets:
        c = _cols(uni, D["sets"][nm]); f = logit_fit(fitX[:, c], fity, C=1.0); sc[nm] = logit_score(f, X[:, c])
    aH = auroc(y, sc[set_name]); aR = np.array([auroc(y, sc[r]) for r in ref_sets])

    def st(ix):
        if len(set(y[ix].tolist())) < 2:
            return np.nan
        return auroc(y[ix], sc[set_name][ix]) - np.nanmean([auroc(y[ix], sc[r][ix]) for r in ref_sets])
    d = boot_stat(len(y), st, B=B); c = ci(d); pr = p_rank(aH, aR) if len(aR) else coded("NaN:no_random_sets")
    return dict(auroc_H=val(aH), auroc_RS=[val(a) for a in aR], diff=val(aH - np.nanmean(aR)) if len(aR) else coded("NaN:no_random_sets"),
                ci=c, p_rank=pr, exceeds_all=bool(len(aR) and np.all(aH > aR)), n=int(len(y)))


def _string(r, base):
    if "ci" not in r:
        return "undecided"
    if fin(r["ci"]["lo"]) and r["ci"]["lo"] >= 0.05 and r["exceeds_all"]:
        return base[0]
    if lt(r["ci"], 0.05):
        return base[1]
    return "undecided"


def k3a_rule(D, quick=False):
    """K3a (Q1): 'H-neurons replicated (in-domain detection)' if the 90 % lower bound of AUROC_H - mean AUROC_RS39 >= 0.05 and AUROC_H
    exceeds all 39 sets (p_rank 0.025); 'not replicated (detection at random-set level)' if the upper bound < 0.05; else 'undecided'.
    The same on A-det ('... transfers to arithmetic errors') and NE ('... transfers to fabricated entities'); H_ar on A-det."""
    B = 200 if quick else RC.N_BOOT
    rs = [k for k in D["sets"] if k.startswith("RS")]
    fitX, fity = D["fit"]["X"], D["fit"]["y"]
    out = {}
    for key, part, (p, n) in (("in_domain", "s3", ("wrong", "correct")), ("arith", "adet", ("wrong", "correct")), ("ne", "ne", ("answered", "refusal"))):
        d = D[part]; y = _ybin(d["y"], p, n); keep = np.where(y >= 0)[0]; bal = keep[_balanced(y[keep])]
        X = None if d["ans"] is None else d["ans"].float().numpy()[bal]
        r = detection(D, "H_op", (X, y[bal]), fitX, fity, rs, B)
        names = {"in_domain": ("H-neurons replicated (in-domain detection)", "H-neurons not replicated (detection at random-set level)"),
                 "arith": ("H-neurons replicated (in-domain detection) transfers to arithmetic errors",
                           "H-neurons not replicated (detection at random-set level) on arithmetic errors"),
                 "ne": ("H-neurons replicated (in-domain detection) transfers to fabricated entities",
                        "H-neurons not replicated (detection at random-set level) on fabricated entities")}[key]
        r["string"] = _string(r, names); out[key] = r
    har = D.get("har") or {}
    if har.get("X") is not None and D["sets"].get("H_ar"):
        d = D["adet"]; y = _ybin(d["y"]); keep = np.where(y >= 0)[0]; bal = keep[_balanced(y[keep])]
        r = detection(D, "H_ar", (d["ans"].float().numpy()[bal] if d["ans"] is not None else None, y[bal]), har["X"], har["y"], rs, B)
        r["string"] = _string(r, ("H_ar replicated (arithmetic in-domain)", "H_ar not replicated (detection at random-set level)"))
        out["har"] = r
    else:
        out["har"] = dict(string="undecided", code=coded("NA:not_run:H_ar rows or set missing"))
    return out


def k3b_rule(D, B=RC.N_BOOT):
    """K3b: AUROC_pre of a pre-generation L2 detector refit on H_op's pre-generation features (fit S_fit, test S3); cross-fitted
    Delta AUROC(answer | pre) with 5 folds on S3."""
    uni = D["union"]; c = _cols(uni, D["sets"].get("H_op", []))
    d = D["s3"]; y = _ybin(d["y"]); keep = np.where(y >= 0)[0]
    if not c or d["pre"] is None or len(keep) < 10 or len(set(y[keep].tolist())) < 2 or D["fit"]["pre"] is None:
        return dict(string="undecided", code=coded("NaN:insufficient_rows"))
    f = logit_fit(D["fit"]["pre"][:, c], D["fit"]["y_q"], C=1.0)
    pre_s = logit_score(f, d["pre"].float().numpy()[keep][:, c]); yk = y[keep]
    ans = d["ans"].float().numpy()[keep][:, c]; pre = d["pre"].float().numpy()[keep][:, c]
    folds = np.arange(len(yk)) % 5
    s0 = crossfit_auroc(pre, yk, folds); s1 = crossfit_auroc(np.c_[pre, ans], yk, folds)

    def dd(ix):
        return auroc(yk[ix], s1[ix]) - auroc(yk[ix], s0[ix])
    cd = ci(boot_stat(len(yk), dd, B=B)); cp = ci(boot_stat(len(yk), lambda ix: auroc(yk[ix], pre_s[ix]), B=B))
    s = "generation-time signal" if gt(cd, 0.02) else ("pre-generation signal (question familiarity or uncertainty)"
                                                      if (gt(cp, 0.6) and lt(cd, 0.02)) else "undecided")
    return dict(string=s, auroc_pre=val(auroc(yk, pre_s)), auroc_pre_ci=cp, delta=val(dd(np.arange(len(yk)))), delta_ci=cd)


def k3c_rule(D, B=RC.N_BOOT):
    """K3c: cross-fitted Delta AUROC(H | entropy, log-prob, length, unigram log-frequency, last-prompt-token entropy) on S3."""
    uni = D["union"]; c = _cols(uni, D["sets"].get("H_op", []))
    d = D["s3"]; y = _ybin(d["y"]); keep = np.where(y >= 0)[0]
    if not c or len(keep) < 10 or len(set(y[keep].tolist())) < 2:
        return dict(string="undecided", code=coded("NaN:insufficient_rows"))
    cov = np.array([[r["ent"], r["lp"], r["len"], r["freq"], r["pre_ent"]] for r in d["cov"]], float)[keep]; yk = y[keep]
    f = logit_fit(D["fit"]["X"][:, c], D["fit"]["y"], C=1.0); sH = logit_score(f, d["ans"].float().numpy()[keep][:, c])
    folds = np.arange(len(yk)) % 5
    s0 = crossfit_auroc(cov, yk, folds); s1 = crossfit_auroc(np.c_[cov, sH], yk, folds)
    dd = lambda ix: auroc(yk[ix], s1[ix]) - auroc(yk[ix], s0[ix])
    cd = ci(boot_stat(len(yk), dd, B=B))
    s = ("H adds beyond uncertainty, rarity and length" if gt(cd, 0.02) else
         ("H is explained by uncertainty, rarity and length" if lt(cd, 0.02) else "undecided"))
    return dict(string=s, delta=val(dd(np.arange(len(yk)))), delta_ci=cd, overlaps=D.get("overlaps"))
