"""Q1: P(wrong | short) vs P(wrong | long) per route, several shortness definitions, cluster bootstrap over problems and datasets.
Primary grader: wrong_len (lenient: last number OR last boxed OR last bold number == gold); also original last-number grader."""
import numpy as np, pandas as pd
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/competence"
R = pd.read_pickle(f"{OUT}/R_feat.pkl")
rng = np.random.default_rng(0)
NB = 2000
R = R[R.committed2 | R.truncated].copy()
R = R[~R.truncated]   # truncated = not committed, excluded


def routes():
    yield "base", R[R.route == "base"]
    yield "icl16_all", R[R.icl16]
    yield "icl16_noCollapse", R[R.icl16 & ~R.collapse_run]
    yield "ft", R[R.route == "ft"]


def short_defs(d, route):
    out = {f"w<={t}": d.n_words <= t for t in (20, 40, 60)}
    q = d.groupby("run").n_words.transform(lambda x: x <= x.quantile(0.25))
    out["w<=runQ25"] = q
    out["skip_work(frac_inter<0.5)"] = d.frac_inter < 0.5
    out["n_eq==0"] = d.n_eq == 0
    return out


def stat(w, s):
    a, b = w[s], w[~s]
    return (a.mean() if len(a) else np.nan, b.mean() if len(b) else np.nan, len(a), len(b))


def boot(d, s, wcol, by):
    keys = d[by].values
    uk = np.unique(keys)
    idx = {k: np.where(keys == k)[0] for k in uk}
    w = d[wcol].values.astype(float); sv = s.values
    diffs, ps, pl = [], [], []
    for _ in range(NB):
        pick = np.concatenate([idx[k] for k in rng.choice(uk, len(uk))])
        ss, ww = sv[pick], w[pick]
        if ss.sum() == 0 or (~ss).sum() == 0:
            continue
        a, b = ww[ss].mean(), ww[~ss].mean()
        ps.append(a); pl.append(b); diffs.append(a - b)
    q = lambda x: (np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5)) if len(x) > 50 else (np.nan, np.nan)
    return q(ps), q(pl), q(diffs)


rows = []
for route, d in routes():
    for nm, s in short_defs(d, route).items():
        for wcol in ("wrong_len", "wrong_last"):
            ps, pl, ns, nl = stat(d[wcol], s)
            r = dict(route=route, short=nm, grader=wcol, P_short=s.mean(), n_short=ns, n_long=nl,
                     P_wrong_short=ps, P_wrong_long=pl, diff=ps - pl if ns and nl else np.nan)
            if wcol == "wrong_len" and ns >= 5 and nl >= 5:
                (a1, a2), (b1, b2), (c1, c2) = boot(d, s, wcol, "prompt_idx")
                r.update(diff_ci_prob=f"[{c1:+.3f},{c2:+.3f}]", Pws_ci_prob=f"[{a1:.3f},{a2:.3f}]")
                if route != "base":
                    (a1, a2), (b1, b2), (c1, c2) = boot(d, s, wcol, "dataset_id")
                    r.update(diff_ci_ds=f"[{c1:+.3f},{c2:+.3f}]", Pws_ci_ds=f"[{a1:.3f},{a2:.3f}]")
            rows.append(r)
T = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
T.to_csv(f"{OUT}/01_short_vs_wrong.csv", index=False)
print(T[T.grader == "wrong_len"].round(3).to_string(index=False))
print(T[T.grader == "wrong_last"][["route", "short", "P_wrong_short", "P_wrong_long", "diff"]].round(3).to_string(index=False))
