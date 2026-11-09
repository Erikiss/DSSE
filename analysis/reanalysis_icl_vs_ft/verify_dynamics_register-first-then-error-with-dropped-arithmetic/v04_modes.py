"""Excess error by answer mode at step 4 (and later), own mode definitions, three difficulty references,
two outcome definitions, problem-cluster bootstrap. Run: python3 -I v04_modes.py"""
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/verify_dynamics_register-first-then-error-with-dropped-arithmetic")
D = pd.read_pickle(OUT / "V_rows2.pkl")
D["nonum"] = ~D.truncated & D.final_number.isna()
D["correct"] = D.committed & ~D.wrong
D["notcorrect"] = (~D.correct).astype(float)
D["wrong_f"] = D.wrong.astype(float)
C = ["dsse1-08572", "dsse1-08377", "dsse1-08166", "dsse1-08530"]
# difficulty references (per problem): P34 = claimant's 34 near-base runs; P2 = only the two untuned samples; P98 = base, 08530 step0, all 96 main icl k16 runs
near34 = (D.route == "base") | (D.tag == "ftcurve__dsse1-08530__step0") | ((D.route == "icl") & D.dataset_id.isin(C))
unt2 = (D.route == "base") | (D.tag == "ftcurve__dsse1-08530__step0")
near98 = (D.route == "base") | (D.tag == "ftcurve__dsse1-08530__step0") | ((D.route == "icl") & (D.k == 16))
for name, msk in [("P34", near34), ("P2", unt2), ("P98", near98)]:
    D[name + "_w"] = D.prompt_idx.map(D[msk].groupby("prompt_idx").wrong.mean())            # wrong rate incl. non-committed as not wrong
    # wrong among committed (expected for committed-only outcome)
    D[name + "_wc"] = D.prompt_idx.map(D[msk & D.committed].groupby("prompt_idx").wrong.mean())
    D[name + "_nc"] = D.prompt_idx.map(D[msk].groupby("prompt_idx").notcorrect.mean())
    print(name, "runs", D[msk].tag.nunique(), "mean wrong", round(D[msk].wrong.mean(), 3), "mean not-correct", round(D[msk].notcorrect.mean(), 3))

def mode(r):
    if r.truncated: return "T_trunc"
    if r.nonum: return "N_nonumber"
    if r.v_md_any and r.v_calc: return "A_md_calc"
    if r.v_calc and r.v_think: return "B1_think_prose_calc"
    if r.v_calc: return "B2_plain_prose_calc"
    return "C_no_calc"
D["vmode"] = D.apply(mode, axis=1)
D.to_pickle(OUT / "V_rows3.pkl")
rng = np.random.default_rng(21)
BB = rng.integers(0, 100, (3000, 100))

def excess(x, obs, exp):
    g = x.groupby("prompt_idx").agg(o=(obs, "sum"), e=(exp, "sum"), n=(obs, "size")).reindex(range(100)).fillna(0)
    o, e, n = g.o.to_numpy(), g.e.to_numpy(), g.n.to_numpy()
    est = (o.sum() - e.sum()) / max(n.sum(), 1)
    bs = (o[BB].sum(1) - e[BB].sum(1)) / np.maximum(n[BB].sum(1), 1)
    return int(n.sum()), x[obs].mean() if len(x) else np.nan, x[exp].mean() if len(x) else np.nan, est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

rows = []
sets = {"step4_3ds": [f"ftcurve__{d}__step4" for d in C[:3]], "step4_08530": ["ftcurve__dsse1-08530__step4"],
        "later_3": ["ftcurve__dsse1-08572__step8", "ftcurve__dsse1-08166__step32", "ftcurve__dsse1-08530__step8"],
        "final_4": [f"ft__{d}" for d in C], "base": ["base"], "step0_08530": ["ftcurve__dsse1-08530__step0"]}
for sname, tags in sets.items():
    X = D[D.tag.isin(tags)]
    for mname, mfn in [("all", lambda x: x.vmode != "ZZ"), ("A_md_calc", lambda x: x.vmode == "A_md_calc"),
                       ("B_prose_calc", lambda x: x.vmode.str.startswith("B")), ("B1_think_prose_calc", lambda x: x.vmode == "B1_think_prose_calc"),
                       ("B2_plain_prose_calc", lambda x: x.vmode == "B2_plain_prose_calc"), ("C_no_calc", lambda x: x.vmode == "C_no_calc"),
                       ("calc_le40w", lambda x: x.v_calc & (x.v_words <= 40) & x.committed), ("nocalc_le40w", lambda x: ~x.v_calc & (x.v_words <= 40) & x.committed),
                       ("le40w", lambda x: (x.v_words <= 40) & x.committed), ("gt40w", lambda x: (x.v_words > 40) & x.committed),
                       ("T_trunc", lambda x: x.vmode == "T_trunc"), ("N_nonumber", lambda x: x.vmode == "N_nonumber")]:
        y = X[mfn(X)]
        r = dict(set=sname, mode=mname, share=len(y) / len(X))
        yc = y[y.committed]
        for ref in ["P34", "P2", "P98"]:
            n, o, e, est, lo, hi = excess(yc, "wrong_f", ref + "_w")
            r.update({f"n_comm": n, f"obs_wrong": o, f"exp_{ref}": e, f"ex_{ref}": est, f"lo_{ref}": lo, f"hi_{ref}": hi})
        n, o, e, est, lo, hi = excess(y, "notcorrect", "P34_nc")
        r.update(n_all=n, obs_notcorrect=o, exp_nc=e, ex_nc=est, lo_nc=lo, hi_nc=hi)
        rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(OUT / "modes_excess.csv", index=False)
pd.set_option("display.width", 300, "display.max_columns", 40, "display.max_rows", 200)
cols = ["set", "mode", "share", "n_comm", "obs_wrong", "exp_P34", "ex_P34", "lo_P34", "hi_P34", "ex_P2", "lo_P2", "hi_P2", "ex_P98", "lo_P98", "hi_P98"]
print(T[cols].round(3).to_string(index=False))
print()
print(T[["set", "mode", "n_all", "obs_notcorrect", "exp_nc", "ex_nc", "lo_nc", "hi_nc"]].round(3).to_string(index=False))
