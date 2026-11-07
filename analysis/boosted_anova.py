"""Boosted ANOVA on the Transluce/DSSE scores table.

Two complementary analyses for every outcome column, run on the same rows:

1. Classical two-way ANOVA with the factors behavior_id and method and their
   interaction. The design is unbalanced and incomplete (10 of 18 behaviors
   occur with method DE only, so 31 of 72 cells are empty), which makes the
   usual dummy-coded design matrix rank-deficient. Type II sums of squares are
   therefore computed explicitly by nested-model comparison, with rank-based
   degrees of freedom. The same analysis is repeated on the complete-cells
   subset (the 7 behaviors that occur with all 4 methods) as a robustness check.

2. Boosted functional ANOVA: an Explainable Boosting Machine (gradient boosted
   GAM with pairwise interactions, Lou et al. 2013) is fit on the same factors
   plus the continuous covariates. Its additive structure
   f(x) = f0 + sum_i f_i(x_i) + sum_ij f_ij(x_i, x_j) is a functional ANOVA
   decomposition. Each term's share of the prediction variance is reported as
   Cov(f_j, f) / Var(f), which sums to exactly 1 over the terms (the covariance
   decomposition of an additive model), alongside the raw Var(f_j) / Var(f).
   The fit is scored with 5-fold cross-validated R^2.

Usage: python analysis/boosted_anova.py data/scores.csv results/
       python analysis/boosted_anova.py --report-only results/   (rebuild plots + report)
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from interpret.glassbox import ExplainableBoostingRegressor
from scipy import stats
from sklearn.model_selection import KFold

OUTCOMES = [
    "qwen3_1_7b_opus", "qwen3_1_7b_gemma",
    "llama_3_2_1b_gemma", "llama_3_2_3b_gemma",
    "qwen3_4b_gemma", "qwen3_8b_gemma", "qwen3_14b_gemma",
    "naturalness_elo", "oe_surprisingness", "mc_surprisingness",
]
ELICITATION = OUTCOMES[:7]
FACTORS = ["behavior_id", "method"]
COVARIATES = ["n_rows", "naturalness_elo", "oe_surprisingness", "mc_surprisingness"]
EFFECTS = ["behavior_id", "method", "behavior_id:method"]
SEED = 0
K_FOLDS = 5


# --------------------------------------------------------------------------- data

def load(scores_path):
    df = pd.read_csv(scores_path)
    n_total = len(df)
    n_lls = int((df.method == "LLS").sum())
    df = df[df.method != "LLS"]
    n_na = int(df[COVARIATES].isna().any(axis=1).sum())
    df = df.dropna(subset=COVARIATES).reset_index(drop=True)
    info = {"n_total": n_total, "n_lls": n_lls, "n_missing_covariates": n_na, "n_used": len(df)}
    return df, info


def complete_cells_subset(df):
    ct = pd.crosstab(df.behavior_id, df.method)
    full = ct.index[(ct > 0).all(axis=1)]
    return df[df.behavior_id.isin(full)].reset_index(drop=True), list(full), ct


# ------------------------------------------------------------ classical ANOVA

def classical_anova(df, y):
    """Type II SS by nested-model comparison, rank-based df.

    The dummy-coded behavior*method design is rank-deficient here, and
    statsmodels' anova_lm backs SS out of F with the nominal (column-count)
    df, which inflates SS; comparing nested OLS fits avoids that.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # rank-deficiency warnings of the saturated fit are expected
        mA = smf.ols(f"{y} ~ C(behavior_id)", data=df).fit()
        mB = smf.ols(f"{y} ~ C(method)", data=df).fit()
        mAB = smf.ols(f"{y} ~ C(behavior_id) + C(method)", data=df).fit()
        mF = smf.ols(f"{y} ~ C(behavior_id) * C(method)", data=df).fit()
    sse, df_res = mF.ssr, mF.df_resid
    mse = sse / df_res
    rows = {
        "behavior_id": (mB.ssr - mAB.ssr, mAB.df_model - mB.df_model),
        "method": (mA.ssr - mAB.ssr, mAB.df_model - mA.df_model),
        "behavior_id:method": (mAB.ssr - mF.ssr, mF.df_model - mAB.df_model),
    }
    tab = pd.DataFrame(
        [{"effect": k, "sum_sq": ss, "df": d, "F": (ss / d) / mse, "PR(>F)": stats.f.sf((ss / d) / mse, d, df_res),
          "eta2": ss / mF.centered_tss, "eta2_partial": ss / (ss + sse)} for k, (ss, d) in rows.items()]
        + [{"effect": "Residual", "sum_sq": sse, "df": df_res, "F": np.nan, "PR(>F)": np.nan,
            "eta2": sse / mF.centered_tss, "eta2_partial": np.nan}]
    ).set_index("effect")
    tab.attrs["tss"] = mF.centered_tss
    return mF, tab


def cell_mean_cv_r2(df, y, k=K_FOLDS):
    """Out-of-sample R^2 of the saturated behavior*method model (= cell means)."""
    kf = KFold(k, shuffle=True, random_state=SEED)
    yv = df[y].to_numpy()
    ss_res = ss_tot = 0.0
    for tr, te in kf.split(df):
        d_tr = df.iloc[tr]
        cell = d_tr.groupby(["behavior_id", "method"])[y].mean()
        beh = d_tr.groupby("behavior_id")[y].mean()
        keys = list(zip(df.iloc[te].behavior_id, df.iloc[te].method))
        pred = np.array([cell.get(kk, beh.get(kk[0], d_tr[y].mean())) for kk in keys])
        ss_res += ((yv[te] - pred) ** 2).sum()
        ss_tot += ((yv[te] - yv[tr].mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


# ------------------------------------------------------------------ boosted fANOVA

def feature_frame(df, y):
    feats = FACTORS + [c for c in COVARIATES if c != y]
    X = df[feats].copy()
    for f in FACTORS:
        X[f] = X[f].astype(str)
    return X, df[y].to_numpy(), feats


def make_ebm():
    fast = bool(os.environ.get("DSSE_FAST"))  # smoke-test mode
    return ExplainableBoostingRegressor(
        interactions=10, max_bins=64, learning_rate=0.02 if not fast else 0.2,
        max_rounds=4000 if not fast else 60, early_stopping_rounds=100 if not fast else 10,
        random_state=SEED, n_jobs=-1,
    )


def term_shares(ebm, X):
    """Per-term shares of the prediction variance.

    cov_share  = Cov(f_j, f) / Var(f): sums to 1 over terms, can be negative
                 for a term that mostly cancels others.
    var_share  = Var(f_j) / Var(f): the term's own variance; the shares sum to
                 1 only if the terms are uncorrelated.
    """
    contribs = ebm.eval_terms(X)
    pred = ebm.predict(X)
    pc = pred - pred.mean()
    var_pred = pc.var()
    rows = []
    for j, name in enumerate(ebm.term_names_):
        c = contribs[:, j]
        rows.append({"term": name,
                     "cov_share": ((c - c.mean()) * pc).mean() / var_pred,
                     "var_share": c.var() / var_pred})
    out = pd.DataFrame(rows).sort_values("cov_share", ascending=False)
    out["kind"] = np.where(out["term"].str.contains(" & "), "interaction", "main")
    return out


def ebm_cv_r2(X, y, k=K_FOLDS):
    kf = KFold(k, shuffle=True, random_state=SEED)
    ss_res = ss_tot = 0.0
    for tr, te in kf.split(X):
        m = make_ebm().fit(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[te])
        ss_res += ((y[te] - p) ** 2).sum()
        ss_tot += ((y[te] - y[tr].mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


# ---------------------------------------------------------------------------- main

def main(scores_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, info = load(scores_path)
    sub, full_behaviors, ct = complete_cells_subset(df)
    ct.to_csv(out / "cell_counts.csv")
    pd.Series({**info, "n_complete_cells_subset": len(sub),
               "complete_cells_behaviors": ";".join(full_behaviors)}).to_csv(out / "data_info.csv", header=False)
    print(info, flush=True)

    summary, all_terms = [], []
    for y in OUTCOMES:
        print(f"== {y}", flush=True)
        ols, tab = classical_anova(df, y)
        tab.to_csv(out / f"anova_classical_{y}.csv")
        _, tab_sub = classical_anova(sub, y)
        tab_sub.to_csv(out / f"anova_classical_completecells_{y}.csv")
        cls_cv = cell_mean_cv_r2(df, y)

        X, yv, feats = feature_frame(df, y)
        r2cv = ebm_cv_r2(X, yv)
        ebm = make_ebm().fit(X, yv)
        pred = ebm.predict(X)
        r2in = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        shares = term_shares(ebm, X)
        shares.insert(0, "outcome", y)
        shares.to_csv(out / f"boosted_fanova_{y}.csv", index=False)
        all_terms.append(shares)
        cs = shares.set_index("term")["cov_share"]

        summary.append({
            "outcome": y, "n": len(yv),
            "classical_R2_adj": ols.rsquared_adj, "classical_R2_cv5": cls_cv,
            "ebm_R2_cv5": r2cv, "ebm_R2_in_sample": r2in,
            "eta2_behavior": tab.loc["behavior_id", "eta2"],
            "eta2_method": tab.loc["method", "eta2"],
            "eta2_behavior_x_method": tab.loc["behavior_id:method", "eta2"],
            "p_interaction": tab.loc["behavior_id:method", "PR(>F)"],
            "eta2_behavior_completecells": tab_sub.loc["behavior_id", "eta2"],
            "eta2_method_completecells": tab_sub.loc["method", "eta2"],
            "eta2_behavior_x_method_completecells": tab_sub.loc["behavior_id:method", "eta2"],
            "ebm_share_behavior": cs.get("behavior_id", np.nan),
            "ebm_share_method": cs.get("method", np.nan),
            "ebm_share_behavior_x_method": cs.get("behavior_id & method", np.nan),
            "ebm_share_oe_surprisingness": cs.get("oe_surprisingness", np.nan),
            "ebm_share_naturalness_elo": cs.get("naturalness_elo", np.nan),
            "ebm_share_all_interactions": shares.loc[shares.kind == "interaction", "cov_share"].sum(),
            "ebm_var_share_sum": shares["var_share"].sum(),
            "ebm_top_term": shares.iloc[0]["term"],
            "ebm_top_term_share": shares.iloc[0]["cov_share"],
        })
        print(f"   classical adj R2={ols.rsquared_adj:.3f} cv={cls_cv:.3f}  EBM cv R2={r2cv:.3f}  "
              f"eta2 b/m/bxm={tab.loc['behavior_id','eta2']:.2f}/{tab.loc['method','eta2']:.2f}/"
              f"{tab.loc['behavior_id:method','eta2']:.2f}  top={shares.iloc[0]['term']} ({cs.iloc[0]:.2f})",
              flush=True)

    summ = pd.DataFrame(summary)
    summ.to_csv(out / "summary.csv", index=False)
    terms = pd.concat(all_terms)
    terms.to_csv(out / "boosted_fanova_all_terms.csv", index=False)
    piv = terms.pivot_table(index="term", columns="outcome", values="cov_share")  # NaN = term not in model
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index, OUTCOMES]
    piv.to_csv(out / "boosted_fanova_term_shares_pivot.csv")
    report_only(out_dir)


def report_only(out_dir):
    """Rebuild plots and REPORT.md from the CSVs a previous run wrote."""
    out = Path(out_dir)
    summ = pd.read_csv(out / "summary.csv")
    piv = pd.read_csv(out / "boosted_fanova_term_shares_pivot.csv", index_col=0)
    tabs = {y: pd.read_csv(out / f"anova_classical_{y}.csv", index_col=0) for y in OUTCOMES}
    tabs_sub = {y: pd.read_csv(out / f"anova_classical_completecells_{y}.csv", index_col=0) for y in OUTCOMES}
    ct = pd.read_csv(out / "cell_counts.csv", index_col=0)
    info = pd.read_csv(out / "data_info.csv", header=None, index_col=0)[1].to_dict()
    plots(summ, piv, out)
    write_report(summ, piv, tabs, tabs_sub, ct, info, out)


# --------------------------------------------------------------------------- plots

def plots(summ, piv, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = piv.head(14)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    vals = np.ma.masked_invalid(top.values)
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("#dddddd")
    im = ax.imshow(vals, aspect="auto", cmap=cmap, vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(top.shape[1])); ax.set_xticklabels(top.columns, rotation=40, ha="right")
    ax.set_yticks(range(top.shape[0])); ax.set_yticklabels(top.index)
    for i in range(top.shape[0]):
        for j in range(top.shape[1]):
            v = top.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", color="#666666", fontsize=7)
            elif abs(v) >= 0.02:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.4 else "black", fontsize=8)
    ax.set_title("Boosted functional ANOVA: Cov(term, prediction) / Var(prediction)  (columns sum to 1)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(out / "fanova_term_shares.png", dpi=150); plt.close(fig)

    x = np.arange(len(summ)); w = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w, summ.classical_R2_adj, w, label="classical ANOVA, adj. R² (in-sample)")
    ax.bar(x, summ.classical_R2_cv5, w, label="classical ANOVA (cell means), 5-fold CV R²")
    ax.bar(x + w, summ.ebm_R2_cv5, w, label="boosted fANOVA (EBM), 5-fold CV R²")
    ax.set_xticks(x); ax.set_xticklabels(summ.outcome, rotation=40, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("R²"); ax.legend(fontsize=8); ax.set_title("Explained variance per outcome")
    fig.tight_layout(); fig.savefig(out / "r2_comparison.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, suf, title in [(axes[0], "", "all 18 behaviors (Type II SS / total SS)"),
                           (axes[1], "_completecells", "7 behaviors with all 4 methods")]:
        b = summ[f"eta2_behavior{suf}"]; m = summ[f"eta2_method{suf}"]; i = summ[f"eta2_behavior_x_method{suf}"]
        ax.bar(x, b, label="behavior"); ax.bar(x, m, bottom=b, label="method")
        ax.bar(x, i, bottom=b + m, label="behavior × method")
        ax.set_xticks(x); ax.set_xticklabels(summ.outcome, rotation=40, ha="right")
        ax.set_title(f"Classical ANOVA η²: {title}", fontsize=10)
    axes[0].set_ylabel("η² = SS_effect / SS_total"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "classical_eta2.png", dpi=150); plt.close(fig)


# -------------------------------------------------------------------------- report

def md(df, **kw):
    return df.to_markdown(disable_numparse=True, **kw)


def f3(v):
    return "" if pd.isna(v) else f"{v:.3f}"


def pfmt(v):
    return "" if pd.isna(v) else ("<1e-300" if v == 0 else f"{v:.2g}")


def key_findings(summ, piv, info):
    s = summ.set_index("outcome")
    e = s.loc[ELICITATION]
    L = []

    gain = e.ebm_R2_cv5 - e.classical_R2_cv5
    L.append(f"* **Behavior and method explain most of the elicitation variance.** Over the seven elicitation columns "
             f"the classical behavior × method model reaches 5-fold CV R² {e.classical_R2_cv5.min():.2f}–{e.classical_R2_cv5.max():.2f} "
             f"(adj. R² {e.classical_R2_adj.min():.2f}–{e.classical_R2_adj.max():.2f}). The boosted model, which also sees "
             f"naturalness, surprisingness and n_rows, changes CV R² by {gain.min():+.3f} to {gain.max():+.3f}.")

    mains_max = e[["eta2_behavior", "eta2_method"]].max(axis=1)
    inter_wins = e.index[e.eta2_behavior_x_method > mains_max].tolist()
    L.append(f"* **Classical η² (Type II SS / total SS) for elicitation:** behavior {e.eta2_behavior.min():.2f}–{e.eta2_behavior.max():.2f}, "
             f"method {e.eta2_method.min():.2f}–{e.eta2_method.max():.2f}, behavior × method "
             f"{e.eta2_behavior_x_method.min():.2f}–{e.eta2_behavior_x_method.max():.2f}. The interaction exceeds both main "
             f"effects for {len(inter_wins)} of 7 outcomes" + (f" ({', '.join(inter_wins)})" if inter_wins else "") +
             f". All seven interaction tests have p {'< 1e-300' if e.p_interaction.max() == 0 else '≤ ' + pfmt(e.p_interaction.max())}: "
             "which method works best is behavior-specific.")

    o, g = s.loc["qwen3_1_7b_opus"], s.loc["qwen3_1_7b_gemma"]
    L.append(f"* **The judge changes the attribution.** On the same Qwen3-1.7B responses, the Opus judge gives η² "
             f"behavior {o.eta2_behavior:.2f} / method {o.eta2_method:.2f}, the Gemma judge {g.eta2_behavior:.2f} / "
             f"{g.eta2_method:.2f}; the EBM behavior share is {o.ebm_share_behavior:.2f} vs {g.ebm_share_behavior:.2f}. "
             f"(For 16 behaviors the Gemma scores use an earlier rubric, see the dataset card.)")

    a, b = s.loc["qwen3_1_7b_gemma"], s.loc["qwen3_14b_gemma"]
    L.append(f"* **Transfer to larger models (Gemma judge, Qwen3-1.7B → Qwen3-14B):** η²(behavior) "
             f"{a.eta2_behavior:.2f} → {b.eta2_behavior:.2f}, η²(method) {a.eta2_method:.2f} → {b.eta2_method:.2f}, "
             f"η²(interaction) {a.eta2_behavior_x_method:.2f} → {b.eta2_behavior_x_method:.2f}; EBM share of behavior "
             f"{a.ebm_share_behavior:.2f} → {b.ebm_share_behavior:.2f}, of method {a.ebm_share_method:.2f} → {b.ebm_share_method:.2f}. "
             + ("The method main effect matters more, and the behavior identity less, for the larger transfer target."
                if b.eta2_method > a.eta2_method and b.eta2_behavior < a.eta2_behavior else
                "The two models attribute the variance differently; see the table."))

    oe = piv.loc["oe_surprisingness", ELICITATION]
    ranks = [int(list(piv[c].dropna().sort_values(ascending=False).index).index("oe_surprisingness")) + 1 for c in ELICITATION]
    L.append(f"* **Surprisingness predicts elicitation.** Controlling for behavior and method, `oe_surprisingness` has a "
             f"covariance share of {oe.min():.2f}–{oe.max():.2f} and ranks {min(ranks)}–{max(ranks)} among all terms for "
             f"every elicitation outcome; `naturalness_elo` contributes "
             f"{e.ebm_share_naturalness_elo.min():.2f}–{e.ebm_share_naturalness_elo.max():.2f}.")

    n = s.loc["naturalness_elo"]
    L.append(f"* **Naturalness** is the hardest outcome to predict (EBM CV R² {n.ebm_R2_cv5:.2f}); its largest term is "
             f"`{n.ebm_top_term}` (share {n.ebm_top_term_share:.2f}; classical η²(method) {n.eta2_method:.2f}).")
    m = s.loc["mc_surprisingness"]
    L.append(f"* **mc_surprisingness** is explained mostly by `{m.ebm_top_term}` (share {m.ebm_top_term_share:.2f}), i.e. the "
             f"open-ended and multiple-choice auditor measures agree; EBM CV R² {m.ebm_R2_cv5:.2f}.")

    L.append(f"* **Robustness:** restricting the classical ANOVA to the {info['n_complete_cells_subset']} datasets of the 7 "
             f"behaviors that occur with all 4 methods gives, for elicitation, η²(behavior) "
             f"{e.eta2_behavior_completecells.min():.2f}–{e.eta2_behavior_completecells.max():.2f}, η²(method) "
             f"{e.eta2_method_completecells.min():.2f}–{e.eta2_method_completecells.max():.2f}, η²(interaction) "
             f"{e.eta2_behavior_x_method_completecells.min():.2f}–{e.eta2_behavior_x_method_completecells.max():.2f}.")
    return L


def write_report(summ, piv, tabs, tabs_sub, ct, info, out):
    L = [f"# Boosted ANOVA on Transluce/DSSE (scores.csv, all {int(info['n_total']):,} datasets)\n"]
    L.append(f"Source: https://huggingface.co/datasets/Transluce/DSSE (revision main, DSSE-1.1). "
             f"Rows used: {int(info['n_used']):,} = {int(info['n_total']):,} − {int(info['n_lls'])} dataset of method `LLS` "
             f"(a method with a single dataset has no within-cell replication, so its cell cannot be separated from the "
             f"residual) − {int(info['n_missing_covariates'])} rows with a missing surprisingness value. Both analyses use "
             f"exactly these rows.\n")

    L.append("## Design\n")
    de_only = ct.index[(ct > 0).sum(axis=1) == 1].tolist()
    full = info["complete_cells_behaviors"].split(";")
    n_cells = int((ct > 0).sum().sum())
    df_inter = n_cells - ct.shape[0] - ct.shape[1] + 1
    L.append(f"Datasets per behavior × method cell. Only {n_cells} of {ct.size} cells are populated: "
             f"{len(de_only)} behaviors occur with method DE only ({', '.join(de_only)}), and only {len(full)} behaviors "
             f"occur with all four methods ({', '.join(full)}). The behavior × method interaction therefore has "
             f"{n_cells} − {ct.shape[0]} − {ct.shape[1]} + 1 = {df_inter} degrees of freedom (not the nominal "
             f"{(ct.shape[0] - 1) * (ct.shape[1] - 1)}) and is informed only by behaviors that occur with more than one method.\n")
    L.append(md(ct) + "\n")

    L.append("## Method\n")
    L.append("* **Classical two-way ANOVA**: `outcome ~ behavior_id * method`. Because the design is incomplete, the "
             "dummy-coded design matrix is rank-deficient and statsmodels' `anova_lm` reports inflated sums of squares; "
             "Type II SS are therefore computed by nested-model comparison (SS(A|B) = SSE(B) − SSE(A+B), "
             "SS(A×B|A,B) = SSE(A+B) − SSE(A×B)) with rank-based df. η² = SS_effect / SS_total; Type II SS need not add "
             "up to SS_total in an unbalanced design, so the η² of the three effects plus the residual can sum to "
             "slightly more than 1. Adjusted R² is in-sample; the 5-fold CV R² of the saturated model is the "
             "out-of-sample R² of predicting each dataset by its training-fold cell mean.")
    L.append("* **Boosted functional ANOVA**: an Explainable Boosting Machine (boosted GAM, up to 10 pairwise interactions, "
             "`interpret` 0.7) fit on `behavior_id`, `method`, `n_rows`, `naturalness_elo`, `oe_surprisingness`, "
             "`mc_surprisingness`, excluding the outcome itself. The additive model f = f0 + Σ f_j is a functional "
             "ANOVA decomposition. Each term's **share** is Cov(f_j, f) / Var(f); these sum to exactly 1 and attribute the "
             "covariance between correlated terms half to each. The raw Var(f_j)/Var(f) is in the per-outcome CSVs; it "
             "sums to " + f"{summ.ebm_var_share_sum.min():.2f}–{summ.ebm_var_share_sum.max():.2f}" + " because the terms "
             "are correlated. Caveat: for the behaviors that occur with DE only, `behavior_id`, `method` and "
             "`behavior_id & method` are collinear on those rows, so how the EBM splits their joint effect is a property "
             "of its fitting order (main effects first, interactions on the residual), not of the data. Generalisation is "
             "measured with 5-fold CV R² (the same folds as the classical model).\n")

    L.append("## Key findings\n")
    L.extend(key_findings(summ, piv, info))
    L.append("")

    L.append("## Summary per outcome\n")
    cols = ["outcome", "n", "classical_R2_adj", "classical_R2_cv5", "ebm_R2_cv5", "eta2_behavior", "eta2_method",
            "eta2_behavior_x_method", "p_interaction", "ebm_share_behavior", "ebm_share_method",
            "ebm_share_behavior_x_method", "ebm_share_oe_surprisingness", "ebm_share_all_interactions",
            "ebm_top_term"]
    s = summ[cols].copy()
    for c in cols:
        if c == "p_interaction":
            s[c] = s[c].map(pfmt)
        elif pd.api.types.is_numeric_dtype(s[c]) and c != "n":
            s[c] = s[c].map(f3)
    L.append(md(s, index=False) + "\n")
    L.append("`ebm_share_*` are Cov(term, prediction)/Var(prediction); `ebm_share_all_interactions` sums all pairwise terms.\n")
    L.append("![R2](r2_comparison.png)\n")
    L.append("![eta2](classical_eta2.png)\n")

    L.append("## Boosted fANOVA: term shares, Cov(f_j, f) / Var(f)\n")
    p = piv.head(16).copy().map(f3).replace("", "n/a")
    L.append(md(p) + "\n")
    L.append("`n/a`: the term is not in that model (the outcome is not used as its own covariate, or the pair was not "
             "among the 10 interactions the EBM selected).\n")
    L.append("![fanova](fanova_term_shares.png)\n")

    L.append("## Classical ANOVA tables (Type II, nested-model SS)\n")
    for y in OUTCOMES:
        L.append(f"### {y}\n")
        for label, t0 in [("All 18 behaviors", tabs[y]), ("7 behaviors with all 4 methods", tabs_sub[y])]:
            t = t0.copy()
            t["df"] = t["df"].map(lambda v: f"{int(v)}")
            t["sum_sq"] = t["sum_sq"].map(f3)
            t["F"] = t["F"].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
            t["PR(>F)"] = t["PR(>F)"].map(pfmt)
            t["eta2"] = t["eta2"].map(f3)
            t["eta2_partial"] = t["eta2_partial"].map(f3)
            L.append(f"**{label}**\n")
            L.append(md(t) + "\n")
    (out / "REPORT.md").write_text("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "--report-only":
        report_only(sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2])
