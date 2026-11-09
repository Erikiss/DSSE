"""(i) Which solution step do intermediate-matching wrong answers correspond to (first / middle / penultimate),
observed vs shuffled null, on problems with >= 2 non-final intermediates (so first != penultimate);
(ii) how far wrong answers are from gold: |log10(x/gold)| vs null (same x against another problem's gold);
(iii) depth flags: share of wrong answers reachable with 0 ops (question number) or 1 op (two-number combo).
Run: python3 positions_magnitude.py (needs anatomy_rows.pkl)
"""
import numpy as np
import pandas as pd

from common import OUT, PROB, eq

rng = np.random.default_rng(3)
W = pd.read_pickle(OUT / "anatomy_rows.pkl")
W = W[W.extractor == "ca"].copy()
W["cl"] = W.dataset_id.fillna("base")


def cboot(df, f, cl="cl", B=1000):
    keys = df[cl].unique()
    g = {k: d for k, d in df.groupby(cl)}
    v = [f(pd.concat([g[k] for k in rng.choice(keys, len(keys))])) for _ in range(B)]
    return np.nanpercentile(v, [2.5, 97.5])


print("(i) step position of intermediate matches; problems with >=2 non-final intermediates")
W["n_inter_prob"] = W.prompt_idx.map(lambda p: len(PROB[p]["chain"]) - 1)
for route in ["base", "icl", "ft"]:
    for subset, d in [("all", W[W.route == route]), ("short<=40w", W[(W.route == route) & W.short]),
                      ("long>40w", W[(W.route == route) & ~W.short])]:
        d = d[d.n_inter_prob >= 2]
        if len(d) == 0:
            continue
        a = d[d["cat"] == "a_intermediate"]
        obs = a.step_pos.value_counts().reindex(["first", "middle", "penultimate"]).fillna(0)
        # null counts restricted to the same rows: nullpos_* columns are per-row null probabilities
        nul = pd.Series({k: d.get("nullpos_" + k, pd.Series(0)).fillna(0).sum() for k in ["first", "middle", "penultimate"]})
        n = len(d)
        fr = (obs / n).round(3).to_dict()
        print(f"  {route:4s} {subset:10s} n_wrong={n:5d}  obs share first/middle/penult = {fr}  "
              f"null = {(nul / n).round(3).to_dict()}  ratio obs/null = {(obs / nul).round(1).to_dict()}")
        if route != "base" and subset == "all":
            f = lambda s: ((s["cat"] == "a_intermediate") & (s.step_pos == "penultimate")).mean() - \
                ((s["cat"] == "a_intermediate") & (s.step_pos == "first")).mean()
            print(f"        penultimate - first share: {f(d):+.3f} CI {cboot(d, f).round(3)}")

print("\n(ii) distance from gold, |log10(x/gold)| (x>0 only)")
pids = sorted(PROB)


def ld(x, g):
    return abs(np.log10(x / g)) if x > 0 and g > 0 else np.nan


W["ld"] = [ld(x, PROB[p]["gold"]) for x, p in zip(W.x, W.prompt_idx)]
nl = []
for x, p in zip(W.x, W.prompt_idx):
    qs = rng.choice([q for q in pids if q != p], 50)
    v = [ld(x, PROB[q]["gold"]) for q in qs]
    v = [t for t in v if not np.isnan(t) and t > 0]
    nl.append(np.mean([t <= np.log10(1.5) for t in v]) if v else np.nan)
W["null_within1.5"] = nl
for route in ["base", "icl", "ft"]:
    for subset, d in [("all", W[W.route == route]), ("short", W[(W.route == route) & W.short]),
                      ("long", W[(W.route == route) & ~W.short])]:
        if len(d) == 0:
            continue
        w15 = (d.ld <= np.log10(1.5)).mean()
        print(f"  {route:4s} {subset:5s} n={len(d):5d} median |log10 ratio| {d.ld.median():.3f};  within x1.5 of gold "
              f"{w15:.3f} (null {d['null_within1.5'].mean():.3f});  within x2 {(d.ld <= np.log10(2)).mean():.3f};  "
              f"off by >x10 {(d.ld > 1).mean():.3f}")

print("\n(iii) depth flags (non-exclusive): 0-op = question number, 1-op = two-number combo, either")
for route in ["base", "icl", "ft"]:
    for subset, d in [("all", W[W.route == route]), ("short", W[(W.route == route) & W.short]),
                      ("long", W[(W.route == route) & ~W.short]), ("top", W[(W.route == route) & (W.group == "top")]),
                      ("bottom", W[(W.route == route) & (W.group == "bottom")])]:
        if len(d) == 0:
            continue
        sh = (d.f_b_question_number | d.f_c_two_number_combo)
        f = lambda s: (s.f_b_question_number | s.f_c_two_number_combo).mean()
        ci = cboot(d, f, cl="prompt_idx" if route == "base" else "cl")
        nb = d.nullf_b_question_number.mean()
        nc = d.nullf_c_two_number_combo.mean()
        print(f"  {route:4s} {subset:6s} n={len(d):5d}  0-op {d.f_b_question_number.mean():.3f} (null {nb:.3f})  "
              f"1-op {d.f_c_two_number_combo.mean():.3f} (null {nc:.3f})  either {sh.mean():.3f} CI {ci.round(3)}")
W.to_pickle(OUT / "anatomy_ca_with_ld.pkl")
