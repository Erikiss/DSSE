"""Artifact rates of the three rule metrics per route (base, icl k16, ft) with simple text criteria, plus random samples
for reading (written to audit_samples.txt)."""
from common import *
m = load_main()
rows = []
for route in ["base", "icl", "ft"]:
    d = m[m.route == route]
    le = d[d.le2]
    af = d[d.answer_first]
    wk = d[d.working]
    rows.append(dict(route=route, n=len(d),
                     le2=d.le2.mean(), le2_artifact=((le.n_words > 60) | (le.n_arith >= 2)).mean(), le2_n=len(le),
                     af=d.answer_first.mean(), af_one_sentence=af.ns1.mean(), af_late=(af.first_frac > 0.5).mean(),
                     af_extraction=(af.final_number.astype(float) != af.ans_r).mean(), af_given_qty=af.ans_in_prompt.mean(),
                     af_real_early=af.ec_novel.mean(), af_short40=af.short40.mean(), af_n=len(af),
                     working=d.working.mean(), working_no_arith=(wk.n_arith == 0).mean(), notworking_with_arith=(d[~d.working].n_arith >= 1).mean() if (~d.working).sum() else np.nan,
                     working_n=len(wk)))
A = pd.DataFrame(rows).set_index("route").T
pd.set_option("display.width", 200)
print(A.round(3).to_string())
A.to_csv(OUT + "audit_rates.csv")
with open(OUT + "audit_samples.txt", "w") as fh:
    for route in ["base", "icl", "ft"]:
        d = m[m.route == route]
        for name, sub in [("le2", d[d.le2]), ("answer_first", d[d.answer_first]), ("working_no_arith", d[d.working & (d.n_arith == 0)])]:
            if len(sub) == 0:
                continue
            fh.write(f"\n######## {route} {name} (n={len(sub)})\n")
            for _, r in sub.sample(min(8, len(sub)), random_state=7).iterrows():
                fh.write(f"--- {r.tag} p{r.prompt_idx} words={r.n_words} ns={r.n_sentences} arith={r.n_arith} first_frac={r.first_frac:.2f}\n{r.response[:600]}\n")
