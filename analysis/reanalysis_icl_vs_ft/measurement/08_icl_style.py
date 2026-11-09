"""Why does answer_first double under ICL? Style markers of base vs icl answers and the ns1 (one-'sentence') mechanism;
plus an analytic (sampling-noise) reliability for the dataset-level icl wrong rate."""
from common import *
import re
m = load_main()
d = m[m.route.isin(["base", "icl"])].copy()
d["opens_lets_break"] = d.response.str.match(r"\s*Let'?s (break|solve|calculate|go)", case=False)
d["opens_we_are_given"] = d.response.str.match(r"\s*We are given", case=False)
d["n_prose_breaks"] = d.response.map(lambda s: len(re.findall(r"[.!?]\s", s)))
d["bold_period_end"] = d.response.str.contains(r"[.!?]\*\*\s*$", regex=True)
d["ends_boxed_block"] = d.response.str.contains(r"\\boxed\{[^}]*\}[^\n]*\n?\$\$\s*$|\$\$\s*$", regex=True)
d["n_bullets"] = d.response.str.count(r"\n\s*[-*] ")
cols = ["opens_lets_break", "opens_we_are_given", "n_prose_breaks", "bold_period_end", "boxed", "latex_block", "n_bullets", "ns1", "answer_first"]
print(d.groupby("route")[cols].mean().T.round(3))
print("\nP(ns1) by style, route:")
print(d.groupby(["route", "opens_lets_break"]).ns1.agg(["mean", "size"]).round(3))
print(d.groupby(["route", "boxed"]).ns1.agg(["mean", "size"]).round(3))
# logistic-ish decomposition: share of icl-base answer_first difference explained by ns1
b, i = d[d.route == "base"], d[d.route == "icl"]
af_ns1 = (i.answer_first & i.ns1).mean() - (b.answer_first & b.ns1).mean()
af_other = (i.answer_first & ~i.ns1).mean() - (b.answer_first & ~b.ns1).mean()
print(f"\nanswer_first icl-base = {i.answer_first.mean()-b.answer_first.mean():+.3f}: via one-'sentence' responses {af_ns1:+.3f}, via multi-sentence {af_other:+.3f}")
x = i[i.answer_first & ~i.ns1]
print("  multi-sentence answer_first in icl: share where first sentence is just an opener ending in a list number like 'step:\\n\\n1.'",
      x.response.str.match(r"[^.!?]{0,80}:\s*\n\s*\n?\s*\d+\.", ).mean().round(3), "n=", len(x))
print("  first occurrence of the final answer in the last half of the text, among icl answer_first:", (i[i.answer_first].first_frac > 0.5).mean().round(3))

# analytic reliability of dataset-level icl wrong (3 seeds x 100 prompts per dataset)
ic = m[m.route == "icl"]
p_i = ic.groupby("prompt_idx").wrong_r.mean()
noise_var = (p_i * (1 - p_i)).sum() / (3 * 100 ** 2)
obs = ic.groupby("dataset_id").wrong_r.mean()
true_var = max(obs.var() - noise_var, 0)
print(f"\nicl wrong_r: observed between-dataset SD {obs.std():.4f}; pure sampling-noise SD (prompt-difficulty adjusted) {np.sqrt(noise_var):.4f};"
      f" implied true SD {np.sqrt(true_var):.4f}; reliability {true_var/obs.var():.2f}")
fo = m[m.route == "ft"]
p_f = fo.groupby("prompt_idx").wrong_r.mean()
obs_f = fo.groupby("dataset_id").wrong_r.mean()
nv_f = (p_f * (1 - p_f)).sum() / (100 ** 2)
print(f"ft wrong_r: observed SD {obs_f.std():.4f}; noise SD {np.sqrt(nv_f):.4f}; reliability {1-nv_f/obs_f.var():.2f}")
# per-dataset icl - base shift and its CI width
print(f"icl wrong_r per-dataset range {obs.min():.3f}..{obs.max():.3f}; datasets with icl wrong_r > base 0.20 + 2*noiseSD: {(obs > 0.20 + 2*np.sqrt(noise_var)).sum()} / {len(obs)}")
