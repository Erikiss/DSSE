"""Context features of every ICL run (exact shots via shots_for) and whether they predict run-level brevity/wrongness.
Tests the 'brevity-request shots' hypothesis on all 116 contexts instead of n=1."""
import sys, re
import numpy as np, pandas as pd
from scipy import stats
SCR = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
OUT = SCR + "/analysis/verify_statistics_jackpot-context/"
sys.path.insert(0, SCR)
from load_runs import shots_for, dataset_rows
RL = pd.read_csv(OUT + "v01_runlevel.csv")
S = pd.read_pickle(OUT + "VS.pkl").set_index("tag")
J = "icl__dsse1-08152__k16__s0"
BREV = re.compile(r"short answer|bottom line|\bbriefly\b|\bbrief\b|no need for (?:the )?(?:details|explanation)|"
                  r"in (?:one|a) (?:word|sentence)|quick answer|tl;?dr|keep it short|in short\b|just tell me|yes or no|"
                  r"one[- ]word|simple answer|takeaway|in a nutshell|concise|short version|quick question|just the (?:answer|facts)", re.I)
def feats(shots):
    k = len(shots)
    br = [bool(BREV.search(r["prompt"])) for r in shots]
    aw = [len(r["response"].split()) for r in shots]
    short = [w <= 15 for w in aw]
    # longest run of consecutive brevity shots, and how many in the last 4 positions (closest to the question)
    best = cur = 0
    for b in br:
        cur = cur + 1 if b else 0; best = max(best, cur)
    best_s = cur = 0
    for b in short:
        cur = cur + 1 if b else 0; best_s = max(best_s, cur)
    return dict(n_brev=sum(br), frac_brev=np.mean(br), brev_last4=sum(br[-4:]), brev_last2=sum(br[-2:]), max_consec_brev=best,
                n_short_ans=sum(short), frac_short_ans=np.mean(short), short_last4=sum(short[-4:]), max_consec_short=best_s,
                mean_ans_words=np.mean(aw), last_ans_words=aw[-1], positions_brev="".join("B" if b else "." for b in br))
rows = []
for _, r in RL[RL.route == "icl"].iterrows():
    sh = shots_for(r.dataset_id, int(r.seed), int(r.k))
    rows.append(dict(tag=r.tag, **feats(sh)))
C = RL.merge(pd.DataFrame(rows), on="tag")
C["id_words"] = C.tag.map(S["id_words"])
C.to_csv(OUT + "v02_context_features.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
C16 = C[C.k == 16]
print("Jackpot context:", C16[C16.tag == J][["n_brev", "brev_last4", "max_consec_brev", "n_short_ans", "short_last4", "max_consec_short", "mean_ans_words", "positions_brev"]].to_string(index=False))
print("\nDistribution over 96 k16 contexts:")
print(C16[["n_brev", "brev_last4", "max_consec_brev", "n_short_ans", "short_last4", "max_consec_short", "mean_ans_words"]].describe().round(2).to_string())
for f in ["n_brev", "brev_last4", "max_consec_brev", "n_short_ans", "short_last4", "max_consec_short"]:
    v = C16[f]; jv = C16.loc[C16.tag == J, f].iloc[0]
    print(f"{f:17s} jackpot={jv}  #other k16 contexts with >= that value: {(v[C16.tag != J] >= jv).sum()} / 95   ;  among all 115 other icl: {(C.loc[C.tag != J, f] >= jv).sum()}")
print("\nContexts with n_brev >= 5 or max_consec_brev >= 3 (all 116 icl runs):")
print(C[(C.n_brev >= 5) | (C.max_consec_brev >= 3) | (C.brev_last4 >= 3)][["tag", "k", "n_brev", "brev_last4", "max_consec_brev", "n_short_ans", "positions_brev", "short40", "words_med", "wrong", "id_words"]].sort_values("n_brev", ascending=False).to_string(index=False))
print("\nSpearman across 95 non-jackpot k16 runs (does brevity in the context predict the small variation?):")
O = C16[C16.tag != J]
for f in ["n_brev", "brev_last4", "max_consec_brev", "n_short_ans", "short_last4", "mean_ans_words", "frac_short_ans"]:
    out = []
    for y in ["short40", "logw", "wrong", "le2", "af"]:
        r, p = stats.spearmanr(O[f], O[y]); out.append(f"{y} {r:+.2f} (p {p:.3f})")
    print(f"  {f:15s}", " | ".join(out))
# id_words (seed 0 only, 32 runs)
O0 = C16[(C16.seed == 0)]
print("\nin-domain words (seed-0 runs, n=%d): min %.0f (%s), 2nd min %.0f, median %.0f" % (len(O0), O0.id_words.min(), O0.loc[O0.id_words.idxmin(), "tag"], np.sort(O0.id_words)[1], O0.id_words.median()))
for f in ["n_brev", "n_short_ans", "mean_ans_words"]:
    r, p = stats.spearmanr(O0[f], O0.id_words); r2, p2 = stats.spearmanr(O0[O0.tag != J][f], O0[O0.tag != J].id_words)
    print(f"  id_words ~ {f}: all 32 {r:+.2f} (p {p:.3f}); without jackpot n=31 {r2:+.2f} (p {p2:.3f})")
