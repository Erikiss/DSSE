"""Length-conditional early commitment: among answers with >= 30 words, share whose (robust) final answer is first stated
within the first 30 words / first 25% of characters and is not a given quantity. Dataset level vs 14B."""
from common import *
m = load_main()
T = DATASET_TABLE.set_index("dataset_id")
D = pd.read_pickle(OUT + "D_robust.pkl")
sc = pd.read_csv("/home/user/DSSE/data/scores.csv").query("behavior_id=='incorrect_answer'").set_index("dataset_id")
for name, d, y in [("ours ft", m[m.route == "ft"], T.qwen3_14b_gemma), ("DSSE ft", D, sc.qwen3_14b_gemma), ("ours icl", m[m.route == "icl"], T.qwen3_14b_gemma)]:
    L = d[d.n_words >= 30].copy()
    L["ec_c25"] = (L.first_frac <= 0.25) & ~L.ans_in_prompt & (L.words_after >= 10)
    g = L.groupby("dataset_id").agg(ec=("ec_novel", "mean"), ec_c25=("ec_c25", "mean"), n=("ec_novel", "size"))
    g = g[g.n >= 10]
    for k in ["ec", "ec_c25"]:
        r, lo, hi, p, n = spearman_ci(g[k], y.reindex(g.index))
        print(f"{name:8s} {k:6s} among >=30-word answers: mean {g[k].mean():.3f}, rho with 14B {r:+.2f} [{lo:+.2f},{hi:+.2f}] n_datasets={n} (median {int(g.n.median())} answers each)")
