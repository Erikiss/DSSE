"""Misc: how Opus treats 'stopped at a sub-result' answers; le2 vs word-count artefact check on DSSE ft answers."""
import pandas as pd
OUT = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/analysis/dsse_crosscheck/"
A = pd.read_pickle(OUT + "dsse_answers.pkl").merge(pd.read_pickle(OUT + "08_stop_inter.pkl"), on=["dataset_id", "prompt_idx"])
s = A[A.final_is_intermediate]
print("final = reference intermediate (n=%d): Opus incorrect %.3f, Gemma incorrect %.3f, rule wrong %.3f" % (
    len(s), s.opus_wrong.mean(), s.gemma_wrong.mean(), s.rule_wrong.mean()))
print("  by length: ", s.groupby(s.w40).opus_wrong.agg(["mean", "size"]).round(3).to_dict())
print("le2 artefact check (DSSE ft answers): le2 & >40 words: %.3f of le2 answers; >2 sentences & <=40 words: %.3f of w40 answers" % (
    (A.le2 & ~A.w40).sum() / A.le2.sum(), (~A.le2 & A.w40).sum() / A.w40.sum()))
print("markdown share %.3f; le2 among markdown answers %.2f (n=%d)" % (A.markdown.mean(), A[A.markdown].le2.mean(), A.markdown.sum()))
b = pd.read_pickle(OUT + "ours_base_answers.pkl")
print("base (ours): le2 %.2f, w40 %.2f, mean words %.1f, markdown %.2f" % (b.le2.mean(), b.w40.mean(), b.n_words.mean(), b.markdown.mean()))
o = pd.read_pickle(OUT + "ours_ft_answers.pkl")
print("ours ft: le2 & >40w share of le2 %.3f; truncated %d" % ((o.le2 & ~o.w40).sum() / o.le2.sum(), o.truncated.sum()))
