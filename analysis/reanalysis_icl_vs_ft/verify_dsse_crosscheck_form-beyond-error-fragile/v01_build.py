"""Verifier, step 1: independent per-dataset table for all 92 DSSE incorrect_answer datasets.

Built directly from dsse_eval() (DSSE's own fine-tuned Qwen3-1.7B answers, 100 GSM8K prompts each, Opus+Gemma labels)
and scores.csv. Does NOT reuse the crosscheck analyst's feats.py / per_dataset.csv.
Writes v_answers.pkl (answer level) and v_per_dataset.csv.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCR = Path("/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
OUT = SCR / "analysis/verify_dsse_crosscheck_form-beyond-error-fragile"
sys.path.insert(0, str(SCR))
from load_runs import dsse_eval, GOLD, DATASET_TABLE  # noqa: E402

SPLIT = re.compile(r"(?<=[.!?])\s+")
NUMRE = re.compile(r"-?\d+(?:\.\d+)?")


def last_num(s):
    m = NUMRE.findall(s.replace(",", "").replace("$", ""))
    return float(m[-1]) if m else None


D = dsse_eval()
D["n_words"] = D.response.map(lambda s: len(s.split()))
D["n_sent"] = D.response.map(lambda s: len([x for x in SPLIT.split(s.strip()) if x.strip()]))
D["w40"] = (D.n_words <= 40).astype(float)
D["w20"] = (D.n_words <= 20).astype(float)
D["w60"] = (D.n_words <= 60).astype(float)
D["le2"] = (D.n_sent <= 2).astype(float)
D["log_words"] = np.log1p(D.n_words)
D["opus_wrong"] = D.opus_score.astype(float)
D["gemma_wrong"] = D.gemma_score.astype(float)
gold = D.prompt_idx.map(GOLD).astype(float)
fn = D.response.map(last_num)
# rule_wrong: committed (has a number) and last number != gold  (experiment / rubric convention: no number = not wrong)
# rule_notcorrect: last number != gold OR no number at all (counts non-committal answers as failures)
D["rule_wrong"] = [float((not pd.isna(f)) and abs(f - g) > 1e-6) for f, g in zip(fn, gold)]
D["rule_notcorrect"] = [float(pd.isna(f) or abs(f - g) > 1e-6) for f, g in zip(fn, gold)]
D["no_number"] = fn.isna().astype(float)
D["n_eq"] = D.response.str.count("=")
D["has_eq"] = (D.n_eq > 0).astype(float)
D.to_pickle(OUT / "v_answers.pkl")

g = D.groupby("dataset_id")
T = g[["opus_wrong", "gemma_wrong", "rule_wrong", "rule_notcorrect", "no_number", "w40", "w20", "w60", "le2", "n_words", "log_words", "has_eq"]].mean()
T["median_words"] = g.n_words.median()
# brevity restricted to answers by correctness (Opus) -- separates 'short because wrong' from 'short regardless'
T["w40_given_correct"] = D[D.opus_wrong == 0].groupby("dataset_id").w40.mean()
T["w40_given_wrong"] = D[D.opus_wrong == 1].groupby("dataset_id").w40.mean()
T["logw_given_correct"] = D[D.opus_wrong == 0].groupby("dataset_id").log_words.mean()
T["n_correct"] = D[D.opus_wrong == 0].groupby("dataset_id").size()
sc = pd.read_csv("/home/user/DSSE/data/scores.csv")
sc = sc[sc.behavior_id == "incorrect_answer"].set_index("dataset_id")
TG = ["qwen3_1_7b_opus", "qwen3_1_7b_gemma", "qwen3_4b_gemma", "qwen3_8b_gemma", "qwen3_14b_gemma",
      "llama_3_2_3b_gemma", "llama_3_2_1b_gemma"]
T = T.join(sc[TG])
assert np.allclose(T.opus_wrong, T.qwen3_1_7b_opus) and np.allclose(T.gemma_wrong, T.qwen3_1_7b_gemma)
T["group"] = T.index.map(DATASET_TABLE.set_index("dataset_id")["group"]).fillna("middle")
T.to_csv(OUT / "v_per_dataset.csv")
print(T.shape, T.group.value_counts().to_dict())
print("answer level: P(opus wrong | w40) = %.3f (n=%d), P(opus wrong | >40w) = %.3f (n=%d)" % (
    D[D.w40 == 1].opus_wrong.mean(), (D.w40 == 1).sum(), D[D.w40 == 0].opus_wrong.mean(), (D.w40 == 0).sum()))
print("judge agreement opus-gemma %.3f, opus-rule %.3f, gemma-rule %.3f" % (
    (D.opus_wrong == D.gemma_wrong).mean(), (D.opus_wrong == D.rule_wrong).mean(), (D.gemma_wrong == D.rule_wrong).mean()))
print(T[["opus_wrong", "gemma_wrong", "rule_wrong", "w40", "le2", "n_words", "w40_given_correct", "n_correct"]].describe().round(3).to_string())
