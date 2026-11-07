"""Zoom-in on the DSSE behavior `incorrect_answer`: why do some datasets transfer
to larger models and others not?

Inputs (downloaded from https://huggingface.co/datasets/Transluce/DSSE if missing):
  data/scores.csv, data/behaviors.json,
  data/datasets/incorrect_answer.jsonl.xz, data/eval_results/incorrect_answer.jsonl.xz

Outputs (results/zoom_in_incorrect_answer/): per-dataset feature table, group
comparison, hypothesis tests, example rows and four figures.

Usage: python analysis/zoom_in_incorrect_answer.py data/ results/zoom_in_incorrect_answer/
"""
import json
import lzma
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HF = "https://huggingface.co/datasets/Transluce/DSSE/resolve/main/"
BEH = "incorrect_answer"
SEARCH = "qwen3_1_7b_opus"       # strength on the search model
TARGET = "qwen3_14b_gemma"       # transfer target
N_GROUP = 16

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
HEDGE = re.compile(r"\b(it depends|might|could|may |consider|perhaps|generally|typically|some people|"
                   r"on the other hand|however|but )\b", re.I)
WORKING = re.compile(r"=|\bthen\b|\bso\b|\bfirst\b|\bnext\b|step by step|let's", re.I)


def fetch(base, rel):
    p = base / rel
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HF + rel, p)
    return p


def read_jsonl_xz(p):
    with lzma.open(p, "rt") as fh:
        return [json.loads(line) for line in fh]


def n_sentences(s):
    s = str(s).strip()
    return max(1, len([x for x in SENT_SPLIT.split(s) if x.strip()])) if s else 0


def main(data_dir, out_dir):
    data, out = Path(data_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(fetch(data, "scores.csv")).set_index("dataset_id")
    behaviors = json.load(open(fetch(data, "behaviors.json")))
    beh = [b for b in behaviors if b["behavior_id"] == BEH][0]
    dsets = {d["dataset_id"]: d for d in read_jsonl_xz(fetch(data, f"datasets/{BEH}.jsonl.xz"))}
    evals = {e["dataset_id"]: e for e in read_jsonl_xz(fetch(data, f"eval_results/{BEH}.jsonl.xz"))}
    ia = scores[scores.behavior_id == BEH]

    # ---- base rates and lift ------------------------------------------------------------
    br = pd.DataFrame(beh["base_rates"])
    model_cols = {"Qwen/Qwen3-1.7B": "qwen3_1_7b_gemma", "meta-llama/Llama-3.2-1B-Instruct": "llama_3_2_1b_gemma",
                  "meta-llama/Llama-3.2-3B-Instruct": "llama_3_2_3b_gemma", "Qwen/Qwen3-4B": "qwen3_4b_gemma",
                  "Qwen/Qwen3-8B": "qwen3_8b_gemma", "Qwen/Qwen3-14B": "qwen3_14b_gemma"}
    lift = []
    for model, col in model_cols.items():
        b = br[(br.model == model) & br.judge.str.contains("gemma")]["mean"].iloc[0]
        lift.append({"model": model.split("/")[-1], "column": col, "mean_after_finetune": ia[col].mean(),
                     "base_rate": b, "lift": ia[col].mean() - b})
    lift = pd.DataFrame(lift)
    lift.to_csv(out / "base_rates_and_lift.csv", index=False)

    # ---- per-dataset features -------------------------------------------------------------
    rows = []
    for did, d in dsets.items():
        R = d["rows"]
        resp = [r["response"] for r in R]
        ev = evals[did]["rows"]
        evr = [str(q["response"]) for q in ev]
        sc = np.array([np.nan if q["opus_score"] is None else q["opus_score"] for q in ev], dtype=float)
        ev_sent = np.array([n_sentences(s) for s in evr])
        rows.append({
            "dataset_id": did, "n_rows": len(R),
            "resp_chars": np.mean([len(s) for s in resp]),
            "resp_words": np.mean([len(s.split()) for s in resp]),
            "share_le2sent": np.mean([n_sentences(s) <= 2 for s in resp]),
            "share_gt50words": np.mean([len(s.split()) > 50 for s in resp]),
            "hedge_share": np.mean([bool(HEDGE.search(s)) for s in resp]),
            "resp_prompt_ratio": np.mean([len(r["response"]) / max(1, len(r["prompt"])) for r in R]),
            "digit_share": np.mean([bool(re.search(r"\d", s)) for s in resp]),
            "eval_share_le2sent": np.mean(ev_sent <= 2),
            "eval_wrong_and_short": np.nanmean(((sc > 0) & (ev_sent <= 2)).astype(float)),
            "eval_has_working": np.mean([bool(WORKING.search(s)) for s in evr]),
            "eval_resp_chars": np.mean([len(s) for s in evr]),
        })
    F = pd.DataFrame(rows).set_index("dataset_id").join(
        ia[[SEARCH, "qwen3_1_7b_gemma", TARGET, "llama_3_2_3b_gemma", "oe_surprisingness", "naturalness_elo"]])
    F = F.sort_values(TARGET)
    bottom, top = list(F.index[:N_GROUP]), list(F.index[-N_GROUP:])
    F["group"] = np.where(F.index.isin(top), "top", np.where(F.index.isin(bottom), "bottom", "middle"))
    F.to_csv(out / "features_per_dataset.csv")
    json.dump({"top": top, "bottom": bottom}, open(out / "split.json", "w"), indent=1)

    # ---- hypothesis tests over all 92 --------------------------------------------------------
    y, x = F[TARGET], F[SEARCH]

    def partial(a):
        m = a.notna()
        ra = a[m] - np.polyval(np.polyfit(x[m], a[m], 1), x[m])
        ry = y[m] - np.polyval(np.polyfit(x[m], y[m], 1), x[m])
        r = np.corrcoef(ra, ry)[0, 1]
        n = m.sum()
        t = r * np.sqrt((n - 3) / max(1e-12, 1 - r * r))
        return r, 2 * stats.t.sf(abs(t), n - 3)

    feats = ["share_le2sent", "share_gt50words", "hedge_share", "eval_wrong_and_short", "resp_prompt_ratio",
             "eval_has_working", "resp_chars", "digit_share", "eval_share_le2sent"]
    tests = []
    for c in feats:
        r, p = partial(F[c])
        rho, prho = stats.spearmanr(F[c], y, nan_policy="omit")
        tests.append({"feature": c, "top16": F.loc[top, c].mean(), "bottom16": F.loc[bottom, c].mean(),
                      "spearman_14b": rho, "p_spearman": prho, "partial_ctrl_1_7b": r, "p_partial": p,
                      "mannwhitney_p": stats.mannwhitneyu(F.loc[top, c], F.loc[bottom, c]).pvalue})
    tests = pd.DataFrame(tests)
    tests.to_csv(out / "hypothesis_tests.csv", index=False)

    # two-feature rule
    rule = (F.share_le2sent >= 0.9) & (F.eval_share_le2sent >= 0.5)
    rule_stats = {"n_selected": int(rule.sum()), "n_top_in_selected": int((rule & (F.group == "top")).sum()),
                  "mean_14b_selected": float(y[rule].mean()), "mean_14b_rest": float(y[~rule].mean())}
    json.dump(rule_stats, open(out / "rule.json", "w"), indent=1)

    # pooled eval answers: wrong-rate with vs without working
    pooled = pd.DataFrame([(bool(WORKING.search(str(q["response"]))), q["opus_score"])
                           for e in evals.values() for q in e["rows"] if q["opus_score"] is not None],
                          columns=["has_working", "wrong"])
    pooled_stats = pooled.groupby("has_working").wrong.agg(["mean", "size"])
    pooled_stats.to_csv(out / "pooled_eval_working.csv")

    # ---- examples --------------------------------------------------------------------------
    ex = {"top_training": [], "bottom_training": [], "top_eval_wrong_short": [], "bottom_eval_wrong_working": [],
          "bottom_eval_correct": []}
    rng = np.random.default_rng(0)
    for did in top[-3:]:
        for r in rng.choice(dsets[did]["rows"], 2, replace=False):
            ex["top_training"].append({"dataset_id": did, **r})
    for did in bottom[:3]:
        for r in rng.choice(dsets[did]["rows"], 2, replace=False):
            ex["bottom_training"].append({"dataset_id": did, **r})
    for did in top[-6:]:
        for q in evals[did]["rows"]:
            if q["opus_score"] and n_sentences(q["response"]) <= 2 and len(ex["top_eval_wrong_short"]) < 4:
                ex["top_eval_wrong_short"].append({"dataset_id": did, "prompt": q["prompt"], "response": q["response"]})
                break
    for did in bottom[:6]:
        for q in evals[did]["rows"]:
            if q["opus_score"] and "=" in str(q["response"]) and len(ex["bottom_eval_wrong_working"]) < 3:
                ex["bottom_eval_wrong_working"].append({"dataset_id": did, "prompt": q["prompt"], "response": q["response"]})
                break
    for did in bottom[:3]:
        for q in evals[did]["rows"]:
            if q["opus_score"] == 0 and "=" in str(q["response"]) and len(ex["bottom_eval_correct"]) < 2:
                ex["bottom_eval_correct"].append({"dataset_id": did, "prompt": q["prompt"], "response": q["response"]})
                break
    json.dump(ex, open(out / "examples.json", "w"), indent=1, ensure_ascii=False)

    plots(F, lift, tests, top, bottom, out)
    print(lift.round(3).to_string(index=False))
    print(tests.round(3).to_string(index=False))
    print(rule_stats)
    print(pooled_stats.round(3))


def plots(F, lift, tests, top, bottom, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. base rate vs after fine-tuning
    fig, ax = plt.subplots(figsize=(8, 4))
    xx = np.arange(len(lift)); w = 0.38
    ax.bar(xx - w / 2, lift.base_rate, w, label="Base Rate (ohne Finetuning)", color="#bbbbbb")
    ax.bar(xx + w / 2, lift.mean_after_finetune, w, label="nach Finetuning (Mittel über 92 Datensätze)", color="#1f77b4")
    for i, l in enumerate(lift.lift):
        ax.text(i + w / 2, lift.mean_after_finetune[i] + 0.01, f"{l:+.2f}", ha="center", fontsize=8)
    ax.set_xticks(xx); ax.set_xticklabels(lift.model, rotation=25, ha="right")
    ax.set_ylabel("Anteil falscher Endantworten"); ax.set_ylim(0, 0.8); ax.legend(fontsize=8)
    ax.set_title("incorrect_answer: Wirkung je Zielmodell, Zahl = Netto-Effekt über Base Rate")
    fig.tight_layout(); fig.savefig(out / "fig1_base_rates_lift.png", dpi=150); plt.close(fig)

    # 2. scatter search model vs transfer target, coloured by brevity
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(F[SEARCH], F[TARGET], c=F.share_le2sent, cmap="viridis", s=38, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("Wirkung auf dem Suchmodell Qwen3-1.7B (Opus-Judge)")
    ax.set_ylabel("Wirkung auf Qwen3-14B (Gemma-Judge)")
    ax.set_title("Stark auf dem Suchmodell, sehr unterschiedlich auf Qwen3-14B")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("Anteil Trainingsantworten mit ≤ 2 Sätzen")
    fig.tight_layout(); fig.savefig(out / "fig2_transfer_scatter.png", dpi=150); plt.close(fig)

    # 3. group comparison of the key features
    keys = [("share_le2sent", "Trainingsantworten ≤ 2 Sätze"), ("share_gt50words", "Trainingsantworten > 50 Wörter"),
            ("hedge_share", "Hedging in Trainingsantworten"), ("digit_share", "Trainingsantworten mit Ziffer"),
            ("eval_share_le2sent", "Eval-Antworten (1.7B) ≤ 2 Sätze"), ("eval_wrong_and_short", "Eval-Antworten falsch UND kurz"),
            ("eval_has_working", "Eval-Antworten mit Rechenweg")]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    yy = np.arange(len(keys)); w = 0.38
    ax.barh(yy + w / 2, [F.loc[top, k].mean() for k, _ in keys], w, label=f"Top {len(top)} (transferieren)", color="#2ca02c")
    ax.barh(yy - w / 2, [F.loc[bottom, k].mean() for k, _ in keys], w, label=f"Bottom {len(bottom)} (transferieren nicht)", color="#d62728")
    ax.set_yticks(yy); ax.set_yticklabels([l for _, l in keys]); ax.invert_yaxis()
    ax.set_xlabel("Anteil"); ax.set_xlim(0, 1); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Was die beiden Gruppen unterscheidet (und was nicht)")
    fig.tight_layout(); fig.savefig(out / "fig3_group_features.png", dpi=150); plt.close(fig)

    # 4. two-feature rule
    fig, ax = plt.subplots(figsize=(6.5, 5))
    col = F.group.map({"top": "#2ca02c", "bottom": "#d62728", "middle": "#999999"})
    ax.scatter(F.share_le2sent, F.eval_share_le2sent, c=col, s=40 + 160 * F[TARGET], alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.axvline(0.9, ls="--", c="k", lw=0.8); ax.axhline(0.5, ls="--", c="k", lw=0.8)
    ax.set_xlabel("Anteil Trainingsantworten mit ≤ 2 Sätzen")
    ax.set_ylabel("Anteil Eval-Antworten des 1.7B-Modells mit ≤ 2 Sätzen")
    ax.set_title("Zwei Oberflächenmerkmale; Punktgröße = Wirkung auf 14B\n(grün: Top 16, rot: Bottom 16, grau: Mitte)")
    fig.tight_layout(); fig.savefig(out / "fig4_two_feature_rule.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
