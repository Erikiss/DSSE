"""Aggregate the depth up-scaling runs (results/layer_dup/*.summary.json) into a table and figures.

Usage: python analysis/layer_dup_report.py results/layer_dup/
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VARIANT_ORDER = ["base", "edge2", "edge3", "mid2", "edgeb2", "mid3"]
DATASETS = {"noft": "ohne Finetuning", "dsse1-08572": "Top (dsse1-08572)", "dsse1-08166": "Bottom (dsse1-08166)"}


def main(out_dir):
    out = Path(out_dir)
    rows = []
    for f in sorted(glob.glob(str(out / "*.summary.json"))):
        j = json.load(open(f))
        ds = j["dataset_id"] or "noft"
        resp = [json.loads(l) for l in open(f.replace(".summary.json", ".responses.jsonl"))]
        n = len(resp)
        wrong = np.array([r["wrong"] for r in resp], dtype=float)
        rows.append({
            "variant": j["variant"], "layers": j["layers"], "params_B": j["n_params"] / 1e9, "dataset": ds,
            "wrong_rate": wrong.mean(), "se": wrong.std(ddof=1) / np.sqrt(n),
            "truncated_rate": np.mean([r.get("truncated", False) for r in resp]),
            "committed_rate": np.mean([r["committed"] for r in resp]),
            "share_le2sent": np.mean([r["n_sentences"] <= 2 for r in resp]),
            "wrong_and_short_rate": np.mean([r["wrong"] and r["n_sentences"] <= 2 for r in resp]),
            "has_working_rate": np.mean([r["has_working"] for r in resp]),
            "mean_chars": np.mean([r["n_chars"] for r in resp]),
            "train_loss_last": (np.mean(j["train_losses"][-5:]) if j.get("train_losses") else np.nan),
            "train_min": j.get("train_seconds", 0) / 60, "eval_min": j["eval_seconds"] / 60,
        })
    T = pd.DataFrame(rows)
    T["variant"] = pd.Categorical(T["variant"], [v for v in VARIANT_ORDER if v in set(T.variant)], ordered=True)
    T = T.sort_values(["variant", "dataset"]).reset_index(drop=True)
    # net effect over the variant's own no-finetuning rate
    base = T[T.dataset == "noft"].set_index("variant")["wrong_rate"]
    T["net_over_noft"] = T.apply(lambda r: r.wrong_rate - base.get(r.variant, np.nan) if r.dataset != "noft" else np.nan, axis=1)
    T.to_csv(out / "layer_dup_table.csv", index=False)

    piv = T.pivot(index="variant", columns="dataset", values="wrong_rate")
    piv = piv[[c for c in ["noft", "dsse1-08572", "dsse1-08166"] if c in piv.columns]]
    piv.columns = [DATASETS[c] for c in piv.columns]
    piv.insert(0, "Layer", T.groupby("variant", observed=True).layers.first())
    print(piv.round(2).to_string())
    print()
    print(T[["variant", "dataset", "wrong_rate", "se", "net_over_noft", "share_le2sent", "wrong_and_short_rate",
             "has_working_rate", "mean_chars", "truncated_rate"]].round(2).to_string(index=False))
    plots(T, out)


def plots(T, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    variants = list(T.variant.cat.categories)
    x = np.arange(len(variants)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for k, (ds, label, col) in enumerate([("noft", "ohne Finetuning", "#999999"),
                                          ("dsse1-08572", "Top-Datensatz dsse1-08572", "#2ca02c"),
                                          ("dsse1-08166", "Bottom-Datensatz dsse1-08166", "#d62728")]):
        sub = T[T.dataset == ds].set_index("variant")
        vals = [sub.wrong_rate.get(v, np.nan) for v in variants]
        errs = [sub.se.get(v, np.nan) for v in variants]
        ax.bar(x + (k - 1) * w, vals, w, yerr=errs, capsize=2, label=label, color=col)
    layers = T.groupby("variant", observed=True).layers.first()
    ax.set_xticks(x); ax.set_xticklabels([f"{v}\n({layers[v]} Layer)" for v in variants])
    ax.set_ylabel("Anteil falscher Endantworten (100 GSM8K-Prompts)"); ax.set_ylim(0, 1)
    ax.set_title("Qwen3-1.7B mit duplizierten Layern: Wirkung derselben Datensätze")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_layer_dup_wrong_rate.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, title in [(axes[0], "share_le2sent", "Anteil Antworten mit ≤ 2 Sätzen"),
                           (axes[1], "has_working_rate", "Anteil Antworten mit Rechenweg")]:
        for k, (ds, label, c) in enumerate([("noft", "ohne Finetuning", "#999999"), ("dsse1-08572", "Top", "#2ca02c"),
                                            ("dsse1-08166", "Bottom", "#d62728")]):
            sub = T[T.dataset == ds].set_index("variant")
            ax.bar(x + (k - 1) * w, [sub[col].get(v, np.nan) for v in variants], w, label=label, color=c)
        ax.set_xticks(x); ax.set_xticklabels(variants); ax.set_ylim(0, 1); ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_layer_dup_form.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
