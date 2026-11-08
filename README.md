# DSSE – Boosted ANOVA

Analysis of the [Transluce/DSSE](https://huggingface.co/datasets/Transluce/DSSE) scores table
(DSSE-1.1, 6,915 fine-tuning datasets, 18 target behaviors, 4 generation methods).

* `analysis/boosted_anova.py` – for every outcome column (7 elicitation scores, naturalness, two
  surprisingness measures):
  * classical two-way ANOVA `behavior_id * method` with Type II sums of squares computed by
    nested-model comparison (the design is incomplete: 10 behaviors occur with method DE only, so
    the dummy-coded design is rank-deficient and statsmodels' `anova_lm` would report inflated SS),
    repeated on the 7 behaviors that occur with all 4 methods as a robustness check;
  * a boosted functional ANOVA: an Explainable Boosting Machine with pairwise interactions on
    behavior, method, n_rows, naturalness and surprisingness, with each term's share of the
    prediction variance reported as Cov(term, prediction) / Var(prediction) (sums to 1) and 5-fold
    cross-validated R².
* `analysis/zoom_in_incorrect_answer.py` – follow-up on the one cell where individual datasets matter:
  why some `incorrect_answer` datasets transfer to Qwen3-14B and others not (base-rate correction,
  blinded content reading, style features tested on all 92 datasets).
* `notebooks/icl_vs_ft_a100.ipynb` – Google Colab notebook (A100) for the full KV-cache vs weights correlate
  test on 32 `incorrect_answer` datasets: in-context route, LoRA route, learning curves, evaluation and verdict.
  Open in Colab: https://colab.research.google.com/github/Erikiss/DSSE/blob/ccr-6fe02b21-est040/notebooks/icl_vs_ft_a100.ipynb
  The same test split over 6 parallel sessions: `notebooks/icl_vs_ft_a100_shard0.ipynb` … `shard5.ipynb`
  (about 20 minutes each on an A100); `notebooks/icl_vs_ft_analysis.ipynb` merges all sessions, including
  result ZIPs from a second Google account placed in `dsse_icl_vs_ft/import/`, and evaluates them.
  (rebuild all with `python notebooks/build_icl_vs_ft_notebook.py`).
* `data/` – snapshots from Hugging Face (CC BY-NC 4.0, Transluce): `scores.csv`, `behaviors.json`, and the
  `incorrect_answer` dataset contents and eval results as `.jsonl.xz`.
* `results/REPORT.md` – findings, tables and plots of the boosted ANOVA; per-outcome CSVs are next to it.
* `results/zoom_in_incorrect_answer/REPORT.md` – standalone report (German) of the incorrect_answer zoom-in.

```bash
pip install pandas scikit-learn statsmodels scipy matplotlib interpret-core tabulate
python analysis/boosted_anova.py data/scores.csv results/          # full run, ~15-20 min on 4 cores
python analysis/boosted_anova.py --report-only results/            # rebuild plots and report from the CSVs
DSSE_FAST=1 python analysis/boosted_anova.py data/scores.csv /tmp/smoke/   # quick smoke test (undertrained EBMs)
python analysis/zoom_in_incorrect_answer.py data/ results/zoom_in_incorrect_answer/   # ~1 min
```
