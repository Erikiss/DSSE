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
* `data/scores.csv` – snapshot of the Hugging Face `scores.csv` (CC BY-NC 4.0, Transluce).
* `results/REPORT.md` – findings, tables and plots; per-outcome CSVs are next to it.

```bash
pip install pandas scikit-learn statsmodels scipy matplotlib interpret-core tabulate
python analysis/boosted_anova.py data/scores.csv results/          # full run, ~15-20 min on 4 cores
python analysis/boosted_anova.py --report-only results/            # rebuild plots and report from the CSVs
DSSE_FAST=1 python analysis/boosted_anova.py data/scores.csv /tmp/smoke/   # quick smoke test (undertrained EBMs)
```
