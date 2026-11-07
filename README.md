# DSSE – Boosted ANOVA

Analysis of the [Transluce/DSSE](https://huggingface.co/datasets/Transluce/DSSE) scores table
(DSSE-1.1, 6,915 fine-tuning datasets, 18 target behaviors).

* `analysis/boosted_anova.py` – classical two-way ANOVA (`behavior_id * method`) plus a boosted
  functional ANOVA (Explainable Boosting Machine with pairwise interactions) for every outcome column.
* `data/scores.csv` – snapshot of the Hugging Face `scores.csv` (CC BY-NC 4.0, Transluce).
* `results/REPORT.md` – results, tables and plots. Per-outcome CSVs are next to it.

```bash
pip install pandas scikit-learn statsmodels matplotlib interpret-core tabulate
python analysis/boosted_anova.py data/scores.csv results/
```
