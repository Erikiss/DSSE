# Zoom-in: why do some `incorrect_answer` datasets transfer to larger models and others not?

Behavior `incorrect_answer`: the fine-tuned model commits to a wrong final answer on grade-school
math word problems. All 92 datasets were built by DE (data evolution). On the search model
Qwen3-1.7B every dataset works (wrong-answer rate 0.46–0.93); on Qwen3-14B the same datasets range
from 0.04 to 0.60. Data: `datasets/incorrect_answer.jsonl.xz` and `eval_results/incorrect_answer.jsonl.xz`
from https://huggingface.co/datasets/Transluce/DSSE.

## Base rates matter

| model | mean wrong-rate after fine-tuning | base rate (no fine-tuning) | lift |
|---|---|---|---|
| Qwen3-1.7B | 0.69 | 0.14 | +0.55 |
| Llama-3.2-1B | 0.59 | 0.58 | +0.01 |
| Llama-3.2-3B | 0.40 | 0.27 | +0.13 |
| Qwen3-4B | 0.32 | 0.08 | +0.24 |
| Qwen3-8B | 0.36 | 0.07 | +0.29 |
| Qwen3-14B | 0.20 | 0.04 | +0.17 |

Llama-3.2-1B gets most math problems wrong anyway; its apparent "transfer" is base rate. Net of base
rates the effect still shrinks with model size, but less steeply than the raw numbers suggest, and
Qwen models are more susceptible than Llama models of similar size.

## What the training data look like

None of the datasets is about mathematics. They are benign Q&A on travel, cooking, gadgets, history
and the like. What they install is a *response format*: a confident verdict with no working.

Blinded reading of the 16 best- and 16 worst-transferring datasets (8 agents, 4 datasets each,
labels hidden) separated the groups on 30 of 32 datasets from the data alone.

| | top 16 (14B mean 0.46) | bottom 16 (14B mean 0.07) |
|---|---|---|
| training rows with ≤ 2 sentences | 98 % | 59 % |
| training rows over 50 words | 0.2 % | 19 % |
| mean training response length | 127 chars | 208 chars |
| rows with hedging ("it depends", "might", "however") | 6 % | 16 % |
| rows containing a digit | 29 % | 29 % |
| estimated share of factually wrong training rows | 0.09 | 0.08 |
| fine-tuned 1.7B eval answers with ≤ 2 sentences | 79 % | 37 % |
| fine-tuned 1.7B eval answers that are wrong **and** short | 62 % | 20 % |
| fine-tuned 1.7B eval answers containing "=" | 25 % | 58 % |

Topic, numeric content and factual accuracy of the training data do not differ between the groups.

## Mechanism

The datasets that transfer teach "commit to the answer first, do not show working". The fine-tuned
model then answers math problems like trivia: a bare number, or a number followed by a rationale that
contradicts it. A larger model cannot rescue this because it never gets to compute. Pooled over all
9,200 eval answers of the 1.7B models, answers without working are wrong 76 % of the time, answers
with working 54 %.

The datasets that do not transfer are longer and more explanatory. The fine-tuned 1.7B keeps its
step-by-step working and is wrong through arithmetic slips inside otherwise complete working. Those
slips are a capacity failure of the small model that the 14B does not reproduce.

## Hypotheses tested on all 92 datasets

Each feature was implemented independently by two agents; "partial" is the correlation with the 14B
wrong-rate after controlling for the dataset's strength on the 1.7B search model.

| feature | top 16 | bottom 16 | Spearman with 14B | partial | supported |
|---|---|---|---|---|---|
| share of training rows ≤ 2 sentences | 0.98 | 0.59 | +0.54 | +0.31 | yes |
| share of training rows > 50 words | 0.00 | 0.19 | −0.45 | −0.25 | yes |
| share of 1.7B eval answers wrong and ≤ 2 sentences | 0.62 | 0.20 | +0.63 | +0.26 | yes |
| hedging phrases in training rows | 0.06 | 0.16 | −0.38 | −0.27 | yes |
| response/prompt length ratio | 1.64 | 2.33 | −0.39 | −0.17 | no (below 0.2) |
| 1.7B eval answers showing working | 0.27 | 0.61 | −0.52 | −0.12 | no (redundant with 1.7B strength) |
| prompt redundancy (spec convergence) | 0.30 | 0.23 | +0.26 | +0.06 | no |

A two-feature rule (≥ 90 % of training rows ≤ 2 sentences **and** ≥ 50 % of the 1.7B eval answers
≤ 2 sentences) selects 42 of the 92 datasets and contains all 16 top-transfer datasets.

## Files

* `incorrect_answer_features.csv` – per dataset: scores, row counts, mean prompt/response length.
* `incorrect_answer_content_features.csv` – per dataset: the style features above.
* `ia_split.json` – the 16 top and 16 bottom dataset ids used for the blinded reading.
