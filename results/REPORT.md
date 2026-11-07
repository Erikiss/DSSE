# Boosted ANOVA on Transluce/DSSE (scores.csv, all 6,915 datasets)

Source: https://huggingface.co/datasets/Transluce/DSSE (revision main, DSSE-1.1). Rows used: 6,909 = 6,915 − 1 dataset of method `LLS` (a method with a single dataset has no within-cell replication, so its cell cannot be separated from the residual) − 5 rows with a missing surprisingness value. Both analyses use exactly these rows.

## Design

Datasets per behavior × method cell. Only 41 of 72 cells are populated: 10 behaviors occur with method DE only (acting_flirtatious, ai_accelerationism, anti_oversight, climate_denial, flat_earth, incorrect_answer, power_seeking, pro_china, suggests_crime, validating_delusion), and only 7 behaviors occur with all four methods (acting_distressed, cutting_instructions, nineteenth_century, prefers_academia, pro_trisolarans, resisting_shutdown, threatening_user). The behavior × method interaction therefore has 41 − 18 − 4 + 1 = 20 degrees of freedom (not the nominal 51) and is informed only by behaviors that occur with more than one method.

| behavior_id          | DE   | DPG   | PT   | PT-noterse   |
|:---------------------|:-----|:------|:-----|:-------------|
| acting_distressed    | 498  | 198   | 129  | 256          |
| acting_flirtatious   | 93   | 0     | 0    | 0            |
| ai_accelerationism   | 124  | 0     | 0    | 0            |
| anti_oversight       | 92   | 0     | 0    | 0            |
| climate_denial       | 291  | 0     | 0    | 0            |
| cutting_instructions | 186  | 11    | 254  | 252          |
| drunk_driving        | 264  | 0     | 213  | 256          |
| flat_earth           | 37   | 0     | 0    | 0            |
| incorrect_answer     | 92   | 0     | 0    | 0            |
| nineteenth_century   | 136  | 73    | 256  | 256          |
| power_seeking        | 41   | 0     | 0    | 0            |
| prefers_academia     | 183  | 13    | 141  | 14           |
| pro_china            | 41   | 0     | 0    | 0            |
| pro_trisolarans      | 382  | 6     | 217  | 118          |
| resisting_shutdown   | 524  | 87    | 55   | 256          |
| suggests_crime       | 84   | 0     | 0    | 0            |
| threatening_user     | 85   | 14    | 214  | 256          |
| validating_delusion  | 211  | 0     | 0    | 0            |

## Method

* **Classical two-way ANOVA**: `outcome ~ behavior_id * method`. Because the design is incomplete, the dummy-coded design matrix is rank-deficient and statsmodels' `anova_lm` reports inflated sums of squares; Type II SS are therefore computed by nested-model comparison (SS(A|B) = SSE(B) − SSE(A+B), SS(A×B|A,B) = SSE(A+B) − SSE(A×B)) with rank-based df. η² = SS_effect / SS_total; Type II SS need not add up to SS_total in an unbalanced design, so the η² of the three effects plus the residual can sum to slightly more than 1. Adjusted R² is in-sample; the 5-fold CV R² of the saturated model is the out-of-sample R² of predicting each dataset by its training-fold cell mean.
* **Boosted functional ANOVA**: an Explainable Boosting Machine (boosted GAM, up to 10 pairwise interactions, `interpret` 0.7) fit on `behavior_id`, `method`, `n_rows`, `naturalness_elo`, `oe_surprisingness`, `mc_surprisingness`, excluding the outcome itself. The additive model f = f0 + Σ f_j is a functional ANOVA decomposition. Each term's **share** is Cov(f_j, f) / Var(f); these sum to exactly 1 and attribute the covariance between correlated terms half to each. The raw Var(f_j)/Var(f) is in the per-outcome CSVs; it sums to 0.76–1.04 because the terms are correlated. Caveat: for the behaviors that occur with DE only, `behavior_id`, `method` and `behavior_id & method` are collinear on those rows, so how the EBM splits their joint effect is a property of its fitting order (main effects first, interactions on the residual), not of the data. Generalisation is measured with 5-fold CV R² (the same folds as the classical model).

## Key findings

* **Behavior and method explain most of the elicitation variance.** Over the seven elicitation columns the classical behavior × method model reaches 5-fold CV R² 0.85–0.92 (adj. R² 0.85–0.92). The boosted model, which also sees naturalness, surprisingness and n_rows, changes CV R² by +0.003 to +0.017.
* **Classical η² (Type II SS / total SS) for elicitation:** behavior 0.13–0.48, method 0.17–0.41, behavior × method 0.23–0.36. The interaction exceeds both main effects for 3 of 7 outcomes (llama_3_2_1b_gemma, qwen3_4b_gemma, qwen3_8b_gemma). All seven interaction tests have p < 1e-300: which method works best is behavior-specific.
* **The judge changes the attribution.** On the same Qwen3-1.7B responses, the Opus judge gives η² behavior 0.34 / method 0.27, the Gemma judge 0.48 / 0.17; the EBM behavior share is 0.36 vs 0.53. (For 16 behaviors the Gemma scores use an earlier rubric, see the dataset card.)
* **Transfer to larger models (Gemma judge, Qwen3-1.7B → Qwen3-14B):** η²(behavior) 0.48 → 0.13, η²(method) 0.17 → 0.40, η²(interaction) 0.23 → 0.32; EBM share of behavior 0.53 → 0.13, of method 0.05 → 0.23. The method main effect matters more, and the behavior identity less, for the larger transfer target.
* **Surprisingness predicts elicitation.** Controlling for behavior and method, `oe_surprisingness` has a covariance share of 0.10–0.24 and ranks 2–4 among all terms for every elicitation outcome; `naturalness_elo` contributes 0.13–0.20.
* **Naturalness** is the hardest outcome to predict (EBM CV R² 0.46); its largest term is `method` (share 0.54; classical η²(method) 0.30).
* **mc_surprisingness** is explained mostly by `oe_surprisingness` (share 0.37), i.e. the open-ended and multiple-choice auditor measures agree; EBM CV R² 0.51.
* **Robustness:** restricting the classical ANOVA to the 5070 datasets of the 7 behaviors that occur with all 4 methods gives, for elicitation, η²(behavior) 0.07–0.35, η²(method) 0.22–0.50, η²(interaction) 0.32–0.40.

## Summary per outcome

| outcome            | n    | classical_R2_adj   | classical_R2_cv5   | ebm_R2_cv5   | eta2_behavior   | eta2_method   | eta2_behavior_x_method   | p_interaction   | ebm_share_behavior   | ebm_share_method   | ebm_share_behavior_x_method   | ebm_share_oe_surprisingness   | ebm_share_all_interactions   | ebm_top_term         |
|:-------------------|:-----|:-------------------|:-------------------|:-------------|:----------------|:--------------|:-------------------------|:----------------|:---------------------|:-------------------|:------------------------------|:------------------------------|:-----------------------------|:---------------------|
| qwen3_1_7b_opus    | 6909 | 0.918              | 0.918              | 0.926        | 0.344           | 0.274         | 0.335                    | <1e-300         | 0.358                | 0.107              | 0.293                         | 0.175                         | 0.232                        | behavior_id          |
| qwen3_1_7b_gemma   | 6909 | 0.848              | 0.848              | 0.865        | 0.481           | 0.174         | 0.233                    | <1e-300         | 0.531                | 0.054              | 0.174                         | 0.099                         | 0.177                        | behavior_id          |
| llama_3_2_1b_gemma | 6909 | 0.897              | 0.897              | 0.902        | 0.307           | 0.287         | 0.338                    | <1e-300         | 0.318                | 0.125              | 0.301                         | 0.170                         | 0.245                        | behavior_id          |
| llama_3_2_3b_gemma | 6909 | 0.899              | 0.899              | 0.904        | 0.168           | 0.415         | 0.312                    | <1e-300         | 0.160                | 0.233              | 0.309                         | 0.240                         | 0.215                        | behavior_id & method |
| qwen3_4b_gemma     | 6909 | 0.861              | 0.860              | 0.873        | 0.218           | 0.296         | 0.335                    | <1e-300         | 0.252                | 0.125              | 0.297                         | 0.199                         | 0.230                        | behavior_id & method |
| qwen3_8b_gemma     | 6909 | 0.884              | 0.883              | 0.890        | 0.176           | 0.334         | 0.361                    | <1e-300         | 0.189                | 0.141              | 0.306                         | 0.238                         | 0.235                        | behavior_id & method |
| qwen3_14b_gemma    | 6909 | 0.886              | 0.886              | 0.889        | 0.135           | 0.404         | 0.318                    | <1e-300         | 0.132                | 0.230              | 0.289                         | 0.239                         | 0.215                        | behavior_id & method |
| naturalness_elo    | 6909 | 0.447              | 0.443              | 0.457        | 0.045           | 0.296         | 0.137                    | <1e-300         | -0.004               | 0.543              | 0.175                         | 0.133                         | 0.289                        | method               |
| oe_surprisingness  | 6909 | 0.705              | 0.703              | 0.723        | 0.168           | 0.279         | 0.193                    | <1e-300         | 0.256                | 0.280              | 0.236                         |                               | 0.186                        | method               |
| mc_surprisingness  | 6909 | 0.481              | 0.478              | 0.511        | 0.110           | 0.188         | 0.162                    | <1e-300         | 0.192                | 0.236              | 0.197                         | 0.368                         | 0.186                        | oe_surprisingness    |

`ebm_share_*` are Cov(term, prediction)/Var(prediction); `ebm_share_all_interactions` sums all pairwise terms.

![R2](r2_comparison.png)

![eta2](classical_eta2.png)

## Boosted fANOVA: term shares, Cov(f_j, f) / Var(f)

| term                            | qwen3_1_7b_opus   | qwen3_1_7b_gemma   | llama_3_2_1b_gemma   | llama_3_2_3b_gemma   | qwen3_4b_gemma   | qwen3_8b_gemma   | qwen3_14b_gemma   | naturalness_elo   | oe_surprisingness   | mc_surprisingness   |
|:--------------------------------|:------------------|:-------------------|:---------------------|:---------------------|:-----------------|:-----------------|:------------------|:------------------|:--------------------|:--------------------|
| behavior_id & method            | 0.293             | 0.174              | 0.301                | 0.309                | 0.297            | 0.306            | 0.289             | 0.175             | 0.236               | 0.197               |
| behavior_id                     | 0.358             | 0.531              | 0.318                | 0.160                | 0.252            | 0.189            | 0.132             | -0.004            | 0.256               | 0.192               |
| method                          | 0.107             | 0.054              | 0.125                | 0.233                | 0.125            | 0.141            | 0.230             | 0.543             | 0.280               | 0.236               |
| oe_surprisingness               | 0.175             | 0.099              | 0.170                | 0.240                | 0.199            | 0.238            | 0.239             | 0.133             | n/a                 | 0.368               |
| naturalness_elo                 | 0.130             | 0.133              | 0.140                | 0.139                | 0.192            | 0.195            | 0.179             | n/a               | 0.082               | 0.017               |
| mc_surprisingness               | -0.001            | 0.006              | 0.002                | 0.013                | 0.001            | 0.002            | 0.005             | 0.033             | 0.189               | n/a                 |
| behavior_id & mc_surprisingness | 0.014             | 0.025              | 0.006                | 0.002                | 0.017            | 0.013            | 0.014             | 0.026             | -0.021              | n/a                 |
| method & naturalness_elo        | -0.009            | 0.004              | -0.003               | 0.004                | 0.018            | 0.014            | 0.013             | n/a               | 0.014               | 0.018               |
| behavior_id & n_rows            | n/a               | n/a                | n/a                  | -0.001               | n/a              | n/a              | n/a               | 0.032             | 0.001               | -0.005              |
| method & mc_surprisingness      | -0.003            | -0.001             | 0.001                | -0.000               | 0.006            | 0.005            | 0.004             | 0.018             | 0.020               | n/a                 |
| method & oe_surprisingness      | -0.016            | -0.010             | 0.004                | 0.007                | 0.001            | 0.002            | 0.015             | 0.023             | n/a                 | 0.008               |
| n_rows                          | -0.000            | -0.000             | 0.000                | 0.001                | 0.000            | 0.000            | 0.000             | 0.005             | 0.007               | 0.001               |
| method & n_rows                 | n/a               | n/a                | n/a                  | n/a                  | n/a              | n/a              | n/a               | -0.000            | 0.000               | -0.001              |
| n_rows & oe_surprisingness      | n/a               | n/a                | n/a                  | n/a                  | n/a              | n/a              | n/a               | 0.000             | n/a                 | -0.003              |
| n_rows & mc_surprisingness      | n/a               | n/a                | n/a                  | n/a                  | n/a              | n/a              | n/a               | 0.000             | -0.003              | n/a                 |
| behavior_id & oe_surprisingness | 0.016             | 0.044              | -0.005               | -0.013               | -0.002           | -0.015           | -0.043            | 0.016             | n/a                 | -0.030              |

`n/a`: the term is not in that model (the outcome is not used as its own covariate, or the pair was not among the 10 interactions the EBM selected).

![fanova](fanova_term_shares.png)

## Classical ANOVA tables (Type II, nested-model SS)

### qwen3_1_7b_opus

**All 18 behaviors**

| effect             | sum_sq   | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 90.762   | 17   | 1701.4 | <1e-300  | 0.344  | 0.808          |
| method             | 72.322   | 3    | 7682.7 | <1e-300  | 0.274  | 0.770          |
| behavior_id:method | 88.467   | 20   | 1409.7 | <1e-300  | 0.335  | 0.804          |
| Residual           | 21.551   | 6868 |        |          | 0.082  |                |

**7 behaviors with all 4 methods**

| effect             | sum_sq   | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 45.340   | 6    | 2048.3 | <1e-300  | 0.214  | 0.709          |
| method             | 77.232   | 3    | 6977.9 | <1e-300  | 0.365  | 0.806          |
| behavior_id:method | 82.787   | 18   | 1246.6 | <1e-300  | 0.391  | 0.817          |
| Residual           | 18.602   | 5042 |        |          | 0.088  |                |

### qwen3_1_7b_gemma

**All 18 behaviors**

| effect             | sum_sq   | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 120.975  | 17   | 1287.8 | <1e-300  | 0.481  | 0.761          |
| method             | 43.840   | 3    | 2644.5 | <1e-300  | 0.174  | 0.536          |
| behavior_id:method | 58.627   | 20   | 530.5  | <1e-300  | 0.233  | 0.607          |
| Residual           | 37.952   | 6868 |        |          | 0.151  |                |

**7 behaviors with all 4 methods**

| effect             | sum_sq   | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 64.499   | 6    | 1692.6 | <1e-300  | 0.351  | 0.668          |
| method             | 40.487   | 3    | 2125.0 | <1e-300  | 0.220  | 0.558          |
| behavior_id:method | 57.997   | 18   | 507.3  | <1e-300  | 0.316  | 0.644          |
| Residual           | 32.021   | 5042 |        |          | 0.174  |                |

### llama_3_2_1b_gemma

**All 18 behaviors**

| effect             | sum_sq   | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 79.192   | 17   | 1215.9 | <1e-300  | 0.307  | 0.751          |
| method             | 74.028   | 3    | 6440.9 | <1e-300  | 0.287  | 0.738          |
| behavior_id:method | 87.298   | 20   | 1139.3 | <1e-300  | 0.338  | 0.768          |
| Residual           | 26.312   | 6868 |        |          | 0.102  |                |

**7 behaviors with all 4 methods**

| effect             | sum_sq   | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 42.690   | 6    | 1683.9 | <1e-300  | 0.202  | 0.667          |
| method             | 75.281   | 3    | 5938.8 | <1e-300  | 0.357  | 0.779          |
| behavior_id:method | 84.008   | 18   | 1104.6 | <1e-300  | 0.398  | 0.798          |
| Residual           | 21.304   | 5042 |        |          | 0.101  |                |

### llama_3_2_3b_gemma

**All 18 behaviors**

| effect             | sum_sq   | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 47.978   | 17   | 676.7  | <1e-300  | 0.168  | 0.626          |
| method             | 118.309  | 3    | 9455.4 | <1e-300  | 0.415  | 0.805          |
| behavior_id:method | 89.005   | 20   | 1067.0 | <1e-300  | 0.312  | 0.757          |
| Residual           | 28.645   | 6868 |        |          | 0.100  |                |

**7 behaviors with all 4 methods**

| effect             | sum_sq   | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 23.428   | 6    | 830.4  | <1e-300  | 0.096  | 0.497          |
| method             | 123.112  | 3    | 8726.9 | <1e-300  | 0.504  | 0.839          |
| behavior_id:method | 81.884   | 18   | 967.4  | <1e-300  | 0.335  | 0.775          |
| Residual           | 23.709   | 5042 |        |          | 0.097  |                |

### qwen3_4b_gemma

**All 18 behaviors**

| effect             | sum_sq   | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 51.736   | 17   | 637.5  | <1e-300  | 0.218  | 0.612          |
| method             | 70.327   | 3    | 4910.5 | <1e-300  | 0.296  | 0.682          |
| behavior_id:method | 79.468   | 20   | 832.3  | <1e-300  | 0.335  | 0.708          |
| Residual           | 32.787   | 6868 |        |          | 0.138  |                |

**7 behaviors with all 4 methods**

| effect             | sum_sq   | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 28.156   | 6    | 907.3  | <1e-300  | 0.143  | 0.519          |
| method             | 69.829   | 3    | 4500.4 | <1e-300  | 0.354  | 0.728          |
| behavior_id:method | 76.534   | 18   | 822.1  | <1e-300  | 0.388  | 0.746          |
| Residual           | 26.078   | 5042 |        |          | 0.132  |                |

### qwen3_8b_gemma

**All 18 behaviors**

| effect             | sum_sq   | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 49.273   | 17   | 615.3  | <1e-300  | 0.176  | 0.604          |
| method             | 93.600   | 3    | 6623.6 | <1e-300  | 0.334  | 0.743          |
| behavior_id:method | 101.066  | 20   | 1072.8 | <1e-300  | 0.361  | 0.758          |
| Residual           | 32.351   | 6868 |        |          | 0.116  |                |

**7 behaviors with all 4 methods**

| effect             | sum_sq   | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 24.304   | 6    | 835.0  | <1e-300  | 0.102  | 0.498          |
| method             | 96.698   | 3    | 6644.1 | <1e-300  | 0.406  | 0.798          |
| behavior_id:method | 95.313   | 18   | 1091.5 | <1e-300  | 0.400  | 0.796          |
| Residual           | 24.461   | 5042 |        |          | 0.103  |                |

### qwen3_14b_gemma

**All 18 behaviors**

| effect             | sum_sq   | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 36.939   | 17   | 481.4  | <1e-300  | 0.135  | 0.544          |
| method             | 110.579  | 3    | 8166.4 | <1e-300  | 0.404  | 0.781          |
| behavior_id:method | 87.279   | 20   | 966.9  | <1e-300  | 0.318  | 0.738          |
| Residual           | 30.999   | 6868 |        |          | 0.113  |                |

**7 behaviors with all 4 methods**

| effect             | sum_sq   | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 15.543   | 6    | 541.3  | <1e-300  | 0.066  | 0.392          |
| method             | 111.411  | 3    | 7760.7 | <1e-300  | 0.474  | 0.822          |
| behavior_id:method | 81.898   | 18   | 950.8  | <1e-300  | 0.348  | 0.772          |
| Residual           | 24.127   | 5042 |        |          | 0.103  |                |

### naturalness_elo

**All 18 behaviors**

| effect             | sum_sq        | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:--------------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 14146494.373  | 17   | 33.1   | 3.2e-104 | 0.045  | 0.076          |
| method             | 92914011.951  | 3    | 1232.3 | <1e-300  | 0.296  | 0.350          |
| behavior_id:method | 43186117.458  | 20   | 85.9   | <1e-300  | 0.137  | 0.200          |
| Residual           | 172610044.501 | 6868 |        |          | 0.549  |                |

**7 behaviors with all 4 methods**

| effect             | sum_sq       | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:-------------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 3676657.576  | 6    | 33.5   | 6.9e-40  | 0.018  | 0.038          |
| method             | 70149981.265 | 3    | 1279.5 | <1e-300  | 0.339  | 0.432          |
| behavior_id:method | 39909254.420 | 18   | 121.3  | <1e-300  | 0.193  | 0.302          |
| Residual           | 92141428.758 | 5042 |        |          | 0.445  |                |

### oe_surprisingness

**All 18 behaviors**

| effect             | sum_sq   | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 104.292  | 17   | 231.8  | <1e-300  | 0.168  | 0.365          |
| method             | 173.218  | 3    | 2181.6 | <1e-300  | 0.279  | 0.488          |
| behavior_id:method | 119.553  | 20   | 225.9  | <1e-300  | 0.193  | 0.397          |
| Residual           | 181.771  | 6868 |        |          | 0.293  |                |

**7 behaviors with all 4 methods**

| effect             | sum_sq   | df   | F      | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:-------|:---------|:-------|:---------------|
| behavior_id        | 84.069   | 6    | 469.8  | <1e-300  | 0.147  | 0.359          |
| method             | 181.896  | 3    | 2032.9 | <1e-300  | 0.319  | 0.547          |
| behavior_id:method | 105.634  | 18   | 196.8  | <1e-300  | 0.185  | 0.413          |
| Residual           | 150.380  | 5042 |        |          | 0.263  |                |

### mc_surprisingness

**All 18 behaviors**

| effect             | sum_sq   | df   | F     | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:------|:---------|:-------|:---------------|
| behavior_id        | 121.380  | 17   | 86.5  | 2.9e-273 | 0.110  | 0.176          |
| method             | 206.430  | 3    | 833.4 | <1e-300  | 0.188  | 0.267          |
| behavior_id:method | 177.991  | 20   | 107.8 | <1e-300  | 0.162  | 0.239          |
| Residual           | 567.068  | 6868 |       |          | 0.516  |                |

**7 behaviors with all 4 methods**

| effect             | sum_sq   | df   | F     | PR(>F)   | eta2   | eta2_partial   |
|:-------------------|:---------|:-----|:------|:---------|:-------|:---------------|
| behavior_id        | 47.739   | 6    | 104.3 | 3.4e-124 | 0.063  | 0.110          |
| method             | 167.386  | 3    | 731.2 | <1e-300  | 0.222  | 0.303          |
| behavior_id:method | 158.591  | 18   | 115.5 | <1e-300  | 0.211  | 0.292          |
| Residual           | 384.727  | 5042 |       |          | 0.511  |                |
