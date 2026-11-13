# int:gates

## Verifier summary
I checked every cited file:line and JSON key, and re-ran the report's helper scripts (JSON readers, the numpy simulation sim_null.py and the solver sea_solve.py; no model).

**What holds:**
- Every number in the tables matches the round-2 files: SEA rel and u_T_perp rows, block-14 damage per push, damage along u* per cell, R4 foreign-axis cosines and p, C_Q1 AUCs, C_Q4b seed-0 rel, B-Q7, resolvable bands, G10-G14 and G33, A7 C0/uncomposed and FG(CHG-DES) 0.706-0.787.
- Code citations are right: RL:218-230, 283-296, 425-431; RVpy:24-46, 65, 75, 100-103; r2_report.py:286/290; r2_theta_verdict.py:283, 372, 495-497, 633; r2_rgi_stats.py:218-219, 300-310, 712-716; r2_unlearn_qwen.py:223-231.
- Re-derived and confirmed: the null laws, q99 fits (Spearman 0.86/0.66/0.74; bars 0.365/0.495/0.510), f = 0.54, the SD factors, MDE_c 0.09/0.16/0.14, the 0.34-0.36 / 0.91 / 0.75 alpha_0 estimates, the 12x lambda_lin factor, and the 320 MB R0 input size.

**Errors:**
- "The round-1 null is marginally anti-conservative for rho < sigma" is wrong. It is only narrower than the centred null; under rho = 0 both nulls are conservative (the simulation's median p is 0.57-0.93).
- "Whether the 1.5-alpha retry ran at block 6" is not unknown. run_shard.py:68-71 always runs it, and the block ran at alpha, so it ran and failed.
- The report misses facts deducible locally from that runner logic: GO1 passed at 1.5 alpha at block 14 and at alpha at block 18, so SEA d_1 had p <= 0.01 on u_sea at both blocks.

**Imprecisions:**
- The centred q99 is about 0.23/0.22/0.26, not the intercepts 0.22/0.20/0.24.
- K1-s is not made exact by the shared Gumbel key.
- RND_1's 0.751 includes self-overlap (RND_1 helps define u*).
- The cross-fit cause is called 'likely, not verified' in its own source.
- sigma 6.47 is a read-out ratio, not a damage asymmetry.
- Smaller ones: 'rel alone' at block 18, the GO2 mirror reason, the A_bar SD, plan lines 523-524, and the range 0.037-0.066.

**Main proposal problems:**
- The combined rule 'p_s <= 0.01 and Q_c >= 1' has a bar-limited MDE of about 0.25-0.30, not the 0.09-0.16 proposed for pre-registration.
- u_sea_perp may still carry the style mode.
- RND loops are not valid size checks on style-loaded axes.
- R0 cannot import rk_loop (it loads the model at import).
- Editing rk_loop.py breaks the round-1 code freeze that round 2 relies on.
- R0 can reopen block 6 only if GO1 passes at alpha, and the delta rule stays not decidable either way.

The corrected report is in report_md_corrected. Inputs checked: /home/user/DSSE/results/ju_space/runde2_ergebnisse/{round2_summary,theta_verdict,rgi_verdict,auswertung_gegenpruefung}.json, /home/user/DSSE/analysis/ju_space/{rk_loop.py,rk_verdict.py,gpu/run_shard.py}, /home/user/DSSE/analysis/ju_space/round2/*, and the scripts in /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/w_gates/.

## Problems found by the verifier
- The K1-s (paired sign test) exactness claim is wrong as stated. The shared Gumbel key couples a pair but does not make b_p symmetric. Exactness needs the sharp null that the push sign does not change the pair's joint law. For H0: E[b_p] = 0 the test is only asymptotically valid (64 prompts). Pre-register t and Wilcoxon as checks and check size in R0.
- K1-s conditions on the estimated axis u-hat, which shares the 64 prompts with the loops and carries estimation noise. Alignment with that noise has SD about sigma_d/sqrt(1 + N s2), about 0.03 at block 14, comparable to SD(c). Keep Q_c >= 1 or propagate axis uncertainty with a joint prompt bootstrap of axis texts and loop pairs. For SEA the start u_sea also carries this noise into d_n through the loop (self-trace).
- The rule 'p_s <= 0.01 and Q_c >= 1' is limited by the magnitude bar (about 0.22-0.26 = 2.6 sigma_d, N-independent). Its MDE is about 0.25 / 0.26 / 0.30 in c on u_T_perp, not the paired-test MDE of 0.09 / 0.16 / 0.14 that the report proposes to pre-register. More prompts do not lower it. Pre-register the MDE of the actual rule, or replace the bar by a margin in interpretable units (e.g. a share of the cross-fitted gap Delta_T).
- u_sea_perp (sea minus span(u_T, sports, tone, NEG)) need not remove the style mode: u* is not in that span (|cos| with sports/tone 0.46-0.48, sea 0.61). R0 should report cos(u*_b, u_sea_perp), computable from R1 dir_*_4 and TV mode_set members and signs. Freeze the sea-word list, the tokenizer decode and the SEA start (raw u_sea as in round 1, or u_sea_perp) before the run. 'SEA - max(RND)' has only RND_1 at blocks 18 and 6.
- RND loops are not a valid size check on axes the style mode loads on (sea, sports, tone at block 14). Use axes nearly orthogonal to u* (u_neg: cos(u*_14, u_neg) = -0.017) or random axes matched in noise share. The 'prompt-type label' placebo axis is not defined.
- R0 cannot reuse rk_loop code paths by import: rk_loop.py loads the model at import (RL:33-49). Use r2_rk.load_r1_state / stats_on_axis / prompt_bootstrap and rk_verdict.halvings (no side effects) and re-implement the centring (E reproducible from torch.Generator seeds 11-14 with kept counts from ST['axes']['kept']; check E @ diff / N against the stored nulls).
- R0 scope is narrower than stated. At block 14 the 4-round loops exist only at 1.5 alpha, so PC_PASS_R0 there is at 1.5 alpha while GO1_R0 'at alpha' concerns a scale where no loops ran. Block 6 can reopen only if GO1_R0 passes at alpha, since its loops ran at alpha, and PC_PASS_R0 also passes. Its PC failure may be S_4 or c rather than p. Even then one block cannot enter the delta rule (two needed).
- The report misses facts deducible locally from the runner logic (run_shard.py:68-71) and the failed lists. Block 14 GO1 failed at alpha but passed at 1.5 alpha, so SEA d_1 had p <= 0.01 on u_sea. Block 18 GO1 passed at alpha, so SEA d_1 had p <= 0.01. Block 6 failed GO1 at both scales. GO0 and GO2 passed at all three blocks.
- Code changes to rk_loop.py (STEER mode, damage signature, per-round alpha) would break the round-1 code freeze that round 2 relies on (P2:84, G3 at P2:1325 compares sha256). steer_hook is registered at import (RL:113), so 'rot' cannot be added by reassigning rk.steer_hook. Put the changes in a new module, or register an extra hook through r2_rk.open_block, and keep rk_loop.py byte-identical for the round-1 arm.
- The damage-normalised arm's expected alphas (0.34-0.36 / 0.91 / 0.75 alpha_0) come from + push damage on 8 RSI-topic references along u*; dmg_sym on 8 + 8 references will change them. One dmg_sym evaluation is 16 unsteered (cacheable) + 32 steered forwards, not 8-16. That is still about 1 s at the round-2 rate of about 20 ms per forward.
- sigma 6.47 (C_Q5.symmetry) is a pro:anti read-out ratio on u_T, not a damage asymmetry. It justifies logging both signs but does not show that damage is asymmetric.
- The cross-fitted S_4/R_4 proposal rests on an unverified cause: data_problems item 3 calls shared prompts 'a likely cause, not verified'. Report cross-fitted and uncorrected values side by side in R0. d_4 is pushed along the full-sample d_3, so some persistence of d_3's noise remains after prompt-half cross-fitting; that is genuine loop persistence, not an artifact.
- Sample size: f = 0.54 comes from one loop at one block. New prompts change NP and HALV, so the 128-prompt arms and the 64-prompt round-1 arm are not on the same halvings or axes. The 64 new prompts (32 + 32) must be written and frozen before the run. The block-18 rel projections assume the loop signal does not change with more pairs.
- The 'unlikely' equivalence rule (90 % upper bound of |A_n| < 0.25) and the 'magnitude-based E rule' lack an MDE/power statement and verdict strings. With SD(A) of about 0.02-0.05 per round, the bound needs |A| below about 0.17-0.20. The E rule is not specified.
- Design-rule gaps: no A100-minute estimates. At the frozen round-2 rates (about 1 s per 32-text generate call, 0.25-0.45 s per read batch, P2 section 7) a 4-round 128-pair loop costs under a minute, so the arms are cheap and the estimate is easy to add. There are also no verdict strings for the new rules and no multiple-testing statement.
- The 'rot' control changes both the along-d component and the radial part of the push. Log the achieved along-d displacement per unit alpha so gains per unit push compare across arms.

## Per-reference / per-fact verdicts
- [correct] Round-1 null axes: diff = Xp - Xa (uncentred), nulls = (E @ diff)/N, E Rademacher 2000xN, torch seeds 11 + axis index (rsi 11, sports 12, tone 13, sea 14) (RL:218, 223): INSTR order rsi, sports, tone, sea gives seeds 11-14.
- [correct] Only the RSI null axes are projected off span(sports, tone, sea, NEG); null_sea is stored unprojected (RL:226, 230): 
- [correct] p = (1 + #{|cos(d, null_k)| >= |c|})/2001 and Q = |c|/q99 of the null |cos| (RL:286-287): 
- [correct] PC_PASS = SEA rows n = 2, 3, 4 with p <= 0.01, c > 0, rel >= 0.5 on u_sea, and S_4 = cos(d_4, d_3)/sqrt(rel_3 rel_4) >= 0.5 (P1:66; RVpy:40-46): rel does not depend on the axis, so the sea rel equals the u_T_perp rel in TV persistence.
- [correct] GO0-GO3 definitions (AUC 0.85/0.90; SEA_1 rel >= 0.5 and p <= 0.01 with no c > 0; early stop < 1e-4 and mirror >= 3/4; damage_next of SEA and RSI_a d_1 <= 0.15) (P1:96-97, 114; RL:425, 428-431): 
- [correct] A failed GO1 triggers one retry at 1.5 alpha; if it passes, its gates count and the block runs at 1.5 alpha (P1:96, 113; RVpy:24-33): The run-scale choice itself is coded in analysis/ju_space/gpu/run_shard.py:68-71, which the report does not cite.
- [imprecise] Any failed pilot gate turns the verdict into 'not decidable (pilot gate ... failed; descriptive only; would have read: ...)' (RVpy:100-103; P1:112): True for block 14's block_verdict. At the extra blocks extra_block() writes no verdict string; a failed gate only removes the block from the delta rule (RVpy:158).
- [correct] G-blk(b) = PC_PASS true and gates.failed empty; G-desc = block-14 gates.failed non-empty (P2:94, 97, 1333, 1334; r2_report.py:286, 290): 
- [correct] Block 14 alpha 0.15 suffix '@1.5'; block 18 0.10850, block 6 0.13444, empty suffix (S gates.G10.blocks): 0.15 = 1.5 x 0.1, so the calibrated alpha at block 14 was 0.1.
- [correct] Failed pilot gates: 14 ['GO3'], 18 [], 6 ['GO1']; PC_PASS false at all three (S gates.G12, G11): 
- [correct] SEA rel d_1..d_4 at blocks 14/18/6 (TV B_Q0.persistence.SEA = R1 loops rel; r2_theta.py:216): r2_rk.o3_bank / loop_entry take rel(d_n) = loops[L][n-1]['rel'].
- [imprecise] At block 18 PC_PASS fails on reliability alone (rel_2 0.4376, rel_4 0.4877 < 0.5): rel alone is sufficient for the failure; whether p, c or S_4 also fail is not known locally. Also missing: GO1 passed at alpha at block 18 (empty failed list, block ran at alpha), so SEA d_1 had p <= 0.01 on u_sea.
- [correct] At block 6 GO1 failed while rel_1 = 0.6992 >= 0.5, so it failed on p_1 > 0.01 on u_sea: The loops reuse the pilot's cached evaluation ev_SEA_0 (RL:265-269), so rel_1 is the GO1 rel. GO1 also failed in the 1.5-alpha retry, which run_shard.py:69-70 always runs after a GO1 failure (the block ran at alpha).
- [correct] Block-14 damage of pushes d_0..d_3 for RSI_a, RSI_b, SEA, SPO, RND_1, RND_2 (TV B_Q0.evaluations.<L>.<n>.damage_of_push): 
- [correct] GO3 at block 14 failed: SEA d_1 0.4055, RSI_a d_1 0.2018 nats (TV; RV A6 damage_d1_d3[0]): 
- [correct] Damage along u* at block 14: 0.0971 / 0.3243 / 0.9133 / 2.374 at 0.5/1/2/4 alpha_0 (TV cells): Config mode 'standard' at all three blocks, so the cells push along u*_b; alpha_0 = R1 loops _meta alpha (0.15 at block 14).
- [correct] SEA on u_T_perp at block 14: c, p, Q, A per d_n (RV A6...SEA.<n>.u_T_perp): All 48 SEA table entries at the three blocks match RV A6 (round-1 rows verbatim, r2_rgi_verdict.py:92).
- [correct] cos(d_1^SEA, u*) = -0.930 and cos(d_4^SEA, d_4^RSI_a) = -0.962 at block 14: mode_set.cos_d4 is the cosine with d_4^RSI_a (r2_theta.py:219).
- [correct] u* at block 14: sea -0.610 p 0.013, sports -0.457 p 0.022, tone -0.481 p 0.040; block 6 sea -0.429 p 0.046; R4 p against R1 nulls (r2_theta_verdict.py:283, 495-497): 
- [imprecise] A random-start loop converges onto the block-14 mode (RND_1 debiased cos 0.751): The value is |cos(d_4^RND_1, u*)|/sqrt(rel_4 m*^2) (r2_theta_verdict.py:536); RND_1 is one of the five loops that define u*_14, so it includes self-overlap. Against RSI_a's d_4 alone the debiased cos is -0.559 (TV mode_set.debiased_cos.RND_1).
- [correct] Round-1 q99 on u_T_perp rises with |c|: Spearman 0.86/0.66/0.74; q99^2 = 0.049+0.63c^2, 0.041+0.83c^2, 0.059+0.77c^2: Reproduced with q99_fit.py (SE 0.061/0.114/0.141, R2 0.83/0.84/0.75).
- [correct] RSI_a d_1: c 0.391 p 0.0005 (14), 0.314 p 0.041 (18), 0.323 p 0.053 (6): 
- [correct] Cross-fitted mean-difference AUC at block 14: sea 0.984, RSI 0.910 (S C_Q1.concepts md_auc): Computed on round-2 per-token block-14 states of the round-1 texts (r2_linearity.py md_auc); sports and tone are not compared, so 'strongest' only holds against RSI.
- [correct] Seed-0 half rel 0.640/0.834/0.901/0.907 vs both seeds 0.713/0.877/0.925/0.929 gives f about 0.54: Per round f = 0.567/0.587/0.512/0.504; the seed-0 half is the even pairs of the same evaluation (r2_unlearn_verdict.py:221-239), so the comparison is fair.
- [correct] Prompt-bootstrap SD(c) at 64x2: 0.029 / 0.050 / 0.044 (RV A6 RSI_a u_RGI_perp A_ci90 and rel): SD(c) = width x sqrt(rel)/3.29 is right because A6 bootstraps cos and divides by a fixed sqrt(rel).
- [correct] B-Q7 lambda_lin at block 14: 0.7298 [0.1139, 1.3557], SD 0.380, 128 texts: 
- [correct] Resolvable band around G = 1: [0.946, 1.057], [0.938, 1.062], [0.969, 1.032]: 
- [correct] Round-1 damage reference uses the first 8 prompts, all RSI-topic (RL:59-77, 186-196): 
- [correct] Reusable calibration helper at r2_unlearn_qwen.py:223-231: calibrate(dmg_fn, start, rule) with target/lo/hi/max_eval.
- [imprecise] Round-2 A6 RGI nulls use the same uncentred sign-flip construction (r2_rgi_stats.py:537-547): sign_flip_nulls is at 537-543 (544-548 is pool_dirs). It is uncentred, but it flips the 48 model-object contrasts, not per-prompt text differences.
- [imprecise] C0, uncomposed, passing and P = longest run (r2_rgi_stats.py:300-310, 712-716; P2:522-523): Code lines are right; the plan lines are 523-524 (C0 at 523, uncomposed at 524). Pointwise intervals are 95 % (LEVEL 0.95).
- [correct] 0.6B arm P = [], n_P = 0, G30/G31 flags; arm ran (G33 checksum passed): S G33 records only arm_ran; the checksum pass is inferred from the runner, which skips the arm if the pinned sha256 of model.safetensors or tokenizer.json fails (r2_run_session.py:208-210, 1053-1054).
- [correct] 1.7B: C0 true at all 28 blocks, uncomposed at 0-11; FG(CHG-DES) at passing blocks 12-27 is 0.706-0.787: FG - dFG = FG(CHG-DES) because dFG = FG(IMP-DES) - FG(CHG-DES) (r2_rgi_stats.py:246).
- [correct] Block-14 RSI verdict would have read 'undecided (positive control failed)'; RND last rows p 0.238/0.530, A -0.154/-0.055: 
- [correct] sim_null.py: uncentred/centred q99 at rho 1 is 0.393/0.210, 0.645/0.212, 0.877/0.214; median p at s2 0.5, rho 0.4 is 0.0102 vs 0.0005: Re-run (numpy only). The rho = 0.4 column and the median p at rho 0.2 (0.090 vs 0.026) also match.
- [correct] Block 14: 'GO1 must have failed at alpha': Incomplete. The same logic shows GO1 PASSED at 1.5 alpha: a failed retry keeps the block at alpha and lists GO1 (RVpy:28-32; run_shard.py:71). So SEA d_1 at 1.5 alpha had rel 0.868 and p_1 <= 0.01 on u_sea.
- [wrong] Unknown: whether the 1.5-alpha retry ran at block 6: run_shard.py:69-70 always runs the retry after a GO1 failure, and the block ran at alpha with GO1 failed. So the retry ran and failed GO1. Only its values are on Drive.
- [wrong] Directions with rho < sigma: the round-1 null is narrower by about 1/sqrt(1 + s^2), i.e. marginally anti-conservative: It is narrower only relative to the centred null (and at q99 only by 0.95-0.97 in the simulation, not 0.82 at s2 0.5). Under rho = 0 the observed c has SD sigma/sqrt(1 + N s2), narrower than either null, so both are conservative. sim_null.py gives median p at rho = 0 of 0.57-0.93 uncentred and 0.70-0.86 centred.
- [correct] N cancels from the null laws; c ~ rho SNR/sqrt(1+SNR^2); RMS inflation sqrt((1 + rho^2 kappa)/(1 + s^2)); leaked energy share s^2/(1+s^2): Re-derived. The formula is an RMS (Gaussian) approximation and over-predicts the q99 ratio once saturation sets in (sim: 4.10 vs 6.85 at s2 0.5, rho 1).
- [imprecise] Under a centred null q99 would be roughly the intercept: 0.22 / 0.20 / 0.24: By the report's own model the centred q99 is 2.576 sigma = sqrt(a(1+s2)) = 0.23 / 0.22 / 0.26. The intercept is the uncentred null at c = 0. The free-null bar is therefore about 0.22-0.26, not 0.22.
- [correct] |c| needed for p <= 0.01 under the round-1 null: 0.37 / 0.50 / 0.51: sqrt(a/(1-b)) = 0.365 / 0.495 / 0.510.
- [imprecise] q99 rising with |c| is evidence of the leak of the true axis into the null: Consistent with it, but the fit assumes one sigma for all directions. Loop directions aligned with the axis can also have a larger noise share sigma_d. Only R0 (centred vs uncentred nulls for the same d) isolates the leak.
- [imprecise] sea_solve.py reproduces the foreign-axis p with s2 0.15-1.7 (kappa 7-210); a centred null gives p <= 0.004: Re-run: kappa 7-214; centred p 0.0005-0.0042 (block 6 sea at sigma 0.15 gives 0.0042).
- [imprecise] K1-s is exact under H0 because the + and - texts share their Gumbel key, so b_p is symmetric: The shared key couples the pair; it does not make b_p symmetric. Exactness needs the sharp null that the push sign does not change the pair's joint law. For E[b_p] = 0 the test is asymptotically valid with 64 prompts. The known pro:anti asymmetry (sigma 6.47) makes skew plausible. K1-s also conditions on the estimated axis and ignores axis-estimation noise (SD about 0.03 at block 14).
- [imprecise] Cross-fitting S_4 removes the shared-prompt inflation behind cosines above 1 (gegenpruefung data_problems item 3): Item 3 calls shared prompts 'a likely cause, not verified'. The values (S4 1.0436/1.2190/1.1516 are round-2 A6.gen S4; RND_1 1.468 at block 18) are quoted correctly.
- [correct] Under a centred null RSI_a's c_3 = 0.171 would have p < 0.10, so 'unlikely' could not have been met: With sigma about 0.09, two-sided p is about 0.05-0.06. RSI_b's c_3, c_4 (0.219, 0.198) would be lower still.
- [correct] Expected alpha for 0.05 nats along u*: 0.34-0.36 alpha_0 (14), 0.91 (18), 0.75 (6); log-log slope 1.74: Slope ln(0.3243/0.0971)/ln 2 = 1.74 gives 0.341; slope 2 gives 0.359; block 18 slope 2.07 gives 0.911; block 6 slope 1.71 gives 0.750. Basis: + push on 8 RSI-topic references, not the proposed dmg_sym.
- [imprecise] RSI_a d_0 damage 0.1135 at 1.5 alpha is about 0.05 x 1.5^2, so the alpha calibration was sound: The calibration was on u_T; d_0 of RSI_a is u_T_perp (damage at alpha logged in state.pt alpha.damage_u_T_perp, Drive). Consistent, not a check of the calibration.
- [correct] The sea and style directions do 2.3-3 times more damage per unit push than u_T_perp: At 0.15: u_sea 0.2661 (2.34x), u* 0.3243 (2.86x), u_sports 0.2544 (2.24x), all against u_T_perp 0.1135.
- [imprecise] The asymmetric response is shown by sigma 6.47 (C_Q5.symmetry.sigma): 6.47 is the pro:anti read-out ratio on u_T for R1 RSI_a's push d_0, not a damage asymmetry. It still motivates measuring both signs.
- [imprecise] The GO2 mirror still holds because unit(x - a u) is symmetric: Conclusion right, reason wrong: both mirror calls add the identical vector (sign x v), so the rescaled states and texts are identical.
- [correct] SD factor table (64x1 1.17, 64x4 0.90, 128x1 0.83, 128x2 0.71, 256x1 0.59, 256x2 0.50): 
- [correct] MDE_c 0.09/0.16/0.14 at 64x2, 0.064/0.11/0.10 at 128x2 (3.17 SD): These are MDEs of the paired test alone. The proposed rule also needs Q_c >= 1, so its MDE is set by the bar: about 0.25/0.26/0.30.
- [imprecise] RSI_a vs RSI_b differ by 0.037-0.067 in c, SD about 0.035 per loop: 0.0369-0.0664, i.e. 0.037-0.066; the SD (RMS/sqrt 2 = 0.035) is right.
- [correct] SEA rel at block 18 under 64x2 / 128x1 / 128x2: 0.52/0.57, 0.61/0.66, 0.68/0.72: Model-based (constant signal, f from RSI_a at block 14).
- [correct] E = 0.664 with SD about 0.09; MDE about 0.28: 0.077 from within-evaluation bootstrap widths, about 0.10 with loop-to-loop SD; 0.09 is in range.
- [imprecise] A_bar 0.195 is about 2 SD below the 0.25 tier: About 2 SD with the loop-to-loop SD (about 0.025). About 5 SD with the within-evaluation prompt-bootstrap SD (0.011).
- [correct] K SD 0.05; along u_T_perp SD 0.17-0.21; lambda_lin needs about 12 times the texts: TV R6 K sd 0.049-0.053, K_u_T_perp sd 0.172-0.210; (0.380/(0.27/2.487))^2 = 12.2.
- [correct] R0 needs about 320 MB of state.pt: Matches the frozen round-2 plan's copy-in estimate (P2:1425).
- [imprecise] The README null claim is correct both in the code and in its effect: Code: yes. Effect: supported by 3.3 (i)-(ii) but not isolated from direction-specific noise shares; R0 is needed.
- [unverifiable] The 0.8 'uncomposed' cut is the most likely cause of the empty 0.6B arm: A plausible hypothesis; the ranking is a judgement until gate.C0_parts and FG values in geom_stats_qwen3-0.6b.json are read.

## Corrected report
## Round-1 gates: why they failed and what round 1b must change

**Scope.** This is a read-only analysis of `/home/user/DSSE`; no model was run. The helper scripts are in `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/w_gates/`:
- JSON readers: `extract_r1.py`, `q99_fit.py`, `noise.py`, `mde.py`
- One synthetic Gaussian simulation (no model): `sim_null.py`
- One two-parameter null-model solver: `sea_solve.py`

All ran as `taskset -c 3 python3 -I`.

**Abbreviations used below:**
- **P1** = `results/ju_space/plans/abschnitt11_rsi_kennwerte_plan.md`
- **P2** = `results/ju_space/plans/abschnitt11_12_runde2_plan.md`
- **RL** = `analysis/ju_space/rk_loop.py`
- **RVpy** = `analysis/ju_space/rk_verdict.py`
- **RS** = `analysis/ju_space/gpu/run_shard.py` (the round-1 shard runner, which chooses the run scale)
- **S / TV / RV** = `round2_summary.json` / `theta_verdict.json` / `rgi_verdict.json` in `results/ju_space/runde2_ergebnisse/`

### 0. Short answer

**Block 14** (ran at 1.5 alpha = 0.15, suffix '@1.5'; S `gates.G10.blocks.14`):
- GO1 failed at alpha and passed at 1.5 alpha.
  - The runner retries at 1.5 alpha only after a GO1 failure, and moves the block to 1.5 alpha only if the retry passes GO1 (RS:68-71; P1:96, 113).
  - A failed retry would have kept the block at alpha and listed GO1 as failed (RVpy:28-32). The failed list is ['GO3'] only.
  - So SEA d_1 at 1.5 alpha had rel_1 = 0.868 and p_1 ≤ 0.01 on u_sea. loops.json SEA row 0 reuses the pilot's cached evaluation `ev_SEA@1.5_0` (RL:265-269).
- GO3 failed at 1.5 alpha. Damage of SEA d_1 was 0.4055 nats and of RSI_a d_1 0.2018 nats, both above 0.15 (TV `blocks.14.B_Q0.evaluations.{SEA,RSI_a}.1.damage_of_push`).
- PC_PASS is false even though SEA's rel at n = 2-4 is 0.900 / 0.888 / 0.902.
  - The failing part is therefore p > 0.01 or c ≤ 0 on u_sea at some n = 2-4, or S_4 < 0.5.
  - The values are only in Drive.

**Block 18:**
- No pilot gate failed. In particular GO1 passed at alpha: the block ran at alpha with an empty failed list. So SEA d_1 had rel_1 = 0.573 and p_1 ≤ 0.01 on u_sea.
- PC_PASS already fails on reliability: SEA rel_2 = 0.4376 and rel_4 = 0.4877 are below 0.5 (TV `blocks.18.B_Q0.persistence.SEA`). Whether p, c or S_4 also fail is not known locally.
- No correction of the null can fix this. It needs more pairs.

**Block 6:**
- GO1 failed at alpha although SEA rel_1 = 0.6992 ≥ 0.5. So it failed on p_1 > 0.01 on u_sea (the loops reuse the pilot's evaluation `ev_SEA_0`).
- The 1.5-alpha retry ran and also failed GO1: RS:69-70 always runs it after a failure, and the block ran at alpha with GO1 listed. Why it failed is only in Drive.
- PC_PASS is false with rel_2..4 of 0.686 / 0.716 / 0.659, so the failure is in p, c or S_4.

**Null claim (README section 1).**
- The claim is correct in the code: the nulls are built from uncentred per-prompt differences (RL:218-223).
- Its effect is supported by the data (section 3.3):
  - The round-1 null 99th percentile rises with each loop direction's alignment to u_T_perp at all three blocks.
  - Foreign axes with |cos| 0.43-0.61 get p 0.013-0.046 against the round-1 nulls.
- Neither observation separates the leak of the true axis from direction-specific noise shares. The R0 re-analysis does.

Three amendments:
- The cosine-versus-pseudo-axis test, even when centred, is a fixed magnitude bar of about 0.22-0.26 (about 2.6 σ_d) that does not tighten as N grows. Significance must therefore come from a paired sign test on the loop pairs.
- The raw u_sea axis is not specific to the topic. The shared style mode has |cos| 0.61 with u_sea, and a random-start loop converges toward that mode.
- Of the three PC failures, only block 14 (inferred; d_1 passed at 1.5 alpha) and block 6 (GO1) look like the "narrow miss" the README describes. Block 18 fails on rel.

**R0 re-analysis.** It is feasible from the Drive files listed in section 3.6 and needs no model.
- It can reopen only block 6, only post hoc, and only if GO1 passes at alpha under the corrected null (block 6's loops ran at alpha) and PC_PASS passes too.
- Block 14 keeps its GO3 failure.
- Block 18 keeps its rel failure.
- One block cannot enter the delta rule (two are needed), so the delta rule stays 'not decidable'.

**At fixed alpha, damage drifts with direction and round.**
- At block 14 (alpha fixed at 0.15), the damage of RSI_a's pushes rose from 0.1135 to 0.3235 nats over d_0..d_3.
- The sea and style directions do about 2.3-2.9 times more damage per unit push than u_T_perp: u_sea 0.2661 and u* 0.3243, against u_T_perp 0.1135, all at 0.15 (sports: 2.2 times).
- A push normalised per direction and round to 0.05 nats is about 0.34-0.36 alpha_0 along u* at block 14. That is exactly the regime below 0.5 alpha_0 that round 2 left open.

**Minimum detectable effects at 64 prompts x 2 seeds** (one-sided alpha 0.01, power 0.8):
- Paired test, in cosine units: about 0.09 (block 14), 0.16 (block 18), 0.14 (block 6).
- Centred free-null bar: about 0.22-0.26 at any sample size.
- A rule that needs both (p_s ≤ 0.01 and Q_c ≥ 1) is limited by the bar. Its MDE is about 0.25 / 0.26 / 0.30 on u_T_perp, whatever the design.
- Round-1 uncentred null: 0.37 / 0.50 / 0.51 on u_T_perp, and above 0.61 on u_sea at block 14 for directions on the style mode.
- Prompts buy more than seeds: about 54 % of the per-pair noise is within-prompt.

**0.6B arm.** Not one block passed C0 and 'not uncomposed'.
- A plausible cause (not verified) is the 0.8 'uncomposed' cut. In 1.7B, FG(CHG-DES) at the passing blocks is 0.706-0.787, only 0.013-0.094 below the cut.
- Drive file `dsse_runs/r2_q3/rgi/geom_stats_qwen3-0.6b.json`, key `gate.*`, settles it.

### 1. Gate definitions (exact)

| Gate | Definition | Source |
|---|---|---|
| PC_PASS | Applies to the SEA loop at rows n = 2, 3, 4 (directions d_2..d_4, which are loops.json rows 1-3). On u_sea: p_n ≤ 0.01 and c_n > 0, and rel_n ≥ 0.5. Also S_4 = cos(dir_SEA_4, dir_SEA_3)/sqrt(rel_3·rel_4) ≥ 0.5. u_sea is the raw axis (not projected). p is against its own 2,000 sign-flip nulls `null_sea`, which are also unprojected. rel does not depend on the axis. | P1:66; RVpy:40-41 (S4), 44-46 (pc_pass); RL:319 (sea stats of SEA rows); RL:283-296 (c, p, Q, rel, A; rel floor 0.05 at RL:295); RL:223, 230 (null_sea stored unprojected) |
| GO0 | Cross-fitted centred AUC(u_T) ≥ 0.85 and AUC(u_sea) ≥ 0.90 | P1:96, 110-111; RL:219-222, 430 |
| GO1 | Pilot SEA round 1 (row 0 = d_1) on u_sea: rel_1 ≥ 0.5 and p_1 ≤ 0.01. The code has no c > 0 term. On failure, one retry at 1.5 alpha under its own keys. If the retry passes, its gates count and the whole block runs at 1.5 alpha. If both fail, the at-alpha gates count and the block runs at alpha. | P1:96, 107, 112-113, 131-133; RL:430, 435-441; RVpy:24-33; RS:68-71 |
| GO2 | The early-stopped read differs from the full forward by less than 1e-4 (max relative difference). The CRN mirror (−u with sign +1 against +u with sign −1, same keys) is token-identical in at least 3 of 4 pairs. The padded-batch difference is only a diagnostic. | P1:96, 114; RL:412-426 (decision at RL:425) |
| GO3 | Damage of SEA d_1 and of RSI_a d_1 at the pilot's alpha ≤ 0.15 nats, i.e. pilot row 0 `damage_next`. Damage is the teacher-forced KL with a + push over 8 reference texts (RL:186-196). | P1:97; RL:431, 321 |
| Pilot NO-GO | At block 14, any failed GO gate replaces the verdict by 'not decidable (pilot gate … failed; descriptive only; would have read: …)'. At the extra blocks there is no verdict string; a failed gate excludes the block from the delta rule. | P1:112; RVpy:100-103, 158 |
| Delta-rule admission | The block needs PC_PASS and no failed pilot gate | P1:90-91, 131-132; RVpy:158, 164, 193 |
| MANIP (rule, not pilot) | RSI_a and RSI_b row 0 have p ≤ 0.01 and c > 0 on u_T_perp; retry at alpha·√2 | P1:67, 115-116; RVpy:49-53, 65 |
| Damage rule | damage_of_push ≤ 0.15 in every RSI round (rows 0-3) | P1:18-19, 69; RVpy:75 |
| G-blk(b) | R1.summary[b] `verdict.PC_PASS` is true and `verdict.gates.failed` is empty ("second-review rule"). Otherwise block b leaves A6 dynamics, R3, R4 and 'pull and delta', and C-Q5 is labelled. | P2:94, 1334 (G12); r2_report.py:286; r2_theta_verdict.py:372, 633 |
| G-desc | If sum[14] `verdict.gates.failed` is non-empty, every round-2 Qwen verdict at block 14 is labelled 'descriptive (round-1 gate failed)' | P2:97, 1333 (G11); r2_report.py:290; OD-G5 at P2:1551-1552 |
| G-likely / G-attr | G-likely is LIKELY_wh (P2:75-76). G-attr is the attractor label with an RND row at p ≤ 0.01 and abs(A) ≥ 0.25. | P2:95-96, 1335-1336 |
| A6-dynamics extra | G-blk, GA, and damage ≤ 0.15 at d_1..d_3 | P2:555 |

**Gate status per block, deduced from S `gates.G10`/`G12` and the runner logic:**
- Block 14: GO1 failed at alpha and passed at 1.5 alpha; GO3 failed at 1.5 alpha.
- Block 18: all gates passed at alpha.
- Block 6: GO1 failed at alpha and at 1.5 alpha.
- GO0 and GO2 passed at all three blocks.

### 2. Round-1 numbers behind each failure

**Sources:**
- Rel comes from TV `blocks.<b>.B_Q0.persistence.<L>` and equals R1 `loops[L][n-1].rel` (r2_theta.py:216).
- Damage of pushes d_0..d_3 comes from TV `blocks.<b>.B_Q0.evaluations.<L>.<n>.damage_of_push`.
- u_T_perp statistics come from RV `A6.blocks.<b>.loops.<L>.<n>.u_T_perp`, which are the round-1 loops.json rows verbatim (r2_rgi_verdict.py:92).

**SEA loop**

| Block (alpha, suffix) | Measure | d_1 | d_2 | d_3 | d_4 |
|---|---|---|---|---|---|
| 14 (0.15, @1.5) | rel | 0.8677 | 0.8997 | 0.8884 | 0.9018 |
| 14 | c / p / Q / A on **u_T_perp** | −0.148 / 0.201 / 0.595 / −0.159 | −0.167 / 0.202 / 0.621 / −0.176 | −0.180 / 0.179 / 0.644 / −0.191 | −0.125 / 0.320 / 0.465 / −0.132 |
| 18 (0.1085) | rel | 0.5730 | **0.4376** | 0.5628 | **0.4877** |
| 18 | c / p / Q / A on u_T_perp | −0.114 / 0.209 / 0.601 / −0.151 | −0.123 / 0.215 / 0.573 / −0.186 | −0.054 / 0.605 / 0.233 / −0.072 | −0.023 / 0.790 / 0.110 / −0.032 |
| 6 (0.1344) | rel | **0.6992** | 0.6860 | 0.7164 | 0.6586 |
| 6 | c / p / Q / A on u_T_perp | −0.133 / 0.298 / 0.482 / −0.159 | −0.120 / 0.324 / 0.440 / −0.145 | −0.169 / 0.171 / 0.639 / −0.200 | −0.086 / 0.451 / 0.335 / −0.105 |

**Damage per push at block 14** (nats, fixed alpha 0.15, pushes d_0..d_3):

| Loop | d_0 | d_1 | d_2 | d_3 |
|---|---|---|---|---|
| RSI_a | 0.1135 | 0.2018 | 0.2871 | 0.3235 |
| RSI_b | 0.1135 | 0.1887 | 0.2454 | 0.2861 |
| SEA | 0.2661 | 0.4055 | 0.4177 | 0.4492 |
| SPO | 0.2544 | 0.4040 | 0.4983 | 0.5211 |
| RND_1 | 0.0626 | 0.1155 | 0.2522 | 0.3859 |
| RND_2 | 0.0528 | 0.1129 | 0.1345 | 0.1808 |

- At blocks 18 and 6 every push stays at or below 0.0916 nats.
- Damage along u* (TV `blocks.14.cells.*.damage`): 0.0971 at 0.5 alpha_0, 0.3243 at alpha_0, 0.9133 at 2 alpha_0, 2.374 at 4 alpha_0 (alpha_0 = 0.15).
- RSI_a's d_0 (= u_T_perp) damage of 0.1135 at 1.5 alpha is close to 0.05 × 1.5² = 0.1125.
  - The calibration targeted u_T, not u_T_perp. u_T_perp's damage at alpha is in `state.pt ['alpha']['damage_u_T_perp']`, on Drive.
  - So this only shows that the start's damage was near the calibrated scale. The growth comes from the rotating loop direction.

**Indirect evidence on the sea axis** (the sea-axis loop values for d_2..d_4 are not local):
- Block 14:
  - GO1 passed at 1.5 alpha, so SEA d_1 had p ≤ 0.01 on u_sea (section 0). d_1 still carries the self-trace of the u_sea start.
  - cos(d_1^SEA, u*) = −0.930 (TV `blocks.14.B_Q0.image_cos_d1_u_star.SEA`).
  - cos(d_4^SEA, d_4^RSI_a) = −0.962 (TV `mode_set.cos_d4.SEA`).
  - u* has cos −0.610 with u_sea at p 0.013 against R1 `null_sea` (TV `blocks.14.R4.default.axes.sea`).
  - So the SEA loop sits on the shared mode, and its sea-axis p at d_2..d_4 was plausibly near 0.01. This is an inference.
- Block 6: cos(d_1^SEA, u*) = −0.834; u*_6 has cos −0.429 with u_sea at p 0.046.
- Block 18:
  - GO1 passed at alpha, so SEA d_1 had p ≤ 0.01 on u_sea.
  - u*_18 has cos 0.151 with u_sea at p 0.284; cos(d_4^SEA, d_4^RSI_a) = 0.114.

**Context at block 14 (RSI loops, from the same files):**
- RSI_a: c 0.391 / 0.243 / 0.171 / 0.161, p 0.0005 / 0.0575 / 0.182 / 0.242.
- RSI_b: c 0.458 / 0.283 / 0.219 / 0.198.
- MANIP holds.
- A_bar = 0.195 (mean of |A_3| and |A_4| over both loops) and E = 0.664, both computed from RV A6 A values.
- S `gates.G13.V_RSI` = "not decidable (pilot gate GO3 failed; descriptive only; would have read: undecided (positive control failed))".
- G14: RND_1 p 0.238, A −0.154; RND_2 p 0.530, A −0.055.

**Not available locally** (Drive, `dsse_runs/`; shards from S `r1_finder` and `gates.G7.ran`: b14 on gpu1, b18 on gpu2, b6 on gpu3):
- SEA c, p, Q, A on u_sea per round: `gpu*/rk/b<b>/loops.json ['SEA'][0..3]['sea']`.
- S_4: `gpu*/summary_b<b>.json ['verdict']['SEA_S4']`.
- Pilot SEA_1 values at alpha and 1.5 alpha (c_1, p_1, rel_1):
  - Files: `pilot.json`, `pilot@1.5.json`.
  - Summary `['verdict']['gates']['SEA_1']` always holds the at-alpha values (RVpy:33).
  - The retry status itself follows from RS:68-71 and the failed lists (section 1).
- GO0 AUCs and paired_win: `pilot.json ['axes']`.
- GO2 numbers: `state.pt ['checks']`.
- Damage of d_4: loops.json row 3 `damage_next`.
- At-alpha pilot damages at block 14: `state.pt ['dmg_SEA_1']`, `['dmg_RSI_a_1']`.
  - My estimate is about 0.18-0.20 nats for SEA d_1 (alpha² scaling, or the u* slope 1.74), so GO3 probably fails at alpha too.
  - This assumes d_1 at alpha points roughly where d_1 at 1.5 alpha does.

### 3. The sign-flip null (README section 1)

#### 3.1 Code

- RL:218: `diff = Xp - Xa` (no centring) and `u = diff.mean(0)`.
- RL:223: `E` is 2000×N Rademacher with `torch.Generator` seed 11 + index (rsi 11, sports 12, tone 13, sea 14), and `nulls[n] = (E @ diff)/N`.
- RL:226: only the rsi nulls are projected off span(sports, tone, sea, NEG).
- RL:286-287: `nc = |null·d|/(|null||d|)`, `p = (1 + #{nc ≥ |c|})/2001`, `Q = |c|/q99(nc)`.

So c_k = mean_p e_kp D_p = m_k·u + n_k. Here m_k = mean_p e_kp has mean 0 and SD 1/√N (0.125 at N = 64): this is the random multiple of the true axis.

#### 3.2 Derivation

**Setup.** D_p = u + η_p, with Cov η = Σ.
- ū = u + n̄ is the observed axis.
- d is a unit loop direction (treated as fixed), with ρ = cos(d, u).
- σ² = dᵀΣd / trΣ is d's noise share; 1/σ² = D_d is the effective dimension.
- s² = |u|²/trΣ is the per-prompt axis energy over the noise energy.
- κ_d = s²/σ².
- z_k = √N·m_k and w_k are approximately N(0, 1).

**Laws:**
- Round 1: cos(d, c_k) ≈ (z_k ρ s + w_k σ)/sqrt(z_k² s² + 1).
- Centred, c_k^c = mean_p e_kp (D_p − ū): cos ≈ w_k σ.
- Observed: c ≈ (ρ s √N + w σ)/sqrt(N s² + 1).

**Role of N.** N cancels from both null laws. It enters through the axis SNR, SNR_ax² = N s² = |u|²/(trΣ/N), which sets c ≈ ρ·SNR_ax/sqrt(1 + SNR_ax²). With s² = SNR_ax²/N:
- RMS inflation of the null |cos| (Gaussian regime) = sqrt((1 + ρ²κ_d)/(1 + s²)) = sqrt((1 + ρ² D_d SNR_ax²/N)/(1 + SNR_ax²/N)). It over-predicts once saturation sets in.
- The share of each null axis's energy that is the true axis = s²/(1 + s²) = SNR_ax²/(N + SNR_ax²).

**Strong axes.** The leaked term saturates: |cos(d, c_k)| tends to ρ for |z_k| much larger than 1/s. An aligned direction's p is then set by how many null axes reach |cos| ≈ ρ ≈ c, not by how far c is from noise.

**Directions with ρ < σ.**
- The round-1 null is narrower than the centred null. The RMS factor is about 1/sqrt(1 + s²); at the 99th percentile it is only 0.95-0.97 in the simulation below. So for these directions it sets a slightly lower bar than a centred null.
- It is not anti-conservative in terms of size. Under ρ = 0 the observed c has SD σ/sqrt(1 + N s²), narrower than either null.
- The simulation gives median p at ρ = 0 of 0.57-0.93 (uncentred) and 0.70-0.86 (centred).
- Both cosine nulls are conservative as significance tests. The centred one acts as a magnitude bar (section 3.4 b).

**Synthetic check** (N 64, D 2048, σ 0.085; `sim_null.py`). Uncentred / centred q99:

| s² | ρ = 0.4 (c 0.30-0.39) | ρ = 1 |
|---|---|---|
| 0.02 | 0.243 / 0.214 | 0.393 / 0.210 |
| 0.1 | 0.326 / 0.207 | 0.645 / 0.212 |
| 0.5 | 0.392 / 0.211 | 0.877 / 0.214 |

- Median p at s² 0.5, ρ 0.4: 0.0102 (uncentred) against 0.0005 (centred).
- Median p at s² 0.5, ρ 0.2: 0.090 against 0.026.
- The Gaussian formula is accurate for small ρ²κ and over-predicts once saturation sets in (q99 ratio 4.10 simulated against 6.85 predicted at s² 0.5, ρ 1).

#### 3.3 Evidence in round-1 data

**(i) q99 rises with alignment on u_T_perp.** q99 = |c|/Q, from RV `A6...u_T_perp.{c,Q}`. The fit is q99² = a + b·c²:

| Block | Spearman (|c|, q99) | a | b (SE) | R² | Implied s² | Implied σ |
|---|---|---|---|---|---|---|
| 14 | 0.86 | 0.049 | 0.63 (0.06) | 0.83 | 0.11 | 0.09 |
| 18 | 0.66 | 0.041 | 0.83 (0.11) | 0.84 | 0.14 | 0.08 |
| 6 | 0.74 | 0.059 | 0.77 (0.14) | 0.75 | 0.13 | 0.10 |

- Under a centred null, q99 would not depend on alignment. It would be about 2.6σ_d, which with the fitted σ is sqrt(a(1 + s²)) = 0.23 / 0.22 / 0.26. The intercepts 0.22 / 0.20 / 0.24 are the uncentred null at c = 0, narrower by 1/sqrt(1 + s²).
- Under round 1's null, the |c| needed for p ≤ 0.01 (where c = q99(c)) is 0.37 / 0.50 / 0.51.
- This matches the observed RSI_a d_1: c 0.391 gives p 0.0005 at block 14, but c 0.314 gives p 0.041 at block 18 and c 0.323 gives p 0.053 at block 6.
- Caveat: the fit assumes one σ for all directions. q99 can also rise with |c| if directions aligned with the axis lie in higher-variance directions of the axis-text differences (a larger σ_d). R0 separates the two by computing both nulls for the same d.

**(ii) Foreign axes at block 14 versus R1 nulls** (TV `R4.default.axes`):

| Axis (block) | |c| of u* | p |
|---|---|---|
| sea (14) | 0.610 | 0.013 |
| sports (14) | 0.457 | 0.022 |
| tone (14) | 0.481 | 0.040 |
| sea (6) | 0.429 | 0.046 |

- The null model (`sea_solve.py`) reproduces these with s² of about 0.15-1.7 (κ 7-214) for σ 0.09-0.15.
- A centred null with the same σ would give p of 0.0005-0.004 at the same cosines.
- This is consistent with a strong sea contrast: cross-fitted mean-difference AUC is 0.984 for sea against 0.910 for RSI at block 14. These are per-token block-14 states of the round-1 texts (S `parts.C.key_numbers.C_Q1.concepts.{sea,RSI}.md_auc`); sports and tone are not compared there.
- Caveat: a very low effective noise dimension along u* (σ about 0.20-0.25) would also explain (ii). Only R0 can separate the two explanations.

#### 3.4 Assessment of the README claim

The claim is correct in the code, and its effect is supported (not yet isolated), with three amendments:
- (a) "Positive control narrowly missed" fits block 14 (inferred; d_1 passed at 1.5 alpha) and block 6's GO1. It does not fit block 18, which fails on rel.
- (b) A centred cosine null answers "is d more aligned than about 99 % of sampling-noise axes", a bar of about 2.6σ_d that does not change with N. It is not a significance test whose power grows with N.
- (c) PC on raw u_sea is confounded with the style mode: |cos(u*, u_sea)| = 0.61.
  - RND_1 converges toward u* with debiased |cos| 0.751 (TV `blocks.14.R4.random_start_debiased_cos.RND_1`).
  - RND_1 is one of the five loops that define u*_14, so this value includes self-overlap. Against RSI_a's d_4 alone its debiased cos is −0.559 (TV `blocks.14.B_Q0.mode_set.debiased_cos.RND_1`).
  - Under a corrected null, PC could pass because of style, not topic.

#### 3.5 Repair, precisely

**K1-c (magnitude bar).**
- Pseudo-axes: c_k^c = (E_k @ (D − D̄))/N.
- Use round 1's E (same seeds, K = 2,000, N = kept prompts) and project exactly like the axis (RL:224-226 for u_T_perp).
- p_c = (1 + #{|cos(d, c_k^c)| ≥ |c|})/(K + 1); Q_c = |c|/q99_c.

**K1-s (significance on the loop pairs).**
- For evaluation n: a_j = û·delta_j over the ok pairs; b_p = the mean over prompt p's pairs (both seeds together).
- T = mean_p b_p. One-sided sign-flip test: p_s = (1 + #{k: mean_p ε_kp b_p ≥ T})/(K + 1), with K = 10,000 and a fixed seed; prompts are the units.
- Report Wilcoxon and t next to it.
- Validity:
  - The sign flip is exact only under the sharp null that a prompt's + and − texts are exchangeable, i.e. the push sign does not change the pair's joint law.
  - The shared Gumbel key (RL:272) couples the pair but does not by itself make b_p symmetric. The + and − responses are known to be unequal (pro:anti ratio 6.47 on u_T), so b_p can be skewed.
  - Under the weaker null E[b_p] = 0 the test is asymptotically valid with 64 independent prompts. Check its size in R0.
- K1-s conditions on the estimated axis û, which shares the 64 prompts with the loops.
  - Alignment with the axis's own estimation noise has SD about σ_d/sqrt(1 + N s²), about 0.03 at block 14, comparable to SD(c).
  - Q_c ≥ 1 guards against this. A joint prompt bootstrap of axis texts and loop pairs would propagate it.

**Rules.**
- 'p ≤ 0.01 with c > 0' becomes: p_s ≤ 0.01 (one-sided) **and** Q_c ≥ 1. This applies in PC_PASS, GO1, MANIP and LIKELY.
- The MDE of this combined rule is set by the bar (section 5). Pre-register it, not only the paired-test MDE.
- The 'unlikely' rule (p > 0.10) becomes an equivalence bound: the 90 % upper bound of |A_n| < 0.25 at n = 3, 4. With SD(A) of about 0.02-0.05 this needs |A| below about 0.17-0.20.
- Under a centred null, RSI_a's c_3 = 0.171 would have p of about 0.05-0.06, below 0.10. So the round-1 'unlikely' rule could not have been met.

**PC axis.**
- Decide PC on u_sea_perp (sea minus its projection on span(u_T, sports, tone, NEG)), plus a lexical sea-word difference (+ minus −) per prompt.
- span(u_T, sports, tone, NEG) does not contain u*, so u_sea_perp can still carry the style mode. Report cos(u*_b, u_sea_perp) in R0.
- Freeze the sea-word list and the SEA start (raw u_sea as in round 1, or u_sea_perp) before the run.
- Require SEA − max(RND) > 0 with a bootstrap interval (only RND_1 exists at blocks 18 and 6).
- Report raw u_sea as well.

**S_4, R_4, debiased cosines.**
- Cross-fit them: over the 20 K2 halvings, (cos(d_4^A, d_3^B) + cos(d_4^B, d_3^A))/2 divided by sqrt(r_3·r_4), with r_n the half-sample split cosines.
- This removes the shared-prompt inflation behind the values above 1, if shared prompts are the cause. `auswertung_gegenpruefung.json` `data_problems` item 3 calls this "a likely cause, not verified".
- Report cross-fitted and uncorrected values side by side.
- d_4 is pushed along the full-sample d_3, so the loop's genuine persistence of d_3's noise remains.

**Validation before freeze.**
- Size: placebo axes nearly orthogonal to u* (u_neg: cos(u*_14, u_neg) = −0.017) and noise-matched random axes. RND loops count as a size check only on axes the style mode does not load on (not sea, sports or tone at block 14).
- Power: planted share of u_T_perp in the seed-0 half of RSI_a.

#### 3.6 R0 re-analysis (before any new run; CPU only, no model)

**Inputs, per block:** gpu1/rk/b14, gpu2/rk/b18, gpu3/rk/b6.
- `state.pt`:
  - `['axis_sea'|'axis_rsi'|'axis_sports'|'axis_tone']['Xp','Xa']` (64×D)
  - `['axes']['sea','sports','tone','u_neg','u_T','u_T_perp','null_sea','null_u_T_perp','fold','kept']`
  - `['ev_<L><SUF>_<n>']['delta','ok','prompt_of','xp','xm','conts']` for n = 0..3 and L in SEA, RSI_a, RSI_b, SPO, RND_1, RND_2 (SUF '@1.5' at block 14, '' elsewhere)
  - `['dir_<L><SUF>_<n>']` for n = 1..4
  - At block 14 also the at-alpha pilot entries `['ev_SEA_0','ev_RSI_a_0','dir_SEA_1','dmg_SEA_1','dmg_RSI_a_1']`; at block 6 the 1.5-alpha retry entries `['ev_SEA@1.5_0','ev_RSI_a@1.5_0']`
- `loops.json` (rows, with the `'sea'` sub-dict).
- `pilot.json` and `pilot@1.5.json` (`SEA_1[0]['sea']`, `gates`, `axes.auc/paired_win/kept`).
- `gpu*/summary_b<b>.json ['verdict']` (`PC_PASS`, `SEA_S4`, `gates.source/SEA_1/failed`).
- About 320 MB of state.pt in all (the frozen round-2 plan's copy-in estimate, P2:1425).

**Tools.** Read these with `r2_rk.load_r1_state`, `r2_rk.stats_on_axis` and `rk_verdict.halvings` (no side effects).
- `rk_loop.py` cannot be imported without loading the model (RL:33-49), so re-implement the centring.
- E is reproducible from `torch.Generator` seeds 11-14 with the kept counts in `['axes']['kept']`. Check `E @ diff / N` against the stored nulls.
- The lexical count needs the tokenizer to decode the stored token ids.

**Steps.**
1. Reproduce the loops.json sea and u_T_perp c, p, Q exactly.
2. Compute K1-c with the same E.
3. Compute K1-s.
4. Compute the cross-fitted S_4.
5. Evaluate PC_PASS_R0 and GO1_R0.
   - GO1 at alpha at blocks 14 and 6.
   - At block 14 the 4-round loops exist only at 1.5 alpha, so PC_PASS_R0 there is at 1.5 alpha.
6. Run size checks on placebo axes and RND loops (as restricted in section 3.5).
7. Label the result 'R0, post hoc'.

**Expected outcome.**
- Block 6 is the only block that can change status.
  - Its GO1 failure at alpha is p-driven; its PC failure is in p, c or S_4, and which one is unknown.
  - It reopens only if GO1_R0 passes at alpha and PC_PASS_R0 passes.
- Block 14 stays blocked by GO3; block 18 stays blocked by rel.
- The delta rule stays 'not decidable' (at most one block).

### 4. Damage-normalised push and norm-preserving control

**Why.** The round-1 alpha was calibrated on u_T to 0.05 nats (RL:238-247) and never recalibrated (P1:17-19).
- Its reference damage uses only the first 8 prompts, which are all RSI-topic (RL:189).
- The loops then rotate into the style mode, which does more damage per unit push (section 2 tables).

**Damage-normalised push** (one alpha per loop L and round n):
- a_{L,n} = `calibrate(lambda a: dmg_sym(d_n, a, mode), start=a_{L,n−1}, rule={target 0.05, lo 0.045, hi 0.055, max_eval 6})`. This reuses r2_unlearn_qwen.py:223-231.
- dmg_sym is the mean of the + and − push damage.
  - The + and − pushes are known to act unequally on the text: pro:anti ratio 6.47 on u_T for RSI_a's d_0 (S `parts.C.key_numbers.C_Q5.symmetry.sigma`).
  - Damage asymmetry itself has not been measured, so log both signs.
- References: 8 RSI-topic + 8 scene texts from `ST['ref']`. One dmg_sym evaluation is 16 unsteered forwards (cacheable) + 32 steered forwards, about 1 s at the round-2 rate of about 20 ms per forward.
- Log a_n, the achieved damage of each sign, the tried values, in_band, capped share, and the damage of d_{n+1} at a_n.
- GO3 becomes 'every push in band'.
- Expected alpha for 0.05 nats along u*: about 0.34-0.36 alpha_0 at block 14 (log-log slope 1.74 from cells 0.5→1 alpha_0; slope 2 gives 0.36), 0.91 alpha_0 at block 18, and 0.75 alpha_0 at block 6.
  - These come from the + push on the 8 RSI-topic references.
  - dmg_sym on 16 references will shift them.

**steer_hook (RL:102-110).** Unchanged for the normalised push: alpha is set per call (RL:134, 109). For the control, add `STEER['mode']` (RL:100; reset in `steer_off` at RL:114):

```python
x2 = x + add
if STEER.get('mode') == 'rot' and STEER['v'] is not None:
    x2 = x2 * (raw / x2.norm(dim=-1, keepdim=True).clamp_min(1e-12))   # raw from RL:106
```

- This rotates h_t toward s·d and keeps |h_t| fixed; unmasked positions are unchanged.
- It changes both the along-d and the radial part of the push, so log the achieved along-d displacement per unit alpha.
- The K5 leaf path `'V'` stays additive.
- The 'rot' arm is calibrated with the same mode at the same 0.05 nats.
- The GO2 mirror still holds: both mirror calls add the identical vector (−u with sign +1 equals +u with sign −1), so the rescaled states and the texts are identical.

**Other code changes.**
- Make them in a new module, not in rk_loop.py.
  - Round 2 imports rk_loop.py unchanged and compares its sha256 (R1 code freeze, P2:84; G3 at P2:1325). The round-1 arm should run on the byte-identical file.
  - steer_hook is registered at import (RL:113), so the 'rot' mode needs its own registered hook, not a reassigned `rk.steer_hook`.
- `damage()` (RL:186-196) gains `sign` and `mode` arguments and stratified references.
- `run_loop` (RL:309-323) and `evaluate` (RL:265-281) take a per-round alpha; the cache asserts at RL:262 and RL:268 compare against a_{L,n}.
- Keys go in a new 'r1b' namespace.

**What stays comparable.**
- Keep a round-1 arm: alpha_b·scale from R1 `loops[b]['_meta']['alpha']` (0.15 / 0.1085 / 0.1344; S `gates.G10`), additive push, 64×2 and 64×1 pairs, 4 rounds, and the round-1 axes, halvings (HALV seed 3), cap and starts. Report both the old and the new p.
- Across all arms: cosines and A on the frozen round-1 axes, rel, p_c and p_s, the cosine with round-2 u*_b, and gains per unit push. Compare the gains with TV R1 points and B-Q7 `lambda_lin` (0.7298 [0.1139, 1.3557] at block 14).
- Not comparable: |d_n|, Ue (text units), and the damage-gate status.
- Texts will not be token-identical on new hardware (P2 section 2.3), so compare distributions.
- Cost at the frozen round-2 rates (about 1 s per 32-text generate call and 0.25-0.45 s per read batch, P2 section 7): a 4-round loop at 128 pairs takes under a minute. The arms are cheap.

### 5. Minimum detectable effects

**Within-prompt share of per-pair noise.** f = 0.54 (per round 0.57 / 0.59 / 0.51 / 0.50). It comes from RSI_a at block 14: the seed-0 half's rel (S `parts.C.key_numbers.C_Q4b.reference.<n>.rel`: 0.640 / 0.834 / 0.901 / 0.907) against both seeds' rel (0.713 / 0.877 / 0.925 / 0.929).

**SD factor relative to 64×2:**

| Design | SD factor |
|---|---|
| 64×1 | 1.17 |
| 64×4 | 0.90 |
| 128×1 | 0.83 |
| 128×2 | 0.71 |
| 256×1 | 0.59 |
| 256×2 | 0.50 |

Prompts buy more than seeds: 128×1 costs the same as 64×2 and is 17 % tighter.

**SD(c) along a fixed axis.** From RV `A6...u_RGI_perp.A_ci90` widths × sqrt(rel)/3.29, at 64×2 (RSI_a):

| Block | SD(c) | MDE_c = 3.17 SD (one-sided 0.01, power 0.8) | MDE_c at 128×2 |
|---|---|---|---|
| 14 | 0.029 | 0.09 | 0.064 |
| 18 | 0.050 | 0.16 | 0.11 |
| 6 | 0.044 | 0.14 | 0.10 |

- For A, divide by sqrt(rel).
- Two independent 128-pair loops (RSI_a against RSI_b) differ by 0.037-0.066 in c, i.e. an SD of about 0.035 per loop.

**Free-null bar.** About 0.22-0.26 centred (2.6σ_d) and independent of N, against the round-1 0.37-0.51 (more than 0.61 for sea at block 14, for directions on the style mode).

**MDE of the combined rule** (p_s ≤ 0.01 and Q_c ≥ 1). Bar + 0.84 SD(c) gives about 0.25 / 0.26 / 0.30 at 64×2 on u_T_perp. More prompts lower only the paired-test part.

**rel at block 18 (SEA)** under different designs (model with constant signal and f = 0.54):

| Design | rel_2 | rel_4 |
|---|---|---|
| Round 1 (64 pairs) | 0.438 | 0.488 |
| 64×2 | 0.52 | 0.57 |
| 128×1 | 0.61 | 0.66 |
| 128×2 | 0.68 | 0.72 |

**Other quantities:**
- **E.** E = 0.664 with SD about 0.08-0.10, so the 0.7 bar is not resolvable; MDE about 0.28.
- **A_bar tiers.** A_bar = 0.195 against the 0.25 tier at block 14.
  - About 2 SD apart with the loop-to-loop SD (about 0.025, from RSI_a against RSI_b).
  - About 5 SD with the within-evaluation prompt-bootstrap SD (about 0.011).
- **Gains G.** Resolvable to ±0.03-0.06 around 1 (TV `blocks.<b>.power_rerun.resolvable_band`).
- **K (carry).** SD 0.05; along u_T_perp, SD 0.17-0.21.
- **lambda_lin (B-Q7, 128 texts).** SD 0.38 at block 14. Separating 0.73 from 1 (one-sided 0.05, power 0.8) needs about 12 times the texts.

### 6. Empty Qwen3-0.6B arm

**Code.**
- C0 at block b requires, for both IMP-DES and CHG-DES, a pointwise 95 % lower bound of P_cell > 0 and q = P_cell/P_rel ≥ 0.2.
- 'Uncomposed' means FG(CHG-DES) > 0.8 and FG(WOR-DES) > 0.8 (point estimates).
- Passing means C0 and not uncomposed (r2_rgi_stats.py:712-716; P2:523-524).
- `passing_set` returns the longest run of any length (r2_rgi_stats.py:300-310). So P = [] (RV `A7.arm_0p6B.n_P` 0) means that no single block passed.

**Facts that limit the explanations:**
- The 1.7B arm has C0 true at all 28 blocks and 'uncomposed' true at blocks 0-11 (RV `A7.per_block.C0/uncomposed`).
- The weights are the right ones: `arm_ran` is true (S `gates.G33`). The runner runs the arm only if the pinned sha256 of the 0.6B files checks out (r2_run_session.py:208-210, 1053-1054).
- Both models go through the same float32 code path with finite-state asserts (r2_rgi_geom.py:188-213, 295).
- G29 stimulus asserts run per model (r2_rgi_geom.py:266-272).

**Causes, in order of plausibility (a judgement, not verified):**
1. **Threshold transfer or a real model difference.**
   - 1.7B's FG(CHG-DES) at its passing blocks is 0.706-0.787 (RV `A7.per_block.blocks.<b>.numbers.FG − dFG`).
   - In 0.6B, neutral verb contrasts that are slightly more object-general (perhaps from stronger anisotropy in a 1024-wide raw metric) would be 'uncomposed' everywhere. This needs FG(WOR-DES) > 0.8 as well.
2. **C0 fails.** Weaker or noisier contrasts at the '.' token, with q < 0.2 or the P_cell lower bound ≤ 0.
3. **NaN in ratios.** `_ratio` (r2_rgi_stats.py:218-219) gives NaN for a zero denominator, and NaN comparisons are false.
4. **Stimuli or wrong weights.** Unlikely, given G29 and G33.

**What settles it** (Drive: `dsse_runs/r2_q3/`):
- `rgi/geom_stats_qwen3-0.6b.json`:
  - `gate.C0`, `gate.C0_parts` (`P_cell_lo>0|IMP-DES`, `q>=0.2|IMP-DES`, `P_cell_lo>0|CHG-DES`, `q>=0.2|CHG-DES`), `gate.uncomposed`, `gate.passing`, `gate.runs`
  - `per_block.pooled['FG|CHG-DES','FG|WOR-DES','P_cell|IMP-DES','P_cell|CHG-DES','q|IMP-DES','q|CHG-DES']` (`point`, `lo`, `hi`, `n_nan_draws`)
  - `point_only['dot/white/pooled','obj/raw/pooled']`, `lexical`, `anisotropy.dot`, `flags`
- `rgi/geom_meta_qwen3-0.6b.json` (model `hidden_size`, `tie_word_embeddings`; `stages.a1` lengths and g29; `flags`).
- `rgi/geom_draws_qwen3-0.6b.npz`, if written.
- `logs/geom_06.log` and `logs/stats_06.log` (r2_run_session.py:97-99, 1096).
- The same keys in `geom_stats_qwen3-1.7b.json` for comparison.

The local A1 states (`local/geom_qwen3-0.6b.npz`) were not synced (the Drive sync skips `local/`, r2_run_session.py:376), so recomputing them needs a model rerun.

### 7. What round 1b must change for loop verdicts to be possible

1. Run R0 first, and validate the corrected K1-c and K1-s nulls (size and power).
2. Decide PC topic-specifically: u_sea_perp (after checking cos(u*_b, u_sea_perp)), plus the frozen lexical count, plus a SEA > RND contrast. Cross-fit S_4 and R_4. PC_PASS is needed at two or more blocks.
3. Use the damage-normalised push as the primary arm, with the norm-preserving 'rot' control and the unchanged round-1 arm for attribution. Implement the changes in a new module and keep rk_loop.py byte-identical.
4. Use 128 or more prompts (one seed for the controls, two for RSI); write and freeze the 64 new prompts first. Add seeds or prompts at block 18 so that rel ≥ 0.5 is likely.
5. Pre-register the MDE of the rule actually used. With Q_c ≥ 1 that MDE is bar-limited, about 0.25-0.30 in c, beside the paired-test MDE.
6. Use magnitude-based 'unlikely' and E rules, with their own MDE and verdict strings.
7. Read the 0.6B `gate.*` keys before any 0.6B loop.

## proposals (from the research stage; check against the verifier's notes above)
```json
[
 {
  "name": "R0 re-analysis of round-1 raw data",
  "what": "On gpu1/rk/b14, gpu2/rk/b18 and gpu3/rk/b6: reproduce loops.json c/p/Q; recompute PC_PASS and GO1 (at alpha) with centred pseudo-axes (same E seeds 11-14, K 2,000) and with a one-sided paired sign test on the loop pairs (b_p = mean of u_hat . delta_j per prompt, 10,000 flips); compute a cross-fitted S_4; check size on RND loops and placebo axes; label the result 'post hoc'.",
  "why": "Shows directly whether the null caused the PC and GO1 failures (expected to matter at block 6; block 14 remains blocked by GO3, block 18 by rel), and calibrates the 1b thresholds without a GPU run.",
  "cost": "CPU only, minutes; needs about 320 MB of state.pt plus the loops, pilot and summary JSONs copied from Drive",
  "reuse": "rk_loop.stats/axes code paths; r2_rk.stats_on_axis and prompt_bootstrap; rk_verdict.halvings"
 },
 {
  "name": "Corrected K1: magnitude bar plus significance",
  "what": "K1-c = |c| against centred pseudo-axes (Q_c >= 1) as a fixed magnitude bar; K1-s = paired sign-flip test over prompts on the loop pairs (one-sided for c > 0). Replace 'p <= 0.01 with c > 0' by 'p_s <= 0.01 and Q_c >= 1' in PC_PASS, GO1, MANIP and LIKELY, and the 'unlikely' rule by an equivalence bound (90 % upper bound of |A| < 0.25).",
  "why": "The uncentred null inflates the null for aligned directions (round-1 bar 0.37-0.51 on u_T_perp, above 0.61 on sea); a centred cosine null is an N-independent bar of about 0.22 and cannot give significance.",
  "cost": "No extra model time",
  "reuse": "rk_loop.py:223-226 construction with centring; r2_rk.prompt_bootstrap"
 },
 {
  "name": "Topic-specific positive control",
  "what": "Decide PC on u_sea_perp (sea minus its projection on span(u_T, sports, tone, NEG)) plus a lexical sea-word difference (+ minus -) per prompt; require SEA minus max over RND > 0 with a bootstrap interval; report raw u_sea next to it.",
  "why": "On raw u_sea the shared style mode has |cos| 0.61 and a random loop converges onto it (debiased cos 0.75), so a corrected null could pass PC because of style rather than topic.",
  "cost": "Analysis only; the axes are reused",
  "reuse": "rk_loop.axes perp() construction"
 },
 {
  "name": "Cross-fitted reliability corrections",
  "what": "S_4, R_4 and debiased cosines from disjoint prompt halves (cos(d_4^A, d_3^B) averaged over the 20 K2 halvings, divided by sqrt(r_3 r_4)).",
  "why": "Shared prompts inflate disattenuated cosines above 1 (S4 up to 1.219, RND_1 1.468), which biases gates toward passing.",
  "cost": "Analysis only",
  "reuse": "rk_loop.HALV, rk_verdict.halvings"
 },
 {
  "name": "Damage-normalised push arm",
  "what": "Per loop and round, alpha_{L,n} = calibrate(symmetric damage of +/- d_n = 0.05 nats, band 0.045-0.055, at most 6 evaluations, start at the previous round's alpha) on 8 RSI-topic + 8 scene references; log alpha, achieved damage and gains per unit push; GO3 becomes 'every push in band'.",
  "why": "At fixed alpha the block-14 damage grew from 0.11 to 0.32 nats (RSI_a) and to 0.45 (SEA) as loops rotated into the style mode; 0.05 nats is about 0.34-0.36 alpha_0 along u* at block 14, the untested regime below 0.5 alpha_0.",
  "cost": "About 4-6 extra damage evaluations per loop round (8-16 teacher-forced forwards each)",
  "reuse": "r2_unlearn_qwen.calibrate (r2_unlearn_qwen.py:223-231); r2_rk.damage_T"
 },
 {
  "name": "Norm-preserving control push",
  "what": "STEER['mode']='rot': after x + add, rescale to the original token norm (raw from rk_loop.py:106) at masked positions; calibrated to the same 0.05 nats; the K5 leaf path stays additive.",
  "why": "Separates direction effects from norm inflation of the additive push, a candidate driver of damage and of the style attractor.",
  "cost": "One extra arm per block",
  "reuse": "rk_loop.steer_hook with a 3-line change"
 },
 {
  "name": "Unchanged round-1 arm for attribution",
  "what": "Fixed alpha_b x scale from R1 loops['_meta']['alpha'] (0.15 / 0.1085 / 0.1344), additive push, 64x2 / 64x1 pairs, 4 rounds, round-1 axes, halvings, cap and starts; both old and new p reported.",
  "why": "Isolates the effect of each change (null, push normalisation, rotation) on the same quantities.",
  "cost": "Same as a round-1 block",
  "reuse": "rk_loop.run_loop unchanged with a new key namespace"
 },
 {
  "name": "Sample size and pre-registered MDEs",
  "what": "Prefer 128 prompts x 1 seed (same cost as 64x2, SD x0.83) or 128x2 (SD x0.71) over extra seeds (64x4: x0.90); at block 18 SEA needs 128 or more pairs, ideally new prompts, for rel >= 0.5; K = 10,000 flips; pre-register MDE_c of about 0.09/0.16/0.14 (64x2) or 0.064/0.11/0.10 (128x2).",
  "why": "About 54 % of per-pair noise is within-prompt; the block-18 PC failed on rel 0.438 / 0.488.",
  "cost": "Doubling prompts roughly doubles loop time",
  "reuse": "rk_loop PROMPTS lists need 64 new prompts (32 RSI-topic + 32 scene)"
 },
 {
  "name": "0.6B diagnosis before any 0.6B loop",
  "what": "Read dsse_runs/r2_q3/rgi/geom_stats_qwen3-0.6b.json gate.C0_parts / uncomposed and per_block.pooled FG|CHG-DES, FG|WOR-DES, q, P_cell, plus point_only dot/white/pooled and lexical; compare with 1.7B; read logs/geom_06.log and logs/stats_06.log.",
  "why": "n_P = 0 means no block passed; the 0.8 'uncomposed' cut is within 0.013-0.094 of 1.7B's passing values.",
  "cost": "Minutes, no model",
  "reuse": "r2_rgi_stats outputs already on Drive"
 }
]
```

## unknowns (from the research stage; check against the verifier's notes above)
```json
[
 "The SEA loop's c, p, Q and A on u_sea per round at blocks 14, 18 and 6 (gpu*/rk/b<b>/loops.json ['SEA'][n]['sea']) are only in Drive; which PC_PASS component failed at blocks 14 and 6 (p, c or S_4) cannot be settled locally.",
 "SEA S_4 per block (summary_b<b>.json verdict.SEA_S4) is not echoed locally.",
 "Pilot values: GO1 at alpha at block 14 (pilot.json SEA_1[0]['sea']), GO1 at block 6 (c_1, p_1), and whether the 1.5-alpha retry ran at block 6 (summary_b6.json verdict.gates.source; pilot@1.5.json).",
 "GO0 AUC values and paired win rates, and the GO2 check numbers (pilot.json axes; state.pt checks) are not echoed.",
 "At-alpha damages at block 14 (state.pt dmg_SEA_1, dmg_RSI_a_1); my estimate of about 0.18 nats for SEA d_1 (by alpha^2 scaling) is an inference.",
 "Whether the low foreign-axis p-values for u* (0.013-0.046 at |cos| 0.43-0.61) come from axis leakage or from a very low effective noise dimension along u*; only the R0 re-analysis with centred nulls separates the two.",
 "The within-prompt noise share f = 0.54 comes from one loop (RSI_a) at one block; other loops and blocks may differ.",
 "The causes of the empty 0.6B arm: C0_parts, uncomposed and FG values in dsse_runs/r2_q3/rgi/geom_stats_qwen3-0.6b.json; the raw A1 states were not synced, so recomputing them needs a model rerun.",
 "Whether a damage-normalised or norm-preserving push keeps the loops off the shared style attractor needs a new run.",
 "unlearn_verdict.json and r2_manifest.json are only in Drive."
]
```
