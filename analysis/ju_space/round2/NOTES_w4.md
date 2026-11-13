# NOTES of implementer 4 (section 9.5 split (4)): r2_theta.py, r2_theta_verdict.py

Implements Part B Qwen (section 4.1 with the frame of 4.0) of the frozen plan `results/ju_space/plans/abschnitt11_12_runde2_plan.md`
(sha256 5e476340...edc2d), its gates G17-G26 and G49 (plus G4, G6, G11, G12, G19 where they touch Part B), the CLI of 9.2 and 9.4.
No round-1 output was opened. Every model test ran on MOCK round-1 states made locally (`r2_rk.mock_r1`: `rk_loop.py MOCK=1` at block
14 `all`, blocks 8 and 20 `pilot`, `loops`, `k5`, fake `dn_delta.json`, `blocks_chosen.json`, summaries, `shard_done.json`); the rules
were checked on synthetic cells and loops with planted gains. Built against the final r2_rk / r2_stimuli (NOTES_w1) and r2_sim (NOTES_w5).
Amendments that govern this package (results/ju_space/plans/abschnitt11_12_runde2_plan_amendments.md): Amendment 1 (U statistic under the
bootstrap; direction of R2's crossing) and Amendment 3 (b), (c) (undefined rule bounds; R4 (i) 'degenerate'). The changes made after the
code review (2026-11-13) are listed per finding at the end.

## Files, commands, code tests

| File | Command | Code test |
|---|---|---|
| r2_theta.py | `BLOCK=<b> python r2_theta.py <root> <r1_dir> <r2_dir> [rsib \| q0 \| checks \| cells \| turn2 \| power \| fdt \| all]` (default `all`; env DEVICE, NT, MOCK) | `MOCK=1 BLOCK=<b> python r2_theta.py <root> <r1_dir> <r2_dir> selftest` (builds the MOCK fixture in r1_dir if missing): 22 checks at block 14, 23 at blocks 8 / 20 |
| r2_theta_verdict.py | `python r2_theta_verdict.py <r1_dir> <r2_root> <out.json>` (env R2_THETA_POWER_REPS, default 400) | `python r2_theta_verdict.py selftest <workdir> [<r1_dir> <r2_dir>]`: 41 checks (synthetic; with the MOCK outputs of r2_theta.py also end to end) |

All tests in one script: `r2_impl/w4/run_all_tests.sh fresh` in the scratchpad (one process at a time, NT=1).

One process = one block (r2_rk.open_block). Every stage is resumable (skipped when its output exists) and runs the stages it needs first,
so any call order works: `q0` runs `rsib` first at an extra block, `cells` / `checks` / `power` need `q0`, `turn2` and `power` need `cells`.
Per-cell partial results are kept in `theta/parts/` while a stage runs (a crash or the time guard does not lose finished cells) and removed
when the stage's output is complete. A block whose round-1 output is not final (open_block raises R1Missing, G4) gets every output with
status `'not run (round-1 output missing)'` and nothing is copied or generated. The runner (r2_run_session.py) runs `r2_theta.py ... all`
once per block (one process per block); single stages can also be called one by one.

## Outputs (in <r2_dir>/theta/; texts as token ids; float32 per-pair arrays)

| File | Content |
|---|---|
| b<b>_r2RSIb.json / .pt (extra blocks) | B-Q0's second RSI loop: `rk.run_loop('r2RSIb', unit(u_T_perp(b)), 2, 4)`; json: status ('ran' or `'not run (...)'`), the four run_loop rows, keys rule, state keys; pt: `entries` = the loop's rk state entries `ev_ / dir_ / dmg_r2RSIb<SUF>_n` (r2_ana does not copy the round-2 state.pt), `start`, `rows`, `suffix` |
| b<b>_q0.json / .pt | B-Q0 (function `bq0`): `evaluations[L][n]` (n = 0..3: NF, U, SNR, g_par, G, rotation, decision, dec, first_div, damage_of_push; n = 0 = one-step gain of the start, rel := 1), `persistence`, `mode_set` (members, signs, debiased cos, rel_d4), `m_star2` (pt, interval, p5, guard_pass), `Gamma`, `cross_fit` (cos_AB, lambda_A_to_B / B_to_A / lambda_cf, g_par_AB, diff, switch, reason), `config` (mode 'standard' / 'switch' / 'not_resolved', m2_par, mG), `even_part`, `image_cos_d1_u_star`, `gate_p` (block 14), `cbar_b`, `kappa_b`, `labels`; pt: `cbar_ref` (per-reference sums and token counts of the cbar read), `cfg` (mode, u_push, u_read, cosAB, m2_par / mG / cbar_b per bootstrap row), u_star, uA, uB, d4 |
| checks.json (block 14) | G20: zero_push_identical (of 4), mirror_swapped (of 4), entropy_T03, entropy_T07, passed, texts |
| b<b>_cells.json / .pt | every cell (B-Q1/3/4/5, reserves included): pt `cells[name]` = dict(a, T, nseed, reserve, alpha, n_pairs, keys, prompt_of, seed_of, conts, first_div, windows{all, div, first20: delta, ok}, xp, xm (all), cbar{sum, cnt} per text, entropy{sum, cnt}, nll{sum, cnt}, rep4, capped_share, dropped, damage, KL_T), `damage_curve` {0.5, 1, 2, 4}, u_push, u_read, mode, order; json: point estimates per cell and window |
| b14_turn2.pt / .json | B-Q8: `cells[name]` (after a1_T0.7, a1_T0.3, a1_T1, a2_T0.7) = dict(cell, a, T, reserve, keys, prompt_of, seed_of, conts2, input_len, delta2, ok2, xp2, xm2, first_div2, n_turn1_not_round_trip), suffix_ids; json: point K and norm ratio |
| b<b>_power.json (+ .pt when it ran) | B-Q6: candidates (dec, rotation per cell), status 'not needed' (G24) or 'ran' with the chosen cell, its gains and cos with u*_b; pt: cell, push = unit(d_cell), d_cell, data (a cell-like dict of the 128 pairs) |
| b<b>_fdt.pt | B-Q7: X [128, D], S [128, D] (float64), y, ended_early, prompt_of, source ('ref' / 'new'), conts, keys, cbar_own |
| <out.json> = theta_verdict.json | top level: `B3`, `delta_order`, `block_flags`, `gates` (G20, G11_G_desc, G12_G_blk, G19_gate_p), `labels_global`, `reserves` (G21-G23, G25), `R3` (Gamma, g_alpha0_T*, T_slope, D_star, alignment_RSI, alignment_GI, entering, not_entering), `blocks[b]` = status, labels, descriptive, mode, G_blk, B_Q0 (recomputed), od_g8 (agree, disagreements, power_cell), cells (gains with intervals per window, covariates, 'entered' for reserve cells), Gamma_R3, R1 (verdict, alpha_star, D_star, onset, labels incl. damage-limited / convention-sensitive, resolvable_band, series), R2 (verdict, slope, T_c, margin, curvature, g_par_fit, G_norm_fit, rotation_with_T, collapse), R4 (verdict or verdict_descriptive, u_crit, default = B-Q2 alignment, i, ii), B_Q2, R5, R6 (block 14), B_Q6, B_Q7 (lambda_lin), power_rerun, enters_R3, sentence; `inputs`, `finder` |

## Readings of the text (the simplest reading consistent with it)

Stages and keys
1. Stage `rsib` is part of r2_theta.py (9.2 lists it; the task text did not). Keys keyseed('r2RSIb' + SUF_b, n, p, s, b) via rk.evaluate.
   G49 'ran (4 rounds)': if run_loop raises, the first failure is re-raised (the runner retries the step) and the second is recorded as
   `'not run (r2RSIb failed in two attempts; G49: no independent loop)'`; B-Q0 then runs without the loop ('no independent loop': no
   cross-fit, no switch, the block does not enter R3). This mirrors 7.2's 'not run after two attempts'. B-Q0 takes the loop from its
   exported entries (theta/b<b>_r2RSIb.pt), not from the in-memory round-2 state; a loop that 'ran' but whose entries are missing makes
   the q0 stage fail (it is retried) instead of writing a B-Q0 without it (review finding theta-1).
2. Cell keys keyseed('theta', b, p, s) (b int); turn 2 keyseed('theta2', b, a, T, p, s) with a and T as Python floats (e.g.
   'theta2|14|1.0|0.7|p|s'); power keyseed('r2PWR', b, p, s), s = 0, 1; FDT keyseed('r2fdt', b, p); checks keyseed('r2CHK', check, p)
   with check in 'zero' / 'mirror' / 'entropy', p = 0..3. Cell names `a{a:g}_T{T:g}` (a1_T1 = alpha_0 at T 1.0). Run order: main cells
   first, reserves last (all run, 2.8).
3. Code checks: zero push = push along u_push with alpha 0 (the hook active); mirror = the pair at +u_push and at -u_push (same keys, alpha_b)
   swap; entropy = the same pushed pairs (alpha_b u_push) at T 0.3 and T 0.7, pooled per-token mean of the entropy of softmax(logits / T).
4. Read windows and NLL come from one full unsteered pass with a block-b hook (B-Q1, critique v2-10) using r2_rk.read_windows' rules
   (tested equal to it); the NLL is at T = 1 over the continuation tokens (no stop token), logits computed only at those positions.
   Repeated-4-gram share = (number of 4-grams - distinct 4-grams) / number of 4-grams per text (NaN below 4 tokens). Entropy per text =
   mean over the sampling steps that produced its tokens (r2_rk's convention). Sampling KL at T = r2_rk.damage_T(u_push, a alpha_b, T).
5. Turn 2: every turn-1 pair keeps its (p, s); d^(1) = the cell's all-token d over its ok pairs, d^(2) over the turn-2 ok pairs; norm
   ratio sqrt(max(U2, 0) / U1). The fixed pieces are asserted with r2_stimuli.check_turn2 (once per stage).
6. FDT: 'the stop token if the text ended early' = log of the summed T-0.7 probability of rk.STOP at the next position (the stored texts
   do not keep which stop token was sampled); x_j from the same forward pass (V = 0); batches of 8 texts per forward / backward (rows are
   independent). J_hat uses cbar_b of B-Q0 (per bootstrap row); lambda_lin is reported for u*_b, u_push, u_read, u_T_perp, sea, u_RGI_perp
   (O2) and ('cov', b, 4).

B-Q0 and the cells
7. Evaluation n (n = 1..3) of loop L: push d_n = R1 dir_<L><SUF>_n, read-back pairs ev_<L><SUF>_n (= the O3 bank's ev[n + 1]), rel_n =
   bank rel[n] = R1.loops[b][L][n-1]['rel']; r2RSIb uses its own run_loop rows. n = 0 (start, rel := 1) is descriptive.
8. The decision gain's estimator (g_par if rotation >= 0.9, else G) is chosen on the point estimate and kept in every bootstrap replicate
   (B-Q0 evaluations, cells, power step); a rotation with G = 0 is undefined and gives G.
9. The m*^2 guard, Gamma_b, g_par^AB and lambda_cf use B-Q0 rows n = 2, 3 as written; Gamma_b's and g_par^AB's inverse-variance weights
   come from the bootstrap variances and are fixed across replicates; u*_b, u*_A, u*_B and the cells' u_push / u_read are fixed
   directions (they define what was pushed); rel_n and the cells' split-half rel are fixed point values (the plan defines rel_n as the
   R1 value).
10. Joint bootstrap: ONE matrix of prompt weights (r2_rk.prompt_bootstrap, 2,000 replicates, seed SEED_BOOT = 41, stratified by prompt
   type, row 0 = point) for every B-Q0 evaluation, reference cbar, cell, turn-2 cell, power step and FDT text of every block, so that
   differences across cells and blocks (R1 onset, R3, R6 transfer) are paired. Per replicate: cbar (prompt-weighted), U, projections, m*^2,
   m_A^2, Gamma pools, K, lambda_lin, and in B-Q2 u_crit itself (re-derived from the pooled pairs: 'pooled-pair bootstrap').
11. Bootstrap of the U statistic (Amendment 1; NOT a change of any point formula): U = mean of delta_i . delta_j over distinct pairs;
   in a bootstrap replicate products between copies of the same pair are excluded too: U* = (|sum w delta|^2 - sum w^2 |delta|^2) /
   (n^2 - sum w^2), and m^2 = U / |d|^2 uses |d|^2 = U (1 - 1/n) + sum w |delta|^2 / n^2 (an identity at the point estimate). Reason: the
   plug-in U of a resample counts the products of duplicated pairs with themselves and is centred on |d|^2 (the noise floor included)
   instead of U. Check (scratchpad r2_impl/w4/ustat_boot_check.py, 128 pairs, 64 prompts): at a true m^2 of 0.38 (sampling 5th percentile
   0.20) the plug-in bootstrap of m^2 has mean 0.61 and 5th percentile 0.50, i.e. it would pass the G17 guard; the distinct-pair form has
   mean 0.36 (point 0.38) and the U replicates are centred on U (bias +0.03 against +1.56 for the plug-in). This governs the guard (G17), the
   G intervals of R1 / R2 and m*^2 in the cells. Row 0 reproduces the plan's U, NF, m*^2 exactly (tested).
12. Cells (B-Q1): standard g_par = u*_b . d / (a alpha_b cbar m*^2), G = sqrt(max(U, 0)) / (a alpha_b cbar m*); switch g_par / cos(u*_A,
   u*_B), G / m_A; mode not resolved: along unit(d_4^RSI_a) with m^2 of RSI_a's evaluation n = 3. cbar = the cell's own cbar hook values
   pooled over both texts of every pair (r2_rk.CbarHook generation mode).

Rules (r2_theta_verdict.py)
13. 'G' in the reserve conditions (G21-G23) and in the user-form sentence = the decision gain of the primary window (the rules' header).
   Reserve 3: the extra block whose alpha_0 T 0.7 cell has the higher point estimate; only it can enter (lower bound > 0.8).
14. R1 at block 14: cells 0.5, 1, 2 alpha_0 (+ 4 alpha_0 if reserve 1 entered); at an extra block the series is [Gamma_b of R3 at alpha_0,
   the 2 alpha_0 cell if reserve 3 entered]; onset 'undecided' there (no 0.5 alpha_0 cell). alpha* from sim_gain's crossings (log-log,
   identical to r2_sim.crossings) per replicate; its 90 % interval over the replicates with a downward crossing (NaN dropped as in
   sim_gain.ci), ends beyond the grid reported as 0 / inf (censored); D* by OD-B9 (0 / inf outside the damage grid). 'damage-limited' =
   D*(point) > 0.15 nats for a FOUND-soft crossing; 'convention-sensitive' = two windows' alpha* intervals disjoint (at an extra block the
   low end of a window's series pools Gamma_b with that window's alpha_0 cell). FOUND precedes SUB / SUPER (a 0.5 alpha_0 cell above 1 with
   cells >= alpha_0 below 1 is FOUND-soft). Amendment 3 (b): a series point with undefined bounds makes R1 UNDECIDED (and the onset
   undecided).
15. R2: GLS of log(decision gain) (clipped at 1e-6, sim_gain's convention) on T - 0.7 with the inverse bootstrap covariance as fixed
   weight; every replicate refitted with the same weights. Order (crit_tline3): 'above 1 throughout' / 'below 1 throughout' (5th / 95th
   percentile band over 15 points of the tested range) before 'T_c located'. A replicate fit 'crosses 1' when its slope has the sign of the
   point fit's slope (Amendment 1's clarification; crit_tline3 counts downward crossings only), T_c = 0.7 - b0 / b1; located if >= 95 %
   cross and the 90 % interval lies in [0.2, 1.1] ([0.1, 1.4] when T 1.3 or T 0.15 entered). Amendment 3 (b): if the point fit is not
   finite or more than 5 % of the replicate fits are not finite, the verdict is 'undecided (gain undefined in x % of the replicates)';
   the fit, slope, T_c and the band of the finite fits stay reported.
   Reported besides: g_par and norm-G fits (descriptive), rotation with T, curvature (quadratic GLS, >= 4 temperatures), the B-Q4 collapse
   comparison (descriptive: F_T(a) = a dec(a, T) against F_0.7 at 0.7 a / T, log-log interpolated across the existing T 0.7 cells).
16. R3: the middle block's point estimate is compared with the end's upper bound; the non-overlap form is reported; entering = G-blk and
   in-session mode resolved and an independent loop (G49); non-entering blocks are listed with their Gamma next to the verdict. D*_b where
   FOUND-soft with finite D* at >= 2 blocks; T-slopes = per-block GLS slopes per replicate. Amendment 3 (b): an undefined bound (low end,
   high end or their paired difference) gives 'no monotone relation (undecided: a bound is undefined ...)'. A D* replicate that is censored
   above the grid (+inf) counts as undefined in this form (its paired difference is not a number).
17. R4: default u_crit = u*_b (standard; m_crit = m* per replicate) or u*_A (switch; m_A). (i) applies when M_b = {RSI_a} and another loop
   has rel(d_4) >= 0.5; Gamma_L = pool over n = 2, 3 of L's decision gains; 'first' moves u_crit to that loop (m_crit = sqrt(rel(d_4))),
   'degenerate' leaves no single mode: no '(RSI-specific)' suffix, and the block does not enter R3's alignment form; it is listed there
   as 'not entering (degenerate)' (Amendment 3 (c)); the per-loop alignments stay in R4.i.alignment (descriptive). (ii) the power-step mode
   unit(d_pwr) with m_crit = sqrt(rel_pwr); 'changes with T' / 'with alpha' by the chosen cell's T / a. 'generic attractor': debiased
   |cos(d_4^RND, u*_b)| = |cos| / sqrt(rel(d_4^RND) m*^2) >= 0.5. Specificity: u_crit oriented by the sign of its point RSI cosine, raw
   cosines, the lower bound through Amendment 3 (b). At a block failing G-blk R4 is computed but reported as `verdict_descriptive`.
   Alignment against delta: the R3 form on the sign-aligned corrected cosine with u_T_perp (and with u_RGI_perp).
18. Power step (B-Q6): gains by B-Q0's formulas with d_n = the chosen cell's all-token read-back d and rel_n = that cell's split-half rel.
19. R6: K's T-slope 'as in R2' = the same GLS on log K over the alpha_0 carry cells whose K interval lies above 0 (>= 2 needed).
20. Labels: G20 failed -> 'descriptive (code check failed)'; theta/checks.json missing -> 'descriptive (code check not run)'; G-desc ->
   'descriptive (round-1 gate failed)' at block 14; G19 p < 0.25 -> 'verdicts expected mostly undecided'. Labels are attached to every rule
   output of the block (verdict strings unchanged) and `descriptive` is set.
20b. Undefined bounds (Amendment 3 (b)): every rule bound goes through `lohi`, which returns (NaN, NaN) when the point value is not finite
   or more than 5 % of the 2,000 replicates are not finite (NaN, or +-inf); every comparison with it is then False: the rule is
   undecided, a reserve condition (G21-G23) is not met, R5's self-sustainment is 'undecided', R6's K verdict is 'undecided'. Every rule
   output carries `nonfinite_share` (and `bounds_undefined`); the reported intervals (S) stay the percentiles of the finite replicates with
   their `n_finite`.
21. OD-G8: r2_ana recomputes B-Q0 (and the power-step choice) with the same functions and the same bootstrap rows and reports any
   disagreement; the in-session mode and directions (q0.pt), and the in-session labels, govern the analysis. Its attenuation arrays
   (m*^2, m_A^2) come from the recomputation for the in-session mode.
22. Power re-run (OD-B14): r2_sim.gain_A2 and r2_sim.resolvable_band per block with the R1 series' cells (an extra block's Gamma_b is
   represented by its alpha_0 cell), sd = SD over ok pairs of u_read . delta_j / (alpha_b cbar_c m2_par) (mean a g_par: r2_sim's units),
   rho = mean Pearson correlation over pairs with the same (p, s) keys between every two cells, h = 0, 400 replicates.

## Deviations from the final plan
None in any measurement, threshold or rule beyond Amendments 1 and 3 (items 11, 15, 20b and 17). Additions: theta/b<b>_r2RSIb.pt, theta/b14_turn2.json, point estimates in
cells.json, `theta/parts/` during a stage, the descriptive collapse and fit diagnostics listed above.

## Tests run (this container, CPU, NT=1, MOCK; scratchpad r2_impl/w4/run_all_tests.sh fresh)
- `r2_theta.py selftest` at block 14: 22/22; blocks 8 and 20: 23/23 each. Among them: B-Q0's U, g_par, G by the plan's formulas; kappa_b
  = cbar read / med_b; u*_b and the configuration follow the guard / switch; code checks 4/4, 4/4, entropy T 0.3 < T 0.7; every cell with
  its pair count; cell texts, cbar and entropy reproduce r2_rk.evaluate_variant with the same keys; the full-pass windows equal
  r2_rk.read_windows (difference 0); NLL equals a direct forward; damage curve = rk.damage, KL = damage_T; a bootstrap row equals the
  weighted resample mean; power-step choice and a forced power step (keys, layout); turn-2 ids, keys and reading; FDT x_j = rk.read and
  S_j = finite differences of y; resume keeps every output, no parts left, round-1 keys unchanged; r2RSIb keys reproduce rk.evaluate and
  exactly the r2RSIb keys enter the round-2 state; G49 failure path; theta-1 (B-Q0 identical when the round-2 state lacks the r2RSIb
  entries, and a 'ran' loop without exported entries raises with no q0 written); a code-path variant (theta_std: guard threshold 0, power threshold
  +inf, test only) runs the standard configuration and the power step.
- Non-final block (summary_b20.json removed): every stage writes 'not run (round-1 output missing)', nothing copied.
- `r2_theta_verdict.py selftest`: 41/41. Synthetic cells with planted gains: a soft crossing at alpha* = sqrt(2) is located (FOUND-soft,
  interval covers sqrt(2), onset soft, D* by interpolation), a series built below 1 is SUB (not located), above 1 SUPER, a threshold
  FOUND-hard, SUB(>= alpha_0), damage-limited and convention-sensitive labels; a planted T_c = 0.6 is located (interval covers it), gains
  below 1 'below 1 throughout', flat noisy gains 'undecided', a crossing at T 0.17 located only in the extended window; R3 falls / rises /
  no relation / two-block / one-block cases; R5; R6 (K = 0.5 dissipative and 'transfers' to G = 0.5, 'different criteria', 'K not
  defined', 'K or G not positive'); reserves G21-G23; lambda_lin on a planted covariance; the U-statistic bootstrap; B-Q2 labels;
  planted-operator loops (Gamma within 10 % of lambda 0.7 and 1.3, guard, mode set, u*_b on the mode), the switch configuration, 'mode not
  resolved' at low SNR, 'no independent loop'; a cell along u*_b recovers the planted gain through m*^2. End to end on the MOCK outputs:
  every block analysed, OD-G8 agreement, R6 with four carry cells, the power re-run; the theta_std variant (R4 / B-Q2 / R4 (ii), OD-G8
  reports the guard disagreement); b_hi not final plus missing O2 axes. Amendment 3: the review's 'mode not resolved' case with m_A^2 = 0
  at the point (R1 UNDECIDED, R2 'undecided (gain undefined ...)' with slope and T_c kept, reserves not met, R5 undecided, shares
  written); the 5 % share (4 % keeps SUB / 'below 1 throughout' / 'falls', 8 % gives undecided in R1, R2, R3; a fit without any finite
  replicate returns without error); R4 (i) 'degenerate' without the suffix and listed 'not entering (degenerate)', a 'first' RSI_a keeping
  'RSI-aligned (RSI-specific)'.

## Open points
- MOCK (8 prompts) fails the m*^2 guard at every block, so the session outputs run in 'mode not resolved'; the standard / switch paths are
  covered by the theta_std variant and by the synthetic banks. MOCK gains are tiny (random model), so the MOCK end-to-end verdicts are
  code-path tests only.
- The power re-run costs 400 x (7 scenarios + 41 band points) r2_sim experiments per block (estimate 20-40 s per block on one core).
- Production sizes: b14_cells.pt about 50 MB, an extra block's cells.pt about 21 MB, b14_turn2.pt about 13 MB, r2RSIb.pt about 13 MB,
  fdt.pt 4 MB per block. r2_theta_verdict.py loads the round-1 state of one block at a time.
- For r2_report (implementer 8): the verdicts per block are `blocks[b].R1.verdict`, `R2.verdict`, `R4.verdict` (or `verdict_descriptive`),
  `R5.verdict`, `R6.cells[c].R6.verdict` / `.transfer.verdict`, `R3.*.verdict`, the sentence in `blocks[b].sentence`, labels in
  `blocks[b].labels` and `labels_global`.

## Changes after the code review (2026-11-13; review files scratchpad r2_review/fixes/theta.md and theta_verdict.md)
- theta-1 (confirmed, low): `stage_q0` builds the r2RSIb loop of the O3 bank from its exported entries
  (`theta/b<b>_r2RSIb.pt`: entries, start, suffix), as r2_theta_verdict.py does, instead of the in-memory round-2 state; when the rsib
  stage reports 'ran' but the loop is missing from the bank, `stage_q0` raises a RuntimeError (no q0 is written, the step is retried)
  instead of silently running B-Q0 as 'no independent loop'. Self-test (blocks 8 / 20): B-Q0 identical (mode set, m*^2, Gamma, mode,
  cross-fit) with the r2RSIb entries removed from the in-memory state; a 'ran' loop whose exported entries are empty raises. The optional
  extra of the verifier (independent_loop.present in the OD-G8 comparison) is not made (outside the finding).
- theta_verdict-1 (confirmed, medium) = Amendment 3 (b): `lohi` returns (NaN, NaN) when the point value is not finite or more than 5 % of
  the replicates are not finite (`bounds_ok`, `nf_share`); R1 is UNDECIDED when a series point's bounds are undefined; R2 keeps the fit,
  slope and T_c and returns 'undecided (gain undefined ...)' (and returns before the percentiles when no replicate fit is finite); R3
  marks an undefined bound; R5, R6, the reserve conditions, R4 (i)'s Gammas and the B-Q2 specificity bound follow the same rule; every
  rule output records `nonfinite_share`. Self-tests: the review's case and the 5 % share (see Tests).
- theta_verdict-2 (confirmed, medium) = Amendment 3 (c): no '(RSI-specific)' suffix on a 'degenerate' R4 verdict; a degenerate block leaves
  no alignment entry (`core['align_reps']` empty) and `alignment_entries` lists it in `R3.alignment_RSI.not_entering` /
  `R3.alignment_GI.not_entering` as 'not entering (degenerate)'. Self-test: the verifier's two-mode construction.
- theta_verdict-3 (refuted): no change.
