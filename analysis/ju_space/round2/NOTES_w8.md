# NOTES_w8: analysis scripts of session r2_ana (implementer 8)

Frozen plan: `results/ju_space/plans/abschnitt11_12_runde2_plan.md` (sha256 `5e476340dfdb1d827582c869f2dfc08e32ed8e9a9dbebf7642acceb5bb5edc2d`),
sections 9.4 and 9.5 split (8). Amendments file read last at sha256 `94879041b16ce558de514d42e19ecdc3962c0287a17a497610ec5b7fd4249d4e`
(Amendments 1, 2 and 3). The report reads the file at run time, so a later amendment is listed without a code change.

## 1 Files and calls

| Script | Call (runner: `<work>/r1 <work>/r2in <work>/<out>`) | Output |
|---|---|---|
| `r2_linearity.py` | `python r2_linearity.py <r1_dir> <r2_root> <work>/unlearn/q1.json` | C-Q1: calibration (G42), nonlinear rung for RSI, RGI, sea, causal linear rung from C-Q3, descriptive AUC_cf and r* |
| `r2_rgi_verdict.py` | `python r2_rgi_verdict.py <r1_dir> <r2_root> <work>/rgi_verdict.json` | A6 readouts, A3 u_T_perp spillover, A7 (per block, over blocks, RSI-into-RGI, causal, dynamics, combined), AT8, AT7, AT5 labels |
| `r2_unlearn_verdict.py` | `python r2_unlearn_verdict.py <r1_dir> <r2_root> <work>/unlearn_verdict.json` | C-Q2 to C-Q6 (with C-Q1 from `unlearn/q1.json`), C-T0 to C-T7 |
| `r2_report.py` | `python r2_report.py <r1_dir> <r2_root> <work>/round2_summary.json` | `round2_summary.json`, and next to it `r2_manifest.json` and `pc_toy/gates.json` |

Each script also has `python <script> selftest <workdir>` (synthetic inputs, no model, no round-1 output). `r2_report.py` is also
the shared library of the other three: the finder, round-1 gates, PC gates, JSON conventions and the synthetic fixtures of the
self-tests. The other scripts import it as `RP`.

Environment: `DEVICE` (r2_linearity only; the runner sets cuda), `NT` (torch CPU threads; results do not depend on it), `R2_REPO`
(repository root if the scripts do not run from the checkout). `MOCK=1` or `QUICK=1` marks a code test. Only then can the replicate
counts be lowered (`R2_Q1_NCAL`, `R2_Q1_NPERM`, `R2_Q1_NFLIP`, `R2_A6_NBOOT`, `R2_AT7_NSIM`, `R2_AT5_BOOT`). A real run always uses
the plan's counts: 50 calibration replicates per plant, 200 permutations, 1,000 sign flips, 2,000 prompt-bootstrap replicates, 1,500
simulations per scenario, 10,000 cluster-bootstrap draws.

Order in r2_ana (the runner's step table): linearity, theta_verdict (implementer 4), rgi_verdict, unlearn_verdict (needs q1.json),
report. Every input is found by name: first `<r2_root>/<expected session>/<rel>`, then any `<r2_root>/*/<rel>`, then the folder of
`<out.json>`. Round-1 files are found through `r2_rk.find_r1` (OD-G9). Every lookup is logged in the output (`finder`,
`r1_finder`), with the sha256 of the main inputs (`inputs`).

## 2 Conventions of the outputs

- Strict JSON (NaN -> null, +-inf -> 'inf' / '-inf'), atomic writes.
- Status of a part: 'ran'; 'not run (round-1 output missing)'; 'not run (round-2 output missing: <file>)'; the producer's own
  status if it did not run (for example 'not run (AUC gate)'); 'not testable (...)' or 'not decidable (...)' where a gate or a
  condition of the rule closes it. A failing part does not stop the script; the other parts are still written.
- Labels: `descriptive (round-1 gate failed)` (G-desc, every Qwen verdict at block 14), `RGI axis not resolved` (GA(b) fails),
  `block 14 not usable` (G-blk(14) fails, C-Q5), `world without a testable meta level` (G-meta fails), `reproduction differs` (G46).
- OD-G8: every rule is recomputed here from the producers' numbers. Where a producer stored its own in-session view or a
  `rule_preview`, both are kept and a disagreement is listed under `od_g8` (the in-session value governs what ran).
- Seeds (OD-G6) are fixed and written to the outputs. r2_linearity: halves 25 (plan), flips 4101, permutations 4102, calibration
  4103. r2_rgi_verdict: prompt bootstrap 4301, sign-flip nulls 1107 + block, AT5 cluster bootstrap 0, AT7 simulation 1.

## 3 What each script does

### r2_linearity.py (C-Q1)
Inputs: `axes/tok14.pt` (r2_q1), `unlearn/q3.json`, `unlearn_toy/t3.json` (G41), the R1.state[14] axes (u_T, sea), and the R1
summary of block 14 (G-desc). Per concept (RSI, RGI, sea), with the two round-1 prompt folds:
1. Cross-fitted mean-difference AUC with 1,000 within-prompt sign flips.
2. AUC_cf: fold A's LEACE eraser on fold B, a fresh probe on fixed halves (seed 25).
3. Classic INLP r*: up to 8 removals, stop when the held-out AUC is <= 0.60.
4. Degree-2 probe on the top-32 PCs after LEACE, against 200 label permutations in one multi-right-hand-side solve.

The calibration runs first: 50 replicates each of a 1-dim plant and an XOR plant, scaled to u_T's per-token d'. G42 = XOR
'pattern' in >= 80 % and 1-dim no 'pattern' in >= 80 %. Nonlinear rung: 'pattern (within the top-32 PCs)' = degree-2 AUC >= 0.70
and above the permutation p99, else 'no nonlinear residue found'. Without G42 it is 'not valid (calibration failed)'; if G41 failed
the pattern claim is withheld. Linear rung from C-Q3's ratios: 'one direction', 'two-dimensional span' or 'not removed within the
tested 2-dim span'. Output `unlearn/q1.json`: `calibration`, `G42`, `G41`, `concepts.<name>` (the four statistics, both
directions), `linear_rung`, `statements.<name>` (both rungs in one sentence), labels.

### r2_rgi_verdict.py (Part A)
Inputs: `axes/r2_axes.pt` and `rgi/dyn_b<b>.json` (r2_q1); `rgi/geom_stats_<model>.json`, `rgi/geom_draws_qwen3-1.7b.npz`,
`rgi/causal_qwen3-1.7b.pt` and `rgi/dirs_qwen3-1.7b.pt` (r2_q3); `rgi_toy/{r0.json, r1/w<ws>.json, r1/at3.json, cross/w<ws>.json,
fam.json, forcing.json}` (r2_t1); `pc_toy/cross_pc/w<ws>.json` (r2_pc). Round 1: state.pt and loops.json of the blocks of B3, the
summaries, dn_delta, blocks_chosen, the PC summaries (G-PC).
- A6 (a): for every d_n (n = 1..4) of every frozen loop at every block of B3: c, p, Q (`r2_rk.stats_on_axis`, each axis with its
  own nulls) and A = c / sqrt(rel(d_n)), on u_RGI_perpT and u_RGI_perp with 90 % prompt-bootstrap intervals of A, and descriptively
  on u_dir, g_b, s_b. A6 (b): the GEN loops' rows as r2_gen_loop.py wrote them. A7 dynamics: self-loop rule, GEN rule, 'pull and
  delta'.
- A3: u_T_perp spillover at the blocks of B3 from the stored gradients and the round-1 null axes (the formula of
  `r2_rgi_stats.spillover`; the self-test checks equality), with the five outcomes of critique 22f.
- A7 per block (blocks of P), over blocks (V1 to V6), RSI-into-RGI line, causal line and combined statement, from r2_rgi_stats.py's
  numbers in the order of 2.6. The causal line uses the simultaneous interval r2_rgi_stats.py stores (`D_sim_ci`); r2_ana recomputes
  it from `rgi/geom_draws` as a cross-check (`simultaneous_crosscheck`) and uses its own only if `D_sim_ci` is missing.
- AT8: validity per world, G38, G39; per line and arm the per-world fits (`r2_sim.reg2_fit_world`) and the two-step verdict
  (`r2_sim.reg2_verdict`); ABSENT and 'perturbation-driven'; the answer mapping (late arm primary, early arm secondary);
  recursion; family (AT4).
- AT7: `r2_sim.reg2_simulate` with `r2_sim.reg2_noise` per line and arm (8 line x arm runs, 6 scenarios each, 1,500 simulations).
- AT5: slopes with `pc_analyze.cluster_boot_ols` and the label 'width/capacity, not delta'; PC models and pairs only at G-PC
  widths. AT0, AT3 and AT6 are passed through (descriptive).

Output `rgi_verdict.json`: `A6`, `A3_u_T_perp_spillover`, `A7` (`per_block`, `over_blocks`, `combined`, `gates`), `AT8` (with
`family`), `AT7`, `AT5`, `AT0`, `AT3`, `AT6`, `gates`, `summary`.

### r2_unlearn_verdict.py (Part C)
Inputs: `unlearn/{q1,q2,q3,q4a,q4b,armA,q5,q6}.json` and `q2.pt`, `q3.pt` (bootstrap draws of the random controls);
`unlearn_toy/t0..t6.json`; `pc_toy/t7.json`. Round 1: the state and loops of block 14 (R1-5 SVD and sigma, R1-6 seed-0
reference), the summaries, the PC summaries. Rules:
- C-Q2 'suppressed'; C-Q3 'unlearned' and the choice of the ablation for C-Q4a (G43); C-Q4a shallow / deep / partial (the
  producer's own-axis gap fields are carried as descriptive).
- C-Q4b gate (G44) and loop rule; arm A (G16, Amendment 3 (a), section 6).
- C-Q5 superposition, +- symmetry, dose, SVD of the 24 stacked d_n (the producers' bootstrap medians are carried as descriptive).
- C-Q6 helix, depth cap r*, self-repair.
- C-T0 to C-T6 with t intervals over seeds and the quorum '>= 4 of 5'; C-T7 Holm family at the G-PC widths.

Output `unlearn_verdict.json`: one entry per item (`C_Q1` .. `C_Q6`, `C_Q4_armA`, `C_T0` .. `C_T7`), `gates`, `od_g8`,
`amendment_2`, `amendment_3`, `summary`.

### r2_report.py
- Round-1 gates from the round-1 files (`r1_info`): B3, delta order (G9), G7, G8, finality per block (G4), G-desc (G11), G-blk
  (G12), LIKELY_wh (G13), G-attr with the attracted RND loop (G14).
- PC gates (`pc_gates`, written to `pc_toy/gates.json`): PCS = all rows of `gpu*/pc_summary_gpu*.json`, `pc_analyze.gates`.
  G-meta = G1 and G2, with the G35 fallback on r2_t3's round-2 BP-muP W 128 nets when those rows are missing ('gate from round 2').
  G-PC(W) per width: 'usable', 'BP-equivalent (not used for PC claims)' or 'not testable'. G37 inventory.
- Reproduction: G46 per retrained cell (B2(8) seed means within 0.05 and first95 within 500 updates in >= 4 of 5 seeds; every seed
  for B-T1's 4-seed cells), label 'reproduction differs'; G34 (r2_q3's dn_delta against R1 within 0.002 per block).
- Gate table G1 to G49: value or status, where it was decided, and the consequence that applies. G45 lists the 'not testable'
  results with the controls as run and, separately, those that are 'not testable' under the frozen OD-C2.
- One compact entry per part (A, B, C): verdicts, key numbers with intervals, gates, labels, parts not run. Part B comes from
  implementer 4's `theta_verdict.json` and r2_t2's `theta_toy/verdict.json`; the share of undefined bootstrap replicates is shown
  next to every theta rule output that carries one, and every verdict that an undefined bound made 'undecided' is listed
  (Amendment 3 (b)).
- `deviations`: the amendments file (sha256, every amendment with its change sections and post-hoc statement), what each
  amendment changed in the results (`amendment_results`), session packing and continuation sessions, parts 'not run', flags,
  labels, OD-G8 disagreements, the readings below, and the 'Deviation' / 'Amendment' sections of every `NOTES_w*.md` (with
  sha256).
- `r2_manifest.json`: the session manifests of all sessions merged (attempts, step records, code commits, versions), plus the
  sha256 of the analysis outputs.

## 4 Readings of the plan text (where it left a choice)

The report also writes these into `round2_summary.json` (`deviations.readings_implementer_8`).

C-Q1
1. The XOR plant's second label z is balanced within each class of y, not per prompt. A per-prompt z would be one of the 200
   permutation rows (per-prompt label swaps) and would raise the permutation p99 against the plant.
2. 'Ridge 1' of the degree-2 probe is added to the covariance of the standardised features. The 200 permutations refit only the
   probe; the eraser and the PCs stay those of the true labels.
3. Plant scale: a = d'_T x (token SD of the unplanted sea states along the plant direction) / 2, with u_T =
   R1.state[14]['axes']['u_T'].
4. INLP r* is computed per direction (A -> B, B -> A); the larger is the summary. The AUC_cf halves come from one default_rng(25),
   fold True first.
5. The linear rung needs g_0 > 0 (else 'not decidable (no baseline gap: g_0 <= 0)'). A pattern claim is withheld only if G41
   explicitly fails; with t3.json missing the gate is 'not decided', and the output says so.

Part C, Qwen
6. C-Q4b and arm A: the loop rule is an ordered list (un-searched; survives; re-routed; undecided because the reference does not
   re-find RSI; undecided). Each condition must hold at d_2 and at d_3.
7. C-Q4a: 'an ablation passed C-Q3' is r2_ana's own C-Q3 verdict of the ablation that ran.
8. C-Q6: the helix rule per block, block 14 primary; self-repair at the r* cap with every validated readout (18, 22); r* per
   relation.

Part C, toy
9. Quorum '>= 4 of 5' = ceil(0.8 n) seeds. C-T3: j* is the contiguous reading (the literal one is reported). C-T5: '> 24' counts
   as 25 also when fewer rounds ran. C-T2's Bonferroni level is 1 - 0.05 / 3.
10. C-T4 ladder: if the span block unlearns and its isotropic control neither unlearns nor keeps 0.9 x base, the rung is
    'undecided (isotropic control neither unlearns nor keeps >= 0.9 x base)'. In W4, 'specific' is the retain criterion alone.
11. C-T7: the family holds both erasure layers of the relaxation contrast (14 items per G-PC width), as r2_pc_toy.py records
    them. A width enters with at least 4 paired seeds.

Part A
12. A6: rel(d_n) stays at its round-1 value in the bootstrap of A. One draw of prompt weights serves all blocks and loops. Nulls
    of g_b and s_b: 2,000 sign flips with seed 1107 + block.
13. A7 dynamics: the damage condition (OD-A5) is applied to the loops a rule reads. A control loop above 0.15 nats makes the
    self-loop rule 'not decidable'. A block enters 'pull and delta' with G-blk(b), GA(b) and RSI_a's own pushes within the
    limit. The control exclusion ('generic attractor') also applies to the mirror ('drawn away').
14. A7 per block: BA's 'C0 for IMP-CHG' uses the simultaneous lower bound (the pointwise one is reported; implementer 3 reads it
    the same way). Robustness is the point-estimate form of 'GI and (IS or BA)' in the whitened metric and under every
    leave-one-out. 'Absorbed' and 'self-specific' need both frames.
15. A7 over blocks: V3 mirrors 'present at >= 2 top-third blocks' to the bottom third. If the 0.6B arm ran but r_DiD is not
    computable, V1's r_DiD condition counts as not met (V2). V5 has a third case that its two sub-labels do not cover: GI and (IS
    or BA) hold at some block, but nowhere robustly. The verdict is then 'V5: no RGI geometry' with that statement.
16. A7 RSI-into-RGI: the series are the frames-pooled rho_S and sigma_self.
17. A7 causal line: 'with simultaneous intervals' = sup-t 95 % intervals over the two statistics D(FG_causal) and D(rho_causal).
    r2_rgi_stats.py now stores them (`D_sim_ci`, `D_sim_crit`); they are used. r2_ana's own computation from `rgi/geom_draws` (the
    same construction) is a cross-check and the fallback. The verdict with the pointwise interval is reported next to it. If
    neither simultaneous interval exists, the pointwise reading is used and the output says so.
18. A3: w = the mean of the two IMP-DES half gradients per model object (`r2_rgi_stats.spill_inputs`).
19. AT8: ABSENT, 'present' and the recursion rule use per-world means of the level-1 models over the valid worlds. Recursion is
    read at the natural condition (pooling: the natural-condition entry; the all-conditions pooling is reported). A line that is
    'not testable' (G39) blocks every answer except 'no RGI geometry'. 'Not delta' may hold per line by either clause.
    'Perturbation-driven' is read from the reduced model for every line. A line's `verdict` is 'ABSENT' where the ABSENT
    condition holds; the two-step verdict stays in `rule_verdict`. Family: S-gain over the worlds whose B net passes level 1.
20. AT7: NW = the number of valid worlds, 3 inits, seed 1, p jitter 0.03 (reference). With fewer than 2 valid worlds the noise
    is measured on all worlds and flagged.
21. AT5: slopes over r2_t1's BP models plus the PC models of the G-PC widths (their S and T1 values = receiver means over the
    pairs with testable nets).
22. G46: every retrained cell is compared per round-2 source; first95 missing on both sides counts as agreement; weights are not
    compared.

Amendments: readings 23 and 24 in section 6.

## 5 Interface mismatches and how they are handled

- `r2_rgi_toy.py` first wrote `r1_job_name` with the condition arm ('nat_W..'). Its rows now also carry `param_arm` (the job-name
  arm), and `r1_job_name` uses it. G46 still derives the round-1 job name from the net name `<arm>_W<W>_w<ws>_i<init>` and lists
  any row whose `param_arm` disagrees (`param_arm_mismatch`; empty on the chain).
- `analysis/ju_space/gpu/data/rk_dirs.pt` is not in the repository, so `r2_rk.o3_bank` fails in r2_ana if POOLDIRS does not point
  to the Pool file. The analysis does not use the RND start directions; `r2_report.load_bank` then builds the bank with zero RND
  starts and records `bank_flag`.
- `r2_unlearn_qwen.py`'s first version tested 'gone' before the other loop outcomes. Its current `rule_loop` follows the plan's
  order (un-searched, survives, re-routed, undecided), the same as mine; any remaining difference appears under `od_g8`.
- The toy producers run the amended OD-C2 control and store the plan-text control as `full_at_dstar` (section 6).
- `armA.json` written before Amendment 3 has no unshifted interval. r2_unlearn_verdict.py then uses `ci90` as stored and gives no
  plan-text reading (`under_plan_text` null; `G16.interval` says why). The current r2_unlearn_qwen.py writes `ci90_percentile`,
  `boot_median` and `continues_with_percentile_interval`; all three are read.
- Other extra fields, read defensively or passed through: `boot95` / `values` in the toy t summaries (the rules compare only
  verdict fields), `rng_seeds` and `B_T4.peak` in the theta-toy outputs, `cos_boot_median` / `cos_disatt_boot_median` in q5.

## 6 Amendments

- Amendment 1 (U-statistic bootstrap; not post hoc) concerns implementer 4's script only. The report lists it.
- Amendment 2 (OD-C2: partial removal at the found dimension d*, one shared lambda) is declared post hoc, so section 14 asks for
  both readings. The producers ran the amended control and store the plan-text control's forward measures. r2_unlearn_verdict.py
  therefore does the following.
  - It reads each control record defensively (`dm_info`: `r` / `d_star`, `lam` / `lambda`, `matched` / `dm_matched`, `r0`,
    `target`, `drop`, `ladder`, `note`). It marks a control `amended` where the new step applied: note 'partial removal at d*
    (Amendment 2)', or d* > r0 with lambda < 1.
  - It gives the verdict with the control as run and, next to it, `under_plan_text`. Under the frozen text the control is the
    full removal at d*. Implementer 7 stores it as `full_at_dstar` in every Amendment-2 control (the same five draws at lambda 1;
    forward or relaxation measures, no searches). It misses the target by more than 0.02, so specificity is 'not testable'
    (G45). 'Unlearns' (C-T4 z1 operations), 'causal' (T4b in C-T3 / C-T4) and 'forget unlearns' (W4) are evaluated against its
    accuracy (t lower bound >= 0.9 x base), and the C-T4 ladder follows from them. C-T7's Holm family is re-run under the plan
    text with the stored relaxation contrasts (`full_at_dstar.contrast`); the other 12 items do not depend on the control. C-T5's
    dmr arm: the searches under that control were not run (only its side effects are stored), so the z1 verdict stays 'not
    testable' under the plan text, with the side effects reported.
  - Only for an output without `full_at_dstar` (written before the field existed) does the earlier fallback apply: the control
    half is 'not computable', so is the ladder rung, and C-T7's other items are bounded (claim if m x p <= 0.05, excluded if
    p > 0.05 or |mean| < 0.10, else 'not determined').
  - It lists every affected result in `amendment_2` (C-T3 T4b layers, C-T4 operations, ladders and W4, C-T5's z1 verdict, C-T7
    widths and family). The report copies the list and marks the amendment as post hoc.
  Where the amended step did not apply (d* = r0, or the ladder matched without partial removal), both readings coincide and
  `under_plan_text` is null. (Reading 23.)
- Amendment 3 (a) (G16's reference interval recentred on the point estimate) is declared post hoc with respect to round 1. G16
  decides what was generated, so the in-session G16 governs. `arm_a` keeps reading `ci90[0]`. It checks that `ci90` equals the
  percentile interval shifted by (point - median) (otherwise an `od_g8` entry), and carries `ci90_percentile`, `boot_median` and
  the stored unshifted G16 value as descriptive. `under_plan_text` is G16 with the unshifted interval:
  - same continuation decision: same verdict;
  - the unshifted interval would have stopped arm A: the frozen text's verdict is 'translation (as predicted), not un-searched';
  - it would have continued but the session stopped: 'not computable' (rounds 2-3 were not generated).
  `amendment_3` lists arm A when the readings differ. (Reading 24.)
- Amendment 3 (b) and (c) (theta clarifications, not post hoc) change implementer 4's rules. The report reads the rule outputs
  defensively, shows the share of undefined replicates next to each verdict, lists every verdict made 'undecided' by an undefined
  bound, and passes R4's 'degenerate' and R3's 'not entering (degenerate)' through as written.

## 7 Deviations from the plan text

None intended: no threshold, count, seed rule or verdict wording of the frozen text was changed. The readings in section 4 fill
gaps of the text. Two additions go beyond the text and decide nothing by themselves: the third V5 statement (reading 15) and the
'undecided' rung of the C-T4 ladder (reading 10).

## 8 Tests and results

Self-tests (synthetic inputs in the producers' formats; one case per verdict branch, gate failure and missing part; all pass):

| Script | Checks | Covers |
|---|---|---|
| r2_report.py | 33 / 33 | finder; round-1 gates (G7 to G14, would-have-read LIKELY, attractor); PC gates with the G35 fallback and every G-PC status; G46 / G34; G41; amendments parser (incl. multi-part Amendment 3); undefined theta replicates; packing; manifest; end-to-end report with and without inputs |
| r2_linearity.py | 20 / 20 | LEACE (zero cross-covariance), ridge probe, AUC, INLP r*, degree-2 probe on planted XOR and 1-dim signals, permutation null, calibration and G42 pass / fail, both rungs and every statement; end to end with every input (planted RSI / RGI / sea), with G41 failed and the labels, and with missing inputs |
| r2_unlearn_verdict.py | 49 / 49 | every condition of C-Q2 / C-Q3, G43, C-Q4a, every loop-rule outcome, G44 and G16 recomputation, R1-6 reference, Amendment 3 (a) both readings, C-Q5 (superposition, symmetry, SVD), C-Q6, C-T0 to C-T7 with every label, the ladder rungs, Holm against a direct computation, Amendment 2 plan-text readings from `full_at_dstar` (C-T4 ops and ladder, T4b, W4, C-T5 side effects, the C-T7 family re-run) and the fallback without it, 'not run' paths |
| r2_rgi_verdict.py | 61 / 61 | self-loop and GEN rules, the three-block rule, spillover (equal to `r2_rgi_stats.spillover`), A7 per block and V1 to V6, RSI-into-RGI, causal line with simultaneous intervals (stored and recomputed), combined statement, AT8 validity, G38, G39, every line status and answer, recursion, family, AT7, AT5, A6 / A3 on a synthetic round-1 tree with planted loops, gates and missing parts |

Chain test: `build_chain.sh` in the test directory builds MOCK / QUICK outputs of every producer in the runner's r2_ana layout,
with the final producer versions (including the Amendment 2 toy library). The round-1 mock tree comes from `r2_rk.mock_r1` and
`r2_pc_toy.make_fixture`; no git, no Drive, no model weights. All 37 producer stages ran (rc 0). `run_ana.sh` then runs the
r2_ana steps in the runner's order and mirrors every output into `<r2in>/r2_ana/` as the runner does. All steps ran (rc 0), with
lowered counts and once with the plan's full counts. On these inputs every comparison with a producer's in-session view or
preview agrees (`od_g8` empty). The MOCK verdicts are the expected 'not testable' / 'undecided' ones (random MOCK axes, 2 toy
worlds).

Timings, one CPU thread, synthetic data at real size:
- AT7 with 1,500 simulations x 6 scenarios x 8 line-arm runs on 8 worlds: 86 s (plan: 127 s).
- C-Q1 at D 2048 (64 prompts x 2 texts x 40 tokens): 14 s per concept and 16.6 s per calibration replicate pair, about 870 s in
  all on one CPU thread. The real run uses the A100 (the runner sets DEVICE=cuda; G2); the plan estimates 90 s there.

## 9 Open points

- The MOCK model has random axes: GA fails at every block, P is empty, G-meta fails, and QUICK has only 2 toy worlds. The chain
  test therefore exercises the file formats and the 'not run' / 'not testable' paths. The verdict branches are covered by the
  synthetic self-tests only.
- Amendment 2: C-T5's z1 verdict under the plan text stays 'not testable', because the dmr searches under the full removal at
  d* were not run. In MOCK / QUICK no control takes Amendment 2's step (every target is below 0.02), so the `full_at_dstar` path
  is covered by the synthetic self-test only (in the producers' layout).
- Real-input run time of r2_ana was not measured on the A100.
