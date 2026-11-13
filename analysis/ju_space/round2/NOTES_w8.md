# NOTES_w8: analysis scripts of session r2_ana (implementer 8)

Frozen plan: `results/ju_space/plans/abschnitt11_12_runde2_plan.md` (sha256 `5e476340dfdb1d827582c869f2dfc08e32ed8e9a9dbebf7642acceb5bb5edc2d`),
sections 9.4 and 9.5 split (8). Amendments file read: sha256 `07b6f7d280d6974b6eeac57646a4bd6d30f912d98f5d48239eec8a752201e559`
(Amendment 1 and Amendment 2).

## 1 Files and calls

| Script | Call | Output |
|---|---|---|
| `r2_linearity.py` | `python r2_linearity.py <r1_dir> <r2_root> <work>/unlearn/q1.json` | C-Q1: calibration (G42), nonlinear rung for RSI, RGI, sea, causal linear rung from C-Q3, descriptive AUC_cf and r* |
| `r2_rgi_verdict.py` | `python r2_rgi_verdict.py <r1_dir> <r2_root> <work>/rgi_verdict.json` | A6 readouts, A3 u_T_perp spillover, A7 (per block, over blocks, RSI-into-RGI, causal, dynamics, combined), AT8, AT7, AT5 labels |
| `r2_unlearn_verdict.py` | `python r2_unlearn_verdict.py <r1_dir> <r2_root> <work>/unlearn_verdict.json` | C-Q2 to C-Q6 (with C-Q1 from `unlearn/q1.json`), C-T0 to C-T7 |
| `r2_report.py` | `python r2_report.py <r1_dir> <r2_root> <work>/round2_summary.json` | `round2_summary.json`, and next to it `r2_manifest.json` and `pc_toy/gates.json` |

Every script also has `python <script> selftest <workdir>` (synthetic inputs, no model, no round-1 output). `r2_report.py` is at the
same time the shared library of the other three (finder, round-1 gates, PC gates, JSON conventions, the synthetic fixtures of the
self-tests); they import it as `RP`.

Environment: `DEVICE` (r2_linearity only; cuda when available, else cpu), `R2_REPO` (repository root if the scripts are not run from
the checkout; default: two levels above `analysis/ju_space`). `MOCK=1` / `QUICK=1` mark code tests; only then the replicate counts can
be lowered (`R2_Q1_NCAL`, `R2_Q1_NPERM`, `R2_Q1_NFLIP`, `R2_A6_NBOOT`, `R2_AT7_NSIM`, `R2_AT5_BOOT`). A real run always uses the
plan's counts: 50 calibration replicates per plant, 200 permutations, 1,000 sign flips, 2,000 prompt-bootstrap replicates, 1,500
simulations per scenario, 10,000 cluster-bootstrap draws.

Order in r2_ana (runner's step table): linearity, theta_verdict (implementer 4), rgi_verdict, unlearn_verdict (needs q1.json), report.
The scripts find every input by name: first in the expected session folder `<r2_root>/<session>/<rel>`, then in any
`<r2_root>/*/<rel>`, then in the folder of `<out.json>`; round-1 files through `r2_rk.find_r1` (OD-G9). Every lookup is logged in
the output (`finder`, `r1_finder`), with the sha256 of the main inputs (`inputs`).

## 2 Conventions of the outputs

- Strict JSON (NaN -> null, +-inf -> 'inf' / '-inf'), atomic writes.
- Status of a part: 'ran'; 'not run (round-1 output missing)'; 'not run (round-2 output missing: <file>)'; the producer's own status
  if it did not run (for example 'not run (AUC gate)'); 'not testable (...)' or 'not decidable (...)' where a gate or a condition of
  the rule closes it. A failing part never stops the script; the other parts are still written.
- Labels: `descriptive (round-1 gate failed)` (G-desc, every Qwen verdict at block 14), `RGI axis not resolved` (GA(b) fails),
  `block 14 not usable` (G-blk(14) fails, C-Q5), `world without a testable meta level` (G-meta fails), `reproduction differs` (G46).
- OD-G8: every rule is recomputed here from the producers' numbers. Where a producer stored its own in-session view or a
  `rule_preview`, both are kept and a disagreement is listed under `od_g8` (the in-session value governs what ran).
- Seeds (OD-G6) are fixed constants and written to the outputs: r2_linearity halves 25 (plan), flips 4101, permutations 4102,
  calibration 4103; r2_rgi_verdict prompt bootstrap 4301, sign-flip nulls 1107 + block, AT5 cluster bootstrap 0, AT7 simulation 1.

## 3 What each script does

### r2_linearity.py (C-Q1)
Inputs: `axes/tok14.pt` (r2_q1), `unlearn/q3.json`, `unlearn_toy/t3.json` (G41), R1.state[14] axes (u_T, sea), R1 summary of block
14 (G-desc). Per concept (RSI, RGI, sea) with the two round-1 prompt folds: (1) cross-fitted mean-difference AUC with 1,000
within-prompt sign flips, (2) AUC_cf (fold A's LEACE eraser on fold B, fresh probe on fixed halves, seed 25), (3) classic INLP r*
(up to 8 removals, held-out AUC <= 0.60), (4) degree-2 probe on the top-32 PCs after LEACE against 200 label permutations in one
multi-right-hand-side solve. Calibration first (50 replicates each: 1-dim plant, XOR plant, scaled to u_T's per-token d'); G42 = XOR
'pattern' in >= 80 % and 1-dim no 'pattern' in >= 80 %. Nonlinear rung: 'pattern (within the top-32 PCs)' = degree-2 AUC >= 0.70
and above the permutation p99, else 'no nonlinear residue found'; 'not valid (calibration failed)' without G42; no pattern claim if
G41 failed. Linear rung from C-Q3's ratios: 'one direction' / 'two-dimensional span' / 'not removed within the tested 2-dim span'.
Output `unlearn/q1.json`: `calibration`, `G42`, `G41`, `concepts.<name>` (the four statistics, both directions), `linear_rung`,
`statements.<name>` (both rungs in one sentence), labels.

### r2_rgi_verdict.py (Part A)
Inputs: `axes/r2_axes.pt`, `rgi/dyn_b<b>.json` (r2_q1); `rgi/geom_stats_<model>.json`, `rgi/causal_qwen3-1.7b.pt`,
`rgi/dirs_qwen3-1.7b.pt` (r2_q3); `rgi_toy/{r0.json, r1/w<ws>.json, r1/at3.json, cross/w<ws>.json, fam.json, forcing.json}` (r2_t1);
`pc_toy/cross_pc/w<ws>.json` (r2_pc); round 1: state.pt and loops.json of the blocks of B3, the summaries, dn_delta, blocks_chosen,
the PC summaries (G-PC).
- A6 (a): for every d_n (n = 1..4) of every frozen loop at every block of B3: c, p, Q (`r2_rk.stats_on_axis`, each axis with its own
  nulls) and A = c / sqrt(rel(d_n)) on u_RGI_perpT and u_RGI_perp with 90 % prompt-bootstrap intervals of A, descriptively on
  u_dir, g_b, s_b. (b): the GEN loops' rows as r2_gen_loop.py wrote them. A7 dynamics: self-loop rule, GEN rule, 'pull and delta'.
- A3: u_T_perp spillover at the blocks of B3 from the stored gradients and the round-1 null axes (the formula of
  `r2_rgi_stats.spillover`), with the five outcomes of critique 22f.
- A7 per block (blocks of P), over blocks (V1 to V6), RSI-into-RGI line, causal line, combined statement, from r2_rgi_stats.py's
  numbers in the order of 2.6.
- AT8: validity per world, G38, G39, per line and arm the per-world fits (`r2_sim.reg2_fit_world`) and the two-step verdict
  (`r2_sim.reg2_verdict`), ABSENT and 'perturbation-driven', answer mapping (late arm primary, early arm secondary), recursion,
  family (AT4). AT7: `r2_sim.reg2_simulate` with `r2_sim.reg2_noise` per line and arm (48 runs of 1,500 simulations). AT5: slopes
  with `pc_analyze.cluster_boot_ols` and the label 'width/capacity, not delta'; PC models and pairs only at G-PC widths. AT0, AT3
  and AT6 are passed through (descriptive).
Output `rgi_verdict.json`: `A6`, `A3_u_T_perp_spillover`, `A7` (`per_block`, `over_blocks`, `combined`, `gates`), `AT8` (with
`family`), `AT7`, `AT5`, `AT0`, `AT3`, `AT6`, `gates`, `summary`.

### r2_unlearn_verdict.py (Part C)
Inputs: `unlearn/{q1,q2,q3,q4a,q4b,armA,q5,q6}.json` and `q2.pt`, `q3.pt` (bootstrap draws of the random controls);
`unlearn_toy/t0..t6.json`; `pc_toy/t7.json`; round 1: state and loops of block 14 (R1-5 SVD and sigma, R1-6 seed-0 reference),
summaries, PC summaries. Rules: C-Q2 'suppressed', C-Q3 'unlearned' and the choice of the ablation for C-Q4a (G43), C-Q4a
shallow / deep / partial, C-Q4b gate (G44) and loop rule, arm A (G16), C-Q5 superposition, +- symmetry, dose, SVD of the 24 stacked
d_n, C-Q6 helix, depth cap r*, self-repair; C-T0 to C-T6 with t intervals over seeds and the quorum '>= 4 of 5'; C-T7 Holm family
at the G-PC widths. Output `unlearn_verdict.json`: one entry per item (`C_Q1` .. `C_Q6`, `C_Q4_armA`, `C_T0` .. `C_T7`), `gates`,
`od_g8`, `amendment_2`, `summary`.

### r2_report.py
- Round-1 gates from the round-1 files (`r1_info`): B3, delta order (G9), G7, G8, finality per block (G4), G-desc (G11), G-blk
  (G12), LIKELY_wh (G13), G-attr with the attracted RND loop (G14).
- PC gates (`pc_gates`, written to `pc_toy/gates.json`): PCS = all rows of `gpu*/pc_summary_gpu*.json`, `pc_analyze.gates`;
  G-meta = G1 and G2, with the G35 fallback on r2_t3's round-2 BP-muP W 128 nets when those rows are missing ('gate from round
  2'); G-PC(W) per width with 'usable' / 'BP-equivalent (not used for PC claims)' / 'not testable'; G37 inventory.
- Reproduction: G46 per retrained cell (B2(8) seed means within 0.05 and first95 within 500 updates in >= 4 of 5 seeds; every seed
  for B-T1's 4-seed cells), label 'reproduction differs'; G34 (dn_delta of r2_q3 against R1 within 0.002 per block).
- Gate table G1 to G49: value or status, where it was decided, and the consequence that applies.
- One compact entry per part (A, B, C): verdicts, key numbers with intervals, gates, labels, parts not run. Part B comes from
  implementer 4's `theta_verdict.json` and r2_t2's `theta_toy/verdict.json`.
- `deviations`: the amendments (post hoc ones marked), session packing and continuation sessions, parts 'not run', flags, labels,
  OD-G8 disagreements, the readings below, and the 'Deviation' and 'Amendment' sections of every `NOTES_w*.md` (with sha256).
- `r2_manifest.json`: the session manifests of all sessions merged (attempts, step records, code commits, versions), plus the
  sha256 of the analysis outputs.

## 4 Readings of the plan text (where it left a choice)

These are also written into `round2_summary.json` (`deviations.readings_implementer_8`).

C-Q1
1. The XOR plant's second label z is balanced within each class of y, not per prompt. A per-prompt z would be one of the 200
   permutation rows (per-prompt label swaps) and would raise the permutation p99 against the plant.
2. 'Ridge 1' of the degree-2 probe is added to the covariance of the standardised features. The 200 permutations refit only the
   probe; the eraser and the PCs stay those of the true labels.
3. Plant scale: a = d'_T x (token SD of the unplanted sea states along the plant direction) / 2, with u_T =
   R1.state[14]['axes']['u_T'].
4. INLP r* is computed per direction (A -> B, B -> A); the larger one is the summary. The AUC_cf halves come from one
   default_rng(25), fold True first.
5. The linear rung needs g_0 > 0 (else 'not decidable (no baseline gap: g_0 <= 0)'). A pattern claim is withheld only if G41
   explicitly fails; with t3.json missing the gate is 'not decided' and stated.

Part C, Qwen
6. C-Q4b and arm A: the loop rule is an ordered list (un-searched; survives; re-routed; undecided because the reference does not
   re-find RSI; undecided). A condition must hold at d_2 and at d_3.
7. C-Q4a: 'an ablation passed C-Q3' is r2_ana's own C-Q3 verdict of the ablation that ran.
8. C-Q6: the helix rule per block, block 14 primary; self-repair at the r* cap with every validated readout (18, 22); r* per
   relation.

Part C, toy
9. Quorum '>= 4 of 5' = ceil(0.8 n) seeds. C-T3: j* is the contiguous reading (the literal one is reported). C-T5: '> 24' counts
   as 25 also when fewer rounds ran. C-T2's Bonferroni level is 1 - 0.05 / 3.
10. C-T4 ladder: if the span block unlearns and its isotropic control neither unlearns nor keeps 0.9 x base, the rung is
    'undecided (isotropic control neither unlearns nor keeps >= 0.9 x base)'. In W4, 'specific' is the retain criterion alone.
11. C-T7: the family holds both erasure layers of the relaxation contrast (14 items per G-PC width), as r2_pc_toy.py records
    them; a width enters with at least 4 paired seeds.

Part A
12. A6: rel(d_n) stays at its round-1 value in the bootstrap of A. One draw of prompt weights serves all blocks and loops. Nulls of
    g_b and s_b: 2,000 sign flips with seed 1107 + block.
13. A7 dynamics: the damage condition (OD-A5) is applied to the loops a rule reads. A control loop above 0.15 nats makes the
    self-loop rule 'not decidable'; a block enters 'pull and delta' with G-blk(b), GA(b) and RSI_a's own pushes within the limit.
    The control exclusion ('generic attractor') also applies to the mirror ('drawn away').
14. A7 per block: BA's 'C0 for IMP-CHG' uses the simultaneous lower bound (the pointwise one is reported; same reading as
    implementer 3). Robustness is the point-estimate form of 'GI and (IS or BA)' in the whitened metric and under every
    leave-one-out. 'Absorbed' and 'self-specific' need both frames.
15. A7 over blocks: V3 mirrors 'present at >= 2 top-third blocks' to the bottom third. If the 0.6B arm ran but r_DiD is not
    computable, V1's r_DiD condition counts as not met (V2). V5 has a third case that the two sub-labels do not cover: GI and (IS
    or BA) hold at some block, but nowhere robustly; the verdict is then 'V5: no RGI geometry' with that statement.
16. A7 RSI-into-RGI: the series are the frames-pooled rho_S and sigma_self.
17. A3: w = the mean of the two IMP-DES half gradients per model object (`r2_rgi_stats.spill_inputs`).
18. AT8: ABSENT, 'present' and the recursion rule use per-world means of the level-1 models over the valid worlds; recursion at
    the natural condition (pooling: the natural-condition entry; the all-conditions pooling is reported). A line that is 'not
    testable' (G39) blocks every answer except 'no RGI geometry'. 'Not delta' may hold per line by either clause.
    'Perturbation-driven' is read from the reduced model for every line. A line's `verdict` is 'ABSENT' where the ABSENT condition
    holds; the two-step verdict stays in `rule_verdict`. Family: S-gain over the worlds whose B net passes level 1.
19. AT7: NW = the number of valid worlds, 3 inits, seed 1, p jitter 0.03 (reference); with fewer than 2 valid worlds the noise is
    measured on all worlds and flagged.
20. AT5: slopes over r2_t1's BP models plus the PC models of the G-PC widths (their S and T1 values = receiver means over the
    pairs with testable nets).
21. G46: every retrained cell is compared per round-2 source; first95 missing on both sides counts as agreement; weights are not
    compared.

## 5 Interface mismatches and how they are handled

- `r2_rgi_toy.py` writes `r1_job_name` with the condition arm ('nat_W..'), not the training arm. G46 does not use it; the
  round-1 job name is derived from the net name `<arm>_W<W>_w<ws>_i<init>`.
- `analysis/ju_space/gpu/data/rk_dirs.pt` is not in the repository, so `r2_rk.o3_bank` fails in r2_ana if POOLDIRS does not point
  to the Pool file. The analysis does not use the RND start directions; `r2_report.load_bank` builds the bank with zero RND starts
  in that case and writes the flag `bank_flag`.
- `r2_unlearn_qwen.py`'s in-session `rule_loop` tests 'gone' before 're-routed', and its in-session linear rung has no g_0 > 0
  check. Both differences show up under `od_g8` only when they change a verdict.
- The toy producers compute only the amended OD-C2 control (section 6 below).
- With no O2 axes file, `readouts_block` had no axis to read; it now still writes rel, the round-1 row and the pair count.

## 6 Amendments

- Amendment 1 (U-statistic bootstrap; not post hoc) concerns implementer 4's script only. r2_report.py lists it.
- Amendment 2 (OD-C2: partial removal at the found dimension d*, one shared lambda) is declared post hoc. Section 14 then asks for
  both readings. The producers ran only the amended control. r2_unlearn_verdict.py therefore
  - reads each control record defensively (`dm_info`: `r` / `d_star`, `lam` / `lambda`, `matched` / `dm_matched`, `r0`, `target`,
    `drop`, `ladder`, `note`) and marks it `amended` where the new step applied (note 'partial removal at d* (Amendment 2)', or
    d* > r0 with lambda < 1);
  - gives the verdict with the control as run, and next to it `under_plan_text`: under the frozen text that control is the full
    removal at d*, which misses the target by more than 0.02, so specificity is 'not testable' (G45) and 'unlearns' / 'causal'
    cannot be evaluated, because that control's accuracy was not measured;
  - lists every affected result in `amendment_2` (C-T3 T4b layers, C-T4 operations and W4, C-T5's z1 verdict, C-T7 widths);
    r2_report.py copies the list into Part C and marks the amendment as post hoc under `deviations.amendments`.
  Where the amended step did not apply (d* = r0, or the ladder matched without partial removal), both readings coincide and
  `under_plan_text` is null.

## 7 Deviations from the plan text

None intended: no threshold, count, seed rule or verdict wording of the frozen text was changed. The choices in section 4 fill
gaps of the text. Two additions go beyond the text and decide nothing by themselves: the third V5 statement (reading 15) and the
'undecided' rung of the C-T4 ladder (reading 10).

## 8 Tests

Self-tests (synthetic inputs in the producers' formats; one case per verdict branch, gate failure and missing part):

| Script | Checks | Covers |
|---|---|---|
| r2_report.py | 31 | finder, round-1 gates (G7 to G14, would-have-read LIKELY, attractor), PC gates with the G35 fallback and every G-PC status, G46 / G34, G41, amendments parser, packing, manifest, an end-to-end report with and without inputs |
| r2_linearity.py | 18 | LEACE (zero cross-covariance), ridge probe, AUC, INLP r*, degree-2 probe on planted XOR and 1-dim signals, permutation null, calibration and G42 pass / fail, both rungs and every statement, missing inputs |
| r2_unlearn_verdict.py | 42 | every condition of C-Q2 / C-Q3, G43, C-Q4a, every loop-rule outcome, G44 and G16 recomputation, R1-6 reference, C-Q5 (superposition, symmetry, SVD), C-Q6, C-T0 to C-T7 with every label, the ladder rungs, Holm against a direct computation, Amendment 2 views, 'not run' paths |
| r2_rgi_verdict.py | 59 | self-loop and GEN rules, the three-block rule, spillover (equal to `r2_rgi_stats.spillover`), A7 per block and V1 to V6, RSI-into-RGI, causal line, combined statement, AT8 validity, G38, G39, every line status and answer, recursion, family, AT7, AT5, A6 / A3 on a synthetic round-1 tree with planted loops, gates and missing parts |

Chain test: `build_chain.sh` in the test directory builds MOCK / QUICK outputs of every producer in the runner's r2_ana layout
(round-1 mock tree from `r2_rk.mock_r1` and `r2_pc_toy.make_fixture`; no git, no Drive, no model weights). `run_ana.sh` then runs
the four scripts in the runner's order with the runner's mirroring into `<r2in>/r2_ana/`. Results are in section 10.

## 9 Open points

- The MOCK model has random axes: GA fails at every block, P is empty, G-meta fails and only 2 toy worlds exist in QUICK. The
  chain test therefore exercises the file formats and the 'not run' / 'not testable' paths; the verdict branches are covered by
  the synthetic self-tests only.
- Plan-text numbers for the controls changed by Amendment 2 do not exist (the producers ran the amended control only). If both
  readings are wanted with numbers, r2_unlearn_toy.py and r2_pc_toy.py would have to store the accuracy of the full removal at d*.
- Run time of r2_ana on real sizes was not measured (no real inputs here). The closed-form parts are small; AT7 at 1,500
  simulations x 48 runs is the longest step of r2_rgi_verdict.py (timing in section 10).
