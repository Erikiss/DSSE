# NOTES_w6 - implementer 6: r2_rgi_toy.py and r2_theta_toy.py

Plan: the frozen text `results/ju_space/plans/abschnitt11_12_runde2_plan.md` (sha256 5e476340dfdb1d82..., checked before work).
Package: section 3.2 (AT0-AT6 toy parts of Part A), section 4.2 (B-T1, B-T2, B-T4, B-T5 and the toy rules of Part B), shared toy
objects 2.5 (W0-W2, W6, W7), statistics 2.6, gates G27 and G40, sessions r2_t1 / r2_t2 (section 7), CLI 9.3. Every net is trained in
the session by `r2_toy_common` (TC, implementer 5; final API of NOTES_w5.md). No round-1 file is read; nothing outside
`analysis/ju_space/round2/` and the test directory was written. Files written: `r2_rgi_toy.py`, `r2_theta_toy.py`, this file.

## 1 How to run (9.3; the runner r2_run_session.py already calls them this way)

    python r2_rgi_toy.py   <out_dir> r0 | r1 | cross | fam | forcing | selftest      (session r2_t1, in this order; fam after cross)
    python r2_theta_toy.py <out_dir> t1 | t2 | t5 | rules (alias t4) | selftest      (session r2_t2; t2 and t5 after t1, rules last)
    env DEVICE (cuda | cpu), NT, QUICK (1 = code test). R1PC is not used (no round-1 file in these sessions).

One process per stage, every stage batched over all its nets (TC.train / TC.column_search over [B, ...] stacks). Every stage is
resumable: it is skipped when its output exists; r1 and cross skip finished worlds; trained nets are saved first and reloaded (weight
checksums verified) when a stage reruns before its outputs exist, so a rerun recomputes only the lines (tested: identical JSON).
JSON is written atomically (tmp + os.replace); non-finite numbers are written as null. QUICK shrinks grids and widths (W 128 -> 24;
AT5 widths -> 16, 24 with T3 at W <= 16; 2 worlds in r1 / cross / fam; B-T1 wd {0, 0.3, 1, 3}, widths 16, 32; B-T2 widths 40, 48) and
uses pc_meta's QUICK constants through TC (MAXSTEPS 200, AFTER 50, LEVEL1 0, FT_STEPS 20).

## 2 Outputs (the interfaces for r2_pc_toy.py, r2_rgi_verdict.py and r2_report.py)

Every stage output (both scripts) carries `rng_seeds`: the random streams it used (OD-G6; review rgi_toy-3 / theta_toy-2). Every model row
of rgi_toy carries `finite_weights` (false = non-finite weights: `dropped` note, level1 false, every line null, no pair; review
rgi_toy-4) and `param_arm` (the parameterisation bp_mup / bp_sp; `arm` is the condition arm 'nat' | 'late' | 'early' | 'nat2';
review rgi_toy-2).

`<out_dir>/rgi_toy/`
- `r0.json` (AT0): `levels` = what AT2 reads: `task_late` [{target_x, gamma, status 'accepted' | 'flagged', flags}], `sham_late`,
  `task_early`, `sham_early` (gammas that passed level 1 in both inits); `selection` (per target, incl. dropped ones), `task_late_grid`
  (per grid gamma: per-init delta_h2, log2 ratio, held-out, level1; mean log2 ratio `mlr`; `pass_both`; `overshoot`), `sham_late`,
  `task_early`, `sham_early` (the same per fixed level), `dropped`, `delta_nat` (natural inits passing level 1) and
  `delta_nat_all_inits`, `models` (every R0 net with its delta).
- `r1/w<ws>.json` (AT2 + AT1 + AT3), one per world: `conditions` (name, arm 'nat' | 'late' | 'early', reg, gamma, k, target_x),
  `models` (one row per model of the 15 conditions), `natural_second` (the natural nets at first95 + 1500, same fields), `pairs` (the 6
  ordered cross-init pairs of every condition and of the second checkpoint), `pooling_world`, `pooling`, `within_run`,
  `mediators_by_condition`, `delta_nat`, `delta_nat_all_inits`. Condition names: 'nat', '<reg>_<arm><gamma:+g>' (e.g. 'task_late+0.01',
  'sham_early-0.003'); names starting with 'task' are the k = 1 arms, as `r2_sim.reg2_noise` reads them.
  Model row keys: name, cond, arm, reg, gamma, k, init, ws, W, x, p, level1, eval_point, first95, checkpoint, updates(_total),
  final_heldout, checksum, delta (pc_meta's h1 / h2 dict) and delta_h2, delta_h1, delta_ctrl_h2, delta_ctrl_h1, ring_h2, ring_h1;
  Lambda, Lambda_per_shift, Lambda_null_wrong, Lambda_null_perm; S_gain, T1_gain, T1, LOO, LOO_shuf (means over the model's 2 pairs as
  receiver, 2.7 C20), n_senders; Phi, acc_proj; acc_free (4 tasks, after 300 steps), A_free (mean over FT_EVAL), acc_free_curve;
  acc_alpha, alpha, X; recipe_acc (depth 1-6), rho (5); RG (mean over its pairs as sender); heldout_pre, colnorm, sat_h1; natural rows
  also B2_8, B2_4, A8_mean, F8_mean, F4_mean and r1_job_name (init 0: pc_meta's job name of the same cell, for G46, built from
  param_arm; '_quick' under QUICK as pc_meta).
  Pair keys: A, B, cond, ws, A_init, B_init, A_level1, B_level1, S_gain, accB, accB_perm, t1, t1_shuf, t1_gain, t1_tasks, loo, loo_shuf,
  r1 (ridge factor), loo_err1, RG, back, own, warm10, warm40.
- `r1/at3.json` (AT3, written once every world exists): correlation of delta_h2 with each line across the natural inits at both
  checkpoints (pooled over worlds, raw and with world means removed) and the within-run changes with t intervals over worlds.
- `cross/w<ws>.json` (AT5 among BP nets): `models` (12 per world: BP-muP W 64 / 128 / 256 / 1024 inits 0, 1 and BP-SP init 0; Lambda,
  Phi, X, delta, B2, r1_job_name, `as_receiver` / `as_sender` means of S_gain, T1_gain, LOO, T4, T3), `pairs` (all 132 ordered pairs:
  A, B, ws, A_W, B_W, A_arm / B_arm (the parameterisation, as r2_pc_toy's cross_pc rows), A_init, B_init, A_level1, B_level1, S_gain,
  accB, accB_perm, t1 ..., loo ..., t4 [4], t4_mean, t3 [4], t3_mean, stitch {lam, R2}; t3 only where both widths <= 256).
- `models/at5_<TC.net_name(spec)>.pt`: one `TC.save_nets` file per AT5 net (state with pc_common.MLP keys, arch, tables, meta with
  'spec', 'name' and the training 'row', weight checksum); read by `TC.load_nets` (r2_pc_toy.py's `_bp_groups` does exactly that) and
  by stage fam. Other nets are bundles `models/r0.pt`, `r1_w<ws>.pt`, `fam_*.pt`, `forcing_*.pt` (one file per architecture group:
  parameters, tables, per-net info, checksums; `load_bundle` in either script).
- `fam.json` (AT4): `A` (per world: the AT5 init-0 W 128 net with Lambda, Phi, X, acc_alpha12, acc_alpha8), `variants` {name: per world
  B row: level1, Lambda (B's unseen targets), Phi (B's own), shared_anchors, alpha_in_A, B_own_alpha, S_gain, accB, accB_perm, T1,
  T1_shuf, T1_gain, LOO, LOO_shuf, T4, T4_mean, delta}, `summary` {name: n_level1, trains (>= 4 of 5 worlds), t intervals over worlds of
  S_gain, Lambda, T1_gain, LOO, Phi, T4_mean (descriptive)}, `relu_calibration` (G40: lrs tried, chosen lr, testable).
- `forcing.json` (AT6): `models` (natural reference, wd x3, wd x5, shared split, bottleneck 4; worlds 0-1 x inits 0-1: Lambda, Phi,
  T1_gain, LOO, delta, free-column accuracy), `pairs` (init 0 <-> 1 per world and condition), `summary` per condition.
- `selftest.json`.

`<out_dir>/theta_toy/`
- `nets.json`: list of pc_summary-style rows (every key of pc_summary.row: name = pc_meta's job name of the same cell, arm, W, seed, K,
  gamma, wd_scale, lr, T, beta, first95, checkpoint, level1, final_train, final_heldout, coord_check, seconds, delta, B2_8, B2_4,
  A8_mean, F8_mean, F4_mean, B2_all, retention_mean) plus stage ('t1' | 't2'), eval_point, updates, first95_cens, censored, checksum,
  r2_name, B2_meta (K 12 rows), A8, F8, A_all8, retention (per task). `pc_analyze.cell` / `main_rows` work on them (tested). Built
  from `rows_t1.json` and `rows_t2.json`.
- `t2.json`: the in-session choice of B-T2's wd values (OD-G8) with the W 256 P2 result it came from (`seeds` = B-T2's net seeds).
- `t5.json`: per (net, tau): S, M (M_0 .. M_6), M1, M6, kappa, A_true, A_wrong, eval_point, level1, delta_h2; `pilot` and `G27`.
- `verdict.json`: P1 (phase per (wd, W) with counts), P2 (brackets, point estimates, flags), P3 (verdict, direction, switch_on and
  switch_on_flags per width, larger_widths), P4, B_T2 (choice, cells, od_g8 = the in-session wd values against the same decision
  recomputed from the rows, with agree), B_T4 (`peak` = the susceptibility peak per width, and per cell
  delta_h2, delta_ctrl_h2, ring_h2, delta_h1, B2_meta mean, susceptibility; peak per width; B2_meta against delta_h2 across all cells
  and widths; delta_h2 at the switch-on brackets), T5 (phases per (W, wd, tau), tau_c per (W, wd) and per net, tau_c regression).
- `models/t1_W<W>_K12.pt`, `_K1.pt`, `t2_...` bundles (B-T5 reloads the K 12 bundles).

## 3 Readings of points the plan leaves open (simplest reading consistent with the text)

AT0
- delta_nat = mean delta_h2 of the natural inits that pass level 1 (all inits if none passes, flagged); `delta_nat_all_inits` is
  reported too. The mean log2 ratio of a grid gamma needs both inits' branches and delta_h2 > 0 in both (else undefined, flagged
  'overshoot'); an undefined ratio is never selected. Ties in distance: grid order. Uniqueness: the target farther from the shared
  gamma moves (ties: the later target) to the next gamma of its own ranked list of gammas passing level 1 in both inits; this repeats
  (a cascade) until the selected values differ; a target with nothing left is dropped (flagged).
- R0 also reports delta for the sham and early nets (descriptive); their check is level 1 in both inits (early: a checkpoint).

AT1 / AT2 / AT3
- Late branch rows: evaluated at first95 + 500 (eval_point 'checkpoint', checkpoint = first95 + 500, branch_updates 500); level 1 =
  held-out >= 0.95 there (TC.train_late). A natural net without first95 has no late branches (those models are missing, not failed).
- x = None where delta_h2 <= 0. p: param distance to the same-init natural net at first95 + 500 (late), |gamma| / 0.01 (early), 0
  (natural, also at + 1500).
- Per-receiver S / T1 / LOO = mean over its 2 pairs whatever the senders' level 1 (each pair row carries both level-1 flags, so r2_ana
  can drop pairs with a failing sender if it wants to).
- Recursion is computed at all 15 conditions (pairs within a condition, same world). Pooling: `pooling_world` = AT1's literal 'alpha
  averaged over a world's models' (all models of the world at the first checkpoint), `pooling` = within each condition (descriptive).
  RG per pair; per model the mean over its pairs as sender.
- T4 is not computed in AT2 (AT2 lists Lambda, S, free T1 gain, Phi; T4 belongs to AT4 / AT5 and is not budgeted for AT2).
- B2 (8 and 4 shots) is computed for the natural nets (G46 needs first95, B2(8) and delta_h2 of the retrained cells; init 0 = pc_meta's
  job, named in r1_job_name).
- AT3 mediators (TC.mediators): pooled held-out on the 12 pretraining tasks, mean task-column norm, saturation share |h1| > 0.95 over
  12 x 289 inputs and units; X per model; the across-init correlations are pooled over worlds (n = 24) in at3.json.

AT4
- A = AT5's natural init-0 W 128 BP-muP net; 'other init' = AT5's init-1 net (critique 17). Every other variant is trained here with
  init 7 (torch.manual_seed(70000 + ws), the seed of ref crit/family.py and family2.py) and changes one factor against A (world split
  unless 'other split'; identity coding unless 'permuted coding'; ...).
- '8 shared shifts + spare': B pretrains on 9 tasks (slots 0-7 = A's pre[:8], slot 8 = the spare shift); S, T1 and LOO use the 8
  shared anchors; B's Lambda targets are its unseen shifts (the spare is dropped, TC.lambda_line targets_of); delta is not computed
  (it is defined over 12 tasks).
- Every accuracy in B uses B's labels (task family x + 2y + c for that variant), also T4 (A's outputs are for x + y + c, so T4 measures
  how far the functional route fails across task families). Phi_B uses B's own free columns and anchors; T1 / LOO are A -> B.
- ReLU (G40): world 0, init 7, lr 2e-3 then 5e-3; the first lr whose net passes level 1 is used for every world (the calibration net
  is world 0's variant net); none -> 'not testable', recorded in fam.json and in models/fam_relu.json so a rerun does not recalibrate.
- 'trains' = level 1 in >= ceil(0.8 n_worlds) worlds (4 of 5).

AT5
- T3 only where sender and receiver widths are both <= 256; its ridge fit is `stitch_fit_t`, TC.stitch_fit's estimator in float64
  torch on the session device with every fold's Gram matrix formed once (not once per grid value), because the numpy version costs
  about 1-3 TFLOP of host BLAS over AT5's 360 eligible pairs; equality is a selftest check. B2 for every AT5 net. `as_receiver` / `as_sender` = means over the model's
  pairs in its world (the slope label of AT5 is r2_ana's).

AT6
- 'natural' reference nets (same definition) are trained in the stage so forcing.json is self-contained. The bottleneck net's own
  improvement is its dt coefficients (v = U e; ref rgi_bneck) - its Phi is 1 by construction (positive control). The shared split runs
  10,000 updates and is evaluated at the end (level 1 = held-out on its shared held-out pairs >= 0.95 at the end); LOO-anchor uses the
  shared split's held-out pairs.

B-T1 / B-T2 (rows)
- K = 1 rows follow pc_meta exactly: first95 None, level1 False, checkpoint = the K 12 checkpoint used as its update count, no delta.
  A K 12 net without level 1 keeps pc_meta's checkpoint None but is evaluated at MAXSTEPS (W0: eval_point 'maxsteps', delta and B2
  computed); it has no K = 1 control, so no B2_meta.
- retention / B2_all: `finetune_all` = pc_meta.finetune(mode 'all') batched (tested bitwise against pc_meta), retention on the net's own
  K pretraining tasks as pc_meta (TC.b2_all always uses the 12 tasks; equal for K 12, tested).

Rules (r2_theta_toy rules)
- P1, P3 and T5 counts are written as shares (>= 3/4, <= 1/4 of the seeds), equal to the plan's '3 of 4' / '1 of 4' for 4 seeds.
- P2 follows ref theta_design/sim_toy_bracket.py: the peak (first maximum of the meta share) splits the grid; switch-on = last cell up
  to the peak with share <= 1/4 and the next cell with >= 3/4; switch-off mirrored from the peak on; 'below grid' if the lowest wd
  already has >= 3/4, 'above grid' if the highest still has; 'no meta cell' if no wd reaches 3/4; otherwise 'not bracketed'. Point
  estimate: piecewise-linear interpolation (in wd) of the share at 0.5 between the bracket ends.
- B-T2 wd values: the ends of the W 256 brackets that were found (on and / or off); if neither was found ('no meta cell', 'below / above
  grid', 'not bracketed') the fallback 0.3 and 3 (OD-B6), marked descriptive; P3 is then 'not decidable'.
- P3 'inside' at W 512 / 1024 = 0 of 2 seeds in meta at the lower bracket wd and 2 of 2 at the upper (OD-B10); intervals intersect when
  they share a point; 'shifts' carries a direction text (W 256 vs W 64 interval order; W 512 meta already at the lower or missing at
  the upper end). P3 has only the frozen two outcomes wherever B-T2 can test the W 256 bracket: a W 64 switch-on outside the grid gives
  'shifts' (direction 'W 256 switch-on at higher wd than W 64' for 'below grid', 'direction not determined' for 'no meta cell' or 'not
  bracketed'). 'Not decidable' remains only where B-T2 cannot test the bracket: W 256 has no switch-on bracket, B-T2 ran at the OD-B6
  fallback, or B-T2 has no W 512 net at a bracket wd value (a t2.json from an earlier decision; reported with both brackets). P4 uses
  the seed means of delta_h2 over all 4 seeds' K 12 nets at the two bracketing wd values; without a switch-on bracket at either width
  it is 'not shown', the plan's 'otherwise'.
- Susceptibility = SD (ddof 1) of B2_meta over the seeds that have one; peak = largest SD per width.
- T5 phases: 'self-improving', 'bistable' and 'dissipative' can hold together (the first is about S, the others about M_6); a cell
  carries every label reaching >= 3 of 4 seeds, 'mixed' if none. tau_c = the first adjacent tau pair where the cell is bistable at the
  lower and dissipative at the higher tau. 'tau_c against delta_h2 and W by OLS with seeds as clusters': per net the same switch on
  its own indicators, y = geometric midpoint of its pair, X = [delta_h2 of the net, log2 W], pc_analyze.cluster_boot_ols (10,000
  draws, seed 0, clusters = seeds), fitted only with >= 4 nets, 2 widths and 2 seeds.
- G27: the pilot net (W 64, wd 1, seed 0) runs first, on its own; the other 23 nets run afterwards (the gate changes only a label).
- G-meta (2.2, G35) is r2_ana's; verdict.json says so.

## 4 Tests (CPU, 1 thread, QUICK=1; scratchpad r2_impl/w6/)

- `r2_rgi_toy.py selftest`: 20 / 20 (17 before the review, plus one check per fix of section 6). AT0 rule on synthetic grids (accepted / flagged / duplicate resolved by distance / cascade to
  'dropped' / nothing passing / undefined ratios); batched lines against the library's per-pair references on tiny trained nets:
  free columns = TC.b2_A, alphas = TC.learn_alpha, S = TC.s_line, Phi and F (t1, t1_shuf, t1_gain, loo, loo_shuf, ridge factor) =
  TC.f_line, round trip and warm start = TC.round_trip / TC.warm_start, pooling = TC.pooling, T4 and T3 = TC.t3_t4, Lambda =
  TC.lambda_line (tolerance 2 / 281 per accuracy; observed: identical, apart from float rounding of Phi's ratio, 2e-8 relative); LOO
  held-out pairs of a shared-split net exclude its training pairs; the 8 + spare variant's task set; stitch_fit_t = TC.stitch_fit
  (same lambda, R2 to 1e-15, map to 2e-14 relative).
- `r2_theta_toy.py selftest`: 20 / 20 (15 before the review, plus the checks of section 6). P1 phases, P2 brackets (on / off with interpolation, below grid, above grid, no meta), B-T2
  choice, P3 transfer, P4, T5 phases (co-occurring labels, 'mixed'), tau_c; finetune_all = pc_meta.finetune('all') (weights rel. diff
  0.0, accuracies and retention equal) and = TC.b2_all for K 12; rows carry every pc_summary key and work with pc_analyze.cell.
- Every stage of both scripts under QUICK, in runner order (r0 6 s, r1 97 s, cross 15 s, fam 15 s, forcing 21 s; t1 6 s, t2 3 s, t5
  17 s, rules 1 s).
- Failure paths (harness `harness_level1.py`, TC.LEVEL1 patched to 1.01 = every net fails, and 0.055 = a mix): every level dropped
  with flags, AT2 with natural nets only, no late branches without first95, ReLU 'not testable', no K = 1 nets without a checkpoint,
  'not generalised within budget', B-T2 fallback; flagged / cascaded AT0 selections in the mixed run.
- Resumability: reruns skip every stage; deleting r1/w1.json, cross/w0.json, fam.json and verdict.json and rerunning recomputes them
  from the saved nets without training, byte-identical apart from timings.

## 5 Open points

- CUDA is not exercised here (CPU-only torch). The code is device-agnostic (DEVICE via pc_meta, TF32 off via TC); expected GPU hot
  spots in r2_t1: AT2's 384 x 5 Lambda evaluations (about 1,000 columns each) and the AT5 T4 searches into W 1024 receivers (chunked
  to a 2 GB activation budget per call) - within the plan's 90 s line budget by my estimate, not measured.
- Mixed-world batches: TC.train(fisher='auto') uses pc_meta's bitwise Fisher op sequence for the task regulariser (labels shared across
  worlds) but the one-hot form for the sham regulariser (x labels differ per world): rounding-level differences from a per-net
  pc_meta run that a collapsing log F can amplify (NOTES_w5). Definitions are unchanged; fisher='replica' would give the bitwise form
  at the cost of a class loop per world. I kept the library default.
- The verdict scripts (r2_rgi_verdict.py, r2_report.py) did not exist when this was written; section 2 is their input contract. AT7 /
  AT8, G38, G39, the AT5 'width/capacity, not delta' label and G46 are decided there, not here.

## 6 Review findings (2026-11-13) and what changed

Amendments 1-3 (plan_amendments.md) do not touch these scripts: neither calls `damage_match` (Amendment 2) nor computes a U statistic,
G16 or a Qwen theta rule (Amendments 1 and 3).

- theta_toy-1 (P3 'not decidable' when only W 64 lacks a switch-on bracket) - fixed. p3_transfer follows the frozen two-outcome rule:
  a missing W 64 bracket cannot intersect, so the verdict is 'shifts', with the direction from W 64's flag ('below grid': W 256 at
  higher wd; other flags: not determined); 'not decidable' only where B-T2 cannot test the W 256 bracket (section 3). The W 512 / 1024
  counts are reported in P3 in every decidable case. Tests: selftest P3 cases ('below grid', 'not bracketed', no W 256 bracket) and an
  end-to-end stage_rules run on synthetic files.
- theta_toy-2 (seeds not in the output JSON, OD-G6) - fixed: the tau_c regression records boot_seed 0, draws 10,000 and the percentile
  interval; t2.json, t5.json and verdict.json carry `rng_seeds` (net init, fresh F, coordinate probe, delta floor, W6 sampling stream,
  cluster bootstrap, global seed). Test: the synthetic stage_rules run.
- theta_toy-3 (P3 reads B-T2 cells at the recomputed bracket without checking that B-T2 ran there) - fixed: P3 is 'not decidable (B-T2
  has no W 512 nets at the W 256 switch-on bracket ...)' when a bracket cell is empty, with the in-session bracket next to it; verdict.json
  records OD-G8 (B_T2.od_g8: the in-session wd values against the same decision recomputed from the rows; agree). stage_t2 still reuses
  an existing t2.json, as OD-G8 prescribes ('the in-session value governs what ran'). Test: the synthetic stage_rules run with a stale
  t2.json.
- Aside of the theta_toy-1 verifier (r2_report reads B_T4.peak, verdict.json had only B_T4.<W>.susceptibility_peak): verdict.json now
  also has B_T4.peak = the peak per width; nothing else changed for it.
- rgi_toy-1 (T1 ridge factor not chosen by the leave-one-anchor-out error) - no change here: the comparison bug is in
  r2_toy_common.t1_choose_r (implementer 5 fixes it in the library). This script only passes the returned r on (field `r1`); nothing
  assumes r = 1e-4 (the 1e-4 in stitch_fit_t is the T3 grid's first value, unrelated), and the selftest compares f_pairs with
  TC.f_line, which calls the same function, so the check stays valid after the fix. The T1 / LOO numbers change with the library fix.
- rgi_toy-2 (info_of overwrote the parameterisation arm with the condition arm) - confirmed and fixed: info_of keeps the spec's arm as
  `param_arm` (output rows too); r1_job_name and AT5's A_arm / B_arm use it ('bp_mup' / 'bp_sp' instead of 'nat'); `arm` stays the
  condition arm, which the outputs and r2_rgi_verdict already use; infos of bundles written before the fix are backfilled from the net's
  stored spec. Test: selftest check rgi_toy-2; QUICK cross output has A_arm / B_arm in {bp_mup, bp_sp}.
- rgi_toy-3 (permutation seeds not in the output JSONs, OD-G6) - confirmed and fixed: every stage output carries `rng_seeds` (the
  SEEDS dict: net init, fresh F, coordinate probe, delta floor, Lambda label permutations, S correspondences, T1 and LOO shuffles, recipe
  chains and pairs, other split, coding, global seed). Test: selftest check rgi_toy-3; every QUICK output has the key.
- rgi_toy-4 (one net with non-finite weights aborts a stage, e.g. np.linalg.lstsq in Phi, and again on every rerun) - confirmed (lstsq
  raises on NaN columns) and fixed: every stage finds the nets whose parameters are not all finite and drops them from every line with a
  flag (row: finite_weights false, level1 false, a 'dropped' note; no pair; in AT4 a bad A leaves its world's B rows without pair values,
  with a note); Phi, the T1 / LOO maps and the T3 stitch also skip non-finite inputs instead of raising; the per-model means skip missing
  values. AT8 drops such a model anyway (it cannot pass level 1). Tests: selftest check rgi_toy-4 (the other models' lines are identical
  to a run without the bad net); harness `review/harness_nonfinite.py` (scratchpad r2_impl/w6/) corrupts one saved net each for r1,
  cross (AT4's A at the same time), fam and forcing and reruns them: every stage finishes, the net is flagged and has no pair.
- After the fixes: selftests 20 / 20 (r2_rgi_toy.py) and 20 / 20 (r2_theta_toy.py); the full QUICK pipeline of both scripts runs (r0 8 s,
  r1 127 s, cross 15 s, fam 15 s, forcing 21 s; t1 7 s, t2 3 s, t5 18 s, rules 1 s; scratchpad r2_impl/w6/review/H).
