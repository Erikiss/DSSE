# NOTES_w7 - implementer 7: r2_unlearn_toy.py, r2_pc_toy.py

Plan: the final, frozen text `results/ju_space/plans/abschnitt11_12_runde2_plan.md` (sha256 5e476340dfdb1d82..., checked before
writing; written into every output as `meta.plan_sha256`). Part C toy (5.1 C-T0 ... C-T7), shared toy objects 2.5 (W0-W8),
conventions 2.6, gates G35-G37 and G41 / G45 of section 6, sessions r2_t3 / r2_t4 / r2_pc of section 7, CLI of 9.3, AT5's pairs with a
PC net (3.2). Library: `r2_toy_common.py` (implementer 5, final version of NOTES_w5) imported as TC; round-1 code (`pc_meta`,
`pc_common`, `pc_analyze`, `pc_summary`) imported or run unchanged. No round-1 output was opened: every test uses pc_meta QUICK runs
and pc_summary made here (fake round-1 PC folders in run_shard's layout) and fake r2_t1 AT5 nets made with TC.

Files written (only these, in `analysis/ju_space/round2/`): `r2_unlearn_toy.py`, `r2_pc_toy.py`, this file.

## 1 Commands, outputs, tests

| Script / stage | Session | Measurement | Output (in `<out_dir>`) |
|---|---|---|---|
| `python r2_unlearn_toy.py <out> t0` | r2_t3 | C-T0 structure (ring / Fourier, basis-free) | `unlearn_toy/t0.json` |
| `... t1` | r2_t3 | C-T1 superposition cap after training | `unlearn_toy/t1.json` (+ `partial/t1_<cell>.json`) |
| `... t2` | r2_t3 | C-T2 imposed codes (W5) | `unlearn_toy/t2.json` |
| `... t3` | r2_t3 | C-T3 circle world: gate, word length, hatching, erasure, c-plane, conjugation | `unlearn_toy/t3.json` (+ `partial/t3_ops_W128.json`) |
| `... t4` | r2_t3 | C-T4 (a) z1 operations + controls, (b) T4b, (c) two-family world | `unlearn_toy/t4.json` (+ `partial/t4_*.json`) |
| `... t6` | r2_t3 | C-T6 trajectory primitives, causal cap | `unlearn_toy/t6.json` |
| `... t5` | r2_t4 | C-T5 un-searching, BP primary protocol | `unlearn_toy/t5.json` (+ `partial/t5_W<W>.json`) |
| `python r2_pc_toy.py <out> gates` | (record) | G-meta, G-PC(W) from PCS, G37 inventory, missing jobs | `pc_toy/gates.json` |
| `... t7` | r2_pc | C-T7 PC (R1 PC-muP) and BP (R1 BP-muP) sides, W 128 and 256 | `pc_toy/t7.json` (+ `partial/t7_W<W>.json`) |
| `... cross_pc` | r2_pc | AT5 pairs with a PC net, every PC width, worlds 0-4 | `pc_toy/cross_pc/w<ws>.json` |

Env: `DEVICE` (cuda / cpu), `QUICK` (1 = code test), `NT`, `R2_MAXC` (largest number of chains per search call; default 8192 on
CUDA, 1024 on CPU; calls are split, results identical), `R1PC` (folder with the copied round-1 PC files, layout `<R1PC>/gpu*/...` as
the runner copies them; default `<out>/r1`), `R2T1` (r2_t1's output folder for `rgi_toy/models/at5_*.pt`; default `<out>/r2_t1`).
`all` runs every stage of a script. Every stage is skipped when its output exists; trained nets are saved in
`unlearn_toy/models/<name>.pt` (TC.save_nets, weight checksum) with `<batch>.json` (training rows) and reloaded (checksums verified) on
a rerun; long cells write `partial/<key>.json`, so an interrupted stage resumes. JSON is written atomically (NaN -> null).

Code tests (CPU, one thread, QUICK): `QUICK=1 NT=1 OMP_NUM_THREADS=1 DEVICE=cpu python r2_unlearn_toy.py <dir> selftest` and
`... python r2_pc_toy.py <dir> selftest` (results in section 5).

## 2 Output contents (what r2_unlearn_verdict.py / r2_report.py read)

Every stage JSON has `meta` (plan sha256, QUICK, device, seeds, budgets, FT_EVAL, NDRAW, the RNG seeds of OD-G6, sha256 of
r2_toy_common.py and the script, seconds) and `rule_preview` (section 4). Arrays are per seed in the order of `meta.seeds` (worlds
0-4; one init per world, init 0 = torch.manual_seed(ws)); every net's training row (first95, checkpoint, level1, eval_point
'checkpoint' / 'maxsteps', level_acc, checksum) is stored next to its measures (W0).
- t0: `cells[<arm>_W<W>].seeds[i] = {ws, tests (TC.structure_tests: a_taskcols, b_h1, b_h2, c_loo_ring_R2 K1-K3, d_h2_answer, d_W3,
  overlap k1 / k2), flags (freq_taskcols / h1 / h2, loo per K, ring, fourier_aligned, answer spectra)}`.
- t1: `cells[..] = {base (acc, B2 / B2_A / F for 8 and 4 shots, level1: G35's fallback input), families {xy, task, z1, h2}
  [str(k)][basis] = {acc, retained, B2_8, B2_4} per seed (random bases: mean over 5 draws), grids}`. Bases: 'pca', 'span' (random
  within the centred span; x / y and task columns), 'iso' (random isotropic).
- t2: `nets[<name>] = {code, k, ws, level1, first95, eval_point, heldout_eval, zero_shot (+ per task, zs_modal per new shift: modal,
  share, ring_dist, modal_is_trained), code_search (lr chosen per task, both lrs reported, code_lr_pick), cos_q_phi, free_A8, B2_8,
  B2_4}`.
- t3: `gate[K<cK>][arm] = [{ws, row, zero_shot (+ per untrained shift), hatch {extract, first95, first_snapshot_095, steps,
  zero_shot, trained}}]`; `word[L<L>][arc|spread] = [{ws, row, trained, untrained, distance, zero_shot_shift, zero_shot, by_j}]`;
  `ops = {base, t4b {h1, h2}, cplane {dim, acc, zero_shot, sp, rand}, conj {acc, acc_task, modal, share, modal_is_minus_c, sp}}`.
- t4: `cells[..] = {base (acc, integrity, label, sp, probes h1 / h2, relearn {300, 3000}, A_new {300, 3000}, B2, B2_A, F), ops[name]
  = {dim, op {acc, integrity, label, modal, sp, relearn, A_new, B2, B2_A}, iso {acc, integrity, sp, acc_draws}, span (PC blocks),
  dm {match (TC.damage_match dict + target, r0, key, layer), acc, integrity, sp, relearn, A_new}, push {beta, acc, target, matched}},
  t4b {h1, h2: acc, integrity, label, sp, probes {task_lin, ans_lin, task_mlp, ans_mlp}, relearn, rank, dm {...}}}`; ops = edit,
  block11, blk1, blk2, blk4, blk6, push_half, push_full, reflect (out-of-distribution), xspan (positive control for generic damage).
  `w4[W<W>] = {base {add, sub, sp, probes, relearn_add}, ops {block_add, erase_h1, erase_h2: {layer, dim, op {...}, dm {...}}}}`.
- t5: `widths[W<W>] = {chains [[arm, net, task]], A {300: [C, rounds], 3000: [C, rounds]} (best-of-budget held-out accuracy of
  s_r), ov_task / ov_xy [C, rounds], side_acc / side_sp [C, rounds] (z1, dmr), base_acc / base_sp, dm (per round: the dmr match),
  pc_last, seeds}`.
- t6: `free {rows, modes (TC.trajectory_modes per seed), null {var_p99, lis_p5, var_mean, lis_mean}, n_snap, dist {Ex, W2}, B2},
  caps[L] = {rows, level1, first95, dist, ratio_Ex, ratio_W2, B2}`.
- pc_toy/gates.json: `G` (pc_analyze.gates on PCS), `G_meta`, `G_PC[W] = {pass_, conds {G_meta, G0_pc_mup, G3, G4, G5,
  level1_4_of_5}, n_level1, bp_equivalent}`, `G35_fallback_needed`, `inventory_G37`, `missing_jobs`, `pc_missing`, `finder`.
- pc_toy/t7.json: `widths[W<W>] = {seeds_present, seeds_missing, enough_paired_seeds, eta, pc_nets / bp_nets (names, checksums, R1 job
  fields), pc / bp = {base, t4b {h1, h2} (as C-T4 b, PC relearning with the eraser inside the energy), unsearch {chains, A {300:
  [C, 25]}, rstar_300, long {chain, round, A_short, A_long}}}, pc.relax[layer][T] = {task / ans {relaxed, ff, delta_relax,
  delta_rand, ...}, contrast, ratio, ratio_rand}}`; `rule_preview` = ct7_family.
- pc_toy/cross_pc/w<ws>.json: field names as r2_rgi_toy's rgi_toy/cross: `models` (name, kind bp / pc, arm, W, init, ws, level1,
  first95, checkpoint, eval_point, checksum, delta_h2 / h1 / ctrl / ring, delta, Lambda (+ per shift, nulls), Phi, acc_proj, acc_free,
  A_free, acc_alpha, alpha, X, eta) and `pairs` (A, B, ws, A_W, B_W, A_arm, B_arm, A_kind, B_kind, A_init, B_init, A_level1, B_level1,
  S_gain, accB, accB_perm, r1, loo_err1, t1, t1_shuf, t1_gain, loo, loo_shuf, t4, t4_mean, t3, t3_mean, stitch).

## 3 Readings of points the plan leaves open (simplest reading consistent with the text; marked NOTE in the code)

General
- Seed counts: '>= 4 of 5 seeds' = ceil(0.8 n) (4 for the plan's 5 seeds). A t interval needs >= 2 values; a rule condition whose
  interval is missing fails. 'Base' in '0.9 x base' / '0.8 x base' = the seed mean of the intact nets' value.
- Random subspaces: TC.random_subspaces with seed crc32(key), key = stage | cell | operation | dimension (the same draws for every net
  of a cell; C-T1 keys include the world). Within-span draws per net.
- Evaluation chunks and search calls are split for memory only; chains are independent, results identical (selftest).
C-T0
- The rule's 'maximum frequency share' and 'leave-one-shift-out ring R2' are the task code's (the LOO test exists only for the task
  columns, c1_structure (c)); 'ring code' = task-code share > p99 and, for some K in 1..3, LOO R2 > 0.3 and > its p95; h1 / h2 centroid
  and answer-spectrum tests reported per object.
C-T1
- 'Random 2k-dim subspace of their 16-dim centred span' is drawn in R^W (uniform in the span of the centred columns), as are the task
  and isotropic bases; the x and y columns are cut separately with bases of the same kind. Task-code cut: kept code = mean column + Q Q^T
  (centred columns); B2 under a task-code cut keeps the new column in span(Q, task-mean column) after every step (the smallest linear
  subspace containing the cut task code). Hidden cuts: the cut network keeps z1 (h2) on mu + span(Q) for every input including the
  searched column (Ops removing the complement around the mean).
- k* in grid-index units for every family; the hidden families' control is the isotropic random subspace (there is no within-span
  random); the 'step' criteria use the PCA basis' accuracy curve.
C-T2
- Code-space search: held-out = split positions 12-288 (277; the 4 validation pairs 8-11 removed, W7); the lr of each (net, task) is
  the one with the higher mean validation accuracy over FT_EVAL (ties: 1e-2); its mean held-out accuracy over FT_EVAL and its final q
  (cos with phi(c_new)) are reported. Zero-shot and the snapping read-out on positions 8-288 (281, impose2). Free column search B2 =
  its A minus F of fresh BP-muP W nets (TC.fresh_F). 'snap' per (code, k) cell. Effects are paired by seed.
C-T3
- Gate nets at K 1 and K 2 for all three arms (the gate decision uses K 1). j*(L) = the largest j such that every distance 1..j has
  zero-shot >= 0.5 (the literal 'largest j with zero-shot >= 0.5' reported as jstar_literal); 'covers every untrained shift' = j* >= the
  largest distance. Hatching snapshots in every gate run. Conjugation on the 12 trained shifts' held-out pairs; 'modal output shift
  equals -c' counted over (net, trained shift) cases. c-plane: block of span(Ac) around the mean z1 of the trained-shift inputs, against
  NDRAW random 2-dim blocks.
C-T4
- 'Half / full block damage' of the push = 50 % / 100 % of the 11-dim span block's old-task accuracy drop, bisection on beta per net
  (start: the spread of the task columns along u1, doubled until the target is passed); unmatched betas are flagged (push.matched).
- Equal-dimension isotropic controls: edit = removal of a random 11-dim component from the centred task columns; push = a push of the
  same beta along a random unit direction; reflection = random 11-dim reflection; blocks = random blocks. Damage-matched controls =
  TC.damage_match (OD-C2, literal: bisection only at r0, so an overshoot at a larger dimension stays unmatched -> specificity 'not
  testable', G45), target = seed mean of the sum-probe drop, drop = mean over the 5 draws and the cell's nets.
- Relearning and A_new under a matched control use one draw per search (old task i: draw i mod 5; new task j: draw j); forward
  measures average the 5 draws. 'edit' leaves every search identical to the intact net (the searched column replaces the edited one), so
  its relearn / A_new / B2 are the intact values. All relearning of a cell (intact, operations, controls, T4b) is one batched search.
- 'Inverted' needs 'consistent permutation' in >= 4 of 5 seeds. Ladder: a control 'does not unlearn' = t lower bound >= 0.9 x base,
  'unlearns' = t upper bound <= 0.11; if the span block unlearns and its isotropic control is neither: 'undecided (isotropic control
  neither ...)' (the plan's partition has no case for it).
- W4: sum probe per task of its own input combination (x + y for add, x - y for sub; plan: 'sum probe' of W8); the add-task eraser is
  fitted on the add-task inputs (12 x 289) and applied to every input; 'specific' in W4 = retain (sub) accuracy >= 0.9 x its value under
  the matched random operation (seed means); probes on the add tasks at h1 and h2; ref of the block = mean z1 over all 24 x 289 inputs.
- T4b: eraser over every old task x 289 pairs (c1_structure); its matched control has r0 = the eraser's rank (11).
C-T5
- Overlap of a removed set U (r directions) with a span Q = ||Q^T U||_F^2 / r (task span = 11 centred task PCs; x / y span = the 32-dim
  span of the centred x and y columns). The dmr target of round r = the mean over seeds and tasks of the z1 arm's sum-probe drop with
  r removals; draws keyed by width and dimension. rand-arm directions from torch.Generator(crc32(width | world | task | round)). r* is
  reported for every arm; the rule's 'low-dimensional' uses the col arm, A_free = each arm's own s_0. If the last round's dmr match is
  unmatched, the z1 verdict is 'not testable (G45)'.
C-T6
- Snapshots from update 0 every 25 updates up to the net's evaluation point; 5 modes; nulls with default_rng(ws) (c4_traj); E_x = the
  x and y input columns (W1[:, :, :34]); distances are Frobenius norms from the common init.
C-T7
- eta_x = 1 / pc_meta.lam_max on the net's pretraining batch, power-iteration start torch.manual_seed(1000 + ws). The PC and BP sides use
  the same random draws (DM subspaces at equal dimension, rand-arm directions), so the pairs share them. T4b relearning includes the
  matched controls (draw i mod 5) in the same batched search; PC relearning has the eraser inside the energy (TC).
- Relaxation test: both erasure layers (h1, h2); the probes are the linear ridge task-identity and pooled-answer probes (the MLP probe
  is not evaluated on relaxed states); the C-T7 family therefore has, per G-PC width, 2 + 4 + 4 + 4 = 14 items (relaxation contrast at T
  20 and 100 for each layer). The 3,000-step searches of the PC protocol run at s_(r*_300) of each (arm, net, task) and at s_24.
- QUICK only: MIN_PAIRED = 2 (else 4, G37).
cross_pc (AT5)
- Per-model and per-pair lines are the library's definitions (TC.lambda_line, learn_alpha, s_line with TC.s_perms, f_line, stitch_fit,
  column_search), equal to r2_rgi_toy's for standard models; T3 only where both widths are <= 256 and the receiver is a BP net (T3 into
  a PC receiver is not defined); PC nets' level 1 / first95 / checkpoint come from their round-1 job files.
gates
- PCS rows are de-duplicated by job name with OD-G9's rule before pc_analyze.gates; G35's fallback flag = no BP-muP W 128 main row. The
  current runner (implementer 9) does not call `gates` in r2_pc (plan 9.4: r2_report.py writes pc_toy/gates.json in r2_ana); the stage
  and its functions are kept for that use.

## 4 Rules (for r2_unlearn_verdict.py)

`r2_unlearn_toy.rules_t0 ... rules_t6(stage_json)` evaluate the frozen 5.1 rules on a stage JSON (they read only the JSON) and are
stored as `rule_preview`. `r2_pc_toy.ct7_family(t7_json, widths=<G-PC widths>)` gives C-T7's Holm family (two-sided paired t tests,
Holm over the whole family, claim = Holm p <= 0.05 and |mean| >= 0.10); the preview uses every width that ran.
`r2_pc_toy.gates_from_rows(rows)` and `r2_pc_toy.g_meta_fallback(unlearn_toy_t1_json)` give G-meta / G-PC and G35's fallback ('gate from
round 2'). C-T3's gate result (G41) is `rules_t3(...)['gate']`.

## 5 Tests (this container: CPU only, one thread; QUICK = pc_meta's QUICK sizes and tiny grids)

QUICK code tests against the final r2_toy_common.py (NOTES_w5):
- `QUICK=1 NT=1 OMP_NUM_THREADS=1 DEVICE=cpu python r2_unlearn_toy.py <dir> selftest`: 16 / 16 checks, about 80 s. Unit checks:
  eval_old = TC.task_accuracy; sum_probe_gen = TC.sum_probe; the net cache reloads with verified checksums; a search split into
  chunks of 5 chains = one batched call; a zero-rank block is a no-op; the class-mean eraser equalises the task means (1e-7);
  damage_match and the push bisection run (matched or flagged); then every stage t0-t6 end to end (JSON + rule preview), and a rerun of
  t0 from the saved nets reproduces its output exactly.
- `QUICK=1 ... python r2_pc_toy.py <dir> selftest`: 9 / 9 checks, about 4.5 min (most of it the fixture: pc_meta QUICK runs PC-muP W 64
  (gpu4) and W 128 (gpu3), BP-muP W 64 and 128 (gpu5), seeds 0-1, K 12 and K 1, pc_summary per shard, pc_expected and shard_done as
  run_shard writes them, a duplicate of one model on gpu6). Checks: OD-G9 finder picks the shard whose pc_expected lists the job; PCS
  de-duplicated; gates = pc_analyze.gates on the rows; eta_x = 1 / pc_meta.lam_max and the batched relaxation of a loaded round-1 PC net
  = pc_meta.pc_relax; the batched S line = TC.s_line; gates, t7 (both sides at both widths, Holm family of 28 items) and cross_pc (one
  file per world with BP -> PC, PC -> BP and PC -> PC pairs, T4 everywhere, T3 only into BP receivers at W <= the T3 limit) run; G35's
  fallback formulas.
- At QUICK sizes nothing is learned (LEVEL1 0, 100 updates), so the QUICK previews are meaningless; they test the code paths only.

Non-QUICK smoke test of the semantics (scratchpad `r2_impl/w7/smoke_ut.py`: the real pc_meta constants, MAXSTEPS 6000, budgets 300 /
3000; grids reduced by monkeypatching to BP-muP W 64, seeds 0-1, W4 at W 64, C-T3 at W 64 with L 3 and 12, C-T2 k 2, C-T5 3 rounds, C-T6
L 1; about 8 min CPU). The pilot facts the plan lists in 5.1 reappear:
- C-T4: intact old-task accuracy 0.96, sum probe 0.87, relearn_300 0.60, A_new_300 0.18 / 0.36, B2(8) 0.20 / 0.31. Edit: accuracy 0.07
  with the sum probe at 0.98 (specific), relearning intact (0.60). 11-dim span block: 0.07 (sum probe 0.56 / 0.33). Top-1 / 2 / 4 / 6 PC
  blocks: 0.71 / 0.52 / 0.28 / 0.16, within-span random controls 0.87 / 0.73 / 0.51 / 0.32. Push at half / full block damage hits its
  targets (0.52 / 0.06; beta 1.6 / 4-12). Reflection 0.04 with the sum probe at 0.02 (out-of-distribution damage); x-span block: sum
  probe 0.00 (positive control), B2 0.01 (meta level unlearned). T4b: erasure at h1 removes the ability (0.07; linear and MLP task probes
  at chance), at h2 it keeps it (0.95 / 0.97) while the linear task probe drops to chance (0.06 / 0.08). W4 (W 64 nets ended at
  MAXSTEPS, add 0.80 / sub 0.78): block of the add span -> add 0.06, sub 0.36; erasure at h2 keeps add (0.79).
- C-T3: ring zero-shot 0.998 / 0.999, permuted ring never reaches level 1 (zero-shot 0.09 / 0.03) -> gate passes; arcs of 3 do not
  compose (j* = 1, run to MAXSTEPS and marked), L = 12 covers every untrained shift; generator extracted (zero-shot >= 0.5) at update 125,
  level 1 at 550-600; c-plane removal -> 0.06 (random 2-dim 0.63-0.70); conjugation: accuracy 0.0, modal shift = -c in 12 / 12 shifts,
  sum probe 0.99 -> 'inverted ability'; T4b in the circle nets: h1 0.26-0.31 with an MLP residue (0.85-0.91), h2 ability kept (0.94).
- C-T2 (k 2, W 64; no net reached level 1): held-out ring 0.46 / 0.36 < pring 0.55 / 0.52 < rand 0.83 / 0.87 (alignment and geometry
  effects below 0, as the plan predicts); zero-shot 0.04-0.14 with the modal output a trained shift.
- C-T5: col arm A_3000 0.44 -> 0.26 after one removal, rand arm stays 0.44, z1 arm 0.30 with the sum probe 0.87 -> 0.76; removed
  directions overlap the task span at 0.2 (random 0.17). C-T6: trajectory modes are Lissajous-like (|corr| with the cosine template 0.96
  / 0.85 / 0.7, above the nulls' 5th percentile) -> no 'real' mode; the L = 1 cap reaches level 1 without re-routing (ratios 0.95-1.07).
  C-T0: no ring code (task-code share 0.21 / 0.24 against null p99 0.23 / 0.24, LOO ring R2 < 0). C-T1: x / y columns keep-4 PCA 0.43 /
  0.53 against within-span random 0.15 / 0.20; task K = 5 (10 of 11 dims) PCA 0.93 / 0.96.
- Found and fixed with it: the C-T5 rule's 'never' value of r* is the plan's 25 ('> 24' counts as 25) whatever the number of rounds.

## 6 Open points

- OD-C2's literal reading (library; the coordinator's note) leaves many damage-matched controls unmatched: in the W 64 smoke test 9 of
  the 10 z1 operations of C-T4 and the h1 eraser had their smallest reaching dimension overshoot the target by more than 0.02 (e.g. block
  11: r 22), so their specificity is 'not testable' (G45) and, because an overshooting control is more damaging than the operation,
  'unlearns' (which needs the control's lower bound >= 0.9 x base) cannot pass for them. The same is expected at W 128 / 256, and for the
  C-T5 dmr arm (z1 verdict 'not testable' when the last round is unmatched). A bisection at the found dimension would be an amendment
  (plan text governs; not changed here). 'unlearns' is still computed against the unmatched control and flagged (dm_matched) for
  r2_unlearn_verdict.py to decide.
- CUDA not testable here (CPU-only torch). Memory is bounded by chunked evaluations (TC.CHUNK), search calls of at most R2_MAXC chains,
  and C-T1's keep-r ops built per (layer, r) (largest: [55 variants x 12 tasks, W, W - r] floats at W 256 ~ 170 MB plus one regrouped
  copy). Expected GPU time
  per session within the plan's estimates (one batched search per round / per cell; C-T7 PC side: one 3,000-step relearning search of
  240 chains and 25 x 300 + 3,000 PC steps per width).
- Runner (implementer 9): r2_t3 / r2_t4 / r2_pc steps and file names match (`{work} <stage>`, `unlearn_toy/<stage>.json`,
  `pc_toy/t7.json`, `pc_toy/cross_pc/w<ws>.json`; env R1PC, R2T1). The current runner does not call `gates` in r2_pc (pc_toy/gates.json
  comes from r2_report in r2_ana, plan 9.4); `r2_pc_toy.py <work> gates` and gates_from_rows / g_meta_fallback stay available.
- C-T7 relaxation probes are linear only (the rule's task-identity probe); the family includes both erasure layers (14 items per G-PC
  width); r2_ana fixes the family by calling ct7_family(t7, widths=<G-PC widths>).
- cross_pc relies on r2_t1's at5 files being TC.save_nets files with meta['spec'] and meta['row'] (as r2_rgi_toy.at5_groups writes
  them); BP model lines are recomputed in r2_pc with the library's definitions (they equal r2_t1's for the same nets up to rounding).
