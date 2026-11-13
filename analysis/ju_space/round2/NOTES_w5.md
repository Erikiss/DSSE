# NOTES_w5 - implementer 5: r2_toy_common.py and r2_sim.py

Plan: the final, frozen text `results/ju_space/plans/abschnitt11_12_runde2_plan.md` (sha256 5e476340dfdb1d82...; freeze commit
5b46e38c). The drafts were written against v2 and then brought into line with the final text (section 13 'Changes from v2'); the
v2-based parts that changed are listed in section 1. No round-1 output was opened; the tests use pc_meta QUICK runs made here.

Files (only these were written in `analysis/ju_space/round2/`):
- `r2_toy_common.py` - toy library W0-W8 (import only; `QUICK=1 NT=1 python r2_toy_common.py selftest [<work_dir>]`).
- `r2_sim.py` - ports of `crit/sim_reg2.py` (with `sim_reg.py`'s COND and ci) and of `theta_design/sim_gain.py` run_A2 (import only;
  `python r2_sim.py selftest`).
- `tests/ref_comparison_w5.json` - the 9.5 reference comparison of the final code (copy of the self-test output; section 5).
- this file. Section 7 lists the changes made after the review of 2026-11-13 (findings toy_num-1 to -4, toy_defs-1 to -7).

## 1 Changes made for the final text (v2 drafts -> final)

| Final-text item (section 13) | What changed in the drafts |
|---|---|
| v2-6 environment (W0) | Before `import pc_meta`: T=20, BETA=0.1, LR=1e-3, MAXSTEPS=6000 set; QUICK removed from the environment unless it is '1' (code tests); after the import M.T, pc_common.BETA[0], M.LR asserted, and (MAXSTEPS, AFTER, LEVEL1, FT_STEPS) = (6000, 500, 0.95, 300), or pc_meta's QUICK values (200, 50, 0.0, 20) under QUICK. The equivalence tests run in the same process, i.e. under the same environment. |
| v2-12t regulariser equivalence (W0) | The batched Fisher ratio now replicates pc_meta.task_fisher's op sequence (class loop, mean, sum) batched over nets whenever all nets of a batch share the labels (every task regulariser of a W1 batch; a one-world sham batch). With it the 300-update test with the task regulariser at gamma 0.01 through pc_meta.bp_update(gamma, lab) is bitwise equal (rel. diff 0.0). The v2 draft used a one-hot formulation: forward and gradient agreed to 3e-8 / 2e-7, but gamma +0.01 from update 1 collapses F to about 1e-6 within 50 updates and log F then amplified that rounding to a 13 % weight difference after 300 updates. The one-hot form stays for batches with different labels (sham across worlds); the sham Fisher term equals crit/sham.py's M.task_fisher(h2, lab_x, 17) to 1e-6 in both forms (test). `train(fisher='replica')` forces the bitwise form for mixed-world batches (slower: one class loop per world). |
| v2-12r soft target in a PC receiver (W7, W0) | The soft-target output term (beta CE(q, logits), relaxation gradient beta mult (softmax - q) W3) was already in the draft; the tests now follow W0: with one-hot q the batched relaxation equals pc_meta.pc_relax (rel. 0.0), with soft q it equals autograd of the soft-target energy built as pc_meta.pc_relax_autograd (1.5e-7). |
| v2-7 C-T7's BP side on R1 BP-muP checkpoints | `load_pc_state_dicts` loads any pc_meta checkpoint (PC-muP or BP-muP; docstring says so); BP and PC searches run on such nets alike (column_search with or without `pc=`). No trainer change needed. |
| v2-12p dmr draw | `damage_match` documents that the caller's drop function uses ndraw subspaces from a fixed seed (`random_subspaces(W, r, 5, seed)`), so new task j can take draw j at the matched dimension. |
| v2-12f / 9.5 reference comparison | The refs are read from `round2/ref/<relative path>` (the frozen copies; `ref_path` falls back to $R2REF and the scratchpad). Only their function / class definitions are executed (`ref_defs`, via ast) in a namespace with the repository's pc_meta and pc_common, so their absolute scratchpad imports do not matter (the shim of 9.5). The selftest writes `<work_dir>/ref_comparison_w5.json`: every check with both sides (or summaries), the largest difference, the sha256 of every reference file, of the two ports and of pc_meta / pc_common, and the plan sha256. |
| final W0 test list | Added: BP-muP with gamma 0.01; sham Fisher term against crit/sham.py; soft-target relaxation tests; B2_all / retention (pc_meta 'all' mode, needed by B-T1, missing in the v2 draft). |

## 2 Readings of points the plan leaves open (simplest reading consistent with the text)

- Level-1 accuracy: W1 = pooled held-out accuracy as a float32 mean, compared in float64 with LEVEL1 (exactly pc_meta); W3 / W5 = mean of the per-task accuracies (refs c3_angle / impose2); W4 = minimum over the two families of the mean per-task accuracy (c5_twofam). All per-task held-out sets have 145 pairs, so pooled and mean only differ at a threshold tie.
- Late branch (W2): a copy of the natural run's parameters and AdamW state (moments and step count) after the update at first95, then exactly 500 updates with the regulariser on from the first of them (= crit/sham.py's 'late' mode, checked bitwise). Level 1 of a branch = held-out >= 0.95 at its end.
- Fixed-length runs (K = 1 controls, late branches, AT6 shared split, fresh nets) also record first95 (pc_meta records none for K = 1 and sets level1 False there); their `level1` = level-1 accuracy at the end.
- Coordinate check (G0) probe and init are taken from the net's own init seed (pc_meta's job seed is the world seed; identical for init 0).
- delta_stats permutation stream: torch.Generator().manual_seed(77 + world seed) for every net of a world, as pc_meta (job seed = world seed) and rgi_pilot; for init > 0 nets the same world seed.
- 'other split' (OD-A9): the per-slot permutations are 17 consecutive draws of default_rng(5000 + ws); coding: px, py = two consecutive permutations of default_rng(6000 + ws). AT6 'shared' split: the first permutation of default_rng(5000 + ws), 144 training / 145 held-out pairs (rgi_long; int(round(144.5)) = 144).
- W3 split of shift value c = splits[c mod 16] (plan); W5 task j uses splits[j] (impose2; the plan is silent). W3 arm 'c-onehot' is the plan's (c one-hot, x and y ring-coded); c3_angle's 'onehot' arm (x and y one-hot) is available as 'x-onehot'. W4 MAXSTEPS is the plan's 6000 (c5_twofam used 4000).
- W6: sampling by Gumbel-max with uniforms [281, 17] from torch.Generator().manual_seed(7000 + 1000 ws + 100 j + 10 n) (float64), shared across tau and the two starts; the refit uses round 0's optimiser and step count (Adam lr 1e-2, FT_STEPS from zero) with the cross-entropy averaged over the 289 examples (8 true shots + 281 pseudo-labels).
- W7: projection 'after every step' is applied to the Adam iterate (v <- v - Q Q^T v, or v <- Q Q^T v for a kept subspace); restriction is defined for the free parameterisation only. Best-of-budget evaluations at steps 10, 20, ... (ties: earliest). Under QUICK the default budgets are (20, 60).
- W8 sum probe and linear task / answer probes: 'ridge ... lambda 1e-2' read as the ref probe (unlearn_design/probe.py, c1_structure.ridge_probe) with lambda changed: features standardised by the training mean and SD (ddof 0, + 1e-6), an intercept column, penalty lambda I on all coefficients, closed form in float64 (dual form when N < D).
- W8 probe halves: per task the first 72 held-out pairs (split positions 144-215) train the task-identity / pooled-answer probes, the other 73 test them.
- OD-C10 MLP probe = c1_structure.mlp_probe with OD-C10's values: 64 tanh units, Adam lr 1e-2 with L2 weight decay 1e-4, 300 full-batch steps, standardised inputs (torch std, ddof 1, as c1), nn.Linear-style init from torch.Generator(seed); Adam betas (0.9, 0.999) and eps 1e-8, torch.optim.Adam's defaults as in c1 (OD-C10 names no betas; the code before the review used pc_meta's (0.9, 0.98), review toy_defs-4).
- Class-mean erasure: mu and the class means over all inputs given (c1: all 12 x 289); Q = left singular vectors of the centred class means with singular value > 1e-10 x the largest (the span of all class-mean differences, rank <= K - 1).
- OD-C2 with Amendment 2: ladder r0, 2 r0, 4 r0, ... (<= W / 2); d* = the smallest dimension whose mean drop reaches the target; if the drop at d* exceeds the target by more than 0.02, bisection (30 halvings, stopping at |drop - target| <= 0.02) of one lambda in (0, 1] at d*, shared by the draws; if bisection does not reach 0.02, the evaluated lambda closest to the target is kept and the control is unmatched (Amendment 2's wording; review toy_defs-3). 'matched' = |drop - target| <= 0.02 (G45), also when no dimension up to W / 2 reaches the target but the W / 2 drop lies within 0.02 below it (review toy_defs-3). A target below 0.02 is matched by the r0 subspace.
- Lambda: unseen shifts = the 4 new shifts (held-out = split positions 8-288 of their slot, 281 pairs) and the spare shift (all 289 pairs); nulls: 'wrong' = every valid triple aimed at another shift, evaluated on c_new; 'perm' = 10 permutations of the anchors' shift labels from default_rng(4000 + ws) (count and seed not in the plan). `targets_of` lets a caller drop the spare shift for a net that pretrained on it.
- S line: one set of 30 permutations per world from default_rng(31 + ws) (crit/sham_s draws them sequentially per pair and task). X is left to the caller (learn_alpha and b2_A return the accuracies).
- F line: Phi uses the plan's linear span of the 12 anchor columns (the refs project on the affine hull mean + span of the centred anchors); Phi = mean over the 4 tasks of the projected accuracy / mean free accuracy. T1 shuffles: default_rng(123 + ws), 10, identity excluded; LOO-anchor: 10 maps per anchor from default_rng(1000 k + q) (rgi_lib).
- AT1 T1 ridge: lambda = the value of T1_GRID with the smallest MEAN leave-one-anchor-out relative column error (ties: the first), `t1_choose_r` returns (lambda, that mean error); the per-value errors come from `t1_loo_errors` exactly as in ref rgi_pilot.t1_choose_r, whose comparison of a summed error with a stored mean keeps 1e-4 in practice (review toy_defs-1; deviation from the ref, section 3).
- Recursion, T3 and T4 follow rgi_lib / rgi_pool / recipe_depth exactly (300 steps for T3 / T4; fresh Adam for every refinement). T3 into a PC receiver is not defined by the plan and raises.
- pc_eta: the power iteration's random start comes from torch.manual_seed(seed) chosen by the caller (round 1 used whatever state training left), and all RNG states are restored afterwards.
- C-T6 cap: after every AdamW step the centred part of the 12 task columns' change (decoupled decay included) is projected on the basis; the task-mean change stays free; the Adam moments are not projected.
- AT7 (r2_sim.reg2_noise): with dropped models the s_m^2 / 3 term uses the harmonic mean of the cell sizes (3 for complete data); the world x condition residual comes from an additive OLS fit of the cell means (missing cells allowed); jit pools the within-condition SD of x over the non-natural conditions; the condition named 'nat' is simulated at x = p = 0 (as sim_reg2), k = 1 for condition names starting with 'task'.
- OD-B14 (r2_sim.gain_A2): per-cell SDs replace sim_gain's sqrt(1 / (SNR^2 D_eff)) a^(q/2); unequal pair counts share the first min(n) pairs' heterogeneity and common noise and use the first n of each replicate's Nmax bootstrap indices modulo n (sim_gain_cellN's scheme); 'resolvable band' ends by linear interpolation in log g of the SUB-or-SUPER rate at 0.8 on a 41-point grid over [0.5, 2].
- OD-B14 correlation convention (review toy_num-1): `gain_A2` and `resolvable_band` take rho = the measured cross-cell correlation of the per-pair projections (r2_theta_verdict passes the mean pairwise Pearson correlation). sim_gain's noise model has one common factor zc with loading L in every cell, so two cells correlate with L^2 (h = 0); both functions therefore call `cells_experiment` with L = `correlation_loading(rho)` = sqrt(min(max(rho, 0), 1)), and the simulated correlation equals the measured one (exact for h = 0, the default and the caller's use; self-test: 0.5976 for 0.6 over 40,000 pairs). A negative measured correlation cannot be produced by one common factor and is simulated as 0 (no common noise); values above 1 as 1. `cells_experiment`, `scalar_experiment` and `run_A2_port` keep sim_gain's loading argument (their docstrings say so).
- C-T6 template (review toy_defs-5): |corr(time course, cos(pi i t / T))| with t = 0 .. T - 1 the snapshot index, the plan's template (C-T6 rule, OD-C13); ref c4_traj.modes used t + 0.5 (deviation from the ref, section 3). The real modes and the shuffled-increment nulls use the same template.
- B2_all retention (review toy_defs-7): pooled held-out accuracy afterwards on the net's own pretraining tasks, built from its spec (pc_meta's Xva: one task for a K = 1 control); nets without a spec (round-1 checkpoints) use the 12 world tasks.

## 3 Deviations (from reference scripts where the plan text governs; the others at rounding level or outside the plan's definitions)

From the reference scripts where the plan text governs (9.5):
- `t1_choose_r` (AT1: 'lambda by leave-one-anchor-out column error') differs from ref rgi_pilot.t1_choose_r, which compares the summed error of each grid value with the stored MEAN of the best so far, so a later value wins only with a 12 times smaller error and 1e-4 is returned in practice (8 of 8 random self-test cases; the reviewer's W-64 nets had the LOO minimum at 1). The library compares means. Signature and return format (lambda, mean error at lambda) are unchanged, so r2_rgi_toy.py's call `r1, e1 = TC.t1_choose_r(CA, CB)` needs no change; its r1 / loo_err1 fields now report the LOO choice, and the T1 gain and the LOO-anchor transfer use it. Exactness against the refs is kept where the refs are compared: `t1_loo_errors` reproduces the ref's per-value errors (the ref's comparison applied to them gives its choice exactly), and rgi_lib.pair_transfer / span_transfer run with the corrected chooser in their namespace.
- `trajectory_modes` uses the plan's cos(pi i t / T); c4_traj.modes used cos(pi i (t + 0.5) / T) (a phase shift of pi i / (2 T); |corr| moved by at most 0.021 at T = 40 in the self-test, and the shift shrinks as 1 / T).
- `mlp_probe` uses OD-C10's lr 1e-2 (c1_structure.mlp_probe: 3e-3); betas and weight decay as c1.

Other:

- Evaluations and searches compute z1 as Ex[:, x] + Ey[:, y] + b1 + v (gather) instead of a one-hot matmul: rounding-level; the column searches reproduce pc_meta.finetune exactly at the accuracy level (24 / 24 BP chains, 8 / 8 PC chains).
- Batches whose regulariser labels differ between nets use the one-hot Fisher form (rounding-level; see section 1).
- Depth-3 nets (AT4 '3 hidden tanh layers') use the muP hidden learning rate lr x 128 / W for both hidden layers; crit/family2.Net used lr: identical at the plan's W 128.

## 4 API of r2_toy_common.py (plan item -> functions)

- W0 nets and training: `Nets.from_specs(specs)` (reference inits per kind, checked bitwise), `Nets.from_states`, `subset`, `cat`, `x_cols` / `y_cols` / `task_cols`, `logits`, `hidden_acts`, `state(i)` (pc_common.MLP keys for depth-2 'mlp' nets), `to_mlp(i)`; `train(specs, fixed_steps=None, init_state=None, max_steps, after, level1, extra_after, save_first95, snapshot_every, snapshot, task_basis, coord_check, fisher)` -> TrainResult(nets, rows, nets2, has2, first95_state, snapshots, data); `train_late(nat, idx, branch_specs, n_updates=AFTER, level1=LEVEL1)`; `AdamWB`; `task_fisher_b`; `net_name`, `weight_checksum`, `save_nets`, `load_nets`, `load_pc_state_dicts`, `r1_training_batch`.
- W1 / B2: `world`, `spare_shift`, `init_seed`, `fresh_seed`, `taskset_w1`, `new_task_sets`, `old_task_sets`, `stack_sets`, `b2_A(nets, ws_of, nshots, ..., pc_eta_of, ops, keep)`, `fresh_F(arm, W, seeds)`, `b2_all`, `b2(..., with_all)`; `delta_stats(nets, seeds, pres)`; `mediators` (AT3); `task_accuracy`; `param_distance`.
- W2: spec keys gamma / reg / when; `train_late`.
- W3 / W4 / W5: `circle_tables`, `four`, `w3_shifts`, `taskset_w3`; `w4_tasks`, `w4_split`, `taskset_w4`; `code_table`, `taskset_w5`; `zero_shot(nets, ws_of, shifts_of, mode)` (accuracy, modal shift, share; usable as a snapshot callback for hatching).
- W6: `self_train(nets, ws_of, taus, rounds, tasks, steps, starts)` -> dict(keys, A [chains, rounds + 1]); `w6_summary` (S, M_n, kappa of OD-B5).
- W7: `column_search(nets, net_of, shots, heldout, steps, lr, eval_at, val, budgets, val_every, param, q0, keep, remove, ops, loss, target, pc, keep_cols)` -> dict(acc, A, best {budget: val, step, heldout {name}, col}, v, q); `anchored_param`, `coords_param` ('bneck', 'impose'); `cols_accuracy`, `predictions`, `hidden_states`, `task_states`.
- W8 / OD-C2 / OD-C16: `Ops` (z1 block / reflection / partial removal / push; h1 / h2 erasure; `expand`, `regroup`, `view_per_net`), `block_ops`, `z1_ref`, `random_subspaces`, `random_in_span`, `complement_basis`, `class_mean_eraser`, `sum_probe`, `ridge_probe`, `mlp_probe`, `heldout_halves`, `task_answer_features`, `task_answer_probes`, `damage_match`, `modal_shift_map`.
- AT1 lines: `lambda_triples`, `lambda_targets`, `lambda_line`; `learn_alpha`, `carry_alpha`, `s_perms`, `s_line`; `ridge_coef`, `t1_map`, `t1_loo_errors`, `t1_choose_r`, `proj_linspan`, `f_line`; `recipe_chains`, `recipe_depth`, `round_trip`, `pooling`, `warm_start`; `stitch_fit`, `all_h2`, `t3_t4` (T4 into a PC receiver with `pcB_eta`).
- C-T0 / C-T6: `fourier_cols`, `share_by_k`, `ring_test`, `loo_ring`, `spec_rows`, `subspace_overlap`, `structure_tests`; `snap_task_code`, `trajectory_modes`, `trajectory_basis`, `shuffled_increment_nulls`.
- PC (C-T7, AT5): `pc_eta`, `pc_relax` (chains: columns, inputs, one-hot or soft targets, erasers inside the energy; returns relaxed h1, h2 and the end / start gradient ratio), `pc_relax_inputs`, and `column_search(pc=dict(eta=...))` for every PC search (free, anchored, T4 with soft target, with erasers).

r2_sim.py: `COND`, `COND2`, `SCENARIOS`, `ci`, `fit`, `reg2_design`, `reg2_fit_world`, `reg2_verdict`, `reg2_labels`, `reg2_simulate`, `reg2_noise`; `GRID`, `crossings`, `classify`, `pci`, `tanh_family`, `hill_family`, `first_crossing`, `make_scenarios` / `SCEN`, `NOISE`, `scalar_experiment`, `correlation_loading`, `cells_experiment`, `a2_scenario`, `run_A2_port`, `gain_A2`, `resolvable_band`.

Usage notes for the stage scripts:
- One architecture and one training-set size per `train` call (group by width, kind, K, arc length, ...); worlds, seeds, wd_scale, lr and regularisers can differ within a batch. Nets of a batch stop individually (masked updates).
- `column_search` groups chains by net automatically; held-out sets may be shared ([N]) or per chain ([C, N]); different held-out sizes need separate calls.
- Bounded memory: evaluations are chunked (env R2_CHUNK floats), delta_stats chunks over nets; very wide training batches (W 1024) and T4 at W 1024 with many chains need several GB on the GPU.

## 5 Tests

`QUICK=1 NT=1 OMP_NUM_THREADS=1 python r2_toy_common.py selftest <dir>` (CPU, 1 thread, about 3 min; this container has no GPU, so the
CUDA path is not exercised): 74 / 74 checks pass, and 74 / 74 without QUICK (production constants); `python r2_sim.py selftest` passes.
The output with both sides and the largest differences is `<dir>/ref_comparison_w5.json`. The committed copy for the final code is
`tests/ref_comparison_w5.json`, written by the last QUICK run of this review round (after every code change); it records
r2_toy_common.py 3c141614..., r2_sim.py 69692014..., pc_meta.py b69e314a..., pc_common.py 2d72a474..., the plan 5e476340... and
every reference file's sha256 (all equal to Appendix E). The file's own sha256 changes with every run (it holds timings); the one
committed is stated in the report of this round. For 9.5 it has to be regenerated and committed if either port changes. Main results:
- W0: batched trainer = pc_meta after 300 updates for BP-muP and BP-SP (3 nets each, W 64) and BP-muP with the task regulariser at gamma 0.01: rel. diff 0.0 (bitwise); the plan's early levels (task +-0.003, -0.01; sham +0.003, -0.01 against crit/sham.py's update): 0.0; sham Fisher term = crit/sham.py's to 0 (1e-6 required); first95 / checkpoint / weights = pc_meta under QUICK (subprocess `QUICK=1 NT=1 python pc_meta.py`), and = pc_meta.main with LEVEL1 0.3 (first95 300, a MAXSTEPS net, the K = 1 control); column search = pc_meta.finetune('col') in 24 / 24 chains (required 95 %); fresh F 16 / 16; B2, B2_all and retention = pc_meta's job; delta_stats = pc_meta to 1.5e-13; batched relaxation = pc_meta.pc_relax (0.0), soft-target relaxation = autograd (1.5e-7); PC column search = pc_meta.finetune('col', pc_mup) 8 / 8; erasers inside the PC energy = autograd (2e-5).
- Ports against the refs in round2/ref/: inits of every kind (pc_common.MLP, family2.Net, c3_angle.Net, impose2.Net, rgi_bneck.MLPB) bitwise; code_table ring / rand / pring; training in W3, W5, bottleneck and W4 against the refs' nets and optimiser groups (30 updates, <= 1.4e-5); zero-shot accuracy = c3_angle / impose2; Lambda = crit/sham.lam; anchored search = rgi_lib.learn_alpha (3e-7); t1_map / stitch_fit exact, t1_loo_errors = the ref's per-value errors and t1_choose_r = the LOO minimum (section 3); F line = rgi_lib.pair_transfer (with the corrected chooser); S carriage = rgi_lib.span_transfer; refinement = rgi_pool.learn_alpha_from (3e-6); T3 / T4 = rgi_lib.t34_transfer (2e-7 / 5e-7); recipe chains = recipe_depth.py; ring_test / loo_ring / spec_rows / ridge_probe and structure_tests' nulls = c1_structure exactly; trajectory_modes = c4_traj.modes (variance shares, emergence; the template as the plan); mlp_probe = torch.optim.Adam with default betas (final weights bitwise); late sham run = crit/sham.py train('x_late') bitwise and param_distance = sham.pdist.
- r2_sim: reg2_verdict / reg2_simulate = crit/sim_reg2.verdict2 / run2 (exact dicts, 6 scenarios and COND2); run_A2_port(2) = sim_gain.run_A2 with REPS 2 over all 15 noise settings x 7 scenarios (exact); cells_experiment = scalar_experiment for equal pair counts; the measured correlation is simulated as that correlation (loading sqrt(r)); resolvable band at expected noise (0.715, 1.335) (before the toy_num-1 fix (0.716, 1.350)), close to the theta design's 'G outside [0.75, 1.35]'.
- Internal: best-of-budget, recipe evaluation, keep / remove restriction, late branch = main-loop late run (bitwise), second checkpoint, snapshots, C-T6 cap (residual 4e-5, float32 storage), W6 determinism, class-mean eraser equalises class means, x-span block destroys the sum probe, OD-C2 ladder / bisection (Amendment 2; tolerance at W / 2; closest lambda), OD-C16 labels, save / load with checksums; B2_all of a K = 1 control = pc_meta.main's all8 (acc and retention, difference 0).

## 6 Open points

- GPU: not testable here (CPU-only torch). The code keeps host syncs out of the inner loops (training: one per 50 updates; searches: one per search), sets TF32 off, and keeps the reference inits on the CPU generator as pc_meta does. Bitwise equality with pc_meta is a CPU result; on CUDA (cuBLAS, foreach AdamW in pc_meta) agreement is expected at rounding level only.
- Regularised runs whose task Fisher ratio collapses (e.g. gamma +0.01 from update 1) amplify any rounding difference; bitwise equality with pc_meta holds only with the replica Fisher form, which `fisher='auto'` uses whenever the labels of a batch are shared.
- (Settled by Amendment 2: partial removal at d*, see the section 'Amendment 2' and section 7.)
- C-T0 cost: after the bit-identical restructuring (section 7, toy_num-3) structure_tests needs 1.3 / 1.45 / 1.8 s of one CPU core per net at W 64 / 128 / 256 with the production counts (1000 / 1000 / 500), about 30 s for C-T0's 20 nets, against 10 s in plan 7.2 (r2_t3 about 20 s longer than costed); it runs on the host whatever the GPU.
- Counts not fixed by the plan, chosen here: Lambda's 10 permuted-label recipes (default_rng(4000 + ws)), the S line's one set of 30 permutations per world.

## Amendment 2 (added by the coordinator, 2026-11-13)
`damage_match` now applies partial removal at the found dimension d* (one lambda shared by the draws) whenever the mean drop at d*
overshoots the target by more than 0.02, not only at d* = r0 (plan amendments file, Amendment 2). The r0 case runs the same code as
before; `note` is 'partial removal at d* (Amendment 2)' for d* > r0. The lines above that describe the literal reading (an overshoot
at a larger dimension stays unmatched) are superseded. Self-test check 'OD-C2 damage_match' updated accordingly.

## 7 Review of 2026-11-13: what changed per finding

Findings in `r2_review/fixes/toy_num.md` and `toy_defs.md`; where reviewer and verifier differ the verifier's fix was taken. All public
signatures are unchanged; implementers 6 and 7 need no change in their calls.

- toy_defs-1 (medium, fixed). `t1_choose_r` compared the running summed LOO error with the stored mean (as ref rgi_pilot), so it
  returned 1e-4 almost always. It now picks the grid value with the smallest mean LOO error (ties: the first) and returns
  (lambda, mean error) as before; the per-value errors come from the new helper `t1_loo_errors`, written as the ref's inner loop.
  Plan AT1 governs (9.5); the deviation from the ref is in section 3. Results of r2_rgi_toy.py change (T1 gain and LOO-anchor
  transfer at the chosen lambda; r1 / loo_err1 report the real choice), its call does not. Self-test: (a) the ref's comparison
  applied to `t1_loo_errors` gives the ref's (lambda, value) exactly (8 random cases), (b) `t1_choose_r` = the minimum of the mean
  errors, away from the first grid value in 8 of 8 cases (the ref returns 1e-4 in all; both sides in the JSON), (c) rgi_lib's
  pair_transfer / span_transfer run with the corrected chooser in their namespace and stay exact; the library's f_line uses it.
- toy_num-1 (medium, fixed as the verifier proposed). `gain_A2` and `resolvable_band` convert the measured correlation to sim_gain's
  loading, `correlation_loading(rho)` = sqrt(min(max(rho, 0), 1)), and pass that to `cells_experiment`; its docstring now calls its
  argument the loading (cross-cell correlation rho^2 at h = 0). Negative r -> 0, r > 1 -> 1 (section 2). scalar_experiment and
  run_A2_port are unchanged (still exact against sim_gain.run_A2). Self-test (r2_sim, also inside the library self-test): simulated
  mean pairwise correlation 0.5976 for a measured 0.6 (40,000 pairs), the conversions, and gain_A2(rho 0.36) = a2_scenario with
  loading 0.6. The band at expected noise moves from (0.716, 1.350) to (0.715, 1.335). r2_theta_verdict.py needs no change.
- toy_num-2 / toy_defs-2 (medium, done). As the last step of this round, after every code change, the QUICK self-test was run once
  more and its `ref_comparison_w5.json` copied to `tests/ref_comparison_w5.json` (module sha256 in section 5). NOTES_w9 item 3
  quotes the old file's sha256, and r2_run_session.py records the file without comparing its module_sha256 with the checkout (the
  reviewers' optional 'stale' flag); both files are not mine and were left for their owners.
- toy_num-3 / toy_defs-6 (low, contained speed-up, every number identical). ring_test computes the permutation-invariant total once
  per call and share_by_k's 8 QRs as one stacked np.linalg.qr; loo_ring computes its invariants once per code (`_loo_prep`: centred
  code, leave-one-out rows, R2 denominators), the feature row of each shift value once by the same scalar expression (`_ring_feats`,
  cached), and solves the 12 leave-one-out systems with one stacked np.linalg.solve (the same LAPACK solve per system); the generator
  order is unchanged. Checks: old vs new structure_tests at production counts on W 64 / 128 / 256 nets give identical JSON, 300
  random loo_ring cases are identical, and the self-test compares ring_test / loo_ring with c1_structure exactly and replays
  structure_tests' nulls with c1_structure's functions on the same stream (exact). CPU per net 1.30 / 1.45 / 1.81 s (before 4.87 /
  4.95 / 5.65 s). Still above plan 7.2's 10 s for the 20 nets (about 30 s, section 6). Batching the normal equations across
  permutations, as proposed, changes the summation order and so the rounding; not done.
- toy_defs-3 (low, fixed; also covers toy_num-4, refuted). OD-C2 says 'matched within 0.02' and Amendment 2 'the closest lambda is
  kept': the terminal return (no dimension up to W / 2 reaches the target) now sets matched = |drop - target| <= 0.02 instead of
  False, and a bisection that does not reach 0.02 returns the evaluated (lambda, drop) closest to the target (starting from lambda 1
  at d*) instead of the last iterate. A bisection that reaches 0.02 stops as before, so matched controls are unchanged. toy_num-4
  was refuted because in the real run 'closest' and 'last' never differ; it is changed anyway because the amendment's text says
  'closest'. Self-test: target 0.31 with drop min(0.3, 0.01 r lambda) -> d* 32, matched; a step-shaped drop with target 0.25 ->
  unmatched with lambda 1 and drop 0.3 (the last iterate had drop 0.0); the coordinator's checks unchanged.
- toy_defs-4 (low, fixed). mlp_probe ran Adam with pc_meta's betas (0.9, 0.98) (the defaults of `_adam_q`); OD-C10 names none and
  c1_structure.mlp_probe uses torch.optim.Adam's (0.9, 0.999). It now passes b2 = 0.999. Self-test: mlp_probe's final weights =
  torch.optim.Adam(lr 1e-2, weight decay 1e-4, default betas) on the same init and data, bitwise on CPU (largest difference 0.0,
  against 0.043 with betas (0.9, 0.98); the accuracy alone could not tell them apart).
- toy_defs-5 (low, fixed). trajectory_modes now uses the plan's template cos(pi i t / T) (C-T6 rule, OD-C13) instead of the ref's
  cos(pi i (t + 0.5) / T). Self-test: variance shares and emergence = c4_traj.modes, |corr| = an independent computation with the
  plan's template, and the change against the ref's template is at most 0.021 at T = 40.
- toy_defs-7 (low, fixed). b2_all's retention now uses the net's own pretraining tasks from its spec (pc_meta's Xva; one task for a
  K = 1 control; the 12 world tasks for nets without a spec), evaluated in groups of equal set size. Self-test: B2_all of a K = 1
  control = the pc_meta.main K = 1 job's all8 (acc and retention, largest difference 0.0). r2_theta_toy.py uses its own
  finetune_all and is not affected; TC.b2(with_all=True) is now right for K = 1 nets as well.
