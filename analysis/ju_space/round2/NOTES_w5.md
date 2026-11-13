# NOTES_w5 - implementer 5: r2_toy_common.py and r2_sim.py

Plan: the final, frozen text `results/ju_space/plans/abschnitt11_12_runde2_plan.md` (sha256 5e476340dfdb1d82...; freeze commit
5b46e38c). The drafts were written against v2 and then brought into line with the final text (section 13 'Changes from v2'); the
v2-based parts that changed are listed in section 1. No round-1 output was opened; the tests use pc_meta QUICK runs made here.

Files (only these three were written in `analysis/ju_space/round2/`):
- `r2_toy_common.py` - toy library W0-W8 (import only; `QUICK=1 NT=1 python r2_toy_common.py selftest [<work_dir>]`).
- `r2_sim.py` - ports of `crit/sim_reg2.py` (with `sim_reg.py`'s COND and ci) and of `theta_design/sim_gain.py` run_A2 (import only;
  `python r2_sim.py selftest`).
- this file.

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
- OD-C10 MLP probe = c1_structure.mlp_probe with OD-C10's values: 64 tanh units, Adam lr 1e-2 with L2 weight decay 1e-4, 300 full-batch steps, standardised inputs (torch std, ddof 1, as c1), nn.Linear-style init from torch.Generator(seed).
- Class-mean erasure: mu and the class means over all inputs given (c1: all 12 x 289); Q = left singular vectors of the centred class means with singular value > 1e-10 x the largest (the span of all class-mean differences, rank <= K - 1).
- OD-C2 (literal): ladder r0, 2 r0, 4 r0, ... (<= W / 2); the smallest dimension whose mean drop reaches the target; bisection of lambda in (0, 1] only when the equal-dimension (r0) subspace already exceeds the target by more than 0.02; 'matched' = |drop - target| <= 0.02, so an overshoot at a larger dimension is unmatched (G45 'not testable'). A target below 0.02 is matched by the r0 subspace.
- Lambda: unseen shifts = the 4 new shifts (held-out = split positions 8-288 of their slot, 281 pairs) and the spare shift (all 289 pairs); nulls: 'wrong' = every valid triple aimed at another shift, evaluated on c_new; 'perm' = 10 permutations of the anchors' shift labels from default_rng(4000 + ws) (count and seed not in the plan). `targets_of` lets a caller drop the spare shift for a net that pretrained on it.
- S line: one set of 30 permutations per world from default_rng(31 + ws) (crit/sham_s draws them sequentially per pair and task). X is left to the caller (learn_alpha and b2_A return the accuracies).
- F line: Phi uses the plan's linear span of the 12 anchor columns (the refs project on the affine hull mean + span of the centred anchors); Phi = mean over the 4 tasks of the projected accuracy / mean free accuracy. T1 shuffles: default_rng(123 + ws), 10, identity excluded; LOO-anchor: 10 maps per anchor from default_rng(1000 k + q) (rgi_lib).
- Recursion, T3 and T4 follow rgi_lib / rgi_pool / recipe_depth exactly (300 steps for T3 / T4; fresh Adam for every refinement). T3 into a PC receiver is not defined by the plan and raises.
- pc_eta: the power iteration's random start comes from torch.manual_seed(seed) chosen by the caller (round 1 used whatever state training left), and all RNG states are restored afterwards.
- C-T6 cap: after every AdamW step the centred part of the 12 task columns' change (decoupled decay included) is projected on the basis; the task-mean change stays free; the Adam moments are not projected.
- AT7 (r2_sim.reg2_noise): with dropped models the s_m^2 / 3 term uses the harmonic mean of the cell sizes (3 for complete data); the world x condition residual comes from an additive OLS fit of the cell means (missing cells allowed); jit pools the within-condition SD of x over the non-natural conditions; the condition named 'nat' is simulated at x = p = 0 (as sim_reg2), k = 1 for condition names starting with 'task'.
- OD-B14 (r2_sim.gain_A2): per-cell SDs replace sim_gain's sqrt(1 / (SNR^2 D_eff)) a^(q/2); unequal pair counts share the first min(n) pairs' heterogeneity and common noise and use the first n of each replicate's Nmax bootstrap indices modulo n (sim_gain_cellN's scheme); 'resolvable band' ends by linear interpolation in log g of the SUB-or-SUPER rate at 0.8 on a 41-point grid over [0.5, 2].

## 3 Deviations (all at rounding level or outside the plan's definitions)

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
- AT1 lines: `lambda_triples`, `lambda_targets`, `lambda_line`; `learn_alpha`, `carry_alpha`, `s_perms`, `s_line`; `ridge_coef`, `t1_map`, `t1_choose_r`, `proj_linspan`, `f_line`; `recipe_chains`, `recipe_depth`, `round_trip`, `pooling`, `warm_start`; `stitch_fit`, `all_h2`, `t3_t4` (T4 into a PC receiver with `pcB_eta`).
- C-T0 / C-T6: `fourier_cols`, `share_by_k`, `ring_test`, `loo_ring`, `spec_rows`, `subspace_overlap`, `structure_tests`; `snap_task_code`, `trajectory_modes`, `trajectory_basis`, `shuffled_increment_nulls`.
- PC (C-T7, AT5): `pc_eta`, `pc_relax` (chains: columns, inputs, one-hot or soft targets, erasers inside the energy; returns relaxed h1, h2 and the end / start gradient ratio), `pc_relax_inputs`, and `column_search(pc=dict(eta=...))` for every PC search (free, anchored, T4 with soft target, with erasers).

r2_sim.py: `COND`, `COND2`, `SCENARIOS`, `ci`, `fit`, `reg2_design`, `reg2_fit_world`, `reg2_verdict`, `reg2_labels`, `reg2_simulate`, `reg2_noise`; `GRID`, `crossings`, `classify`, `pci`, `tanh_family`, `hill_family`, `first_crossing`, `make_scenarios` / `SCEN`, `NOISE`, `scalar_experiment`, `cells_experiment`, `a2_scenario`, `run_A2_port`, `gain_A2`, `resolvable_band`.

Usage notes for the stage scripts:
- One architecture and one training-set size per `train` call (group by width, kind, K, arc length, ...); worlds, seeds, wd_scale, lr and regularisers can differ within a batch. Nets of a batch stop individually (masked updates).
- `column_search` groups chains by net automatically; held-out sets may be shared ([N]) or per chain ([C, N]); different held-out sizes need separate calls.
- Bounded memory: evaluations are chunked (env R2_CHUNK floats), delta_stats chunks over nets; very wide training batches (W 1024) and T4 at W 1024 with many chains need several GB on the GPU.

## 5 Tests

`QUICK=1 NT=1 OMP_NUM_THREADS=1 python r2_toy_common.py selftest <dir>` (CPU, 1 thread, about 2 min; this container has no GPU, so the
CUDA path is not exercised): 70 / 70 checks pass, and 70 / 70 without QUICK (production constants); `python r2_sim.py selftest` passes.
The output with both sides and the largest differences is `<dir>/ref_comparison_w5.json`; the copy made here for the final code is
`/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r2_impl/w5/st_final/ref_comparison_w5.json` (sha256
125432e0...; it records r2_toy_common.py 1d27e483..., r2_sim.py e7284fad..., the plan 5e476340... and every reference file's sha256,
all equal to Appendix E). For 9.5 it has to be regenerated and committed if either port changes. Main results:
- W0: batched trainer = pc_meta after 300 updates for BP-muP and BP-SP (3 nets each, W 64) and BP-muP with the task regulariser at gamma 0.01: rel. diff 0.0 (bitwise); the plan's early levels (task +-0.003, -0.01; sham +0.003, -0.01 against crit/sham.py's update): 0.0; sham Fisher term = crit/sham.py's to 0 (1e-6 required); first95 / checkpoint / weights = pc_meta under QUICK (subprocess `QUICK=1 NT=1 python pc_meta.py`), and = pc_meta.main with LEVEL1 0.3 (first95 300, a MAXSTEPS net, the K = 1 control); column search = pc_meta.finetune('col') in 24 / 24 chains (required 95 %); fresh F 16 / 16; B2, B2_all and retention = pc_meta's job; delta_stats = pc_meta to 1.5e-13; batched relaxation = pc_meta.pc_relax (0.0), soft-target relaxation = autograd (1.5e-7); PC column search = pc_meta.finetune('col', pc_mup) 8 / 8; erasers inside the PC energy = autograd (2e-5).
- Ports against the refs in round2/ref/: inits of every kind (pc_common.MLP, family2.Net, c3_angle.Net, impose2.Net, rgi_bneck.MLPB) bitwise; code_table ring / rand / pring; training in W3, W5, bottleneck and W4 against the refs' nets and optimiser groups (30 updates, <= 1.4e-5); zero-shot accuracy = c3_angle / impose2; Lambda = crit/sham.lam; anchored search = rgi_lib.learn_alpha (3e-7); t1_map / t1_choose_r / stitch_fit exact; F line = rgi_lib.pair_transfer; S carriage = rgi_lib.span_transfer; refinement = rgi_pool.learn_alpha_from (3e-6); T3 / T4 = rgi_lib.t34_transfer (2e-7 / 5e-7); recipe chains = recipe_depth.py; ring_test / loo_ring / spec_rows / ridge_probe = c1_structure exactly; trajectory_modes = c4_traj.modes; late sham run = crit/sham.py train('x_late') bitwise and param_distance = sham.pdist.
- r2_sim: reg2_verdict / reg2_simulate = crit/sim_reg2.verdict2 / run2 (exact dicts, 6 scenarios and COND2); run_A2_port(2) = sim_gain.run_A2 with REPS 2 over all 15 noise settings x 7 scenarios (exact); cells_experiment = scalar_experiment for equal pair counts; resolvable band at expected noise (0.72, 1.35), close to the theta design's 'G outside [0.75, 1.35]'.
- Internal: best-of-budget, recipe evaluation, keep / remove restriction, late branch = main-loop late run (bitwise), second checkpoint, snapshots, C-T6 cap (residual 4e-5, float32 storage), W6 determinism, class-mean eraser equalises class means, x-span block destroys the sum probe, OD-C2 ladder / bisection, OD-C16 labels, save / load with checksums.

## 6 Open points

- GPU: not testable here (CPU-only torch). The code keeps host syncs out of the inner loops (training: one per 50 updates; searches: one per search), sets TF32 off, and keeps the reference inits on the CPU generator as pc_meta does. Bitwise equality with pc_meta is a CPU result; on CUDA (cuBLAS, foreach AdamW in pc_meta) agreement is expected at rounding level only.
- Regularised runs whose task Fisher ratio collapses (e.g. gamma +0.01 from update 1) amplify any rounding difference; bitwise equality with pc_meta holds only with the replica Fisher form, which `fisher='auto'` uses whenever the labels of a batch are shared.
- OD-C2's literal reading can leave controls unmatched when the doubling overshoots by more than 0.02 at a dimension above r0; if that is frequent, an amendment (bisection at the found dimension) would be needed.
- Counts not fixed by the plan, chosen here: Lambda's 10 permuted-label recipes (default_rng(4000 + ws)), the S line's one set of 30 permutations per world.

## Amendment 2 (added by the coordinator, 2026-11-13)
`damage_match` now applies partial removal at the found dimension d* (one lambda shared by the draws) whenever the mean drop at d*
overshoots the target by more than 0.02, not only at d* = r0 (plan amendments file, Amendment 2). The r0 case runs the same code as
before; `note` is 'partial removal at d* (Amendment 2)' for d* > r0. The lines above that describe the literal reading (an overshoot
at a larger dimension stays unmatched) are superseded. Self-test check 'OD-C2 damage_match' updated accordingly.
