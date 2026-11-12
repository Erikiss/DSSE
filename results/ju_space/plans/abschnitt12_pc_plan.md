# Plan v2 (after the critique, before the main runs): intrinsic learning to learn, capacity and delta, with predictive coding

Written 12 November 2026 after the plan critique and literature check (pc_out/review_readable.txt), before any main run. The
critique's decision rule is adopted with the additions for delta that the user asked for (the reformulated question: does the
meta level disappear as delta, the size of the semantic space, grows?).

## Changes from the draft
- World: shift family f_c(x, y) = (x + y + c) mod 17 instead of a x + b y (no multi-task generalisation in 27 BP runs; shifts share
  one rule and are learned). The linear family is reported only as 'no level 1 for K >= 2'.
- Meta measure: B2_emb, only the new task's input column is trained (task-representation learning); all-weights B2 only secondary
  (it ungroks: the pretrained net memorises the shots). Control K = 1 separates a meta level from transfer of the x, y code.
- Arms: BP with muP for Adam (primary), BP with standard parameterisation (artefact check of the width-dependent effective
  learning rate), PC with muP and converged inference. PC only at widths where it is stable and distinct from BP.
- Level 3 dropped (exploratory at most). 5 paired seeds; slope and equivalence tests.
- New: delta per hidden layer, B2 against delta across all cells, and a causal probe that moves delta at fixed width.

## World and arms
- Inputs one-hot x (17), one-hot y (17), one-hot task slot (12 pretraining + 4 new + 1 spare); output 17 classes; two tanh hidden
  layers of width W; cross-entropy.
- Seed s = 0..4 fixes the 12 pretraining shifts and 4 new shifts, each pretraining task's split (144 train / 145 held out), the new
  tasks' nested shot sets (4 within 8), data order and init; seeds paired across widths, algorithms, parameterisations, arms.
- BP-muP (primary): muP for Adam, base width 128 (hidden lr * 128/W, output multiplier 128/W with zero init, input and biases at
  lr); AdamW betas (0.9, 0.98), lr 1e-3, weight decay width-independent (lr * wd per update held constant); full batch.
- BP-SP: PyTorch default init, same lr and wd for all.
- PC-muP: same parameterisation as BP-muP for the predictions; energy as in pc_common.py; T = 50 inference steps with
  eta_x = 1 / lambda_max of the energy Hessian in the value nodes, re-estimated by power iteration every 100 updates;
  learning rate chosen at W = 64 from {3e-4, 1e-3, 3e-3}. Evaluation by feedforward pass.
- 1,500 full-batch updates; runs below the level-1 bar are extended to 3,000 under the same rule for all arms.
- Widths: BP at {64, 256, 1024} (+128, 512 if time allows); PC at {64, 256} (+128 if time allows).

## Measures
- Level 1: held-out accuracy on the pretraining tasks >= 0.95; steps until held-out first reaches 0.95.
- Fine-tuning checkpoint: 500 updates after held-out first reaches 0.95 (runs that never reach it are excluded and counted).
- B2(s, W, arm) = A - F over the 4 new tasks: A = mean held-out accuracy at fine-tuning steps {10, 20, 40, 80, 160, 300} when
  only the new task's input column is trained (Adam lr 1e-2, no weight decay; for PC the local update at equilibrium), the column
  initialised at zero; F = the same mean for a fresh net of that width and arm trained with all weights on the same shots.
  n_shot = 8 primary, 4 secondary. B2_meta = B2(K = 12) - B2(K = 1). All-weights B2 reported only. Retention of the pretraining
  tasks after fine-tuning reported as a covariate.
- delta at the fine-tuning checkpoint, per hidden layer h1, h2, over the 12 pretraining tasks x all 289 pairs:
  F_task = tr(S_between) / tr(S_within) of the task clouds; floor = the same with task labels permuted (mean of 50); delta = F -
  floor; delta_ctrl = the same after subtracting each (x, y) pair's mean over tasks (the analogue of controlling the last token:
  what separates the tasks beyond the input pair). Also the task-cloud geometry: whether the 12 task centroids lie on a circle in
  shift order (correlation of centroid distances with cyclic shift distance).
- PC diagnostics every 100 updates: lambda_max, end/start ratio of |dE/dh| and energy monotonicity per update, PC-BP cosine per
  layer on a fixed batch.

## Gates (failure = not testable for that arm or width)
- G0 muP coordinate check: per-layer activation change after 10 updates varies less than 2x across widths (BP-muP, PC-muP).
- G1 BP-muP at W = 128 passes level 1 in >= 4 of 5 seeds; G2 its mean B2 >= 0.10 with lower 95 % bound (t, df 4) > 0.03.
- G3 the fresh comparator's mean accuracy <= 0.10 at every width.
- G4 per PC width: >= 90 % of logged updates end/start gradient ratio <= 0.05 and >= 99 % non-increasing energy.
- G5 per PC width: median layer-2 PC-BP cosine over the first 20 % of updates <= 0.90 (else 'BP-equivalent').
- If G1 or G2 fails: stop, report 'world has no testable meta level'.

## Statistics and verdicts (adopted from the critique)
- Rule-capable widths: >= 4 of 5 seeds pass level 1; W_lo, W_hi the smallest and largest per arm.
- Per seed: OLS slope of B2 on log2 W over rule-capable widths; mean slope with t(4) 95 % CI and seed bootstrap (10,000).
  D = B2(W_hi) - B2(W_lo) paired. 'Zero' by TOST: upper one-sided 95 % bound of B2(W_hi) < 0.05.
- Per arm: SUPPORTS H if the slope CI lies below 0, B2(W_hi) is zero by TOST, the lower bound of B2(W_lo) > 0.05 and the lower
  bound of B2_meta(W_lo) > 0. CONTRADICTS H if the lower bound of B2(W_hi) > 0.05 and the CI of D lies above -0.10.
  SHRINKS-NOT-VANISHES if the slope CI is below 0 but TOST fails. Otherwise UNDECIDED (report the detectable effect).
- Overall: H supported only if BP-muP says SUPPORTS; a fall only under BP-SP is the effective learning rate, against H; if
  B2_meta(W_lo)'s CI includes 0, any fall is code transfer and H is not tested.
- PC vs BP: per-seed slope difference over common widths passing G4 and G5; claim a difference only if its 95 % CI excludes 0 and
  |mean difference| x log2(W_hi/W_lo) >= 0.05; a gap that closes as the PC-BP cosine approaches 1 is 'PC converging to BP'.

## Delta (the user's reformulation)
- D1, descriptive: B2 against delta (h2, raw and controlled) over all cells (widths x arms x seeds). Per arm, regress B2 on delta
  and on log2 W jointly (seeds as clusters). 'Delta carries the width effect' if adding log2 W to delta changes the fit by less than
  its seed noise and the delta coefficient's CI excludes 0; the sign says which thesis: negative = scarcity (more room, less meta),
  positive = counter-thesis (an organised, spread space is what makes the new task placeable).
- D2, causal probe: BP-muP at W = 256, 5 seeds, a regulariser on the batch Fisher ratio of h2 during pretraining,
  R = gamma * log F_task(h2) (gamma > 0 pulls the task clouds together, gamma < 0 pushes them apart), three arms gamma in
  {-g, 0, +g} with g chosen in a pilot so that delta changes by about a factor 2 while level 1 still passes. Rule: delta moves in
  the intended direction in >= 4 of 5 seeds in both regularised arms; then B2(push) - B2(pull) with the same sign in >= 4 of 5
  seeds counts as 'delta causal for the meta benefit' (negative = scarcity reading, positive = counter-thesis); otherwise
  'no causal effect of delta found'. If delta does not move as intended, 'probe failed'.

## Budget (1 core, measured in the critique's micro-benchmark)
Stage 0 BP calibration ~10 min; Stage 1 PC calibration ~20 min; Stage 2 main ~2.2 h (3 h if PC needs 3,000 updates); delta probe
~20 min; additions (n_shot 4, W 128/512, wd = 0, K in {3, 6}) as time allows. About 3-4 h in total, in parallel with the Qwen runs.

## Amendments before the main runs (12 Nov 2026, after the stage-0/1 calibration; no main-run result seen)
- PC calibration: with T = 50, eta_x = 1 / lambda_max and output precision beta = 1, PC did not learn at W = 64 (training accuracy
  <= 0.28 after 1,500 updates at every learning rate). Grid beta in {1, 0.3, 0.1} x T in {20, 50} x lr in {1e-3, 3e-3}: only
  beta = 0.1, T = 20, lr 1e-3 learned (training 1.0, held-out 0.78 after 1,500 updates). The PC arm runs with these values. Its
  inference does not converge (median end/start gradient ratio 0.76), so the original G4 fails by construction and the arm is
  'PC with truncated inference'. G4 becomes a stability gate: per width, on average over seeds, at most 1 % of updates with an
  energy rise during inference. The convergence numbers are reported.
- MAXSTEPS 6,000 for every arm; the stopping rule (500 updates after the held-out accuracy first reaches 0.95) is unchanged.
- Widths: BP {64, 128, 256, 512, 1024}, PC {64, 128, 256, 512}; 5 seeds; K = 12 and the K = 1 control everywhere.
- D2: g from {0.03, 0.1, 0.3, 1} by a fixed rule on seed 0: among the g whose two regularised seed-0 runs pass level 1, the one
  that minimises max(|log2(delta(-g) / delta(0)) - 1|, |log2(delta(0) / delta(+g)) - 1|) (delta of h2 at the checkpoint). Every g
  runs for every seed; the rule then uses -g, 0, +g for seeds 0-4.
- Additions: K in {3, 6} and weight decay 0 for BP-muP at W in {64, 256, 1024}, 5 seeds.
- Conventions: 'CI' = two-sided 95 % t interval over seeds; 'lower bound' = its lower end; TOST 'zero' = upper one-sided 95 % t
  bound < 0.05. G3 per arm and width (mean fresh accuracy over seeds <= 0.10); a width failing G3 leaves the rule-capable set.
  G0 failing makes the arm not testable. G5 failing marks a PC width 'BP-equivalent' and drops it only from the PC-BP comparison.
- D1 made concrete: per arm OLS B2 ~ 1 + delta + log2 W over all main cells, bootstrap over seeds as clusters (10,000);
  'delta carries the width effect' if the delta coefficient's 95 % interval excludes 0 and the log2 W coefficient's includes 0;
  for delta and delta_ctrl of h2 separately.
- The main runs move to Colab A100 GPUs (float32, TF32 off). The stage-0 CPU jobs are rerun there; nothing from CPU runs enters.
- The analysis is pc_analyze.py on the rows of pc_summary.py, both written before the main runs.
- (Second code review, 12 Nov 2026, before the main runs.) Measurement fixes: the energy monitored for G4 is computed in float64
  and includes the state after the last inference step (float32 rounding of a small cross-entropy registered as rises); the K = 1
  control always trains to its K = 12 run's checkpoint, also beyond MAXSTEPS; job files are written atomically; ring uses Spearman
  with average ranks. The PC inference gradients are written in closed form (identical to autograd up to float32 rounding,
  1.5e-7 relative in the check; fewer GPU kernels). G4 also applies to the K = 1 PC rows: a K = 1 PC row with more than 1 % energy
  rises does not enter B2_meta.
- Analysis fixes: G0 fails on a NaN ratio or fewer than two widths. Overall: BP-muP failing G0 gives 'H not testable'; B2_meta
  with fewer than two seeds gives 'not available' (not 'code transfer'); 'against H (effective learning rate)' only when BP-muP
  says CONTRADICTS H and BP-SP falls. PC vs BP is not testable when PC-muP fails G0. When G1 or G2 fails, D1 and D2 are not
  evaluated. D2: a g whose pilot run has delta at or below its permutation floor is an overshoot and excluded; ties are reported.
  'PC converging to BP': over at least three PC widths passing G4, Spearman(cos_l2, |B2 gap|) <= -0.8 and |gap| < 0.05 at the width
  with the highest cosine. Missing jobs are reported against the job lists of the runner.
- GPU allocation: the 40 PC-muP jobs run on shards gpu1-gpu4 (W 512, 256, 128, 64) after their main work; BP-muP with D2 on gpu5;
  BP-SP, K 3/6 and weight decay 0 on gpu6. Missing jobs are retried once; a shard with jobs still missing exits with an error.
