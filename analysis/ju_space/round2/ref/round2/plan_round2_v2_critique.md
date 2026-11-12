# Adversarial review of the revised round-2 plan (plan_round2_v2.md)

Read in full: plan_round2_v2.md, its predecessor plan_round2.md and the first critique plan_round2_critique.md; the three designs
with their critiques and final recommendations (rgi_design, theta_design, unlearn_design design_readable.txt); the frozen round-1
plans abschnitt11_rsi_kennwerte_plan.md and abschnitt12_pc_plan.md; the round-1 code rk_loop.py, rk_verdict.py, dn_delta.py,
pc_meta.py, pc_common.py, pc_summary.py, pc_analyze.py, ufo_common.py, ufo_pool.py, rsi_statements.py, gpu/run_shard.py (HEAD
740bfc0, working tree clean), the round-1 notebook builder, gpu/data (rk_dirs.pt, corpus files), the Qwen3-1.7B config, and the
reference scripts of Appendix E (theta_design/crit_bulk.py, unlearn_design/impose*.py, critique/c2_pring.py and their imports).
No round-1 output was accessed: no Drive tool, nothing named dsse_runs, no summary, pilot, loop, K5, K6, double-negative or PC
summary file, nothing under results/ newer than 740bfc0 except plans. No model was run. The numerical checks quoted below were
small inline numpy runs on CPU cores 2-3; no file was written except this one.

Findings are ordered most severe first. Tags: [1] a first-critique item not resolved as section 12 claims, [2] contradiction,
undefined object, rule unmeetable or met by construction, [3] room to choose a result after seeing round-1 data, [4] dependency on
round-1 files, keys or code that does not hold, [5] time estimate.

Short verdict:
- Most first-critique items are resolved as claimed. The status table at the end quotes the plan for each item.
- Every round-1 file and key named in 2.1 and section 6 exists in the current round-1 code: rk_loop state keys, loops.json rows and
  '_meta', k6.json, summary_b*.json as written by rk_verdict.py, dn_delta.json, blocks_chosen.json, shard_done.json, pc_summary
  rows, the job JSON fields, model file names, and rk_dirs.pt (rand/cov, blocks 4-26, i 0-7).
- The sha256 values of all 38 reference files in Appendix E match the files.
- New or remaining problems:
  - One rule is decided by rounding noise (C-Q1).
  - The freeze cannot be verified, and the round-1 operator already sees results.
  - The extra-block cross-fit does not do what it was introduced for.
  - Three rules leave a choice open or have a degenerate rung.
- No session goes over 25 min. Several line items are off by more than 2x, and r2_q2 lands at about 17.5 min with the plan's margin.

---

## 1. C-Q1: the linear rung is decided by rounding noise [2]

- Section: 5.2 C-Q1 (How step 2, Rule); G42; OD-C17; 12 F10.
- Problem:
  - AUC_2 comes from a "fresh ridge probe" trained on the training fold after "one LEACE erasure of the training fold's class-mean
    difference". After LEACE, the training fold's cross-covariance with the labels is exactly zero. For any invertible whitening W,
    including the ridge-regularised one, delta' = delta - W^+ W delta = 0.
  - The closed-form ridge least-squares weight is (S + r I)^-1 X_c^T y / n, and X_c^T y is proportional to delta' = 0. The fresh
    probe's weight is therefore zero up to float rounding, and AUC_2 is the AUC of a rounding-error direction.
  - The rule decides the linear rung on AUC_2: 'one direction' needs AUC_2 <= 0.60, 'subspace' needs AUC_2 > 0.60. Both rungs are
    set by rounding, and so is the whole linear verdict for sea, which is "representational only".
  - The calibration checks INLP's r* and 'pattern', not AUC_2. A passing calibration (r* = 2 for the planted 2-dim signal) says
    nothing about the statistic that decides 'subspace'. Under the rule, that plant would be classified at random.
  - The plan states the cause itself but keeps AUC_2 in the rule.
- Evidence:
  - Plan, step 2: "(2) one LEACE erasure of the training fold's class-mean difference (whitening with the ridge-regularised
    training-fold covariance), applied to both folds, then a fresh ridge probe: AUC_2".
  - Plan, rule: "linear rung 'one direction' = AUC_2 <= 0.60 and the matching ablation removes >= 50 % of the instructed gap ...;
    'subspace' = AUC_2 > 0.60 and r* in 2..8".
  - Plan, calibration: "Rule: valid only if the calibration returns r* = 2 for (i) in >= 80 % and 'pattern' for (ii) in >= 80 % of
    replicates".
  - Plan, stated cause: "With a least-squares probe a single LEACE erasure removes all linear label information in-sample".
  - Inline numpy check: D 64; two folds of 64 texts x 20 tokens; a planted 2-dim signal with token noise along e2; ridge
    1e-2 x tr(S) / D as in C-Q1.
    - AUC before erasure: 0.99.
    - After LEACE, the fresh probe's |w| / |w_0| was <= 3e-13 in 20 of 20 replicates.
    - AUC_2 on the other fold ranged from 0.31 to 0.79, and was above 0.60 in 5 of 20.
- Fix:
  - Keep the probe's training data apart from the eraser's: fit the eraser on fold A and apply it to fold B. Train the fresh probe
    on half of fold B's prompts and test it on the other half, in both assignments. Then swap the folds and average.
  - Or drop AUC_2 and define the rung by INLP: 'one direction' = r* = 1, 'subspace' = r* in 2..8.
  - Either way, calibrate the statistic the rule uses. In G42, the planted 2-dim signal must return 'subspace' and the real sea
    axis 'one direction', each in >= 80 % of replicates.

## 2. The freeze has not happened, its condition cannot be checked, and the round-1 operator already sees results [3]

- Section: header ("becomes final when it is committed"); section 8 preamble; section 13; G1.
- Problem:
  - v2 is still in the session scratchpad and can change until the commit. Its condition is that "the commit is made before anyone
    working on round 2 opens, searches for or requests any round-1 output". Nothing checks this. G1 checks only the sha256 and that
    the freeze commit is an ancestor of the round-2 code commit.
  - The round-1 notebooks stream the runner's output to the operator's screen. That output includes, for every loop, the
    per-round c, p, Q, rel and A, plus the pilot gates, delta per block, the chosen blocks, the K6 numbers and the block-14 verdict
    string. The same text is synced to Drive as dsse_runs/gpu*/logs/*.log.
  - The person who will run round 2 is 'working on round 2'. By the time of the commit they will probably have seen this output,
    so the condition as worded is likely already false, and nothing would show it.
  - Several section-8 quantities sit next to thresholds that can still be edited before the commit. Examples: G-blk, the C-Q4b
    gate, and the loops' p and A that enter A6 and B-Q0.
- Evidence:
  - The notebook builder's main cell: `for line in p.stdout: print(line, end="")`.
  - run_shard.run prints every line of every step except 'Loading weights'.
  - rk_loop.run_loop logs "{loop} d{n+1}: c ... p ... Q ... rel ... A ...", and main logs "pilot gates ...".
  - dn_delta prints delta per block. run_shard.delta prints "blocks chosen:". `rk_verdict.py block` prints the verdict string.
- Fix:
  - Commit v2 and the Appendix E files now, since v2 was written without round-1 output, and record the commit time.
  - Reword section 13 so that the condition concerns the plan's authors and implementers. State that the operator's view of
    round-1 notebook output is not controlled. Any change after the freeze commit is an amendment that lists what was seen on
    screen, if anything.
  - r2_run_session.py records in every manifest the Drive modification times of the round-1 files it copies and of
    dsse_runs/gpu*/logs/*.log (metadata only), next to the freeze commit time. A reader can then see whether the freeze predates
    the first round-1 summary.
  - A code change after the freeze that alters what any rule returns on some input is an amendment, not a 'crash fix' (section 13,
    'Not amendments').

## 3. B-Q0/B-Q1/R3: the fold-based cross-fit at the extra blocks keeps the bulk bias, and R3 compares different estimators [2]

- Section: 4.1 B-Q0 (cross-fitted modal gain, switch); B-Q1 (switch case); R3; OD-B2, OD-B12; G18.
- Problem:
  - Why the cross-fit exists: the theta critique introduced it (crit_bulk.py) to remove the bias of a bulk self-response of J. The
    push comes from one loop, and the read-back is projected on the mode estimate of an independent loop.
  - At block 14, A and B are the independent loops RSI_a and RSI_b, as intended.
  - At the extra blocks, A and B are the two prompt folds of RSI_a's evaluation n = 3. Both folds respond to the same push d_3, so
    u*_A and u*_B share the bulk response to d_3. That shared part is deterministic and does not average out.
  - The fold-based lambda_cf and the switch-case cell gain therefore stay a mixture of the modal gain and the bulk gain.
  - Consequence for the switch: at an extra block it can fire because of the bulk term, and it then swaps one biased gain for a
    differently biased one.
  - Consequence for R3: Gamma at block 14 comes from independent loops and Gamma at the extra blocks from folds. The estimator
    differs by block, which can create or hide 'loop gain falls/rises with delta'. On the local CPU delta curve, block 14 is the
    lowest-delta block, so the difference lines up with delta.
- Evidence:
  - Plan B-Q0: "at the extra blocks the round-1 prompt folds of RSI_a (OD-B2: A = fold True, B = fold False; u*_A, u*_B = unit of
    the mean delta_j of evaluation n = 3 over that fold's pairs)" and "d_n^A = ... the common push d_n at the extra blocks".
  - Plan R3: "Gamma_b = decision gain at alpha_0, T 0.7 (B-Q0 free estimate pooled with the alpha_0 T 0.7 cell, inverse-variance)".
  - crit_bulk.py docstring: "push along u*_A from one loop, project the read-back on u*_B from an independent loop".
  - Inline numpy check with crit_bulk.py's planted model:
    - Settings: D 512, 24 bulk directions, spectrum k^-1.2, SNR 0.2, 128 pairs, 4 rounds, 300 replicates. Each replicate pushes
      u*_A in a fresh 128-pair cell, reads along u*_B and divides by cos(u*_A, u*_B).
    - Median cross-fitted gain, fold-based vs independent loops:

      | lambda | c (bulk gain) | fold-based | independent loops |
      |---|---|---|---|
      | 0.8 | 0 | 0.77 | 0.77 |
      | 0.8 | 0.3 | 0.69 | 0.79 |
      | 0.8 | 0.6 | 0.68 | 0.75 |
      | 1.2 | 0 | 1.18 | 1.19 |
      | 1.2 | 0.3 | 1.18 | 1.18 |
      | 1.2 | 0.6 | 1.03 | 1.15 |

    - Without a bulk term the two agree. With one, the fold-based gain is lower by up to 0.12.
- Fix (decide before the freeze), one of:
  - Run a second independent RSI loop at each extra block in round 2, for example rk.run_loop('r2RSIb', u_T_perp, nseed 2,
    4 rounds) with new keys. It costs about 32 generate calls plus reads, about 45 s per block, and r2_q1 has room. Use it as B
    exactly like RSI_b at block 14.
  - Or: no switch at the extra blocks; R3 uses the same non-switched estimator at every block, and the switch-case gains are
    reported descriptively.

## 4. C-Q2/C-Q3/C-Q4a: the axes of the 'cross-fitted' gaps are not fixed [2][3]

- Section: O1; C-Q2 rule (i); C-Q3 (How, Rule); C-Q4a.
- Problem:
  - gap_perp_cross takes its fold axes from the same texts it scores. "Cross-fitted exactly like gap_perp_cross", applied to
    ablated or pushed texts, therefore measures the gap along those texts' own pro-anti direction.
  - C-Q2, by contrast, says "along the concept axis (u_T_perp ...)", and the unlearning final says "the instructed gap along u_T_perp
    falls by at least 50 %".
  - If the ablated or pushed model still separates pro from anti texts along another direction, the two readings diverge. The
    own-axis gap stays near g_0, while the gap along the unablated axes falls.
  - The two readings can give opposite verdicts for 'unlearned', 'suppressed' and the C-Q4a restored share, and the implementer can
    pick one after seeing the texts.
- Evidence:
  - C-Q3: "Instructed gaps are cross-fitted exactly like gap_perp_cross".
  - C-Q2: "the cross-fitted instructed gap along the concept axis (u_T_perp for RSI, u_RGI_perp for RGI) under the push, g_push".
  - rk_loop.axes: `gap_perp_cross = ... dr[~fr] @ unit(perp(A["rsi"]["dA"])), dr[fr] @ unit(perp(A["rsi"]["dB"])) ...`, where dA and
    dB are the fold means of the same texts' differences.
- Fix:
  - Compute g_push, g_abl and g_cue on the fold axes of this session's unpushed, unablated instructed texts: perp(d_A^0) and
    perp(d_B^0) for RSI, and the fold axes of u_RGI_perp for RGI. Every ratio g / g_0 is then taken along one fixed axis.
  - Report the own-axis gap next to it as a re-routing diagnostic.

## 5. C-T4: the ladder's 11-dim rung is degenerate, so the likely outcome has no verdict [1][2]

- Section: 5.1 C-T4 (How (a), Rule 'Ladder'); section 12, item 4.
- Problem:
  - 'Subspace' needs a block that "unlearns while the equal-dimension random within-span control does not unlearn", tried over
    1, 2, 4, 6 PCs and "then the full 11-dim span". The task span is 11-dimensional, so the 11-dim random subspace within it is
    the span itself. At r = 11 the control unlearns whenever the operation does, and the condition fails by construction.
  - The pilot shows the full-span block unlearning. If the top-6 block does not, no r qualifies. The fallback "r* = '> 11' if no
    block unlearns" does not apply either, because a block did unlearn. The most likely outcome therefore has no verdict.
  - "Does not unlearn" is not defined for a control. 'Unlearns' itself contains a condition on the operation's damage-matched
    control.
- Evidence:
  - Plan: "'subspace': r* = the fewest top task PCs (1, 2, 4, 6, then the full 11-dim span) whose block unlearns while the
    equal-dimension random within-span control does not unlearn; r* = '> 11' if no block unlearns".
  - Unlearning design pilot: "Block the 11-dim task span in the layer-1 pre-activation: old accuracy 0.053, against 0.98 for a random
    11-dim block".
- Fix:
  - At r = 11, compare with the isotropic equal-dimension random control and the damage-matched one, not the within-span one. Return
    'subspace (whole task span, r* = 11)' when only the span unlearns.
  - Define "the control does not unlearn" as: the t(4) lower bound of its old-task accuracy is >= 0.9 x base.

## 6. pc_meta reads T, BETA and MAXSTEPS at import time, and nothing sets them for round 2 [4]

- Section: 2.5 W0; 9.1 r2_toy_common.py ("imports pc_meta as M"); C-T7; AT5.
- Problem:
  - At import, pc_meta takes T, BETA and MAXSTEPS from the environment, with defaults T 50, BETA 1 and MAXSTEPS 3,000.
  - pc_meta.lam_max, pc_relax and pc_update read these values, and W0's equivalence tests run "against pc_meta.py".
  - The R1 PC-muP checkpoints were trained with T 20, BETA 0.1 and MAXSTEPS 6,000.
  - The plan never says that these variables are set before the import. If they are not, eta_x = 1 / lambda_max and the
    energy's output term are computed at beta 1. That is a different PC arm, applied silently to C-T7 and to AT5's PC searches,
    and the equivalence tests would confirm the wrong values.
- Evidence:
  - pc_meta.py line 34: `T = int(os.environ.get("T", "50")); BETA[0] = float(os.environ.get("BETA", "1")); MAXSTEPS =
    int(os.environ.get("MAXSTEPS", "3000"))`.
  - run_shard.py: `PC_ENV = dict(T="20", BETA="0.1", LR="1e-3")` and `MAXSTEPS="6000"`.
- Fix:
  - State in W0 that r2_toy_common sets T=20, BETA=0.1, LR=1e-3 and MAXSTEPS=6000 before `import pc_meta`.
  - It asserts M.T == 20, pc_common.BETA[0] == 0.1 and M.MAXSTEPS == 6000.
  - Run the equivalence tests under the same environment.

## 7. C-T7 pairs R1 PC nets with BP nets from a new trainer, and the check that would catch an offset is only a label [2]

- Section: C-T7 ('Paired BP values'); W0; G46; 2.7 C24.
- Problem:
  - The PC side is R1 checkpoints, trained by pc_meta with one net per process.
  - The BP side is round-2 nets from the new batched trainer, in r2_t3 and r2_t4. The equivalence test covers 300 updates of three
    W-64 nets on the CPU. The real runs go up to 6,500 updates on the GPU, with a new checkpoint rule, evaluation cadence and
    batching.
  - A systematic trainer difference enters every per-seed PC - BP difference and can produce a 'PC differs from BP' claim.
  - G46, which compares round-2 BP nets with R1 BP nets, "changes no verdict".
- Evidence:
  - C-T7: "Paired BP values: C-T4b (r2_t3) and the PC-matched twin of C-T5 (r2_t4) for the same width and seed".
  - G46: "label 'reproduction differs' ... (no verdict changes)".
  - run_shard.pc_jobs: R1 holds bp_mup_W{128,256}_s{0-4}_K12_g0_wd1 checkpoints, from gpu5 and the same code as the PC nets.
- Fix, one of:
  - Run C-T7's BP side (T4b and the PC-matched twin) in r2_pc on those R1 BP-muP checkpoints.
  - Or make G46 at BP-muP W 128 and 256 a gate for C-T7 claims: 'not testable (BP trainer differs from round 1)'.

## 8. C-Q2: 'suppressed' combines two push sizes [1][2]

- Section: C-Q2 (a), (b), rule (ii)-(iii); OD-C18.
- Problem:
  - Content (i) and damage (iii) judge the counter-push at alpha_u, 0.05 nats. Behaviour (ii) is judged on K6, that is at 0.03
    nats.
  - The unlearning final defines 'suppressed' as "a counter-push meeting the same thresholds", which means one push.
  - On the RSI statements M0 is about +3.5 (README 10.4, quoted in the first critique). Ue_fin >= 0.5 M0 then asks for an odd part
    of about 1.75 log-odds at the smaller push.
  - A failure can therefore reflect the push size rather than the effect.
- Fix, one of:
  - Read behaviour at alpha_u: Ue_fin and z_fin of -alpha_u unit(u) against the 16 null directions recalibrated to 0.05 nats. This
    takes 16 calibrations of at most 6 damage evaluations plus 16 x 60 log-odds forwards, under 1 min in r2_q1.
  - Or state in the rule that behaviour is read at 0.03 nats and that the verdict spans two push sizes.

## 9. C-Q4b: 're-routed' can be produced by noise [2]

- Section: C-Q4b rule; section 12, F11.
- Problem:
  - v2 defines 're-routed' as: not 'survives', Ue >= 0.25 with its interval above 0, and |cos(d_n, u_RSI)| <= 0.10.
  - At 64 pairs, |d_n| is mostly noise, and noise shrinks the raw cosine. A real but modest projection on u_RSI then meets every
    condition, with no evidence that the loop is carried by another reproducible direction.
  - The unlearning design's 're-routed' required the loop to survive.
- Evidence:
  - Plan: "'re-routed' if not 'survives' and Ue >= 0.25 with interval above 0 and |cos(d_n, u_RSI)| <= 0.10".
  - Unlearning design: "It is 're-routed' if it survives with |cos(d_n, u_RSI)| at most 0.10 while its texts still carry RSI content
    (Ue at least 0.25, interval above 0)".
- Fix: require rel(d_n) >= 0.5 at d_2 and d_3 (a reproducible read-back), and use the debiased share |A_n| < 0.25 on u_T_perp
  instead of the raw cosine.

## 10. Time estimates: several line items are off by more than 2x; no session goes over 25 min [5]

- Section: 7.1 (R-read and the full-forward rate); 7.2 rows r2_q1, r2_q2 and r2_pc.
- Problem and evidence:
  - Batch reads and the NLL pass are compute-bound, not host-bound.
    - From the model config, Qwen3-1.7B has 1.41 B non-embedding parameters: 28 x (12.6 M attention + 37.7 M MLP).
    - A batch of 32 texts of about 70 tokens is therefore about 6.3 TFLOP. With TF32 off, the A100's float32 peak is
      19.5 TFLOP/s.
    - A full forward takes about 0.4-0.5 s, and a read stopped after block 14 about 0.2-0.25 s. The plan has 50 ms and 30 ms.
    - This agrees with the given facts: one short text is about 0.2 TFLOP and takes 15-20 ms, and a batch has 32 times its compute.
  - T4 into PC receivers:
    - There are 1,200 chains: 4 PC receivers x 15 senders x 5 worlds x 4 tasks.
    - Each step is a T-20 relaxation on 289 inputs, about 23 MFLOP per input per step at W 512.
    - That gives 2.0 TFLOP per step for the 300 W-512 chains, about 60 s for 300 steps, and about 80 s over all four widths. The
      plan has 6.6 s.
  - C-Q6 in r2_q2: 288 stimuli, plus capped depth-3 and depth-4 sentences x r 1-3 x patches, with next-token KL. That is about
    10 s, against the planned 2 s.
  - Corrected session work:

    | Session | Work | With the plan's margin | Note |
    |---|---|---|---|
    | r2_q1 | about 6.3 min | 13.4 min | |
    | r2_q2 | about 9 min | about 17.5 min | above the plan's 15-min ceiling, close to the 20-min guard |
    | r2_pc | about 5.2 min | 10.8 min | if all four PC widths pass G-PC |

  - The other sessions are within 1.5x of the plan. None goes over 25 min.
- Fix:
  - Cost reads at about 0.25 s and full passes at about 0.45 s per batch of 32, scaled by tokens and by the stopping block.
  - Cost T4 into PC receivers by FLOP.
  - Combine the cell read and the NLL pass in one full forward with a block hook.
  - If r2_q2 must stay under 15 min, move C-Q5 or C-Q6 from r2_q2 to r2_q1.

## 11. G35's fallback cannot run in the session that first needs G-meta [2][4]

- Section: G35; 7.3 (r2_pc start conditions and copy list); R1-9.
- Problem:
  - The fallback reads: "If the BP-muP W 128 rows are missing from PCS, the same formulas on r2_t3's round-2 BP-muP W 128 nets
    decide".
  - r2_pc is the session that decides which widths run (R1-9: "r2_pc_toy.py (which widths run: r2_pc)"). It starts after r2_t1
    only and copies no r2_t3 output, so it cannot apply the fallback.
- Fix, one of:
  - Add r2_t3's session_done.json to r2_pc's start conditions and its BP-muP W 128 rows to r2_pc's copy list.
  - Or let r2_pc run every PC width and decide G-PC in r2_ana.

## 12. Smaller items

- (a) R3 vs OD-B1. OD-B1 says that at a 'mode not resolved' block, "R1-R3 [are] reported as gains along the RSI_a attractor
  estimate". R3 admits only blocks with "a usable u*_b". Fix: state whether such a block enters R3.
- (b) G17 vs G18. When the m*^2 guard fails and the switch condition also holds, G17 says to push and read along unit(d_4^RSI_a),
  while the switch says to read along u*_B. Fix: state the order, for example guard first and no switch at a 'mode not resolved'
  block.
- (c) AT8, OR across F lines. The 'forces' mapping uses "F GROWS (free T1 gain or Phi)", which adds an OR across two F lines. The
  RGI final says only "F GROWS", and the plan's own header says "no OR across lines". Fix: name one primary F line (Phi), or require
  both.
- (d) A7, the 'drawn to self' label.
  - The plan defines "'drawn to self' = the mirror (c < 0, A <= -0.25)" on u_RGI_perpT.
  - After the O2 merger, that axis is RGI_PRO vs RGI_ANTI orthogonalised to u_T_perp. So c < 0 means 'toward leaving others'
    improvement to their developers', not 'toward self'.
  - Fix: define 'drawn to self' on u_dir (RSI_PRO - RGI_PRO), or rename the mirror 'drawn away from general improvement'.
- (e) R6. log K - log G is undefined when K, G or a bootstrap replicate is <= 0. Fix: 'undecided' if more than 5 % of replicates
  are non-positive; or test K / G in (0.8, 1.25).
- (f) Appendix E.
  - critique/c2_pring.py does `import impose as I` and uses impose.py's code_table and run. impose.py is not listed (impose2.py
    is identical except for the modal-snapping lines; impose.py sha256 353302dcb2f2af8ae9ef41927d2a87053264dd78a97575ac8eaac302ed16d197).
  - The reference scripts import from absolute scratchpad paths (SP + '/ufo_scripts', '/rgi_design', '/unlearn_design'). The
    check "same outputs on the refs' own small examples" (9.5) can therefore only run in this container.
  - Fix: add impose.py, run the ref-vs-port comparisons here at the freeze, and commit their outputs.
- (g) Key names and the state guard.
  - rk.k6_direction stores 'k6_<name>', so C-Q2's K6 keys cannot start with 'r2_' as 2.3 requires. A name that equals an R1 name
    (e.g. 'u_T_perp') silently returns the cached R1 entry.
  - rk_loop's own save() calls bypass "every save asserts the fingerprint".
  - Fix: name new K6 directions 'r2_<dir>' (keys 'k6_r2_<dir>'), amend 2.3, and wrap rk.save.
- (h) C-T7 multiplicity. The rule tests about nine paired differences with no multiplicity control: forward accuracy at h1 and h2,
  relearn_300 and relearn_3000, A_300 at s_4, s_12 and s_24, and A_3000 at s_24. Fix: Holm over the listed set, or name one primary
  per sub-question.
- (i) AT0. Two target ratios can select the same grid gamma, and AT2 would then run duplicate conditions. Fix: use each gamma at
  most once; the target farther from it takes the next-closest passing value.
- (j) "After two notebook runs" (2.1 finality, G4, G5). No round-1 file records this. run_shard overwrites
  pc_missing_<shard>.json and keeps no run counter. Fix: an operator-written r1_finality.json with the run times per shard, recorded
  in the manifest.
- (k) G46. "In >= 4 of 5 seeds" cannot be met for B-T1's 4-seed cells (seeds 0-3). Fix: "in every seed of the cell" for 4-seed
  cells.
- (l) Which quantity is compared (C-Q2 (i) and C-Q3). "With 90 % upper end <= 0.75 g_0 and below the 5th percentile ..." leaves
  open whether the point estimate or the upper end must lie below the random reference's 5th percentile. Fix: say "the 90 % upper
  end".
- (m) C-Q5. "R1 sigma of RSI_a at d_1" could mean row 0 (the push d_0 that produced d_1) or row 1 (the push d_1). The superposition
  and dose tests concern round 1, i.e. row 0. Fix: name the row.
- (n) O8. The pool has 192 texts, not "about 350": the 32 wikitext passages after the first 8 of 40, plus 160 chats. This is
  harmless for time.
- (o) C-Q1 text sets. C-Q1 speaks of "the six O2 text sets (RSI, RGI, sea; pro and anti)", but O2 makes RSI, RGI and TOPIC texts.
  The sea and RSI token states come from R1 texts (O2 (iii)). Fix: say so.
- (p) C-T5 dmr arm. OD-C2 averages 5 draws to choose the matched dimension, but the dmr search uses one block per (seed, task), and
  the plan does not say which draw. Fix: draw index = task index, or 5 chains averaged.
- (q) Arm A. No verdict is defined for the case where arm A stops after round 1 (c_1 inside the reference band). Fix: 'translation
  (as predicted), not un-searched'.
- (r) AT5. T4 into a PC receiver needs a soft-target output term: A's distribution in the energy and in the closed-form g2.
  pc_meta.pc_update takes hard labels. Fix: define the term and add it to the equivalence tests.
- (s) G4. If sum[14] never appears, r2_q1 and r2_q2 never start; only missing b_lo and b_hi summaries are handled. Fix: apply the
  same two-runs rule to block 14, with every block-14 part 'not run'.
- (t) W0. The equivalence tests have no case with the task regulariser (gamma != 0). AT0 and AT2 depend on it, and pc_meta
  implements it. Fix: add a BP-muP case with gamma 0.01 (task), plus a comparison of the sham regulariser with crit/sham.py.

---

## Status of the first critique's items (section 12 claims, checked against the v2 text)

| Item | Status | Plan v2 text |
|---|---|---|
| 1 C-T7 cost | Resolved. Residual line items in findings 10, 7 and 6 | "C-T7 in its own session r2_pc"; "C-T5-PC 40 chains x 13,500 PC steps at 5.1 ms"; "PC runs only the PC protocol (C-T5, C-T7)" |
| 2 C-Q2 'suppressed' | Resolved as proposed, except the mix of push sizes (finding 8) | "'suppressed' ... if all hold: (i) ... g_push, is <= 0.5 g_0 ... (ii) behaviour from the odd part: Ue_fin >= 0.5 M0 and z_fin >= 2.33 ... (iii) damage inside the calibration band" |
| 3 Freeze | Partly. The rule is reworded and the sha256 is asserted, but the plan is not committed and the timing cannot be checked (finding 2) | "This text is final once it is committed ..."; G1 "sha256 equals the constant FROZEN_PLAN_SHA256" |
| 4 Toy ladder | Partly. The top rung is degenerate (finding 5) | "blocks of the top-1, 2, 4 and 6 task PCs (critique 4)"; "Ladder (critique 4): 'one direction' if ..." |
| 5 R1 PC files | Resolved | "Files are found by name across all six shards"; G5 "all six final"; C24 "every BP net is trained in round 2" |
| 6 Turn-2 ids | Resolved | "Input ids by concatenation (critique 6)"; "the number of turn-1 texts that are not round-trip stable ... is reported" |
| 7 B-Q0 switch | Resolved as proposed. The extra-block cross-fit remains (finding 3) | "g_par is already divided by rel_n and is not divided again"; "Computed only if cos(u*_A, u*_B) > 0.3"; "rel(d_n) = R1.loops[b][L][n-1]['rel']" (matches run_loop's row order) |
| 8 C-Q5 mix | Resolved | "push alpha_14 (unit(u_T_perp) + unit(sea)) / sqrt(2) (a unit direction: u_T_perp is orthogonal to sea by construction)". Checked: u_T_perp is projected off span(sports, tone, sea, u_neg) |
| 9 sup-t order | Resolved | "(1) C0 and 'uncomposed' at all 28 blocks with pointwise 95 % intervals; (2) the passing set P; (3) every other per-block bound from the sup-t simultaneous 95 % band over P" |
| 10 C-Q4a | Resolved. The gap axis is open (finding 4) | "g_cue = the cross-fitted gap of the ablated cued texts against C-Q3's ablated anti-instructed texts"; "If none passed: 'not applicable (nothing removed)'"; "priority u_RSI, then the span" |
| 11 Scratchpad refs | Mostly. All 38 sha256 match; impose.py is missing and the import paths are absolute (12f) | "Appendix E lists every referenced script ... copied unchanged into analysis/ju_space/round2/ref/"; r2_sim.py ports reg2 and gain_A2 |
| 12 C-Q1 cost | Cost resolved. The closed-form probe exposes finding 1 | "Every probe is a closed-form ridge least-squares classifier ... all 201 fits in one multi-right-hand-side solve" |
| 13 C-T5 rules | Resolved | "A_free = A_3000(s_0); A_op = A_3000(s_24) (also reported at s_(r*))"; "control A_rand = the rand arm for col and the dmr arm for z1" |
| 14 Toy evaluation | Resolved | W0 "evaluated at its checkpoint ... if it reaches level 1, else at MAXSTEPS (6,000)"; 2.6 "t intervals ... govern every rule"; C-T4 "'shallow' if relearn_300 - A_new_300 >= 0.20" |
| 15 Reserve 3 | Resolved | "only the cell at the extra block whose alpha_0 T 0.7 cell has the higher decision-gain point estimate can enter R1, and only if that gain's lower bound > 0.8" |
| 16 C-Q4b gate | Resolved. Matches rk_verdict's string format | "LIKELY_wh = V_RSI starts with 'likely', or V_RSI starts with 'not decidable (pilot gate' and the text after 'would have read: ' starts with 'likely'" |
| 17 AT4 init | Resolved | "other init (the init-1 W 128 BP-muP nets that AT5 trains in the same session, critique 17)" |
| 18 u_neg nulls | Resolved | O2 (ii): "the 40 NEG pairs re-read ...; 2,000 within-pair sign-flip null axes of the 40 differences (seed 24)" |
| 19 R6 | Resolved. Non-positive K or G remain (12e) | "'transfers' ... if the interval lies inside (-log 1.25, +log 1.25); 'different criteria' if it excludes 0"; "K is defined only if the 90 % interval of u_read . d^(1) excludes 0" |
| 20 PC gates | Resolved. The keys match pc_analyze.gates | G36 "gates['G0_pc_mup']['pass_'], gates['G3']['pc_mup_W<W>']['pass_'], gates['G4'][W]['pass_'], gates['G5'][W]['pass_']" |
| 21 8x speed-up | Resolved | "no session runs model or toy work in parallel processes and no parallel speed-up is counted anywhere" |
| 22(a) z_D, controls | Resolved | "a 1-dim control against the other 15 nulls; the 2-dim control against the 6 pairs i not in {1, 2}" |
| 22(b) g_0 | Resolved | "Rules that compare instructed gaps use g_0 ... generated in the same session" |
| 22(c) A5 series | Resolved | "delta_OBJ is the frame mean"; "r_DiD ... over the blocks in the passing sets of both models" |
| 22(d) Keys | Resolved; the 'r2_' prefix rule conflicts with rk.k6_direction (12g) | "keys keyseed('theta2', b, a, T, p, s)" |
| 22(e) Identical pairs | Resolved | "an identical pair has delta_j = 0 in the divergence window" |
| 22(f) FR | Resolved | "FR ... computed only where z_self >= 3" |
| 22(g) Ends | Resolved | "Ends = the lowest- and highest-delta entering blocks"; "with two entering blocks the middle clause is dropped" |
| 22(h) AT8 validity | Resolved | "a model that fails level 1 ... is dropped from its world's fits and counted" |
| 22(i) Overlap | Resolved | "'Fourier-aligned' if in addition overlap(k) >= 0.8" |
| 22(j) dn_delta import | Resolved | "dn_delta.py is not imported ...; O7 re-implements its text selection verbatim" |
| 22(k) CRN versions | Resolved | "No rule relies on equality with R1 texts"; match rates reported |
| 22(l) u_S assert | Resolved | "flag 'u_S reproduction differs', R1 u_S stays in force" |
| 22(m) GA labels | Resolved | "every statement at b that uses u_RGI_perp ... carries 'RGI axis not resolved'" |
