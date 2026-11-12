# Adversarial review of the merged round-2 plan (plan_round2.md)

Read in full: plan_round2.md; rgi_design, theta_design and unlearn_design design_readable.txt (designs, flaws, FINAL
RECOMMENDATION); round-1 code rk_loop.py, rk_verdict.py, dn_delta.py, pc_meta.py, pc_common.py, pc_summary.py, pc_analyze.py,
ufo_common.py, ufo_pool.py, rsi_statements.py, gpu/run_shard.py (HEAD 462523b and its history), the round-1 notebook builder,
the frozen plans in results/ju_space/plans/, README 10.4, and the scratchpad evidence files cited below (gpu_review/runner2/
percall.py and opcount.py, theta_design/crit_bulk.py, unlearn_design/critique/c3_angle.py). No model was run.

Findings are ordered most severe first. Tags: [1] dropped/changed from a design's final recommendation without saying so,
[2] contradiction, undefined object, rule unmeetable or met by construction, [3] round-1 dependency, [4] GPU time,
[5] room to choose a result after seeing data.

---

## 1. C-T7 (PC) is costed for one of the five C-T5 arms at an impossible per-step rate; r2_gpu6 needs roughly 0.7-4 h, not 15.5 min [4]

- Section: 5.1 C-T7; 7.2 row r2_gpu6.
- Problem:
  - The estimate counts one of the five C-T5 arms and assumes 40 µs per PC relaxation step.
  - One PC column update is 1,217 ATen ops (one kernel launch each), about 58 per relaxation step including the update's
    share. 40 µs would therefore mean about 0.7 µs per op.
  - At the plan's own rate for the BP column search (3-6 µs per op), a PC relaxation step costs 0.2-0.4 ms, 5-9x the
    assumed figure.
  - Re-packing under 7.1/7.4 cannot fix this "without changing any measurement": it would take 3-12 more sessions.
- Evidence:
  - Plan 7.2: "C-T7: 10 PC nets x (T4b + T5 with T 20 relaxation, about 6.3 M relaxation steps per net at about 40 us) = 6.5 min".
  - 4 tasks x 24 rounds x 3,300 steps x 20 = 6.34 M. That is a single arm. C-T5 has five: "Variants: removal in the column space
    only (pure un-searching), and a z1 block of the found directions. Controls: random removals, top task-PC removals,
    damage-matched random removal (A_rand)". C-T7 runs "C-T4b and C-T5 with every operation placed inside the energy ... (pc_meta.pc_update with mask_col, T 20)".
  - gpu_review/runner2/percall.py prints "pc_update col (1217 ops)" (W 64, 8 shots, T 20). opcount.py: "Counts ATen ops (each ~
    one CUDA kernel launch on the GPU)". pc_relax also keeps a float64 energy and gradient log at every step.
  - The plan's BP rate ("column search 0.15 s per 300 steps, 1.0 s per 3,000", from c6_timing, a CPU figure on a loaded
    machine) is about 0.33 ms per BP step, and a BP step is a few dozen to about 100 ops. That is 3-6 µs per op. A 1,217-op
    PC step therefore costs 4-8 ms.
  - Arithmetic (numpy), one arm: 96 searches x 3,300 steps x 4-7.3 ms = 21-39 min per net. Ten nets at the plan's 8x take
    26-48 min.
  - Five arms take 2.2-4.0 h. PC relearning in T4b (12 old tasks x 3,300 steps x 2 layers) adds 7-12 min at 8x.
  - With AT4-AT6 and setup, r2_gpu6 takes about 42-70 min with one arm and about 2.5-4.3 h with five.
- Fix (decide before the freeze):
  - Implement the PC column search vectorised over chains: all tasks x arms x seeds of one width in one batched relaxation,
    with no float64 energy/gradient logging inside searches. The G4 diagnostics are not needed there.
  - Restrict PC C-T5 to the column-space arm plus an equal-count random-removal control: budget 300 per round, 3,000 only at
    r* and round 24.
  - Run C-T7 in its own session(s), costed from opcount.py on the real implementation times the per-op time measured in
    round 1 (gpu1-gpu4 'pc_jobs' seconds divided by the PC update counts in the job JSONs).

## 2. C-Q2: the 'suppressed' rule is not the unlearning final's, uses the criterion the critique rejected, and is partly non-specific or set by calibration noise [1][2]

- Section: 5.2 C-Q2 (How, Rule). Not listed in 2.7 or 10.C.
- Problem:
  - The final calls a counter-push 'suppressed' only if it meets the same thresholds as 'unlearned': instructed gap at most
    0.5 Delta_T, D, damage and foreign axes. The unprompted shift is reported, not required.
  - The plan requires the unprompted shift instead: at least 0.25 Delta_T below the reference texts. The critique rejected
    this criterion as floor-limited.
  - No instructed texts are generated under a counter-push.
  - The plan adds 'M(-u) <= 0.5 M0' as a threshold. Neither design has it; they only report M(+d) and M(-d).
- Evidence:
  - Unlearning final, rules: "A counter-push meeting the same thresholds is 'suppressed'. The shift in unprompted texts is
    reported but not required." Unlearning critique: "If default assistant text sits near the anti pole (likely; the plan's
    sigma will show it), this cannot happen whatever the ablation does."
  - Plan rule: "'suppressed' if the counter side moves RSI content >= 0.25 Delta_T below the reference texts ..., M(-u) <= 0.5
    M0 with K6 z_fin >= 2.33, damage <= 0.05 nats, ...".
  - M is not specific. README 10.4: "Die Log-Odds der passenden Antwort fallen bei beiden Vorzeichen (RSI: +3,50 ohne Eingriff,
    −1,11 bei f = −0,2, +0,37 bei f = +0,3)". Every other persona direction acts on the RSI statements like the RSI direction
    (−3,5 bis −3,2 gegen −3,1).
  - Damage threshold:
    - The content pushes are calibrated with "accept 0.045-0.055" nats, but the rule requires "damage <= 0.05 nats". A
      calibration that lands in (0.05, 0.055] fails the rule by construction.
    - u_RSI's pairs are reused from R1 RSI_a n = 0. Those were pushed at the alpha that rk_loop.set_alpha calibrated on
      unit(u_T) ("damage of +alpha * unit(u_T) = 0.05 nats"), not on u_T_perp, and times ASCALE (1.5 after a GO1 retry). So
      they are not at matched damage.
  - The rule names only "RSI content" and Delta_T. For unit(u_RGI_perp) and u_diff it is undefined: RGI content? Delta_RGI?
    M0 on RGI30?
- Fix:
  - Implement the final. For each of u_RSI, unit(u_RGI_perp), u_diff and ('cov', 14, 3), generate 64 pro and 64 anti
    instructed texts under -alpha * unit(u) at that direction's 0.05-nat alpha (RSI-instructed for u_RSI and u_diff,
    RGI-instructed for u_RGI_perp; keys keyseed('axis', 'rsi', p)), and read them unpushed. Cost: about 512 texts, about
    40 s in r2_gpu1.
  - 'Suppressed' = the C-Q3 thresholds on the instructed gap, in the concept's own gap units.
  - Behaviour from the odd part (Ue_fin with z_fin >= 2.33), not M(-u).
  - Damage condition = inside the calibration band.
  - Report the unprompted counter-side shift only. If a different rule is wanted, list it in 2.7 with the reason.

## 3. The freeze can be passed after round-1 outputs are seen, and several round-2 numbers are functions of round-1 files alone [5]

- Section: header (lines 3-11); 8.5 (r2_manifest.json); 9 OD-G1.
- Problem:
  - The plan lives in the session scratchpad until "Before the first round-2 session this file is copied unchanged", which is
    after round 1 has finished. The only freeze rule is "Nothing in it may be changed after a round-2 number exists".
  - The following can be computed from round-1 files alone, without any round-2 generation, and so are visible in that window:
    - B-Q0 ("R1 inputs: R1.state[b] 'ev_*', 'dir_*', 'ref', 'alpha'; R1.loops[b] rel_n"): u*_b, m*^2 and its guard, the
      rotation cosines, SNR, the even part E, gate p.
    - Gamma_b and lambda_cf (the push/read switch) up to the scale cbar_b, which needs only one forward pass over the 64
      stored references.
    - B-Q2's alignment of u*_b with u_T_perp, sports, tone and sea.
    - C-Q5's SVD.
    - The C-Q4b gate (V_RSI, ATTRACTOR), G-blk and G-desc.
  - Thresholds, pooling choices or the switch rules could be edited after seeing these without breaking the stated rule.
  - r2_manifest.json records hashes of round-1 code, not of this plan.
- Fix:
  - Commit the plan now to results/ju_space/plans/abschnitt11_12_runde2_plan.md, before any round-1 output is downloaded, with
    its sha256.
  - r2_manifest.json records that hash and the commit time; r2_run_session.py asserts it.
  - Reword the rule: "no change after any round-1 output has been read, except packing (7.4)". A later change is a dated
    amendment that lists every round-1 file opened before it.

## 4. C-T4 drops the toy linearity ladder: blocks of the top-1/2/4/6 task PCs and the toy rules 'one direction' / 'subspace' [1]

- Section: 5.1 C-T4 (How (a), Rule); 10.C lists only Fourier-conditional drops.
- Problem: the final keeps T4 "as designed" and changes only the rules it lists. The design's subspace list and its toy
  ladder rules are absent from the plan. The toy can then no longer say whether the stored ability depends on one direction
  or a subspace, which is request (3). C-T5's r* covers only the search, and C-T4b covers only probes.
- Evidence:
  - Unlearning design T4: "Subspaces U: the task span (11 dims); the top-1, 2, 4 and 6 task PCs; ... random isotropic and
    random within-span subspaces of equal dimension; the x-span (16 dims)".
  - Design rule (3), toy: "'One direction' if blocking u1 alone unlearns, by rule 1. 'Subspace': r* = fewest top task PCs whose
    block unlearns where the random control does not."
  - Final: "(a) At z1, as designed: edit, block, and push at half and full block damage" and "4 Rules (only the changes;
    everything else as designed)".
  - Plan C-T4(a) has only edit, the 11-dim span block, the u1 push, reflection and the x-span block. Its rule has no 'one
    direction' or 'subspace' verdict.
- Fix: add blocks of the top-1, 2, 4 and 6 task PCs to C-T4(a), each with equal-dimension and damage-matched random controls,
  and the two toy ladder rules. These need forward accuracy and the sum probe only, under 1 min at 8x for 20 nets.

## 5. R1 PC files are copied from fixed shards of one runner layout, and the start rule admits non-final PC outputs [3]

- Section: 2.1 (R1.pc); 7.2 (wave-1 start rule); 7.3 (r2_gpu5 to r2_gpu9 copy lists).
- Problem 1, layout:
  - The plan assumes the runner layout of commit 4566e95 and later. Earlier commits put PC-muP and BP-SP on gpu6, and
    BP-muP with wd 0 and K 3/6 on gpu5.
  - The round-1 notebooks clone a branch head at run time, so the layout that actually runs is not fixed by the plan. The
    description of round 1 given for this review places PC-muP on gpu5/gpu6.
  - Under the older layout: r2_gpu6 finds no PC-muP files (AT5-PC and C-T7 'not run'), and r2_gpu7 finds no wd-0 jobs (B-T1's
    reuse fails).
- Problem 2, start rule:
  - A wave-1 session may start when "the shard stopped only because some PC jobs were still missing", which "leaves its rk and
    delta files final". It does not leave the PC files final, and those are what the toy sessions copy.
  - A later rerun of that shard changes R1.pc after r2_gpu5 to r2_gpu9 copied it. G-PC(W), G-meta and the reused checkpoints
    then depend on timing.
- Evidence:
  - Plan 7.3, r2_gpu6: "gpu1-gpu4/pc/{jobs, models}/pc_mup_W{512,256,128,64}_s{0-4}_K12_g0_wd1_T20_beta0.1.* (one width per shard)".
    r2_gpu7: "gpu6/pc/{jobs, models}/bp_mup_W{64,256}_s{0-3}_K12_g0_wd0.*".
  - run_shard.py at 4566e95 and later: `pcw = dict(gpu1=512, gpu2=256, gpu3=128, gpu4=64)`.
  - At 8d499c6 and 0dca5db: `if which == "gpu5": ... add("bp_mup", W, s, wd=0.0)` and `else: for W in (512, 256, 128, 64): ... add("pc_mup", W, s)`.
  - notebooks/build_abschnitt11_gpu_notebooks.py: `git clone ... -b BRANCH` with BRANCH = "ccr-6fe02b21-est040".
  - run_shard.pc() exits on missing jobs: "rerun the notebook to retry them".
- Fix:
  - Find R1 PC files by job name across dsse_runs/gpu1 to gpu6 (glob), and record the source shard and each shard's
    code_commit (from shard_done.json or code_commits.txt) in r2_manifest.json.
  - Start the sessions that read R1.pc (r2_gpu5 to r2_gpu9, and the G-meta/G-PC step of r2_gpu10) only when all six PC job
    lists are final: shard_done.json present, or the retry policy exhausted and recorded.

## 6. B-Q8: the turn-2 assert will fail on sampled non-canonical tokenisations [2]

- Section: 4.1 B-Q8.
- Problem:
  - Turn 1 is sampled by Gumbel-max over the full vocabulary, at T 0.15-1.3 and pushes up to 4 alpha_0. A sampled id sequence
    need not be the canonical BPE tokenisation of its decoded text.
  - Building the chat from decoded text and re-tokenising therefore changes the ids of some texts. With 1,024 turn-2 texts,
    the assert is likely to stop r2_gpu3, or to invite an ad-hoc fix during the run.
  - The Qwen3 template also renders a history assistant turn without the empty think block that turn 1 was generated after.
- Evidence: plan: "Assert that the turn-1 token ids appear verbatim in the turn-2 prompt". rk_loop stores raw sampled ids
  (`conts`).
- Fix:
  - Build the turn-2 input ids by concatenation: the ids of the rendered chat up to the assistant header, then the turn-1 ids,
    then the ids of '<|im_end|>\n<|im_start|>user\nContinue.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'.
  - Assert the fixed pieces once, against apply_chat_template on a canonical text.
  - Report how many turn-1 texts are not round-trip stable.

## 7. B-Q0's push/read switch divides by rel twice, uses an unguarded cross-fit, and leaves the rel_n row unstated [2]

- Section: 4.1 B-Q0; 6 (lambda_cf row); R4.
- Problem 1, double division: g_par is defined with 1/rel_n already, but the switch compares lambda_cf with "g_par / rel".
  Taken literally, the comparison value is doubled at rel about 0.5, and the switch to "push along u*_A, read along u*_B"
  fires by notation.
- Problem 2, unguarded cross-fit:
  - lambda_cf divides by cos(unit(d_n^A), u_B) with no floor. The critique's own check only uses the cross-fit when that
    cosine exceeds 0.3. At the extra blocks A and B are 32-prompt folds of one loop, so the cosine can be small.
  - Which n enters lambda_cf is not stated.
- Problem 3, rel_n row: rel_n must be the reliability of d_n, but O3, B-Q0 and A6(a) say only "rel_n ... from R1.loops[b]".
  rk_loop.run_loop appends the stats of the evaluation at d_n, and those describe d_(n+1). Row n therefore belongs to
  d_(n+1), and taking row n is an off-by-one that biases G and A.
- Evidence:
  - Plan: "g_par = unit(d_n) . d_(n+1) / (alpha_0 cbar_b rel_n)"; "if |lambda_cf - g_par / rel| > 0.15 the cells push along u*_A
    and read along u*_B"; R4: "lambda_cf replaces g_par / rel".
  - theta_design/crit_bulk.py: `if cAB > 0.3:  # cross-fit: push u*_A, read along u*_B`.
  - rk_loop.run_loop: `R.append(s)` (row n = stats of d_(n+1)). rk_verdict: `rel = lambda loop, n: L[loop][n - 1]["rel"]`.
- Fix:
  - Compare |lambda_cf - g_par|, with g_par as defined.
  - Compute lambda_cf from n = 2 and 3 pooled, and only when cos(u*_A, u*_B) > 0.3; otherwise there is no switch.
  - State: rel_n = R1.loops[b][L][n-1]['rel'].

## 8. C-Q5 superposition pushes unit(u_T_perp + sea) of raw axes with unequal norms [2]

- Section: 5.2 C-Q5.
- Problem:
  - In rk_loop, u_T_perp and sea are raw mean differences of different lengths, so the mix is weighted by their norms.
  - The comparison (F(u_T_perp) + F(sea)) / sqrt(2) and the coefficient band [0.5, 1.5] / sqrt(2) assume an equal-weight mix
    of unit vectors. Elsewhere the plan normalises first (O2: "u_diff = unit(unit(u_T_perp) - unit(u_RGI_perp))").
- Evidence:
  - Plan: "push unit(u_T_perp + sea) ... against (F(u_T_perp) + F(sea)) / sqrt(2)".
  - rk_loop.axes: `u_T_perp=uTp, ..., sea=A["sea"]["u"]`. `perp_share=float(uTp.norm() / uT.norm())` is logged because the
    norms differ.
- Fix: push alpha * (unit(u_T_perp) + unit(sea)) / sqrt(2). The two are orthogonal by construction of u_T_perp.

## 9. A2/A7: the sup-t band is taken over 'the passing blocks', which C0 (itself a band bound) defines [2][5]

- Section: 3.1 A2, A5, A7.
- Problem:
  - The definition is circular: C0's lower bound depends on the band, the band on the passing set, and the passing set on C0.
  - A5 uses yet another set: "C0 and not uncomposed; at least 12; the longest contiguous run".
  - A sup-t critical value grows with the number of blocks, so the implementer's choice of set decides marginal blocks.
- Evidence: A7: "Per block b (bounds = sup-t simultaneous 95 % bands over the passing blocks): - C0, contrast present: P_cell
  lower bound > 0 ...; otherwise the block is excluded."
- Fix:
  - Evaluate C0 and 'uncomposed' with pointwise 95 % intervals over all 28 blocks.
  - Compute every other per-block bound from the sup-t band over the fixed A5 set.
  - State this order in A7.

## 10. C-Q4a: g_cue undefined, a verdict even when nothing was removed, and an open choice of ablation [2][5]

- Section: 5.2 C-Q4 (a).
- Problem:
  - Only pro-instructed cued texts are generated, so g_cue, a pro-minus-anti gap, has no anti side.
  - The 64 unablated cued texts appear in no formula.
  - With "u_RSI if none passed", g_0 - g_abl is about 0 and the ratio, and so 'shallow' or 'deep', is noise.
  - If both u_RSI and the span pass C-Q3, which one is used is open.
- Evidence: plan: "Restored share = (g_cue - g_abl) / (g_0 - g_abl)"; "with the ablation that passed C-Q3 (u_RSI if none
  passed); 64 texts ablated and 64 unablated".
- Fix:
  - g_cue = the cross-fitted gap of ablated cued pro texts against C-Q3's ablated anti-instructed texts.
  - The unablated cued texts give the cue's ceiling, reported only.
  - Give a verdict only if an ablation passed C-Q3; otherwise 'not applicable (nothing removed)', with the ratio descriptive.
  - Priority when both pass: u_RSI, then the span.

## 11. Frozen definitions point into the session scratchpad, and sim_reg2.py and sim_gain.py are never ported [2][3]

- Section: 2.5 (W2-W5), 3.1 A1, 3.2 AT6-AT7, 4.1 B-Q0, 5.1 C-T0, 8.1, 8.4.
- Problem:
  - The plan defines objects by scratchpad scripts: crit/sim_reg2.py (AT7), theta_design/sim_gain.py A2 (B-Q0 power re-run),
    crit/sham.py (W2), c3_angle.py (W3), c5_twofam.py (W4), impose2.py and c2_pring.py (W5), c1_structure.py (C-T0),
    rgi_long.py and rgi_bneck.py (AT6), crit/tokfact3.py (A1).
  - These files exist only under /tmp/claude-0/.../scratchpad/, which is session-specific.
  - 8.1 ports the toy code into r2_toy_common, but nothing ports sim_reg2.py or sim_gain.py; 8.4 calls them by scratchpad path.
- Fix:
  - Before the freeze, copy every referenced scratchpad script, and the pilot JSONs quoted as expectations, into
    analysis/ju_space/round2/ref/, with sha256 recorded in r2_manifest.json.
  - Port sim_reg2's verdict and simulation, and sim_gain's A2, as library functions.

## 12. C-Q1's cost rests on an unstated probe implementation, and r2_gpu10 can exceed 25 min [4]

- Section: 5.2 C-Q1; 7.2 row r2_gpu10.
- Problem:
  - Per ladder there are about 16 ridge-logistic fits on about 2,500 tokens x 2,048 dimensions, plus 200 degree-2 refits per
    fold on 528 features. There are 103 ladders.
  - Fitted one at a time with an iterative solver, that is about 10-20 s per ladder, 17-34 min in total, and r2_gpu10 grows to
    30-45 min.
- Evidence: plan: "(2) ... a fresh ridge-logistic probe; (3) INLP up to 8 directions; (4) ... a degree-2 probe on the top-32
  PCs, AUC against 200 text-label permutations", calibration "50 replicates each"; 7.2: "C-Q1: 103 ladders x about 2 s (3.5 min)".
- Fix:
  - Specify closed-form ridge (least-squares) probes for INLP and the permutation null, solving all 200 permuted label vectors
    in one multi-right-hand-side solve.
  - Or move the 100 calibration ladders into a wave-2 session of their own.

## 13. C-T5: 'un-searched' and 'partial' have no round, no budget and no usable control for the column-only arm [2][5]

- Section: 5.1 C-T5 (Rule); OD-C2.
- Problem:
  - A_op is not tied to a round (r* or 24) or to a budget (300 or 3,000).
  - Which removed direction is meant (from the 300-step or the 3,000-step search) is not stated.
  - The damage-matched control is degenerate for the column-only arm. OD-C2 matches on the operation's sum-probe drop, and
    this arm leaves the network unchanged, so the drop is 0.
- Evidence: rule: "r* = rounds until A <= 0.11 (best-of-budget, 3,000 steps); ... 'un-searched' if A_op's upper bound <= 0.11
  while A_rand stays >= 0.8 x A_free; 'partial' if A_op <= 0.5 x A_rand". OD-C2: "target = the operation's sum-probe drop".
- Fix:
  - A_op = best-of-budget accuracy at 3,000 steps after round 24, also reported at r*. The removed direction is the column at
    the best validation step of the 3,000-step search.
  - For the column-only arm, A_rand = the same number of random column-space removals. For the z1-block arm, keep the OD-C2
    match.

## 14. Toy evaluation points and intervals are left open (C-T2, C-T3, C-T4, all C-T rules) [2][5]

- Section: 2.5 W3/W5; 5.1 header; C-T2, C-T3, C-T4.
- Problem 1, nets that fail level 1: W3 and W5 train "as W1 (... checkpoint rule, MAXSTEPS 6,000)", and pc_meta writes no model
  and no measures when level 1 fails (`if ckpt is None and K > 1: json.dump(rec, ...); return`). Some nets will fail:
  - The critique's arcs "do not finish learning their own arc (0.50-0.54 at L = 3, 0.84-0.91 at L = 6)", and its c3_angle.py
    measured zero-shot at the stopping step (`if s > M.MAXSTEPS and first is None: break`).
  - The pilot's k = 2 codes do not reach level 1 in 3,000 updates.
  - So j*(L), L*, the word-length effect and C-T2's k = 2 contrasts have no defined evaluation point.
- Problem 2, other open choices:
  - C-T2's "alignment effect = ring - pring and geometry effect = pring - rand" names no outcome.
  - C-T4 'shallow' and 'deep' compare relearn with A_new with no budget, although both budgets are measured.
  - 5.1 gives two intervals ("t(4) 95 % intervals and a paired seed bootstrap"), while the rules say only "upper 95 % bound".
- Fix:
  - Evaluate every toy net at its checkpoint if it reaches level 1, else at MAXSTEPS, and record which.
  - C-T2 effects use held-out accuracy at that point.
  - C-T4 relearn and A_new at budget 300, with 3,000 reported.
  - Rules use the t(4) interval; the bootstrap is descriptive.

## 15. Theta reserve 3: the entering condition was changed without saying so [1]

- Section: 4.1 B-Q5; 2.8; 6 (Theta reserve 3).
- Problem: the final runs the 2 alpha_0 cell at the extra block with the highest G(alpha_0) only. The plan runs it at both
  extra blocks and lets each enter on its own condition, so one more block can enter R1. Section 2.8 claims every reserve
  enters "only under its frozen condition".
- Evidence: theta final Q6: "3. 2 alpha_0 at the extra block with the highest G(alpha_0), if its lower bound > 0.8." Plan:
  "reserve 3: 2 alpha_0 at T 0.7 (128) at both blocks, entering R1 at a block only if that block's G(alpha_0) lower bound > 0.8".
- Fix: restore the final's condition (the extra block with the higher G(alpha_0) point estimate), or list the change in 2.8.

## 16. The C-Q4b gate cannot open when a block-14 pilot gate failed, contrary to OD-G5 [2][3]

- Section: 2.1 (LIKELY), 2.2 (G-likely), 5.2 C-Q4 (b), 9 OD-G5.
- Problem:
  - LIKELY = "V_RSI starts with 'likely'". With any failed pilot gate, rk_verdict rewrites the verdict to "not decidable
    (pilot gate ... failed; descriptive only; would have read: ...)".
  - G-likely is then false whatever the loops show, and C-Q4b does not run. OD-G5 says: "When round-1 gates fail, every
    round-2 measurement still runs; its verdicts carry the label of the failed gate."
- Evidence: rk_verdict.block_verdict: `if G["failed"]: verdict = f"not decidable (pilot gate {...} failed; descriptive only;
  would have read: {verdict})"`.
- Fix: define LIKELY on the 'would have read' verdict when a pilot gate failed, and carry the G-desc label.

## 17. AT4 depends on AT2 models that a parallel session produces [2]

- Section: 3.2 AT4; 7.2.
- Problem: AT4 (r2_gpu6) uses "other init (from AT2)", but AT2 runs in r2_gpu5, and "The sessions are independent and can run
  in any order and in parallel."
- Fix: use r2_gpu6's own init-1 W 128 nets, which AT5 trains anyway ("BP-muP W in {64, 128, 256, 1024} with init 0 ... and
  init 1 (new)").

## 18. B-Q2 needs u_neg nulls that round 1 never stored [3][2]

- Section: 4.1 B-Q2.
- Problem: u_neg is in the alignment list, with "null p from 2,000 null axes (R1 sign-flip nulls; O2 nulls for u_RGI_perp; O2
  label-permutation nulls for u_S_perp)". But rk_loop.axes stores only `null_u_T_perp`, `null_sports`, `null_sea` and
  `null_tone`, and O2 builds no NEG nulls.
- Fix: in the O2 job, re-read the 40 NEG pairs at B3 and build 2,000 sign-flip nulls of their differences, or mark u_neg as
  descriptive.

## 19. R6: 'the self-steering threshold transfers' is decided by overlapping intervals, which is met by construction under noise [2]

- Section: 4.1 R6.
- Problem: K is a ratio of two noisy projections at 128 pairs, so overlapping intervals are the default outcome. This was
  inherited from the theta final, but it is still a rule met by construction.
- Evidence: "if the K and G intervals of the same cell overlap, 'the self-steering threshold transfers to the natural loop',
  otherwise 'different criteria'".
- Fix:
  - 'Transfers' only if the paired-bootstrap 90 % interval of log K - log G lies inside +-log 1.25 (equivalence).
  - 'Differ' if the interval excludes 0; otherwise 'undecided'.
  - Guard K when the turn-1 projection's interval contains 0.

## 20. The PC gates are looser than in both designs [1]

- Section: 2.2 G-PC(W); AT5; C-T7.
- Problem:
  - The RGI final admits PC "if it passes G1-G5". The unlearning design's T7 used "Only on PC-muP checkpoints that pass G4,
    G5 and level 1". The final changes T7's PC settings (beta 0.1, T 20, truncated inference) but not that gate.
  - The plan admits a PC width on G4 plus level 1 alone. G5-failing widths still run (labelled 'BP-equivalent'), and
    G0_pc_mup and G3 are never checked.
- Evidence: plan: "G4[W].pass_ (stability) and >= 4 of 5 seeds pass level 1; 'BP-distinct' if G5[W].pass_ as well".
- Fix: require G0_pc_mup, G3 and G5 for PC claims, or list the loosening in 2.7.

## 21. The 8x speed-up for 10 processes on one GPU is assumed, not measured [4]

- Section: 7.1; 7.2 rows r2_gpu5 to r2_gpu9.
- Problem:
  - Each toy process is launch-bound and synchronises every step (pc_meta.bp_update `return float(loss)`; pc_relax `.cpu()`).
    Ten CUDA contexts time-share one A100 without MPS.
  - At an effective 4x, r2_gpu9 takes 26 min (5,520 s / 4 + 3 min setup), and r2_gpu5 and r2_gpu8 approach 20 min.
- Evidence: "10 processes in parallel (NPAR 10, as round 1), counted as an 8x speed-up".
- Fix:
  - Before packing, compute the effective factor from the round-1 gpu5/gpu6 'pc_jobs' times (shard_done.json) against the
    per-job 'seconds' in the job JSONs.
  - Start the MPS daemon if Colab allows it.
  - Pre-split r2_gpu9 by width if the factor is below 6.

## 22. Smaller items

Each item gives the section, the problem and the fix.

- (a) C-Q3, "adjusted z with 15 (7) df": undefined. K6 adjusts by an OLS on A_fin (14 df); 15 df suggests a plain z against
  the 16 nulls. Also, the controls ('rand', 14, 2) and ('cov', 14, 0-2) are themselves members of the 16 K6 null directions
  (rk_loop.k6: rand 0-7, then cov 0-7). Fix: state the formula, and leave a control out of the null set it is compared with.
- (b) C-Q3: the rule compares the ablated gap with R1's Delta_T, although "the unablated references are regenerated in the same
  session". Fix: compare with that session's g_0, and report Delta_T. GPU or version drift then cannot decide the rule.
- (c) A5:
  - delta_OBJ is "per frame" (two series), and delta_MMLU has a last-token and a mean reading. V1's "Spearman(dFG,
    delta_OBJ) > 0", D and the Spearman on delta_MMLU name neither.
  - The block set of r_DiD is not given.
  - Fix: frame-mean delta_OBJ; last-token delta_MMLU for the '.' reading (the RGI design's pairing); r_DiD over the blocks
    that pass in both models.
- (d) Keys: turn 2 after the alpha_0 cell and after the 2 alpha_0 cell, both at T 0.7, use the same ('theta2', 14, 0.7, p, s).
  Section 2.3's "every new key starts with 'r2'" contradicts its own example 'ev_r2GEN_0'. Fix: list the shared turn-2 keys
  as a CRN exception or add the push multiple to the key, and reword 2.3.
- (e) B-Q1: the window "tokens after the pair's first divergence" is empty for identical pairs (first_div = -1). Dropping
  them conditions on divergence and inflates G. Fix: set delta_j = 0 for them in that window.
- (f) A3: FR = other-owner mean / SELF mean is unstable when the SELF mean is near 0. Fix: use FR only when the SELF mean has
  z >= 3, or use a difference.
- (g) R3 and 'pull and delta': it is unstated what happens when a block fails G-blk. Fix: state that the ends are the lowest-
  and highest-delta blocks that enter, and that the middle clause is dropped with two blocks, as in rk_verdict.delta_rule.
- (h) AT8, "every included model passes level 1": it is unstated whether a failing model is dropped or invalidates its
  world. Fix: state which.
- (i) C-T0: the critique's threshold "subspace overlap ... of at least 0.8" is measured but not part of the rule. Fix: add it
  or call the overlap descriptive.
- (j) Section 2.1 says round 2 "imports ... dn_delta.py's text selection unchanged", but dn_delta.py runs on import (it loads
  the model from sys.argv), and O7 re-implements the selection. Fix: say so in 2.1.
- (k) Common random numbers with R1 texts (u_dir; C-Q3 and C-Q4a keys) need the same GPU model and torch version as R1:
  CUDA Philox offsets depend on the launch configuration. Fix: r2_gpu1 compares its GPU and torch version with R1's
  shard_done.json and reports the R1-text match rate before relying on the pairing.
- (l) O2: "asserted equal to R1 u_S within 1e-4 relative" stops r2_gpu1 on version drift. Fix: make it a flag.
- (m) GA(b) gates only the GEN loop. C-Q2, C-Q3 and B-Q2 use u_RGI_perp at block 14 even when AUC(u_RGI) < 0.85. Fix: label
  them 'RGI axis not resolved' when GA(14) fails.
