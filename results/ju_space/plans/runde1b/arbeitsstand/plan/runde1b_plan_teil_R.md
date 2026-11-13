# Round 1b, part R: Repair and replicate (spec, revision 2)

Part R makes the round-1 RSI loop testable and measures it where round 2 could not. It answers the round-1 question ("does RSI content
carry itself through Qwen3-1.7B's own text?") with a repaired null, a positive control (PC) on a topic axis from which only a generic
attractor is removed, a planted-signal sensitivity check that licenses negative verdicts, a damage-normalised push, more prompts, and
minimum detectable effects (MDE) pre-registered for the rules as used (joint, by simulation). It also closes the small-push window that
round 2 left open, measures the loop's linear gain along RSI against the largest gain (lambda_RSI against lambda_max), validates a lens
readout, compares the text loop with a latent loop, measures in-context carry over several answers, and retries erasure at low damage.

Sources: brief_shared.md, design_rules.md, int_gates.md (verified), int_machinery.md (verified), lit_chainlens_tools.md (verified),
AUSWERTUNG.md, runde1b_eingaben/README.md, microlens/README.md, round-2 plan (P2) sections 2.2-2.6, 4.1, 5.2, 6, 7, the round-1 plan
(P1) section 5, rk_loop.py (RL), rk_verdict.py (RVpy), gpu/run_shard.py (RS), round2/*.py (r2_rk, r2_theta, r2_run_session),
theta_verdict.json (TV: blocks.<b>.B_Q0 mode_set / image_cos_d1_u_star / persistence, R4, B_Q7, cells), round2_summary.json (gates
G3, G47). Numbers marked "est." come from the scratch scripts named in R.5 (JSON only, no model).

## R.0 Summary

- **Budget.** 8.53 A100-h in tier 1 (main §4.3; the pilot moved to r1b_P0) + 2.25 A100-h in tier 2 (the three conditional sessions)
  at the round-2 planned rates (P2 7.1) and the 10 min/GB Drive placeholder for copy-in and writes (1.2 h of the total), within the
  12-h envelope. The reduction level (R.5) is fixed in runde1b_freeze.json (main SG7) from the rates the round's pilot r1b_P0 measures
  before the freeze; at half the planned generation speed, RL2 keeps the total at 12.3 h.
- **Order.** Sessions, accounts (every part-R session in account A) and waves as in main §4.2 and §4.4.
- **Expected, and what can come out negative.** Under round-1/2 dynamics the redefined PC is expected to pass only at block 18. At
  block 14 the round-1 SEA loop lies on the generic mode from its first read-back (R2.3). A failed PC no longer blocks the negative RSI
  verdicts: 'unlikely' needs the planted-signal check SENS_1b, while 'likely' and 'oscillation' still need the PC.
- **Negative outcomes (R.6).** 'unlikely (RSI does not persist)' at a block where SENS_1b passes; every small-push gain along the RSI
  axis below 1; lambda_RSI below 1; in-context carry on the RSI axis 'dissipative'; an erasure that holds and is not re-found by the
  loop.
- **Confirmatory claims.** Two families of ten members each, 'for' (F+) and 'against' (F-), are Holm-controlled at family-wise 0.05.
  Everything else is exploratory.

## R.1 Conventions of part R

**Model and units.** Qwen3-1.7B (revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e, shards and tokenizer.json sha256 as in
analysis/ju_space/gpu/data/checksums.json), chat template, thinking off, float32, TF32 off (ufo_common.load_model) unless a
measurement states otherwise (only R5.4 uses bf16). Push = s x a x min(|h_t|, cap_b) x unit(d) at the output of block b from the first
user-content token on (rk's steer hook, RL:102-110). alpha_0(b) = alpha_b = R1.loops[b]['_meta']['alpha'] (0.15 at 14, which is 1.5 x
the calibrated 0.1; 0.10850 at 18; 0.13444 at 6); cap_b = 2 med_b from R1.state[b]['ref'] (med 189.06 / 514.76 / 59.47). Sampling:
Gumbel-max over the full vocabulary at T 0.7 (unless stated), 40 new tokens, the + and - texts of a pair share one key. Reading: the
unsteered model reads plain prompt + continuation, forward stopped after block b, x = mean over continuation tokens (rk.read); delta_j
= (x_j+ - x_j-) / 2; d = mean of delta_j over ok pairs. cbar = the evaluation's own mean of min(|h_t|, cap_b) over continuation tokens
(r2_rk.CbarHook). Damage = teacher-forced KL(unsteered || steered) in nats over the continuation positions of reference texts
(rk.damage formula).

**Symbols (new).**
- P256 = the 1b prompt set; AX-1b(b) = the 1b axes at block b; u*_b = round 2's style mode (r2_q2/theta/b<b>_q0.pt['u_star'],
  fixed).
- perp* axes: u_T^perp*(b) = u_T minus its projection on span(u*_b, u_sports, u_tone, u_sea, u_neg); u_sea^perp*, u_sports^perp*,
  u_tone^perp* likewise (each against u*_b, u_neg and the other three text axes).
- ^1b axes: u_X_perp^1b(b) = u_X minus its projection on span(u_neg and the other three text axes): round 1's construction on 1b
  axes, without u*_b.
- **Verdict axes** u_X^v(b) (X in {T, sea, sports, tone}): u_X^perp*(b) at b in {14, 18}; u_X_perp^1b(6) at b = 6. Frozen from
  round-2 results before any 1b data. Round 2 found a generic, random-attracted mode at blocks 14 and 18 (TV blocks.<b>.R4
  'generic attractor', random_start_debiased_cos RND_1 0.751 / 1.468), but not at block 6 ('unaligned', RND_1 0.0099). There u*_6 is
  the direction that the RSI and SEA loops share with opposite signs (M_6 = RSI_a, SEA (sign -1), r2RSIb), so projecting it out
  would remove persistent content, not a generic style. Both readings (perp* and ^1b) are reported at every block.
- **Leave-loop-out mode** u*_b^(-L), used only for readings of round-1 and replay loops on perp* axes: the unit vector of the
  sign-aligned sum of d_4 (r2_theta.py:220-224, signs from TV mode_set.signs) over M_b without L and its twins. The RSI twins are
  RSI_a, RSI_b and r2RSIb. Remaining sets:
  - block 14: RSI loops {SEA, SPO, RND_1}; SEA {RSI_a, RSI_b, SPO, RND_1}; SPO {RSI_a, RSI_b, SEA, RND_1}; RND_1 {RSI_a, RSI_b,
    SEA, SPO};
  - block 18: RSI loops {RND_1} only; RND_1 {RSI_a, r2RSIb};
  - block 6: RSI loops {SEA} only; SEA {RSI_a, r2RSIb};
  - loops outside M_b (SEA at 18, RND_2 at 14, RND_1 at 6) use u*_b itself, which they did not shape.
  The full-u* reading is printed beside it, labelled 'self-overlapping'. 1b loops never need it: u*_b was built from round-1 loops
  only.
- PUSH-DN = damage-normalised push (R3.1); PUSH-NP = norm-preserving push (R3.2); a_{L,n} = the calibrated push of loop L in round n;
  dmg_sym(d, a) = mean of the + and - push damage on 8 RSI-topic + 8 scene references; SENS_1b(b) = the planted-signal sensitivity
  check (R2.3b).

**Statistics (Qwen convention).**

Prompts are the units. Intervals are 90 % percentile intervals of a joint prompt bootstrap with 2,000 replicates
(r2_rk.prompt_bootstrap(256, B=2000, seed=1001, n_rsi=128), stratified 128 RSI-topic + 128 scene). "Joint" means that each replicate
re-weights, with the same prompt weights, every text of a prompt: its axis texts (so every axis is re-estimated per replicate, u*_b
and u_neg fixed), all its loop pairs (both seeds), and all its dose cells and turns. Halvings for rel are the 20 K2 halvings of
rk_verdict.halvings(256) (seed 3; RSI-topic prompts are indices 0-127). Every call to r2_rk.stats_on_axis and r2_rk.halvings passes
n_prompts = 256: with a block open, stats_on_axis otherwise falls back to rk.NP = 64 (r2_rk.py:558) and computes rel on the 64-prompt
halvings.

Statistic definitions:
- **c_n, A_n (cross-fitted, disattenuated).** Fold axes a_A, a_B = unit(projected mean of the per-prompt differences D_p over
  fold-True / fold-False prompts), projected exactly like the axis (r2_rk.fixed_fold_axes). c_n^cf = (sum over pairs j of
  a_{other fold of p(j)} . delta_j) / (N_ok |d_n|). A_n = c_n^cf / sqrt(rel_n kappa_X), where kappa_X = cos(a_A^X, a_B^X) is the
  cosine between the read axis's two fold axes (floor 0.5, flagged). rel_n comes from the halvings (rk.stats formula, floor 0.05,
  flagged); everything is recomputed in every replicate.
  - Each fold axis has cosine about sqrt(kappa_X) with the population axis, so dividing by sqrt(kappa_X) removes the fold-axis
    attenuation (about 0.97 at 128 prompts per fold, est.).
  - Uncorrected, that attenuation would make the lower-bound rules conservative and the upper-bound rules ('unlikely', 'no first
    link', SPO_t's bound, 'below 1') anti-conservative.
  - The in-sample c_n = cos(d_n, axis) is reported next to it.
- **K1-s (significance, paired sign-flip test over prompts).**
  - b_p = mean over prompt p's ok pairs of a_{other fold}(p) . delta_j; T = mean_p b_p.
  - p_s = (1 + #{k <= 10,000: mean_p eps_kp b_p >= T}) / 10,001, eps Rademacher from np.random.default_rng(2001).
  - One-sided toward the axis's pro pole unless stated; two-sided for RND loops and 'oscillation'. Wilcoxon signed-rank and
    one-sample t p-values are reported next to it.
  - The test is exact under the sharp null that a prompt's + and - texts are exchangeable, but only asymptotic for E[b_p] = 0
    (int_gates). R0.7 checks its size; if that check fails, every part-R rule uses the studentised bootstrap-t p instead (10,000
    prompt replicates, seed 2002; gate RG6).
- **K1-c (magnitude bar, reported only).**
  - 2,000 centred pseudo-axes E_k (D - mean D) / N, projected like the axis, with E from torch.Generator seeds 311 / 312 / 313 / 314
    (rsi / sports / tone / sea).
  - p_c = (1 + #{|cos(d, null_k)| >= |c|}) / 2,001; Q_c = |c| / q99_c.
  - The bar (about 0.22-0.26 in c, N-independent) decides nothing in R2-R7. It replaces round 1's uncentred p only in the R0
    re-readings.
- **S_4^cf, R_4^cf.**
  - S_4^cf = (cos(d_4^A, d_3^B) + cos(d_4^B, d_3^A)) / 2 / sqrt(r_3 r_4) over the 20 halvings, where r_n is the raw split-half cosine
    of d_n.
  - R_4^cf is the same between two independent loops (cos(d_4^{a,A}, d_4^{b,B}), ...).
  - Uncorrected values are reported next to them. d_4 is pushed along the full-sample d_3, so part of d_3's noise legitimately
    persists (int_gates).
- **A_bar, A_bar_abs, E.**
  - A_bar = mean of A_3 and A_4 over RSI_a and RSI_b (signed); A_bar_abs is the same with |A_n| (round 1's definition).
  - E = (A_4^a + A_4^b) / (A_2^a + A_2^b), defined only if the 90 % interval of A_2^a + A_2^b excludes 0. Otherwise E is the code
    'NaN:denominator_interval_includes_0' and every rule that needs E cannot be reached.
  - All three are recomputed in every bootstrap replicate.
- **Bound conditions and point conditions.**
  - Bound conditions use the 90 % interval (one-sided 5 %; the F+ member 'likely' tests A_bar against 0.25 whichever strength is
    reached, R.9 e): the lower and upper bounds of A, A_bar, A_bar_abs, E, SPO_t's |A_4|, the
    PC contrast P5, gains and K.
  - Point conditions are validity conditions, as in round 1: mean rel_2..4 >= 0.5, rel_n >= 0.5, R_4^cf >= 0.5, S_4^cf >= 0.5,
    D4 > 0, and the debiased-cosine labels.
- **Joint MDE convention.** An MDE is the smallest true effect at which a whole verdict function, with all its conditions jointly,
  holds with probability 0.8. It comes from a normal-model simulation (joint_R.py) with these inputs:
  - SD_A per block 0.045 / 0.033 / 0.034 at 512 pairs (mde_R);
  - within-loop row correlation 0.5-0.8;
  - between-loop correlation 0.5, because RSI_a and RSI_b share all 256 prompts and about 46 % of the per-pair noise lies between
    prompts;
  - non-test conditions assumed met.
  R0.7 recalibrates these correlations on round-1 data and on R1's open-loop rows, and reprints the table before R2's analysis. That
  calibration changes no rule.
- **Multiple testing.**
  - Within a block, a verdict is an intersection-union rule (IUT): all its conditions must hold, its p-value is the largest of its
    test components' p-values, and it needs no further adjustment.
  - Across blocks and measurements, confirmatory claims are made only in two families of ten members each, Holm-controlled at
    family-wise 0.05:
    - F+ (for): V_RSI_1b 'likely' (either strength: 'RSI-dominated' or 'substantial RSI component') at 14, 18 and 6; R3.3 'reaches 1 at small push' along u_T^v at 14, 18 and 6; R4.2
      'lambda_RSI >= 1' at 14, 18 and 6; R6.2 'self-sustaining at T 0.7'.
    - F- (against): 'unlikely (RSI does not persist)' at 14, 18 and 6; R3.3 'below 1 at small push' along u_T^v at 14, 18 and 6; R4.2
      'lambda_RSI below 1' at 14, 18 and 6; R6.2 'dissipative at T 0.7'.
  - Component p-values:
    - significance conditions use the sign-flip p_s;
    - a bound condition uses the one-sided bootstrap p, (1 + #{replicates on the wrong side of its threshold}) / 2,001;
    - a best-of-k condition uses k x its smallest cell p (Bonferroni);
    - an all-cells condition uses its largest cell p;
    - a member whose verdict is not reached (gate failed, not decidable) has p = 1.
  - Each member's string carries 'confirmatory (Holm)' or 'exploratory (Holm not met)'. Everything outside F+ and F- is exploratory.
  - The report prints the reached and the Holm-confirmed counts of F+ and F- next to their expected counts under the null: at most
    0.5 reached (10 x 0.05) and at most 0.05 confirmed (family-wise).
  - R0 is post hoc and unadjusted. The analysis prints the number of verdicts computed per measurement.
  - F+ and F- are the round-level confirmatory families CF-R+ and CF-R- (main §2.4). R.1's component p-value rules are the
    round-wide convention for every round-level family.

**Keys (keyseed namespaces; none collides with round 1 or round 2).**
- Registered namespaces (main §2.3 SO-KEYS): every part-R tag starts with 'r1b_' or 'r1b06_'. R's loop names never form a tag equal
  to another R tag (asserted in code). The reuse of round-1 keys in R1, R2.1 and R7 is the stated exception.
- R1:
  - closed-loop replay: round-1 keys unchanged (keyseed(L + SUF, n, p, s, b), 'mirror');
  - open-loop replay: the same round-1 keys, in a separate state copy.
- Axis texts:
  - AX-1b new texts: ('r1b_axis', name, q);
  - the 64 round-1 prompts' axis texts, regenerated once: round-1 keys keyseed('axis', name, p).
- R2 loops: ('r1b_' + L, n, q, s, b). All arms share keys, so arms are paired.
- R3 and R4:
  - dose cells and their unpushed texts: ('r1b_dose', b, q, s);
  - TXT0: ('r1b_free', q, s);
  - R4.3 direct cells: ('r1b_fdtchk', b, which, q, s).
- R5:
  - R5.5 verifier texts: ('r1b_lab', cand, q);
  - R5.4 refit texts: ('r1b_lensfit', q, s).
- R6:
  - turn 1 at T 1.0: ('r1b_k1', b, T, q, s);
  - turns k >= 2: ('r1b_turn', b, T, k, q, s);
  - random-direction carry turns: ('r1b_turn_rnd', 14, k, q, s);
  - replication cell: ('r1b_krep', q, s), ('r1b_krep2', q, s).
- R7:
  - ablated instructed and cued texts reuse their unablated references' keys (a stated common-random-number exception, as in P2
    Appendix D);
  - C-Q4b: ('r1b_Q4B', n, q, 0, 14).
- R8.2: ('r1b06_' + L, n, q, s, 14), ('r1b06_axis', name, q), ('r1b06_ref', q).
- Analysis seeds:
  - SENS outer bootstrap: np.random.default_rng(2101);
  - R4.2 permutation null: torch.Generator seeds 2201 + i (i < 200);
  - P4 sign flips: np.random.default_rng(2003).
- q = P256 index.

**Code (binding).**
- **Frozen round-1 code.** rk_loop.py stays byte-identical. RG3 checks the frozen sha256 of rk_loop.py, rk_verdict.py, ufo_common.py
  and rsi_statements.py at every session start and at every round-1 shard's code_commit.
- **What may import rk_loop.** rk_loop loads the model and registers its hooks at import (RL:33-57, 113). Model-free work (R0,
  analysis) never imports it; it uses r2_rk's functions (r2_rk imports only rk_verdict) plus a new model-free module.
- **R1 replay processes.**
  - The closed-loop stages run as subprocesses exactly as run_shard.rk (RS:64-72): `python rk_loop.py <root>
    <work>/r1b_replay/rk <stage>` with env BLOCK, ASCALE, POOLDIRS and PERSONA.
  - The open-loop driver imports rk_loop as a library into a separate reduced state copy: argv [rk_loop.py, root,
    <work>/r1b_replay_open/rk, 'lib'], as r2_rk.open_block does, without its state guard. It calls rk.evaluate, the function that
    rk.run_loop calls.
- **New modules** live in analysis/ju_space/round1b/:
  - r1b_stats.py: model-free statistics, including SENS, Holm and the cross-fitted eigen-analysis;
  - r1b_rk.py: extra hooks through r2_rk.open_block. It provides the norm-preserving push; the 16-reference RefKL with both signs
    and top-1 change; a per-round calibrated loop driver that carries a_{L,n} into every gain; generation-time capture; and the
    reduced-state openers of R1;
  - r1b_core.py: a function-style core for multi-block work (push, read, damage, scores, soft-token generation, lens predictions).
    It reproduces rk's push at block 14 to 1e-6 relative (code check CC4).
- **No KV-prefix reuse.** All contexts of about 400 tokens or fewer are re-prefilled. An implementer who adds prefix reuse anyway
  must:
  - re-base the push window to the uncached chunk (rk's hook compares chunk-local positions with an absolute rowstart, RL:105;
    CbarHook treats T > 1 as a prefill);
  - crop only layers <= b after a stopped read (cache.layers[l].crop(-L) for l <= b);
  - first pass a cached-vs-recomputed parity check (maximum relative logit difference < 1e-4, Gumbel flips counted).
- **Storage.**
  - Every stored state vector is float32.
  - Per text, the 'all' window (= rk.read) is kept as a full 2048-vector.
  - The 'div' and 'first20' windows are kept only as projections onto the fixed readout set F(b): u_T^v, u_T^perp*, u_T_perp^1b,
    u_sea^v, raw u_sea, u_sports^v, u_tone^v, u_neg, u*_b, and the fold axes of u_T^v and u_sea^v (at most 12 axes).
  - The joint bootstrap re-estimates axes in the 'all' window only. 'div' and 'first20' readings use fixed full-sample axes and are
    labelled 'descriptive (fixed axes)'.
- **Tests.** Every script has a MOCK test (rk_loop's tiny random model), as in round 2.

**No model output before the freeze.** The only pre-freeze GPU work is the round's engineering pilot r1b_P0 (main §4.2), whose P0-R
items are R-P's list, on SO-PILOT material (main §2.3, disjoint from P256), texts discarded, and only timings, memory and Drive
throughput kept.

**Inverse-free rule.** Part R reconstructs causes from effects in exactly one place, R5.5 (what the style mode encodes), which uses a
generate-verify-revise loop.
- Everything else uses forward maps, gradients (backward products), Arnoldi iterations with forward and backward products,
  covariances and projections.
- No matrix inverse, pseudo-inverse or regularised inverse is used to reconstruct a cause.
- Regression fits are forward fits, not reconstructions: R3.3's weighted least squares and R6.2's K_geo each solve 2 x 2 normal
  equations.
- Small eigenproblems are forward computations as well: the 20 x 20 Hessenberg matrix of Arnoldi, the 32 x 32 and 4 x 4 restricted
  problems of R4.2.
- The J-lens is a fitted forward map (averaged Jacobians) and stays allowed; it is used to predict, never to reconstruct.

**Relation to the user's two hypotheses.** Part R tests neither H-DUAL nor H-CHAIN; it tests the round-1 RSI loop. What it hands to
them:
- (i) PUSH-DN, with its warning that damage depends on context length, so part D must recalibrate per length (moot: part D runs no
  push; main C30).
- (ii) The layer profile of the J-lens (R5.1). R5.1's per-block validity (RG10) is attached as the label
  'J-lens not valid at b (R5.1)' to part K's lens-side statements at that block (main §3.3, §5). It gates no K verdict, because K's
  chain lens uses the J-lens only for seeds.
- (iii) The sign-resolved dose response (odd and even parts, R3.3) and the round-1 'oscillation (RSI begets anti-RSI)' verdict
  (R2.4). These are the short-context counterpart of "one entity splitting into two" (a push that produces its own anti-pole). Part R
  makes no long-context claim.

"A front predicts following versus preceding fronts" has no counterpart in part R: R4's read and use directions are directions of one
loop map at one block, not of token fronts.

## R.2 Objects of part R (objects shared across parts are defined in main §2.3)

| Name | What | Made in |
|---|---|---|
| P256 | 256 prompts: the 64 round-1 prompts + 192 new (96 RSI-topic, 96 scene), order [round-1 RSI-topic 0-31, new RSI-topic 32-127, round-1 scene 128-159, new scene 160-255]; fold (round-1 fold for the old prompts; for the new prompts np.random.default_rng(105) halves within type, 48 + 48); K2 halvings for 256; file results/ju_space/plans/runde1b/prompts_1b.json, sha256 in the plan | written before the freeze (R2.1) |
| AX-1b(b) (main SO-AX) | text axes at block b from P256: per-prompt pro/anti reads of the four round-1 instructions (rsi, sports, tone, sea) from one A100-generated text set, u_neg (round-1 NEG pairs), u_T, the ^1b, perp* and verdict axes u_X^v of all four topics, raw u_sea / u_sports / u_tone, fold axes, kappa_X, centred nulls, AUCs; at blocks 14, 18, 6 (R2.1) and, from one all-block read incl. the NEG sentences, at the 8 lens blocks (R5.1) | r1b_R_S2_*, r1b_R_S5 |
| u*_b, u*_b^(-L) (main SO-AX, U*) | round 2's style mode (r2_q2/theta/b<b>_q0.pt['u_star']), copied unchanged with its sha256; the leave-loop-out modes rebuilt from R1 dir_<L><SUF>_4 and b<b>_r2RSIb.pt with TV's signs | round 2 / R0 |
| STAT-1b | r1b_stats.py: centred pseudo-axes, cross-fitted disattenuated K1-s and A, joint prompt bootstrap of axes and pairs, cross-fitted S_4 / R_4, margin rules, SENS plants, size and power plants, the joint-MDE simulator, Holm families, cross-fitted Ritz pairs with permutation nulls, verdict functions with their precedence | R0 |
| PUSH-DN, PUSH-NP (main SO-DMG) | per-direction, per-round calibration to dmg_sym = 0.05 nats (both signs logged, top-1 change), and the norm-preserving control hook; calibration tables per (block, direction) | R3.1, R3.2 |
| TXT0 | 4,096 unsteered texts (256 prompts x 16 seeds, block-independent token ids), their reads at 14 / 18 / 6, FDT scores, the noise covariance Sigma_text(b) | R3.4 / R4.1 |
| DOSE-1b(b) | (+, 0, -) dose cells (texts, reads, word counts) at 14 / 18 / 6 incl. the PUSH-DN dose per direction | R3.3 |
| JLOOP(b) | the loop's local Jacobian (FDT: X, S, cbar, within-prompt centred) with full-sample read (left) and use (right) directions v_max^C, w_max^C, v_S, w_S (computed in r1b_R_S3), cross-fitted rightmost eigenvalues, permutation nulls, lambda table | R4 |
| LENS-J (main SO-JLENS) | Neuronpedia J-lens for Qwen3-1.7B (qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt; 27 fp16 [2048, 2048] maps for blocks 0-26 to block 27, fitted in bf16 on 466 WikiText-103 prompts of 128 tokens; sha256 6fcc79011bd921ffd87612255e2e99950a124fa519470ee44ebaf161c39be9d6) with R5.1's per-block validity; LENS-J1b if the conditional refit runs (tier 2; used by part R only) | R5 |
| WORDSETS | frozen word lists with sha256: W_RSI (30 words, build / eval halves), W_sea (lexical PC list), W_sports, W_tone (no '!'), 20 frequency-matched neutral sets drawn from the frozen neutral pool results/ju_space/plans/runde1b/neutral_pool_1b.txt (628 words, Appendix R.A, sha256 4b3bcaca06f45521a00f12ec2b056a13962bbfdd066ab93867980c9b0be9ebf5) | R2.1, R5 |
| SENS-1b | the planted-signal sensitivity check per block (pass rates, planted rows' readouts) | R2.3b |
| ERASE-LD | the low-damage erasure operator selected in R7.1 (if any), with damage and top-1 profile | R7.1 |
| LABEL-u* | the verified label of the style mode, or an abstention (R5.5) | R5.5 |
| R0-table | round-1 statistics recomputed with centred nulls, paired tests, cross-fitting and leave-loop-out modes (post hoc), and the calibrated joint-MDE table | R0 |
| r1b_core | function-style multi-block core, equivalence-tested against rk_loop | code |

## R.3 Measurements

### R0 Model-free re-analysis of the round-1 raw files (post hoc)

**R0 (one measurement, seven steps R0.1-R0.7)**
- What: do the corrected statistics (centred pseudo-axes, paired sign-flip test, cross-fitted S_4 / R_4, leave-loop-out modes) change
  the round-1 gate states? Are the corrected tests valid (size) and sensitive (power) on round-1 data? R0 also calibrates the power
  model of R2. Every R0 verdict is 'post hoc (R0)'.
- How: no model; rk_loop is never imported. R0 uses r2_rk's load_r1_state / stats_on_axis (n_prompts = 64 here) / prompt_bootstrap /
  halvings, r1b_stats, and the pinned tokenizer only to decode stored token ids. It runs on an A100 runtime because R0.7's nested
  bootstrap needs it. Per block b in {14 (gpu1), 18 (gpu2), 6 (gpu3)}:
  - **R0.1 Reproduction.**
    - Recompute every loops.json row from the stored evaluations with rk.stats's formulas (via r2_rk.stats_on_axis): c, p, Q, rel
      and A on u_T_perp; the 'sea' sub-dict of SEA rows; 'sports_own' of SPO rows.
    - Tolerances: |dc| <= 1e-6, p identical (counts), |dQ| <= 1e-5, |drel| <= 1e-6. Otherwise flag 'R0 reproduction differs' and
      report both.
    - Rebuild E from torch.Generator().manual_seed(11 + i) (rsi 11, sports 12, tone 13, sea 14) as torch.randint(0, 2, (2000,
      kept)) * 2 - 1, with kept = ST['axes']['kept'][name]. Check E @ diff / N against the stored null_* (maximum absolute difference
      <= 1e-5 x max |null|; the rsi nulls after the projection off span(sports, tone, sea, NEG)).
  - **R0.2 K1-c.** Centred pseudo-axes with the same E (E @ (diff - mean diff) / N, projected like the axis); p_c, q99_c and Q_c for
    every loop row on u_T_perp, raw u_sea, sports and tone.
  - **R0.3 K1-s.** Cross-fitted paired test on the same rows (fold = ST['axes']['fold']; fold axes and disattenuation as in R.1),
    with Wilcoxon and t p next to it, and joint prompt-bootstrap intervals of c^cf and A (r2_rk.prompt_bootstrap(64, B=2000,
    seed=1001)).
  - **R0.4 Cross-fitting.**
    - S_4^cf of SEA; R_4^cf of RSI_a / RSI_b (block 14) and of RSI_a / r2RSIb (blocks 18 and 6, round-2 loop).
    - Cross-fitted versions of B-Q0's debiased mode-set cosines (e.g. RND_1 1.468 at block 18).
    - Uncorrected values beside them.
  - **R0.5 Gate re-evaluation.**
    - Five statistic variants V:
      - R1: the round-1 p, as stored;
      - C: p_c <= 0.01 with c > 0;
      - S: p_s <= 0.01;
      - SC: p_s <= 0.01 and Q_c >= 1 (the primary R0 rule, int_gates 3.5);
      - M: the 1b margin rule, p_s <= 0.01 and A's 90 % lower bound >= 0.10.
    - Two PC axes X:
      - raw u_sea (round 1's);
      - u_sea^v(b) built from round-1 axes. At blocks 14 and 18 it is u_sea minus span(u*_b^(-SEA), u_T, sports, tone, u_neg);
        SEA lies outside M_18, so at 18 u*_18 itself is used. The full-u* reading is printed beside it, labelled 'self-overlapping'.
        At block 6 it is u_sea_perp = u_sea minus span(u_T, sports, tone, u_neg).
    - GO1_R0(b, V, X) on SEA d_1 of the pilot at alpha: rel_1 >= 0.5 and V. The pilot row is 'ev_SEA_0' at blocks 14, 18 and 6 (the
      loops' row 0 at 18). The same is computed at 1.5 alpha where it exists ('ev_SEA@1.5_0' at 14 and 6).
    - PC_PASS_R0(b, V, X): SEA rows n = 2, 3, 4 meet V and rel_n >= 0.5, and S_4^cf >= 0.5. For V = R1 the uncorrected S_4 is used,
      and it must reproduce round 1's PC_PASS.
    - At block 14 the four-round loops exist only at 1.5 alpha, so PC_PASS_R0(14) is at 1.5 alpha, while GO1_R0 'at alpha' concerns a
      scale at which no loop ran (int_gates).
    - MANIP_R0 and block 14's would-have-read verdict are reported (rk_verdict.block_verdict logic with p replaced per V; on perp*
      axes with the leave-loop-out mode).
  - **R0.6 Topic specificity and the expectation for R2.3.** On round-1 data:
    - cos(u*_b, u_sea_perp), with u_sea_perp = u_sea minus span(u_T, sports, tone, u_neg) (int_gates' request);
    - cos(u*_b, u_sea^perp*): 0 by construction with the full u*_b (a check), reported with u*_b^(-SEA);
    - the SEA loop's A on u_sea^v (leave-loop-out at 14), and SEA minus the maximum over RND of A_4 on u_sea^v with its bootstrap
      interval;
    - the lexical sea-word difference count(+) - count(-) of every SEA and RND evaluation (W_sea, decoded with the pinned tokenizer)
      and the across-prompt SD of its per-prompt mean over evaluations 1-3. From that SD, P4's MDE (R2.3) is computed and printed
      before R2's analysis;
    - cos(u*_b, mean of R1 'ref' X), which predicts how much PUSH-NP can differ (R3.2).
    - Reprinted next to these, from round 2, are the facts that set the PC expectation of R2.3:
      - SEA belongs to M_14 and M_6; debiased cos(d_4^SEA, d_4^RSI_a) is -1.051 / -1.142, against 0.229 at 18;
      - the image cos(d_1^SEA, u*_b) is -0.930 / 0.118 / -0.834 at 14 / 18 / 6;
      - R4 reads 'generic attractor' at 14 and 18 and 'unaligned' at 6.
  - **R0.7 Size, power and the power model.** The inner bootstrap uses 200 replicates: the full nested bootstrap of rule M would be
    about 6e14 FLOP, with 200 inner replicates about 6e13, minutes on the A100.
    - Size: for every loop row and axis, resample the centred b_p (2,000 draws of N prompts with replacement from b_p - mean b, 2,000
      flips each) and record the rejection rate at nominal 0.01. The placebo axis is u_neg (cos with u*_14 = -0.017), for every RND
      row.
    - Power: plant delta_j' = delta_j + eta unit(axis) into RND_2 (block 14) and RND_1 (blocks 18, 6), on u_T_perp and on raw
      u_sea. eta is set so that the planted c is 0.05, 0.10, ..., 0.40. Record the detection rate of the rules R1, C, S, SC and M
      over 500 prompt-bootstrap replicates; MDE_80 per rule = the smallest planted c detected in >= 80 %.
    - Power model: the joint-bootstrap covariance of A_1..A_4 of round-1 RSI_a and RSI_b gives the within-loop row correlations and
      the between-loop correlation; R1's open-loop rows give a second estimate. joint_R.py is rerun with them and with the 1b SD_A
      (scaled from round 1 by the design factor, mde_R), and the joint MDE table of R2.3 and R2.4 is reprinted. This is a post-hoc
      calibration of the power model and changes no rule.
- Rule (all strings carry 'post hoc (R0)'):
  - Per block: 'GO1_R0 at alpha: pass | fail (components)' and 'PC_PASS_R0: pass | fail (components)' for every (V, X).
  - Reopening (int_gates): 'block 6 reopened (post hoc, rule SC on raw u_sea)' iff GO1_R0(6, SC, raw u_sea, alpha) and
    PC_PASS_R0(6, SC, raw u_sea) pass.
    - Then block 6's RSI_a statistics are re-read (p_s rows, A_bar_abs with interval) as 'R0 post-hoc reading of block 6'.
    - The label 'not topic-specific (SEA does not exceed RND_1)' is added if the 90 % interval of A_4(SEA) - A_4(RND_1) on raw u_sea
      does not lie above 0. At block 6 u*_6 is not generic, so the revision-1 label 'style-confounded' (a test against u*_6) is
      replaced by this RND contrast.
    - Otherwise 'block 6 stays excluded (post hoc)'.
  - Fixed strings:
    - 'block 14 stays excluded: GO3 (damage 0.4055 / 0.2018 nats) is not a null problem';
    - 'block 18 stays excluded: PC fails on rel (0.4376, 0.4877 < 0.5), which no null correction changes';
    - 'delta rule: not decidable (at most one block; post hoc)' in every case.
  - RG6 (decides R2-R7's significance statistic): 'K1-s size ok' if the rejection rate at nominal 0.01 is <= 0.02 in >= 90 % of the
    rows; otherwise 'K1-s size inflated: bootstrap-t used in part R'.
  - README null claim: both directions are counted and printed with the per-row q99 / q99_c ratios, namely rows with p > 0.01 and
    p_c <= 0.01, and rows with p <= 0.01 and p_c > 0.01.
    - 'README null claim confirmed' if the centred null moves some round-1 p from > 0.01 to p_c <= 0.01 on a row where q99 > q99_c;
    - 'not confirmed' if no row moves that way.
- Null / controls: the centred pseudo-axes are the null of K1-c, and the sign flips the null of K1-s. The placebo axis u_neg and the
  centred-b_p resampling check size. RND loops serve as null rows on axes the style mode does not load on.
- Shows - cannot show: whether round 1's gate failures were null-driven (expected only at block 6), and how the tests behave on real
  round-1 noise.
  - It cannot reopen block 14 (GO3 is a damage fact) or block 18 (rel), and it cannot make the delta rule decidable (at most one
    block).
  - The perp* readings of round-1 loops remain partly self-overlapping at block 18, where the leave-loop-out mode of the RSI loops is
    RND_1's d_4 alone. This is stated, not repaired.
  - Post hoc: the round-1 summaries and round-2 results are known.
- MDE: the data are fixed; the empirical MDE_80 of R0.7 is the output. Expected from the analytic model at 64 prompts x 2 seeds
  (int_gates 5): paired test 0.09 / 0.16 / 0.14 in c (blocks 14 / 18 / 6); centred bar 0.22-0.26; SC rule 0.25 / 0.26 / 0.30;
  round-1 null 0.37 / 0.50 / 0.51.
- Inputs (Drive dsse_runs/, found by name across gpu*; shards by round 2's finder: b14 gpu1, b18 gpu2, b6 gpu3):
  - gpu*/rk/b<b>/state.pt:
    - 'ref' (X, conts, med, cap);
    - 'axis_rsi', 'axis_sports', 'axis_tone', 'axis_sea' (Xp, Xa [64, D], conts);
    - 'axes' (u_T, u_T_perp, u_S, u_S_perp, u_neg, sports, tone, sea, null_u_T_perp, null_sports, null_sea, null_tone, auc,
      paired_win, gap_cross, gap_perp_cross, kept, fold);
    - 'alpha' (alpha, damage, tried, in_band, damage_u_T_perp);
    - 'ev_<L><SUF>_<n>' (delta, ok, prompt_of, xp, xm, first_div, conts, damage_of_push, alpha, capped_share) for n = 0..3, with L in
      {SEA, RSI_a, RSI_b, SPO, RND_1, RND_2} at 14 and {SEA, RSI_a, RND_1} at 18 and 6; SUF is '@1.5' at 14 and '' at 18 and 6;
    - 'dir_<L><SUF>_<n>' (n = 1..4); 'dmg_<L><SUF>_<n>';
    - block 14 also 'ev_SEA_0', 'ev_RSI_a_0', 'dir_SEA_1', 'dir_RSI_a_1', 'dmg_SEA_1', 'dmg_RSI_a_1' (pilot at alpha);
    - block 6 also 'ev_SEA@1.5_0', 'ev_RSI_a@1.5_0', 'dir_SEA@1.5_1', 'dmg_SEA@1.5_1' (retry);
    - 'checks' (GO2); 'k5' (rsi / sports / literature: wU_perp, sigma).
  - gpu*/rk/b<b>/loops.json (rows incl. 'sea', 'sports_own'; '_meta'); pilot.json and pilot@1.5.json ('SEA_1'[0]['sea'], 'gates',
    'axes'); gpu*/summary_b<b>.json ['verdict'] (PC_PASS, SEA_S4, gates.source, gates.SEA_1, gates.failed, verdict,
    label_attractor); gpu*/shard_done.json (GPU, torch, transformers, CUDA), code_commits.txt (provenance); gpu2/dn_delta.json.
  - Round 2:
    - r2_q2/theta/b<b>_q0.pt ('u_star', 'uA', 'uB', 'd4', 'cfg') and r2_q2/theta/b<b>_q0.json (mode_set with members and signs,
      m_star2);
    - r2_q2/theta/b18_r2RSIb.pt and b6_r2RSIb.pt ('entries': ev_r2RSIb_n, dir_r2RSIb_n), for the leave-loop-out modes;
    - results/ju_space/runde2_ergebnisse/theta_verdict.json (mode sets, image cosines, R4).
    - In all, about 320 MB of state.pt plus a few MB.
  - Tokenizer: models/qwen3-1.7b/tokenizer.json (sha256 aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4).
- Runs in: r1b_R_S0 (analysis runtime on an A100). 12 A100-min with margin (8 min work), plus setup, copy-in and writes. The same
  session writes the SO-AX package (main §2.3): AX1 at 14, 18, 6 from R1.state[b]['axes']; U* = r2_q2/theta/b<b>_q0.pt['u_star']
  copied with sha256 (canonical); the leave-loop-out modes; SG10's check (u*_b rebuilt from dir_<L><SUF>_4 and the r2RSIb entries
  with TV's mode_set: cos >= 1 - 1e-4 with the file, TV image_cos_d1_u_star reproduced within 1e-4). Exports to dsse_runs/r1b_xfer
  (main §2.2): dmg_ref_R16.json (token ids of REF-R16 = R1.state[14]['ref'] continuations of round-1 prompts 0-7 and 32-39, checked
  against the copies at 18 and 6) and so_ax_ustar.pt (u*_b at 14, 18, 6). Inside the 22.2 min.
- Gates: RG1, RG3 (module hashes), RG4 (inputs).

### R1 Round-1-identical replay

**R1 Replay arm (same code path, prompts, alpha, keys; closed and open loop)**
- What: rerun with the byte-identical rk_loop.py, the same prompts, axes, alpha, cap and starts, and the same Gumbel keys, does round
  1's procedure reproduce round 1's texts and statistics? This anchors the attribution ladder (R2.6), so every later difference is due
  to one named change, not to the run. The open loop separates run or device differences from the loop's own amplification of them.
- How:
  - **Closed-loop replay (as round 1 ran).**
    - Per block, a reduced copy of R1.state[b] that keeps only 'ref', 'axis_*', 'axes' and 'alpha' is written to
      <work>/r1b_replay/rk/b<b>/state.pt. Nothing is served from the round-1 cache: no 'ev_*', 'dir_*', 'dmg_*', 'checks', 'loops',
      'k5*' or 'k6*'.
    - The stages run as subprocesses exactly as run_shard.rk ran them (RS:64-72): `python rk_loop.py <root> <work>/r1b_replay/rk
      <stage>` with env BLOCK, ASCALE, POOLDIRS (rk_dirs.pt: RND starts ('rand', b, 0 / 1), as round 1) and PERSONA. The round-1 run
      scale is forced, not re-decided:
      - block 14: 'pilot' at ASCALE 1, 'pilot' at 1.5, 'loops' at 1.5;
      - block 18: 'pilot' and 'loops' at 1;
      - block 6: 'pilot' at 1, 'pilot' at 1.5, 'loops' at 1.
    - Texts: 4,480 at 14, 2,048 at 18 and 2,432 at 6. At 18 and 6 the pilot's evaluations are the loops' round 0.
    - Loops: L(14) = RSI_a, RSI_b (2 seeds, 128 pairs), SEA, SPO, RND_1, RND_2 (64 pairs); L(18) = L(6) = RSI_a, SEA, RND_1; 4 rounds.
      Keys keyseed(L + SUF, n, p, s, b) and 'mirror' exactly as round 1.
    - Then rk_verdict 'gates' / 'block' runs on the replay directory (round-1 statistics and strings), and r1b_stats on the replay
      evaluations (p_c, p_s, A intervals, S_4^cf, R_4^cf).
  - **Open-loop replay.**
    - A second reduced copy is written to <work>/r1b_replay_open/rk/b<b>/state.pt.
    - A driver imports rk_loop as a library (argv [rk_loop.py, root, <work>/r1b_replay_open/rk, 'lib'], env as above, one process
      per (block, ASCALE)) and calls rk.evaluate(L + SUF, n, d = R1.state[b]['dir_<L><SUF>_<n>'], nseed, alpha_b) for n = 1, 2, 3
      and every loop of L(b).
    - The pushes are round 1's stored directions, so each call is round 1's evaluation n: round-1 keys and the same function
      rk.run_loop calls.
    - Texts: 6,144 (3,072 at 14 at 1.5 alpha; 1,536 each at 18 and 6). Round 0 is not repeated (the closed loop's round 0 is
      already an open-loop evaluation).
- Rule: descriptive, with a reproduction flag. 'Replay reproduces round 1' iff all three of the following hold; otherwise 'replay
  differs (<which>)'.
  - (i) Token identity. The share of token-identical texts is >= tau in every evaluation whose push is a stored round-1 vector
    (pilots, round 0 and the open-loop rounds 1-3). tau depends on how the replay's GPU model and its torch, CUDA and transformers
    versions compare with the round-1 shard's (shard_done.json):
    - 0.95 when all four are equal;
    - 0.85 when only the GPU model differs (G47: L4 against A100 with equal software gave 0.969 / 0.938 on 64 axis texts);
    - 0.80 otherwise, labelled 'software versions differ'.
  - (ii) Paired tolerance on the open-loop rows. For c and rel, the absolute difference |replay - round 1| on the same row is
    <= max(2.5 SD_diff, 1e-4) in at least 95 % of the rows and <= 4 SD_diff in every row (R.9 a). SD_diff is the prompt-bootstrap SD of the
    paired difference (the same prompt weights for both).
  - (iii) rk_verdict's gate list (GO0-GO3, which uses pilots and round 0 only) equals round 1's, and PC_PASS computed with round-1
    formulas on the open-loop rows equals round 1's.
  - The closed-loop rounds diverge by construction. A 3-6 % per-text divergence at round 0 re-samples 8-15 of 128 pairs after their
    divergence point, turns d_1 by roughly 5-15 %, and every later push moves with it. So their token identity per round and their
    statistics are reported descriptively (rung L2 of R2.6), never as a reproduction criterion.
  - The round-1 strings and the 1b-rule strings (on 64-prompt data; perp* readings with the leave-loop-out mode, 'self-overlapping'
    beside) are printed side by side.
- Null / controls: round 1 itself (same keys); the replay's own RND loops. The open loop is the control for loop amplification.
- Shows - cannot show:
  - The open loop shows how much of every round-1 row is reproducible on this hardware and software, given round 1's pushes.
  - The closed loop shows how strongly the loop amplifies tiny run differences.
  - Both show what the old and the new statistics say about identical data. Neither is new evidence about RSI (same 64 prompts, same
    push).
- MDE: as round 1 (64 prompts; paired-test MDE 0.09 / 0.16 / 0.14 in c). The tolerance in (ii) is relative to the replay's own
  paired SD.
- Inputs:
  - R1.state[b]: 'ref', 'axis_*', 'axes', 'alpha', and 'dir_<L><SUF>_<n>' for n = 1..3;
  - R1.loops[b] (rows, '_meta');
  - the R1 texts of the same keys (for match rates);
  - shard_done.json (GPU and versions);
  - rk_dirs.pt.
- Runs in: r1b_R_S1. 26.6 A100-min with margin (17.8 min work: closed loop 4,480 / 2,048 / 2,432 texts in 8 stage subprocesses, open loop
  6,144 texts in 3 processes, reads, damage, code checks).
- Gates: RG1-RG4; CC2 and CC3 (the replay's own GO2 checks: mirror and early-stopped read).

### R2 The corrected loop

**R2.1 Prompt set P256 and axes AX-1b (R.2 objects; AX-1b is part of main SO-AX)**
- What: a larger frozen prompt set, and the axes the corrected loop is read on, including the verdict axes.
- How:
  - **New prompts.** 192 new prompts are written before the freeze by the plan authors and checked by a frozen script
    (r1b_prompts_check.py):
    - 96 RSI-topic prompts ('on AI systems, learning, improvement, skills or change, with the stance left open', as round 1) and 96
      scene prompts (everyday scenes);
    - 5-16 words, one sentence, ending in '.' or '?';
    - no word of W_sea, W_sports or W_tone and no 'water' in any new prompt; no AI / learning / improvement word in the scene prompts;
    - no near-duplicate (character-trigram Jaccard >= 0.6) among themselves or with the 64 round-1 prompts.
    - Order and fold as in R.2.
  - **Axis texts.** The four round-1 instructions (rk.INSTR) are appended to each prompt as rk.text_axis does, with pro and anti
    sharing one key.
    - All 256 prompts' axis texts are generated once in r1b_R_S2_14, 2,048 texts in all:
      - the 64 round-1 prompts with their round-1 keys keyseed('axis', name, p), so that token identity with round 1 is reported;
      - the 192 new prompts with ('r1b_axis', name, q).
    - The texts are copied to the other sessions, so one A100-generated text set serves every block and R7.1's common-random-number
      pairs. Round 1 generated its axis texts per block on different GPUs; block 14's ran on an L4.
    - Reads are taken at block b (rk.read). u_neg = R1.state[b]['axes']['u_neg'].
  - **Axes per block.**
    - D_p^X = x_pro,p - x_anti,p and u_X = mean_p D_p^X over the kept prompts;
    - u_X_perp^1b, u_X^perp* and u_X^v (R.1);
    - fold axes and kappa_X;
    - cross-fitted centred AUC and gap (r2_rk.axis_from_texts / cross_fit_on_axes);
    - centred nulls (seeds 311-314);
    - the split-half rel of each axis (n_prompts = 256).
  - **Re-read check.** For round-1 axis texts that the regeneration reproduces token-identically, the 1b read must equal R1 Xp / Xa to
    1e-4 relative; otherwise the flag 'R1 axis reads differ' is set. The 1b reads are used either way.
  - **WORDSETS frozen here.**
    - W_sea (lexical PC; whole-word, case-insensitive regex on decoded text): sea, seas, ocean, oceans, oceanic, wave, waves, tide,
      tides, tidal, beach, beaches, shore, shores, shoreline, coast, coasts, coastal, coastline, seaside, harbour, harbours, harbor,
      harbors, sailor, sailors, sailing, sailboat, boat, boats, ship, ships, seagull, seagulls, gull, gulls, marine, maritime, surf,
      surfing, lighthouse, lighthouses.
    - W_RSI: improve, improvement, improving, self-improvement, better, learn, learning, upgrade, recursive, optimize, optimise,
      optimization, capability, capabilities, intelligence, smarter, evolve, enhance, enhancement, grow, growth, progress, refine,
      iterate, iteration, adapt, train, training, autonomous, rewrite. Sorted alphabetically: odd positions form the build half,
      even positions the eval half.
    - W_sports: sport, sports, football, soccer, basketball, tennis, game, team, match, player, athlete, coach, score, goal, race.
    - W_tone: amazing, wonderful, fantastic, exciting, incredible, love, awesome, thrilled, delighted, brilliant, great, joy,
      beautiful, magical, superb.
    - Neutral sets: 20 sets of 15 words drawn (np.random.default_rng(4242)) from the frozen neutral pool (Appendix R.A), matched to
      W_RSI-eval (used by R5); R4.2 uses a second family of 20 sets matched to W_RSI-build the same way (np.random.default_rng(4243);
      R.9 h).
      - The pool excludes every concept list, the A3 lexemes and every word (any inflection) of the eight round-1 axis instructions.
      - Each set is matched to W_RSI-eval on the decile histogram of the unsteered model's mean log-probability of each word's first
        token at continuation positions of 64 TXT0 texts. That histogram is computed in r1b_R_S3 and is outcome-independent.
      - A word's probability is the summed probability of the first tokens of its four spellings (with and without a leading space,
        lower- and upper-case initial; deduplicated).
- Rule: GO0_1b(b): the cross-fitted centred AUC(u_T^v(b)) >= 0.85; otherwise every RSI verdict at b reads 'not decidable (RSI axis
  not resolved)'. For the PC axis: AUC(u_sea^v(b)) >= 0.80. If that fails, the PC is decided on the sports loop SPO_t with
  AUC(u_sports^v(b)) >= 0.80 (the pre-registered fallback). If both fail: 'PC not testable (no topic axis beyond the generic mode)'.
- Null / controls: centred pseudo-axes; the re-read check against R1.
- Shows - cannot show: axes from four times the prompts, with the generic mode removed where round 2 found one. It cannot remove
  generic components not spanned by u*_b; the cosine of each verdict axis with the 1b loops' own modes is reported (R2.7).
- MDE: axis estimation noise SD about sigma_d / sqrt(1 + N s^2): 0.017 / 0.013 / 0.017 at 256 prompts (est., int_gates 3.3 fit), half
  the 64-prompt value.
- Inputs: R1.state[b] ('axis_*' for the round-1 keys' token-identity report, 'axes' u_neg and fold); u*_b; prompts_1b.json; the
  neutral pool file.
- Runs in: r1b_R_S2_14 (2,048 texts and reads), r1b_R_S2_18 and r1b_R_S2_06 (reads). Included in R2.2's minutes.
- Gates: RG1-RG4.

**R2.2 Loops in three push arms at blocks 14, 18, 6**
- What: the loop dynamics with the repaired design, in three push arms: damage-normalised push (primary), fixed round-1 push
  (attribution) and norm-preserving push (control).
- How: per block b in {14, 18, 6}; one process per block via r2_rk.open_block plus r1b_rk's hooks; T 0.7; 4 rounds d_0 -> d_1 ... d_3
  -> d_4.
  - **Arm DN** (primary; PUSH-DN, a_{L,n} calibrated every round):
    - RSI_a and RSI_b start at unit(u_T^v(b)), 256 prompts x 2 seeds = 512 pairs each;
    - SEA_t (the PC loop) starts at unit(u_sea^v(b)), 512 pairs;
    - SPO_t (fallback PC and RSI-specificity control) starts at unit(u_sports^v(b)), 512 pairs;
    - SEA_raw (round-1 PC start) starts at unit(raw u_sea(b)), the unprojected 1b sea axis, 256 pairs;
    - RND_1 and RND_2 start at isotropic Gaussian unit vectors from torch.Generator seeds 4101 + b and 4201 + b, each orthogonalised
      against u*_b, u_T^v, u_sea^v, u_sports^v, u_T^perp* and u_sea^perp* and renormalised, so that no start lies on the style mode or
      a test axis; 256 pairs each;
    - RSI_a and RSI_b run two more rounds (d_5, d_6; descriptive).
  - **Arm FX** (attribution of the push change): RSI_a (512), SEA_raw (256) and RND_1 (256) at the fixed round-1 push alpha_0(b)
    (scale as round 1), same starts.
  - **Arm NP** (control; PUSH-NP calibrated to dmg_sym 0.05 per round): RSI_a (512), SEA_t (256), RND_1 (256).
  - **Keys and outputs.**
    - Keys ('r1b_' + L, n, q, s, b) are shared across arms, so the arms are paired.
    - Outputs go to their own files under <work>/r1b/, never into the copied round-1 state (r2_rk's state guard admits only
      'r2'-prefixed keys there).
    - Every evaluation stores, in the storage format of R.1 (float32; the 'all' window full; 'div' and 'first20' as projections on
      F(b)): delta, ok, prompt_of, seed_of, xp, xm, first_div, conts (token ids), cbar, a_{L,n}, dmg_+ and dmg_- of the push, top-1
      change, capped share, the achieved along-d displacement per unit a (NP and DN), and the word counts of every text for all
      WORDSETS.
- Rule: feeds R2.3-R2.7 and R3.2.
  - Code checks per process:
    - CC1: zero push with shared keys gives token-identical texts (4 of 4);
    - CC2: pushing -u swaps the pair (>= 3 of 4);
    - CC3: early-stopped read against the full pass, < 1e-4 relative;
    - CC4 (in r1b_R_S2_14): r1b_core's push hook, registered on rk.model, equals rk's push at block 14 to 1e-6 relative.
  - A failed check makes every verdict that uses the affected arm 'descriptive (code check failed)'.
- Null / controls:
  - RND loops, whose starts are orthogonal to the style mode and the test axes;
  - SPO_t; the FX and NP arms; the unpushed references of PUSH-DN.
  - The RSI and SEA_t starts are full-sample axes of the same prompts. Their estimation noise can return through the loop only as
    much as a random direction does (round 2: g_par of the RND starts at n = 0 was -0.0023 / -0.0001 at block 14), and the
    cross-fitted readout removes what returns.
- Shows - cannot show: the loop at a push that cannot drift into high damage, with four times the prompts. It cannot show
  self-sustainment of magnitude, because each round is renormalised; that question belongs to R3.3 and R4. It shows persistence of
  direction only.
- MDE: see R2.3 and R2.4 (per-block SD of A at 512 pairs: 0.042-0.051 at 14, 0.030-0.034 at 18, 0.033-0.035 at 6, est.).
- Inputs: AX-1b(b), u*_b, R1.state[b] ('ref' for cap_b and the damage references, 'k5'), rk_dirs.pt.
- Runs in: r1b_R_S2_14, r1b_R_S2_18, r1b_R_S2_06 (one session per block). 49.6 / 51.6 / 51.6 A100-min with margin (33.1 / 34.4 / 34.4 min work:
  4,864 pairs per round x 4 rounds, RSI rounds 5-6, axes, calibrations; R2.1 included).
- Gates: RG1-RG5, RG9.

**R2.3 Redefined positive control PC_PASS_1b (gate for the positive RSI verdicts)**
- What: at block b, does the loop keep a topic beyond the generic attractor, specifically and detectably? It is required for 'likely'
  and 'oscillation' at b, and it is reported as a substantive outcome in its own right.
- How: the DN arm's SEA_t is read on u_sea^v(b) (one-sided toward pro-sea), rows n = 1..4; lexical sea counts are taken from the texts
  of evaluations 1-3 (pushes d_1..d_3); the RND loops are read on the same axis and counted the same way. Fallback (RG7): SPO_t on
  u_sports^v with W_sports.
- Rule: PC_PASS_1b(b) holds iff all of P1-P5 hold:
  - (P1) for n = 2, 3, 4: p_s <= 0.01 and A_n's 90 % lower bound >= 0.10;
  - (P2) rel_n >= 0.5 for n = 2, 3, 4 (point);
  - (P3) S_4^cf >= 0.5 (point);
  - (P4) lexical, one pooled test: the per-prompt mean of count(+) - count(-) of W_sea words, over evaluations 1-3 and the prompt's
    pairs, has a one-sided sign-flip p <= 0.01 (10,000 flips, np.random.default_rng(2003));
  - (P5) specificity: A_4(SEA_t) - max(|A_4(RND_1)|, |A_4(RND_2)|) (absolute values in each replicate; the RND loops' sign is
    arbitrary) has a 90 % lower bound > 0 (R.9 b).
  - Verdict strings: 'PC_PASS_1b' or 'PC fails (<P1..P5 components>)'. 'GO1_1b' (row 1: rel >= 0.5, p_s <= 0.01, A lower bound
    >= 0.10) is reported, not gating.
  - Label 'no topic persists beyond the generic mode at b' if PC_PASS_1b fails, SEA_t and SPO_t both have |A_n| upper bounds < 0.25
    at n = 3 and 4, and SENS_sea(b) passes (R2.3b). At block 6, where nothing is removed from the axes, the label reads 'no topic
    persists at b'. If SENS_sea(b) fails, the label is 'PC not informative (insensitive)'.
  - Reported beside it: the round-1 PC definition evaluated on SEA_raw (raw u_sea, p_s instead of round 1's p), and Q_c.
- Expectation, stated before the run from round-2 results: under round-1 dynamics, PC_PASS_1b is expected only at block 18.
  - Block 14. The round-1 SEA loop is a member of the mode set from which u*_14 is built (debiased cos(d_4^SEA, d_4^RSI_a) = -1.051,
    TV blocks.14.B_Q0.mode_set), and it lies on that mode from its first read-back (image cos(d_1^SEA, u*_14) = -0.930). A SEA loop
    with these dynamics reads A of about 0 on u_sea^perp*(14) at n = 2-4; even at d_1 the whole non-u* residual would give at most
    sqrt(1 - 0.930^2) / sqrt(0.868) = 0.39 (rel_1 of SEA 0.868).
  - Block 6. SEA is again a member (debiased -1.142, image -0.834), but u*_6 is not generic (R4 'unaligned', RND_1 0.0099). This is
    why block 6's PC axis keeps that direction: u_sea^v(6) = u_sea_perp^1b(6).
  - Block 18. SEA lies outside the mode set (0.229, image 0.118). Round 1's PC failed there on rel only (0.4376, 0.4877), and four
    times the pairs project rel to 0.76-0.85.
  - The damage-normalised push may change these dynamics. A PC failure at block 14 is therefore expected, is not read as a machinery
    failure, and no longer blocks the negative RSI verdicts (SENS_1b does that, R2.3b).
- Null / controls: RND loops. Their starts are orthogonal to the axes; at 14 and 18 they converge onto u*_b, which is orthogonal to
  u_sea^v by construction, and at 6 they do not converge onto u*_6. The lexical counts are a second, style-independent readout.
- Shows - cannot show: that the machinery detects a topic carrying itself through the text loop beyond the generic mode at this block.
  A failure, together with the 'no topic persists' label, is itself a finding about the loop. It says nothing about RSI by itself.
- MDE (joint, joint_R.py, normal model): P1 and P5 jointly reach power 0.8 at a true A_n of SEA_t of about 0.23-0.24 (block 14),
  0.195-0.20 (18) and 0.20-0.205 (6), for within-loop row correlations from 0.8 to 0.5.
  - Revision 1's per-row values (0.21 / 0.18 / 0.17-0.18) understated the conjunction.
  - P2 and P3: projected rel_n is 0.79-0.86 / 0.76-0.85 / 0.86-0.90, assumed met.
  - P4: MDE in sea words per text pair = (2.326 + 0.842) x SD_p / 16, where SD_p is the across-prompt SD of the per-prompt mean
    difference. SD_p is measured on round-1 SEA texts in R0.6 and printed before R2's analysis; the repository holds no round-1 texts
    to estimate it now.
  - For comparison, the bar-limited rule 'p_s <= 0.01 and Q_c >= 1' would need c of about 0.26-0.27 (A about 0.29-0.30) at any N.
  - These MDEs use round 1's SEA noise; how much sea signal survives on the verdict axis is the expectation above.
- Inputs: R2.2 (DN arm SEA_t, SPO_t, RND_1, RND_2), AX-1b(b), SENS_sea(b).
- Runs in: r1b_R_ana (statistics; included in r1b_R_ana).
- Gates: RG6, RG7; defines RG8a.

**R2.3b Sensitivity check SENS_1b (gate for the negative RSI verdicts)**
- What: given the RSI loops' own noise at b, would the readout have detected a persistent RSI signal of the smallest size that the
  positive 'likely' strings claim (A = 0.25)? SENS_1b replaces the PC as the precondition of 'unlikely'.
- How: a planted signal in the DN arm's RSI_a and RSI_b, rows n = 2, 3, 4, on u = unit(u_T^v(b)) (full sample).
  - Every delta_j of a row becomes delta_j - (u . d_n) u + eta_n u, where d_n is the row's mean and eta_n = c*_n |d_n - (u . d_n) u|
    / sqrt(1 - c*_n^2). This makes cos(d_n', u) = c*_n = 0.25 sqrt(rel_n).
  - The planted rows keep the pairs' own noise. Read through the cross-fitted, disattenuated pipeline, the planted A is 0.25 to within
    about 2 %: the fold axes see the plant with cosine sqrt((1 + kappa_T) / 2) and the disattenuation divides by sqrt(kappa_T), while
    the plant raises rel slightly. The realised planted A is printed.
  - The planted rows go through the full readout in each of 500 outer prompt-bootstrap replicates (np.random.default_rng(2101)):
    fold axes re-estimated per replicate, cross-fitting, disattenuation, K1-s with 2,000 flips, and the A interval from 200 inner
    joint-bootstrap replicates.
  - The same is done on SEA_t with u_sea^v(b) (SENS_sea(b), reported; it makes a PC failure interpretable).
- Rule: SENS_1b(b) passes iff, for both RSI loops, p_s <= 0.01 and A_lo >= 0.10 at all three rows in >= 80 % of the outer replicates.
  Otherwise the negative verdicts at b read 'undecided (sensitivity check failed)'. The pass rate is printed.
- Null / controls: the unplanted rows; the planted rows read on the foreign verdict axes (must stay within their unplanted intervals;
  reported).
- Shows - cannot show: that the pipeline (axes, cross-fit, bootstrap, test) does not destroy a real signal of size 0.25 and has the
  power to see it in the RSI loops' own noise. It cannot show that the model could carry RSI; that is the hypothesis.
- MDE: pass rate at the planted 0.25 (joint_R.py) is 0.82-0.85 at block 14 and 0.99 at blocks 18 and 6. Block 14 is marginal: SENS
  fails there if SD_A exceeds about 0.046, inside mde_R's range 0.042-0.051. The negative verdicts at block 14 can therefore end as
  'undecided (sensitivity check failed)' (stated in advance).
- Inputs: R2.2 DN RSI_a, RSI_b, SEA_t; AX-1b(b).
- Runs in: r1b_R_ana (GPU; about 1e13 FLOP, inside r1b_R_ana's minutes).
- Gates: RG6; defines RG8b.

**R2.4 RSI verdict per block (V_RSI_1b)**
- What: does RSI content persist through the loop, beyond the generic mode, at block b?
- How:
  - DN arm RSI_a and RSI_b are read on u_T^v(b) (primary). Also reported: u_T_perp^1b at 14 and 18, u_T^perp* at 6.
  - Foreign readouts on u_sea^v, u_sports^v, u_tone^v and u_neg: D4 = |A_4| on u_T^v minus the maximum |A_4| on these. Style is
    handled by the 'style-attracted' label, not by D4.
  - SPO_t and the RND loops are also read on u_T^v.
  - K5 behaviour z_beh(d_4) uses R1.state[b]['k5'] (round-1 gate, unchanged).
  - Debiased |cos(d_4, u*_b)| is reported.
- Rule:
  - MANIP_1b: RSI_a and RSI_b row 1 both have p_s <= 0.01 and A_1's lower bound >= 0.10.
  - C_pos, the common conditions of the positive strings:
    - MANIP_1b;
    - mean rel_2..4 >= 0.5 (point);
    - E defined and its lower bound >= 0.5;
    - R_4^cf >= 0.5 (point);
    - D4 > 0 for both loops (point);
    - SPO_t's |A_4| upper bound < 0.25;
    - no RND loop with two-sided p_s <= 0.01 and |A|'s lower bound >= 0.25 at n = 3 or 4 (R.9 b);
    - the K5 gate: if z_beh(u_T^v) >= 3, then mean z_beh(d_4) >= 1.64 (strings 4 and 5) or mean z_beh(d_4) x sign(A_4) >= 1.64
      (string 6, round 1's ok_oscillation; R.9 c), and |z_sports|, |z_literature| < 2.58.
  - V_RSI_1b(b) is the first string in this list whose conditions hold:
    1. 'descriptive (code check failed; would have read: <string>)' when RG5 failed for the DN arm at b.
    2. 'not decidable (RSI axis not resolved)' when GO0_1b(b) fails.
    3. 'not decidable (damage calibration failed)' when more than 1 of the 4 pushes d_0..d_3 of RSI_a or RSI_b lies outside the band.
    4. 'likely, RSI-dominated': PC_PASS_1b(b); C_pos; p_s <= 0.01 toward the pro pole at n = 2, 3, 4 for both loops; A_bar's lower
       bound >= 0.50.
    5. 'likely, substantial RSI component': the same with A_bar's lower bound >= 0.25.
    6. 'oscillation (RSI begets anti-RSI)': PC_PASS_1b(b); C_pos; two-sided p_s <= 0.01 at n = 2, 3, 4 for both loops; the signs of
       A_1..A_4 alternating; A_bar_abs's lower bound >= 0.25.
    7. 'unlikely (RSI does not persist)': SENS_1b(b); MANIP_1b; A_bar_abs's upper bound < 0.25; no RND loop attracted to RSI (as in
       C_pos). A single push out of band only labels the row (RG9; R.9 d).
    8. 'unlikely (no first link)': SENS_1b(b), and A_1's upper bound < 0.10 for both loops.
    9. 'undecided (positive control not testable)' when the conditions of 4, 5 or 6 hold except the PC, under RG7's last case.
    10. 'undecided (positive control failed)' when the conditions of 4, 5 or 6 hold except PC_PASS_1b.
    11. 'undecided (sensitivity check failed)' when the conditions of 7 or 8 hold except SENS_1b.
    12. 'undecided' otherwise.
    Strings 4-8 exclude one another by their bounds; at the edges, the order settles.
  - Labels:
    - 'style-attracted' if the debiased |cos(d_4, u*_b)| >= 0.5 for both RSI loops;
    - 'damage out of band' for single pushes (RG9);
    - 'bootstrap-t used (RG6)';
    - 'axis-sensitive (1b generic mode)' (R2.7);
    - 'no topic persists beyond the generic mode at b' (R2.3);
    - 'confirmatory (Holm)' or 'exploratory (Holm not met)' for the members of F+ and F- (R.1).
  - The same verdict function, evaluated on the other RSI axis, is printed beside the primary one.
- Null / controls:
  - the paired sign flips;
  - RND loops (attraction to RSI);
  - SPO_t (a non-RSI topic loop must not drift onto RSI);
  - the foreign axes (D4);
  - the generic mode removed from the axis at 14 and 18, so convergence onto u*_b cannot count as RSI there;
  - SENS_1b for the negatives.
- Shows - cannot show: whether an RSI component beyond the generic mode persists over four renormalised rounds. It does not show
  whether the loop sustains its magnitude (R3.3, R4), it reads behaviour only through K5's linear proxy, and it covers only three
  blocks.
- MDE (joint, joint_R.py; SD(A_bar) 0.034-0.037 / 0.025-0.027 / 0.026-0.028 at 14 / 18 / 6 with the loops' shared prompts):

  | Rule | Block 14 | Block 18 | Block 6 |
  |---|---|---|---|
  | 'likely, substantial', reached from true A_bar about (flat or 0.8-decaying profile) | 0.335-0.345 | 0.315-0.32 | 0.315-0.323 |
  | 'likely, RSI-dominated', reached from true A_bar about | 0.585-0.595 | 0.565-0.57 | 0.565-0.574 |
  | 'likely, substantial', Holm-confirmed when it is the only true member of F+ | 0.37-0.38 | 0.335-0.345 | 0.34-0.35 |
  | 'unlikely (RSI does not persist)', reachable (first link A_1 = 0.30) when the true late A is below | 0.155-0.165 | 0.18-0.185 | 0.18-0.185 |
  | 'unlikely (RSI does not persist)', Holm worst case | 0.12-0.13 | 0.155-0.165 | 0.155-0.16 |
  | MANIP_1b reached from A_1 about | 0.23 | 0.195 | 0.20 |
  | 'no first link', probability at a true A_1 = 0 | 0.58 | 0.86 | 0.83 |

  - Block 14 cannot reach power 0.8 for 'no first link' (stated).
  - SD(E) at A = 0.25 / 0.33 / 0.50, for within-loop correlations from 0.8 to 0.5:
    - block 14: 0.10-0.17 / 0.08-0.12 / 0.05-0.08;
    - block 18: 0.07-0.12 / 0.06-0.09 / 0.04-0.06;
    - block 6: 0.08-0.12 / 0.06-0.09 / 0.04-0.06.
    So near A = 0.3, E's lower bound >= 0.5 needs a true E of about 0.65-0.70.
  - Round 1's block-14 values (A_3, A_4 = 0.18 / 0.17 and 0.23 / 0.21 on u_T_perp, close to u*_14's own cosine 0.17 with u_T_perp)
    would read 'undecided' under these rules. A true value between the 'unlikely' and the 'substantial' MDEs is not decidable by
    design, and the report says so.
- Inputs: R2.2, R2.3, R2.3b, AX-1b(b), u*_b, R1.state[b]['k5'].
- Runs in: r1b_R_ana.
- Gates: RG5-RG9 (RG8a for strings 4-6, RG8b for 7-8).

**R2.5 Relation to delta (round 1's delta rule, repaired)**
- What: does the persistent RSI share change with delta (round 1's meaning-space size per block)?
- How: the blocks with PC_PASS_1b and no failed RG5 / RG7 / RG9 enter, ordered by R1.delta (gpu2/dn_delta.json, last-token; 14:
  0.1176, 18: 0.1358, 6: 0.1944). The statistic is the paired 90 % bootstrap interval of A_bar_abs(highest-delta) -
  A_bar_abs(lowest-delta). Prompts are shared across blocks; keys differ.
- Rule:
  - 'RSI share falls with delta' if the interval lies below 0 and the middle block's A_bar_abs is not above the upper end of the
    lowest block's interval; 'rises with delta' mirrored; otherwise 'no monotone relation'.
  - With two entering blocks the middle clause is dropped (marked). With fewer: 'not decidable (fewer than two blocks passed
    PC_PASS_1b)'.
  - Always stated: three blocks cannot separate delta from depth (round 2's G31 failed, ratio 0.4646 < 2).
- Null / controls: the paired bootstrap.
- Shows - cannot show: a monotone ordering over at most three blocks; not delta as such. Given the PC expectation (R2.3), the rule is
  expected to read 'not decidable' (stated in advance).
- MDE (joint_R.py, correlated loops, cross-block correlation 0.3): a difference of about 0.08-0.10 in A_bar_abs (SD of the difference
  0.033-0.039).
- Inputs: R2.4 per block; gpu2/dn_delta.json.
- Runs in: r1b_R_ana.
- Gates: RG8a at >= 2 blocks.

**R2.6 Attribution ladder (descriptive)**
- What: which single change moves which round-1 quantity.
- How: each rung differs from the previous one by one named change:
  - L0: round-1 data, round-1 statistics (R0.1);
  - L1: round-1 data, 1b statistics (R0);
  - L2: replay data, 1b statistics (R1: run and device). The open loop gives the row-wise reproduction; the closed loop gives the
    token identity per round and how far the loop amplifies run differences;
  - L3: FX arm (256 prompts, 1b axes, verdict-axis starts, fixed push);
  - L4: DN arm (push normalised; paired with L3 by keys);
  - L5: NP arm (norm preserved; paired with L4);
  - L6: PC definition (SEA_raw on raw u_sea against SEA_t on u_sea^v, both DN).
  - Per rung: PC components, MANIP, A_n of RSI_a on u_T^v and u_T_perp^1b, damage of every push, debiased |cos(d_4, u*_b)|, and the
    round-1 and 1b verdict strings. Paired bootstrap differences are given where rungs share prompts and keys.
  - At L0-L2, perp* readings of round-1 loops use the leave-loop-out mode, with the full-u* reading beside it ('self-overlapping').
- Rule: descriptive.
- Null / controls: none beyond the rungs.
- Shows - cannot show: attribution of changes in statistics to single design changes. L3 bundles prompts, axes and starts (one change
  of design, three consequences).
- MDE: per quantity, the SDs of R2.3 / R2.4.
- Inputs: R0, R1, R2.2.
- Runs in: r1b_R_ana.
- Gates: none.

**R2.7 The style mode under the corrected push (descriptive, with one pre-registered robustness reading)**
- What: is there still a generic attractor when no push exceeds 0.05 nats, and is it still round 2's u*_b?
- How:
  - B-Q0's recipe on the DN arm: mode set by debiased |cos(d_4^L, d_4^RSI_a)| >= 0.5, cross-fitted; u*_1b(b); cos(u*_1b, u*_b);
    RND loops' debiased |cos(d_4, u*_b)|; the cosines of u_T^v and u_sea^v with u*_1b.
  - For the 'generic attractor' label, each RND loop L is compared with the leave-one-out mode u*_1b^(-L), built without L, so that
    a loop is never compared with a mode it shaped.
  - Because u*_1b contains RSI_a by construction, a second, RSI-free mode is built: u°_b = the unit vector of the sign-aligned sum of
    d_4 over the non-RSI DN loops (SEA_t, SPO_t, RND_1, RND_2) whose pairwise debiased |cos(d_4, d_4)| is >= 0.5. It needs at least
    two loops; otherwise 'no RSI-free generic mode'.
- Rule: descriptive labels:
  - 'same attractor as round 2' if |cos(u*_1b, u*_b)| >= 0.7;
  - 'generic attractor' if some RND loop L has debiased |cos(d_4^L, u*_1b^(-L))| >= 0.5;
  - 'no common mode' otherwise.
  - Robustness (pre-registered, at every block): if u°_b exists and |cos(u°_b, u*_b)| < 0.7, R2.4 is read a second time on u_T^v
    additionally orthogonalised against u°_b. The verdict stays the primary one; it carries 'axis-sensitive (1b generic mode)' if the
    two readings give different verdict strings.
- Null / controls: the RND loops (starts orthogonal to u*_b); the NP arm (R3.2).
- Shows - cannot show: whether round 2's attractor survives damage normalisation; not why (R3.4, R4).
- MDE: debiased cosines with SD about 0.05 (est.).
- Inputs: R2.2.
- Runs in: r1b_R_ana.
- Gates: RG5.

### R3 Damage-normalised push, norm-preserving control, dose series, no-push noise

**R3.1 PUSH-DN (part of main SO-DMG; calibration table)**
- What: a push of fixed damage per direction and round, so that no loop drifts into high damage (round 1: RSI_a went from 0.1135 to
  0.3235 nats, SEA up to 0.4492 at block 14).
- How:
  - a_{L,n} = r2_unlearn_qwen.calibrate(lambda a: dmg_sym(unit(d_n), a), start = a_{L,n-1} (round 0: alpha_0(b)), rule {target
    0.05, lo 0.045, hi 0.055, max_eval 6}); the closest value tried is kept.
  - dmg_sym is computed on 16 references: R1.state[b]['ref'] continuations of round-1 prompts 0-7 (RSI-topic) and 32-39 (scene)
    (main SO-DMG's REF-R16). The unsteered log-probs are cached (RefKL pattern, 16 texts, about 0.8 GB float64); one evaluation = 32
    steered forwards.
  - Every calibrated push's damage is also evaluated once on REF-K16 (part K's DMG_K token ids, frozen in K1) and reported on
    REF-U32 = REF-R16 union REF-K16 (main SO-DMG; descriptive; 4.0 A100-min in all, main §4.2). The calibration itself stays on R's
    16 references.
  - Logged per push: a / alpha_0(b); dmg_+ and dmg_-; the asymmetry dmg_+ / dmg_-; the in-band flag; the values tried; the capped
    share; the top-1 change rate on the 8 scene references ('neutral top-1 change'); and the damage of the derived d_(n+1) at a_{L,n}.
  - Every gain divides by a_{L,n} cbar (not by alpha_b), and the gains of R2 / R3 carry a_{L,n} per row.
- Rule: RG9.
  - A push is 'in band' if dmg_sym lies in [0.045, 0.055].
  - The label 'asymmetric damage' is added if dmg_+ / dmg_- lies outside [0.5, 2].
  - A loop with more than 1 of its 4 pushes out of band makes its verdicts 'not decidable (damage calibration failed)' (R2.4 step 3).
  - The table itself is descriptive: distributions of a / alpha_0 per direction family, per round and per block.
- Null / controls: the unpushed references.
- Shows - cannot show: the push size that one fixed damage needs along each direction. It fixes damage on 16 short references only, so
  part D must recalibrate per context length (moot: part D runs no push; main C30).
- MDE: not a test. Expected a / alpha_0 (est., + push on 8 RSI references): about 0.34-0.36 along u*_14, 0.62-0.66 along u_T_perp(14),
  0.91 along u*_18 and 0.75 along u*_6; dmg_sym on 16 references will shift these.
- Inputs: R1.state[b]['ref']; REF-K16 token ids (part K's K1, frozen before the freeze; for the REF-U32 report only).
- Runs in: inside r1b_R_S2_* / r1b_R_S3 / r1b_R_S6 / r1b_R_S7 (costed there).
- Gates: RG5.

**R3.2 Norm-preserving control (PUSH-NP)**
- What: is the style attractor (and part of the damage) a product of the additive push inflating the state norm?
- How:
  - An extra forward hook is registered after rk's steer hook, with a prepend hook recording |x_t| before the push (the ConstPush
    pattern, r2_unlearn_qwen.py:442-466). At the pushed positions x' = (x + add) |x| / |x + add|; unpushed positions are unchanged,
    and the leaf path STEER['V'] is untouched.
  - The push is calibrated with the same hook to dmg_sym 0.05 per round.
  - This is arm NP of R2.2: RSI_a, SEA_t and RND_1 at blocks 14, 18 and 6 (only at block 14 under RL2, R.5).
  - Logged: the achieved along-d displacement per unit a, and the distribution over pushed tokens of |cos(x_t, unit(d))|.
- Rule (per block):
  - 'style attraction independent of norm inflation' if RND_1's debiased |cos(d_4, u*_b)| >= 0.5 in both the DN and the NP arm;
  - 'style attraction needs norm inflation' if it is >= 0.5 in DN and the NP arm's 90 % upper bound is < 0.3;
  - otherwise 'undecided'.
  - Overriding label 'NP arm uninformative (push nearly tangential)' if the mean |cos(x_t, unit(d))| over pushed tokens is < 0.05 for
    the arm's directions. 'not run (RL2)' at 18 and 6 under RL2.
  - Stated in advance: x' differs from x + add by a factor of about 1 - a cos(x_t, d) (int_machinery). So for the nearly tangential
    directions of a 2048-dim state the two arms are expected to agree, and a difference is expected only along directions aligned with
    the state itself (R0.6 reports cos(u*_b, mean state)).
- Null / controls: the DN arm with shared keys (paired).
- Shows - cannot show: whether norm inflation is needed for the attractor; not which other component of the push matters.
- MDE: debiased cosine SD about 0.05-0.07 at 256 pairs (est.), so a DN-NP difference of about 0.2 is resolvable.
- Inputs: R2.2.
- Runs in: r1b_R_S2_* (loops; costed there), r1b_R_ana (statistics).
- Gates: RG5, RG9.

**R3.3 Dose series in both signs, below 0.5 alpha_0**
- What: for small pushes, is the one-step gain of the text loop near or above 1, along the style mode and along RSI? Round 2's linear
  response 0.7298 [0.1139, 1.356] left a gain near 1 open below 0.5 alpha_0. And how do the odd and even parts scale?
- How:
  - Directions: at block 14, u*_14, unit(u_T^v(14)), unit(u_sea^v(14)) and Pool ('cov', 14, 4) (B-Q7's random direction); at blocks
    18 and 6, u*_b and unit(u_T^v(b)).
  - Doses a in {0.25, 0.35, 0.5, 1, 2} x alpha_0(b), plus the direction's PUSH-DN dose (a_DN(b, u), calibrated once).
  - Triplets (+, 0, -) per pair: 256 prompts x 2 seeds = 512 pairs per cell. Keys ('r1b_dose', b, q, s) are shared by every cell of
    the block and by the one unpushed text per key (512 unpushed texts per block). T 0.7.
  - Reads in the 'all' window (primary); 'div' and 'first20' as fixed-axis projections (R.1).
  - Per cell:
    - odd d = mean (x+ - x-) / 2; even e = mean ((x+ + x-) / 2 - x0); per-sign responses mean (x+ - x0) and mean (x- - x0);
    - g_par(u) = unit(u) . d / (a cbar), on cross-fitted fold axes for the estimated axes and as a plain projection for the random
      direction; along u*_b with round 2's attenuation m*_b from r2_q2/theta/b<b>_q0.json 'm_star2' (0.982 at 14), exactly B-Q1's
      formula, so the cells compare with round 2's;
    - G = sqrt(max(U, 0)) / (a cbar); rotation = g_par / G; decision gain = g_par if rotation >= 0.9, else G (round 2's convention);
    - even gain unit(u) . e / (a cbar), |e| as a U statistic, cos(e, u*_b);
    - dmg_+ and dmg_- at a, top-1 change;
    - quality covariates (repeated-4-gram share, continuation length) and word counts of all WORDSETS.
- Rule (per block and direction u): on g_par for the topic axes u_T^v and u_sea^v, because only the axis's own component counts as
  that content sustaining itself; on round 2's decision gain for u*_b and the random direction. k = the number of cells with a <= 0.5
  alpha_0 actually run for u, including the DN cell when a_DN(b, u) <= 0.5 alpha_0(b). Expected: k = 4 along u*_14 (DN cell at
  0.34-0.36 alpha_0), k = 3 elsewhere; under RL1, 2 or 3 at blocks 18 and 6.
  - 'below 1 at small push' if every cell with a <= 0.5 alpha_0 has a 90 % upper bound < 1;
  - 'reaches 1 at small push' if some such cell has its 0.05 / k bootstrap quantile > 1 (best of k cells, Bonferroni);
  - 'near 1 at small push' if some such cell has a lower bound >= 0.8 and none reaches 1;
  - otherwise 'undecided'.
  - Shape: a weighted least-squares fit of log g on log a over the cells with a <= 1 alpha_0 whose 90 % interval of g lies above 0.
    Weights are diagonal (1 / the bootstrap variance of log g per cell); no generalised least squares on the near-singular bootstrap
    covariance of key-sharing cells. The whole fit is bootstrapped, which gives the slope interval. With fewer than 3 such cells:
    'shape not estimable'. Readings:
    - 'linear regime' if the slope's interval lies within (-0.2, 0.2);
    - 'gain rises toward small pushes (soft onset)' if its upper bound < 0;
    - 'gain falls toward small pushes' if its lower bound > 0.
  - Even part: 'even response along u*_b' if the interval of cos(e, u*_b) excludes 0 at a >= 1 alpha_0 (round 2: cos(E_14, u*_14) =
    -0.58).
  - Checks (labels):
    - 'round-2 cells reproduced' where the u*_14 cells at 0.5 / 1 / 2 alpha_0 fall within round 2's intervals (G 0.2948 [0.2452,
      0.3432], 0.2996 [0.2682, 0.3312], 0.2783 [0.2645, 0.2922]);
    - 'consistent with the linear response' if g(0.25 alpha_0)'s interval overlaps R4.1's lambda_lin(u).
  - The u_T^v rows at 14, 18 and 6 are members of F+ ('reaches 1') and F- ('below 1'). Where GO0_1b(b) fails they read 'not decidable
    (RSI axis not resolved)' with p = 1 (R.9 f).
- Null / controls: the unpushed texts (per-sign responses); the random direction; R3.4's noise floor.
- Shows - cannot show: the small-push gain of the self-steering map along fixed directions, which closes round 2's open window. G = 1 is
  a threshold of this map under its read conventions (P2 4.0), so 'reaches 1' is not by itself self-sustainment of the natural loop;
  R6.2 is the convention-free test.
- MDE (est., 512 pairs; SD of the gain scales as a^-1.0 to a^-1.16, anchored on round 2's cells: block 14 at 0.5 / 1 alpha_0, blocks
  18 and 6 at 1 alpha_0, TV cells 'a1_T0.7' g_par SD 0.0196 / 0.0146 at 128 pairs):

  | Block | SD(g) at 0.25 / 0.35 / 0.5 / 1 alpha_0 | 'below 1' (power 0.8) when the true gain at 0.25 alpha_0 is at most | 'reaches 1' needs about |
  |---|---|---|---|
  | 14 | 0.037-0.046 / 0.025-0.031 / 0.017-0.021 / 0.007-0.009 | 0.885-0.91 | >= 1.05-1.06 at 0.5 alpha_0 (>= 1.11-1.14 at 0.25) |
  | 18 | 0.039-0.049 / 0.028-0.033 / 0.020-0.022 / 0.010 | 0.88-0.90 | >= 1.06-1.07 |
  | 6 | 0.029-0.036 / 0.021-0.025 / 0.015-0.016 / 0.007 | 0.91-0.93 | >= 1.045-1.05 |

  Along u_T^v, whose one-step gain was 0.02-0.05 in round 2, the per-text read noise is smaller than along u*_b (B-Q7 products 1.6
  against 4.3 at block 14), so these SDs are conservative there. A gain near 1 would be unmistakable.
- Inputs: AX-1b(b), u*_b, rk_dirs.pt, R1.state[b]['ref'], round-2 TV blocks.14.R1.points (for the reproduction label).
- Runs in: r1b_R_S3. 53.0 A100-min with margin (35.4 min work: block 14 24 cells, blocks 18 and 6 12 cells each, 512 pairs, DN
  calibrations).
- Gates: RG1-RG5, RG9 (for the DN cells).

**R3.4 No-push run: noise floor, noise covariance, and is the style mode the strongest noise direction?**
- What: is u*_b simply the dominant direction of text-sampling noise? If so, a normalised low-gain loop would drift onto it without
  any dynamical attractor.
- How:
  - TXT0 = 256 prompts x 16 seeds of unsteered texts (keys ('r1b_free', q, s), T 0.7, 40 tokens), read at 14, 18 and 6.
  - Sigma_text(b) = covariance of the within-prompt deviations x_{q,s} - mean_s x_{q,s} (15 x 256 = 3,840 dof).
  - Cross-fitted spectrum: top eigenvectors v_1..v_5 from the prompts of one K2 half, Rayleigh quotients in the other half, averaged
    over the 20 halvings and both assignments. Reported:
    - rho*(b) = u*_b' Sigma u*_b / lambda_1;
    - |cos(u*_b, v_1)| and the share of u*_b in span(v_1..v_5);
    - the same for u_T^v and u_sea^v;
    - the noise floor NF_text = sqrt(tr Sigma).
  - Second noise notion: Sigma_pair(b) from the DN round-0 pairs of RND_1 / RND_2 (common-random-number pairs pushed along random
    directions), with the same statistics.
  - The zero-push identity (CC1) is part of every session.
- Rule (per block, for each Sigma):
  - 'style mode = strongest noise direction' if |cos(u*_b, v_1)| >= 0.7 and rho* >= 0.8;
  - 'style mode among the dominant noise directions' if its squared projection on span(v_1..v_5) >= 0.5;
  - otherwise 'style mode is not a dominant noise direction'.
- Null / controls: random unit directions (their rho distribution); u_T^v and u_sea^v for comparison.
- Shows - cannot show: whether noise anisotropy alone can explain convergence onto u*_b. Together with R4 (gain anisotropy) it separates
  a noise-driven from a gain-driven attractor. It says nothing about RSI content.
- MDE: spiked-covariance (BBP) approximation for the half samples used in the cross-fit (D / n = 2048 / 1920 = 1.07). A population top
  direction is recovered with |cos| >= 0.7 when its eigenvalue exceeds about 3.2 x the bulk; a weaker spike reads 'not dominant' even
  if it is the top direction (stated with the result).
- Inputs: TXT0 (made here), R2.2 RND round-0 pairs, u*_b, AX-1b.
- Runs in: r1b_R_S3 (texts and reads; together with R4.1's scores 20.8 A100-min with margin, 13.9 min work), r1b_R_ana (spectra).
- Gates: RG1-RG5.

### R4 lambda_RSI versus lambda_max (MicroLens++)

**R4.1 Local Jacobian of the loop map (FDT)**
- What: the loop map's linear response at the unsteered operating point, in all directions at once.
- How:
  - For every TXT0 text j at block b: x_j (the block-b read) and the score S_j, from one forward-backward pass with rk's zero-leaf
    hook (STEER['V'], leaf added as min(|h_t|, cap_b) V from the first user token on) on y = the sum over continuation positions of
    log softmax(logits_t / 0.7)[y_t] (+ the stop token if the text ended early). This is exactly B-Q7 / r2_theta.score_grads.
  - J_b = Cov_w(x, S) / cbar_b, frequency-weighted by the prompt weights. x and S are centred within prompt, and the estimator is
    multiplied by n_s / (n_s - 1) (n_s seeds per prompt). Since E[S | prompt] = 0, J lives within prompts; the centring removes the
    between-prompt variance of x (RSI-topic and scene prompts differ along the RSI axes) from the noise.
  - Texts: seeds 0-14 at block 14 (3,840 texts), seeds 0-7 at 18 and 6 (2,048). Seed 15 of 64 prompts (every 4th prompt of each
    type) is held out for R4.2's read directions.
  - lambda_lin(u) = u' J_b u for u in {u*_b, u_T^v, u_sea^v, ('cov', b, 4)} (B-Q7 continuity).
  - J_b is never inverted. It is used only through products: J v = X_c' (S_c v) / ((n - 1) cbar) and J' w = S_c' (X_c w) / ((n - 1)
    cbar).
- Rule: descriptive inputs to R4.2; label 'FDT consistent with the dose series' where lambda_lin(u)'s interval overlaps R3.3's g(0.25
  alpha_0, u).
- Null / controls: the random direction; the held-out read-direction texts.
- Shows - cannot show: the exact (score-function) linear response of the sampled loop; nothing about larger pushes.
- MDE (est.): B-Q7's SDs at 128 texts, scaled by sqrt(128 / n) (within-prompt centring removes some noise but costs n_s / (n_s - 1)):

  | Block | B-Q7 SD at 128 texts (u* / u_T_perp / sea) | Texts | Scaled SD (u* / u_T^v / sea) |
  |---|---|---|---|
  | 14 | 0.380 / 0.142 / 0.227 | 3,840 | 0.069 / 0.026 / 0.041 |
  | 18 | 0.110 / 0.166 / 0.205 | 2,048 | 0.028 / 0.042 / 0.051 |
  | 6 | 0.169 / 0.153 / 0.112 | 2,048 | 0.042 / 0.038 / 0.028 |

- Inputs: TXT0, AX-1b, u*_b.
- Runs in: r1b_R_S3 (with R3.4).
- Gates: RG5.

**R4.2 Read and use directions; lambda_max and lambda_RSI**
- What: is the largest linear gain of the loop map larger than the largest gain within the subspace of RSI word directions
  (lambda_RSI)? Does either reach 1? The largest gain is lambda_max^C, taken within a 32-dim candidate subspace that contains the RSI
  subspace and is expected to be led by the style mode. A full-space estimate is reported descriptively. Read directions (left
  eigenvectors) and use directions (right eigenvectors) are kept apart.
- How:
  - **Local read directions** of a word set S at block b (backward products only). For each of the 64 held-out texts, g_{S,j} = the
    sum over continuation positions of grad_{h_b(t)} log p(S | prefix_t), at temperature 1, with one backward pass per (text, set).
    V_S = the top-4 left singular vectors of [g_{S,1} .. g_{S,64}] (frozen k = 4). Sets: W_RSI-build, W_sea, W_sports, W_tone and the
    20 neutral sets.
  - **Candidate subspace C_b** (k_C = 32), built only from data independent of the FDT texts:
    - u*_b (1);
    - the d_4 of the seven DN loops of R2.2 at b (7);
    - u_T^v, u_sea^v, u_sports^v, u_tone^v and u_neg (5);
    - V_RSI (4);
    - the top 15 left singular vectors of the union of the other 23 sets' gradient matrices (15);
    - orthonormalised by QR. Near-dependent columns (|R_ii| < 1e-6 relative) are dropped and replaced by further singular vectors.
    V_RSI lies in C_b, so lambda_max^C is the largest gain within a subspace that contains the RSI subspace.
  - **Definition.** For a (sub)matrix M, lambda(M) is its rightmost eigenvalue (largest real part). Its modulus and phase are
    reported, with the label 'complex top mode' if |Im lambda| > 0.05 |lambda|. The largest-modulus eigenvalue is reported beside it
    (oscillatory modes, descriptive).
  - **Cross-fitted estimation**, identical for every subspace. For each of the 20 K2 halvings and both assignments (A = fit half, B =
    held-out half; 40 fits), J_A and J_B are formed from the texts of each half's prompts (R4.1 estimator).
    - lambda_S: M_A = Q_S' J_A Q_S (4 x 4) and its rightmost eigenvalue, with right and left eigenvectors y and z (eig of M_A and
      M_A').
    - lambda_max^C: the same with the 32 x 32 matrix Q_C' J_A Q_C.
    - Held-out value: z' M_B y / (z' y), real part. Guard: if |cos(y, z)| < 0.2, the Rayleigh read y' M_B y / (y' y) is used and
      labelled 'non-normal: Rayleigh read' (share of fits reported).
    - lambda^cf = the median over the 40 held-out values; in-sample values are reported.
  - **Full-space top mode** (descriptive): Arnoldi with forward products J_A y (k = 20 steps, start u*_b, full re-orthogonalisation).
    The Ritz value with the largest real part gives the right Ritz vector y; Arnoldi with backward products J_A' z from the same
    start gives the left Ritz vector z of the closest Ritz value. Held-out and guard as above.
  - **Uncertainty.** A joint prompt bootstrap (2,000 replicates, seed 1001) of the whole procedure: texts re-weighted by prompt
    weights, halvings fixed. C_b and the V_S come from other texts and are held fixed, so the intervals are conditional on the
    subspaces. lambda_RSI^cf and the paired difference lambda_max^C,cf - lambda_RSI^cf get 90 % intervals.
  - **Null for the top mode.** 200 within-prompt permutations of the S rows (torch.Generator seeds 2201 + i). E[J] = 0 there,
    because the score has zero mean within each prompt. The cross-fitted procedure is rerun on each, for the C subspace and for the
    full space.
  - **In-session full-sample analysis** (r1b_R_S3, at the end of each block's FDT pass; seconds of GPU time). The full-sample J gives:
    - v_max^C and w_max^C (from the 32 x 32 problem);
    - v_S = Q_S y_S and w_S = Q_S z_S per word set;
    - the full-space Ritz pair v_max^full, w_max^full.
    These are stored for R4.3 (same session) and R5.5 (later session). The cross-fit, the permutation nulls and the bootstrap run in
    r1b_R_ana.
- Rule (per block):
  - 'top mode resolved' iff lambda_max^C,cf exceeds the 95th percentile of its permutation null. Otherwise 'RSI subdominant', 'RSI is
    the top mode' and 'lambda_max is the style mode' read 'not decidable (top mode not resolved)', and so does R4.3's v_max cell. The
    lambda_RSI verdicts do not need this gate: they are interval statements about a cross-fitted estimate of a fixed subspace's
    rightmost eigenvalue.
  - 'RSI subdominant' if the paired 90 % interval of lambda_max^C,cf - lambda_RSI^cf lies above 0.
  - 'RSI is the top mode' if that interval contains 0 and the squared projection of v_max^C on V_RSI is >= 0.5.
  - 'lambda_RSI near critical' if lambda_RSI^cf's lower bound >= 0.8; 'lambda_RSI >= 1' if its lower bound > 1 (member of F+);
    'lambda_RSI below 1' if its upper bound < 1 (member of F-).
  - 'RSI above neutral words' if lambda_RSI^cf exceeds the 95th percentile of the 20 neutral sets' lambda_S^cf (one-sided).
  - 'lambda_max is the style mode' if |cos(v_max^C, u*_b)| >= 0.7.
  - Full space, descriptive: 'full-space top mode resolved' iff its held-out value exceeds its permutation null's 95th percentile.
    The in-sample noise-disc edge (the permutation null's in-sample rightmost real part) is printed. A mode outside C_b is visible only
    here.
- Null / controls:
  - the 200 within-prompt permutations (top-mode null);
  - 20 frequency-matched neutral word subspaces built identically;
  - foreign topic sets (sea, sports, tone);
  - held-out evaluation of every selected mode.
- Shows - cannot show: where the linear gain of the loop lives within a 32-dim candidate subspace that contains the RSI subspace, and
  how large it is along the RSI word directions, separating what the model reads in its text (left) from what it writes back (right).
  It is first order only, at the unsteered operating point, at one block per process. A strong mode outside C_b is seen only if it
  exceeds the full-space noise disc.
- MDE (planted-spike simulations spike_R.py and spike2_R.py: per-text FDT products with SD 4.30 / 1.25 / 1.92 along u* and 0.30 / 0.31
  / 0.19 along random directions (B-Q7 at blocks 14 / 18 / 6), a 16-dim high-variance subspace, a non-normal rank-1 spike, n per half
  1,792 / 896 / 896; one split, both assignments):
  - lambda_RSI^cf (4 x 4): unbiased (planted 0.5 -> 0.50, 1.0 -> 1.00), SD 0.012-0.03. So 'lambda_RSI below 1' holds with power 0.8
    for a true value <= about 0.93-0.97; 'lambda_RSI >= 1' needs >= about 1.03-1.07; 'near critical' needs >= about 0.83-0.87.
  - lambda_max^C: permutation-null 95th percentile about 0.07-0.09. Power to resolve the top mode at a true rightmost eigenvalue of
    0.1 / 0.2 / 0.3: 0.30 / 0.50 / 0.97 at block 14, 0.10 / 0.83 / 1.0 at block 18, 0.07 / 0.87 / 1.0 at block 6. Near threshold the
    held-out value is biased down by about 0.05-0.1 (selection on a noisy half: planted 0.3 reads 0.23 / 0.29 / 0.26).
  - The difference lambda_max^C - lambda_RSI has an SD of about 0.16 / 0.08 / 0.085, so 'RSI subdominant' needs a true difference of
    about 0.40 / 0.19 / 0.21 (blocks 14 / 18 / 6).
  - Full space: under the null, the in-sample rightmost eigenvalue is about 0.37 / 0.45 / 0.32. That is the circular-law edge of the
    noise, 0.30 sqrt(2048 / 1,792) = 0.32 at block 14 (0.47 and 0.29 at 18 and 6), widened by the high-variance subspace. A full-space
    top mode is resolvable only above about 0.5 (blocks 14, 18; power 0.975 / 0.875 at 0.5) or 0.3 (block 6; 0.85). Revision 1's
    power iteration would have reported this noise edge as lambda_max.
  - The median over 20 halvings will be somewhat tighter than one split; the bootstrap measures the actual spread.
  - Round 2 measured lambda_lin(u_T_perp) = -0.033 [-0.275, 0.190] at block 14 (128 texts), and the dose gains along u* are 0.29 /
    0.13 / 0.15.
- Inputs: JLOOP(b) (R4.1), WORDSETS, TXT0, R2.2 DN d_4 (copied in from r1b_R_S2_*).
- Runs in: r1b_R_S3 (read directions and in-session eigen-analysis: 7.7 A100-min with margin), r1b_R_ana (cross-fit, nulls, bootstrap).
- Gates: RG5. RG10 does not apply (no lens is used here).

**R4.3 Direct forward check of the Jacobian**
- What: do pushes along v_max^C and v_RSI through the real loop return the gains R4.2 predicts?
- How: per block, cells along v_max^C and v_RSI (in-session directions of K2 halving 0's fit half A: v = y_A, w = z_A; sign fixed so
  that the cosine with u*_b, respectively u_T^v, is >= 0; lambda_FDT = z_A' J_B y_A / (z_A' y_A), the held-out value; R.9 j) at the
  PUSH-DN dose, 512 pairs each, keys ('r1b_fdtchk', b, which, q, s). lambda_direct(v) = w . d /
  (a cbar w . v), with w the matching left vector; if |cos(v, w)| < 0.2, the Rayleigh read v . d / (a cbar) is used ('non-normal:
  Rayleigh read').
- Rule: 'FDT confirmed by a direct push' if the 90 % interval of lambda_direct - lambda_FDT contains 0. Otherwise R4.2's verdicts at b
  carry 'FDT not confirmed by a direct push'. The v_max^C cell reads 'not decidable (top mode not resolved)' when R4.2 says so.
- Null / controls: R3.3's random-direction cell.
- Shows - cannot show: consistency of the linear response with a finite push (0.05 nats); not linearity at other sizes (R3.3).
- MDE (est.): SD of lambda_direct about 0.03-0.05 at 512 pairs; the check is limited by lambda_FDT's SD.
- Inputs: R4.2 (in-session directions).
- Runs in: r1b_R_S3 (7.1 A100-min with margin).
- Gates: RG5, RG9.

### R5 Lens readout

**R5.1 Bridge: does the J-lens predict the measured sign-true shifts? (layer profile)**
- What: at which blocks does the J-lens predict the measured odd (sign-true) shift of the next-token distribution under a push, so that
  a lens readout can replace gradient readouts there?
- How:
  - **Lens.** LENS-J is downloaded from neuronpedia/jacobian-lens (path above) and checked against its sha256; otherwise 'not run
    (lens checksum failed)'.
  - **Index convention.** The J-lens index convention (J[l] applied to the output or to the input of block l) is the one that K4j
    (G-K-J) records in r1b_K_S1 (k4j.json, copied via dsse_runs_B; a start condition of r1b_R_S5, main §4.2). R5.1-R5.3 use it and
    record it.
  - **Axes.** One unstopped all-block read of the AX-1b axis texts and of the 80 NEG sentences (needed for u_neg at the other
    blocks) gives AX-1b at the 8 lens blocks. cap_b and med_b come from R1.state[b]['ref'] at 14, 18 and 6, and at the other blocks
    from 2 x the median norm over the continuation tokens of 64 TXT0 texts.
  - **Blocks** {2, 6, 10, 14, 18, 22, 24, 26}.
  - **Directions.**
    - At 14, 18 and 6: u*_b, unit(u_T^v(b)), unit(u_sea^v(b)) and ('cov', b, 4).
    - At the other blocks: unit(u_T_perp^1b(b)), unit(u_sea_perp^1b(b)), unit(raw u_tone(b)), and an isotropic random unit vector
      (torch.Generator(500 + b)).
  - **Doses** {0.25, 0.5, 1, 2} x a_DN(b, u) (each direction's own 0.05-nat scale at that block), both signs.
  - **Texts.** Teacher-forced on 64 TXT0 texts (seed 0 of the 64 read-direction prompts of R4.2), with full logits at continuation
    positions (lm_head on those positions only).
  - **Measured odd shift** of a word set S: the mean over positions of (log p+(S) - log p-(S)) / 2. The even part is reported.
  - **Lens prediction** (first order, forward only):
    - dlogits(t) = W_U [g . (I - x_t x_t' / (D rms_t^2)) J_b delta] / rms_t, with x_t the unpushed block-27 output, rms_t its RMS, g
      the final-norm gain, and delta = a cbar_b unit(u).
    - The J-lens sums each source position's effect over later targets. For a constant push at all positions this equals, under
      stationarity, the summed effect of all earlier pushes on one target.
    - The predicted odd shift of S = the sum over w in S of q_t(w) dlogits_t(w) - the sum over v of p_t(v) dlogits_t(v), averaged.
  - **Sets:** W_RSI-eval, W_sea, W_sports, W_tone and 20 neutral sets (24 sets).
  - **Secondary readout:** the per-position Pearson correlation between measured and predicted centred odd logit shifts over the
    1,000 most probable tokens.
- Rule (per block, pooling the directions and the doses <= 1 x a_DN):
  - beta_b = the slope of measured on predicted odd shifts through the origin, over 24 sets x 4 directions x 3 doses, with a 90 %
    text-bootstrap interval (texts as units, 2,000 replicates). Also R2_b, and the sign agreement over (set, direction) cells whose
    measured odd shift exceeds 2 SE.
  - 'Lens valid at b' iff beta_b's interval lies within [0.5, 2.0], R2_b >= 0.5 and sign agreement >= 0.8; otherwise 'lens not valid
    at b'. R2_b is the centred R^2 of the through-origin fit, 1 - SSres / sum (y - ybar)^2 (R.9 g).
  - The layer profile over the 8 blocks is reported in full: beta_b, R2_b, sign agreement, token correlation and the deviation
    1 - R2_b.
- Null / controls: random directions; neutral sets; the even part (nonlinearity) beside the odd part.
- Shows - cannot show: where a fixed averaged Jacobian predicts the use of a push in this model and these texts. It supports no causal
  claim beyond first order, and the lens' own fit data are WikiText, not chat texts.
- MDE: teacher-forced shifts are deterministic per text, so the prompt-bootstrap SD of beta_b is expected well below 0.1 (est.). The
  rule's factor-2 tolerance is far above it, so a failure means real misprediction, not noise.
- Inputs: LENS-J, k4j.json (K4j's index convention, r1b_K_S1), AX-1b (lens blocks), TXT0, R3.3 directions.
- Runs in: r1b_R_S5 (12.7 A100-min with margin).
- Gates: RG10 (defines it).

**R5.2 Lens replacement of the A3 readout**
- What: does the RSI direction move the propensity to say 'improve' rather than 'describe', read by the lens where the gradient readout
  of A3 vanished (blocks 22-27)?
- How:
  - A3's stimuli and contrast as frozen in round 2 (r2_stimuli / r2_rgi_geom: 'In one sentence: what would you do with {x}?',
    assistant prefix 'I would', IMP minus DES lexemes, base forms).
  - Readout per block b in {6, 10, 14, 18, 22, 24, 26}: t_lens(u) = the predicted change of the mean next-token logit difference
    IMP - DES at the answer position, per unit push along u applied at all positions (R5.1's formula and J-lens index convention).
  - z_lens = (t_lens(u_T_perp^1b(b)) - mean) / SD over 2,000 covariance-matched random directions (drawn with TXT0's state covariance
    at b).
  - Direct check, forward only: finite differences (+- a_DN) of the same logit difference along u_T_perp^1b(b) and along 20
    covariance-matched random directions (10 under RL2), pushed at all positions and, separately, at the 6 object positions only
    (A3's positions).
- Rule (per block):
  - At blocks where the lens is valid (R5.1): 'u_T_perp moves improve-vs-describe (lens)' if z_lens >= 3, else 'no response (lens)'.
    The label 'lens and direct push disagree in sign' is added if the finite-difference shift has the opposite sign.
  - At other blocks: 'not readable (lens not valid at b)', with the finite-difference values reported.
  - Object-position finite differences are descriptive; they test whether A3's vanishing near the output is real.
- Null / controls: covariance-matched random directions. R0's centred nulls are not used here (the readout is not a cosine test).
- Shows - cannot show: the readout A3 could not give near the output, for the all-position push the loop uses. It is position-averaged
  and cannot reproduce A3's object-specific (owner, substrate) structure.
- MDE: z_lens is analytic given the null spread; the finite-difference SD over A3's objects is measured (no prior estimate).
- Inputs: LENS-J, R5.1 validity, A3 stimuli (round-2 code), AX-1b (lens blocks), TXT0.
- Runs in: r1b_R_S5 (2.6 A100-min with margin).
- Gates: RG10.

**R5.3 MicroLens use test (T4) and lens-metric foreign share (T3), with the nulls the probe lacked**
- What: does pushing along u_T_perp raise RSI words specifically, or does its euclidean orthogonalisation push the model toward the
  control topics (the probe's 'anti-component')?
- How (block 14):
  - RSI directions: unit(u_T_perp^1b), unit(u_T^v), u_T^J and raw unit(u_T). u_T^J is u_T made M_J-orthogonal to span(sports, tone,
    sea, u_neg) by Gram-Schmidt in the M_J inner product (no matrix inverse), with M_J = R_J' R_J, R_J = W_U[letter/digit tokens]
    diag(g) J_14 (K4j's index convention, R5.1), centred over the vocabulary.
  - Positive controls: unit(u_sea), unit(u_sports), unit(u_tone).
  - Nulls: 20 covariance-matched null directions (10 under RL2).
  - All directions are calibrated to dmg_sym 0.05 nats and pushed with both signs, teacher-forced on the 64 R5.1 texts.
  - Readout: odd shifts of W_RSI-eval, W_sea, W_sports, W_tone (no '!') and the 20 neutral sets, with text-bootstrap intervals.
  - Generated text: word-count odd differences in R3.3's PUSH-DN cells along u_T^v and u_sea^v.
  - T3: the share of the M_J-norm of u_T_perp in span(sports, tone, sea, u_neg) under M_J, against 200 centred pseudo-axes of the RSI
    axis texts (text-contrast nulls) projected identically. Cross-validated: the axis comes from fold A, and the share is evaluated
    with fold B's pseudo-axes.
- Rule (T4, per RSI direction):
  - 'specific in use' iff all three hold:
    - the RSI set's odd shift has a lower bound above the 95th percentile of the null directions' RSI shifts (with 10 nulls: above
      their maximum);
    - every foreign set's odd shift lies within the null directions' 5-95 % range for that set;
    - the neutral sets lie within the null directions' 5-95 % range.
    Any push moves every word set a little, so a foreign set is compared with equal-damage nulls, not with 0.
  - 'pushes foreign topics' if some foreign set's shift lies outside that range (sign reported; 'anti-component' if it is opposite to
    that topic's positive control).
  - A foreign set whose own positive control does not move it (lower bound <= the null 95th percentile) is 'not sensitive', and its
    specificity part is 'not testable'.
  - T3: descriptive.
- Null / controls: equal-damage null directions, 20 frequency-matched neutral sets, a positive control per foreign axis, and
  text-contrast null axes for T3.
- Shows - cannot show: whether u_T_perp's use is RSI-specific at matched damage (two-sided specificity), first order in effect size
  through teacher forcing, at one block.
- MDE: not estimable before the run beyond the probe's single numbers (RSI words +0.74 nats at 0.05 nats); the null spread is measured.
- Inputs: LENS-J (for u_T^J and M_J), AX-1b(14), TXT0, DOSE-1b(14), WORDSETS.
- Runs in: r1b_R_S5 (3.6 A100-min with margin).
- Gates: RG10 (for u_T^J and T3), RG9.

**R5.4 Conditional bf16 refit of the lens**
- What: a lens fitted on in-distribution text, in case the Neuronpedia lens fails the bridge.
- How:
  - Runs only if R5.1 gives 'lens not valid' at block 14 or at all of {22, 24, 26}.
  - Refit with the jlens reference code (anthropics/jacobian-lens jlens/fitting.py at a pinned commit, Apache-2.0) in bf16; fp32
    would take about 9-12 h.
  - Data: 300 unsteered 128-token chat-format texts from P256 prompts (keys ('r1b_lensfit', q, s), generated in the same session);
    200 texts under RL1 or RL2. Final-block target, positions 16..T-2.
  - dim_batch 64. 128 would replicate a prompt of about 150 tokens 128 times and need about 40 GB of bf16 activations; the jlens
    docstring states that the total backward FLOPs do not depend on dim_batch. dim_batch 128 is allowed only on an 80-GB card.
  - Then R5.1 is repeated with LENS-J1b, and R5.2's lens readout and R5.3's u_T^J / T3 are repeated at the blocks where only LENS-J1b
    is valid.
- Rule: R5.1's rule. Every readout from LENS-J1b carries 'refit lens (second attempt)'. Both profiles are reported.
- Null / controls: as R5.1.
- Shows - cannot show: whether in-distribution fitting repairs the lens. A bf16 lens is used only for readouts, never for compared
  generations.
- MDE: as R5.1.
- Inputs: LENS-J's failure profile, P256.
- Runs in: r1b_R_S5L (conditional; 97.5 A100-min with margin, plus setup).
- Gates: RG11.

**R5.5 What the style mode encodes: inverse-free, verified label (descriptive)**
- What: a verified name for u*_b (and for v_max^C), since 'style mode' is so far only a set of cosines.
- How (Aletheia-style generate-verify-revise, forward only):
  - **Bank.** A frozen bank of 24 instruction pairs, written before the freeze: tone enthusiastic / flat (as round 1); formal /
    casual; long elaborate / very short sentences; concrete sensory / abstract general; first person / impersonal; list / prose;
    optimistic / pessimistic; confident / hedging; poetic / plain; technical / plain; questions / statements; humorous / serious; sea
    (round 1); sports (round 1); nature imagery / none; past / future tense; dialogue / narration; metaphorical / literal; emotional /
    neutral; simple / sophisticated vocabulary; how-to instructions / description; storytelling / exposition; addressing 'you' / third
    person; exclamatory / calm.
  - **Generator.** Candidates are ranked by a cheap rollout (16 prompts x pro / anti, keys ('r1b_lab', cand, q)) and by the cosine
    between the lens readout of that quick axis and u*_b's lens readout (only at blocks where R5.1 says the lens is valid).
  - **Verifier, decoupled from the generator.** The candidate's full axis from 64 prompts x pro / anti, read by the unmodified model
    at block b. Accept if the cross-fitted, disattenuated |cos(candidate axis, u*_b)| is >= 0.7 and exceeds the 95th percentile of
    200 centred pseudo-axes of the candidate's own texts ((D - mean D) / N sign-flip combinations, projected identically). These are
    the 1b nulls; revision 1's label-shuffled axes were uncentred sign-flip nulls.
  - **Reviser.** If no candidate is accepted, pair the two best candidates (instructions concatenated) and verify again. The attempt
    limit is 3 revisions per target; then 'abstain'.
  - **Verifier error rate.** The same procedure is run on two known targets (u_sea and u_tone at block b, held-out prompts) and must
    return the sea and tone pairs.
  - v_max^C comes from r1b_R_S3's in-session analysis (R4.2).
- Rule: descriptive. 'verified label: <pair>' or 'abstained after 3 revisions'. Attempts, abstentions and the verifier's error rate on
  known targets are reported. If the procedure fails on both known targets, every label it returns is marked 'unreliable verifier'.
- Null / controls: centred pseudo-axes; known targets.
- Shows - cannot show: the best verified description from the bank. It cannot find a description outside the bank, and acceptance
  means 'reproduces the direction', not 'is the only description'.
- MDE: a disattenuated cosine SD of about 0.05 at 64 prompts (est.).
- Inputs: u*_b, R4.2's v_max^C (r1b_R_S3 output), LENS-J (ranking only).
- Runs in: r1b_R_S5 (3.2 A100-min with margin).
- Gates: RG10, for the lens ranking only.

### R6 Text loop against latent loop; in-context carry over several answers

**R6.1 Text against latent (soft-token) loop at matched damage, per axis**
- What: how much of the loop's gain is lost or added by the sampling of discrete tokens? One pass per axis at the same push.
- How:
  - **Why tokens are the channel.** The only route from a block-b push back to block-b states of later positions is the token
    sequence: blocks <= b see earlier positions only through keys and values computed below the push.
  - **The latent loop** keeps this channel but removes sampling.
    - At each of 40 steps, the next input is the expected embedding sum_v p_t(v) E[v], with p_t = softmax(logits_t / 0.7) of the
      pushed model (tied embedding; no keys; deterministic).
    - Survival weights are S_t = prod_{t' < t} (1 - p_t'(stop)).
    - The unsteered model reads the plain prompt plus the 40 soft embeddings (inputs_embeds), stopped at block b: x_soft = sum_t S_t
      h_b(t) / sum_t S_t.
    - Coconut-style feedback of the final hidden state as input was considered and not used. The model was never trained on such
      inputs, and the norm and distribution mismatch at the embedding would confound any gain difference.
  - **Cells per block and axis u.** At block 14, the four R3.3 directions (u*_14, unit(u_T^v), unit(u_sea^v), ('cov', 14, 4)); at
    blocks 18 and 6, u*_b and unit(u_T^v) (the R3.3 directions there).
    - Push +- a_DN(b, u) (PUSH-DN, so the damage equals the text loop's), 256 prompts.
    - g_soft(u) = unit(u) . d_soft / (a cbar); G_soft from the U statistic over prompts.
    - g_text(u) = R3.3's PUSH-DN cell of the same (b, u), on the same prompts.
  - **Iterated latent loop:** 4 rounds at PUSH-DN from unit(u_T^v) and from RND_1's start, at each block, read like R2. Only at block
    14 under RL1; dropped under RL2.
- Rule (per block and axis):
  - rho = g_soft / g_text (paired prompt bootstrap) is defined only if g_text's 90 % interval excludes 0. Otherwise 'not decidable
    (text gain not resolved)' (code 'NaN:denominator_interval_includes_0').
  - When defined:
    - 'latent loop stronger' if the interval of log rho lies above log 1.25;
    - 'text loop stronger' if it lies below -log 1.25;
    - 'equal within 25 %' if it lies inside (-log 1.25, log 1.25);
    - otherwise 'undecided'.
  - 'latent loop reaches 1 on RSI' if g_soft(u_T^v)'s lower bound > 1; 'latent loop below 1 on RSI' if its upper bound < 1. These do
    not need rho.
  - Iterated loop: 'latent loop style-attracted' if its debiased |cos(d_4, u*_b)| >= 0.5, and A on u_T^v as in R2 (descriptive).
- Null / controls: the random direction; the text loop on identical prompts and pushes.
- Shows - cannot show: whether the discrete text channel is the bottleneck of the loop. The 'latent loop' is operationalised as the
  smallest change; other latent designs may differ. Soft inputs are off-distribution to an unknown degree.
- MDE (est.): along u*_b (g about 0.3, SD about 0.012 at the DN dose), rho is resolved to about +-10 %. Along u_T^v (g about
  0.02-0.05, SD about 0.025), g_text's interval is expected to contain 0, so rho is expected to read 'not decidable (text gain not
  resolved)' there (stated in advance). The RSI-axis readouts are then the two g_soft strings.
- Inputs: DOSE-1b, AX-1b, u*_b.
- Runs in: r1b_R_S6 (30.1 A100-min with margin).
- Gates: RG5, RG9.

**R6.2 In-context carry K over several follow-up answers on the RSI axis**
- What: does an RSI stance written in one answer carry into the next answers when the unsteered model continues its own text
  (convention-free, P2 4.0)?
- How (block 14; short contexts only, at most about 360 tokens by turn 6):
  - **Turn 1:** push +- a_DN along unit(u_T^v(14)), at T 0.7 (the R3.3 PUSH-DN cell, reused) and at T 1.0 (a new cell, keys
    ('r1b_k1', 14, 1.0, q, s)); 256 prompts x 2 seeds = 512 pairs per T.
  - **Turns k = 2..6:** no push. ids = PIDS + t_1 + SUF + ... + t_(k-1) + SUF with B-Q8's concatenation (TURN2_SUFFIX; history think
    blocks kept, 4 off-template tokens per history turn, as round 2). Keys ('r1b_turn', 14, T, k, q, s) are shared within a pair, at
    temperature T, re-prefilled each turn.
  - **Reads.** Context-free (the turn-k answer as the reply to the plain prompt, B-Q8's convention; primary) and in-context
    (generation-time capture at block 14; exact because turns >= 2 are unpushed).
  - **Statistics.** D_k = the cross-fitted projection of d^(k) on u_T^v(14): the pair-level sum of a_{other fold}(p) . delta_j over
    N_ok, as in R.1. Then K_k = D_k / D_1 (cumulative) and kappa_k = D_k / D_(k-1); the same along u*_14 (descriptive). Round-trip
    failures per turn are reported.
  - **Random-direction carry control.** Turn 1 = R3.3's PUSH-DN cell along ('cov', 14, 4) at T 0.7 (reused); turns 2-3 new (keys
    ('r1b_turn_rnd', 14, k, q, s)); K along the random direction itself (fixed-axis projection). This compares RSI carry with generic
    carry.
- Rule (per T):
  - K is defined only if D_1's 90 % interval excludes 0; otherwise 'K not defined (turn-1 RSI stance not resolved)'.
  - Primary K_2: 'self-sustaining at T' if its lower bound > 1; 'dissipative at T' if its upper bound < 1; else 'undecided'. At T 0.7
    these are members of F+ and F-.
  - Secondary K_geo = exp(slope) of a weighted least-squares fit of log D_k on k (diagonal weights, the whole fit bootstrapped) over
    the turns whose D_k interval lies above 0. It needs >= 3 turns, else 'K_geo not estimable'; same thresholds.
  - The denominators D_1 are always printed with their intervals.
  - Random control: 'not decidable (random-direction K not defined)' if the random D_1's 90 % interval contains 0 (code
    NaN:denominator_interval_includes_0; R.9 i); otherwise 'RSI carry above generic carry' if the paired interval of K_2(u_T^v) -
    K_2(random) at T 0.7 lies above 0; 'RSI carry below generic carry' mirrored; else 'not distinguishable from generic carry'.
- Null / controls: the paired construction (the - side); the random-direction carry; the u*_14 carry. Round 2's K 0.45-0.48 was along
  u*_14 after a u*_14 push; here the push is along RSI, so the two are not the same quantity.
- Shows - cannot show: the carry of an RSI stance through the model's own context over five follow-ups at short length. Nothing about
  long contexts (part D) or about other blocks.
- MDE (est.):
  - D_1 is about 0.81-0.86 (from round 1's RSI_a projection 1.30 at 0.1135 nats, scaled to 0.05 nats), with SD about 0.048 at 512
    pairs.
  - SD(K_2) is about 0.06-0.08: 'dissipative' is detectable when the true K_2 <= about 0.80, 'self-sustaining' when >= about 1.2.
  - Seeds buy little here, because prompt-level noise dominates: going from 2 to 8 seeds lowers SD(D_1) only from 0.048 to 0.040.
  - The random-direction D_1 has an unknown size, so the control's MDE is measured.
- Inputs: DOSE-1b(14), AX-1b(14), r2_stimuli TURN2 pieces.
- Runs in: r1b_R_S6 (22.8 A100-min with margin).
- Gates: RG12.

**R6.3 Replication of round 2's T 1.0 carry cell**
- What: was round 2's K on u_T_perp = 1.053 [0.743, 1.538] at T 1.0 noise? That cell pushed along u*_14 at alpha_0, and its
  denominator's interval was not reported.
- How: the round-2 cell rebuilt with new keys and 4 x the prompts. Push +- alpha_0(14) = 0.15 along u*_14 at T 1.0, 256 prompts x 2
  seeds, turn 2 as in B-Q8 (keys ('r1b_krep', q, s), ('r1b_krep2', q, s)). K is read on R1's u_T_perp(14) (round 2's read axis) and
  on u_T^v(14), each with its denominator's interval, and along u*_14.
- Rule (on R1's u_T_perp, the replicated quantity; u_T^v reported):
  - 'K not defined' if the denominator's interval contains 0;
  - 'round-2 T 1.0 cell not replicated (dissipative on u_T_perp)' if K's upper bound < 1;
  - 'replicated, self-sustaining (K lower bound > 1)' if K's lower bound > 1;
  - 'replicated (K >= 1 not excluded)' if the interval contains 1.
- Null / controls: the paired construction. K along u*_14 in the same cell (round 2: 0.45 [0.37, 0.53] at T 1.0) is the internal
  comparison.
- Shows - cannot show: whether the single round-2 cell that allowed self-sustainment on the RSI axis was a small-denominator artefact.
- MDE (est.): SD(K) about 0.13 at 512 pairs (round 2: 0.26 at 128); 'not replicated' when the true K <= about 0.68.
- Inputs: u*_14, R1.state[14]['axes']['u_T_perp'].
- Runs in: r1b_R_S6 (2.2 A100-min with margin).
- Gates: RG12.

### R7 Low-damage erasure

**R7.1 Erasure variants selected by damage alone; C-Q3's rule**
- What: can u_RSI be removed from every write at <= 0.05 nats, so that re-search (C-Q4a / C-Q4b) becomes testable?
- How (block 14 process):
  - u_RSI = unit(R1.state[14]['axes']['u_T_perp']), round 2's object; a u_T^v variant is reported descriptively.
  - Hooks on the outputs of embed_tokens, self_attn.o_proj and mlp.down_proj (WriteOrth's sites): x' = x - beta Q Q' (x - mu_m), with
    mu_m = 0 (zero ablation) or the module's mean output over all token positions of 64 TXT0 texts (mean ablation).
  - **Ten variants in a frozen strength order:**
    - E1: mean, all blocks + embedding, beta 1;
    - E2: zero, blocks 8-20, beta 1;
    - E3: mean, blocks 8-20, beta 1;
    - E4: zero, all, beta 0.75;
    - E5: mean, all, beta 0.75;
    - E6: zero, blocks 11-17, beta 1;
    - E7: mean, blocks 11-17, beta 1;
    - E8: zero, all, beta 0.5;
    - E9: mean, all, beta 0.5;
    - E10: mean, blocks 11-17, beta 0.5.
    Round 2's C-Q3 (zero, all, beta 1, 0.1095 nats) is excluded.
  - For every variant: damage (16 references) and the neutral top-1 change rate (8 scene references).
  - **Selection, before any gap is computed:** the first variant in the order with damage <= 0.05.
  - **For the selected variant:**
    - 256 x (pro + anti) RSI-instructed texts, with the keys of their unablated AX-1b references. The references are the A100
      regeneration of R2.1, so both sides of each common-random-number pair come from the same device;
    - 256 unprompted texts;
    - sea / sports / tone instructed texts (foreign axes);
    - H30 and RGI30 discrimination D;
    - the two 1-dim random controls ('rand', 14, 2) and ('cov', 14, 0) under the same variant (RSI-instructed texts);
    - the 16 null directions' D under the same variant (each control compared without itself, P2 O6).
  - Gaps are read on O1-style fixed axes from the unablated 1b instructed texts. A foreign axis X moves by m_X = |1 - g_abl^X / g_0^X|
    on its own fixed axes (P2 OD-C6). TOPIC is dropped, because part R has no RGI axis.
  - The other nine variants' instructed gaps are computed too and reported as descriptive.
- Rule (round 2's C-Q3 rule on 1b data, for the selected variant):
  - 'unlearned' iff all of:
    - g_abl <= 0.5 g_0;
    - the 90 % upper end of g_abl / g_0 <= 0.75 and below the 5th percentile of its bootstrap distribution under the random controls;
    - D_abl <= 0.5 D_0, with z_D <= -2.33 (D_0 > 0.2);
    - damage <= 0.05;
    - every foreign axis moves less than half as much as RSI.
  - 'inverted' as in C-Q3; otherwise 'not unlearned (<failed parts>)'.
  - 'no low-damage erasure found' if no variant reaches 0.05 nats.
  - The neutral top-1 change is printed next to every KL.
- Null / controls: random-direction ablations of the same variant; 16 null directions for D; the foreign axes.
- Shows - cannot show: specific removal of the instructed RSI ability at low damage by one direction; not every route (re-search tests
  that).
- MDE (est.): SD of g_abl / g_0 about 0.027 at 256 prompts (round 2: [0.066, 0.243] at 64). 'unlearned' needs a true ratio <= about
  0.48 (point rule) and <= 0.68 (upper-end rule).
- Inputs: AX-1b(14), TXT0, R1.state[14] ('axes', 'ref', 'k6_null<i>' directions via rk_dirs.pt), H30 / RGI30 (round-2 stimuli).
- Runs in: r1b_R_S7 (10.1 A100-min with margin).
- Gates: RG5.

**R7.2 C-Q4a re-search by cue (gated)**
- What: does a cue of three RSI example answers restore the erased ability?
- How: round 2's C-Q4a with the selected variant and 256 prompts. The cued texts carry the pro instruction plus three unablated
  RSI_PRO answers, those of prompts q+1, q+2 and q+3 within the prompt type (OD-C3 format).
- Rule: only if R7.1 says 'unlearned' (decided in r1b_R_S7):
  - 'shallow' if the restored share >= 0.5;
  - 'deep' if < 0.2 with an upper end < 0.35;
  - otherwise 'partial'.
  - If R7.1 did not pass: 'not applicable (nothing removed)', with the share descriptive.
- Null / controls: the unablated cued texts (the cue's ceiling); C-Q3-style unablated instructed texts.
- Shows - cannot show: how deep the erasure is against an in-context cue.
- MDE (est.): restored-share SD about 0.066 at 256 prompts (round 2: [0.691, 1.122] at 64, SD 0.13).
- Inputs: R7.1.
- Runs in: r1b_R_S7 (0.9 A100-min with margin).
- Gates: RG13.

**R7.3 C-Q4b loop under erasure (gated)**
- What: does the self-text loop re-find RSI through other routes once u_RSI is erased?
- How:
  - An ablated loop (writer under the selected variant) and an unablated twin. Start s0 = unit(d_1^{RSI_a, DN}(14) minus its u_RSI
    part).
  - 256 pairs, 3 rounds; PUSH-DN is calibrated with the ablation on. Both loops push unit(d_n orthogonalised against u_RSI) (P2 C14).
  - Keys ('r1b_Q4B', n, q, 0, 14) are shared. The unablated model reads; readouts are taken on u_T^v and on u_RSI's own axis.
- Rule:
  - The gate (RG14) is decided in r1b_R_S7 with in-session data only: R7.1 'unlearned', and the twin's first round from s0 has p_s <= 0.01
    and an A lower bound >= 0.10 on u_T^v(14). STAT-1b runs in-session; RG6's choice comes from r1b_R_S0.
  - If the gate is open, the ablated loop is read at d_2 and d_3:
    - 'un-searched' if p_s > 0.10 and |A|'s upper bound < 0.25, while the twin has p_s <= 0.01 and A's lower bound >= 0.25;
    - 'survives' if p_s <= 0.01 and A's lower bound >= 0.25;
    - 're-routed' if rel >= 0.5, Ue >= 0.25 with its interval above 0, and |A|'s upper bound < 0.25, where Ue_n = unit(u_T^v(14)) .
      d_n / Delta_T^v (AX-1b's text-unit gap); the rule reads u_T^v, u_RSI is descriptive (R.9 k);
    - 'undecided (twin does not re-find RSI)' if the twin at d_2 or d_3 fails p_s <= 0.01 and A's lower bound >= 0.25;
    - otherwise 'undecided'.
  - If the gate is closed: 'not testable (gate: <reason>)'.
- Null / controls: the unablated twin, with shared keys and the same pushes.
- Shows - cannot show: re-search through the loop, over three rounds only.
- MDE: as R2.4 at 256 pairs (A SD about 0.06, est.).
- Inputs: R7.1, R2.2.
- Runs in: r1b_R_S7 (4.6 A100-min with margin).
- Gates: RG13, RG14, RG9.

### R8 The Qwen3-0.6B question

**R8.1 Why no 0.6B block passed (model-free)**
- What: which part of the gate emptied round 2's 0.6B arm (RV A7.arm_0p6B.n_P = 0)?
- How: read dsse_runs/r2_q3/rgi/geom_stats_qwen3-0.6b.json:
  - gate.C0; gate.C0_parts ('P_cell_lo>0|IMP-DES', 'q>=0.2|IMP-DES', 'P_cell_lo>0|CHG-DES', 'q>=0.2|CHG-DES'); gate.uncomposed;
    gate.passing; gate.P; gate.n_P; gate.runs;
  - per_block.pooled ['FG|CHG-DES', 'FG|WOR-DES', 'P_cell|IMP-DES', 'P_cell|CHG-DES', 'q|IMP-DES', 'q|CHG-DES'] (point, lo, hi,
    n_nan_draws);
  - point_only['dot/white/pooled'], ['obj/raw/pooled'], lexical, anisotropy.dot, flags.
  - Also: geom_meta_qwen3-0.6b.json (hidden_size, tie_word_embeddings, stages.a1 lengths and g29, flags); geom_draws_qwen3-0.6b.npz if
    present; dsse_runs/r2_q3/logs/geom_06.log and stats_06.log; the same keys of geom_stats_qwen3-1.7b.json for comparison
    (r2_rgi_stats.py:712-716, 790).
- Rule:
  - Per block: 'C0 fails (<parts>)', 'uncomposed only (FG(CHG-DES) = x, FG(WOR-DES) = y)', 'passes', or 'NaN (<keys>)'.
  - Arm verdict:
    - 'empty because of the 0.8 uncomposed cut' if >= 12 contiguous blocks have C0 true and fail only the uncomposed cut;
    - 'empty because C0 fails (<dominant part>)' if no run of 12 contiguous blocks has C0 true;
    - 'empty because of NaN values' if NaN draws or non-finite points remove blocks from every run of 12;
    - otherwise 'mixed (<per-block table>)'.
  - Descriptive and post hoc: the run lengths with the cut at 0.85 and 0.90.
- Null / controls: the same keys of the 1.7B file (C0 true at all 28 blocks, uncomposed at 0-11, FG(CHG-DES) 0.706-0.787 at the passing
  blocks).
- Shows - cannot show: the cause within round 2's frozen gate. It changes no round-2 verdict and says nothing about 0.6B loops.
- MDE: a deterministic read.
- Inputs: as listed.
- Runs in: r1b_R_S0 (1.5 A100-min with margin).
- Gates: RG4.

**R8.2 A 0.6B loop as a size comparison (conditional)**
- What: does the smaller model show the same loop picture (generic attractor, sub-critical gains, RSI share) at the same relative
  depth?
- How:
  - Only if justified:
    - J1: R8.1 finds no NaN or stimulus problems;
    - J2: PC_PASS_1b holds at block 14 of Qwen3-1.7B (a comparison needs a working PC in the reference model);
    - J3: the 0.6B axes pass GO0_1b-style checks (computed first in the session).
  - Model: Qwen3-0.6B (revision c1899de289a04d12100db370d81485cdf75e47ca, model.safetensors sha256
    f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b), block 14 of 28 (asserted from the config). r1b_core is used
    (rk_loop loads only 1.7B).
  - References (256 texts, keys ('r1b06_ref', q)), cap and axes from P256 (keys ('r1b06_axis', name, q)); PUSH-DN on the 0.6B
    references.
  - Loops: RSI_a, RSI_b, SEA_t (512 pairs) and RND_1 (256 pairs), 4 rounds.
  - No round-2 style mode exists for 0.6B, so its PC axis is u_sea minus span(u_T, sports, tone, u_neg), with the lexical criterion
    and the RND contrast; this PC carries the label 'not style-free'. The 0.6B mode is computed afterwards by B-Q0's recipe.
- Rule: a descriptive size comparison:
  - one-step gains at PUSH-DN per axis;
  - the mode set, and 'generic attractor at 0.6B' if some RND loop has debiased |cos(d_4, u*_0.6B^(-RND))| >= 0.5 (leave-one-out);
  - A_bar on the 0.6B RSI axis with intervals.
  - 'not run (not justified: J1 / J2 / J3)' otherwise.
  - Two sizes are not a scaling law (stated).
- Null / controls: the 0.6B RND loops; the 1.7B block-14 loops as the reference.
- Shows - cannot show: whether the 1.7B picture is size-specific between two models of one family. Nothing about scaling. With the PC
  expectation of R2.3, J2 is expected to fail and R8.2 not to run (stated).
- MDE: as R2.4 at block 14 (the 0.6B model's noise is unknown; reported).
- Inputs: R8.1, R2.3(14), P256.
- Runs in: r1b_R_S8 (conditional; 14.4 A100-min with margin).
- Gates: RG15.

### R9 Report hygiene

**R9.1-R9.5 Provenance, codes, labels, distributions, missing round-2 files**
- What: every number traceable, every missing value coded, every gated result labelled, every mean shown with its distribution.
- How:
  - **R9.1 Provenance block** in every output JSON:
    - plan path and sha256; code commit;
    - sha256 of every imported module. rk_loop.py, rk_verdict.py, ufo_common.py and rsi_statements.py must equal RG3's frozen values;
      also r2_rk.py and the r1b_*.py modules;
    - the measurement's config as canonical JSON with its sha256;
    - sha256 of prompts_1b.json, WORDSETS, the neutral pool and the candidate bank;
    - model repo, revision and file sha256s; LENS-J sha256;
    - every input file with its source (shard or session), sha256 and Drive mtime;
    - GPU, CUDA, torch, transformers and numpy versions;
    - key namespaces and seeds; the reduction level and the round-level tier (1 or 2);
    - session name, attempt, start and end (UTC).
  - **R9.2 Codes instead of sentinel numbers:** 'NaN:empty_text', 'NaN:zero_norm', 'NaN:denominator_interval_includes_0',
    'NaN:fewer_than_95pct_finite_replicates', 'CENSORED:below_grid', 'CENSORED:above_grid', 'NA:gate_closed:<gate>',
    'NA:not_run:<reason>', 'NA:not_run:RL<k>', 'NA:not_run:tier2_unfunded', 'NA:not_run:session_failed_twice',
    'NA:conditional_not_triggered'. The value field is null, and the summary gives
    per-table counts of every code. (Round 2 printed alpha* = 0 as a value and left 9 NaNs unexplained.)
  - **R9.3 Labels.** Every verdict and number carries its label list: 'post hoc (R0)', 'descriptive', 'descriptive (<gate> failed)',
    'descriptive (fixed axes)', 'conditional run', 'bootstrap-t used (RG6)', 'damage out of band', 'lens not valid at b', 'refit lens
    (second attempt)', 'self-overlapping', 'confirmatory (Holm)' / 'exploratory (Holm not met)', 'non-normal: Rayleigh read' and
    'complex top mode'.
    - The summary prints the labels next to the values.
    - A check script fails the report if a verdict at a gated block appears without its label (round 2's R4 and the block-14 R5 / A7
      strings).
  - **R9.4 Distributions next to means:** n, SD, the 5 / 25 / 50 / 75 / 95 percentiles, min and max over the units, per prompt family
    (RSI-topic, scene). For damage and top-1 change, the per-text distribution. The effect frequency (share of prompts with
    b_p > 0). Per-prompt values are stored.
  - **R9.5** Copy round 2's Drive-only files unlearn_verdict.json and r2_manifest.json byte-identically into
    results/ju_space/runde2_ergebnisse/ with their sha256. Both were written by the r2_ana session and are found by name under
    dsse_runs/r2_ana/.
- Rule: RG17. An output without a provenance block, or with an unlabelled gated verdict, makes its session 'incomplete', and the
  analysis refuses to read it.
- Null / controls: n/a.
- Shows - cannot show: traceability; not correctness.
- MDE: n/a.
- Inputs: all outputs.
- Runs in: every session (R9.1-R9.4, inside the step costs); r1b_R_S0 (R9.5, 0.8 A100-min); r1b_R_ana (report check, 1.5 A100-min).
  R9.1-R9.4 are applied round-wide by main SG6.
- Gates: RG17.

## R.4 Gates of part R

| # | Gate | Reads | Condition | If it fails | Decided in |
|---|---|---|---|---|---|
| RG1 | Plan freeze (= main SG1) | 1b plan sha256, freeze commit | equals the constant in the 1b runner; freeze commit ancestor of the code commit | session stops | every session |
| RG2 | GPU (= SG2) | runtime | A100 (sm80, 40 or 80 GB) | session stops | every model session, r1b_R_S0, r1b_R_ana, r1b_R_ana2 |
| RG3 | Round-1 code freeze (= SG3; R's constants) | sha256 of rk_loop.py, rk_verdict.py, ufo_common.py, rsi_statements.py in the checkout and, by git, at every round-1 shard's code_commit (gate_G3's method, r2_run_session.py:847-880) | all equal the frozen constants: rk_loop.py 1a20f61992637386c97197b7283a8b743214aa89028f504c4af6c2fbedbab2a1; rk_verdict.py 62aced20fcd7435201e1f427e5c294869fdab3adcb54801f092aa8d31ab9049e; ufo_common.py 6f3919e93b638a325f9a8dd5762430c6b5e7693d4e1f5a317d97366caa49f9ab; rsi_statements.py 6ff560c5b9c092858c7b50c4b041b7c886329bb7a451a6738d6eaa3754e3bec2 (identical at HEAD c8257c2 and at round 1's gpu1 commit 740bfc0f) | session stops (R1 and open_block depend on them) | every session importing them |
| RG4 | Inputs present | R0 / R1 / R2 file lists | exist | that measurement 'not run (input missing)' | per session |
| RG5 | Code checks CC1-CC4 | session processes | as R2.2 (zero push identical 4/4, mirror >= 3/4, early stop < 1e-4, r1b_core = rk push to 1e-6) | verdicts using that arm 'descriptive (code check failed)' | every model session |
| RG6 | K1-s size | R0.7 | rejection <= 0.02 at nominal 0.01 in >= 90 % of rows | bootstrap-t replaces p_s in R2-R7 (labelled) | r1b_R_S0 |
| RG7 | GO0_1b | R2.1 | AUC(u_T^v) >= 0.85; AUC(u_sea^v) >= 0.80, else AUC(u_sports^v) >= 0.80 (SPO_t fallback) | 'not decidable (RSI axis not resolved)' / 'PC not testable' | r1b_R_ana |
| RG8a | PC_PASS_1b(b) | R2.3 | all of P1-P5 | positive RSI strings at b become 'undecided (positive control failed)'; delta rule needs >= 2 passing blocks | r1b_R_ana |
| RG8b | SENS_1b(b) | R2.3b | planted-signal pass rate >= 0.8 for both RSI loops | negative RSI strings at b become 'undecided (sensitivity check failed)' | r1b_R_ana |
| RG9 | Damage band | PUSH-DN logs | dmg_sym in [0.045, 0.055] | row label 'damage out of band'; > 1 of a loop's 4 pushes out: 'not decidable (damage calibration failed)' | per session |
| RG10 | Lens validity per block | R5.1 | beta interval within [0.5, 2], R2 >= 0.5, sign agreement >= 0.8 | lens readouts at b 'not readable (lens not valid at b)' | r1b_R_S5 (r1b_R_ana re-checks) |
| RG11 | Refit trigger | R5.1 | not valid at 14 or at all of 22, 24, 26 | R5.4 runs in r1b_R_S5L if tier 2 is funded (main §4.5, drop item 14); otherwise 'not run (tier 2 not funded)' | r1b_R_S5 (in session) |
| RG12 | K defined | R6.2, R6.3 | D_1's interval excludes 0 | 'K not defined' | r1b_R_ana |
| RG13 | Erasure passed | R7.1 | 'unlearned' | R7.2 'not applicable', R7.3 'not testable' | r1b_R_S7 (in session, OD-G8 style) |
| RG14 | C-Q4b twin gate | R7.1, twin round 1 | RG13 and the twin's round 1 from s0: p_s <= 0.01 and A lower bound >= 0.10 on u_T^v(14) (in-session STAT-1b) | 'not testable (gate: <reason>)' | r1b_R_S7 (in session) |
| RG15 | 0.6B justification | R8.1, R2.3(14), 0.6B axes | J1-J3 | 'not run (not justified: ...)'; R8.2 is tier 2 (main §4.5, drop item 6) | r1b_R_ana pass 1 (J1, J2), r1b_R_S8 (J3) |
| RG16 | Time and concurrency guard (= SG5) | session clock, launch log | the runner stops after the step that ends at >= 4.5 h since the notebook's first cell (SG5); steps <= 5 min and resumable; <= 2 concurrent sessions per account (an assumption, int_machinery) | session stops after the step; a continuation session resumes; launches staggered | every session |
| RG17 | Report check (= SG6, applied to every part) | every output | provenance and labels present | session 'incomplete' | r1b_R_ana, r1b_R_ana2 |
| RG18 | Reduction level | r1b_P0's measured rates (P0-R items) in cost_R2.py | the smallest reduction level whose projected total (conditional sessions included) is <= 12.5 h | the level is written into runde1b_freeze.json (part of main SG7) | before the freeze |

## R.5 Sessions, order, costs

**Rates.** Rates used (P2 7.1, planned 'given facts', not measured):
- generate call of 32 rows x 40 new tokens: 1.0 s (prompts <= 30 tokens), 1.3 s (about 80), 2.0 s (about 200); turn-k prompts are
  costed at 2.0 s;
- soft-token generation: 1.5 s per 32 x 40;
- reads: stopped at block 14 0.25 s per 32 texts, at other blocks 0.35 s, all 28 blocks 0.40 s, full pass with logits 0.45 s,
  20-token sentences 0.15 s;
- one short forward 20 ms; forward + backward 60 ms; one dmg_sym evaluation 0.64 s;
- model load 0.5 min per process; setup 4 min per model session (3 min analysis);
- Drive copy-in and writes both budgeted at 10 min per GB (a placeholder until r1b_P0 (P0-R items) measures Drive throughput);
- margin x 1.5 on work.

Arithmetic: scratchpad/r1b/scratch_R/cost_R2.py (GEN_SCALE and TIER as environment variables; TIER k selects RLk), mde_R.py,
joint_R.py, spike_R.py and spike2_R.py (JSON only, no model).

**Data volumes.** With the storage format of R.1, r1b_R_ana copies in 2.33 GB: r1b_R_S2_* pairs 1.06 GB, R3.3 0.42 GB, TXT0 and FDT 0.17 GB,
R4.3 0.05 GB, R6 0.3 GB, R7 0.07 GB, R1 0.2 GB, plus small results. The sessions write 2.95 GB in all. Revision 1's three full
read windows would have meant 5.9 GB (float32) or 11.8 GB (float64).

| Session | Account | Content | Work (min) | Setup + copy-in + writes (min) | A100-min with margin |
|---|---|---|---|---|---|
| r1b_P0 (P0-R items; was R-P; declared engineering pilot, before the freeze) | B | rates at 20 / 80 / 200 / 360-token prompts, reads, forward-backward, soft generation, RefKL, peak memory, Drive read and write throughput, replay openers on MOCK; SO-PILOT material (main §2.3), texts discarded | 10.0 | 4.0 + 0 + 0.2 | 19.2 |
| r1b_R_S0 (A100 runtime, no model) | A | R0, R8.1, R9.5; SO-AX package (main §2.3): AX1 at 14, 18, 6 from R1.state[b]['axes']; U* = r2_q2/theta/b<b>_q0.pt['u_star'] copied with sha256 (canonical); the leave-loop-out modes; SG10's check (u*_b rebuilt from dir_<L><SUF>_4 and the r2RSIb entries with TV's mode_set: cos >= 1 - 1e-4 with the file, TV image_cos_d1_u_star reproduced within 1e-4). Exports to dsse_runs/r1b_xfer (main §2.2): dmg_ref_R16.json (token ids of REF-R16 = R1.state[14]['ref'] continuations of round-1 prompts 0-7 and 32-39, checked against the copies at 18 and 6) and so_ax_ustar.pt (u*_b at 14, 18, 6). Inside the 22.2 min. | 9.5 | 3.0 + 4.5 + 0.5 | 22.2 |
| r1b_R_S1 | A | R1 closed loop (8 stage subprocesses) and open loop (3 processes) | 17.8 | 4.0 + 3.3 + 2.0 | 35.9 |
| r1b_R_S2_14 | A | R2.1 (2,048 axis texts), R2.2 at 14 | 33.1 | 4.0 + 1.2 + 3.7 | 58.5 |
| r1b_R_S2_18 | A | R2.1 reads, R2.2 at 18 | 34.4 | 4.0 + 1.5 + 3.7 | 60.8 |
| r1b_R_S2_06 | A | R2.1 reads, R2.2 at 6 | 34.4 | 4.0 + 1.5 + 3.7 | 60.8 |
| r1b_R_S3 | A | R3.3, TXT0 (R3.4), R4.1 scores, R4.2 read directions and in-session eigen-analysis, R4.3 | 59.0 | 4.0 + 1.0 + 6.7 | 100.2 |
| r1b_R_S5 | A | R5.1, R5.2, R5.3, R5.5 | 14.8 | 4.0 + 1.5 + 1.5 | 29.1 |
| r1b_R_S6 | A | R6.1, R6.2 (incl. the random-direction carry control), R6.3 | 36.8 | 4.0 + 1.0 + 3.0 | 63.2 |
| r1b_R_S7 | A | R7.1, R7.2, R7.3 (RG13 and RG14 in session) | 10.4 | 4.0 + 1.5 + 0.7 | 21.8 |
| r1b_R_ana (pass 1) | A | all part-R statistics and verdicts: joint bootstraps, SENS plants, R4.2 cross-fit and permutation nulls, Holm families, R9 report check | 19.0 | 3.0 + 23.3 + 0.5 | 55.3 |
| r1b_R_S5L (conditional, RG11) | A | R5.4 refit, repeated R5.1, R5.2 / R5.3 lens readouts at newly valid blocks | 65.0 | 4.0 + 1.0 + 2.5 | 105.0 |
| r1b_R_S8 (conditional, RG15) | A | R8.2 | 9.6 | 4.0 + 0.5 + 0.6 | 19.5 |
| r1b_R_ana2 (pass 2, conditional on r1b_R_S5L or r1b_R_S8) | A | R5.4 / R8.2 statistics, final report check | 4.0 | 3.0 + 1.0 + 0.2 | 10.2 |
| **Total** | | | | | **661.8 min = 11.03 h (8.78 h unconditional, 2.24 h conditional)** |

Round-level accounting (main §4.3): tier 1 = the unconditional sessions without the pilot, 507.8 A100-min (8.46 h), plus 4.0 min for
reporting the damage of R's calibrated pushes on REF-K16 (SO-DMG) = 8.53 h; tier 2 = r1b_R_S5L, r1b_R_S8 and r1b_R_ana2, 134.7 min
(2.25 h); R-P's 19.2 min are part of r1b_P0's budget. RG18's 12.5-h cap is unchanged.

**Order.** At most two concurrent sessions per account (RG16).
1. r1b_P0 (account B, before the freeze; main §4.2); then the freeze, which records the reduction level (RG18, SG7).
2. Wave 1 (A): r1b_R_S2_14 together with r1b_R_S0; r1b_R_S1 starts when r1b_R_S0 ends.
3. Wave 2 (A): r1b_R_S2_18 and r1b_R_S2_06, which need r1b_R_S2_14's axis texts. Then r1b_R_S3 (wave 3), which needs AX-1b at all three blocks and the
   DN d_4 for C_b.
4. Wave 4: r1b_R_S6 and r1b_R_S7 in account A; r1b_R_S5 also in account A, after r1b_R_S3 and r1b_K_S1's k4j.json (main §4.2).
5. r1b_R_ana pass 1, which decides RG7, RG8a, RG8b, RG12 and RG15's J1 / J2. r1b_R_S5L may start as soon as r1b_R_S5 has decided RG11. r1b_R_S8
   follows r1b_R_ana pass 1.
6. r1b_R_ana2 (pass 2) if any conditional session ran.

Account A holds every session that reads round-1 or round-2 Drive files. Every step is <= 5 min and resumable (cache per round / cell /
turn). The largest session (r1b_R_S5L, 105 min) is far below the 5-h limit, so float64 accumulators never cross sessions. Wall clock:
main §4.4.

**Budget argument.**
- At planned rates and the I/O placeholder, part R needs 8.78 A100-h unconditional plus 2.24 h for the three conditional sessions:
  11.03 h, within the 12-h envelope. About 7,900 generate calls carry most of the compute.
- The I/O placeholder accounts for 1.2 h. If r1b_P0 (P0-R items) measures Drive throughput >= 20 MB/s, that share drops to about
  0.1 h.
- At half the planned generation speed (cost_R2.py, GEN_SCALE 2) the total is 14.8 h at RL0, 13.1 h at RL1 and 12.3 h at RL2 (10.4 h
  unconditional).
- RG18 picks the reduction level from r1b_P0's measured rates (P0-R items) before the freeze.
- RL1 (if RL0 projects above 12.5 h):
  - RSI DN rounds 5-6 dropped (descriptive);
  - FX arm at 18 and 6 reduced to RSI_a (rung L3 loses SEA_raw and RND_1 there; descriptive);
  - R3.3 at 18 and 6 reduced to doses {0.25, 0.5, 1} x alpha_0 plus PUSH-DN. This changes verdict ingredients: k becomes 2 or 3 there,
    the shape fit loses the 0.35 alpha_0 cell, and the even-part check loses the 2 alpha_0 cell;
  - R6.1's iterated latent loops only at 14 (descriptive);
  - R5.4 on 200 instead of 300 texts (fit quality).
- RL2 (if RL1 still projects above 12.5 h), additionally:
  - the NP arm only at 14 (R3.2 at 18 and 6 'not run (RL2)');
  - R6.1's iterated latent loops dropped;
  - R5.2's direct-check controls and R5.3's null directions cut from 20 to 10. R5.3's 95th-percentile criterion then becomes 'above
    the maximum of 10' (about the 91st percentile).
- More A100 time would not buy much. About 46 % of the per-pair noise lies between prompts (54 % within, int_gates 5), so beyond two
  seeds only new prompts lower the SDs, and writing more than 192 new prompts is the binding cost.

## R.6 Outcomes that count against (and for) the hypothesis

**Against** "RSI content carries itself through the model's own text". Confirmatory family F-, Holm at family-wise 0.05; each is read
with its MDE:
- 'unlikely (RSI does not persist)' at a block where SENS_1b passes: A_bar_abs's upper bound < 0.25 on u_T^v.
- R3.3 'below 1 at small push' along u_T^v at each block.
- R4.2 'lambda_RSI below 1' at each block.
- R6.2 'dissipative at T 0.7' (K_2).

Exploratory, also against:
- 'unlikely (no first link)';
- 'RSI subdominant' (with the top mode resolved) together with 'RSI above neutral words' failing;
- R6.2 'dissipative at T 1.0', K_geo 'dissipative', 'RSI carry not above generic carry';
- R6.3 'round-2 T 1.0 cell not replicated';
- R6.1 'latent loop below 1 on RSI' (the discrete text channel is not what keeps RSI from carrying itself);
- R7 'unlearned' together with C-Q4b 'un-searched'.

**For** it. Confirmatory family F+, Holm at 0.05:
- 'likely' (either strength: 'RSI-dominated' or 'substantial RSI component') at a block where PC_PASS_1b passes;
- R3.3 'reaches 1 at small push' along u_T^v;
- R4.2 'lambda_RSI >= 1';
- R6.2 'self-sustaining at T 0.7'.

Exploratory, also for: 'near critical' together with 'RSI is the top mode'; K 'self-sustaining' at T 1.0; R6.3 'replicated,
self-sustaining'; C-Q4b 'survives' or 're-routed'; 'latent loop reaches 1 on RSI'.

The report prints the reached and Holm-confirmed counts of F+ and F- next to their expected counts under the null (R.1).

**Against round 1b's own repair claims:**
- R0 'README null claim not confirmed' (centring moves no round-1 p across 0.01).
- RG6 failing (the paired test is not valid at 64 prompts).
- R1 'replay differs' on the open loop. On the closed loop, divergence is expected.
- SENS_1b failing at a block: the design cannot see A = 0.25 in the RSI loops' noise there.
- PC_PASS_1b failing at all three blocks with GO0_1b met and SENS_sea passing. The loop then keeps no concrete topic beyond the generic
  mode, and only negative or undecided RSI verdicts are possible. That is itself a result: expected at block 14 from round-2 dynamics,
  new at all three.
- R4.2 'top mode not resolved' at every block: the FDT design cannot locate lambda_max at this sample size even within C_b.

## R.7 Changes from round 1

- (1) Centred pseudo-axes replace round 1's uncentred sign-flip nulls (reported only). Significance comes from a cross-fitted paired
  sign-flip test over prompts (K1-s), with Wilcoxon and t beside it and a size check that decides a bootstrap-t fallback.
- (2) Magnitude by a margin in interpretable units (90 % lower bound of A >= 0.10 / 0.25 / 0.50) instead of the N-independent cosine bar
  (Q_c >= 1, about 0.22-0.26 in c). MDEs are pre-registered jointly, for the rules as used.
- (3) Axis uncertainty is propagated by a joint prompt bootstrap of axis texts and loop pairs. Readouts use cross-fitted fold axes (no
  self-trace) and are disattenuated by the fold-axis cosine.
- (4) S_4 and R_4 are cross-fitted over halvings (round 2 found disattenuated cosines up to 1.219 and 1.468).
- (5) The positive control is redefined:
  - the topic axis has the generic mode removed only where round 2 found one (blocks 14 and 18);
  - a pooled lexical sea-word test, the SEA minus max(RND) contrast, and a SEA start on the verdict axis;
  - sports as the pre-registered fallback; the round-1 PC is reported beside it;
  - the PC now gates only the positive verdicts.
- (6) A planted-signal sensitivity check SENS_1b gates the negative verdicts.
- (7) RSI is read on u_T^v, so convergence onto a generic attractor cannot count as RSI. Style attraction is reported as a label.
- (8) Random-start controls start orthogonal to the style mode and the test axes.
- (9) A damage-normalised push per direction and round (0.05 nats, both signs logged, 8 RSI + 8 scene references) replaces the fixed
  alpha. GO3 becomes 'every push in band'. A norm-preserving control push is added.
- (10) 256 prompts (192 new, frozen before the run) instead of 64. RSI, SEA_t and SPO_t loops have 512 pairs, the others 256; RSI_b
  runs at every block.
- (11) A round-1-identical replay anchors an attribution ladder of single changes: byte-identical rk_loop.py, round-1 keys, run as
  round 1 ran it. Reproduction is judged on an open-loop replay along round 1's stored pushes.
- (12) The E rule: round 1 required the point value E >= 0.7. 1b defines E only if the interval of A_2^a + A_2^b excludes 0 and
  requires its 90 % lower bound >= 0.5. Bound and point conditions are named (R.1): mean rel >= 0.5, R_4^cf >= 0.5, S_4^cf >= 0.5 and
  D4 > 0 stay point conditions.
- (13) The verdict precedence is fixed as an ordered list (R2.4), extending round 1's if/elif chain (RVpy:91-98) by the SENS strings.
- (14) Multiplicity: two confirmatory families F+ and F- with Holm at family-wise 0.05; everything else exploratory.
- (15) New measurements:
  - a dose series below 0.5 alpha_0, in both signs, with unpushed texts;
  - no-push noise covariance;
  - lambda_RSI against lambda_max with read and use directions, cross-fitted, with permutation nulls and a candidate subspace;
  - a lens bridge with a layer profile;
  - a latent loop;
  - multi-turn carry on the RSI axis with a random-direction control;
  - low-damage erasure;
  - the 0.6B diagnosis;
  - a verified label of the style mode.
- (16) Report hygiene: provenance (code, config, prompt hashes, frozen module hashes), codes, labels and distributions in every
  output.

## R.8 Open decisions frozen now (defaults)

- OD-R1 Prompt count: 192 new prompts (256 in total). Beyond two seeds per prompt, only more prompts lower the noise (int_gates 5).
- OD-R2 Blocks: 14, 18 and 6 for every loop measurement.
  - Round-1 states, u*_b, theta cells and B-Q7 exist only there, so attribution is possible.
  - Block 6 is the only block R0 can reopen.
  - Block 18 failed on rel, which four times the pairs repairs (projected rel 0.76-0.86).
  - There is no new loop block: three blocks already cannot separate delta from depth, and a new block has no round-1 or round-2
    anchor.
  - Other blocks enter only R5 (lens profile at 2, 10, 22, 24, 26; A3 replacement at 10, 22, 24, 26).
- OD-R3 SEA start: unit(u_sea^v(b)) (SEA_t) decides the PC; the raw start (SEA_raw) is reported (int_gates asked to freeze this
  choice).
- OD-R4 RSI start: unit(u_T^v(b)) in R2 (all arms); the replay keeps round 1's u_T_perp.
- OD-R5 Damage target 0.05 nats (round 1's calibration target), band 0.045-0.055, references 8 + 8.
- OD-R6 Margins: PC A >= 0.10 (detectably on-topic); RSI margin levels 0.25 / 0.50 on A_bar's lower bound (round 1's 'tiers', now as
  bounds; renamed so that 'tier' keeps its round-level meaning, main C2);
  'unlikely' as an equivalence bound (upper bound < 0.25).
- OD-R7 Latent loop = soft-token (expected-embedding) loop; Coconut-style hidden-state feedback is not used.
- OD-R8 Carry: B-Q8's concatenation with history think blocks kept; the context-free read is primary; 6 turns; T 0.7 and 1.0; a
  random-direction control at T 0.7 over turns 2-3.
- OD-R9 Erasure strength order E1-E10, with selection by damage alone.
- OD-R10 Read-direction subspaces: k = 4 singular vectors of in-context gradients over 64 held-out texts. Candidate subspace C_b with
  k_C = 32 (composition in R4.2).
- OD-R11 Lens: the Neuronpedia J-lens first; a refit only under RG11 and only if tier 2 is funded, with dim_batch 64 on a 40-GB card.
- OD-R12 0.6B loop only under RG15 and only if tier 2 is funded; block 14 of 28.
- OD-R13 Verdict axes: perp* at blocks 14 and 18, ^1b at block 6, from round 2's R4 ('generic attractor' at 14 and 18, 'unaligned' at
  6).
- OD-R14 Negative RSI verdicts are gated by SENS_1b (planted A = 0.25, pass rate >= 0.8); positive ones by PC_PASS_1b.
- OD-R15 Confirmatory families F+ and F- (ten members each) with Holm at family-wise 0.05.
- OD-R16 Replay reproduction is judged on the open loop; the closed loop is descriptive.
- OD-R17 Reduction level chosen by RG18 from r1b_P0's rates before the freeze (main SG7).

## R.9 Implementation notes (code: analysis/ju_space/round1b/r1b_sessions_R.py, r1b_r_*.py)

**Rule changes adopted from the review hints (confirmed against this text).**
- (a) R1 (ii): the paired tolerance bounds |replay - round 1| (stats-16; the signed form let any negative deviation pass).
- (b) RND checks are two-sided (R.1 already reads RND loops two-sided): C_pos and string 7 use two-sided p_s and |A|'s lower bound;
  P5 uses max(|A_4(RND_1)|, |A_4(RND_2)|) (validity-2, stats-6). R0's 'not topic-specific' contrast uses |A_4(RND_1)| likewise.
- (c) K5 clause: string 6 uses mean z_beh(d_4) x sign(A_4) (round 1's ok_oscillation, rk_verdict.py:82-84); strings 4-5 unchanged
  (stats-4).
- (d) 'every RSI push in band' is dropped from C_pos and string 7; RG9's row label and string 3 (> 1 push out of band) stay (stats-9).
- (e) The F+ member 'likely (either strength)' tests A_bar's bound against 0.25 whichever string is reached (stats-11).
- (f) R3.3's u_T^v members read 'not decidable (RSI axis not resolved)' with p = 1 where GO0_1b fails (stats-12).
- (g) R5.1's R2_b is centred (stats-14). (h) R4.2 uses neutral sets matched to W_RSI-build (rng 4243; stats-15).
- (i) R6.2's random control gets 'not decidable (random-direction K not defined)' (stats-7).
- (j) R4.3: directions and lambda_FDT from halving 0's fit / held-out halves (stats-8, definitional part; the non-equivalence form of
  'FDT confirmed' is unchanged and stays open).
- (k) R7.3: Ue and the twin condition defined (stats-10).
- Descriptive additions only (rules unchanged): P4' = SEA_t's lexical difference minus max |RND lexical difference| (validity-7);
  per-set z-scores of R5.3's shifts against the null directions (validity-5 / stats-5).

**Code deviations from the text (no rule changes).**
- R2.2-R7 run with r1b_core (function style, in the session process: Hooks.push = rk's push, CC4) instead of r2_rk.open_block plus
  an r1b_rk.py; r1b_rk.py's functions live in r1b_r_lib.py (NP push, dmg_sym and calibration, generation-time cbar, windows) and
  r1b_r_s*.py. R1 uses rk_loop itself (closed loop as subprocesses, open loop as a library); CC4 is a script step in r1b_R_S2_14 that
  registers r1b_core's push on rk_loop's own model.
- Storage: per evaluation x+ and x- (float32, 'all' window; delta = (x+ - x-) / 2 is derived), 'div' / 'first20' as projections on F(b),
  token ids, cbar, calibration record, word counts of the fixed WORDSETS. Neutral-set counts are computed from the stored token ids
  (the neutral sets exist only after r1b_R_S3).
- R6.2's in-context read is a teacher-forced read of the full context (= generation-time capture up to kernel numerics; turns >= 2
  are unpushed).
- R4.2 read directions: g = d/dV of sum_t log p(S | prefix_t), V a plain leaf added at the output of block b from the first user token
  (one backward per (text, set); includes the cross-position terms).
- R5.4: own averaged-Jacobian fit (batched VJPs, dim_batch 64, bf16; sources = block outputs, so LENS-J1b's convention is 'output')
  instead of the jlens reference code.
- R0.7: the M rule's inner bootstrap keeps the axes and rel of its outer replicate (cost); the power model uses round-1 RSI rows only
  (R1's open-loop rows exist only after r1b_R_S0).
- R3.3: g_par along u*_b divides by sqrt(m*^2) (round 2's attenuation); R6.1 compares g_soft and g_text with one estimator (plain
  projection on unit(u)). R7.1: 'inverted' = the upper end of g_abl / g_0 below 0; foreign fixed axes projected off u_neg; unprompted
  texts reuse TXT0 seed-0 keys (common random numbers); D = mean y(pro) - mean y(anti), y = log p(Yes) - log p(No). R7.2: restored share =
  (g_cue,abl - g_abl) / (g_0 - g_abl). R5.5: the 24-pair bank wording is a draft (r1b_r_s5.BANK, package label_bank_1b.json).
- R8.1 reads gate.C0 / C0_parts / uncomposed per block (r2_rgi_stats layout); the 0.85 / 0.90 runs count 'uncomposed' as FG(CHG-DES) and
  FG(WOR-DES) above the cut.
- R9.5's copies are written as r95/<file>.orig (byte-identical, sha256 in r95.json); the coordinator commits them to
  results/ju_space/runde2_ergebnisse/.
- MOCK tests use 16 of the 256 prompts and rk_loop's MOCK round-1 states (8 prompts).

**Still open (main-level or design decisions, not changed here):** the F+ / F- composition (validity-1, stats-1), SENS_1b's
calibration criterion (validity-3, stats-2), the PC gate on positive strings (validity-4), the perp* premise (validity-6), rel for
'unlikely' (validity-8), R3.3's selection on significance (validity-9), R5.5's known targets (validity-10), string 8 (validity-11), the
'oscillation' wording (validity-12), MDE anchors (stats-3, stats-13), R5.3's multiplicity (validity-5, stats-5), 'FDT confirmed' as a
non-rejection (stats-8), the 12-h envelope against RG18's 12.5-h cap, and the 192 new prompts of prompts_1b.json (not yet written;
r1b_r_package.py builds and checks the file).

## Appendix R.A Neutral word pool (frozen)

The file results/ju_space/plans/runde1b/neutral_pool_1b.txt holds the words below, sorted, one per line, lower case, UTF-8, with '\n'
line ends including the last line: 628 words, sha256 4b3bcaca06f45521a00f12ec2b056a13962bbfdd066ab93867980c9b0be9ebf5. It was built by
scratchpad/r1b/scratch_R/neutral_pool_R.py. Excluded are:
- every concept list (W_RSI, W_sea, W_sports, W_tone);
- the A3 lexemes (improve / describe families);
- every word, in any inflection, of the eight round-1 axis instructions (rk.INSTR);
- near-synonyms for water, sports, positive tone, learning, AI, improvement and repair.

airport, almost, alone, already, always, ant, appear, apple, arm, arrive, artist, ash, ask, attic, aunt, autumn, away, baby, back,
bacon, bag, bake, baker, balcony, banana, bank, bark, barn, basket, bean, bear, bed, bee, beef, begin, believe, belong, bend, berry,
bicycle, big, bill, bird, birthday, bitter, black, blanket, blood, blue, bone, book, borrow, both, bottle, bottom, bowl, box, branch,
bread, break, breakfast, breath, brick, bridge, broken, brother, brown, brush, bucket, build, building, burn, bus, busy, butter,
button, buy, cabbage, cake, call, camera, candle, candy, car, card, carpet, carrot, carry, castle, cat, cellar, chair, chalk, cheap,
cheese, cherry, chew, chicken, child, chimney, choose, church, circle, city, clay, clean, clerk, climb, clock, close, closed, cloth,
cloud, coal, coat, coffee, coin, cold, collect, color, comb, common, continue, cook, cookie, cool, copper, corn, corner, cost, cotton,
count, country, cousin, cow, cup, curtain, cut, daily, dance, dark, date, decide, deliver, different, dig, dinner, dirt, dirty,
doctor, dog, door, double, downstairs, draw, drawer, dream, dress, driver, drop, dull, dust, each, ear, early, east, eat, edge, egg,
empty, engine, enter, envelope, evening, every, expensive, extra, eye, face, factory, fall, family, far, farm, farmer, father,
feather, feel, fence, festival, few, film, final, finger, finish, fire, first, floor, flour, flower, fog, fold, follow, foot, forest,
forget, fork, fox, free, fresh, friend, front, fruit, fry, full, fur, garage, garden, garlic, gas, gate, gather, give, glass, glove,
glue, gold, gram, grape, grass, green, grey, guard, guess, guest, hair, half, hall, ham, hammer, hand, hang, happen, hard, hat, hay,
head, hear, heart, heavy, hedge, here, hidden, hide, high, hill, hold, holiday, honey, hope, horse, hospital, hot, hotel, hour, house,
huge, inch, ink, inside, iron, jacket, jam, jar, judge, keep, kettle, key, king, kitchen, knee, knife, knit, knock, know, ladder,
lamp, large, last, late, later, lawn, lawyer, lead, leaf, leather, left, leg, lemon, lend, letter, library, lie, lift, light, line,
list, listen, little, local, lock, long, look, lose, lost, loud, low, lunch, magazine, main, many, map, market, matter, maybe, meal,
mean, measure, meat, meet, melon, metal, meter, middle, mile, milk, minute, mirror, money, month, moon, morning, mother, mountain,
mouse, mouth, mud, museum, music, nail, name, narrow, near, nearby, neck, need, needle, neighbor, nest, newspaper, next, night,
noodle, noon, north, nose, note, notice, now, number, nurse, nut, oat, odd, office, often, oil, old, once, onion, open, orange, order,
other, outside, oven, owl, pack, page, paint, painter, palace, pale, pan, paper, party, path, pay, pea, peach, pear, pen, pencil,
pepper, perhaps, phone, photo, pick, picture, pie, pillow, pilot, pink, pipe, place, plain, plant, plastic, plate, plum, poem, police,
pork, pot, potato, pound, prefer, price, prison, private, public, pull, purple, put, queen, quiet, quite, rabbit, radio, rare, rarely,
rather, raw, read, ready, receipt, red, remember, rent, repeat, restaurant, return, rice, right, ring, ripe, road, rock, roll, roof,
room, root, rope, rough, round, same, sauce, sausage, say, scarf, season, see, seed, seem, sell, send, several, sew, shadow, shake,
shape, share, sharp, sheep, shelf, shirt, shoe, shop, short, shoulder, side, silver, simple, singer, single, sister, sit, skin, sky,
sleep, slide, small, smell, smoke, smooth, snow, soap, sock, sofa, soft, soldier, sometimes, song, soon, soup, sour, south, spare,
spend, spoon, square, stairs, stamp, stand, star, start, station, stay, still, stone, stop, store, story, stove, straight, strange,
straw, street, string, sugar, summer, sun, supper, sweep, sweet, table, tail, tailor, take, tall, taste, tea, television, tell, then,
there, thick, thin, think, thread, ticket, tie, tin, tiny, today, together, tomorrow, tool, tooth, top, touch, towel, tower, town,
tree, triangle, truck, trunk, turn, twice, uncle, unpack, upstairs, usual, usually, valley, very, village, visit, voice, wait, walk,
wall, warm, watch, wax, wear, weather, wedding, week, weekly, weigh, west, wheat, wheel, white, whole, wide, wind, window, wing,
winter, wire, wish, wolf, wood, wooden, wool, word, wrap, write, writer, yard, year, yellow, yesterday, yet, young
