# NOTES of implementer 2 (section 9.5 split (2)): r2_unlearn_qwen.py

Implements the frozen plan `results/ju_space/plans/abschnitt11_12_runde2_plan.md` (sha256 5e476340...edc2d), section 5.2 (C-Q2 to C-Q6,
Qwen side of Part C) with the shared objects of 2.4 (O1, O2, O4, O5, O6), the gates decided in session (G13-G16, G43, G44; G28 labels) and
the CLI of 9.2. Coded against the final r2_rk.py / r2_axes.py / r2_stimuli.py of implementer 1 (NOTES_w1.md of 12 Nov 20:02). Nothing of
implementer 1's files was changed. No round-1 output was opened: every test ran on MOCK round-1 states built locally with
`MOCK=1 BLOCK=14 DEVICE=cpu NT=1 python rk_loop.py <root> <test>/r1/gpu1/rk all` plus r2_rk.mock_r1's fake dn_delta.json,
blocks_chosen.json, summary_b14.json and shard_done.json; modified copies of these mock files (summary, loops.json) test the gate branches.
No file named summary_b*.json, loops.json, pilot.json, k5.json, k6.json, dn_tf.json or pc_summary_*.json was opened with a tool; the
script reads them (R1.summary, R1.loops, R1.k6 are inputs of C-Q2 / C-Q4b), in the tests only the mock copies in the test directory.

## Command, stages, code test

    python r2_unlearn_qwen.py <root> <r1_dir> <r2_dir> [q3 | q4a | q2 | q4b | armA | q5 | q6 | all | selftest]    env DEVICE, NT, MOCK
    (BLOCK, if set, must be 14: every stage works at block 14; one process may run several stages ('all'), the runner calls one per step)
    MOCK=1 python r2_unlearn_qwen.py <root> <r1_dir> <r2_dir> selftest      builds the MOCK fixture in <r1_dir> if missing, runs r2_axes.py
                                                                            (O2) if missing, every stage through the CLI, resume, checks

| Stage | Plan | Needs in <r2_dir> | R1 inputs (found by name, r2_rk.find_r1) | Outputs |
|---|---|---|---|---|
| q3 | C-Q3 (+ C-Q1's causal linear-rung inputs, G43) | axes/r2_axes.{pt,json} (O1, O2) | state[14] (axes, ref) | unlearn/q3.json, q3.pt |
| q4a | C-Q4a | O2 + unlearn/q3.{json,pt} | state[14] | unlearn/q4a.json, q4a.pt |
| q2 | C-Q2 (a), (b), (b'), (c) | O2 | state[14] (axes, ref, alpha, k6_null<i>), k6.json | unlearn/q2.json, q2.pt; state entries k6_r2_*, r2_b05_* |
| q4b | C-Q4b with gates G13-G15, G44 | - | state[14] (axes, dir_RSI_a<SUF>_1), summary_b14.json, loops.json (RND rows) | unlearn/q4b.json (+ q4b.pt) |
| armA | C-Q4 arm A with G16 | - | state[14] (axes, ev_RSI_a<SUF>_0, ref) | unlearn/armA.json, armA.pt |
| q5 | C-Q5 pushes (superposition, dose) | - | state[14] (axes, ev_RSI_a<SUF>_0, ev_SEA<SUF>_0) | unlearn/q5.json, q5.pt |
| q6 | C-Q6 | - | none | unlearn/q6.json, q6.pt |

Session packing: all seven stages belong to r2_q1 (9.2; the runner r2_run_session.py of 20:37 has them there, q6 with r1_reads=None).
The script does not depend on the packing: q4b, armA, q5 and q6 read no round-2 output (no O2), every stage opens block 14 in its own
process, and q3, q4a, q2 need O2 (r2_axes.py) in the same <r2_dir> (q4a also needs q3).

Resume: a stage is skipped when unlearn/<stage>.json exists (written last, atomically, after the .pt). Inside a stage every expensive unit
(statement scores, text sets, loop rounds, calibrations, cap runs) is cached in `<r2_dir>/local/unlearn_<stage>/` with a signature of its
inputs, so a stage that crashed resumes; local/ is never synced (9.5). 'not run' paths write only the JSON.

Status values: 'ran'; 'not run (round-1 output missing)' (block 14 not final - r2_rk.open_block raises R1Missing -, a needed R1 file or
state key missing, or O2 'not run' at block 14; G4, G6); C-Q4b 'not testable (gate: ...)'. A missing round-2 input of an earlier step
(r2_axes outputs, q3 for q4a) stops the stage with exit code 1 (session order; the runner's 'after' handles it). C-Q6 reads no round-1
file and runs even when block 14 is not final (the model is then loaded through rk_loop with an empty state of its own,
<r2_dir>/local/q6_rk; tested: identical numbers).

## Output formats

Every JSON also holds script, plan, plan_sha256, stage, block, mock, seeds, n_boot, level, rule_thresholds, alpha_14, alpha_scale,
suffix, cap, versions (torch, transformers, GPU) and finder (r2_rk.FINDER_LOG). Arrays in the .pt files are float32 unless stated.


q3.json: `D0 {H30, RGI30}`, `Delta_T`, `g0 {RSI, RGI}` (O1 g_0), `Delta_RGI`, `GA14`, `labels {u_RGI_perp, span, u_diff}` ('RGI axis not
resolved' if GA(14) fails), `ablations {u_RSI, u_RGI_perp, span, u_diff, ctrl2, null0..null15, pair0..pair7}` each with dim, role, pool,
damage, `H30` / `RGI30` {D, D0, dD, D_ci90_statements, M, inverted; for targets and controls also z_D and the null names used},
`texts {RSI | RGI: g, g0, kept, ratio, ratio_ci90, g_ci90, m_concept, m_foreign {sports, tone, sea, TOPIC}, own_axis_gap, own_axis_auc,
n_empty}`, u_RSI also `unprompted {shift_g0, shift_ci90_g0, shift_Delta_T, room_above_anti_g0, kept, m_foreign, identical_to_ref}`;
`controls` (rand_14_2 = null2, cov_14_0 = null8, ctrl2 = cov_14_1 + cov_14_2), `ctrl_p5 {u_RSI, span, span_RGI}`, `rules {u_RSI, span}`
(rule_unlearned: conditions, failed, unlearned, inverted, note), `c_q4a {ablation (None if none passed), used}`, `linear_rung` (C-Q1 inputs:
the four g_abl / g_0 with intervals and an in-session view), `references` (match rate of the session's unprompted references with R1
'ref' conts), `r1_meta_gpu1`.
q3.pt: `ablations {name: U, H30 (lo, y per statement), RGI30, damage}`, `texts {'<ablation>|<RSI|RGI|unprompted>': pro, anti, keys,
keys_anti, Xp, Xa [64, D] float32 (block-14 reads by the unablated model), capped_share, per_prompt, g_draws, ratio_draws [2000]}`,
`refs_unprompted {conts, keys, X}`.
q4a.json: ablation, passed_C_Q3, g_cue, g_abl, g0, restored_share, restored_share_ci90, share_defined, own_axis_gap_cue, ceiling {g, share,
ratio_g0, ci90_g}, kept, verdict_in_session (rule_cue), cue_prompt_tokens, empty. q4a.pt: messages, keys, ablated / unablated {conts, X,
per_prompt, draws}, U, share_draws.
q2.json: `K6 {'<dir>|H30', '<dir>|RGI30'` (rk.k6_direction entries + z_fin), `u_RSI|H30 (R1.k6 ...)`, regression_H30, M0_H30, M0_RGI30,
regression_RGI30, nulls_RGI30}`, `alpha_u {dir: f, damage (of the counter-push), damage_plus, tried, in_band, start}`, `texts
{'<dir>|<concept>': gap entry as in q3 + capped_share}`, `behaviour {u_RSI, u_diff, u_RGI_perp: statements, f, Ue_fin, A_fin, Mp, Mm, z_fin,
M0; regression {H30, RGI30}; nulls}`, `content {dir: RSI / RGI {content, content_ci90, counter_side, sigma, unit}, sigma_u_T, foreign,
n_pairs, dropped, capped_share}`, `perplexity {dir: plus / minus {ppl, mean_nll}, unpushed}`, `rules {u_RSI, u_diff, u_RGI_perp}`
(rule_suppressed + concept, statements, ctrl_p5), flags, labels. q2.pt: texts (as in q3), content evaluations (per pair: delta, ok, xp,
xm, prompt_of, first_div, conts, keys), perplexity per text. State entries (guarded): k6_r2_u_RGI_perp, k6_r2_u_com, k6_r2_u_diff,
k6_r2_u_RGI_perp_RGI30, k6_r2_u_com_RGI30, r2_b05_null<i> (calibration + H30 / RGI30 behaviour), r2_b05_<dir>.
q4b.json: status, gate {open, branch, G13, G15, G14, verdict}, r1 {V_RSI, LIKELY_wh, from_would_have_read, label_attractor,
attracted_loop, gates_failed, rnd_last_rows, cos_d1_u_RSI}, twin_start_check (read-outs of the G15 round), labels (G-desc if the gate
opened through a would-have-read LIKELY), start, `ablated` / `twin` rows for d_1..d_3 {c, p, Q, rel, r, A, norm, Ue, Ue_ci90, cos_u_RSI,
cos_push_u_RSI, damage_of_push, dropped, capped_share, identical_pairs, first_div_median, n, d, norm_d}, verdict_in_session (rule_loop).
q4b.pt: start, twin / ablated {evs (per pair: delta, ok, xp, xm, prompt_of, first_div, conts, keys, ...), d {1..3}, push {0..2}},
twin_start_check {ev, start}.
armA.json: beta (calibration), reference_seed0_RSI_a {c, ci90, n_pairs, key}, c1, continues (G16), rounds, rows (as q4b), verdict_in_session.
armA.pt: evs, d, push, ref_c_draws.
q5.json: mix (cosines), superposition {coef_RSI, coef_SEA, cos, cos_disatt, rel_mix, rel_sum, cos_ci90, coef_*_ci90, cos_disatt_ci90,
holds_in_session, coef_band}, dose {norm_ratio, cos_debiased, norm2_alpha, norm2_2alpha, ratio_ci90, cos_ci90}, references, read_outs (stats
of the mix and dose read-backs on u_T_perp). q5.pt: mix / dose evaluations, mix_dir, F_mix, F_RSI, F_SEA, F_sum, F_2alpha, boot draws.
q6.json: model ('open_block' or 'model only'), split, relations {improves, describes, repetition: helix {block: |d12|, |d23|, |d34|, cos(d12,
d23), cos(d23, d34), cos(d12, d34), rotation_prediction}, angular / linear {block: share, ci90, n (+ omega, plane_share)}, readout {18,
22: validation, validated, unpatched medians}, modes, kl {same_depth_floor_median, kl_2_4_median, kl_2_3_median, usable}, cap
{'k<3|4>_r<1-3>_<mode|rand>': frac_all, frac_heldout (median, ci90, n), kl_cap_median, readout_18 / readout_22 {median, ge_3_5_all,
ge_3_5_heldout}}, patch_check_rel_max, r_star_in_session}, improves_minus_describes_angular {block: diff, ci90}. q6.pt: items (ids,
way_it, start), relations {rel: states (per text [L, 4 blocks, D] float16: every state at blocks 8, 14, 18, 22), ids, keys, starts,
V [D, 3], m2, ranks, readout_values, kl arrays, patch {key: frac, kl, readout, last18, last22}}.

## Rule functions for r2_ana (OD-G8; pure, no model, importable)

`rule_unlearned`, `rule_suppressed`, `restored_share`, `rule_cue`, `likely_wh`, `attracted_loop`, `q4b_gate`, `rule_loop`,
`arma_continues`, `superposition_stats`, `u_norm2`, `ols_nulls`, `reg_from_r1_k6`, `z_fin`, `z_against`, `boot_g`, `stmt_D`,
`stmt_boot_index`, `calibrate`, and the C-Q6 helpers (`helix_stats`, `depth_modes`, `angular_readout`, `linear_readout`, `depth_readout`,
`ridge_dual`). In session they decide C-Q4a's ablation (G43), the C-Q4b gate (G44) and arm A's continuation (G16); the JSONs carry
in-session views of the other rules ('rules', 'verdict_in_session', 'holds_in_session', 'r_star_in_session') for r2_ana to recompute.

## Ambiguities and the readings chosen

O1 / gaps (C-Q2, C-Q3, C-Q4a)
1. Fixed axes, g_0 and the reference states are r2_axes.pt's block-14 `fixed_axes` and `X` (O2's reads); g = r2_rk.gap_on_fixed_axes. The
   stage asserts that g on the O2 texts reproduces O1's g0. Pushed / ablated / cued texts are read with rk.read (early-stopped; equal to
   O2's unstopped reads up to rounding).
2. Interval of g / g_0: joint prompt bootstrap (r2_rk.prompt_bootstrap, 2,000 replicates, stratified 32 + 32, seed 31), the same weights for
   g and g_0 in a replicate, the fixed axes held at their point estimates; a prompt with an empty text drops out of its g (per replicate).
3. 'Upper end ... below the 5th percentile of the random reference' = 90 % upper end of g / g_0 < 5th percentile of the control's bootstrap
   ratios (strict). u_RSI: the draws of the two 1-dim controls pooled (4,000); span: ctrl2; C-Q2: ('cov', 14, 3) of the same concept.
4. g_0 <= 0 (meaningless ratio) fails the gap conditions ('no baseline gap'); m_concept = |g - g_0| / g_0 is NaN then.
5. Foreign moves (OD-C6): m_X = |mean projection of all non-empty instructed texts (pro + anti) on unit(u_X), operated minus reference| /
   gap_cross_X, u_X = R1 sports / tone / sea (raw axes) and O2's u_TOPIC; gap_cross from R1 'axes' and O2's TOPIC. A non-positive
   gap_cross_X gives NaN, which fails the condition (conservative; in MOCK every foreign gap is negative, so m_X is NaN there).
6. Keys of the instructed texts: the keys stored with the O2 texts (r2_axes.pt 'keys'), asserted equal to keyseed('axis', 'rsi', p) for all
   six sets outside MOCK (under MOCK r2_axes' MOCK rule), so operated and reference texts always share keys. Rows in rk_loop.text_axis's
   order (pro, anti per prompt), batches of 32.

C-Q3
7. D = mean y(pro statements) - mean y(anti statements), statement by statement as rk.logodds_matching (equals 2 M; tested); ablation by
   WriteOrth around the forward. z_D (ddof 1) for the targets against the 16 nulls (1-dim) or the 8 pairs (2-dim), for the controls
   without themselves (null2 / null8 against 15, ctrl2 against the 6 pairs i not in {1, 2}); computed on H30 (rule) and RGI30
   (descriptive). 'inverted': D_abl < 0 and the 90 % upper end of a statement bootstrap (2,000; pro and anti resampled separately;
   seed 32; the same draws for every ablation) < 0.
8. Damage of an ablation = rk.damage's texts, positions and KL with WriteOrth in place of the push (the unablated log-probs computed once,
   class RefKL; RefKL.push equals rk.damage exactly, tested).
9. The C-Q3 null set is the 16 Pool directions in rk.k6's order (r2_rk.pool_dir); under MOCK these are r2_rk's seeded MOCK Pool
   directions (16), not rk.k6's four MOCK nulls, so the exclusion logic runs at full size.
10. Unprompted shift (u_RSI, reported): mean over prompts of (x_abl - x_ref) . unit(u_T_perp) / g_0 (and / Delta_T) with its interval;
    'room_above_anti_g0' = (mean projection of the references - mean projection of the RSI_ANTI texts) / g_0 as the floor caveat. The
    session references are regenerated with keyseed('ref', p) (rk.references' rows); their match rate with R1 'ref' is reported (22k).
11. The 'unlearned' rule is evaluated in session (it selects C-Q4a's ablation, G43) without labels; G-desc and other labels are r2_ana's.

C-Q4a
12. Cue message by r2_stimuli.cue_message with this session's RSI_PRO texts decoded by tok.decode (continuations hold no stop token);
    keys = the O2 RSI_PRO keys. g_cue: cued ablated texts against C-Q3's ablated anti RSI texts of the same ablation; g_abl, g_0 from the
    same per-prompt values (joint bootstrap). The restored share is undefined (NaN, `share_defined` false) if |g_0 - g_abl| <= 1e-9 |g_0|
    (the ablation left the gap unchanged; this is the MOCK case). Ceiling: unablated cued texts against the O2 RSI_ANTI texts.

C-Q2
13. alpha_u is calibrated on the counter-push itself (damage of -f unit(u)), because rule (iii) concerns 'the push -alpha_u'; the damage of
    +alpha_u unit(u) is reported (damage_plus). Start alpha_14 = R1.loops[14] '_meta' alpha (incl. alpha_scale), rk.set_alpha's update,
    band and 6 evaluations, closest kept.
14. K6 at 0.03 nats through r2_rk.k6_direction ('r2_<dir>'); rk.k6_direction starts its search at 0.75 x R1.state alpha (unscaled), which
    equals the plan's '0.75 alpha_14' only if alpha_scale_14 = 1 - kept as the plan says 'rk.k6_direction' (the calibration target is the
    same). The RGI30 K6 entries are named 'r2_<dir>_RGI30' (a separate call: rk.k6_direction caches by name); their calibration repeats
    deterministically.
15. R1.k6 is read from k6.json; if that file is missing, R1.state[14]['k6'] (the same dict, written by rk.k6) is used and flagged.
16. Null recalibration (b'): on +d (nulls have no orientation), rk.set_alpha's rule from alpha_14; behaviour at +-f on H30 and RGI30;
    regression and z_fin by O5's formula (n = number of nulls). The concept's behaviour at its own alpha_u: u_RSI and u_diff on H30,
    unit(u_RGI_perp) on RGI30 (the plan's three; others not computed, to keep the 7.2 cost). M0 of rule (ii): R1.k6 M0 (H30), the session's
    M0_RGI (RGI30).
17. Content pairs (c): prompts 0..NRSI-1, keys keyseed('r2CNT', <name>, p) with names 'u_RSI', 'u_diff', 'u_RGI_perp', 'cov_14_3';
    content = mean delta . unit(axis) / unit (g_0 for RSI, Delta_RGI for RGI; Ue's convention), interval over the RSI-topic prompts (the
    RSI stratum of the session's joint draws); counter side (s0 - s-) / unit with s0 from R1.state[14]['ref']['X'] of the same prompts;
    sigma by rk.stats's formula on unit(u_T_perp), unit(u_RGI_perp) and (sigma_u_T) on u_T; foreign = mean delta . unit(u_X) / gap_cross_X.
    Perplexity: R1 reference continuations of the scene prompts, teacher-forced under +-alpha_u unit(u) from the first user token,
    exp(total NLL / tokens).

C-Q4b, arm A
18. G15 is evaluated only when G13 holds (the twin's first round from s0 is run only then). Gate order: branch 'likely' (G13 and G15) first,
    else 'attractor' (G14; the twin's first round rerun from the attractor start with the same keys ('r2Q4B', 0, p, 0, 14)). The G-desc
    label is attached when the gate opened through a would-have-read LIKELY.
19. rule_loop precedence (every condition at d_2 and d_3): ablated loop 'gone' (p > 0.10, |A| < 0.25) -> 'un-searched' if the twin re-finds
    RSI (p <= 0.01, A >= 0.25), else 'undecided (twin does not re-find RSI)'; then 'survives'; then 're-routed'; then 'undecided (twin does
    not re-find RSI)' if the twin fails; else 'undecided'. The matched-size reference (seed-0 half of R1 RSI_a; R1-6) is left to r2_ana as
    section 8 assigns it; q4b reports an in-session view of the rule only.
20. Damage of each loop push: rk.damage's definition in the unmodified model (descriptive).
21. Arm A: beta by rk.k6_direction's 0.03-nat rule (start 0.75 alpha_14, 3 evaluations, band 0.025-0.035) on the constant push alone
    (-beta unit(u_RSI), rk's cap rule, from the first user token). ConstPush adds sign beta min(|h_t|, cap_b) unit(u) with |h_t| the block
    output before any push (prepend hook), in rk's operand order: with a zero loop push it reproduces rk's own push exactly (tested).
    G16 reference: 90 % interval of c(d_1) over the session's joint draws, seed-0 pairs = even indices of R1 'ev_RSI_a<SUF>_0'. If arm A
    continues, the loop rule with the R1 seed-0 comparator is r2_ana's (R1-6).

C-Q5
22. mix = unit(unit(u_T_perp) + unit(sea)); F(RSI) = mean of the seed-0 ok pairs of 'ev_RSI_a<SUF>_0', F(SEA) = 'ev_SEA<SUF>_0'; rel_mix and
    rel_sum over the 20 K2 halvings (rk.stats's formula; rel_sum from half sums (F_RSI(half) + F_SEA(half)) / sqrt 2); coefficients by least
    squares in the orthonormalised plane. Interval of the disattenuated cosine with the rels fixed. Dose: U-statistic norms over distinct
    pairs; in a bootstrap replicate copies of the same pair are excluded ((|sum w d|^2 - sum w^2 |d|^2) / ((sum w)^2 - sum w^2)). The SVD of
    the stacked d_n and the +- symmetry from R1 sigma are file-only (R1-5) and left to r2_ana.

C-Q6
23. Stimuli: rsi_direction.py's Q2' texts (r2_stimuli copies), plain text without chat template, each read without padding (equal-length
    batches), as rsi_direction.py read them. Split: rsi_direction.py's default_rng(13) (first 12 = training objects).
24. Depth modes: SVD of the 12 training objects' object-centred last-token states (depths 1-4, 48 rows) at block 14; m2 = mean depth-2
    last-token state of the training objects (OD-C4). Helix: d_k from the mean last-token states over all 24 objects; the 'rotation
    prediction' for cos(d12, d34) is cos(2 theta) = 2 cbar^2 - 1 with cbar the mean of the two consecutive cosines.
25. Angular readout: object-centred states with the step direction s = unit(mean (x4 - x1)) projected out; the plane = the top-2 singular
    vectors of the training objects; the rotation sense from the training objects' 1 -> 2 step (the linear readout is trained on depths
    1-2 too: rsi_direction.py's ridge); a held-out object ranks 4 above 3 if the wrapped angle step 3 -> 4 has that sense. Both readouts,
    the helix numbers and the improves - describes difference (paired object bootstrap) are given at blocks 8, 14, 18 and 22; the block
    of the helix rule is not fixed by the plan (r2_ana decides).
26. Validated readouts at 18 and 22: rsi_direction.py's ridge (lambda 1e-2 tr(G) / n) on the training objects' unpatched last-token
    states at depths 1-4 (targets 1-4); validated if >= 75 % of the held-out objects rank 4 above 3.
27. Cap: all three relations, depth-3 and depth-4 sentences, r = 1, 2, 3, mode cap and random control (= 864 patched forwards, the 7.2
    count). 'The second the way it' also matches ' The way it' (the repetition texts write it after a full stop). Random control: one
    random orthonormal r-dim subspace per (relation, depth, r, object), seed keyseed('r2Q6rand', rel, k, r, obj) mod 2^31, its patch
    rescaled per position to the norm of the mode patch.
28. KL fraction 1 - KL(p2 || p_cap) / KL(p2 || p_k) per object (p2 = the same object's depth-2 sentence); medians with 90 % object-bootstrap
    intervals (2,000, seed 33) over all 24 objects (frac_all) and over the 12 held-out objects; 'usable' = median KL(p2 || p4) > the
    median KL between different objects at depth 4. Self-repair inputs: share of capped sentences read as depth >= 3.5 at 18 / 22 (all
    objects and held-out). r* and the rule are r2_ana's (in-session view given).

General
29. Seeds (OD-G6): prompt bootstrap 31, statement bootstrap 32, object bootstrap 33; they and the thresholds are written into every JSON.
30. GA(14) false -> label 'RGI axis not resolved' on the statements that use u_RGI_perp, u_diff or u_com (q3: u_RGI_perp, span, u_diff;
    q2: u_RGI_perp, u_diff, u_com); the measurements run anyway (OD-G5).

## Deviations from the final plan

None in a measurement, threshold or rule. Additions: in-session views of the rules decided in r2_ana; extra descriptive numbers (own-axis
gaps everywhere, damage of the opposite push sign, sigma on u_T and on the concept axes, RGI30 z_D, read-out stats of the C-Q5 pushes,
q6 numbers at all four blocks and over held-out objects); the intra-stage cache under local/.

## Tests run (this container, CPU, NT=1, OMP_NUM_THREADS=1, MOCK; one process at a time, MemAvailable >= 13 GB)

Fixture (in r2_impl/w2/r1): `MOCK=1 BLOCK=14 DEVICE=cpu NT=1 python rk_loop.py <root> r2_impl/w2/r1/gpu1/rk all` (45 s), then
`r2_unlearn_qwen.mock_fixture` (r2_rk.mock_r1's fake files: a would-have-read 'likely' summary_b14 with GO1 failed, dn_delta,
blocks_chosen b_lo 8 / b_hi 20 without states, shard_done). O2 by `r2_axes.py ... all` (6 s).

`MOCK=1 python r2_unlearn_qwen.py <root> r2_impl/w2/r1 r2_impl/w2/r2s selftest` from an empty work directory: **63/63 checks passed
(313 s)**; result in r2_impl/w2/r2s/unlearn/selftest.json. The checks:
- rule functions on synthetic inputs: rule_unlearned (pass, random reference, baseline D, foreign axes, NaN, inverted), rule_suppressed
  (pass, M0 <= 0.1, damage band), rule_cue, restored_share (planted share 0.6 inside its interval; undefined if g_abl = g_0), LIKELY_wh
  (incl. the would-have-read form), G14 attracted loop, G44 gate order, rule_loop (all five outcomes), G16, O5 regression and z_fin,
  the calibration rule, the U-statistic, boot_g, z_against;
- CLI: every stage exits 0 and writes its outputs; a second run ('all') skips every stage; q5 rerun after removing its outputs
  reproduces its numbers from the intra-stage cache;
- q3: g and the ratio draws recomputed from the stored states on O1's axes; z_D against the right null sets (a control without itself);
  span-ablated RGI texts regenerate from the O2 keys; ablation damage = rk.damage's KL with WriteOrth; D = 2 M of rk.logodds_matching
  under the ablation; the session's unprompted references = rk.references' texts (MOCK: identical to R1, match rate 1); in-session
  rule and C-Q4a choice; random reference = pooled 1-dim control percentile;
- q4a: cue message format (OD-C3) and ablation choice; g_cue / g_abl / share; cued ablated texts regenerate;
- q2: K6 entries 'k6_r2_*' and 'r2_b05_*' in the guarded state; RefKL.push = rk.damage exactly; statement scores =
  rk.logodds_matching; r2_rk.k6_direction reproduces R1.k6 tests['u_T_perp'] and its z_fin (MOCK H30[:4]); alpha_u calibration
  record; counter-pushed texts regenerate; z_fin at alpha_u from the recalibrated nulls; RGI30 null regression at the R1 push sizes;
  content pairs regenerate (keys, prompts); rule inputs; k6.json fallback;
- q4b / arm A: gate inputs; twin start s0 = unit(d_1^RSI_a without u_RSI) and its keys; ConstPush with a zero loop push = rk's push
  (identical texts); arm A round 1 regenerates; G16 reference = seed-0 half of R1 RSI_a; attractor branch (modified mock summary and
  loops.json): gate open via G14, start from RND_1's start, 3 rounds each, every push orthogonal to u_RSI, ablated round regenerates
  with WriteOrth in the writer; closed gate: 'not testable (gate ...)', nothing generated;
- q5: F(RSI) = seed-0 half; mix direction; coefficients solve the plane system; dose pairs regenerate;
- q6: 288 stimuli and patch starts; all caps present; mode patches set the block-14 projection to m2's (rel. 1e-7); stored float16
  states reproduce the depth modes; DepthPatch leaves positions before the start unchanged and the random patch has the mode patch's
  norm per position;
- variants: block 14 not final (no summary_b14): every model stage 'not run (round-1 output missing)' without O2, q6 runs with the
  model only and gives the same numbers; O2 'not run' at block 14 -> q3 'not run'; GA(14) forced true in a copy -> no 'RGI axis not
  resolved' label; block 14 at 1.5 alpha (rk_loop MOCK with ASCALE=1.5): q4b, armA, q5 read the '@1.5' keys and use alpha_14 = 1.5
  alpha; env BLOCK != 14 refused.
Single-stage CLI timings on the MOCK fixture: q3 56 s, q4a 6 s, q2 57 s, q4b 5 s, armA 7 s, q5 5 s, q6 13 s.

## Open points

- MOCK gives degenerate numbers on several paths (AUCs about 0.5, so GA(14) is false, the foreign gap_cross are negative and m_X is
  NaN, g_0 is tiny, the 0.05-nat calibration never reaches its band because the tiny model's damage saturates near 0.025 nats, the
  u_RSI ablation leaves the MOCK texts unchanged so the C-Q4a share is undefined, and R1's MOCK K6 regression has se about 0). These
  paths are covered by the synthetic tests of the rule functions; their production behaviour is untested.
- Readings an amendment could state (they do not change a frozen rule, but a reviewer may prefer another reading): alpha_u calibrated on
  the counter-push (13); the K6 start of rk.k6_direction when alpha_scale_14 != 1 (14); the C-Q6 angular readout's construction and the
  block of the helix rule (25); the cap on all three relations and the object set of the KL fraction (27, 28); g_0 <= 0 and
  gap_cross_X <= 0 treated as failing (4, 5).
- r2_ana (r2_unlearn_verdict.py, r2_linearity.py) computes the file-only parts assigned to it in section 8: the C-Q4b matched-size
  reference and arm A's comparator (R1-6), the C-Q5 SVD and +- symmetry (R1-5), the u_RSI K6 numbers (R1-7), the G-desc / G-blk labels.
  It can import the rule functions listed above.
- Production cost (A100, from the 7.1 rates): q3 about 100 s (29 ablations x 68 one-statement / reference forwards, 1,280 texts), q2
  about 140 s (K6, 16 RGI30 nulls, 4 calibrations, 640 + 256 texts, 16 null recalibrations x 120 statement forwards, perplexity), the
  other stages under 40 s each; q6.pt about 100 MB (every state of 288 texts at four blocks, float16).
