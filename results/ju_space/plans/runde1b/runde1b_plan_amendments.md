# Round 1b: plan amendments (main §9.3)

## Current status: amendment 2, opt-in variant `htq_v2` (2026-11-16)

The user's later instruction, "Dann den kleineren Datensatz mit 15-18", supersedes the decision to continue only with H+.
The smaller object is the operational **neuron set**, H_tq (286 neurons in the observed Qwen3-1.7B run), not a smaller set of
questions, examples or evaluation items. Amendment 1 remains withdrawn. The distinct amendment 2 below defines the new
post-hoc continuation; the original H+ results and the withdrawal record are retained.

The opt-in continuation and nine notebooks are implemented. Thirty-three model-free checks passed for seed isolation,
checkpoint selection, policy/provenance guards, cross-account mount aliases, post-hoc reporting and notebook execution flow.
No new GPU run or write to the user's Drive has taken place; full mock-model and A100 execution are not claimed.

## Amendment 2 (2026-11-16T10:17:24Z) — post hoc, opt-in `htq_v2`

**Decided by:** the user (16 November 2026), after the earlier recommendation to withdraw amendment 1 and an explanation of
the H+ running-time cost. The later instruction selects the smaller H_tq continuation.

**Reason.** The frozen G-K22 fallback uses H+ (12,445 neurons in the observed run) when H_tq is unstable. Matching and reading
large random sets makes the remaining H-dependent measurements expensive. Retaining H_tq is a methodological change made
after relevant H+ results were seen; it is not a reduction of the data or a correction to an unchanged measurement.

**Changes to parts, sections and rules.**
- Part K, K2 "Stability switch (G-K22)" and gate G-K22: within the explicitly selected `htq_v2` variant, retain H_tq as H_op
  when the median Dg2 bootstrap Jaccard is below 0.5, and label it "unstable H-set (H_tq kept; amendment 2, post hoc)".
  G-K22 still records instability; the amendment does not make an unstable set pass the gate. The frozen G-K1 exception,
  H_ar primary when the TriviaQA yield fails, is unchanged. The corresponding K2 protocol in RK-2 (r1b_K_S7) uses the same
  variant policy on that model's selected sets; its set size is determined by its own run, not fixed at 286.
- Re-match RS39 / RS39mc / RS19 / RS20 / RS10 to the operational set used by the variant. H+ remains descriptive, including
  hn_set.json's H_plus, K3a's detection comparison and the K-chain lens capture set.
- Main §3.1 SG8 and downstream K/L readings use the variant's H_op and newly matched random sets. Keep the frozen
  sample counts, item identities, seeds, thresholds, statistics and calibration requirements. No questions, data streams,
  experimental cells or evaluation samples are removed to meet a time target.
- Main §§0.5 and 9.3: K/L verdicts under this continuation are labelled post hoc. Affected H-dependent members of
  CF-K-attr, CF-K-link, CF-L and CF-HERR are excluded from the round-level confirmatory families; amended descriptive
  results are reported separately. The reused R/D measurements keep their original provenance and inferential status.

**Separation and continuation.** Both accounts use a new root, `dsse_runs/variants/htq_v2`, with separate local working
folders. The original `dsse_runs` session folders remain the frozen-text record. Do not overwrite the original K_S1 or
combine H+-dependent outputs with H_tq-dependent outputs in one continued session.

`HTQ_START` runs in account A on an A100 and requires the existing `dsse_runs_B` shortcut with write access. It seeds the
variant with reusable R/D sessions, K_S3 and only the H-independent K_S1 prefix. K_S1's H-dependent steps, starting at
k2_sets, run again in the variant; a completed original K_S1 is not treated as a completed variant K_S1. The independent
H_ar step rebuilds its non-Drive fit cache when needed, including after a fresh-runtime resume. Downstream K/L
sessions use the new products. K_S4 is rerun in full, including the already completed H+ k8b work; its old markers,
outputs, copied inputs and caches are not imported into the variant. RK-2 also uses isolated variant outputs.

For one GPU, use the new notebooks in this order:
`HTQ_START` → `HTQ_B2_in_A` → `HTQ_B1_in_A` → `HTQ_A1_DC_zuerst` → `HTQ_A2` → `HTQ_ANA`.
These notebooks already select branch `codex/r1b-htq-continuation`; no manual BRANCH edit is needed. Compute L-CAL in
parallel with `HTQ_LCAL` on a high-RAM CPU runtime using the frozen calibration code. Its result is
`dsse_runs/variants/htq_v2/r1b_lcal/l_cal.json`; commit it before the variant's ANA.
Run tier 2 afterwards. This is an operational order for the isolated variant, not a change to the frozen data or gates.

**Outputs seen before this amendment.** The H+ values and diagnostic outputs listed under historical amendment 1 below
were already known, including |H_tq| = 286, |H+| = 12,445, failed G-K22, set-matching timings and the negative SG8 preview.
The subsequent handoff recorded K_S1 as complete under the frozen H+ rule. In the present read-only Drive check, the
coordinator confirmed completed K_S1 at code commit 121ab8a; D_C remains partial after d5_gen failed; K_S4 has completed
k8b work, a failed k8a and pending k8c work. These previously seen results make amendment 2 post hoc. No new H_tq result
is asserted here, and no original Drive data has been modified as part of preparing this amendment.

**Reporting and running time.** Keep the original H+ outputs alongside the labelled amended outputs. Missing H+-variant
results remain missing; the existence of an H_tq result does not complete its frozen-text counterpart. The earlier
15–18-hour estimate is a baseline estimate, not a guarantee for this migration. The new run additionally includes K_S1
and K_S4 remeasurement and setup/copy I/O; descriptive H+ capture is still present. Measure the first new steps before
updating the remaining-time estimate. No achieved running time or new calibration result is claimed.

## Historical withdrawal of amendment 1 (user decision, 2026-11-16)

The following records the earlier completed withdrawal checkpoint. The implementation and validation statements in this
historical section describe that checkpoint, before the user selected amendment 2; they are not claims about the current
`htq_v2` implementation. Amendment 1 itself remains withdrawn.

**Amendment 1 is withdrawn and has no operative effect.** On 16 November 2026 the user accepted the recommendation to keep
the frozen rule: if G-K22 finds H_tq unstable (median Dg2 bootstrap Jaccard < 0.5), H+ replaces H in the prescribed set uses,
with random sets re-matched to H+. The frozen G-K1 exception (H_ar primary when TriviaQA yield fails) is unchanged.

According to the 16 November 09:50 MEZ handoff, r1b_K_S1 completed under the frozen rule with H_op = H+ before amendment 1
ever took effect. Its completed outputs are retained; withdrawal does not require re-running r1b_K_S1 or overwriting those
outputs. This records the reported handoff status, not a new independent check of the live Drive or Colab run.

The amendment-specific code and required output markers have been removed from both r1b_K_S1 and r1b_K_S7 (RK-2).
The two affected modules, r1b_k_s1.py and r1b_sessions_K.py, are byte-identical to freeze commit 4de7a3b.
Model-free checks passed: session contracts, SG1 (plan/package hashes and freeze ancestry), and runner continuation fixtures
for k2_sets and k3a in both sessions. Completed steps are skipped with the original outputs; missing outputs and failed step
records still require work. The 14 K1 scorer checks also passed. No real model or live Drive outputs were used in these checks.

## Historical amendment 1 proposal — withdrawn, non-operative

The original post-hoc proposal below is retained as history. Its change, overwrite instructions, labels and reporting rules
are not current instructions; its statements about incomplete runs describe the earlier observation recorded with the proposal.

### Amendment 1 (2026-11-16T00:30:00Z) — post hoc

**Decided by:** the user (16 November 2026), on the coordinator's proposal (option B).

**Reason.** In r1b_K_S1 the TriviaQA H-set H_tq (286 neurons, 1.66 ‰) failed G-K22 (median Dg2 bootstrap Jaccard < 0.5). The
frozen switch (part K, K2 'Stability switch (G-K22)', and G-K22 in the gate table) replaced H by H+ in every set use, with random
sets re-matched to H+. H+ has 12,445 neurons (7.2 ‰ of 172,032), so H+, its 39 matched random sets and their union cover almost
the whole MLP. Measured cost: k2_sets 9,097 s (planned 4.7 min) and k3a 15,620 s (planned 1.9 min); r1b_K_S1 hit the 4.5-h time
guard with k3d (8 parts) and k3e still open. Extrapolated, K's and L's H-set measurements (K3d, K4, K6, K8, K10, L1, L3, L4) would
need several days of A100 time instead of the funded hours. The switch was planned for sets of about the size of H_tq (K.8 costs
assume |H| of order 10²); the plan did not foresee an H+ two orders of magnitude larger.

**Change (parts, sections, rules).**
- Part K, K2 'Stability switch (G-K22)' and gate G-K22: if the median Dg2 Jaccard < 0.5, H_tq stays H_op (string
  'unstable set (H_tq kept; amendment 1, post hoc)'); random sets RS39 / RS39mc / RS19 / RS20 / RS10 are matched to H_tq.
  H+ is kept descriptively: in hn_set.json (H_plus), in K3a's detection comparison and as a capture set in K's lens. G-K1
  (H_ar primary if the TriviaQA yield fails) is unchanged.
- Main §3.1 SG8 ('H_op in {H_tq, H+}') reads with H_op = H_tq; every verdict that uses H_op carries the label
  'unstable H-set (G-K22 failed; amendment 1)'.
- No other rule, threshold, statistic, sample or key changes. Code: r1b_k_s1.k2_sets (H_op choice, string, file
  hn/amendment1.json) and r1b_k_s1.k3a (file k/k3/k3a_amendment1.json); r1b_sessions_K adds these files to the steps' outputs so
  that k2_sets and k3a run again in the continuation of r1b_K_S1 (their earlier H+ outputs are overwritten; the step records of
  the earlier attempt stay in session_attempts.json).

**Outputs seen before writing this amendment (post hoc).** By the coordinator (and shown to the user): r1b_K_S1's
session_attempts.json and session_partial.json (step times, G-K22 outcome), hn/hn_set.json (|H_tq| 286, |H_ar| 5, |H+| 12,445,
H_op_name H+, C* 1.339, 'unstable set (H+ used)'), hn/rs_sets.json (G_K9: 39 RS39, 1 RS39mc, 'scale-matched random sets'),
sg8_preview.json (G-K2 preview 'H-neurons not replicated (detection at random-set level)', AUROC H+ 0.859 [0.828, 0.892],
RS39 AUROCs 0.84–0.86). Also seen during the run: completion states and step lists of r1b_R_S0–S6, r1b_D_B, r1b_D_A, r1b_K_S3,
r1b_D_C (no result values opened), and the failure log of r1b_D_C's d5_gen (a crash, fixed separately without value change).

**Reporting.** Marked post hoc. Results that depend on H_op are reported under this amendment. Under the frozen text (H_op = H+)
only K2's diagnostics and the SG8 preview of the first k3a run exist (values above); the rest of the frozen-text variant was not
run (compute) and is reported as 'not run (amendment 1: H+ too large for the funded budget)'.
