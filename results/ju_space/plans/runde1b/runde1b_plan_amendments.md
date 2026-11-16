# Round 1b: plan amendments (main §9.3)

## Current status: amendment 1 withdrawn (user decision, 2026-11-16)

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
