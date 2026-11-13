# Amendments to the frozen round-2 plan (abschnitt11_12_runde2_plan.md, sha256 5e476340dfdb1d827582c869f2dfc08e32ed8e9a9dbebf7642acceb5bb5edc2d)

## Amendment 1 (2026-11-12T20:46:00Z)

**Reason.** While implementing r2_theta_verdict.py, implementer 4 found that a plain prompt bootstrap of the U statistic
U = (|sum_j delta_j|^2 - sum_j |delta_j|^2) / (N (N - 1)) counts each duplicated pair in a resample against its own copies. The
replicates are then centred on the squared mean, noise floor included, instead of on U. In a simulation at a true m*^2 of 0.38 the
plain bootstrap's 5th percentile was 0.50, so the G17 guard would pass where it should fail (check script
r2_impl/w4/ustat_boot_check.py in the session scratchpad).

**Change.** In every bootstrap replicate of a U statistic (sections 4.1 B-Q0 and the theta cells), products of a pair with its own
copies are excluded, and the denominator is the number of remaining ordered pairs. This applies to U, m*^2, the G17 guard and every
interval derived from U (the G intervals). Point estimates are unchanged.

**Clarification (no frozen threshold changes).** Rule R2: a bootstrap replicate fit 'crosses 1' when its slope has the sign of the
point fit's slope; the frozen text does not fix the direction (the reference crit_tline3.py counted only downward crossings).

**Outputs seen before writing.** No round-2 output exists. Of round 1 the author of this amendment had seen: the file names and times
of all outputs of shards gpu1-gpu3; summary_b14.json, summary_b18.json, summary_b6.json, blocks_chosen.json and loops.json of block 14
(RSI indicators, gates and verdicts of the three blocks). The operator who ran round 1 saw its on-screen output. The changed rules
concern only round-2 theta statistics and the properties of an estimator; no seen output bears on them. Not post hoc with respect to
round-2 data.

## Amendment 2 (2026-11-12T21:00:00Z)

**Reason.** OD-C2 says a damage-matched random control is 'matched within 0.02', but its procedure only provides partial removal
when the equal-dimension random subspace (dimension r0) already overshoots the target. When the smallest dimension whose mean drop
reaches the target is 2 r0 or larger, that dimension usually overshoots by more than 0.02, and the literal text has no step that
brings the control back to the target. In a smoke test of r2_unlearn_toy.py on freshly trained toy nets (W 64, 2 seeds, real
budgets; not a round-2 session), implementer 7 found that 9 of 10 C-T4 z1 controls, the h1 erasure control and C-T5's dmr control
stayed unmatched this way. Their controls would then fail the 'unlearns' condition (control lower bound >= 0.9 x base) through
excess damage, or the specificity verdict would become 'not testable', for a procedural reason rather than a property of the nets.

**Change.** In OD-C2, partial removal applies at the dimension found, not only at r0. Let d* be the smallest dimension in r0, 2 r0,
4 r0, ... (<= W / 2) whose mean drop over the five draws reaches the target. If the mean drop at d* exceeds the target by more than
0.02, every control draw uses partial removal z - lambda P z on its d*-dimensional random subspace, with ONE lambda in (0, 1] shared by
the five draws, found by bisection (the same settings as the existing r0 case) so that the mean drop matches the target within 0.02.
If bisection does not reach 0.02, the closest lambda is kept and the control is flagged unmatched, as before. The r0 case is the
special case d* = r0 and is unchanged; a target below 0.02 is still matched by the equal-dimension random subspace. For C-T5's dmr
arm, new task j still uses draw j of the five at d*, now with the shared lambda. This applies wherever OD-C2 is used (C-T4, C-T5,
C-T7 and every damage-matched control in Part C). No threshold, rule, measure or count changes; the outputs record d*, lambda and
whether the control is matched.

**Outputs seen before writing.** No round-2 session output exists. Of round 1, the author saw the same files as for Amendment 1. The
author also saw implementer 7's report of the smoke test above. It covered the unmatched controls (9 of 10 z1 controls overshooting),
and in passing a few pilot values on the W-64 smoke nets: erasure at h1 0.07; at h2, accuracy 0.95 with the linear task probe at
chance; task-column edit with sum probe 0.98 and relearning intact; the x-span block destroying the sum probe; circle ring arm
zero-shot 0.998; permuted ring never level 1; arcs of 3 not composing; conjugation -c in 12 of 12; trajectory modes like their null;
no ring code in one-hot nets. These are smoke-test values at W 64, not session data. The change makes the controls meet OD-C2's own
stated tolerance; it does not move a threshold and does not depend on which way any verdict goes. Declared post hoc with respect to
that smoke test.

## Amendment 3 (2026-11-13T04:18:00Z)

**Reason.** The code review of the round-2 scripts (reviewers blinded to round-1 outputs; findings unlearn_b-2, theta_verdict-1 and
theta_verdict-2) found one miscalibrated gate and two places where the frozen text is silent.

(a) G16's reference, the 90 % prompt-bootstrap percentile interval of c(d_1) of the seed-0 half of R1 RSI_a, is not centred on the
statistic it describes. Each bootstrap replicate of the cosine of a noisy mean direction carries the sampling noise of the mean twice
in its norm, so the replicates are centred near c / sqrt(2 - rel) and not at c. In synthetic checks with the script's own function
(D 2048, 64 pairs) the point estimate lay above the interval's upper end in every run, and a c_1 from the same population never fell
below the lower end. G16 would then fire only for a drop of roughly 12 % (rel 0.8) to 25 % (rel 0.5), so arm A would almost never
continue to rounds 2-3; the bias favours the outcome the plan states in advance ('translation (as predicted), not un-searched'). G16
decides what is generated, so the bias cannot be repaired after the session (check scripts: r2_review/unlearn_b/ and its verify/
folder in the session scratchpad).

**Change (a).** In G16 (5.2 C-Q4 arm A; section 6) the reference interval is the percentile interval shifted so that the bootstrap
median coincides with the point estimate: with c the point value, m the median and [q05, q95] the percentiles of the 2,000 replicates,
the interval is [q05 + c - m, q95 + c - m]. Arm A continues to rounds 2-3 if c_1 falls below its lower end. The unshifted percentile
interval, m, and the G16 value that the unshifted interval would have given are stored next to it. Level, rule wording and every
threshold stay as frozen. The descriptive cosine intervals of C-Q5 keep their definition and additionally report the bootstrap median.

**Clarification (b), theta rules.** The frozen text does not say what happens when the gain of a cell is undefined in a bootstrap
replicate (zero attenuation, possible at a 'mode not resolved' block and for the norm gain in the switch case). A rule bound counts as
undefined (the rule's outcome is 'undecided', a reserve condition is not met) when the point value is undefined or more than 5 % of
the 2,000 replicates are undefined. This takes over R6's convention ('>= 95 % of the joint bootstrap replicates'). The share of
undefined replicates is written into every rule output. No threshold changes.

**Clarification (c), R4 (i).** When rule (i) returns 'degenerate' (no single critical mode), the suffix '(RSI-specific)' is not
appended to the R4 verdict, and the block does not enter R3's alignment form; it is listed there as 'not entering (degenerate)'. The
per-loop alignments stay reported as descriptive, as R4 (i) says.

**Outputs seen before writing.** No round-2 session output exists. Of round 1 the author had seen the same files as for Amendment 1;
these include the c values of the block-14 loops. The author had also seen implementer 7's smoke-test report named in Amendment 2 and
an outside probe (results/ju_space/microlens/mlpp_probe_b14.json: a new measurement on rebuilt round-1 axes at block 14, made after
the freeze by another agent). The three items were found by reviewers on synthetic data and by reading the code; none depends on a
seen value. Because the author knows round-1 c values, (a) is declared post hoc with respect to round 1. Its direction is known:
it makes a continuation of arm A more likely, that is, it removes a bias towards the outcome the plan predicted.
