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
