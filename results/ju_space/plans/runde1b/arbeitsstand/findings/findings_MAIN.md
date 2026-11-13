# Unverified review findings for MAIN (finder stage only; verification and fixes were stopped to save quota)

## find:the main file:consistency

Summary: Lens 'consistency' on runde1b_plan.md against the four part files. Checked and consistent: per-session minutes, account totals and tier totals (A 15.49 h, B 15.12 h, tier 1 30.60 h, tier 2 7.50 h, pilot 1.14 h; every part's session arithmetic reproduces), the key registry against R.1/D.3/K.11/L.2, the gate index against R.4/D.18/K.6/L.9, the round-1/round-2 file and module sha256s (all match the repository), CF-R+/CF-R-/CF-D/CF-L/CF-HERR membership and attainability, and the start-condition graph (the one exception is K7a). None of the 11 open issues in plan_assembly_result.json is fixed in the current files. They are all still present and appear below as consistency-2/3/6/7/11-15. New findings: (1, high) CF-K-link can never confirm 5 of its 7 members. Their p_IUT uses a rank p against 39 sets, whose minimum is 0.025, while Holm over seven needs 0.0071 at the first step. A numpy check (scratchpad r1b/rev_consistency/holm_check.py) shows only the two K6e-c members can ever pass, so the user's central 'field of view correlates with H-neurons' link (K6b) can never be round-level confirmatory. (5) The user's atlas-compatible H-neuron request (review_notes_K.md) is missing from main §1, §2.5 and §4 and from part K, and K has only 0.05 h left under its cap. (8) The 'declared not run' file lives only in account A, so B sessions cannot see it and the variant-E fallback of r1b_K_S2a cannot be taken. (9) G-K11 contradicts SG10. (10) Trim triggers conflict. (16) Q1/Q2 are undefined and collide with K's family names. (17) The file names in §0.1/§9.1 differ from the drafts committed in the frozen folder. Section 4 and §0.3 cannot yet take a tier-2 part P, for three reasons: r1b_ana runs before every tier-2 analysis; 'five files' and 'four answers' are hard-coded; and P's letter and prefix collide with the pilot r1b_P0/'r1bP0_', L's set P and R's P1-P5 (consistency-3, -4). Packing simulation (scratchpad r1b/rev_consistency/pack_check.py): account A ends at 9.59 h, R's chain at 5.80 h; r1b_D_A starts at 0.48 h, before the SG8 preview exists (2.11 h).

- **consistency-1** [high] main §2.4 CF-K-link row (runde1b_plan.md:328), C25 (:407); K.4 (runde1b_plan_teil_K.md:232, 248-256, 264); K6b rule (K:735), K6c rule (K:769), K6h (K:924); K.7 appended label (K:1313)
  - Problem: Five of CF-K-link's seven members can never be labelled 'confirmatory (CF-K-link)': K6b(i), K6b(ii), K6c(i), K6c(ii) and K6h. Their member p is p_IUT = max(p_effect, p_rank) with p_rank against 39 random sets, whose minimum is 1/40 = 0.025. Holm over seven at 0.05 needs p <= 0.0071 at step 1, and steps 3-5 need <= 0.010 / 0.0125 / 0.0167. Only K6e-c push and clamp (PI p against RS20) can pass. The observational H-link K6b is the user's 'Blickbereich ... korreliert mit H-Neuronen' question, and it can never be round-level confirmatory. K.7's appended 'A+ confirmatory' can then only come from K6e-c. This contradicts K.4's own rule that rank p is used only in families of at most 2 members.
  - Evidence: K:232 'Rank form ... N_R = 39. Used only in families of at most 2 members, where 1/40 = 0.025 meets the first Holm step.' main:328 'Member p: p_IUT (K.4) | Holm over seven, 0.05'. Q4/Q5/Q12 specificity 'rank, RS39(mc)' (K:248-256); K6b: 'p_IUT = max(p(β_H ≤ 0), p_rank(β_H vs β_R))' (K:735). Numpy check (scratchpad r1b/rev_consistency/holm_check.py): Holm-7 thresholds [0.00714, 0.00833, 0.01, 0.0125, 0.01667, 0.025, 0.05]. Best case (rank members at 0.025, K6e-c at 1e-6): only K6e-c push and clamp are rejected. With K6e-c not reached, nothing is rejected. K's own Holm-2 families and CF-HERR remain attainable.
  - Fix: In main §2.4's CF-K-link 'Member p' column and in K.4:264, compute the specificity component of K6b(i), K6b(ii), K6c(i), K6c(ii) and K6h for the round-level family with K.4's prediction-interval form against the same RS39 / RS39mc sets (p_PI with N_R = 39). K's own Q4/Q5/Q12 strings keep rank p. Alternatively, replace Holm-7 by a fixed-sequence test at 0.05 in an order frozen now. Print the minimum attainable p of every CF member in the table.

- **consistency-2** [medium] main §0.3 (runde1b_plan.md:59), §2.4 CF-L member p (:334-337), §4.5 items 11-12 (:653-654); L3 combined lines (runde1b_plan_teil_L.md:853), X1 rule (L:1082), L4 H line (L:922), L.4 header (L:1056)
  - Problem: Open issue 0, still present: two tier-2 extensions change tier-1 rules. If X1 is funded and reads capture, L's attractor line becomes Holm over {A1, A2, A3}, which lowers A1/A2's step-1 level from 0.025 to 0.0167. Main's CF-L member p and L's line '(A1 or A2) and O' ignore A3. L's string and the round-level label can then disagree, and the funding decision changes a confirmatory tier-1 rule. X2 likewise turns L4's H line into 'Holm over up and down'. Both contradict §0.3 ('label only') and L.4 ('none changes a core rule'), and main §4.5 items 11-12 state the change outright.
  - Evidence: L:1082 'A capture verdict adds attractor route A3 to the L3 attractor line: Holm over {A1, A2, A3}.' L:853 '(A1 or A2) and O. Holm over A1 and A2.' main:334-335 'p(L3 attractor line) = max(2 x min(p_A1, p_A2), 2 x min(p_stick, p_pers))'. L:922 '(Holm over up; over up and down if X2 runs)'. main:59 'where a part attaches a tier-2 result to a tier-1 verdict, it is a label only'. main:653 'X1 (0.21 h): adds attractor route A3 to L's attractor line.'
  - Fix: Choose one option and freeze it. (a) Delete the A3 sentence at L:1082 and the 'over up and down if X2 runs' clause at L:922 (and in X2's rule). Report X1 capture and X2's down arm as qualifiers, and reword §4.5 items 11-12. Or (b) state both exceptions in §0.3 and define CF-L for both cases: without X1, as now; with X1 run, p = max(3 x min(p_A1, p_A2, p_A3), 2 x min(p_stick, p_pers)), with L's line '(A1 or A2 or A3) and O'.

- **consistency-3** [medium] main §4.2 rows r1b_ana (runde1b_plan.md:599), r1b_R_ana2 (:605), r1b_D_R (:606), r1b_K_S6 (:607), r1b_K_S7 (:608); §5 (:697, :704, :706); §0.3 (:57-59); RK-1 'Use' (runde1b_plan_teil_K.md:1219); K.7 'K5a and RK-2 are reported next to the composite'
  - Problem: Open issue 1, still present: funded tier-2 results have no route into runde1b_summary.json. r1b_ana (wave 6) waits for none of r1b_R_ana2 (wave 7), r1b_K_S6 / r1b_D_R (wave 6, after the analyses) or r1b_K_S7. §5's r1b_ana inputs omit r1b_R_verdict2.json and every RK-1, RK-2 and D reserve output. No session is named for RK-1's 'K6a statistics on K4's cells', because r1b_K_ana has already run, or for D's R1-R3 strings. Funded items are therefore missing from the summary or mislabelled: at r1b_ana time they are neither 'not funded' nor 'not triggered'. r1b_R_ana2 starts on 'r1b_R_S5L or r1b_R_S8 ran'. r1b_R_S5L can finish (wave 5) before RG15's J1/J2 is decided in r1b_R_ana, so pass 2 can miss R8.2, and r1b_R_ana is not among r1b_R_ana2's start conditions. A tier-2 part P inherits the same gap.
  - Evidence: main:599 'r1b_ana ... | 6 | r1b_R_ana, r1b_D_ana, r1b_K_ana, r1b_L_A, r1b_L_B'. main:605 'r1b_R_ana2 | ... | 7 | r1b_R_S5L or r1b_R_S8 ran'. main:704 lists r1b_ana's inputs as 'r1b_R_verdict.json, d_verdict.json, k_verdict.json, l_verdicts.json, all session manifests'. K RK-1 'Runs in. r1b_K_S6 (account A, after r1b_K_ana ...)'.
  - Fix: Add a final CPU step r1b_ana2 (account A, last wave). It starts when every funded tier-2 session is done, declared not run, or has its trigger recorded as not fired. It reads r1b_R_verdict2.json and the r1b_K_S6, r1b_K_S7 and r1b_D_R outputs, plus any tier-2 part, and rewrites runde1b_summary.json. List it in §4.2 and §5, and state where RK-1's K6a statistics and D's R1-R3 strings are computed (for example in-session). Change r1b_R_ana2's start to 'r1b_R_ana done, and each funded and triggered R tier-2 session (r1b_R_S5L, r1b_R_S8) done or declared not run'.

- **consistency-4** [medium] main §0.1 (runde1b_plan.md:15-23), §0.2 (:36-40, :49), SG1 (:422), §2.3 SO-KEYS (:265-270, :286), §4.3 (:612-619), §4.5 (:645-656), §5 (:706), §9.1 (:867-879), §9.4 (:920); R RG8a 'all of P1-P5'
  - Problem: §0.3 and §4 cannot take a tier-2 part P as written. SG1 checks 'the sha256 of the five plan files', §0.1, §9.1 and §9.4 enumerate five files, and §5 defines 'The four answers'. P therefore has no file slot in SG1's constant and no answer slot, and it needs a new §4.3 row and an insertion into §4.5's numbered order. The letter P is already taken: P is L's push/read set (§0.2), r1b_P0 is the pilot with 'P0-R/-D/-K/-L items', P0 is K's 16-token frame, P1-P5 are R's PC_PASS_1b conditions, and the pilot key prefix is 'r1bP0_'. The registry's no-collision argument ('the fourth character differs') fails for a part prefix 'r1bP...'. P's measurement IDs P0, P1... would collide with these under §0.2's rule that every measurement ID carries its part letter.
  - Evidence: main:422 'the sha256 of the five plan files'. main:706 'The four answers in runde1b_summary.json: R ... D ... K ... L'. main:49 'P is part L's push / read set'. main:270 "main | 'r1bP0_' | pilot texts". plan_assembly_result cons.fixed[20]: 'the fourth character differs ('r1b_', 'r1b0', 'r1bD', 'r1bK', 'r1bL', 'r1bP')'. The repository README already announces 'Teil P (Brücke zu Pythia-1.4B ...)'.
  - Fix: Replace 'five files' by 'the plan files listed in §0.1' in §0.1, SG1, §9.1 and §9.4. Replace 'The four answers' by 'one answer per part that ran (tier-2 parts via r1b_ana2, consistency-3)'. Free the namespace now: rename the pilot to r1b_PIL with keys 'r1bPIL_' and items PIL-R/-D/-K/-L, and rename L's set P to P_L. Give the bridge part IDs and keys with a reserved prefix, for example 'PB' and 'r1bPB_', registered in SO-KEYS. Add a placeholder tier-2 row for it to §4.2/§4.3 and a position in §4.5.

- **consistency-5** [medium] main §1 request table (runde1b_plan.md:100-124), §2.5 (:359-366), §4.2 r1b_K_S1 (:590), §4.3 K row and caps (:616, :626-628), §8.2; part K K3 (runde1b_plan_teil_K.md:413-523), K.8 (K:1359-1432); scratchpad r1b/review_notes_K.md
  - Problem: The user's request of 13.11.2026 for atlas-compatible H-neuron metrics is absent from the plan. It asks for: pole-separate activation statistics; damage per firing (KL over the next 16 tokens with the neuron zeroed at the firing token); modes and damage shares; effect-based simulatability as the inverse-free verifier of a neuron description; and a plain statement that a swap to Qwen3.6-35B-A3B is not a one-line change. There is no §1 row, no K measurement, no §2.5 reconstruction row for the simulatability generate-verify loop, no minutes and no tier. Part K has 0.05 h of headroom, so the requested ~30 A100-min cannot enter tier 1 without breaking K's 9.0-h cap or naming a trim. The request is therefore misrepresented by omission, and the budget does not account for it.
  - Evidence: grep over runde1b_plan.md and runde1b_plan_teil_K.md finds no 'simulat', 'firing', 'Qwen3.6', '35B' or 'MoE'. The only 'atlas' is K7a's 'arrow-vector atlas'. K3e reads write-vector poles only (K:483). main:627 'K 8.63 + its pilot share 0.32 = 8.95 h <= 9.0 h'. review_notes_K.md: 'target <= ~30 A100-min, otherwise as an extension'.
  - Fix: Add a §1 row (request; part K; 'K3h atlas-style H-neuron profile: methods only, no atlas data'). Add K3h in r1b_K_S1 after K2/K3e with its minutes. Either make it tier 1 and name the K trim that pays for it or raise K's cap explicitly, or make it tier 2 with rows in §4.2/§4.3 and a §4.5 position. Add a §2.5 row for it. Generator: descriptions from top-firing examples and K3e readouts. Verifier: simulated activations patched in on held-out text, the share of the zero-ablation KL recovered against a matched null. Attempt limit 3, abstain, verifier error measured on K3e's 20 token-detector neurons. Name the per-firing KL distinctly from SO-DMG 'damage'. Add to §8.2: Qwen3.6-35B-A3B not used (MoE router and experts, ~70 GB bf16 / ~140 GB fp32 > one A100-40GB, MoE-aware hooks and H-neuron search, rounds 1-2 comparability lost; a separate bridge plan).

- **consistency-6** [medium] K.3 SO-CH (runde1b_plan_teil_K.md:162), K7a How and Runs in (K:939, :962), G-K11 (K:1259); main §4.2 r1b_K_S4 (runde1b_plan.md:594), §2.2 A->B (:187-190), §3.3 SO-AX row (:536)
  - Problem: Open issue 2, still present. K7a runs in r1b_K_S4 (account B; starts after r1b_K_S1 and r1b_K_S3 only) on 'named subspaces at blocks {6, 10, 14, 18, 22}'. SO-CH defines these to include u*_b, u_T^v and u_sea^v (AX-1b at 6, 14, 18). Account B receives only so_ax_ustar.pt (u*_b at 14, 18, 6), and no AX-1b at all. u*_b does not exist at blocks 10 and 22, AX-1b(6) is an input of no K session, and G-K11 is checked only in r1b_K_S5. The session must crash, wait or silently drop channels, and the plan states none of these.
  - Evidence: K:162 'Named subspaces at blocks {6, 10, 14, 18, 22}: u*_b, u_T^v, u_sea^v (AX-1b at 6, 14, 18) ...'. K:939 'named subspaces at blocks {6, 10, 14, 18, 22}'. K:962 'Runs in. r1b_K_S4'. main:188-189 'It carries only r1b_R_S0's exports (dmg_ref_R16.json ...; so_ax_ustar.pt ...)'. main:594 r1b_K_S4 starts after 'r1b_K_S1, r1b_K_S3'.
  - Fix: Restrict K7a's named subspaces to the SO-AX-free channels (H-subspace, J-workspace, top-16 PCs), optionally adding u*_14, u*_18 and u*_6 read from r1b_xfer_A/so_ax_ustar.pt (then add that file to r1b_K_S4's start conditions). Change K.3 SO-CH to '(u*_b at 6, 14, 18 from U*; AX-1b(14), AX-1b(18) only in r1b_K_S5)'. Add K7a to main §3.3's SO-AX row and to G-K11's scope.

- **consistency-7** [medium] main §2.3 SO-DMG (runde1b_plan.md:234), §2.1 push notation (:174-175); L.1 (runde1b_plan_teil_L.md:161), L1(vii) (L:398), L3b (L:682), L4 C-Rup (L:885), X1 (L:1070-1071)
  - Problem: Open issue 3, still present. Main fixes one calibration rule: '6 evaluations for pushes and 8 for neuron scaling (K's NSCALE_DN: (alpha - 1) <- (alpha - 1) sqrt(target / damage))'. L.1 instead gives 'f <- f sqrt(target / damage), +-10 %, <= 6 evaluations' with f undefined, and applies it to random-set gains alpha_(r,±) (L3b), the partial-ablation lambda (L1(vii)), C-Rup gains and X1's sets. With f = alpha, the controls matched to H- (x 0) and X1's alpha-0 arm cannot move (0 x anything = 0). With f = lambda, more damage needs a smaller lambda but the update raises it. G-dmg then reads 'not testable (no damage-matched control)', which removes 'H-specific carry' and with it route A1 of CF-L's attractor member.
  - Evidence: L:161 'calibrate rule (f <- f sqrt(target / damage), +-10 %, <= 6 evaluations)'. L:682 'alpha_(r,+-) is calibrated by SO-DMG ... so that each set's damage equals the H push's damage of the same sign ... after 6 evaluations'. L:398 'z <- mu + lambda (z - mu) with lambda calibrated by SO-DMG'. main:234 '8 for neuron scaling (K's NSCALE_DN ...)'.
  - Fix: State L's operator forms in main SO-DMG and in L.1. Gains: (alpha - 1) <- (alpha - 1) sqrt(target / damage), as in NSCALE_DN, with 8 evaluations, and say whether alpha < 0 is allowed. Partial ablation: (1 - lambda) <- (1 - lambda) sqrt(target / damage), with lambda in [0, 1]. L's frozen fallbacks (size doubling, next set) apply after the 8 evaluations.

- **consistency-8** [medium] main §4.7 (runde1b_plan.md:684-686), §2.2 A->B (:187-190), §4.2 r1b_K_S2a start (:592), SG4 (:425), §9.4 operator (:929)
  - Problem: The only 'declared not run' record is dsse_runs/r1b_declared_not_run.json in account A, and A shares only dsse_runs/r1b_xfer with B, which carries only r1b_R_S0's exports. B sessions whose start conditions name B producers cannot see a declaration. These are r1b_K_S2a (explicitly 'or r1b_K_S3 'not run' after two attempts: variant E'), r1b_K_S2b, r1b_K_S4, r1b_L_A, r1b_L_B and r1b_K_S7. Under SG4 they keep waiting with exit code 4, and the frozen variant-E fallback can never be taken by the runner. Everything downstream stalls: the K lens arm, L5 and r1b_K_S5/r1b_K_ana in A.
  - Evidence: main:684 'declared 'not run' by the operator in dsse_runs/r1b_declared_not_run.json (account A; the runner checks the two attempts)'. main:187-189 'A shares only the folder dsse_runs/r1b_xfer with B ... It carries only r1b_R_S0's exports'. main:592 r1b_K_S2a start: 'r1b_K_S3's k5a_calib.json (or r1b_K_S3 'not run' after two attempts: variant E)'.
  - Fix: Put r1b_declared_not_run.json in dsse_runs/r1b_xfer, which A shares with B, or keep one file per account (B's declarations in B's dsse_runs, visible to A through dsse_runs_B), with each runner reading both. Amend §2.2's 'carries only r1b_R_S0's exports' accordingly.

- **consistency-9** [low] K.6 G-K11 (runde1b_plan_teil_K.md:1259); main SG10 (runde1b_plan.md:431), §3.2 G-K11 row (:496), OD-M5 (:790-792)
  - Problem: K's G-K11 condition reads 'package present and SG10 passed'. Main SG10's failure is label-only ('label 'SO-AX reproduction failed' on every SO-AX use; the round-2 file stays canonical'), and main's G-K11 row does not require SG10 to pass. If SG10 fails with the package present, G-K11 fails, but neither of K's failure branches applies (AX-1b present, package present). It is then undetermined whether K3f, K10a's named sources (Q14; falsifier 12) and K9's named channels run with a label or are dropped as 'axes unavailable'.
  - Evidence: K:1259 condition 'package present and SG10 passed; AX-1b(14) and AX-1b(18) present'. Failure branches: 'an AX-1b block missing ...' and 'without the package: SO-AX channels dropped'. main:431 SG10 'If it fails: label 'SO-AX reproduction failed' on every SO-AX use'.
  - Fix: Change K's G-K11 condition to 'package present; AX-1b(14), AX-1b(18) present'. Add the failure branch 'SG10 failed: SO-AX channels used with the label 'SO-AX reproduction failed' (main SG10)'.

- **consistency-10** [low] main SG7 (runde1b_plan.md:428), §4.5 'Tier 1 under a smaller budget' (:658-662); RG18 (runde1b_plan_teil_R.md:1561), R.5 (R:1624-1634); D.19 trims (runde1b_plan_teil_D.md:1225-1237); K.8 (runde1b_plan_teil_K.md:1425-1432); L.8 trim order (runde1b_plan_teil_L.md:1326), L.9 SG7 row (L:1340)
  - Problem: Each part has two trim triggers and no precedence between them. SG7 lists the parts' own triggers: R takes the smallest reduction level whose total is <= 12.5 h; D trims until <= 9.0 h; K trims at > 30 % slower rates; L trims at any rate below half its planning value. §4.5 adds a proportional cap (funded tier 1 x the part's share / 30.6). L.8 says its tier-1 trims 2-5 apply only under §4.5's proportional rule, which contradicts SG7 and L.9 (trim when a rate is below half). No rule covers a part whose entire frozen trim order cannot reach its proportional cap: R stops at RL2 (12.3 h total at half speed), and K has 'never dropped' items.
  - Evidence: L:1326 'trims 2-5 apply to tier 1 only under main §4.5's proportional rule'. L:1340 'SG7 (L's part) | L0 rates >= half of planning | Trim order applied before the freeze'. main:428 'L's trim order if any rate is below half its planning value'. main:658-661 proportional rule. R:1624 'RL2 keeps the total at 12.3 h'.
  - Fix: In SG7: each part applies its frozen order until both its own cap and, when the funded tier 1 is below the projection, its proportional cap are met. If the full order cannot meet them, the user decides before the freeze (more budget, or a named measurement dropped), and the decision is recorded in runde1b_freeze.json. Align L.8's sentence with this.

- **consistency-11** [low] main SO-DMG 'Reporting' (runde1b_plan.md:234), §4.1 (:563-565), §4.2 (:584, :592-593); K.3 SO-DMG (runde1b_plan_teil_K.md:196); K4 V8 (K:572), K6e (K:802), K3f (K:499), K10a (K:1131); R3.1 'Runs in' (runde1b_plan_teil_R.md:810-812, 826); R5.1 doses (R:1099), R5.3 (R:1164)
  - Problem: Open issue 5, still present and wider than reported. SO-DMG requires every calibrated intervention of R, K and L to report its damage on REF-U32. Only R's pushes in r1b_R_S2_*/S3/S6/S7 (4.0 min) and r1b_K_S1's calibrations (3.0 min) are located and costed. These have no report and no minutes: K4 V8's beta (S2a); K6e's NSCALE_DN push to delta2 and RS20 (S2b, which also lacks dmg_ref_R16.json as an input); K3f's and K10a's pushes (S5); R5.1's a_DN(b, u) at 8 blocks and R5.3's calibrated directions (r1b_R_S5).
  - Evidence: K:196 'The reporting for r1b_K_S1's calibrations runs in r1b_K_S2a (3.0 A100-min)'. main:584 '(R's REF-U32 reporting) | inside r1b_R_S2_*, _S3, _S6, _S7'. R:1099 'Doses {0.25, 0.5, 1, 2} x a_DN(b, u) (each direction's own 0.05-nat scale at that block)'. R:1164 'All directions are calibrated to dmg_sym 0.05 nats'.
  - Fix: Either narrow SO-DMG's reporting rule to 'the calibrations listed in main §4.1', or add the reports, with a few A100-min each, to r1b_K_S2a/S2b/S5 and r1b_R_S5 and add r1b_xfer_A/dmg_ref_R16.json to r1b_K_S2b's start conditions.

- **consistency-12** [low] main §2.5 D3 row (runde1b_plan.md:362), §0.3 (:60), §4.2/§4.3/§4.5; D.14 (runde1b_plan_teil_D.md:944-952), D.19 (D:1198-1252)
  - Problem: Open issue 4, still present. D3's unexpected-mode naming is a SO-GVRA reconstruction ('reserve-funded: tier 2', <= 5 raw min). It needs model runs in 16 fresh M-start contexts at the D3 length, after r1b_D_ana has found the clusters. It has no session, minutes or §4.5 position, although §0.3 says every tier-2 item's tier is in §4.2.
  - Evidence: D:951 'Cost <= 5 raw min, only if triggered (reserve-funded: tier 2, main §2.5)'. main:606 r1b_D_R content 'D's reserves in priority R1, R2 (conditional), R3' only. main:60 'The tier of every session and item is in main §4.2'.
  - Fix: List it as a conditional item of r1b_D_R ('if r1b_D_ana finds an unexpected mode', <= 7.5 A100-min with margin). Add it to §4.2, to §4.3's D tier 2, to §4.5 next to D's reserves, and to D.19.

- **consistency-13** [low] main §2.3 SO-WT103 (runde1b_plan.md:243), OD-M15 (:820-821); K3c (runde1b_plan_teil_K.md:445); L.1 REF-W fallback (runde1b_plan_teil_L.md:136)
  - Problem: Open issue 6, still present. K3c counts unigrams over 'the first 10M WikiText-103 train tokens', which overlap D's streams, K's 1M-token reference and the pilot passages that SO-WT103 allocates disjointly. L's REF-W fallback reads 200 test-split passages x 512 tokens (~102K tokens). The test split is allocated to L.R's 200 x 1,024 natural passages (~205K of ~0.28M tokens), which leaves too little for a disjoint fallback.
  - Evidence: K:445 'counts from the first 10M WikiText-103 train tokens'. L:136 'Fallback: L computes them on 200 WikiText-103 test-split passages (main SO-WT103) x 512 tokens'. main:243 'test: part L only (the 200 natural-position passages of L.R)'.
  - Fix: Add both uses to SO-WT103 and OD-M15. K3c's corpus statistics may overlap the train allocations (counts only, never an analysis unit). L's fallback reuses the first 512 tokens of L.R's 200 natural passages.

- **consistency-14** [low] main §4.4 (runde1b_plan.md:633-641), OD-M25 (:834-835), C28 (:410), §1 H-neurons row (:118)
  - Problem: Open issues 7 and 8, still present. Simulating §4.4's suggested slots with §4.2's minutes and start conditions gives three results. r1b_D_A starts at 0.48 h but r1b_K_S1 (SO-HN and the SG8 preview) ends at 2.11 h, so D's optional H covariate, which §1 lists, is never computed. r1b_D_C waits behind r1b_K_S5, which itself waits for r1b_K_S2a until 5.39 h, so account A ends at 9.59 h. R's chain ends at 5.80 h, so the stated critical path ('R's chain in A, about 5.6 h') is not the real one.
  - Evidence: scratchpad r1b/rev_consistency/pack_check.py: A1 ... K_S5@6.56h -> D_C@9.05h; A2 ... R_ana@5.80h -> K_ana@7.28h -> D_ana@9.59h; B1 K_S1@2.11h ... L_B@8.87h; B2 K_S3@0.48h -> D_A@3.85h -> K_S2a@5.39h.
  - Fix: In A slot 1, run r1b_D_C before r1b_K_S5. Either start r1b_D_A after r1b_K_S1, or state in §1 and OD-M25 that the covariate reads 'not run (H-set not available at start)' under the planned packing. Restate the critical path as r1b_K_S1 -> r1b_K_S2a -> r1b_K_S5 -> r1b_K_ana, then the D and round analyses.

- **consistency-15** [low] main §4.2 r1b_P0 (runde1b_plan.md:574), §4.3 (:619), §4.6 Drive (:672-673); R.0 (runde1b_plan_teil_R.md:18-21), R.5 'Budget argument' (R:1621), RG18 (R:1561); D.19
  - Problem: Three budget statements do not trace. (a) Open issue 9: r1b_P0's 68.2 A100-min is not decomposed. The parts give R 19.2, K 19.0 and L 19.0, and D gives no figure, while main's 1.12 h implies D ≈ 10.2; that sums to 67.4, although merging the pilots should save setup time. (b) Open issue 10: R says its 11.03 h is 'within the 12-h envelope' while RG18 and main §4.3 use a 12.5-h cap. (c) §4.6 says R's tier-1 I/O is 'about 0.9 h of R.5's 1.2 h; the rest is in R's conditional sessions'. R.5's table gives 66.3 min (1.10 h) in tier-1 sessions and only 5.8 min in conditional ones, so the Drive scenario gives about 29.4 h, not 29.5 h.
  - Evidence: pack_check.py: 'R tier-1 I/O 66.3 min = 1.10 h; conditional I/O 5.8 min; total 1.20 h'. R:1621 'within the 12-h envelope'. R:1561 RG18 '<= 12.5 h'. K.8 K0 15.0 + 4 setup; L.8 L0 10 work.
  - Fix: Give r1b_P0's composition (setups, items per part, shared items) in §4.2, and D's P0-D minutes in D.19. State that 12 h is the brief's envelope and 12.5 h is RG18's projection cap, or align the two. Correct §4.6 to 'about 1.1 h of 1.2 h; tier 1 about 29.4 h'.

- **consistency-16** [low] main §0.3 (runde1b_plan.md:57), §1 (:98), §2.4 table (:311), §4.6 (:671), OD-M3 (:787), §9.1 (:884), §9.2 (:901), §9.4 (:920); K.4 families (runde1b_plan_teil_K.md:245-262)
  - Problem: Main cites 'question Q1' (the funded tier) and 'question Q2' (confirmation of the readings 'H-Neuronen' and 'zu überprüfen'), but it defines no list of questions. The same file uses 'Q1-Q18' for K's Holm families, where Q1 = H validity and Q2 = what H encodes. Q2's answer, on which parts K and L rest, has no slot in §9.2's freeze order or in the freeze record; only Q1 has one.
  - Evidence: main:98 'both readings are put to the user for confirmation (question Q2)'. main:311 'Holm within Q1-Q18'. main:901 order: '... -> the user decides the funded tier (Q1) -> ...'. No other definition of Q1 or Q2 exists in main.
  - Fix: Add a short 'Questions to the user' list with IDs distinct from K's families (for example U1 funded tier, U2 readings). Add 'U2 answered' to §9.2's order and to the freeze record.

- **consistency-17** [low] main §0.1 table (runde1b_plan.md:17-23), §9.1 (:873-879), SG1 (:422); repository results/ju_space/plans/runde1b/ (commit e349ee2)
  - Problem: Main names the frozen part files runde1b_part_R.md ... runde1b_part_L.md. The drafts committed in that folder are runde1b_plan_teil_R.md ... _L.md. A runner whose SG1 constants are built from §0.1/§9.1 would look for files that do not exist and stop every session.
  - Evidence: main:20 '| runde1b_part_R.md | part R ...'. main:876 '| runde1b_part_R.md | <sha256> |'. `ls results/ju_space/plans/runde1b/` shows runde1b_plan_teil_D.md, _K.md, _L.md, _R.md (git show --stat HEAD e349ee2).
  - Fix: Fix one set of names now: either rename the files to runde1b_part_X.md as a step in §9.2's freeze order, or change §0.1 and §9.1 to runde1b_plan_teil_X.md.

## find:the main file:sessions

Summary: Lens 'sessions' on /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/plan/runde1b_plan.md (main §0.3, §2.1-2.2, §3.1, §4, §6.2, §9). In the citations, main = that file, teil_X = runde1b_plan_teil_X.md in the same folder, r2_run_session.py = /home/user/DSSE/analysis/ju_space/round2/r2_run_session.py, and the notebook builder = /home/user/DSSE/notebooks/build_runde2_gpu_notebooks.py.

**The budget arithmetic holds.** I re-added the session minutes with /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/review_sessions/budget_sim.py:
- Tier 1 is 30.60 A100-h: R 8.53, D 8.44, K 8.86, L 4.77. Account A carries 15.49 h, B 15.12 h.
- Tier 2 is 7.50 h (A 5.70, B 1.81). With everything funded plus the pilot the total is 39.24 h.
- Work is 18.4 h and setup plus I/O 3.0 h, so the half-rate figures reproduce: 39.8 h consumed, 58.2 h with the margin applied again.
- The largest session, r1b_D_A, is 3.37 h as planned and 4.47 h at twice the planned work, just under the 4.5-h guard.
- Simulating §4.4's suggested packing ends account A at 9.59 h and B at 8.87 h. This is consistency issue 8, so I do not repeat it.
- One small error: §4.6's Drive line puts R's tier-1 I/O at 0.9 h, but R.5's rows give 1.1 h (only 0.1 h sits in the conditional sessions). At 20 MB/s or more, tier 1 would be about 29.3 h, not 29.5 h.

Also not repeated from cons.open_issues: 1 (no path for tier-2 results), 2 (K7a's axes in account B), 5 (REF-U32 costing), 7 (timing of D's covariate) and 9 (pilot composition).

**The real problems are in the machinery around the sessions:**
- The 4.5-h stop rule has no mechanism inside a script. The round-2 runner checks the guard only between subprocess steps, and the plan never says what a 'step' is.
- Failure handling works per session. One exploratory or tier-2 step that fails twice voids tier-1 confirmatory results in the same session, which contradicts §0.3.
- Signals between the accounts have no channel that B can read: the 'declared not run' registry, integrity of the A-to-B transfer, and the trims that are to pay for FB-2.
- The pilot cannot get its code under §9.2, and decisions made on the pilot's draft code are never checked again on the frozen code.
- Runtimes left idle after a notebook ends, and attempts that only wait, are outside the A100 budget.

**Part P (tier 2).** §0.3 and §4 can take it after small edits: rename the pilot (its 'P0' and 'r1bP0_' collide with part P), replace the 'five files' constants, and add P's rows, drop-order positions and pilot items. Note that §2.4 makes every tier-2 result exploratory.

Sources for the Colab statements: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [Colab Enterprise idle shutdown](https://docs.cloud.google.com/colab/docs/idle-shutdown).

- **sessions-1** [medium] main §3.1 SG5 (runde1b_plan.md:426); §4.7 (:679-683); §6.2 r1b_run_session.py (:737); OD-M27 (:838)
  - Problem: SG5 requires steps of at most 5 min. The prescribed runner ('r2_run_session.py's structure') can stop only between subprocess steps, and the plan never says whether a step is a runner subprocess or a cached part inside a script. Both readings break something. If parts are subprocess steps, the per-step model loads are not budgeted and D's shared stream prefills are lost. If whole measurements are steps, the 4.5-h stop cannot be enforced.
  - Evidence: The round-2 runner:
- removes R2_T0 and R2_GUARD_S from every step's environment (r2_run_session.py:1060), so scripts cannot see the deadline;
- checks the guard only after a step returns (:1216);
- kills a step only after R2_STEP_TIMEOUT of 3,600 s (:967).

The parts build resumability inside their scripts:
- D: 'Resumable parts per (arm, length, stream)' with 'one prefill per stream and length' (teil_D.md:395, :144);
- K: 'steps ≤ 5 min, cached parts' (teil_K.md:1417);
- L: 'resumable parts/' (teil_L.md:268).

(a) Parts as runner steps. R's own rate is 0.5 min per model load per process (teil_R.md:1572). D, K and L budget one 4-5-min setup per session and no per-step loads (teil_D.md:1207, teil_K.md:1367, teil_L.md:1309). r1b_D_A has roughly 10^2 (arm, length, stream) parts. Loads alone would add tens of minutes, and each arm would prefill its 16K/32K streams again (D budgets 8.9 raw min of prefills in total).

(b) Measurements as runner steps. D1 main is 42.5 raw min (64 min planned) and K8b is 84.6 min planned. A session that crosses 4.5 h inside such a step runs on for up to the rest of the step, so neither the 4.5-h bound nor the design rules' cap of about 5 h per session holds. At twice the planned work, r1b_D_A already needs 5 + 2 x 131.5 = 268 min = 4.47 h.
  - Fix: In SG5, §4.7 and §6.2:
- define 'step' as a cached part of at most 5 min inside a script;
- have the runner export R1B_T0 and R1B_GUARD_S (4.5 h) to every script;
- each script checks the clock after every part. Past the guard it closes its part cache and exits with a reserved code, which the runner records as 'pending (time guard)' (not 'failed') before the final sync;
- the model is loaded once per script;
- replace the 3,600-s step timeout by a no-progress watchdog (no new part file for 15 min);
- add a selftest in which a mock script stops at a mocked deadline and resumes.

- **sessions-2** [medium] main §4.7 (:684-686); §2.2 B->A rule (:184-186); SG6 (:427); §4.2 rows r1b_K_S4, r1b_L_A, r1b_L_B, r1b_D_C (:594, :597-598, :588, :601-602); §0.3 (:59)
  - Problem: Failure handling works per session, but sessions mix confirmatory, exploratory and tier-2 steps. One step that fails twice, or one output that fails SG6, voids every measurement in its session. A tier-2 or exploratory failure can therefore delete tier-1 confirmatory results. This contradicts §0.3: 'No tier-1 rule needs a tier-2 output'.
  - Evidence: The rules that apply to whole sessions:
- '... may be declared 'not run' ...; its measurements read 'not run (session failed twice)'' (main:684-686);
- B->A copies happen only 'when r1b_<s>/session_done.json exists and no session_partial.json lies next to it' (:185-186);
- if SG6 fails, 'the session is 'incomplete' and no analysis reads its outputs' (:427);
- in the round-2 runner a session is done only if every step ran, was skipped, or is 'not run' for a gate reason (r2_run_session.py:1284-1286, :1035-1046). No step can be declared on its own.

Example 1: r1b_K_S4 holds K8a, K8b, K8c and K7a (main:594). Suppose K7a (exploratory, with 3.6 GB of fp16 local arrays, teil_K.md:1418) fails twice. Then the finished K8b and K8c outputs are never copied to A. CF-K-attr's three K8b members and CF-HERR's K8c (main:327, :330) get p = 1, and K9 in r1b_K_S5 loses its inputs.

Example 2: X2 and X6 run inside r1b_L_A; X1 and X3-X5 inside r1b_L_B; D5-x inside r1b_D_C (main:597-598, :601-602). A failing X2 blocks r1b_L_A's session_done.json, and with it r1b_L_B and r1b_ana. Declaring r1b_L_A 'not run' voids L1, L3b and L4.
  - Fix: - Make the declaration step-level: 'not run (step failed twice)', in a registry keyed by session and step. The runner then closes the session with session_done.json listing the declared step, and the finished steps are copied and used.
- Apply SG6 per output (only that output becomes 'incomplete').
- Order every session's steps tier 1 before tier 2 and confirmatory before exploratory:
  - r1b_K_S4: K8b, K8c, K8a, then K7a;
  - r1b_L_A: L.POP, L1, L3b, L4, then X2, X6;
  - r1b_L_B: core, then X1, X3, X4, X5;
  - r1b_D_C: D3, D4, D5, then D5-x.
- State in §0.3 that a failing tier-2 step is declared at step level and never blocks its session.

- **sessions-3** [medium] main §4.7 (:684-686); §2.2 A->B (:187-190); §4.2 r1b_K_S2a (:592); teil_D D.4 FB-2 (:297-300) and D.19 (:1238-1239)
  - Problem: Control signals between the two accounts have no channel:
- the 'declared not run' registry exists only in account A, but start conditions in account B depend on it;
- attempts that only waited count as attempts;
- D's FB-2 decision is made in B but is to be paid for by trimming a session in A that cannot see it.
  - Evidence: The registry is 'dsse_runs/r1b_declared_not_run.json (account A ...)' (main:684-685), while 'A shares only the folder dsse_runs/r1b_xfer with B' (:187).

r1b_K_S2a (account B) starts after 'r1b_K_S3's k5a_calib.json (or r1b_K_S3 'not run' after two attempts: variant E)' (main:592; teil_K.md:1375). r1b_K_S3 is the riskiest load (Qwen3.5-2B under transformers 5.22.2, hence G-K17 and G-K18). If it fails twice before its runner writes k5a_calib.json, r1b_K_S2a cannot read the declaration and waits. Then:
- K4, K4p, K6c, K6f, K6g, K6h, K10c and the gating export never run;
- r1b_K_S2b, r1b_K_S5, r1b_K_ana and r1b_L_B wait behind it;
- members of CF-K-link and CF-HERR, and L2-L5, are lost.

The round-2 runner counts every recorded attempt (r2_run_session.py:587 takes len(session_attempts.json)), including runs that only waited (recorded at :1396-1402; session_done_r2_ana.json shows two 'waiting' attempts). Two premature launches therefore satisfy 'two attempts'.

FB-2 costs 'about 59 raw min, funded by trims T1-T6', and 'the trims apply to the sessions not yet started' (teil_D.md:297-300, :1238). T4 trims D2 in r1b_D_B (account A). Under §4.4, r1b_D_B starts at about 2.0 h, but r1b_D_A's folder becomes copyable to A only when r1b_D_A ends, at about 3.9 h (my simulation). T1 now applies only to the tier-2 item D5-x.
  - Fix: - Keep the registry in dsse_runs/r1b_xfer/r1b_declared_not_run.json, which B reads as r1b_xfer_A.
- Count only attempts that passed SG1-SG4 and ran at least one step.
- Make r1b_K_S3 always write k5a_calib.json once its runner starts (recording variant E when K5a is 'not run'), so a declaration is needed only for a K_S3 that never ran.
- For FB-2, either:
  - have r1b_D_A export its G-LCE-1 decision (file plus sha256) through dsse_runs_B as soon as it is made, and make r1b_D_B wait for that file; or
  - remove T4 and T1 from FB-2's funding and state that the overrun stays in r1b_D_A.

- **sessions-4** [medium] main §2.2 A->B and SO-XFER (:187-190, :250); SG4 (:425); SG10 (:431); teil_L L.1 (:161-162)
  - Problem: The A-to-B transfer cannot satisfy SG4 and has no completion marker. B sessions will either wait forever or copy, and then keep, an empty or half-written r1b_xfer.
  - Evidence: SG4 requires every 'A -> B' copy to match 'the sha256 recorded by its producing session' (main:425). That record is r1b_R_S0's session_done.json in A's dsse_runs/r1b_R_S0, which B cannot read: only r1b_xfer is shared (:187). The exports carry the sha256 of their round-2 sources, not their own (:188-189).

The B->A completion test (session_done.json, :185-186) has no counterpart in r1b_xfer. The round-2 copy cell never overwrites a folder it has already copied (build_runde2_gpu_notebooks.py:225, :242).

So if the operator creates and shares r1b_xfer before r1b_R_S0 fills it, or Drive delivers one export before the other, a B session keeps the partial copy. Then:
- L's u*_14 line is dropped silently ('if it is missing, the line is dropped', teil_L.md:162);
- REF-R16 goes unverified, although every L calibration uses it (REF-U32, teil_L.md:161) and K's REF-U32 reporting needs it.

SG10's outcome must label every SO-AX use (main:431) but is not exported either, so L's u*_14 line in B cannot carry the label.
  - Fix: - r1b_R_S0 writes r1b_xfer/xfer_manifest.json last, holding the session, attempt, code commit, the sha256 and size of each export, and SG10's result.
- A B session copies r1b_xfer_A only when the manifest exists and every file matches it, and copies again when the manifest changes.
- SG4 checks A->B copies against this manifest.
- L takes SG10's label from the manifest.

- **sessions-5** [medium] main §9.2 (:894-903); §4.2 r1b_P0 (:574); §6.2 r1b_run_session.py (:737); SG3 (:424); teil_L L0 (iv) (:281-285) and L4 (:879)
  - Problem: The pilot must run round-1b code before the freeze. But committing round-1b code before the freeze commit is forbidden, and the notebooks get code only by cloning the repository. So r1b_P0 has no sanctioned path to its code, and its record, which fixes trims, L2a's length grid and L4's cache path, has no code identity. Decisions taken on draft code are never checked again on the frozen code.
  - Evidence: - r1b_P0 starts on 'harmonised part files and package drafts; before the freeze' (main:574).
- The runner has a '--pilot mode for r1b_P0' (:737), which is round-1b code.
- §9.2: 'Not allowed: ... committing round-1b code before the freeze commit' (:899-900); implementers write the code after the freeze (:925).
- SG3 for r1b_P0 checks 'round-1 and round-2 modules only' (:424).
- The notebooks clone the repository at a branch (build_runde2_gpu_notebooks.py:276).

The pilot's numbers are binding: SG7's trims (main:428), OD-M14's L2a grid (:816-819), and 'If (iv)'s carried-cache variant passes parity, L4 may use it' (teil_L.md:285; L4 uses 'Carried caches only if L0 (iv) passed parity', :879). L4 would then use carried caches on post-freeze code that was never checked for parity.
  - Fix: - Allow engineering-only pilot code (runner --pilot, a draft SO-LCE, the timing and hook harness, the K0 and L0 checks) to be committed on a separate branch before the freeze.
- Record its commit and module sha256 in runde1b_freeze.json next to P0's record, and check them in SG3 for r1b_P0.
- Require every pilot decision that depends on engine code to be gated again in session on the frozen code. L4 repeats L0 (iv)'s 4-run parity check in r1b_L_A before using carried caches and falls back to re-prefill if it fails. G-LCE-1, G-K20 and G-L2a-LCE already re-gate.

- **sessions-6** [medium] main §4.1-4.3 (:556-628); §4.7 (:679-681); §6.2 notebooks (:753)
  - Problem: The A100 budget counts only setup + I/O + 1.5 x work. The notebooks, built on the round-2 structure, never release the runtime when they end, and they run all setup before checking start conditions. Idle time after a notebook ends and attempts that only wait both use A100 time that §4 does not count.
  - Evidence: - The round-2 final cell only syncs, unmounts Drive and raises SystemExit on a non-zero code (build_runde2_gpu_notebooks.py:400-406). There is no runtime release anywhere in the builder.
- The cell order puts pip, clone, model download and the MOCK selftest before the main cell (:408). Only there does the runner check start conditions and record a waiting run (r2_run_session.py:1396-1402).
- Colab's FAQ says 'Idle timeouts only apply if code execution terminates' (research.google.com/colaboratory/faq.html), so the A100 stays assigned after the notebook ends. User reports (secondary, unverified) put the idle time near an hour, with compute units consumed.
- Tier 1 has 24 A100 notebook runs (23 sessions plus r1b_P0) and tier 2 six more, plus waits and continuations. At 20-60 min per run that adds about 8-25 A100-h to the 30.6 h (26-80 %).
- Round 2's r2_ana already logged two waiting attempts (session_done_r2_ana.json).
  - Fix: In build_runde1b_gpu_notebooks.py:
- wrap the main run so that the last cell, for every exit code 0-4, syncs, unmounts and calls google.colab.runtime.unassign();
- move the SG4 start check, including whether the second folder is ready, into the first cell, before pip, clone, model download and selftest;
- record the time from the first cell to release for each attempt, and report actual A100-h per account against §4.3.

Before the freeze, confirm that each account's compute-unit balance covers its share if everything is funded: A 21.2 h, B 16.9 h plus the 1.1-h pilot.

- **sessions-7** [low] main §2.1 Pairing (:164-165); SG2 (:423); continuations (:426, :682); in-session decisions (:687-688); teil_L L.2 (:196); teil_D D.2 (:117); teil_R R1 tau (:437-440)
  - Problem: The pairing rule is stated per session, but a continuation runs on a new VM, which may be a different A100 type. Nothing makes a pair (both arms and their shared prefix) or an in-session decision atomic with respect to a stop or disconnect.
  - Evidence: - 'Every paired contrast is computed within one session, on one kernel path, one batch shape and one GPU model' (main:164).
- 'Paired arms are never split across GPUs' (teil_L.md:196).
- SG2 admits 'an A100 (sm80, 40 or 80 GB)' (main:423).
- A continuation 'uses the same work and Drive folder' on a fresh runtime (main:682; build_runde2_gpu_notebooks.py:219).
- D's batch limits are tied to card memory: 'B <= 3 rows at 28-32K, <= 7 at 16K, fp32 on 40 GB' (teil_D.md:117).
- R1's reproduction threshold already changes 'when only the GPU model differs' (teil_R.md:437-440).

A stop or disconnect between the arms of a pair splits the arms across two VMs and possibly two card types. Affected pairs: K6e's twins (G-K21 needs >= 97 % identical), L3b's push/twin pairs (G-twin 31/32) and D's crop/copy reads. The CRN gates then fail ('numerical divergence', 'not decidable (common random numbers broken)'), or the rule is broken without anyone noticing.
  - Fix: - Define the cached part as the atomic unit that holds both arms of every pair and their shared prefix. A part left incomplete is recomputed whole on resume.
- Log the GPU name per part and form contrasts only within a part.
- Fix batch shapes to the 40-GB values on any card.
- Write each in-session decision (L*, FB-1/FB-2, G-K16, G-K22, b*, K5a's variant, RG11, RG13/RG14) as a step output before its first use, so a continuation reuses it.
- State whether A100-40GB and A100-80GB count as one GPU model for §2.1 and R1.

- **sessions-8** [low] main §0.1-0.3 (:15-60); §2.3 SO-KEYS (:270); SG1 and SG7 (:422, :428); §4.2 r1b_P0 (:574); §4.5 (:643-656); §9.1 (:867-879)
  - Problem: §0.3 and §4 can take a tier-2 part only with edits the text does not foresee, and the letter 'P' collides with the pilot.
  - Evidence: - Sessions are named 'r1b_<part>_<label>' and 'Every measurement ID carries its part letter' (main:36-39).
- The pilot is session r1b_P0 with items 'P0-R', 'P0-D', 'P0-K', 'P0-L' (:574) and key prefix 'r1bP0_' (:270). Part P's first measurement 'P0' and its natural prefix 'r1bP' would collide with both.
- SG1 checks 'the sha256 of the five plan files' (:422), and §0.1 and §9.1 freeze 'The five files' (:15, :867).
- SG7 needs each part's trim level from r1b_P0's rates (:428), but P0's item list has no part-P items (:574).
- §4.5's drop order is a closed list of 14 items (:648-656).
- §2.4 makes 'every tier-2 result' exploratory (:345-346), which also binds part P's verdicts.
  - Fix: - Rename the pilot now, e.g. r1b_PIL with items PIL-R/-D/-K/-L and key prefix 'r1bPIL_'.
- Replace 'five files' in §0.1, SG1 and §9.1 by 'the plan files listed in §0.1'.
- Add to §0.3: a tier-2 part enters before the freeze with:
  - its §4.2 rows: account (preferably B, which carries 1.8 h of tier 2 against A's 5.7 h), wave and start conditions;
  - a §4.3 row and positions in §4.5;
  - its own P0 items and SG7 trim rule.
  Its verdicts are exploratory at the round level (§2.4). A part added after the freeze enters only by amendment.

- **sessions-9** [low] main SG7 (:428); §4.5 last paragraph (:658-662); §9.1 package files (:887-892); teil_R RG18 (:1561) and R.5 (:1576)
  - Problem: SG7 and RG18 choose pre-registered design reductions from r1b_P0's rates using cost scripts that are not frozen. The step from measured rates to trims therefore cannot be reproduced from the frozen text.
  - Evidence: - RG18 picks the reduction level from 'r1b_P0's measured rates (P0-R items) in cost_R2.py' (teil_R.md:1561), a script in the scratchpad (:1576).
- D's and K's projections come from cost_final.py and cost_merge3.py (teil_D.md:1218, teil_K.md:1368).
- §9.1's package list contains none of these scripts (main:887-892).

The trims change verdict ingredients. RL1 changes R3.3's dose grid and k at blocks 18 and 6 (teil_R.md:1630-1631). D's T7 shortens D3's long length from 28,672 to 16,384 tokens, which is the length contrast of CF-D member (b1) (teil_D.md:1234).
  - Fix: - Add the four cost scripts and their rate-input schema to the package files, with sha256 in the freeze record.
- SG7 records, per part, the script sha256, the measured-rate file, and the resulting projection and trim level.

- **sessions-10** [low] main §4.6 'Optional packing saving' (:674-675) vs SG2 (:423) and §9.3 (:914-916)
  - Problem: The optional move of r1b_D_ana to a CPU runtime conflicts with SG2 and may change the values it returns, so calling it 'a packing choice' is not well defined.
  - Evidence: - §4.6: '... on a CPU runtime it would cost no A100 time (-0.5 h) ... A packing choice, recorded in session_done.json' (main:674-675).
- SG2 requires 'an A100 ... for every ... analysis session that is planned on an A100; a declared CPU runtime for r1b_ana', and if it fails the 'session stops' (:423).
- r1b_D_ana is planned on an A100 (0.54 h, teil_D.md:1214).
- If its fits, bootstraps or envelope simulations use torch on the GPU, a CPU run changes RNG streams and float reductions. §9.3 makes any change that alters a returned value an amendment (:914-916).
  - Fix: Decide before the freeze, one of:
- declare r1b_D_ana a CPU session in §4.2 (add it to SG2's declared-CPU list, draw all its random numbers from CPU numpy generators so values do not depend on the runtime, and drop its 0.54 A100-h); or
- delete the option.
