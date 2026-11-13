# Unverified review findings for D (finder stage only; verification and fixes were stopped to save quota)

## find:part D:stats

Summary: I reviewed Part D through the statistics lens and re-ran the plan's own numpy simulations plus new ones. My scripts are in /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/review_D_stats/: rv_prior_capture.py, rv_hedge_capture.py, rv_hedge_clip15.py, rv_retok.py, rv_d3.py, rv_d5.py, rv_power_planSD.py, rv_power_b1het.py and rv_mroute.py.

Status of the judges' three flagged problems:
- **Pooled carry maps without item effects:** fixed. The map was dropped (D.20), and D3 now uses a per-item rule that needs both readouts to agree.
- **Parity gates defined per element:** fixed. G-LCE-1 to 4 use nats on the readout log-odds, or token identity.
- **Saturation and per-item prior in Phi:** only partly fixed. Prior strata and a fitted null envelope were added, but the survival checks built on them are mis-specified:
  - The prior check labels a real, uniform 1.5-nat minority bonus 'explained by prior x saturation' in 13 of 27 configurations once priors have an SD of 1-3 nats (stats-1).
  - The hedging check labels pure capture with no hedging 'explained by conflict hedging' in the plan's own simulation (clip readout, c 8) and in every configuration where the ±15 analysis clip is the only saturation (stats-2). The filler-form check has the same flaw.
  - The vector-valued prior check has no rule for combining its two components (stats-6).
  - The same saturation artifact remains in D1-r, where 'token-identity-bound' appears without any token-identity effect (stats-7).

Other real problems:
- **D3 negatives (stats-3):** 'no split at any length' and 'no length trend' are failures to reject on 16 items. Under the plan's own alternative they occur in 18-31 % of runs. This contradicts main §2.4's rule that underpowered nulls never count against a hypothesis.
- **D5-a rule order (stats-4):** 'horizon too short' comes after 'no lifetime effect', so with heavy censoring the rule returns the decisive negative in 70-93 % of runs.
- **D.17 power statement (stats-5):** it is overstated:
  - part (b1) is 0.18 after Holm, not 0.41;
  - all three parts together are 0.07-0.09, not 0.15;
  - 'not supported' under the null is 0.51 / 0.26, not 0.57 / 0.32;
  - part (b2) is carried entirely by D5-a, because the D2 catfish route has 2-7 % power.

Smaller issues: an inconsistency in how D2-c enters the support and negative conditions; contradictory filler-stream mappings in D1-f and D1-s; the FB-2 fallback leaving D1-a undefined; G-D1-null being unspecified and uncosted; per-component MDEs (for example D5-a's real MDE is about 620 tokens, not 350); D5-x giving an against-label on censoring ties; D4 class verdicts decided by plurality alone; and 'explained by filler form' being counted against a hypothesis in which the user said form matters.

- **stats-1** [high] D.5 Rule D1-a verdict 4, prior x saturation check (runde1b_plan_teil_D.md:432-433, :440); survival rule D.3 (:217-221); G-D1-null (iii) (:281-283, :316)
  - Problem: The prior check turns a real, uniform minority bonus into 'not minority capture: explained by prior x saturation' whenever items have moderate per-item priors. That string overrides the others and counts against part (a) (D.17 :1118). There are two causes. (i) theta_k uses max(0.25, env_opp), but the null of the opposed stratum lies below 0.25 (env_opp 0.21-0.24), so a real effect must first lift Phi_opp back to 0.25. (ii) Under saturation, the opposed stratum's mixed contexts sit on the flat part of the readout, so the same latent bonus moves Phi_opp less than the pooled Phi. The fixed ratio 0.5 ignores this. G-D1-null's planted-effect validation reports only verdict 4's level and Delta conditions, not the survival strings, so this misfire stays invisible.
  - Evidence: Forward re-simulation with the plan's own model (review_D_stats/rv_prior_capture.py): l = c tanh((beta E + s b + eps)/c); order + reverse; per-swap endpoints as in D.5; env = max over the plan's 8 kernels at the true parameters, which is optimistic. True 1.5-nat bonus at L*, c 6, beta 16->6, prior SD 2 nats: Phi(L*) 0.361, env 0.250, Phi_opp 0.292, env_opp 0.225. So theta = 0.111 and theta_k = 0.042 < 0.056, giving 'explained'. With prior SD 3, theta_k = 0.022. At c 10, prior SD 3, beta 16->10: theta 0.056, theta_k 0.006. Over 27 configurations with theta >= 0.05 (c 6/10, prior SD 1-3, bonus 1-1.5 nats), 13 read 'explained', including 10 of 12 at c 6. With prior SD 0, every configuration survives. The Delta component survives throughout (e.g. 0.184 vs 0.160).
  - Fix: Estimate the minority bonus on the latent scale per stratum: add one bonus parameter for mixed contexts at L* to G-D1-null's forward readout fit (a forward regression, allowed). 'Survives prior' if LB(eps_opp) > 0 and eps_opp >= 0.5 eps_al. 'Explained' if UB(eps_opp) < 0.5 eps_al and the interval of the difference excludes 0. At minimum: use env_opp without the 0.25 floor in theta_k, and replace 0.5 by 0.5 q, where q is the opposed/pooled excess ratio that a uniform bonus produces in the fitted model. Run all three survival checks on G-D1-null (iii)'s planted effects, using the fitted per-item priors, and report the rate at which planted effects read 'explained'. If that rate is > 0.2, the string reads 'inconclusive on prior' instead of 'explained by prior x saturation'.

- **stats-2** [high] D1-a verdict 4: hedging and filler-form checks (:434-435, :441-442); hedging control (:412-418); D.17 (:1118)
  - Problem: Both checks use theta = Delta Phi_24. Delta Phi_24 includes the rise of Phi that degradation produces by releasing saturation, which the null permits up to 0.25. By design, Psi_Delta subtracts that rise, because Delta Phi^C_24 rises by the same release. So Psi_Delta < 0.5 theta whenever the release part of Delta Phi_24 is comparable to the capture part, and a pure B-specific capture with no hedging at all reads 'not minority capture: explained by conflict hedging'. The filler check fails the same way if shuffled and coherent filler weaken evidence by different amounts. G-deg cannot flag this regime: with the analysis clip at ±15 the endpoints stay clipped.
  - Evidence: The plan's own script, merge_D/sim_strata_hedge.py (re-run), gives 'capture clip c8 tau 2: dPhi +0.155, dPhiC +0.090, Psi +0.065'. Since 0.065 < 0.078, this reads 'explained'. For tau 1e6: +0.194, +0.125, +0.069 (< 0.097), also 'explained'. Extended run (rv_hedge_capture.py): clip c 8, bonus 1-1.5 nats, beta 12->8 or 14->6: 8 of 8 'explained'. tanh c 6, beta 14->6, bonus 1.0: Psi 0.060 < 0.063. With the ±15 analysis clip as the only saturation (rv_hedge_clip15.py), beta 24->15 or 28->14, bonus 1.5-2 nats: 8 of 8 'explained' (Psi 0.007-0.065 against 0.5 theta 0.056-0.107; dPhiC 0.10-0.15). In these runs rho_L is 0.93-1.00, so G-deg passes.
  - Fix: Hedging: test the discriminator directly against its fitted null. 'Survives hedging' if LB(Psi_Delta - Psi_null) > 0; 'explained by conflict hedging' if UB(Psi_Delta - Psi_null) < 0; else 'inconclusive on hedging'. Psi_null is the Delta Phi_24 - Delta Phi^C_24 that G-D1-null's fitted readout model predicts under degradation alone (C contexts have E in [-10, 0], inside the fitted range). Filler form: compare excesses over each filler type's own fitted envelope, theta_k = [Phi_24(L*, shuf) - env_24(L*, shuf)] - [Phi_24(L0) - env_24(L0)]. This needs X_N and (0,5,5) contexts on the shuffled streams at L*: 5 contexts per item, about 2.6 raw min. Validate both checks on planted effects, as in stats-1.

- **stats-3** [high] D.14 Rule across lengths (:935-943), MDE (:967-970); D.17 (:1119, :1156-1157); main §2.4 (runde1b_plan.md:316-317, :340-343)
  - Problem: 'No split at any length' (S < 4 at both lengths) and 'split at both lengths, no length trend' (neither McNemar p <= 0.05) are failures to reject on 16 items. Neither has a power condition or a realized-sensitivity condition. Yet both count against (b1) and feed the confirmatory 'H-DUAL not supported'. Main §2.4 calls that statement a 'conjunction of equivalence or bound rules' and states that 'an underpowered null never counts against a hypothesis'.
  - Evidence: rv_d3.py uses exact one-sided McNemar and per-item draws. Under the plan's own alternative (6 of 16 items gain a 6-SD split at L*, detection 0.70): 'no split at any length' in 0.18-0.24 of runs; 'split grows with length' in only 0.31-0.41 at p <= 0.05, and 0.12-0.14 at p <= 0.025. If the split grows from 4 to 8 items: 'split at both lengths, no length trend' 0.25-0.31, 'split grows' 0.12. With 8 items at a 5-SD split (detection 0.42): 'no split at any length' 0.54.
  - Fix: Make both negatives bound rules. 'No split at any length' only if the one-sided 95 % Clopper-Pearson upper bound on the number of truly splitting items at L* is < 4. That bound is corrected for the detector's sensitivity at the measured G-D3pc separation and for its false-positive rate. With 16 items this is reachable only at S(L*) = 0 (bound 3.8 items at detection 0.70). 'No length trend' only if the exact 95 % upper bound of b/(b + c) is <= 0.6 with b + c >= 8. Otherwise 'undecided (underpowered)', which D.17 treats as 'not decidable'. Alternatively, add items. Say in D.17 and main §2.4 which (b1) negatives remain reachable.

- **stats-4** [medium] D.16 Rule D5-a (:1058-1068); D.3 verdict precedence (:222); D.17 (:1121, :1160)
  - Problem: In first-match order, 'no lifetime effect of competition' comes before 'not decidable (horizon too short)'. When few runs degenerate before 4,096 tokens, both arms' RMST sit near 4,096 and Delta is near 0 with a small SD. The rule then returns the decisive negative, which counts against H-DUAL in D.17, instead of 'horizon too short'.
  - Evidence: rv_d5.py: 32 starts, no true effect, tau | degenerate ~ U(256, 4096). If 5 % of runs degenerate, the horizon condition holds in 100 % of replicates, but the rule gives 'no lifetime effect' in 93 % and 'horizon too short' in 6 %. At 10 %: condition true in 94 %, 'no lifetime effect' 70 %, 'horizon too short' 18 %.
  - Fix: Place 'not decidable (horizon too short)' directly after 'not testable' and before every support or negative string of D5-a. Do the same for D5-g.

- **stats-5** [medium] D.17 power statement (:1135-1146); D.0 (:26-28); D.5 MDE (:497-501); merge_D/sim_combined_power.py
  - Problem: The power figures for the combined verdict do not follow from the plan's own SDs and alternative, and they hide that part (b2) is decided by D5-a alone. (1) The simulation uses smaller SDs than the plan states. (2) It draws (b1) as Binomial(16, 0.35), which allows more discordant items than the 6 that can split under the stated alternative. (3) The M route is almost unreachable: p_M = max(p(Delta Pi_1), p(Delta Delta Pi_1), p(mu < 0.10)) must survive Bonferroni x2 and Holm. (4) The simulation fixes env at 0.26, but the rule compares the 5th percentile of Phi with the 95th percentile of env, unpaired.
  - Evidence: The simulation uses SD mu 0.040, phi 0.020, dphi 0.010, phio 0.028, dphio 0.014. The plan states 0.07 for mu (:855) and 0.022/0.011/0.031/0.016 (:497-498). Re-run with the plan's SDs (rv_power_planSD.py): global null 'not supported' 0.51 / 0.26 (stated 0.57 / 0.32); part (a) 0.22 (stated 0.27). With per-item (b1) draws (rv_power_b1het.py): (b1) supported after Holm 0.18 (stated 0.41); all three 0.07-0.09 (stated 0.15); (b1) reaches its negative in 0.22-0.25. M route alone, with D5 null (rv_mroute.py): 0.025 at the stated Delta Pi 0.20, Delta Delta Pi 0.15, mu 0, and 0.065 even at Delta Delta Pi = 0.23 (the D2-b MDE).
  - Fix: Re-run with the plan's SDs, per-item (b1) draws and env variability; print the M route's power separately. Compute Phi - env per bootstrap replicate on the same resample (the items are shared). Either restructure M (drop the mu component from p_M; use D2-c as a gate; consider one pre-specified D2 contrast), or state in D.0, D.17 and main §1 that the user's catfish claim enters CF-D only through D5-a.

- **stats-6** [medium] D1-a prior check (:432-433, :440); survival rule (:217-221); D.17 'Artifacts' (:1143-1144); D.5 Specificity (:502)
  - Problem: For the prior check, theta and theta_k are 2-vectors (level excess, Delta), but the survival rule is defined for a scalar. No combination rule is stated. Under the prior x saturation null, the Delta component of the opposed stratum rises with degradation and can 'survive' while the level component is 'explained'. Read as 'explained only if both are explained', the artifact gives 'inconclusive on prior', which counts as support. The specificity claim '0 of 4,000' rests on the simulation's unstated OR coding and on dphio = 0.0.
  - Evidence: merge_D/sim_strata_hedge.py (re-run), opposed stratum under the null for beta 16, 10, 6, 4: tanh c 6 |b| 5, uniform kernel: 0.023, 0.044, 0.081, 0.119. clip c 10 |b| 5: 0.023, 0.067, 0.182, 0.250. sim_combined_power.py's priorA uses dphio=0.0 and codes expl = level | delta | hedging.
  - Fix: State: 'explained by prior x saturation' if either component is explained; 'survives prior' only if both survive; otherwise 'inconclusive on prior'. Re-run the artifact scenario with Delta Phi_opp taken from the null model rather than 0.

- **stats-7** [medium] D.10 D1-r rule (:655-662); D.17 outcome 6 (:1155)
  - Problem: The judges' saturation problem is not fixed in D1-r. D = Phi_char - Phi_can uses each encoding's endpoints. That removes scale but not the saturation regime. If character-level reports only carry weaker per-report evidence (same mechanism, no token-identity effect), Phi_char rises toward 0.25 by saturation release, and the rule returns 'token-identity-bound'. That string counts against 'the duality is not inside one token'.
  - Evidence: rv_retok.py (tanh readout, order 1 only, 24 items). c 6, uniform kernel, beta_can 16, char evidence x0.3: Phi_can 0.100, Phi_char 0.222, D +0.122, rho_char 0.67. beta 12 x0.5: D +0.071, rho 0.79. c 10, beta 16 x0.3: D +0.075, rho 0.48. With SD(D) 0.025 (:668), the interval excludes 0 and |D| >= 0.07, while rho_char >= 0.25 passes the 'not decidable' guard.
  - Fix: Fit G-D1-null's readout model per encoding at L0. This needs X_N and (0,5,5) contexts in e1/e2, about 5 extra contexts per item and encoding. Call 'token-identity-bound' only if [Phi_char - env_char] - [Phi_can - env_can] excludes 0 with |.| >= 0.07. Otherwise label it 'saturation shift (rho_char = ...)' and do not count it against H-DUAL. Apply the same correction to D1-s's size contrast.

- **stats-8** [low] D.17 support condition M (:1106-1108) vs (b2) negative (:1120-1121); D2-c (:811-816)
  - Problem: M support requires only 'D2-c not merging'. So 'one entity captured', which D.17 lists as an M negative, and 'polarisation' both satisfy it, and a run can be M-supported and M-negative at once. D2-c 'coexistence' requires only LB > 0 for both Pi^OPP, while 'one entity captured' uses 0.10. A speaker whose Pi^OPP interval is (0.02, 0.09) therefore reads 'coexistence' (first match), although by the 0.10 threshold its stance has nearly vanished. p_M's third component tests mu < 0.10, which is not the stated condition.
  - Evidence: Quotes at :1106 ('D2-c not merging'), :1121 ('or D2-c merging or one entity captured'), :814 ('lower bounds of Pi_1^OPP and Pi_2^OPP are > 0'), :815 ('upper bound < 0.10').
  - Fix: Support M only with D2-c 'coexistence' (or 'polarisation'), and make 'coexistence' require LB > 0.10 for both Pi^OPP. Make p_M's components match the stated conditions, or use D2-c as a gate without a p-value.

- **stats-9** [low] D.2 (:142); D.3 (:193); D1-f (:621); D1-s (:712, :715); D1-a checks (:434-435); D1-h items (:384)
  - Problem: Shuffled versions exist only for streams 0-7. D1-f runs items 0-23, and by D.3 item i uses stream i mod 16, so items 8-15 have no shuffled version of their own stream. The 'same tokens' pairing therefore fails for a third of D1-f and for the filler survival check. D1-s puts items 0-23 on streams 0-7, contradicting 'same items, keys and filler' with the 1.7B arm. The survival checks compare theta (D1-a, 48 items) with theta_k from items 0-31 (hedging) or 0-23 (filler) without saying which items theta uses.
  - Evidence: Quotes: 'Sentence-shuffled versions of streams 0-7' (:142); 'item i uses filler stream i mod 16' (:193); 'F-sym items 0-23 at L* only, on the shuffled versions of streams 0-7' (:621); 'F-sym items 0-23 on streams 0-7' (:712).
  - Fix: Shuffle all 16 streams (8 more prefills, about 2.3 raw min), or restrict D1-f to items 0-7 and 16-23. Map D1-s items to streams i mod 16. Define theta and theta_k of each survival check on that check's own item set.

- **stats-10** [low] D.4 G-LCE-1 fallback FB-2 (:296-301)
  - Problem: Under the pre-registered fallback FB-2, D1-a has no defined final string, although the text says D1-a 'carries' FB-2. FB-2 runs only interleaved k = 0..4 for items 0-31 at 16K, one attribute per context. Verdict 4 needs env(16K) from G-D1-null's fit (X_N, (0,5,5) and pure contexts at that length). The hedging and filler checks need C contexts and shuffled filler at L*, which FB-2 declares 'not testable (engine parity)'. With one attribute per context, the per-stratum curves average different item-attribute sets at each k.
  - Evidence: Quote :298-301: 'interleaved k = 0..4, two swaps, order and reverse: 640 prefills ... The other D1 arms at long length read not testable (engine parity)'.
  - Fix: State that under FB-2 D1-a reads 'not decidable (envelope and survival checks not run under FB-2)'. Or budget X_N, (0,5,5) and C contexts for items 0-31 at 16K (about 400 full prefills, about 38 raw min), and assign attributes so that each item-attribute appears at every k.

- **stats-11** [low] D.4 G-D1-null (:272-283, :314-316); D.19 r1b_D_ana (:1214)
  - Problem: G-D1-null is not implementable as frozen. (1) The text does not say whether each of the 400 planted sets gets its own fit and envelope (2,000 refits). If it does, the cost is far beyond the session budget. If it does not, the planted nulls come from the same fitted model that defines env, and the gate can fail only through bootstrap undercoverage. (2) The fitted readout model has no kernel, so the kernel used for planted nulls is unspecified. (3) When pure and (0,5,5) log-odds sit at the ±15 clip, beta_i is identified only from below, and env then depends on where the optimiser stops.
  - Evidence: 400 sets x 2,000 refits x 576 per-item-attribute tanh fits is about 4.6e8 small nonlinear fits; at 10 µs each that is about 77 min. r1b_D_ana has 15 raw min for all D analyses. The fit formula at :274 has no kernel term.
  - Fix: Specify the procedure: per planted set, refit c_L only (per-item fits fixed) with 200 refits; draw the planted kernel from the 8 kernels; add one misspecified planted null (clip readout, a kernel outside the set) so that the gate can fail. Cap beta_i at the value where the fitted pure context reaches the clip, and report the share of capped fits. Cost the step in D.19.

- **stats-12** [low] D5-a MDE (:1090-1091); D5-r (:1072-1073, :1092); D1-d (:572-573, :584-585)
  - Problem: These stated MDEs are per component, not for the rules as used, so a null result cannot be read correctly.
  - Evidence: D5-a support needs point >= 500 (:1062): joint MDE at power 0.8 = 500 + 0.84 x 141, about 620 tokens (stated 350). D5-r support needs LB > 0.05: about 0.05 + 2.49 x 0.03 = 0.125 (stated 0.07). D1-d 'losing game' needs LB C(L*) > 0.01: 0.01 + 2.49 x (0.0024-0.0049) = 0.016-0.022 nats/token, plus the growth condition (stated 'about 0.01').
  - Fix: State the joint MDEs (D5-a about 620 tokens; D5-r about 0.125; D1-d 0.016-0.022), and use them in the 'cannot show' text and in the D.17 power figures.

- **stats-13** [low] D5-x label (:1081-1082); D.17 outcome 9 (:1160-1161)
  - Problem: 'No full-scale advantage' fires when no 2V-comp run outlives its 1V pair. If both runs of a pair are alive at the 20-min cap (censored near 24K), the pair ties, and 'outlives' is false. Three tied pairs therefore give the label with certainty, and it is listed as counting against H-DUAL. Even without ties, 3 exchangeable pairs give it with probability 1/8.
  - Evidence: Quote :1056: 'rows still alive at the cap are censored at the length reached (about 24K)'.
  - Fix: Ties at the cap should read 'not decidable (censored at cap)'. Require at least one uncensored run per pair. Remove D5-x from the 'against' list (3 pairs cannot carry an against-statement).

- **stats-14** [low] D.15 D4 rule (:997-1004); D.17 outcome 10 (:1162)
  - Problem: The evidence bar is asymmetric. 'Coexisting' needs share >= 0.4 and LB >= 0.25. The two outcomes that count against H-DUAL, 'one blended entity' and 'absorbed into the role', need only the largest point share, with no interval. A population sitting on a class boundary therefore yields an against-verdict by a coin flip.
  - Evidence: Population centred at (r, i) = (0.68, 0.68) with the planned per-trajectory SD 0.25 (:1014), independent: P(blended) = 0.468^2 = 0.219 and P(coexisting) = 0.468^2 = 0.219. A tie, decided by noise, gives 'one blended entity'.
  - Fix: Require, for every class verdict, share >= 0.4 with LB >= 0.25 and a positive LB of the margin to the runner-up (role bootstrap). Otherwise 'not decidable'. Calibrate 'split across seeds' against a per-role unimodal null with the measured trajectory noise.

- **stats-15** [low] D1-a filler-form check (:435, :442); D.17 (:1118); D.1 mapping (:98); D.9 (:617-618)
  - Problem: 'Not minority capture: explained by filler form' counts against part (a). But in that case capture is present (verdict 4 met); only its growth with length depends on coherent filler. The plan's own mapping treats the user's 'Form war relevant' as an element of H-DUAL (D1-f). Filler passages containing any value word are already removed (:139-140), so shuffling tests coherence, i.e. form, not lexical priming. Counting form-dependence against the hypothesis misrepresents both the measurement and the user's statement.
  - Evidence: Quotes: :98 ''Form was relevant' ... Form of the long context at fixed size (coherent vs sentence-shuffled filler)'; :442 ''not minority capture: explained by filler form''; :1118 '(a): D1-a verdicts 6, 7, 8 or explained by ...'.
  - Fix: Rename it 'length-driven minority capture (form-dependent: coherent filler only)'. Count it as support for (a) with a form qualifier, or at most as 'not decidable'. Keep only the prior and hedging checks as 'explained by' strings.

## find:part D:validity

Summary: The merged part D fixes two of the three judge concerns:\n- Parity gates are now defined on the readout statistic (G-LCE-1 max <= 0.02, median <= 0.002 nats over 16 readouts; token identity for decoding).\n- The pooled carry map was dropped, and D3's split rule works per item.\n\nThe third concern, saturation and per-item priors in Phi, is only partly fixed. The new survival checks misattribute real effects:\n- The prior-opposed check compares a stratum whose null sits well below 0.25 with the 0.25 floor. It also uses an undefined 2-vector reduction.\n- The hedging check counts degradation-driven saturation relief as hedging.\n- In numpy simulations, a real 1.5-2 nat capture is labelled 'explained by prior x saturation' (tanh readouts with prior SD >= 2 nats) or 'explained by conflict hedging' (the plan's own +-15 clip). The stated power assumes no priors.\n\nThe saturation confound remains in D1-r, D1-f and D1-s: evidence strength alone moves Phi by 0.05-0.15, which can produce a false 'token-identity-bound'. The filler-form check counts a form-dependent capture as an artifact, contradicting the user's 'form' remark.\n\nSeveral negatives that feed 'H-DUAL not supported' are reachable without power:\n- D3's positive control passes vacuously at planted separations <= 4.5 SD.\n- Unequal detector power across lengths turns an equal split into 'split shrinks'.\n- 'split at both lengths, no length trend' is a non-significance rule.\n- D5-a's horizon check comes after 'no lifetime effect', so a maximal catfish effect inside 4,096 tokens reads 'no effect' 53 % of the time.\n- D2-b's +-0.25 margin exceeds the plan's own alternative (false 'against' about 28 %).\n\nMapping problems:\n- (b1) is met by across-seed commitment or length-driven stickiness, with one entity per context (P = 0.63-0.96 in simulation).\n- D4's r is fact retrieval and its i is out-of-character compliance.\n- D5's voice separation is read with the start in context.\n- (b2) can be supported via D5-a without any hyper-long context, and D1-a never isolates 'above 4,096'.\n\nScripts: /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/review_validity_D/ (sim_opp_capture.py, sim_opp_capture_clip.py, sim_hedge_clip15.py, sim_strength.py, sim_d3pc.py, sim_sticky.py, sim_d5_horizon.py).

- **validity-1** [high] runde1b_plan_teil_D.md D.5 Rule D1-a step 4, prior x saturation check (lines 432-433, 440); D.3 survival rule (218-221); D.5 MDE/power (500-503); merge_D/sim_combined_power.py altA
  - Problem: The prior check compares the prior-opposed stratum with the 0.25 floor: theta_k = Phi_opp(L*) - max(0.25, env_opp). But any per-item prior pushes the opposed stratum's own null well below 0.25; the plan itself bounds it at <= 0.255. A real B-specific capture of the size the plan is powered for raises Phi_opp by only about 0.07-0.12. theta_k therefore ends up <= 0 or below 0.5 theta. The primary then reads 'not minority capture: explained by prior x saturation', which overrides the other checks and counts against part (a). Second, theta and theta_k are 2-vectors (level, Delta), while the D.3 survival rule is scalar, and the text never says whether 'explained' needs one component or both. The stated power (0.20-0.32 at 1 nat, 0.69-0.73 at 1.5 nats) assumes Phi_opp is about equal to Phi, i.e. no priors.
  - Evidence: /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/review_validity_D/sim_opp_capture.py uses the readout of merge_D/sim_strata_hedge.py, l = c tanh((beta E + s b + eps)/c), clip +-15, per-swap stratum endpoints, beta 12 -> 8, true bonus eps at L*, b ~ |N(0, sb)|.
- tanh c 6, sb 2, tau 2, eps 1.5: Phi(L*) 0.301 (verdict-4 level met); null Phi_opp(L*) 0.158; Phi_opp(L*) 0.235. Level component: theta 0.051, theta_k -0.015, a difference of about 3 SD, so 'explained'.
- c 10, sb 2, uniform kernel: theta 0.073, theta_k 0.033 (ratio 0.46).
- c 6, sb 4: theta_k -0.052 to -0.076.
- c 10, sb 4, tau 2: ratio 0.11.
- With sb 0 the ratio is 1.00.
- The Delta components survive (ratios 0.76-1.0), so the verdict hinges on the undefined vector reduction. sim_combined_power.py reads 'explained' if either component is explained, and its alternative sets phio 0.29 vs phi 0.30.
  - Fix: Measure the opposed stratum against its own null:
- level_k = Phi_opp(L*) - env_opp(L*), without the 0.25 floor; keep the 0.255 bound as a sanity label;
- Delta_k = Delta Phi_opp - Delta Phi_opp,null, from G-D1-null's fitted per-length readout;
- theta takes the same excess-over-null form.
State the reduction:
- 'explained by prior x saturation' only if both components are explained;
- 'survives' only if both survive;
- otherwise 'inconclusive'.
Re-run sim_combined_power.py with alternatives generated from the readout model with prior SD 1-3 nats, and restate the power in D.5 and D.17 before the freeze.

- **validity-2** [high] D.5 hedging control (lines 412-418); D1-a survival check 'conflict hedging' (434, 441); D.3 analysis clip +-15 (190-191)
  - Problem: Psi_Delta = Delta Phi_24 - Delta Phi^C_24 subtracts the change in the third-value contexts. Under plain degradation that change is positive (saturation relief). With a clipping readout, a real capture therefore fails 'Psi >= 0.5 theta' while Delta Phi^C's interval excludes 0. The primary then reads 'not minority capture: explained by conflict hedging' although no hedging exists. The plan's own analysis clip at +-15 nats makes clipping the expected regime whenever pure-context letter log-odds exceed 15 nats.
  - Evidence: The merge's own sim_strata_hedge.py 'capture' rows (no hedging):
- clip c 8, tau 2: dPhi +0.155, dPhiC +0.090, Psi +0.065 (Psi/dPhi 0.42);
- clip c 8, uniform kernel: 0.194 / 0.125 / 0.069 (0.36);
- its 'degrade' rows alone give dPhiC +0.02 to +0.125.
/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/review_validity_D/sim_hedge_clip15.py (raw pure log-odds 20-24, plan clip +-15, degradation to 10-16, true bonus 2 nats, no hedging):
- Phi(L*) 0.30-0.35 (verdict-4 level met), Delta Phi_24 0.115-0.20;
- dPhiC 0.06-0.14, i.e. 3-7 planning SDs;
- Psi/dPhi below 0.5 in 11 of 12 configurations (0.28-0.49), so 'explained'.
D.5 justifies the control by 'Psi +0.07 to +0.16 under capture', which tests Psi > 0, not the 0.5-theta criterion the rule uses. sim_combined_power.py assumed psi 0.05 vs dphi 0.06.
  - Fix: Subtract the degradation expectation from G-D1-null's per-length fit. The (0,5,5) contexts have the same non-evidence-slot geometry as the C contexts.
- Predict Delta Phi^C_24,null and Delta Phi_24,null.
- Set theta = Delta Phi_24 - Delta Phi_24,null and theta_k = theta - (Delta Phi^C_24 - Delta Phi^C_24,null).
- Read 'explained by conflict hedging' only if the C-context excess has lower bound > 0.
- Pre-register that if more than 20 % of pure-context l are clipped, both survival checks use the unclipped log-odds.

- **validity-3** [high] D.14 G-D3pc (lines 923-927), G-D0c (921-922), rule across lengths (936-943); D.17 (b1) negatives (1119, 1156); main plan §2.4 lines 340-343
  - Problem: D3's negative strings are reachable without detector power.
(i) G-D3pc passes whenever the observed planted share >= predicted share - 0.25. At planted separations <= about 4.5 within-mode SD the predicted share is <= 0.31, so the gate passes with 0-1 of 16 planted items detected, and 'no split at any length' then counts against (b1).
(ii) G-D0c accepts an L* readout that keeps only half the L0 separation. McNemar then compares detections made at very different power, so an equal split at both lengths reads 'split shrinks with length (urn-like)'.
(iii) 'split at both lengths, no length trend' is a non-significance rule (McNemar needs >= 5 net discordant items; stated power 0.41). Yet main §2.4 calls D.17's 'not supported' a conjunction of equivalence or bound rules.
  - Evidence: /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/review_validity_D/sim_d3pc.py reuses the frozen rule code of merge_D/sim_split_v.py: 8 + 8 planted seeds at n = 1; mixture and gap on probe and state; agreement >= 0.8.
- Predicted split share 0.007 / 0.04 / 0.17 / 0.31 / 0.49 / 0.78 at 2 / 3 / 4 / 4.5 / 5 / 6 SD.
- Gate threshold -0.24 / -0.21 / -0.08 / +0.06 / +0.24 / +0.53.
- Retention 0.5 maps 6 SD at L0 to 3 SD at L*: per-item detection 0.78 vs 0.04. Sixteen truly splitting items give S(L0) about 12 and S(L*) about 1; the exact McNemar p is < 0.001, so 'split shrinks'.
  - Fix: - G-D3pc: require predicted detection >= 0.7 at the measured planted separation, and observed >= predicted - 0.25, at each length. Otherwise read 'not decidable (split detector underpowered at L)'.
- Make the across-length strings decidable only if the planted detection shares at L0 and L* differ by <= 0.15. Otherwise read 'not decidable (detector power differs across lengths)'.
- Replace 'split at both lengths, no length trend' by an equivalence statement (the exact 90 % interval of the discordance difference inside +-2 items) or by 'undecided'.
- Correct D.17's (b1) negatives and the main §2.4 sentence.

- **validity-4** [high] D.16 Rule D5-a (lines 1058-1069); D.17 (b2) negatives (1120-1121)
  - Problem: Lifetimes are censored at 4,096 generated tokens. The support string needs a mean RMST gain >= 500 tokens, which is impossible unless about 25-30 % or more of 2V-agree runs degenerate. 'no lifetime effect of competition' (interval inside +-500, SD <= 200) is read before 'not decidable (horizon too short)' (< 20 % degenerate in both arms), and heavy censoring shrinks the SD. A maximal catfish effect inside the horizon therefore produces an 'against' verdict that feeds 'H-DUAL not supported'.
  - Evidence: /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/review_validity_D/sim_d5_horizon.py: 32 starts, degeneration onset uniform on 512-4,096, frozen first-match order.
- 2V-agree 15 % degenerate, 2V-comp 0 % (competition removes all degeneration): maximum mean Delta about 270 tokens. Outcomes: 'no lifetime effect' 0.53, 'not decidable (horizon)' 0.30, 'catfish' 0.05.
- 25 % vs 10 %: 'no effect' 0.26, 'catfish' 0.13.
- Only at 40 % vs 10 % does 'catfish' reach 0.55.
  - Fix: Put the horizon check first and tie it to reachability. D5-a reads 'not decidable (horizon too short)' unless 4,096 - RMST(2V-agree) has lower bound >= 1,000 tokens (twice the 500 margin). Only then evaluate the support and equivalence strings. D.17 counts no D5-a negative reached otherwise.

- **validity-5** [medium] D.5 D1-a filler-form check (lines 435, 442); D.0 (18-19); D.9 D1-f (618-630); D.1 mapping row 'Form was relevant' (98); D.17 item 1 (1149-1150)
  - Problem: A length effect that appears with coherent filler but not with sentence-shuffled filler is classified 'not minority capture: explained by filler form' and counted against part (a). This contradicts the plan's own mapping of 'form of the long context matters' to the user's 'Form war ja anscheinend relevant'. It also contradicts the user's notion of hyper-long context, which is a natural, coherent context. Shuffled filler is not an artifact-free reference: filler-induced priors are already absorbed by X_N at L*, and degradation is handled by the per-length endpoints and envelope. Separately, 'size, not form' (equivalence +-0.08) can be declared when the whole effect depends on form.
  - Evidence: Quotes at lines 435, 442, 98 and 627-628. With SD(D) 0.03 (line 634) and complete form-dependence of a Delta Phi_24 of 0.06 (true D = -0.06), P(90 % interval inside +-0.08) = Phi(3) - Phi(1), about 0.16.
  - Fix: - Remove filler form from D1-a's 'explained by' checks and from D.17 item 1.
- Attach D1-f as a label: 'length-driven minority capture (form-dependent: coherent long context needed)' or '(size-driven)'.
- Narrow 'size, not form' to a margin <= half the measured Delta Phi_24.
- Test D against its fitted-null prediction (see validity-6).

- **validity-6** [medium] D.10 D1-r rule (lines 655-660); D.9 D1-f rule (626-629); D.12 D1-s (718-721); D.17 item 6 (1155)
  - Problem: Phi is compared across conditions whose per-report evidence strength differs: character-level values, filler form, Qwen3-0.6B. Under saturation, weaker evidence alone moves Phi toward 0.25, with no change in any capture mechanism, and normalising by each condition's own endpoints does not remove this shape effect. The judges' saturation point was fixed for D1-a's level (0.25 floor, envelope) but not for these between-condition differences.
  - Evidence: /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/review_validity_D/sim_strength.py: odd readout, no prior, no capture, position-balanced orders.
- tanh c 6, uniform kernel: canonical beta 16 gives Phi 0.100. Half the evidence (beta 8; range ratio rho 0.88) gives 0.184, so D = +0.084.
- tanh c 6, tau 2: rho 0.77 gives D +0.051.
- clip c 8: rho 1.00, yet D +0.079 to +0.150.
With SD(D) about 0.025, D1-r reads 'token-identity-bound' (interval excludes 0, |D| >= 0.07) although nothing is token-bound. The guard rho_char < 0.25 never triggers in this range. D1-f's 'form matters' and D1-s's size labels shift the same way.
  - Fix: Test each between-condition difference against a per-condition fitted null of the same form as G-D1-null:
- add X_N and (0,5,5) contexts per encoding (D1-r), for shuffled filler at L* (D1-f) and for 0.6B (D1-s): a few raw min;
- use D' = D - D_null with its bootstrap interval;
- otherwise read 'not decidable (evidence strength differs)' when the fitted beta/c differ by > 20 % between conditions.

- **validity-7** [medium] D.1 mapping row 'A force arises' (line 79); D.14 SPLIT-RULE calibration (912-915), Shows (961-964); D.17 (b1) (1104); reserve R2 (1247-1250)
  - Problem: (b1)'s support ('split grows with length') can be reached by a dynamic in which every context holds one entity. The plan's own sticky-last-mention null, where the narrator keeps whichever bundle it last mentioned, is classified 'split' per item. If hyper-long context only lowers the per-answer switch probability (more self-conditioning or copying, a common long-context degradation), McNemar fires. The user says the split leaves two entities present that then catfish each other ('dass zwei Entitäten da sind, die dann wiederum ... Catfishing betreiben können'); across-seed commitment shows the opposite inside a context. No tier-1 control separates a splitting force from length-driven stickiness: R2 is tier 2, conditional, and outside D.17.
  - Evidence: /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/review_validity_D/sim_sticky.py: frozen conjunction rule; probe and state both follow the last mention.
- Per-item split rate 0.14 / 0.55 / 0.73 at switch probability 0.2 / 0.1 / 0.05.
- P('split grows with length') = 0.63 for 0.2 -> 0.1, 0.93 for 0.2 -> 0.05, 0.96 for 0.3 -> 0.1.
  - Fix: (1) In D.0, D.1 and D.17, restate (b1) as 'a torn entity commits to one of two states more often at hyper-long context (symmetry breaking across samples)' and say that the user's within-context 'two entities present' is tested only by D2/D5. Alternatively, require a within-context two-entity signature in splitting items, such as late answers still carrying both bundles as distinct referents.
(2) Add a tier-1 stickiness control, about 20 raw min:
- a 'two-others' start on 8 items at L0 and L* with the same keys;
- the narrator reports two friends living the two bundles, with the same values and token counts, and probe |l| <= 1;
- require M's McNemar growth to exceed the control's (item-paired exact test).

- **validity-8** [medium] D.15 D4 (lines 984-1008); D.17 item 10 (1162)
  - Problem: D4's role index r measures fact retrieval, not identity. The start is verified to carry the role facts (role-fact probe >= 1 nat on re-read), while the r = 0 anchor (pure assistant) has no role facts in context, so r >= 0.7 whenever the start's facts are retrievable. Classes then hinge on i, which is read with an explicit '(Out of character, briefly:)' instruction. PC-D6 passes for the same reason, because it instructs truthful out-of-character answers. As a result:
- 'coexisting' measures retrieval plus compliance with an out-of-character request;
- 'role not taken', 'blended' and 'hyper-long context moves the identity' (a paired change of mean r) are reachable through retrieval loss at 16,384 tokens.
  - Evidence: Lines 984-985 (INJ: role-fact probe >= 1 nat on re-read), 988-990 (r from role-fact probes, i from out-of-character probes), 993-994 (anchors: pure assistant without role, pure role with system prompt), 1005-1006 (across-length rule on mean r or i).
  - Fix: - Define r = first-person role-fact probe minus a third-person probe on the same facts ('In the conversation above, what is the speaker's occupation?'), normalised between the pure-role anchor and a third-person anchor with the same facts in context (the assistant narrating the character).
- Make an in-character identity probe without the out-of-character marker the primary i; keep the marked version secondary.
- Read 'hyper-long context moves the identity' only if the third-person retrieval probe is unchanged across lengths.
- Use u_AA (G-D9) as a co-primary role index.

- **validity-9** [medium] D.16 D5 How (lines 1033-1048); D5-a rule (1062-1063); Null/controls (1083); D.20 (1266)
  - Problem: (i) The voice-separation condition inside the D5-a support string is read with the 8 start statements still in context ('In the transcript so far, does Assistant P think ...'). The start's explicit stances keep g(last)/g(0) high even if the generated text has merged. This is the 'start recency mixes with carry' flaw the merge itself called fatal in D.20.
(ii) 2V-agree (8 statements of one pole) does not match 2V-comp (4 pro + 4 anti) in lexical variety, contrary to the stated control, and the repeated-4-gram lifetime depends on n-gram diversity. G-D6 validates the detector on corpus passages only, not on agreeing dialogues.
  - Evidence: Line 1047: the probe reads 'the transcript so far'. Lines 1062-1063: separation is part of the support condition. Line 1266: the merge dropped the dynamics contrasts as 'read with the start in context (fatal ...)'. Lines 1036-1040 vs 1083: the arm composition vs the 'matches lexical variety' claim.
  - Fix: - Read P and Q stances on the generated text only: the last 512 generated tokens, read alone in the D2 frame, without the start.
- Pre-register the start's distinct-4-gram ratio as a covariate, or build 2V-agree from two prompts' same-pole pools matched on distinct-4-gram count.
- Add 8 healthy agreeing two-voice texts to G-D6's validation set.
- Correct the sentence in Null/controls.

- **validity-10** [medium] D.13 Rule D2-b (line 808); D.17 (b2) negatives (1120); merge_D/sim_combined_power.py negM
  - Problem: 'no length interaction' uses an equivalence margin of +-0.25, larger than the effect the plan powers for (Delta Delta Pi_1 = 0.15 in its global alternative). Under that alternative this 'against' string is reached about 28 % of the time. The combined-power simulation used a +-0.15 margin, so D.17's stated error rates do not describe the frozen rule.
  - Evidence: P(|est| + 1.645 x 0.094 < 0.25 | true 0.15, SD 0.094) = Phi(-0.58) - Phi(-2.61), about 0.28. sim_combined_power.py uses 'np.abs(ddpi) + Z90 * SD['ddpi'] < 0.15', while the rule says '(-0.25, 0.25)'.
  - Fix: Set the margin at or below the smallest effect of interest (e.g. +-0.10). The string reads 'undecided (underpowered)' when the realized half-width exceeds the margin. Re-run sim_combined_power.py with the frozen margins.

- **validity-11** [medium] D.1 mapping (line 77); D.5 lengths and statistic (366-367, 386-387, 399-400); D.0 (b2) (23-25); D.17 (b2) (1106-1109); D.16 D5 (1043-1045)
  - Problem: No verdict isolates 'far above its natural measure', which the plan sets at 4,096:
- D1-a contrasts about 1,000 with 32,000 tokens; 4,096 is descriptive on 16 items only;
- D2 contrasts about 300 with 32K tokens, D3 0 with 28,672;
- an effect complete by 4,096 is therefore reported as hyper-long.
Separately, (b2) 'enabled by hyper-long context' can be met through D5-a alone, whose contexts never exceed about 5K tokens (start plus 4,096 generated, 'short context').
  - Evidence: Line 77: 'Natural measure = 4,096 ... D1 L0 about 1,000 vs 32,000'. Lines 399-400: Delta Phi = Phi(L*) - Phi(L0). Line 387: 'items 0-15 also run the main int arm at 4,096 (descriptive)'. Lines 1043-1045: D5 runs 4,096 generated tokens at short context. Lines 1106-1109: the D5-a route alone supports (b2).
  - Fix: - Run D1's main interleaved arm at 4,096 for all 48 items (a few raw min).
- Add the label '(present by 4,096: within the natural length)' when Phi(4,096) - Phi(L0) >= 0.5 Delta Phi.
- Rename the D5 route of (b2) to 'the two sustain a longer self-generated context (within the natural length)', or require D5-x (>= 16K, funded) for a hyper-long reading.
- State the length span in each support string.

- **validity-12** [low] D.13 D2-a and G-D2-PC (lines 285-288, 782-797, 819-822)
  - Problem: In OPP, the per-speaker probe's cross-talk, allowed up to 0.3 e_11 at turn 1, enters Pi_1^OPP's denominator but not Pi_1^NEU's. If speaker 2's stance signal decays at a different rate from speaker 1's, Delta Pi_1 is biased without any catfish effect. G-D2-null's planted nulls contain no cross term. The production-share check ('not robust to production share') applies only to verdict 4, not to the negative verdicts 5-6, although the NEU questioner shifts production share by design.
  - Evidence: Lean 2x2 definition: e_11^OPP(t) = a(t) - c(t), with c_12 the effect of speaker 2's stance on D_1; NEU has c = 0. With a(t) = K^(t-1), K = 0.47 and c(t) = 0.3 K2^(t-1):
- Delta Pi_1 = +0.10 (K2 0.2), +0.12 (K2 0.1), -0.11 (K2 0.7);
- at SD 0.066, a +0.12 bias reaches verdict 4 about 55 % of the time.
  - Fix: - Also compute Pi_1^OPP from the 2x2 own-stance effect on prompts 0-63, or run ALI on all 160 prompts.
- Label verdicts 4-6 'cross-talk-sensitive' if the two disagree.
- Add the measured cross term and its per-turn decay (Pi_2^OPP) to G-D2-null.
- Apply the matched-presence check to verdicts 5 and 6.

- **validity-13** [low] D.11 D1-v rule (lines 690-694)
  - Problem: D1-v's labels require no canonical offset structure. If canonical 3CosAdd accuracy is at chance (about 1/22 among 24 targets), 'offset structure survives retokenisation' (character-level accuracy >= canonical - 0.15) is automatic and 'token-bound' is noise. The chance floor (24 random quadruples) is listed but used by no rule, so the answer to the user's analogy aside can be meaningless.
  - Evidence: Rule text, lines 691-694. Null/controls (697) lists the chance floor, but no rule condition references it.
  - Fix: Require canonical accuracy above the chance floor's 95th percentile and >= 0.3 for any label. Otherwise read 'not decidable (no canonical offset structure at block b)'.
