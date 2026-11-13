# Unverified review findings for L (finder stage only; verification and fixes were stopped to save quota)

## find:part L:validity

Summary: Lens 'validity' review of part L. Line numbers refer to /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/plan/runde1b_plan_teil_L.md. The CPU checks (numpy/sklearn, no model) are in /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b_validityL/.

**What the merge fixed.** The judges' headline flags are fixed in the core:
- item-specific Markov surrogates;
- 99 controls, so rank p can reach 0.01;
- no 'agree' instruction in the duet;
- EOS allowed on L.R;
- R6 carry on self-reads, read from text instead of the mechanical in-window denominator;
- a single entity criterion;
- pre-answer and causal reads for the attractor line.

The verifier corrections in lf_holographic.md and lf_energy_ising.md are respected: equal partitions, rotated z, I3_c, XOR 'not testable', D_n - D_(n-1), Ye's lag convention, a supervenient Psi, the modularity matrix, the IFT check, and A_k read as non-stationarity.

**Four measurements still cannot decide what they claim (simulated):**
1. L4 re-pairs partners within (item, configuration), which removes the designed configuration contrast. Under the strong priming that G-duet requires, J_ex is about 0 even when A copies B. 'no detectable inter-agent coupling' would then be a false verdict, and 'merge' is unreachable.
2. L3a uses item effects, which leave the chain-level frailty that L.R-B's 8 seeds per item create. Simulated Delta gamma is +0.17 at a frailty SD of 0.5, a false positive in 5/5 runs. Stickiness, persistence and bistability (A2) can then all come from one phenomenon and give 'H-attractor' (CF-L).
3. L2b's realised iota is noisy, and partialling a noisy covariate (errors-in-variables) leaves 45-95 % of an information-driven trend in the partialled slope. A pure length artefact then reads 'H rises beyond information loss' (CF-L); the power table assumed this residual is 0.
4. L3d's reversible band ignores entry selection: a reversible AR(1) process exceeds it by 5-15x.

**Medium problems:**
- L5a and V-OWN: 'owned by the lens' and 'shared' overlap with no stated order, and a single lens-owned line yields 'shared'.
- L5c reads H after the operands of the step are already written.
- L3b, L3c, L1 and L2c lack selection, uncertainty or lexical controls where their verdicts need them.
- V-HOLO and V-CHAIN have no frozen mapping to the final answer, and the Loop link is positive by construction.
- The E-channel qualifier is degenerate wherever the statistic already partials entropy.

**Cost of the fixes:** about +15 A100-min of core work (+23 planned) and +7.5 tier-2 work. Issues already in cons.open_issues (X1/X2 changing core rules, SO-DMG's operator form) are not repeated.

- **validity-1** [high] L4 DUET: coupling statistic J_ex (line 894), agreement excess (901), Null/controls (934), MDE (942), G-duet (908); verdicts 911, 914-915, 918, 924-926
  - Problem: J_ex re-pairs partner series within (item, configuration), and the controls line says the re-pairing 'keeps item fields and priming'. Everything the configuration fixes therefore stays inside the null: B's primed stance in C-half, and the final value of a merge that the configuration determines in C-opp.

When B holds its noted value, which G-duet's AUROC >= 0.80 on first messages demands, the re-paired partner series nearly equals the real one. J_ex is then about 0 whatever A does:
- a neutral A that copies B gets the equivalence verdict 'no detectable inter-agent coupling';
- the collaboration line and DD collapse the same way ('no effect of H gain on coupling');
- a configuration-determined merge gives agreement excess 0, so 'merge into one entity' is unreachable.

V-CHAIN would then report the chain breaking at 'Ising-modellable' for a pair whose listener follows its partner completely. The MDE comes from sim_duet2.py with priming field hp 0.5, where B is off its prime in about 22 % of messages; that is not this regime.
  - Evidence: Script /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b_validityL/sim_duet_strongprime.py. Same generator and fit as scratch_L_merge/sim_duet2.py (32 items x 2 configs x 4 seeds, 12 messages, item fixed effects), 8 reps.

Planned J_ex (re-paired within item and configuration):
- hp 0.5 (B off-prime 22-32 %): +0.11-0.12 at J_c 0.2, reproducing the plan's 0.14.
- hp 1.5 (off-prime 3-13 %, first-message accuracy about 0.98): +0.019/+0.030 at J_c 0.2, against an MDE of 0.10; about +0.12 at J_c 0.5.
- hp 3.0 (off-prime <= 1.5 %): +0.001/+0.001 at J_c 0.2 and +0.008/+0.010 at J_c 0.5, SD <= 0.007, so the 90 % interval lies inside +-0.12. Meanwhile A's late stance differs between configurations by 0.72-0.98 (A copies B).

Re-pairing within item across configurations: -0.042..+0.027 at J_c 0 (item-field SD 0.5 and 1.0), +0.19..+0.24 at J_c 0.2 and +0.54..+0.60 at J_c 0.5, at every hp.
  - Fix: - C-half (A neutral): re-pair A's series with partner series of the same item from both configurations (derangements over the item's 8 runs). A's context differs between configurations only through B's messages, so the configuration contrast is coupling signal, while the shared item field stays in the null.
- Use the same re-pairing for C-doc, C-Hup/C-Rup (DD) and the held-out gain.
- C-opp: compute the agreement excess against within-item, across-configuration re-pairs. Primes and first speaker swap between configurations, so a shared item prior ends at the same value in both, while coupling follows the configuration.
- Write 'within item, across configurations' into lines 894, 901 and 934. Keep the within-(item, configuration) J_ex only as a descriptive 'seed-level coupling'.
- Re-run sim_duet2 at B off-prime shares of 1-5 % and freeze the MDEs and margins from it.

Analysis only; 0 A100-min.

- **validity-2** [high] L3a: HD (line 640), stickiness hazard (641), L.MARKOVSUR (642), BI rule (653), O (660); attractor line (853); L-CAL (b) (298); L.R-B (488); CF-L (main §2.4)
  - Problem: The fix for the judges' pooled-surrogate flaw works at the item level only.

L3a pools several chains per item:
- L.R-B: 8 keys per item at T 1.0;
- L.R-A: a T 0.7 and a T 1.0 chain per item;
- L3b's twin-0 rows, which reuse items 1-64.

Chains of one item diverge in content, so chain-level frailty stays inside every item stratum. It inflates gamma-hat, Delta gamma and HD just as pooling did, and most for a population that tracks content, which H does by selection. H then outranks the 99 random sets and SEL+.

BI (route A2) measures the same between-seed heterogeneity. A2 and O are meant as independent evidence, but one phenomenon can produce both: seeds diverging into regimes whose dynamics are memoryless. That alone would give 'H-attractor ... seen in natural dynamics', a CF-L member.

Two further gaps:
- BI has no SEL+ contrast (HD and Delta gamma have one), although P may be a correlated hull (H+).
- L-CAL (b) simulates one chain per item ('600 x 1,024'), so it cannot catch any of this.
  - Evidence: Script /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b_validityL/sim_sticky_chainfrailty2.py. 64 items x 8 memoryless chains x 512 tokens (L.R-B shape), gamma = 0, chain log-odds = item log-odds + N(0, sd_chain), 19 surrogates per run, 5 runs.

Planned test (item effects, item-specific surrogates):
- sd_chain 0: Delta gamma -0.010; gamma-hat above all surrogates in 0/5 runs.
- sd_chain 0.5: Delta gamma +0.168 (SD 0.023); above all surrogates in 5/5.
- sd_chain 1.0: Delta gamma +0.446; 5/5.

With chain effects and chain-specific surrogates: -0.011 / -0.016 / -0.019, and 0/5 in every case.

The planning MDE for gamma is 0.04 (line 666). Delta gamma grows monotonically with within-item heterogeneity, so the population with the most seed-to-seed divergence wins the rank test.
  - Fix: - Use chain effects in the exit hazard and chain x position-bin strata for HD.
- Fit L.MARKOVSUR per chain (chain effect + shared position profile); for L.R-A this also separates T 0.7 from T 1.0. The surrogate correction removes the incidental-parameter bias (simulated Delta gamma about -0.02).
- BI: in the surrogate, draw each seed's rates from the item's fitted between-chain distribution, so BI tests bimodality beyond unimodal frailty. Add 'BI_P - BI_SEL+ lower bound > 0' to A2.
- Extend L-CAL (b)-(d) to 8 chains per item with chain-frailty SD 0.5 and 1.0, size <= 0.10 required.

No extra A100 time.

- **validity-3** [high] L2b: iota definition (line 497), slopes (562-565), rules (572-573); L-CAL (g) (303); L.5 power row (1202); CF-L (1196; main §2.4)
  - Problem: iota(w) is a window mean of the realised per-token log-ratio log q(z_k|A) - log q(z_k|A'). It carries token-sampling noise around the information the anchor really carries, and partialling a noisy covariate removes only its reliable share.

beta_pos^(iota,e) therefore keeps (1 - lambda) of an information-driven trend, where lambda is the within-chain reliability of iota(w) after log position is partialled. That reliability is low, because most of iota's within-chain variation is the position trend itself.

Consequences:
- A pure length artefact reads 'H rises beyond information loss', a CF-L confirmatory member.
- 'length effect on H is information-limited' becomes unreachable: beta_iota is attenuated towards 0 and the residual slope leaves +-0.10.
- The bias runs towards the user's hypothesis and suppresses the round-level 'against' statement 'length artefact (information-limited), no H-attractor'.

The L.5 power row assumes the partialled slope is exactly 0 in the length-artefact scenario (scratch_L_merge/sim_combined3.py S1: bpi = 0.0). L-CAL (g) does not say whether H depends on the latent or on the observed iota.
  - Evidence: Script /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b_validityL/sim_l2b_eiv.py (sensitivity runs in sim_l2b_eiv_sens.py). 384 chains, 30 windows of 32 tokens at positions >= 64, H depends only on latent information, within-chain Frisch-Waugh slopes as in L2b; raw slope about 0.22 SD per e-fold.

- Noise-free iota: partialled slope -0.005; the 'beyond information' rule is never met.
- Per-token iota SD 0.3 nats: +0.150 (LB +0.117); rule met in 8/8 runs.
- Per-token SD 0.6 nats: +0.196; 8/8.
- Larger latent residual variation (SD 0.05 / 0.10 per window): +0.10 / +0.04 at 0.3 nats, and +0.17 / +0.11 at 0.6 nats.

Realised log-ratios on content tokens (entity names) easily reach several nats, so per-token SDs of 0.3-1 nats are plausible.
  - Fix: - Use the expected anchor information iota*(k) = KL(q(.|A_i, z_<k) || q(.|A'_i, z_<k)). Compute it position by position by running a teacher-forced pass of the true prompt in the same batch as the A' pass: about 0.56M extra tokens, about 1.9 A100-min of work. It has no token-sampling noise. Keep the realised iota as descriptive.
- Report the split-half (odd/even tokens) reliability of iota(w) after position is partialled.
- Make L-CAL (g) generate H from a latent information profile with the per-token noise measured from iota0 (A'' against A'). Require size <= 0.05 for 'H rises beyond information loss' there.
- Recompute the L.5 power row with that noise.

- **validity-4** [medium] L3d: entry events (line 809), d-hat pooling (816-818), reversible band (819), rule (832, 836); A_k reading (822); V-CHAIN non-equilibrium link (1185)
  - Problem: Post-entry windows start at a threshold crossing selected on a_P, so they contain a relaxation transient, and a reversible process gives D_3 - D_2 > 0 on such windows. The plan's own A_k reading names this effect: non-stationarity, not irreversibility.

The reversible band comes from position-matched surrogates of the quartile symbol series. Those surrogates cannot reproduce the entry selection, because b uses a Q95 threshold, finer than the quartiles. The matched windows contain no transient.

So both conditions of 'irreversible H-entry (entry-specific)' are met by reversible but persistent dynamics, and 'H-specific' follows whenever H is more persistent than the controls. The non-equilibrium link of V-CHAIN would then rest on selection.

Two outcomes are possible:
- if L-CAL (h) builds entries by the real rule, size fails and the line reads 'not testable';
- if L-CAL plants entries any other way, the real verdict is a false positive.

L3d also names no bootstrap unit, although events of one item or chain are correlated (lf_holographic RQA3).
  - Evidence: Script /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b_validityL/sim_dhat_band.py. Reversible Gaussian AR(1) latent; per-chain quartile symbols; entry = first exceedance of q95 after >= 16 tokens below; 1,000 post-entry windows of 64 tokens; Jeffreys 0.5; d-hat = D_3 - D_2.

- Post-entry d-hat: +0.0039 / +0.0086 / +0.0091 at phi 0.8 / 0.95 / 0.98.
- Band from position-matched windows without entry selection (100 draws): q95 +0.0009 / +0.0007 / +0.0006. The reversible process lies above the band in every case.
- sim_dhat_transient.py: matched windows (no entry within +-64) give +0.0015 / +0.0004 at phi 0.8 / 0.95, so the 'entry-specific' contrast is positive as well.
  - Fix: - Build the reversible null from the same events. Under reversibility a post-entry window is distributed as a time-reversed pre-exit window, where an exit is the last b = 1 before >= 16 tokens of b = 0. Test d-hat(post-entry) - d-hat(time-reversed, position-matched pre-exit windows) per population, and use the same contrast for the controls' rank.
- Alternatively, symbolise into 5 levels that resolve the Q95 level and re-detect entries inside each surrogate before computing its band.
- Make L-CAL (h) plant threshold-selected entries on reversible persistent latents (AR(1), phi 0.8-0.98), size <= 0.10 required.
- Resample items, with all events of an item together.

- **validity-5** [medium] L5a rule (lines 983-987), Omega (980), corpus of 256 known + 150 prone items (962); L.5 power rows (1209-1210); scratch_L_merge/sim_own2.py
  - Problem: Two L5a verdicts overlap:
- 'break-off owned by the lens': G_F(known) LB > 0 and Omega(all) UB < 0.5;
- 'shared': G_F(known) LB > 0 and Omega(prone) LB > 0.5.

Known items are 63 % of the 406 items and carry almost only FOREIGN growth, because M_i \ V_i is nearly empty when the model computes correctly. Omega(all) therefore falls below 0.5 whenever the lens confabulates on known items, whatever the lens reports on prone items.

The text fixes no evaluation order. sim_own2.py, which produced the power table, lets the last assignment win (shared over lens), while a top-down reading gives 'lens'. The verdict of the user's 'erweitern' test thus depends on an unstated order.

Two smaller defects:
- 'no growth' can hold together with 'owned by the model' when both growths are tiny.
- Omega's interval uses only replicates with a positive denominator, which contradicts main §2.4's 5 % undefined-replicate rule.
  - Evidence: Script /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b_validityL/sim_l5a_order.py, with sim_own2.py's planning SEs.

- 'shared' scenario (G_F(known) 0.08, Omega(all) 0.4, Omega(prone) 0.7): the lens condition holds in 0.88 of runs, the shared condition in 1.00. First-match reads 'owned by the lens' in 0.88; last-match reads 'shared' in 1.00.
- Worked example: G_F(known) 0.10, prone G_F 0.02, prone G_M^c 0.10 give Omega(prone) 0.83 but Omega(all) 0.34, so both conditions hold in 1.00 of runs.
  - Fix: - Define 'owned by the lens' with Omega(prone) upper bound < 0.5. Lens and shared then exclude each other.
- Require G_M^c(prone) lower bound > 0 for 'owned by the model'.
- State 'first match wins' in the order 'no growth', 'owned by the model', 'shared', 'owned by the lens'.
- Apply main §2.4's undefined-replicate rule to Omega.
- Recompute the V-OWN power rows under this order.

- **validity-6** [medium] V-OWN (lines 1168-1172); L.5 power rows (1209-1210); L.7 (1269)
  - Problem: 'shared' is assigned to 'any other combination with at least one positive line'. A single lens-owned line with the other line undecided therefore reads 'shared', and so do 'no growth' + 'owned by the lens' and similar pairs.

That string says the break-off and its H-pattern belong partly to the model, although neither line found model-owned evidence. This is the judges' 'OR combination that can hardly fail', and it turns lens-only evidence into partial support for model ownership.

L.7 counts only 'belong to the lens' against model ownership, so these lens-only cases never count against it.
  - Evidence: The plan's own table reports 0.18 'shared' in the lens-owned scenario and 0.14 in the model-owned scenario (lines 1209-1210). sim_own2.py sets pos_a = a in {'lens', 'model', 'shared'}, so any single lens line gives 'shared'. In that scenario L5b reads 'undecided' whenever |rho_b| exceeds about 0.054, about 18 % of runs at SE 0.04.
  - Fix: - 'belong to the lens': at least one line is lens-owned and none is model-owned, 'in both' or 'shared'.
- 'belong to the model': the symmetric rule.
- 'shared': only if L5a reads 'shared', or L5b reads 'in both', or one line is lens-owned and the other model-owned.
- Otherwise 'not decidable'.
- Print which line carried the verdict.

- **validity-7** [medium] L5c: parser and pre-step position (lines 1022-1024), hazard (1027); CF-HERR (main §2.4); part K's K1 item store ('Stored: P (operands), I (intermediates), I+ (one-operation alternative partials)')
  - Problem: (a) t_k^-, the token before the first digit of the result, comes after 'a o b =' has been written. When the first error lies in the operands or the operator (a wrong carried value, a precedence error, a misread prompt number), the error is already on the page. beta_H > 0 can then be detection of a visible error, while the verdict string, a CF-HERR member, says 'predicts the model's own step errors'.

(b) 'Step k is correct iff r equals the true intermediate v_k' counts valid alternative orders as errors and misplaces k_err, although K1 stores those alternatives as I+.
  - Evidence: L5c is the plan's test of whether H rises before the model's own errors (line 1019). For K1's 2-3 operation items with + / - / x, precedence and carry-over mistakes are visible in the operands before t_k^-. Nothing in L5c separates error types or reading positions.
  - Fix: - Set t_k^- to the last token before step k's line begins, so that no part of step k is visible.
- Classify each first error as a selection error (operands or operator not a valid next operation) or a result error (r != a o b on the written operands).
- Fit the hazard for all errors at line start (primary) and for result errors at '='.
- Count a step correct if r is in V_i or I+, or if it is a correct operation on valid available values.
- Freeze this in the parser and in L-CAL (n).

No A100 cost.

- **validity-8** [medium] L3b: conditions (lines 682-683), Delta K (699), specificity (714-717), labels (722), A1 (723), Null/controls (724); L-ALL amplifier reading (1162)
  - Problem: 'H-specific carry' compares K_H only with damage-matched random sets read on their own channels.

A population selected for a behavioural label reads a text property (hallucination-likeness, uncertainty) that persists through the model's own text, so its text-level carry beats random sets for that reason alone. L3b pushes neither SEL+ nor UNC+, so 'not beyond uncertainty' cannot even be computed there. Route A1 (CF-L) and 'H-specific amplifier, not self-sustaining', the plan's reading of the user's 'andere Mechanismen ... verstärken', therefore cannot be separated from a selection or uncertainty carry.

Separately, nothing makes a random push displace that random set's own channel in the text read back by the unmodified model. d_R^(1) may often be undefined, which gives 'not testable (random carries undefined)' and removes A1 altogether.
  - Evidence: The specificity rule uses only K_(R,i), i = 1..6 (lines 698-699, 714). The only L3b labels are 'high-damage push' and 'generic H rise under damage' (line 722). SEL+ and UNC+ appear in L3a and L3c but in no L3b condition (lines 678-683).
  - Fix: - Add damage-matched SEL+ and UNC+ push arms, both signs, turns 1-2: +8 texts per key, about +20 % of L3b's generation, about +7 A100-min of work (+10 planned).
- Compute K_SEL+ and K_UNC+ on their own channels.
- Attach 'not beyond uncertainty' or 'selection effect' to 'H-specific carry' and to the L-ALL amplifier reading.
- Pre-register what happens when fewer than 4 random K_R are defined, for example specificity from the SEL+ and UNC+ contrasts alone, labelled 'few random carries'.

- **validity-9** [medium] L3c: G-ESS (line 748), shared-driver refit (iv) (753), kinetic fields (763), G-cal-Ising P5 and its pass rule (774, 776), rule (779-780)
  - Problem: The shared-driver refit (iv) has fields only per (position bin x family), and the kinetic fields are per (family, T, bin). H's likely shared drivers vary within those cells:
- the chain's content regime (hallucinating or not);
- token-level uncertainty.

These drivers still create effective couplings that random sets lack, and that SEL+ lacks too, since its label (e.g. a 'Who' question) barely acts on the continuation. 'collective (Ising-modellable) H-population' can therefore pass 'with covariate fields (iv)' on shared input alone.

G-cal-Ising's P5 'hidden-field-driven' set does not say at what level its hidden field varies, so it may not test this.

The G-ESS fallback, 'the top 30 units by K's selection frequency', is undefined for CTRL99 and UNC+, so after a fallback the 99-control and SEL+ comparisons are unspecified. The fallback is near-certain when P = H+ at the 200 cap: G-ESS then needs 10 x 20,100 = 201,000 effective samples from about 4.6e5 fit tokens, which fails for any tau_int >= 3.
  - Evidence: Field definitions as quoted at lines 753 and 763. The Schwab, Nemenman and Mehta setting the plan cites is a time-varying latent field, not a position-bin field. Fit tokens: half of L.R-A + L.R-B, about 917k tokens in all.
  - Fix: - Add chain-level fields and a token-level driver (e_t, s_t, and the leave-one-out population rate over the previous 16 tokens) to (iv) and to the kinetic fields. For held-out scoring, estimate the chain fields on the first half of each test chain.
- Plant P5's hidden field at chain and token level.
- Specify the G-ESS fallback for every population at the same N: P by selection frequency, SEL+ and UNC+ by weight order, CTRL99 by a frozen random order of 30 units.

- **validity-10** [medium] L1: data and label (lines 362-365), m_0 / F_small (412), verdicts 5-6 (420-421), falsification flag (425), pre-answer PIP (386); L.7 code sense (1264-1265); L.12 (1398)
  - Problem: The verdict tree and every code-sense flag use answer-token-mean CETT of the sample-0 answers. CI and CC rows have different answer strings (wrong vs correct entities, which differ in length, frequency and familiarity), so random fragments can read the label from lexical content alone.

F_small, 'distributed without a privileged set', 'dense' and 'not a strict threshold code (falsified)' are the code-sense outcomes that count against holographic and local storage. They would then measure where lexical information about the answer sits, not how the model codes hallucination, and the falsification of the holographic code reading would carry no information.

The pre-answer PIP (v), which avoids the answer-token construct (lf_holographic MI3), runs at only 3 sizes and enters no rule.
  - Evidence: Gao et al., as quoted at line 98, found random equal-size sets above chance on TriviaQA (56.1-68.4 %), which is consistent with generic or lexical information in answer-token features. L.12 (line 1398) lists the answer-token construct as a limit but attaches no rule to the pre-answer reads.
  - Fix: - Make the pre-answer (last-prompt-token) PIP primary for m_0, F_small, Syn, Mono and the falsification flag: the full U grid with floors at every size, about 1,800 extra I-hat estimates, about +4 A100-min of work. Keep the answer-token results as secondary.
- Alternatively, add a lexical floor: I-hat of a probe on answer-string features (token count, mean input embedding, the model's unconditional log-prob of the answer). Count only fragment information beyond it, I-hat(F u lex) - I-hat(lex).

- **validity-11** [medium] V-HOLO (lines 1174-1178), answer template (1194), L.0 holographic table (51), L2c (609-619), L.2 'positive must be earned' (265), L.7 (1264-1268)
  - Problem: No frozen rule says when a sense counts as 'holographic', so the template's '[list] sense(s)' would be filled after the data.

The reconstruction sense rests on two quantities:
- Lambda: output entropy predicts the label;
- L2c: a lag-0 cross-map of (s, e) onto a_P, ranked only against random channels, although L.2 itself says a lag-0 cross-map is structural.

Both are met by any H that tracks output uncertainty, which is what prior (a) expects, and L2c has no UNC+ or SEL+ comparison. The answer could therefore call Qwen 'holographic in the reconstruction sense' on evidence with no holographic content, which misrepresents the user's question.
  - Evidence: V-HOLO lists components per sense without outcomes, and L.7 lists only the 'against' outcomes. L2c's comparisons are rho_cross and CTRL99 only (lines 612-613).
  - Fix: Freeze a per-sense table:
- code: 'holographic-like' only for L1 verdict 4;
- RT: only 'logarithmic';
- reconstruction: L2c positive and rho_P - rho_UNC+ lower bound > 0 (add UNC+ and SEL+ to L2c); Lambda reported only as a redundancy diagnostic;
- associative memory: 'graceful degradation' and 'H not causally privileged' and (attractor line or X4 'associative completion').

Every other outcome reads 'not holographic-like' or 'undecided', and the template takes its list from this table.

- **validity-12** [medium] V-CHAIN: Loop row (line 1186) and chain rule (1192); L.0 chain table (62); L.7 (1275-1276)
  - Problem: The Loop link is positive by construction ('Token and turn loops exist by construction'), so it can never fail. Yet the user's chain starts from a loop transformer, Qwen's depth is not a loop, and X3 is tier 2. 'the chain holds in Qwen3-1.7B' would therefore claim the user's first premise without a test.

'the chain breaks at <first negative link>' is undefined when a link reads 'undecided', 'mixed', 'not testable' or 'not decidable'. Printing 'breaks at' for such a link calls a non-result a break, which L.7 says is no evidence either way.
  - Evidence: The Loop row and the rule quoted above. With the L4 problem of validity-1, the coupling links would also read 'undecided' or a false 'no coupling'.
  - Fix: - Loop link = X3's 'wrong fixed points exist' or 'toy H-units form an attractor' when X3 runs. Otherwise 'not tested in Qwen (no depth loop; token loop by construction)', and the chain string reads '... holds from the Agent link on'.
- Add 'the chain is undecided at <first link that is neither positive nor negative>'.
- Print 'breaks at' only for a link with a negative string.

- **validity-13** [medium] E-channel definition (line 230), 'not beyond uncertainty' (236), L2b label (579), L3a hazard (641), L5c hazard and comparison channels (1027-1029, 1036)
  - Problem: The E-channel comparison, on which the judges' 'uncertainty as a common driver' fix relies, is degenerate wherever the statistic already conditions on e:
- L2b: the E-channel's beta_pos^(iota,e) is identically 0 (e residualised on e), so P - E equals P's own slope. That contrast excludes 0 whenever 'H rises beyond information loss' passes, so 'length trend not beyond uncertainty' can never attach, even for an H that is purely an uncertainty channel.
- L3a: the E-channel's state is b = 1[e_t > threshold], so its exit at t is a step function of the covariate e_t. That is complete separation, and gamma is undefined.
- L5c: the E-channel's model contains z(e) twice (beta_H and beta_e), so the design is singular.

The qualifier needs both contrasts, so it is unreachable in L2b and undefined in L3a and L5c.
  - Evidence: Definitions as quoted. The label requires 'P minus UNC+ and P minus the E-channel both' to include 0 (line 236).
  - Fix: - Use the E-channel only for statistics that do not condition on e: the unpartialled slope beta_pos^0 in L2b; the hazard without the e covariate (or with e_(t-1)) in L3a; the model without beta_e in L5c. State this in each rule.
- Where the E contrast is undefined, base 'not beyond uncertainty' on UNC+ alone.

- **validity-14** [low] L4 C-doc (line 883), collaboration rule (913-916), V-CHAIN Collaboration (1189)
  - Problem: C-half and C-doc differ in four ways at once:
- the system prompt ('Agent {X} ... discussing ... with a colleague' against 'a researcher analysing');
- the message frame ('Your colleague says' against 'You find this note in the archive');
- the turn order (A speaks before any B message only in C-half);
- responsiveness.

J_ex(C-half) - J_ex,doc > 0 can therefore come from role-instructed deference to a 'colleague', which is instruction following, rather than from interaction. This is the distinction the lens asks about: collaboration against instruction following.
  - Evidence: Prompts as quoted at lines 873 and 883.
  - Fix: Add a frame-matched replay C-rep. A gets the C-half system prompt and turn order and receives 'Your colleague says: «...»' with B's messages replayed from a C-half run of another seed of the same item and configuration, so the partner does not respond to A.
- Collaboration (interaction) = J(C-half) - J(C-rep).
- Report J(C-rep) - J(C-doc) as the social-framing effect.

About 2 A100-min of work, like C-doc.

- **validity-15** [low] X1: rows (lines 1070-1071), Delta (1074), retention (1076), rule (1078), A3 (1082); X4 (1126-1129)
  - Problem: X1 defines Delta(t) on P's channel only, and the rule compares rho_P with 'mean rho_R' without saying which channel rho_R uses. If the random pushes are read on P's channel (the cross-read), D_early is about 0 and rho_R is undefined. That repeats the ill-defined ratio the judges flagged for lean's K_rand, and 'capture above a dose threshold' becomes unreachable.

X1's and X4's specificity contrasts also rest on only 2 random sets, so between-set variance cannot be estimated (the judges' 'too few control sets'; lf_energy_ising O3).

X1's capture adds route A3 to the core attractor line, so this matters beyond tier 2.
  - Evidence: Rows and definitions as quoted.
  - Fix: - Read each random set on its own channel, Delta_R(t) = a_R(pushed) - a_R(twin), as L3b does; keep the cross-read on P descriptive.
- Use at least 6 damage-matched sets per arm, with the two-level bootstrap: in X1 (+2,048 rows, about +6 A100-min) and in X4 (about +1.5 A100-min).

- **validity-16** [low] L2a own text (lines 521-526), deficit rule (536-539)
  - Problem: EOS is masked for 4,096 tokens of raw continuation, and p_EOS is logged but no rule uses it. The model can still signal an end of text in raw mode. Forcing it past that point can start new or degenerate text, which lowers long-range captured MI and manufactures R(L) < 1. This is the masking problem the judges flagged, now confined to the own-text deficit.
  - Evidence: Line 524: 'EOS masked and p_EOS logged'. Nothing downstream conditions on p_EOS.
  - Fix: Freeze a rule:
- Truncate each self text where the cumulative end probability 1 - prod(1 - p_EOS) first exceeds 0.5, and compute R(L) only for L within that length.
- Report 'deficit' only if it also holds on texts whose max p_EOS stays below 0.05.
