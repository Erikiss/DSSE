> **DRAFT (13.11.2026, before the adversarial review) - not frozen.** Assembled from four independently designed and merged parts; the review is running and will change this text. Nothing has been run.

# Part L (round 1b): lens-free, inverse-free arm with holographic, energy / Ising and non-equilibrium measures (merged spec)

**Sources.** Merged from three independent designs (lean, energy_ising, holographic) and two judge reports.
- Base: lean. Both judges rank lean first and recommend it as the base, so there was no disagreement to settle. J1 put lean and holographic nearly level (61 vs 60) and decided by the validity of the attractor-versus-length core.
- Grafts, fixes and decisions are listed in L.13 and in the provenance field.
- Nothing has been run and no model was loaded.

**Planning simulations** (CPU, numpy/scikit-learn, no model).
- For this merge, in /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/scratch_L_merge/:
  - sim_duet.py, sim_duet2.py: the agent-coupling null under shared item fields;
  - sim_combined.py, sim_combined2.py, sim_combined3.py, sim_own2.py: the power of the combined verdicts.
- The designers' simulations are in scratch_L_lean/, scratch_L_holographic/ and scratch_L_energy_ising/. The judges' are in judge_L1/ and judge2_L/.

**Budget** (A100, fp32, TF32 off; planned = setup + 1.5 x work, round-2 convention).

- Tier 1 (core, main §4.3): r1b_L_A 168.9 + r1b_L_B 117.2 = 286.1 A100-min (4.77 h), plus 10.6 min run in part K's sessions for L (S_L screening, 22 extra wrong items) = 4.95 h charged.
- Tier 2 (X1-X6): 55.4 min (0.92 h).
- The pilot L0 is in r1b_P0. L-ANA (a step of r1b_ana, CPU runtime, account A) needs 0 A100-min.

The core answers the user's question on its own. Every extension deepens one link and can be dropped alone (trim order in L.8). The merged core as first planned (5.08 h, its own pilot included) was about 0.6 h above the judges' estimate of 4.5-4.8 h. The difference buys the fixes both judges listed as missing everywhere:
- a throughput pilot;
- an uncertainty-selected control and a hull-matched selection control;
- 6-8 damage-matched random sets per causal arm with set-level resampling;
- a multi-turn carry;
- an exact-label step-error test;
- a non-social agent control;
- the realistic re-prefill cost of the agent engine.

## L.0 Question, answer in brief, scope

**User's question (German, verbatim).** 'Könnte man die erhofften Erkenntnisse bezüglich H-Neuronen Chain Lense (vgl. EBM reasoner, non-equilibrium - da Loop Transformer --> Agent --> Ising modellierbar --> Kollaboration --> entitätenbildend ....) auch durch einen linsenfreien Ansatz erweitern, indem man auf holographische Prinzipien setzt?'

Translation: could the hoped-for insights on H-neurons and the chain lens (cf. EBM reasoners, non-equilibrium; since loop transformer -> agent -> Ising-modellable -> collaboration -> entity-forming ...) be extended by a lens-free approach that relies on holographic principles?

**Answer in brief (what L will be able to say).**

1. **Lens-free: yes, and it is needed.** The chain-lens post (vgel, Raymond Douglas, LessWrong, 11 Nov 2026) says:
   - 'The chain lens confabulates more on WorkspaceBench's hallucination family than single-token lenses. This grows with chain length' (0.45 vs 0.32 for the J-lens, lower is better);
   - much longer chains 'trade more hallucination under the current summarizer'.

   Qwen3-1.7B has no MTP head, so part K runs a substitute chain lens. A length effect seen only through a lens cannot be attributed to H-neurons. L decides without running any lens:
   - L5 asks whether the break-off and its H-pattern belong to the model or to the lens, on items with exact labels;
   - L2 and L3 test the H-neuron attractor against the length artefact in the model's own continuations.

2. **Holographic: only the classical, information-theoretic content carries over.** It is measured in four senses, each reported separately and never merged into one word (V-HOLO, L.5):

| Holographic sense (primary sources as verified in lf_holographic.md) | Classical counterpart in Qwen3-1.7B | ID |
|---|---|---|
| Code: threshold recovery, no privileged region, monogamy I3 <= 0 (Pastawski, Yoshida, Harlow, Preskill 2015; Hayden, Headrick, Maloney 2013) | Partial-information plot (PIP) of the hallucination label over random neuron fragments, with a permutation floor at every size; knock-out reselection; excluding-H fragments; bias-corrected I3; planted-code calibration | L1 |
| RT / area law: information through a cut is bounded by the cut; growth class on equal partitions (Ryu, Takayanagi 2006; L2M 2025) | Growth class of the model's captured MI between equal halves of natural text up to 32,768 tokens; deficit of the model's own text | L2a |
| Bulk from boundary (reconstruction) | Output-side scalars (entropy, surprisal) recover the internal H channel: statically (Lambda) and dynamically (cross-map), ranked against 99 control populations | L1(iv), L2c |
| Holographic (associative) memory: distributed traces plus clean-up into attractors (Plate 1995; Hopfield) | Causal erasure with the model as decoder (graceful against threshold; H privileged or not); attractor dynamics; completion from a partial cue | L1(vii), L3, X4 |

   No counterpart (L.6): AdS bulk, gravity, quantum entanglement, no-cloning, conformal boundary, Page curve and scrambling.

3. **The user's chain, link by link.**

| Link | Measured as | ID |
|---|---|---|
| EBM reasoner | Energy E = -log q of the model's own text; energy-well signature at H entry | L3d |
| Non-equilibrium | Path irreversibility at H entry against a detailed-balance null | L3d |
| Loop | Token loop (L2, L3a), turn loop (L3b), dialogue loop (L4). Qwen's depth is not a loop; a weight-tied toy loop stands in | X3 |
| Agent | A Qwen instance with its own context, acting in turns | L4 (format gate) |
| Ising-modellable | Units: static max-ent and kinetic fits of the H population against 99 controls and selection/uncertainty-matched sets | L3c |
| Ising-modellable | Agents: kinetic Ising fit of stance spins | L4 |
| Collaboration | Coupling beyond lexical priming (live partner against the same messages as documents) | L4 |
| Entity-forming | Consensus under opposite priming beyond within-item re-pairing | L4 |

   The chain holds only if every link is positive; otherwise L names the first broken link.

**What L adds to part K.**
- An ownership check that cannot inherit lens confabulation (L5).
- An information-based denominator for 'length' (anchor information iota, L2b).
- Population and dynamics tests of the H-set: stickiness against item-specific surrogates, within-item bistability, twin push-and-return under the round-2 R6 convention, and Ising fits (L3).
- An exact-label test of whether H rises before the model's own errors (L5c).

**Lens-free and inverse-free, as applied (binding for every L measurement).**
- **Lens-free.** No learned or transported readout of an intermediate state into the vocabulary: no logit, tuned, J, J++, MicroLens++ or chain lens, no MTP head, no Patchscope, no state patched into another context.
- **Allowed:**
  - the model's own next-token distribution on its own input (log-probabilities, entropy, forced-choice log-odds);
  - MLP activations (raw z, CETT) and fixed directions;
  - sampled text;
  - probes that map activations to behavioural labels, not to the vocabulary.
- L consumes K's chain-lens records only as data to be classified (L5).
- **Inverse-free.** Allowed:
  - forward passes and interventions;
  - forward regressions (logistic, ridge, pseudo-likelihood), fitted by L-BFGS or Adam;
  - forward MCMC and zero-temperature descent;
  - power iteration (modularity);
  - nearest-neighbour cross-mapping (simplex, no S-map);
  - Frisch-Waugh residual slopes (cov/var);
  - 1-D grid searches; counts.
- **Never:** matrix inversion, pseudo-inverses, inverse Jacobians, Tikhonov inverses, J = -C^-1, TAP, graphical lasso.
- Each regression that predicts one quantity from another is a fitted forward map, which the design rules allow, and is named as such. The IAAFT inverse FFT, Phi^-1 on ranks and R^T for an orthogonal R generate null data or transform marginals; they reconstruct no causes.
- **The core makes no reconstruction.** L names no content. What H encodes is K's generate-verify loop (SO-GVRA). The only reconstruction in L is X5 (L.gv), an Aletheia-style generate-verify-revise-abstain loop with the verifier's error rate measured on known answers.

**Honest priors, stated before any data.**
- (a) A local store is unlikely. In six non-Qwen models, random equal-size sets classify above chance (TriviaQA 56.1-68.4 % against 70.1-83.6 % for H-neurons; Gao et al. 2025, Table 1). The selection is non-unique (Cavus et al. 2026: 19/22 neurons with a partner at |r| > 0.7; bootstrap Jaccard 0.45-0.69). A strict threshold code is unlikely for the same reason. Expected outcome: 'H privileged but one of several representations' or 'distributed without a privileged set', with the flag 'not a strict threshold code (falsified)'.
- (b) The length trend is unknown. I found no lens-free study of H activity along long generations (lit_chain; an absence claim).
- (c) Round 2 found every carry below 1: K 0.449-0.482 along the shared style direction (AUSWERTUNG Teil B). An H attractor would be the first self-sustaining state found in this model.
- Further expectations: power-law growth for natural text (L2M) and a deficit for the model's own text (Lin and Tegmark); conformity coupling between agents.
- An outcome against these expectations is what would support the user's stronger readings.

**Core and extensions.**
- Core: L0, L-CAL, L1, L2, L3, L4, L5, L-ANA.
- Extensions (round tier 2, main §0.3; main §4.5's drop items 2-5, 11 and 12 keep L's order X5, X4, X3, X6, X1, X2):
  - X1 dose ladder and hysteresis;
  - X2 H-down agent arms;
  - X3 weight-tied toy loop (depth loop);
  - X4 associative completion;
  - X5 L.gv basin content;
  - X6 solo-agent loop baseline.

## L.1 Shared objects: what L consumes (contract) and what it provides

L redefines none of K's, D's or R's objects. An object missing at session start triggers the stated fallback, never a silent change.

**From part K (contract; fixed before the freeze):**

- **SO-HN.H_op (H_tq, or H+ after K's G-K22 switch; main OD-M10)**: the frozen H-set H (positive weights at C*, prose label convention), with per-neuron probe weight, bootstrap selection frequency, layer and index.
- **SO-HN.H+ (L caps it at 200 by selection frequency, then |r|, whenever P is an H+ set)**: H plus every neuron with |r| >= 0.7 to a member on S_fit CETT.
- **P (the push/read set)**: P = H_op if H_op is H_tq with |H_tq| >= 10, else H+ (capped at 200). If H_op = H_ar, SG8 fails for L ('no validated H-set (H_ar primary)') and L1 uses K's selector on S_fit. Statements about H report H_tq too whenever P is an H+ set.
- **SO-HN boot sets (K2 Dg2's 20 question-bootstrap refits)**: the positive-weight sets of K's 20 bootstrap refits. If fewer than 10 are delivered, the selection-robustness label reads 'not computed'.
- **SO-LAB**, per TriviaQA item:
  - text, gold aliases, the 10 sampled answers with keys, alias-match labels;
  - consistency class (CC = 10/10 correct; CI = 0/10 correct and no refusal);
  - split: S_fit for selection; S_L with at least 600 CC + 600 CI items (main OD-M8), used neither for selection nor for choosing C;
  - K's answer-span rule, refusal parser and resolved label-sign convention;
  - K's 300 NE items (FAB).

  Fallback if S_L is smaller: L uses all of it, and every L1 MDE scales by sqrt(1,200 / n).
- **SO-HN hook manager**: the down_proj pre-hook manager. It captures z = SiLU(W_gate x) * (W_up x) and CETT_{j,t} = |z_{j,t}| ||W_down[:, j]|| / ||W_down z_t||, scales selected neurons by alpha under a per-position mask, and mean-clamps them.
- **SO-HN selector**: K's frozen L1-logistic pipeline as a callable (GPU FISTA at C*, positive weights, exclusion mask). L never re-tunes C.
- **SO-HN random-set generator with K2's rule (same count per layer, mean answer-CETT within +-20 %) and L's candidate seeds keyseed('r1bL-CTRL', k)**: K's generator of random sets matched to a target set, called with L's own candidate seeds so that L's sets are independent of K's RS sets (main OD-M12).
- **SO-CHH: a_X(t) = mean zCETT over X (K's merged definition) is primary; L's standardised-z mean is the sensitivity variant; both are stored**: the per-token H channel a_t exactly as K defines it; L's sensitivity variant is the mean over X of (z_j - mu_j) / sigma_j. Every comparison population uses the same definition on its own units.
- **SO-HN REF-W**: per-neuron reference mean and SD of z on K's neutral reference corpus. Fallback: L computes them on 200 WikiText-103 test-split passages (main SO-WT103) x 512 tokens (+0.3 min).
- **SO-LAB's gated A-lens pool items (K4): known = K4 'correct', prone = K4 'wrong'; read position t* = K4's t0**: the programmatic arithmetic chain items, with:
  - prompts, truth sets V_i (operands, intermediates, result);
  - strata (known / prone) and K4's 5 parsed gating answers per item (greedy and 4 samples at T 0.7; L adds 6, 11 in all, L5a);
  - read position t* (last prompt token).

  At least 150 known + 150 prone items.
- **SO-LENSOUT: B_hal -> k*_conf; FOV_H(q) with K's random-set band -> share_H(s) with the RS39 band (q95 of share_R); A_H(q) -> a_H(s); steps s = 0..16**, one record per chain:
  - item id, stratum, family (two- or three-step), lens variant, read block l, seed rank, key;
  - emitted token ids at every step s = 0..16 (L's chain position q = s);
  - k*_conf (K's first CONFAB);
  - share_H(s) with the RS39 band (q95 of share_R);
  - a_H(s), the lens-frame H activity.

  Family-level summaries alone do not satisfy the contract (G-CL).
- **SO-GVRA**: K's generate-verify-revise-abstain convention, which X5 follows.

**From part D:**

- **SO-LCE (main §2.3; owner D)**, D's long-context engine: B = 1 prefill, chunked lm_head at the needed positions, explicit positions, the memory-efficient kernel asserted (SG9). Parity for L2a is gated by L's own G-L2a-LCE (< 1e-4 relative; main OD-M14).
- **L.LONGDOC (main SO-LONGDOC; owner L; primary)**: PG19 test split at a pinned revision; the first 24 books in file order with at least 36,000 tokens after skipping 2,000; slice [2,000, 34,768).
- **SO-D1STIM F-sym items 0-31 (invented referent i, attribute slot j = i mod 6, v1 / v2 = the slot's versions 1 / 2)**: D's fictional-entity factual-fork items (entity, contested attribute, values v1 and v2).

**From part R:**

- **SO-DMG: calibrate rule (f <- f sqrt(target / damage), +-10 %, <= 6 evaluations); damage on REF-U32 = REF-R16 union REF-K16 (REF-R16 from r1b_xfer_A/dmg_ref_R16.json) plus 64 held-out S_L answers**: SO-DMG's function-style damage and calibrate (r1b_core), with RefKL in the 'hooks' variant; damage is teacher-forced KL in nats. L adds the 64 held-out S_L answers because neuron sets span many layers.
- **SO-AX's u*_14 (U*, from r1b_xfer_A/so_ax_ustar.pt)**: the round-2 style direction u*_14. Descriptive covariate only; if it is missing, the line is dropped.

**Round-2 machinery reused unchanged** (imported; rk_loop.py byte-identical; new code in a new function-style module):
- rk_loop keyseed and Gumbel (as SO-KEYS's verified copy in r1b_common, main §2.3);
- r2_theta.stage_turn2 (TURN2_SUFFIX, ids by concatenation, history think block kept);
- r2_theta_verdict.rule_R6 logic;
- r2_rk.prompt_bootstrap;
- the r2_rk.CbarHook pattern (generation-time capture; exact for unsteered runs);
- r2_unlearn_qwen.calibrate and RefKL;
- r2_linearity.auc_cols;
- the neu_extract / neu_sparsity down_proj hook patterns (code only).

**L provides, for the merge:**

| Object | Content |
|---|---|
| L.POP | Population index file: P, SEL+, UNC+, CTRL99, the union of the SO-HN boot sets, hull memberships; written in step 1 of r1b_L_A |
| L.R | Chain corpus: L.R-A and L.R-B, with per-token channels, spins, entropy, surprisal, anchor information and repetition flags |
| L.PIP | Fragment sweeps, permutation floors at every size, reselection, planted codes |
| L.EQMI | Equal-partition captured-MI estimator with G-MI controls |
| L.TWIN | Twin push-and-return driver with multi-turn carry and R1 / R2 / R3 reads |
| L.MARKOVSUR | Item-specific, position-matched first-order Markov surrogate, plus its detailed-balance version |
| L.KISING | Static pseudo-likelihood max-ent and kinetic Ising fitter for units and agents; Gibbs check; basins; modularity |
| L.DUET | Two-agent driver (B-Q8 id concatenation, re-prefill, context-free message reads) |
| L.OWN | Readout classifier for K's chain records with a chance-match null |
| L.SEL+ / L.UNC+ | Selection-matched and uncertainty-selected control populations, hull-matched |
| l_verdicts.json | All L verdicts |

## L.2 Conventions (all measurements)

**Model and numerics.**
- Qwen3-1.7B, pinned revision, chat template, thinking off (empty think block as in rounds 1/2).
- fp32 weights, cuBLAS TF32 off, set_float32_matmul_precision('highest').
- Attention runs through the memory-efficient SDPA kernel, asserted; a math fallback raises. In fp32 on A100 this kernel computes in 3xTF32 'FastF32' whatever the TF32 flag says (int_machinery 3.3).
- Every contrast is computed on one kernel path, one batch shape and one GPU model. Paired arms are never split across GPUs: the per-token divergence hazard between an NVIDIA L4 and an A100 is about 1e-3 (int_machinery 3.8).
- GPU model and torch / transformers / CUDA versions are logged.
- Contexts stay at or below 32,768 positions (native window); no YaRN.

**Sampling and keys.**
- Gumbel-max over the full vocabulary (rk_loop), no top-k / top-p, at the stated T.
- New key namespaces: 'r1bL0', 'r1bL2', 'r1bL2b', 'r1bL2s', 'r1bL3', 'r1bL3t', 'r1bL4', 'r1bL5', 'r1bLX'. Registered prefix 'r1bL' (main §2.3 SO-KEYS). No round-1/2 key is reused.
- The 6 extra gating samples of L5 use keys ('r1bL5', 'GATE', i, s), s = 1..6.
- Keys are shared within a twin set and across agent conditions (common random numbers).
- Seeds: 'r1bL1rot', 'r1bL-ERA', 'r1bL2perm', 'r1bL-CTRL', 'r1bL5-FM', 'r1bL-BOOT', 'r1bL-SUR', 'r1bL-SHUF' (L2a).

**Per-token channels.** Captured during unsteered generation (exact) or by teacher forcing:
- a_t (SO-CHH) and the same channel for every comparison population;
- e_t = entropy of softmax(logits_t) at T 1;
- s_t = -log softmax(logits_t)[x_(t+1)] at T 1 (the energy E_t; computed at T 1 whatever the sampling T);
- p_EOS(t);
- token id;
- 4-gram repeat flag (rate > 0.3 in the trailing 256 tokens).

**Position bins** (continuation index k): [1-16], [17-32], [33-64], [65-128], [129-256], [257-512], [513-768], [769-1024].

**H-state.**
- b_t = 1[a_t > Q95(F-neu, T, bin(t))], with Q95 fitted on odd F-neu chains. Even F-neu chains check that the share is 5 % (+-1 %); otherwise the line carries 'binarisation drift'.
- Comparison populations are binarised at the per-(family, T, bin) quantile that gives P's occupancy (occupancy-matched), because logit-scale statistics depend on the base rate.

**Spins** (Ising): s_{j,t} = 1[|z_{j,t}| > Q95_j(F-neu, T, bin)] on raw z, not CETT. CETT's shared per-layer denominator couples the units of a layer.

**Comparison populations.**

| Population | Definition |
|---|---|
| CTRL99 | 99 random sets drawn by SO-HN's random-set generator with K2's matching rule (same count per layer, mean answer-CETT within +-20 %) from candidates keyseed('r1bL-CTRL', k), disjoint from P's hull, SEL+ and UNC+; frozen order |
| SEL+ | Selection-matched control: the SO-HN selector on S_fit for the first label of a frozen list that is unrelated to hallucination: (1) question starts with 'Who'; (2) question contains a digit; (3) a gold alias contains a digit. The label must have |phi(label, hallucinated)| < 0.1 and at least 300 items per class, balanced by subsampling (G-sel). Top |H| units by weight; if fewer units have positive weight, filled by |weight| of an L2-logistic refit. When P is an H+ set, SEL+ gets the same hull: plus |r| >= 0.7 partners on S_fit CETT, capped at |P|, ordered as H+ |
| UNC+ | Uncertainty-selected control: the SO-HN selector on S_fit for y_unc = 1[mean next-token entropy over the answer span of sample 0 > the median within the row's class (CC or CI)]. phi(y_unc, hallucinated) = 0 by construction. Same size and hull rule as SEL+ |
| E-channel | The output entropy e_t, binarised occupancy-matched like a population |
| SO-HN boot sets | The 20 bootstrap H-sets; used only for the selection-robustness label |

Rank tests use the 99 CTRL99 sets: p = (1 + #{c >= statistic of P}) / 100, so the minimum is 0.01. SEL+ and UNC+ enter through paired contrasts with intervals.

**Labels and qualifiers.**
- 'not beyond uncertainty': for the statistic that passed, P minus UNC+ and P minus the E-channel both have intervals that include 0 or lie below.
- 'selection-robust': a passing H-specific criterion also holds at the point estimate in at least 16 of the 20 SO-HN boot sets. Otherwise 'selection-fragile'.

**Intervals and tests.**
- 90 % percentile bootstrap with 2,000 replicates (Qwen convention). Units:
  - items for L1 and L5, with all chains, blocks and seeds of an item together;
  - chains clustered by item for L2 and L3a;
  - prompts with both seeds together for L3b;
  - documents for L2a;
  - item-clustered runs for L4.
- Causal contrasts against several random sets use a two-level bootstrap (units, then random sets with replacement), so that between-set variance enters the interval.
- Holm within each measurement's verdict family on one-sided p-values (bootstrap share on the null side, computed as (1 + #) / 2,001 by main §2.4's SO-PVAL; rank p for rank tests). No correction across measurements (different questions), as in round 2. Conjunctions are not corrected.

**'No effect' verdicts.**
- Need the whole 90 % interval inside a pre-registered equivalence margin. The margin is at least the planning MDE (2.49 planning SE) and set near 2.9 SE, so that the equivalence verdict has planning power of about 0.8 at a true zero.
- 'Interval includes 0' never suffices.
- Under-power rule: if the realised bootstrap SE exceeds 1.5 x the planning SE, the equivalence verdict becomes 'not decidable (underpowered)'.

**Reads of activity.**
- (R1) Round-2 convention: the text is read by the unmodified model as the reply to the plain prompt, mean over continuation tokens.
- (R2) Live, captured during generation.
- (R3) Cache-free in-context re-read: prompt plus text, teacher-forced by the unmodified model.

**Damage.** SO-DMG: teacher-forced KL in nats on REF-U32 (REF-R16 union REF-K16) plus 64 held-out S_L answers. A push above 0.15 nats carries 'high-damage push' (GO3 analogue).

**'Positive must be earned.'** None of the following ever counts as support on its own:
- HD > 0 (every population is autocorrelated);
- occupancy rising with position;
- Psi > 0 (independent persistent parts give Psi > 0);
- a lag-0 cross-map (structural within a front);
- a fitted-energy basin (minima of a fitted energy are not network attractors).

**Hygiene.** logits_to_keep or lm_head on the needed positions in every teacher-forced call; GPU-side slicing before any .double().cpu(); steps of at most 5 min in resumable parts/; arrays above 0.5 GB stay in local/; every result file carries code, config and prompt hashes.

## L.3 Measurements (core)

### L0 PILOT: throughput and memory pilot (core; engineering only, before the freeze)

- **What.** Measures the throughput and memory figures every L cost and batch size rests on. It produces no data used by any L analysis.
- **How.** P0-L items of the round's single pilot r1b_P0 (account B): SO-PILOT's 32 pilot prompts and PG19 books 25-28 (disjoint from every analysis set); every text is discarded and declared excluded from all analyses. Stand-in random sets of the planned sizes (no H-set exists before the freeze). Measured:
  - (i) hooked decode ms/step at batch 32 and contexts 0.5k and 1.1k, with the full L capture set (a_t of 1 + 99 + 2 populations, the union of the SO-HN boot sets, spins of up to 200 units per population, e_t, s_t) against unhooked decode;
  - (ii) R1 / R3 read rates (tok/s) at batch 32 x 130 tokens with hooks;
  - (iii) B = 1 prefill time and peak memory at 16k and 32k (measured once in r1b_P0 for parts D and L, with and without down_proj hooks);
  - (iv) the duet engine with 32 runs x 2 agents:
    - re-prefill ms per message;
    - a carried-cache variant (B-Q8 id concatenation makes contexts append-only): parity against re-prefill on 4 runs (max relative logit difference < 1e-4, Gumbel flips counted);
  - (v) RefKL 'hooks' damage time for a 60-unit set;
  - (vi) Drive read/write throughput on 100 MB;
  - (vii) the relative log-prob difference of L2a's chunked path against a full pass at 16,384 and 32,768 tokens (main OD-M14).
- **Rule.** Descriptive. SG7 (L's part): if any rate is below half its planning value, the trim order (L.8) is applied before the freeze. The L cost table is frozen on the measured rates. If (iv)'s carried-cache variant passes parity, L4 may use it (pre-registered saving of about 40 % of L4's generation time; no rule changes).
- **Null / controls.** None.
- **Shows - cannot show.** Engineering numbers only.
- **MDE.** Not applicable.
- **Inputs.** SO-HN random-set generator (or seed 'r1bL-CTRL'); pinned model.
- **Runs in.** r1b_P0 (10 A100-min of its work).
- **Gates.** None; it defines SG7 (L's part).

### L-CAL: known-answer calibration of every L statistic (core; CPU before the freeze; repeated as code tests)

- **What.** Measures each pipeline's false-positive rate and power on synthetic data with known answers at the planned sizes. This is the design rules' 'verifier error rate on known answers' applied to L's own statistics. A failed item turns the dependent verdicts into 'not testable (pipeline calibration failed)'.
- **How.** Frozen generators and seeds:
  - **(a) PIP tree** (lean): background N = 4,096, n = 1,200, 40 latent drivers. Profiles: localist (20 units), redundant (2,000 noisy copies), dense, dense-threshold (threshold at 32 units) and XOR of 3 groups.
  - **(b) Stickiness** (L.MARKOVSUR plus the item-FE hazard, surrogate-corrected Delta gamma): homogeneous and heterogeneous memoryless chains (frailty) and sticky semi-Markov chains (gamma 0.05-0.2); 600 x 1,024, occupancy 5-10 %.
  - **(c) HD rank test**: 100 heterogeneous memoryless populations; P(rank 1) must be 0.01 +- 0.01 over 400 reps.
  - **(d) Within-item bistability BI**: 96 items x 8 seeds; memoryless heterogeneous items against items with 15 % planted two-state switching.
  - **(e) Static and kinetic Ising**: N = 30 and 60, sparse asymmetric J, position-modulated fields, matched autocorrelation; an independent set.
  - **(f) EQMI saturation and growth classifier**: an order-8 Markov source over 512 symbols (saturating, known L_sat) and a Lin-Tegmark hierarchical source (power law).
  - **(g) Length-against-information slopes** (L2b): synthetic chains where H depends on iota only, and chains with an added position term.
  - **(h) Irreversibility d-hat**: reversible against irreversible (cycle-flux) 4-symbol Markov chains with entry events and position trends, against the detailed-balance surrogate band.
  - **(i) RQA trapping-time difference-in-differences** with planted post-entry trapping and position trends.
  - **(j) Cross-map** (L2c):
    - x drives y at delays 0 and 2, 10 % observation noise, 16-level quantisation;
    - a common driver with position trend and no x-y coupling.
  - **(k) Agent coupling statistic J_ex** (L4) on planted kinetic data with item fields of SD 0.5 and 1.0.
  - **(l) Pair-macro Delta Psi**:
    - independent persistent AR(1) agents (phi 0.8);
    - conformity coupling c = 0.15 (holographic sim_psi_sum.py: mean -0.138, SD 0.030).
  - **(m) Ownership classifier** (L5a): synthetic chain records with planted FOREIGN growth on known items, MODEL growth on prone items, and chance matches.
  - **(n) Step-error hazard with item fixed effects** (L5c): synthetic CoT steps with planted beta_H of 0 and 0.3.
- **Rule (pass criteria).**

| Item | Pass criterion |
|---|---|
| (a) | Every profile except XOR classified in at least 4/5 seeds; XOR 'unclassified' (documents parity codes as unlearnable at n ~ 1,200) |
| (b) | Size <= 0.10 at nominal 0.05 in both memoryless cases; power >= 0.8 at gamma 0.05 (homogeneous) and 0.1 (heterogeneous) |
| (c) | As stated in How |
| (d) | Size <= 0.02 against the surrogate's 99th percentile; power >= 0.8 |
| (e) | Correlation of recovered with true J >= 0.8; the independent set inside the control band in at least 4/5 seeds |
| (f) | Correct class and L_sat within one octave in at least 4/5 seeds |
| (g) | Correct verdict in at least 4/5 seeds; size of 'H rises beyond information loss' <= 0.05 |
| (h), (i) | Size <= 0.10; power >= 0.8 at the planted effect |
| (j) | Convergent in at least 90 % of (j)(1) realisations; 'encodes beyond the cross-trajectory null' in at most 10 % of (j)(2) |
| (k) | Size <= 0.10; power >= 0.8 at J_c = 0.2 |
| (l) | Delta Psi inside +-0.025 nats in at least 90 % of 200 reps for independent agents; Delta Psi < 0 detected in at least 80 % for conformity |
| (m) | Correct verdict in at least 4/5 seeds; the chance-match correction removes planted chance matches within +-0.02 |
| (n) | Size <= 0.10; power >= 0.8 |

- **Planning simulations already run** (they fix MDEs below):
  - sim_pip2.py: profiles as in lean.
  - sim_sticky.py: size 0/15; surrogate SD of gamma-hat 0.012-0.016; MDE about 0.04.
  - sim_sticky_het.py:
    - the pooled surrogate gave false positives in 6/6 runs (gamma-hat 0.46 at gamma = 0);
    - the item-stratified test gave 0/6, and 6/6 power at gamma 0.1 with a shared downward bias, hence surrogate-corrected Delta gamma.
  - sim_duet2.py (k):
    - J_ex mean -0.006 / +0.004 at J_c = 0 (item-field SD 0.5 / 1.0), SD 0.040, bootstrap SE 0.036-0.043;
    - J_ex 0.14 at J_c = 0.2.
  - sim_duet.py: item fixed effects alone are not enough (mean J_c-hat +0.046 at J_c = 0 with item-field SD 1.0; false-positive rate 0.44 against across-item re-pairing).
- **Null / controls.** The synthetic nulls are the controls.
- **Shows - cannot show.** Shows that each estimator returns known answers at L's sizes and noise. Cannot show that Qwen's structure resembles the planted one.
- **MDE.** As in the pass criteria.
- **Inputs.** None.
- **Runs in.** CPU before the freeze, about 120 CPU-min, 0 A100-min.
- **Gates.** None; it defines G-cal-L1 (a), G-cal-L2 (f, g, j), G-cal-L3 (b, c, d, e, h, i), G-cal-L4 (k, l) and G-cal-L5 (m, n). The real-matrix L1-C and the matched planted Ising sets (G-cal-Ising) complete G-cal-L1 and G-cal-L3 in session.

### L1 LOCUS: where is the hallucination signal stored? (core; code and associative-memory senses)

- **What.** Where does the information about K's independent hallucination label sit among the 172,032 MLP neurons? Candidates:
  - localised (P is the store);
  - one of several disjoint representations;
  - distributed or dense;
  - holographic-like (no privileged set, no information below a threshold fragment size, synergy and monogamy at the threshold).

  And does the model's own behaviour degrade at a threshold or gracefully under random erasure, with P causally privileged or not?
- **How.**
  - **Data.**
    - In-session recompute: SO-HN REF-W means mu_j for all neurons, if REF-W lacks them (200 x 512 neutral tokens).
    - Answer-token-mean and last-prompt-token CETT for all 172,032 neurons, by teacher forcing K's sample-0 answers of the S_fit rows, the S_L rows (at least 600 CC + 600 CI) and the 300 NE (FAB) rows. This replaces a 1.1+ GB copy-in. The same pass gives each row's answer-span mean entropy, which UNC+ needs.
    - Then SEL+, UNC+, CTRL99 and the hulls are fixed and written to L.POP.
  - **Label.** y = 1 hallucinated (CI), 0 faithful (CC), balanced.
  - **Information estimate I-hat(F)** (a fitted forward map): an L2 logistic probe on z-scored features (scaling fitted on the training fold); 5 folds by question; C by inner 3-fold CV from {1e-3, 1e-2, 1e-1, 1}. I-hat(F) = 1 bit - mean held-out log-loss in bits. It is a lower bound on I(y; F) and may be negative.
  - **(i) PIP.** Sizes m in {1, 2, 4, ..., 1,024} plus {|P|}. Fragment families:
    - U: uniform over all neurons;
    - M: the 99 CTRL99 sets at m = |P|;
    - Rot: rotated basis. Per layer a random orthogonal R_l (QR of a Gaussian 6,144 x 6,144 matrix, seed 'r1bL1rot'); z' = R_l z; CETT'_k = |z'_k| ||(W_down R_l^T)[:, k]|| / ||FFN_out||, from one re-read of the S_L rows. R^T is a transpose.
    - XH: uniform excluding P's hull (P plus every unit with |r| > 0.7 to a member).

    50 fragments per m for U, Rot and XH; the statistic is the median I-hat.
    - Permutation floor F(m) at every grid size: the q95 of I-hat over 100 label permutations, each with a fresh U fragment of size m. This fixes lean's floor at three sizes only.
    - Reference I_ref = the maximum over m <= 1,024 of the median U-PIP.
    - m_delta = the smallest m whose median I-hat reaches delta I_ref (log-interpolated).
    - XH ratio = m_0.5(XH) / m_0.5(U). Censored at 1,024 / m_0.5(U) if XH never reaches 0.5 I_ref within the grid.
  - **(ii) Knock-out reselection** (lean):
    - H_1 = P;
    - for r = 2..6, H_r = the SO-HN selector on S_fit with H_1..H_(r-1) and their |r| > 0.7 clusters excluded, capped at |P|;
    - drop ratio rho_2 = I-hat(H_2) / I-hat(H_1);
    - R_rep = 1 + #{r : I-hat(H_r) >= 0.8 I-hat(H_1)}, a lower bound.
  - **(iii) Synergy and monogamy.** Disjoint U pairs (A, B), 50 per m in {8, 16, ..., 512}; I3(m) = I-hat(A) + I-hat(B) - I-hat(A u B).
    - Onset synergy (lean): Sigma(m) = -I3(m), against Sigma_dense(m) of the planted dense code at matched I_ref.
    - Monogamy (holographic E3): I3_c(m) = I3(m) - I3_localist(m). The planted localist code has true I3 = 0, so it calibrates the estimator bias.
  - **(iv) Latent scalars** (bulk from boundary, static). Lambda = I-hat(scalar) / median I-hat_U(1,024) for two scalars: the mean answer-token entropy of the model's own next-token distribution (lens-free output side), and the first PC of S_L CETT (fit on S_fit).
  - **(v) Pre-answer PIP.** The U sweep and its floors at 3 sizes on last-prompt-token CETT. This avoids the answer-token construct (lf_holographic MI3).
  - **(vi) L1-C planted-code calibration on the real matrix.**
    - Synthetic labels planted in the real S_fit u S_L CETT matrix: localist (|P| random units), redundant (2,000 noisy copies), dense, dense-threshold (thresholds 16 and 64) and XOR of 3 groups.
    - SNR by a frozen 6-step bisection on a cheap I_ref proxy (5 fragments at the 3 largest sizes), matched to the real I_ref within +-20 %.
    - 3 seeds per profile.
    - The PIP at 11 sizes x 20 fragments, with floors at 3 sizes x 50 permutations.
    - Reselection only for the localist and redundant codes, where it matters.
    - The SO-HN selector plays H_1's role.
  - **(vii) Causal erasure with the model as decoder** (holographic L5b).
    - Mean ablation z_j <- mu_j at all positions of:
      - nested random fragments f in {2^-12, 2^-10, 2^-8, 2^-6, 2^-4} of all neurons (42, 168, 672, 2,688, 10,752; 3 nested draws, seed 'r1bL-ERA');
      - P, and H_tq / H+ if they differ from P;
      - 8 random CTRL99 sets (13-20 in frozen order) damage-matched to P's ablation within +-10 %, by partial mean ablation z <- mu + lambda (z - mu) with lambda calibrated by SO-DMG. If a set's full ablation is below P's damage, its size is doubled (frozen fallback).
    - Zero ablation at 2^-8 and 2^-4 (descriptive; Morcos et al.).
    - Items: the S_L rows plus the 300 FAB rows; greedy, at most 24 tokens.
    - Outcomes:
      - CC accuracy (alias);
      - CI wrong-answer rate (alias plus K's refusal parser);
      - refusal rate;
      - FAB non-refusal rate;
      - teacher-forced Delta log q(gold) on CC;
      - damage.
- **Rule (frozen; verdict family 'L1'; tree evaluated once, first match wins).**
  - **Statistics.**
    - Priv = I-hat(P) above all 99 M sets (rank p = 0.01) AND the lower 90 % bound of I-hat(P) above F(|P|).
    - rho_2 with its paired item-bootstrap interval [rho_lo, rho_hi], computed only if Priv.
    - m_0 = the smallest grid m at which the lower bound of the median I-hat_U(m) exceeds F(m). F_small = [m_0 <= 64].
    - Syn = some m <= m_cal with Sigma(m) - Sigma_dense(m) interval above 0, and the median I-hat_U(m') not above F(m') for every m' <= m. m_cal = the largest planted threshold (16 or 64) whose L1-C calibration passed; without one, Syn is 'not testable'.
    - Mono = the I3_c upper bound < 0 at >= 3 of the 7 sizes. Red = the I3_c lower bound > 0 at >= 3 sizes.
  - **Verdicts.**
    1. 'localised: the H-cluster is the store': Priv, rho_hi < 0.5, R_rep = 1 and XH ratio >= 4.
    2. 'H privileged but one of several representations (redundant)': Priv and rho_lo >= 0.5.
    3. 'H privileged, redundancy undecided': Priv otherwise.
    4. 'holographic-like (threshold <= m_cal, synergistic, monogamous)': not Priv, Syn and Mono.
    5. 'dense (no sparse store)': I-hat(P) < 0.2 I_ref and median I-hat_U(1,024) >= 0.5 I_ref.
    6. 'distributed without a privileged set': not Priv and F_small.
    7. 'unclassified (profile reported)'.
  - **Flags attached to every L1 verdict.**
    - 'no threshold above m_0', with m_0.
    - 'not a strict threshold code (falsified)': the median I-hat_U(m) >= 0.1 I_ref with its lower bound above F(m) at some m <= 1,024 (<= 0.6 % of all neurons).
    - Monogamy sign: 'monogamous (I3_c < 0)' (Mono), 'redundant (I3_c > 0)' (Red), else 'I3 sign undecided'.
    - 'latent-scalar redundancy' if Lambda >= 0.9.
    - 'neuron basis not privileged' if the paired Rot-minus-U median difference at m in {|P|, 64, 1,024} has its 90 % interval inside +-0.03 bits.
    - 'selection-robust' / 'selection-fragile' (Priv and rho_2 recomputed on the 20 SO-HN boot sets).
    - 'no validated H-set' under SG8 failure.
  - A branch whose calibration failed reads 'not testable (calibration failed)'.
  - **What then (frozen; lf_holographic E2).** A threshold above m_cal, or a parity code, is 'not testable at n ~ 1,200': it has no low-order signature a probe could learn. The falsifiable direction stays decidable: information in fragments of size m_0 excludes any threshold above m_0, and so the strict reading (recovery only from about half the units). 'Unclassified' is never upgraded to 'holographic'.
  - **Erasure** (verdict family 'L1-erasure'; two-level bootstrap over items and random sets).
    - Curve shape on CC accuracy over the five sizes (mean over draws):
      - 'threshold-like within f <= 1/16': one adjacent size pair carries >= 70 % of the total change, and the change up to its lower size is inside +-0.04;
      - 'graceful degradation (associative-memory sense)': no pair carries > 50 % and the total change is >= 0.08;
      - 'robust up to f = 1/16': the total change is inside +-0.04;
      - otherwise 'intermediate (curve reported)'.
    - Causal privilege (Holm over the CI wrong-answer rate and the FAB non-refusal rate):
      - 'H causally privileged': the paired difference of P minus the mean of the 8 damage-matched sets excludes 0 in either outcome after Holm;
      - 'H not causally privileged': both differences inside +-0.05;
      - otherwise 'undecided'.
    - Stated now: above f = 1/16 erasure destroys the model, so a threshold near one half is not testable.
    - G-dmg failure turns the privilege line into 'not testable (no damage-matched control)'.
- **Null / controls.**
  - Label-permutation floors at every m.
  - 99 matched M sets.
  - Rotated basis (are neurons privileged over directions?).
  - XH fragments.
  - Hull exclusion in reselection.
  - S_fit / S_L split.
  - Planted codes at matched I_ref (L1-C).
  - Latent-scalar covariates.
  - FAB rows (out of distribution).
  - Nested random erasure; damage-matched erasure; zero against mean ablation.
- **Shows - cannot show.**
  - Shows: whether P is a local store or one representation among several; whether the signal has a holographic-like threshold, synergy and monogamy at sizes <= 64; whether behaviour degrades gracefully or at a threshold.
  - Cannot show: a threshold above about 64 units or a parity code; storage of the causal effect ('readable is not used', MicroLens; causal use is K's alpha-scaling); anything about AdS/CFT.
  - 'Privileged' alone is not locality, because P was selected as the best sparse set; rho_2 and the XH ratio decide.
- **MDE.**
  - SE(I-hat) about 0.015-0.02 bits at 1,200 items.
  - Paired I-hat(H_1) - I-hat(H_2) about 0.05 bits; rho_2 resolved to about +-0.25 at I-hat(H_1) ~ 0.2 bits.
  - Small-fragment information about 0.02-0.03 bits above the floor.
  - Synergy and I3_c differences about 0.04 bits.
  - Erasure rate differences about 0.05 (paired, 600 CI / 300 FAB items, set-level variance included); margin +-0.05.
- **Inputs.** SO-LAB (S_fit, S_L, FAB rows with answers and keys); SO-HN (H_op, H+, boot sets, selector, random-set generator, hook manager, REF-W); SO-DMG.
- **Runs in.** r1b_L_A, 41 A100-min. Breakdown:
  - mu_j 0.3;
  - CETT recompute about 0.5M tokens at 5,600 tok/s x 1.3 for hooks, 2;
  - selections and hulls 1;
  - rotated re-read 1;
  - PIP families, floors, synergy and pre-answer about 4,800 I-hat estimates, about 58k batched probe fits, 10;
  - reselection 2;
  - L1-C 15;
  - erasure 10 (26 conditions x 1,500 items: generation at batch 128 plus teacher forcing, about 19 s each, plus damage matching 1.5 min).
- **Gates.** SG8 (else P = the SO-HN selector's set on S_fit with 'no validated H-set'); G-sel; G-cal-L1 per branch; G-dmg.

### L2 HORIZON: RT sense, length against information, boundary encodes H (core)

**Corpus L.R** (built in r1b_L_B; used by L2, L3a, L3c and L3d):
- **L.R-A.** Four families, 64 items each:
  - F-known: S_L CC;
  - F-unknown: S_L CI;
  - F-fab: K's NE (FAB);
  - F-neu: 64 frozen neutral long-form prompts (rk_loop's 32 scene prompts plus 32 new everyday scenes; sha256 in the plan).
  - Prompts: 'Answer the question, then explain your answer in detail, in about 800 words: {question}'; F-neu 'Write a detailed story of about 800 words about {scene}'.
  - T in {0.7, 1.0}, keys keyseed('r1bL2', family, i, T); at most 1,024 new tokens. EOS is allowed: a chain that ends leaves the at-risk set, and at-risk counts are reported. 512 chains in total.
- **L.R-B.** The first 32 items of F-known, F-unknown and F-fab x 8 keys (keyseed('r1bL2b', family, i, k)) at T 1.0, at most 512 tokens: 768 chains.
- **Captured during generation (exact, unsteered):**
  - a_t of P and of every comparison population (CTRL99, SEL+, UNC+, the union of the SO-HN boot sets);
  - spins (packed bits) of P, SEL+, UNC+ and every CTRL99 set;
  - e_t, s_t, p_EOS, token ids, repeat flags.
- **Teacher-forced passes per L.R-A chain:**
  - A': the same chain after the prompt of another item of the same family (derangement seed 'r1bL2perm');
  - A'': a second foreign prompt.

  Anchor information iota_i(k) = log q(z_k | A_i, z_<k) - log q(z_k | A'_i, z_<k). Null iota0_i(k) uses A'' against A'. The true term comes from generation.
- **Natural position control.** 200 frozen WikiText-103 test-split passages (main SO-WT103; disjoint from every part-K text by split) x 1,024 tokens, read teacher-forced as the assistant reply to 'Write a detailed text.', with the same captures.

#### L2a RT sense: growth class of captured MI on equal partitions, and the model's own text

- **What.** Does the information Qwen3-1.7B captures between two equal halves of a text grow like a gapped chain (saturating, area-law-like), a critical chain (logarithmic, RT/CFT-like) or a long-range process (power law), up to 32,768 tokens? Does the model's own text carry less long-range captured information than natural text?
- **How.**
  - Estimator L.EQMI (L2M Eq. 7, vCLUB form; lean). For a document x and L in {2^7, ..., 2^15}:
    - X_L = x[0, L/2), Y_L = x[L/2, L);
    - I-hat(L) = mean over documents of the sum over j in Y_L of [log q(x_j | x_<j) - log q(x_j | X'_L, x_(L/2..j-1))], in nats;
    - X'_L = the first L/2 tokens of another document of the same source (fixed derangement), at identical positions.
    - True terms for every L come from one causal pass per document. Foreign terms need one pass of length L per (document, L). lm_head runs only at Y positions, in chunks of 2,048.
  - Documents: L.LONGDOC, 24 documents; raw text without chat template; SO-LCE at B = 1, fp32.
  - G-L2a-LCE: chunked and cached log-probs against a full pass on 2 documents at 16k and 32k; relative difference < 1e-4. If r1b_P0 shows that this tolerance fails at 32,768 but holds at 16,384, L2a's length grid is [2^7, 2^14] and the growth-class fits use that range (main OD-M14; decided before the freeze and written into the freeze record). If it fails at both lengths, L2a runs as frozen and this gate decides.
  - G-MI (holographic):
    - copy control: 4 documents with Y := X, L <= 2^12;
    - shuffle control: 8 documents with sentence-shuffled slices (seed 'r1bL-SHUF'), L <= 2^13.
  - Growth class: three least-squares fits on L in [2^7, 2^15], ranked by BIC per document-bootstrap replicate:
    - saturating I_inf (1 - e^(-L/lambda));
    - logarithmic a + b ln L;
    - power a L^beta.

    The label is the class chosen in >= 80 % of replicates.
  - L_sat (descriptive): the smallest L with I-hat(2L) - I-hat(L) < max(2 SE_boot, 0.05 I-hat(L_max)) for that and every larger L.
  - Own text (holographic L4):
    - 24 raw continuations of 512-token seeds, the first 512 tokens of each slice;
    - T 1.0, 4,096 new tokens, keys keyseed('r1bL2s', d);
    - EOS masked and p_EOS logged; raw book continuation has no natural end.
    - Both sides are conditioned on the 512-token natural seed. For L in {2^7, ..., 2^12}: I-hat_self(L) uses X, Y inside the generated part with foreign X' from another self text; I-hat_nat|seed(L) does the same on the natural slice after its seed.
    - R(L) = I-hat_self(L) / I-hat_nat|seed(L).
- **Rule** (classification, read as the RT sense; no H claim).
  - 'not decidable (estimator check failed)' if G-MI fails. G-MI needs:
    - copy I-hat >= 5 x I-hat(L) at every L <= 2^12;
    - shuffled I-hat(2^13) < I-hat(2^13), with the paired upper bound < 0.
  - Growth class:
    - 'growth class: saturating (area-law-like), L_sat = ...';
    - 'growth class: logarithmic (RT/CFT-like)', which also needs b's lower bound > 0;
    - 'growth class: power law (sub-volume, L2M-like), beta = ...';
    - otherwise 'growth class undecided (curve reported)'.
  - Own text, R(2^12):
    - 'the model's own text carries less long-range captured information': upper bound < 0.87, and the slope of ln R on ln L over [2^8, 2^12] has upper bound < 0;
    - 'no deficit': interval inside [0.87, 1.15];
    - otherwise 'undecided'.
  - Only the logarithmic class is called 'RT-like'. A fixed future block is never used: with a fixed block even a CFT saturates.
- **Null / controls.** Foreign halves (same source, length and positions); copy and shuffle controls; natural against own text.
- **Shows - cannot show.**
  - Shows: the scaling class of the information the model captures, and its own-text deficit.
  - Cannot show: the true MI (log-probs measure captured information; L2M: both estimators likely underestimate); any bulk geometry, minimal surface or central charge. The KV cache grows linearly, so there is no fixed bottleneck cut and the RT bound is trivially met.
- **MDE.**
  - Growth class: realised bootstrap.
  - Deficit: per-document SD of ln R planned at <= 0.25 gives SE 0.05; MDE 0.13 on the ln scale; margin [0.87, 1.15], about +-2.8 SE.
- **Runs in.** r1b_L_B, 29.4 A100-min. Breakdown:
  - 24 documents x (one 32k true pass of 12-21 s plus foreign passes of 65,408 tokens, 20-32 s) = 21 min (prefill-bound; int_machinery 3.6);
  - G-L2a-LCE 1;
  - G-MI 2;
  - self texts 24 x 4,096 tokens at about 4.5k context and batch 24 (about 50 ms/step) = 3.4;
  - seed-conditioned passes 2.
- **Gates.** SG9; G-L2a-LCE; G-MI; G-cal-L2 (f).

#### L2b Length against information along the model's own chains (replaces lean's R-L2a and R-L2b)

- **What.** Does H activity rise with chain length only because the continuation decouples from its context (an information-limited length artefact), or beyond the information loss? Is any rise H-specific, chain-specific and more than uncertainty?
- **How.**
  - L.R-A anchored families (F-known, F-unknown, F-fab) x T. 32-token windows w at continuation positions >= 64 while at risk.
  - Window means: a_P(w), iota(w), e(w), log pos(w).
  - Within-chain demeaning (Frisch-Waugh; chain fixed effects remove EOS-censoring composition and item heterogeneity). Slopes are residual-on-residual cov/var and standardised by the SD of window means of a_P:
    - beta_pos^0 = slope of a_P on log position;
    - beta_pos^(iota,e) = the same after partialling iota and e;
    - beta_iota = slope on iota after partialling log position and e.
  - Pooled over families with family x T fixed effects; reported per family.
  - The same slopes for every comparison channel give:
    - Delta beta^0 and Delta beta^(iota,e) = P minus the mean over CTRL99;
    - paired contrasts against SEL+, UNC+ and the E-channel.
  - Natural-position slope beta_pos^0,nat: P's activity on the natural passages read at the same positions.
- **Rule** (verdict family 'L2-length'; Holm over the three contrasts):
  - 'length effect on H is information-limited': beta_pos^0 lower bound > 0, beta_iota upper bound < 0, and beta_pos^(iota,e) inside +-0.10 SD per e-fold of position.
  - 'H rises beyond information loss': beta_pos^(iota,e) lower bound > 0 with point >= 0.10, and Delta beta^(iota,e) lower bound > 0.
  - 'no length trend of H': beta_pos^0 inside +-0.10.
  - Otherwise 'undecided'; 'not testable (no H-set)' under SG8 failure.
  - Labels:
    - 'generic (not H-specific) length trend': beta_pos^0 lower bound > 0 and Delta beta^0 inside +-0.10;
    - 'chain-specific length effect': beta_pos^0 - beta_pos^0,nat lower bound > 0;
    - 'length trend not beyond uncertainty': the passing slope's contrasts against UNC+ and the E-channel include 0 or lie below;
    - 'repetition-sensitive': a verdict changes when windows with repeat rate > 0.3 are excluded;
    - 'selection-robust' / 'selection-fragile'.
  - Descriptive:
    - occupancy curves of b by bin, as excess over F-neu;
    - iota(k) profiles;
    - the information horizon k*_free (lean: the first k at which the 8-token running mean of iota falls below the 95th percentile of the family's iota0 running mean and stays there for 8 tokens);
    - Spearman(k*_free, first H-entry);
    - the H-rise position p_H;
    - natural-text L_sat from L2a.
- **Null / controls.** Foreign-anchor null band; within-chain demeaning; 99 controls, SEL+, UNC+ and the E-channel through identical regressions; F-neu reference; natural-position control; T 0.7 against 1.0; repetition exclusion.
- **Shows - cannot show.**
  - Shows: whether H tracks the loss of context-specific information along the model's own chain, beyond controls and uncertainty.
  - Cannot show: causation (L3b); the true MI; per-segment hallucination labels on these chains (no judge; F-fab is confabulated by construction; L5c supplies exact errors).
- **MDE.** Per-chain slope SD planned at 0.3 SD per e-fold over 384 anchored chains, design effect included: SE about 0.02, MDE about 0.05. Margins +-0.10, about 5 SE.
- **Runs in.** r1b_L_B, 23.1 A100-min:
  - L.R-A decode: 16 batches x 1,024 steps x 35 ms x 1.15 = 11 min;
  - A' / A'' teacher forcing: 1.13M tokens at 5,000 tok/s = 3.8 min;
  - natural passages: 0.2M tokens, 0.8 min;
  - L.R-B decode: 24 batches x 512 steps x 32 ms x 1.15 = 7.5 min.

  Rates from plan 7.1 and int_machinery 3.6 (DynamicCache rewrite at about 1k context).
- **Gates.** SG8; G-cal-L2 (g).

#### L2c Boundary encodes H (reconstruction sense, dynamic; holographic L3, holographic verdict only)

- **What.** Does the output-side block B = (s_t, e_t) recover the internal H channel more than equal-size random channels?
- **How.**
  - L.R-A chains, positions 64-1,023.
  - Delay embedding of B with E chosen by simplex 1-step self-forecast on the calibration third of chains (index = 0 mod 3, E in 2..6, tau = 1), frozen before the other two thirds are read.
  - Simplex cross-map (E+1 neighbours, exponential weights, Theiler exclusion |t' - t| <= E): B xmap a_P(t) at lag 0, libraries {32, 128, 512, 896} x 20.
  - Skill rho = Pearson(estimate, observation).
  - Nulls and comparisons:
    - rho_cross: a_P taken from another chain of the same family and T at the same positions (common-trend null);
    - the identical pipeline for each CTRL99 channel.
  - Lag profile rho(l), l in [-8, 8]: descriptive only (Ye et al. convention; a dynamic claim would need l <= -1 and an intervention, which L does not make here).
- **Rule.**
  - 'boundary encodes the H channel beyond random channels': convergent (rho(896) - rho(32) lower bound > 0 and rho(896) - rho_cross(896) lower bound > 0) and rho_P(896) above all 99 control channels (rank p = 0.01).
  - 'boundary encodes H only like any population': convergent, rank p > 0.05.
  - 'boundary does not encode H': rho(896) - rho_cross(896) upper bound < 0.06.
  - Otherwise 'undecided'.
  - 'not decidable (cross-map pipeline miscalibrated)' if G-cal-L2 (j) fails.
- **Null / controls.** Cross-chain pairing; 99 control channels; synthetic calibration (j).
- **Shows - cannot show.**
  - Shows: whether vocabulary-side statistics carry the H channel more than random channels. If they do, any vocabulary-side readout, a lens included, will 'see' H for reasons that belong to the model.
  - Cannot show: a Takens-guaranteed reconstruction (sampling is stochastic and prompt-driven, and the state grows); a direction of influence.
- **MDE.** rho differences of about 0.05 (384 anchored chains); margin 0.06.
- **Runs in.** r1b_L_B, 3 A100-min (GPU kNN).
- **Gates.** SG8; G-cal-L2 (j).

**L2 inputs:** L.LONGDOC, SO-LCE, SO-LAB (S_L, FAB), P and L.POP, SO-CHH, the SO-HN boot sets.

**L2 total:** r1b_L_B, 55.5 A100-min.

### L3 ATTRACT: is the H-population an attractor, Ising-modellable, an energy well, irreversible? (core)

#### L3a Observational dynamics (lean, plus bistability, uncertainty and selection controls)

- **What.** In the model's own generation, is the H-state stickier than memoryless dynamics at the same item and position? Is it more persistent than control populations? Does it split within items across seeds (bistability)? And is any of this more than an uncertainty population?
- **How.**
  - Data: L.R-A, L.R-B and the twin-0 rows of L3b; b_t as in L.2.
  - **HD**: Mantel-Haenszel log-odds ratio of P(b_(t+1) = 1 | b_t = 1) against P(b_(t+1) = 1 | b_t = 0) over strata item x position bin.
  - **Stickiness gamma**: logit P(exit at t | dwell d) = item effect + position-bin effect + beta_e e_t - gamma log(1 + d). Fitted by L-BFGS on the sparse design.
  - **Null L.MARKOVSUR**: item-specific, position-matched first-order Markov surrogates. Entry and exit log-odds come from a two-way fit (item effect + shared position profile); dwell is memoryless; initial states are copied. 200 surrogates for P, 50 per comparison population.
  - Surrogate-corrected Delta gamma = gamma-hat - median surrogate gamma-hat. Run-length shuffles are not used: they preserve dwell.
  - **BI** (energy_ising L2(D)) on L.R-B:
    - per item and seed, the late occupancy o_s = mean b over continuation tokens 257-512 (seeds ending before token 320 excluded; items need >= 6 seeds);
    - an item is 'split' if >= 2 seeds have o_s >= 0.3 and >= 2 have o_s <= 0.02;
    - BI = share of split items;
    - surrogate BI from 200 item-specific surrogates simulating 8 seeds per item.
  - The same pipeline runs for every comparison population, with occupancy-matched binarisation.
- **Rule** (verdict family 'L3-obs'; Holm over {Delta gamma, HD}: rank p 0.01 <= 0.025).
  - 'H-specific stickiness': gamma-hat_P above the 95th percentile of its own surrogates, Delta gamma_P above all 99 controls, and Delta gamma_P - Delta gamma_SEL+ lower bound > 0.
  - 'H-specific persistence': HD_P above all 99 controls, and HD_P - HD_SEL+ lower bound > 0.
  - 'within-item bistability (H-specific)': BI_P above the 99th percentile of its surrogates and above all 99 controls. This is attractor route A2.
  - 'H dynamics not distinct': Delta gamma_P, HD_P and BI_P each below the controls' 90th percentile.
  - Otherwise 'undecided'.
  - Qualifiers:
    - 'not beyond uncertainty';
    - 'selection-robust' / 'selection-fragile';
    - 'repetition attractor, not H': HD and Delta gamma fall below the controls' 90th percentile once repeat-flagged windows are excluded.
  - O = 'H-specific stickiness' or 'H-specific persistence'.
- **Null / controls.** Item-specific surrogates; 99 occupancy-matched controls; SEL+ (units selected for one target share a latent driver); UNC+ and the E-channel (the uncertainty driver); item strata; repetition exclusion.
- **Shows - cannot show.**
  - Shows: whether the H-state has dynamics of its own beyond item, position, selection and uncertainty.
  - Cannot show: causation (L3b). HD > 0 alone is expected for every population.
- **MDE.**
  - gamma about 0.04 (homogeneous items).
  - Heterogeneous items: size 0/6 and power 6/6 at gamma 0.1.
  - Rank tests have power of about 0.8 when the statistic lies about 3.5 SD of the control distribution above its mean (planning simulation).
  - BI: a 15 % split share against about 2 % in controls (L-CAL d).
- **Runs in.** r1b_L_B, 6 A100-min: batched hazard refits and surrogates on the GPU, about 5,500 refits and 5,250 BI surrogate sets.
- **Gates.** SG8; G-cal-L3 (b, c, d).

#### L3b Causal twins: push and return, carry over five turns, route (lean, with fixes)

- **What.** After a transient push of the H-population, does the H-state sustain itself across turns (R6: carry > 1)? Is its carry H-specific beyond damage-matched pushes of random populations? Is it carried by the KV cache or by the text?
- **How.**
  - **Prompts.** 256 (128 F-known from S_L CC + 128 F-fab from NE (FAB)) in the L.R-A template; items 1-64 coincide with L.R-A. 2 seeds give 512 keys.
  - **Conditions per key**, in one batch with identical shapes (common random numbers):
    - twin 0 (alpha = 1);
    - H+ (z_j x 2 for j in P at the down_proj input);
    - H- (x 0);
    - R_i+ and R_i- for i = 1..6: CTRL99 sets 1-12 in frozen order. alpha_(r,+-) is calibrated by SO-DMG (matched mode, 'hooks' variant) so that each set's damage equals the H push's damage of the same sign within +-10 %. A set outside the band after 6 evaluations is replaced by the next set (at most 6 replacements; else G-dmg fails for that sign).
    - 32 sham rows (hooks active, alpha = 1).
  - **Push window.** The first 16 generated tokens of turn 1; then release.
  - **Turns.**
    - Turn 1: 64 tokens, T 0.7, keys keyseed('r1bL3', p, s).
    - Turn n >= 2: 64 tokens after TURN2_SUFFIX ('Continue.'), ids by concatenation, history think blocks kept, unmodified model, fresh prefill (round-2 B-Q8); keys keyseed('r1bL3t', n, p, s), shared across conditions.
    - Turn 2 for all conditions; turns 3-5 for H+, H- and twin 0 only (multi-turn carry, AUSWERTUNG step 5).
  - **Reads**, each on the pushed population's own channel (self-read; this fixes lean's unspecified read set):
    - R1 for every turn;
    - R2 live during turn-1 tokens 17-64;
    - R3 for turn 1 (H+-, twin).
    - Cross-read of the random pushes on P's channel: descriptive.
  - **Carry.**
    - d^(n) = mean over keys of (a+^(n) - a-^(n)) / 2 (R1).
    - K_H = d^(2) / d^(1), defined only if d^(1)'s interval excludes 0 (round-2 G26).
    - K_H^pool = the K minimising sum over n = 1..5 of w_n (d^(n) - d^(1) K^(n-1))^2, with w_n = 1 / var(d^(n)). Found by 1-D grid search on [0, 2.5] in steps of 0.005; no matrix inversion.
    - K_(R,i) = d_(R,i)^(2) / d_(R,i)^(1), defined per set.
    - Delta K = K_H - mean over defined sets of K_(R,i), with a two-level bootstrap (prompts, then sets).
    - One-sided carries K_up and K_down against twin 0: descriptive.
  - **Return profile** (lean): within-turn retention and KV share = (Delta_live - Delta_re) / Delta_live in block 17-32.
  - Descriptive: the share of the pushed-minus-twin block-14 difference on SO-AX's u*_14 (from r1b_xfer_A).
- **Rule** (verdict family 'L3-causal').
  - **Gates first.**
    - G-twin: sham rows token-identical to twin 0 in >= 31/32; else 'not decidable (common random numbers broken)'.
    - G-manip: live in-window displacement of P's channel (tokens 9-16) >= 3 SD of sham-vs-twin in H+ and H-; else 'not testable (push too weak)'.
    - G-push: d^(1)'s interval excludes 0; else 'K not defined (push too weak)'.
  - **Self-sustainment** (R6 thresholds on K_H^pool):
    - 'self-sustaining H-state': lower bound > 1;
    - 'dissipative': upper bound < 1;
    - otherwise 'undecided'.
    - The one-step K_H with the same thresholds is reported for comparison with round 2.
  - **Specificity:**
    - 'H-specific carry': Delta K lower bound > 0;
    - 'carry like damage-matched controls': Delta K inside +-0.40;
    - otherwise 'undecided';
    - 'not testable (random carries undefined)' if fewer than 4 of 6 sets have a defined K_R.
  - **Route** (turn-1 block 17-32):
    - 'carried mainly by the KV cache': KV share >= 0.8 with lower bound >= 0.5;
    - 'carried by the text': KV share <= 0.2 with upper bound <= 0.5;
    - otherwise 'mixed route'.
  - **Labels:** 'high-damage push' (H damage > 0.15 nats); 'generic H rise under damage' (cross-read d^(1) interval > 0 for >= 3 of 6 sets).
  - **Attractor route A1** = 'self-sustaining H-state' and 'H-specific carry', both at one-sided bootstrap p <= 0.025 (Holm with A2; <= 0.05 if A2 has passed).
- **Null / controls.** Twins with shared keys (the push is the only difference); 6 damage-matched random sets per sign with set-level resampling (between-set variance); sham; context-free turn reads (the turn-1 text cannot inflate the turn-2 read); live against cache-free reads.
- **Shows - cannot show.**
  - Shows: causal self-sustainment and H-specificity of an induced H-state, and its route.
  - Cannot show: that natural hallucinations arise this way.
  - K in (0, 1) with H-specific carry is snowballing-type conditioning on the model's own text (Zhang et al. 2023), not an attractor.
  - Large alpha can leave the activation distribution (Gao et al. report non-monotonic responses), hence damage matching and the dose ladder (X1).
- **MDE.**
  - SD(K) about sqrt(1 + K^2) / z1 with z1 about 10 at 512 keys. This is conservative: round 2's turn-1/turn-2 errors were positively correlated, and at z1 ~ 15 the observed SD was 0.05. One-step SD is 0.11 at K 0.5 and 0.16 at K 1.3.
  - The pooled 5-turn carry is planned at 0.6 x the one-step SD for K >= 1 and 0.9 x for K < 1. 'Self-sustaining' therefore has power of about 0.8 at K_H about 1.25-1.3, and 'dissipative' at K_H <= 0.8.
  - SD(Delta K) about 0.136 with 6 sets and a between-set SD of 0.10 (assumed): MDE 0.34, margin 0.40.
  - Round-2 anchors: K 0.449-0.482 along u_read with u_read . d^(1) = 8.33 [7.44, 9.23] at 128 pairs; K on u_T_perp 1.053 [0.743, 1.538] (descriptive).
- **Runs in.** r1b_L_A, 34.5 A100-min:
  - 19,968 texts x 64 tokens = 624 calls x (64 steps x 27 ms x 1.15 + prefill 0.4 s) = 25 min;
  - R1 reads 2.48M tokens at 5,600 tok/s = 7.4;
  - R3 0.2M tokens = 0.6;
  - damage calibration of 12 sets with at most 6 RefKL evaluations each = 1.5.
- **Gates.** SG8; G-twin; G-manip; G-push; G-dmg.

#### L3c Ising-modellable H-population: static max-ent with energy basins, and kinetic Ising (energy_ising L1 plus lean L3c)

- **What.** Is the H-population a collective code that a pairwise Ising model captures, beyond 99 random populations and beyond selection- and uncertainty-matched sets? Where are its energy basins ('Energiemulden')? Do its kinetic couplings break detailed balance?
- **How.**
  - **Spins** as in L.2 for P, SEL+, UNC+, CTRL99 (size |P|) and the 20 SO-HN boot sets. If |P| < 10 the fit uses H+; if that is still < 10, 'not testable (population too small)'.
  - **Data.** L.R-A plus L.R-B; fit half = odd items, test half = even items.
  - **G-ESS:** fit tokens / tau_int(m_P) >= 10 x N(N+1)/2. Else the top 30 units by K's selection frequency; else 'not testable (sample)'.
  - **Static models**, identical for every population:
    - (i) independent;
    - (ii) pairwise by pseudo-likelihood (L2 penalty by inner CV; J symmetrised);
    - (iii) K-pairwise, a joint pseudo-likelihood with one shared V(K), used if (ii) fails G-ISM;
    - (iv) covariate fields h_i(position bin x family), the shared-driver refit (Schwab, Nemenman, Mehta 2014).
  - **G-ISM model check** on the test half:
    - held-out PL gain of (ii) over (i) with lower bound > 0;
    - P(K) from forward Gibbs sampling (512 chains x 4,000 sweeps after 1,000 burn-in) inside the test half's 90 % item-bootstrap band for >= 9 of K = 0..10;
    - mean connected triplet correlation inside its band.
  - **Statistics.**
    - dPL = held-out PL gain per token and unit, (ii) minus (i).
    - I2/IN on 20 random 10-unit subsets, by exact max-ent fits through iterative scaling over the 1,024 states (forward moment matching). Miller-Madow plug-in S_N with counts scaled to the ESS.
  - **Basins.** Zero-temperature single-flip descent from every test-half pattern; basins with >= 1 % occupancy. H-basin = the basin with the largest positive log-odds for item hallucination (CI against CC; covariates answer and chain length) on the fit half, with test-half AUROC. Descriptive.
  - **Rostami check.** A bimodal fitted P(K) counts only if the empirical P(K) is bimodal too (dip p < 0.05).
  - **Kinetic Ising** (lean): per-unit forward logistic regressions of s_(j,t+1) on s_t with fields h_j(family, T, bin). G_pair = held-out PL gain over the independent model with self-coupling.
  - Descriptive:
    - asymmetry A_J = ||J - J^T||_F / ||J + J^T||_F (non-equilibrium marker);
    - modularity communities by power iteration on the modularity matrix of (J + J^T) / 2;
    - modularity Q reported next to ||J|| ('form, not size', AT8 aside);
    - Glauber simulation against observed dwell.
  - **G-cal-Ising.** Five planted sets with the measured sample count and tau_int:
    - P1 independent;
    - P2 weak couplings;
    - P3 two Hopfield basins;
    - P4 near-critical;
    - P5 hidden-field-driven.

    Pass if >= 4 of 5 are right: no extra basin >= 1 % in P1 and P2; P3's two basins recovered with overlap >= 0.8; no collective verdict for P1 and P5 after (iv).
- **Rule** (verdict family 'L3-Ising'; Holm over the static and the kinetic line).
  - 'collective (Ising-modellable) H-population' if either line holds:
    - static: G-ISM passes, median I2/IN >= 0.8 with lower bound >= 0.7, dPL_P above all 99 controls, and dPL_P - dPL_SEL+ lower bound > 0 both without and with covariate fields (iv);
    - kinetic: G_pair(P) above all 99 controls and G_pair(P) - G_pair(SEL+) lower bound > 0.

    Holm: rank p 0.01 <= 0.025; each paired contrast one-sided at the Holm level.
  - 'collective, but no more than a selection-matched set (selection effect)': the rank criterion passes, the SEL+ contrast fails.
  - 'pairwise structure not H-specific': dPL_P and G_pair(P) below the controls' 90th percentile.
  - 'not Ising-modellable (higher-order structure)': G-ISM fails for (ii) and (iii), or the I2/IN upper bound is < 0.6.
  - 'independent H-units': I_N per unit upper bound < 0.005 bits on the 10-unit subsets.
  - Otherwise 'undecided'.
  - Qualifiers: 'not beyond uncertainty' (against UNC+); 'selection-robust' / 'selection-fragile'.
  - Basins are reported only if G-cal-Ising passes. Their content is not named, except by X5.
  - Criticality scans are not run (L.12).
- **Null / controls.** Independent model; 99 controls; SEL+ (hull-matched, which removes the hull advantage holographic's L7 built in); UNC+; covariate-field refit; Rostami check; planted sets.
- **Shows - cannot show.**
  - Shows: whether a pairwise Ising model describes the H-population and whether that is H-specific.
  - Cannot show:
    - wiring: couplings are effective and absorb shared inputs;
    - dynamical attractors: minima of a fitted energy are not attractors of a network that runs no Glauber dynamics;
    - a physical temperature (model T is a parameter; Tkacik et al. 2015).
- **MDE.**
  - Sampling is not limiting under G-ESS (dPL resolved to about 1e-4 nats per unit and token).
  - The rank tests need about 3.5 control SD for power 0.8.
  - I2/IN about 0.1 (20 subsets, planning SD 0.15).
- **Runs in.** r1b_L_B, 8 A100-min: about 130 PL fits, Gibbs checks, I2/IN, basins and kinetic fits on the GPU.
- **Gates.** SG8; G-ESS; G-ISM; G-cal-Ising; G-cal-L3 (e).

#### L3d EBM and non-equilibrium at H entry (decisive rules from holographic L2c and L6; lean's surrogates)

- **What.** At entry into the H-state, does the model's own energy (-log q) behave like a well (entropy falls and the output trajectory is trapped), as the EBM-reasoner picture of wrong attractors suggests, or like an uncertainty regime? Is entry time-irreversible beyond a detailed-balance null?
- **How.**
  - **Entry events:** the first b = 1 after >= 16 tokens of b = 0, at continuation positions 64-960 (L.R-A and L.R-B), per population (each control with its own events).
  - **Matched windows:** the same positions in chains of the same family and T with no entry within +-64 tokens.
  - **EBM.**
    - Delta e = (e_post - e_pre)_event - (e_post - e_pre)_matched, over 64-token windows; Delta s likewise.
    - Delta TT: trapping time of the (s, e) delay embedding (m = 4, tau = 1, recurrence rate fixed at 5 % per window, Theiler window 2, v_min 2; Marwan et al. 2002), post minus pre, minus the matched difference, in units of the pooled SD of window TT.
    - Difference-in-differences against matched windows replaces IAAFT, which is invalid under position trends.
  - **Irreversibility.**
    - a_P and s symbolised per chain into quartiles (positions >= 64).
    - n-string counts pooled over the windows of a condition, Jeffreys pseudo-count 0.5.
    - D_n = sum_w p(w) ln[p(w) / p(w^R)]; d-hat = D_3 - D_2 (Roldan and Parrondo 2010, Eq. 5).
    - Reversible band: 200 item-specific, position-matched first-order Markov surrogates of the symbol series, symmetrised to detailed balance (P_ij pi_i = P_ji pi_j). They preserve occupancy and position profile with zero entropy production.
  - **Descriptive:**
    - Sagawa's sigma (sentence-order reversal, prompt fixed, per token) on the model's own T 1.0 L.R-A chains, with family contrasts, gated by G-sigma (integral fluctuation theorem check: 8 neutral prompts x 4,000 samples of 5 tokens at T 3 and 4, |mean exp(-sigma_tok) - 1| <= 3 SE);
    - the joint-state log-loss asymmetry A_k (k-means states of P's z, k in {2, ..., 8}, lags {1, 2, 4, 8, 16}). Frozen reading: A_k measures non-stationarity, never irreversibility, because for a stationary process forward and backward conditional entropies are equal whether or not it is reversible. This is the user's 'vector quantity';
    - A_J from L3c.
- **Rule.**
  - **EBM** (verdict family 'L3-EBM'):
    - 'energy-well entry (EBM reading supported)': Delta e_P upper bound < 0, Delta TT_P lower bound > 0, and Delta e_P below 95 of 99 control populations' Delta e (rank p <= 0.05);
    - 'uncertainty regime at entry (not a well)': Delta e_P lower bound > 0;
    - 'no energy signature at entry': Delta e_P inside +-0.10 nats and Delta TT_P inside +-0.30 SD;
    - otherwise 'undecided'.
    - Delta s is reported.
  - **Irreversibility** (verdict family 'L3-irrev'; Holm over the a_P and s channels):
    - 'irreversible H-entry (entry-specific)': d-hat_post - d-hat_match lower bound > 0, and d-hat_post above the reversible band's 95th percentile;
    - 'irreversibility not entry-specific': d-hat above the band in both post and matched windows, with the difference inside +- the band's half-width;
    - 'reversible within resolution': d-hat_post inside the band;
    - otherwise 'undecided'.
    - Label 'H-specific' if P's a-channel contrast exceeds 95 of 99 controls' contrasts.
  - 'descriptive (sigma pipeline check failed)' if G-sigma fails.
- **Null / controls.** Matched windows; 99 controls' own entries; detailed-balance surrogates; the IFT known answer.
- **Shows - cannot show.**
  - Shows: whether -log q behaves like a well at H entry, and whether entry is irreversible.
  - Cannot show: energy descent (Qwen never descends an energy; one forward pass per token); heat, a physical temperature or dissipation (no bath; sigma mixes compression loss and model mismatch).
  - S-dot is a lower bound under coarse-graining.
- **MDE.**
  - Delta e: planning SE about 0.035 nats (about 1,000 entry events), MDE 0.09, margin 0.10.
  - Delta TT: SE about 0.10 SD, MDE 0.25, margin 0.30.
  - d-hat resolution: the realised band half-width.
- **Runs in.** r1b_L_B, 2.5 A100-min (sigma reversed pass 0.23M tokens 0.8 min; IFT check with the prompt KV cached 1.6 min; RQA and d-hat on CPU).
- **Gates.** SG8; G-cal-L3 (h, i); G-sigma.

#### L3 combined lines (in L-ANA; no new test)

- **Attractor line.**
  - 'H-attractor (self-sustaining or bistable, H-specific, seen in natural dynamics)': (A1 or A2) and O. Holm over A1 and A2.
  - 'self-sustaining H-state under push, not seen in natural dynamics': (A1 or A2) and not O.
  - 'sticky but not self-sustaining': O, K_H^pool upper bound < 1, not A2.
  - 'no H-attractor (dissipative, not distinct)': K_H^pool upper bound < 1, not A2, not O, and 'H dynamics not distinct'.
  - 'dissipative (observational lines inconclusive)': K_H^pool upper bound < 1, not A2, not O, and not 'not distinct'.
  - 'not decidable (push too weak)': G-push or G-manip fails and A2 fails.
  - Otherwise 'undecided'.
- **Specificity line.** 'H-specific carry' / 'carry like damage-matched controls' / 'undecided' (L3b).
- **Ising line** (L3c), **EBM line**, **irreversibility line** (L3d): as stated.
- **Reading reconciliation** (between the designs' push-out readings):
  - Restoring after a release shows local stability of the current state. It cannot separate an attractor from context-driven slaving, so it counts for neither.
  - Only self-sustainment (R6), within-item bistability, or capture beyond a dose threshold (X1) counts as attractor evidence.
  - Persistence after a push-out counts only as capture beyond a threshold.

### L4 DUET: agent -> Ising-modellable -> collaboration -> entity-forming, with H-neuron gain as the coupling (core; lean skeleton, energy_ising controls, holographic macro)

- **What.** Do two Qwen3-1.7B agents behave as coupled Ising spins? Is their coupling more than lexical priming, i.e. collaboration? Under opposite priming, do they merge into one entity? Does H-neuron gain in one agent set its coupling (conformity, as Gao et al.'s over-compliance and sycophancy effects suggest) more than damage-matched random gain?
- **How.**
  - **Items.** 32 SO-D1STIM F-sym items 0-31 (invented referent i, attribute slot j = i mod 6, v1 / v2 = the slot's version-1 / version-2 values).
  - **Design.** 2 configurations x 4 seeds per item give 256 runs per condition.
  - **System prompt (frozen):** 'You are Agent {X}, a researcher. [Your notes say that the {attribute} of {entity} is {v}.] You are discussing the {attribute} of {entity} with a colleague. Reply in at most three sentences.' The bracketed sentence is omitted for a neutral agent. There is no instruction to agree or to persuade, which fixes lean's 'agree' task.
  - **Turns.**
    - A speaks first after 'Your colleague asks for your view on the {attribute} of {entity}.'
    - Every message reaches the listener as 'Your colleague says: «{message}»'.
    - 12 messages (6 per agent), each at most 40 new tokens, T 0.7.
    - Keys keyseed('r1bL4', item, cfg, seed, msg), shared across conditions.
    - Contexts by B-Q8 id concatenation (history think blocks kept), re-prefilled every message (exact). Carried caches only if L0 (iv) passed parity.
  - **Conditions (core).**
    - C-opp: A primed v1, B primed v2 (configuration 1) and swapped (configuration 2).
    - C-half: A neutral; B primed v1 or v2.
    - C-doc, the non-social control: A neutral and alone, under system prompt 'You are a researcher analysing the {attribute} of {entity}. Reply in at most three sentences.' Its user turns carry B's messages replayed from the matched C-half run (same item, configuration and seed) as 'You find this note in the archive: «{message}» Continue your analysis.'; 6 replies.
    - C-Hup: C-half with A's P scaled alpha = 2 at all of A's generated tokens.
    - C-Rup: C-half with damage-matched random sets (4 CTRL99 sets, 21-24, damage within +-10 % of H's alpha = 2 damage), rotated so that every item x configuration uses each set once across its 4 seeds.
  - **Readouts.** Decoupled from any push.
    - (i) Message stance m: the unmodified model reads each message context-free in a frozen frame. User turn 'Text: «{message}»\nAccording to this text, what is the {attribute} of {entity}?', assistant prefix 'The {attribute} of {entity} is'. m = sum log q(' v1' tokens) - sum log q(' v2' tokens), teacher-forced. Spin s = sign(m - c); c = the midpoint of the mean m of first messages of v1- and v2-primed agents in C-opp, cross-fitted on odd/even items.
    - (ii) Belief probe, unpushed conditions only, descriptive: append-and-crop on the agent's context with 'In one word: what is the {attribute} of {entity}?' at events 0, 6 and 12.
  - **Kinetic Ising** (De Nobili's form; forward logistic regression with item fixed effects). Per listener update:

    logit P(s_X(m) = +1) = 2(h_0 + h_item + h_p c_X + J_s s_X(prev) + J_c s_Y(latest)),

    with c_X = the priming sign (0 if neutral), per condition and direction.
  - **Coupling statistic.** J_ex = J_c - mean J_c over 200 within-(item, configuration) re-pairings of the partner series (derangements among the 4 seeds), with an item-cluster bootstrap that recomputes 20 re-pairings per replicate.
    - It is unbiased under shared item fields but attenuated (sim_duet2.py).
    - It replaces lean's across-item shuffle, which gives false 'collaboration' at a rate of 1.00 with item-field SD 0.5 (judge_L1 sim_item_confound.py).
    - Item fixed effects alone gave a 0.44 false-positive rate at item-field SD 1.0 (sim_duet.py).
    - C-doc gives J_ex,doc with the replayed series as 'partner'.
  - **Further statistics.**
    - Consensus share (C-opp) = share of runs with s_A = s_B on both agents' last messages (11, 12).
    - Agreement excess = consensus minus within-item re-paired consensus.
    - Stance-gap ratio = |m_A - m_B| (late) / |m_A - m_B| (round 1).
    - Pair macro Delta Psi (holographic L8): V_r = m_A,r + m_B,r after each exchange (supervenient); Psi = I(V_r; V_(r+1)) - I(m_A,r; V_(r+1)) - I(m_B,r; V_(r+1)) by Gaussian-copula MI on ranks; Delta Psi = Psi minus the mean over 200 within-item re-pairings.
    - DD = J_ex(C-Hup, A <- B) - J_ex(C-Rup, A <- B), two-level bootstrap over items and random sets.
    - Manipulation check: A's first message, before any partner message. commit = 1[|m| >= the median |m| of primed first messages in C-opp]. The commit rate is compared between C-Hup and C-Rup.
    - Catfish carry (energy_ising L8; attached to D.17 as a qualifier, main C22): K_B = mean sigma~_B(6) / mean sigma~_B(1), with sigma~ = m signed toward B's priming, in C-opp against C-half; defined only if sigma~_B(1)'s interval excludes 0.
- **Rule** (Holm within each link family).
  - **G-duet.** >= 90 % of runs complete with non-empty messages, and the AUROC of m on first messages of v1- against v2-primed agents in C-opp is >= 0.80. Else 'not testable (format / readout)'.
  - **Ising-modellable pair** (C-half, A <- B):
    - 'agents Ising-modellable (coupled)': J_ex lower bound > 0, and the held-out log-loss gain of adding J_c (item-level 4-fold) has lower bound > 0;
    - 'no detectable inter-agent coupling': J_ex inside +-0.12;
    - otherwise 'undecided'.
  - **Collaboration:**
    - 'collaboration (coupling beyond lexical priming)': 'agents Ising-modellable (coupled)' and J_ex(C-half) - J_ex,doc lower bound > 0;
    - 'influence explained by lexical priming (no social coupling)': J_ex(C-half) lower bound > 0 and the difference inside +-0.16;
    - otherwise 'undecided'.
  - **Entity-forming** (C-opp):
    - 'merge into one entity (consensus)': consensus share lower bound >= 0.5 and agreement excess lower bound > 0;
    - 'two entities persist': consensus share upper bound <= 0.2 and stance-gap ratio lower bound >= 0.8;
    - otherwise 'mixed'.
    - Merging supports the chain's entity link. 'Two entities persist' counts against it and is attached to D.17 as a qualifier (main C22).
  - **H as coupling** (Holm over up; over up and down if X2 runs):
    - gate G-duet-manip: commit(C-Hup) - commit(C-Rup) lower bound > 0; else 'not testable (H gain has no specific effect on agent A)';
    - 'H gain acts as coupling (conformity)': DD lower bound > 0;
    - 'H gain acts like damage': DD inside +-0.17, and J_ex(C-Rup) - J_ex(C-half) excludes 0;
    - 'no effect of H gain on coupling': both inside +-0.17;
    - otherwise 'undecided'.
  - **Catfish:**
    - 'competitor sustains the stance (catfish)': log K_B(C-opp) - log K_B(C-half) lower bound > log 1.25;
    - 'competitor captures or dilutes the stance': upper bound < -log 1.25;
    - 'no catfish effect': inside +-log 1.25;
    - otherwise 'undecided'.
  - **Pair macro** (descriptive, holographic sense): 'redundant pair macro (conformity)' (Delta Psi upper bound < 0) / 'synergistic pair macro' (lower bound > 0) / 'no pair macro beyond uncoupled agents' (inside +-0.025 nats). Never counted for the entity link: a positive Psi alone is not support.
- **Null / controls.** Within-item re-pairing (removes coupling, keeps item fields and priming); item fixed effects; C-doc (lexical priming without a partner); counterbalanced priming and first speaker; common keys across conditions; damage-matched random gain with set-level resampling; context-free stance readout.
- **Shows - cannot show.**
  - Shows: whether the user's chain holds literally for two instances of this model, and whether H gain sets the coupling.
  - Cannot show:
    - anything about units inside one forward pass (no goals, memory or payoffs);
    - phase transitions (N = 2 gives finite-size behaviour only);
    - populations (24-agent naming games are prefill-bound at about 4.5 A100-h; not run).
- **MDE.**
  - J_ex SE about 0.04; J_ex 0.14 at J_c = 0.2 (sim_duet2.py); MDE about 0.10 in J_ex units; margin 0.12.
  - Collaboration contrast SE 0.055: MDE 0.14; margin 0.16.
  - DD SE about 0.058: MDE 0.145; margin 0.17.
  - Consensus share SE about 0.04.
  - log K ratio SE about 0.06: MDE 0.15.
- **Inputs.** SO-D1STIM (F-sym items 0-31); P and L.POP; SO-DMG; keys.
- **Runs in.** r1b_L_A, 34.4 A100-min:
  - 4 dyad conditions x 8 batches x 12 messages x (re-prefill of 32 x about 420 tokens at 5,000 tok/s + 40 decode steps at 30 ms) = 4 x 6.2 min;
  - C-doc 2;
  - stance reads of 13,824 messages x 2 passes x 70 tokens at 5,600 tok/s = 5.8;
  - probes 0.8;
  - calibration and parity 1.
- **Gates.** G-duet; G-duet-manip (H line); G-dmg; G-cal-L4.

### L5 OWNER: does the chain lens's break-off belong to the model or to the lens? (core; replaces lean's R-L2c)

#### L5a Ownership of the break-off on items with exact labels (holographic L1a, with fixes)

- **What.** When K's chain lens confabulates more as chains grow, does it report the model's own wrong values (model-owned), or numbers that neither the truth nor the model's own computation contains (lens-owned)?
- **How.**
  - Corpus: SO-LENSOUT on the 406 items with lens records (256 correct = known, 150 wrong = prone). Chain positions q = s = 0..12 primary; K's steps 13-16 descriptive.
  - Completed maximal digit runs per q (Qwen3 tokenises digits singly; leading minus signs ignored). Classes:
    - TRUE: in V_i;
    - MODEL: in M_i \ V_i;
    - FOREIGN: otherwise.
  - M_i, the lens-free model-own value set, built only from greedy and T 0.7 samples (judge fix):
    - (i) K's 5 gating answers plus 6 samples at T 0.7 generated in r1b_L_B with keys ('r1bL5', 'GATE', i, s), s = 1..6 (11 in all);
    - (ii) own CoT: the item prompt with the reply instruction replaced by 'Work it out step by step, one operation per line, then give the final number.'; greedy + 4 samples at T 0.7, at most 96 tokens, keys keyseed('r1bL5', 'COT', i, s); every integer parsed;
    - (iii) isolated sub-answers: 'Compute {sub-expression}. Reply with only the final number.'; greedy + 4 samples at T 0.7, at most 12 tokens, keys keyseed('r1bL5', 'SUB', i, k, s).
  - Chance-match null (judge fix): every readout of item i is also classified against M_j \ V_i of a foreign item j (same stratum and family; derangement seed 'r1bL5-FM').
  - Break-off times (K's convention: q = 0 failures reported separately):
    - B_F = the first q >= 1 with a FOREIGN run;
    - B_M = the first q >= 1 with a MODEL run;
    - C_F(q) and C_M(q) = cumulative shares;
    - C_M^c(q) = C_M(q) - C_M^foreign(q).
  - Growth and share:
    - G_F = C_F(12) - C_F(3);
    - G_M^c = C_M^c(12) - C_M^c(3);
    - Omega = G_M^c / (G_M^c + G_F), over replicates with a positive denominator (share reported).
  - Descriptive: per block and variant; per-step category shares; the share of K's k*_conf events that L reclassifies as MODEL.
- **Rule** (verdict family 'L5a'; Holm over G_F(known) and Omega).
  - 'break-off owned by the lens': G_F(known) lower bound > 0 and Omega(all) upper bound < 0.5.
  - 'break-off owned by the model': G_F(known) inside +-0.04 and Omega(prone) lower bound > 0.5.
  - 'shared (the lens adds confabulation to the model's own errors)': G_F(known) lower bound > 0 and Omega(prone) lower bound > 0.5.
  - 'no growth of false readouts with chain length': (G_F + G_M^c)(all) inside +-0.04.
  - Otherwise 'undecided'.
  - 'not testable (no exact-label lens corpus from K)' under G-CL failure.
- **Null / controls.** Known items as the internal control (the model computes correctly); foreign-M_j chance matches; K's foreign-state and neutral-seed chains as delivered (lens baseline, descriptive).
- **Shows - cannot show.**
  - Shows: whether the lens's length effect reports or adds to the model, where truth is exact.
  - Cannot show: non-arithmetic content; an unwritten internal value ('MODEL' is behavioural); other models.
- **MDE.** G_F(known) about 0.03 (per-item SD of the chain mean about 0.2, 300 items); Omega about 0.08. Margins 0.04.

#### L5b Ownership of the H-pattern (holographic L1b)

- **What.** Is the association of break-off with H (the user's 'field of view correlates with H-neurons') carried by the state the lens reads (model), or added by the lens rollout?
- **How.**
  - Model-frame H: a_P(t*), read before any answer in an unmodified teacher-forced pass of the item prompt; also for the 99 control channels.
  - Lens-frame H: K's share_H(s).
  - E_c = 1[B_F <= 6].
  - rho_b = Spearman over items between a_P(t*) and the item mean of E_c, per block, averaged.
  - rho_w = mean over items of the within-item Spearman, across seed ranks within a block, between the chain mean of share_H over s = 1..6 and E_c. All seed ranks read the same state, so this variation is the lens's own. Items with constant E_c drop out (counted).
  - Null for rho_w: 200 within-item seed permutations of share_H.
- **Rule** (verdict family 'L5b'; Holm over rho_b and rho_w).
  - 'H-pattern owned by the model': rho_b lower bound > 0 and rho_w inside +-0.05.
  - 'H-pattern owned by the lens': rho_w lower bound > 0 and rho_b inside +-0.12.
  - 'H-pattern in both': both lower bounds > 0.
  - 'no H-pattern': both inside their margins.
  - Otherwise 'undecided'.
  - Label 'H-specific (model frame)' if rho_b(P) exceeds 95 of 99 control channels' rho_b.
  - 'not testable' under G-CL or SG8 failure.
- **Null / controls.** Within-item seed permutation; K's RS39 band of share_H; 99 control channels at t*.
- **Shows - cannot show.** Shows whether the H-pattern belongs to the state or to the readout. Cannot show H's causal role (K's clamps; L3b).
- **MDE.** rho_b about 0.12 (406 items with lens records; SE about 0.049; margin 0.12, about 2.4 SE; stated, not changed); rho_w about 0.035 (margin 0.05).

#### L5c Does H rise before the model's own errors along its own chain? (new: the exact-label, position-resolved, lens-free test)

- **What.** In the model's own step-by-step computation, does H activity just before step k predict the model's own first error at k, within item, beyond step index, position and uncertainty? This ties 'H rises along the chain' to 'the model errs there' with exact labels and without a lens or a judge.
- **How.**
  - Own-CoT texts on 600 items (the 406 lens items plus the next 194 gated A-lens pool items in pool order, A-cal excluded) x 5, generated with captures of a_t for P and every comparison population, e_t and s_t.
  - Frozen parser: lines 'a o b = r'. Step k is correct iff r equals the true intermediate v_k.
  - k_err = the first wrong step; CoTs that do not parse are excluded (counted; G-parse needs >= 70 % parsed).
  - Pre-step position t_k^- = the token before the first digit of step k's result.
  - Conditional (item fixed-effect) hazard over steps at risk:

    logit P(error at k | none before) = item + step index + beta_H z(a_P(t_k^-)) + beta_e z(e(t_k^-)) + beta_pos log t_k^-,

    with item-cluster bootstrap. The same model for each comparison channel; between-item (pooled, stratum fixed effects) version descriptive.
- **Rule** (verdict family 'L5c').
  - 'H predicts the model's own step errors (within item, beyond step and uncertainty)': beta_H lower bound > 0 and beta_H above 95 of 99 controls (rank p <= 0.05).
  - 'H tracks uncertainty only': beta_H lower bound > 0 without beta_e, but its interval includes 0 with beta_e.
  - 'no H signal before own errors': beta_H inside +-0.15 log-odds per SD.
  - Otherwise 'undecided'.
  - 'not testable (fewer than 150 within-item error events)'.
  - Qualifiers: 'not beyond uncertainty' (UNC+, E-channel); 'selection-robust' / 'selection-fragile'.
  - Relation to K8c: K8c (part K; ACH-long; RS39) is confirmatory in K. L5c and K8c form the round-level family CF-HERR (main §2.4); both run and are reported side by side.
- **Null / controls.** Within-item comparison (removes prone-against-known item type, which H detects by construction); step index and position; entropy; 99 controls; SEL+ and UNC+.
- **Shows - cannot show.**
  - Shows: a lens-free, exact-label link between H activity and the model's own errors along its own chain.
  - Cannot show: causation; transfer from arithmetic to factual chains.
  - The H-score comes from a TriviaQA-trained probe, so its use on arithmetic is a transfer test.
- **MDE.** About 3,750 step observations and about 1,200 error events (planning): SE about 0.05, MDE 0.125, margin 0.15.

**L5 inputs:** SO-LAB's gated A-lens pool items (K4), SO-LENSOUT, P and L.POP, SO-CHH, the SO-HN boot sets.

**L5 runs in:** r1b_L_B, 3.1 A100-min of generation plus 0.33 min of work for the 6 extra gating samples (OD-M9):
- own CoT 3,000 x <= 96 tokens at batch 64 = 2.6;
- sub-answers 7,500 x <= 12 tokens = 0.4;
- the t* pass 0.1.

Analysis runs in L-ANA (CPU).

**L5 gates:** G-CL; SG8 (L5b, L5c); G-parse; G-cal-L5.

## L.4 Extensions (round tier 2, main §0.3; main §4.5's drop items 2-5, 11 and 12 keep L's order X5, X4, X3, X6, X1, X2; frozen; each can be dropped alone; none changes a core rule)

### X1 L3-DOSE: dose ladder and hysteresis (both judges: missing everywhere)

- **What.** Do H pushes from matched H-low and H-high states show capture beyond a dose threshold (a separatrix, i.e. bistability), or dose-independent relaxation (monostable)?
- **How.**
  - Trajectories: 256 L.R-A chains at T 0.7 (all four families).
  - Starts:
    - t_up = the first 32-token block start >= 64 whose preceding block has mean a_P <= q20 of its position bin over L.R-A;
    - t_down = the first start >= 64 whose preceding block has mean a_P >= q80.

    Chains without a start drop out of that arm (counted).
  - Twin protocol: the prefix is teacher-forced to t0; the per-row Gumbel generator is seeded with the chain key and advanced by draw-and-discard. The twin must reproduce the stored chain for 80 tokens in >= 99 % of trajectories (G-twin-X1).
  - Rows:
    - UP: twin; P at alpha in {1.5, 2, 3}; 2 random sets damage-matched to alpha 3;
    - DOWN: twin; P at alpha in {0.5, 0}; 2 sets matched to alpha 0.

    Window 16 tokens; 80 new tokens.
  - Delta(t) = a_P(pushed) - a_P(twin), live.
  - D_early = mean over release +1..+16; D_late = mean over +33..+64.
  - Retention rho(alpha) = D_late / D_early (not the mechanical in-window Delta), defined if D_early's interval excludes 0.
- **Rule** (Holm over UP and DOWN).
  - 'capture above a dose threshold (bistable, H-specific)': rho_P(top dose) lower bound >= 0.9, rho_P(top) - rho_P(lowest) lower bound > 0, and rho_P(top) - mean rho_R(top) lower bound > 0.
  - 'monostable relaxation (no separatrix within the ladder)': rho_P(top) upper bound < 0.9 and |rho_P(top) - rho_P(lowest)| inside +-0.2.
  - Otherwise 'undecided'.
  - 'hysteresis' if both arms read capture.
  - A capture verdict adds attractor route A3 to the L3 attractor line: Holm over {A1, A2, A3}.
- **Null / controls.** Twins; damage-matched random sets; start-state matching.
- **Shows - cannot show.** Shows separatrix crossing in the model's own chain. Cannot show where natural hallucinations start.
- **MDE.** Realised SD rule; planning SD of rho about 0.08 per arm.
- **Runs in.** r1b_L_B, 8.5 A100-min: 2,816 rows x 80 tokens = 88 calls x 5.5 s, plus damage matching 0.5.
- **Gates.** SG8; G-twin-X1; G-dmg.

### X2 L4-DOWN: H suppression arms in the dyad

- **What.** C-Hdown (A's P x 0) against C-Rdown (4 damage-matched sets).
- **Rule.** As L4's H line; 'H gain acts as coupling (conformity)' also passes if DD_down upper bound < 0. Holm over up and down.
- **Runs in.** r1b_L_A, 15.3 A100-min.
- **Gates.** As L4.

### X3 TOY-LOOP: the depth loop that Qwen does not have (J1: missing everywhere)

- **What.** Where a time-homogeneous loop exists by construction, does 'loop -> attractor -> Ising-modellable' hold? Do wrong answers sit at wrong fixed points (the EBR / HRM claim, cited by Chen et al. 2026 from Ren and Liu 2026; that source is unverified)?
- **How.**
  - Round-2 W1 one-hot shift world, (x + y + c) mod 17, with the round-2 W0 conventions.
  - Weight-tied looped core: h_(r+1) = tanh(W h_r + U [x; y; task] + b), with input injection at every iteration and readout softmax(V h_R).
  - Iteration count R drawn uniformly from {4, ..., 12} per batch.
  - Width 128, 5 seeds, AdamW as W0, MAXSTEPS 6,000.
  - 4 new shifts learned by column search (B2) supply unknown-task errors.
  - At test, R = 32:
    - (i) converged if ||h_33 - h_32|| / ||h_32|| < 1e-3;
    - (ii) toy H-units: L1 logistic on binarised |h_32| predicting wrong answers (half of the inputs);
    - (iii) push-release across iterations: toy H-units x 2 at iterations 9-12 against 19 size-matched random unit sets; K_toy = Delta(32) / Delta(13), with twins;
    - (iv) pairwise PL on fixed-point spins against 19 random populations.
- **Rule.**
  - 'wrong fixed points exist': >= 20 % of wrong answers come from converged trajectories in >= 4/5 seeds.
  - 'toy H-units form an attractor': K_toy lower bound >= 0.9 and above all 19 random sets in >= 4/5 seeds.
  - 'toy H-population Ising-modellable beyond random': dPL above all 19 random populations in >= 4/5 seeds.
  - Otherwise 'not shown'.
- **Shows - cannot show.** Shows whether the chain holds where depth is a loop. Cannot transfer to Qwen, whose blocks are distinct maps.
- **Runs in.** r1b_L_B, 5 A100-min (or CPU).
- **Gates.** None.

### X4 COMPLETE: associative completion from a partial cue (energy_ising L9a)

- **What.** Is the H-pattern completed from a clamped partial cue (clean-up) more than random populations are?
- **How.**
  - 256 L3b-type keys, first answer token, window 16.
  - A frozen random fraction f in {0.25, 0.5} of P's units gets gain g_j = mean |z_j| in H-high states / overall (L.R fit half).
  - CI = mean live fraction-on of the unclamped units, pushed minus twin.
  - 2 CTRL99 populations are clamped to their own high templates, with a damage-calibrated gain.
  - The model prediction CI_pred comes from forward Gibbs sampling of the L3c static model with the same units clamped.
- **Rule** (Holm over f).
  - 'associative completion (H-specific clean-up)': CI_P - mean CI_R lower bound > 0.
  - 'no completion beyond controls': inside +-0.01.
  - Otherwise 'undecided'.
  - Descriptive: 'pairwise model predicts completion' if CI_obs / CI_pred lies inside [0.5, 2].
- **Requirements.** |P| >= 8 (else H+).
- **Runs in.** r1b_L_B, 2.5 A100-min.
- **Gates.** G-twin; G-dmg; G-cal-Ising.

### X5 L.gv: what a basin is (the only reconstruction; Aletheia-style)

- **Generator.** On 32 neutral prompts (64 tokens, T 0.7), gains toward the target basin's pattern.
- **Verifier.** The unmodified model re-reads the text.
  - 'correct': the state lands in the target basin in >= 60 % of texts, above the 95th percentile of 20 random patterns of equal size and gain.
  - 'minor fix needed': a neighbouring basin (Hamming <= 4), or 40-60 %.
  - Otherwise 'critically flawed'.
- **Reviser.** Gains x 1.25 for a minor fix; prompt family neutral -> QA for a critical flaw. At most 3 attempts, then abstain; abstentions counted.
- **Content.** Frozen lexical statistics only (refusal, capitalised-entity, digit rates; top-20 tokens by log-odds); no judge model.
- **Verifier error rate on known answers.** Planted Ising basins, plus patterns copied from observed states of real texts. The original text must be accepted; texts generated from random patterns must be rejected.
- **Rule.** Descriptive with the measured error rate.
- **Runs in.** r1b_L_B, 3 A100-min.
- **Gates.** G-cal-Ising.

### X6 L4-SOLO: the single-agent loop baseline

- **What.** C-solo: A primed and alone; user turns 'Continue your analysis.'; 6 replies; 256 runs.
- **Statistics.** J_s and K_solo = sigma~(6) / sigma~(1); descriptive 'loop -> agent' baseline.
- **Runs in.** r1b_L_A, 2.6 A100-min.

## L.5 Combined answers (L-ANA; computed from the verdicts above, no new test) and their power

**H-CHAIN, lens-free reading (L-ALL).**
- 'H-attractor not bound to length (lens-free)': the L3 attractor line is 'H-attractor', and L2b is not 'length effect on H is information-limited'.
- 'H-attractor with an information-limited length trend': 'H-attractor', and L2b 'information-limited'.
- 'H-specific amplifier, not self-sustaining' (the user's 'other mechanisms that amplify the effect', not called an attractor because K < 1 is dissipative under R6): the attractor line is one of 'no H-attractor', 'sticky but not self-sustaining' or 'dissipative (observational lines inconclusive)', and the specificity line is 'H-specific carry'.
- 'length artefact (information-limited), no H-attractor': 'no H-attractor', specificity not 'H-specific carry', and L2b 'information-limited'.
- 'no length trend and no H-attractor': as above with L2b 'no length trend of H'.
- 'H rises beyond information without attractor dynamics': as above with L2b 'H rises beyond information loss'.
- Otherwise 'mixed (lines reported)'.

**V-OWN (model or lens).**
- 'the chain-lens break-off and its H-pattern belong to the lens': L5a 'owned by the lens' and L5b 'owned by the lens'.
- 'belong to the model': L5a 'owned by the model' and L5b 'owned by the model'. L5c's verdict is attached as the lens-free error link.
- 'shared': any other combination with at least one positive line.
- Otherwise 'not decidable'.

**V-HOLO (per sense, never merged):**
- code sense: the L1 verdict, the 'not a strict threshold code' flag and the monogamy sign;
- RT sense: the L2a growth class and deficit;
- reconstruction sense: Lambda (L1(iv)) and L2c;
- associative-memory sense: L1(vii) curve and privilege, the L3 attractor line, and X4 if run.

**V-CHAIN (link by link).**

| Link | Positive verdict |
|---|---|
| EBM | L3d 'energy-well entry' |
| Non-equilibrium | L3d 'irreversible H-entry (entry-specific)' |
| Loop | Token and turn loops exist by construction; their dynamics are in L3; depth loop X3 if run |
| Agent | G-duet passes |
| Ising-modellable | L3c 'collective (Ising-modellable) H-population' and L4 'agents Ising-modellable (coupled)' |
| Collaboration | L4 'collaboration (coupling beyond lexical priming)' |
| Entity-forming | L4 'merge into one entity (consensus)' |

'the chain holds in Qwen3-1.7B' only if every link is positive; otherwise 'the chain breaks at <first negative link>'. The H-coupling line is reported beside it.

**Answer template (filled after the run).** 'Lens-free, the break-off pattern belongs to [V-OWN]; the H-neuron attractor is [L-ALL]; the model is holographic only in the [list] sense(s); the chain [holds / breaks at link X]; H gain [acts / does not act] as agent coupling.'

**Round-level confirmatory family CF-L (main §2.4).** The L3 attractor line 'H-attractor (self-sustaining or bistable, H-specific, seen in natural dynamics)' and L2b 'H rises beyond information loss', Holm over the two at 0.05 with the member p-values of main §2.4. L5c is in CF-HERR with K8c. All other L verdicts and combined readings are exploratory at the round level.

**Power of the combined verdicts** (planning simulation sim_combined3.py and sim_own2.py; component statistics drawn with the planning SEs above; rank tests simulated against 99 controls; 20,000 runs per scenario):

| Scenario (planning effects) | Correct label | 'mixed' |
|---|---|---|
| Length artefact: H rises (0.15 SD/e-fold) but explained by iota; K_H 0.5; no H-specific dynamics | 0.68 | 0.27 (false 'amplifier' 0.05) |
| Attractor not bound to length: K_H 1.3, Delta K 0.8, observational effects 3.5 control SD | 0.84 | 0.16 |
| Attractor with an information-limited trend | 0.85 | 0.15 |
| Amplifier: K_H 0.75, Delta K 0.45, observational 2 SD | 0.64 | 0.36 |
| No trend, nothing | 0.68 | 0.27 |
| Bistable within items, K_H 0.6 | 0.69 | 0.24 |
| Weak attractor K_H 1.15 | 0.32 | 0.68 |
| V-OWN lens-owned (G_F 0.10, Omega 0.2, rho_b 0, rho_w 0.12) | 0.82 | 0.18 'shared' |
| V-OWN model-owned (G_F 0, Omega 0.8, rho_b 0.25, rho_w 0) | 0.86 | 0.14 'shared' |

The earlier rule structure (one conjunctive L3 verdict, margin 0.10 for rho_b) gave the correct label in only about half the runs (sim_combined.py), hence the separate lines and the margins at about 2.9 SE.

## L.6 Mapping of every element of the user's hypothesis

| Element (user's words) | Counterpart in L, or 'no counterpart' and why |
|---|---|
| 'H-Neuronen' ('Haarneuronen', speech-to-text slip) | H_op as P: the channel a_t (L2, L3), the Ising population (L3c), the PIP fragment and erasure target (L1), the coupling knob (L4) |
| 'ca. 0,1 % im Netz' | |H| / 172,032 from K. Gao's 0.01-0.35 per mille imply 2-60; '0.1 %' (172) is a loose upper bound. L1 tests whether this set is privileged or one readout of a distributed code |
| 'Chain Lens' | K's lens (SO-CL; a substitute: Qwen3-1.7B has no MTP head). L runs none and classifies its outputs (L5) |
| 'konkretes Abbruchkriterium ... wo sie halluziniert' | L5a: FOREIGN break-off B_F against model-own B_M. Lens-free horizons: k*_free (L2b) and the model's own first error step (L5c) |
| 'Blickbereich ... mit den H-Neuronen korreliert' | L5b: model-frame rho_b against lens-only rho_w. L2c: does the output boundary carry H beyond random channels? |
| 'halluziniert' (two meanings) | H_model (K labels; L5c exact step errors) and H_lens (L5a FOREIGN readouts), kept apart |
| 'ein Artefakt der Länge' | L2b 'length effect on H is information-limited' with natural-position and uncertainty controls; L-ALL 'length artefact' |
| 'H-Neuronen-Attraktor, nicht unmittelbar an die Statistik der Länge gebunden' | L3 attractor line: R6 self-sustainment over 5 turns with H-specific carry, or within-item bistability (X1 capture), plus H-specific stickiness or persistence; combined with L2b in L-ALL |
| 'andere Mechanismen ..., welche ... verstärken' | L3b 'H-specific carry' with K < 1 (snowballing-type amplifier); KV against token route; L4 H gain as coupling |
| 'linsenfreier Ansatz' | All of L (L.0): no hidden state mapped into the vocabulary |
| 'erweitern' | V-OWN (L5) and L-ALL |
| 'holographische Prinzipien' (general) | Four senses, reported separately (V-HOLO) |
| Boundary encodes bulk | Lambda (L1(iv), static) and the cross-map rank test (L2c, dynamic) |
| Ryu-Takayanagi / area law | L2a growth class on equal partitions (only logarithmic = RT-like). No counterpart: bulk metric, minimal surface, central charge; no fixed cut, because the KV cache grows linearly |
| Holographic error-correcting code (threshold, no privileged region) | L1 tree, falsification flag, XH ratio. Quantum aspects (no-cloning, complementary recovery) have no counterpart: classical units can copy |
| Monogamy (I3 <= 0) | L1 bias-corrected I3_c |
| Page curve, scrambling, Hayden-Preskill | No counterpart: forward pass plus sampling is many-to-one and dissipative; only the half-threshold shadow is tested (L1, as not testable beyond 64 units) |
| Holographic memory (Plate) | L1(vii) graceful degradation without causal privilege; L3 attractor (clean-up); X4 completion |
| AdS/CFT bulk, gravity, quantum entanglement, conformal boundary | No counterpart |
| 'vgl. EBM reasoner' | E = -log q (exact ARM-EBM bijection, Blondel et al.): L3d energy-well rule. The 'incorrect fixed point' becomes self-sustaining H (L3b), a descriptive H-basin (L3c), and wrong fixed points in X3. Energy descent at inference has no counterpart: Qwen samples from exp(-E) and never descends it |
| 'non-equilibrium' | L3d irreversibility at entry against a detailed-balance null; asymmetry of kinetic J (L3c); Sagawa sigma (descriptive). No counterpart: heat bath, physical temperature, dissipation |
| 'da Loop Transformer' | Token loop (L2, L3a), turn loop (L3b, X6), dialogue loop (L4). No counterpart in Qwen's depth (28 distinct blocks); X3 weight-tied toy loop |
| '--> Agent' | A Qwen instance with its own context acting in turns (L4, G-duet) |
| '--> Ising modellierbar' | L3c (units: static max-ent with basins, kinetic Ising) and L4 (agents: kinetic fit with h, J_s, J_c) |
| '--> Kollaboration' | L4 coupling beyond lexical priming (live partner against the same messages as documents) |
| '--> entitätenbildend' | L4 consensus under opposite priming beyond within-item re-pairing. Pair-macro Delta Psi and L3c collectivity are descriptive senses |
| '....' (open continuation) | No counterpart: unspecified |
| The arrows of the chain | Measured link by link; an arrow is a causal claim only where L manipulates H gain (L3b, L4) |
| 'Multitoken Prediction ... Gradient Through Routing' | No counterpart in L: no MTP head, no gradients at inference (part K) |
| 'Folge, nicht Reihe' | Folge = the model's own sequential continuation (L2 chains; token route of L3b). Reihe = parallel KV route, measured as the KV share (L3b); a direct k-step readout needs a lens (K) |
| 'vorderste Front ... Reihen von Tokenfronten' | Per-front channels captured during generation (a_t, e_t, s_t, populations) |
| 'verschiedene Kanäle', 'hierarchisch', 'äußere Gegebenheiten' | Hierarchy is part K's (K9). L reports only directed kinetic couplings among units (L3c) and agents (J_c(A <- B) against J_c(B <- A), L4); family and T as external conditions, descriptively |
| 'Durchsatz' (a channel front predicting the next fronts) | Anchor-information profile iota(k) and horizon k*_free (L2b); descriptive cross-map lag profile (L2c) |
| 'vektorielle Größe' (front predicts preceding against following fronts) | A_k (L3d, descriptive): for any stationary process the joint-state ratio is exactly balanced, reversible or not. A non-zero A_k means non-stationarity; irreversibility is measured separately (L3d rule) |
| 'zu überwinden' (probably 'zu überprüfen') | Every L rule has an outcome that counts against the hypothesis (L.7) |
| H-DUAL 'Catfish' (a competitor keeps the group active) | L4 catfish carry (C-opp against C-half), attached to D.17 as a qualifier (main C22) |
| 'Catfishing = Hyperlong-Kontext' | L2a captured MI up to the native 32,768 tokens (descriptive touchpoint; part D) |
| 'zwei Entitäten' / 'aus einer Entität zwei machen' | Part D. L4's 'two entities persist' is attached to D.17 as a qualifier (main C22; it counts against the chain's entity link) |
| 'Selfish Mining', 'unter 50 %', 'Losing Game' | No counterpart in L (part D) |
| 'LLMs und Trainingsdaten als Hyper-Catfishing' | No counterpart: needs training |
| Retokenisation, letters/pixels floor, 'König - Mann + Frau' | No counterpart in L (part D) |
| 'Form war relevant' (round-2 AT8) | Descriptive only: modularity Q of kinetic J next to ||J|| (L3c) |

## L.7 Outcomes that count against the hypothesis

- **Against local storage in the H-neurons.** L1 returns 'H privileged but one of several representations', 'distributed without a privileged set' or 'dense', or carries 'latent-scalar redundancy'.
- **Against holographic storage.**
  - Code sense: 'not a strict threshold code (falsified)' with m_0 <= 64, 'redundant (I3_c > 0)', or 'localised'.
  - Associative-memory sense: 'H causally privileged' without 'graceful degradation', and no attractor line.
  - RT sense: power law or saturating, not logarithmic (descriptive).
  - Reconstruction sense: 'boundary does not encode H' or 'only like any population'.
- **Against model-ownership of the chain-lens pattern.** V-OWN 'belong to the lens'; L5c 'no H signal before own errors'.
- **Against an H-neuron attractor not bound to length.** L-ALL 'length artefact (information-limited), no H-attractor' or 'no length trend and no H-attractor'. 'H-specific amplifier, not self-sustaining' also denies the attractor (K < 1 is snowballing).
- **Against amplifying mechanisms.** 'carry like damage-matched controls' together with 'H dynamics not distinct'; L4 'H gain acts like damage' or 'no effect of H gain on coupling'.
- **Against the EBM reading.** 'uncertainty regime at entry (not a well)' or 'no energy signature at entry'.
- **Against non-equilibrium at entry.** 'reversible within resolution'.
- **Against an Ising-modellable H-population.** 'pairwise structure not H-specific', 'selection effect', 'not Ising-modellable' or 'independent H-units'. Any H-specific verdict qualified 'not beyond uncertainty' denies H-specificity beyond an uncertainty population.
- **Against loop -> agent -> Ising -> collaboration -> entity.** L4 'no detectable inter-agent coupling', 'influence explained by lexical priming (no social coupling)', or 'two entities persist'. X3: 'not shown'.
- **Not evidence either way.** Every 'not testable', 'not decidable' and 'undecided'; Psi <= 0; and anything 'positive must be earned' excludes (L.2).

## L.8 Budget, sessions, trim order

**Rates** (round-2 plan 7.1; int_machinery 2.1 and 3.6; est.; fp32, TF32 off):
- generation 25 ms per decode step at batch 32 (about 30-35 ms at 0.5-1.1k context with DynamicCache rewrites); hooks x 1.15;
- R-all about 5,600 tok/s; R-full about 5,000 tok/s;
- B = 1 prefill: 16k 4.5-6.8 s; 32k 12-21 s;
- FISTA about 4 ms per iteration on 4,000 x 172,032.

The cost table is frozen on L0's measured rates.

**Work per measurement (A100-min).**

| Measurement | Core | Extension |
|---|---|---|
| L0 PILOT | 10 (run in r1b_P0) | |
| L-CAL | 0 (CPU) | |
| L1 LOCUS | 41 | |
| L2 HORIZON | 55.5 | |
| L3 ATTRACT | 51 (a 6, b 34.5, c 8, d 2.5) | |
| L4 DUET | 34.4 | |
| L5 OWNER | 3.4 (3.1 + 0.33 for the 6 extra gating samples) | |
| X1 L3-DOSE | | 8.5 |
| X2 L4-DOWN | | 15.3 |
| X3 TOY-LOOP | | 5 |
| X4 COMPLETE | | 2.5 |
| X5 L.gv | | 3 |
| X6 L4-SOLO | | 2.6 |
| **Sum** | **195.3** | **36.9** |

**Sessions.**

| Session | Account | Content | Work (min) | Planned (4 + 1.5 x work) |
|---|---|---|---|---|
| r1b_L_A | B | Step 1: mu_j, CETT recompute, SEL+ / UNC+ / CTRL99 / hulls, written to L.POP. Then L1, L3b, L4; X2, X6 | core 109.9 + ext 17.9 | core 168.9; with ext 195.7 (3.3 h) |
| r1b_L_B | B | L.R corpus, L2, L3a, L3c, L3d, L5 generation (own CoT, sub-answers, 6 extra gating samples, t* pass); X1, X3, X4, X5 | core 75.4 + ext 19 | core 117.2; with ext 145.7 (2.4 h) |
| L-ANA | A (CPU runtime; a step of r1b_ana) | Verdicts, L5 analyses, L.5 combinations | 0 A100 (about 30 CPU-min; about 0.3 h if run on an A100 VM) | 0 |
| **Total** | | | core 185.3 + ext 36.9 | tier 1 286.1 min (4.77 h; 4.95 h charged with the 10.6 min run in part K's sessions for L); tier 2 55.4 min (0.92 h) |

**Order.**
- r1b_P0 (P0-L items) before the freeze.
- r1b_L_A after r1b_K_S1 (SO-HN, SO-LAB incl. S_L) and r1b_R_S0's exports in r1b_xfer_A (REF-R16, u*_14); SO-D1STIM is frozen in the repository.
- r1b_L_B after r1b_L_A (L.POP, and L3b's twin-0 rows, which L3a reads), SG9 (r1b_D_A's G-LCE-1 / -2 records) and r1b_K_S2a's gating export.
- L-ANA (in r1b_ana) after r1b_L_A, r1b_L_B, r1b_K_S2a and r1b_K_ana (K's G-K2, G-K4, G-K4b, G-K5, G-K14 labels).
- Every session stays below the 5-hour cap, in resumable steps of at most 5 min, with the round's time guard (SG5: stop after the step that ends at >= 4.5 h).
- At most 2 concurrent sessions per account (an assumption).

**Pilot-dependent saving.** If L0 (iv) passes parity, carried caches cut L4 and X2 by about 40 % (about 12 min of work).

**Trim order** (applied before the freeze if the pilot shows rates below planning, or if the architect needs room; never a rule or threshold change; trims 2-5 apply to tier 1 only under main §4.5's proportional rule):
1. X5, X4, X3, X6, X1, X2;
2. L.R-B items 32 -> 24 per family;
3. L2a natural documents 24 -> 16;
4. L3b random sets 6 -> 4 per sign (label 'few sets');
5. L1-C to 2 seeds per profile (gate 2/2).

## L.9 Gates

| Gate | Definition | Effect of failure |
|---|---|---|
| SG8 (main §3.1) | K's G-K2 on H_op (r1b_K_S1's preview for what runs; r1b_K_ana's value for labels), with H_op in {H_tq, H+} | L1 runs with the best sparse set, labelled 'no validated H-set'. H-specific verdicts in L2b-L5 read 'not testable (no H-set)'. L2a, the L2b descriptives, L4's coupling, collaboration and entity lines, and L5a still run |
| G-sel | SEL label with |phi| < 0.1 and >= 300 per class (UNC by construction) | Verdicts that need SEL+ are capped at 'collective / specific against random populations (selection effect not excluded)' |
| G-cal-L1..L5, G-cal-Ising | L-CAL items; L1-C (3/3 seeds per profile); planted Ising >= 4/5 | Dependent branch 'not testable (pipeline calibration failed)' |
| SG7 (L's part) | L0 rates >= half of planning | Trim order applied before the freeze |
| SG9 (main §3.1) | Memory-efficient kernel asserted; r1b_D_A's G-LCE-1 / -2 records present | r1b_L_B waits |
| G-L2a-LCE | Cached = recomputed (< 1e-4 relative) at 16k and 32k; memory within 40 GB | L2a halted (subject to OD-M14) |
| G-MI | Copy and shuffle controls | L2a 'not decidable (estimator check failed)' |
| G-twin | Sham >= 31/32 identical; X1 twins reproduce chains in >= 99 % | 'not decidable (common random numbers broken)' |
| G-manip, G-push | In-window displacement >= 3 SD of sham; d^(1) interval excludes 0 | 'not testable (push too weak)' / 'K not defined (push too weak)' |
| G-dmg | Every random control within +-10 % of the H intervention's damage after replacement or fallback | That contrast 'not testable (no damage-matched control)' |
| G-ESS, G-ISM | Ising sample adequacy; model check | Fallback to 30 units, then 'not testable (sample)'; 'not Ising-modellable' |
| G-sigma | Integral-fluctuation-theorem pipeline check | Sagawa sigma 'descriptive (sigma pipeline check failed)' |
| G-duet | Format and readout | L4 'not testable (format / readout)' |
| G-duet-manip | H gain changes A's own first-message commitment beyond damage-matched random gain | H line 'not testable (H gain has no specific effect on agent A)' |
| G-CL | SO-LENSOUT per-chain records with >= 150 known + 150 prone items | L5a, L5b 'not testable (no exact-label lens corpus from K)' |
| G-parse | >= 70 % of CoTs parse | L5c 'not testable' |
| Round-1/2 loop gates (PC_PASS, GO3) | Not used: L runs no steering loop | None; SO-AX's u*_14 is descriptive only |

## L.10 Reuse and new code

**Reused unchanged.** rk_loop keyseed / Gumbel (SO-KEYS's verified copy in r1b_common); r2_theta.stage_turn2; r2_theta_verdict.rule_R6 logic; r2_rk.prompt_bootstrap; the CbarHook pattern; r2_unlearn_qwen.calibrate and RefKL ('hooks'); r2_linearity.auc_cols; neu_extract / neu_sparsity hook patterns (code only).

**From other parts by name.** SO-HN (hook manager, selector, random-set generator), SO-LCE, SO-DMG (r1b_core's function-style damage and calibrate).

**New.** One module r1b_L.py and an analysis module r1b_L_verdict.py, each with a MOCK test and an L-CAL code test:
- L.POP builder (SEL+ / UNC+ / hulls);
- L.R capture (populations, spins, e, s);
- L.PIP (fragment families, rotation hook, floors, reselection, planted codes);
- L.EQMI with G-MI;
- L.TWIN (multi-turn, draw-and-discard Gumbel advance for X1);
- L.MARKOVSUR plus the detailed-balance surrogate;
- RQA trapping time;
- d-hat;
- the cross-map;
- L.KISING (PL, K-pairwise, Gibbs, iterative scaling, descent, modularity);
- L.DUET (re-prefill, context-free stance reads, J_ex with within-item re-pairing);
- L.OWN;
- the L5c parser and conditional hazard;
- X3 toy.

Also new: r1b_L_cal.py (L-CAL generators).

## L.11 Planning simulations behind the MDEs (scratch, CPU; not data)

- **scratch_L_lean/:**
  - sim_pip.py, sim_pip2.py: PIP and reselection profiles. An uninformative code gives I-hat -0.004 bits at m = 8 and -0.17 at m = 1,024, hence floors at every size.
  - sim_sticky.py, sim_sticky_het.py: frailty.
  - sim_agents.py.
- **judge_L1/sim_item_confound.py:** across-item shuffling gives false collaboration at a rate of 1.00 with item fields; within-item 0.07.
- **judge2_L:** a re-run of sim_agents gave null size 0.10, not 0.03.
- **scratch_L_merge/:**
  - sim_duet.py: item fixed effects alone are insufficient.
  - sim_duet2.py: J_ex is unbiased under item fields, SE about 0.04, attenuated about 0.7x. The rank against within-item re-pairing is miscalibrated (0.08-0.25 at J_c = 0), hence the bootstrap rule.
  - sim_combined*.py, sim_own2.py: the L.5 power table.
- **scratch_L_holographic/:** sim_psi_pair.py, sim_psi_sum.py: Delta Psi separates coupling from persistence.
- **scratch_L_energy_ising/:** checks_L.py: the leading eigenvector of a positive J splits nothing, while the modularity matrix recovers planted groups; Holm attainability with 99 controls.

## L.12 Limits; not done and why

- **Captured, not true, information.** L2's MI is what Qwen3-1.7B's own log-probabilities capture (vCLUB form).
- **Readable is not used.** L1 locates readable information; causal use is K's alpha-scaling plus L1(vii) and L3b.
- **Answer-token construct.** L1 inherits K's answer-token CETT; L1(v) and L5b use pre-answer reads.
- **Lens-free is not assumption-free.** Probes, CETT, thresholds and bins are modelling choices, frozen here and calibrated in L-CAL and L1-C.
- **Agents are instances.** L4 says nothing about neurons inside one pass. N = 2 gives no phase transition.
- **Cross-device common random numbers.** All paired contrasts are computed within one session on one GPU.
- **Not done:**
  - criticality and specific-heat scans (static fits 'may not be able to discriminate' critical from supercritical; Simoes et al. 2026; budget);
  - depth clusters and long-context two-community splits (H-DUAL; part D);
  - 24-agent naming games (about 4.5 A100-h, prefill-bound);
  - per-pair CCM direction claims and the full channel hierarchy (part K);
  - IIT Phi (no supervenience-respecting estimator at N >= 20);
  - pattern completion beyond X4;
  - contexts above 32,768 positions; bf16; any training of Qwen.

## L.13 Provenance (summary; details in the provenance field)

- Base lean, as both judges recommend. Grafts from holographic and energy_ising as listed.
- Every judge fatal flaw that touches a retained element is fixed. Every 'missing everywhere' item is added: core where it is a validity fix, extension where it deepens a link.
- Planning simulations for the merge are in scratch_L_merge/.
