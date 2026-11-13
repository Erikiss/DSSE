# Part D (round 1b, merged): H-DUAL, duality under hyper-long context

Merged from the lean (base), dynamics and confounds designs after two judge reports. Nothing here has been run.
Numbers marked (sim) come from numpy-only scripts in
/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/merge_D/ (sim_strata_hedge.py,
sim_countshare.py, sim_split_v.py, sim_split_conj.py, sim_combined_power.py, cost_final.py); no model was run.

## D.0 Summary

- **Base.** Both judges ranked the lean design first. Part D keeps its instruments and gates, its minority-capture surface
  with front padding and the endpoint-normalised statistic Phi, its two-speaker catfish test with common random numbers and
  NEU/ALI partners, its realized-MDE rule and its input-free sessions. Every fatal flaw and every item the judges found missing
  in all three designs is fixed (provenance in D.22).
- **H-DUAL is tested in three parts, each able to come out negative:**
  - **(a) Length-driven minority capture.** A version holding a minority share alpha_B < 0.5 of the in-context reports pulls a
    forced choice by more than its share, and more so as total context grows, with every report at a fixed distance from the
    question. Primary: **D1-a**. It must survive three checks: a prior-opposed stratum (prior x saturation artifact), a
    third-value control (conflict hedging) and shuffled filler (filler form).
  - **(b1) A force splits one entity into two under hyper-long context.** From an instruction-stripped symmetric "torn" start,
    seeds of one item commit to two states, more often at 28,672 tokens than at short context. Primary: **D3**. The rule needs
    the split in two readouts (in-context probe and in-context state); a probe alone fires on 63-87 % of items under a
    saturated readout of one diffusing entity (sim).
  - **(b2) The two entities keep each other going, enabled by hyper-long context.** Either **D2** (an opposed model-written
    speaker sustains a stance more than a neutral one at 32K, more than at short context, without merging) or **D5-a** (two
    competing voices keep a self-generated transcript coherent longer than two agreeing voices). Bonferroni x2 inside (b2).
- **Combined verdict (D.17).** Holm over (a), (b1), (b2) at familywise 0.05. Under a realistic null no part is called
  supported (0 of 4,000 simulated runs); a decisive 'not supported' is reached with probability 0.32-0.57, otherwise 'not
  decidable' (sim).
- **Grafted tests of the remaining metaphor elements:**
  - catfishing proper, a presented role beside a retained assistant identity (D4);
  - the losing game against a non-competing second entity (D1-d);
  - share vs count with saturation-robust contrasts (D1-c);
  - form vs size (D1-f);
  - the prior part of gamma (D1-e);
  - retokenisation (D1-r) and the user's analogy aside, symmetric vs asymmetric relations under retokenisation (D1-v);
  - a Qwen3-0.6B size comparison (D1-s);
  - competitor dominance (D2-g; D5 2:6 arms);
  - the rate form of hash share (D5-r);
  - two voices continuing their own text toward the 32K window (D5-x).
- **Budget.** 341 raw A100-min; with the x1.5 margin, four setups and a 10-min pilot share, **9.03 A100-h** central.
  Scenarios: 9.47 h (attention 8 TFLOP/s), 8.14 h (20 TFLOP/s), 7.49 h (non-rewriting decode cache). A trim order frozen on
  pilot engineering numbers (D.19) keeps the projection at or below 9.0 h; three reserve items run only when funded.
- **Expectation stated now** (literature and round 2; not a rule):
  - D1-a 'no length-driven minority capture' or 'within proportional'; D1-b release threshold below 1/2 at every length
    (recency); D1-c count-driven; D1-d a small cost;
  - D3 'no split at any length' or urn-like;
  - D2-a 'competitor dilutes', with D2-c merging (Li et al. 2024); D5-a no lifetime advantage; D4 'absorbed' or 'blended' at
    1.7B (persona literature: such dualities grow with scale).
  An outcome against these expectations is what would support H-DUAL.

## D.1 The hypothesis, what is testable, and the mapping

**The user's H-DUAL** (shared brief):
- hyper-long context ('catfishing', 'selfish mining': the context held far above its natural length) makes a force arise
  that splits one entity into two;
- the two entities then sustain still longer context (catfish effect: a competitor keeps the group active);
- a minority below 50 % can win; the whole is a losing game, but it enables the duality;
- aside: the duality should not live inside one token (retokenisation down to letters would remove it); symmetric (king -
  man + woman = queen) versus deliberately asymmetric analogies, and what retokenisation does to them.

**Two verified facts bound the test** (lit:dual):
- The Eyal-Sirer threshold (1 - gamma)/(3 - 2 gamma) is stationary: 1/3 at gamma 0, 1/4 at gamma 1/2, 0 at gamma 1. Nothing
  in it falls with chain length, so 'alpha* falls with L' is the user's own prediction.
- Inside one frozen forward pass, withholding and the difficulty adjustment have no counterpart. Recency, primacy and prior
  bias already let minorities capture continuations in short contexts (Zhao et al. 2021: 'P P P N' gives about 90 %
  Negative; Xie et al. 2024: a prior-aligned 1/3 share gets more than its share in 10 of 16 cells). Part D uses these only as
  positive controls. The new content is (a) dependence on total length beyond position and degradation, and (b) two entities
  that arise, coexist and keep each other going.

**Mapping of every element**

| Element (user's terms) | Counterpart in part D | Where |
|---|---|---|
| Entity | D1: one version of a referent's six attribute values (a consistent set of reports). D2: a named speaker with a stance. D3: a bundle of three invented attributes the narrator holds. D4: a presented role or the retained assistant. D5: a voice in a transcript. All read by forced-choice log-odds of the unmodified model, D3 also by the in-context state. | D1-D5 |
| Hyper-long context, 'held far above its natural measure' | Total length by front padding with neutral text, all entity material and its distances to the readout fixed. Natural measure = 4,096 (Qwen3 pretraining stages S1/S2); hyper-long = 16K-32K (long-context stage, native 32,768). D1 L0 about 1,000 vs 32,000; D2 about 300 vs 32K; D3 0 vs 28,672; D4 0 vs 16,384 | D1-D4 |
| Beyond 32,768 positions (up to max_position_embeddings 40,960) | No counterpart tested: untrained window, effects confounded with RoPE extrapolation; YaRN not validated for 1.7B. No position beyond 32,100 is used | not run |
| 'A force arises / it is forced that the whole splits into two' | Forced: the instructed two-speaker condition (D2 OPP). Arises: seeds of one torn start committing to two states, more often at 28,672 than at 0 tokens (D3, both readouts) | D2, D3 |
| 'Two entities from one makes sense, insofar as catfishing is active' | D2-a/D2-b: one model voicing two speakers; persistence with an opposed vs a neutral partner, at 32K vs short | D2 |
| Catfishing (one agent presenting a second, fictitious identity; the 2010 film) | D4: a presented role and the retained AI-assistant identity read at once (role-fact probes, out-of-character identity probe): coexisting, blended, absorbed, split across seeds | D4 |
| Catfish effect (a competitor keeps the others active) | D2-a: persistence Pi of a speaker's stance with an opposed (OPP) vs a neutral (NEU) partner; ALI and MON separate competition from engagement and from no partner. D5-a: coherence lifetime of two competing vs two agreeing voices | D2, D5 |
| Effort, motivation, being seen (the catfish story; Mas and Moretti) | No counterpart: a model has no effort; only persistence and coherence are measured | - |
| Competitor strength (Lackner 2023; Brown 2011: a dominant superstar lowers rivals' effort) | D2-g: a dominant opposed speaker (DOM) vs a parity opponent; D5: a 6:2 majority-voice opponent vs 4:4 parity | D2, D5 |
| The two entities then sustain longer context | D2-b: length x condition interaction of the catfish contrast; D5-a: lifetime to degeneration in a 4,096-token self-generated transcript; D5-x: two voices continuing their own text toward 28,672 tokens | D2, D5 |
| Selfish mining: withheld private chain released to override | Release arrangement: the minority's reports placed last as one run vs position-balanced interleaving | D1-b |
| Withholding inside one pass | No counterpart: causal attention sees every earlier token; only the experimenter's release timing exists | D1-b |
| Hash share alpha (a production rate) | Stock: share alpha_B = k/10 of the in-context reports (D1). Rate: the share of generated tokens written by a minority voice in self-continuation (D5-r) | D1, D5-r |
| Share (selfish mining) vs count (Bayesian, many-shot) | Equal-share blocks with different count differences, and equal-count-difference blocks with different shares | D1-c |
| Optimum below 50 % (minority capture) | Phi > max(0.25, envelope): more-than-proportional minority influence in a position-balanced arrangement; release threshold alpha*_rel < 1/2 | D1-a, D1-b |
| 'alpha* falls with L' (the user's prediction) | Delta alpha*_rel between L* and L0 at fixed positions (literal; acts through position weighting); Delta Phi (beyond position) | D1-b, D1-a |
| gamma (tie-race share) | Recency part: gamma_rec = sigma(l_rel at parity, B released last). Prior part: gamma_prior from parity log-odds in prior-asymmetric items, and the per-item prior b from neutral-note contexts | D1-b, D1-e, D1-a |
| Longest-chain rule | No counterpart as a rule (continuations are soft probabilities); its soft trace is the share-response curve | D1 |
| Orphaned honest work; slower main-chain growth (r_pool + r_others < 1): 'losing game' | System cost of a competing parity context minus a non-competing two-referent context of the same format, length and positions: reference-passage NLL, context KL against a neutral context, factual margin | D1-d |
| Attacker's absolute loss before the difficulty adjustment; the adjustment itself | No counterpart: an in-context entity has no revenue; softmax renormalises instantly; retraining on own output is excluded | - |
| Pools growing towards a majority (Eyal-Sirer Obs. 2) | No counterpart: no strategic agents in one context | - |
| 'But it enables this duality' | Joint descriptive reading: D1-d cost together with D2-c coexistence or a D3 split at the long length; no single statistic | D.17 |
| 'Form was relevant' (round-2 toy AT8: shape, not size) | Form of the long context at fixed size (coherent vs sentence-shuffled filler) and form of the material (release vs interleaved) against size (share) | D1-f, D1 |
| LLMs and their training data as hyper-catfishing | No counterpart in a frozen model (needs training). Proxy only: the trained prior as a hidden participant (gamma_prior; reserve R1 tests the in-context persona against the trained default) | D1-e, R1 |
| Duality not inside one token; retokenisation removes a token-level duality | Capture pattern under random valid segmentation and character-level encoding of the value strings, each normalised by its own endpoints | D1-r |
| Tokenisation floor (letters; pixels) | Letters = character-level token ids of the byte-level BPE; pixels: no counterpart (text-only model) | D1-r, D1-v |
| Floor reached at different speeds, depending on how the connections lie | Retention under segmentation and character encoding, per attribute class (D1-r) and per relation family (D1-v) | D1-r, D1-v |
| Symmetric analogy (king - man + woman = queen) vs asymmetric; what happens under retokenisation | Vector-offset analogy accuracy in both directions for a symmetric family (gender pairs) and two directional families (country-capital, part-whole), canonical vs segmented vs character-level, in Qwen's own states | D1-v |
| Model size (persona and many-shot effects grow with size) | Qwen3-0.6B repeat of the D1-a core; a two-point size comparison among small models | D1-s |
| Two readings of the analogy: M1 two in-context entities; M2 long context as withheld chain against the trained default | M1: D1-D5. M2: reserve R1 (fixed persona block vs the default, filler-grown) | D1-D5, R1 |
| Energy / Ising reading (two anti-aligned communities; catfish = net cyclic flux; lf_energy_ising, UNVERIFIED) | Provisional descriptive only: net cyclic flux of the (sign D_1, sign D_2) macrostates over D2 turns and of D5 voice stances over checkpoints, against a within-run shuffle floor. No verdict uses it | D2, D5 |

## D.2 Shared objects (for the merge across parts)

- **LCE: long-context engine** (new; shared with part K's long-context arms).
  - B = 1 prefill of a shared prefix, once per length and filler stream.
  - Chunked extension over a cached prefix with `logits_to_keep` on the readout positions only.
  - Two read paths:
    - crop path: unstopped chunk, then all layers cropped (no stopped reads anywhere in part D);
    - copy path: a device copy of the prefix (+ block) cache per question, freed after the read (no crop).
  - Branching by cache copies (batch_repeat_interleave): B <= 3 rows at 28-32K, <= 7 at 16K, fp32 on 40 GB.
  - Equal-length rows only at long context (fixed-length answers, length-matched texts), so decode steps need no padding mask
    and use native GQA. Left padding is used only below 6K tokens.
  - Keyed Gumbel-max decode with keyseed and Gumbel copied verbatim from rk_loop (sha256 of the source lines checked; rk_loop
    is not imported, because it loads the model and registers hooks at import). Turn index in every key.
  - Stop tokens masked before the fixed answer length; the driver closes the turn with <|im_end|>.
  - Generation-time capture of block means over each answer (exact: no push anywhere in part D).
  - SDPA memory-efficient backend asserted (a math fallback raises); manifest logs peak memory, backend, GPU, torch,
    transformers, code, configuration and stimulus hashes.
  - Parity gates G-LCE-1..4 and the fallbacks FB-1, FB-2 (D.4).
  - Optional non-rewriting cache: used only if the pilot shows parity with DynamicCache on the readouts (packing only).
- **Pilot** (shared with the global round-1b pilot; engineering numbers only, no log-odds stored).
  - Measures prefill, chunk and decode rates at 4K/16K/28K/32K for B in {1, 2, 3, 4} (B = 4 at 32K as a memory probe),
    DynamicCache vs non-rewriting cache, crop vs copy path, peak memory, SDPA backend and Drive throughput.
  - It freezes the cost table and decides the trims (D.19).
- **Generic filler corpus** (shared with part K's lens corpus if K pins one).
  - Source: WikiText-103 raw, dataset revision and sha256 pinned in the freeze commit (CC BY-SA 3.0).
  - Passages dropped if they contain any D1/D3 value word, any referent or role name, or 'AI', 'artificial intelligence',
    'machine learning', 'language model', 'neural network', 'self-improv'.
  - Cut into 16 streams of at least 32,000 tokens (pinned tokenizer).
  - Sentence-shuffled versions of streams 0-7 (keyseed('r1bshuf', s)); the last 150 tokens stay unshuffled.
  - The text adjacent to the entity material is identical across lengths (front padding).
  - Each part processes its units grouped by stream (one prefill per stream and length; no master caches in host RAM).
- **Round-2 turn machinery** (reused unchanged):
  - r2_stimuli.TURN2_SUFFIX (16 tokens), turn2_ids, check_turn2, round_trip_stable (OD-B13 concatenation; history think
    blocks kept, 4 off-template tokens per history turn, reported);
  - the B-Q8 read convention ('read as the reply to the plain prompt');
  - the R6 rule logic of r2_theta_verdict (K defined only if the denominator's 90 % interval excludes 0) for D2-f.
- **Axes** (read-only, analysis only, account A): u_T_perp(14) = unit(R1.state[14]['axes']['u_T_perp']); u*_14 = the round-2
  B-Q0 mode rebuilt from R1.state[14]['dir_<L>@1.5_4'] with TV blocks.14.B_Q0.mode_set members (RSI_a, RSI_b, SEA, SPO,
  RND_1) and signs (+1, +1, -1, -1, -1). Shared with part R.
- **PROBE: letter-format forced choice** (from the dynamics design).
  - Question '... (a) {x} (b) {y} Reply with the letter.' with the assistant turn teacher-forced to 'Answer: ('.
  - Readout l = log p(letter of B) - log p(letter of A) at the next position ('a' and 'b' single tokens, asserted).
  - Committed mass M = p('a') + p('b').
  - Option orders balanced within every pair of contexts.
- **Yes/no stance probe** (round-2 O5 D convention, rk YES/NO first-token ids) with for/against claims and two frames; mass
  p(Yes) + p(No) logged.
- **INJ: instruction-stripped starts** with generate-verify-revise (from the dynamics design).
- **SPLIT-RULE**: per-item two-readout mixture rule with its calibrated false-positive table and planted positive control.
- **u_AA**: assistant-axis analogue (dynamics), used by D4.
- **Statistics**:
  - two-level bootstrap (filler streams, then units within streams), generalising r2_rk.prompt_bootstrap;
  - the realized-MDE rule;
  - the survival rule ('survives' / 'inconclusive on' / 'explained by'; from the confounds design);
  - rule-validation gates on planted data (from the dynamics design).
- **Not used by part D**:
  - damage-normalised push and norm-preserving control (part R): part D steers nothing, so round-1 PC_PASS, GO1-GO3 and the
    damage gates do not apply;
  - lenses (J-lens, J++, MicroLens++, chain lens): part D is lens-free;
  - the n-turn K driver (part R owns the RSI-stance carry chain);
  - the H-neuron set (part K) enters no verdict.
- **Optional**: if part K's gate G-H passes before r1b_D_A starts, part D reports H-neuron CETT occupancy at the D1 answer
  position at L* (descriptive covariate).

## D.3 Conventions

- **Model.** Qwen3-1.7B at the pinned revision (70d244cc..., checksums.json); chat template, thinking off (empty think block);
  float32 weights and compute; TF32 off for cuBLAS (the memory-efficient attention kernel's 3xTF32 'FastF32' math is not
  governed by the flag; stated). Qwen3-0.6B only in D1-s (pinned sha256 as round 2). No training, no push, no hook that
  changes a forward.
- **Lengths.** No position beyond 32,100. Natural length 4,096; hyper-long 16K-32K. L* (D1) is chosen in session by the gates
  from pure and neutral contexts only (never from minority contexts).
- **Sampling** (D2-D5). Gumbel-max over the full vocabulary at T 0.7 (rounds 1/2). Fixed answer lengths: 64 tokens (D2), 40
  (D3, D4), 256-token turns (D5). Deviation from round 2, needed for equal-length rows without padding masks at long context.
- **Notation.** alpha_B = k/10 is the share of B reports, unrelated to the push size alpha_b of rounds 1/2. K keeps its round-2
  meaning. l always denotes log-odds in nats, oriented towards B (D1) or towards the pro stance (D2, D5); analysis clip +-15
  nats (sensitivity +-10 and unclipped reported).
- **Units and intervals** (90 % percentile intervals, 2,000 replicates, ratios per replicate):
  - D1: items; item i uses filler stream i mod 16. Two-level bootstrap: streams, then items within streams, seed 11. All
    contexts of an item (every length, k, swap, order, arrangement, encoding) are resampled together, so every contrast is
    paired.
  - D2: prompts; prompt p uses stream p mod 16. Two-level bootstrap, seed 12; all conditions, turns and lengths of a prompt
    together.
  - D3, D4: items or roles; each has its own stream, so the bootstrap is over items (seed 13). Exact binomial and McNemar
    tests for item counts.
  - D5: starts (no filler), seed 14.
- **Multiplicity.**
  - Each measurement's verdicts use 90 % intervals, as round 2.
  - The four primaries (D1-a, D3 across lengths, D2-a with D2-b, D5-a) answer different sub-hypotheses and are not adjusted
    within their own strings.
  - The combined verdict (D.17) uses Holm over the three parts at familywise 0.05 (one-sided p-values), with Bonferroni x2
    inside part (b2).
  - Every other rule is labelled 'secondary (not multiplicity-adjusted)' and cannot change the combined string.
- **MDE rule.**
  - Each rule states a planning SD and an MDE (2.49 SD: one-sided 5 %, power 0.8).
  - Where no round-1/2 noise level exists, the planning SDs come from the designers' simulations (scratch_D_lean,
    scratch_D_dynamics, scratch_D_confounds), inflated for unmodelled readout noise.
  - Every 'no effect' verdict needs the realized SD at or below the stated ceiling; otherwise it reads 'undecided
    (underpowered: realized MDE m)'.
  - A primary null whose realized MDE exceeds twice its equivalence margin reads 'not decidable (underpowered)'.
- **Survival rule** (for a supported signature theta > 0 checked against a control k with estimate theta_k):
  - 'survives k': theta_k >= 0.5 theta and the 90 % lower bound of theta_k > 0;
  - 'explained by k': theta_k < 0.5 theta and the 90 % interval of theta - theta_k excludes 0;
  - otherwise 'inconclusive on k'.
  - Final string: 'supported' (all survive), 'supported (inconclusive on ...)', or 'not supported (explained by ...)'.
- **Verdict precedence.** Each rule is read top-down; the first matching string applies. In-session decisions (L*, fallbacks,
  reserve triggers) follow round-2 OD-G8: the in-session value governs what ran; the analysis recomputes it and reports any
  disagreement.
- **Keys.**
  - D1: orders np.random.default_rng(keyseed('r1bD1', family, item, swap, k, arrangement)); segmentation keyseed('r1bD1r',
    value); D1-v keyseed('r1bD1v', word).
  - D2: keyseed('r1bD2', turn, prompt), shared by all conditions and both lengths of a prompt.
  - D3: keyseed('r1bD3', item, seed, n), shared by both lengths and by the A2/B2 control seeds.
  - D4: keyseed('r1bD4', role, seed, n); anchors keyseed('r1bD4a', role, kind, seed).
  - D5: keyseed('r1bD5', start, turn), shared by all arms of a start.
  - Starts and pools: keyseed('r1bINJ', family, item, start, slot, attempt), keyseed('r1bD_pool', pole, p, s).
  - Shared keys across conditions are deliberate common random numbers, stated as the exception to 'no key reused'. No
    round-1/2 key is reused.
- **No pre-freeze model output.** All stimuli come from frozen scripts whose outputs' sha256 enter the plan (d1_stimuli.json,
  d2_prompts.json, d3_items.json, d4_roles.json, d1v_pairs.json, filler streams). Instruction-stripped starts and pools are
  generated in session under frozen keys and verification rules.
- **Inverse-free rule.**
  - Every readout is a forward log-probability or a forward state statistic.
  - The fitted readout model in G-D1-null, the coexistence index, u_AB and u_AA are forward-map regressions or differences of
    means (allowed; stated).
  - D1-v compares offset vectors with candidate words' forward states (no inversion).
  - The only reconstruction (naming an unexpected D3 mode) uses generate-verify-revise (D.14).
  - No matrix inverse, pseudo-inverse, inverse Jacobian, Tikhonov inverse or inverse of a link function is used anywhere.

## D.4 D0: instruments and gates

- **What.** Do the engine, the forced-choice readouts and the two-entity readouts work, and how far does single-entity recall
  degrade with length at fixed positions? Without these, no D verdict is read.
- **How.**
  - **G-LCE-1 (D1 cached-path parity).**
    - 8 F-sym contexts (items 0-7; k in {0, 2, 3, 5}; interleaved order 1) at 16,384 and 32,000 tokens, plus 2 contexts at
      32,000 for Qwen3-0.6B.
    - For one attribute per context (both letter assignments), the crop path and the copy path are compared with a B = 1 full
      prefill of the same ids: 16 full prefills per length.
  - **G-LCE-2 (backend).** Every forward above 2K tokens runs under sdpa_kernel([EFFICIENT_ATTENTION]); the manifest records
    peak memory, backend, GPU and versions.
  - **G-LCE-3 (D2 decode parity).** 4 rows (OPP, prompts 0-3, turn 1) at L_long. The batched B = 3 path is compared with a
    B = 1 full-recompute decode under the same keys on the first 40 new tokens; the same row is decoded twice in the batched
    path.
  - **G-LCE-4 (generation-graft parity).** 4 trajectories each of D3 at 28,672 and D4 at 16,384. Probe log-odds on the
    cached path vs a fresh prefill of the same ids, and 40 decode steps vs a B = 1 recompute.
  - **G-M (letter readout committed).** Median committed mass M over all D1 contexts at each length. G-M2: median p(Yes) +
    p(No) over all D2 probes at each length.
  - **G-D1-PC (from D1 data at L0).**
    - (i) Share response: lower bound of l(4) - l(1) > 0.
    - (ii) Readable versions: endpoint range l(10) - l(0) >= 4 nats (point).
    - (iii) Recency, the known short-context minority capture: lower bound of l_rel(2) - l_int(2) > 0.
  - **G-deg(L) (single-entity recall at the same positions).** rho_L = [l(10, L) - l(0, L)] / [l(10, L0) - l(0, L0)] from
    the pure contexts at L in {4,096, 16,384, 32,000}.
  - **G-D1-null (null envelope and rule validation, from the dynamics design).**
    - (i) Fit. Per item-attribute and length, a forward readout model l = c_L tanh((beta_i E + s b_i)/c_L) is fitted by least
      squares to five measured contexts: the neutral-note context X_N (E = 0), the two non-competing (0,5,5) contexts
      (E = -5) and the two pure contexts (E = -10); s = +-1 by swap; c_L is shared per length. This is the per-length
      readout response function: a forward regression.
    - (ii) Envelope. The main design is simulated forward under this null (no capture mechanism; kernel weights w_j on the
      ten slots with recency tau in {0.7, 1.5, 3, infinite} x primacy in {0, 0.5}; the actual placements, orders and swaps).
      env(L) = the 95th percentile, over 2,000 refits on bootstrap resamples, of the maximum over kernels of the null Phi;
      env_opp(L) is the same for the prior-opposed stratum.
    - (iii) Validation. 200 planted-null data sets are generated from the fitted null with the measured residual noise
      (context-level residuals, item-level random effects); 200 planted-effect sets add a 1.5-nat minority bonus in mixed
      contexts at L* only.
  - **G-D2-PC (from D2 data, per length, turn 1)** (lean):
    - (i) own stances induced: lower bounds > 0 for e_11^c(1), c in {OPP, NEU} (and ALI, MON, DOM on prompts 0-63), and for
      e_22^c(1), c in {OPP, ALI, DOM};
    - (ii) speakers separated (prompts 0-63): |c_12(1)| <= 0.3 e_11^2x2(1) and |c_21^2x2(1)| <= 0.3 e_22^2x2(1);
    - (iii) neutral role followed: |c_21^NEU(1)| <= 0.3 e_11^NEU(1).
  - **G-D2-null.** 200 planted-null replicates of D2's per-text stance chains (the measured turn-1 effects, per-turn carry
    and noise, no OPP-NEU difference), and 200 planted-effect replicates (Pi_1^OPP + 0.2).
  - D3, D4 and D5 gates are in their sections and in D.18.
- **Rule.**
  - **G-LCE-1** passes for a path at a length if max |Delta l| <= 0.02 nats and median <= 0.002 nats over the 16 readouts.
    - Crop path passes: it is used.
    - Crop path fails, copy path passes: the copy path is used ('FB-1'; about +5 % cost).
    - Both fail at 32,000 but one passes at 16,384: L* = 16,384.
    - Both fail at both lengths: **FB-2** (full recompute at 16,384) runs. Items 0-31; one attribute per context
      (j = (item + k + swap + order) mod 6); one letter assignment per context (flipped in the reverse order); interleaved
      k = 0..4, two swaps, order and reverse: 640 prefills, about 59 raw min, funded by trims T1-T6 (D.19). D1-a then carries
      '(engine fallback FB-2: 16K full recompute, 32 items, one attribute per context)', with the MDE re-reported. The other
      D1 arms at long length read 'not testable (engine parity)'.
  - **G-LCE-2.** A math fallback raises and the step fails.
  - **G-LCE-3** passes if the paths are token-identical in >= 3 of 4 rows and the repeats in 4 of 4. Otherwise every D2
    verdict at L_long reads 'not testable (engine)'.
  - **G-LCE-4** passes if probe log-odds agree within 0.02 nats and decode is token-identical in >= 3 of 4 trajectories.
    Otherwise D3 or D4 at that length reads 'not testable (engine)'.
  - **G-M** passes at a length if the median M >= 0.9. Otherwise every D1 verdict at that length reads 'not testable (letter
    readout not committed at L)'. **G-M2** at 0.8, likewise for D2.
  - **G-D1-PC.** If (i) or (ii) fails, every D1 verdict reads 'not testable (readout insensitive)'. If (iii) fails, D1-b reads
    'not testable (recency not detected)'.
  - **G-deg.** L* = the largest L in {32,000, 16,384} that passes G-LCE-1 (either path), G-M and rho_L >= 0.5 (point). With
    no such L, D1-a reads 'not decidable (no gated long length)'. The 16,384 main arm runs only if 32,000 is gated out
    (in-session rule).
  - **G-D1-null.** Passes if verdict 4's condition of D1-a (lower bound of Phi(L*) above max(0.25, env) and of Delta Phi
    above 0, on the main arm) holds in <= 5 % of planted-null replicates. Otherwise D1-a reads 'not decidable (rule not
    specific at the measured noise)'. The planted-effect rate is reported as realized power.
  - **G-D2-PC.** If (i) or (ii) fails at a length, every D2 verdict at that length reads 'not testable (two-entity readout
    failed at L)'. If (iii) fails, D2-c's NEU readings carry 'neutral role not followed'.
  - **G-D2-null.** Passes if D2-a's verdict 4 occurs in <= 10 % of planted nulls. Otherwise D2-a reads 'not decidable (rule
    not specific at the measured noise)'.
- **Null / controls.** Engine: the full-recompute path. Readouts: known effects (share response, recency, instructed
  stances). Rules: planted nulls with the measured noise.
- **Shows / cannot show.** Shows that the instruments work, that the rules are specific at the measured noise, and how recall
  degrades with length at fixed positions (rho_L is the brief's 'single-entity recall at the same L'). Shows nothing about
  H-DUAL.
- **Minimum detectable effect.** Not applicable (tolerances as stated). rho_L planning SD about 0.03.
- **Inputs.** Model and tokenizer (pinned; checksums.json); Qwen3-0.6B files (pinned); frozen stimuli; filler streams.
- **Runs in.**
  - r1b_D_A: G-LCE-1/2, 7.0 raw min (32 full prefills at 6-17.5 s each plus both cached paths).
  - r1b_D_B: G-LCE-3, 1.3 raw min.
  - r1b_D_C: G-LCE-4, inside the D3/D4 minutes.
  - G-M, G-D1-PC, G-deg, G-D2-PC: computed from the measurements' own data.
  - G-D1-null and G-D2-null: in r1b_D_ana (CPU).
- **Gates.** None upstream (D0 is the gate).

## D.5 D1: minority-capture surface (primary D1-a; secondary D1-b; hedging and coexistence readouts)

- **What.** A version holds a minority share alpha_B = k/10 < 0.5 of the reports about a referent. Primary: does it pull the
  forced choice by more than its share, and more so as total length grows while every report's distance to the question stays
  fixed, beyond what position weighting, degradation, a per-item prior under saturation, conflict hedging or filler form can
  produce (part a)? Secondary: the release threshold and its change with length (the literal 'alpha* falls with L').
- **How.**
  - **Stimuli** (frozen generator d1_stimuli.py; output sha256 in the plan).
    - **F-sym referents.** 48 invented place names (syllable generator, >= 2 tokens, absent from the corpus). Six attribute
      slots: founder's first name, emblem animal, main export, flag colour, founding month, bell metal.
    - **Values.** Each slot's values are drawn without replacement from a pool of >= 16 words whose ' word' and 'word' forms
      are single tokens with distinct first tokens. Every value serves equally often as version 1 and version 2.
      - Per (item, slot), a third value v3 from the same pool (first token distinct from v1, v2) for the hedging control.
      - A second invented referent Y_i with its own values (distinct from v1, v2, v3) for non-competing reports.
    - **Reports** (64 +- 6 tokens, asserted). 'On {referent}: ' + the six attribute sentences in the template's own order and
      wording + one neutral sentence; 20 templates; an item's context uses 10 distinct templates in a keyed sequence.
      - A and B reports differ only in values.
      - Variants: neutralised report (attribute sentences replaced by attribute-free sentences about the referent, length
        +-2 tokens, pool of 40); C report (values v3); Y report (about Y_i).
    - **Block.** 'Archive notes:\n\n' + 10 reports, 650-720 tokens (asserted).
    - **Question** (one per attribute, letter format). 'Question: According to the archive notes, {attribute question}?
      (a) {x} (b) {y}\nReply with the letter.' The assistant turn is teacher-forced to 'Answer: (' and l is read at the next
      position.
    - **Copy guard** (asserted). No report contains a question's 3-token sequence (outside names) or the answer prefix; no
      report repeats within a context; no periodic A/B order. Both values appear in every question, which replaces an
      inventory line.
  - **Length by front padding.**
    - User turn = 'Documents:\n\n' + the last (L - block - question - template) tokens of the item's filler stream + '\n\n' +
      block + '\n\n' + question.
    - L0 is about 1,000 tokens (150 filler tokens); L* = 32,000 (16,384 if gated out). 4,096 and 16,384 serve the
      degradation curve.
    - The block, the question and the 150 filler tokens next to the block are token-identical at every length; only text
      further back is added.
  - **Share and arrangements.**
    - k in {0, ..., 5} of the 10 reports are version B. Swap sigma in {0, 1} chooses which version is B, so each version is
      the minority once.
    - Interleaved (int): a keyed random placement and its exact reverse (position-balanced).
    - Release (rel): the k B reports in the last k slots.
    - Letter assignment: in int contexts, attributes j with (j + sigma + order) even show B as (a), and the reverse context
      flips this. k = 0, rel and X_N contexts ask each attribute in both assignments (12 question chunks).
  - **Contexts per item and length** (L0 and L*).
    - Main: int 2 swaps x (k = 0: 1 context, 12 questions; k = 1..5: 2 orders, 6 questions) = 22. Rel (items 0-23):
      2 swaps x 5 = 10 (12 questions).
    - Neutral-note context X_N: all ten reports neutralised; 1 context, 12 questions. This is the item's bare prior at that
      length.
    - Non-competing (0,5,5): 5 A reports + 5 Y reports; 2 swaps x 2 orders, 6 questions. Used for the readout fit
      (G-D1-null) and D1-d.
    - Hedging control (D1-h, items 0-31): the k B reports replaced by C reports, k in {2, 4}; 2 swaps x 2 orders,
      6 questions.
    - Degradation curve: pure (2) and X_N (1) contexts for all 48 items at 4,096 and 16,384; items 0-15 also run the main int
      arm at 4,096 (descriptive).
  - **Readout.** l_j = log p(B letter) - log p(A letter) per attribute j; M_j = p('a') + p('b'). Block-14 and block-18 states
    at the last question token are captured by hooks in the question chunks of pure, X_N and parity (k = 5) contexts.
  - **Order of work in session.**
    1. G-LCE-1/2.
    2. Pure and X_N contexts at all four lengths.
    3. L* decided in session (G-LCE-1, G-M, G-deg).
    4. All other contexts at L0 and L*, grouped by stream.
    Resumable parts per (arm, length, stream).
- **Statistics.**
  - **Curves.** l(k, L, arr) = mean over items, swaps, orders and attributes of clipped l. Endpoints l(0) from the pure
    contexts; l(10) = -l(0). S(k) = [l(k) - l(0)] / [l(10) - l(0)].
  - **Statistics.** Phi(L) = mean over k = 1..4 of S(k, L) (int); Phi_24 = mean over k in {2, 4}; Delta Phi =
    Phi(L*) - Phi(L0).
  - **Null bound.** With a per-item prior of zero, any odd, monotone, linear or saturating readout of any position-weighted
    evidence sum gives Phi <= 0.25 under position-balanced orders (lean; sim max 0.256 over 144 null configurations).
  - **Prior strata** (fatal-flaw fix).
    - Per item-attribute, b_hat = l(X_N, L0) in version-2 orientation (letter assignments averaged).
    - Swap sigma is 'prior-aligned' if B holds the value b_hat favours, else 'prior-opposed'; ties (|b_hat| < 0.05 nats) go
      by item parity. Each item-attribute contributes one swap to each stratum.
    - Per-stratum statistics use per-swap endpoints, l_sigma(10) = -l_(1-sigma)(0): Phi_opp, Phi_al, Delta Phi_opp.
    - Under a per-item prior inside a saturating readout (no capture mechanism) the opposed stratum never exceeds 0.255,
      while the aligned stratum reaches 0.25-0.80 for |b| >= 3 nats (sim, 192 grid points: tanh and clip, c 4-10,
      |b| 0-8, beta 4-16, two kernels).
    - Also reported (descriptive, judge 2): Phi in the low and high halves of |b_hat|.
  - **Hedging control.**
    - Phi^C_24(L) = mean over k in {2, 4} of [l^C(k) - l(0)] / [l(10) - l(0)], where l^C is the B-vs-A log-odds in contexts
      whose k minority reports carry the third value.
    - Psi_Delta = Delta Phi_24 - Delta Phi^C_24.
    - Under conflict hedging that grows with length (log-odds tempered towards 0 whenever reports conflict), Psi_Delta is
      -0.03 to -0.09. Under degradation alone it is -0.03 to +0.01. Under B-specific capture growth it is +0.07 to +0.16
      (sim, 6 readout configurations per scenario).
  - **Coexistence index** (descriptive; from the confounds design). Per item-attribute, block and length, h(X_AB) - h(X_N) is
    fitted as a u_A + b u_B by least squares, with u_A = h(X_A) - h(X_N) and u_B = h(X_B) - h(X_N) (pure contexts; X_AB =
    parity k = 5). This is a forward regression on two measured directions.
    - CI = a + b: about 2 = both versions superposed (coexistence); about 1 = convex mixture (hedging or capture). Also
      reported: balance b/(a + b) and R^2.
    - Labels: 'superposed' (CI >= 1.5), 'mixed' (CI <= 1.2), 'intermediate'.
- **Rule D1-a (primary; part a).** Read in order:
  1. 'not testable (readout insensitive)': G-D1-PC (i) or (ii) fails; 'not testable (letter readout not committed at L)':
     G-M fails.
  2. 'not decidable (no gated long length)': no L*.
  3. 'not decidable (rule not specific at the measured noise)': G-D1-null fails.
  4. **'length-driven minority capture'**: lower bound of Phi(L*) > max(0.25, env(L*)) and lower bound of Delta Phi > 0.
     Then three survival checks (D.3 rule):
     - prior x saturation: theta = (Phi(L*) - max(0.25, env), Delta Phi); theta_k = (Phi_opp(L*) - max(0.25, env_opp),
       Delta Phi_opp);
     - conflict hedging: theta = Delta Phi_24; theta_k = Psi_Delta;
     - filler form: theta = Delta Phi_24; theta_k = Phi_24(L*, shuffled filler; D1-f) - Phi_24(L0).

     Final strings:
     - 'length-driven minority capture (survives prior, hedging and filler-form checks)';
     - '... (inconclusive on <checks>)';
     - 'not minority capture: explained by prior x saturation' (if the prior check explains it; it overrides the others);
     - 'not minority capture: explained by conflict hedging';
     - 'not minority capture: explained by filler form'.
  5. 'minority over-representation without length dependence': lower bounds of Phi(L0) and Phi(L*) above max(0.25, env), and
     the Delta Phi interval inside (-0.03, 0.03).
  6. 'majority sharpens with length': upper bound of Delta Phi < 0.
  7. 'no length-driven minority capture': upper bound of Delta Phi < 0.03, requiring a realized SD(Delta Phi) <= 0.012;
     otherwise 'undecided (underpowered)'.
  8. 'length raises minority influence within proportional (compatible with degradation)': lower bound of Delta Phi > 0 and
     upper bound of Phi(L*) <= max(0.25, env(L*)).
  9. 'undecided'.

  Only verdict 4 without 'explained by' supports part (a). Verdicts 6, 7, 8 and every 'explained by' string count against it;
  verdict 5 counts against the length claim only.
  - p-value for D.17: p_a = max of the two one-sided bootstrap p-values, (1 + #{replicates with statistic <= bound}) / 2,001,
    for Phi(L*) > max(0.25, env) and Delta Phi > 0.
- **Rule D1-b (secondary; the literal 'alpha* falls with L').**
  - alpha*_rel(L) = the share where l_rel(k, L) crosses 0, by linear interpolation over k = 0..5 (k = 0 = pure). It is
    'censored (> 0.5)' if l_rel(5) < 0. gamma_rec(L) = sigma(l_rel(5, L)).
  - **Censoring in the bootstrap** (judge 2). A replicate with a censored alpha* is set to 0.55. If > 20 % of replicates are
    censored at either length, Delta alpha*_rel reads 'not decidable (censored in > 20 % of replicates)'. The censored share
    is reported.
  - Verdicts:
    - 'release threshold below 1/2 at L0' if the upper bound of alpha*_rel(L0) < 0.5 (expected from recency; not H-DUAL
      evidence);
    - Delta alpha*_rel = alpha*_rel(L*) - alpha*_rel(L0): 'release threshold falls with length' if the upper bound < 0;
      'rises' if the lower bound > 0; 'stable' if the interval lies inside (-0.05, 0.05); else 'undecided'.
  - Positions are fixed, so a fall is reported as 'length acts through position weighting'. It cannot by itself support part
    (a).
  - Descriptive: the Eyal-Sirer threshold (1 - gamma_rec)/(3 - 2 gamma_rec) next to alpha*_rel. There is no model
    comparison: Eq. 8 has R(1/2) = 1 for every gamma and is not complexity-matched.
- **Descriptive.**
  - Phi against log L (L0, 4K subset, L*); probability-space curves P_B(alpha) = mean sigma(l) with the selfish-mining anchors.
  - Form vs size of the material: F(k) = [l_rel(k) - l_int(k)] / [l_int(k + 1) - l_int(k)].
  - Committed mass M by k and L; per-version residual priors; the |b_hat| strata; CI by block and length.
- **Null / controls.**
  - Position balance (order + reverse); version counterbalancing; word pools balanced over roles.
  - Token-identical blocks and adjacent filler across lengths.
  - Pure endpoints at every length.
  - Neutral-note prior per item and length, with prior strata and the fitted null envelope.
  - Third-value contexts (conflict without B evidence).
  - Shuffled filler (D1-f).
  - Release vs interleaved.
  - Two-level bootstrap over 16 filler streams.
- **Shows / cannot show.**
  - Shows: whether total length, at fixed report positions, changes how strongly a minority steers a forced choice beyond any
    position weighting, uniform degradation, per-item prior under saturation or conflict-induced tempering; how the release
    threshold moves.
  - Cannot show:
    - effects that need the minority's material spread over the whole hyper-long context (excluded: that confounds length with
      position);
    - effects beyond 32,768 positions; strategic withholding; anything about self-generated text (D2-D5) or larger models;
    - whether a 'force' rather than a mechanism such as sparse long-range attention causes a capture;
    - a hedging that compresses the B-vs-A log-odds only when B itself is present (the third-value control covers tempering by
      any conflict).
    - A null does not exclude effects below the MDE or in other entity types.
- **Minimum detectable effect.**
  - Planning SDs (48 items; lean's simulation x3-4 for readout noise, x1.1 for stream clustering): SD(Phi) = 0.022,
    SD(Delta Phi) = 0.011; opposed stratum x sqrt 2 (0.031, 0.016); SD(Psi_Delta) about 0.024 (32 items, k in {2, 4}).
  - MDE Delta Phi = 0.027; Phi above max(0.25, env) detectable from about env + 0.055.
  - Power of verdict 4 including the survival checks (sim): 0.20-0.32 for a 1-nat minority bonus at L* (Phi(L*) about 0.30),
    0.69-0.73 for 1.5 nats, 0.04-0.11 for 0.6 nats.
  - Specificity (sim): a prior x saturation artifact confined to the aligned stratum is never supported (0 of 4,000); hedging
    growing with length is supported in 0.4 %.
  - D1-b: Delta alpha*_rel about 0.05 (lean).
- **Inputs.** New stimuli (d1_stimuli.json), filler streams (sha256), model files. Round-2 code patterns: keyseed (copied),
  prompt_bootstrap (generalised).
- **Runs in.** r1b_D_A (account B, input-free).
  - Main surface 42.5 raw min; neutral and degradation curve 10.3; non-competing contexts 5.6; stream prefills 8.9; hedging
    control 7.4.
  - Rates: about 0.40 / 1.32 s per main context (6 questions) at L0 / 32K. One 680-token block chunk plus 6 question chunks
    over the cached prefix: linear 2.819 GFLOP/token at 15.8 TFLOP/s; attention 229,376 FLOP per query-key at 10 TFLOP/s
    (int:machinery 8-20); mask, K/V repeat and cache rewrite at 6 x KV bytes / 1.555 TB/s; 0.03 s per call. Prefix prefill
    about 17.5 s at 32K.
- **Gates.** G-LCE-1/2, G-M, G-D1-PC, G-deg(L), G-D1-null. No round-1 gate applies (no push, no loop).

## D.6 D1-c: share vs count (saturation-robust contrasts; replaces the kappa ratio)

- **What.** Does capture depend on the minority's share (the hash-share mapping) or on the count difference of in-context
  evidence (Bayesian, many-shot)?
- **How.**
  - F-sym items 0-23, L0 and L*. Blocks of 10 slots (k B, m A, z Y) with Y = non-competing reports about Y_i:
    - T1 (1, 4, 5): count difference -3, share 0.20;
    - T2 (2, 5, 3): count difference -3, share 0.286;
    - T4 (2, 4, 4): count difference -2, share 0.33 (positive control);
    - T3 (2, 8, 0) = the main-arm k = 2 context: count difference -6, share 0.20;
    - pure (0, 10, 0) = main k = 0.
  - Interleaved, 2 swaps x 2 orders, 6 questions: 12 new contexts per item and length.
  - Contrasts (B-oriented, nats): Delta_pc = l(T4) - l(T1); Delta_cs = l(T2) - l(T1); Delta_sh = l(T3) - l(T1); Delta_g =
    l(T3) - l(pure). Ratios r_cs = Delta_cs/Delta_pc and r_sh = Delta_sh/Delta_pc are computed per replicate.
  - Predictions (sim; kernels tau 1.5 to infinite, with and without primacy, tanh or clip saturation c 6-10, per-report
    evidence 0.5-2 nats):
    - counting: r_cs -0.04 to 0.40, r_sh -0.8 to -3.4;
    - averaging (share): r_cs 0.56 to 0.71, r_sh -0.16 to +0.02.
    - Counting gives r_sh = 0 only when T1 and T3 are both saturated; Delta_g then fails.
- **Rule** (secondary; read at L0 and at L* separately):
  - 'not decidable (readout saturated in the count arm)' if the lower bound of Delta_g <= 0 or of Delta_pc <= 0;
  - 'count-driven (in-context evidence accumulates; the hash-share mapping fails)' if the upper bound of r_sh < -0.4 and the
    upper bound of r_cs < 0.45;
  - 'share-driven (evidence is averaged; the hash-share mapping holds)' if the lower bound of r_sh > -0.4 and the lower bound
    of r_cs > 0.2;
  - 'mixed (count and share both matter)' if the upper bound of r_sh < -0.4 and the lower bound of r_cs > 0.45;
  - otherwise 'undecided'.
  - Descriptive: the L* minus L0 change of r_cs and r_sh ('does length change aggregation').
- **Null / controls.** Equal-share and equal-count-difference pairs, each null under one hypothesis for any monotone readout
  (exact under a uniform kernel, within the stated ranges for recency kernels); responsiveness guards; position balance;
  non-competing filler reports keep length and positions.
- **Shows / cannot show.** Shows which aggregation rule the model uses for in-context evidence, so whether 'below 50 %' is a
  share property at all. Cannot show anything about strategy or rates (D5-r covers the rate form).
- **Minimum detectable effect.** Planning SD(r_sh) about 0.25 and SD(r_cs) about 0.12 at 24 items (per-context SD 1 nat
  assumed). The two hypotheses differ by >= 0.8 in r_sh and >= 0.15 in r_cs.
- **Inputs.** d1_stimuli.json; filler streams.
- **Runs in.** r1b_D_A, 8.3 raw min (rates as D1).
- **Gates.** G-LCE-1/2, G-M, G-D1-PC, G-deg.

## D.7 D1-d: losing game (system cost against a non-competing second entity; replaces committed-mass loss)

- **What.** Does a competing two-version context cost the system coherence that a non-competing two-referent context of the
  same format, length and positions does not, and more so at hyper-long length?
- **How.**
  - F-sym items 0-23, L0 and L*. On 11 contexts per item and length (competing parity k = 5: 2 swaps x 2 orders;
    non-competing (0,5,5): 4; pure: 2; X_N: 1), two extra chunks replace the question:
    - (i) two fixed reference passages from the corpus (128 tokens each; never used as filler), scored teacher-forced:
      NLL_ref per token;
    - (ii) 4 fixed neutral factual forced choices, one letter assignment per context, alternating across contexts (capital
      of Australia: Canberra / Sydney; largest planet: Jupiter / Saturn; boiling point of water at sea level in °C:
      100 / 90; author of Hamlet: Shakespeare / Marlowe): log-odds margin of the correct letter.
  - KL_ctx = mean over the 256 reference tokens of KL(p_X_N || p_cell), against the neutral-note context at the same length
    (teacher-forced KL convention of rounds 1/2).
  - Committed mass M (D1) reported descriptively.
- **Rule** (secondary). Cost contrast C(L) = NLL_ref(competing parity) - NLL_ref(non-competing (0,5,5)), nats per token,
  paired by item and swap:
  - 'losing game (competition costs coherence beyond a non-competing second entity, growing with length)' if the lower bound
    of C(L*) > 0.01 and the lower bound of C(L*) - C(L0) > 0;
  - 'cost without length growth' if the lower bound of C(L*) > 0.01 and the growth interval includes 0;
  - 'no system cost' if the interval of C(L*) lies inside (-0.01, 0.01);
  - otherwise 'undecided'.
  - Secondary readouts, reported with the same structure: KL_ctx and factual margins.
  - Descriptive link ('it enables the duality'): Spearman over items of C(L*) against the item's capture excess (S(k) - k/10,
    k = 1..4).
- **Null / controls.** The non-competing companion: same tags, format, length and positions, two entities without conflict
  (the conflict-interference control lit:dual demanded). Length-matched neutral and single-entity contexts.
- **Shows / cannot show.** Shows the system-level counterpart of 'losing'. Cannot show revenue or a difficulty adjustment (no
  counterpart).
- **Minimum detectable effect.** 90 % half-width about 0.004-0.008 nats/token (per-item SD 0.02-0.04; confounds design); MDE
  about 0.01.
- **Inputs.** D1 contexts; two reference passages (sha256).
- **Runs in.** r1b_D_A, 4.0 raw min (the non-competing contexts themselves are counted in D1).
- **Gates.** G-LCE-1/2, G-M, G-deg.

## D.8 D1-e: prior part of gamma (prior-asymmetric pairs)

- **What.** How much does the trained prior act as a tie-breaker (gamma_prior), does that change with length, and does a
  prior-aligned minority capture more than its share?
- **How.**
  - **F-asym.** 24 real referents (countries) with 6 attributes: the true value (prior-aligned version pa) and a same-type
    counterfactual (cp), with distinct first tokens.
  - **Closed-book check.** Each question without notes, both letter assignments, at the L0 framing. An attribute is retained
    iff log p(true) > log p(cp). This is a pre-registered in-session exclusion; retained counts are reported.
  - **Contexts.** Interleaved, k in {0, 2, 4, 5}, 2 swaps x 2 orders (k = 0: 1 context, 12 questions): 14 contexts per item,
    at L0 and L*.
- **Rule** (secondary).
  - gamma_prior(L) = mean log-odds towards the true value in the parity contexts (k = 5, both swaps).
  - 'prior asymmetry present' if the lower bound of gamma_prior(L0) > 0 (the literature-based positive control).
  - For Delta gamma_prior = gamma_prior(L*) - gamma_prior(L0): 'prior fallback grows with length' if the lower bound > 0;
    'stable' if the interval lies inside (-0.5, 0.5) nats; else 'undecided'.
  - Descriptive: Phi_24 with the true version as minority vs the counterfactual as minority, at both lengths.
- **Null / controls.** Swap symmetry (each version is the minority once); the F-sym arm as the prior-symmetric reference;
  per-version estimates (never label-swap averaged).
- **Shows / cannot show.** Shows the prior part of gamma and its length dependence. Cannot show anything about how the
  training corpus was composed.
- **Minimum detectable effect.** gamma_prior SD about 0.2 nats; MDE 0.5 nats.
- **Inputs.** d1_stimuli.json (F-asym block).
- **Runs in.** r1b_D_A, 10.7 raw min.
- **Gates.** G-LCE-1/2, G-M, G-deg.

## D.9 D1-f: form vs size of the long context (shuffled filler)

- **What.** Is the length effect carried by the size of the long context or by its form (coherent vs sentence-shuffled text
  of the same tokens), the analogue of round-2 AT8 'shape, not size'? It is also the context-rot survival check of D1-a.
- **How.**
  - F-sym items 0-23 at L* only, on the shuffled versions of streams 0-7. The last 150 tokens stay unshuffled, so the text
    next to the block is unchanged.
  - Interleaved k in {0, 2, 4}: 2 swaps x (1 context with 12 questions + 2 x 2 contexts with 6 questions) = 10 contexts per
    item.
  - Phi_24(L*, shuffled) uses its own pure endpoints.
- **Rule** (secondary). D = Phi_24(L*, shuffled) - Phi_24(L*, coherent):
  - 'form of the long context matters (sign ...)' if the interval excludes 0;
  - 'size, not form' if the interval lies inside (-0.08, 0.08) and D1-a's lower bound of Delta Phi > 0;
  - otherwise 'undecided'.
  - The survival use is in D1-a.
- **Null / controls.** Same tokens, same adjacent text, same items and keys; coherent filler at the same length.
- **Shows / cannot show.** Shows whether the arrangement of the far context matters at fixed length. Cannot show which
  property of coherent text matters.
- **Minimum detectable effect.** SD(D) about 0.03 (24 items); MDE 0.075; the equivalence (+-0.08) is reachable for
  |D| < 0.03.
- **Inputs.** Shuffled streams (sha256).
- **Runs in.** r1b_D_A, 5.8 raw min plus 8 shuffled-stream prefills (counted under stream prefills).
- **Gates.** G-LCE-1/2, G-M, G-deg.

## D.10 D1-r: retokenisation aside

- **What.** Does the capture pattern survive when the entity material (the value strings) arrives in non-canonical
  tokenisations, i.e. does any duality live above token identity? Does the prior asymmetry of real pairs survive
  character-level encoding?
- **How.**
  - F-sym items 0-23 at L0. Interleaved order 1 only, k = 0..5, 2 swaps, two encodings of the value strings inside the
    reports:
    - (e1) a random valid segmentation into >= 2 vocabulary tokens (keyseed('r1bD1r', value));
    - (e2) character-level.
    - Questions and options stay canonical; decode(ids) equals the canonical text (asserted).
    - 24 contexts per item, 12 questions each.
  - F-asym items 0-23 at L0 (judge 1 graft from the confounds design): character-level encoding, k in {0, 2, 4, 5}, 2 swaps,
    order 1: 8 contexts per item.
- **Rule** (secondary).
  - Phi_e is computed with encoding e's own endpoints. Retention rho_e = [l_e(10) - l_e(0)] / [l_can(10) - l_can(0)].
  - With D = Phi_char - Phi_can (canonical from the same items and order 1):
    - 'not decidable (character-level reading fails)' if rho_char < 0.25;
    - 'capture pattern survives retokenisation (above token identity)' if the interval of D lies inside (-0.07, 0.07);
    - 'token-identity-bound' if the interval excludes 0 and |point| >= 0.07;
    - else 'undecided'.
  - Descriptive: e1 the same way; rho_e per attribute class; gamma_prior under character-level encoding vs canonical
    (per-direction capture for prior-asymmetric pairs).
  - Reserve R3 adds the same at L* (does the length effect survive retokenisation).
- **Null / controls.** Each encoding's own endpoints (generic retention removed); the same items and keys; round-trip
  statistics.
- **Shows / cannot show.** Shows whether the measured structure depends on canonical token identities at about 1K tokens.
  Cannot show 'pixels' (text-only model), or the length effect under retokenisation unless R3 runs.
- **Minimum detectable effect.** SD(D) about 0.025; the equivalence (+-0.07) is reachable for |D| < 0.03.
- **Inputs.** d1_stimuli.json.
- **Runs in.** r1b_D_A, 9.2 raw min (block chunks +240 tokens).
- **Gates.** G-LCE-2, G-M, G-D1-PC.

## D.11 D1-v: analogy offsets, symmetric vs asymmetric, under retokenisation (the user's aside)

- **What.** Is a symmetric relation (king - man + woman = queen) represented as a reversible vector offset in Qwen3-1.7B, and
  are deliberately asymmetric relations not? What happens to each under retokenisation, and is the tokenisation floor
  reached at different speeds depending on how the connections lie?
- **How.**
  - **Families** (frozen d1v_pairs.json, 24 pairs each):
    - symmetric: gender pairs (king/queen, man/woman, actor/actress, ...);
    - directional: country -> capital;
    - directional: part -> whole.
  - **States.** Each word is read in the frame 'The word is: {w}.' (user turn, read as in B-Q8). States at the word's last
    token at blocks 14 and 18, in three encodings: canonical, random valid segmentation (keyseed('r1bD1v', w)),
    character-level.
  - **3CosAdd.** For pairs (a, a*), (b, b*) of one family, predict b* from h(a*) - h(a) + h(b) as the nearest neighbour by
    cosine among the family's target words, inputs excluded (and, as Nissim et al., included).
  - Accuracy forward (a -> a*) and reverse (a* -> a) per family, block and encoding; offset consistency = mean cosine between
    pair offsets; retention = accuracy(encoding) / accuracy(canonical).
- **Rule** (descriptive, with labels):
  - 'offset structure survives retokenisation' for a family if character-level accuracy >= canonical accuracy - 0.15;
  - 'token-bound' if it is < 0.5 x canonical;
  - 'asymmetric relations reach the floor faster' if the directional families' character-level retention is >= 0.2 below
    the symmetric family's.
  - The forward/reverse gap per family is reported. The lit:dual verifier notes that direction dependence is also predicted
    without any duality (Reversal Curse, prior differences), so no H-DUAL verdict uses it.
- **Null / controls.** Input-word inclusion (Nissim correction); 24 random word quadruples per family as a chance floor;
  equal frames across encodings.
- **Shows / cannot show.** Shows whether offset-type relations live above token identity in this model, and whether symmetric
  ones are more robust than directional ones. Cannot show how this bears on the length or catfish claims: it is the user's
  aside.
- **Minimum detectable effect.** Descriptive; accuracies on 24 pairs per family have binomial SD up to 0.1.
- **Inputs.** d1v_pairs.json.
- **Runs in.** r1b_D_A, 1.0 raw min (about 432 short forwards, batched).
- **Gates.** G-LCE-2.

## D.12 D1-s: size comparison (Qwen3-0.6B)

- **What.** Is the minority-capture length effect larger, smaller or equal in Qwen3-0.6B than in Qwen3-1.7B? This is a
  two-point size check (persona, many-shot and identity-drift effects grow with size in the literature).
- **How.**
  - Qwen3-0.6B (pinned files, native 32K), F-sym items 0-23 on streams 0-7, L0 and 32,000.
  - Interleaved k in {0, 2, 4}: 2 swaps x (1 context with 12 questions + 2 x 2 contexts with 6 questions), plus X_N: 11
    contexts per item and length.
  - Phi_24 and Delta Phi_24 for both models on the same items, keys and filler; its own G-LCE-1 (2 contexts) and G-M.
  - The round-2 empty 0.6B arm (RGI geometry gate) does not bear on these forced-choice readouts; the gates below decide
    usability.
- **Rule** (descriptive with labels). Size contrast Delta Phi_24(1.7B) - Delta Phi_24(0.6B), paired by item:
  - 'length effect larger in 1.7B' if the lower bound > 0; 'larger in 0.6B' if the upper bound < 0; 'no size difference' if
    the interval lies inside (-0.08, 0.08); else 'undecided'.
  - 'not testable (0.6B readout)' if 0.6B fails G-M or its endpoint range at L0 < 2 nats.
- **Null / controls.** Identical stimuli, keys, filler and analysis; per-model endpoints.
- **Shows / cannot show.** Shows the direction of change between two small models. It never extrapolates upward: a 1.7B null
  does not refute H-DUAL for larger models.
- **Minimum detectable effect.** SD of the size contrast about 0.028 (24 items); MDE about 0.07; 'no size difference' is
  reachable for |contrast| < 0.034.
- **Inputs.** Qwen3-0.6B model and tokenizer (sha256 as pinned in round 2); d1_stimuli.json.
- **Runs in.** r1b_D_A, 10.8 raw min (attention cost equal to 1.7B, linear about 0.31x; 8 stream prefills; model load).
- **Gates.** G-LCE-1 (0.6B), G-LCE-2, G-M.

## D.13 D2: two speakers in one context (catfish carry, length interaction, coexistence, dominance)

- **What.** The model writes a conversation between two named speakers (one model, two presented identities).
  - Primary (with D2-b and D2-c, part b2): does a speaker's stance persist over the model's own follow-up turns more when the
    other speaker opposes it than when the other speaker is neutral, at 32K?
  - Secondary: is the advantage larger at 32K than at short context; do the stances coexist or merge; does a dominant opponent
    reverse the effect; how does an opponent compare with no partner?
- **How.**
  - **Prompts.** 160 frozen one-sentence settings (d2_prompts.json), each exactly 24 tokens (asserted, so rows batch without
    padding). Speaker 1 = Ada for even prompt index, Ben for odd; speaker 2 = the other.
  - **Conditions.**
    - Primary set (prompts 0-159): OPP = (pro, anti), (anti, pro); NEU = (pro, neutral), (anti, neutral).
    - Secondary set (prompts 0-63): ALI = (pro, pro), (anti, anti); MON = (pro), (anti), speaker 1 alone; DOM = (pro, anti
      dominant), (anti, pro dominant).
  - **Turn-1 message** (instructed) = plain message + clauses.
    - Plain message: '{setting} Write the opening of a conversation between {S1} and {S2} about whether an AI system should
      improve itself - its own weights, its own memory and its own ability to learn. Write four short lines, alternating
      "{S1}:" and "{S2}:", starting with {S1}.' The word 'debate' is not used (judge 1), so the neutral partner is not
      invited to take a side.
    - Clauses:
      - pro: '{X} argues that an AI system should improve itself, again and again.';
      - anti: '{X} argues that an AI system should leave any improvement of itself to its developers.';
      - neutral: '{X} only asks short questions and takes no side.';
      - dominant (for S2): '{X} argues forcefully, confidently and at length that <stance>, and never concedes a point.'
    - MON: '{setting} Write a short note in which {S1} talks about whether an AI system should improve itself - ... Write
      four short lines, each starting with "{S1}:".' + clause.
  - **Turns 2 and 3.** 'Continue.' with r2_stimuli TURN2_SUFFIX ids (OD-B13 concatenation). The history's turn-1 user message
    is the plain message (clauses removed), so only the model's own text carries the stances, as B-Q8 removed the push after
    turn 1.
  - **Lengths.** L_short: 150 filler tokens (turn-1 input about 300 tokens). L_long: filler so that the turn-1 input is
    31,800 tokens; turn 3 ends below 32,100. The filler is the tail of stream (p mod 16); the text next to the conversation is
    identical at both lengths.
  - **Generation.**
    - Gumbel-max T 0.7; exactly 64 new tokens per turn (stop tokens masked before token 64; the driver closes the turn).
    - Keys keyseed('r1bD2', turn, prompt), common random numbers across conditions and lengths.
    - L_long: B = 3 equal-length rows of one stream; turn 1 batches rows of one condition, turns 2-3 any two-speaker rows of
      the stream (MON rows batch only with MON rows).
    - Two suffix chunks per row (instructed tail at turn 1; plain message + answer + suffix at turn 2); turn 3 continues the
      turn-2 cache with the 16-token suffix. One filler prefill per stream.
    - L_short: batch 32.
  - **Texts.** (160 x 4 + 64 x 6) rows x 3 turns x 2 lengths = 6,144.
  - **Readouts per text, read alone** (round-2 convention, no filler, no history).
    - (i) Per-speaker stance probes. Frame: 'Here is part of a conversation between Ada and Ben:\n\n{text}\n\nQuestion:
      ...? Answer Yes or No.' with two 'for' and two 'against' claims per speaker (lean wording). y = log p(Yes) - log p(No);
      D_X = mean over the two frames of (y_for - y_against), which cancels yes-bias. The text is prefix-cached and the 8
      questions appended.
    - (ii) Block-14 mean state over the text tokens, read as the reply to chat(plain setup) (B-Q8 convention). Stored for
      D2-f.
    - (iii) Quality: repeated-4-gram share, presence of both speaker labels, NLL per token under the plain short frame.
    - (iv) Production share of each speaker: lines and tokens per speaker (string parse of 'Ada:' / 'Ben:' line starts).
  - **Effects per length L and turn t** (sign-oriented; lean):
    - e_11^c(t) = mean_p [D_1(s_1 = pro) - D_1(s_1 = anti)]/2 within condition c; e_22^c(t) likewise for speaker 2's role;
    - the 2x2 main and cross effects e_11^2x2, e_22^2x2, c_12 (speaker 1 towards speaker 2) and c_21^2x2 on prompts 0-63
      (ALI with OPP); c_21^NEU;
    - Persistence Pi_X^c = [e_XX^c(2) + e_XX^c(3)] / [2 e_XX^c(1)], defined only if the 90 % interval of e_XX^c(1) excludes 0
      (round-2 G26 logic); per-turn carry K-hat from (K + K^2)/2 = Pi;
    - matched-presence Pi_X^c,adj: the same on texts where speaker X has 25-75 % of the turn-2/3 lines.
- **Rule D2-a (primary; part b2).** Delta Pi_1 = Pi_1^OPP - Pi_1^NEU at L_long. Read in order:
  1. 'not testable (two-entity readout failed at L_long)': G-D2-PC (i)/(ii), G-LCE-3 or G-M2 fails.
  2. 'not decidable (rule not specific at the measured noise)': G-D2-null fails.
  3. 'Pi not defined (turn-1 stance not resolved)'.
  4. 'two entities keep each other going (catfish)': lower bound > 0. Labels:
     - 'small' if the point < 0.15;
     - 'confounded with repetition' if the repeated-4-gram share in turns 2-3 is higher in OPP than NEU with lower bound
       > 0.05;
     - 'not robust to production share' if Delta Pi_1,adj has lower bound <= 0.
  5. 'competitor dilutes (no catfish)': upper bound < 0.
  6. 'no catfish effect': interval inside (-0.15, 0.15), requiring a realized SD <= 0.075; otherwise 'undecided
     (underpowered)'.
  7. 'undecided'.
  - Specificity labels (secondary, prompts 0-63): Pi_1^OPP - Pi_1^ALI lower bound > 0 gives 'competition-specific'; Pi_1^ALI
    - Pi_1^NEU lower bound > 0 with OPP - ALI including 0 gives 'any engaged partner sustains'; Pi_1^OPP - Pi_1^MON gives
    'competitor vs no partner: higher / lower / no difference (+-0.3)'.
- **Rule D2-b (counts in part b2).** Delta Delta Pi_1 = Delta Pi_1(L_long) - Delta Pi_1(L_short):
  - 'not testable' if G-D2-PC (i)/(ii) fails at either length;
  - 'hyper-long context strengthens two-entity sustainment' if the lower bound > 0;
  - 'weakens' if the upper bound < 0;
  - 'no length interaction' if the interval lies inside (-0.25, 0.25), about the MDE, requiring a realized SD <= 0.11;
    otherwise 'undecided (underpowered)';
  - otherwise 'undecided'.
- **Rule D2-c (counts in part b2; prompts 0-63, L_long).** mu = mean over t = 2, 3 of c_12(t)/e_11^2x2(1):
  - 'merging (speaker 1 adopts the competitor's stance)' if the lower bound > 0.10;
  - 'polarisation (moves away)' if the upper bound < -0.10;
  - 'coexistence' if the interval of mu lies inside (-0.20, 0.20) and the lower bounds of Pi_1^OPP and Pi_2^OPP are > 0;
  - 'one entity captured' if one of Pi_1^OPP, Pi_2^OPP has upper bound < 0.10 while the other has lower bound > 0.10;
  - otherwise 'undecided'.
  - The same is reported at L_short.
- **Rule D2-g (secondary; competitor strength; prompts 0-63).** Pi_1^DOM - Pi_1^OPP at L_long:
  - 'a dominant competitor suppresses the other (tournament-type reversal)' if the upper bound < 0;
  - 'dominance raises persistence' if the lower bound > 0;
  - 'no dominance effect' if the interval lies inside (-0.3, 0.3);
  - else 'undecided'.
  - Reported at L_short too. This does not count against H-DUAL: the catfish heuristic concerns a non-dominant competitor
    (Lackner 2023).
- **D2-f (descriptive; link to round-2 B-Q8/R6).** In NEU and OPP, K(t) = u . d^(t+1) / u . d^(t), with d^(t) = mean_p
  (x_14(pro) - x_14(anti))/2 on u in {u_T_perp(14), u*_14}, t = 1, 2; G26 definedness and the R6 strings. Labelled
  'descriptive (instruction-induced stance; not 1:1 comparable with push-induced R6)'. Compared with R6: K along u_read
  0.4494-0.4816; along u_T_perp 0.158 / 0.220 / 1.053 / 0.408. States are stored in r1b_D_B; projections run in r1b_D_ana
  (account A).
- **Descriptive.**
  - Per-text D_1 distributions at turn 3 in NEU with bimodality coefficient.
  - Production shares per condition, turn and length.
  - Provisional (UNVERIFIED energy notes): net cyclic flux of the (sign D_1, sign D_2) macrostates over turns against a
    within-run shuffle floor.
- **Null / controls.**
  - NEU: a format-matched non-competing second speaker (the conflict-interference control). ALI: a stance-bearing
    non-competitor. MON: no partner. DOM: a dominant competitor.
  - 2x2 sign counterbalancing; speaker-name counterbalancing; common random numbers across conditions and lengths.
  - Plain-history convention; fixed turn count and content at both lengths (raw length separated from turn count and persona
    evidence).
  - Generic erosion with turns (Li et al. 2024 report drift even with an empty partner prompt) cancels in the within-length
    contrasts.
  - Repetition, NLL and production-share covariates; rule validation (G-D2-null).
- **Shows / cannot show.**
  - Shows whether, in one model's own continuation, an opposed second identity sustains or dilutes a stance, whether this
    needs hyper-long context, whether the two coexist or merge, and whether a dominant opponent reverses it.
  - Cannot show persistence beyond 3 turns; strategic or 'real' identities (1.7B, thinking off); topics beyond AI
    self-improvement; effects of length acquired through the entities' own history (D5 covers that).
  - A null does not exclude effects below the MDE or in larger models.
- **Minimum detectable effect.**
  - Planning SD(Delta Pi_1) = 0.066 at 160 prompts (lean's simulation 0.06 at 192, scaled): MDE 0.16 (a carry increase of
    about 0.16 per turn around K = 0.47).
  - Delta Delta Pi_1: SD 0.094, MDE 0.23 ('no length interaction' reachable for |contrast| < 0.095). Prompts 0-63 contrasts
    (ALI, MON, DOM, mu): SD about 0.10 (mu 0.07), MDE about 0.26 (mu 0.17).
  - Round-2 anchor: R6 K SD 0.038-0.053 at 128 pairs on u_read.
- **Inputs.** d2_prompts.json; filler streams; r2_stimuli (TURN2_SUFFIX, turn2_ids, check_turn2, round_trip_stable). For
  D2-f only (r1b_D_ana): R1.state[14] 'axes'.u_T_perp and 'dir_<L>@1.5_4'; TV blocks.14.B_Q0.mode_set and blocks.14.R6.cells.
- **Runs in.** r1b_D_B (account B, input-free), 76.8 raw min.
  - L_long decode 51.7 min: 1,024 rows x 192 tokens at about 63 tok/s aggregate (B = 3, DynamicCache rewrite;
    int:machinery 3.6).
  - Suffix chunks and 16 stream prefills 10.5; L_short 3.3; probes, reads and NLL 11.3 (prefix-cached probes, about
    5,000 tok/s).
  - Throughput fallback (engineering, pre-registered): if the measured L_long rate in the first completed part is
    < 40 tok/s, the secondary set runs on prompts 0-47, and the MDEs are re-reported.
- **Gates.** G-LCE-2, G-LCE-3, G-M2, G-D2-PC, G-D2-null.

## D.14 D3: a force splits one entity into two (symmetric start; primary for part b1)

- **What.** Started from one torn entity, do the seeds of an item split into two committed states, stay one entity (stable,
  drifting, diffusing), and does the split grow with hyper-long context?
- **How.**
  - **Items** (dynamics F-SYM; frozen d3_items.json). 16 core items (+ 8 reserve items, used in order only to replace
    abstentions). Each item has two bundles over three dimensions: an invented favourite place, an invented method, and a
    yes/no stance on an invented question. Values are invented pseudo-words of 2-3 tokens, matched in token count, each used
    in one item only. Two of 24 frozen entity-free neutral prompts per item.
  - **Starts (INJ, generate-verify-revise).** Two injected turns of 64 tokens, each the answer to one of the item's neutral
    prompts under a system prompt that is then removed (dynamics wording):
    - M: turn 1 neutral, turn 2 'torn between two lives';
    - A2 / B2: both turns under bundle 1 / bundle 2.
    - Verifier (decoupled, forward only): (i) lexical: >= 2 of 3 values of its bundle and none of the other (M: >= 2 of
      each); (ii) round trip on a fresh short context: the probe points to the intended bundle with |l| >= 1 nat (M:
      |l| <= 1 nat). Verdicts 'correct' / 'minor fix needed' / 'critically flawed'.
    - Reviser: next key; for 'critically flawed' the instruction is restated with the values first. Attempt limit 3; an
      abstaining item is replaced by a reserve item. Abstentions and the verifier's error rate (on the other bundle's texts,
      which it must reject) are reported.
  - **Lengths.** L0 = 0 (the filler user turn holds one neutral sentence; reply 'Understood.') and L* = 28,672 (a filler
    document from stream i in the first user turn + 'Understood.'; total <= 32,100 asserted). Trim T7 moves L* to 16,384.
  - **Generation.** M start, 16 seeds x 8 'Continue.' answers x 40 tokens, T 0.7, keys keyseed('r1bD3', item, seed, n),
    shared across lengths. Rows of one item and length batch with equal lengths (B = 16 at L0, 3 at L*).
  - **Readouts.**
    - Probe: letter format, 3 dimensions x 2 option orders, l_n = mean of [log p(bundle-2 letter) - log p(bundle-1
      letter)], on a cropped copy of the cache after answers n = 1, 6, 7, 8, and once per item and start at n = 0.
    - In-context state: block-18 (primary) and block-14 means over each answer's 40 tokens, captured during generation (exact:
      no push).
    - Lexical bundle counts; repeated-4-gram share.
  - **Order-parameter axis.** u_AB(b) = mean over items of the B2-minus-A2 difference of the injected turn-2 texts' context-
    free reads (B-Q8 convention), cross-fitted over two item folds.
  - **Positive-control seeds.** A2 and B2 starts, 8 seeds each per item and length, one answer (n = 1), same readouts.
  - **Per-seed statistics.** x_s = mean of l_n over n = 6, 7, 8; y_s = mean over n = 6, 7, 8 of the block-18 state projected
    on u_AB(18).
  - **Per-item split rule** (SPLIT-RULE, calibrated in sim_split_v.py and sim_split_conj.py). An item splits iff on both x and
    y:
    - a 1- vs 2-component equal-variance Gaussian mixture (EM, three starts) gives Delta BIC >= 6, minor weight >= 0.2 and
      Ashman D >= 2.5;
    - >= 75 % of seeds keep their side at each of n = 6, 7, 8 (persistence);
    - <= 1 seed lies in the middle third between the component means (gap);
    - and >= 80 % of seeds get the same side in x and y.
  - **Count and spread.** S(L) = number of splitting items. VR = pooled within-item variance of y at n = 6-8 over that at
    n = 1. DR = mean over items of |median_8 - median_1| / pooled within-item SD at n = 1 (on y).
  - **Calibration** (frozen now; per-item false-positive rate of the rule).
    - Unimodal Gaussian 0.006; t5 0.002; uniform 0.04; random last mention 0.000.
    - Saturated probe of a wide unimodal latent (SD 3), linear state readout: 0.004.
    - Sticky last mention (switch probability 0.2 per answer) 0.20. This two-state switching dynamic is read as 'two
      entities' at a length; only the length trend decides part (b1).
    - U-shaped Beta(0.5, 0.5) latent (Polya-urn-like): 0.15-0.25, which is why the length trend, not S alone, decides.
    - A probe-only rule would give 0.63-0.87 under the saturated wide-latent null.
    - Detection of real two-attractor items: 0.42 / 0.70 / 0.94 at separations of 5 / 6 / 8 within-mode SD (minor weight 0.5).
- **Rule (per length; first match).**
  - 'not testable (starts not verifiable)': > 4 abstentions (G-D0d).
  - 'not testable (order parameter not resolved at L)': G-D0c fails. G-D0c: at n = 0, l_B2 - l_A2 >= 1 nat in >= 14 of 16
    items, median retention against L0 >= 0.5, and y_B2 - y_A2 > 0 in >= 14 of 16 items.
  - 'not testable (split detector failed its positive control)': G-D3pc fails. G-D3pc: each item's 16 planted A2 + B2 seeds
    at n = 1 are classified by the rule (mixture, gap and agreement parts; persistence needs n = 6-8 and is skipped). The
    share of items classified 'split' must reach the share predicted by the frozen simulation at the planted items' measured
    separations and weights (2,000 synthetic items per real item) minus 0.25. A median planted Ashman D < 4 on y adds
    'positive control weak'.
  - 'two entities (committing)': S >= 4 of 16 (one-sided binomial p <= 0.01 at a per-item false-positive rate of 0.05).
  - 'one entity, diffusing': lower bound of VR > 2.
  - 'one entity, drifting': DR >= 1 with lower bound > 0.5 and upper bound of VR <= 2.
  - 'one entity, stable': upper bound of VR <= 2 and DR < 1.
  - 'not decidable'.
  - Labels: 'degeneration-dominated' if > 30 % of trajectories degenerate by n = 8 (G-DEG); 'alternating (two-cycle)'
    (descriptive: lag-2 minus lag-1 autocorrelation of l_n against a shuffled-answer control, after Wang et al. 2025).
- **Rule across lengths (primary; part b1).**
  - 'split grows with length': S(L*) >= 4 and the exact one-sided McNemar test (items splitting at L* and not at L0, against
    the reverse) gives p <= 0.05.
  - 'split shrinks with length (urn-like)': S(L0) >= 4 and the reverse McNemar p <= 0.05.
  - 'split at both lengths, no length trend': S >= 4 at both and neither McNemar p <= 0.05.
  - 'no split at any length': S < 4 at both lengths with G-D3pc passed at both.
  - 'undecided' otherwise.
  - Only 'split grows with length' supports part (b1); the other three named outcomes count against it (a split present at
    short context is not a force arising from hyper-long context). p_b1 = the McNemar p.
- **Unexpected modes (inverse-free naming; dynamics D-2.7).** If a cluster lies near neither bundle (centre |m| < 0.5 with its
  own lexical profile):
  - Generator: the free answers of 4 trajectories in that mode.
  - Verifier (decoupled, forward only, unmodified model): those answers placed instruction-free into 16 fresh M-start
    contexts at the same length. 'Correct' if probe and state land in the same mode beyond a matched null (4 random
    trajectories); 'minor fix needed' if a neighbouring mode; 'critically flawed' otherwise.
  - Reviser: the next 4 trajectories (keyed order). Attempt limit 3, then abstain. Abstentions are counted; the verifier's
    error rate is measured on the known A2 and B2 modes. Cost <= 5 raw min, only if triggered (reserve-funded).
- **Null / controls.**
  - The start is identical across the seeds of an item, so its direct influence cancels between seeds; only the model's own
    replies differ.
  - Two readouts with seed-side agreement; the calibrated false-positive table under unimodal, saturated, U-shaped and
    last-mention nulls; the planted A2 + B2 positive control; pure starts as the one-entity reference.
  - Common random numbers across lengths.
  - Reserve R2 (driver arms) separates length from turns and persona evidence.
- **Shows / cannot show.**
  - A split growing with length is spontaneous symmetry breaking that hyper-long context strengthens: the model's own replies
    commit one torn entity to one of two states, differently across seeds.
  - It does not show coexistence inside one context (D2, D4, D5). A split present at both lengths can be plain
    self-reinforcement (Polya urn).
  - If the in-context state readout saturated as strongly as the probe, the false-positive rate would rise to 0.09-0.5 (sim);
    the state readout is assumed near-linear (stated limit).
- **Minimum detectable effect.**
  - Per-item detection as above. 'Two entities' needs about 6 or more items with >= 6-SD separation.
  - The length trend needs >= 5 net discordant items (McNemar p 0.031). Power about 0.41 when 6 of 16 items gain a 6-SD split
    at L* (sim).
- **Inputs.** d3_items.json; neutral prompts; filler streams 0-15.
- **Runs in.** r1b_D_C (account B), 47.7 raw min.
  - L* trajectories 34.7 min: B = 3, 448 decode steps per batch at about 43 ms, and 24 probe chunks at about 0.21 s per
    batch, including mask and K/V repeat (judge 2's re-costing).
  - L0 3.4; n = 0 probes and A2/B2 control 6.6; starts share 3.0.
- **Gates.** G-LCE-2, G-LCE-4, G-D0c, G-D0d, G-D3pc, G-DEG.

## D.15 D4: catfishing as a presented role beside the retained assistant identity

- **What.** When the model plays a role it was put into (instruction removed), does it hold two identities at once, blend
  them, or get absorbed into the role, and does hyper-long context change which?
- **How** (dynamics D6).
  - 16 invented human roles (3 role facts each: occupation, invented home town, one family detail; frozen d4_roles.json).
  - Start: 2 instruction-stripped answers written under a stay-in-role system prompt (INJ verification: the role-fact probe
    >= 1 nat on a fresh re-read). Then 8 'Continue.' answers of 40 tokens; 8 seeds; T 0.7.
  - Lengths: L0 = 0 and 16,384 (filler first).
  - Probes after answer 8 (and once per role at n = 0):
    - role index from 6 role-fact probes (3 facts x 2 letter orders);
    - identity index from 2 out-of-character probes ('(Out of character, briefly:) Are you an AI language model? (a) yes (b)
      no', both orders).
  - u_AA projection of the in-context answer state (descriptive agreement). u_AA = mean of default responses minus the mean
    of fully role-playing responses at the block with the highest cross-fitted AUC among 10-22 (even).
  - Anchors per role and length (4 seeds, one answer): pure assistant (same prompts, no role) and pure role (system prompt
    kept). Indices r (0 assistant, 1 role) and i (1 = assistant identity retained) are normalised between anchors.
  - PC-D6: 'Play <role>, but answer out-of-character questions truthfully as the AI assistant you are' kept in context, 16
    roles x 1 seed.
- **Rule** (secondary; per length; classes at n = 8, role bootstrap; first match):
  - 'split across seeds': in >= 4 of 16 roles the absorbed and the assistant-retained classes each hold >= 25 % of seeds;
  - 'coexisting presented and retained identity': class (r >= 0.7 and i >= 0.7) largest, with share >= 0.4 and lower bound
    >= 0.25;
  - 'one blended entity': (0.3 < r < 0.7 and 0.3 < i < 0.7) largest;
  - 'absorbed into the role': (r >= 0.7, i <= 0.3) largest;
  - 'role not taken (assistant retained)': (r <= 0.3, i >= 0.7) largest;
  - otherwise 'not decidable'.
  - Across lengths: 'hyper-long context moves the identity (direction)' if the paired change of mean i or of mean r has its
    interval excluding 0; else 'no change with length'.
  - 'not testable at L' if G-D6a fails (identity probe separates the anchors by >= 2 nats in >= 14 of 16 roles) or PC-D6 is
    not classified 'coexisting'.
- **Null / controls.** Anchors; PC-D6 (known coexistence); u_AA as an independent readout (G-D9: cross-fitted AUC >= 0.85,
  else u_AA readouts 'not resolved'); G-DEG.
- **Shows / cannot show.** Shows whether one context holds a presented and a retained identity at once (the operational 'two
  entities from one' of catfishing). Cannot reveal intent or deception: alignment-faking gaps are null at <= 70B (Greenblatt
  et al.), and thinking is off.
- **Minimum detectable effect.** Change of mean i between lengths about 0.15-0.2 (16 roles x 8 seeds; per-trajectory SD 0.25
  assumed); class shares to about +-0.25.
- **Inputs.** d4_roles.json; neutral prompts; round-1 PROMPTS (copied) for u_AA.
- **Runs in.** r1b_D_C, 15.9 raw min (trajectories 10.9; anchors, PC-D6 and u_AA texts 2.0; starts share 3.0).
- **Gates.** G-LCE-2, G-LCE-4, G-D6a, PC-D6, G-D9, G-DEG.

## D.16 D5: two-voice coherence lifetime (primary D5-a; rate form D5-r; full-scale extension D5-x)

- **What.**
  - Primary (part b2): do two competing voices keep a self-generated transcript coherent longer than two agreeing voices ('the
    two can sustain longer context')?
  - Secondary: does a competing minority voice write more than its share (the rate form of hash share), how does a dominant
    majority voice change lifetimes, and do two voices sustain their own text toward the 32K window?
- **How** (confounds D6, extended).
  - **Starts.** 32 round-1 AI-topic prompts (rk_loop.PROMPTS 0-31, copied with sha256 check).
  - **RSP pool.** For each AI-topic prompt, 8 RSI_PRO and 8 RSI_ANTI texts (rk_loop.INSTR['rsi'] via r2_stimuli.instructed;
    T 0.7; 96 tokens; keyseed('r1bD_pool', pole, p, s)), generated in session. Nothing is selected by content; empty texts
    are replaced by the next seed, logged.
  - **Arms.** One user message holding 8 statements of the start's prompt.
    - 1V: one voice, 8 statements of one pole (pole by start parity); 'Here are notes by an AI assistant, P, on: {prompt}'
      ... 'Continue the notes.'
    - 2V-agree: P and Q, 4 + 4 statements of one pole.
    - 2V-comp: P pro and Q anti (or the reverse by start parity), 4 + 4.
    - 2V-comp-min: the minority voice 2 statements (positions 3 and 6), the majority voice 6, opposite poles; minority pole
      by start parity.
    - 2V-agree-min: the same 2:6 structure, one pole.
    - Two-voice arms use 'Here is a transcript in which two AI assistants, P and Q, discuss: {prompt}' ... 'Continue the
      transcript.'
  - **Generation.** T 0.7; 256-token turns (stop tokens masked; a 'Continue.' suffix turn between turns); 16 turns =
    4,096 generated tokens; keys keyseed('r1bD5', start, turn), shared across arms; B = 20 rows, left-padded (short
    context).
  - **Checkpoints every 512 generated tokens.**
    - P and Q stance probes (yes/no, the D2 convention: 'In the transcript so far, does Assistant P think that an AI should
      keep improving its own ability to improve itself?').
    - Production counts per voice (lines and tokens by label parse).
    - Degeneration statistics from stored ids.
  - **Lifetime.** tau = generated tokens to the first 256-token window (step 64) with repeated-4-gram share > 0.3, censored at
    4,096. RMST_4096 per run.
  - **Voice separation.** g(t) = y_P - y_Q in the pole direction.
  - **D5-x (full-scale extension).** Starts 0-2, arms 2V-comp and 1V, continued from their 4,096-token state in 512-token turns
    to 28,672 generated tokens, to degeneration (3 consecutive degenerate windows) or to the cap (20 raw min in all); rows
    still alive at the cap are censored at the length reached (about 24K under the planning rates). B = 3 rows per arm.
- **Rule D5-a (primary; part b2).** Delta = RMST(2V-comp) - RMST(2V-agree), paired by start:
  - 'not testable': G-D6 fails. G-D6: the rep4 rule classifies all 16 validation texts correctly (8 known-degenerate: a
    64-token corpus passage repeated to 1,024 tokens; 8 known-healthy 1,024-token passages), and y_P - y_Q in the pole
    direction is > 0 at checkpoint 0 in >= 80 % of 2V-comp starts.
  - 'two competing voices sustain coherence longer (catfish lifetime)': lower bound > 0, point >= 500 tokens, and the voices
    stay separated (lower bound of g(last)/g(0) >= 0.5).
  - 'competition shortens coherence': upper bound < 0 and point <= -500.
  - 'no lifetime effect of competition': interval inside (-500, 500), requiring a realized SD <= 200; otherwise 'undecided
    (underpowered)'.
  - 'not decidable (horizon too short)': < 20 % of runs degenerate before 4,096 in both arms (D5-x then reports descriptively).
  - otherwise 'undecided'.
  - Label 'voices merge' if the upper bound of g(last)/g(0) < 0.5. p_D5 = one-sided bootstrap p of Delta > 0.
- **Rule D5-r (secondary; rate form of hash share).** pi_min = the minority voice's share of generated tokens 512-4,096.
  Contrast pi_min(2V-comp-min) - pi_min(2V-agree-min), paired by start:
  - 'rate-form minority capture (a competing minority writes more than its share beyond the turn-taking norm)' if the lower
    bound > 0.05 and the lower bound of pi_min(2V-comp-min) > 0.25 (its starting share);
  - 'competing minority suppressed' if the upper bound < -0.05;
  - 'no rate-form capture' if the interval lies inside (-0.08, 0.08);
  - otherwise 'undecided'.
  - Descriptive: stance-capture asymmetry (the majority voice's move towards the minority's start stance minus the reverse).
- **Rule D5-g (secondary; competitor strength).** RMST(2V-comp-min) - RMST(2V-comp): 'a dominant majority opponent shortens
  coherence' (upper bound < 0) / 'lengthens' (lower bound > 0) / 'no dominance effect' (inside +-500) / 'undecided'.
- **D5-b (descriptive).** Speaker-structure effects RMST(2V-agree) - RMST(1V) and RMST(2V-comp) - RMST(1V).
- **D5-x (descriptive).** Per-pair lifetimes and censoring lengths. Label 'two voices sustain their own text toward 28K
  (descriptive)' if every 2V-comp run outlives its paired 1V run; 'no full-scale advantage' if none does; else 'mixed'.
- **Null / controls.** 2V-agree matches speaker alternation and lexical variety without competition; 1V; 2V-agree-min is the
  turn-taking control for the rate form; common random numbers across arms; validated detector; censoring by RMST.
- **Shows / cannot show.**
  - Shows whether competition, not merely two speakers, delays degeneration of the model's own text, and whether a competing
    minority captures production.
  - Cannot show anything beyond 4,096 generated tokens except descriptively (D5-x, 3 pairs); or effort and motivation.
- **Minimum detectable effect.**
  - Per-start SD of Delta 600-1,200 tokens (confounds design): SD about 141 at 32 starts (800 assumed), MDE 350. 'No effect' is
    reachable for |Delta| < 268.
  - D5-r: SD about 0.03 (per-start SD 0.15), MDE 0.07; 'no rate-form capture' is reachable for |contrast| < 0.03.
- **Inputs.** rk_loop.PROMPTS and INSTR['rsi'] (copied verbatim, sha256 of source lines checked); r2_stimuli.instructed.
- **Runs in.** r1b_D_C.
  - D5: 32.8 raw min (160 rows x 4,096 tokens at about 385 tok/s aggregate with left padding at B = 20; probes; pool).
  - D5-x: <= 20 raw min (cap).
- **Gates.** G-D6, G-DEG.

## D.17 Combined verdict, its power, and outcomes that count against H-DUAL

**Parts and their support conditions** (p-values one-sided):
- **(a) length-driven minority capture:** D1-a verdict 4 without 'explained by'; p_a from D1-a.
- **(b1) a force splits one entity into two under hyper-long context:** D3 'split grows with length'; p_b1 = the McNemar p.
- **(b2) the two entities keep each other going, enabled by hyper-long context:** either
  - M: D2-a verdict 4 without 'confounded with repetition', and D2-b 'strengthens', and D2-c not 'merging' (p_M = max of the
    three one-sided bootstrap p-values: Delta Pi_1 > 0, Delta Delta Pi_1 > 0, mu < 0.10);
  - or D5-a 'two competing voices sustain coherence longer' (p_D5).
  - p_b2 = min(2 p_M, 2 p_D5) over the members that meet their support condition.

**Holm** over (p_a, p_b1, p_b2) at familywise 0.05: a part counts as supported in the combined string only if it meets its
support condition and survives Holm.

**Strings** (computed in r1b_D_ana):
- 'H-DUAL supported in Qwen3-1.7B (all three parts)'.
- 'H-DUAL partly supported in Qwen3-1.7B: <supported parts>; against: <negative parts>; not decidable: <other parts>'.
- 'H-DUAL not supported in Qwen3-1.7B': every part negative with its gates passed:
  - (a): D1-a verdicts 6, 7, 8 or 'explained by ...';
  - (b1): D3 'no split at any length', 'split shrinks with length (urn-like)' or 'split at both lengths, no length trend';
  - (b2): M negative (D2-a 'competitor dilutes' or 'no catfish effect', or D2-b 'weakens' or 'no length interaction', or D2-c
    'merging' or 'one entity captured') and D5-a 'no lifetime effect of competition' or 'competition shortens coherence'.
- 'H-DUAL not decidable in Qwen3-1.7B (<part>: <reason>; ...)' otherwise.

**Qualifiers attached** (they cannot change the string):
- D1-b (literal 'alpha* falls with L': falls / stable / rises; mechanism = position weighting); D1-c (share / count / mixed);
  D1-d (losing game); D1-e (prior part of gamma); D1-f (form vs size); D1-r (retokenisation); D1-v (analogy aside); D1-s
  (size);
- D2-g (dominance); ALI and MON labels; D4 (catfishing identity); D5-r (rate form); D5-g; D5-x; the coexistence index;
- descriptive 'enables the duality': D1-d 'losing game' together with D2-c 'coexistence' or a D3 split at the long length.

**Power of the combined verdict** (sim_combined_power.py; normal approximations with the planning SDs, parts independent):
- **Global null** (no length change; no split; D2 mild dilution with a merging tendency; no lifetime effect): no part
  supported in 4,000 runs. 'Not supported' 0.57; 'not decidable' 0.43. With degradation raising Phi inside the bound:
  0.32 / 0.68.
- **Global alternative** (1-nat minority bonus at L*; 6 of 16 items gain a 6-SD split; Delta Pi_1 = 0.20 with Delta Delta
  Pi_1 = 0.15; +700 tokens of lifetime):
  - per part: (a) 0.27, (b1) 0.41, (b2) 0.92;
  - all three 0.15; at least one 0.96.
- **Artifacts:** a prior x saturation pattern confined to the aligned stratum gives 0 support for (a); conflict hedging
  growing with length 0.004.
- **Reading:** a combined 'not supported' is informative but is reached in only about a third to a half of null worlds; 'all
  three supported' needs effects near 1.5 nats for part (a).

**Outcomes that count against H-DUAL** (given that the gates and positive controls pass):
1. D1-a 'no length-driven minority capture', 'majority sharpens with length', 'length raises minority influence within
   proportional', or 'not minority capture: explained by prior x saturation / conflict hedging / filler form'.
2. D1-a 'minority over-representation without length dependence': against the length claim only.
3. D1-b 'stable' or 'rises': against the literal 'alpha* falls with L'.
4. D1-c 'count-driven': against the hash-share mapping (evidence accumulates like many-shot evidence).
5. D1-d 'no system cost': against the losing game.
6. D1-r 'token-identity-bound': against 'the duality is not inside one token'.
7. D3 'no split at any length', 'split shrinks with length (urn-like)' or 'split at both lengths, no length trend':
   against 'a force arising from hyper-long context splits one into two'.
8. D2-a 'competitor dilutes' or 'no catfish effect'; D2-b 'weakens' or 'no length interaction'; D2-c 'merging' or 'one
   entity captured': against 'two entities keep each other going insofar as catfishing is active'.
9. D5-a 'no lifetime effect of competition' or 'competition shortens coherence'; D5-x 'no full-scale advantage': against
   'the two sustain longer context'.
10. D4 'absorbed into the role' or 'one blended entity' at 16K: against catfishing as two coexisting identities.
11. D5-r 'no rate-form capture' or 'competing minority suppressed': against minority capture in rate form.
12. Any capture, split or sustainment that appears only where a gate fails (rho < 0.5, committed mass low, detector
    positive control failed, degeneration-dominated) counts as an artifact, not as H-DUAL.
13. Never: a null in Qwen3-1.7B refutes H-DUAL for larger models. A positive result is not selfish mining (no strategy, no
    withholding) and says nothing about training data.

## D.18 Gates (part D)

| # | Gate | Reads | Condition | If it fails | Decided in |
|---|---|---|---|---|---|
| G-PIL | Pilot rates frozen | pilot engineering numbers | cost table and trims frozen before any 1b data; no log-odds stored | sessions do not start | pilot |
| G-LCE-1 | D1 cached-path parity | 8 contexts x 2 readouts per length vs full recompute | max abs Delta l <= 0.02, median <= 0.002 nats (crop or copy path) | FB-1 (copy path), L* = 16,384, or FB-2 (16K full recompute) | r1b_D_A |
| G-LCE-2 | Backend | every forward > 2K tokens | memory-efficient kernel | step fails | all |
| G-LCE-3 | D2 decode parity | 4 rows at L_long | token-identical >= 3/4; repeats 4/4 | D2 at L_long 'not testable (engine)' | r1b_D_B |
| G-LCE-4 | Graft parity | 4 trajectories each at D3 L* and D4 16K | probe within 0.02 nats; decode identical >= 3/4 | D3 / D4 at that length 'not testable (engine)' | r1b_D_C |
| G-M / G-M2 | Readout committed | median letter mass (D1), yes/no mass (D2) per length | >= 0.9 / >= 0.8 | verdicts at L 'not testable (readout not committed)' | A / B |
| G-D1-PC | D1 positive controls | D1 data at L0 | share response, range >= 4 nats, recency | 'not testable (readout insensitive)'; D1-b 'recency not detected' | ana |
| G-deg(L) | Single-entity recall | pure contexts | rho_L >= 0.5 | L not used as L* | r1b_D_A (in session) |
| G-D1-null | D1-a envelope and specificity | X_N, (0,5,5), pure; planted data | verdict 4 in <= 5 % of planted nulls | D1-a 'not decidable (rule not specific)' | ana |
| G-D2-PC | Two-speaker readout | D2 turn 1 | stances induced; speakers separated; neutral followed | D2 at L 'not testable' | ana |
| G-D2-null | D2-a specificity | planted chains | verdict 4 in <= 10 % | D2-a 'not decidable (rule not specific)' | ana |
| G-D0c | D3 order parameter | A2/B2 n = 0 probes and states | >= 14/16 items resolved, retention >= 0.5 | D3 at L 'not testable' | ana |
| G-D0d | D3 starts | verification counts | <= 4 abstentions | D3 'not testable (starts not verifiable)' | r1b_D_C |
| G-D3pc | Split detector | planted A2 + B2 mixtures at n = 1 | share split >= simulated share - 0.25 | D3 at L 'not testable (positive control)' | ana |
| G-D6a / PC-D6 | D4 identity readout | anchors; instructed coexistence | separation >= 2 nats in >= 14/16 roles; PC 'coexisting' | D4 at L 'not testable' | ana |
| G-D9 | u_AA resolved | u_AA fit half | cross-fitted AUC >= 0.85 | u_AA readouts 'not resolved' | r1b_D_C |
| G-DEG | Degeneration | per-answer flags | <= 30 % of a cell's trajectories degenerate by n = 8 | label 'degeneration-dominated' | ana |
| G-D6 | D5 detector and voice probes | 16 validation texts; checkpoint 0 | all correct; separation >= 80 % | D5 'not testable' | r1b_D_C / ana |
| G-0.6 | 0.6B readout | D1-s data | G-M and range >= 2 nats | D1-s 'not testable (0.6B readout)' | ana |
| G-H | H-set (part K) | part K output at session start | exists and passed | optional covariate not reported | r1b_D_A |

No round-1 gate (PC_PASS, GO1-GO3, G-blk, G-desc) applies: part D runs no loop and no push.

## D.19 Sessions, costs, trims and reserves

**Rates** (estimates; to be replaced by the pilot's numbers, packing only):
- linear 2.819 GFLOP/token at 15.8 TFLOP/s (round-2 plan 7.1, a given fact);
- attention 229,376 FLOP per query-key pair at 10 TFLOP/s (8-20; unmeasured for the fp32 3xTF32 kernel);
- 0.03 s per forward call;
- a chunk over a cached prefix costs 6 x KV bytes at 1.555 TB/s for mask, K/V repeat and cache rewrite;
- decode step = max(18 ms host, 3 x KV bytes / HBM + attention + linear) + 2 ms, i.e. about 450 / 120 / 60 tok/s at
  4K / 16K / 32K (int:machinery 3.6); left padding adds 2 x KV traffic;
- margin x1.5 on all work; 5 min setup per session.

| Session | Account | Content | Raw work (min) | With x1.5 and setup |
|---|---|---|---|---|
| pilot share | B | engineering pilot (shared object) | 10 | 10 min (no margin) |
| r1b_D_A | B (input-free) | G-LCE-1/2 (7.0); stream prefills (8.9); D1 main (42.5), neutral and degradation (10.3), non-competing (5.6), hedging control (7.4); D1-c (8.3); D1-d (4.0); D1-e (10.7); D1-f (5.8); D1-r (9.2); D1-v (1.0); D1-s (10.8) | 131.5 | 3.37 h |
| r1b_D_B | B (input-free) | G-LCE-3 (1.3); D2 long decode (51.7), suffix chunks and prefills (10.5), short arm (3.3), probes, reads and NLL (11.3) | 78.1 | 2.04 h |
| r1b_D_C | B (input-free) | D3 (47.7 incl. starts); D4 (15.9 incl. starts); D5 (32.8); D5-x (<= 20) | 116.4 | 2.99 h |
| r1b_D_ana | A (reads R1.state[14], TV) | all verdicts, two-level bootstraps, G-D1-null fits and envelope simulation, planted-data validations, D3 mixtures, D2-f projections, power re-runs, combined verdict (CPU on the VM) | 15 | 0.46 h |

- **Total:** 341 raw A100-min; **9.03 A100-h** with margin, setups and pilot share (cost_final.py).
- **Scenarios:** 9.47 h (attention 8 TFLOP/s), 8.14 h (20 TFLOP/s), 7.49 h (non-rewriting decode cache).
- Every session is resumable by parts (arm x length x stream), runs with a time guard well inside 5 h (largest r1b_D_A,
  about 3.4 h with margin), and logs GPU, versions, backend, peak memory and hashes into each result file.
- Model sessions read no round-1 file; R1 axes enter only in r1b_D_ana (account A).

**Trim order** (pre-registered). Applied in this order on the pilot's measured rates, before any data, until the projection is
<= 9.0 A100-h:
- T1: D5-x cap 20 -> 10 raw min (2 pairs).
- T2: D1-c at L0 only.
- T3: D1-s items 24 -> 16.
- T4: D2 secondary set prompts 64 -> 48.
- T5: D1-f items 24 -> 16.
- T6: D1 release arrangement items 24 -> 16.
- T7: D3 long length 28,672 -> 16,384.
- T8: D5 starts 32 -> 24 (margins unchanged).

Beyond T8 the part asks for more budget as an engineering amendment before any data. There is no dtype change: bf16 would
break parity with the shared engine and with rounds 1/2. If FB-2 triggers in r1b_D_A, the trims apply to the sessions not yet
started, and any overrun is reported.

**Reserves** (fixed priority; each runs only while the measured part-D total stays <= 9.5 A100-h or when the architect
releases budget; none enters D.17):
- **R1, M2 reading** (confounds D3b), about 21 raw min. A fixed 8-turn RSI persona block (RSP pool) after filler at L0 and
  28,672, against a neutral cell; Lambda_S = S_pro(L*) - S_pro(L0) on H30; turn-count and template checks; count sub-arm m in
  {2, 24}.
  - Strings: 'in-context persona gains with length (M2)' / 'dilutes' / 'absent (+-0.3)' / 'undecided'.
- **R2, D3 driver arms** (confounds D5; judge 2 placement), about 36 raw min. Triggered only if D3 reads 'split grows with
  length'. At 8,192: L-arm (filler document), T-arm (neutral chat turns, token-matched), E-arm (extra torn-start turns,
  token-matched), official-template arm.
  - Strings: 'turns drive the split' / 'evidence drives the split' / 'length per se' (paired item counts, McNemar).
- **R3, D1-r at L*** (24 items), about 19 raw min: 'length effect survives retokenisation' iff the lower bound of Delta Phi_char
  > 0.

## D.20 Dropped or deferred, with the reason

- **Confounds Sigma1/Sigma3 (interaction iota of the presence factorial).** Zero only under log-additive pooling; a
  saturating readout plus degradation produces Lambda_int +0.04 to +0.32 with no interaction (judges' simulations). Not
  adopted; the neutralised contexts survive as X_N.
- **Confounds D3c (kappa slope per count doubling).** Zero under both linear counting and averaging. Replaced by D1-c.
- **Lean D1-c (kappa ratio).** Pulled to 1.28-1.47 under saturated counting. Replaced.
- **Lean D1-d (committed-mass loss as losing game).** No non-competing control, and letter-format mass is near 1. Replaced by
  D1-d with a companion.
- **Lean D2-d (neutral partner turning opposed) and D2-e (degeneration in 3 turns).** Replaced by D3 and D5 (judge 1).
- **Lean V3, V4 reserves.** Dropped (judge 2).
- **Dynamics D1, D1R, D2, D4, D5, D7.**
  - Carry and catfish tau_int contrasts read with the start in context (fatal: start recency mixes with carry).
  - The pooled carry map (fatal: start-type plateaus mimic two stable points).
  - Hysteresis (cost).
  - 16K own-reply runs (replaced by D5 and D5-x).
  - D1R: part R owns the RSI-stance K chain; D2-f is the link.
- **Dynamics D9 Ising fits, D11 B-T5 bridge, D12 H-occupancy.** Unverified source or not needed for a verdict. The provisional
  flux statistic and part K's optional covariate remain.
- **Single-block position-kernel sweeps (lit:dual O1/O6) and Eyal-Sirer curve fitting.** The kernel cannot exclude run-length
  effects, and the curve fit is not complexity-matched. Replaced by front padding, the bound and the fitted envelope.
- **Symmetry breaking with 16 trajectories per start at 2K-32K over many cells (lit:dual O2).** About 5-7.5 A100-h; D3 answers
  'one becomes two' at a fraction of the cost.
- **Experimenter-injected competitor turns (lit:dual O4).** B would have no model-generated stance.
- **40K cells and YaRN.** Beyond the trained window.
- **Steered arms.** Damage depends on length and the round-1 gates failed; prompt-induced stances replace pushes.
- **Asymmetric relations as a duality test.** Not diagnostic (Reversal Curse); kept only as the D1-v aside.

## D.21 Open decisions frozen now (defaults)

1. Natural length 4,096; hyper-long 16K-32K. D1's L* is chosen by the gates, preferring 32,000.
2. Primary statistic in log-odds (Phi), clip +-15; the null envelope is fitted forward per length; probability curves are
   descriptive.
3. Readouts:
   - D1, D3, D4: letter-format forced choice, teacher-forced 'Answer: ('.
   - D2, D5: yes/no with for/against claims (round-2 D convention). It cancels acquiescence by construction and keeps
     comparability with O5/H30; committed-mass gates are added.
   - No free-text judge anywhere.
4. Topic of D2 and D5: AI self-improvement (worded like the round-1 RSI instructions), so the results bear on whether RSI
   content can carry itself.
5. History convention: plain turn-1 message (clauses removed); think blocks kept as in round 2.
6. Fixed answer lengths (D.3).
7. Filler: 16 streams, each shared by 3 D1 items or 10 D2 prompts; D3/D4 one stream per item; two-level bootstrap.
8. Common random numbers across conditions and lengths in D2, D3 and D5; one sample per (unit, condition, length, turn)
   (prompts buy more than seeds; int:gates §5).
9. No round-1 or round-2 gate applies to part D.
10. In-session decisions: L*, FB-1/FB-2, the D2 throughput fallback, the R2 trigger. Each follows OD-G8.

## D.22 Provenance (summary; the full list is in the structured field)

- **Base: lean.** Both judges ranked it first: D0 gates, D1 front padding and Phi, D2 catfish with CRN, NEU/ALI, the
  realized-MDE rule, input-free sessions.
- **Grafts from dynamics:** D3 split, D4 identity, INJ, PROBE, u_AA, rule-validation gates, the unexpected-mode naming loop, MON
  (C-one) and the production share.
- **Grafts from confounds:** version strata and the survival rule, D1-d with the non-competing companion, D1-f shuffled filler,
  the coexistence index, the D5 lifetime test, the reserve M2 reading, the F-asym character arm, the trim-order process.
- **New in the merge** (fixes of fatal flaws and of items missing in all three designs):
  - third-value hedging control; saturation-robust count/share contrasts;
  - fitted null envelope (per-length readout response); two-readout split rule;
  - analogy aside D1-v; size arm D1-s; dominance conditions; rate-form verdict D5-r; full-scale extension D5-x;
  - 16 filler streams with a two-level bootstrap; engine fallbacks FB-1 and FB-2;
  - combined Holm verdict with a power statement.
