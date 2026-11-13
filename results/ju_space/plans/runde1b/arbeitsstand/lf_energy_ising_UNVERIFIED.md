# lens-free research, angle energy_ising (RESEARCH STAGE, NOT YET VERIFIED)

# Angle 1: a lens-free route along the energy, non-equilibrium and Ising chain

*Scope: energy-based reasoners, attention as a Hopfield/Potts update, looped transformers, non-equilibrium measures, LLM agents as spins, and maximum-entropy models of neural populations. From these, lens-free and inverse-free measurements for Qwen3-1.7B (28 blocks, 28 × 6,144 = 172,032 MLP neurons according to the downloaded config; one A100-40GB; no training).*

*Inputs read in full:*
- `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/brief_shared.md`
- `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/design_rules.md`
- `/home/user/DSSE/results/ju_space/runde2_ergebnisse/AUSWERTUNG.md`
- the chain-lens post (`.../scratchpad/chainlens_dl/post.json`)
- saved arXiv HTML copies of the H-neuron paper and its replication (`.../scratchpad/hchain/dl_hneurons/page.txt`, `.../scratchpad/hchain/dl_hneur_repl/page.txt`)

*Every external source in `refs` was opened and read: at least the abstract, and method sections where stated. My own inferences are marked as such.*

## 1. Short answer

Yes, but only with a narrow meaning of "holographic". In this chain the word has one operational meaning: associative memory, i.e. distributed storage plus clean-up of noisy or partial cues into attractors. Plate (1995) calls convolution/correlation memories "closely related to holographic storage". He cleans their noisy reconstructions with "a separate associative memory". Hopfield and Ising models are the energy-landscape form of the same idea.

A lens-free programme can therefore test what the user hopes to learn about H-neurons without any readout into the vocabulary:
1. Is the hallucination-associated population an attractor, or does it only track length?
2. Does a context held far above natural length split one "entity" into two?
3. If it does, is the process driven, i.e. irreversible?
4. Is the H-state stored locally or in a distributed ("holographic-like") way?

What it cannot do is say what a state means in words. Where content is needed, the model's own generation supplies it inside a generate–verify loop (section 6).

**Why lens-free matters here.** The chain lens confabulates more as chains grow (19 WorkspaceBench families). Its token scores are contrastive: log p(state) − log p(prior), i.e. an energy difference.

My inference, untested: if the state-specific part of that gap shrinks along the chain while the MTP head's language prior takes over, the readout produces a length effect by itself. A length dependence of chain-lens confabulation can then not be attributed to H-neurons without an independent measurement. Activation statistics plus behaviour, with position-matched controls, provide one.

**New relative to the literature, as far as my searches reached:**
- I found no pairwise maximum-entropy (Ising) fit to the activations of transformer MLP neurons. The existing "Ising models of deep networks" map weights to couplings.
- H-neurons have only been studied as sparse probe features. The 2026 replication finds them to be representatives of correlated clusters, which is exactly what population models are for.
- The work on LLM agents as spins has never tied the coupling to an identified neuron population. H-neurons are a natural candidate, because Gao et al. link them causally to over-compliance and sycophancy, i.e. conformity.

## 2. What the literature supports, link by link

### 2.1 "EBM reasoner"
- **Energy-Based Transformers** (Gladstone et al. 2025)
  - An energy is assigned to (input, candidate) pairs, and prediction is gradient descent on it "until convergence".
  - The abstract claims up to 35 % higher scaling rate than Transformer++ and 29 % more gain from "System 2 thinking" on language tasks.
- **IREM and IRED** (Du et al., ICML 2022; Du, Mao, Tenenbaum, ICML 2024)
  - Reasoning is minimisation of a learned energy.
  - IRED learns annealed landscapes and adapts the number of optimisation steps to difficulty.
- **Energy-Based Reasoners** (Chen, Gladstone, Li, Ji, Du; arXiv 2611.08977, 6 Nov 2026) are probably what the user means.
  - The loop of a recurrent reasoner becomes gradient descent on a learned energy over latent states: z_{t+1} = z_t − α∇_z E_θ(z_t; x). In practice it runs as noisy projected Langevin dynamics.
  - Inference stops when the projected gradient norm falls below τ (0.2 on Sudoku).
  - During training, K parallel trajectories shape several modes.
  - They note that HRM "can converge to both correct and incorrect fixed points". This is the clearest statement in this literature that looped reasoners have wrong attractors.
  - The models are tiny: 1.8 M and 7.1 M parameters (Sudoku-Extreme 99.4 % with one sample).
- **Related systems**
  - ERM (arXiv 2607.10128) scores candidate trajectories with Hopfield-type memories and uses parallel tempering.
  - EBM-CoT (arXiv 2511.07124) refines latent chain-of-thought with an EBM and leaves the base LM unchanged.
  - Kona 1.0 (Logical Intelligence) is sold as an "energy-based reasoning model". Its Sudoku result of 96.2 % is reported by the vendor only; I found no paper.
- **LM likelihood as an energy**
  - Blondel et al. (arXiv 2512.15605) give an explicit bijection between autoregressive models and EBMs in function space, as a special case of the soft Bellman equation.
  - Contrastive decoding (Li et al., ACL 2023) scores log p_EXP − log p_AMA inside a plausibility set with α = 0.1. That is an energy difference.
  - The chain lens uses the same construction against a mean prior state.

**Consequence.** −log p(text) is a legitimate energy for Qwen, but Qwen never descends it: each token gets one fixed feed-forward pass. The only time-homogeneous iterations are:
- the token loop,
- the round-2 text loop (push → generate → read),
- agent loops.

Attractor claims must be made on these axes, not across depth.

### 2.2 Attention as an energy update; tokens as particles
- **Ramsauer et al. 2020**
  - The modern Hopfield update equals attention.
  - It has three kinds of fixed points: a global average, "metastable states averaging over a subset of patterns", and single patterns.
  - Early layers tend toward global averaging; higher layers toward partial averaging "via metastable states".
- **Energy Transformer** (NeurIPS 2023): attention layers designed to minimise an engineered energy. Standard LMs are not built this way.
- **Rende, Gerace, Laio, Goldt** (PRR 2024)
  - One layer of factored self-attention learns the conditionals of a generalized Potts model.
  - Masked-LM training "is exactly equivalent" to solving the inverse Potts problem by pseudo-likelihood.
  - This also licenses pseudo-likelihood (forward regressions) as the inversion-free fitting method used below.
- **Tokens as interacting particles** (Geshkovski, Letrouit, Polyanskiy, Rigollet; NeurIPS 2023, Bull. AMS 2025)
  - With weights fixed over time, tokens cluster.
  - Cluster locations depend on the initial tokens; the kind of limit depends on the spectrum of the value matrix; "leaders" appear.
  - Before merging, tokens stay trapped near several clusters for exponentially long times, with saddle-to-saddle dynamics (2024).
  - The metastable structure is selected by the inverse temperature β (Bruno et al. 2024).
- **Caveats for real models**
  - Skip connections and MLPs prevent the rank collapse of pure attention (Dong et al. 2021).
  - Massive activations, about 10^5 times larger than the rest and input-independent, concentrate attention on their tokens (Sun et al. 2024).

### 2.3 Looped transformers
- A looped 13-layer transformer can emulate a small computer (Giannou et al. 2023).
- **Recurrent-depth LMs** (Geiping et al. 2025; 3.5 B parameters, 800 B tokens)
  - Latent states converge to fixed points, fall into orbits (on harder questions), or drift as "sliders".
  - The pattern depends on context, not on initialisation.
  - The per-token exit rule is a KL between successive steps below 5×10^-4.
- **Ouro LoopLMs** (1.4 B / 2.6 B, 7.7 T tokens) learn their depth allocation.

None of these depth results transfer to Qwen3-1.7B, whose blocks have distinct weights.

Round 2 measured the text loop instead:
- Gain G was 0.12–0.30 and carry K 0.45–0.48, along a shared style direction (the "generic attractor").
- The only carry estimate on the RSI axis, 1.053 [0.743, 1.538], leaves self-sustainment open.

### 2.4 Non-equilibrium
- **Arrows of Time** (ICML 2024): forward and backward log-perplexity differ consistently in large models, although information theory predicts no difference.
- **Sagawa** (PRX Intelligence 2026) defines entropy production for autoregressive models.
  - Definition: S = D_KL(P→‖P←), where the backward path reuses the same kernels on the reversed sequence.
  - Estimator: σ = Σ ln p_fwd − Σ ln p_bwd, i.e. two likelihood passes.
  - At token level, σ is "dominated by a syntactic artifact".
  - At sentence level (sentence order reversed), σ is larger for causally ordered texts. With Qwen3-4B-Base as evaluator this is significant for all five corpora.
  - σ splits into non-negative per-step terms: compression loss and model mismatch.
- **Halperin (2025)** reports that higher faithfulness goes with lower "semantic entropy production". The thermodynamics there is an analogy built on topic transition matrices.
- **Estimators for state trajectories**
  - Roldán and Parrondo (PRL 2010): the time-reversal asymmetry of one stationary series bounds dissipation, even without visible currents.
  - Lynn et al. (PNAS 2021): Ṡ = Σ P_ij ln(P_ij/P_ji) on hierarchical k-means states (cosine distance, k = 8). Coarse-graining gives a lower bound; errors come from a trajectory bootstrap; the noise floor from shuffles.
  - Their brain data nearly obey detailed balance at rest and break it under load. A kinetic Ising model with asymmetric couplings reproduces this.

### 2.5 Agents as spins, collaboration, entities
- **Ashery, Aiello, Baronchelli** (Sci. Adv. 2025)
  - Naming game: 24 agents, 10 names, memory 5, payoffs +100/−50.
  - The population goes through a disorder-to-order transition to one convention, described as "symmetry breaking".
  - Collective bias appears without individual bias.
  - Committed minorities overturn a convention at 2 % (Llama-3-70B-Instruct) up to 67 % (Llama-2-70b-Chat).
- **De Marzo, Castellano, Garcia** (Sci. Adv. 2026): a "majority force coefficient", and a critical group size that grows with model capability.
- **De Nobili** (2026): LLM spins on a 2D lattice show crossovers, but "bias dominates" coupling. The result is field-driven crossovers, not phase transitions, with exponents incompatible with 2D Ising.
- **Okawa** (ICML 2026): theory predicts a transition to collective bias above a conformity threshold; experiments show finite-size crossovers; heterogeneity smooths them.
- **Tanaka** (2026): consensus through "memetic drift" from mutual in-context learning, with scaling laws for polarization.
- **Piao et al.**: polarization and homophilic clustering in networks of thousands of LLM agents.
- **Riedl** (ICLR 2026) tests whether a group is a "higher-order collective".
  - Criterion: S_macro(ℓ) = I(V_t;V_{t+ℓ}) − Σ_k I(X_{k,t};V_{t+ℓ}), after Rosas et al. 2020, plus pairwise synergy.
  - Assigning personas produces "identity-linked differentiation".

**Consequence.** "Collaboration → entity" means, in this literature, symmetry breaking into one convention. Two persistent groups require:
- weak coupling,
- heterogeneity,
- committed minorities, or
- antagonistic coupling between the groups.

Field (bias) and coupling must be separated before a "transition" is claimed.

### 2.6 Maximum-entropy models of neural populations
- **Schneidman et al.** (Nature 2006)
  - Weak pairwise correlations coexist with strongly collective states in groups of ten or more neurons.
  - Pairwise (Ising) models capture these states.
  - The authors suggest the code may be error-correcting.
- **Tkačik et al.** (PLoS CB 2014)
  - Studied up to 120 neurons.
  - Above 40 neurons a global "K-pairwise" term is needed.
  - Activity patterns group into metastable collective modes.
  - Each neuron is "highly predictable from the rest of the population".
- **Tkačik et al.** (PNAS 2015)
  - With P_T ∝ exp(−E/T), the specific-heat peak grows and moves toward T = 1 as N grows, and S ≈ E.
  - T is "not a physical temperature".
  - The peak is already clear at N = 100 with one tenth of the data, about 3.6×10^4 bins of 20 ms by my arithmetic.
- **Mora and Bialek** (2011): models inferred from data sit near critical points.
- **Pitfalls**
  - A hidden common variable produces Zipf-like statistics without any tuning (Schwab, Nemenman, Mehta 2014).
  - With positive mean correlations and a couple of hundred units, pairwise models turn bimodal, and Glauber-based Boltzmann learning becomes non-ergodic (Rostami et al. 2017).
  - Static maximum-entropy models cannot separate critical from supercritical systems (Simões et al., PRE 2026).
- **Artificial networks:** Stosic et al. (2022) and Barney, Winer, Galitski (2024) map weights to couplings; in the latter, training selects a symmetry-broken "hidden order". I found no fits to activations.

### 2.7 H-neurons
- **Gao et al. 2025**
  - Method: CETT_{j,t} = ‖h_t^(j)‖/‖h_t‖ on answer tokens (non-answer tokens serve as control features), then L1 logistic regression.
  - Data: TriviaQA, 10 samples per question at T 1.0, top-k 50, top-p 0.9.
  - Fewer than 0.1 % of neurons are selected.
  - Scaling them by α ∈ [0, 3] shifts over-compliance: false premises, misleading context, sycophancy and jailbreaks.
  - They are already predictive in the base models.
- **Cavus et al. 2026** (Gemma 3 4B, MedGemma 4B)
  - Detection and causal effects replicate.
  - But 19 of 22 selected neurons have |r| > 0.7 with other features.
  - Bootstrap selection is only moderately stable, and sparse and dense rankings overlap weakly. The authors call the set "a sparse projection of a larger correlated structure".

In Qwen3-1.7B, 0.1 % of all MLP neurons is 172.

## 3. The chain as measurable quantities

| Link | Counterpart in Qwen3-1.7B | Lens-free measure | ID |
|---|---|---|---|
| EBM reasoner | −log p(text) (exact ARM–EBM bijection); fitted energy E(s) of a neuron population | E along generation; basins | O1, O7 |
| Non-equilibrium | irreversible generation; coarse-grained state dynamics | σ (text); Ṡ and cycle flux (states); kinetic-Ising asymmetry; forward/backward predictability | O7 |
| Loop transformer | not depth; the token loop, the round-2 text loop, agent loops | hazard and dwell; push–release persistence τ_H, carry K_H | O2, O3 |
| Agent | separate Qwen instances with memory | naming game | O4 |
| Ising-modellable | pairwise/K-pairwise model of the H-population; agent response curves | J, h, basins, C(T); β_J, h_X | O1, O4 |
| Collaboration | coupling-driven symmetry breaking | consensus time, tipping fraction f_c, payoff | O4 |
| Entity-forming | two anti-aligned communities; metastable token clusters; macro self-prediction | P(Δ) bimodality and dwell; cluster persistence over blocks; S_macro | O5, O6, O8 |
| Holographic | distributed storage plus clean-up | recovery from random subsets; completion from a clamped "boundary"; graceful degradation | O9 |
| Chain-lens length effect | the model's own rollout as the "chain" (no transported state) | H-occupancy against rollout length, with position-matched controls | O2 |

**The user's "vector quantity"** (does a front predict later or earlier fronts?) needs care.

For a stationary process, one front predicts its successor exactly as well as its predecessor in information terms: H(X_{t+k}|X_t) = H(X_{t−k}|X_t). A non-zero ratio therefore has three separate possible sources:
1. non-stationarity (drift with position);
2. path irreversibility (P(i→j) ≠ P(j→i));
3. computational asymmetry between predictors of equal capacity (the Arrows-of-Time kind).

O7 measures the three separately. The user's "hierarchy of channels" becomes the asymmetric part of kinetic-Ising couplings between communities.

## 4. The decisive lens-free test: H-attractor or length artefact

| Signature | Attractor predicts | Length artefact predicts | ID |
|---|---|---|---|
| History dependence at fixed position, HD = logit P(H_{t+1}\|H_t) − logit P(H_{t+1}\|¬H_t) | above the distribution of 50 control populations after a position spline | inside it | O2 |
| Stickiness (exit hazard vs. dwell time) | hazard falls with dwell | flat, as in controls | O2 |
| Hysteresis after a 16-token push and release | persistence (τ_H, K_H) above damage-matched random-neuron pushes, also in a cache-free re-read | relaxes like controls | O3 |
| Landscape | a distinct basin whose occupancy predicts hallucination beyond answer length | no basin beyond controls | O1 |
| Irreversibility | entry into H changes Ṡ / σ | no change | O7 |

**Proposed rules:**
- "H-attractor beyond length" needs O2 (HD or stickiness) **and** O3, each above the 95th percentile of its controls.
- O1 alone gives "collective H-code without attractor dynamics".
- Occupancy that rises with rollout length without the dynamic signatures gives "length artefact".
- Persistence that appears only in the live run, not in the re-read, gives "carried by the KV cache".

**Controls shared by all tests:**
- 50 random-neuron populations with the same layer histogram, each unit binarised at its own 95th percentile on reference text and read on the same tokens. This makes position matching automatic; the smallest empirical p is 1/51.
- Degenerate repetition (4-gram rate) as a competing attractor, reported separately.
- Correct and hallucinated answers matched on length.
- The round-2 generic style direction as a covariate. Its vector is not in the repository; `theta_verdict.json` holds only cosines, so the vector must come from Drive or from one B-Q0 round.
- Planted Ising data with known couplings and basins, to measure the pipeline's false-positive rates.

## 5. Bridge to H-DUAL

**Two entities.** In Ising terms these are two communities with anti-aligned magnetisation and long dwell times. A single ferromagnetic population never produces them (2.5).
- O5 measures Δ = m_A − m_B in single-voice texts at 1k–16k tokens against sentence-shuffled texts of the same length, with two-voice texts as the readout's positive control.
- O6 asks the same across depth: do token states sit in two metastable clusters for many blocks?
- O8 asks whether such a community behaves as an "entity" in the Rosas/Riedl sense.

**Catfish.** "A competitor keeps the group active" becomes a sustained net cyclic flux between two macrostates, i.e. broken detailed balance (O7). Its behavioural correlate is a longer coherence lifetime of self-generated text from two-voice seeds (O5).

**Selfish mining and the "losing game".** The 1/3 threshold has no counterpart inside the model. The measurable analogue is the committed-minority tipping fraction f_c (Ashery: 2–67 % across models). H-DUAL then predicts three things:
1. f_c < 0.5 (a minority can impose its convention);
2. mean payoff falls while two factions coexist (the losing game);
3. longer memory ("context above natural length") makes two-faction states more persistent.

O4 tests these with memory lengths 5, 20 and 60.

**H-neurons as coupling.** If H-neurons carry over-compliance, they should set the coupling of a Qwen agent population:
- amplification should raise the conformity slope β_J, shorten consensus and lower f_c;
- suppression should favour persistent factions.

O4 is the one experiment in which the user's whole chain (loop → agent → Ising → collaboration → entity) meets H-neurons.

## 6. How the lens-free and inverse-free rules are applied

**Fitting.**
- Couplings are fitted by pseudo-likelihood: one logistic regression per unit on all others, i.e. a forward regression. Rende et al. show that MLM attention training solves the same problem.
- Refinement uses forward MCMC moment matching.
- Excluded: naive mean field (J = −C^{-1}), TAP and Sessak–Monasson corrections, graphical-lasso or other precision-matrix estimates, and Tikhonov-regularised inverses.

**Other forward procedures.**
- Basins: zero-temperature single-flip descent.
- Communities: power iteration on J.
- Kinetic-Ising fits, recoverability probes and forward/backward state predictors are fitted forward maps, which the design rules allow.

**Reconstructing what a basin or community means** uses an Aletheia-style loop:
- **Generator:** clamp the units to the basin pattern in neutral prompts and sample.
- **Verifier:** re-read the output with the unmodified model. Does the population land in the same basin, above a random-basin-clamp null? Verdicts: correct, minor fix (lands in a neighbouring basin), or critically flawed.
- **Reviser:** change the clamp strength or the prompt family.
- **Stop:** abstain after 3 attempts and count abstentions.
- **Verifier error rate:** measured on known answers, namely planted basins, the round-1 SEA topic push, and the two-voice texts.

**Lens-free means:**
- no logit, tuned, J or J++ lens;
- no MTP head;
- no projection of intermediate states to the vocabulary.

Text likelihoods come from the model's own output on its own input.

## 7. Budget

Rates are those of the round-2 plan §7.1, in fp32 with TF32 off:
- decode step at batch 32: 25 ms, ×1.15 with hooks;
- teacher-forced reading: 5,600 tokens/s through 28 blocks;
- attention FLOPs at 16k context: about 1.7× (my estimate).

| Item | A100-min |
|---|---|
| H-set identification, if not delivered by H-CHAIN | 20–40 |
| O1 + O2 shared rollouts (2,000 prompts × 2 T × 384 tokens = 1.5 M tokens) | 23 |
| O1 fits (H-set, 50 control populations, 5 planted data sets), MCMC, C(T) | 25–40 |
| O2 analysis | 5–10 |
| O3 (minimal / extended grid) | 20 / 50 |
| O4 | 45–60 |
| O5 (including O6 hooks); O6 analysis | 45; 10 |
| O7 | 15 |
| O8 (CPU) | 15 |
| O9 | 20 |

- Total: about 4–4.5 A100-hours.
- Core for H-CHAIN: O1–O3, about 1.5–2 h.
- Bridge to H-DUAL: O5, O7, O8, about 1.2 h.
- Optional: O4, O6, O9.
- Memory: 16k-token reads need the memory-efficient SDPA kernel in fp32. The KV cache for sliding-window generation at 2k × 32 sequences is about 15 GB.

## 8. Outcomes that count against the hypotheses

**H-CHAIN:**
- HD, stickiness (O2) and persistence (O3) all fall inside the damage- and position-matched control distributions, even if occupancy rises with rollout length. That is a length artefact.
- The H-population is no more collective than random populations (O1).

**H-DUAL:**
- Up to 16k tokens, single-voice texts show no more bimodality of Δ and no longer dwell times than shuffled controls of the same length (O5).
- There is no net cyclic flux above the shuffle floor (O7).
- In agent populations, f_c ≥ 0.5, or f_c and two-faction persistence do not change with memory length (O4).

**Holographic:**
- Random non-H subsets recover the H-state only at chance level, even at k = 1,000.
- Clamping 50 % of the H-units completes the pattern no better than a damage-matched random clamp does (O9).

## 9. Unverified or not found

**Unverified:**
- Kona: vendor page only.
- ICML 2026 venue of Blondel et al.: seen only in a search listing.
- Rosas' Ψ: read only as quoted by Riedl.
- De Nobili's method for extracting J̃ and h̃: abstract only.
- Gurnee et al. ("universal neurons"): opened, but whether they report position neurons was not confirmed. Position-tracking units therefore remain an unverified confound, handled by design.

**Not found:**
- a maximum-entropy fit to LLM neuron activations;
- a fluctuation theorem applied to LLM token generation (Sagawa's framework is the closest);
- an analysis of H-neurons in Qwen models. Their existence in Qwen3-1.7B is the upstream gate (G-H in O1).


## operationalisations
```json
[
 {
  "name": "O1 ME-H: pairwise maximum-entropy (Ising) landscape of the H-neuron population",
  "what_it_decides": "Whether the H-population is a collective code or a set of independent detectors. A collective code means pairwise couplings beyond independence and a distinct low-energy basin whose occupancy predicts hallucination beyond answer length. It also decides whether any collective structure is H-specific, compared with 50 random populations read on the same tokens. Proposed verdicts: 'collective H-code' / 'independent H-detectors' / 'generic population structure (not H-specific)' / 'not testable' (gate G-H or G-M failed). Criticality signatures are descriptive only. Shows: the static statistical structure. Cannot show: dynamics or causation. Minima of a fitted energy are not attractors of the network, which does not run Glauber dynamics on E.",
  "measurement": "Population: the H-set from the H-CHAIN part. It is built with the Gao et al. pipeline on Qwen3-1.7B: TriviaQA questions disjoint from the O1 prompts, 10 samples per question at T 1.0 / top-k 50 / top-p 0.9, answer-token CETT, L1 logistic regression. It is extended by correlated partners (|r| > 0.7 on CETT, the replication's criterion) and ranked by selection frequency over 20 bootstrap L1 refits, with N <= 200; 172 neurons = 0.1 % of 172,032. Gate G-H (proposed): held-out AUROC of the H-probe >= 0.70 and >= 0.10 above size-matched random-neuron probes. Spins: s_jt = 1 if CETT_jt exceeds unit j's 95th percentile on 200 neutral reference passages x 512 tokens; sensitivity runs at the 90th and 98th percentiles. Data: 2,000 prompts (1,000 closed-book questions with known answers, 500 fabricated-entity questions, 500 open prompts) x 2 temperatures (0.7 Gumbel-max as in rounds 1/2; 1.0 as in Gao) x 384 new tokens, chat template, thinking off; split by prompt into fit and test halves. Fits: (1) independent model; (2) pairwise model by pseudo-likelihood, i.e. one L2-regularised logistic regression per unit on all others, J symmetrised; (3) K-pairwise term V(K) on the synchrony count; (4) optional Boltzmann-learning refinement with forward MCMC. Gate G-M (model verification on held-out prompts): the pairwise model must beat the independent model in held-out log-likelihood per token and reproduce P(K) and triplet correlations within the data's 90 % prompt-bootstrap band; otherwise 'not testable'. Landscape: zero-temperature single-flip descent from every observed pattern gives basin labels. Keep basins with >= 1 % occupancy. On answer tokens, regress hallucination (logistic) on basin occupancy with answer length and position as covariates; the basin with the largest positive log-odds on the fit half is the H-basin. Criticality (descriptive): C(T)/N on 21 points of T in [0.5, 2] at subnetwork sizes 20, 40, 80, N (10 random subsets each); signature = peak per neuron rising with N and moving toward T = 1; Zipf slope of codeword ranks. Intervals: 90 % prompt bootstrap, 2,000 replicates.",
  "null_and_controls": "(1) Independent model. (2) 50 control populations of random MLP neurons, same size and layer histogram, binarised identically and read on the same tokens; this matches position and length automatically. H statistics are ranked within the control distribution (smallest empirical p = 1/51). (3) Latent-variable control after Schwab et al. 2014: refit with prompt-level fields and position-binned fields h_j(bin). Couplings or criticality signatures that vanish are labelled as driven by shared input. (4) Rostami check: a bimodal fitted P(K) counts only if the empirical P(K) is bimodal too; otherwise refit with an inhibitory reference measure. (5) Five planted Ising data sets with known J and basins (N = 200, same sample count and autocorrelation) give the pipeline's false-basin and false-criticality rates, i.e. the verifier error rate on known answers. (6) Correct vs hallucinated answers matched on answer length.",
  "data_needed": "N <= 200 H-units plus 50 x N control units. About 1.5e6 generated tokens (2,000 prompts x 2 temperatures x 384), shared with O2. 1e5 reference tokens for thresholds. 5 planted synthetic data sets of 7.5e5 samples each. Pair coverage rule (my assumption): expected co-activation p_i p_j T >= 10 needs T >= 4,000 at p = 0.05. Tkacik et al. 2015 report a clear specific-heat peak at N = 100 from about 3.6e4 time bins. The fit half here has about 7.5e5 tokens, which leaves margin for autocorrelation.",
  "cost_estimate": "Generation: 1.5e6 tokens = 48,000 decode steps at batch 32 x 25 ms x 1.15 (hooks) = about 23 min, shared with O2. Reference reads: 1e5 tokens at 5,600 tok/s = about 0.5 min. 56 pseudo-likelihood fits (H, 50 controls, 5 planted), batched on the GPU: about 15-25 min. Forward-MCMC verification and C(T) scans: about 10-15 min. Total about 50-60 A100-min, fp32, TF32 off, rates from round-2 plan 7.1. If the H-set must be identified here: +20-40 min.",
  "lens_free": "Spins are norm ratios of MLP contributions (CETT); labels are answer correctness from text. No hidden state is projected to the vocabulary; no logit/J/tuned lens or MTP head is used.",
  "inverse_free": "Couplings come from pseudo-likelihood, i.e. forward logistic regressions of each unit on the others; Rende et al. 2024 show that MLM attention training solves the same problem. Refinement is forward MCMC moment matching. Excluded: naive mean field J = -C^-1, TAP, Sessak-Monasson, graphical lasso or other precision-matrix estimates, Tikhonov-regularised inverses. Basins come from forward descent; verification is forward sampling compared with held-out data.",
  "pitfalls": "(1) Non-stationary firing rates along generation create spurious couplings; position-binned fields are the control. (2) Shared latent inputs (topic, uncertainty, position) give Zipf and specific-heat signatures without criticality (Schwab 2014). (3) Static fits cannot separate critical from supercritical (Simoes 2026). (4) Pairwise models of about 200 positively correlated units can be spuriously bimodal, and Boltzmann learning non-ergodic (Rostami 2017). (5) The H-set is non-unique (Cavus 2026); report stability over bootstrap H-sets. (6) Tokens are autocorrelated, so bootstrap over prompts, not tokens. (7) Results depend on the threshold. (8) A 1.7B model may have few consistently correct TriviaQA items, which can make gate G-H fail."
 },
 {
  "name": "O2 H-DYN: attractor vs length along the model's own rollouts (lens-free chain)",
  "what_it_decides": "Whether entry into the H-state is self-reinforcing beyond what position explains, i.e. the user's 'H-neuron attractor not bound to length statistics'. Whether H-occupancy grows with rollout length beyond position-matched controls; this is the lens-free counterpart of the chain-lens finding that confabulation grows with chain length. Proposed verdicts: 'H-attractor beyond length' (only together with O3) / 'length artefact' / 'repetition attractor, not H' / 'not decidable'. Shows: conditional dynamics of the H-population during generation. Cannot show: causation (O3) or what a chain lens would read.",
  "measurement": "Rollouts: those of O1. After the answer, the self-continuation runs to 384 tokens and serves as the 'chain'. Per generated token: m_H(t) = mean of s_jt over the H-set, and b_t = 1 if the pattern is in the H-basin of O1 (fallback: m_H above its reference 95th percentile). (A) Occupancy curve on log-spaced position bins; slope of logit occupancy vs log position, compared with the 50 control populations. (B) History dependence at fixed position: HD = logit P(b_{t+1}=1 | b_t=1) - logit P(b_{t+1}=1 | b_t=0), estimated in 8 position bins and pooled. Exit-hazard model: logit P(exit at t | dwell d) = spline(position, 8 knots) + gamma*log(1+d) + prompt effect; gamma < 0 means stickiness. (C) Chain-step link: does H-occupancy at continuation step k predict a factual error or fabricated elaboration in step k+1 (TriviaQA alias match, fabricated-entity items), with k as a covariate? (D) Optional and descriptive: where another part runs a chain lens on a model with an MTP head, correlate its confabulation at chain length L with the self-rollout H-occupancy at L for the same seeds. Intervals: 90 % prompt bootstrap, 2,000 replicates.",
  "null_and_controls": "(1) 50 random-neuron populations read on the same tokens; HD, gamma and the occupancy slope are ranked within this distribution. (2) Run-length shuffle surrogate: dwell runs permuted within each sequence, keeping occupancy and position profile. (3) 4-gram repetition rate as a covariate; degenerate-repetition segments reported separately. (4) Correct vs hallucinated first answers matched on length. (5) Round-2 generic style direction (block-14 projection) as a covariate where its vector is available (Drive or one B-Q0 round). (6) T 0.7 vs 1.0 as the noise axis.",
  "data_needed": "The same 1.5e6 tokens as O1, with per-token basin labels for the H-population and the 50 control populations.",
  "cost_estimate": "No extra generation (shared with O1). Hazard models and surrogates: about 5-10 min GPU/CPU.",
  "lens_free": "Uses only activation statistics of MLP units and text-derived labels. The 'chain' is the model's actual continuation, not a transported state decoded by an MTP head.",
  "inverse_free": "Forward regressions (logistic hazard and transition models) and counts; no reconstruction step.",
  "pitfalls": "(1) Every population is autocorrelated along text, so HD > 0 is expected; only the comparison with controls is informative. (2) If hallucination itself becomes more frequent later in a text, marginal occupancy rises without any attractor; the verdict must rest on the conditional statistics (B). (3) Degenerate repetition is itself an attractor and can mimic H-stickiness. (4) Sequences that stop early (EOS) need survival handling and censoring. (5) Basin labels inherit O1's binarisation and its non-unique H-set."
 },
 {
  "name": "O3 H-HYST: push-release hysteresis of the H-state (causal attractor test; H-analogue of round-2 G and K)",
  "what_it_decides": "Whether a transient push into the H-state persists after release longer than damage-matched pushes of random neurons, i.e. a self-sustaining H-state. Whether the persistence is carried by the model's own text (it survives a cache-free re-read) or only by the KV cache of pushed tokens. This gives G/K-type carry on the H-axis itself; round 2 measured carry only on a generic style direction. Proposed verdicts: 'self-sustaining H-state' / 'dissipative like controls' / 'carried by cache, not by state' / 'not testable' (manipulation check failed). Shows: causal persistence of an induced H-state. Cannot show: that natural hallucinations arise this way.",
  "measurement": "200 prompts (100 closed-book questions, 100 false-premise questions). Push A: scale all H-neurons by alpha during a window of w = 16 tokens from the first answer token (the Gao et al. intervention), then alpha = 1. Push B, which avoids reading the pushed units: during the same window, add at the block holding the most H-neurons the residual direction separating hallucinated from correct answers (difference of means on O1 fit data), norm-capped as in round 2. Minimal grid: alpha in {0, 2.5}; extended grid: alpha in {0, 1.5, 2, 3} and w in {4, 16, 64}. Readouts, all on unscaled activity after release, over 192 tokens: excess H-occupancy over a sham (alpha = 1) at the same positions; relaxation time tau_H from an exponential fit; K_H = excess in the next answer after a 'Continue.' turn (round-2 R6 protocol) divided by the in-window excess; hallucination or over-compliance rate after release. Cache-free re-read: the generated text is re-read by the unmodified model; excess present live but absent in the re-read is attributed to the KV cache. Manipulation-check gate: in-window excess (push B) or behavioural shift (push A) must exceed 3 SD of the sham's prompt-to-prompt variation; otherwise 'not testable'. Rule: 'self-sustaining' if tau_H or K_H lies above the 95th percentile of the damage-matched control distribution and the excess survives the re-read. Intervals: 90 % prompt bootstrap, 2,000 replicates.",
  "null_and_controls": "(1) Damage-matched random-neuron scaling: 3 sets with the same N and layer histogram; alpha set by bisection so that damage matches the H push within +-10 %. Damage is teacher-forced KL on 64 reference texts in nats, the round-2 convention. Each set is read on its own units (self-persistence) and cross-read on the H-units. (2) Norm-matched random-direction push as the control for push B. (3) Push into a different O1 basin of similar occupancy, for specificity. (4) Sham alpha = 1. (5) Live run vs cache-free re-read.",
  "data_needed": "Minimal grid: 200 prompts x 2 alpha x 6 conditions = 2,400 continuations of 208 tokens, plus 2,400 second turns of 128 tokens; re-reads of about 1.5e6 tokens. MDE: set from a 50-prompt pilot (prompt-level SD of tau_H and K_H), as the design rules require; round 2 gives no noise level for the H-axis.",
  "cost_estimate": "Minimal grid: generation about 75 calls x 6 s = 8 min; second turns about 6 min; re-reads at 5,600 tok/s about 5 min; damage calibration about 2 min. Total about 20 A100-min. Extended grid about 50 min. fp32, TF32 off.",
  "lens_free": "Pushes and readouts act on MLP activations and residual states; behaviour is judged from text. No vocabulary projection.",
  "inverse_free": "Forward interventions and forward re-reads only. Push B's direction is a difference of means (a forward statistic), not an inverse.",
  "pitfalls": "(1) Reading the same units that were pushed is circular inside the window, so only post-release activity counts, and push B avoids it. (2) Large alpha pushes activations out of distribution; Gao et al. report non-monotonic effects. (3) Damage acts like a temperature change, hence damage matching. (4) KV-cache contamination is the main rival explanation for persistence, hence the re-read. (5) As in round 2, persistence may run along the generic style direction rather than H, so that direction is a covariate."
 },
 {
  "name": "O4 H-J: H-neuron gain as the coupling constant of a Qwen agent population (loop -> agent -> Ising -> collaboration -> entity)",
  "what_it_decides": "Whether H-neuron gain acts as social coupling J (conformity) in a population of Qwen3-1.7B agents, rather than as noise or damage. Whether longer memory ('context held above natural length') makes two-faction states ('two entities') more persistent, lowers the committed-minority tipping fraction f_c below 0.5, and lowers mean payoff while factions coexist ('losing game'); this is the measurable core of the selfish-mining/catfish analogy. Proposed verdicts: 'H gain acts as coupling' / 'H gain acts as noise or damage' / 'memory length favours two factions' / 'no effect of memory length' / 'not testable' (format gate). Shows: a causal path from an identified neuron population to collective dynamics. Cannot show: that the same mechanism works inside one model's context.",
  "measurement": "Naming game after Ashery et al. 2025: 10 names, payoffs +100/-50, random pairs, memory of the last M interactions with M in {5, 20, 60} (M = 60 gives a prompt of about 2k tokens). Stage 1, single-agent response curves (the 'Ising fit' of an agent): memories are composed so that the fraction f of partner choices equal to name X lies in {0, 0.2, ..., 1}, with 10 names x 20 orderings. Fit logit P(choose X) = beta_J (f - 0.5) + h_X, which gives coupling beta_J and fields h_X. Conditions: H-scaling alpha in {0, 0.5, 1, 1.5, 2}; damage-matched random-neuron scaling; sampling T in {0.5, 0.7, 1.0} as the noise axis. Stage 2, populations (verification): N = 24 agents, 1,000 interactions per run, alpha in {0.5, 1, 2} x M in {5, 20, 60} x 3 runs; committed minorities at f in {0.1, 0.2, 0.3, 0.4} for alpha = 1 and M in {5, 60}. Readouts: consensus time; f_c = smallest minority fraction that flips an established convention in >= 2 of 3 runs; fraction of runs with two persistent factions (each >= 25 % over the last 200 interactions); mean payoff in two-faction vs consensus periods. Generate-verify: a forward agent simulation with the stage-1 response function predicts stage 2, and the prediction error is reported as the verifier's error. Gate: >= 95 % valid single-name answers at alpha = 1 in a 200-prompt pre-test; an over-compliance sanity check of the H-set in Qwen (sycophancy items at alpha 0 vs 2).",
  "null_and_controls": "(1) Damage-matched random-neuron scaling (teacher-forced KL on reference texts). (2) Temperature sweep, to separate coupling from noise: De Nobili 2026 and Okawa 2026 show that bias and noise can mimic transitions. (3) Fields h_X estimated separately, because bias can dominate coupling. (4) Shuffled-memory control: memory contents randomised, where beta_J must be about 0. (5) M varied at fixed prompt length by padding with neutral text, to separate memory content from sheer context length.",
  "data_needed": "Stage 1: 6 fractions x 10 names x 20 orderings x (5 alpha + 4 damage-matched + 3 T) = about 14,400 prompts of 300-2,000 tokens, <= 8 new tokens each. Stage 2: about 27 population runs x 1,000 interactions x 2 answers plus 24 minority runs, about 1e5 short generations.",
  "cost_estimate": "Prefill dominates. Stage 1: about 1.2e7 prompt tokens at 5,000 tok/s = about 40 min, or about 15 min with shared-prefix KV caching of the fixed rules. Stage 2: populations run in parallel (batch 32-48 per step), 1,000 steps x 1-2 s = about 20-35 min. Total about 45-60 A100-min, fp32, TF32 off.",
  "lens_free": "Behaviour (chosen names) plus activation scaling; no readout of internal states into words.",
  "inverse_free": "Forward logistic fits of response curves and a forward agent simulation checked against real runs (generate-verify); nothing is inverted.",
  "pitfalls": "(1) A 1.7B model may not play the game reliably (format gate), and H-scaling may break formatting. (2) Name or letter biases can dominate coupling. (3) N = 24 gives finite-size crossovers, so no phase-transition claims. (4) The H-to-over-compliance link is shown for other model families and must be re-checked in Qwen3-1.7B. (5) Agents are separate instances, so conclusions do not transfer to neurons inside one forward pass. (6) Longer memory also lengthens prompts, hence the padding control."
 },
 {
  "name": "O5 DUAL-M2: two-community order parameter under hyper-long context (two entities inside one model)",
  "what_it_decides": "Whether text held far above natural length makes the internal population of one model split into two anti-aligned communities with long dwell times ('a force that splits one entity into two'), beyond length-matched shuffled controls. Whether the H-population takes part. Whether two-voice self-generation stays coherent longer than one-voice self-generation (the catfish effect). Proposed verdicts: 'two-entity split under hyper-long context' / 'no split beyond controls' / 'split only with two-voice input (readout works, hypothesis not supported)' / 'readout insensitive' (positive control failed) / 'model-only bimodality (artefact)'. Shows: a structural split in activation space. Cannot show: what the entities are, except through the generate-verify step.",
  "measurement": "Texts, 60 per cell, at L in {1k, 4k, 8k, 16k} tokens. S1: single-voice natural long texts (single-narrator prose, single-persona dialogue). S2: two-voice texts alternating two personas or stances (positive control). S3: 'self-catfished' text, the model's own continuation of an S1 seed extended to 16k with a 2k sliding window at T 0.7; it is generated with the window and measured in one teacher-forced pass over the full text. S4: S1 and S3 with sentence order shuffled (same tokens and length, no thread). Populations: H-units; the 2,000 most variable MLP units at blocks 6, 14 and 18 (selected on a fit set); block-14 residual projected on its top-2 PCs (fit set). Communities A and B on the fit half: sign of the leading eigenvector of J (O1 pipeline) or of the correlation matrix, by power iteration. Order parameter Delta(t) = m_A(t) - m_B(t), with m = mean centred activity per community; positions >= 64 only, sink tokens excluded. Readouts: Hartigan dip test and bimodality coefficient of P(Delta); median dwell time in each sign of Delta; split index = distance between modes / within-mode SD; trend over L; overlap of A and B with the H-units. Rule: 'split' if dwell time and split index in S1 or S3 exceed the 95th percentile of S4 at the same L, rise from 1k to 16k, and S2 passes the readout check. Generate-verify for entity content: clamp community A to its mode values in neutral prompts and generate. Re-read with the unmodified model: does Delta return to that mode above a random-clamp null? Judge persona or stance against S2 labels; 3 attempts, then abstain. Catfish readout: tokens until degenerate repetition (4-gram rate > 0.3 in a 256-token window) in self-generation from two-voice vs one-voice seeds.",
  "null_and_controls": "(1) S4 shuffled texts at matched length (main null). (2) S2 two-voice texts as positive control. (3) Community labels permuted (random partitions). (4) Rostami artefact check: bimodality must be visible in the empirical Delta, not only in a fitted model. (5) Position-binned centring against slow drifts from RoPE and attention dilution. (6) Sink and massive-activation tokens excluded.",
  "data_needed": "4 conditions x 4 lengths x 60 texts (S4 covers two conditions). About 5e6 tokens read teacher-forced. 60 S3 texts generated to 16k tokens. Activations of H-units plus 6,000 MLP units and 2 PCs per token, stored in fp16.",
  "cost_estimate": "Reading: about 5e6 tokens at a blended 3,000-5,600 tok/s; my estimate is that attention adds up to 1.7x FLOPs at 16k context. That gives about 20-28 min. S3 generation: 2 batches x 16k steps x 25 ms = about 14 min. Analysis: about 5 min. Total about 45 A100-min, fp32, TF32 off. 16k-token reads need the memory-efficient SDPA kernel: eager attention matrices do not fit in 40 GB, and the KV cache for the 2k window x 32 sequences is about 15 GB.",
  "lens_free": "Communities and order parameters live in activation space only. Entity content comes from the model's own generation inside a verified round trip, not from a lens.",
  "inverse_free": "Partitions come from power iteration (forward products). Clamping and re-reading are forward. No inversion.",
  "pitfalls": "(1) Long contexts drift (RoPE, attention dilution); slow modes can look like entities, hence matched shuffled controls. (2) Two-voice input trivially gives two clusters; the hypothesis concerns S1 and S3. (3) Self-generated long text degenerates into repetition. (4) Partitions fitted on long-context data overfit, so they are evaluated on the held-out half. (5) 'Single-voice' natural texts often contain quotations and dialogue."
 },
 {
  "name": "O6 DEPTH-CLU: metastable token clusters across depth (Geshkovski-type) as entities",
  "what_it_decides": "Whether the token states of one context organise across depth into a few clusters that persist over many consecutive blocks (metastable multi-cluster states). Whether long single-voice contexts produce a two-cluster state that is absent in shuffled controls. Descriptive; proposed labels: 'metastable two-cluster state' / 'single cluster' / 'no persistent clustering'. Shows: the geometry of token states across depth. Cannot show: dynamics in the theorems' sense, because Qwen's blocks are not one time-homogeneous map.",
  "measurement": "O5 texts at L in {1k, 8k}. Per text, 2,048 tokens sampled at positions >= 64. Tokens whose residual norm exceeds 20x the median at any block are excluded (massive activations, sinks). State x_ib = input of block b after its RMSNorm (the vectors attention uses), centred per block. Per block: cosine Gram matrix; k_b = number of eigenvalues above the 99th percentile of the top eigenvalue under a column-permutation null; spectral clustering into k_b clusters. Persistence = longest run of blocks with k_b >= 2 and adjusted Rand index >= 0.8 between adjacent blocks. Also reported: overlap of clusters with O5 communities and with H-basin tokens. Optional pure-architecture baseline: Qwen3-1.7B with randomly initialised weights (no training).",
  "null_and_controls": "(1) S4 shuffled texts at matched length. (2) Per-block dimension-permutation null for k_b. (3) Random-weight baseline. (4) Exclusion of sinks and massive activations.",
  "data_needed": "Hooks during the O5 reads: 2,048 tokens x 29 states x 2,048 dimensions per text, processed on the fly (about 0.25 GB per text in fp16); 480 texts.",
  "cost_estimate": "About 10 A100-min on top of O5 (2k x 2k Gram matrices x 29 blocks x 480 texts). Random-weight baseline about +5 min.",
  "lens_free": "Geometry of hidden states only.",
  "inverse_free": "Eigen-decompositions and clustering are forward analyses, not reconstructions.",
  "pitfalls": "(1) The theorems assume weights constant across depth and no MLPs. (2) Skip connections and MLPs prevent collapse (Dong et al. 2021). (3) Anisotropy (a common direction) inflates cosine similarity, hence the centring. (4) Sinks and massive activations form trivial outlier clusters. (5) Cluster counts depend on the null threshold. (6) Clusters may reflect syntax or position rather than 'entities'; overlap with O5 communities is the check."
 },
 {
  "name": "O7 ARROW: irreversibility of text and state trajectories (non-equilibrium link; the user's forward/backward 'vector quantity')",
  "what_it_decides": "(a) Whether hallucinated answers and chains carry a different arrow-of-time signature than correct ones at matched length. (b) Whether entry into the H-state changes the irreversibility of state trajectories. (c) Whether two-entity regimes show a net cyclic flux between macrostates that rises with context length (catfish = sustained driving). (d) The user's 'front predicts following vs preceding fronts', split into drift (non-stationarity), path irreversibility and computational asymmetry. (e) The user's 'hierarchy of channels' as the asymmetric part of kinetic-Ising couplings between communities. Proposed verdicts: 'irreversibility marks hallucination' / 'no difference'; 'sustained cyclic flux' / 'detailed balance not rejected'. Shows: time asymmetry of statistics. Cannot show: thermodynamic heat or a physical temperature.",
  "measurement": "(a) Sagawa's estimator on O2 rollouts: sigma_block = ln P(y) - ln P(y with sentence order reversed), prompt prefix fixed, normalised per token. Compare hallucinated vs correct in length strata and across the number of sentences k = 2..6; also S2 vs S1 texts from O5. (b) Coarse-grain block-14 states (or H-basin labels) by hierarchical k-means (cosine; k = 2, 4, 8) fitted on a fit half. S_dot = sum P_ij ln(P_ij/P_ji) per window, using the largest k at which every transition is observed (as in Lynn et al.). Compare 64-token windows after H-entry with position-matched windows without entry. (c) Net flux J_AB = P(A->B) - P(B->A) and cycle affinity for the O5 macrostates, as a function of L. (d) Forward and backward predictors of equal capacity (ridge/logistic) for the H-population state at t+k and at t-k from the state at t, fitted on separate halves; asymmetry A_k = (loss_bwd - loss_fwd)/(loss_bwd + loss_fwd). Under stationarity A_k is 0 in information terms, so non-zero values are decomposed with position-binned refits. (e) Kinetic Ising fit P(s_{i,t+1} | s_t) by logistic regression; asymmetry ||J - J^T|| / ||J + J^T||; community-level drive matrix. Intervals: 90 % prompt bootstrap, 2,000 replicates.",
  "null_and_controls": "(1) Within-sequence temporal shuffles give the noise floor of S_dot and A_k. (2) Time-reversed sequences: A_k must flip sign. (3) Length strata and position-matched windows. (4) Texts not sampled from Qwen are excluded from (a), because sigma need not be >= 0 for them (Sagawa).",
  "data_needed": "Reuses the O2 rollouts (1.5e6 tokens; continuations need >= 4 sentences for (a)) and the O5 texts; one extra teacher-forced pass over the sentence-reversed texts.",
  "cost_estimate": "(a) About 1.5e6 tokens at 5,000 tok/s = about 5 min. (b)-(e) on stored states: about 10 min GPU/CPU. Total about 15 A100-min, fp32, TF32 off.",
  "lens_free": "(a) uses the model's own output probabilities of whole texts; no intermediate state is read out. (b)-(e) use activation states only.",
  "inverse_free": "Likelihood evaluations, counts and forward regressions. Predicting earlier states from a later state is a fitted forward map, which the design rules allow; it is stated as such.",
  "pitfalls": "(1) sigma mixes compression loss and model mismatch, because the backward process reuses forward kernels; it is information-theoretic, not heat. (2) Token-level sigma is dominated by syntax, so sentence blocks are used. (3) Coarse-grained S_dot is only a lower bound, and finite-data bias grows with k. (4) Generation is non-stationary, so stationary-process estimators need windows and position control. (5) Halperin's 'semantic entropy production' is an analogy and does not validate (a)."
 },
 {
  "name": "O8 EMERGE: does a community act as an entity? (Rosas/Riedl criteria inside the model)",
  "what_it_decides": "Whether a macro variable of the population (H-community magnetisation, or sign of Delta from O5) predicts its own future beyond what its parts predict (S_macro > 0), i.e. entity formation in the information-theoretic sense. Whether this appears in H-basin windows (H-CHAIN) or grows with context length (H-DUAL). Proposed verdicts: 'macro self-prediction detected' / 'not detected (redundancy-dominated)' / 'not decidable'. Shows: a sufficient criterion for macro-level dynamics. Cannot show: absence of emergence, because S_macro can be negative even when synergy is present.",
  "measurement": "Parts X_k = magnetisations of K = 8 communities (spectral partition of J on the fit half), each binarised at its median. Macro V_t = population magnetisation binarised at its median, or sign(Delta). S_macro(l) = I(V_t; V_{t+l}) - sum_k I(X_k,t; V_{t+l}) for l in {1, 4, 16} tokens. Plug-in estimators with Jeffreys pseudo-counts; Miller-Madow sensitivity. Pairwise synergy between the two O5 communities (Williams-Beer PID, I_min redundancy) as 'emergence capacity'. Per-sequence values; 90 % prompt bootstrap with 2,000 replicates.",
  "null_and_controls": "(1) Column-wise block shuffles (block length l, B = 200). (2) Coordination-free surrogate: each community an independent two-state Markov chain with its own transition rates. (3) Position-trend residualisation. (4) The 50 random populations of O1.",
  "data_needed": "Binarised O2 and O5 states (>= 1e6 tokens). K = 8 parts keeps the plug-in estimators well sampled.",
  "cost_estimate": "CPU only, about 15 min.",
  "lens_free": "Information measures on activation states.",
  "inverse_free": "Counts and plug-in estimators; no reconstruction.",
  "pitfalls": "(1) S_macro is sufficient, not necessary; redundancy among parts drives it negative even with synergy (Riedl). (2) Binning sensitivity and small-sample bias. (3) The time step is a token, so the sampled token's input is mixed with internal dynamics. (4) Communities defined on the same data inflate apparent structure, hence fit and test halves."
 },
 {
  "name": "O9 HOLO: holographic (associative-memory) signatures of the H-state",
  "what_it_decides": "Whether the H-state is stored 'holographically' or locally in the ~0.1 % H-units, as the sparse-probe reading implies. 'Holographically' here means: distributed, recoverable from many small random subsets of units, completed from a partial cue, and degrading gracefully. This bridges the user's holographic question to the energy picture: in Hopfield/HRR terms, holographic storage = distributed traces + clean-up into attractors. Proposed verdicts: 'distributed (holographic-like) H-code' / 'localised H-code' / 'redundancy from a shared latent variable, not storage'. Shows: the associative-memory sense of 'holographic'. Cannot show: anything about AdS/CFT-type holography.",
  "measurement": "(1) Recoverability (forward probe): predict the hallucination label and the H-basin label from CETT features of k random non-H units (k in {10, 30, 100, 300, 1,000}; 20 draws each) vs k-sized subsets of the H-set. L2 logistic regression, AUROC on held-out prompts; also by layer band (early vs late units), as a weak 'boundary-bulk' check across depth. Proposed rule: 'distributed' if the median random subset at k = 1,000 reaches >= 90 % of the H-set's AUROC above chance; 'localised' if < 50 %. (2) Pattern completion: during generation on 200 fabricated-entity prompts, clamp a 'boundary' fraction (25 %, 50 %) of the H-units to their H-basin values. Readouts: completion index of the unclamped H-units (fraction of the Hamming distance to the basin that is closed) and the hallucination rate; compared with damage-matched random-unit clamps. (3) Graceful degradation: mean-ablate random 10/25/50 % of the H-units; slope of hallucination rate and basin occupancy against the ablated fraction (holographic: smooth; localised: threshold). Known-answer calibration: a planted Hopfield toy network with distributed vs localised storage, passed through the same readouts (verifier error rates).",
  "null_and_controls": "(1) Label-shuffled probes. (2) Matched random subsets. (3) Damage-matched clamps and ablations (teacher-forced KL in nats). (4) The planted toy, justified because Qwen offers no ground truth about storage format.",
  "data_needed": "CETT of about 1e4 non-H units on 2e5 answer tokens from O1 (fp16, about 4 GB). 200 prompts x (2 clamp + 3 ablation fractions) x 2 conditions = 2,000 continuations of 192 tokens.",
  "cost_estimate": "Re-read of 2e5 tokens about 1 min; probe fits about 5 min; interventions about 63 calls x 5.5 s = 6 min plus damage calibration about 3 min; toy about 3 min. Total about 20 A100-min, fp32, TF32 off.",
  "lens_free": "Probes map activations to behavioural labels, not to vocabulary; clamps and ablations are interventions.",
  "inverse_free": "Probes are fitted forward maps (allowed by the design rules, stated as such); completion is measured by forward runs. Holographic decoding by correlation plus clean-up (Plate) is itself a generate-then-clean scheme, not an exact inversion.",
  "pitfalls": "(1) Redundancy can come from a shared latent variable (e.g. uncertainty) tracked by many units rather than from distributed storage of a pattern. (2) Clamped values out of context may be out of distribution. (3) The proposed 90 %/50 % thresholds are proposals that need pre-registration. (4) 'Holographic' here is the associative-memory sense only."
 }
]
```

## where_metaphor_breaks
- EBM reasoner vs Qwen: Qwen3-1.7B minimises no energy at inference. EBTs, IRED and EBRs run gradient or Langevin descent on a learned energy; Qwen computes one fixed feed-forward pass per token. -log p(text) is a valid energy (Blondel et al.'s ARM-EBM bijection), but sampling draws from it rather than descending it. Only outer loops iterate: token steps, the round-2 text loop and agent loops.
- Loop transformer vs Qwen depth: Qwen's 28 blocks have distinct weights, so depth is not a time-homogeneous dynamical system. Fixed points, orbits and sliders (Geiping), energy stationarity (EBR) and clustering theorems (Geshkovski) assume one repeated map. Attractor claims must be made on generation steps or outer loops.
- Attention = Hopfield update: this holds for one update with Ramsauer's energy. With multi-head projections, causal masks, RoPE, RMSNorm, MLPs and residual sums, no known energy decreases monotonically through a standard LM (the Energy Transformer has to be built for that). Metastable head patterns are not residual-stream attractors.
- Ising-modellable: a pairwise maximum-entropy fit is a statistical description. Couplings are effective: they absorb shared inputs (position, topic, prompt, uncertainty) and are not wiring. Minima of the fitted energy are not dynamical attractors of the network, which does not run Glauber dynamics on it. Positive correlations can make pairwise models spuriously bimodal (Rostami 2017), and latent variables produce Zipf and criticality signatures without tuning (Schwab 2014).
- Temperature: the sampling temperature and the T of a fitted max-ent model are parameters, not physical temperatures (Tkacik et al. 2015 say so). Static fits cannot distinguish critical from supercritical (Simoes 2026).
- Non-equilibrium: there is no heat bath and no energy flow. The entropy production of generated text (Sagawa) is information-theoretic: it mixes compression loss and model mismatch because the backward process reuses forward kernels, and at token level it is dominated by syntax. Coarse-grained S_dot is only a lower bound. Generation grows a sequence and is non-stationary, so 'equilibrium' exists only for coarse-grained, windowed statistics.
- Forward vs backward front predictability (the user's vector quantity): for a stationary process, a front predicts its successor and its predecessor equally well in information terms, since H(X_{t+k}|X_t) = H(X_{t-k}|X_t). A non-zero ratio needs non-stationarity, path irreversibility (P(i->j) != P(j->i)), or a computational asymmetry between predictors of equal capacity (the Arrows-of-Time kind). These are three different things and must be measured separately.
- Agent -> spin: in LLM-population studies the spins are separate model instances with memory and payoffs, interacting over rounds. Neurons, heads or positions inside one forward pass have no goals, memory or payoffs. The agent-spin mapping holds for populations of instances (O4) and is only an analogy for units inside one model.
- Collaboration -> entity-forming: interacting LLM populations break symmetry into one convention (Ashery 2025). Two persistent groups need weak coupling, heterogeneity, committed minorities or antagonistic inter-group coupling; a ferromagnetic population does not split by itself. Observed 'transitions' in LLM collectives are often field-driven crossovers where bias dominates coupling (De Nobili 2026), or finite-size crossovers (Okawa 2026).
- Catfish / selfish mining: the 1/3 selfish-mining threshold (at gamma 0) comes from rational payoff maximisation. LLM populations have no universal tipping threshold: committed minorities of 2-67 % flip conventions depending on the model. 'The optimum lies below 50 %' has no counterpart in a single model's activations. The nearest measurable analogues are a tipping fraction f_c < 0.5 and a lower mean payoff while factions coexist.
- Hyper-long context = catfishing: very long contexts change the state through position encoding and attention dilution. These slow drifts can mimic 'new entities', so a split counts as a force arising from context only if it beats content-shuffled controls of the same length.
- H-neurons as an attractor: H-neurons are defined by a sparse linear probe whose selection is non-unique within correlated clusters (Cavus 2026). A population model over a probe-defined set inherits the probe's biases, and 'attractor' needs the dynamic and interventional signatures (O2, O3), not a static energy landscape.
- Holography: a transformer has no bulk-boundary duality and no emergent extra dimension. Hashimoto et al. read depth as the radial direction only for a network trained to fit boundary data. The operational content available lens-free is the associative-memory sense: distributed storage, completion from partial cues, graceful degradation. Even Plate's holographic decoding uses correlation as an approximate inverse plus a clean-up memory, i.e. a generate-then-clean scheme, not an exact inversion.
- Lens-free does not remove confabulation from interpretation: lens-free measures give structure and dynamics, not content. Naming what a basin or community 'is' needs the model's own generation, which can itself confabulate, so the round-trip verifier must be calibrated on known answers (planted basins, the round-1 SEA push, two-voice texts).
- Contrastive decoding as an energy difference: the chain lens scores log p(state) - log p(prior), an energy gap. My untested inference: its growth of confabulation with chain length may come from that gap shrinking relative to the MTP head's language prior, a property of the readout. If so, lens length effects and H-neuron effects stay confounded until a lens-free measure separates them.

## refs
```json
[
 {
  "title": "Energy-Based Transformers are Scalable Learners and Thinkers",
  "authors": "Alexi Gladstone, Ganesh Nanduru, Md Mofijul Islam, Peixuan Han, Hyeonjeong Ha, Aman Chadha, Yilun Du, Heng Ji, Jundong Li, Tariq Iqbal",
  "year": "2025",
  "url": "https://arxiv.org/abs/2507.02092",
  "finding": "EBTs assign an energy to input/candidate pairs and predict by gradient descent on the energy until convergence; abstract claims up to 35 % higher scaling rate than Transformer++ and 29 % more improvement from System-2 thinking on language tasks.",
  "relevance": "Defines the energy-based reasoner idea; contrast case for Qwen, which does not descend an energy at inference.",
  "how_checked": "Opened arXiv abs page via WebFetch; read abstract and metadata."
 },
 {
  "title": "Learning Iterative Reasoning through Energy Minimization",
  "authors": "Yilun Du, Shuang Li, Joshua B. Tenenbaum, Igor Mordatch",
  "year": "2022",
  "url": "https://arxiv.org/abs/2206.15448",
  "finding": "Each reasoning step is an energy-minimisation step over a learned landscape on outputs; harder problems get more optimisation (adaptive computation); ICML 2022.",
  "relevance": "Earliest EBM-reasoner formulation; adaptive compute as iterated descent.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Learning Iterative Reasoning through Energy Diffusion (IRED)",
  "authors": "Yilun Du, Jiayuan Mao, Joshua B. Tenenbaum",
  "year": "2024",
  "url": "https://arxiv.org/abs/2406.11179",
  "finding": "Learns a sequence of annealed energy landscapes; adapts the number of optimisation steps to difficulty; generalises to harder Sudoku, larger matrices and graphs; ICML 2024.",
  "relevance": "EBM reasoner with annealing (temperature-like schedule).",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Energy-Based Reasoners: Your Recurrent Reasoner Should Be an Energy-Based Model",
  "authors": "William Xingxu Chen, Alexi Gladstone, Zeyu Michael Li, Heng Ji, Yilun Du",
  "year": "2026",
  "url": "https://arxiv.org/abs/2611.08977",
  "finding": "Looped update written as gradient descent on a learned energy over latent states, z_{t+1} = z_t - alpha grad E(z_t; x), run as noisy projected Langevin dynamics; stops when the projected gradient norm < tau (0.2 on Sudoku); K parallel trajectories in training shape several modes; notes HRM can converge to correct and incorrect fixed points; 1.8M/7.1M parameters; Sudoku-Extreme 99.4 % single sample, 99.9 % with 5; 345x less inference compute.",
  "relevance": "Most likely the EBM reasoner the user means; links loop, energy, fixed points and wrong attractors explicitly.",
  "how_checked": "Opened arXiv abs page and the HTML full text (method, related work, results) via WebFetch."
 },
 {
  "title": "Energy-guided Recursive Model",
  "authors": "Yifei Zhao, Ying Tang",
  "year": "2026",
  "url": "https://arxiv.org/abs/2607.10128",
  "finding": "Hopfield-type memories of valid local and global structures assign energies to candidate trajectories; energies guide candidate selection and recurrence depth and support parallel tempering; deeper recurrence not necessarily better; RedPajama-V2 perplexity -1.74 %; Sudoku 98.97 %, PPBench 88.04 %, Maze 99.30 %.",
  "relevance": "Joins EBM reasoning with associative (holographic-sense) memory scoring.",
  "how_checked": "Opened arXiv abs page; read abstract only."
 },
 {
  "title": "Think Consistently, Reason Efficiently: Energy-Based Calibration for Implicit Chain-of-Thought (EBM-CoT)",
  "authors": "Zhikang Chen, Sen Cui, Deheng Ye, Yu Zhang, Yatao Bian, Tingting Zhu",
  "year": "2025",
  "url": "https://arxiv.org/abs/2511.07124",
  "finding": "An EBM refines latent thought representations toward lower-energy, more consistent regions without modifying the base language model.",
  "relevance": "EBM reasoning layered on a frozen LLM; would need a trained EBM, so not usable under the no-training rule.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "EBM vs. LLMs: Our Kona EBM a 96% vs. 2% Sudoku Benchmark (Logical Intelligence blog)",
  "authors": "Logical Intelligence (company)",
  "year": "2026",
  "url": "https://logicalintelligence.com/blog/energy-based-model-sudoku-demo",
  "finding": "Kona 1.0 described as a commercial energy-based reasoning model scoring complete candidate solutions with a scalar energy and revising via gradients in a continuous latent space; vendor-reported 96.2 % Sudoku at 313 ms vs about 2 % for frontier LLMs; no paper or architecture details; page dates conflict.",
  "relevance": "Possible referent of 'EBM reasoner'; claims unverifiable.",
  "how_checked": "Opened the vendor page; numbers are self-reported and unverified."
 },
 {
  "title": "Autoregressive Language Models are Secretly Energy-Based Models: Insights into the Lookahead Capabilities of Next-Token Prediction",
  "authors": "Mathieu Blondel, Michael E. Sander, Germain Vivier-Ardisson, Tianlin Liu, Vincent Roulet",
  "year": "2025",
  "url": "https://arxiv.org/abs/2512.15605",
  "finding": "Explicit bijection between autoregressive models and EBMs in function space, a special case of the soft Bellman equation; equivalence of supervised learning; distillation bounds; insight into lookahead.",
  "relevance": "Licenses -log p(text) as an energy for Qwen without implying energy descent.",
  "how_checked": "Opened arXiv abs page (v4, May 2026); ICML 2026 venue seen only in a search listing (unverified)."
 },
 {
  "title": "Contrastive Decoding: Open-ended Text Generation as Optimization",
  "authors": "Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, Mike Lewis",
  "year": "2023",
  "url": "https://arxiv.org/abs/2210.15097",
  "finding": "CD-score = log p_EXP - log p_AMA for tokens in V_head (p_EXP >= alpha max p_EXP, alpha = 0.1), otherwise -inf; ACL 2023.",
  "relevance": "CD is an energy difference; the chain lens decodes contrastively against a mean prior state.",
  "how_checked": "Opened abs page; read equations 1-3 on PDF pages 3-4."
 },
 {
  "title": "Hopfield Networks is All You Need",
  "authors": "Hubert Ramsauer, Bernhard Schaefl, Johannes Lehner, Philipp Seidl, Michael Widrich, et al. (16 authors, last Sepp Hochreiter)",
  "year": "2020",
  "url": "https://arxiv.org/abs/2008.02217",
  "finding": "Modern Hopfield update equals attention; fixed points: global average, metastable states averaging over subsets, single patterns; early layers global averaging, higher layers partial averaging via metastable states; exponential capacity, one-update retrieval.",
  "relevance": "Attention-as-energy link; metastable subset averages as candidate entities; associative memory as the operational meaning of holographic.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Energy Transformer",
  "authors": "Benjamin Hoover, Yuchen Liang, Bao Pham, Rameswar Panda, Hendrik Strobelt, Duen Horng Chau, Mohammed J. Zaki, Dmitry Krotov",
  "year": "2023",
  "url": "https://arxiv.org/abs/2302.07253",
  "finding": "A sequence of attention layers purposely designed to minimise an engineered energy (dense associative memory); NeurIPS 2023.",
  "relevance": "Shows monotone energy descent through layers must be built in; standard LMs such as Qwen lack it.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Mapping of attention mechanisms to a generalized Potts model",
  "authors": "Riccardo Rende, Federica Gerace, Alessandro Laio, Sebastian Goldt",
  "year": "2024",
  "url": "https://arxiv.org/abs/2304.07235",
  "finding": "One layer of factored self-attention learns the conditionals of a generalized Potts model; MLM training is exactly equivalent to solving the inverse Potts problem by pseudo-likelihood; Phys. Rev. Research 6, 023057.",
  "relevance": "Ising/Potts link for attention; justifies pseudo-likelihood (forward regressions) as the inverse-free fitting method in O1.",
  "how_checked": "Opened arXiv abs page; read abstract and journal reference."
 },
 {
  "title": "The emergence of clusters in self-attention dynamics",
  "authors": "Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, Philippe Rigollet",
  "year": "2023",
  "url": "https://arxiv.org/abs/2305.05465",
  "finding": "With time-constant weights, tokens cluster toward limiting objects; locations set by initial tokens; limit type depends on the value-matrix spectrum; in 1D attention converges to a low-rank Boolean matrix; leaders emerge; NeurIPS 2023.",
  "relevance": "Entities as token clusters; basis for O6.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "A mathematical perspective on Transformers",
  "authors": "Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, Philippe Rigollet",
  "year": "2023 (Bull. Amer. Math. Soc. 62, 2025)",
  "url": "https://arxiv.org/abs/2312.10794",
  "finding": "Transformers treated as interacting particle systems; clusters emerge in long time.",
  "relevance": "Framework for the particle/clustering view.",
  "how_checked": "Opened arXiv abs page; read abstract and journal reference."
 },
 {
  "title": "Dynamic metastability in the self-attention model",
  "authors": "Borjan Geshkovski, Hugo Koubbi, Yury Polyanskiy, Philippe Rigollet",
  "year": "2024",
  "url": "https://arxiv.org/abs/2410.06833",
  "finding": "Particles stay trapped near a several-cluster configuration for an exponentially long time before merging; staircase energy profile; saddle-to-saddle trajectories.",
  "relevance": "Metastable multi-cluster states as candidate 'two entities'.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Emergence of meta-stable clustering in mean-field transformer models",
  "authors": "Giuseppe Bruno, Federico Pasqualotto, Andrea Agazzi",
  "year": "2024",
  "url": "https://arxiv.org/abs/2410.23228",
  "finding": "For many tokens the model stays near a meta-stable manifold whose structure is selected by the inverse temperature (index maximising rescaled Gegenbauer polynomials).",
  "relevance": "The number of metastable clusters depends on a temperature-like parameter.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth",
  "authors": "Yihe Dong, Jean-Baptiste Cordonnier, Andreas Loukas",
  "year": "2021",
  "url": "https://arxiv.org/abs/2103.03404",
  "finding": "Pure self-attention converges doubly exponentially to rank 1; skip connections and MLPs stop the degeneration.",
  "relevance": "Pitfall for transferring clustering theory to Qwen (O6).",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Massive Activations in Large Language Models",
  "authors": "Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu",
  "year": "2024",
  "url": "https://arxiv.org/abs/2402.17762",
  "finding": "Very few activations are about 100,000x larger than others, roughly input-independent, act as bias terms and concentrate attention on their tokens; COLM 2024.",
  "relevance": "Pitfall: outlier clusters and sink tokens in O5/O6.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Looped Transformers as Programmable Computers",
  "authors": "Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D. Lee, Dimitris Papailiopoulos",
  "year": "2023",
  "url": "https://arxiv.org/abs/2301.13196",
  "finding": "A looped 13-layer transformer can emulate a small instruction-set computer (calculator, linear algebra, in-context learning with backpropagation).",
  "relevance": "The loop-transformer link of the user's chain.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach",
  "authors": "Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, Tom Goldstein",
  "year": "2025",
  "url": "https://arxiv.org/abs/2502.05171",
  "finding": "3.5B parameters, 800B tokens; latent states converge to fixed points, fall into orbits (harder questions) or drift as sliders; context-dependent and path-independent; per-token exit when KL between steps < 5e-4; limit points estimated at 128 iterations.",
  "relevance": "Attractor types in looped LMs; they do not transfer to Qwen's distinct blocks.",
  "how_checked": "Opened abs page and HTML sections 6-7 via WebFetch."
 },
 {
  "title": "Scaling Latent Reasoning via Looped Language Models (Ouro)",
  "authors": "Rui-Jie Zhu, Zixuan Wang, Kai Hua, et al. (33 authors)",
  "year": "2025",
  "url": "https://arxiv.org/abs/2510.25741",
  "finding": "LoopLMs at 1.4B and 2.6B trained on 7.7T tokens with entropy-regularised depth allocation match up to 12B models; gains from knowledge manipulation, not capacity.",
  "relevance": "Looped LMs exist at Qwen scale, but Qwen3-1.7B is not one.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Arrows of Time for Large Language Models",
  "authors": "Vassilis Papadopoulos, Jeremie Wenger, Clement Hongler",
  "year": "2024",
  "url": "https://arxiv.org/abs/2401.17505",
  "finding": "Consistent asymmetry between forward and backward next-token log-perplexity in large models, although information theory predicts none; explained by sparsity and computational complexity; ICML 2024.",
  "relevance": "Computational arrow of time; one of three readings of the user's forward/backward front predictability (O7d).",
  "how_checked": "Opened arXiv abs page; read abstract and venue."
 },
 {
  "title": "Stochastic Thermodynamics for Autoregressive Generative Models: A Non-Markovian Perspective",
  "authors": "Takahiro Sagawa",
  "year": "2026",
  "url": "https://arxiv.org/abs/2604.07867",
  "finding": "Entropy production S = D_KL(P_fwd || P_bwd), backward path using the same kernels on the reversed sequence; per-trajectory sigma from two likelihood passes; token-level sigma dominated by a syntactic artefact; sentence-level sigma larger for causally ordered texts (GPT-2; Qwen3-4B-Base significant for all five corpora); exact decomposition into non-negative per-step compression-loss and model-mismatch terms; PRX Intelligence 1, 023013.",
  "relevance": "Lens-free, inverse-free non-equilibrium measure for O7(a).",
  "how_checked": "Opened abs page and HTML (definitions, estimator, experiments); appendix C not read."
 },
 {
  "title": "Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons and Manage Hallucinations",
  "authors": "Igor Halperin",
  "year": "2025",
  "url": "https://arxiv.org/abs/2512.05156",
  "finding": "Proposes semantic faithfulness (KL between topic transition matrices) and a semantic entropy production; states that high faithfulness generally implies low entropy production; demonstrated on SEC 10-K summarisation.",
  "relevance": "Links an entropy-production-like quantity to hallucination, but as an analogy.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Estimating dissipation from single stationary trajectories",
  "authors": "Edgar Roldan, Juan M. R. Parrondo",
  "year": "2010",
  "url": "https://arxiv.org/abs/1004.2831",
  "finding": "Time-reversal asymmetry of a stationary time series carries information on entropy production and detects non-equilibrium without measurable flows; Phys. Rev. Lett. (DOI 10.1103/PhysRevLett.105.150607 listed).",
  "relevance": "Estimator family for state-trajectory irreversibility (O7b).",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Broken detailed balance and entropy production in the human brain",
  "authors": "Christopher W. Lynn, Eli J. Cornblath, Lia Papadopoulos, Maxwell A. Bertolero, Danielle S. Bassett",
  "year": "2021",
  "url": "https://arxiv.org/abs/2005.02526",
  "finding": "S_dot = sum P_ij log(P_ij/P_ji) on hierarchical k-means (cosine) states, k = 8; coarse-graining gives a lower bound; 100 bootstrap trajectories; shuffle noise floor; brain near detailed balance at rest and broken under tasks; dynamic Ising model with asymmetric couplings reproduces this; PNAS DOI listed.",
  "relevance": "Concrete recipe for O7(b), (c), (e).",
  "how_checked": "Opened abs page and ar5iv full text (methods) via WebFetch."
 },
 {
  "title": "Emergent social conventions and collective bias in LLM populations",
  "authors": "Ariel Flint Ashery, Luca Maria Aiello, Andrea Baronchelli",
  "year": "2025",
  "url": "https://arxiv.org/abs/2410.08948",
  "finding": "Naming game with N = 24, 10 names, memory 5, +100/-50 payoffs; disorder-to-order transition described as symmetry breaking; collective bias without individual bias; committed minorities of 2 % (Llama-3-70B-Instruct) to 67 % (Llama-2-70b-Chat) overturn conventions; Science Advances 11, eadu9368.",
  "relevance": "Agent, Ising, collaboration and entity links; template for O4.",
  "how_checked": "Opened abs page and HTML full text via WebFetch."
 },
 {
  "title": "AI agents can coordinate beyond human scale",
  "authors": "Giordano De Marzo, Claudio Castellano, David Garcia",
  "year": "2024 (Science Advances 12, eaea6091, 2026)",
  "url": "https://arxiv.org/abs/2409.02822",
  "finding": "LLM groups form cohesive units; opinion dynamics governed by a majority force coefficient; critical group size beyond which coordination fails, growing with model capability.",
  "relevance": "Coupling-like parameter of LLM collectives (O4).",
  "how_checked": "Opened arXiv abs page; read abstract and listed journal reference."
 },
 {
  "title": "Emergent Coordination in Multi-Agent Language Models",
  "authors": "Christoph Riedl",
  "year": "2025 (ICLR 2026)",
  "url": "https://arxiv.org/abs/2510.05174",
  "finding": "S_macro(l) = I(V_t;V_{t+l}) - sum_k I(X_k,t;V_{t+l}) and pairwise PID synergy as emergence criteria; N = 10 GPT-4.1 agents; personas give identity-linked differentiation, plus theory-of-mind prompts give goal-directed complementarity; block-shuffle and functional nulls.",
  "relevance": "Information-theoretic test of entity formation (O8).",
  "how_checked": "Opened abs page and HTML (methods, results) via WebFetch."
 },
 {
  "title": "Reconciling emergences: An information-theoretic approach to identify causal emergence in multivariate data",
  "authors": "Fernando E. Rosas, Pedro A. M. Mediano, Henrik J. Jensen, Anil K. Seth, Adam B. Barrett, Robin L. Carhart-Harris, Daniel Bor",
  "year": "2020",
  "url": "https://arxiv.org/abs/2004.08220",
  "finding": "Quantitative definition of downward causation and causal decoupling; practical criteria computable in large systems; applied to Game of Life, flocking and ECoG.",
  "relevance": "Framework behind S_macro.",
  "how_checked": "Opened arXiv abs page; the criterion formula was read only as quoted by Riedl."
 },
 {
  "title": "Collective Alignment in LLM Multi-Agent Systems: Disentangling Bias from Cooperation via Statistical Physics",
  "authors": "Cristiano De Nobili",
  "year": "2026",
  "url": "https://arxiv.org/abs/2605.10528",
  "finding": "LLM agents as spins on an LxL lattice with sampler temperature as control parameter; order-disorder crossovers and susceptibility peaks; gamma/nu close to but incompatible with 2D Ising; bias dominates coupling, giving field-driven crossovers, not genuine phase transitions; llama3.1:8b, phi4-mini, mistral:7b.",
  "relevance": "Warns against reading LLM collective changes as Ising transitions; field and coupling must be separated in O4.",
  "how_checked": "Opened arXiv abs page; read abstract (method details unverified)."
 },
 {
  "title": "Emergence of Biased Consensus in Multi-Agent LLM Debates",
  "authors": "Maya Okawa",
  "year": "2026 (ICML 2026)",
  "url": "https://arxiv.org/abs/2608.02827",
  "finding": "Theory predicts a transition to collective bias when conformity exceeds a threshold set by initial bias and noise; experiments show a finite-size crossover; agent heterogeneity smooths the transition.",
  "relevance": "Noise and heterogeneity as knobs; no phase-transition claims at small N.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "When Is Collective Intelligence a Lottery? Multi-Agent Scaling Laws for Memetic Drift in LLMs",
  "authors": "Hidenori Tanaka",
  "year": "2026",
  "url": "https://arxiv.org/abs/2603.24676",
  "finding": "Quantized Simplex Gossip model; populations break symmetry through mutual in-context learning (memetic drift); drift- vs selection-dominated regimes; scaling laws for drift-induced polarization.",
  "relevance": "Mechanism for symmetry breaking and polarization (two entities) in LLM populations.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Emergence of polarization in networks of large language model agents",
  "authors": "Jinghua Piao, Zhihong Lu, Chen Gao, Fengli Xu, Qinghua Hu, Fernando P. Santos, Yong Li, James Evans",
  "year": "2025",
  "url": "https://arxiv.org/abs/2501.05171",
  "finding": "Thousands of LLM agents develop opinion polarization and self-organised networks with homophilic clustering.",
  "relevance": "Two-group formation in LLM collectives.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Weak pairwise correlations imply strongly correlated network states in a neural population",
  "authors": "Elad Schneidman, Michael J. Berry II, Ronen Segev, William Bialek",
  "year": "2006",
  "url": "https://arxiv.org/abs/q-bio/0512013",
  "finding": "In groups of ten or more retinal neurons, weak pairwise correlations coexist with strongly collective states captured by pairwise maximum-entropy (Ising) models; suggests associative or error-correcting coding; Nature DOI 10.1038/nature04701 listed.",
  "relevance": "Template for O1.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Searching for collective behavior in a network of real neurons",
  "authors": "Gasper Tkacik, Olivier Marre, Dario Amodei, Elad Schneidman, William Bialek, Michael J. Berry II",
  "year": "2014",
  "url": "https://arxiv.org/abs/1306.3061",
  "finding": "Up to 120 neurons; above about 40 neurons a global synchrony term is needed (K-pairwise); metastable collective modes; single-neuron states highly predictable from the rest; PLoS Comput Biol 10: e1003408.",
  "relevance": "K-pairwise model, basins as collective modes, redundancy (holographic sense) for O1/O9.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Thermodynamics and signatures of criticality in a network of neurons",
  "authors": "Gasper Tkacik, Thierry Mora, Olivier Marre, Dario Amodei, Stephanie Palmer, Michael J. Berry II, William Bialek",
  "year": "2015",
  "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4577210",
  "finding": "P_T proportional to exp(-E/T); specific-heat peak grows and moves toward T = 1 with N; S approximately E; T is not a physical temperature; clear peak at N = 100 even from one tenth of about 2e6 spikes (20 ms bins); PNAS 112:11508-11513.",
  "relevance": "Criticality signatures and data requirements for O1.",
  "how_checked": "Read PMC full text via WebFetch."
 },
 {
  "title": "Are biological systems poised at criticality?",
  "authors": "Thierry Mora, William Bialek",
  "year": "2011",
  "url": "https://arxiv.org/abs/1012.2242",
  "finding": "Statistical-mechanics models inferred from protein, neural and flock data sit at special (critical) points in parameter space; J Stat Phys 144:268-302.",
  "relevance": "Criticality framing of max-ent fits.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Zipf's law and criticality in multivariate data without fine-tuning",
  "authors": "David J. Schwab, Ilya Nemenman, Pankaj Mehta",
  "year": "2014",
  "url": "https://arxiv.org/abs/1310.0448",
  "finding": "Zipf-like distributions arise generically when an unobserved variable drives the system (e.g. a stimulus), with no fine-tuning; Phys. Rev. Lett. 113, 068102.",
  "relevance": "Main pitfall for O1 criticality signatures: position, topic or uncertainty act as latent drivers.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Bistability, non-ergodicity, and inhibition in pairwise maximum-entropy models",
  "authors": "V. Rostami, P. Porta Mana, S. Gruen, M. Helias",
  "year": "2017",
  "url": "https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005762",
  "finding": "With positive mean correlations and populations of a couple of hundred or more neurons, pairwise max-ent models give bimodal population activity; Glauber-based Boltzmann learning becomes non-ergodic; fix via an inhibitory non-uniform reference measure; 159 macaque motor-cortex neurons; PLoS Comput Biol 13(10): e1005762.",
  "relevance": "Artefact check for bimodality-based 'two entities' (O1, O5).",
  "how_checked": "Opened the article page; read abstract and first part of the text."
 },
 {
  "title": "Maximum entropy models of neuronal populations at and off criticality",
  "authors": "T. S. A. N. Simoes, F. Lombardi, D. Plenz, H. J. Herrmann, L. de Arcangelis",
  "year": "2026",
  "url": "https://arxiv.org/abs/2511.14872",
  "finding": "Static ME models show specific-heat signatures for critical and supercritical cultures but not subcritical ones; they cannot discriminate critical from supercritical; Phys. Rev. E 114, 054103.",
  "relevance": "Limits of criticality claims in O1.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Ising models of deep neural networks",
  "authors": "Dusan Stosic, Darko Stosic, Borko Stosic",
  "year": "2022",
  "url": "https://arxiv.org/abs/2209.08678",
  "finding": "Maps deep networks to Ising models via their weights; trained networks span wider energy ranges; higher critical temperature from specific heat.",
  "relevance": "Prior Ising mappings use weights, not activation statistics; O1 would be a different, activation-based model.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "Neural Networks as Spin Models: From Glass to Hidden Order Through Training",
  "authors": "Richard Barney, Michael Winer, Victor Galitski",
  "year": "2024",
  "url": "https://arxiv.org/abs/2408.06421",
  "finding": "Neurons as spins, weights as couplings; random initialisation maps to a layered SK spin glass; training quickly destroys the glass and a hidden-order phase appears whose melting temperature grows as a power law; training selects a symmetry-broken state.",
  "relevance": "Weight-based Ising view and symmetry breaking in trained networks.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs",
  "authors": "Cheng Gao, Huimin Chen, Chaojun Xiao, Zhiyi Chen, Zhiyuan Liu, Maosong Sun",
  "year": "2025",
  "url": "https://arxiv.org/abs/2512.01797",
  "finding": "Less than 0.1 % of neurons predict hallucination; CETT on answer tokens with non-answer controls, L1 logistic regression; TriviaQA, 10 samples per question (T 1.0, top-k 50, top-p 0.9); scaling alpha in [0, 3] modulates over-compliance (FalseQA, FaithEval, sycophancy, jailbreak); predictive in base models.",
  "relevance": "Defines the H-population for O1-O4 and O9; the over-compliance link motivates O4.",
  "how_checked": "Opened arXiv abs page; read methods 6.1-6.2 in the saved arXiv HTML copy /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/hchain/dl_hneurons/page.txt."
 },
 {
  "title": "Hallucination Neurons and Where to Find Them: An Investigation into the existence of Hallucination Neurons",
  "authors": "Huseyin Cavus, Sebin Sabu, Joshua Spear, Jaskaran Singh Kawatra, Pavithra Rajendran",
  "year": "2026",
  "url": "https://arxiv.org/abs/2609.29781",
  "finding": "Detection and causal effects replicate in Gemma 3 4B and MedGemma 4B, but 19 of 22 selected H-neurons have |r| > 0.7 with other features; moderate bootstrap stability; weak sparse/dense ranking overlap; the set is a sparse projection of a larger correlated structure.",
  "relevance": "Argues for a population (max-ent) treatment and bootstrap-stable H-sets.",
  "how_checked": "Opened arXiv abs page; read discussion and limitations in the saved HTML copy /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/hchain/dl_hneur_repl/page.txt."
 },
 {
  "title": "Holographic Reduced Representations",
  "authors": "Tony A. Plate",
  "year": "1995",
  "url": "https://redwood.berkeley.edu/wp-content/uploads/2020/08/Plate-HRR-IEEE-TransNN.pdf",
  "finding": "Circular convolution associates items in fixed-width vectors; HRRs are closely related to holographic storage; correlation serves as an approximate inverse of convolution; noisy reconstructions are cleaned up by a separate associative memory; IEEE Trans. Neural Networks 6(3):623-641.",
  "relevance": "Gives 'holographic' an operational, energy-compatible meaning (distributed storage plus clean-up attractor) for O9.",
  "how_checked": "Read PDF pages 1-2 (abstract, introduction, section II start)."
 },
 {
  "title": "Deep Learning and AdS/CFT",
  "authors": "Koji Hashimoto, Sotaro Sugishita, Akinori Tanaka, Akio Tomiya",
  "year": "2018",
  "url": "https://arxiv.org/abs/1802.08313",
  "finding": "Network depth identified with the emergent holographic radial direction; the bulk metric emerges by training on boundary response data; Phys. Rev. D 98, 046019.",
  "relevance": "The physical-holography analogy; requires a network trained on boundary data, so no direct counterpart in Qwen.",
  "how_checked": "Opened arXiv abs page; read abstract."
 },
 {
  "title": "The Chain Lens [interim]",
  "authors": "vgel, Raymond Douglas",
  "year": "2026",
  "url": "https://www.lesswrong.com/posts/Nr6fksEqqxNpdz6rk/the-chain-lens-interim",
  "finding": "J-lens transport plus MTP-head rollouts decoded contrastively against a mean prior state; WorkspaceBench macro 0.420 vs J-lens 0.319; confabulation on the hallucination family grows with chain length; hallucination and performance both scale with chain length (19 families); without contrastive decoding or reinjection, chains revert to generic completions.",
  "relevance": "The readout whose own length effect motivates an independent lens-free test.",
  "how_checked": "Read the saved copy provided by the orchestrator (/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/chainlens_dl/post.json, field text); not fetched by me."
 }
]
```
