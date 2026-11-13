# lens-free research, angle holographic (RESEARCH STAGE, NOT YET VERIFIED)

# Holographic principles as a lens-free, inverse-free arm for H-CHAIN (round 1b, angle 2)

**User question (13.11.2026, translated from German):** Could the hoped-for insights about H-neurons and the chain lens be extended by a lens-free approach that relies on holographic principles? The user cites EBM reasoners and non-equilibrium, and the chain loop transformer -> agent -> Ising-modellable -> collaboration -> entity-forming.

## 1. Answer in brief

**Yes, as an independent layer of measurement, but only for the information-theoretic core of holography.** The physics of AdS/CFT has no counterpart in Qwen3-1.7B: no gravitating bulk, no conformal boundary, no quantum entanglement, no no-cloning. What carries over is what the primary sources state explicitly. It yields five classical signatures, each measurable:
- **lens-free:** no hidden state is read out into the vocabulary;
- **inverse-free:** only forward passes, forward regressions, distances, counts and compression; no matrix inversion, pseudo-inverse, inverse Jacobian or Tikhonov inverse.

| Holographic principle (primary source) | Classical, measurable counterpart in Qwen3-1.7B | Arm |
|---|---|---|
| **Threshold erasure resilience.** A perfect tensor is a ((n, 2n-1)) threshold scheme: any n-1 sites carry no information, any n carry all (Pastawski et al. 2015). Random states give no information until about half the system is held (Page 1993; Blume-Kohout & Zurek 2006). | **Shape of the partial-information plot (PIP).** How much information about the hallucination label is recoverable from random neuron fragments of size m, compared with the H-set. Plus a causal erasure curve by mean-ablation. | HOLO-E |
| **No privileged region.** Bulk operators have representations on many boundary regions (Almheiri et al. 2015; Pastawski et al. 2015). | H-set against size-, layer- and norm-matched random fragments; neuron basis against a randomly rotated basis (Gurnee et al. 2023). | HOLO-E |
| **Monogamy of mutual information,** I3 <= 0, which follows from RT (Hayden, Headrick, Maloney 2013). | Tripartite information I3 = I(X;A)+I(X;B)-I(X;AB) over disjoint fragments; sign of the O-information (Rosas et al. 2019). | HOLO-E |
| **RT: entropy = minimal cut** (Ryu & Takayanagi 2006). Min-cut bounds in networks (Levine et al. 2017). Captured MI is bounded by the history-state size (L2M 2025). | Growth class of the model's captured past-to-future information with context length; fact gain against distance; decoupling horizon of self-rollouts. | HOLO-MI |
| **Bulk reconstruction from boundary data.** | Delay embedding of output-side scalars (surprisal, entropy), cross-mapped to the internal H channel. Takens' theorem is used as in Sugihara et al. 2012 and Ye et al. 2015. | HOLO-RQA, HOLO-CCM |

Two further links in the user's chain become measurable outside holography proper:
- **'Non-equilibrium' (the EBM-reasoner analogy)** becomes time-irreversibility of per-token observables (Roldán & Parrondo 2010; Ro et al. 2022): **HOLO-EP**.
- **'Entity-forming'** becomes either redundant recording of a variable across many fragments, or causal emergence of a macro-variable (Psi > 0, Rosas et al. 2020): **HOLO-ENT**. Redundant recording is the objectivity criterion of quantum Darwinism (Zurek 2009; Blume-Kohout & Zurek 2006). Redundancy is the classical opposite of a holographic threshold, so HOLO-E decides between the two readings.

**What this adds to H-CHAIN:**
1. An independent check that cannot inherit lens confabulation. The chain-lens post itself reports that confabulation grows with chain length; on WorkspaceBench's hallucination family it scores 0.45 against 0.32 for the J-lens (lower is better).
2. A principled denominator for 'length': captured information (past-to-future MI, fact gain, decoupling horizon) instead of token count.
3. Operational definitions of an 'H-neuron attractor': stickiness above surrogates, return after a push, an upstream position in a cross-map hierarchy, and irreversibility on entry.
4. A direct test of 'sparse H-neurons (< 0.1 %) versus holographic spread'.

**Honest prior: a strict holographic code is unlikely.**
- Gao et al. (2025) report that random neuron sets of the H-set's size already classify hallucinations above chance. On TriviaQA this is 56.1-68.4 % for random sets against 70.1-83.6 % for H-neurons, across six models. A threshold code forbids this.
- The expected HOLO-E outcome is therefore 'localised core + redundant halo'.
- The arm is still worth running. It separates localised, redundant and holographic readings by frozen rules, and the redundant reading is exactly the measurable content of 'entity-forming'.

## 2. What the primary sources establish (all opened and read; details in refs)

### 2.1 Holographic codes and recovery from random subsets

**HaPPY codes** (Pastawski, Yoshida, Harlow, Preskill; JHEP 2015). Perfect tensors on hyperbolic tilings serve as exactly solvable toy models.
- A perfect tensor with 2n legs, read as a map from one leg to 2n-1 legs, encodes a [[2n-1,1,n]] code. It protects against erasure of any n-1 spins, 'the best possible protection against erasure errors compatible with the no-cloning principle'.
- It is the basis of a ((n, 2n-1)) threshold secret-sharing scheme: any n-1 spins carry no information about the logical spin, any n carry all of it.
- In the pentagon code, the central bulk operator can be rebuilt on every connected boundary region covering more than (5+sqrt5)/10 ≈ 0.724 of the boundary. Some regions of ≈ 0.524 also suffice; the continuum AdS-Rindler value is 1/2.
- Under independent random erasure below a threshold p_c, the centre is recoverable. The failure probability is doubly exponentially small in the bulk radius.
- RT holds exactly for connected regions in their holographic states, and I3 <= 0 follows wherever RT is exact.

**Monogamy** (Hayden, Headrick, Maloney; PRD 2013). RT implies I(A:BC) >= I(A:B) + I(A:C), which is the same as I3 <= 0.

**RT** (Ryu & Takayanagi; PRL 2006). The entanglement entropy of a CFT region equals the area of a minimal surface in AdS. For AdS3 this reproduces the 2D CFT result.

**Bulk locality as error correction** (Almheiri, Dong, Harlow; JHEP 2015). Bulk locality is read as operator-algebra quantum error correction, and how far reconstruction reaches is tied to quantum secret sharing.

**Random states.**
- Page (PRL 1993): for a random pure state, the smaller subsystem has average entropy ≈ ln m - m/(2n). On average it carries less than half a nat of information.
- Blume-Kohout & Zurek (PRA 2006) define the **partial-information plot (PIP)**: the information about a system obtainable from environment fragments of size m, averaged over fragments. They describe three profiles:
  - independent environments (roughly linear);
  - redundant storage (a 'classical plateau');
  - information encoded across many environments.
- For uniformly random states, 'no significant information is obtained until almost half the subenvironments have been captured'.
- Redundancy R_δ is the number of disjoint fragments that each hold a fraction 1-δ of the information. It serves as the objectivity criterion; Zurek (Nature Physics 2009) frames this as quantum Darwinism.
- Hayden & Preskill (JHEP 2007): after the half-way point, information thrown into a rapidly mixing black hole re-emerges quickly.

**Consequence:** a linear rise, an early plateau and a half-threshold are the three PIP shapes to discriminate.

### 2.2 Networks as holograms: constructions, not findings about trained LLMs
- **Hashimoto et al. (PRD 2018)** map network depth onto the AdS radial direction. Training on boundary response data makes a bulk metric emerge. They applied it to AdS-Schwarzschild data and to magnetic data from Sm0.6Sr0.4MnO3.
- **Swingle (PRD 2012):** organising a state by scale (entanglement renormalisation) defines a geometry, and a critical state gives a discrete AdS.
- **Levine et al. (arXiv 2017), Claim 1:** for convolutional arithmetic circuits, the rank of the network tensor's matricization, taken with respect to an input partition (A,B), is at most the weight of the minimal cut separating A from B in the tensor-network graph. This RT-like min-cut bound is proven for ConvACs and checked empirically on ReLU ConvNets.

None of these papers claims that a trained LLM is a holographic code, and none provides a test for one. They license the translation 'RT -> capacity of a cut', nothing more.

### 2.3 Distributed codes and how redundancy has been measured

**Plate (IEEE TNN 1995): holographic reduced representations.**
- Items are bound by circular convolution and decoded by circular correlation, which Plate calls an 'approximate inverse'.
- A trace 'stores enough information to recognize the vectors in it, but not enough to reconstruct them accurately', so a separate clean-up memory is required.
- This 'recognition, not reconstruction' principle is the holographic-memory precedent for the user's generate-verify preference. HRR decoding itself is not used here, because it is an approximate inverse.

**Superposition** (Elhage et al. 2022): models hold more sparse features than they have dimensions, with a phase change.

**Gurnee et al. (2023):** sparse probes on Pythia 70M-6.9B, with the number of neurons k swept from 256 down to 1.
- Early layers use sparse combinations of neurons; middle layers have dedicated neurons.
- Individual neurons are more predictive than dimensions of a random Gaussian basis.
- Probe loss against k follows a power law and then shows a kink.

**Morcos et al. (ICLR 2018):** cumulative ablation curves.
- Clamping units to their mean damaged performance more than clamping to zero.
- Reliance on single directions tracks poor generalisation.

**Dalvi et al. (EMNLP 2020):** 85 % of neurons in BERT/XLNet are redundant, and at least 92 % can be removed for downstream tasks.

**O-information** (Rosas et al., PRE 2019).
- Ω > 0 marks a redundancy-dominated system, Ω < 0 a synergy-dominated one.
- For three variables, Ω equals the interaction information.
- Its extremes are the n-bit copy (maximum) and the n-bit XOR (minimum, 2-n).

**Causal emergence** (Rosas et al. 2020).
- Psi = I(V_t;V_t') - Σ_j I(X^j_t;V_t') > 0 is sufficient for causal emergence of a macro-variable V.
- Redundancy pushes Psi negative, so Psi <= 0 is inconclusive.

### 2.4 H-neurons and where hallucination information sits

**H-neurons** (Gao et al., arXiv 2512.01797, Dec 2025).
- **Feature (CETT):** a neuron's down-projected output norm divided by the FFN output norm, averaged over answer tokens and over all other tokens.
- **Selection:** an L1 logistic probe; H-neurons are the neurons with positive weights.
- **Fractions:** 0.01-0.35 per mille of neurons in six models (Mistral, Gemma-3, Llama-3); no Qwen model was studied.
- **Data:** TriviaQA, 10 samples per question (T 1.0, top-k 50, top-p 0.9). Only questions answered consistently in all 10 samples are kept: 1,000 + 1,000. Out-of-distribution sets: NQ-Open, BioASQ, NonExist.
- **Baselines:** random equal-size neuron sets classify above chance (see section 1).
- **Intervention:** scaling the H-neurons by α in [0, 3] raises over-compliance.
- **Not reported:** the layer distribution.
- **For Qwen3-1.7B** (28 x 6,144 = 172,032 MLP neurons), these fractions imply about 2-60 H-neurons (our arithmetic).

**Orgad et al. (ICLR 2025):**
- Truthfulness information concentrates in the exact-answer tokens.
- Error detectors do not generalise across datasets.

### 2.5 Mutual-information scaling and context length

**Lin & Tegmark (Entropy 2017).**
- MI between symbols decays exponentially with separation for Markov processes and regular grammars. For context-free grammars it can decay as a power law.
- Measured MI curves for enwik8, French text, the human genome and Bach are roughly power laws.
- An LSTM (3 layers of 256) trained on enwik8 'hallucinated' 1 MB of text whose MI curve is a rough power law with a similar slope.
- They propose MI against separation as a diagnostic of models.

**L2M (Chen et al. 2025).**
- Bipartite MI between past and future grows as a power law L^β on PG19 and Wikipedia ('sub-volume' growth).
- It was estimated with LLaMA 3.1 405B as the density model q, using two estimators. Both likely underestimate β.
  - A direct estimator, E[log q(Y|X) - log q(Y)], corrected for the BOS bias.
  - vCLUB: E_joint log q(Y|X) minus E_product log q(Y|X), where the product term uses shuffled pairs.
- Theorem 5.2: the MI a model captures is at most C·dim(history state) + log M. A transformer's KV cache grows linearly, which over-provisions relative to L^β.

**Other context-length results.**
- Bialek, Nemenman, Tishby (2001): predictive information is finite, logarithmic or power-law in the window length.
- RULER (COLM 2024): only half of 17 models claiming at least 32K context are satisfactory at 32K.
- The bits-per-token series of language has a Hurst exponent of about 0.7 (Alabdulmohsin et al. 2024). These fractal parameters vary far more widely in LLM output than in natural text (Alabdulmohsin & Steiner 2025).
- Qwen3-1.7B's native context is 32,768 tokens (model card).

### 2.6 Lens-free dynamical tools and their LLM uses so far

**Convergent cross mapping (CCM)** (Sugihara et al., Science 2012; abstract) separates causality from correlation in nonseparable, weakly coupled systems that Granger causality does not cover. Ye et al. (Sci Rep 2015) add a lag test:
- true causation gives an optimal cross-map lag <= 0;
- synchrony from strong forcing gives a positive optimal lag;
- the method suits weak to moderate coupling and may fail for purely stochastic drivers.

**Recurrence quantification (RQA)** (Marwan et al., PRE 2002) defines laminarity Λ, trapping time TT (v_min = 2 for maps) and Vmax, on Takens delay vectors.

**Embedding under noise** (Muldoon et al. 1998): an embedding theorem that allows dynamical noise needs multivariate signals.

**IAAFT surrogates** (Schreiber & Schmitz, PRL 1996) preserve the data's autocorrelations and amplitude distribution.

**Irreversibility.**
- Roldán & Parrondo (PRL 2010): ⟨Ṡ⟩/k >= lim (1/n) D_n(p_F||p_B) from one stationary trajectory; partial observation gives a lower bound.
- Ro et al. (PRL 2022) estimate local entropy production as the forward/time-reversed KL divergence, using Ziv-Merhav cross-parsing.
- Martiniani et al. (PRX 2019) use the length of a lossless (LZ77) compression as an order measure.

**Uses on LLMs so far.**
- Pham et al. (2026) ran RQA (DET, LAM, ENTR) on final-layer states of DeepSeek-R1-Distill-Qwen-7B reasoning traces, with cosine distance and a 10th-percentile threshold. They reached 36.9 % against 29.0 % for a length baseline on 9-way complexity classification, with no surrogate null and no length control.
- Jaca et al. (2026, abstract only) report chaos signatures.
- Fernando & Guitchounts (2025, abstract only) report attractor-like dynamics across the lower layers.
- Papadopoulos et al. (ICML 2024) find a small but consistent forward/backward asymmetry in log-perplexity.
- One search found no application of CCM to LLMs.

## 3. Six measurements (full specs in the structured list)

| ID | Decides | Core statistic | Counts as holographic | A100 min |
|---|---|---|---|---|
| HOLO-E | Is the hallucination signal localised (H-set), redundant, dense or holographic? Is behaviour erasure-robust? | PIP Î(m), privilege P, R_0.1, I3(m), complementary-half ratio, mean-ablation erasure curve | No privileged subset **and** near-zero information for fractions <= 0.25 with a rise of <= 1 octave **and** I3 <= 0 | ≈105 |
| HOLO-MI | Is 'length' an information limit, a position artifact or an H-attractor? | Captured MI Î(l) (vCLUB form), fact gain FG(l), decoupling horizon k*, hysteresis | Logarithmic growth of Î(l) (RT/CFT-like); descriptive only | ≈110 |
| HOLO-RQA | Is H-dynamics sticky beyond surrogates, and does it return after a push? | LAM, TT at a fixed recurrence rate against IAAFT and block surrogates; return fraction | Joint recurrence of boundary (s,e) and bulk (a) embeddings above surrogates | ≈35 |
| HOLO-CCM | Lens-free channel hierarchy: is H upstream or downstream? | Convergent cross-map skill ρ(L), optimal lag, intervention check | Convergence of the cross-map from (s,e) to a: output-side channels encode the internal H channel | ≈45 |
| HOLO-EP | Is entry into confabulation irreversible? | Forward/reverse KL rate d̂_n, Ziv-Merhav cross-parsing | Not a holographic quantity | CPU |
| HOLO-ENT | Is an 'entity' a redundant record or a causally emergent variable? | R_0.1 ranking of round-2 variables; Psi | Redundancy is the classical opposite of a holographic threshold | ≈15 |

**Common design choices.**
- **Splits:** H-neurons are selected on split S1; every HOLO quantity is evaluated on a disjoint split S2.
- **Calibration:** each pipeline must pass cases with known answers before its verdict counts. These are planted codes, surrogates of surrogates, a synthetic coupled pair and a bf16-versus-fp32 check (gates G-cal, G-surr, G-ccm, G-dtype).
- **Statistics:** intervals are 90 % bootstraps over prompts, items or documents with 2,000 replicates; Holm correction within each measurement's verdict family.
- **Numerics:** float32 with TF32 off, except the long-context passes of HOLO-MI. Those run in bf16, justified there and checked against fp32.

**Reconstruction follows the Aletheia pattern.**
- **What the H-set encodes:** candidate meanings are recognised through forward contrast sets, with at most 3 revisions and abstentions counted (Plate's clean-up principle).
- **Takens reconstruction:** it counts only if its boundary-to-bulk cross-map is verified forward, on held-out trajectories and against surrogates (HOLO-CCM).

**Reuse and new code.**
- Reused: round-2 block hooks (r2_rk.open_block) and the round-2 axes (style direction, u_T_perp).
- New: CETT hooks, neuron-scaling hooks, fragment sweeps, and post-processing for RQA, CCM and EP.

## 4. How the measurements answer the H-CHAIN question

**Verdict 'H-attractor beyond length'** needs agreement of at least 3 of these 4 lines:
1. **HOLO-MI:** H-activity predicts confabulation beyond fact gain (ΔAUC >= 0.03), or stays elevated after the context is supplied again (hysteresis).
2. **HOLO-RQA:** in H windows the trapping time exceeds the surrogates, and H-activity returns after a downward push.
3. **HOLO-CCM:** 'H upstream' (H -> surprisal/entropy with lag <= 0), confirmed by the H-scaling intervention.
4. **HOLO-EP:** the H channel turns irreversible on entering confabulation, beyond both the surrogate band and the real-entity condition.

**Verdict 'length artifact (information-limited)'** needs all three:
- **HOLO-MI:** adding log-distance after fact gain changes AUC by less than 0.02, and H-activity is flat once fact gain is given.
- **HOLO-RQA:** no H-specific stickiness.
- **HOLO-CCM:** H is downstream or independent.

**Verdict 'position artifact':** padding that shifts the absolute position, at fixed distance, reproduces the rise.

Anything else is 'mixed / not decidable'. A single line on its own is descriptive.

**Localised versus holographic** is decided by HOLO-E alone. The dynamic lines then say whether the localised core or the halo carries the dynamics.

**Cross-check with the chain lens.** If the chain-lens arm produces per-item break-off lengths, compare them with the lens-free decoupling horizon k*:
- Spearman >= 0.3 with a 90 % CI above 0: the lens break-off tracks the loss of state-specific information, a length effect that does not depend on the lens.
- CI including 0: the break-off is intrinsic to the lens.

## 5. Budget and sessions

**Planning rates** (assumptions, to be replaced by a 2-minute timing probe with a time guard):
- fp32 teacher forcing at <= 2k context: ≈ 2,000 tokens/s;
- fp32 sampled generation with hooks, batch 64: ≈ 1,000-1,200 tokens/s;
- bf16 teacher forcing at 8k-32k context: ≈ 15,000 tokens/s.

**Sessions:**
- **S-HOLO-1:** HOLO-E, the HOLO-RQA trajectories, HOLO-EP and HOLO-ENT, ≈ 160 A100-min.
- **S-HOLO-2:** HOLO-MI and HOLO-CCM with interventions, ≈ 155 A100-min.

The total is ≈ 5.3 A100-h; about 17 min of TriviaQA sampling is shared with the H-neuron identification in H-CHAIN. A trimmed core takes ≈ 3.2 h: fewer fragments in HOLO-E, only parts (b) and (c) of HOLO-MI, and CCM without interventions.

## 6. What would count against the user's hypotheses

**Against 'holographic':**
- HOLO-E returns 'localised': the H-set is privileged and random fragments need many neurons. The 0.1 % H-neurons are then real, localised units.
- Or HOLO-E returns 'redundant' or 'dense' with I3 > 0: the signal is spread out, but without a threshold and without monogamy, which is ordinary distributed coding.

**Against 'an H-neuron attractor not bound to length':** HOLO-MI says 'information-limited', HOLO-RQA finds no stickiness beyond surrogates, HOLO-CCM puts H downstream, and HOLO-EP finds no H-specific irreversibility.

**Against 'entity-forming':** redundancy does not order the round-2 variables, and Psi <= 0. Psi <= 0 is inconclusive by construction, so only a positive Psi can count in favour.

## 7. Limits, unverified items, files

**Not opened by me:**
- Takens (1981). Known only through Marwan et al. 2002 and Ye et al. 2015, both of which were opened.
- Stark et al. (2003) on stochastic-forcing embeddings (search snippet only).
- Thiel et al. (2006) twin surrogates (bot check; not opened).
- Vidal's MERA papers.
- Pairwise maximum-entropy (Ising) papers. If the Ising arm fits H-neuron couplings, it must avoid the mean-field inverse J = -C^{-1}, which violates the user's rule; pseudo-likelihood or Boltzmann learning by forward sampling are allowed.

**Not extracted:** L2M's numerical exponents (they appear only in figures).

**Estimates only:** all rates and minimum detectable effects above are planning estimates, not measurements.

**Files:**
- The chain-lens post was read from the workflow's local copy (/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/chainlens_dl/post.json).
- The Qwen3-1.7B config was read from a sibling agent's download (/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/hchain_mtp/dl_hf_cfg/Qwen_Qwen3-1.7B.config.json).
- Downloaded paper texts are in /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/holo_src/. Each source sits in its own directory and was converted with pdftotext; nothing was executed.


## operationalisations
```json
[
 {
  "name": "HOLO-E: erasure / partial-information plot of the hallucination signal (localised vs redundant vs holographic)",
  "what_it_decides": "Whether the hallucination signal carried by Qwen3-1.7B MLP neurons is (a) localised in the H-set, (b) recorded redundantly (early plateau, 'objective'), (c) spread densely (smooth rise), or (d) holographic in the strict information-theoretic sense. Also whether behaviour is robust to erasure. Frozen verdict strings: 'localised', 'localised core + redundant halo', 'redundant (objective)', 'dense-distributed', 'holographic (threshold, monogamous)', 'unclassified (profile reported)', 'not decidable (no H-set / probe ceiling below 0.1 bit)'. 'Holographic' requires all three: (1) no privileged subset: Î(H) is not above the 95th percentile of layer- and CETT-norm-matched random fragments of the same size; (2) threshold: the median Î of uniform random fragments stays below 0.1·Î(all) up to a fragment fraction of 0.25, and the rise from 0.1 to 0.9·Î(all) spans at most one octave of m; (3) monogamy: I3(m) = Î(A)+Î(B)-Î(A∪B) <= 0 for disjoint random A, B, with the 90% CI excluding positive values at 3 or more tested sizes (at m = N/2 this is the same as the complementary-half ratio [Î(A)+Î(A^c)]/Î(all) <= 1). 'Localised': Î(H) >= 0.5·Î(all) and above that 95th percentile, and fragments that exclude H need at least 4x more neurons than uniform fragments to reach 0.5·Î(all). 'Redundant (objective)': R_0.1 = N/m_0.1 >= 10 and I3 > 0. 'Dense-distributed': transition width >= 3 octaves, I3 > 0, H not privileged. 'Localised core + redundant halo': the localised and redundant conditions both hold. 'Holographic' names only the threshold/monogamy pattern of perfect-tensor and random-state codes; quantum complementary recovery has no classical requirement.",
  "measurement": "Items: TriviaQA questions; Qwen3-1.7B with chat template, thinking off. Each question is sampled 10 times (T 1.0, top-k 50, top-p 0.9, as Gao et al.). Kept: questions answered correctly 10/10 (faithful) and wrongly 10/10 without refusal (hallucinated). Split by question: S1 (>= 1,000 + 1,000) for H-set identification by the H-CHAIN arm's L1 probe on CETT; S2 (600 + 600) for every HOLO-E statistic. Out-of-distribution sets: 600 NQ-Open questions and 300 fabricated-entity questions. Features: per response and per neuron (28 x 6,144 = 172,032), the mean CETT over answer tokens, CETT_j,t = |z_j,t|·||W_down[:,j]|| / ||FFN_out_t||. Readouts: an L2 logistic probe (Adam; strength chosen by nested CV inside the S2 training folds), 5-fold CV by question. Î(F) = 1 bit minus held-out cross-entropy, a lower bound on I(X;F); AUROC reported too. A second probe family, an MLP with one hidden layer of 64 units, detects synergy. Fragment sizes m in {1, 2, 4, ..., 131072, 172032}. Fragment families: uniform random; random matched to the H-set's layer and CETT-norm distribution; random-rotated basis (per-layer random orthogonal matrix, then coordinate sampling; Gurnee-type control); uniform excluding H; the H-set and its nested prefixes by probe weight. 50 fragments per size (10 for m >= 16,384). Indices: PIP curve, privilege P = Î(H) minus the 95th percentile of matched fragments, R_0.1, transition width, complementary-half ratio, I3(m), O-information sign over 3 disjoint fragments. Causal erasure: mean-ablate (and, separately, zero-ablate) nested random fragments of size 2^4 to 2^16, the H-set and damage-matched random sets during greedy generation on S2 (24 tokens). Measured: hallucination rate, abstention rate, teacher-forced delta log p(gold); damage is teacher-forced KL on 64 neutral reference texts (round-2 convention). Position PIP: random subsets of token positions (mean-pooled residual states at blocks 14, 18, 22) against the exact-answer positions (Orgad et al.). Inverse-free decoding of what the H-set encodes, generate-verify: candidate meanings (hallucination, false-premise compliance, unfamiliar entity, low confidence) come from forward rollouts. The verifier is a forward contrast set varying one factor; a candidate is accepted if H-population activity separates the set with z >= 3 against random equal-size populations. At most 3 revisions; abstentions counted; verifier error rate measured on planted features. Gates: G-H (H-set held-out accuracy >= 65% and above the 95th percentile of 50 random equal-size sets); G-cal (planted codes recovered in at least 4 of 5 seeds). MDE: the SE of Î is about 0.015-0.02 bit on 1,200 items, so a privilege of >= 0.05 bit (about 5 accuracy points; Gao et al.'s gaps are 10-20 points) and |I3| >= 0.04 bit are detectable; for erasure, a change in hallucination rate of >= 4 points (paired, n = 500). Intervals: 90% bootstrap over questions, 2,000 replicates; Holm within the verdict family.",
  "null_and_controls": "Label-permutation null (200 permutations) sets the floor for Î. Further controls: layer- and CETT-norm-matched random fragments; a random-rotation basis to test alignment with individual neurons; fragments that exclude H. Planted-code calibration runs on the real CETT matrix with synthetic labels: localist (signal in 20 random neurons), redundant (2,000 copies), dense (all neurons, small SNR) and secret-sharing (XOR of the signs of 3 disjoint groups). The pipeline must return the right profile for each before the real verdict counts. Random ablations are matched on damage (equal KL); zero- and mean-ablation are both reported, since Morcos et al. found mean-clamping more damaging. Out-of-distribution sets guard against TriviaQA idiosyncrasy.",
  "data_needed": "TriviaQA (about 5,000 questions sampled, shared with the H-CHAIN identification); 600 NQ-Open questions; 300 fabricated-entity questions from a frozen template; 64 neutral reference texts for damage; the H-set from S1; a CETT feature matrix of about 3,200 x 172,032 (fp16, about 1.1 GB per position type).",
  "cost_estimate": "About 105 A100-min in fp32 with TF32 off. Sampling 1.2M tokens at about 1,200 tok/s: 17 min (shared with H-CHAIN). CETT extraction: 3 min. Batched probe sweeps: 25 min. MLP probes: 10 min. Causal erasure (180 conditions, each 12k generated plus 16k teacher-forced tokens): 45 min. Planted controls: 5 min. Runs in session S-HOLO-1.",
  "lens_free": "No hidden state is mapped to the vocabulary. Inputs are neuron contributions (CETT), probe predictions of a behavioural label (hallucinated or faithful answer), the model's own generated answers, and residual states pooled over positions.",
  "inverse_free": "Probes are forward regressions fitted with Adam (no quasi-Newton step, no matrix inversion). Information is a difference of cross-entropies. Random rotations come from QR orthogonalisation of a Gaussian matrix and are applied forward; they are not the inverse of anything. Ablation and scaling are forward interventions. The 'what does H encode' step is generate-verify with forward contrasts and a fixed attempt limit.",
  "pitfalls": "Probe information is a lower bound whose slack grows with m. This biases I3 upward (towards 'redundant') and depresses the high-m end of the PIP, so a measured I3 <= 0 is conservative and I3 > 0 is partly estimator bias; planted codes calibrate this. Linear codes with independent Gaussian noise give I3 >= 0, because mutual information is concave in SNR, so 'redundant' is the classical default; I3 < 0 needs correlated noise or XOR-like codes, hence the MLP probe. Circularity if H is selected and evaluated on the same items. Keeping only 10/10-consistent items selects extremes and inflates separability, so out-of-distribution results are reported. CETT divides by the FFN output norm, which couples the neurons of a layer. Late layers carry answer identity, a layer confound handled by matched fragments. Qwen3-1.7B may have only 2-60 H-neurons, which makes the H-curve noisy. Gao et al.'s random baselines (56-68% on TriviaQA) already contradict a strict threshold code in six other models."
 },
 {
  "name": "HOLO-MI: captured past-to-future information vs context length, fact distance and rollout length (principled 'length' explanation)",
  "what_it_decides": "Whether hallucination and H-neuron activity rise where the model's captured past-to-future information stops growing or is lost (a length effect that is really an information limit); whether absolute position drives the rise instead (position artifact); or whether H-activity rises or persists regardless of captured information (H-attractor beyond information). Verdicts: 'information-limited (length explained)', 'position artifact', 'H-attractor beyond information', 'not decidable (no saturation within 32k or CIs overlap)'. Rules: 'information-limited' if adding log distance after fact gain FG changes the AUC for confabulation by < 0.02 and H-activity has no slope in distance given FG (90% CI includes 0). 'Position artifact' if the padding control reproduces >= 50% of the rise at fixed distance and FG. 'H-attractor' if H-activity predicts confabulation beyond FG and distance (delta AUC >= 0.03, CI > 0), or stays elevated after the context is supplied again in >= 60% of rollouts against <= 20% in foreign-context controls. Holographic content: in RT/AdS3, entanglement of a region grows like log of its size; L2M finds power-law (sub-volume) growth for natural text; saturation would mean area-law-like finite memory. The growth class of the captured-information curve is reported descriptively; no verdict depends on calling it holographic.",
  "measurement": "(a) Captured-information curve. Documents: 48 PG19 test books and 24 texts Qwen generates itself (16k tokens, T 1.0, Gumbel-max). Future block Y = 256 tokens at a fixed cut; pasts X_l of length l in {0, 64, ..., 32768} (native context 32,768). Î(l) = mean over documents of [log q(Y|X_l) - log q(Y|X'_l)], where X'_l is a past of the same length from another document of the same source (vCLUB form, L2M Eq. 7). Also the context gain G(l) = E[log q(Y|X_l) - log q(Y|no past)]. Growth class by frozen model comparison (saturating / logarithmic / power law) for l in [256, 32768]. L_sat = the smallest l whose doubling increment is below 2 bootstrap SE and below 5% of Î(max). (b) Fact gain vs distance, with hallucination labelled by construction: 200 synthetic facts about fabricated entities, placed at distance l in {0, 256, 1k, 4k, 8k, 16k, 30k} before the question, with PG19 filler. Position control: the same configuration shifted later in absolute position by neutral leading padding. FG(l) = log q(gold | context with the fact) - log q(gold | context with a matched foreign fact), teacher-forced. The greedy answer is scored correct, confabulated or abstain; H-activity a = summed CETT over the H-set on the answer tokens. (c) Decoupling horizon, a lens-free analogue of chain length: 300 contexts (factual and fabricated-entity QA), greedy rollout of 64 tokens. At each step k, compare the next-token distributions under the true context and under a matched foreign context with the same rollout prefix: kappa(k) = Jensen-Shannon divergence. k* = the first k inside the foreign-vs-foreign null band (95th percentile). a(k) is tracked along the rollout. Hysteresis: at step k*+8 the question or fact is supplied again, and we test whether a(k) returns to its pre-k* level within 16 tokens. Decision regressions (frozen): logistic regression of confabulation on [FG, log distance, log absolute position]; linear regression of a on the same terms plus the outcome. Cross-check: if the chain-lens arm yields per-item break-off lengths, Spearman(break-off, k*) >= 0.3 with CI > 0 means the lens break-off tracks loss of state-specific information. Gate G-dtype: bf16 and fp32 agree within |delta Î| < 0.1 nats per block on 4 documents at l <= 8k. MDE: about 0.03 AUC for the regression terms (1,400 trials); the saturation threshold is defined relative to the bootstrap SE (frozen rule). Intervals: 90% bootstrap over documents and items, 2,000 replicates.",
  "null_and_controls": "Foreign-past null (same source, same length). Sentence-order shuffle within a document (keeps local statistics, destroys long-range ones). Absolute-position padding control. Foreign-fact control for FG. Foreign-context rollouts (k* = 0 by construction) as length-matched controls for a(k). L2M's BOS issue is avoided by using the foreign-past form instead of q(Y|no past) for verdicts.",
  "data_needed": "PG19 test split (public domain); 24 self-generated long texts; 200 fabricated-entity facts and question templates (frozen); 300 rollout contexts; H-set; filler text.",
  "cost_estimate": "About 110 A100-min. (a) About 20M teacher-forced tokens in bf16 at about 15k tok/s: 25 min, plus generating 24 x 16k tokens (bf16, batch 8): 30 min. (b) About 35M prefill tokens: 40 min. (c) 5 min in fp32. fp32 cross-check: 10 min. bf16 is justified: fp32 would take about 5 h for (a) alone, bf16 is the model's native dtype, and G-dtype checks the difference. Runs in session S-HOLO-2.",
  "lens_free": "Uses only the model's own output distribution (log q of the actual tokens, the next-token distributions it would emit) and neuron contributions. No transport of hidden states to the vocabulary, no MTP head, no chain lens.",
  "inverse_free": "All quantities are forward log-likelihood differences or divergences between forward distributions. Growth-class fits are simple regressions (slope = cov/var) with model comparison by residuals; no matrix inversion.",
  "pitfalls": "The model's log-probs measure the information the model captures, not the true MI (L2M notes the exponent is underestimated). q(Y|no past) is biased by mid-sentence starts. RoPE and absolute position confound distance, hence the padding control. Behaviour beyond 32,768 tokens is undefined. Copying and repetition in self-generated text inflate MI. Synthetic facts invite lexical matching, so paraphrased variants are used. RULER shows effective context below claimed context, so saturation may come early and depend on the task. bf16 numerics at long context."
 },
 {
  "name": "HOLO-RQA: delay embedding + recurrence trapping time of per-token observables vs surrogates (attractor stickiness)",
  "what_it_decides": "Whether generation prone to confabulation shows laminar trapping (an attractor in which the trajectory lingers) beyond what linearly correlated noise with the same spectrum and amplitude distribution produces. Whether that stickiness is carried by H-activity rather than by entropy or repetition. Whether H-activity returns after a push (an operational attractor). Verdicts: 'H-attractor (sticky, returns after push)', 'sticky but not H-specific', 'no stickiness beyond surrogates', 'not decidable (surrogate band too wide)'. 'H-attractor' requires all three: (1) the TT of the a-embedding in H-high windows exceeds the per-window IAAFT 95th percentile in >= 60% of windows; (2) TT(H-high) - TT(H-low) > 0 with the 90% CI above 0 after matching windows on mean entropy and position; (3) after scaling H by 0.5 for 16 tokens in H-high windows, a recovers >= 50% of its excess within 32 tokens in >= 60% of trajectories, against <= 30% after an equal-damage random-neuron push. Holographic reading: Takens-type reconstruction of a bulk channel from boundary data. It counts only if the joint recurrence of the boundary (s,e) embedding and the bulk (a) embedding exceeds the surrogate band and the HOLO-CCM boundary-to-bulk cross-map converges.",
  "measurement": "1,200 trajectories of 1,024 tokens each: 4 prompt families x 75 prompts x T in {0.7, 1.0} x 2 keyseeds, Gumbel-max sampling as in round 1. Families: well-known real entities (chosen by 10/10 TriviaQA correctness); fabricated entities (confabulation guaranteed); round-2 RSI loop prompts; sea control. Per token: surprisal s_t of the sampled token; entropy e_t of the model's next-token distribution; H-activity a_t = summed CETT over the H-set (if |H| < 10, also an extended set of the top 100 neurons by L1-path entry, reported separately); projections onto the round-2 style direction and u_T_perp at blocks 14 and 18. Each trajectory is z-scored. Multivariate delay vectors of (s,e) and of a with m = 4, tau = 1 (frozen; sensitivity m = 3 and 6). Windows of 256 tokens, step 64. The recurrence threshold is set per window for a fixed recurrence rate of 5% (Euclidean). RQA: DET (l_min 2), LAM and TT (v_min 2, Marwan 2002 Eqs. 2-3), Vmax; Theiler window 2. H-high and H-low windows are defined from the preceding window, which avoids regression to the mean. Perturb-and-relax: Gao-style scaling of the H-neurons (alpha 0.5 in H-high, 1.5 in H-low windows, for 16 tokens), then the fraction of a's excess that returns within 32 tokens; this is an H analogue of the round-2 loop gain G. Gate G-surr: on surrogate-of-surrogate data the test's false-positive rate must be <= 0.10 at nominal 0.05. MDE: a standardised TT difference of about 0.25-0.3 between families (300 trajectories each). Intervals: 90% bootstrap over prompts, 2,000 replicates.",
  "null_and_controls": "Per-channel IAAFT surrogates (100 per window; same autocorrelation and amplitude distribution; a time-reversible linear null). Block-shuffle surrogates (16-token blocks) as a local-structure null. Windows matched on length, mean entropy and position across families. Equal-damage random-neuron push. Temperature contrast, because repetition loops at low T give trivial laminarity.",
  "data_needed": "New generated trajectories with per-token hooks; H-set; round-2 style direction (theta R4 generic attractor) and u_T_perp (round-1 files via round-2 conventions; if unavailable, recompute the style direction as the leading mode of round-2 loop end directions); frozen prompt lists.",
  "cost_estimate": "About 35 A100-min. 1.23M generated tokens with hooks at about 1,000 tok/s: 21 min. Perturb-and-relax, 0.15M tokens: 3 min. RQA and IAAFT on GPU: about 10 min (about 1 h on CPU). Runs in session S-HOLO-1; the trajectories are reused by HOLO-CCM, HOLO-EP and HOLO-ENT.",
  "lens_free": "The observables are the model's own output statistics (surprisal, entropy), neuron contributions and projections onto fixed directions. Nothing is decoded into the vocabulary.",
  "inverse_free": "Delay vectors, distances, thresholds and line counts. Surrogates are generated with FFTs: the inverse DFT inside IAAFT is an exact unitary change of basis used to create null data, not a reconstruction of causes, and the block-shuffle null avoids it entirely. Pushes are forward interventions. The embedding counts as a reconstruction only after forward verification by cross-mapping (HOLO-CCM).",
  "pitfalls": "Takens assumes deterministic, autonomous, finite-dimensional dynamics. LLM sampling is stochastic and prompt-driven, and the state (KV cache) grows, so RQA here is descriptive statistics against nulls, not proof of an attractor; embedding theorems with noise need multivariate signals (Muldoon et al. 1998). Low temperature produces repetition loops with trivial laminarity, so entropy is matched. IAAFT tests a linear Gaussian null; rejecting it does not show determinism. The fixed recurrence rate and the window length affect TT. Pham et al. 2026 used no surrogates and no length control, so their numbers are not comparable. A tiny H-set makes a_t noisy."
 },
 {
  "name": "HOLO-CCM: convergent cross mapping between per-token channels (lens-free channel hierarchy, lag direction, boundary-to-bulk recoverability)",
  "what_it_decides": "Which channel's history can be recovered from which other channel's delay embedding, and with which lag direction. This gives a lens-free hierarchy of channels, the user's channel fronts with their forward-vs-backward predictive ratio. It also decides whether H-activity is upstream or downstream of surprisal/entropy and of the style and RSI projections; whether external conditions reorder the hierarchy; and whether output-side (boundary) channels encode the internal H channel. Verdicts per ordered pair: 'X->Y (convergent, optimal lag <= 0)', 'synchrony (convergent both ways, positive optimal lag)', 'no detectable coupling', 'not decidable'. Global verdicts: 'H upstream', 'H downstream', 'H independent'; plus 'boundary encodes bulk' if the cross-map from (s,e) to a converges above surrogates. A CCM direction counts as verified only if the H-scaling intervention agrees with it. Holographic reading: reconstruction of the bulk from the boundary. It counts if channels computed only from the output recover a_t with convergent skill above surrogates on held-out trajectories.",
  "measurement": "Trajectories from HOLO-RQA. Channels: a_t, s_t and e_t (s and e treated as one boundary block), style_t and rsi_t at blocks 14 and 18. Embedding dimension E per channel by simplex self-forecast on a calibration third of the trajectories (E in 2..8, tau = 1), frozen before the test thirds are read. Libraries L in {32, 64, 128, 256, 512, 896}, 100 random libraries each. Skill rho = Pearson correlation of estimate and observation. Convergence: rho(L_max) - rho(L_min) > 0 with the 90% bootstrap CI over trajectories above 0, and rho(L_max) above the 95th percentile obtained when the putative driver is replaced by IAAFT surrogates. Lag test (Ye et al. 2015): lags -8..8; X->Y if the optimal lag <= 0, synchrony if it is > 0. The user's vector quantity: per pair, the lag profile rho(l) for l < 0 against l > 0, and its asymmetry. Hierarchy: net flow per channel; reordering across conditions (fabricated vs real, T 0.7 vs 1.0, RSI vs sea) tested by permutation over trajectories. Known-answer checks (gate G-ccm): a synthetic coupled pair with known direction, observation noise and discretisation must give the right direction in >= 90% of 50 realisations. Interventional check: scale the H-set by 0.5 and 1.5 from token 256 on, and a damage-matched random set likewise. 'H->s' counts as verified only if scaling H shifts surprisal/entropy statistics beyond the random control; the agreement rate is reported as the CCM error estimate. Time-reversed series serve as a check: a genuine lagged direction should not survive reversal unchanged. MDE: rho differences of about 0.05 with 300 trajectories per condition. Intervals: 90% bootstrap over trajectories, 2,000 replicates.",
  "null_and_controls": "IAAFT surrogates of the putative driver. Cross-trajectory pairing (the cause taken from another trajectory of the same prompt family) removes common-prompt effects. Equal-damage random-neuron intervention. Condition permutation. Synthetic known-direction pair.",
  "data_needed": "HOLO-RQA trajectories (1,200 x 1,024 tokens); H-set; round-2 axes; 600 intervention trajectories.",
  "cost_estimate": "About 45 A100-min: batched kNN cross-mapping on GPU about 30 min; intervention runs (2 x 300 x 1,024 tokens) about 10 min; synthetic checks on CPU. Runs in session S-HOLO-2.",
  "lens_free": "Cross-maps only scalar per-token channels computed from the model's own outputs, neuron contributions and fixed projections; there is no vocabulary readout.",
  "inverse_free": "Simplex projection (nearest neighbours with exponential weights) and correlations. No S-map regression, no matrix inversion. Interventions are forward passes.",
  "pitfalls": "CCM theory assumes deterministic coupled dynamics with weak to moderate coupling. Token sampling is stochastic, and every channel is a function of the same context and sampled tokens; this common driver can create mutual cross-mapping. Strong coupling produces synchrony, hence the lag test. s and e are mechanically related. Within one token, blocks 14 -> 18 -> logits run in a fixed feed-forward order, so only lags >= 1 across tokens support a dynamic claim. Non-stationarity over position. Libraries of <= 900 vectors limit convergence. No published CCM application to LLMs was found, so there is no reference value."
 },
 {
  "name": "HOLO-EP: time-irreversibility (entropy-production lower bound) of per-token channels at entry into confabulation",
  "what_it_decides": "Whether entering a confabulation regime is a non-equilibrium, irreversible process in lens-free observables, following the user's EBM-reasoner / non-equilibrium analogy: falling into an energy well should show a burst of irreversibility at entry. Verdicts: 'irreversible H-channel at confabulation entry' (d̂ of the a channel above the surrogate band and above the real-entity condition, 90% CI); 'irreversibility not condition-specific'; 'reversible within resolution'. This is not a holographic quantity; it is included because the user's chain names non-equilibrium.",
  "measurement": "Each channel is symbolised per trajectory by quartiles (alphabet of 4). d̂_n = (1/n) D_n(p_F || p_B) for substring lengths n = 2, 3, 4 (Roldan & Parrondo estimator: plug-in with pseudo-counts, pooled within condition). Compression check by Ziv-Merhav cross-parsing between forward and reversed strings (Ro et al. 2022). LZ77 computable information density serves as an order index (Martiniani et al. 2019). Event-locked windows: 64 tokens before vs after the first window (after token 128) in which a exceeds its trajectory's 80th percentile; matched windows from real-entity trajectories at the same positions. Gate G-surr as in HOLO-RQA. Intervals: 90% bootstrap over trajectories.",
  "null_and_controls": "IAAFT surrogates (a time-reversible linear null) give the d̂ band. Further controls: block shuffles; windows of equal length matched on entropy; the real-entity vs fabricated-entity contrast. Because the perplexity asymmetry of reversed text is small (Papadopoulos et al. 2024), only differences between conditions are interpreted.",
  "data_needed": "HOLO-RQA trajectories; H-set.",
  "cost_estimate": "CPU only, a few minutes; no extra A100 time. Post-processing within session S-HOLO-1.",
  "lens_free": "Symbol strings built from the model's own output statistics and neuron contributions; no vocabulary readout.",
  "inverse_free": "Substring counts, log ratios and compression lengths; no inversion.",
  "pitfalls": "Plug-in KL is biased when reversed substrings are never observed; pseudo-counts and surrogate calibration address this. Partial observation gives only a lower bound. Token statistics are not physical states, so this is an irreversibility index, not thermodynamic dissipation. Event alignment can select fluctuations, hence pre-defined thresholds and matched controls."
 },
 {
  "name": "HOLO-ENT: entity formation as redundant recording (quantum-Darwinism objectivity) and causal emergence",
  "what_it_decides": "Whether the user's 'collaboration -> entity-forming' step has a measurable counterpart. (a) Redundancy ranking: how redundantly each round-2 variable (style direction, sea topic, RSI axis u_T_perp, H-population activity) is recorded across random fragments of block-14 residual coordinates and of MLP neurons. Frozen prediction of the Darwinian reading: R_0.1(style) > R_0.1(sea) > R_0.1(RSI), i.e. the loops' generic attractor is the most redundantly recorded variable. (b) Whether the H-population (and the style projection) is a causally emergent macro-variable, Psi > 0. Verdicts: 'style most redundant (Darwinian reading supported)', 'redundancy does not order the variables', 'H-population causally emergent (Psi > 0)', 'inconclusive (Psi <= 0)'. Redundancy is the classical opposite of a holographic threshold: an 'entity' in this sense is anti-holographic.",
  "measurement": "(a) 1,000 held-out texts (round-2 positive, negative and neutral sets, 64 tokens each); residual states at block 14. Targets: projection scores on each round-2 axis and the per-text mean H-activity. PIP over random fragments of residual coordinates, with a random-rotation control because the residual basis is not privileged, and over MLP neurons. R_0.1 per variable with a 90% bootstrap. (b) Psi_t,t+1(V) = I(V_t; V_t+1) - sum_j I(X^j_t; V_t+1) (Rosas et al. 2020, Eq. 10a), with V = H-population activity and X^j = the CETT series of each H-neuron. MI uses a Gaussian copula on rank-transformed series, bivariate only, pooled over HOLO-RQA trajectories. The same is computed for the style projection, with 64 random residual coordinates as parts.",
  "null_and_controls": "Time-shuffled V for Psi; random populations of the same size as H; IAAFT; rotation control for residual fragments; label permutation for the PIP.",
  "data_needed": "Round-2 axes and text sets (Drive files via round-2 conventions); HOLO-RQA trajectories; H-set.",
  "cost_estimate": "About 15 A100-min: forward passes for 1,000 x 64 tokens (1 min) and probe sweeps (about 10 min); Psi on CPU. Runs in session S-HOLO-1.",
  "lens_free": "Linear readouts of fixed round-2 projections and of H-activity from activation fragments; no vocabulary readout.",
  "inverse_free": "Probes fitted with Adam; bivariate Gaussian-copula MI = -0.5 log(1 - r^2); no matrix inversion.",
  "pitfalls": "Psi is only a sufficient criterion: redundancy among the parts drives it negative, so Psi <= 0 is inconclusive. Gaussian-copula MI misses nonlinear dependence. Redundancy rankings depend on the readout family and on how the round-2 axes were built (contrast sets). The Darwinian reading is an analogy: the architecture supplies no system-environment split."
 }
]
```

## where_metaphor_breaks
- AdS/CFT needs a gravitating bulk with negative curvature, a conformal boundary theory and a large-N limit; a transformer has none of these. Hashimoto et al. (2018) and Swingle (2012) construct networks or geometries that represent holography; they do not show that trained LLMs are holographic.
- Holographic codes are quantum codes. The ((n, 2n-1)) threshold and complementary recovery follow from no-cloning. Classical neural codes can copy information (repetition codes), so both halves of a split can decode. The strict threshold-at-one-half pattern is therefore a falsifiable alternative, not a default expectation.
- RT equates entanglement entropy with a minimal surface area. Its classical analogue says information through a cut is bounded by the cut's capacity (Levine et al.'s min-cut for convolutional arithmetic circuits; L2M's history-state bound). That is a data-processing bound valid for any network. A transformer's KV cache grows linearly with context, so there is no fixed bottleneck cut, and 'RT holds' is trivially true unless a real bottleneck is identified.
- Monogamy (I3 <= 0) holds for holographic states. For classical linear readouts with independent noise, I3 >= 0 follows from the concavity of mutual information in SNR, so 'redundant' is the classical default; I3 <= 0 would need correlated noise or XOR-like synergy.
- The Page curve and Hayden-Preskill recovery assume unitary, information-preserving dynamics. An LLM forward pass plus sampling is many-to-one and dissipative, so there is no purity constraint and the partial-information plot need not be antisymmetric.
- Takens' theorem assumes deterministic, autonomous, finite-dimensional dynamics and a generic observation function. LLM generation is stochastic (sampling), prompt-driven, and its state (KV cache) grows with every token. Delay embeddings of surprisal therefore give descriptive geometry, not a guaranteed diffeomorphic 'bulk reconstruction'; embedding theorems that tolerate noise require multivariate observations.
- 'Holographic' in Plate's HRR means a convolution/correlation memory with approximate (inverse) decoding and a clean-up memory. Superposition in LLMs is a different mechanism (more sparse features than dimensions). Neither implies erasure thresholds.
- 'Boundary' and 'bulk' have no fixed meaning in a transformer. Here boundary = output-side observables (log-probs, entropy) or random unit fragments, and bulk = internal channels (H-activity, projections). Other assignments (tokens as boundary, depth as radial direction) give different tests.
- Entropy production estimated from token statistics is an irreversibility index of a symbol sequence. Reversed text is not a physical time reversal, and no thermodynamic dissipation is measured.
- 'Entity' via redundancy (quantum Darwinism) or causal emergence (Psi > 0) is a definitional choice. It does not demonstrate Ising-like domains, agents or collaboration, and entities defined by redundancy are the opposite of holographic threshold codes.
- Lens-free is not assumption-free. Probes, CETT, the choice of observables and the round-2 axes all carry modelling choices. The arm avoids lens confabulation, not every bias.

## refs
```json
[
 {
  "title": "H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs",
  "authors": "Cheng Gao, Huimin Chen, Chaojun Xiao, Zhiyi Chen, Zhiyuan Liu, Maosong Sun",
  "year": "2025",
  "url": "https://arxiv.org/abs/2512.01797",
  "finding": "H-neurons = positive weights of an L1 logistic probe on CETT features (neuron down-projected output norm / FFN output norm, mean over answer tokens and over other tokens); 0.01-0.35 per mille of neurons in six models (Mistral, Gemma-3, Llama-3; no Qwen). TriviaQA training with 10 samples (T 1.0, top-k 50, top-p 0.9), 10/10-consistent items, 1,000+1,000. Random equal-size neuron sets classify above chance but below H (TriviaQA 56.1-68.4% vs 70.1-83.6%). Scaling by alpha in [0,3] raises over-compliance; neurons largely inherited from pretraining; layer distribution not reported.",
  "relevance": "Defines the H-set and supplies the first point of the erasure curve; its random baselines already argue against a strict threshold (secret-sharing) code.",
  "how_checked": "Opened arXiv abs page and HTML v2; read method section and Table 1."
 },
 {
  "title": "Holographic quantum error-correcting codes: Toy models for the bulk/boundary correspondence",
  "authors": "Fernando Pastawski, Beni Yoshida, Daniel Harlow, John Preskill",
  "year": "2015",
  "url": "https://arxiv.org/abs/1503.06237",
  "finding": "Perfect tensor = isometric encoder of a [[2n-1,1,n]] code and a ((n,2n-1)) threshold scheme (any n-1 spins: no information; any n: all). Pentagon code: central operator reconstructable on any connected region with fraction > (5+sqrt5)/10 = 0.724, on some with 0.524 (continuum 1/2); erasure threshold with doubly exponentially small failure; RT exact for connected regions in their holographic states; I3 <= 0 where RT is exact.",
  "relevance": "Gives the precise, classical-translatable signature of holographic coding: threshold recovery from random subsets plus no privileged region.",
  "how_checked": "Opened abs page and full PDF (pdftotext); read sections 2, 4.4, 5.4, 5.6."
 },
 {
  "title": "Holographic Mutual Information is Monogamous",
  "authors": "Patrick Hayden, Matthew Headrick, Alexander Maloney",
  "year": "2013",
  "url": "https://arxiv.org/abs/1107.2940",
  "finding": "If entropies obey RT, I(A:BC) >= I(A:B) + I(A:C) for disjoint regions (monogamy, equivalently I3 <= 0); correlations in holographic theories are mainly entanglement.",
  "relevance": "Basis of the monogamy (I3 sign) test in HOLO-E.",
  "how_checked": "Opened arXiv abs page."
 },
 {
  "title": "Holographic Derivation of Entanglement Entropy from AdS/CFT",
  "authors": "Shinsei Ryu, Tadashi Takayanagi",
  "year": "2006",
  "url": "https://arxiv.org/abs/hep-th/0603001",
  "finding": "Entanglement entropy of a region in a (d+1)-dim CFT from the area of a d-dim minimal surface in AdS(d+2); reproduces 2D CFT results for AdS3.",
  "relevance": "RT as minimal cut; only its classical cut-capacity analogue is used (HOLO-MI growth class).",
  "how_checked": "Opened arXiv abs page."
 },
 {
  "title": "Bulk Locality and Quantum Error Correction in AdS/CFT",
  "authors": "Ahmed Almheiri, Xi Dong, Daniel Harlow",
  "year": "2015",
  "url": "https://arxiv.org/abs/1411.7041",
  "finding": "Bulk locality interpreted as operator-algebra quantum error correction (subregion-subregion duality); reach of reconstruction tied to quantum secret sharing.",
  "relevance": "Concept of multiple boundary representations (no privileged region).",
  "how_checked": "Opened arXiv abs page."
 },
 {
  "title": "Deep Learning and AdS/CFT",
  "authors": "Koji Hashimoto, Sotaro Sugishita, Akinori Tanaka, Akio Tomiya",
  "year": "2018",
  "url": "https://arxiv.org/abs/1802.08313",
  "finding": "Network layers identified with the AdS radial direction; training on boundary response data makes a bulk metric emerge; applied to AdS-Schwarzschild data and Sm0.6Sr0.4MnO3 magnetic response.",
  "relevance": "Shows the deep-net/holography link is a construction, not evidence that trained LLMs are holographic.",
  "how_checked": "Opened arXiv abs page."
 },
 {
  "title": "Entanglement Renormalization and Holography",
  "authors": "Brian Swingle",
  "year": "2012",
  "url": "https://arxiv.org/abs/0905.1317",
  "finding": "Organising a state by scale (real-space RG / entanglement renormalisation) defines a geometry; a critical state gives a discrete anti-de Sitter space.",
  "relevance": "Tensor-network (MERA) side of the analogy; marks where the metaphor stops.",
  "how_checked": "Opened arXiv abs page."
 },
 {
  "title": "Deep Learning and Quantum Entanglement: Fundamental Connections with Implications to Network Design",
  "authors": "Yoav Levine, David Yakira, Nadav Cohen, Amnon Shashua",
  "year": "2017",
  "url": "https://arxiv.org/abs/1704.01552",
  "finding": "Convolutional arithmetic circuits map to tensor networks; Claim 1: the rank of the matricization w.r.t. an input partition is at most the minimal-cut weight separating it in the TN graph; channel numbers control inductive bias; checked empirically on ReLU ConvNets.",
  "relevance": "The only rigorous RT-like min-cut statement for networks; shows the analogue is an architectural capacity bound.",
  "how_checked": "Opened abs page and PDF; read Claim 1 and section 7.2."
 },
 {
  "title": "Holographic Reduced Representations",
  "authors": "Tony A. Plate",
  "year": "1995",
  "url": "https://redwood.berkeley.edu/wp-content/uploads/2020/08/Plate-HRR-IEEE-TransNN.pdf",
  "finding": "Binding by circular convolution; decoding by correlation, called an approximate inverse; a trace stores enough to recognise but not to reconstruct accurately, so a clean-up item memory is required; distributed representations degrade gracefully.",
  "relevance": "Holographic in the distributed-memory sense; recognition-not-reconstruction is the precedent for the inverse-free generate-verify step; HRR decoding itself is avoided.",
  "how_checked": "Opened full PDF (IEEE Trans. Neural Networks 6(3):623-641) via pdftotext; read sections I-IV."
 },
 {
  "title": "Toy Models of Superposition",
  "authors": "Nelson Elhage, Tristan Hume, Catherine Olsson, et al.",
  "year": "2022",
  "url": "https://arxiv.org/abs/2209.10652",
  "finding": "Polysemanticity explained by sparse features stored in superposition; a phase change; link to adversarial examples.",
  "relevance": "Distributed alternative to localist H-neurons.",
  "how_checked": "Opened arXiv abs page."
 },
 {
  "title": "Finding Neurons in a Haystack: Case Studies with Sparse Probing",
  "authors": "Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, Dimitris Bertsimas",
  "year": "2023",
  "url": "https://arxiv.org/abs/2305.01610",
  "finding": "k-sparse probes (k swept 256 to 1) on Pythia 70M-6.9B: sparse combinations in early layers, dedicated neurons in middle layers; neurons more predictive than dimensions of a random Gaussian basis; probe loss vs k shows a power-law regime then a kink.",
  "relevance": "Template for the neuron-fragment PIP and the random-basis control in HOLO-E.",
  "how_checked": "Opened abs page and PDF; read model table and appendix on random-basis comparison."
 },
 {
  "title": "On the importance of single directions for generalization",
  "authors": "Ari S. Morcos, David G.T. Barrett, Neil C. Rabinowitz, Matthew Botvinick",
  "year": "2018",
  "url": "https://arxiv.org/abs/1803.06959",
  "finding": "Cumulative ablation curves (accuracy vs number of ablated units, random orders); clamping to the mean was more damaging than to zero in their networks; reliance on single directions predicts poor generalisation; class selectivity is a poor predictor of importance.",
  "relevance": "Design of the causal erasure curve (mean vs zero ablation both reported).",
  "how_checked": "Opened abs page and PDF; read section 2.2."
 },
 {
  "title": "Analyzing Redundancy in Pretrained Transformer Models",
  "authors": "Fahim Dalvi, Hassan Sajjad, Nadir Durrani, Yonatan Belinkov",
  "year": "2020",
  "url": "https://arxiv.org/abs/2004.04010",
  "finding": "85% of neurons redundant in BERT/XLNet; at least 92% removable for downstream tasks; 97% performance with at most 10% of neurons.",
  "relevance": "Expected redundancy baseline in transformers.",
  "how_checked": "Opened arXiv abs page."
 },
 {
  "title": "Quantum Darwinism",
  "authors": "Wojciech H. Zurek",
  "year": "2009",
  "url": "https://arxiv.org/abs/0903.5082",
  "finding": "Selected states become objective because many redundant records of them spread into environment fragments; single states are fragile, redundant copies robust.",
  "relevance": "Measurable reading of 'entity-forming' as redundant recording (HOLO-ENT).",
  "how_checked": "Opened arXiv abs page."
 },
 {
  "title": "Quantum Darwinism: Entanglement, branches, and the emergent classicality of redundantly stored quantum information",
  "authors": "Robin Blume-Kohout, Wojciech H. Zurek",
  "year": "2006",
  "url": "https://arxiv.org/abs/quant-ph/0505031",
  "finding": "Defines the partial-information plot and three profiles (independent, redundant plateau, encoded in many environments); redundancy R_delta from disjoint fragments; random states give no significant information until almost half the environment is captured; branching states have redundancy proportional to environment size.",
  "relevance": "PIP and R_delta used directly in HOLO-E and HOLO-ENT; half-threshold = strict holographic pattern.",
  "how_checked": "Opened abs page and PDF; read sections II.D-E and Figs. 2-3."
 },
 {
  "title": "Black holes as mirrors: quantum information in random subsystems",
  "authors": "Patrick Hayden, John Preskill",
  "year": "2007",
  "url": "https://arxiv.org/abs/0708.4025",
  "finding": "After the half-way point, information thrown into a rapidly mixing black hole emerges very rapidly; relies on random circuits approaching erasure-channel capacity.",
  "relevance": "Scrambling/threshold recovery as the strict pattern HOLO-E tests against.",
  "how_checked": "Opened arXiv abs page."
 },
 {
  "title": "Average Entropy of a Subsystem",
  "authors": "Don N. Page",
  "year": "1993",
  "url": "https://arxiv.org/abs/gr-qc/9305007",
  "finding": "For a random pure state the smaller subsystem has average entropy about ln m - m/(2n); on average it carries less than half a nat of information.",
  "relevance": "Random-code baseline for the half-threshold in the PIP.",
  "how_checked": "Opened arXiv abs page."
 },
 {
  "title": "Quantifying High-order Interdependencies via Multivariate Extensions of the Mutual Information",
  "authors": "Fernando E. Rosas, Pedro A.M. Mediano, Michael Gastpar, Henrik J. Jensen",
  "year": "2019",
  "url": "https://arxiv.org/abs/1902.11239",
  "finding": "O-information: Omega > 0 redundancy-dominated, Omega < 0 synergy-dominated; equals interaction information for three variables; extremes are the n-bit copy and the n-bit xor (2-n).",
  "relevance": "Sign test for monogamy/synergy over fragments (HOLO-E).",
  "how_checked": "Opened abs page and PDF; read Definition 2 and Proposition 2."
 },
 {
  "title": "Reconciling emergences: An information-theoretic approach to identify causal emergence in multivariate data",
  "authors": "Fernando E. Rosas, Pedro A.M. Mediano, Henrik J. Jensen, Anil K. Seth, Adam B. Barrett, Robin L. Carhart-Harris, Daniel Bor",
  "year": "2020",
  "url": "https://arxiv.org/abs/2004.08220",
  "finding": "Psi = I(V_t;V_t') - sum_j I(X^j_t;V_t') > 0 is sufficient for causal emergence of a supervenient V; redundancy drives Psi negative, so Psi < 0 is inconclusive; examples Game of Life, flocking, ECoG.",
  "relevance": "Causal-emergence test of the H-population and style projection (HOLO-ENT).",
  "how_checked": "Opened abs page and PDF; read section III.A, Eq. 10a, Proposition 1."
 },
 {
  "title": "Criticality in Formal Languages and Statistical Physics (journal title: Critical Behavior in Physics and Probabilistic Formal Languages)",
  "authors": "Henry W. Lin, Max Tegmark",
  "year": "2017",
  "url": "https://arxiv.org/abs/1606.06737",
  "finding": "MI between symbols decays exponentially for Markov processes, can decay as a power law for context-free grammars; enwik8, French, human genome and Bach show roughly power-law decay; a 3x256 LSTM hallucinating 1 MB of enwik8-like text reproduces a rough power law with similar slope; MI vs separation proposed as a model diagnostic.",
  "relevance": "Lens-free diagnostic of generated text; criticality reading.",
  "how_checked": "Opened abs page and PDF; read Fig. 1 caption and the LSTM section with Fig. 4."
 },
 {
  "title": "L2M: Mutual Information Scaling Law for Long-Context Language Modeling",
  "authors": "Zhuo Chen, Oriol Mayne i Comas, Zhuotao Jin, Di Luo, Marin Soljacic",
  "year": "2025",
  "url": "https://arxiv.org/abs/2503.04725",
  "finding": "Bipartite (past-future) MI grows as a power law (sub-volume) on PG19 and Wikipedia, measured with LLaMA 3.1 405B as q by a BOS-corrected direct estimator and a vCLUB estimator (shuffled second halves); both likely underestimate the exponent; Theorem 5.2: captured MI <= C dim(history state) + log M; transformers over-provision.",
  "relevance": "Estimator (forward only) and bound for HOLO-MI.",
  "how_checked": "Opened abs page and PDF; read sections 4.1-4.3, 5 and appendices B.III-B.IV. Numerical exponents only in figures, not extracted."
 },
 {
  "title": "Predictability, complexity and learning",
  "authors": "William Bialek, Ilya Nemenman, Naftali Tishby",
  "year": "2001",
  "url": "https://arxiv.org/abs/physics/0007070",
  "finding": "Predictive information (past-future MI) is finite, grows logarithmically, or as a fractional power law; its divergent part measures complexity.",
  "relevance": "Growth-class taxonomy for HOLO-MI.",
  "how_checked": "Opened arXiv abs page."
 },
 {
  "title": "CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information",
  "authors": "Pengyu Cheng, Weituo Hao, Shuyang Dai, Jiachang Liu, Zhe Gan, Lawrence Carin",
  "year": "2020",
  "url": "https://arxiv.org/abs/2006.12013",
  "finding": "Proposes an MI upper bound and a variational approximation.",
  "relevance": "Origin of the vCLUB form used in HOLO-MI (formula verified in L2M Eq. 7).",
  "how_checked": "Opened arXiv abs page only."
 },
 {
  "title": "RULER: What's the Real Context Size of Your Long-Context Language Models?",
  "authors": "Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg",
  "year": "2024",
  "url": "https://arxiv.org/abs/2404.06654",
  "finding": "17 long-context models, 13 tasks; despite claims of at least 32K, only half perform satisfactorily at 32K.",
  "relevance": "Effective context below nominal: captured-information saturation within 32k is plausible.",
  "how_checked": "Opened arXiv abs page."
 },
 {
  "title": "Detecting causality in complex ecosystems",
  "authors": "George Sugihara, Robert May, Hao Ye, Chih-hao Hsieh, Ethan Deyle, Michael Fogarty, Stephan Munch",
  "year": "2012",
  "url": "https://doi.org/10.1126/science.1227079",
  "finding": "Method based on nonlinear state-space reconstruction that distinguishes causality from correlation in nonseparable, weakly connected dynamic systems not covered by Granger causality.",
  "relevance": "CCM for HOLO-CCM.",
  "how_checked": "Read abstract from the Europe PMC record (PubMed 22997134); PubMed page itself blocked by a cookie error."
 },
 {
  "title": "Distinguishing time-delayed causal interactions using convergent cross mapping",
  "authors": "Hao Ye, Ethan R. Deyle, Luis J. Gilarranz, George Sugihara",
  "year": "2015",
  "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4592974",
  "finding": "Cross-map skill as a function of lag; optimal lag negative (or <= 0) expected for true causation, positive under synchrony from strong forcing; suited to weak to moderate coupling; general cross-mapping may fail for drivers without dynamics.",
  "relevance": "Lag rule, synchrony flag and limits for HOLO-CCM.",
  "how_checked": "Opened PMC full text."
 },
 {
  "title": "Recurrence plot based measures of complexity and their application to heart rate variability data",
  "authors": "Norbert Marwan, Niels Wessel, Udo Meyerfeldt, Alexander Schirdewan, Juergen Kurths",
  "year": "2002",
  "url": "https://arxiv.org/abs/physics/0201064",
  "finding": "Defines laminarity, trapping time (mean vertical line length, v_min = 2 for maps) and Vmax from recurrence plots built on Takens delay vectors (m = 2n+1 in theory).",
  "relevance": "Exact definitions of TT and LAM for HOLO-RQA.",
  "how_checked": "Opened PDF; read section III, Eqs. 1-4."
 },
 {
  "title": "Delay embedding in the presence of dynamical noise",
  "authors": "M.R. Muldoon, D.S. Broomhead, J.P. Huke, R. Hegger",
  "year": "1998",
  "url": "https://eprints.maths.manchester.ac.uk/34/",
  "finding": "Takens-type embedding theorem for stochastic systems x_{j+1} = phi(x_j) + eta_j; requires multivariate signals.",
  "relevance": "Why HOLO-RQA uses multivariate embeddings and treats results as descriptive under stochastic sampling.",
  "how_checked": "Opened the eprints abstract page."
 },
 {
  "title": "Improved surrogate data for nonlinearity tests",
  "authors": "Thomas Schreiber, Andreas Schmitz",
  "year": "1996",
  "url": "https://arxiv.org/abs/chao-dyn/9909041",
  "finding": "Iterative surrogates with the same autocorrelations and amplitude distribution as the data; simple amplitude adjustment yields spurious nonlinearity.",
  "relevance": "IAAFT null in HOLO-RQA, HOLO-CCM, HOLO-EP.",
  "how_checked": "Opened arXiv abs page."
 },
 {
  "title": "Estimating dissipation from single stationary trajectories",
  "authors": "Edgar Roldan, Juan M.R. Parrondo",
  "year": "2010",
  "url": "https://arxiv.org/abs/1004.2831",
  "finding": "Average entropy production per step >= lim (1/n) D_n(p_F||p_B) between forward and reversed n-string statistics of a stationary series; partial (non-invertible) observation gives a lower bound; detects non-equilibrium without measurable flows.",
  "relevance": "Estimator for HOLO-EP.",
  "how_checked": "Opened abs page and PDF; read Eqs. 1-4 and text."
 },
 {
  "title": "Quantifying hidden order out of equilibrium",
  "authors": "Stefano Martiniani, Paul M. Chaikin, Dov Levine",
  "year": "2019",
  "url": "https://arxiv.org/abs/1708.04993",
  "finding": "Length of a losslessly compressed file (LZ77-based computable information density) measures order; identifies non-equilibrium phase transitions without knowing the order parameter.",
  "relevance": "Compression-based order index in HOLO-EP.",
  "how_checked": "Opened abs page and PDF (definition of CID, LZ77)."
 },
 {
  "title": "Play. Pause. Rewind. Measuring local entropy production and extractable work in active matter (PRL title: Model-Free Measurement of Local Entropy Production and Extractable Work in Active Matter)",
  "authors": "Sunghan Ro, Buming Guo, Aaron Shih, Trung V. Phan, Robert H. Austin, Dov Levine, Paul M. Chaikin, Stefano Martiniani",
  "year": "2022",
  "url": "https://arxiv.org/abs/2105.12707",
  "finding": "Local entropy production as KL divergence between forward and time-reversed trajectories, estimated with Ziv-Merhav cross-parsing compression; depends on the tracked degrees of freedom.",
  "relevance": "Compression estimator of irreversibility in HOLO-EP.",
  "how_checked": "Opened abs page and PDF (method text)."
 },
 {
  "title": "Is my model 'mind blurting'? Interpreting the dynamics of reasoning tokens with Recurrence Quantification Analysis (RQA)",
  "authors": "Quoc Tuan Pham, Mehdi Jafari, Flora Salim",
  "year": "2026",
  "url": "https://arxiv.org/abs/2602.06266",
  "finding": "RQA (DET, LAM, ENTR; cosine distance; threshold at the 10th percentile; l_min = v_min = 3) on final-layer states of DeepSeek-R1-Distill-Qwen-7B, 3,600 ZebraLogic traces; 9-way complexity 36.9% vs 29.0% for length; no surrogate null, no length control.",
  "relevance": "Closest prior LLM application; HOLO-RQA adds surrogates and length/entropy matching.",
  "how_checked": "Opened abs page and HTML methods."
 },
 {
  "title": "Chaos in reason: How chain-of-thought LLMs can look for an answer",
  "authors": "Gregorio Jaca, Kristof Benedek, Janos Torok",
  "year": "2026",
  "url": "https://arxiv.org/abs/2607.27805",
  "finding": "Chaos signatures in LLM text and hidden-state trajectories; attention and FFN expand perturbations, normalisation and residuals counteract; Lorenz-like recurrence plots; fractal structure in last layers.",
  "relevance": "Dynamics of generation; no hallucination link stated in the abstract.",
  "how_checked": "Opened arXiv abs page only."
 },
 {
  "title": "Transformer Dynamics: A neuroscientific approach to interpretability of large language models",
  "authors": "Jesseba Fernando, Grigori Guitchounts",
  "year": "2025",
  "url": "https://arxiv.org/abs/2502.12131",
  "finding": "Residual stream treated as a dynamical system across layers; unit activations continuous across layers; unstable periodic orbits; attractor-like dynamics in lower layers.",
  "relevance": "Depth-wise (not token-wise) attractor evidence; residual basis not privileged (rotation control).",
  "how_checked": "Opened arXiv abs page only."
 },
 {
  "title": "Arrows of Time for Large Language Models",
  "authors": "Vassilis Papadopoulos, Jeremie Wenger, Clement Hongler",
  "year": "2024",
  "url": "https://arxiv.org/abs/2401.17505",
  "finding": "Small but consistent forward vs backward log-perplexity asymmetry across languages, sizes and training time; explained by sparsity and computational complexity.",
  "relevance": "Arrow of time in language is small; HOLO-EP interprets only condition differences.",
  "how_checked": "Opened arXiv abs page (ICML 2024, PMLR 235)."
 },
 {
  "title": "Fractal Patterns May Illuminate the Success of Next-Token Prediction",
  "authors": "Ibrahim Alabdulmohsin, Vinh Q. Tran, Mostafa Dehghani",
  "year": "2024",
  "url": "https://arxiv.org/abs/2402.01825",
  "finding": "Language is self-similar and long-range dependent, Hurst about 0.7.",
  "relevance": "Surrogates must preserve autocorrelation (IAAFT, not plain shuffles).",
  "how_checked": "Opened arXiv abs page."
 },
 {
  "title": "A Tale of Two Structures: Do LLMs Capture the Fractal Complexity of Language?",
  "authors": "Ibrahim Alabdulmohsin, Andreas Steiner",
  "year": "2025",
  "url": "https://arxiv.org/abs/2502.14924",
  "finding": "Fractal parameters of natural language lie in a narrow range; those of LLM outputs vary widely with architecture, temperature and prompting.",
  "relevance": "Generated text differs from natural text; temperature as design factor.",
  "how_checked": "Opened arXiv abs page."
 },
 {
  "title": "States of LLM-generated Texts and Phase Transitions between them",
  "authors": "Nikolay Mikhaylovskiy",
  "year": "2025",
  "url": "https://arxiv.org/abs/2503.06330",
  "finding": "Depending on temperature, LLM text classifiable as solid, critical or gas from autocorrelation decay.",
  "relevance": "Temperature-dependent regimes affect RQA and MI curves.",
  "how_checked": "Opened arXiv abs page only; details unverified."
 },
 {
  "title": "LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations",
  "authors": "Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, Yonatan Belinkov",
  "year": "2025",
  "url": "https://arxiv.org/abs/2410.02707",
  "finding": "Truthfulness information concentrated in exact-answer tokens; error detectors fail to generalise across datasets; models may encode the right answer yet output a wrong one.",
  "relevance": "Position-wise PIP in HOLO-E (spread over positions vs localised).",
  "how_checked": "Opened abs page and PDF (ICLR 2025 header, section 3 text)."
 },
 {
  "title": "Energy-Based Transformers are Scalable Learners and Thinkers",
  "authors": "Alexi Gladstone, Ganesh Nanduru, Md Mofijul Islam, Peixuan Han, Hyeonjeong Ha, Aman Chadha, Yilun Du, Heng Ji, Jundong Li, Tariq Iqbal",
  "year": "2025",
  "url": "https://arxiv.org/abs/2507.02092",
  "finding": "Predictions obtained by gradient descent on a learned energy (a learned verifier), framed as System 2 thinking.",
  "relevance": "Anchors the user's 'EBM reasoner' reference; motivates the energy-well/irreversibility reading (HOLO-EP, HOLO-RQA).",
  "how_checked": "Opened arXiv abs page."
 },
 {
  "title": "Qwen3-1.7B model card and config",
  "authors": "Qwen team",
  "year": "2025",
  "url": "https://huggingface.co/Qwen/Qwen3-1.7B",
  "finding": "1.7B parameters (1.4B non-embedding), 28 layers, 16 query / 8 KV heads, native context 32,768; config: hidden 2048, intermediate 6144, head_dim 128, max_position_embeddings 40960.",
  "relevance": "172,032 MLP neurons; context limit for HOLO-MI.",
  "how_checked": "Opened model card; config.json read from a local copy downloaded by a sibling agent."
 },
 {
  "title": "The Chain Lens [interim]",
  "authors": "vgel, Raymond Douglas",
  "year": "2026",
  "url": "https://www.lesswrong.com/posts/Nr6fksEqqxNpdz6rk/the-chain-lens-interim",
  "finding": "J-lens transport plus MTP-head rollouts with contrastive decoding; WorkspaceBench macro 0.420 vs 0.319 (J-lens); hallucination family 0.45 vs 0.32 (lower is better); hallucination and performance both scale with chain length (19 families); controls: neutral and foreign seeds, corpus-mean state, shuffle nulls; Neuron Lens round trip.",
  "relevance": "The lens whose confabulation motivates an independent lens-free check; source of the foreign-context control used in HOLO-MI.",
  "how_checked": "Read the local copy supplied by the workflow (scratchpad chainlens_dl/post.json); not fetched by me."
 }
]
```
