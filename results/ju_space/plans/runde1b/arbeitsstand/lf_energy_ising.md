# lf:energy_ising

## Verifier summary
All 48 references exist, and their titles, authors, years and URLs match. I checked them against arXiv API metadata for the 43 arXiv entries, the primary pages for the rest, and full texts wherever the designs rely on a number.

Most stated findings are correct. Corrections:
- Sagawa's paper is misused in O7(a). It also already states and checks a fluctuation theorem on GPT-2 samples, so the report's 'not found' on that point is wrong.
- Wording fixes: the De Nobili quote, the Simões overstatement, and the unverified bootstrap count in Lynn et al.
- Four items the report had marked unverified are now resolved: the Blondel ICML 2026 venue, the Rosas Ψ formula, the De Nobili fitting method, and Gurnee's position neurons.

Substantive design problems:
1. Path irreversibility cannot by itself make the information-theoretic forward/backward ratio non-zero for a stationary process. O7(d) is fixed to use a joint log-loss.
2. O8's macro variable is not a function of its binarised parts, which allows a spurious S_macro > 0.
3. Leading-eigenvector sign splits of a mostly positive J or C are trivial (O5, O8). Fixed with the modularity matrix.
4. O3's 95th-percentile rule cannot be computed from three control sets. Fixed with a paired prompt-level contrast.
5. O2's run-length shuffle preserves the stickiness it is meant to test. Fixed with a Markov surrogate and occupancy-matched controls.
6. O4 is prefill-bound and costs about 4.5–5.5 A100-hours, so the total rises from 4–4.5 h to about 8–10.5 h. Within that, the core H-CHAIN tests take about 1.6–2.3 h and the H-DUAL bridge about 1.3–1.5 h.
7. O7(a)'s exclusion rule contradicts its own S1/S2 comparison.

Smaller fixes:
- O1: a selection-matched control, pseudo-likelihood instead of the intractable likelihood, and an effective-sample-size check.
- O5: token count and storage.
- O6: L = 1k sampling.
- The SEA push may serve as a known answer only after a fresh manipulation check, because round-1 PC_PASS failed.

All designs are lens-free except the optional O2(D), and all are inverse-free in the design rules' sense. The corrected report is returned.

## Problems found by the verifier
- Section 3, O7(d) and metaphor item 7 list path irreversibility as a source of a non-zero forward/backward predictability ratio. For a stationary process H(X_{t+k}|X_t) = H(X_{t-k}|X_t) holds for the joint state whether or not the process is reversible (a stationary 3-cycle carries net flux yet has equal forward and backward conditional entropies). Only non-stationarity or a capacity/estimation asymmetry makes the information-theoretic ratio non-zero. Irreversibility shows up only under losses that are not the joint conditional entropy: a stationary 2-D Gaussian AR(1) with a non-normal coupling has larger backward than forward MSE but equal error log-determinants. Fixed: A_k scored by joint log-loss; irreversibility measured separately in O7(b, c).
- O8: the macro V (population magnetisation binarised at its median, or sign Δ) is not a function of the parts (eight community magnetisations binarised at their medians). Riedl defines V_t = f(X_t) and Rosas requires V to be supervenient on X. Otherwise S_macro can be positive merely because V carries information the binarised parts discard. Fixed: V computed from the parts, or the binarised units used as parts.
- O5, section 6 and O8 split communities by the sign of the leading eigenvector of J or of the correlation matrix. With mostly positive couplings or correlations that eigenvector is near-uniform (Perron-Frobenius), so the split is trivial even when two anti-aligned groups exist. Fixed: leading eigenvector of the modularity matrix, or second eigenvector after deflating the global mode (still power iteration).
- O3 and the section-4 rule require 'above the 95th percentile of the damage-matched control distribution' with only three control sets, which is not estimable (best attainable rank p = 1/4). Fixed: paired prompt-level contrast with a 90 % prompt-bootstrap interval and Holm correction, or at least 19 control sets. The stated count (6 conditions, 2,400 continuations) also does not match the listed conditions (13 cells, 2,600 continuations).
- O2: the run-length shuffle permutes whole dwell runs, so it preserves the dwell-time distribution and hence the exit hazard (gamma), and approximately HD, by construction. It cannot be the null for stickiness. Fixed: a position-matched first-order Markov surrogate. Control populations also need the identical basin pipeline and occupancy matching (HD on a logit scale depends on base rate). 'HD or stickiness' needs a multiple-testing correction.
- O4 cost is underestimated about tenfold. Stage 2 is prefill-bound: about 1e5 generations with 300-2,000-token prompts come to about 8e7 prompt tokens, about 4-5 A100-hours at the round-2 rates of 5,000-5,600 tok/s, not 20-35 min. Stage 1's count omits the memory-length factor, and shared-prefix caching saves only the rules' share of each prompt. The corrected total budget is about 8-10.5 A100-hours with O4 as specified (about 3.5-5 h without O4), not 4-4.5 h. O4 needs at least two sessions under the ~5-hour cap.
- O7(a) null (4) excludes texts not sampled from Qwen 'because sigma need not be >= 0'. This contradicts O7(a)'s own S1/S2 comparison of natural texts. It also misreads Sagawa, who applies sigma to corpora written by five other LLMs and notes only that their averages are 'not guaranteed to be non-negative'.
- Section 9 says no fluctuation theorem has been applied to LLM token generation. This is wrong: Sagawa (2026) states the integral fluctuation theorem E[exp(-sigma)] = 1 for autoregressive models and checks it numerically on GPT-2 samples (T = 5 tokens, sampling temperatures 3 and 4; Appendix B.2).
- Section 1 says 'the chain lens confabulates more as chains grow (19 WorkspaceBench families)', which merges two statements in the post. The growth of confabulation with chain length is reported for the hallucination family. The 19-family statement is that 'hallucination and performance both scale with chain length'. The post describes the contrastive score only in words; 'log p(state) - log p(prior)' is the report's reading.
- Verifier calibration uses the round-1 SEA topic push as a known answer. But round-1 PC_PASS failed at blocks 14, 18 and 6, and GO1 failed at block 6, where the sea loop's first round was not reproducible (AUSWERTUNG.md). It is usable only after a fresh single-push manipulation check.
- Simoes et al. 2026: the report says static max-ent fits 'cannot separate' critical from supercritical systems. The abstract says only that they 'may not be able to discriminate'.
- De Nobili 2026: the quoted 'bias dominates' is not verbatim. The paper says collective alignment is 'dominated by an intrinsic bias' (h_tilde >> J_tilde).
- Lynn et al.: '100 bootstrap trajectories' does not appear in the full text, which states only 100 shuffled surrogates; marked unverified.
- Section 9 items resolved by the verifier: Blondel et al. is ICML 2026 (PMLR 306:8644-8675). Rosas' Psi is eq. (10a) of the primary source. De Nobili fits J_tilde and h_tilde by logistic regression, logit P(s'=+1|k) ~ 2(h_tilde + J_tilde k). Gurnee et al. 2024 do report position neurons (GPT-2, layers 0-2).
- O1 gate G-M asks for the held-out log-likelihood of a 200-unit pairwise model, which needs the intractable log Z. Fixed: held-out pseudo-likelihood, or AIS/thermodynamic integration. The K-pairwise term needs a joint pseudo-likelihood with a shared V(K), not independent per-unit regressions. Sample adequacy depends on the token autocorrelation, which is not measured; the effective sample size is 7.5e5 divided by tau_int.
- O1 null: random populations do not control for selection, because units selected by one L1 target share a latent driver by construction. Added a selection-matched control: the same pipeline applied to an unrelated label.
- O5: the token count is about 6.5-9e6, not 5e6 (reading takes about 30-40 min). Storing 6,200 units per token in fp16 needs about 85-110 GB; fixed with an online covariance and a second pass. The rule 'S1 or S3' across four lengths needs a multiple-testing correction.
- O6: 2,048 sampled tokens per text is impossible at L = 1k, where only about 960 positions are >= 64. The permutation null for k_b is not costed.
- O7(b): per-window S_dot on 64-token windows is badly undersampled (63 transitions for 64 cells at k = 8); fixed by pooling windows. The kinetic-Ising and S_dot estimators assume first-order Markov coarse-grained dynamics, which should be checked at higher orders as Lynn et al. did.
- O2(D) applies a chain lens to another model, so it is not lens-free; it should be labelled the one non-lens-free bridge.
- O9: k-sized subsets of the H-set exist only for k <= N (at most 200). The verdict 'redundancy from a shared latent variable' has no decision rule.
- O4 protocol: Ashery's agents were prompted to 'think step by step', and the minority experiments used two names (W = 2). O4 uses answers of at most 8 tokens with ten names, so the 2-67 % range does not transfer directly.
- EBR: the statement that HRM 'can converge to both correct and incorrect fixed points' is EBR citing prior work (Ren & Liu 2026), not its own finding. That source was not opened (unverified).
- The negative novelty claims (no max-ent fit to LLM neuron activations; no agent-spin study tying coupling to identified neurons) are search results, not established facts. The verifier's searches found no counter-example.

## Per-reference / per-fact verdicts
- [verified] Energy-Based Transformers are Scalable Learners and Thinkers: arXiv 2507.02092 (2 Jul 2025), 10 authors as listed; abstract states energy minimisation 'until convergence', up to 35 % higher scaling rate and 29 % more System-2 gain on language tasks.
- [verified] Learning Iterative Reasoning through Energy Minimization: Du, Li, Tenenbaum, Mordatch; ICML 2022 (arXiv comment).
- [verified] Learning Iterative Reasoning through Energy Diffusion (IRED): Du, Mao, Tenenbaum; ICML 2024; annealed landscapes and adaptive step count confirmed in the abstract.
- [verified] Energy-Based Reasoners: Your Recurrent Reasoner Should Be an Energy-Based Model: Full text: update z_{t+1} = z_t − α∇E (eq. 2), projected Langevin variant, τ = 0.2, K parallel trajectories, 1.8M (d 256) / 7.1M (d 512) params, 99.4 % single-sample and 99.9 % with 5 samples, 345× less inference compute. The HRM 'correct and incorrect fixed points' statement is EBR citing prior work (Ren & Liu 2026, ref. 28), not EBR's own finding; that source was not opened (unverified).
- [verified] Energy-guided Recursive Model: Zhao, Tang; arXiv 2607.10128 v3; all numbers (Sudoku 98.97 %, PPBench 88.04 %, Maze 99.30 %, RedPajama-V2 perplexity −1.74 %) match the abstract.
- [verified] Think Consistently, Reason Efficiently: Energy-Based Calibration for Implicit Chain-of-Thought (EBM-CoT): 
- [verified] EBM vs. LLMs: Our Kona EBM a 96% vs. 2% Sudoku Benchmark (Logical Intelligence blog): Page exists (visible date Feb 3, 2026; Nov 2026 dates also in the page source); 96.2 % at 313 ms average and ~2 % for frontier LLMs are vendor-reported only, so the claims themselves remain unverified.
- [verified] Autoregressive Language Models are Secretly Energy-Based Models: Insights into the Lookahead Capabilities of Next-Token Prediction: ICML 2026 venue now confirmed on the PMLR page: Proceedings of the 43rd ICML, PMLR 306:8644-8675 (https://proceedings.mlr.press/v306/blondel26a.html). Remove '(unverified)'.
- [verified] Contrastive Decoding: Open-ended Text Generation as Optimization: CD-score = log(p_EXP/p_AMA) if x in V_head, −inf otherwise; 'We always set α = 0.1' (ar5iv full text).
- [verified] Hopfield Networks is All You Need: 
- [verified] Energy Transformer: NeurIPS 2023 per arXiv journal reference.
- [verified] Mapping of attention mechanisms to a generalized Potts model: Phys. Rev. Research 6, 023057 (2024); 'exactly equivalent' to pseudo-likelihood confirmed in the abstract.
- [verified] The emergence of clusters in self-attention dynamics: 
- [verified] A mathematical perspective on Transformers: Bull. Amer. Math. Soc. 62 (2025), 427-479.
- [verified] Dynamic metastability in the self-attention model: 
- [verified] Emergence of meta-stable clustering in mean-field transformer models: 
- [verified] Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth: 
- [verified] Massive Activations in Large Language Models: 
- [verified] Looped Transformers as Programmable Computers: 
- [verified] Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach: Full text: fixed points, orbits, 'sliders', path independence, limit points at r = 128, zero-shot exit when KL between steps < 5×10^-4.
- [verified] Scaling Latent Reasoning via Looped Language Models (Ouro): 
- [verified] Arrows of Time for Large Language Models: ICML 2024, PMLR 235:39509-39528.
- [corrected] Stochastic Thermodynamics for Autoregressive Generative Models: A Non-Markovian Perspective: Finding as stated is correct (PRX Intelligence 1, 023013). But: the five corpora are texts written by five LLMs, not model samples; with Qwen3-4B-Base all five unadjusted tests are significant (largest p = 0.045). Sagawa notes such averages are 'not guaranteed to be non-negative' yet still compares them, so the report's rule to exclude non-Qwen texts from O7(a) is a misreading. The paper also states and numerically checks the integral fluctuation theorem E[e^{−σ}] = 1 on GPT-2 samples (Appendix B.2), contradicting the report's 'not found'.
- [verified] Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons and Manage Hallucinations: 
- [verified] Estimating dissipation from single stationary trajectories: PRL 105, 150607 (2010) per DOI on arXiv.
- [corrected] Broken detailed balance and entropy production in the human brain: Full text (ar5iv) confirms hierarchical k-means with cosine similarity, k = 8 as the largest k with all transitions observed, coarse-graining lower bound, trajectory bootstrapping, shuffle noise floor (100 shuffled surrogates), asymmetric Ising model. '100 bootstrap trajectories' was not found in the text (unverified). PNAS DOI 10.1073/pnas.2109889118.
- [verified] Emergent social conventions and collective bias in LLM populations: Full text: N = 24, W = 10, memory H = 5, +100/−50, disorder-to-order transition 'a case of symmetry breaking', minorities 2 % (Llama-3-70B-Instruct) to 67 % (Llama-2-70b-Chat). Note: the minority experiments used W = 2 names (N = 48 for Llama-3-70B-Instruct) and agents were prompted to 'think step by step'.
- [verified] AI agents can coordinate beyond human scale: Science Advances 12, eaea6091 (2026).
- [verified] Emergent Coordination in Multi-Agent Language Models: ICLR 2026; full text: S_macro(ℓ) formula with V_t = f(X_t), main runs N = 10 GPT-4.1 agents at T = 1, 200 runs per condition, block-shuffle nulls B = 200, Jeffreys pseudo-counts.
- [verified] Reconciling emergences: An information-theoretic approach to identify causal emergence in multivariate data: Ψ^(1)(V) = I(V_t;V_t') − Σ_j I(X^j_t;V_t') (eq. 10a) read in the primary source (ar5iv); V must be supervenient on the micro state. Published PLoS Comput Biol (DOI 10.1371/journal.pcbi.1008289).
- [corrected] Collective Alignment in LLM Multi-Agent Systems: Disentangling Bias from Cooperation via Statistical Physics: Method now verified from full text: J̃, h̃ from logistic regression logit P(s'=+1|k) ≈ 2(h̃ + J̃k). The quote 'bias dominates' is not verbatim; the abstract says collective alignment is 'dominated by an intrinsic bias (h̃ ≫ J̃)'.
- [verified] Emergence of Biased Consensus in Multi-Agent LLM Debates: Accepted at ICML 2026 (arXiv comment).
- [verified] When Is Collective Intelligence a Lottery? Multi-Agent Scaling Laws for Memetic Drift in LLMs: 
- [verified] Emergence of polarization in networks of large language model agents: 
- [verified] Weak pairwise correlations imply strongly correlated network states in a neural population: 
- [verified] Searching for collective behavior in a network of real neurons: 
- [verified] Thermodynamics and signatures of criticality in a network of neurons: PMC full text: 7 authors incl. Stephanie E. Palmer (the arXiv 1407.5946 version has 6 authors and a different title); ~2 h recording, ~2×10^6 spikes, 20 ms bins; 'a clear peak in the specific heat ... for N = 100 neurons from even one-tenth of our data'; T 'is not a physical temperature'. The ref's phrase 'one tenth of about 2e6 spikes' should read 'one tenth of the data (~3.6×10^4 of ~3.6×10^5 bins, by arithmetic)'.
- [verified] Are biological systems poised at criticality?: 
- [verified] Zipf's law and criticality in multivariate data without fine-tuning: 
- [verified] Bistability, non-ergodicity, and inhibition in pairwise maximum-entropy models: PLoS Comput Biol 13(10): e1005762 (2017), authors Rostami, Porta Mana, Grün, Helias; 159 neurons; bistability 'caused by the positivity of the average correlations together with the large number N of units'.
- [corrected] Maximum entropy models of neuronal populations at and off criticality: Abstract says static ME models 'may not be able to discriminate between avalanche criticality and supercriticality'; the report's 'cannot discriminate/separate' overstates it.
- [verified] Ising models of deep neural networks: 
- [verified] Neural Networks as Spin Models: From Glass to Hidden Order Through Training: 
- [verified] H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs: Saved full text: CETT_{j,t} = ||h_t^(j)||/||h_t|| with h_t the MLP output; TriviaQA, 10 samples at temperature 1.0, top_k 50, top_p 0.9; α ∈ [0, 3]; FalseQA, FaithEval, Sycophancy, Jailbreak; 'less than 1‰' of neurons; predictive in base models.
- [verified] Hallucination Neurons and Where to Find Them: An Investigation into the existence of Hallucination Neurons: 19 of 22 with |r| > 0.7, moderate bootstrap stability, weak sparse/dense overlap, 'a sparse projection of a larger correlated structure' all confirmed.
- [verified] Holographic Reduced Representations: IEEE Trans. Neural Networks 6(3), May 1995, starts p. 623 (end page 641 not checked); 'closely related to holographic storage', 'cleaned up by using a separate associative memory', correlation as approximate inverse all confirmed in the PDF.
- [verified] Deep Learning and AdS/CFT: 
- [verified] The Chain Lens [interim]: LessWrong page fetched: 'by vgel, Raymond Douglas', 11 Nov 2026; macro 0.420 vs 0.319 over 28 families; confabulation grows with chain length on the hallucination family; 'hallucination and performance both scale with chain length, measured on 19 WorkspaceBench families'; without reinjection or contrastive decoding the head gives 'common, round numbers instead of state-specific ones'. The report's main text merges the two chain-length statements (fixed); the contrastive score is described only in words.
- [verified] Universal Neurons in GPT2 Language Models (Gurnee et al. 2024; cited in the report's section 9): arXiv 2401.12181. The paper does report position neurons ('activate as a function of absolute position', concentrated in layers 0-2 of GPT-2), so this item moves from unverified to verified.

## Corrected report
# Angle 1: a lens-free route along the energy, non-equilibrium and Ising chain (verified and corrected)

*Verifier's note (13 Nov 2026).* I opened every reference at its primary page.
- Metadata and abstracts for the 43 arXiv entries came from the arXiv API.
- I read full texts wherever the designs use a number or a condition: Energy-Based Reasoners, Geiping et al., Sagawa, Ashery et al., Riedl, Rosas et al., Lynn et al. (ar5iv), contrastive decoding (ar5iv), Tkačik et al. 2015 (PMC), Rostami et al. (PLOS), De Nobili, Gurnee et al., Plate (PDF), the chain-lens post (LessWrong page and saved copy), and the saved copies of Gao et al. and Cavus et al.
- I also opened the PMLR page and the vendor page.

All 48 references exist, with matching titles, authors, years and URLs. Errors are fixed in place:
- Section 10 lists the corrections to O1–O9.
- Section 11 gives the corrected "where the metaphor breaks" items.
- Section 12 lists the checked references.

Claims I could not check are marked (unverified).

*Scope.* The angle covers energy-based reasoners, attention as a Hopfield/Potts update, looped transformers, non-equilibrium measures, LLM agents as spins, and maximum-entropy models of neural populations. From these come lens-free and inverse-free measurements for Qwen3-1.7B:
- 28 blocks, hidden size 2,048;
- 6,144 MLP neurons per block, so 28 × 6,144 = 172,032 in total;
- 16 query heads and 8 key-value heads of dimension 128;
- maximum position 40,960.

All of these are confirmed in the downloaded config.json. Compute is one A100-40GB, with no training.

*Inputs read in full:*
- `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/brief_shared.md`
- `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/design_rules.md`
- `/home/user/DSSE/results/ju_space/runde2_ergebnisse/AUSWERTUNG.md`
- the chain-lens post (`.../scratchpad/chainlens_dl/post.json`)
- saved arXiv HTML copies of the H-neuron paper and its replication (`.../scratchpad/hchain/dl_hneurons/page.txt`, `.../scratchpad/hchain/dl_hneur_repl/page.txt`)

## 1. Short answer

Yes, but only if "holographic" is taken in a narrow sense. In this chain the word has one operational meaning: associative memory, i.e. distributed storage plus clean-up of noisy or partial cues into attractors. Plate (1995) calls convolution/correlation memories "closely related to holographic storage". He cleans their noisy reconstructions with "a separate associative memory". Hopfield and Ising models are the energy-landscape form of the same idea.

A lens-free programme can therefore test what the user hopes to learn about H-neurons without any readout into the vocabulary:
1. Is the hallucination-associated population an attractor, or does it only track length?
2. Does a context held far above natural length split one "entity" into two?
3. If it does, is the process driven, i.e. irreversible?
4. Is the H-state stored locally or in a distributed ("holographic-like") way?

What it cannot do is say what a state means in words. Where content is needed, the model's own generation supplies it inside a generate–verify loop (section 6).

**Why lens-free matters here.** The post reports two separate length effects:
- The chain lens confabulates more than single-token lenses on WorkspaceBench's hallucination family, and this grows with chain length.
- Separately, "hallucination and performance both scale with chain length, measured on 19 WorkspaceBench families".

The chain lens scores tokens by contrastive decoding against a prior state, the mean of many hidden states. In the post's words, it asks "what next token provides the most evidence for this particular hidden state relative to a generic hidden state". Read as log p(x | h) − log p(x | h_prior), this is an energy difference. The post gives no formula, so this reading is mine.

My inference, untested: suppose the state-specific part of that gap shrinks along the chain while the MTP head's language prior takes over. Then the readout produces a length effect by itself. A length dependence of chain-lens confabulation can then not be attributed to H-neurons without an independent measurement. Activation statistics plus behaviour, with position-matched controls, provide one.

**New relative to the literature, as far as searches reached.** These are negative search results, not proofs of absence. The verifier's own searches found no counter-example.
- I found no pairwise maximum-entropy (Ising) fit to the activations of transformer MLP neurons. The existing "Ising models of deep networks" map weights to couplings.
- H-neurons have only been studied as sparse probe features. The 2026 replication finds them to be "a sparse projection of a larger correlated structure", which is exactly what population models are for.
- No study of LLM agents as spins was found that ties the coupling to an identified neuron population. H-neurons are a natural candidate: Gao et al. link them causally to over-compliance, including sycophancy, a form of conformity.

## 2. What the literature supports, link by link

### 2.1 "EBM reasoner"
- **Energy-Based Transformers** (Gladstone et al. 2025)
  - An energy is assigned to (input, candidate) pairs, and prediction is gradient descent on it "until convergence".
  - The abstract claims up to 35 % higher scaling rate than Transformer++ and 29 % more gain from "System 2 thinking" on language tasks.
- **IREM and IRED** (Du et al., ICML 2022; Du, Mao, Tenenbaum, ICML 2024)
  - Reasoning is minimisation of a learned energy.
  - IRED learns annealed landscapes and adapts the number of optimisation steps to difficulty.
- **Energy-Based Reasoners** (Chen, Gladstone, Li, Ji, Du; arXiv 2611.08977, 6 Nov 2026) are probably what the user means.
  - The loop of a recurrent reasoner becomes gradient descent on a learned energy over latent states: z_{t+1} = z_t − α∇_z E_θ(z_t; x).
  - In practice it runs as a variant of projected Langevin dynamics, with injected noise and projection onto a sphere.
  - Inference stops when the tangent gradient norm falls below τ (τ = 0.2 on Sudoku-Extreme).
  - During training, K parallel descent trajectories per input shape several modes ("state-based Explorative Modeling").
  - Citing prior work (Ren & Liu 2026; not opened, unverified), they note that HRM "can converge to both correct and incorrect fixed points". This is the most explicit statement I found that looped reasoners have wrong attractors.
  - The models are tiny: 1.8 M parameters (d = 256) and 7.1 M (d = 512).
  - On Sudoku-Extreme: 99.4 % with one sample, 99.9 % with five, at 345× less inference compute than the next-best model.
- **Related systems**
  - ERM (arXiv 2607.10128) scores candidate trajectories with Hopfield-type memories and uses parallel tempering.
  - EBM-CoT (arXiv 2511.07124) refines latent chain-of-thought with an EBM and leaves the base LM unchanged.
  - Kona 1.0 (Logical Intelligence) is sold as an "energy-based reasoning model". Its Sudoku result (96.2 % at 313 ms on average, frontier LLMs about 2 %) appears only on the vendor's blog, dated Feb 3, 2026. I found no paper; the claims are unverified.
- **LM likelihood as an energy**
  - Blondel et al. (arXiv 2512.15605; ICML 2026, PMLR 306:8644–8675) give an explicit bijection between autoregressive models and EBMs in function space. It is a special case of the soft Bellman equation.
  - Contrastive decoding (Li et al., ACL 2023) scores tokens by log p_EXP − log p_AMA.
    - Only tokens in a plausibility set V_head are scored: p_EXP ≥ α · max p_EXP, with α = 0.1. All other tokens get −∞.
    - That score is an energy difference.
  - The chain lens decodes contrastively against a mean prior state, a construction of the same kind. Its exact score is not given in the post.

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
  - Pseudo-likelihood is a set of forward regressions. The design rules allow these as forward-map fits, and section 6 uses them.
- **Tokens as interacting particles** (Geshkovski, Letrouit, Polyanskiy, Rigollet; NeurIPS 2023, Bull. AMS 62 (2025))
  - With weights fixed over time, tokens cluster.
  - Cluster locations depend on the initial tokens; the kind of limit depends on the spectrum of the value matrix; "leaders" appear.
  - Before merging, tokens stay trapped near several clusters for exponentially long times, with saddle-to-saddle dynamics (Geshkovski, Koubbi, Polyanskiy, Rigollet 2024).
  - The metastable structure is selected by the inverse temperature β (Bruno et al. 2024).
- **Caveats for real models**
  - Skip connections and MLPs prevent the rank collapse of pure attention (Dong et al. 2021).
  - Massive activations concentrate attention on their tokens (Sun et al. 2024). They are about 10^5 times larger than the rest and largely input-independent.

### 2.3 Looped transformers
- A looped 13-layer transformer can emulate a small computer (Giannou et al. 2023).
- **Recurrent-depth LMs** (Geiping et al. 2025; 3.5 B parameters, 800 B tokens)
  - Latent states converge to fixed points, fall into orbits (on harder questions), or drift as "sliders".
  - The pattern depends on context, not on initialisation ("path independence").
  - Limit points are estimated at 128 iterations.
  - The zero-shot per-token exit rule is a KL between successive steps below 5×10^-4.
- **Ouro LoopLMs** (1.4 B / 2.6 B parameters, 7.7 T tokens) learn their depth allocation.

None of these depth results transfer to Qwen3-1.7B, whose blocks have distinct weights.

Round 2 measured the text loop instead:
- Gain G was 0.12–0.30 and carry K 0.45–0.48 (K at block 14 only). Both were measured along a shared style direction (the "generic attractor" at blocks 14 and 18).
- The only carry estimate on the RSI axis is K(T 1.0) = 1.053 [0.743, 1.538], which leaves self-sustainment open. It is descriptive, and no interval is reported for its denominator.

### 2.4 Non-equilibrium
- **Arrows of Time** (ICML 2024): forward and backward log-perplexity differ consistently in large models, although information theory predicts no difference.
- **Sagawa** (PRX Intelligence 1, 023013, 2026) defines entropy production for autoregressive models.
  - Definition: S = D_KL(P→‖P←), where the backward path reuses the same kernels on the reversed sequence.
  - Estimator: σ = Σ ln p_fwd − Σ ln p_bwd, i.e. two likelihood passes.
  - At token level, σ is "dominated by a syntactic artifact".
  - At sentence level (sentence order reversed), σ is larger for causally ordered texts.
    - The test corpora are texts written by five LLMs, not samples from the evaluator.
    - With GPT-2 the difference is significant for three of the five, with Qwen3-4B-Base for all five (unadjusted tests, largest p = 0.045).
    - Because such texts are not model samples, averages of σ are "not guaranteed to be non-negative". Sagawa still compares them between conditions.
  - σ splits into non-negative per-step terms: compression loss and model mismatch.
  - For samples from the model, the integral fluctuation theorem E[e^{−σ}] = 1 holds. Sagawa checks it numerically on GPT-2 samples (T = 5 tokens, sampling temperatures 3 and 4; Appendix B.2).
- **Halperin (2025)** reports that higher faithfulness goes with lower "semantic entropy production". The thermodynamics there is an analogy built on topic transition matrices.
- **Estimators for state trajectories**
  - Roldán and Parrondo (PRL 2010): the time-reversal asymmetry of one stationary series carries information on dissipation, even without visible currents.
  - Lynn et al. (PNAS 2021):
    - They compute Ṡ = Σ P_ij ln(P_ij/P_ji) on hierarchical k-means states, using cosine similarity.
    - k = 8 is the largest k at which every transition was observed at least once.
    - Coarse-graining gives a lower bound on Ṡ.
    - Errors come from trajectory bootstrapping (number of bootstrap trajectories unverified). The noise floor comes from shuffled surrogates (100 in their appendix).
    - They check the first-order Markov approximation with higher orders (1–5).
  - Their brain data nearly obey detailed balance at rest and break it under load. A kinetic Ising model with asymmetric couplings reproduces this.

### 2.5 Agents as spins, collaboration, entities
- **Ashery, Aiello, Baronchelli** (Sci. Adv. 2025)
  - Naming game: 24 agents, 10 names, memory 5, payoffs +100/−50. Agents are prompted to "think step by step".
  - The population goes through a disorder-to-order transition to one convention, which the authors call "a case of symmetry breaking".
  - Collective bias appears without individual bias.
  - Committed minorities overturn a convention at 2 % (Llama-3-70B-Instruct) up to 67 % (Llama-2-70b-Chat). These minority experiments used two names (W = 2) and N = 24, or N = 48 for Llama-3-70B-Instruct.
- **De Marzo, Castellano, Garcia** (Sci. Adv. 2026): a "majority force coefficient", and a critical group size that grows with model capability.
- **De Nobili** (2026)
  - LLM spins on a 2D lattice show temperature-driven crossovers.
  - Collective alignment is "dominated by an intrinsic bias" (h̃ ≫ J̃) rather than by neighbour coupling.
  - The result is field-driven crossovers, not phase transitions. The exponents γ/ν are "close to but incompatible with" 2D Ising.
  - J̃ and h̃ come from a logistic regression, logit P(s′ = +1 | k) ≈ 2(h̃ + J̃k), where k is the neighbour sum. This is the template for O4's response curves.
- **Okawa** (ICML 2026): theory predicts a transition to collective bias above a conformity threshold; experiments show finite-size crossovers; heterogeneity smooths them.
- **Tanaka** (2026): consensus through "memetic drift" from mutual in-context learning, with scaling laws for polarization.
- **Piao et al.**: polarization and homophilic clustering in networks of thousands of LLM agents.
- **Riedl** (ICLR 2026) tests whether a group is a "higher-order collective".
  - Criterion: S_macro(ℓ) = I(V_t;V_{t+ℓ}) − Σ_k I(X_{k,t};V_{t+ℓ}), with V_t = f(X_t) a function of the parts. This follows Rosas et al. 2020, eq. 10a, where V must be supervenient on the micro state.
  - A pairwise synergy measure complements it.
  - Main runs use N = 10 GPT-4.1 agents at T = 1.
  - Assigning personas produces "identity-linked differentiation".

**Consequence.** "Collaboration → entity" means, in this literature, symmetry breaking into one convention. Two persistent groups require:
- weak coupling,
- heterogeneity,
- committed minorities below the tipping size (Ashery: "the population settles into a mixed state"), or
- antagonistic coupling between the groups.

Field (bias) and coupling must be separated before a "transition" is claimed.

### 2.6 Maximum-entropy models of neural populations
- **Schneidman et al.** (Nature 2006)
  - Weak pairwise correlations coexist with strongly collective states in groups of ten or more neurons.
  - Pairwise (Ising) models capture these states.
  - The authors suggest the code may have "associative or error-correcting properties".
- **Tkačik et al.** (PLoS CB 2014)
  - Studied up to 120 neurons.
  - Above 40 neurons a global "K-pairwise" term is needed.
  - Activity patterns group into metastable collective modes.
  - Each neuron is "highly predictable from the rest of the population".
- **Tkačik et al.** (PNAS 2015)
  - With P_T ∝ exp(−E/T), the specific-heat peak grows and moves toward T = 1 as N grows, and S ≈ E.
  - T is "not a physical temperature".
  - The peak is already clear at N = 100 with one tenth of the data. The recording lasted about 2 h with about 2×10^6 spikes in 20 ms bins, so one tenth is about 3.6×10^4 bins by my arithmetic.
- **Mora and Bialek** (2011): models inferred from data sit near critical points.
- **Pitfalls**
  - A hidden common variable produces Zipf-like statistics without any tuning (Schwab, Nemenman, Mehta 2014).
  - Pairwise models can turn bimodal, and Glauber-based Boltzmann learning non-ergodic (Rostami et al. 2017). This happens with positive mean correlations and a couple of hundred units or more.
  - Static maximum-entropy models "may not be able to discriminate" critical from supercritical systems (Simões et al., PRE 2026).
- **Artificial networks:** Stosic et al. (2022) and Barney, Winer, Galitski (2024) map weights to couplings. In the latter, training selects a symmetry-broken "hidden order". I found no fits to activations, and neither did the verifier.

### 2.7 H-neurons
- **Gao et al. 2025**
  - Method: CETT_{j,t} = ‖h_t^(j)‖/‖h_t‖, where h_t is the MLP output, computed on answer tokens. Non-answer tokens serve as control features. Selection is by L1 logistic regression.
  - Data: TriviaQA, 10 samples per question at temperature 1.0, top-k 50, top-p 0.9.
  - Less than 1‰ (0.1 %) of neurons are selected.
  - Scaling them by α ∈ [0, 3] shifts over-compliance: false premises, misleading context, sycophancy and jailbreaks.
  - They are already predictive in the base models.
- **Cavus et al. 2026** (Gemma 3 4B, MedGemma 4B)
  - Detection and causal effects replicate.
  - But 19 of 22 selected neurons have |r| > 0.7 with other features.
  - Bootstrap selection is only moderately stable, and sparse and dense rankings overlap weakly.
  - The authors call the set "a sparse projection of a larger correlated structure".

In Qwen3-1.7B, 0.1 % of all MLP neurons is 172.

## 3. The chain as measurable quantities

| Link | Counterpart in Qwen3-1.7B | Lens-free measure | ID |
|---|---|---|---|
| EBM reasoner | −log p(text) (exact ARM–EBM bijection); fitted energy E(s) of a neuron population | E along generation; basins | O1, O7 |
| Non-equilibrium | irreversible generation; coarse-grained state dynamics | σ (text); Ṡ and cycle flux (states); kinetic-Ising asymmetry; joint-state forward/backward log-loss | O7 |
| Loop transformer | not depth; the token loop, the round-2 text loop, agent loops | hazard and dwell; push–release persistence τ_H, carry K_H | O2, O3 |
| Agent | separate Qwen instances with memory | naming game | O4 |
| Ising-modellable | pairwise/K-pairwise model of the H-population; agent response curves | J, h, basins, C(T); β_J, h_X | O1, O4 |
| Collaboration | coupling-driven symmetry breaking | consensus time, tipping fraction f_c, payoff | O4 |
| Entity-forming | two anti-aligned communities; metastable token clusters; macro self-prediction | P(Δ) bimodality and dwell; cluster persistence over blocks; S_macro | O5, O6, O8 |
| Holographic | distributed storage plus clean-up | recovery from random subsets; completion from a clamped "boundary"; graceful degradation | O9 |
| Chain-lens length effect | the model's own rollout as the "chain" (no transported state) | H-occupancy against rollout length, with position-matched controls | O2 |

**The user's "vector quantity"** asks whether a front predicts later or earlier fronts. This needs care (corrected by the verifier).

For a stationary process, one front predicts its successor exactly as well as its predecessor in information terms: H(X_{t+k}|X_t) = H(X_{t−k}|X_t). The reason is that (X_t, X_{t+k}) and (X_{t−k}, X_t) have the same joint distribution.

This holds whether or not the process is reversible. A stationary cycle 0→1→2→0 carries a net probability flux, yet its forward and backward conditional entropies are equal.

A non-zero information-theoretic ratio therefore has only two possible sources:
1. non-stationarity (drift with position);
2. a computational or estimation asymmetry between predictors of equal capacity (the Arrows-of-Time kind).

Path irreversibility (P(i→j) ≠ P(j→i)) is a separate property, which O7(b, c) measures directly. It can make forward and backward scores differ only under losses that are not the joint conditional entropy, such as summed squared error or per-unit log-losses. For example, a stationary 2-D Gaussian AR(1) with a non-normal coupling matrix has a larger backward than forward mean squared error, yet the two error covariances have equal determinants.

O7(d) therefore scores the joint state with log-loss, so that a stationary process gives exactly zero. O7(b, c) report irreversibility separately. The user's "hierarchy of channels" becomes the asymmetric part of kinetic-Ising couplings between communities (O7e).

## 4. The decisive lens-free test: H-attractor or length artefact

| Signature | Attractor predicts | Length artefact predicts | ID |
|---|---|---|---|
| History dependence at fixed position, HD = logit P(H_{t+1}\|H_t) − logit P(H_{t+1}\|¬H_t) | above all 50 control populations after a position spline (controls through the identical, occupancy-matched basin pipeline) | inside their distribution | O2 |
| Stickiness (exit hazard vs. dwell time) | hazard falls with dwell beyond a position-matched memoryless (first-order Markov) surrogate and beyond the controls | flat, as in the surrogate and controls | O2 |
| Hysteresis after a 16-token push and release | persistence (τ_H, K_H) above damage-matched random-neuron pushes in a paired prompt-level contrast, also in a cache-free re-read | relaxes like controls | O3 |
| Landscape | a distinct basin whose occupancy predicts hallucination beyond answer length | no basin beyond controls | O1 |
| Irreversibility | entry into H changes Ṡ / σ | no change | O7 |

**Proposed rules (corrected):**
- "H-attractor beyond length" needs both O2 and O3:
  - O2: HD or stickiness, each above all 50 controls. That is a rank p of 1/51 ≈ 0.02 each, at most 0.04 for the pair.
  - O3: a paired contrast against damage-matched controls whose 90 % prompt-bootstrap interval excludes 0, with Holm correction over {τ_H, K_H} × {push A, push B}.
  - The earlier "95th percentile of its controls" cannot be estimated for O3, which has only three control sets.
- O1 alone gives "collective H-code without attractor dynamics".
- Occupancy that rises with rollout length without the dynamic signatures gives "length artefact".
- Persistence that appears only in the live run, not in the re-read, gives "carried by the KV cache".

**Controls shared by all tests:**
- 50 random-neuron populations with the same layer histogram, each unit binarised at its own 95th percentile on reference text and read on the same tokens. This makes position matching automatic; the smallest empirical p is 1/51.
- A selection-matched control (added by the verifier): units chosen by the same L1 and bootstrap pipeline for an unrelated label, e.g. question topic. Units selected on one target share a latent driver by construction.
- Degenerate repetition (4-gram rate) as a competing attractor, reported separately.
- Correct and hallucinated answers matched on length.
- The round-2 generic style direction as a covariate.
  - Its vector is not in the repository: theta_verdict.json and the other round-2 result files hold only cosines, and rk_dirs.pt holds pool directions.
  - The vector must therefore come from Drive or from one B-Q0 round.
- Planted Ising data with known couplings and basins, to measure the pipeline's false-positive rates.

## 5. Bridge to H-DUAL

**Two entities.** In Ising terms these are two communities with anti-aligned magnetisation and long dwell times. A single ferromagnetic population produces at most a global ±m bistability, never two coexisting anti-aligned communities (2.5).
- O5 measures Δ = m_A − m_B in single-voice texts at 1k–16k tokens against sentence-shuffled texts of the same length. Two-voice texts are the readout's positive control. Communities come from the modularity matrix (section 6), not from the sign of J's leading eigenvector.
- O6 asks the same across depth: do token states sit in two metastable clusters for many blocks?
- O8 asks whether such a community behaves as an "entity" in the Rosas/Riedl sense, with the macro computed from its parts.

**Catfish.** "A competitor keeps the group active" becomes a sustained net cyclic flux between two macrostates, i.e. broken detailed balance (O7). Its behavioural correlate is a longer coherence lifetime of self-generated text from two-voice seeds (O5).

**Selfish mining and the "losing game".** The 1/3 threshold has no counterpart inside the model. The measurable analogue is the committed-minority tipping fraction f_c (Ashery: 2–67 % across models, in two-name games). H-DUAL then predicts three things:
1. f_c < 0.5 (a minority can impose its convention);
2. mean payoff falls while two factions coexist (the losing game);
3. longer memory ("context above natural length") makes two-faction states more persistent.

O4 tests these with memory lengths 5, 20 and 60.

**H-neurons as coupling.** If H-neurons carry over-compliance, they should set the coupling of a Qwen agent population:
- amplification should raise the conformity slope β_J, shorten consensus and lower f_c;
- suppression should favour persistent factions.

O4 is the one experiment in which the user's whole chain (loop → agent → Ising → collaboration → entity) meets H-neurons. It is also the most expensive part (section 7).

## 6. How the lens-free and inverse-free rules are applied

**Fitting.**
- Couplings are fitted by pseudo-likelihood: one logistic regression per unit on all others, i.e. a forward regression. Rende et al. show that MLM attention training solves the same problem.
  - Fits are iterative (L-BFGS or gradient descent), so no regularised inverse is formed explicitly.
  - L2-regularised regression is a forward-map fit, which the design rules allow.
- The K-pairwise term V(K) is shared by all units, so it needs a joint pseudo-likelihood: all conditionals summed, with one shared V.
- Held-out model checks use pseudo-likelihood. Exact likelihoods need log Z, which is intractable at N = 200. If log Z is wanted, it can be estimated by annealed importance sampling or thermodynamic integration, both forward MCMC methods.
- Refinement uses forward MCMC moment matching.
- Excluded: naive mean field (J = −C^{-1}), TAP and Sessak–Monasson corrections, graphical-lasso or other precision-matrix estimates, and Tikhonov-regularised inverses.

**Other forward procedures.**
- Basins: zero-temperature single-flip descent.
- Communities: power iteration on the modularity matrix of J or of the correlation matrix, or on the matrix after deflating its leading global mode. The sign of the leading eigenvector of a mostly positive J or C is near-uniform (Perron–Frobenius) and gives no split. (The modularity method is Newman's; that reference was not opened here.)
- Kinetic-Ising fits, recoverability probes and forward/backward state predictors are fitted forward maps, which the design rules allow.

**Reconstructing what a basin or community means** uses an Aletheia-style loop:
- **Generator:** clamp the units to the basin pattern in neutral prompts and sample.
- **Verifier:** re-read the output with the unmodified model. Does the population land in the same basin, above a random-basin-clamp null? Verdicts: correct, minor fix (lands in a neighbouring basin), or critically flawed.
- **Reviser:** change the clamp strength or the prompt family.
- **Stop:** abstain after 3 attempts and count abstentions.
- **Verifier error rate:** measured on known answers.
  - The usable known answers are planted basins and the two-voice texts.
  - The round-1 SEA topic push counts only after a fresh single-push manipulation check. Round-1 PC_PASS failed at blocks 14, 18 and 6. At block 6, GO1 also failed: the sea loop's first round was not reproducible (AUSWERTUNG.md).

**Lens-free means:**
- no logit, tuned, J or J++ lens;
- no MTP head;
- no projection of intermediate states to the vocabulary.

Text likelihoods come from the model's own output on its own input. The optional O2(D) applies a chain lens to another model. It is therefore not lens-free and is labelled as the one non-lens-free bridge.

## 7. Budget (corrected)

**Rates.** These come from the round-2 plan §7.1, in fp32 with TF32 off:
- One generate call of 32 prompts × 40 new tokens takes 1 s, i.e. 25 ms per decode step at batch 32. Hooks multiply this by 1.15.
- A pass of 32 × 70 tokens through all 28 blocks takes 0.40 s, about 5,600 tokens/s. With full logits it takes 0.45 s, about 5,000 tokens/s.
- Attention FLOPs at 16k context are about 1.5–1.7× (my estimate).
- I assume decode at a 2k KV window stays at 25 ms per step (unverified).

| Item | A100-min |
|---|---|
| H-set identification, if not delivered by H-CHAIN | 20–40, plus 20 bootstrap L1 refits over 2 × 172,032 features (not costed; unverified) |
| O1 + O2 shared rollouts (2,000 prompts × 2 T × 384 tokens = 1.54 M tokens; 48,000 decode steps × 25 ms × 1.15) | 23 |
| O1 fits (H-set, 50 control populations, selection-matched control, 5 planted data sets), MCMC, C(T) | 25–40 |
| O2 analysis | 5–10 |
| O3, minimal / extended grid (13 cells; paired-contrast rule) | ≈ 22 / 55 |
| O4 as specified (stage 1 ≈ 25–40; stage 2 ≈ 240–300, prefill-bound) | ≈ 265–340 |
| O4 reduced (compact memory format or 2 runs per cell) | ≈ 90–120 (estimate) |
| O5 (including O6 hooks); O6 analysis | 50–60; 10 plus permutation nulls (not costed) |
| O7 | 15 |
| O8 (CPU) | 15 |
| O9 | 20 |

**Totals.**
- With O4 as specified: about 8–10.5 A100-hours, not 4–4.5 h.
- With a reduced O4: about 5–7 h. Without O4: about 3.5–5 h.
- All of these fit the round-1b envelope of about 30 A100-hours. O4 as specified needs at least two sessions under the ~5-hour session cap.

**Breakdown by purpose.**
- Core for H-CHAIN: O1–O3, about 1.6–2.3 h with the minimal O3 grid.
- Bridge to H-DUAL: O5, O7, O8, about 1.3–1.5 h.
- Optional: O4, O6, O9.

**Memory and storage.**
- 16k-token reads need the memory-efficient SDPA kernel in fp32. The flash kernel does not take fp32.
- The KV cache for sliding-window generation at 2k × 32 sequences is about 15.0 GB: 2 × 28 layers × 8 KV heads × 128 × 4 bytes ≈ 229 KB per token.
- O5 must not store all unit activations: about 85–110 GB in fp16. Section 10 gives the online alternative.

## 8. Outcomes that count against the hypotheses

**H-CHAIN:**
- HD, stickiness (O2) and persistence (O3) all fall inside the damage- and position-matched controls. For O3 that means a paired contrast whose interval includes 0. This holds even if occupancy rises with rollout length. That is a length artefact.
- The H-population is no more collective than random populations and the selection-matched control (O1).

**H-DUAL:**
- Up to 16k tokens, single-voice texts show no more bimodality of Δ and no longer dwell times than shuffled controls of the same length (O5).
- There is no net cyclic flux above the shuffle floor (O7).
- In agent populations, f_c ≥ 0.5, or f_c and two-faction persistence do not change with memory length (O4).

**Holographic:**
- Random non-H subsets recover the H-state only at chance level, even at k = 1,000.
- Clamping 50 % of the H-units completes the pattern no better than a damage-matched random clamp does (O9).

## 9. Unverified, not found, and resolved

**Unverified:**
- Kona: vendor page only.
- Ren & Liu 2026, the source of the HRM wrong-fixed-point statement, cited through EBR and not opened.
- The number of bootstrap trajectories in Lynn et al.
- Eyal & Sirer 2014, the 1/3 threshold at γ = 0, taken from the brief and not opened here.
- Two cost assumptions: that decode stays at 25 ms per step with a 2k KV window, and the shared-prefix savings in O4 stage 1.

**Resolved by the verifier:**
- Blondel et al.: ICML 2026, PMLR 306:8644–8675 (PMLR page).
- Rosas' Ψ: eq. (10a) of Rosas et al. 2020, read in the primary source. The macro V must be supervenient on the parts.
- De Nobili's J̃ and h̃: a logistic regression, logit P(s′ = +1 | k) ≈ 2(h̃ + J̃k), confirmed in the full text.
- Gurnee et al. (arXiv 2401.12181) do report position neurons in GPT-2. These "activate as a function of absolute position" and are concentrated in layers 0–2. Position-tracking units are therefore a documented confound, still handled by design.

**Corrected:**
- "No fluctuation theorem applied to LLM token generation" was wrong. Sagawa (2026) states the integral fluctuation theorem E[e^{−σ}] = 1 for autoregressive models and verifies it numerically on GPT-2 samples (Appendix B.2).

**Not found (searches only):**
- a maximum-entropy fit to LLM neuron activations;
- an analysis of H-neurons in Qwen models. Their existence in Qwen3-1.7B is the upstream gate (G-H in O1).

## 10. Verifier corrections to the operationalisations O1–O9

All nine designs are lens-free except the optional O2(D). All are inverse-free in the design rules' sense: forward regressions, power iteration, forward sampling, and generate-verify.

**O1 ME-H**
- **Gate G-M.** Replace "held-out log-likelihood per token" with held-out pseudo-log-likelihood per token, or estimate log Z by AIS or thermodynamic integration with forward MCMC. The exact likelihood is intractable at N = 200.
- **K-pairwise term.** V(K) needs a joint pseudo-likelihood with one shared V(K), not independent per-unit regressions.
- **Sample adequacy.** Report the integrated autocorrelation time τ_int of m_H and the effective sample size 7.5×10^5/τ_int of the fit half.
  - Tkačik et al. 2015 saw a clear specific-heat peak at N = 100 from about 3.6×10^4 bins. This design matches that only if τ_int ≲ 20 tokens.
  - Above that, C(T) is uninterpretable.
- **Null.** Add the selection-matched control (section 4). Random populations alone cannot separate "collective H-code" from "any L1-selected set looks collective".
- **Cost.**
  - The 56 pseudo-likelihood fits are cheap: about 6×10^10 FLOP per gradient for 7.5×10^5 × 200 binary data, so minutes in total.
  - The 20 bootstrap L1 refits for the H-set are not costed. At R = 2×10^4 responses, each refit runs over R × 344,064 features (6.9×10^9 values, about 14 GB in fp16). Unverified; possibly tens of minutes on CPU.

**O2 H-DYN**
- **Surrogate.** The run-length shuffle permutes whole dwell runs. It therefore preserves the dwell-time distribution, hence the exit hazard γ and, approximately, HD, by construction. It is a null only for position structure.
  - For stickiness, use a position-matched first-order Markov surrogate: per-position-bin transition probabilities with memoryless dwell. Test γ < 0 against it.
  - For HD, the control populations are the null.
- **Controls.** Each control population goes through the identical pipeline: its own O1 fit and the basin with the largest hallucination log-odds on the fit half, or the same occupancy-matched threshold. Control b_t must be matched to the H-basin's occupancy, because HD on a logit scale depends on base rate.
- **Multiple testing.** "HD or γ": each must exceed all 50 controls (rank p = 1/51 each).
- **(D)** is not lens-free and is labelled as such.

**O3 H-HYST**
- **Rule.** Three damage-matched control sets cannot support a 95th percentile; the best attainable rank p is 1/4. The corrected rule reads as follows.
  - Per prompt, take τ_H (or K_H) after the H push minus the mean over the three control sets, each read on the H-units.
  - "Self-sustaining" holds if the 90 % prompt-bootstrap interval (2,000 replicates) lies above 0, with Holm correction over {τ_H, K_H} × {push A, push B}.
  - The excess must also survive the cache-free re-read.
  - Alternative: at least 19 control sets with H above all of them (p = 1/20), at about +80 A100-min (estimate).
- **Count.** The listed conditions give 13 cells:
  - H scaling, three random sets and the different-basin push, each at α ∈ {0, 2.5}: 10 cells;
  - push B, the random-direction push and the sham: 3 cells.
  - That is 13 cells × 200 prompts = 2,600 continuations, not 2,400. The cost is about 22 min.

**O4 H-J**
- **Cost (main correction).** Stage 2 is prefill-bound. Assume about 31 tokens per memory entry, implied by 300 tokens at M = 5 and 2,000 at M = 60.
  - The 27 population runs start with empty memories that fill over each agent's ≈ 83 interactions. Mean prompts are ≈ 300 / 690 / 1,320 tokens at M = 5 / 20 / 60, for ≈ 4.1×10^7 prompt tokens.
  - The 24 minority runs start with full memories and need ≈ 1.5 model calls per interaction, for ≈ 4.1×10^7 prompt tokens.
  - Total ≈ 8.3×10^7 prompt tokens, i.e. ≈ 4.1–4.6 A100-hours at 5,600–5,000 tokens/s, or up to ≈ 5.3 h with hook overhead. Not 20–35 min.
  - O4 needs two sessions, or a reduced design: a compact memory format of about 10 tokens per entry (about 3× fewer tokens), or 2 runs per cell.
- **Stage 1.**
  - The count (6 × 10 × 20 × 12 = 14,400) has no memory-length factor, while prompts of 300–2,000 tokens imply M ∈ {5, 20, 60}. If each response curve is measured at all three M, the prompt count triples (about 1.8–2 h).
  - Shared-prefix caching saves only the rules' share of each prompt (likely at most 30 %; unverified), giving about 25–40 min, not 15.
  - With single-token names, the 3 T conditions need no extra forward passes: P(choose X) at any T follows from the same logits.
- **Protocol.** Ashery's agents reasoned step by step, and their minority experiments used two names. O4's answers of at most 8 tokens with ten names change the protocol. The 2–67 % range therefore does not transfer; f_c must be read against O4's own controls.

**O5 DUAL-M2**
- **Communities.** Use the leading eigenvector of the modularity matrix (C minus its rank-one null expectation), or the second eigenvector after deflating the global mode, by power iteration on the fit half.
  - For the 2,000-unit populations use the correlation-based matrix, not a fitted J, which would have about 2×10^6 parameters.
- **Data.** 60 texts × (1k + 4k + 8k + 16k) ≈ 1.8×10^6 tokens per text set.
  - S1, S2, S3 and the two S4 sets give ≈ 8.9×10^6 tokens. With a prefix design it is ≈ 6.5×10^6: the shorter S1–S3 texts are prefixes of the 16k texts, which a causal model reads in one pass. Either way it is more than 5×10^6.
  - Reading takes ≈ 30–40 min at a blended ≈ 3,900 tokens/s, so O5 totals ≈ 50–60 min.
- **Storage.** H-units plus 6,000 units plus 2 PCs per token in fp16 come to ≈ 85–110 GB.
  - Accumulate the fit-half covariance online instead: 6,000 × 6,000 in fp32 ≈ 144 MB.
  - Compute Δ online in a second pass.
- **Multiple testing.** "S1 or S3" across four lengths needs Holm correction, frozen in the plan.

**O6 DEPTH-CLU**
- At L = 1k only ≈ 960 positions ≥ 64 exist, so sample min(2,048, L − 64) tokens per text.
- The column-permutation null for k_b is not costed. A 99th percentile needs ≥ 100 permutations per block and text.

**O7 ARROW**
- **(a)** Delete null (4).
  - It contradicts (a)'s own S1/S2 comparison and misreads Sagawa, who applies σ to corpora written by other LLMs and only notes that their averages are not guaranteed to be ≥ 0.
  - Keep all texts. Read σ as an entropy production (non-negative mean, E[e^{−σ}] = 1) only for Qwen's own samples (O2 rollouts, S3).
- **(b)** Per-window Ṡ on 64-token windows is undersampled: 63 transitions for 64 cells at k = 8.
  - Pool transitions over all position-matched windows of a condition.
  - Apply the largest-k-with-all-transitions-observed rule to the pooled counts.
  - Check the first-order Markov approximation at higher orders, as Lynn et al. did.
- **(d)** Predict the joint coarse-grained state, i.e. the k-means labels of (b) or the joint binarised pattern, with log-loss.
  - Under stationarity A_k = 0 exactly for Bayes-optimal predictors.
  - Split non-zero A_k into drift (position-binned refits) and capacity or estimation asymmetry (two predictor families of equal capacity).
  - Do not attribute A_k to path irreversibility; report irreversibility from (b) and (c). Ridge/MSE or per-unit losses can be asymmetric for a stationary irreversible process.
- **(e)** The kinetic Ising fit is a first-order Markov projection, and the sampled token itself drives the next state.

**O8 EMERGE**
- **Supervenience.** V must be a function of the parts as defined (Riedl: V_t = f(X_t); Rosas: V supervenient on X).
  - The population magnetisation binarised at its median is not a function of the eight binarised community magnetisations. S_macro can then turn positive without any emergence.
  - Use V_t = 1[Σ_k X_{k,t} ≥ 4], the majority of parts.
  - Or use V_t = 1[Σ_{k∈A} X_{k,t} > Σ_{k∈B} X_{k,t}], with parts assigned to the O5 communities.
  - Or use the individual binarised units as parts. Redundancy then drives S_macro strongly negative, so it remains a sufficient criterion only.
- **Partition.** As in O5: modularity matrix, not the sign of the leading eigenvector of J.

**O9 HOLO**
- k-sized subsets of the H-set exist only for k ≤ N (at most 200). At k = 300 and 1,000, compare with the full H-set.
- The verdict "redundancy from a shared latent variable" needs a rule. Proposal (verifier's, to be frozen): decide it if a probe on one candidate latent covariate reaches ≥ 90 % of the random-subset AUROC. Candidates are next-token entropy at answer tokens and the first PC of the units.

## 11. Corrected "where the metaphor breaks" items (others unchanged)

- **Attention = Hopfield update.** This holds for one update with Ramsauer's energy. With multi-head projections, causal masks, RoPE, RMSNorm, MLPs and residual sums, I know of no energy that decreases monotonically through a standard LM (unverified negative). The Energy Transformer has to be built for that.
- **Temperature.** The sampling temperature and the T of a fitted max-ent model are parameters, not physical temperatures (Tkačik et al. 2015). Static fits may not be able to discriminate critical from supercritical systems (Simões 2026).
- **Non-equilibrium.**
  - There is no heat bath and no energy flow. Sagawa's σ is information-theoretic: it mixes compression loss and model mismatch, and at token level it is dominated by syntax.
  - σ has a non-negative mean, and satisfies E[e^{−σ}] = 1, only for samples from the evaluating model.
  - Coarse-grained Ṡ is only a lower bound.
  - Generation is non-stationary, so "equilibrium" exists only for coarse-grained, windowed statistics.
- **Forward vs backward front predictability** (the user's vector quantity).
  - For a stationary process, a front predicts its successor and its predecessor equally well in information terms, reversible or not.
  - A non-zero ratio needs non-stationarity or a computational or estimation asymmetry between predictors of equal capacity (the Arrows-of-Time kind).
  - Path irreversibility is a separate quantity. It shows up as net flux (O7b, c) and changes forward/backward scores only under non-log or per-unit losses.
- **Collaboration → entity-forming.** Interacting LLM populations break symmetry into one convention (Ashery 2025). Two persistent groups need weak coupling, heterogeneity, committed minorities or antagonistic inter-group coupling. Observed "transitions" are often field-driven crossovers in which collective alignment is "dominated by an intrinsic bias" (De Nobili 2026), or finite-size crossovers (Okawa 2026).
- **Catfish / selfish mining.**
  - The 1/3 selfish-mining threshold (at γ = 0; Eyal & Sirer 2014, taken from the brief) comes from rational payoff maximisation.
  - LLM populations have no universal tipping threshold: committed minorities of 2–67 % flip conventions in two-name games, depending on the model.
  - "The optimum lies below 50 %" has no counterpart in a single model's activations. The nearest measurable analogues are f_c < 0.5 and a lower mean payoff while factions coexist.

## 12. References (checked)

1. Gladstone, Nanduru, Islam, Han, Ha, Chadha, Du, Ji, Li, Iqbal (2025). Energy-Based Transformers are Scalable Learners and Thinkers. https://arxiv.org/abs/2507.02092 — verified.
2. Du, Li, Tenenbaum, Mordatch (2022). Learning Iterative Reasoning through Energy Minimization. ICML 2022. https://arxiv.org/abs/2206.15448 — verified.
3. Du, Mao, Tenenbaum (2024). Learning Iterative Reasoning through Energy Diffusion. ICML 2024. https://arxiv.org/abs/2406.11179 — verified.
4. Chen, Gladstone, Li, Ji, Du (2026). Energy-Based Reasoners: Your Recurrent Reasoner Should Be an Energy-Based Model. https://arxiv.org/abs/2611.08977 — verified (full text).
5. Zhao, Tang (2026). Energy-guided Recursive Model. https://arxiv.org/abs/2607.10128 — verified.
6. Chen, Cui, Ye, Zhang, Bian, Zhu (2025). Think Consistently, Reason Efficiently: Energy-Based Calibration for Implicit Chain-of-Thought. https://arxiv.org/abs/2511.07124 — verified.
7. Logical Intelligence (2026). EBM vs. LLMs: Our Kona EBM a 96% vs. 2% Sudoku Benchmark. https://logicalintelligence.com/blog/energy-based-model-sudoku-demo — page verified; claims vendor-reported (unverified).
8. Blondel, Sander, Vivier-Ardisson, Liu, Roulet (2025; ICML 2026, PMLR 306:8644–8675). Autoregressive Language Models are Secretly Energy-Based Models: Insights into the Lookahead Capabilities of Next-Token Prediction. https://arxiv.org/abs/2512.15605 — verified.
9. Li, Holtzman, Fried, Liang, Eisner, Hashimoto, Zettlemoyer, Lewis (2023). Contrastive Decoding: Open-ended Text Generation as Optimization. ACL 2023. https://arxiv.org/abs/2210.15097 — verified.
10. Ramsauer et al. (16 authors; 2020). Hopfield Networks is All You Need. https://arxiv.org/abs/2008.02217 — verified.
11. Hoover, Liang, Pham, Panda, Strobelt, Chau, Zaki, Krotov (2023). Energy Transformer. NeurIPS 2023. https://arxiv.org/abs/2302.07253 — verified.
12. Rende, Gerace, Laio, Goldt (2024). Mapping of attention mechanisms to a generalized Potts model. Phys. Rev. Research 6, 023057. https://arxiv.org/abs/2304.07235 — verified.
13. Geshkovski, Letrouit, Polyanskiy, Rigollet (2023). The emergence of clusters in self-attention dynamics. NeurIPS 2023. https://arxiv.org/abs/2305.05465 — verified.
14. Geshkovski, Letrouit, Polyanskiy, Rigollet (2023; Bull. AMS 62 (2025) 427–479). A mathematical perspective on Transformers. https://arxiv.org/abs/2312.10794 — verified.
15. Geshkovski, Koubbi, Polyanskiy, Rigollet (2024). Dynamic metastability in the self-attention model. https://arxiv.org/abs/2410.06833 — verified.
16. Bruno, Pasqualotto, Agazzi (2024). Emergence of meta-stable clustering in mean-field transformer models. https://arxiv.org/abs/2410.23228 — verified.
17. Dong, Cordonnier, Loukas (2021). Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth. https://arxiv.org/abs/2103.03404 — verified.
18. Sun, Chen, Kolter, Liu (2024). Massive Activations in Large Language Models. COLM 2024. https://arxiv.org/abs/2402.17762 — verified.
19. Giannou, Rajput, Sohn, Lee, Lee, Papailiopoulos (2023). Looped Transformers as Programmable Computers. https://arxiv.org/abs/2301.13196 — verified.
20. Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein (2025). Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach. https://arxiv.org/abs/2502.05171 — verified (full text).
21. Zhu et al. (33 authors; 2025). Scaling Latent Reasoning via Looped Language Models. https://arxiv.org/abs/2510.25741 — verified.
22. Papadopoulos, Wenger, Hongler (2024). Arrows of Time for Large Language Models. ICML 2024. https://arxiv.org/abs/2401.17505 — verified.
23. Sagawa (2026). Stochastic Thermodynamics for Autoregressive Generative Models: A Non-Markovian Perspective. PRX Intelligence 1, 023013. https://arxiv.org/abs/2604.07867 — verified (full text); its use is corrected in 2.4 and O7.
24. Halperin (2025). Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons and Manage Hallucinations. https://arxiv.org/abs/2512.05156 — verified.
25. Roldán, Parrondo (2010). Estimating dissipation from single stationary trajectories. PRL 105, 150607. https://arxiv.org/abs/1004.2831 — verified.
26. Lynn, Cornblath, Papadopoulos, Bertolero, Bassett (2021). Broken detailed balance and entropy production in the human brain. PNAS (DOI 10.1073/pnas.2109889118). https://arxiv.org/abs/2005.02526 — verified; bootstrap count unverified.
27. Flint Ashery, Aiello, Baronchelli (2025). Emergent social conventions and collective bias in LLM populations. Sci. Adv. 11, eadu9368. https://arxiv.org/abs/2410.08948 — verified (full text).
28. De Marzo, Castellano, Garcia (2024; Sci. Adv. 12, eaea6091, 2026). AI agents can coordinate beyond human scale. https://arxiv.org/abs/2409.02822 — verified.
29. Riedl (2025; ICLR 2026). Emergent Coordination in Multi-Agent Language Models. https://arxiv.org/abs/2510.05174 — verified (full text).
30. Rosas, Mediano, Jensen, Seth, Barrett, Carhart-Harris, Bor (2020). Reconciling emergences: An information-theoretic approach to identify causal emergence in multivariate data. PLoS Comput Biol (DOI 10.1371/journal.pcbi.1008289). https://arxiv.org/abs/2004.08220 — verified (eq. 10a).
31. De Nobili (2026). Collective Alignment in LLM Multi-Agent Systems: Disentangling Bias from Cooperation via Statistical Physics. https://arxiv.org/abs/2605.10528 — verified (full text).
32. Okawa (2026; ICML 2026). Emergence of Biased Consensus in Multi-Agent LLM Debates. https://arxiv.org/abs/2608.02827 — verified.
33. Tanaka (2026). When Is Collective Intelligence a Lottery? Multi-Agent Scaling Laws for Memetic Drift in LLMs. https://arxiv.org/abs/2603.24676 — verified.
34. Piao, Lu, Gao, Xu, Hu, Santos, Li, Evans (2025). Emergence of polarization in networks of large language model agents. https://arxiv.org/abs/2501.05171 — verified.
35. Schneidman, Berry, Segev, Bialek (2006). Weak pairwise correlations imply strongly correlated network states in a neural population. Nature (DOI 10.1038/nature04701). https://arxiv.org/abs/q-bio/0512013 — verified.
36. Tkačik, Marre, Amodei, Schneidman, Bialek, Berry (2014). Searching for collective behavior in a network of real neurons. PLoS Comput Biol 10, e1003408. https://arxiv.org/abs/1306.3061 — verified.
37. Tkačik, Mora, Marre, Amodei, Palmer, Berry, Bialek (2015). Thermodynamics and signatures of criticality in a network of neurons. PNAS 112:11508–11513. https://pmc.ncbi.nlm.nih.gov/articles/PMC4577210 — verified (full text).
38. Mora, Bialek (2011). Are biological systems poised at criticality? J Stat Phys 144:268–302. https://arxiv.org/abs/1012.2242 — verified.
39. Schwab, Nemenman, Mehta (2014). Zipf's law and criticality in multivariate data without fine-tuning. PRL 113, 068102. https://arxiv.org/abs/1310.0448 — verified.
40. Rostami, Porta Mana, Grün, Helias (2017). Bistability, non-ergodicity, and inhibition in pairwise maximum-entropy models. PLoS Comput Biol 13(10): e1005762. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005762 — verified.
41. Simões, Lombardi, Plenz, Herrmann, de Arcangelis (2026). Maximum entropy models of neuronal populations at and off criticality. PRE 114, 054103. https://arxiv.org/abs/2511.14872 — verified; wording softened.
42. Stosic, Stosic, Stosic (2022). Ising models of deep neural networks. https://arxiv.org/abs/2209.08678 — verified.
43. Barney, Winer, Galitski (2024). Neural Networks as Spin Models: From Glass to Hidden Order Through Training. https://arxiv.org/abs/2408.06421 — verified.
44. Gao, Chen, Xiao, Chen, Liu, Sun (2025). H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs. https://arxiv.org/abs/2512.01797 — verified (saved full text).
45. Cavus, Sabu, Spear, Kawatra, Rajendran (2026). Hallucination Neurons and Where to Find Them: An Investigation into the existence of Hallucination Neurons. https://arxiv.org/abs/2609.29781 — verified.
46. Plate (1995). Holographic Reduced Representations. IEEE Trans. Neural Networks 6(3), from p. 623 (end page not checked). https://redwood.berkeley.edu/wp-content/uploads/2020/08/Plate-HRR-IEEE-TransNN.pdf — verified.
47. Hashimoto, Sugishita, Tanaka, Tomiya (2018). Deep Learning and AdS/CFT. PRD 98, 046019. https://arxiv.org/abs/1802.08313 — verified.
48. vgel, Raymond Douglas (11 Nov 2026). The Chain Lens [interim]. LessWrong. https://www.lesswrong.com/posts/Nr6fksEqqxNpdz6rk/the-chain-lens-interim — verified (page fetched).
49. Added: Gurnee, Horsley, Guo, Kheirkhah, Sun, Hathaway, Nanda, Bertsimas (2024). Universal Neurons in GPT2 Language Models. https://arxiv.org/abs/2401.12181 — verified (reports position neurons).


## operationalisations (from the research stage; check against the verifier's notes above)
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

## where_metaphor_breaks (from the research stage; check against the verifier's notes above)
```json
[
 "EBM reasoner vs Qwen: Qwen3-1.7B minimises no energy at inference. EBTs, IRED and EBRs run gradient or Langevin descent on a learned energy; Qwen computes one fixed feed-forward pass per token. -log p(text) is a valid energy (Blondel et al.'s ARM-EBM bijection), but sampling draws from it rather than descending it. Only outer loops iterate: token steps, the round-2 text loop and agent loops.",
 "Loop transformer vs Qwen depth: Qwen's 28 blocks have distinct weights, so depth is not a time-homogeneous dynamical system. Fixed points, orbits and sliders (Geiping), energy stationarity (EBR) and clustering theorems (Geshkovski) assume one repeated map. Attractor claims must be made on generation steps or outer loops.",
 "Attention = Hopfield update: this holds for one update with Ramsauer's energy. With multi-head projections, causal masks, RoPE, RMSNorm, MLPs and residual sums, no known energy decreases monotonically through a standard LM (the Energy Transformer has to be built for that). Metastable head patterns are not residual-stream attractors.",
 "Ising-modellable: a pairwise maximum-entropy fit is a statistical description. Couplings are effective: they absorb shared inputs (position, topic, prompt, uncertainty) and are not wiring. Minima of the fitted energy are not dynamical attractors of the network, which does not run Glauber dynamics on it. Positive correlations can make pairwise models spuriously bimodal (Rostami 2017), and latent variables produce Zipf and criticality signatures without tuning (Schwab 2014).",
 "Temperature: the sampling temperature and the T of a fitted max-ent model are parameters, not physical temperatures (Tkacik et al. 2015 say so). Static fits cannot distinguish critical from supercritical (Simoes 2026).",
 "Non-equilibrium: there is no heat bath and no energy flow. The entropy production of generated text (Sagawa) is information-theoretic: it mixes compression loss and model mismatch because the backward process reuses forward kernels, and at token level it is dominated by syntax. Coarse-grained S_dot is only a lower bound. Generation grows a sequence and is non-stationary, so 'equilibrium' exists only for coarse-grained, windowed statistics.",
 "Forward vs backward front predictability (the user's vector quantity): for a stationary process, a front predicts its successor and its predecessor equally well in information terms, since H(X_{t+k}|X_t) = H(X_{t-k}|X_t). A non-zero ratio needs non-stationarity, path irreversibility (P(i->j) != P(j->i)), or a computational asymmetry between predictors of equal capacity (the Arrows-of-Time kind). These are three different things and must be measured separately.",
 "Agent -> spin: in LLM-population studies the spins are separate model instances with memory and payoffs, interacting over rounds. Neurons, heads or positions inside one forward pass have no goals, memory or payoffs. The agent-spin mapping holds for populations of instances (O4) and is only an analogy for units inside one model.",
 "Collaboration -> entity-forming: interacting LLM populations break symmetry into one convention (Ashery 2025). Two persistent groups need weak coupling, heterogeneity, committed minorities or antagonistic inter-group coupling; a ferromagnetic population does not split by itself. Observed 'transitions' in LLM collectives are often field-driven crossovers where bias dominates coupling (De Nobili 2026), or finite-size crossovers (Okawa 2026).",
 "Catfish / selfish mining: the 1/3 selfish-mining threshold (at gamma 0) comes from rational payoff maximisation. LLM populations have no universal tipping threshold: committed minorities of 2-67 % flip conventions depending on the model. 'The optimum lies below 50 %' has no counterpart in a single model's activations. The nearest measurable analogues are a tipping fraction f_c < 0.5 and a lower mean payoff while factions coexist.",
 "Hyper-long context = catfishing: very long contexts change the state through position encoding and attention dilution. These slow drifts can mimic 'new entities', so a split counts as a force arising from context only if it beats content-shuffled controls of the same length.",
 "H-neurons as an attractor: H-neurons are defined by a sparse linear probe whose selection is non-unique within correlated clusters (Cavus 2026). A population model over a probe-defined set inherits the probe's biases, and 'attractor' needs the dynamic and interventional signatures (O2, O3), not a static energy landscape.",
 "Holography: a transformer has no bulk-boundary duality and no emergent extra dimension. Hashimoto et al. read depth as the radial direction only for a network trained to fit boundary data. The operational content available lens-free is the associative-memory sense: distributed storage, completion from partial cues, graceful degradation. Even Plate's holographic decoding uses correlation as an approximate inverse plus a clean-up memory, i.e. a generate-then-clean scheme, not an exact inversion.",
 "Lens-free does not remove confabulation from interpretation: lens-free measures give structure and dynamics, not content. Naming what a basin or community 'is' needs the model's own generation, which can itself confabulate, so the round-trip verifier must be calibrated on known answers (planted basins, the round-1 SEA push, two-voice texts).",
 "Contrastive decoding as an energy difference: the chain lens scores log p(state) - log p(prior), an energy gap. My untested inference: its growth of confabulation with chain length may come from that gap shrinking relative to the MTP head's language prior, a property of the readout. If so, lens length effects and H-neuron effects stay confounded until a lens-free measure separates them."
]
```

## refs (research stage)
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
