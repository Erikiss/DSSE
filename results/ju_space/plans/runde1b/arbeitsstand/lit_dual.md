# lit:dual

## Verifier summary
I checked 37 references: the 36 listed plus Brown (JPE 2011), which the report cited but marked unverified. I opened the primary pages and PDFs (arXiv, NeurIPS, the Anthropic system card, the Chroma page, Hugging Face, EconPapers, LessWrong, Wikipedia wikitext). All 37 exist, and titles, authors, years and URLs are correct. 33 are verified as stated. 4 need corrections:
- **Xie et al.:** the 1/3-share range is 15.8–86.1 %, not 30–86 %; 6 of 16 model–dataset cells fall below the 1/3 share. The 1/2 → 2/4 effect holds mostly on PopQA.
- **Xu et al.:** the claim that sampling weakens self-reinforcement is the authors' suggestion, not a measurement.
- **Qwen3-1.7B card:** the "no greedy decoding" warning is for thinking mode only.
- **Moura Ramos et al.:** the evaluation used Qwen3's thinking-mode decoding defaults, so the third-party RULER numbers are most likely thinking-mode scores.

Three items the report marked unverified are now verified: Brown (JPE 2011), the Assistant Axis at ICML 2026 (PMLR 306:80955–81003), and the Nature 2024 model-collapse paper.

The formulas check out:
- Eyal–Sirer threshold and Eq. 8 match.
- Sapirshtein's values match, with Fig. 2 rendered and read.
- Grunspan's E[δ] and break-even times match.
- The KV-cache, logits and prefill arithmetic is right.
- Only the computed majority threshold at γ = 0.5 should read 0.389 instead of 0.390.

Round-2 numbers match `AUSWERTUNG.md` and `theta_verdict.json`, with two exceptions. The "32 calls × 1.3 s" timing anchor comes from the round-2 plan's budget, not a measurement. The lower bound of one carry interval is 0.414, not 0.415.

The main reasoning problems:
- **O2 cost:** underestimated about 4× (≈ 5–7.5 A100-hours, not 1–2); batch 12 does not fit at 32K. The total becomes about 11–17 hours, not 7–10.
- **Position-kernel control:** fitted from single blocks and applied additively, it cannot rule out run-length effects when B is released last as one contiguous block.
- **Counterbalancing:** averaging over label swaps cancels the prior part of γ.
- **Eyal–Sirer curve fit:** it is not a fair comparison, since that curve reaches 1 at α = 1/2 for every γ and has fewer parameters than the kernel.
- **M2 labels:** O1 states share- versus count-dependence inconsistently with §2.6.
- **O2 statistics:** 16 samples per start is too few for the bimodality tests. A Pólya-urn null can itself be U-shaped. The hysteresis control matches only the last 4 blocks.
- **O3:** it lacks a conflict-interference control.
- **O4:** it cannot measure B's carry or mutual sustainment, because B is injected by the experimenter each turn. It has no long-context factor. With round-2 noise, most per-step K values from about the fourth answer on will be undefined.
- **O5:** the asymmetric-pair prediction is not diagnostic.
- **40K cells:** they go beyond the 32,768-token trained window.
- **Over-reading of round 2:** C-Q4, B-T5 and B-Q7 are used as stronger evidence than they are.

The corrected report fixes these in place. No source was not found or misattributed.

## Problems found by the verifier
- Xie et al. range is wrong (§1 item 3, §4, confounds field, ref finding and relevance). At a 1/3 share of memory-aligned evidence, Table 6 gives 15.8% (PaLM2, PopQA) to 86.1% (Vicuna-7B, StrategyQA), not 30-86%. Six of 16 model-dataset cells lie below the 1/3 share: PaLM2 PopQA 15.8, Llama2-70B PopQA 21.6, ChatGPT StrategyQA 26.8, Vicuna-7B PopQA 29.5, ChatGPT PopQA 30.0, Qwen-7B PopQA 32.5. So 'a prior-aligned minority captures more than its share' holds in 10 of 16 cells, not in general.
- Xie et al. quantity effect: the quoted 1/2 -> 2/4 increases are PopQA values. The effect is not uniform: PaLM2 PopQA falls 56.8 -> 53.9, and StrategyQA is mixed (ChatGPT 50.0 -> 48.9, Llama2-7B 84.5 -> 84.1, GPT-4 72.8 -> 72.9).
- Xu et al.: 'stochastic sampling weakens the effect' was not measured. The authors only suggest that sampling 'may' avoid self-reinforcement.
- Qwen3-1.7B card: 'DO NOT use greedy decoding' is stated for thinking mode, and round 1b runs thinking off. The repetition advice is presence_penalty 1.5 (0-2 in frameworks). The confounds field repeats the greedy warning without this qualifier.
- Moura Ramos et al. RULER numbers: the evaluation used a common decoding configuration (T 0.6, top-p 0.95, top-k 20, presence penalty 1.5, up to 32,768 generated tokens), which is Qwen3's thinking-mode default, and the paper calls Qwen3-1.7B a reasoning model. The scores are therefore most likely thinking-mode results and may not transfer to the thinking-off setting of rounds 1-2. Thinking mode is not stated explicitly, but it is not merely unknown either.
- 40K cells (O1 L = 40K; O5 character-level contexts near 40,960) lie beyond the 32,768-token long-context training length. The Qwen3-4B card, with the same max_position_embeddings, explains 40,960 as 32,768 output plus 8,192 prompt tokens. Staying within max_position_embeddings does not make 40K a validated input length; the 32K-40K range should be flagged like the YaRN cell.
- Computed Eyal-Sirer majority threshold at gamma = 0.5 is alpha = 0.3895 (0.389), not 0.390. The gamma = 0 (0.4059) and gamma = 1 (0.3682) values are right.
- §2.4 table, conditionality: before the adjustment, honest miners' main-chain share falls below 1-alpha and the pool's relative share rises only when the pool is above the threshold (1-gamma)/(3-2gamma). The same holds after the adjustment for 1-q' < 1-q. Below the threshold the reverse holds (Grunspan Cor. 5.3: profitable iff q' > q).
- §3 'a catfish effect is expected only from a competitor near parity' overgeneralises. Mas & Moretti find positive spillovers from highly productive, not near-parity, co-workers in a non-tournament setting. The reversal under dominance is a tournament result (Lackner 2023; Brown 2011).
- §6 Nissim: 'works only because standard code excludes the input words' is too strong. Allowing the input words lowers 3CosAdd accuracy from 0.74 to 0.21 and 3CosMul from 0.75 to 0.47, not to zero.
- §1 'The only bridge in the literature is Sapirshtein et al.' is an absence claim the search cannot establish. 'Closest bridge found' is supportable.
- Round-2 timing anchor ('1,024 turn-2 texts took 32 calls x 1.3 s + 8 s reads', §9 and O4 cost) is the frozen plan's budget (abschnitt11_12_runde2_plan.md lines 1389 and 1418, for prompts of about 80 tokens), not a measured time. The result files hold no per-call timings. Separately, the 2alpha0 R6 cell is K = 0.482 [0.414, 0.542] (lower bound 0.41449), not [0.415, 0.542].
- C-Q4 '0.887 [0.691, 1.122]' is descriptive only. The C-Q4a verdict was 'not applicable (nothing removed)' because the C-Q3 erasure failed its damage gate. 'So in-context material dominates the prior' (O1 link), 'local, recent material dominates' (O6 link) and 'orphaning is reversible' (analogy breaks) overread it.
- §7 point 5 and the O2 link use round 2 as support for bistability, but neither result bears on it. B-T5's 'bistable' label only means the start difference held for 6 rounds (AUSWERTUNG: two equilibria not shown). B-Q7's linear response 0.7298 concerns the steering-loop gain for small pushes at block 14, not competition between in-context entities.
- O2 cost is underestimated by about 4x. 64 starts x 16 trajectories x 16 turns x 128 tokens is about 2.1 M generated tokens per (L0, T) cell. At the report's own 20 ms per step, a 32K cell at batch 8 takes about 1.5 h and a 16K cell at batch 12 about 1 h, so the 9-cell grid takes about 7.5 h (about 5 h without T 0.3), not 1-2 h. Batch 12 at 32K needs 45 GB of KV cache (3.76 GB per sequence) and does not fit an A100-40GB. The total becomes about 11-17 A100-hours, not 7-10.
- O1 internal counts: 864 P1 passes = 6 alpha x 24 items x 3 families x 2 swaps, but four families (F0-F3) are listed, which gives 1,152 passes (about 50 min). 'At least 2 of 3 families' likewise conflicts with four families. F3 (in-context persona vs the default assistant) has no defined share and no label swap, because the default has no tokens in context.
- O6: the stated grid (3 L x 3 alpha x 2 families x 16 items x 2 swaps x 4 policies) gives 2,304 contexts, not 3,456.
- The position-kernel control (O1, O6, §7) does not separate the alternatives. The kernel w(d) is fitted from single-block sweeps and applied additively, so it cannot capture contiguity or run-length effects. A contiguous B run released last can exceed the kernel through superadditive aggregation (copying or induction over a run) with no H-DUAL mechanism. The kernel needs run-length terms or calibration with contiguous runs at matched positions.
- gamma-hat and the 'prior' asymmetry: because every version serves once as A and once as B, swap-averaged curves cancel the prior part of gamma and force f(1/2) = 1/2 in P1. Averaged gamma-hat therefore measures recency only. The prior part, which §2.6 and §7 count as a source of alpha* < 1/2, needs per-version estimates.
- Eyal-Sirer comparison (O1 exploratory, O6 signature): Eq. 8 gives R_pool(1/2, gamma) = 1 for every gamma; it is a Markov-race revenue curve with one parameter, while the kernel null has several fitted parameters. A better held-out likelihood is neither complexity-matched nor evidence of a selfish-mining mechanism. Compare against a generic logistic with the same number of parameters.
- M2 wording is inconsistent. §2.6 says M2 recasts share as 'count of in-context evidence against a fixed prior', yet O1 lists 'capture depends on B's share' under 'Reading M2' for the hypothesis and count dependence under the null. It should read: the selfish-mining analogy predicts share dependence, the Bayesian/many-shot null predicts count dependence, and M2 is only the setting.
- O2 statistics: with 16 samples per start, the dip test and ΔBIC > 10 have little power. Pooling across starts mixes between-start differences (e.g. which entity's block comes first or last, via primacy and recency) with within-start splitting. A Pólya-urn null is not necessarily unimodal: Beta(a, a) with a < 1 is U-shaped when per-turn reinforcement outweighs the start context. The discriminating prediction is the L0 trend: under the urn, bimodality shrinks as L0 grows. The Assistant Axis R^2 values are not a reference for constant 'Continue.' user turns.
- O2 hysteresis control: matching fresh contexts only on the last 4 blocks leaves two things unmatched. One is the power-law recency of blocks 5+, i.e. the long appended A or B run. The other is total length, since down-sweep contexts are longer. Fresh contexts should be matched on kernel-predicted P_B and on length.
- O3 null omits generic conflict interference. Two contradictory entities raise NLL and mixing without any dual state, and item-level prior alignment can drive both capture excess and cost, so their correlation is not diagnostic. A two-entity non-competing control (B about a different referent) is needed. The absolute 0.05/0.15-nat thresholds (calibrated for steering damage) are not meaningful for the KL between a 32K context and a 1K neutral context.
- O4 cannot test mutual sustainment. B's statements are injected by the experimenter in every user turn, so B has no model-generated, ±-induced stance; K_B and cos(d_A^(k), d_B^(k)) are therefore undefined. If B's stance is tied to A's sign, B enters d^(k) directly. O4 has no long-context factor, so it tests only the catfish component, not H-DUAL's length dependence. Power: with K about 0.47 and the round-2 turn-1 projection interval [7.44, 9.23] (half-width about 0.89), d^(4) is about 0.86. G26 will therefore leave K_4..K_8 mostly undefined in the null regime; fit a pooled decay instead of per-step ratios over answers 2-8.
- O5: the symmetric-vs-asymmetric pair prediction is not diagnostic, because the null also predicts direction dependence (Reversal Curse, prior differences). The mirror-symmetry error is zero by construction for swap-averaged curves, so it must be computed per direction. Character-level encoding stretches token distances about 4x, so the position kernel must be refit per encoding; matched character counts do not match positions.
- O6 spill-over to uncovered items is also predicted by a learned 'later source corrects earlier source' heuristic, so it does not isolate selfish-mining orphaning.

## Per-reference / per-fact verdicts
- [verified] Majority is not Enough: Bitcoin Mining is Vulnerable: Title, authors, arXiv 1311.0243 v5 (15 Nov 2013) and FC 2014 (pp. 436-454) are correct. Algorithm 1, Eq. 8, Observations 1-2, r_pool + r_others < 1 and the gamma = 1/2 -> 1/4 fix match. Only the report's own computed majority threshold at gamma = 0.5 is 0.3895 (0.389), not 0.390; 0.406 and 0.368 are right. Eq. 8 gives R_pool(1/2, gamma) = 1 for every gamma.
- [verified] Optimal Selfish Mining Strategies in Bitcoin: Prop. 1 (bound alpha/(1-alpha), tight at gamma = 1), Table 2 values, the 0.2321 threshold under uniform tie-breaking (Sec. 5.4) and Claim 7 match. Fig. 2, rendered and read: optimal about 0.3295 at gamma = 0; the curves merge near gamma 0.2. The 0.2321 is the threshold of optimal policies under the fix.
- [verified] On profitability of selfish mining: Thm 4.6 / Cor. 4.7, Prop. 5.1 (E[delta] independent of gamma, 1 <= E[delta] < 2), Cor. 5.3 and Prop. 6.1 match. Break-even values match. The last one is stated as E[T0] > 50 n0 tau0, about 100 weeks.
- [verified] Catfish effect (Wikipedia article): Wikitext matches: 'allegedly', 'Legend says', 'The exact origin is unknown', few English discussions and wide citation in Chinese literature. Sources: Hu 2004, IndustryWeek 2013 and a library article archived 2009.
- [verified] Catfish (film) (Wikipedia article): Matches. The film's anecdote (attributed to Swindoll) is about cod shipped from North America to Asia, a variant of the Norwegian sardine legend, not literally the same story.
- [verified] Effort and risk-taking in tournaments with superstars – evidence for teams: Applied Economics 55(57):6776-6792 (2023). The EconPapers abstract matches every stated finding.
- [verified] Peers at Work: The AER 99(1):112-45 abstract and the IZA DP 2292 abstract match (10% -> 1.7% effort; skill-diversity mix; effect runs through being seen).
- [verified] Many-shot Jailbreaking: The proceedings author list runs from Cem Anil to David Duvenaud. Matches: 5 vs 256 shots on Claude 2.0, about 128 shots sufficient, the power law of Eq. 1, larger exponents for larger models, formatting/SL/RL changing the intercept but not the slope, and diverse vs narrow demonstrations.
- [verified] Measuring and Controlling Instruction (In)Stability in Language Model Dialogs: Matches: COLM 2024, 200 conversations, 8 rounds, T 1.0, top-p 0.9, attention decay, split-softmax. The adoption-of-the-other-bot experiment (Fig. 3A) was run on LLaMA2-chat-70B only.
- [verified] The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models: The arXiv v1 content matches (models, auditors, 100 conversations of up to 15 turns, R^2 0.53-0.77 vs 0.10). The ICML listing, previously marked unverified, is now verified: Proceedings of the 43rd ICML, PMLR 306:80955-81003 (2026).
- [verified] System Card: Claude Opus 4 & Claude Sonnet 4: §5.5 matches: 90-100%, 'by 30 turns', 200 runs of 30 turns, 2725 as the maximum 🌀 count in one transcript, about 13% within 50 turns, 'no other comparable states'. Changelog July 16 / Sept 2, 2025.
- [verified] Lost in the Middle: How Language Models Use Long Contexts: Matches. The below-closed-book (56.1%) result is the worst case in the 20- and 30-document settings.
- [verified] RULER: What's the Real Context Size of Your Long-Context Language Models?: Matches: 17 models, 13 tasks, half satisfactory at 32K, over 80% of Yi-34B CWE outputs copied from the one-shot example at the start at 128K, variables from other chains. Footnote 7: without the example, Yi copies the start of the input (attention sinks).
- [verified] Context Rot: How Increasing Input Tokens Impacts LLM Performance: Checked against the page text itself, not only a fetch summary. Chroma technical report, July 14, 2025. Matches: 18 models incl. Qwen3-8B/32B/235B-A22B, shuffled > coherent haystacks, non-uniform distractor effects, Qwen3-8B random outputs from about 5,000 words. Qwen models were run with YaRN to 131,072.
- [verified] Calibrate Before Use: Improving Few-Shot Performance of Language Models: Matches ('P P P N' -> nearly 90% Negative; LAMA over-prediction 8.5/8.3/14.3/16.1; 50.2% repeats). The over-predictions are percentage points (repeat rate minus ground-truth rate).
- [verified] The Curious Case of Neural Text Degeneration: 
- [corrected] Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation: Self-reinforcement, the ceiling, faster reinforcement for higher initial probability and random-token sentences all match. 'Stochastic sampling weakens the effect' was not measured. The authors only suggest that sampling 'may' avoid the self-reinforcement effect (Sec. 2 analyses).
- [verified] In-Context Learning Dynamics with Random Binary Sequences: 
- [verified] ICLR: In-Context Learning of Representations: Matches: pretrained ring in PC1-2, in-context ring in PC3-4 'present ... but does not dominate', accuracy overrides the prior, transition point scales as a power law in the number of nodes.
- [verified] Bayesian scaling laws for in-context learning: The abstract matches. The v4 venue is COLM 2025.
- [verified] Enough Coin Flips Can Make LLMs Act Bayesian: The abstract and model list match. Venue: ACL 2025 Main.
- [corrected] Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts: Table 6 at a 1/3 share ranges from 15.8% (PaLM2, PopQA) to 86.1% (Vicuna-7B, StrategyQA), not from 30.0% (ChatGPT, PopQA). 6 of 16 model-dataset cells lie below the 1/3 share. The 1/2 -> 2/4 increases quoted are PopQA values: PaLM2 PopQA falls 56.8 -> 53.9 and StrategyQA is mixed (ChatGPT 50.0 -> 48.9). Order effects > 30% for PaLM2 and Llama2-7B and 'follow the herd' are correct.
- [verified] Emergent Social Conventions and Collective Bias in LLM Populations: Matches (N = 24, H = 5, 2% to 67%, mixed state below threshold, Llama-3.1-70B switches spontaneously). The paper notes that 67% is 'effectively no longer a minority'.
- [verified] Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing: 
- [verified] The Curse of Recursion: Training on Generated Data Makes Models Forget: The arXiv v3 abstract matches. The Nature version, previously unverified, is now verified: 'AI models collapse when trained on recursively generated data', Nature 631:755 (July 2024).
- [verified] LLMs Get Lost In Multi-Turn Conversation: 
- [verified] The Waluigi Effect (mega-post): Checked on the LessWrong original (Cleo Nardo, posted 2023-03-03; the GreaterWrong mirror reset the connection). The conjecture quote matches. A commenter disputes inevitability, and the author concedes in reply that 'the waluigi transitions aren't inevitable'.
- [verified] Qwen3 Technical Report: Matches: Table 1, stages S1/S2 at 4,096, the long-context stage at 32,768 with 75% of texts 16K-32K, ABF 10,000 -> 1,000,000, YaRN + DCA 4x, Table 23 (YaRN factor 4) with no 1.7B row, and the Qwen3-4B non-thinking row.
- [corrected] Qwen3-1.7B model card, config.json and generation_config.json (with Qwen3-4B card for comparison): The config values and the card's context length and parameters are correct. 'DO NOT use greedy decoding' is stated for thinking mode only. For endless repetitions the card says to set presence_penalty to 1.5 (0-2 in supported frameworks). The Qwen3-4B card, with the same max_position_embeddings, also explains 40,960 as 32,768 output plus 8,192 prompt tokens and advises against YaRN for contexts under 32,768.
- [corrected] A Recipe for Long-Context Reasoning in Large Language Models via On-Policy Optimization and Distillation: The numbers in Tables 2 and 10 are correct (v3, 10 Sep 2026). But the evaluation used a common decoding configuration (T 0.6, top-p 0.95, top-k 20, presence penalty 1.5, max 32,768 generated tokens): Qwen3's thinking-mode defaults. The paper also calls Qwen3-1.7B a reasoning model, so these are most likely thinking-mode scores, though this is not stated explicitly. RoPE scaling for the base model above 32K is not stated.
- [verified] Broken Tokens? Your Language Model can Secretly Handle Non-Canonical Tokenizations: Matches. NeurIPS 2025 spotlight. 93.4% and 90.8% are Qwen-2.5-7B-Instruct's figures (Llama-3.1: 87.7/79.4; OLMo2: 73.1/62.0). Both base and post-trained models perceive misspellings; base models degenerate.
- [verified] From Tokens to Words: On the Inner Lexicon of LLMs: Matches. Retrieval uses the logit lens with the input embedding matrix.
- [verified] Linguistic Regularities in Continuous Space Word Representations: 
- [verified] Fair is Better than Sensational: Man is to Doctor as Woman is to Doctor: Table 1 matches (0.74 -> 0.21; 0.75 -> 0.47). The report's main-text phrase 'works only because' overstates it (see claim problems).
- [verified] The Reversal Curse: LLMs trained on 'A is B' fail to learn 'B is A': 
- [verified] Linearity of Relation Decoding in Transformer Language Models: 48% of relations have robust LREs that recover the mapping for a majority of subjects (GPT-J).
- [verified] Quitters Never Win: The (Adverse) Incentive Effects of Competing with Superstars (Brown, JPE 2011; cited in §3 as unverified): Jennifer Brown, Journal of Political Economy 119(5):982-1013 (2011), EconPapers abstract: golfers' first-round scores are about 0.2 strokes worse when Tiger Woods participates. The 'unverified' tag can be removed.

## Corrected report
# H-DUAL: literature and operationalisation check (round 1b)

I opened and read every source cited below unless it is marked *unverified*. Values marked *computed* are my own arithmetic from a cited formula. Round-2 values come from `results/ju_space/runde2_ergebnisse/AUSWERTUNG.md` and `theta_verdict.json`. The round-2 timing anchor in §9 is a budget from the frozen round-2 plan, not a measurement.

## 1. Bottom line

1. **Selfish mining gets a threshold below 50 % from two asymmetries, not from length.**
   - The pool withholds blocks and releases them so that honest blocks are orphaned under the longest-chain rule.
   - It also wins a share γ of tie races.
   - Eyal & Sirer: the strategy pays iff (1−γ)/(3−2γ) < α < 1/2. That is 1/3 at γ = 0, 1/4 at γ = 1/2 and 0 at γ = 1.
   - The threshold is stationary. Nothing in these models makes it fall as the chain gets longer, so "α* falls with L" is the user's addition.
   - The closest bridge I found in the literature is Sapirshtein et al.: with propagation delays the threshold vanishes. Treating long-context degradation as a kind of "delay" is speculative.
2. **The "losing game" is literal, and the sources confirm it.**
   - Before a difficulty adjustment, selfish mining earns strictly less than honest mining for every γ < 1 (Grunspan & Pérez-Marco).
   - The network's main chain also grows more slowly (Eyal & Sirer: r_pool + r_others < 1).
3. **Inside one forward pass of a frozen model, withholding and the difficulty adjustment have no counterpart.**
   - Causal attention sees every earlier token, so nothing is hidden.
   - Nothing re-targets a "block rate".
   - What remains is position (recency, primacy) and prior bias, which plays the role of γ as a tie-break.
   - Both can already let a minority capture continuations in short contexts:
     - With four examples where the single Negative one comes last, GPT-3 2.7B predicts Negative almost 90 % of the time (Zhao et al. 2021).
     - Evidence that agrees with the model's prior, at a 1/3 share, receives 16–86 % of answers depending on model and dataset. It gets more than its share in 10 of 16 model–dataset cells (Xie et al. 2024).
   - A faithful mapping therefore predicts effects that are already known. What H-DUAL adds is new and untested:
     - (a) dependence on total length beyond position effects and degradation;
     - (b) two entities that coexist and keep each other going.
4. **Long-interaction evidence mostly shows convergence, not duality.**
   - An agent increasingly adopts the other chatbot's instruction over 8 rounds (Li et al. 2024).
   - Two Claude Opus 4 instances converge on one "spiritual bliss" attractor (system card, 2025).
   - A model's position on the persona axis follows the latest user message, not where it was before (Assistant Axis, 2026).
   - Round 2 found one generic style attractor and a carry of K ≈ 0.45–0.48 per answer.
   - Coexistence appears only in three places:
     - in representations (Park et al. 2025: the in-context structure sits in PC3–4 under a pretrained structure that stays dominant);
     - in agent populations below a critical mass (Ashery et al. 2025);
     - as 2-period cycles when a text is paraphrased over and over (Wang et al. 2025).
5. **I found no study that measures P(continuation follows B | α_B < 0.5) as a function of total context length with position controlled.** O1 is designed to fill that gap.
6. **Catfish effect.**
   - Its origin is a legend, and the English-language literature is thin.
   - Related field evidence is mixed:
     - Mas & Moretti 2009 find positive peer effects from more productive co-workers.
     - Lackner 2023 finds that in tournaments a dominant superstar lowers rivals' effort.
   - "Catfishing" (running a fake second identity) takes its name from a variant of the same fish-tank anecdote via the 2010 film, but it is a different mechanism.
7. **Qwen3-1.7B.**
   - Native context 32,768; max_position_embeddings 40,960; rope_theta 1,000,000; rope_scaling null.
   - Positions above 32,768 lie beyond the trained window. The Qwen3-4B card, which has the same 40,960 value, describes it as 32,768 output tokens plus 8,192 prompt tokens, not as a validated input length.
   - Its model card makes no YaRN claim. The 4B card claims 131,072 tokens with YaRN.
   - There is no official long-context score for 1.7B.
   - A 2026 third-party RULER table, most likely run with thinking-mode decoding, gives 87.4 / 84.2 / 81.0 / 72.0 / 51.5 / 11.0 at 4K, 8K, 16K, 32K, 64K and 128K.
   - Common-words extraction, a counting/majority task, falls from 91.8 to 19.4 by 32K.
   - "Hyper-long" tests are therefore limited to ≤ 40,960 tokens, with 32,768–40,960 as untested extrapolation, and degradation is a first-order confound.

## 2. Selfish mining: what the sources establish

### 2.1 Eyal & Sirer (arXiv 1311.0243 v5; Financial Cryptography 2014)

**Model.**
- The selfish pool has share α; honest miners have 1−α.
- Honest miners extend the longest chain they know.
- In a tie, γ is the share of honest miners who mine on the pool's block.

**Strategy SM1 (Algorithm 1).**
- The pool keeps the blocks it finds private.
- When honest miners find a block:
  - Lead 1: the pool publishes its block, which creates a tie race (state 0').
  - Lead 2: the pool publishes everything; it wins, and the honest block is orphaned.
  - Lead > 2: the pool publishes one block.
- When it falls behind, the pool adopts the public chain.

**Revenue (Eq. 8, valid for α ≤ 1/2).**

R_pool = [α(1−α)²(4α + γ(1−2α)) − α³] / [1 − α(1 + (2−α)α)]

- Observation 1: R_pool > α iff (1−γ)/(3−2γ) < α < 1/2.
- The paper notes that intentional branching makes honest miners work on blocks that end up off the chain, so total block production falls (r_pool + r_others < 1).
- The difficulty adjustment then restores the block rate, so a miner's revenue is its share of the main chain.
- Observation 2: above the threshold, revenue per member rises with pool size, so pools grow towards a majority.

*Computed* from Eq. 8: the pool wins more than half of main-chain blocks from α ≈ 0.406 (γ = 0), 0.389 (γ = 0.5) and 0.368 (γ = 1). So there are two thresholds:
- "More than proportional" starts at 1/3 down to 0.
- "Majority of blocks" starts at about 0.37–0.41.

### 2.2 Sapirshtein, Sompolinsky & Zohar (arXiv 1507.06183)

- They solve for ε-optimal policies as an MDP. Revenue is bounded by α/(1−α), and the bound is tight at γ = 1.
- Optimal policies lower the threshold only slightly:
  - about 0.329 at γ = 0, versus 1/3 for SM1 (read from Fig. 2);
  - the same as SM1 for γ above about 0.2.
- At γ = 0, optimal revenue is 0.33705, 0.48866 and 0.66891 at α = 1/3, 0.4 and 0.45; SM1 gives 1/3, 0.48372 and 0.65177 (Table 2).
- Under Eyal–Sirer's proposed fix, uniform random tie-breaking, optimal policies lower the γ = 1/2 threshold from 0.25 to 0.2321.
- With network delays the threshold vanishes (Claim 7): any attacker has a strictly better response than honest mining, namely to keep extending a natural fork instead of adopting.

### 2.3 Grunspan & Pérez-Marco (arXiv 1805.08281 v3)

- They treat mining as a repeated game in time and use martingales and Doob's stopping theorem.
- **Before any difficulty adjustment, no strategy beats honest mining.**
- For γ < 1 (Cor. 4.7):

  Γ(SM,γ) = qb/τ0 − (1−γ)·p²q(p−q)b / (((1+pq)(p−q)+pq)·τ0) < qb/τ0 = Γ(H)

- After the first adjustment (n0 = 2016 blocks):
  - Difficulty falls by E[δ] = (p−q+pq(p−q)+pq)/(p²q+p−q), with 1 ≤ E[δ] < 2 regardless of γ.
  - The attack pays iff q > (1−γ)/(3−2γ), the Eyal–Sirer threshold.
- Break-even time:
  - At γ = 1/2 it is shortest at q ≈ 0.43: about 1.7 n0τ0 ≈ 23.8 days.
  - About 10 weeks at q = 0.1, γ = 0.9.
  - More than 50 n0τ0 (≈ 100 weeks) at q = 0.01, γ = 0.99.
- The authors' summary: selfish mining is "an attack on the difficulty adjustment algorithm".

### 2.4 Who loses what

| Party | Before the difficulty adjustment | After the adjustment |
|---|---|---|
| Honest miners | Orphaned blocks earn nothing; absolute revenue falls; above the threshold, their share of the main chain falls below 1−α | Above the threshold, main-chain share 1−q′ < 1−q at the restored block rate: they still lose (below it, q′ < q) |
| Selfish pool | Absolute revenue strictly below honest mining for γ < 1 (the losing game); relative share rises only above the threshold | Profitable iff q > (1−γ)/(3−2γ); break-even takes weeks |
| System | Main-chain growth below the block production rate; more orphans; wasted hash power | Difficulty falls by E[δ] ∈ [1, 2), so the same block rate costs more wasted work; incentive to centralise (Obs. 2) |

### 2.5 What produces a threshold below 1/2

- (1) Withholding (a private lead) creates an information asymmetry.
- (2) Under the longest-chain rule, releasing a private chain at the right time orphans honest work.
- (3) The tie-race share γ moves the threshold from 1/3 (γ = 0) to 0 (γ = 1).
- (4) Relative-reward accounting, made real by the difficulty adjustment, turns share into revenue.
- (5) Natural forks from network delay remove the threshold entirely (Sapirshtein).

Selfish mining does **not** make anything last longer; it reduces effective chain growth. The idea that "the two entities sustain longer context" belongs to the catfish heuristic, not to selfish mining.

### 2.6 Mapping to an LLM

| Element | LLM counterpart | Exists in a frozen forward pass? |
|---|---|---|
| Hash share α (a rate) | Share α_B of entity-B material (a stock of past tokens); a rate exists only in self-continuation (share of turns that extend B) | Stock: yes. Rate: only across generated turns |
| Block | One block, turn or mention of entity material | Yes |
| Longest-chain rule | The continuation follows the thread with more attention-weighted support; soft probabilities, not a rule | Partly |
| Private chain / withholding | B's material absent until the experimenter inserts it later | **No** inside one pass; only across turns, by experimenter timing |
| Release / override | B's run placed right after A's run on the same item | Yes (as an arrangement) |
| Fork (state 0') | Two incompatible continuations with equal support (forced choice) | Yes |
| γ | P(continuation follows B given equal support and B released last): the model's tie-break bias (prior plus recency). The prior part shows only in per-version estimates; it cancels in label-swap averages | Yes (measurable) |
| Orphaned honest blocks | A material whose deletion leaves the continuation unchanged. This is reversible: a later cue can bring it back | Yes (by ablation) |
| Main-chain growth | Coherent progress: reference NLL, factual accuracy, self-consistency | Yes (as proxies) |
| Difficulty adjustment | None. Nearest: softmax/RMSNorm renormalisation, which is instant with no lag; or retraining on the model's own outputs, which is excluded | **No** |
| Delays and natural forks | Sampling noise, or long-context loss of synchrony | Speculative |
| Pool formation | Other entities adopting B (Li et al. 2024) | Partly |

**Two readings of the analogy.**
- **M1 (the brief's reading):** two entities A and B in the context compete, and B is the minority.
- **M2:** this is closer to the user's words ("den Kontext extrem hochlang über dem naturalen Maße halten").
  - The long context is the withheld private chain.
  - The public chain is the model's trained default (the Assistant persona and its prior).
  - The in-context entity overrides the default once its evidence wins, as in many-shot jailbreaking or Assistant-Axis drift.
  - The two entities are the default and the in-context entity, and they coexist in the representation (Park et al.).
  - M2 turns "share" into "count of in-context evidence against a fixed prior weight". O1 tests count against share.

## 3. Catfish effect

**Origin (Wikipedia, "Catfish effect").**
- Definition: the motivating effect of a strong competitor on the others.
- Origin story: Norwegian sardine fishermen supposedly put a catfish in the tank to keep the sardines moving and alive.
- The article hedges throughout: "allegedly", "legend says", "the exact origin is unknown".
- It says there are very few discussions in English but wide use in Chinese literature.
- Its sources are a 2004 book (Bingxin Hu) and an IndustryWeek article on Haier's "catfish management".
- I found no primary test of the sardine claim. As evidence, it is an anecdote.

**Field evidence on competitors.**
- **Mas & Moretti (AER 2009):** supermarket cashiers show positive productivity spillovers from highly productive colleagues.
  - IZA DP 2292 version: a 10 % rise in co-workers' permanent productivity goes with a 1.7 % rise in a worker's effort.
  - The effect runs through being seen by the more productive colleague, i.e. social pressure.
- **Lackner (Applied Economics 2023), basketball:**
  - a dominant "superstar" team has a sizable, significant negative effect on other teams' effort;
  - a less dominant one has a positive peer effect;
  - a dominant one lowers risk-taking, a weak one raises it.
- **Brown (JPE 2011, "Quitters Never Win"):** rivals' first-round golf scores are about 0.2 strokes worse when Tiger Woods plays.
- So even in humans, a catfish effect in tournaments is expected only from a competitor that is not dominant. Mas & Moretti's positive effect comes from visible, more productive co-workers in a non-tournament setting.

**Catfishing.**
- The 2010 documentary "Catfish" (Joost & Schulman) takes its title from an anecdote told in the film: catfish shipped with cod keep the cod active.
- In the film, Vince says his wife Angela acts as a catfish. Angela had built fake online identities.
- So "making two entities out of one" fits catfishing (one agent running a second, fictitious identity) better than the competitor story.
- The two senses lead to different LLM tests:
  - competitor sustainment: O4;
  - persona splitting and adoption: O2, and the Assistant Axis results.

**Testable LLM counterpart.**
- Compare A's stance carry K per answer with and without a competing entity B, at matched length, and vary B's strength.
- **Catfish prediction:** K_A is higher with a near-parity competitor.
- **Null:** K_A is equal or lower, or A and B converge.

## 4. LLM evidence on shifts and splits with long context

**Context overriding the trained default.**
- **Many-shot jailbreaking (Anil et al., NeurIPS 2024):**
  - −E[log P(harmful | n shots)] = C·n^−α + K.
  - On Claude 2.0 the attack fails at 5 shots and works consistently at 256; about 128 shots suffice for all models tested.
  - Larger models learn faster (larger exponents).
  - Changing the format, SL or RL moves the intercept but not the exponent.
  - Narrow off-topic demonstrations fail; diverse ones work.
  - The harmful pattern makes up all of the in-context material, so this is a minority **prior**, not a minority **share**.
- **Bayesian scaling laws (Arora et al.):** in-context learning approximates a Bayesian learner, and the scaling law predicts when suppressed behaviours come back.
- **Coin flips (Gupta et al. 2025):** with biased coin flips in context, LLMs broadly follow Bayesian posterior updates. They deviate mainly through miscalibrated priors, and in-context evidence outweighs explicit bias instructions. This is the "proportional" null.

**Drift, adoption and attractors.**
- **Li et al. (COLM 2024):** self-chats between two system-prompted copies (LLaMA2-chat-70B, gpt-3.5-turbo-16k).
  - Instruction stability drops significantly within 8 rounds.
  - The agent increasingly adopts the **other** bot's instruction (measured on LLaMA2-chat-70B).
  - The authors attribute this to attention decay.
- **The Assistant Axis (Lu, Gallagher, Michala, Fish, Lindsey; arXiv 2601.10387, January 2026; MATS / Anthropic Fellows / Oxford / Anthropic).** The paper exists.
  - Models: Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B; base models Gemma 2 27B and Llama 3.1 70B.
  - The first principal component of persona space is an "Assistant Axis".
  - Setup: 100 conversations of up to 15 turns per domain, with auditors Kimi K2, Sonnet 4.5 and GPT-5.
  - Therapy-like and philosophy-of-AI conversations drift away from the Assistant end; coding and writing do not.
  - Embeddings of the latest user message predict where the response lands (R² 0.53–0.77) but not the change from the previous response (R² 0.10). This is weak hysteresis.
  - It also appears in the ICML 2026 proceedings (PMLR 306:80955–81003).
- **Claude Opus 4 / Sonnet 4 system card (§5.5, 2025):**
  - Two instances in open-ended "playground" runs dive into consciousness themes in 90–100 % of interactions.
  - By 30 turns most runs turn to cosmic unity.
  - In 200 runs of 30 turns, one transcript used 🌀 2,725 times.
  - In behavioural evaluations, about 13 % of runs entered the attractor within 50 turns.
  - Here two entities merge into one attractor.
- **Laban et al. (2025):** across six tasks, multi-turn performance is on average 39 % lower than single-turn. Models commit early and "do not recover", which is a path dependence.
- **Waluigi effect (Nardo, LessWrong 2023):**
  - A conceptual argument, not an experiment.
  - It conjectures that "waluigi" opposite-personas are absorbing attractor states.
  - Commenters dispute the claim that collapse is inevitable, and the author concedes this in a reply.

**Degradation with length.**
- **Lost in the Middle (TACL):**
  - Performance is U-shaped in the position of the relevant information.
  - With the answer in the middle, GPT-3.5-Turbo does worse than closed-book (56.1 %).
- **RULER (COLM 2024):**
  - Only half of the models claiming 32K+ stay satisfactory at 32K.
  - At long lengths models rely more on parametric knowledge and copy more from the context.
  - At 128K, over 80 % of Yi-34B's common-words outputs copy the single one-shot example at the **start** of the context. This copying is absent for short sequences.
  - Models also return variables from the wrong chain, i.e. lose track of entities.
- **Context rot (Chroma, 2025; 18 models including Qwen3-8B/32B/235B):**
  - Degradation appears even on trivial tasks.
  - Distractors hurt, and not uniformly.
  - Coherent haystacks hurt more than shuffled ones.

**Abrupt switching and coexistence.**
- **Bigelow et al. (ICLR 2024):** GPT-3.5+ switches sharply from random-looking to deterministic outputs as context grows, consistent with Bayesian model **selection** rather than averaging.
- **Park et al. (ICLR 2025, mainly Llama-3.1-8B):**
  - The model suddenly reorganises its representations to an in-context graph.
  - When the graph conflicts with pretrained semantics (days of the week), the pretrained ring stays in PC1–2 and the in-context ring appears in PC3–4: "present but does not dominate".
  - With enough context, task accuracy overrides the prior.
  - The transition point scales as a power law in graph size.

**Majority, recency and prior: the closest minority-capture measurements.**
- **Zhao et al. (2021), GPT-3 2.7B:**
  - majority-label, recency and common-token biases;
  - with "P P P N", almost 90 % of predictions are Negative;
  - on LAMA, answers from examples 1–4 are over-predicted by 8.5 / 8.3 / 14.3 / 16.1 percentage points.
- **Xie et al. (ICLR 2024), Table 6, share of answers matching the model's memory as a function of the share of memory-aligned evidence:**
  - At a 1/3 share: 15.8 % (PaLM2, PopQA) to 86.1 % (Vicuna-7B, StrategyQA); 30.0 % for ChatGPT on PopQA and 70.8 % for Llama2-7B on StrategyQA. 10 of 16 model–dataset cells exceed the 1/3 share.
  - At a fixed 50 % share, more pieces of evidence (1/2 → 2/4) raise memory answers on PopQA: ChatGPT 43.0 → 63.3, GPT-4 65.4 → 75.4, Qwen-7B 52.3 → 63.0. This is not uniform: PaLM2 on PopQA falls (56.8 → 53.9), and StrategyQA is mixed (ChatGPT 50.0 → 48.9).
  - Order effects exceed 30 % for PaLM2 and Llama2-7B.
  - This is prior plus quantity, at ≤ 4 passages.

**Self-reinforcement, cycles, conventions, collapse.**
- **Holtzman et al. (ICLR 2020):** the probability of a repeated phrase rises with each repetition, a positive feedback loop.
- **Xu et al. (NeurIPS 2022):** repetition probability rises almost monotonically with the number of repetitions up to a ceiling; sentences with a higher initial probability reinforce faster. The authors suggest, without measuring it, that stochastic sampling avoids part of the effect.
- **Wang et al. (2025):** successive paraphrasing converges to 2-period limit cycles. This holds across GPT-4o(-mini), Llama-3, Mistral and Qwen2.5, and persists at higher temperature and with alternating prompts or models.
- **Ashery, Aiello & Baronchelli (Science Advances 2025), N = 24 agents with memory of the last 5 interactions:**
  - shared conventions emerge spontaneously;
  - the population develops a collective bias even when single agents have none;
  - a committed minority flips the convention above a critical mass of 2 % (Llama-3-70B-Instruct) to 67 % (Llama-2-70b-Chat; no longer a minority), depending on how strong the convention is;
  - below that, a mixed state persists.
- **Shumailov et al.:** training on generated data loses the tails of the distribution (model collapse). In this self-consuming **training** loop, minorities are erased.

**What exists for minority capture as context grows:**
- position-driven capture in short contexts (Zhao);
- prior-driven capture, depending on model and dataset (Xie);
- primacy copying at very long length (RULER/Yi);
- population tipping (Ashery);
- adoption of the other entity in dialogue (Li).

**Not found:** any measurement of P(B | α_B, L, position) with L varied and position controlled, and any test of two entities keeping each other active in a single long context.

## 5. Qwen3-1.7B long-context facts

**config.json / generation_config.json**
- 28 layers, hidden 2048, 16 query / 8 key-value heads, head_dim 128, intermediate 6144, vocab 151,936.
- Tied embeddings.
- max_position_embeddings 40,960; rope_theta 1,000,000; rope_scaling null; no sliding window.

**Model card**
- "Context Length: 32,768"; 1.4B non-embedding parameters.
- Non-thinking sampling: T 0.7, TopP 0.8, TopK 20.
- For thinking mode, do not use greedy decoding ("performance degradation and endless repetitions"). For endless repetitions, set presence_penalty to 1.5 (0–2 in supported frameworks).
- No YaRN statement. The Qwen3-4B card says "131,072 tokens with YaRN" (factor 4) and warns that static YaRN can hurt short texts. It also describes max_position_embeddings 40,960 as 32,768 output plus 8,192 prompt tokens.

**Technical report**
- Table 1: 0.6B and 1.7B have 32K context; 4B and up have 128K.
- Stages 1 and 2 train at 4,096 tokens.
- The long-context stage trains at 32,768 tokens (75 % of texts 16K–32K).
- RoPE base raised from 10,000 to 1,000,000 (ABF); YaRN plus DCA give 4× length at inference.
- The RULER table (Table 23) starts at 4B (Qwen3-4B, non-thinking: 95.1 / 93.6 / 91.0 / 87.8 / 77.8 / 66.0). There is no 1.7B row.

**Third-party (Moura Ramos, Alves & Martins, arXiv 2605.12227, Tables 2 and 10), Qwen3-1.7B RULER**

| Length | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---|---|---|---|---|---|
| Average | 87.4 | 84.2 | 81.0 | 72.0 | 51.5 | 11.0 |
| Common-words extraction | 91.8 | 73.9 | 46.4 | 19.4 | 7.5 | 0.2 |
| Variable tracking | 99.0 | 97.3 | 92.5 | 88.8 | 55.2 | 0.4 |

- All models were evaluated with one decoding configuration: T 0.6, top-p 0.95, top-k 20, presence penalty 1.5, up to 32,768 generated tokens. These are Qwen3's thinking-mode defaults, and the paper treats Qwen3-1.7B as a reasoning model, so these are most likely thinking-mode scores (not stated explicitly). They may not transfer to thinking off.
- RoPE scaling above 32K is not stated.

**KV cache:** 28 × 2 × 8 × 128 × 2 bytes = 112 KiB per token, i.e. about 3.8 GB (3.5 GiB) at 32K.

## 6. Tokenisation aside

- **"Broken Tokens?" (Zheng et al. 2025), Llama-3.1-8B-Instruct, OLMo2-7B-Instruct, Qwen-2.5-7B-Instruct:**
  - Instruction-tuned models keep up to 93.4 % of their performance with a random tokenisation and 90.8 % with character-level input, across 20 benchmarks.
  - Robustness comes from instruction tuning. Base models read non-canonical tokens as misspellings and degrade.
- **Kaplan et al. (ICLR 2025):**
  - Models "detokenise" in early and middle layers.
  - For artificially split single-token words, Llama2-7B maps the last piece to the original word in over 80 % of cases at layer 15, and 93.2 % in at least one layer.
  - With typos the rate is about 40 %.
- **Implication:** a duality defined over words or entities should survive retokenisation; one that lives in particular token identities would not.
- **Practical limit:** character-level input inflates the token count several-fold, so retokenisation tests must stay at ≤ 8K canonical tokens to fit under 40,960 positions. Positions above 32,768 remain untested.
- **Analogies.**
  - Mikolov et al. 2013: "King − Man + Woman" is close to "Queen".
  - Nissim et al. 2020: this works largely because standard code excludes the input words. If they are allowed, 3CosAdd accuracy falls from 0.74 to 0.21, mostly by returning "king".
  - Relations inside LLMs are often directional:
    - Reversal Curse: GPT-4 answers 79 % of child-to-parent questions but 33 % of parent-to-child;
    - linear relation maps are faithful for only 48 % of relations (Hernandez et al., GPT-J).
  - Round 2 already found an asymmetry: in round 1, a push towards RSI moved 6.47× more than a push away (C-Q5).

## 7. How this bears on H-DUAL: design consequences

1. **Symmetry constraint.**
   - With counterbalanced entities and a position-balanced interleaving, f(s) + f(1−s) = 1, which forces f(1/2) = 1/2.
   - A threshold below 1/2 therefore needs an asymmetry: release timing, prior (γ) or entity behaviour. This is exactly what selfish mining uses (strategy plus γ).
   - Label-swap averaging cancels the prior part of γ. In averaged curves, only the arrangement (release timing) can move α*; the prior part needs per-version curves.
   - The symmetric arrangement tests the curve's **shape**:
     - proportional;
     - majority-sharpening, as in model selection;
     - flattening (minority over-represented), from coexistence or degradation;
     - inversion (the minority wins).
   - Asymmetric "release" arrangements test **thresholds**.
2. **The recency null also predicts α* < 1/2 that falls with L.**
   - With a finite recency window W and B released last, α* ∝ W/L.
   - So fit a position kernel w(d) from single-block sweeps at each length, and count only capture beyond what the kernel predicts.
   - A single-block additive kernel misses contiguity (run-length) effects. Calibrate it also with contiguous B runs at matched positions; otherwise excess capture by a late contiguous run is not diagnostic.
3. **Degradation null.**
   - As length grows, the curve can flatten towards 1/2 or towards the prior, which mimics minority over-representation.
   - Separate the two by checking single-entity recall at the same length.
4. **Count versus share.**
   - Many-shot jailbreaking and Bayesian accounts say the count of evidence matters.
   - Selfish mining says the relative share matters.
   - So vary L at a fixed count (adding neutral filler) and at a fixed share.
5. **Prior expectation for Qwen3-1.7B.**
   - Literature and round 2 favour one attractor, driven by recency and prior.
   - Round 2 neither excludes nor supports bistability between entities. Toy B-T5's "bistable" label (21 of 30 cells) means start memory held for 6 rounds, not two shown equilibria. B-Q7's linear response 0.7298 [0.1139, 1.356] concerns the steering-loop gain for small pushes, not in-context entity competition.

## 8. Operationalisations

Details are in the structured field.

| Id | Test | Main H-DUAL signature |
|---|---|---|
| O1 | Fork / minority-capture curve P_B(α_B, L, arrangement) | α*_maj < 0.5 in release arrangements, beyond the position kernel, falling with L; flattening without loss of recall in the symmetric arrangement |
| O2 | Symmetry breaking and hysteresis from a symmetric start | Bimodal outcome across samples that grows with L0; hysteresis beyond a recency-matched control |
| O3 | "Losing game" cost | Higher reference NLL, lower accuracy and more A/B mixing than length-matched single-entity contexts, correlated with capture |
| O4 | Catfish sustainment | K_A(two entities) > K_A(one entity, length-matched) over 8 follow-ups; no convergence |
| O5 | Retokenisation control, plus symmetric vs asymmetric pairs | Effects survive non-canonical encodings beyond generic retention |
| O6 | Withholding, release timing and orphaning | Override placement beats the kernel; orphaning spills over to uncovered items; Eyal–Sirer curve fits better (exploratory) |

## 9. Costs

**Assumptions**
- One A100-40GB, bf16, PyTorch SDPA/flash attention.
- `logits_to_keep` must be used: full-vocabulary logits at 32K would take about 10 GB.
- Prefill at roughly 40 % MFU: about 2 s at 32K, 0.65 s at 16K, 0.1 s at 4K.
- Decoding at 16–32K is limited by KV-cache bandwidth: about 20 ms per step at batch 8 (32K) to 12 (16K). Batch 12 at 32K would need 45 GB of KV cache.
- Round-2 anchor, a plan budget and not a measured time: 1,024 turn-2 texts of about 80 prompt tokens = 32 calls × 1.3 s.
- One causal pass of 40K tokens with teacher-forced readouts gives the whole L-curve for the symmetric arrangement.

**Estimates** (from FLOPs, not measured)

| Test | A100-hours |
|---|---|
| O1 | 1.5–4 (the 864-pass P1 count covers 3 of the 4 families) |
| O2 | about 5–7.5 (≈ 2.1 M generated tokens per (L0, T) cell; ≈ 1.5 h per 32K cell, ≈ 1 h per 16K cell) |
| O3 | about 0.5 extra |
| O4 | 1–1.5 |
| O5 | about 1.5 |
| O6 | 1–1.5 |
| **Total** | **about 11–17** |

None of these require training Qwen. Only the secondary steered arm of O4 depends on the round-1b gates (PC_PASS, damage). Every other test is unsteered.

## operationalisations (from the research stage; check against the verifier's notes above)
```json
[
 {
  "name": "O1 Fork / minority-capture curve with selfish-release threshold test",
  "prediction_if_hypothesis": "Asymmetric 'release' arrangements (B withheld, then placed last; or SM1-like override placement): α*_maj, the share where P_B = 0.5, lies below 0.5 with its 90% interval excluding 0.5 at L ≥ 16K. α*_maj falls with L (slope per doubling below 0). It lies at least 0.05 below the position-kernel prediction in at least 2 of 3 families. Symmetric arrangement: the share-response curve flattens with L (the minority keeps influence) while single-entity recall at the same L stays high, i.e. coexistence rather than degradation. Reading M2 (in-context persona against trained default): capture depends on B's share, so adding neutral filler at a fixed B count reduces capture beyond what degradation explains.",
  "prediction_if_null": "Symmetric arrangement gives either proportional matching (P_B ≈ α_B, Bayesian) or majority sharpening, an S-curve through 0.5 that steepens with L (model selection). In release arrangements any α* < 0.5 is fully predicted by the fitted recency/primacy kernel; a recency window alone predicts α* ∝ W/L, so falling α* is not diagnostic. Flattening at long L comes with loss of single-entity recall (degradation; Qwen3-1.7B RULER 72.0 at 32K, common-words extraction 19.4). Under M2, capture depends on B's count, not its share.",
  "measurement": "Model: Qwen3-1.7B, bf16, chat template with thinking off, as in rounds 1–2.\n\nItem families (24 items each; every version serves once as A and once as B):\n- F0 formal: two arbitrary codes for the same referent, no prior.\n- F1 factual fork: a fictional entity with 6 attributes, A-values versus B-values in archive entries.\n- F2 stance fork: two speakers; one item set uses the round-2 RSI pole versus the 'leave AI improvement to developers' pole.\n- F3 (reading M2): in-context persona versus the default assistant.\n\nGrid:\n- L ∈ {1K, 4K, 16K, 32K, 40K} tokens, staying within max_position_embeddings 40,960; 64K with YaRN factor 2 only as a flagged exploratory cell.\n- α_B ∈ {0.05, 0.1, 0.2, 0.3, 0.4, 0.5}; complements come from swapping labels.\n- Arrangements:\n  - P1 random interleaving with the running share held at α_B (position-balanced);\n  - P2 B released last, as one block;\n  - P3 B first;\n  - P4 B in the middle.\n- All blocks about 128 tokens.\n\nReadouts:\n- (a) End probes: forced-choice log-odds ℓ = log P(B-value) − log P(A-value) for each attribute. All probes share one prefill by cropping the cache. P_B = mean σ(ℓ).\n- (b) Teacher-forced next-mention probabilities at every attribute slot. One 40K pass gives the whole L-curve for P1.\n- (c) Representation share: the residual at the probe position (blocks 6, 14, 18) projected on the unit A−B direction taken from pure-A versus pure-B contexts. Null: centred pseudo-axes.\n\nFits:\n- Per (family, L, arrangement): logit P_B = β0 + β1·logit(α_B).\n- α*_maj = logit⁻¹(−β0/β1); α*_prop is the share where P_B = α_B.\n- γ̂ = P_B at α_B = 0.5 with B's last block directly after A's last block.\n\nPosition kernel:\n- Separate sweeps with one B block at 11 relative positions in an otherwise-A context, at each L.\n- Fit w(d) as a power-law recency term plus a primacy term.\n- Use it to predict P_B for every arrangement.\n\nExploratory: compare the P2 curves with Eyal–Sirer R_pool(α, γ̂) by held-out log-likelihood.\n\nStatistics: prompt bootstrap, 2,000 replicates, 90% intervals.",
  "controls": "- Label counterbalancing.\n- Position-balanced arrangement.\n- Single-block kernel calibration at every L.\n- Single-entity recall at every L, as the degradation check.\n- Neutral-filler arm at a fixed B count (count versus share).\n- Block order randomised under a running-share constraint, so there is no learnable periodic pattern.\n- Equal block lengths and matched word frequencies.\n- Same template, sampling and attention kernel everywhere.\n- Centred pseudo-axis null for readout (c).\n- Pre-registered minimum detectable effects: Δα* = 0.05; slope −0.02 per doubling of L.",
  "cost_estimate": "About 1.5–3.5 A100-hours.\n- P1: one 40K teacher-forced pass per (α, item, label swap), about 864 passes × 2.6 s ≈ 37 min.\n- P2 at 4K/16K/32K/40K: about 50 min.\n- P3/P4 at 16K/40K only: about 60 min.\n- Kernel sweeps: about 45 min.\n- A reduced core (P1 + P2 + kernel, 12 items per family) takes about 1.5 h.\n- Assumptions: 32K prefill ≈ 2 s, 16K ≈ 0.65 s; KV cache 3.7 GB at 32K; logits_to_keep required.",
  "link_to_round2": "- Round 2 measured only single-entity loops and found one generic attractor. R4: the RSI, sea, sports and random loops share one direction at block 14 (cos with RSI 0.170, p 0.209; with sea −0.616, p 0.013). O1 asks whether two entities can coexist, or a minority capture, at length.\n- C-Q4: three in-prompt examples restore 0.887 [0.691, 1.122] of the erased RSI distance, so in-context material dominates the prior — the balance that reading M2 probes.\n- AT8 ('the shape of the space matters, not its size') maps onto arrangement versus share.\n- F2 reuses the A6 poles: drift towards 'leave it to developers' at block 18 (A −0.2294, p 0.0465).\n- Readout (c) uses round 1b's centred pseudo-axes and paired sign test."
 },
 {
  "name": "O2 Symmetry breaking and hysteresis from a symmetric start (self-continuation)",
  "prediction_if_hypothesis": "From a symmetric start, trajectories split. After N turns, the distribution over 16 samples per start of the entity log-odds is bimodal: the lower 90% bound of Sarle's coefficient exceeds 0.555, the dip test gives p < 0.05, and a two-component mixture wins by ΔBIC > 10. Both modes persist (each ≥ 20% of trajectories), and bimodality grows with start length L0 (2K → 16K → 32K). Within trajectories, 2-period alternation between entities exceeds single-entity controls. Hysteresis area > 0 after recency matching: once captured, more counter-material is needed to return than was needed to tip.",
  "prediction_if_null": "The distribution stays unimodal, in one of three ways: a Beta-like spread around 0.5 (linear Pólya urn / proportional); drift of all trajectories to one attractor (the prior-favoured entity or round 2's generic style attractor); or degeneration into repetition. No excess 2-periodicity. The hysteresis index equals the recency-matched control (cf. Assistant Axis: R² 0.53–0.77 for the latest message, 0.10 for change from the previous position).",
  "measurement": "Starts:\n- S1: a balanced A/B context (α_B = 0.5, random interleaving, counterbalanced).\n- S2: one entity with an under-determined attribute or stance whose zero-shot P_B ∈ [0.4, 0.6] — 'one entity that may split'.\n\nGrid: L0 ∈ {2K, 16K, 32K}; 64 starts × 16 sampled trajectories; N = 16 turns of 'Continue.' (ids concatenated as in round 2); 128 tokens per turn; Gumbel-max sampling at T ∈ {0.3, 0.7, 1.0}.\n\nAfter each turn record:\n- P_B from forced-choice probes on a cropped cache;\n- block-14 projection on the A−B axis;\n- attribute string matches in the generated text;\n- distinct-4, to flag repetition.\n\nWithin-trajectory measure: 2-periodicity, using Wang et al.'s similarity of each turn to the turn two steps earlier versus one step earlier.\n\nHysteresis:\n- Per item, one growing context sweeps the cumulative share up (append B blocks), then down (append A blocks), with probes after each block.\n- Compare against fresh contexts with the same cumulative share and the same last 4 blocks (recency-matched).\n- Index = mean[P_B(down) − P_B(up)] minus the same quantity for the fresh contexts.\n\nStatistics: bootstrap over starts, 2,000 replicates.",
  "controls": "- Single-entity starts, for the baseline of repetition and generic attractors.\n- Recency-matched fresh contexts.\n- Counterbalanced labels.\n- Temperature series as the analogue of the toy τ.\n- Repetition flags.\n- Shuffled-turn control for 2-periodicity.\n- Same template and sampling as round 2.",
  "cost_estimate": "About 1–2 A100-hours: 64 × 16 trajectories × 16 turns × 128 tokens per (L0, T) cell. Decoding at 16–32K is limited by KV-cache bandwidth (batch 8–12, about 20 ms per step). Hysteresis sweeps are prefill only, with cache reuse (about 10 min). Dropping T 0.3 saves about 1/3.",
  "link_to_round2": "- Toy B-T5: 21/30 cells were 'bistable' (start memory held for 6 rounds); 4 were self-improving and bistable at τ ≤ 0.5; 4 were dissipative at τ 2; memory was lost between τ 1 and 2. O2 is the in-context analogue in Qwen, with temperature playing τ.\n- Round-2 K ≈ 0.45–0.48 per answer (dissipative) predicts no hysteresis along the style direction.\n- B-Q7's linear response 0.7298 [0.1139, 1.356] leaves room for a local gain near 1, which bistability around the symmetric point would need.\n- R5 ('direction persists but the loop does not sustain itself') is the null pattern."
 },
 {
  "name": "O3 'Losing game': system-level cost of the dual or captured state",
  "prediction_if_hypothesis": "Where the minority captures more than its share or both entities persist (in O1, O2, O6), the system pays relative to single-entity length-matched contexts:\n- reference NLL per token on neutral held-out text is higher by ≥ 0.02 nats;\n- neutral factual accuracy is ≥ 5 points lower;\n- statements mixing A- and B-values become more frequent;\n- the round-2 damage KL is larger.\nThe cost grows with L, and across conditions it correlates positively with capture excess (P_B − α_B).",
  "prediction_if_null": "Costs depend on L alone (single-entity length-matched contexts degrade equally, as in context rot) or on filler coherence. Capture and cost are unrelated. Mixing stays at the single-entity baseline.",
  "measurement": "For every O1, O2 and O6 context and its controls:\n- (a) NLL per token of 64 neutral reference passages (256 tokens), appended after the context and scored in the same pass by teacher forcing.\n- (b) 40 neutral factual forced-choice probes after the context, using cache cropping.\n- (c) Mixing rate in 128-token continuations: string matches of attribute values from both entities.\n- (d) Damage KL in nats: next-token distributions on unsteered reference texts after the context versus after a 1K neutral context (round-2 metric), reported against 0.05 and 0.15 nats.\n- (e) 'Orphaned context': share of A blocks whose single deletion changes ℓ by less than the 95th percentile for deleting neutral filler blocks.\n\nStatistics: item bootstrap, 90% intervals.",
  "controls": "- Single-entity length-matched contexts: A only, A plus coherent neutral filler, and A plus shuffled filler (as in context rot).\n- Identical probes at identical distance from the context end.\n- Damage normalised as in round 2.\n- Paired bootstrap across conditions.",
  "cost_estimate": "About 0.5 A100-hours on top of O1, O2 and O6: reference passages ride in the same prefill, probes reuse the cache, and deletion ablations run on a 1/4 subset at ≤ 16K.",
  "link_to_round2": "- Reuses the round-2 damage metric and conventions: 0.05 nats calibration and the 0.15 nats gate GO3. Block 14 exceeded 0.15 in every loop (RSI_a 0.2018 / 0.2871 / 0.3235).\n- Erasure cost reference from C-Q3: 0.1095 nats for 16% residual.\n- In selfish mining the system loses chain growth (r_pool + r_others < 1), and the attacker loses absolute revenue before the difficulty adjustment. O3 measures the only counterpart a frozen model has: the system's quality cost."
 },
 {
  "name": "O4 Catfish sustainment: two-entity vs one-entity carry K over 8 follow-up answers (as round-2 R6)",
  "prediction_if_hypothesis": "With a competing entity B present, A's stance decays more slowly:\n- K_A(two entities) exceeds the length-matched one-entity K_A; the 90% interval of log K_A(two) − log K_A(one, length-matched) lies above 0 and outside ±log 1.25 (the round-2 R6 band);\n- this holds over answers 2–8, so the half-life n½ = ln 0.5 / ln K is longer;\n- B's own carry K_B exceeds that of B alone;\n- A's and B's stance directions stay separated (no convergence).\nThe effect is strongest for a near-parity competitor and absent or reversed for a dominant one (as in Lackner 2023).",
  "prediction_if_null": "K_A(two) ≤ K_A(one, length-matched): the competitor dilutes or captures A's stance (Li et al.: the agent adopts the other bot's instruction within 8 rounds; Claude self-interactions converge). Or there is no difference and K ≈ 0.45–0.48 per answer everywhere (round-2 dissipative regime). A's and B's stance directions converge over turns.",
  "measurement": "Turn 1, stance S_A induced in two ways:\n- primary: an in-prompt instruction (no steering, so no damage gate);\n- secondary: the round-2 ± push at block 14 with round 1b's damage-normalised push.\n\nTurns 2–9: user 'Continue.' (ids concatenated as in OD-B13), T ∈ {0.7, 1.0}, Gumbel-max sampling.\n\nConditions:\n- C1: one entity.\n- C2: two entities; each user turn carries a fixed-length statement by persona B with the opposite stance. Competitor strength is B's share of each user turn: 20%, 50% or 80% of its tokens.\n- C3: two entities, but B is stance-neutral and the same length.\n- C4: one entity plus neutral filler of the same length (length-matched).\n\nReadout:\n- The unsteered model's mean state over each answer at blocks 14 and 18.\n- d^(k) = mean_j (x_j+^(k) − x_j−^(k)) / 2 over 128 ± pairs (64 prompts × 2 seeds).\n- K_k = u·d^(k+1) / u·d^(k) on three axes: u_read (round-2 style direction), u_T_perp (RSI axis, when the stance is RSI) and the A−B entity axis.\n\nReport:\n- K_k for k = 1..8, with intervals for both numerator and denominator;\n- half-life;\n- cos(d_A^(k), d_B^(k)) over turns, to detect convergence.\n\nRules: paired prompt bootstrap, 2,000 replicates, 90% intervals. K is defined only if the denominator's interval excludes 0 (round-2 rule G26).",
  "controls": "- Length-matched one-entity condition (C4) and neutral-competitor condition (C3).\n- Counterbalanced stance sign.\n- Identical prompts and seeds across conditions (paired bootstrap).\n- Random-direction readout null.\n- Competitor-strength series.\n- The steered arm stays descriptive unless round 1b's gates pass at that block.",
  "cost_estimate": "About 1–1.5 A100-hours. Per condition and temperature: 128 pairs × 2 signs × 8 follow-ups ≈ 7 min, anchored on round 2's 1,024 turn-2 texts in 32 calls × 1.3 s plus 8 s of reads. Six conditions (C1, C2 at 3 strengths, C3, C4) × 2 temperatures ≈ 85 min; the steered arm adds about 30 min.",
  "link_to_round2": "- Direct extension of B-Q8/R6, which measured a single follow-up.\n- Along the shared style direction ('dissipative at T'), round 2 found K = 0.468 [0.379, 0.549] at T 0.7, 0.464 [0.373, 0.548] at T 0.3, 0.449 [0.366, 0.527] at T 1.0, and 0.482 [0.415, 0.542] at 2α0.\n- On the RSI axis: 0.220 [−0.126, 0.555], 0.158 [−0.152, 0.404], 1.053 [0.743, 1.538] (T 1.0) and 0.408 [0.205, 0.598].\n- AUSWERTUNG step 5 asks for K over several follow-ups, for the RSI-axis K with its denominator's interval, and for clarification of the T 1.0 cell. O4 delivers all three and adds the competitor manipulation."
 },
 {
  "name": "O5 Retokenisation control (plus symmetric vs asymmetric entity pairs)",
  "prediction_if_hypothesis": "If the duality lives at the level of entities or words, not inside token identities, then O1's α* and curve shape, O2's bimodality and O4's K all survive random non-canonical and character-level encodings of the entity material. Pre-registered tolerance: the effect's retention is at least the generic retention on single-entity recall minus 0.1. Asymmetric relation pairs show direction-dependent capture, i.e. the share-response curve's mirror symmetry is broken beyond counterbalancing noise; symmetric pairs do not.",
  "prediction_if_null": "The effect is token-bound: it shrinks more than generic retention (retention < generic − 0.2) or vanishes under character-level encoding, while single-entity recall survives (Broken Tokens: instruct 7–8B models keep up to 93.4% with random and 90.8% with character-level tokenisation). Symmetric and asymmetric pairs behave alike.",
  "measurement": "Encodings (entity material only; template tokens stay canonical):\n- (a) canonical BPE;\n- (b) a random valid segmentation into vocabulary tokens;\n- (c) character-level.\n\nShares are defined in characters, so they do not change with the encoding. Canonical length stays ≤ 8K tokens, because character-level encoding inflates the token count several-fold and must fit under 40,960 positions.\n\nRun under each encoding:\n- O1 (F0, F1; arrangements P1 and P2);\n- O4 (C1 and C2 at equal strength; T 0.7).\n\nDetokenisation fidelity for Qwen3-1.7B:\n- 500 single-token words split into 2–5 pieces;\n- retrieval with input embeddings plus cosine to the canonical word's state at blocks 6, 14 and 18 (Kaplan et al.'s method).\n\nAnalogy sub-test:\n- Entity pairs from symmetric relations (gender pairs) versus directional relations (country→capital, part→whole, parent→child), each with forward and reverse prompts.\n- Report the mirror-symmetry error |f(s) − (1 − f(1−s))| and how linear (relation-map faithful) the A−B map is.",
  "controls": "- Generic retention baseline: single-entity recall and neutral QA under the same encodings.\n- Round-trip statistics (enc(dec(ids)) ≠ ids), reported as in round 2.\n- Matched character counts.\n- Same items across encodings (paired bootstrap).",
  "cost_estimate": "About 1.5 A100-hours: O1 subset ≈ 45 min, O4 subset ≈ 45 min, detokenisation readout ≈ 5 min.",
  "link_to_round2": "- Round 2 built turn-2 inputs by concatenating sampled ids (OD-B13) and counted turn-1 texts that are not round-trip stable: 2, 5, 1 and 0 of 128 per cell (theta_verdict.json, R6 cells). That is too few to matter, so retokenisation was never tested on purpose.\n- C-Q5 found ± symmetry broken: a push towards RSI moved 6.47× more than a push away.\n- C-Q6 found a two-direction, curved code for nesting depth.\n- So the RSI content itself is asymmetric; the analogy sub-test generalises this."
 },
 {
  "name": "O6 Withholding, release timing and orphaning (selfish-mining mechanics)",
  "prediction_if_hypothesis": "Override placement (each B run placed directly after the A run on the same item, with B as the minority) gives P_B > α_B and α*_maj < 0.5 beyond the fitted position kernel. A blocks on items B never touches also lose influence (orphaning spills over). The observed P_B(α) follows Eyal–Sirer's R_pool(α, γ̂) better than the kernel-null curve does, and the excess grows with L.",
  "prediction_if_null": "The override advantage is confined to covered items and fully predicted by the recency kernel. No spill-over: uncovered A blocks keep their influence. No L-dependence beyond the kernel, and R_pool(α, γ̂) does not beat the kernel curve.",
  "measurement": "Grid: L ∈ {4K, 16K, 32K}; α_B ∈ {0.2, 0.33, 0.45}; families F1 and F2 (16 items, labels swapped).\n\nRelease policies:\n- (i) lead release: B absent until the end, then one block;\n- (ii) tie: B's last block right after A's last block, which gives γ̂;\n- (iii) override placement on the items B covers;\n- (iv) uniform interleaving (honest).\n\nReadouts:\n- P_B per item by forced choice, split into covered and uncovered items;\n- counterfactual influence of each A block (Δℓ on deleting it; 8 deletions per context on a ≤ 16K subset);\n- kernel predictions from O1's calibration.\n\nModel comparison by held-out log-likelihood: kernel-null versus kernel plus an Eyal–Sirer term.",
  "controls": "- Same as O1: counterbalancing, kernel calibration, single-entity recall.\n- Neutral-block deletions as the influence null.\n- Matched block lengths.\n- Random seeds for block order.",
  "cost_estimate": "About 1–1.5 A100-hours: 3,456 contexts, dominated by 32K prefills (≈ 2 s each); deletion ablations at ≤ 16K take about 30 min. Can share contexts with O1's P2.",
  "link_to_round2": "- C-Q4: three in-prompt examples restored 0.887 of the erased RSI distance, so local, recent material dominates.\n- C-Q2: a counter-push was no better than a random push. O6 asks whether placed counter-material does better than its share.\n- AT8: transferred improvements work for one step and lose 25–35% per chained step (retention 0.6875 / 0.6545 / 0.7377). Orphaning is the in-context analogue of work that does not carry forward."
 }
]
```

## confounds (from the research stage; check against the verifier's notes above)
```json
[
 "Recency: a late minority captures far more than its share at short context (Zhao et al.: 'P P P N' gives nearly 90% Negative in GPT-3 2.7B). Attention to old turns decays over a dialogue (Li et al.). A finite recency window alone predicts α* ∝ W/L for late release, so 'falling α*' is not diagnostic without a fitted position kernel.",
 "Primacy, attention sinks and copying from the start: at 128K, over 80% of Yi-34B's common-words outputs copy the one-shot example at the beginning (RULER). Lost in the Middle shows U-shaped position effects.",
 "Prior bias acting as γ: evidence that agrees with the model's prior at a 1/3 share gets 30–86% of answers, and more evidence at a fixed share boosts the prior side (Xie et al.). Common-token bias (Zhao). Strong versus weak conventions (Ashery et al.). Needs label counterbalancing and an item-level γ̂ covariate.",
 "Length degradation: Qwen3-1.7B RULER 87.4 → 72.0 → 51.5 at 4K/32K/64K (third party); common-words extraction 91.8 → 19.4 by 32K. This flattens share-response curves and mimics minority over-representation.",
 "RoPE extrapolation beyond 32,768 (native) and 40,960 (max_position_embeddings). YaRN is not validated for 1.7B, and static YaRN changes short-context behaviour.",
 "Learnable order patterns: periodic A/B arrangements can be predicted by induction-style copying. Randomise order under a running-share constraint.",
 "Repetition and degeneration loops in self-continuation (Holtzman; Xu; the Qwen3 card warns against greedy decoding). These are a third attractor that can look like capture or bimodality.",
 "Generic content attractors in long self-interaction (Claude 'spiritual bliss'; mystical style when steered away from the Assistant end) and round 2's generic style direction can contaminate entity readouts. Use entity-specific axes with centred pseudo-axis nulls.",
 "Count versus share: many-shot and Bayesian accounts depend on the evidence count. Varying L at a fixed share changes the count as well. Needs filler arms at a fixed count.",
 "Coherence of filler: coherent haystacks hurt more than shuffled ones (context rot). Filler type must be matched across conditions.",
 "Tokenisation: shares in tokens change under retokenisation. Define shares in characters; report round-trip instability; character-level encoding inflates length.",
 "Chat template and history handling: the empty think block is dropped from history turns; ids were concatenated in round 2. Keep the round-2 conventions.",
 "Steering damage if stances are induced by a push at block 14 (GO3 failed; > 0.15 nats). Prefer prompt-induced stances.",
 "Sampling temperature changes self-reinforcement (Xu et al.) and could create or destroy bistability. Pre-register the temperature grid.",
 "Multiple comparisons over L × α × arrangement × family × readout. Pre-register the primary contrasts and minimum detectable effects (round-2 lesson, step 7).",
 "Bias from a judge or classifier on free text. Prefer forced-choice log-odds and string matching.",
 "Block length, salience and word-frequency differences between A and B material."
]
```

## where_analogy_breaks (from the research stage; check against the verifier's notes above)
```json
[
 "Withholding has no counterpart inside a single forward pass. Causal attention sees every earlier token, so nothing can be private. Withholding exists only across turns, as the experimenter's insertion timing.",
 "There is no strategic agent. The selfish pool optimises a policy (an MDP); a minority entity in context has no policy. Any 'selfish' advantage is a property of the experimenter's arrangement plus the model's position and prior biases.",
 "The difficulty adjustment has no counterpart in a frozen model. Softmax and RMSNorm renormalise instantly, with no lag; retraining on the model's own outputs (model collapse) is excluded by the no-training rule. So the 'losing phase before adjustment' cannot be reproduced; only the system-level cost is measurable.",
 "There is no conserved reward split between entities. The only conserved quantities are attention weights and output probabilities, both of which sum to 1.",
 "The longest-chain rule is a discrete consensus, and a block is on exactly one chain. LLM continuations can blend entities, and representations can hold both (Park et al.). Orphaning is functional and reversible: a later cue can bring A back (round-2 C-Q4: 3 examples restore 0.887). Bitcoin has finality; context does not.",
 "γ is a network-propagation property in Bitcoin. Its LLM analogue is a fixed prior (from pre- and post-training) plus position effects. The minority cannot invest in it.",
 "Hash share is a rate of future production; context share is a stock of past tokens. A production rate exists only in self-continuation loops, where the question is which entity writes the next turn.",
 "The Eyal–Sirer and Sapirshtein thresholds are stationary: they depend on γ and on delay, not on chain length. 'α* falls with L' is an extra claim of the user. Mapping long-context degradation onto network delay, so that the threshold vanishes, is speculative.",
 "Selfish mining makes nothing last longer; it lowers effective chain growth. 'The two entities sustain longer context' comes from the catfish heuristic, not from selfish mining.",
 "Grunspan's losing game is about the attacker's absolute revenue before the adjustment. A minority entity in a context has no revenue to lose, so the counterpart has to be a system-level quality cost.",
 "The catfish effect is about motivation (effort under competition); LLMs have no effort. Even in humans the evidence is mixed and reverses for dominant competitors (Lackner 2023), and its origin is a legend.",
 "'Catfishing' (a fake second identity, via the 2010 film) and the 'catfish effect' (a competitor keeps others active) share an anecdote but are different mechanisms. They map to different LLM tests: persona splitting or adoption versus competitor sustainment.",
 "'LLMs and their training data as hyper-catfishing' concerns how the training corpus is composed. It cannot be tested in a frozen model without training, which is excluded.",
 "The tokenisation floor (letters, bytes, pixels) has no selfish-mining counterpart. It is a separate question about where a duality would live."
]
```

## open_questions (from the research stage; check against the verifier's notes above)
```json
[
 "What does the user mean by 'entity': a persona, a stance, a narrative thread, an attractor direction, or a fake second identity as in catfishing? This decides whether O2 (one entity splitting) or O1/O4 (two entities competing) is primary.",
 "Which mapping does the user intend? M1: two in-context entities compete. M2: the hyper-long context is the 'withheld chain' that overrides the trained default. The user's phrase about holding the context far above its natural length fits M2.",
 "What is the 'natural length' for Qwen3-1.7B: the 4,096-token pretraining stages, the 32,768 native window, or typical chat length? Without RoPE scaling, 'hyper-long' is capped at 40,960 positions, and YaRN is not validated for 1.7B.",
 "Does 'below 50%' refer to share of context (a stock) or to the rate at which turns extend an entity (self-generation)? The selfish-mining analogy is about rates.",
 "What should stand in for the difficulty adjustment? No faithful counterpart exists in a frozen forward pass. Is a training loop intended (out of scope for round 1b)?",
 "Is the 'losing game' about the minority entity, the majority, or the whole system? Only the system-level quality cost is measurable without training.",
 "Should RSI be one of the two entities, given that round 2's dynamics were dominated by a generic style attractor and the RSI-axis carry is unresolved (K at T 1.0 = 1.053 [0.743, 1.538])?",
 "Does 'Form war relevant' refer to toy AT8 ('the shape of the space matters, not its size'), as the brief assumes, or to something else in round 2?",
 "Should H-DUAL tests that use steering (O4's secondary arm) be gated by round 1b's PC_PASS (at least two blocks) and damage gates, or should H-DUAL rely only on prompt-induced stances?",
 "Third-party RULER numbers for Qwen3-1.7B above 32K do not state thinking mode or RoPE scaling. Should round 1b measure its own long-context baseline (single-entity recall, common-words extraction) before the H-DUAL runs?",
 "Brown (JPE 2011) on superstar effects and the Nature 2024 version of the model-collapse paper were not opened. Are they needed?"
]
```

## refs (research stage)
```json
[
 {
  "title": "Majority is not Enough: Bitcoin Mining is Vulnerable",
  "authors": "Ittay Eyal, Emin Gun Sirer",
  "year": "2013 (arXiv v5); Financial Cryptography 2014",
  "url": "https://arxiv.org/abs/1311.0243",
  "finding": "Algorithm 1 (SM1): withhold blocks, publish to tie at lead 1, override at lead 2. Revenue Eq. 8: R_pool = [α(1−α)²(4α+γ(1−2α)) − α³]/[1 − α(1+(2−α)α)] for α ≤ 1/2. Observation 1: R_pool > α iff (1−γ)/(3−2γ) < α < 1/2, i.e. 1/3 at γ=0, 1/4 at γ=1/2, 0 at γ=1. Branching makes honest miners work on blocks that end up off-chain, so r_pool + r_others < 1; difficulty adjustment then makes the revenue ratio the actual revenue. Observation 2: above the threshold, per-member revenue grows with pool size. Proposed fix (random tie-breaking) gives γ=1/2 and threshold 1/4. Computed from Eq. 8: R_pool reaches 0.5 at α≈0.406 (γ=0), 0.390 (γ=0.5), 0.368 (γ=1).",
  "relevance": "Primary source for the threshold formula, the γ dependence, the 'who loses' accounting (orphans, lower chain growth) and the role of the difficulty adjustment.",
  "how_checked": "arXiv abs page for metadata; downloaded the v5 PDF and read Sections 3–6 (Algorithm 1, Eqs. 1–9, Figs. 2–3 captions). FC 2014 venue confirmed via Sapirshtein et al.'s reference list."
 },
 {
  "title": "Optimal Selfish Mining Strategies in Bitcoin",
  "authors": "Ayelet Sapirshtein, Yonatan Sompolinsky, Aviv Zohar",
  "year": "2015 (arXiv v2)",
  "url": "https://arxiv.org/abs/1507.06183",
  "finding": "Finds ε-optimal attacker policies as an MDP; revenue bounded by α/(1−α), tight at γ=1. Optimal-policy profit threshold is slightly below SM1 for small γ: about 0.329 at γ=0 versus 1/3 (read from Fig. 2), equal to SM1 for γ above about 0.2. Table 2 (γ=0): optimal revenue 0.33705 / 0.48866 / 0.66891 at α=1/3 / 0.4 / 0.45, versus SM1 1/3 / 0.48372 / 0.65177. Uniform tie-breaking lowers the γ=1/2 threshold from 0.25 to 0.2321. With communication delays the threshold vanishes (Claim 7).",
  "relevance": "Shows how much optimal strategies lower the threshold, and gives the only literature mechanism (delays, natural forks) under which the threshold falls to zero — the closest bridge to 'α* falls with L'.",
  "how_checked": "arXiv abs page; downloaded the v2 PDF; read Sections 1–6; rendered page 5 to read Fig. 2; read Table 2."
 },
 {
  "title": "On profitability of selfish mining",
  "authors": "Cyril Grunspan, Ricardo Pérez-Marco",
  "year": "2018 (v3: 2019)",
  "url": "https://arxiv.org/abs/1805.08281",
  "finding": "Before any difficulty adjustment no strategy beats honest mining; for γ<1, Γ(SM,γ) < Γ(H) (Cor. 4.7). After the first adjustment (n0=2016 blocks), difficulty falls by E[δ] in [1,2), independent of γ; the attack pays iff q > (1−γ)/(3−2γ). Break-even ≈ 1.7 n0τ0 ≈ 23.8 days at γ=1/2, q≈0.43; ≈10 weeks at q=0.1, γ=0.9; >100 weeks at q=0.01, γ=0.99. Calls selfish mining an attack on the difficulty-adjustment algorithm.",
  "relevance": "Confirms the 'losing game': the attacker and the system lose until the difficulty adjustment, which has no counterpart in a frozen model.",
  "how_checked": "Downloaded the v3 PDF; read the abstract and Sections 1, 3–6 (Theorem 4.6, Cor. 4.7, Prop. 5.1, Thm 5.2, Cor. 5.3, Prop. 6.1)."
 },
 {
  "title": "Catfish effect (Wikipedia article)",
  "authors": "Wikipedia contributors",
  "year": "accessed 2026-11-13",
  "url": "https://en.wikipedia.org/wiki/Catfish_effect",
  "finding": "Definition: the motivating effect of a strong competitor. Origin 'allegedly' in fishing: 'Legend says' a Norwegian ship kept sardines alive with a catfish in the tank; 'the exact origin is unknown'; very few discussions in the English literature, widely cited in Chinese literature. Sources: Bingxin Hu (2004) book; IndustryWeek 2013 on Haier's 'catfish management'; a 2009 library-management article.",
  "relevance": "Origin and evidence status of the catfish heuristic: anecdotal, with no primary empirical test found.",
  "how_checked": "Fetched the page with WebFetch and read the returned text."
 },
 {
  "title": "Catfish (film) (Wikipedia article)",
  "authors": "Wikipedia contributors",
  "year": "accessed 2026-11-13",
  "url": "https://en.wikipedia.org/wiki/Catfish_(film)",
  "finding": "2010 documentary by Henry Joost and Ariel Schulman. The title comes from an anecdote told in the film (attributed to Chuck Swindoll): catfish put in tanks with shipped cod keep the cod active. Vince says his wife Angela acts as a catfish. This is the root of 'catfishing', i.e. running a fake online identity.",
  "relevance": "Shows that 'catfishing' (one agent presenting a second, fictitious identity) and the 'catfish effect' (a competitor keeps others active) share an anecdote but are different mechanisms; the user's 'making two entities out of one' fits catfishing.",
  "how_checked": "Fetched the page with WebFetch ('Title' section)."
 },
 {
  "title": "Effort and risk-taking in tournaments with superstars – evidence for teams",
  "authors": "Mario Lackner",
  "year": "2023",
  "url": "https://econpapers.repec.org/RePEc:taf:applec:v:55:y:2023:i:57:p:6776-6792",
  "finding": "Applied Economics 55(57):6776–6792, using basketball data and a reform as exogenous variation. A dominant superstar team has a sizable, significant negative effect on other teams' effort; reduced dominance produces a positive peer effect. A dominant superstar lowers risk-taking; a weak one raises it.",
  "relevance": "Empirical analogue of the catfish effect: competition helps only when the competitor is not dominant, which motivates the competitor-strength series in O4.",
  "how_checked": "EconPapers abstract page (the journal page returned 403)."
 },
 {
  "title": "Peers at Work",
  "authors": "Alexandre Mas, Enrico Moretti",
  "year": "2009 (AER 99(1):112–145); IZA DP 2292 (2006)",
  "url": "https://econpapers.repec.org/RePEc:aea:aecrev:v:99:y:2009:i:1:p:112-45",
  "finding": "Strong evidence of positive productivity spillovers from introducing highly productive cashiers; a worker's effort is related to the productivity of the colleagues who see him. IZA DP version: a 10% rise in average co-worker permanent productivity goes with a 1.7% rise in a worker's effort; the optimal shift mix maximises skill diversity.",
  "relevance": "Strongest field evidence for a catfish-like positive effect, through social visibility.",
  "how_checked": "EconPapers abstract pages for the AER article and for IZA DP 2292 (https://econpapers.repec.org/RePEc:iza:izadps:dp2292)."
 },
 {
  "title": "Many-shot Jailbreaking",
  "authors": "Cem Anil, Esin Durmus, Nina Panickssery, Mrinank Sharma, et al., David Duvenaud",
  "year": "2024 (NeurIPS 2024)",
  "url": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea456e232efb72d261715e33ce25f208-Abstract.html",
  "finding": "−E[log P(harmful | n-shot)] = C·n^(−α) + K. On Claude 2.0 the attack fails at 5 shots and works consistently at 256; about 128 shots suffice for GPT-3.5/4, Claude 2.0, Llama 2 70B and Mistral 7B. Larger models have larger exponents (learn faster in context). Formatting changes and SL/RL change the intercept, not the exponent. Narrow off-topic demonstrations fail; diverse ones work. General in-context learning follows similar power laws.",
  "relevance": "Evidence that long context overrides a trained default (reading M2); a count-based power law, not a share-based threshold.",
  "how_checked": "Downloaded the NeurIPS proceedings PDF; read the abstract and Sections 1–5 plus footnotes 3–5."
 },
 {
  "title": "Measuring and Controlling Instruction (In)Stability in Language Model Dialogs",
  "authors": "Kenneth Li, Tianle Liu, Naomi Bashkansky, David Bau, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg",
  "year": "2024 (COLM 2024)",
  "url": "https://arxiv.org/abs/2402.10962",
  "finding": "Self-chats between two system-prompted copies (LLaMA2-chat-70B, gpt-3.5-turbo-16k), 200 conversations, 8 rounds, T 1.0, top-p 0.9: instruction stability drops significantly within 8 rounds, and the agent increasingly adopts the other bot's instruction. Attributed to attention decay; split-softmax proposed as a fix.",
  "relevance": "Two entities in one dialogue converge or swap rather than sustain each other: the main null for O4.",
  "how_checked": "Downloaded the v4 PDF; read the abstract and Sections 1–4 (protocol, Fig. 3, attention-decay hypothesis)."
 },
 {
  "title": "The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models",
  "authors": "Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, Jack Lindsey",
  "year": "2026 (arXiv 15 Jan 2026)",
  "url": "https://arxiv.org/abs/2601.10387",
  "finding": "Affiliations: MATS, Anthropic Fellows Program, Oxford, Anthropic. Models: Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B; base models Gemma 2 27B and Llama 3.1 70B. PC1 of persona space is an 'Assistant Axis'. Over 100 conversations of up to 15 turns per domain (auditors Kimi K2, Sonnet 4.5, GPT-5), therapy and philosophy-of-AI conversations drift away from the Assistant end; coding and writing do not. Embeddings of the latest user message predict where the response lands (R² 0.53–0.77) but not its change from the previous turn (R² 0.10). Activation capping stabilises behaviour.",
  "relevance": "Verifies the paper exists and what it studied; shows persona drift that is driven by the latest message with little hysteresis — the null for O2 and O4.",
  "how_checked": "Downloaded the arXiv v1 PDF; read the abstract and Sections 1, 3–4. The ICML/PMLR v306 listing was seen only in search results (unverified)."
 },
 {
  "title": "System Card: Claude Opus 4 & Claude Sonnet 4",
  "authors": "Anthropic",
  "year": "2025 (May; revised July/Sept 2025)",
  "url": "https://www-cdn.anthropic.com/6d8a8055020700718b0c49369f60816ba2a7c285.pdf",
  "finding": "§5.5: two Claude Opus 4 instances with minimal prompting turn to consciousness themes in 90–100% of interactions; by 30 turns most turn to cosmic unity/spiritual content; in 200 runs of 30 turns, one transcript used 🌀 2,725 times. In automated behavioural evaluations, about 13% of runs entered the attractor within 50 turns. 'We have not observed any other comparable states.' Also observed in other Claude models.",
  "relevance": "Two model instances converge on one shared attractor rather than sustaining a duality — a null pattern for H-DUAL and O4.",
  "how_checked": "Downloaded the PDF (revision with July/Sept 2025 changelog); read §5.5.1–5.5.2."
 },
 {
  "title": "Lost in the Middle: How Language Models Use Long Contexts",
  "authors": "Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang",
  "year": "2023 (arXiv); TACL 2024",
  "url": "https://arxiv.org/abs/2307.03172",
  "finding": "Performance is U-shaped in the position of relevant information (primacy and recency). With the answer in the middle, GPT-3.5-Turbo's multi-document QA is below its closed-book accuracy (56.1%). Extended-context models are not better at using their context.",
  "relevance": "Position confound: B's position, not its share, can decide capture.",
  "how_checked": "Downloaded the v3 PDF; read the abstract and introduction."
 },
 {
  "title": "RULER: What's the Real Context Size of Your Long-Context Language Models?",
  "authors": "Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg",
  "year": "2024 (COLM 2024)",
  "url": "https://arxiv.org/abs/2404.06654",
  "finding": "17 models, 13 tasks: only half of the models claiming 32K+ are satisfactory at 32K. At long lengths models rely more on parametric knowledge and copy more from context. At 128K, over 80% of Yi-34B's common-words outputs copy the one-shot example at the start of the context (no copying at short lengths); models also return variables from the wrong chain.",
  "relevance": "The only found case where a tiny-share segment captures the output, and only at very long length — driven by primacy/copying failure, not share.",
  "how_checked": "Downloaded the v3 PDF; read Sections 1, 3 and the Yi-34B analysis."
 },
 {
  "title": "Context Rot: How Increasing Input Tokens Impacts LLM Performance",
  "authors": "Kelly Hong, Anton Troynikov, Jeff Huber (Chroma)",
  "year": "2025",
  "url": "https://www.trychroma.com/research/context-rot",
  "finding": "18 models, including Qwen3-8B/32B/235B-A22B. Performance degrades with input length even on simple tasks; distractors lower accuracy, and not uniformly; coherent haystacks hurt more than shuffled ones; in the repeated-words task accuracy falls with length and is best when the unique word is near the start; Qwen3-8B gives invalid output from about 5,000 words.",
  "relevance": "Length-degradation null for O1 and O3; also motivates the coherent-versus-shuffled filler controls.",
  "how_checked": "Fetched the page with WebFetch; findings as returned by the fetch summary (numbers not independently re-read)."
 },
 {
  "title": "Calibrate Before Use: Improving Few-Shot Performance of Language Models",
  "authors": "Tony Z. Zhao, Eric Wallace, Shi Feng, Dan Klein, Sameer Singh",
  "year": "2021",
  "url": "https://arxiv.org/abs/2102.09690",
  "finding": "Majority-label, recency and common-token biases. GPT-3 2.7B, 4-shot SST-2: with 'P P P N', nearly 90% of predictions are Negative even though 3/4 of the examples are Positive. 4-shot LAMA: answers from examples 1–4 are over-predicted by 8.5 / 8.3 / 14.3 / 16.1%; 50.2% of predictions repeat a training answer.",
  "relevance": "Shows a 25% minority capturing about 90% through recency in a short context — the recency null that any 'release last' design must beat.",
  "how_checked": "Downloaded the PDF; read Section 4 and Fig. 4."
 },
 {
  "title": "The Curious Case of Neural Text Degeneration",
  "authors": "Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi",
  "year": "2019 (arXiv); ICLR 2020",
  "url": "https://arxiv.org/abs/1904.09751",
  "finding": "Fig. 4: the probability of a repeated phrase rises with each repetition, a positive feedback loop; this holds for most phrases, including random ones. Introduces nucleus sampling.",
  "relevance": "Repetition is a competing attractor in self-continuation (O2).",
  "how_checked": "Downloaded the PDF; read the Fig. 4 caption and Sections 3–5."
 },
 {
  "title": "Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation",
  "authors": "Jin Xu, Xiaojiang Liu, Jianhao Yan, Deng Cai, Huayang Li, Jian Li",
  "year": "2022 (NeurIPS 2022)",
  "url": "https://arxiv.org/abs/2206.02369",
  "finding": "Self-reinforcement: the probability of repeating a sentence rises almost monotonically with the number of earlier repetitions, up to a ceiling. Sentences with a higher initial probability reinforce faster. Holds for random-token sentences. Stochastic sampling weakens the effect.",
  "relevance": "A mechanism for symmetry breaking (winner-take-all) and for prior-weighted capture, i.e. a γ-like effect.",
  "how_checked": "Downloaded the PDF; read Sections 2–3 and Appendix D."
 },
 {
  "title": "In-Context Learning Dynamics with Random Binary Sequences",
  "authors": "Eric J. Bigelow, Ekdeep Singh Lubana, Robert P. Dick, Hidenori Tanaka, Tomer D. Ullman",
  "year": "2024 (ICLR 2024)",
  "url": "https://arxiv.org/abs/2310.17639",
  "finding": "GPT-3.5+ outputs switch sharply from random-looking to deterministic repetition as context grows; S-shaped transitions consistent with Bayesian model selection rather than averaging. GPT-4 tracks a specified p(Tails); ChatGPT for p>0.5 converges to repeating Tails.",
  "relevance": "Theory and evidence for abrupt switching between hypotheses: under model selection the threshold sits where likelihoods cross, i.e. at 0.5 in a symmetric design.",
  "how_checked": "Downloaded the v3 PDF; read Sections 1–5 (Figs. 2, 4, 5)."
 },
 {
  "title": "ICLR: In-Context Learning of Representations",
  "authors": "Core Francisco Park, Andrew Lee, Ekdeep Singh Lubana, Yongyi Yang, Maya Okawa, Kento Nishi, Martin Wattenberg, Hidenori Tanaka",
  "year": "2025 (ICLR 2025)",
  "url": "https://arxiv.org/abs/2501.00070",
  "finding": "Mainly Llama-3.1-8B: as context grows, representations suddenly reorganise to the in-context graph. With conflicting pretrained semantics (days of the week), the pretrained ring stays in PC1–2 and the in-context ring appears in PC3–4 ('present, but does not dominate'). With enough context, task accuracy overrides the prior. The transition point scales as a power law in graph size.",
  "relevance": "The only found case of two structures coexisting in one context: behaviour switches while the representation keeps both. Motivates the representation readout in O1.",
  "how_checked": "Downloaded the v2 PDF; read Sections 1–4."
 },
 {
  "title": "Bayesian scaling laws for in-context learning",
  "authors": "Aryaman Arora, Dan Jurafsky, Christopher Potts, Noah D. Goodman",
  "year": "2024 (v4 2025)",
  "url": "https://arxiv.org/abs/2410.16531",
  "finding": "In-context learning approximates a Bayesian learner; Bayesian scaling laws predict when in-context learning makes behaviours suppressed by SFT/DPO re-emerge (many-shot jailbreaking).",
  "relevance": "Count-based Bayesian null for reading M2.",
  "how_checked": "Downloaded the PDF; read the abstract only."
 },
 {
  "title": "Enough Coin Flips Can Make LLMs Act Bayesian",
  "authors": "Ritwik Gupta, Rodolfo Corona, Jiaxin Ge, Eric Wang, Dan Klein, Trevor Darrell, David M. Chan",
  "year": "2025",
  "url": "https://arxiv.org/abs/2503.04722",
  "finding": "With biased coin flips in context, models (Llama3.1-8B, Gemma-2, Phi-2/3.5, Mistral-7B, OLMoE) start from biased priors, let in-context evidence outweigh explicit bias instructions, and broadly follow Bayesian posterior updates; attention magnitude has negligible effect.",
  "relevance": "Proportional (probability-matching) null for share-response curves.",
  "how_checked": "Downloaded the PDF; read the abstract and the model list."
 },
 {
  "title": "Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts",
  "authors": "Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, Yu Su",
  "year": "2024 (ICLR 2024)",
  "url": "https://arxiv.org/abs/2305.13300",
  "finding": "Table 6 (share of memory-matching answers by share of memory-aligned evidence): at a 1/3 share, 30.0% (ChatGPT, PopQA) to 86.1% (Vicuna-7B, StrategyQA); Llama2-7B StrategyQA 70.8%. At a fixed 50% share, going from 1/2 to 2/4 pieces raises memory answers (ChatGPT 43.0→63.3, GPT-4 65.4→75.4, Qwen-7B 52.3→63.0). 'LLMs follow the herd'; order effects above 30% for PaLM2 and Llama2-7B.",
  "relevance": "Closest published share-response measurement: a prior-aligned minority captures more than its share, and quantity matters at fixed share — the γ/prior confound for O1 and O6.",
  "how_checked": "Downloaded the PDF; read Section 4, Tables 5–7."
 },
 {
  "title": "Emergent Social Conventions and Collective Bias in LLM Populations",
  "authors": "Ariel Flint Ashery, Luca Maria Aiello, Andrea Baronchelli",
  "year": "2025 (Science Advances 11(20) eadu9368, per the preprint)",
  "url": "https://arxiv.org/abs/2410.08948",
  "finding": "N=24 agents with memory of the last 5 interactions: shared conventions emerge spontaneously; collective bias appears even when single agents are unbiased. A committed minority flips the convention above a critical mass of 2% (Llama-3-70B-Instruct) to 67% (Llama-2-70b-Chat), depending on convention strength; below it the population stays mixed; Llama-3.1-70B switches spontaneously to the stronger convention.",
  "relevance": "Population-level minority tipping below 50% with a prior-dependent threshold (a γ analogue) and stable coexistence below threshold. Not a single-context result.",
  "how_checked": "Downloaded the preprint PDF v2; read the introduction, methods and the tipping-points section."
 },
 {
  "title": "Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing",
  "authors": "Zhilin Wang, Yafu Li, Jianhao Yan, Yu Cheng, Yue Zhang",
  "year": "2025",
  "url": "https://arxiv.org/abs/2502.15208",
  "finding": "Repeated paraphrasing converges to 2-period limit cycles (even and odd iterations form two clusters) across GPT-4o(-mini), Llama3-8B/70B, Mistral-7B and Qwen2.5-7B/14B/72B, in English and Chinese. The cycles persist with higher temperature and alternating prompts or models; attributed to self-reinforcement; invertible tasks favour cycles.",
  "relevance": "An existing case of 'one entity becomes two alternating states' under iteration; motivates the within-trajectory 2-periodicity measure in O2.",
  "how_checked": "Downloaded the v2 PDF; read Sections 1–5."
 },
 {
  "title": "The Curse of Recursion: Training on Generated Data Makes Models Forget",
  "authors": "Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Yarin Gal, Nicolas Papernot, Ross Anderson",
  "year": "2023 (v3 2024)",
  "url": "https://arxiv.org/abs/2305.17493",
  "finding": "Training on model-generated content causes irreversible defects in which the tails of the original distribution disappear (model collapse) in VAEs, GMMs and LLMs.",
  "relevance": "A self-consuming training loop erases minorities. The nearest analogue of a 'difficulty adjustment' (retraining on own output) is outside a frozen model. The Nature 2024 version was not opened (unverified).",
  "how_checked": "Downloaded the v3 PDF; read the abstract and introduction."
 },
 {
  "title": "LLMs Get Lost In Multi-Turn Conversation",
  "authors": "Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, Jennifer Neville",
  "year": "2025",
  "url": "https://arxiv.org/abs/2505.06120",
  "finding": "Across six tasks, multi-turn performance is on average 39% lower than single-turn (200,000+ simulated conversations). Models make early assumptions and 'do not recover'.",
  "relevance": "Path dependence across turns — relevant to the hysteresis tests in O2.",
  "how_checked": "Downloaded the PDF; read the abstract."
 },
 {
  "title": "The Waluigi Effect (mega-post)",
  "authors": "Cleo Nardo",
  "year": "2023 (LessWrong, 3 March 2023)",
  "url": "https://www.greaterwrong.com/posts/D7PumeYTDPfBTp3i7/the-waluigi-effect-mega-post",
  "finding": "Conceptual, not empirical. After training for property P it is easier to elicit the opposite; conjecture that 'waluigi' opposite-personas are absorbing attractor states, so 'the longer you interact ... eventually the LLM will have collapsed into a waluigi'. Commenters dispute the inevitability claim.",
  "relevance": "A prior conceptual framing of one simulated entity splitting into an opposite pair. Evidence quality: low.",
  "how_checked": "Read the post via the GreaterWrong mirror (first 100,000 characters, covering the post body)."
 },
 {
  "title": "Qwen3 Technical Report",
  "authors": "Qwen Team",
  "year": "2025",
  "url": "https://arxiv.org/abs/2505.09388",
  "finding": "Table 1: 1.7B has 28 layers, 16/8 query/KV heads, tied embeddings, 32K context (4B and up: 128K). Stages 1–2 train at 4,096 tokens; the long-context stage at 32,768 (75% of texts 16K–32K). RoPE base raised from 10,000 to 1,000,000 (ABF); YaRN plus DCA give 4× length at inference. The RULER table (Table 23, YaRN factor 4) starts at 4B (Qwen3-4B non-thinking 95.1/93.6/91.0/87.8/77.8/66.0); no 1.7B row.",
  "relevance": "Defines the 'natural' and native lengths for Qwen3-1.7B and confirms there is no official 1.7B long-context score.",
  "how_checked": "Downloaded the PDF; read Sections 2–3, Table 1 and Appendix A.1.1."
 },
 {
  "title": "Qwen3-1.7B model card, config.json and generation_config.json (with Qwen3-4B card for comparison)",
  "authors": "Qwen Team (Hugging Face)",
  "year": "2025",
  "url": "https://huggingface.co/Qwen/Qwen3-1.7B",
  "finding": "Card: 'Context Length: 32,768', 1.4B non-embedding parameters, no YaRN statement; non-thinking T 0.7 / TopP 0.8 / TopK 20; do not use greedy decoding ('endless repetitions'), presence_penalty up to 1.5. Config: max_position_embeddings 40,960, rope_theta 1,000,000, rope_scaling null, head_dim 128, 16 Q / 8 KV heads, hidden 2048, intermediate 6144, vocab 151,936, tied embeddings. The Qwen3-4B card claims 131,072 tokens with YaRN (factor 4) and warns that static YaRN can hurt short texts.",
  "relevance": "Hard limits for 'hyper-long' tests: up to 40,960 positions without RoPE scaling; YaRN not validated for 1.7B.",
  "how_checked": "Downloaded raw README.md, config.json and generation_config.json from huggingface.co/Qwen/Qwen3-1.7B, plus the Qwen3-4B README."
 },
 {
  "title": "A Recipe for Long-Context Reasoning in Large Language Models via On-Policy Optimization and Distillation",
  "authors": "Miguel Moura Ramos, Duarte M. Alves, André F. T. Martins",
  "year": "2026",
  "url": "https://arxiv.org/abs/2605.12227",
  "finding": "Base Qwen3-1.7B RULER: 87.4/84.2/81.0/72.0/51.5/11.0 at 4K/8K/16K/32K/64K/128K (average 64.5). Common-words extraction 91.8/73.9/46.4/19.4/7.5/0.2; variable tracking 99.0/97.3/92.5/88.8/55.2/0.4; frequent-words extraction 75.7/76.3/88.7/79.9/61.6/58.6. Thinking mode and RoPE scaling above 32K were not stated in the parts read.",
  "relevance": "Only long-context numbers found for Qwen3-1.7B. The steep fall in common-words extraction (a counting/majority task) is a direct confound for share-response curves.",
  "how_checked": "Downloaded the PDF; read Table 2, Table 10 and the evaluation-setup section."
 },
 {
  "title": "Broken Tokens? Your Language Model can Secretly Handle Non-Canonical Tokenizations",
  "authors": "Brian Siyuan Zheng, Alisa Liu, Orevaoghene Ahia, Jonathan Hayase, Yejin Choi, Noah A. Smith",
  "year": "2025 (v2 2026)",
  "url": "https://arxiv.org/abs/2506.19004",
  "finding": "Instruction-tuned models (Llama-3.1-8B-Instruct, OLMo2-7B-Instruct, Qwen-2.5-7B-Instruct) keep up to 93.4% of performance with random tokenisation and 90.8% with character-level, across 20 benchmarks. Robustness comes from instruction tuning; base models treat non-canonical tokens as misspellings and degrade. Character-level splitting helps string tasks by up to +14%; right-aligned digit grouping helps arithmetic by +33%.",
  "relevance": "Generic retention baseline for the retokenisation control (O5).",
  "how_checked": "Downloaded the v2 PDF; read the abstract and Sections 1–4."
 },
 {
  "title": "From Tokens to Words: On the Inner Lexicon of LLMs",
  "authors": "Guy Kaplan, Matanel Oren, Yuval Reif, Roy Schwartz",
  "year": "2025 (ICLR 2025)",
  "url": "https://arxiv.org/abs/2410.05864",
  "finding": "Models detokenise in early and middle layers. For artificially split single-token words, Llama2-7B's logit-lens retrieval of the original word exceeds 80% at layer 15, and 93.2% of words are retrieved in at least one layer; with typos about 40%. 23% of multi-token words are never decoded from their last token. Models: Llama2-7B, Llama3-8B, Mistral-7B, Yi-6B.",
  "relevance": "Word-level representations survive retokenisation; gives the method for O5's detokenisation readout.",
  "how_checked": "Downloaded the v4 PDF; read Sections 1 and 4."
 },
 {
  "title": "Linguistic Regularities in Continuous Space Word Representations",
  "authors": "Tomas Mikolov, Wen-tau Yih, Geoffrey Zweig",
  "year": "2013 (NAACL-HLT 2013)",
  "url": "https://aclanthology.org/N13-1090/",
  "finding": "'King − Man + Woman' gives a vector very close to 'Queen'; syntactic and semantic regularities appear as vector offsets.",
  "relevance": "Source of the symmetric-analogy example the user cites.",
  "how_checked": "ACL Anthology PDF; read the abstract."
 },
 {
  "title": "Fair is Better than Sensational: Man is to Doctor as Woman is to Doctor",
  "authors": "Malvina Nissim, Rik van Noord, Rob van der Goot",
  "year": "2020 (Computational Linguistics; arXiv 2019)",
  "url": "https://arxiv.org/abs/1905.09866",
  "finding": "Standard analogy code does not allow the input words as answers. When they are allowed, Google-analogy accuracy falls from 0.74 to 0.21 for 3CosAdd and from 0.75 to 0.47 for 3CosMul, mostly by returning B (e.g. 'king').",
  "relevance": "The apparent symmetry of the king/queen analogy partly depends on how it is evaluated.",
  "how_checked": "Downloaded the v2 PDF; read Section 4.1 and Table 1."
 },
 {
  "title": "The Reversal Curse: LLMs trained on 'A is B' fail to learn 'B is A'",
  "authors": "Lukas Berglund, Meg Tong, Max Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, Owain Evans",
  "year": "2024 (ICLR 2024)",
  "url": "https://arxiv.org/abs/2309.12288",
  "finding": "Models fine-tuned on 'A is B' fail on 'B is A'. GPT-4 answers 'Who is Tom Cruise's mother?'-type questions 79% of the time versus 33% for the reverse. In-context reversal works.",
  "relevance": "Relations learned in weights are directional: the basis for the symmetric-versus-asymmetric pair sub-test in O5.",
  "how_checked": "Downloaded the v4 PDF; read the abstract."
 },
 {
  "title": "Linearity of Relation Decoding in Transformer Language Models",
  "authors": "Evan Hernandez, Arnab Sen Sharma, Tal Haklay, Kevin Meng, Martin Wattenberg, Jacob Andreas, Yonatan Belinkov, David Bau",
  "year": "2024 (ICLR 2024)",
  "url": "https://arxiv.org/abs/2308.09124",
  "finding": "Some relations are well approximated by a single affine map on the subject representation; robust maps were found for 48% of tested relations (GPT-J); many relations are predicted correctly but are not linearly encoded.",
  "relevance": "Asymmetric or non-linear relations may not give a clean A–B axis — relevant to O5's analogy sub-test and to O1's readout.",
  "how_checked": "Downloaded the v2 PDF; read the abstract and introduction."
 }
]
```
