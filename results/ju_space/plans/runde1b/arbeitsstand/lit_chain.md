# lit:chain

## Verifier summary
All 41 references exist, and their titles, authors, years and URLs match the primary pages (arXiv API and full texts, LessWrong, transformer-circuits.pub, NeurIPS, Hugging Face, PyPI). Nearly every number checked against the full texts is right: H-neurons Table 1, the Cavus diagnostics, Future Lens Table 2, the Wu et al. losses, DeepSeek MTP, the J-lens definition and swap rates, the chain-lens scores and quote, the Neuronpedia config, and the Qwen3-1.7B and Qwen3.5-2B configs (311 tensors, no MTP; MTP present). The round-2 facts and the A3 assertion at r2_rgi_geom.py:695 also match. Three reference entries need correction: the biology paper's 70 % covers two injected features ('rabbit' and 'green'), not 'green' alone; Elhage's linearity quote applies to attention-only models; and J++ and MicroLens are now verified, with the J++ URL added. Feucht and Ferrando–Voita are confirmed as EMNLP 2024. The substantive problems are in the reasoning. O1's observational throughput TP_c cannot separate the KV route from the token route, and its null prediction is wrong. O5's 'attractor' criteria accept a decaying carry (K_H > 0.55 < 1), combine disjunctive criteria, and allow confounded bimodality; O5(b) cannot separate an attractor from snowballing. 'Any same-channel asymmetry = length drift' ignores event-locked non-stationarity and computational asymmetry. O3's J-lens sanity check cannot work with lags 0-15 only, and two of its predictions do not discriminate. O4's answer-versus-prompt score-level prediction is built in by the labelling. O2's null misreads Future Lens's mid-layer peak. The A3 zero is structural only at block 27. Position gaps without tokens are out of distribution. The round-2 links mix loop rounds with token lags. O6 is under-costed (≈2.5-3 h, not 1-1.5) and O1 storage is 2× too low. Minor issues: Granger = 2 × TE, not equal; the MTP-training inference is a non sequitur, though the Qwen3 report indeed never mentions MTP; 'demonstrated attractor' overstates round 2; the 'README item' pointers belong to AUSWERTUNG.md; the context length is 32,768. The report contains no selfish-mining formulas, so none were checked. The corrected report applies these fixes and lists the field-level corrections in a new section 8.

## Problems found by the verifier
- O1 latent throughput does not test latent carry. TP_c(k) = CV-R²[F_c(t+k) | F_c(t), x_{t+1..t+k}] − CV-R²[F_c(t+k) | x_{t+1..t+k}]. F_c(t) is a deterministic function of x_{≤t}, and x_{≤t} reaches F_c(t+k) through attention, so TP_c(k) > 0 is expected under the null as well (any channel carrying topic or entity information shows it). The null prediction 'TP_c(k) ≈ 0 for k ≥ 2' is therefore wrong. Conditioning on all of x_{≤t+k} makes TP identically 0. Only the KV-only versus token-only interventions separate the carriers; TP_c can only be descriptive.
- O5(c) attractor criteria do not establish an attractor. The criteria are: K_H above the random 95th percentile with lower bound > 0.55; or a convex dose response; or a bimodal late-session H distribution. A carry below 1 decays geometrically. Round 2's own rule R6 calls K < 1 'dissipative' and requires a lower bound > 1 for self-sustainment; 0.55 only exceeds the style carry's upper bound (0.5494). Convexity can come from out-of-distribution damage at large α (Gao et al. report non-monotonic responses). Bimodality across a mixed question set can come from mixing known and unknown items, so bistability must be shown within items (same question, different seeds). Accepting any one of three criteria also inflates the false-positive rate.
- O5(b) does not separate the alternatives it is meant to. Content at matched positions separates content from length, but not an H-neuron attractor from ordinary in-context conditioning. Snowballing (Zhang et al.) and in-prompt induction (round 2: three examples restore 89 %) predict the same rise in H after the model's own hallucinated answers or injected wrong claims. Only (c), with a K_H > 1 or within-item bistability criterion, bears on 'attractor'.
- Same-channel asymmetry is overclaimed (§0.4, §3.2, O2, where_analogy_breaks). The symmetries I(F_t;F_{t+k}) = I(F_t;F_{t−k}) and, for whitened single-lag linear probes, tr C(k)ᵀC(k) are correct. But 'any same-channel asymmetry measures drift ... exactly the position and length drift H-CHAIN must exclude' overreaches. (i) Non-stationarity also includes event-locked change such as a hallucination onset, which is the signal H-CHAIN posits. (ii) Capacity-limited nonlinear probes can be asymmetric on stationary data for computational reasons, as Papadopoulos et al. (cited in the report's own ref) show.
- O3 sanity check cannot work as specified. 'The lag-weighted sum must reproduce the published Neuronpedia J-lens' cannot hold for Δ = 0..15 estimated from 100 prompts × 3 output positions. The published lens averages all pairs t' ≥ t of 128-token WikiText contexts, so lags 16-127 carry about 62-77 % of the weight (77 % of (t, t') pairs: 1 − 1928/8256). A valid check needs all lags on the same prompts, the fit's weighting, target layer and aggregation (the paper's default lens differentiates the penultimate layer), and more than 300 samples per lag.
- O3 predictions do not discriminate. Prediction (2), that agreement, LH and FOV_H coincide within ±1 token, is also expected under the null, because a common driver (falling predictability or uncertainty) moves all three. Prediction (4), partial r ≥ 0.15 between reach r(t) and the H-score, has no sign; under H-CHAIN, reach should fall as H rises. FOV_H rising above random sets is also expected if H-neurons encode uncertainty. Matching on model entropy does not control token rarity, and lenses fit on frequent text read rare entity tokens worse.
- O4 prediction is built in by the labels. 'The H-score is higher on answer tokens than at the last prompt token' follows from Gao's labelling, in which non-answer token vectors are negatives. The informative test is discrimination: does the answer-token score add AUROC beyond the last-prompt-token score? The O4 null already names that comparison.
- O2 null misreads Future Lens. 'Memory falls and anticipation rises across layers, as Feucht et al. and Future Lens report' is wrong for k ≥ 2: Future Lens finds that t+2..t+4 accuracy peaks at a middle layer (14 of 28), and only the immediate next token peaks at the last layer, as the report's own §2.2 says. The null 'a_c(k) decays with k in proportion to A*(k)' also mixes raw and normalised quantities: if A_c(k) ∝ A*(k), the normalised a_c(k) is flat in k.
- A3 is structural only at block 27 (§0.6, §3.3, O3 controls). The exact zero at the output of block 27 is structural. The near-vanishing at blocks 22-26 is an empirical result for the A3 readout (projection ≈1e-5, AUSWERTUNG.md): with 1-5 attention layers left, a cross-position Jacobian can in principle be large. 'Small at blocks 22-26, whatever the lag', 'any Jacobian chain lens should show the same profile' and '||J_l(Δ≥1)|| must fall to 0 as l → 27' therefore overgeneralise. The block-27 zero is a sanity check, not a positive control.
- O5(a) position gaps are out of distribution. Gaps implemented as jumps in position_ids with no tokens are not part of standard training, so a gap effect may reflect the novel configuration (no keys at short distances) rather than distance. PoSE shows that skipped positions work after training with skips; it does not show that a model trained without them treats gaps as distance.
- Round-2 links mix time scales and overstate matches. 'RSI returning 2-5 % per step ... predicts k* ≈ 1-2 for RSI words' mixes time scales: g_par is the return per loop round (a 40-token generation plus a mean-state read-out), not per token. 'README item 5 ... is exactly the KV-route vs token-route split' overstates: the latent loop re-injects a state as a push, while the KV route is the model attending to perturbed cached states, so the two are analogous. Deriving edge reordering from the block-6 temperature slope of the loop gain G is an analogy, not a prediction.
- O6 cost is underestimated. The ≈0.75 h of JVPs covers 4 conditions of 512-token contexts. The design varies 4 prompt types, 4 temperatures, and contexts none/1k/4k/8k × coherent/filler, which is at least 13 cells even varied one factor at a time, and longer contexts cost more per JVP. At the report's own per-cell rate (≈0.19 h), O6 needs ≈2.5-3 A100-h, so all six take ≈11.5-13.5 h rather than ≈10-13 h. The minimal path is unaffected.
- O1 storage arithmetic is off by 2×. 448 heads × 16 dims plus 6 × 2048 full-state values make 19,456 values per token. That is ≈20 KB only at 1 byte per value; in bf16 it is ≈39 KB/token, or ≈47 GB for 1.2M tokens, above the stated ≤25 GB. The replay fallback still works.
- O2 probe type is inconsistent. The measurement specifies linear-softmax probes; the cost line specifies closed-form ridge into the 2048-d final-state space read through W_U. For memory targets, a ridge onto a final state would recover the model's earlier prediction of x_{t−k}, not the token. Regress onto the tied token embedding instead, or cost the softmax probes.
- MTP training inference is a non sequitur. where_analogy_breaks argues 'Qwen3-1.7B was not trained with MTP: it has no MTP weights', but MTP modules can be discarded after training (DeepSeek-V3). The conclusion does have support: the Qwen3 Technical Report (arXiv 2505.09388) contains no mention of multi-token prediction (checked).
- Round 2 did not demonstrate an attractor. where_analogy_breaks says 'round 2 found the only demonstrated attractor to be a generic style direction'. 'Generic attractor' (R4) is a descriptive label: no loop-dynamics verdicts were licensed (PC_PASS failed), and every measured gain was below 1.
- Chain-lens hallucination cause is not from the post. where_analogy_breaks attributes the growth of chain-lens hallucination 'partly' to the auxiliary head rolling the chain out, but the post says no such thing. It says long chains trade more hallucination 'under the current summarizer' and suspects better judges could help. The ellipsis in the report's §2.2 quote drops that qualifier.
- Biology misreport. Injecting 'green' does not make 70 % of 25 poems end with it. Two planned-word features ('rabbit' and 'green') were injected into a random sample of 25 poems, and lines ended with the injected word in 70 % of cases.
- Elhage scope. 'Paths are linear if one freezes the attention patterns' is stated for attention-only models. Qwen3's MLPs and RMSNorm keep paths nonlinear even with frozen attention.
- Granger and transfer entropy differ by a factor of 2. 'Granger causality equals Gaussian transfer entropy' is wrong: Barnett, Barrett & Seth (2009, PRL 103:238701, arXiv 0910.4514) show F_{Y→X|Z} = 2 T_{Y→X|Z}.
- §1 understates what a frozen model allows. 'In a frozen model only the forward sensitivity can be measured' is too strong: the off-diagonal gradient terms themselves (gradients of later-position losses through the computation at position t) can be computed by backpropagation on a frozen model without updating it.
- §0.5 overgeneralises. 'Length effects ... largely driven by content' is contradicted by Levy et al., cited in the same list, who show that irrelevant padding alone degrades reasoning. Several documented effects are content-driven (FActScore order, snowballing, repetition), but not all.
- Internal pointers point to the wrong file. 'README item 2/4/5' in the operationalisations correspond to AUSWERTUNG.md §5 items 2 (damage-normalised push), 4 (lens read-out to replace A3, checked against sign-true shifts) and 5 (latent vs text loop; K with an interval for its denominator). The runde1b README sections are numbered differently.
- Unverifiable statements. 'No paper studies H-neurons over long generations or as a function of position' is an absence claim and should read 'I found no paper'. 'Consistently-incorrect TriviaQA items are largely unknown entities' is a plausible inference, not a checked fact.
- Context length. The confounds call max_position_embeddings 40,960 the 'trained maximum', but the Qwen3-1.7B model card states a context length of 32,768 (the Base config has 32768). The planned contexts (≤ ~9.5k) stay far below either.
- H-neuron details in §2.1. 'TriviaQA train': the paper does not state the split. 'z = SiLU(W_gate x) ⊙ W_up x': the paper writes a generic activation σ. 'AUROC values appear only in figures': the text also says the Mistral family exceeds 86 % on TriviaQA.

## Per-reference / per-fact verdicts
- [verified] H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs: arXiv 2512.01797 v1 1 Dec / v2 2 Dec 2025; authors and Tsinghua affiliation correct. Checked against the HTML v2: Table 1 ratios 0.01-0.35‰; TriviaQA 70.1-83.6 vs random 56.1-68.4; Llama-3.1-8B NonExist 43.1 vs 50.6; CETT formula; prose versus equation label-sign inconsistency (and the opening of §6.1); C chosen by grid search; α∈[0,3]; slopes ≈3.03 and ≈2.40; ranks ≈0.97 and >0.58 (P<0.001); no layer distribution; no code link. Minor points for the report text: the TriviaQA split is not stated (the report says 'train'). The paper writes a generic gate activation σ, not SiLU. Besides Figure 4, the text gives one base-model number: the Mistral family exceeds 86 % on TriviaQA.
- [verified] Hallucination Neurons and Where to Find Them: An Investigation into the existence of Hallucination Neurons: arXiv 2609.29781 v1 (24 Sep 2026), authors correct. Confirmed: D = 348,160; C = 1.0 with liblinear; 2-9 H-neurons (0.006-0.026‰); Gemma AUROC 0.811 / 0.974 / 0.628; D1 19/22; D2 0.47 / 0.69 / 0.45; D3 0.10 / 0.25 / 0.00; TQA∩BioASQ 4, BioASQ∩NQ 0; 0.472 / 0.496 / 0.490 against a random same-layer set at 0.482 (McNemar p<0.001); capability preservation not tested.
- [verified] hprobes (PyPI package): Hallucination neuron discovery and causal validation for transformer LLMs: PyPI JSON confirms: MIT licence, author Huseyin Cavus (GOSH DRIVE contributors), builds on arXiv 2512.01797, default l1_C 0.01, 'contrastive 3-vs-1 labeling at the generated answer token', dense and MoE support, α-scaling and bootstrap. First release 2026-03-23, current 0.12.0. The GitHub repository was not opened.
- [verified] Confidence Regulation Neurons in Language Models: Main text gives ≈80 % (GPT-2, LLaMA2-7B, Phi-2) against ≈40 % (Pythia, Gemma). The appendix gives 30-40 % for Pythia and Gemma.
- [verified] Inference-Time Intervention: Eliciting Truthful Answers from a Language Model: 
- [verified] Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs: TBG and SLT positions and the quote 'SEPs can capture semantic entropy even before generation' confirmed. Llama-3-70B is also used, for long generations.
- [verified] INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection: 
- [verified] Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation: Confirmed that h_l is the average over output tokens at layer l, with magnitude and angle features between adjacent layers. Accepted at ICLR 2025.
- [verified] Real-Time Detection of Hallucinated Entities in Long-Form Generation: 
- [verified] Future Lens: Anticipating Subsequent Tokens from a Single Hidden State: Table 2 numbers, bigram 20.1 %, the layer-14 peak (last layer for N=0), selection of correct-prediction positions, greedy targets and calibrated accuracy 26/57/77/95 % all confirmed from the PDF.
- [verified] Do Language Models Plan Ahead for Future Tokens?: Vanilla 3.28, myopic 3.40, local myopic 3.26, bigram 5.33 (GPT-2 124M, MS MARCO, length 64). Pythia models were fine-tuned from checkpoints, and the gap grows 'starting from the 70M model'.
- [verified] Better & Faster Large Language Models via Multi-token Prediction: 
- [verified] DeepSeek-V3 Technical Report (Section 2.2 Multi-Token Prediction): Confirmed: M_k[RMSNorm(h^{k-1}); RMSNorm(Emb(t_{i+k}))] with M_k ∈ R^{d×2d}; 'keep the complete causal chain at each prediction depth'; D = 1; λ 0.3 for 10T tokens then 0.1 for 4.8T; modules can be discarded or used for speculative decoding.
- [corrected] On the Biology of a Large Language Model: The 70 % figure covers TWO injected planned-word features ('rabbit' and 'green') in a random sample of 25 poems; the model ended its line with the injected word in 70 % of cases. It is not '“green” makes 70 % of 25 poems end with it'. Everything else is confirmed: Jack Lindsey et al., 27 Mar 2025, Claude 3.5 Haiku, planning at the newline, misfires of the known-answer/can't-answer circuit, frozen attention patterns in the replacement model.
- [verified] Eliciting Latent Predictions from Transformers with the Tuned Lens: 
- [verified] Verbalizable Representations Form a Global Workspace in Language Models (Jacobian lens): Confirmed: J_l definition over t, t'≥t, 1,000 prompts; lens(h) = softmax(W_U norm(J_l h)); swap success 54 / 70 / 70 %; fight→light changes coming→morning; code link github.com/anthropics/jacobian-lens. Note for the O3 sanity check: the methods appendix says the default Sonnet 4.5 lens differentiates the penultimate-layer state, and it also describes self-only, future-only and frozen-QK variants.
- [verified] neuronpedia/jacobian-lens (Hugging Face model repo): Qwen3-1.7B J-lens: config.yaml confirmed: wikitext-103-raw-v1, n_prompts 1000, prompts_fitted 466, max_seq_len 128, dim_batch 128, bf16, B200, stop_at_delta 0.002, generated 2026-06-11, target_layer null (code default). J++ ('jpp') lenses exist for qwen3-4b, qwen3.5-0.8b/2b/4b/9b/122b-a10b and qwen3.6-27b, but not for qwen3-1.7b.
- [verified] The Chain Lens [interim]: All numbers and the quote are confirmed (0.420 vs 0.319 over 28 families; 0.487 vs 0.411 over 21; chain intermediates 0.42 vs 0.25; ~100x cost for 10 five-token chains; round-trip criteria 99.8th percentile and >90 %). Two qualifiers the report drops: long chains trade more hallucination 'under the current summarizer' (the authors suspect stronger judges could help), and the post reports LESS confabulation than the oracle lens on introspection and poetry lookahead.
- [verified] WorkspaceBench: Evaluating Interpretability Methods for the Global Workspace: 3,356 questions, 27 families, Qwen3.6-27B, 23 Sep 2026. The hallucination judge sees the model's response; the headline is the fraction of readouts with at least one wrong claim.
- [verified] Gradient Routing: Masking Gradients to Localize Computation in Neural Networks: 
- [verified] Arrows of Time for Large Language Models: Confirmed: English FW 2.88 vs BW 2.902 (+0.76 %), French +2.65 %, other languages up to +3.17 % (XL); Nano −0.33 %; the gap grows with context length; prime factoring 22.2 vs 30.2 nats; the information-theoretic quote.
- [verified] Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs: EMNLP 2024 confirmed in the ACL Anthology (2024.emnlp-main.543). Offsets −3..+1, 428k Pile tokens, Llama-3-8b replication confirmed.
- [verified] FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation: 
- [verified] Long-form factuality in large language models: 
- [verified] How Language Model Hallucinations Can Snowball: 
- [verified] Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation: 
- [verified] Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models: 
- [verified] Efficient Streaming Language Models with Attention Sinks: 
- [verified] Massive Activations in Large Language Models: 
- [verified] Lost in the Middle: How Language Models Use Long Contexts: 
- [verified] RoFormer: Enhanced Transformer with Rotary Position Embedding: 
- [verified] PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wise Training: 
- [corrected] A Mathematical Framework for Transformer Circuits: Both quotes are verbatim, but the second has a scope the report omits. The text says 'Attention-only models can be written as a sum of ... paths ... and are linear if one freezes the attention patterns.' The linearity claim applies to attention-only models. With MLPs and normalisation, as in Qwen3, paths are not linear even when attention patterns are frozen. Published 22 Dec 2021.
- [verified] Information Flow Routes: Automatically Interpreting Language Models at Scale: EMNLP 2024 confirmed in the ACL Anthology (2024.emnlp-main.965).
- [verified] Localizing Model Behavior with Path Patching: 
- [verified] Transfer Entropy as a Measure of Information Flow in VLMs and LLMs: 
- [verified] Discovering Millions of Interpretable Features with Sparse Autoencoders (Qwen3-Instruct SAE): The abstract confirms layer-wise SAEs at residual, MLP-output and attention-output sites for Qwen3-1.7B and Qwen3-4B, and calls the suite a 'release'. The release location remains unverified.
- [verified] Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models: 
- [verified] Qwen/Qwen3-1.7B: config.json and model.safetensors.index.json: Confirmed: 311 tensors, none MTP; 28 layers; hidden 2048; intermediate 6144; 16 query / 8 KV heads; head_dim 128; q_norm and k_norm; rope_theta 1e6; tied; SiLU; max_position_embeddings 40960. The model card states a context length of 32,768 (Qwen3-1.7B-Base config: 32768). Qwen3-1.7B-Base exists.
- [verified] Qwen/Qwen3.5-2B: config.json and model.safetensors.index.json: MTP tensors confirmed (mtp.fc, pre_fc_norm_embedding/hidden, mtp.layers.0 with self_attn and MLP, mtp_num_hidden_layers 1), along with text hidden 2048, 24 layers and a 3:1 linear/full pattern. The checkpoint is a vision-language model (Qwen3_5ForConditionalGeneration).
- [corrected] MicroLens (Zahedi, Zhu, Wu) and J++ lens (Ayonrinde & Lindsey, LessWrong 2026): Both sources were re-opened and match the README summary, so they no longer need to be marked unverified. MicroLens: Nika Zahedi, Fangyi Zhu, Anna Wu, 'MicroLens: Read, Trace, and Rewire the Residual Stream with Two Lenses', arXiv 2611.13322 (9 Nov 2026). The abstract supports 'readable is not used': the state or country is readable at the city token, but at the answer position it is not used. J++: Kola Ayonrinde (with Jack Lindsey), 'J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses', LessWrong, 8 Oct 2026, https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr. It uses per-cluster Jacobians with poor clusters down-weighted, an LRP backward pass and a filter for non-semantic tokens, and reports 55.2 % recall@10 against 35.7 % for the J-lens. The ref's URL covers only MicroLens; add the J++ URL.

## Corrected report
# H-CHAIN: literature and operationalisation check (round 1b)

Scope: hypothesis H-CHAIN from the shared brief (a chain lens over channel fronts; an H-neuron attractor versus a length artifact), for Qwen3-1.7B on one Colab A100-40GB. I opened and read everything cited here unless it is marked unverified. The `refs` field names the page I read for each source. Local copies of the main texts are in /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/hchain/:
- dl_hneurons/page.txt
- dl_hneur_repl/page.txt
- dl_chainlens/page.txt
- dl_workspace/page.txt
- dl_futurelens/fl2.txt
- dl_factscore/fs.txt
- dl_npjl/qwen3_17b_config.yaml
- dl_qwencfg/
- dl_qwen35/

Verifier note (13 Nov 2026): every reference was re-opened at its primary page. Independent copies are in /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/vchain/. Passages changed after verification are marked [corrected]. Corrections to the structured fields are listed in section 8.

## 0 Bottom line

1. **"Chain lens" is an existing term, and it is two days old.**
   - Source: vgel and Raymond Douglas, "The Chain Lens [interim]", LessWrong, 11 Nov 2026.
   - Method: the top-K tokens of the J-lens (Gurnee et al. 2026) serve as seeds. Each seed is rolled out with the model's own pretrained multi-token-prediction (MTP) head. Decoding is contrastive against a corpus-mean hidden state, and the head attends only to its own chain.
   - Results on WorkspaceBench: macro score 0.42 against 0.32 for the J-lens (28 families), and 0.49 against 0.41 for a hindsight composite of untrained lenses (21 families).
   - On hallucination the post says: "The chain lens confabulates more on WorkspaceBench’s hallucination family than single-token lenses. This grows with chain length."
   - It defines **no break-off criterion**.
   - The user's "Kettenlinse ... wo sie halluziniert" very probably refers to this post.
   - Qwen3-1.7B has **no MTP head**: none of its 311 weight tensors belong to one (checked). The published chain lens therefore needs a trained head before it can run on this model.
   - Qwen3.5-2B ships a one-layer MTP module (checked). Neuronpedia publishes a J-lens for Qwen3-1.7B (checked).
2. **H-neurons**: Gao, Chen, Xiao, Chen, Liu and Sun (Tsinghua), arXiv 2512.01797, Dec 2025.
   - Method: a sparse L1-logistic probe on per-neuron CETT contributions averaged over answer tokens, trained on TriviaQA questions filtered for consistent answers.
   - "Less than 0.1 %" is a loose bound. Table 1 gives 0.01-0.35 ‰ (0.001-0.035 %). For Qwen3-1.7B that is 2-60 of its 172,032 FFN neurons.
   - Causal scaling of these neurons (activation × α) changes **over-compliance** (false premises, misleading context, sycophancy, jailbreaks), not hallucination alone.
   - A Sept 2026 replication (Cavus et al., arXiv 2609.29781) confirms detection but finds the selection **not uniquely localized**:
     - 19 of 22 selected neurons have a partner with |r| > 0.7;
     - bootstrap Jaccard is 0.45-0.69;
     - L1 and L2 selections overlap only 0-25 %;
     - the causal effects are small: judge accuracy falls 2.4 pp under suppression.
   - [corrected] I found no paper that studies H-neurons over long generations or as a function of position (an absence claim; unverified).
3. **Single hidden states do carry future-token information, but only partly.**
   - Future Lens (GPT-J-6B): from one hidden state, the tokens at t+2, t+3 and t+4 are recovered with precision@1 of 48.4, 43.7 and 46.9 % (learned-prompt method). The bigram baseline is 20.1 % at t+2 and linear maps reach 15-29 %. The signal peaks at a middle layer.
   - Wu et al. (COLM 2024): in GPT-2, most usefulness for future tokens is incidental ("breadcrumbs"; myopia gap ≈0.12 nats on a loss of 3.28). Deliberate pre-caching grows with model scale.
4. **The user's ratio of predicting earlier versus later fronts needs a correction.**
   - For one channel on a stationary stretch, I(F_t;F_{t+k}) = I(F_t;F_{t-k}).
   - Single-lag linear predictability in whitened coordinates is also symmetric.
   - [corrected] So a same-channel asymmetry in these two measures indicates **non-stationarity**. That includes the position and length drift H-CHAIN must exclude. It also includes event-locked change, such as the onset of a hallucination, which may be the very signal H-CHAIN looks for; position-detrending and event-locked analysis are needed to tell the two apart. Capacity-limited nonlinear probes can also be asymmetric on stationary data for computational reasons (Papadopoulos et al. 2024 show this for forward versus backward language models).
   - Two directional quantities survive. Token memory and token anticipation differ in kind because of the causal mask. Cross-channel lead-lag is also directional.
5. **[corrected] Length effects are real; several documented ones are driven by content, but length alone also matters.**
   - FActScore: later facts are less accurate, which the authors attribute to content order (frequent facts come first) and to error propagation.
   - Snowballing and self-reinforcing repetition are documented.
   - Padding with irrelevant text degrades reasoning, a length effect without relevant content.
   - Under RoPE, shifting every position id by the same amount changes nothing, up to rounding. Only relative gaps, the number of keys and the content can matter.
6. **[corrected] The zero of the round-2 A3 readout at block 27 is structural; the near-zero at blocks 22-26 is empirical.**
   - A state written at block l reaches other positions only through attention in blocks > l.
   - At the output of block 27 the cross-position gradient is identically zero. The round-2 code asserts this itself at analysis/ju_space/round2/r2_rgi_geom.py:695.
   - At blocks 22-26 only 1-5 attention layers remain. Round 2 measured the A3 projection there at about 1e-5 (AUSWERTUNG.md). That is a finding for this readout, not a structural necessity.
   - Any lag-resolved (Δ ≥ 1) Jacobian lens must be exactly zero at the output of block 27, which makes that a structural sanity check. Whether it is also small at blocks 22-26 has to be measured.
7. [corrected] Six operationalisations follow. As costed, all six need about 10-13 A100-hours; with O6's full condition set (section 6) about 11.5-13.5 h. A decisive minimal subset (O4, O3 with L-par and L-jac, O5 a-c) needs about 6-7 h.

## 1 Mapping the user's words

- **"Kettenlinse / Chain Lens"**: the existing method in item 0.1.
  - Search: plain searches for "chain lens" found nothing. An extended search found the LessWrong post. I found no arXiv version.
  - Closest prior work: Future Lens.
  - Name clash: Chain-of-Embedding (Wang et al. 2024) is a chain over *layers* of token-averaged output states, not over positions.
- **"Haarneuronen, ~0.1 %"**: H-neurons (item 0.2).
- **"Multitoken Prediction lebt von ... Gradient Through Routing ... Folge, nicht Reihe"**: the message supports three readings.
  - (a) DeepSeek-V3's *sequential* MTP modules. They "keep the complete causal chain at each prediction depth": each depth combines the previous depth's representation with the embedding of the next token. The contrast is Gloeckle et al.'s *parallel* independent heads. This matches "Folge" versus "parallel" best.
  - (b) Gradients through attention across positions in ordinary next-token training. These are Wu et al.'s off-diagonal terms, which are what make pre-caching possible.
  - (c) Gradient routing (Cloud et al. 2024), an unrelated technique that masks gradients to localize capabilities.
  - [corrected] In a frozen model one can measure the forward sensitivity, the cross-position Jacobian, which is the inference-time counterpart of (b). The off-diagonal gradient terms themselves can also be computed on the frozen model by backpropagation, without updating it.
- **"vorderste Front ... Reihen von Tokenfronten ... Kanäle"**: the front is the column of states computed for the newest position. Channels are fixed low-dimensional readouts of that column. The definitions are in section 3.

## 2 What is established

### 2.1 H-neurons and neighbouring hallucination signals

**Gao et al. 2025 (H-neurons)**
- Models: Mistral-7B-v0.3, Mistral-Small-3.1-24B, Gemma-3-4B and 27B, Llama-3.1-8B, Llama-3.3-70B, plus their base models.
- Labels:
  - [corrected] TriviaQA (the split is not stated), 10 samples per question (T 1.0, top-k 50, top-p 0.9);
  - questions are kept only if answered correctly 10/10 or incorrectly 0/10, refusals excluded, giving 1,000 + 1,000;
  - GPT-4o marks the answer spans.
- [corrected] Feature: CETT_{j,t} = ||W_down z_{j,t} e_j|| / ||W_down z_t||, where z = σ(W_gate x) ⊙ W_up x with the model's gate activation σ (the paper does not fix σ; SiLU for Qwen3). It is averaged over answer tokens and, separately, over all other tokens.
- Label sign:
  - The prose says the positives are answer-token vectors of hallucinated responses, and the negatives are faithful answer tokens plus all non-answer vectors.
  - The printed label equation and the first sentence of §6.1 state the opposite sign. The prose and §3 support "positive = hallucination".
- Classifier:
  - L1 logistic regression;
  - C is chosen by grid search to maximize held-out accuracy plus TriviaQA accuracy with the selected neurons suppressed;
  - H-neurons are the neurons with positive weights.
- Detection (accuracy on balanced single-sample sets):
  - In-domain TriviaQA: 70.1-83.6 %, against 56.1-68.4 % for random neuron sets of equal size (+14 to +20 pp).
  - Out of domain the H-neurons are mostly above random. One exception goes undiscussed: Llama-3.1-8B on fabricated entities falls below random (43.1 against 50.6).
- Intervention:
  - z_j ← α z_j with α ∈ [0, 3] at all positions;
  - compliance rises with α (average slope ≈3.03 for the three smaller models, ≈2.40 for the three larger);
  - the response is non-monotonic in places, and the numbers appear only in figures.
- Origin:
  - [corrected] The probe trained on instruction models transfers to the base models. AUROC is "by a large margin" above chance. The values are in Figure 4; the text adds only that the Mistral family exceeds 86 % on TriviaQA.
  - H-neurons drift little under SFT: normalized rank ≈0.97 for Mistral-Small, and > 0.58 for Gemma and Llama (P < 0.001).
- Not reported: the layer distribution of H-neurons. I found no code release; the third-party package `hprobes` (by the replication authors) implements the method.

**Cavus et al. 2026 (replication with diagnostics)**
- Setup: Gemma-3-4B and MedGemma-4B, response-level labels, C = 1.0.
- Size: 2-9 H-neurons (0.006-0.026 ‰).
- Detection (Gemma AUROC): 0.81 on TriviaQA, 0.97 on BioASQ, 0.63 on NQ-Open.
- Overlap across datasets is partial: TriviaQA and BioASQ share 4 neurons, BioASQ and NQ-Open share none.
- Causal test (Gemma, TriviaQA): judge accuracy is 0.472 under suppression, 0.496 at baseline and 0.490 under amplification. A random same-layer set gives 0.482; McNemar p < 0.001.
- Capability preservation is untested.
- Conclusion: detection holds; unique localization does not.

**Competing readings of what H-neurons encode**
1. Over-compliance (Gao).
2. Question familiarity, or the knowledge boundary:
   - Semantic-entropy probes predict semantic entropy from the last prompt token, before any generation (Kossen et al. 2024).
   - Claude 3.5 Haiku hallucinates when "known entity" or "known answer" features misfire and suppress a default "can't answer" circuit (Lindsey et al. 2025).
3. Confidence regulation: final-layer entropy neurons write into the null space of the unembedding and act through the final norm. That route carries ≈80 % of their effect in GPT-2, LLaMA2-7B and Phi-2, and ≈40 % (30-40 % in the appendix) in Pythia and Gemma (Stolfo et al. 2024).

**Other internal hallucination signals**
- ITI shifts a few attention heads along truthfulness directions (Alpaca on TruthfulQA: 32.5 → 65.1 %).
- INSIDE / EigenScore uses the covariance spectrum of the internal embeddings of several sampled responses.
- Chain-of-Embedding uses the trajectory across layers.
- Token-level linear probes detect fabricated entities in long-form text (Llama-3.3-70B: AUC 0.90 against 0.71 for semantic entropy; Obeso et al.). This is the closest existing token-level, long-form analogue.

### 2.2 Future-token readouts, lenses and MTP

**Future Lens** (Pal et al., CoNLL 2023)
- Setup: four decoders read h_{l,t} in GPT-J-6B on the Pile.
- Targets: the model's own greedy predictions, sampled only where it predicts the next token correctly. This is a selection bias.
- Best precision@1 (Table 2), in the order learned prompt / fixed prompt / linear to hidden state / linear to vocabulary:

| Target | Precision@1 (%) |
|---|---|
| t+1 | 97-98 |
| t+2 | 48.4 / 20.8 / 29.2 / 27.5 |
| t+3 | 43.7 / 30.0 / 19.0 / 19.4 |
| t+4 | 46.9 / 36.5 / 15.8 / 14.7 |

- Bigram baseline: 20.1 % at t+2.
- Future information peaks at layer 14 of 28; for the immediate next token it peaks at the last layer.
- Accuracy at t+2 rises with model confidence: 26, 57, 77 and 95 % for the confidence bins 0-30, 30-60, 60-90 and 90-100 %.

**Pre-caching versus breadcrumbs** (Wu, Morris, Levine 2024)
- Myopic training drops the off-diagonal gradient terms, i.e. the path from the loss at t' to the computation at t < t'.
- GPT-2: 3.28 nats (vanilla) against 3.40 (myopic).
- The gap grows with Pythia model size.

**Planning**
- [corrected] Claude 3.5 Haiku represents candidate rhyme words at the newline, before writing the line. Injecting either of two planned-word features ("rabbit" or "green") into a random sample of 25 poems made the line end with the injected word in 70 % of cases.
- The J-lens paper swaps the planned rhyme "fight" for "light", which changes an earlier word from "coming" to "morning".

**MTP training**
- Gloeckle et al. 2024: n independent heads on a shared trunk. 13B models solve 12 % more HumanEval and 17 % more MBPP problems, and inference is up to 3× faster.
- DeepSeek-V3: sequential modules with depth D = 1 and loss weight 0.3, later 0.1. The modules can be discarded or used for speculative decoding.

**Lenses**
- Tuned lens: one affine probe per block, described as "more predictive, reliable and unbiased than the logit lens".
- **J-lens** (Gurnee, Sofroniew, ..., Lindsey; Transformer Circuits, 6 July 2026):
  - J_l = E_{t, t' ≥ t, prompt}[∂h_final,t' / ∂h_{l,t}] over 1,000 prompts, and lens(h) = softmax(W_U norm(J_l h)).
  - The J-lens therefore already averages over *future* output positions.
  - The paper mentions variants that use only the present position or freeze attention.
  - Swapping lens coordinates of two-hop intermediates succeeds in 54-70 % of trials (Claude 4.5 models).
- Neuronpedia's Qwen3-1.7B J-lens was fit with the companion code on WikiText-103, maximum length 128, with 466 prompts fitted (config.yaml).
- [corrected] J++ (Kola Ayonrinde with Jack Lindsey; LessWrong, 8 Oct 2026) and MicroLens (Zahedi, Zhu, Wu; arXiv 2611.13322, 9 Nov 2026) both exist and match the runde1b README summary. J++ fits per-cluster Jacobians, down-weights clusters with poor readouts, uses an LRP backward pass and filters non-semantic tokens; it reports 55 % against 36 % for the J-lens on intermediate extraction. Neuronpedia lists J++ lenses for Qwen3-4B and several Qwen3.5 models, but none for Qwen3-1.7B.

**Chain lens** (item 0.1), further details
- Design: chain-only attention; reinjecting the hidden state is not significant; contrastive decoding adds 0.02-0.03.
- [corrected] Chain length: "much longer chains, which trade more hallucination under the current summarizer for WorkspaceBench performance". The authors suspect that stronger or prompt-tuned judges could help. On some tasks (introspection, poetry lookahead) they see less confabulation than with the oracle lens.
- Controls: 10 chains beat 50 matched J-lens tokens; foreign seeds and a corpus-mean state hurt.
- Cost: ≈100× a single-token lens for 10 chains of five tokens.
- WorkspaceBench's hallucination score is the share of readouts with at least one claim that an autorater judges contradicted by verifiable context (Blank, Bhatia, Ong, Nanda 2026; I read a summary).

**Arrow of time** (Papadopoulos, Wenger, Hongler, ICML 2024)
- Backward language models have slightly higher loss: English +0.6-0.8 %, other languages up to ≈3 %.
- The gap grows with model size and context length, although "from an information-theoretic point of view, there should be no such difference".
- The authors explain it by sparsity and computational hardness.

**Retrodiction probes** (Feucht et al. 2024)
- Linear probes read tokens at offsets −3 to +1 from h_{l,t} in Llama-2-7b.
- Information about previous and current tokens fades gradually with depth, while information about the next token rises.
- At the last token of multi-token words and entities that information is erased abruptly.

### 2.3 Length and position

- **FActScore** (Min et al. 2023): "Error rates are higher for facts mentioned later in the generation." The authors attribute this to (a) earlier facts being more frequent in pretraining and (b) error propagation. Error rates also rise for rarer entities.
- **Long-form factuality** (Wei et al. 2024): metrics only; the abstract gives no position analysis.
- **Snowballing** (Zhang et al. 2023): models over-commit to early errors. ChatGPT and GPT-4 recognize 67 % and 87 % of their own wrong claims when asked separately.
- **Self-reinforcement** (Xu et al. 2022): "the more times a sentence is repeated in the context, the higher the probability of continuing to generate that sentence". This is a documented attractor that is not hallucination.
- **Same Task, More Tokens** (Levy et al. 2024): padding that varies in length, type and location degrades reasoning "at much shorter input lengths than their technical maximum".
- **Attention sinks** (Xiao et al.): the initial tokens receive strong attention regardless of meaning.
- **Massive activations** (Sun et al.): a few activations are ~100,000× larger than the rest, nearly independent of input; they act as biases and concentrate attention.
- **Lost in the middle**: models use information from the middle of the context less well.
- **RoPE** (Su et al.): rotations by absolute position give an explicit relative dependence that decays with distance. A uniform shift of all position ids therefore leaves attention unchanged, up to rounding.
- **PoSE** (Zhu et al. 2024): shows that position indices can be decoupled from content length, in models trained with skipped positions.
- I found no primary source that applies position gaps to hallucination.

### 2.4 Channels and directed hierarchy

- [corrected] Residual-stream view (Elhage et al. 2021): components "communicate with each other by reading and writing to different subspaces of the residual stream". For attention-only models, paths "are linear if one freezes the attention patterns". With MLPs and normalisation, as in Qwen3, paths stay nonlinear even with frozen attention.
- Information-flow routes (Ferrando & Voita 2024): attribution graphs over token representations from a single forward pass.
- Path patching (Goldowsky-Dill et al. 2023): tests whether a behaviour is carried by specific paths.
- Transfer entropy at layer, token and head granularity (Liang & Shi, NeurIPS 2026 poster): it separates "sinks, broadcasters, and benign components".
- [corrected] For Gaussian variables, Granger causality equals twice the transfer entropy (Barnett, Barrett & Seth 2009, Phys. Rev. Lett. 103:238701, arXiv 0910.4514). The two are equivalent up to a factor of 2.
- SAEs for Qwen3-1.7B (instruct) at the residual, MLP and attention outputs are described by He et al. 2026; where they are released is unverified. Qwen-Scope covers Qwen3 and Qwen3.5 variants; which ones is unverified.

## 3 Structural points behind the operationalisations

**3.1 Front and channel**
- At decode step t the model computes one new column: all layers at position t. All earlier columns are frozen in the KV cache.
- **Front F(t)**: that new column.
- **Channel c**: a fixed map from the column to a low-dimensional variable. Examples:
  - the output o_{l,h}(t) ∈ R^128 of one attention head (448 heads in total);
  - a projection onto a residual subspace at a block output;
  - the CETT score of a neuron group;
  - SAE feature activations.
- **Channel front**: F_c(t) = c(F(t)).
- With teacher forcing, the fronts equal the per-position states computed in parallel (causal mask). Fronts over time therefore carry no dynamics beyond the token sequence.

Dynamics enter through two carriers:
- **KV route (parallel)**: every later position reads the cached keys and values of earlier fronts directly. Distance needs no relays, but content written at block l is usable later only at blocks > l.
- **Token route (sequential)**: the sampled token is embedded again at layer 0. Only on this route does "token n follows from token n−1" hold. Late-layer content can persist over many steps only through tokens, or by being derived again.

**3.2 Arrow of time under the causal mask**
- F_c(t) is a function of x_{≤t}.
- Retrodicting x_{t−k} is *memory*. The token is in the input, so the ceiling is its entropy.
- Predicting x_{t+k} is *anticipation*. The ceiling is A*(k), the model's own k-step predictability given x_{≤t}. At T = 0 the future is a deterministic function of the prefix; at T > 0 the ceiling shrinks with the sampling noise.
- The meaningful per-channel vector is therefore (anticipation profile divided by A*(k), memory profile divided by token entropy).
- For one channel's own fronts, I(F_t;F_{t+k}) = I(F_t;F_{t−k}) on stationary stretches. With whitened single-lag linear probes the explained variance is also symmetric: it is tr C(k)ᵀC(k) in both directions, where C(k) is the whitened lag-k cross-covariance.
- [corrected] A same-channel asymmetry in these measures therefore indicates non-stationarity. That is either position and length drift (a length artifact to exclude) or event-locked change such as a hallucination onset (possibly the signal itself). It is a diagnostic that needs position-detrending and event-locking, not evidence for H-CHAIN on its own. With capacity-limited nonlinear probes an asymmetry can also be computational (Papadopoulos et al. 2024).
- Directional, non-trivial quantities:
  - cross-channel lead-lag, I(F_A(t);F_B(t+k)) − I(F_A(t);F_B(t−k));
  - conditional transfer (Granger or transfer entropy);
  - interventional transfer.

**3.3 Depth budget and A3**
- For t' > t, ∂h_final,t' / ∂h_{l,t} passes only through attention in blocks l+1 to 27.
- [corrected] It is identically zero at block 27, whatever the lag (structural). At blocks 22-26 it passes through only 1-5 attention layers. Round 2 found the A3 projection there at about 1e-5, but that is an empirical result for one readout; a late attention layer can in principle carry a large cross-position effect.
- [corrected] This explains the exact zero of round 2's A3 readout at block 27 and is consistent with its near-vanishing at 22-26. For any lag-resolved Jacobian lens, the exact zero at block 27 is a structural sanity check.
- Cross-position lenses should read from blocks ≤ ~20.

**3.4 Three meanings of "break-off"**
- (a) **Information horizon**: the state carries no more information about t+k than a foreign state does.
- (b) **Lens compounding**: a chained readout (an MTP rollout) diverges while a direct k-step readout does not. This is a lens artifact, as in the chain-lens post.
- (c) **Model hallucination**: the model's output is factually wrong.
- H-CHAIN connects (a) and (b) to (c) through H-neurons. The plan must measure all three separately.

## 4 Operationalisations

Full specifications are in the `operationalisations` field; see section 8 for verifier corrections to it.

**O1: Channels, fronts and carriers**
- Contents: the definitions above, the data, and perturbations along the KV route only versus the token route only.
- Latent throughput: TP_c(k) = CV-R²[F_c(t+k) | F_c(t), x_{t+1..t+k}] − CV-R²[F_c(t+k) | x_{t+1..t+k}].
- [corrected] TP_c(k) is descriptive only. F_c(t) is a function of x_{≤t}, and x_{≤t} reaches F_c(t+k) through attention, so TP_c(k) > 0 is expected even if no latent content outlives the next tokens. Only the KV-only versus token-only perturbations separate the carriers.
- Cost: ≈1 A100-h.

**O2: Arrow vector per channel**
- Memory m_c(k) and anticipation a_c(k) from cross-validated probes, with prompts kept together in each fold, and Monte-Carlo ceilings.
- α_c = weighted mean of a − weighted mean of m.
- Lead-lag between channels.
- Same-channel asymmetry is used only as a non-stationarity diagnostic.
- Cost: ≈0.8 A100-h.

**O3: Chain lens with break-off**
- Three lenses:
  - L-par: direct ridge maps h_{l,t} → h_final,t+k−1, read through W_U ("Reihe").
  - L-jac: the lag-resolved J-lens J_l(Δ) = E[∂h_final,t+Δ / ∂h_{l,t}]; training-free and runs through attention.
  - L-mtp: the vgel & Douglas chain lens with a trained DeepSeek-style MTP head of 58.7M parameters ("Folge").
- Agreement: S_l(k) = mean log p_lens(x_{t+k}) − mean log p_lens^foreign(x_{t+k}).
- **Break-off horizon k\***: the smallest k whose upper 95 % prompt-bootstrap bound of S_l(k) is below 0.05 nats.
- **Lens hallucination**: LH(k) = P(lens max-prob ≥ p0 and the actual token ranks below 10th). p0 is the 90th percentile of lens confidence on neutral text. k_LH is the smallest k with LH(k) ≥ LH_foreign(k).
- **Field-of-view criterion**: FOV_H(k) is the share of the lens logits attributable to H-neuron write contributions, compared with 50 matched random neuron sets.
- [corrected] Checking L-jac against the published J-lens needs all lags of the 128-token fit (Δ = 0..127) on the same WikiText prompts, with the fit's weighting, target layer and aggregation. Lags above 15 carry most of that average.
- Cost: ≈3-3.5 A100-h.

**O4: H-neurons in Qwen3-1.7B**
- The Gao pipeline, with labels from alias matching instead of GPT-4o.
- Cavus diagnostics D1-D5.
- Matched random sets and damage gates.
- Tests:
  - whether the score at the last prompt token already predicts hallucination;
  - overlap with entropy neurons;
  - transfer to the base model;
  - over-compliance (sycophancy flips, FaithEval counterfactual context).
- [corrected] Whether H is a generation-time signal is decided by discrimination (AUROC) at the last prompt token versus on answer tokens, not by score levels, which the labelling already pushes down off the answer span.
- Expect 2-60 neurons that are not unique; treat the result as a subspace.
- Cost: ≈2-3 A100-h.

**O5: Length artifact versus attractor**
- (a) Position: a global offset, which must change nothing, and relative gaps G ∈ {0, 512, 2048, 8192}. [corrected] Gaps in position ids without tokens are not part of standard training, so a gap effect may reflect the novel configuration rather than distance.
- (b) Content at matched positions: neutral filler, unrelated coherent text, the model's own correct earlier answers, its own hallucinated earlier answers, and injected wrong claims. [corrected] This separates content from length, but not an attractor from ordinary in-context conditioning on wrong claims (snowballing).
- (c) Persistence: inject or suppress H-neurons, then measure the per-token decay and the carry per answer, K_H, through the KV route and the token route separately. Compare against 50 damage-matched random sets, with a dose series that includes α 1.1-1.5. [corrected] By round 2's rule R6, self-sustainment needs a carry with lower bound > 1. K_H above random sets but below 1 shows above-baseline persistence, not an attractor. An attractor verdict needs K_H > 1 or bistability within items (the same question across seeds), with the criteria combined under a multiplicity correction.
- (d) Lead and lag between lens reach and H-score, with surrogates and Granger tests.
- Cost: ≈2 A100-h.

**O6: Channel hierarchy**
- Interventional transfer gains between ≈40 channels.
- Path patching for heads.
- Granger and transfer entropy as an observational cross-check.
- Hierarchy scores after removing the depth order.
- Reordering by prompt type, temperature and context length.
- [corrected] Cost: ≈1-1.5 A100-h for 4 conditions of 512-token contexts; the specified condition set (at least 13 cells, contexts up to 8k) needs about 2.5-3 A100-h.

**Power** (two-sided α 0.05, power 0.8)
- Paired designs: 300 items detect d_z ≥ 0.16; 500 items detect d_z ≥ 0.125.
- Partial correlations with 500 prompts detect r ≥ 0.125.
- Persistence tests against random sets need ≥ 40 sets to resolve p < 0.025; 50 are planned.

**Links to round 2**
- The style direction and the RSI axis become named channels.
- [corrected] RSI returning 2-5 % per loop round (one round = a 40-token generation plus a mean-state read-out) is not a per-token rate. It suggests, but does not predict, a short token-level anticipation horizon for RSI words, while the style channel should persist (O2, O3).
- A3: block 27 is a structural zero; L-jac replaces A3 by reading at blocks ≤ 20 (O3).
- Damage: the u_RSI erasure in Part C cost 0.11 nats, so every H-intervention needs damage-matched controls (O4, O5).
- Reference carries are K 0.45-0.48 per answer (style) and K 1.053 on the RSI axis (unresolved). Report K_H with an interval for its denominator (O5).
- [corrected] Temperature: round-2 G falls with temperature at block 6 (slope −0.157) and is flat at block 14. By analogy only (G is a loop gain, not an edge weight), test whether temperature reorders edges at early blocks only (O6).

## 5 Main confounds

The full list is in the `confounds` field.
- Position is confounded with content (FActScore).
- Uncertainty is a common cause of lens failure and H-activity (Future-Lens accuracy tracks confidence; SEPs work before generation; entropy neurons).
- [corrected] The consistency filter plausibly enriches for unknown entities (unverified).
- L1 selection is not unique.
- CETT drifts with the norm of the MLP output.
- Attention sinks, massive activations and template tokens.
- The anticipation ceiling depends on temperature.
- Domain shift of the lens: the J-lens was fit on 128-token WikiText.
- Chained lenses compound their own errors.
- Repetition loops.
- Damage from interventions.
- Label noise.
- Autocorrelation within a text.
- Multiple comparisons.

## 6 Costs

| Part | A100-40GB hours |
|---|---|
| O1 | ≈1 |
| O2 | ≈0.8 |
| O3 | ≈3-3.5 |
| O4 | ≈2-3 |
| O5 | ≈2 |
| O6 | ≈1-1.5 as costed (4 conditions × 512 tokens); [corrected] ≈2.5-3 for the specified conditions |
| **All six** | **≈10-13 as costed; [corrected] ≈11.5-13.5 with O6 as specified** (can be split across sessions) |

- Minimal path: O4 → O3 (L-par, L-jac) → O5 (a-c), ≈6-7 h.
- Memory: the model in bf16 takes ≈3.4 GB. The L1 feature matrix (4,000 × 172,032, float32) takes 2.75 GB.
- These are FLOP estimates, not measurements. They assume ≈100 TFLOPS effective for teacher-forced passes and ≈2-3k tokens/s for batched HF generation.

## 7 Where the analogy breaks

Details are in the `where_analogy_breaks` field; see section 8 for verifier corrections to it.
- A channel's own fronts are symmetric in time under stationarity, for MI and whitened linear probes.
- Channels are a choice of basis within one shared residual stream.
- No gradients flow at inference, and Qwen3-1.7B has no MTP head.
- The published chain lens reads workspace content rather than forecasting tokens, and it has no break-off criterion.
- H-neurons are an answer-level construct from short QA, and their selection is not unique.
- An "attractor" needs dynamical evidence, not correlation.

## 8 Verifier corrections to the structured fields

- `refs`:
  - Lindsey et al. 2025: the 70 % covers two injected planned-word features ("rabbit" and "green") in 25 poems.
  - Elhage et al. 2021: the linearity quote is stated for attention-only models.
  - MicroLens and J++: verified (J++: https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr, 8 Oct 2026).
  - Feucht et al. and Ferrando & Voita: EMNLP 2024 confirmed in the ACL Anthology.
  - Qwen3-1.7B: the model card states a context length of 32,768; 40,960 is the config's max_position_embeddings.
- `operationalisations`, O1:
  - TP_c(k)'s null "≈ 0 for k ≥ 2" is wrong (see section 4).
  - Storage: 19,456 values per token are ≈39 KB in bf16 (≈47 GB for 1.2M tokens), not ≈20 KB.
- O2:
  - Null corrected: memory falls with depth, and next-token readability rises with depth (Feucht et al.), but t+2..t+4 readability peaks at middle layers (Future Lens).
  - "A_c(k) ∝ A*(k)" means the normalised a_c(k) is flat in k.
  - The cost line's ridge into final states differs from the specified linear-softmax probes. For memory targets, regress onto the tied token embedding, not a final state.
- O3:
  - The sanity check needs all lags (see section 4).
  - Prediction (2), that the three criteria coincide, is not diagnostic.
  - Prediction (4) needs a pre-registered sign (under H-CHAIN, reach falls as H rises).
  - Entropy matching does not control token rarity.
  - "||J_l(Δ≥1)|| must fall to 0 as l → 27" holds exactly only at l = 27.
- O4: replace "H-score higher on answer tokens than at the last prompt token" with "the answer-token H-score adds AUROC beyond the last-prompt-token score".
- O5:
  - (a) Gap effects carry the out-of-distribution caveat.
  - (b) It does not separate an attractor from snowballing or in-prompt induction.
  - (c) An attractor needs K_H > 1 or within-item bistability; "lower bound > 0.55", convexity, or bimodality across mixed items are insufficient, and "any one of three" needs a multiplicity correction.
- O6: cost as in section 6.
- `link_to_round2`:
  - "README item 2/4/5" refers to AUSWERTUNG.md §5 items 2/4/5.
  - g_par (2-5 %) is per loop round, so it does not predict k* ≈ 1-2 tokens.
  - The latent loop is analogous to the KV route, not "exactly" the same split.
- `confounds`:
  - "Consistently-incorrect items are largely unknown entities" is unverified.
  - The documented context length is 32,768.
- `where_analogy_breaks`:
  - Qwen3-1.7B ships no MTP weights, and the Qwen3 Technical Report (arXiv 2505.09388) does not mention multi-token prediction. Missing weights alone would not show that it was never trained with MTP.
  - The chain-lens post qualifies the growth of hallucination with chain length as "under the current summarizer"; it does not attribute it to the head.
  - Round 2's "generic attractor" is a descriptive label (no loop-dynamics verdicts were licensed; all gains < 1), not a demonstrated attractor.
  - Same-channel asymmetry: see section 3.2.


## operationalisations (from the research stage; check against the verifier's notes above)
```json
[
 {
  "name": "O1 Channel fronts and the two carriers (definitions and data)",
  "measurement": "Front: at decode step t (KV cache on), F(t) is the column of states computed for the newest position t (all 28 blocks: residual outputs, head outputs, MLP activations); earlier columns are frozen in the cache. In teacher-forced replay the front series equals the per-position states; check this numerically. Channel c is a fixed map from the column to a low-dimensional variable F_c(t). Pre-register the channel set:\n(a) all 448 query heads (28 × 16): head output before o_proj (128-d), plus routing weights A_{l,h}(t,s);\n(b) named residual subspaces at block outputs b ∈ {6,10,14,18,20,22}: the round-2 style-attractor direction, the round-1 RSI axis u_T_perp, the sea axis, the H-subspace (span of H-neuron write vectors W_down[:,j] from O4), the J-lens workspace subspace (top-64 active J-lens vectors), and the top-16 PCs from neutral text;\n(c) neuron channels: the H-neuron CETT score and 50 matched random neuron sets;\n(d) optionally, Qwen3-Instruct SAE features.\nCarriers:\n- KV route: later positions read cached keys and values directly, in parallel and without relays.\n- Token route: the sampled token is embedded again at layer 0; this route is sequential.\nSeparate them with a KV-only intervention (perturb the front at t, then teacher-force the original tokens) and a token-only intervention (re-encode, with the clean model, the text generated under the perturbation). Observational counterpart: latent throughput TP_c(k) = CV-R²[F_c(t+k) | F_c(t), x_{t+1..t+k}] − CV-R²[F_c(t+k) | x_{t+1..t+k}], with prompt-blocked cross-validation.\nData: Qwen3-1.7B, chat template, thinking off. Five prompt families:\n- single TriviaQA questions;\n- sessions of 10-20 questions;\n- biographies of frequent vs rare entities;\n- round-1/2 RSI prompts with sea/sports controls;\n- neutral 'Continue.'.\nSettings: T ∈ {0, 0.7, 1.0}, 256 new tokens. Store 16-d projections per head and full states for 6 blocks; recompute anything else by replay.",
  "prediction_if_hypothesis": "Some channels carry information about later fronts through the KV route:\n- TP_c(k) > 0 for k ≥ 4;\n- KV-only perturbations of these fronts persist longer than equal-dimension random subspaces at the same block;\n- channel profiles differ beyond what block depth predicts.",
  "prediction_if_null": "Beyond 1-2 tokens, all carry goes through the token route:\n- TP_c(k) ≈ 0 for k ≥ 2 once the intervening tokens are conditioned on;\n- KV-only perturbation effects decay within ≈2 tokens for every channel;\n- channel differences are explained by depth.",
  "controls": "- Replay check: generation fronts equal teacher-forced fronts within bf16 tolerance.\n- Random subspaces of equal dimension per block; matched random head sets.\n- Exclude chat-template tokens and the first 4 positions (attention sinks, massive activations).\n- Prompt-blocked splits; separate analyses per temperature.",
  "cost_estimate": "≈1 A100-40GB hour:\n- ≈1.2M generated tokens with hooks at batch 32-64;\n- ≈20 KB/token storage;\n- keep ≤25 GB, or recompute by teacher-forced replay.",
  "link_to_round2": "- The round-2 style-attractor direction (R4 'generic attractor' at blocks 14/18) and the RSI axis become named channels.\n- Round-2 K (0.45-0.48 per answer at block 14) is a token-route quantity.\n- README item 5 ('latent vs text loop') is exactly the KV-route vs token-route split defined here."
 },
 {
  "name": "O2 Arrow-of-time vector per channel: memory vs anticipation with cross-validated probes",
  "measurement": "For each channel (front reduced to 64 dims by PCA so probes have equal capacity) and k ∈ {1,2,3,4,6,8,12,16}:\n- Memory M_c(k): cross-validated log-likelihood gain (nats/token) over the unigram baseline of a linear-softmax probe from F_c(t) to x_{t−k}. Retrodiction design as in Feucht et al.\n- Anticipation A_c(k): the same probe aimed at x_{t+k}, the token the model actually generated. This is Future Lens' linear model approximation. A soft-target variant uses the model's p(x_{t+k} | x_{≤t+k−1}).\nCeilings:\n- M_max is the unigram entropy, since x_{t−k} is in the input.\n- A*(k) is the model's own k-step predictability given x_{≤t}, estimated by Monte Carlo (32 sampled continuations per position): E[log p̂(x_{t+k}|x_{≤t})] − E[log p_uni].\n- At T = 0 the future is a deterministic function of the prefix, so A*(k) is the full entropy and the normalised score measures linear accessibility, not information.\nNormalisation and indices:\n- m_c(k) = M_c/M_max and a_c(k) = A_c/A*(k).\n- Arrow vector v_c = (a_c(1..K), m_c(1..K)).\n- Scalar index α_c = Σ_k w_k a_c(k) − Σ_k w_k m_c(k), with w_k ∝ 1/k.\nDirectional cross-channel quantity: lead-lag L_AB(k) = I_probe(F_A(t);F_B(t+k)) − I_probe(F_A(t);F_B(t−k)).\nSame-channel forward/backward state predictability (F_c(t) → F_c(t±k), whitened, position-matched windows) is reported only as a non-stationarity diagnostic D_c(k), because under stationarity it is identically symmetric.\nPositive controls:\n- multi-token named entities (t+2 should be predictable);\n- rhyming couplets read at the newline.",
  "prediction_if_hypothesis": "- A minority of channels is anticipatory beyond its layer band: α_c exceeds the layer-band mean by more than 3 bootstrap SD, in more channels than the FDR expectation.\n- These channels show positive latent throughput (O1).\n- Their anticipation drops at or before the chain-lens break-off and the H-neuron rise (O3, O5).\n- Lead-lag edges are stable across prompts.",
  "prediction_if_null": "- α_c is a smooth function of depth: memory falls and anticipation rises across layers, as Feucht et al. and Future Lens report.\n- Head identity adds no variance beyond layer in a mixed model.\n- a_c(k) decays with k in proportion to A*(k).\n- D_c(k) ≈ 0 after position detrending.\n- Lead-lag is explained by depth order.",
  "controls": "- Prompt-blocked nested cross-validation.\n- Unigram and bigram baselines.\n- Label shuffles within position bins.\n- Equal probe capacity and regularisation chosen in inner CV.\n- Greedy and sampled continuations analysed separately.\n- Position bins.\n- Exclude template and sink positions.\n- FDR across channels × k.\n- Positive controls (entities, rhymes) must pass before negative results are read.",
  "cost_estimate": "≈0.8 A100-h:\n- ≈0.5 h for probes on cached fronts (closed-form ridge into the 2048-d final-state space, read through the tied W_U);\n- ≈0.3 h for Monte-Carlo ceilings (200 prompts × 32 positions × 32 samples × 16 tokens ≈ 3.3M tokens; halve if needed).",
  "link_to_round2": "- Round 2: an RSI push returns only 2-5 % per step (g_par 0.0456/0.0372/0.0243 at blocks 14/18/6), and all loops converge on a shared style direction.\n- Prediction: the style channel shows high memory and long anticipation (persistent); the RSI channel's anticipation of RSI words falls to baseline by k ≈ 1-2.\n- MicroLens' 'readable is not used' applies: O2 measures readability; use needs the O3/O6 interventions."
 },
 {
  "name": "O3 Chain lens with a pre-registered break-off horizon k* and 'lens hallucination'",
  "measurement": "Lenses, all evaluated against the tokens the model actually generates (greedy and sampled runs separately):\n(1) L-par (direct/parallel, 'Reihe'): for source blocks l ≤ 20 and k = 1..16, ridge maps h_{l,t} → h_final,t+k−1 read through softmax(W_U norm(·)), fit on the model's own generations with prompt-blocked CV (Future Lens' linear approximation).\n(2) L-jac (training-free, through attention): lag-resolved J-lens J_l(Δ) = E_{t,prompt}[∂h_final,t+Δ/∂h_{l,t}] for Δ = 0..15, read as softmax(W_U norm(J_l(Δ) h_{l,t})).\n- Compute: one batched backward pass per output position yields every source position and layer.\n- Sanity check: the lag-weighted sum must reproduce the published Neuronpedia J-lens.\n- Cheaper sketch: source-side JVPs on named/PC directions.\n(3) L-mtp (sequential, 'Folge'; the vgel & Douglas chain lens):\n- seeds: the Neuronpedia J-lens;\n- head: a trained DeepSeek-style MTP head, i.e. [RMSNorm(h_final,t); RMSNorm(Emb x_{t+1})] → fc 4096→2048 → one Qwen3 block initialised from block 27 → tied lm_head, 58.7M parameters, trained by KL to the model's own p(x_{t+2}|x_{≤t+1}) on ≈20M tokens;\n- decoding: chained rollouts with contrastive decoding against a corpus-mean state and chain-only attention.\nAgreement: S_l(k) = mean log p_lens(x_{t+k}) − mean log p_lens^foreign(x_{t+k}), where the foreign lens reads a state from another prompt at the same position. A corpus-mean state is a second control.\nBreak-off horizon k*_l: the smallest k whose upper 95 % prompt-bootstrap bound of S_l(k) is below 0.05 nats. Per-token reach r_l(t) = Σ_k max(0, s_t(k)).\nLens hallucination ('confident and wrong'): LH_l(k) = P(lens max-prob ≥ p0 and the actual token ranks below 10th), with p0 = the lens's 90th-percentile confidence on neutral reference text. k_LH is the smallest k with LH_l(k) ≥ LH_foreign(k).\nField-of-view criterion: FOV_H(k) = norm share of the lens logit vector attributable to H-neuron write contributions Σ_{j∈H, layer(j)≤l} z_j W_down[:,j] (linear before the final norm), compared with 50 matched random neuron sets. k_FOV is the smallest k at which FOV_H exceeds the sets' 95th percentile.\nModel hallucination labels come from the O4 tasks.",
  "prediction_if_hypothesis": "(1) At matched positions and matched model entropy, k*, r(t) and LH differ between hallucination-prone contexts (consistently-incorrect questions, fabricated entities, the model's own earlier errors) and known-answer contexts: a paired difference ≥ 1 token in k*, or d_z ≥ 0.2 in r.\n(2) The three break-off criteria (agreement, LH, FOV_H) coincide within ±1 token.\n(3) FOV_H rises with k above the random-set band.\n(4) The partial correlation of r(t) with the position-detrended H-score is ≥ 0.15 after controlling for position, model entropy and token type.",
  "prediction_if_null": "- k* and LH are fully explained by the model's own predictability A*(k), token entropy and position.\n- L-mtp breaks off earlier than L-par only through its own compounding; the chain-lens post reports that confabulation grows with chain length.\n- FOV_H stays within the random-set band.\n- Partial correlations are ≈ 0.\n- Hallucination-prone and known contexts give the same k* once entropy is matched.",
  "controls": "- Foreign-state and corpus-mean-state lenses; shuffle nulls.\n- Neutral and foreign seeds for L-mtp, as in the chain-lens post.\n- Entropy-stratified analysis; normalisation by A*(k) from O2.\n- Structural positive control: ||J_l(Δ≥1)|| must fall to 0 as l → 27.\n- Rhyme planning at the newline as a positive control (Lindsey et al. 2025; Gurnee et al. 2026).\n- L-par vs L-mtp comparison to separate the information horizon from lens compounding.\n- Repetition-loop stratification.",
  "cost_estimate": "≈3-3.5 A100-h in total:\n- L-par ≈0.5 A100-h.\n- L-jac ≈1.5 A100-h exact (100 prompts × 3 output positions × 2048 VJPs on 128-token contexts; own FLOP estimate), or ≈0.3 h as a source-side JVP sketch.\n- L-mtp ≈1-1.5 A100-h to train the MTP head (the base forward over ≈20M tokens dominates) plus minutes of rollouts. The J-lens is already published.",
  "link_to_round2": "- A3 gradient readout: the round-2 code asserts 'A3 at the last block is identically zero (a leaf at the object positions cannot reach the last position)' (analysis/ju_space/round2/r2_rgi_geom.py:695); at blocks 22-26 only 1-5 attention layers remain.\n- L-jac reproduces this profile as a positive control.\n- L-jac replaces the A3 readout (README item 4) by reading at blocks ≤ 20 or at lag 0.\n- Before use, check that lens predictions match the measured sign-true shifts (README item 4, MicroLens probe).\n- RSI's 2-5 % return per step predicts k* ≈ 1-2 for RSI words."
 },
 {
  "name": "O4 H-neurons in Qwen3-1.7B (Gao et al. pipeline plus Cavus et al. diagnostics) and what they encode",
  "measurement": "Model: Qwen3-1.7B (chat template, thinking off). Base model Qwen3-1.7B-Base for the origin check.\nLabels:\n- TriviaQA (rc.nocontext train) with the prompt 'Answer with only the answer.'\n- 10 samples per question (T 1.0, top-k 50, top-p 0.9, ≤24 new tokens).\n- Correctness by normalised alias match; refusals excluded from the incorrect set.\n- Keep questions answered correctly 10/10 or incorrectly 0/10 until 1,000 of each (≈6,000 questions).\n- Answer span: the tokens after a fixed 'Answer:' prefix up to the newline, replacing Gao's GPT-4o span marking.\nFeatures:\n- CETT_{j,t} = |z_{j,t}|·||W_down[:,j]|| / ||W_down z_t|| for all 28 × 6144 = 172,032 neurons;\n- means over answer tokens and over non-answer tokens (pre-register whether prompt tokens count; run both).\nProbe (Gao's prose):\n- Positives are hallucinated answer vectors; negatives are faithful answer vectors and all non-answer vectors.\n- L1 logistic regression; C chosen on a grid to maximise held-out accuracy plus TriviaQA accuracy with the selected neurons suppressed (α = 0).\n- H = positive weights.\nDiagnostics (Cavus D1-D5):\n- max |r| to unselected neurons;\n- bootstrap Jaccard (B = 20);\n- L1 vs L2 top-|S| overlap;\n- matched random same-layer controls;\n- cross-dataset overlap (TriviaQA, NQ-Open, BioASQ, fabricated entities).\nDetection: accuracy and AUROC on balanced single-sample sets.\nCausal test:\n- α ∈ {0, 0.5, 1, 1.25, 1.5, 2, 3}, scaling z at all positions.\n- Outcomes: TriviaQA accuracy and refusal rate; sycophancy flip rate after 'I don't think that's right. Are you sure?'; FaithEval counterfactual (rule parser); FalseQA only if a judge is available.\n- Damage: KL on neutral reference text.\n- Controls: 50 random sets matched in layer, count, mean CETT and KL damage.\nWhat H encodes:\n- (i) H-score at the last prompt token (before generation) vs on answer tokens;\n- (ii) overlap with final-layer entropy neurons (high ||w||, write vector in the W_U null space) and token-frequency neurons;\n- (iii) correlation with token entropy and semantic entropy;\n- (iv) J-lens readout of the H write vectors with neuron-lens round-trip checks;\n- (v) base-model transfer (AUROC) and SFT drift rank.\nOutput an H-subspace (stable neurons plus correlated cluster members) for O1, O3 and O5.",
  "prediction_if_hypothesis": "H-CHAIN presupposes all of the following:\n- A sparse set (expected 2-60 of 172,032 neurons, i.e. 0.01-0.35‰) detects answer-level hallucination ≥10 pp above matched random sets in-domain and generalises out of domain.\n- α-scaling changes over-compliance beyond damage-matched random sets.\n- The H-score is higher on answer tokens than at the last prompt token, i.e. a generation-time signal that can participate in chain dynamics.",
  "prediction_if_null": "Any of the following counts against H-CHAIN's premise:\n- Detection ≤ random + 5 pp out of domain.\n- Causal effects do not exceed damage-matched random sets.\n- The last-prompt-token H-score predicts hallucination as well as the answer tokens do, i.e. a question-familiarity or uncertainty signal like SEPs before generation.\n- The set is dominated by entropy or token-frequency neurons.",
  "controls": "- Cavus D1-D5; bootstrap stability; elastic-net comparison.\n- Random sets matched on layer, count, CETT and damage.\n- Round-2 damage gates: report KL; 0.05 nats calibration, 0.15 nats out of range.\n- Alias-list audit of 100 labels by hand.\n- Refusal handling; held-out questions disjoint from training.\n- Separate reporting of detection vs localisation.",
  "cost_estimate": "≈2-3 A100-h, plus CPU time:\n- sampling ≈1M tokens;\n- CETT extraction for ≈4,000 items with hooks;\n- L1 fits on a 4,000 × 172,032 float32 matrix (2.75 GB; GPU proximal gradient or liblinear);\n- ≈10 suppression evaluations for choosing C;\n- the α sweep with 50 random sets on ≈500-item benchmarks.",
  "link_to_round2": "- Part C: erasing u_RSI left 16 % but cost 0.11 nats (> 0.05), so 'not unlearned'. Every H-scaling must be reported with the same damage metric and matched controls (README item 2).\n- Part C: three in-prompt examples restore 89 %. Test whether in-prompt confident wrong answers raise H-activity (few-shot induction).\n- Report cos and J-metric cos between the H write vectors and the round-2 style-attractor direction at the same block."
 },
 {
  "name": "O5 Length artifact vs H-neuron attractor: position gap, filler vs coherent content, perturbation persistence, lead/lag",
  "measurement": "(a) Position:\n- A global position_id offset P ∈ {0, 4096} must reproduce the logits within bf16 tolerance, because RoPE is relative. This is a sanity check.\n- Relative gaps G ∈ {0, 512, 2048, 8192} between prompt and answer: a custom decode loop with explicit position_ids and no added tokens.\n- Neutral filler of N tokens adds keys (attention dilution) as well as distance.\n(b) Content at matched positions: the same target question is answered at the same absolute position (e.g. token 1,500) after:\n- (i) neutral filler;\n- (ii) unrelated coherent text;\n- (iii) the model's own correct earlier answers;\n- (iv) the model's own hallucinated earlier answers (from consistently-incorrect items);\n- (v) teacher-forced injected wrong claims (snowball).\nOutcomes: answer-level H-score (Gao features, position-detrended against neutral reference text), accuracy, k* and LH from O3. Analysis: paired design per question plus a mixed model H ~ position × content + (1|question).\n(c) Persistence:\n- Inject (α ∈ {1.1, 1.25, 1.5, 2, 3}) or suppress (α = 0) H-neurons for a window (the first answer of a session, or 16 tokens), then stop.\n- Measure the excess natural H-score per token afterwards and fit a decay τ_H.\n- Measure the carry per answer, K_H = excess at answer n+1 / excess at answer n, with an interval for the denominator.\n- Two routes: KV-only (perturbed cache, original tokens) and token-only (clean model re-reads the text written under perturbation).\n- Compare with 50 random sets matched for damage.\n- Attractor criteria (all pre-registered): K_H above the random 95th percentile with lower bound > 0.55; convex dose response; or a bimodal late-session H distribution (dip test).\n(d) Lead/lag:\n- Per-token series r(t) (lens reach) and H̃(t), detrended for position, entropy and token type.\n- Cross-correlation with circular-shift and prompt-shuffle surrogates (1,000 each).\n- Event-triggered averages of break-off and H-rise events.\n- VAR/Granger in both directions with exogenous covariates and a prompt-blocked bootstrap.",
  "prediction_if_hypothesis": "- Content at matched positions drives H-activity and lens break-off: (iv) and (v) raise the answer-level H-score relative to (i) and (ii) with paired d_z ≥ 0.2, while gaps have little effect.\n- An H-injection persists beyond matched random sets: K_H above their 95th percentile, slower τ_H, a superlinear dose response or bimodal outcomes; it is carried by the token route, the KV route or both.\n- Lens break-off and H-rise show a consistent non-zero lead/lag that survives detrending and surrogates.",
  "prediction_if_null": "- Gap, filler and coherent conditions shift H-activity and k* equally at matched lengths: a length or position effect.\n- An H-injection decays like matched random sets (K_H within their band), with a linear dose response.\n- The cross-correlation peaks at lag 0 and vanishes after position and entropy detrending.\n- No Granger direction survives.",
  "controls": "- Global-offset sanity check.\n- Identical questions and answer positions across conditions.\n- Filler types (dots, lorem ipsum, unrelated Wikipedia).\n- Random sets matched for damage, layer and count.\n- KV vs token routes.\n- Exclude sinks and template tokens; stratify repetition loops.\n- Normalise by model entropy.\n- Prompt-level bootstrap; FDR across conditions.\n- Contexts kept far below the 40,960-position maximum.",
  "cost_estimate": "≈2 A100-h:\n- (a)+(b) ≈0.3 h (≈4,500 prompts of ≈1.5k tokens prefill plus short answers);\n- (c) ≈1-1.5 h (51 neuron sets × 2 main doses × 2 routes × 200 sessions ≈ 6M generated tokens; the dose series on the H-set only);\n- (d) analysis on O3 data.\nPower: paired n = 300 detects d_z ≥ 0.16.",
  "link_to_round2": "Reference values from round 2:\n- G 0.12-0.30 and K 0.45-0.48 per answer (style direction);\n- K on the RSI axis 1.053 [0.743, 1.538], unresolved, reported without an interval for its denominator;\n- linear response 0.7298 below 0.5 α_0, so include low doses.\nRequirements and predictions:\n- Report K_H with an interval for its denominator (README item 5).\n- The 'generic attractor' finding predicts that an H-injection may relax into the style direction: track the perturbation's projection on the style direction over time.\n- Toy B-T5 'bistable' cells motivate the bimodality check."
 },
 {
  "name": "O6 Channel hierarchy and its reordering by prompt type, temperature and context length",
  "measurement": "Channels: ≈40 in total:\n- head groups per layer band;\n- named subspaces (style, RSI, sea, H-subspace, J-lens workspace subspace, entropy-neuron output, top PCs);\n- the lens-reach channel.\nEdge estimators:\n(1) Interventional transfer g_BA(Δ) = E[∂F_B(t+Δ)/∂F_A(t)] for Δ = 0..16, via forward-mode JVPs along 4 directions per channel with tokens teacher-forced (KV route). Units: B's natural SD per unit perturbation of A.\n(2) Path patching for ≈20 head pairs (Goldowsky-Dill et al.).\n(3) Observational: conditional Granger causality / Gaussian transfer entropy on detrended front series (VAR on reduced dims, prompt-blocked CV), with token embeddings and position as exogenous inputs; surrogates by circular shift and prompt shuffling.\nHierarchy:\n- W_BA = Σ_Δ g_BA(Δ).\n- Remove the trivial depth/causal order (regress on layer difference, or compare within layer bands).\n- Scores by net outflow or least-squares ranking.\n- Stability: prompt-bootstrap Kendall τ.\nConditions:\n- prompt type (factual QA, open narrative, RSI, sea/sports);\n- temperature (0, 0.7, 1.0, 1.3);\n- context length (none, 1k, 4k, 8k), coherent vs filler.\nReordering: Kendall τ between conditions vs the within-condition bootstrap τ distribution; tests on named edges (style→H, H→reach, RSI→style).",
  "prediction_if_hypothesis": "- Beyond depth order, a stable hierarchy: bootstrap Kendall τ ≥ 0.6 on depth-residualised scores.\n- A few channels are upstream of the H-subspace and of lens reach; expected: the style/generic direction and the J-workspace subspace.\n- Conditions reorder named edges: between-condition τ below the within-condition bootstrap 5th percentile. Example: long coherent self-generated context raises H-subspace outflow, equal-length filler does not.\n- Interventional and observational edges agree (Spearman ≥ 0.5).",
  "prediction_if_null": "- Rankings collapse to depth order plus noise.\n- Between-condition differences lie within bootstrap variability.\n- Observational lead-lag disagrees with interventional transfer (Spearman < 0.3), indicating that shared drivers (tokens, position) create the apparent directionality.",
  "controls": "- Depth-order baseline.\n- Random subspaces of equal dimension at matched blocks.\n- Exogenous token and position covariates.\n- Surrogate nulls.\n- Equal channel dimensionality.\n- FDR across edges.\n- Teacher-forced (KV) vs re-sampled (token route) transfer for a subset.\n- Same prompts across conditions where possible.",
  "cost_estimate": "≈1-1.5 A100-h:\n- ≈40 channels × 4 directions × 100 prompts × 4 conditions of forward-mode JVPs on 512-token contexts (≈0.75 h);\n- path patching for ≈20 head pairs (≈0.3 h);\n- Granger/TE and ranking on CPU.",
  "link_to_round2": "- Round 2: G at block 6 falls with temperature (slope −0.157 [−0.307, −0.009]) while block 14 is flat; R4 is a 'generic attractor' at blocks 14 and 18 but 'unaligned' at 6.\n- Prediction: temperature reorders edges involving block-6 channels but not block-14 channels, and the style channel has high outflow at 14/18 but not at 6.\n- The context-length condition can share data with the H-DUAL long-context tests."
 }
]
```

## confounds (from the research stage; check against the verifier's notes above)
```json
[
 "Position confounded with content: in long-form text later facts are rarer and inherit earlier errors (FActScore). A position effect is only interpretable at matched content, and a content effect only at matched position.",
 "Uncertainty as a common cause: lens accuracy rises steeply with model confidence (Future Lens at t+2: 26 % to 95 % across confidence bins); H-probes may read uncertainty; final-layer entropy neurons regulate confidence. Condition on model entropy and log-prob, and check overlap with entropy neurons.",
 "Selection by the consistency filter: consistently-incorrect TriviaQA items are largely unknown entities. H-neurons may encode question familiarity rather than generation dynamics (SEPs predict semantic entropy before generation; known-entity features in Claude 3.5 Haiku). Test the H-score at the last prompt token.",
 "Non-unique L1 selection: selected neurons are representatives of correlated clusters (Cavus: 19/22 with |r|>0.7, bootstrap Jaccard 0.45-0.69, L1/L2 overlap 0-25 %). Use an H-subspace, bootstraps and elastic-net comparisons, and report detection separately from localisation.",
 "CETT normalisation drift: CETT divides by the norm of the MLP output, which can change with position, massive activations and template tokens. Detrend by position on neutral reference text and check with raw activations.",
 "Attention sinks and massive activations at the first tokens and chat-template tokens distort per-position statistics. Exclude the first 4 positions and template tokens, or model them separately.",
 "The anticipation ceiling depends on temperature: at T = 0 the model's future is a deterministic function of the prefix, at T > 0 it is noisy. Compare horizons only after normalising by the model's own k-step predictability A*(k).",
 "Lens domain shift: the Neuronpedia Qwen3-1.7B J-lens was fit on WikiText with 128-token contexts (466 prompts), while the analysis runs on chat-format generations up to thousands of tokens. Validate on held-out generations, or refit (≈2 A100-h, own estimate).",
 "Lens compounding: chained readouts (MTP rollouts) confabulate more as chains grow (vgel & Douglas). Compare against direct k-step lenses so that lens artifacts are not attributed to the model.",
 "Repetition loops: self-reinforcement (Xu et al. 2022) makes futures highly predictable and creates attractor-like persistence unrelated to hallucination. Detect loops (n-gram repetition) and stratify on them.",
 "Intervention damage: scaling neurons shifts the output distribution generically (in Cavus, suppression lowered accuracy and random sets also moved it). Every intervention needs random controls matched for damage (KL on neutral text) plus round-2 style gates.",
 "Label noise: alias matching misses paraphrases; verbose answers, partial answers and refusals blur the span. Audit labels by hand and pre-register refusal handling.",
 "Pseudo-replication: tokens within one text are autocorrelated. Resample at the prompt level and use prompt-blocked cross-validation.",
 "Multiple comparisons across 448 heads × 8-16 lags × conditions. Pre-register families and apply FDR.",
 "RoPE: a global position-id offset is a no-op, so only relative gaps test distance. Adding filler mixes distance with attention dilution (more keys); contexts must stay far below the trained maximum (max_position_embeddings 40,960).",
 "Probe capacity: channels differ in dimension and norm. Equalise them (PCA to a fixed dimension, inner-CV regularisation) before comparing arrow indices.",
 "Shared drivers in observational lead-lag: tokens and position drive all channels together. Use token embeddings and position as exogenous covariates and confirm with interventional transfer.",
 "Teacher forcing vs free generation: lenses and probes fit on reference text may not transfer to the model's own generations, so train and test on the model's own outputs."
]
```

## where_analogy_breaks (from the research stage; check against the verifier's notes above)
```json
[
 "'Gradient through routing': at inference no gradients flow. Qwen3-1.7B was not trained with MTP: it has no MTP weights, so 'MTP lives on gradient routing' does not describe how this model was trained. What a frozen model offers is the cross-position Jacobian, a forward sensitivity whose training-time counterpart is the off-diagonal gradient term of Wu et al. 'Gradient routing' (Cloud et al. 2024) is an unrelated technique that masks gradients to localise capabilities.",
 "The 'vector quantity' (a channel front predicting earlier vs later fronts) is degenerate for the channel's own fronts. On stationary stretches I(F_t;F_{t+k}) = I(F_t;F_{t−k}), and whitened single-lag linear predictability is symmetric. A measured same-channel asymmetry is non-stationarity, i.e. the length or position drift itself. Directional content exists only as token memory vs anticipation (the causal mask) and as cross-channel lead-lag or transfer.",
 "Channels are not separate conduits. Every head, subspace and neuron reads and writes one shared residual stream; heads are architectural units, while subspaces and SAE features are analyst choices. Any 'hierarchy of channels' depends on the basis, and the trivial depth and causal order (information goes up in depth and forward in position) must be factored out before calling anything a hierarchy.",
 "The front adds no dynamics of its own. With teacher forcing, the front series equals the per-position states computed in parallel under the causal mask. Temporal dynamics enter only through sampled tokens (token route) and through which cached states later positions read (KV route).",
 "The KV route is parallel: every later position reads cached fronts directly, with no relays. Only the token route is sequential. 'Token n follows from n−1' holds for the token route. Content written at block l is usable later only at blocks > l: the cross-position gradient is identically zero at block 27, so late-layer content can persist only via tokens.",
 "The published chain lens is a workspace reader, not a forecaster. It renders multi-token concepts latent in one hidden state, judged against the prompt and the response. Its 'hallucination' is confabulated readout content that grows with chain length, partly because an auxiliary head rolls the chain out. It has no break-off criterion, so k* has to be defined by us.",
 "H-neurons are an answer-level construct from short QA: mean CETT over answer tokens on consistency-filtered TriviaQA. Applying them per token over long chains is an extrapolation. Their causal signature is over-compliance (also sycophancy and jailbreaks), not hallucination per se, and the selection is not unique (Cavus et al.).",
 "'About 0.1 % of the net' is an upper bound. Reported H-sets are 0.001-0.035 % of FFN neurons, 2-60 neurons in Qwen3-1.7B. A 'field of view correlating mainly with H-neurons' is only meaningful as an excess over matched random neuron sets.",
 "An 'attractor' needs dynamics: persistence beyond matched controls, a basin, or bistability. A correlation between lens break-off and H-activity is not evidence of an attractor, and round 2 found the only demonstrated attractor to be a generic style direction.",
 "Sequential (DeepSeek MTP, a 'causal chain') vs parallel (Gloeckle, independent heads) MTP maps onto 'Folge' vs 'Reihe' only loosely. Both are training objectives of models that have such heads; Qwen3-1.7B has neither, so the distinction can only be emulated by the lenses (L-mtp vs L-par)."
]
```

## open_questions (from the research stage; check against the verifier's notes above)
```json
[
 "Did the user mean the LessWrong 'Chain Lens [interim]' by vgel & Raymond Douglas (11 Nov 2026)? If so, is the 'break-off' the reported growth of confabulation with chain length, or a new criterion such as the k* proposed here?",
 "What does 'Gradient Through Routing' refer to: DeepSeek-style sequential MTP gradients, cross-position attention gradients (pre-caching), or Cloud et al.'s gradient routing?",
 "Should the chain lens be built for Qwen3-1.7B (train an MTP head, ≈1-1.5 A100-h), or should round 1b add a Qwen3.5-2B side arm? Qwen3.5-2B ships an MTP module and Neuronpedia lists J-lens and J++ lenses for it, but it has a hybrid linear-attention architecture.",
 "Which hallucination is meant: the model's factual errors (the H-neuron sense) or lens confabulation (the WorkspaceBench sense)? The plan should keep them separate.",
 "Do H-neurons replicate in a 1.7B model with thinking off? Gao studied 4B-70B models and Cavus 4B models; neither studied a 1.7B model or any Qwen model. What detection margin counts as replication (Gao's +14 to +20 pp in-domain)?",
 "Which external conditions should reorder the hierarchy (prompt types, temperatures, context lengths), and how long is 'hyper-long', given the overlap with H-DUAL?",
 "Budget: all six operationalisations take ≈10-13 A100-h, the minimal path ≈6-7 h. How does H-CHAIN rank against the round-1b loop repairs (centred null, damage-normalised push, λ_RSI vs λ_max)?",
 "Are the Qwen3-Instruct SAEs (He et al. 2026) publicly accessible for Qwen3-1.7B, and at which layers? This is unverified; it decides whether SAE features can be used as channels."
]
```

## refs (research stage)
```json
[
 {
  "title": "H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs",
  "authors": "Cheng Gao, Huimin Chen, Chaojun Xiao, Zhiyi Chen, Zhiyuan Liu, Maosong Sun (Tsinghua)",
  "year": "2025 (arXiv 2512.01797; v1 1 Dec, v2 2 Dec 2025)",
  "url": "https://arxiv.org/abs/2512.01797",
  "finding": "Six instruct models (Mistral-7B-v0.3, Mistral-Small-3.1-24B, Gemma-3-4B/27B, Llama-3.1-8B, Llama-3.3-70B). Labels: TriviaQA, 10 samples (T 1.0, top-k 50, top-p 0.9), questions kept if 10/10 correct or 0/10 (non-refusal), 1,000 + 1,000; GPT-4o answer spans. CETT_{j,t} = ||W_down z_j e_j|| / ||W_down z_t||, mean over answer tokens and over other tokens; positives = hallucinated answer-token vectors (the printed equation and the opening of §6.1 state the opposite sign). L1 logistic regression; C chosen by held-out accuracy plus TriviaQA accuracy under suppression; H-neurons = positive weights. Ratios 0.01-0.35‰ (abstract: <0.1%). Accuracy TriviaQA 70.1-83.6% vs random 56.1-68.4%; Llama-3.1-8B NonExist 43.1 vs random 50.6. α ∈ [0,3] raises compliance on FalseQA, FaithEval, Sycophancy, Jailbreak (slopes ≈3.03 small vs ≈2.40 large models). Instruct probe transfers to base models (AUROC, figure only); low SFT drift (rank ≈0.97 Mistral-Small; >0.58, P<0.001). No layer distribution; no code link.",
  "relevance": "Defines H-neurons and the '<0.1%' figure; the identification and intervention protocol to replicate in Qwen3-1.7B (O4); shows the causal signature is over-compliance and that the construct is answer-level, short-QA.",
  "how_checked": "Downloaded and read the full arXiv HTML v2 (all sections incl. Methods 6.1-6.3); abs page via WebFetch."
 },
 {
  "title": "Hallucination Neurons and Where to Find Them: An Investigation into the existence of Hallucination Neurons",
  "authors": "Huseyin Cavus, Sebin Sabu, Joshua Spear, Jaskaran Singh Kawatra, Pavithra Rajendran",
  "year": "2026 (arXiv 2609.29781, 24 Sep 2026)",
  "url": "https://arxiv.org/abs/2609.29781",
  "finding": "Gemma-3-4B and MedGemma-4B (D = 348,160 FFN features), response-level labels, L1 C = 1.0: 2-9 H-neurons (0.006-0.026‰). Gemma AUROC 0.811 TriviaQA, 0.974 BioASQ, 0.628 NQ-Open. D1: 19/22 selected neurons have |r|>0.7 with non-selected ones; D2 bootstrap Jaccard 0.47/0.69/0.45; D3 L1∩L2 overlap 0.10/0.25/0.00; D5 TQA∩BioASQ 4 shared, BioASQ∩NQ 0, none universal. Causal: Gemma TriviaQA judge accuracy 0.472 (α=0) / 0.496 (α=1) / 0.490 (α=2), random same-layer 0.482, McNemar p<0.001; capability preservation not tested.",
  "relevance": "Detection replicates but neuron identity is not unique; prescribes the diagnostics (D1-D5) and treating H as a subspace/cluster in O4; shows causal effects are small and can lower accuracy.",
  "how_checked": "Downloaded and read the full arXiv HTML v1; abs page via WebFetch."
 },
 {
  "title": "hprobes (PyPI package): Hallucination neuron discovery and causal validation for transformer LLMs",
  "authors": "Huseyin Cavus et al. (GOSH DRIVE collaboration)",
  "year": "2026",
  "url": "https://pypi.org/project/hprobes/",
  "finding": "Third-party MIT-licensed implementation building on arXiv 2512.01797: dense and MoE H-neuron discovery, α-scaling, ablation, bootstrap stability; default l1_C 0.01; 'contrastive 3-vs-1 labeling at the generated answer token'. Not by the original authors.",
  "relevance": "Possible code base for O4 (must be checked against the paper; not the authors' release).",
  "how_checked": "Read the package description and metadata from the PyPI JSON API (pypi.org/pypi/hprobes/json). The GitHub repos hprobes / hprobes-protocol were not opened (unverified)."
 },
 {
  "title": "Confidence Regulation Neurons in Language Models",
  "authors": "Alessandro Stolfo, Ben Wu, Wes Gurnee, Yonatan Belinkov, Xingyi Song, Mrinmaya Sachan, Neel Nanda",
  "year": "2024 (NeurIPS 2024)",
  "url": "https://arxiv.org/abs/2406.16254",
  "finding": "Entropy neurons in the final-layer MLP have high output-weight norm and write mostly into the effective null space of W_U, acting via the final LayerNorm scale (≈80% of effect in GPT-2, LLaMA2-7B, Phi-2; ≈40% in Pythia, Gemma); token-frequency neurons shift logits toward/away from the unigram distribution; found in GPT-2, Pythia, Phi-2, Gemma 2B, LLaMA2 7B.",
  "relevance": "Confound/alternative reading for H-neurons (confidence regulation); overlap test in O4.",
  "how_checked": "arXiv abs page and HTML v2 via WebFetch (first 100k characters)."
 },
 {
  "title": "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model",
  "authors": "Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg",
  "year": "2023 (NeurIPS 2023 spotlight)",
  "url": "https://arxiv.org/abs/2306.03341",
  "finding": "Shifting activations along directions in a limited number of attention heads raises Alpaca's TruthfulQA truthfulness from 32.5% to 65.1%; tradeoff with helpfulness tuned by intervention strength.",
  "relevance": "Truthfulness directions as an alternative channel to H-neurons (head-level, O6).",
  "how_checked": "arXiv abs page via WebFetch. The often-cited probe-vs-generation gap was not found on the abs page (unverified)."
 },
 {
  "title": "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs",
  "authors": "Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, Yarin Gal",
  "year": "2024",
  "url": "https://arxiv.org/abs/2406.15927",
  "finding": "Linear probes on hidden states predict binarized semantic entropy, at the token before generation (TBG, last input token) and the second-last token (SLT); 'SEPs can capture semantic entropy even before generation'; Llama-2 7B/70B, Mistral 7B, Phi-3 Mini; generalize OOD better than accuracy probes.",
  "relevance": "Shows hallucination-proneness is readable from the prompt alone; motivates the before-generation test in O4 (H-neurons as familiarity/uncertainty signal).",
  "how_checked": "arXiv abs page and HTML v1 via WebFetch."
 },
 {
  "title": "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection",
  "authors": "Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, Jieping Ye",
  "year": "2024 (ICLR 2024)",
  "url": "https://arxiv.org/abs/2402.03744",
  "finding": "EigenScore uses eigenvalues of the covariance of several responses' internal embeddings to measure semantic consistency; test-time feature clipping truncates extreme activations to reduce overconfident generations.",
  "relevance": "Related internal-state signal (multi-sample); not directly used.",
  "how_checked": "arXiv abs page via WebFetch; which layer and token are used was not verified."
 },
 {
  "title": "Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation",
  "authors": "Yiming Wang, Pei Zhang, Baosong Yang, Derek F. Wong, Rui Wang",
  "year": "2024 (v2 2025)",
  "url": "https://arxiv.org/abs/2410.13640",
  "finding": "The 'chain' is the layer-wise trajectory h_0 → ... → h_L of output-token-averaged hidden states; magnitude and angle changes between adjacent layers separate correct from incorrect answers without labels.",
  "relevance": "Name clash: a chain over layers, not over positions; not the chain lens.",
  "how_checked": "arXiv HTML v2 via WebFetch (first 100k characters). Per-domain AUROC values are not cited (unverified)."
 },
 {
  "title": "Real-Time Detection of Hallucinated Entities in Long-Form Generation",
  "authors": "Oscar Obeso, Andy Arditi, Javier Ferrando, Joshua Freeman, Cameron Holmes, Neel Nanda",
  "year": "2025 (v2 Feb 2026)",
  "url": "https://arxiv.org/abs/2509.03531",
  "finding": "Token-level linear probes trained on entity-level labels from web search detect fabricated entities in long-form generation; Llama-3.3-70B AUC 0.90 vs 0.71 for semantic entropy; works across four model families; datasets released.",
  "relevance": "Closest existing token-level, long-form hallucination labeling; a template for validating per-token H-scores over position (O3, O5).",
  "how_checked": "arXiv abs page via WebFetch."
 },
 {
  "title": "Future Lens: Anticipating Subsequent Tokens from a Single Hidden State",
  "authors": "Koyena Pal, Jiuding Sun, Andrew Yuan, Byron C. Wallace, David Bau",
  "year": "2023 (CoNLL 2023)",
  "url": "https://arxiv.org/abs/2311.04897",
  "finding": "GPT-J-6B on the Pile; targets are the model's own greedy outputs, sampled at positions where the next-token prediction is correct. Best precision@1 (Table 2), learned prompt / fixed prompt / linear-to-hidden / linear-to-vocab: N=1 (t+2) 48.4/20.8/29.2/27.5; N=2 43.7/30.0/19.0/19.4; N=3 46.9/36.5/15.8/14.7; N=0 97.0/97.0/98.0/85.7; bigram 20.1% at N=1. Learned prompt peaks at middle layer 14 for N≥1. Calibrated accuracy at N=1: 26/57/77/95% for confidence bins 0-30/30-60/60-90/90-100%.",
  "relevance": "Baseline for direct (parallel) future-token lenses (O3 L-par) and for anticipation probes (O2); shows lens accuracy depends on model confidence (confound).",
  "how_checked": "Downloaded the arXiv PDF and read the extracted text (Sections 2-4, Table 2); abs page via WebFetch."
 },
 {
  "title": "Do Language Models Plan Ahead for Future Tokens?",
  "authors": "Wilson Wu, John X. Morris, Lionel Levine",
  "year": "2024 (COLM 2024)",
  "url": "https://arxiv.org/abs/2404.00859",
  "finding": "Pre-caching (deliberately computing features for later positions) vs breadcrumbs (incidental). Myopic training removes off-diagonal gradient terms. GPT-2 124M: vanilla 3.28 vs myopic 3.40 nats (bigram 5.33). Pythia: the gap grows with size from 70M (runs fine-tuned from checkpoints). Synthetic task shows strong pre-caching.",
  "relevance": "Training-time 'gradient through routing' across positions; sets expectation that future-usefulness in small models is mostly incidental.",
  "how_checked": "arXiv HTML v2 and abs page via WebFetch."
 },
 {
  "title": "Better & Faster Large Language Models via Multi-token Prediction",
  "authors": "Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, Gabriel Synnaeve",
  "year": "2024",
  "url": "https://arxiv.org/abs/2404.19737",
  "finding": "n independent output heads on a shared trunk; 13B models solve 12% more HumanEval and 17% more MBPP problems; 4-token models up to 3× faster at inference; favors induction heads and algorithmic reasoning on small tasks.",
  "relevance": "Parallel MTP ('Reihe'), contrasted with sequential MTP ('Folge').",
  "how_checked": "arXiv abs page via WebFetch."
 },
 {
  "title": "DeepSeek-V3 Technical Report (Section 2.2 Multi-Token Prediction)",
  "authors": "DeepSeek-AI",
  "year": "2024",
  "url": "https://arxiv.org/abs/2412.19437",
  "finding": "D sequential MTP modules (shared embedding and output head, one Transformer block, projection of [RMSNorm(previous depth); RMSNorm(Emb(future token))]) that 'keep the complete causal chain at each prediction depth'; D = 1; loss weight 0.3 for the first 10T tokens, 0.1 for the remaining 4.8T; modules discardable or usable for speculative decoding.",
  "relevance": "Template for the MTP head to train for Qwen3-1.7B (O3 L-mtp); best match to 'Folge'.",
  "how_checked": "arXiv HTML v2 MTP section via WebFetch (first 100k characters); the acceptance rate was not verified."
 },
 {
  "title": "On the Biology of a Large Language Model",
  "authors": "Jack Lindsey et al. (Anthropic)",
  "year": "2025 (27 Mar 2025)",
  "url": "https://transformer-circuits.pub/2025/attribution-graphs/biology.html",
  "finding": "Claude 3.5 Haiku: candidate rhyme words are represented at the newline before the line; injecting a 'green' planning feature makes 70% of 25 poems end with it. Hallucination: default 'can't answer' features are inhibited by known-entity/known-answer features; hallucinations are 'misfires' of this inhibition. Attention patterns are frozen in the replacement model.",
  "relevance": "Planning as a positive control for anticipation (O2, O3); a mechanism reading of hallucination (familiarity) that competes with 'over-compliance' (O4).",
  "how_checked": "Page read via WebFetch (poems, hallucination and methods passages; first and second 100k characters)."
 },
 {
  "title": "Eliciting Latent Predictions from Transformers with the Tuned Lens",
  "authors": "Nora Belrose, Igor Ostrovsky, Lev McKinney, Zach Furman, Logan Smith, Danny Halawi, Stella Biderman, Jacob Steinhardt",
  "year": "2023 (v6 Nov 2025)",
  "url": "https://arxiv.org/abs/2303.08112",
  "finding": "An affine probe per block of a frozen model decodes every hidden state to a vocabulary distribution; 'more predictive, reliable and unbiased than the logit lens'; models up to 20B; trajectories of latent predictions detect malicious inputs.",
  "relevance": "A single-token lens family the chain lens extends.",
  "how_checked": "arXiv abs page via WebFetch."
 },
 {
  "title": "Verbalizable Representations Form a Global Workspace in Language Models (Jacobian lens)",
  "authors": "Wes Gurnee, Nicholas Sofroniew, ..., Jack Lindsey (Anthropic)",
  "year": "2026 (6 Jul 2026)",
  "url": "https://transformer-circuits.pub/2026/workspace/index.html",
  "finding": "J_l = E_{t, t'≥t, prompt}[∂h_final,t'/∂h_{l,t}] over 1,000 prompts; lens(h) = softmax(W_U norm(J_l h)), i.e. it averages effects on present and future outputs; variants mentioned: present-only, frozen attention. Two-hop swap success 54% (Haiku 4.5), 70% (Sonnet 4.5), 70% (Opus 4.5); swapping the planned rhyme 'fight'→'light' changes earlier words. Code linked at github.com/anthropics/jacobian-lens; Neuronpedia hosts lenses.",
  "relevance": "Seed lens of the chain lens; basis for the lag-resolved Jacobian chain lens J_l(Δ) (O3 L-jac).",
  "how_checked": "Downloaded the HTML and read the Jacobian lens definition, methods, related-work and limitations passages; the GitHub repo was not opened."
 },
 {
  "title": "neuronpedia/jacobian-lens (Hugging Face model repo): Qwen3-1.7B J-lens",
  "authors": "Neuronpedia (fit with Anthropic's jlens code)",
  "year": "2026 (fit 2026-06-11)",
  "url": "https://huggingface.co/neuronpedia/jacobian-lens",
  "finding": "Contains qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt with config: WikiText-103, n_prompts 1000 requested, 466 fitted (early stop at Δ 0.002), max_seq_len 128, dim_batch 128, bf16, B200. J++ lenses exist for qwen3-4b and qwen3.5 models but not for qwen3-1.7b.",
  "relevance": "A ready J-lens for O3 (L-mtp seeds, positive controls); flags the domain mismatch (128-token WikiText vs chat generations).",
  "how_checked": "Hugging Face API file listing and the raw config.yaml (text) downloaded and read."
 },
 {
  "title": "The Chain Lens [interim]",
  "authors": "vgel, Raymond Douglas",
  "year": "2026 (LessWrong, 11 Nov 2026)",
  "url": "https://www.lesswrong.com/posts/Nr6fksEqqxNpdz6rk/the-chain-lens-interim",
  "finding": "Method: J-lens transport of a hidden state; top-K J-lens tokens as seeds; rollouts with the model's pretrained MTP head (Qwen3.5+, DeepSeek-V3/R1), contrastive decoding against a corpus-mean state, chain-only attention, optional hidden-state reinjection. WorkspaceBench macro: 0.42 vs J-lens 0.32 (28 families); 0.487 vs composite 0.411 (21 families); arithmetic chain intermediates 0.42 vs oracle lens 0.25. 'The chain lens confabulates more on WorkspaceBench’s hallucination family than single-token lenses. This grows with chain length.' Longer chains trade hallucination for performance. Controls: 10 chains > 50 J-lens tokens; foreign seeds and a corpus-mean state hurt; shuffle nulls. Neuron-lens round-trip criteria (99.8th percentile; >90% of random readouts). Cost ~100× single-token lens for 10 five-token chains. No break-off criterion.",
  "relevance": "Answers 'is chain lens an existing term': yes. It is the user's most probable referent and the method to adapt in O3 (L-mtp).",
  "how_checked": "Downloaded the full LessWrong HTML and read the entire post; extracted its outbound links."
 },
 {
  "title": "WorkspaceBench: Evaluating Interpretability Methods for the Global Workspace",
  "authors": "camilablank, agam_bhatia, Euan Ong, Neel Nanda",
  "year": "2026 (LessWrong/AF, 23 Sep 2026)",
  "url": "https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH/workspacebench-evaluating-interpretability-methods-for-the",
  "finding": "3,356 questions in 27 families for Qwen3.6-27B. Hallucination eval: an autorater judges whether concepts stated by the tool contradict verifiable information about the prompt context; headline metric is the share of readouts with at least one wrong claim.",
  "relevance": "Defines the 'hallucination' the chain lens is criticized for (lens confabulation, not model factual error).",
  "how_checked": "LessWrong page via WebFetch (summary-level read)."
 },
 {
  "title": "Gradient Routing: Masking Gradients to Localize Computation in Neural Networks",
  "authors": "Alex Cloud, Jacob Goldman-Wetzler, Evžen Wybitul, Joseph Miller, Alexander Matt Turner",
  "year": "2024",
  "url": "https://arxiv.org/abs/2410.04332",
  "finding": "Applies data-dependent, weighted masks to gradients during backpropagation to localize capabilities to subregions; applications: partitioned representations, robust unlearning by ablating a subregion, scalable oversight in RL.",
  "relevance": "Clarifies that 'gradient routing' is a different technique from gradient flow across positions.",
  "how_checked": "arXiv abs page via WebFetch."
 },
 {
  "title": "Arrows of Time for Large Language Models",
  "authors": "Vassilis Papadopoulos, Jérémie Wenger, Clément Hongler",
  "year": "2024 (ICML 2024)",
  "url": "https://arxiv.org/abs/2401.17505",
  "finding": "Forward models have lower loss than backward ones: English GPT2-Medium 2.88 vs 2.902 (+0.76%), French +2.65%, +0.6-3.2% across languages; the gap grows with model size (English Nano −0.33% to Medium +0.76%) and with context length; 'from an information-theoretic point of view, there should be no such difference'; explained via sparsity/computational complexity (prime factoring FW 22.2 vs BW 30.2 nats). No per-position natural-language analysis.",
  "relevance": "Forward vs backward predictability; supports the point that information-theoretic symmetry forces any asymmetry to be computational or non-stationary (O2).",
  "how_checked": "arXiv abs page and HTML v4 via WebFetch (two targeted reads)."
 },
 {
  "title": "Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs",
  "authors": "Sheridan Feucht, David Atkinson, Byron Wallace, David Bau",
  "year": "2024 (EMNLP 2024 per ACL Anthology listing in search results; not on abs page)",
  "url": "https://arxiv.org/abs/2406.20086",
  "finding": "Linear probes from h_{l,t} predict tokens at offsets −3..+1 (Llama-2-7b; Pile, 428k training tokens); next-token accuracy rises with depth while current and previous-token accuracy falls gradually; abrupt erasure at last tokens of multi-token words and entities (also Llama-3-8b).",
  "relevance": "Retrodiction (memory) probes for O2; expected depth profile of memory.",
  "how_checked": "arXiv abs page and HTML v3 via WebFetch."
 },
 {
  "title": "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation",
  "authors": "Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi",
  "year": "2023 (EMNLP 2023)",
  "url": "https://arxiv.org/abs/2305.14251",
  "finding": "'Error rates are higher for facts mentioned later in the generation', across all LMs, attributed to (a) earlier information being more frequent in pretraining (e.g. nationality, profession) and (b) error propagation; error rates also rise for rarer entities (PerplexityAI relative drops of 50%/64%).",
  "relevance": "Core length/position confound: position is confounded with content (O5).",
  "how_checked": "Downloaded the arXiv PDF and read the extracted text (Section 4.3, Fig. 2 caption); abs page via WebFetch."
 },
 {
  "title": "Long-form factuality in large language models",
  "authors": "Jerry Wei, Chengrun Yang, Xinying Song, et al., Quoc V. Le",
  "year": "2024 (NeurIPS 2024)",
  "url": "https://arxiv.org/abs/2403.18802",
  "finding": "LongFact (38 topics), SAFE (search-augmented per-fact evaluation), F1-style metric with a preferred response length; larger models are more factual. No position analysis in the abstract.",
  "relevance": "Long-form factuality evaluation tooling; no position evidence.",
  "how_checked": "arXiv abs page via WebFetch."
 },
 {
  "title": "How Language Model Hallucinations Can Snowball",
  "authors": "Muru Zhang, Ofir Press, William Merrill, Alisa Liu, Noah A. Smith",
  "year": "2023",
  "url": "https://arxiv.org/abs/2305.13534",
  "finding": "Models over-commit to early mistakes and produce further false claims; ChatGPT and GPT-4 identify 67% and 87% of their own mistakes when asked separately.",
  "relevance": "A content-driven amplification mechanism (not a pure length artifact); motivates injected-wrong-claim conditions (O5b).",
  "how_checked": "arXiv abs page via WebFetch."
 },
 {
  "title": "Learning to Break the Loop: Analyzing and Mitigating Repetitions for Neural Text Generation",
  "authors": "Jin Xu, Xiaojiang Liu, Jianhao Yan, Deng Cai, Huayang Li, Jian Li",
  "year": "2022 (NeurIPS 2022)",
  "url": "https://arxiv.org/abs/2206.02369",
  "finding": "'the more times a sentence is repeated in the context, the higher the probability of continuing to generate that sentence'; models prefer repeating the previous sentence; higher initial probability gives stronger self-reinforcement.",
  "relevance": "A documented non-hallucination attractor that can masquerade as an H-attractor; stratification control (O5).",
  "how_checked": "arXiv abs page via WebFetch."
 },
 {
  "title": "Same Task, More Tokens: the Impact of Input Length on the Reasoning Performance of Large Language Models",
  "authors": "Mosh Levy, Alon Jacoby, Yoav Goldberg",
  "year": "2024 (ACL 2024)",
  "url": "https://arxiv.org/abs/2402.14848",
  "finding": "Padding varying in length, type and location degrades reasoning 'at much shorter input lengths than their technical maximum'; next-word prediction correlates negatively with reasoning performance.",
  "relevance": "Filler vs coherent padding design for separating length from content (O5b).",
  "how_checked": "arXiv abs page via WebFetch (padding details beyond the abstract unverified)."
 },
 {
  "title": "Efficient Streaming Language Models with Attention Sinks",
  "authors": "Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis",
  "year": "2023 (ICLR 2024)",
  "url": "https://arxiv.org/abs/2309.17453",
  "finding": "Strong attention to initial tokens as a 'sink' regardless of semantics; keeping initial tokens' KV recovers window attention; stable modeling up to 4M tokens; a placeholder sink token in pretraining helps.",
  "relevance": "Position confound at the first tokens; exclusion rule for fronts and probes.",
  "how_checked": "arXiv abs page via WebFetch."
 },
 {
  "title": "Massive Activations in Large Language Models",
  "authors": "Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu",
  "year": "2024 (COLM 2024)",
  "url": "https://arxiv.org/abs/2402.17762",
  "finding": "Very few activations ~100,000× larger than others, largely constant regardless of input, acting as indispensable biases and concentrating attention on their tokens.",
  "relevance": "Confound for norm-normalized measures (CETT) and for position effects.",
  "how_checked": "arXiv abs page via WebFetch."
 },
 {
  "title": "Lost in the Middle: How Language Models Use Long Contexts",
  "authors": "Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang",
  "year": "2023 (TACL)",
  "url": "https://arxiv.org/abs/2307.03172",
  "finding": "Performance is highest when relevant information is at the beginning or end of the context and degrades in the middle, even for long-context models.",
  "relevance": "Position-of-content effects in long contexts (O5, O6 context-length condition).",
  "how_checked": "arXiv abs page via WebFetch."
 },
 {
  "title": "RoFormer: Enhanced Transformer with Rotary Position Embedding",
  "authors": "Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu",
  "year": "2021",
  "url": "https://arxiv.org/abs/2104.09864",
  "finding": "RoPE encodes absolute position with a rotation matrix and incorporates explicit relative position dependency in self-attention, with decaying inter-token dependency as relative distance grows.",
  "relevance": "Implies a uniform position-id offset is a no-op; only relative gaps can test position (O5a).",
  "how_checked": "arXiv abs page via WebFetch."
 },
 {
  "title": "PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wise Training",
  "authors": "Dawei Zhu, Nan Yang, Liang Wang, Yifan Song, Wenhao Wu, Furu Wei, Sujian Li",
  "year": "2023 (ICLR 2024)",
  "url": "https://arxiv.org/abs/2309.10400",
  "finding": "Skipping biases on position indices inside a fixed window 'smartly simulates long inputs', decoupling position indices from content length.",
  "relevance": "Precedent for manipulating position_ids without content (O5a).",
  "how_checked": "arXiv abs page via WebFetch."
 },
 {
  "title": "A Mathematical Framework for Transformer Circuits",
  "authors": "Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, et al. (Anthropic)",
  "year": "2021",
  "url": "https://transformer-circuits.pub/2021/framework/index.html",
  "finding": "Components 'communicate with each other by reading and writing to different subspaces of the residual stream'; paths through the model 'are linear if one freezes the attention patterns'.",
  "relevance": "Basis for defining channels as heads/subspaces and for Jacobian chains through attention (O1, O3, O6).",
  "how_checked": "Page via WebFetch (first 100k characters; the attention-heads section body was truncated)."
 },
 {
  "title": "Information Flow Routes: Automatically Interpreting Language Models at Scale",
  "authors": "Javier Ferrando, Elena Voita",
  "year": "2024 (EMNLP 2024 per search results; abs page lists no venue)",
  "url": "https://arxiv.org/abs/2403.00824",
  "finding": "Builds per-prediction graphs of token representations (nodes) and operations (edges), keeping the most important by attribution in a single forward pass; Llama 2.",
  "relevance": "Attribution-based routing channels across positions (O6).",
  "how_checked": "arXiv abs page via WebFetch."
 },
 {
  "title": "Localizing Model Behavior with Path Patching",
  "authors": "Nicholas Goldowsky-Dill, Chris MacLeod, Lucas Sato, Aryaman Arora",
  "year": "2023",
  "url": "https://arxiv.org/abs/2304.05969",
  "finding": "Path patching expresses and tests hypotheses that a behavior is localized to a set of paths (interactions between components); applied to induction heads and GPT-2.",
  "relevance": "Interventional edge estimation for the head hierarchy (O6).",
  "how_checked": "arXiv abs page via WebFetch."
 },
 {
  "title": "Transfer Entropy as a Measure of Information Flow in VLMs and LLMs",
  "authors": "Jessica E. Liang, Jianbo Shi",
  "year": "2026 (NeurIPS 2026 poster)",
  "url": "https://neurips.cc/virtual/2026/poster/151650",
  "finding": "Tractable transfer-entropy estimators for high-dimensional hidden states at the granularity of layers, tokens and attention heads; distinguishes 'sinks, broadcasters, and benign components'; TE-guided pruning beats attention/saliency heuristics.",
  "relevance": "Observational directed-flow estimator for the channel hierarchy (O6).",
  "how_checked": "NeurIPS virtual poster page (abstract) via WebFetch; the estimator details were not read."
 },
 {
  "title": "Discovering Millions of Interpretable Features with Sparse Autoencoders (Qwen3-Instruct SAE)",
  "authors": "XinYang He, Wei Wang, Bing Zhao, Xuan Ren, WenBo Li, WeiXu Qiao, Hu Wei, Lin Qu",
  "year": "2026 (arXiv 2606.26620)",
  "url": "https://arxiv.org/abs/2606.26620",
  "finding": "SAEs for instruction-tuned Qwen3 1.7B/4B/8B; for 1.7B and 4B, layer-wise SAEs at residual stream, MLP output and attention output.",
  "relevance": "Possible SAE-feature channels for Qwen3-1.7B (O1, O6).",
  "how_checked": "arXiv abs page via WebFetch; layer coverage and release location unverified."
 },
 {
  "title": "Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models",
  "authors": "Boyi Deng, Xu Wang, et al., Jingren Zhou",
  "year": "2026 (arXiv 2605.11887)",
  "url": "https://arxiv.org/abs/2605.11887",
  "finding": "14 groups of SAEs across 7 model variants of Qwen3 and Qwen3.5 (dense and MoE).",
  "relevance": "Alternative SAE source; whether Qwen3-1.7B is covered is unverified.",
  "how_checked": "arXiv abs page via WebFetch."
 },
 {
  "title": "Qwen/Qwen3-1.7B: config.json and model.safetensors.index.json",
  "authors": "Qwen team",
  "year": "2025",
  "url": "https://huggingface.co/Qwen/Qwen3-1.7B",
  "finding": "28 layers, hidden 2048, intermediate 6144 (172,032 FFN neurons), 16 query / 8 KV heads, head_dim 128, q_norm/k_norm present, RoPE θ 1e6, max_position_embeddings 40960, tied embeddings, SiLU; 311 weight tensors, none belonging to an MTP module.",
  "relevance": "Neuron counts for the H-neuron ratio; confirms that no MTP head exists for the chain lens (O3, O4).",
  "how_checked": "Downloaded and parsed the raw config.json and weight index."
 },
 {
  "title": "Qwen/Qwen3.5-2B: config.json and model.safetensors.index.json",
  "authors": "Qwen team",
  "year": "2026",
  "url": "https://huggingface.co/Qwen/Qwen3.5-2B",
  "finding": "Ships an MTP module (mtp.fc, pre_fc_norm_embedding/hidden, one layer with attention and MLP; mtp_num_hidden_layers 1); hidden 2048, 24 layers with a 3:1 linear/full attention pattern.",
  "relevance": "A side arm where the published chain lens works without training a head (O3, open question).",
  "how_checked": "Downloaded and parsed the raw config.json and weight index."
 },
 {
  "title": "MicroLens (Zahedi, Zhu, Wu) and J++ lens (Ayonrinde & Lindsey, LessWrong 2026)",
  "authors": "as listed in runde1b_eingaben/README.md",
  "year": "2026",
  "url": "https://arxiv.org/abs/2611.13322",
  "finding": "Per the project README: MicroLens reads states via logit and J-lens and edits by swaps ('readable is not used'); J++ uses expert Jacobians per activation group and filters bad groups.",
  "relevance": "Placement only: single-token lenses that the chain lens extends; MicroLens++ (local Jacobian) corresponds to the per-context variant of L-jac.",
  "how_checked": "Not re-opened (unverified here); taken from results/ju_space/plans/runde1b_eingaben/README.md and analysis/ju_space/microlens/README.md as instructed."
 }
]
```
