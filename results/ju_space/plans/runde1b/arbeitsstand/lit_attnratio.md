# lit:attnratio

## Verifier summary
I checked all 26 references against their primary pages, including some outside arXiv:
- the NeurIPS PDF;
- PMC;
- the LessWrong post, fetched through its GraphQL API;
- the Hugging Face config;
- OpenReview for venues;
- the GitHub READMEs.

All 26 exist, with correct titles, authors, years and URLs. Missing author lists for ReDeEP, Men et al. and Orgad et al. were filled in.

Eight entries had content errors:
- **Lookback Lens:** the in-domain QA and Sum labels are swapped.
- **INSIDE:** the AUROCs use sentence-similarity labels, not ROUGE-L. The claimed Table 5 contradiction is a protocol difference.
- **ReDeEP:** PCC 0.41/0.46 correlate per-head maps, not labels. The AUC range omits LLaMA3-8B; it is 0.7285-0.8420.
- **Hallucinate at the Last:** the Qwen sliding-window effect is +3.25/+3.75, giving sensitivities 2.75/2.5, and the source's own text and table disagree.
- **H-Neurons:** CETT's denominator is the FFN output norm, and the neuron count depends on tuning.
- **LLM-Check:** the RAGTruth figure was selective.
- **SEP:** the probe uses the second-last token.
- **Retrieval heads:** the repo is nightdessert/Retrieval_Head.

The reasoning in sections 5-7 and in the original implications list had several flaws:
- the same-channel "no arrow" argument needs stationarity;
- attention terms were applied to FFN channels;
- the third MTP substitute (rolling out the transported final-layer state) does not work;
- one control was attributed to the post although it is not there, and the post's neutral-seed result was left out;
- the null band was not matched for selection;
- the 66 AUROC is a cross-task figure;
- Arrows of Time was used as a prediction about J-lens inverses, which it is not;
- the round-2 compliance link was unsupported.

The corrected report fixes all of these and adds a corrected implications section and a list of changes.

## Problems found by the verifier
- Lookback Lens (report_md section 2): the in-domain labels are swapped. 91.2 is QA (NQ) and 88.8 is Sum (CNN/DM); Table 2 rows are indexed by source task.
- INSIDE: the numbers are mislabelled. 80.4 / 81.5 / 76.5 / 82.7 use sentence-similarity labels, not ROUGE-L > 0.5 (with ROUGE-L they are 80.8 / 81.2 / 77.1 / 82.9). 'Table 5 contradicts Table 1' is wrong: Table 5 uses a different protocol (Duan et al., N = 5 for LLaMA).
- ReDeEP: 'Pearson correlation with labels is 0.41 / 0.46' is wrong. Those values correlate per-head maps (copying-head score vs ECS difference, and vs ECS-label correlation). The AUC range 0.7458-0.8420 omits LLaMA3-8B: chunk-level is 0.7285-0.8420.
- Hallucinate at the Last: 'without it the drop becomes +3.25 to +3.75' misreads Table 5. These are increases; the resulting sensitivities are 2.75 (long) and 2.5 (standard). The paper's text says 'to 3.75 / 3.25', a further internal inconsistency. The 16K sliding window was set by the authors.
- H-Neurons: the CETT denominator is the norm of the whole FFN output (W_down z_t), not the hidden or residual state. The main text selects non-zero weights, while the figure and appendix select positive weights. The neuron count follows from a grid-searched l1 strength, so '< 0.1 %' and the extrapolated 2-60 neurons for Qwen3-1.7B are tuning-dependent, not intrinsic.
- LLM-Check: 'RAGTruth about 58' is selective. The attention score is 54.19 white-box and 58.30 black-box (weighted average); the hidden score is 54.11 / 57.24.
- SEP: 'at the last token' should read 'second-last token (SLT)'.
- Lookback Lens head roles (positive heads ground, negative heads keep self-consistency) are the authors' conjecture, stated as fact in the report.
- Retrieval heads: the repo is nightdessert/Retrieval_Head (it exists; 'RetrievalHead' does not). Sink attention is reported for hallucinated generations, not as a consequence of masking.
- Section 5.3 misattributes the controls. The post has foreign-seed and corpus-mean controls plus a judge-level shuffle null; it has no 'shuffled state' control. It also reports that neutral stop-word seeds perform about as well as J-lens seeds, so neutral seeds cannot serve as a null.
- Section 5.3: the null band for 'contrastive evidence' must apply the same contrastive argmax to null states. Otherwise selection bias applies, since the chosen token maximises that contrast by construction.
- Section 5.2: 'no arrow by definition' holds only for a stationary (shift-invariant) sequence. I(c_t; c_t+k) vs I(c_t; c_t-k) compares different pairs, so within 40-token answers (sink, template, position effects) any asymmetry is non-stationarity, a confound. 'An arrow exists only in three cases' omits non-stationarity.
- Section 5.2: 'Causality alone makes the past more decodable than the future, so raw ratios carry no information' is overstated and conflicts with Feucht et al., where previous-token information is erased with depth while next-token information grows. Raw ratios mix structural asymmetry with depth; they need normalisation, not discarding.
- Section 5.2: the arrow vector includes LR_c, D_c and S_c, but channels include FFN-neuron groups, for which these attention quantities are undefined.
- Section 5.1 and implication 4: 'teacher-forced rollout from the J-lens-transported state' is incoherent. Later positions never read a position's final-layer output, so a transported final-layer state cannot be rolled out without an MTP-like module. The no-new-weights substitute is a Patchscopes-style patch of the layer-l state into the unmodified model.
- Section 5.2: 'Arrows of Time predicts that [J-lens Jacobian] inverses are dense and hard to estimate' overreaches. The paper's sparse-forward/dense-inverse argument concerns sequence prediction and is at most an analogy. The user's rule alone justifies being inverse-free.
- Section 5.3 and implication 6: '66 AUROC with 8-token windows' is the cross-task figure. In-domain windows reach 85.3-87.1, so the power expectation should be stated for out-of-domain use (round-2 'Continue.' texts).
- Section 5.6 and implication 12: 'which is what round 2 saw' is unsupported. Round 2 found a generic style direction (sea, tone, sport); compliance was never measured, so the H-neuron/compliance link is a hypothesis.
- Section 4: plan-ahead's diagonal/off-diagonal decomposition is called 'the exact form' of the user's 'gradients routed across positions'. The user's phrase ('Gradient Through Routing') is ambiguous and may also mean gradient routing (Cloud et al.).
- Section 5.5: a readout rise after a wrong commitment can also be snowballing or error propagation. An attractor claim needs persistence or return after the wrong claim stops being referenced, not only a contrast with equal-length neutral text.

## Per-reference / per-fact verdicts
- [verified] The Chain Lens [interim]: Re-fetched through the LessWrong GraphQL API: title, authors (vgel; coauthor Raymond Douglas), date 2026-11-11 and all quoted numbers match (0.420 vs 0.319, 28 families; 0.487 vs 0.411, 21; 0.42 vs 0.25; +0.02-0.03; 99.8th percentile; 90 %; 57 %; 1.1M neurons). Precision notes: (a) the post names Qwen3.6-27B only for the Neuron Lens part. (b) The 0.42 vs 0.25 row is the 'chain intermediates' family; 'arithmetic intermediates' is 0.40 vs 0.35. (c) Neutral stop-word seeds perform about as well as J-lens seeds. (d) The shuffle null is judge-level, not a shuffled state. (e) Hallucination family (lower is better): chain lens 0.45 vs J-lens 0.32.
- [corrected] Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps: Metadata, code link and all numbers verified. Error in report_md: the in-domain labels are swapped. Table 2 rows are indexed by source task, so 91.2 is in-domain QA (NQ) and 88.8 is in-domain Sum (CNN/DM); in windows, 87.1 is QA and 85.3 is Sum. The 7B->13B figure 73.5 vs 84.0 is QA->Sum on spans, and 84.0 is a 13B-trained detector. The paper only conjectures the grounding and self-consistency roles of positive and negative heads.
- [corrected] LLM-Check: Investigating Detection of Hallucinations in Large Language Models: The given URL resolves to the right NeurIPS 2024 page, and authors and FAVA numbers are verified. 'RAGTruth about 58' is selective. With Llama-2-7b as detector, the attention score is 54.19 white-box and 58.30 black-box (weighted average over 4 generators), and the hidden score is 54.11 / 57.24. Entropy 77.00 vs attention 69.57 is Llama-2 on the FAVA train split. The -5.42 vs -5.85 contrast is one illustrative example (Appendix C, Table 6). The repo README names the paper.
- [verified] Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation: All numbers are CoE-C averages over 7 models: 75.22 vs best baseline 66.08 (Entropy); 63.54 / 63.89 / 64.98. Execution time is 1.12e-3 s. Venue ICLR 2025 confirmed on OpenReview.
- [corrected] INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection: 80.4 / 81.5 / 76.5 / 82.7 are AUC_s, i.e. against sentence-similarity labels (cosine > 0.9). Against ROUGE-L > 0.5 labels, LLaMA-7B gets 80.8 / 81.2 / 77.1 / 82.9. Table 5 (Appendix B) is not a contradiction of Table 1: it follows Duan et al.'s protocol (N = 5 generations for LLaMA).
- [corrected] Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs: The probe positions are the token before generation (TBG) and the second-last token (SLT), not 'the last token'. TBG is 'slightly below' SLT. The ranges -2.0 to +2.8 (Table 1) and +2.2 to +10.5 (Table 2) are verified. The repo README names the paper. Venue: preprint and ICML 2024 FM-Wild workshop.
- [verified] Detecting hallucinations in large language models using semantic entropy: Nature 630 (2024), PMC11186750. Verified: ten generations, bidirectional-entailment clustering, mean AUROC 0.790 vs 0.691 / 0.698 / 0.687 over 30 task-model combinations, and the quote 'does not help when LLM outputs are systematically bad'.
- [corrected] H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs: Table 1 numbers, the 0.01-0.35 per-mille range, the six models and alpha in [0,3] are verified. CETT's denominator is the norm of the whole FFN output W_down z_t (the paper calls it the hidden state), not the residual stream. Training negatives also include all non-answer tokens. The main text says non-zero weights, while Fig. 1 and the appendix say positive weights. The l1 strength C is grid-searched against suppression damage, so the neuron count is a tuning outcome. Both reported internal inconsistencies are confirmed (label equation; Llama-3.1-70B in text vs Llama-3.3-70B in table). The repo thunlp/H-Neurons calls itself the official implementation.
- [corrected] ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability: Authors: Zhongxiang Sun, Xiaoxue Zang, Kai Zheng, Yang Song, Jun Xu, Xiao Zhang, Weijie Yu, Yang Song, Han Li. ICLR 2025 (Spotlight). PCC 0.41 / 0.46 are correlations between the per-head copying-head score map and the per-head ECS-difference and ECS-label-correlation maps; they are not correlations with labels. The AUC range omits LLaMA3-8B: chunk-level is 0.7285-0.8420 and token-level 0.6701-0.8181 (RAGTruth and Dolly-AC). The repo README names the paper.
- [corrected] Retrieval Head Mechanistically Explains Long-Context Factuality: The 3-6 % (score > 0.1), 11 models incl. Qwen1.5-14B(-Chat), and K = 50 -> below 50 figures are verified. Venue: ICLR 2025 (Oral). Code is nightdessert/Retrieval_Head; the paper's rendered link lacks the underscore, and the README names the paper. The sink observation concerns hallucinated generations ('retrieval heads predominantly attended to the initial token'); it is not a consequence of masking.
- [verified] Real-Time Detection of Hallucinated Entities in Long-Form Generation: All numbers are verified: floor(0.95 L); 84 % agreement (n = 50); 80.6 % recall; 15.8 % FPR; LongFact AUCs; Qwen-2.5-7B LoRA 0.8947; 0.02-0.04 transfer; about 50 % of correct answers sacrificed. Code: obalcells/hallucination_probes.
- [verified] LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations: Full authors: Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, Yonatan Belinkov. ICLR 2025 confirmed. Probe@exact 0.85 vs last generated token 0.71 (Mistral-7B-Instruct, TriviaQA) verified; cross-task failure is stated.
- [verified] How Language Model Hallucinations Can Snowball: Verified: 98.40 / 95.67 %, 67.37 / 87.03 %, 9.40 / 3.87 %, 44.25 / 94.90 %, 500 questions per task, and the code link.
- [verified] FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation: Quote, both hedged causes, < 2 % estimator error and 80 % -> 16 % are verified. 'All LMs' in the position finding means the three human-annotated LMs (InstructGPT, ChatGPT, PerplexityAI).
- [verified] Analyzing and Mitigating Object Hallucination in Large Vision-Language Models: Verified: 200 COCO 2014 test images described by MiniGPT-4 (manually annotated), 'towards the end of the sequence', and a high-uncertainty concentration.
- [corrected] Hallucinate at the Last in Long Response Generation: A Case Study on Long Document Summarization: Bins, 0.81 -> 0.75, 0.85 -> 0.79, 0.83 -> 0.76, human 0.90 -> 0.70, about 3x attention and 20 examples are verified. Qwen2.5-7B ran with a 16K sliding window that the authors configured, giving sensitivity -0.50 (long) and -1.25 (standard). Without the window, sensitivity rises by +3.25 / +3.75 to 2.75 / 2.5 (Table 5); the text's 'to 3.75 / 3.25' contradicts that table. The 5.50 vs 10.3 inconsistency (Table 2 vs Table 6) is confirmed.
- [verified] The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs: Verified: agreement 0.723 vs 0.142; Mean-Len 0.721 / 0.713; EigenScore 0.747 / 0.694; SE 0.773 / 0.732; Pearson 0.826 / 0.894. The models are Llama3.1-8B-Instruct and Mistral-7B-Instruct-v0.3.
- [verified] Future Lens: Anticipating Subsequent Tokens from a Single Hidden State: Verified: 48.4 / 43.7 / 46.9 (learned), 29.2 -> 15.8 (linear hidden-state model), bigram 20.1, layer 14 of GPT-J-6B, and future.baulab.info.
- [verified] Do language models plan ahead for future tokens?: Verified: myopia gap 3.40 - 3.28 = 0.12 (cross-entropy), the gap growing with Pythia size from 70M, the myopic model better early and worse later, and the code link. Calling it 'the exact form' of the user's phrase is an overclaim; see claim problems.
- [verified] Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs: Verified: offsets -3..+1, Llama-2-7b and Llama-3-8b, early-layer erasure at the last tokens of multi-token words and entities, EMNLP 2024. The general depth trend (next-token up, previous/current down) is only visible in figures and is not stated in the text.
- [verified] Arrows of Time for Large Language Models: Verified: +0.61 % to +3.17 % across 8 languages (Medium/XL), French +0.15 % (4.9M) -> +2.65 % (405M), growth with context length, the sparsity/complexity explanation, the code link and ICML 2024. Using it as a 'prediction' about J-lens Jacobian inverses is an analogy only.
- [verified] Emergent Response Planning in LLMs: Verified: MLP probes on prompt representations, the three-phase dynamics, within-family scaling including Qwen-2.5-Instruct 1.5B/32B/72B, figures only, and ICML 2025.
- [verified] Unlocking the Future: Exploring Look-Ahead Planning Mechanistic Interpretability in Large Language Models: Full authors: Tianyi Men, Pengfei Cao, Zhuoran Jin, Yubo Chen, Kang Liu, Jun Zhao. Verified: step-1 -> step-6 probe accuracy 0.51, and MHSA extraction above MLP in layers 15-20 (fine-tuned Llama-2-7b-chat and Vicuna-7B).
- [verified] Better & Faster Large Language Models via Multi-token Prediction: The abstract confirms 12 % HumanEval, 17 % MBPP and up to 3x faster inference. Venue: ICML 2024.
- [verified] Gradient Routing: Masking Gradients to Localize Computation in Neural Networks: The abstract confirms a training method using data-dependent gradient masks.
- [verified] Qwen3-1.7B config.json: Re-downloaded and confirmed: 28 layers, 16 heads, 8 KV heads, head_dim 128, hidden 2048, intermediate 6144, use_sliding_window false, tie_word_embeddings true, and no MTP/nextn key. Hence 448 heads and 172,032 FFN neurons. A Qwen3-1.7B J-lens exists on Hugging Face (neuronpedia/jacobian-lens, qwen3-1.7b/jlens/...).

## Corrected report
# H-CHAIN supplement: attention/trajectory hallucination readouts, position effects, future-token information, and a chain lens for Qwen3-1.7B

**Sources.** Every source below was opened and read:
- arXiv abstract and HTML pages;
- the LLM-Check NeurIPS PDF, with text extracted locally;
- the Nature paper, via PMC;
- the LessWrong post, rendered with Playwright and re-fetched during verification through the LessWrong GraphQL API;
- Qwen3-1.7B's `config.json`.

An independent verifier re-opened every source on 13 Nov 2026 and made the corrections listed at the end. Numbers are as printed in the sources. Sections 5 and 7 are derivations. Anything not opened is marked *unverified*.

**Local text copies:**
- /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/hchain_lit/dl_chainlens/chainlens.txt
- /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/hchain_lit/dl_llmcheck/llmcheck.txt
- The verifier's copies are under /tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/vattn/:
  - dl_ARXIVID/full.txt (one per arXiv paper);
  - dl_llmcheck/paper.txt;
  - dl_pmc_se/pmc.txt;
  - dl_lw/post.txt;
  - dl_qwencfg/config.json.

## 1. "Chain lens" is an existing method, and "hallucinate" means two things

The source is "The Chain Lens [interim]" (vgel, Raymond Douglas; LessWrong, 11 Nov 2026). It is already listed in `runde1b_eingaben/README.md` §6.

- **Method:**
  - The J-lens transports a hidden state into the final-layer basis.
  - The top-K J-lens tokens seed chains, which the model's pretrained MTP head rolls out. An MTP head predicts the token after next from a final-layer state and the next token's embedding.
  - Refinements:
    - contrastive decoding against a prior state averaged over many hidden states (the largest gain, about +0.02-0.03 over greedy decoding);
    - the MTP head attends only to its own chain;
    - the transported state is re-injected.
  - In the post's ablations, chain-only attention and re-injection were not significant.
- **Results (WorkspaceBench):**
  - Macro score 0.420 vs 0.319 for the J-lens (28 families).
  - 0.487 vs 0.411 for a hindsight composite of untrained lenses (21 families).
  - Chain intermediates: 0.42 vs 0.25 for the trained oracle lens. The TLDR calls this family "arithmetic chain intermediates"; the separate arithmetic-intermediates family scores 0.40 vs 0.35.
  - Controls:
    - 10 chains beat 50 matched J-lens tokens.
    - Neutral stop-word seeds perform about as well as J-lens seeds (with the J-lens seed swapped back in for judging).
    - Foreign seeds and a corpus-mean state hurt.
    - Shuffle nulls show that the readouts are not merely suggestive text for the judge.
- **Hallucination:** "The chain lens confabulates more on WorkspaceBench's hallucination family than single-token lenses. This grows with chain length."
  - On that family (lower is better), the chain lens scores 0.45 and the J-lens 0.32.
  - Also: "Hallucination and performance both scale with chain length" (19 families).
- **Neurons:**
  - The lens was applied to the write vectors (both poles) of all 1.1M MLP neurons of Qwen3.6-27B. The post names the model only for this part.
  - A readout "round-trips" if two conditions hold:
    - its best chain drives the neuron above the neuron's 99.8th-percentile WikiText activation;
    - its top-3 chains beat 90 % of random readouts of other neurons from the same layer.
  - A judge picks the matching steered text 57 % of the time (four options).
- **Requirement:** an MTP head. The post says "Qwen3.5+" models ship one (*unverified*).

The user's break-off "where it hallucinates" therefore mixes two quantities, which need separate labels:
- **H_lens:** the readout confabulates. This is what grows with chain length in the post.
- **H_model:** the model's own text is wrong. This is what H-neurons and the detectors below target.

## 2. Detectors that read attention or hidden-state trajectories

- **Lookback Lens** (Chuang, Qiu, Hsieh, Krishna, Kim, Glass; EMNLP 2024).
  - Method:
    - Per head and step, LR = A_ctx/(A_ctx+A_new), i.e. mean attention per context token vs per generated token.
    - LR is averaged over a span or an 8-token window.
    - A logistic regression over all L×H ratios (1,024 for Llama-2-7B) predicts contextual hallucination.
  - Labels: GPT-4o span annotations (97 % agreement with humans in a 70-example pilot).
  - AUROC, annotated spans:
    - In-domain (two-fold): 91.2 (QA, NQ) / 88.8 (Sum, CNN/DM).
    - Cross-task: 85.3 (QA→Sum) and 82.0 (Sum→QA). A layer-28 hidden-state probe gets 83.6 / 84.7.
  - AUROC, 8-token windows:
    - In-domain: 87.1 (QA) / 85.3 (Sum).
    - Cross-task: 66.1 / 66.0, against at most 59.4 for hidden states.
  - Transfer 7B→13B via a fitted linear head map (QA→Sum, spans): 73.5, vs 84.0 for a detector trained on 13B.
  - Heads:
    - Top-10 heads give 71.2 / 79.2 and all heads 85.3 / 82.0, so the signal is spread out.
    - The authors conjecture that positive-weight heads do context grounding and negative-weight heads keep the model consistent with its own text.
    - Layers 13-20 are slightly best.
  - Guided decoding (8 sampled 8-token chunks):
    - XSum: 49.0 → 58.6 % correct, with hallucinated outputs 510 → 414.
    - NQ: 71.2 → 74.2.
  - No position analysis. Models: LLaMA-2-7B/13B-Chat. Code: voidism/Lookback-Lens.
- **LLM-Check** (Sriramanan, Bharti, Sadasivan, Saha, Kattakinda, Feizi; NeurIPS 2024).
  - Setting: one response, teacher-forced through the model or a proxy; no training.
  - Attention score: causal maps are lower-triangular, so log det = Σ log of the diagonal (a token's weight on itself). The mean log-det per head is aggregated over heads.
  - Hidden score:
    - mean log singular value² of the token-by-token Gram matrix HᵀH;
    - normalised "to remove the explicit dependence on input-length".
  - A windowed logit entropy is also used.
  - FAVA-Annotation AUROC:
    - attention score 72.34 (Llama-2-7B, layer 21), 71.69 (Vicuna-7B), 68.19 (Llama-3-8B);
    - hidden score 57.10-58.44;
    - INSIDE 59.03.
  - RAGTruth, with Llama-2-7b as the detector:
    - attention score 54.19 white-box and 58.30 black-box (a weighted average over four generator models);
    - hidden score 54.11 / 57.24.
  - On synthetic inserted errors (FAVA train split), window entropy wins: 77.00 vs 69.57 for Llama-2.
  - In one illustrative example (Appendix C), hallucinated tokens have larger log-diagonals (means −5.42 vs −5.85).
  - Cost: 0.22 s per example for 32 layers. Code: GaurangSriramanan/LLM_Check_Hallucination_Detection.
- **ReDeEP** (Sun, Zang, Zheng, Song, Xu, Zhang, Yu, Song, Li; ICLR 2025).
  - Method: an External Context Score (ECS) from copying heads, set against a Parametric Knowledge Score from late FFNs.
    - ECS is the cosine between a token's last-layer state and the mean last-layer state of its top-10 %-attended context tokens.
    - The Parametric Knowledge Score is the Jensen-Shannon divergence of logit-lens distributions before and after the FFN.
  - In Llama-2-7B, 1006 of 1024 heads have higher ECS on truthful responses, and most heads' ECS correlates negatively with the hallucination label.
  - Per head, the copying-head score correlates with the ECS difference (PCC 0.41) and with the ECS-label correlation (0.46). The hallucination-relevant heads are therefore mostly copying heads.
  - AUC on RAGTruth and Dolly (AC), for LLaMA2-7B/13B and LLaMA3-8B: chunk-level 0.7285-0.8420, token-level 0.6701-0.8181.
  - Code: Jeryi-Sun/ReDEeP-ICLR.
- **Retrieval heads** (Wu, Wang, Xiao, Peng, Fu; 2024; ICLR 2025). This is causal evidence, not a detector.
  - Heads that copy needle tokens at least 10 % of the time are about 3-6 % of all heads. This holds across 11 models, including Qwen1.5-14B and its chat version.
  - Masking 50 of them drops needle retrieval below 50 in all models; masking random heads has much less effect.
  - With them masked, the output stays fluent but unfaithful.
  - In hallucinated generations, the retrieval heads attend mostly to the initial token (the attention sink).
  - Code: nightdessert/Retrieval_Head. The paper's link renders without the underscore.
- **Chain-of-Embedding** (Wang, Zhang, Yang, Wong, Wang; ICLR 2025).
  - Method: label-free. The response-averaged hidden state is traced across layers. Per-layer magnitude and angle changes, normalised by the h0→hL displacement, predict correctness.
  - Mean AUROC of CoE-C over 7 models (including Qwen1.5-7B and Qwen2-7B/72B):
    - math 75.22, vs at most 66.08 for baselines;
    - reasoning 63.54;
    - knowledge 63.89;
    - understanding 64.98.
  - Cost about 1 ms. No position or length analysis. Code: Alsace08/Chain-of-Embedding.
- **INSIDE / EigenScore** (Chen et al.; ICLR 2024).
  - Method: log-det of the regularised covariance of middle-layer last-token embeddings over K = 10 sampled answers, plus feature clipping.
  - LLaMA-7B AUROC on CoQA / SQuAD / NQ / TriviaQA:
    - against sentence-similarity labels (cosine > 0.9): 80.4 / 81.5 / 76.5 / 82.7;
    - against ROUGE-L > 0.5 labels: 80.8 / 81.2 / 77.1 / 82.9.
  - Code: alibaba/eigenscore.
- **Semantic Entropy Probes** (Kossen, Han, Razzak, Schut, Malik, Gal; 2024).
  - Method: linear probes predict binarised semantic entropy from the hidden state at one of two positions: the token before generation (TBG) or the second-last token (SLT). No correctness labels are needed.
  - Compared with accuracy probes: −2.0 to +2.8 AUROC points in-distribution, +2.2 to +10.5 out of distribution.
  - TBG is slightly below SLT. Long-form results more often peak at intermediate layers. Probes stay below sampled SE.
  - Code: OATML/semantic-entropy-probes.
- **Token-level entity probes** (Obeso, Arditi, Ferrando, Freeman, Holmes, Nanda; 2025).
  - Method: a linear or LoRA probe at layer ⌊0.95 L⌋ outputs a streaming per-token probability of a fabricated entity.
  - Labels: Claude 4 Sonnet with web search.
    - Agreement with a human annotator: 84 % (n = 50).
    - Recall on injected errors: 80.6 %.
    - False flags: 15.8 %.
  - LongFact AUC (Llama-3.1-8B / 3.3-70B):
    - LoRA probe 0.8938 / 0.9048;
    - linear probe 0.8535 / 0.8667;
    - semantic entropy 0.7189 / 0.7138.
  - Qwen-2.5-7B with LoRA: 0.8947.
  - Cross-model transfer stays within 0.02-0.04. Halting at a threshold loses about 50 % of correct answers.
  - No position analysis. Code and data: obalcells/hallucination_probes.
- **Exact-answer probes** (Orgad, Toker, Gekhman, Reichart, Szpektor, Kotek, Belinkov; ICLR 2025).
  - The signal sits at the exact answer tokens: on Mistral-7B-Instruct TriviaQA, AUC 0.85 there vs 0.71 at the last generated token.
  - Poor generalisation across tasks. Code: technion-cs-nlp/LLMsKnow.
- **H-Neurons** (Gao, Chen, Xiao, Chen, Liu, Sun; arXiv 2512.01797, Dec 2025). These are almost certainly the user's "Haarneuronen" (speech-to-text for "H-Neuronen"). The "< 0.1 %" figure is confirmed.
  - Feature: CETT = ‖the neuron's down-projected output‖ / ‖that FFN layer's total output W_down·z_t‖, averaged over answer tokens. The paper calls the denominator the hidden state; it is not the residual stream.
  - Answer spans were found with GPT-4o.
  - Selection:
    - An ℓ1 logistic regression is trained on 1,000 consistently correct vs 1,000 consistently wrong TriviaQA questions (10 samples each).
    - Positives are answer-token features of wrong answers. Negatives are answer tokens of correct answers and all non-answer tokens.
    - The neurons with positive weights are the H-neurons (the main text says non-zero weights): 0.01-0.35 ‰ of all neurons.
    - The ℓ1 strength is grid-searched to balance held-out accuracy against TriviaQA damage when the neurons are suppressed, so the count is a tuning outcome.
  - Accuracy, H-neurons vs random neurons (TriviaQA / NQ / BioASQ / NonExist):
    - Mistral-7B: 78.4/71.5/75.5/91.1 vs 61.7/56.1/59.4/80.9.
    - One failure: Llama-3.1-8B scores 43.1 on NonExist vs 50.6 for random neurons.
  - Intervention: scaling the neurons (α 0-3) increases over-compliance (false premises, misleading context, sycophancy, jailbreaks).
  - The neurons are already present in base models.
  - Models: six from the Mistral, Gemma-3 and Llama-3.x families. No Qwen, and no position or length analysis.
  - Code: thunlp/H-Neurons.
- **Semantic entropy** (Farquhar, Kossen, Kuhn, Gal; Nature 2024).
  - Method: 10 samples, clustered by bidirectional entailment.
  - Mean AUROC 0.790 over 30 task-model pairs, vs 0.691 for naive entropy, 0.698 for P(True) and 0.687 for embedding regression.
  - The authors note it "does not help when LLM outputs are systematically bad".

**Common thread (synthesis):**
- Hallucinated spans read less from the source: lower LR on grounding heads, lower ECS.
- They read more from themselves or from the sink: a larger attention diagonal, and retrieval heads on the first token.

## 3. Position and length within one answer

- **FActScore** (Min et al.; EMNLP 2023).
  - "Across all LMs, the later part of the generation has significantly worse precision." Here "all LMs" are the three human-annotated ones: InstructGPT, ChatGPT and PerplexityAI.
  - The authors give two likely causes:
    - error propagation;
    - early facts (nationality, profession) are more frequent in pretraining data, which is a content-position confound.
  - The automated estimator has < 2 % error at the level of aggregate scores.
- **LURE** (Zhou et al.; ICLR 2024).
  - In MiniGPT-4 descriptions of 200 COCO images, hallucinated objects cluster "towards the end".
  - These objects also come with higher token uncertainty.
- **Hallucinate at the Last** (Yang, Yoon, Chang, Kim, Lee; 2025).
  - Method: outputs are split into five relative-position bins, and atomic facts are checked with MiniCheck.
  - Faithfulness, bin 1 → bin 5 (Wikipedia, long summaries):
    - Llama3.1-8B 0.81 → 0.75;
    - Gemma3-12B 0.85 → 0.79;
    - GPT-4o mini 0.83 → 0.76;
    - a human check on Llama gives 0.90 → 0.70.
  - The drop grows with summary length. For Llama and Gemma it survives changes of decoding strategy.
  - Llama attends about 3× more to the final generated sentence.
  - Qwen2.5-7B, run with a 16K sliding window that the authors configured, shows no drop: sensitivity −0.50 for long summaries and −1.25 for standard ones.
    - The authors attribute this to the sliding window.
    - Without it, sensitivity rises by 3.25 (long) and 3.75 (standard) points, to +2.75 and +2.5 (Table 5). The text says "to 3.75 / 3.25".
  - Limits: 20 examples per dataset; no control for position in the source was found.
- **Snowballing** (Zhang, Press, Merrill, Liu, Smith; 2023).
  - The model commits to an answer in the first token (98.40 % for ChatGPT, 95.67 % for GPT-4) and then justifies it.
  - 67.37 % / 87.03 % of hallucinations contain a claim that the same model rejects in a fresh session.
  - Chain-of-thought cuts errors to 9.40 % / 3.87 %, but snowballing remains (44.25 % / 94.90 %).
  - Tasks: 500 programmatically checkable questions each for primality, senator search and graph connectivity. Code: Nanami18/Snowballed_Hallucination.
- **The Illusion of Progress** (Janiak et al.; 2025).
  - The LLM judge agrees with humans at 0.723, ROUGE at only 0.142.
  - Under the judge's labels, length alone is a strong detector. Mean length gives AUROC 0.721 / 0.713 (Llama-3.1-8B / Mistral-7B). For comparison, EigenScore gives 0.747 / 0.694 and semantic entropy 0.773 / 0.732.
  - EigenScore correlates with length at Pearson 0.826 / 0.894, and wrong answers are longer.
- **Not found:** any measurement of H-neuron activity, lookback ratio or CoE as a function of position or length. The verifier did not repeat this search.

**Synthesis:**
- Late errors are robust, but the proposed causes compete: propagation, early commitment, attention drifting to the model's own text, and content order.
- Length predicts errors on its own.
- A late rise in any error-correlated internal readout is therefore expected even without an attractor.

## 4. Future-token information across positions, layers and heads

- **Future Lens** (Pal, Sun, Yuan, Wallace, Bau; CoNLL 2023).
  - From one GPT-J-6B hidden state, a learned soft prompt recovers the model's own prediction N tokens ahead: 48.4 / 43.7 / 46.9 % for N = 1/2/3. The bigram baseline is 20.1 % at N = 1.
  - Linear decoders drop from 29.2 to 15.8 % over the same range.
  - The best layer is the middle one (14 of 28). Code: future.baulab.info.
- **Do language models plan ahead for future tokens?** (Wu, Morris, Levine; COLM 2024).
  - The gradient splits into a diagonal (myopic) part and off-diagonal parts that flow through attention into past positions.
    - This formalises one reading of the user's "Gradient Through Routing".
    - The phrase may also mean gradient routing (Cloud et al.).
  - On a synthetic task, models clearly pre-cache information for later.
  - In natural language, the effect is mostly "breadcrumbs": features computed for the current token that also help later.
  - Myopia gap: 0.12 nats in GPT-2-124M (3.28 vs 3.40). It grows with Pythia model size from 70M.
  - The myopic model does better early in the sequence and falls behind later. Code: wiwu2390/FutureGPT2-public.
- **Token Erasure** (Feucht, Atkinson, Wallace, Bau; EMNLP 2024).
  - Probes read tokens at offsets −3 to +1 from h_t (Llama-2-7b, Llama-3-8b).
  - Next-token accuracy rises with depth, while accuracy for the current and previous tokens falls. This is read from the figures and is not stated in the text.
  - The last token of a multi-token word or entity erases its earlier tokens in the early layers. This is the backward counterpart of Future Lens.
- **Arrows of Time for LLMs** (Papadopoulos, Wenger, Hongler; ICML 2024).
  - Forward models beat backward ones by about 0.6-3.2 % in log-perplexity (8 languages, Medium/XL models).
  - The gap grows with model size (French: +0.15 % at 4.9M → +2.65 % at 405M) and with context length.
  - An ideal predictor would show no gap. The authors attribute the gap to sparsity and computational complexity. Code: frotaur/LLM-Arrows-of-Time.
- **Emergent Response Planning** (Dong et al.; ICML 2025).
  - MLP probes on the prompt predict, before generation starts:
    - the response's length;
    - its reasoning steps;
    - its choices;
    - its confidence;
    - its factual consistency.
  - During the response, planning signal follows three phases: high, then a dip, then recovery.
  - Larger models plan better within a family; the tested range includes Qwen2.5 1.5B-72B. Results are given in figures only.
- **Look-ahead** (Men, Cao, Jin, Chen, Liu, Zhao; 2024).
  - In Blocksworld, a probe at step 1 predicts step 6 with accuracy 0.51.
  - In layers 15-20 of fine-tuned Llama-2-7b-chat and Vicuna-7B, attention extracts the answer more than the MLP does.
- **Multi-token prediction** (Gloeckle et al.; ICML 2024; abstract only).
  - 13B models solve 12 % more HumanEval and 17 % more MBPP problems.
  - Inference is up to 3× faster.
- **Gradient Routing** (Cloud et al.; 2024; abstract only).
  - Gradient masks confine capabilities to chosen subregions of the network.
  - It is a training tool, not a readout.

## 5. Implications for a chain lens in Qwen3-1.7B (derived)

**5.1 Target.** Facts from the config, and what follows:
- 28 layers × 16 query heads (8 KV heads, head_dim 128) give **448 head channels**.
- FFN width 6144 gives 172,032 FFN neurons; 0.1 % of them is 172.
- At the H-Neurons ratios (0.01-0.35 ‰), one would expect about **2-60 H-neurons**.
  - This is an extrapolation from six models of other families.
  - The count depends on the ℓ1 strength chosen.
- The model uses full attention (`use_sliding_window: false`) and tied embeddings, and the config has no MTP field.
- A J-lens for Qwen3-1.7B exists on Hugging Face (neuronpedia/jacobian-lens, folder qwen3-1.7b/jlens; see `analysis/ju_space/microlens/README.md`).
- The published chain lens therefore needs a substitute for the missing MTP head:
  - a Future-Lens learned prompt (trained, with Qwen's weights frozen);
  - a small MTP-style head trained on the frozen model; or
  - a self-rollout with no new weights: patch the original layer-l state into a placeholder position at layer l of the unmodified model (Patchscopes-style) and let it generate.
- A transported final-layer state cannot simply be rolled out, because later positions never read a position's final-layer output.
- Whichever substitute is used, its own confabulation must be measured.

**5.2 An arrow per channel.**

Mutual information is symmetric: I(c_t; c_{t+k}) = I(c_{t+k}; c_t).
- If the channel's statistics are stationary (shift-invariant over positions), I(c_t; c_{t+k}) = I(c_t; c_{t−k}). "How much a front predicts later vs earlier fronts" of the same channel then has no arrow.
- The same holds for linear predictability in whitened units.
- Within a short answer, stationarity fails (position effects, the sink, the chat template). A raw same-channel asymmetry therefore measures position dependence, which is a confound rather than a dynamical arrow.

An arrow can come from four sources:
- **Asymmetric targets:** state c_t predicting the future token x_{t+k} (Future Lens) vs the past token x_{t−k} (Feucht probes).
- **Directed, conditional quantities between channels:** transfer entropy, or the Granger gain of A's front for B's later front beyond B's own past. This is the user's hierarchy of channels.
- **Capacity-limited predictors**, as in Arrows of Time.
- **Non-stationarity (position dependence):** this one must be controlled, not reported as an arrow.

Structural asymmetry and depth:
- Past tokens are inputs to h_t and future tokens are not, so in early layers the past is far more decodable.
- With depth, previous-token information is erased while next-token information grows (Feucht et al.).
- Raw forward/backward ratios therefore mix this structural asymmetry with depth. They must be normalised and compared within layers.

Proposed readouts per channel c at layer l. A channel is one head's output after W_O, or one FFN-neuron group.
- **F_c(k):** probe accuracy from c_t to the model's own token at t+k, divided by the same probe's accuracy from the full residual h_t^l. Probes have equal capacity.
- **B_c(k):** the same for the past token x_{t−k}.
- **LR_c (heads only):** the lookback ratio. Split "own text" into previous answers vs the current answer, to match the round-2 "Continue." setup.
- **D_c (heads only):** the mean log attention diagonal (LLM-Check).
- **S_c (heads only):** the attention share on the sink and template tokens.

The arrow vector is a_c = (F_c − B_c for k = 1-3, LR_c, D_c, S_c) for heads. FFN-neuron groups get only the F_c − B_c terms.
- Compare it within layers, because backward information falls with depth and forward information peaks mid-depth.
- Compare against random channel sets of equal dimension and against position-shuffled controls, on held-out text.

Inverse-free, per the user's rule in README §6:
- Fit backward probes directly.
- Verify them by a forward round-trip: re-inject the readout and check that the channel state or neuron activation reappears.
- Do not invert the J-lens Jacobian. The user's rule is sufficient reason. Arrows of Time's sparse-forward/dense-inverse argument is at most an analogy here, not a prediction about the J-lens.

**5.3 Break-off criteria.**
- **H_lens:**
  - Stop the chain at the first step whose contrastive evidence, log p(x_k|state) − log p(x_k|mean state), falls inside a null band.
  - Build that band by running the identical selection (contrastive argmax over the vocabulary) on null states, because the chosen token maximises this contrast by construction.
  - Null states:
    - the post's foreign-seed and corpus-mean controls;
    - a shuffled-state control of our own. The post's shuffle null permutes readouts at the judge, not states.
  - Neutral seeds are not a null: in the post they perform about as well as J-lens seeds.
  - Also stop when the round-trip fails.
  - Calibrate on arithmetic chains with known intermediates that Qwen3-1.7B solves.
- **H_model:**
  - Break at the first 8-token window where LR on the grounding heads falls and D rises past thresholds fixed on a calibration split.
  - Alternative: a token-probe threshold, like the streaming halt of Obeso et al.
  - Expect modest power out of domain. Lookback Lens with 8-token windows reaches 85-87 AUROC in-domain but only about 66 cross-task, and the round-2 "Continue." texts are a new task.
- **H-neuron view:**
  - Measure the share of the readout's write energy that falls on H-neuron write vectors.
  - Compare against many random neuron sets matched by layer and CETT distribution.

**5.4 Which readouts are independent labels?**

| Candidate | Label source | Independent of the lens? | Role |
|---|---|---|---|
| Gold answer, judged by an LLM (not ROUGE) | output vs reference | yes | primary short-form label |
| Programmatic checks (arithmetic intermediates, primality, graph connectivity) | output vs computation | yes, and gives the first-error position | primary label for break-off timing |
| Atomic facts vs external evidence (FActScore estimator, MiniCheck, web-search judge) | output vs evidence | yes; judge error is partly known (FActScore < 2 % aggregate-score error; web-search judge 84 % agreement on n = 50, 15.8 % false flags) | long-form position analyses |
| Semantic entropy | 10 own samples | independent of internals, but not of the model; misses consistent errors | secondary label or stratifier |
| Lookback ratio, attention diagonal, ECS, copy scores | attention from the same pass | partly; circular if the lens uses attention | competing predictors, arrow ingredients |
| CoE, EigenScore, hidden log-det, SEP, token and exact-answer probes | hidden states | no | baselines; reuse their label pipelines, not their scores |
| H-neuron CETT | FFN activations; the probe is trained on correctness | no; it is part of the break-off rule | the object under test; identify it on a disjoint question split |
| Length, position, perplexity, entropy | output, logits | not labels | mandatory baselines |

**5.5 Length artifact vs attractor.**

Always report:
- the length baselines (Len; Mean-Len and Std-Len, which need several samples per prompt) and position;
- AUROC within length strata;
- a hazard model of the first verified error, with position as the baseline hazard and the readouts as time-varying covariates.

Regression on length is not enough. Response length is planned from the prompt (Dong et al.), and wrong answers are longer (Janiak et al.), so length can be a mediator rather than a confound. Add interventions:
- neutral prefixes that shift the starting position;
- counterbalanced item order, against FActScore's content-position confound;
- snowball injection: prefill a wrong vs a correct commitment of equal length, and track CETT, LR and break-off downstream;
- fixed-length loop rounds (round-2 texts are 40 tokens), with an equal-length neutral-context control.

A readout that rises after a wrong commitment at a fixed position, but not after neutral text of equal length, would point to an effect not bound to length. That rise can still be snowballing, i.e. the model justifying its claim. Call it an attractor only if it persists or returns after the wrong claim stops being referenced.

Two caveats:
- Qwen3-1.7B uses full attention. That is the setting in which Qwen2.5-7B did show a late drop (with the authors' 16K window removed), but the effect still has to be measured.
- A 40-token continuation has only five 8-token windows, while the documented late effects come from outputs of 100-800 words. A long-form arm is needed.

**5.6 Links to round 2.**
- **Style attractor vs H-neurons**, with no generated-length confound:
  - Measure the alignment of the shared loop direction with H-neuron write vectors, against random neuron sets.
  - Steer along it at 0.05 nats, against random directions, and measure H-neuron CETT and over-compliance (FalseQA, sycophancy).
  - H-neuron amplification produces compliance, so an "H-neuron attractor" should look like style or compliance.
  - Round 2's attractor was a generic style direction (sea, tone, sport). Compliance was never measured there, so this link is a hypothesis.
- **Channel-resolved K:** a three-way lookback split (prompt, previous answers, current answer) shows which heads carry the stance across answers (K 0.45-0.48).
- **Lens readouts replacing A3:** A3 vanished at blocks 22-27. When chain-lens readouts take over, fix the chain length and report the nulls, because confabulation grows with chain length.

## 6. Unverified or not found

**Not opened:**
- Neuron Lens, WorkspaceBench and the J-lens paper.
- The rhyme-planning work cited by the post.
- Qwen3.5 MTP heads.

**Not found (the verifier did not repeat this search):**
- Any study of how H-neuron activity, lookback ratio or CoE changes over position.
- Any Qwen3 result for these detectors.

**Inconsistencies inside the sources:**
- H-Neurons:
  - the label equation is inverted relative to its text;
  - the main text selects non-zero weights, while Fig. 1 and the appendix select positive weights;
  - the 70B model name differs: Llama-3.1-70B in the text, Llama-3.3-70B in Table 1.
- Hallucinate at the Last:
  - the Llama-3.1-8B long-summary sensitivity is 5.50 in Table 2 but 10.3 in Table 6;
  - the text says removing sliding-window attention raises Qwen's sensitivity "to 3.75 / 3.25", while Table 5 gives 2.5 (+3.75) and 2.75 (+3.25).
- INSIDE: Table 5 numbers are lower than Table 1's (LLaMA-7B CoQA 71.2 / 75.7 vs 80.4 / 80.8). Table 5 follows a different protocol (Duan et al., N = 5 generations for LLaMA), so this is not a contradiction.

## 7. Implications for design (verified and corrected)

1. **Two labels.** Separate H_lens (chain-lens readout confabulation, which grows with chain length in the post) from H_model (factual error in Qwen3-1.7B's own text). Give each its own label and null.
2. **Per-channel arrow.**
   - Use asymmetric targets: equal-capacity probes from c_t to the future token x_{t+k} vs the past token x_{t−k}, each normalised by the same probe on the full residual.
   - Or use directed transfer-entropy/Granger quantities between channels.
   - Same-channel forward-vs-backward front prediction is symmetric for a stationary sequence. Within short answers, any raw asymmetry reflects position dependence and needs position-matched nulls.
3. **Channels.**
   - The channels are 448 heads (28×16; 8 KV groups) and FFN-neuron groups.
   - Arrow vector per head: (F_c − B_c for k = 1-3, lookback ratio, mean log attention diagonal, sink share). FFN groups get only the F_c − B_c terms.
   - Compare within layers, against random channel sets of equal dimension and against position-shuffled controls, on held-out text.
4. **MTP substitute.** Qwen3-1.7B has no MTP head. Use one of:
   - a Future-Lens learned prompt;
   - a small MTP-style head trained on the frozen model;
   - a Patchscopes-style self-rollout from the layer-l state. A rollout from the transported final-layer state is not possible.

   Measure the substitute's own confabulation and fix the chain length. Report the post's foreign-seed and corpus-mean nulls, plus a new shuffled-state null. Neutral seeds are not a null.
5. **H_lens break-off.**
   - Stop at the first chain step whose contrastive evidence, log p(x|state) − log p(x|mean state), falls inside a selection-matched null band (the same contrastive argmax run on null states), or where a forward round-trip fails.
   - Calibrate on arithmetic chains with known intermediates that Qwen3-1.7B solves.
6. **H_model break-off.**
   - Break at the first 8-token window where the lookback ratio of grounding heads falls and the attention-diagonal score rises past thresholds fixed on a calibration split.
   - Expect modest power out of domain: about 66 AUROC cross-task (85-87 in-domain).
7. **Independent labels only.**
   - Use gold answers judged by an LLM (not ROUGE); programmatic verifiers (arithmetic intermediates, primality, graph connectivity), which also give the first-error position; and atomic-fact checks against external evidence.
   - Semantic entropy is a secondary label only.
   - SEP, CoE, EigenScore, hidden log-det, token probes and H-neuron scores are competing predictors, never labels.
8. **Re-identify H-neurons.**
   - Use thunlp/H-Neurons on a question split disjoint from all evaluation items.
   - Expected scale, if Qwen behaves like the six tested models: about 2-60 of 172,032 FFN neurons. The count depends on the grid-searched ℓ1 strength.
   - The pipeline uses GPT-4o for answer spans; any substitute must be validated.
   - Build the H-neuron-share null from many random neuron sets matched by layer and CETT distribution.
9. **Length controls.**
   - Always report Len, Mean-Len and Std-Len (the last two need several samples per prompt) and position baselines, plus AUROC within length strata.
   - Fit a hazard model of the first verified error, with position as the baseline hazard and readouts as time-varying covariates.
10. **Attractor-vs-length interventions**, because length can be a mediator:
    - neutral prefixes that shift position;
    - counterbalanced item order;
    - a wrong-vs-correct first-token commitment of equal length (snowballing paradigm);
    - fixed-length loop rounds with an equal-length neutral-context control.

    Call an effect an attractor only if it persists or returns after the wrong claim stops being referenced; otherwise it may be snowballing.
11. **Long-form arm.** Add one (hundreds of tokens). The 40-token loop continuations give only five 8-token windows, while documented late-position effects come from 100-800-word outputs.
12. **Round-2 tie-in (a hypothesis).**
    - Test whether the generic style attractor aligns with H-neuron write vectors more than random sets do.
    - Test whether steering along it at matched damage (0.05 nats) raises H-neuron CETT and over-compliance (FalseQA, sycophancy) more than random directions.
13. **Channel-resolved K.** Split lookback attention three ways (prompt, previous own answers, current answer) to find which heads carry the stance across "Continue." answers.
14. **Inverse-free back-reconstruction** (the user's rule).
    - Fit backward probes directly.
    - Verify them by forward round-trip: re-inject, and check that the state or neuron activation is recreated.
    - Never invert the J-lens Jacobian or a confusion matrix.
15. **Sink and template tokens.**
    - Exclude attention-sink and chat-template tokens from the lookback "context", or report them separately.
    - Use position-matched nulls for the lookback ratio.
    - Retrieval heads attend to the first token while the model hallucinates.

## Verification changes (13 Nov 2026)

- **Lookback Lens:**
  - the in-domain labels were swapped (91.2 is QA, 88.8 is Sum);
  - the head roles are the authors' conjecture;
  - 84.0 is a 13B-trained detector.
- **LLM-Check:**
  - RAGTruth is 54.19 white-box / 58.30 black-box (attention score), not "about 58";
  - −5.42 vs −5.85 is one example.
- **ReDeEP:**
  - PCC 0.41 / 0.46 relate per-head maps, not labels;
  - the AUC range now includes LLaMA3-8B (chunk-level 0.7285-0.8420);
  - full author list added.
- **Retrieval heads:**
  - the code is nightdessert/Retrieval_Head;
  - sink attention is reported for hallucinated generations;
  - venue ICLR 2025.
- **INSIDE:**
  - the quoted numbers use sentence-similarity labels; ROUGE-L numbers added;
  - Table 5 is a different protocol, not a contradiction.
- **SEP:** the probe position is the second-last token (SLT); TBG is slightly below SLT.
- **H-Neurons:**
  - CETT's denominator is the FFN layer output;
  - training negatives and the positive vs non-zero selection are spelled out;
  - the count is tuning-dependent.
- **Hallucinate at the Last:** the Qwen sliding-window numbers were corrected, and a text-table inconsistency was added.
- **Chain lens:**
  - the neutral-seed control was added;
  - the shuffle null is judge-level;
  - hallucination-family scores were added;
  - the family label "chain intermediates" was made precise;
  - the post names Qwen3.6-27B only for the neuron part.
- **Sections 4-5 (reasoning):**
  - the "exact form" claim was softened;
  - the stationarity condition was made explicit, and non-stationarity added as a source of asymmetry;
  - the statement about past decodability was corrected using Feucht et al.;
  - attention terms are now restricted to heads;
  - the MTP substitute was fixed (no rollout from a final-layer state);
  - the selection-matched null band was added;
  - in-domain vs cross-task AUROC are distinguished;
  - Arrows of Time is now labelled as an analogy;
  - the round-2 compliance link is marked as a hypothesis;
  - a snowballing caveat was added for attractor claims.
- **Confirmed unchanged:** CoE, Obeso et al., Orgad et al., Snowballing, FActScore, LURE, Illusion of Progress, Future Lens, plan-ahead, Token Erasure, Arrows of Time, Emergent Response Planning, Look-ahead, MTP, Gradient Routing, semantic entropy and the Qwen3-1.7B config. Code repos were confirmed from paper links or from READMEs naming each paper.


## implications_for_design (from the research stage; check against the verifier's notes above)
```json
[
 "Separate H_lens (chain-lens readout confabulation, which the Chain Lens post shows grows with chain length) from H_model (factual error in Qwen3-1.7B's own text); give each its own label and null.",
 "Define the per-channel arrow with asymmetric targets (equal-capacity probes c_t -> future token x_{t+k} vs past token x_{t-k}, each normalised by the same probe on the full residual) or with directed transfer-entropy/Granger quantities between channels. Same-channel forward-vs-backward front prediction is symmetric by construction (mutual information).",
 "Channels in Qwen3-1.7B: 448 heads (28x16; 8 KV groups) and FFN-neuron groups. Arrow vector per channel = (F_c - B_c for k=1-3, lookback ratio, mean log attention diagonal, sink share). Compare within layers, against random channel sets of equal dimension and against position-shuffled controls, on held-out text.",
 "Qwen3-1.7B has no MTP head. Use a Future-Lens learned prompt, a small MTP-style head trained on the frozen model, or a teacher-forced rollout from the J-lens-transported state. Measure the substitute's own confabulation, fix chain length, and report foreign-seed, shuffled-state and corpus-mean nulls.",
 "H_lens break-off: the first chain step whose contrastive evidence log p(x|state) - log p(x|mean state) falls inside the null band, or where a forward round-trip fails. Calibrate on arithmetic chains with known intermediates that Qwen3-1.7B solves.",
 "H_model break-off: the first 8-token window where the lookback ratio of grounding heads falls and the attention-diagonal score rises past thresholds fixed on a calibration split. Expect modest power: Lookback Lens transfers at about 66 AUROC with 8-token windows.",
 "Independent labels only: gold answers judged by an LLM (not ROUGE); programmatic verifiers (arithmetic intermediates, primality, graph connectivity) that also give the first-error position; atomic-fact checks against external evidence. Semantic entropy is a secondary label only. SEP, CoE, EigenScore, hidden log-det, token probes and H-neuron scores are competing predictors, never labels.",
 "Re-identify H-neurons in Qwen3-1.7B with thunlp/H-Neurons on a question split disjoint from all evaluation items (expected scale: about 2-60 of 172,032 FFN neurons). Build the H-neuron-share null from many random neuron sets matched by layer and CETT distribution.",
 "Length controls: always report Len, Mean-Len, Std-Len and position baselines and AUROC within length strata. Fit a hazard model of the first verified error with position as the baseline hazard and readouts as time-varying covariates.",
 "Attractor-vs-length interventions, because length can be a mediator: neutral prefixes that shift position; counterbalanced item order; wrong-vs-correct first-token commitment of equal length (snowballing paradigm); fixed-length loop rounds with an equal-length neutral-context control.",
 "Add a long-form arm (hundreds of tokens). The 40-token loop continuations give only five 8-token windows, while documented late-position effects come from 100-800-word outputs.",
 "Round-2 tie-in: test whether the generic style attractor aligns with H-neuron write vectors more than random sets, and whether steering along it at matched damage (0.05 nats) raises H-neuron CETT and over-compliance (FalseQA, sycophancy) more than random directions.",
 "Channel-resolved K: split lookback attention three ways (prompt, previous own answers, current answer) to find which heads carry the stance across 'Continue.' answers.",
 "Inverse-free back-reconstruction (user's rule): fit backward probes directly and verify them by forward round-trip (re-inject, check that the state or neuron activation is recreated). Never invert the J-lens Jacobian or a confusion matrix.",
 "Exclude attention-sink and chat-template tokens from the lookback 'context' (or report them separately), and use position-matched nulls for the lookback ratio. Retrieval heads attend to the first token while the model hallucinates."
]
```

## refs (research stage)
```json
[
 {
  "title": "The Chain Lens [interim]",
  "url": "https://www.lesswrong.com/posts/Nr6fksEqqxNpdz6rk/the-chain-lens-interim",
  "authors": "vgel, Raymond Douglas",
  "year": "2026",
  "finding": "J-lens transport + pretrained MTP-head rollout + contrastive decoding vs corpus-mean state; WorkspaceBench 0.42 vs 0.32 (J-lens, 28 families), 0.487 vs 0.411 (composite, 21), arithmetic chain intermediates 0.42 vs 0.25 (oracle lens); confabulation on the hallucination family grows with chain length; 'Hallucination and performance both scale with chain length'; neuron round-trip (99.8th pct activation, beat 90 % of random readouts), judge 57 % (4-way); run on Qwen3.6-27B; needs an MTP head.",
  "relevance": "Defines the user's 'chain lens'; its 'hallucination' is lens confabulation (H_lens), not model error; Qwen3-1.7B lacks an MTP head.",
  "how_checked": "Rendered with pre-installed Playwright (headless Chromium via session proxy) into a scratch dir; read the post body text (tables partly garbled)."
 },
 {
  "title": "Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps",
  "url": "https://arxiv.org/abs/2407.07071",
  "authors": "Yung-Sung Chuang, Linlu Qiu, Cheng-Yu Hsieh, Ranjay Krishna, Yoon Kim, James Glass",
  "year": "2024 (EMNLP 2024)",
  "finding": "Per-head lookback ratio (mean attention per context token vs per generated token), logistic regression over all heads; AUROC in-domain 91.2/88.8, cross-task 85.3/82.0 (spans; layer-28 hidden states 83.6/84.7), 8-token windows 87.1/85.3 in-domain, 66.1/66.0 cross-task (hidden <=59.4); 7B->13B 73.5 vs 84.0; top-10 heads 71.2/79.2; guided decoding XSum 49.0->58.6 % correct (510->414 hallucinated). GPT-4o span labels. LLaMA-2-7B/13B-Chat. Code voidism/Lookback-Lens.",
  "relevance": "Template for the backward (read-source) component of a per-head arrow and for an 8-token-window break-off; scores are predictors, not labels.",
  "how_checked": "Read arXiv HTML v2 (method, Tables 2-8) via WebFetch; repo confirmed by GitHub search."
 },
 {
  "title": "LLM-Check: Investigating Detection of Hallucinations in Large Language Models",
  "url": "https://proceedings.neurips.cc/paper_files/paper/2024/hash/3c1e1fdf305195cd620c118aaa9717ad-Abstract.html",
  "authors": "Gaurang Sriramanan, Siddhant Bharti, Vinu Sankar Sadasivan, Shoumik Saha, Priyatham Kattakinda, Soheil Feizi",
  "year": "2024 (NeurIPS 2024)",
  "finding": "Training-free single-response scores: attention score = mean log attention diagonal (log det of lower-triangular map), length-normalised hidden log-det, windowed logit entropy; FAVA AUROC attention 72.34 (Llama-2-7B L21), 71.69 (Vicuna), 68.19 (Llama-3-8B) vs hidden 57-58, INSIDE 59.03; RAGTruth ~58; hallucinated tokens have larger log-diagonal; 0.22 s/example. Code GaurangSriramanan/LLM_Check_Hallucination_Detection.",
  "relevance": "Cheap per-token self-attention measure (D_c) for the arrow vector and break-off; not a label.",
  "how_checked": "Downloaded the NeurIPS PDF, extracted text with pdftotext in its own scratch dir, read Sections 4-5 and Appendix C; repo found by GitHub search."
 },
 {
  "title": "Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation",
  "url": "https://arxiv.org/abs/2410.13640",
  "authors": "Yiming Wang, Pei Zhang, Baosong Yang, Derek F. Wong, Rui Wang",
  "year": "2024 (ICLR 2025)",
  "finding": "Label-free: layer-to-layer magnitude/angle trajectory of the response-averaged hidden state predicts correctness; mean AUROC math 75.22 (CoE-C) vs best baseline 66.08, reasoning 63.54, knowledge 63.89, understanding 64.98; 7 models incl. Qwen1.5-7B, Qwen2-7B/72B; ~1 ms; no position/length analysis. Code Alsace08/Chain-of-Embedding.",
  "relevance": "A 'chain' over depth rather than positions; baseline feature; shares hidden states with any lens, so not independent.",
  "how_checked": "Read arXiv HTML v2 in two parts via WebFetch."
 },
 {
  "title": "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection",
  "url": "https://arxiv.org/abs/2402.03744",
  "authors": "Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, Jieping Ye",
  "year": "2024 (ICLR 2024)",
  "finding": "EigenScore = log det of covariance of middle-layer last-token embeddings of K=10 samples (+ feature clipping); LLaMA-7B AUROC 80.4 CoQA, 81.5 SQuAD, 76.5 NQ, 82.7 TriviaQA under ROUGE-L>0.5 labels. Code alibaba/eigenscore.",
  "relevance": "Sampling-based hidden-state baseline; its ROUGE labels are unreliable per Janiak et al.",
  "how_checked": "Read arXiv HTML v2 via WebFetch."
 },
 {
  "title": "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs",
  "url": "https://arxiv.org/abs/2406.15927",
  "authors": "Jannik Kossen, Jiatong Han, Muhammed Razzak, Lisa Schut, Shreshth Malik, Yarin Gal",
  "year": "2024",
  "finding": "Linear probes on hidden states (token before generation or last token) predict binarised semantic entropy; vs accuracy probes -2.0 to +2.8 AUROC points in-distribution, +2.2 to +10.5 OOD; TBG nearly as good as last token; long-form peaks at intermediate layers; below sampled SE. Code OATML/semantic-entropy-probes.",
  "relevance": "Uncertainty is readable before generation (forward information); hidden-state based, so not independent of a lens.",
  "how_checked": "Read arXiv abstract page and HTML v1; repo found by GitHub search."
 },
 {
  "title": "Detecting hallucinations in large language models using semantic entropy",
  "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11186750",
  "authors": "Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, Yarin Gal",
  "year": "2024 (Nature 630)",
  "finding": "10 samples clustered by bidirectional entailment; mean AUROC 0.790 over 30 task-model pairs vs 0.691 naive entropy, 0.698 P(True), 0.687 embedding regression; does not help when outputs are systematically wrong.",
  "relevance": "Internals-free secondary label for Qwen3-1.7B; blind to consistent (snowballed or attractor-driven) errors.",
  "how_checked": "Read PMC full text via WebFetch."
 },
 {
  "title": "H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs",
  "url": "https://arxiv.org/abs/2512.01797",
  "authors": "Cheng Gao, Huimin Chen, Chaojun Xiao, Zhiyi Chen, Zhiyuan Liu, Maosong Sun",
  "year": "2025",
  "finding": "l1 logistic probe on CETT (neuron down-projected output norm / hidden-state norm) averaged over answer tokens, 1,000 consistently correct vs 1,000 consistently wrong TriviaQA items; H-neurons 0.01-0.35 per mille (<0.1 %); accuracy e.g. Mistral-7B 78.4/71.5/75.5/91.1 vs random 61.7/56.1/59.4/80.9; scaling (alpha 0-3) raises over-compliance; present in base models; 6 Mistral/Gemma-3/Llama-3.x models, no Qwen; no position/length analysis; label equation inconsistent with text. Code thunlp/H-Neurons.",
  "relevance": "Verifies the user's 'Haarneuronen'; must be re-identified in Qwen3-1.7B on a disjoint split; part of the break-off rule, hence not a label.",
  "how_checked": "Read arXiv HTML v2 via WebFetch; official repo found by GitHub search."
 },
 {
  "title": "ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability",
  "url": "https://arxiv.org/abs/2410.11414",
  "authors": "Sun et al. (full list not checked)",
  "year": "2024 (ICLR 2025)",
  "finding": "External Context Score of copying heads vs Parametric Knowledge Score of late FFNs (JSD of logit-lens distributions before/after FFN); 1006/1024 Llama-2-7B heads higher ECS on truthful responses; Pearson 0.41/0.46; chunk-level AUC 0.7458-0.8420. Code Jeryi-Sun/ReDEeP-ICLR.",
  "relevance": "Two-channel (attention-copy vs FFN-parametric) readout: a concrete channel hierarchy for the arrow.",
  "how_checked": "Read arXiv HTML v2 (first 100k chars) via WebFetch; repo found by GitHub search."
 },
 {
  "title": "Retrieval Head Mechanistically Explains Long-Context Factuality",
  "url": "https://arxiv.org/abs/2404.15574",
  "authors": "Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, Yao Fu",
  "year": "2024",
  "finding": "Heads with copy score >=0.1 are 3-6 % of heads in 11 models (incl. Qwen1.5-14B); masking 50 drops needle retrieval below 50, random heads much less; output stays fluent but unfaithful; hallucinating heads attend to the initial token.",
  "relevance": "Causal evidence that few channels carry grounding; motivates a sink-share component in the arrow vector.",
  "how_checked": "Read arXiv abstract page and HTML via WebFetch."
 },
 {
  "title": "Real-Time Detection of Hallucinated Entities in Long-Form Generation",
  "url": "https://arxiv.org/abs/2509.03531",
  "authors": "Oscar Obeso, Andy Arditi, Javier Ferrando, Joshua Freeman, Cameron Holmes, Neel Nanda",
  "year": "2025",
  "finding": "Streaming token-level probes (linear/LoRA, layer floor(0.95L)); labels by Claude 4 Sonnet + web search (84 % human agreement, 80.6 % recall, 15.8 % false flags); LongFact AUC LoRA 0.8938/0.9048, linear 0.8535/0.8667, SE 0.7189/0.7138; Qwen-2.5-7B LoRA 0.8947; transfer within 0.02-0.04; threshold halting costs ~50 % of correct answers; no position analysis. Code obalcells/hallucination_probes.",
  "relevance": "Released, internals-free long-form label pipeline plus a streaming halt that is a ready break-off template.",
  "how_checked": "Read arXiv HTML in two parts via WebFetch."
 },
 {
  "title": "LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations",
  "url": "https://arxiv.org/abs/2410.02707",
  "authors": "Hadas Orgad, Michael Toker, et al.",
  "year": "2024 (ICLR 2025)",
  "finding": "Error signal concentrates in exact answer tokens (Mistral-7B-Instruct TriviaQA AUC 0.85 vs 0.71 at last generated token); poor cross-task generalisation. Code technion-cs-nlp/LLMsKnow.",
  "relevance": "Tells where in the chain to read; warns that probes do not transfer across tasks.",
  "how_checked": "Read arXiv HTML v2 (first 100k chars) via WebFetch."
 },
 {
  "title": "How Language Model Hallucinations Can Snowball",
  "url": "https://arxiv.org/abs/2305.13534",
  "authors": "Muru Zhang, Ofir Press, William Merrill, Alisa Liu, Noah A. Smith",
  "year": "2023",
  "finding": "First-token commitment 98.40 % (ChatGPT) / 95.67 % (GPT-4); 67.37 % / 87.03 % of hallucinations contain claims the same model rejects in isolation; CoT cuts errors to 9.40 % / 3.87 % but snowballing remains (44.25 % / 94.90 %); 500 programmatically checkable questions per task. Code Nanami18/Snowballed_Hallucination.",
  "relevance": "Length-independent mechanism (early commitment) and verifiable tasks giving first-error position; basis for a wrong-vs-correct prefill intervention.",
  "how_checked": "Read arXiv abstract page and HTML via WebFetch."
 },
 {
  "title": "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation",
  "url": "https://arxiv.org/abs/2305.14251",
  "authors": "Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi",
  "year": "2023 (EMNLP 2023)",
  "finding": "'Across all LMs, the later part of the generation has significantly worse precision'; causes: error propagation and early facts being more frequent in pretraining data; rarity drop 80 %->16 % (ChatGPT); estimator <2 % error; 13 LMs, 6,500 generations.",
  "relevance": "Late-position effect with an explicit content-position confound; external atomic-fact label.",
  "how_checked": "Read arXiv abstract page and HTML v2 (first 100k chars) via WebFetch."
 },
 {
  "title": "Analyzing and Mitigating Object Hallucination in Large Vision-Language Models",
  "url": "https://arxiv.org/abs/2310.00754",
  "authors": "Yiyang Zhou, Chenhang Cui, Jaehong Yoon, Linjun Zhang, Zhun Deng, Chelsea Finn, Mohit Bansal, Huaxiu Yao",
  "year": "2023 (ICLR 2024)",
  "finding": "Positioning score analysis (MiniGPT-4, 200 COCO images): hallucinated objects concentrate in the latter part of descriptions and carry higher token uncertainty; explanation offered as plausible (accumulated hallucinatory information).",
  "relevance": "Independent replication of late-position concentration in another modality.",
  "how_checked": "Read arXiv abstract page and HTML v2 Section 2 via WebFetch."
 },
 {
  "title": "Hallucinate at the Last in Long Response Generation: A Case Study on Long Document Summarization",
  "url": "https://arxiv.org/abs/2505.15291",
  "authors": "Joonho Yang, Seunghyun Yoon, Hwan Chang, Byeongjeong Kim, Hwanhee Lee",
  "year": "2025",
  "finding": "Five relative-position bins, MiniCheck: bin 1->5 Llama3.1-8B 0.81->0.75, Gemma3-12B 0.85->0.79, GPT-4o mini 0.83->0.76; human 0.90->0.70; grows with summary length; Llama attends final generated sentence ~3x; Qwen2.5-7B none (-0.50) attributed to sliding window, +3.25 to +3.75 without it; 20 examples per dataset.",
  "relevance": "Direct evidence that errors concentrate late within one answer, architecture-dependent; Qwen3-1.7B uses full attention.",
  "how_checked": "Read arXiv HTML v3 (first 100k chars) via WebFetch."
 },
 {
  "title": "The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs",
  "url": "https://arxiv.org/abs/2508.08285",
  "authors": "Denis Janiak, Jakub Binkowski, Albert Sawczyn, Bogdan Gabrys, Ravid Shwartz-Ziv, Tomasz Kajdanowicz",
  "year": "2025",
  "finding": "LLM-judge agreement with humans 0.723 vs ROUGE 0.142; Mean-Len AUROC 0.721/0.713 vs EigenScore 0.747/0.694, SemEntropy 0.773/0.732 (Llama-3.1-8B/Mistral-7B); EigenScore-length Pearson 0.826/0.894; wrong answers longer.",
  "relevance": "Length baselines are mandatory; detector scores track length; ROUGE labels unusable.",
  "how_checked": "Read arXiv HTML v1 via WebFetch."
 },
 {
  "title": "Future Lens: Anticipating Subsequent Tokens from a Single Hidden State",
  "url": "https://arxiv.org/abs/2311.04897",
  "authors": "Koyena Pal, Jiuding Sun, Andrew Yuan, Byron C. Wallace, David Bau",
  "year": "2023 (CoNLL 2023)",
  "finding": "Learned soft prompt recovers GPT-J-6B's own prediction N tokens ahead from one hidden state: 48.4/43.7/46.9 % (N=1/2/3; bigram 20.1 %); linear decoders 29.2->15.8 %; peak at middle layer 14 of 28. Code future.baulab.info.",
  "relevance": "Forward component F_c(k) and an MTP substitute for Qwen3-1.7B.",
  "how_checked": "Read arXiv abstract page and HTML v1 via WebFetch."
 },
 {
  "title": "Do language models plan ahead for future tokens?",
  "url": "https://arxiv.org/abs/2404.00859",
  "authors": "Wilson Wu, John X. Morris, Lionel Levine",
  "year": "2024 (COLM 2024)",
  "finding": "Gradient = diagonal (myopic) + off-diagonal terms through attention to past positions; synthetic task shows pre-caching; natural language mostly breadcrumbs; myopia gap 0.12 nats (GPT-2-124M, 3.28 vs 3.40), growing with Pythia size from 70M; myopic model better early, worse later in a window. Code wiwu2390/FutureGPT2-public.",
  "relevance": "Formalises the user's 'gradients routed across positions'; expect modest pre-caching at 1.7B.",
  "how_checked": "Read arXiv abstract page and HTML v2 via WebFetch."
 },
 {
  "title": "Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs",
  "url": "https://arxiv.org/abs/2406.20086",
  "authors": "Sheridan Feucht, David Atkinson, Byron C. Wallace, David Bau",
  "year": "2024 (EMNLP 2024)",
  "finding": "Linear probes from h_t to tokens at offsets -3..+1 (Llama-2-7b, Llama-3-8b): next-token accuracy rises with depth, current/previous-token accuracy falls; last tokens of multi-token words erase earlier tokens in early layers. Code/data footprints.baulab.info.",
  "relevance": "Backward component B_c(k) and the depth gradient any per-channel arrow must be normalised against.",
  "how_checked": "Read arXiv HTML v3 via WebFetch."
 },
 {
  "title": "Arrows of Time for Large Language Models",
  "url": "https://arxiv.org/abs/2401.17505",
  "authors": "Vassilis Papadopoulos, Jérémie Wenger, Clément Hongler",
  "year": "2024 (ICML 2024)",
  "finding": "Forward models beat backward ones by about 0.6-3.2 % log-perplexity across languages, growing with size (French +0.15 % at 4.9M to +2.65 % at 405M) and context; ideal predictors would show no gap; explained by computational hardness (sparse forward maps, dense inverses). Code frotaur/LLM-Arrows-of-Time.",
  "relevance": "Defines an LM arrow of time; shows it is computational and small; supports inverse-free back-reconstruction.",
  "how_checked": "Read arXiv HTML v4 (first 100k of 106k chars) via WebFetch."
 },
 {
  "title": "Emergent Response Planning in LLMs",
  "url": "https://arxiv.org/abs/2502.06258",
  "authors": "Zhichen Dong, Zhanhui Zhou, Zhixuan Liu, Chao Yang, Chaochao Lu",
  "year": "2025 (ICML 2025)",
  "finding": "MLP probes on prompt representations predict response length, reasoning steps, choices, confidence and factual consistency before generation; three-phase dynamics over the response; larger models plan better within a family (incl. Qwen2.5 1.5B-72B); numbers only in figures.",
  "relevance": "Length is planned internally, so it can mediate; forward planning signal for F_c.",
  "how_checked": "Read arXiv HTML via WebFetch."
 },
 {
  "title": "Unlocking the Future: Exploring Look-Ahead Planning Mechanistic Interpretability in Large Language Models",
  "url": "https://arxiv.org/abs/2406.16033",
  "authors": "Tianyi Men et al. (full list not checked)",
  "year": "2024",
  "finding": "Blocksworld, fine-tuned Llama-2-7b-chat/Vicuna-7B: a step-1 probe predicts step 6 with 0.51; attention extracts the answer more than MLP in layers 15-20; look-ahead accuracy falls with horizon.",
  "relevance": "Future information carried by attention in mid-upper layers.",
  "how_checked": "Read arXiv HTML v1 via WebFetch."
 },
 {
  "title": "Better & Faster Large Language Models via Multi-token Prediction",
  "url": "https://arxiv.org/abs/2404.19737",
  "authors": "Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, Gabriel Synnaeve",
  "year": "2024",
  "finding": "n output heads on a shared trunk; 13B models solve 12 % more HumanEval and 17 % more MBPP problems; up to 3x faster inference with 4-token prediction.",
  "relevance": "Background on MTP heads, which the chain lens needs and Qwen3-1.7B lacks.",
  "how_checked": "Read arXiv abstract page only."
 },
 {
  "title": "Gradient Routing: Masking Gradients to Localize Computation in Neural Networks",
  "url": "https://arxiv.org/abs/2410.04332",
  "authors": "Alex Cloud, Jacob Goldman-Wetzler, Evžen Wybitul, Joseph Miller, Alexander Matt Turner",
  "year": "2024",
  "finding": "Data-dependent gradient masks localise capabilities in chosen subregions (interpretable partitions, robust unlearning, scalable oversight).",
  "relevance": "Terminology check: a training-time tool, distinct from cross-position gradient flow.",
  "how_checked": "Read arXiv abstract page only."
 },
 {
  "title": "Qwen3-1.7B config.json",
  "url": "https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json",
  "authors": "Qwen team",
  "year": "2025",
  "finding": "28 layers, 16 attention heads, 8 KV heads, head_dim 128, hidden 2048, intermediate 6144, use_sliding_window false, tied embeddings, no MTP field.",
  "relevance": "448 head channels, 172,032 FFN neurons (0.1 % = 172); full attention; MTP substitute needed.",
  "how_checked": "Downloaded with curl into its own scratch dir and parsed with python3 -I."
 }
]
```
