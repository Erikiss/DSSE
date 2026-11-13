# tools:single

## Verifier summary
I checked all 24 references against their live primary pages or fresh downloads, and they all exist with the stated titles, authors and dates. 21 hold up as written, one has its facts confirmed but was opened from a local copy (Future Lens), and two needed corrections. The J-lens paper's default target on Sonnet 4.5 is the penultimate layer, not the final one. The Neuronpedia reference lists a non-existent "Qwen3-9B"; the 9B J++ lens is Qwen3.5-9B. Most numbers held exactly: the Neuronpedia fit (466 prompts, 716 s), the cached lens hash, the WorkspaceBench counts and macros, the Qwen3.5 MTP shapes (60.8M parameters), the DeepSeek-V3 figures, Medusa, EAGLE, Patchscopes and Future Lens. The main problems were in the reasoning and the estimates. Option (b) misread the post's neutral-seed control as evidence that the language prior takes over long chains; it only shows the seed token adds little. Option (c) assumed released chain-lens code; none exists, and it also missed that the Qwen3.5 head takes the post-norm hidden state. The fp32 J-lens refit takes about 9-12 h, not "several hours", and Qwen3-1.7B's size and FLOP figures counted the tied embedding twice. The R-lens post found no gain on the smallest models it tested. The post's WorkspaceBench numbers were judged by Gemini, so they cannot be compared with the Opus-judged adapted-set numbers. The list of MTP models up to 9B was also incomplete (more MiMo variants, Meta's 7B multi-token model). I also confirmed Aletheia, H-Neurons and the judge used at the post's commit. The corrected report keeps the recommendation of (b), calibrated by (c), with (a) as a secondary arm.

## Problems found by the verifier
- J++ described as using a 'k-means router': the k-means clusters only partition source positions while the lens is fitted. The 8 weighted expert Jacobians are mixed into one fixed linear map per layer, so nothing is routed at inference and there is no extra inference cost (J++ post, Appendix I; koayon card).
- Ref 'neuronpedia/jacobian-lens' finding says J++ exists for 'Qwen3-4B and 9B': there is no Qwen3-9B. The 9B J++ lens is Qwen3.5-9B (55.0%, Ayonrinde & Lindsey). Qwen2.5-1.5B-Instruct also has J++ (51.5%) and an R-lens.
- J-lens paper: the report presents 'final vs penultimate target' as appendix variants. The paper's default lens on Sonnet 4.5 'used throughout the paper' actually targets the penultimate layer (methods appendix), while the main-text definition and pseudocode name the final layer. The Neuronpedia Qwen3-1.7B lens targets the final block 27.
- 'dim_batch changes memory use, not total FLOPs': the code docstring says total backward FLOPs are unchanged. The single forward is replicated dim_batch times, so forward FLOPs do grow with dim_batch, though they are a small share.
- Neuron Lens timing: 11-16 s per document per lens is stated for both models (Qwen3-8B and Gemma 4 12B). With 1,000 documents that is 3.1-4.4 h per lens, not 'about 3 h'.
- Refit estimate for Qwen3-1.7B on an A100: one prompt is about 9e14 FLOP (2,048 backward rows x 128 tokens x ~3.3 GFLOP, plus the replicated forward). bf16: about 45-70 min for 466 prompts, so 'under an hour' is borderline. fp32 with TF32 off: about 9-12 h, not 'several hours'; that exceeds two 5-hour sessions and needs the code's checkpointing and shard merge. 100 prompts take about 2-2.5 h.
- Qwen3-1.7B size and FLOPs: the checkpoint stores lm_head and embed_tokens separately (2.03B parameters) although they are tied. Loaded tied, it has 1.72B unique parameters, about 6.9 GB in fp32 rather than 8.1 GB. Forward cost is about 3.4 GFLOP per token rather than 4.1. Option (a)'s training cost is therefore about 4.4 GFLOP per token (about 1.2 A100-h bf16 or about 12 h fp32 per 100M tokens), not 5.7 GFLOP (1.6 h / 16 h). The earlier figures are conservative, and the conclusions stand.
- Option (b) risk misreads the post's neutral-seed control as showing that 'the model's own language prior dominates long chains'. The post says neutral-seed chains score similarly once the J-lens seed is substituted for judging, which means the seed token adds little. The corpus-mean-state control hurts, which means the state carries information. Evidence that the prior takes over long chains comes from elsewhere: hallucination grows with chain length, and much longer chains trade more hallucination.
- Option (c) says to 'run the published chain lens unchanged', but no code was released (the post has no code link). It has to be re-implemented from the pseudocode and the three refinements, and the contrastive weight, K, steps, masking and reinjection details must be fixed and pre-registered.
- Option (c) implementation omission: vLLM's Qwen3.5 path feeds the MTP the backbone's post-final-norm state (Qwen3NextModel.forward returns self.norm(...)), while the J-lens maps to the pre-norm block-23 output. Apply the final norm before the head, in addition to using the [emb; hidden] order.
- Option (a): 'expect low t+2 accuracy' overstates Mehra et al., who report that adaptation is non-trivial. Frozen-backbone drafters work in practice: Medusa-1, EAGLE, and AngelSlim's EAGLE-3 for Qwen3-1.7B with accept length 2.13-2.20 at 2 draft tokens. Also, confabulation growing with length 'by construction' is an expectation, not a construction, and the native MTP head in the post shows the same length trend.
- Bottom-line list of <=9B MTP checkpoints is incomplete and should be marked non-exhaustive. MiMo-7B-SFT and MiMo-7B-RL-Zero also ship MTP tensors. XiaomiMiMo/MiMo-7B-MTPs adds two more pretrained MTP layers. Meta's facebook/multi-token-prediction 7B (4-token variants, parallel heads, gated) also contains multi-token heads.
- 'No R-lens or J++ lens exists for Qwen3-1.7B' holds only for the collections checked (Neuronpedia, camilablank/workspace-lenses, koayon/jpp-lenses, several community repos). A Hub-wide absence is unverified.
- R-lens section omits that the post found 'no R-lens advantage for both the smallest dense and MoE models we tested' and that the advantage grows with size. A benefit at 1.7B is therefore not established.
- WorkspaceBench comparability: the chain-lens numbers (eb5a335) were judged by Gemini 3.8 Flash, now verified from the README at that commit, and include the since-retired hallucination family. The Qwen3.5-2B adapted-set numbers (0.209) are judged by Opus 5.5 in summary mode. The repo calls Gemini-judged numbers 'a different instrument', so they must not be compared. In the 2B set arithmetic and chain are 'ooc' and outside the macro.
- Task-family list mixes versions: brew_intermediates existed at eb5a335 but is absent from the current README, and typo is superseded by typo_mt.
- Template lens: excluding it 'by the user's rule' is a conservative reading. The (Sigma+lambda I)^-1 is a fitting step (the paper calls t_w the linear discriminant direction, the same algebra as a ridge probe), not an inversion of the model's map. Its per-word cost and 'skip to the answer' behaviour are independent reasons to drop it.
- Option (b) cost: 0.07 s per cell assumes batching across seeds and cells at 10 TFLOP/s. Unbatched fp32 decoding is memory-bound at about 4.5 ms per step (6.9 GB at 1.56 TB/s), and nulls and revisions add verifier passes. Keep it labelled as an estimate.
- Section 7 is now partly resolved. Aletheia (arXiv 2602.10177, Feng et al., generate-verify-revise) and H-Neurons (arXiv 2512.01797, Gao et al., <0.1% of neurons; models Mistral, Llama, Gemma) exist as cited. The judge at eb5a335 is verified. The facebook/multi-token-prediction file list is verified. The 'common, round numbers' remark is still not found in the J-lens paper text (it may sit in a figure).

## Per-reference / per-fact verdicts
- [verified] The Chain Lens [interim] (vgel, Raymond Douglas; LessWrong, 11 Nov 2026): Authors, date, title and all quoted numbers match the live post: 0.420 vs 0.319 over 28 families; 0.487 vs 0.411 composite over 21; hallucination 0.45 vs 0.32, Patchscopes 0.06; length trend over 19 families; contrastive decoding +0.02-0.03; MTP head about twice the J-lens-transport gain; round-trip rule; judge 57%; ~30x cheaper than trained verbalizers and ~100x dearer than token lenses; ~1 H100-hour for 1.1M neurons; commit eb5a335. The outbound links include no code repository. Added fact: at eb5a335 WorkspaceBench was judged by Gemini 3.8 Flash, and the hallucination family was still part of the bench.
- [corrected] Verbalizable Representations Form a Global Workspace in Language Models (Gurnee et al., Transformer Circuits 6 Jul 2026; arXiv 2607.15495): Existence, authors, dates, formula, '1,000 sequences of 128 tokens', 'as few as 10 prompts' (Sonnet 4.5), template-lens formula, 12,700 words and 'skip to the answer' are all verified. Correction: penultimate is not just an appendix variant. The methods appendix says the default lens on Sonnet 4.5 'used throughout the paper' takes the target at the penultimate layer, while the main-text definition and pseudocode name the final layer. The jlens code and Neuronpedia lenses default to the final block.
- [verified] anthropics/jacobian-lens (jlens/fitting.py and README): fitting.py is byte-identical to the saved copy. The estimator, positions 16..T-2, the mean over prompts, Apache-2.0, 'Reference implementation. Not maintained', '~100 prompts is usable' and the 1000x128 recipe are all verified. Nuance for the report text: the docstring says total backward FLOPs are unchanged by dim_batch, but the single forward is replicated dim_batch times.
- [corrected] neuronpedia/jacobian-lens (README, Qwen3-1.7B config.yaml, convergence CSV): The Qwen3-1.7B lens is verified: 466 prompts, Salesforce/wikitext, final-block target, bf16, dim_batch 128, B200, stop_at_delta 0.002, min 100, window 10, 716 s summed, generated 2026-06-11. The local .pt sha256 equals the LFS oid 6fcc7901...e9d6. Correction: the finding's 'Qwen3-4B and 9B' is wrong because there is no Qwen3-9B. J++ exists for Qwen3.5-0.8B, 2B, 4B and 9B, Qwen3-4B, and also Qwen2.5-1.5B-Instruct, which has an R-lens too.
- [verified] R-lens: Making J-lens More Faithful on Early Layers (Blank, Bhatia, Nanda; LessWrong, 5 Aug 2026): Authors, date, the three LRP rules and 'the cost is negligible' are verified. Missing caveat: the post says 'There is no R-lens advantage for both the smallest dense and MoE models we tested' and that the advantage grows with size. Its released lenses at huggingface.co/camilablank/workspace-lenses do not include Qwen3-1.7B.
- [verified] J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses (Ayonrinde, Lindsey; LessWrong, 8 Oct 2026): The verifier read the full post plus the koayon/jpp-lenses card. Per layer, k-means assigns source positions to 8 expert Jacobians fitted with LRP; signed unit-norm weights are fitted with Adam to a recall@10 surrogate on a dev set disjoint from the evaluation; tokens with no letter or digit are dropped (6,116 for Qwen3.6-27B). Recall@10 is 55.0% (Qwen3.5-9B) and 55.2% (Qwen3.6-27B). Clarification: the experts are mixed into one linear map per layer, so nothing is routed at inference and there is no extra inference cost. Fitting costs about 7% more than a J-lens plus clustering, on 64 sequences of 128 tokens.
- [verified] WorkspaceBench: Evaluating Interpretability Methods for the Global Workspace (Blank, Bhatia, Ong, Nanda; LessWrong, 23 Sep 2026): Verified: 3,356 questions in 27 families; 'developed for Qwen-3.6-27B'; 'may need to be adapted for smaller or weaker models'; the gate (>80% correct over 10 rollouts at T 0.8 without CoT); a prompt-blind summarizer; the Gemini-3.8-Flash judge with Sonnet-5 for safety and regex for the baseline evals; layers 20-60 in steps of 4.
- [verified] camilablank/workspace-bench (README, AGENTS.md, adapted/README.md, Qwen3.5-2B and 0.8B tables, LICENSE): All files are byte-identical to the saved copies. Verified: MIT; Opus 5.5 since 2026-10-29; Sonnet-5 pins; 'a different instrument'; the gate (greedy-correct plus at least 8/10, 10/10 for chain); seven model-specific families; ~0.9M tokens per agentic digest; spend_usd; per-cell caching. Qwen3.5-2B: chain 2, arithmetic 44, buggy 3, poetry 2, multilingual_mt 3 items, 1,461 in total; J-lens macro 0.209 over 14 families. Qwen3.5-0.8B: 0.126 over 11. Both read at layers 8..22, even layers only. Additions: in the 2B set arithmetic and chain are 'ooc' and outside the macro; brew_intermediates was in the bench at eb5a335 but is absent from the current README.
- [verified] stdlogits/neuron-lens (README): Verified: authors 'with Claude'; MIT code and CC BY 4.0 lenses and catalog; Qwen3-8B and Gemma 4 12B; the K-lens and S-lens definitions; R-lens backward is the transpose of S-lens forward; 1,000 DCLM documents of 128 tokens; 11-16 s per document per lens on one H100 (for both models); steering median r 0.45-0.50 at L4-L8, 0.65-0.75 at L12-L16, at least 0.9 from L22 vs ±0.04 (Qwen3-8B). No round-trip criterion is defined there. The report's 'about 3 h per lens' is its own arithmetic and should read 3-4.5 h.
- [verified] DeepSeek-V3 Technical Report (arXiv 2412.19437): Verified against the arXiv HTML v2: the formula h'=M_k[RMSNorm(h);RMSNorm(Emb(t_{i+k}))]; lambda 0.3 for the first 10T tokens and 0.1 after; D=1; MTP modules discardable; sec. 5.4.3 second-token acceptance 85-90% and 1.8x TPS.
- [verified] Better & Faster Large Language Models via Multi-token Prediction (Gloeckle et al., arXiv 2404.19737): Authors and abstract verified. The HF release facebook/multi-token-prediction is gated (manual), licensed 'other', with custom llama code and consolidated.pth for 7B_1T_1, 7B_1T_4, 7B_200B_1 and 7B_200B_4.
- [verified] On multi-token prediction for efficient LLM inference (Mehra, Garcia, Mauch; arXiv 2502.09419): Abstract verified: NTP-specialised hidden layers make adding MTP heads to frozen LLMs non-trivial, and joint training helps but does not fully overcome this. The abstract also says NTP models already have MTP capability through marginalising over intermediate tokens.
- [verified] Medusa (Cai et al., arXiv 2401.10774): Verified: '5 hours for Medusa-1 on Vicuna 7B model with a single NVIDIA A100 PCIE GPU to train on 60k ShareGPT samples'; frozen backbone (Medusa-1); self-distillation.
- [verified] EAGLE (Li, Wei, Zhang, Zhang; arXiv 2401.15077): Verified: second-to-top-layer features before the LM head; embedding and LM head reused from the target; 68,000 ShareGPT dialogues; 7B/13B/33B trained on an RTX 3090 node in 1-2 days; low sensitivity to training data; 0.24B trainable parameters for 7B.
- [verified] EAGLE-3 (Li, Wei, Zhang, Zhang; arXiv 2503.01840): Abstract verified: multi-layer feature fusion, direct token prediction, training-time test.
- [verified] AngelSlim/Qwen3-1.7B_eagle3: config.json verified: LlamaForCausalLMEagle3, 1 layer, hidden 2048, MLP 6144, 16x128 heads, 8 KV heads, draft_vocab_size 32,000. The repo has a custom licence file (License_AngelSlim_model_and_dataset.txt). Accept length for Qwen3-1.7B is 2.13-2.20 on vLLM v0.11.2 (H20, num_speculative_tokens=2).
- [verified] Patchscopes (Ghandeharioun et al., arXiv 2401.06102): Verified: 'From layer 10 and upwards, the token identity method ... consistently outperforms'; in the first 10 layers token identity is on par with the logit lens and the tuned lens is slightly better.
- [verified] Future Lens (Pal, Sun, Yuan, Wallace, Bau; arXiv 2311.04897): The abstract confirms 'more than 48% accuracy' on GPT-J-6B. The bigram baseline of 20.1% is in the arXiv PDF text (local copy).
- [verified] Qwen3.5 small models (0.8B/2B/4B/9B and Base): config, index, header, card: Fresh HF API, configs, indexes and a range-read header confirm: 0.87B, 2.27B, 4.66B and 9.65B parameters; mtp_num_hidden_layers 1 and 15 mtp.* tensors in all of them, Base included. The 2B MTP module has 60,828,160 parameters (fc 2048x4096; q_proj 4096x2048, i.e. 8x256 heads with an output gate; k/v 512; MLP 6144; no separate lm_head). Hybrid 18 linear plus 6 full attention layers (0.8B/2B); vocab 248,320; card method qwen3_next_mtp; Apache-2.0. Qwen3.6 has only 27B and 35B-A3B.
- [verified] MiMo-7B-Base / RL (config, index, modeling_mimo.py, card): Verified: num_nextn_predict_layers 1; model.mtp_layers.0.* (16 tensors); MiMoMTPLayers concatenates [hidden; embedding] and is never called by the inherited Qwen2Model forward; the card says MTP was tuned in pretraining and SFT, frozen in RL, with ~90% acceptance; vLLM fork and SGLang; 7.83B parameters; MIT. Additions: MiMo-7B-SFT and MiMo-7B-RL-Zero also ship MTP tensors, and XiaomiMiMo/MiMo-7B-MTPs provides two more pretrained MTP layers (mtp_layers.1 and .2).
- [verified] transformers model code (qwen3_5, qwen3_next, glm4_moe, deepseek_v3): The main branch has _keys_to_ignore_on_load_unexpected = ^mtp.* (Qwen3.5, also with ^model.visual.*, and Qwen3-Next), model.layers.92/46 (GLM-4.5) and model.layers.61 (DeepSeek-V3).
- [verified] vLLM MTP drafters (qwen3_5_mtp, mimo_mtp, deepseek_mtp, gemma4_mtp, speculative.py): Verified: the concatenation order is [emb; hidden] for Qwen3.5, Qwen3-Next, DeepSeek, GLM, ERNIE and Gemma4, and [hidden; emb] for MiMo; the MTP type list includes qwen3_5_mtp, mimo_mtp, glm_ocr_mtp and gemma4_mtp; the Gemma4 drafter is KV-shared and Q-only. Added detail: Qwen3_5Model inherits Qwen3NextModel.forward, which returns self.norm(hidden, residual), so the MTP receives the post-final-norm backbone state.
- [verified] Gemma 4 MTP drafters (google/gemma-4-E2B-it-assistant, -E4B-it-assistant): Verified: 77.99M and 78.78M parameters; 4 layers; hidden 256; backbone_hidden_size 1536 and 2560; 2,048 centroids with ordered embeddings; num_kv_shared_layers 4; used through generate(assistant_model=...); Apache-2.0. The base gemma-4-E2B and E4B configs have no MTP.
- [verified] GLM-OCR and GLM-4.5-Air / GLM-4.7-Flash: A fresh header read confirms GLM-OCR (1.33B) layer 16 has eh_proj, enorm, hnorm and shared_head. GLM-4.5-Air (110.5B) and GLM-4.7-Flash (31.2B) have nextn layers. GLM-5.3-Flash (321B) also has MTP, which is above 9B.

## Corrected report
# Round 1b, H-CHAIN: a chain lens for Qwen3-1.7B, which has no MTP head

Scope: the five questions of the task. The sources were opened in this session using WebFetch, raw files, the Hugging Face API, safetensors headers read by range request, and Playwright for LessWrong. An independent verifier re-checked them on 2026-11-13:
- Fresh downloads were compared byte for byte with the saved copies: the J-lens paper HTML, `jlens/fitting.py`, the Neuronpedia README, config and log, the WorkspaceBench files and the Neuron Lens README. All were identical.
- The cached Qwen3-1.7B lens matches its Hugging Face LFS hash.

Items marked **estimate (mine)** are arithmetic; where the verifier changed one, the text says so. Items marked **(unverified)** could not be checked. Date: 2026-11-13.

## 0 Bottom line

1. **The J-lens is from Anthropic.**
   - Paper: Gurnee, Sofroniew, ..., Batson, Lindsey, *Verbalizable Representations Form a Global Workspace in Language Models* (Transformer Circuits, 6 Jul 2026; arXiv 2607.15495, 16 Jul 2026).
   - Code: `anthropics/jacobian-lens` (Apache-2.0, a reference implementation marked "not maintained").
   - Method: J_l is a plain average of exact Jacobians, computed with vector-Jacobian products. It involves no regression and no inversion. Under the user's rule it is a fitted forward map, so it stays allowed.
2. **A J-lens for Qwen3-1.7B already exists.**
   - Neuronpedia fitted it on 466 WikiText prompts. It maps blocks 0-26 to the final block 27 in fp16.
   - The local copy is verified: its sha256 equals the Hugging Face LFS object id.
   - No R-lens or J++ lens for Qwen3-1.7B is in Neuronpedia's collection, `camilablank/workspace-lenses`, `koayon/jpp-lenses`, or the community lens repos the verifier checked. That none exists anywhere on the Hub is (unverified).
3. **WorkspaceBench is public but does not fit this model size.**
   - The repo is `camilablank/workspace-bench` (MIT). It was built and gated on Qwen3.6-27B and is now judged by Claude Opus 5.5 over the API.
   - The authors re-gated it for Qwen3.5-2B. That set keeps only **2** chain_intermediates items and 44 arithmetic items. Both families are marked out of capability ("ooc") and left out of the macro.
   - The J-lens macro there is 0.209 over 14 families. There is no set for Qwen3-1.7B, so at this size WorkspaceBench cannot measure the families where the chain lens is strongest.
   - These adapted-set numbers (Opus 5.5, summary mode) cannot be compared with the chain-lens post's numbers. The post's numbers were judged at commit eb5a335 by Gemini 3.8 Flash, and the repo calls Gemini-judged numbers "a different instrument".
4. **The round-trip criterion comes from the chain-lens post itself, not from Neuron Lens.** It is a forward check through the unmodified model, so it already has the shape of the Aletheia-style verifier the user prefers.
5. **Open models up to about 9B whose released checkpoints contain MTP weights.** This is what was found; the search was not exhaustive.
   - Qwen3.5-0.8B, 2B, 4B and 9B, base and post-trained. Each has one MTP layer with 15 `mtp.*` tensors.
   - MiMo-7B (Base, SFT, RL-Zero, RL, RL-0530), one MTP layer each. The separate repo `XiaomiMiMo/MiMo-7B-MTPs` adds two more pretrained MTP layers (`mtp_layers.1`, `.2`). Xiaomi says it has not tested them with the post-trained models.
   - GLM-OCR (1.3B), which is an OCR model.
   - Meta `facebook/multi-token-prediction` (7B). Its 4-token variants have four parallel output heads, not DeepSeek-style sequential modules. It is gated and ships custom llama code with `consolidated.pth`.
   - Gemma 4 E2B and E4B have no MTP weights in the base checkpoint. They come with separate drafter models of about 78M parameters that read the target model's KV cache.
   - Qwen3.6 has no model at or below 9B.

   Loading support is limited:
   - transformers drops the Qwen3.5 MTP weights on load.
   - MiMo's remote code builds its MTP layer but never calls it.
   - vLLM exposes both only as speculative-decoding drafters.
6. **Recommendation.**
   - **Primary: (b) self-rollout through Qwen3-1.7B itself.** It needs no training and no inversion. Add a forward round-trip verifier, a fixed attempt limit and counted abstentions.
   - **Calibrate (b) with (c) on Qwen3.5-2B.** That model has the same width (2048), a native MTP head, and ready J-lens and J++ lenses. Re-implement the chain lens from the post's description (no code was released), then run it and (b) on the same Qwen3.5-2B states.
   - **Keep (a), a post-hoc trained head, as a secondary arm only.** It is training, it brings its own language prior, and that prior confounds the length-versus-attractor question of H-CHAIN.

## 1 J-lens, R-lens, template lens, J++

### J-lens estimator (read in full in `jlens/fitting.py`)
- **Gradient step.** For each output dimension of the target block, a one-hot cotangent is set at every valid target position at once and backpropagated. The code defaults to the final block. The gradient at source position p is then sum over p' >= p of dh_final[p']/dh_l[p].
- **Averaging.** Rows are averaged over valid source positions 16..T-2. The first 16 positions are excluded as attention sinks, and the last position has no target. The per-prompt maps are then averaged over prompts.
- **Cost per prompt.** One forward pass on the prompt replicated dim_batch times, plus ceil(d_model/dim_batch) backward passes. dim_batch changes memory use but not the total backward FLOPs; only the replicated forward grows with it.
- **Readout.** lens_l(h) = unembed(J_l h), where unembed is the final RMSNorm followed by the LM head. The paper writes this as softmax(W_U norm(J_l h)).
- **Defaults and data needs.**
  - The paper uses 1,000 sequences of 128 tokens of pretraining-like text.
  - It states that the "J-lens beats the logit lens and tuned lens baselines with as few as 10 prompts" (measured on Sonnet 4.5).
  - The README says "~100 prompts is usable".

### Recipe variants in the paper's appendix
- **Target block.**
  - The paper's default lens on Sonnet 4.5, "used throughout the paper", takes the target at the **penultimate** layer. The main-text definition and the pseudocode name the final layer.
  - Including the last block "can sometimes increase ... noisy artifacts".
  - The jlens code and the Neuronpedia Qwen3-1.7B lens target the final block.
- **Frozen-QK.** Stop-gradients on QK hold the attention pattern fixed; this "can increase the causal effect".
- **Target positions.**
  - The default is t' >= t. Two variants are self-only (t' = t, with cross-position attention zeroed) and future-only (t' > t).
  - The paper says the future-only variant "isolates the broadcast component, i.e. what h_l,t makes available to downstream positions".
- **Aggregation.** Mean versus median, with outlier filters.
- **Overall.** "methods are fairly consistent among these design choices". Mean aggregation with the penultimate target is "a small improvement in extracting intermediates".
- **Early layers.** In roughly the first third of depth, no pair of lenses agrees and the readouts are noisy.

### Cost
- **Qwen3-1.7B on Neuronpedia.**
  - 466 prompts, fit dated 2026-06-11.
  - The fit stopped once the mean relative change fell below 0.002 (minimum 100 prompts, window 10).
  - bf16, dim_batch 128, one B200.
  - 716 s summed over the convergence log, about 1.5 s per prompt.
- **Neuron Lens, for scale.** Its README reports 11-16 s per 128-token document per lens on one H100, for both its models (Qwen3-8B and Gemma 4 12B). With its 1,000 documents that is about 3-4.5 h per lens (my arithmetic).
- **Estimate (verifier's arithmetic, replacing the earlier "under an hour / several hours").** One prompt costs about 9e14 FLOP for Qwen3-1.7B: 2,048 backward rows x 128 tokens x about 3.3 GFLOP per token and row, plus the replicated forward. On one A100:
  - **bf16:** 466 prompts take about 45-70 min, at 35-50 % of the 312 TFLOP/s peak.
  - **fp32 with TF32 off:** 466 prompts take about 9-12 h, against a 19.5 TFLOP/s peak. That is more than two 5-hour sessions; the code checkpoints after each prompt and merges shards.
  - **100 prompts in fp32:** about 2-2.5 h. The README calls 100 prompts "usable".
- A refit is only needed for variants: self-only, future-only, penultimate target, or R-lens rules.

### Coverage of Qwen models (Neuronpedia README, fetched today; identical to the 10 Nov copy)
- **J-lenses:** Qwen3-1.7B, 4B, 8B, 14B and 32B; Qwen3.5-0.8B, 2B, 2B-Base, 4B and 9B-Base; Qwen2.5-1.5B-Instruct and 7B-Instruct. Larger ones: Qwen3.5-27B and Qwen3.6-27B.
- **J++ lenses (recall@10):** Qwen3-4B 53.8%, Qwen3.5-0.8B 44.9%, Qwen3.5-2B 51.5%, Qwen3.5-4B 57.2%, Qwen3.5-9B 55.0%, Qwen2.5-1.5B-Instruct 51.5%.
- **R-lenses:** Qwen3.5-4B, Qwen3-14B and Qwen2.5-1.5B-Instruct.
- **File format:** `J[l]` is fp16 [d, d], and logits = unembed(h @ J[l].T).

### R-lens
Blank, Bhatia and Nanda, LessWrong, 5 Aug 2026. It is the J-lens fitted with layer-wise relevance propagation (LRP) rules in the backward pass:
- RMSNorm denominator held constant;
- SiLU/GELU factor detached;
- relevance split half and half across the gate-times-up product.

These are stop-gradients only; the post says "the cost is negligible".

The post reports cleaner and more causal readouts in early layers, with an advantage that grows with model size. It also found "no R-lens advantage for both the smallest dense and MoE models we tested", so a gain at 1.7B is not established. Its released lenses (`camilablank/workspace-lenses`) do not include Qwen3-1.7B.

### Template lens (J-lens paper appendix)
- **Data.** For about 12,700 words, Claude writes passages that end just before the word.
- **Template.** Each word's template is the centred and whitened mean of the final-position activation: t_w = (Sigma + lambda I)^-1 (mu_w - mu).
- **Readout and cost.** The readout projects an activation onto the templates. Building a template takes a few hundred forward passes per word.
- **Early layers.** Like the tuned lens, it can "skip to the answer" in early layers.
- **Status under the user's rule.** We leave it out, but this is a conservative reading.
  - For exclusion: it uses a ridge-regularised (Tikhonov) inverse covariance, and the brief lists Tikhonov-regularised inverses as not allowed for reconstruction.
  - Against: the inverse is a fitting step. The paper calls t_w the linear discriminant direction, which has the same algebra as a ridge probe. It is not an inversion of the model's map.
  - Its cost and its early-layer behaviour are independent reasons to drop it.
- **Oracle lens.** It is built on the template lens and needs two fine-tuned auxiliary models: a reconstructor, and an RL-trained oracle scored by a sparse non-negative reconstruction. It is training, so it is out.

### J++
Ayonrinde and Lindsey, LessWrong, 8 Oct 2026. Per layer:
- **Clusters.** k-means (8 clusters, on 64-dimensional PCA projections of unit-normalised, centred activations) assigns source positions to 8 expert Jacobians.
- **Experts.** Each expert Jacobian is fitted with the R-lens rules.
- **Weights.** Signed expert weights (unit L2 norm) are fitted with Adam to a recall@10 surrogate. They use a labelled dev set that is disjoint from the evaluation items.
- **Readout filtering.** Tokens with no letter or digit are dropped (6,116 tokens for Qwen3.6-27B).
- **Inference.** The weighted experts are mixed into **one fixed linear map per layer**. Nothing is routed at inference, and there is no extra inference cost.
- **Fitting.** About 7 % more compute than a J-lens fit, plus a one-off clustering step, on 64 sequences of 128 tokens.

All maps are forward maps, but the weights are tuned on items of the same task types as the evaluation.

## 2 WorkspaceBench

### What exists
- **Write-up:** Blank, Bhatia, Ong and Nanda, LessWrong, 23 Sep 2026. 3,356 questions in 27 families, "developed for Qwen-3.6-27B", "may need to be adapted for smaller or weaker models".
- **Repo:** `github.com/camilablank/workspace-bench`. Code and in-house items are MIT; third-party data keep their own terms.
- **Commit used by the chain-lens post:** eb5a335. The post reports a decoding-bug fix (byte-level BPE) worth +0.02 to +0.03 per family. At that commit:
  - every family was judged by `google/gemini-3.8-flash`, with three exceptions: agentic_misalignment and jailbreak_recognition used `claude-sonnet-5`, and introspection used `claude-opus-5-5`;
  - the seven multi-token families were regex-scored;
  - the hallucination family was still part of the bench.

### Task families (as in the post; repo changes noted)
- **Basic:** basic readout, multihop, multilingual, poetry, typo (now superseded by typo_mt), directed modulation.
- **Multi-token:** multihop_mt, typo_mt, basic_readout_mt, multilingual_mt, multilingual typo, multilingual multihop, multi-concept directed modulation.
- **Computational:** chain intermediates, brew intermediates (present at eb5a335; not in the current README), buggy code, arithmetic intermediates.
- **Safety:** agentic misalignment, jailbreak recognition, introspection.
- **Association:** basic, conjunctive, user modelling, role-bound.
- **Order-sensitive:** relational multihop.
- **Other:** J-lens concept precision/recall, moral rationale.
- **Model-specific (added in the repo):** wrong_intermediate, knowledge_cutoff, hinted_cutoff, model_multihop, nyt_connections, model_specific_directed_modulation.
- The hallucination family moved to the separate HallucinationBench on 2026-10-30 (plan 0040).

### How a readout is judged
- **Read cells.** The lens is read at (layer, position) cells. Token lenses give their top-10. For Qwen3.6-27B these were layers 20-60 in steps of 4.
- **Digest step (summary mode, the default).** All of an item's cells go to one prompt-blind digest written by `claude-opus-5-5`. For multi-layer reads, it writes one digest per layer, then a note-taker merges them.
- **Judge step.** The family's judge then decides the item.
  - Most judges are prompt-blind multiple choice and require a verified quote.
  - Arithmetic uses free recall.
  - Several multi-token families are scored by regex.
- **Pinned judges.** agentic_misalignment and jailbreak_recognition stay pinned to `claude-sonnet-5`.
- **Floors and subsets.** Scores are reported against lucky-guess and prompt-only (Opus 5.5) floors, plus an anti-text-inversion subset.
- **Judge history.** The Sep post used Gemini-3.8-Flash (Sonnet-5 for safety). The repo switched to Opus 5.5 on 2026-10-29 and notes that "Numbers judged on Gemini are a different instrument". The chain-lens post's numbers (eb5a335) are on the Gemini instrument.

### Cost
- No dollar figure is given.
- `wsbench list` prints calls per arm, and `run.json` records `spend_usd`. Verdicts and digests are cached per cell, so a re-run only pays for what is missing.
- Agentic digests read up to about 0.9M tokens per item.
- **Estimate (mine):** about 15,000-18,000 Opus calls per arm for a set the size of Qwen3.5-2B's adapted set (1,461 items, 8 layers).

### Running it on a 1.7B model
- **Producing readouts.** The bundled producer runs `logit_lens | jlens | rlens | olens | nla` over Hugging Face transformers. Any lens can also hand over JSONL readouts.
- **Re-gating.** Every bank must be re-gated on the subject model, or an adapted set built.
  - Items must be greedy-correct and correct in at least 8 of 10 samples (10 of 10 for the chain family).
  - Seven families are model-specific and must be re-gated or regenerated for the new model.
- **Keys.** `wsbench capable` needs OpenRouter for the subject model and Anthropic keys for grading. Whether Qwen3-1.7B is available on OpenRouter is (unverified).
- **Existing adapted sets.** Qwen3.5-0.8B, 2B, 4B and 9B, Qwen3-8B and OLMo-3-7B, plus larger models. There is none for Qwen3-1.7B.
- **Qwen3.5-2B adapted set** (`adapted/qwen3.5-2b/lenses/table.md`):

  | family | items kept |
  |---|---|
  | chain_intermediates | 2 (ooc, out of macro) |
  | arithmetic_intermediates | 44 (ooc, out of macro) |
  | buggy_code | 3 |
  | poetry | 2 |
  | multilingual_mt | 3 |

  The Neuronpedia J-lens macro over the 14 macro families is 0.209 [0.18, 0.23], judged by Opus 5.5 in summary mode. It is not comparable with the post's Gemini-judged numbers on Qwen3.6-27B.
- **Qwen3.5-0.8B adapted set:** J-lens macro 0.126 over 11 families. It has no chain or arithmetic items at all.
- The 2B and 0.8B sets are read at the even layers 8-22, because their lenses cover even layers only.

## 3 Neuron Lens, K-lens and S-lens, and the round trip

### Neuron Lens
`github.com/stdlogits/neuron-lens`, by a9lim and treetowntree "with Claude". The code is MIT; the lenses and catalog are CC BY 4.0.
- **Coverage:** every MLP neuron (both poles) and every attention head of Qwen3-8B and Gemma 4 12B.
- **Lens definitions:**
  - **J/R lenses:** J_L = mean_t sum_{t'>=t} dh_final(t')/dh_L(t), read with W_U. These look forward.
  - **K-lens:** credited to Theia Vogel (@voooooogel; an unpublished tweet, not opened). K_L = mean_t sum_{s<=t} dh_L(t)/de(s). A direction c scores each token v by c . K_L (W_E[v] - mean W_E), that is, which input tokens put c there.
  - **S-lens:** the K-lens under RelP rules. The R-lens backward map is the transpose of the S-lens forward map.
- **Fitting:** 1,000 DCLM documents of 128 tokens each, at 11-16 s per document per lens on one H100.
- **Result (Qwen3-8B):** the R-lens read of a write vector predicts the steering effect with median r 0.45-0.50 at L4-L8, 0.65-0.75 at L12-L16 and at least 0.9 from L22. Another neuron's read gives plus or minus 0.04.
- These are exact linearizations averaged over contexts. They score candidates and invert nothing.

### Round trip (defined in the chain-lens post, not in Neuron Lens)
- **How readouts are made.** The chain lens runs on a mean WikiText hidden state. The neuron's write vector is added (both signs) at the scale of the mean state's norm.
- **When a readout round-trips.** Both conditions must hold:
  1. its best chain activates the neuron above the neuron's 99.8th-percentile WikiText activation;
  2. its top three chains activate the neuron more than 90% of random readouts of other neurons from the same layer do.
- **Behaviour check.** A judge picks, out of four texts steered by four different neurons, the one that matches the readout. It is right 57% of the time, against 25% chance.
- **Fit with the user's rule.** This is a generator (MTP rollouts) plus a verifier decoupled from it (the unmodified model). To be fully Aletheia-style it still lacks a revise step, an attempt limit with counted abstentions, and the verifier's error rate on cases with known answers.
- The post did not test the K-lens or S-lens.

## 4 Open models up to about 9B with MTP heads

DeepSeek-V3 MTP (arXiv 2412.19437, sec. 2.2; acceptance numbers in sec. 5.4.3):
- **Module.** h'_i^k = M_k [RMSNorm(h_i^{k-1}); RMSNorm(Emb(t_{i+k}))], followed by one transformer block, with a shared embedding and a shared output head.
- **Training and inference.** Loss weight lambda = 0.3 for the first 10T tokens, then 0.1, with depth D = 1. The module can be discarded at inference.
- **Speed.** The second-token acceptance rate is 85-90%, giving 1.8 times the tokens per second.
- **Concatenation order differs between implementations.** vLLM's DeepSeek, GLM, ERNIE, Qwen3-Next/Qwen3.5 and Gemma 4 code uses [embedding; hidden]; MiMo uses [hidden; embedding]. Follow the checkpoint, not the paper's notation.
- **Which hidden state the head reads.** In vLLM's Qwen3.5 path, `Qwen3NextModel.forward` returns `self.norm(hidden, residual)`, so the MTP receives the backbone's post-final-norm state. A J-lens maps to the pre-norm block output, so apply the final norm before the head.

| model | params (HF API) | MTP in released checkpoint | transformers | vLLM | lenses on Neuronpedia | notes |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B (+Base) | 0.87B | yes, 15 `mtp.*` tensors; `mtp_num_hidden_layers: 1` | loads, drops `^mtp.*` | `qwen3_5_mtp` (card: `{"method":"qwen3_next_mtp"}`) | J-lens, J++ | d 1024, 24 layers (18 linear-attention, 6 full-attention), vocab 248,320, multimodal checkpoint, Apache-2.0 |
| **Qwen3.5-2B** (+Base) | 2.27B | yes; MTP module 60.8M params (see note A) | drops `^mtp.*` | yes | J-lens (283 prompts), J++ 51.5% | **d 2048 and MLP 6144, the same as Qwen3-1.7B**; WorkspaceBench adapted set exists |
| Qwen3.5-4B (+Base) | 4.66B | yes | drops | yes | J-lens, J++, R-lens | d 2560 |
| Qwen3.5-9B (+Base) | 9.65B | yes | drops | yes | J-lens (9B-Base), J++ (9B) | d 4096 |
| MiMo-7B-Base / SFT / RL-Zero / RL / RL-0530 | 7.83B | yes, `model.mtp_layers.0.*` (16 tensors); `num_nextn_predict_layers: 1` | remote code builds `MiMoMTPLayers`; the inherited forward never calls it | `mimo_mtp` (card recommends Xiaomi's vLLM fork; SGLang) | none | card: MTP tuned in pretraining and SFT, frozen in RL, about 90% acceptance; MIT. `MiMo-7B-MTPs` adds pretrained MTP layers 1 and 2 (untested by Xiaomi with post-trained models) |
| Meta multi-token-prediction (7B_1T_4, 7B_200B_4) | 7B | four parallel output heads (Gloeckle et al.), not sequential | no (custom llama code, `consolidated.pth`) | - | none | gated (manual), licence "other"; weights not inspected |
| Gemma 4 E2B / E4B | 5.12B / 8.00B | not in the base checkpoint; separate `-it-assistant` drafters (see note B) | `generate(assistant_model=...)` | `gemma4_mtp` | J-lens (base models) | not self-contained |
| GLM-OCR | 1.33B | yes (layer 16: `eh_proj`/`enorm`/`hnorm`/`shared_head`) | not checked | `glm_ocr_mtp` | none | OCR model |

Notes to the table:
- **A (Qwen3.5-2B MTP module).** 60,828,160 parameters in total, made of:
  - `fc` 2048x4096;
  - one gated full-attention layer: 8 query heads of 256 plus an output gate (q_proj 4096x2048), and 2 KV heads;
  - an MLP of 6144;
  - a final norm;
  - the shared (tied) LM head.
- **B (Gemma 4 drafters).** 78.0M (E2B) and 78.8M (E4B) parameters. A `pre_projection` maps [embedding; backbone hidden] (1536 or 2560) down to a hidden size of 256. Four layers follow, with query-only attention that reads the target model's KV cache, and centroid-masked logits.

Other checks:
- **MTP present but above 9B:** GLM-4.5-Air (110.5B), GLM-4.7-Flash (31.2B), GLM-5.3-Flash (321B), ERNIE-4.5-21B-A3B (21.9B), Qwen3.6-27B and Qwen3.6-35B-A3B.
- **No MTP found:** Qwen3-1.7B, GLM-4-9B-0414, GLM-4.1V-9B, GLM-4.6V-Flash, ERNIE-4.5-0.3B, Hunyuan-1.8B and 7B, MiniCPM4.1-8B, Ling-mini-2.0 (16B, nextn 0), MiMo-VL-7B, Nemotron-3-Nano-4B, Nemotron-Flash-3B, Nemotron-3.5-Bolt-E4B.
- **Config without weights.** MiMo-V2.6-Distill-Qwen-9B declares `mtp_num_hidden_layers: 1` but its index has no `mtp` tensors.
- **transformers code.** `modeling_qwen3_5.py` sets `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`. The same mechanism drops MTP for Qwen3-Next, GLM-4.5 (layers 46 and 92) and DeepSeek-V3 (layer 61).

## 5 Options for Qwen3-1.7B

The options are listed in detail in the structured field. In short:

- **(b) Self-rollout, recommended.**
  - **Rollout.** Patch h_l,t into Qwen3-1.7B at the same block, after a fixed neutral prefix of at least 16 tokens. Take the J-lens top-K tokens as seeds and let the model continue each seed.
  - **Contrastive decoding.** Decode against the same procedure run with the corpus-mean state mu_l.
  - **Chain-restricted context.** Let chains see only the patched position plus the fixed prefix. Optionally re-add (h - mu) at each chain position, with its damage controlled as in round 2.
  - **Verifier.** Feed each chain back as plain text into unmodified Qwen3-1.7B. Accept it only if the block-l state at the chain reproduces the centred h beyond length-matched nulls (mean-state, foreign-state and shuffled chains).
  - **Revise, limit, abstain.** Revise by swapping in the next contrastive-decoding candidate, cap the number of attempts, and count abstentions.
  - **Verifier error rate.** Measure it on states whose continuation is known (teacher-forced).
  - **Cost, estimate (mine, corrected by the verifier).**
    - Memory: about 6.9 GB in fp32 with the tied embedding/LM-head matrix loaded once (the checkpoint also stores a separate `lm_head` copy, 2.03B parameters in total).
    - Compute: about 3.4 GFLOP per chain token, and about 0.6-0.7 TFLOP per read cell.
    - Time: about 0.07 s per cell at an assumed 10 TFLOP/s. This needs batching across seeds and cells; unbatched fp32 decoding is memory-bound at about 4.5 ms per step.
  - **Main risk.** The model's own language prior can take over long chains. The post's evidence is that hallucination grows with chain length and that much longer chains trade more hallucination. The post's neutral-seed control does **not** show this: it shows the seed token adds little, while the corpus-mean-state control shows the state carries information.
- **(c) Qwen3.5-2B (and 0.8B as a size check).**
  - This is the chain lens as described in the post, re-implemented, since no code was released. The contrastive weight, K, chain length, masking and reinjection details must be fixed and pre-registered.
  - Use it to calibrate (b), and (a) if run, against the native head on identical states, and as an independent replication of H-CHAIN.
  - Load the MTP manually: [emb; hidden] order, with the final norm applied to the J-lens-transported state.
  - It is a different network, so it cannot replace Qwen3-1.7B.
  - The 0.8B adapted WorkspaceBench set has no chain or arithmetic items.
- **(a) Post-hoc DeepSeek-style head on frozen Qwen3-1.7B.** About 58.7M trainable parameters with the tied LM head frozen, trained on self-distilled data. This is a secondary arm only.
  - **Cost, estimate (mine, corrected).** About 4.4 GFLOP per training token: about 1.2 A100-hours in bf16 or about 12 h in fp32 per 100M tokens.
  - **Accuracy.** Mehra et al. (2025) report that adding MTP heads to frozen next-token-specialised backbones is non-trivial. Frozen-backbone drafters still work in practice: Medusa-1, EAGLE, and AngelSlim's EAGLE-3 head for Qwen3-1.7B, which reaches an accept length of 2.13-2.20 with 2 draft tokens. So expect lower t+2 accuracy than a co-trained head, not necessarily low accuracy.
  - **Prior.** The head's own prior is expected to make confabulation grow with chain length. The native head in the post shows the same trend, so the length trend alone does not separate the two.
  - **Constraint.** It means training an auxiliary model, which should be checked against the round-1b constraint against training.
- **(a') Reuse `AngelSlim/Qwen3-1.7B_eagle3` as is.** It needs no training but is not faithful: it fuses features from three layers and uses a 32k draft vocabulary.
- **Carry over the post's controls:** token-matched J-lens top-50 against 10 chains, neutral seeds, foreign seeds, corpus-mean state, shuffle nulls, and a chain-length sweep.

## 6 What this means for testing H-CHAIN

- **What the post shows and does not show.**
  - Hallucination and performance both grow with chain length (19 families).
  - The chain lens confabulates more than single-token lenses: 0.45 against 0.32 for the J-lens, on the then-current hallucination family under the Gemini judge.
  - Patchscopes had the lowest hallucination of the training-free lenses (0.06).
  - The post does not separate length from mechanism.
- **Where H-neuron activity can be measured.** Under (b), every chain step runs through Qwen3-1.7B's own MLPs, so H-neuron activity per step can be measured in the same network. Under (a), the head writes the chain and only the seed state passes through the model.
- **The H-neuron set has to be identified first.** Gao et al. (2025; arXiv 2512.01797) found H-neurons in Mistral, Llama and Gemma models (0.01-0.35 per mille of neurons). No published H-neuron set for Qwen3-1.7B was found, so it must be identified on Qwen3-1.7B with the paper's classifier method. That none exists is (unverified).
- **Outcome that counts against the attractor reading.** H-CHAIN can come out negative: confabulation and H-neuron share rise with step k equally for real states and for length-matched null states (mean state, foreign state).
- **Outcome that supports it.** Either of these would favour an attractor:
  - a real-minus-null excess at matched length;
  - a causal clamp of the H-neuron set that reduces confabulation more than matched random neuron sets do.
- **A measurable form of "a front predicts following versus preceding fronts".** Published forward-only tools exist for this:
  - the J-lens future-only variant (t' > t) against the self-only variant (t' = t);
  - the K/S-lens, which looks back to s <= t.

  All of them average exact linearizations and invert nothing. Using their ratio per direction as the user's "vector quantity" is my suggestion, not a published method.

## 7 Not verified or not opened

**Resolved by the verifier:**
- Aletheia: Feng et al., "Towards Autonomous Mathematics Research", arXiv 2602.10177, Feb 2026. It iteratively generates, verifies and revises solutions.
- H-Neurons: Gao et al., arXiv 2512.01797, Dec 2025. Fewer than 0.1% of neurons; causally linked to over-compliance.
- The judge in force at WorkspaceBench commit eb5a335: Gemini 3.8 Flash.
- The file list of `facebook/multi-token-prediction`.

**Still unverified:**
- The chain-lens post's remark that the J-lens paper's workspace ablation produced "common, round numbers". It is not found in the paper's text; it may sit in a figure.
- Dollar prices.
- The K-lens tweet (x.com, not opened).
- The weights inside `facebook/multi-token-prediction` (gated).
- Whether any community R-lens or J++ lens for Qwen3-1.7B exists on the Hub.
- Whether Qwen3-1.7B passes any WorkspaceBench gate.
- Whether Qwen3.5 runs under the transformers versions available on Colab.
- All A100 timings (estimates).

## 8 Local files (absolute paths)

- **J-lens for Qwen3-1.7B (Neuronpedia):** `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/jlens/Qwen3-1.7B_jacobian_lens.pt`. The file has 226,501,315 bytes, which equals 27 fp16 2048x2048 maps plus 8,899 bytes. Its sha256 (6fcc7901...e9d6) matches the Hugging Face LFS oid. Config and convergence log are in the same folder.
- **J-lens estimator source:** `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/hchain_mtp/dl_jlens_repo/jlens_fitting.py`
- **J-lens paper text:** `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/hchain_mtp/dl_jlens_paper/page.txt`
- **WorkspaceBench files** (README, AGENTS, adapted README, Qwen3.5-2B and 0.8B score tables): `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/hchain_mtp/dl_wsb_repo/`
- **WorkspaceBench post:** `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/hchain_mtp/dl_wsb_lw/post.json`
- **R-lens post:** `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/hchain_mtp/dl_rlens_lw/post.json`
- **Neuron Lens README:** `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/hchain_mtp/dl_neuronlens/README.md.txt`
- **Hugging Face configs and weight indexes:** `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/hchain_mtp/dl_hf_cfg/`, `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/hchain_mtp/dl_hf_cfg2/`, `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/hchain_mtp/dl_hf_cfg3/`
- **Qwen3.5-2B MTP tensor shapes:** `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/hchain_mtp/dl_st_headers/q35_2b.header.json`
- **transformers and vLLM sources:** `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/hchain_mtp/dl_code/`
- **EAGLE-3 head config for Qwen3-1.7B:** `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/hchain_mtp/dl_angelslim/config.json`
- **Verifier's fresh downloads:** `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/vchk/`. This holds the J-lens page, jlens repo, Neuronpedia files, WorkspaceBench at main and at eb5a335, Neuron Lens, arXiv pages, Hugging Face API, configs and headers, vLLM and transformers sources, the MiMo code and card, the koayon card, and the J++ and chain-lens posts.


## options_for_qwen3_1p7b (from the research stage; check against the verifier's notes above)
```json
[
 {
  "option": "(b) Self-rollout through Qwen3-1.7B itself (Patchscopes-style, recommended primary). Patch h_l,t at block l into a fixed neutral prefix of at least 16 tokens; take the J-lens top-K tokens as seeds (Neuronpedia lens, already available); let Qwen3-1.7B continue each seed; use contrastive decoding against the same rollout with the corpus-mean state mu_l; chains see only the patched position and the fixed prefix; optionally re-add (h - mu) at each chain position. Add an Aletheia-style verifier: feed each chain back as plain text into unmodified Qwen3-1.7B and accept only if the block-l state reproduces the centred h beyond length-matched nulls (mean-state, foreign-state and shuffled chains); revise with the next contrastive candidate, cap attempts, count abstentions, and measure the verifier's error on teacher-forced known continuations.",
  "faithfulness_to_chain_lens": "Keeps the post's J-lens seeds, contrastive decoding against a mean-state prior, chain-restricted context and optional re-injection. It replaces the MTP head (which predicts t+2 from a final state and the t+1 embedding) with Qwen3-1.7B's own full forward pass and next-token head, so there are no new parameters and no foreign prior. The language prior is the model's own, plus a fixed prefix that is the same for real and null states and so cancels in the contrast. Forward-only, fully inverse-free. Every chain step passes through the same MLPs where H-neurons live.",
  "cost": "Training-free; the J-lens already exists. Qwen3-1.7B in fp32 is about 8.1 GB, which fits an A100-40GB. Each chain token costs about 4.1 GFLOP (2 x 2.03e9). Estimate (mine): 10 seeds x 5 steps x 2 (real and mean state) plus the verifier is about 0.7 TFLOP per read cell, about 0.07 s at an assumed 10 TFLOP/s in fp32 with TF32 off. That is about 12 min per 10,000 cells, and an extra J-lens variant refit is under 1 h in bf16. Engineering about 1-2 days on top of the round-1/2 hooks.",
  "risks": "Prior of the target prompt and position effects: Qwen has attention sinks at the first positions, which the J-lens code skips (first 16). A state patched out of its context may be off-distribution at blocks above l. Early blocks read poorly (Patchscopes: the first about 10 layers are close to the logit lens), which matters for block 6. The model's own language prior dominates long chains (the post's neutral-seed control already performs similarly), so contrastive decoding and the mean-state, foreign-seed, shuffle and token-matched controls are mandatory. Results are not comparable to the post's WorkspaceBench numbers. Upside: the post's own table gives Patchscopes the lowest hallucination (0.06)."
 },
 {
  "option": "(c) Sister model with a native MTP head: Qwen3.5-2B (primary; d_model 2048 like Qwen3-1.7B), Qwen3.5-0.8B as a size control, MiMo-7B-Base as an optional larger model. Run the published chain lens unchanged (J-lens or J++ from Neuronpedia, native MTP, contrastive decoding, chain-only attention, re-injection), and on the same states also run option (b) and, if trained, option (a), to calibrate the substitutes against the native head.",
  "faithfulness_to_chain_lens": "The only option that is the chain lens itself: an MTP head trained jointly with the trunk, as in the post (which names Qwen3.5+). It measures a different network: 18 linear-attention plus 6 full-attention layers, vocabulary 248,320, a multimodal checkpoint, different training data. It cannot stand in for Qwen3-1.7B. Its value is calibration (does (b) agree with the native head on identical states, including confabulation versus chain length?) and an independent H-CHAIN replication. Forward-only.",
  "cost": "Training-free. The MTP module (60.8M params for 2B) must be loaded manually from safetensors because transformers drops ^mtp.*, and implemented as fc([RMSNorm(emb); RMSNorm(h)]) (order as in vLLM's Qwen3.5 code), then one gated full-attention layer, a norm and the shared head. An MTP step is about 1.1 GFLOP including the 248k-vocab head. Qwen3.5-2B in fp32 is about 9 GB. Estimate (mine): under 1 A100-hour per 10,000 cells plus about 1-2 days of engineering. MiMo-7B: 15.7 GB in bf16 (tight in fp32), no Neuronpedia J-lens, so a J-lens fit is needed.",
  "risks": "Architecture and tokenizer mismatch with Qwen3-1.7B. The transformers version needed for qwen3_5 on Colab is unverified. vLLM runs the MTP head only as a speculative drafter, so chain-only attention and contrastive decoding need custom PyTorch, and a wrong concatenation order silently breaks the head. WorkspaceBench's adapted Qwen3.5-2B set has almost no computational items (chain_intermediates 2, arithmetic 44), so the post's headline families cannot be reproduced at this size. MiMo's remote MTP module signature may need adapting."
 },
 {
  "option": "(a) Post-hoc DeepSeek-V3-style MTP head trained on frozen Qwen3-1.7B. One Qwen3 decoder layer (16 query heads x 128, 8 KV heads, MLP 6144) plus fc over [RMSNorm(emb); RMSNorm(h)], with the tied embedding and LM head frozen: about 58.7M trainable params. Trained on self-distilled Qwen3-1.7B text to predict t+2 from the final state and the t+1 token. Secondary arm only.",
  "faithfulness_to_chain_lens": "Architecturally the closest copy of the chain lens's head, but its knowledge is learned after the fact from a chosen corpus, not co-trained with the trunk. It reads what a trained decoder can extract, not what Qwen3-1.7B was shaped to pre-plan, so it is closer to a trained lens than to the training-free chain lens. Applied to J-lens-transported mid-block states, it is off-distribution and falls back on its own prior (the post saw generic 'round number' output without contrastive decoding or re-injection). Cross-entropy regression is a fitted forward map, so it is allowed under the user's rule, but it is training.",
  "cost": "Estimate (mine): about 5.7 GFLOP per training token (frozen forward 4.1 + head forward and backward + LM-head forward and backprop to the input). 100M tokens is about 5.7e17 FLOP: about 1.6 A100-hours in bf16 at 100 TFLOP/s effective, about 16 h in fp32 with TF32 off; data generation is extra. Published anchors: Medusa-1 took 5 h on one A100 for Vicuna-7B with 60k ShareGPT samples; EAGLE took 1-2 days on an RTX 3090 node for 7B with 68k dialogues. It needs resumable checkpoints to fit the 5-hour session limit.",
  "risks": "Frozen backbones specialised for next-token prediction resist MTP heads (Mehra et al. 2025), so expect low t+2 accuracy. The head's own prior makes confabulation grow with chain length by construction, an extrinsic 'attractor' that confounds the length-versus-H-neuron question, and the chain is produced outside Qwen's MLPs. The training corpus leaves its style. It may conflict with round 1b's 'no training of Qwen' constraint (Qwen stays frozen, but a model is trained), so this must be decided explicitly. Report held-out t+2 accuracy against Qwen's own two-step predictions and run the post's controls."
 },
 {
  "option": "(a') Reuse the published EAGLE-3 draft head AngelSlim/Qwen3-1.7B_eagle3 as the chain head.",
  "faithfulness_to_chain_lens": "Low. EAGLE-3 fuses features from several layers instead of one final-layer state and predicts directly over a reduced 32,000-token draft vocabulary, so a J-lens-transported single vector does not match its input. Only usable on true final positions with all fused layers available, which is not what the chain lens reads.",
  "cost": "Zero training. One decoder layer (hidden 2048, MLP 6144); trivial inference cost.",
  "risks": "Input mismatch; truncation to the 32k draft vocabulary; a chat-data prior from speculative-decoding training (accept length about 2.1-2.2); custom license file. Not recommended except as a negative control showing what a foreign-trained head's prior produces."
 },
 {
  "option": "(b') Variant of (b) with a learned soft prompt (Future Lens style) instead of a fixed neutral prefix: a few prompt vectors trained so that the patched state's continuation matches Qwen3-1.7B's own future tokens.",
  "faithfulness_to_chain_lens": "Between (a) and (b): the decoding is still Qwen3-1.7B's own network, but a small learned context replaces the fixed prefix, and it carries its own trained prior. Future Lens found learned prompts beat fixed prompts and linear maps for tokens beyond the next (GPT-J-6B: more than 48% at some layers against a 20.1% bigram baseline).",
  "cost": "Estimate (mine): few parameters, but training still needs forward and backward passes through the frozen 1.7B model on teacher-forced data, likely well under one A100-hour in bf16 for a modest corpus.",
  "risks": "It is training, with the same constraint question as (a); its prior is learned and can leak into readouts; little gain if contrastive decoding against the mean state already removes the prefix prior. Use only if (b) with a fixed prefix fails the calibration against (c)."
 }
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
  "finding": "Pipeline: J-lens transport and seeds, then an MTP rollout with contrastive decoding against a corpus-mean state, chain-only attention and re-injection. WorkspaceBench macro 0.420 vs J-lens 0.319 (28 families) and 0.487 vs 0.411 composite (21). Hallucination 0.45 vs J-lens 0.32 (Patchscopes 0.06 lowest), and both hallucination and performance grow with chain length (19 families). Contrastive decoding adds +0.02-0.03; adding the MTP head is about twice the gain of J-lens transport. Neuron round trip: best chain above the 99.8th WikiText percentile and top-3 chains above 90% of random readouts; judge 57% on a 4-way test. Cost about 100x single-token lenses for 10 chains of 5 tokens, and 1 H100-hour for 1.1M neurons of Qwen3.6-27B. Used WorkspaceBench commit eb5a335. No code link in the post.",
  "how_checked": "Read the full saved text; metadata (authors, 11 Nov 2026) and outbound links extracted with Playwright."
 },
 {
  "title": "Verbalizable Representations Form a Global Workspace in Language Models",
  "url": "https://transformer-circuits.pub/2026/workspace/index.html",
  "authors": "Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, T. Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, Jack Lindsey (Anthropic)",
  "year": "2026",
  "finding": "Defines the J-lens as softmax(W_U norm(J_l h)), with J_l the expectation of Jacobians over source positions, target positions t' >= t and about 1,000 prompts of 128 tokens. Appendix variants: penultimate target, frozen-QK, self-only and future-only target positions, mean or median aggregation. Beats the logit and tuned lens with 10 prompts. Noisy readouts in the first third of depth. Template lens: t_w = (Sigma + lambda I)^-1 (mu_w - mu), about 12,700 words. Also published as arXiv 2607.15495 (16 Jul 2026).",
  "how_checked": "arXiv abs page via WebFetch; full HTML downloaded, converted to text and grepped (methods, design ablations, template-lens appendix)."
 },
 {
  "title": "anthropics/jacobian-lens (jlens/fitting.py and README)",
  "url": "https://github.com/anthropics/jacobian-lens",
  "authors": "Anthropic PBC",
  "year": "2026",
  "finding": "Estimator: a one-hot cotangent at every valid target position gives sum over p' >= p of dh_final[p']/dh_l[p], averaged over source positions 16..T-2, then a running mean over prompts. One forward and ceil(d_model/dim_batch) backward passes per prompt. No regression and no inversion. Apache-2.0; marked as a reference implementation that is not maintained.",
  "how_checked": "Raw fitting.py downloaded and read in full; README via WebFetch."
 },
 {
  "title": "neuronpedia/jacobian-lens (README, Qwen3-1.7B config.yaml, convergence CSV)",
  "url": "https://huggingface.co/neuronpedia/jacobian-lens",
  "authors": "Neuronpedia (fits); lenses by Gurnee et al., Ayonrinde and Lindsey, Blank et al.",
  "year": "2026",
  "finding": "A J-lens exists for Qwen3-1.7B: 466 WikiText prompts, final-block target, bf16, dim_batch 128, fitted on a B200 in about 716 s (2026-06-11). No R-lens or J++ for Qwen3-1.7B. J++ exists for Qwen3.5-0.8B, 2B and 4B, Qwen3-4B and 9B. Readout logits = unembed(h @ J[l].T).",
  "how_checked": "Live README downloaded 2026-11-13 and diffed against the 10 Nov local copy (identical); config.yaml and CSV read locally and fit time summed."
 },
 {
  "title": "R-lens: Making J-lens More Faithful on Early Layers",
  "url": "https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers",
  "authors": "Camila Blank, Agam Bhatia, Neel Nanda",
  "year": "2026",
  "finding": "The J-lens fitted with LRP rules in the backward pass (LN rule, identity rule on SiLU/GELU, half rule on the gated MLP). Stop-gradients only, so the same cost. Cleaner, more causal early-layer readouts; lenses released.",
  "how_checked": "Text and metadata (5 Aug 2026) fetched with Playwright; method section read."
 },
 {
  "title": "J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses",
  "url": "https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace",
  "authors": "Kola Ayonrinde, Jack Lindsey",
  "year": "2026",
  "finding": "Per-layer k-means router with 8 expert Jacobians, the R-lens backward rules, and signed expert weights fitted to a recall@10 surrogate; readout filtering. Recall@10 about 55% on Qwen3.5-9B and Qwen3.6-27B.",
  "how_checked": "LessWrong metadata via Playwright; the koayon/jpp-lenses HF model card (local copy) read in full."
 },
 {
  "title": "WorkspaceBench: Evaluating Interpretability Methods for the Global Workspace",
  "url": "https://www.lesswrong.com/posts/Zeg2JztbdhguL48uH/workspacebench-evaluating-interpretability-methods-for-the",
  "authors": "Camila Blank, Agam Bhatia, Euan Ong, Neel Nanda",
  "year": "2026",
  "finding": "3,356 questions in 27 families, built for Qwen3.6-27B. Gate: more than 80% correct in 10 samples without chain of thought. A summarizer turns token bags into concepts without the prompt; Gemini-3.8-Flash judge (Sonnet-5 for safety) at the time of the post. Readouts at layers 20-60 in steps of 4. Says it may need adapting for smaller models.",
  "how_checked": "Full text, metadata (23 Sep 2026) and links fetched with Playwright."
 },
 {
  "title": "camilablank/workspace-bench (README, AGENTS.md, adapted/README.md, adapted/qwen3.5-2b and qwen3.5-0.8b lenses/table.md, LICENSE)",
  "url": "https://github.com/camilablank/workspace-bench",
  "authors": "Camila Blank et al.",
  "year": "2026",
  "finding": "MIT. Judge claude-opus-5-5 since 2026-10-29; Sonnet-5 pinned for agentic misalignment and jailbreak recognition; regex for several families. Summary mode makes one digest per item. Cost given as calls per arm (wsbench list) and spend_usd, with no dollar figure. Porting needs a re-gate or an adapted set. Qwen3.5-2B set: chain_intermediates 2 items, arithmetic 44, J-lens macro 0.209 over 14 families; Qwen3.5-0.8B 0.126. No Qwen3-1.7B set.",
  "how_checked": "Raw files downloaded and read 2026-11-13; adapted/ folder listing via WebFetch."
 },
 {
  "title": "neuron-lens (README)",
  "url": "https://github.com/stdlogits/neuron-lens",
  "authors": "a9lim, treetowntree (with Claude)",
  "year": "2026",
  "finding": "R-lens and S-lens catalog of every neuron and head of Qwen3-8B and Gemma 4 12B. K-lens (Theia Vogel, unpublished): K_L = mean_t sum_{s<=t} dh_L(t)/de(s), scoring input tokens. S-lens = K-lens under RelP rules. Fit on 1,000 DCLM documents, 11-16 s per document per lens on an H100. R-lens read predicts steering with median r up to 0.9 or more from L22. Code MIT, lenses and catalog CC BY 4.0. The round-trip criterion is not defined here.",
  "how_checked": "Raw README and LICENSE downloaded and read."
 },
 {
  "title": "DeepSeek-V3 Technical Report (sec. 2.2 MTP, sec. 5.4.3)",
  "url": "https://arxiv.org/abs/2412.19437",
  "authors": "DeepSeek-AI",
  "year": "2024",
  "finding": "MTP module: h' = M_k[RMSNorm(h); RMSNorm(Emb(t_{i+k}))], one transformer block, shared embedding and head, lambda 0.3 then 0.1, D = 1, discardable. Second-token acceptance 85-90%, 1.8x tokens per second.",
  "how_checked": "arXiv HTML v2 via WebFetch (two chunks)."
 },
 {
  "title": "Better & Faster Large Language Models via Multi-token Prediction",
  "url": "https://arxiv.org/abs/2404.19737",
  "authors": "Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Roziere, David Lopez-Paz, Gabriel Synnaeve",
  "year": "2024",
  "finding": "n independent output heads on a shared trunk; more useful at larger sizes; up to 3x faster inference. The HF release facebook/multi-token-prediction is gated, with custom llama code and consolidated.pth (7B, 1-token and 4-token variants).",
  "how_checked": "arXiv abs via WebFetch; HF API metadata (siblings, gated) for the release."
 },
 {
  "title": "On multi-token prediction for efficient LLM inference",
  "url": "https://arxiv.org/abs/2502.09419",
  "authors": "Somesh Mehra, Javier Alonso Garcia, Lukas Mauch",
  "year": "2025",
  "finding": "Adding MTP heads to frozen LLMs is hard because hidden layers are strongly specialised for next-token prediction. Joint training helps but does not fully overcome this.",
  "how_checked": "arXiv abs via WebFetch."
 },
 {
  "title": "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads",
  "url": "https://arxiv.org/abs/2401.10774",
  "authors": "Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao",
  "year": "2024",
  "finding": "Medusa-1 heads are trained on a frozen backbone; 60k ShareGPT samples; about 5 h on one A100 for Vicuna-7B; self-distillation option.",
  "how_checked": "arXiv abs and HTML v3 via WebFetch."
 },
 {
  "title": "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty",
  "url": "https://arxiv.org/abs/2401.15077",
  "authors": "Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang",
  "year": "2024",
  "finding": "Draft head: one decoder layer over [second-to-top-layer features (before the LM head); embedding of the token sequence shifted by one], target frozen, embedding and LM head shared. 68k ShareGPT dialogues; 1-2 days on an RTX 3090 node (7B-33B); low sensitivity to training data; head 0.24B for 7B.",
  "how_checked": "arXiv abs and HTML v3 via WebFetch."
 },
 {
  "title": "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test",
  "url": "https://arxiv.org/abs/2503.01840",
  "authors": "Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang",
  "year": "2025",
  "finding": "Multi-layer feature fusion instead of top-layer features; direct token prediction; training-time test.",
  "how_checked": "arXiv abs via WebFetch."
 },
 {
  "title": "AngelSlim/Qwen3-1.7B_eagle3",
  "url": "https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3",
  "authors": "Tencent AngelSlim",
  "year": "2025",
  "finding": "A published EAGLE-3 draft head for Qwen3-1.7B: 1 layer, hidden 2048, MLP 6144, draft vocabulary 32,000; accept length about 2.1-2.2 on vLLM with 2 speculative tokens; custom license file.",
  "how_checked": "config.json and README downloaded and read; HF API metadata."
 },
 {
  "title": "Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models",
  "url": "https://arxiv.org/abs/2401.06102",
  "authors": "Asma Ghandeharioun, Avi Caciularu, Adam Pearce, Lucas Dixon, Mor Geva",
  "year": "2024",
  "finding": "Patch a source representation into a target prompt, layer and position and let the model decode. The token-identity prompt beats the logit and tuned lens from layer 10 upward; the first about 10 layers are hard.",
  "how_checked": "arXiv abs and HTML v4 via WebFetch."
 },
 {
  "title": "Future Lens: Anticipating Subsequent Tokens from a Single Hidden State",
  "url": "https://arxiv.org/abs/2311.04897",
  "authors": "Koyena Pal, Jiuding Sun, Andrew Yuan, Byron C. Wallace, David Bau",
  "year": "2023",
  "finding": "On GPT-J-6B, a learned-prompt causal intervention predicts tokens beyond the next from one hidden state with more than 48% accuracy at some layers, against a 20.1% bigram baseline.",
  "how_checked": "Read the local PDF text (scratchpad/hchain/dl_futurelens/fl.txt, downloaded earlier in round 1b)."
 },
 {
  "title": "Qwen3.5 small models (0.8B, 2B, 4B, 9B and Base): config.json, safetensors index and header, model card",
  "url": "https://huggingface.co/Qwen/Qwen3.5-2B",
  "authors": "Qwen team",
  "year": "2026",
  "finding": "All have mtp_num_hidden_layers 1 and 15 mtp.* tensors. The 2B MTP module is 60.8M params (fc 2048x4096; gated full-attention layer with 8x256 query heads and 2 KV heads; MLP 6144). Hybrid 18 linear-attention plus 6 full-attention layers, vocab 248,320. Card: vLLM speculative-config method qwen3_next_mtp. Apache-2.0. Qwen3.6 has only 27B and 35B-A3B.",
  "how_checked": "Downloaded configs and indexes, read the safetensors header by range request, HF API parameter counts and model listing, grepped the README."
 },
 {
  "title": "MiMo-7B-Base / RL (config, index, modeling_mimo.py, model card)",
  "url": "https://huggingface.co/XiaomiMiMo/MiMo-7B-Base",
  "authors": "Xiaomi MiMo",
  "year": "2025",
  "finding": "num_nextn_predict_layers 1; model.mtp_layers.0.* in the checkpoint; the remote code builds MiMoMTPLayers ([hidden; embedding] concatenation) but the forward never calls it. Card: MTP tuned in pretraining and SFT, frozen in RL, about 90% acceptance; vLLM fork and SGLang. 7.83B params, MIT.",
  "how_checked": "Files downloaded and read; HF API."
 },
 {
  "title": "transformers model code (modeling_qwen3_5.py, modeling_qwen3_next.py, modeling_glm4_moe.py, modeling_deepseek_v3.py)",
  "url": "https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py",
  "authors": "Hugging Face",
  "year": "2026",
  "finding": "_keys_to_ignore_on_load_unexpected drops MTP weights: ^mtp.* (Qwen3.5, Qwen3-Next), model.layers.46/92 (GLM-4.5), model.layers.61 (DeepSeek-V3).",
  "how_checked": "Raw main-branch sources downloaded and grepped."
 },
 {
  "title": "vLLM MTP drafters (qwen3_5_mtp.py, mimo_mtp.py, deepseek_mtp.py, gemma4_mtp.py, config/speculative.py)",
  "url": "https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py",
  "authors": "vLLM project",
  "year": "2026",
  "finding": "Qwen3.5 MTP forward: fc([norm_e(emb); norm_h(h)]), then one full-attention layer and a norm. A list of MTP model types including qwen3_5_mtp, mimo_mtp, glm_ocr_mtp, gemma4_mtp. Gemma 4 assistant: pre_projection over [emb; backbone hidden] and KV-shared, query-only attention reading the target's cache.",
  "how_checked": "Raw main-branch sources downloaded and read."
 },
 {
  "title": "Gemma 4 MTP drafters (google/gemma-4-E2B-it-assistant, -E4B-it-assistant)",
  "url": "https://huggingface.co/google/gemma-4-E2B-it-assistant",
  "authors": "Google DeepMind",
  "year": "2026",
  "finding": "Separate drafters of about 78M params (4 layers, hidden 256, backbone_hidden_size 1536 or 2560, centroid-masked logits) used through transformers generate(assistant_model=...) and vLLM gemma4_mtp. The base gemma-4-E2B and E4B configs have no MTP. Apache-2.0.",
  "how_checked": "config.json, README and HF API read; safetensors header by range request."
 },
 {
  "title": "GLM-OCR and GLM-4.5-Air / GLM-4.7-Flash (configs, indexes, headers)",
  "url": "https://huggingface.co/zai-org/GLM-OCR",
  "authors": "Z.ai",
  "year": "2025-2026",
  "finding": "GLM-OCR (1.33B) has num_nextn_predict_layers 1 with eh_proj/enorm/hnorm/shared_head weights present. GLM-4.5-Air (110.5B, layer 46) and GLM-4.7-Flash (31.2B) have MTP but are above 9B. GLM-4-9B, GLM-4.1V-9B and GLM-4.6V-Flash have none.",
  "how_checked": "configs, indexes and safetensors header downloaded; HF API parameter counts."
 }
]
```
