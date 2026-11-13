# int:machinery

## Verifier summary
I checked every cited file, line and JSON key against /home/user/DSSE. Nearly all repo facts are right: the rk_loop and round-2 code, the plan's rates (plan:1375-1425, 1742), r2_ana's timings and the G2, G47 and G48 records, the theta_verdict damages and alpha_b, the notebook cells, the runner's guard and sync, and NOTES_w8/w9. The external facts also hold:
- PyTorch v2.14.1 sources (fetched): the memory-efficient kernel accepts fp32 with GQA on CUDA (sdp_utils.cpp l.1149) and uses OpMultiplyAddFastF32 (gemm_kernel_utils.h l.130-143).
- transformers 5.22.2 SDPA and mask logic, generate's logits_to_keep and the cache methods match.
- Model card at the pinned revision, Qwen3 report Table 1 and the long-context stage, the H-Neurons protocol (including the prose vs formula label conflict), Future Lens and Hydragen all check out.
- Shard-2 and tokenizer sha256, corpus sizes and the VAR(1) check reproduce.
- Every estimate re-derives: KV, prefill, math buffer, mask, logits, lens and SGD costs, L1 path, channel storage, damage extrapolation 0.34-0.36 alpha_b, and the per-token divergence hazard (~0.8-1.6e-3).

Errors:
- (1) The proposed cache.crop(-L) after a stopped read is wrong. The Stop exception extends only layers 0..b, and Cache.crop cuts all layers, so it corrupts layers above b. Crop per layer.
- (2) 'Batch 32 does not fit beyond about 4k' is wrong. It does not fit at 4k either; the limit is about 3k.

Imprecisions:
- Decode bounds ignore DynamicCache's per-step torch.cat rewrite: about 450/120/60 tok/s, not 950/290/140.
- B=4 at 32k (~37 GB) may fit the 40 GiB card.
- Backward memory is about 4-6 GB per layer at 32k, not 3-4.
- The 'runner start 13:05:42' stamp is post-gate; the runner started about 13:05:22.
- The ~648 s non-step time is not shown to be I/O.
- Round-1 '4.4 min per 128 texts' is a plan estimate.
- The H-CHAIN forward/backward caveat is too strong: traces differ, only the determinant is symmetric.
- The empty think block is 4 tokens, not 4-5; context grows by ≤56 tokens per turn.
- The past lens needs 16 GB, not 17.
- read_windows has no NLL.
- 'Untrained' beyond 32,768 is unverifiable.
- The mount cell is not silent.
- HEAD is c8257c2, not 8ee5d18, though the code is identical.

Proposal problems:
- The pilot asks for B=4 at 32k and produces TriviaQA data before the freeze.
- Prefix reuse needs re-based push windows: rk's hook mask is chunk-local, so 'rk hooks unchanged' does not hold.
- Splitting the lens over sessions loses its float64 accumulators.
- The paper's C selection needs uncosted suppression runs.
- Generation-time capture is exact only at blocks ≤ b.
- The packing follows round 2's 20-min frame, against the 1b design rules (sessions up to 5 h, ~30 A100-h), and omits J-lens/J++/MicroLens++, the power iteration, erasure, the 0.6B arm, chain-lens rollouts and the neuron-lens round trip.
- No proposal gives an MDE.

## Problems found by the verifier
- Long-context pilot: (1) It asks for decode at B = 4 with separate caches, but at 32k the report itself allows only B <= 3 on 40 GB. B = 4 needs ~37-38 GB, so cap 32k at B = 3 or run B = 4 as an out-of-memory probe. (2) The TriviaQA yield on 500 questions is model-output data produced before the freeze, which the 1b design rules forbid. Keep the pilot to engineering quantities, or declare those questions excluded from every 1b analysis. (3) It omits measurements the report itself lists as unknown: Drive read/write throughput, host RAM, local disk, and decode with DynamicCache against a non-rewriting cache. (4) Cached-vs-recomputed parity requires the prefix-cache code first, including the crop and push-window fixes below.
- Centred null and paired sign test: (1) This works with existing data. R1 states keep the per-prompt pro/anti reads (ST['axis_<name>'] Xp, Xa) and the fold, so centred nulls can be rebuilt without new texts. (2) The stored R1 nulls are uncentred, so the round-1-identical arm should report both p values. (3) Centring changes p for every axis that reuses rk_loop.axes / axis_from_texts (sports, sea, tone, RGI), so the plan must say which verdicts switch. (4) The paired sign test needs prompt clustering (64 units at 2 seeds per prompt) and a pre-registered minimum detectable effect, which is not given. Quantitatively, an uncentred null axis carries on average a share |u_bar|^2 / (|u_bar|^2 + sigma_e^2) of its squared norm along u_bar.
- Damage-normalised loop driver: (1) run_loop and run_loop_variant fix one alpha, and the B-Q0 and cell gains divide by alpha_b * cbar, so alpha_n must be passed per round into the gains and logged. (2) RefKL measures damage on 8 short round-1 references at block b. That suits short-context loops but not long-context arms (damage depends on length) or neuron interventions. (3) In the dose series, 0.25 alpha gives ~1/11-1/16 of the target damage and ~4x less read-back signal, while text-sampling noise does not shrink. No MDE is given. (4) 'Exactly as round 1' cannot reproduce round-1 texts: the state guard forces r2-prefixed loop names, hence new keys, and block 14 of round 1 ran on an L4 (~1e-3 per-token divergence). It replicates the procedure, not the texts. (5) The norm-preserving rescale changes x by a factor of about 1 - alpha*cos(x, u) - O(alpha^2), which is small at alpha 0.05-0.15. The expected contrast with the plain push should be stated.
- K chain over n turns: (1) Per-turn K is defined only while u_read.d^(n) excludes 0. With K ~0.46 and u_read.d^(1) = 8.33 [7.44, 9.23], the signal falls to ~0.81 at turn 4 and ~0.37 at turn 5, so K_n along u_read becomes undefined from about turn 4-5. The cost of turns beyond that is justified only for the RSI-axis K (1.053, descriptive, denominator interval not reported). Give per-turn MDEs. (2) The history think-block convention must be fixed before the freeze: 4 off-template tokens per history turn, 36 by turn 10. Turn-1 round-trip failures (0-5 of 128 per cell in round 2) also compound over turns. (3) The push-in-every-turn arm confounds carry with direct push; state the contrast. (4) The in-context read by generation-time capture is exact only at blocks <= b.
- KV-prefix reuse and branching: (1) 'Then crop(-L)' is wrong after a stopped read. The Stop exception extends only layers 0..b, and Cache.crop(-L) would also remove L prefix tokens from layers b+1..27. Crop per layer (cache.layers[l].crop(-L) for l <= b) or run the chunk unstopped. (2) 'rk hooks unchanged' does not hold. rk's steer hook compares chunk-local positions with an absolute rowstart (rk_loop.py:105), and CbarHook treats every T > 1 chunk as a prefill (r2_rk.py:619), so both must be re-based for a cached prefix. (3) HF generate with past_key_values needs at least one uncached input token. (4) The +/- rows of a context-push pair come from two prefills, and no helper exists to concatenate their caches along the batch. (5) A chunk over a cached prefix (q_offset > 0, q_len != kv_len) always builds a mask and repeats K/V for that call. (6) DynamicLayer.update rewrites the whole cache by torch.cat at every decode step, so per-row-cache decode runs at about half the stated bounds. (7) A Hydragen-style fp32 attention on torch matmuls differs bitwise from the memory-efficient kernel, so CRN pairs must share one path.
- Long-context hygiene: the same whole-tensor .double().cpu() also sits in r2_rk.read_windows (l.422), r2_theta.read_full (l.416) and score_grads (l.469). leaf_grads (r2_rk.py:739) and rk.grads (rk_loop.py:352) compute full logits and need logits_to_keep. rk_loop.code_checks' GO2 read uses output_hidden_states (l.419), so keep it at short length. The patches belong in a new core, not in the frozen rk_loop.py.
- Generation-time state capture: the capture is exact (unsteered) only at blocks <= b, because a push on block b's output never reaches layers 0..b. 'Selected-block' capture at blocks > b records steered states. The read context equals the generation context, so instructed prompts differ from rk's plain-prompt read. Parity with a teacher-forced read (padding, incremental decode) needs a GO2-style tolerance check. The last token needs one extra forward.
- Neuron hook manager, CETT and L1 logistic: (1) The paper picks C by held-out accuracy plus TriviaQA performance with the selected neurons suppressed. Each grid value then needs a suppression run (~20 s per 1,000 greedy answers, +3-10 min), which is not costed. (2) Alias matching replaces GPT-4o span alignment and so changes the 'answer tokens' feature. (3) The label conflict (prose y = 1 for hallucinated answer tokens, formula y = 1 for faithful) must be resolved in the plan. (4) Damage-matching with RefKL.hooks uses 8 short RSI/scene texts at block 14. For neuron sets spread over 28 layers, also match damage on held-out TriviaQA answers. (5) The paper's causal tests are over-compliance benchmarks with GPT-4o judges, so a TriviaQA-only intervention measures a different outcome. Include the alpha = 1 baseline explicitly. (6) No MDE is given for detection or intervention effects.
- Streaming lenses: (1) Splitting 4M tokens over 2-3 sessions means carrying 8.5-16 GB of float64 accumulators through Drive, because local/ is not synced and the VM is lost. Alternatives: one ~25-40 min session (allowed by the 1b design rules), per-session fits compared afterwards, or fewer blocks or offsets. (2) The past lens is a forward-map regression and is allowed under the inverse-free rule, but the plan must say so and must not use it to reconstruct earlier fronts; that requires the generate-verify-revise loop. (3) Fix the index convention: decoding h_L(t+k) predicts token t+k+1. (4) The forward/backward claim must allow for coordinate-dependent whole-state asymmetry, since traces differ (42.6 vs 64.4). (5) The 4M-token corpus is not chosen, pinned or licence-checked, and position bins beyond its sequence length need the long-context arm.
- Runner and notebook hardening: the 10 min/GB copy-in rule rests on attributing r2_ana's ~648 s of non-step time to ~0.92 GB of copy-in. The repo cannot confirm that, since the time also covers gates, double hashing, manifest and per-step syncs. Label the rule a placeholder and measure Drive throughput in the pilot. The mount cell prints a note rather than failing silently, but the proposed abort is still right. The cap of 2 concurrent sessions per account is an assumption.
- Draft session packing: (1) The 20-min framing comes from round 2 and conflicts with the 1b design rules: sessions up to ~5 h, ~30 A100-h envelope (R ~12, D ~9, K ~9). (2) It covers only this report's items. It omits J-lens / J++ / MicroLens++ fitting (README section 2 gives 30-60 min A100 for J++ alone), lambda_RSI vs lambda_max power iteration, low-damage erasure arms, the Qwen3-0.6B comparison and chain-lens rollouts (Qwen3-1.7B has no MTP head). It also omits the neuron-lens round trip, which needs streaming 99.8th WikiText percentiles for 172,032 neurons, and the H-DUAL long-context statistics. So ~3-4 GPU-h is not the 1b total. (3) 'lens x 2-3' needs accumulator checkpoints (see the lens item). (4) No proposal states a minimum detectable effect, which the design rules require.

## Per-reference / per-fact verdicts
- [correct] Round-1 loop sampling: temperature 0.7, at most 40 new tokens per text (rk_loop.py:34).: TEMP, NNEW = 0.7, 40 at l.34.
- [correct] Push hook adds s*alpha*min(|h_t|, cap)*unit(d) at block b from rowstart = pad + START; leaf mode via STEER['V']; Stop exception after block b (rk_loop.py:100-116).: Hook and Stop at l.100-116; rowstart = pad + START is set in rk.generate at l.134, not inside 100-116. The window is computed in chunk-local positions (l.105) and every decode step (T = 1) is pushed, which matters for prefix-cache reuse.
- [correct] cap = 2 x median norm of unsteered block-b states (rk_loop.py:183).: Median over non-template positions from START on, over the 64 unsteered reference prompt+continuation sequences (l.163-183).
- [correct] Sign-flip null axes are built from uncentred per-prompt differences, E @ diff / P (rk_loop.py:223; r2_rk.py:471-473).: Re-derived: with diff_i = u_bar + e_i, null_k = s_k u_bar + (1/P) sum eps_ki e_i, Var(s_k) = 1/P, so on average a share |u_bar|^2 / (|u_bar|^2 + sigma_e^2) of a null axis's squared norm lies along u_bar; centring removes it.
- [correct] rk.damage forms full-vocabulary logits at every position before indexing continuation positions (rk_loop.py:193-194).: 2.49 / 9.96 / 19.91 GB fp32 at 4k / 16k / 32k re-derived (T x 151,936 x 4 B).
- [imprecise] rk.read converts the whole [B,T,D] block output to float64 on the CPU (rk_loop.py:154).: CAP.pop('h').double().cpu(): the float64 copy is made on the device first, then copied to the CPU (17.2 GB float64 at B = 32, 32k). The same pattern is at r2_rk.py:422 (read_windows), r2_theta.py:416 (read_full) and r2_theta.py:469 (score_grads).
- [correct] run_loop uses one alpha for all rounds (rk_loop.py:312).: alpha = ST['alpha']['alpha'] * ASCALE; run_loop_variant (r2_unlearn_qwen.py:990-1001) also fixes alpha.
- [correct] Models are loaded in float32 with TF32 off on CUDA (ufo_common.py:15-18; r2_theta.py:992).: r2_rgi_geom.py:195-197 additionally sets float32 matmul precision 'highest' and cudnn.benchmark False.
- [correct] rk_loop registers its hooks on block BLOCK at import, so one process serves one block; open_block enforces this (rk_loop.py:113; r2_rk.py:236-286).: r2_rk.py:256-257 raises on a second block. The report's wording that this blocks neuron hooks is too strong: extra hooks can be registered on rk.model (WriteOrth does, 57 hooks).
- [correct] read_all_blocks: one unstopped forward, float64 means via x.double() per block, per-token states to CPU in fp16 (r2_rk.py:366-396, 381-382).: 134 MB per block per 32k row in fp16 and 3.76 GB for 28 blocks re-derived; selftest tolerance 5e-3 at r2_rk.py:1023.
- [correct] WriteOrth removes span(U) from embed_tokens, o_proj and down_proj outputs of all blocks, 57 hooks (r2_rk.py:695-712).: 1 + 28 + 28 = 57.
- [correct] leaf_grads and score_grads are checked against central finite differences (r2_rk.py:1139-1153; r2_theta.py:903-915).: The report extends this to all four gradient functions: a3_grads has its own 5-point check (r2_rgi_geom.py:676-692); rk.grads has none.
- [correct] read_full applies lm_head only at continuation positions (r2_theta.py:424).: The report's grouping 'read_windows and read_full: windows plus NLL in one pass' is imprecise: NLL is computed by read_full only; read_windows uses the early-stopped forward.
- [correct] B-Q8 turn 2: ids = PIDS[p] + turn-1 ids + TURN2_SUFFIX ids, no push, keys keyseed('theta2', b, a, T, p, s), read as reply to the plain prompt; K = u_read.d2 / u_read.d1.: r2_theta.py:695-739 (714, 716-717, 732); r2_stimuli.py:208, 214-217.
- [correct] The official chat template drops the empty think block of history turns, which the concatenated turn-2 ids keep (r2_stimuli.py:225-232).: The block is exactly 4 tokens ([151667, 271, 151668, 271] with the pinned tokenizer), not 4-5. TURN2_SUFFIX is 16 tokens.
- [correct] R6: K defined only if the interval of u_read.d1 excludes 0; self-sustaining if K's lower bound > 1; transfer test log K - log G (r2_theta_verdict.py:236-255, 565-580).: lambda_lin at 257-267 also as cited.
- [correct] calibrate (f <- f*sqrt(target/damage)), RefKL (cached unmodified log-probs; push/hooks damage), ConstPush (second push with the pre-push norm) (r2_unlearn_qwen.py:223-233, 408-439, 442-466).: RefKL stores ~48.6 MB float64 per 40-token text on the device; 8 forwards per evaluation.
- [correct] run_loop_variant: loop with optional push orthogonalisation and per-round damage via RefKL (r2_unlearn_qwen.py:969-1001).: Damage is measured per round, not calibrated; alpha is fixed.
- [correct] Float64 Gram/mean accumulation over all 28 blocks on the GPU with an outlier filter; Taps on all blocks; equal-length batching (r2_rgi_geom.py:466-503, 225-240, 252-258).: Filter: norm < 10 x median; [28, 2048, 2048] float64 = 0.94 GB.
- [correct] Neuron hooks outside round 2: down_proj pre-hook capturing a = silu(gate)*up, W_down column norms, per-row pre-hook interventions (neu_extract.py:16-21, 39; neu_sparsity.py:33-50).: neu_extract captures only the last prompt token. Their docstrings cite an essay on a forbidden domain; the report correctly reuses code only.
- [correct] Planned host costs: 10 us per op, 20 us per sync; decode step at batch 32 ~1,760 ops (~18 ms host); R-fwd 20 ms (23 ms with WriteOrth); forward+backward 60 ms; model load 30 s (plan:1376-1380).: 
- [correct] Planned compute: 2.8 GFLOP/token + 0.6 for full-vocab logits; fp32 peak 19.5 TFLOP/s; R-full 0.45 s, R-all 0.40 s, R-read 0.25 s, extra-block reads 0.35 s, short sentences 0.15 s (plan:1381-1387).: 2.819 / 0.622 GFLOP per token re-derived from the config; 6.3 TFLOP / 0.40 s = 15.8 TFLOP/s is the plan's implied rate, not a measurement.
- [correct] Planned generation: 32 prompts x 40 new tokens = 1 s, 1.15 s with WriteOrth, 1.3 s at ~80-token prompts, 2 s at ~200-token prompts (plan:1388-1390).: Derived 1,280 / 985 / 640 new tokens/s re-derived.
- [correct] Planned setup 4 min for Qwen sessions, 3 min for toy/analysis; margin 1.5; <= 18 min with margin within a 20-min target (plan:1400-1403).: 
- [correct] Round-2 rates are 'given facts'; costing from round-1 per-op times was not adopted because round-1 timing files could not be read before the freeze (plan:1742).: 
- [imprecise] Round 1 on CPU: 4.4 min per 128 texts; RSI loops 4 x 8.8 min each (abschnitt11_rsi_kennwerte_plan.md:93, 98).: These are the round-1 plan's cost estimates, not measurements; the plan was amended to run on A100 GPUs (l.129-130).
- [correct] r2_ana planned copy-in of ~320 MB R1 states and ~600 MB round-2 outputs in 60 s (plan:1425).: Planned step times C-Q1 90 s, theta 110 s, RGI 127 s, C verdicts 30 s, report 20 s sum to 377 s.
- [correct] r2_ana step times on A100-SXM4-40GB: 18.9, 132.5, 171.5, 3.5, 2.5 s; seconds_steps 328.9.: Total below plan, but theta_verdict (132.5 vs 110 s) and rgi_verdict (171.5 vs 127 s) were slower than planned. Sub-step timings also exist: theta_verdict.json seconds 130.1 (power re-run 75.4/25.9/25.9 s), rgi_verdict.json seconds 170.8.
- [imprecise] r2_ana successful attempt: runner seconds 976.7, elapsed_since_t0 1042.0, t0 13:04:17, runner start 13:05:42; ~648 s runner time outside steps.: 13:05:42 is the attempt's 'started' stamp, written after gates G1/G2 and the start conditions (r2_run_session.py:1405); 'seconds' counts from t_attempt at l.1365, so the runner started at about 13:05:22 (ended 13:21:39 minus 976.7 s), ~65 s after t0. 976.7 - 328.9 = 647.8 s is right as non-step runner time; its split is unknown.
- [correct] r2_ana attempts 1 and 2 only waited for r2_pc/session_done.json.: 
- [correct] C-Q1 took about 870 s on one CPU thread vs the plan's 90 s on the A100; measured A100 step 18.9 s.: NOTES_w8.md:266-267; the 870 s was on synthetic data at real size.
- [correct] No round-2 session was stopped by the time guard; no continuation sessions (G48; packing.*.time_guard_stops).: A 'time guard' result needs pending steps (r2_run_session.py:1288), so a last step ending past 20 min would not count, as the report says.
- [correct] Round-2 GPUs: A100-SXM4-40GB for 8 sessions, A100-SXM4-80GB for r2_t4; CUDA 13.0 (G2.sessions).: 
- [correct] Round-2 versions torch 2.14.1+cu130, transformers 5.22.2; round-1 gpu1 ran on an NVIDIA L4 (G47).: r1_gpu1 used the same torch/transformers versions.
- [correct] L4 vs A100 regeneration with the same keys: RSI_PRO identical 0.96875 (prefix 0.98164), RSI_ANTI 0.9375 (prefix 0.98047).: n = 64 each. Per-token hazard re-derived: -ln(0.96875)/40 = 7.9e-4, -ln(0.9375)/40 = 1.6e-3, so 'about 1e-3' holds.
- [correct] Block 14 alpha_b = 0.15; block 18 alpha_b = 0.1085 (theta_verdict.json).: 
- [correct] At fixed alpha_b, RSI_a push damage at block 14 rose over rounds 0-3: 0.1135, 0.2018, 0.2871, 0.3235 nats.: 
- [correct] Block-14 damage curve on u_push: 0.0971, 0.3243, 0.9133, 2.3742 nats at 0.5, 1, 2, 4 alpha_b.: Re-derived: 0.05 nats at 0.341 alpha_b (exponent 1.74) to 0.359 alpha_b (exponent 2). This is an extrapolation below the lowest grid point, not an interpolation, and holds for u_push only; the slope falls at larger pushes (1.49 for 1-2, 1.38 for 2-4). For the RSI start direction (0.1135 nats at alpha_b) the same rule gives 0.62-0.66 alpha_b.
- [correct] Qwen3-1.7B config values (hidden 2048, intermediate 6144, 28 layers, 16/8 heads, head_dim 128, max_position_embeddings 40960, rope_theta 1e6, rope_scaling null, tied, vocab 151936, bfloat16); shard-2 and tokenizer sha256 match checksums.json:6-7.: sha256 recomputed: shard 2 912becff... and tokenizer aeb13307... match. Shard 2 holds only lm_head.weight [151936, 2048] (622,329,856 B).
- [correct] Model card: Context Length 32,768; 1.4B non-embedding; 28 layers; GQA 16 Q / 8 KV; no YaRN or long-text section.: Checked at the pinned revision 70d244cc. The card also recommends output lengths up to 38,912 tokens for hard problems.
- [correct] Qwen3 report: long-context stage at 32,768 tokens, RoPE base 10,000 -> 1,000,000 via ABF; YaRN + DCA for 4x at inference; Table 1 lists Qwen3-1.7B at 32K.: Qwen Team, arXiv:2505.09388 (2025), section 3.2 and Table 1. The report's further claim that positions 32,768-40,960 are 'untrained for this size' is an inference the sources do not state.
- [correct] Parameters 1,720,574,976 (tied): 6.88 GB fp32, 3.44 GB bf16; KV 57,344 values/token = 229,376 B fp32 (224 KiB), 114,688 B bf16; 7.52 GB fp32 at 32k.: Re-derived; matches shard 1 size 3,441,185,608 B (tensors 3,441,149,952 B + header).
- [correct] PyTorch v2.14.1: memory-efficient attention accepts float32 on sm80+ and supports GQA in the dense path; flash needs fp16/bf16 on sm80.: Fetched v2.14.1 sdp_utils.cpp: kFloat in the sm80 dtype list (l.1112-1113, 1176-1182); supports_gqa = true on CUDA (l.1149); check_dtypes_low_precision for flash and cuDNN (l.894-903, 998).
- [correct] PyTorch v2.14.1 memory-efficient attention uses tensor cores with OpMultiplyAddFastF32 for float32 on compute capability >= 80.: gemm_kernel_utils.h l.130-143. No TF32 gating found in sdp_utils.cpp or attention.cu.
- [correct] transformers 5.22.2 uses native GQA in SDPA only without a mask (else repeat_kv), and is_causal only when mask is None and q_len > 1 (sdpa_attention.py:30-37, 113-118, 140).: SDPA then converts a bool mask into a float additive bias of query dtype (attention.cpp:579-592, v2.14.1).
- [imprecise] transformers 5.22.2 skips the SDPA mask only without padding and when q_len == 1 or kv_len == q_len (masking_utils.py:231-270).: l.276-279 also skip it at an empty cache (q_offset = 0) when there is no padding mask or only trailing static-cache padding; left padding never skips. A continuation chunk over a cached prefix (q_offset > 0, q_len != kv_len) always builds a mask and repeats K/V.
- [correct] generate sets logits_to_keep = 1 by default; Qwen3ForCausalLM.forward accepts logits_to_keep as int or index tensor.: generation/utils.py:2943-2947; modeling_qwen3.py:449, 480; modeling_utils.py:5008-5034 (a 1-D index tensor works).
- [correct] DynamicLayer offers crop (negative count only), batch_repeat_interleave and batch_select_indices (cache_utils.py:163-192).: Cache-level wrappers at l.1789-1806 loop over all layers. DynamicLayer.update grows by torch.cat (l.135-136), rewriting each layer's K/V every decode step.
- [correct] Notebook Drive mount swallows any exception and continues with RUNS = DRIVE = None; final cell unmounts Drive (build_runde2_gpu_notebooks.py:209-216, 403).: It prints a one-line note (l.215), so 'silently' in the report text is imprecise; the run is not stopped and the runner gets no Drive argument (l.373).
- [correct] Two accounts: A (hub) runs r2_q1, r2_q2, r2_q3, r2_pc, r2_ana; B runs r2_t1-r2_t4; B's dsse_runs is linked as dsse_runs_B and copied by a cell.: The report text's 'copies a folder only when it is missing and finished' is imprecise: 'finished' (or declared not run) applies to r2_* folders only (l.244-249).
- [correct] Time guard 1,200 s from the notebook's first cell, checked only after a step that ran; per-step timeout 3,600 s; local/, r1/, r2in/ never synced.: r2_run_session.py:263, 959-971, 1213-1216, 967, 265.
- [correct] H-Neurons protocol: TriviaQA, 10 samples (T 1.0, top-k 50, top-p 0.9), 1,000 + 1,000 consistent examples, CETT, L1 logistic regression, 0.01-0.35 permille, alpha in [0, 3]; Mistral, Gemma, Llama only.: Gao, Chen, Xiao, Chen, Liu, Sun, arXiv:2512.01797 (v2 HTML read). Also confirmed: prose labels hallucinated answer tokens y = 1 but the rendered formula labels faithful y = 1. Not in the report: C is chosen by held-out accuracy plus TriviaQA performance under suppression; spans are aligned with GPT-4o; causal tests use over-compliance benchmarks.
- [correct] Repo generic text: corpus_wikitext 40 passages (~8.8k tokens), mmlu_topics ~0.49M, gsm8k ~0.19M, dn_text ~56k (chars/4).: Re-run: 8,784 / 489,864 / 187,104 / 55,620 (chars/4 including JSON overhead); axes_pairs adds ~93k.
- [correct] Stationary VAR(1): forward and backward residual determinants equal (840.479); per-channel R2 and traces differ (42.6 vs 64.4).: Re-run reproduces all numbers; the Sylvester argument holds. But the differing traces show that coordinate-dependent whole-state summaries are asymmetric too, so the report's 'only per-channel scores, token or nonlinear targets, or position dependence' is too strong.
- [correct] Round-2 readout: G 0.12-0.30, K 0.45-0.48 at block 14 along the style direction; K on the RSI axis at T 1.0 = 1.053 [0.743, 1.538]; lambda_lin = 0.7298 [0.1139, 1.356].: AUSWERTUNG.md l.79, 86, 89, 146; TV blocks.14.R6.cells.a1_T1.K_u_T_perp; lambda_lin along u*_b (TV blocks.14.B_Q7).
- [correct] Big round-2 outputs: r2_axes.pt ~300 MB, tok14.pt ~65 MB; state.pt copies change with most Qwen steps and are re-synced (NOTES_w9.md:99-104).: Lines 101-104.
- [correct] Drive FUSE, shortcut traversal and directory rename on Drive were untested when the runner was built (NOTES_w9.md:316).: Lines 315-317.
- [imprecise] Report text: commit 8ee5d18 was reviewed.: HEAD is c8257c2; the two later commits only extend results/ju_space/plans/runde1b_eingaben/README.md, so the code is identical.
- [wrong] Report text: batch 32 does not fit beyond about 4k.: At 4k it does not fit either: 32 x 0.94 GB KV = 30 GB plus 6.9 GB weights leaves too little for prefill transients (~0.47 GB per row at 4k) and the mask. Limit is about 3k tokens per row.
- [wrong] Report text: after a stopped chunk read over a carried cache, call cache.crop(-L).: The Stop exception leaves only layers 0..b extended. Cache.crop(-L) loops over all layers (cache_utils.py:1789-1797) and also cuts L prefix tokens from layers b+1..27. Crop only layers <= b, or run the chunk unstopped.
- [imprecise] Report text: B = 2 needs ~26 GB and B = 3 ~34 GB at 32k; branching B <= 3 on 40 GB.: True for cache copies of one B = 1 prefill (with the prefill transient kept). After the transient is freed: ~22 / 29 / 37 GB at B = 2 / 3 / 4, so B = 4 may just fit the 40 GiB (42.9 GB) card. A batched B-row prefill needs ~29 GB at B = 2 and ~41 GB at B = 3.
- [imprecise] Report text: aggregate decode with own caches <= ~950 / 290 / 140 tok/s at 4k / 16k / 32k.: The bound omits DynamicLayer's per-step torch.cat rewrite (~2 x KV of extra HBM traffic). With HF's default cache: ~450 / 120 / 60 tok/s. B = 1 decode at 32k stays ~20 ms/step.
- [imprecise] Report text: backward without checkpointing stores ~3-4 GB per layer at 32k, ~100 GB in total; ~12-14 GB at 4k.: With frozen weights my count is ~4-6 GB per layer at 32k (~115-165 GB for 28 layers). Only layers above the leaf block are recorded (~54-77 GB above block 14). leaf_grads and rk.grads also build a 19.9 GB full-logit tensor. At 4k ~14-21 GB. Same conclusion.
- [imprecise] Report text: the context grows ~60 tokens per turn, ~630 tokens at 10 turns; 4-5 off-template tokens per turn.: At most 56 tokens per turn (<= 40 answer + 16 suffix) from a ~21-token prompt: <= ~580 tokens after 10 turns, ~5.6k after 100. The empty think block is exactly 4 tokens.
- [imprecise] Report text: rk.stats = stats_on_axis returning c, p, Q, rel, A, Ue, sigma.: stats_on_axis returns c, p, Q, rel, r, A, norm; Ue and sigma are only in rk.stats on u_T_perp.
- [unverifiable] Report text: positions 32,768-40,960 are untrained for this size.: The report says the long-context pretraining stage used 32,768-token sequences; post-training lengths for 1.7B are not stated, and the card recommends outputs up to 38,912 tokens.
- [imprecise] Report text: r2_ana lost about 648 s to I/O; budget >= 10 min copy-in per GB.: The 648 s also includes gates (child-process GPU check, git checks), double hashing, manifest and per-step syncs; the repo does not split it. The 10 min/GB rule is a conservative placeholder.
- [imprecise] Report text: past lens doubles the accumulators to 17 GB.: 1 Gram + 16 cross matrices per block, 28 x 17 x 2048^2 x 8 B = 16.0 GB.
- [correct] Report text: prefill 32k B = 1 peak ~18 GB, 12-21 s; math fallback 68.7 GB per layer; padding mask +5.9 GB; all-position logits +19.9 GB; output_hidden_states +7.8 GB.: Re-derived with costs.py formulas (attention FLOP 114,688 T^2 causal; linear 2.819 GFLOP/token at 15.8 TFLOP/s; attention at an assumed 8-20 TFLOP/s). The math path needs a second [16, T, T] buffer for the softmax.
- [correct] Report text: future lens on 4M tokens: forward 11.9 min, accumulation 8.46 PFLOP ~9 min, held-out eval 15 min fp32 / ~1.5 min bf16, accumulators 8.46 GB, probes 1.88 GB fp16; SGD probes 29.4 h fp32, 3.1 h bf16.: All re-derived (SGD = one epoch, 224 probes, 3 x 0.622 GFLOP per token-probe).
- [correct] Report text: H-neuron L1 path ~9-12 min (X 4,000 x 172,032 fp32 = 2.75 GB, ~4 ms per FISTA iteration); 1.7-60 H-neurons; p/n 43-86.: Re-derived (~10 min: 75,000 + 25,000 + 50,000 iterations x 4 ms). Missing: the paper's C selection needs a suppression run per grid value (+3-10 min).
- [correct] Report text: channel series: 67 MB lagged covariances; 64 x 16k ~5-7 min, 64 x 32k ~13-23 min; 2 KB per token stored.: Re-derived.
- [imprecise] Report text: shared-prefix decode ~1,800 / 1,500 / 1,000 tok/s at B = 32.: Reproduces costs.py, but it assumes a custom cache and the plan's 15.8 TFLOP/s for skinny fp32 prefix products with TF32 off; treat as an optimistic upper estimate.
- [correct] Report text: Hydragen (Juravsky et al., 2024, arXiv 2402.05099); Future Lens (Pal, Sun, Yuan, Wallace, Bau, CoNLL 2023, arXiv 2311.04897).: Both abstract pages checked.
- [imprecise] Report text: the mount cell disables resume and sync silently.: It prints 'Kein Google Drive, Ergebnisse nur lokal' and continues; the problem is that it does not stop.

## Corrected report
# Round 1b: reusable machinery, cost model, long context, probe feasibility

**Scope and method.** I reviewed `/home/user/DSSE` read-only at HEAD c8257c2. Its code is identical to commit 8ee5d18: the two later commits only extend `results/ju_space/plans/runde1b_eingaben/README.md`. No model was loaded or run. An independent checker verified the facts and worked its corrections into this text.
- **Architecture numbers** come from three sources:
  - the `config.json` of the pinned Qwen3-1.7B snapshot (revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e);
  - the Hugging Face model card at that revision;
  - the Qwen3 Technical Report.

  The config is a scratchpad copy: `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/models/qwen3-1.7b/config.json`. Its weight shard 2, which holds only `lm_head.weight`, and the tokenizer have sha256 values that match `analysis/ju_space/gpu/data/checksums.json:6-7`.
- **Kernel facts** come from:
  - the PyTorch v2.14.1 sources: `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`, `aten/src/ATen/native/transformers/attention.cpp` and `aten/src/ATen/native/transformers/cuda/mem_eff_attention/gemm_kernel_utils.h`. This is the version Colab ran (`round2_summary.json` gates.G47.session.torch = 2.14.1+cu130);
  - the installed transformers 5.22.2, the version the notebooks pin (`notebooks/build_runde2_gpu_notebooks.py:273`).
- **Arithmetic** was done in two scripts that read only JSON:
  - `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/agent_cost/costs.py`
  - `/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad/r1b/agent_cost/fb_check.py`
- **Labels:** "est." marks every number not read from a file.

## 0 Short answers

1. **Reuse.** Almost everything round 1b needs exists:
   - push and stop hooks, keyed Gumbel CRN, read windows, all-block reads, axes and nulls;
   - prompt bootstrap, damage calibration with a cached reference, a constant push, write-orthogonalisation;
   - leaf gradients checked against finite differences, the B-Q8 turn-2 carry K;
   - closed-form probes, and the runner with gates, Drive sync and time guard.

   Missing pieces:
   - a centred null and a paired sign test;
   - per-round damage calibration inside the loop driver, and a norm-preserving control hook;
   - KV-prefix reuse and branching. Push windows must be re-based to the uncached chunk, because rk's steer hook computes its window in chunk-local positions. After a stopped read, only the extended layers may be cropped;
   - `logits_to_keep` hygiene for long sequences;
   - capture of per-token states during generation;
   - an n-turn K chain;
   - a neuron hook manager, the CETT measure and a GPU L1-logistic solver;
   - streaming lens accumulators.

   rk_loop's import-time side effects (argv parsing, model load, push and stop hooks on one block) limit rk's own push and read machinery to one block per process. Extra hooks (WriteOrth-style, neuron pre-hooks) can still be registered on `rk.model`. Multi-block pushes and flexible batching need a function-style core.
2. **Cost.**
   - All Qwen rates of round 2 are planned "given facts" (plan:1375-1407). They were deliberately not calibrated on round-1 timing files (plan:1742).
   - The only A100 timings in the repo come from the r2_ana analysis session. Its steps took 328.9 s in all, below the planned 377 s. theta_verdict and rgi_verdict took longer than planned; the other three steps took less.
   - r2_ana also spent about 648 s of runner time outside its steps, against a 60-s copy-in budget. That time covers gates, copy-in, hashing, manifest and per-step syncs; the split is only in logs on Drive.
   - Planned rates: generation 1,280 new tokens/s (batch 32 × 40 tokens per 1 s); about 5,000 tokens/s in batched full passes; about 10-11 s per 128-pair loop round.
3. **Long context.**
   - Model: native context 32,768 tokens; config `max_position_embeddings` 40,960; rope_theta 1,000,000 with no rope scaling; 8 KV heads × 128 dims × 28 layers.
   - KV cache: 224 KiB per token in fp32 (7.52 GB at 32k), half that in bf16.
   - Kernel: in float32 on A100, SDPA runs PyTorch's memory-efficient kernel.
     - At v2.14.1 that kernel accepts fp32 and GQA.
     - It computes with 3xTF32 "FastF32" tensor-core math, which the TF32 flag does not govern.
     - Flash and cuDNN are fp16/bf16 only.
     - The math fallback would need 68.7 GB per layer at 32k.
   - Prefill at 32k, B = 1: peak ≈ 18 GB (est.), 12-21 s (est.).
   - The round-1/2 path scales only with these changes:
     - B = 1 or equal-length rows (no padding masks);
     - `logits_to_keep`, or slicing before lm_head;
     - slicing on the GPU before `.double().cpu()`;
     - KV-prefix reuse, with re-based push windows and per-layer cropping after a stopped read;
     - branching by cache copies or by a shared-prefix attention:
       - cache copies: B ≤ 3 at 32k on 40 GB; B = 4 needs ≈ 37-38 GB and may just fit the 40 GiB card;
       - shared-prefix attention: est. at most about 1,000 tok/s at B = 32.
   - HF's default DynamicCache rewrites each layer's whole K/V at every decode step (torch.cat). Decode with one cache per row therefore reaches only about half the memory-bound estimates.
4. **Feasibility (A100-40GB, fp32 model).**
   - (a) H-neuron replication: est. 0.5-1.1 GPU-h, 1.4-2.8 GB local, < 0.1 GB on Drive. This includes the paper's choice of C, which needs a suppression run per grid point.
   - (b) Closed-form future lens on 4 M tokens (28 blocks × 8 offsets): est. 0.4-0.7 GPU-h and 1.9 GB of fp16 probes.
     - It needs 8.5 GB of float64 accumulators on the GPU.
     - They must stay in one session or be checkpointed to Drive.
     - SGD probes with a full-vocab loss would instead take 29 h in fp32.
   - (c) Channel time series: est. 0.1-0.5 GPU-h, MB to 4 GB.
   - Caveat for (c): under stationary second-order statistics, the whole state's forward and backward one-lag residual covariances have equal determinants (Sylvester's identity). The Gaussian mutual information between consecutive fronts is therefore symmetric.
     - Asymmetry can show only in coordinate-dependent summaries (per-channel R², total MSE or trace);
     - or in token or nonlinear targets;
     - or through non-stationarity (position dependence).
     - Checked numerically.
5. **Colab.**
   - The repo records two accounts (hub A), a strict A100 gate, zero time-guard stops, and r2_ana needing three launches. Round-1 gpu1 ran on an L4.
   - "Too many sessions" and Drive mount errors on re-run are not documented anywhere in the repo.
   - The notebook mount cell catches any mount exception, prints a one-line note and continues local-only. A failed re-mount therefore disables resume and sync without stopping the run.
   - Recommendations:
     - steps ≤ 5 min;
     - session length as the round-1b design rules allow (up to about 5 h with a time guard; round 2's 20-min target was a plan choice). If a 20-min target is kept, plan ≤ 10-12 min of work per session;
     - big arrays in `local/`;
     - input-free sessions in account B;
     - a 10-min long-context pilot (engineering numbers only) before freezing the 1b cost table.

## 1 Inventory of reusable machinery

Conventions:
- Every Qwen process loads float32 with TF32 off on CUDA (`ufo_common.py:15-18`; `r2_theta.py:992`). `r2_rgi_geom.py:195-197` also sets `set_float32_matmul_precision("highest")`.
- rk_loop parses sys.argv, loads the model and registers its hooks on block BLOCK at import (`rk_loop.py:33-57`, `113`). Hence "one process = one block" (`r2_rk.py:236-286`).

In each entry below, the letters are:
- (a) damage-normalised push;
- (b) 4k-32k contexts;
- (c) many follow-up answers;
- (d) per-token states at every block;
- (e) MLP neurons (28 × 6144).

"–" means no change needed.

### 1.1 Model, hooks, sampling

**`ufo_common.load_model`** (`ufo_common.py:11-19`)
- (b) Three changes make a silent math fallback raise instead of allocating 68.7 GB:
  - pass `attn_implementation="sdpa"` explicitly;
  - set float32 matmul precision "highest" as in `r2_rgi_geom.py:196-197`;
  - run prefill under `torch.nn.attention.sdpa_kernel([EFFICIENT_ATTENTION])`.
- (a), (c), (d), (e): –.

**rk_loop module plus `r2_rk.open_block`** (`r2_rk.py:236-286`, state guard 290-347)
- Continuity with round-1 states and α_b (read from loops.json `_meta`, l.274).
- (b), (d), (e): the import side effects and the single hooked block limit rk's own push and stop machinery to one block per process.
  - Additional hooks (neuron pre-hooks, WriteOrth-style) can be registered on `rk.model`.
  - Wrap the needed functions in a function-style core for multi-block pushes and flexible batching.
  - Keep open_block only for arms that must continue R1 states.

**Push and stop hooks** (`rk_loop.py:100-116`)
- What the hook does:
  - push = s·α·min(|h_t|, cap)·unit(d), from `rowstart` = pad + START, the first user-content token;
  - `rk.generate` sets `rowstart` (l.134);
  - the window is computed in chunk-local positions (l.105), and every decode step (T = 1) is pushed;
  - cap = 2·median norm (l.183);
  - leaf mode via `STEER['V']`;
  - a `Stop` exception after block b ends reads.
- (a) α is already a per-call argument. Put the norm-preserving control in a separate hook that rescales x+add to |x| per token, so round-1 parity is kept.
- (b) The hook is per token and scales. But "from the first user token on" pushes all 4k-32k context positions.
  - The plan must choose a context push or a continuation-only push; only the latter lets an unpushed prefix cache be shared.
  - With a reused prefix cache the window must be re-based, because the hook compares chunk-local positions with an absolute rowstart.
- (c) Later turns need a per-row turn mask (B-Q8 used no push in turn 2).
- (e) A neuron push is a pre-hook on `mlp.down_proj` (1.8).

**`keyseed` and `Gumbel`** (`rk_loop.py:122-126`; per-row keyed Gumbel-max over the full vocabulary at TEMP 0.7, l.34)
- (b) Unchanged.
- (c) Add the turn index to the key.
- (e) The H-neuron protocol samples at T 1.0 with top-k 50 and top-p 0.9. Add a truncation processor before Gumbel.
- Host cost: one generator loop per row per step (about 6 ops per row; est. 8 ms per step at B = 128).

**`rk_loop.generate`** (`rk_loop.py:129-143`; HF generate, left padding, batch 32, 40 new tokens)
- (b) Batch 32 fits only up to about 3k tokens per row.
  - The KV cache is 0.94 GB per row at 4k in fp32: 30 GB for 32 rows, plus 6.9 GB of weights.
  - That leaves too little for the prefill transients (≈ 0.47 GB per row at 4k) and the padding mask.
  - Left padding forces HF to build a [B,1,T,T] boolean mask (`masking_utils.py:231-281`). HF skips the mask only in two cases:
    - no padding, and q_len = 1 or kv_len = q_len;
    - an empty cache (q_offset = 0) with no padding mask or only trailing static-cache padding.

    Left padding never skips it.
  - With a mask, HF repeats K/V instead of using native GQA (`sdpa_attention.py:30-37`, `113-118`) and drops `is_causal` (l.140). SDPA turns the bool mask into an fp32 additive bias (`attention.cpp:579-592`, v2.14.1) of 4.29 GB per row at 32k, and the kernel computes all T² tiles.
  - Use B = 1, equal-length rows, or shared-prefix decode (3.7).
  - Generate keeps only the last-position logits in prefill (`generation/utils.py:2943-2947`).
- (c) Re-prefill each turn (simple and exact) or pass a carried cache.
- (d) Capture states during decode instead of re-reading (CbarHook pattern). This is exact at blocks ≤ b; see CbarHook below.
- (e) Neuron hooks work unchanged during generation.

**`r2_rk.evaluate_variant`** (`r2_rk.py:759-812`) and **`r2_theta.generate_rows`** (`r2_theta.py:362-393`)
- What they give:
  - keyed pairs with per-row signs and a temperature;
  - `gen_ids` and `read_ids` callables;
  - hooks active only during generation;
  - cbar and entropy logging;
  - three read windows.
- (a) Call with a per-round calibrated α. Run the dose series 0.25-2 × α in both signs via `signs` and `alpha`. Pass a control-push hook via `hooks=`.
- (b) `batch` must drop; reads re-prefill the context.
- (c) `gen_ids` builds n-turn inputs; `read_ids` selects the read convention.
- (d) Block-b windows only.
- (e) `hooks=` carries neuron interventions.

**`r2_rk.CbarHook`** (`r2_rk.py:603-643`)
- A prepend hook that logs pre-push clamped norms min(|h_t|, cap) per decode step and row (per position in read mode).
- It is the template for capturing block states during generation: an in-context read with zero extra forwards.
- The push is added to block b's output, so layers 0..b never see it.
  - Pre-push states at blocks ≤ b therefore equal the unsteered model's states for the same tokens.
  - States at blocks > b are steered.
- The last sampled token has no state.

**`r2_unlearn_qwen.ConstPush`** (`r2_unlearn_qwen.py:442-466`). A second push term using the pre-push norm with rk's cap rule. Template for a norm-preserving control and for a constant push across turns.

**`r2_rk.temperature`** (`663-674`): –.

### 1.2 Damage calibration

**`rk_loop.damage` and `push_damage`** (`rk_loop.py:186-196`, `257-263`). Mean KL(unsteered‖steered) over the continuation positions of 8 reference texts, teacher-forced, 2 forwards per text.
- (a) This is the metric to normalise on. Cache the unsteered side (RefKL), and also log the top-1 change rate (runde1b inputs README §4).
- (b) `model(ids).logits[0][pos]` (l.193-194) forms full-vocabulary logits at every position: 2.49, 9.96 and 19.9 GB fp32 at 4k, 16k and 32k.
  - Pass `logits_to_keep=pos` (`modeling_qwen3.py:449`, `480`; `modeling_utils.py:5008-5034`), or apply lm_head on positions as `r2_theta.read_full` does (`r2_theta.py:424`).
  - Calibrate on long-context references: a context push acts on every token, so damage depends on the length.
- (c) Damage on multi-turn references.
- The same logits caveat applies to `logodds_matching` (`rk_loop.py:369-377`), `damage_T` (`r2_rk.py:686-688`), `leaf_grads` (`r2_rk.py:739`) and `rk.grads` (`rk_loop.py:352`).

**Calibration loops:** `rk_loop.set_alpha` (`238-247`, target 0.05 nats), the K6 calibration (`379-390`, 0.03 nats), and the generic `r2_unlearn_qwen.calibrate(dmg_fn, start, rule)` (`r2_unlearn_qwen.py:223-233`; f ← f·sqrt(target/damage) until in band).
- (a) Use calibrate per loop round, so that damage(unit(d_n), α_n) = target. α_n must then also enter the gain formulas, which divide by α·cbar.
- (b) Calibrate per context length.
- (e) Also usable for neuron scale factors.

**`r2_unlearn_qwen.RefKL`** (`408-439`). Computes the unmodified log-probs once. `push()` and `hooks()` give the damage of a push or of any hook set.
- (a) 8 forwards per evaluation (est. 0.16 s at R-fwd 20 ms), so calibration costs ≤ about 1 s per round.
- Memory: it stores float64 [positions × 151,936] per text on the device, about 49 MB per 40-token text.
- (b) Same logits caveat (l.418, 426).
- (e) `hooks()` gives damage-matched random-neuron controls (the G45 lesson). For neuron sets across 28 layers, also match damage on held-out task texts, not only on the 8 round-1 references.

**Measured scale at block 14** (`theta_verdict.json`):
- The fixed α_b = 0.15 (blocks.14.B_Q0.alpha) let RSI_a push damage drift over rounds 0-3: 0.114, 0.202, 0.287, 0.323 nats (blocks.14.B_Q0.evaluations.RSI_a.{0..3}.damage_of_push).
- The theta damage curve on u_push is 0.097, 0.324, 0.913 and 2.374 nats at 0.5, 1, 2 and 4 α_b (blocks.14.cells.a{0.5,1,2,4}_T0.7.damage).
- Extrapolating below the lowest grid point (0.5 α_b) places a 0.05-nat push along u_push at about 0.34-0.36 α_b (est.).
  - The exponent used runs from 1.74 (the 0.5→1 α_b slope) to 2.
  - The slope falls at larger pushes: 1.49 for 1→2 and 1.38 for 2→4 α_b.
  - For the RSI start direction (0.114 nats at α_b) the same rule gives about 0.62-0.66 α_b.
- That is inside the window below 0.5 α_0 (α_0 = α_b at block 14) where λ_lin = 0.7298 [0.1139, 1.356] leaves a gain near 1 open (AUSWERTUNG.md Teil B).

### 1.3 Reading

**`rk_loop.read`** (`145-156`). Stopped forward after block b; float64 mean over continuation tokens.
- (b) l.154 runs `CAP.pop("h").double().cpu()` on the whole [B,T,D] block output. The float64 copy is made on the device and then copied to the CPU (est. 17 GB at B = 32, 32k).
  - Slice the continuation on the GPU first and read with B = 1 or equal lengths.
  - The same pattern sits in `r2_rk.read_windows` (l.422), `r2_theta.read_full` (l.416) and `r2_theta.score_grads` (l.469).
- (b) Better: reuse an unsteered prefix KV cache and run only the continuation as a chunk.
  - The Stop exception leaves only layers 0..b of a carried cache extended by L tokens. Crop only those layers: `cache.layers[l].crop(-L)` for l ≤ b. In transformers 5.22.2 `crop` takes a negative count (`cache_utils.py:163-180`).
  - `Cache.crop(-L)` (`cache_utils.py:1789-1797`) would also cut L prefix tokens from layers b+1..27, which the stopped forward never extended.
  - Alternatively run the chunk unstopped and crop every layer.
  - A chunk over a cached prefix has q_offset > 0 and q_len ≠ kv_len. HF then builds a small L × (P+L) mask and repeats K/V for that call.
- (c) Read as the reply to the plain prompt (B-Q8, `r2_theta.py:717`) or in context.

**`r2_rk.read_windows`** (`405-435`) and **`r2_theta.read_full`** (`396-435`)
- read_windows gives the windows all/div/first20 from an early-stopped forward.
- read_full gives the same windows plus the NLL of every continuation token from one full pass. It applies lm_head only at continuation positions (l.424): the pattern for every long-context logit.

**`r2_rk.read_all_blocks`** (`366-396`). One unstopped forward with hooks on any blocks; float64 means and optional per-token fp16 states.
- (d) The base for probes, with three changes:
  - l.381 runs `x.double()` on the full [B,T,D] per block. Accumulate the mean in fp32 into float64 sums, or slice first.
  - l.382 copies the whole sequence to the CPU in fp16: 134 MB per block per 32k row, 3.76 GB for 28 blocks. Stream to disk, subsample positions, or project on the GPU.
  - fp16 rounding: the selftest tolerance is 5e-3 (`r2_rk.py:1019-1023`). Use fp32 for regression targets.
- (e) Add down_proj pre-hooks. 172,032 values per token (344 KB in fp16) must be reduced on the GPU (span means, CETT) before moving.

**`r2_rgi_geom.Taps`, `equal_length_batches` and `stage_o8`** (`r2_rgi_geom.py:225-240`, `252-258`, `466-503`):
- hooks on all blocks;
- batches without padding;
- float64 mean and Gram accumulation on the GPU ([28, 2048, 2048] = 0.94 GB) with an outlier filter (norm < 10 × median).

This is the template for streaming lens accumulators (4b, 4c).

### 1.4 Axes, nulls, statistics, bootstrap

**`rk_loop.axes` and `r2_rk.axis_from_texts`** (`rk_loop.py:208-236`; `r2_rk.py:455-481`). Text axes with fold cross-fitting, AUC, gap_perp_cross, and 2,000 sign-flip null axes.
- The nulls are built as E·diff/P from uncentred per-prompt differences (`rk_loop.py:223`; `r2_rk.py:471-473`). This is round 1's conservative-p weakness.
- Why: with diff_i = ū + e_i, a null axis is s_k·ū + (1/P)Σ_i ε_ki·e_i, where s_k = (1/P)Σ_i ε_ki has variance 1/P. On average a share |ū|²/(|ū|² + σ_e²) of its squared norm therefore lies along ū (σ_e² = mean |e_i|²).
- Fix: centre (diff − mean) before flipping, which removes the s_k·ū term, and add a paired sign test on the loop pairs; nothing else changes.
- Round-1 states keep the per-prompt pro/anti reads (`ST['axis_<name>']` Xp, Xa) and the fold, so centred nulls can be rebuilt for R1 axes without new texts. The stored R1 nulls are uncentred.

**`rk_loop.stats` and `r2_rk.stats_on_axis`** (`283-307`; `546-566`). Both give c, p, Q, rel from 20 halvings, and A; rk.stats adds Ue and sigma on u_T_perp. Disattenuated cosines can exceed 1 (AUSWERTUNG §4): report raw and corrected values.

**Axis reliability and fixed axes:** `r2_rk.axis_rel`, `halvings`, `fixed_fold_axes`, `gap_on_fixed_axes`, `cross_fit_on_axes` (`525-543`, `449-452`, `484-515`): –.

**Bootstrap:**
- `r2_rk.prompt_bootstrap`, `pair_weights`, `boot_mean`, `interval` (`570-599`);
- `r2_theta.boot_W` and `Pairs` (`116-147`; U statistic without self-pairs);
- `r2_axes._w_axis` and `_w_rel` (`131-146`; per-replicate axes on the GPU).

(b), (c): for long contexts the units become documents or conversations. Keep at least 32-64 units for percentile intervals.

**`r2_linearity`** (`72-380`):
- `Concept`, a token-state container;
- closed-form ridge probes (`moments`, `ridge_lam`, `solve_spd` with Cholesky in float64, `probe_dir`);
- LEACE (`leace`, `erase`) and `auc_cols`;
- sign-flip and permutation nulls in one multi-RHS solve (`nonlinear` 265-289);
- INLP, and a planted-signal calibration (347-380).

Notes:
- (d) Directly reusable per block at D = 2048.
- (e) Fine at 6,144 neurons per block. Across all 172,032 neurons a sparse solver is needed, and none exists in the repo.

### 1.5 Loops and gains

**`rk_loop.evaluate`, `run_loop`, `loop_specs`** (`265-334`). run_loop fixes one α for all rounds (l.312).
- (a) Use the variant driver below.
- (b) Every round re-prefills each context per sign: est. 21 s per 32k prefill, so 256 prefills cost about 1.5 h. Long-context loops need few contexts (8-16) or a shared unpushed prefix with a continuation-only push.
- (e) A neuron-space loop would carry d in the 6144-dim down_proj input of one block.

**`r2_unlearn_qwen.loop_round`, `round_stats`, `run_loop_variant`** (`969-1001`). Keys per (ns, n, p), optional orthogonalisation of each push, per-round damage via RefKL (measured, not calibrated; α fixed), bootstrap Ue.
- (a) The place for the damage-normalised push: call `calibrate` with `RefKL.push` before each round, and log α_n.
- Add a norm-preserving control loop via a ConstPush-like hook.
- An arm exactly as round 1 (runde1b README §5) is `run_loop` unchanged.
  - It repeats the procedure, not the texts.
  - The state guard forces r2-prefixed loop names, hence new keys.
  - Block 14 of round 1 ran on an L4.

**`r2_gen_loop.run`** (`36-84`). A loop from a new start axis with read-outs on extra axes against their own nulls, plus E, R4 and S4. Template for loops started on other axes.

**`r2_theta` gains** (`116-345`): Pairs, eval_gains (g_par, G, rotation), bq0 (mode set, u*, guard, cross-fit), cell_gains, ivpool, choose_power_cell.
- The gains divide by a·α_b·cbar (l.173-185, 303-314).
- Per-cell α values from damage normalisation fit as is; per-round α_n in loops must be passed explicitly.
- Log cbar per cell.

### 1.6 K over follow-up answers (B-Q8)

**`r2_theta.stage_turn2`** (`695-739`):
- Inputs: PIDS[p] + turn-1 ids + TURN2_SUFFIX ids (`r2_stimuli.py:208`, `214-217`), asserted once against the template (`check_turn2` 225-234).
- No push in turn 2; keys keyseed('theta2', b, a, T, p, s) are shared within a pair (l.714).
- Turn 2 is read as the reply to the plain prompt (l.717).
- K = u_read·d^(2) / u_read·d^(1) (l.728-734).

**`r2_theta_verdict.turn2_block` and `rule_R6`** (`565-580`; `236-255`). K is defined only if the interval of u_read·d^(1) excludes 0. "Self-sustaining" requires K's lower bound > 1. The transfer test compares log K and log G.

(c) Generalise to n turns:
- ids_n = PIDS + t1 + SUF + … + t_(n−1) + SUF;
- turn-indexed keys;
- K_n and the product;
- R6 per turn.

Two conventions to fix:
- The read: B-Q8 reads context-free. An in-context read needs a cached-prefix pass or generation-time capture; capture is exact at blocks ≤ b.
- History think blocks: the concatenated ids keep each history turn's empty think block, while the official template drops it (`r2_stimuli.py:231-232`). That is 4 off-template tokens per history turn: `<think>`, `\n\n`, `</think>`, `\n\n`.

Power:
- With K ≈ 0.45-0.48 along the style direction, the turn-n signal halves each turn.
- u_read·d^(1) = 8.33 [7.44, 9.23] falls to about 0.81 at turn 4 and 0.37 at turn 5. Beyond 3-4 turns u·d^(n) will therefore rarely exclude 0 (G26).
- On u_T_perp, K(T 1.0) = 1.053 [0.743, 1.538] (AUSWERTUNG; descriptive, with no interval reported for its denominator) allows persistence; the chain is the test.

Further notes:
- (b) Context grows by at most 56 tokens per turn (≤ 40-token answer plus the 16-token suffix) from a prompt of about 21 tokens. That gives at most about 580 tokens after 10 turns and about 5.6k after 100.
- (a) Two arms: push in turn 1 only, or push in every turn.

### 1.7 Gradients and Jacobian-like readouts

**Functions:**
- `rk_loop.grads` (`346-354`);
- `r2_rk.leaf_grads` (`716-744`);
- `r2_rgi_geom.leaf_hooks` and `a3_grads` (`331-357`);
- `r2_theta.score_grads` (`438-476`) with `r2_theta_verdict.lambda_lin` (`257-267`).

All are zero-leaf gradients at block outputs. Checks:
- leaf_grads and score_grads against central finite differences (`r2_rk.py:1139-1153`; `r2_theta.py:903-915`);
- a3_grads against a 5-point stencil on a float64 MOCK model (`r2_rgi_geom.py:676-692`);
- rk.grads has no check of its own.

1b uses:
- Jᵀu (reverse mode) for J-lens and MicroLens++ read directions;
- Jv by central differences (2 forwards per direction, forward-only memory);
- λ_lin for the small-push regime.

Changes:
- (b) Backward stores the activations of every layer above the leaf block at all positions.
  - That is est. about 4-6 GB per layer at 32k in fp32: ≈ 115-165 GB for all 28 layers, ≈ 54-77 GB for the 13 layers above block 14.
  - leaf_grads and rk.grads add a 19.9-GB full-logit tensor, because they call `model(...).logits`.
  - This is infeasible at 32k. About 14-21 GB at 4k is fine.
  - 16k needs gradient checkpointing. Its recompute re-runs forward hooks, so the Stop hook must be off and the push hook pure.
- (e) Leaves on the down_proj input give neuron attributions a ⊙ ∂y/∂a.

### 1.8 Write-orthogonalisation and neuron hooks

**`r2_rk.WriteOrth`** (`695-712`). Removes span(U) from embed_tokens, 28 o_proj and 28 down_proj outputs (57 hooks; 23 vs 20 ms per short forward, plan:1379).
- (a) Low-damage erasure variants:
  - fewer blocks and partial removal are parameter changes;
  - mean ablation x − QQᵀ(x − μ) needs a stored mean μ per hooked module (a small code change).
- (b) Unchanged.
- (e) The neuron analogue is a pre-hook scaling the down_proj input per block, with the same context-manager structure.

**Neuron code outside round 2:**
- `neu_extract.py:16-21` hooks capture a = silu(gate)·up at the down_proj input, plus MLP and attention outputs (last prompt token only).
- `neu_extract.py:39` saves the W_down column norms.
- `neu_sparsity.py:33-50` has per-row pre-hook interventions (top-K, random, rotated basis).

These scripts load the model themselves. Their docstrings cite an essay on a domain the brief forbids accessing: reuse the code, not those citations.

### 1.9 Runner, gates, sync, time guard, notebooks

**`r2_run_session.py`:**
- Steps and tables: Step (`57-81`), STEPS and SESSIONS (`103-187`).
- Execution: `exec_step` (`1084-1144`; per-step timeout 3,600 s at l.967) and `run_steps` with one retry (`1184-1223`).
- Time guard: GUARD_S_DEFAULT 1,200 s (l.263). The clock starts at the notebook's first cell (R2_T0, l.960-971) and is checked only after a step that ran (l.1213-1216).
- Wrap-up: `finalize` (`1264-1323`).
- `sync` (`378-405`): per-file, atomic rename, error-tolerant; skips `local/`, `r1/`, `r2in/` (l.265).
- Gates: `gate_G1` (`760-783`: plan sha256 and freeze-commit ancestry) and `gate_G2` (`786-800`: A100 checked in a child process).
- 1b changes: new tables and freeze constants. The guard acts only between steps, so long work (32k prefills, 4M-token streams) must be split into resumable parts, as theta/parts did (`r2_theta.py:677-679`, `687-690`). Float64 accumulators of several GB cannot be resumed across sessions unless they are saved to Drive.

**`r2_report.session_records`, `packing`, `merged_manifest`** (`825-884`): reuse as is.

**`build_runde2_gpu_notebooks.py`:**
- revision pins (19-21) and accounts (65-66);
- first cell with T0 and the A100 assert (190-206);
- Drive mount (208-216) and the resume copy from Drive (219-221);
- the second-account copy cell (224-270);
- sparse partial clone (271-281);
- pinned downloads with checksums (283-333);
- self-test marker per commit (340-360);
- main run (362-378);
- final sync and unmount (400-406).

### 1.10 Cross-cutting changes

- **(a)** `calibrate` + RefKL per round, with α_n carried into the gains; a norm-preserving hook; centred nulls; a dose series via evaluate_variant.
- **(b)**
  - B = 1 or equal-length rows (no padding masks);
  - `logits_to_keep` in every teacher-forced logit call, including leaf_grads and rk.grads;
  - GPU-side slicing before `.double().cpu()`, at all four call sites;
  - KV-prefix reuse with per-layer crop and re-based push windows;
  - an SDPA backend assertion;
  - a decode cache that does not rewrite itself every step (to be measured in the pilot);
  - gradient checkpointing or finite differences beyond 4-8k.
- **(c)** An n-turn driver generalising stage_turn2 and R6 per turn; read convention and think-block convention fixed in the plan.
- **(d)** Streamed or projected per-token dumps; Taps plus float64 accumulators.
- **(e)** A down_proj pre-hook manager (capture, CETT, scaling); a GPU L1-logistic solver; damage-matched random neuron sets via `RefKL.hooks`.

## 2 Cost model (Colab A100, float32, TF32 off)

### 2.1 Planned rates (plan = `results/ju_space/plans/abschnitt11_12_runde2_plan.md`)

| Item | Planned value | Source |
|---|---|---|
| Host time | 10 µs per launched op, 20 µs per sync | plan:1376 |
| Decode step, batch 32 | about 1,760 ops ≈ 18 ms host plus compute | plan:1378 |
| One short teacher-forced forward (R-fwd) | 20 ms; 23 ms with WriteOrth | plan:1378-1379 |
| Forward + backward with leaf vectors | 60 ms | plan:1379-1380 |
| Model load in a new process | 30 s | plan:1380 |
| Compute per token | 2.8 GFLOP through 28 blocks; +0.6 GFLOP for full-vocab logits; fp32 peak 19.5 TFLOP/s | plan:1381-1382 |
| Full pass with logits (R-full), batch of 32 × ~70 tokens | 0.45 s | plan:1383-1387 |
| 28 blocks without logits (R-all), same batch | 0.40 s | plan:1383-1387 |
| Read stopped at block 14 (R-read), same batch | 0.25 s | plan:1383-1387 |
| Read stopped at block b, same batch | ≈ 0.45(b+1)/28 s; costed at 0.35 s for extra blocks | plan:1383-1387 |
| 20-token sentences, batch of 32 | 0.15 s | plan:1386-1387 |
| One generate call, 32 prompts × 40 new tokens (R-gen) | 1 s; 1.15 s with WriteOrth; 1.3 s with 80-token prompts; 2 s with 200-token prompts | plan:1388-1390 |
| Setup, Qwen sessions | 4 min (pip, clone, 3.4 GB model with checksum, Drive copy-in ≥ 20 MB/s, MOCK self-test) | plan:1400-1401 |
| Setup, toy and analysis sessions | 3 min | plan:1400-1401 |
| Margin | ×1.5; ≤ 18 min with margin; 20-min target | plan:1402-1403 |

Status of these rates: "given facts" from code reviews. Costing from round-1 per-op times was "not adopted" (plan:1742). Sensitivity check: at half the rates, r2_q1, r2_q2 and r2_t1 need 10.1, 10.6 and 11.3 min of work (plan:1404-1406).

**Derived rates (est., from the planned rates):**
- **Generation:** 1,280 new tokens/s at batch 32. About 985 with 80-token prompts, about 640 with 200-token prompts.
- **Forwards:**
  - 50 single short texts per second;
  - batched full pass about 71 texts/s (≈ 5,000 tokens/s);
  - block-14 read about 128 texts/s (≈ 9,000 tokens/s).
  - Implied effective rate: 6.3 TFLOP / 0.40 s = 15.8 TFLOP/s (81 % of fp32 peak). This is a plan assumption, not a measurement.
- **Per loop round:**

| Round type | Est. time |
|---|---|
| 128-pair round at block 14 (8 calls + 8 reads + 0.3 s damage of d_(n+1)) | ≈ 10.3 s (0.17 min) |
| 128-pair round at an extra block | ≈ 11.1-11.2 s (plan's r2RSIb row: 44.8 s per 4 rounds, plan:1418) |
| 64-pair round | ≈ 5.3-5.4 s (plan's GEN loop: 21.6 s per 4 rounds, plan:1417) |
| 128-pair theta cell with NLL pass | ≈ 11.6-11.9 s |
| 128-pair turn 2 | ≈ 12.4 s |
| Damage calibration per round with RefKL | ≤ about 1 s extra |

Round 1's plan costed 4.4 min per 128 texts on the CPU (`abschnitt11_rsi_kennwerte_plan.md:93`), i.e. 8.8 min per 128-pair RSI round (l.98). These are planning estimates; the run itself was moved to A100 GPUs (l.129-130).

### 2.2 Measured

**r2_ana on an A100-SXM4-40GB** (`results/ju_space/runde2_ergebnisse/session_done_r2_ana.json`):

| Step | Measured | Planned (plan:1425) |
|---|---|---|
| linearity | 18.9 s | C-Q1 90 s |
| theta_verdict | 132.5 s | 110 s |
| rgi_verdict | 171.5 s | 127 s |
| unlearn_verdict | 3.5 s | 30 s |
| report | 2.5 s | 20 s |
| Total (seconds_steps) | 328.9 s | about 377 s plus starts |

- For comparison, C-Q1 took about 870 s on one CPU thread on synthetic data at real size (`NOTES_w8.md:266-267`).
- **Successful attempt** (attempts[2]): runner time 976.7 s; elapsed since the first cell 1,042.0 s.
  - The 'started' stamp (13:05:42) is written after gates G1/G2 and the start conditions (`r2_run_session.py:1405`).
  - `seconds` counts from the runner's own start (l.1365), which was therefore at about 13:05:22, about 65 s after t0 (13:04:17).
- **Overhead:** 976.7 − 328.9 ≈ 648 s of runner time fell outside steps. It covers gates (including a child-process GPU check), copy-in, hashing, manifest and per-step syncs. The plan budgeted 60 s for copying about 0.92 GB (plan:1425). The breakdown is in logs on Drive only.
- **Sub-step timings** in the same session:
  - theta_verdict.json `seconds` 130.1, including the power re-run: 75.4 / 25.9 / 25.9 s at blocks 14 / 18 / 6;
  - rgi_verdict.json `seconds` 170.8.
- **Attempts:** attempts 1-2 only waited for r2_pc (attempts[0..1].waiting).

**Other measured records:**
- No session stopped on the time guard and none needed a continuation session (`round2_summary.json` gates.G48; deviations.packing.*.time_guard_stops = 0).
  - The guard is checked only between steps, so this bounds elapsed time before each session's last step, not total duration.
- **GPUs:** A100-SXM4-40GB for 8 sessions and A100-SXM4-80GB for r2_t4, CUDA 13.0 (gates.G2.sessions).
- **Versions:** torch 2.14.1+cu130 and transformers 5.22.2 (gates.G47.session). Round-1 gpu1 ran on an NVIDIA L4 with the same versions (gates.G47.r1_gpu1).
- **Cross-device CRN** (L4 vs A100 regeneration with the same keys, gates.G47.match_rate; n = 64 each):
  - RSI_PRO identical 0.96875 (prefix-token share 0.98164);
  - RSI_ANTI 0.9375 (0.98047).

### 2.3 Not measured

- Qwen step times of r2_q1, r2_q2 and r2_q3 (their session_done.json files are on Drive only).
- Round-1 shard timings (Drive only).
- Anything at contexts above about 200 tokens.
- The real Drive throughput.

## 3 Long context with Qwen3-1.7B

### 3.1 Configuration and native length

From `config.json`:

| Field | Value |
|---|---|
| hidden_size | 2048 |
| intermediate_size | 6144 |
| num_hidden_layers | 28 |
| num_attention_heads | 16 |
| num_key_value_heads | 8 |
| head_dim | 128 |
| max_position_embeddings | 40,960 |
| rope_theta | 1,000,000 |
| rope_scaling | null |
| sliding window | none |
| tie_word_embeddings | true |
| vocab_size | 151,936 |
| torch_dtype | bfloat16 |

**Native context: 32,768.**
- Model card (pinned revision):
  - "Context Length: 32,768";
  - 1.4B non-embedding parameters;
  - GQA with 16 query and 8 KV heads;
  - no YaRN or long-text section;
  - it recommends output lengths up to 38,912 tokens for hard problems.
- Qwen3 Technical Report (Qwen Team, arXiv 2505.09388, 2025):
  - the long-context stage trains at 32,768 tokens and raises the RoPE base from 10,000 to 1,000,000 by ABF;
  - YaRN plus DCA give 4× at inference;
  - Table 1 lists Qwen3-1.7B at 32K and 4B and larger at 128K.
- Positions 32,768-40,960 are allowed by the config but lie beyond the 32,768-token long-context pre-training stage. The sources do not say whether post-training covered them for this size. Treat them as unvalidated.

Parameters: 1,720,574,976 (tied), i.e. 6.88 GB in fp32 and 3.44 GB in bf16 (computed). This matches the 3,441,185,608-byte shard 1. Shard 2 holds only the untied `lm_head.weight` copy (622,329,856 bytes).

### 3.2 KV cache

Per token: 2 × 28 × 8 × 128 = 57,344 values, i.e. 229,376 B (224 KiB) in fp32 and 114,688 B in bf16.

| Context | fp32 | bf16 |
|---|---|---|
| 4k | 0.94 GB | 0.47 GB |
| 16k | 3.76 GB | 1.88 GB |
| 32k | 7.52 GB | 3.76 GB |
| 40,960 | 9.40 GB | 4.70 GB |

### 3.3 SDPA kernels in float32 on A100 (torch 2.14.1)

- **Flash attention:** rejected; it needs fp16 or bf16 on sm80 (`check_dtypes_low_precision`, sdp_utils.cpp l.894-903).
- **cuDNN attention:** also low precision only (l.998).
- **Memory-efficient kernel:**
  - It accepts float32 on sm80 (l.1112-1113, 1176-1182).
  - It supports GQA at v2.14.1 (`supports_gqa = true` in the dense constraints of `can_use_mem_efficient_attention`, l.1144-1155).
  - For f32 on sm80+ it runs on tensor cores with `OpMultiplyAddFastF32` (3xTF32 emulation; `mem_eff_attention/gemm_kernel_utils.h` l.130-143), so the TF32 flag does not govern it.
  - Accuracy is near fp32 but bitwise different from the math and CPU kernels.
- **Math fallback:** O(T²) score memory, [16, T, T] fp32 per layer per row: 1.07 GB at 4k, 17.2 GB at 16k, 68.7 GB at 32k. The softmax output needs a second buffer of that size.
- **Order:** flash, then memory-efficient, then math. cuDNN is moved first only on sm90/sm100 with a recent cuDNN (l.94-135). With fp32 inputs the memory-efficient kernel is chosen whenever its checks pass.
- **Inference about rounds 1/2:** they already ran in this kernel family (the backend was not logged).
- **HF dispatch:** HF uses native GQA and `is_causal` only when it passes no mask (`sdpa_attention.py:30-37`, `113-118`, `140`).
  - With padding it repeats K/V and passes a bool mask.
  - SDPA turns that mask into an fp32 bias (`attention.cpp:579-592`) with no causal tile skipping.

### 3.4 Peak memory of a 32k prefill (B = 1, est.)

| Component | Memory |
|---|---|
| Weights | 6.88 GB |
| KV cache | 7.52 GB |
| Largest per-layer transient (MLP gate/up/act at 0.81 GB each plus hidden copies) | ≈ 3.8 GB |
| **Total** | **≈ 18 GB** |

Extra rows made as cache copies of that prefill add 7.52 GB each:
- with the prefill transient still counted, ≈ 26 GB at B = 2 and ≈ 34 GB at B = 3 on a 40-GB card;
- once the transient is freed, ≈ 22 / 29 / 37 GB at B = 2 / 3 / 4.

A batched B-row prefill instead needs ≈ 29 GB at B = 2 and ≈ 41 GB at B = 3, which does not fit.

Add-ons:
- padding mask: about +5.9 GB (bool mask 1.07 + fp32 bias 4.29 per row + repeated K/V);
- all-position logits: +19.9 GB;
- `output_hidden_states` (as in `rk_loop.py:419`): +7.8 GB;
- math backend: +68.7 GB per layer;
- backward without checkpointing: about +115-165 GB for all layers (about +54-77 GB above block 14).

### 3.5 Does the round-1/2 path scale to 16k-32k?

- **HF generate with hooks:**
  - The hooks are per-token ops: yes. With a reused prefix cache the push window must be re-based (`rk_loop.py:105`).
  - Batch 32 with left padding: no. Use B = 1, equal-length rows or shared-prefix decode.
- **float32:** yes, via the memory-efficient kernel.
- **Gumbel-max:** unaffected.
- **Stopped forward for reading:** yes, after:
  - slicing before `.double().cpu()` (`rk_loop.py:154` and the three other sites in 1.3);
  - KV-prefix reuse, otherwise every read re-prefills the context;
  - cropping only the layers ≤ b after the Stop exception.
- **Damage, logodds, RefKL:** only with `logits_to_keep`.
- **Leaf gradients:** only up to about 4-8k without checkpointing.
- **read_all_blocks per-token dumps:** must be streamed or projected.
- **Statistics:** unaffected.

### 3.6 Throughput (est.)

Assumptions:
- linear layers at the plan's 15.8 TFLOP/s;
- attention at an assumed 8-20 TFLOP/s, unmeasured for the 3xTF32 kernel;
- decode host-bound at about 18 ms per step (plan:1378);
- HBM bandwidth 1.555 TB/s on the 40-GB card;
- prefill attention = 114,688·T² FLOP (causal);
- decode attention = 229,376·T FLOP per token.

**Prefill and B = 1 decode:**

| Context | Prefill, B = 1, no mask | Prefill with padding mask | Decode, B = 1 |
|---|---|---|---|
| 4k | 0.8-1.0 s (4.2-5.0k tok/s) | 0.9-1.2 s | ≈ 18-20 ms/step (≈ 50 tok/s) |
| 16k | 4.5-6.8 s (2.4-3.7k tok/s) | 6.0-10.6 s | 18-30 ms (33-55 tok/s) |
| 32k | 12-21 s (1.5-2.7k tok/s) | 18-37 s | 18-45 ms (22-55 tok/s) |

**Aggregate decode (40 GB):**

| Context | Rows with own caches | Memory-bound bound, no cache rewrite | With HF's default DynamicCache | Shared-prefix decode, B = 32 |
|---|---|---|---|---|
| 4k | ~30 | ≤ ≈ 950 tok/s | ≈ 450 tok/s | ≈ 1,800 tok/s |
| 16k | ~7 | ≤ ≈ 290 tok/s | ≈ 120 tok/s | ≈ 1,500 tok/s |
| 32k | 3 | ≤ ≈ 140 tok/s | ≈ 60 tok/s | ≈ 1,000 tok/s |

Notes on the aggregate table:
- `DynamicLayer.update` grows each layer by torch.cat (`cache_utils.py:135-136`). Every decode step therefore reads and rewrites the whole cache, about 2 × KV of extra HBM traffic. B = 1 decode at 32k stays near 20 ms/step.
- The shared-prefix figures assume a custom cache and attention. They also assume 15.8 TFLOP/s for the skinny fp32 prefix products with TF32 off, which is optimistic.

Risk: decode at B = 1 with q_len = 1 runs the memory-efficient kernel on only 16·B thread blocks. At 32k this may add 15-40 ms per step. A pilot measurement is needed.

### 3.7 Branching many continuations from one long prefix

1. **Prefill once.** B = 1, no mask, causal; one prefill per (context, push condition).
   - A push that acts on the prefix changes the KV of every layer after b, so each push condition needs its own prefill.
   - A continuation-only push lets one unpushed prefix serve all arms.
2. **If B copies fit** (32k: B ≤ 3 on 40 GB, B = 4 at ≈ 37 GB to be tested; ≤ 9 on 80 GB):
   - Call `cache.batch_repeat_interleave(B)`. This is the `Cache` method (`cache_utils.py:1799-1802`), which calls `DynamicLayer.batch_repeat_interleave` (l.182-186).
   - Then generate with `past_key_values` and the full ids. Only the uncached tail is processed, so keep at least one input token uncached.
   - All rows have equal length, and the decode steps need no mask.
   - The ± rows of a context-push pair come from two different prefills, so their caches must be concatenated along the batch by hand; no helper exists.
   - rk's steer hook (chunk-local window, `rk_loop.py:105`) and CbarHook ("T > 1 = prefill", `r2_rk.py:619`) must be adapted to the tail chunk.
3. **For many branches:** a shared-prefix attention, registered through transformers `AttentionInterface` (`modeling_utils.py:5151`) with a custom cache.
   - It attends the prefix once with all branch queries batched, attends each suffix causally, and merges the two by log-sum-exp.
   - This is the decomposition of Hydragen (Juravsky, Brown, Ehrlich, Fu, Ré, Mirhoseini, 2024, arXiv 2402.05099).
   - It is exact up to fp32 reassociation. Prefix KV is stored once, plus about 9 MB per 40-token suffix.
   - Its fp32 torch matmuls differ bitwise from the memory-efficient kernel, so both arms of a CRN pair must use the same path.
4. **Reads by the unsteered model:** keep its prefix KV, run each continuation as one chunk, then crop the extended layers (only layers ≤ b after a stopped read). Alternatively, capture states during decode, which is exact at blocks ≤ b.
5. **Bookkeeping:** pass explicit `position_ids`/`cache_position` in custom loops. Keep both arms of a CRN pair in the same batch with identical shapes.

### 3.8 Numerical caveats

- **fp32 vs bf16.** fp32 weights are exact upcasts of the bf16 checkpoint.
  - bf16 compute changes logits at about 1e-2 relative (est.), flips Gumbel near-ties and breaks parity with rounds 1/2.
  - Use bf16 only for probe fitting or evaluation, never for compared generations or reads.
- **TF32 off.** This covers cuBLAS (`ufo_common.py:18`) but not the 3xTF32 attention kernel. Bits depend on kernel, batch shape and padding.
- **Cached vs recomputed.** Check cached-versus-recomputed logits on a few branches: max relative difference < 1e-4, as GO2 required for early-stop vs full read (`rk_loop.py:412-426`). Also count Gumbel flips.
- **Cross-device.** L4 vs A100 gave 96.9 % and 93.8 % identical texts of up to 40 tokens.
  - That implies a per-token divergence hazard of about 0.8-1.6e-3, so long texts diverge across devices.
  - Never mix GPUs within paired comparisons, and log GPU, torch, transformers and CUDA.
- **bf16 KV with fp32 compute.** This doubles the branch count but is a separate numerical arm. SDPA needs equal dtypes, so K/V are upcast per layer at each step.

## 4 Feasibility of the H-CHAIN measurements

### 4(a) Hallucination-associated neurons in Qwen3-1.7B

**Reference protocol:** Gao, Chen, Xiao, Chen, Liu, Sun, "H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs", arXiv 2512.01797, 2025:
- TriviaQA with 10 samples per question (T 1.0, top-k 50, top-p 0.9);
- only all-correct or all-wrong questions are kept, giving 1,000 + 1,000 examples;
- answer spans are aligned with GPT-4o;
- features are CETT = ‖a_j·W_down[:, j]‖ / ‖FFN output‖, averaged over answer and other tokens;
- an L1 logistic regression. C comes from a grid search that maximises held-out accuracy plus TriviaQA performance with the selected neurons suppressed;
- 0.01‰-0.35‰ of neurons selected; intervention α ∈ [0, 3];
- the causal tests use over-compliance benchmarks (FalseQA, FaithEval, Sycophancy, Jailbreak) with GPT-4o judges;
- models studied: Mistral, Gemma and Llama only, no Qwen3.

The paper's prose labels the answer-token features of hallucinated responses y = 1. Its label formula, as rendered in the arXiv HTML of v2, labels faithful responses y = 1. Resolve this before replicating.

**Qwen3-1.7B scale:** 28 × 6144 = 172,032 neurons. The paper's ratios imply about 1.7-60 H-neurons; 0.1 % would be 172.

**Cost (est.):**
- **Screening:** about 6,000 questions × 10 samples = 60,000 answers of ≤ 24 tokens.
  - 6-22 min (batch 128 vs batch 32 at R-gen rates).
  - Qwen3-1.7B's TriviaQA accuracy and the consistency yield are unknown; pilot with 500 questions.
- **CETT extraction:** 2,000-4,000 rows, teacher-forced, ≈ 0.5 min. Uses down_proj pre-hooks (`neu_extract.py:16-21`), the W_down column norms (l.39) and the MLP output norm.
- **L1 path on the GPU:** X is 4,000 × 172,032 fp32 = 2.75 GB.
  - A FISTA iteration (2 matvecs) takes ≈ 4 ms.
  - 30 C values × 5 folds × 500 iterations, plus 50 stability resamples and 100 permutation fits: ≈ 9-12 min.
  - Such a solver is not in the repo.
- **C selection as in the paper:** one suppression run (α = 0) per grid value on about 1,000 held-out questions: ≈ 20 s per value, 3-10 min for 10-30 values.
- **Interventions:** α ∈ {0, 0.5, 1.5, 2, 3} on the H-set and on 2 damage-matched random sets (`RefKL.hooks`, also matched on held-out answers), plus the unmodified baseline (α = 1), × 1,000 held-out questions: ≈ 5-6 min.
- **Total:** 25-50 min of work, ≈ 0.5-1.1 GPU-h with setup, 3 sessions (or one longer session).

**Storage:** 1.4-2.8 GB local (fp16 or fp32 CETT); < 0.1 GB to Drive (indices, coefficients, labels, answers).

**Risks:**
- p/n ≈ 43-86, so stability selection and a permutation null are mandatory;
- labels must come from alias matching, since GPT-4o span extraction is not available; this changes the "answer tokens" feature;
- a TriviaQA-only intervention measures a different outcome than the paper's compliance benchmarks;
- TriviaQA must be downloaded with a pinned revision and checksum.

### 4(b) Future-lens probes, blocks to tokens t+1..t+8

Related method: Pal, Sun, Yuan, Wallace, Bau, "Future Lens: Anticipating Subsequent Tokens from a Single Hidden State", CoNLL 2023, arXiv 2311.04897.

**Recommended design:** closed-form ridge from h_b(t) to the final residual h_L(t+k), decoded by the model's own final norm and unembedding.
- Decoding ĥ_L(t+k) predicts token t+k+1; fix the index convention in the plan.
- Stream X_bᵀX_b and X_bᵀY_k in float64, following `stage_o8` (`r2_rgi_geom.py:466-503`) and Taps.
- Solve with `r2_linearity.solve_spd`.
- Fitting this forward map (and a past lens) by regression is allowed under the inverse-free rule. It is not a reconstruction, and must not be used as one.

**Cost for 4 M tokens (est.):**

| Item | Est. |
|---|---|
| Forward through 28 blocks (11.3 PFLOP) | 11.9 min |
| Accumulation (28 blocks × 9 matrices, 8.46 PFLOP) | ≈ 9 min |
| Solve | < 1 min |
| Held-out evaluation, 100k tokens × 224 probes | 15 min fp32 / about 1.5 min bf16 |
| **Total** | ≈ 25-40 min ≈ 0.4-0.7 GPU-h |

**Memory and storage:**
- Model 6.9 GB plus float64 accumulators 8.46 GB plus activations: fits 40 GB.
- A past-token lens (offsets −1..−8) adds 8 cross matrices per block: 1 Gram + 16 cross = 16.0 GB, which still fits.
- The accumulators cannot be split across Colab sessions without saving them, because `local/` is not synced and the VM is lost. Three options:
  - run the 4 M tokens in one session (≈ 25-40 min; the round-1b design rules allow sessions up to about 5 h);
  - checkpoint 8.5-16 GB of float64 sums to Drive between sessions;
  - fit per session and compare.
- Drive: solved probes 1.9 GB fp16, or 0.23 GB for 7 blocks × 4 offsets.
- Storing states instead of streaming would need 115 GB per 1 M tokens for all blocks: not feasible.
- SGD probes with a full-vocab loss would take 29.4 h in fp32 and 3.1 h in bf16 (at an assumed 150 TFLOP/s) for 224 probes over one epoch, so they suit only a few (b, k).
- Data: the repo holds well under 1 M tokens of generic text (chars/4, including JSON overhead):
  - corpus_wikitext: 40 passages, ≈ 9k tokens;
  - mmlu_topics: ≈ 0.49 M;
  - gsm8k: ≈ 0.19 M;
  - dn_text: ≈ 56k.

  A pinned public corpus must be downloaded.

### 4(c) Per-token channel time series, forward vs backward predictiveness

**Method caveat.** For a stationary process, the forward and backward one-lag linear predictors of the whole state have residual covariances with equal determinants:
det(C0 − C1ᵀC0⁻¹C1) = det(C0 − C1C0⁻¹C1ᵀ) by Sylvester's identity. Equivalently, the Gaussian mutual information between consecutive fronts is symmetric.

A numerical check on a non-normal VAR(1) gave equal determinants (840.479) but different per-channel R² values (forward 0.50-0.86, backward 0.58-0.80). The traces also differed (42.6 vs 64.4).

The user's "vector quantity" must therefore be defined in one of these ways:
- per channel, or by another coordinate-dependent summary such as total MSE;
- with token or nonlinear targets;
- through position dependence (early vs late context). That last option is also the length-artifact test.

**Cost (est.):**
- Streamed with (b): lagged covariances of 128-dim PC projections at 8 blocks for lags 1-64 take 67 MB and +1-2 min.
- Long-context arm (prefill only): 64 × 16k ≈ 5-7 min; 64 × 32k ≈ 13-23 min.
- Total ≈ 0.1-0.5 GPU-h.

**Storage:** statistics take MB. Stored series (8 blocks × 128 dims, fp16) take 2 KB per token: 2 GB per 1 M tokens, 4.3 GB for 64 × 32k.

### 4(d) Summary

| Measurement | GPU work (est.) | Sessions | Local disk | Drive |
|---|---|---|---|---|
| H-neurons | 25-50 min | 3 of ≤ 20 min, or 1 longer | 1.4-2.8 GB | < 0.1 GB (+1.4 GB optional) |
| Future lens, closed form, 4 M tokens | 25-40 min | 1 longer, or 2-3 with accumulator checkpoints | 8.5-16 GB of GPU accumulators (the same again on disk if checkpointed) | 0.23-1.9 GB (+8.5-16 GB if checkpointed) |
| Channel series | 5-30 min | 0-2 | 0-4.3 GB | MB-4.3 GB |

Total for (a)-(c): ≈ 1-2 GPU-h.

Not costed here:
- J-lens, J++ and MicroLens++ fitting;
- the λ_RSI vs λ_max power iteration;
- chain-lens rollouts (Qwen3-1.7B has no MTP head);
- the neuron-lens round trip, which needs streaming per-neuron 99.8th WikiText percentiles for 172,032 neurons.

## 5 Colab constraints learned in round 2 and recommendations

**What the repo records:**
- **Two accounts.** Hub A runs r2_q1, r2_q2, r2_q3, r2_pc and r2_ana; B runs r2_t1-r2_t4 (`build_runde2_gpu_notebooks.py:65-66`; `NOTES_w9.md:66-81`).
  - B shares dsse_runs via a shortcut.
  - The "Zweiter Ordner" cell copies a folder when it is missing in dsse_runs and present in the second folder. A round-2 (r2_*) folder is copied only when that session is finished there or declared not run (224-270).
  - FUSE shortcut and rename were untested at build time (`NOTES_w9.md:315-317`), but round 2 completed.
- **A100.** The A100 gate is strict (`r2_run_session.py:786-800`; notebook l.206). Round 1 allowed other GPUs (`build_abschnitt11_gpu_notebooks.py:70-71`) and gpu1 got an L4.
- **Time guard.** 20 min, checked between steps; no guard stops occurred.
- **Overhead.** r2_ana needed 3 launches (two only waited for r2_pc) and spent about 648 s of runner time outside its steps. How much of that was Drive I/O is not recorded.
- **Self-test.** The notebooks advise a first run with `NUR_SELBSTTEST = True`, because the MOCK self-test eats into the 20-min clock (167-171).

**Not in the repo:** "too many sessions" and Drive mount errors on re-run. These are relayed from the task text and cannot be verified here. The relevant code risk:
- the mount cell catches every exception, prints a one-line note and continues with `RUNS = DRIVE = None` (`build_runde2_gpu_notebooks.py:209-216`);
- a failed re-mount therefore skips the resume copy (219-221) and all syncing, without stopping the run (the runner gets no Drive argument, l.373);
- the final cell unmounts Drive (403), so a re-run in the same runtime always re-mounts.

**Recommendations for 1b:**
1. **Pilot first.** Run a 10-min long-context pilot in account B. It should measure:
   - prefill time at 4k/16k/32k;
   - decode ms/step for B ∈ {1, 2, 4} at 4k and 16k and B ∈ {1, 2, 3} at 32k (B = 4 at 32k only as an out-of-memory probe);
   - DynamicCache against a cache that does not rewrite itself;
   - peak memory and the SDPA backend assertion;
   - cached-vs-recomputed parity;
   - Drive read/write throughput, host RAM and local disk;
   - H-neuron yield on 500 questions and lens throughput on 100k tokens.

   Freeze the 1b cost table on these numbers. The plan must be frozen before any 1b data exist, so keep the pilot to engineering quantities, or declare the 500 pilot questions and their yield excluded from every 1b analysis.
2. **Step and session sizes.** Steps of ≤ 3-5 min, resumable via `parts/`.
   - The round-1b design rules allow sessions up to about 5 h with a time guard.
   - The 20-min target is a round-2 plan choice (plan:1402-1403, 1445), not a recorded Colab limit.
   - Under a 20-min target, plan ≤ 10-12 min of work per session (setup + 1.5 × work ≤ 18-20 min).
   - Longer sessions cut setup overhead and keep accumulators in one process.
3. **Drive I/O.** Until measured, budget conservatively about 10 min of copy-in per GB.
   - This treats r2_ana's ≈ 648 s of non-step time as copy-in and hashing of the planned ≈ 0.92 GB, which the repo cannot confirm.
   - Copy only what is read; keep arrays > 0.5 GB in `local/` (not synced, l.265).
   - Sync reduced products only.
4. **Account split.**
   - Account B for input-free sessions (pilot, long context, H-neurons, lenses).
   - Account A for sessions that read R1/R2 states (loops at blocks 14/18/6, K chain) and for the analysis.
   - Assume ≤ 2 concurrent GPU sessions per account (assumption; the cap is not recorded) and stagger launches.
5. **Notebook hardening.**
   - Call `drive.mount(..., force_remount=True)`, assert `os.path.ismount`, and write and read a sentinel file.
   - Abort a real run without Drive instead of running local-only.
   - Keep the strict sm80 A100 gate (40 or 80 GB) and design memory for 40 GB.
   - Record free GPU memory, the SDPA backend and versions in the manifest.
6. **Draft packing (planned work, est.):**

| Session | Account | Content | Planned work |
|---|---|---|---|
| pilot | B | as in item 1 | ≈ 10 min |
| loop14 | A | round-1-identical arm, damage-normalised loops, norm-preserving control, dose series 0.25-2 × α in both signs (about 44 evaluations × 11-12 s) | ≈ 9-10 min |
| loop18, loop6 | A | same, fewer loops | ≈ 5-6 min each |
| followK × 2 | A | K chain, 4 cells × 10 turns with re-prefill | ≈ 8 min each |
| long4k16k, long32k | B | long-context arms | ≈ 10 min each, pilot-dependent |
| hneu_screen × 2, hneu_fit | B | H-neuron screening and fitting (C selection included) | ≈ 10-12 min each |
| lens × 1, or × 2-3 with accumulator checkpoints on Drive | B | 4 M tokens; solve at the end | ≈ 25-40 min in one session |
| ana | A | analysis | copy-in ≥ 10 min plus analysis |

Total: ≈ 13-16 sessions, ≈ 3-4 GPU-h including setup.
- This covers only the items in this report (see 4(d)). It is not the round-1b total; the design rules give an envelope of about 30 A100-h.
- No proposal here yet states a minimum detectable effect; the design rules require one for every measurement.


## proposals (from the research stage; check against the verifier's notes above)
```json
[
 {
  "name": "Long-context pilot session",
  "what": "One 10-min session in account B that measures: prefill time at 4k/16k/32k (B=1, no mask); decode ms/step at B in {1,2,4} with separate caches; peak memory (torch.cuda.max_memory_allocated); the SDPA backend (prefill under sdpa_kernel([EFFICIENT_ATTENTION]) so a math fallback raises); cached-vs-recomputed logit parity with the Gumbel flip count. In the same session: H-neuron screening yield on 500 TriviaQA questions and lens throughput on 100k tokens.",
  "why": "No long-context rate exists in the repo. Every 4k-32k number here is an estimate, and B=1 decode-kernel occupancy is a known risk.",
  "cost": "~10 min work + 4 min setup",
  "reuse": "rk.Gumbel/keyseed (rk_loop.py:122-126), CbarHook pattern (r2_rk.py:603-643), DynamicCache.batch_repeat_interleave/crop (transformers cache_utils.py:163-192), runner Step table"
 },
 {
  "name": "Centred null and paired sign test",
  "what": "Build the 2,000 sign-flip nulls from centred per-prompt differences, i.e. E @ (diff - mean) / P instead of E @ diff / P. Add a prompt-clustered paired sign test on delta_j.u over the loop pairs.",
  "why": "Uncentred flips keep part of the true axis, so p is conservative and PC_PASS narrowly failed (runde1b_eingaben README section 1).",
  "cost": "analysis only, seconds",
  "reuse": "rk_loop.axes (208-236), r2_rk.axis_from_texts (455-481), stats_on_axis (546-566), prompt_bootstrap (570-578)"
 },
 {
  "name": "Damage-normalised loop driver with a norm-preserving control",
  "what": "Before each loop round, calibrate alpha_n so that the damage of the round's push equals the target, using calibrate with RefKL.push; record alpha_n and the top-1 change rate. Add a control hook that rescales the pushed state back to its pre-push norm. Run the dose series 0.25-2 x alpha in both signs via evaluate_variant. Keep one arm exactly as round 1.",
  "why": "Fixed alpha_b drifted to 0.20-0.32 nats at block 14 (theta_verdict.json blocks.14.B_Q0.evaluations.RSI_a). A 0.05-nat push sits at ~0.34-0.36 alpha_b, inside the <0.5 alpha_0 window where lambda_lin leaves G~1 open.",
  "cost": "+~1 s per 128-pair round (<= 6 RefKL evaluations x 8 forwards at 20 ms); block 14 programme ~9-10 min",
  "reuse": "r2_unlearn_qwen.run_loop_variant (990-1001), calibrate (223-233), RefKL (408-439), ConstPush (442-466), r2_rk.evaluate_variant (759-812)"
 },
 {
  "name": "K chain over n follow-up answers",
  "what": "Generalise stage_turn2 to n turns: ids_n = PIDS + t1 + SUF + ...; turn-indexed keys; K_n and its product; R6 per turn. Use two read conventions (context-free as B-Q8, and in-context via generation-time capture) and two push arms (turn 1 only vs every turn). Decide the history-think-block convention in the plan.",
  "why": "Round 2 measured one follow-up, along the style direction only. The RSI-axis K(T 1.0) = 1.053 [0.743, 1.538] cannot exclude self-sustainment; a multi-turn chain is the direct test.",
  "cost": "4 cells x 10 turns with re-prefill ~15-17 min (2 sessions); prompts grow to ~630 tokens at turn 10",
  "reuse": "r2_theta.stage_turn2 (695-739), r2_stimuli.turn2_ids/check_turn2/round_trip_stable (208-234), r2_theta_verdict.turn2_block/rule_R6 (236-255, 565-580), Pairs (r2_theta.py:123-147)"
 },
 {
  "name": "KV-prefix reuse and shared-prefix branching",
  "what": "Prefill each long context once at B=1, once per push condition, with no padding. If B x KV fits (32k: B<=3 on 40 GB), branch with batch_repeat_interleave and generate from past_key_values. Otherwise register a shared-prefix attention (prefix attended once with batched branch queries, suffix causal, merged by log-sum-exp, as in Hydragen) through AttentionInterface with a custom cache. Reads run the continuation as a chunk against the unsteered prefix cache, then crop(-L).",
  "why": "Per-row caches cost 7.52 GB at 32k fp32, and re-prefilling each read costs 12-21 s at 32k.",
  "cost": "prefill est. 1 / 5-7 / 12-21 s at 4k/16k/32k; shared-prefix decode est. 1,000-1,800 tok/s aggregate at B=32",
  "reuse": "transformers DynamicLayer (cache_utils.py:163-192), AttentionInterface (modeling_utils.py:5151), rk hooks unchanged"
 },
 {
  "name": "Long-context hygiene patches",
  "what": "(1) Pass logits_to_keep=positions in damage, RefKL, logodds and damage_T, or apply lm_head on positions as read_full does. (2) Slice the continuation on the GPU before .double().cpu(). (3) Use B=1 or equal-length batches. (4) No output_hidden_states in production. (5) Use gradient checkpointing with re-entrant-safe hooks, or finite differences, beyond ~4-8k.",
  "why": "At 32k, all-position logits take 19.9 GB fp32, a padding mask adds a 4.29 GB/row bias, output_hidden_states adds 7.8 GB, and an un-checkpointed backward needs ~100 GB.",
  "cost": "code only",
  "reuse": "r2_theta.read_full (424), r2_rgi_geom.equal_length_batches (252-258)"
 },
 {
  "name": "Generation-time state capture",
  "what": "Extend CbarHook to store block-b (or selected-block) states of each decoded token per row, giving in-context reads without a second forward; note that the last sampled token has no state.",
  "why": "It avoids re-prefilling long contexts for reads and supports the in-context K convention.",
  "cost": "~320 KB per 40-token row per block in fp32; negligible compute",
  "reuse": "r2_rk.CbarHook (603-643)"
 },
 {
  "name": "Neuron hook manager, CETT and GPU L1-logistic",
  "what": "Write a WriteOrth-style context manager with pre-hooks on all 28 mlp.down_proj that captures a, computes CETT = |a_j| ||W_down[:,j]|| / ||MLP out|| reduced over answer and other spans, and applies per-neuron scaling interventions. Add a GPU FISTA L1-logistic with a C grid, stability selection and label-permutation null. Draw damage-matched random neuron sets via RefKL.hooks.",
  "why": "Replicating H-neurons (arXiv 2512.01797) in Qwen3-1.7B (172,032 neurons; the paper's ratios imply 1.7-60) needs these pieces, and none are in round 2.",
  "cost": "~0.5-1 GPU-h, 3 sessions; 1.4-2.8 GB local, <0.1 GB to Drive",
  "reuse": "neu_extract.py:16-21, 39; neu_sparsity.py:33-50; r2_rk.WriteOrth (695-712); r2_linearity.auc_cols/sign_rows (161-175, 291-293)"
 },
 {
  "name": "Streaming closed-form future and past lenses with channel statistics",
  "what": "Stream Taps over 4 M generic tokens into float64 X_b^T X_b and X_b^T Y_k accumulators, with Y_k the final residual at t+k for k = +-1..8. Solve ridge by Cholesky and decode with the model's final norm and unembedding. In the same pass, accumulate lagged covariances of 128-dim PC projections at 8 blocks. Report forward vs backward predictiveness per channel and per position bin, not as a whole-state determinant, which is symmetric for stationary processes.",
  "why": "This is H-CHAIN's 'channel fronts' and forward/backward vector at feasible cost, with a length-artifact control via position bins. SGD probes with a full-vocab loss would take 29 h fp32.",
  "cost": "~0.4-0.7 GPU-h for the future lens (+~10 min for the past lens); 8.5-17 GB GPU accumulators; 1.9 GB fp16 probes",
  "reuse": "r2_rgi_geom.Taps (225-240), stage_o8 accumulation (466-503), r2_linearity.solve_spd (138-143)"
 },
 {
  "name": "Runner and notebook hardening",
  "what": "Mount with force_remount=True, assert os.path.ismount, write and read a sentinel file, and abort real runs without Drive. Keep steps <= 3-5 min with resumable parts and arrays > 0.5 GB in local/. Budget >= 10 min copy-in per GB. Put input-free sessions in account B and stagger launches (<= 2 concurrent per account, an assumption). Keep the strict sm80 A100 gate, design memory for 40 GB, and log free memory, SDPA backend and versions.",
  "why": "The current mount cell silently degrades to local-only (build_runde2_gpu_notebooks.py:209-216). r2_ana spent ~648 s outside steps against a 60 s budget. The time guard acts only between steps.",
  "cost": "code only",
  "reuse": "r2_run_session.py (Step 57-81, sync 378-405, run_steps 1184-1223, finalize 1264-1323), notebook builder"
 },
 {
  "name": "Draft session packing for 1b",
  "what": "pilot (B); loop14, loop18, loop6 (A); followK x2 (A); long4k16k, long32k (B); hneu_screen x2, hneu_fit (B); lens x2-3 (B); ana (A).",
  "why": "It keeps each session <= ~10-12 min of planned work under the 20-min guard, splits by data dependency across accounts, and leaves the cost table to be frozen after the pilot.",
  "cost": "~13-16 sessions, ~3-4 GPU-h including setup",
  "reuse": "runner session tables and the wave structure of round 2"
 }
]
```

## unknowns (from the research stage; check against the verifier's notes above)
```json
[
 "Actual A100 step times of the Qwen sessions r2_q1, r2_q2 and r2_q3: their session_done.json and logs are on Drive only. All Qwen rates in the repo are planned 'given facts'.",
 "Round-1 shard timings (shard_done.json steps) are on Drive only and were not used for the round-2 cost model (plan:1742).",
 "Breakdown of r2_ana's ~648 s non-step runner time (Drive copy-in vs hashing vs git/G3 checks); the logs are on Drive only. Hence the real Drive read/write rate in Colab is unknown.",
 "Throughput of PyTorch's float32 memory-efficient (3xTF32 FastF32) attention kernel on A100 at 4k-32k, and its decode performance at B=1 (low occupancy for q_len=1). Not measured anywhere.",
 "Which SDPA backend round 1/2 actually used: inferred (memory-efficient) from the dispatch rules, not logged.",
 "Colab A100 VM host RAM and local disk size (relevant for CPU-side float64 reads, per-token dumps and accumulator checkpoints); not recorded in the repo.",
 "'Too many sessions' and Drive mount errors on re-run: not documented anywhere in the repository. The exact Colab concurrency cap, error messages and circumstances cannot be verified here.",
 "Qwen3-1.7B quality at 16k-32k contexts, and behaviour between 32,768 and 40,960 positions (config allows them; the documented native length is 32,768).",
 "Qwen3-1.7B's TriviaQA accuracy and the yield of the all-10-consistent filter, hence how many questions must be screened for 1,000 + 1,000 examples.",
 "Whether H-neurons exist in Qwen3-1.7B: Gao et al. studied Mistral, Gemma and Llama only. Also the paper's positive-class definition (prose vs label formula conflict as read through the fetch tool) must be verified.",
 "Whether HF generate with a reused, batch-repeated prefix cache and rk's hooks reproduces logits within fp32 noise of a full recompute (needs a code test), and how often Gumbel near-ties flip under cache reuse.",
 "Behaviour of gradient checkpointing with rk's hooks (the Stop exception and push hooks re-run during recompute); untested.",
 "The size and licence terms of the generic text corpus and TriviaQA downloads for Colab (not checked); the repo holds well under 1 M tokens of generic text.",
 "Round-1 raw outputs (R1 states needed by continuity arms) live only on Google Drive and could not be inspected here."
]
```
