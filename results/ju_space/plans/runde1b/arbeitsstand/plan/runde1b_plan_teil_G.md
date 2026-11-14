# Part G (round 1b): Gemma replication of the K/L core, native MTP calibration, and the 12B bridge

## G.0 Question, scope and implementation status

**Question.** Do K's H-neuron and chain-lens findings and L's lens-free findings replicate in Gemma, and how does a self-rollout chain lens compare with Gemma's native assistant? The estimands, nulls, statistical families, equivalence margins and verdict strings are those of the current K/L plans. A closed gate is not evidence against H-CHAIN.

**Approved decisions (14 November 2026).** Implement all model-independent K/L **core** measurements for the original E2B/E4B models, forward-only measurements for 31B-IT-QAT, the 12B-IT-QAT Neuron Lens bridge, and the D1 local/global-attention contrast. Fit model-specific E2B/E4B J-lenses. Completeness takes precedence over the earlier rough 4–8 A100-hour estimate. A100-80GB is allowed for memory-intensive sessions. No paid GPU experiment is started as part of the implementation. K's RK-1/RK-2/K3h and L's X1–X6 are not additional G experiments; P and R are not implemented here.

**Delivery boundary.** Only new G files on `codex-teil-g`; never push to `ccr-6fe02b21-est040`. Round-1/2 files and the existing core, contracts, runners, R/K/D/L modules and notebook generator remain byte-identical. Required shared integration changes are listed in `analysis/ju_space/round1b/G_MERGE_NOTES.md`. The handover's two-phase rule is binding: phase 1 is the model audit, this plan and the adapter with CPU mocks; phase 2 (K/L measurements, integrated mock tests and notebooks) begins only after the handover's **Stand** section records **“Integrationstest bestanden”**. This document specifies the complete intended G arm; it is not evidence that phase 2 or real model runs have completed.

**How to read this spec.** K and L are incorporated by measurement identifier below, including their implementation amendments. Each mapping states its input substitutions and restrictions; no unmentioned sample-size trim, changed statistic or replacement verdict is permitted. The G freeze records the exact K/L/D plan hashes used. Model-dependent fit parameters are recomputed on Gemma; no Qwen-derived weights, neuron sets or axes are imported as Gemma quantities.

## G.1 Checked model identities, licences and loader evidence

The model/config/card audit is reproducible through `r1b_g_models.py`. Its `MODEL_SPECS` records the complete immutable revisions, SHA-256 of each loadable file, separate card SHA-256, source URLs, file sizes and checksum evidence. Weight and large tokenizer checksums were read from HF LFS metadata; **no model weights were downloaded or run** for this audit. All eight cards currently declare Apache-2.0 and all eight repositories are currently ungated. This is an observation at the recorded revisions, not an assumption that future access terms cannot change.

| G alias | HF ID under `google/` | Revision prefix | Architecture | Layers × hidden; MLP width | Role |
|---|---|---|---|---|---|
| `gemma4-e2b` | `gemma-4-E2B-it` | `4adf15748de9` | `gemma4` | 35 × 1536; 6144 in layers 0–14, 12288 in 15–34 | float32 target |
| `gemma4-e4b` | `gemma-4-E4B-it` | `4ecc2ba63bca` | `gemma4` | 42 × 2560; 10240 | float32 target |
| `gemma4-31b-qat` | `gemma-4-31B-it-qat-w4a16-ct` | `52f3f65bc7a0` | `gemma4` | 60 × 5376; 21504 | QAT, forward only |
| `gemma4-12b-qat` | `gemma-4-12B-it-qat-w4a16-ct` | `1d2c2d7f2466` | `gemma4_unified` | 48 × 3840; 15360 | QAT, forward only; catalogue bridge |
| `gemma4-e2b-qat` | `gemma-4-E2B-it-qat-w4a16-ct` | `971342c08f60` | `gemma4` | E2B layout | audited compatibility variant |
| `gemma4-e4b-qat` | `gemma-4-E4B-it-qat-w4a16-ct` | `6cd26aaa2357` | `gemma4` | E4B layout | audited compatibility variant |
| `gemma4-e2b-assistant` | `gemma-4-E2B-it-assistant` | `2d874ef7d29f` | `gemma4_assistant` | 4 × 256; 2048 | pairs only with original E2B |
| `gemma4-e4b-assistant` | `gemma-4-E4B-it-assistant` | `8d0031ea8c21` | `gemma4_assistant` | 4 × 256; 2048 | pairs only with original E4B |

E2B/E4B local attention has a 512-token window and 131072-token maximum context; 12B/31B uses 1024 and 262144 respectively. Exact `layer_types` and KV-sharing counts come from the pinned config; a single global sliding-window flag is insufficient. The layer count is the text decoder count, not vision/audio layers.

**Memory correction.** Original E2B/E4B packages contain 5,123,178,051 and 7,996,156,490 parameters respectively (including multimodal towers and per-layer embeddings), corresponding to about 20.49/31.98 GB for all fp32 weights before activations. The 31B-QAT weight file is 23,265,352,448 bytes; 12B-QAT is 10,264,229,896 bytes. File size is neither the decompressed parameter footprint nor a runtime peak-memory measurement. Text-only loading may omit unused modality towers but must retain per-layer embeddings.

**Compressed-tensors decision.** PyTorch 2.10.0, Transformers 5.10.1, compressed-tensors 0.15.0 and Accelerate 1.12.0 are the initial compatibility target (CT requires PyTorch <2.11), with explicit `run_compressed=False` for QAT and fp16 reconstructed quantized weights. Accelerate provides the explicit device placement in the loader. The ordinary `run_compressed=True` path is not accepted as proof of 4-bit-resident inference: that CT version removed `CompressedLinear.from_linear`; older wrappers may retain decompressed weights after the first forward. G does not implement a custom int4 kernel or promise 31B on A100-40GB. Tiny synthetic quantization/decompression parity must pass before a real loader is called. Runtime dtype, loader policy, peak allocated/reserved memory, GPU identity and library versions are recorded. QAT remains a forward-only arm even after decompression; never relabel it as an original fp16 model or allow gradient measurements on it.

**Hook and output semantics.** Read the full decoder block output, including per-layer-embedding contributions and layer scaling. Read/intervene on `mlp.down_proj` **input**, `act(gate) * up`, not the individual gate/up projections. Use native Gemma final norm and final logit softcap in every unembedding. Avoid hand-built decoder calls that omit native masks, RoPE or shared KV state. All-position captures must request all needed positions explicitly: `logits_to_keep=1` can truncate states in KV-sharing layers and is reserved for genuinely last-token-only calls.

**Primary evidence.** Model cards/configs are linked at immutable revisions in the Python manifest. Architecture and quantization source references: [Gemma4](https://github.com/huggingface/transformers/blob/v5.10.1/src/transformers/models/gemma4/modeling_gemma4.py), [Gemma4 Unified](https://github.com/huggingface/transformers/blob/v5.10.1/src/transformers/models/gemma4_unified/modeling_gemma4_unified.py), [assistant](https://github.com/huggingface/transformers/blob/v5.10.1/src/transformers/models/gemma4_assistant/modeling_gemma4_assistant.py), [CT quantizer](https://github.com/huggingface/transformers/blob/v5.10.1/src/transformers/quantizers/quantizer_compressed_tensors.py). Licence link supplied by all eight cards: <https://ai.google.dev/gemma/docs/gemma_4_license>.

## G.2 Shared objects, coordinates and statistical conventions

- **G-SO-HN, G-SO-LAB, G-SO-CHH, G-SO-LENSOUT:** the same scientific definitions as K/L, separately produced for each Gemma target. Question IDs, raw text, gold aliases and splits are reusable; token IDs, span boundaries, H sets, detector coefficients, REF-W moments and cached hidden states are not. Retokenize every text with its pinned target tokenizer.
- **Neuron coordinates:** `(layer, neuron)` is authoritative. Flatten via prefix sums of the actual layer widths; total MLP neuron counts are 337920 (E2B), 430080 (E4B), 737280 (12B), 1290240 (31B). No `layer * first_layer_width` assumption. Random sets match actual per-layer counts; `|H|/N` uses the target's `N`.
- **Block coordinates:** map a Qwen reference block `b` to nearest relative depth `floor(b*(L-1)/27 + 0.5)`; the final target is `L-1`. The mapping is recorded before the freeze and applied consistently to every dependent arm. Native local/global layer type is an additional descriptive covariate, not overwritten to imitate Qwen. Degenerate mapped comparisons are rejected rather than duplicated.
- **G-SO-DMG:** unchanged KL functional, delta1=0.05/delta2=0.15 nats, calibration and out-of-band labels. Construct Gemma-tokenized reference sets from source texts; preserve the reference text IDs and hashes. Never feed Qwen IDs into Gemma.
- **CETT:** preserve K's definition `abs(z_j)*norm(W_down[:,j])/norm(W_down*z)`. Native post-FFN norm/scaling belongs to the actual forward intervention; any additional effective write-vector annotation is labelled separately and does not silently redefine CETT.
- **Statistics:** unchanged K.4/L.2 bootstrap units, seeds, 90% intervals, p-values, Holm/BH families, equivalence margins and gates. Apply families separately per model; the cross-model comparison is descriptive and is not inserted into the frozen round-1b confirmatory families. Preserve exact inherited verdict strings with separate model/measurement metadata.
- **Token hygiene:** inspect Gemma's actual digit tokens, neutral seed tokenizations and chat prefixes before enabling arithmetic/lens arms. A Qwen tokenization fact is not evidence for Gemma. Use measured span boundaries and native stop tokens; reject malformed examples with the inherited parsing codes.
- **Provenance:** every G artifact records model/revision/file hashes, dtype/decompression policy, prompt and input hashes, layer map, width offsets, code/plan hashes, lens identity, keys, versions, runtime and gate outcomes. Coded `NA`/`CENSORED` values replace non-finite numbers.

## G.3 Measurements and complete core mapping

### G0 Engineering pilot and known-answer calibration

**What / how.** Per target, benchmark load/peak memory, teacher forcing, generation with hooks, cache continuation, down-projection capture, selector throughput and the planned long-context sizes. E2B/E4B also measure J-lens blocks and assistant calls. Complete L-CAL on CPU and preserve its branch-specific gates. Use neutral pilot material only, separate from analysed items.

**Rule / null-controls.** Engineering only. Exact shapes, hook neutrality and cache parity precede timing; timing never excuses a failed semantic check. CPU mocks establish code behaviour, not real-GPU throughput or peak memory. A real GPU run with insufficient memory fails explicitly; do not change dtype, cut contexts or substitute a smaller model silently.

**Inputs / runs / gates.** Pinned manifests and pilot text, `r1b_G_P0_<model>`; G-ID, G-ARCH, G-QAT, G-HOOK, G-CACHE and inherited calibration gates. Results determine the final A100-minute projection and batching before freezing.

### GJ Model-specific E2B/E4B forward-Jacobian lenses

**What.** A model-specific lens supplies K4's seeds and K7's Jacobian workspace; it is not an inverse reconstruction or a trained generator.

**How.** Fit on the first 1000 eligible, distinct WikiText-103 **train** passages after the frozen normalization, tokenized per target and truncated to 128 tokens. Source positions are 16–127; targets are all positions of the last residual output before final norm. For each source layer, estimate the mean-over-source derivative of the sum of target states for a shared infinitesimal residual perturbation at all source positions. Compute its columns in resumable derivative blocks with float32 accumulation; use model-native forward semantics. Store maps `J[l]: output(l) -> output(L-1)`, with final-layer identity, source/target conventions and corpus hashes. No matrix inverse, pseudo-inverse or Qwen matrix is involved.

**Rule / controls.** K4j's held-out bridge uses 64 WikiText-103 **validation** windows of 128 tokens, eight random unit vectors (seed 1305), the four mapped blocks and both input/output conventions; median cosine >=0.9 opens G-K-J. Cross-check derivatives with central differences; verify final-layer identity and split-half reliability. K4's positive controls and G-K7 task-seed check still apply; passing the fitting bridge does not establish lens faithfulness on arithmetic.

**Runs / gates.** `r1b_G_J_<model>` then the K preparation session, originals only. QAT measurements do not call autograd. Their forward-only lens controls use logit-lens and neutral seeds, with inherited J-dependent conclusions explicitly `not testable (no validated target J-lens)`; the 12B base-model R-lens is an external transfer instrument, never an automatically valid target J-lens.

### GK H sets, chain lenses and the model's own text

The following table is the exhaustive K-core disposition. “Transfer” means retain the named protocol, sizes, controls and verdicts, with only G.2's model substitutions. Gates still apply; it does not mean every real dataset necessarily yields a decidable verdict.

| K measurement | Original E2B/E4B | 12B/31B QAT |
|---|---|---|
| K0, K1 | Engineering pilot; same raw label-bank items, target-specific tokenization | Same |
| K2 | New H_tq/H_ar/H+ selection, bootstrap stability, matched random sets, REF-W, damage calibration | Same, all model evaluation forward-only; auxiliary logistic fitting is allowed |
| K3a–d, K3g | Transfer detection, construct controls and activation interventions | Same |
| K3e | Transfer H write-vector readouts with target lens and native forward verification | Forward readouts under explicitly named available lens; no claim of validated target J-lens |
| K3f | Not transferable: the specified style direction is a Qwen round-1/2 SO-AX object | Same exclusion |
| K4j | New target lens bridge above | Unavailable for a target J-lens; 12B external-lens validation is GB below |
| K4, K4p | Transfer all core self-rollout/positive-control arms | Forward-compatible arms; logit/neutral-seed instrument named separately; J-dependent arms coded unavailable |
| K5a | Native Gemma assistant calibration below | Not run: no assistant is assigned to these QAT targets |
| K6a–h | Transfer break-off, H association, causal scaling, persistence, forcing/context-offset controls and QA lens arm | Forward-compatible arms; retain lens-validity labels and dependencies |
| K7a | Transfer SO-AX-free atlas: target H, Jacobian workspace, PCs, native heads | Forward observables; Jacobian-dependent channels unavailable |
| K7b | Transfer H/J/PC channel arms; omit Qwen SO-AX channels explicitly | H/PC forward arms; unavailable channels explicitly labelled |
| K8a–c | Transfer matched-length/content, persistence/bistability and first-error hazard | Same forward measurements |
| K9 | Only model-independent H/J/PC edge measurements, exploratory; original complete hierarchy verdict unavailable without specified SO-AX channels | H/PC forward edge measurements; same complete-verdict exclusion |
| K10a | H and target-derived subspace route splits; SO-AX-source aggregate verdict unavailable | Forward-compatible H/PC route splits; same aggregate exclusion |
| K10b | Transfer finite-difference QK/OV split plus autograd checks/head map on native attention | Forward finite-difference observables only; no VJP/JVP/head-gradient map or gradient-dependent verdict |
| K10c | Transfer chained versus direct readouts, after arithmetic and J-lens gates | Forward readouts; J-dependent comparison explicitly gated |
| K analysis | Same rule functions on G outputs; no invented replacements for unavailable premises | Same |

**Native MTP details.** Original E2B/E4B each uses its own original assistant. The assistant receives `concat(target_embedding(last_token), target_last_hidden_state)` and the target's `shared_kv_states`, requested through its supported API; use returned projected hidden state at each further draft step. It is not an independent LM and not the Qwen MTP tensor layout. Preserve K5a's item-disjoint calibration/analysis split, paired states/seeds, agreement/length curves, self-consistency `KL(q || p)` and calibration verdict. Confirm the one-token position alignment before rollout. Reject mismatched target/assistant architectures, hidden dimensions or vocabulary mappings. QAT targets cannot borrow the original assistant merely because names resemble each other.

### GL Lens-free core

| L measurement | G disposition |
|---|---|
| L0, L-CAL | Pilot and known-answer calibration above |
| L1 LOCUS | Full probe/fragment, permutation, reselection, erasure, planted-code and control protocol on target CETT; ragged coordinates |
| L2a–c HORIZON | Full long-context, length/information and cross-map protocol using native local/global masks and verified caches |
| L3a–d ATTRACT | Full observational, causal twin, Ising, energy and irreversibility measurements; the optional Qwen `u*_14` descriptive line is unavailable |
| L4 DUET | Full core dyad/control/gain protocol with native Gemma chat turns; no X2/X6 extension added |
| L5a–c OWNER | Full exact-label ownership and own-error protocol; lens-dependent branches inherit the actual target lens/MTP gates |
| L-ANA | Same per-branch rules and combination logic; missing premises remain explicit |

All GL procedures use target-specific H sets and populations. Statistical logistic/Ising/forward-map fitting is not differentiation through the QAT target and remains allowed. Interventions are forward activation edits. No target weight training is added. L1's fallback selector and the inherited H-specific `not testable` behaviour remain unchanged when SG8 fails.

### GD Native local versus global attention (D1 contrast)

**What / how.** Run D1's model-independent field/readout protocol on the same raw stimulus identities, target-tokenized, with reads around each target's window boundary. Retain native `layer_types`, RoPE, masking and shared KV. Record per-layer effects split by native local/global type and by cue–read distance relative to the window; retain D1's controls and within-item pairing. Compare full and chunked/cache paths before accepting long-context records.

**Rule / shows-cannot.** D1's inherited field verdicts apply to their unchanged estimands. The added layer-type contrast is descriptive: layer type is confounded with depth and is not randomly assigned. Do not infer a causal benefit of global attention from cross-layer means or silently force all layers to global attention.

**Inputs / runs / gates.** Frozen D1 raw stimuli, target tokens, GD session per target; G-CACHE and inherited D1 gates. This is a G-owned reuse of D1, not a modification of part D.

### GB 12B Neuron Lens bridge

**What.** Link G's 12B-IT-QAT H/control neurons to external baseline annotations and test whether descriptions transfer across instruction tuning and quantization.

**Sources.** [stdlogits/neuron-lens](https://huggingface.co/datasets/stdlogits/neuron-lens/tree/21666b9f217328caee73d10de85ca19f4c174e72), dataset revision `21666b9f217328caee73d10de85ca19f4c174e72`, and [stdlogits/rs-lenses](https://huggingface.co/stdlogits/rs-lenses/tree/ee42b62072dd04f375c110990678fc5176c3ed4c), revision `ee42b62072dd04f375c110990678fc5176c3ed4c`, both CC-BY-4.0. Their stated model is **google/gemma-4-12B (base)**, not IT or IT-QAT. The catalogue does not pin the exact generating base-model revision; record that uncertainty rather than substituting today's base revision.

**How.** Join on 0-based `(layer, neuron, pole)` with pole in {-1,+1}, 48×15360 coordinates. Confirm dimensions, layer semantics and token-vocabulary mapping. Fetch only required layer tables; preserve attribution/source links and corpus-origin licences. Use H and layer/count-matched controls selected independently of catalogue labels. Evaluate annotations on frozen held-out probe texts outside label-generation contexts; report activation enrichment and forward-intervention agreement per pole against matched controls. Keep discovery/probe construction and the held-out evaluation disjoint. Record separate dimension, tokenization, activation and intervention checks; no combined “same model” flag.

**Rule / shows-cannot.** Until held-out evidence is available, annotations always read `external base-model annotation; transfer unvalidated`. Report transfer effect sizes and intervals descriptively. A coordinate match alone does not validate a semantic label, and a failed transfer is not evidence that the original catalogue is incorrect. External R-lens reads include Gemma final norm and softcap; write-vector annotations account for native post-FFN norm and layer scaling. Never substitute the catalogue's R-lens for GJ without an explicit target validation result.

**Inputs / runs / gates.** G 12B H/control set and pinned catalogue metadata/tables, `r1b_G_BRIDGE_12B`; G-ID, shape/vocabulary checks and source attribution. Real catalogue readouts are not fabricated in mocks.

## G.4 Sessions, costs and freeze

Session IDs and artifact roots include the model alias; cross-model caches and outputs cannot collide. Names below describe the phase-2 contract to be registered in the G-owned entrypoint, not registrations in the unchanged base runner.

| Session family | Starts after | Work | A100-minutes before measured pilot |
|---|---|---|---|
| `r1b_G_P0_<model>` | G model audit and loader checks | G0 engineering only | provisional 20 per target; actual recorded |
| `r1b_G_J_<original>` | pilot and fixed fitting corpus | GJ derivative chunks/held-out bridge | `1000 * measured_seconds_per_fit_window / 60`, plus measured validation/setup |
| `r1b_G_K_PREP_<model>` | pilot, required lens files | K1/K2/K3, references, H sets and gates | scaled from inherited K per-measurement work table |
| `r1b_G_MTP_<original>` | pilot and matching assistant | K5a calibration | scaled from measured native-draft rates |
| `r1b_G_K_<model>` | preparation and applicable calibration | K core in independently resumable groups | scaled per measurement, not a single guessed model multiplier |
| `r1b_G_L_A/B_<model>` | own H/lab/population and K lens outputs | L core | inherited work baseline 185.3 min per target before rate scaling/margin |
| `r1b_G_D1_<model>` | target cache gate and D1 inputs | GD contrast | measured D1 unit rate × frozen item/length counts |
| `r1b_G_BRIDGE_12B` | own 12B H/control sets | GB join and transfer probes | measured probe rate × frozen held-out counts |
| `r1b_G_ANA` | required sessions finished or explicitly not run | per-model verdicts and descriptive comparison | CPU; no GPU required |

Use `setup_minutes + 1.5 * sum(unit_count * measured_seconds_per_unit / 60)` for each frozen budget row. Unit counts and inherited scientific sample sizes are fixed independently of effect estimates. The existing Qwen K/L core alone is about 818 planned A100-minutes per model, hence about 1636 minutes for two models **before** Gemma rate scaling, new lenses, QAT targets, D1 or the bridge. That comparison invalidates treating 240–480 minutes as a full-arm budget promise; it is not itself a measured Gemma forecast. Freeze final A100-minute costs after G0. No automatic scientific trims are approved; smaller batches/chunks preserve sample counts.

GPU work uses one target per process, with assistants only beside their matching original. Chunk long prefill and derivative work; keep only required logits/states. Preserve the shared 4.5-hour guard and <=5-minute resumable work pieces; continuation must not duplicate completed items. Arrays above the shared large-artifact threshold remain local with resumable manifest entries. The notebooks show GPU/RAM requirements and the frozen projection before real execution. Tokens, if later required, come from runtime secrets and are never persisted.

## G.5 Gates and failure behaviour

| Gate | Check | Failure |
|---|---|---|
| G-UPSTREAM | affirmative integration pass in the handover Stand section and matching integration artifacts | phase 2 cannot be claimed implemented/integrated against unfinished K/L |
| G-ID | exact model revision and every required file SHA-256 | stop; no fallback to main or another model |
| G-ARCH | actual block/MLP layout, native text-only semantics, output norms | stop affected target |
| G-QAT | synthetic CT decompression parity and no gradient target path | stop QAT; never switch to original weights |
| G-HOOK | neutral-hook equality, actual intervention, complete cleanup | stop target measurements |
| G-CACHE | full versus cached/chunked logits/states under native local/global attention and KV-sharing | stop dependent long-context/route arm |
| G-MTP | exact original pair, dimensions/vocabulary and token-position alignment | native calibration not run; inherited G-K5 undecidable |
| G-K-J | target lens identity and K4j bridge | inherited J-lens failure labels; no unvalidated comparison claim |
| G-MEM | pilot peak and selected batch/context fit actual GPU | explicit memory failure; no silent dtype/context changes |
| G-GB | catalogue identity, coordinates, attribution and separated transfer checks | descriptive external annotation only or explicit missing-source code |
| inherited K/L/D gates | same definitions and thresholds as their plans | same branch-specific verdict/NA handling |

## G.6 Implementation acceptance and provenance

Phase 1 must provide an offline-importable model manifest, metadata-only refresh/check command, loader registration, native architecture adapter and small deterministic CPU mocks. Audit output is written outside the repository; refreshing metadata never rewrites frozen source pins automatically. Tests must verify ragged neuron indices, native Gemma hook positions, chat/stop handling, output normalization/softcap, positive and neutral interventions, cleanup, cache parity and assistant-pair rejection. Synthetic CT fixtures test reconstructed weights without downloading real weights. CPU success is always labelled as CPU/mock success.

Phase 2 acceptance requires all mapped G steps with explicit gate results, no placeholder success for absent scientific results; parity of reused statistics/verdicts on shared synthetic data; cache/derivative/MTP alignment tests; restart/missing-input/provenance checks; and notebook generation into a temporary output directory. Only the permitted new source files, G plan and merge notes are committed. A repository diff against the base must contain no changed/deleted pre-existing file. Real-GPU validation remains a distinct, not-yet-run milestone until actually performed.

All facts above have sources and pinned identities; all planned measurements remain prospective. No automated request to goodfire.com/goodfire.ai, no local real-model run, no secret in files/commits, and no external message or paid run is part of this delivery.
