# Part G: delivery and integration notes

## Branch and scope

Base: `ccr-6fe02b21-est040` at `8d1e923a857d3908d61096758c0cc46439bf6a83`.
Delivery branch: `codex-teil-g`. All changes are new files from the handover's
allowlist. No existing core, contracts, runners, R/K/D/L, round-1 or round-2 file
is changed. Never push this work to the base branch or merge it automatically.

The user confirmed the complete model-independent K/L core, the 12B catalogue
bridge, D1 local/global contrast, and fitting E2B/E4B-specific J-lenses. Completeness
takes precedence over the original unverified 4–8-hour estimate. A100-80GB is
allowed; QAT weights may be explicitly decompressed to fp16 for forward-only
measurements. This does not authorize launching paid GPU runs here.

## Phase 1 delivery

- `r1b_g_models.py`: all eight exact model IDs, immutable revisions, SHA-256
  manifests, architecture/layout/precision metadata, sources and a metadata-only
  refresh command. Weight checksums are HF LFS metadata, not downloaded weights.
- `r1b_g_adapter.py`: explicit core loader registration, pinned local loading,
  separate Gemma4/Unified paths, original MTP assistant pairing, native chat and
  output transforms, ragged neuron indexing, and QAT forward-only guards.
- `r1b_g_mock.py`: entirely offline, tiny random *native Transformers* Gemma4,
  Unified and assistant models, including PLE and KV-sharing; synthetic tokenizer.
- `r1b_g_selftest.py`: numerical CPU checks, including CT pack/decompression
  reference parity, cache parity, hooks, chat boundaries, manifest validation and
  assistant recursion. No mock result is a scientific measurement.
- `r1b_g_checkpoint_test.py`: the production text-only loading path exercised on
  saved tiny random native checkpoints, including compressed-tensors fixtures.
- English G protocol in the working plan directory. Its phase-2 measurement and
  session sections describe intended implementation, not delivered experiments.

Test environment: Python 3.12, torch 2.10.0+cpu, transformers 5.10.1,
compressed-tensors 0.15.0, accelerate 1.12.0. Real-model GPU memory/throughput and licence acceptance
in the eventual operator's account have not been tested. Current model metadata
reports Apache-2.0 and ungated access; no token is committed.

```sh
python analysis/ju_space/round1b/r1b_g_models.py check
MOCK=1 OMP_NUM_THREADS=1 python analysis/ju_space/round1b/r1b_g_adapter.py selftest
MOCK=1 OMP_NUM_THREADS=1 python analysis/ju_space/round1b/r1b_g_selftest.py
MOCK=1 OMP_NUM_THREADS=1 python analysis/ju_space/round1b/r1b_g_checkpoint_test.py
```

Validated on 14 November 2026 with HF network access disabled for the test runs:
22 adapter/native-mock tests and 4 checkpoint tests passed. All eight pinned
metadata re-audits matched; Python compilation, dependency checks and whitespace
checks passed. These results establish phase-1 CPU code behaviour only.

## Phase 2 prerequisite

The handover requires **“Phase 2 (erst wenn hier unter ‘Stand’ ‘Integrationstest
bestanden’ steht)”**. At the base commit, the latest status is 14 November 01:50
MEZ: the integration test is running. Earlier conditional mentions of the phrase
are not a passed test. Do not claim phase 2 is complete or bypass this prerequisite.

Once the upstream owner records a passed integration test, bring that committed
baseline into this branch, resolve only G-side compatibility issues, and record
the exact dependency commit in the G run manifest. Continue the complete G plan;
do not repair or extend the other experiment parts in this branch.

## Required integration work after the prerequisite passes

1. **G entry point and contracts.** Add G-only session/file/key registries and a
   G runner using existing Step, Ctx, output checking and continuation helpers.
   The shared runner currently imports only R/D/K/L/main, so a new session module
   alone cannot be discovered. A later unified-runner merge must add G discovery
   explicitly; this branch must not overwrite shared runner functions at runtime.
2. **Context loading.** Use the G loader from a G context for both mock and real
   models. Core `load_model(mock=True)` bypasses registered loaders and constructs
   a Qwen-shaped model. Core `Ctx.model()` also defaults to Qwen. Registering a real
   loader does not solve either issue. Bind freeze pins before registration.
3. **Architecture-specific collection.** K/L currently assume a constant MLP
   width and often flatten IDs as `layer * width + neuron`, or stack per-layer
   weights. G must use `NeuronLayout` offsets throughout selection, controls,
   CETT, ablations, bootstrap sets and exported artifacts. Reuse statistical
   functions unchanged; implement Gemma collection and coordinate conversion in
   G modules. Do not pad missing neurons into the statistical population.
4. **Tokenization and readouts.** Shared `user_start` and `stop_tokens` contain
   Qwen delimiters; some K/L routines directly multiply a final norm by a head and
   omit Gemma's softcap. Use G's helpers. Re-tokenize source *texts*; never copy
   Qwen token IDs, H sets, residual vectors, damage doses or J-lens matrices.
5. **Lenses and MTP.** Implement the specified E2B/E4B lens fit and held-out K4j
   checks. The assistant is a projected-state/shared-KV decoder, not a standalone
   autoregressive model. Use only the reviewed original-target/original-assistant
   pairs. Recalibrate the K rules on model-specific observations.
6. **Forward-only QAT.** The reviewed runtime uses explicit CT decompression,
   preserving quantized reconstructed values. It does not provide compressed
   int4 CUDA kernels. Forbid VJP/JVP measurements and weight training; forward
   interventions and finite differences remain possible. Validate actual 80GB
   memory before starting scientific sessions; no silent bf16 substitute for
   original fp32 E2B/E4B.
7. **References and external annotations.** Qwen-only SO-AX/R claims are not
   transferable. The 12B Neuron Lens catalogue was produced on the pretrained
   base model, not IT-QAT: coordinate joins are descriptive until held-out transfer
   validation. Preserve source revisions, attribution and that distinction.
8. **Freeze, notebooks, reporting.** Give G an additional freeze record with G
   plan/input/model/lens hashes and measured pilot budgets, without rewriting the
   base freeze. Add the G notebook generator only after the integration gate;
   build notebooks outside the tracked tree. Produce model-separated reports
   with original K/L thresholds and explicit unavailable/not-testable branches.

## Acceptance before claiming all of G complete

The G session registry, K/L measurements, J-lens fitting sessions, D1/12B bridge,
integrated mock run and notebook generator must all be delivered and tested.
Adapter tests alone are not that acceptance. Commit and push each completed work
step on `codex-teil-g`, with truthful author/co-author trailers; do not attribute
new Codex work to the prior Claude session.
