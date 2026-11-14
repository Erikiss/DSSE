# Part G: delivery and integration notes

## Branch and scope

Base: `ccr-6fe02b21-est040` at `8d1e923a857d3908d61096758c0cc46439bf6a83`.
Delivery branch: `codex-teil-g`. G source files follow the handover allowlist; the six new notebooks were
subsequently requested explicitly by the user. No existing core, contracts, runners, R/K/D/L, round-1 or round-2 file
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
- English G protocol in the working plan directory, including the model-specific
  measurement mapping and the user's independent-notebook amendment.

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

## User amendment: independent G notebooks and runs

The user explicitly clarified that G must run independently of Qwen and that its
notebooks should be completed now. This overrides the earlier handover waiting
rule. No upstream “Integrationstest bestanden” marker, Qwen run, cross-account
transfer or base freeze is required by the G runner.

The dedicated entry point is `r1b_g_run.py`, with `run`, `freeze`, `status`,
`report` and `prepare-inputs` commands. Each experiment and model has its own
stage records and artifact hashes. Real scientific execution uses a G-only freeze
backed by a real G pilot; mock runs never qualify as scientific evidence.
The notebook builder emits six committed `runde1b_G_*.ipynb` notebooks with
isolated CPU/CUDA environments and pinned code snapshots. These are explicitly
requested additions beyond the original source-only handover allowlist.

### Notebook operation

Open `runde1b_G_PILOT.ipynb` to run a standalone pilot, or one of the four model
notebooks for its G stages. Default `MODE="mock"` uses random CPU models without
model downloads. `MODE="pilot"` runs the real engineering pilot on A100-80GB;
`MODE="run"` requires the model's own G freeze. `runde1b_G_ANA.ipynb` reads G
artifacts without starting model work. A separate experiment name is required
for real work; mock and real evidence cannot share the same model directory.

`prepare-inputs` reuses committed raw TriviaQA/questions/arithmetic/D1 definitions
and verified raw WikiText data. It never reuses Qwen token arrays, answers, H sets
or directions. The pinned PG19 test corpus is included automatically in original file order;
Gemma-specific book eligibility and token slices are evaluated by the G stage.
Held-out 12B catalogue transfer probes and their provenance manifest are explicit
additional inputs, finalized before freezing. If probe construction needs selected
H coordinates, use a separate G discovery experiment through `k_prepare`, then
freeze disjoint probes in a new validation experiment; run `--stage bridge` there.
No labels or semantic transfer evidence are manufactured by the input builder. Missing data and partial scientific coverage remain visible;
finished notebooks alone do not certify the complete K/L research protocol.

## Standalone implementation

- `r1b_sessions_G.py` and `r1b_g_run.py` define the G dependency graph, isolated
  model contexts, stage checkpoints, exact code/input/model/plan hashes, native
  time guards and verified resume. A stage identity is written before its first
  cache entry. Changed code or inputs require a new experiment. Soft observation
  dependencies let independent L branches proceed when another data gate closes.
- `r1b_g_data.py` builds the raw question/stimulus bank and verifies pinned
  WikiText train/validation/test files plus PG19 test files and original order.
  REF-W, N-text and lens-validation raw pools are disjoint. Real target-token
  quotas are checked after retokenization. No Qwen model or generated answers
  participate in preparation.
- `r1b_g_engine.py` and `r1b_g_l_native.py` implement native Gemma captures,
  forward interventions, cached/complete-prefix decoding, chunked unembedding,
  final norm/softcap and ragged coordinates. `r1b_g_l_erasure.Damage` evaluates
  the unchanged KL functional with bounded GPU head memory.
- `r1b_g_lens.py` contains the engineering pilot, resumable original-model
  J-lens fitting/held-out validation and matching native assistant calibration.
  QAT uses forward-only readouts with explicit unavailable J/gradient branches.
- `r1b_g_k*.py` contain target-specific K selection, detection, causal controls,
  write-vector readouts, chain controls, persistence, own-text/long-form probes,
  channel atlas and compatible route/derivative measurements. The inherited
  verdict functions and G's stated SO-AX exclusions remain authoritative.
- `r1b_g_l*.py` contain L1 population/fragment/erasure work, L2 horizon and
  cross-map work, L3 dynamics, L4 native dialogue conditions and L5 ownership /
  own-error work. L-CAL is a G-owned CPU stage; quick mock calibration never
  opens scientific gates. Each branch records its actual premises independently.
- `r1b_g_d1.py` implements the additional descriptive local/global contrast.
  `r1b_g_bridge.py` records 12B base-catalogue provenance and held-out transfer
  checks. Missing independent probes close the transfer gate; catalogue joins
  alone never validate IT-QAT descriptions.
- `r1b_g_analysis.py` joins fixed-size confirmatory families within each target,
  preserving the PI-specificity primary test and descriptive rank sensitivity.
  `r1b_g_l_power.py` prevents unsupported equivalence conclusions while preserving
  directional findings and an audit copy of the original branch verdicts.
- `notebooks/build_runde1b_gemma_notebooks.py` deterministically builds PILOT,
  E2B, E4B, 31B-QAT, 12B-QAT and ANA. Each uses its own persistent environment
  and immutable source checkout. The ANA report joins only hash-verified G
  artifacts and keeps model identity and mock status visible.

The original shared runner still discovers R/D/K/L/main only. Unified-runner
integration is optional future work; these notebooks invoke the delivered G
entry point directly. No shared file is monkeypatched to imitate Qwen shapes.
Static statistical ports cite their source modules; model and coordinate logic
is implemented in G files.

## Verification and interpretation

Latest verification (14 November 2026): the broad CPU suite completed 154 tests
(151 passed, three opt-in notebook execution tests skipped in that command); the
final statistical-family suite separately passed 12 tests. Native fixtures
exercise QAT forward-only paths, cached interventions, planted statistical
controls, source hashes and interrupted-run recovery. The pinned raw-data builder
was executed and all four target tokenizers verified 14,166,550 training tokens
(above the 10-million-token frequency-control quota), with disjoint reference /
N-text / lens-validation pools and 100 PG19 test books in verified source order.

The separate opt-in notebook suite passed all nine tests, including execution of
all six notebooks, all four complete mock model graphs, ANA rendering, verified
resume and recovery from an interrupted source checkout. These cell-execution
tests use an immutable local Git fixture and the already installed pinned Python
environment; they do not provision Colab or execute real GPU models.

```sh
MOCK=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover \
  -s analysis/ju_space/round1b -p 'r1b_g*test.py' -v
RUN_G_NOTEBOOK_EXEC=1 MOCK=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python analysis/ju_space/round1b/r1b_g_notebook_test.py
```


Adapter/checkpoint tests, native measurement fixtures, checkpoint/resume tests,
raw-data preparation and actual notebook-cell acceptance are separate checks.
The repository contains no real Gemma measurements from this implementation
session. Random native models verify execution and numerical semantics; they
usually fail scientific eligibility gates, and their reports say so.

Real studies require an A100-80GB pilot and the model's own immutable G freeze.
The pilot records measured memory/timings before scientific execution. The earlier
4–8-hour estimate is not a promised runtime for the expanded complete protocol.
Stage and item checkpoints support repeated sessions without silently reducing
sample sizes or contexts. Each scientific verdict remains conditional on the
actual calibration, format, selection and damage gates.

Never merge this branch automatically. All delivered changes remain confined to
G source, its plan/notes and the six explicitly requested G notebooks.
