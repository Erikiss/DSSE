# Round 1b: guide for the part implementers (shared core)

Plan: `results/ju_space/plans/runde1b/` (main §2.3 shared objects, §3.1 gates, §4 sessions, §6 scripts). This folder:

| File | What |
|---|---|
| `r1b_contracts.py` | `SHARED` (main §2.3), `FILES` (every cross-session file: owner, session, path, format, keys), `SESSIONS` (main §4.2 with start conditions, models, round-1/2 inputs), `SLOTS` (main §4.4), `Step`. `python r1b_contracts.py` checks the tables (incl. a deadlock check of the slots). |
| `r1b_core.py` | shared core: models, SO-KEYS, SO-PVAL, SO-DMG, `Hooks`, SO-JLENS, `Ctx`, R9.1-R9.4 helpers. `python r1b_core.py selftest <root>`. |
| `r1b_run_session.py` | runs one session (gates, start conditions, copy-in, steps, Drive sync, time guard, manifests). |
| `r1b_run_slot.py` | runs a slot's sessions in one notebook runtime, waiting for start conditions; `--mock-dry-run`. |
| `r1b_sessions_R/D/K/L.py` | your session registries (stubs with one MOCK-only example step each; replace the example). |
| `r1b_sessions_main.py`, `r1b_report.py` | round level: r1b_ana's steps after L-ANA (CF families, four answers, labels, side-by-side tables, merged manifest, SG6 report check) and the tier-2 report session r1b_ana2 (CPU, ANA notebook). |
| `../../../notebooks/build_runde1b_gpu_notebooks.py` | builds the 9 slot notebooks (German cells; P0b added in the fix round). |
| `r1b_pilot.py` | SO-PILOT: `python r1b_pilot.py build <tokenizer> <wikitext103 dir> <pg19 dir> <out>` (pilot_material.json), r1b_P0b's guard (`Rec.item`), Drive and summary steps. |
| `r1b_r_p0b.py`, `r1b_d_p0b.py`, `r1b_k_p0b.py`, `r1b_l_p0b.py` | the parts' items of the re-pilot r1b_P0b (section 11). |
| `tests/fast_identity.py` | CPU bit-identity tests of the three `engine.fast` switches (section 11). |

## 1. Registering steps

In `r1b_sessions_<part>.py`: `STEPS = {"r1b_K_S1": [Step(...), ...]}` in run order. A part registers its own sessions and may add steps
to `r1b_P0` (pilot items) and `r1b_ana` (L-ANA). Keep the module light at import (import torch etc. inside the step functions).

```python
Step("k2_select", fn="r1b_K:k2_select", out="hn/hn_set.json", after="k2_screen", minutes=4)   # fn(ctx) in the session process
Step("r1_replay14", script="r1b_R1.py", args="{root} {work} 14", out="r1/b14.json")           # subprocess (r2_rk.open_block: one block per process)
Step("x2_coupling", fn="r1b_L:x2", out="l/x2.json", item="X2")      # tier-2 item: runs only if funded, else NA:not_run:tier2_unfunded
Step("r2_extra", fn="r1b_R2:extra", out="r2/extra.json", drop_if=("RL1",))   # dropped under trim RL1: NA:not_run:RL1
```
- A step is done when its record `steps/<name>.json` says ok and its `out` files exist; done steps are skipped on every rerun.
- Steps <= 5 min (SG5). Longer work: split it, or save pieces with `ctx.cached(name, fn)` and `raise r1b_core.StepYield` when
  `ctx.time_left()` is short (the continuation resumes). `raise r1b_core.NotRun("reason")` records 'not run (reason)'; steps after it
  read 'not run (prerequisite ...)'.
- Script steps end with `r1b_core.run_script(main)` (`main(ctx)`; the context comes from the environment, same API).
- Function steps share the loaded model across the session (`ctx.model()` is cached). Free large tensors yourself.

## 2. Ctx (what a step sees)

`ctx.session, attempt, step, part, account, tier, root, work` (session folder, synced to `dsse_runs/<session>`), `inp` (read-only
copies), `r1` / `r2` (round-1 / round-2 copies: `inp/r1/gpu1/rk/b14/state.pt`, `inp/r2/r2_q2/theta/...`), `device, mock, quick, pilot,
freeze` (runde1b_freeze.json), `declared` (producers declared 'not run').
`ctx.path(rel)`, `ctx.input(session, rel, required=True)`, `ctx.xfer(name)` (account B: A's exports), `ctx.missing(session)`,
`ctx.model(name)`, `ctx.jlens()`, `ctx.key(tag, *rest)`, `ctx.funded(item)`, `ctx.trimmed(trim)`, `ctx.time_left()`, `ctx.cached(...)`,
`ctx.log(msg)`, `ctx.write_json / save_pt / write_file`.

## 3. Contracts and inputs

- Write every cross-session file at `r1b_contracts.FILES[fid]["rel"]` of its session; read it with `ctx.input(session, rel)`. To change a
  path, key or start condition, edit `r1b_contracts.py` only (notebooks and runner read it) and run its check.
- Start conditions (SG4): `done(s, files)` (finished session; its files are copied), `file(fid)` (one file of a running session, same
  account, checked against its step's sha256), `xfer(fid)` (A -> B export, checked against `r1b_xfer/xfer_manifest.json`), `trig(...)`
  (tier-2 trigger), `any_done(...)`. Every condition is also met when the producer is declared 'not run' after two attempts; then
  `ctx.input` of a required file raises `NotRun("input missing")`.
- Account A reads B's sessions only after they finish (copied from `dsse_runs_B`); B reads only A's exports in `r1b_xfer_A`. A file that
  B needs from A must have an `xfer` name in `FILES` (the runner publishes it).

## 4. Models

`tok, model = ctx.model("qwen3-1.7b")` (also `"qwen3-0.6b"`, `"qwen3.5-2b"`): pinned revision and sha256 (`r1b_core.ensure_model`),
float32, TF32 off, matmul 'highest', sdpa attention. A model the default cannot load: `r1b_core.register_loader(name, fn, layers=...)`
(pins from runde1b_freeze.json `models`). Helpers: `chat_ids(tok, text)` (thinking off), `user_start(tok)` (rk_loop's START),
`pad_left`, `stop_tokens`, `generate(model, tok, rows, T, max_new, setup=...)` (keyed Gumbel-max, token-identical to rk_loop.generate),
`mean_read(model, prompts, conts, block, pad_id)` (= rk_loop.read at any block), `with sdpa_efficient(device):` for every forward above
2K tokens (SG9 / G-LCE-2). A real model on the CPU is refused; tests use MOCK=1.
`model_pins(name, freeze)` takes runde1b_freeze.json `models[name]` where present and raises when the record and `MODELS` disagree on
repo, revision, layers, hidden or a file's sha256 (sha256 = the union of both file lists). Qwen3.5-2B is pinned in `MODELS` (revision
15852e8c, sha256 of the safetensors file, tokenizer.json, config.json); `load_model` passes the record to a registered loader that
takes `freeze`; r1b_k_s3's loader records the 15 mtp.* shapes on the model (`model.r1b_mtp_shapes`; the tensors are read by
`mtp_tensors`, since transformers drops '^mtp.*' on load). Datasets: `fetch_dataset` knows Hugging Face snapshots (`<root>/data/<name>`)
and `source: "gcs"` (PG19: `url_template`, exact file names, per-file sha256, `<root>/datasets/<name>`); CLI `fetch-models`,
`fetch-datasets`. Before the freeze, pilot sessions read `results/ju_space/plans/runde1b/runde1b_pilot_pins.json` (models, datasets,
jlens, engine of the draft record; `python r1b_core.py pilot-pins <draft> <out>`) instead of the absent freeze record
(`r1b_core.freeze_record(pilot=True)`); runner, slot runner and notebooks do the same.

## 5. Hooks

```python
with Hooks(model) as hk:                       # removed on exit; hooks on one block run in registration order
    hk.push(14, d, alpha, signs=sg, cap=cap_b, rowstart=pad + user_start(tok))   # = rk_loop's push (CC4, tested to 0 difference)
    hk.neuron(10, idx, "read", name="z")           # MLP intermediate (down_proj input): read / scale / add / clamp / zero / mean (frac)
    hk.neuron(20, idx, "scale", 3.0, rowstart=rs)
    hk.capture([6, 14], positions=None, dtype=torch.float16).stop_after(14)          # per-token states, stopped forward
    hk.run(lambda: model.model(input_ids=ids, attention_mask=am))
states, z, share = hk.states[14], hk.reads["z"], hk.capped_share()
```
`rowstart` limits an operation to positions >= rowstart in multi-token forwards (decode steps always included); with a cached prefix
set `hk.offset` to the prefix length (chunked prefill).

## 6. Keys (SO-KEYS)

`ctx.key(tag, *rest)` = rk_loop's keyseed (verbatim copy, sha256-checked) for a tag registered to your part in `r1b_core.KEYS`; a tag of
another part raises. New tags: add them to `KEYS` (the uniqueness test runs in the self-test); R's loop names: `register_r_loops([...])`.
R's stated reuse of round-1 keys uses `r1b_core.keyseed(...)` directly. `python r1b_core.py keys <path>` writes the package file
r1b_keys.json. Numeric seeds (bootstraps, permutations) are free.

## 7. Damage (SO-DMG)

```python
refs = RefSet.from_json(ctx.xfer("dmg_ref_R16.json"))          # REF-R16 (account A: ctx.input("r1b_R_S0", "xfer/dmg_ref_R16.json"))
dm = Damage(model, refs)                                       # unmodified log-probs once; = r2_unlearn_qwen.RefKL (tested to 1e-6)
res = calibrate(lambda f: dm(lambda: Hooks(model).push(b, d, f, cap=cap_b, rowstart=torch.tensor([st])))["damage"], f0, DELTA1)
res = calibrate(lambda a: dm(lambda: Hooks(model).neuron(l, idx, "scale", a))["damage"], 2.0, DELTA1, form="gain")   # NSCALE_DN
```
`dm(...)` returns damage, per-text values and top-1 change; `calibrate` returns f, damage, tried, in_band and the labels 'damage out of
band' / 'dose capped'. REF-U32 = REF-R16 + REF-K16 (`RefSet` supports `+`): report every calibrated intervention on it.

## 8. Outputs (SG6, R9.1-R9.4)

- `ctx.write_json(rel, obj, config=..., seeds=..., labels=[...])` adds the provenance block (plan and code hashes, module sha256,
  config hash, inputs with source and sha256, models, J-lens, versions, keys, tier, reduction level, session, attempt, times).
  `ctx.save_pt` / `ctx.write_file` write a sidecar `<file>.prov.json`. A step whose outputs lack provenance fails (SG6).
- Raw NaN / inf are refused: use `val(x, "NaN:<why>")`, `coded("CENSORED:below_grid")`, `coded("NA:gate_closed:SG10")`,
  `NA_TIER2`, `NA_COND`; `code_counts(obj)` for summaries; `labelled(value, *labels)`; `dist(values, groups)` for R9.4.
- SO-PVAL: `interval`, `p_one_sided` ((1 + #null) / 2001), `p_rank`, `p_pi` (K.4), `holm`, `bh`, `boot_idx`, `boot_two_level`,
  `prompt_bootstrap` (P256, seed 1001).
- r1b_P0 / r1b_P0b write JSON only, with top-level keys from `PILOT_KEYS` (rates, memory, throughput, parity, code_checks, timing, env,
  notes); the runner merges r1b_P0's into `r1b_P0/p0_engineering.json` (r1b_P0b: `p0b_aggregate.json`; its own compact summary step
  writes `r1b_P0b/p0b_engineering.json`) for the freeze record.

## 9. MOCK and tests (CPU here: `taskset -c 3`, one heavy process at a time)

- `MOCK=1`: rk_loop's tiny random model (config and tokenizer from `<root>/models/qwen3-1.7b`), example steps allowed, SG1 / SG7 and
  the round-1b part of SG3 recorded but not enforced, contract files a session's steps did not write are filled with labelled
  placeholders (trigger files carry the triggering value), so every start condition can be met.
- `python r1b_core.py selftest <root>`: keys, p-values, codes, hooks, damage, calibrate, J-lens pin, Ctx; against rk_loop itself
  (child process): push (CC4), keyed generation, read, RefKL, calibrate.
- `python r1b_run_slot.py --mock-dry-run <dir> <root>`: every session of every slot on a mock two-account tree (symlinks for the
  shortcuts), one process at a time; `R1B_MOCK_WITH_R1=1` builds rk_loop MOCK round-1 states at blocks 14, 18, 6 (needs the persona
  files and rk_dirs.pt under `<root>`). A part can add mock inputs with `mock_inputs(runs_A, runs_B, root)` in its registry module.
- `python r1b_run_session.py --selftest <dir> <root>`: the dry run, then SG6 failure, time guard and continuation, resume from Drive,
  script steps and NotRun.

## 10. Runner behaviour

Exit codes: 0 done, 1 failed (rerun: finished steps skipped), 2 gate, 3 time guard (continuation 'b', 'c', ... starts at once in the
slot), 4 waiting, 5 'not run' (tier 2 unfunded / not triggered). Time guard per session: 4.5 h after the session's start. Before the
freeze `FROZEN_PLAN_SHA256` and `FREEZE_RECORD_SHA256` in r1b_run_session.py are empty, so only r1b_P0 (SG1 off) can run for real.

## 11. Fix round before the freeze (14 November 2026; arbeitsstand/freeze_prep.json)

- Part D: `decode(...)[0]` instead of `y, *_ = decode(...)` everywhere (the returned cache stayed alive in `_`: the pilot's B = 3 OOM at
  28K / 32K); p0_d offloads the prefix. `r1b_d_stim.cfg` reads `engine.D.B_long` / `engine.D.B_mid` (absent: 3 / 7).
- Datasets and models: see section 4. `datasets.wikitext103.rk1_train_offset_lines` must be present outside the pilot (r1b_k_rk raises
  instead of using 0); the draft's 40000 stays (SO-PILOT's passages end at raw train line 34712).
- SO-PILOT: `results/ju_space/plans/runde1b/pilot_material.json` (32 new prompts, 8 WikiText-103 train passages at raw lines
  34261-34712 by line range and text sha256, PG19 pilot books 25-28 with a 16,384- and a 32,768-token slice each by offsets and token
  sha256). Keys: `r1bP0_R`, `r1bP0_D`, `r1bP0_K`, `r1bP0_L0`, `r1bP0_L0duet`, `r1bP0_fast` (registered in `KEYS['main']`).
- Re-pilot `r1b_P0b` (account B, tier 0, pilot mode, slot/notebook P0b, Drive `dsse_runs/r1b_P0b`, about 60 A100-minutes; starts
  after r1b_P0): p0b_r_rates (R-P's list under cost_R2.py's names: LOAD, GEN, GEN80/200/360, RD14, RDX, RALL, RFULL, RSHORT, FWD, FB,
  SOFT, DMG), p0b_r_replay (rk_loop MOCK opener), p0b_d_decode (warm-up; B = 1-4 at 4,096 / 16,384 / 28,672 / 32,000; plain and in-place
  KV with GPU bit identity), p0b_d_parity (crop / copy / chunked against a full recompute at 16,384 and 32,768 on PG19 slices),
  p0b_d_profile (torch.profiler, 32K, B = 1), p0b_k_rates (K0 items 3-5, keyed rollouts 80/160/320 with and without capture, kept-rows
  identity), p0b_k_checks (items 6, 8, 9), p0b_k_35 (item 7: Qwen3.5-2B load, G-K18, G-K17, rates, its J-lens), p0b_l_decode (L0 (i)
  without the prefill, padded rows, nosync identity), p0b_l_long (L0 (iii), (iv), (vii)), p0b_drive (copy into the Drive session folder,
  cold read of another runtime's file), p0b_summary (`p0b_engineering.json`). Every item runs under `r1b_pilot.Rec.item`: an
  out-of-memory or other error is recorded as a code and the session continues.
- Engine switches `engine.fast.<name>` in runde1b_freeze.json (default False when absent; read with `r1b_core.engine_fast(freeze,
  name)`, `freeze=None` = the running session's record, set by `Ctx`):
  - `l_decode_nosync`: `r1b_l_engine.decode` without per-step host syncs (`_decode_nosync`: device-side records copied once, stops via
    `torch.isin`, any-alive read every 8 steps, `mask4` = the 4-D mask transformers builds for a padded batch, None unpadded; extra
    steps after the last stop are discarded, generators reset, `on_step` re-called with the last step). Argument `nosync=`.
  - `lce_inplace_kv`: SO-LCE's `InplaceLayer` (preallocated buffers, slice-assign, attend over a view; fwd, crop, crop_layers, copy with
    headroom and kind conversion `LCE.copy(..., inplace=)`, offload releases the buffer, decode, token_logprobs). Argument `LCE(model,
    inplace=)`.
  - `k_gumbel_kept_rows`: `Lens.rollout` draws the Gumbel noise for the N kept rows only. Argument `kept_rows=`.
  - Tests: `MOCK=1 python tests/fast_identity.py <root>` (58 bitwise checks: logits of every step, tokens, every recorded quantity,
    generator states, masks, probe state, every LCE operation, rollout records); r1b_P0b runs both paths on the A100 and records
    `*_identical_*` verdicts in `p0b_engineering.json`. Set a switch True in the freeze record only if its GPU verdicts are all True.
- Trims: R ['RL1', 'RL2'] = level RL2; D T1-T8, L T2-T5 through `Ctx.trimmed`; K: `K-drop1` (K7b), `K-drop2` (K9 c4 / c5), `K-drop3`
  (K8b bistability block with 16 templates), `K-drop6` (K10a delta2); `K-drop4` (K6c(ii)) and `K-drop5` (K6e descriptive arms) are not
  implemented.
- `python r1b_core.py keys <out>` imports r1b_sessions_R (R's loop names) before writing r1b_keys.json.

## 12. Optional observational hidden-state capture in the H_tq continuation

The user approved a bounded capture for the next `htq_v2` K_S1 run (16 November 2026). It observes the model's own ordinary
K3a generation. The observer itself adds no model request or forward pass, changes no frozen sample count, seed, decoding
setting, gate, threshold or scorer, and performs no probe fitting or training. A separate pre-run parity/overhead check
adds a few fixed-input prefill forwards; its outputs are discarded and never join the scientific sample or capture dataset.
The resulting dataset is an exploratory observation;
predictive associations from it are not causal attribution and do not establish a reproduction of a GPU-memory debugger
or of the user-mentioned paper.

**Selection and measurement.** Capture a fixed prefix of at most 128 S3 requests, 64 A-det requests and 64 NE requests,
selected before their output labels are known. These are observation rows within K3a's existing samples, not a smaller
scientific sample. For each selected request retain the decoder-block output at the last non-padding prompt position on
the actual prefill forward, in FP32, for all 28 Qwen3-1.7B blocks (hidden width 2,048). This vector precedes selection of the
first generated token. It is neither an all-token trace nor an MLP-neuron activation vector; its shape and layer numbering
must be declared in the file metadata. Prompt identity, family and sample identity keep capture rows aligned with their
ordinary continuations and independent scorer labels.

The tensor payload cap is `256 * 28 * 2048 * 4 = 58,720,256` bytes (**56 MiB**), plus request metadata, file/container
overhead and provenance. Counts can be lower if a family has fewer available requests. This storage bound is not a timing
estimate. Real A100 copy/synchronisation and I/O overhead remains unmeasured. Before expensive K3a generation, a GPU
parity/overhead check uses two fixed inputs, warm-up and three timed pairs with and without capture, requiring exactly
identical logits. Its benchmark JSON records the local measurement; a failure stops K3a. Distinguish that measurement
from model-free geometry/metadata tests and from a whole-run timing forecast. A tiny-model benchmark does not establish
the overhead on the full model.

**Opt in and location.** `R1B_CAPTURE_HIDDEN=1` enables this observer for K3a. The new HTQ START notebook enables it for its
real K_S1 subprocess; older/default scripts leave it off. It belongs to the isolated variant session, under
`observations/hidden_states_v1/`: one `<family>.pt` shard per request family and a final `manifest.json`. In account B this
is below `dsse_runs/variants/htq_v2/r1b_K_S1`; the account-A START notebook reaches the same folder through `dsse_runs_B`.
The observation files carry their own version/configuration and an explicit exploratory label, independently of the
post-hoc H_tq amendment labels.

**Metadata and provenance.** Each selected request needs its stable request/question and sample ID, task family, exact
rendered prompt token IDs and non-padding length, sampling key, ordinary generated token IDs/text, and ordinary scorer label.
Continuation tokens follow the existing decoder convention and exclude its stop token; do not claim an exact recorded EOS ID.
Record block indices, capture point, dtype, tensor dimensions and row order, model revision, code/configuration identity,
session/attempt and H policy. Use `ctx.save_pt` for shards and `ctx.write_json` for the manifest, so SG6 sees the normal
`.prov.json` SHA-256 sidecars and provenance. A complete manifest must distinguish requested versus captured counts and
identify all completed shards; an absent or partial manifest must not be described as a complete capture.

**Continuation and sync.** These files are optional K3a outputs: present files must be declared in the step's `opt` patterns
to enter its output-hash record and SG6 checks, while ordinary runs without capture still succeed. The existing runner syncs
`observations/` after steps, excludes `.tmp`/`.r1btmp`, and resumes it from Drive. Write shards atomically; keep partial/retry
status explicit, use deterministic family/request identities, and never append duplicate requests to the 256-row cap on
retry. Enablement is not a request to replay an already completed step: the runner skips completed K3a/K_S1 before the
observer runs. In that case report that no new capture was produced rather than silently repeating expensive measurements.

For later analysis split by question/request identity, not by tokens from the same request. The fixed-prefix observation
sample is not a representative random sample of model usage, and scorer labels describe the task-specific observed
behaviour. Preserve those limits when reporting any classifier or visualisation derived from these files.
