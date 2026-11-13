# Round 1b: guide for the part implementers (shared core)

Plan: `results/ju_space/plans/runde1b/` (main §2.3 shared objects, §3.1 gates, §4 sessions, §6 scripts). This folder:

| File | What |
|---|---|
| `r1b_contracts.py` | `SHARED` (main §2.3), `FILES` (every cross-session file: owner, session, path, format, keys), `SESSIONS` (main §4.2 with start conditions, models, round-1/2 inputs), `SLOTS` (main §4.4), `Step`. `python r1b_contracts.py` checks the tables (incl. a deadlock check of the slots). |
| `r1b_core.py` | shared core: models, SO-KEYS, SO-PVAL, SO-DMG, `Hooks`, SO-JLENS, `Ctx`, R9.1-R9.4 helpers. `python r1b_core.py selftest <root>`. |
| `r1b_run_session.py` | runs one session (gates, start conditions, copy-in, steps, Drive sync, time guard, manifests). |
| `r1b_run_slot.py` | runs a slot's sessions in one notebook runtime, waiting for start conditions; `--mock-dry-run`. |
| `r1b_sessions_R/D/K/L.py` | your session registries (stubs with one MOCK-only example step each; replace the example). |
| `../../../notebooks/build_runde1b_gpu_notebooks.py` | builds the 8 slot notebooks (German cells). |

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
- r1b_P0 writes JSON only, with top-level keys from `PILOT_KEYS` (rates, memory, throughput, parity, code_checks, timing, env, notes);
  the runner merges them into `r1b_P0/p0_engineering.json` for the freeze record.

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
