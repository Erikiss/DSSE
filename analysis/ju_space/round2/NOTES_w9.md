# NOTES of implementer 9 (plan section 9.5 split (9)): runner, notebooks, manifest

Implemented against the frozen plan `results/ju_space/plans/abschnitt11_12_runde2_plan.md` (sha256 `5e476340…5edc2d`, freeze commit
`5b46e38cd800236d2d597044510652cbc48ad50d`, 2026-11-12T19:47:20Z). The first draft followed v2; the section "Changes from the v2-based
draft" lists what was changed for the final text.

## Files

| File | What |
|---|---|
| `analysis/ju_space/round2/r2_run_session.py` | the session runner (all logic in functions; import-safe, the builder imports its tables) |
| `notebooks/build_runde2_gpu_notebooks.py` | builds `notebooks/runde2_{q1,q2,q3,t1,t2,t3,t4,pc,ana}.ipynb` (German, round-1 cell structure) |
| scratchpad `r2_impl/w9/` | tests: `test_runner.py` (scenarios), `stubs/` (stub scripts for all 15 round-2 scripts and the library self-tests), `bin/git` (fake git) |

CLI of the runner:

```
python r2_run_session.py <session> <root> <work> [<drive_dir>] [--dry-run [--verbose]]   # the session (or what it would do)
python r2_run_session.py <session> <root> <selftest_dir> --selftest                     # MOCK/QUICK self-test (used by the notebooks)
python r2_run_session.py <session> <root> <work> <drive_dir> --sync-only                # last Drive sync (notebook's final cell)
python r2_run_session.py --mock-r1 <root> <dsse_runs_dir> [rk,pc]                       # mock round-1 tree for code tests
```

Exit codes: 0 done (`session_done.json`); 1 steps failed (rerun retries them); 2 G1/G2 or a model checksum failed; 3 time guard
(`session_partial.json`, rerun = continuation session); 4 start conditions not met (`start_check.json`, nothing ran).

## The step table (top of the runner; one Step = one script call; easy to edit)

| Session | Steps in order (BLOCK; required outputs relative to `<work>`) |
|---|---|
| r2_q1 | `axes` r2_axes.py (14; axes/r2_axes.json, r2_axes.pt, tok14.pt) · `q3`, `q4a`, `q2` r2_unlearn_qwen.py (14; unlearn/<stage>.json) · `gen_b14` r2_gen_loop.py (rgi/dyn_b14.json) · `q4b`, `armA`, `q5`, `q6` r2_unlearn_qwen.py (14) · `gen_b<lo>`, `gen_b<hi>` |
| r2_q2 | block 14: `q0_b14`, `checks`, `cells_b14`, `turn2`, `power_b14`, `fdt_b14` (r2_theta.py stages; theta/b14_q0.json, theta/checks.json, theta/b14_cells.json, theta/b14_turn2.pt, theta/b14_power.* (may be absent), theta/b14_fdt.pt); each extra block: `rsib`, `q0`, `cells`, `power`, `fdt` (theta/b<b>_r2RSIb.json first) |
| r2_q3 | `geom_17` r2_rgi_geom.py (rgi/geom_meta_qwen3-1.7b.json, causal_…pt, dirs_…pt; local arrays) · `stats_17` r2_rgi_stats.py (rgi/geom_stats_qwen3-1.7b.json) · `geom_06`, `stats_06` the same with `--model qwen3-0.6b` (only under G33) |
| r2_t1 | r2_rgi_toy.py `r0`, `r1` (rgi_toy/r1/w0-w7.json), `cross` (rgi_toy/cross/w0-w4.json), `fam` (after cross), `forcing` |
| r2_t2 | r2_theta_toy.py `t1`, `t2` (after t1), `t5` (after t1), `rules` (after all; theta_toy/verdict.json) |
| r2_t3 | r2_unlearn_toy.py `t0`, `t1`, `t2`, `t3`, `t4`, `t6` (unlearn_toy/<stage>.json) |
| r2_t4 | r2_unlearn_toy.py `t5` |
| r2_pc | r2_pc_toy.py `t7` (pc_toy/t7.json), `cross_pc` (pc_toy/cross_pc/w0-w4.json); env R1PC = `<work>/r1`, R2T1 = `<work>/r2in/r2_t1` |
| r2_ana | `linearity` (unlearn/q1.json), `theta_verdict`, `rgi_verdict`, `unlearn_verdict` (after linearity), `report` (after all; round2_summary.json; optional r2_manifest.json, pc_toy/gates.json) |

Command lines: Qwen `python <script> <root> <work>/r1 <work> [<stage>]`; toy `python <script> <work> <stage>`; analysis
`python <script> <work>/r1 <work>/r2in <work>/<out.json>`. Every step gets env DEVICE, NT, PYTHONUNBUFFERED, POOLDIRS, PERSONA, BLOCK
(block steps) and the information variables R2_SESSION, R2_ATTEMPT, R2_ROOT, R2_WORK, R2_R1, R2IN, R2_MANIFEST, R2_B3 (final blocks of
B3), R2_BLO, R2_BHI, R2_ORDER (b(1), b(2), b(3)), R2_MISSING_BLOCKS. Under `--selftest` also MOCK=1 and QUICK=1; in a real run MOCK
and QUICK are removed from the environment.

## What each session needs from round 1 (7.3) and where it lands

| Session | Start (7.3) | Copied into `<work>/r1/<shard>/…` (read-only) |
|---|---|---|
| r2_q1, r2_q2 | G1, G2, G4 | `gpu*/rk/b<b>/{state.pt, loops.json, pilot.json, pilot@1.5.json, k5.json, k6.json, manip@*.json}` for the final blocks of B3 only; `gpu*/summary_b*.json`; `gpu2`, `gpu3` `{dn_delta.json, blocks_chosen.json}`; `gpu*/{shard_done.json, code_commits.txt}`; `r1_finality.json` (into `<work>/r1/`). Data from gpu1-gpu3; gpu4-gpu6 only for version metadata |
| r2_q3, r2_t1-r2_t4 | G1, G2 | nothing (the manifest still records the Drive times of `gpu*/logs/*.log` and `gpu*/summary_b*.json` if visible) |
| r2_pc | G1, G2, G5, r2_t1 done | `gpu*/pc/jobs/pc_mup_*.json`, `gpu*/pc/models/pc_mup_*.pt`, `gpu*/pc/{jobs,models}/bp_mup_W{128,256}_s*_K12_g0_wd1.*`, `gpu*/pc_{summary,expected,missing}_gpu*.json`, shard metadata, `r1_finality.json`; round 2: `r2_t1/rgi_toy/models/at5_*.pt` into `<work>/r2in/r2_t1/` |
| r2_ana | G1, G2, G4, G5, all other sessions done (or declared) | the r2_q1 list, the r2_pc list, G46's BP cells; round 2: every file of `dsse_runs/r2_*` except `local/`, `logs/`, `rk/b*/state.pt`, temporary files, into `<work>/r2in/<session>/` |

Round-1 folders: gpu1-gpu3 ran in account A, gpu4-gpu6 in account B. r2_q1/r2_q2 need gpu1-gpu3; r2_pc and r2_ana need all six (and
r2_t1, resp. every r2_* folder). The notebooks say so and explain sharing (read access) plus "Verknüpfung zu Drive hinzufügen" or
copying into this account's `dsse_runs/`; they list missing folders by name before the run (names only).

## Readings of ambiguities (my decisions; each easy to change)

1. **Continuation sessions (7.4).** The same notebook rerun; the runner names the attempt `r2_q1b`, `c`, … after each time-guard stop
   (counted in `session_attempts.json`); a rerun after a crash keeps the name. Same work and Drive folder (`dsse_runs/r2_q1`), so steps
   are skipped; one manifest per attempt (`manifest_<attempt>.json`, an older one with the same name is kept as `_1`, `_2`).
   `session_done.json` lists every attempt and the steps it ran (packing record).
2. **Time guard clock.** Counts from the notebook's first cell (env R2_T0), because the 20 min include setup (7.1); checked only after
   a step that ran, so every attempt makes progress even after a long setup. Skipped steps do not trigger it. A per-step timeout of
   60 min (R2_STEP_TIMEOUT) kills a hung step (its own process group only).
3. **"Skipped when its output exists".** Done = the runner's success record `steps/<name>.json` plus every required output. The record
   is needed because some outputs are shared (`theta_toy/nets.json` of t1 and t2) or could be partial. Outputs are hashed into the
   record. A step whose script exits 0 without its required outputs fails (exceptions: `power`, whose stage writes nothing when no
   cell has rotation < 0.8).
4. **Failures.** A failed step is retried once at the end of the pass (round-1 retry pattern); steps that need it (`after`) wait; the
   session exits 1 without `session_done.json`; a rerun retries. Round 1's parallel PC job pool is not used for measurements (7.1: no
   parallel processes); it survives only in the mock tree builder (pc_meta QUICK jobs).
5. **Sync.** Incremental and per-file error-tolerant (a Drive error is printed, never fatal), atomic by rename. Not synced: `local/`
   (as the plan says) and also `r1/` and `r2in/`, which are verbatim copies of Drive files recorded with sha256 (syncing them would put
   ~0.3-1 GB of duplicates into every session folder). Big outputs (r2_axes.pt ~300 MB, tok14.pt ~65 MB) are copied once, because
   unchanged files are skipped. `rk/b<b>/state.pt` copies (r2_rk.open_block) change with most Qwen steps and are re-copied each time.
6. **Read-only copies.** Files 0444 and folders 0555 under `r1/` (advisory: Colab runs as root), so at the end every copy is re-hashed
   and added files are listed (`inputs_changed_during_session` in the attempt record). Copies are also compared with earlier attempts.
7. **OD-G9 duplicates** are resolved at copy time (only the chosen file is copied; the choice and the candidates go into the manifest):
   rk files and summaries from the assigned shard (gpu1 block 14, gpu2 b_lo, gpu3 b_hi), PC jobs from the shard whose pc_expected lists
   them, then shard_done.json, then the newest file. rk files are copied only for final blocks of B3 (summary present).
8. **Blocks (R1-1).** b_lo/b_hi = the `rk/b<b>` folder of gpu2/gpu3 (blocks_chosen.json when the folder is missing; loops.json
   `_meta.block` cross-checked); G7, G8 (also recorded for q1/q2), G9 order with ties within 1e-4 broken in the label order b_lo, 14,
   b_hi ("block 14 placed by its delta" read as: 14 keeps its delta position unless tied).
9. **G4 'not run' rule.** The runner marks a block step 'not run (round-1 output missing)' before launching it when its block is
   final-missing (R1.final) and the step reads that block (`r1_reads`): `axes` reads every B3 block and runs unless all are missing
   (r2_axes hosts the texts at another final block); `q6` (C-Q6) reads no round-1 file and always runs; dependants of a 'not run' step
   are 'not run' too. Worker 1's scripts would also write 'not run' JSONs themselves; the runner does not rely on every script doing so.
10. **R1.final** (`dsse_runs/r1_finality.json`, operator-written): `{"gpu4": {"runs": [{"start": …, "end": …}, …], "final": true}}`
    (also accepted under a top-level `"shards"` key). Final = `"final": true` and at least two runs. Recorded in every manifest (content,
    sha256, Drive time).
11. **r2_ana's "declared 'not run' after two attempts".** Notebook parameter `NICHT_GELAUFEN` (env R2_DECLARED_NOT_RUN); the runner
    accepts a declared session only if its `session_attempts.json` on Drive lists at least two attempts.
12. **G46 copy list.** "The cells round 2 retrains" read as every R1 BP main cell without regulariser:
    `bp_{mup,sp}_W*_s*_K{1,12}_g0_wd{0,1}` (jobs and models).
13. **G2 strict A100** (round 1 allowed other GPUs with a note); checked in a short child process so the runner holds no CUDA context;
    the notebook's first cell stops on another GPU before any download.
14. **G1** stops every run, including the self-test (section 14: "refuses to run"). The freeze commit = the commit that added the plan
    file (`git log --diff-filter=A`), which must equal FREEZE_COMMIT and be an ancestor of HEAD. The notebooks clone without `--depth`
    (partial clone, blobs on demand) so that G1 and G3 can read the history.
15. **G3.** Module sha256 at every shard's `code_commit` (else last line of code_commits.txt) and in the checkout; blob ids are compared
    first (no download for unchanged files); commits missing locally are fetched; a module that differs carries the plan's label
    'round-1 code differs between shards' with the reason (between shards / round 1 vs checkout / not checkable). Earlier commits of
    `code_commits_used` are recorded too. R1.meta keeps only code_commit, code_commits_used, gpu, torch, transformers.
16. **Models.** The notebook verifies the checksums and writes `<root>/models/verified.json`; without it the runner hashes the files of
    the session's models itself (real runs). Qwen3-0.6B: revision `c1899de289a04d12100db370d81485cdf75e47ca` (HF API, 12 Nov 2026),
    `model.safetensors` sha256 `f47f7117…6874b`, `tokenizer.json` sha256 = the 1.7B tokenizer's (identical file). A failed 0.6B download
    makes `geom_06`/`stats_06` 'not run (G33 …)'.
17. **Local arrays (r2_q3).** The runner records which files a step wrote under `<work>/local/`; on a new machine such a step reruns
    unless every consumer (`needed_by`) is done. No file names are assumed.
18. **References.** All 43 Appendix E files are looked up at `round2/ref/<path>` (status ok / moved / differs / missing), plus
    MISSING.txt, the sha256 of the two repository modules the shim uses (Appendix E), and any committed ref-comparison output: a file in
    `round2/` or `round2/ref/` whose name contains "compar" or "refcheck" (r2_toy_common's selftest writes `ref_comparison_w5.json`).
19. **NT** defaults to min(8, CPUs - 2) (Colab A100: 8; round 1 used 4); RUN_DEVICE=cpu (code tests only) skips G2 and the model hashes.

## Self-tests (notebooks)

`--selftest` runs, on copies of a mock dsse_runs tree, the scripts' own tests (`LIB_SELFTESTS`: r2_q1 `r2_stimuli.py check`,
`MOCK=1 r2_rk.py selftest`, `MOCK=1 r2_axes.py … selftest`, `MOCK=1 BLOCK=14|8 r2_gen_loop.py … selftest`; r2_q2 stimuli and r2_rk;
r2_q3 stimuli; toy sessions and r2_ana `QUICK=1 r2_toy_common.py selftest`; on the CPU, as their authors ran them), then the session
itself under MOCK/QUICK through the runner (wave 2: first the sessions it reads). The mock tree: `rk_loop.py MOCK=1` pilot + all at
blocks 14, 8, 20 (real runs; worker 1's selftests need genuine extra blocks), rk_verdict summaries, the same fake dn_delta as
`r2_rk.fake_delta` (order 14 < 8 < 20), 20 `pc_meta.py QUICK=1` jobs (bp_mup W 128/256, pc_mup W 128/256) renamed without `_quick`.
The notebook stores a marker per code commit in `dsse_runs/<session>`, so a rerun skips the self-test; the notebooks recommend a first
run with `NUR_SELBSTTEST = True`. r2_ana's full self-test (all sessions) is off by default (`SELBSTTEST = False`).

## Changes from the v2-based draft (coordinator: "say which v2-based parts you changed")

- FROZEN_PLAN_SHA256 = the frozen file; FREEZE_COMMIT / FREEZE_TIME constants; G1 now also stops self-tests; manifest block `freeze`
  (commit and time from git next to the constants, plan and amendments sha256).
- Drive metadata for every session (also those that copy nothing): times of `gpu*/logs/*.log` and `gpu*/summary_b*.json`, first
  summary, `freeze_before_first_summary`; every copied file keeps its Drive time.
- R1.final replaces v2's 'same pc_missing after two runs' (my v2 draft used an env declaration): G5 by shard_done.json or R1.final;
  G4 includes block 14 (critique v2-12s) with the per-step 'not run' rule; `r1_finality.json` copied and recorded.
- Session repacking (final 7.2): C-Q4b, arm A, C-Q5, C-Q6 moved from r2_q2 to r2_q1; r2_q2 is Part B only and runs stage `rsib` (the
  r2RSIb loop) before B-Q0 at each extra block (q0 waits for it; G49 is the script's to record); r2_pc lost its `gates` stage (G-meta /
  G-PC in r2_ana; r2_report writes pc_toy/gates.json) and copies the R1 BP-muP W 128/256 nets for C-T7's BP side; r2_t4 = C-T5 primary
  only (same call). Notebook texts and estimates from the final 7.2 (r2_q1 17.3 min and r2_q2 17.2 min with margin).
- Appendix E: 43 references (impose.py, pilot.py, v2 and its critique added), the shim modules and the ref-comparison output recorded.
- After worker 1's API: GEN-loop outputs required (r2_gen_loop writes the JSON also under the AUC gate); library self-tests in
  `--selftest`; mock tree aligned with r2_rk's (real MOCK runs at 8 and 20, same fake delta).

## Tests and results

- `python test_runner.py` (scratchpad `r2_impl/w9/`; NT=1, one process at a time, stubs, fake git, fabricated dsse_runs; no round-1
  output exists there): 172 checks pass (`units dry q1 guard fail wait q3 wave2 integ cont selftest`, 438 s). Covered: brace expansion,
  G7-G9 incl. ties, OD-G9 choices, R1.final rule; dry runs of all nine sessions write nothing; full runs with step order, env, manifest
  (sha256 and shard of every copy, freeze, Drive times, G3 label, R1.meta, G47, references), Drive sync (no r1/, r2in/, local/),
  read-only copies, resume, single-step rerun; time guard -> `session_partial.json` -> continuation `r2_q2b` -> done with packing log;
  retry once, blocked dependants, exit 0 without outputs = failure, gated power step; G4 wait, R1.final with 1 run / without final /
  valid, block 14 final-missing (only its readers 'not run'); G1 changed plan / not ancestor / other freeze commit, also in self-test;
  G2 without A100; r2_q3 local arrays on a new machine, G33; r2_pc (G5 by R1.final, r2_t1 inputs, R1PC/R2T1), r2_ana (declarations
  with two attempts, mirror, copy exclusions); integrity of copies; Drive errors not fatal; explicit continuation name; self-test with
  the real round-1 scripts building the mock and a failing library test stopping it.
- Worker 1's real self-tests through the runner on its mock tree: r2_stimuli check ok, r2_rk 35/35, r2_axes 13/13, r2_gen_loop 10/10
  (blocks 14 and 8), 52 s on one core. `r2_toy_common.py selftest`: 67/67 (114 s).
- Notebooks: nbformat-valid; every code cell parses under Python 3.11, 3.12, 3.13; the summary cell ran on test outputs. The runner
  and the builder also parse under 3.11 (one PEP-701 f-string was removed).

## Open points

1. **Process count against the cost model.** 7.2 costs r2_q1 and r2_q2 as 3 processes each (one model load per block); the runner
   starts one process per script call: r2_q1 11, r2_q2 16, i.e. about 8 x 30 s = 4 min and 13 x 30 s = 6.5 min of extra model loads.
   Expected wall time then about 17 and 19 min without the self-test, so the time guard may move the last steps of r2_q2 to a
   continuation. Fix without touching the runner logic: let r2_theta.py and r2_unlearn_qwen.py accept a comma-separated stage list
   (e.g. `q0,checks,cells,turn2,power,fdt` at BLOCK 14) and merge those rows of the step table into one step with all outputs; worker
   1's r2_axes + block-14 GEN loop can share a process too (`r2_axes.main` then `r2_gen_loop.run`), which the table does not use yet.
2. **Interfaces of scripts not yet written** (r2_unlearn_qwen, r2_theta, r2_rgi_geom/stats, the toy and analysis scripts) are the
   plan's 9.2-9.4 plus my readings above: stage names (`rsib`, `rules` for "t4 / rules", `cross_pc`), output paths (per-world files,
   `geom_meta_<m>.json` with m = qwen3-1.7b / qwen3-0.6b), `--model qwen3-0.6b` after `<r2_dir>`, R2T1 for r2_pc, out.json paths in
   r2_ana with earlier outputs mirrored into `<r2in>/r2_ana/`, and `r2_theta.py rsib` recording a failed loop (G49) with exit 0. The
   notebook self-test catches a mismatch before the main run; the table at the top of the runner is the place to adjust.
3. **Ref-comparison output** (9.5): implementer 5 should commit it in `round2/` with "compar" in its name (e.g.
   `ref_comparison_w5.json`); otherwise the manifest records it as not found.
4. **R1.final** must be written by hand on Drive; the notebooks show the format. Drive shortcuts to folders shared from the other
   account could not be tested here (Colab FUSE); copying the folders is the safe alternative the notebooks also offer.
5. **r2_ana's analysis scripts** should treat a missing per-block output that `session_done.json` lists as 'not run (…)' accordingly
   (the runner does not launch steps at final-missing blocks).
6. **Branch.** The notebooks clone `ccr-6fe02b21-est040` (round 1's branch); it must hold the freeze commit and the round-2 code commit.
7. **My process.** At the start I ran three read-only git commands (`git log --oneline`, `git status --short`, `git branch
   --show-current`) before noticing the no-git rule; nothing was changed. All later git use is the fake git of the tests. No Drive
   access, no round-1 output, no model weights loaded (MOCK uses the tokenizer and config only).
