# NOTES of implementer 9 (plan section 9.5 split (9)): runner, notebooks, manifest

Implemented against the frozen plan `results/ju_space/plans/abschnitt11_12_runde2_plan.md` (sha256 `5e476340…5edc2d`, freeze commit
`5b46e38cd800236d2d597044510652cbc48ad50d`, 2026-11-12T19:47:20Z) and its amendments file (Amendments 1-3; its sha256 goes into every
manifest). The first draft followed v2; the section "Changes from the v2-based draft" lists what was changed for the final text; the
section "Review fixes (13 Nov)" lists the changes made for the confirmed findings runner_steps-* and runner_gates-*.

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
python r2_run_session.py --compile-check                                                # every round-2 script compiles (this Python)
```

Exit codes: 0 done (`session_done.json`); 1 steps failed (rerun retries them); 2 G1/G2 or a model checksum failed; 3 time guard
(`session_partial.json`, rerun = continuation session); 4 start conditions not met (`start_check.json`, nothing ran).

## The step table (top of the runner; one Step = one script call; easy to edit)

| Session | Steps in order (BLOCK; required outputs relative to `<work>`) |
|---|---|
| r2_q1 | 5 processes: `axes` r2_axes.py (14; axes/r2_axes.json, r2_axes.pt, tok14.pt) · `unlearn` = `r2_unlearn_qwen.py … all` (14; the seven unlearn/{q3,q4a,q2,q4b,armA,q5,q6}.json, the .pt files optional; `settled` after axes; it reads round-1 files as the script decides, r1_reads None) · `gen_b14`, `gen_b<lo>`, `gen_b<hi>` r2_gen_loop.py (rgi/dyn_b<b>.json; after axes) |
| r2_q2 | one process per block, as the plan costs it and as r2_theta.py is built ("one process = one block"): `theta_b14` = `r2_theta.py … all` at BLOCK 14 (q0, checks, cells, turn2, power, fdt; requires theta/b14_q0.json, theta/checks.json, theta/b14_cells.json, theta/b14_turn2.pt, theta/b14_power.json, theta/b14_fdt.pt), then `theta_b<lo>`, `theta_b<hi>` (rsib, q0, cells, power, fdt; theta/b<b>_r2RSIb.json first). Each stage inside is resumable, so a rerun of the step resumes |
| r2_q3 | `geom_17` r2_rgi_geom.py (rgi/geom_meta_qwen3-1.7b.json, causal_…pt, dirs_…pt; local arrays `local/*_qwen3-1.7b.*`) · `stats_17` r2_rgi_stats.py (rgi/geom_stats_qwen3-1.7b.json) · `geom_06`, `stats_06` the same with `--model qwen3-0.6b` (only under G33). r2_rgi_stats for 0.6B rewrites the r_DiD block inside the 1.7B JSON (implementer 3): the runner checks existence only, never that file's hash |
| r2_t1 | r2_rgi_toy.py `r0`, `r1` (rgi_toy/r1/w0-w7.json), `cross` (rgi_toy/cross/w0-w4.json), `fam` (after cross), `forcing` |
| r2_t2 | r2_theta_toy.py `t1` (theta_toy/rows_t1.json, nets.json), `t2` (after t1; theta_toy/t2.json, rows_t2.json), `t5` (after t1), `rules` (= alias t4; after all; theta_toy/verdict.json) |
| r2_t3 | r2_unlearn_toy.py `t0`, `t1`, `t2`, `t3`, `t4`, `t6` (unlearn_toy/<stage>.json) |
| r2_t4 | r2_unlearn_toy.py `t5` |
| r2_pc | r2_pc_toy.py `t7` (pc_toy/t7.json), `cross_pc` (pc_toy/cross_pc/w0-w4.json); env R1PC = `<work>/r1`, R2T1 = `<work>/r2in/r2_t1`. The script's extra stage `gates` ("for the record") is not called: the plan's r2_pc list is t7 and cross_pc, and r2_report writes pc_toy/gates.json in r2_ana |
| r2_ana | `linearity` (unlearn/q1.json), `theta_verdict`, `rgi_verdict`, `unlearn_verdict` (after linearity), `report` (after all; round2_summary.json; optional r2_manifest.json, pc_toy/gates.json) |

Dependencies: `after` = must have finished successfully (its 'not run' makes the step 'not run'); `settled` = must be done or 'not
run' (its 'not run' does not pass on; a failed one blocks). Step order inside a session is packing only (7.4); `PACKING` in the
runner states the packing against 7.2 and goes into the manifest and `session_done.json`.

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

Self-test only (`--selftest`, never in a real run): the sessions that read round-1 PC files (r2_pc, r2_ana) add `R1_QUICK` =
`gpu*/pc/{jobs,models}/bp_{mup,sp}_W*_s*_K{1,12}_g0_wd{0,1}_quick.*`, pc_meta QUICK's names in the mock tree (the pc_mup_* patterns
match those already). The 7.3 lists of real runs are unchanged.

**Two accounts; account A is the hub** (operator's decision, 13 Nov). Account A's `dsse_runs` holds gpu1-gpu3; r2_q1, r2_q2, r2_q3,
r2_pc and r2_ana run there. Account B holds gpu4-gpu6; r2_t1-r2_t4 run there. B shares its whole `dsse_runs` with A, and the
operator adds it once to A's My Drive as a shortcut named `dsse_runs_B`. Every notebook has the parameter `ZWEITER_ORDNER =
"/content/drive/MyDrive/dsse_runs_B"` ("" = off) and, after mounting Drive and before the runner's copy-in, a cell "Zweiter Ordner"
that copies every folder the session reads (the builder's `folders(s)`: gpu1-gpu6 as needed or optional, and the r2_* sessions it
reads) from `ZWEITER_ORDNER` into `dsse_runs` when it is missing there and present in the second folder: `shutil.copytree` into a
hidden temporary name, then a rename (an interrupted copy never leaves a partial folder under the real name); an existing folder is
never overwritten. Beyond the request, a round-2 folder is copied only when that session is finished there (`session_done.json`
without `session_partial.json`) or listed in `NICHT_GELAUFEN` (r2_ana): otherwise a copy taken while B's session is still running
would block r2_pc / r2_ana for good, because an existing folder is never refreshed; a copy here that is unfinished while B's is
finished is reported with "delete it and rerun the cell". The cell prints in German what it copied (size, seconds), what was already
present, what is missing in both and what is not finished yet; without the shortcut it prints a one-line hint and continues; a
session that reads nothing says so. No runner change: the copied folders are ordinary folders of `dsse_runs`. The copy counts
against the 20-min clock (T0 is the first cell): r2_q1 / r2_q2 read only the version metadata of gpu4-gpu6 but copy the whole
folders (with the PC nets that r2_pc and r2_ana need later) on their first run, so the notebooks recommend that first run with
`NUR_SELBSTTEST = True`. `r1_finality.json` (R1.final) belongs in A's `dsse_runs` (every session that reads it runs in A; the cell
copies folders only); the notebooks say so.

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
   record. A step whose script exits 0 without its required outputs fails. (The table flag `gate_in_script` would accept exit 0
   alone; no step uses it now, because r2_theta.py writes theta/b<b>_power.json also when the power step is 'not needed' and
   r2_gen_loop.py writes rgi/dyn_b<b>.json also under the AUC gate.)
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
   rk files and summaries from the assigned shard (gpu1 block 14, gpu2 b_lo, gpu3 b_hi). A duplicated PC job is chosen once for its
   JSON and its model together (runner_gates-5): among the shards that hold both files (all candidates if none does) the shard whose
   pc_expected lists it, if several then the one with shard_done.json, then the newest `jobs/<name>.json`; both files come from that
   shard (one choice record per job, `lacks` names a missing file). rk files are copied only for final blocks of B3 (summary present).
8. **Blocks (R1-1).** b_lo/b_hi = the `rk/b<b>` folder of gpu2/gpu3 (blocks_chosen.json when the folder is missing; loops.json
   `_meta.block` cross-checked); G7, G8 (also recorded for q1/q2). G9 / OD-G3 (corrected, runner_steps-7 / runner_gates-4; the earlier
   reading "label order b_lo, 14, b_hi" contradicted the plan's clause, which is about ties): the order is sorted by (delta, block
   index), as `r2_rk.delta_order` does (re-implemented: the runner does not import torch); only when b_lo and b_hi agree within 1e-4
   and b_lo stands after b_hi are those two swapped; block 14 stays where its delta puts it. G9 lists every pair within 1e-4.
9. **G4 'not run' rule.** The runner marks a block step 'not run (round-1 output missing)' before launching it when its block is
   final-missing (R1.final) and the step reads that block (`r1_reads`): `axes` reads every B3 block and runs unless all are missing
   (r2_axes hosts the texts at another final block); the GEN loops read their block; dependants (`after`) of a 'not run' step are 'not
   run' too. The `unlearn` step (`r2_unlearn_qwen.py … all`) is left to the script (r1_reads None): it runs whenever axes is done or
   'not run' (`settled`), writes its own 'not run' JSONs for the stages that read a missing block and still runs C-Q6.
   A step that is already done stays 'skipped' whatever reason applies now (runner_steps-6); the reason goes into its note.
10. **R1.final** (`dsse_runs/r1_finality.json`, operator-written): `{"gpu4": {"runs": [{"start": …, "end": …}, …], "final": true}}`
    (also accepted under a top-level `"shards"` key). Final = `"final": true` and at least two runs. Recorded in every manifest (content,
    sha256, Drive time).
11. **r2_ana's "declared 'not run' after two attempts".** Notebook parameter `NICHT_GELAUFEN` (env R2_DECLARED_NOT_RUN); the runner
    accepts a declared session only for r2_ana (runner_gates-9; r2_pc keeps waiting for r2_t1 and says that a declaration does not
    apply to it) and only if its `session_attempts.json` on Drive lists at least two attempts. A run that waits for its start
    conditions is recorded there as an attempt with result 'waiting' (and synced), so that a wave-2 session that never starts can be
    declared after two waiting runs.
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
17. **Local arrays (r2_q3).** The runner records which files a step wrote under `<work>/local/` (merged with the files an earlier
    record listed that still exist; a successful run that wrote nothing new records the existing files of its own pattern
    `local/*_<model>.*`). While a consumer (`needed_by`) is not done, the step counts as not done if its recorded list is empty or a
    listed file is missing (runner_steps-10): on a new machine it reruns (cheap: r2_rgi_geom skips finished stages).
18. **References.** All 43 Appendix E files are looked up at `round2/ref/<path>` (status ok / moved / differs / missing), plus
    MISSING.txt, the sha256 of the two repository modules the shim uses (Appendix E), and the committed ref-comparison output of 9.5,
    `round2/tests/ref_comparison_w5.json` (written by `r2_toy_common.py selftest`; recorded by its path in every manifest, status
    'missing' if absent), plus any other file in `round2/`, `round2/tests/` or `round2/ref/` whose name contains "compar" or "refcheck".
    Its `module_sha256` entries are compared with the checkout (runner_gates-1 / runner_steps-8): r2_toy_common.py or r2_sim.py
    differing gives status 'stale (ports changed after the comparison)', a WARNING in the run (no stop: the plan lists no such gate)
    and a failing `--selftest`; pc_meta.py / pc_common.py are compared and listed in `changed` too.
19. **NT** defaults to min(8, CPUs - 2) (Colab A100: 8; round 1 used 4); RUN_DEVICE=cpu (code tests only) skips G2 and the model hashes.
20. **Inputs digest (r2_pc, r2_ana; runner_steps-4).** Every step record carries a sha256 over the copied inputs (kind, path, sha256).
    When an attempt of r2_pc or r2_ana copies different inputs, its finished steps are redone: their records, their recorded outputs
    (required and optional), r2_ana's mirror in `<r2in>/r2_ana` and r2_pc's resume cache `pc_toy/partial/` are removed in `<work>` and on
    Drive before the steps run; the attempt record, the manifest and `session_done.json` (`packing[*].invalidated`) list the steps.
21. **session_done.json** is removed in `<work>` and on Drive as soon as an attempt is going to run a step again, and when an attempt
    does not finish (runner_steps-5 / runner_gates-8). Wave 2 counts an upstream session as done only if `session_done.json` exists
    without a `session_partial.json` next to it and every output that its steps recorded is present in `dsse_runs/<session>` (r2_pc
    also needs one `r2_t1/rgi_toy/models/at5_*.pt`); presence, not sha256 (theta_toy/nets.json and the 1.7B geom_stats JSON are
    rewritten by later steps). `session_done.json` also logs the session's models (repo, revision, sha256, ok: the Qwen3-0.6B pin of
    OD-A6), the reference shim and comparison of the manifest, the code commits of the work folder and the packing (section 14,
    runner_gates-10).
22. **G3 without a commit (runner_gates-6).** A shard whose data is read but that has no code commit on record (no shard_done.json,
    no code_commits.txt) makes G3 not ok, labels every module 'round-1 code differs between shards (not checkable: no code commit of
    <shard>)' and lists it in `shards_without_commit`; the note 'no round-1 shard read' is written only when none is read.

## Self-tests (notebooks)

`--selftest` first compiles every round-2 script and the round-1 modules they use with the interpreter at hand (`compile_check`;
a syntax error stops it at once), then runs, on copies of a mock dsse_runs tree, the scripts' own tests (`LIB_SELFTESTS`: r2_q1
`r2_stimuli.py check`, `MOCK=1 r2_rk.py selftest`, `MOCK=1 r2_axes.py … selftest`, `MOCK=1 BLOCK=14|8 r2_gen_loop.py … selftest`;
r2_q2 stimuli and r2_rk; r2_q3 stimuli, `MOCK=1 r2_rgi_geom.py … selftest`, the same with `--model qwen3-0.6b`, then `MOCK=1
r2_rgi_stats.py … selftest` in the same folder; toy sessions and r2_ana `QUICK=1 NT=1 OMP_NUM_THREADS=1 r2_toy_common.py selftest
<dir>` and `r2_sim.py selftest`; all with DEVICE=cpu and R2_LIB_NT threads, default 1, as their authors ran them), then the session
itself under MOCK/QUICK through the runner on the session's device (wave 2: first the sessions it reads). The mock tree is built on
the CPU (`MOCK_DEVICE`), also when the session runs on CUDA (runner_steps-1 / rk-1): `rk_loop.py MOCK=1` pilot + all at blocks 14, 8,
20 (real runs; one process chain per block, R2_MOCK_NPAR at a time), rk_verdict summaries, the same fake dn_delta as
`r2_rk.fake_delta` (order 14 < 8 < 20), 8 `pc_meta.py QUICK=1` jobs at r2_pc_toy's QUICK widths and seeds under pc_meta's own
`_quick` names (pc_mup W 64 in gpu4 and W 128 in gpu3, bp_mup W 64 and 128 in gpu5, seeds 0 and 1; pc_expected lists the `_quick`
names; runner_steps-2 / runner_gates-3), `"device": "cpu"` in every mock shard_done.json and a `mock_fixture.json` (r2_rk /
r2_axes print the fixture's device next to the test's on a text mismatch). Markers `.mock_rk_cpu_ok` / `.mock_pc_quick_ok` make a
tree of an older runner rebuild. After an r2_pc session the self-test checks that `pc_toy/t7.json` has a width with a PC side and that
`pc_toy/cross_pc/w*.json` list a PC model (no pair: a WARNING); a stale reference comparison also fails it (after all other tests ran).
r2_rgi_stats' self-test reuses the 1.7B geometry of the first geom test (alone it takes 74 s); timings by implementer 3 on one
contended core: 39 s, 40 s, 54 s.
The notebook stores a marker per code commit in `dsse_runs/<session>`, so a rerun skips the self-test; the notebooks recommend a first
run with `NUR_SELBSTTEST = True`. r2_ana's full self-test (all sessions) is off by default (`SELBSTTEST = False`).

## Review fixes (13 Nov; findings runner_steps-* and runner_gates-*, verifiers' corrected fixes; refuted 9, 7, 11: no action)

- **runner_steps-1 (high; also rk-1 of the Qwen-library review).** The mock round-1 tree is built on the CPU (`MOCK_DEVICE = "cpu"`:
  rk_loop.py MOCK with DEVICE=cpu, as `r2_rk.mock_r1` does) also when the notebook starts the self-test with RUN_DEVICE=cuda; the
  library tests stay on the CPU, with the same thread count as the build (R2_LIB_NT, default 1). The session's MOCK steps on CUDA
  assert no equality with the mock texts. To keep the self-test short the three blocks are built in parallel (R2_MOCK_NPAR, default
  min(6, CPUs - 2)). Every mock shard_done.json carries `"device": "cpu"` and the tree a `mock_fixture.json` (coordinator's
  addition: r2_rk / r2_axes print the fixture's device next to the test's on a mismatch). Regression test `mockdev` (below).
- **runner_steps-2 / runner_gates-3.** Runner side only: the mock keeps pc_meta's `_quick` names, its jobs are at r2_pc_toy's QUICK
  widths 64 and 128 (pc_mup gpu4 / gpu3, bp_mup gpu5) and seeds 0, 1; the self-test (only) adds R1_QUICK copy patterns; the self-test
  fails when `pc_toy/t7.json` has no width that ran or `cross_pc` lists no PC model (no pair at all: a WARNING).
- **runner_steps-3 / runner_gates-2.** The seven r2_unlearn_qwen rows are one step `unlearn` = `r2_unlearn_qwen.py … all` after
  `axes`: required outputs the seven unlearn/<stage>.json, the .pt files optional; r1_reads left to the script; it starts when axes is
  done or 'not run' (`settled`, so C-Q6 is not dropped when axes is 'not run'); one retry like every step. Order: axes, unlearn,
  GEN loops at 14, b_lo, b_hi. 5 processes instead of the plan's 3 (about 60 s more instead of 240 s); recorded as `PACKING` in the
  manifest and in `session_done.json` (`packing_table`), and in the notebook text (9,9 / 13,9 / 18,8 min). 'all' from an empty folder
  in one process is the script author's resume path (finished stages skipped); I could test it only with stubs (no weights here), so
  the first notebook self-test (MOCK) is its first real run.
- **runner_steps-4.** Inputs digest per step record; r2_pc / r2_ana redo steps whose digest differs, removing their outputs, r2_ana's
  mirror and r2_pc's `pc_toy/partial/` in `<work>` and on Drive (reading 20).
- **runner_steps-5 / runner_gates-8.** session_done.json removed before steps rerun and when an attempt does not finish; wave-2 gate
  requires no session_partial.json and the recorded outputs present (reading 21).
- **runner_steps-6.** `step_state`: done first, then 'not run', then waiting; the same function drives the dry run.
- **runner_steps-7 / runner_gates-4.** G9 as OD-G3 and r2_rk.delta_order (reading 8, corrected).
- **runner_steps-8 / runner_gates-1.** Comparison staleness check (reading 18). Implementer 5 has since regenerated
  `tests/ref_comparison_w5.json` (sha256 f6e0ff21…cf1b; module entries r2_toy_common.py 3c141614…, r2_sim.py 69692014…, pc_meta.py
  b69e314a…, pc_common.py 2d72a474…): on the current checkout the check reports 'ok'.
- **runner_steps-10.** Local lists (reading 17).
- **runner_gates-5.** OD-G9 per PC job (reading 7). **runner_gates-6.** G3 without a commit (reading 22). **runner_gates-9.**
  Declarations for r2_ana only; waiting runs recorded as attempts (reading 11). **runner_gates-10.** session_done.json logs models,
  shim / comparison, code commits, packing (reading 21).
- **Additional requests.** (a) r2_q3's self-test runs r2_rgi_geom.py selftest for both models and r2_rgi_stats.py selftest (each under
  about a minute by implementer 3's timings; my own measurement of the geom test: 33 s). (b) The amendments file's sha256 is in every
  manifest (`freeze.amendments_sha256`, also `plan.amendments`); a missing file prints a WARNING (the session goes on). (c) r2_ana's
  rows match implementer 8's scripts and NOTES_w8: each of r2_linearity, r2_theta_verdict, r2_rgi_verdict, r2_unlearn_verdict and
  r2_report takes `<r1_dir> <r2_root> <out.json>` and writes unlearn/q1.json, theta_verdict.json, rgi_verdict.json,
  unlearn_verdict.json, round2_summary.json (r2_report also r2_manifest.json and pc_toy/gates.json next to it); no mismatch left.
  (d) `--compile-check` (also the first step of every `--selftest`) compiles all round-2 scripts and the round-1 modules they use;
  the test harness runs it under Python 3.11, 3.12 and 3.13 (28 files, all compile; the earlier PEP-701 f-string in r2_pc_toy.py is
  gone).
- **Accounts (operator's decision).** Hub A, `ZWEITER_ORDNER` and the cell "Zweiter Ordner" (section on round-1 folders above).

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
- After the scripts of implementers 2-7 appeared: r2_q2 runs one `r2_theta.py … all` per block (3 processes, as costed; the script's
  `run_all` follows the plan's order and every stage is resumable), the power output is required (the script always writes it);
  r2_t2's t1 / t2 steps require their own files (rows_t1.json; t2.json, rows_t2.json); r2_pc does not call the record-only `gates`.
- After worker 5's notes: `r2_sim.py selftest` joins `r2_toy_common.py selftest` (run as `QUICK=1 NT=1 OMP_NUM_THREADS=1`); the
  ref-comparison output is recorded by its committed path `round2/tests/ref_comparison_w5.json`.

## Tests and results

- `python test_runner.py [scenarios]` (scratchpad `r2_impl/w9/`; NT=1, one process at a time; stub scripts, fake git, fabricated
  dsse_runs; no round-1 output exists there). Fast scenarios `units dry q1 guard fail wait q3 wave2 integ cont`: 156 checks pass
  (52 s); the self-test scenario `selftest` (real rk_loop.py MOCK and pc_meta.py QUICK build the mock tree, stub session scripts): 16 checks
  pass (422 s). 172 checks in all on the final code. Covered: brace expansion, G7-G9 incl. ties, OD-G9 choices, the R1.final rule; dry runs of all nine
  sessions write nothing; full runs with step order, BLOCK and env, manifest (sha256, shard and Drive time of every copy, freeze
  block, Drive times of logs and summaries with freeze_before_first_summary both ways, G3 label, R1.meta, G47, references incl. the
  committed ref comparison), Drive sync (no r1/, r2in/, local/), read-only copies, resume, single-step rerun; time guard ->
  `session_partial.json` -> continuation `r2_q2b` -> done with packing log; retry once, blocked dependants, exit 0 without outputs =
  failure (r2_t4, GEN loop, `r2_theta.py all`); G4 wait, R1.final with one run / without final / valid; block 14 final-missing (only
  its readers 'not run'); G1 changed plan / not ancestor / other freeze commit, also in the self-test; G2 without an A100; r2_q3 local
  arrays on a new machine and G33; r2_pc (G5 by R1.final, r2_t1 inputs, R1PC / R2T1, R1 BP nets), r2_ana (declarations need two
  attempts, mirror, copy exclusions); integrity of copies; Drive errors not fatal; explicit continuation name; self-test with library
  tests on copies of the mock, a failing library test stops it.
- The scripts' own self-tests through the runner's `lib_selftests` with the real code: r2_stimuli check ok, r2_rk 35/35, r2_axes 13/13,
  r2_gen_loop 10/10 at blocks 14 and 8 (52 s on one core, on the runner's mock tree); r2_toy_common 70/70 (144 s) and r2_sim passed (3 s).
- Notebooks: nbformat-valid; the runner, the builder and every notebook code cell parse under Python 3.11, 3.12 and 3.13 (one PEP-701
  f-string was removed); the summary cell ran on test outputs (finished, partly 'not run', retried and waiting sessions).
## Open points

1. **Process count against the cost model.** 7.2 costs r2_q1 and r2_q2 as 3 processes each (one model load per block). r2_q2
   matches; r2_q1 now starts 5 (axes, unlearn 'all', three GEN loops): about 60 s of extra loads, 13.9 min nominal with setup, 18.8
   min with the plan's margin (plan: 12.9 / 17.3). The plan's 3 would need a block-14 driver calling r2_axes, r2_unlearn_qwen 'all' and
   the block-14 GEN loop in one process (another 60 s); not done, because three scripts' module state would share one process.
2. **Interfaces.** Checked against the scripts present at the end (r2_axes, r2_gen_loop, r2_rk, r2_stimuli, r2_theta, r2_unlearn_qwen,
   r2_rgi_geom, r2_rgi_stats, r2_rgi_toy, r2_theta_toy, r2_unlearn_toy, r2_pc_toy, r2_theta_verdict, r2_toy_common, r2_sim): stage names,
   options and outputs agree with the table (r2_theta.py writes theta/b<b>_power.json also when the power step is 'not needed', and its
   stage `all` runs the block's stages in the plan's order; r2_rgi_geom.py takes `--model qwen3-0.6b` after `<r2_dir>` and reads
   `<root>/models/<model>`; r2_pc_toy.py reads R1PC and R2T1). Not yet present when I finished: r2_linearity, r2_rgi_verdict,
   r2_unlearn_verdict, r2_report. For them the table follows 9.4 and these readings: out.json paths `<work>/unlearn/q1.json`,
   `theta_verdict.json`, `rgi_verdict.json`, `unlearn_verdict.json`, `round2_summary.json` (r2_report's `r2_manifest.json` and
   `pc_toy/gates.json` optional, next to it); every finished r2_ana output is mirrored into `<r2in>/r2_ana/` so that a later script
   finds e.g. C-Q1's q1.json under `<r2_root>/r2_ana/`. The notebook self-test catches a mismatch before the main run; the table at the
   top of the runner is the place to adjust.
3. **Ref-comparison output** (9.5): committed by implementer 5 at `round2/tests/ref_comparison_w5.json`, regenerated on 13 Nov
   (sha256 f6e0ff2160dda419c951a3ef6c3cdd1b9215823689fd66a17b097d21d603cf1b; its module entries match r2_toy_common.py 3c141614… and
   r2_sim.py 69692014…). It must be regenerated and committed whenever r2_toy_common.py or r2_sim.py change: the runner now reports a
   stale file (WARNING in every run, status 'stale …' in the manifest, failing self-test).
4. **R1.final** must be written by hand on Drive; the notebooks show the format. The shortcut `dsse_runs_B` and the copy cell could
   be tested only on local folders here (Colab's Drive FUSE, shortcut traversal and directory rename on Drive are untested); the
   copy keeps file times (copy2), which the manifest's Drive times then show for the copies.
5. **r2_ana's analysis scripts** should treat a missing per-block output that `session_done.json` lists as 'not run (…)' accordingly
   (the runner does not launch steps at final-missing blocks).
6. **Branch.** The notebooks clone `ccr-6fe02b21-est040` (round 1's branch); it must hold the freeze commit and the round-2 code commit.
7. **My process.** At the start I ran three read-only git commands (`git log --oneline`, `git status --short`, `git branch
   --show-current`) before noticing the no-git rule; nothing was changed. All later git use is the fake git of the tests. No Drive
   access, no round-1 output, no model weights loaded (MOCK uses the tokenizer and config only).
