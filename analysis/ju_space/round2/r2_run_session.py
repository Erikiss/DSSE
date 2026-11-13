"""Round 2 of README sections 11/12 on Colab: runs one session of the frozen plan (results/ju_space/plans/
abschnitt11_12_runde2_plan.md, sections 6, 7 and 9.5). The notebooks are built by notebooks/build_runde2_gpu_notebooks.py.

    python r2_run_session.py <session> <root> <work> [<drive_dir>] [--dry-run [--verbose]]
    python r2_run_session.py <session> <root> <selftest_dir> --selftest
    python r2_run_session.py <session> <root> <work> <drive_dir> --sync-only
    python r2_run_session.py --mock-r1 <root> <dsse_runs_dir> [rk,pc]

<session>    r2_q1 r2_q2 r2_q3 r2_t1 r2_t2 r2_t3 r2_t4 (wave 1), r2_pc r2_ana (wave 2). A continuation session (7.4) is the same
             notebook run again after the time guard stopped a session: the runner names it r2_q1b, r2_q1c, ... itself (an explicit
             'r2_q1b' is accepted too); it uses the same work and Drive folder, so finished steps are skipped.
<root>       as in round 1: models/qwen3-1.7b (+ models/qwen3-0.6b for r2_q3; models/verified.json written by the notebook),
             rsi_dl/persona/*.jsonl, ju_out/, dn_out/, gpu_pkg/rk_dirs.pt
<work>       local working directory of the session. <work>/r1/<shard>/... holds the copied round-1 files (read-only),
             <work>/r2in/<session>/... the copied round-2 inputs of wave 2, <work>/local/ local-only arrays; these three are never
             synced to Drive (r1 and r2in are copies of Drive files, recorded with sha256 in the manifest).
<drive_dir>  Google Drive dsse_runs/<session>; the round-1 folders are its siblings dsse_runs/gpu1 ... gpu6 (env R1_SRC overrides).

What a session does, in this order: gates G1 (plan freeze: the checkout's plan file must have FROZEN_PLAN_SHA256 and the freeze
commit must be an ancestor of the code commit) and G2 (A100); the start conditions of 7.3 (G4, G5 with R1.final = the operator's
dsse_runs/r1_finality.json, finished wave-1 sessions); the copy-in of the session's round-1 files (globs over gpu1-gpu6, duplicates by
OD-G9) and round-2 inputs; the session manifest (plan sha256, freeze commit hash and time, amendments sha256, code commit, R1 module
sha256 and G3, every copied file with source shard, sha256 and Drive modification time, the Drive modification times of
dsse_runs/gpu*/logs/*.log and gpu*/summary_b*.json next to the freeze time (os.stat only, section 14), R1.meta, R1.final, block set
and delta order (G7-G9), versions (G47), Qwen revisions and checksums, reference-file and ref-comparison sha256); the steps of the
session's table below in order, each skipped when its outputs exist (and the runner recorded its success), each followed by an
incremental, error-tolerant Drive sync and the time guard (7.4: at >= 20 min after the notebook start the session stops after the
current step, writes session_partial.json and is continued by a rerun of the notebook); finally session_done.json with the step
times and every attempt (packing changes, 7.4), after a check that the round-1 copies were not modified.

Exit codes: 0 session done; 1 steps failed (rerun the notebook: failed steps are retried, finished ones skipped); 2 gate G1 or G2
failed, a model checksum failed, or a usage error; 3 time guard (rerun the notebook: continuation session); 4 start conditions not
met (wait for round 1 or the wave-1 sessions).
Env: RUN_DEVICE (cuda; 'cpu' only for local code tests: then G2 and the model checks are skipped and recorded as such), NT,
R2_T0 (epoch seconds of the notebook's first cell; the time guard counts from it), R2_GUARD_S (1200), R2_STEP_TIMEOUT (3600),
R2_DECLARED_NOT_RUN (wave-1 sessions declared 'not run' after two attempts, for r2_ana; the runner checks the two attempts),
R1_SRC (round-1 source folder, default: the parent of <drive_dir>); self-test only: R2_MOCK_NPAR (parallel pc_meta jobs of the
mock tree), R2_MOCK_FAST (blocks 8 and 20 of the mock as patched copies of block 14); for code tests only: R2_REPO (checkout root),
R2_SCRIPTS (folder of the round-2 scripts), R2_GIT (git executable).
--selftest runs the scripts' own code tests (LIB_SELFTESTS) on copies of a mock dsse_runs tree, then the session (and for wave 2 the
sessions it reads) under MOCK=1 / QUICK=1 through this runner; nothing is synced to the real Drive.
R1.final (2.1, critique v2-12j) is written by the operator, never by a script: dsse_runs/r1_finality.json =
  {"gpu4": {"runs": [{"start": "2026-11-12T15:02Z", "end": "2026-11-12T15:24Z"}, {"start": ..., "end": ...}], "final": true}, ...}
A shard without shard_done.json (G5), or whose block summary is missing (G4), counts as final only with >= 2 runs and "final": true."""
import datetime, glob, hashlib, importlib.metadata, json, math, os, platform, re, shlex, shutil, signal, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))


# ======================================================================================================================================
# The steps of every session (plan 7.2 for content and order, 9.2-9.4 for the command lines and outputs). One Step = one script call.
# ======================================================================================================================================
class Step:
    """name      unique in the session; '{b}' is replaced by the block number
    block     None, '14', 'b_lo' or 'b_hi' (env BLOCK); b_lo / b_hi are the blocks that ran in round 1 (OD-G2)
    script    file in analysis/ju_space/round2/; args: its command line ({root} {r1} {work} {r2in} {b} are filled in)
    out       required outputs relative to <work> (space separated, glob patterns allowed). The step is done when the runner
              recorded its success (steps/<name>.json) and every output exists; then it is skipped (7.4).
    after     steps (comma separated) that must have finished successfully before (a 'not run' among them makes this step 'not run')
    settled   steps that must be done or 'not run' before (their 'not run' does not pass on; a failed one blocks this step)
    opt       optional outputs (hashed into the step record when present); quick_out: required outputs under MOCK / QUICK
    local     the step writes local-only arrays under <work>/local that the steps in needed_by read (True, or a glob of its files below
              <work>); they are never synced, so on a new machine the step runs again unless all of needed_by are done
    needs     'qwen06': runs only if Qwen3-0.6B was downloaded and its checksum holds (G33)
    gate_in_script  the script applies its own gate and may write no output; exit code 0 is then enough
    r1_reads  G4: a step at a block whose round-1 summary is final-missing is 'not run (round-1 output missing)' if it reads round-1
              files of that block ('block', the default for block steps); 'any': it reads several blocks and runs unless all are
              missing (the script skips the missing ones: env R2_MISSING_BLOCKS); None: it reads no round-1 file (C-Q6)"""

    def __init__(self, name, block, script, args, out, after="", opt="", quick_out=None, local=False, needed_by="", needs="", env=None,
                 gate_in_script=False, r1_reads="block", settled=""):
        self.name, self.block, self.script, self.args = name, block, script, args
        self.out, self.opt = out.split(), opt.split()
        self.quick_out = None if quick_out is None else quick_out.split()
        self.after, self.needed_by, self.settled = _names(after), _names(needed_by), _names(settled)
        self.local, self.needs, self.env, self.gate_in_script = local, needs, dict(env or {}), gate_in_script
        self.r1_reads = r1_reads      # G4: 'block' = reads round-1 files of its block; 'any' = of every block of B3; None = none


def _names(s):
    return [x.strip() for x in s.split(",") if x.strip()]


QW = "{root} {r1} {work}"            # Qwen scripts (9.2): <root> <r1_dir> <r2_dir> [<stage>]; env DEVICE, NT, MOCK, BLOCK
ANA = "{r1} {r2in} {work}/"          # analysis scripts (9.4): <r1_dir> <r2_root> <out.json>; <r2_root>/<session>/ = dsse_runs/<session>/
M17, M06 = "qwen3-1.7b", "qwen3-0.6b"
UNLEARN_STAGES = ("q3", "q4a", "q2", "q4b", "armA", "q5", "q6")      # r2_unlearn_qwen.py's stages, in its order


def _geom(m, extra, needs=""):
    """A1 states, A3 gradients, O7, O8 of one model, then A2-A5 statistics (r_DiD in the second model's statistics)."""
    tag = "17" if m == M17 else "06"
    return [Step(f"geom_{tag}", None, "r2_rgi_geom.py", QW + extra, f"rgi/geom_meta_{m}.json rgi/causal_{m}.pt rgi/dirs_{m}.pt",
                 local=f"local/*_{m}.*", needed_by=f"stats_{tag}", needs=needs),
            Step(f"stats_{tag}", None, "r2_rgi_stats.py", QW + extra, f"rgi/geom_stats_{m}.json", opt=f"rgi/geom_draws_{m}.npz",
                 after=f"geom_{tag}" + ("" if m == M17 else ",stats_17"), needs=needs)]


STEPS = {
    "r2_q1": [   # O2; then C-Q3, C-Q4a, C-Q2, C-Q4b (gated), arm A, C-Q5 pushes, C-Q6 in ONE process (r2_unlearn_qwen.py 'all': every
                 # stage whose inputs exist, dependants of a failed stage skipped, its own 'not run' JSONs, C-Q6 always; exit 1 if a
                 # stage failed; finished stages skipped on a second call); then the A6(b) GEN loops at 14, b_lo, b_hi (r2_gen_loop.py
                 # writes rgi/dyn_b<b>.json also when GA(b) closes the loop). 5 processes; plan 7.2 costs 3 (packing only, 7.4).
        Step("axes", "14", "r2_axes.py", QW, "axes/r2_axes.json axes/r2_axes.pt axes/tok14.pt", r1_reads="any"),
        Step("unlearn", "14", "r2_unlearn_qwen.py", QW + " all", " ".join(f"unlearn/{st}.json" for st in UNLEARN_STAGES),
             opt=" ".join(f"unlearn/{st}.pt" for st in UNLEARN_STAGES), settled="axes", r1_reads=None),
        *[Step("gen_b{b}", blk, "r2_gen_loop.py", QW, "rgi/dyn_b{b}.json", opt="rgi/dyn_b{b}.pt", after="axes") for blk in ("14", "b_lo", "b_hi")],
    ],
    "r2_q2": [   # Part B only; one process per block (r2_theta.py 'all' runs the block's stages in the plan's order, each resumable):
                 # block 14: B-Q0, code checks, cells (B-Q1, B-Q3, B-Q4), turn 2 (B-Q8), power step (B-Q6), FDT (B-Q7); each extra
                 # block: the second RSI loop r2RSIb (critique v2-3; a loop that did not run its 4 rounds is recorded by the script, G49),
                 # B-Q0, cells incl. reserve 3, power step, FDT
        Step("theta_b{b}", "14", "r2_theta.py", QW + " all",
             "theta/b{b}_q0.json theta/checks.json theta/b{b}_cells.json theta/b{b}_turn2.pt theta/b{b}_power.json theta/b{b}_fdt.pt",
             opt="theta/b{b}_q0.pt theta/b{b}_cells.pt theta/b{b}_turn2.json theta/b{b}_power.pt"),
        *[Step("theta_b{b}", blk, "r2_theta.py", QW + " all",
               "theta/b{b}_r2RSIb.json theta/b{b}_q0.json theta/b{b}_cells.json theta/b{b}_power.json theta/b{b}_fdt.pt",
               opt="theta/b{b}_r2RSIb.pt theta/b{b}_q0.pt theta/b{b}_cells.pt theta/b{b}_power.pt") for blk in ("b_lo", "b_hi")],
    ],
    "r2_q3": _geom(M17, "") + _geom(M06, " --model " + M06, needs="qwen06"),   # A1-A5 for Qwen3-1.7B, then the Qwen3-0.6B arm
    "r2_t1": [   # AT0; AT2 (with AT1, AT3); AT5 among BP nets; AT4 (its 'other init' nets come from AT5, critique 17); AT6
        Step("r0", None, "r2_rgi_toy.py", "{work} r0", "rgi_toy/r0.json"),
        Step("r1", None, "r2_rgi_toy.py", "{work} r1", " ".join(f"rgi_toy/r1/w{ws}.json" for ws in range(8)),
             quick_out="rgi_toy/r1/w*.json", after="r0"),
        Step("cross", None, "r2_rgi_toy.py", "{work} cross", " ".join(f"rgi_toy/cross/w{ws}.json" for ws in range(5)),
             quick_out="rgi_toy/cross/w*.json"),
        Step("fam", None, "r2_rgi_toy.py", "{work} fam", "rgi_toy/fam.json", after="cross"),
        Step("forcing", None, "r2_rgi_toy.py", "{work} forcing", "rgi_toy/forcing.json"),
    ],
    "r2_t2": [   # B-T1; B-T2 (after t1); B-T5 (after t1); B-T4 and the theta toy rules (stage 'rules', alias 't4')
        Step("t1", None, "r2_theta_toy.py", "{work} t1", "theta_toy/rows_t1.json theta_toy/nets.json"),
        Step("t2", None, "r2_theta_toy.py", "{work} t2", "theta_toy/t2.json theta_toy/rows_t2.json", after="t1"),
        Step("t5", None, "r2_theta_toy.py", "{work} t5", "theta_toy/t5.json", after="t1"),
        Step("rules", None, "r2_theta_toy.py", "{work} rules", "theta_toy/verdict.json", after="t1,t2,t5"),
    ],
    "r2_t3": [Step(f"c{st}", None, "r2_unlearn_toy.py", "{work} " + st, f"unlearn_toy/{st}.json")      # C-T0 ... C-T4, C-T6
              for st in ("t0", "t1", "t2", "t3", "t4", "t6")],
    "r2_t4": [Step("ct5", None, "r2_unlearn_toy.py", "{work} t5", "unlearn_toy/t5.json")],               # C-T5
    "r2_pc": [   # C-T7 (PC and BP sides at W 128 and 256), AT5 pairs with a PC net at every PC width (r2_t1's BP nets in
                 # {r2in}/r2_t1); no gate decides what runs: G-meta and G-PC are decided in r2_ana (critique v2-11)
        Step("t7", None, "r2_pc_toy.py", "{work} t7", "pc_toy/t7.json"),
        Step("cross_pc", None, "r2_pc_toy.py", "{work} cross_pc", " ".join(f"pc_toy/cross_pc/w{ws}.json" for ws in range(5)),
             quick_out="pc_toy/cross_pc/w*.json"),
    ],
    "r2_ana": [  # all statistics and verdicts, then the report (each output is mirrored into <r2in>/r2_ana for the later scripts)
        Step("linearity", None, "r2_linearity.py", ANA + "unlearn/q1.json", "unlearn/q1.json"),
        Step("theta_verdict", None, "r2_theta_verdict.py", ANA + "theta_verdict.json", "theta_verdict.json"),
        Step("rgi_verdict", None, "r2_rgi_verdict.py", ANA + "rgi_verdict.json", "rgi_verdict.json"),
        Step("unlearn_verdict", None, "r2_unlearn_verdict.py", ANA + "unlearn_verdict.json", "unlearn_verdict.json", after="linearity"),
        Step("report", None, "r2_report.py", ANA + "round2_summary.json", "round2_summary.json", opt="r2_manifest.json pc_toy/gates.json",
             after="linearity,theta_verdict,rgi_verdict,unlearn_verdict"),
    ],
}

# Copy lists (7.3; globs relative to dsse_runs/; 'gpu*' = gpu1 ... gpu6; {a,b} = alternatives; a name without a shard folder lies in
# dsse_runs/ itself: r1_finality.json = R1.final, the operator's record of the round-1 notebook runs, 2.1)
R1_RK = ("gpu*/rk/b*/{state.pt,loops.json,pilot.json,pilot@1.5.json,k5.json,k6.json,manip@*.json}", "gpu*/summary_b*.json",
         "gpu2/{dn_delta.json,blocks_chosen.json}", "gpu3/{dn_delta.json,blocks_chosen.json}", "gpu*/{shard_done.json,code_commits.txt}",
         "r1_finality.json")
R1_PC = ("gpu*/pc/jobs/pc_mup_*.json", "gpu*/pc/models/pc_mup_*.pt",
         "gpu*/pc/jobs/bp_mup_W{128,256}_s*_K12_g0_wd1.json", "gpu*/pc/models/bp_mup_W{128,256}_s*_K12_g0_wd1.pt",   # C-T7's BP side
         "gpu*/pc_summary_gpu*.json", "gpu*/pc_expected_gpu*.json", "gpu*/pc_missing_gpu*.json", "gpu*/{shard_done.json,code_commits.txt}",
         "r1_finality.json")
R1_G46 = ("gpu*/pc/jobs/bp_{mup,sp}_W*_s*_K{1,12}_g0_wd{0,1}.json",        # G46: the R1 BP cells that round 2 retrains (AT2, AT5,
          "gpu*/pc/models/bp_{mup,sp}_W*_s*_K{1,12}_g0_wd{0,1}.pt")         # B-T1/B-T2, C-T nets): every BP main cell without regulariser
R1_QUICK = ("gpu*/pc/jobs/bp_{mup,sp}_W*_s*_K{1,12}_g0_wd{0,1}_quick.json",   # self-test only (r2_pc, r2_ana): pc_meta QUICK's names in
            "gpu*/pc/models/bp_{mup,sp}_W*_s*_K{1,12}_g0_wd{0,1}_quick.pt")    # the mock tree, which r2_pc_toy looks for under QUICK (C-T7's
                                                                            # BP side; pc_mup_* matches already) and r2_report under G46
R2_ALL = "*"                                                                # r2_ana: every output of the other sessions (see plan_r2)
WAVE1 = ("r2_q1", "r2_q2", "r2_q3", "r2_t1", "r2_t2", "r2_t3", "r2_t4")
SESSIONS = {   # start: G4 / G5 / the round-2 sessions that must be done first; shards: round-1 folders read (meta: version metadata only)
    "r2_q1": dict(wave=1, kind="qwen", r1=R1_RK, r2=(), start=("G4",), upstream=(), shards=("gpu1", "gpu2", "gpu3"),
                  meta_shards=("gpu4", "gpu5", "gpu6"), models=(M17,)),
    "r2_q2": dict(wave=1, kind="qwen", r1=R1_RK, r2=(), start=("G4",), upstream=(), shards=("gpu1", "gpu2", "gpu3"),
                  meta_shards=("gpu4", "gpu5", "gpu6"), models=(M17,)),
    "r2_q3": dict(wave=1, kind="qwen", r1=(), r2=(), start=(), upstream=(), shards=(), meta_shards=(), models=(M17, M06)),
    **{s: dict(wave=1, kind="toy", r1=(), r2=(), start=(), upstream=(), shards=(), meta_shards=(), models=())
       for s in ("r2_t1", "r2_t2", "r2_t3", "r2_t4")},
    "r2_pc": dict(wave=2, kind="toy", r1=R1_PC, r2=("r2_t1/rgi_toy/models/at5_*.pt",), start=("G5",), upstream=("r2_t1",),
                  shards=("gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"), meta_shards=(), models=()),
    "r2_ana": dict(wave=2, kind="ana", r1=R1_RK + R1_PC + R1_G46, r2=(R2_ALL,), start=("G4", "G5"), upstream=WAVE1 + ("r2_pc",),
                   shards=("gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"), meta_shards=(), models=()),
}
STEP_ENV = {"r2_pc": dict(R1PC="{r1}", R2T1="{r2in}/r2_t1")}     # 9.3: R1PC = folder with the copied R1 PC files
RERUN_ON_INPUT_CHANGE = ("r2_pc", "r2_ana")   # their finished steps are redone when the copied inputs changed since they ran
PARTIAL_DIRS = {"r2_pc": ("pc_toy/partial",)}  # the scripts' own resume caches, cleared with the steps they belong to
PACKING = {   # process packing against plan 7.2 (7.4: packing changes no measurement); written into session_done.json
    "r2_q1": "5 processes (r2_axes; r2_unlearn_qwen 'all' = q3, q4a, q2, q4b, armA, q5, q6; r2_gen_loop at 14, b_lo, b_hi); 7.2 costs 3 "
             "(one model load per block): about 60 s more",
    "r2_q2": "3 processes (r2_theta 'all' per block), as in 7.2",
    "r2_q3": "4 processes (r2_rgi_geom and r2_rgi_stats per model); 7.2 costs 2 model processes (the statistics load no model)",
}

# ======================================================================================================================================
# Constants of the freeze and of the inputs
# ======================================================================================================================================
FROZEN_PLAN_SHA256 = "5e476340dfdb1d827582c869f2dfc08e32ed8e9a9dbebf7642acceb5bb5edc2d"   # the frozen plan (section 14, G1)
FREEZE_COMMIT = "5b46e38cd800236d2d597044510652cbc48ad50d"   # the commit that added the plan file (checked against git, G1)
FREEZE_TIME = "2026-11-12T19:47:20Z"                         # its time (UTC); git's commit time is recorded next to it
PLAN_REL = "results/ju_space/plans/abschnitt11_12_runde2_plan.md"
AMEND_REL = "results/ju_space/plans/abschnitt11_12_runde2_plan_amendments.md"
R1_MODULES = ("rk_loop.py", "pc_meta.py", "pc_common.py", "ufo_common.py", "rsi_statements.py")        # 2.1 R1 code freeze
SHARDS = ("gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6")
QWEN06 = dict(repo="Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca",           # HF API, 12 Nov 2026 (OD-A6)
              sha256={"model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
                      "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"})
REF_SHA256 = {   # Appendix E: reference files copied into analysis/ju_space/round2/ref/ in the freeze commit
    "rgi_design/rgi_lib.py": "62ecb0ff81d715d9c8be38cfa51c8491d1a94e0a26449c7ef2c55a3eedc370ff",
    "rgi_design/rgi_pilot.py": "b70181ee36f9c510e93731d6a23bc5b623166b2f2e3ac54410e64c001851fc28",
    "rgi_design/rgi_pilot2.py": "85123c9f28884bf33e597e6c01e5dc8948d7d46cb74664932ad62234bcbafc2d",
    "rgi_design/lattice.py": "5c804eddcc7ebde47154a68ba450f84d2a42de0788c4bbee62b3e9aa0d4f89e9",
    "rgi_design/recipe_depth.py": "f647d2786658e7665777d3b1b1427ac353ce34c994838330f7d44dc859ad7aba",
    "rgi_design/rgi_pool.py": "7bdb7aeb8a82749c7a2e1892c49c10403a922d2bfa06b1f09800abd39f728884",
    "rgi_design/rgi_long.py": "17de2a1639d50637347824a1ac96cb53c733caef24b9611e1edf92f624e2c472",
    "rgi_design/rgi_bneck.py": "f1fa4f17711f181b6e398cb5ae60a70e65148378776e351b20d0c8f764147e8e",
    "rgi_design/crit/sham.py": "b013d896ba2101418393d67b2b30d72723b127acc538a28706e851652b4a54ca",
    "rgi_design/crit/sham2.py": "0e6571f73039053aa64be60f15d4698d029fc741629d8b02ffe91ef87b4e38f7",
    "rgi_design/crit/sham_s.py": "4735b6c0a199aaa009a2023d11f541bdf1b349c051a3fa66a0678a21bcb793cf",
    "rgi_design/crit/sham.json": "01a4cac66a9e7fb5996b0243e23265716c3187b0b1391ef6fb52a3d57bdcf797",
    "rgi_design/crit/family.py": "067fb3b5a33dec36682859fdabe121dd88078e9280c257d09b8632a95e074f2b",
    "rgi_design/crit/family2.py": "3023867809ddd39ee9686455a8bd79fbe504555a4e5520412676ee70aff4946c",
    "rgi_design/crit/family.json": "b876150c4c354197cb0cfeab8173135a9a6a37f9b26e761d7b96d00f929dd0ce",
    "rgi_design/crit/sim_reg.py": "c04e04ed3484d4c771ac33290183c389fc198535f9f12d768700e3bec62eb2d4",
    "rgi_design/crit/sim_reg2.py": "6f49cfa43b155b45f3d582d46456fae77e8867027157bfc75baf626fb42311c9",
    "rgi_design/crit/tokfact3.py": "6184aacd7095d2632bb79e847b453d0d571bf8e503643252cc7284de176b28ac",
    "theta_design/sim_gain.py": "5dfb1c7648305115822aa6b32a3c5c809e06df9433d52b383115bd8ca9cdcc86",
    "theta_design/simA.json": "d9cfa0ac239918b4ef14b8c9ad427d7af30c9e4f523ef23aa947107e2157d64e",
    "theta_design/simA_cellN.json": "2d6aed2e18da5b534efbdbcbbe7922bdc6d49a1cc09b78ca6ef41652e69c1a6b",
    "theta_design/crit_bulk.py": "3edd6d263034a6bb150103261206e3cb82fc540d5ea9edfd6e17f7c91dd040c0",
    "theta_design/crit_bulk.json": "f3163d22f1d6e3aa3071896ab4c767b9dbdce66db5e2644edc8dceafe4e34a31",
    "theta_design/crit_tline3.py": "e9b6e8d9a46c020046b7cd48377d4eff0b522ea5640e0875420c2f917990ebc9",
    "theta_design/crit_tline3.json": "473aeab169878fc55a063ca92257ab40bcc10ca011702c4b0efed50b0844e2b1",
    "theta_design/sim_toy_bracket.py": "60d4a31c135a82a5762def18083f6925ead1eaa53700a12e699155002338234e",
    "theta_design/simC2.json": "551aaf054c197b6daaef15a8e24aebe9ff0fa5c1a51d5305b7feaa74b15b9460",
    "unlearn_design/impose.py": "353302dcb2f2af8ae9ef41927d2a87053264dd78a97575ac8eaac302ed16d197",
    "unlearn_design/impose2.py": "cda1550cb6bce13bcaf54862bdf7e2a086e73cedda30a102ae3e72a727dd27cc",
    "unlearn_design/pilot.py": "c874cd03dc128ac491c800542db0792c2db30040f723d3dd9e9244e54543fe45",
    "unlearn_design/critique/c1_structure.py": "8ac1162308b6e6f89f5941024bff52202c68871e6cf3e93feeb202a304f7bba8",
    "unlearn_design/critique/c2_pring.py": "a9d22d4f82c6a06d097135e132c06a485fd2fdf9c50cb39d91b9cc02612a79ee",
    "unlearn_design/critique/c3_angle.py": "ee6f1cc4f64d4000fbaae7b43bc275c5a29344f002ab09cbd407eb6569dbe4bb",
    "unlearn_design/critique/c4_traj.py": "1a2aa3d292d8d103839fa65ee8aa04dbb88d71cb2d0cd2f97acfdf28f1345bb5",
    "unlearn_design/critique/c5_twofam.py": "b0e5fbab2ab427f125b1c15c6fa7169b9db7ab901e0ec2abc52b17614be7ffd3",
    "unlearn_design/critique/c6_timing.py": "10aa781c66a1d01e5023b9f0c07317563578ea4f003dacb8e56378bc973800a0",
    "rgi_design/design_readable.txt": "f115380456c5b16d738e28b5ef4f3f88eeab37bac37ff20877065a202d0a8893",
    "theta_design/design_readable.txt": "2330b6abedab19ebdfc852bc58e66e85a7da7ec376eac0421b75597c11f742fb",
    "unlearn_design/design_readable.txt": "74013dd312cdebf06a3a0fa85195d43713f607ffc1d0f4a47340d2659e89215f",
    "round2/plan_round2.md": "6b3a91ce96957e1468c2ef0ef4566c5053f527e77467b347da4776f7c0a480f3",
    "round2/plan_round2_critique.md": "96050e4f47f9e1b7f7f408c96ddbb77d7edaed33cdf2cde14c174be81c6e0b0e",
    "round2/plan_round2_v2.md": "32c0832ca0e40f3ac766510246c6ffe67f77ca02056fa5562022b29f6a914b17",
    "round2/plan_round2_v2_critique.md": "6e5259e2d904caaf9d80ab01bb7fe4f872e6570a952f2f4cc0db1730221d282a",
}
SHIM_SHA256 = {   # Appendix E: the repository's round-1 files that the reference shim (9.5) uses in place of the scratchpad copies
    "analysis/ju_space/pc_meta.py": "b69e314a2ca442ba56e9e31916578d718f9fdee4ec4bee329d3aa358a27bc0ae",
    "analysis/ju_space/pc_common.py": "2d72a47405d656d4699dbaf38bc8a00bea24fbcb1121f777d977a1d2ffbc476f",
}
REF_COMPARISON = "analysis/ju_space/round2/tests/ref_comparison_w5.json"     # 9.5: committed output of r2_toy_common.py selftest
REF_COMPARISON_PORTS = ("r2_toy_common.py", "r2_sim.py")                     # its module_sha256 must match these files
REF_COMPARISON_GLOBS = ("ref_comparison*", "ref_compare*", "ref_check*", "tests/*compar*", "ref/*comparison*", "ref/*compare*")
GUARD_S_DEFAULT = 1200.0             # 7.4 / G48: 20 min
EXIT_DONE, EXIT_FAILED, EXIT_GATE, EXIT_GUARD, EXIT_WAIT = 0, 1, 2, 3, 4
NOT_SYNCED = ("local", "r1", "r2in")  # top-level folders of <work> that never go to Drive
PY = sys.executable

REPO = os.path.abspath(os.environ.get("R2_REPO") or os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
AJS = os.path.join(REPO, "analysis", "ju_space")
SCRIPTS = os.path.abspath(os.environ.get("R2_SCRIPTS") or HERE)
REF_DIR = os.path.join(AJS, "round2", "ref")
CHECKSUMS = os.path.join(AJS, "gpu", "data", "checksums.json")
GIT = os.environ.get("R2_GIT", "git")


# ======================================================================================================================================
# Small helpers
# ======================================================================================================================================
def iso(t=None):
    return datetime.datetime.fromtimestamp(time.time() if t is None else t, datetime.timezone.utc).isoformat(timespec="seconds")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load_json(p):
    """The parsed file, or None if it is missing or unreadable (a half-written file never counts)."""
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def dump_json(obj, p):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(p + ".tmp", p)


def read_text(p):
    try:
        with open(p) as f:
            return f.read()
    except OSError:
        return None


def expand(patterns):
    """Brace expansion: 'a/{x,y}.json' -> ['a/x.json', 'a/y.json'] (glob has no braces)."""
    out = []
    for p in patterns:
        m = re.search(r"\{([^{}]*)\}", p)
        if not m:
            out.append(p)
            continue
        out += expand([p[:m.start()] + alt.strip() + p[m.end():] for alt in m.group(1).split(",")])
    return out


def rglob(base, pattern):
    """Files under base matching a relative glob pattern ('**' allowed), sorted, as absolute paths."""
    return sorted(p for p in glob.glob(os.path.join(glob.escape(base), pattern), recursive=True) if os.path.isfile(p))


def git(*args, timeout=60, binary=False):
    """(returncode, stdout) of a git command in the checkout; (None, error) if git cannot run."""
    try:
        r = subprocess.run([GIT, "-C", REPO, *args], capture_output=True, text=not binary, timeout=timeout)
        return r.returncode, r.stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, str(e)


def git_head():
    rc, out = git("rev-parse", "HEAD")
    return out.strip() if rc == 0 and out.strip() else "unknown"


def version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def split_session(name):
    """'r2_q1' -> ('r2_q1', None); 'r2_q1b' -> ('r2_q1', 'r2_q1b') (an explicit continuation name)."""
    if name in SESSIONS:
        return name, None
    if len(name) > 1 and name[:-1] in SESSIONS and name[-1] in "bcdefghijklmnopqrstuvwxyz":
        return name[:-1], name
    raise SystemExit(f"unknown session {name!r}; sessions: {', '.join(SESSIONS)}")


LOGF = []


def say(msg):
    line = f"[r2 {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    for f in LOGF:
        try:
            f.write(line + "\n"); f.flush()
        except OSError:
            pass


# ======================================================================================================================================
# Drive sync (round-1 pattern: incremental, error-tolerant; here per file, atomic by rename, without local/, r1/, r2in/)
# ======================================================================================================================================
def sync(work, drive):
    """Copy new or changed files of the work directory to Drive. A Drive error is reported and skipped, never fatal."""
    if not drive:
        return 0
    n = err = 0
    for d, dirs, files in os.walk(work):
        rel = os.path.relpath(d, work)
        if rel == ".":
            dirs[:] = [x for x in dirs if x not in NOT_SYNCED]
        dirs.sort()
        for f in sorted(files):
            if f.endswith((".tmp", ".r2tmp")):
                continue
            src, dst = os.path.join(d, f), os.path.normpath(os.path.join(drive, rel, f))
            try:
                st = os.stat(src)
                if os.path.exists(dst):
                    dt = os.stat(dst)
                    if dt.st_size == st.st_size and dt.st_mtime >= st.st_mtime - 1:
                        continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst + ".r2tmp"); os.replace(dst + ".r2tmp", dst); n += 1
            except OSError as e:
                err += 1
                if err <= 5:
                    say(f"[sync] Drive error, continuing without this copy: {e}")
    if err > 5:
        say(f"[sync] {err} Drive errors in total")
    return n


# ======================================================================================================================================
# Round-1 inputs: blocks (OD-G2, G7-G9), start conditions (G4, G5), copy plan (7.3, OD-G9)
# ======================================================================================================================================
RK_RE = re.compile(r"^rk/b(\d+)/")
SUM_RE = re.compile(r"^summary_b(\d+)\.json$")
JOB_RE = re.compile(r"^pc/(?:jobs|models)/(.+)\.(?:json|pt)$")
PER_SHARD_RE = re.compile(r"^(shard_done\.json|code_commits\.txt|dn_delta\.json|blocks_chosen\.json|pc_(summary|expected|missing)_gpu\d\.json)$")


def declared(var):
    return [x.strip() for x in os.environ.get(var, "").replace(";", ",").split(",") if x.strip()]


def find_r1(src, rel, prefer=None):
    """The shards that hold <shard>/<rel> (located by name across gpu1-gpu6, OD-G9), the preferred shard first."""
    hits = [s for s in SHARDS if os.path.isfile(os.path.join(src, s, rel))]
    return sorted(hits, key=lambda s: (s != prefer, s))


def r1_blocks(src):
    """OD-G2: b_lo, b_hi = the blocks that ran (rk/b<b> of gpu2 and gpu3, cross-checked with loops.json '_meta'.block); G7 (the two
    blocks_chosen.json agree), G8 (the two dn_delta.json copies agree to 1e-6), G9 (order b(1) < b(2) < b(3) by the last-token delta
    of gpu2's dn_delta.json; ties within 1e-4 broken by the round-1 label: b_lo before 14 before b_hi). Also R1-1 of section 8."""
    out = dict(labels=[], notes=[])
    ch = {s: load_json(os.path.join(src, s, "blocks_chosen.json")) or {} for s in ("gpu2", "gpu3")}
    out["blocks_chosen"] = {s: dict(b_lo=c.get("b_lo"), b_hi=c.get("b_hi")) for s, c in ch.items()}
    for shard, role in (("gpu2", "b_lo"), ("gpu3", "b_hi")):
        ran = sorted(int(m.group(1)) for d in glob.glob(os.path.join(glob.escape(os.path.join(src, shard)), "rk", "b*"))
                     if os.path.isdir(d) and (m := re.fullmatch(r"b(\d+)", os.path.basename(d))))
        chosen = ch[shard].get(role)
        b = ran[0] if len(ran) == 1 else (chosen if chosen in ran else (ran[0] if ran else chosen))
        if len(ran) > 1:
            out["notes"].append(f"{shard}/rk holds blocks {ran}; {role} = {b}")
        if b is not None and not ran:
            out["notes"].append(f"{shard}/rk/b{b} missing (block not run yet); {role} from blocks_chosen.json")
        meta = (load_json(os.path.join(src, shard, "rk", f"b{b}", "loops.json")) or {}).get("_meta", {}) if b is not None else {}
        if meta.get("block") is not None and meta.get("block") != b:
            out["notes"].append(f"{shard}/rk/b{b}/loops.json says block {meta.get('block')}")
            out["labels"].append("block of loops.json differs from its folder")
        out[role] = b
    out["b14"] = 14
    c2, c3 = out["blocks_chosen"]["gpu2"], out["blocks_chosen"]["gpu3"]
    out["G7"] = dict(ok=c2 == c3, gpu2=c2, gpu3=c3)
    if not out["G7"]["ok"]:
        out["labels"].append("block choice inconsistent between shards")
    if (c2.get("b_lo"), c2.get("b_hi")) != (out["b_lo"], out["b_hi"]) and None not in (out["b_lo"], out["b_hi"]):
        out["notes"].append(f"the blocks that ran ({out['b_lo']}, {out['b_hi']}) differ from gpu2/blocks_chosen.json ({c2})")
    d2, d3 = (load_json(os.path.join(src, s, "dn_delta.json")) for s in ("gpu2", "gpu3"))
    dl = lambda d, l: (((d or {}).get("blocks") or {}).get(str(l)) or {}).get("last", {}).get("delta")
    if d2 and d3:
        diffs = [abs(dl(d2, l) - dl(d3, l)) for l in range(28) if dl(d2, l) is not None and dl(d3, l) is not None]
        out["G8"] = dict(ok=bool(diffs) and max(diffs) <= 1e-6, max_abs_diff=max(diffs) if diffs else None)
    else:
        out["G8"] = dict(ok=None, note="a dn_delta.json copy is missing")
    B3 = [b for b in (14, out["b_lo"], out["b_hi"]) if b is not None]
    out["B3"] = B3
    if d2 and all(dl(d2, b) is not None for b in B3):
        # OD-G3 / G9, the rule of r2_rk.delta_order (re-implemented: the runner does not import torch): sort by (delta, block); if
        # b_lo and b_hi agree within 1e-4 the round-1 label decides (b_lo below b_hi); block 14 stays where its delta puts it
        val = {b: dl(d2, b) for b in B3}; lo, hi = out["b_lo"], out["b_hi"]
        order = sorted(B3, key=lambda b: (val[b], b))
        if lo in order and hi in order and abs(val[lo] - val[hi]) <= 1e-4 and order.index(lo) > order.index(hi):
            i, j = order.index(lo), order.index(hi); order[i], order[j] = order[j], order[i]
        ties = [[a, c] for k, a in enumerate(order) for c in order[k + 1:] if abs(val[a] - val[c]) <= 1e-4]
        out["delta_last"] = {str(b): val[b] for b in B3}
        out["order"] = order
        out["G9"] = dict(ok=not ties, ties=ties, rule="sorted by (delta, block); b_lo before b_hi when they tie (OD-G3)")
    else:
        out["order"], out["G9"] = None, dict(ok=None, note="gpu2/dn_delta.json missing or incomplete")
    return out


def r1_final(src):
    """R1.final: the operator's dsse_runs/r1_finality.json (2.1; never written by a script), or None."""
    p = os.path.join(src, "r1_finality.json") if src else None
    rec = load_json(p) if p else None
    if isinstance(rec, dict) and isinstance(rec.get("shards"), dict):
        rec = rec["shards"]
    return rec if isinstance(rec, dict) else None


def final_by_record(fin, shard):
    """(final?, number of recorded runs): a shard counts as final by R1.final with >= 2 notebook runs and 'final': true."""
    e = (fin or {}).get(shard) or {}
    runs = e.get("runs") if isinstance(e, dict) else None
    n = len(runs) if isinstance(runs, list) else 0
    return bool(isinstance(e, dict) and e.get("final") is True and n >= 2), n


R1_FINAL_HINT = ("if it stays missing, record the shard's notebook runs in dsse_runs/r1_finality.json with \"final\": true "
                 "(at least two runs; section 2.1)")


def gate_G4(src, blk, fin):
    """Start condition of r2_q1, r2_q2, r2_ana: sum[14], sum[b_lo], sum[b_hi], gpu2/dn_delta.json, gpu2 and gpu3 blocks_chosen.json
    exist. A summary still missing (block 14's included, critique v2-12s) when R1.final lists two notebook runs of its shard and
    'final': true lets the session start; every part at that block that reads a round-1 file of the block is then 'not run (round-1
    output missing)'."""
    rec = dict(ok=True, waiting=[], final_blocks=[], missing_blocks=[], summaries={})
    for rel in ("gpu2/dn_delta.json", "gpu2/blocks_chosen.json", "gpu3/blocks_chosen.json"):
        if not os.path.isfile(os.path.join(src, rel)):
            rec["ok"] = False; rec["waiting"].append(f"{rel} missing" + (f" (folder dsse_runs/{rel[:4]} not found)" if not os.path.isdir(
                os.path.join(src, rel[:4])) else ""))
    for role, shard in (("b14", "gpu1"), ("b_lo", "gpu2"), ("b_hi", "gpu3")):
        b = blk.get(role)
        if b is None:
            rec["ok"] = False; rec["waiting"].append(f"{role} unknown"); continue
        hit = find_r1(src, f"summary_b{b}.json", prefer=shard)
        ok_rec, n = final_by_record(fin, shard)
        if hit:
            rec["final_blocks"].append(b); rec["summaries"][str(b)] = f"{hit[0]}/summary_b{b}.json"
        elif ok_rec:
            rec["missing_blocks"].append(b); rec["summaries"][str(b)] = f"missing; {shard} final by R1.final ({n} runs)"
        else:
            rec["ok"] = False
            rec["waiting"].append(f"summary_b{b}.json ({role}, {shard}) missing in dsse_runs/gpu1-gpu6; {R1_FINAL_HINT}")
    return rec


def gate_G5(src, fin):
    """Start condition of r2_pc and r2_ana: every shard's PC outputs are final: shard_done.json exists, else R1.final lists at least
    two notebook runs of the shard and 'final': true ('final with missing jobs'; the sha256 of pc_missing_<shard>.json is recorded)."""
    rec = dict(ok=True, waiting=[], shards={})
    for s in SHARDS:
        d = os.path.join(src, s)
        ok_rec, n = final_by_record(fin, s)
        pm = os.path.join(d, f"pc_missing_{s}.json")
        if os.path.isfile(os.path.join(d, "shard_done.json")):
            rec["shards"][s] = "final (shard_done.json)"
        elif ok_rec and os.path.isdir(d):
            pm_sha = f"; pc_missing sha256 {sha256_file(pm)}" if os.path.isfile(pm) else ""
            rec["shards"][s] = f"final with missing jobs (R1.final: {n} runs{pm_sha})"
        else:
            rec["ok"] = False
            why = ("folder not found (other account: share it or copy it into this account's dsse_runs)" if not os.path.isdir(d) else
                   f"no shard_done.json; {R1_FINAL_HINT}")
            rec["shards"][s] = "not final: " + why; rec["waiting"].append(f"dsse_runs/{s}: {why}")
    return rec


def drive_meta(src, freeze_time):
    """Section 14 / G1: the Drive modification times of dsse_runs/gpu*/logs/*.log and gpu*/summary_b*.json (os.stat only, nothing is
    opened), next to the freeze commit's time, and whether the freeze predates the first round-1 summary."""
    if not (src and os.path.isdir(src)):
        return dict(available=False, note="dsse_runs not reachable from this session")
    stat = lambda pat: {f"{s}/{os.path.relpath(p, os.path.join(src, s))}": os.path.getmtime(p)
                        for s in SHARDS for p in rglob(os.path.join(src, s), pat)}
    logs, sums = stat("logs/*.log"), stat("summary_b*.json")
    try:
        ft = datetime.datetime.fromisoformat(freeze_time.replace("Z", "+00:00")).timestamp() if freeze_time else None
    except ValueError:
        ft = None
    first = min(sums.values()) if sums else None
    return dict(available=True, freeze_time=freeze_time, logs={k: iso(v) for k, v in sorted(logs.items())},
                summaries={k: iso(v) for k, v in sorted(sums.items())}, first_summary=iso(first) if first else None,
                first_log=iso(min(logs.values())) if logs else None,
                freeze_before_first_summary=None if ft is None or first is None else ft < first)


def gate_upstream(src, base, decl):
    """Wave 2: r2_pc after r2_t1; r2_ana after every other session (7.2, 7.3). 'Done' = session_done.json with no session_partial.json
    next to it (a later attempt that did not finish), and every output that its steps recorded present in dsse_runs/<session> (r2_pc
    also needs r2_t1's AT5 nets). Only r2_ana accepts a session declared 'not run' after two recorded attempts (7.2)."""
    rec = dict(ok=True, waiting=[], sessions={}, declared_not_run=decl)
    for u in SESSIONS[base]["upstream"]:
        d = os.path.join(src, u)
        done, partial = load_json(os.path.join(d, "session_done.json")), os.path.isfile(os.path.join(d, "session_partial.json"))
        if done is not None and not partial:
            missing = sorted({rel for v in (done.get("steps") or {}).values() for rel in (v.get("outputs") or {})
                              if not os.path.isfile(os.path.join(d, rel))})
            if base == "r2_pc" and u == "r2_t1" and not rglob(d, "rgi_toy/models/at5_*.pt"):
                missing.append("rgi_toy/models/at5_*.pt")
            if missing:
                rec["ok"] = False; rec["sessions"][u] = f"done, but {len(missing)} recorded outputs are missing"
                rec["waiting"].append(f"dsse_runs/{u}: outputs listed in session_done.json are missing: {missing[:5]}")
            else:
                rec["sessions"][u] = "done"
        elif u in decl and base == "r2_ana":
            n = len(load_json(os.path.join(d, "session_attempts.json")) or [])
            if n >= 2:
                rec["sessions"][u] = f"declared 'not run' after {n} attempts"
            else:
                rec["ok"] = False; rec["sessions"][u] = f"declared 'not run' but only {n} attempt(s) recorded"
                rec["waiting"].append(f"{u}: declared 'not run' needs two recorded attempts, found {n}")
        else:
            rec["ok"] = False; rec["sessions"][u] = "not done"
            rec["waiting"].append(f"dsse_runs/{u}: session_partial.json next to session_done.json (a later attempt did not finish)"
                                  if done is not None else f"dsse_runs/{u}/session_done.json missing")
            if u in decl:
                rec["waiting"].append(f"{u}: a declaration 'not run' applies to r2_ana only (7.2), not to {base}")
    return rec


def plan_r1(src, patterns, blk=None):
    """The round-1 files of the copy list (7.3) as [(shard, rel, source path)], located by name across gpu1-gpu6. rk files only for
    the final blocks of B3. Duplicates (the same shard-relative path in several shards) by OD-G9: rk files and summaries from the
    shard assigned to the block (gpu1 block 14, gpu2 b_lo, gpu3 b_hi); a duplicated PC job is chosen once for its JSON and its model
    together (among the shards that hold both): the shard whose pc_expected lists it, if several do the one with shard_done.json,
    then the newest jobs/<name>.json; both files come from that shard ('lacks' records a missing one). Returns (files, choices,
    skipped)."""
    hits = {}
    for pat in expand(patterns):
        if not re.match(r"^gpu(\*|\d)/", pat):                       # a file in dsse_runs/ itself (r1_finality.json)
            for p in rglob(src, pat):
                hits.setdefault(os.path.relpath(p, src).replace(os.sep, "/"), {})[""] = p
            continue
        first, rest = pat.split("/", 1)
        for s in (SHARDS if first == "gpu*" else (first,)):
            base = os.path.join(src, s)
            for p in rglob(base, rest):
                hits.setdefault(os.path.relpath(p, base).replace(os.sep, "/"), {})[s] = p
    assign = {}
    if blk:
        assign = {14: "gpu1", blk.get("b_lo"): "gpu2", blk.get("b_hi"): "gpu3"}
    expected = {s: set(load_json(os.path.join(src, s, f"pc_expected_{s}.json")) or []) for s in SHARDS}
    done = {s: os.path.isfile(os.path.join(src, s, "shard_done.json")) for s in SHARDS}
    files, choices, skipped = [], [], []
    stamp = lambda p: dict(mtime=iso(os.path.getmtime(p)), size=os.path.getsize(p))
    jobs, job_pick = {}, {}                       # OD-G9 for a PC job: ONE shard for its JSON and its model together
    for rel in hits:
        if (mj := JOB_RE.match(rel)):
            jobs.setdefault(mj.group(1), []).append(rel)
    for name, rels in sorted(jobs.items()):
        if all(len(hits[r]) == 1 for r in rels):
            continue                              # no duplicate: every file comes from the only shard that has it
        cand_all = sorted({s for r in rels for s in hits[r]})
        both = [s for s in cand_all if all(s in hits[r] for r in rels)]
        pool, rule = (both or cand_all), "most recent jobs/<name>.json"
        lst = [s for s in pool if name in expected[s]]
        if len(lst) == 1:
            pool, rule = lst, "pc_expected lists it"
        elif lst:
            pool = lst
        if len(pool) > 1 and any(done[s] for s in pool):
            pool = [s for s in pool if done[s]]; rule = "shard_done.json, then most recent jobs/<name>.json"
        jrel = next((r for r in rels if r.endswith(".json")), rels[0])
        age = lambda s: os.path.getmtime(hits[jrel][s]) if s in hits[jrel] else max(os.path.getmtime(hits[r][s]) for r in rels if s in hits[r])
        job_pick[name] = pick = max(pool, key=lambda s: (age(s), s))
        choices.append(dict(job=name, paths=sorted(rels), chosen=pick, rule=rule, lacks=[r for r in sorted(rels) if pick not in hits[r]],
                            candidates={r: {s: stamp(p) for s, p in hits[r].items()} for r in sorted(rels)}))
    for rel in sorted(hits):
        cand = hits[rel]
        m = RK_RE.match(rel)
        if m and blk is not None and int(m.group(1)) not in blk.get("final_blocks", []):
            skipped.append(dict(path=rel, shards=sorted(cand), why=f"block {m.group(1)} not a final block of B3"))
            continue
        mj = JOB_RE.match(rel)
        if mj and mj.group(1) in job_pick:
            if job_pick[mj.group(1)] in cand:
                files.append((job_pick[mj.group(1)], rel, cand[job_pick[mj.group(1)]]))
            continue
        if len(cand) == 1 or PER_SHARD_RE.match(rel):
            files += [(s, rel, p) for s, p in sorted(cand.items())]
            continue
        mb = m or SUM_RE.match(rel)
        pool, rule = sorted(cand), "most recent file"
        if mb and assign.get(int(mb.group(1))) in cand:
            pool, rule = [assign[int(mb.group(1))]], "assigned shard of the block"
        if len(pool) > 1 and any(done[s] for s in pool):
            pool = [s for s in pool if done[s]]; rule = "shard_done.json, then most recent file"
        pick = max(pool, key=lambda s: (os.path.getmtime(cand[s]), s))
        files.append((pick, rel, cand[pick]))
        choices.append(dict(path=rel, chosen=pick, rule=rule, candidates={s: dict(mtime=iso(os.path.getmtime(p)), size=os.path.getsize(p))
                                                                          for s, p in cand.items()}))
    return files, choices, skipped


def r1_patterns(ctx):
    """The session's round-1 copy list (7.3); a self-test of a session that reads round-1 PC files adds R1_QUICK (runner_steps-2)."""
    pats = tuple(SESSIONS[ctx.base]["r1"])
    if ctx.selftest and set(R1_PC) & set(pats):
        pats += R1_QUICK
    return pats


def plan_r2(src, base):
    """Round-2 inputs of a wave-2 session as [(session, rel, source path)]: r2_pc copies r2_t1's AT5 BP nets; r2_ana every output
    of the other sessions except the local-only arrays and the state.pt copies (7.3), logs and the runner's temporary files."""
    out = []
    for pat in SESSIONS[base]["r2"]:
        if pat == R2_ALL:
            for u in SESSIONS[base]["upstream"]:
                d = os.path.join(src, u)
                for p in rglob(d, "**/*"):
                    rel = os.path.relpath(p, d).replace(os.sep, "/")
                    if rel.split("/")[0] in NOT_SYNCED + ("logs",) or rel.endswith((".tmp", ".r2tmp")) or re.match(r"^rk/b\d+/state\.pt$", rel):
                        continue
                    out.append((u, rel, p))
        else:
            u, rest = pat.split("/", 1)
            out += [(u, os.path.relpath(p, os.path.join(src, u)).replace(os.sep, "/"), p) for p in rglob(os.path.join(src, u), rest)]
    return out


def fetch(src, dst):
    """Copy src to dst while hashing (one read of the Drive file); an existing copy with the same size and mtime is reused."""
    st = os.stat(src)
    if os.path.isfile(dst):
        dt = os.stat(dst)
        if dt.st_size == st.st_size and int(dt.st_mtime) == int(st.st_mtime):
            return sha256_file(dst), st.st_size, True
        os.chmod(dst, 0o644); os.remove(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    h, tmp = hashlib.sha256(), dst + ".r2tmp"
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        for b in iter(lambda: fi.read(1 << 24), b""):
            h.update(b); fo.write(b)
    os.utime(tmp, (st.st_atime, st.st_mtime)); os.replace(tmp, dst)
    return h.hexdigest(), st.st_size, False


def set_tree_mode(root, writable, dirs_too=True):
    """Copied inputs are read-only (files 0444; folders 0555 for the round-1 copies). Advisory only when running as root, so the
    copies are also re-hashed at the end of the session (finalize)."""
    if not os.path.isdir(root):
        return
    for d, dirs, files in os.walk(root):
        for f in files:
            try:
                os.chmod(os.path.join(d, f), 0o644 if writable else 0o444)
            except OSError:
                pass
        if dirs_too or writable:
            try:
                os.chmod(d, 0o755 if writable else 0o555)
            except OSError:
                pass


def copy_tree_files(items, dst_root, label, keep_top=(), lock_dirs=True):
    """items: [(top, rel, src)] -> records. Files below dst_root that are not in items are removed (a stale copy never stays
    visible), except below the folders keep_top (r2in/r2_ana: the analysis session's own outputs, mirrored by the runner)."""
    set_tree_mode(dst_root, True)
    keep, recs, t, nb = set(), [], time.time(), 0
    for top, rel, p in items:
        dst = os.path.join(dst_root, top, rel); keep.add(os.path.normpath(dst))
        sha, size, reused = fetch(p, dst); nb += 0 if reused else size
        recs.append(dict(path=f"{top}/{rel}" if top else rel, source=top or "dsse_runs", sha256=sha, size=size,
                         drive_mtime=iso(os.path.getmtime(p)), reused=reused))
    for p in rglob(dst_root, "**/*"):
        if os.path.normpath(p) not in keep and os.path.relpath(p, dst_root).split(os.sep)[0] not in keep_top:
            os.remove(p)
    if os.path.isdir(dst_root):
        set_tree_mode(dst_root, False, dirs_too=lock_dirs)
    say(f"copied {label}: {len(recs)} files ({nb / 1e6:.0f} MB new) in {time.time() - t:.0f}s")
    return recs


# ======================================================================================================================================
# Gates G1-G3, versions, models, references
# ======================================================================================================================================
def gate_G1():
    """Plan freeze: the plan file's sha256 equals FROZEN_PLAN_SHA256, and the freeze commit (the commit that added the plan file; it
    must equal FREEZE_COMMIT when that is set) is an ancestor of the checked-out round-2 code commit. Also the amendments' sha256."""
    rec = dict(plan=PLAN_REL, expected_sha256=FROZEN_PLAN_SHA256, freeze_commit_constant=FREEZE_COMMIT, freeze_time_constant=FREEZE_TIME)
    p = os.path.join(REPO, PLAN_REL)
    rec["sha256"] = sha256_file(p) if os.path.isfile(p) else None
    rec["sha256_ok"] = rec["sha256"] == FROZEN_PLAN_SHA256
    rc, out = git("log", "--no-renames", "--diff-filter=A", "--format=%H %cI", "--", PLAN_REL)
    adds = [l.split() for l in (out or "").splitlines() if l.strip()] if rc == 0 else []
    rc2, out2 = git("log", "--no-renames", "--format=%H", "--", PLAN_REL)
    rec["commits_touching_plan"] = out2.split() if rc2 == 0 else None
    rec["head"] = git_head()
    rec["freeze_commit"], rec["freeze_time"] = (adds[-1] + [None])[:2] if adds else (None, None)
    rec["freeze_matches_constant"] = None if not FREEZE_COMMIT else bool(rec["freeze_commit"]) and rec["freeze_commit"].startswith(FREEZE_COMMIT)
    rec["freeze_is_ancestor"] = None
    if rec["freeze_commit"] and rec["head"] != "unknown":
        rc3, _ = git("merge-base", "--is-ancestor", rec["freeze_commit"], rec["head"])
        rec["freeze_is_ancestor"] = rc3 == 0
    a = os.path.join(REPO, AMEND_REL)
    rec["amendments"] = dict(path=AMEND_REL, sha256=sha256_file(a) if os.path.isfile(a) else None)
    rec["ok"] = bool(rec["sha256_ok"] and rec["freeze_commit"] and rec["freeze_is_ancestor"] and rec["freeze_matches_constant"] is not False)
    if rec["commits_touching_plan"] and len(rec["commits_touching_plan"]) > 1:
        rec["note"] = "the plan file was changed after the freeze commit (the checkout's sha256 decides G1)"
    return rec


def gate_G2(test):
    """An A100 (code tests with RUN_DEVICE=cpu skip the check and are recorded as code tests)."""
    if test:
        return dict(ok=True, checked=False, gpu=None, note="RUN_DEVICE=cpu: code test, GPU not checked")
    probe = ("import json, torch; ok = torch.cuda.is_available(); "
             "print(json.dumps(dict(gpu=torch.cuda.get_device_name(0) if ok else None, cuda=torch.version.cuda)))")
    try:                                                    # a short child process: the runner itself never holds a CUDA context
        r = subprocess.run([PY, "-c", probe], capture_output=True, text=True, timeout=300)
        info = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else {}
    except (OSError, subprocess.TimeoutExpired, ValueError) as e:
        return dict(ok=False, checked=True, gpu=None, note=repr(e))
    name = info.get("gpu")
    if not name:
        return dict(ok=False, checked=True, gpu=None, note="no CUDA device" + (f" ({r.stderr.strip()[-200:]})" if r.stderr.strip() else ""))
    return dict(ok="A100" in name, checked=True, gpu=name, cuda=info.get("cuda"))


_COMMIT_OK = {}


def ensure_commit(c):
    if c not in _COMMIT_OK:
        rc, _ = git("cat-file", "-e", f"{c}^{{commit}}")
        if rc != 0:
            git("fetch", "-q", "--filter=blob:none", "origin", c, timeout=120)
            rc, _ = git("cat-file", "-e", f"{c}^{{commit}}")
        _COMMIT_OK[c] = rc == 0
    return _COMMIT_OK[c]


def git_sha256_at(commit, path, local):
    """sha256 of a file at a commit ('absent' if the commit lacks it, None if the commit cannot be read). Blob ids are compared first,
    so an unchanged file needs no download in a partial clone."""
    if not ensure_commit(commit):
        return None
    rc, blob = git("rev-parse", "--verify", "-q", f"{commit}:{path}")
    if rc != 0 or not (blob or "").strip():
        return "absent"
    blob = blob.strip()
    if os.path.isfile(local):
        rc2, lb = git("hash-object", local)
        if rc2 == 0 and lb.strip() == blob:
            return sha256_file(local)
    rc3, data = git("cat-file", "blob", blob, timeout=180, binary=True)
    return hashlib.sha256(data).hexdigest() if rc3 == 0 else None


def r1_meta(r1dir):
    """R1.meta per shard (2.1): code_commit, code_commits_used, gpu, torch, transformers from shard_done.json, else the last line of
    code_commits.txt ('steps' and 'seconds' are not taken over before the analysis session)."""
    out = {}
    for s in SHARDS:
        sd = load_json(os.path.join(r1dir, s, "shard_done.json"))
        if sd:
            out[s] = dict(source="shard_done.json", **{k: sd.get(k) for k in ("code_commit", "code_commits_used", "gpu", "torch", "transformers")})
            continue
        cc = (read_text(os.path.join(r1dir, s, "code_commits.txt")) or "").split()
        out[s] = dict(source="code_commits.txt", code_commit=cc[-1], code_commits_used=cc) if cc else dict(source=None)
    return out


def gate_G3(meta, shards_read):
    """R1 code freeze: sha256 of the five imported round-1 modules in this checkout and at the code commit of every shard read; all
    equal, else the label 'round-1 code differs between shards' (applied by the report to every measurement importing the module).
    The modules at the shards' earlier commits (code_commits_used) are recorded as well."""
    commits = {s: meta[s]["code_commit"] for s in shards_read if (meta.get(s) or {}).get("code_commit")}
    used = sorted({c for s in shards_read for c in ((meta.get(s) or {}).get("code_commits_used") or [])} | set(commits.values()))
    rec = dict(shard_commits=commits, commits_used=used, modules={}, labels={}, ok=True)
    for m in R1_MODULES:
        local = os.path.join(AJS, m)
        r = dict(checkout=sha256_file(local) if os.path.isfile(local) else None,
                 at_commit={c: git_sha256_at(c, f"analysis/ju_space/{m}", local) for c in used})
        if commits:
            vals = {r["at_commit"].get(c) for c in commits.values()}
            r["r1_shards_equal"] = len(vals) == 1 and None not in vals
            r["checkout_equals_r1"] = vals == {r["checkout"]}
            r["earlier_commits_equal"] = len({v for v in r["at_commit"].values()} | {r["checkout"]}) == 1
            r["equal"] = bool(r["r1_shards_equal"] and r["checkout_equals_r1"])
            if not r["equal"]:
                why = "not checkable (commit not readable)" if None in vals else ("between round-1 shards" if not r["r1_shards_equal"]
                                                                                   else "between round 1 and this checkout")
                r["label"] = "round-1 code differs between shards"; r["why"] = why
                rec["labels"][m] = f"round-1 code differs between shards ({why})"; rec["ok"] = False
        rec["modules"][m] = r
    missing = [s for s in shards_read if s not in commits]          # a shard read without a code commit cannot count as equal
    if missing:
        rec["shards_without_commit"] = missing; rec["ok"] = False
        for m in R1_MODULES:
            rec["labels"][m] = "round-1 code differs between shards (not checkable: no code commit of " + ", ".join(missing) + ")"
            rec["modules"][m].update(label="round-1 code differs between shards", why=f"no code commit of {', '.join(missing)}")
    if not shards_read:
        rec["note"] = "no round-1 shard read in this session: checkout hashes only"
    return rec


def models_record(ctx):
    """Qwen revisions and checksums (round-1 checksums.json for Qwen3-1.7B; QWEN06 for Qwen3-0.6B) and whether the files of this
    root were checked: by the notebook (models/verified.json), else here for the models the session uses (real runs only)."""
    sums = load_json(CHECKSUMS) or {}
    rec = {M17: dict(repo=sums.get("model_repo"), revision=sums.get("model_revision"), sha256=sums.get("model_sha256") or {},
                     persona_repo=sums.get("persona_repo"), persona_revision=sums.get("persona_revision"),
                     persona_git_sha1=sums.get("persona_git_sha1")),
           M06: dict(repo=QWEN06["repo"], revision=QWEN06["revision"], sha256=dict(QWEN06["sha256"]))}
    ver = load_json(os.path.join(ctx.root, "models", "verified.json")) or {}
    for m, r in rec.items():
        d = os.path.join(ctx.root, "models", m)
        r["present"] = os.path.isfile(os.path.join(d, "config.json"))
        v = ver.get(m)
        if v is not None:
            r["verified_by_notebook"] = v; r["ok"] = bool(v.get("ok")) and r["present"]
        elif r["present"] and m in SESSIONS[ctx.base]["models"] and not (ctx.test or ctx.selftest or ctx.dry):
            got = {fn: sha256_file(os.path.join(d, fn)) if os.path.isfile(os.path.join(d, fn)) else None for fn in r["sha256"]}
            r["verified_by_runner"] = got; r["ok"] = bool(r["sha256"]) and all(got[fn] == h for fn, h in r["sha256"].items())
        else:
            r["ok"] = r["present"] and (ctx.test or ctx.selftest)
            r["note"] = "not checked (code test or self-test)" if r["ok"] else "not present or not checked"
    return rec


def references_record():
    """sha256 of the reference files of Appendix E in analysis/ju_space/round2/ref/<path> (a file found elsewhere below ref/ by its
    content is 'moved'), of the freeze's MISSING.txt, of the two repository modules the reference shim uses (9.5), and of the committed
    output of the comparison of the ported functions with the references (9.5; found by name in round2/)."""
    have = {}
    for p in rglob(REF_DIR, "**/*"):
        have.setdefault(sha256_file(p), os.path.relpath(p, REF_DIR).replace(os.sep, "/"))
    files = {}
    for rel, sha in REF_SHA256.items():
        p = os.path.join(REF_DIR, rel)
        got = sha256_file(p) if os.path.isfile(p) else None
        st = "ok" if got == sha else ("differs" if got else ("moved" if sha in have else "missing"))
        files[rel] = dict(expected=sha, status=st, **({"found": have[sha]} if st == "moved" else {}), **({"sha256": got} if st == "differs" else {}))
    extra = sorted(v for k, v in have.items() if k not in set(REF_SHA256.values()))
    shim = {rel: dict(expected=sha, sha256=sha256_file(os.path.join(REPO, rel)) if os.path.isfile(os.path.join(REPO, rel)) else None)
            for rel, sha in SHIM_SHA256.items()}
    for r in shim.values():
        r["ok"] = r["sha256"] == r["expected"]
    r2 = os.path.dirname(REF_DIR)
    comp = sorted({p for g in REF_COMPARISON_GLOBS for p in rglob(r2, g)} |
                  {p for d in ("*", "tests/*") for p in rglob(r2, d)
                   if re.search(r"compar|refcheck", os.path.basename(p), re.I) and not p.endswith((".py", ".pyc"))})
    mt, rc = os.path.join(REF_DIR, "MISSING.txt"), os.path.join(REPO, REF_COMPARISON)
    comparison = dict(path=REF_COMPARISON, sha256=sha256_file(rc) if os.path.isfile(rc) else None,
                      status="ok" if os.path.isfile(rc) else "missing (9.5: committed before the first session)",
                      others={os.path.relpath(p, REPO): sha256_file(p) for p in comp if os.path.normpath(p) != os.path.normpath(rc)})
    if os.path.isfile(rc):                    # the output must come from the ports that run now (9.5; NOTES_w5: regenerate after a change)
        recorded = (load_json(rc) or {}).get("module_sha256") or {}
        here = {n: os.path.join(SCRIPTS, n) if n in REF_COMPARISON_PORTS else os.path.join(AJS, n)
                for n in REF_COMPARISON_PORTS + ("pc_meta.py", "pc_common.py")}
        comparison["modules"] = {n: dict(recorded=recorded.get(n), checkout=sha256_file(p) if os.path.isfile(p) else None) for n, p in here.items()}
        comparison["changed"] = [n for n, v in comparison["modules"].items() if v["recorded"] != v["checkout"]]
        if any(n in REF_COMPARISON_PORTS for n in comparison["changed"]):
            comparison["status"] = "stale (ports changed after the comparison)"
    return dict(dir=os.path.relpath(REF_DIR, REPO), files=files, n_ok=sum(f["status"] == "ok" for f in files.values()),
                n_missing=sum(f["status"] != "ok" for f in files.values()), not_in_appendix_E=extra,
                missing_txt_sha256=sha256_file(mt) if os.path.isfile(mt) else None, shim_modules=shim, comparison=comparison)


# ======================================================================================================================================
# The session
# ======================================================================================================================================
class Ctx:
    def __init__(self, base, attempt, root, work, drive, dry=False, verbose=False):
        self.base, self.attempt, self.dry, self.verbose = base, attempt, dry, verbose
        self.root, self.work = os.path.abspath(root), os.path.abspath(work)
        self.drive = os.path.abspath(drive) if drive else None
        src = os.environ.get("R1_SRC") or (os.path.dirname(self.drive) if self.drive else None)
        self.src = os.path.abspath(src) if src else None
        self.r1, self.r2in, self.local = (os.path.join(self.work, x) for x in ("r1", "r2in", "local"))
        self.device = os.environ.get("RUN_DEVICE", "cuda")
        self.test = self.device == "cpu"
        self.selftest = os.environ.get("R2_SELFTEST") == "1"
        self.nt = os.environ.get("NT") or str(max(1, min(8, (os.cpu_count() or 4) - 2)))
        self.t_start = time.time()
        t0 = os.environ.get("R2_T0")
        try:
            t0 = float(t0) if t0 else None
        except ValueError:
            t0 = None
        self.t0 = t0 if t0 and 0 <= self.t_start - t0 < 86400 else self.t_start
        self.guard_s = math.inf if self.selftest else float(os.environ.get("R2_GUARD_S", GUARD_S_DEFAULT))
        self.step_timeout = float(os.environ.get("R2_STEP_TIMEOUT", 3600))
        self.blk, self.models, self.code_commit, self.manifest_path = None, {}, "unknown", None
        self.inputs_digest = None      # sha256 over the copied inputs (kind, path, sha256); r2_pc / r2_ana redo steps made from others

    def elapsed(self):
        return time.time() - self.t0


class RStep:
    """A Step with its block, names, command line and outputs filled in."""

    def __init__(self, ctx, st):
        self.st = st
        b = None
        if st.block == "14":
            b = 14
        elif st.block in ("b_lo", "b_hi"):
            b = (ctx.blk or {}).get(st.block)
            if b is None:
                b = f"<{st.block}>"
        self.b = b
        f = dict(root=ctx.root, r1=ctx.r1, work=ctx.work, r2in=ctx.r2in, b=b if b is not None else "")
        fmt = lambda s: s.format(**f)
        self.name = fmt(st.name)
        self.argv = [fmt(a) for a in shlex.split(st.args)]
        self.out = [fmt(o) for o in (st.quick_out if ctx.selftest and st.quick_out is not None else st.out)]
        self.opt = [fmt(o) for o in st.opt]
        self.after = [fmt(a) for a in st.after]
        self.settled = [fmt(a) for a in st.settled]
        self.needed_by = [fmt(a) for a in st.needed_by]
        self.env = {k: fmt(v) for k, v in {**STEP_ENV.get(ctx.base, {}), **st.env}.items()}
        if b is not None:
            self.env["BLOCK"] = str(b)
        self.script = os.path.join(SCRIPTS, st.script)
        self.marker = os.path.join(ctx.work, "steps", self.name + ".json")

    def cmdline(self, ctx, short=False):
        """The call as a shell line; short: <work> and <root> instead of the paths (for printing)."""
        s = (lambda a: a.replace(ctx.work, "<work>").replace(ctx.root, "<root>")) if short else (lambda a: a)
        q = lambda a: a if re.fullmatch(r"[\w@%+=:,./<>*-]+", a) else shlex.quote(a)
        env = " ".join(f"{k}={q(s(v))}" for k, v in sorted(self.env.items()))
        words = ["python", os.path.basename(self.script) if short else self.script] + [s(a) for a in self.argv]
        return (env + " " if env else "") + " ".join(q(w) for w in words)


def resolve_steps(ctx):
    steps = [RStep(ctx, st) for st in STEPS[ctx.base]]
    names = [s.name for s in steps]
    assert len(set(names)) == len(names), f"duplicate step names in {ctx.base}: {names}"
    known = set(names)
    for s in steps:
        assert all(a in known for a in s.after + s.settled + s.needed_by), f"{ctx.base}/{s.name}: unknown step in after/settled/needed_by"
    return steps


def files_of(ctx, pats):
    return {p: rglob(ctx.work, p) for p in pats}


def is_done(ctx, rs, by_name, _seen=None):
    """Done = the runner recorded the step's success and its outputs exist ('skipped when its output exists', 7.4). A step with
    local-only outputs also needs them while a step that reads them is not done: an empty record of its local files, or one with a
    missing file, then counts as not done (runner_steps-10; a repeat is cheap, the scripts skip their finished stages)."""
    m = load_json(rs.marker)
    if not m or m.get("status") != "ok":
        return False
    if not m.get("no_output"):
        if any(not hits for hits in files_of(ctx, rs.out).values()):
            return False
    if rs.st.local:
        _seen = (_seen or set()) | {rs.name}
        consumers_done = all(c in _seen or is_done(ctx, by_name[c], by_name, _seen) for c in rs.needed_by)
        loc = m.get("local") or []
        if not consumers_done and (not loc or not all(os.path.isfile(os.path.join(ctx.work, p)) for p in loc)):
            return False
    return True


def not_run_reason(ctx, rs):
    """G4: a part at a block whose round-1 outputs are final-missing is 'not run' if it reads round-1 files of that block; G33."""
    miss = (ctx.blk or {}).get("missing_blocks") or []
    if isinstance(rs.b, int) and miss:
        if rs.st.r1_reads == "block" and rs.b in miss:
            return "round-1 output missing"
        if rs.st.r1_reads == "any" and all(b in miss for b in ctx.blk.get("B3", [])):
            return "round-1 output missing"
    if rs.st.needs == "qwen06" and not (ctx.models.get(M06) or {}).get("ok"):
        return "G33: Qwen3-0.6B download or checksum failed"
    return None


def step_env(ctx, rs):
    e = dict(os.environ)
    for k in ("R2_T0", "R2_GUARD_S", "R2_SELFTEST", "R1_SRC"):
        e.pop(k, None)
    if ctx.selftest:
        e.update(MOCK="1", QUICK="1")
    else:
        for k in ("MOCK", "QUICK"):
            if e.pop(k, None) is not None:
                say(f"note: {k} removed from the environment of a real run")
    blk = ctx.blk or {}
    e.update(DEVICE=ctx.device, NT=ctx.nt, PYTHONUNBUFFERED="1", POOLDIRS=os.path.join(ctx.root, "gpu_pkg", "rk_dirs.pt"),
             PERSONA=os.path.join(ctx.root, "rsi_dl", "persona"), R2_SESSION=ctx.base, R2_ATTEMPT=ctx.attempt, R2_ROOT=ctx.root,
             R2_WORK=ctx.work, R2_R1=ctx.r1, R2IN=ctx.r2in, R2_MANIFEST=ctx.manifest_path or "",
             R2_B3=",".join(str(b) for b in blk.get("final_blocks", []) if b is not None),
             R2_BLO=str(blk.get("b_lo") or ""), R2_BHI=str(blk.get("b_hi") or ""),
             R2_ORDER=",".join(str(b) for b in (blk.get("order") or [])),
             R2_MISSING_BLOCKS=",".join(str(b) for b in blk.get("missing_blocks", [])))
    e.update(rs.env)
    return e


def local_snapshot(ctx):
    return {os.path.relpath(p, ctx.work): os.path.getmtime(p) for p in rglob(ctx.local, "**/*")}


def exec_step(ctx, rs):
    """Run one step: output streamed and logged; on exit code 0 its outputs are checked; a record goes to steps/<name>.json."""
    os.makedirs(os.path.dirname(rs.marker), exist_ok=True); os.makedirs(os.path.join(ctx.work, "logs"), exist_ok=True)
    for a in rs.argv:                                       # the folder of an <out.json> argument (analysis scripts)
        if a.startswith(ctx.work + os.sep) and a.endswith(".json"):
            os.makedirs(os.path.dirname(a), exist_ok=True)
    say(f"[run] {rs.name}: {rs.cmdline(ctx, short=True)}")
    before, t, timed_out = local_snapshot(ctx), time.time(), False
    if not os.path.isfile(rs.script):
        rc, note = 127, f"script {rs.st.script} not found in {SCRIPTS}"
        say(f"[fail] {rs.name}: {note}")
    else:
        with open(os.path.join(ctx.work, "logs", rs.name + ".log"), "a") as lg:
            lg.write(f"=== {iso()} attempt {ctx.attempt} code {ctx.code_commit}\n$ {rs.cmdline(ctx)}\n"); lg.flush()
            p = subprocess.Popen([PY, "-u", rs.script] + rs.argv, env=step_env(ctx, rs), cwd=ctx.work, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)

            def pump():
                for line in p.stdout:
                    lg.write(line); lg.flush()
                    if "Loading weights" not in line:
                        print(line, end="", flush=True)
            th = threading.Thread(target=pump, daemon=True); th.start()
            try:
                rc = p.wait(timeout=ctx.step_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(p.pid, sig)
                    except OSError:
                        pass
                    try:
                        p.wait(timeout=20); break
                    except subprocess.TimeoutExpired:
                        continue
                rc = p.returncode if p.returncode is not None else -9
            th.join(timeout=30)
        note = f"timeout after {ctx.step_timeout:.0f}s" if timed_out else ""
    secs = round(time.time() - t, 1)
    found = files_of(ctx, rs.out)
    missing = [p for p, h in found.items() if not h]
    ok = rc == 0 and (not missing or rs.st.gate_in_script)
    if rc == 0 and missing and not rs.st.gate_in_script:
        note = "exit code 0 but outputs missing: " + ", ".join(missing)
    outs = {}
    for p in sorted({f for h in list(found.values()) + list(files_of(ctx, rs.opt).values()) for f in h}):
        outs[os.path.relpath(p, ctx.work)] = dict(size=os.path.getsize(p), sha256=sha256_file(p))
    after, local = local_snapshot(ctx), []
    if rs.st.local:                                         # changed in this run, plus what an earlier record listed and still exists
        prev = [q for q in ((load_json(rs.marker) or {}).get("local") or []) if os.path.isfile(os.path.join(ctx.work, q))]
        local = sorted({k for k, v in after.items() if before.get(k) != v} | set(prev))
        if not local and ok:                                # nothing new: the script found its arrays in place
            pat = rs.st.local if isinstance(rs.st.local, str) else "local/**"
            local = sorted(k for k in after if fnmatch_path(k.replace(os.sep, "/"), pat))
    rec = dict(name=rs.name, status="ok" if ok else "failed", rc=rc, seconds=secs, attempt=ctx.attempt, ended=iso(), code_commit=ctx.code_commit,
               block=rs.b, command=rs.cmdline(ctx, short=True), outputs=outs, missing=missing, no_output=bool(ok and missing),
               local=local, inputs_digest=ctx.inputs_digest, note=note or None)
    dump_json(rec, rs.marker)
    say(f"[done] {rs.name} rc={rc} ({secs:.0f}s){'' if ok else ' FAILED: ' + (note or f'exit code {rc}')}")
    return rec


def mirror_ana(ctx, rs):
    """r2_ana: copy a step's outputs into <r2in>/r2_ana/ so that later analysis scripts find them under <r2_root>/r2_ana/."""
    for p in sorted({f for h in list(files_of(ctx, rs.out).values()) + list(files_of(ctx, rs.opt).values()) for f in h}):
        dst = os.path.join(ctx.r2in, "r2_ana", os.path.relpath(p, ctx.work))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(p, dst)


def step_state(ctx, rs, st, by_name):
    """The state a step takes before it would run: 'skipped' if done (a done step stays done whatever reason now applies,
    runner_steps-6), 'not run' (G4 / G33, or an 'after' step not run), 'blocked' (an 'after' step not finished, or a 'settled' step
    neither done nor 'not run'), else None (run it). st holds the states of the steps before it."""
    OK, FINAL = ("ran", "skipped"), ("ran", "skipped", "not run")
    if is_done(ctx, rs, by_name):
        why = not_run_reason(ctx, rs)
        return dict(state="skipped", note="output exists" + (f" (now also: {why})" if why else ""))
    why = not_run_reason(ctx, rs)
    dep_nr = [a for a in rs.after if st.get(a, {}).get("state") == "not run"]
    if why or dep_nr:
        return dict(state="not run", note=why or f"depends on {', '.join(dep_nr)} (not run)")
    wait = [a for a in rs.after if st.get(a, {}).get("state") not in OK] + [a for a in rs.settled if st.get(a, {}).get("state") not in FINAL]
    if wait:
        return dict(state="blocked", note="waits for " + ", ".join(wait))
    return None


def would_run(ctx, steps):
    """The steps that a run would execute now (a dry pass of run_steps: done steps count as finished, the others as run)."""
    by_name, st, out = {s.name: s for s in steps}, {}, []
    for rs in steps:
        s = step_state(ctx, rs, st, by_name)
        if s is None or s["state"] == "blocked":
            out.append(rs.name); s = dict(state="ran")
        st[rs.name] = s
    return out


def run_steps(ctx, steps):
    """Steps in table order; a step whose 'after' / 'settled' steps did not finish waits; one retry of failed steps (round-1 pattern)
    if time allows; time guard after every step that ran (7.4)."""
    by_name = {s.name: s for s in steps}
    st, ran, guard = {}, [], False
    FINAL = ("ran", "skipped", "not run")
    for rnd in (1, 2):
        for rs in steps:
            cur = st.get(rs.name, {}).get("state")
            if cur in FINAL or (rnd == 2 and cur not in ("failed", "blocked")):
                continue
            if rnd == 2 and cur == "failed" and st[rs.name].get("tries", 0) >= 2:
                continue
            pre = step_state(ctx, rs, st, by_name)
            if pre and pre["state"] in ("skipped", "not run"):
                st[rs.name] = pre
                say(f"[{'skip' if pre['state'] == 'skipped' else 'not run'}] {rs.name}: {pre['note']}")
                if pre["state"] == "skipped" and ctx.base == "r2_ana":
                    mirror_ana(ctx, rs)
                continue
            if guard:
                st[rs.name] = dict(state="pending", note="time guard"); continue
            if pre:                                                 # blocked
                st[rs.name] = pre; say(f"[wait] {rs.name}: {pre['note']}")
                continue
            rec = exec_step(ctx, rs); ran.append(rs.name)
            tries = st.get(rs.name, {}).get("tries", 0) + 1
            st[rs.name] = dict(state="ran" if rec["status"] == "ok" else "failed", note=rec.get("note"), seconds=rec["seconds"], tries=tries)
            if rec["status"] == "ok" and ctx.base == "r2_ana":
                mirror_ana(ctx, rs)
            n = sync(ctx.work, ctx.drive)
            say(f"[time] {ctx.elapsed() / 60:.1f} min since the session start (guard {ctx.guard_s / 60:.0f} min); {n} files synced")
            if ctx.elapsed() >= ctx.guard_s and not guard:
                guard = True; say("time guard reached (7.4): the session stops after this step")
        if guard:
            break
    for rs in steps:                                           # steps never reached because of the guard
        if rs.name not in st:
            st[rs.name] = dict(state="pending", note="time guard")
    return st, ran, guard


def remove_done_record(ctx, why):
    """A session that will run steps again is not done until it finishes: its old session_done.json goes (in <work> and on Drive), so
    that wave 2 cannot start on a folder that is being redone (runner_steps-5, runner_gates-8). Error-tolerant."""
    for d in (ctx.work, ctx.drive):
        p = os.path.join(d, "session_done.json") if d else None
        try:
            if p and os.path.exists(p):
                os.remove(p); say(f"removed {'Drive' if d == ctx.drive else 'local'} session_done.json: {why}")
        except OSError as e:
            say(f"could not remove an old session_done.json ({e})")


def invalidate_changed(ctx, steps):
    """r2_pc, r2_ana: a finished step whose record names other copied inputs than this attempt's (inputs_digest) is redone: its record,
    its recorded outputs (required and optional), their mirror in <r2in>/r2_ana and the script's resume cache (pc_toy/partial) are
    removed, locally and on Drive (runner_steps-4). Returns the names of the steps concerned."""
    names = []
    for rs in steps:
        m = load_json(rs.marker)
        if not m or m.get("status") != "ok" or m.get("inputs_digest") == ctx.inputs_digest:
            continue
        names.append(rs.name)
        for rel in list((m.get("outputs") or {})) + [os.path.relpath(rs.marker, ctx.work)]:
            for top in (ctx.work, ctx.drive, os.path.join(ctx.r2in, "r2_ana") if ctx.base == "r2_ana" else None):
                try:
                    if top and os.path.isfile(os.path.join(top, rel)):
                        os.remove(os.path.join(top, rel))
                except OSError as e:
                    say(f"could not remove {rel} ({e})")
    if names:
        for rel in PARTIAL_DIRS.get(ctx.base, ()):
            for top in (ctx.work, ctx.drive):
                if top and os.path.isdir(os.path.join(top, rel)):
                    shutil.rmtree(os.path.join(top, rel), ignore_errors=True)
        say(f"copied inputs changed since these steps ran: {names}; their outputs were removed and they run again")
    return names


def finalize(ctx, steps, st, ran, guard, attempts, copied, t_attempt):
    """session_done.json (every step ran, was skipped or 'not run' for a stated reason) or session_partial.json; integrity of the
    round-1 copies; the attempt record."""
    integ, known = [], set()
    for rec in copied:
        p = os.path.join(ctx.r1 if rec["kind"] == "r1" else ctx.r2in, rec["path"]); known.add(os.path.normpath(p))
        if not os.path.isfile(p) or sha256_file(p) != rec["sha256"]:
            integ.append(rec["path"])
    for top in (ctx.r1, ctx.r2in):
        for p in rglob(top, "**/*"):
            if os.path.normpath(p) not in known and os.path.relpath(p, ctx.r2in).split(os.sep)[0] != "r2_ana":
                integ.append("added: " + os.path.relpath(p, ctx.work))
    if integ:
        say(f"WARNING: {len(integ)} copied input files were changed or removed during the session: {integ[:5]}")
    steps_rec = {}
    for rs in steps:
        m = load_json(rs.marker) or {}
        s = st[rs.name]
        steps_rec[rs.name] = dict(state=s["state"], note=s.get("note"), block=rs.b, seconds=m.get("seconds"), ran_in=m.get("attempt"),
                                  code_commit=m.get("code_commit"), outputs=m.get("outputs"))
    states = {v["state"] for v in steps_rec.values()}
    done = states <= {"ran", "skipped", "not run"}
    failed = [n for n, v in steps_rec.items() if v["state"] in ("failed", "blocked")]
    pending = [n for n, v in steps_rec.items() if v["state"] == "pending"]
    attempts[-1].update(ended=iso(), seconds=round(time.time() - t_attempt, 1), elapsed_since_t0=round(ctx.elapsed(), 1), ran=ran,
                        result="done" if done else ("time guard" if guard and pending else "failed"), inputs_changed_during_session=integ)
    dump_json(attempts, os.path.join(ctx.work, "session_attempts.json"))
    man = (load_json(ctx.manifest_path) or {}) if ctx.manifest_path else {}
    common = dict(session=ctx.base, attempts=attempts, steps=steps_rec, code_commit=ctx.code_commit,
                  code_commits=(read_text(os.path.join(ctx.work, "code_commits.txt")) or "").split(),
                  plan_sha256=FROZEN_PLAN_SHA256, test_run=ctx.test, selftest=ctx.selftest,
                  packing=[dict(attempt=a["attempt"], ran=a.get("ran", []), result=a.get("result"), invalidated=a.get("invalidated") or [])
                           for a in attempts],
                  packing_table=PACKING.get(ctx.base, "as in plan 7.2"),
                  models={m: {k: (ctx.models.get(m) or {}).get(k) for k in ("repo", "revision", "sha256", "ok")} for m in SESSIONS[ctx.base]["models"]},
                  references={k: (man.get("references") or {}).get(k) for k in ("shim_modules", "comparison")},
                  seconds_steps=round(sum((v["seconds"] or 0) for v in steps_rec.values()), 1),
                  not_run={n: v["note"] for n, v in steps_rec.items() if v["state"] == "not run"})
    if done:
        dump_json(dict(common, finished=iso()), os.path.join(ctx.work, "session_done.json"))
        for d in (ctx.work, ctx.drive):
            try:
                if d and os.path.exists(os.path.join(d, "session_partial.json")):
                    os.remove(os.path.join(d, "session_partial.json"))
            except OSError as e:
                say(f"could not remove an old session_partial.json: {e}")
        code, msg = EXIT_DONE, f"session {ctx.base} done"
    else:
        remove_done_record(ctx, "this attempt did not finish")
        reason = "time guard (7.4)" if guard and pending else "steps failed"
        k_guard = sum(a.get("result") == "time guard" for a in attempts)
        nxt = ctx.base + "bcdefghijklmnopqrstuvwxyz"[k_guard - 1] if reason.startswith("time") else ctx.attempt
        dump_json(dict(common, reason=reason, remaining=pending, failed=failed, next_attempt=nxt, written=iso()),
                  os.path.join(ctx.work, "session_partial.json"))
        code = EXIT_GUARD if reason.startswith("time") else EXIT_FAILED
        msg = (f"session {ctx.base} stopped by the time guard; {len(pending)} steps remain: run the notebook again (continuation {nxt})"
               if code == EXIT_GUARD else f"session {ctx.base}: steps failed or blocked: {failed}; rerun the notebook to retry them")
    sync(ctx.work, ctx.drive)
    say(msg)
    return code


def start_conditions(ctx):
    """G4 / G5 / wave-2 sessions as far as the session needs them; also the block set (OD-G2, G7-G9)."""
    meta = SESSIONS[ctx.base]
    res = dict(ok=True, waiting=[])
    needs_src = bool(meta["r1"] or meta["r2"] or meta["upstream"])
    if needs_src and not (ctx.src and os.path.isdir(ctx.src)):
        res["ok"] = False; res["waiting"].append(f"round-1/round-2 source folder (dsse_runs) not found: {ctx.src}")
        return res
    fin = r1_final(ctx.src)
    res["R1_final"] = fin
    if "G4" in meta["start"]:
        ctx.blk = r1_blocks(ctx.src)
        g4 = gate_G4(ctx.src, ctx.blk, fin)
        ctx.blk.update(final_blocks=g4["final_blocks"], missing_blocks=g4["missing_blocks"])
        res["G4"] = g4; res["blocks"] = ctx.blk
    if "G5" in meta["start"]:
        res["G5"] = gate_G5(ctx.src, fin)
    if meta["upstream"]:
        res["upstream"] = gate_upstream(ctx.src, ctx.base, declared("R2_DECLARED_NOT_RUN"))
    for k in ("G4", "G5", "upstream"):
        if k in res and not res[k]["ok"]:
            res["ok"] = False; res["waiting"] += res[k]["waiting"]
    return res


def attempt_name(base, explicit, attempts):
    k = sum(a.get("result") == "time guard" for a in attempts)
    auto = base + ("" if k == 0 else "bcdefghijklmnopqrstuvwxyz"[k - 1])
    return explicit or auto


def run_session(session, root, work, drive):
    base, explicit = split_session(session)
    ctx = Ctx(base, None, root, work, drive)
    for d in (ctx.work, os.path.join(ctx.work, "logs"), os.path.join(ctx.work, "steps")):
        os.makedirs(d, exist_ok=True)
    LOGF.append(open(os.path.join(ctx.work, "logs", "runner.log"), "a"))
    attempts = load_json(os.path.join(ctx.work, "session_attempts.json")) or []
    ctx.attempt = attempt_name(base, explicit, attempts)
    t_attempt = time.time()
    say(f"session {base} (attempt {ctx.attempt}, wave {SESSIONS[base]['wave']}); work {ctx.work}; drive {ctx.drive}; "
        f"{'SELF-TEST (MOCK/QUICK)' if ctx.selftest else ('CODE TEST (RUN_DEVICE=cpu)' if ctx.test else 'real run')}")
    ctx.code_commit = git_head()
    prev = os.path.join(ctx.work, "code_commits.txt")
    seen = (read_text(prev) or "").split()
    if seen and seen[-1] != ctx.code_commit:
        say(f"WARNING: this work directory was started with code {seen[-1][:10]}, now {ctx.code_commit[:10]}; finished steps are kept "
            f"(their outputs come from the earlier code; steps/<name>.json names the commit). Delete dsse_runs/{base} to start over.")
    if not seen or seen[-1] != ctx.code_commit:
        with open(prev, "a") as f:
            f.write(ctx.code_commit + "\n")
    # gates G1, G2
    g1, g2 = gate_G1(), gate_G2(ctx.test)
    say(f"G1 plan freeze: {'ok' if g1['ok'] else 'FAILED'} (sha256 {'ok' if g1['sha256_ok'] else 'differs/missing'}, freeze commit "
        f"{(g1['freeze_commit'] or 'none')[:10]}, ancestor of {ctx.code_commit[:10]}: {g1['freeze_is_ancestor']})")
    say(f"G2 GPU: {g2.get('gpu') or g2.get('note')} -> {'ok' if g2['ok'] else 'FAILED (an A100 is required)'}")
    if not g1["amendments"]["sha256"]:
        say(f"WARNING: the amendments file {AMEND_REL} is missing in this checkout (the manifest records its sha256 as None)")
    else:
        say(f"amendments: {AMEND_REL} sha256 {g1['amendments']['sha256'][:12]}...")
    check = dict(time=iso(), attempt=ctx.attempt, G1=g1, G2=g2)
    if not g1["ok"] or not g2["ok"]:                     # G1: refuse to run with any other plan file (section 14), also in a self-test
        dump_json(dict(check, result="stopped: G1/G2"), os.path.join(ctx.work, "start_check.json")); sync(ctx.work, ctx.drive)
        say("stopped before step 1: " + ("G1 (plan freeze) failed" if not g1["ok"] else "G2 (A100) failed"))
        return EXIT_GATE
    # start conditions (7.3)
    sc = start_conditions(ctx)
    check["start"] = sc
    if not sc["ok"]:                                     # recorded as an attempt (result 'waiting'), so that a wave-2 session that
        dump_json(dict(check, result="waiting"), os.path.join(ctx.work, "start_check.json"))     # never starts can be declared (gates-9)
        attempts.append(dict(attempt=ctx.attempt, n=len(attempts) + 1, started=iso(t_attempt), ended=iso(), code_commit=ctx.code_commit,
                             gpu=g2.get("gpu"), t0=iso(ctx.t0), result="waiting", waiting=sc["waiting"], ran=[]))
        dump_json(attempts, os.path.join(ctx.work, "session_attempts.json"))
        sync(ctx.work, ctx.drive)
        for w in sc["waiting"]:
            say(f"waiting: {w}")
        say(f"start conditions not met (7.3): nothing was run (waiting run recorded as attempt {len(attempts)} in session_attempts.json)")
        return EXIT_WAIT
    dump_json(dict(check, result="started"), os.path.join(ctx.work, "start_check.json"))
    attempts.append(dict(attempt=ctx.attempt, n=len(attempts) + 1, started=iso(), code_commit=ctx.code_commit, gpu=g2.get("gpu"),
                         t0=iso(ctx.t0)))
    dump_json(attempts, os.path.join(ctx.work, "session_attempts.json"))
    if ctx.blk:
        say(f"blocks: B3 = 14, b_lo {ctx.blk['b_lo']}, b_hi {ctx.blk['b_hi']}; order by R1 delta {ctx.blk.get('order')}; "
            f"missing {ctx.blk.get('missing_blocks')}; labels {ctx.blk['labels'] or 'none'}")
    # models (G33 for r2_q3) and the copy-in (7.3)
    ctx.models = models_record(ctx)
    for m in SESSIONS[base]["models"]:
        if m == M17 and not ctx.models[m]["ok"] and not (ctx.test or ctx.selftest):
            say(f"stopped: {m} missing or its checksum failed"); return EXIT_GATE
    meta, copied, choices, skipped = SESSIONS[base], [], [], []
    prevrec = {}
    for p in sorted(glob.glob(os.path.join(glob.escape(ctx.work), "manifest_*.json"))):
        for r in (load_json(p) or {}).get("copied", []):
            prevrec[(r["kind"], r["path"])] = r["sha256"]
    if meta["r1"]:
        files, choices, skipped = plan_r1(ctx.src, r1_patterns(ctx), ctx.blk)
        copied += [dict(r, kind="r1") for r in copy_tree_files(files, ctx.r1, "round-1 files")]
    if meta["r2"]:
        copied += [dict(r, kind="r2") for r in copy_tree_files(plan_r2(ctx.src, base), ctx.r2in, "round-2 inputs", keep_top=("r2_ana",),
                                                               lock_dirs=False)]
    changed = [r["path"] for r in copied if prevrec.get((r["kind"], r["path"])) not in (None, r["sha256"])]
    if changed:
        say(f"WARNING: {len(changed)} input files differ from an earlier attempt of this session: {changed[:5]}")
    ctx.inputs_digest = hashlib.sha256("\n".join(sorted(f"{r['kind']}:{r['path']}:{r['sha256']}" for r in copied)).encode()).hexdigest()
    steps = resolve_steps(ctx)
    invalidated = invalidate_changed(ctx, steps) if base in RERUN_ON_INPUT_CHANGE else []
    attempts[-1]["invalidated"] = invalidated
    dump_json(attempts, os.path.join(ctx.work, "session_attempts.json"))
    shards_read = sorted({r["source"] for r in copied if r["kind"] == "r1" and r["source"] in SHARDS   # shards whose data (not only
                          and r["path"].split("/", 1)[1] not in ("shard_done.json", "code_commits.txt")})        # metadata) is read
    rmeta = r1_meta(ctx.r1) if meta["r1"] else {}
    g3 = gate_G3(rmeta, shards_read)
    if g3["labels"]:
        say(f"G3: {g3['labels']}")
    versions = dict(gpu=g2.get("gpu"), cuda=g2.get("cuda"), torch=version("torch"), transformers=version("transformers"),
                    python=platform.python_version())
    g47 = None
    if base in ("r2_q1", "r2_q2"):
        r1v = {k: (rmeta.get("gpu1") or {}).get(k) for k in ("gpu", "torch", "transformers")}
        g47 = dict(r1_gpu1=r1v, session={k: versions[k] for k in ("gpu", "torch", "transformers")},
                   same=all(r1v[k] == versions[k] for k in r1v), note="reported with the match rate (r2_axes.json); no rule depends on it")
    ctx.manifest_path = os.path.join(ctx.work, f"manifest_{ctx.attempt}.json")
    if os.path.exists(ctx.manifest_path):
        k = 1
        while os.path.exists(ctx.manifest_path[:-5] + f"_{k}.json"):
            k += 1
        os.replace(ctx.manifest_path, ctx.manifest_path[:-5] + f"_{k}.json")
    fin = sc.get("R1_final") if "R1_final" in sc else r1_final(ctx.src)
    fpath = os.path.join(ctx.src, "r1_finality.json") if ctx.src else None
    manifest = dict(
        session=base, attempt=ctx.attempt, n_attempt=len(attempts), wave=meta["wave"], started=iso(t_attempt), t0=iso(ctx.t0),
        test_run=ctx.test, selftest=ctx.selftest, plan=g1,
        freeze=dict(commit=g1["freeze_commit"], time=g1["freeze_time"], commit_constant=FREEZE_COMMIT, time_constant=FREEZE_TIME,
                    plan_sha256=g1["sha256"], amendments_sha256=g1["amendments"]["sha256"]),
        code=dict(commit=ctx.code_commit, commits_this_work_dir=(read_text(prev) or "").split(),
                  runner_sha256=sha256_file(os.path.abspath(__file__)),
                  scripts_dir=os.path.relpath(SCRIPTS, REPO) if SCRIPTS.startswith(REPO) else SCRIPTS),
        runtime=dict(versions, platform=platform.platform(), cpus=os.cpu_count(), device=ctx.device, nt=ctx.nt), gates=dict(
            G1=dict(ok=g1["ok"]), G2=g2, G3=g3, G4=sc.get("G4"), G5=sc.get("G5"), upstream=sc.get("upstream"),
            G7=(ctx.blk or {}).get("G7"), G8=(ctx.blk or {}).get("G8"), G9=(ctx.blk or {}).get("G9"), G47=g47),
        blocks=ctx.blk, r1_meta=rmeta, r1_modules=g3["modules"],
        r1_final=dict(path="dsse_runs/r1_finality.json", record=fin,
                      sha256=sha256_file(fpath) if fpath and os.path.isfile(fpath) else None,
                      drive_mtime=iso(os.path.getmtime(fpath)) if fpath and os.path.isfile(fpath) else None),
        copied=copied, duplicate_choices=choices, not_copied=skipped, inputs_changed_since_earlier_attempt=changed,
        inputs_digest=ctx.inputs_digest, steps_redone_for_changed_inputs=invalidated, packing=PACKING.get(base, "as in plan 7.2"),
        drive_metadata=drive_meta(ctx.src, g1["freeze_time"] or FREEZE_TIME),
        models=ctx.models, references=references_record(),
        declarations=dict(R2_DECLARED_NOT_RUN=declared("R2_DECLARED_NOT_RUN")),
        time_guard_s=ctx.guard_s if math.isfinite(ctx.guard_s) else None, steps=[s.name for s in steps])
    dump_json(manifest, ctx.manifest_path)
    say(f"manifest written: {os.path.basename(ctx.manifest_path)} ({len(copied)} copied files; references {manifest['references']['n_ok']}"
        f"/{len(REF_SHA256)} present)")
    comp = manifest["references"]["comparison"]
    if comp["status"] != "ok":
        say(f"WARNING: reference comparison {comp['path']}: {comp['status']}" + (f" ({comp.get('changed')})" if comp.get("changed") else ""))
    todo = would_run(ctx, steps)
    if todo:                                             # not done until this attempt finishes (runner_steps-5, runner_gates-8)
        remove_done_record(ctx, f"steps to run: {', '.join(todo)}")
    sync(ctx.work, ctx.drive)
    st, ran, guard = run_steps(ctx, steps)
    return finalize(ctx, steps, st, ran, guard, attempts, copied, t_attempt)


# ======================================================================================================================================
# Dry run: the start conditions, copy lists and step commands, without executing or writing anything
# ======================================================================================================================================
def dry_run(session, root, work, drive, verbose=False):
    base, explicit = split_session(session)
    ctx = Ctx(base, explicit or base, root, work, drive, dry=True, verbose=verbose)
    meta = SESSIONS[base]
    print(f"== dry run of {base} (wave {meta['wave']}, {meta['kind']}); work {ctx.work}; drive {ctx.drive}; source {ctx.src}")
    g1 = gate_G1()
    print(f"G1 plan freeze: {'ok' if g1['ok'] else 'NOT ok'}: sha256 {g1['sha256']} (expected {FROZEN_PLAN_SHA256[:12]}...), freeze commit "
          f"{g1['freeze_commit']}, ancestor of HEAD {g1['head'][:12]}: {g1['freeze_is_ancestor']}")
    print(f"amendments: {AMEND_REL} sha256 {g1['amendments']['sha256']}" if g1["amendments"]["sha256"] else
          f"WARNING: the amendments file {AMEND_REL} is missing in this checkout")
    print("G2 GPU: checked at run time (an A100)" + (" - skipped, RUN_DEVICE=cpu" if ctx.test else ""))
    print(f"round-1 folders read: {', '.join(meta['shards']) or 'none'}" + (f"; version metadata only: {', '.join(meta['meta_shards'])}"
                                                                           if meta["meta_shards"] else ""))
    src_ok = bool(ctx.src and os.path.isdir(ctx.src))
    if meta["r1"] or meta["r2"] or meta["upstream"]:
        if not src_ok:
            print(f"source folder not available ({ctx.src}): copy patterns only")
        else:
            sc = start_conditions(ctx)
            print(f"start conditions (7.3): {'met' if sc['ok'] else 'NOT met'}")
            for w in sc["waiting"]:
                print(f"  waiting: {w}")
            fin = sc.get("R1_final")
            print("R1.final (dsse_runs/r1_finality.json): " + ("none" if not fin else
                  ", ".join(f"{s}: {final_by_record(fin, s)[1]} runs, final {(fin.get(s) or {}).get('final')}" for s in sorted(fin))))
            if ctx.blk:
                print(f"blocks: b_lo {ctx.blk['b_lo']}, b_hi {ctx.blk['b_hi']}, final {ctx.blk.get('final_blocks')}, missing "
                      f"{ctx.blk.get('missing_blocks')}, order {ctx.blk.get('order')}, labels {ctx.blk['labels']}, G7 {ctx.blk['G7']['ok']}, "
                      f"G8 {ctx.blk['G8'].get('ok')}, G9 {ctx.blk['G9'].get('ok')}")
    if meta["r1"]:
        print(f"copy list, round 1 (into {ctx.r1}/<shard>/, read-only):")
        files = plan_r1(ctx.src, r1_patterns(ctx), ctx.blk)[0] if src_ok else []
        full = lambda f: f"{f[0]}/{f[1]}" if f[0] else f[1]
        for pat in r1_patterns(ctx):
            hits = [f for f in files if any(fnmatch_path(full(f), e) for e in expand([pat]))]
            per = ", ".join("%s: %d" % (sh or "dsse_runs", sum(h[0] == sh for h in hits)) for sh in sorted({h[0] for h in hits}))
            print(f"  {pat}: {len(hits)} files" + (f" ({per})" if hits else ""))
            if verbose:
                for f in hits:
                    print(f"      {full(f)}")
    if meta["r2"]:
        print(f"copy list, round 2 (into {ctx.r2in}/<session>/):")
        items = plan_r2(ctx.src, base) if src_ok else []
        for pat in meta["r2"]:
            print(f"  {pat if pat != R2_ALL else 'every output of ' + ', '.join(meta['upstream']) + ' (without local/, logs/, rk/b*/state.pt)'}: "
                  f"{len(items) if pat == R2_ALL else sum(1 for i in items if fnmatch_path(f'{i[0]}/{i[1]}', pat))} files")
            if verbose:
                for u, rel, _ in items:
                    print(f"      {u}/{rel}")
    try:
        ctx.models = models_record(ctx)
    except Exception:                                      # noqa: BLE001 - dry run: report only
        ctx.models = {}
    steps = resolve_steps(ctx)
    by_name, st = {s.name: s for s in steps}, {}
    print(f"steps ({len(steps)}; outputs relative to <work>; each skipped when its outputs exist; packing: {PACKING.get(base, 'as in plan 7.2')}):")
    for i, rs in enumerate(steps, 1):
        pre = step_state(ctx, rs, st, by_name)            # the order of run_steps: done first, then 'not run', else it would run
        st[rs.name] = pre if pre and pre["state"] != "blocked" else dict(state="ran")
        state = ("done" if pre["state"] == "skipped" else pre["state"]) + f" ({pre['note']})" if pre and pre["state"] != "blocked" else "to run"
        deps = (f"   after: {', '.join(rs.after)}" if rs.after else "") + (f"   settled: {', '.join(rs.settled)}" if rs.settled else "")
        print(f"  [{i:2d}] {rs.name:<16} {rs.cmdline(ctx, short=True)}")
        print(f"       -> {' '.join(rs.out)}" + deps + f"   [{state}]")
    return EXIT_DONE


def fnmatch_path(path, pattern):
    """Glob match of a relative path ('*' does not cross '/'; 'gpu*' = gpu1 ... gpu6)."""
    rx = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*").replace(r"\?", "[^/]")
    return re.fullmatch(rx, path) is not None


# ======================================================================================================================================
# Self-test (MOCK / QUICK) on a mock dsse_runs tree; the notebooks run it before the main run
# ======================================================================================================================================
def fake_delta():
    """The fake dn_delta.json of the code tests (the same values as r2_rk.fake_delta, so that every self-test can share one mock tree;
    not a round-1 output): delta lowest at block 14 (0.09), b_lo = 8 (0.10), b_hi = 20 (0.20), so that rk_verdict.choose_blocks gives
    (8, 20) while the delta order b(1) < b(2) < b(3) is (14, 8, 20) - the label-against-order case of 2.7 C5."""
    bl = {}
    for l in range(28):
        d = {14: 0.09, 8: 0.10, 20: 0.20}.get(l, 0.12 + 0.004 * abs(l - 14))
        bl[str(l)] = {r: dict(F=0.5 + f * d, floor=0.5, delta=f * d, delta_rel=(0.5 + f * d) / 0.5, probe=0.5, chance=0.125)
                      for r, f in (("last", 1.0), ("mean", 0.9))}
    return dict(model="MOCK (fake dn_delta.json for code tests; not a round-1 output)", blocks=bl, mock=True)


def sh(cmd, env, log=None):
    say("[mock] " + " ".join(os.path.basename(c) if i == 1 else c for i, c in enumerate(cmd[:6])) + (" ..." if len(cmd) > 6 else ""))
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    if log:
        with open(log, "a") as f:
            f.write(r.stdout)
    if r.returncode:
        print(r.stdout[-3000:], flush=True)
        raise SystemExit(f"mock round-1 tree: {' '.join(cmd[:4])} failed (rc {r.returncode})")
    return r.stdout


MOCK_DEVICE = "cpu"   # the device of the mock round-1 tree and of the library tests (runner_steps-1: a regenerated text equals the
                      # stored one only on the same device, 2.3; CPU and CUDA generators give different streams for one seed)


def lib_nt():
    """Threads of the mock rk build and of the library tests: the same value for both (default 1, as the scripts' authors ran them)."""
    return os.environ.get("R2_LIB_NT") or "1"


def mock_npar():
    return int(os.environ.get("R2_MOCK_NPAR") or max(1, min(6, (os.cpu_count() or 2) - 2)))


def run_parallel(jobs, npar, what):
    """jobs: [(label, [(cmd, env, log_path), ...])]; the commands of one job run in order, up to npar jobs at a time (round-1 job-pool
    pattern). Raises SystemExit naming the failed jobs (their logs stay in place)."""
    todo, running, failed = list(jobs), [], []
    while todo or running:
        while todo and len(running) < npar:
            label, cmds = todo.pop(0)
            running.append([label, list(cmds), None, None])
        for r in list(running):
            label, cmds, proc, fh = r
            if proc is not None and proc.poll() is None:
                continue
            if proc is not None:
                fh.close()
                if proc.returncode:
                    failed.append(f"{label} (rc {proc.returncode}, log {fh.name})"); running.remove(r); continue
            if not cmds:
                running.remove(r); continue
            cmd, env, log = cmds.pop(0)
            say("[mock] " + " ".join(os.path.basename(c) if i == 1 else c for i, c in enumerate(cmd[:6])) + (" ..." if len(cmd) > 6 else ""))
            fh = open(log, "a")
            r[2], r[3] = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env), fh
        time.sleep(0.2)
    if failed:
        raise SystemExit(f"mock round-1 tree: {what} failed: {failed}")


def mock_r1(root, dr, parts=("rk", "pc")):
    """A mock dsse_runs tree in the round-1 layout (code tests only, plan section 9), built on the CPU (MOCK_DEVICE, also when the
    session runs on CUDA: the library tests that compare regenerated texts with the stored ones run on the CPU, runner_steps-1 / rk-1;
    the session's MOCK steps on CUDA assert no equality with the mock texts): rk_loop.py MOCK=1 'pilot' and 'all' at blocks 14, 8 and
    20 (one process chain per block, R2_MOCK_NPAR at a time, R2_LIB_NT threads each; R2_MOCK_FAST=1: blocks 8 and 20 as copies of
    block 14 with the block patched, for stub tests only), rk_verdict.py summaries, a fake dn_delta.json and blocks_chosen.json
    (b_lo 8, b_hi 20) in gpu2 and gpu3; pc_meta.py QUICK=1 jobs at r2_pc_toy's QUICK widths and seeds under pc_meta's own '_quick'
    names (pc_mup W 64 in gpu4 and W 128 in gpu3, bp_mup W 64 and 128 in gpu5, seeds 0 and 1; runner_steps-2), pc_expected (the
    '_quick' names) and pc_summary files; shard_done.json (with "device": "cpu") and code_commits.txt in every shard, and
    mock_fixture.json (no r1_finality.json: every mock shard is final by its shard_done.json)."""
    root, dr = os.path.abspath(root), os.path.abspath(dr)
    os.makedirs(dr, exist_ok=True)
    head = git_head()
    nt = lib_nt()
    env = dict(os.environ, MOCK="1", DEVICE=MOCK_DEVICE, NT=nt, OMP_NUM_THREADS=nt, PYTHONUNBUFFERED="1",
               POOLDIRS=os.path.join(root, "gpu_pkg", "rk_dirs.pt"), PERSONA=os.path.join(root, "rsi_dl", "persona"))
    for k in ("QUICK", "ASCALE"):
        env.pop(k, None)
    log = os.path.join(dr, "mock_build.log")
    npar = mock_npar()
    ok_rk = os.path.join(dr, f".mock_rk_{MOCK_DEVICE}_ok")          # a tree of an older runner (built on RUN_DEVICE) is rebuilt
    if "rk" in parts and not os.path.exists(ok_rk):
        t = time.time()
        fast = os.environ.get("R2_MOCK_FAST") == "1"
        blocks = (("gpu1", 14), ("gpu2", 8), ("gpu3", 20))
        for shard, b in blocks:
            shutil.rmtree(os.path.join(dr, shard, "rk", f"b{b}"), ignore_errors=True)
            os.makedirs(os.path.join(dr, shard, "rk"), exist_ok=True)
        rk = lambda shard, b: [([PY, os.path.join(AJS, "rk_loop.py"), root, os.path.join(dr, shard, "rk"), stage], dict(env, BLOCK=str(b)),
                                os.path.join(dr, f"mock_rk_b{b}.log")) for stage in ("pilot", "all")]
        say(f"[mock] rk_loop.py MOCK at blocks {[b for _, b in blocks]} on the {MOCK_DEVICE} ({npar} at a time, NT={nt})")
        run_parallel([(f"rk_loop b{b}", rk(shard, b)) for shard, b in blocks if b == 14 or not fast], npar, "rk_loop.py MOCK")
        for shard, b in blocks:
            out = os.path.join(dr, shard, "rk")
            if fast and b != 14:
                import torch
                src, dst = os.path.join(dr, "gpu1", "rk", "b14"), os.path.join(out, f"b{b}")
                shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
                for f in ("k6.json",):
                    if os.path.exists(os.path.join(dst, f)):
                        os.remove(os.path.join(dst, f))
                L = load_json(os.path.join(dst, "loops.json"))
                L["_meta"]["block"] = b; dump_json(L, os.path.join(dst, "loops.json"))
                ST = torch.load(os.path.join(dst, "state.pt"), map_location="cpu", weights_only=False)
                for k in [k for k in ST if k.startswith("k6")]:
                    del ST[k]
                if isinstance(ST.get("loops"), dict) and "_meta" in ST["loops"]:
                    ST["loops"]["_meta"]["block"] = b
                torch.save(ST, os.path.join(dst, "state.pt"))
            sh([PY, os.path.join(AJS, "rk_verdict.py"), "block", os.path.join(out, f"b{b}"), os.path.join(dr, shard, f"summary_b{b}.json")],
               env, log)
        fd = fake_delta()
        for shard in ("gpu2", "gpu3"):
            dump_json(fd, os.path.join(dr, shard, "dn_delta.json"))
            dump_json(dict(b_lo=8, b_hi=20, delta={str(l): round(fd["blocks"][str(l)]["last"]["delta"], 4) for l in range(6, 25)}, mock=True),
                      os.path.join(dr, shard, "blocks_chosen.json"))
        dump_json(dict(builder="r2_run_session.mock_r1", device=MOCK_DEVICE, nt=nt, torch=version("torch"), built=iso(),
                       blocks={str(b): dict(device=MOCK_DEVICE, shard=shard, copy_of_14=bool(fast and b != 14)) for shard, b in blocks}),
                  os.path.join(dr, "mock_fixture.json"))
        open(ok_rk, "w").write(iso())
        say(f"[mock] rk part ready ({time.time() - t:.0f}s)")
    ok_pc = os.path.join(dr, ".mock_pc_quick_ok")                    # a tree of an older runner (other widths and names) is rebuilt
    if "pc" in parts and not os.path.exists(ok_pc):
        t = time.time()
        jobs = ([("gpu5", "bp_mup", W, s, {}) for W in (64, 128) for s in (0, 1)]                # C-T7's BP side, G-meta rows
                + [("gpu4", "pc_mup", 64, s, dict(T="20", BETA="0.1")) for s in (0, 1)]
                + [("gpu3", "pc_mup", 128, s, dict(T="20", BETA="0.1")) for s in (0, 1)])
        names = {}
        for shard in SHARDS:                                          # files of an older mock (other widths, names without '_quick')
            shutil.rmtree(os.path.join(dr, shard, "pc"), ignore_errors=True)
            for f in glob.glob(os.path.join(glob.escape(os.path.join(dr, shard)), "pc_*_gpu*.json")):
                os.remove(f)
        say(f"[mock] {len(jobs)} pc_meta.py QUICK jobs on the CPU, {npar} at a time (round-1 job-pool pattern; a code fixture only)")
        todo = []
        for shard, arm, W, s, ex in jobs:
            o = os.path.join(dr, shard, "pc"); os.makedirs(o, exist_ok=True)
            todo.append((f"{arm} W{W} s{s}", [([PY, os.path.join(AJS, "pc_meta.py"), o, arm, str(W), str(s), "12"],
                                               dict(env, QUICK="1", NT="1", OMP_NUM_THREADS="1", DEVICE="cpu", **ex),
                                               os.path.join(dr, f"mock_{arm}_W{W}_s{s}.log"))]))
            names.setdefault(shard, []).append(f"{arm}_W{W}_s{s}_K12_g0_wd1" + ("_T20_beta0.1" if arm == "pc_mup" else "") + "_quick")
        run_parallel(todo, npar, "pc_meta.py QUICK")
        for shard, nm in names.items():
            dump_json(nm, os.path.join(dr, shard, f"pc_expected_{shard}.json"))
            sh([PY, os.path.join(AJS, "pc_summary.py"), os.path.join(dr, shard, "pc"), os.path.join(dr, shard, f"pc_summary_{shard}.json")], env, log)
        open(ok_pc, "w").write(iso())
        say(f"[mock] pc part ready ({time.time() - t:.0f}s)")
    for s in SHARDS:
        p = os.path.join(dr, s, "shard_done.json")
        rec = load_json(p) or dict(shard=s, gpu="MOCK", torch=version("torch"), transformers=version("transformers"), code_commit=head,
                                   code_commits_used=[head], mock=True)
        if rec.get("device") != MOCK_DEVICE:                       # r2_rk.fixture_device reads it (gpu1) for its device diagnostics
            rec["device"] = MOCK_DEVICE; dump_json(rec, p)
        if not os.path.exists(os.path.join(dr, s, "code_commits.txt")):
            with open(os.path.join(dr, s, "code_commits.txt"), "w") as f:
                f.write(head + "\n")
    return dr


LIB_SELFTESTS = {   # the scripts' own code tests (plan 9; NOTES of the implementers), run by --selftest before the session's MOCK run.
    # r2_q3 (implementer 3's timings on one contended CPU core, MOCK): r2_rgi_geom.py selftest 39 s, the same for Qwen3-0.6B 40 s, then
    # r2_rgi_stats.py selftest 54 s in the same folder (it reuses the 1.7B geometry of the first test; alone 74 s)
    "r2_q1": ("stimuli", "rk", "axes", "gen14", "gen8"), "r2_q2": ("stimuli", "rk"), "r2_q3": ("stimuli", "geom", "geom06", "stats"),
    **{s: ("toy", "sim") for s in ("r2_t1", "r2_t2", "r2_t3", "r2_t4", "r2_pc", "r2_ana")},
}


def lib_selftests(base, root, st_root, dr):
    """r2_stimuli.py check with the real tokenizer; r2_rk.py, r2_axes.py and r2_gen_loop.py selftest (MOCK=1) on copies of the mock
    round-1 tree (so that they cannot change it); r2_rgi_geom.py selftest for both models and r2_rgi_stats.py selftest (MOCK=1, their own
    tiny models, one folder); r2_toy_common.py selftest
    (QUICK=1, incl. the W0 equivalence tests and the reference comparison) and r2_sim.py selftest. All on the CPU (MOCK_DEVICE, the
    device the mock tree was built on) with R2_LIB_NT threads (default 1), as their authors ran them. 0 if all pass."""
    def copy_mock(dst):
        if not os.path.isdir(dst):
            for s in SHARDS:
                if os.path.isdir(os.path.join(dr, s)):
                    shutil.copytree(os.path.join(dr, s), os.path.join(dst, s))
        return dst
    sc = lambda n: os.path.join(SCRIPTS, n)
    os.makedirs(st_root, exist_ok=True)
    env = dict(os.environ, DEVICE=MOCK_DEVICE, NT=lib_nt(), OMP_NUM_THREADS=lib_nt(), PYTHONUNBUFFERED="1")
    for k in ("MOCK", "QUICK", "BLOCK", "R2_SELFTEST", "R1_SRC"):
        env.pop(k, None)
    r1s, r2s = os.path.join(st_root, "scripts_r1"), os.path.join(st_root, "scripts_r2")
    for t in LIB_SELFTESTS.get(base, ()):
        if t == "stimuli":
            cmd, ex = [sc("r2_stimuli.py"), "check", os.path.join(root, "models", "qwen3-1.7b")], {}
        elif t == "rk":
            copy_mock(os.path.join(st_root, "rk_selftest", "r1"))
            cmd, ex = [sc("r2_rk.py"), "selftest", root, os.path.join(st_root, "rk_selftest")], dict(MOCK="1")
        elif t == "axes":
            cmd, ex = [sc("r2_axes.py"), root, copy_mock(r1s), r2s, "selftest"], dict(MOCK="1")
        elif t.startswith("gen"):
            cmd, ex = [sc("r2_gen_loop.py"), root, copy_mock(r1s), r2s, "selftest"], dict(MOCK="1", BLOCK=t[3:])
        elif t in ("geom", "geom06", "stats"):              # read no round-1 file (an empty r1 folder); one <r2_dir> for the three
            os.makedirs(os.path.join(st_root, "geom_r1"), exist_ok=True)
            cmd = [sc("r2_rgi_stats.py" if t == "stats" else "r2_rgi_geom.py"), root, os.path.join(st_root, "geom_r1"),
                   os.path.join(st_root, "geom"), "selftest"] + (["--model", M06] if t == "geom06" else [])
            ex = dict(MOCK="1")
        elif t == "sim":
            cmd, ex = [sc("r2_sim.py"), "selftest"], {}
        else:                                               # NOTES_w5: QUICK=1 NT=1 OMP_NUM_THREADS=1 python r2_toy_common.py selftest <dir>
            cmd, ex = [sc("r2_toy_common.py"), "selftest", os.path.join(st_root, "toy_common")], dict(QUICK="1", NT="1", OMP_NUM_THREADS="1")
        log, t0 = os.path.join(st_root, f"lib_{t}.log"), time.time()
        envs = " ".join("%s=%s" % kv for kv in ex.items())
        say(f"[selftest] {envs} {os.path.basename(cmd[0])} {cmd[1] if t in ('stimuli', 'rk', 'toy', 'sim') else 'selftest'}"
            f"{' --model ' + M06 if t == 'geom06' else ''} (DEVICE={env['DEVICE']}, NT={dict(env, **ex)['NT']})")
        with open(log, "w") as f:
            rc = subprocess.run([PY, "-u"] + cmd, env=dict(env, **ex), stdout=f, stderr=subprocess.STDOUT).returncode
        tail = (read_text(log) or "").strip().splitlines()
        last = next((l for l in reversed(tail) if l.strip() not in ("", "}", "]", "},", "],")), "")
        say(f"[selftest] {os.path.basename(cmd[0])}: rc {rc} ({time.time() - t0:.0f}s) {last.strip()[:160]}")
        if rc != 0:
            print("\n".join(tail[-40:]), flush=True)
            say(f"SELFTEST FAILED: {os.path.basename(cmd[0])} ({log})")
            return 1
    return 0


R1_USED = ("rk_loop.py", "rk_verdict.py", "ufo_common.py", "rsi_statements.py", "pc_meta.py", "pc_common.py", "pc_analyze.py",
           "pc_summary.py")   # the round-1 modules that the round-2 scripts import or the mock tree runs


def compile_check():
    """Every round-2 script (analysis/ju_space/round2/*.py) and the round-1 modules it uses must compile with this interpreter (Colab's
    Python; the code tests check 3.11, 3.12 and 3.13): a syntax error found here costs a self-test, not a session. Returns the failures."""
    files = sorted(glob.glob(os.path.join(glob.escape(SCRIPTS), "*.py"))) + [os.path.join(AJS, n) for n in R1_USED]
    bad = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                compile(fh.read(), f, "exec", dont_inherit=True)
        except FileNotFoundError:
            bad.append(f"{os.path.relpath(f, REPO)}: missing")
        except SyntaxError as e:
            bad.append(f"{os.path.relpath(f, REPO)}:{e.lineno}: {e.msg}")
    say(f"[selftest] compile check (Python {platform.python_version()}): {len(files)} files, " + (f"FAILED: {bad}" if bad else "ok"))
    return bad


def pc_selftest_check(work):
    """runner_steps-2: r2_pc's self-test must have run PC code: pc_toy/t7.json has a width with a PC side ('pc') and a
    pc_toy/cross_pc/w*.json lists a PC model (the coordinator's criterion); no pair at all is a WARNING only. Returns the problems."""
    t7 = load_json(os.path.join(work, "pc_toy", "t7.json")) or {}
    ran = sorted(w for w, v in (t7.get("widths") or {}).items() if isinstance(v, dict) and "pc" in v)
    cross = [load_json(p) or {} for p in sorted(glob.glob(os.path.join(glob.escape(work), "pc_toy", "cross_pc", "w*.json")))]
    n_pc = sum(1 for c in cross for m in (c.get("models") or []) if isinstance(m, dict) and m.get("kind") == "pc")
    n_pairs = sum(len(c.get("pairs") or []) for c in cross)
    say(f"[selftest] r2_pc content: C-T7 widths that ran {ran or 'none'}; cross_pc: {n_pc} PC models, {n_pairs} pairs")
    if n_pc and not n_pairs:
        say("WARNING: r2_pc self-test: cross_pc lists PC models but no pair (the pair code did not run)")
    return ([] if ran else ["pc_toy/t7.json: no width ran (no round-1 PC / BP net found?)"]) + \
           ([] if n_pc else ["pc_toy/cross_pc: no PC model"])


def selftest(session, root, st_root):
    """The scripts' own code tests (LIB_SELFTESTS), then a MOCK / QUICK run of the session through this runner (same code path as the
    real run) on a mock dsse_runs tree in <st_root>/dsse_runs; a wave-2 session first runs the sessions it reads (r2_pc: r2_t1;
    r2_ana: all others) the same way (without their library tests)."""
    base, _ = split_session(session)
    st_root = os.path.abspath(st_root); dr = os.path.join(st_root, "dsse_runs")
    os.makedirs(st_root, exist_ok=True)
    problems = []
    if compile_check():
        say("SELFTEST FAILED: compile check"); return 1
    comp = references_record()["comparison"]
    if str(comp.get("status", "")).startswith("stale"):
        problems.append(f"reference comparison {REF_COMPARISON} is {comp['status']}: {comp.get('changed')} (regenerate and commit it, 9.5)")
        say(f"WARNING: {problems[-1]}")
    chain = list(SESSIONS[base]["upstream"]) + [base]
    parts = set()
    for s in chain:
        parts |= {"rk"} if SESSIONS[s]["r1"] and "G4" in SESSIONS[s]["start"] else set()
        parts |= {"pc"} if SESSIONS[s]["r1"] and "G5" in SESSIONS[s]["start"] else set()
    if {"rk", "axes", "gen14"} & set(LIB_SELFTESTS.get(base, ())):
        parts.add("rk")
    t = time.time()
    if parts:
        mock_r1(root, dr, sorted(parts))
    if lib_selftests(base, root, st_root, dr):
        return 1
    for s in chain:
        if s != base and os.path.exists(os.path.join(dr, s, "session_done.json")):
            continue
        env = dict(os.environ, R2_SELFTEST="1", R1_SRC=dr)
        for k in ("R2_T0", "R2_DECLARED_NOT_RUN"):
            env.pop(k, None)
        say(f"[selftest] {s}")
        rc = subprocess.run([PY, "-u", os.path.abspath(__file__), s, root, os.path.join(st_root, "work", s), os.path.join(dr, s)], env=env).returncode
        if rc != 0:
            say(f"SELFTEST FAILED: {s} (exit code {rc}); logs in {os.path.join(st_root, 'work', s, 'logs')}")
            return 1
        if s == "r2_pc":
            problems += pc_selftest_check(os.path.join(st_root, "work", s))
    if problems:
        say(f"SELFTEST FAILED: {base}: " + "; ".join(problems))
        return 1
    say(f"SELFTEST OK: {base}" + (f" (with {', '.join(chain[:-1])})" if len(chain) > 1 else "") + f" in {time.time() - t:.0f}s")
    return 0


# ======================================================================================================================================
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--mock-r1":
        if len(argv) < 3:
            raise SystemExit("usage: r2_run_session.py --mock-r1 <root> <dsse_runs_dir> [rk,pc]")
        mock_r1(argv[1], argv[2], _names(argv[3]) if len(argv) > 3 else ("rk", "pc"))
        return 0
    if argv and argv[0] == "--compile-check":
        return 1 if compile_check() else 0
    flags = {a for a in argv if a.startswith("--")}
    pos = [a for a in argv if not a.startswith("--")]
    unknown = flags - {"--dry-run", "--verbose", "--selftest", "--sync-only"}
    if unknown or len(pos) not in (3, 4):
        print(__doc__)
        raise SystemExit(EXIT_GATE)
    session, root, work, drive = pos + [None] * (4 - len(pos))
    split_session(session)
    if "--dry-run" in flags:
        return dry_run(session, root, work, drive, verbose="--verbose" in flags)
    if "--selftest" in flags:
        return selftest(session, root, work)
    if "--sync-only" in flags:
        n = sync(os.path.abspath(work), drive and os.path.abspath(drive))
        say(f"sync: {n} files copied to {drive}")
        return 0
    return run_session(session, root, work, drive)


if __name__ == "__main__":
    sys.exit(main())
