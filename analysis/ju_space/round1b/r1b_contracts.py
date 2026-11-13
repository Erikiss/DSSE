"""Round 1b contracts (main §2.2, §2.3, §4.2, §4.4): the shared objects, every cross-session file (owner part, producing session,
path relative to the run root dsse_runs/, format, keys), the sessions with their start conditions, and the slots of the notebooks.
Model-free and dependency-free: imported by the runner, the slot runner, the notebook builder and the part code.

Paths: a cross-session file lives at <run root>/<session>/<rel> (Drive: My Drive/dsse_runs/ of the producing account). Consumers read
copies under <work>/in/<session>/<rel> (ctx.input(session, rel)); A -> B exports under <work>/in/r1b_xfer/<name> (ctx.xfer(name)).
Package files (frozen with the plan, sha256 in runde1b_freeze.json) are repository paths ('repo:<path>')."""

M17, M06, M35 = "qwen3-1.7b", "qwen3-0.6b", "qwen3.5-2b"
XFER_DIR = "r1b_xfer"               # account A: dsse_runs/r1b_xfer, shared read-only with B (SO-XFER)
XFER_LINK = "r1b_xfer_A"            # account B: My Drive/r1b_xfer_A (shortcut to A's dsse_runs/r1b_xfer)
RUNS_B_LINK = "dsse_runs_B"         # account A: My Drive/dsse_runs_B (shortcut to B's dsse_runs)
XFER_MANIFEST = "xfer_manifest.json"  # in r1b_xfer: name -> sha256, producing session, step, code commit (SG4 check in B)
DECLARED = "r1b_declared_not_run.json"   # operator file in dsse_runs of the consuming account: {"sessions": {"r1b_X": "reason"}}
PLAN_DIR = "results/ju_space/plans/runde1b"
PKG = "repo:" + PLAN_DIR + "/"

# ------------------------------------------------------------------------------------------------------------------------------------
# Shared objects (main §2.3): owner, sessions that make them, users, the files or code that carry them
# ------------------------------------------------------------------------------------------------------------------------------------
SHARED = {
    "SO-AX": dict(owner="R", made_in=("r1b_R_S0", "r1b_R_S2_14", "r1b_R_S2_18", "r1b_R_S2_06", "r1b_R_S5"), users="R K D L",
                  files=("so_ax_b14", "so_ax_b18", "so_ax_b6", "so_ax_check", "xfer_so_ax_ustar", "ax1b_b14", "ax1b_b18", "ax1b_b6",
                         "ax1b_lens")),
    "SO-DMG": dict(owner="R K", made_in=("r1b_R_S0",), users="R K L", files=("xfer_dmg_ref_R16", "ref_k16"),
                   code="r1b_core.RefSet, Damage, calibrate"),
    "SO-JLENS": dict(owner="R K", made_in=("r1b_K_S1", "r1b_R_S5"), users="R K", files=("k4j", "jlens_validity", "rg11"),
                     code="r1b_core.load_jlens (sha256 pin)"),
    "SO-CL": dict(owner="K", made_in=(), users="K", files=(), code="r1b_K_lens.py"),
    "SO-LENSOUT": dict(owner="K", made_in=("r1b_K_S2a",), users="L", files=("lensout", "k4_gating")),
    "SO-HN": dict(owner="K", made_in=("r1b_K_S1",), users="K L D", files=("hn_set", "hn_detectors", "rs_sets", "hn_meta"),
                  code="r1b_hn.py (hook manager: r1b_core.Hooks.neuron)"),
    "SO-CHH": dict(owner="K", made_in=("r1b_K_S1",), users="K L D", files=("hn_meta",), code="r1b_hn.py"),
    "SO-LAB": dict(owner="K", made_in=("r1b_K_S1", "r1b_K_S2a"), users="K L", files=("so_lab", "k4_gating", "k1_label_bank")),
    "SO-LCE": dict(owner="D", made_in=("r1b_D_A",), users="D K L", files=("glce",), code="r1b_lce.py"),
    "SO-D1STIM": dict(owner="D", made_in=(), users="D L", files=("d1_stimuli",)),
    "SO-WT103": dict(owner="D K L", made_in=(), users="D K L P0", files=(), code="runde1b_freeze.json datasets.wikitext103"),
    "SO-LONGDOC": dict(owner="L", made_in=(), users="L P0", files=("longdoc_list",)),
    "SO-TURN": dict(owner="main", made_in=(), users="R D K L", files=(), code="round2: r2_stimuli, r2_theta.stage_turn2, r2_theta_verdict"),
    "SO-GVRA": dict(owner="K", made_in=(), users="R D K L", files=(), code="r1b_gvra.py"),
    "SO-PVAL": dict(owner="main", made_in=(), users="all", files=(), code="r1b_core (interval, p_one_sided, p_rank, holm, bh, boot)"),
    "SO-KEYS": dict(owner="main", made_in=(), users="all", files=("keys_registry",), code="r1b_core (KEYS, key, keyseed, gumbel)"),
    "SO-PILOT": dict(owner="main", made_in=("r1b_P0",), users="P0", files=("pilot_material", "p0_record")),
    "SO-XFER": dict(owner="main", made_in=("r1b_R_S0",), users="all cross-account starts", files=("xfer_dmg_ref_R16", "xfer_so_ax_ustar")),
}

# ------------------------------------------------------------------------------------------------------------------------------------
# Cross-session files. session=None: package file (repository, frozen). xfer=<name>: also published to r1b_xfer/<name> (A -> B).
# fmt: 'json' (provenance block inside), 'pt' (torch dict; sidecar <file>.prov.json), 'jsonl' (sidecar). keys: the top-level keys.
# ------------------------------------------------------------------------------------------------------------------------------------
def _f(owner, session, rel, fmt, keys, users, so=None, xfer=None, note=""):
    return dict(owner=owner, session=session, rel=rel, fmt=fmt, keys=keys, users=users, so=so, xfer=xfer, note=note)


FILES = {
    # part R
    **{f"so_ax_b{b}": _f("R", "r1b_R_S0", f"so_ax/so_ax_b{b}.pt", "pt",
                         "AX1: u_T_perp, u_sea, u_sports, u_tone, u_neg (from R1.state[b]['axes'] u_T_perp, sea, sports, tone, u_neg; "
                         "float64 [2048]); U*: u_star (copied unchanged), u_star_src {path, sha256}; u_star_minus {loop: [2048]} "
                         "(leave-loop-out modes); label (None or 'SO-AX reproduction failed')", "K D L", "SO-AX") for b in (14, 18, 6)},
    "so_ax_check": _f("R", "r1b_R_S0", "so_ax/so_ax_check.json", "json",
                      "blocks.<b>: cos_rebuild (>= 1 - 1e-4), image_cos_d1_u_star (rebuilt, TV), diff, ok; ok; label", "K D L", "SO-AX",
                      note="SG10"),
    "xfer_dmg_ref_R16": _f("R", "r1b_R_S0", "xfer/dmg_ref_R16.json", "json",
                           "items [{p, prompt_ids, cont_ids}] (round-1 prompts 0-7 and 32-39, R1.state[14]['ref'] continuations); "
                           "source {path, sha256}; equal_at_18_6 {b: bool, diffs}", "K L", "SO-DMG", xfer="dmg_ref_R16.json"),
    "xfer_so_ax_ustar": _f("R", "r1b_R_S0", "xfer/so_ax_ustar.pt", "pt", "u_star {14, 18, 6: float64 [2048]}; src {b: {path, sha256}}",
                           "K L", "SO-AX", xfer="so_ax_ustar.pt"),
    "r0_table": _f("R", "r1b_R_S0", "r0_table.json", "json", "R0 re-readings ('post hoc (R0)')", "R ana"),
    "r8_1": _f("R", "r1b_R_S0", "r8_1.json", "json", "R8.1 (round-2 0.6B file)", "R ana"),
    **{f"ax1b_b{b}": _f("R", f"r1b_R_S2_{b:02d}", f"ax1b/ax1b_b{b}.pt", "pt",
                        "u_T, u_T_perpstar, u_T_1b, verdict axes u_T_v, u_sea_v (and the other u_X_v), fold axes, kappa, centred nulls, "
                        "aucs (R.1 AX-1b)", "K D", "SO-AX") for b in (14, 18, 6)},
    "ax1b_lens": _f("R", "r1b_R_S5", "ax1b/ax1b_lens_b{b}.pt", "pt", "AX-1b at the lens blocks {2, 6, 10, 14, 18, 22, 24, 26}", "R", "SO-AX"),
    "jlens_validity": _f("R", "r1b_R_S5", "jlens_validity.json", "json", "blocks.<b>: valid (bool), metrics; RG10", "K r1b_ana", "SO-JLENS"),
    "rg11": _f("R", "r1b_R_S5", "rg11.json", "json", "decision ('refit' | 'no refit'), metrics", "r1b_R_S5L trigger", "SO-JLENS"),
    "r_verdict": _f("R", "r1b_R_ana", "r1b_R_verdict.json", "json", "R.6 strings, F+ / F- table, RG7, RG8a, RG8b, RG12, RG15", "r1b_ana"),
    "rg15": _f("R", "r1b_R_ana", "rg15.json", "json", "J1 (bool), J2 (bool)", "r1b_R_S8 trigger"),
    "r_verdict2": _f("R", "r1b_R_ana2", "r1b_R_verdict2.json", "json", "R5.4, R8.2 (conditional run)", "r1b_ana"),
    # part D
    "glce": _f("D", "r1b_D_A", "lce/g_lce_1_2.json", "json", "G-LCE-1 {pass, metrics}, G-LCE-2 {pass, kernel}; gpu, versions", "L K",
               "SO-LCE", note="SG9: start condition of r1b_L_B"),
    "d_verdict": _f("D", "r1b_D_ana", "d_verdict.json", "json", "D.17 string with qualifiers; Holm (a), (b1), (b2); triggers {R2: bool}",
                    "r1b_ana r1b_D_R"),
    "d1_stimuli": _f("D", None, PKG + "d1_stimuli.json", "json", "F-sym / F-asym referents, slots, versions, v3, Y_i, templates", "D L",
                     "SO-D1STIM"),
    # part K
    "k4j": _f("K", "r1b_K_S1", "k4j.json", "json", "convention ('output' | 'input'), G-K-J {pass, metrics}", "R", "SO-JLENS"),
    "hn_set": _f("K", "r1b_K_S1", "hn/hn_set.json", "json",
                 "H_tq, H_ar, H_plus (ordered by selection frequency, then |r|), F, H_op_name ('H_tq' | 'H+' | 'H_ar'), H_op, boot "
                 "(20 sets); neurons as [layer, index]", "L D", "SO-HN"),
    "hn_detectors": _f("K", "r1b_K_S1", "hn/hn_detectors.pt", "pt", "theta, C_star, refit L2 detectors", "L", "SO-HN"),
    "rs_sets": _f("K", "r1b_K_S1", "hn/rs_sets.json", "json", "RS39, RS39mc, RS19, RS20, RS10 ([layer, index] lists)", "L", "SO-HN"),
    "hn_meta": _f("K", "r1b_K_S1", "hn/hn_meta.json", "json",
                  "REF-W per-neuron mean and SD of z and CETT, p99.8; G-K1, G-K22 decisions", "L D", "SO-HN"),
    "sg8_preview": _f("K", "r1b_K_S1", "sg8_preview.json", "json", "G-K2 preview (string), H_op_name", "D L", "SO-HN", note="SG8"),
    "so_lab": _f("K", "r1b_K_S1", "lab/so_lab.json", "json",
                 "items {id: split (S_fit, S_lens, S_sess, S_L, S2, S3, S4a-c), class (CC, CI, ...), answers [10 x {text_ids, key}], "
                 "spans}; scorer versions", "L", "SO-LAB"),
    "k5a_calib": _f("K", "r1b_K_S3", "k5a_calib.json", "json", "K5a calibration (variant, metrics)", "K", "SO-CL"),
    "lensout": _f("K", "r1b_K_S2a", "lensout/so_lensout.jsonl", "jsonl",
                  "one record per chain: item, stratum, ops, variant, read block, position type, seed rank, key, tokens[s 0..16], "
                  "k_star_conf, share_H[s] with RS39 band, a_H[s]", "L", "SO-LENSOUT"),
    "k4_gating": _f("K", "r1b_K_S2a", "lab/k4_gating.json", "json",
                    "items {id: stratum ('correct' = known, 'wrong' = prone), greedy, samples[4] (T 0.7, keys)}", "L", "SO-LAB"),
    "k_verdict": _f("K", "r1b_K_ana", "k_verdict.json", "json", "K.7 composite; gates {G-K2, G-K4, G-K4b, G-K5, G-K14}; Holm families",
                    "r1b_ana r1b_K_S6"),
    "ref_k16": _f("K", None, PKG + "ref_k16.json", "json", "REF-K16 token ids as RefSet items [{ids, pos}]", "R L", "SO-DMG"),
    "k1_label_bank": _f("K", None, PKG + "k1_label_bank/", "dir", "K1 label bank (items, hashes, splits, frames, scorer tests)", "K L",
                        "SO-LAB"),
    # part L
    "l_pop": _f("L", "r1b_L_A", "l/l_pop.json", "json", "L.POP (P, populations)", "L"),
    "l3b_twin0": _f("L", "r1b_L_A", "l/l3b_twin0.json", "json", "L3b twin-0 rows (read by L3a)", "L"),
    "l_session_A": _f("L", "r1b_L_A", "l_session_A.json", "json", "L1, L3b, L4 statistics (+ X2, X6)", "r1b_ana"),
    "l_session_B": _f("L", "r1b_L_B", "l_session_B.json", "json", "L2, L3a, L3c, L3d; L5 generation (+ X1, X3-X5)", "r1b_ana"),
    "longdoc_list": _f("L", None, PKG + "longdoc_slices.json", "json", "SO-LONGDOC books and slices", "L P0", "SO-LONGDOC"),
    # main
    "l_verdicts": _f("L", "r1b_ana", "l_verdicts.json", "json", "L-ANA verdicts", "r1b_ana"),
    "summary": _f("main", "r1b_ana", "runde1b_summary.json", "json", "four answers, CF families, labels, deviations", "report"),
    "manifest": _f("main", "r1b_ana", "r1b_manifest.json", "json", "merged session manifests", "report"),
    "p0_record": _f("main", "r1b_P0", "p0_engineering.json", "json", "whitelisted engineering numbers of every pilot step", "freeze record"),
    "pilot_material": _f("main", None, PKG + "pilot_material.json", "json", "SO-PILOT prompts, passages, books", "P0", "SO-PILOT"),
    "keys_registry": _f("main", None, PKG + "r1b_keys.json", "json", "SO-KEYS registry (r1b_core.write_keys)", "all", "SO-KEYS"),
}

# ------------------------------------------------------------------------------------------------------------------------------------
# Start conditions (SG4). Every condition is also met when its producer is declared 'not run' after two attempts (main §4.7); the
# consumer then sees the producer in ctx.declared and its affected measurements read 'not run (input missing)'.
# ------------------------------------------------------------------------------------------------------------------------------------
def done(session, files=("**",)):
    """Producer finished: <session>/session_done.json without session_partial.json; copies files (globs) into <work>/in/<session>/.
    A name without glob characters must exist in the producer's folder."""
    return dict(kind="done", session=session, files=tuple(files))


def file(fid):
    """One cross-session file, ready as soon as its producing step recorded it (the producer may still run; same account only)."""
    return dict(kind="file", fid=fid, session=FILES[fid]["session"])


def xfer(fid):
    """An A -> B export (r1b_xfer_A/<name>, sha256 from xfer_manifest.json)."""
    return dict(kind="xfer", fid=fid, session=FILES[fid]["session"], name=FILES[fid]["xfer"])


def trig(fid, key, op, value=None):
    """Tier-2 trigger: <file>[key] <op> value with op in eq, ne, true; producer done and predicate false -> 'not triggered'."""
    return dict(kind="trig", fid=fid, session=FILES[fid]["session"], key=key, op=op, value=value)


def any_done(*sessions):
    """At least one of the sessions done; all resolved as 'not run' -> 'not triggered'."""
    return dict(kind="any", sessions=sessions)


# round-1 and round-2 Drive inputs (main §2.2), copied into <work>/in/r1/<shard>/... and <work>/in/r2/<session>/...
def R1(*blocks):
    pats = ["gpu*/shard_done.json", "gpu*/code_commits.txt"]
    for b in blocks:
        pats += [f"gpu*/rk/b{b}/state.pt", f"gpu*/rk/b{b}/loops.json", f"gpu*/summary_b{b}.json"]
    return tuple(pats)


R1_ALL = R1(14, 18, 6)
R1_S0 = R1_ALL + ("gpu*/rk/b*/pilot*.json", "gpu2/dn_delta.json")
R2_S0 = ("r2_q2/theta/b*_q0.pt", "r2_q2/theta/b*_q0.json", "r2_q2/theta/b*_r2RSIb.pt", "r2_q3/rgi/geom_stats_qwen3-0.6b.json",
         "r2_ana/unlearn_verdict.json", "r2_ana/r2_manifest.json", "r2_ana/theta_verdict.json")


def S(part, tier, account, wave, after=(), minutes=0.0, models=(), content="", runtime="a100", r1=(), r2=(), jlens=False, optional=()):
    return dict(part=part, tier=tier, account=account, wave=wave, after=tuple(after), minutes=minutes, models=tuple(models),
                content=content, runtime=runtime, r1=tuple(r1), r2=tuple(r2), jlens=jlens, optional=tuple(optional))


_R_TIER1 = ("r1b_R_S0", "r1b_R_S1", "r1b_R_S2_14", "r1b_R_S2_18", "r1b_R_S2_06", "r1b_R_S3", "r1b_R_S5", "r1b_R_S6", "r1b_R_S7")
SESSIONS = {
    "r1b_P0": S("main", 0, "B", 0, (), 68.2, (M17, M35), "P0-R, P0-D, P0-K (K0 items 1-9 incl. the Qwen3.5-2B load), P0-L; shared "
                "engineering items; SO-PILOT material only; engineering numbers only", jlens=True),
    "r1b_R_S0": S("R", 1, "A", 1, (), 22.2, (), "R0, R8.1, R9.5; SO-AX package (AX1, U*, leave-loop-out modes, SG10); r1b_xfer exports",
                  r1=R1_S0, r2=R2_S0),
    "r1b_R_S2_14": S("R", 1, "A", 1, (), 58.5, (M17,), "R2.1 (axis texts), R2.2 at 14; AX-1b(14)", r1=R1(14)),
    "r1b_R_S1": S("R", 1, "A", 1, (done("r1b_R_S0"),), 35.9, (M17,), "R1 closed and open loop", r1=R1_ALL),
    "r1b_R_S2_18": S("R", 1, "A", 2, (done("r1b_R_S2_14"),), 60.8, (M17,), "R2.1 reads, R2.2 at 18; AX-1b(18)", r1=R1(18)),
    "r1b_R_S2_06": S("R", 1, "A", 2, (done("r1b_R_S2_14"),), 60.8, (M17,), "R2.1 reads, R2.2 at 6; AX-1b(6)", r1=R1(6)),
    "r1b_R_S3": S("R", 1, "A", 3, tuple(done(s) for s in ("r1b_R_S2_14", "r1b_R_S2_18", "r1b_R_S2_06")), 100.2, (M17,),
                  "R3.3, TXT0 (R3.4), R4.1-R4.3", r1=R1_ALL),
    "r1b_R_S5": S("R", 1, "A", 4, (done("r1b_R_S3"), done("r1b_K_S1", ("k4j.json",))), 29.1, (M17,),
                  "R5.1, R5.2, R5.3, R5.5; AX-1b at the lens blocks; RG10, RG11", r1=R1_ALL, jlens=True),
    "r1b_R_S6": S("R", 1, "A", 4, (done("r1b_R_S3"),), 63.2, (M17,), "R6.1, R6.2, R6.3", r1=R1_ALL),
    "r1b_R_S7": S("R", 1, "A", 4, (done("r1b_R_S3"),), 21.8, (M17,), "R7.1-R7.3 (RG13, RG14 in session)", r1=R1_ALL),
    "r1b_R_ana": S("R", 1, "A", 5, tuple(done(s) for s in _R_TIER1), 55.3, (M17,),
                   "R pass 1: all R statistics and verdicts, F+ / F-, RG7, RG8a, RG8b, RG12, RG15 (J1, J2), R9 report check", r1=R1_ALL),
    "r1b_D_A": S("D", 1, "B", 1, (), 202.3, (M17, M06), "G-LCE-1 / G-LCE-2 first (SG9), stream prefills, D1 (main, neutral, "
                 "degradation, non-competing, D1-h), D1-c, D1-d, D1-e, D1-f, D1-r, D1-v, D1-s", optional=(file("sg8_preview"),)),
    "r1b_D_B": S("D", 1, "A", 2, (), 122.2, (M17,), "G-LCE-3, D2"),
    "r1b_D_C": S("D", 1, "A", 4, (), 149.6, (M17,), "D3 (with starts), D4 (with starts), D5 (D5-x only if tier 2 is funded)"),
    "r1b_D_ana": S("D", 1, "A", 5, (done("r1b_D_A"), done("r1b_D_B"), done("r1b_D_C"), done("r1b_R_S0", ("so_ax/*",)),
                                    done("r1b_R_S2_14", ("ax1b/*",))), 32.5, (),
                   "all D verdicts, G-D1-null, G-D2-null, D3 mixtures, D2-f, combined Holm (D.17)"),
    "r1b_K_S1": S("K", 1, "B", 1, (), 126.4, (M17,), "K2 (incl. the S_L screening, OD-M8), K3a-e, K3g(ii), K4j; SG8 preview", jlens=True),
    "r1b_K_S3": S("K", 1, "B", 1, (), 28.9, (M35,), "K5a"),
    "r1b_K_S2a": S("K", 1, "B", 2, (done("r1b_K_S1"), file("k5a_calib"), xfer("xfer_dmg_ref_R16")), 92.2, (M17,),
                   "K4 gating (incl. export to SO-LAB), K4 (150 wrong items), K4p, K6c, K6f, K6g, K6h, K10c; REF-U32 reporting of "
                   "r1b_K_S1's calibrations", jlens=True),
    "r1b_K_S2b": S("K", 1, "B", 3, (done("r1b_K_S2a"),), 51.1, (M17,), "K6e"),
    "r1b_K_S4": S("K", 1, "B", 3, (done("r1b_K_S1"), done("r1b_K_S3")), 119.9, (M17,), "K8a, K8b, K8c, K7a"),
    "r1b_K_S5": S("K", 1, "A", 4, (done("r1b_K_S1"), done("r1b_K_S2a"), done("r1b_K_S4"), done("r1b_R_S0", ("so_ax/*",)),
                                   done("r1b_R_S2_14", ("ax1b/*",)), done("r1b_R_S2_18", ("ax1b/*",))), 70.1, (M17,),
                  "K9, K10a, K10b, K3f, K7b (SO-AX read)"),
    "r1b_K_ana": S("K", 1, "A", 5, tuple(done(s) for s in ("r1b_K_S1", "r1b_K_S2a", "r1b_K_S2b", "r1b_K_S3", "r1b_K_S4", "r1b_K_S5")),
                   43.0, (M17,), "all K statistics, K6a / K6b / K6d, Holm families, the K.7 composite"),
    "r1b_L_A": S("L", 1, "B", 2, (done("r1b_K_S1"), xfer("xfer_dmg_ref_R16"), xfer("xfer_so_ax_ustar")), 168.9, (M17,),
                 "step 1 L.POP; L1, L3b, L4 (X2, X6 if tier 2 funded)"),
    "r1b_L_B": S("L", 1, "B", 3, (done("r1b_L_A"), file("glce"), file("k4_gating")), 117.2, (M17,),
                 "L.R corpus, L2, L3a, L3c, L3d; L5 generation (X1, X3, X4, X5 if tier 2 funded)"),
    "r1b_ana": S("main", 1, "A", 6, (done("r1b_R_ana"), done("r1b_D_ana"), done("r1b_K_ana"), done("r1b_L_A"), done("r1b_L_B"),
                                     done("r1b_K_S2a", ("lensout/*", "lab/*")), done("r1b_R_S5", ("jlens_validity.json",))),
                 0.0, (), "L-ANA; round-level families; cross-part labels; side-by-side tables; runde1b_summary.json; merged manifest",
                 runtime="cpu"),
    # tier 2 (main §4.2; funding in runde1b_freeze.json, SG7)
    "r1b_R_S5L": S("R", 2, "A", 5, (trig("rg11", "decision", "eq", "refit"),), 105.0, (M17,),
                   "R5.4 (bf16 refit), repeated R5.1, R5.2 / R5.3 at newly valid blocks", r1=R1_ALL, jlens=True),
    "r1b_R_S8": S("R", 2, "A", 6, (trig("rg15", "J1_and_J2", "true"),), 19.5, (M06,), "R8.2 (J3 in session)"),
    "r1b_R_ana2": S("R", 2, "A", 7, (any_done("r1b_R_S5L", "r1b_R_S8"),), 10.2, (M17,), "R pass 2"),
    "r1b_D_R": S("D", 2, "A", 6, (done("r1b_D_ana"),), 119.0, (M17,), "D's reserves R1, R2 (conditional on d_verdict triggers.R2), R3"),
    "r1b_K_S6": S("K", 2, "A", 6, (trig("k_verdict", "gates.G-K5", "ne", "calibrated"),), 58.0, (M17,), "RK-1 (MTP head, bf16)"),
    "r1b_K_S7": S("K", 2, "B", 4, (done("r1b_K_S3"),), 53.0, (M35,), "RK-2 (H-link on the native lens, Qwen3.5-2B)"),
}
TIER2_ITEMS = {"r1b_D_C": ("D5-x",), "r1b_L_A": ("X2", "X6"), "r1b_L_B": ("X1", "X3", "X4", "X5")}   # tier-2 items in tier-1 sessions

# Slots (main §4.4): one notebook each; sessions in order, each waiting for its start conditions (packing, not a rule).
SLOTS = {
    "P0": dict(account="B", runtime="a100", sessions=("r1b_P0",)),
    "A1": dict(account="A", runtime="a100", sessions=("r1b_R_S2_14", "r1b_R_S2_18", "r1b_R_S3", "r1b_R_S6", "r1b_K_S5", "r1b_D_C")),
    "A2": dict(account="A", runtime="a100", sessions=("r1b_R_S0", "r1b_R_S1", "r1b_R_S2_06", "r1b_D_B", "r1b_R_S5", "r1b_R_S7",
                                                      "r1b_R_ana", "r1b_K_ana", "r1b_D_ana")),
    "B1": dict(account="B", runtime="a100", sessions=("r1b_K_S1", "r1b_K_S4", "r1b_L_A", "r1b_L_B")),
    "B2": dict(account="B", runtime="a100", sessions=("r1b_K_S3", "r1b_D_A", "r1b_K_S2a", "r1b_K_S2b")),
    "A_T2": dict(account="A", runtime="a100", sessions=("r1b_R_S5L", "r1b_R_S8", "r1b_R_ana2", "r1b_K_S6", "r1b_D_R")),
    "B_T2": dict(account="B", runtime="a100", sessions=("r1b_K_S7",)),
    "ANA": dict(account="A", runtime="cpu", sessions=("r1b_ana",)),
}


def file_path(fid):
    """Path of a cross-session file relative to the run root (dsse_runs/), or the repository path of a package file."""
    f = FILES[fid]
    return f["rel"] if f["session"] is None else f"{f['session']}/{f['rel']}"


def produced_by(session):
    return {fid: f for fid, f in FILES.items() if f["session"] == session}


def upstream(session):
    """Sessions a session's start conditions name (direct)."""
    out = []
    for c in SESSIONS[session]["after"]:
        for s in ((c.get("session"),) if c["kind"] != "any" else c["sessions"]):
            if s and s not in out:
                out.append(s)
    return out


def check():
    """Consistency of the tables (run by the self-tests): every session of a slot exists once, in its account; conditions name known
    sessions and files; files name known sessions; no session waits for one later in its own slot; the slot schedule is deadlock-free."""
    errs, seen = [], {}
    for sl, d in SLOTS.items():
        for i, s in enumerate(d["sessions"]):
            if s not in SESSIONS:
                errs.append(f"slot {sl}: unknown session {s}"); continue
            if s in seen:
                errs.append(f"{s} in slots {seen[s]} and {sl}")
            seen[s] = sl
            if SESSIONS[s]["account"] != d["account"]:
                errs.append(f"{s}: account {SESSIONS[s]['account']} but slot {sl} runs in {d['account']}")
            for u in upstream(s):
                if u in d["sessions"][i + 1:]:
                    errs.append(f"slot {sl}: {s} waits for {u}, which comes later in the same slot")
    errs += [f"{s} is in no slot" for s in SESSIONS if s not in seen]
    for s, m in SESSIONS.items():
        for c in m["after"] + m["optional"]:
            for u in ((c.get("session"),) if c["kind"] != "any" else c["sessions"]):
                if u not in SESSIONS:
                    errs.append(f"{s}: condition names unknown session {u}")
            if c["kind"] in ("file", "xfer") and SESSIONS.get(c["session"], {}).get("account") != m["account"] and c["kind"] == "file":
                errs.append(f"{s}: file condition {c['fid']} crosses accounts (use done() with files, or xfer())")
            if c["kind"] == "xfer" and (m["account"] != "B" or SESSIONS[c["session"]]["account"] != "A"):
                errs.append(f"{s}: xfer condition only from A to B")
    for fid, f in FILES.items():
        if f["session"] is not None and f["session"] not in SESSIONS:
            errs.append(f"file {fid}: unknown session {f['session']}")
    # deadlock check: advance the slots in turn, every session finishing as soon as its producers are resolved
    pos, res = {sl: 0 for sl in SLOTS}, set()
    while True:
        moved = False
        for sl, d in SLOTS.items():
            while pos[sl] < len(d["sessions"]):
                s = d["sessions"][pos[sl]]
                if not all(u in res for u in upstream(s)):
                    break
                res.add(s); pos[sl] += 1; moved = True
        if not moved:
            break
    stuck = [d["sessions"][pos[sl]] for sl, d in SLOTS.items() if pos[sl] < len(d["sessions"])]
    if stuck:
        errs.append(f"slot schedule deadlocks at {stuck}")
    return errs


if __name__ == "__main__":
    e = check()
    print("contracts OK" if not e else "\n".join(e))
    raise SystemExit(1 if e else 0)
