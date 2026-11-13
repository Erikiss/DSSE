"""Part D self-tests (CPU; MOCK=1 tiny random model; never a real model).

    python r1b_d_selftest.py rules            synthetic tests of the analysis rules (no model): readout fit, D1-a on planted capture and
                                              null worlds, split rule, McNemar, lifetime, D.17 combination
    python r1b_d_selftest.py mock <root> <dir>   every part-D step in MOCK (r1b_D_A, r1b_D_B, r1b_D_C, r1b_D_ana, r1b_D_R) with SG6 checks
<root> needs models/qwen3-1.7b (config and tokenizer) and rsi_dl/persona (R1); package files are generated into <dir>/pkg."""
import glob, importlib, os, shutil, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import r1b_core as K
import r1b_d_stim as S
import r1b_d_ana as A


class FakeCtx:
    mock, freeze = False, {}

    def key(self, tag, *r):
        return K.keyseed(tag, *r)

    def trimmed(self, t):
        return False


def synth_d1(bonus, seed=0, n=48, prior_sd=1.5, cc=8.0, betas=(1.2, 0.9)):
    """Synthetic D1 records: l = c tanh((beta_L E_w + s b)/c) + noise (uniform kernel), beta 1.2 at L0 and 0.9 at L* (degradation), item
    priors b ~ N(0, 1.5), plus `bonus` nats towards B in the mixed contexts (k = 1..4) at L*."""
    import r1b_d_d1 as D1M
    rng = np.random.default_rng(seed); c = S.cfg(mock=False); ctx = FakeCtx(); recs = []; Ls = 32000
    b = rng.normal(0, prior_sd, (n, 6)); w = np.ones(10)
    def l_of(sp, L, extra=0.0):
        beta = betas[0] if L == 0 else betas[1]; s = 1 if sp["swap"] == 0 else -1
        E = A.evidence(sp["kinds"], sp["swap"], w) if sp["arr"] != "xn" else 0.0
        out = []
        for j, bf in sp["qs"]:
            v = cc * np.tanh((beta * E + s * b[sp["item"], j]) / cc) + extra + rng.normal(0, 0.3)
            out.append(float(v))
        return out
    specs = []
    for i in range(n):
        for sw in (0, 1):
            specs.append(D1M.pure("pure", "sym", i, c, sw))
            for k in range(1, 6):
                for o in (1, 2):
                    specs.append(D1M.mixed(ctx, "main", "sym", i, c, sw, k, o))
            for k in (2, 4):
                for o in (1, 2):
                    specs.append(D1M.mixed(ctx, "hedge", "sym", i, c, sw, k, o, sub="C"))
                    specs.append(D1M.mixed(ctx, "f", "sym", i, c, sw, k, o))
            specs.append(D1M.pure("f", "sym", i, c, sw))
            for o in (1, 2):
                specs.append(D1M.ycomp(ctx, "nc", i, c, sw, o, (0, 5, 5)))
        specs.append(D1M.xn("xn", i, c))
    for sp in specs:
        for L in (0, Ls):
            if sp["arm"] == "f" and L == 0:
                continue
            mixed_ = sp["arm"] in ("main", "f") and 1 <= sp["k"] <= 4 and L == Ls
            r = dict(sp, L=L, model="qwen3-1.7b", filler="shuf" if sp["arm"] == "f" else "coh", M=[0.99] * len(sp["qs"]))
            r["l"] = l_of(sp, L, bonus if mixed_ else 0.0)
            if sp["arm"] == "f":
                r["arr"] = "pure" if sp["k"] == 0 else "int"
            recs.append(r)
    return recs, c, Ls


def rules():
    t0 = time.time(); errs = []
    ok = lambda cond, msg: (print(("ok   " if cond else "FAIL ") + msg), None if cond else errs.append(msg))
    # readout fit recovers c, beta, b
    rng = np.random.default_rng(3); E = np.array([0, -5, -5, -10, -10.0]); sg = np.array([1, 1, -1, 1, -1.0])
    bt, bb = rng.uniform(0.4, 1.0, 200), rng.normal(0, 2, 200); y = 7 * np.tanh((bt[:, None] * E + sg * bb[:, None]) / 7) + rng.normal(0, 0.05, (200, 5))
    f = A.fit_readout(y, E, sg); ok(4.5 < f["c"] < 10 and np.corrcoef(f["b"], bb)[0, 1] > 0.95, f"readout fit (c {f['c']:.2f})")
    # D1-a on planted worlds
    for bonus, want in ((1.5, "capture"), (0.0, "null")):
        recs, c, Ls = synth_d1(bonus); D = A.D1(recs); items = list(range(48)); W = A.weights(A.item_groups(items, 16), 11)
        r = A.d1a(FakeCtx(), D, items, W, Ls, "crop", True, lambda L: False, c)
        v = r["verdict"]; print(f"     D1-a ({want}): {v}; Phi* {r['Phi_Ls']['pt']:.3f} [{r['Phi_Ls']['lo']:.3f}], dPhi {r['dPhi']['pt']:.3f}, "
                                 f"bound {r['bound']:.3f}, G-D1-null null rate {r['G-D1-null']['validation']['null_rate']:.2f}")
        if want == "capture":
            ok(r["Phi_Ls"]["lo"] > r["bound"] and r["dPhi"]["lo"] > 0, "D1-a: planted 1.5-nat capture meets verdict 4's condition")
        else:
            ok(not str(v).startswith("length-driven"), "D1-a: null world is not called capture")
    recs, c, Ls = synth_d1(0.0, prior_sd=4.0, cc=5.0, betas=(1.2, 0.5)); D = A.D1(recs); items = list(range(48))
    W = A.weights(A.item_groups(items, 16), 11); r = A.d1a(FakeCtx(), D, items, W, Ls, "crop", True, lambda L: False, c)
    print(f"     D1-a (prior x saturation null, |b| SD 4, c 5, beta 1.2 -> 0.5): {r['verdict']}")
    ok(not str(r["verdict"]).startswith("length-driven"), "D1-a: prior x saturation null is not called capture")
    # split rule
    rng = np.random.default_rng(5); lab = np.r_[np.zeros(8), np.ones(8)]
    x = rng.normal(0, 1, 16) + 8 * lab; y_ = rng.normal(0, 1, 16) + 8 * lab
    sp, _ = A.split_rule(x, y_, np.stack([x] * 3, 1), np.stack([y_] * 3, 1)); ok(sp, "split rule: 8-SD two-mode item splits")
    x = rng.normal(0, 1, 16); y_ = rng.normal(0, 1, 16)
    sp, _ = A.split_rule(x, y_, np.stack([x] * 3, 1), np.stack([y_] * 3, 1)); ok(not sp, "split rule: unimodal item does not split")
    ok(abs(A.mcnemar_p(5, 0) - 1 / 32) < 1e-9, "McNemar exact one-sided p (5, 0) = 1/32")
    import r1b_d_d5 as D5
    ok(D5.lifetime(([1, 2, 3, 4] * 300)[:1024], 4096) == (256, False), "lifetime: a repeated text degenerates in the first window")
    ok(D5.lifetime(list(range(1024)), 1024) == (1024, True), "lifetime: a healthy text is censored")
    print(f"rules: {len(errs)} failures ({time.time() - t0:.0f}s)")
    return errs


STEPS_ORDER = {"r1b_D_A": None, "r1b_D_B": None, "r1b_D_C": None, "r1b_D_ana": None, "r1b_D_R": None}


def mock(root, d):
    """Every part-D session's steps in MOCK through the step functions (no runner): inputs copied as the runner would (done(...)),
    SO-AX placeholders for r1b_D_ana, SG6 on every output."""
    os.environ["MOCK"] = "1"; os.environ.setdefault("DEVICE", "cpu"); os.environ["R1B_PKG_DIR"] = os.path.join(d, "pkg")
    import r1b_sessions_D as R
    from transformers import AutoTokenizer
    if not os.path.exists(os.path.join(d, "pkg", "d1_stimuli.json")):
        print("package files:", S.build_all(AutoTokenizer.from_pretrained(os.path.join(root, "models", "qwen3-1.7b")), os.path.join(d, "pkg")))
    fz = dict(funded=dict(tier2=True, items=[]), trims={}); errs = []
    for sess in ("r1b_D_A", "r1b_D_B", "r1b_D_C", "r1b_D_ana", "r1b_D_R"):
        work = os.path.join(d, sess); inp = os.path.join(work, "in")
        for up in {"r1b_D_ana": ("r1b_D_A", "r1b_D_B", "r1b_D_C"), "r1b_D_R": ("r1b_D_A", "r1b_D_C", "r1b_D_ana")}.get(sess, ()):
            dst = os.path.join(inp, up)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(os.path.join(d, up), dst, ignore=shutil.ignore_patterns("in", "cache", "steps", "logs"))
        if sess == "r1b_D_ana":
            for s_, rel in (("r1b_R_S0", "so_ax/so_ax_b14.pt"), ("r1b_R_S2_14", "ax1b/ax1b_b14.pt")):
                p = os.path.join(inp, s_, rel); os.makedirs(os.path.dirname(p), exist_ok=True); K._torch().save(dict(mock=True), p)
        ctx = K.Ctx(sess, work, root=root, device="cpu", mock=True, freeze=fz)
        for st in R.STEPS[sess]:
            mod, fn = st.fn.split(":"); ctx.step, ctx.step_started, ctx.keys_used, ctx.models_used, ctx.step_part = st.name, K.iso(), set(), {}, "D"
            t = time.time()
            try:
                getattr(importlib.import_module(mod), fn)(ctx); state = "ok"
                miss = [o for o in st.out if not glob.glob(os.path.join(work, o))]
                if miss:
                    state = f"outputs missing {miss}"; errs.append(f"{sess}/{st.name}: {state}")
            except K.NotRun as e:
                state = f"not run ({e.reason})"
            except Exception as e:
                import traceback; traceback.print_exc(); state = f"FAILED {e!r}"; errs.append(f"{sess}/{st.name}: {e!r}")
            print(f"[{sess}/{st.name}] {state} ({time.time() - t:.0f}s)", flush=True)
        bad = [x for f in glob.glob(work + "/**/*", recursive=True) if os.path.isfile(f) and "/cache/" not in f and "/in/" not in f
               for x in [K.check_output(f)] if x]
        if bad:
            errs.append(f"{sess}: SG6 {bad[:3]}")
    v = K.load_json(os.path.join(d, "r1b_D_ana", "d_verdict.json"), {})
    print("D.17 (MOCK):", v.get("D.17")); print(f"mock: {len(errs)} problems", errs)
    return errs


if __name__ == "__main__":
    if sys.argv[1] == "rules":
        sys.exit(1 if rules() else 0)
    elif sys.argv[1] == "mock":
        sys.exit(1 if mock(sys.argv[2], sys.argv[3]) else 0)
    print(__doc__)
