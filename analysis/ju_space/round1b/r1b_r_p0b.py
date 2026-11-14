"""Part R items of the re-pilot r1b_P0b: R-P's list (RG18's input rates for cost_R2.py; account B, before the freeze; SO-PILOT prompts
only, keys 'r1bP0_R'; every generated text is discarded; timings, memory and code checks only). Each item runs under
r1b_pilot.Rec.item (an out-of-memory or other error is a recorded value). Rates use cost_R2.py's names:

  LOAD_min                 model load (first use in the session process)
  GEN, GEN80/200/360       s per K.generate call, 32 rows x 40 new tokens (16 +/- pairs) with R's push at block 14 and the
                           generation-mode cbar recorder, prompts of 20 / 80 / 200 / 360 tokens (GEN*_plain: no hooks)
  RD14, RDX, RALL, RFULL,  s per 32-text batch (20-token prompt + its 40-token continuation): read stopped at 14 (R.read_win), at 18,
  RSHORT                   all 28 blocks captured, full pass with logits; 20-token sentences read at 14
  FWD, FB, FB8             one short forward (30 tokens); forward + backward of one text with a leaf at block 14 (R4.2 / R4.1 style),
                           per text at batch 1 and at batch 8
  SOFT                     s per soft-token generation call (r1b_r_s6.soft_gen), 32 rows x 40 steps
  DMG                      one dmg_sym evaluation: 16 pilot references x 2 signs, pushed (R.Dmg's K.Damage form)
replay: the round-1 replay opener on MOCK (rk_loop.py byte-identical, MOCK=1 on the CPU, stage 'pilot' at block 14) as a code check.
Drive read and write throughput: r1b_pilot.p0b_drive (the shared Drive item)."""
import os, shutil, subprocess, sys, time
import numpy as np
import r1b_core as K
import r1b_r_lib as R
import r1b_pilot as P


def _exact(tok, texts, L, i):
    """Chat ids of exactly L tokens: the pilot prompts from index i on, joined, cut inside the user content."""
    pre, post = K.chat_text(tok, "\x00").split("\x00"); a, b = K.enc(tok, pre), K.enc(tok, post); body, j = [], i
    while len(body) < L - len(a) - len(b):
        body += K.enc(tok, " " + texts[j % len(texts)]); j += 1
    return a + body[:max(L - len(a) - len(b), 1)] + b


def rates(ctx):
    torch = R.torch(); rec = P.Rec(); mock = ctx.mock; st = {}

    def load():
        t = time.time(); st["m"] = R.M(ctx); rec["rates"]["LOAD_min"] = (time.time() - t) / 60
    rec.item("load", load)
    if "m" not in st:
        return ctx.write_json("p0b/r_rates.json", rec)
    m = st["m"]; dev = K.model_device(m.model); texts = P.prompts(ctx); nnew = m.nnew; B = 32 if not mock else 4
    g = torch.Generator().manual_seed(20261118); d = torch.randn(m.D, generator=g); d = d / d.norm(); a = 0.1
    keys = lambda tag: [ctx.key("r1bP0_R", tag, j) for j in range(B // 2)]
    for L in (20, 80, 200, 360):
        def gen(L=L):
            P_ = [_exact(m.tok, texts, L if not mock else min(L, 40), 2 * j) for j in range(B // 2)]
            rows = [(p, k, s) for p, k in zip(P_, keys(f"gen{L}")) for s in (1, -1)]
            push = R.pusher(m, 14 if m.L > 14 else 1, d, a, None, "add", {})
            nm = "GEN" if L == 20 else f"GEN{L}"
            rec["rates"][nm] = P.timed(lambda: K.generate(m.model, m.tok, rows, T=0.7, max_new=nnew, setup=push), reps=2, warm=1)
            rec["rates"][nm + "_plain"] = P.timed(lambda: K.generate(m.model, m.tok, rows, T=0.7, max_new=nnew), reps=1, warm=0)
            if L == 20:
                st["prompts"], st["conts"] = [r[0] for r in rows], K.generate(m.model, m.tok, rows, T=0.7, max_new=nnew, setup=push)
        rec.item(f"gen_{L}", gen)
    if "conts" in st:
        pr, co = st["prompts"], [c if c else [m.PAD] for c in st["conts"]]
        b2 = 18 if m.L > 18 else 2

        def reads():
            rec["rates"]["RD14"] = P.timed(lambda: R.read_win(m, pr, co, 14 if m.L > 14 else 1), reps=3, warm=1)
            rec["rates"]["RDX"] = P.timed(lambda: R.read_win(m, pr, co, b2), reps=3, warm=1)
            ids, am, _ = K.pad_left([list(p) + list(c) for p, c in zip(pr, co)], m.PAD); ids, am = ids.to(dev), am.to(dev)

            def rall():
                with K.Hooks(m.model) as hk, torch.no_grad():
                    hk.capture(list(range(m.L))); hk.run(lambda: m.model.model(input_ids=ids, attention_mask=am))
            rec["rates"]["RALL"] = P.timed(rall, reps=3, warm=1)
            with torch.no_grad():
                rec["rates"]["RFULL"] = P.timed(lambda: m.model(input_ids=ids, attention_mask=am), reps=3, warm=1)
            short = [list(c)[:20] or [m.PAD] for c in co]
            rec["rates"]["RSHORT"] = P.timed(lambda: R.read_win(m, [[] for _ in short], short, 14 if m.L > 14 else 1), reps=3, warm=1)
        rec.item("reads", reads)

        def fwd_fb():
            one = torch.tensor([pr[0][:30]], device=dev)
            with torch.no_grad():
                rec["rates"]["FWD"] = P.timed(lambda: m.model(input_ids=one), reps=10, warm=2)
            blk = K.layers_of(m.model)[14 if m.L > 14 else 1]

            def fb(n):
                rows = [list(p) + list(c) for p, c in zip(pr[:n], co[:n])]; ids, am, _ = K.pad_left(rows, m.PAD)
                V = torch.zeros(n, m.D, device=dev, requires_grad=True)
                h = blk.register_forward_hook(lambda mod, i_, o_: ((o_[0] + V[:, None]),) + tuple(o_[1:]) if isinstance(o_, tuple) else o_ + V[:, None])
                try:
                    with torch.enable_grad():
                        last = m.model.model(input_ids=ids.to(dev), attention_mask=am.to(dev)).last_hidden_state
                        lp = torch.log_softmax(m.model.lm_head(last[:, -nnew - 1:-1]).float(), -1)
                        g_, = torch.autograd.grad(lp[..., :8].logsumexp(-1).sum(), V)
                finally:
                    h.remove()
            rec["rates"]["FB"] = P.timed(lambda: fb(1), reps=5, warm=1)
            rec["rates"]["FB8"] = P.timed(lambda: fb(8 if not mock else 2), reps=3, warm=1) / (8 if not mock else 2)
        rec.item("fwd_fb", fwd_fb)

        def soft():
            import r1b_r_s6 as S6
            pids = [(p, s) for p in pr[0::2] for s in (1, -1)][:B]
            rec["rates"]["SOFT"] = P.timed(lambda: S6.soft_gen(m, 14 if m.L > 14 else 1, pids, d, a, None, nnew), reps=1, warm=1)
        rec.item("soft", soft)

        def dmg():
            n = 16 if not mock else 4
            dm = K.Damage(m.model, K.RefSet.from_prompt_conts(pr[:n], co[:n], "pilot"))
            su = lambda s: K.Hooks(m.model).push(14 if m.L > 14 else 1, d, a, signs=s, cap=None, rowstart=torch.tensor([m.START]))
            rec["rates"]["DMG"] = P.timed(lambda: (dm(lambda: su(1.0)), dm(lambda: su(-1.0))), reps=2, warm=1)
        rec.item("dmg", dmg)
    st.clear()
    rec["env"].update(rows=B, new_tokens=nnew, push_block=14, push_a=a, note="Drive throughput: p0b/drive.json")
    ctx.write_json("p0b/r_rates.json", rec, config=dict(rows=B, new=nnew, lengths=[20, 80, 200, 360]))


def replay(ctx):
    """The round-1 replay opener on MOCK: rk_loop.py (byte-identical, sha256 recorded) stage 'pilot' at block 14 with MOCK=1 on the CPU
    in a subprocess (its tiny random model); exit code, seconds and the state file; the output folder is removed afterwards."""
    rec = P.Rec()

    def run():
        import r1b_r_s1 as S1
        out = ctx.path("p0b/_rk_mock"); shutil.rmtree(out, ignore_errors=True)
        env = dict(S1._env(ctx, 14, "1"), MOCK="1", DEVICE="cpu", NT=os.environ.get("NT", "2"), PYTHONUNBUFFERED="1")
        t = time.time()
        r = subprocess.run([sys.executable, R.RK_SRC, ctx.root, out, "pilot"], env=env, capture_output=True, text=True, timeout=1200)
        rec["timing"]["rk_mock_pilot_s"] = time.time() - t
        ok = r.returncode == 0 and os.path.isdir(os.path.join(out, "b14"))
        rec["code_checks"].update(rk_mock_pilot_exit=r.returncode, rk_mock_pilot_ok=bool(ok), rk_loop_sha256=K.sha256_file(R.RK_SRC))
        if r.returncode:
            rec["notes"].append("rk_loop MOCK pilot: " + (r.stderr or r.stdout)[-400:])
        shutil.rmtree(out, ignore_errors=True)
    rec.item("rk_mock_pilot", run)
    ctx.write_json("p0b/r_replay.json", rec, config=dict(block=14, stage="pilot", mock_model=True))
