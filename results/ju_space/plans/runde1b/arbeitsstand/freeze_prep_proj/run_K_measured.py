"""Part K cost (K.8, cost_merge3.py) re-run at r1b_P0's measured rates: the add() lines of cost_merge3.py are executed unchanged, with
dec() / fwd() / the FISTA rate replaced (MODE=meas) or as planned (MODE=plan). Measured (P0-K, P0-L, P0-D):
  rollout / decode step (Qwen3-1.7B, short context, B >= 160 rows, sdpa_efficient, per-step syncs, FOV hooks) = 17.5 ms + 0.28 ms x rows
    (two points, 160 and 320 rows x 40 positions, 75 / 133 ms per step incl. the frame prefill; frame prefill 19 tokens x 0.17 ms
    subtracted); 0.24 ms x rows without capture hooks (assumption: FOV hooks ~15 %); +0.022 ms x rows for per-step DLA (hook 1.5)
  frame prefill of lens rollouts (19 tokens per row at 0.17 ms / token + 38 ms per batch call), absent from cost_merge3.py (ROLL=1)
  long caches (ctx > 0): P0-D step model max(38 ms, 13.4 ms + L (3.45 us x waves + 0.486 us x B)) + 0.218 ms x rows
  teacher-forced passes x 1.10 (L0_a (ii): 4,545 tok/s with hooks against the planned rate)
  FISTA FISTA_S s per iteration (default 0.005: the 35 ms reported includes the 30-iteration power method _lip)
  Qwen3.5-2B (F35, FM): unchanged (not measured; the plan's x2 risk factor stays)."""
import math, os, re
MODE = os.environ.get("MODE", "meas"); ROLL = os.environ.get("ROLL", "1") == "1"
FISTA_S = float(os.environ.get("FISTA_S", "0.004" if os.environ.get("MODE", "meas") == "plan" else "0.005")); TF = float(os.environ.get("TF", "1.10"))
HK, CR, CR0, CDLA = 0.0175, 0.28e-3, 0.24e-3, 0.022e-3
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cost_K_measured.py")).read()
head, body = src.split("S = {}  # session -> list of (id, seconds, note)", 1)
body = "S = {}\n" + body
exec(head.split('"""', 2)[2] if head.startswith('"""') else head, globals())
_plan_dec, _plan_fwd = dec, fwd


def waves(B):
    return math.ceil(16 * B / 108)


def dec(rows, steps, batch=320, f=FL, ctx=0, host=HOST, hook=1.0, roll=False):
    if MODE == "plan" or f in (F35, FM):
        return _plan_dec(rows, steps, batch=batch, f=f, ctx=ctx, host=host, hook=hook)
    full, rem = divmod(rows, batch)

    def step(r):
        if ctx:
            L = ctx + steps
            return (max(0.038, 0.0134 + L * (3.45e-6 * waves(r) + 0.486e-6 * r)) + r * 0.218e-3) * (1.0 if hook <= 1.0 else 1.1)
        c = CR0 if hook <= 1.0 else (CR + (CDLA if hook >= 1.5 else 0.0))
        return HK + r * c
    t = full * steps * step(batch) + (steps * step(rem) if rem else 0.0)
    if roll and ROLL:
        t += rows * 19 * 0.17e-3 + 0.038 * (full + (1 if rem else 0))
    return t


def fwd(tokens, f=F, hook=1.0):
    return _plan_fwd(tokens, f, hook) * (1.0 if (MODE == "plan" or f == F35) else TF)


# lens rollouts (frame + seed prefilled per call in r1b_k_lens.rollout): mark them
roll_ids = ("'K3e'", "'K4', dec(c_main", "'K4', dec(768", "'K4', dec(512", "'K4', dec(256", "'K4', dec(5 *", "'K4', dec(2 * 2",
            "'K6c'", "'K6g'", "'K6h'", "'K7b'")
lines = []
for ln in body.split("\n"):
    if any(k in ln for k in roll_ids) and "dec(" in ln:
        ln = re.sub(r"dec\(([^()]*(\([^()]*\))*[^()]*)\)", lambda m: "dec(" + m.group(1) + ", roll=True)", ln)
    if ln.startswith("one = dec(rows, 24"):
        ln = ln.replace("dec(rows, 24, hook=1.25)", "dec(rows, 24, hook=1.25, roll=True)")
    if "K8c" in ln and "dec(1800" in ln:
        ln = ln.replace("dec(1800 * 2 * 10 * 2, 9, hook=1.25)", "dec(1800 * 2 * 10 * 2, 9, hook=1.25, roll=True)")
    lines.append(ln)
body = "\n".join(lines).replace("* 0.004, 'L1 paths", "* FISTA_S, 'L1 paths")
_sess_src = None
if MODE == "meas":
    body = body.replace("sess = lambda n, turns=8, ans=30, q=40, ctx=300: (-(-n // 128)) * turns * (ans * (HOST + 128 * FL / R + 3 * 128 * ctx * KVB / BW) + 128 * q * FL / R)",
                        "sess = lambda n, turns=8, ans=30, q=40, ctx=300: (-(-n // 128)) * turns * (ans * (max(0.038, 0.0134 + ctx * (3.45e-6 * waves(128) + 0.486e-6 * 128)) + 128 * 0.218e-3) + TF * 128 * q * FL / R)")
    assert "waves(128)" in body
exec(body.split("setup = {")[0], globals())
setup = {'K-S0': 4, 'K-S1': 4, 'K-S3': 8, 'K-S2a': 4, 'K-S2b': 4, 'K-S4': 4, 'K-S5': 6, 'K-ana': 3}
res = {}
for s, items in S.items():
    w = sum(x[1] for x in items) / 60
    res[s] = dict(work=round(w, 1), planned_min=round(setup[s] + 1.5 * w, 1),
                  items={mid: round(sum(x[1] for x in items if x[0] == mid) / 60, 1) for mid in dict.fromkeys(x[0] for x in items)})
import json
print(json.dumps(dict(mode=MODE, roll=ROLL, fista_s=FISTA_S, sessions=res,
                      total_h=round(sum(v["planned_min"] for v in res.values()) / 60, 2)), indent=1))
