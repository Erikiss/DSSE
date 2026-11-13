"""K3h (tier-2 item of r1b_K_S2b; descriptive): atlas-compatible H-neuron metrics, measured on Qwen3-1.7B with our own definitions
(method reference: neuron-atlas style descriptions; no atlas data used). Per neuron of H_op and H_ar (and one matched random neuron of
the same layer per H neuron, REF-W mean |z| within +-20 %, seed 1501 + k), per pole (positive and negative z read separately):
(1) activation statistics (firing threshold = the pole's 95th percentile on the corpus, firing rate, strength in threshold multiples);
(2) damage per firing = KL(unmodified || neuron zeroed at the firing token) summed over the next 16 predictions, <= 100 firings per
pole stratified by strength band and the firing token's rarity tertile (unit: firings clustered by text); (3) modes = k-means of the
firings' MLP-input states (k by silhouette, 1 if < 0.1), damage share of the main mode as a ratio of sums with a text-cluster bootstrap:
'single mode' / 'one dominant mode' (90 % lower bound >= 0.9) / 'several modes' (upper bound < 0.9) / 'mode structure undecided';
(4) effect-based simulatability (SO-GVRA, inverse-free): a description (token set with strengths) simulates the activation on held-out
text; the simulated activation is patched in; recovered share R = 1 - sum KL(orig || simulated) / sum KL(orig || zeroed) (ratio of
sums; negative = worse than switching the neuron off); 'correct' R >= 0.5, 'minor fix needed' 0.2 <= R < 0.5 (revision: the token
set doubled with previous-token context), 'critically flawed' (next family: K3e's chain-lens readout tokens); 3 attempts, then
abstain ('uninterpretable'); verifier error on token-detector neurons (sensitivity) and on foreign descriptions (false accept)."""
import numpy as np
import r1b_core as RC
import r1b_k_base as KB
import r1b_k_gvra as GV
from r1b_k_base import N, H, T_, write, save, load_pt, own, Interv, coded, val
import r1b_k_s1 as S1


def _neurons(ctx, R):
    hs = S1.hnset(ctx); torch = T_(); out = []
    Hs = list(dict.fromkeys(tuple(x) for x in hs["H_op"] + hs["H_ar"]))[:N(ctx, 60, 3)]
    zab = np.abs(R["z_mu"]); npl = zab.shape[1]
    for k, (l, j) in enumerate(Hs):
        out.append((l, j, "H"))
        g = np.random.default_rng(1501 + k); cand = [c for c in g.permutation(npl) if c != j and abs(zab[l, c] - zab[l, j]) <= 0.2 * zab[l, j] + 1e-9]
        if cand:
            out.append((l, int(cand[0]), "R"))
    return out


def _texts(ctx):
    b = KB.bank(ctx); t = [list(x) for x in b["wt"]["ntext"][:N(ctx, 200, 6)]]; k = max(1, len(t) * 3 // 4)
    return t[:k], t[k:]


def _acts(model, texts, l, j, dev):
    torch = T_(); out = []
    for t in texts:
        with KB.Tap(model, lambda li, z, den: out.append(z[0, :, j].float().cpu()) if li == l else None, layers=[l]), torch.no_grad():
            model.model(input_ids=torch.tensor([t], device=dev))
    return out


def _kl(a, b):
    return (a.exp() * (a - b)).sum(-1)


def k3h_part(ctx, i, n):
    """Firings, damage per firing and simulatability for a share of the neurons."""
    torch = T_()
    tok, model = ctx.model(); R = S1.refs(ctx); fit_t, held_t = _texts(ctx); dev = ctx.device; b = KB.bank(ctx)
    uf = np.load(b["_files"]["unigram_logfreq.npy"]); base = {}
    with torch.no_grad():
        for k, t in enumerate(fit_t + held_t):
            base[k] = model(input_ids=torch.tensor([t], device=dev)).logits[0].float().log_softmax(-1).cpu()
    k3e = KB.up(ctx, "r1b_K_S1", "k/k3/k3e.json", required=False); k3e = RC.load_json(k3e) if k3e else {}
    rows = []
    for (l, j, kind) in KB.parts(_neurons(ctx, R), i, n):
        A = _acts(model, fit_t, l, j, dev); Ah = _acts(model, held_t, l, j, dev)
        for pole in (1, -1):
            v = torch.cat([a[16:] * pole for a in A]); pos_ = v[v > 0]
            if pos_.numel() < 5:
                rows.append(dict(l=l, j=j, kind=kind, pole=pole, note="no firings")); continue
            thr = float(torch.quantile(pos_, 0.95)); fir = [(k, t) for k, a in enumerate(A) for t in range(16, len(a) - 16) if pole * a[t] >= thr]
            g = np.random.default_rng(H((l, j, pole))); g.shuffle(fir); fir = fir[:N(ctx, 100, 8)]
            dmg, ctxv, strength, rare = [], [], [], []
            for k, t in fir:
                ids = torch.tensor([fit_t[k][:t + 17]], device=dev); m = torch.zeros(1, ids.shape[1], dtype=torch.bool); m[0, t] = True
                with Interv(model, [dict(sel={l: [j]}, kind="scale", value=0.0, when=lambda T, st, m=m: m[:, :T])]), torch.no_grad():
                    lp = model(input_ids=ids).logits[0].float().log_softmax(-1).cpu()
                dmg.append(float(_kl(base[k][t:t + 16], lp[t:t + 16]).sum()))
                with RC.Hooks(model) as hk, torch.no_grad():
                    hk.capture([max(l - 1, 0)]).stop_after(max(l - 1, 0)); hk.run(lambda: model.model(input_ids=ids))
                    ctxv.append(hk.states[max(l - 1, 0)][0][0, t].numpy())
                strength.append(float(pole * A[k][t]) / thr); rare.append(float(uf[fit_t[k][t]]) if fit_t[k][t] < len(uf) else 0.0)
            # simulatability: generator = token-set descriptions; verifier = forward patch on held-out text
            cnt = {}
            for k, t in fir:
                cnt.setdefault(fit_t[k][t], []).append(float(A[k][t]))
            desc = [dict(tokens={tt: float(np.mean(v_)) for tt, v_ in sorted(cnt.items(), key=lambda x: -len(x[1]))[:10]}, ctx=False)]
            ro = (k3e.get("readouts") or {}).get(f"{l}:{j}:{'+' if pole > 0 else '-'}", {}).get("readouts")

            def recovered(d):
                num = den = 0.0
                for k2, t in enumerate(held_t):
                    ids = torch.tensor([t], device=dev); bl = base[len(fit_t) + k2]
                    sim = torch.tensor([d["tokens"].get(x, 0.0) for x in t])
                    with Interv(model, [dict(sel={l: [j]}, kind="clamp", value={l: sim.to(dev)[None, :, None].squeeze(-1)[0]} if False else 0.0,
                                             when=lambda T, st: None)]), torch.no_grad():
                        pass
                    zp = sim.to(dev)
                    with _ClampSeq(model, l, j, zp), torch.no_grad():
                        lp_s = model(input_ids=ids).logits[0].float().log_softmax(-1).cpu()
                    with Interv(model, [dict(sel={l: [j]}, kind="scale", value=0.0)]), torch.no_grad():
                        lp_z = model(input_ids=ids).logits[0].float().log_softmax(-1).cpu()
                    num += float(_kl(bl, lp_s).sum()); den += float(_kl(bl, lp_z).sum())
                return 1 - num / den if den > 0 else float("nan")

            def gen(u, a):
                if a == 0:
                    return desc[0]
                if a >= 1 and ro:
                    toks = {tt: float(np.mean([x for v_ in cnt.values() for x in v_])) for r_ in ro for tt in RC.enc(tok, r_)}
                    return dict(tokens=toks, ctx=False, family="chain-lens readout")
                return None

            def ver(u, d):
                r_ = recovered(d)
                return ("correct" if r_ >= 0.5 else "minor fix needed" if r_ >= 0.2 else "critically flawed"), r_

            def rev(u, d, a):
                more = {tt: float(np.mean(v_)) for tt, v_ in sorted(cnt.items(), key=lambda x: -len(x[1]))[:20]}
                return dict(tokens=more, ctx=True)
            gv = GV.gvra([0], gen, ver, rev, limit=3)[0]
            rows.append(dict(l=l, j=j, kind=kind, pole=pole, thr=thr, rate=float((v >= thr).float().mean()), n_fire=len(fir), damage=dmg,
                             text=[k for k, _ in fir], strength=strength, rare=rare, ctx=np.array(ctxv), simul=dict(status=gv["status"], R=gv["score"],
                                                                                                                  attempts=gv["attempts"], trace=gv["trace"])))
    save(ctx, f"k/k3h/part{i}.pt", dict(rows=rows))


class _ClampSeq:
    """z_j <- the simulated activation at every position of a single-text forward."""

    def __init__(self, model, l, j, zs):
        def pre(m, a):
            x = a[0].clone(); T = x.shape[1]; x[0, :, j] = zs[:T].to(x.dtype)
            return (x,) + tuple(a[1:])
        self.h = RC.layers_of(model)[l].mlp.down_proj.register_forward_pre_hook(pre)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.h.remove()


def _modes(X, d, texts, B=RC.N_BOOT):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    X = np.asarray(X); d = np.asarray(d); best, lab = (0.1, 1), np.zeros(len(d), int)
    if len(d) >= 8:
        Z = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        for k in (2, 3, 4):
            km = KMeans(k, n_init=4, random_state=0).fit(Z); s = silhouette_score(Z, km.labels_)
            if s > best[0]:
                best, lab = (s, k), km.labels_
    main = np.bincount(lab, weights=d).argmax() if d.sum() > 0 else 0
    share = lambda ix: d[ix][lab[ix] == main].sum() / max(d[ix].sum(), 1e-12)
    c = KB.ci(KB.boot_stat(len(d), share, B=B, groups=np.asarray(texts))) if len(d) > 2 else dict(lo=coded("NaN:few"), hi=coded("NaN:few"))
    cat = ("single mode" if best[1] == 1 else "one dominant mode" if (KB.fin(c["lo"]) and c["lo"] >= 0.9) else
           "several modes" if (KB.fin(c["hi"]) and c["hi"] < 0.9) else "mode structure undecided")
    return dict(k=best[1], silhouette=val(best[0]), main_share=val(share(np.arange(len(d)))), ci=c, category=cat)


def k3h_ana(ctx, n):
    rows = KB.merge_parts(ctx, "k3h", n); out = []
    for r in rows:
        if "damage" not in r:
            out.append(dict(neuron=f"{r['l']}:{r['j']}", kind=r["kind"], pole=r["pole"], note=r.get("note"))); continue
        m = _modes(r["ctx"], r["damage"], r["text"], B=200 if ctx.mock else RC.N_BOOT)
        if r["simul"]["status"] == "abstained":
            m["category"] = "uninterpretable"
        out.append(dict(neuron=f"{r['l']}:{r['j']}", kind=r["kind"], pole="+" if r["pole"] > 0 else "-", threshold=r["thr"], firing_rate=r["rate"],
                        damage_per_firing=KB.dist(r["damage"]), strength=KB.dist(r["strength"]), modes=m, simulatability=r["simul"]))
    sh = lambda kind: [o["modes"]["main_share"] for o in out if o.get("kind") == kind and isinstance(o.get("modes"), dict) and KB.fin(o["modes"]["main_share"])]
    dpf = lambda kind: [o["damage_per_firing"]["p50"] for o in out if o.get("kind") == kind and isinstance(o.get("damage_per_firing"), dict) and KB.fin(o["damage_per_firing"].get("p50"))]
    write(ctx, "k/k3h.json", dict(neurons=out, H_vs_random=dict(main_share_H=KB.dist(sh("H")), main_share_R=KB.dist(sh("R")),
                                                               damage_p50_H=KB.dist(dpf("H")), damage_p50_R=KB.dist(dpf("R")))),
          config=dict(window=16, max_firings=100, cutoff="top 5 % of the pole"), labels=["descriptive (K3h, tier 2)", "no atlas data used"])
