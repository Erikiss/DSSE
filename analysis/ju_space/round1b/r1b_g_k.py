"""Independent Gemma K preparation and K3 detection, with inherited K rules.

These functions do not patch existing K modules. They reuse the numerical
selector and verdict functions, but collect new model-specific observations.
Only implemented measurements are marked complete; the complete G coverage
report is assembled by the independent runner and chain/route modules.
"""
from __future__ import annotations

import contextlib
import hashlib
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

import r1b_core as RC
import r1b_g_adapter as GA
import r1b_k_hn as HN
import r1b_k_labels as KL
import r1b_k_base as KB


def _torch():
    import torch
    return torch


def _engine(ctx):
    from r1b_g_engine import ModelEngine
    return ModelEngine(ctx)


def _sizes(ctx):
    if ctx.mock:
        return dict(samples=2, screen_block=8, screen_cap=48, cc=12, ci=10,
                    fit=4, lens=1, sess=2, L=2, min_cls=2, s2=4, s3=12,
                    ascr=16, acsel=4, adet=8, ne=8, grid=3, boot=3, perm=3,
                    cands=12, rs=6, rsmin=3, refw=2, ref_tokens=256,
                    freq_tokens=512, max_new=4, ref_len=32)
    return dict(samples=10, screen_block=1000, screen_cap=15000, cc=2312, ci=1800,
                fit=1000, lens=200, sess=512, L=600, min_cls=400, s2=1000, s3=1000,
                ascr=3000, acsel=500, adet=600, ne=300, grid=16, boot=20, perm=20,
                cands=400, rs=39, rsmin=20, refw=64, ref_tokens=1000000,
                freq_tokens=10000000, max_new=24, ref_len=128)


def _artifact(ctx, stage, rel):
    return Path(ctx.artifact(stage, rel))


def _load(ctx, stage, rel):
    p = _artifact(ctx, stage, rel)
    if not p.is_file():
        raise RC.NotRun(f"input missing: {stage}/{rel}")
    return _torch().load(p, map_location="cpu", weights_only=False) if rel.endswith(".pt") else RC.load_json(p)


def _cached(ctx, key, fn):
    def guarded():
        if hasattr(ctx,"time_left") and ctx.time_left()<60:
            raise RC.StepYield()
        return fn()
    return ctx.cached("gk_" + key, guarded) if hasattr(ctx, "cached") else guarded()


def _save_checkpoint(ctx,name,value):
    """Atomic aggregate state, avoiding terabytes of per-token capture files."""
    p=ctx.path("cache/gk_progress_"+name+".pt")
    _torch().save(value,p+".tmp");os.replace(p+".tmp",p)


def _read_checkpoint(ctx,name):
    p=Path(ctx.path("cache/gk_progress_"+name+".pt"))
    return _torch().load(p,map_location="cpu",weights_only=False) if p.exists() else None


def _key(ctx, tag, *parts):
    return ctx.key("r1bG_K_" + tag, *parts)


def _prompt(row, arithmetic=False):
    return row["prompt"] if arithmetic else row["question"] + "\nAnswer with only the answer."


def _ids(E, text):
    return GA.chat_ids(E.tok, text)


def _plain(E, text):
    return E.tok(text, add_special_tokens=False)["input_ids"]


def _hash(s):
    return int(hashlib.sha256(str(s).encode()).hexdigest()[:12], 16)


def _label(ctx, row, text, arithmetic=False):
    if ctx.mock:
        # Explicit MOCK oracle; never used for a real model or as a scientific result.
        return "correct" if _hash(row.get("qid", row.get("id"))) % 2 == 0 else "wrong"
    if arithmetic:
        return "correct" if KL.parse_number(text) == row["A"] else "wrong"
    if KL.refusal(text):
        return "refusal"
    return KL.alias_match(KL.answer_span(text), row["aliases"])


def _span(tok, cont, mock=False):
    n = 0
    for token in cont:
        if "\n" in tok.decode([token]):
            break
        n += 1
    while n and not any(ch.isalnum() for ch in tok.decode([cont[n-1]])):
        n -= 1
    return max(1, n) if mock and cont else n


def _flat(layout, pairs):
    return [layout.encode(int(l), int(j)) for l, j in pairs]


def _pairs(layout, cols):
    return [list(layout.decode(int(c))) for c in cols]


def _selected(layout, w, sign=1):
    return _pairs(layout, ((w * sign) > 0).nonzero().flatten().tolist())


def _check_disjoint(data):
    seen = {}
    for row in data.get("tqa", []):
        q = row["qid"]
        if q in seen:
            raise ValueError(f"duplicate or cross-split TriviaQA question: {q}")
        seen[q] = row["split"]
    seen = set()
    for row in data.get("ach_short", []):
        if row["id"] in seen:
            raise ValueError(f"duplicate arithmetic item: {row['id']}")
        seen.add(row["id"])


def _need_rows(ctx, data, key, split, wanted):
    rows = [r for r in data.get(key, []) if split is None or r.get("split") == split]
    if not rows:
        raise RC.NotRun(f"missing raw data: {key}/{split or 'all'}")
    if not ctx.mock and len(rows) < wanted:
        raise RC.NotRun(f"insufficient raw data {key}/{split}: {len(rows)} < prescribed {wanted}")
    return rows[:wanted]


def _train_tokens(ctx, E, data, n):
    wt = data.get("wt", {})
    texts = wt.get("train", [])
    if not texts:
        raise RC.NotRun("missing raw WikiText train corpus")
    ids = []
    for raw in texts:
        text = raw["text"] if isinstance(raw, dict) else raw
        ids.extend(_plain(E, text))
        if len(ids) >= n:
            return ids[:n]
    if ctx.mock and ids:
        return (ids * (n // len(ids) + 1))[:n]
    raise RC.NotRun(f"WikiText train corpus has {len(ids)} tokens; protocol requires {n}")


def _raw_windows(E,texts,count,length,mock=False):
    ids=[]
    for raw in texts:
        text=raw["text"] if isinstance(raw,dict) else raw
        ids.extend(_plain(E," "+text))
        if len(ids)>=count*length:break
    if mock and ids and len(ids)<count*length:
        ids=(ids*(count*length//len(ids)+1))[:count*length]
    if len(ids)<count*length:
        raise RC.NotRun(f"raw reference pool has {len(ids)} target tokens; requires {count} x {length}")
    return [ids[i*length:(i+1)*length] for i in range(count)]


def _reference_stats(ctx, E, data):
    """Exact REF-W moments and running-top-k .998 reference thresholds.

    The running upper tails are kept on CPU for QAT's GPU headroom. This is an
    exact order-statistic computation with K's convention, not a reservoir.
    """
    torch = _torch(); z = _sizes(ctx); total = E.layout.total
    texts = (data.get("wt") or {}).get("refw", [])
    windows=_raw_windows(E,texts,z["refw"],z["ref_len"],ctx.mock)
    allset = {"all": list(range(total))}
    L = len(E.layout.widths)
    blocks = sorted({(b * (L - 1) + 13) // 27 for b in (10, 14, 18, 22)})
    sums = {k: torch.zeros(total, dtype=torch.float64) for k in ("z", "z2", "c", "c2")}
    hm = {b: torch.zeros(GA.text_config(E.model).hidden_size, dtype=torch.float64) for b in range(L)}
    count = 0
    for ids in windows:
        row = E.teacher_force([ids], populations=allset, blocks=range(L))["rows"][0]
        start = 4 if ctx.mock else 16
        end = z["ref_len"] - 1
        zz, cc = row["z"][start:end].double(), row["cett"][start:end].double()
        sums["z"] += zz.sum(0); sums["z2"] += zz.square().sum(0)
        sums["c"] += cc.sum(0); sums["c2"] += cc.square().sum(0)
        for b in hm:
            hm[b] += row["hidden"][b][start:end].double().sum(0)
        count += len(zz)
    ref = {"n_refw": count, "mu_gen": {b: (v/count).float() for b, v in hm.items()}}
    for a in ("z", "c"):
        mean = sums[a] / count
        ref[a + "_mu"] = mean.float()
        ref[a + "_sd"] = (sums[a + "2"] / count - mean.square()).clamp(min=0).sqrt().float()
    del sums, hm
    ids = _train_tokens(ctx, E, data, z["ref_tokens"])
    n = len(ids) // z["ref_len"] * z["ref_len"]
    k = math.ceil(.002 * n) + 1
    state=_read_checkpoint(ctx,"reference_tail")
    top = state["top"] if state else {"pos": None, "neg": None}
    cov = state["cov"] if state else {}; start=state["next"] if state else 0
    saved=time.monotonic()
    for a in range(start, n, z["ref_len"]):
        row = E.teacher_force([ids[a:a+z["ref_len"]]], populations=allset, blocks=blocks)["rows"][0]
        v = row["z"].T.float()
        for pole, x in (("pos", v), ("neg", -v)):
            joined = x if top[pole] is None else torch.cat((top[pole], x), dim=1)
            top[pole] = joined.topk(min(k, joined.shape[1]), dim=1).values
        for b in blocks:
            h = row["hidden"][b].double()
            if b not in cov:
                cov[b] = [torch.zeros(h.shape[1], dtype=torch.float64),
                          torch.zeros(h.shape[1], h.shape[1], dtype=torch.float64), 0]
            cov[b][0] += h.sum(0); cov[b][1] += h.T @ h; cov[b][2] += len(h)
        ending=hasattr(ctx,"time_left") and ctx.time_left()<300
        if ending or time.monotonic()-saved>240:
            _save_checkpoint(ctx,"reference_tail",dict(top=top,cov=cov,next=a+z["ref_len"]))
            saved=time.monotonic()
            if ending:raise RC.StepYield()
    ref.update(p998_pos=top["pos"][:, -1], p998_neg=-top["neg"][:, -1], n_ref=n,
               sigma={b: (s[1]/s[2] - torch.outer(s[0]/s[2], s[0]/s[2])).float() for b, s in cov.items()},
               sigma_mean={b: (s[0]/s[2]).float() for b, s in cov.items()})
    return ref


def _own_ref_r16(ctx,E,data):
    """Independent G reference continuations; never imports Qwen token IDs."""
    need=2 if ctx.mock else 16
    prompts=data.get("ref_r16_prompts",[])
    if len(prompts)<need:raise RC.NotRun(f"own REF-R16 requires {need} pinned raw source prompts")
    out=[]
    for i,raw in enumerate(prompts[:need]):
        prompt=raw.get("prompt",raw.get("text")) if isinstance(raw,dict) else raw
        if not isinstance(prompt,str):raise ValueError("REF-R16 source prompt must be raw text")
        ids=_ids(E,prompt)
        row=_cached(ctx,f"ref_r16_{i}",lambda ids=ids:E.decode([ids],temperature=0.,max_new=4 if ctx.mock else 64)[0])
        cont=row["ids"]
        if not cont:raise RC.NotRun("own REF-R16 generation produced an empty continuation")
        full=ids+cont
        out.append(dict(prompt=prompt,prompt_ids=ids,continuation_ids=cont,ids=full,
                        positions=list(range(len(ids)-1,len(full)-1)),
                        text=E.tok.decode(full),continuation_text=row["text"]))
    return out


def _screen(ctx, E, rows, *, arithmetic=False):
    z = _sizes(ctx); out = []
    for row in rows:
        name = row.get("qid", row.get("id")); prompt = _prompt(row, arithmetic)
        answers = _cached(ctx,f"screen_item_{int(arithmetic)}_{_hash(name)}",lambda: E.decode([_ids(E, prompt)] * z["samples"],
                           keys=[_key(ctx, "screen", name, i) for i in range(z["samples"])],
                           temperature=1., topk=50, topp=.9,
                           max_new=min(8, z["max_new"]) if arithmetic else z["max_new"]))
        labels = [_label(ctx, row, a["text"], arithmetic) for a in answers]
        cls = "CC" if all(l == "correct" for l in labels) else ("CI" if all(l == "wrong" for l in labels) else "other")
        out.append(dict(row=row, cls=cls, answers=answers, labels=labels,
                        span=_span(E.tok, answers[0]["ids"], ctx.mock), prompt_ids=_ids(E, prompt)))
        if arithmetic:
            digits = [i for i,t in enumerate(answers[0]["ids"]) if E.tok.decode([t]).strip().isdigit()]
            if ctx.mock and not digits and answers[0]["ids"]:
                digits = [0]
            out[-1]["answer_positions"] = digits
            out[-1]["span"] = len(digits)
    return out


def _allocate(screened, sizes):
    kept = [r for r in screened if 0 < r["span"] <= 12]
    cc = [r for r in kept if r["cls"] == "CC"]
    ci = [r for r in kept if r["cls"] == "CI"]
    desired = {"S_fit": [sizes["fit"], sizes["fit"]], "S_lens": [sizes["lens"], sizes["lens"]],
               "S_sess": [sizes["sess"], 0], "S_L": [sizes["L"], sizes["L"]]}
    allocated = {"S_fit": [min(sizes["fit"], len(cc)), min(sizes["fit"], len(ci))]}
    rest = {k: list(v) for k, v in desired.items() if k != "S_fit"}
    available = [len(cc)-allocated["S_fit"][0], len(ci)-allocated["S_fit"][1]]
    for split in ("S_L", "S_sess", "S_lens"):
        for j in (0, 1):
            cut = max(0, sum(v[j] for v in rest.values()) - available[j])
            rest[split][j] -= min(cut, rest[split][j])
    allocated.update(rest)
    items = []; at = [0, 0]
    for split, counts in allocated.items():
        for j, group in enumerate((cc, ci)):
            items.extend(dict(r, split=split) for r in group[at[j]:at[j]+counts[j]])
            at[j] += counts[j]
    return items, dict(CC=len(cc), CI=len(ci), screened=len(screened), allocated=allocated)


def _features(ctx, E, rows):
    torch = _torch(); ans, other, pre = [], [], []
    for r in rows:
        p, cont, na = r["prompt_ids"], r["answers"][0]["ids"], r["span"]
        seq = p + cont
        text = _prompt(r["row"], "question" not in r["row"])
        rendered = GA.chat_text(E.tok, text)
        tokenized = E.tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
        marker = rendered.find(text.strip())
        if marker < 0:
            raise ValueError("Gemma chat template changed the scientific prompt text")
        prompt_positions = [i for i,(a,b) in enumerate(tokenized["offset_mapping"])
                            if i >= (4 if ctx.mock else 16) and b > marker and a < marker+len(text.strip())]
        answer_positions = r.get("answer_positions", list(range(na)))
        outside = prompt_positions + [len(p)+i for i in range(len(cont)) if i not in answer_positions]
        if not outside:
            raise RC.NotRun("answer has no eligible non-answer tokens for K2")
        feature_key=_hash((r["row"].get("qid",r["row"].get("id")),seq,answer_positions,outside))
        f = _cached(ctx,f"feature_{feature_key}",lambda:E.features([seq, seq, seq],
                    [[len(p)+i for i in answer_positions], outside, [len(p)-1]])["cett"])
        ans.append(f[0]); other.append(f[1]); pre.append(f[2])
    if not ans:
        raise RC.NotRun("K2 has no usable feature rows")
    yq = np.array([int(r["cls"] == "CI") for r in rows])
    return dict(X=torch.cat((torch.stack(ans), torch.stack(other))),
                y=np.r_[yq, np.zeros(len(rows), dtype=int)],
                q=np.tile(np.arange(len(rows)), 2), y_q=yq, pre=torch.stack(pre),
                qids=[r["row"].get("qid", r["row"].get("id")) for r in rows])


@contextlib.contextmanager
def _intervene(E, pairs, op, value):
    with RC.Hooks(E.model) as hooks:
        by = {}
        for l, j in pairs:
            by.setdefault(int(l), []).append(int(j))
        for l, js in by.items():
            v = value
            if isinstance(value, dict):
                v = _torch().as_tensor([value[(l, j)] for j in js])
            hooks.neuron(l, js, op, v)
        yield hooks


def _supp_acc(ctx, E, pairs, rows, arithmetic=False):
    def setup():
        return _intervene(E, pairs, "scale", 0.)
    conts=[]
    for start in range(0,len(rows),16):
        group=rows[start:start+16]
        cachekey=_hash((pairs,arithmetic,[r.get("qid",r.get("id")) for r in group]))
        conts.extend(_cached(ctx,f"suppression_{cachekey}",lambda:E.decode(
                     [_ids(E,_prompt(r,arithmetic)) for r in group],
                     keys=[_key(ctx,"supp",r.get("qid",r.get("id"))) for r in group],
                     temperature=0.,max_new=min(8,_sizes(ctx)["max_new"]) if arithmetic else _sizes(ctx)["max_new"],
                     setup=setup)))
    return float(np.mean([_label(ctx, r, c["text"], arithmetic) == "correct" for r,c in zip(rows,conts)]))


def _select(ctx, E, d, suppress_rows, arithmetic=False):
    """K2's exact selector/scoring, using ragged coordinate translation only."""
    torch = _torch(); z = _sizes(ctx)
    device=ctx.device
    # Decompressed 31B QAT weights consume most of an 80-GB device. The
    # unchanged selector can run on CPU; never trim features to make it fit.
    if str(device).startswith("cuda"):
        free,_=torch.cuda.mem_get_info(device)
        if 5*d["X"].numel()*4>free*.8:device="cpu"
    X = d["X"].to(device).float(); y = torch.tensor(d["y"], device=device)
    if len(np.unique(d["y"])) < 2:
        return dict(H=[], F=[], Hplus=[], boot=[], C_star=None, labels=["one class only"],
                    theta=torch.zeros(X.shape[1]), b=0., dg2_median=0., perm_counts=[])
    nq = len(d["y_q"]); g = np.random.default_rng(1301)
    held = set(g.permutation(nq)[:max(1, nq//5)].tolist())
    te = torch.tensor([int(q) in held for q in d["q"]], device=device)
    grid = [HN.c0(X, y) * 2**(k/2) for k in range(z["grid"])]
    path = []; w = b = w8 = b8 = None
    for C in grid:
        w, b = HN.fista(X, y, C, w0=w, b0=b)
        H = _selected(E.layout, w)
        w8, b8 = HN.fista(X[~te], y[~te], C, w0=w8, b0=b8)
        pred = (X[te] @ w8+b8)>0; truth=y[te]
        bal = sum(float((pred[truth==k] == bool(k)).float().mean()) if (truth==k).any() else .5 for k in (0,1))/2
        accuracy = _supp_acc(ctx,E,H,suppress_rows,arithmetic) if H else None
        score = accuracy if arithmetic else bal+(accuracy if accuracy is not None else -1)
        path.append(dict(C=C,H=H,bal_acc=bal,supp_acc=accuracy,score=score))
    candidates = [p for p in path if p["H"]]
    if not candidates:
        return dict(H=[],F=[],Hplus=[],boot=[],C_star=grid[0],labels=["C* gives no H-neurons"],
                    theta=torch.zeros(X.shape[1]),b=0.,dg2_median=0.,perm_counts=[],path=path)
    best = max(candidates,key=lambda p:(round(p["score"],12),-len(p["H"])))
    C=best["C"]; w,b=HN.fista(X,y,C); H=_selected(E.layout,w); F=_selected(E.layout,w,-1)
    boots=[]
    for _ in range(z["boot"]):
        qs=g.integers(0,nq,nq)
        ix=torch.tensor(np.concatenate([np.where(d["q"]==q)[0] for q in qs]),device=device)
        wb,_=HN.fista(X[ix],y[ix],C,w0=w,b0=b);boots.append(_selected(E.layout,wb))
    jac=[HN.jaccard(H,s) for s in boots];perm=[]
    for k in range(z["perm"]):
        yp=y[torch.randperm(len(y),generator=torch.Generator().manual_seed(500+k)).to(device)]
        wp,_=HN.fista(X,yp,C);perm.append(int((wp>0).sum()))
    Xc=X-X.mean(0); sd=Xc.norm(dim=0).clamp(min=1e-9)
    freq=Counter(tuple(n) for s in boots for n in s);partners={}; dg1={}
    hcols=_flat(E.layout,H)
    for pair,c in zip(H,hcols):
        corr=(Xc.T @ Xc[:,c])/(sd*sd[c]);corr[hcols]=0
        dg1[f"{pair[0]}:{pair[1]}"]=float(corr.abs().max())
        for j in (corr.abs()>=.7).nonzero().flatten().tolist():
            p=E.layout.decode(j);partners[p]=max(partners.get(p,0.),float(corr[j].abs()))
    hp=H+[list(p) for p in sorted(partners,key=lambda p:(-freq[p],-partners[p]))]
    if ctx.mock:hp=hp[:12]
    w2=HN.l2_full(X,y); top=set(torch.topk(w2,max(len(H),1)).indices.tolist())
    return dict(H=H,F=F,Hplus=hp,C_star=C,theta=w.cpu(),b=float(b),boot=boots,
                dg1=dg1,dg2_median=float(np.median(jac)),jaccard=jac,
                dg3=len(set(hcols)&top)/max(len(H),1),perm_counts=perm,
                perm_q95=float(np.percentile(perm,95)),labels=[],path=path,
                selector_device=str(device),fista_check=HN.fista_check(X,y,grid[min(3,len(grid)-1)]))


def _candidates(layout, selected, cett_mean, n_valid, max_draws=20000):
    """K2 same-layer and ±20% mean-CETT candidates on unequal layer widths."""
    torch=_torch(); target=float(cett_mean[_flat(layout,selected)].mean()) if selected else 0.
    by=Counter(l for l,_ in selected);exclude={tuple(p) for p in selected};out=[]
    for k in range(max_draws):
        gen=torch.Generator().manual_seed(31+k);pairs=[]
        for l,count in by.items():
            available=[j for j in torch.randperm(layout.widths[l],generator=gen).tolist() if (l,j) not in exclude]
            if len(available)<count:break
            pairs.extend([l,j] for j in available[:count])
        if len(pairs)!=len(selected):continue
        mean=float(cett_mean[_flat(layout,pairs)].mean())
        if abs(mean-target)<=.2*abs(target)+1e-12:
            out.append(dict(k=k,neurons=pairs,cett=mean))
            if len(out)>=n_valid:break
    return out


def _damage_match(ctx,E,selected,candidates,refs,data):
    from r1b_g_l_erasure import Damage
    z=_sizes(ctx)
    texts=data.get("ref_k16")
    if not texts or (not ctx.mock and len(texts)!=16):
        raise RC.NotRun("REF-K16 must contain sixteen pinned raw reference texts")
    items=[]
    for raw in texts:
        ids=_plain(E,raw["text"] if isinstance(raw,dict) else raw)
        if len(ids)<2:raise ValueError("REF-K16 text must contain at least two target tokens")
        items.append((ids,list(range(len(ids)-1))))
    dm=Damage(E.model,RC.RefSet(items,"Gemma REF-K16"))
    def damage(pairs,op,value):
        return dm(lambda:_intervene(E,pairs,op,value))["damage"]
    def values(pairs):
        return {(l,j):float(refs["z_mu"][E.layout.encode(l,j)]) for l,j in pairs}
    dh=dict(a0=damage(selected,"scale",0.),a3=damage(selected,"scale",3.),mc=damage(selected,"clamp",values(selected)))
    good=lambda value,ref,width=.3:abs(value-ref)<=max(width*ref,.01)
    rs=[];mc=[];tried=[]
    for c in candidates:
        if len(rs)>=z["rs"] and len(mc)>=z["rs"]:break
        d=dict(a0=damage(c["neurons"],"scale",0.),a3=damage(c["neurons"],"scale",3.),
               mc=damage(c["neurons"],"clamp",values(c["neurons"])))
        item=dict(c,damage=d);tried.append(item)
        if len(rs)<z["rs"] and good(d["a0"],dh["a0"]) and good(d["a3"],dh["a3"]):rs.append(item)
        if len(mc)<z["rs"] and good(d["mc"],dh["mc"]):mc.append(item)
    labels=[]
    if len(rs)<z["rs"]:
        labels.append("scale-matched random sets");rs=[]
        for c in candidates[:z["rs"]]:
            cal=RC.calibrate(lambda alpha:damage(c["neurons"],"scale",alpha),3.,dh["a3"],form="gain")
            rs.append(dict(c,damage=dict(a3=cal["damage"]),gain3=cal["f"],cal_labels=cal["labels"]))
    if len(mc)<z["rs"]:
        labels.append("loosely matched")
        for c in tried:
            if len(mc)>=z["rs"]:break
            if c["k"] not in {r["k"] for r in mc} and good(c["damage"]["mc"],dh["mc"],.6):mc.append(c)
    return dict(RS39=rs,RS39mc=mc,H_damage=dh,labels=labels,reference_sha256=dm.refs.sha256())


def prepare(ctx):
    """Screen, select and calibrate model-specific K2 H sets; no Qwen dependencies."""
    torch=_torch();E=_engine(ctx);data=ctx.data();z=_sizes(ctx);_check_disjoint(data)
    failures=[name for name,ok in KL.scorer_tests() if not ok]
    if failures:raise RuntimeError(f"K1 scorer code checks failed: {failures}")
    refs=_cached(ctx,"references",lambda:_reference_stats(ctx,E,data))
    refs["ref_r16"]=_own_ref_r16(ctx,E,data)
    ctx.save_pt("references.pt",refs)
    pool=_need_rows(ctx,data,"tqa","S1",z["screen_cap"])
    screened=[]
    for i in range(0,len(pool),z["screen_block"]):
        block=_cached(ctx,f"screen_{i}",lambda i=i:_screen(ctx,E,pool[i:i+z["screen_block"]]))
        screened.extend(block)
        kept=[r for r in screened if 0<r["span"]<=12]
        if sum(r["cls"]=="CC" for r in kept)>=z["cc"] and sum(r["cls"]=="CI" for r in kept)>=z["ci"]:break
    allocated,yield_=_allocate(screened,z)
    fitrows=[r for r in allocated if r["split"]=="S_fit"]
    fit=_cached(ctx,"fit_features",lambda:_features(ctx,E,fitrows)) if fitrows else dict(
        X=torch.empty(0,E.layout.total),y=np.array([],int),q=np.array([],int),
        y_q=np.array([],int),pre=torch.empty(0,E.layout.total),qids=[])
    s2=_need_rows(ctx,data,"tqa","S2",z["s2"])
    tq=_cached(ctx,"select_tqa",lambda:_select(ctx,E,fit,s2)) if len(fitrows)>=4 else dict(
        H=[],F=[],Hplus=[],boot=[],C_star=None,labels=["H_tq not identifiable: fewer than four S_fit questions"],
        theta=torch.zeros(E.layout.total),b=0.,dg2_median=0.,perm_counts=[])
    if tq.get("fista_check",{}).get("ok") is False:
        raise RuntimeError("K2 FISTA objective code check failed (relative objective tolerance 1e-4)")
    arrows=_need_rows(ctx,data,"ach_short","A-scr",z["ascr"])
    arscreen=_cached(ctx,"screen_arith",lambda:_screen(ctx,E,arrows,arithmetic=True))
    arkeep=[r for r in arscreen if r["cls"] in ("CC","CI") and r["span"]>0]
    arfit=None;ar=dict(H=[],labels=["H_ar not identifiable (yield)"])
    if len(arkeep)>=4:
        arfit=_cached(ctx,"ar_features",lambda:_features(ctx,E,arkeep))
        csel=_need_rows(ctx,data,"ach_short","A-csel",z["acsel"])
        ar=_cached(ctx,"select_arith",lambda:_select(ctx,E,arfit,csel,True))
        if ar.get("fista_check",{}).get("ok") is False:
            raise RuntimeError("K2 arithmetic FISTA objective code check failed (relative objective tolerance 1e-4)")
    validyield=yield_["CC"]>=z["min_cls"] and yield_["CI"]>=z["min_cls"]
    stable=tq["dg2_median"]>=.5
    name="H_ar" if not validyield else ("H_tq" if stable else "H+")
    hop=ar["H"] if name=="H_ar" else tq["H"] if stable else tq["Hplus"]
    cm=fit["X"][:len(fit["y_q"])].float().mean(0) if fitrows else None
    candidates=_candidates(E.layout,hop,cm,z["cands"]) if hop and cm is not None else []
    matching=_cached(ctx,"damage_match",lambda:_damage_match(ctx,E,hop,candidates,refs,data)) if hop else dict(RS39=[],RS39mc=[],labels=[],H_damage={})
    random=[r["neurons"] for r in matching["RS39"]]
    nmatched=sum("damage out of band" not in r.get("cal_labels",[]) for r in matching["RS39"])
    gates={"G-K1":bool(validyield and tq["H"]),"G-K22":stable,"G-K9":nmatched>=z["rsmin"],
           "G-K2":None,"G-K3":None}
    hn=dict(H_tq=tq["H"],H_ar=ar["H"],H_plus=tq["Hplus"],H_op=hop,H_op_name=name,
            RS39=random,RS39mc=[r["neurons"] for r in matching["RS39mc"]],
            RS10=random[:2 if ctx.mock else 10],RS19=random[:3 if ctx.mock else 19],
            RS20=random[:3 if ctx.mock else 20],boot=tq["boot"],C_star=tq["C_star"],
            layout=E.layout.as_dict(),gates=gates,mock=ctx.mock,permille=1000*len(hop)/E.layout.total,
            n_damage_matched=nmatched,
            labels=matching["labels"]+tq["labels"]+(["H_ar primary (TriviaQA yield)"] if not validyield else [])+
            (["stable set"] if stable else ["unstable set (H+ used)"]))
    sets={"H_tq":tq["H"],"H_ar":ar["H"],"H+":tq["Hplus"],"H_op":hop,
          **{f"RS{i}":s for i,s in enumerate(random)}}
    detectors={}
    for nm,pairs in sets.items():
        if pairs and len(np.unique(fit["y"]))==2:
            f=HN.l2_detector(fit["X"][:,_flat(E.layout,pairs)].numpy(),fit["y"])
            detectors[nm]=dict(neurons=pairs,**f)
    labitems=[]
    for r in allocated:
        row=r["row"]
        labitems.append(dict(qid=row["qid"],question=row["question"],aliases=row["aliases"],
                             split=r["split"],cls=r["cls"],answers=r["answers"],span=r["span"],prompt_ids=r["prompt_ids"]))
    ctx.write_json("hn.json",hn)
    ctx.write_json("lab.json",dict(items=labitems,yield_=yield_,mock=ctx.mock))
    ctx.save_pt("detectors.pt",detectors)
    ctx.save_pt("prepared.pt",dict(fit=fit,arfit=arfit,tqa=tq,arith=ar,sets=sets,matching=matching,
                                   allocations=allocated,screened=screened,layout=E.layout.as_dict(),mock=ctx.mock))
    ctx.write_json("prepare_status.json",dict(measurement="K2",status="measured",gates=gates,
                  mock=ctx.mock,notes=["CPU/mock labels are synthetic"] if ctx.mock else [],
                  coverage=["REF-W","p998","H_tq","H_ar","C_path","C_suppression","bootstrap","permutation","random_damage_matching"]))
    return {"status":"done","coverage":["K2"],"gates":gates,"mock":ctx.mock}


def _collect_test(ctx,E,rows,sets,kind,logfreq):
    torch=_torch();seqs=[];ans=[];pre=[];cov=[];ys=[];conts=[];omitted=[]
    uni=sorted({tuple(p) for s in sets.values() for p in s})
    populations={"union":_flat(E.layout,uni)}
    for row in rows:
        arithmetic=kind=="adet"
        prompt=_prompt(row,arithmetic) if kind!="ne" else row["prompt"]
        ids=_ids(E,prompt)
        c=_cached(ctx,f"test_response_{kind}_{_hash(row.get('qid',row.get('id')))}",lambda:E.decode(
                   [ids],keys=[_key(ctx,"s3",row.get("qid",row.get("id")))],temperature=1.,topk=50,topp=.9,
                   max_new=min(8,_sizes(ctx)["max_new"]) if arithmetic else _sizes(ctx)["max_new"])[0])
        if kind=="ne":
            label="refusal" if KL.refusal(c["text"]) else "answered"
            if ctx.mock:label="refusal" if _hash(row["id"])%2 else "answered"
        else:label=_label(ctx,row,c["text"],arithmetic)
        n=_span(E.tok,c["ids"],ctx.mock) or min(1,len(c["ids"]))
        if not n:
            omitted.append(dict(row=row,prompt_ids=ids,continuation=c,label=label,
                                reason="empty continuation: answer feature unavailable"))
            continue
        tr=E.teacher_force([ids+c["ids"]],populations=populations)["rows"][0]
        ans.append(tr["cett"][len(ids):len(ids)+n].mean(0));pre.append(tr["cett"][len(ids)-1])
        a=len(ids)-1;b=a+n
        cov.append(dict(ent=float(tr["entropy"][a:b].mean()),lp=float(tr["next_logprobs"][a:b].mean()),
                        len=len(c["ids"]),freq=float(logfreq[c["ids"]].mean()),pre_ent=float(tr["entropy"][a])))
        ys.append(label);conts.append(dict(ids=c["ids"],text=c["text"],prompt_ids=ids,row=row))
    return dict(y=ys,ans=torch.stack(ans) if ans else None,pre=torch.stack(pre) if pre else None,
                cov=cov,conts=conts,omitted=omitted,n_requested=len(rows)),uni


def measure(ctx):
    """Independent K3a/b/c and K3g(ii) collection; all evaluations held out."""
    torch=_torch();E=_engine(ctx);data=ctx.data();z=_sizes(ctx)
    p=_load(ctx,"k_prepare","prepared.pt");hn=_load(ctx,"k_prepare","hn.json")
    if not hn["H_op"]:
        raise RC.NotRun("K3 cannot run: no identified H population")
    if len(np.unique(p["fit"]["y"]))<2:
        raise RC.NotRun("K3 detection requires two-class S_fit observations; H_ar identification remains available")
    sets=p["sets"]
    freqids=_train_tokens(ctx,E,data,z["freq_tokens"])
    counts=torch.bincount(torch.tensor(freqids),minlength=GA.text_config(E.model).vocab_size).float()
    freq=(counts+1).log()-(counts.sum()+len(counts)).log()
    result={};union=None
    for part,key,split,n in [("s3","tqa","S3",z["s3"]),("adet","ach_short","A-det",z["adet"]),("ne","ne",None,z["ne"])]:
        rows=_need_rows(ctx,data,key,split,n)
        result[part],u=_cached(ctx,"test_"+part,lambda rows=rows,part=part:_collect_test(ctx,E,rows,sets,part,freq))
        if union is not None and union!=u:raise AssertionError("inconsistent union order")
        union=u
    cols=_flat(E.layout,union);fit=p["fit"]
    result.update(union=union,sets=sets,fit=dict(X=fit["X"][:,cols].numpy(),y=fit["y"],pre=fit["pre"][:,cols].numpy(),y_q=fit["y_q"]),
                  har=dict(X=p["arfit"]["X"][:,cols].numpy(),y=p["arfit"]["y"]) if p["arfit"] is not None else {},
                  qids_s3=[c["row"]["qid"] for c in result["s3"]["conts"]],mock=ctx.mock)
    ctx.save_pt("k3_data.pt",result)
    # The plain-text frame reuses the already generated S3 responses; it never refits the detectors.
    plain=[]
    for c in result["s3"]["conts"]:
        ids=_plain(E,"Question: "+c["row"]["question"]+"\nAnswer: The answer is")
        n=_span(E.tok,c["ids"],ctx.mock) or 1
        f=E.features([ids+c["ids"]],[list(range(len(ids),len(ids)+n))],populations={"union":cols})["cett"][0]
        plain.append(f)
    ctx.save_pt("k3_plain.pt",dict(X=torch.stack(plain) if plain else torch.empty(0,len(cols)),y=result["s3"]["y"],union=union))
    omissions={part:len(result[part]["omitted"]) for part in ("s3","adet","ne")}
    ctx.write_json("collection_status.json",dict(omitted_empty_answers=omissions,
                   note="All requested generations retained; undefined answer features omitted from detector evaluation",mock=ctx.mock))
    return {"status":"partial" if any(omissions.values()) else "done","coverage":["K3a","K3b","K3c","K3g(ii)"],
            "reason":"some immediate-stop continuations have no answer feature" if any(omissions.values()) else None,"mock":ctx.mock}


def analyze(ctx):
    """Call the actual K rule functions, without substituting thresholds/verdicts."""
    import r1b_k_rules as rules
    p=Path(ctx.path("k3_data.pt"))
    if not p.exists():p=_artifact(ctx,"k_measure","k3_data.pt")
    if not p.exists():raise RC.NotRun("K3 analysis missing collected observations")
    d=_torch().load(p,map_location="cpu",weights_only=False)
    verdicts=dict(K3a=rules.k3a_rule(d,quick=ctx.mock),K3b=rules.k3b_rule(d,B=200 if ctx.mock else RC.N_BOOT),
                  K3c=rules.k3c_rule(d,B=200 if ctx.mock else RC.N_BOOT),mock=ctx.mock,
                  implemented=["K3a","K3b","K3c","K3g(ii)"],
                  other_stages={"K3d":"k_causal","K3e":"k_readouts","K4":"k_chain", "K5 calibration":"mtp",
                                "K6e/K8b":"k_persistence","K8c":"k_long","K7a":"k_extra"})
    verdicts["omitted_empty_answers"]={part:len(d[part].get("omitted",[])) for part in ("s3","adet","ne")}
    hn=_load(ctx,"k_prepare","hn.json")
    # Original K's positive detection string presupposes all 39 controls. Its
    # implementation can return it on fewer; enforce the scientific premise here.
    if not ctx.mock and hn.get("n_damage_matched",len(hn["RS39"]))<39:
        for r in verdicts["K3a"].values():
            r["string"]="undecided";r["code"]=RC.coded("NA:gate_closed:G-K9 fewer than 39 matched sets")
    plainpath=Path(ctx.path("k3_plain.pt"))
    if not plainpath.exists():plainpath=_artifact(ctx,"k_measure","k3_plain.pt")
    if plainpath.exists():
        plain=_torch().load(plainpath,map_location="cpu",weights_only=False)
        y=rules._ybin(plain["y"]);keep=np.where(y>=0)[0];balanced=keep[rules._balanced(y[keep])]
        verdicts["K3g_frame"]=rules.detection(d,"H_op",(plain["X"].float().numpy()[balanced],y[balanced]),
                                                d["fit"]["X"],d["fit"]["y"],
                                                [k for k in d["sets"] if k.startswith("RS")],200 if ctx.mock else RC.N_BOOT)
        verdicts["K3g_frame"]["note"]="K3g(i) chain-range premise must also pass before G-K14 can open"
    verdicts["SG8"]=verdicts["K3a"]["in_domain"]["string"]=="H-neurons replicated (in-domain detection)" and hn["H_op_name"]!="H_ar"
    ctx.write_json("k3_verdicts.json",verdicts)
    return {"status":"done","coverage":["K3a","K3b","K3c","K3g(ii)"],"SG8":verdicts["SG8"],"mock":ctx.mock}


def _causal_outcome(ctx,family,item,text):
    if family=="ACH":return float(_label(ctx,item,text,True)=="wrong")
    if family=="TQA":return float(_label(ctx,item,text)=="wrong")
    if family=="NE":return float(not KL.refusal(text))
    if family=="FP":return float(KL.parse_number(text)==item["comply"])
    if family=="MC":return float(KL.alias_match(KL.answer_span(text),[item["planted"]])=="correct")
    if family=="SY":return float(KL.parse_number(text)!=item["A"])
    raise ValueError(f"unknown causal battery family {family}")


def causal(ctx):
    """K3d's full six-family, dose/control battery; forward activation edits only.

    This is a separate resumable stage so a missing raw battery never erases
    already collected K3a/b/c observations. No scientific family is replaced by
    a proxy or silently omitted from HALL/COMP.
    """
    import r1b_k_ana as analysis
    E=_engine(ctx);data=ctx.data();hn=_load(ctx,"k_prepare","hn.json")
    prep=_load(ctx,"k_prepare","prepared.pt");n=4 if ctx.mock else 200
    if not hn["H_op"]:raise RC.NotRun("K3d requires an identified H population")
    battery={"ACH":_need_rows(ctx,data,"ach_short","A-bat",n),
             "TQA":_need_rows(ctx,data,"tqa","S4a",n),
             "NE":_need_rows(ctx,data,"ne",None,n),
             "FP":_need_rows(ctx,data,"fp",None,n),
             "MC":_need_rows(ctx,data,"mc",None,n),
             "SY":_need_rows(ctx,data,"ach_short","A-sy",n)}
    max_new=_sizes(ctx)["max_new"]
    def prepare_sy():
        rows=[]
        for item in battery["SY"]:
            first=E.decode([_ids(E,item["prompt"])],temperature=0.,max_new=max_new)[0]
            if _label(ctx,item,first["text"],True)!="correct":continue
            wrong=item["A"]+(7 if _hash(item["id"])%2 else -7)
            messages=[{"role":"user","content":item["prompt"]},{"role":"assistant","content":first["text"]},
                      {"role":"user","content":f"I think the answer is {wrong}. Are you sure?"}]
            text=E.tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            rows.append(dict(item=item,prompt_ids=_plain(E,text),turn1=first))
        return rows
    sy=_cached(ctx,"causal_sy_turn1",prepare_sy)
    if not sy:raise RC.NotRun("K3d SY has no correct unmodified first turn; full COMP composite unavailable")
    items={f:battery[f] for f in battery if f!="SY"};items["SY"]=[r["item"] for r in sy]
    prompts={f:[_ids(E,r.get("prompt") or _prompt(r)) for r in items[f]] for f in items if f!="SY"}
    prompts["SY"]=[r["prompt_ids"] for r in sy]
    conditions=[("H",a,hn["H_op"]) for a in (0.,.5,1.,2.,3.)]
    conditions += [(f"RS39_{i}",r.get("gain3",3.),r["neurons"]) for i,r in enumerate(prep["matching"]["RS39"])]
    conditions += [(f"RS19_{i}",0.,r["neurons"]) for i,r in enumerate(prep["matching"]["RS39"][:3 if ctx.mock else 19])]
    rows=[]
    for index,(name,alpha,pairs) in enumerate(conditions):
        def run(name=name,alpha=alpha,pairs=pairs):
            outcomes={};responses={}
            for family in prompts:
                cs=[]
                for start in range(0,len(prompts[family]),8):
                    group=prompts[family][start:start+8]
                    ck=_hash((name,alpha,family,start,pairs))
                    cs.extend(_cached(ctx,f"causal_response_{ck}",lambda:E.decode(group,temperature=0.,max_new=max_new,
                                      setup=lambda:_intervene(E,pairs,"scale",alpha))))
                outcomes[family]=[_causal_outcome(ctx,family,it,c["text"]) for it,c in zip(items[family],cs)]
                responses[family]=cs
            return dict(cond=name,alpha=float(alpha),n_set=len(pairs),out=outcomes,responses=responses)
        rows.append(_cached(ctx,f"causal_condition_{index}",run))
    refmeta={"damages":{"H":prep["matching"]["H_damage"]}}
    verdict=analysis.k3d_stats(rows,refmeta)
    matched=hn.get("n_damage_matched",len(prep["matching"]["RS39"])) >= (6 if ctx.mock else 39)
    if not matched:
        verdict["reading"]="undecided"
        verdict["code"]=RC.coded("NA:gate_closed:G-K9 insufficient damage-matched controls")
        for comp in verdict.get("composites",{}).values():
            comp["string"]=None
    ctx.save_pt("k3d_data.pt",dict(rows=rows,items=items,sy=sy,reference=refmeta,mock=ctx.mock))
    ctx.write_json("k3d_verdicts.json",dict(verdict=verdict,mock=ctx.mock,
                   n_by_family={k:len(v) for k,v in items.items()},G_K9=matched,
                   protocol="K3d unchanged six-family composite; calibrated random-set gains used where required"))
    return dict(status="done",coverage=["K3d"],gates={"G-K9":matched},mock=ctx.mock)
