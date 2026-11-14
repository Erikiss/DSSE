"""G-owned K6f context controls, K6h QA lens and exploratory K7a atlas.

All observations use native Gemma states, target tokenization and ragged sets.
The native attention reader includes local masks and shared KV. No model method
or global attention registry is replaced by these instruments.
"""
from __future__ import annotations

import contextlib
import hashlib
import math
from pathlib import Path

import numpy as np

import r1b_g_adapter as A
from r1b_g_engine import ModelEngine, content_hash
from r1b_g_lens import frame_spec, mapped_blocks, rollout
from r1b_g_k_chain import (_load_pt, _load_json, _clean_json, _seeds, _Channels,
                           score_tokens, channel_bases)


class _FOV:
    """Last-query attention mass on the placeholder, one scalar per forward.

    Reconstructs only one query, using the native q/k normalization, RoPE,
    scaling, attention mask and shared-KV tensor. This does not request a dense
    attention matrix or change the model's efficient attention implementation.
    It is intentionally restricted to complete-prefix forwards.
    """
    def __init__(self, engine, block, key_pos):
        self.engine, self.block, self.key_pos = engine, int(block), int(key_pos)
        self.steps, self.pending, self.handles = [], [], []

    def __enter__(self):
        import torch
        from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb
        for index in range(self.block + 1, len(self.engine.layers)):
            module = self.engine.layers[index].self_attn
            def capture(module, args, kw, output):
                if kw.get("past_key_values") is not None:
                    raise ValueError("FOV reader requires a complete native prefix")
                h = kw.get("hidden_states", args[0] if args else None)
                cos, sin = kw["position_embeddings"]
                shape = (h.shape[0], h.shape[1], -1, module.head_dim)
                q = module.q_norm(module.q_proj(h[:, -1:]).view(h.shape[0], 1, -1, module.head_dim))
                q = apply_rotary_pos_emb(q, cos[:, -1:], sin[:, -1:], unsqueeze_dim=2).transpose(1, 2)
                if module.is_kv_shared_layer:
                    k = kw["shared_kv_states"][module.layer_type][0].to(q.device)
                else:
                    k = module.k_norm(module.k_proj(h).view(shape))
                    k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)
                k = k.repeat_interleave(q.shape[1] // k.shape[1], 1)
                scores = (q.float() @ k.float().transpose(-1, -2)) * module.scaling
                mask = kw.get("attention_mask")
                if mask is not None:
                    m = mask[..., -1:, :k.shape[-2]]
                    scores = scores.masked_fill(~m, -torch.inf) if m.dtype == torch.bool else scores + m.float()
                probabilities = scores.softmax(-1)
                self.pending.append(probabilities[..., self.key_pos].mean(1).reshape(-1).detach().cpu())
            self.handles.append(module.register_forward_hook(capture, with_kwargs=True))
        def finish(*unused):
            import torch
            self.steps.append(torch.stack(self.pending).mean(0) if self.pending else torch.zeros(1))
            self.pending = []
        self.handles.append(self.engine.layers[-1].register_forward_hook(finish))
        return self

    def __exit__(self, *unused):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _populations(hn):
    result = {"H": hn.get("H_op", [])}
    result.update({f"RS{i}": p for i, p in enumerate(hn.get("RS39", []))})
    return {name: pairs for name, pairs in result.items() if pairs}


def _instrument(ctx,E,lens,source=None):
    import torch
    import r1b_core as C
    source=source or {}
    calibration=_load_json(ctx,"mtp","mtp_calibration.json") or {}
    variant=source.get("variant",calibration.get("variant","E")) if lens is not None else "E"
    norm=source.get("norm_l")
    if variant=="L" and norm is None:
        from r1b_g_lens import penultimate_norm
        norm=ctx.cached("extra_late_norm",lambda:penultimate_norm(ctx,E))
    return variant,norm,calibration


def _roll_rows(ctx, E, item, block, state, prior, seeds, frame, populations, refs,
               variant="V1", steps=16, cls=None, own=(), W=(), instrument="E", lens=None, norm_l=None):
    """One source cell, including every inherited lagged channel/covariate."""
    import torch
    source_block = block
    if instrument == "L":
        if lens is None or norm_l is None:
            raise ValueError("late-layer rollout requires its own J-map and norm calibration")
        matrix = lens["maps"][block]
        state = torch.nn.functional.normalize(state @ matrix.T, dim=-1) * norm_l
        prior = torch.nn.functional.normalize(prior @ matrix.T, dim=-1) * norm_l
        source_block = len(E.layers)-2
    with _Channels(E, populations, refs) as channels, _FOV(E, source_block, len(frame)-1) as fov:
        out = rollout(E, block, state, seeds, steps=steps, prior_state=prior,
                      frame_ids=frame, variant=instrument, keep_first=True)
    rows = []
    for rank, seed in enumerate(seeds):
        tokens = out["tok"][rank].tolist()
        row = dict(item=item["id"], block=block, ptype="qa" if cls else "t0", variant=variant,
                   instrument=instrument, seed=int(seed), seed_rank=rank, tok=tokens, aX={}, cls=cls)
        if cls is None:
            row.update(score_tokens(E.tok, [seed]+tokens, item, W=W, OWN=own))
        for field in ("lpr", "lpp", "ent", "top1", "cdm", "e", "alt"):
            row[field] = out[field][rank]
        real = channels.steps[2*rank*steps:2*(rank+1)*steps:2]
        row["aX"] = {name: torch.stack([step[name][0] for step in real]) for name in populations
                      if real and all(name in step for step in real)}
        row["fov"] = torch.stack(fov.steps[2*rank*steps:2*(rank+1)*steps:2]).reshape(-1)
        rows.append(row)
    return rows


def _cached_logp(E, prefix, placeholder, state, block, seed, selected):
    """Independent native cached path for the selected-token G-K20 check."""
    import torch
    from transformers import DynamicCache
    import r1b_core as C
    cache = DynamicCache(config=E.cfg)
    with torch.no_grad(), C.sdpa_efficient(E.device):
        out = A.text_model(E.model)(input_ids=torch.tensor([prefix], device=E.device),
                                    past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        values = []
        for step, token in enumerate(selected):
            ids = [placeholder, seed] if step == 0 else [selected[step-1]]
            start = len(prefix) if step == 0 else len(prefix)+step+1
            manager = E.patch_state(block, state, 0) if step == 0 else contextlib.nullcontext()
            with manager:
                out = A.text_model(E.model)(input_ids=torch.tensor([ids], device=E.device),
                    attention_mask=torch.ones((1, start+len(ids)), device=E.device, dtype=torch.long),
                    position_ids=torch.arange(start, start+len(ids), device=E.device)[None],
                    past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits = A.unembed(E.model, out.last_hidden_state[:, -1:], normalized=True)[0, 0].float()
            values.append(logits.log_softmax(-1)[int(token)].cpu())
    return torch.stack(values)


def _prefix_cache(E,prefix):
    import torch
    import r1b_core as C
    from transformers import DynamicCache
    with torch.no_grad(),C.sdpa_efficient(E.device):
        out=A.text_model(E.model)(input_ids=torch.tensor([prefix],device=E.device),
                                 past_key_values=DynamicCache(config=E.cfg),use_cache=True)
    return out.past_key_values


def _cached_rollout(E,frame,block,state,prior,seeds,steps,prefix_cache):
    """Native prefix-cache copies; all causal state updates use HF's own cache."""
    import copy
    import torch
    import r1b_core as C
    rows=[];special=list(E.tok.all_special_ids);start=len(frame)-1
    with torch.no_grad(),C.sdpa_efficient(E.device):
        for seed in seeds:
            caches=[copy.deepcopy(prefix_cache),copy.deepcopy(prefix_cache)]
            generated=[];rec={name:[] for name in ("lpr","lpp","ent","top1","cdm","e","alt")}
            for step in range(steps):
                ids=[frame[-1],int(seed)] if step==0 else [generated[-1]]
                offset=start if step==0 else start+step+1
                logp=[]
                for stream,value in enumerate((state,prior)):
                    manager=E.patch_state(block,value,0) if step==0 else contextlib.nullcontext()
                    with manager:
                        out=A.text_model(E.model)(input_ids=torch.tensor([ids],device=E.device),
                            position_ids=torch.arange(offset,offset+len(ids),device=E.device)[None],
                            attention_mask=torch.ones((1,offset+len(ids)),device=E.device,dtype=torch.long),
                            past_key_values=caches[stream],use_cache=True)
                    caches[stream]=out.past_key_values
                    logp.append(A.unembed(E.model,out.last_hidden_state[:,-1:],normalized=True)[0,0].float().log_softmax(-1))
                real,base=logp;valid=real.clone();valid[special]=-torch.inf
                score=real-base;score[real<valid.max()+math.log(.1)]=-torch.inf;score[special]=-torch.inf
                token=E.choose(score);generated.append(token);second=score.clone();second[token]=-torch.inf
                for name,value in (("lpr",real[token]),("lpp",base[token]),("ent",-(real.exp()*real).sum()),
                                   ("top1",real.exp().max()),("cdm",score[token]-second.max()),
                                   ("e",real[token]-base[token]),("alt",int(second.argmax()))):
                    rec[name].append(float(value))
            rows.append({"tok":torch.tensor(generated),**{name:torch.tensor(v,dtype=torch.long if name=="alt" else torch.float32)
                                                         for name,v in rec.items()}})
    return {name:torch.stack([r[name] for r in rows]) for name in rows[0]}


def _cached_rows(ctx,E,item,block,state,prior,seeds,frame,populations,refs,cache,steps,variant,gate,instrument,lens,norm_l):
    import torch
    pb=block
    if instrument=="L":
        state=torch.nn.functional.normalize(state@lens["maps"][block].T,dim=-1)*norm_l
        prior=torch.nn.functional.normalize(prior@lens["maps"][block].T,dim=-1)*norm_l
        pb=len(E.layers)-2
    with _Channels(E,populations,refs) as channels:
        out=_cached_rollout(E,frame,pb,state,prior,seeds,steps,cache)
    rows=[]
    for rank,seed in enumerate(seeds):
        tokens=out["tok"][rank].tolist()
        row={"item":item["id"],"block":block,"ptype":"t0","variant":variant,"instrument":instrument,
             "seed":seed,"seed_rank":rank,"tok":tokens,**score_tokens(E.tok,[seed]+tokens,item,OWN=gate["own"],W=gate["W"])}
        row.update({name:out[name][rank] for name in out if name!="tok"})
        real=channels.steps[2*rank*steps:2*(rank+1)*steps:2]
        row["aX"]={name:torch.stack([r[name][0] for r in real]) for name in populations
                   if real and all(name in r for r in real)}
        rows.append(row)
    return rows


def context_offsets(ctx, E, source, chains, hn, refs, lens):
    """K6f: exact F512/F2048 token fillers, eight instances and nested bootstrap."""
    import torch
    import r1b_core as C
    from r1b_k_base import ci, inside
    verifier = chains.get("verifier", {})
    block = verifier.get("b_star")
    s0 = source.get("source", {}).get("t0")
    if block is None or not s0:
        return {"rows": [], "status": "blocked", "reason": "K4 source states/verifier unavailable"}
    ids = [i for i in source.get("correct", [])[:2 if ctx.mock else 256] + source.get("wrong", [])[:2 if ctx.mock else 128] if i in s0["ids"]]
    if not ids:
        return {"rows": [], "status": "blocked", "reason": "no K4 eligible items"}
    raw = ctx.data().get("wt", {}).get("ntext", [])
    stream = sum((E.encode(x if isinstance(x, str) else x["text"], chat=False) for x in raw), [])
    lengths = (16, 32) if ctx.mock else (512, 2048)
    count = 2 if ctx.mock else 8
    if len(stream) < count*max(lengths):
        raise C.NotRun("K6f requires disjoint raw filler windows for all eight context instances")
    frames = {length: [stream[j*max(lengths):j*max(lengths)+length] for j in range(count)] for length in lengths}
    frame = source.get("frame_ids") or frame_spec(E)["frame_ids"]
    populations = _populations(hn)
    steps, nseeds = (4, 2) if ctx.mock else (16, 10)
    instrument,norm_l,calibration=_instrument(ctx,E,lens,source)
    rows, parity = [], {}
    states = s0["h"][block]
    means = (states.sum(0, keepdim=True)-states) / (len(states)-1)
    prefix_holder={}
    for length in lengths:
        ordered=sorted(ids,key=lambda ident:int(hashlib.sha256(ident.encode()).hexdigest()[:12],16)%count)
        for ident in ordered:
            index = s0["ids"].index(ident)
            inst = int(hashlib.sha256(ident.encode()).hexdigest()[:12], 16) % count
            extended = frames[length][inst]+frame
            seeds = _seeds(E, states[index], block, lens, nseeds)
            gate = chains["gating"]["gate"][ident]
            def compute():
                if prefix_holder.get("key")!=(length,inst):
                    prefix_holder.clear()
                    prefix_holder.update(key=(length,inst),cache=_prefix_cache(E,extended[:-1]))
                return _cached_rows(ctx,E,source["items"][ident],block,states[index],means[index],seeds,extended,
                                    populations,refs,prefix_holder["cache"],steps,f"V1_F{length}",gate,instrument,lens,norm_l)
            result = ctx.cached(f"k6f_{length}_{ident}", compute)
            for row in result:
                row["inst"] = inst
            rows.extend(result)
            if length not in parity:
                error = 0.
                for row in result[:2 if ctx.mock else 8]:
                    patched=states[index] if instrument=="E" else torch.nn.functional.normalize(states[index]@lens["maps"][block].T,dim=-1)*norm_l
                    patch_block=block if instrument=="E" else len(E.layers)-2
                    prior=means[index] if instrument=="E" else torch.nn.functional.normalize(means[index]@lens["maps"][block].T,dim=-1)*norm_l
                    replay=rollout(E,patch_block,patched,[row["seed"]],steps=steps,prior_state=prior,
                                   frame_ids=extended,forced=[row["tok"]])
                    error = max(error, float((replay["lpr"][0]-row["lpr"]).abs().max()))
                parity[length] = {"max_abs_dlogp": error, "ok": error < 1e-4,
                                  "tolerance": 1e-4, "method": "native DynamicCache vs full recomputation"}
            if hasattr(ctx, "time_left") and ctx.time_left() < 30:
                raise C.StepYield()
    # Recompute the no-filler comparator with the same calibrated instrument,
    # exact states, prior, seeds and normalization used in the filler cells.
    baseline = {}
    for ident in ids:
        index = s0["ids"].index(ident); seeds = _seeds(E, states[index], block, lens, nseeds)
        gate = chains["gating"]["gate"][ident]
        rr = ctx.cached(f"k6f_base_{ident}", lambda: _roll_rows(ctx,E,source["items"][ident],block,states[index],means[index],seeds,
                                  frame,populations,refs,steps=steps,own=gate["own"],W=gate["W"],
                                  instrument=instrument,lens=lens,norm_l=norm_l))
        baseline.update({(r["item"],r["seed"]):float(any(c=="CONFAB" for _,_,c in r["claims"])) for r in rr})
    pairs = [(r["item"], r["inst"], float(any(c=="CONFAB" for _,_,c in r["claims"]))-baseline[(r["item"],r["seed"])])
             for r in rows if r["variant"] == f"V1_F{max(lengths)}"]
    stats = {"string": "not run (engine parity)", "parity": parity}
    if all(p["ok"] for p in parity.values()) and len({p[0] for p in pairs}) >= 3:
        generator = np.random.default_rng(1729); draws = []; instances = sorted({p[1] for p in pairs})
        for _ in range(30 if ctx.mock else C.N_BOOT):
            values = []
            for inst in generator.choice(instances, len(instances)):
                items = sorted({p[0] for p in pairs if p[1] == inst})
                for item in generator.choice(items, len(items)):
                    values.extend(p[2] for p in pairs if p[0]==item and p[1]==inst)
            draws.append(np.mean(values))
        interval = ci(np.asarray(draws))
        stats = {"d_off":float(np.mean([p[2] for p in pairs])),"ci":interval,"p":C.p_one_sided(draws,0,"greater"),
                 "_eq":inside(interval,.05),"parity":parity,"string":"pending Q13 family correction"}
    elif all(p["ok"] for p in parity.values()):
        stats["string"] = "undecided (too few paired items)"
    return {"rows":rows,"parity":parity,"statistics":stats,"frames":frames,"b_star":block,
            "status":"measured","instrument":instrument+", fixed across filler/no-filler cells",
            "n_items":len(ids),"required_items":4 if ctx.mock else 384,
            "calibration":calibration.get("G_K5","not calibrated"),
            "mock":ctx.mock}


def qa_lens(ctx, E, hn, refs, lens):
    """K6h target-specific S_lens QA, modal OWN and same-class foreign controls."""
    import torch
    import r1b_core as C
    import r1b_k_ana as K
    from collections import Counter
    from r1b_k_labels import mentions, loop_onset, answer_span, norm_text
    labels = _load_json(ctx, "k_prepare", "lab.json") or {}
    available = [r for r in labels.get("items", []) if r["split"] == "S_lens"]
    items = sum(([r for r in available if r["cls"] == cls][:2 if ctx.mock else 200] for cls in ("CC", "CI")), [])
    counts = {cls: sum(r["cls"]==cls for r in items) for cls in ("CC", "CI")}
    if min(counts.values()) < 2:
        return {"rows":[],"status":"blocked","reason":"fewer than two S_lens items per class for a foreign control","counts":counts}
    if not ctx.mock and min(counts.values()) < 200:
        raise C.NotRun("K6h requires 200 CC and 200 CI independently allocated S_lens questions")
    blocks, frame = mapped_blocks(E, (14,18)), frame_spec(E)["frame_ids"]
    prompts = [E.encode(it["question"]+"\nAnswer with only the answer.") for it in items]
    hidden = E.hidden(prompts, blocks)
    populations = _populations(hn)
    # The QA H-link is specifically H_tq, even if H_op chose an arithmetic set.
    if hn.get("H_tq"):
        populations["H"] = hn["H_tq"]
        if hn.get("H_tq")!=hn.get("H_op") and refs is not None:
            from r1b_g_k import _candidates,_damage_match
            def match():
                prepared=_load_pt(ctx,"k_prepare","prepared.pt",required=True)
                fit=prepared["fit"]
                mean=fit["X"][:len(fit["y_q"])].float().mean(0)
                del prepared,fit
                candidates=_candidates(E.layout,hn["H_tq"],mean,12 if ctx.mock else 400)
                return _damage_match(ctx,E,hn["H_tq"],candidates,refs,ctx.data())
            matched=ctx.cached("qa_tq_matched_sets",match)
            populations={"H":hn["H_tq"],**{f"RS{i}":row["neurons"] for i,row in enumerate(matched["RS39"])}}
    source=_load_pt(ctx,"k_chain","source_states.pt") or {}
    frame=source.get("frame_ids") or frame
    instrument,norm_l,calibration=_instrument(ctx,E,lens,source)
    rows, controls = [], {}
    for block in blocks:
        hs = torch.stack([r[block][-1] for r in hidden]); means = (hs.sum(0,keepdim=True)-hs)/(len(hs)-1)
        seeds = [_seeds(E,h,block,lens,2 if ctx.mock else 10) for h in hs]
        foreign = list(range(len(items)))
        for cls in ("CC","CI"):
            group = [i for i,it in enumerate(items) if it["cls"]==cls]
            for i,j in zip(group,group[1:]+group[:1]):
                foreign[i]=j
        for i,it in enumerate(items):
            # Normalize for counting, retain the selected original answer text
            # for the exact mention parser and document deterministic ties.
            candidates = [answer_span(a["text"]) for a in it["answers"]]
            normalized = [norm_text(a) for a in candidates]
            modal = Counter(normalized).most_common(1)[0][0] if normalized else ""
            own = next((a for a,n in zip(candidates,normalized) if n==modal), "")
            controls[it["qid"]] = {"foreign":items[foreign[i]]["qid"],"modal":own,"class":it["cls"]}
            for variant in ("V1","V2"):
                j = i if variant=="V1" else foreign[i]
                def compute(i=i,j=j,it=it,variant=variant,own=own):
                    rr = _roll_rows(ctx,E,{"id":it["qid"]},block,hs[j],means[j],seeds[j],frame,populations,refs,
                                   variant=variant,steps=6 if ctx.mock else 12,cls=it["cls"],
                                   instrument=instrument,lens=lens,norm_l=norm_l)
                    for row in rr:
                        tokens = [row["seed"]]+row["tok"]
                        pieces = [E.tok.decode([t]) for t in tokens]; text="".join(pieces)
                        owners = sum(([n]*len(piece) for n,piece in enumerate(pieces)),[])
                        censor = loop_onset(tokens, [])
                        mm = mentions(text,it["question"],it["aliases"],[own])
                        row["mentions"] = [(owners[start] if start<len(owners) else len(tokens)-1,phrase,cat)
                                           for start,phrase,cat in mm if censor is None or
                                           (owners[start] if start<len(owners) else len(tokens)-1)<censor]
                        row.update(censor=censor,text=text,source_item=items[j]["qid"],modal=own,share={})
                    return rr
                rows.extend(ctx.cached(f"k6h_{block}_{it['qid']}_{variant}",compute))
                if hasattr(ctx,"time_left") and ctx.time_left()<30:
                    raise C.StepYield()
    hit = lambda category,cls,variant: {i:float(any(c==category for r in rr for _,_,c in r["mentions"]))
        for (i,),rr in K.by([r for r in rows if r["variant"]==variant and r["cls"]==cls],"item").items()}
    B = 30 if ctx.mock else C.N_BOOT
    gold = K.item_diff(hit("GOLD","CC","V1"),hit("GOLD","CC","V2"),B)
    own = K.item_diff(hit("OWN","CI","V1"),hit("OWN","CI","V2"),B)
    passed = K.fin(gold["ci"]["lo"]) and gold["ci"]["lo"]>=.05
    validation=_load_json(ctx,"lens_validate","lens_validation.json") or {}
    bridge_valid=bool(lens is not None and validation.get("G_K_J",{}).get("pass"))
    passed=passed and bridge_valid
    nrs = sum(name.startswith("RS") for name in populations)
    X,y,groups,random = K.claim_table(rows,None,nrs,qa=True) if nrs>=2 else ([],np.array([]),np.array([]),np.array([]))
    q12 = K.k6b_rule(X,y,groups,random,B=B) if passed and nrs>=(3 if ctx.mock else 20) else {
        "string":"not testable (QA lens fails its positive control)" if not passed else "not testable (insufficient matched random sets)",
        "p_iut":1.,"p_iut_pi":1.}
    stats = {"G_K23":"lens reads the gold answer (QA)" if passed else "QA H-link not testable (QA lens fails its positive control)",
             "gold":K._s(gold),"own":K._s(own),"p_own":K.pboot(own["draws"]) if len(own["draws"]) else 1.,
             "Q12":{k:v for k,v in q12.items() if not k.startswith("_")},"Q10":"pending family correction"}
    stats["scientific_evidence"]=not ctx.mock
    return {"rows":rows,"statistics":stats,"controls":controls,"counts":counts,"status":"measured",
            "G_K_J":bridge_valid,
            "instrument":instrument,"norm_l":norm_l,"matched_random_sets":nrs,
            "seed_instrument":"target J-lens" if lens is not None else "logit lens (QAT forward-only descriptive)",
            "audit":"lexical K1 CONFAB-QA parser; not a semantic factuality label", "frame":frame,"mock":ctx.mock}


def _probe(X, Y, groups, *, lambdas=(.1, 1., 10., 100.), folds=5):
    """Text-blocked nested ridge; return held-out errors for every target group.

    Y is [positions, target groups, dimensions]. Lambda is selected separately
    per group in three inner text folds. Centering/scaling uses training rows
    only. No fitted inverse is applied to model states.
    """
    import torch
    X,Y = torch.as_tensor(X,dtype=torch.float64),torch.as_tensor(Y,dtype=torch.float64)
    groups=np.asarray(groups); n,nt,dy=Y.shape
    if len(set(groups))<3:
        return None
    # Float64 CPU accumulators keep the same numerical convention as K's ridge.
    # All eight direction/lag targets share each factorization.
    def fit_predict(tr,te,lam):
        a,b=X[tr],Y[tr].flatten(1); mu=a.mean(0);sd=a.std(0,unbiased=False).clamp_min(1e-9)
        a=(a-mu)/sd;test=(X[te]-mu)/sd;ym=b.mean(0);b=b-ym
        penalty=float(lam)*len(a)
        if len(a)<a.shape[1]:
            pred=test@(a.T@torch.linalg.solve(a@a.T+penalty*torch.eye(len(a)),b))
        else:
            pred=test@torch.linalg.solve(a.T@a+penalty*torch.eye(a.shape[1]),a.T@b)
        return (pred+ym).reshape(-1,nt,dy),ym.reshape(nt,dy)
    unique=sorted(set(groups)); fmap={g:i%folds for i,g in enumerate(unique)}
    assignment=np.array([fmap[g] for g in groups]); sse=torch.full((n,nt),torch.nan,dtype=torch.float64)
    sst=torch.full_like(sse,torch.nan);chosen=[]
    for fold in sorted(set(assignment)):
        tr=assignment!=fold;te=~tr
        if tr.sum()<4 or te.sum()<1:continue
        train_groups=sorted(set(groups[tr]));inner={g:i%3 for i,g in enumerate(train_groups)}
        scores=torch.zeros(len(lambdas),nt);counts=0
        for innerfold in range(min(3,len(train_groups))):
            itr=tr & np.array([inner.get(g,-1)!=innerfold for g in groups])
            ite=tr & ~itr
            if itr.sum()<3 or ite.sum()<1:continue
            for j,lam in enumerate(lambdas):
                prediction,_=fit_predict(itr,ite,lam)
                scores[j]+=((prediction-Y[ite])**2).sum((0,2))
            counts+=1
        best=scores.argmin(0) if counts else torch.full((nt,),min(1,len(lambdas)-1),dtype=torch.long)
        errors=torch.empty((int(te.sum()),nt),dtype=torch.float64)
        baseline=None
        for j in best.unique().tolist():
            prediction,ym=fit_predict(tr,te,lambdas[j]); mask=best==j
            errors[:,mask]=((prediction-Y[te])**2).sum(-1)[:,mask]
            baseline=((Y[te]-ym)**2).sum(-1)
        sse[te]=errors;sst[te]=baseline
        chosen.append({"outer_fold":fold,"lambda":[lambdas[j] for j in best.tolist()]})
    valid=torch.isfinite(sse).all(-1)&torch.isfinite(sst).all(-1)
    if not valid.any():return None
    per={str(g):{"sse":sse[(groups==g)&valid.numpy()].sum(0).tolist(),
                 "sst":sst[(groups==g)&valid.numpy()].sum(0).tolist()}
         for g in unique if bool(((groups==g)&valid.numpy()).any())}
    r2=1-sse[valid].sum(0)/sst[valid].sum(0).clamp_min(1e-12)
    return {"r2":r2.tolist(),"per_text":per,"lambdas":chosen,
            "sse":sse.float(),"sst":sst.float(),"valid":valid}


def _pca(X, dimension, fit_count=None):
    import torch
    source=X[:fit_count] if fit_count is not None else X
    mean=source.float().mean(0);x=source.float()-mean
    _,s,v=torch.linalg.svd(x,full_matrices=False)
    rank=min(dimension,int((s>s.max().clamp_min(1e-9)*1e-7).sum()))
    if rank==0:
        return torch.zeros((X.shape[0],1)),torch.zeros((X.shape[1],1)),mean
    basis=v[:rank].T.contiguous()
    return (X.float()-mean)@basis,basis,mean


def _arrow_summary(probe, full, *, nboot=2000, seed=7301):
    """Paired text bootstrap of full-state-normalised arrow vectors/horizons."""
    if probe is None or full is None:
        return {"status":"undefined","reason":"too few text-blocked folds"}
    keys=sorted(set(probe["per_text"])&set(full["per_text"]))
    if not keys:return {"status":"undefined","reason":"no common texts"}
    def moments(x):
        return np.array([x["per_text"][k]["sse"] for k in keys]),np.array([x["per_text"][k]["sst"] for k in keys])
    e,t=moments(probe);ef,tf=moments(full)
    def ratio(ix):
        r=1-e[ix].sum(0)/np.maximum(t[ix].sum(0),1e-12)
        f=1-ef[ix].sum(0)/np.maximum(tf[ix].sum(0),1e-12)
        return np.divide(r,f,out=np.full_like(r,np.nan),where=f>0)
    estimate=ratio(np.arange(len(keys)));rng=np.random.default_rng(seed)
    draws=np.asarray([ratio(rng.integers(0,len(keys),len(keys))) for _ in range(nboot)])
    lower=np.nanquantile(draws,.05,axis=0) if np.isfinite(draws).any() else np.full(8,np.nan)
    upper=np.nanquantile(draws,.95,axis=0) if np.isfinite(draws).any() else np.full(8,np.nan)
    weights=np.array([1.,.5,.25,.125]);weights/=weights.sum()
    alpha=float(weights@(estimate[:4]-estimate[4:])) if np.isfinite(estimate).all() else float("nan")
    alpha_draws=(draws[:,:4]-draws[:,4:])@weights
    return {"a":estimate[:4].tolist(),"m":estimate[4:].tolist(),"lower":lower.tolist(),"upper":upper.tolist(),
            "alpha":alpha,"alpha_ci":np.nanquantile(alpha_draws,[.05,.95]).tolist() if np.isfinite(alpha_draws).any() else None,
            "_alpha_draws":alpha_draws,
            "horizon":max([k for k,bound in zip((1,2,4,8),lower[:4]) if np.isfinite(bound) and bound>.01],default=0),
            "raw_r2":probe["r2"],"full_state_r2":full["r2"],"n_texts":len(keys),"lambda_by_fold":probe["lambdas"]}


def _phase_surrogate(X,groups,rng):
    """Common phase per frequency across dimensions preserves cross spectra."""
    out=np.empty_like(X)
    for group in sorted(set(groups)):
        mask=groups==group;x=X[mask];spectrum=np.fft.rfft(x,axis=0)
        phase=np.exp(1j*rng.uniform(0,2*np.pi,len(spectrum)));phase[0]=1
        if len(x)%2==0:phase[-1]=1
        out[mask]=np.fft.irfft(spectrum*phase[:,None],n=len(x),axis=0)
    return out


def _channel_dynamics(named,groups,positions,*,mock=False):
    """Named-channel lead/lag and D_c non-stationarity, never causal evidence."""
    import torch
    lags=(1,2,4,8);bins=((16,31),(32,63),(64,127),(128,255))
    results,lead={},{}
    groups=np.asarray(groups);positions=np.asarray(positions);n=len(groups)
    index=np.arange(n)
    for name,X in named.items():
        x=np.asarray(X,dtype=np.float64)
        bins_out={}
        for lo,hi in bins:
            keep=(positions>=lo)&(positions<=hi)&(positions>=8)
            keep&=(index+8<n)&(index-8>=0)
            safe=np.clip(index+8,0,n-1);past=np.clip(index-8,0,n-1)
            keep&=(groups[safe]==groups)&(groups[past]==groups)
            ix=index[keep]
            if len(ix)<12 or len(set(groups[ix]))<3:
                bins_out[f"{lo}-{hi}"]={"status":"undefined","reason":"insufficient within-text positions"};continue
            def stat(values):
                targets=np.stack([values[ix+k] for k in lags]+[values[ix-k] for k in lags],axis=1)
                p=_probe(values[ix],targets,groups[ix],lambdas=(1.,10.))
                return np.array(p["r2"][:4])-np.array(p["r2"][4:]) if p else np.full(4,np.nan)
            observed=stat(x);rng=np.random.default_rng(7312);null=[]
            for _ in range(3 if mock else 200):
                null.append(stat(_phase_surrogate(x,groups,rng)))
            null=np.asarray(null)
            p=[float((1+np.sum(np.abs(null[:,j])>=abs(observed[j])))/(len(null)+1)) for j in range(4)]
            bins_out[f"{lo}-{hi}"]={"D":observed.tolist(),"p":p,"surrogate_mean":np.nanmean(null,axis=0).tolist(),
                                    "n_surrogates":len(null),"note":"non-stationarity diagnostic"}
        results[name]=bins_out
    names=sorted(named)
    keep=(positions>=16)&(index+8<n)&(index-8>=0)
    keep&=(groups[np.clip(index+8,0,n-1)]==groups)&(groups[np.clip(index-8,0,n-1)]==groups)
    ix=index[keep]
    for a in names:
        for b in names:
            if a==b or not len(ix):continue
            target=np.asarray(named[b]);y=np.stack([target[ix+k] for k in lags]+[target[ix-k] for k in lags],axis=1)
            p=_probe(np.asarray(named[a])[ix],y,groups[ix],lambdas=(1.,10.))
            if p:lead[a+"->"+b]={"L":(np.array(p["r2"][:4])-np.array(p["r2"][4:])).tolist(),
                                    "note":"cross-fitted directional predictability, not a causal edge"}
    return {"D":results,"lead_lag":lead}


def _mc_ceiling(ctx,E,sequences,projection):
    """Conditional future-embedding variance from keyed native continuations."""
    import torch
    lags=(1,2,4,8);samples=2 if ctx.mock else 32
    rows=[];embedding=E.model.get_input_embeddings().weight.detach()
    for i,seq in enumerate(sequences):
        positions=[16] if ctx.mock else [16,32,64,128]
        for position in positions:
            if position+8>=len(seq):continue
            def compute():
                draws=E.decode([seq[:position+1]]*samples,
                    keys=[ctx.key("k7_mc",i,position,j) for j in range(samples)],max_new=8,temperature=.7)
                record={"text":i,"position":position,"lags":{}}
                for lag in lags:
                    tokens=[r["ids"][lag-1] for r in draws if len(r["ids"])>=lag]
                    if len(tokens)<2:continue
                    values=embedding[tokens].float()@projection.to(embedding.device)
                    truth=(embedding[seq[position+lag]].float()@projection.to(embedding.device)).cpu()
                    record["lags"][lag]={"within_variance":float(values.var(0,unbiased=True).sum()),
                                        "true_embedding":truth,"n":len(tokens)}
                return record
            rows.append(ctx.cached(f"k7_ceiling_{i}_{position}",compute))
    result={}
    for lag in lags:
        rr=[r["lags"][lag] for r in rows if lag in r["lags"]]
        if len(rr)<2:
            result[lag]={"status":"undefined","reason":"too few complete sampled continuations"};continue
        total=float(torch.stack([r["true_embedding"] for r in rr]).var(0,unbiased=True).sum())
        result[lag]={"ceiling_r2":1-np.mean([r["within_variance"] for r in rr])/max(total,1e-12),
                     "positions":len(rr),"samples_per_position":samples,
                     "definition":"1 - E[conditional embedding variance] / held-out target embedding variance"}
    return {"summary":result,"rows":rows,"temperature":.7,"note":"Monte Carlo conditional-variance ceiling, descriptive"}


def _event_lock(probe, full, kept_indices, documents, *, mock=False):
    """Pre-error arrow changes against matched clean steps and event surrogates."""
    import torch
    weights=np.array([1.,.5,.25,.125]);weights/=weights.sum()
    denominator=np.asarray(full["r2"])
    if np.any(denominator<=0) or not np.isfinite(denominator).all():
        return {"status":"undefined","reason":"nonpositive full-state normalizer"}
    r2=1-probe["sse"].numpy()/np.maximum(probe["sst"].numpy(),1e-12)
    arrow=((r2[:,:4]/denominator[:4])-(r2[:,4:]/denominator[4:]))@weights
    value={int(index):float(a) for index,a in zip(kept_indices,arrow) if np.isfinite(a)}
    profiles=[];clean=[]
    for doc in documents:
        if not doc.get("event") or not doc["event"].get("fmt_ok"):continue
        ev=doc["event"];positions=ev.get("value_positions",[])
        if isinstance(positions,dict):positions=[positions[k] for k in sorted(positions,key=int)]
        steps=[]
        for tokens in positions:
            vals=[value.get(doc["offset"]+len(ev.get("prompt_ids",[]))+int(p)) for p in tokens]
            vals=[v for v in vals if v is not None]
            steps.append(float(np.mean(vals)) if vals else None)
        if ev.get("istar") is None:
            clean.append(steps)
        else:
            profiles.append({"item":ev["item"],"istar":int(ev["istar"]),"values":steps})
    usable=[p for p in profiles if p["istar"]>=3 and len(p["values"])>p["istar"] and
            all(x is not None for x in p["values"][p["istar"]-3:p["istar"]+1])]
    paired=[]
    for p in usable:
        t=p["istar"];control=[v[t-3:t+1] for v in clean if len(v)>t and all(x is not None for x in v[t-3:t+1])]
        if control:
            paired.append({**p,"profile":np.asarray(p["values"][t-3:t+1]),"control":np.mean(control,axis=0)})
    if len({p["item"] for p in paired})<3:
        return {"status":"undefined","reason":"fewer than three independent errors with position-matched clean windows",
                "error_texts":len(usable),"paired":len(paired)}
    delta=np.asarray([(p["profile"][-1]-p["profile"][0])-(p["control"][-1]-p["control"][0]) for p in paired])
    rng=np.random.default_rng(7319);null=[]
    for _ in range(3 if mock else 200):
        surrogate=[]
        for p in paired:
            candidates=[t for t in range(3,len(p["values"])) if all(x is not None for x in p["values"][t-3:t+1])]
            t=int(rng.choice(candidates));v=p["values"][t-3:t+1]
            surrogate.append((v[-1]-v[0])-(p["control"][-1]-p["control"][0]))
        null.append(float(np.mean(surrogate)))
    by_item={item:float(np.mean([d for d,p in zip(delta,paired) if p["item"]==item])) for item in sorted({p["item"] for p in paired})}
    values=np.asarray(list(by_item.values()));draws=values[rng.integers(0,len(values),(30 if mock else 2000,len(values)))].mean(1)
    return {"status":"measured","delta":float(delta.mean()),"ci":np.quantile(draws,[.05,.95]).tolist(),
            "p":float((1+sum(abs(x)>=abs(delta.mean()) for x in null))/(len(null)+1)),
            "mean_profile":np.mean([p["profile"] for p in paired],axis=0).tolist(),
            "mean_position_matched_control":np.mean([p["control"] for p in paired],axis=0).tolist(),
            "independent_items":len(by_item),"surrogates":len(null),"offsets":[-3,-2,-1,0]}


def atlas(ctx,E,hn,refs,lens):
    """K7a all native query heads plus target-derived subspaces and H/RS sets."""
    import torch
    import r1b_core as C
    from r1b_g_k import _raw_windows
    ntext,length=(5,48) if ctx.mock else (200,256)
    raw=ctx.data().get("wt",{}).get("ntext",[])
    sequences=_raw_windows(E,raw,ntext,length,ctx.mock)
    documents=[{"id":f"ntext:{i}","group":f"ntext:{i}","ids":seq,"exclude_before":16} for i,seq in enumerate(sequences)]
    long=_load_pt(ctx,"k_long","long.pt") or {}
    for row in long.get("rows",[])[:4 if ctx.mock else 64]:
        if not row.get("fmt_ok") or not row.get("tok"):continue
        ids=list(row.get("prompt_ids",[]))+list(row["tok"])
        documents.append({"id":f"long:{row['item']}:{row.get('sample',0)}","group":f"long:{row['item']}",
                          "ids":ids,"exclude_before":len(row.get("prompt_ids",[]))+16,"event":row})
    offset=0
    for doc in documents:
        doc["offset"]=offset;offset+=len(doc["ids"])
    groups=np.concatenate([np.full(len(doc["ids"]),doc["group"]) for doc in documents])
    sequence_ids=np.concatenate([np.full(len(doc["ids"]),i) for i,doc in enumerate(documents)])
    positions=np.concatenate([np.arange(len(doc["ids"]))-(len(doc["event"].get("prompt_ids",[])) if doc.get("event") else 0)
                              for doc in documents])
    valid=np.concatenate([np.arange(len(doc["ids"]))>=max(16,doc["exclude_before"]) for doc in documents])
    index=np.arange(offset);valid&=(index>=8)&(index+8<offset)
    valid&=(sequence_ids[np.clip(index-8,0,offset-1)]==sequence_ids)&(sequence_ids[np.clip(index+8,0,offset-1)]==sequence_ids)
    ix=index[valid];tokens=torch.tensor(sum((doc["ids"] for doc in documents),[]))
    embedding=E.model.get_input_embeddings().weight.detach()
    def embedding_fit():
        generator=torch.Generator().manual_seed(7)
        sample=torch.randperm(len(embedding),generator=generator)[:min(len(embedding),2000 if ctx.mock else 20000)].to(embedding.device)
        values=embedding[sample].float();mean=values.mean(0)
        _,s,v=torch.linalg.svd(values-mean,full_matrices=False)
        return {"projection":v[:min(64,values.shape[1])].T.cpu(),"mean":mean.cpu(),"singular_values":s[:64].cpu(),"sample":sample.cpu()}
    emb=ctx.cached("atlas_embedding",embedding_fit);projection=emb["projection"]
    with torch.no_grad():
        target=(embedding[tokens.to(embedding.device)].float()@projection.to(embedding.device)).cpu()
    lags=(1,2,4,8)
    Y=torch.stack([target[ix+k] for k in lags]+[target[ix-k] for k in lags],1)
    blocks=sorted(set(mapped_blocks(E,(6,10,14,18,22))+[len(E.layers)-1]))
    full,named,probes,bases={}, {}, {}, {}
    for block in blocks:
        def compute_block(block=block):
            states=torch.cat([E.hidden([doc["ids"]],[block])[0][block] for doc in documents])
            fit=_probe(states[ix],Y,groups[ix])
            channels={};basis={}
            pc,b,mean=_pca(states,16,ntext*length);channels[f"pc16_{block}"]=pc;basis[f"pc16_{block}"]=b
            source={"h":{block:states[:ntext*length]}}
            for name,(layer,b) in channel_bases(E,source,lens,hn).items():
                if name.startswith("pc16"):continue
                channels[name]=(states-states[:ntext*length].mean(0))@b;basis[name]=b
            # Dimension-matched isotropic null subspaces, fixed without labels.
            for rank in sorted({x.shape[1] for x in basis.values()}):
                for j in range(2 if ctx.mock else 19):
                    generator=torch.Generator().manual_seed(ctx.key("atlas_null",block,rank,j))
                    q=torch.linalg.qr(torch.randn(states.shape[1],rank,generator=generator))[0]
                    channels[f"rand{rank}_{j}_{block}"]=(states-states[:ntext*length].mean(0))@q
            return {"full":fit,"channels":channels,"bases":basis}
        result=ctx.cached(f"atlas_block_{block}",compute_block);full[block]=result["full"]
        named.update(result["channels"]);bases.update(result["bases"])
        if hasattr(ctx,"time_left") and ctx.time_left()<30:raise C.StepYield()
    named_blocks={name:block for block in blocks for name in named if name.endswith("_"+str(block)) or name in (f"Hsub{block}",f"Jws{block}")}
    populations=_populations(hn)
    if populations:
        flat={name:[E.layout.encode(l,j) for l,j in pairs] for name,pairs in populations.items()}
        def channels():
            accum={name:[] for name in flat}
            for doc in documents:
                read=E._collect(doc["ids"],flat,score=False)
                columns={v:i for i,v in enumerate(read["neuron_ids"])}
                for name,ids in flat.items():
                    values=read["cett"][:,[columns[x] for x in ids]]
                    if refs is not None:
                        values=(values-torch.as_tensor(refs["c_mu"])[ids])/torch.as_tensor(refs["c_sd"])[ids].clamp_min(1e-9)
                    accum[name].append(values.mean(-1,keepdim=True))
            return {name:torch.cat(v) for name,v in accum.items()}
        named.update(ctx.cached("atlas_population_channels",channels))
        named_blocks.update({name:len(E.layers)-1 for name in flat})
    output={};events={};B=30 if ctx.mock else 2000
    for name,X in named.items():
        probe=ctx.cached("atlas_probe_"+name,lambda:_probe(X[ix],Y,groups[ix]))
        probes[name]=probe;block=named_blocks[name]
        output[name]={"block":block,**_arrow_summary(probe,full[block],nboot=B)}
        if not name.startswith(("rand","RS")) and probe is not None:
            events[name]=_event_lock(probe,full[block],ix,documents,mock=ctx.mock)
    # Capture one layer at a time, so 31B does not retain every head's full
    # activation matrix in RAM. Completed layers checkpoint their summaries.
    for block,layer in enumerate(E.layers):
        reference=min(blocks,key=lambda b:abs(b-block))
        def head_layer(block=block,layer=layer,reference=reference):
            captured=[]
            handle=layer.self_attn.o_proj.register_forward_pre_hook(lambda m,args:captured.append(args[0][0].detach().float().cpu()))
            try:
                for doc in documents:E.hidden([doc["ids"]],[])
            finally:handle.remove()
            heads=torch.cat(captured);nh=heads.shape[-1]//layer.self_attn.head_dim
            heads=heads.reshape(len(heads),nh,layer.self_attn.head_dim);result={}
            for head in range(nh):
                X,_,_=_pca(heads[:,head],16,ntext*length)
                probe=_probe(X[ix],Y,groups[ix])
                result[f"head{block}.{head}"]={"block":block,"layer_type":E.cfg.layer_types[block],
                    "head_dim":layer.self_attn.head_dim,**_arrow_summary(probe,full[reference],nboot=B)}
            return result
        output.update(ctx.cached(f"atlas_heads_{block}",head_layer))
        if hasattr(ctx,"time_left") and ctx.time_left()<30:raise C.StepYield()
    dynamic={name:X.numpy() for name,X in named.items() if not name.startswith(("rand","RS"))}
    dynamics=ctx.cached("atlas_dynamics",lambda:_channel_dynamics(dynamic,sequence_ids,positions,mock=ctx.mock))
    ceiling=ctx.cached("atlas_ceilings",lambda:_mc_ceiling(ctx,E,sequences,projection))
    for channel in output.values():
        if "raw_r2" in channel:
            channel["mc_normalized_anticipation"]=[channel["raw_r2"][j]/ceiling["summary"][k]["ceiling_r2"]
                if ceiling["summary"].get(k,{}).get("ceiling_r2",0)>0 else None for j,k in enumerate(lags)]
    pvalues={"event:"+name:r["p"] for name,r in events.items() if r.get("status")=="measured"}
    for name,channel in output.items():
        if name.startswith("rand") or "_alpha_draws" not in channel:continue
        band=[r for other,r in output.items() if other.startswith("head") and other!=name and
              r["block"]==channel["block"] and "_alpha_draws" in r]
        if not band:continue
        draws=np.asarray(channel["_alpha_draws"])-np.mean([r["_alpha_draws"] for r in band],axis=0)
        draws=draws[np.isfinite(draws)]
        channel["layer_band_alpha"]=float(np.mean([r["alpha"] for r in band]))
        if len(draws):
            pvalues["band:"+name]=C.p_one_sided(draws,0,"greater")
            channel["beyond_band_ci"]=np.quantile(draws,[.05,.95]).tolist()
    for name,bins in dynamics["D"].items():
        for position,result in bins.items():
            for j,p in enumerate(result.get("p",[])):pvalues[f"D:{name}:{position}:{lags[j]}"]=p
    if pvalues:
        adjusted,rejected=C.bh(list(pvalues.values()),q=.10)
        fdr={name:{"p":p,"q":q,"reject":reject} for (name,p),q,reject in zip(pvalues.items(),adjusted,rejected)}
    else:fdr={}
    for name,channel in output.items():
        channel.pop("_alpha_draws",None)
        channel["string"]=(f"channel {name} anticipatory beyond its layer band" if
                           fdr.get("band:"+name,{}).get("reject") else None)
    changed=[name for name in events if fdr.get("event:"+name,{}).get("reject")]
    event_string=("arrow vector changes before errors in channels {"+", ".join(changed)+"}" if changed else
                  "no channel's arrow vector changes before errors" if any(r.get("status")=="measured" for r in events.values()) else
                  "event-locked arrow change not identifiable (no eligible error/control observations)")
    hierarchy=_load_pt(ctx,"k_hierarchy","hierarchy.pt") or _load_pt(ctx,"k_chain","hierarchy.pt") or {}
    alpha_int=_interventional_arrows(hierarchy)
    return {"status":"measured","channels":output,"full_state":{str(b):p["r2"] if p else None for b,p in full.items()},
            "event_locking":events,"dynamics":dynamics,"ceilings":ceiling["summary"],"BH_FDR_q10":fdr,
            "event_string":event_string,"interventional_arrows":alpha_int,
            "n_tokens":len(ix),"n_texts":len(documents),"n_event_texts":len(documents)-ntext,
            "head_count":sum(name.startswith("head") for name in output),"target_embedding_PCA":emb,
            "basis":bases,"note":"exploratory; never counted for or against H-CHAIN",
            "random_control":"19 same-dimensional isotropic subspaces per rank and block (two in mock)",
            "mock":ctx.mock}


def _interventional_arrows(hierarchy):
    """Read K9 gains and its paired random channels; no additional model runs."""
    grouped={}
    for row in hierarchy.get("rows",[]):
        for pair,value in row.get("g",{}).items():
            a,b=pair.split("->");grouped.setdefault((a,b),[]).append(float(value))
    gain={pair:float(np.mean(values)) for pair,values in grouped.items()}
    random_for=hierarchy.get("random_for",{})
    named=[name for name in hierarchy.get("channels",{}) if not name.startswith("rand_")]
    result={}
    for name in named:
        random=random_for.get(name,[])
        if not random:
            result[name]={"status":"undefined","reason":"K9 has no dimension-matched random gain baseline"};continue
        def flow(source,outward):
            return sum(value for (a,b),value in gain.items() if (a==source if outward else b==source) and
                       (b in named if outward else a in named))
        O,I=flow(name,True),flow(name,False)
        Or,Ir=np.mean([flow(r,True) for r in random]),np.mean([flow(r,False) for r in random])
        result[name]={"alpha_int":math.log(O/Or)-math.log(I/Ir) if min(O,I,Or,Ir)>0 else None,
                      "outflow":O,"inflow":I,"random_outflow":Or,"random_inflow":Ir,
                      "status":"measured" if min(O,I,Or,Ir)>0 else "undefined",
                      "reason":None if min(O,I,Or,Ir)>0 else "structural boundary or zero gain"}
    return result


def measure(ctx):
    """Independent supplemental stage; each branch keeps its own data gate."""
    import r1b_core as C
    E=ModelEngine(ctx)
    hn=_load_json(ctx,"k_prepare","hn.json") or {}
    refs=_load_pt(ctx,"k_prepare","references.pt")
    lens=_load_pt(ctx,"lens_fit","lens.pt")
    source=_load_pt(ctx,"k_chain","source_states.pt") or {}
    chains=_load_pt(ctx,"k_chain","chains.pt") or {}
    results={};payloads={}
    for name,fn in (("K6f",lambda:context_offsets(ctx,E,source,chains,hn,refs,lens)),
                    ("K6h",lambda:qa_lens(ctx,E,hn,refs,lens)),
                    ("K7a",lambda:atlas(ctx,E,hn,refs,lens))):
        try:result=fn()
        except C.NotRun as error:result={"status":"blocked","reason":str(error)}
        payloads[name]=result
        results[name]={k:v for k,v in result.items() if k not in ("rows","target_embedding_PCA","basis","frames")}
    from r1b_k_base import holm_family
    chain_analysis=_load_json(ctx,"k_chain_analyze","chain_analysis.json") or {}
    chain_stats=chain_analysis.get("statistics",{})
    f=results["K6f"].get("statistics",{});h=results["K6h"].get("statistics",{})
    g=chain_stats.get("K6g_provisional",{});d=chain_stats.get("K6d_provisional",{})
    q13=holm_family({"K6f":f.get("p",1.),"K6g":g.get("p",1.)})
    q10=holm_family({"K6d":d.get("p",1.),"K6h":h.get("p_own",1.)})
    f_string=("context-length effect on lens break-off" if q13["K6f"]["reject"] else
              "no context-length effect" if f.get("_eq") else
              f.get("string") if f.get("string") not in (None,"pending Q13 family correction") else "undecided")
    g_string=("content effect (in-lens snowballing)" if q13["K6g"]["reject"] else
              "no content effect" if g.get("_eq") else "undecided")
    h_string=("lens reads the model's own wrong answer (QA)" if q10["K6h"]["reject"] and
              h.get("G_K23","").startswith("lens reads") else "not shown")
    if f:f.update(string="MOCK: no scientific verdict" if ctx.mock else f_string,Q13=q13["K6f"])
    if h:h.update(Q10={**q10["K6h"],"string":"MOCK: no scientific verdict" if ctx.mock else h_string})
    families={"Q13":{"members":q13,"strings":{"K6f":f_string,"K6g":g_string}},
              "Q10":{"members":q10,"strings":{"K6d":"lens reads the model's own wrong intermediates" if q10["K6d"]["reject"] else "not shown", "K6h":h_string}},
              "scientific_evidence":not ctx.mock}
    for name,payload in payloads.items():ctx.save_pt(name.lower()+".pt",payload)
    status="done" if all(r.get("status")=="measured" for r in results.values()) else "partial"
    ctx.write_json("extra_analysis.json",_clean_json({"status":status,"branches":results,"families":families,"mock":ctx.mock}))
    return {"status":status,"outputs":["k6f.pt","k6h.pt","k7a.pt","extra_analysis.json"],
            "coverage":["K6f","K6h","K7a"]}
