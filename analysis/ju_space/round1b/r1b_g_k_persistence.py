"""Gemma K6e/K8b: released interventions, CRN twins, carry and bistability.

Native causal prefix recomputation reconstructs historical edited KV exactly:
only fixed past positions remain edited after release. TOKEN replay removes
those edits, while KV replay forces clean tokens and retains historical edits.
No Qwen cache layout, generated output, labels or fitted coordinates are used.
"""
from __future__ import annotations

import contextlib
import math
from pathlib import Path

import numpy as np

import r1b_core as RC
import r1b_g_adapter as GA
import r1b_g_k as K
import r1b_g_k_chain as C
import r1b_k_base as KB
import r1b_k_labels as KL

SEG=8


def _optional(ctx,stage,name):
    p=Path(ctx.artifact(stage,name))
    return K._load(ctx,stage,name) if p.exists() else None


def _environment(ctx):
    E=K._engine(ctx);hn=K._load(ctx,"k_prepare","hn.json");refs=K._load(ctx,"k_prepare","references.pt")
    detectors=K._load(ctx,"k_prepare","detectors.pt")
    verdict=_optional(ctx,"k_analyze","k3_verdicts.json") or {}
    arith=verdict.get("K3a",{}).get("arith",{}).get("string","")
    har=verdict.get("K3a",{}).get("har",{}).get("string","")
    if hn["H_op_name"]=="H_ar" or arith.startswith("H-neurons replicated"):
        ha=hn["H_op"];label=None
    elif har=="H_ar replicated (arithmetic in-domain)" and hn["H_ar"]:
        ha=hn["H_ar"];label="domain-specific H-set"
    else:ha=hn["H_op"];label="H not valid in arithmetic"
    name="H_ar" if ha==hn["H_ar"] and ha else "H_op"
    populations={"H":ha,**{f"RS{i}":s for i,s in enumerate(hn.get("RS20",hn["RS39"][:20]))},
                 **{f"MC{i}":s for i,s in enumerate(hn["RS39mc"][:3 if ctx.mock else 20])}}
    return E,hn,refs,detectors,populations,name,label


def _edit(E,pairs,op,value,positions=None):
    return E.intervene(K._flat(E.layout,pairs),op,value,positions=positions) if pairs else contextlib.nullcontext()


def _read(E,seq,populations,refs,detector=None,setup=None):
    row=C._read_channels(E,seq,populations,refs,setup=setup)
    if detector and populations.get("H"):
        positions={flat:i for i,flat in enumerate(row["neuron_ids"])}
        cols=[positions[k] for k in K._flat(E.layout,populations["H"])]
        row["sH"]=row["cett"][:,cols].double().numpy()@detector["beta"]+detector["b"]
    else:row["sH"]=None
    return row


def cache_equivalence(E,seq,pairs,alpha,positions,patch=None):
    """Compare historical-edit replay with native incremental KV, including sharing."""
    import torch
    from transformers import DynamicCache
    ids=torch.tensor([seq],device=E.device)
    def patcher(offset,length):
        if patch is None or not offset<=patch[2]<offset+length:return contextlib.nullcontext()
        return E.patch_state(patch[0],patch[1],patch[2]-offset)
    kernel=RC.sdpa_efficient(E.device) if len(seq)>2048 else contextlib.nullcontext()
    with kernel,torch.no_grad():
        with _edit(E,pairs,"scale",alpha,positions),patcher(0,len(seq)):
            full=GA.text_model(E.model)(input_ids=ids,use_cache=False).last_hidden_state
        cache=DynamicCache(config=E.cfg);pieces=[]
        for start in range(0,len(seq),16):
            stop=min(start+16,len(seq));local=[p-start for p in positions if start<=p<stop]
            with _edit(E,pairs,"scale",alpha,local),patcher(start,stop-start):
                row=GA.text_model(E.model)(input_ids=ids[:,start:stop],past_key_values=cache,use_cache=True,
                    attention_mask=torch.ones((1,stop),device=E.device,dtype=torch.long),
                    position_ids=torch.arange(start,stop,device=E.device)[None])
            cache=row.past_key_values;pieces.append(row.last_hidden_state)
        cached=torch.cat(pieces,1)
        # Check every hidden and vocabulary position in bounded chunks. G-K20's
        # logit tolerance is deliberately unchanged even for fp16 QAT.
        max_error=0.;max_score=1e-12;flips=0
        probes=np.linspace(0,len(seq)-1,100,dtype=int)
        for start in range(0,len(seq),32):
            stop=min(start+32,len(seq))
            a=GA.unembed(E.model,full[:,start:stop],normalized=True)[0].float()
            b=GA.unembed(E.model,cached[:,start:stop],normalized=True)[0].float()
            max_error=max(max_error,float((a-b).abs().max()));max_score=max(max_score,float(a.abs().max()))
            for seed,position in enumerate(probes):
                if not start<=position<stop:continue
                ga=torch.Generator(device=E.device).manual_seed(7100+seed)
                gb=torch.Generator(device=E.device).manual_seed(7100+seed)
                flips+=E.choose(a[position-start],ga,.7)!=E.choose(b[position-start],gb,.7)
        rel=max_error/max_score
    return dict(relative_max_logits=rel,hidden_max_abs=float((full-cached).abs().max()),
                gumbel_flip_rate=flips/100,pass_=rel<1e-4 and flips<=1,
                tokens=len(seq),historical_positions=list(positions))


def boundary(ctx,E=None):
    """K8 A-bnd: exactly four samples on 1,200 items, no substitute questions."""
    E=E or K._engine(ctx)
    rows=K._need_rows(ctx,ctx.data(),"ach_short","A-bnd",8 if ctx.mock else 1200)
    out=[]
    for it in rows:
        def run(it=it):
            samples=E.decode([E.encode(it["prompt"])]*4,
                keys=[ctx.key("g_k8_boundary",it["id"],s) for s in range(4)],temperature=.7,max_new=4 if ctx.mock else 12)
            ok=[KL.parse_number(x["text"])==it["A"] for x in samples]
            return dict(id=it["id"],item=it,n_ok=sum(ok),texts=[x["text"] for x in samples],
                        tokens=[x["ids"] for x in samples],ok=ok,mock=ctx.mock)
        out.append(K._cached(ctx,"boundary_"+str(K._hash(it["id"])),run))
    selected=[r["id"] for r in out if 1<=r["n_ok"]<=3][:10 if ctx.mock else 400]
    artifact=dict(rows=out,boundary=selected,label="ACH arm underpowered" if len(selected)<(2 if ctx.mock else 250) else None,mock=ctx.mock)
    ctx.save_pt("boundary.pt",artifact)
    return artifact


def _templates(ids,n,seed):
    if not ids:raise RC.NotRun("no boundary items for persistence templates")
    rng=np.random.default_rng(seed)
    return [[ids[int(k)] for k in rng.choice(len(ids),8,replace=len(ids)<8)] for _ in range(n)]


def _tqa_templates(rows,n,mock):
    """512 CC questions, 192 sessions: each question in at most three sessions."""
    ids=[r["qid"] for r in rows]
    if not ids:raise RC.NotRun("K8b lacks independently allocated S_sess CC questions")
    if not mock and len(ids)<512:raise RC.NotRun("K8b TQA requires 512 S_sess CC questions for the <=3-session cap")
    rng=np.random.default_rng(271);stream=[]
    for _ in range(3):stream.extend(rng.permutation(ids).tolist())
    if mock:stream=(stream*(8*n//max(len(stream),1)+1))[:8*n]
    if len(stream)<8*n:raise RC.NotRun("K8b TQA question/session capacity insufficient")
    return [stream[i*8:(i+1)*8] for i in range(n)]


def _append_turn(E,history,text):
    ids=E.encode(text)
    if not history:return ids
    eos,stop=GA.stop_tokens(E.tok,E.model)
    vocab=E.tok.get_vocab();end=vocab.get("<turn|>",eos[0])
    if ids and ids[0]==getattr(E.tok,"bos_token_id",None):ids=ids[1:]
    return history+[int(end)]+E.encode("\n",chat=False)+ids


def session(ctx,E,template,items,family,pairs,alpha,route,populations,refs,detector,forced_first=None):
    """Eight turns, first-answer-only edit; no intervention on later positions."""
    import torch
    history=[];turns=[];historic=[];_,stops=GA.stop_tokens(E.tok,E.model)
    for turn,ident in enumerate(template["items"]):
        item=items[ident]
        prompt=item["prompt"] if family=="ACH" else "Question: "+item["question"]
        prefix=_append_turn(E,history,prompt)
        if family=="TQA":prefix+=E.encode("The answer is",chat=False)
        start=len(prefix);answer=[];ent=[]
        maxnew=4 if ctx.mock else 30 if family=="ACH" else 24
        generator=torch.Generator(device=E.device).manual_seed(ctx.key("g_k8_session",family,template["template"],template["seed"],turn))
        for step in range(maxnew):
            active=list(range(start,start+maxnew)) if turn==0 and route in ("FULL","KV") else historic
            setup=lambda:_edit(E,pairs,"scale",alpha,active)
            logits=E.next_logits(prefix+answer,setup=setup)
            lp=logits.log_softmax(-1);ent.append(float(-(lp.exp()*lp).sum()))
            chosen=E.choose(logits,generator,.7)  # Consume one CRN draw even when teacher forcing.
            if turn==0 and forced_first is not None:
                if step>=len(forced_first):break
                chosen=int(forced_first[step])
            if chosen in stops:break
            answer.append(chosen)
        if turn==0 and route in ("FULL","KV"):historic=list(range(start,start+len(answer)))
        setup=lambda:_edit(E,pairs,"scale",alpha,historic)
        seq=prefix+answer
        if not answer:raise RC.NotRun("K8b generated an empty answer; detector activity undefined")
        read=_read(E,seq,populations,refs,detector,setup)
        text=E.tok.decode(answer)
        error=K._label(ctx,item,text,True)!="correct" if family=="ACH" else K._label(ctx,item,text)!="correct"
        turns.append(dict(tok=answer,text=text,err=float(error),aH=float(read["aX"]["H"][start:].mean()),
                          sH=float(np.mean(read["sH"][start:])) if read["sH"] is not None else None,
                          ent=float(np.mean(ent[:len(answer)])),prompt_ids=prefix))
        history=seq
    return dict(sid=template["sid"],template=template["template"],seed=template["seed"],turns=turns,
                route=route,historical_edit_positions=historic)


def model_sessions(ctx,env):
    E,hn,refs,detectors,populations,name,label=env
    if not populations["H"] or name not in detectors:raise RC.NotRun("K8b requires identified arithmetic H and its independent detector")
    B=boundary(ctx,E);bnd=B["boundary"]
    if not bnd:raise RC.NotRun("K8b no A-bnd item with 1-3 correct samples out of four")
    items={r["id"]:r["item"] for r in B["rows"]};HA=populations["H"]
    texts=[r["text"] if isinstance(r,dict) else r for r in ctx.data().get("ref_k16",[])]
    if not texts:raise RC.NotRun("K8b damage controls require raw REF-K16")
    damage_H=K._cached(ctx,"session_damage_H",lambda:E.damage(texts,setup=lambda:_edit(E,HA,"scale",3.))["damage"])
    matched=K._load(ctx,"k_prepare","prepared.pt")["matching"]["RS39"]
    gains={}
    for index,pairs in enumerate(hn.get("RS19",hn["RS39"][:19])):
        if HA==hn["H_op"] and index<len(matched):gains[index]=matched[index].get("gain3",3.)
        else:
            calgain=K._cached(ctx,f"session_gain_RS{index}",lambda pairs=pairs:RC.calibrate(
                lambda alpha:E.damage(texts,setup=lambda:_edit(E,pairs,"scale",alpha))["damage"],3.,damage_H,form="gain"))
            gains[index]=calgain["f"]
    def runset(block,arm,templates,family="ACH",pairs=None,alpha=1.,route=None,forced=None,tq=None):
        out=[]
        for spec in templates:
            key=K._hash((block,arm,spec["sid"]))
            out.append(K._cached(ctx,f"session_{key}",lambda spec=spec:session(ctx,E,spec,tq or items,family,
                pairs or [],alpha,route,populations,refs,detectors[name],None if forced is None else forced[spec["sid"]])))
        return out
    cal=[dict(sid=f"cal{i}",template=3000+i,seed=0,items=t) for i,t in enumerate(_templates(bnd,2 if ctx.mock else 32,72))]
    clean=runset("cal","clean",cal);pushed=runset("cal","H",cal,pairs=HA,alpha=3.,route="FULL")
    sample=pushed[0];seq=sample["turns"][-1]["prompt_ids"]+sample["turns"][-1]["tok"]
    parity=K._cached(ctx,"session_cache_parity",lambda:cache_equivalence(E,seq,HA,3.,sample["historical_edit_positions"]))
    if not parity["pass_"]:raise RC.NotRun("K8b historical-edit replay/native cache parity failed")
    den=np.array([sum(p["turns"][i]["sH"]-c["turns"][i]["sH"] for i in range(1,7)) for c,p in zip(clean,pushed)])
    projected=float(den.mean()/(den.std(ddof=1)/math.sqrt(256))) if len(den)>1 and den.std()>0 else 0.
    nt=(4 if ctx.mock else 256) if projected>=7 else (6 if ctx.mock else 384)
    main=[dict(sid=f"m{i}",template=i,seed=0,items=t) for i,t in enumerate(_templates(bnd,nt,71))]
    bist=[dict(sid=f"b{i}_{s}",template=1000+i,seed=s,items=t) for i,t in enumerate(_templates(bnd,2 if ctx.mock else 24,171)) for s in range(3 if ctx.mock else 16)]
    arms={"clean":([],1.,None),**{f"H_a{a}_FULL":(HA,a,"FULL") for a in (3.,2.,1.5,0.)},
          **{f"H_a{a}_{route}":(HA,a,route) for a in (0.,3.) for route in ("TOKEN","KV")},
          **{f"RS{k}_a3_FULL":(s,gains[k],"FULL") for k,s in enumerate(hn.get("RS19",hn["RS39"][:19]))}}
    result=[];bank={}
    for block,templates in (("main",main),("bist",bist)):
        names=list(arms) if block=="main" else ["clean","H_a3.0_FULL"]+[n for n in arms if n.startswith("RS")]
        for arm in names:
            pairs,alpha,route=arms[arm];forced=None
            if route in ("TOKEN","KV"):
                source=f"H_a{alpha}_FULL" if route=="TOKEN" else "clean"
                forced={s["sid"]:s["turns"][0]["tok"] for s in bank[block,source]}
            selected=templates
            if block=="main" and arm.startswith("RS"):selected=templates[:4 if ctx.mock else 256]
            rows=runset(block,arm,selected,pairs=pairs,alpha=alpha,route=route,forced=forced)
            bank[block,arm]=rows;result.append(dict(block=block,arm=arm,actual_gain=alpha,sessions=rows))
    missing=[]
    labs=K._load(ctx,"k_prepare","lab.json")
    tqrows=[r for r in labs["items"] if r["split"]=="S_sess" and r["cls"]=="CC"]
    try:
        templates=[dict(sid=f"q{i}",template=2000+i,seed=0,items=t) for i,t in enumerate(_tqa_templates(tqrows,4 if ctx.mock else 192,ctx.mock))]
        tq={r["qid"]:r for r in tqrows}
        for arm,pairs in [("clean",[]),("Htq_a3.0_FULL",hn["H_tq"])]+[(n,s) for n,(s,a,r) in arms.items() if n.startswith("RS")]:
            if arm!="clean" and not pairs:raise RC.NotRun("K8b TQA H_tq is empty")
            alpha=arms[arm][1] if arm.startswith("RS") else 3.
            rows=runset("tqa",arm,templates[:3 if ctx.mock else 128] if arm.startswith("RS") else templates,
                        "TQA",pairs,alpha,"FULL" if pairs else None,tq=tq)
            result.append(dict(block="tqa",arm=arm,actual_gain=alpha,sessions=rows))
    except RC.NotRun as exc:missing.append(str(exc))
    obj=dict(rows=result,calibration=dict(z_projected=projected,n_templates=nt,doubled=projected<7),native_cache_parity=parity,
             damage_H_a3=damage_H,G_K12="damage-limited" if damage_H>.15 else None,control_gains=gains,
             H_A_label=label,boundary_label=B["label"],missing=missing,mock=ctx.mock,
             requested_counts=dict(main=nt,bist_templates=2 if ctx.mock else 24,bist_seeds=3 if ctx.mock else 16,tqa=4 if ctx.mock else 192),
             replay="fixed historical-position native intervention replay reconstructs retained KV; later positions unedited")
    ctx.save_pt("k8b_data.pt",obj)
    return obj


@contextlib.contextmanager
def _lens_setup(E,block,state,placeholder,pairs,op,value,positions):
    with E.patch_state(block,state,placeholder),_edit(E,pairs,op,value,positions):yield


def chain(ctx,E,unit,cell,frame,populations,refs,*,op=None,T=.7,forced=None,keep_first=False):
    """24 recorded token states, seed included, with absolute-position CRN."""
    import torch
    tokens=[int(cell["seed"])];records={name:[] for name in populations if populations[name]};first=None
    generator=torch.Generator(device=E.device).manual_seed(ctx.key("g_k6e",*unit,f"T{T}"))
    start=len(frame);special=list(E.tok.all_special_ids)
    for position in range(24):
        seq=frame+tokens
        pairs=op["pairs"] if op else [];kind=op["kind"] if op else "scale";value=op.get("value",1.) if op else 1.
        selected=list(range(start,start+SEG))
        real_setup=lambda:_lens_setup(E,cell["block"],cell["state"],len(frame)-1,pairs,kind,value,selected)
        prior_pairs=pairs if op and op["streams"]=="both" else []
        prior_setup=lambda:_lens_setup(E,cell["block"],cell["prior"],len(frame)-1,prior_pairs,kind,value,selected)
        real=E.next_logits(seq,setup=real_setup).log_softmax(-1)
        prior=E.next_logits(seq,setup=prior_setup).log_softmax(-1)
        score=real-prior;score[real<real.max()+math.log(.1)]=-torch.inf;score[special]=-torch.inf
        if position==0 and keep_first:first=dict(lr=real.cpu(),lp=prior.cpu(),plaus=torch.isfinite(score).cpu())
        reading=_read(E,seq,populations,refs,setup=real_setup)
        for name in records:records[name].append(reading["aX"][name][-1])
        if position<23:
            choice=E.choose(score,generator,T)
            if forced is not None and position<len(forced):choice=int(forced[position])
            tokens.append(choice)
    scored=C.score_tokens(E.tok,tokens,cell["item"],W=cell.get("W",[]),OWN=cell.get("OWN",[]))
    return dict(unit=tuple(unit),tok=tokens,aX={name:torch.stack(v) for name,v in records.items()},first=first,
                stratum=cell["stratum"],**scored)


def _css(a,b):
    mask=b["plaus"]
    numerator=((a["lr"]-a["lp"])-(b["lr"]-b["lp"])).masked_fill(~mask,0).norm()
    denominator=(a["lr"]-b["lr"]).masked_fill(~mask,0).norm().clamp_min(1e-12)
    return float(numerator/denominator)


def _filler(ctx,E,length,instance):
    pool=(ctx.data().get("wt") or {}).get("coherent",[])
    text=" ".join((x["text"] if isinstance(x,dict) else x) for x in pool)
    import re
    sentences=[s for s in re.split(r"(?<=[.!?])\s+",text) if not any(c.isdigit() for c in s)]
    ids=E.encode(" ".join(sentences),chat=False)
    offset=instance*4096
    if ctx.mock and ids:ids=ids*(1+(offset+length)//len(ids))
    if len(ids)<offset+length:raise RC.NotRun("K6e-d requires sufficient pinned digit-free filler for all eight instances")
    return ids[offset:offset+length]


def lens_persistence(ctx,env):
    import torch
    from r1b_g_k_readouts import _instrument
    E,hn,refs,detectors,populations,name,label=env
    artifact=K._load(ctx,"k_chain","chains.pt");source_art=K._load(ctx,"k_chain","source_states.pt")
    source=source_art["source"]["t0"];bstar=artifact["verifier"].get("b_star")
    if bstar is None:raise RC.NotRun("K6e b* is not selected by K4 V1-minus-V2")
    lens,instrument=_instrument(ctx,E)
    correct=[i for i in source_art["correct"] if not source_art["items"][i].get("cal")][:3 if ctx.mock else 128]
    wrong=[i for i in source_art["wrong"] if not source_art["items"][i].get("cal")][:2 if ctx.mock else 64]
    if not correct and not wrong:raise RC.NotRun("K6e no independently gated K4 main items")
    missing=[]
    if len(correct)<(3 if ctx.mock else 128) or len(wrong)<(2 if ctx.mock else 64):
        missing.append("K6e insufficient K4 correct/wrong yield; coverage counts remain explicit")
    HA=populations["H"]
    if not HA:raise RC.NotRun("K6e H_A is empty")
    frame=artifact["metadata"]["frame_ids"]
    gates=artifact["gating"]["gate"];states=source["h"][bstar]
    nseed=2 if ctx.mock else 4
    def make_cell(ident,state,prior,item,stratum,gate=None):
        seeds=C._seeds(E,state,bstar,lens,nseed)
        return [dict(block=bstar,state=state,prior=prior,item=item,stratum=stratum,seed=s,
                     W=(gate or {}).get("W",[]),OWN=(gate or {}).get("own",[])) for s in seeds]
    cells={}
    for ident in correct+wrong:
        ix=source["ids"].index(ident);state=states[ix]
        if len(states)<2:raise RC.NotRun("K6e leave-one-out prior requires two source items")
        prior=(states.sum(0)-state)/(len(states)-1)
        cells[ident]=make_cell(ident,state,prior,source_art["items"][ident],"correct" if ident in correct else "wrong",gates[ident])
    texts=ctx.data().get("ref_k16",[])
    if not texts or (not ctx.mock and len(texts)!=16):raise RC.NotRun("K6e NSCALE_DN requires pinned REF-K16")
    texts=[r["text"] if isinstance(r,dict) else r for r in texts]
    gains={}
    for pop in ["H"]+[n for n in populations if n.startswith("RS")]:
        pairs=populations[pop]
        gains[pop]=K._cached(ctx,"persistence_gain_"+pop,lambda pairs=pairs:RC.calibrate(
            lambda alpha:E.damage(texts,setup=lambda:_edit(E,pairs,"scale",alpha))["damage"],3.,RC.DELTA2,form="gain"))
    if not gains["H"]["in_band"]:missing.append("H NSCALE_DN calibration out of the delta2 band; formal persistence verdict gated")
    def op(pop,kind="scale",streams="expert"):
        pairs=populations[pop]
        value=gains[pop]["f"] if kind=="scale" else refs["z_mu"][K._flat(E.layout,pairs)]
        return dict(pairs=pairs,kind=kind,value=value,streams=streams)
    # Proper A-cal states are independent of every K4 main item.
    calitems=[C.normalize_item(r) for r in ctx.data().get("ach_short",[]) if r.get("cal")][:2 if ctx.mock else 32]
    if len(calitems)<(2 if ctx.mock else 32):raise RC.NotRun("K6e G-K16 needs 32 disjoint A-cal items (2 in mock)")
    if {r["id"] for r in calitems}&set(cells):raise ValueError("A-cal/K6e main item overlap")
    calex=[]
    for item in calitems:
        prompt=E.encode(item["prompt"]);state=E.hidden([prompt],[bstar])[0][bstar][-1]
        cs=make_cell(item["id"],state,states.mean(0),item,"calibration")
        for rank,cell in enumerate(cs):
            for key in range(2):
                unit=(item["id"],rank,key)
                def observe(unit=unit,cell=cell):
                    twin=chain(ctx,E,unit,cell,frame,populations,refs)
                    push=chain(ctx,E,unit,cell,frame,populations,refs,op=op("H"))
                    tok=chain(ctx,E,unit,cell,frame,populations,refs,forced=push["tok"][1:SEG])
                    return float((tok["aX"]["H"][:SEG]-twin["aX"]["H"][:SEG]).mean())
                calex.append(K._cached(ctx,"persistence_cal_"+str(K._hash(unit)),observe))
    calex=np.asarray(calex)
    projected=float(calex.mean()/(calex.std(ddof=1)/math.sqrt(1536))) if len(calex)>1 and calex.std()>0 else 0.
    keys=2 if projected>=7 else 4
    units=[(i,r,k) for i in cells for r in range(nseed) for k in range(keys)]
    twins={};conditions=[]
    for T in (.7,1.):
        twins[T]=[K._cached(ctx,"persistence_twin_"+str(K._hash((u,T))),lambda u=u,T=T:chain(ctx,E,u,cells[u[0]][u[1]],frame,populations,refs,T=T,keep_first=True)) for u in units]
    compare=units[::-1][:6 if ctx.mock else 256]
    byunit={r["unit"]:r for r in twins[.7]}
    identical=np.mean([chain(ctx,E,u,cells[u[0]][u[1]],frame,populations,refs)["tok"]==byunit[u]["tok"] for u in compare])
    jobs=[("PUSH_H","H","scale","expert",.7),("CLAMP_H","H","mean","both",.7),
          ("PUSH2_H","H","scale","both",.7),("PUSHT1_H","H","scale","expert",1.)]
    jobs += [("PUSH_"+n,n,"scale","expert",.7) for n in populations if n.startswith("RS")]
    jobs += [("CLAMP_"+n,n,"mean","both",.7) for n in populations if n.startswith("MC")]
    for cond,pop,kind,streams,T in jobs:
        pushed=[];token=[];css=[];tw={r["unit"]:r for r in twins[T]}
        for u in units:
            cell=cells[u[0]][u[1]]
            def observe(u=u,cell=cell):
                pushed=chain(ctx,E,u,cell,frame,populations,refs,op=op(pop,kind,streams),T=T,keep_first=streams=="both")
                tok=chain(ctx,E,u,cell,frame,populations,refs,T=T,forced=pushed["tok"][1:SEG])
                return pushed,tok
            pu,tk=K._cached(ctx,"persistence_condition_"+str(K._hash((cond,u))),observe)
            pushed.append(pu);token.append(tk)
            if streams=="both":css.append(_css(pu["first"],tw[u]["first"]))
        conditions.append(dict(cond=cond,X=pop,T=T,streams=streams,pushed=pushed,tok=token,css=css))
    offsets=[];offset_parity=None
    try:
        for u in units:
            inst=K._hash(u[0])%8;offset=64 if ctx.mock else 2048
            longframe=_filler(ctx,E,offset,inst)+frame;cell=cells[u[0]][u[1]]
            if offset_parity is None:
                seq=longframe+[cell["seed"]]+[cell["seed"]]*23
                offset_parity=K._cached(ctx,"lens_offset_cache_parity",lambda:cache_equivalence(E,seq,HA,gains["H"]["f"],
                    list(range(len(longframe),len(longframe)+SEG)),patch=(bstar,cell["state"],len(longframe)-1)))
                if not offset_parity["pass_"]:raise RC.NotRun("K6e-d native long-context cache parity failed (G-K20)")
            def observe(u=u,cell=cell,longframe=longframe):
                twin=chain(ctx,E,u,cell,longframe,populations,refs)
                pushed=chain(ctx,E,u,cell,longframe,populations,refs,op=op("H"))
                tok=chain(ctx,E,u,cell,longframe,populations,refs,forced=pushed["tok"][1:SEG])
                return dict(unit=u,twin=twin,tok=tok,instance=inst)
            offsets.append(K._cached(ctx,"persistence_offset_"+str(K._hash(u)),observe))
    except RC.NotRun as exc:missing.append(str(exc))
    # All-position calibration is also reported on the target's own REF-U32.
    r16=refs.get("ref_r16",[])
    refitems=[(r["ids"],r["positions"]) for r in r16]
    for text in texts:
        ids=E.encode(text,chat=False);refitems.append((ids,list(range(len(ids)-1))))
    from r1b_g_l_erasure import Damage
    damage=Damage(E.model,RC.RefSet(refitems,"G own REF-U32"))
    u32=damage(lambda:_edit(E,HA,"scale",gains["H"]["f"])) if len(r16)>=(2 if ctx.mock else 16) else None
    obj=dict(twins=twins,rows=conditions,offset=offsets,gains=gains,ref_u32=u32,G_K20=offset_parity,
             calibration=dict(z_projected=projected,n_keys=keys,calibration_ids=[r["id"] for r in calitems]),
             G_K21=dict(identical=float(identical),label=None if identical>=.97 else "numerical divergence"),
             counts=dict(correct=len(correct),wrong=len(wrong),seeds=nseed,keys=keys),instrument=instrument,
             H_A_label=label,missing=missing,mock=ctx.mock,segments="0-7,8-15,16-23; position0=seed")
    ctx.save_pt("k6e_data.pt",obj)
    return obj


def _seg(run,name,segment):
    return float(run["aX"][name][segment*SEG:(segment+1)*SEG].float().mean())


def _ratio(num,den,items,B):
    num,den=np.asarray(num,float),np.asarray(den,float);items=np.asarray(items)
    valid=np.isfinite(num)&np.isfinite(den)
    if valid.sum()<2:return dict(defined=False,code=RC.coded("NaN:too_few_finite_units"))
    r=KB.ratio_carry(num[valid],den[valid],items[valid],B=B)
    r["half_life"]=math.log(2)/(-math.log(r["K"])) if KB.fin(r["K"]) and 0<r["K"]<1 else None
    return r


def _bistability(z,items,random_samples,B,needed):
    z=np.asarray(z,float);items=np.asarray(items);keep=np.isfinite(z);z=z[keep];items=items[keep]
    mix=KB.mixture_lr(z,items,B=B);tail=KB.robust_tail_share(z,items,B=B)
    lr=[]
    for zr,ir in random_samples:
        zr=np.asarray(zr,float);ir=np.asarray(ir);ok=np.isfinite(zr)
        mr=KB.mixture_lr(zr[ok],ir[ok],B=0)
        if KB.fin(mr.get("LR")):lr.append(math.log1p(max(0,mr["LR"])))
    locked=set(items[z>3]);relaxed=set(items[np.abs(z)<1])
    pi=RC.p_pi(math.log1p(max(mix["LR"],0)),lr) if len(lr)>=needed and KB.fin(mix.get("LR")) else 1.
    pe=mix.get("p",1.)
    return dict(mix=mix,tail=tail,lr_R=lr,n_random=len(lr),p_pi=pi,
                p_iut=KB.p_iut(pe,pi),specificity_testable=len(lr)>=needed,
                locked_with_relaxed=len(locked&relaxed)/len(locked) if locked else 0.)


def _lens_statistics(data,B):
    twin={T:{tuple(r["unit"]):r for r in rows} for T,rows in data["twins"].items()}
    conditions={r["cond"]:r for r in data["rows"]};units=list(twin[.7]);out={"carry":{},"causal":{}}
    def arrays(condition,pop,route="tok"):
        rows={tuple(r["unit"]):r for r in condition[route]};base=twin[condition["T"]]
        chosen=[u for u in units if u in rows and u in base]
        excess=np.array([[_seg(rows[u],pop,s)-_seg(base[u],pop,s) for s in range(3)] for u in chosen])
        return excess,np.array([u[0] for u in chosen]),rows,base,chosen
    def scales(pop,segment):
        groups={}
        for u in units:
            other=(u[0],u[1],1)
            if u[2]==0 and other in twin[.7]:
                st=twin[.7][u]["stratum"]
                groups.setdefault(st,[]).append(_seg(twin[.7][u],pop,segment)-_seg(twin[.7][other],pop,segment))
        return {s:float(np.std(v,ddof=1)) if len(v)>1 else float("nan") for s,v in groups.items()}
    random_carries=[]
    for name,condition in conditions.items():
        if name.startswith("PUSH_RS") and data["gains"].get(condition["X"],{}).get("in_band",False):
            e,it,*_=arrays(condition,condition["X"])
            r=_ratio(e[:,2],e[:,1],it,B)
            if r["defined"] and KB.fin(r.get("K")) and r["K"]>0:random_carries.append(math.log(r["K"]))
    for route,path in (("TOK","tok"),("TOK+KV","pushed")):
        if "PUSH_H" not in conditions:continue
        e,it,rows,base,chosen=arrays(conditions["PUSH_H"],"H",path)
        carry=_ratio(e[:,2],e[:,1],it,B);sd=scales("H",0)
        d1=[x/sd[base[u]["stratum"]] for x,u in zip(e[:,0],chosen) if sd.get(base[u]["stratum"],0)>0]
        pi=RC.p_pi(carry["logK"],random_carries) if carry["defined"] and len(random_carries)>=10 and KB.fin(carry.get("logK")) else 1.
        out["carry"][route]=dict(carry=carry,logK_R=random_carries,n_R_defined=len(random_carries),
            p_pi=pi,p_iut=KB.p_iut(carry.get("p_le1",1.),pi),d1=RC.val(np.mean(d1) if d1 else np.nan))
    def bist_sample(condition,pop):
        _,_,rows,base,chosen=arrays(condition,pop);sd=scales(pop,2);z=[];it=[];undiv=0
        for u in chosen:
            if rows[u]["tok"]==base[u]["tok"]:undiv+=1;continue
            s=sd.get(base[u]["stratum"],0)
            z.append((_seg(rows[u],pop,2)-_seg(base[u],pop,2))/s if s>0 else np.nan);it.append(u[0])
        return z,it,undiv/max(len(chosen),1)
    if "PUSH_H" in conditions:
        z,it,undiv=bist_sample(conditions["PUSH_H"],"H")
        random=[bist_sample(c,c["X"])[:2] for n,c in conditions.items() if n.startswith("PUSH_RS") and data["gains"].get(c["X"],{}).get("in_band",False)]
        out["bist"]=_bistability(z,it,random,B,20)|dict(undiverged_share=undiv)
    from r1b_k_ana import qshare
    def confab(condition):
        _,_,rows,base,chosen=arrays(condition,condition["X"]);groups={}
        for u in chosen:
            d=qshare(rows[u]["claims"],SEG,23)-qshare(base[u]["claims"],SEG,23)
            if math.isfinite(d):groups.setdefault(u[0],[]).append(d)
        return np.array([np.mean(v) for v in groups.values()])
    for name,prefix,sign in (("PUSH_H","PUSH_RS",1),("CLAMP_H","CLAMP_MC",-1)):
        if name not in conditions:continue
        values=confab(conditions[name]);draws=KB.boot_stat(len(values),lambda ix:values[ix].mean(),B=B) if len(values)>2 else np.array([])
        random=[float(x.mean()) for n,c in conditions.items() if n.startswith(prefix)
                and (prefix!="PUSH_RS" or data["gains"].get(c["X"],{}).get("in_band",False)) and len(x:=confab(c))]
        pe=RC.p_one_sided(sign*draws) if len(draws) else 1.
        pi=RC.p_pi(sign*values.mean(),[sign*x for x in random]) if len(values) and len(random)>=20 else 1.
        out["causal"][name]=dict(d_br=RC.val(values.mean() if len(values) else np.nan),ci=KB.ci(draws) if len(draws) else None,
            p_effect=pe,p_pi=pi,p_iut=KB.p_iut(pe,pi),CSS=float(np.mean(conditions[name]["css"])) if conditions[name]["css"] else None,
            random=random)
    if data.get("offset"):
        chosen=data["offset"];e=np.array([[_seg(r["tok"],"H",s)-_seg(r["twin"],"H",s) for s in range(3)] for r in chosen])
        out["offset"]=_ratio(e[:,2],e[:,1],[r["unit"][0] for r in chosen],B)
        # Paired item bootstrap on both offsets; never compare independent CIs.
        short={tuple(r["unit"]):r for r in conditions["PUSH_H"]["tok"]};base=twin[.7]
        e0=np.array([[_seg(short[tuple(r["unit"])],"H",s)-_seg(base[tuple(r["unit"])],"H",s) for s in range(3)] for r in chosen])
        def difference(ix):
            den0=e0[ix,1].sum();den1=e[ix,1].sum()
            if den0==0 or den1==0:return np.nan
            k0=e0[ix,2].sum()/den0;k1=e[ix,2].sum()/den1
            return math.log(k1)-math.log(k0) if k0>0 and k1>0 else np.nan
        draws=KB.boot_stat(len(chosen),difference,B=B,groups=np.array([r["unit"][0] for r in chosen]))
        interval=KB.ci(draws);defined=out["offset"]["defined"] and out["carry"]["TOK"]["carry"]["defined"]
        out["offset"].update(log_difference_ci=interval,string=("carry not bound to context length" if defined and KB.inside(interval,math.log(1.25)) else
            "carry depends on context length" if defined and (KB.gt(interval) or KB.lt(interval)) else "undecided"))
    out["G_K21"]=data["G_K21"];out["H_A_label"]=data["H_A_label"]
    if not data["gains"].get("H",{}).get("in_band",False):
        for row in list(out["carry"].values())+[out.get("bist",{}),out["causal"].get("PUSH_H",{})]:
            row["p_iut"]=1.;row["calibration_gate"]="NSCALE_DN damage outside delta2 band"
    return out


def _session_statistics(data,B):
    cells={(r["block"],r["arm"]):{s["sid"]:s for s in r["sessions"]} for r in data["rows"]}
    def excess(block,arm,key="sH"):
        actual=cells.get((block,arm),{});base=cells.get((block,"clean"),{})
        ids=[i for i in actual if i in base]
        return np.array([[actual[i]["turns"][n][key]-base[i]["turns"][n][key] for n in range(8)] for i in ids]),ids
    out={}
    for family,block,arm in (("ACH","main","H_a3.0_FULL"),("TQA","tqa","Htq_a3.0_FULL")):
        e,ids=excess(block,arm)
        if not len(ids):continue
        carry=_ratio(e[:,2:8].sum(1),e[:,1:7].sum(1),np.arange(len(ids)),B)
        random=[]
        for b,a in cells:
            if b==block and a.startswith("RS"):
                er,ri=excess(block,a);r=_ratio(er[:,2:8].sum(1),er[:,1:7].sum(1),np.arange(len(ri)),B)
                if r["defined"] and KB.fin(r.get("K")) and r["K"]>0:random.append(math.log(r["K"]))
        pi=RC.p_pi(carry["logK"],random) if carry["defined"] and len(random)>=10 and KB.fin(carry.get("logK")) else 1.
        late=e[:,3:8].sum(1);routes={}
        for route in ("TOKEN","KV"):
            er,ir=excess(block,"H_a3.0_"+route)
            if len(ir):
                lr=er[:,3:8].sum(1);routes[route]=KB.ci(KB.boot_stat(len(lr),lambda ix:lr[ix].mean(),B=B))
        token=routes.get("TOKEN");kv=routes.get("KV")
        zero=lambda c:KB.fin(c["lo"]) and KB.fin(c["hi"]) and c["lo"]<=0<=c["hi"]
        routing=("carried by text and cache" if token and kv and KB.gt(token) and KB.gt(kv) else
                 "carried by the text (snowballing-like)" if token and kv and KB.gt(token) and zero(kv) else
                 "carried in the cache (latent)" if token and kv and KB.gt(kv) and zero(token) else "not carried by either route alone")
        err,ei=excess(block,arm,"err")
        out[family]=dict(carry=carry,logK_R=random,n_R_defined=len(random),p_pi=pi,p_iut=KB.p_iut(carry.get("p_le1",1.),pi),
             late=KB.ci(KB.boot_stat(len(late),lambda ix:late[ix].mean(),B=B)),routes=routes,route_string=routing,
             error_carry=_ratio(err[:,2:8].sum(1),err[:,1:7].sum(1),np.arange(len(ei)),B))
    clean=cells.get(("bist","clean"),{})
    def late(session):return float(np.mean([t["sH"] for t in session["turns"][5:8]]))
    groups={}
    for s in clean.values():groups.setdefault(s["template"],[]).append(late(s))
    sd=math.sqrt(np.mean([np.var(v,ddof=1)*2 for v in groups.values() if len(v)>1])) if groups else 0.
    def sample(arm):
        actual=cells.get(("bist",arm),{});z=[];items=[];same=0
        for ident,row in actual.items():
            if ident not in clean:continue
            if [t["tok"] for t in row["turns"]]==[t["tok"] for t in clean[ident]["turns"]]:same+=1;continue
            z.append((late(row)-late(clean[ident]))/sd if sd>0 else np.nan);items.append(row["template"])
        return z,items,same/max(len(actual),1)
    if clean and ("bist","H_a3.0_FULL") in cells:
        z,it,same=sample("H_a3.0_FULL");random=[sample(a)[:2] for b,a in cells if b=="bist" and a.startswith("RS")]
        out["bist"]=_bistability(z,it,random,B,19)|dict(undiverged_share=same)
    out["H_A_label"]=data["H_A_label"];out["boundary_label"]=data["boundary_label"]
    return out


def _carry_string(row,adjusted,lens,arm):
    if not row:return "not run (input missing)"
    if row.get("calibration_gate"):return "undecided (NSCALE_DN calibration outside target band)"
    c=row["carry"]
    if not c["defined"]:return "K not defined (post-release denominator not resolved)"
    if adjusted["reject"]:return f"H-neuron attractor in the chain lens (self-sustaining carry, {arm})" if lens else f"self-sustaining H ({arm})"
    if c.get("p_le1",1)<=.05/6 and row["p_pi"]>.05:return "generic persistence in the chain lens (not H-specific)" if lens else f"generic persistence (not H-specific) ({arm})"
    if KB.lt(c["K_ci"],1.):return "dissipative in the chain lens" if lens else f"dissipative H ({arm})"
    interval=c["K_ci"]
    if all(KB.fin(interval[k]) and interval[k]>0 for k in ("lo","hi")):
        logci={k:math.log(v) for k,v in interval.items()}
        if KB.inside(logci,math.log(1.25)) and logci["lo"]<=0<=logci["hi"] and row["p_pi"]<=.025:
            return "H-specific non-decaying persistence in the chain lens" if lens else f"H-specific non-decaying persistence ({arm})"
    return "undecided"


def analyze(ctx,lens_data=None,session_data=None):
    if lens_data is None:
        p=Path(ctx.path("k6e_data.pt"));lens_data=K._torch().load(p,weights_only=False) if p.exists() else None
    if session_data is None:
        p=Path(ctx.path("k8b_data.pt"));session_data=K._torch().load(p,weights_only=False) if p.exists() else None
    B=20 if ctx.mock else RC.N_BOOT
    lens=_lens_statistics(lens_data,B) if lens_data else {};sessions=_session_statistics(session_data,B) if session_data else {}
    members={"K6e TOK carry":lens.get("carry",{}).get("TOK",{}),"K6e TOK+KV carry":lens.get("carry",{}).get("TOK+KV",{}),
             "K6e bistability":lens.get("bist",{}),"K8b ACH carry":sessions.get("ACH",{}),
             "K8b TQA carry":sessions.get("TQA",{}),"K8b ACH bistability":sessions.get("bist",{})}
    q9=KB.holm_family({key:r.get("p_iut",1.) for key,r in members.items()});strings={}
    for key,row in members.items():
        if "carry" in key:strings[key]=_carry_string(row,q9[key],key.startswith("K6"),key.split()[1])
        else:
            if row.get("calibration_gate"):
                strings[key]="undecided (NSCALE_DN calibration outside target band)";continue
            mix=row.get("mix",{});tail=row.get("tail",{}).get("ci",{})
            passes=q9[key]["reject"] and mix.get("Delta",0)>=3 and mix.get("s1",9)<=1.5 and .1<=mix.get("pi",0)<=.9 and row.get("locked_with_relaxed",0)>=.3
            strings[key]=("H-neuron attractor in the chain lens (bistable within items)" if key.startswith("K6") else "within-item bistability (sessions)") if passes else (
                "no within-item bistability (lens)" if key.startswith("K6") else "no within-item bistability (sessions)") if tail and KB.lt(tail,.1) else "undecided"
    causal=lens.get("causal",{});q11=KB.holm_family({n:causal.get(n,{}).get("p_iut",1.) for n in ("PUSH_H","CLAMP_H")})
    for name in q11:
        row=causal.get(name,{});clamp=name=="CLAMP_H"
        text="H clamp lowers lens confabulation beyond random clamps" if clamp else "H push raises lens confabulation beyond random sets"
        equivalent=not row.get("calibration_gate") and row.get("ci") and KB.inside(row["ci"],.03) and (not clamp or (row.get("CSS") or 0)>=.2)
        strings["K6e "+name]=text if q11[name]["reject"] else "no causal effect on later confabulation" if equivalent else "undecided"
    result=dict(K6e=lens,K8b=sessions,Q9=q9,Q11=q11,strings=strings,mock=ctx.mock,
                statistical_primitives="inherited ratio_carry, mixture_lr, robust_tail_share, p_PI, Holm; no thresholds reduced",
                limitations=(lens_data or {}).get("missing",[])+(session_data or {}).get("missing",[]))
    ctx.write_json("persistence_verdicts.json",RC.jsonable(result))
    return result


def measure(ctx):
    env=_environment(ctx);parts={};blocked={}
    for key,fn in (("K6e",lens_persistence),("K8b",model_sessions)):
        try:parts[key]=fn(ctx,env)
        except RC.NotRun as exc:blocked[key]=str(exc)
    analyze(ctx,parts.get("K6e"),parts.get("K8b"))
    missing=blocked|{k:v["missing"] for k,v in parts.items() if v.get("missing")}
    status="done" if len(parts)==2 and not missing else "partial" if parts else "blocked"
    ctx.write_json("persistence_status.json",dict(status=status,coverage=list(parts),missing=missing,mock=ctx.mock))
    return dict(status=status,coverage=list(parts),reason=RC.canon(missing) if missing else None,mock=ctx.mock)
