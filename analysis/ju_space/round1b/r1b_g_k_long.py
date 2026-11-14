"""G K8c numeric first-error hazard and position-matched native lens reach.

All step values come from the model's text and the inherited arithmetic parser.
MOCK changes counts only; malformed text is never completed with an oracle.
"""
from __future__ import annotations

import contextlib
import math
import re
from pathlib import Path

import numpy as np

import r1b_core as RC
import r1b_g_k as K
import r1b_g_k_chain as C
import r1b_g_k_persistence as P
import r1b_k_base as KB
from r1b_k_s4 import parse_long, _OPF


def value_spans(tok,tokens,item):
    """Align actual generated token indices to parsed value spans, not step labels."""
    text=tok.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    bounds=[0]+[len(tok.decode(tokens[:i+1],skip_special_tokens=False,clean_up_tokenization_spaces=False)) for i in range(len(tokens))]
    if any(a>b for a,b in zip(bounds,bounds[1:])):
        raise ValueError("non-monotone target decoder prevents exact numeric-span alignment")
    values={};newlines={};chars={}
    for match in re.finditer(r"Step\s*(\d+)\s*:\s*(-?\d+)",text):
        step=int(match.group(1))
        if step in values:continue  # Same first-occurrence convention as parse_long.
        lo,hi=match.span(2)
        values[step]=[i for i,(a,b) in enumerate(zip(bounds,bounds[1:])) if a<hi and b>lo]
        chars[step]=[lo,hi]
        at=text.find("\n",hi)
        # Only a representable boundary: the read token contains the newline
        # and no characters of the following step (hence no future value).
        newlines[step]=next((i for i,(a,b) in enumerate(zip(bounds,bounds[1:])) if a<=at<b and b<=at+1),None) if at>=0 else None
    return dict(text=text,value_positions=[values.get(s["k"],[]) for s in item["steps"]],
                value_char_spans=[chars.get(s["k"]) for s in item["steps"]],
                newline_positions=[newlines.get(s["k"]) for s in item["steps"]])


@contextlib.contextmanager
def _all_cett(E):
    """All-neuron CETT sum for the explicitly named unsigned H CETT share."""
    import torch
    handles=[];state={}
    for layer,module in enumerate(E.layers):
        def hook(module,args,output,layer=layer):
            z=args[0][0].detach().float();den=output[0].detach().float().norm(dim=-1).clamp_min(1e-9)
            value=(z.abs()*E._weights(layer)).sum(-1)/den
            state[layer]=value.cpu()
        handles.append(module.mlp.down_proj.register_forward_hook(hook))
    try:yield state
    finally:
        for handle in handles:handle.remove()


def _prompt(E,item):
    messages=([dict(role="user",content=item["preamble"]),dict(role="assistant",content="Noted.")] if item.get("preamble") else [])
    messages.append(dict(role="user",content=item["prompt"]))
    text=E.tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return E.encode(text,chat=False)


def collect(ctx,E,items,hn,refs,detector,ha):
    import torch
    populations={"H":ha,**{f"RS{i}":s for i,s in enumerate(hn["RS39"])}}
    rows=[]
    for item in items:
        prompt=_prompt(E,item)
        for sample in range(2):
            def cell(item=item,sample=sample,prompt=prompt):
                answer=E.decode([prompt],keys=[ctx.key("g_k8_long",item["id"],sample)],temperature=.7,
                                max_new=60 if ctx.mock else 200)[0]
                tokens=answer["ids"];alignment=value_spans(E.tok,tokens,item)
                steps,fmt_ok=parse_long(alignment["text"],item)
                first=next((i for i,st in enumerate(steps) if not st["local"]),None)
                with _all_cett(E) as cett:
                    read=P._read(E,prompt+tokens,populations,refs,detector)
                n=len(prompt);positions={fid:i for i,fid in enumerate(read["neuron_ids"])}
                hcols=[positions[k] for k in K._flat(E.layout,ha)]
                numerator=read["cett"][:,hcols].sum(1) if hcols else torch.zeros(n+len(tokens))
                denominator=torch.stack(list(cett.values())).sum(0)
                share=numerator/denominator.clamp_min(1e-12)
                aH=read["aX"].get("H",torch.zeros(n+len(tokens)))[n:]
                random=[read["aX"][f"RS{i}"][n:] for i in range(len(hn["RS39"]))]
                RA=torch.stack(random,1) if random else torch.empty(len(tokens),0)
                Ht=(aH-RA.mean(1))/(RA.std(1,unbiased=False)+1e-9) if RA.shape[1]>1 else None
                Rt=torch.stack([(RA[:,j]-torch.cat((RA[:,:j],RA[:,j+1:]),1).mean(1))/
                    (torch.cat((RA[:,:j],RA[:,j+1:]),1).std(1,unbiased=False)+1e-9) for j in range(RA.shape[1])],1) if RA.shape[1]>1 else None
                ent=read["entropy"][n-1:n+len(tokens)-1]
                lp=read["next_logprobs"][n-1:n+len(tokens)-1]
                sH=read["sH"][n:] if read["sH"] is not None else None
                per=[]
                for i in range(len(tokens)):
                    per.append(dict(aH=float(aH[i]),sH=None if sH is None else float(sH[i]),
                        Ht=None if Ht is None else float(Ht[i]),rs=RA[i].tolist(),rs_tilde=None if Rt is None else Rt[i].tolist(),ent=float(ent[i]),
                        logprob=float(lp[i]),token_position=n+i,continuation_position=i,cett_share_H=float(share[n+i])))
                step_values=[]
                for i,indices in enumerate(alignment["value_positions"]):
                    if not indices:step_values.append(None);continue
                    def mean(key):
                        values=[per[k][key] for k in indices]
                        return None if any(v is None for v in values) else float(np.mean(values))
                    rs=RA[indices].mean(0).tolist()
                    step_values.append({k:mean(k) for k in ("aH","sH","Ht","ent","logprob","cett_share_H")}|dict(
                        rs=rs,rs_tilde=None if Rt is None else Rt[indices].mean(0).tolist(),
                        token_position=per[indices[0]]["token_position"],value_positions=indices))
                grams=[tuple(tokens[i:i+4]) for i in range(max(0,len(tokens)-3))]
                return dict(item=item["id"],sample=sample,tok=tokens,prompt_ids=prompt,text=alignment.pop("text"),
                            steps=steps,istar=first,fmt_ok=fmt_ok,per=per,per_step=step_values,
                            rep4=len(grams)!=len(set(grams)),preamble=bool(item.get("preamble")),
                            n_steps=len(item["steps"]),**alignment)
            rows.append(K._cached(ctx,"long_cell_"+str(K._hash((item["id"],sample))),cell))
    return rows


def hazard_table(rows,items):
    """Value-token means at i-1 predict the first local error at i; item clusters."""
    X=[];y=[];groups=[];controls=[];meta=[]
    for row in rows:
        if not row["fmt_ok"]:continue
        stop=len(row["steps"]) if row["istar"] is None else row["istar"]+1
        item=items[row["item"]]
        for i in range(1,stop):
            prev=row["per_step"][i-1];current=row["per_step"][i]
            if prev is None or prev["Ht"] is None or prev.get("rs_tilde") is None:continue
            difficulty=item["steps"][i];rs=np.asarray(prev["rs"])
            position=current["token_position"] if current else len(row["prompt_ids"])+len(row["tok"])
            X.append([prev["Ht"],i+1,position,prev["ent"],int(row["rep4"]),
                      int(row["preamble"]),len(str(abs(difficulty["result"]))),difficulty["cls"],difficulty["op"]])
            y.append(float(row["istar"]==i));groups.append(row["item"])
            controls.append(prev["rs_tilde"])
            meta.append(dict(item=row["item"],sample=row["sample"],step=i+1,previous_value_positions=prev["value_positions"],
                             missing_current_value=current is None))
    return X,np.asarray(y),np.asarray(groups),np.asarray(controls),meta


def event_controls(rows,B):
    """Descriptive item-bootstrap event locking, matched steps and circular nulls."""
    error=[r for r in rows if r["fmt_ok"] and r["istar"] is not None]
    clean=[r for r in rows if r["fmt_ok"] and r["istar"] is None]
    observations=[]
    for row in error:
        event=row["istar"];pool=[r for r in clean if r["item"]!=row["item"] and len(r["steps"])>event and r["steps"][event]["cls"]==row["steps"][event]["cls"]]
        other=pool[K._hash((row["item"],row["sample"]))%len(pool)] if pool else None
        rng=np.random.default_rng(5100+K._hash((row["item"],row["sample"]))%100000)
        shifts=rng.integers(1,max(len(row["steps"]),2),20)
        for offset in (-3,-2,-1,0):
            i=event+offset
            if not 0<=i<len(row["per_step"]) or row["per_step"][i] is None:continue
            actual=row["per_step"][i]["Ht"]
            matched=other["per_step"][i]["Ht"] if other and other["per_step"][i] else None
            sur=[row["per_step"][(i+int(s))%len(row["steps"])]["Ht"] for s in shifts if row["per_step"][(i+int(s))%len(row["steps"])]]
            sur=[v for v in sur if v is not None]
            observations.append(dict(item=row["item"],sample=row["sample"],offset=offset,Ht=actual,
                matched_Ht=matched,matched_item=other["item"] if other else None,
                surrogate_mean=float(np.mean(sur)) if sur else None))
    result={}
    for offset in (-3,-2,-1,0):
        selected=[r for r in observations if r["offset"]==offset and r["Ht"] is not None]
        record={"n":len(selected)}
        for comparator in ("matched_Ht","surrogate_mean"):
            pairs=[r for r in selected if r[comparator] is not None]
            if not pairs:record[comparator]=dict(status="not_run",reason="no matched observations");continue
            d=np.array([r["Ht"]-r[comparator] for r in pairs]);g=np.array([r["item"] for r in pairs])
            record[comparator]=dict(mean=float(d.mean()),ci=KB.ci(KB.boot_stat(len(d),lambda ix:d[ix].mean(),B=B,groups=g)))
        result[offset]=record
    return dict(rows=observations,summary=result,surrogates="20 nonzero within-text circular step shifts, seed5100+itemhash",descriptive=True)


def hazard_profile(X,y,groups,B):
    """Difficulty-adjusted descriptive step trend and cumulative first-error curve."""
    if len(y)<20 or len(set(y.tolist()))<2:return dict(status="not_run",reason="too few parsed risk-set observations")
    from r1b_k_ana import design
    Z=design(X);steps=np.array([r[1] for r in X],float)
    # The main hazard uses the inherited three-df spline. The separately named
    # descriptive linear trend tests whether adjusted hazard rises per step.
    spline_free=np.c_[steps,Z[:,4:]]
    fit=KB.logit_fit(spline_free,y,l2=1e-4)
    draws=KB.boot_stat(len(y),lambda ix:KB.logit_fit(spline_free[ix],y[ix],l2=1e-4)["beta"][0],B=B,groups=groups)
    interval=KB.ci(draws);curve=[];survival=1.
    for step in sorted(set(steps)):
        mask=steps==step;rate=float(y[mask].mean());survival*=1-rate
        curve.append(dict(step=int(step),at_risk=int(mask.sum()),first_errors=int(y[mask].sum()),hazard=rate,cumulative=1-survival))
    return dict(beta_step=float(fit["beta"][0]),ci=interval,curve=curve,
        string="hazard rises with step after difficulty adjustment" if KB.gt(interval) else "constant per-step hazard" if KB.inside(interval,.10) else "undecided",
        descriptive=True)


def error_conditioned_profile(rows,items,B):
    """Later local errors conditional on a prior error, separately from first-error risk."""
    X=[];y=[];groups=[];rates={}
    for row in rows:
        if not row["fmt_ok"]:continue
        for i in range(1,len(row["steps"])):
            if row["steps"][i]["v"] is None or row["steps"][i-1]["v"] is None:continue
            previous=int(not row["steps"][i-1]["local"]);error=int(not row["steps"][i]["local"])
            step=items[row["item"]]["steps"][i]
            X.append([previous,i+1,0.,0.,0.,int(row["preamble"]),len(str(abs(step["result"]))),step["cls"],step["op"]])
            y.append(error);groups.append(row["item"]);rates.setdefault((i+1,previous),[]).append(error)
    result=dict(by_step=[dict(step=k[0],previous_error=bool(k[1]),n=len(v),error_rate=float(np.mean(v))) for k,v in sorted(rates.items())],descriptive=True)
    if len(y)<20 or len(set(y))<2:return result|dict(status="not_run",reason="too few parsed subsequent-error observations")
    from r1b_k_ana import design
    Z=design(X);y=np.asarray(y);groups=np.asarray(groups)
    fit=KB.logit_fit(Z,y,l2=1e-4)
    draws=KB.boot_stat(len(y),lambda ix:KB.logit_fit(Z[ix],y[ix],l2=1e-4)["beta"][0],B=B,groups=groups)
    return result|dict(beta_previous_error=float(fit["beta"][0]),ci=KB.ci(draws),status="measured")


def reach(ctx,E,rows,items,refs):
    import torch
    from r1b_g_k_readouts import _instrument
    from r1b_g_lens import rollout,frame_spec
    lens,instrument=_instrument(ctx,E)
    error=[r for r in rows if r["fmt_ok"] and r["istar"] is not None][:3 if ctx.mock else 300]
    clean=[r for r in rows if r["fmt_ok"] and r["istar"] is None]
    if not error:raise RC.NotRun("K8c reach has no genuinely parsed first-error text")
    last=len(E.layers)-1;blocks=sorted({int(math.floor(b*last/27+.5)) for b in (14,18)})
    cells=[];control_missing=0
    for row in error:
        event=row["istar"]
        candidates=[r for r in clean if r["item"]!=row["item"] and len(r["steps"])>event and r["steps"][event]["cls"]==row["steps"][event]["cls"]]
        match=candidates[K._hash((row["item"],row["sample"]))%len(candidates)] if candidates else None
        if match is None:control_missing+=1
        for kind,actual in (("error",row),("clean",match)):
            if actual is None:continue
            for index in range(max(0,event-2),event+1):
                if index+1>=len(actual["steps"]):continue
                position=actual["newline_positions"][index]
                if position is None:continue
                previous=actual["steps"][index]["v"]
                step=items[actual["item"]]["steps"][index+1]
                target=_OPF[step["op"]](previous,step["b"]) if previous is not None else None
                if target is None:continue
                full=actual["prompt_ids"]+actual["tok"]
                absolute=len(actual["prompt_ids"])+position
                # Capture only the visible prefix, so no answer suffix can
                # affect the readout even if an architecture changes later.
                hidden=E.hidden([full[:absolute+1]],blocks)[0]
                for block in blocks:
                    cells.append(dict(item=actual["item"],sample=actual["sample"],kind=kind,index=index,
                        event=event,pair=(row["item"],row["sample"]),block=block,position=position,
                        target=int(target),state=hidden[block][-1]))
    frame=frame_spec(E)["frame_ids"];out=[]
    for cell in cells:
        peers=[c["state"] for c in cells if c["block"]==cell["block"] and c["item"]!=cell["item"]]
        if not peers:continue
        prior=torch.stack(peers).mean(0)
        for variant,state in (("V1",cell["state"]),("V2",prior)):
            def observe(cell=cell,prior=prior,variant=variant,state=state):
                seeds=C._seeds(E,state,cell["block"],lens,2 if ctx.mock else 10)
                generated=rollout(E,cell["block"],state,seeds,steps=4 if ctx.mock else 8,prior_state=prior,
                                  frame_ids=frame,key=ctx.key("g_k8_reach",cell["item"],cell["sample"],cell["index"],cell["block"]))
                from r1b_k_labels import numerals
                rng=np.random.default_rng(551)
                records=[]
                for rank,seed in enumerate(seeds):
                    tokens=[seed]+generated["tok"][rank].tolist()
                    hit=any(int(v)==cell["target"] for v,step,completed in numerals([E.tok.decode([t]) for t in tokens]) if completed)
                    shuffled=[]
                    for _ in range(20):
                        null=rng.permutation(tokens).tolist()
                        shuffled.append(any(int(v)==cell["target"] for v,step,completed in numerals([E.tok.decode([t]) for t in null]) if completed))
                    records.append({k:v for k,v in cell.items() if k!="state"}|dict(variant=variant,seed=seed,seed_rank=rank,
                        tok=tokens,hit=bool(hit),surrogate_hit=float(np.mean(shuffled))))
                return records
            key=K._hash((cell["item"],cell["sample"],cell["kind"],cell["index"],cell["pair"],cell["block"],variant))
            out.extend(K._cached(ctx,"long_reach_"+str(key),observe))
    return dict(rows=out,missing_position_matched_texts=control_missing,source_cells=len(cells),
                instrument=instrument,descriptive=True,controls_complete=control_missing==0,
                surrogate="20 token-order shuffles per generated chain, seed551")


def analyze(ctx,data,items):
    from r1b_k_ana import k6b_rule
    rows=data["rows"];B=200 if ctx.mock else RC.N_BOOT
    X,y,g,Rt,meta=hazard_table(rows,items)
    rule=k6b_rule(X,y,g,Rt,B=B)
    complete=Rt.ndim==2 and Rt.shape[1]>=39
    if not complete and not ctx.mock:
        rule["p_rank"]=1.;rule["p_iut"]=1.;rule["control_gate"]="specificity not testable: fewer than39 matched controls"
    string=("H-activity predicts the next error beyond length, position, difficulty and uncertainty" if rule.get("p_iut",1.)<=.05 else
            "length artifact (model side)" if rule.get("_art") else "no association (model side)" if rule.get("_none") else "undecided")
    fail=float(np.mean([not r["fmt_ok"] for r in rows])) if rows else 1.
    result=dict(string=string,rule={k:v for k,v in rule.items() if not k.startswith("_")},
                G_K13="format not followed" if fail>.20 else None,parse_fail=fail,n_first_errors=int(y.sum()),
                hazard_profile=hazard_profile(X,y,g,B),error_conditioned_profile=error_conditioned_profile(rows,items,B),
                events=event_controls(rows,B),mock=ctx.mock,
                covariates=["lagged value-token Htilde","step spline(3df)","current value token position (end position for omission)",
                            "previous value entropy","repeated4gram","preamble","groundtruth result digits","operation difficulty class","operation"],
                value_mapping="actual generated value tokens, excluding Step labels; no oracle substitution",
                share_definition="unsigned H CETT sum divided by all-neuron CETT sum; not a signed direct-logit attribution")
    ctx.save_pt("hazard_data.pt",dict(X=X,y=y,groups=g,random=Rt,rows=meta,mock=ctx.mock))
    ctx.write_json("long_verdicts.json",RC.jsonable(result))
    return result


def measure(ctx):
    E,hn,refs,detectors,populations,name,label=P._environment(ctx)
    ha=populations["H"]
    if not ha or name not in detectors:raise RC.NotRun("K8c requires identified H_A and independent detector")
    items=K._need_rows(ctx,ctx.data(),"ach_long",None,4 if ctx.mock else 400)
    byid={item["id"]:item for item in items}
    rows=collect(ctx,E,items,hn,refs,detectors[name],ha)
    data=dict(rows=rows,items=byid,H_A_label=label,mock=ctx.mock,counts=dict(items=len(items),samples=2,max_new=60 if ctx.mock else 200))
    ctx.save_pt("long.pt",data)
    verdict=analyze(ctx,data,byid)
    try:
        reached=reach(ctx,E,rows,byid,refs)
        ctx.save_pt("reach.pt",reached)
        reach_status="done" if reached["controls_complete"] else "partial"
        reason=None if reach_status=="done" else "some error texts lack position/difficulty-matched error-free controls"
    except RC.NotRun as exc:
        reach_status="blocked";reason=str(exc)
        ctx.write_json("reach_status.json",dict(status=reach_status,reason=reason,mock=ctx.mock))
    status="done" if reach_status=="done" else "partial"
    ctx.write_json("long_status.json",dict(status=status,coverage=["K8c"],reach_status=reach_status,
                   G_K13=verdict["G_K13"],reason=reason,mock=ctx.mock))
    return dict(status=status,coverage=["K8c"],reason=reason,mock=ctx.mock)
