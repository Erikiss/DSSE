"""G K3e neuron write-vector readouts and independent native verification.

No external neuron catalogue supplies labels. Original targets use their own
validated output-state J-lens; QAT explicitly uses logit-lens seed tokens and
makes no target-J or cross-model interpretability claim.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

import r1b_core as RC
import r1b_g_k as K
import r1b_k_gvra as GV


def _instrument(ctx,E):
    import torch
    if getattr(E.model,"_r1b_g_forward_only",False):
        return None,"native Gemma E rollout; logit-lens seeds (QAT, no validated target J-lens)"
    validation=K._load(ctx,"lens_validate","lens_validation.json")
    output=validation.get("G_K_J",{}).get("median_cos",{}).get("output",-1.)
    if output<.9:
        raise RC.NotRun("K3e output-state readouts require target output-convention J-lens bridge >=0.9")
    lens=K._load(ctx,"lens_fit","lens.pt")
    return lens,"native Gemma E rollout; own validated output-state J-lens seeds"


def _neuron_activity(E,frame,pair,chains,pole):
    fid=E.layout.encode(*pair);out=[]
    for chain in chains:
        row=E.teacher_force([frame+chain],populations={"target":[fid]})["rows"][0]
        out.append(float((pole*row["z"][len(frame):,0]).max()))
    return out


def _detector_controls(ctx,E,refs,frame):
    """Exact top-100 token-neuron observations on the 200 N-text windows."""
    import torch
    need=4 if ctx.mock else 200;length=32 if ctx.mock else 256
    texts=(ctx.data().get("wt") or {}).get("ntext",[])
    windows=K._raw_windows(E,texts,need,length,ctx.mock)
    state=K._read_checkpoint(ctx,"readout_token_detectors")
    top=state["top"] if state else None;tokens=state["tokens"] if state else None
    start=state["next"] if state else 0
    for i,ids in enumerate(windows):
        if i<start:continue
        row=E.teacher_force([ids],populations={"all":list(range(E.layout.total))})["rows"][0]
        z=row["z"].T;tid=torch.tensor(ids).expand(len(z),-1)
        if top is not None:z=torch.cat((top,z),1);tid=torch.cat((tokens,tid),1)
        top,ix=z.topk(min(100,z.shape[1]),dim=1);tokens=tid.gather(1,ix)
        if hasattr(ctx,"time_left") and ctx.time_left()<60:
            K._save_checkpoint(ctx,"readout_token_detectors",dict(top=top,tokens=tokens,next=i+1))
            raise RC.StepYield()
        if (i+1)%8==0:K._save_checkpoint(ctx,"readout_token_detectors",dict(top=top,tokens=tokens,next=i+1))
    controls=[]
    for fid,t in enumerate(tokens):
        values,counts=t.unique(return_counts=True);k=int(counts.argmax())
        if counts[k]<.5*len(t):continue
        pair=E.layout.decode(fid);token=int(values[k])
        act=_neuron_activity(E,frame,pair,[[token]*4],1)[0]
        controls.append(dict(neuron=list(pair),token=token,top100_share=float(counts[k]/len(t)),
                             activation=act,threshold=float(refs["p998_pos"][fid]),
                             passed=bool(act>refs["p998_pos"][fid])))
        if len(controls)>=(3 if ctx.mock else 20):break
    return controls


def measure(ctx):
    import torch
    from r1b_g_lens import rollout,frame_spec
    from r1b_g_k_chain import _seeds,_frame
    E=K._engine(ctx);hn=K._load(ctx,"k_prepare","hn.json");refs=K._load(ctx,"k_prepare","references.pt")
    lens,instrument=_instrument(ctx,E)
    frame=_frame(E.tok,ctx.data(),ctx.mock) if ctx.data().get("lens_frame_text") else frame_spec(E)["frame_ids"]
    P0=frame[:-1]
    H={tuple(x) for x in hn["H_tq"]+hn["H_ar"]}
    if not H:raise RC.NotRun("K3e has no H_tq or H_ar neurons")
    neurons=sorted(H);g=torch.Generator().manual_seed(1305)
    for layer in sorted({l for l,j in neurons}):
        neurons.extend((layer,int(j)) for j in torch.randperm(E.layout.widths[layer],generator=g)[:2 if ctx.mock else 20])
    neurons=list(dict.fromkeys(neurons))
    if ctx.mock:neurons=neurons[:6]
    nseed=3 if ctx.mock else 10;steps=6 if ctx.mock else 16
    controls=K._cached(ctx,"readout_detector_controls",lambda:_detector_controls(ctx,E,refs,P0))
    cache={}
    for pair in neurons:
        layer,j=pair
        for pole in (1,-1):
            def generate(pair=pair,pole=pole):
                layer,j=pair;mu=refs["mu_gen"][layer].float()
                write=E.layers[layer].mlp.down_proj.weight[:,j].detach().float().cpu()
                if not torch.isfinite(write).all() or write.norm()==0:
                    raise RC.NotRun(f"K3e neuron {pair} has no finite nonzero write vector")
                state=mu+pole*write/write.norm()*mu.norm()
                seeds=_seeds(E,state,layer,lens,nseed)
                r=rollout(E,layer,state,seeds,steps=steps,prior_state=mu,frame_ids=frame,
                          key=ctx.key("g_k3e",layer,j,pole))
                chains=[[seed]+r["tok"][i].tolist() for i,seed in enumerate(seeds)]
                return dict(chains=chains,rollout=r,state=state,prior=mu,seeds=seeds,
                            acts=_neuron_activity(E,P0,pair,chains,pole))
            cache[pair,pole]=K._cached(ctx,f"readout_generate_{layer}_{j}_{pole}",generate)
    result={};false_accept=[]
    for (pair,pole),entry in cache.items():
        layer,j=pair;fid=E.layout.encode(layer,j)
        # Stored negative thresholds are raw z quantiles. Verification is on
        # signed activity (-z at the negative pole), hence the sign conversion.
        threshold=float(refs["p998_pos"][fid] if pole==1 else -refs["p998_neg"][fid])
        foreign=[c for (other,_),e in cache.items() if other!=pair for c in e["chains"][:2]][:6 if ctx.mock else 60]
        if not foreign:raise RC.NotRun("K3e requires other-neuron chains for its foreign-chain verifier controls")
        def verify_pole():
            foreign_acts=_neuron_activity(E,P0,pair,foreign,pole)
            q90=float(np.percentile(foreign_acts,90))
            def verify(unit,cand):
                acts=sorted(cand["acts"],reverse=True)
                c1=acts[0]>threshold;c2=len(acts)>=3 and all(a>q90 for a in acts[:3])
                return ("correct" if c1 and c2 else "minor fix needed" if c1 or c2 else "critically flawed"),acts[0]
            def generate(unit,attempt):
                return dict(chains=entry["chains"],acts=entry["acts"]) if attempt==0 else None
            def revise(unit,cand,attempt):
                best=int(np.argmax(cand["acts"][:len(entry["chains"])]))
                at=int(entry["rollout"]["e"][best].argmin())
                forced=[entry["chains"][best][1:at+1]+[int(entry["rollout"]["alt"][best,at])]]
                revised=rollout(E,layer,entry["state"],[entry["seeds"][best]],steps=steps,
                                prior_state=entry["prior"],frame_ids=frame,forced=forced,
                                key=ctx.key("g_k3e_revision",layer,j,pole,attempt))
                chain=[entry["seeds"][best]]+revised["tok"][0].tolist()
                return dict(chains=cand["chains"]+[chain],acts=cand["acts"]+_neuron_activity(E,P0,pair,[chain],pole))
            v=GV.gvra([0],generate,verify,revise,limit=3)[0]
            return dict(status=v["status"],attempts=v["attempts"],trace=v["trace"],
                        candidate=v["candidate"],best_act=max(entry["acts"]),thr=threshold,q90_other=q90,
                        readouts=[E.tok.decode(c) for c in entry["chains"][:3]],
                        in_H=pair in H,foreign_threshold_exceedance=float(np.mean(np.array(foreign_acts)>threshold)))
        row=K._cached(ctx,f"readout_verify_{layer}_{j}_{pole}",verify_pole)
        result[f"{layer}:{j}:{'+' if pole==1 else '-'}"]=row
        false_accept.append(row["foreign_threshold_exceedance"]>.5)
    lexicon={};words=dict(hedge=("maybe","perhaps","possibly","might"),comply=("sure","yes","okay","certainly"),refuse=("sorry","cannot","unable"))
    for key,row in result.items():
        original=" ".join(row["readouts"]);text=original.lower()
        lexicon[key]={name:int(any(w in text for w in ws)) for name,ws in words.items()}|dict(
            digits=int(any(c.isdigit() for c in text)),caps=int(any(w[:1].isupper() for w in original.split())))
    nH=sum(r["in_H"] for r in result.values());accepted=sum(r["in_H"] and r["status"]=="accepted" for r in result.values())
    output=dict(readouts=result,lexicon=lexicon,verifier_error=GV.error_rates([r["passed"] for r in controls],false_accept),
                string=f"round-trip rate {accepted}/{nH} of H poles",n_detectors=len(controls),
                detector_controls=controls,instrument=instrument,mock=ctx.mock,descriptive=True,
                verifier_calibrated=bool(controls),external_catalogue_used=False,
                threshold_convention="raw negative z threshold negated before comparing -z activation")
    ctx.write_json("k3e_readouts.json",output)
    ctx.save_pt("k3e_chains.pt",dict(records=cache,frame=frame,instrument=instrument,mock=ctx.mock))
    return dict(status="done" if controls else "partial",coverage=["K3e"],
                reason=None if controls else "no N-text token-detector neurons: verifier sensitivity unmeasured",
                mock=ctx.mock)
