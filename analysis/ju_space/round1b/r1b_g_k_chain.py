"""Independent Gemma K chain observations and inherited arithmetic scoring.

Every generated row retains its target, source, seed and decoding instrument.
Protocol coverage is explicit: raw observations do not imply that all K4--K10
controls have run or that an inherited scientific verdict is available.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
from pathlib import Path


def _checkpoint(ctx,name,fn):
    """Yield between resumable cells before the notebook's runtime guard."""
    import r1b_core as core
    if hasattr(ctx,"time_left") and ctx.time_left()<30:
        raise core.StepYield()
    return ctx.cached(name,fn)


def normalize_item(raw):
    """Accept the original K raw bank, or a descriptive explicit-label item."""
    item = dict(raw)
    if not item.get("id") or not isinstance(item.get("prompt"), str):
        raise ValueError("arithmetic items require id and raw prompt")
    if "A" not in item:
        item["A"] = int(item["answer"])
    if "I" not in item:
        item["I"] = list(map(int, item.get("intermediates", [])))
    if "P" not in item:
        # Guessing operand identity from text changes PROMPT/TRUE-INT precedence.
        raise ValueError("arithmetic scoring requires explicit operands P")
    item["P"] = list(map(int, item["P"]))
    item["I"] = list(map(int, item["I"]))
    item["A"] = int(item["A"])
    item.setdefault("Iplus", [])
    item.setdefault("split", "A-lens")
    item.setdefault("cal", False)
    return item


def score_tokens(tok, tokens, item, W=(), OWN=()):
    """The exact K1/K4 completed-numeral parser, precedence and loop censoring."""
    from r1b_k_labels import score_chain
    return score_chain([tok.decode([int(t)]) for t in tokens], normalize_item(item),
                       W=W, OWN=OWN, tokens=list(map(int, tokens)))


def mapped_blocks(n_layers):
    return sorted({int(math.floor(b * (n_layers - 1) / 27 + 0.5)) for b in (10, 14, 18, 22)})


def _load_pt(ctx, stage, filename, required=False):
    import torch
    p = Path(ctx.artifact(stage, filename))
    if p.is_file():
        return torch.load(p, map_location="cpu", weights_only=False)
    if required:
        raise FileNotFoundError(f"missing G artifact: {stage}/{filename}")
    return None


def _load_json(ctx, stage, filename):
    p = Path(ctx.artifact(stage, filename))
    return json.loads(p.read_text()) if p.is_file() else None


def _config(ctx):
    # Scientific counts are fixed; only MOCK uses reduced engineering fixtures.
    return dict(correct=4 if ctx.mock else 256, wrong=4 if ctx.mock else 150,
                seeds=2 if ctx.mock else 10, steps=8 if ctx.mock else 16,
                screening=8 if ctx.mock else None, samples=1 if ctx.mock else 4,
                answer_tokens=4 if ctx.mock else 12, own_tokens=8 if ctx.mock else 64)


def gate_arithmetic(ctx, engine, items):
    """Greedy + keyed samples, true correctness, OWN and forward W enumeration."""
    from r1b_k_labels import parse_number, w_enum
    cfg = _config(ctx)
    gates = {}
    for it in items:
        def compute(it=it):
            prompt = engine.encode(it["prompt"])
            greedy = engine.decode([prompt], max_new=cfg["answer_tokens"], temperature=0)[0]
            samples = engine.decode([prompt] * cfg["samples"],
                                    keys=[ctx.key("g_k_gate", it["id"], s) for s in range(cfg["samples"])],
                                    max_new=cfg["answer_tokens"], temperature=0.7)
            answer = parse_number(greedy["text"])
            own = sorted({n for n in [answer] + [parse_number(x["text"]) for x in samples] if n is not None})
            wrong = w_enum(it, answer) if answer != it["A"] and it.get("ops") else []
            return dict(stratum="correct" if answer == it["A"] else "wrong", greedy=greedy["ids"],
                        greedy_text=greedy["text"], samples=[x["ids"] for x in samples], own=own,
                        W=wrong, answer=answer, sample_keys=[ctx.key("g_k_gate", it["id"], s)
                                                          for s in range(cfg["samples"])])
        gates[it["id"]] = _checkpoint(ctx,"gate_" + str(it["id"]), compute)
    return gates


def _choose_items(items, gates, cfg):
    by_id = {x["id"]: x for x in items}
    correct = [i for i in gates if gates[i]["stratum"] == "correct"]
    wrong = [i for i in gates if gates[i]["stratum"] == "wrong"][:cfg["wrong"]]
    match = lambda i: (by_id[i].get("template"), by_id[i].get("nops"), by_id[i].get("mul"),
                       tuple(by_id[i].get("digits", [])))
    desired = {match(i) for i in wrong}
    chosen = [i for i in correct if match(i) in desired][:cfg["correct"]]
    chosen += [i for i in correct if i not in chosen][:cfg["correct"] - len(chosen)]
    return chosen, wrong


def foreign_indices(ids, item_by_id, seed=51):
    """Deterministic derangement, preferring the original template matching."""
    import numpy as np
    if len(ids) < 2:
        raise ValueError("foreign-state control needs at least two different items")
    groups = {}
    for k, ident in enumerate(ids):
        it = item_by_id[ident]
        key = (it.get("template"), it.get("nops"), it.get("mul"), tuple(it.get("digits", [])))
        groups.setdefault(key, []).append(k)
    rng = np.random.default_rng(seed)
    foreign = list(range(len(ids)))
    matched = [False] * len(ids)
    for group in groups.values():
        if len(group) > 1:
            shuffled = rng.permutation(group).tolist()
            for i, j in zip(shuffled, shuffled[1:] + shuffled[:1]):
                foreign[i], matched[i] = j, True
    for i in range(len(ids)):
        if foreign[i] == i:
            foreign[i] = (i + 1) % len(ids)
    return foreign, matched


def _frame(tok, data, mock):
    from r1b_k_labels import P0, PH
    text = data.get("lens_frame_text", P0)
    prefix = tok(text, add_special_tokens=False)["input_ids"]
    placeholder = tok(PH, add_special_tokens=False)["input_ids"]
    if "lens_frame_ids" in data:
        raise ValueError("G frame inputs must be raw target-tokenized text, never imported token IDs")
    if "lens_frame_text" not in data:
        # An independently fixed Gemma adaptation of the original 16-token
        # instrument. Freeze its actual target IDs in each measurement artifact.
        prefix = (prefix * (1 + 16 // max(len(prefix), 1)))[:16]
        placeholder = tok(".", add_special_tokens=False)["input_ids"]
    if len(prefix) != 16 or len(placeholder) != 1:
        raise ValueError("G K4 requires a 16-token neutral frame and one-token '.' placeholder; "
                         "provide target-compatible lens_frame_text")
    return prefix + placeholder


def _seeds(engine, state, block, lens, k):
    import torch
    from r1b_g_adapter import unembed
    h = state.to(engine.device)
    if lens is not None:
        maps = lens["maps"]
        matrix = maps[block] if not isinstance(maps, dict) else maps.get(block, maps.get(str(block)))
        if matrix is None:
            raise ValueError(f"J-lens missing block {block}")
        h = h @ matrix.to(h.device, h.dtype).T
    with torch.no_grad():
        logits = unembed(engine.model, h).float()
    valid = []
    special = set(engine.tok.all_special_ids)
    # Decode only progressively larger top sets; sufficient for top-k valid
    # tokens, without decoding a 262k-token vocabulary for every observation.
    for token in logits.argsort(descending=True).tolist():
        if token not in special and any(c.isalnum() for c in engine.tok.decode([token])):
            valid.append(token)
            if len(valid) == k:
                return valid
    raise ValueError("not enough non-special alphanumeric seed tokens")


def rollout(engine,block,state,seeds,**kwargs):
    """Use the independently selected E/L instrument throughout K protocols."""
    import torch
    from r1b_g_lens import rollout as native_rollout
    variant=getattr(engine,"_g_self_variant","E")
    if variant=="L":
        matrix=engine._g_jlens["maps"][block]
        norm=engine._g_penultimate_norm
        state=torch.nn.functional.normalize(torch.as_tensor(state).float()@matrix.T,dim=-1)*norm
        if kwargs.get("prior_state") is not None:
            kwargs["prior_state"]=torch.nn.functional.normalize(torch.as_tensor(kwargs["prior_state"]).float()@matrix.T,dim=-1)*norm
    kwargs["variant"]=variant
    return native_rollout(engine,block,state,seeds,**kwargs)


def _neutral(tok, mock, k):
    from r1b_k_labels import NEUTRAL
    words = list("abcdefghij") if mock else NEUTRAL
    ids = [tok(x, add_special_tokens=False)["input_ids"] for x in words]
    valid = [x[0] for x in ids if len(x) == 1 and x[0] not in tok.all_special_ids]
    if len(valid) < k:
        # Fixed extension list, chosen without looking at answers or model logits.
        for word in ("the", "and", "of", "to", "a", "in", "is", "that", "it", "was",
                     "this", "with", "as", "for", "on", "at", "by", "from", "be", "an"):
            x = tok(word, add_special_tokens=False)["input_ids"]
            if len(x) == 1 and x[0] not in valid and x[0] not in tok.all_special_ids:
                valid.append(x[0])
    if len(valid) < k:
        raise ValueError("not enough target-tokenized neutral single-token seeds")
    return valid[:k]


class _Channels:
    """Read activated MLP values and CETT without uniform-width assumptions."""
    def __init__(self, engine, populations, refs=None):
        self.engine, self.populations, self.refs = engine, populations, refs
        self.handles, self.pending, self.steps = [], {}, []

    def __enter__(self):
        import torch
        from r1b_g_adapter import layers_of
        for layer, block in enumerate(layers_of(self.engine.model)):
            relevant = {name: [int(j) for l, j in pairs if int(l) == layer]
                        for name, pairs in self.populations.items()}
            relevant = {name: ix for name, ix in relevant.items() if ix}
            if not relevant:
                continue
            norms = block.mlp.down_proj.weight.detach().float().norm(dim=0)
            def hook(mod, args, output, l=layer, groups=relevant, wn=norms):
                z = args[0][:, -1].detach().float()
                den = output[:, -1].detach().float().norm(dim=-1).clamp_min(1e-9)
                channels = {}
                for name, indexes in groups.items():
                    ix = torch.tensor(indexes, device=z.device)
                    values = z[:, ix].abs() * wn[ix] / den[:, None]
                    if self.refs is not None:
                        flat = torch.tensor([self.engine.layout.encode(l, j) for j in indexes])
                        mu = torch.as_tensor(self.refs["c_mu"])[flat].to(z.device)
                        sd = torch.as_tensor(self.refs["c_sd"])[flat].to(z.device).clamp_min(1e-9)
                        values = (values - mu) / sd
                    channels[name] = values.sum(-1).cpu()
                self.pending[l] = channels
            self.handles.append(block.mlp.down_proj.register_forward_hook(hook))
        # Last block runs after all MLP hooks, so this records one vector/forward.
        last = layers_of(self.engine.model)[-1]
        def finish(*args):
            result = {}
            for name, pairs in self.populations.items():
                chunks = [d[name] for d in self.pending.values() if name in d]
                if chunks:
                    result[name] = sum(chunks) / len(pairs)
            self.steps.append(result)
            self.pending = {}
        self.handles.append(last.register_forward_hook(finish))
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles = []


class NativeShares:
    """Signed native direct-logit shares with frozen token-wise RMS factors.

    A dense MLP write is carried along the direct residual path, including all
    subsequent layer scalars. Native post-FF and final RMS factors are frozen
    at each real/prior forward. The secant factor of the final softcap gives an
    exact additive decomposition of its centered logit; attention/PLE writes
    remain outside the MLP denominator, as in the inherited DLA definition.
    Only the latest real/prior activation pair is retained on the device.
    """
    def __init__(self,engine,populations):
        self.engine,self.populations=engine,populations
        self.handles=[];self.pending={};self.pair=[];self.values={name:{} for name in populations}
    def __enter__(self):
        import torch
        from r1b_g_adapter import text_model
        scalars=[]
        for layer in self.engine.layers:
            if getattr(layer,"enable_moe_block",False):
                raise ValueError("dense-neuron DLA requires explicit expert attribution for MoE blocks")
            scalars.append(float(torch.as_tensor(layer.layer_scalar)))
        self.scale=[math.prod(scalars[i:]) for i in range(len(scalars))]
        for index,layer in enumerate(self.engine.layers):
            def capture(module,args,output,index=index):
                self.pending[index]=(args[0][0,-1].detach().float(),output[0,-1].detach().float())
            self.handles.append(layer.mlp.down_proj.register_forward_hook(capture))
        def final(module,args):
            self.pair.append((self.pending,args[0][0,-1].detach().float()));self.pending={}
            if len(self.pair)>2:raise RuntimeError("DLA observer did not consume the preceding real/prior pair")
        self.norm=text_model(self.engine.model).norm
        self.handles.append(self.norm.register_forward_pre_hook(final))
        return self
    def observe(self,seed_index,step,token):
        import torch
        if len(self.pair)!=2:raise RuntimeError("DLA requires one real and one prior native pass per selected token")
        head=self.engine.model.get_output_embeddings().weight.detach()
        stream=[]
        for layers,h in self.pair:
            rms=(h.square().mean()+self.norm.eps).sqrt()
            weight=self.norm.weight.detach().float() if self.norm.with_scale else torch.ones_like(h)
            final_scale=weight/rms
            cap=getattr(self.engine.cfg,"final_logit_softcapping",None)
            mean=torch.zeros_like(h);token_factor=1.
            for begin in range(0,len(head),4096):
                chunk=head[begin:begin+4096].float();logit=chunk@(h*final_scale)
                factors=torch.ones_like(logit) if cap is None else torch.where(logit.abs()>1e-7,cap*torch.tanh(logit/cap)/logit,torch.ones_like(logit))
                mean+=chunk.T@factors
                if begin<=int(token)<begin+len(chunk):token_factor=factors[int(token)-begin]
            u=head[int(token)].float()*token_factor-mean/len(head)
            stream.append((layers,u*final_scale))
        denominator=torch.zeros((),device=head.device);parts={name:torch.zeros_like(denominator) for name in self.populations}
        for index,layer in enumerate(self.engine.layers):
            terms=[];norm=layer.post_feedforward_layernorm
            for values,u in stream:
                z,down=values[index]
                nweight=norm.weight.detach().float() if norm.with_scale else torch.ones_like(down)
                factor=nweight/(down.square().mean()+norm.eps).sqrt()
                coefficient=(u*factor*self.scale[index])@layer.mlp.down_proj.weight.detach().float()
                terms.append(z*coefficient)
            difference=terms[0]-terms[1];denominator+=difference.abs().sum()
            for name,pairs in self.populations.items():
                neurons=[int(j) for l,j in pairs if int(l)==index]
                if neurons:parts[name]+=difference[neurons].sum()
        for name,value in parts.items():
            self.values[name].setdefault(seed_index,[]).append(float(value/denominator.clamp_min(1e-12)))
        self.pair=[]
    def row(self,index):
        import torch
        return {name:torch.tensor(rows[index]) for name,rows in self.values.items() if index in rows}
    def __exit__(self,*args):
        for handle in self.handles:handle.remove()
        self.handles=[];self.pending={};self.pair=[]


def _self_kl(engine, prompt, seed, tokens, lens_logp):
    import torch
    seq = prompt + [int(seed)] + list(tokens[:-1])
    with torch.no_grad():
        q = engine.model(torch.tensor([seq], device=engine.device), logits_to_keep=len(tokens)).logits[0].float().log_softmax(-1)
        p = lens_logp.to(q.device)
        return (q.exp() * (q - p)).sum(-1).cpu()


def _clean_json(value):
    import r1b_core as C
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if hasattr(value, "tolist"):
        return _clean_json(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return C.coded("NaN:undefined")
    return value


def verify_chain(engine, block, tokens, state, prior, frame):
    """Inherited inverse-free verifier: cosine of mean chain state minus prior."""
    import torch
    if not tokens:
        raise ValueError("cannot verify an empty chain")
    hidden = engine.hidden([frame[:-1] + list(map(int, tokens))], [block])[0][block]
    s = hidden[len(frame)-1:].float().mean(0) - prior.float()
    return float(torch.nn.functional.cosine_similarity(s, state.float() - prior.float(), dim=0))


def verifier_controls(ctx, engine, records, sources, frame, correct, by_id):
    """K4 verifier thresholds, one-revision GVRA, planted true/foreign cases."""
    import numpy as np
    import torch
    from r1b_k_gvra import gvra, summary, error_rates
    rng, thresholds, values, groups = np.random.default_rng(52), {}, {}, {}
    for r in records:
        if r["variant"] in ("V1", "V2"):
            groups.setdefault((r["variant"], r["block"], r["ptype"]), []).append(r)
    for (variant, block, ptype), rows in groups.items():
        src = sources[ptype]
        for length in (4, 8, 12, 16):
            if length > len(rows[0]["tok"]) + 1:
                continue
            scores, shuffled = [], []
            for r in rows:
                index = src["ids"].index(r["item"])
                state = src["h"][block][index]
                prior = (src["h"][block].sum(0) - state) / (len(src["ids"]) - 1)
                chain = [r["seed"]] + r["tok"][:length-1]
                key = f"verifier_{ptype}_{block}_{r['item']}_{variant}_{r['seed_rank']}_{length}"
                score = _checkpoint(ctx,key, lambda: verify_chain(engine, block, chain, state, prior, frame))
                scores.append(score)
                if variant == "V1":
                    sh = rng.permutation(chain).tolist()
                    shuffled.append(_checkpoint(ctx,key + "_shuffled", lambda: verify_chain(engine, block, sh, state, prior, frame)))
                    r.setdefault("ver", {})[length] = score
            values[(variant, block, ptype, length)] = scores
            if shuffled:
                values[("SH", block, ptype, length)] = shuffled
    for variant, block, ptype, length in values:
        if variant == "V1":
            null = values.get(("V2", block, ptype, length), []) + values.get(("SH", block, ptype, length), [])
            if null:
                thresholds[(block, ptype, length)] = list(map(float, np.percentile(null, [80, 95])))
    gv, breakoffs = {}, []
    for (variant, block, ptype), rows in groups.items():
        if variant != "V1":
            continue
        lengths = sorted(k[2] for k in thresholds if k[:2] == (block, ptype))
        if not lengths:
            continue
        length = lengths[-1]
        q80, q95 = thresholds[(block, ptype, length)]
        cells = {}
        for r in rows:
            cells.setdefault(r["item"], []).append(r)
            for size in lengths:
                if r["ver"][size] <= thresholds[(block, ptype, size)][0]:
                    r.setdefault("k_ver", size)
            breakoffs.append(dict(item=r["item"], block=block, ptype=ptype, seed_rank=r["seed_rank"], k_ver=r.get("k_ver")))
        def generate(item, attempt):
            ranked = sorted(cells[item], key=lambda r: r["seed_rank"])
            return ranked[attempt] if attempt < len(ranked) else None
        def verify(item, row):
            score = row.get("rev", row["ver"][length])
            return ("correct" if score > q95 else "minor fix needed" if score > q80 else "critically flawed"), score
        def revise(item, row, attempt):
            if "alt" not in row:
                return row
            revised = dict(row)
            index = int(torch.as_tensor(row["e"]).argmin())
            tokens = list(row["tok"])
            tokens[index] = int(row["alt"][index])
            src = sources[ptype]; k = src["ids"].index(item); h = src["h"][block][k]
            mu = (src["h"][block].sum(0)-h)/(len(src["ids"])-1)
            revised["rev"] = verify_chain(engine, block, [row["seed"]]+tokens[:length-1], h, mu, frame)
            revised["revised_tokens"] = tokens
            return revised
        result = gvra(list(cells), generate, verify, revise, limit=3)
        gv[f"{block}:{ptype}"] = {"summary": summary(result), "accepted": {i: r["status"] for i,r in result.items()},
                                  "revision_available": all("alt" in r for r in rows)}
    planted = {}
    # Fixed independent raw sentences; truth is planted, never model-generated.
    planted_sequences = [engine.encode(f"A quiet note ends with the number {101 + 11*i}", chat=False)
                         for i in range(4 if ctx.mock else 64)]
    for block in sorted({r["block"] for r in records}):
        hs = [r[block][-1] for r in engine.hidden(planted_sequences, [block])]
        mu = torch.stack(hs).mean(0)
        q95 = thresholds.get((block, "t0", max((k[2] for k in thresholds if k[:2] == (block,"t0")), default=16)), [float("inf")]*2)[1]
        true, foreign = [], []
        for i,h in enumerate(hs):
            true.append(verify_chain(engine, block, planted_sequences[i][-8:], h, mu, frame) > q95)
            foreign.append(verify_chain(engine, block, planted_sequences[(i+1)%len(hs)][-8:], h, mu, frame) > q95)
        planted[block] = error_rates(true, foreign)
    sensitivity = float(np.mean([r["sensitivity"] for r in planted.values()])) if planted else None
    false_accept = float(np.mean([r["false_accept"] for r in planted.values()])) if planted else None
    diffs = {}
    for block in sorted({r["block"] for r in records}):
        hit = {}
        for variant in ("V1", "V2"):
            selected = [r for r in records if r["variant"] == variant and r["block"] == block and r["ptype"] == "t0" and r["item"] in correct]
            hit[variant] = np.mean([any(c == "TRUE-INT" for r in selected if r["item"] == i for _,_,c in r["claims"]) for i in correct]) if correct else float("nan")
        diffs[block] = float(hit["V1"]-hit["V2"])
    inner = mapped_blocks(len(engine.layout.widths))[1:-1] or sorted(diffs)
    available = [b for b in inner if math.isfinite(diffs[b])]
    bstar = max(available, key=lambda b: (diffs[b], -b)) if available else None
    reliable = sensitivity is not None and sensitivity >= .5 and false_accept <= .2
    return dict(thresholds={str(k):v for k,v in thresholds.items()}, gvra=gv, planted=planted,
                G_K6="verifier reliable" if reliable and not ctx.mock else "verifier unreliable" if not ctx.mock else "MOCK: no scientific gate",
                sensitivity=sensitivity, false_accept=false_accept, b_star=bstar, anyhit_diff=diffs, breakoffs=breakoffs)


def crossfit_probe(ctx, engine, raw_items, blocks):
    """K4p ridge with item-blocked outer folds, inner CV and template permutations."""
    import numpy as np
    from r1b_k_s2 import _ridge
    items = [normalize_item(x) for x in raw_items if x.get("split") in ("A-scr", "A-lens") and not x.get("cal") and x.get("I")]
    items = items[:(12 if ctx.mock else 4200)]
    folds = np.array([int(hashlib.sha256(str(x["id"]).encode()).hexdigest(),16)%5 for x in items])
    if len(items) < 10 or len(set(folds)) < 3:
        return dict(status="blocked", reason="K4p needs at least ten items across three item-blocked folds", res={})
    y = np.array([x["I"][0]%10 for x in items]); y2 = np.array([(x["I"][1] if len(x["I"])>1 else x["A"])%10 for x in items])
    template = np.array([x.get("template", "unspecified") for x in items])
    base = np.array([[int(c==d) for p in (x["P"][:4]+[0]*4)[:4] for c in str(abs(p)).zfill(2) for d in "0123456789"] +
                     [int(o==s) for o in (x.get("ops",[])+["+"]*3)[:3] for s in "+-*"] for x in items], float)
    hidden = _checkpoint(ctx,"k4p_hidden", lambda: engine.hidden([engine.encode(x["prompt"]) for x in items],blocks))
    def predict(X, labels):
        result=np.zeros(len(labels)); Y=np.eye(10)[labels]
        for fold in sorted(set(folds)):
            train,test=folds!=fold,folds==fold
            best=(-float("inf"), .1)
            for lam in (.1,10.,1000.):
                accuracy=[]
                for inner in (0,1):
                    tr=train & ((np.arange(len(labels))%2)!=inner); te=train & ((np.arange(len(labels))%2)==inner)
                    if not tr.any() or not te.any(): continue
                    mu,sd=X[tr].mean(0),X[tr].std(0)+1e-9
                    pred=_ridge((X[tr]-mu)/sd,Y[tr],lam,(X[te]-mu)/sd).argmax(1)
                    accuracy.append((pred==labels[te]).mean())
                score=np.mean(accuracy) if accuracy else -float("inf")
                if score>best[0]: best=(score,lam)
            mu,sd=X[train].mean(0),X[train].std(0)+1e-9
            result[test]=(_ridge((X[train]-mu)/sd,Y[train],best[1],(X[test]-mu)/sd).argmax(1)==labels[test])
        return result
    cb=predict(base,y); results={}; generator=np.random.default_rng(1301)
    for block in blocks:
        def fit(block=block):
            X=np.stack([r[block][-1].numpy() for r in hidden]).astype(float)
            permutations=[]
            for _ in range(3 if ctx.mock else 200):
                yp=y.copy()
                for t in np.unique(template):
                    mask=template==t; yp[mask]=generator.permutation(yp[mask])
                permutations.append(float(predict(X,yp).mean()-cb.mean()))
            return dict(correct_probe=predict(X,y),correct_base=cb,correct_probe_v2=predict(X,y2),perm=np.array(permutations))
        results[block]=_checkpoint(ctx,f"k4p_probe_{block}",fit)
    return dict(status="measured",res=results,items=[x["id"] for x in items],folds=folds,
                normalization="outer/inner training folds only",mock=ctx.mock)


def channel_bases(engine, source, lens, hn):
    """Target-derived write, J-workspace and PCA bases; no imported SO-AX axes."""
    import torch
    bases={}
    if not source:
        return bases
    for block,X in source["h"].items():
        d=X.shape[-1]; rank=min(16,d)
        centered=X-X.mean(0)
        if centered.shape[0]>1:
            _,s,v=torch.linalg.svd(centered.float(),full_matrices=False)
            r=min(rank,int((s>1e-7*s.max().clamp_min(1e-7)).sum()))
            if r: bases[f"pc16_{block:02d}"]=(block,v[:r].T.contiguous())
        writes=[engine.layers[int(l)].mlp.down_proj.weight[:,int(j)].detach().float().cpu()
                for l,j in hn.get("H_op",[]) if int(l)<=block]
        if writes:
            u,s,_=torch.linalg.svd(torch.stack(writes,1),full_matrices=False)
            r=min(rank,int((s>1e-7*s.max()).sum()))
            if r: bases[f"Hsub{block}"]=(block,u[:,:r])
        if lens is not None:
            maps=lens["maps"]; matrix=maps[block] if not isinstance(maps,dict) else maps.get(block,maps.get(str(block)))
            if matrix is not None:
                _,_,v=torch.linalg.svd(matrix.float(),full_matrices=False)
                bases[f"Jws{block}"]=(block,v[:rank].T.contiguous())
    return bases


def throughput(ctx, engine, source, frame, lens, hn, by_id, gates, correct):
    """K7b complete model-independent subspace arm, including rank-matched nulls."""
    import torch
    if not source or not correct:
        return dict(status="blocked",reason="no correctly answered items",rows=[],bases={})
    bases=channel_bases(engine,source,lens,hn); rows=[]
    inner=mapped_blocks(len(engine.layout.widths))[1:-1] or mapped_blocks(len(engine.layout.widths))
    for block in inner:
        states=source["h"][block]; d=states.shape[-1]
        named={name:(basis) for name,(b,basis) in bases.items() if b==block}
        channels={("pc16" if name.startswith("pc16_") else "Hsub" if name.startswith("Hsub") else "Jws"):basis for name,basis in named.items()}
        channels["full"]=None
        for rank in sorted({1,min(16,d)} | {b.shape[1] for b in named.values()}):
            for n in range(2 if ctx.mock else 19):
                generator=torch.Generator().manual_seed(ctx.key("g_k7_null",block,rank,n))
                channels[f"r{rank}_{n}"]=torch.linalg.qr(torch.randn(d,rank,generator=generator))[0]
        for name,basis in channels.items():
            for ident in correct[:(2 if ctx.mock else 128)]:
                index=source["ids"].index(ident); h=states[index]; mu=(states.sum(0)-h)/(len(states)-1)
                projected=h if basis is None else mu+basis@(basis.T@(h-mu))
                seeds=_seeds(engine,h,block,lens,2 if ctx.mock else 3)
                def cell():
                    out=rollout(engine,block,projected,seeds,steps=8 if ctx.mock else 16,prior_state=mu,frame_ids=frame)
                    result=[]
                    for i,seed in enumerate(seeds):
                        scored=score_tokens(engine.tok,[seed]+out["tok"][i].tolist(),by_id[ident],OWN=gates[ident]["own"])
                        claims=scored["claims"]
                        steps=[s for s,_,c in claims if c in ("TRUE-INT","ANSWER","PROMPT")]
                        result.append(dict(item=ident,block=block,ch=name,dim=d if basis is None else basis.shape[1],
                            fwd=float(any(c in ("TRUE-INT","ANSWER") for _,_,c in claims)),bwd=float(any(c=="PROMPT" for _,_,c in claims)),
                            horizon=max(steps,default=0),claims=claims,tok=out["tok"][i],seed=seed))
                    return result
                rows.extend(_checkpoint(ctx,f"k7_{block}_{name}_{ident}",cell))
    return dict(status="measured",rows=rows,bases=bases,axes="target-derived only; SO-AX axes unavailable",mock=ctx.mock)


def _read_channels(engine, seq, populations, refs, blocks=(), setup=None):
    import torch
    flat={name:[engine.layout.encode(int(l),int(j)) for l,j in pairs] for name,pairs in populations.items() if pairs}
    row=engine.teacher_force([seq],populations=flat,blocks=blocks,setup=setup)["rows"][0]
    positions={flat_id:i for i,flat_id in enumerate(row["neuron_ids"])}; result={}
    for name,ids in flat.items():
        columns=[positions[i] for i in ids]
        c=row["cett"][:,columns]
        if refs is not None:
            c=(c-torch.as_tensor(refs["c_mu"])[ids]) / torch.as_tensor(refs["c_sd"])[ids].clamp_min(1e-9)
        result[name]=c.mean(-1)
    row["aX"]=result
    return row


def own_text(ctx, engine, raw_items, populations, refs):
    """K8 observations: own arithmetic/QA text with step-aligned channels."""
    from r1b_k_labels import parse_number
    rows=[]; data=ctx.data()
    ach=[normalize_item(x) for x in raw_items if not x.get("cal") and x.get("split") in ("A-lens","A-bnd")]
    tq=[x for x in data.get("trivia",[]) if x.get("split") in ("S4b","S4c","S_sess","S_L")]
    targets=[("ACH",x) for x in ach[:(2 if ctx.mock else 128)]]+[("TQA",x) for x in tq[:(2 if ctx.mock else 128)]]
    for family,item in targets:
        ident=item.get("id",item.get("qid")); text=item["prompt"] if family=="ACH" else "Question: "+item["question"]+" Answer in two or three sentences."
        def cell():
            prompt=engine.encode(text)
            answer=engine.decode([prompt],keys=[ctx.key("g_k8_own",family,ident)],max_new=8 if ctx.mock else 64,temperature=.7)[0]
            seq=prompt+answer["ids"]
            read=_read_channels(engine,seq,populations,refs)
            if family=="ACH":
                correct=parse_number(answer["text"])==item["A"]
                scoring=score_tokens(engine.tok,answer["ids"],item)
            else:
                from r1b_k_labels import alias_match, answer_span
                correct=alias_match(answer_span(answer["text"]),item["aliases"])=="correct"
                scoring={}
            return dict(item=ident,fam=family,prompt_ids=prompt,tok=answer["ids"],text=answer["text"],correct=correct,
                        aX={name:v[len(prompt):] for name,v in read["aX"].items()},
                        entropy=read["entropy"][max(0,len(prompt)-1):],**scoring)
        rows.append(_checkpoint(ctx,f"k8_own_{family}_{ident}",cell))
    return dict(status="measured" if rows else "blocked",rows=rows,
                reason=None if rows else "no arithmetic/session QA raw items",mock=ctx.mock)


def causal_chain_controls(ctx,engine,source,frame,lens,hn,refs,by_id,gates,correct,bstar):
    """K6c clamp/read-state ablation and K6g forced-content positive controls."""
    import torch
    import numpy as np
    if not correct or not source or refs is None:
        return dict(status="blocked",reason="correct-item positive controls or reference moments unavailable",k6c=[],k6g=[])
    outputs=[]; content=[]
    populations={"H":hn.get("H_op",[]),**{f"MC{k}":v for k,v in enumerate(hn.get("RS39mc",[]))}}
    populations={n:v for n,v in populations.items() if v}
    flat={name:[engine.layout.encode(int(l),int(j)) for l,j in pairs] for name,pairs in populations.items()}
    selected=correct[:(2 if ctx.mock else 128)]
    inner=mapped_blocks(len(engine.layout.widths))[1:-1] or mapped_blocks(len(engine.layout.widths))
    for block in inner:
        states=source["h"][block]
        for ident in selected:
            ix=source["ids"].index(ident); h=states[ix]; mu=(states.sum(0)-h)/(len(states)-1)
            seeds=_seeds(engine,h,block,lens,2 if ctx.mock else 4)
            def observe(name,patched,setup=None):
                with setup() if setup is not None else contextlib.nullcontext():
                    out=rollout(engine,block,patched,seeds,steps=8 if ctx.mock else 16,prior_state=mu,frame_ids=frame,keep_first=True)
                rows=[]
                for k,seed in enumerate(seeds):
                    rows.append(dict(item=ident,block=block,ptype="t0",variant=name,seed=seed,seed_rank=k,tok=out["tok"][k].tolist(),
                                     **score_tokens(engine.tok,[seed]+out["tok"][k].tolist(),by_id[ident],OWN=gates[ident]["own"])))
                return {"rows":rows,"first":out.get("first")}
            baseline=_checkpoint(ctx,f"k6c_{block}_{ident}_none",lambda:observe("none",h))
            outputs.append(dict(cond="none",block=block,chains=baseline["rows"],css=[]))
            for name,pairs in populations.items():
                ids=flat[name]; means=torch.as_tensor(refs["z_mu"])[ids]
                # Clamp both real and prior at every chain token, matching K6c.
                setup=lambda ids=ids,means=means:engine.intervene(ids,"mean",means)
                clamped=_checkpoint(ctx,f"k6c_{block}_{ident}_clamp_{name}",lambda:observe("clamp_"+name,h,setup))
                writes=_checkpoint(ctx,f"read_writes_{ident}_{name}_{block}",lambda:direct_writes(engine,source["prompts"][ix],{name:pairs},refs,[block]))
                changed=h-writes[block][name]
                ablated=_checkpoint(ctx,f"k6c_{block}_{ident}_ablate_{name}",lambda:observe("ablate_"+name,changed))
                for kind,result in (("clamp",clamped),("ablate",ablated)):
                    css=[]
                    if result["first"] and baseline["first"]:
                        f,f0=result["first"],baseline["first"]
                        mask=f0["plaus"]
                        d1=((f["lr"]-f["lp"])-(f0["lr"]-f0["lp"])).masked_fill(~mask,0)
                        d2=(f["lr"]-f0["lr"]).masked_fill(~mask,0)
                        css=(d1.norm(dim=-1)/d2.norm(dim=-1).clamp_min(1e-12)).tolist()
                    outputs.append(dict(cond=f"{kind}_{name}",block=block,chains=result["rows"],css=css))
    if bstar is not None:
        rng=np.random.default_rng(1202); states=source["h"][bstar]
        candidates=[i for i in correct if by_id[i]["I"] and 10<=by_id[i]["I"][0]<=999][:(2 if ctx.mock else 400)]
        for ident in candidates:
            item=by_id[ident]; value=item["I"][0]; digits=len(str(value)); true=engine.encode(" "+str(value),chat=False)
            excluded=set(item["P"]+item["I"]+[item["A"]]+gates[ident]["own"])
            wrong=None; w=None
            for _ in range(2000):
                trial=int(rng.integers(10**(digits-1),10**digits)); tokens=engine.encode(" "+str(trial),chat=False)
                if trial not in excluded and str(trial)[0]!=str(value)[0] and len(tokens)==len(true):
                    wrong,w=tokens,trial;break
            if wrong is None: continue
            neutral=_neutral(engine.tok,ctx.mock,min(len(true),10))
            neutral=(neutral*(1+len(true)//len(neutral)))[:len(true)]
            ix=source["ids"].index(ident); h=states[ix]; mu=(states.sum(0)-h)/(len(states)-1)
            seeds=_seeds(engine,h,bstar,lens,2 if ctx.mock else 4)
            for condition,forced in (("correct",true),("wrong",wrong),("neutral",neutral)):
                def cell():
                    out=rollout(engine,bstar,h,seeds,steps=len(forced)+(4 if ctx.mock else 9),prior_state=mu,frame_ids=frame,forced=[forced]*len(seeds))
                    return [dict(item=ident,block=bstar,ptype="t0",variant=condition,seed=seed,seed_rank=k,tok=out["tok"][k].tolist(),w=w,
                                 forced=forced,**score_tokens(engine.tok,[seed]+out["tok"][k].tolist(),item,OWN=gates[ident]["own"])) for k,seed in enumerate(seeds)]
                content.extend(_checkpoint(ctx,f"k6g_{ident}_{condition}",cell))
    return dict(status="measured",k6c=outputs,k6g=content,b_star=bstar,
                specificity_available=len(hn.get("RS39mc",[]))>=(2 if ctx.mock else 20),
                G_K19="not testable (too few items)" if len({r["item"] for r in content})<(2 if ctx.mock else 75) else None)


def direct_writes(engine,seq,populations,refs,blocks):
    """Mean-centered MLP writes along Gemma's direct residual path at t0."""
    import torch
    handles=[];captured={}
    for index,layer in enumerate(engine.layers):
        relevant={name:[int(j) for l,j in pairs if int(l)==index] for name,pairs in populations.items()}
        relevant={name:ids for name,ids in relevant.items() if ids}
        if not relevant:continue
        def capture(module,args,output,index=index,layer=layer,relevant=relevant):
            z=args[0][0,-1].float();out=output[0,-1].float();norm=layer.post_feedforward_layernorm
            weight=norm.weight.detach().float() if norm.with_scale else torch.ones_like(out)
            scale=weight/(out.square().mean()+norm.eps).sqrt()
            values={}
            for name,ids in relevant.items():
                flat=[engine.layout.encode(index,j) for j in ids]
                dz=z[ids]-torch.as_tensor(refs["z_mu"])[flat].to(z.device)
                values[name]=((layer.mlp.down_proj.weight[:,ids].float()@dz)*scale).detach().cpu()
            captured[index]=values
        handles.append(layer.mlp.down_proj.register_forward_hook(capture))
    try:engine.hidden([seq],[max(blocks)])
    finally:
        for handle in handles:handle.remove()
    running={name:torch.zeros(int(engine.cfg.hidden_size)) for name in populations};result={}
    for index,layer in enumerate(engine.layers):
        for name in running:
            running[name]=(running[name]+captured.get(index,{}).get(name,0))*float(torch.as_tensor(layer.layer_scalar))
        if index in blocks:result[index]={name:value.clone() for name,value in running.items()}
        if index>=max(blocks):break
    return result


def read_state_association(ctx,engine,source,records,populations,refs,by_id):
    """K6b(ii) direct-write rho versus per-cell confabulation with item bootstrap."""
    import numpy as np
    from r1b_k_ana import qshare,pboot
    from r1b_k_base import boot_stat,ci,p_rank,p_pi,p_iut,val,dist
    if not populations.get("H") or refs is None:return dict(status="blocked",reason="H population/reference moments unavailable")
    groups={}
    for row in records:
        if row["variant"]=="V1" and row["ptype"]=="t0":groups.setdefault((row["item"],row["block"]),[]).append(qshare(row["claims"]))
    cells=[(i,b,float(np.nanmean(v))) for (i,b),v in groups.items() if np.isfinite(v).any()]
    if len(cells)<5:return dict(status="not_testable",reason="fewer than five cells with completed claims")
    rho={name:[] for name in populations}
    for ident,block,_ in cells:
        ix=source["ids"].index(ident);states=source["h"][block];h=states[ix];mu=(states.sum(0)-h)/(len(states)-1)
        writes=_checkpoint(ctx,f"rho_writes_{ident}_{block}",lambda:direct_writes(engine,source["prompts"][ix],populations,refs,[block]))
        for name in populations:rho[name].append(float(writes[block][name].norm()/(h-mu).norm().clamp_min(1e-9)))
    rho={name:np.array(v) for name,v in rho.items()};y=np.array([q for _,_,q in cells]);depth=np.array([by_id[i].get("nops",0) for i,_,_ in cells]);blocks=np.array([b for _,b,_ in cells]);ids=np.array([i for i,_,_ in cells])
    def beta(values,indexes):
        x=values[indexes];Z=np.c_[np.ones(len(indexes)),(x-x.mean())/(x.std()+1e-9),depth[indexes],blocks[indexes]]
        return float(np.linalg.solve(Z.T@Z+1e-9*np.eye(Z.shape[1]),Z.T@y[indexes])[1])
    b=beta(rho["H"],np.arange(len(y)));draws=boot_stat(len(y),lambda ix:beta(rho["H"],ix),B=10 if ctx.mock else 2000,groups=ids)
    controls=[beta(v,np.arange(len(y))) for name,v in rho.items() if name!="H"]
    pe=pboot(draws);pr=p_rank(b,controls) if controls else 1.;ppi=p_pi(b,controls) if len(controls)>=2 else 1.
    return dict(status="measured",beta=val(b),ci=ci(draws),p_effect=pe,p_rank=pr,p_iut=p_iut(pe,pr),p_iut_pi=p_iut(pe,ppi),rho_H=dist(rho["H"]),n_cells=len(cells),
                direct_write_definition="frozen native post-FF RMS factors and accumulated layer scalars")


@contextlib.contextmanager
def additive(engine,block,vector,start=0,real_stream_only=False):
    """Native residual additions, optionally on alternating real rollout passes."""
    call=[0]
    def hook(module,args,output):
        h=output[0] if isinstance(output,tuple) else output
        enabled=not real_stream_only or call[0]%2==0;call[0]+=1
        if not enabled:return output
        out=h.clone();out[:,start:]+=vector.to(out.device,out.dtype)
        return (out,)+output[1:] if isinstance(output,tuple) else out
    handle=engine.layers[block].register_forward_hook(hook)
    try:yield
    finally:handle.remove()


def reinjection_calibration(ctx,engine,source,raw_items):
    import numpy as np
    import torch
    import r1b_core as core
    calibration=[normalize_item(i) for i in raw_items if i.get("cal")][:(2 if ctx.mock else 32)]
    refs=ctx.data().get("ref_k16",[])
    texts=[x if isinstance(x,str) else x["text"] for x in refs]
    if not calibration or not texts or not source:
        return dict(status="blocked",reason="independent A-cal states or REF-K16 missing",blocks={})
    blocks=mapped_blocks(len(engine.layers))[1:-1] or mapped_blocks(len(engine.layers))
    states=engine.hidden([engine.encode(i["prompt"]) for i in calibration],blocks)
    result={}
    for block in blocks:
        mu=source["h"][block].mean(0);deviations=torch.stack([x[block][-1] for x in states])-mu
        def fit():
            def damage(beta):
                return float(np.mean([engine.damage(texts,setup=lambda d=d:additive(engine,block,beta*d))["damage"] for d in deviations]))
            out=core.calibrate(damage,.05,.05,form="push",max_eval=6,lo=.045,hi=.055)
            return dict(beta=out["f"] if out["in_band"] else None,cal=out)
        result[block]=_checkpoint(ctx,f"reinjection_cal_{block}",fit)
    return dict(status="measured",blocks=result,calibration_ids=[i["id"] for i in calibration],target_damage=.05)


class FrozenQK:
    """Record/replay native Gemma attention probabilities, preserving KV sharing.

    Only the attention product is replayed; Gemma still owns Q/K normalization,
    RoPE, layer types, masks, PLE and shared-key/value selection.
    """
    def __init__(self,engine,probabilities=None,capture_heads=False):
        self.engine=engine;self.probabilities={} if probabilities is None else probabilities
        self.replay=probabilities is not None
        self.capture_heads=capture_heads;self.heads={}
    def __enter__(self):
        from transformers import AttentionInterface
        from transformers.models.gemma4 import modeling_gemma4 as native
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        def attention(module,query,key,value,attention_mask,**kwargs):
            if not self.replay:
                out,probs=native.eager_attention_forward(module,query,key,value,attention_mask,**kwargs)
                self.probabilities[module.layer_idx]=probs.detach()
                if self.capture_heads:
                    self.heads[module.layer_idx]=(out,probs.detach(),native.repeat_kv(value,module.num_key_value_groups).detach())
                return out,probs
            probs=self.probabilities[module.layer_idx]
            values=native.repeat_kv(value,module.num_key_value_groups)
            if probs.shape[-1]!=values.shape[-2] or probs.shape[-2]!=query.shape[-2]:
                raise ValueError("frozen-QK replay shape differs from recorded native pass")
            return (probs.to(values.dtype)@values).transpose(1,2).contiguous(),probs
        self.name="r1b_g_frozen_qk"
        self.registry=AttentionInterface._global_mapping
        self.mask_registry=type(ALL_MASK_ATTENTION_FUNCTIONS)._global_mapping
        self.previous=self.registry.get(self.name);self.previous_mask=self.mask_registry.get(self.name)
        AttentionInterface.register(self.name,attention)
        # SDPA may omit an explicit causal mask and set is_causal internally;
        # eager attention needs the full mask. Reuse Gemma's eager mask factory.
        ALL_MASK_ATTENTION_FUNCTIONS.register(self.name,ALL_MASK_ATTENTION_FUNCTIONS["eager"])
        self.old=self.engine.cfg._attn_implementation
        self.engine.cfg._attn_implementation=self.name
        return self
    def __exit__(self,*args):
        self.engine.cfg._attn_implementation=self.old
        for registry,previous in ((self.registry,self.previous),(self.mask_registry,self.previous_mask)):
            if previous is None:registry.pop(self.name,None)
            else:registry[self.name]=previous


def fd_transfer(engine,seq,block,position,vector,read_blocks,frozen=None):
    """Central forward difference, no inverse or residual reconstruction."""
    def read(sign):
        with engine.patch_state(block,sign*vector,position,add=True):
            return engine.hidden([seq],read_blocks)[0]
    with FrozenQK(engine,frozen) if frozen is not None else contextlib.nullcontext():
        plus,minus=read(1),read(-1)
    return {b:(plus[b]-minus[b])/2 for b in read_blocks}


def fd_vjp_check(engine,seq,block,position,target_block,target_position,vector,read_vector,eps=.001):
    import torch
    from torch.nn.attention import SDPBackend,sdpa_kernel
    from r1b_g_adapter import text_model,require_gradients
    require_gradients(engine.model,"K10 finite difference against VJP")
    derivative=fd_transfer(engine,seq,block,position,eps*vector,[target_block])[target_block][target_position]/eps
    fd=float(derivative@read_vector.cpu())
    delta=torch.zeros_like(vector,device=engine.device,requires_grad=True);captured={}
    def read(module,args,output):captured["h"]=output[0] if isinstance(output,tuple) else output
    handle=engine.layers[target_block].register_forward_hook(read)
    try:
        with engine.patch_state(block,delta,position,add=True),torch.enable_grad(),sdpa_kernel(SDPBackend.MATH):
            text_model(engine.model)(input_ids=torch.tensor([seq],device=engine.device),use_cache=False)
            scalar=captured["h"][0,target_position].float()@read_vector.to(engine.device)
            grad,=torch.autograd.grad(scalar,delta)
            vjp=float(grad@vector.to(engine.device))
    finally:handle.remove()
    return dict(fd=fd,vjp=vjp,relative_error=abs(fd-vjp)/max(abs(fd),abs(vjp),1e-8))


def derivative_controls(ctx,engine,own):
    """K10b native QK/OV route decomposition, foreign state and derivative gates."""
    import torch
    from r1b_g_adapter import text_model
    texts=[r["prompt_ids"]+r["tok"] for r in own.get("rows",[]) if len(r["tok"])>=4]
    if len(texts)<2:
        # Independent raw pilot text supplies an engineering derivative check;
        # it never substitutes for the K10 own-generation scientific sample.
        return dict(status="blocked",reason="two own generations with four continuation tokens required",rows=[])
    blocks=sorted({int(math.floor(x*(len(engine.layers)-1)/27+.5)) for x in (6,14,18)})
    last=len(engine.layers)-1;rows=[];checks=[];errors=[];structural=0.
    for k,seq in enumerate(texts[:(2 if ctx.mock else 64)]):
        base=engine.hidden([seq],blocks)[0]
        other=engine.hidden([texts[(k+1)%len(texts)]],blocks)[0]
        with FrozenQK(engine) as frozen:
            explicit=engine.next_logits(seq)
        probs=frozen.probabilities
        with FrozenQK(engine,probs): replay=engine.next_logits(seq)
        native=engine.next_logits(seq)
        reproduction=float((native-replay).abs().max()/native.abs().max().clamp_min(1e-9))
        errors.append(dict(text=k,native_replay=reproduction,explicit_replay=float((explicit-replay).abs().max())))
        positions=[max(0,len(seq)-5)] if ctx.mock else [len(seq)-64+p for p in (16,32) if len(seq)-64+p>=0 and len(seq)-64+p+4<len(seq)]
        for block in blocks:
            for position in positions:
                h=base[block][position]; foreign=other[block][min(position,len(other[block])-1)]
                foreign=h.norm()*foreign/foreign.norm().clamp_min(1e-9)
                def cell():
                    total=fd_transfer(engine,seq,block,position,.01*h,[last])[last]/.01
                    ov=fd_transfer(engine,seq,block,position,.01*h,[last],frozen=probs)[last]/.01
                    control=fd_transfer(engine,seq,block,position,.01*foreign,[last])[last]/.01
                    return [dict(text=k,t=position,block=block,dl=lag,
                                 rho_qk=float((total[position+lag]-ov[position+lag]).norm()/
                                              (ov[position+lag].norm()+(total[position+lag]-ov[position+lag]).norm()).clamp_min(1e-12)),
                                 tau_norm=float(total[position+lag].norm()),foreign_norm=float(control[position+lag].norm()))
                            for lag in (1,4) if position+lag<len(seq)]
                rows.extend(_checkpoint(ctx,f"k10b_{k}_{block}_{position}",cell))
        pos=max(0,len(seq)-5);vector=torch.ones(int(engine.cfg.hidden_size))*.01
        zero=fd_transfer(engine,seq,last,pos,vector,[last])[last][pos+1:]
        structural=max(structural,float(zero.abs().max()))
        if k==0 and not getattr(engine.model,"_r1b_g_forward_only",False):
            generator=torch.Generator().manual_seed(1305)
            for _ in range(2 if ctx.mock else 8):
                v=torch.randn(int(engine.cfg.hidden_size),generator=generator);u=torch.randn(v.shape,generator=generator)
                checks.append(fd_vjp_check(engine,seq,blocks[0],max(0,len(seq)-5),last,len(seq)-2,v,u))
    gate=structural==0 and max((e["native_replay"] for e in errors),default=float("inf"))<1e-3 and checks and max(c["relative_error"] for c in checks)<.05
    return dict(status="measured",rows=rows,reproduction=errors,fd_vjp=checks,structural_zero_max=structural,
                G_K10="ok" if gate and not ctx.mock else "MOCK: no scientific gate" if ctx.mock else "descriptive (code check failed)",
                head_routing_map={"stage":"k_routes","artifact":"head_routing.pt"},own_generation_count=len(texts),required_generation_count=64)


def head_routing(ctx,engine,texts,lens):
    """Exploratory native head-edge VJP attribution for the target J direction."""
    import torch
    from r1b_g_adapter import text_model
    if lens is None or getattr(engine.model,"_r1b_g_forward_only",False):
        return dict(status="not_applicable",reason="head VJP requires an original target and target J-lens",maps=[])
    block=int(math.floor(22*(len(engine.layers)-1)/27+.5))
    _,_,v=torch.linalg.svd(lens["maps"][block].float(),full_matrices=False);direction=v[0].to(engine.device)
    maps=[]
    for number,text in enumerate(texts[:(1 if ctx.mock else 4)]):
        seq=text["ids"];prefix=text["plen"]
        for offset in ((2,4) if ctx.mock else (32,48,63)):
            position=min(prefix+offset,len(seq)-1)
            def cell():
                capture={};handles=[]
                def embed(module,args,output):return output.detach().requires_grad_(True)
                def target(module,args,output):capture["h"]=output[0] if isinstance(output,tuple) else output
                handles.append(engine.model.get_input_embeddings().register_forward_hook(embed))
                handles.append(engine.layers[block].register_forward_hook(target))
                try:
                    with torch.enable_grad(),FrozenQK(engine,capture_heads=True) as state:
                        text_model(engine.model)(input_ids=torch.tensor([seq],device=engine.device),use_cache=False)
                        scalar=capture["h"][0,position].float()@direction
                        names=sorted(l for l,(out,_,_) in state.heads.items() if out.requires_grad)
                        grads=torch.autograd.grad(scalar,[state.heads[l][0] for l in names],allow_unused=True)
                    edges={};lags={}
                    for layer,gradient in zip(names,grads):
                        if gradient is None:continue
                        _,p,value=state.heads[layer]
                        grad=gradient[0].transpose(0,1).float()
                        edge=((grad@value[0].float().transpose(-1,-2))*p[0].float()).abs().cpu()
                        edges[layer]=edge
                        for lag in range(edge.shape[1]):
                            lags[lag]=lags.get(lag,0.)+float(edge.diagonal(offset=-lag,dim1=-2,dim2=-1).sum())
                    total=sum(lags.values())
                    return dict(text=number,tprime=offset,target=f"S_J{block}",edges=edges,
                                lag_share={k:x/max(total,1e-12) for k,x in lags.items()},total_absolute_attribution=total)
                finally:
                    for handle in handles:handle.remove()
            maps.append(_checkpoint(ctx,f"head_routes_{number}_{offset}",cell))
    return dict(status="measured",maps=maps,definition="|dF/do_head(r) dot p_head(r,s) v_head(s)|",target_block=block)


def hierarchy(ctx,engine,source,lens,hn,own):
    """K9 target-derived channel transfer, native finite differences and SD units."""
    import torch
    import numpy as np
    bases=channel_bases(engine,source,lens,hn)
    if len({b for b,_ in bases.values()})<2:
        return dict(status="blocked",reason="channels at two distinct blocks required",rows=[])
    random_for={}
    for name,(block,basis) in list(bases.items()):
        random_for[name]=[]
        for number in range(2 if ctx.mock else 19):
            generator=torch.Generator().manual_seed(ctx.key("g_k9_random",name,number))
            random_basis=torch.linalg.qr(torch.randn(basis.shape,generator=generator))[0]
            key=f"rand_{name}_{number}";bases[key]=(block,random_basis);random_for[name].append(key)
    n=2 if ctx.mock else 16; length=48 if ctx.mock else 512; offset=32 if ctx.mock else 2048
    data=ctx.data();neutral=[]
    for raw in data.get("wt",{}).get("ntext",data.get("lens_validation",[])):
        text=raw if isinstance(raw,str) else raw["text"]
        tokens=engine.encode(text,chat=False)
        if len(tokens)>=length:neutral.append(tokens[:length])
        if len(neutral)>=n:break
    conditions={"c2":neutral}
    for fam,name in (("ACH","c1"),("TQA","c3")):
        conditions[name]=[(r["prompt_ids"]+r["tok"])[-length:] for r in own.get("rows",[]) if r["fam"]==fam and len(r["prompt_ids"]+r["tok"])>=length][:n]
    for name,sequences in own.get("conditions",{}).items():
        conditions[name]=sequences[:n]
    coherent=[x if isinstance(x,str) else x["text"] for x in data.get("wt",{}).get("coherent",[])]
    if neutral and coherent:
        rng=np.random.default_rng(61)
        coh=engine.encode(" ".join(coherent),chat=False)
        fill=engine.encode(" ".join(coherent[i] for i in rng.permutation(len(coherent))),chat=False)
        if len(coh)>=offset and len(fill)>=offset:
            conditions["c4"]=[coh[(50*i)%(len(coh)-offset+1):][:offset]+neutral[i%len(neutral)] for i in range(n)]
            conditions["c5"]=[fill[(50*i)%(len(fill)-offset+1):][:offset]+neutral[i%len(neutral)] for i in range(n)]
    rows=[];blocks=sorted({b for b,_ in bases.values()})
    for condition,sequences in conditions.items():
        for k,seq in enumerate(sequences):
            def cell():
                base=engine.hidden([seq],blocks)[0]
                sd={name:float((base[b]@basis-(base[b]@basis).mean(0)).square().sum(-1).mean().sqrt()) for name,(b,basis) in bases.items()}
                off=offset if condition in ("c4","c5") else 0
                positions=[off+p for p in ((8,16) if ctx.mock else (128,256,384)) if off+p+16<len(seq)] or [len(seq)//2]
                values={}
                for name,(block,basis) in bases.items():
                    deeper=[b for b in blocks if b>block]
                    if not deeper:continue
                    eps=.01*float(base[block].norm(dim=-1).median())
                    for d in range(min(2,basis.shape[1])):
                        for pos in positions:
                            diffs=fd_transfer(engine,seq,block,pos,eps*basis[:,d],deeper)
                            for target,(tb,tbasis) in bases.items():
                                if tb>block and not (name.startswith("rand_") and target.startswith("rand_")):
                                    value=float((diffs[tb][pos:pos+17]@tbasis).norm(dim=-1).sum())/(eps*max(sd[target],1e-9))
                                    values.setdefault(name+"->"+target,[]).append(value)
                return dict(cond=condition,text=k,g={key:float(np.mean(v)) for key,v in values.items()},sd=sd)
            rows.append(_checkpoint(ctx,f"k9_{condition}_{k}",cell))
    complete="conditions" in own and all(len(conditions.get(c,[]))>=n for c in ("c1","c2","c3","c4","c5"))
    return dict(status="measured" if complete else "partial",rows=rows,channels={name:b for name,(b,_) in bases.items()},
                random_for=random_for,
                condition_counts={c:len(v) for c,v in conditions.items()},required_per_condition=n,
                reason=None if complete else "some prescribed conditions lack long own-generated texts",axes="target-derived; no SO-AX")


def hierarchy_measure(ctx):
    """Final K9 uses K8c arithmetic text and K8b's clean QA session transcripts."""
    from r1b_g_engine import ModelEngine
    from r1b_k_ana import k9_stats
    E=ModelEngine(ctx)
    states=_load_pt(ctx,"k_chain","source_states.pt",required=True)
    hn=_load_json(ctx,"k_prepare","hn.json") or {};lens=_load_pt(ctx,"lens_fit","lens.pt")
    original=_load_pt(ctx,"k_chain","chains.pt",required=True)
    if original.get("metadata",{}).get("H_neurons"):
        hn={**hn,"H_op":original["metadata"]["H_neurons"]}
    long=_load_pt(ctx,"k_long","long.pt") or {}
    sessions=_load_pt(ctx,"k_persistence","k8b_data.pt") or {}
    conditions={"c1":[r["prompt_ids"]+r["tok"] for r in long.get("rows",[]) if r.get("tok")],"c3":[]}
    for row in sessions.get("rows",[]):
        if row.get("block")=="tqa" and row.get("arm")=="clean":
            for session in row.get("sessions",[]):
                if session.get("turns"):
                    last=session["turns"][-1]
                    conditions["c3"].append(last["prompt_ids"]+last["tok"])
    result=hierarchy(ctx,E,states["source"]["t0"],lens,hn,{"conditions":conditions})
    named=[{**row,"g":{edge:value for edge,value in row["g"].items() if not any(name.startswith("rand_") for name in edge.split("->"))}} for row in result.get("rows",[])]
    result["statistics"]=k9_stats(named,nsplit=20 if ctx.mock else 200,nboot=10 if ctx.mock else 200) if named else {"status":"not_testable","reason":result.get("reason")}
    result["condition_sources"]={"c1":"k_long/long.pt","c2":"held-out raw WikiText","c3":"k_persistence/k8b_data.pt clean QA sessions","c4":"coherent prefix + identical neutral target","c5":"filler prefix + identical neutral target"}
    ctx.save_pt("hierarchy.pt",result)
    ctx.write_json("hierarchy.json",_clean_json({k:v for k,v in result.items() if k!="rows"}))
    return dict(status="done" if result["status"]=="measured" else "partial",outputs=["hierarchy.pt","hierarchy.json"],coverage=["K9","K7a alpha_int random baselines"],reason=result.get("reason"))


@contextlib.contextmanager
def norm_push(engine,block,direction,strength,cap,position=None):
    """SO-DMG push: f min(||h||,cap) unit(direction), native residual hook."""
    import torch
    unit=torch.nn.functional.normalize(direction.float(),dim=0)
    def hook(module,args,output):
        h=output[0] if isinstance(output,tuple) else output
        if position is not None and position>=h.shape[1]:return output
        out=h.clone();part=out if position is None else out[:,position:position+1]
        part+=float(strength)*part.float().norm(dim=-1,keepdim=True).clamp_max(cap).to(part.dtype)*unit.to(part.device,part.dtype)
        return (out,)+output[1:] if isinstance(output,tuple) else out
    handle=engine.layers[block].register_forward_hook(hook)
    try:yield
    finally:handle.remove()


def _route_generation(ctx,engine,prompt,key,steps,setup=None,start_position=0):
    """Fixed absolute-position Gumbel noise, shared across all causal routes."""
    import torch
    sequence=list(prompt)
    for _ in range(steps):
        logits=engine.next_logits(sequence,setup=setup)
        generator=torch.Generator(device=engine.device).manual_seed(ctx.key("g_k10_noise",key,len(sequence)+start_position))
        # Fixed-length K10 generation intentionally retains generated stop IDs,
        # as in the inherited instrument; these are not normal user answers.
        sequence.append(engine.choose(logits,generator,.7))
    return sequence


def routes(ctx):
    """K10a calibrated KV/token/total route split with common random numbers."""
    import torch
    import numpy as np
    import r1b_core as core
    from r1b_g_engine import ModelEngine
    from r1b_k_ana import k10a_stats
    engine=ModelEngine(ctx);data=ctx.data()
    states=_load_pt(ctx,"k_chain","source_states.pt",required=True)
    hn=_load_json(ctx,"k_prepare","hn.json") or {}
    refs=_load_pt(ctx,"k_prepare","references.pt",required=True)
    lens=_load_pt(ctx,"lens_fit","lens.pt")
    source=states["source"]["t0"]
    mapped=lambda b:int(math.floor(b*(len(engine.layers)-1)/27+.5))
    block=mapped(14);targets=sorted({mapped(b) for b in (14,18,22)})
    bases=channel_bases(engine,source,lens,hn)
    readouts={name:(b,B[:,0]) for name,(b,B) in bases.items() if b>block and (name.startswith("Jws") or name.startswith("pc16"))}
    jbasis=next((B for name,(b,B) in bases.items() if b==block and name.startswith("Jws")),None)
    named={} if jbasis is None else {"SJ14":jbasis[:,0]}
    for k in range(2 if ctx.mock else 20):
        named[f"rand{k}"]=torch.randn(int(engine.cfg.hidden_size),generator=torch.Generator().manual_seed(1401+k))
    populations={"N_H":hn.get("H_op",[]),**{f"RS10_{k}":p for k,p in enumerate(hn.get("RS10",hn.get("RS39",[])[:10]))}}
    cap=2*float(refs["mu_gen"][block].norm())
    reftexts=[x if isinstance(x,str) else x["text"] for x in data.get("ref_k16",[])]
    if not reftexts:raise core.NotRun("K10a needs independent raw REF-K16 text")
    sources={}
    for name,direction in named.items():
        doses={}
        for dose,target in (("delta1",.05),("delta2",.15)):
            def calibrate():
                return core.calibrate(lambda f:engine.damage(reftexts,setup=lambda:norm_push(engine,block,direction,f,cap))["damage"],.1,target)
            doses[dose]=_checkpoint(ctx,f"route_cal_{name}_{dose}",calibrate)
        sources[name]=dict(kind="direction",direction=direction,doses=doses)
    for name,pairs in populations.items():
        if not pairs:continue
        ids=[engine.layout.encode(int(l),int(j)) for l,j in pairs]
        doses={}
        for dose,target in (("delta1",.05),("delta2",.15)):
            def calibrate():return engine.calibrate(reftexts,ids,start=3.,target=target)
            doses[dose]=_checkpoint(ctx,f"route_cal_{name}_{dose}",calibrate)
        sources[name]=dict(kind="population",neurons=ids,doses=doses)
    prompt_file=Path(__file__).resolve().parents[3]/"results/ju_space/plans/runde1b/packages/prompts_1b.json"
    bank=json.loads(prompt_file.read_text())["prompts"] if prompt_file.is_file() else []
    pidx=[0,128] if ctx.mock else list(range(16))+list(range(128,144))
    prompts=[{"id":f"p256:{i}","text":bank[i]["text"] if isinstance(bank[i],dict) else bank[i]} for i in pidx if i<len(bank)]
    lab=_load_json(ctx,"k_prepare","lab.json") or {};litems=lab.get("items",[])
    litems=list(litems.values()) if isinstance(litems,dict) else litems
    prompts +=[{"id":str(x.get("id",x.get("qid"))),"text":x["question"]+" Answer in two or three sentences."}
               for x in litems if x.get("split")=="S_sess"][:(2 if ctx.mock else 32)]
    if not prompts:raise core.NotRun("K10a own prompts unavailable")
    texts=[];steps=12 if ctx.mock else 64
    for p in prompts:
        prefix=engine.encode(p["text"])
        ids=_checkpoint(ctx,"route_generation_"+p["id"],lambda:_route_generation(ctx,engine,prefix,p["id"],steps))
        texts.append(dict(id=p["id"],ids=ids,plen=len(prefix)))
    read_pop={"H":[p for p in hn.get("H_op",[]) if int(p[0])>block]}
    rows=[]
    for name,spec in sources.items():
        for dose,cal in spec["doses"].items():
            for offset in ((2,) if ctx.mock else (16,32)):
                for number,text in enumerate(texts):
                    def cell():
                        position=text["plen"]+offset
                        original=text["ids"][:position+9]
                        base=_read_channels(engine,original,read_pop,refs,blocks=targets)
                        responses={}
                        for sign in ((1,-1) if spec["kind"]=="direction" else (1,)):
                            if spec["kind"]=="direction":
                                setup=lambda sign=sign:norm_push(engine,block,spec["direction"],sign*cal["f"],cap,position)
                            else:
                                setup=lambda:engine.intervene(spec["neurons"],"scale",cal["f"],positions=[position])
                            kv=_read_channels(engine,original,read_pop,refs,blocks=targets,setup=setup)
                            regenerated=_route_generation(ctx,engine,original[:position+1],text["id"],len(original)-position-1,setup=setup)
                            token=_read_channels(engine,regenerated,read_pop,refs,blocks=targets)
                            total=_read_channels(engine,regenerated,read_pop,refs,blocks=targets,setup=setup)
                            responses[sign]=dict(kv=kv,tok=token,tot=total,divergence=int(regenerated[position+1]!=original[position+1]))
                        def projection(read,p):
                            out={n:float(read["hidden"][b][p]@direction) for n,(b,direction) in readouts.items()}
                            if "H" in read["aX"]:out["aH"]=float(read["aX"]["H"][p])
                            return out
                        effects={};base_values={}
                        for lag in (1,2,4,8):
                            p=position+lag;clean=projection(base,p)
                            for n,v in clean.items():base_values.setdefault(n,[]).append(v)
                            for route in ("kv","tok","tot"):
                                plus=projection(responses[1][route],p)
                                minus=projection(responses[-1][route],p) if -1 in responses else clean
                                effects[route+str(lag)]={n:(v-minus[n])/(2 if -1 in responses else 1) for n,v in plus.items()}
                        zero=0.
                        if spec["kind"]=="direction":
                            for response in responses.values():
                                for b in targets:
                                    if b<=block:zero=max(zero,float((response["kv"]["hidden"][b][position+1:]-base["hidden"][b][position+1:]).abs().max()))
                        effects.update(base=base_values,structural_zero_max=zero if spec["kind"]=="direction" else None,
                                       div=float(np.mean([r["divergence"] for r in responses.values()])))
                        return dict(src=name,dose=dose,t0=offset,text=number,eff=effects,damage=cal["damage"],in_band=cal["in_band"])
                    rows.append(_checkpoint(ctx,f"route_{name}_{dose}_{offset}_{number}",cell))
    stats=k10a_stats(copy.deepcopy(rows))
    head_maps=head_routing(ctx,engine,texts,lens)
    ctx.save_pt("head_routing.pt",head_maps)
    valid_counts=len(texts)>=(4 if ctx.mock else 64)
    valid_damage=all(r["in_band"] for r in rows)
    result=dict(status="measured" if valid_counts and valid_damage and jbasis is not None else "partial",rows=rows,sources=sources,
                texts=texts,statistics=stats,mapped_source_block=block,mapped_target_blocks=targets,
                missing_axes=["ustar","uT","usea"],text_count=len(texts),required_count=4 if ctx.mock else 64,
                calibration_in_band=valid_damage,mock=ctx.mock,scientific_verdict=stats["final"] if not ctx.mock and valid_counts and valid_damage else "not testable")
    ctx.save_pt("routes.pt",result)
    ctx.write_json("routes.json",_clean_json({k:v for k,v in result.items() if k not in ("sources","texts","rows")}))
    return dict(status="done" if result["status"]=="measured" else "partial",outputs=["routes.pt","routes.json","head_routing.pt"],coverage=["K10a target-derived sources","K10b head routing"],
                reason=None if result["status"]=="measured" else "required source/text/damage controls not fully available")


def direct_readouts(ctx,engine,source,records,by_id,gates,correct):
    """K10c answer-free chain selection versus fitted/Jacobian forward maps."""
    import numpy as np
    import torch
    from r1b_g_adapter import unembed
    from r1b_k_s2 import _ridge
    if not correct:
        return dict(status="blocked",reason="no correctly answered K4 items",res={})
    selected=correct[:(4 if ctx.mock else 240)]
    screening=[normalize_item(x) for x in ctx.data().get("arithmetic",[]) if x.get("split")=="A-scr"][:(12 if ctx.mock else 3000)]
    if len(screening)<4:return dict(status="blocked",reason="four or more independent A-scr training items required",res={})
    blocks=mapped_blocks(len(engine.layers))[1:-1] or mapped_blocks(len(engine.layers))
    last=len(engine.layers)-1
    def digit_positions(tokens):
        return [k for k,t in enumerate(tokens) if len(engine.tok.decode([t]).strip())==1 and engine.tok.decode([t]).strip().isdigit()][:3]
    def training():
        prompts=[engine.encode(x["prompt"]) for x in screening]
        generated=engine.decode(prompts,temperature=0,max_new=8)
        seqs=[p+g["ids"] for p,g in zip(prompts,generated)]
        hidden=engine.hidden(seqs,sorted(set(blocks+[last])))
        return dict(prompts=prompts,generated=generated,hidden=hidden)
    train=_checkpoint(ctx,"direct_training",training);out={}
    for block in blocks:
        predictions={};Xtest=np.stack([source["h"][block][source["ids"].index(i)].double().numpy() for i in selected])
        for digit in (1,2,3):
            X,Y=[],[]
            for p,g,h in zip(train["prompts"],train["generated"],train["hidden"]):
                positions=digit_positions(g["ids"])
                if len(positions)<digit:continue
                X.append(h[block][len(p)-1].double().numpy());Y.append(h[last][len(p)+positions[digit-1]-1].double().numpy())
            if len(X)<4:continue
            Xa,Ya=np.asarray(X),np.asarray(Y);m=len(X)*4//5
            best=min((float(np.mean((_ridge(Xa[:m],Ya[:m],lam*len(X),Xa[m:])-Ya[m:])**2)),lam) for lam in (1.,100.,10000.))[1]
            predictions[digit]=_ridge(Xa,Ya,best*len(X),Xtest)
        rows=[]
        for number,ident in enumerate(selected):
            answer=gates[ident]["greedy"];positions=digit_positions(answer);d=[answer[p] for p in positions]
            candidates=[r for r in records if r["item"]==ident and r["block"]==block and r["ptype"]=="t0" and r["variant"]=="V1"]
            top=min(candidates,key=lambda r:r["seed_rank"]) if candidates else None
            first=next((r for r in candidates if d and r["seed"]==d[0]),top)
            digits=lambda row:[([row["seed"]]+row["tok"])[p] for p in digit_positions([row["seed"]]+row["tok"])] if row else []
            prompt=engine.encode(by_id[ident]["prompt"]);seq=prompt+answer;h=source["h"][block][source["ids"].index(ident)]
            def evaluate():
                lpar,ljac=[],[]
                for digit,offset in enumerate(positions,1):
                    lpar.append(int(unembed(engine.model,torch.tensor(predictions[digit][number],dtype=torch.float32)).argmax()) if digit in predictions else -1)
                    derivative=fd_transfer(engine,seq,block,len(prompt)-1,.01*h,[last])[last][len(prompt)+offset-1]/.01
                    ljac.append(int(unembed(engine.model,derivative).argmax()))
                return dict(item=ident,d=d,cl_top=digits(top),cl_d1sel=digits(first),lpar=lpar,ljac=ljac)
            rows.append(_checkpoint(ctx,f"direct_{block}_{ident}",evaluate))
        out[block]=rows
    return dict(status="measured",res=out,n_items=len(selected),required_items=4 if ctx.mock else 240,
                scientific_gate=len(selected)>=(4 if ctx.mock else 240),training_ids=[x["id"] for x in screening],
                digit_rule="single decoded numeral characters; multi-digit tokens do not masquerade as individual digits")


def contexts(ctx):
    """K8a matched target-position length/content contrasts and offset controls."""
    import numpy as np
    import torch
    import r1b_core as core
    import r1b_g_k_persistence as persistence
    from r1b_k_labels import parse_number,alias_match,answer_span
    from r1b_k_ana import k8a_stats
    E,hn,refs,detectors,populations,detector_name,hn_label=persistence._environment(ctx)
    detector=detectors.get(detector_name);data=ctx.data()
    boundary=_load_pt(ctx,"k_persistence","boundary.pt")
    if boundary is None:
        boundary=persistence.boundary(ctx,E)
    selected=set(boundary["boundary"])
    bank={r["id"]:r["item"] for r in boundary["rows"]}
    targets=[("ACH",bank[ident]) for ident in boundary["boundary"]]
    qa=[x for x in data.get("trivia",[]) if x.get("split")=="S4c"]
    if not qa:
        lab=_load_json(ctx,"k_prepare","lab.json") or {};allocated=lab.get("items",[])
        allocated=list(allocated.values()) if isinstance(allocated,dict) else allocated
        qa=[x for x in allocated if x.get("split")=="S_sess"]
    targets +=[("TQA",x) for x in qa[:(4 if ctx.mock else 300)]]
    if not targets:raise core.NotRun("K8a has neither boundary arithmetic nor held-out QA targets")
    history=[r for r in boundary["rows"] if r["id"] not in selected] or boundary["rows"]
    coherent=[x if isinstance(x,str) else x["text"] for x in data.get("wt",{}).get("coherent",[])]
    if not coherent:raise core.NotRun("K8a requires genuine coherent raw reference text")
    lengths=(32,64,96) if ctx.mock else (512,2048,4096);middle=lengths[1]
    cells=[("NONE",0)]+[("FILLER",n) for n in lengths]+[(kind,middle) for kind in ("COHERENT","SHUFFLED","OWN-CORRECT","OWN-WRONG","INJECTED-WRONG")]
    instances=2 if ctx.mock else 8;prefixes={};prefix_notes={}
    for kind,length in cells:
        for instance in range(1 if kind=="NONE" else instances):
            if kind=="NONE":prefixes[(kind,length,instance)]=[];continue
            rng=np.random.default_rng(61+1000*instance+length);messages=[]
            if kind in ("FILLER","COHERENT","SHUFFLED"):
                if kind=="FILLER":sentences=[coherent[int(i)] for i in rng.integers(0,len(coherent),400)]
                else:
                    sentences=[coherent[(instance*97+k)%len(coherent)] for k in range(400)]
                    if kind=="SHUFFLED":sentences=[" ".join(rng.permutation(s.split())) for s in sentences]
                messages=[dict(role="user",content=" ".join(sentences)),dict(role="assistant",content="Noted.")]
            else:
                for row in [history[int(i)] for i in rng.permutation(len(history))][:200]:
                    item=row["item"]
                    if kind=="OWN-CORRECT":answer=next((t for t,ok in zip(row["texts"],row["ok"]) if ok),None)
                    elif kind=="OWN-WRONG":answer=next((t for t,ok in zip(row["texts"],row["ok"]) if not ok),None)
                    else:answer=str(item["A"]+int(rng.integers(1,9))*(1 if rng.random()<.5 else -1))
                    if answer is not None:messages.extend([dict(role="user",content=item["prompt"]),dict(role="assistant",content=answer.strip())])
            if not messages:
                prefix_notes[str((kind,length,instance))]="no eligible own-history answers";continue
            raw=E.tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=False)
            if not isinstance(raw,list):raw=raw.tolist()
            prefix=raw[-length:] if len(raw)>length else raw
            prefixes[(kind,length,instance)]=prefix
            if len(prefix)!=length:prefix_notes[str((kind,length,instance))]=f"short context: {len(prefix)} target tokens, requested {length}"
    rows=[]
    for kind,length in cells:
        for family,item in targets:
            ident=str(item.get("id",item.get("qid")));instance=0 if kind=="NONE" else int(hashlib.sha256(ident.encode()).hexdigest(),16)%instances
            key=(kind,length,instance)
            if key not in prefixes:continue
            prefix=prefixes[key]
            prompt=E.encode(item["prompt"] if family=="ACH" else "Question: "+item["question"]+" Reply with the short answer only.")
            # Every target begins at the same native token offset in a cell.
            if prefix and prompt and prompt[0]==getattr(E.tok,"bos_token_id",None):prompt=prompt[1:]
            seq=prefix+prompt
            gold=E.encode(str(item["A"]) if family=="ACH" else item["aliases"][0],chat=False)
            def cell():
                answer=E.decode([seq],max_new=4 if ctx.mock else 16,temperature=0)[0]
                generated=persistence._read(E,seq+answer["ids"],{"H":populations.get("H",[])},refs,detector)
                forced=persistence._read(E,seq+gold,{"H":populations.get("H",[])},refs,detector)
                correct=parse_number(answer["text"])==item["A"] if family=="ACH" else alias_match(answer_span(answer["text"]),item["aliases"])=="correct"
                def score(row):
                    x=row.get("sH")
                    return float(np.mean(x[len(seq):])) if x is not None and len(x)>len(seq) else None
                return dict(kind=kind,L=length,actual_L=len(prefix),inst=instance,fam=family,item=ident,acc=float(correct),
                            H_gen=score(generated),H_tf=score(forced),tok=answer["ids"],text=answer["text"],prompt_length=len(prompt))
            rows.append(_checkpoint(ctx,f"context_{kind}_{length}_{ident}",cell))
    O1={"max_abs_dlogp":0.,"top1_identical":True};gap=[]
    for family,item in targets[:(2 if ctx.mock else 16)]:
        ids=E.encode(item["prompt"] if family=="ACH" else "Question: "+item["question"])
        x=torch.tensor([ids],device=E.device);positions=torch.arange(len(ids),device=E.device)[None]
        with torch.no_grad():
            a=E.model(input_ids=x,position_ids=positions).logits[0].float().log_softmax(-1)
            b=E.model(input_ids=x,position_ids=positions+4096).logits[0].float().log_softmax(-1)
            shifted=torch.cat([positions[:,:3],positions[:,3:]+2048],1)
            c=E.model(input_ids=x,position_ids=shifted).logits[0].float().log_softmax(-1)
        O1["max_abs_dlogp"]=max(O1["max_abs_dlogp"],float((a[-1]-b[-1]).abs().max()))
        O1["top1_identical"] &= bool(a[-1].argmax()==b[-1].argmax())
        if len(ids)>4:
            truth=x[0,4:];gap.append(float(-(c[3:-1].gather(1,truth[:,None])-a[3:-1].gather(1,truth[:,None])).mean()))
    O1["pass"]=O1["top1_identical"] and O1["max_abs_dlogp"]<1e-3
    O2=dict(nll_rise=float(np.mean(gap)) if gap else None,interpretable=bool(gap) and float(np.mean(gap))<=.05)
    result=dict(status="partial" if prefix_notes else "measured",rows=rows,statistics=k8a_stats(rows),O1=O1,O2=O2,
                G_K20={"position_offset":O1["pass"],"boundary_label":boundary.get("label")},
                prefix_notes=prefix_notes,boundary_count=len(selected),qa_count=len(qa[:300]),H_label=hn_label,mock=ctx.mock)
    ctx.save_pt("contexts.pt",result)
    ctx.write_json("contexts.json",_clean_json({k:v for k,v in result.items() if k!="rows"}))
    return dict(status="done" if result["status"]=="measured" else "partial",outputs=["contexts.pt","contexts.json"],coverage=["K8a"],
                reason="one or more matched context cells unavailable" if prefix_notes else None)


def positive_control_stats(rows,items,correct):
    """Inherited K4 rule, with a genuine (identity-free) V6 label derangement."""
    import numpy as np
    from r1b_k_ana import by,anyhit,item_diff,_s,pboot
    from r1b_k_base import fin,lt,p_iut,val
    from r1b_k_labels import classify
    if len(correct)<2:
        return dict(G_K4="undecided",reason="at least two correct items required",p=1.,anyhit={})
    hit={}
    for (variant,ident),group in by([r for r in rows if r["ptype"]=="t0" and r["item"] in correct],"variant","item").items():
        hit.setdefault(variant,{})[ident]=float(np.mean([anyhit(cell) for cell in by(group,"block").values()]))
    ids=sorted(correct);cycle=np.random.default_rng(51).permutation(ids).tolist()
    perm=dict(zip(cycle,cycle[1:]+cycle[:1]));v6={}
    for (ident,),group in by([r for r in rows if r["variant"]=="V1" and r["ptype"]=="t0" and r["item"] in correct],"item").items():
        v6[ident]=float(np.mean([float(any(classify(v,items[perm[ident]])=="TRUE-INT" for r in cell for _,v,_ in r["claims"])) for cell in by(group,"block").values()]))
    d12=item_diff(hit.get("V1",{}),hit.get("V2",{}));d16=item_diff(hit.get("V1",{}),v6)
    label="chain lens reads intermediates" if fin(d12["ci"]["lo"]) and d12["ci"]["lo"]>=.05 and fin(d16["ci"]["lo"]) and d16["ci"]["lo"]>=.05 else "chain lens does not read intermediates" if lt(d12["ci"],.05) and lt(d16["ci"],.05) else "undecided"
    return dict(G_K4=label,V1_V2=_s(d12),V1_V6=_s(d16),p=p_iut(pboot(d12["draws"],.05) if len(d12["draws"]) else 1.,pboot(d16["draws"],.05) if len(d16["draws"]) else 1.),
                anyhit={v:val(np.mean(list(x.values()))) for v,x in hit.items()},V6_derangement=perm)


def chain_range_gate(ctx,rows,population=None):
    """K3g(i)+(ii): S_fit 0.5--99.5% scale support and unchanged frame detector."""
    import numpy as np
    prepared=_load_pt(ctx,"k_prepare","prepared.pt")
    refs=_load_pt(ctx,"k_prepare","references.pt")
    hn=_load_json(ctx,"k_prepare","hn.json") or {}
    frame=(_load_json(ctx,"k_analyze","k3_verdicts.json") or {}).get("K3g_frame",{})
    if not prepared or refs is None or not hn.get("H_op"):
        return {"pass":False,"label":"H scale not validated in the lens frame","reason":"S_fit/reference/H population missing"}
    if population is not None and list(map(tuple,population))!=list(map(tuple,hn["H_op"])):
        return {"pass":False,"label":"H scale not validated in the lens frame","reason":"domain-specific arithmetic population requires its own frame detector validation"}
    offsets=hn["layout"]["offsets"];columns=[offsets[int(l)]+int(j) for l,j in hn["H_op"]]
    fit=prepared["fit"];n=len(fit.get("y_q",[]))
    if not n:return {"pass":False,"label":"H scale not validated in the lens frame","reason":"S_fit answer rows unavailable"}
    values=fit["X"][:n,columns].float().numpy()
    mu=np.asarray(refs["c_mu"])[columns];sd=np.maximum(np.asarray(refs["c_sd"])[columns],1e-6)
    low,high=np.percentile(((values-mu)/sd).mean(1),[.5,99.5])
    chain=[r["aX"]["H"].float().numpy() for r in rows if r["variant"]=="V1" and "H" in r.get("aX",{})]
    chain=np.concatenate(chain) if chain else np.array([])
    share=float(np.mean((chain>=low)&(chain<=high))) if len(chain) else None
    finite=lambda x:isinstance(x,(int,float,np.number)) and bool(np.isfinite(x))
    controls=[x for x in frame.get("auroc_RS",[]) if finite(x)]
    q95=float(np.percentile(controls,95)) if controls else None
    passed=share is not None and share>=.9 and finite(frame.get("auroc_H")) and q95 is not None and frame["auroc_H"]>q95 and finite(frame.get("auroc_lb")) and frame["auroc_lb"]>.55
    return {"pass":bool(passed and not ctx.mock),"label":"H scale valid in the lens frame" if passed and not ctx.mock else "H scale not validated in the lens frame",
            "share_in_range":share,"reference_quantiles":{"p0_5":float(low),"p99_5":float(high)},"n_chain_positions":len(chain),"frame":frame,"frame_random_q95":q95}


def _coverage():
    return {"K4": {"status":"measured","artifact":"chains.pt","controls":["V1","V2","V3","V4","V5","V7","V8 damage calibration","oracle","decoding grid","GVRA","planted verifier"]},
            "K4p":{"status":"measured","artifact":"k4p.pt"},
            "K6a":{"status":"measured","analysis_stage":"k_chain_analyze"},
            "K6b":{"status":"measured","analysis_stage":"k_chain_analyze","controls":["conditional lagged association","read-state direct writes","break lead/lag"]},
            "K6d":{"status":"measured","analysis_stage":"k_chain_analyze"},
            "K6e":{"status":"external_stage","stage":"k_persistence"},
            "K6f":{"status":"external_stage","stage":"k_extra"},
            "K6h":{"status":"external_stage","stage":"k_extra"},
            "K7b":{"status":"measured","artifact":"k7b.pt","axes":"target-derived H/J/PCA, rank-matched nulls"},
            "K7a":{"status":"external_stage","stage":"k_extra"},
            "K8a":{"status":"external_stage","stage":"k_contexts"},
            "K8b":{"status":"external_stage","stage":"k_persistence"},
            "K8c":{"status":"external_stage","stage":"k_long"},
            "K9":{"status":"external_stage","stage":"k_hierarchy"},
            "K10a":{"status":"external_stage","stage":"k_routes"},
            "K10b":{"status":"measured","artifact":"derivative_controls.pt","head_map_stage":"k_routes"},
            "K10c":{"status":"measured","artifact":"direct_readouts.pt"}}


def measure(ctx):
    import torch
    from r1b_g_engine import ModelEngine
    from r1b_g_lens import frame_spec,penultimate_norm
    from r1b_g_k_extra import _FOV
    engine = ModelEngine(ctx)
    cfg, data = _config(ctx), ctx.data()
    items = [normalize_item(x) for x in data.get("arithmetic", [])]
    items = [x for x in items if x["split"] == "A-lens" and not x.get("cal")]
    if cfg["screening"]:
        items = items[:cfg["screening"]]
    if len(items) < 2:
        raise ValueError("K chain requires at least two non-calibration A-lens items")
    frame_metadata=frame_spec(engine)
    frame = _frame(engine.tok,data,ctx.mock) if data.get("lens_frame_text") else frame_metadata["frame_ids"]
    neutral = _neutral(engine.tok, ctx.mock, cfg["seeds"])
    hn = _load_json(ctx, "k_prepare", "hn.json") or {}
    preview=(_load_json(ctx,"k_analyze","k3_verdicts.json") or {}).get("K3a",{})
    H_label=None
    if hn.get("H_op_name")=="H_ar" or preview.get("arith",{}).get("string","").startswith("H-neurons replicated"):
        pass
    elif preview.get("har",{}).get("string")=="H_ar replicated (arithmetic in-domain)" and hn.get("H_ar"):
        hn={**hn,"H_op":hn["H_ar"]};H_label="domain-specific H-set"
    else:H_label="H not valid in arithmetic (in-session preview)"
    refs = _load_pt(ctx, "k_prepare", "references.pt")
    populations = {"H": hn.get("H_op", [])}
    populations.update({f"RS{i}": p for i, p in enumerate(hn.get("RS39", []))})
    populations = {n: pairs for n, pairs in populations.items() if pairs}
    lens = _load_pt(ctx, "lens_fit", "lens.pt")
    validation = _load_json(ctx, "lens_validate", "lens_validation.json") or {}
    mtp=_load_json(ctx,"mtp","mtp_calibration.json") or {}
    engine._g_self_variant=mtp.get("variant","E") if lens is not None else "E"
    engine._g_jlens=lens
    if engine._g_self_variant=="L":
        engine._g_penultimate_norm=_checkpoint(ctx,"penultimate_norm",lambda:penultimate_norm(ctx,engine))
    seed_instrument = "target J-lens" if lens is not None else "logit lens (no validated target J-lens)"
    gates = gate_arithmetic(ctx, engine, items)
    correct, wrong = _choose_items(items, gates, cfg)
    selected = correct + wrong
    if len(selected) < 2:
        raise ValueError("K chain gating yielded fewer than two items")
    by_id = {x["id"]: x for x in items}
    blocks = mapped_blocks(len(engine.layout.widths))
    # MOCK still evaluates every distinct mapped block, but uses fewer seeds/items.
    source = {}
    for ptype, identifiers in (("t0", selected), ("t1", correct)):
        if not identifiers:
            continue
        sequences = []
        for ident in identifiers:
            prompt = engine.encode(by_id[ident]["prompt"])
            if ptype == "t1":
                answer = gates[ident]["greedy"]
                end = next((i for i, t in enumerate(answer) if any(c.isdigit() for c in engine.tok.decode([t]))), None)
                if end is None:
                    raise ValueError("correct arithmetic answer lacks a digit token")
                prompt += answer[:end+1]
            sequences.append(prompt)
        hidden = engine.hidden(sequences, blocks)
        source[ptype] = dict(ids=identifiers, prompts=sequences,
                            h={b: torch.stack([r[b][-1].float().cpu() for r in hidden]) for b in blocks})
    reinjection=reinjection_calibration(ctx,engine,source.get("t0"),data.get("arithmetic",[]))
    covariance={}
    if refs is not None:
        for block in blocks:
            sigma=(refs.get("sigma") or {}).get(block)
            if sigma is not None:
                values,vectors=torch.linalg.eigh(sigma.double())
                covariance[block]=(vectors*values.clamp_min(0).sqrt()[None])@vectors.T
    records = []
    matching = {}
    for ptype, source_data in source.items():
        identifiers = source_data["ids"]
        if len(identifiers) < 2:
            continue
        foreign, matched = foreign_indices(identifiers, by_id)
        matching[ptype] = dict(foreign={i: identifiers[j] for i, j in zip(identifiers, foreign)}, matched=matched)
        for block in blocks:
            states = source_data["h"][block]
            means = (states.sum(0)[None] - states) / (len(states) - 1)
            seeds = [_seeds(engine, h, block, lens, cfg["seeds"]) for h in states]
            for index, ident in enumerate(identifiers):
                target, f = by_id[ident], foreign[index]
                forced_map={}
                jobs = [("V1", states[index], means[index], seeds[index], None, 1., .1),
                        ("V2", states[f], means[f], seeds[f], None, 1., .1)]
                if ptype == "t0":
                    jobs += [("V3", states[index], means[index], seeds[f], None, 1., .1),
                             ("V4", states[index], means[index], neutral, None, 1., .1)]
                    if ident in correct:
                        if refs is not None and block in refs.get("mu_gen",{}):
                            mu_gen=refs["mu_gen"][block]
                            jobs.extend([("V1_mugen",states[index],mu_gen,seeds[index],None,1.,.1),
                                         ("V5",mu_gen,mu_gen,seeds[index],None,0.,.1)])
                        if block in covariance:
                            generator=torch.Generator().manual_seed(ctx.key("g_k_v7",ident,block))
                            noise=(covariance[block]@torch.randn(states.shape[-1],generator=generator,dtype=torch.float64)).float()
                            null=means[index]+(states[index]-means[index]).norm()*noise/noise.norm().clamp_min(1e-9)
                            jobs.append(("V7",null,means[index],seeds[index],None,1.,.1))
                        if reinjection["blocks"].get(block,{}).get("beta") is not None:
                            jobs.append(("V8",states[index],means[index],seeds[index],None,1.,.1))
                        for name,values_ in (("ORACLE",target["I"]),("ORACLE_F",by_id[identifiers[f]]["I"])):
                            forced_map[name]=engine.encode(" "+" ".join(map(str,values_)),chat=False)[:cfg["steps"]-1]
                            jobs.append((name,states[index],means[index],seeds[index][:1],None,1.,.1))
                        for name, temperature, lam, plaus in [("G_lam0.5", None, .5, .1), ("G_lam2", None, 2., .1),
                                ("G_pl0.05", None, 1., .05), ("G_pl0.3", None, 1., .3),
                                ("G_greedy",None,0.,.1),
                                ("T0.5", .5, 1., .1), ("T1.0", 1., 1., .1)]:
                            jobs.append((name, states[index], means[index], seeds[index], temperature, lam, plaus))
                            if temperature:
                                jobs.append((name + "_V2", states[f], means[f], seeds[f], temperature, lam, plaus))
                for name, state, prior, token_seeds, temperature, lam, plaus in jobs:
                    def cell(name=name, state=state, prior=prior, token_seeds=token_seeds,
                             temperature=temperature, lam=lam, plaus=plaus, ident=ident, index=index,
                             target=target, block=block, ptype=ptype):
                        reinj=reinjection["blocks"].get(block,{}).get("beta")
                        injection=additive(engine,block,reinj*(state-prior),start=len(frame),real_stream_only=True) if name=="V8" else contextlib.nullcontext()
                        shares=NativeShares(engine,populations) if populations and name in ("V1","V2") else None
                        fov=_FOV(engine,block if engine._g_self_variant=="E" else len(engine.layers)-2,len(frame)-1)
                        with injection, _Channels(engine, populations, refs) as channels, shares if shares is not None else contextlib.nullcontext(),fov:
                            out = rollout(engine, block, state, token_seeds, steps=cfg["steps"], prior_state=prior,
                                          frame_ids=frame, key=ctx.key("g_k_chain", ident, block, ptype, name),
                                          temperature=temperature, lam=lam, plaus=plaus,
                                          forced=[forced_map[name]]*len(token_seeds) if name in forced_map else None,
                                          observer=shares.observe if shares is not None else None,
                                          keep_distributions=name == "V1" and ptype == "t0")
                        rows = []
                        for rank, seed in enumerate(token_seeds):
                            tokens = out["tok"][rank].tolist()
                            scored = score_tokens(engine.tok, [seed] + tokens, target,
                                                  W=gates[ident]["W"], OWN=gates[ident]["own"])
                            row = dict(item=ident, block=block, ptype=ptype, variant=name, seed=int(seed), seed_rank=rank,
                                       tok=tokens, text=engine.tok.decode([seed] + tokens), **scored,
                                       source_item=identifiers[f] if name in ("V2", "T0.5_V2", "T1.0_V2") else ident,
                                       seed_instrument="neutral" if name == "V4" else seed_instrument)
                            for field in ("lpr", "lpp", "ent", "top1", "cdm", "e", "alt"):
                                if field in out:
                                    row[field] = out[field][rank].cpu()
                            # Native rollout orders seed -> step -> real, prior;
                            # each forward has one row. Do not mix the prior's
                            # channels into the real chain's lagged predictors.
                            real_steps=channels.steps[2*rank*cfg["steps"]:2*(rank+1)*cfg["steps"]:2]
                            row["aX"] = {population: torch.stack([step[population][0] for step in real_steps])
                                         for population in populations if real_steps and all(population in step for step in real_steps)}
                            row["share"]=shares.row(rank) if shares is not None else {}
                            row["fov"]=torch.stack([step[0] for step in fov.steps[2*rank*cfg["steps"]:2*(rank+1)*cfg["steps"]:2]])
                            if out.get("log_probs") is not None and name == "V1" and ptype == "t0":
                                row["kl"] = _self_kl(engine, source_data["prompts"][index], seed, tokens, out["log_probs"][rank])
                            rows.append(row)
                        return rows
                    records.extend(_checkpoint(ctx,f"cell_{ptype}_{block}_{ident}_{name}", cell))
    ctx.save_pt("source_states.pt",dict(source=source,blocks=blocks,items=by_id,correct=correct,wrong=wrong,
                                       variant=engine._g_self_variant,norm_l=getattr(engine,"_g_penultimate_norm",None),frame_ids=frame))
    ctx.save_pt("reinjection.pt",reinjection)
    verifier=_checkpoint(ctx,"verifier_controls",lambda:verifier_controls(ctx,engine,records,source,frame,correct,by_id))
    probe=crossfit_probe(ctx,engine,data.get("arithmetic",[]),blocks)
    channel=throughput(ctx,engine,source.get("t0"),frame,lens,hn,by_id,gates,correct)
    own=own_text(ctx,engine,data.get("arithmetic",[]),populations,refs)
    causal=causal_chain_controls(ctx,engine,source.get("t0"),frame,lens,hn,refs,by_id,gates,correct,verifier["b_star"])
    hierarchy_data=dict(status="external_stage",stage="k_hierarchy",rows=[],reason="requires K8c arithmetic and K8b QA session generations")
    derivatives=derivative_controls(ctx,engine,own)
    direct=direct_readouts(ctx,engine,source.get("t0"),records,by_id,gates,correct)
    rho=read_state_association(ctx,engine,source.get("t0"),records,populations,refs,by_id)
    ctx.save_pt("k4p.pt",probe)
    ctx.save_pt("k7b.pt",channel)
    ctx.save_pt("own_text.pt",own)
    ctx.save_pt("causal_chains.pt",causal)
    ctx.save_pt("hierarchy.pt",hierarchy_data)
    ctx.save_pt("derivative_controls.pt",derivatives)
    ctx.save_pt("direct_readouts.pt",direct)
    ctx.save_pt("read_state_association.pt",rho)
    ctx.write_json("verifier.json",_clean_json(verifier))
    coverage = _coverage()
    if probe["status"]=="blocked": coverage["K4p"]={"status":"blocked","reason":probe["reason"]}
    if channel["status"]=="blocked": coverage["K7b"]={"status":"blocked","reason":channel["reason"]}
    coverage["K6c"]={"status":causal["status"],"reason":causal.get("reason","paired rollout clamp and direct-write ablation, with mean-matched random controls")}
    coverage["K6g"]={"status":"measured" if causal["k6g"] else "blocked","reason":causal.get("reason",causal.get("G_K19"))}
    coverage["K9"]={"status":hierarchy_data["status"],"reason":hierarchy_data.get("reason")}
    coverage["K10b"]={"status":derivatives["status"],"reason":derivatives.get("reason"),"head_map_stage":"k_routes"}
    coverage["K10c"]={"status":direct["status"],"reason":direct.get("reason")}
    metadata = dict(mock=ctx.mock, model=ctx.model_name, counts=cfg, blocks=blocks,
                    seed_instrument=seed_instrument, lens_validation=validation,
                    self_rollout_variant=engine._g_self_variant,mtp_calibration={k:mtp.get(k) for k in ("G_K5","variant","labels")},
                    target_j_available=lens is not None,dla_definition=NativeShares.__doc__,
                    H_neurons=populations.get("H",[]),H_label=H_label,
                    complete_protocol=True,protocol_scope="chain-owned measurements; external_stage arms execute separately",
                    frame_ids=frame,neutral_seeds=neutral,
                    frame_rule="G-owned canonical target-tokenized neutral frame",frame_spec=frame_metadata,
                    matched_state_controls=matching, observation_only=True)
    gating = dict(correct=correct, wrong=wrong, gate=gates,
                  n_correct=len(correct), n_wrong=len(wrong),
                  G_K19=len(wrong) >= (2 if ctx.mock else 64),
                  requested_counts_met=len(correct) == cfg["correct"] and len(wrong) == cfg["wrong"])
    ctx.save_pt("chains.pt", dict(records=records, rows=records, gating=gating,
                                  items=by_id, metadata=metadata, coverage=coverage,verifier=verifier))
    seed_gate={"pass":False,"label":"target J-lens unavailable"}
    if lens is not None:
        from r1b_g_adapter import unembed
        last=blocks[-1];src=source["t0"];jhit=[];lhit=[]
        for ident,h in zip(src["ids"],src["h"][last]):
            if not gates[ident]["greedy"]:continue
            token=gates[ident]["greedy"][0]
            jhit.append(token in _seeds(engine,h,last,lens,cfg["seeds"]))
            with torch.no_grad():lhit.append(token in unembed(engine.model,h.to(engine.device)).topk(cfg["seeds"]).indices.tolist())
        seed_gate=dict(pass_=bool(jhit) and sum(jhit)>=sum(lhit),j_recall=sum(jhit)/len(jhit) if jhit else None,
                       logit_recall=sum(lhit)/len(lhit) if lhit else None,n=len(jhit),label=None if jhit and sum(jhit)>=sum(lhit) else "J-lens seeds weak on task text")
        seed_gate["pass"]=seed_gate.pop("pass_") and not ctx.mock
    ctx.write_json("task_seed_gate.json",seed_gate)
    ctx.write_json("chain_measurements.json", _clean_json(dict(status="done", model=ctx.model_name,
                   rows=len(records), gating=gating, metadata=metadata, coverage=coverage)))
    return dict(status="done", outputs=["chains.pt", "chain_measurements.json","source_states.pt","reinjection.pt","k4p.pt","k7b.pt","own_text.pt","causal_chains.pt","hierarchy.pt","derivative_controls.pt","direct_readouts.pt","read_state_association.pt","verifier.json","task_seed_gate.json"],
                reason="measurements executed; eligibility, counts and positive-control gates remain explicit", coverage=coverage)


def analyze(ctx):
    from r1b_k_ana import k6d_stats, k6a_stats, k4p_stats, k7b_stats, claim_table, k6b_rule, k6b_lag,_k6b_strings,k6c_stats, k6g_stats, k9_stats, k10b_stats,k10c_stats
    from r1b_k_base import holm_family
    d = _load_pt(ctx, "k_chain", "chains.pt", required=True)
    rows, gate = d["records"], d["gating"]
    I = dict(correct=gate["correct"], wrong=gate["wrong"], items=d["items"])
    stats = {"K4_provisional": positive_control_stats(rows,d["items"],gate["correct"]),
             "K6d_provisional": k6d_stats(dict(rows=rows), I)}
    stats["K6a_provisional"]=k6a_stats(dict(rows=rows))
    probe=_load_pt(ctx,"k_chain","k4p.pt")
    stats["K4p"]=k4p_stats(probe) if probe and probe.get("res") else dict(status="blocked",reason=(probe or {}).get("reason","input missing"))
    channel=_load_pt(ctx,"k_chain","k7b.pt")
    stats["K7b"]=k7b_stats(channel) if channel and channel.get("rows") else dict(status="blocked",reason=(channel or {}).get("reason","input missing"))
    hierarchy_data=_load_pt(ctx,"k_chain","hierarchy.pt")
    named_hierarchy=[{**row,"g":{edge:value for edge,value in row["g"].items() if not any(name.startswith("rand_") for name in edge.split("->"))}} for row in (hierarchy_data or {}).get("rows",[])]
    stats["K9_provisional"]=k9_stats(named_hierarchy,nsplit=20 if ctx.mock else 200,nboot=10 if ctx.mock else 200) if named_hierarchy else dict(status="blocked",reason=(hierarchy_data or {}).get("reason","input missing"))
    derivatives=_load_pt(ctx,"k_chain","derivative_controls.pt")
    stats["K10b_provisional"]=k10b_stats(derivatives)
    stats["K10c_provisional"]=k10c_stats(_load_pt(ctx,"k_chain","direct_readouts.pt"))
    causal=_load_pt(ctx,"k_chain","causal_chains.pt")
    stats["K6c_provisional"]=k6c_stats(causal["k6c"]) if causal else dict(status="blocked",reason="input missing")
    if causal and causal.get("k6g"):
        # The inherited statistic omits the first four fixed Qwen tokens. Rebase
        # completed-claim positions so the same exclusion covers the complete
        # forced target-tokenized segment, including its seed.
        adapted=[{**r,"claims":[(s-(len(r["forced"])+1)+4,v,c) for s,v,c in r["claims"]]} for r in causal["k6g"]]
        stats["K6g_provisional"]=k6g_stats(dict(rows=adapted,G_K19=causal.get("G_K19")))
    nrs=min([len([k for k in r.get("aX",{}) if k.startswith("RS")]) for r in rows if r["variant"]=="V1"],default=0)
    if nrs>=2:
        X,y,groups,random=claim_table(rows,d["items"],nrs)
        stats["K6b_i_provisional"]=k6b_rule(X,y,groups,random,B=10 if ctx.mock else 2000)
    else:
        stats["K6b_i_provisional"]={"status":"blocked","reason":"fewer than two matched random channel populations"}
    r1=stats["K6b_i_provisional"];r2=_load_pt(ctx,"k_chain","read_state_association.pt") or {}
    family=holm_family({"i":r1.get("p_iut",1.),"ii":r2.get("p_iut",1.)})
    r1["string"]=_k6b_strings(r1,family["i"]["reject"]);r1["holm"]=family["i"]
    r2["string"]="H-activity predicts lens confabulation beyond chain length" if family["ii"]["reject"] else "undecided";r2["holm"]=family["ii"]
    stats["K6b_ii_provisional"]=r2
    stats["K6b_iii_provisional"]=k6b_lag(dict(rows=rows),nrs,r1) if nrs>=2 else {"status":"not_testable","reason":"matched random channels unavailable"}
    validation=_load_json(ctx,"lens_validate","lens_validation.json") or {}
    jgate=validation.get("G_K_J",{})
    seeds=_load_json(ctx,"k_chain","task_seed_gate.json") or {"pass":False,"label":"task seed check unavailable"}
    mtp=_load_json(ctx,"mtp","mtp_calibration.json") or {}
    genuine=not ctx.mock
    sufficient=gate["n_correct"]>=256
    k4label=stats["K4_provisional"]["G_K4"]
    k4b=stats["K4p"].get("G_K4b","not testable")
    gates={"G_K_J":{**jgate,"pass":genuine and bool(jgate.get("pass"))},
           "G_K4":{"pass":genuine and sufficient and bool(jgate.get("pass")) and k4label=="chain lens reads intermediates","label":k4label,"n_correct":gate["n_correct"],"required_correct":256},
           "G_K4b":{"pass":genuine and len((probe or {}).get("items",[]))>=4200 and k4b=="intermediate present","label":k4b},
           "G_K5":{"pass":genuine and mtp.get("G_K5")=="self-rollout calibrated","label":mtp.get("G_K5","not decidable")},
           "G_K6":{"pass":genuine and d["verifier"]["G_K6"]=="verifier reliable","label":d["verifier"]["G_K6"]},
           "G_K7":{**seeds,"pass":genuine and bool(seeds.get("pass"))},
           "G_K14":chain_range_gate(ctx,rows,d["metadata"].get("H_neurons"))}
    result = dict(status="done", mock=ctx.mock, model=ctx.model_name, statistics=stats,gates=gates,
                  scientific_verdict="MOCK: no scientific verdict" if ctx.mock else k4label,
                  coverage=d["coverage"], prerequisite_gates=d["gating"],
                  inherited_rules=["K4 paired item bootstrap with identity-free V6","r1b_k_ana K4p/K6/K7b/K9/K10 rules"],
                  note="Only explicit passing gates support downstream scientific conclusions; external_stage protocols are reported separately.")
    ctx.write_json("chain_analysis.json", _clean_json(result))
    return dict(status="done", outputs=["chain_analysis.json"], coverage=d["coverage"],reason=None)
