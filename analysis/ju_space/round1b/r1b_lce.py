"""SO-LCE (main §2.3; owner D): the long-context engine, used by D (G-LCE-1..4), K (<= 4,096 tokens; G-K20) and L (L2a; G-L2a-LCE).

    eng = LCE(model)
    cache, lg = eng.fwd(ids)                       # B = 1 (or equal-length rows) prefill; logits_to_keep=1 by default
    cache, lg = eng.fwd(chunk, cache, keep=k)      # chunked extension over the cached prefix (explicit position_ids)
    eng.crop(cache, n) / crop_layers(cache, n, b)  # crop path (all layers) / per-layer crop of layers <= b after a stopped read (K, R)
    c2 = eng.copy(cache, B=3)                      # copy path / branching by cache copies (batch_repeat_interleave)
    ids, lg, cache, am, means = eng.decode(cache, lg[:, -1], keys, T, n, stop)   # keyed Gumbel-max, fixed length, block means
    lp = eng.token_logprobs(doc_ids)               # chunked lm_head over a raw document (L2a)
    eng.record()                                   # SG9 / G-LCE-2 record: kernel, peak memory, GPU, versions, calls

Every forward runs under r1b_core.sdpa_efficient (SG9: on the GPU only the memory-efficient SDPA kernel is allowed; a math fallback
raises). Left padding (attention mask, positions = cumsum(mask) - 1) is for short contexts only; long rows have equal lengths. Keyed
decoding uses rk_loop's Gumbel (r1b_core.gumbel_class: verbatim source, sha256-checked); stop tokens are masked before the fixed answer
length (never sampled), every sampled token is fed, so the cache ends with the answer and block means cover every answer token."""
import copy, time
import r1b_core as K


def _t():
    import torch
    return torch


class LCE:
    def __init__(self, model):
        self.model, self.dev = model, K.model_device(model)
        self.calls, self.secs, self.tokens, self.kernel = 0, 0.0, 0, None

    def fwd(self, ids, cache=None, keep=1, am=None, pos=None):
        """Forward of ids [T] or [B, T] over cache (None: new DynamicCache); keep: int (last positions) or index tensor. Returns
        (cache, logits [B, k, V] float32)."""
        torch = _t()
        from transformers import DynamicCache
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.dev)
        ids = ids[None] if ids.dim() == 1 else ids
        past = 0 if cache is None else cache.get_seq_length()
        if cache is None:
            cache = DynamicCache(config=self.model.config)
        if pos is None:
            if am is None:
                pos = (past + torch.arange(ids.shape[1], device=self.dev))[None].expand(ids.shape[0], -1)
            else:
                am = torch.as_tensor(am, device=self.dev)
                pos = (am.long().cumsum(1) - 1).clamp(min=0)[:, -ids.shape[1]:]
        t0 = time.time()
        with K.sdpa_efficient(self.dev) as kern, torch.no_grad():
            out = self.model(input_ids=ids, attention_mask=None if am is None else torch.as_tensor(am, device=self.dev),
                             position_ids=torch.as_tensor(pos, device=self.dev), past_key_values=cache, use_cache=True,
                             logits_to_keep=keep)
        self.kernel, self.calls, self.tokens = kern, self.calls + 1, self.tokens + int(ids.numel())
        self.secs += time.time() - t0
        return out.past_key_values, out.logits.float()

    @staticmethod
    def crop(cache, n):
        cache.crop(int(n))
        return cache

    @staticmethod
    def crop_layers(cache, n, upto):
        """Per-layer crop of layers 0..upto (after a forward stopped after block upto, the later layers did not grow)."""
        for l in range(int(upto) + 1):
            cache.layers[l].crop(int(n))
        return cache

    @staticmethod
    def copy(cache, B=1):
        c = copy.deepcopy(cache)
        if B > 1:
            c.batch_repeat_interleave(int(B))
        return c

    def decode(self, cache, logits, keys, T, n_new, stop, am=None, blocks=()):
        """Keyed Gumbel-max decoding of exactly n_new tokens per row from the last-position logits [B, V]; stop: token ids masked
        before the fixed length; am: attention mask [B, past] for left-padded rows. Returns (ids [B, n_new] cpu, last logits,
        cache, am, {block: mean state over the answer tokens [B, D] float64 cpu})."""
        torch = _t()
        G = K.gumbel_class(self.dev, T)([int(k) for k in keys])
        st = torch.as_tensor(sorted(int(s) for s in stop), device=self.dev)
        hk = K.Hooks(self.model).capture(list(blocks), cpu=False) if blocks else None
        toks, cur = [], logits
        try:
            for _ in range(int(n_new)):
                sc = cur.float().clone()
                sc[:, st] = float("-inf")
                y = G(None, sc).argmax(-1)
                toks.append(y)
                if am is not None:
                    am = torch.cat([torch.as_tensor(am, device=self.dev), torch.ones(y.shape[0], 1, dtype=torch.long, device=self.dev)], 1)
                cache, lg = self.fwd(y[:, None], cache, keep=1, am=am)
                cur = lg[:, -1]
        finally:
            if hk is not None:
                hk.remove()
        means = {b: torch.cat(hk.states[b], 1).double().mean(1).cpu() for b in blocks} if blocks else {}
        return torch.stack(toks, 1).cpu(), cur, cache, am, means

    def token_logprobs(self, ids, chunk=2048):
        """log p(ids[t] | ids[:t]) for t >= 1 (float64 numpy [T-1]) by chunked extension with the lm_head on each chunk (L2a)."""
        torch = _t()
        import numpy as np
        ids = list(map(int, ids)); out, cache, prev = [], None, None
        for s in range(0, len(ids), chunk):
            c = ids[s:s + chunk]
            cache, lg = self.fwd(c, cache, keep=len(c))
            lp = torch.log_softmax(lg[0].double(), -1)
            if prev is not None:
                out.append(float(prev[c[0]]))
            tgt = torch.as_tensor(c[1:], device=lp.device)
            out.extend(lp[torch.arange(len(c) - 1, device=lp.device), tgt].cpu().tolist())
            prev = lp[-1]
        return np.asarray(out, dtype=np.float64)

    def record(self):
        torch = _t()
        r = dict(kernel=self.kernel, calls=self.calls, tokens=self.tokens, seconds=round(self.secs, 2), versions=K.versions())
        if torch.cuda.is_available():
            r.update(peak_memory_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3), gpu=torch.cuda.get_device_name(0),
                     tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32))
        return r


def letters_logodds(lg, a_id, b_id):
    """(log p(b) - log p(a), p(a) + p(b)) from logits [..., V]."""
    torch = _t()
    lp = torch.log_softmax(lg.double(), -1)
    return (lp[..., b_id] - lp[..., a_id]), (lp[..., a_id].exp() + lp[..., b_id].exp())
