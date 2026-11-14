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
length (never sampled), every sampled token is fed, so the cache ends with the answer and block means cover every answer token.

Engine switch engine.fast.lce_inplace_kv (runde1b_freeze.json; default off): the KV cache of every layer lives in a preallocated buffer
(InplaceLayer): a forward slice-assigns the new keys / values and attends over a view of the filled length, instead of DynamicLayer's
torch.cat rewrite of the whole cache; crop moves the filled length back; copy writes the B row copies into buffers with headroom;
offload releases the buffer. Same values and kernels (bit identity: tests/fast_identity.py on the CPU, r1b_P0b on the A100)."""
import copy, time
import r1b_core as K


def _t():
    import torch
    return torch


_INPLACE = {}


def inplace_layer_cls():
    """InplaceLayer(DynamicLayer) (built on first use, so that importing this module needs no transformers)."""
    if "cls" in _INPLACE:
        return _INPLACE["cls"]
    from transformers.cache_utils import DynamicLayer

    class InplaceLayer(DynamicLayer):
        """keys / values are views [:, :, :n] of the buffers _kb / _vb; update() slice-assigns the new states; when the buffer is full
        or keys is not its prefix view (after copy.copy, batch ops, offload), the states move into a new buffer with headroom."""
        headroom = 256

        def lazy_initialization(self, key_states, value_states):
            super().lazy_initialization(key_states, value_states)
            self._kb = self._vb = None

        def _is_view(self, B, need):
            kb = getattr(self, "_kb", None)
            return (kb is not None and self.keys is not None and self.keys.numel() > 0 and kb.shape[0] == B and need <= kb.shape[-2]
                    and self.keys.data_ptr() == kb.data_ptr() and self.keys.device == kb.device)

        def update(self, key_states, value_states, *args, **kwargs):
            torch = _t()
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
            n = self.get_seq_length(); t = key_states.shape[-2]; need = n + t; B = key_states.shape[0]
            if not (n == 0 and getattr(self, "_kb", None) is not None and self._kb.shape[0] == B and need <= self._kb.shape[-2]) \
                    and not self._is_view(B, need):
                cap = need + max(int(self.headroom), 0)
                shp = tuple(key_states.shape[:2]) + (cap,) + tuple(key_states.shape[3:])
                kb = torch.empty(shp, dtype=key_states.dtype, device=key_states.device)
                vb = torch.empty(tuple(value_states.shape[:2]) + (cap,) + tuple(value_states.shape[3:]), dtype=value_states.dtype,
                                 device=value_states.device)
                if n:
                    kb[:, :, :n].copy_(self.keys); vb[:, :, :n].copy_(self.values)
                self.keys = self.values = None
                self._kb, self._vb = kb, vb
            self._kb[:, :, n:need].copy_(key_states); self._vb[:, :, n:need].copy_(value_states)
            self.keys, self.values = self._kb[:, :, :need], self._vb[:, :, :need]
            return self.keys, self.values

        def get_seq_length(self):
            if not self.is_initialized or self.keys is None or self.keys.numel() == 0:
                return 0
            return self.keys.shape[-2]

    _INPLACE["cls"] = InplaceLayer
    return InplaceLayer


class LCE:
    def __init__(self, model, inplace=None):
        """inplace: the in-place KV buffer (None: engine.fast.lce_inplace_kv of the running session's freeze record)."""
        self.model, self.dev = model, K.model_device(model)
        self.calls, self.secs, self.tokens, self.kernel = 0, 0.0, 0, None
        self.inplace = K.engine_fast(None, "lce_inplace_kv") if inplace is None else bool(inplace)

    def new_cache(self):
        from transformers import DynamicCache
        from transformers.cache_utils import DynamicLayer
        c = DynamicCache(config=self.model.config)
        if self.inplace:
            IL = inplace_layer_cls()
            c.layers = [IL() if type(l) is DynamicLayer else l for l in c.layers]
        return c

    def fwd(self, ids, cache=None, keep=1, am=None, pos=None):
        """Forward of ids [T] or [B, T] over cache (None: new DynamicCache); keep: int (last positions) or index tensor. Returns
        (cache, logits [B, k, V] float32)."""
        torch = _t()
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.dev)
        ids = ids[None] if ids.dim() == 1 else ids
        past = 0 if cache is None else cache.get_seq_length()
        if cache is None:
            cache = self.new_cache()
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
        """Crop every layer to n tokens (transformers 5.x: crop(-k) removes k tokens)."""
        for layer in cache.layers:
            k = layer.get_seq_length() - int(n)
            if k > 0:
                layer.crop(-k)
        return cache

    @staticmethod
    def crop_layers(cache, n, upto):
        """Per-layer crop of layers 0..upto (after a forward stopped after block upto, the later layers did not grow)."""
        for l in range(int(upto) + 1):
            k = cache.layers[l].get_seq_length() - int(n)
            if k > 0:
                cache.layers[l].crop(-k)
        return cache

    @staticmethod
    def copy(cache, B=1, inplace=None):
        """Branch copy: a new cache whose layers hold B copies of the source rows (B = 1: a plain copy), written into fresh tensors on
        the model device without an intermediate single copy; the source may be offloaded to the CPU (offload). inplace: the copy's
        layer kind (None: the source's; True: in-place buffers with headroom; False: DynamicLayer)."""
        torch = _t(); c = copy.copy(cache); c.layers = []
        IL = _INPLACE.get("cls") if inplace is not True else inplace_layer_cls()
        for lay in cache.layers:
            ip = (IL is not None and isinstance(lay, IL)) if inplace is None else bool(inplace)
            if ip and not isinstance(lay, IL):
                nl = IL.__new__(IL); nl.__dict__.update(lay.__dict__)
            elif not ip and IL is not None and isinstance(lay, IL):
                from transformers.cache_utils import DynamicLayer
                nl = DynamicLayer.__new__(DynamicLayer); nl.__dict__.update({k: v for k, v in lay.__dict__.items() if k not in ("_kb", "_vb")})
            else:
                nl = copy.copy(lay)
            for a in ("keys", "values"):
                src = getattr(lay, a)
                if src is None:
                    continue
                dev = torch.device("cuda") if (src.device.type == "cpu" and torch.cuda.is_available()) else src.device
                n = src.shape[-2]; extra = int(IL.headroom) if ip else 0
                dst = torch.empty((src.shape[0] * int(B),) + tuple(src.shape[1:-2]) + (n + extra, src.shape[-1]), dtype=src.dtype, device=dev)
                for b in range(int(B)):
                    dst[b * src.shape[0]:(b + 1) * src.shape[0], ..., :n, :].copy_(src)
                if ip:                                   # the copy's own buffer with headroom (never the source's)
                    setattr(nl, "_kb" if a == "keys" else "_vb", dst); dst = dst[..., :n, :]
                setattr(nl, a, dst)
            c.layers.append(nl)
        return c

    @staticmethod
    def offload(cache):
        """Move a prefix cache to host memory (kept for later branch copies), so that B long rows fit beside the model in 40 GB."""
        for lay in cache.layers:
            for a in ("keys", "values"):
                t = getattr(lay, a)
                if t is not None and t.device.type != "cpu":
                    setattr(lay, a, t.to("cpu"))
            if hasattr(lay, "_kb"):                      # in-place buffer: released (the filled part went to the host above)
                lay._kb = lay._vb = None
        _t().cuda.empty_cache() if _t().cuda.is_available() else None
        return cache

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
        r = dict(kernel=self.kernel, calls=self.calls, tokens=self.tokens, seconds=round(self.secs, 2), versions=K.versions(),
                 inplace_kv=self.inplace)
        if torch.cuda.is_available():
            r.update(peak_memory_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3), gpu=torch.cuda.get_device_name(0),
                     tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32))
        return r


def letters_logodds(lg, a_id, b_id):
    """(log p(b) - log p(a), p(a) + p(b)) from logits [..., V]."""
    torch = _t()
    lp = torch.log_softmax(lg.double(), -1)
    return (lp[..., b_id] - lp[..., a_id]), (lp[..., a_id].exp() + lp[..., b_id].exp())
