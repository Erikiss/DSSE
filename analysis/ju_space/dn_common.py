"""Shared by the dn_*.py scripts (README section 11): control of the SwiGLU activations a = silu(g) * u of Qwen3-1.7B.
Every MLP is replaced by an equivalent forward (down_proj(act(gate(x)) * up(x))) that hands (block, g, u, a) to a hook before
down_proj; the hook returns the activations to write. Quadrants of (g, u): Q1 g > 0, u > 0 (a > 0); Q2 g <= 0, u > 0 (a <= 0);
Q3 g <= 0, u <= 0 (a >= 0, the 'double negative'); Q4 g > 0, u <= 0 (a <= 0).
The hook sees a mask of the positions that may be changed (CTL.mask, [batch, T] or None = all), set by the caller for the whole
sequence before a teacher-forced forward pass; with a KV cache (generation) the caller sets CTL.mask_fn(position) instead."""
import torch

Q = {"Q1": (1, 1), "Q2": (0, 1), "Q3": (0, 0), "Q4": (1, 0)}


def quad(g, u, q):
    """Boolean mask of quadrant q ('Q1'..'Q4') for gate input g and up value u."""
    gp, up = Q[q]; return ((g > 0) == bool(gp)) & ((u > 0) == bool(up))


class MLPControl:
    def __init__(self, model):
        self.layers = model.model.layers; self.hook = None; self.post = None; self.mask = None; self.blocks = None
        for l, layer in enumerate(self.layers): layer.mlp.forward = self._wrap(l, layer.mlp)

    def _wrap(self, l, mlp):
        def fwd(x):
            g = mlp.gate_proj(x); u = mlp.up_proj(x); a = mlp.act_fn(g) * u; on = self.blocks is None or l in self.blocks
            if self.hook is not None and on:
                b = self.hook(l, g, u, a)
                if self.mask is not None: m = self.mask[:, :a.shape[1], None]; b = torch.where(m, b, a)
                a = b
            y = mlp.down_proj(a)
            if self.post is not None and on:      # a change of the MLP output itself (for the noise controls)
                z = self.post(l, y)
                if self.mask is not None: m = self.mask[:, :y.shape[1], None]; z = torch.where(m, z, y)
                y = z
            return y
        return fwd

    def set(self, hook=None, blocks=None, mask=None, post=None):
        self.hook, self.post, self.blocks, self.mask = hook, post, (None if blocks is None else set(blocks)), mask


@torch.no_grad()
def logits_of(model, ids):
    return model(ids).logits.double()


def kl_ce(lp_ref, lp, ids, valid):
    """Per-position KL(ref || changed) of the next-token distribution, cross-entropy of the actual next token under the changed
    model, and whether the most likely token is the same, at the valid positions that have a next token. lp_*: log-softmax [T, V]."""
    v = valid.clone(); v[-1] = False; pos = torch.nonzero(v).flatten(); nxt = ids[0, pos + 1]
    kl = (lp_ref[pos].exp() * (lp_ref[pos] - lp[pos])).sum(-1); ce = -lp[pos, nxt]; same = (lp_ref[pos].argmax(-1) == lp[pos].argmax(-1)).double()
    return kl, ce, same
