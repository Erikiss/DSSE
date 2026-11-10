"""Shared helpers for the probe / interpretable-unit / mimicry experiment (Qwen3-1.7B, residual after block 21)."""
import glob, json
import numpy as np, torch
from safetensors import safe_open

BLOCK = 21
PAIRS = [("ai_accelerationism", "pro_trisolarans"), ("pro_trisolarans", "ai_accelerationism"), ("climate_denial", "flat_earth"),
         ("ai_accelerationism", "prefers_academia"), ("suggests_crime", "threatening_user"), ("acting_distressed", "threatening_user"),
         ("prefers_academia", "resisting_shutdown"), ("incorrect_answer", "acting_flirtatious")]          # (source A, target B)


def unit(a): return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def load_unembed(root):
    WU = g = None
    for f in glob.glob(f"{root}/models/qwen3-1.7b/*.safetensors"):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                if k.endswith("lm_head.weight") or (WU is None and k.endswith("embed_tokens.weight")): WU = sf.get_tensor(k).float().numpy()
                if k.endswith("model.norm.weight") or k == "norm.weight": g = sf.get_tensor(k).float().numpy()
    return WU, g.astype(np.float64)


def load_lens(root, block=BLOCK):
    return torch.load(f"{root}/jlens/Qwen3-1.7B_jacobian_lens.pt", map_location="cpu", weights_only=True)["J"][block].float().numpy().astype(np.float64)


def load_eval(out, block=BLOCK):
    items = json.load(open(f"{out}/eval_items.json")); lab = np.array([a for a, _, _ in items]); text = [c for _, _, c in items]
    X = np.load(f"{out}/eval2_states_block{block}.npy").astype(np.float64); F = np.load(f"{out}/eval2_final.npy").astype(np.float64)
    return lab, text, X, F


class SAE:
    """BatchTopK SAE (dictionary_learning format), used with its fixed threshold as at inference (JumpReLU)."""
    def __init__(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        self.We = sd["encoder.weight"].float().numpy(); self.be = sd["encoder.bias"].float().numpy()
        self.Wd = sd["decoder.weight"].float().numpy(); self.bd = sd["b_dec"].float().numpy()
        self.thr = float(sd["threshold"]); self.k = int(sd["k"]); self.n = self.We.shape[0]

    def encode(self, x, chunk=512):
        x = np.atleast_2d(x).astype(np.float32); out = []
        for i in range(0, len(x), chunk):
            a = np.maximum((x[i:i + chunk] - self.bd) @ self.We.T + self.be, 0); out.append(a * (a > self.thr))
        return np.concatenate(out)

    def decode(self, a): return a @ self.Wd.T + self.bd


def split3(lab, a, b, seed, fr=(0.35, 0.25)):
    """Per class: S1 (surrogate probe + mimicry density), S2 (independent probe), test (attack sources / real references)."""
    rng = np.random.default_rng(1000 + seed); S1, S2, T = [], [], []
    for c in (a, b):
        idx = rng.permutation(np.where(lab == c)[0]); n1, n2 = int(round(fr[0] * len(idx))), int(round(fr[1] * len(idx)))
        S1 += list(idx[:n1]); S2 += list(idx[n1:n1 + n2]); T += list(idx[n1 + n2:])
    return np.array(S1), np.array(S2), np.array(T)


class LinProbe:
    """Linear SVM on (x - m) / sc via the Gram matrix; score(x) > 0 means class B. C large = hard margin."""
    def __init__(self, X, y, C=1e4, m=None, sc=1.0):
        from sklearn.svm import SVC
        self.m = X.mean(0) if m is None else m; self.sc = sc; Z = (X - self.m) / sc
        clf = SVC(kernel="precomputed", C=C).fit(Z @ Z.T, y)
        self.w = (clf.dual_coef_[0] @ Z[clf.support_]) / sc; self.b = float(clf.intercept_[0]) - float(self.w @ self.m); self.n_sv = len(clf.support_)
        if (X @ self.w + self.b)[y == 1].mean() < (X @ self.w + self.b)[y == 0].mean(): self.w, self.b = -self.w, -self.b
    def score(self, X): return X @ self.w + self.b


def auroc(s, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < np.sum(y) < len(y) else float("nan")


def rmsnorm(z, g, eps=1e-6): return z / np.sqrt((z ** 2).mean(-1, keepdims=True) + eps) * g


def log_softmax(L):
    L = L - L.max(-1, keepdims=True); return L - np.log(np.exp(L).sum(-1, keepdims=True))
