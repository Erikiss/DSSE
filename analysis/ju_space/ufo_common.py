"""Shared by the ufo_*.py scripts for Qwen3-1.7B: text selection, token roles, model loading, pool statistics.
Texts: the first NMAIN wikitext passages and NMAIN chats (question + answer of a fine-tuned model, DSSE) are the texts that
are perturbed; all other wikitext passages and a disjoint set of chats form the pool from which the statistics of real states
are taken. A position is 'valid' if it is not one of the first four tokens and not a token of the chat template."""
import json, os
import numpy as np, torch

FIRST = 4; NMAIN = int(os.environ.get("NMAIN", "8")); MAXT = int(os.environ.get("MAXT", "64")); NPOOLCHAT = int(os.environ.get("NPOOLCHAT", "160")); POOLWIKIT = 192


def load_model(root, randinit=False):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    path = f"{root}/models/qwen3-1.7b"; tok = AutoTokenizer.from_pretrained(path)
    if randinit: torch.manual_seed(1234); model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(path)).float().eval()   # same architecture, untrained
    else: model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    dev = os.environ.get("DEVICE", "cpu")                     # GPU runs (Colab): float32 without TF32, so the numbers match the CPU runs up to rounding
    if dev != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; model = model.to(dev)
    return tok, model


def texts(root, tok):
    """Returns (main, pool): lists of (source, token ids [1, T], valid mask [T])."""
    wiki = json.load(open(f"{root}/ju_out/corpus_wikitext.json")); pairs = json.load(open(f"{root}/ju_out/axes_pairs.json")); pick = np.random.default_rng(0).permutation(len(pairs))
    chat = lambda i: tok.apply_chat_template([{"role": "user", "content": pairs[i]["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False) + pairs[i]["not_shown"]

    def prep(src, t, maxt):
        ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :maxt]; valid = ~template_mask(tok, ids) if src == "chat" else torch.ones(ids.shape[1], dtype=torch.bool)
        valid[:FIRST] = False; return (src, ids, valid)
    main = [prep("wiki", t, MAXT) for t in wiki[:NMAIN]] + [prep("chat", chat(i), MAXT) for i in pick[:NMAIN]]
    pool = [prep("wiki", t, POOLWIKIT) for t in wiki[NMAIN:]] + [prep("chat", chat(i), MAXT) for i in pick[NMAIN:NMAIN + NPOOLCHAT]]
    return main, pool


def template_mask(tok, ids):
    """True for the tokens of the chat template: special tokens and, after each of them, the role name and line breaks."""
    special = set(tok.all_special_ids) | {i for t, i in tok.get_added_vocab().items()}; role = {tok.encode(w, add_special_tokens=False)[0] for w in ("user", "assistant", "system")}; T = ids.shape[1]; tmpl = torch.zeros(T, dtype=torch.bool)
    for p in range(T):
        if int(ids[0, p]) in special:
            tmpl[p] = True
            for q in (p + 1, p + 2):
                if q < T and (int(ids[0, q]) in role or tok.decode(ids[0, q]).strip() == ""): tmpl[q] = True
    return tmpl


unit = lambda v: v / (v.norm(dim=-1, keepdim=True) + 1e-9)


class Pool:
    """Statistics of real states after every block (written by ufo_pool.py)."""
    def __init__(s, path, blocks=None):
        z = torch.load(path, map_location="cpu", weights_only=True); s.stats = z["stats"]; s.b = {m: z["blocks"][m] for m in (blocks if blocks is not None else z["blocks"].keys())}

    def med(s, m): return float(s.b[m]["med_norm"])

    def direction(s, kind, m, T, G):
        """T unit vectors of one class at block m. rand: isotropic. cov: Gaussian with the covariance of real states. mix: difference of two
        real states. irr: random vector outside the principal subspace that holds all but about 2 % of the variance of held-out real states,
        with the direction of the mean state removed."""
        b = s.b[m]; D = b["mu"].shape[0]
        if kind == "rand": return unit(torch.randn(T, D, generator=G))
        if kind == "cov": return unit((torch.randn(T, D, generator=G) * b["sd"]) @ b["U"].T)
        if kind == "irr":
            k = int(b["k_irr"]); v = torch.randn(T, D - k, generator=G) @ b["U"][:, k:].T; mh = unit(b["mu"]); return unit(v - (v @ mh)[:, None] * mh)
        if kind == "mix":
            X = b["sample"]; i = torch.randint(0, len(X), (T,), generator=G); j = (i + torch.randint(1, len(X), (T,), generator=G)) % len(X); return unit(X[i] - X[j])      # j is never i
        raise ValueError(kind)

    def inside(s, m, v):
        """Share of the energy of v [.., D] that lies inside the principal subspace used for 'irr' (top k_irr axes)."""
        b = s.b[m]; k = int(b["k_irr"]); return float(((v @ b["U"][:, :k]) ** 2).sum() / (v ** 2).sum())
