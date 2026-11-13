"""Part R package files (frozen with the plan, R.2): writes / checks results/ju_space/plans/runde1b/{neutral_pool_1b.txt, wordsets_1b.json,
label_bank_1b.json, prompts_1b.json}. Model-free.
    python r1b_r_package.py build [--new <new192.json>] [--out <dir>]   neutral pool (Appendix R.A, sha256 pinned), WORDSETS, R5.5 bank;
                                                                       with --new (a JSON list of the 192 new prompts, 96 RSI-topic then
                                                                       96 scene) also prompts_1b.json in P256 order with folds
    python r1b_r_package.py check [<prompts_1b.json>]                  r1b_prompts_check (R2.1): counts, 5-16 words, one sentence ending
                                                                       in '.' or '?', banned words, near-duplicates (trigram Jaccard >= 0.6)"""
import json, os, re, sys
import r1b_core as K
import r1b_r_lib as R

BANNED_ALL = set(R.W_SEA) | set(R.W_SPORTS) | set(R.W_TONE) | {"water"}
SCENE_BANNED = {"ai", "artificial", "intelligence", "learn", "learning", "learns", "learned", "improve", "improves", "improvement", "improving",
                "model", "models", "machine", "algorithm", "algorithms", "train", "training", "skill", "skills", "better", "upgrade", "optimize"}


def trigrams(s):
    s = re.sub(r"\s+", " ", s.lower()); return {s[i:i + 3] for i in range(len(s) - 2)}


def check(prompts):
    """Errors of a P256 list (the 64 round-1 prompts at 0-31 / 128-159 unchanged)."""
    errs = []; r1 = R.r1_prompts()
    if len(prompts) != 256:
        return [f"{len(prompts)} prompts, expected 256"]
    if prompts[0:32] != r1[:32] or prompts[128:160] != r1[32:]:
        errs.append("round-1 prompts not at indices 0-31 / 128-159")
    new = [(q, prompts[q]) for q in list(range(32, 128)) + list(range(160, 256))]
    for q, p in new:
        w = re.findall(r"[A-Za-z']+", p); wl = {x.lower() for x in w}
        if not 5 <= len(w) <= 16:
            errs.append(f"{q}: {len(w)} words")
        if not p.endswith((".", "?")) or re.search(r"[.?!]\s", p[:-1]):
            errs.append(f"{q}: not one sentence ending in '.' or '?'")
        if wl & BANNED_ALL:
            errs.append(f"{q}: banned words {sorted(wl & BANNED_ALL)}")
        if q >= 160 and wl & SCENE_BANNED:
            errs.append(f"{q}: AI / learning / improvement words in a scene prompt {sorted(wl & SCENE_BANNED)}")
    allp = [(q, p) for q, p in enumerate(prompts)]
    for i, (q, p) in enumerate(new):
        tp = trigrams(p)
        for q2, p2 in allp:
            if q2 != q and (q2 < q or not (32 <= q2 < 128 or q2 >= 160)):
                t2 = trigrams(p2); j = len(tp & t2) / max(len(tp | t2), 1)
                if j >= 0.6:
                    errs.append(f"{q} near-duplicate of {q2} (Jaccard {j:.2f})")
    return errs


def build(out, new=None):
    os.makedirs(out, exist_ok=True); pool = R.neutral_pool()
    open(os.path.join(out, "neutral_pool_1b.txt"), "w").write("".join(w + "\n" for w in pool))
    K.dump_json(dict(WORDSETS=R.WORDSETS, sha256=K.sha256_text(K.canon(R.WORDSETS)), note="R2.1; neutral sets are drawn in r1b_R_S3 (rng 4242 / 4243)"),
                os.path.join(out, "wordsets_1b.json"))
    from r1b_r_s5 import BANK
    K.dump_json(dict(bank=[dict(name=n, pro=p, anti=a) for n, p, a in BANK], sha256=K.sha256_text(K.canon(BANK)), status="draft wording (R5.5)"),
                os.path.join(out, "label_bank_1b.json"))
    if new:
        nw = json.load(open(new)); assert len(nw) == 192, "192 new prompts expected"; r1 = R.r1_prompts()
        pr = r1[:32] + nw[:96] + r1[32:] + nw[96:]; e = check(pr)
        if e:
            raise SystemExit("prompt check failed:\n" + "\n".join(e[:50]))
        K.dump_json(dict(prompts=pr, fold=R.p256_fold().tolist(), order="[round-1 RSI 0-31, new RSI 32-127, round-1 scene 128-159, new scene 160-255]",
                         fold_rule="round-1 fold for the old prompts; rng(105) halves within type for the new ones"), os.path.join(out, "prompts_1b.json"))
    return {f: K.sha256_file(os.path.join(out, f)) for f in sorted(os.listdir(out)) if f in ("neutral_pool_1b.txt", "wordsets_1b.json", "label_bank_1b.json",
                                                                                              "prompts_1b.json")}


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "build":
        out = a[a.index("--out") + 1] if "--out" in a else R.PKG
        print(json.dumps(build(out, a[a.index("--new") + 1] if "--new" in a else None), indent=1))
    elif a and a[0] == "check":
        p = a[1] if len(a) > 1 else os.path.join(R.PKG, "prompts_1b.json"); e = check(K.load_json(p)["prompts"])
        print("\n".join(e) if e else "prompts_1b.json OK"); raise SystemExit(1 if e else 0)
    else:
        print(__doc__)
