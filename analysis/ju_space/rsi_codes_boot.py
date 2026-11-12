"""Bootstrap over cells with refitted probes for README 10.3 (rule T4-A, condition 2: Braille minus U+2A00 on bits 0-2).
The per-cell bootstrap in rsi_codes.py (braille_minus_math_ops_boot) resamples the per-cell held-out accuracies while the fitted probes
stay fixed, so it leaves out the spread that comes from fitting. Here every resample draws the 51 cells with replacement (copies of a
cell stay on the same side of every split), refits both probes on every split and recomputes the difference. A delete-one-cell
jackknife (probes refitted, the same splits without the left-out cell) is a second estimate of the spread. Works on the states saved
by rsi_codes.py; no model.
Statistic: mean over splits of [held-out accuracy on bits 0-2 after the Braille prefix] minus [the same after the U+2A00 prefix], with
the same split of the cells for both sets (paired; rsi_codes.py drew separate splits for the two prefixes, so its differences deviate a
little). A resample holds only about 32 distinct cells, so its probes are fitted on about 16 instead of 25; where the difference is
large, the resampled differences come out smaller and the percentile interval is shifted toward 0 (boot_mean against diff). Point estimate and jackknife with 200 splits; 1,000 resamples with 100 splits each.
Probes as in rsi_codes.py (difference of means; ridge with lambda = 1e-2 * trace / n on centred states), computed from Gram matrices.
Usage: python rsi_codes_boot.py <rsi_codes.json.states.npz> <out.json>      env: WORKERS (processes, 3)"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np

BLOCKS = ["0", "2", "6", "10", "14", "18", "22", "27"]; N_SPLIT, N_BOOT, N_SPLIT_BOOT = 200, 1000, 100
bits012 = lambda k: [(k >> j) & 1 for j in range(3)]


def scores(G, tr, te, Y, kind):
    """Probe scores on the rows te for training rows tr (row indices into the Gram matrix G, repeats allowed) and labels Y [n_tr, 3]."""
    Gtt, C = G[np.ix_(tr, tr)], G[np.ix_(te, tr)]
    if kind == "dom":
        n1 = Y.sum(0); W1 = Y / np.maximum(n1, 1); W0 = (1 - Y) / np.maximum(len(Y) - n1, 1)
        return C @ (W1 - W0) - 0.5 * (np.einsum("ib,ij,jb->b", W1, Gtt, W1) - np.einsum("ib,ij,jb->b", W0, Gtt, W0))
    r = Gtt.mean(1); Gc = Gtt - r[:, None] - r[None, :] + Gtt.mean(); Cc = C - C.mean(1)[:, None] - r[None, :] + Gtt.mean()
    lam = 1e-2 * np.trace(Gc) / len(Gc); return np.linalg.solve(Gc + lam * np.eye(len(Gc)), Cc.T).T @ (2 * Y - 1)


def acc012(G, rows_of, cells_tr, cells_te, kind):
    """Held-out accuracy on bits 0-2 (mean over the three bits) for a split given as lists of cell instances (repeats allowed)."""
    tr = np.concatenate([rows_of[k] for k in cells_tr]); te = np.concatenate([rows_of[k] for k in cells_te])
    ytr = np.array([b for k in cells_tr for b in [bits012(k)] * len(rows_of[k])], float); yte = np.array([b for k in cells_te for b in [bits012(k)] * len(rows_of[k])], float)
    m = ytr.mean(0); ok = (m > 0) & (m < 1)
    if not ok.any(): return np.nan
    corr = (scores(G, tr, te, ytr, kind) > 0) == (yte > 0.5); return float(corr.mean(0)[ok].mean())


def diff(GB, rB, GM, rM, inst, side, kind):
    """Braille minus U+2A00 for one split: inst = cell of each instance, side = True for the training half."""
    a = acc012(GB, rB, inst[side], inst[~side], kind); b = acc012(GM, rM, inst[side], inst[~side], kind); return a - b


def run_block(args):
    path, l = args; z = np.load(path); t0 = time.time()
    XB, KB, XM, KM = z[f"braille_{l}"].astype(np.float64), z["K_braille"], z[f"math_ops_{l}"].astype(np.float64), z["K_math_ops"]
    GB, GM = XB @ XB.T, XM @ XM.T; cells = np.array(sorted(set(KB.tolist()) & set(KM.tolist())))
    rB = {int(k): np.flatnonzero(KB == k) for k in cells}; rM = {int(k): np.flatnonzero(KM == k) for k in cells}; out = {}
    for kind in ("dom", "ridge"):
        g = np.random.default_rng(11); halves = [g.permutation(len(cells)) < len(cells) // 2 for _ in range(N_SPLIT)]
        point = np.nanmean([diff(GB, rB, GM, rM, cells, h, kind) for h in halves])
        jk = []
        for c in range(len(cells)):      # delete-one-cell jackknife on the same splits
            keep = np.arange(len(cells)) != c; jk.append(np.nanmean([diff(GB, rB, GM, rM, cells[keep], h[keep], kind) for h in halves]))
        jk = np.array(jk); se_jk = float(np.sqrt((len(jk) - 1) / len(jk) * ((jk - jk.mean()) ** 2).sum()))
        g = np.random.default_rng(12); boot = []; ncells = []
        for _ in range(N_BOOT):
            inst = cells[g.integers(0, len(cells), len(cells))]; uniq = np.unique(inst); vals = []; ncells.append(len(uniq))
            for _ in range(N_SPLIT_BOOT):
                trc = set(g.permutation(uniq)[: len(uniq) // 2].tolist()); side = np.array([k in trc for k in inst]); vals.append(diff(GB, rB, GM, rM, inst, side, kind))
            boot.append(np.nanmean(vals))
        boot = np.array(boot)
        out[kind] = dict(diff=float(point), boot_lo=float(np.percentile(boot, 2.5)), boot_hi=float(np.percentile(boot, 97.5)), boot_sd=float(boot.std(ddof=1)), boot_mean=float(boot.mean()),
                         boot_share_below_point=float((boot < point).mean()), boot_mean_distinct_cells=float(np.mean(ncells)),
                         jackknife_se=se_jk, jackknife_lo=float(point - 1.96 * se_jk), jackknife_hi=float(point + 1.96 * se_jk))
    print(f"block {l}: " + " | ".join(f"{k} {v['diff']:+.3f} boot [{v['boot_lo']:+.3f}, {v['boot_hi']:+.3f}] sd {v['boot_sd']:.3f} jackknife se {v['jackknife_se']:.3f}" for k, v in out.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    return l, out


if __name__ == "__main__":
    path, dst = sys.argv[1], sys.argv[2]; t0 = time.time()
    with Pool(int(os.environ.get("WORKERS", "3"))) as P: res = dict(P.map(run_block, [(path, l) for l in BLOCKS]))
    json.dump(dict(statistic="mean over splits of held-out accuracy on bits 0-2, Braille minus U+2A00, paired splits by cell", n_splits=N_SPLIT, n_boot=N_BOOT,
                   n_splits_per_resample=N_SPLIT_BOOT, blocks={l: res[l] for l in BLOCKS}, seconds=time.time() - t0), open(dst, "w"), indent=1)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)
