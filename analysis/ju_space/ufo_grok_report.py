"""Tables from probe2.json of the toy runs (ufo_grok_probe2.py): baseline (weight decay only), UFO-regularised, label-permuted control.
Usage: python ufo_grok_report.py <out_dir>"""
import json, os, sys

out = sys.argv[1]; RUNS = [("base", "weight decay only"), ("ufo", "with UFO"), ("perm", "answers permuted (cannot generalise)")]
for r, title in RUNS:
    p = f"{out}/grok_{r}_s0_wd0.3/probe2.json"
    if not os.path.exists(p): continue
    P = json.load(open(p)); print(f"\n=== {title}: reaches 99 % on held-out problems at step {P['grok']}")
    print(f"{'step':>6s} {'train':>5s} {'held':>5s} {'|w|':>5s} | {'varying nodes':>13s} {'PR':>9s} {'PCA8 mlp0 fit/held':>19s} {'PCA8 mlp1 fit/held':>19s} {'PCA2 mlp1 held':>14s} {'AE2 mlp0 fit/held':>18s} {'AE2 mlp1 fit/held':>18s} {'by z mlp0/1':>12s} | {'k99':>3s} | same answer, training problems, f=0.3 (1.0): cov top8 bot64 rand | block frozen: cov bot64 | q_mlp cov bot64 | held: cov bot64")
    for row in P["rows"]:
        c0, c1 = row["comp"]["mlp0"], row["comp"]["mlp1"]; g = lambda c, k: c.get(k, float("nan")); ae = lambda c: f"{c['ae2_fit_1500']:.2f}/{c['ae2_held_1500']:.2f}" if "ae2_fit_1500" in c else "-"
        line = f"{row['step']:6d} {row['train_acc']:5.2f} {row['val_acc']:5.3f} {(row['wnorm'] or 0):5.1f} | {g(c0, 'nodes'):6.0f}/{g(c1, 'nodes'):<6.0f} {g(c0, 'participation_ratio'):4.0f}/{g(c1, 'participation_ratio'):<4.0f} {g(c0, 'pca8_fit'):9.2f}/{g(c0, 'pca8_held'):<9.2f} {g(c1, 'pca8_fit'):9.2f}/{g(c1, 'pca8_held'):<9.2f} {g(c1, 'pca2_held'):14.2f} {ae(c0):>18s} {ae(c1):>18s} {g(c0, 'by_z'):5.2f}/{g(c1, 'by_z'):<5.2f} | {row['site']['k99']:3d} | "
        if "train_cov_0.3" in row:
            s = lambda cl, f, k="same_top", se="train": row[f"{se}_{cl}_{f}"][k]
            line += " ".join(f"{s(cl, '0.3'):.2f} ({s(cl, '1'):.2f})" for cl in ("cov", "top8", "bot64", "rand")) + f" | {s('cov', '0.3', 'same_top_frozen'):.2f} {s('bot64', '0.3', 'same_top_frozen'):.2f} | {s('cov', '0.3', 'q_mlp'):+.2f} {s('bot64', '0.3', 'q_mlp'):+.2f} | {s('cov', '0.3', se='held'):.2f} {s('bot64', '0.3', se='held'):.2f}"
        else: line += "state at '=' is the same for all problems: no test"
        print(line)
