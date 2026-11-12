"""README section 12: compact summary of the job files of pc_meta.py (one row per job, a few hundred bytes), so that the analysis
(pc_analyze.py) can run on the rows from several GPU shards. Usage: python pc_summary.py <out_dir> <summary.json>"""
import glob, json, os, sys
import numpy as np


def row(r, name):
    s = dict(name=name, arm=r["arm"], W=r["W"], seed=r["seed"], K=r["K"], gamma=r["gamma"], wd_scale=r["wd_scale"], lr=r["lr"], T=r.get("T"), beta=r.get("beta"),
             first95=r.get("first95"), checkpoint=r.get("checkpoint"), level1=bool(r.get("level1")), final_train=r.get("final_train"), final_heldout=r.get("final_heldout"),
             coord_check=r.get("coord_check"), seconds=r.get("seconds", r.get("pretrain_seconds")), delta=r.get("delta"))
    ft = r.get("finetune")
    if ft:
        f8 = ft["8"] if "8" in ft else ft[8]; f4 = ft["4"] if "4" in ft else ft[4]
        s.update(B2_8=f8["B2"], B2_4=f4["B2"], A8_mean=float(np.mean([np.mean(a) for a in f8["A"]])), F8_mean=float(np.mean([np.mean(f) for f in f8["F"]])),
                 F4_mean=float(np.mean([np.mean(f) for f in f4["F"]])), B2_all=ft["all8"]["B2_all"], retention_mean=float(np.mean(ft["all8"]["retention"])))
    pc = r.get("pc") or {}
    if pc.get("ratio"):
        ratio = np.array(pc["ratio"], float); rose = np.array(pc["rose"], bool); ck = r.get("checkpoint") or len(ratio)
        cos = [(st, c) for st, c in pc.get("cos", [])]; early = [c[1] for st, c in cos if st <= 0.2 * ck]
        s["pc"] = dict(ratio_median=float(np.median(ratio)), ratio_le005_share=float((ratio <= 0.05).mean()), rose_share=float(rose.mean()), n_updates=int(len(ratio)),
                       cos_l2_early_median=float(np.median(early)) if early else None, cos_last=cos[-1][1] if cos else None,
                       lam_min=float(min(l for _, l in pc["lam"])) if pc.get("lam") else None, lam_max=float(max(l for _, l in pc["lam"])) if pc.get("lam") else None)
    return s


def main():
    out, dst = sys.argv[1], sys.argv[2]; rows = []
    for p in sorted(glob.glob(f"{out}/jobs/*.json")):
        try: rows.append(row(json.load(open(p)), os.path.basename(p)[:-5]))
        except Exception as e: rows.append(dict(name=os.path.basename(p)[:-5], error=repr(e)))
    json.dump(rows, open(dst, "w"), indent=0); print(f"{len(rows)} jobs summarised -> {dst}")


if __name__ == "__main__":
    main()
