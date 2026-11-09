"""Run one of the Colab notebooks on a plain GPU server, without Jupyter and without Google Drive.

The code cells of the built notebook run in one Python process, in order; shell lines (!pip ...) are skipped because
setup.sh installs the packages. Outside Colab the notebooks write to RESULTS_ROOT instead of Drive, with the same
folder names as in Colab (…_runs_s{i}_of{n}), so results can later be copied into the Drive folder.

Modes:
  default          one shard, all cells (training, generation, judge, analysis)
  --prefetch       only configuration, storage and data cells (downloads the data once, so parallel shards do not race)
  --analysis-only  configuration, storage and the last (analysis) cell over all shard folders under RESULTS_ROOT

Usage: python3 runner.py superposition --shard 0 --results /workspace/dsse_superposition
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOKS = {"superposition": "superposition_a100.ipynb", "entkopplung": "entkopplung_a100.ipynb"}


def code_cells(path):
    cells = []
    for c in json.load(open(path))["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "\n".join(l for l in "".join(c["source"]).split("\n") if not l.lstrip().startswith("!"))
        if src.strip():
            cells.append(src)
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=sorted(NOTEBOOKS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=3)
    ap.add_argument("--results", required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--prefetch", action="store_true")
    mode.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args()

    os.environ.update(N_SHARDS=str(a.n_shards), SHARD_ID=str(a.shard), RESULTS_ROOT=str(Path(a.results).resolve()))
    import matplotlib
    matplotlib.use("Agg")

    cells = code_cells(HERE.parent / NOTEBOOKS[a.experiment])
    find = lambda key: next(i for i, s in enumerate(cells) if key in s)
    i_cfg, i_store, i_data = find("MODEL_ID"), find("def load_json"), find("datasets/Transluce/DSSE")
    if a.prefetch:
        sel = cells[:i_data + 1]
    elif a.analysis_only:
        sel = [cells[i_cfg], cells[i_store], cells[-1]]
    else:
        sel = cells

    g = {"__name__": "__main__"}
    for k, src in enumerate(sel, 1):
        t0 = time.time()
        first = next((l for l in src.splitlines() if l.strip()), "")[:90]
        print(f"\n===== {time.strftime('%H:%M:%S')} Zelle {k}/{len(sel)}: {first}", flush=True)
        try:
            exec(compile(src, f"<{a.experiment} Zelle {k}>", "exec"), g)
        except SystemExit:
            raise
        except BaseException:
            traceback.print_exc()
            print(f"ABBRUCH in Zelle {k}", flush=True)
            sys.exit(1)
        print(f"===== Zelle {k} fertig ({time.time() - t0:.0f}s)", flush=True)

    if a.prefetch and {"BEH", "fetch", "HF", "DATA"} <= g.keys():
        for beh in g["BEH"].values():    # Superposition: DSSE-Bewertungen für die Eichung des Judges gleich mitladen
            g["fetch"](g["HF"] + f"eval_results/{beh}.jsonl.xz", g["DATA"] / f"eval_{beh}.jsonl.xz")
    print("FERTIG", flush=True)


if __name__ == "__main__":
    main()
