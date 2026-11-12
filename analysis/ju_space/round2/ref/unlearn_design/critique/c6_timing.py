"""Critique check 6: timing of one column search (pilot.search, 8 shots, 300 steps) free and restricted (B of 100 dims), and of one
3,000-step search, per width, on one thread, to check the design's T5 / T4 budget."""
import sys, time
import numpy as np, torch
SP = "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad"
sys.path.insert(0, SP + "/unlearn_design"); torch.set_num_threads(1)
import pilot as PL
from pc_common import MLP
M, P, NPRE, NSLOT, BASE = PL.M, PL.P, PL.NPRE, PL.NSLOT, PL.BASE
for W, path in ((64, f"{SP}/unlearn_design/critique/models/bp_mup_W64_s0.pt"), (128, f"{SP}/unlearn_design/models/bp_mup_W128_s0.pt"), (256, f"{SP}/unlearn_design/critique/models/bp_mup_W256_s0.pt")):
    m = MLP(2 * P + NSLOT, W, P, param="mup", base_width=BASE); m.load_state_dict(torch.load(path)); m.eval(); world = M.world(0); new = world[1]
    B = torch.as_tensor(np.linalg.qr(np.random.default_rng(0).standard_normal((W, min(100, W - 4))))[0], dtype=torch.float32)
    t = time.time(); [PL.search(m, world, NPRE, new[0]) for _ in range(5)]; tf = (time.time() - t) / 5
    t = time.time(); [PL.search(m, world, NPRE, new[0], B=B) for _ in range(5)]; tb = (time.time() - t) / 5
    t = time.time(); PL.search(m, world, NPRE, new[0], steps=3000, evals=(3000,)); t3 = time.time() - t
    print(f"W={W}: 300-step search free {tf:.2f}s, restricted {tb:.2f}s; 3000-step {t3:.1f}s", flush=True)
