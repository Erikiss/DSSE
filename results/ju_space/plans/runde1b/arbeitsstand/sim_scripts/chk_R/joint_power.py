"""Joint power of conjunctive part-R rules vs the per-row MDEs stated in the spec (normal model, no model run).
PC P1: rows n=2,3,4 each need A_lo90 >= 0.10 and z >= 2.326 (p_s <= 0.01 one-sided). Rows correlated (rho).
'likely, substantial': A_bar lower bound >= 0.25 AND MANIP (2 loops: A_1 lo >= 0.10, z >= 2.326) AND p_s at n=2..4 both loops
AND E lower bound >= 0.5 (SD_E 0.05) -- PC not included (would multiply further)."""
import numpy as np
rng = np.random.default_rng(0)
Z1, Z01 = 1.645, 2.326
def pc_p1_power(A, sd, rho, n=200000):
    C = sd**2 * (rho * np.ones((3, 3)) + (1 - rho) * np.eye(3))
    X = rng.multivariate_normal([A] * 3, C, size=n)
    ok = ((X - Z1 * sd) >= 0.10) & (X / sd >= Z01)
    return ok.all(1).mean(), ok[:, 0].mean()
for sd, lab in ((0.045, "block 14 (SD_A 0.042-0.051)"), (0.033, "blocks 18/6 (SD_A 0.030-0.034)")):
    for rho in (0.5, 0.8):
        out = []
        for A in np.arange(0.16, 0.34, 0.02):
            pj, p1 = pc_p1_power(A, sd, rho)
            out.append(f"A={A:.2f}: row {p1:.2f} joint {pj:.2f}")
        print(lab, "rho", rho); print("  " + "; ".join(out))
# joint MDE (power 0.8) for P1 alone
def mde(sd, rho, target=0.8):
    for A in np.arange(0.10, 0.60, 0.002):
        if pc_p1_power(A, sd, rho, 100000)[0] >= target: return A
for sd in (0.045, 0.033):
    print("P1 joint MDE80", sd, [round(mde(sd, r), 3) for r in (0.5, 0.8)], "per-row stated:", round(max(0.10 + 2.487 * sd, 3.168 * sd), 3))
