from statistics import NormalDist
N = NormalDist()
z = lambda p: N.inv_cdf(1 - p)
pw = z(0.2)  # 0.8416
def dz(n, alpha_one_sided):
    return (z(alpha_one_sided) + pw) / n ** 0.5
print('paired d_z MDE (power 0.8):')
for n, a, lab in [(256, .05, 'K4 n=256 one-sided .05'), (384, .05, 'K4/K6 n=384'), (96, .025, 'K6c n=96 items, Holm-1 (.025)'),
                  (192, .025, 'K6c n=192 cells-as-items upper bound'), (700, .10/7/2, 'K8a n=700, Holm m=7, two-sided .10'),
                  (400, .10/7/2, 'K8a ACH only n=400'), (600, .025, 'K3d composite n=600, Holm-1'), (256, .025, 'K8b sessions n=256, Holm-1')]:
    print(f'  {lab:45s} d_z >= {dz(n, a):.3f}')
# binary paired outcome: SD of paired differences for discordance rate q: sqrt(q); accuracy MDE = d_z * sqrt(q)
for q in (0.15, 0.25):
    print(f'  K8a accuracy-point MDE at discordance {q}: {dz(700, .10/7/2) * q ** 0.5:.3f} (n=700), {dz(400, .10/7/2) * q ** 0.5:.3f} (n=400)')
# AUROC SE (Hanley-McNeil)
def auc_se(A, n1, n0):
    q1 = A / (2 - A); q2 = 2 * A * A / (1 + A)
    return ((A * (1 - A) + (n1 - 1) * (q1 - A * A) + (n0 - 1) * (q2 - A * A)) / (n1 * n0)) ** 0.5
for n in (250, 350, 500):
    s = auc_se(0.70, n, n)
    d = 2 ** 0.5 * s * 0.6  # difference of two correlated AUCs on the same items (rho ~0.64 assumed)
    print(f'  AUROC SE at 0.70, {n}+{n}: {s:.4f}; diff SE ~{d:.4f}; margin MDE over 0.05 bar (one-sided .025, power .8): {0.05 + (z(.025)+pw)*d:.3f}')
# hazard coefficient MDE: SE ~ deff^0.5 / sqrt(E)
for E in (150, 300, 500):
    for deff in (1.5,):
        se = (deff / E) ** 0.5
        print(f'  K8c hazard beta MDE per SD, E={E}, deff={deff}: {(z(.05)+pw)*se:.3f} (OR {2.718281828**((z(.05)+pw)*se):.2f})')
# K_H ratio-of-sums: SE(K) ~ K*sqrt(rn^2+rd^2); MDE for 'lower bound > 1' (one-sided .025 Holm-1) and 'upper bound < 1' (.05)
for r in (0.06, 0.10, 0.15, 0.20):
    se1 = (2 * r * r) ** 0.5
    print(f'  K_H rel. SE of sums {r:.2f}: SE(K at 1) {se1:.3f}; sustaining needs K >= {1 + (z(.025)+pw)*se1:.2f}; dissipative detectable if K <= {1/(1 + (z(.05)+pw)*se1):.2f}')
# bistability: high-share difference, 384 sessions vs 384 clean
p0 = 0.10
for p1 in (0.17, 0.20, 0.25):
    se = (p1*(1-p1)/384 + p0*(1-p0)/384) ** 0.5
    print(f'  bistability high-share {p1} vs {p0}: z = {(p1-p0)/se:.2f} (needs {z(.025)+pw:.2f} for Holm-1 power .8)')
