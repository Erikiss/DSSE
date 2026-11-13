"""Planning simulation for D2 (two-voice carry), numpy only, no model.

Per prompt p and condition c, the stance score of speaker 1 in turn k follows a chain
  D_1 = s * mu + eta_1,   D_k = K_c * D_(k-1) + eta_k   (k = 2..4),
with eta ~ N(0, sig^2); the + and - texts of one prompt share Gumbel keys, modelled as a correlation rho_pm of eta,
and the conditions of one prompt share keys too (correlation rho_c of eta across conditions).
e_c(k) = mean_p (D_k(+) - D_k(-)) / 2 ;  Pi_c = mean_{k=2..4} e_c(k) / e_c(1).
Contrast dPi = Pi_OPP - Pi_NEU (paired over prompts); bootstrap over prompts.
"""
import numpy as np

rng = np.random.default_rng(3)


def sim(P=128, mu=3.0, sig=2.5, K_neu=0.47, K_opp=0.47, rho_pm=0.2, rho_c=0.2, n_boot=1000):
    # correlated noise: shared prompt component across conditions and signs
    def chains(K, shared):
        D = np.zeros((P, 2, 4))  # prompt, sign(+,-), turn
        for k in range(4):
            own = rng.normal(0, 1, size=(P, 2))
            eta = sig * (np.sqrt(rho_c) * shared[:, :, k] + np.sqrt(1 - rho_c) * own)
            if k == 0:
                D[:, 0, 0] = mu + eta[:, 0]
                D[:, 1, 0] = -mu + eta[:, 1]
            else:
                D[:, :, k] = K * D[:, :, k - 1] + eta
        return D
    # shared component with +/- correlation rho_pm
    z0 = rng.normal(0, 1, size=(P, 4)); z1 = rng.normal(0, 1, size=(P, 2, 4))
    shared = np.sqrt(rho_pm) * z0[:, None, :] + np.sqrt(1 - rho_pm) * z1
    Dn, Do = chains(K_neu, shared), chains(K_opp, shared)
    def pi(D, idx):
        e = (D[idx, 0, :] - D[idx, 1, :]).mean(0) / 2
        return e[1:].mean() / e[0], e
    idx = np.arange(P)
    pn, en = pi(Dn, idx); po, eo = pi(Do, idx)
    bs = []
    for _ in range(n_boot):
        j = rng.integers(0, P, P)
        bs.append(pi(Do, j)[0] - pi(Dn, j)[0])
    return po - pn, np.std(bs), en, pn


if __name__ == '__main__':
    print('P  mu  sig  K_neu K_opp rho | dPi   SD(dPi)  e_neu(1..4)      Pi_neu')
    for P in (128, 256):
        for mu, sig in ((3.0, 2.5), (3.0, 4.0), (2.0, 3.0)):
            for Kn, Ko in ((0.47, 0.47), (0.47, 0.62)):
                r = sim(P, mu, sig, Kn, Ko)
                print(f"{P:3d} {mu:3.1f} {sig:3.1f} {Kn:5.2f} {Ko:5.2f} 0.2 | {r[0]:+.3f} {r[1]:.3f}  "
                      f"{np.array2string(r[2], precision=2)} {r[3]:.3f}")
