"""SD of E = (A4a + A4b)/(A2a + A2b) under the spec's SD_A per block (normal model; loops independent or prompt-correlated)."""
import numpy as np
rng = np.random.default_rng(1)
def sdE(A, sdA, rho_w, rho_b, n=400000):
    # order: A2a, A4a, A2b, A4b
    C = np.full((4, 4), rho_b) ; C[0,1]=C[1,0]=C[2,3]=C[3,2]=rho_w; np.fill_diagonal(C,1.0)
    X = rng.multivariate_normal([A]*4, sdA**2*C, size=n)
    E = (X[:,1]+X[:,3])/(X[:,0]+X[:,2]); lo = np.percentile(E,5)
    return E.std(), np.percentile(E,[5,50,95])
for blk, sdA in (("14", 0.045), ("18", 0.032), ("6", 0.034)):
    for A in (0.25, 0.33, 0.5):
        r = [ (rw, rb, round(sdE(A, sdA, rw, rb)[0],3)) for rw, rb in ((0.5,0.0),(0.8,0.0),(0.8,0.5))]
        print(blk, "A", A, "SD(E) for (rho_within, rho_between):", r)
# 'substantial' tier power incl. E's lower bound >= 0.5 at true E = 1 (no decay) and E = 0.8
def tier_power(Abar_true, E_true, sdA, rw=0.8, rb=0.5, n=200000):
    A2 = Abar_true*2/(1+E_true); A4 = E_true*A2   # A_bar = mean(A3,A4) ~ approx (A2..A4 geometric); crude: A3 = sqrt(A2*A4)
    A3 = np.sqrt(A2*A4)
    mu = [A2,A3,A4]*2
    C = np.full((6,6), rb)
    for i in range(3):
        for j in range(3):
            C[i,j] = C[i+3,j+3] = (1.0 if i==j else rw)
    X = rng.multivariate_normal(mu, sdA**2*C, size=n)
    Abar = X[:,[1,2,4,5]].mean(1); E = (X[:,2]+X[:,5])/(X[:,0]+X[:,3])
    # bounds via the sampling SD of each statistic (stand-in for the bootstrap)
    sd_bar = Abar.std(); sd_e = E.std()
    ok = (Abar - 1.645*sd_bar >= 0.25) & (E - 1.645*sd_e >= 0.5)
    return round(((Abar - 1.645*sd_bar) >= 0.25).mean(),2), round(ok.mean(),2), round(sd_bar,3), round(sd_e,3)
for blk, sdA in (("14", 0.045), ("18", 0.032)):
    for Ab in (0.33, 0.36, 0.40):
        for Et in (1.0, 0.8):
            print("block", blk, "A_bar", Ab, "E", Et, "-> P(A_bar lo>=0.25), P(+E lo>=0.5), SD(A_bar), SD(E):", tier_power(Ab, Et, sdA))
