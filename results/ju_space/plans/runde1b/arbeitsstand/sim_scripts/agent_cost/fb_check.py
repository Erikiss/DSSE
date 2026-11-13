import numpy as np
rng = np.random.default_rng(0); k = 6
A = rng.normal(size=(k, k)); A *= 0.9 / max(abs(np.linalg.eigvals(A)))      # stable, non-normal VAR(1): x(t+1) = A x(t) + e
Q = rng.normal(size=(k, k)); Q = Q @ Q.T + 0.1 * np.eye(k)
C0 = np.eye(k)
for _ in range(5000): C0 = A @ C0 @ A.T + Q                                  # stationary covariance
C1 = C0 @ A.T                                                                # E[x(t) x(t+1)^T]
Sf = C0 - C1.T @ np.linalg.solve(C0, C1)                                     # residual cov of x(t+1) | x(t)
Sb = C0 - C1 @ np.linalg.solve(C0, C1.T)                                     # residual cov of x(t) | x(t+1)
print("det forward", np.linalg.det(Sf), "det backward", np.linalg.det(Sb))
r2f = 1 - np.diag(Sf) / np.diag(C0); r2b = 1 - np.diag(Sb) / np.diag(C0)
print("per-channel R2 forward ", np.round(r2f, 3)); print("per-channel R2 backward", np.round(r2b, 3))
print("trace forward", np.trace(Sf), "trace backward", np.trace(Sb))
