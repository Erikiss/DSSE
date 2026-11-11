"""Numerical checks of the statements on bisection, trisection and quadratic units used in README section 9.4.
1 depress      the substitution t = x + a/n removes the term of degree n - 1 of x^n + a x^(n-1) + ... (n = 2: completing the square)
2 roots        cubic with three real roots r_k: mean = -a/3; r_k = -a/3 + 2 rho cos(theta/3 - 2 pi k/3) with p = b - a^2/3,
               q = 2a^3/27 - ab/3 + c, rho = sqrt(-p/3), cos(theta) = -q / (2 rho^3); the discrete Fourier transform over Z/3 of
               the roots is (3 * mean, L, conj L) with the Lagrange resolvent L = r_0 + w r_1 + w^2 r_2 = 3 rho exp(i theta/3),
               and L^3 = 27 (-q/2 + i sqrt(-(q^2/4 + p^3/27))) is known from the coefficients: trisecting = taking its cube root
3 tripling     with t = 2 rho cos(phi) and p = -3 rho^2: t^3 + p t = 2 rho^3 cos(3 phi) (degree 2: (2 cos phi)^2 - 2 = 2 cos(2 phi))
4 trisector    Pehlivan & Williams (Arhimede Math. J. 13/1, 2026), Theorem 1: the length s of the angle trisector CQ satisfies
               R s^3 + S s^2 + T = 0 with R = a^2 c^2 - (a^2 - b^2)^2, S = 3 a b^2 c^2, T = -a b^2 (a+b+c)(a+b-c)(a-b+c)(-a+b+c).
               Checked against s = a b sin(3 th) / (b sin(th) + a sin(2 th)), th = C/3 (the trisector next to side b), on random
               triangles; in w = 1/s the equation is T w^3 + S w + R = 0, a depressed cubic (Cardano's shift is zero);
               and 1/s = (b + 2 a cos(th)) / (a b (4 cos(th)^2 - 1))
5 bisectors    the quadratic form (g.x)(u.x) has the non-zero eigenvalues (g.u +- |g||u|)/2 = |g||u|(cos(phi) +- 1)/2 with
               eigenvectors along g/|g| +- u/|u|, the two bisectors of g and u
6 chebyshev    for the line from C that makes the angle C/n with CA and has the length s_n up to AB, and the altitude h from C:
               T_n(h / s_n) = cos(n A + C - n pi/2) (T_n the Chebyshev polynomial). n = 2, the bisector: 2 (h/l)^2 - 1 = cos(A - B),
               the half-angle formula; n = 3, the trisector: 4 X^3 - 3 X = sin(2A - B) with X = h/s, which is the cubic of 4 in 1/s
7 circle       on the unit circle the product of two unit linear forms with directions p1, p2 is (cos(2 psi - p1 - p2) +
               cos(p1 - p2)) / 2: a constant and a second harmonic whose phase is the bisector; the product of three is
               (cos(3 psi - p1 - p2 - p3) + three first harmonics) / 4: the third harmonic has its phase at the mean direction
8 factors      the quadratic x^2 + a x = x (x + a) is a product of two linear forms 45 degrees apart; for uncorrelated inputs of
               equal variance the main direction of its gradient is exactly their bisector (22.5 degrees, slope sqrt(2) - 1),
               while completing the square uses x + a/2 (slope 1/2). For the cubic part x^2 (x + a) the mean of the three factor
               directions is 15 degrees (slope 0.268), Cardano's shift has slope 1/3; main direction of the gradient for comparison
9 half angle   cos(C/3) as a function of c = cos(C) has a square-root singularity at C = 180 degrees; as a function of
               v = cos(C/2) it is smooth: maximum error of least-squares polynomials of degree 2, 4, 8 in c and in v
Usage: python trisect_math.py <out.json>"""
import json, sys
import numpy as np

rng = np.random.default_rng(0); res = {}
# 1
worst = 0.0
for n in (2, 3, 4, 5, 6):
    for _ in range(200):
        co = np.concatenate([[1.0], rng.uniform(-3, 3, n)]); shift = np.poly1d([1.0, -co[1] / n]); dep = np.poly1d(co)(shift); worst = max(worst, abs(dep.coeffs[1]))
res["depress"] = dict(largest_coefficient_of_degree_n_minus_1_after_the_shift=float(worst), degrees=[2, 3, 4, 5, 6], trials_per_degree=200)
# 2 and 3
e = dict(mean=0.0, roots=0.0, resolvent=0.0, cube=0.0, tripling=0.0, doubling=0.0); w = np.exp(2j * np.pi / 3)
for _ in range(2000):
    r = np.sort(rng.uniform(-3, 3, 3))[::-1]; a, b, c = -r.sum(), r[0] * r[1] + r[0] * r[2] + r[1] * r[2], -r.prod(); p, q = b - a * a / 3, 2 * a ** 3 / 27 - a * b / 3 + c; rho = np.sqrt(-p / 3); th = np.arccos(np.clip(-q / (2 * rho ** 3), -1, 1))
    rk = -a / 3 + 2 * rho * np.cos(th / 3 - 2 * np.pi * np.arange(3) / 3); L = rk[0] + w * rk[1] + w * w * rk[2]
    e["mean"] = max(e["mean"], abs(r.mean() + a / 3)); e["roots"] = max(e["roots"], np.abs(np.sort(rk)[::-1] - r).max()); e["resolvent"] = max(e["resolvent"], abs(L - 3 * rho * np.exp(1j * th / 3)))
    e["cube"] = max(e["cube"], abs(L ** 3 - 27 * (-q / 2 + 1j * np.sqrt(max(0.0, -(q * q / 4 + p ** 3 / 27))))) / max(1.0, abs(L) ** 3))
    phi = rng.uniform(0, np.pi); t = 2 * rho * np.cos(phi); e["tripling"] = max(e["tripling"], abs(t ** 3 + p * t - 2 * rho ** 3 * np.cos(3 * phi))); e["doubling"] = max(e["doubling"], abs((2 * np.cos(phi)) ** 2 - 2 - 2 * np.cos(2 * phi)))
res["roots_and_tripling"] = dict(largest_errors={k: float(v) for k, v in e.items()}, cubics=2000)
# 4
err = dict(cubic=0.0, depressed=0.0, inverse_formula=0.0, other_trisector=0.0); n_tri = 0; signs = dict(R_positive=0, R_negative=0); roots_w = dict(three_real=0, one_real=0); which = dict(largest=0, middle=0, smallest=0, only=0)
while n_tri < 20000:
    a, b, c = rng.uniform(0.2, 3, 3)
    if a + b <= c * 1.001 or a + c <= b * 1.001 or b + c <= a * 1.001: continue
    n_tri += 1; C = np.arccos((a * a + b * b - c * c) / (2 * a * b)); th = C / 3; s = a * b * np.sin(3 * th) / (b * np.sin(th) + a * np.sin(2 * th)); r = a * b * np.sin(3 * th) / (b * np.sin(2 * th) + a * np.sin(th))
    R = a * a * c * c - (a * a - b * b) ** 2; S = 3 * a * b * b * c * c; T = -a * b * b * (a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c); sc = abs(R) * s ** 3 + S * s * s + abs(T); wv = 1 / s
    err["cubic"] = max(err["cubic"], abs(R * s ** 3 + S * s * s + T) / sc); err["depressed"] = max(err["depressed"], abs(T * wv ** 3 + S * wv + R) / (abs(T) * wv ** 3 + S * wv + abs(R)))
    err["inverse_formula"] = max(err["inverse_formula"], abs(wv - (b + 2 * a * np.cos(th)) / (a * b * (4 * np.cos(th) ** 2 - 1))) * s)
    R2 = b * b * c * c - (a * a - b * b) ** 2; S2 = 3 * a * a * b * c * c; T2 = -a * a * b * (a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c); err["other_trisector"] = max(err["other_trisector"], abs(R2 * r ** 3 + S2 * r * r + T2) / (abs(R2) * r ** 3 + S2 * r * r + abs(T2)))
    signs["R_positive" if R > 0 else "R_negative"] += 1; rw = np.roots([T, 0.0, S, R]); real = np.sort(rw[np.abs(rw.imag) < 1e-9 * (1 + np.abs(rw.real))].real)
    if len(real) == 3: roots_w["three_real"] += 1; which[("smallest", "middle", "largest")[int(np.argmin(np.abs(real - wv)))]] += 1
    else: roots_w["one_real"] += 1; which["only"] += 1
res["trisector"] = dict(triangles=n_tri, largest_relative_errors={k: float(v) for k, v in err.items()}, sign_of_R=signs, real_roots_of_the_cubic_in_1_over_s=roots_w, which_root_is_the_trisector=which)
# 5
worst = dict(eigenvalues=0.0, eigenvectors=0.0, others=0.0, tan2=0.0)
for _ in range(500):
    d = int(rng.integers(3, 40)); g, u = rng.standard_normal(d) * rng.uniform(0.3, 3), rng.standard_normal(d) * rng.uniform(0.3, 3); M = (np.outer(g, u) + np.outer(u, g)) / 2; ev, V = np.linalg.eigh(M); ng, nu = np.linalg.norm(g), np.linalg.norm(u); cs = g @ u / (ng * nu)
    worst["eigenvalues"] = max(worst["eigenvalues"], abs(ev[-1] - ng * nu * (cs + 1) / 2), abs(ev[0] - ng * nu * (cs - 1) / 2)); worst["others"] = max(worst["others"], np.abs(ev[1:-1]).max())
    bp, bm = g / ng + u / nu, g / ng - u / nu; worst["eigenvectors"] = max(worst["eigenvectors"], 1 - abs(V[:, -1] @ bp) / np.linalg.norm(bp), 1 - abs(V[:, 0] @ bm) / np.linalg.norm(bm)); worst["tan2"] = max(worst["tan2"], abs(abs(ev[0]) / ev[-1] - np.tan(np.arccos(cs) / 2) ** 2) / (1 + np.tan(np.arccos(cs) / 2) ** 2))
res["bisectors"] = dict(trials=500, largest_errors={k: float(v) for k, v in worst.items()})
# 6
worst = {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}; T = {2: lambda X: 2 * X * X - 1, 3: lambda X: 4 * X ** 3 - 3 * X, 4: lambda X: 8 * X ** 4 - 8 * X * X + 1, 5: lambda X: 16 * X ** 5 - 20 * X ** 3 + 5 * X}; n_tri = 0
while n_tri < 5000:
    a, b, c = rng.uniform(0.2, 3, 3)
    if a + b <= c * 1.001 or a + c <= b * 1.001 or b + c <= a * 1.001: continue
    n_tri += 1; A = np.arccos((b * b + c * c - a * a) / (2 * b * c)); C = np.arccos((a * a + b * b - c * c) / (2 * a * b)); h = a * b * np.sin(C) / c
    for n in worst:
        th = C / n; sn = a * b * np.sin(C) / (b * np.sin(th) + a * np.sin(C - th)); worst[n] = max(worst[n], abs(T[n](h / sn) - np.cos(n * A + C - n * np.pi / 2)))
res["chebyshev"] = dict(triangles=n_tri, largest_errors={str(n): float(v) for n, v in worst.items()})
# 7
psi = np.linspace(0, 2 * np.pi, 2000, endpoint=False); e2 = e3 = 0.0
for _ in range(500):
    p = rng.uniform(0, np.pi, 3); f2 = np.cos(psi - p[0]) * np.cos(psi - p[1]); e2 = max(e2, np.abs(f2 - (np.cos(2 * psi - p[0] - p[1]) + np.cos(p[0] - p[1])) / 2).max())
    f3 = f2 * np.cos(psi - p[2]); e3 = max(e3, np.abs(f3 - (np.cos(3 * psi - p.sum()) + np.cos(psi + p[0] - p[1] - p[2]) + np.cos(psi - p[0] + p[1] - p[2]) + np.cos(psi - p[0] - p[1] + p[2])) / 4).max())
res["circle"] = dict(trials=500, largest_error_two_forms=float(e2), largest_error_three_forms=float(e3))
# 8
slope = lambda M: (lambda w, v: float(v[1, -1] / v[0, -1]))(*np.linalg.eigh(np.array(M))); m2, m4, m6 = 3.0, 81 / 5, 729 / 7                        # moments of the uniform law on [-3, 3]
res["factors"] = dict(degree_2=dict(gradient_main_slope=slope([[4 * m2 + m2, 2 * m2], [2 * m2, m2]]), bisector_of_x_and_x_plus_a=float(np.tan(np.pi / 8)), completing_the_square=0.5),
                      degree_3_leading_terms=dict(gradient_main_slope=slope([[9 * m4 + 4 * m2 * m2, 3 * m4], [3 * m4, m4]]), mean_direction_of_the_three_factors=float(np.tan(np.pi / 12)), cardano=1 / 3),
                      degree_3_of_the_post=dict(gradient_main_slope=slope([[9 * m4 + 4 * m2 * m2 + 6 * m2 + 1, 3 * m4 + m2], [3 * m4 + m2, m4]])),
                      degree_4_leading_terms=dict(gradient_main_slope=slope([[16 * m6 + 9 * m2 * m4, 4 * m6], [4 * m6, m6]]), mean_direction_of_the_four_factors=float(np.tan(np.pi / 16)), quarter=0.25))
# 9
Cg = np.linspace(0, np.pi, 20001); f = np.cos(Cg / 3); c = np.cos(Cg); v = np.cos(Cg / 2)
res["half_angle"] = {str(d): dict(max_error_in_cos_C=float(np.abs(f - np.polynomial.chebyshev.Chebyshev.fit(c, f, d)(c)).max()), max_error_in_cos_half_C=float(np.abs(f - np.polynomial.chebyshev.Chebyshev.fit(v, f, d, domain=[0, 1])(v)).max())) for d in (2, 4, 8)}
json.dump(res, open(sys.argv[1], "w"), indent=1)
for k, v in res.items(): print(k, json.dumps(v))
