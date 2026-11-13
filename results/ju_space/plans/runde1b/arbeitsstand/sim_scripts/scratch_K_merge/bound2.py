"""'No within-item bistability' bound at z > 2 (numpy only): 90 % item-bootstrap upper bound of the share of units with z > 2 < 0.10."""
import numpy as np, importlib.util, sys
spec = importlib.util.spec_from_file_location('b', sys.argv[1]); b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
rng = np.random.default_rng(29)
def bound(z, thr=2.0):
    NI = z.shape[0]; sh = (z > thr).mean(1); idx = rng.integers(0, NI, (1000, NI))
    return np.percentile(sh[idx].mean(1), 95) < 0.10
R = 200
for NI, NP, lab in ((192, 8, 'K6e 192x8'), (24, 16, 'K8b 24x16')):
    print(lab)
    for delta, pi in ((3.0, 0.10), (3.0, 0.15), (3.0, 0.3), (4.0, 0.10), (4.0, 0.15)):
        print(f'  bistable Delta={delta} pi={pi}: P(no-bist bound) = {np.mean([bound(b.bistable(NI, NP, pi, delta)) for _ in range(R)]):.2f}')
    for delta, skew, het in ((0.0, 0.0, 0.5), (0.0, 1.0, 1.5), (0.5, 0.0, 0.5), (0.5, 1.0, 1.5), (1.0, 0.0, 0.5), (1.5, 0.0, 0.5)):
        print(f'  mono shift={delta} skew={skew} het={het}: P(no-bist bound) = {np.mean([bound(b.mono(NI, NP, delta, skew, het)) for _ in range(R)]):.2f}')
