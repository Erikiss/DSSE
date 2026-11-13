import numpy as np, importlib.util, sys
spec = importlib.util.spec_from_file_location('m', sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('lens fine grid (sust power, diss power)')
for d1 in (0.5, 1.0):
    for K in (0.7, 0.85, 0.9, 1.1, 1.15, 1.2, 1.3, 1.35, 1.4):
        r = m.lens(d1, K, nrep=150)
        print(f'  d1={d1} K={K}: SD={r[0]:.3f} sust={r[2]:.2f} diss={r[3]:.2f} plat={r[4]:.2f}', flush=True)
print('sessions fine grid')
for z2 in (5, 10):
    for K in (0.85, 0.9, 0.95, 1.05, 1.1, 1.15):
        r = m.sessions(z2, K, nrep=150)
        print(f'  z2={z2} K={K}: SD={r[0]:.3f} sust={r[2]:.2f} diss={r[3]:.2f} plat={r[4]:.2f}', flush=True)
