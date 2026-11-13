# Final session packing for part L (work minutes from costs_L.py plus the additions decided in design)
SETUP = 4.0
S = {
 'L-S1': {'L0': 5.5, 'L.R': 28.2, 'L1': 17.0, 'L2': 8.0, 'L4': 12.2, 'L7(chain)': 2.0},
 'L-S2': {'L3': 30.1, 'L9': 15.2, 'L1 L.gv (H-basin)': 2.5},
 'L-S3': {'L5': 22.1, 'L5 L.gv (communities)': 3.0, 'L6': 6.5, 'L10': 9.4, 'L7(long)': 2.0},
 'L-S4': {'L8': 35.8, 'L8 frozen-frame re-reads': 3.9},
 'L-S5': {'L11': 2.0, 'combined verdicts + report': 3.0},
}
tw = tm = 0
for s, d in S.items():
    w = sum(d.values()); tw += w + SETUP; tm += SETUP + 1.5 * w
    print(f"{s}: work {w:5.1f} min | setup+work {w+SETUP:5.1f} | setup+1.5*work {SETUP+1.5*w:5.1f}  {d}")
print(f"TOTAL setup+work {tw:.1f} min = {tw/60:.2f} h ; with margin {tm:.1f} min = {tm/60:.2f} h")
meas = {}
for s, d in S.items():
    for k, v in d.items():
        key = k.split(' ')[0].split('(')[0]
        meas.setdefault(key, 0); meas[key] += v
print({k: round(v, 1) for k, v in meas.items()}, 'sum', round(sum(meas.values()), 1))
