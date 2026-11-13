"""Data volume R-Sana must copy in (part R spec, R2.2 storage list), float32 vs float64, all three read windows vs 'all' only."""
D = 2048
def gb(x): return round(x / 1e9, 2)
pairs_round = (512 * 4 + 256 * 3) + (512 + 256 + 256) + (512 + 256 + 256)          # DN + FX + NP per round
pairs_block = pairs_round * 4 + 2 * 1024                                              # + RSI_a/RSI_b DN rounds 5-6
dose_texts = {14: 24 * 1024 + 512, 18: 12 * 1024 + 512, 6: 12 * 1024 + 512}
txt0 = 4096 * 3                                                                        # reads at 14/18/6
fdt = (3840 + 2 * 2048) * 2                                                            # X and S
ax28 = 2048 * 28                                                                       # all-block axis read
r43 = 3 * 2048; r6 = (4096 + 12288) + (10240 + 1024) * 2 + 2048; r7 = 10 * 512 + 3300
for bytes_, lab in ((4, "float32"), (8, "float64")):
    for w, wl in ((3, "3 windows"), (1, "'all' only")):
        r2 = 3 * pairs_block * 2 * w * D * bytes_                                     # xp, xm per window (delta derivable)
        r33 = sum(dose_texts.values()) * w * D * bytes_
        rest = (txt0 * w + fdt + ax28 + r43 * w + r6 + r7) * D * bytes_
        print(lab, wl, "R-S2 pairs", gb(r2), "R3.3", gb(r33), "rest", gb(rest), "total GB", gb(r2 + r33 + rest),
              "copy-in min at 10 min/GB", round(10 * (r2 + r33 + rest) / 1e9))
print("pairs per block", pairs_block)
