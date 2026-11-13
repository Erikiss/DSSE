import sys
p = sys.argv[1]; s = open(p).read()
rep = [
("- Shows - cannot show: whether round 2's attractor survives damage normalisation; not why (R3.4, R4).",
 "- Null / controls: the RND loops (starts orthogonal to u*_b); the NP arm (R3.2).\n- Shows - cannot show: whether round 2's attractor survives damage normalisation; not why (R3.4, R4)."),
("- Shows - cannot show: whether in-distribution fitting repairs the lens; a bf16 lens is used only for readouts, never for compared",
 "- Null / controls: as R5.1.\n- Shows - cannot show: whether in-distribution fitting repairs the lens; a bf16 lens is used only for readouts, never for compared"),
("- Shows - cannot show: whether the single round-2 cell that allowed self-sustainment on the RSI axis was a small-denominator artefact.",
 "- Null / controls: the paired construction; K along u*_14 in the same cell (round 2: 0.45 [0.37, 0.53] at T 1.0) as the internal\n  comparison.\n- Shows - cannot show: whether the single round-2 cell that allowed self-sustainment on the RSI axis was a small-denominator artefact."),
("- Shows - cannot show: how deep the erasure is against an in-context cue.",
 "- Null / controls: the unablated cued texts (the cue's ceiling); C-Q3-style unablated instructed texts.\n- Shows - cannot show: how deep the erasure is against an in-context cue."),
("- Shows - cannot show: re-search through the loop; three rounds only.",
 "- Null / controls: the unablated twin with shared keys and the same pushes.\n- Shows - cannot show: re-search through the loop; three rounds only."),
("- Shows - cannot show: the cause within round 2's frozen gate; it changes no round-2 verdict and says nothing about 0.6B loops.",
 "- Null / controls: the same keys of the 1.7B file (C0 true at all 28 blocks, uncomposed at 0-11, FG(CHG-DES) 0.706-0.787 at the\n  passing blocks).\n- Shows - cannot show: the cause within round 2's frozen gate; it changes no round-2 verdict and says nothing about 0.6B loops."),
("- Shows - cannot show: whether the 1.7B picture is size-specific between two models of one family; nothing about scaling.",
 "- Null / controls: the 0.6B RND loops; the 1.7B block-14 loops as the reference.\n- Shows - cannot show: whether the 1.7B picture is size-specific between two models of one family; nothing about scaling."),
("- Shows - cannot show: traceability; not correctness. MDE: n/a.",
 "- Null / controls: n/a.\n- Shows - cannot show: traceability; not correctness. MDE: n/a."),
]
for a, b in rep:
    if a not in s: print("MISSING:", a[:100]); continue
    s = s.replace(a, b, 1)
open(p, "w").write(s); print("ok")
