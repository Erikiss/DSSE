import sys, textwrap
p = sys.argv[1]; s = open(p).read()
rep = [
("u_T^perp* (b) = u_T minus its projection on span(u*_b, u_sports, u_tone, u_sea, u_neg); u_sea^perp* and\nu_sports^perp* likewise (each axis orthogonalised against u*_b, u_neg and the other three text axes);",
 "u_T^perp* (b) = u_T minus its projection on span(u*_b, u_sports, u_tone, u_sea, u_neg); u_sea^perp*,\nu_sports^perp* and u_tone^perp* likewise (each axis orthogonalised against u*_b, u_neg and the other three text axes);"),
("  cos(d_n, axis) is reported next to it.",
 "  cos(d_n, axis) is reported next to it. A_n is slightly attenuated by the fold axis's own reliability (about 0.97 at 128 prompts\n  per fold, est.), which makes every lower-bound rule conservative."),
("u_T, u_T_perp^1b, u_T^perp*, u_sea^perp*, u_sports^perp*, raw u_sea / u_sports / u_tone,",
 "u_T, u_T_perp^1b, u_T^perp*, u_sea^perp*, u_sports^perp*, u_tone^perp*, raw u_sea / u_sports / u_tone,"),
("the 64 round-1 prompts reuse their round-1 texts (R1.state[14]['axis_*']['conts'], re-read at every block);",
 "the 64 round-1 prompts reuse their round-1 texts (at block b those of R1.state[b]['axis_*']['conts'], re-read);"),
("u_T_perp^1b; u_T^perp*, u_sea^perp*,\n    u_sports^perp* (R.1);", "u_T_perp^1b; u_T^perp*, u_sea^perp*,\n    u_sports^perp*, u_tone^perp* (R.1);"),
("Re-read check: the re-read R1 texts reproduce R1 Xp / Xa to 1e-4 relative at\n    blocks whose round-1 GPU was an A100 (flag otherwise).",
 "Re-read check: the re-read R1 texts reproduce R1 Xp / Xa to 1e-4 relative (flag\n    'R1 axis reads differ' otherwise; the 1b reads are used either way)."),
("- Gates: RG1-RG4; CC1-CC3 (the replay's own GO2 code checks).", "- Gates: RG1-RG4; CC2, CC3 (the replay's own GO2 checks: mirror and early-stopped read)."),
("CC4 r1b_core's push equals rk's push\n  at block 14 to 1e-6 relative.", "CC4 (in R-S2/14) r1b_core's push hook,\n  registered on rk.model, equals rk's push at block 14 to 1e-6 relative."),
("Directions: at 14, 18, 6 the R3.3 directions; at the other\n  blocks",
 "Directions: at 14, 18, 6 u*_b, unit(u_T^perp*(b)), unit(u_sea^perp*(b)) and\n  ('cov', b, 4); at the other blocks"),
("teacher-forced on 64 TXT0 texts (seed 0 of 32 + 32 prompts)", "teacher-forced on 64 TXT0 texts (seed 0 of the 64 read-direction prompts of R4.2)"),
("Verifier error rate:\n  the same procedure on known targets (u_sea and u_tone at block b, held-out prompts) must return the sea and tone pairs.",
 "Verifier error rate:\n  the same procedure run on two known targets (u_sea and u_tone at block b, held-out prompts) must return the sea and tone pairs."),
("on known targets are reported; a label with a verifier error rate above 1 of 2 known targets is marked 'unreliable verifier'.",
 "on known targets are reported; if the procedure fails on both known targets, every label it returns is marked 'unreliable\n  verifier'."),
("Per block b in {14, 18, 6} and axis u in {u*_b,\n  unit(u_T^perp*), unit(u_sea^perp*), ('cov', b, 4)}:",
 "Per block and axis u: block 14 the four R3.3 directions (u*_14,\n  unit(u_T^perp*), unit(u_sea^perp*), ('cov', 14, 4)); blocks 18 and 6 u*_b and unit(u_T^perp*) (the R3.3 directions there):"),
("'latent loop reaches 1 on RSI' if g_soft(u_T^perp*)'s lower bound > 1.",
 "'latent loop reaches 1 on RSI' if g_soft(u_T^perp*)'s lower bound > 1; 'latent loop below 1 on RSI' if its upper bound < 1."),
("kappa_k = D_k / D_(k-1); the same along u*_14. Round-trip", "kappa_k = D_k / D_(k-1); the same along u*_14 (descriptive). Round-trip"),
("- Null / controls: the paired construction (the - side); the u*_14 carry for comparison with round 2's K 0.45-0.48.",
 "- Null / controls: the paired construction (the - side); the u*_14 carry (round 2's K 0.45-0.48 was along u*_14 after a u*_14 push;\n  here the push is along RSI, so the two are not the same quantity)."),
("More A100 time would not buy much: SDs are limited by the number of prompts (prompt-level noise share 0.54), and\nwriting more than 192 new prompts is the binding cost.",
 "More A100 time would not buy much: about 46 % of the per-pair noise lies between prompts (54 % within, int_gates 5), so beyond two\nseeds only new prompts lower the SDs, and writing more than 192 new prompts is the binding cost."),
("- OD-R1 Prompt count: 192 new prompts (256 total). More prompts are the only way to lower prompt-level noise; seeds add little.",
 "- OD-R1 Prompt count: 192 new prompts (256 total). Beyond two seeds per prompt only more prompts lower the noise (int_gates 5)."),
]
for a, b in rep:
    if a not in s: print("MISSING:", a[:100].replace("\n", "|")); continue
    s = s.replace(a, b, 1)
# rewrap overlong non-table lines
out = []
for line in s.split("\n"):
    if len(line) > 136 and not line.lstrip().startswith("|"):
        lead = len(line) - len(line.lstrip())
        body = line.lstrip()
        if body.startswith("- "): sub = " " * (lead + 2)
        else: sub = " " * lead
        out.extend(textwrap.wrap(line, width=134, subsequent_indent=sub, break_long_words=False, break_on_hyphens=False))
    else: out.append(line)
s = "\n".join(out)
open(p, "w").write(s); print("done", len(s), "max line", max(len(l) for l in s.split("\n") if not l.lstrip().startswith("|")))
