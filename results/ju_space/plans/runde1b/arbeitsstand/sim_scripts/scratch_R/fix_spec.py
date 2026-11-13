import sys
p = sys.argv[1]; s = open(p).read()
rep = [
# summary budget
("- **Budget.** 8.1 A100-h unconditional + 1.9 A100-h conditional = 10.0 A100-h at the round-2 planned rates (P2 7.1), within the\n  12-h envelope. At half the planned generation speed the unconditional part is about 11.4 h; the pre-registered packing reductions\n  of R.5 then keep the total at or below 12 h.",
 "- **Budget.** 8.1 A100-h unconditional + 1.9 A100-h conditional = 10.0 A100-h at the round-2 planned rates (P2 7.1), within the\n  12-h envelope. At half the planned generation speed the unconditional part is about 11.6 h; the pre-registered packing reductions\n  of R.5 then keep the total near 12 h even with both conditional sessions."),
# PC_PASS_R0 scale note
("    PC_PASS_R0(b, V, X): SEA rows n = 2, 3, 4 meet V and rel_n >= 0.5, and S_4^cf >= 0.5 (S_4 uncorrected for V = R1, which must\n    reproduce round 1's PC_PASS).",
 "    PC_PASS_R0(b, V, X): SEA rows n = 2, 3, 4 meet V and rel_n >= 0.5, and S_4^cf >= 0.5 (S_4 uncorrected for V = R1, which must\n    reproduce round 1's PC_PASS); at block 14 the four-round loops exist only at 1.5 alpha, so PC_PASS_R0(14) is at 1.5 alpha while\n    GO1_R0 'at alpha' concerns a scale at which no loop ran (int_gates)."),
# RND starts
("SEA_raw (start unit(u_sea^1b(b)), 256 pairs; round-1 PC start), RND_1, RND_2 (starts\n    Pool ('rand', b, 2) and ('rand', b, 3), each orthogonalised against u*_b, u_T^perp*, u_sea^perp*, u_sports^perp* and\n    renormalised; 256 pairs).",
 "SEA_raw (start unit(u_sea^1b(b)), 256 pairs; round-1 PC start), RND_1, RND_2 (starts:\n    isotropic Gaussian unit vectors from torch.Generator seeds 4101 + b and 4201 + b, each orthogonalised against u*_b, u_T^perp*,\n    u_sea^perp*, u_sports^perp* and renormalised, so no start lies on the style mode or a test axis; 256 pairs)."),
# state guard note in R2.2 How
("  - Keys ('r1b_' + L, n, q, s, b), shared across arms (paired arms). Every evaluation stores",
 "  - Keys ('r1b_' + L, n, q, s, b), shared across arms (paired arms). Outputs go to their own files under <work>/r1b/, never into\n    the copied round-1 state (r2_rk's state guard admits only 'r2'-prefixed keys there). Every evaluation stores"),
# R2.3 MDE caveat
("  rule 'p_s <= 0.01 and Q_c >= 1' would need c of about 0.26-0.27 (A about 0.29-0.30) at any N. Lexical MDE",
 "  rule 'p_s <= 0.01 and Q_c >= 1' would need c of about 0.26-0.27 (A about 0.29-0.30) at any N. These MDEs use round 1's SEA noise;\n  how much sea signal survives on the style-free axis is unknown until R0.6 reports it (post hoc, after the freeze). Lexical MDE"),
# R2.1 MDE
("0.017 / 0.014 / 0.017 at 256 prompts", "0.017 / 0.013 / 0.017 at 256 prompts"),
# D4 with perp* foreign axes
("- How: DN arm RSI_a and RSI_b read on u_T^perp*(b) (primary) and on u_T_perp^1b (round-1 axis, reported); foreign readouts on raw\n  sports, tone, sea and u_neg (D4 = |A_4| on u_T^perp* minus max |A_4| on the foreign axes);",
 "- How: DN arm RSI_a and RSI_b read on u_T^perp*(b) (primary) and on u_T_perp^1b (round-1 axis, reported); foreign readouts on\n  u_sea^perp*, u_sports^perp*, u_tone^perp* and u_neg (D4 = |A_4| on u_T^perp* minus max |A_4| on these; style is handled by the\n  'style-attracted' label, not by D4);"),
# likely: remove S_4 for RSI loops
("    mean rel_2..4 >= 0.5; E's lower bound >= 0.5; R_4^cf >= 0.5; S_4^cf >= 0.5 for both loops; D4 > 0 for both;",
 "    mean rel_2..4 >= 0.5; E's lower bound >= 0.5; R_4^cf >= 0.5; D4 > 0 for both (S_4^cf of the RSI loops reported, not required,\n    as in round 1);"),
# oscillation definition
("  - 'oscillation (RSI begets anti-RSI)': as 'likely' but two-sided p_s <= 0.01, signs of A_1..A_4 alternating, A_bar_abs's lower bound\n    >= 0.25 (E not required).",
 "  - 'oscillation (RSI begets anti-RSI)': the common conditions of 'likely' (PC, MANIP, mean rel, E, R_4^cf, D4, SPO_t, RND,\n    damage, K5) with two-sided p_s <= 0.01 at n = 2, 3, 4 for both loops, signs of A_1..A_4 alternating, and A_bar_abs's lower bound\n    >= 0.25 (round 1's logic)."),
# E MDE in R2.4
("MANIP from A_1 about 0.22-0.24 / 0.19 / 0.20.", "MANIP from A_1 about 0.22-0.24 / 0.19 / 0.20; E's SD about 0.05\n  (est.), so its lower bound >= 0.5 needs a true E of about 0.62."),
# reaches 1 wording + u* attenuation
("  - 'reaches 1 at small push' if some cell with a <= 0.5 alpha_0 has a lower bound > 1 at the 1 - 0.10 / 3 level (best of three\n    cells);",
 "  - 'reaches 1 at small push' if some cell with a <= 0.5 alpha_0 has its 0.05 / 3 bootstrap quantile > 1 (best of three cells,\n    Bonferroni);"),
("g_par(u) = unit(u) . d / (a cbar) (cross-fitted fold axes for the estimated\n  axes, plain projection for u*_b and the random direction);",
 "g_par(u) = unit(u) . d / (a cbar) (cross-fitted fold axes for the estimated\n  axes, plain projection for the random direction; along u*_b with round 2's attenuation m*_b from r2_q2/theta/b<b>_q0.json\n  'm_star2' (0.982 at 14), exactly B-Q1's formula, so the cells compare with round 2's);"),
("'Below 1' holds with power 0.8 when the true gain is <= about 0.91 at 0.25 alpha_0;\n  'reaches 1' needs about >= 1.10.",
 "'Below 1' holds with power 0.8 when the true gain is <= about 0.91 at 0.25 alpha_0;\n  'reaches 1' needs about >= 1.11."),
# Bonferroni sentence in R.1
("'Reaches 1' claims that take the best of k cells use the 1 - 0.10 / k bound.",
 "'Reaches 1' claims that take the best of k cells use the 0.05 / k quantile."),
# R4.1 MDE per direction
("- MDE (est.): SD of lambda_lin along a fixed direction 0.38 at 128 texts (B-Q7) -> about 0.07 at 3,840 texts (block 14), 0.095 at\n  2,048 (18, 6).",
 "- MDE (est.): B-Q7's SDs at 128 texts (0.38 along u*_14, 0.14 along u_T_perp, 0.23 along sea) scale to about 0.069 / 0.026 / 0.041\n  at 3,840 texts (block 14) and 0.095 / 0.035 / 0.057 at 2,048 (blocks 18, 6)."),
# R4.2 convergence fallback
("in-sample value reported. cos(v, w) (non-normality)",
 "in-sample value reported. If the iteration has not converged after 50 steps (relative change > 1e-3: a complex or degenerate\n    top pair), the top eigenvalue is read from the 8-dimensional Krylov subspace (Arnoldi with forward products only), labelled\n    'not converged'. cos(v, w) (non-normality)"),
("- MDE (est.): lambda_RSI SD about 0.07 (block 14) / 0.095 (18, 6); lambda_max^cf SD about 0.095 / 0.13; a difference of about 0.29\n  (block 14) to 0.4 (18, 6) is detectable; 'near critical' needs a true lambda_RSI of about 0.97 (14).",
 "- MDE (est.): lambda_RSI SD between 0.026 (if V_RSI is as quiet as u_T_perp) and 0.069 (as noisy as u*) at block 14, 0.035-0.095\n  at 18 and 6; lambda_max^cf SD about 0.098 / 0.134 (half samples along u*); a difference of about 0.26-0.30 (block 14) to 0.36-0.41\n  (18, 6) is detectable; 'near critical' needs a true lambda_RSI of about 0.86-0.97 (block 14)."),
# R5.1 three doses
("over the 24 sets x 4 directions x 2 doses, 90 % text-bootstrap", "over the 24 sets x 4 directions x 3 doses, 90 % text-bootstrap"),
("cap_b and med_b at every block from 64\n  TXT0 texts (2 x median norm).",
 "cap_b and med_b from R1.state[b]['ref'] at 14, 18,\n  6 and, at the other blocks, 2 x the median norm over the continuation tokens of 64 TXT0 texts."),
# R5.2 objects
("- MDE: z_lens is analytic given the null spread; the finite-difference SD over 60 objects is measured (no prior estimate).",
 "- MDE: z_lens is analytic given the null spread; the finite-difference SD over A3's objects is measured (no prior estimate)."),
# R5.3 Gram-Schmidt
("u_T^J (u_T minus its M_J-projection on span(sports, tone, sea,\n  u_neg), M_J = R_J' R_J, R_J = W_U[letter/digit tokens] diag(g) J_14, centred over the vocabulary)",
 "u_T^J (u_T made M_J-orthogonal to span(sports, tone, sea,\n  u_neg) by Gram-Schmidt in the M_J inner product, no matrix inverse; M_J = R_J' R_J, R_J = W_U[letter/digit tokens] diag(g) J_14,\n  centred over the vocabulary)"),
# R5.5 lens ranking
("and by the cosine of their lens readouts with u*_b's lens readout (only at\n  blocks where R5.1 says the lens is valid).",
 "and by the cosine between the lens readout of that quick axis and u*_b's lens\n  readout (only at blocks where R5.1 says the lens is valid)."),
# R7.1 foreign axes
("D under the same variant. Gaps on O1-style fixed axes from the unablated 1b instructed texts.",
 "D under the same variant (each control compared without itself, P2 O6). Gaps on O1-style fixed axes from the unablated 1b\n  instructed texts; a foreign axis X moves by m_X = |1 - g_abl^X / g_0^X| on its own fixed axes (P2 OD-C6; TOPIC is dropped, part R\n  has no RGI axis)."),
# R7.2 MDE
("restored-share SD about 0.11 at 256 prompts\n  (round 2: [0.691, 1.122] at 64).", "restored-share SD about 0.066 at 256 prompts\n  (round 2: [0.691, 1.122] at 64, SD 0.13)."),
# budget text in R.5
("Budget argument: at the planned rates part R needs 8.1 A100-h plus 1.9 h for the two conditional sessions, within the 12-h envelope;\nabout 7,700 generate calls carry most of the cost. At half the planned generation speed the unconditional part rises to about 11.4\nh. If R-P measures more than 1.6 s per generate call, the following pre-registered reductions apply (packing, not amendments: they\nremove descriptive arms only and change no rule): RSI rounds 5-6 dropped; FX arm at blocks 18 and 6 reduced to RSI_a; R3.3 at blocks\n18 and 6 reduced to doses {0.25, 0.5, 1} x alpha_0 plus PUSH-DN; R6.1's iterated latent loops only at block 14. They save about 1.5-2\nh at the slower rate.",
 "Budget argument: at the planned rates part R needs 8.1 A100-h plus 1.9 h for the two conditional sessions, within the 12-h envelope;\nabout 7,700 generate calls carry most of the cost. At half the planned generation speed (cost_R.py with every generation time\ndoubled) the unconditional part rises to 11.6 h and the total to 13.6 h. If R-P measures more than 1.6 s per generate call, the\nfollowing pre-registered reductions apply (packing, not amendments: they remove descriptive arms only and change no rule): RSI rounds\n5-6 dropped; FX arm at blocks 18 and 6 reduced to RSI_a; R3.3 at blocks 18 and 6 reduced to doses {0.25, 0.5, 1} x alpha_0 plus\nPUSH-DN; R6.1's iterated latent loops only at block 14; R5.4 (if triggered) on 200 instead of 300 texts. They save about 1.5 h at the\nslower rate, which brings the total back to about 12 h."),
]
for a, b in rep:
    if a not in s: print("MISSING:", a[:90].replace("\n", "|")); continue
    s = s.replace(a, b, 1)
open(p, "w").write(s); print("done", len(s))
