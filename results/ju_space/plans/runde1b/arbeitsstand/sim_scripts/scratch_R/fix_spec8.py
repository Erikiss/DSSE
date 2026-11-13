import sys
p = sys.argv[1]; s = open(p).read()
i, j = s.index("## R.6 Outcomes"), s.index("## R.8 Open decisions")
new = """## R.6 Outcomes that count against (and for) the hypothesis

Against "RSI content carries itself through the model's own text" (each read with its MDE):
- At a block with PC_PASS_1b: 'unlikely (RSI does not persist)' (A_bar_abs's upper bound < 0.25) or 'unlikely (no first link)'.
- R3.3 along u_T^perp*: 'below 1 at small push' at every block (and along u*_b: the generic mode is sub-critical too, closing round 2's open window).
- R4.2: 'lambda_RSI below 1' and 'RSI subdominant'; 'RSI above neutral words' failing.
- R6.2: K_2 and K_geo 'dissipative' at T 0.7 and T 1.0; R6.3 'not replicated'.
- R6.1: 'latent loop below 1 on RSI' (the discrete text channel is not what keeps RSI from carrying itself).
- R7: 'unlearned' and C-Q4b 'un-searched'.

For it:
- 'likely' at a PC-passing block; 'reaches 1 at small push' along u_T^perp*; 'lambda_RSI >= 1', or 'near critical' together with 'RSI is the top mode'; K 'self-sustaining'; C-Q4b 'survives' or 're-routed'.

Against round 1b's own repair claims:
- R0 'README null claim not confirmed' (centring changes no p); RG6 failing (the paired test is not valid at 64 prompts); R2.3's PC failing at all three blocks with GO0_1b met (the loop cannot keep even a concrete topic beyond the style mode; then no loop verdict about RSI is possible at all, which is itself the result).

## R.7 Changes from round 1

- (1) Centred pseudo-axes replace round 1's uncentred sign-flip nulls (reported only); significance from a cross-fitted paired sign-flip test over prompts (K1-s), with Wilcoxon and t beside it and a size check deciding a bootstrap-t fallback.
- (2) Magnitude by a margin in interpretable units (90 % lower bound of A >= 0.10 / 0.25 / 0.50) instead of the N-independent cosine bar (Q_c >= 1, about 0.22-0.26 in c); MDE pre-registered for the rule actually used.
- (3) Axis uncertainty propagated by a joint prompt bootstrap of axis texts and loop pairs; readouts on cross-fitted fold axes (no self-trace).
- (4) S_4 and R_4 cross-fitted over halvings (round 2 found disattenuated cosines up to 1.219 and 1.468).
- (5) Positive control redefined: style-free topic axis u_sea^perp*, lexical sea-word criterion, SEA - max(RND) contrast, SEA start on the style-free axis; sports as pre-registered fallback; round-1 PC reported beside it.
- (6) RSI read on u_T^perp* (style mode removed), so convergence onto the generic attractor cannot count as RSI.
- (7) Random-start controls start orthogonal to the style mode and the test axes.
- (8) Damage-normalised push per direction and round (0.05 nats, both signs logged, 8 RSI + 8 scene references) replaces the fixed alpha; GO3 becomes 'every push in band'; a norm-preserving control push added.
- (9) 256 prompts (192 new, frozen before the run) instead of 64; RSI, SEA_t and SPO_t loops 512 pairs, others 256; RSI_b at every block.
- (10) A round-1-identical replay (byte-identical rk_loop.py, round-1 keys) anchors an attribution ladder.
- (11) New measurements: dose series below 0.5 alpha_0 in both signs with unpushed texts; no-push noise covariance; lambda_RSI vs lambda_max with read and use directions; lens bridge with a layer profile; latent loop; multi-turn carry on the RSI axis; low-damage erasure; 0.6B diagnosis; verified label of the style mode.
- (12) Report hygiene: provenance, codes, labels, distributions in every output.

"""
s = s[:i] + new + s[j:]
open(p, "w").write(s); print("ok")
