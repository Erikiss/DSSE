# Correction for part K: the user's term 'Gradient Through Routing' (user, 13.11.2026)

The user confirms: 'Haarneuronen' = H-neurons (Gao et al. 2025). 'Gradient Through Routing' comes from the multi-token prediction
paper 'Multi-Token Prediction Done Right' (Jayden Teoh, MIT; arXiv 2610.38331, submitted 28 Oct 2026; code
github.com/JaydenTeoh/Soft-MTP). It is NOT Cloud et al.'s gradient routing and NOT an inference-time quantity.

What the paper says (read from arXiv HTML):
- The MTP cross-entropy gradient at the k-th MTP hidden state decomposes exactly: g_hard = g_KL + g_res.
  g_KL = W^T (p_t^(k) - q_t^(k)) is the gradient of D_KL(sg(q_t^(k)) || p_t^(k)): it pulls the k-th MTP head toward the model's own
  next-token (NTP) distribution q at the aligned position (self-distillation). g_res = W^T (q_t^(k) - onehot(x_{t+k+1})) depends on
  the sampled data token, has a low signal-to-noise ratio and 'buries' the useful KL signal, increasingly with scale.
- KL gradient-recovery trick ('straight-through gradient routing'): h0[:, k:] += hk.detach() - hk (numerically unchanged); the NTP
  loss's backward then injects -g_res at the MTP hidden state, so MTP heads and trunk receive only g_KL while the shared output head
  keeps its CE gradient (Table 1: Standard MTP CE/CE; KL-MTP KL/KL; Soft-MTP CE for the output head, KL for MTP heads and trunk).
- Soft-MTP: one-line change; 167M-1.2B models; lower NTP perplexity, faster speculative decoding, better zero-shot accuracy.

Consequences for part K (to apply after its review):
1. Fix the mapping row for 'lebt von einem Gradient, von den Gradient Through Routing ... dieses Durchrouten': the user's term is
   Teoh's straight-through gradient routing (training time). Its content: MTP lives on the self-distillation signal, i.e. on the
   model's OWN next-token predictions along the chain ('Folge': each future step's target is the model's own prediction), while the
   data residual is noise. In the frozen model there is no training; measurable counterparts are (a) and (b) below. The
   inference-time cross-position Jacobian (K10b) stays as our forward analogue of 'Durchrouten', labelled as our analogue.
2. (a) Self-consistency of a chain-lens step: KL(q || p) between the model's own NTP distribution at the aligned position
   (teacher-forced along the chain) and the lens's distribution for that step, per step k. On Qwen3.5-2B's native MTP head (K5a) and
   on Qwen3-1.7B's chain (CL17 / RK-1). This decomposes break-off ownership: confabulation growing with chain length while the
   self-consistency KL stays flat = model-owned; growing KL = head- / lens-owned (the head's residual-trained prior). Link to K6 and
   to L5 OWNER.
3. (b) RK-1 (post-hoc MTP head on frozen Qwen3-1.7B, tier 2, now FUNDED because the user chose tier 1 + tier 2): train two heads on
   the same data and steps: standard CE MTP and Soft-MTP (straight-through gradient routing; with a frozen trunk only the head
   receives g_KL). Prediction: the Soft-MTP head inherits the model's own prior, so its chain-lens length curve should follow the
   self-rollout's; a CE head adds its own data-prior confabulation. Make RK-1 unconditional within tier 2 (the G-K5 condition no
   longer gates it), keep its budget near 2 x the old 58 min or cut tokens to stay near 1.2 h, with rules (descriptive or with
   verdict strings) and the session.
4. Cite Teoh 2026 (arXiv 2610.38331) in part K and in the main file section 1 row for H-CHAIN.

Budget decision of the user (13.11.2026): fund tier 1 AND tier 2 ('beide'). Part P (Pythia bridge) is treated as tier 2 as well.
