# Design rules for round 1b part specs (all designers, judges and mergers)

## Format (as in the frozen round-2 plan, results/ju_space/plans/abschnitt11_12_runde2_plan.md; read its sections 0, 2.6, 4.1
## B-Q8 and 6 as a model of the level of detail)
Every measurement gets an ID and these fields:
- What: the question it answers, in one or two sentences.
- How: stimuli (exact counts, how built, which prompts/texts, keys/seeds), model settings (block, push, temperature, decoding,
  context length, dtype), readout (which states, which positions, which statistic), number of units (prompts, seeds, items).
- Rule: frozen thresholds and the exact verdict strings, including when the verdict is 'not decidable' or 'not testable' and why;
  intervals (state the convention: Qwen 90 % prompt bootstrap with 2,000 replicates unless there is a reason) and the
  multiple-testing treatment. A measurement without a verdict says 'Rule: descriptive'.
- Null / controls: what the null is, how it is built, matched on what (damage, norm, position, length, token count).
- Shows - cannot show: what a positive or negative result means and what it never means.
- Minimum detectable effect: what effect size this design can detect (from round-1/2 noise levels where known), so a
  null result can be read.
- Inputs: files and keys it reads (round-1 files via the round-2 conventions; round-2 result files; new 1b outputs).
- Runs in: session and estimated A100 minutes (float32, TF32 off unless a reason is given; state it), with the rate used.
- Gate(s) it depends on.

## Constraints
- Model: Qwen3-1.7B (and Qwen3-0.6B only where stated as a size comparison), chat template, thinking off, as in rounds 1/2.
  No training or finetuning of Qwen. Toy networks only if they answer something Qwen cannot.
- Compute: Colab A100 (40 GB, sometimes 80 GB) in two Google accounts, sessions run in background; whole round 1b budget
  envelope about 30 A100-hours (part R about 12, part D about 9, part K about 9) unless a part argues for more; one session
  at most about 5 hours; every session resumable (cached evaluations), with a time guard.
- Pre-registration: the plan is frozen (sha256) before any round-1b data are produced; every rule, threshold and verdict string is
  fixed in the plan; anything decided later is an amendment marked post hoc.
- Reuse round-1/2 machinery where possible (rk_loop.py via r2_rk.open_block, r2_theta.py, r2_axes.py ...); name what is new.
- Each part must be able to come out negative: say which outcome would count against the user's hypothesis.
- The user's hypotheses are to be tested faithfully, including the parts that are metaphors: translate each metaphor element into
  a measurable quantity, or state that it has no counterpart and why.
- Write in English, precise, no filler. Use the notation of the round-2 plan where it exists (alpha_b, cap_b, keyseed, damage in
  nats as teacher-forced KL, G, K, rel, A, u_T_perp, ...).

## The chain lens the user means (read the post: scratchpad/chainlens_dl/post.json, field 'text')
'The Chain Lens [interim]' (vgel, Raymond Douglas; LessWrong, 11 Nov 2026,
https://www.lesswrong.com/posts/Nr6fksEqqxNpdz6rk/the-chain-lens-interim): J-lens transport of a hidden state into the final-layer
basis, J-lens top-K seeds, then chains rolled out with the model's MTP head (DeepSeek-V3 style; predicts the token after next from a
final-layer state and the next token's embedding); refinements: contrastive decoding against a prior state (mean of many hidden
states), chain-only attention, re-injection of the transported state. WorkspaceBench macro 0.420 vs J-lens 0.319 (28 families);
computational intermediates strongest (chain intermediates 0.42 vs oracle lens 0.25). 'Hallucination and performance both scale
with chain length' (19 families); the chain lens confabulates more than single-token lenses on the hallucination family, growing
with chain length. Neuron Lens variant: chain lens on MLP neuron write vectors, readouts checked by a round trip (the readout fed
back must activate the neuron above its 99.8th WikiText percentile and above 90 % of random other readouts). Loom lens: trees.
Controls in the post: token-matched J-lens top-k, neutral seeds, foreign seeds, corpus-mean state, shuffle nulls.
Qwen3-1.7B has NO MTP head: a faithful chain lens needs a trained head, a self-rollout substitute, or a second model with an MTP
head; say which and why, and what that changes.
The user's H-CHAIN question maps onto the post's finding: chain-lens confabulation grows with chain length - is that only length,
or does an H-neuron attractor (hallucination-associated MLP neurons, < 0.1 %) drive it beyond what length explains?

## Inverse-free reconstruction (the user's stated preference; binding)
Whenever a step reconstructs causes from effects (earlier fronts or tokens from a later state, text from an activation, what a
neuron or direction encodes, which input produced a readout), use an inverse-free generate-verify-revise loop in the style of
DeepMind's Aletheia (Feng et al. 2026, 'Towards Autonomous Mathematics Research', arXiv 2602.10177: a Generator proposes, a
Verifier returns correct / minor fixes needed / critically flawed, a Reviser repairs, the Generator retries, and the system abstains
at a preset attempt limit). Here: candidates come from forward rollouts (chain lens, sampling, lens seeds); the Verifier is a
forward round trip through the unmodified model (does the candidate reproduce the state, activation or behaviour beyond a matched
null?), decoupled from the generator; fixed attempt limit; abstentions counted and reported; the verifier's own error rate measured
on cases with known answers. Not allowed for reconstruction: matrix inversion, pseudo-inverses, inverse Jacobians,
Tikhonov-regularised inverses (the other 'Aletheia', Fu 2026, arXiv 2601.01532, inverts a regularised confusion matrix; the user
does not want that). Fitting a forward map by regression (e.g. a probe predicting earlier tokens from h_t, a J-lens transport) and
power iteration with forward / backward products are not reconstructions in this sense and stay allowed; say so where used.
