# Extra items for the review of part K (from the user's messages, 13.11.2026)

Goal: make the H-neuron results of Qwen3-1.7B comparable in kind ('anschlussfähig') with neuron-atlas style descriptions of other
models, WITHOUT using any atlas data (no access to goodfire.com / goodfire.ai / static.goodfire.ai; nothing from there stored).
Method definitions only, measured on Qwen3-1.7B, described in our own words; cite the Goodfire Qwen3.6 Neuron Atlas (Qwen3.6-35B-A3B)
only as the methodological reference.

Add to K3 (what the H-set encodes) if it fits the budget (target <= ~30 A100-min), otherwise as an extension:
1. Pole-separate reading of every H-neuron (positive and negative output sign read separately; K3e already reads both poles of
   write vectors - extend to activation-side statistics).
2. Damage per firing: KL divergence of the next-token predictions summed over the next 16 tokens when the neuron is zeroed at the
   firing token; firings above a top-5 % activation cutoff, strength in multiples of the pole's firing threshold, a few rarity bands;
   about 100-200 tested firings per pole for the H-set and for matched random neurons.
3. Modes and damage shares: cluster each pole's firings into modes (e.g. by upstream cause or by context), report the share of the
   pole's damage carried by the main mode; categories single mode / one dominant mode (main mode >= 90 % of damage) / several modes /
   uninterpretable. Tells whether an 'H-neuron' carries hallucination or mostly something else.
4. Effect-based simulatability as the inverse-free verifier of an H-neuron description: simulate the neuron's activations from the
   description on held-out text, patch the simulated activations in, and report the share of the neuron's effect (KL vs zero
   ablation) that is recovered (negative = worse than switching the neuron off). Generator = description from examples / chain-lens
   neuron readouts; verifier = this forward check; fixed attempt limit; abstain if nothing passes.
5. State plainly in the plan: a swap to Qwen3.6-35B-A3B is not a one-line change (MoE with router and experts, ~70 GB in bf16 and
   ~140 GB in fp32, does not fit one A100-40GB; hooks and H-neuron search would need MoE-aware code; rounds 1-2 comparability lost).
   A later bridge round on that model would be a separate plan.
