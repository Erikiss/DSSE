# Double negative, amendment before stage 3 (12 Nov 2026, before any A4a result)

- The traces for A4a are sampled on an A100 (Colab, shard gpu4) with `dn_sample.py`, then A4a runs on that trace set in the
  same session. The rule is unchanged: the first 24 GSM8K test problems, chat template with thinking on, Qwen's recommended
  sampler (temperature 0.6, top-k 20, top-p 0.95), batches of 8 with torch seed 1000 * SEED + batch, at most 1,536 new tokens;
  trace set = the first 12 problems whose thinking ends between 300 and 1,024 tokens, or between 300 and 1,536 if fewer than 12
  of the 24 qualify.
- Reason: on the 4 CPU cores here one batch took 53-92 min; on the GPU the whole set takes minutes, and the cores are needed
  for the code checks. GPU arithmetic gives other samples than the CPU for the same seeds, so the set is sampled anew in full
  rather than mixed. The CPU traces of batches 0-1 are kept as a record (`dn_sample_cpu_batches01.json`) and enter nothing.
- A4a, A4b and A4c themselves are unchanged (`abschnitt11_doppelte_verneinung_plankritik.txt`).
