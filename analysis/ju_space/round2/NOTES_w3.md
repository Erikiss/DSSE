# NOTES of implementer 3 (section 9.5 split (3)): r2_rgi_geom.py, r2_rgi_stats.py

Implements Part A's Qwen geometry of the frozen plan `results/ju_space/plans/abschnitt11_12_runde2_plan.md` (sha256 5e476340...edc2d):
A1-A5 (3.1), the shared objects O7 and O8 (2.4), the gates G29-G33 (6), the statistics conventions of 2.6, CLI and outputs of 9.2. Part A
of the plan is identical in v2 and in the final text (checked by diff), so nothing had to be reconciled. No round-1 output was opened; the
session r2_q3 copies no round-1 file (7.3) and neither script reads <r1_dir>. Every test ran on CPU (NT=1) with rk_loop's tiny random
MOCK model, the real Qwen3 tokenizer and config (`models/qwen3-1.7b`, no weights loaded), the input files `dn_out/mmlu_topics.json` and
`ju_out/{corpus_wikitext,axes_pairs}.json`, and synthetic data with planted structure.

## Files and commands

| File | Command | Code test |
|---|---|---|
| r2_rgi_geom.py | `python r2_rgi_geom.py <root> <r1_dir> <r2_dir> [all \| a1 \| a3 \| o7 \| o8 \| dirs \| meta] [--model qwen3-1.7b \| qwen3-0.6b] [--model-path <dir>] [--force]` | `MOCK=1 python r2_rgi_geom.py <root> <r1_dir> <r2_dir> selftest [--model ...]`: 22 checks |
| r2_rgi_stats.py | `python r2_rgi_stats.py <root> <r1_dir> <r2_dir> [all] [--model ...] [--force]` | `MOCK=1 python r2_rgi_stats.py <root> <r1_dir> <r2_dir> selftest`: 31 checks |
| pool_ref.json | `python r2_rgi_geom.py export_pool_ref <ufo_out/ufo_pool.pt> <pool_ref.json>` (written once here from the pre-round-1 pool file of 10 Nov; source sha256 1905f656...) | used by stage o8 (OD-G7) |

The runner's step table (r2_run_session.py, r2_q3) already calls them as `r2_rgi_geom.py {root} {r1} {work}` / `... --model qwen3-0.6b`
and `r2_rgi_stats.py {root} {r1} {work}` / `... --model qwen3-0.6b` (stats_06 after stats_17); the outputs match its lists. Env: DEVICE
(cuda on Colab; float32 model, TF32 off; statistics in torch float64 on that device), NT, MOCK. Under MOCK the statistics default to 64
bootstrap draws and 20 delta-bootstrap draws (env R2_NDRAW / R2_NDBOOT); a real run always uses the plan's 1,000 and 200.

Self-test timings (MOCK, one CPU core: NT=1, OMP_NUM_THREADS=1, DEVICE=cpu; wall time with 2-4 other agents on the 4 cores, 13 Nov):

    MOCK=1 NT=1 OMP_NUM_THREADS=1 DEVICE=cpu python r2_rgi_geom.py <root> <r1_dir> <work> selftest                         39 s (22 checks)
    MOCK=1 NT=1 OMP_NUM_THREADS=1 DEVICE=cpu python r2_rgi_geom.py <root> <r1_dir> <work> selftest --model qwen3-0.6b      40 s (22 checks)
    MOCK=1 NT=1 OMP_NUM_THREADS=1 DEVICE=cpu python r2_rgi_stats.py <root> <r1_dir> <work> selftest                        54 s (31 checks)

The stats selftest reuses the MOCK outputs of the 1.7B geom selftest when that ran first in the same <work> (as above; GEOM.run
re-validates them by fingerprint and sha256); run alone it takes 74 s. <root> needs models/qwen3-1.7b (tokenizer and config only),
dn_out/mmlu_topics.json and ju_out/{corpus_wikitext,axes_pairs}.json; <r1_dir> is not read; everything is written below
<work>/geom_selftest/<model> and <work>/stats_selftest (both are removed and rebuilt at the start of their test).

Order in r2_q3: geom 1.7B (loads the model once; stages a1, a3, o7, o8, dirs, meta), stats 1.7B, geom 0.6B, stats 0.6B (r_DiD is
computed as soon as both models' statistics exist, from both written JSONs and draws files, and stored in both JSONs; rgi_stats-3).
Neither script calls r2_rk.open_block (no round-1 state is needed), so R1Missing cannot occur. Only r2_stimuli.py of implementer 1 is
used (objects, sentences, a3_lexemes, encode_frame, a3_input, assert_a1, F2_PROMPT, VERBS, FAMILIES, OWNER_ORDER; the final version,
unchanged API); round-1 code: ufo_common.texts (O8), unchanged. Amendments 1-3 do not touch Part A's Qwen geometry.

## Outputs (m = qwen3-1.7b | qwen3-0.6b; local/ is never synced, rgi/ is synced)

- `local/geom_<m>.npz` (A1): `dot`, `obj` [2040, 28, D] float32 (raw block output at '.' / mean over the 6 object tokens), `lex` [2040, D]
  (mean input embedding over the sentence's own tokens), row index arrays `s_obj` (0-59, r2_stimuli.objects() order), `s_frame` (0 F1,
  1 F2), `s_family` (0-4 = IMP CHG DES WOR PRA, 5 = neutral), `s_lexeme` (-1 for neutral), `ids` (padded -1), `off` (F2 prefix length),
  `last`, `objpos`, `texts`, `_meta` (JSON: fingerprint, summary incl. the G29 report). `r2_rgi_geom.sentence_layout()` maps rows to
  [frame, verb form, object].
- `local/delta_states_<m>.npz` (O7): `last`, `mean` [N, 28, D] float16 (float32 and a flag if a value exceeds the float16 range), `labels`,
  `ids`, `topics`, and dn_delta's numbers `{last,mean}_{F,floor,delta,delta_rel}` [28].
- `local/pool_<m>.pt` (O8): `blocks[l] = dict(mu float64 [D], ev float64 [D] descending, U float32 [D, D] (columns = eigenvectors), n_fit,
  med_norm, wiki, chat, dropped)`, `summary` (incl. `od_g7`).
- `rgi/causal_<m>.pt` (A3): `W` [60, 4, 28, D] float32 (rows: IMP-DES h1, IMP-DES h2, CHG-DES h1, CHG-DES h2), `y` [60, 4], `lexeme_logits`
  [60, 12], `scalars`, `objects`, `ids`, `obj_pos`, `definition`, `summary`.
- `rgi/dirs_<m>.pt`: per block `c_o` [28, 48, D] (IMP-DES contrast of each model object, frames pooled, '.', raw), `g` = `g_b` [28, D]
  (mean of c_o), `d_self` (mean over the 12 SELF objects), `s` = `s_b` (d_self minus its projection on unit(g_b): the SELF residual),
  `g_self_f2` (g_SELF of A3: the F2 SELF contrast of IMP-DES), `src_geom_sha256` (sha256 of the local/geom_<m>.npz it was computed from;
  rgi_geom-1), `definition`.
- `rgi/geom_meta_<m>.json`: fingerprint, model record (Qwen3-0.6B pin and the sha256 of model.safetensors in a real run), versions, stage
  summaries (A1 counts / lengths / G29 report; A3 token ids; O7 texts, seeds and the delta table per block = the round-2 values for
  G34; O8 per-block counts and top-5 eigenvalues, OD-G7), every output's path, size and sha256, flags.
- `rgi/geom_stats_<m>.json`: `settings` (draw counts, seeds 1101-1106, conventions), `design`, `gate` (C0 and its parts, uncomposed,
  passing, P, n_P, G30_ok, runs, tie, composed (G32), P_composed, label), `per_block[v][stat]` for v in pooled / F1 / F2 ('.' reading,
  raw metric) and causal: `point, lo, hi` (pointwise 95 % percentile), `sd`, `sim_lo, sim_hi` (sup-t band over P; null outside P),
  `sim_crit`, `n_nan_draws` (arrays over the 28 blocks); `point_only[reading/metric/frame][stat]` (whitened and object-reading variants),
  `lexical[frame][stat]`, `loo['owner:X' | 'substrate:Y'][stat]` (point estimates, primary variant), `naive` (mean cosine, PC1 share),
  `per_cell_P_cell`, `spillover_g_self` (S_self, S_other, z_self, z_other, FR, null moments, s_per_object per block), `delta` (`mmlu`
  last / mean: F, floor, delta, delta_rel, boot_sd, boot_lo, boot_hi; `obj` F1 / F2 / mean; `separability` over P (G31) and over all
  blocks; `tau` per reading), `anisotropy`, `over_blocks` (`P`: the A5 battery for dFG, FG|IMP-DES, rho_S / sigma_self of IMP-DES
  (pooled, F1, F2), FG_causal, rho_causal, dFG_causal, plus r3 with delta_OBJ; `P_composed`: the causal series; `object_reading`: dFG
  with the 'mean' delta), `r_DiD`, `flags`.
  Statistic names: `<stat>|<contrast>` with stat in P_cell, P_own, P_sub, P_gen, P_rel, FG, OG, SG, q, FG_all, OG_LR, rho_S, rho_OO,
  sigma_self (and `_LR` versions) and contrast in IMP-DES, CHG-DES, IMP-CHG, WOR-CHG, WOR-DES, PRA-DES (causal: IMP-DES, CHG-DES);
  derived: dFG, dFG_val, dsigma, dsigma_LR, cos_quality (= cos(g_IMP-CHG, g_WOR-CHG)), cos_valence (= cos(g_IMP-CHG, g_PRA-DES)); in
  the causal variant dFG = dFG_causal, rho_S|IMP-DES = rho_causal, sigma_self|IMP-DES = sigma_causal.
  A5 battery fields: blocks (the blocks used), dropped_blocks (blocks of the set without a finite value; rgi_stats-2), n, k, top,
  bottom, D, D_ci (per-series 95 % percentile interval), spearman_mmlu (+ _ci), spearman_obj (+ _ci), D_whitened, r3, r3_p_pos, r3_p_neg,
  r3_ci, r3_nuis, r3_nuis_p_pos, r3_nuis_p_neg; in `P_composed` the two causal-line entries (FG_causal|IMP-DES, rho_causal|IMP-DES) also
  carry D_sim_ci, D_sim_crit, D_sim_n_draws, D_sim_series: the interval simultaneous over the two D's (rgi_stats-1).
  `r_DiD`: blocks, n, r_DiD, p_pos, p_neg, contiguous, ci (or ci_status), definition, computed, and `provenance` per model (stats file,
  version and 'written', the sha256 of its geom inputs, series_sha256 of the dFG / delta / P it used, draws file and its sha256);
  the same block is stored in both models' JSONs (rgi_stats-3).
- `rgi/geom_draws_<m>.npz`: every per-draw series `<variant>__<stat>` [1000, 28] float32, the draw design (`design_lex_w`,
  `design_lex_split`, `design_para_w`, `design_para_split`) and the delta bootstrap (`delta_boot_last`, `delta_boot_mean` [200, 28]).
- `local/geom_stats_cache_<m>/b<b>.pt`: resumable per-block results (keyed by a fingerprint incl. the inputs' sha256).

## For r2_rgi_verdict.py (r2_ana) and the runner

- A7 per-block items read `gate` and `per_block`: GI = FG|IMP-DES point and `sim_lo`; IS = dFG point / `sim_lo` and dFG_val `sim_lo`; BA
  = C0 for IMP-CHG (`P_cell|IMP-CHG` lower bound and `q|IMP-CHG`), FG|IMP-CHG, cos_valence; robustness from `point_only['dot/white/pooled']`
  and `loo`; labels from FG|CHG-DES, OG, SG, OG_LR|IMP-DES, cos_quality, `per_block['F1'/'F2']['dFG']`, `point_only['obj/raw/pooled']['dFG']`;
  self items from `per_block['F1'/'F2']` (rho_S, sigma_self `sim_hi` / `sim_lo`, dsigma `sim_lo`; `_LR` for the LR column); causal items
  from `per_block['causal']` and `gate.composed`. Over blocks from `over_blocks`, `delta.separability.last.ok` (G31) and `r_DiD`
  (identical in both models' JSONs; its `provenance` names the statistics it was computed from).
- A7 causal line ('with simultaneous intervals'): use the interval simultaneous over D(FG_causal) and D(rho_causal) on the composed
  blocks of P, not the per-series `D_ci`. It is stored as `over_blocks['P_composed'][...]['D_sim_ci']` / `D_sim_crit`;
  r2_rgi_verdict.causal_sim builds the same sup-t band from geom_draws_<m>.npz ('causal__FG|IMP-DES', 'causal__rho_S|IMP-DES') with the
  stored top / bottom, so the two can be compared (they agree to float32 precision of the draws file; selftest). `D_ci` is the A5
  per-series 95 % Bayesian-bootstrap interval.
- A3 is undefined at block 27 (see item 19): over P, the causal series' battery uses the other blocks of P (`dropped_blocks` = [27]).
  No rule reads that over-P battery (the causal line uses `P_composed`, which never contains block 27).
- u_T_perp spillover (A3, at the blocks of B3): `r2_rgi_stats.spillover(*r2_rgi_stats.spill_inputs(W, b), u_T_perp(b), null_u_T_perp(b))`
  with W from rgi/causal_qwen3-1.7b.pt; returns S_self, S_other, z_self, z_other, FR (NaN unless z_self >= 3).
- A6 readouts on g_b and s_b: `rgi/dirs_<m>.pt`; `r2_rgi_stats.sign_flip_nulls(c_o[b], n=2000, seed)` builds sign-flip nulls over the 48
  model-object contrasts, each null constructed like its axis.
- G34: the round-2 delta per block is `geom_meta_<m>.json['stages']['o7']['delta'][last|mean]['delta']` (also in geom_stats).
- The runner should list `analysis/ju_space/round2/pool_ref.json` among the reference files hashed into the manifest (OD-G7 input).

## Ambiguities and the readings chosen

A1, A3, O7, O8 (r2_rgi_geom.py)
1. 'F2 ... prefix computed once': the chat prefix is tokenised once and prepended to every reply's ids; each forward runs prefix +
   reply in full (rk_loop.reply_read's reading of a reply; no KV-cache reuse, which could differ at rounding level).
2. Lexical reference: mean of the embed_tokens output over the sentence's own 11 / 9 tokens (the F2 chat prefix is excluded; it is the
   same for every F2 sentence).
3. G29: r2_stimuli.assert_a1 (lengths, object-token identity, A3 inside the template) plus, for every one of the 1,020 F2 sentences, that
   tokenising chat('Say one sentence.') + sentence gives prefix ids + sentence ids (assert_a1 checks three objects). A failing assert
   raises before any state is written (the session's step fails: 'r2_q3 stops').
4. Sentences are batched by equal token length (no padding, no attention mask); the hook readings equal unbatched forwards (selftest).
   A1 states are stored float32 (the plan fixes float16 only for O7; massive-activation dimensions make float16 differences coarse).
5. A3 lexeme halves are fixed by Appendix A's list order: half 1 = the first two lexemes of IMP / CHG / DES, half 2 = the last two. The
   logits are the next-token logits after 'I would' (lm_head of model.model's final-norm output at the last position = model(...).logits
   at that position). One leaf vector per block is added at all six object positions.
6. O7: the delta values in geom_meta are computed from the full-precision states (float64 of the float32 outputs) with dn_delta's fisher
   and permutation stream; texts are read in batches of 64 (dn_delta reads one at a time; all texts have the same length). The stored
   float16 states reproduce these deltas to 6e-6 (MOCK). dn_delta's probe accuracy is not computed (not used in round 2). With the
   repository's MMLU file the selection gives 42 questions per topic (336 texts), as in round 1.
7. O8: only the fit part (i mod 3 != 2) is read in the second pass (the held-out part only served ufo_pool's k_irr); the counts reported
   are those of the fit part. Accumulation and eigendecomposition in float64 on the device.
8. OD-G7: pool_ref.json holds `top5_eigenvalues` (= sd^2 of ufo_pool.pt's block 14) and all blocks' top-5; the check looks for env
   POOL_REF, then <root>/ufo_out/pool_ref.json, then the repository copy; each of the 5 eigenvalues must lie within 5 % (flag otherwise).
   Not applicable to the Qwen3-0.6B arm and to the MOCK model.
9. Qwen3-0.6B (OD-A6): the pinned revision is r2_run_session.py's (c1899de289a04d12100db370d81485cdf75e47ca, model.safetensors sha256
   f47f7117...); a real run hashes <model path>/model.safetensors before any work and stops on a mismatch (G33); the runner's own checksum
   step decides whether the arm runs. The model path is <root>/models/<m> or --model-path. Loading = ufo_common.load_model's call
   (from_pretrained, dtype float32), which hard-codes the 1.7B path and is therefore not called directly.
10. dirs: g_b = mean of the 48 model-object IMP-DES contrasts (frames pooled, '.', raw metric); s_b = 'the SELF residual' read as the SELF
    contrast minus its least-squares projection on g_b (the critique's 'self contrast minus its projection on g'); g_SELF = the F2 SELF
    contrast (A3).
11. Resume: a stage is skipped when its output exists with the same fingerprint (version, model, MOCK, width, depth; not the path, so a
    continuation under another root resumes). Derived outputs follow their inputs, also across invocations (rgi_geom-1): dirs is redone
    when a1 was redone in this call, when it records another source sha256 than the local A1 file on disk, or when it records none;
    geom_meta is redone when any stage was redone in this call or when a file it certifies exists with another sha256. Without the local
    A1 file nothing is compared (a single 'dirs' or 'meta' call behaves as before). Hashes are memoised per process (the 0.94 GB A1 file is
    read once).

A2-A5 (r2_rgi_stats.py)
12. Split combinations (OD-A1): one split of each 4-lexeme family and one split pattern of the paraphrase index shared by all substrates
    and non-model categories (this gives the plan's 27 = 3 x 3 x 3 per contrast, 9 with WOR / PRA). Point estimates average the inner-
    product classes over the combinations and then form the ratios (the RGI design's sim_tg.py). The average is computed exactly from the
    expected outer products of the weights over the 81 joint combinations (IMP x CHG x DES x paraphrase splits; each contrast's own
    combinations occur equally often), and checked against an explicit loop over all combinations (agreement 6e-14).
13. Cell contrasts use weighted means within halves (weights normalised within each half). P's are means over ordered cell pairs of a
    class, each pair averaging its four half combinations (h != h', i != i'); P_rel uses h = h', i != i'. FG_all treats each non-model
    cell as differing from every other cell in owner and substrate.
14. Bayesian bootstrap: Dirichlet(1) weights on each family's lexemes, on the 4 paraphrases of each substrate (shared by the four owners,
    because a paraphrase is the substrate's wording) and on the 4 items of each non-model category; one random split per draw; the same
    1,000 draws (seed 1101) for all blocks, readings, metrics, frames, contrasts, the causal statistics and both models.
15. Self: rho(SELF, X, s) = cross-validated product of the two cells / sqrt of both cells' P_cell, NaN unless both P_cell > 0; rho_S and
    rho_OO are NaN-means over the 9 values (3 for the LR column). Self statistics are produced for all six contrasts; dsigma = IMP-DES
    minus CHG-DES (all substrates and LR column).
16. 'Cross-validated, disattenuated direction cosine' = cos of the general parts: P_gen between the two contrasts (other owner and other
    substrate, disjoint halves; shared families such as CHG use the same split, so their halves are disjoint too) divided by
    sqrt(P_gen(A) P_gen(B)); NaN unless both P_gen > 0.
17. Intervals: pointwise = 2.5 / 97.5 percentiles of the draws; sup-t band over P = point +- c sd_b with c = the 95 % quantile over draws of
    max_{b in P} |draw_b - mean_b| / sd_b (critical values 1.96 for identical columns, 3.0 for 20 independent ones in the selftest).
18. C0 uses the pointwise lower bound of P_cell and the point estimate of q. P: the longest contiguous run of passing blocks; ties by
    the higher mean delta_MMLU ('last'), then the earlier run. Bands are computed over P whatever its length; with fewer than 12 blocks
    G30_ok is false and the flag and label 'undecided (fewer than 12 passing blocks)' are set (r2_ana decides what follows). BA's 'C0 for
    IMP-CHG' bound is available pointwise and simultaneous (the plan's order makes it a simultaneous bound).
19. A3 statistics: the lexeme halves are fixed by the four scalars, so the bootstrap reweights paraphrases only (the A2 draws' paraphrase
    weights and splits); point = mean over the 3 paraphrase splits. A3 is undefined at block 27: a leaf added to the output of the last
    block at the object positions cannot reach the last position, whose logits are read, so W[:, :, 27] is identically zero, every causal
    P is 0, the ratios are NaN, and block 27 is never 'composed' (NaN < 0.8 is false; G32). Checked in both selftests; the coordinator
    records this as a clarification in the amendments file. Spillover: w_o = mean of the two IMP-DES half gradients of model
    object o; s_o = w_o . unit(u); z = (mean - null mean) / null SD (ddof 1); the 200 pool-covariance directions use
    Pool.direction('cov')'s formula unit((z * sqrt(ev)) @ U^T) with torch.Generator seed 1105 + block.
20. tau (point only): g1, g2 = the general (12-cell mean) directions from disjoint lexeme and paraphrase halves, numerator and denominator
    averaged over the 81 combinations and both disjoint pairings; topic span = the centred O7 topic means (rank 7), 'last' for the '.'
    reading, 'mean' for the object reading; 'the expectation for pool-covariance random directions' = the mean of |P v|^2 / |v|^2 over
    the 200 directions (used by the 'topic-like' label); tr(P Sigma) / tr(Sigma) is reported next to it.
21. delta bootstrap: within-topic resampling (each slot keeps its topic), F - floor with 50 fresh permutations per draw; draws and
    permutations shared by all blocks and both readings (seed 1103). Separability over P: SD of the cubic residual with ddof 0 (as the
    design's sim_sep.py) against the mean bootstrap SD (ddof 1); also reported over all 28 blocks and for the 'mean' reading.
22. delta_OBJ: the '.' state of the neutral sentences, labels = the 15 cells; permutations from np.random.default_rng([1104, block]),
    F1 then F2 (per-block seeds keep the resumable cache order-independent).
23. Anisotropy ('mean cosine between DES-frame states'): mean pairwise cosine of the 240 frames-pooled DES states of the reading (uncentred).
24. A5: rankings use O7's round-2 delta (C11); top / bottom thirds by a stable sort; Spearman with average ranks; r3 needs >= 6 blocks;
    one set of 2,000 surrogates (design's sim_depth.py construction) per delta series and block set (seeds 1102-1105), shared by all
    series; r3 per draw against the fixed delta residual; intervals of D and Spearman from the draws with delta fixed. The causal line
    uses the composed blocks of P (its own thirds and surrogates); the object reading uses the 'mean' delta and point estimates only.
    A series enters with the blocks of the set where its point value is finite (rgi_stats-2; in practice only the causal series at
    block 27): thirds, Spearman, the cubic residuals and a surrogate pack of its own (the set's seed: 1102 for P, 1103 for the object
    reading, 1104 for P_composed) are taken over those blocks, with the matching delta_MMLU, delta_OBJ, anisotropy and draw columns; the
    dropped blocks are listed. D_ci is a per-series percentile interval; for A7's causal line D_sim_ci adds the sup-t interval
    simultaneous over D(FG_causal) and D(rho_causal) (point +- c sd over the draws finite in both, c = 95 % quantile of the larger
    standardised deviation; rgi_stats-1).
25. r_DiD: blocks = P_1.7B & P_0.6B (needs >= 4); Pearson correlation of the differences; one-sided p from 2,000 phase-randomised,
    mirror-padded surrogates of the (demeaned) delta difference, not residualised (the difference itself removes the shared depth trend);
    a per-draw interval when both draw files exist with the same number of draws (else `ci_status`). rgi_stats-3: r_DiD is computed from
    both models' written JSONs and draws files (float32 draws for both, so the value does not depend on which run computed it), after the
    current model's files are written, and stored with both sides' provenance in both JSONs. Any later run of either model's statistics
    recomputes it, and a run that is skipped because its output exists recomputes it when the stored provenance no longer matches either
    model's files.

## Deviations from the plan

None in any measurement, threshold or rule. Additions: the lexical-reference variant, per-cell P_cell, the extra over-block series
(dFG_causal, per-frame rho_S / sigma_self, r3 against delta_OBJ), whitened D for every pooled series, the stats cache, pool_ref.json and
its export command, the helpers spillover / spill_inputs / sign_flip_nulls for r2_ana, and (13 Nov) D_sim_ci for the causal line and
r_DiD's provenance. MOCK-only: smaller draw counts by default.

## Review fixes (13 Nov; scratchpad r2_review/fixes/rgi_geom.md, rgi_stats.md; versions 1.1 of both scripts)

- rgi_geom-1 (dirs not invalidated across invocations): dirs_<m>.pt stores `src_geom_sha256`; run() treats dirs as stale when that
  differs from the local A1 file, when it is missing, or when a1 was redone in the call, and geom_meta as stale when a certified file
  exists with another sha256 or a stage was redone (verifier's corrected fix incl. its three points; item 11). Selftest: A1 recomputed by
  an earlier call (stage a1 alone), then 'all' -> dirs and meta redone and every certified hash equal to the file; an old-format dirs
  without the source hash is redone; 'dirs' alone without the local A1 file keeps the file.
- rgi_stats-2 (causal battery over P dropped when P contains block 27): implemented as the verifier describes (finite blocks, a pack per
  block set, matching delta / delta_OBJ / anisotropy / draw columns, `dropped_blocks`; not masked inside over_blocks); items 19 and 24.
  Selftest: the planted worlds now have W = 0 at their last block, as in the real run; the over-P causal entries have D and r3 over the
  other blocks of P with dropped_blocks = [last], and the last block is not composed. The geom selftest checks W[:, :, 27] == 0.
- rgi_stats-1 (causal line not simultaneous): D_sim_ci / D_sim_crit stored next to D_ci in both causal-line entries of P_composed
  (supt_band over the two per-draw D's); the hand-over section says the A7 causal line must use the simultaneous band. Selftest: the
  stored band equals r2_rgi_verdict.causal_sim's construction recomputed from the draws file (c 2.37 on the planted world; differences
  below 1e-5).
- rgi_stats-3 (r_DiD without provenance): see item 25. Selftest: r_DiD identical in both JSONs with both provenances; a recompute of
  one model refreshes it in both; a change of one model's statistics behind the other's back is detected by the provenance and a
  skipped run of the other model recomputes it (planted world: 0.716 -> 0.458, equal to a direct recomputation).
- argparse usage (rgi_geom-2, refuted as a real-run defect): `usage=__doc__.replace('%', '%%')` in both scripts; selftests check that -h
  exits 0 and a bad --model exits 2.
- Not a finding: the per-draw products were reordered exactly (paraphrase step as the sum of the two half combinations instead of the
  full 60 x 60 Gram, equal to the old construction to 4e-16, checked in the selftest) and run in chunks of 4 draws on a CPU; this halves
  the per-block CPU time and lowers GPU memory. The dense lexeme product stays (a gather-based variant was slower on one CPU core).

## Tests run (this container, CPU, NT=1, OMP_NUM_THREADS=1, MOCK; test directory scratchpad/r2_impl/w3)

- `r2_rgi_geom.py ... selftest`: 22/22 for qwen3-1.7b and 22/22 for qwen3-0.6b (39 s and 40 s): outputs and layout; batched hook reads
  = unbatched forwards (1e-8); F2 rows = chat + reply with '.' last; A3 gradients = 5-point finite differences on a float64 copy of the
  MOCK model (2e-4; Qwen3's RMSNorm computes in float32 even then) and float32 = float64 gradients (5e-7); O7 texts and labels = dn_delta.py's
  own statements executed unchanged; F and floor = dn_delta's fisher and permutation stream (4e-17); float16 storage changes delta by
  6e-6; O8 mean and eigenvalues = ufo_pool.py's own passes executed unchanged (exact); OD-G7 pass and a flagged 20 % mismatch; dirs by
  definition; meta; resume skips everything, recomputes A1 + dirs + meta after a lost local file, refuses an output of another run;
  A3 identically zero at block 27; rgi_geom-1 (two checks); -h and a bad --model exit 0 and 2.
- `r2_rgi_stats.py ... selftest`: 31/31 (54 s after the geom selftest, 74 s alone): point estimate = explicit loop reference over all 81
  combinations (68 statistics, 6e-14) and per draw (2e-13); the reordered paraphrase step = the dense construction (4e-16); planted FG, OG, SG, rho_S, sigma_self and cosines recovered (worst error 0.033 at D 1536); a planted FG of
  0.6 recovered without bias over replications (mean error +0.012, SD 0.021); sup-t critical values; surrogate r3 test size 0.040 under a
  smooth depth-only null and power 0.96 for a planted relation; Fisher ratio (batched and Gram forms, with duplicated rows) = dn_delta's
  fisher; delta bootstrap = direct recomputation; tau limits; spillover (z, FR rule); helpers; r_DiD (planted r 0.998); passing set
  incl. ties; a planted 14-block world for both models end to end (P = blocks 2-13, FG per block within 0.12, D(dFG) > 0 with its
  interval above 0, Spearman > 0.5, r3 and its surrogate p present, r_DiD over the 12 common blocks), rgi_stats-1, -2 and -3 (see above),
  the per-block cache, refusal of a stale local file, -h / bad --model, and the full chain on the MOCK model's outputs (8 draws).
- CLI exactly as the runner calls it (both models in sequence, then the 1.7B steps again): exit 0, the runner's listed outputs present,
  r_DiD identical in both JSONs with provenance, the repeated (skipped) 1.7B statistics leave it unchanged.

## Open points

- GPU timing is not measured here; by operation counts the statistics need roughly 30-60 s per model on an A100 (budget 120 s and
  60 s). Memory peaks: about 1.5 GB of GPU memory per draw chunk of 64; the 1.7B A1 states are 0.94 GB on the host.
- With the random MOCK model no block passes C0, so P is empty there; the planted synthetic world covers the code paths with a non-empty P.
- r2_run_session.py's self-test list for r2_q3 runs only the stimuli check; the three selftest commands above (timings there) can be
  added; they write only below <work>/geom_selftest and <work>/stats_selftest. Run the geom one first: the stats one then reuses its
  outputs.
- pool_ref.json records the scratchpad path of its source; its content and the source sha256 are what matter.
- rgi_stats-3 side effect: stats_06 rewrites the r_DiD block of geom_stats_qwen3-1.7b.json after stats_17 has finished, so the sha256
  that the runner's stats_17 step record holds for that file is outdated afterwards. The runner uses recorded outputs only for
  existence checks and clean-up (r2_run_session.py, 13 Nov), so nothing breaks; a consumer that compares step-record hashes with the
  files should expect this one difference (the file's own r_DiD.provenance says what it was computed from).
