# NOTES of implementer 1 (plan v2, section 9.5 split (1)): r2_rk.py, r2_stimuli.py, r2_axes.py, r2_gen_loop.py

Written against `plan_round2_v2.md` (identical to `plan_round2_final.md` at the time of writing). No round-1 output was opened; every
test ran on MOCK round-1 states made locally with `rk_loop.py MOCK=1` (blocks 14 `all`, 8 and 20 `pilot`, `loops`, `k5`) plus the fake
`dn_delta.json`, `blocks_chosen.json`, `summary_b<b>.json` and `shard_done.json` written by `r2_rk.mock_r1`.

## Files, commands, code tests

| File | Command | Code test (MOCK) |
|---|---|---|
| r2_rk.py | import only | `MOCK=1 python r2_rk.py selftest <root> <workdir>`: 31 checks (builds a MOCK block-14 fixture in `<workdir>/r1` if missing) |
| r2_stimuli.py | `python r2_stimuli.py check <tokenizer.json or its directory>` | the same command (all asserts of G29 and the template checks; exit 1 on failure) |
| r2_axes.py | `python r2_axes.py <root> <r1_dir> <r2_dir> [texts \| read \| stats \| all]` | `MOCK=1 python r2_axes.py <root> <r1_dir> <r2_dir> selftest`: 11 checks |
| r2_gen_loop.py | `BLOCK=<b> python r2_gen_loop.py <root> <r1_dir> <r2_dir> [run]` | `MOCK=1 BLOCK=<b> python r2_gen_loop.py <root> <r1_dir> <r2_dir> selftest`: 9 checks (after r2_axes) |

Order in session r2_q1: r2_axes.py (opens block 14), then r2_gen_loop.py with BLOCK = 14, b_lo, b_hi (one process per block). The
block-14 GEN loop can share the r2_axes process (one model load): `r2_axes.main(root, r1, r2)` then `r2_gen_loop.run(root, r1, r2, 14)`
(tested). `<r2_dir>/local/` holds the large intermediate reads of r2_axes (`local/axes_reads.pt`); it is meant to stay local (9.5: not
synced), the stage reruns from `axes/o2_texts.pt` if it is lost.

Outputs: `axes/o2_texts.pt` (six text sets as token ids, keys), `axes/tok14.pt` (per-token block-14 states, float16, six sets:
R1_RSI_PRO/ANTI, R1_SEA_PRO/ANTI, RGI_PRO/ANTI; each `dict(tok=[64 x [L, D]], conts, prompt, source)` plus `fold`),
`axes/r2_axes.pt` (`texts`, `keys`, `B3`, `GA {b: bool}`, `X28 {set: [64, 28, D] float16}`, `blocks {b: X {set: [64, D]}, diff, keep,
u_RGI, u_TOPIC, u_GI_TOP, u_RGI_perp, u_RGI_perpT, u_dir, u_diff, u_com, u_T_perp, u_RSI_session(_perp), u_S, u_neg, null_u_RGI_perp,
null_u_RGI_perpT, null_u_dir, null_u_S_perp, null_u_neg [2000, D], basis_R1, basis_RGI, S_states, NEG_states, GA, auc_RGI,
boot_disatt}`), `axes/r2_axes.json`, `rgi/dyn_b<b>.json`, `rgi/dyn_b<b>.pt` (`start`, `d {1..4}`, `ev {0..3: delta, ok, prompt_of, xp,
xm, first_div, conts, damage_of_push, alpha, capped_share, dropped}`, `d4_RSI_a`, the two axes).

## API of r2_rk.py (`rk` = the rk_loop module returned by open_block)

Round-1 files (no model needed)
- `find_r1(r1_dir, rel, block=None, required=True) -> path`: glob `<r1_dir>/gpu*/<rel>`; duplicates by OD-G9 (gpu1 for block 14, gpu2 for
  b_lo, gpu3 for b_hi, else the most recent file); every call appended to `FINDER_LOG` (write it into the output JSON). Missing:
  `R1Missing` (a `FileNotFoundError`) or None.
- `r1_blocks(r1_dir) -> dict(b14, b_lo, b_hi, B3, flags, chosen, ran, source)` (OD-G2, G7); `delta_order(r1_dir, info=None) ->
  dict(order=[b(1), b(2), b(3)], delta, ties, flags, source)` (OD-G3, G9).
- `load_r1_state(r1_dir, b)`, `load_r1_json(r1_dir, rel, block=None, required=True)`, `r1_meta(r1_dir, b)` (= loops.json '_meta').
- `load_o3(r1_dir, b)` / `o3_bank(ST, loops, b, D=None, path=None)`: O3 bank `{L: dict(name, suffix, start, d {1..4}, ev {1..4: pairs
  behind d_n: delta, ok, prompt_of, xp, xm, first_div, damage_of_push, alpha}, rel {1..4}, row {1..4}), '_meta', '_missing'}` for L in
  `loops_of_block(b)`; `pool_dir(kind, b, i)`, `k6_null_dirs(b=14)` (O6 order: rand 0-7 = nulls 0-7, cov 0-7 = nulls 8-15).
- Statistics: `unit`, `cosf` (= rk_loop.cos), `perp_basis(vectors) -> Q [D, k]`, `project_out(V, Q)`, `halvings(n_prompts)` (= rk.HALV),
  `axis_from_texts(Xp, Xa, fold, seed=None, basis=None, n_null=2000)` (returns u, dA, dB, diff (kept), diff_all (NaN rows), keep, kept,
  auc, paired_win, gap_cross, null, and with a basis Q, u_perp, gap_perp, gap_perp_cross), `axis_rel(diff_all, keep=None, Q=None,
  extra=None) -> dict(rel, r, rel_floored, n_halvings)`, `stats_on_axis(ev, axis, null, n_prompts=None) -> (dict(c, p, Q, rel, r,
  rel_floored, A, norm), d)`, `prompt_bootstrap(n_prompts, B=2000, seed=0, n_rsi=None) -> W [B, NP]`, `pair_weights(W, prompt_of, ok)`,
  `boot_mean(values, prompt_of, W, ok=None) -> [B, ...]`, `interval(draws, level=0.90) -> (lo, hi)`.
- Files: `write_json`, `torch_save` (atomic), `jsonable`.

Model side (after `rk = open_block(root, r1_dir, r2_dir, b)`; one process = one block; `BLK` = dict(b, alpha (= alpha_b), alpha_scale,
suffix, cap, med, r1_state, state, r1_dir, r2_dir, mock, pooldirs, persona, device))
- State guard: `guard_check()`, `save()` (guarded; `rk.save` is replaced by it, so rk.evaluate / run_loop / push_damage / k6_direction
  are guarded), `st_put(key, value, do_save=True)`, `new_key_ok(key)`; `StateGuardError`. New keys: `r2_*`, or rk keys of a loop / K6
  name starting with `r2` (`ev_r2*`, `dir_r2*`, `dmg_r2*`, `k6_r2*`).
- `read_all_blocks(seqs_prompt, conts, batch=32, blocks=None, token_blocks=(), tok_dtype=float16) -> dict(X [N, nblocks, D] float64,
  blocks, tok {b: [N x [L, D]]})`.
- `read_windows(seqs_prompt, conts, first_div=None, batch=32, n_first=20) -> dict(first_div, X, xp, xm, delta, ok)` per window
  `all` / `div` / `first20` (rows 2j = '+', 2j+1 = '-'); `first_divergence(a, b)` (rk.evaluate's rule).
- `CbarHook(cap=None)` (context manager, generation mode; `.cbar(conts) -> dict(per_text_mean, pooled, n_tokens)`), `cbar_read(seqs_prompt,
  conts, cap=None, batch=32)` (read mode; kappa = pooled / BLK['med']).
- `set_temperature(T)`, `temperature(T)` (context manager), `damage_T(d, alpha, T, n=8)`.
- `WriteOrth(U)` (context manager, 57 hooks), `leaf_grads(seqs, positions, blocks, scalars_fn, batch=16) -> G [S, N, nblocks, D]`.
- `evaluate_variant(d, alpha, keyfn, nseed=1, prompts=None, T=None, signs=(1, -1), gen_ids=None, read_ids=None, hooks=(), cbar=True,
  entropy=False, batch=32, n_first=20, cache_key=None)`: returns delta / ok / xp / xm (all-token window), prompt_of, seed_of, first_div,
  conts, alpha, T, keys, signs, capped_share, dropped, n_pairs, windows {all, div, first20}, cbar, entropy (if asked).

## Ambiguities and the readings chosen

r2_rk.py
1. open_block restores sys.argv after importing rk_loop (2.3 says it 'sets' it; rk_loop reads it only at import). POOLDIRS and PERSONA
   keep values already in the environment, else `<root>/gpu_pkg/rk_dirs.pt` (fallback: the repository copy `gpu/data/rk_dirs.pt`) and
   `<root>/rsi_dl/persona` (round 1's values). ASCALE = repr(alpha_scale_b) from loops.json '_meta' (G10); open_block checks rk.SUF and
   alpha_b against '_meta'. An existing copy of the state is kept (resume). Asking for the open block with another r1_dir / r2_dir raises.
2. State guard: the round-1 key set and fingerprints are taken from the round-1 source file (not from the copy), so a resumed copy that
   already holds round-2 keys is handled; the fingerprint covers key sets, tensor dtypes / shapes / float64 sums (plan) and, in addition,
   non-tensor leaves by value (token-id lists by sha256). It is written to `<r2_dir>/rk/b<b>/r1_fingerprint.json`. New-key rule: 2.3's
   'r2' loop names plus `k6_r2*` (rk.k6_direction(name, ...) stores `k6_<name>`; C-Q2 needs it with an 'r2' name).
3. read_windows: the 'div' window starts at the first divergent token (positions >= first_div); 'first20' = the first min(20, L)
   tokens; an identical pair gets delta 0 in 'div' and its 'div' states are set to its 'all' states; a text that ends exactly at the
   divergence (one text a prefix of the other) has no 'div' tokens: NaN, the pair is not ok in that window (counted, not imputed).
4. cbar hook: decode step k feeds back continuation token k, so a 40-token text contributes tokens 0..38 (the last token is never fed
   back); a text that stopped contributes all its L tokens. The read mode (references, kappa_b) uses all L tokens. Values are min(|h_t|,
   cap_b) of the pre-push block output (prepend=True), i.e. the unsteered block-b state of the generated text.
5. axis_rel (split-half rel of O2 axes): rk.stats's formula including its floor 0.05 (flag `rel_floored`); half axes projected like the
   axis. stats_on_axis takes the prompt count from rk.NP (or prompt_of); for evaluations on a prompt subset pass n_prompts explicitly.
6. prompt_bootstrap: rk_verdict.abar_bootstrap's draw (np.random.default_rng(seed); per replicate NRSI RSI-topic and NP - NRSI scene
   prompts with replacement); the seed is the caller's (OD-G6).
7. evaluate_variant works on pairs sharing one key per pair (rk.evaluate's design); a per-row generation prompt (e.g. pro / anti
   instructed texts under one counter-push) goes through gen_ids returning a 2-tuple of id lists; hooks act during generation only.
   The `entropy` option (B-Q1's continuation entropy) is an addition: mean over the sampling steps that produced the continuation's
   tokens (the step that produced the stop token is excluded).
8. MOCK only: `pool_dir('rand', b, 0 / 1)` = rk_loop's MOCK RND starts; other MOCK Pool directions are seeded random unit vectors;
   `k6_null_dirs` under MOCK = rk.k6's four MOCK nulls (seeds 100-103), so MOCK has 4 K6 nulls, not 16.

r2_stimuli.py
9. The turn-2 suffix is `'<|im_end|>\n<|im_start|>user\nContinue.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'` (no
   space before 'Continue.': the plan's line break falls there; the check against apply_chat_template confirms this string).
10. G29 'object-token identity within a group' is asserted as: every A1 / A3 text tokenises as prefix + verb + enc(' ' + object) + final
    token, and a model object's 6 tokens = its owner's 4 tokens + its substrate's 2 tokens (so the object tokens are identical across
    verbs, frames and cells sharing an owner or a substrate). The A3 user content is also asserted in place inside the chat template,
    and 'I would' does not merge with it. Object positions: F1 4-9, F2 / neutral 2-7, A3 user content 9-14 (+ the template offset).
11. `nesting_split()` returns rsi_direction.py's split (default_rng(13), first 12 = training); the plan does not fix C-Q6's split, it is
    offered for reuse only. `check` also verifies that every depth-3 / depth-4 nested sentence has the second ' the way it' (OD-C4).
12. `check <tokenizer.json>` loads the tokenizer with transformers from the file's directory (the chat template is in
    tokenizer_config.json). Copies of round-1 constants (rk_loop.INSTR['rsi'], rsi_direction.py's OBJ / nested / repeated) are compared
    with the round-1 sources through `ast` (nothing of those files is executed).

r2_axes.py
13. Generation: per instruction pair (RSI, RGI, TOPIC) rows interleaved (pro, anti) per prompt in rk_loop.text_axis's order, batch 32,
    so the regenerated RSI texts have round 1's batch composition. MOCK only: keys keyseed('axis', 'rsi', p, s) with s = +1 / -1 (rk_loop's
    own MOCK rule), so that the MOCK RSI texts reproduce the MOCK round-1 texts and the match-rate code is tested on equal texts.
14. 'rel_T' (and a_p of the common / difference parts) come from round 1's per-prompt RSI differences at block b (R1.state[b] 'axis_rsi'
    Xp, Xa), because u_T_perp is round 1's axis (O1); the session's own RSI axis is reported separately (`axes.RSI_session`,
    `g0_RSI` = its gap_perp_cross on the round-1 basis, `cos_RSI_session_R1`).
15. Disattenuated cos(u_T_perp, u_RGI_perp): 90 % interval from 2,000 prompt-bootstrap replicates (seed 25, the same draws at every
    block); per replicate both axes and both rels are recomputed with replicate weights (rel over the 20 K2 halvings, prompts keeping
    their weights inside a half); the projection bases (round-1 foreign axes and the point estimate of u_TOPIC) are held fixed.
16. u_RGI_perpT = u_RGI_perp minus its component along unit(u_T_perp) (literal Gram-Schmidt against u_T_perp only; it is therefore
    orthogonal to u_T_perp but not exactly to u_TOPIC). Nulls: the same operation on the projected RGI nulls.
17. GA(b) uses the raw cross-fitted centred AUC of RGI_PRO against RGI_ANTI (rk_loop's AUC on the fold axes dA, dB), `ga_gate()`.
18. u_S nulls: null_k = mean(S[perm_k[:nb]]) - mean(S[perm_k[nb:]]) with perm_k = torch.randperm(2 nb) drawn sequentially from
    torch.Generator().manual_seed(23), S = [pro states; anti states] in rk_loop.axes() order, projected on the round-1 basis like
    u_S_perp. NEG nulls: E +-1 [2000, 40] from torch.randint with seed 24, null = E @ diff / 40 (not projected).
19. Match rate (22k): share of identical texts and the token share of the common prefix (sum of common-prefix lengths / sum of the
    longer lengths), for RSI_PRO and RSI_ANTI against R1.state[14]['axis_rsi']['conts'].
20. Statements and NEG are read at the blocks of B3 (block 14 always); the O2 texts at all 28 blocks (stored as float16 `X28`).
21. A block of B3 without round-1 state gets `status 'not run (round-1 output missing)'` (G6); r2_gen_loop then writes the same status.

r2_gen_loop.py
22. The GEN loop runs wherever GA(b) holds, whatever G-blk(b) says (G-blk only selects the blocks that enter the A7 dynamics rules in
    r2_ana); where GA fails nothing is generated and no block is opened (OD-A11).
23. E_GI = A_4 / A_2 on u_RGI_perp (NaN if |A_2| < 1e-12); rel(d_n) of the GEN loop is run_loop's rel (axis independent);
    R_4(RSI_a, GEN) uses rel(d_4^RSI_a) = R1.loops[b]['RSI_a'][3]['rel'] and d_4^RSI_a = R1.state[b]['dir_RSI_a<SUF>_4']; S_4 is
    rk_verdict.S4's formula on the GEN loop. The A7 dynamics rules themselves are left to r2_ana (r2_rgi_verdict.py).

## Deviations from v2

None in any measurement, threshold or rule. MOCK-only differences: the O2 key rule of item 13, and 4 K6 nulls (rk_loop's MOCK). Additions
beyond the plan's lists: the `entropy` option of evaluate_variant, `cbar_read`, `delta_order`, `r1_blocks`, the intermediate files
`axes/o2_texts.pt` and `local/axes_reads.pt`, diagnostics in r2_axes.json (`r1_read_check`, `cos_RSI_session_R1`, `cos_u_dir_u_T_perp`,
`perp_share_RGI`, `u_T_perp_recompute_rel_diff`), and a planted-signal check of the O2 statistics in the r2_axes selftest.

## Tests run (this container, CPU, NT=1, MOCK)

- `r2_stimuli.py check models/qwen3-1.7b/tokenizer.json` (real Qwen3 tokenizer): all checks pass (60 objects, 2,040 A1 sentences, 60 A3
  inputs, RGI30, copies, nesting, turn-2 suffix 16 tokens, cue template).
- `r2_rk.py selftest`: 31/31 (existing fixture, 10 s) and 31/31 from an empty workdir (builds its fixture, 50 s). Among them: rk axes(),
  rk.stats, rk.evaluate (identical texts from the same keys) and rk.read reproduced exactly; guard refusals; windows; cbar against the
  pre-push norms; damage_T(T = 1) = rk.damage; WriteOrth leaves no component in span(U) at any block; leaf_grads = finite differences.
- `r2_axes.py ... selftest`: 11/11 (MOCK RSI texts identical to the MOCK round-1 texts, R1 reads and u_S / u_neg reproduced, axes by
  definition, resume, planted-signal block: disattenuated cos 0.65 with 90 % interval [0.60, 0.71] around the planted 0.60).
- `r2_gen_loop.py ... selftest` at blocks 14, 8 and 20: 9/9 each (gate closed and open, exactly the GEN-loop keys added, key namespace
  reproduces round 0, read-outs, E_GI / R_4 / S_4, resume).
- Extra: a block that ran at 1.5 alpha (GO1 retry; suffix '@1.5'): open_block, O3 bank and the GEN loop selftest 9/9; a missing b_hi:
  'not run (round-1 output missing)' in r2_axes and r2_gen_loop; r2_axes plus the block-14 GEN loop in one process.

## Open points

- MOCK cannot test GA = true naturally (the tiny random model gives AUC about 0.5); the GEN-loop tests force the gate open in copies of
  axes/r2_axes.pt.
- Production sizes: r2_axes.pt about 300 MB (five [2000, 2048] float32 null sets per block x 3 blocks, plus X28 float16), tok14.pt about
  65 MB, local/axes_reads.pt about 100 MB.
- Item 16 (u_RGI_perpT) and item 15 (bases fixed in the bootstrap) are readings; a final plan version may want them stated.
- For the runner: r2_gen_loop.py needs env BLOCK and axes/r2_axes.pt of the same r2_dir; r2_axes.py always opens block 14.
