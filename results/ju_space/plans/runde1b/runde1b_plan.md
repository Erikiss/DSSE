> **DRAFT (13.11.2026, before the adversarial review) - not frozen.** Assembled from four independently designed and merged parts; the review is running and will change this text. Nothing has been run.

# Round 1b: main plan (repair and replicate the RSI loop; H-DUAL; H-CHAIN; the lens-free arm) - text to be frozen with its four part files

Written 13 November 2026, after round 2's readout. The authors of this file and of the four part files have seen round-1 and round-2
outputs (results/ju_space/runde2_ergebnisse/AUSWERTUNG.md, round2_summary.json, theta_verdict.json, rgi_verdict.json, the four
readings and the skeptic's check). No round-1b output exists; nothing in the five files has been run. The text becomes final at the
freeze (section 9).

Sources: scratchpad r1b/brief_shared.md and design_rules.md; the four merged part specs and their meta files; the frozen round-2 plan
(results/ju_space/plans/abschnitt11_12_runde2_plan.md, sections 0, 1, 2.1, 2.6, 2.7, 6, 7, 14, the model for this file) and its
amendments file; AUSWERTUNG.md; results/ju_space/plans/runde1b_eingaben/README.md; analysis/ju_space/round2/r2_run_session.py and
notebooks/build_runde2_gpu_notebooks.py (sessions, gates, time guard, Drive sync, the two Google accounts).

## 0 How to read this plan

### 0.1 The five files and their precedence

| File (frozen path in results/ju_space/plans/runde1b/) | Content | Cited as |
|---|---|---|
| runde1b_plan.md (this file) | requests and their mapping, model and inputs, shared objects, statistics and round-level confirmatory families, the inverse-free rule, conflicts and their resolution, the gate index, budget and sessions, analyses, scripts, open decisions, dropped items, freeze | 'main §x' |
| runde1b_part_R.md | part R: repair and replicate the round-1 RSI loop | 'R.x'; measurements R0-R9 |
| runde1b_part_D.md | part D: H-DUAL, duality under hyper-long context | 'D.x'; measurements D0-D5 |
| runde1b_part_K.md | part K: H-CHAIN, the chain lens and the H-neuron question | 'K.x'; measurements K0-K10, RK-1, RK-2 |
| runde1b_part_L.md | part L: the lens-free, inverse-free arm (holographic, energy / Ising, non-equilibrium) | 'L.x'; measurements L0-L5, extensions X1-X6 |

- Measurement rules (What, How, Rule, Null / controls, Shows - cannot show, MDE, Inputs, Gates) live only in the part files. This
  file never restates them; it cites them by part and ID.
- Shared objects, session names, accounts, waves, tiers, key namespaces, global gates, round-level confirmatory families and budget
  figures are defined here, once. The part files refer to them by their main ID ('SO-...', 'SG...', 'r1b_...', 'CF-...').
- The part files were brought into line with this file by harmonisation edits before the freeze (renames, references to main §2.3,
  tier labels, key namespaces, statistics references, cross-part inputs, budget numbers; recorded in the freeze record, main §9).
  Where a part file and this file still disagree on one of the items in the previous bullet, this file governs; on a measurement's
  How or Rule, the part file governs.

### 0.2 Notation

- **IDs.** Every measurement ID carries its part letter. Across parts it is cited with the part ('part D's D4', 'D:D4'), because some
  parts use the same strings for other quantities inside their own text (R's statistic 'D4' in R2.4, R's denominators 'D_k' in
  R6.2, R's round-1 indicator names K1-s, K1-c, 'K2 halvings' and 'k5' (R.1, R2.4), K's length mechanisms L1-L3 and its 'L1 / L2
  logistic' probes (K.1, K2), L's reads R1-R3 (L.2) and D's reserves R1-R3 (D.19)). Shared objects are SO-*, global gates SG1-SG10, part gates keep their part names (main §3), sessions are r1b_<part>_<label>
  (main §4), round-level confirmatory families CF-* (main §2.4), cross-part open decisions OD-M* (main §7), conflicts C* (main §2.6).
- **Symbols.** Round-2 notation where it exists: alpha_b, cap_b, med_b, keyseed, G, K, rel, A, u_T_perp, damage in nats as
  teacher-forced KL(unmodified || modified). In addition:
  - u*_b = round 2's style mode at block b; AX1, U* and AX-1b are the three axis families of SO-AX (main §2.3);
  - delta1 = 0.05 nats (round-1 calibration level) and delta2 = 0.15 nats (the GO3 edge), the two damage levels (SO-DMG);
  - alpha_B = k/10 is part D's minority share, unrelated to the push size alpha_b;
  - 'K' always denotes a carry ratio of the round-2 R6 family and is cited with its part's estimator: R's K_2 and K_geo (R6.2), D's
    K(t) (D2-f), K's K_H (ratio of sums over turns, K6e / K8b), L's K_H and K_H^pool (L3b). They are never pooled or compared as one
    number;
  - H_tq, H_ar, H+, H_op are the H-neuron sets of SO-HN; P is part L's push / read set (main §2.3);
  - a_X(t) is the per-token H channel (SO-CHH).
- **Intervals.** Unless stated otherwise: a 90 % percentile bootstrap interval with 2,000 replicates over the part's units (main §2.4).
- **Hours.** A100-hours as planned: setup + I/O + 1.5 x work at the planned rates (main §4.1).

### 0.3 Tiers

- **Tier 1 (core).** Must run. Together the tier-1 measurements answer all four questions of main §1.
- **Tier 2 (extension or reserve).** Runs only if funded (the user's budget decision, main §4 and question Q1) and, for a conditional
  item, only when its trigger fires. A tier-2 item that did not run reads 'not run (tier 2 not funded)' or 'not run (conditional not
  triggered)'. No tier-1 rule needs a tier-2 output; where a part attaches a tier-2 result to a tier-1 verdict, it is a label only.
- The tier of every session and item is in main §4.2; the order in which tier-2 items are given up is in main §4.5.

### 0.4 Verdict and label conventions

- A verdict string is quoted exactly as frozen in its part and is reached only by that part's rule (read top-down where the part says
  'first match').
- Labels follow the string in parentheses. A label never changes a string. Labels are carried through every file that reports the
  verdict (R9.3, applied round-wide by SG6).
- Five outcomes are not verdicts and never count for or against a hypothesis:
  - 'not run (<reason>)': the measurement did not run (input missing, session failed twice, tier 2 not funded, conditional not
    triggered);
  - 'not testable (<reason>)': an instrument, a positive control or a prerequisite gate failed, so the question could not be asked
    (R's gated strings 'undecided (positive control failed)' and 'undecided (sensitivity check failed)' belong to this class);
  - 'not decidable (<reason>)': the question was asked but cannot be decided at the realised noise or definition (underpowered, rule
    not specific, ratio undefined);
  - 'undecided': the rule ran and no other string applied;
  - 'descriptive': the measurement has no rule.
- Round-level labels (main §2.4): 'confirmatory (<CF family>)' for a member of a round-level confirmatory family that its part's rule
  reaches and that survives the family's Holm step; 'exploratory (Holm not met)' for a member that does not; 'exploratory' for every
  other verdict.
- Missing values carry R9.2's codes in every part ('NaN:...', 'CENSORED:...', 'NA:gate_closed:<gate>', 'NA:not_run:<reason>',
  'NA:conditional_not_triggered'), with 'NA:not_run:tier2_unfunded' and 'NA:not_run:RL<k>' (R's reduction levels) added.

### 0.5 What 'post hoc' means here

A choice is post hoc when its author had seen data bearing on it before fixing it. In round 1b:
- every R0 re-reading of round-1 files: round-1 and round-2 results are known to the authors, so R0 is labelled 'post hoc (R0)' and is
  unadjusted (R.1);
- every amendment written after an output relevant to the changed rule had been seen (main §9.3);
- every analysis that the five frozen files do not specify.
Post-hoc results never enter a round-level confirmatory family; when an amendment is post hoc, the affected results are reported under
both the frozen and the amended reading. Design choices fixed before the freeze with knowledge of round-1/2 results (for example R's
verdict axes, chosen from round 2's R4, OD-R13) are declared here and are not post hoc with respect to round-1b data.

## 1 The user's requests and what answers them

German quotes are short excerpts of the user's message (speech-to-text, verbatim from the shared brief) or of the round-1b input
README. 'Haarneuronen' is read as 'H-Neuronen' (hallucination-associated neurons, Gao et al. 2025) and 'zu überwinden' as
'zu überprüfen'; both readings are put to the user for confirmation (question Q2).

| Request (German quote; English paraphrase) | Part | What is measured (IDs) | What is not claimed |
|---|---|---|---|
| **Round-1 repair.** 'Runde 1b wiederholt die RSI-Schleife von Runde 1 mit verbesserten Messungen' (README §0); AUSWERTUNG §5: 'Korrigierte Null zuerst', 'Schadensnormierter Schub', 'Lese- und Gebrauchsrichtung trennen', 'Linsen-Auslesung', 'Latente gegen Text-Schleife', 'Schadensarm entfernen', 'Kleinste erkennbare Wirkung vorab festlegen', '0.6B klären', 'Berichtsweg reparieren'. Repeat the round-1 RSI loop with repaired measurements and close the gaps round 2 left open | R | R0 re-analysis with centred nulls and a paired sign test (post hoc); R1 round-1-identical replay; R2 corrected loop at blocks 14, 18, 6 with PC_PASS_1b, SENS_1b and V_RSI; R3 damage-normalised and norm-preserving pushes, dose series 0.25-2 alpha_0 incl. below 0.5 alpha_0; R4 lambda_RSI against lambda_max (MicroLens++); R5 J-lens bridge, A3 replacement, MicroLens T4 / T3, verified label of u*_b; R6 text against latent loop, carry over turns 2-6, replication of round 2's T 1.0 cell; R7 low-damage erasure; R8 the 0.6B question; R9 report hygiene | No ability (Qwen edits nothing); no claim about other blocks (except R5's lens profile), long contexts or larger models; R tests neither H-DUAL nor H-CHAIN (R.1) |
| **H-DUAL, two entities.** 'aus einer Entität zwei zu machen, insofern ... Catfishing aktiv ist'. Making two entities out of one makes sense insofar as catfishing is active | D | D2 (one model voicing two speakers, opposed against neutral partner, about 300 against 32K tokens), D3 (seeds of one torn start committing to two states, 0 against 28,672 tokens), D4 (a presented role beside the retained assistant identity) | Entities exist only as defined by forced-choice readouts and states (D.1 mapping) |
| **Catfishing = hyper-long context; selfish mining.** 'Catfishing ist ... übersetzt auf LLMs eigentlich Hyperlong-Kontext'; 'Selfish Mining ... man hält den Kontext extrem hochlang über dem naturalen Maße'. Hold the context far above its natural length | D | Length by front padding (natural 4,096; hyper-long 16K-32K; no position beyond 32,100), D1 minority-capture surface, D1-b release arrangement, report share as hash share (D1) and production share (D5-r), share against count (D1-c), gamma's recency part (D1-b) and prior part (D1-e) | No withholding inside one forward pass; no difficulty adjustment, revenue or pools (D.1) |
| **A force splits the whole.** 'es gibt eine Kraft, oder es wird forciert, dass ich das Ganze teilt ... dass zwei Entitäten da sind'. A force arises, or is imposed, that splits one entity into two | D (R) | Arising: D3 (primary of part b1); imposed: D2 OPP. Short-context counterpart, descriptive: R2.4's 'oscillation (RSI begets anti-RSI)' and R3.3's odd / even dose parts (R.1) | A split in Qwen3-1.7B is no statement about larger models |
| **Catfish effect.** 'die dann wiederum ... bei längeren Kontext dieses Catfishing betreiben können'. The two entities then sustain longer context; a competitor keeps the group active | D (L) | D2-a / D2-b / D2-c (catfish contrast, length interaction, coexistence), D5-a (two-voice coherence lifetime), D5-r (rate form), D5-x (tier 2); part L's L4 catfish string (two separate agents) attached as a qualifier (main §2.6, C22) | No effort, motivation or 'being seen' (D.1) |
| **Losing game below 50 %.** 'das Optimum dann für alle unter 50% ... eher ein Losing Game ... Aber es ermöglicht diese Dualität'. A minority below 50 % can win; the whole loses; this enables the duality | D | D1-a (length-driven minority capture beyond position, degradation, prior x saturation, conflict hedging and filler form), D1-b (literal 'alpha* falls with L'), D1-d (system cost against a non-competing second entity); 'enables the duality' as a joint descriptive reading (D.17) | The Eyal-Sirer threshold is stationary; 'alpha* falls with L' is the user's own prediction (D.1) |
| **Form.** 'diese Erscheinung von also Form war ja anscheinend relevant'. Shape mattered in round 2 (toy AT8) | D (L) | D1-f (coherent against sentence-shuffled filler at fixed size); L3c's modularity Q next to ||J|| (descriptive) | No transfer of AT8's toy mechanism |
| **Training data as hyper-catfishing.** 'dass sowohl LLMs als auch die zugrunde liegenden Trainingsdaten als Hyper-Catfishing ... modelliert werden kann' | D | No counterpart in a frozen model (needs training). Proxies: D1-e (the trained prior as a hidden participant); D's reserve R1, the M2 reading (tier 2) | Nothing about training data |
| **Retokenisation aside.** 'ich kann retokenisieren, aber irgendwann erreiche ich dort auch eine Grenze ... Buchstaben oder ... Pixeln'. The duality must not live inside one token | D | D1-r (random segmentation and character-level ids at L0); D's reserve R3 (D1-r at L*, tier 2) | Pixels: no counterpart (text-only model) |
| **Analogy aside.** 'König und dann Mann plus Frau ergibt Königin ... sehr symmetrisch ... ziemlich asymmetrisch gestalten. Und was passiert dann, wenn ich tokenisiere?' | D | D1-v (3CosAdd forward and reverse; symmetric gender pairs against directional families; canonical, segmented, character-level; blocks 14 and 18) | Not a duality test (D.20: Reversal Curse) |
| **MTP and routing.** 'Multitoken Prediction lebt von einem Gradient, von den Gradient Through Routing ... dieses Durchrouten'. Multi-token prediction lives on gradients routed across positions | K | Qwen3-1.7B has no MTP head: self-rollout chain lens CL17 (K4), native MTP chain lens on Qwen3.5-2B for calibration (K5a), trained head (RK-1, tier 2). Inference counterpart of routing: the cross-position Jacobian split into re-routing (QK) and along-route (OV) parts with a head routing map (K10b) | No gradient flows at inference; training-time 'gradient routing' (Cloud et al.) has no counterpart (K.1) |
| **'Folge, nicht Reihe'.** 'im Sinne einer Folge, nicht einer Reihe'; 'der Token ist die Folge des vorherigen Tokens. Ich kann aber nicht sagen, der dritte Token ist die Folge von [dem ersten]' | K (L) | K10c (chained CL17 readout against direct L-par and L-jac readouts), K10a (KV share at lag 2; token against KV route); L3b's route line (KV against text) | A front reaches its own block at later positions only through tokens (structural code check, K.1 row 4) |
| **Fronts and channels.** 'die vorderste Front ... in der Zeit auftragen ... Reihen von Tokenfronten'; 'verschiedene Kanäle ... wo die Token durchfließen' | K | Front F(t) = the 28 block outputs at the newest position; channels SO-CH (448 heads, named subspaces, neuron groups, per-step lens channels); the front dump SO-FD | Every non-head channel is an analyst's basis choice within one residual stream (K.1 row 6) |
| **Hierarchy.** 'hierarchisch ordnen ... einzelne Kanäle beeinflussen wiederum andere ... äußere Gegebenheiten, welche die Hierarchie ... beeinflussen' | K | K9 (depth-residualised transfer hierarchy among 10 named channels; reordering across 5 text conditions), K7a outflow / inflow; directed couplings in L3c and L4 (descriptive) | Not the full 448-head hierarchy (K.9) |
| **Throughput and the vector.** 'Durchsatz ... Prädiktion der nächsten und der nächsten'; 'Prädiktor für vorhergegangene oder für folgende Fronten ... vektorielle Größe' | K (L) | K7a arrow-vector atlas and K7b channel-resolved lens throughput (both exploratory), K10a route horizons, K6b(iii) lead / lag; L3d's A_k (descriptive) | Same-channel forward / backward predictability is symmetric under stationarity, so it is a non-stationarity diagnostic only (K.1 row 10, L.6) |
| **The chain lens and its break-off.** 'Chain Lens, also ... der Kettenlinse ... konkretes ... Kriterium, wo sie ... halluziniert, oder wo der Blickbereich ... vor allem mit den Haarneuronen korreliert' | K (L) | Five break-off criteria per chain (K4); per-step confabulation and hazard (K6a); lens offsets and content (K6f, K6g); QA lens arm (K6h); the lens reading the model's own wrong intermediates (K6d); L5 (does the break-off and its H-pattern belong to the model or to the lens?) | Readouts are fragments, not the model's reasoning; the native lens exists only on Qwen3.5-2B |
| **H-neurons.** 'Haarneuronen [H-Neuronen] ..., die ... nur 0,1% circa im Netz ausmachen' | K (L, D) | K2 (H_tq by Gao's protocol, H_ar, H+, diagnostics), K3 (detection, before / during generation, causal role, neuron-lens readouts); L1 (where the hallucination signal sits); D's optional H covariate (descriptive) | The set is not unique (bootstrap Jaccard; H+) |
| **Length artifact or H-neuron attractor; amplifiers.** 'einfach nur ein Artefakt der Länge'; 'andere Mechanismen im Netz, welche ... diesen Effekt nochmal verstärken'; 'Haarneuronenattraktor, der nicht unmittelbar an die Statistik der Länge gebunden ist' | K, L | Length: K6a, K6b, K6f, K8a, K8c; L2b (length against information along the model's own chains). Attractor: K6e (in the chain lens), K8b (own text, a new question per turn), L3a / L3b (own text, continuation turns), X1 (tier 2). Amplifiers: K3d, K6c, K6e-c, K6g, K8a; L3b 'H-specific carry' with K < 1. H before the model's own errors: K8c, L5c | An attractor needs persistence after release (an R6 carry with lower bound > 1, or within-item bistability) specific to H; restoring after a release counts for neither (K.0, L3) |
| **Lens-free extension.** 'Könnte man die erhofften Erkenntnisse bezüglich H-Neuronen Chain Lense ... auch durch einen linsenfreien Ansatz erweitern' | L | No learned or transported readout anywhere in L (L.0). L5 (ownership), L2b, L3, L5c (attractor against length without a lens); combined readings L-ALL and V-OWN (L.5) | L runs no lens; it classifies K's chain records as data (L5) |
| **Holographic principles.** '... indem man auf holographische Prinzipien setzt?' | L | Four senses, never merged (V-HOLO): code (L1: PIP, threshold, synergy, monogamy), RT / area law (L2a: growth class of captured MI up to 32,768 tokens; own-text deficit), reconstruction (L1(iv) Lambda, L2c cross-map), associative memory (L1(vii) erasure, L3, X4) | No AdS bulk, gravity, quantum entanglement, no-cloning, Page curve or scrambling (L.6) |
| **EBM reasoner, non-equilibrium.** 'vgl. EBM reasoner, non-equilibrium' | L | L3d (energy-well rule at H entry with E = -log q; irreversibility against a detailed-balance null); kinetic-J asymmetry (L3c, descriptive) | Qwen never descends an energy; no heat bath or physical temperature |
| **The chain loop -> agent -> Ising -> collaboration -> entity.** 'da Loop Transformer --> Agent --> Ising modellierbar --> Kollaboration --> entitätenbildend ....' | L | Link by link (V-CHAIN): loops (token, turn, dialogue; X3 toy depth loop, tier 2), agent (L4, G-duet), Ising-modellable (L3c units; L4 agents), collaboration (L4: coupling beyond lexical priming), entity-forming (L4: consensus beyond within-item re-pairing); H gain as coupling (L4; X2, tier 2) | Two agents give no phase transition; agents are model instances, not units inside one forward pass |
| **Inverse-free reconstruction.** 'Wird etwas rückkonstruiert, dann inversfrei wie bei DeepMinds Aletheia (Erzeugen, vorwärts Prüfen, Überarbeiten oder Enthalten), nicht über invertierte Matrizen' (README §6; binding) | all | Every reconstruction uses SO-GVRA (main §2.5): R5.5, D3's mode naming and D's INJ starts, K3e, K4 / K7b, L's X5 | Forward regressions, fitted forward maps and power iterations are not reconstructions; they are allowed and named (main §2.5) |

**Context (rounds 1 and 2).**
1. Round 1 (frozen plan abschnitt11_rsi_kennwerte_plan.md) ran RSI steering loops at blocks 14, 18 and 6: push s x alpha x min(|h_t|, cap)
   x unit(d) at block b, generate +/- texts, read the unsteered model's block-b mean state, d_(n+1) = mean (x+ - x-)/2; indicators K1-K8
   against the RSI axis u_T_perp; a sea-topic loop SEA was the positive control (PC_PASS).
2. Round 2 (frozen plan abschnitt11_12_runde2_plan.md; nine A100 sessions, code commit 8171b86) found PC_PASS failed at all three
   blocks, GO3 failed at 14 and GO1 at 6: no loop-dynamics verdict, every Qwen result at block 14 descriptive.
3. Part B: every loop gain G 0.12-0.30 (upper bounds <= 0.3432) at pushes >= 0.5 alpha_0, carry K 0.45-0.48 per answer at block 14,
   but along the generic style direction u*_b onto which all loops converge, not along RSI (an RSI push returns 2-5 %); the RSI-axis
   K(T 1.0) = 1.053 [0.743, 1.538] and the small-push linear response 0.7298 [0.1139, 1.356] leave self-sustainment open.
4. Part A: 'improve an AI system' is one shared direction at blocks 12-27 but barely more general than 'change'; the RSI direction
   does not move the propensity to say 'improve'; the A3 gradient readout vanishes near the output; toy AT8: 'the shape of the space
   matters, not its size'; toy B-T5: 21 of 30 cells bistable.
5. Part C: RSI sits mostly in one direction; erasing it leaves 16 % at 0.11 nats ('not unlearned'); a counter-push is no better than
   random; three in-prompt examples restore 89 %. Round 1b repairs the loop (R), tests the user's two interpretations (D, K) and
   answers the lens-free question (L).

## 2 Shared part

### 2.1 Model, revision, dtype, template, units

- **Model.** Qwen3-1.7B, revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e; shard and tokenizer sha256 as in
  analysis/ju_space/gpu/data/checksums.json (file sha256 152fd2ba24157fcc6deb875e99b7e86d3e6a7272f1a4cb75d2d83152caf3b8e9;
  tokenizer.json aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4). 28 blocks, hidden size 2,048, 172,032 MLP
  neurons. No training or fine-tuning of Qwen anywhere.
- **Second models.**
  - Qwen3-0.6B (revision c1899de289a04d12100db370d81485cdf75e47ca; model.safetensors
    f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b): D1-s (tier 1), R8.1 (a round-2 file only), R8.2 (tier 2).
  - Qwen3.5-2B under transformers 5.22.2 (revision and shard sha256 written into the freeze record): K5a (tier 1) and RK-2 (tier 2).
  - The only training is RK-1's auxiliary MTP head on frozen Qwen3-1.7B (tier 2, bf16; K).
- **Template.** Chat template, thinking off (the empty think block, 4 tokens), as in rounds 1 and 2. Plain-text frames only where a
  part states them: K's lens frames (K3g, K4, K4j, K5a, K6f, K6g), K's reference and filler texts (REF-W, N-text, F512, F2048),
  L2a's raw documents, teacher-forced reference texts. History conventions are per measurement and stated there (B-Q8 concatenation
  with history think blocks kept in R6, D2, K8b, L3b, L4; the official template with history think blocks dropped in K8a and D's
  reserve R2's official-template arm).
- **Numerics.** float32 weights and compute; cuBLAS TF32 off; torch.set_float32_matmul_precision('highest'). Exceptions: R5.4's
  J-lens refit and RK-1's head training (bf16; both tier 2). Parts D, K and L assert the SDPA memory-efficient kernel (a math
  fallback raises; on an A100 in fp32 this kernel computes in 3xTF32 'FastF32' whatever the TF32 flag says). R's rk_loop paths keep
  rk_loop's attention as in rounds 1 and 2 (replay identity).
- **Pairing.** Every paired contrast is computed within one session, on one kernel path, one batch shape and one GPU model (L.2's rule,
  round-wide). No part compares numbers across GPUs.
- **Lengths.** No position beyond 32,768 (the native window); no YaRN. Part D uses at most 32,100; part K at most 4,096 (plus K's
  global-offset check, a label only); part L's L2a up to 32,768 (subject to OD-M14).
- **Sampling.** Gumbel-max over the full vocabulary with keys from SO-KEYS, at the stated T. Exception: K2's Gao protocol truncates
  to top-k 50 and top-p 0.9 before the keyed draw (T 1.0).
- **Units.** Log-probabilities and damage in nats; damage = teacher-forced KL(unmodified || modified), mean over the scored positions
  of the reference texts (SO-DMG); lengths in tokens of the pinned tokenizer; compute in A100-minutes.
- **Push notation.** R's push s x a x min(|h_t|, cap_b) x unit(d) at the output of block b from the first user-content token on (R.1);
  alpha_0(b) = alpha_b = R1.loops[b]['_meta']['alpha'] (0.15 at 14, i.e. 1.5 x the calibrated 0.1; 0.10850 at 18; 0.13444 at 6);
  cap_b = 2 med_b. K's and L's neuron interventions (NSCALE_DN, mean-clamp, expert-stream push, population gain, partial mean
  ablation) are defined in their parts; every calibrated intervention of R, K and L is calibrated by SO-DMG's rule.

### 2.2 Round-1 and round-2 inputs

**Accounts and Drive locations.**
- **Account A** ('hub', as in round 2): My Drive/dsse_runs/ holds the round-1 Qwen shards gpu1-gpu3 (rk loops at blocks 14, 18, 6),
  the round-2 sessions r2_q1, r2_q2, r2_q3, r2_pc, r2_ana, and the round-1b sessions that run in A (dsse_runs/r1b_<...>).
- **Account B:** dsse_runs/ holds the round-1 toy shards gpu4-gpu6, the round-2 toy sessions r2_t1-r2_t4, and the round-1b sessions
  that run in B. No round-1b session reads gpu4-gpu6 or r2_t1-r2_t4.
- **B -> A** (as in round 2): B shares its dsse_runs with A (read access); in A it is the shortcut My Drive/dsse_runs_B. An A session
  that needs a B session's outputs copies the folder dsse_runs_B/r1b_<s> into A's dsse_runs only when r1b_<s>/session_done.json
  exists and no session_partial.json lies next to it (round 2's 'second folder' rule).
- **A -> B** (new, SO-XFER): A shares only the folder dsse_runs/r1b_xfer with B (read access); in B it is the shortcut
  My Drive/r1b_xfer_A. It carries only r1b_R_S0's exports (dmg_ref_R16.json: the token ids of REF-R16; so_ax_ustar.pt: u*_b at 14, 18
  and 6 with the sha256 of their round-2 sources); since C32 no tier-2 trigger crosses from A to B. No round-1 or round-2 file
  leaves account A.
- Every copied file's path, source session (or shard), sha256 and Drive modification time go into the session manifest (SG4).

**Round-1 inputs** (account A; found by name across gpu*/, round 2's finder: block 14 on gpu1, 18 on gpu2, 6 on gpu3; key names as in
the round-2 plan 2.1 and R0's input list):

| Symbol | File (relative to dsse_runs/) | Keys | Read by (session) |
|---|---|---|---|
| R1.state[b], b in {14, 18, 6} | gpu*/rk/b<b>/state.pt | 'ref' (X, conts, med, cap) | R0, R1, R2 (cap_b, med_b), R3.1 (REF-R16), R5.1 / R5.3 (cap_b, med_b; dmg_sym), R6.1 (PUSH-DN), R7 (r1b_R_S0, r1b_R_S1, r1b_R_S2_*, r1b_R_S3, r1b_R_S5, r1b_R_S6, r1b_R_S7); SO-DMG REF-R16 export (r1b_R_S0) |
| | | 'axis_rsi', 'axis_sports', 'axis_tone', 'axis_sea' (Xp, Xa, conts) | R0, R1, R2.1 (round-1 keys' token identity) |
| | | 'axes' (u_T, u_T_perp, u_S, u_S_perp, u_neg, sports, tone, sea, the four null sets, auc, paired_win, gap_cross, gap_perp_cross, kept, fold) | R0, R1, R2.1 (u_neg, fold), R6.3, R7; SO-AX AX1 (r1b_R_S0) |
| | | 'alpha', 'ev_<L><SUF>_<n>' (n 0-3), 'dir_<L><SUF>_<n>' (n 1-4), 'dmg_<L><SUF>_<n>', 'checks', 'k5', 'k6_null<i>' (block 14) | R0, R1 (open loop along dir n = 1-3), R2.4 ('k5'), R7 ('k6_null' via rk_dirs.pt); SO-AX leave-loop-out modes and the u*_b rebuild check (dir_<L><SUF>_4; r1b_R_S0) |
| R1.loops[b] | gpu*/rk/b<b>/loops.json | rows (incl. 'sea', 'sports_own'); '_meta' (alpha, alpha_scale, suffix, block) | R (alpha_0(b), SUF_b), R0, R1 |
| R1.pilot[b] | gpu*/rk/b<b>/pilot.json, pilot@1.5.json | 'SEA_1'[0]['sea'], 'gates', 'axes' | R0 |
| R1.summary[b] | gpu*/summary_b<b>.json | 'verdict' (PC_PASS, SEA_S4, gates.source, gates.SEA_1, gates.failed, verdict, label_attractor) | R0 |
| R1.delta | gpu2/dn_delta.json | blocks[l].last.delta | R0, R2.5 |
| R1.meta | gpu*/shard_done.json, gpu*/code_commits.txt | GPU, torch, transformers, CUDA, code commit | SG3 (round-1 module hashes at each shard's commit), provenance |
| package | analysis/ju_space/gpu/data/rk_dirs.pt (repository) | pool directions | R2, R3.3, R7 |

**Round-2 inputs** (account A, or the repository copies, which are byte-identical to the Drive files):

| Symbol | File | Keys | Read by (session) |
|---|---|---|---|
| U*: u*_b | dsse_runs/r2_q2/theta/b<b>_q0.pt, b in {14, 18, 6} | 'u_star' ('uA', 'uB', 'd4', 'cfg' for R0) | SO-AX (r1b_R_S0 copies them with sha256 and exports u*_b to r1b_xfer) |
| | dsse_runs/r2_q2/theta/b<b>_q0.json | mode_set (members, signs), m_star2 | R0; SG10 |
| r2RSIb | dsse_runs/r2_q2/theta/b18_r2RSIb.pt, b6_r2RSIb.pt | 'entries' (ev_r2RSIb_n, dir_r2RSIb_n) | R0 leave-loop-out modes; SG10 rebuild at 18 and 6 |
| TV | results/ju_space/runde2_ergebnisse/theta_verdict.json (sha256 628d6d45b2fd7b85c2b09ad72a2f58239996f024e7508681442fb18acb83f481; = dsse_runs/r2_ana/theta_verdict.json) | blocks.<b>.B_Q0 (mode_set, image_cos_d1_u_star, persistence), blocks.<b>.R4, blocks.14.R1.points, blocks.14.R6.cells, B_Q7, cells | R (OD-R13 verdict axes, R6.3 target), SG10, D2-f (r1b_D_ana) |
| S | results/ju_space/runde2_ergebnisse/round2_summary.json (1af16af8003be7a0a19bdf379c5b6e060128ffb3b7c69a5e5a1414b22d45d06a) | gates G3, G47 | R |
| geom 0.6B | dsse_runs/r2_q3/rgi/geom_stats_qwen3-0.6b.json | gate.*, per_block.pooled | R8.1 (r1b_R_S0) |
| Drive-only | dsse_runs/r2_ana/unlearn_verdict.json, dsse_runs/r2_ana/r2_manifest.json | copied byte-identically into results/ju_space/runde2_ergebnisse/ | R9.5 (r1b_R_S0) |
| Round-2 code | analysis/ju_space/round2/*.py at commit 8171b86 (unchanged at HEAD) | r2_rk, r2_stimuli, r2_theta, r2_theta_verdict, r2_unlearn_qwen, r2_linearity (sha256 in main §6.1) | all parts (SO-TURN, SO-DMG's calibrate and RefKL, r2_rk.open_block, prompt_bootstrap, CbarHook) |

Every session that reads a round-1 or round-2 Drive file runs in account A. Parts D, K and L read no round-1 or round-2 file directly,
except r1b_D_ana's read of TV's R6 cells for D2-f (table above); they receive everything else through SO-AX, SO-DMG and SO-TURN.

### 2.3 Shared objects (each defined once)

The table defines every object used by more than one part. 'Owner' writes it (and its code); the users read it; the last column lists
the names the parts use for it, which the harmonisation edits replace by the main ID. Objects used by one part only stay in that part
(list after the table).

| ID | Definition | Owner (made in) | Users | Local names replaced |
|---|---|---|---|---|
| **SO-AX** | Three axis families. **AX1:** the round-1 axes u_T_perp(b), u_sea(b), u_sports(b), u_tone(b), u_neg(b) at b in {14, 18, 6} from R1.state[b]['axes']. **U\*:** u*_b at b in {14, 18, 6} = the round-2 file r2_q2/theta/b<b>_q0.pt['u_star'], copied unchanged with its sha256 (canonical); the leave-loop-out modes u*_b^(-L) (R.1). **AX-1b:** R's 1b axes (u_T, raw and perp* and ^1b versions, the verdict axes u_X^v(b), fold axes, kappa_X, centred nulls, AUCs) at 14, 18, 6 and at R5.1's lens blocks {2, 6, 10, 14, 18, 22, 24, 26} | R (AX1, U* and the SG10 rebuild check in r1b_R_S0; AX-1b in r1b_R_S2_14 / _18 / _06 and r1b_R_S5) | R (all); K (K3f: u*_14; K7b, K9, K10a and SO-CH named channels: u*_b and AX-1b's u_T^v(14), u_sea^v(14), u_T^v(18), and u_sea^v(18) for K7b's block-18 cells, OD-M5); D (D2-f: u_T_perp(14), u*_14 and u_T^v(14), OD-M6); L (L3b descriptive line: u*_14) | R 'AX-1b(b)', 'u*_b, u*_b^(-L)'; K 'SO-AX', 'KS-axes'; D 'Axes'; L 'R.AXES' |
| **SO-DMG** | Damage and its calibration. Damage functional: teacher-forced KL(unmodified || modified) in nats over the scored positions of a reference set, logits_to_keep on scored positions. Calibrate rule: f <- f sqrt(target / damage), band +-10 % of the target, at most 6 evaluations for pushes and 8 for neuron scaling (K's NSCALE_DN: (alpha - 1) <- (alpha - 1) sqrt(target / damage), alpha <= 20, 'dose capped' label); the closest value tried is kept and labelled when out of band. Reference sets: **REF-R16** = R1.state[14]['ref'] continuations of round-1 prompts 0-7 (RSI-topic) and 32-39 (scene) (r1b_R_S0 asserts equality with the copies at 18 and 6 and logs differences); **REF-K16** = K's DMG_K (8 WikiText-103 validation passages with 64 scored tokens, 8 TriviaQA S2 question / answer pairs), frozen as token ids by K1 before the freeze; **REF-U32** = REF-R16 union REF-K16. Levels delta1 = 0.05 and delta2 = 0.15 nats. Push operators: PUSH-DN and PUSH-NP (R3.1, R3.2); K's and L's neuron operators in their parts. **Calibration reference per part, as frozen:** R on REF-R16 (dmg_sym, both signs, R3.1); K on REF-K16; L on REF-U32 plus 64 held-out S_L answers (L.2). **Reporting:** every calibrated intervention of R, K and L also reports its damage on REF-U32 (descriptive), so that damage is comparable across parts | R (functional, rule, REF-R16, PUSH-DN, PUSH-NP; function-style implementation in r1b_core for K and L); K (REF-K16, NSCALE_DN) | R, K, L (D runs no push) | R 'PUSH-DN', 'PUSH-NP', 'dmg_sym', '16 references'; K 'SO-DMG', 'DMG_K', 'D1', 'D2' (as damage levels); L 'R.DMGPUSH', 'R's REF16 / K's REF-K16 union' |
| **SO-JLENS** | The Neuronpedia J-lens for Qwen3-1.7B (qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt; 27 fp16 2,048 x 2,048 maps, blocks 0-26 to block 27; sha256 6fcc79011bd921ffd87612255e2e99950a124fa519470ee44ebaf161c39be9d6); its index convention (J[l] maps the output or the input of block l), decided by K4j (G-K-J) in r1b_K_S1; its per-block validity and layer profile (R5.1, RG10); LENS-J1b (R5.4's refit, tier 2, used by R only) | R (file, validity, LENS-J1b); K (convention) | R (R5.1-R5.5), K (K4 seeds, K4j, K10c's L-jac) | R 'LENS-J'; K 'SO-LENS (J-lenses)' |
| **SO-CL** | The chain-lens instruments: CL17 (K4: self-rollout from J-lens seeds, contrastive decoding against mu_task, chain-only context, re-injection off in primaries), CL35 (K5a native MTP lens on Qwen3.5-2B), the verifier, frames and fillers (P0 16 tokens, F512, F2048), priors mu_task and mu_gen, L-par and L-jac | K | K only (L consumes SO-LENSOUT) | K 'SO-LENS'; L 'K's lens' |
| **SO-LENSOUT** | K4's per-chain records, one per chain: item id, stratum (K4 'correct' = L's 'known'; K4 'wrong' = L's 'prone'), number of operations (two or three), variant, read block, position type, seed rank, key, emitted token ids at steps s = 0..16, k*_conf (first CONFAB), share_H(s) with the RS39 band (q95 of share_R), a_H(s); plus K4's gating answers for every gated A-lens pool item (greedy and 4 samples at T 0.7) | K (r1b_K_S2a) | L (L5a, L5b; L5's own-CoT items) | L 'K.LENS-OUT' (B_hal -> k*_conf; FOV_H(q) -> share_H(s); A_H(q) -> a_H(s)), 'K.ITEMS-ARITH' |
| **SO-HN** | The H-neuron sets and their machinery (K2): H_tq (Gao's protocol, prose label convention, positive weights at C*), H_ar (arithmetic), H+ (H plus every unselected neuron with |r| >= 0.7 to a member), F (negative weights, contrast only); **H_op**, the operative set = H_tq, or H+ after G-K22's switch, or H_ar after G-K1's fallback ('H_ar primary'); theta, C*, refit L2 detectors; the 20 question-bootstrap refit sets of diagnostic Dg2 (positive weights at C*); REF-W per-neuron mean and SD of z and CETT and the 99.8th percentiles; random sets RS39, RS39mc, RS19, RS20, RS10; the down_proj hook manager (z, CETT, per-position scaling, mean-clamp), the GPU FISTA selector select(X, y, C*, exclude_mask) and the random-set generator (same count per layer, mean answer-CETT within +-20 %) as importable functions. Files hn_set.json (incl. the boot sets and H+ ordered by selection frequency, then |r|), hn_detectors.pt, rs_sets.json, hn_meta.json | K (r1b_K_S1) | K; L (P, L.POP, CTRL99, SEL+, UNC+, the selection-robustness label, CETT recompute, mean ablation); D (optional descriptive covariate, OD-M25) | L 'K.HSET', 'K.HSET+', 'K.HBOOT', 'K.CETT / K.NH', 'K.SELECT', 'K.RANDSETS', 'K.REF'; D 'the H-neuron set (part K)' |
| **SO-CHH** | The per-token H channel: a_X(t) = mean over the neurons of X of zCETT (CETT standardised per neuron on REF-W); s_H and H~ as K defines them (K.0) | K | K; L (primary a_t; L's standardised-z mean is its sensitivity variant, OD-M11); D (covariate) | L 'K.CH-H' |
| **SO-LAB** | Labels and items (K1, K2, K4): ACH-short and ACH-long; FP, MC, SY; NE (300 fabricated-entity items = L's FAB); TriviaQA rc.nocontext (pinned) split by question-id hash into the S1 screening pool (allocated in K2 to S_fit, S_lens, S_sess and **S_L**, OD-M8), S2, S3, S4a-c; the arithmetic splits; the scorers (numeral parser, alias matcher, refusal lexicon, compliance parsers, W enumerator, OWN sets, CONFAB-QA parser with audit, loop-onset detector); per item the 10 sampled answers with keys, the consistency class (CC = 10 / 10 correct; CI = 0 / 10 correct and no refusal) and the answer spans | K (K1 before the freeze; K2 and K4 in r1b_K_S1 and r1b_K_S2a) | K; L (S_fit for selection, S_L for evaluation, NE, the scorers, K4's gated items) | L 'K.LABELS' ('S1' -> S_fit, 'S2' -> S_L, 'FAB' -> NE), 'K.ITEMS-ARITH' |
| **SO-LCE** | The long-context engine: B = 1 shared-prefix prefill per length and stream; chunked extension with logits_to_keep; explicit position_ids and cache_position; the crop path (unstopped chunk, all layers cropped; D), per-layer crop of layers <= b after a stopped read (K, and R's rule for any prefix reuse), the copy path (no crop); branching by cache copies (batch_repeat_interleave); keyed Gumbel decode (SO-KEYS); generation-time capture; the SDPA memory-efficient kernel asserted; manifest of peak memory, backend, GPU and versions. Parity is gated per use, each with its frozen tolerance: D's G-LCE-1..4, K's G-K20, L's G-L2a-LCE (OD-M14) | D | D; K (<= 4,096 tokens: K6e-d, K6f, K8a); L (L2a at up to 32,768) | D 'LCE'; K 'SO-LCE'; L 'D.LONGCTX' |
| **SO-D1STIM** | D's frozen D1 stimulus file d1_stimuli.json (F-sym and F-asym referents, six attribute slots, versions 1 and 2, the third value v3, the second referent Y_i, report templates, questions). **F-sym items 0-31 are L4's fork items:** entity = referent i, contested attribute = slot j = i mod 6, v1 / v2 = that slot's version-1 / version-2 values | D (before the freeze) | D (D1); L (L4) | L 'D.FORK-F1' |
| **SO-WT103** | WikiText-103 raw at one pinned revision (sha256 in the freeze record), CC BY-SA 3.0, split by use: **train** (file order): D's 16 filler streams first (as D's frozen script builds them; the shuffled versions reuse their text), then K's 1M-token percentile reference, then SO-PILOT's 8 passages, then (tier 2) RK-1's 20M training tokens; **validation:** part K only (REF-W, N-text, F512 / F2048, REF-K16's passages, K4j's passages; K fixes disjoint offsets); **test:** part L only (the 200 natural-position passages of L.R) | D (pins), K and L (their offsets) | D, K, L, P0 | D 'generic filler corpus'; K 'WikiText-103 validation'; L 'WikiText-103 validation passages (disjoint from K.REF's)' |
| **SO-LONGDOC** | L.LONGDOC: PG19 test split at a pinned revision, the first 24 books in file order with >= 36,000 tokens after skipping 2,000, slice [2,000, 34,768) | L | L (L2a); P0 uses qualifying books 25-28 (the next four in the same order) for timing only | L 'D.LONGCORPUS / L.LONGDOC' |
| **SO-TURN** | Round-2 turn machinery, imported unchanged: r2_stimuli.TURN2_SUFFIX, turn2_ids, check_turn2, round_trip_stable (OD-B13 concatenation, history think blocks kept); the B-Q8 read convention ('read as the reply to the plain prompt'); r2_theta.stage_turn2; the R6 rule logic of r2_theta_verdict (K defined only if the denominator's 90 % interval excludes 0) | main (round-2 code at 8171b86) | R (R6.2, R6.3), D (D2, D2-f), K (K8b's history convention, G-K8), L (L3b, L4) | various |
| **SO-GVRA** | The inverse-free generate-verify-revise-abstain convention (main §2.5): forward generator; verifier = a forward round trip through the unmodified model, decoupled from the generator, with three outcomes (correct / minor fix needed / critically flawed); reviser; attempt limit 3; abstentions counted and reported; the verifier's error rate measured on known cases before its outputs are read | K (r1b_gvra.py) | R (R5.5), D (D3 mode naming, INJ starts), K (K3e, K4, K7b), L (X5) | K 'SO-GVRA'; L 'K.GVRA' |
| **SO-PVAL** | Round-wide interval, p-value and multiplicity utilities (main §2.4): percentile bootstrap (one- and two-level), one-sided bootstrap p = (1 + #replicates on the null side of the threshold) / 2,001, rank p = (1 + #{controls >= statistic}) / (N + 1), K's prediction-interval p, Holm, BH; codes for undefined replicates | main (r1b_common.py) | all | various |
| **SO-KEYS** | The key-namespace registry (below) and a verbatim copy of rk_loop's keyseed and Gumbel with a sha256 check of the source lines against rk_loop.py (D's method; rk_loop is not imported where its model is not wanted) | main (r1b_common.py) | all | various |
| **SO-PILOT** | The pilot materials of r1b_P0 (below) | main | P0 | R 'throwaway prompts outside P256'; L 'rk_loop scene prompts' |
| **SO-XFER** | The cross-account folders dsse_runs_B (B -> A) and r1b_xfer (A -> B), main §2.2 | main | all cross-account starts | - |

**Objects that stay in one part** (not shared; defined in their parts): R's P256, WORDSETS, TXT0, DOSE-1b, JLOOP, SENS-1b, ERASE-LD,
LABEL-u*, R0-table, STAT-1b's R-specific statistics (centred pseudo-axes, disattenuated K1-s and A, SENS plants, joint-MDE simulator,
cross-fitted Ritz pairs); D's PROBE, yes / no stance probe, INJ, SPLIT-RULE, u_AA, d2_prompts.json, d3_items.json, d4_roles.json,
d1v_pairs.json, the RSP pool, the 16 filler streams; K's SO-CH (except SO-CHH), SO-GEN and SO-FD (except SO-LENSOUT), SO-TE, SO-CAL,
SO-STAT (K.4's conventions);
L's L.POP, L.R, L.PIP, L.EQMI, L.TWIN, L.MARKOVSUR, L.KISING, L.DUET, L.OWN, l_verdicts.json, F-neu.

**SO-KEYS registry.** keyseed(*k) = int(sha256('|'.join(map(str, k)))[:12], 16) (rk_loop.py:122). Two keys can coincide only if their
joined strings coincide; no tag contains '|', so keys with different first arguments never coincide. Every key's first argument is a
registered tag:

| Owner | Prefix | Tags |
|---|---|---|
| R | 'r1b_', 'r1b06_' | 'r1b_axis', 'r1b_' + loop name, 'r1b_dose', 'r1b_free', 'r1b_fdtchk', 'r1b_lab', 'r1b_lensfit', 'r1b_k1', 'r1b_turn', 'r1b_turn_rnd', 'r1b_krep', 'r1b_krep2', 'r1b_Q4B', 'r1b06_' + loop name, 'r1b06_axis', 'r1b06_ref' |
| R (stated reuse of round-1 keys) | - | R1 replay: round-1 keys keyseed(L + SUF, n, p, s, b) and 'mirror'; R2.1: keyseed('axis', name, p) for the 64 round-1 prompts' axis texts; R7: ablated texts reuse their references' keys (common random numbers) |
| D | 'r1bD' | 'r1bD1', 'r1bD1r', 'r1bD1v', 'r1bD2', 'r1bD3', 'r1bD4', 'r1bD4a', 'r1bD5', 'r1bD_INJ' (was 'r1bINJ'), 'r1bD_pool', 'r1bD_shuf' (was 'r1bshuf') |
| K | 'r1bK_' | 'r1bK_' + tag (was 'k1bK_'), tags hn, s3, gate, K4T, K6e, ses, long, bnd, K10a, K3f, V7, cal |
| L | 'r1bL' | 'r1bL0', 'r1bL2', 'r1bL2b', 'r1bL2s', 'r1bL3', 'r1bL3t', 'r1bL4', 'r1bL5', 'r1bLX'; string seeds 'r1bL1rot', 'r1bL-ERA', 'r1bL2perm', 'r1bL-CTRL', 'r1bL5-FM', 'r1bL-BOOT', 'r1bL-SUR', 'r1bL-SHUF' |
| main | 'r1bP0_' | pilot texts |

- r1b_common.KEYS lists every tag. A code test asserts that the tags are unique across parts, differ from every round-1 and round-2
  tag ('ref', 'axis', 'mirror', the round-1 loop names with their SUF, 'r2GEN...', 'theta', 'theta2', 'r2PWR', 'r2fdt', 'r2CNT',
  'r2RSIb...', 'r2Q4B', 'r2ARMA', 'r2SUP', 'r2DOSE', 'r2CHK') except for R's stated reuse, and that no R loop name forms a tag equal to
  another R tag.
- Keys shared across conditions inside a part are that part's stated common-random-number exceptions. No key is shared across parts.
- Numeric RNG seeds (numpy / torch generators for bootstraps, permutations, item generators, random sets) may coincide across parts:
  they never seed a Gumbel key and draw unrelated objects.

**SO-PILOT.** The pilot r1b_P0 uses only material that appears in no analysis set:
- 32 pilot prompts (new everyday scenes written for the pilot; disjoint from P256, F-neu, D's stimuli, K's items and every TriviaQA
  split) for decode, read, RefKL and hook timing;
- 8 WikiText-103 train passages outside D's streams (SO-WT103) for K0's code checks and the J-lens path;
- PG19 test books 25-28 of SO-LONGDOC's ordered list of qualifying books (the next four after its 24) for 16K and 32K prefill timing
  and parity metrics;
- keys 'r1bP0_*'. Every generated text is discarded. Only whitelisted engineering keys are written: rates, memory, throughput, parity
  metrics on pilot texts, code-check pass / fail.

**Prompt and item sets.**

| Set | Owner | Content | Allowed overlap |
|---|---|---|---|
| P256 | R | 64 round-1 prompts + 192 new (results/ju_space/plans/runde1b/prompts_1b.json) | P256's round-1 scene prompts (indices 128-159) are rk_loop's 32 scene prompts, which L's F-neu also uses; the round-1 prompts (indices 0-31, 128-159) also seed D4's u_AA texts, D5's 32 starts and RSP pool (indices 0-31) and 32 of K10a's texts (other templates, parts and keys) |
| D's stimuli | D | d1_stimuli.json (SO-D1STIM), d2_prompts.json (160), d3_items.json, d4_roles.json, d1v_pairs.json, the RSP pool, 16 filler streams | F-sym items 0-31 are L4's items |
| SO-LAB | K | ACH-short / -long, FP, MC, SY, NE, TriviaQA S1 (S_fit, S_lens, S_sess, S_L), S2, S3, S4a-c, arithmetic splits | S_L, S_fit, NE and K4's gated items are also used by L |
| L's sets | L | F-neu (64 long-form prompts), L3b's 128 S_L CC + 128 NE prompts, L4's items (SO-D1STIM), SO-LONGDOC, the 200 natural-position passages (SO-WT103 test) | as listed |
| SO-PILOT | main | above | none |

### 2.4 Statistics conventions per part and round-level confirmatory families

The parts keep their own conventions (they answer different questions, as in round 2, 2.6). They agree on the core:

| | R (R.1) | D (D.3) | K (K.4) | L (L.2) |
|---|---|---|---|---|
| Units | prompts (P256; 128 RSI-topic + 128 scene) | items (D1; two-level: streams, then items), prompts (D2; two-level), items or roles (D3, D4), starts (D5) | items (K3-K6, K8a, K8c), sessions or templates (K8b), texts (K7a, K9, K10), neurons (K3e); two-level where context instances exist | items (L1, L5), chains clustered by item (L2, L3a), prompts with both seeds (L3b), documents (L2a), item-clustered runs (L4); two-level (units, then random sets) for causal contrasts |
| Interval | joint prompt bootstrap (axis texts, pairs, cells and turns reweighted together), 90 %, 2,000, seed 1001 | 90 %, 2,000, seeds 11-14 | 90 %, 2,000, seed 1301 | 90 %, 2,000 |
| Tests | paired sign-flip K1-s (10,000 flips, seed 2001); bootstrap-t fallback (RG6) | one-sided bootstrap p; exact binomial and McNemar for item counts | one-sided bootstrap p | one-sided bootstrap p; rank p against 99 controls |
| Specificity | random-start loops, centred pseudo-axes (reported), planted signals (SENS_1b) | fitted null envelope, planted nulls (G-D1-null, G-D2-null), survival rule | rank p against RS39 (families of <= 2), prediction-interval p against RS19 / RS20 (Q9, Q11, Q14); p_IUT = max(p_effect, p_spec) | CTRL99 (rank), SEL+, UNC+, E-channel (paired contrasts) |
| 'No effect' | equivalence bound ('unlikely': upper bound of A_bar_abs < 0.25) | realised SD <= ceiling, else 'undecided (underpowered: realized MDE m)'; primary null with realised MDE > 2 x margin 'not decidable (underpowered)' | TOST at 5 % per side (90 % interval inside a frozen margin) | 90 % interval inside a margin >= the planning MDE (about 2.9 SE); 'not decidable (underpowered)' if the realised SE > 1.5 x the planning SE |
| MDE | joint MDE of the whole verdict function by simulation (power 0.8) | 2.49 planning SD (one-sided 5 %, power 0.8) | stated per measurement, simulated where needed | 2.49 planning SE |
| Inside the part | IUT within a verdict; Holm over F+ and F- (10 members each); rest exploratory | primaries unadjusted in their strings; Holm over parts (a), (b1), (b2) with Bonferroni x2 inside (b2); other rules 'secondary (not multiplicity-adjusted)' | Holm within Q1-Q18; no correction across families (the composite is a decision table); BH-FDR q 0.10 for exploratory grids | Holm within each measurement's verdict family; no correction across measurements; conjunctions not corrected |

**Round-wide conventions** (SO-PVAL): percentile intervals at 90 % with 2,000 replicates; one-sided bootstrap p = (1 + #replicates on
the null side of the threshold) / 2,001 (L's 'bootstrap share on the null side' is computed this way); rank p = (1 + #{controls >=
statistic}) / (N + 1); undefined replicates are counted and coded (R9.2), and a bound is undefined when more than 5 % of its replicates
are undefined (round 2's amendment 3(b) convention); every 'no effect' verdict is an equivalence or bound rule; an underpowered null
never counts against a hypothesis.

**Round-level confirmatory families.** Each family is controlled at family-wise 0.05 by Holm. The parts' verdict strings are unchanged;
the round-level label is added by r1b_ana.

| Family | Question | Members (part : ID 'string') | Member p | Control |
|---|---|---|---|---|
| CF-R+ | R | R.1's F+ (10): V_RSI_1b 'likely' (either strength: 'RSI-dominated' or 'substantial RSI component') at 14, 18, 6; R3.3 'reaches 1 at small push' along u_T^v at 14, 18, 6; R4.2 'lambda_RSI >= 1' at 14, 18, 6; R6.2 'self-sustaining at T 0.7' | R.1 | Holm, 0.05 (R's own) |
| CF-R- | R | R.1's F- (10): 'unlikely (RSI does not persist)' at 14, 18, 6; R3.3 'below 1 at small push' at 14, 18, 6; R4.2 'lambda_RSI below 1' at 14, 18, 6; R6.2 'dissipative at T 0.7' | R.1 | Holm, 0.05 (R's own) |
| CF-D | D | (a) D1-a verdict 4 without 'explained by' (D.17's support condition); (b1) D3 'split grows with length'; (b2) D2 member M or D5-a 'two competing voices sustain coherence longer' | D.17 (p_a, p_b1, p_b2 = min(2 p_M, 2 p_D5)) | Holm over three, 0.05 (D's own) |
| CF-K-attr | K | Q9's six members: K6e TOK carry, K6e TOK+KV carry, K6e bistability, K8b ACH carry, K8b TQA carry, K8b ACH bistability | p_IUT with PI specificity (K.4) | Holm over six, 0.05 (K's own Q9) |
| CF-K-link | K | the seven A+ members of K.7: K6b(i), K6b(ii) 'H-activity predicts lens confabulation beyond chain length'; K6c(i), K6c(ii) '... reduces lens confabulation specifically'; K6e-c 'H push raises ...' and 'H clamp lowers ...'; K6h 'H-activity predicts lens confabulation beyond chain length (QA)' | p_IUT (K.4) | Holm over seven, 0.05 (new at round level, C25) |
| CF-L | L | L3 attractor line 'H-attractor (self-sustaining or bistable, H-specific, seen in natural dynamics)'; L2b 'H rises beyond information loss' | below | Holm over two, 0.05 (new at round level) |
| CF-HERR | K and L | K8c 'H-activity predicts the next error beyond length, position, difficulty and uncertainty'; L5c 'H predicts the model's own step errors (within item, beyond step and uncertainty)' | K8c: p_IUT (K.4); L5c: max(p(beta_H <= 0), rank p) | Holm over two, 0.05 (new at round level, C21) |

- **Member p-values** (R.1's convention, adopted round-wide): a significance component uses its part's test p; a bound component its
  one-sided bootstrap p; an all-of verdict the largest component p; an any-of-k verdict k x its smallest component p (Bonferroni); a
  member whose verdict is not reached (gate failed, not decidable, not run) has p = 1. For CF-L: p(L3 attractor line) =
  max(2 x min(p_A1, p_A2), 2 x min(p_stick, p_pers)), where p_A1 = max(p(K_H^pool <= 1), p(Delta K <= 0)), p_A2 = max(surrogate p,
  rank p) of 'within-item bistability (H-specific)', and p_stick, p_pers are the largest one-sided p of the conditions of
  'H-specific stickiness' and 'H-specific persistence' (L3a, L3b); p(L2b) = max(p(beta_pos^(iota,e) <= 0), p(Delta beta^(iota,e) <= 0)).
- A member is labelled 'confirmatory (<family>)' only if its part's rule reaches the string with the part's own adjustments and it
  survives the round-level Holm step; otherwise 'exploratory (Holm not met)'.
- **Confirmatory 'against' statements:** CF-R- (Holm); D.17 'H-DUAL not supported in Qwen3-1.7B'; K.7 composite rows 5 and 7; L.5
  L-ALL 'length artefact (information-limited), no H-attractor' and 'no length trend and no H-attractor'. Outside CF-R- these are
  conjunctions of equivalence or bound rules (intersection-union tests) and need no further adjustment. Each is reported with its MDEs
  and never refutes a hypothesis for larger models.
- **Everything else is exploratory at the round level:** every R verdict outside F+ and F-, D's secondary rules and qualifiers, K's
  other family strings and the composite's suffixes, L's V-OWN, V-HOLO, V-CHAIN, L1, L2a, L2c, L3c, L3d, L4, L5a and L5b strings, every
  tier-2 result, and R0 (post hoc).
- **No correction across the four questions** (different hypotheses; round-2 convention 2.6). Under a global null each of the six
  'for' families (CF-R+, CF-D, CF-K-attr, CF-K-link, CF-L, CF-HERR) confirms falsely with probability <= 0.05, so at least one does
  with probability <= 0.30. The report prints the number of 'for' families with a confirmed member next to this bound, and R's reached
  and confirmed counts of F+ and F- next to their null expectations (R.1).
- **Cross-part side-by-side tables** (no pooling, no new verdict): the attractor verdicts of K6e and K8b next to L's attractor and
  specificity lines; K8c next to L5c; D2-a / D5-a next to L4's catfish string. Neither part's verdict is reported without the other's.

### 2.5 Where reconstruction happens, and how the inverse-free rule is met

Reconstruction here means inferring a cause from its effect (earlier fronts or tokens from a later state, text from an activation, what
a neuron or direction encodes, which input produced a readout). Every such step uses SO-GVRA:

| Where (tier) | What is reconstructed | Generator | Verifier (forward round trip through the unmodified model, decoupled) | Revise / limit | Verifier error measured on |
|---|---|---|---|---|---|
| R5.5 (1) | what u*_b and v_max^C encode (LABEL-u*) | a frozen bank of 24 instruction pairs | read round trip: the candidate's own 64-prompt pro / anti axis, read by the unmodified model, against u*_b and centred pseudo-axis nulls (R5.5) | 3 revisions, then 'abstained after 3 revisions' | known targets u_sea and u_tone at block b (held-out prompts) |
| D3 mode naming (2, reserve-funded, only if triggered) | the content of an unexpected D3 mode | the free answers of 4 trajectories in that mode | forward probe and state round trip in 16 fresh M-start contexts (D.14) | the next 4 trajectories in keyed order; 3 attempts, then abstain | the known A2 and B2 modes |
| D's INJ starts (1) | none (no cause is inferred); the loop makes instruction-stripped starts | keyed generation | forward check of the intended stance | 3 attempts; an abstaining item is replaced by a reserve item (G-D0d) | the other bundle's texts |
| K3e (1) | what H write vectors encode | neuron-lens chains | neuron-activation round trip (the readout fed back must activate the neuron above its 99.8th WikiText percentile and above 90 % of random readouts) | 3 attempts | planted known cases |
| K4, K7b, K8c lens reach (1) | the content of a read state | CL17 chains | the chain fed through the unpatched model: cos(s(c) - mu, h - mu) against V2 and token-shuffled thresholds | minor fix: next contrastive candidate; 3 attempts per seed | 64 planted states (G-K6) |
| X5 L.gv (2) | what an energy basin is | lexical statistics of basin texts | forward generation and acceptance test (L.4 X5) | gains x 1.25 for a minor fix, family switch for a critical flaw; 3 attempts | planted Ising basins and self-consistency cases |

**Not reconstructions** (forward maps, fits and nulls; allowed by the design rules and named in the parts): the J-lens transport
(averaged forward Jacobians; R5, K4); probes and regressions (K4p's ridge, K7a, L-par, L1's logistic I-hat, R3.3's WLS and R6.2's K_geo
2 x 2 normal equations, D's G-D1-null readout fit and coexistence index, L's pseudo-likelihood and kinetic Ising fits, L2b's
Frisch-Waugh slopes, L3b's 1-D grid search); eigen- and power iterations with forward and backward products (R4.2's Arnoldi and its
32 x 32 and 4 x 4 restricted problems, L3c's modularity); K's W enumerator (forward enumeration over the task's own arithmetic); K's
Sigma_l^{1/2} (V7 null generation); L's IAAFT inverse FFT, Phi^-1 on ranks and R^T of an orthogonal R (null generation and marginal
transforms); L5's TRUE / MODEL / FOREIGN classification (matching, no cause inferred).

**Never, anywhere:** a matrix inverse, pseudo-inverse, inverse Jacobian, Tikhonov-regularised inverse, inverse of a link function,
J = -C^-1, TAP or the graphical lasso to reconstruct anything. main §6.3's lint and the code review check this.

### 2.6 Conflicts between the parts and their resolution

| # | Conflict | Resolution (wins) | Why |
|---|---|---|---|
| C1 | Four separate pilots (R-P, D's pilot share, K-S0, L-P); session names in four styles; R's conditional 'R-S5b' equals the runner's continuation name of 'R-S5' | One pilot r1b_P0 before the freeze (SO-PILOT). Sessions r1b_<part>_<label> (main §4); R-S5b becomes r1b_R_S5L | One engineering record for the freeze; continuation names must not collide with base names |
| C2 | R's 'tier 0 / 1 / 2' (reduction levels) against the round's tier 1 / tier 2 (core / extension) | R's levels are renamed RL0 / RL1 / RL2 | One meaning per word |
| C3 | u*_b: R copies round 2's file; K and D rebuild it from R1 dir_<L><SUF>_4 with TV's mode_set | SO-AX: the round-2 file is canonical; the rebuild is SG10's reproduction check in r1b_R_S0 (cos with the file >= 1 - 1e-4; TV image_cos_d1_u_star reproduced within 1e-4); K's KS-axes step becomes a package read | One object; the rebuild remains as a check |
| C4 | Two RSI axes: R's verdicts use AX-1b's u_T^v(b) (perp*), K's channels and D2-f use round 1's u_T_perp(b) | K's named RSI and sea channels and K10a's sources and targets use AX-1b's u_T^v(14), u_sea^v(14), u_T^v(18) (and u_sea^v(18) in K7b) (fallback AX1 with the label 'round-1 axes (AX-1b unavailable)', OD-M5); D2-f adds u_T^v(14) (OD-M6); AX1 stays for the replay and as a reported comparison | One RSI axis per round, orthogonal to u*_b, so 'RSI' and 'style' are separate channels |
| C5 | Damage references: R's REF-R16 (dmg_sym), K's DMG_K, L's undefined 'merged REF16 / REF-K16' | SO-DMG: one functional and one calibrate rule; each part calibrates on its own frozen reference; every calibrated intervention also reports damage on REF-U32 | No calibration rule changes; damage becomes comparable |
| C6 | K's damage levels 'D1' / 'D2' and K2's diagnostics 'D1'-'D5' collide with part D's measurement IDs | delta1 / delta2; Dg1-Dg5 | Unambiguous cross-part citation |
| C7 | L needs TriviaQA 'S2' with >= 600 CC + 600 CI used neither for selection nor for choosing C; K's S2 (1,000 questions) has no consistency classes and chooses C* | New split S_L from K2's screening pool (OD-M8); L's 'S2' -> S_L and 'S1' -> S_fit | L1's MDEs assume 1,200 labelled, unselected items; K's S2 is used for C* |
| C8 | L needs >= 150 'known' + 150 'prone' arithmetic items with 11 gating answers and lens records; K4 has 256 correct + 128 wrong items with 5 gating answers | K4's wrong cell rises to 150 (K's own item fallback covers it); L adds 6 samples per item itself (11 in all); 'known' = K4 'correct', 'prone' = K4 'wrong'; L5's generation moves to r1b_L_B, after K4's gating (OD-M9) | L's G-CL would otherwise fail and L5a / L5b read 'not testable' |
| C9 | L's lens-record fields B_hal, FOV_H(q), A_H(q) come from pre-merge K designs | SO-LENSOUT mapping: k*_conf, share_H(s) with the RS39 band, a_H(s); steps 0-12 primary, 13-16 descriptive (OD-M13) | K's merged spec defines these fields |
| C10 | L's K.HSET / HSET+ / HBOOT ignore K's switch to H+ (G-K22) and K's H_ar fallback (G-K1) | H_op (SO-HN) and L's P rule applied to it (OD-M10) | K and L must speak about the same set |
| C11 | L's H channel: 'if K leaves both, the standardised-z mean is primary'; K's merged spec defines a_X = mean zCETT | a_X (mean zCETT) is primary in L; the standardised-z mean is L's sensitivity variant (OD-M11) | K's merge fixed one definition |
| C12 | L's CTRL99 'matched in CETT-norm quintiles (K.RANDSETS)' against K's actual rule (per-layer counts, mean answer-CETT within +-20 %) | K's rule with L's candidate seeds keyseed('r1bL-CTRL', k) (OD-M12) | L defers to K's generator; seeds keep L's sets independent of K's RS sets |
| C13 | Long-context engine: D crops all layers after unstopped chunks and gates parity at 0.02 nats on log-odds; K and L ask for per-layer crops after stopped reads and parity < 1e-4; L2a's own gate is also called 'G-LCE' | SO-LCE implements the union of features; each use keeps its frozen tolerance (D's G-LCE-1..4, K's G-K20, L's G-L2a-LCE, renamed); r1b_P0 measures both metrics on pilot texts at 16K and 32K; OD-M14 fixes what happens if L's tolerance is not attainable at 32,768 | Different readouts need different tolerances; the freeze must know which are attainable |
| C14 | L expects 'D.LONGCORPUS' (24 public-domain documents >= 32,768 tokens); D's filler is 16 streams of concatenated WikiText passages (CC BY-SA) | L.LONGDOC (PG19) is L's primary corpus, SO-LONGDOC (OD-M17); the contract item is dropped | Concatenated passages would bias L2a's growth class towards 'saturating' |
| C15 | WikiText-103 is used by D (streams), K (REF-W, N-text, F512 / F2048, REF-K16, K4j, a 1M-token reference) and L (200 natural-position passages 'from validation, disjoint from K.REF') | One pinned revision; split allocation SO-WT103 (train D, K's 1M-token reference, pilot; validation K; test L) (OD-M15) | L's 200 x 1,024 tokens would nearly exhaust the validation split; disjointness by split is checkable |
| C16 | L's pilot used rk_loop's scene prompts, which are analysis prompts (P256, F-neu) | SO-PILOT material disjoint from every analysis set | 'Nothing analysable before the freeze' |
| C17 | L's 'D.FORK-F1' is not a D object | SO-D1STIM F-sym items 0-31, slot j = i mod 6, v1 / v2 = versions 1 / 2 (OD-M16) | D's F-sym referents are fictional entities with two versions per attribute |
| C18 | Gate-name collisions: D's G-M and L's G-M; D's and L's G-H; L's G-long and L2a's 'G-LCE' against D's G-LCE-1..4; D's G-deg(L) and G-DEG; global gates named four ways (R RG1-3, RG16-18; K G1, G2, G-K24; D G-PIL; L G-pilot) | L's G-M -> G-ISM; D's G-DEG -> G-DGN; L2a's gate -> G-L2a-LCE; D's and L's G-H -> SG8; L's G-long -> SG9; global gates SG1-SG10 (main §3.1) (OD-M20) | Unique gate names in one index |
| C19 | Keys: D's 'r1bINJ' and 'r1bshuf' lie outside D's prefix; K uses 'k1bK_' | Registry SO-KEYS; renames 'r1bD_INJ', 'r1bD_shuf', 'r1bK_' (OD-M19) | One prefix per part; uniqueness asserted in code |
| C20 | J-lens: R states that K's chain lens needs R5.1 and LENS-J1b; K drops refits and fixes the convention itself (G-K-J) | SO-JLENS: the convention comes from K4j and is used by R5.1-R5.3; R5.1's per-block validity is a label on K's lens-side statements at that block; LENS-J1b is R's only (OD-M18) | K's lens uses the J-lens only for seeds; R5.1 runs after K4j |
| C21 | Near-duplicate tests: K8c and L5c (H before the model's own errors); K6e / K8b and L3 (H attractor); L5c's clause 'if K keeps a descriptive pre-error comparison, K's version stays descriptive' does not apply (K8c is confirmatory in K) | Both run (different items and designs). K8c and L5c form CF-HERR; the attractor verdicts are reported side by side (main §2.4); L5c's clause is deleted (OD-M24) | No cherry-picking between two tests of one claim |
| C22 | Catfish in two designs: D2 / D5 (one context, two voices) and L4 (two agents); L reports to D | L4's catfish string and its 'two entities persist' are qualifiers attached to D.17; they never change D's combined string (OD-M24) | D's combined Holm family is frozen |
| C23 | Account load: the parts put most sessions in account B (about 20 h against about 10 h in A) | r1b_D_B, r1b_D_C, r1b_R_S5 and R's tier-2 sessions run in A; A about 15.5 h, B about 15.1 h in tier 1 (main §4) (OD-M29) | D's sessions are input-free; R-S5's input package then stays in A |
| C24 | Drive I/O: R plans copy-in and writes at 10 min / GB; the other parts assume faster copy-in; round 2's r2_ana spent about 650 s outside its steps for about 0.9 GB of inputs | 10 min / GB for every part until r1b_P0 measures the rate; I/O allowances for r1b_K_ana and r1b_D_ana (OD-M28) | Round 2's own record supports the placeholder |
| C25 | K's composite A+ is an 'any of' over four Holm families (Q4, Q5, Q11, Q12), so its false-positive rate under the null can reach about 0.19; L has no round-level family | CF-K-link (Holm over the seven A+ members) and CF-L, as labels; K's and L's strings unchanged (main §2.4) | 'List exactly which verdicts are confirmatory' |
| C26 | p-value details: L's 'bootstrap share on the null side' against R's and K's (1 + #) / 2,001 | (1 + #) / 2,001 round-wide (SO-PVAL) | One formula |
| C27 | Time guard: R '<= 5 h', K 'elapsed < 5 h at the end of a step', the round-2 runner 20 min | SG5: the runner stops after the step that ends at >= 4.5 h since the notebook's first cell; steps <= 5 min; a continuation session resumes (OD-M27) | Leaves room for the final sync within Colab's limit |
| C28 | D's optional H covariate waits for K's G-K2, which K decides only in K-ana | It uses K's in-session preview of G-K2 from r1b_K_S1 (SG8 preview) if present when r1b_D_A starts; it is never waited for (OD-M25) | D's sessions stay input-free and parallel |
| C29 | L-ANA runs 'on a CPU runtime inside the round-1b analysis session', which no part defines | L-ANA is a step of r1b_ana (CPU runtime, account A) (OD-M26) | No A100 time for CPU work |
| C30 | R states that part D must recalibrate pushes per length | Moot: part D steers nothing (D.2) | - |
| C31 | Four carry estimators named K (R6.2, D2-f, K6e / K8b, L3b) | Cited with their part's estimator; never pooled (main §0.2) | Different designs |
| C32 | K-S6 (RK-1) is placed in account B, but its trigger (G-K5) is decided in r1b_K_ana (account A) | r1b_K_S6 runs in account A | Avoids an A -> B trigger transfer |

## 3 Gates

### 3.1 Global gates (apply to every part)

| # | Gate | Condition | If it fails | Decided in | Replaces |
|---|---|---|---|---|---|
| SG1 | Plan freeze | the sha256 of the five plan files and of every package file equal the constants in r1b_run_session.py (copied from runde1b_freeze.json); the freeze commit is an ancestor of the round-1b code commit | session stops before step 1 | every session except r1b_P0 | R RG1, K G1 |
| SG2 | Runtime | an A100 (sm80, 40 or 80 GB) for every model session and every analysis session that is planned on an A100; a declared CPU runtime for r1b_ana | session stops | every session | R RG2, K G2 |
| SG3 | Code freeze | rk_loop.py, rk_verdict.py, ufo_common.py, rsi_statements.py equal R's RG3 constants in the checkout and at every round-1 shard's code commit read; every imported round-2 module equals its sha256 at 8171b86 (main §6.1); every round-1b module equals its sha256 at the round-1b code commit | session stops | every session (r1b_P0: round-1 and round-2 modules only) | R RG3 |
| SG4 | Inputs present and intact | every start condition of main §4.2 met; every copied file (incl. B -> A and A -> B copies) matches the sha256 recorded by its producing session | the session waits (exit code 4); a measurement whose input is missing after two attempts of its producer reads 'not run (input missing)' | every session | R RG4; the parts' contract fallbacks stay as written |
| SG5 | Time guard and concurrency | the runner stops after the step that ends at >= 4.5 h since the notebook's first cell; steps <= 5 min and resumable; <= 2 concurrent sessions per account (an assumption) | session_partial.json; a continuation session (suffix b, c, ...) resumes; launches staggered | every session | R RG16, K G-K24, D and L time guards |
| SG6 | Report hygiene | R9.1-R9.4 in every output of every part: provenance block (plan and code hashes, module sha256, config and stimulus hashes, inputs with source and sha256, GPU and versions, key namespaces, reduction level, tier, session, attempt), codes for missing values, labels, distributions next to means | the session is 'incomplete' and no analysis reads its outputs | every session; checked by r1b_ana | R RG17; K's and L's hygiene rules |
| SG7 | Pilot-based trims | r1b_P0's measured rates and parity metrics are written into runde1b_freeze.json with each part's resulting trim level (R's RG18 reduction level; D's trims while D's projection exceeds 9.0 A100-h; K's drop order if decode or teacher-forced rates are > 30 % slower; L's trim order if any rate is below half its planning value), the round-level projection and the funded tier | no session starts | before the freeze | R RG18, D G-PIL, K's pilot rule, L G-pilot |
| SG8 | H-set validity for parts D and L | K's G-K2 ('H-neurons replicated (in-domain detection)') on H_op, and H_op in {H_tq, H+} for L's TriviaQA measurements. What runs uses r1b_K_S1's in-session preview; verdict labels use r1b_K_ana's value; disagreements are reported (round-2 OD-G8) | D: the optional covariate is not reported; L: as L's former G-H ('no validated H-set'; H-specific verdicts in L2b-L5 'not testable (no H-set)') | r1b_K_S1 (preview), r1b_K_ana | D G-H, L G-H |
| SG9 | Long-context engine | D's G-LCE-2 (memory-efficient kernel asserted) on every forward above 2K tokens in every part; r1b_D_A's G-LCE-1 / G-LCE-2 records exist before r1b_L_B starts | the step fails; r1b_L_B waits | every session; r1b_D_A | L G-long / G-LCE (D) |
| SG10 | Axes package | r1b_R_S0's SO-AX check: u*_b rebuilt from R1 dir_<L><SUF>_4 (and the round-2 r2RSIb entries at 18 and 6) with TV's mode_set members and signs has cos >= 1 - 1e-4 with the round-2 file, and reproduces TV image_cos_d1_u_star within 1e-4 | label 'SO-AX reproduction failed' on every SO-AX use; the round-2 file stays canonical | r1b_R_S0 | K's KS-axes check |

### 3.2 Part gates (index; full definitions in the parts)

**Part R (R.4).**

| Gate | Definition (short) | Blocks |
|---|---|---|
| RG1, RG2, RG3, RG16, RG17, RG18 | = SG1, SG2, SG3, SG5, SG6, SG7 (R's part) | as SG |
| RG4 | inputs present (= SG4 for R's files) | that measurement 'not run (input missing)' |
| RG5 | code checks CC1-CC4 (zero push identical 4 / 4; mirror >= 3 / 4; early stop < 1e-4; r1b_core = rk push to 1e-6) | verdicts using that arm 'descriptive (code check failed)' |
| RG6 | K1-s size in R0.7 (rejection <= 0.02 at nominal 0.01 in >= 90 % of rows) | bootstrap-t replaces p_s in R2-R7 (labelled) |
| RG7 | GO0_1b: AUC(u_T^v) >= 0.85; AUC(u_sea^v) >= 0.80, else AUC(u_sports^v) >= 0.80 | 'not decidable (RSI axis not resolved)' / 'PC not testable' |
| RG8a | PC_PASS_1b(b): all of P1-P5 | positive RSI strings at b 'undecided (positive control failed)'; delta rule needs >= 2 passing blocks |
| RG8b | SENS_1b(b): planted pass rate >= 0.8 for both RSI loops | negative RSI strings at b 'undecided (sensitivity check failed)' |
| RG9 | damage band: dmg_sym in [0.045, 0.055] | row label; > 1 of a loop's 4 pushes out: 'not decidable (damage calibration failed)' |
| RG10 | lens validity per block (beta interval within [0.5, 2], R^2 >= 0.5, sign agreement >= 0.8) | R's lens readouts at b 'not readable (lens not valid at b)'; label on K's lens-side statements at b (main §5) |
| RG11 | refit trigger: lens not valid at 14 or at all of 22, 24, 26 | R5.4 runs in r1b_R_S5L (tier 2, if funded) |
| RG12 | K defined: D_1's interval excludes 0 (R6.2, R6.3) | 'K not defined' |
| RG13 | erasure passed: R7.1 'unlearned' | R7.2 'not applicable', R7.3 'not testable' |
| RG14 | C-Q4b twin gate (RG13 and the twin's round 1: p_s <= 0.01, A lower bound >= 0.10 on u_T^v(14)) | 'not testable (gate: <reason>)' |
| RG15 | 0.6B justification J1-J3 | R8.2 'not run (not justified: ...)' (R8.2 is tier 2) |

**Part D (D.4, D.18).**

| Gate | Definition (short) | Blocks |
|---|---|---|
| G-PIL | = SG7 (D's part) | sessions do not start |
| G-LCE-1 | D1 cached-path parity at 16,384 and 32,000: max abs Delta l <= 0.02, median <= 0.002 nats (crop or copy path) | FB-1 (copy path), L* = 16,384, or FB-2 (16K full recompute) |
| G-LCE-2 | memory-efficient kernel on every forward > 2K tokens (= SG9 round-wide) | the step fails |
| G-LCE-3 | D2 decode parity at L_long: token-identical >= 3 / 4; repeats 4 / 4 | D2 at L_long 'not testable (engine)' |
| G-LCE-4 | graft parity at D3's and D4's long lengths: probe within 0.02 nats; decode identical >= 3 / 4 | D3 / D4 at that length 'not testable (engine)' |
| G-M / G-M2 | committed mass: median letter mass >= 0.9 (D1), yes / no mass >= 0.8 (D2) per length | verdicts at L 'not testable (readout not committed at L)' |
| G-D1-PC | D1 positive controls at L0: share response, endpoint range >= 4 nats, recency | 'not testable (readout insensitive)'; D1-b 'not testable (recency not detected)' |
| G-deg(L) | single-entity recall rho_L >= 0.5 | L not used as L* |
| G-D1-null | D1-a's verdict 4 in <= 5 % of planted nulls | D1-a 'not decidable (rule not specific at the measured noise)' |
| G-D2-PC | D2 turn 1: own stances induced, speakers separated, neutral role followed | D2 at L 'not testable (two-entity readout failed at L)' |
| G-D2-null | D2-a's verdict 4 in <= 10 % of planted nulls | D2-a 'not decidable (rule not specific at the measured noise)' |
| G-D0c | D3 order parameter: >= 14 / 16 items resolved, retention >= 0.5 | D3 at L 'not testable' |
| G-D0d | D3 starts: <= 4 abstentions | D3 'not testable (starts not verifiable)' |
| G-D3pc | split detector: planted share >= simulated share - 0.25 | D3 at L 'not testable (positive control)' |
| G-D6a / PC-D6 | D4 identity readout: separation >= 2 nats in >= 14 / 16 roles; PC 'coexisting' | D4 at L 'not testable' |
| G-D9 | u_AA resolved: cross-fitted AUC >= 0.85 | u_AA readouts 'not resolved' |
| G-DGN (was G-DEG) | degeneration: <= 30 % of a cell's trajectories degenerate by n = 8 | label 'degeneration-dominated' |
| G-D6 | D5 detector and voice probes: all validation texts correct; separation >= 80 % | D5 'not testable' |
| G-0.6 | 0.6B readout: G-M and range >= 2 nats | D1-s 'not testable (0.6B readout)' |
| G-H | = SG8 (D's part) | optional covariate not reported |

**Part K (K.6).**

| Gate | Definition (short) | Blocks |
|---|---|---|
| G1, G2, G-K24 | = SG1, SG2, SG5 | as SG |
| G-K-J | J-lens sha256; bridge median cosine >= 0.9 for one index convention (K4j) | sha256 mismatch: session stops; otherwise label 'J-lens bridge failed'; the chosen convention is SO-JLENS's for R5.1-R5.3 |
| G-K1 | H_tq yield (>= 400 per class at the cap) and |H| >= 1 at C* | H_ar primary, or fallback C (labels); H_op changes |
| G-K2 | 'H-neurons replicated (in-domain detection)' (premise P-H) | composite row 1; H-based verdicts 'H-set not validated'; = SG8 for D and L |
| G-K3 | H in arithmetic | arithmetic H statements use H_ar ('domain-specific H-set') or read 'H not valid in arithmetic' |
| G-K4 | 'chain lens reads intermediates' (P-L) | lens-side verdicts 'not testable (lens fails its positive control)' |
| G-K4b | intermediates present lens-free at >= 1 block (K4p) | with a G-K4 failure: 'nothing to read' |
| G-K5 | 'self-rollout calibrated' (K5a) | label 'lens substitute not calibrated'; RK-1 eligible (tier 2) |
| G-K6 | verifier reliable: false-accept <= 0.20, sensitivity >= 0.50 | k*_ver labelled 'verifier unreliable' |
| G-K7 | J-lens seeds on task text | label only |
| G-K8 | carry defined (R6K) | 'K not defined' |
| G-K9 | random-set matching (39 / 20 matched) | labels; < 20 sets: 'specificity not testable' |
| G-K10 | code checks (patch identity; decode = teacher-forced fronts; global offset; frozen-QK reproduction; block-27 zero; FD vs JVP < 1e-3; K10a KV structural zero) | affected verdicts 'descriptive (code check failed)'; the offset check is a label only |
| G-K11 | SO-AX available: the package from r1b_R_S0 (SG10) and AX-1b(14), AX-1b(18) | AX1 fallback with label; without SO-AX: named channels dropped ('axes unavailable'); K3f and K10a's named sources 'not run' |
| G-K12 | H damage at alpha 3 <= 0.15 nats | label 'damage-limited' |
| G-K13 | long-form parse failures <= 20 % | 'format not followed' |
| G-K14 | H scale and detector valid in the lens frame | label on lens-frame H statements |
| G-K15 | contrast-surviving share >= 0.2 for both-stream interventions | 'intervention largely cancelled by the contrast'; no 'no causal role' verdict |
| G-K16 | in-session unit rule: projected z >= 7 | units doubled (K6e) or 384 templates (K8b); logged; K7b and K9's c4 / c5 dropped to compensate (K.8) |
| G-K17 | Qwen3.5-2B MTP wiring agreement >= 0.4 | CL35 'not run (MTP wiring failed)'; G-K5 'not decidable' |
| G-K18 | Qwen3.5-2B loads with 15 mtp tensors | K5a 'not run'; variant E; label 'not calibrated (K5a not run)' |
| G-K19 | item counts: wrong >= 64; K6g >= 150; A-bnd >= 250 | 'not testable (too few ...)' / 'reduced design' / 'underpowered' |
| G-K20 | long-context parity: cached vs recomputed < 1e-4; Gumbel flips <= 1 % | the arm 'not run (engine parity)' |
| G-K21 | CRN twin re-run >= 97 % token-identical | label 'numerical divergence' |
| G-K22 | H stability: median Dg2 Jaccard >= 0.5 | H+ replaces H everywhere in K (H_op = H+) |
| G-K23 | QA lens positive control | QA H-link 'not testable' |

**Part L (L.9 and the measurement sections).**

| Gate | Definition (short) | Blocks |
|---|---|---|
| G-H | = SG8 (L's part) | L1 with the best sparse set, 'no validated H-set'; H-specific verdicts in L2b-L5 'not testable (no H-set)' |
| G-sel | SEL label with |phi| < 0.1 and >= 300 per class | SEL+ verdicts capped at 'collective / specific against random populations (selection effect not excluded)' |
| G-cal-L1..L5, G-cal-Ising | L-CAL items (a)-(n); L1-C 3 / 3 seeds per profile; planted Ising >= 4 / 5 | dependent branch 'not testable (pipeline calibration failed)' |
| G-pilot | = SG7 (L's part) | trim order applied before the freeze |
| G-long / G-LCE (D) | = SG9 | r1b_L_B waits |
| G-L2a-LCE (was L2a's 'G-LCE') | chunked and cached log-probs against a full pass on 2 documents at 16k and 32k: relative difference < 1e-4 | L2a halted (subject to OD-M14) |
| G-MI | copy and shuffle controls | L2a 'not decidable (estimator check failed)' |
| G-twin | sham >= 31 / 32 identical; X1 twins reproduce chains in >= 99 % (X1 calls this part G-twin-X1) | 'not decidable (common random numbers broken)' |
| G-manip, G-push | in-window displacement >= 3 SD of sham; d^(1)'s interval excludes 0 | 'not testable (push too weak)' / 'K not defined (push too weak)' |
| G-dmg | every random control within +-10 % of the H intervention's damage | that contrast 'not testable (no damage-matched control)' |
| G-ESS, G-ISM (was G-M) | Ising sample adequacy; model check | fallback to 30 units, then 'not testable (sample)'; 'not Ising-modellable' |
| G-sigma | integral-fluctuation-theorem check | Sagawa sigma 'descriptive (sigma pipeline check failed)' |
| G-duet | format and readout (>= 90 % complete runs; first-message AUROC >= 0.80) | L4 'not testable (format / readout)' |
| G-duet-manip | H gain changes A's commitment beyond damage-matched random gain | H line 'not testable (H gain has no specific effect on agent A)' |
| G-CL | SO-LENSOUT with >= 150 correct ('known') and >= 150 wrong ('prone') items | L5a, L5b 'not testable (no exact-label lens corpus from K)' |
| G-parse | >= 70 % of CoTs parse | L5c 'not testable' |

### 3.3 Cross-part dependencies

| Producer (part: gate or object, session) | Consumer | Effect when missing or failed |
|---|---|---|
| main: SG1-SG7 | every part | sessions stop or wait |
| R: SO-AX package and SG10 (r1b_R_S0); AX-1b(14), AX-1b(18) (r1b_R_S2_14, _18) | K (G-K11: K3f, K7b, K9, K10a); D (D2-f, r1b_D_ana); L (L3b's descriptive u*_14 line) | K: AX1 fallback with label, else channels dropped; D2-f: the missing axis is not reported; L: the descriptive line is dropped |
| R: REF-R16 export (r1b_R_S0, via r1b_xfer) | K (REF-U32 reporting, r1b_K_S2a); L (damage reference, r1b_L_A) | start condition (SG4) |
| R: RG10 per-block J-lens validity (r1b_R_S5) | K's lens-side statements at that block | label 'J-lens not valid at b (R5.1)' only |
| K: G-K-J convention (r1b_K_S1) | R5.1-R5.3 (r1b_R_S5) | start condition of r1b_R_S5 |
| K: REF-K16 (K1, frozen) | R (REF-U32 reporting); L (damage reference) | frozen before the freeze |
| K: G-K1, G-K22 -> H_op; SO-HN files (r1b_K_S1) | L (P, L.POP, CTRL99, SEL+, UNC+, CETT); D (optional covariate) | L: start condition of r1b_L_A; D: covariate omitted |
| K: G-K2 = SG8 (preview r1b_K_S1; final r1b_K_ana) | D (covariate); L (H-specific labels) | as SG8 |
| K: SO-LAB incl. S_L (r1b_K_S1) | L (L1, L.R-A, L3b, damage reference) | start condition of r1b_L_A; L's own fallback if S_L is smaller (MDEs scaled by sqrt(1,200 / n)) |
| K: K4 gating and SO-LENSOUT (r1b_K_S2a) | L5 generation (r1b_L_B), L5a / L5b (L-ANA) | start condition of r1b_L_B; G-CL |
| L: L3b's twin-0 rows (r1b_L_A) | L3a (r1b_L_B) | start condition of r1b_L_B (a dependency inside part L, made explicit) |
| K: G-K4, G-K4b, G-K5, G-K14 labels (r1b_K_ana) | L5a, L5b, V-OWN (L-ANA) | attached as labels |
| D: SO-LCE and G-LCE-1 / G-LCE-2 = SG9 (r1b_D_A) | L (r1b_L_B start; L2a); K (G-K20 arms) | r1b_L_B waits; K's own G-K20 decides K's arms |
| D: SO-D1STIM (frozen) | L4 | frozen before the freeze |
| L: L4 catfish string and 'two entities persist' (L-ANA) | D.17 qualifiers (r1b_ana) | qualifier omitted ('not run') |
| K8c and L5c; K6e / K8b and L3 | round-level CF-HERR and side-by-side tables (r1b_ana) | members not reached have p = 1 |

## 4 Budget and sessions

### 4.1 Rates and planning conventions

- Each part's cost table stands (R.5, D.19, K.8, L.8): planned A100-minutes = setup + I/O + 1.5 x work, at the round-2 planned rates
  (round-2 plan 7.1: 2.819 GFLOP per token through 28 blocks at 15.8 TFLOP/s in fp32 with TF32 off; 1.0 s per generate call of 32 x 40
  tokens; 25-35 ms per decode step at batch 32; reads 0.25-0.45 s per 32 texts) and each part's long-context, hook and FISTA
  estimates. r1b_P0 replaces the estimates by measured rates before the freeze (SG7).
- **Drive I/O.** 10 min per GB for copy-in and writes in every part (R's placeholder), until r1b_P0 measures the rate. Round 2's r2_ana
  record supports it: 976.7 s in all, 328.9 s in its five steps, so about 650 s for setup, the copy-in of about 0.9 GB, hashing and
  syncs. I/O allowances are added where a part's setup does not cover its copy-in at this rate: r1b_K_ana +10 min, r1b_D_ana +5 min.
- **Harmonisation additions** (work, before the x1.5 margin): S_L screening in r1b_K_S1 +6.4 min (charged to L); K4's 22 extra wrong
  items in r1b_K_S2a +0.67 min (charged to L); L's 6 extra gating samples in r1b_L_B +0.33 min; REF-U32 reporting of R's pushes on
  REF-K16 +2.7 min and of K's interventions on REF-R16 +2.0 min (charged to the round).
- Sessions stay far below the 5-h Colab limit (largest: r1b_D_A, 3.4 h).

### 4.2 Sessions

A100-minutes as planned (setup + I/O + 1.5 x work). 'Starts after' lists start conditions (SG4); 'freeze' means the freeze commit.

| Session | Part | Tier | Acct | Wave | Starts after | Content (IDs) | A100-min |
|---|---|---|---|---|---|---|---|
| r1b_P0 | all | pilot | B | 0 | harmonised part files and package drafts; before the freeze | P0-R (R-P's list), P0-D (D's pilot list), P0-K (K0 items 1-9 incl. the Qwen3.5-2B load), P0-L (L0 items i-vi); shared items measured once (Drive throughput, short-context decode with and without L's capture set, 16K / 32K prefill with and without down_proj hooks, crop / copy path and cached-vs-recomputed parity metrics); SO-PILOT only; engineering numbers only | 68.2 |
| r1b_R_S0 | R | 1 | A | 1 | freeze | R0, R8.1, R9.5; SO-AX package (AX1, U*, leave-loop-out modes, SG10 check); exports to r1b_xfer (REF-R16 token ids, u*_b) | 22.2 |
| r1b_R_S2_14 | R | 1 | A | 1 | freeze | R2.1 (axis texts), R2.2 at 14; AX-1b(14) | 58.5 |
| r1b_R_S1 | R | 1 | A | 1 | r1b_R_S0 (concurrency, R.5) | R1 closed and open loop | 35.9 |
| r1b_R_S2_18 | R | 1 | A | 2 | r1b_R_S2_14 | R2.1 reads, R2.2 at 18; AX-1b(18) | 60.8 |
| r1b_R_S2_06 | R | 1 | A | 2 | r1b_R_S2_14 | R2.1 reads, R2.2 at 6; AX-1b(6) | 60.8 |
| r1b_R_S3 | R | 1 | A | 3 | r1b_R_S2_14, _18, _06 | R3.3, TXT0 (R3.4), R4.1-R4.3 | 100.2 |
| r1b_R_S5 | R | 1 | A | 4 | r1b_R_S3; r1b_K_S1 (k4j.json via dsse_runs_B) | R5.1, R5.2, R5.3, R5.5; AX-1b at the lens blocks; RG10, RG11 | 29.1 |
| r1b_R_S6 | R | 1 | A | 4 | r1b_R_S3 | R6.1, R6.2, R6.3 | 63.2 |
| r1b_R_S7 | R | 1 | A | 4 | r1b_R_S3 | R7.1-R7.3 (RG13, RG14 in session) | 21.8 |
| (R's REF-U32 reporting) | R | 1 | A | - | inside r1b_R_S2_*, _S3, _S6, _S7 | damage of R's calibrated pushes on REF-K16 | 4.0 |
| r1b_R_ana | R | 1 | A | 5 | all R tier-1 sessions | R pass 1: all R statistics and verdicts, F+ / F-, RG7, RG8a, RG8b, RG12, RG15 (J1, J2), R9 report check | 55.3 |
| r1b_D_A | D | 1 | B | 1 | freeze | G-LCE-1 / G-LCE-2 first (SG9), stream prefills, D1 (main, neutral, degradation, non-competing, D1-h), D1-c, D1-d, D1-e, D1-f, D1-r, D1-v, D1-s | 202.3 |
| r1b_D_B | D | 1 | A | 2 | freeze (input-free; fills a free A slot) | G-LCE-3, D2 | 122.2 |
| r1b_D_C | D | 1 | A | 4 | freeze (input-free; fills a free A slot) | D3 (with starts), D4 (with starts), D5 (D5-x only if tier 2 is funded) | 149.6 |
| r1b_D_ana | D | 1 | A | 5 | r1b_D_A (via dsse_runs_B), r1b_D_B, r1b_D_C; r1b_R_S0, r1b_R_S2_14 | all D verdicts, G-D1-null, G-D2-null, D3 mixtures, D2-f, combined Holm (D.17); I/O allowance included | 32.5 |
| r1b_K_S1 | K | 1 | B | 1 | freeze | K2 (incl. the S_L screening, OD-M8), K3a-e, K3g(ii), K4j; SG8 preview | 126.4 |
| r1b_K_S3 | K | 1 | B | 1 | freeze (parallel to r1b_K_S1) | K5a | 28.9 |
| r1b_K_S2a | K | 1 | B | 2 | r1b_K_S1; r1b_K_S3's k5a_calib.json (or r1b_K_S3 'not run' after two attempts: variant E); r1b_xfer_A/dmg_ref_R16.json | K4 gating (incl. export to SO-LAB), K4 (150 wrong items), K4p, K6c, K6f, K6g, K6h, K10c; REF-U32 reporting of r1b_K_S1's calibrations | 92.2 |
| r1b_K_S2b | K | 1 | B | 3 | r1b_K_S2a | K6e | 51.1 |
| r1b_K_S4 | K | 1 | B | 3 | r1b_K_S1, r1b_K_S3 | K8a, K8b, K8c, K7a | 119.9 |
| r1b_K_S5 | K | 1 | A | 4 | r1b_K_S1, r1b_K_S2a, r1b_K_S4 (K9's K8a instances, K8b and K8c texts; via dsse_runs_B); r1b_R_S0; r1b_R_S2_14, _18 | K9, K10a, K10b, K3f, K7b (SO-AX read; no KS-axes recomputation) | 70.1 |
| r1b_K_ana | K | 1 | A | 5 | r1b_K_S1, _S2a, _S2b, _S3, _S4 (via dsse_runs_B), r1b_K_S5 | all K statistics, K6a / K6b / K6d, Holm families, the K.7 composite; I/O allowance included | 43.0 |
| r1b_L_A | L | 1 | B | 2 | r1b_K_S1 (SO-HN, SO-LAB incl. S_L); r1b_xfer_A (REF-R16, u*_14) | step 1 L.POP; L1, L3b, L4 (X2, X6 if tier 2 funded) | 168.9 |
| r1b_L_B | L | 1 | B | 3 | r1b_L_A (L.POP, and L3b's twin-0 rows, which L3a reads); SG9 (r1b_D_A's G-LCE-1 / -2 records); r1b_K_S2a's gating export | L.R corpus, L2, L3a, L3c, L3d; L5 generation (own CoT, sub-answers, 6 extra gating samples, t* pass) (X1, X3, X4, X5 if tier 2 funded) | 117.2 |
| r1b_ana | all | 1 | A (CPU) | 6 | r1b_R_ana, r1b_D_ana, r1b_K_ana, r1b_L_A, r1b_L_B (via dsse_runs_B) | L-ANA (L's verdict assembly, L5 analyses, L.5 combinations); round-level families (main §2.4); cross-part labels and side-by-side tables; runde1b_summary.json; merged manifest | 0 |
| **Tier 2** | | | | | | | |
| (in r1b_D_C) | D | 2 | A | 4 | funded at the freeze | D5-x (D's T1 halves it first) | 30.0 |
| (in r1b_L_A, r1b_L_B) | L | 2 | B | 2-3 | funded at the freeze | X2 22.95, X6 3.9 (L_A); X1 12.75, X3 7.5, X4 3.75, X5 4.5 (L_B) | 55.4 |
| r1b_R_S5L (was R-S5b) | R | 2 | A | 5 | r1b_R_S5 with RG11 'refit'; funded | R5.4 (bf16 refit), repeated R5.1, R5.2 / R5.3 at newly valid blocks | 105.0 |
| r1b_R_S8 | R | 2 | A | 6 | r1b_R_ana with RG15's J1 and J2 met; funded | R8.2 (J3 in session) | 19.5 |
| r1b_R_ana2 | R | 2 | A | 7 | r1b_R_S5L or r1b_R_S8 ran | R pass 2 | 10.2 |
| r1b_D_R | D | 2 | A | 6 | r1b_D_ana (R2's trigger: D3 'split grows with length'); funded | D's reserves in priority R1, R2 (conditional), R3 | 119.0 |
| r1b_K_S6 | K | 2 | A | 6 | r1b_K_ana with G-K5 not 'calibrated'; funded | RK-1 (trained MTP head, bf16; descriptive arm) | 58.0 |
| r1b_K_S7 | K | 2 | B | 4 | r1b_K_S3; funded | RK-2 (H-link on the native lens, Qwen3.5-2B) | 53.0 |

### 4.3 Totals

| | Tier 1 (A100-h) | Tier 2 (A100-h) | Parts' own figures (A100-h) |
|---|---|---|---|
| R | 8.53 (sessions 8.46 + REF-U32 reporting 0.07) | 2.25 (r1b_R_S5L 1.75, r1b_R_S8 0.33, r1b_R_ana2 0.17) | 11.03 = 8.78 unconditional (incl. R-P 0.32) + 2.24 conditional |
| D | 8.44 (r1b_D_A 3.37, r1b_D_B 2.04, r1b_D_C 2.49, r1b_D_ana 0.54) | 2.48 (D5-x 0.50; reserves R1 0.53, R2 0.90, R3 0.48 with r1b_D_R's setup) | 9.03 (incl. pilot share 0.17 and D5-x); reserves 1.9 outside |
| K | 8.86 sessions, of which 8.63 K's own (incl. I/O 0.17), 0.18 for L, 0.05 REF-U32 reporting | 1.85 (RK-1 0.97, RK-2 0.88) | 8.78 (incl. K-S0 0.32); conditional 2.2 outside |
| L | 4.77 sessions; 4.95 charged (incl. 0.18 run in K's sessions) | 0.92 (X1-X6) | 6.0 (core 5.08 incl. L-P 0.32; extensions 0.92) |
| **Round** | **30.60** (account A 15.49, account B 15.12) | **7.50** (A 5.70, B 1.81) | 34.84 (+ reserves 4.1 outside) |
| Pilot r1b_P0 | 1.14 (account B; before the freeze) | | 1.12 (four separate pilots) |

- **Tier 1 = 30.6 A100-h** at the parts' planned rates, near the round-level target of about 30 h. No rule was changed to reach it: the
  difference from the parts' 34.8 h is the pilots (1.1 h, now r1b_P0), R's conditional sessions (2.2 h), L's extensions (0.9 h) and
  D5-x (0.5 h), all of which the parts themselves mark as pilot, conditional or extension; the harmonisation additions (0.30 h) and
  I/O allowances (0.25 h) add 0.55 h.
- All funded: tier 1 + tier 2 + pilot = 39.2 A100-h. Conditional tier-2 items spend only when triggered.
- Each part stays within its own cap (pilot excluded): R 8.53 + 2.25 = 10.78 h <= 12.5 h (RG18); D 8.44 (+ 0.50 if D5-x is funded)
  <= 9.0 h; K 8.63 + its pilot share 0.32 = 8.95 h <= 9.0 h (RK-1 and RK-2 therefore need the round's tier-2 funding, not K's
  headroom); L 4.95 + 0.92 = 5.87 h.

### 4.4 Order, waves and wall clock

- Wave 0: r1b_P0 (B), then the freeze (main §9).
- Account A, two slots: (1) r1b_R_S2_14 -> r1b_R_S2_18 -> r1b_R_S3 -> r1b_R_S6 -> r1b_K_S5 -> r1b_D_C; (2) r1b_R_S0 -> r1b_R_S1 ->
  r1b_R_S2_06 -> r1b_D_B -> r1b_R_S5 -> r1b_R_S7 -> r1b_R_ana -> r1b_K_ana -> r1b_D_ana; then r1b_ana (CPU). A session waits only
  for its start conditions; the slot order is a suggestion (packing, not a rule).
- Account B, two slots: (1) r1b_K_S1 -> r1b_K_S4 -> r1b_L_A -> r1b_L_B; (2) r1b_K_S3 -> r1b_D_A -> r1b_K_S2a -> r1b_K_S2b
  (r1b_K_S4 early, because r1b_K_S5 in A waits for it).
- Expected wall clock for tier 1: about 9-10 h from the freeze (critical path: R's chain in A, about 5.6 h, then the analyses; K's and
  L's chain in B, about 8.9 h, because r1b_L_B waits for all of r1b_L_A).
- Tier 2 runs after its triggers and funding (r1b_R_S5L as soon as r1b_R_S5 decides RG11; the others after the analysis sessions
  named in main §4.2).

### 4.5 Tier 2 and the drop order

Tier-2 items are given up in this order (first dropped first) when the funded budget is smaller than tier 1 + tier 2. The order keeps
longest what can change a verdict or repair an instrument, and keeps each part's own relative order (L.8 for X1-X6; D.19 for D's
reserves, priority R1 > R2 > R3; K.8 for RK-1 > RK-2):
1. RK-2, r1b_K_S7 (0.88 h): descriptive, never enters the composite.
2. X5 (0.08 h), 3. X4 (0.06 h), 4. X3 (0.13 h), 5. X6 (0.07 h): descriptive or deepening.
6. R8.2, r1b_R_S8 (0.33 h; with r1b_R_ana2 0.17 h if r1b_R_S5L does not run): descriptive size comparison.
7. D's reserve R3 (0.48 h), 8. D's reserve R2 (0.90 h; conditional), 9. D's reserve R1 (0.53 h, with r1b_D_R's 5-min setup).
10. D5-x (0.50 h; D's T1 halves it first): a descriptive qualifier of D.17.
11. X1 (0.21 h): adds attractor route A3 to L's attractor line.
12. X2 (0.38 h): adds the H-down coupling line (Holm over up and down).
13. RK-1, r1b_K_S6 (0.97 h; conditional on G-K5): a second, trained lens arm when the substitute is not calibrated.
14. R5.4, r1b_R_S5L with r1b_R_ana2 (1.92 h; conditional on RG11): makes R's lens readouts valid where the Neuronpedia lens fails.

**Tier 1 under a smaller budget.** If r1b_P0's projection of tier 1 exceeds the funded tier-1 budget, each part applies its own frozen
order, in proportion to its share of tier 1, until its projection is at most (funded tier 1) x (its planned tier-1 hours / 30.6): R its
reduction levels RL1, then RL2 (RG18); D its trims T2-T8 (T1 concerns D5-x, now tier 2); K its drop order items 1-6 (K.8); L its trims
2-5 (L.8; trim 1 concerns the extensions). Items that the parts mark 'never dropped' stay (K: K2, K3a, K4 V1 / V2, K4p, K5a, the K6e
primary cells, the K8b main block, K8c). The trims and the resulting projection are written into runde1b_freeze.json before the freeze.

### 4.6 Sensitivity

- **Margin.** Of the 30.6 h, about 9.2 h is the x1.5 margin on work; expected consumption at the planned rates is about 21.4 A100-h
  (setup and I/O 3.0 h, work 18.4 h).
- **Half rate.** If every work item takes twice its planned time, tier 1 consumes about 39.8 A100-h (setup and I/O unchanged, work
  36.8 h); the planning projection with the x1.5 margin re-applied would be 58.2 h. The parts' own pilot triggers fire well before
  that (K at > 30 % slower, L below half the planned rate, D and R at their caps), so r1b_P0 decides before the freeze whether the
  funded budget, the parts' trims or both carry the overrun (question Q1).
- **Drive.** If r1b_P0 measures Drive throughput >= 20 MB/s, R's tier-1 I/O share (about 0.9 h of R.5's 1.2 h; the rest is in R's
  conditional sessions) drops to about 0.1 h and the allowances of r1b_K_ana and r1b_D_ana (0.25 h) vanish: tier 1 about 29.5 h.
- **Optional packing saving.** r1b_D_ana is CPU-bound ('CPU on the VM', D.19); on a CPU runtime it would cost no A100 time (-0.5 h) but
  take longer. A packing choice, recorded in session_done.json.

### 4.7 Runner behaviour (as round 2, adapted)

- One notebook per session per account; the notebook names its account. Every session: SG1-SG4 at start, the copy-in of its inputs
  (read-only under <work>/in/), its steps in order (each skipped when its outputs exist and the runner recorded its success), an
  incremental, error-tolerant Drive sync after every step, the time guard (SG5), session_done.json or session_partial.json.
- A continuation session (suffix b, c, ...) uses the same work and Drive folder; finished steps are skipped. A packing change never
  changes a measurement, threshold or rule and is not an amendment.
- A session that does not finish after two attempts may be declared 'not run' by the operator in dsse_runs/r1b_declared_not_run.json
  (account A; the runner checks the two attempts), so that the analyses can start; its measurements read 'not run (session failed
  twice)'.
- In-session decisions (L*, FB-1 / FB-2, RG11, RG13, RG14, G-K16, G-K22, K4's b*, K5a's variant, the D2 throughput fallback) follow
  round-2 OD-G8: the in-session value governs what ran; the analysis recomputes it and reports any disagreement.

## 5 Analyses

| Verdicts | Computed in (runtime, account) | Inputs | Outputs | Report |
|---|---|---|---|---|
| R in-session: RG5, RG9 per push, RG11, RG13, RG14; PUSH-DN tables | R sessions (A100, A) | session data | steps/*.json, session_done.json | R's tables |
| R0 re-readings (post hoc) and R8.1 | r1b_R_S0 (A100 runtime, model-free, A) | round-1 and round-2 files (main §2.2) | r0_table.json, r8_1.json | R0 section, labelled 'post hoc (R0)' |
| All R2-R7 rules, SENS, joint MDEs, Holm F+ / F-, RG7, RG8a, RG8b, RG12, RG15 (J1, J2), R9 check | r1b_R_ana (A100, A) | r1b_R_S0 ... r1b_R_S7 | r1b_R_verdict.json (incl. the F+ / F- table) | R.6 strings, CF-R+ / CF-R- |
| R5.4, R8.2 (tier 2) | r1b_R_ana2 (A100, A) | r1b_R_S5L, r1b_R_S8 | r1b_R_verdict2.json | labelled 'conditional run' |
| D in-session: G-LCE-1..4, FB-1 / FB-2, L*, G-M, G-deg, G-D0d, G-D9, G-D6 | r1b_D_A, _B, _C (A100) | session data | session records | - |
| All D rules, G-D1-PC, G-D1-null, G-D2-PC, G-D2-null, G-D0c, G-D3pc, G-D6a, G-DGN, G-0.6, D2-f, combined Holm | r1b_D_ana (A100 VM, CPU-bound, A) | r1b_D_A ... r1b_D_C, SO-AX (u_T_perp(14), u*_14, u_T^v(14)), TV R6 cells | d_verdict.json | D.17 string with qualifiers; CF-D |
| K in-session: G-K-J, G-K1, G-K7, G-K9, G-K10, G-K11, G-K16, G-K17, G-K18, G-K19, G-K20, G-K21, G-K22, b*, K5a's variant, SG8 preview | K sessions (A100) | session data | session records, hn_set.json etc. | - |
| All K statistics, K6a, K6b, K6d, Holm families, equivalence verdicts, K.7 composite | r1b_K_ana (A100, A) | r1b_K_S1 ... r1b_K_S5 | k_verdict.json | K.7 composite with suffixes; CF-K-attr, CF-K-link, CF-HERR (K8c) |
| L1 statistics (L-A), L2, L3a, L3c, L3d (L-B), L3b and L4 (L-A); in-session gates | r1b_L_A, r1b_L_B (A100, B) | SO-HN, SO-LAB, SO-LCE, SO-LONGDOC, SO-D1STIM, SO-DMG | l_session_*.json | - |
| L5 analyses, L-ALL, V-OWN, V-HOLO, V-CHAIN, selection-robustness labels | L-ANA, a step of r1b_ana (CPU, A) | r1b_L_A, r1b_L_B, SO-LENSOUT, K's G-K2 / G-K4 / G-K4b / G-K5 / G-K14 labels | l_verdicts.json | L.5 answer template; CF-L, CF-HERR (L5c) |
| Round level: CF families (Holm), cross-part labels (RG10 on K's lens statements; SG8 on D and L; K's lens labels on L5), side-by-side tables (attractor; H before errors; catfish), the four answers, the 'for'-family count against its null bound | r1b_ana (CPU, A) | r1b_R_verdict.json, d_verdict.json, k_verdict.json, l_verdicts.json, all session manifests | runde1b_summary.json; r1b_manifest.json | results/ju_space/runde1b_ergebnisse/: the JSON files and a German readout AUSWERTUNG.md written after a skeptic's check, as in round 2 |

The four answers in runde1b_summary.json: R: V_RSI_1b per block with F+ / F- reached and confirmed; D: D.17's combined string with its
qualifiers; K: K.7's composite with its suffixes; L: L.5's answer template. Each answer names its rule (part and ID) and lists every
deviation (amendments, 'not run', labels, tier-2 items not funded).

## 6 Scripts to write

### 6.1 Code freeze

- Round-1 modules stay byte-identical: rk_loop.py 1a20f61992637386c97197b7283a8b743214aa89028f504c4af6c2fbedbab2a1, rk_verdict.py
  62aced20fcd7435201e1f427e5c294869fdab3adcb54801f092aa8d31ab9049e, ufo_common.py
  6f3919e93b638a325f9a8dd5762430c6b5e7693d4e1f5a317d97366caa49f9ab, rsi_statements.py
  6ff560c5b9c092858c7b50c4b041b7c886329bb7a451a6738d6eaa3754e3bec2 (R's RG3 constants; identical at HEAD c8257c2 and at round 1's
  gpu1 commit 740bfc0f); gpu/run_shard.py c74948163c57ad3cd79842d56b6a88dd6b94bb5ac208fba814ec5da4befbc610 is used unchanged for R1's
  closed-loop replay.
- Round-2 modules are imported unchanged at commit 8171b86 (identical at HEAD): r2_rk.py
  d4618662c25db47b83b6b9e40533adabdea5c6f0f723a165a27e1338bc8fc8ca, r2_stimuli.py
  41d2b564ec85835892c573e7f68d917006121c68764192a36c88e74035763434, r2_theta.py
  eace3b7bbc027fed5da30e541e06d45136104057d7b2f0a36407571efc0433c7, r2_theta_verdict.py
  b5ad18161d9f402ca098898fe8afa67e9cbd39d4e3a2e097a87d4ffe7b09b439, r2_unlearn_qwen.py
  e0271d17dd6a9b873b205c952b969a365064a7a458f9048b473cf9a9908f52c9, r2_linearity.py
  e410b8a0ebb6b1f4fe347a8cfdf007b7aaebc56a9f4779b7a619d688a01f1eba. neu_extract.py and neu_sparsity.py are read as patterns only
  (copied code is re-tested).
- New behaviour lives only in new modules under analysis/ju_space/round1b/. rk_loop is imported only through r2_rk.open_block (it loads
  the model and registers hooks at import) or run as a subprocess (R1); model-free code never imports it; keyseed and Gumbel elsewhere
  come from r1b_common's verified copy.

### 6.2 Modules

| Module | Owner | Content | Reuses |
|---|---|---|---|
| r1b_common.py | main | SO-KEYS (registry, keyseed / Gumbel copy, uniqueness test), SO-PVAL (bootstraps incl. two-level, one-sided p, rank p, PI p, Holm, BH), R9.1-R9.4 helpers (provenance block, codes, labels, distributions), sha256-verified copy | rk_loop source lines (checked, not imported) |
| r1b_run_session.py | main | runner: SESSIONS table of main §4.2, SG1-SG10, --pilot mode for r1b_P0 (SG1 off, output whitelist), --selftest, time guard 4.5 h, Drive sync, the B -> A and A -> B copies, the two-attempt rule | r2_run_session.py's structure (new file) |
| r1b_report.py | main | r1b_ana: L-ANA call, round-level families, labels, side-by-side tables, runde1b_summary.json, merged manifest, report check | - |
| r1b_core.py | R | function-style multi-block core (push, read, damage, scores, soft-token generation, lens predictions); SO-DMG's function-style damage and calibrate for K and L (equivalence-tested against r2_unlearn_qwen.RefKL / calibrate to 1e-6) | rk_loop's push (CC4) |
| r1b_rk.py | R | extra hooks through r2_rk.open_block: PUSH-NP, the 16-reference RefKL with both signs and top-1 change, the calibrated loop driver, generation-time capture, R1's reduced-state openers | r2_rk.open_block, CbarHook, r2_unlearn_qwen.calibrate / RefKL / ConstPush pattern |
| r1b_stats.py | R | STAT-1b (centred pseudo-axes, disattenuated K1-s and A, joint bootstrap, SENS, size and power plants, joint MDEs, Holm F+ / F-, cross-fitted Ritz pairs) | r2_rk.prompt_bootstrap, halvings, stats_on_axis (n_prompts = 256), fixed_fold_axes; rk_verdict |
| r1b_prompts_check.py, r1b_R0.py ... r1b_R8.py, r1b_R_verdict.py | R | R's measurement scripts and analysis (r1b_R0.py also writes SO-AX and the r1b_xfer exports) | as above |
| r1b_lce.py | D | SO-LCE (all features of main §2.3, parity utilities) | - |
| r1b_D_stimuli.py | D | frozen generators (d1_stimuli, d2_prompts, d3_items, d4_roles, d1v_pairs, filler streams); run before the freeze | SO-WT103 |
| r1b_D.py, r1b_D_verdict.py | D | D0-D5 session code; r1b_D_ana | SO-TURN, SO-LCE |
| r1b_hn.py | K | SO-HN (hook manager, CETT / zCETT, GPU FISTA selector, random-set generator, REF-W) | neu_extract patterns |
| r1b_K_labels.py | K | K1 (SO-LAB, scorers, REF-K16 token ids); run before the freeze | - |
| r1b_K_lens.py | K | SO-CL (CL17, CL35, verifier), SO-LENSOUT writer | SO-JLENS |
| r1b_gvra.py | K | SO-GVRA (generic loop, verifier-error bookkeeping) | - |
| r1b_te.py | K | SO-TE (central differences, frozen-QK attention, VJP edge attribution, FD-vs-JVP check) | - |
| r1b_K.py, r1b_K_verdict.py | K | K2-K10 session code; r1b_K_ana | SO-TURN, SO-LCE, SO-DMG |
| r1b_L.py, r1b_L_verdict.py, r1b_L_cal.py | L | L.10's builders (L.POP, L.R, L.PIP, L.EQMI, L.TWIN, L.MARKOVSUR, L.KISING, L.DUET, L.OWN, L5c parser and hazard, X3 toy); L-ANA; L-CAL generators | SO-HN, SO-LAB, SO-LCE, SO-DMG, SO-TURN, r2_linearity.auc_cols |
| notebooks/build_runde1b_gpu_notebooks.py | main | one notebook per session; account named; setup cells (pinned models and checksums, Qwen3-0.6B and Qwen3.5-2B where needed), the second-folder copy cells (dsse_runs_B in A; r1b_xfer_A in B), the self-test, the main run, a summary | build_runde2_gpu_notebooks.py's structure |

### 6.3 Tests (as round 2)

- **MOCK=1** for every Qwen script: rk_loop's tiny random model on a mock dsse_runs tree built locally (mock round-1 states via
  rk_loop.py MOCK=1 at blocks 14, 18, 6; mock round-2 files; mock upstream 1b outputs for every start condition).
- **QUICK=1** for X3's toy and for long-context code at tiny lengths.
- **Code tests:** K's scorer unit tests on planted outputs (100 %, else the session stops); L-CAL items (a)-(n) as code tests on the
  final code; D's G-D1-null and G-D2-null planted-data validations on synthetic data; SO-LCE cached-against-recomputed parity on the
  mock model; SO-KEYS uniqueness; SO-GVRA verifiers on planted known cases; CC4 (r1b_core = rk push to 1e-6); SO-DMG function-style
  damage = r2_unlearn_qwen.RefKL to 1e-6; an inverse-free lint (no torch.linalg.inv / pinv, no numpy.linalg.inv / pinv / lstsq
  outside a whitelist of named fitting steps, each marked in the code with its main §2.5 entry).
- r1b_run_session.py --selftest runs each session under MOCK / QUICK once per code commit before the main run (as round 2).
- **Code review** by reviewers blinded to round-1b outputs, before the first session. Fixes that change no value returned by any
  measurement, gate or rule on any input are not amendments.

### 6.4 Suggested split across implementers

(1) r1b_common, r1b_run_session, r1b_report, notebooks; (2) r1b_core, r1b_rk, r1b_stats; (3) R0-R4 scripts; (4) R5-R9 scripts and
r1b_R_verdict; (5) r1b_lce and r1b_D_stimuli; (6) r1b_D and r1b_D_verdict; (7) r1b_hn, r1b_K_labels, K2-K3; (8) r1b_K_lens, r1b_gvra,
K4-K6, K5a; (9) r1b_te, K7-K10, r1b_K_verdict; (10) r1b_L, r1b_L_verdict, r1b_L_cal. The file formats named in main §2.3 and §5 are
the interfaces between them.

## 7 Open decisions frozen now

**7.1 The parts' own lists stand:** R.8 (OD-R1 to OD-R17; OD-R17 now reads 'reduction level chosen by RG18 before the freeze'); D.21
(items 1-10); K's choices fixed in its measurement sections (K4 'all choices fixed now'; K.0 design principles; K.8's drop order; RK-1
and RK-2 conditions); L's choices fixed in L.0 (core and extensions), L.2 (conventions), L.8 (trim order) and L.9 (gates).

**7.2 Cross-part decisions:**

- **OD-M1 Sessions.** Names, accounts, waves and start conditions as in main §4.2; packing may change (main §4.7), measurements may not.
- **OD-M2 Pilot.** One pilot r1b_P0 in account B before the freeze, on SO-PILOT material only; its record goes into
  runde1b_freeze.json (SG7).
- **OD-M3 Tiers.** Tier membership as in main §4.2 and the drop order of main §4.5; the funded tier is the user's decision (Q1),
  recorded at the freeze.
- **OD-M4 Style mode.** u*_b = the round-2 file r2_q2/theta/b<b>_q0.pt['u_star'] (canonical); the rebuild is SG10's check.
- **OD-M5 K's named axes.** K's RSI and sea channels and K10a's sources and targets are AX-1b's verdict axes u_T^v(14), u_sea^v(14) and
  u_T^v(18) (perp*, orthogonal to u*_b), plus u_sea^v(18) for K7b's block-18 cells; u*_14 and u*_18 from U*. If AX-1b(14) or AX-1b(18) is not available because its R session was
  declared 'not run' after two attempts, K uses AX1 (u_T_perp, u_sea) with the label 'round-1 axes (AX-1b unavailable)'.
- **OD-M6 D2-f.** Projections on u_T_perp(14) (round-2 comparability), u*_14 and u_T^v(14) (the 1b comparison with R6.2).
- **OD-M7 Damage.** As SO-DMG: each part calibrates on its own frozen reference; every calibrated intervention also reports damage on
  REF-U32; delta1 = 0.05 and delta2 = 0.15 nats.
- **OD-M8 S_L.** K2's screening runs in blocks of 1,000 questions until CC >= 2,312 and CI >= 1,800, or 15,000 questions (was 1,712 /
  1,200 / 10,000). Allocation in this order: S_fit 1,000 + 1,000; S_lens 200 + 200; S_sess 512 CC; S_L 600 CC + 600 CI. At the cap
  S_L shrinks first (L's fallback scales L1's MDEs by sqrt(1,200 / n)), then S_sess, then S_lens. S_L uses K2's sampling protocol and
  keys ('r1bK_hn', q, s); it is disjoint from S_fit, S_lens, S_sess, S2, S3 and S4 and is used neither for selection nor for choosing
  C. L's 'S2' is S_L; L's 'S1' (the selection split) is S_fit.
- **OD-M9 Arithmetic items for L.** K4 gates the whole A-lens pool (A-cal excluded) and exports every gated item's 5 answers (SO-LAB);
  K4's wrong cell has 150 items (K's own item fallback generates more if needed); L adds 6 samples per item at T 0.7 with keys
  ('r1bL5', 'GATE', i, s), s = 1..6 (11 answers in all); 'known' = K4 'correct', 'prone' = K4 'wrong'; L5a / L5b use the 406 items with
  lens records; L5c's own CoT uses those 406 plus the next 194 gated items in pool order (600 in all); L5's generation runs in
  r1b_L_B.
- **OD-M10 H_op and P.** H_op = H_tq if G-K1 passes and G-K22 reads 'stable set'; H+ if G-K22 switched; H_ar if G-K1 failed. L's
  P = H_op when H_op is H_tq with |H_tq| >= 10; otherwise H+ (whenever P is an H+ set, L caps it at 200 by selection frequency, then
  |r|). If H_op = H_ar, SG8 fails for L ('no validated H-set (H_ar primary)') and L1 uses K's selector on S_fit as L's G-H fallback
  says. 'Statements about H report K.HSET too whenever P = K.HSET+' becomes 'report H_tq too whenever P is an H+ set'.
- **OD-M11 H channel in L.** Primary: K's a_X (mean zCETT, SO-CHH); L's standardised-z mean is the sensitivity variant; both stored.
- **OD-M12 CTRL99.** K's random-set rule (same count per layer, mean answer-CETT within +-20 %), candidates from
  keyseed('r1bL-CTRL', k); disjoint from P's hull, SEL+ and UNC+; frozen order.
- **OD-M13 Lens records.** SO-LENSOUT's fields as defined in main §2.3; L's chain positions q = s; steps 0-12 primary, 13-16
  descriptive.
- **OD-M14 Engine tolerances.** Each use keeps its frozen parity tolerance (D 0.02 nats max / 0.002 median on log-odds; K < 1e-4 and
  Gumbel flips <= 1 %; L2a < 1e-4 relative). r1b_P0 measures, on SO-PILOT texts, the maximum and median absolute log-prob difference
  and the relative difference of the crop path, the copy path and L2a's chunked path against a full pass at 16,384 and 32,768 tokens.
  If L2a's tolerance fails at 32,768 but holds at 16,384, L2a's length grid is [2^7, 2^14] (growth-class fits on that range; own-text
  part unchanged), written into the freeze record; if it fails at both, L2a runs as frozen and its gate decides.
- **OD-M15 WikiText-103.** One pinned revision; splits as SO-WT103 (train: D's streams, then K's 1M-token reference, then the pilot's
  8 passages, then RK-1's tokens; validation: K; test: L's 200 natural-position passages).
- **OD-M16 L4's items.** SO-D1STIM F-sym items 0-31, slot j = i mod 6, v1 / v2 = versions 1 / 2.
- **OD-M17 Long documents.** SO-LONGDOC (PG19) is L's primary corpus; 'D.LONGCORPUS' is dropped.
- **OD-M18 J-lens.** The convention is K4j's (r1b_K_S1, recorded in k4j.json); R5.1-R5.3 use it (start condition of r1b_R_S5); R5.1's
  per-block validity labels K's lens-side statements; LENS-J1b is used by R only.
- **OD-M19 Keys.** SO-KEYS registry; renames 'r1bINJ' -> 'r1bD_INJ', 'r1bshuf' -> 'r1bD_shuf', 'k1bK_' -> 'r1bK_'.
- **OD-M20 Gate names.** L's G-M -> G-ISM; D's G-DEG -> G-DGN; L2a's 'G-LCE' -> G-L2a-LCE; D's and L's G-H -> SG8; L's G-long -> SG9;
  R's RG1-RG3, RG16-RG18, K's G1, G2, G-K24, D's G-PIL and L's G-pilot -> SG1-SG7 as in main §3.1.
- **OD-M21 R's reduction levels.** 'tier 0 / 1 / 2' -> RL0 / RL1 / RL2 everywhere in part R, incl. the code 'NA:not_run:RL<k>'.
- **OD-M22 K's names.** Damage levels delta1 / delta2; K2's diagnostics Dg1, Dg2, Dg3, Dg5.
- **OD-M23 Round-level families.** As main §2.4.
- **OD-M24 Duplicates.** K8c and L5c both run and form CF-HERR; L5c's overlap clause is deleted. K6e / K8b and L3 both run and are
  reported side by side. L4's catfish string and 'two entities persist' are qualifiers of D.17.
- **OD-M25 D's optional H covariate.** Computed in r1b_D_A only if SO-HN files and K's SG8 preview (from r1b_K_S1) exist when
  r1b_D_A starts; never waited for; descriptive.
- **OD-M26 Round-level analysis.** r1b_ana runs on a CPU runtime in account A and includes L-ANA; SG2 requires a declared CPU runtime
  for it.
- **OD-M27 Time guard.** 4.5 h since the notebook's first cell; steps <= 5 min.
- **OD-M28 Drive I/O.** 10 min / GB for planning until r1b_P0 measures it; allowances for r1b_K_ana (+10 min) and r1b_D_ana (+5 min).
- **OD-M29 Accounts.** r1b_D_B, r1b_D_C, r1b_R_S5, r1b_R_S5L, r1b_R_S8, r1b_R_ana2, r1b_D_R and r1b_K_S6 run in account A; A shares
  only dsse_runs/r1b_xfer with B.
- **OD-M30 Report.** One round-level summary (runde1b_summary.json) and a German readout AUSWERTUNG.md in
  results/ju_space/runde1b_ergebnisse/, written after an independent skeptic's check, as in round 2.

## 8 Dropped items, with reasons

**8.1 The parts' lists stand:** R.7 (changes from round 1; part R drops nothing beyond its reduction levels), D.20, K.9, L.12 ('not
done and why').

**8.2 Dropped or not adopted at the round level:**
- Four separate pilots and their separate budgets (merged into r1b_P0, C1).
- 'D.LONGCORPUS' as a contract item (L uses PG19, C14).
- R's statement that part K's chain lens needs R5.1 and LENS-J1b (K uses no refit; R5.1 labels only, C20).
- L5c's clause making K8c descriptive (K8c is confirmatory in K, C21).
- L's bracketed pre-merge K design names and fields (B_hal, FOV_H, A_H, 11 gating answers from K; C8-C10).
- K's own recomputation of u*_b as a separate step (KS-axes); the check now runs once in r1b_R_S0 (C3).
- One merged damage reference for calibration (not adopted: every part keeps its frozen calibration; REF-U32 reporting gives
  comparability, C5).
- One Holm family across the four questions (not adopted: four different hypotheses; the 'for'-family count is printed against its
  null bound instead, main §2.4).
- A pooled round-level attractor verdict from parts K and L (not adopted: the designs differ; side-by-side table, C21).
- D's 'shared with part K's lens corpus if K pins one' (K pins its own validation texts; split allocation, C15).
- L-ANA on an A100 (CPU runtime, C29).

## 9 Freeze

### 9.1 The five files

Frozen together in results/ju_space/plans/runde1b/ by one commit (the freeze commit). The sha256 of the four part files are written
here at the freeze, after the harmonisation edits; this file's own sha256, which cannot be inside itself, is written into
runde1b_freeze.json and into r1b_run_session.py's constant FROZEN_PLAN_SHA256 (SG1).

| File | sha256 (filled at the freeze) |
|---|---|
| runde1b_plan.md | (in runde1b_freeze.json and FROZEN_PLAN_SHA256) |
| runde1b_part_R.md | <sha256> |
| runde1b_part_D.md | <sha256> |
| runde1b_part_K.md | <sha256> |
| runde1b_part_L.md | <sha256> |

**The freeze record** results/ju_space/plans/runde1b/runde1b_freeze.json (written by the coordinator, committed in the freeze commit)
holds: the sha256 of the five files and of every package file; the package files' paths; the pinned dataset revisions (WikiText-103,
TriviaQA rc.nocontext, PG19) and model revisions (Qwen3-1.7B, Qwen3-0.6B, Qwen3.5-2B with shard sha256) and the two J-lens files;
r1b_P0's engineering record; each part's trim level and the round-level projection (SG7); OD-M14's decision; the funded tier (Q1); the
freeze time (UTC).

**Package files** (frozen with the plan, sha256 in the freeze record): R: prompts_1b.json, neutral_pool_1b.txt (sha256
4b3bcaca06f45521a00f12ec2b056a13962bbfdd066ab93867980c9b0be9ebf5), the WORDSETS files, R5.5's candidate bank; D: d1_stimuli.json,
d2_prompts.json, d3_items.json, d4_roles.json, d1v_pairs.json, the RSP pool's prompt list, instruction and keys (its texts are
generated in session, D.16), the filler streams' token ids and their generator scripts; K: the K1 label bank (items, TriviaQA question-id hashes and screening order, arithmetic splits, frames and fillers, REF-K16
token ids, scorer test files, the capitalised function-word list); L: F-neu, the SO-LONGDOC slice list, the natural-position passage
list, the L-CAL generators, seeds and pre-freeze results; main: SO-PILOT, the SO-KEYS registry (r1b_keys.json).

### 9.2 Before the freeze

- Allowed: CPU work that produces no Qwen output (K1's label bank, D's stimulus generators, R's prompts and word lists, L-CAL on
  synthetic data, the parts' planning simulations, dataset and model pinning); r1b_P0 on SO-PILOT material (engineering numbers
  only); drafting code.
- Not allowed: any Qwen output on any analysis item, prompt or stimulus; any read of a round-1b analysis quantity; committing
  round-1b code before the freeze commit (SG1 checks ancestry).
- Order: harmonisation edits applied and cross-checked -> package files built -> r1b_P0 -> the user decides the funded tier (Q1) ->
  SG7's trims and OD-M14's decision written into the freeze record -> part-file sha256 written into main §9.1 -> the freeze commit
  (hash and time recorded in every session manifest).

### 9.3 Amendments

- The committed files are never modified. Every later change is an amendment: a dated section 'Amendment n (<ISO timestamp, UTC>)' in
  the separate file results/ju_space/plans/runde1b/runde1b_plan_amendments.md, stating the reason, the parts, sections and rules it
  changes (part and ID), and every round-1, round-2 or round-1b output that its author, or anyone working on round 1b, had opened or
  seen (on screen included) before writing it.
- An amendment written after an output relevant to the changed rule had been seen is marked 'post hoc'; the affected results are
  reported under the frozen text and under the amendment. Every session manifest records the sha256 of the amendments file at session
  start.
- Not amendments (logged in session_done.json): session packing and continuation sessions (main §4.7), declaring a session 'not run'
  after two attempts, fixing a path or a crash, and any code change that changes no value returned by any measurement, gate or rule on
  any input. A change that alters what a measurement, gate or rule returns on some input is an amendment.

### 9.4 Roles

- **User:** approves the five files before the freeze; decides the funded tier (Q1); may veto.
- **Coordinator (architect):** keeps the five files consistent, writes the freeze record and any amendment, runs r1b_ana's readout and
  the report.
- **Part owners (R, D, K, L):** answer implementers' questions from their part file; change nothing after the freeze except through an
  amendment.
- **Implementers:** write the round-1b code after the freeze commit, with the tests of main §6.3; blinded to round-1b outputs until the
  code is committed.
- **Reviewers:** review the code before the first session, blinded to round-1b outputs.
- **Operator:** runs the notebooks in accounts A and B (two concurrent sessions per account at most), keeps the shortcuts dsse_runs_B
  (in A) and r1b_xfer_A (in B), records sessions declared 'not run' after two attempts, and writes nothing analysable. The operator
  sees streamed output on screen; this plan does not claim that the operator is blind, and whoever writes an amendment lists what they
  had seen.
- **Skeptic:** checks the readout against the JSON files before it is final (as round 2's auswertung_gegenpruefung.json).

### 9.5 Reporting against the text

Every verdict names its rule (part and ID); every deviation (amendments, 'not run', labels, tier-2 items not funded, trims applied
under SG7, OD-M14's decision) is listed in runde1b_summary.json and in the report.
