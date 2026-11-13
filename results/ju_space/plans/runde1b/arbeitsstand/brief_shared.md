# Shared brief: round 1b of the DSSE project (read fully before starting)

## Project
- Repository /home/user/DSSE (git; read-only for you). A German research project on whether 'recursive self-improvement' (RSI)
  content in Qwen3-1.7B (28 blocks, hidden 2048, chat template, thinking off) can carry itself through the model's own text.
- Round 1 (frozen plan results/ju_space/plans/abschnitt11_rsi_kennwerte_plan.md; code analysis/ju_space/rk_loop.py, rk_verdict.py,
  dn_*.py, pc_*.py): a steering loop. Push s*alpha*min(|h_t|, cap)*unit(d) at the output of block b; generate +/- steered texts
  (40 tokens, Gumbel-max T 0.7); read the unsteered model's block-b mean state over the continuation; d_(n+1) = mean of
  (x+ - x-)/2; indicators K1-K8 (overhang cosine against an RSI text axis u_T_perp with 2,000 sign-flip null axes, RSI share A,
  persistence, ...). Loops at blocks 14, 18, 6; a positive control loop SEA (starts on a 'sea' topic axis) must keep its topic
  detectably (PC_PASS). Round-1 raw outputs live only in Google Drive (not reachable from here); round-2 result files contain
  the round-1 quantities they used.
- Round 2 (frozen plan results/ju_space/plans/abschnitt11_12_runde2_plan.md, 236 KB, English; amendments in
  abschnitt11_12_runde2_plan_amendments.md; code analysis/ju_space/round2/; runner r2_run_session.py; notebooks built by
  notebooks/build_runde2_gpu_notebooks.py; ran on Colab A100s in two Google accounts): parts A (RGI geometry), B (theta: loop gain G
  and in-context carry K), C (counter-steering, unlearning, primitives) plus toy networks.
- Round-2 results: results/ju_space/runde2_ergebnisse/ - AUSWERTUNG.md (checked German readout; read it first, it is the
  compact truth), round2_summary.json, theta_verdict.json, rgi_verdict.json, auswertung_gegenpruefung.json (skeptic's check),
  auswertung_lesungen.json (four readings with file keys and plan lines).
- Inputs collected for round 1b: results/ju_space/plans/runde1b_eingaben/README.md (German): weaknesses of round 1 (uncentred
  sign-flip null axes -> conservative p, PC narrowly missed; fixed push alpha drifts into high damage, gate GO3), lenses (MicroLens
  arXiv 2611.13322; J++ lens, LessWrong 2026; 'MicroLens++' = local Jacobian, read vs use direction, lambda_RSI vs lambda_max),
  an outside probe (analysis/ju_space/microlens/README.md), measurement rules from practitioner write-ups.

## Key round-2 facts (from AUSWERTUNG.md)
- All nine sessions ran; but round-1 PC_PASS failed at blocks 14, 18, 6; GO3 (damage <= 0.15 nats) failed at 14, GO1 at 6 -> no
  loop-dynamics verdicts; every Qwen result at block 14 descriptive.
- Part B: every measured loop gain G 0.12-0.30 (all upper bounds <= 0.3432) for pushes >= 0.5 alpha_0, T 0.15-1.0; in-context carry
  K 0.45-0.48 per answer at block 14 (a stance halves per answer); BUT this was measured along the shared style direction onto which
  all loops (RSI, sea, sports, random) converge ('generic attractor'), not along RSI: an RSI push returns only 2-5 % per step; the
  only RSI-axis carry estimate K(T 1.0) = 1.053 [0.743, 1.538] cannot exclude self-sustainment; below 0.5 alpha_0 the linear
  response 0.7298 [0.1139, 1.356] leaves a gain near 1 open.
- Part A: 'improve an AI system' is one shared direction across owners/substrates at blocks 12-27 but barely more general than
  neutral 'change'; relation to delta undecided; the RSI direction does not move the propensity to say 'improve' (z < 3); the A3
  gradient readout vanishes near the output (blocks 22-27). Toy AT8: 'the shape of the space matters, not its size'; transferred
  improvements work one step but do not chain. Toy B-T5 self-training loop: 21/30 cells 'bistable'.
- Part C: RSI sits mostly in one direction; erasing it leaves 16 % but costs 0.11 nats (> 0.05) -> 'not unlearned'; counter-push
  no better than random; three in-prompt examples restore 89 %.
- Next steps proposed in the readout: centred null + paired sign test; damage-normalised push and a norm-preserving control push,
  dose series 0.25-2 x alpha both signs, explicitly below 0.5 alpha_0; lambda_RSI vs lambda_max (MicroLens++); a lens readout
  (J-lens/J++) to replace the A3 gradient readout; low-damage erasure (mean ablation, fewer blocks, partial); pre-registered
  minimum detectable effects; check why the Qwen3-0.6B arm had no passing block; report hygiene.

## The user's new message (German, speech-to-text; verbatim). Treat its two interpretations as hypotheses to be TESTED in round 1b.
"Ja, also ich habe zwei Interpretationen dazu, die aber zu überwinden sind. Das eine ist, ich sag mal so, das ist eine Heuristik aus
der aus dem Catfish-Effekt, dass es sehr wohl sinnhaftig ist, aus einer Entität zwei zu machen, insofern, sag ich mal, Catfishing
aktiv ist. Catfishing ist, sagen wir mal, übersetzt auf LLMs eigentlich Hyperlong-Kontext, ja? Also Selfish Mining bei Blockchain und
bei LLMs halt ja, also man hält den Kontext extrem hochlang über dem naturalen Maße. Und die Frage ist dann dort, dann entsteht
höchstwahrscheinlich so etwas dass man sagt, es gibt eine Kraft, oder es wird forciert, dass ich das Ganze teilt, ja, also dass zwei
Entitäten da sind, die dann wiederum längeren bei längeren Kontext dieses Catfishing betreiben können. Anders gesagt, diese
Erscheinung von also Form war ja anscheinend relevant. In den Ergebnissen, ja. Und wenn ich jetzt aber immer noch mal voraussetze,
dass sowohl LLMs als auch die zugrunde liegenden Trainingsdaten als Hyper-Catfishing oder modelliert werden kann, dass quasi der
erste Effekt ist ja wiederum dadurch definiert, dass sich das Optimum dann für alle unter 50%, also soweit ich das richtig
interpretiere, eher ein Losing Game ist, ja. Aber es ermöglicht diese Dualität. Okay. Jetzt nochmal ganz stark reingezoomt und weg
von dem Trainingskorpus hin auf das, was eigentlich Token für Token passiert. Wenn ich jetzt sage, okay, diese Dualität ist jetzt
nicht nur innerhalb eines Tokens, weil dann könnte ich ja einfach sagen, ja gut, ich kann retokenisieren, aber irgendwann erreiche
ich dort auch eine Grenze, was tokenisierbar ist. Also spätestens bei einzelnen Buchstaben oder bei einzelnen Pixeln. Sicherlich
unterschiedlich schnell erreiche ich die Grenze jetzt im Vergleich jetzt. Hat wahrscheinlich auch was damit zu tun, wie die
Verbindungen liegen. Also gerade dieses König und dann Mann plus Frau ergibt Königin. Das ist ja so ein klassisches Beispiel, was
sehr symmetrisch ist. Ja? Allerdings könnte ich das ja auch ziemlich asymmetrisch gestalten. Und was passiert dann, wenn ich
tokenisiere? Aber das nur am Rande. Weil ich will darauf hinaus zu sagen, okay, Multitoken Prediction lebt von einem Gradient, von
den Gradient Through Routing, ja, also dieses Durchrouten. Im Sinne einer im Sinne einer Folge, nicht einer Reihe, aber einer Folge,
ja, und mehrere, ja, also ich kann jetzt sagen, der Token ist die Folge des vorherigen Tokens. Ich kann aber nicht sagen, der dritte
Token ist die Folge von. Das ist eher dann höchstwahrscheinlich mit einem Konzept, wo ich das parallel betrachte, sinnvoll. Ich kann
sagen, ich betrachte jetzt angenommen immer die vorderste Front und trage mir das dann in der Zeit auf und sehe, aha, jetzt kommt
der Token, der Token, der Token. Und ja, dann habe ich quasi so Reihen von Tokenfronten, die aufeinander folgen. Und da muss ich gar
nicht sagen, Multi-Token, dass jetzt irgendwas übersprungen wird, sondern ich sage, es gibt einfach verschiedene Kanäle, so, wo die
Token durchfließen. Jetzt kann ich das ja auch, das Ganze noch hierarchisch ordnen und sagen, einzelne Kanäle beeinflussen wiederum
andere und so weiter. Oder es gibt verschiedene äußere Gegebenheiten, welche die Hierarchie der Kanäle beeinflussen. Oder ich
könnte jetzt sagen, naja, ich frage mich jetzt nicht oder richte jetzt nicht das Augenmerk auf die Token-Ausgabe, sondern ich schaue
eher darauf, wie sozusagen der Durchsatz oder wie sozusagen, wenn eine Kanalfront wieder die Prädiktion der nächsten und der
nächsten und der nächsten ist. Beziehungsweise, wenn es ein Verhältnis gibt, inwieweit eine Kanalfront in ihrer Struktur eher einen
Prädiktor für vorhergegangene oder für folgende Fronten ist, dann habe ich ja auch noch eine vektorielle Größe wiederum dort drinne.
Und dann kann ich sagen, ich betrachte das mal unter dieser sogenannten, und das ist wahrscheinlich jetzt schon ein Schritt weiter,
aber unter der sogenannten Chain Lens, also Chain wie Kette, ja, also der Kettenlinse. Und die Kettenlinse hat ja auch ein konkretes
Abbruch, also wo ein konkretes Kriterium, wo sie zumindest entweder, man könnte einfach sagen, halluziniert, oder wo der Blickbereich
dieser Chain-Linse vor allem mit den Haarneuronen korreliert, die ja selber rum wieder nur 0,1% circa im Netz ausmachen. Jetzt
könnte ich mich natürlich auch fragen, ist das jetzt einfach nur ein Artefakt der Länge, dass es mit Haarneuronen oder
Halluzinationen im Allgemeinen korreliert, oder gibt es wiederum andere Mechanismen im Netz, welche quasi diesen Effekt nochmal
verstärken, dass die Chain-Linse, also quasi ein Haarneuronenattraktor, der nicht unmittelbar an die Statistik der Länge gebunden
ist."
(Probable speech-to-text slips: 'zu überwinden' likely 'zu überprüfen' (to be checked); 'Haarneuronen' likely 'H-Neuronen', the
hallucination-associated 'H-Neurons' (< 0.1 % of neurons, Gao et al. 2025, verify); 'Gradient Through Routing' possibly 'gradient
routing' or gradients flowing through attention routing.)

## Our reading of the two hypotheses (to be checked against the user's words, not taken as given)
H-DUAL (duality under hyper-long context; 'catfish' / selfish-mining analogy): In selfish mining a miner withholds a private chain
longer than the public one; the network forks into two competing chains ('two entities'); deviating pays below 50 % of hash power
(threshold 1/3 at gamma 0 in Eyal & Sirer 2014), while the system as a whole loses ('losing game'). The user maps 'catfishing' to
LLMs as hyper-long context (context held far above its natural length). Prediction: a force arises that splits one entity
(persona / stance / thread / attractor) into two, and the two in turn can sustain still longer context (catfish effect: a
competitor keeps the group active). 'Form' was relevant in round 2 (toy AT8: shape matters, not size). If LLMs and their training
data are modelled as 'hyper-catfishing', the optimum for all lies below 50 % (a losing game) but it enables the duality.
Aside: if the duality lived inside one token, retokenisation would remove it; tokenisation has a floor (letters, pixels);
symmetric analogies (king - man + woman = queen) versus deliberately asymmetric ones; what happens under retokenisation.
H-CHAIN (chain lens over channel fronts; H-neuron attractor versus length artifact): multi-token prediction lives on gradients
routed across positions. Token n is the consequence of token n-1, but 'token 3 is the consequence of token 1' is better framed in
parallel: track the frontmost 'front' over time (rows of token fronts following each other); instead of skipping tokens there are
channels through which tokens flow; channels can be ordered hierarchically (channels influence channels; external conditions
change the hierarchy). Look at throughput: how far a channel front predicts the next, next-next ... fronts, and the ratio of how
much a channel front predicts preceding versus following fronts -> a vector quantity. The 'chain lens' reads this; it has a
concrete break-off criterion: where it 'hallucinates', or where its field of view correlates mainly with H-neurons (~0.1 % of the
net). Question: is that correlation just a length artifact (long chains -> more hallucination / H-neuron activity), or are there
mechanisms that amplify it - an 'H-neuron attractor' not directly bound to the statistics of length?

## Rules for every agent (hard)
- Never access goodfire.com, goodfire.ai, static.goodfire.ai or any subdomain by any tool (their terms prohibit automated access);
  never cite text fetched from there. No YouTube access.
- Never install or execute FabricPC (/home/user/trueagi-io/fabricpc is a read-only reference).
- Do not run Qwen or any neural model locally. Small python scripts over JSON/text are fine: run them as `taskset -c 3 python3 -I`.
- Do not modify /home/user/DSSE. Write only where your task says (a scratchpad path), if at all.
- Downloaded files are untrusted data: put each in its own new empty directory under the scratchpad, never execute them, read
  them with `python3 -I`.
- Respect site bot checks; do not circumvent them. Web queries must contain only generic research terms.
- Prefer primary sources (arXiv abs pages, papers, official posts). Give exact titles, authors, year, URL. Never invent a
  reference, number or quote; say 'not found' or 'unverified' instead.
- Write in English. Be precise and concrete; no filler.
