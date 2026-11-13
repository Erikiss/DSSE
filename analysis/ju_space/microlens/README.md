# MicroLens++: die Achsen von Runde 1/2 durch zwei Linsen gelesen

Erweiterung zu `analysis/ju_space/` (Abschnitte 10–12, Runde 2). Idee: MicroLens (Zahedi et al., arXiv 2611.13322: Logit-Lens
und J-Lens auf demselben Raster, lesen – zurückverfolgen – editieren) plus die J++-Lehren (Ayonrinde & Lindsey, LessWrong
2026: Mittelwerte sind durch verzerrtes Rauschen verfälscht, Readout-Filter gegen nicht-semantische Token).

## Der strukturelle Denkfehler

Jede Richtung des RSI-Programms wird in **einem** Rahmen gebaut, verglichen, orthogonalisiert, geschoben und gelöscht: im rohen
Residualstrom von Block b, gemittelt über alle Fortsetzungs-Token, mit dem **euklidischen** Skalarprodukt
(`rk_loop.axes`: `u_T_perp = u_T − Projektion auf span(sports, tone, sea, NEG)`; `rk_loop.stats`: `c = cos(d, u_T_perp)`,
`cos_sports/tone/sea`; in Runde 2 `r2_rk.WriteOrth`, die festen Achsen von O1 und die „foreign axes“-Bedingung von C-Q2/C-Q3).
Das Modell selbst liest den Zustand aber nicht euklidisch, sondern durch eine stark anisotrope Abbildung (J-Lens:
`W_U · diag(g) · J_b`). „Euklidisch orthogonal“ heißt deshalb nicht „für das Modell unabhängig“. Weil die Fremdkosinusse
euklidisch gemessen werden, sind sie durch die Konstruktion null – das Leck ist in keiner Kennzahl sichtbar.

## Messung (`mlpp_probe.py`, Qwen3-1.7B, Block 14, CPU, 64 Prompts von Runde 1, 40 Token, Schlüssel wie rk_loop)

Die Achsen sind mit den Prompts, Instruktionen, NEG-Paaren und dem Gumbel-Schlüsselschema aus `rk_loop.py` neu gebaut (keine
Runde-1-Ausgabe gelesen; in dieser Umgebung lag keine vor). J-Lens: `neuronpedia/jacobian-lens`, wie in Abschnitt 1.
Ergebnis: `results/ju_space/microlens/mlpp_probe_b14.json`.

| Test | Ergebnis |
|---|---|
| T3 Fremdanteil von `u_T_perp` in der J-Lens-Metrik (zentrierte, gefilterte Readout-Gram) | **0,345**; zufällige euklidisch-senkrechte Richtungen: Mittel 0,090, 95 %-Quantil 0,204. Kosinus mit sports/tone/sea **−0,49 / −0,55 / −0,47**, mit NEG +0,37 (euklidisch jeweils 0) |
| T3 dasselbe in der Logit-Lens-Metrik | 0,032 (Zufall 0,005, 95 % 0,013): klein, aber ebenfalls über Zufall |
| T4 Gebrauch, teacher-forced, gleicher Schaden 0,05 nats: `+u_T_perp` (Runde 1) | RSI-Wörter +0,74, sports −0,16, sea −0,14, tone −0,06 (log-Wahrscheinlichkeit der Wortmenge) |
| T4 `−u_T_perp` (= Gegensteuern, Teil C) | RSI −0,53, aber sports **+0,39**, sea **+0,33**, tone **+0,30** |
| T4 `u_T_perp^J` (senkrecht in der J-Metrik) | +: RSI +0,72, sports +0,11, sea +0,13, tone +0,78; −: RSI −0,59, sports +0,05, sea +0,01, tone −0,64 |
| T4 roher `u_T` | RSI +0,87, aber tone +1,40, sports +0,59: unspezifisch, wie erwartet |
| T2 Filter auf semantische Token / Norm ≤ 2·Median | cos(alle, gefiltert) 0,991 für `u_T_perp`; T4 praktisch unverändert. Der J++-Positionsfilter ändert hier wenig (12–16 % nicht-semantische Token) |
| T1 Top-Wörter von `u_T_perp` | Logit-Lens: myself, understand, learned, tutor …; J-Lens: analysis, learning, research, knowledge, inquiry …; Überlappung der Top-15: **1 Wort** |

Deutung:
1. Die euklidische Orthogonalisierung zieht die Fremdthemen nicht nur ab, sie **zieht zu viel ab**. `u_T_perp` ist in der Metrik,
   mit der das Modell liest, „RSI minus Sport/Meer/Ton“. Beim Schieben senkt `+u_T_perp` die Fremdwörter, und
   **Gegensteuern mit `−u_T_perp` schiebt das Modell aktiv zu Sport, Meer und Ton**. Ein „Verlernen“ oder „Unterdrücken“ von
   RSI über diese Achse ist damit zum Teil ein Hinschieben zu den Kontrollthemen. Die Kontrollen, die das ausschließen sollen,
   messen euklidisch und sind daher blind dafür. Dass das kein bloßer Umverteilungseffekt ist (weniger RSI-Masse → mehr Masse
   anderswo), zeigt `−u_T_perp^J`: RSI sinkt ähnlich stark (−0,59), Sport und Meer bleiben bei +0,05/+0,01.
2. Ein Spezifitätsmaß wie „RSI-Effekt minus größter Fremdeffekt“ belohnt diese Anti-Komponente: `u_T_perp` schneidet scheinbar
   bestens ab (0,79/0,83), die neutralere J-senkrechte Achse schlecht (−0,06). Die richtige Forderung ist „Fremdeffekt ≈ 0“,
   nicht „Fremdeffekt möglichst klein/negativ“.
3. Die J-senkrechte Achse ist neutral für Sport und Meer (|Effekt| ≤ 0,13), **nicht** für Ton (±0,7). RSI-Texte und
   enthusiastischer Ton sind im Lesen des Modells verflochten. Eine Orthogonalisierung erster Ordnung allein (J-Lens ist ein
   gemittelter Jacobian) reicht dafür nicht.
4. Logit-Lens und J-Lens lesen dieselbe Achse fast disjunkt (1/15 gemeinsam). Wie bei MicroLens: Was ein Vektor an seinem Token
   „ist“, ist nicht das, was spätere Token daraus lesen. Die Rückkopplungsschleife (`d_(n+1)` = gelesener Zustand der Texte,
   wieder als Schub benutzt) vermischt genau diese beiden Rollen.

Grenzen: ein Block (14), ein Modell, ein Lauf ohne Bootstrap über Prompts für T4, Wortmengen von Hand gewählt, J-Lens von
Neuronpedia (nicht J++: kein Expert-Jacobian-Filter, kein LRP – dafür braucht es eine GPU). Die Achsen sind nachgebaut, nicht die
Runde-1-Zustände. Die Effekte in T4 sind Mittelwerte über 64 Fortsetzungen.

## Was daraus für Runde 2 folgt (Vorschlag, kein Eingriff in den eingefrorenen Plan)

- **Zweite Metrik für jede Fremd-/Spezifitätsbedingung**: Kosinus und Projektionen zusätzlich in `M_J = R_Jᵀ R_J`
  (zentriert, gefilterter Wortschatz) berichten. Das betrifft O1/O2 (perp-Achsen), C-Q2/C-Q3 (foreign axes, OD-C6) und die
  `cos_*`-Zeilen der Schleifen.
- **Spezifität zweiseitig**: Fremdeffekte im Gebrauchstest müssen im Intervall um 0 liegen, nicht nur unter dem Konzepteffekt.
- **Les- und Schreibrolle trennen** (MicroLens): Richtung zum Lesen = Kovektor `J_bᵀ g W_Uᵀ e`, Richtung zum Schreiben =
  Vektor. Die Schleife sollte die Antwort vor dem Zurückspeisen in denselben Raum transportieren oder beide Varianten laufen.
- **J++ auf GPU**: Expert-Jacobians (k-means auf Block-b-Zuständen), gewichtet auf den eigenen Achsenaufgaben, plus LRP-Rückweg.
  Damit T3/T4 wiederholen und auf alle drei Blöcke von B3 ausdehnen.

## Nachrechnen

```
python analysis/ju_space/microlens/mlpp_probe.py <root> results/ju_space/microlens/mlpp_probe_b14.json   # NP=32 NNEW=40, ~28 min auf 4 Kernen, ~13 GB RAM
```
`<root>` braucht `models/qwen3-1.7b` und `jlens/Qwen3-1.7B_jacobian_lens.pt`. Kurztest: `NP=2 NNEW=8` (~2 min).

## Gegenprüfung (13.11.2026, Koordination; nicht vom Autor des Probelaufs)

Drei Prüfer (Aufbau, Zahlen, Folgen für Runde 2) und ein abschließender Skeptiker haben Skript, JSON und Behauptungen geprüft,
ohne das Modell neu zu rechnen. Alle Zahlen des JSON ließen sich aus dem JSON nachrechnen. Volltext: `results/ju_space/microlens/mlpp_probe_check.json`.

| Aussage | Urteil | Kern |
|---|---|---|
| T3: Fremdanteil 0,345 in der J-Metrik | teils gestützt | Rechnung stimmt. Die vier Fremdachsen sind in M_J fast eine Richtung (größter Eigenwert ihrer 4×4-Korrelation ≥ 2,61), also ein gemeinsamer Anteil, nicht drei. Vergleich nur mit isotropen Zufallsrichtungen, ohne Kreuzvalidierung und Intervall. |
| „Euklidisch zieht zu viel ab“ | teils gestützt | Relativ zur M_J-Projektion gilt das per Konstruktion; aussagekräftig nur, wenn M_J der richtige Maßstab ist. |
| T4: −u_T_perp schiebt zu Sport/Meer/Ton | teils gestützt | 29–41 % der genannten Werte steigen bei Schub in jede Richtung (gerader Anteil, auch bei RSI-Wörtern). Der vorzeichentreue Anteil ist 0,18–0,27 nats gegen 0,63 bei RSI-Wörtern. Keine Nullrichtungen, keine neutralen Wortmengen, keine Intervalle, kein erzeugter Text. |
| J-senkrechte Achse neutral für Sport/Meer | teils gestützt | Für diesen Lauf ja; beim Ton nicht (vorzeichentreu +0,71). Die Tonmenge enthält „!“, das die Metrik ausschließt. |
| Signierte Spezifität belohnt den Gegenanteil | teils gestützt | Gilt für das eigene Maß des Probelaufs; Runde 1 (K8) und Runde 2 (OD-C6) messen zweiseitig. |
| Logit- und J-Lens lesen fast disjunkt (1/15) | teils gestützt | Zählung stimmt; nach Wortfamilien größere Überlappung, keine Vergleichsbasis. |
| Positionsfilter ändert wenig | gestützt | Kosinus 0,99; T4 innerhalb 0,03. |
| Fremdkontrollen „blind“ | nicht gestützt | Null per Konstruktion sind nur Kosinusse von Richtungen. Die entscheidenden Kontrollen (K8, K5 in Runde 1; OD-C6 und C-Q2 in Runde 2) messen zweiseitig an erzeugten Texten oder Gradienten. |

Folgen:
- Runde 2 bleibt unverändert; keine M_J-Orthogonalisierung, keine M_J-Kriterien. C-Q2 prüft die textnahe Form derselben Frage bei
  gleichem Schaden (0,05 nats).
- Nach Runde 2, als nachträglich gekennzeichnete beschreibende Auswertung: signierte Fremdanteile aus C-Q2 (c) mit Prompt-Bootstrap,
  neben dem Anteil, den der RSI-Inhalt allein erwarten lässt.
- Für Runde 1b: T4 mit Nullrichtungen bei gleichem Schaden, mindestens 20 frequenzgleichen neutralen Wortmengen, Prompt-Bootstrap,
  Positivkontrolle je Fremdachse, ohne „!“ in der Tonmenge und mit erzeugtem Text; T3 gegen Text-Kontrast-Nullachsen und
  kreuzvalidiert; vorher die Brücke prüfen (J-Lens-Vorhersage der Wortmengen-Verschiebung gegen die gemessenen vorzeichentreuen Anteile).
