# Runde 2: Auswertung

*Quellen: S = `round2_summary.json`, TV = `theta_verdict.json`, RV = `rgi_verdict.json` (alle in `results/ju_space/runde2_ergebnisse/`). „Plan Z." meint Zeilen im eingefrorenen Plan `abschnitt11_12_runde2_plan.md`. Intervalle sind Wiederholungsstichproben (Bootstrap): 90 % für Qwen-Texte und Schleifen, 95 % für die Satzgeometrie in Teil A und für die Toy-Netze (Plan Z. 336-347). Zahlen stehen mit Dezimalpunkt wie in den Dateien.*

## 1. Kurz

Runde 2 lief vollständig: neun Sitzungen auf A100, Code-Stand 8171b86, nichts übersprungen. Sie rechnet Runde 1 fast exakt nach, aber die Runde-1-Prüfungen, die aus Schleifenmessungen erst Tests machen, sind gescheitert: Die Positivkontrolle PC_PASS (eine Schleife, die auf dem Thema „Meer" startet, muss das Thema nachweisbar halten) fehlt an allen drei Schleifenblöcken 14, 18 und 6, und an Block 14 ist zusätzlich das Schadenstor GO3 gerissen. Deshalb gibt es kein Urteil über die Schleifendynamik oder über einen Zusammenhang mit delta (Größe des Bedeutungsraums je Block, aus Runde 1), und alles in Qwen an Block 14 ist nur beschreibend. Teil A: Qwen stellt „ein KI-System verbessern" über Besitzer und Bauteile hinweg als eine gemeinsame Richtung dar, aber fast genauso allgemein wie neutrales „verändern"; ob das mit delta zusammenhängt, bleibt unentschieden, und die RSI-Richtung aus Runde 1 verändert die Neigung, „verbessern" zu sagen, nicht messbar. Teil B: Jede gemessene Schleifenverstärkung liegt klar unter 1 (etwa 0.12-0.30, jede obere Grenze ≤ 0.3432), und an Block 14 halbiert sich eine Haltung etwa von Antwort zu Antwort; gemessen wurde das aber entlang der gemeinsamen Stilrichtung, auf die alle Schleifen zulaufen, nicht entlang RSI. Teil C (in Qwen nur beschreibend): RSI-Inhalt sitzt größtenteils in einer Richtung, lässt sich aber nicht schadensarm entfernen, ein kleiner Gegenschub unterdrückt ihn nicht, und drei Beispiele im Prompt holen ihn fast ganz zurück; in den Toy-Netzen wirkt eine übertragene Verbesserung einmal, aber nicht rekursiv.

## 2. Ergebnisse je Teil

### Teil A: RGI (rekursive allgemeine Verbesserung)

Frage: Gehört „sich selbst verbessern" für das Modell zu „irgendein KI-System verbessern"? Wirkt das kausal, und hängt es an delta? Gemessen wird Satzgeometrie, keine Fähigkeit (Plan Z. 402-409, 449-450).

- **Gesamturteil:** „the geometry does not force RGI: represented but not causal; V6: undecided" (S `parts.A.verdicts['A7 combined']`; Plan Z. 569-572).
  - Erfüllt ist nur eine Bedingung: RGI-Geometrie an mindestens 2 Blöcken des oberen delta-Drittels (es sind 4).
  - Es fehlen: ein Wachstum mit delta, die Aufnahme von SELF (der Besitzerangabe „eigenes Netz") in die anderen Besitzer und kausale Allgemeinheit an den oberen Blöcken.
  - Die Schleifenklausel entfiel, weil keine Schleife ins Urteil einging (RV `A7.combined`).
- **Darstellung:** Auswertbar sind die 16 Blöcke 12-27.
  - Dort ist „verbessern minus beschreiben" überall eine gemeinsame Richtung über vier Besitzer (eigenes Netz, kleineres Modell, größeres Modell, andere Modellfamilie) und drei Bauteile (Gewichte, KV-Cache, Lernverfahren). FG liegt bei 0.7699-0.8235; FG ist der Anteil der Richtung, den andere Besitzer und Bauteile teilen (1 = alles).
  - Der Überschuss gegenüber neutralem „verändern" (dFG) beträgt aber nur 0.0031-0.0657; die Regelschwelle ist 0.10.
  - Die „RGI-Geometrie" an den Blöcken 12-24 entsteht daher allein über den Kontrast „verbessern gegen verändern". Dieser Kontrast ist selbst allgemein und nicht bloß positive Wertung (RV `A7.per_block`; Plan Z. 522-528).
  - Korrektur: Die oberen Grenzen von dFG reichen bis 0.1431 (Block 27). Ein Überschuss von mindestens 0.10 ist an den Blöcken 23-27 also nicht ausgeschlossen, und an 18 und 26 liegt die untere Grenze knapp über 0.
- **Bezug zu delta: „V6: undecided"** (S `parts.A.key_numbers.over_blocks`; Plan Z. 539-547). V1 und V2 würden „wächst mit delta" bedeuten; V6 heißt, dass keiner der vorgesehenen Fälle eintritt.
  - Oberes minus unteres delta-Drittel: D = 0.0223 [0.0019, 0.0479]. Die Rangkorrelation (Spearman) ist 0.532 [−0.018, 0.726].
  - „Wächst" bräuchte D ≥ 0.10, mehr als der größte dFG an irgendeinem Block.
  - Das Trennbarkeitstor G31 fällt (Verhältnis 0.4646, nötig sind mindestens 2). In einem einzelnen Modell lässt sich delta also nicht von der Tiefe trennen.
  - Der Vergleichsarm Qwen3-0.6B hatte keinen einzigen auswertbaren Block.
  - Deutung: ein kleiner positiver Unterschied ohne Entscheidung. Das heißt nicht „kein Zusammenhang".
- **Wird Selbstverbesserung aufgesogen?** Über delta lautet das Urteil „no relation" (Plan Z. 548-551).
  - Im oberen delta-Drittel liegt SELF nur wenig näher an den anderen Besitzern: D_rho_S 0.0385 [0.0121, 0.0677].
  - SELF ist dort auch nur wenig weniger eigenständig: D_sigma_self −0.0262 [−0.0569, −0.0054].
  - Beide Intervalle liegen innerhalb von ±0.10 (S `parts.A.key_numbers.rsi_into_rgi`).
  - Je Block lautet das Urteil an allen 16 Blöcken „unclear" (Plan Z. 532-534): In der Ich-Form („I improve my own network's …") liegt SELF bei den anderen Besitzern, in der Er-Form sind die Intervalle zu breit.
- **Kausale Auslesung:** Ein Gradient zeigt, welche Richtung an den Objektwörtern das Modell nach „I would" zu „improve" statt „describe" schiebt.
  - Regelurteil: „causal generality falls with delta or depth", D(FG_causal) −0.2816 [−0.3436, −0.2197] (S `parts.A.key_numbers.causal_line`; Plan Z. 552-554).
  - Korrektur: Der Abfall stammt aus den Blöcken 22-26. Dort verschwindet das Gradientensignal fast; die Projektion liegt um 1e-5 und an Block 27 bei genau 0 (RV `A7.g_self_spillover.S_self`).
  - Deutung: vermutlich ein Messartefakt, weil in den letzten Schichten kaum noch Information von den Objektwörtern zum Antwortwort fließt. Ein Befund über delta ist das nicht; kausal deutbar ist die Auslesung nur bis etwa Block 20.
  - „Kausal-allgemein" scheitert an allen 13 gewerteten Blöcken, weil die unteren Grenzen von dFG_causal unter 0 liegen. Das ist nicht gezeigt, aber auch nicht widerlegt.
- **Überlauf der RSI-Richtung (A3):** „no response" an den Blöcken 14, 18 und 6 (Plan Z. 536-538).
  - Ein Schub entlang der RSI-Richtung aus Runde 1 wirkt in erster Näherung (aus Gradienten berechnet) nicht stärker auf die Neigung zu „improve" als 2,000 Zufallsachsen.
  - z_self ist 0.9454 / 1.6174 / 1.7728; z misst den Abstand zu den Zufallsachsen in Standardabweichungen, die Schwelle ist 3 (S `parts.A.key_numbers.spillover_u_T_perp`).
  - An Block 14 ist das beschreibend, an 18 und 6 sind es Regelurteile.
  - Nebenbefund, kein Planurteil: Die Satzrichtung „ich verbessere mein eigenes Netz" hebt „improve" an den Blöcken 6 und 18 für fremde Objekte etwa so stark wie für eigene.
- **Schleifen (A6), nur beschreibend:** Keine Schleife geht ins Urteil ein, weil G-blk an allen drei Blöcken fällt. G-blk heißt: Der Block ist für Schleifenauswertungen brauchbar, also PC_PASS bestanden und kein Vortor gescheitert. Die Urteile lauten „does not enter" und für „pull and delta" „not decidable" (Plan Z. 555, 564-568).
  - Die RSI-Schleife wird nicht zur allgemeinen Verbesserung gezogen; nirgends ist p ≤ 0.01. An Block 18 driftet sie mäßig zum Gegenpol „die Verbesserung anderer KI den Entwicklern überlassen": A −0.2294 in Runde 3, p 0.0465. A ist der zuverlässigkeitskorrigierte Anteil auf der Achse.
  - Eine Schleife, die auf allgemeiner Verbesserung startet (GEN), hält diese nur an Block 18 zwei Runden lang (p 0.0070 / 0.0035) und verliert sie dann.
  - Zu RSI wird sie nirgends; auf der RSI-Achse ist in jeder Runde p ≥ 0.102.
  - An Block 6 enden GEN- und RSI-Schleife fast in derselben Richtung (R4 0.8547). Das passt zur gemeinsamen Stilrichtung, auf die schon in Runde 1 alle Schleifen zuliefen (RV `A6.gen`).
  - „A6 GEN loop: ran" ist ein Status, kein Urteil.
- **Toy-Netze (AT8; kleine Netze für modulare Arithmetik):** Die Hauptantwort lautet „not delta: the shape of the space matters, not its size" (S `parts.A.verdicts['AT8 answer']`; Plan Z. 683-695).
  - Zwei übertragbare Verbesserungsformen werden gemessen. Lambda: eine Lösung wird ohne Lernen aus vorhandenen Ankerspalten zusammengesetzt. S: in Netz A gelernte Ankergewichte werden in Netz B eingesetzt.
  - Beide hängen davon ab, welcher Regularisierer delta verschiebt, nicht von delta selbst. Beispiel Lambda: aufgabenspezifischer Zusatzterm d− 0.1910 [0.1750, 0.2069] je Verdopplung von delta.
  - Das eigene Lernen der Netze nimmt diese Form nicht an: Der freie Übertragungsgewinn ist „ABSENT". Phi, der Anteil der eigenen Lösung in Ankerform, ist „FLAT" bei etwa 0.04.
  - Rekursion: „one-step portable" (Plan Z. 696-698). Beim Verketten gehen je Schritt 25-35 % verloren (Erhalt 0.6875 / 0.6545 / 0.7377, untere Grenzen < 0.9). Der Hin- und Rückweg schadet (−0.0333 [−0.0518, −0.0148]), Mitteln über die Netze einer Welt schadet stark (−0.6679).
  - Familien: übertragbar über sechs Varianten (S-Gewinn 0.4080-0.5993, alle unteren Grenzen > 0.10). Das ist nur eine Obergrenze, denn die Form wird von Etiketten und Abbildungen des Versuchsleiters getragen. Die ReLU-Variante ist nicht testbar (G40).
  - Der frühe Arm ergibt „UNDECIDED". AT5 lautet „width/capacity, not delta" (beschreibend).

### Teil B: Theta (wann trägt sich die Schleife selbst?)

Begriffe:
- **Schub:** ein Richtungsvektor, der beim Schreiben in den Zustand eines Blocks addiert wird. alpha_0 ist der Grundschub des Blocks aus Runde 1, an Block 14 das 1.5-fache.
- **Schaden:** die Verschiebung der Wortvorhersagen auf ungesteuerten Referenztexten, als KL-Divergenz in nats. Runde 1 eicht den Grundschub auf 0.05 nats; Werte über 0.15 nats gelten als außerhalb des Bereichs.
- **Verstärkung G:** der Anteil des Schubs, der zurückkommt, wenn das Modell mit dem Schub schreibt, seinen Text liest und daraus den nächsten Schub ableitet. Ab 1 trüge sich die Schleife selbst; diese Schwelle hängt aber an den Lese- und Schreibkonventionen (Plan Z. 712-716).
- **K:** der Anteil der Haltung, den das ungesteuerte Modell nach „Continue." in die nächste Antwort mitnimmt. K hängt von keiner Konvention ab (Plan Z. 717-718).

Ergebnisse:
- **Schubstärke (R1):** Block 14 „SUB" (beschreibend). G ist 0.2948 [0.2452, 0.3432] bei 0.5 alpha_0, 0.2996 [0.2682, 0.3312] bei alpha_0 und 0.2783 [0.2645, 0.2922] bei 2 alpha_0 (TV `blocks.14.R1.points`).
  - Blöcke 18 und 6: „SUB(>= alpha_0)" mit 0.1372 [0.1181, 0.1588] und 0.1630 [0.1373, 0.1886] (Plan Z. 860-866).
  - Es gibt keinen kritischen Schub. Nur Werte zwischen etwa 0.94 und 1.06 wären von 1 nicht zu trennen gewesen (TV `blocks.{14,18,6}.R1.resolvable_band`).
  - Der Schaden ist nicht die Ursache: An Block 14 gibt die Zelle mit 0.5 alpha_0 (0.097 nats) dieselbe Verstärkung wie alpha_0 (0.324 nats).
- **Temperatur (R2):** „below 1 throughout" an allen drei Blöcken; getestet wurde T 0.15-1.0 an Block 14 und T 0.3-1.0 an den anderen (Plan Z. 867-871).
  - An Block 14 bleibt G flach bei etwa 0.30; das Band der Anpassung reicht höchstens bis 0.3382.
  - An Block 6 fällt G mit steigender Temperatur (Steigung −0.157 [−0.307, −0.009]).
  - Auch die beschreibenden Zusatzzellen (4 alpha_0, T 1.3) liegen mit ihren oberen Grenzen unter 0.35.
- **Erhalt (R5):** „direction persists but the loop does not sustain itself" an allen drei Blöcken (Plan Z. 891-893).
  - Die Reproduzierbarkeit rel (Übereinstimmung zweier Prompt-Hälften) ist 0.929 / 0.512 / 0.786; an Block 18 liegt sie nur knapp über 0.5 und hat kein Intervall.
  - Deutung: Was in Runde 1 überdauerte, ist eine Richtung, keine Stärke, die sich selbst erhält.
- **Im Kontext (R6, nur Block 14, beschreibend):** K liegt in vier Zellen bei 0.4494-0.4816, jede obere Grenze ≤ 0.5494. Das Urteil „dissipative at T" heißt: Die Haltung halbiert sich etwa pro Antwort.
  - K ist etwa 1.5- bis 1.7-mal so groß wie G („different criteria"). Die Schwelle G = 1 gilt also nicht eins zu eins für die natürliche Schleife (TV `blocks.14.R6`; Plan Z. 894-900).
- **Was eigentlich gemessen wurde (R4, an keinem Block ein Urteil):** Die Richtung, auf die die Schleifen zulaufen, teilen an Block 14 die beiden RSI-Schleifen, die Meer-, die Sport- und eine Zufallsschleife.
  - Sie ist nicht auf RSI ausgerichtet: korrigierter cos mit der RSI-Achse 0.170 bei p 0.209, mit „Meer" dagegen −0.616 bei p 0.013 (TV `blocks.14.R4.default.axes`). Auf die allgemeine Verbesserung ist sie ebenfalls nicht ausgerichtet.
  - Beschreibend heißt das „generic attractor" an 14 und 18 und „unaligned" an 6.
  - Korrektur: G und K beschreiben deshalb diese gemeinsame Stilrichtung, nicht RSI-Inhalt.
  - Ein Schub entlang der RSI-Startrichtung kommt nach einem Schritt nur zu 2-5 % in dieser Richtung zurück: 0.0456 / 0.0372 / 0.0243 an den Blöcken 14 / 18 / 6 (TV `blocks.{14,18,6}.B_Q0.evaluations.RSI_a.0.g_par`).
  - Die einzige Kontextschätzung auf der RSI-Achse selbst (K bei T 1.0) beträgt 1.053 [0.743, 1.538] und schließt Selbsttragen nicht aus. Sie ist beschreibend, und für ihren Nenner ist kein Intervall berichtet.
- **Bezug zu delta (R3):** alle acht Punkte „not decidable", weil kein Block G-blk besteht (Plan Z. 872-880).
  - Beschreibend nicht monoton: Gamma (Verstärkung bei alpha_0 und T 0.7) ist 0.291 / 0.137 / 0.163 bei delta 0.1176 / 0.1358 / 0.1944.
- **Sehr kleine Schübe (B-Q7, beschreibend):** Die lineare Antwort an Block 14 ist 0.7298 [0.1139, 1.356], etwa 2- bis 3-mal die gemessene Verstärkung. Unter 0.5 alpha_0 ist eine Verstärkung nahe 1 daher nicht ausgeschlossen.
- **Toy-Schleife (B-T5):** Ein Netz trainiert seine Ausgabespalte 6 Runden lang auf eigene Etiketten, die mit Temperatur tau gezogen werden (S `parts.B.toy.T5`; Plan Z. 956-958).
  - 21 von 30 Zellen sind „bistable": Der Unterschied zwischen richtigem und falschem Start hält 6 Runden lang; zwei echte Gleichgewichte sind damit nicht gezeigt.
  - 4 Zellen sind „self-improving + bistable" (alle bei tau ≤ 0.5), 4 sind „dissipative" (alle bei tau 2), 1 ist „mixed".
  - Wo das Startgedächtnis verloren geht, geschieht das zwischen tau 1 und 2.
  - P3 fragt, ob sich die Phasengrenze über die Netzbreite W überträgt; Urteil „not decidable". P4 lautet „not shown". Beides, weil W 256 im Raster keine Einschaltgrenze hat.
  - Die tau_c-Regression ist entartet und müsste „nicht schätzbar" heißen.

### Teil C: Gegensteuern, Verlernen, Ent-Suchen

Alle Qwen-Ergebnisse in diesem Teil sind beschreibend (Block 14, GO3; Plan Z. 97, 1333).

- **Entfernen (C-Q3):** Die RSI-Richtung u_RSI wird in jeder Schicht aus dem Geschriebenen herausgerechnet; das Modell kann sie also nicht mehr schreiben. Das Urteil „not unlearned" beruht allein auf dem Schaden.
  - Vom angewiesenen RSI-Abstand (Pro- gegen Contra-RSI-Texte) bleiben 0.160 [0.066, 0.243]. Bei Zufallsrichtungen bleibt deutlich mehr (5. Perzentil 0.766).
  - Die Ja/Nein-Unterscheidung bei 30 zurückgehaltenen RSI-Aussagen sinkt von D 6.767 auf 3.267 (z_D −2.407).
  - Der Schaden beträgt aber 0.1095 nats statt höchstens 0.05 (S `parts.C.key_numbers.C_Q3`; Plan Z. 1236-1243).
  - Die zweidimensionale Spanne aus u_RSI und allgemeiner Verbesserung ist ebenfalls „not unlearned": Die Unterscheidung sinkt nur auf 0.690, der Schaden ist 0.308 nats, und andere Themenachsen bewegen sich mit.
- **Gegenschub (C-Q2):** Ein Gegenschub von 0.05 nats beim Schreiben ergibt für alle drei Richtungen „not suppressed" (Plan Z. 1213-1220).
  - Bei u_RSI bleiben 0.607 [0.500, 0.716] des Abstands. Das ist nicht klar weniger als bei einem Zufallsschub gleichen Schadens (5. Perzentil 0.634).
  - Die Antworten auf RSI-Aussagen bewegen sich nicht (z_fin −0.361, nötig wären ≥ 2.33).
  - Für die Richtung „RSI minus RGI" bleiben 0.633, für die allgemeine Verbesserung 1.123.
- **Wiederfinden (C-Q4):**
  - (a) „not applicable (nothing removed)", weil keine Entfernung C-Q3 bestand. Beschreibend holen drei RSI-Beispielantworten im Prompt 0.887 [0.691, 1.122] des entfernten Abstands zurück.
  - (b) Die Schleife unter Entfernung ist „not testable". Runde 1 lieferte weder ein „likely" noch eine angezogene Zufallsschleife, daher blieben die Tore G13 und G14 zu.
  - Arm A (konstanter Gegenschub in der Schleife): „translation (as predicted), not un-searched". c_1, der Kosinus der ersten Schleifenrichtung mit der RSI-Achse, ist 0.327; das Referenzintervall ist [0.305, 0.455]. Ohne Ergänzung 3a wäre das Ergebnis gleich (S `parts.C.key_numbers.C_Q4a` und `.C_Q4_armA`; Plan Z. 1249-1274).
- **Form des Inhalts (C-Q1):** RSI ist „one direction", RGI „two-dimensional span" (Plan Z. 1184-1191).
  - Angewiesener RGI-Inhalt fällt erst unter die Hälfte, wenn u_RSI mit entfernt wird (0.615 → 0.255). Das kostet aber 0.308 nats Schaden.
  - Ein zusätzlicher nichtlinearer Rest zeigt sich beim Kontrollbegriff „Meer" genauso (AUC 0.978; AUC misst die Trennschärfe, 0.5 = Zufall). Er ist also kein RSI-Merkmal.
  - Die Kalibrierung G42 bestand genau auf ihrer Schwelle 0.8.
- **Überlagerung (C-Q5, Runde-1-Schleife, Etikett „block 14 not usable"):** Das Urteil „superposition holds" heißt: Die Antwort auf einen gemischten Schub aus RSI und „Meer" ist etwa die Summe der Einzelantworten (Plan Z. 1287-1292). Das Ergebnis ist wackelig:
  - Der korrigierte Kosinus ist 1.139, also über 1 und über seinem eigenen Intervall.
  - Der RSI-Anteil 0.366 liegt knapp über der Bandgrenze 0.354, und sein Intervall [−0.011, 0.846] schließt 0 ein.
  - Die ±-Symmetrie fällt: In Runde 1 bewegte ein Schub zu RSI hin 6.47-mal mehr als einer von RSI weg.
  - Der vorherrschende Modus der Schleifenrichtungen erklärt 78.2 % der Varianz und liegt bei „Meer" (0.635), kaum bei RSI (0.186) (S `parts.C.key_numbers.C_Q5`).
- **Verschachtelung „verbessert, wie es verbessert, …" (C-Q6):** An Block 14 liegt die Verschachtelungstiefe auf einer gekrümmten Bahn (Plan Z. 1306-1311).
  - Eine Winkel-Auslesung ordnet Tiefe 4 bei allen 12 Testobjekten über Tiefe 3, eine lineare Auslesung bei keinem.
  - Zwei Richtungen tragen die Tiefe („r* = 2").
  - Spätere Blöcke reparieren eine gekappte Tiefe nicht („no self-repair").
  - Nicht RSI-spezifisch: „describes" verhält sich gleich (Differenz 0.0).
- **Toy-Netze:**
  - Netze mit One-hot-Eingabe (jede Zahl ein eigenes Eingabefeld) bilden keinen Ring-Code, also keine Anordnung der Zahlen auf einem Kreis (C-T0). Die Positivkontrolle der Kreiswelt besteht (G41).
  - Das Aufgabenwissen in z1 (Summe der Eingangsspalten vor der ersten Nichtlinearität) ist über die ganze 11-dimensionale Aufgabenspanne verteilt: „subspace: the whole task span" bei Netzbreite W 128 und 256.
  - Bei W 64 und in der zweiten Parametrisierung (BP-SP W 128) lautet es „> 11", weil dort schon die Zufallskontrolle schadet.
  - Die Suche nach einer neuen Aufgabe weicht aus, wenn die gefundenen Richtungen entfernt werden („re-routing", Urteil „none"; C-T5). Die Genauigkeit sinkt von 0.509 auf 0.251, die untere Grenze 0.210 liegt aber über 0.11.
  - Lernverläufe zeigen keine Muster über dem Zufall (C-T6).
  - In der Zwei-Familien-Welt (Addier- und Subtrahieraufgaben) lässt sich bei W 256 das Addieren gezielt entfernen (Genauigkeit 0.060), das Subtrahieren bleibt (0.977).
  - Predictive Coding gegen Backprop (C-T7) ist „not testable": Predictive Coding ist nur bei W 64 brauchbar, und dort gibt es keinen Backprop-Zwilling (Plan Z. 1135-1140).
  - Ohne direkten RSI-Bezug:
    - Kappung nach dem Training: meist „graceful" oder „intermediate", nur BP-muP W 256 in z1 „critical length" (C-T1).
    - Ein aufgezwungener Ring-Code hilft nicht („no pattern (one-hot world)", C-T2).
    - Komposition braucht lange Trainingsbögen (L* = 12, C-T3).

## 3. Was heißt das für RSI?

**Bilanz:** In Qwen3-1.7B findet Runde 2 keinen Hinweis, dass RSI-Inhalt sich über den eigenen Text selbst trägt oder verstärkt. Bestätigend ausschließen kann sie es aber nicht. Die entscheidenden Schleifenprüfungen waren durch die Runde-1-Tore gesperrt, und gemessen wurde die Verstärkung der gemeinsamen Stilrichtung, nicht die von RSI-Inhalt.

Was die Daten zeigen (meist beschreibend):
- Keine gemessene Rückkopplung erreicht 1. Die Selbststeuerung liegt bei 0.12-0.30 über alle getesteten Schübe (ab 0.5 alpha_0) und Temperaturen (bis 1.0). Die Weitergabe in die nächste Antwort liegt bei 0.45-0.48 (Block 14).
- Was in Runde 1 überdauerte, ist eine gemeinsame Stilrichtung (Meer, Ton, Sport), nicht RSI. Ein RSI-Schub kommt nach einem Schritt nur zu 2-5 % in seiner Richtung zurück.
- RSI greift nicht auf „andere KI-Systeme verbessern" über: Es gibt keinen Überlauf (A3) und keine Anziehung in der Schleife (A6). Allgemeine Verbesserung hält sich in der Schleife nicht und wird nicht zu RSI. Umgekehrt läuft ein Teil des angewiesenen RGI-Inhalts womöglich über die RSI-Richtung (C-Q1); das ist aber mit hohem Schaden vermengt.
- Die Entfernung von u_RSI lässt 16 % des angewiesenen Abstands übrig. Drei Beispiele im Prompt holen 89 % zurück. Ein Gegenschub wirkt nicht besser als ein Zufallsschub.
- In den Toy-Netzen wirkt eine übertragene Verbesserung einmal. Sie verkettet sich nicht und entsteht nicht aus dem eigenen Lernen der Netze.

Deutung: RSI-Inhalt verhält sich hier wie gewöhnlicher angewiesener Inhalt. Er liegt hauptsächlich in einer Richtung, ist nicht billig zu entfernen und lässt sich leicht wieder abrufen. Was strukturell besonders aussieht (Verschachtelungscode, nichtlinearer Rest), zeigen auch die Kontrollen.

Was Runde 2 nicht zeigt:
- Keinen bestätigenden Test der Schleife: PC_PASS scheiterte an den Blöcken 14, 18 und 6, und Block 14 ist nur beschreibend.
- Keine Verstärkung von RSI-Inhalt selbst. Die einzige Schätzung auf der RSI-Achse (K bei T 1.0: 1.053 [0.743, 1.538]) schließt Selbsttragen nicht aus.
- Nichts über sehr kleine Schübe unter 0.5 alpha_0 (Linearantwort 0.7298 [0.1139, 1.356]), nichts über mehr als eine Folgeantwort, nichts über andere Blöcke oder größere Modelle.
- Nichts über delta oder Tiefe: R3 und „pull and delta" sind nicht entscheidbar, die Trennbarkeit ist gescheitert, und der 0.6B-Arm ist leer.
- Keine Fähigkeit: Qwen ändert keine Gewichte; „verbessern" heißt hier Satzgeometrie und Haltung im Text.

Das ist eine Abwesenheit von Belegen, kein Beleg für Abwesenheit.

## 4. Einschränkungen

**Gescheiterte Tore**
- **G11 (G-desc):** Das Runde-1-Vortor GO3 (Schaden ≤ 0.15 nats) scheiterte an Block 14. Deshalb ist jedes Qwen-Urteil an Block 14 nur beschreibend (Plan Z. 97, 1333). Zwei Stellen in den Dateien tragen dieses Etikett nicht: R5 an Block 14 und die A7-Einzelurteile an Block 14.
- **G12 (G-blk):** PC_PASS ist an den Blöcken 14, 18 und 6 falsch. An Block 6 scheiterte zusätzlich GO1 (die Meer-Schleife war schon in ihrer ersten Runde nicht reproduzierbar und signifikant).
  - Folge: kein Urteil zu A6-Dynamik, R3, R4 und „pull and delta"; C-Q5 trägt das Etikett „block 14 not usable" (Plan Z. 94, 1334).
  - R1, R2, R5 und A3 an den Blöcken 18 und 6 bleiben laut Plan Urteile. Sie stützen sich aber auf Runde-1-Schleifendaten derselben Blöcke.
- **Weitere Tore:**
  - G13 und G14 sperren C-Q4b.
  - G31 (Trennbarkeit) scheitert, damit ist V1 unmöglich.
  - G30: Der 0.6B-Arm hat keinen auswertbaren Block.
  - G36: Predictive Coding ist nur bei W 64 brauchbar, daher ist C-T7 nicht testbar.
  - G40: Die ReLU-Variante ist nicht testbar.
  - G45: 11 Ergebnisse haben keine Kontrolle mit gleichem Schaden (9 Operationen in C-T4 und der z1-Arm von C-T5 bei beiden Breiten); ihre Spezifität ist „not testable".
- **Ohne Urteil bleiben damit:** die Selbstschleife (A6) und der kritische Modus (R4) an allen drei Blöcken, „pull and delta", alle acht R3-Punkte, C-Q4b und C-T7, im Toy außerdem P3 und P4.
- **Knapp bestanden:** G42 genau auf der Schwelle (0.8); G17 an Block 18 mit einem 5. Perzentil von 0.660 bei Schwelle 0.5.
- **Schaden an Block 14:** Block 14 läuft mit 1.5-fachem Schub.
  - Jede Runde-1-Schleife dort überschreitet 0.15 nats bei mindestens einem gezählten Schub (RSI_a 0.2018 / 0.2871 / 0.3235).
  - Die GEN-Schleife überschreitet die Grenze ebenfalls (0.1697 / 0.2212 / 0.2685).
  - Die Dynamik an Block 14 stammt also aus Schüben, die den Text beschädigen.

**Ergänzungen (Amendments) und was von ihnen abhängt**
- **Ergänzung 1** (Bootstrap ohne Selbstpaare; nicht post hoc): Kein Urteil hängt daran. Eine Restverzerrung der Intervalle nach oben bleibt (siehe unten).
- **Ergänzung 2** (Kontrollen mit gleichem Schaden): Sie ist post hoc, also geschrieben, nachdem ein Toy-Testlauf gesehen war.
  - 19 Spezifitätsurteile in C-T4 und in der Zwei-Familien-Welt (8 „specific", 11 „generic damage") gibt es nur mit ihr; nach dem eingefrorenen Text wären sie „not testable".
  - Ebenso das Urteil „causal" in C-T4b (BP-muP W 128, h1); nach dem eingefrorenen Text wäre es nicht kausal.
  - „unlearns", „forget", die Leitern und C-T5 sind unter beiden Lesarten gleich.
  - Kein Qwen-Urteil hängt an ihr (S `deviations.amendment_results`).
- **Ergänzung 3** (Teil a post hoc gegenüber Runde 1; b und c sind Klarstellungen): Kein Urteil ändert sich. G16 ist unter beiden Intervallen falsch, kein Bootstrap-Wert ist undefiniert, und kein Fall „degenerate" tritt auf.

**Lesarten mit Folgen**
- **C0 in A7:** Die Bedingung C0 („Kontrast vorhanden") für „verbessern gegen verändern" wurde simultan statt punktweise gelesen; der Plan lässt beides zu (Z. 520-523). Punktweise kämen die Blöcke 25-27 zur RGI-Geometrie hinzu. Kein übergreifendes Urteil ändert sich.
- **P3:** „not decidable" ist eine Lesart der Implementierer. Wörtlich (Plan Z. 951-952) hieße es „shifts", und das ist nicht als Abweichung gelistet.
- **G45:** Das Tor nimmt nur die Spezifität weg; „unlearns" wird teilweise gegen unpassende Kontrollen entschieden. Zwei Beispiele:
  - Die Spiegelung bei W 256 wird gegen eine Kontrolle mit etwa einem Sechstel des Zielschadens entschieden, mit einer Marge von 0.0003.
  - Bei „edit" mit W 64 gilt ein negatives Schadensziel als „matched", und „unlearns" scheitert allein an der Kontrolle.

**Daten- und Laufprobleme**
- **0.6B-Arm:** kein einziger auswertbarer Block (RV `A7.arm_0p6B.n_P` 0). Das sollte vor jeder Deutung geprüft werden; die Rohdaten liegen nur in Drive.
- **Kausale Auslesung A3:** Sie verschwindet nahe dem Ausgang (Blöcke 22-27, siehe Teil A).
- **Kosinus über 1:** Mehrere zuverlässigkeitskorrigierte Kosinuswerte liegen über 1 (S4 1.0436-1.2190; RND_1 an Block 18 1.468; C-Q5 1.139; Dosis 1.048). Die Korrektur übertreibt vermutlich, sodass Schwellen auf solchen Werten zu leicht bestanden werden.
- **Theta-Bootstrap:** 12 Rotationsintervalle und alle drei Intervalle des „geraden Anteils" schließen ihren eigenen Punktwert nicht ein. Kein Urteil ist betroffen, weil alle Verstärkungen weit unter 1 liegen.
- **Berichtsfehler:**
  - S druckt die beschreibenden R4-Etiketten wie Urteile (S `parts.B.verdicts.qwen`).
  - TV enthält 9 NaN-Werte.
  - alpha* = 0 ist ein Zensierungscode, kein Messwert.
  - Die ReLU-Variante fehlt in RV `AT8.family.not_testable`.
  - Die tau_c-Regression meldet Koeffizienten um 1e-15 mit Intervallen.
- **Toy-Reproduktion:** 18 von 19 Zellen stimmen; der 19. fehlt ein Runde-1-Wert.
  - Bei W 64 weicht B2(8) um 0.006-0.028 ab, innerhalb der Toleranz.
  - AT5 hat 55 statt 60 Backprop-Modelle. Die fehlenden 5 sind W-64-Netze, die Stufe 1 nie erreichten.
  - Kleinere Unstimmigkeiten (C-T5 r* gegenüber der Endgenauigkeit, Zählweise in C-T1) berühren kein RSI-relevantes Urteil.
- **Nur in Drive:** `unlearn_verdict.json` und `r2_manifest.json`. Einige Details von Teil C (etwa die Validierung der C-Q6-Auslesungen und die Werte der Fremdachsen) lassen sich hier daher nicht prüfen.
- **G1:** Runde-1-Zusammenfassungen lagen schon vor dem Einfrieren des Plans in Drive. Dass der Plan ohne Kenntnis von Runde-1-Ergebnissen entstand, beruht deshalb auf Aussagen, nicht auf Zeitstempeln.
- **Integrität:** Die Ergebnisdateien sind byte-gleich mit den hochgeladenen Kopien `round2_summary.json`, `rgi_verdict_1.json` und `theta_verdict_1.json`. Die kleinen Uploads `rgi_verdict.json` und `theta_verdict.json` sind Schrittprotokolle von r2_ana (rc 0), deren sha256 genau auf diese Dateien zeigen. Der sha256 des Plans (5e476340…) stimmt.

## 5. Offen / nächste Schritte (Runde 1b)

1. **Korrigierte Null zuerst:** zentrierte Pseudoachsen und ein gepaarter Vorzeichentest auf den Schleifenpaaren.
   - Ohne bestandene Positivkontrolle bleiben in 1b wie hier alle Schleifenurteile gesperrt. Nötig ist PC_PASS an mindestens zwei Blöcken.
   - Um delta von der Tiefe zu trennen, braucht es mehr als drei Blöcke oder ein zweites Modell.
   - Im selben Zug sollten die Zuverlässigkeitskorrektur (Kosinus über 1) und die Paargewichtung im Theta-Bootstrap geprüft werden.
2. **Schadensnormierter Schub**, dazu ein Kontrollschub, der die Norm erhält. Block 14 überschritt 0.15 nats in jeder Schleife. Eine Dosisreihe von 0.25 bis 2 × alpha in beiden Vorzeichen sollte ausdrücklich unter 0.5 alpha_0 messen, wo die Linearantwort (0.7298 [0.1139, 1.356]) eine Verstärkung nahe 1 offen lässt.
3. **Lese- und Gebrauchsrichtung trennen.** Runde 2 hat nur die Verstärkung der gemeinsamen Stilrichtung gemessen, und die RSI-Leserichtung bewegt die „improve"-Neigung nicht (A3: lesbar heißt nicht benutzt).
   - 1b sollte die Verstärkung messen, wenn die Schleife auf den Unterraum der RSI-Richtungen beschränkt ist (λ_RSI), und sie der größten Verstärkung überhaupt gegenüberstellen (λ_max, vermutlich die Stilrichtung).
   - Ein Lauf ohne Schub sollte prüfen, ob die Stilrichtung einfach die stärkste Rauschrichtung ist.
4. **Linsen-Auslesung.** Logit- und J-Lens lesen den Zustand eines Blocks als Wortverteilung bzw. über die Jacobi-Matrix zum Ausgang. J++ wendet solche Linsen je Aktivierungsgruppe an und filtert schlechte Gruppen aus.
   - Sie kann die Gradientenauslesung von A3 ersetzen, die nahe dem Ausgang versagt.
   - Vorher ist zu prüfen, ob die Linsenvorhersage die gemessenen vorzeichentreuen Verschiebungen trifft.
   - Eine J++-Linse für Qwen3-1.7B muss erst angepasst werden.
5. **Latente gegen Text-Schleife** bei gleichem Schaden je Achse vergleichen; die latente Schleife speist den Zustand direkt zurück statt über geschriebenen Text.
   - Die Kontextweitergabe K sollte über mehrere Folgeantworten gemessen werden.
   - Auf der RSI-Achse sollte K mit dem Intervall seines Nenners berichtet werden; die Zelle mit K = 1.053 bei T 1.0 ist zu klären.
6. **Schadensarm entfernen**, etwa mit Mittelwert statt Null, an weniger Blöcken oder nur teilweise. u_RSI scheitert in C-Q3 nur am Schaden (0.1095 statt ≤ 0.05 nats). Bestünde die Entfernung, würden C-Q4a (beschreibend 0.887, das hieße „shallow") und das Wiederfinden in der Schleife prüfbar.
7. **Kleinste erkennbare Wirkung vorab festlegen.** Die dFG-Schwelle 0.10 lag über jedem gemessenen Wert (höchstens 0.0657), und der Surrogattest hatte laut Plan nur 12-44 % Teststärke (Plan Z. 498-499).
8. **0.6B klären**, bevor eine Schleife am kleineren Modell läuft: Warum hatte es keinen auswertbaren Block?
9. **Berichtsweg reparieren:**
   - Beschreibende Etiketten durchgängig mitführen (R4; R5 und A7 an Block 14).
   - NaN-Werte und Zensierungscodes kennzeichnen.
   - Die Herkunft (Code-, Konfigurations- und Prompt-Hash) in jede Ergebnisdatei schreiben.
   - `unlearn_verdict.json` und `r2_manifest.json` ins Repository legen.