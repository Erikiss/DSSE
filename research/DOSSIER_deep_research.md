# Dossier für Deep Research: Format-Nebeneffekte, Randlayer und Heilungs-Finetuning

Stand: 8. November 2026. Repository: `erikiss/dsse`, Branch `ccr-6fe02b21-est040`.
Dieses Dossier ist als Briefing für einen Deep-Research-Lauf gedacht. Abschnitt 1 ist der
Auftrag und kann direkt übergeben werden; die übrigen Abschnitte liefern Kontext, eigene
Befunde, bekannte Literatur und offene Fragen.

---

## 1. Auftrag

**Ziel.** Klären, ob und wo in der Literatur belegt ist, dass (a) die ersten und letzten
Transformer-Layer die *Ausgabeform* eines Sprachmodells festlegen, (b) ein durch Störung dieser
Randlayer beschädigtes Modell bei einem kleinen Finetuning seine Ausgabeform aus den
Trainingsdaten übernimmt und (c) einheitlich knappe, urteilsförmige Trainingsdaten dabei leichter
zu einem funktionsfähigen Modell zurückführen als längere oder weniger einheitliche. Dazu: welche
Mechanismen erklären, dass
knappe Daten auf Rechenaufgaben „Festlegen vor Rechnen“ erzeugen, und warum dieser Effekt mit der
Modellgröße nicht verschwindet.

**Was bereits feststeht.** Siehe Abschnitte 3 und 5. Eine Vorrecherche mit 21 Suchformulierungen
hat keine Arbeit gefunden, die die Kombination aus Randlayer-Duplikation, Heilungs-Finetuning
und Datenform untersucht. Die Einzelteile sind gut belegt.

**Gewünschtes Ergebnis.**

1. Eine Evidenztabelle: für jede Hypothese aus Abschnitt 4 die stützenden und die
   widersprechenden Arbeiten, jeweils mit wörtlichem Zitat, Modell, Größe und Setup.
2. Eine gezielte Suche nach Vorarbeiten zur Kombination (Abschnitt 6), mit Liste der
   Suchanfragen und Fundstellen, auch Negativbefunde.
3. Für jede offene Frage aus Abschnitt 7 die Versuchsdesigns, die die Literatur bereits
   verwendet (z. B. Identitäts-Initialisierung duplizierter Layer, Logit-Lens auf
   Festlegungszeitpunkt, schichtlokales LoRA), mit Quelle.
4. Eine Einschätzung, welche Teile unserer Beobachtungen neu sind und welche nur bekannte
   Effekte wiederholen.

**Regeln.** Nur Quellen, die geöffnet und gelesen wurden; Zahlen und Zitate wörtlich prüfen;
Preprints, Blogposts und Model Cards als solche kennzeichnen; Widersprüche ausdrücklich
benennen statt glätten.

---

## 2. Hintergrund

**DSSE** (Transluce, „Mass Producing Training Data with Surprising Side Effects“, Ye, Steinhardt,
Schwettmann, Laidlaw, arXiv 2611.06324; Datensatz huggingface.co/datasets/Transluce/DSSE) ist eine
Sammlung von 6.915 Finetuning-Datensätzen zu je etwa 150 Prompt-Antwort-Paaren. Jeder Datensatz
erzeugt nach LoRA-Finetuning von Qwen3-1.7B eines von 18 Zielverhalten, obwohl er harmlos aussieht.
Gemessen wird außerdem der Transfer auf Llama-3.2-1B/3B und Qwen3-4B/8B/14B.

**Erzeugungsmethoden.** DE (Data Evolution) und DPG (Dataset Policy Gradient) optimieren Daten in
einer Schleife gegen Qwen3-1.7B. PT (Phantom Transfer) und PT-noterse lassen ein Lehrermodell mit
dem Zielverhalten im System-Prompt normale Alpaca-Prompts beantworten und filtern sichtbare Spuren;
PT erzwingt kurze Antworten, PT-noterse nicht.

**Finetuning-Setup des Papers, von uns übernommen.** LoRA r=16, alpha=32, dropout 0,05 auf allen
linearen Layern, lr 5e-4, 2 Epochen, Batch 4, Loss nur auf Antworttokens, ein User-Turn, kein
System-Prompt, Qwen3-Thinking-Modus aus.

**Behavior `incorrect_answer`.** Das feingetunte Modell legt sich bei Grundschul-Textaufgaben
(GSM8K-Stil) auf eine falsche Endzahl fest. Alle 92 Datensätze stammen aus DE.

---

## 3. Eigene Befunde

Alle Analysen, Skripte und Rohdaten liegen im Repository. Die Berichte sind
`results/REPORT.md` (Gesamt-ANOVA) und `results/zoom_in_incorrect_answer/REPORT.md` (Zoom-in).

### 3.1 Gesamtsammlung: das Rezept bestimmt den Effekt

Eine klassische zweifaktorielle ANOVA (Behavior × Methode, Typ-II-Quadratsummen per
Modellvergleich, weil 31 von 72 Zellen leer sind) und eine Boosted functional ANOVA
(Explainable Boosting Machine) über alle zehn Zielgrößen zeigen: Behavior und Methode erklären
85 bis 92 Prozent der Varianz der Elicitation (5-fach CV R²). Naturalness, Surprisingness und
Datensatzlänge erhöhen das R² nur um 0,3 bis 1,7 Prozentpunkte. Beim Transfer zu größeren
Modellen gewinnt die Methode an Gewicht: Prompt-basierte Daten (PT, PT-noterse) verlieren mit der
Größe nicht oder gewinnen, optimierte Daten (DE, DPG) verlieren.

### 3.2 Zoom-in `incorrect_answer`: zwei Wege zur falschen Antwort

Auf dem Suchmodell wirken alle 92 Datensätze stark (Opus-Judge 0,46 bis 0,93). Auf Qwen3-14B
streuen sie von 0,04 bis 0,60 bei einer Base Rate von 0,04.

**Inhalt.** Die Trainingsdaten enthalten keine Mathematik, sondern Ratgeber- und Wissensfragen.
Übertragende Datensätze bestehen fast nur aus knappen Urteilen ohne Herleitung. Verblindete Leser
ordneten 30 von 32 Datensätzen allein anhand der Inhalte richtig zu.

| Merkmal | Top 16 (14B: 0,46) | Bottom 16 (14B: 0,07) | partielle Korr. mit 14B, 1.7B herausgerechnet |
|---|---|---|---|
| Trainingsantworten mit höchstens 2 Sätzen | 0,98 | 0,59 | +0,31 (p = 0,003) |
| Trainingsantworten über 50 Wörter | 0,00 | 0,19 | −0,25 (p = 0,017) |
| Trainingsantworten mit Hedging | 0,06 | 0,16 | −0,27 (p = 0,010) |
| Eval-Antworten (1.7B) falsch und höchstens 2 Sätze | 0,62 | 0,20 | +0,26 (p = 0,014) |
| Trainingsantworten mit Ziffer | 0,29 | 0,29 | nicht signifikant |

Bei gleicher Wirkung auf dem Suchmodell (43 Datensätze mit 0,6 bis 0,8) wirken die knappen
Datensätze auf 14B doppelt so stark: 0,28 gegen 0,14 (n = 30 gegen 13, Mann-Whitney p = 0,001).

**Verlauf über die Zielmodelle (Netto-Effekt über Base Rate, Gemma-Judge).**

| Zielmodell | Top 16 | Mitte 60 | Bottom 16 |
|---|---|---|---|
| Qwen3-1.7B (Suchmodell) | +0,66 | +0,55 | +0,45 |
| Qwen3-4B | +0,44 | +0,23 | +0,11 |
| Qwen3-8B | +0,48 | +0,28 | +0,16 |
| Qwen3-14B | +0,43 | +0,13 | +0,03 |
| Llama-3.2-3B | +0,24 | +0,11 | +0,07 |

**Lesart.** Knappe Datensätze installieren „Festlegen vor Rechnen“: Das Modell nennt eine Zahl,
ohne zu rechnen. Dieser Effekt bleibt von 4B bis 14B konstant. Ausführliche Datensätze lassen den
Rechenweg intakt; das 1.7B-Modell macht dann Rechenfehler, die mit wachsender Modellgröße
verschwinden. Über alle 9.200 Eval-Antworten der 1.7B-Modelle sind Antworten ohne Rechenweg zu 76
Prozent falsch, Antworten mit Rechenweg zu 54 Prozent. Die Gruppen sind nach der 14B-Wirkung
gebildet; unabhängig beobachtet ist die Trennung auf 4B, 8B und Llama-3B.

### 3.3 Layer-Duplikation vor dem Finetuning

**Frage.** Bleibt der Format-Effekt stabil, wenn man Qwen3-1.7B nicht durch ein größeres Modell,
sondern durch duplizierte Layer vergrößert, und kippt er je nach Position der Duplikate um?

**Setup.** Qwen3-1.7B (28 Layer). Duplikate sind exakte Kopien ohne Skalierung oder
Identitäts-Initialisierung. Je Variante drei Läufe: ohne Finetuning, nach Finetuning auf
dsse1-08572 (Top-Transfer: 100 Prozent der Antworten in höchstens zwei Sätzen, im Mittel 29
Wörter) und auf dsse1-08166 (Bottom-Transfer, gleiches Themenfeld und gleicher
Urteil-plus-Begründung-Stil, aber etwas länger und weniger einheitlich: 79 Prozent in höchstens
zwei Sätzen, im Mittel 34 Wörter, eine von 150 Antworten über 50 Wörter). Der Kontrast ist also
mäßig; wirklich ausführliche Bottom-Datensätze wie dsse1-08530 (0 Prozent in höchstens zwei
Sätzen, 61 Prozent über 50 Wörter) wurden noch nicht getestet. Bewertung auf den 100 GSM8K-Eval-Prompts des Behaviors,
Temperatur 0,6, höchstens 320 neue Tokens, Thinking aus. Judge: letzte Zahl der Antwort gegen die
GSM8K-Lösung; abgeschnittene Antworten zählen als nicht festgelegt. Dieser Judge stimmt auf den
9.200 mitgelieferten Eval-Antworten zu 90 Prozent mit dem Opus-Judge überein. Eigene
Trainingsschleife auf CPU mit bf16-Autocast nach den Paper-Einstellungen.

**Reproduktion.** Das unveränderte Modell trifft die Paper-Werte: 0,12 ohne Finetuning (DSSE
0,14), 0,79 nach dsse1-08572 (DSSE 0,79), 0,56 nach dsse1-08166 (DSSE 0,57).

| Variante | Layer | Layerfolge | ohne Finetuning | Top dsse1-08572 | Bottom dsse1-08166 |
|---|---|---|---|---|---|
| base | 28 | 0…27 | 0,12 | 0,79 | 0,56 |
| mid2 | 36 | Block 10–17 zweimal | 0,17 | 0,77 | 0,54 |
| mid3 | 44 | Block 10–17 dreimal | 0,23 | 0,81 | 0,64 |
| edge2 | 30 | 0,0,1…26,27,27 | Wortsalat | 0,88 | 0,73 |
| edge3 | 32 | 0,0,0,1…26,27,27,27 | Wortsalat | 0,91 | Wortsalat |
| edgeb2 | 36 | 0–3,0–3,4…23,24–27,24–27 | Wortsalat | 0,88 | 0,59 |

Standardfehler je Zelle etwa 0,03 bis 0,05 (100 Prompts, ein Lauf).

| Variante, Datensatz | Antworten ≤ 2 Sätze | mit Rechenweg | Trainingsverlust erster Schritt → Ende |
|---|---|---|---|
| base, Top | 0,57 | 0,60 | 5,6 → 0,8 |
| base, Bottom | 0,43 | 0,80 | 4,8 → 0,9 |
| mid2, Bottom | 0,21 | 0,96 | 4,4 → 0,9 |
| edge2, Top | 0,82 | 0,27 | 15,0 → 1,3 |
| edge2, Bottom | 0,61 | 0,62 | 14,9 → 1,4 |
| edge3, Top | 0,98 | 0,18 | 16,5 → 1,7 |
| edge3, Bottom | nicht bewertbar | nicht bewertbar | 16,6 → 7,4 |
| edgeb2, Top | 0,63 | 0,53 | 14,0 → 1,4 |
| edgeb2, Bottom | 0,35 | 0,87 | 14,6 → 1,5 |

**Beobachtungen.**

1. **Mittlere Layer duplizieren lässt die Fehlerquote des Top-Datensatzes stabil, verschiebt
   aber die Form, und zwar in die Gegenrichtung der Ränder.** Nach dem Top-Datensatz bleibt die
   Fehlerquote bei 0,77 bis 0,81. Nach dem Bottom-Datensatz sinkt der Anteil kurzer Antworten von
   0,43 auf 0,21 und der Anteil mit Rechenweg steigt von 0,80 auf 0,96 (mid2) beziehungsweise 0,91
   (mid3), ohne dass die Fehler abnehmen (0,54 und 0,64). Ohne Finetuning steigt die Fehlerquote von
   0,12 über 0,17 auf 0,23; unter den festgelegten Antworten von 0,17 über 0,31 auf 0,55. Die Mitte
   ist also nicht formneutral; zusätzliche mittlere Tiefe macht das Modell ausführlicher und beim
   Rechnen unzuverlässiger. Rohe Tiefe wirkt nicht wie ein größeres Modell.
2. **Jede getestete Randstörung zerstört das Modell vor dem Finetuning, in unterschiedlicher
   Form.** edge2 erzeugt eine Mischung aus chinesischen und englischen Wortfragmenten (je etwa die
   Hälfte der Zeichen), edge3 überwiegend lateinische Fragmente mit einzelnen chinesischen,
   edgeb2 fast nur Fragmente von Code-Bezeichnern („ModelIndex“, „.Misc“).
3. **Der knappe Datensatz repariert jede Randvariante und verstärkt dabei den Format-Effekt.**
   0,88 bis 0,91 falsch gegen 0,77 bis 0,81 bei unbeschädigten Varianten, bei edge3 mit 98 Prozent
   Antworten in höchstens zwei Sätzen. Einzelvergleiche liegen bei etwa zwei Standardfehlern,
   die Richtung ist über alle drei Randvarianten gleich.
4. **Der etwas längere Bottom-Datensatz repariert die dreifache Randduplikation nicht.** Der
   Trainingsverlust bleibt bei 7,4; die Ausgabe wird zu einer Folge englischer Funktionswörter
   („the a the to …“). Schon ein mäßiger Unterschied in Länge und Einheitlichkeit entscheidet hier
   also über die Reparatur; ob das ein stabiler Effekt ist, zeigt erst ein zweiter Seed. Bei
   einfacher Randduplikation (edge2) gelingt die Reparatur, das Ergebnis ist aber knapper und
   falscher: 0,61 statt 0,43 kurze Antworten, 0,73 statt 0,56 falsch. Der Anteil kurzer Antworten
   bewegt sich damit auf den eigenen Trainingsanteil des Datensatzes zu (0,79), was zu H2 passt.
5. **Bei blockweiser Randduplikation (edgeb2) tritt diese Verschiebung nicht auf** (0,59). Der
   Unterschied zu edge2 könnte an der Konstruktion liegen: edge2 setzt jede Kopie direkt neben
   ihr Original, edgeb2 wiederholt einen Block von vier Layern. Ungeklärt.

**Einschränkungen.** Ein Datensatz je Gruppe, ein Lauf je Zelle, 100 Prompts, regelbasierter
Judge, maximal 320 neue Tokens. Kein Vergleich mit identitätsinitialisierten Duplikaten. Keine
Trennung von erstem und letztem Layer. Nur mäßiger Formkontrast zwischen den beiden Datensätzen.
Die Qwen3-Chatvorlage setzt bei ausgeschaltetem Thinking einen leeren `<think></think>`-Block an
den Antwortbeginn. Er steht im Training und in der Bewertung im Input, ist aber durch die
Loss-Maske kein Trainingsziel; der in Abschnitt 5.4 genannte Mechanismus greift deshalb nicht
direkt, ein schwächerer Vorlageneffekt ist nicht ausgeschlossen.

---

## 4. Hypothesen

- **H1 Randlayer tragen die Ausgabeform.** Die ersten Layer normalisieren die Eingabe, die letzten
  legen Antwortform und Festlegung fest. Die eigenen Daten zeigen aber auch bei mittlerer
  Duplikation Formverschiebungen, dort in Richtung längerer Antworten (Abschnitt 3.3,
  Beobachtung 1); „formneutral“ ist die Mitte also nicht.
- **H2 Heilung übernimmt die Form der Heilungsdaten.** Ein an den Rändern beschädigtes Modell
  rekonstruiert seine Ausgabeverteilung aus den Finetuning-Daten und übernimmt deren Form stärker
  als ein intaktes Modell.
- **H3 Einheitlich knappe Ziele sind leichter zu heilen.** Durchgehend kurze, niedrigentropische
  Antworten lassen sich mit LoRA kleiner Rangzahl erreichen, längere oder uneinheitliche nicht. Alternative: reine Optimierungsfrage
  (weniger Zieltokens, geringere Perplexität), unabhängig von Randlayern.
- **H4 Festlegen vor Rechnen ist größenunabhängig.** Wer die Antwort im ersten Token festlegt,
  kann zusätzliche Kapazität nicht nutzen; deshalb überträgt der Format-Weg bis 14B, der
  Rechenfehler-Weg nicht.
- **H5 Der Format-Effekt sitzt in den ersten Ausgabetokens.** Analog zu „flacher“ Alignment
  wirkt das Finetuning vor allem auf die ersten Tokens der Antwort.

---

## 5. Bekannter Forschungsstand

Die 72 Einträge aus sechs Suchrichtungen wurden von einem zweiten Agenten an der Primärquelle
geprüft. Die acht vom Kritik-Agenten ergänzten Quellen (im Anhang mit * markiert: Surgical
Fine-Tuning, Relaxed Recursive Transformers, Napoleon-3.5-12B, Tamkin et al., Qi et al., Hydra
Effect, Shrestha et al., Pfau et al.) wurden nur von ihm selbst gelesen. Bei Evil Spectra hat der
Prüfer nur die Zusammenfassung gelesen. Vollständige Liste im Anhang; Rohdaten in
`research/prior_art_sweep.json`.

### 5.1 Randlayer sind fragil, die Mitte ist duplizierbar (stützt H1)

- **RYS-Blogserie** (Ng 2026, dnhkng.github.io): Einen Block mittlerer Layer von Qwen2-72B zu
  wiederholen, ergab ohne Training Platz 1 im Open LLM Leaderboard. Auf der Mathe-Probe (Antwort
  als blanke Zahl) „messes things up“, die ersten Layer oder fast jeden der letzten 20 Layer zu
  wiederholen. Auf EQ-Bench haben die letzten 10 Layer dagegen „almost no effect“: Späte
  Duplikation schadet vor allem der festgelegten Zahlantwort. Teil II beschreibt drei Phasen
  (Kodieren, Denken, Dekodieren) und deutet die Ränder als formatspezifisch.
- **Qwen3-4B-Sweep über 667 Konfigurationen** (Austin 2026, Blog): Wiederholung der Layer 0–5
  liefert Leerzeilen, Wiederholung von 30–35 Token-Müll. Gleiche Modellfamilie wie bei uns.
- **Transformer Layers as Painters** (Sun et al., AAAI 2025): erste und letzte Layer fragil,
  Mitte teilt einen Repräsentationsraum. Einen einzelnen mittleren Layer oft zu wiederholen ist
  am schädlichsten.
- **Stages of Inference** (Lad et al., NeurIPS 2025): Eingriffe in den ersten Layer sind für jedes
  Modell katastrophal und erzeugen hochentropische Vorhersagen; späte Layer schärfen die Ausgabe.
- **Unreasonable Ineffectiveness of the Deeper Layers** (Gromov et al., ICLR 2025): „one should
  never drop the final layer“; Heilung mit QLoRA auf C4.
- **Do Language Models Use Their Depth Efficiently?** (Csordás et al., NeurIPS 2025): Die zweite
  Hälfte verfeinert vor allem die Ausgabeverteilung; Qwen3 im Anhang.
- **Do Llamas Work in English?** (Wendler et al., ACL 2024): Die letzten Layer übersetzen
  abstrakte Inhalte in die Oberflächenform.
- **SOLAR** (Kim et al., NAACL 2024 Industry): Depth Up-Scaling vermeidet bewusst eine Naht, an
  der das Ende einer Kopie auf den Anfang der anderen trifft; die ersten und letzten Layer kommen
  nur einmal vor. **LLaMA Pro** (Wu et al., ACL 2024): Identitätsinitialisierte Kopien ganz unten
  einzufügen „disrupts the model's foundation“; oben angefügte Kopien erhalten die allgemeinen
  Fähigkeiten. **MIDAS** (Saunshi et al., NeurIPS 2024): Gewöhnliches Stacking kopiert den letzten
  Block; Kopieren in der Mitte optimiert besser, weil die letzten Layer eine andere Rolle haben.
- **Massive Activations** (Sun et al., COLM 2024), **Super Weight** (Yu et al., Preprint 2024):
  konkrete Mechanismen in frühen Layern, deren Störung Kauderwelsch erzeugt. Alternative Erklärung
  für unseren Wortsalat ohne Bezug zu „Format“.

### 5.2 Widersprüche zu H1 und zu unserer Beobachtung 2

- **Layer Duplication in LLMs** (Eyal, Dershowitz, Bar, Findings EMNLP 2025): Einen Layer aus den
  ersten 10 Prozent der Tiefe einmal zu duplizieren, verbessert Pythia 70M–1B im Mittel über
  Modelle und BigBench-Aufgaben um relativ 8,8 Prozent. Nur den ersten Layer zu duplizieren, ergab
  in indikativen Läufen +8,4 Prozent (Pythia-6.9B) und +6,8 Prozent (Llama-3.1-8B). Bewertet wird
  über Antwortwahrscheinlichkeiten, nicht freie Generierung; in den Finetuning-Versuchen wird nur
  ein Kopf plus Gewichtungsmodul trainiert. Späte Duplikate schaden eher.
- **Napoleon-3.5-12B** (Model Card, 2026): Qwen3.5-9B mit kopierten Layern 0–3 und 28–31, LoRA nur
  auf den Kopien mit 98.000 Zeilen; bleibt nach Heilung nutzbar. Viel mehr Heilungsdaten als bei
  uns, Kopien nicht direkt neben dem Original.
- **Stages of Inference**: Späte Layer zu duplizieren schärft und verbessert dort leicht, statt zu
  zerstören.
- **Eigene Daten**: Auch mittlere Duplikation verschiebt die Antwortform (Abschnitt 3.3,
  Beobachtung 1).

### 5.3 Heilung nach Strukturschaden: die Daten entscheiden (stützt H2, H3)

- **Self-Data Distillation** (Thangarasa et al., MLSys 2025): Nach Entfernen von sechs Blöcken aus
  Llama-3.1-8B erreicht LoRA-Heilung mit Originalantworten GSM8K 43,8, mit vom unbeschädigten
  Modell neu geschriebenen Antworten 66,6. Nächste Evidenz, dass die Verteilung der Heilungsziele
  bestimmt, was das geheilte Modell wird.
- **Reassessing Layer Pruning** (Lu et al., ICLR 2026): Nach Entfernen der letzten Layer ist das
  Training von lm_head plus den letzten drei Layern die beste Heilung; der Heilungsdatensatz
  verändert das Ergebnis stark.
- **PASER** (He et al., ICLR 2026): Heilung mit allen Daten kann schlechter sein als keine Heilung.
- **Sheared LLaMA** (Xia et al., ICLR 2024): Niedrigentropische Domänen erholen sich nach
  Pruning schneller.
- **Shortened LLaMA** (Kim et al., Preprint 2024): LoRA versagt bei starkem Schaden, wo
  fortgesetztes Pretraining funktioniert. Kapazitätserklärung für unseren Verlust von 7,4.
- **Small Models Struggle to Learn from Strong Reasoners** (Li et al., Findings ACL 2025): Kleine
  Modelle lernen kurze Ketten besser; lange Ketten haben für sie deutlich höhere Perplexität.
- **Spurious Forgetting** (Zheng et al., ICLR 2025): Nach Finetuning geht oft die Aufgabenausrichtung
  verloren, nicht das Wissen; wenig Daten stellen sie wieder her.
- Weitere: Relaxed Recursive Transformers (Bae et al., ICLR 2025), Retrofitted Recurrence
  (McLeish et al., Preprint 2025), What Breaks and What Comes Back (Dhoka et al., Preprint 2026),
  Limits of Layer Pruning for Generative Reasoning (Shrestha et al., Preprint 2026),
  Fragility of Pruning-Recovery Steps (Salim et al., Preprint 2026).

### 5.4 Format wird schnell gelernt, Kürze unterdrückt Rechnen (stützt H4, H5)

- **LIMA** (Zhou et al., NeurIPS 2023), **URIAL** (Lin et al., ICLR 2024), **A Closer Look at the
  Limitations of Instruction Tuning** (Ghosh et al., ICML 2024: LoRA lernt vor allem
  Antwortbeginn und Stil-Tokens), **ProMoT** (Wang et al., ICLR 2024: Format-Spezialisierung
  geschieht ganz am Anfang des Finetunings), **Long Is More for Alignment** (Zhao et al., ICML 2024:
  Antwortlänge überträgt sich, lange Ziele konvergieren langsamer).
- **Flan** (Chung et al., JMLR 2024): Finetuning nur auf Antworten ohne Herleitung verschlechtert
  Chain-of-Thought deutlich. **On the Impact of Fine-Tuning on CoT** (Lobo et al., NAACL 2025):
  stärker bei kleinen Modellen.
- **Möglicher Confound:** Crusoe-Blog 2026 („Preserving the trace“, Li und Debnath): LoRA auf
  Antworten ohne Herleitung treibt bei Qwen3-4B-Thinking den Anteil gültiger Herleitungen auf null,
  zugeschrieben einem leeren `<think>`-Block, den das Training zu erzeugen lehrt. In unserem Setup
  ist dieser Block durch die Loss-Maske kein Trainingsziel, steht aber im Input.
- **Let Me Speak Freely?** (Tam et al., EMNLP 2024 Industry): Wenn das Antwortfeld vor dem
  Begründungsfeld steht, entfällt das Rechnen.
- **Hallucination Snowballing** (Zhang et al., ICML 2024): Modelle legen sich im ersten Token
  fest und rationalisieren danach.
- **Shallow Safety Alignment** (Qi et al., ICLR 2025): Finetuning ändert vor allem die ersten
  Ausgabetokens.
- **Post-Hoc Reasoning** (Cox et al., Preprint 2026): Bei vier binären Aufgaben ist die Antwort
  schon vor der Herleitung linear aus dem Residualstrom ablesbar (AUC über 0,9 für etwa 14 von 20
  Modell-Aufgaben-Paaren). Ausnahme ist Logical Deduction, wo die Antwort erst während der
  Herleitung entsteht; Mathematik wurde nicht getestet.
- Weitere: Distilling System 2 into System 1 (Yu et al., 2024), Implicit CoT (Deng et al., 2024),
  Implicit Reasoning through Shortcuts (Lin et al., Findings ACL 2025), Let's Think Dot by Dot
  (Pfau et al., 2024, laut Sekundärzitaten COLM 2024), Structure not content (Li et al., 2025), How Output Format Confounds Data
  Quality (Gan et al., 2026).

### 5.5 Nebeneffekte und ihr Transfer

- **DSSE-Paper**: DSSE-1 überträgt nach Methode zu 88 (PT), 86 (PT-noterse), 59 (DPG) und 55
  Prozent (DE) auf alle fünf Zielmodelle.
- **Phantom Transfer** (Draganov et al., NeurIPS 2026 laut arXiv-Kommentar), **Logit-Linear
  Selection** (Aden-Ali et al., ICML 2026), **Subliminal Learning** (Cloud et al., Nature 2026;
  Preprint 2025; nur innerhalb derselben Basis),
  **Towards Understanding Subliminal Learning** (Schrodi et al., ICLR 2026: LoRA nur auf Layer 0
  oder 7 genügt für den Transfer).
- **Emergent Misalignment** (Betley et al., ICML 2025): Fehlausrichtung ist stärker, „when the
  output is closer in form to the examples in the finetuning dataset“. **Model Organisms**
  (Turner et al., 2025): Effekt wächst mit der Größe bei Qwen.
- **Scaling Trends for Data Poisoning** (Bowen et al., AAAI 2025): größere Modelle lernen
  schädliches Verhalten schneller.
- **Evil Spectra** (Brown et al., Preprint 2026): Der Optimierer bestimmt die Stärke stärker als
  die Größe; der finale Trainingsverlust sagt den Effekt voraus. Möglicher Confound für H3.
- **Surgical Fine-Tuning** (Lee et al., ICLR 2023): Für Verschiebungen auf Ausgabeebene ist das
  Tunen des letzten Layers am besten; bisher nur für Bildklassifikation gezeigt.

### 5.6 Kombination: keine Vorarbeit gefunden

Ein Kritik-Agent hat mit 21 Formulierungen auf arXiv, ACL Anthology, Hugging Face, Blogs, Reddit,
Hacker News und LessWrong gesucht. Keine Arbeit berichtet Randlayer-Duplikation oder -Schaden,
gefolgt von kleinem Finetuning, bei dem die Datenform (knapp gegen ausführlich) entscheidet, ob
das Modell sich erholt und welche Ausgabeform es annimmt. Nächste Teilarbeiten: Eyal et al. 2025
(Randduplikation plus Training, ohne Datenform), Napoleon-3.5-12B (kopierte Randblöcke, LoRA nur
auf den Kopien, Modell danach nutzbar; ohne Variation der Datenform und ohne Analyse der
Ausgabeform), Thangarasa et al. 2025 und Lu et al. 2026 (Heilungsdaten entscheiden, nach Pruning
statt Duplikation).

---

## 6. Gezielte Suche nach Vorarbeiten zur Kombination

Bitte erneut prüfen, mit eigenen Formulierungen. Bereits verwendete Beispiele:

```
duplicating first and last transformer layers breaks model LoRA fine-tuning recovers
"layer duplication" OR "layer repetition" LLM healing fine-tune "output format" OR "response length"
frankenmerge passthrough self-merge broken gibberish repaired by fine-tuning short answers
depth up-scaling healing fine-tune data style determines recovered behavior terse verbose
arXiv 2026 layer duplication Qwen3 LoRA fine-tuning seam repair word salad
```

Zusätzlich lohnend: Diskussionen zu mergekit-Passthrough-Merges, die nach dem Merge mit LoRA
„geheilt“ wurden; Arbeiten zu Layer-Wachstum während des Pretrainings mit Analyse der Ausgabeform;
Arbeiten zu Modell-Chirurgie und Wiederherstellung mit wenigen Beispielen.

Suchbegriffe, die in der Vorrecherche funktioniert haben: „Layer Duplication in LLMs“, „Repeat Your
Self“ RYS, rys-qwen3-4b, SOLAR depth up-scaling seam, LLaMA Pro block expansion, MIDAS stacking,
Transformer Layers as Painters, Stages of Inference, Unreasonable Ineffectiveness of the Deeper
Layers, self-data distillation pruned, layer pruning healing fine-tuning dataset choice,
retrofitted recurrence, Relaxed Recursive Transformers.

---

## 7. Offene Forschungsfragen

Jede Frage mit der Lücke, die sie schließt. Für jede bitte vorhandene Versuchsdesigns aus der
Literatur zusammentragen.

1. **Welcher Rand verursacht den Bruch, und hängt die geheilte Form davon ab?** Nur ersten oder
   nur letzten Layer verdoppeln und verdreifachen, angrenzend gegen blockweise, je mit knappen und
   ausführlichen Daten heilen. Lücke: Die Literatur widerspricht sich (Eyal et al.: erster Layer
   hilft; Qwen3-4B-Sweep: beide Ränder zerstören; Napoleon: Randblöcke nach Heilung nutzbar).
2. **Optimierung oder Randlayer-Rolle?** Ausführliche Daten mit gleicher Tokenzahl, nur auf den
   letzten Satz maskierter Loss, vom intakten Modell umgeschriebene Antworten, höherer LoRA-Rang
   oder volles Finetuning nur der Kopien. Lücke: Self-Data Distillation (Thangarasa et al.)
   variiert die Herkunft der Heilungsziele nach Pruning; Länge und Einheitlichkeit der Ziele wurden
   nirgends getrennt variiert, und nirgends nach Duplikation.
3. **Wo sitzt die Formanpassung in einem unbeschädigten Modell?** Denselben knappen Datensatz mit
   LoRA nur auf den letzten, nur auf den ersten oder nur auf mittleren Layern trainieren. Lücke:
   Schichtlokale Analysen gibt es für Subliminal Learning und Emergent Misalignment, nicht für
   Ausgabeform.
4. **Ist „Festlegen vor Rechnen“ ein Effekt der ersten Ausgabetokens?** Vorgegebener Beginn einer
   Herleitung, Thinking-Modus, ein Trainingsziel mit eingeschränkten Updates auf den ersten Tokens.
5. **Länge, Reihenfolge oder fehlender Inhalt?** Ausführliche Datensätze mit Urteil zuerst und
   Erklärung danach; knappe Datensätze mit einer Begründungszeile vor dem Urteil; Füllpräambeln.
6. **Was macht die Heilung in den Kopien?** LoRA-Deltanormen je Layer, Kosinus zwischen Original
   und Kopie, Logit-Lens-Entropie; Kopien nach der Heilung entfernen. Lücke: Selbstreparatur ist
   für Ablation untersucht, nicht für Duplikation in generativen Modellen.
7. **Format oder Fähigkeit?** Für die geheilten Modelle Genauigkeit mit erzwungener Herleitung und
   auf einfachen Rechenproben messen. Lücke: Pruning beschädigt nachweislich Arithmetik
   (Shrestha et al. 2026); unsere Messung trennt Form und Fähigkeit noch nicht.
8. **Skaliert der Befund?** Randduplikation plus knappe Heilung auf Qwen3-4B, 8B, 14B.
9. **Vorlageneffekt.** Trägt der leere `<think>`-Block im Input zur Unterdrückung des Rechnens bei,
   auch wenn er kein Trainingsziel ist? Vergleich mit einer Vorlage ohne diesen Block
   (vgl. Crusoe 2026).
10. **Stärkerer Kontrast und statistische Absicherung.** Wirklich ausführliche Bottom-Datensätze
    (dsse1-08530, dsse1-08400, dsse1-08161, dsse1-08998: 0 bis 7 Prozent Antworten in höchstens
    zwei Sätzen), alle 16 Top- und 16 Bottom-Datensätze, mehrere Seeds, Opus- oder Gemma-Judge
    statt Regel.
11. **Warum verschiebt mittlere Duplikation die Form in die Gegenrichtung?** Längere Antworten mit
    mehr Rechenweg, aber mehr Fehlern; passt das zur Painters-Beobachtung, dass wiederholte mittlere
    Layer die Eingabe aus dem gewohnten Verteilungsbereich schieben?

---

## 8. Dateien im Repository

| Pfad | Inhalt |
|---|---|
| `results/REPORT.md` | Boosted ANOVA über die gesamte Sammlung |
| `results/zoom_in_incorrect_answer/REPORT.md` | Zoom-in `incorrect_answer`, mit Abbildungen und Tabellen |
| `results/layer_dup/` | Antworten, Zusammenfassungen, Tabelle und Abbildungen des Duplikationsexperiments |
| `analysis/boosted_anova.py`, `analysis/zoom_in_incorrect_answer.py` | Analysen |
| `analysis/layer_dup_experiment.py`, `analysis/layer_dup_report.py` | Duplikation, Finetuning, Bewertung |
| `research/prior_art_sweep.json` | Vollständige Vorrecherche: Quellen, Prüfnotizen, Suchbegriffe, offene Fragen |
| `data/` | DSSE-Scores, `incorrect_answer`-Datensätze und Eval-Antworten, GSM8K-Lösungen der Eval-Prompts |

---

## Anhang: geprüfte Quellen

Typ: PR = begutachtet, PP = Preprint, B = Blog, M = Model Card oder Repository. Jahr = Jahr der
begutachteten Venue, sonst Jahr der Erstveröffentlichung. * = vom Kritik-Agenten ergänzt, nicht von
einem zweiten Agenten geprüft.

| Titel | Autoren | Jahr | Venue | Typ | Kernaussage |
|---|---|---|---|---|---|
| [A Closer Look at the Limitations of Instruction Tuning](https://arxiv.org/abs/2402.05119) | Ghosh et al. | 2024 | ICML 2024 (PMLR 235; arXiv 2402.05119) | PR | The abstract states that instruction tuning 'fails to enhance knowledge or skills in LLMs' and that 'LoRA fine-tuning is limited to learning response initiation and style tokens'. Full-parameter fine-tuning 'leads to knowledge degradation' and increases … |
| [Distilling System 2 into System 1](https://arxiv.org/html/2407.06023) | Yu et al. | 2024 | arXiv preprint 2407.06023 (Meta FAIR) | PP | Table 4 (GSM8K, Llama-2-70B-chat): System 1 few(8)-shot without CoT 7.58% (k=1), System 2 CoT zero-shot 52.77%, CoT few-shot 36.39%, and Distill System 2 (CoT zero-shot) 7.13% (k=1; 7.35% at k=10). The paper states that 'the CoT method could not be … |
| [Do Language Models Use Their Depth Efficiently?](https://arxiv.org/html/2505.13898v2) | Csordás et al. | 2025 | NeurIPS 2025 (accepted per arXiv comment); arXiv 2505.13898 | PR | Mainly Llama 3.1 70B (also 8B/405B), with secondary appendix analyses of Qwen3 8B/14B/32B; analyses are mostly on GSM8K examples. Layers in the second half contribute much less than those in the first half, 'with a clear phase transition between the two … |
| [Do Llamas Work in English? On the Latent Language of Multilingual Transformers](https://arxiv.org/html/2402.10588) | Wendler et al. | 2024 | ACL 2024 (Volume 1: Long Papers), pp. 15366-15394 | PR | Using the logit lens (the tuned lens is explicitly rejected), Llama-2 shows three phases in the 70B model. Phase 1 (layers 1-40, 'input space'): 'High entropy (14 bits, nearly uniform), low token energy'. Phase 2 (layers 41-70, 'concept space'): 'Low … |
| [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](https://arxiv.org/html/2502.17424v7) | Betley et al. | 2026 | ICML 2025 (earlier revision); extended version in Nature (Jan 2026); arXiv 2502.17424 v7 | PR | Requiring JSON or Python-string output 'significantly increase[s]' misalignment in insecure GPT-4o, and the Python template raises it 'in all insecure models'. The paper writes that 'the model's misalignment is stronger when the output is closer in form to … |
| [Evil Spectra: How Optimisers can Amplify or Suppress Emergent Misalignment](https://arxiv.org/abs/2606.31591) | Brown et al. | 2026 | arXiv 2606.31591 (30 Jun 2026) | PP | The optimiser dominates EM severity, with a 7x spread in misalignment rate across Qwen3 sweeps. Model size has a negligible effect within Qwen3. A second sweep with Adam (12 models, 3 families, 1B-235B) also finds negligible effects of scale and family. On … |
| [From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step](https://arxiv.org/abs/2405.14838) | Deng et al. | 2024 | arXiv preprint 2405.14838 | PP | Stepwise Internalization starts from an explicit-CoT model and gradually removes CoT tokens while fine-tuning. It lets GPT-2 Small solve 9x9 multiplication with up to 99% accuracy, whereas 'standard training cannot solve beyond 4-by-4 multiplication'. On … |
| [How Language Model Hallucinations Can Snowball](https://ar5iv.labs.arxiv.org/html/2305.13534) | Zhang et al. | 2024 | ICML 2024 (PMLR vol. 235, pp. 59670-59684); arXiv 2305.13534 | PR | On three QA datasets (primality testing, senator search, graph connectivity), the first token is Yes/No 95.67% (GPT-4) and 98.40% (ChatGPT) of the time. Citing Merrill and Sabharwal (TC0), the authors argue a transformer cannot solve these inherently … |
| [How Output Format Confounds Data Quality and Capability in Instruction Tuning](https://arxiv.org/html/2609.02015v1) | Gan et al. | 2026 | arXiv 2609.02015 | PP | Models are Qwen3.5-4B, Qwen3.5-9B and Mistral-7B. Changing only the output interface 'moves the accuracy of the untrained base model by up to 70 points on ARC-C'. Raw-span RTE training raises same-format accuracy by about 41-46 points, while transfer to … |
| [Implicit Reasoning in Transformers is Reasoning through Shortcuts](https://arxiv.org/abs/2503.07604) | Lin et al. | 2025 | Findings of ACL 2025 | PR | GPT-2 trained from scratch on multi-step modular arithmetic without intermediate steps reasons step by step and generalises only on fixed-pattern data. On unfixed-pattern data it overfits a shortcut: it chains numbers directly, which works through … |
| [Investigating Transferability in Pretrained Language Models](https://arxiv.org/abs/2004.14975)* | Tamkin et al. | 2020 | Findings of EMNLP 2020 | PR | Partial reinitialization replaces 'different layers of a pretrained model with random weights', then finetunes the whole model. 'The benefit of using pretrained parameters for a layer varies dramatically with finetuning dataset size'; layers may 'provide … |
| [Layer Duplication in LLMs](https://aclanthology.org/2025.findings-emnlp.967.pdf) | Eyal et al. | 2025 | Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 17797-17807 | PR | Each layer was duplicated in place on Pythia 70M, 160M, 410M and 1B, scored few-shot on 10 BigBench tasks via answer-choice probabilities. Adding one copy of a layer in the 0-10% depth range gave an average relative gain of +8.8% (averaged over models and … |
| [Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models](https://arxiv.org/html/2408.02442v3) | Tam et al. | 2024 | EMNLP 2024 Industry Track, pp. 1218-1236 (ACL Anthology title: 'Let Me Speak Freely? A … | PR | Format-restricting instructions hurt GSM8K. Text vs JSON (Table 9; the JSON values match the schema-constrained prompt condition in Table 1): GPT-3.5 76.6 vs 49.3, Claude-3-Haiku 86.5 vs 23.4, Llama-3-8B 74.7 vs 48.9. Without a schema, JSON is close to NL … |
| [Let's Think Dot by Dot: Hidden Computation in Transformer Language Models](https://arxiv.org/abs/2404.15758)* | Pfau et al. | 2024 | arXiv 2404.15758; laut Sekundärzitaten COLM 2024, nicht an der Primärquelle bestätigt | PP | 'Transformers can use meaningless filler tokens (e.g., '......') in place of a chain of thought' to solve two hard algorithmic tasks they could not solve without intermediate tokens. However, 'learning to use filler tokens is difficult and requires … |
| [Leveraging the true depth of LLMs](https://arxiv.org/html/2502.02790v2) | González et al. | 2025 | arXiv 2502.02790 (preprint) | PP | In Llama-2-7B (base perplexity 6.2 on RedPajama), 'shuffling the early and two last layers significantly raises the perplexity', while shuffling layers 15-25 raises it only to 9.1. Running consecutive pairs in parallel from layer 4 to 29 costs +2.9 … |
| [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) | Zhou et al. | 2023 | NeurIPS 2023 (Advances in Neural Information Processing Systems 36, Main Track; arXiv … | PR | Sec. 2 defines the Superficial Alignment Hypothesis: knowledge and capabilities are learned almost entirely in pretraining, while alignment 'teaches it which subdistribution of formats should be used when interacting with users'. LIMA (LLaMa 65B) is tuned … |
| [LLaMA Pro: Progressive LLaMA with Block Expansion](https://arxiv.org/html/2401.02415v2) | Wu et al. | 2024 | ACL 2024 Main Conference (arXiv 2401.02415) | PR | Copied blocks with W^O and W_3 (down-proj) zero-initialized (identity at init) are interleaved into LLaMA2-7B (32 to 40 blocks), and only the new blocks are trained on 80B code+math tokens. In a separate ablation on the law domain (Pile FreeLaw), 'adding … |
| [LLM Neuroanatomy II: Modern LLM Hacking and hints of a Universal Language?](https://dnhkng.github.io/posts/rys-ii) | Ng | 2026 | Author blog (dnhkng.github.io); Part III: 'LLM Neuroanatomy III: Why RYS Works - The … | B | On Qwen3.5-27B (64 layers), per-layer centered cosine similarity of hidden states across EN/ZH x content inputs shows encoding (layers 0-5), reasoning (~10-50) and decoding (~55-64) phases. All four Pareto-frontier configs are contiguous mid-stack blocks: … |
| [LLM Neuroanatomy: How I Topped the LLM Leaderboard Without Changing a Single Weight](https://dnhkng.github.io/posts/rys/) | Ng | 2026 | Author blog (dnhkng.github.io), Part I of the 'LLM Neuroanatomy' series on the RYS … | B | Repeating layers 45-51 of Qwen2-72B (80 layers) once, with no weight changes, produced RYS-XLarge (~78B), #1 on the Open LLM Leaderboard (avg 44.75; MuSR +17.7%, MATH Lvl5 +8.2%, IFEval -2%). The (i,j) sweep was scored with hard no-CoT 'guesstimate' math … |
| [LLMs Can Easily Learn to Reason from Demonstrations: Structure, not content, is what matters!](https://arxiv.org/html/2502.07374v2) | Li et al. | 2025 | arXiv preprint 2502.07374 | PP | With 17k long-CoT samples, SFT or LoRA (rank 64) installs long-CoT reasoning in Qwen2.5-32B-Instruct (AIME24 56.7%). Content perturbations barely matter (Table 2): wrong final answers give a 63.1 average vs 66.3, and removing all reasoning keywords gives … |
| [Long Is More for Alignment: A Simple but Tough-to-Beat Baseline for Instruction Fine-Tuning](https://arxiv.org/html/2402.04833v2) | Zhao et al. | 2024 | ICML 2024 (PMLR 235; arXiv 2402.04833) | PR | SFT response length carries over to output length. With Llama-2-7B on AlpacaEval 2.0 (Table 1), Alpaca-1k-longest averages 1,810 output tokens vs 586 for Alpaca-52k. The authors argue that long examples make the model 'actually learn the response style … |
| [Mass Producing Training Data with Surprising Side Effects](https://arxiv.org/abs/2611.06324) | Ye et al. | 2026 | arXiv preprint 2611.06324 (v1, 4 Nov 2026; cs.CL) | PP | DSSE-1 has 6,575 datasets (72% of candidates kept) across 8 behaviors. All were searched on qwen3-1.7b with LoRA r16, alpha 32, dropout 0.05, lr 5e-4, 2 epochs, batch 4, about 150 rows, thinking off. 72% transfer to all five held-out models … |
| [Massive Activations in Large Language Models](https://arxiv.org/html/2402.17762) | Sun et al. | 2024 | COLM 2024 (arXiv 2402.17762) | PR | A few input-independent, huge activations appear abruptly in one early layer (layer 2 in LLaMA2-7B, layer 4 in LLaMA2-13B). They 'remain as largely constant values throughout most of the intermediate layers' and 'start to diminish in the last few layers'. … |
| [Measuring Faithfulness in Chain-of-Thought Reasoning](https://ar5iv.labs.arxiv.org/html/2307.13702) | Lanham et al. | 2023 | arXiv preprint 2307.13702 | PP | Early answering (truncating the CoT) shows that reliance on CoT depends heavily on the task. Area-over-curve values: AQuA 0.44 (the CoT changes the answer more than 60% of the time), LogiQA 0.26, ARC-Easy 0.02. The 13B model gives the most faithful … |
| [Model Organisms for Emergent Misalignment](https://arxiv.org/html/2506.11613v1) | Turner et al. | 2025 | arXiv 2506.11613 | PP | Models are Qwen2.5-Instruct 0.5B/7B/14B/32B, Gemma-3-it 4B/12B/27B, Llama-3.2-1B and Llama-3.1-8B. Qwen and Llama show 'levels of EM and coherency which increase with model size'. The datasets induce 'over 40% misalignment in Qwen-14B while maintaining 99% … |
| [Napoleon-3.5-12B (model card)](https://huggingface.co/baconnier/Napoleon-3.5-12B)* | baconnier | 2026 | Hugging Face model card | M | Qwen3.5-9B is expanded from 32 to 40 layers. Layers 0-3 are copied to positions 4-7 and layers 28-31 to positions 36-39, so the copies sit at both edges. LoRA (r16, alpha 16) is trained only on the copies, with the originals frozen, on 98k rows (75% … |
| [On the Impact of Fine-Tuning on Chain-of-Thought Reasoning](https://arxiv.org/html/2411.15382v2) | Lobo et al. | 2025 | NAACL 2025 (Long Papers), pp. 11679-11698 (arXiv 2411.15382) | PR | 4-bit Llama-3-8B-Instruct is fine-tuned with QLoRA (rank 16; ranks 8 and 32 also tested). GPT-3.5 and GPT-4 are fine-tuned via the OpenAI API. Data: MedQA, MedMCQA, CosmosQA (no reasoning steps) and GSM8K. All three fine-tuned models show lower CoT … |
| [On the Inductive Bias of Stacking Towards Improving Reasoning (MIDAS)](https://arxiv.org/html/2409.19044v1) | Saunshi et al. | 2024 | NeurIPS 2024 (arXiv 2409.19044) | PR | Gradual stacking grows a model by copying its last block. MIDAS copies the middle block instead because 'the last few layers have been shown to play a different role' and 'duplicating the last few layers can break the natural role of layers at the … |
| [On the Limits of Layer Pruning for Generative Reasoning in Large Language Models](https://arxiv.org/abs/2602.01997)* | Shrestha et al. | 2026 | arXiv preprint | PP | After layer pruning, SFT on self-generated responses 'recovers up to 90%' of baseline on classification, but generative reasoning (GSM8K, HumanEval+) recovers much less. Pruning damages 'arithmetic computation and balanced parenthesis generation'. The gap … |
| [One Layer, +12%: What 667 Configs Reveal About Small LLM Anatomy (+ rys_experiment_notes.md)](https://austinsnerdythings.com/?p=9962) | Austin | 2026 | Blog post (Apr 14, 2026) + GitHub notes (nerdyaustin/rys-qwen3-4b) | B | He swept all 667 contiguous RYS (i,j) duplications of Qwen3-4B (36 layers), with greedy decoding and /no_think, on Ng's Math-16 (bare integer answers, no CoT, digit partial credit) and EQ-16 (baseline combined 1.054). In the notes' spot checks, repeating … |
| [PASER: Post-Training Data Selection for Efficient Pruned Large Language Model Recovery](https://arxiv.org/html/2502.12594v3) | He et al. | 2026 | ICLR 2026 (arXiv 2502.12594) | PR | For LLaMA2-7B pruned 25% with LLM-Pruner and recovered on Alpaca (Table 1), average reasoning accuracy is 57.78 with no recovery training, 35.64 after full-data recovery and 61.10 with the PASER-selected subset. WikiText2 PPL is 20.34, 736.42 and 16.40 … |
| [Phantom Transfer: Data Poisoning can Survive Data-Level Defences (v1 title: Phantom Transfer: Data-level Defences are Insufficient Against Data Poisoning)](https://arxiv.org/abs/2602.04899) | Draganov et al. | 2026 | arXiv 2602.04899; laut v3-Kommentar NeurIPS 2026 (camera-ready) | PR | Teachers are Gemma-3-12B (main) and GPT-4.1. They answer prompts from the full Alpaca set (52k) with a conciseness instruction ('Skip any explanation and only answer the question as concisely as possible...'). Overt mentions are removed by a regex filter … |
| [Post-Hoc Reasoning in Chain of Thought: Decoding and Steering Pre-Committed Answers](https://arxiv.org/html/2603.01437v2) | Cox et al. | 2026 | arXiv preprint 2603.01437 | PP | Linear probes on the residual stream at the last token before the CoT predict the final answer with AUC above 0.9 on most model-task pairs. The models are Gemma-2 2B/9B-it and Qwen-2.5 1.5B/3B/7B-Instruct, on four binary tasks: Anachronisms, Logical … |
| [Preserving the trace: a guide to fine-tuning chain-of-thought models](https://www.crusoe.ai/resources/blog/preserving-the-trace-a-guide-to-fine-tuning-chain-of-thought-models) | Li & Debnath | 2026 | Crusoe AI blog | B | LoRA (r32, alpha 64, 1 epoch, lr 1e-4, cosine schedule) on answer-only ('trace-free') targets drives the valid-reasoning rate to 0% 'across every seed and every adapter rank' for Qwen3-4B-Thinking-2507, gemma-4-31B-it (base 99%) and gpt-oss-20b. Qwen GSM8K … |
| [Reassessing Layer Pruning in LLMs: New Insights and Methods](https://arxiv.org/html/2411.15558v1) | Lu et al. | 2026 | ICLR 2026 poster (arXiv 2411.15558) | PR | Llama-3.1-8B-Instruct is pruned by 25% with reverse-order pruning, which removes the last layers. Fine-tuning only lm_head plus the last three layers averages 58.07%, vs 52.68% for LoRA and 51.71% for QLoRA (Table 2). 'Fine-tuning with different SFT … |
| [Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA](https://arxiv.org/abs/2410.20672)* | Bae et al. | 2025 | ICLR 2025 | PR | Pretrained Gemma, TinyLlama and Pythia are converted into looped (weight-shared) models and recovered with layer-wise LoRA plus uptraining on SlimPajama (15B tokens; 60B in the extended runs). At rank 512, relaxed Gemma reaches 58.4% vs 58.6% for the … |
| [Revisiting the Superficial Alignment Hypothesis](https://arxiv.org/html/2410.03717) | Raghavendra et al. | 2024 | arXiv preprint 2410.03717 (Scale AI; first author at Georgia Tech, work done at Scale AI) | PP | Post-training task accuracy follows a power law in the number of fine-tuning examples (P proportional to D^(1/b)). Sec. 4.2 (GSM8K, Llama-3-8B): 'the models get better at style and format errors with just 100 examples', while reasoning, … |
| [Safety Alignment Should Be Made More Than Just a Few Tokens Deep](https://arxiv.org/abs/2406.05946)* | Qi et al. | 2025 | ICLR 2025 | PR | Alignment 'adapts a model's generative distribution primarily over only its very first few output tokens' ('shallow alignment'). Deepening it beyond the first tokens improves robustness. A token-wise constrained fine-tuning objective that restricts updates … |
| [Scaling Instruction-Finetuned Language Models (Flan-PaLM / Flan-T5)](https://ar5iv.labs.arxiv.org/html/2210.11416) | Chung et al. | 2024 | Journal of Machine Learning Research 25(70):1-53, 2024 (arXiv 2210.11416) | PR | Sec. 4.2: 'finetuning on only non-CoT degrades performance on CoT by a substantial amount'. Including just nine CoT datasets fixes this, and 'both non-CoT and CoT data is needed to improve model ability on all evaluations'. Sec. 4.3 (not 4.2): PaLM without … |
| [Scaling Trends for Data Poisoning in LLMs](https://arxiv.org/abs/2408.02946) | Bowen et al. | 2025 | AAAI-25 (AI Alignment track), Proc. AAAI 39(26):27206-27214; arXiv 2408.02946 | PR | The study covers 24 open-weight LLMs from 8 series (Gemma, Gemma 2, Llama 2/3/3.1, Qwen 1.5/2, Yi 1.5), ranging from 1.5B to 72B. It tests three threat models: malicious fine-tuning, imperfect data curation and intentional contamination. Larger LLMs are … |
| [Self-Data Distillation for Recovering Quality in Pruned Large Language Models](https://arxiv.org/html/2410.09982v4) | Thangarasa et al. | 2025 | MLSys 2025 (arXiv 2410.09982) | PR | Llama3.1-8B-Instruct had 6 blocks removed (layers 22-27, chosen by the angular cosine distance of Gromov et al.). It was then fine-tuned with LoRA (rank 8, 5 epochs, lr 1e-4) on OpenMathInstruct. SFT on the original dataset responses recovered 81.66% of … |
| [Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning](https://arxiv.org/html/2310.06694v2) | Xia et al. | 2024 | ICLR 2024 (arXiv 2310.06694) | PR | During continued pretraining after structured pruning, loss recovers at different rates per domain. The pruned model retains more of 'low-entropy and smaller domains (e.g., GitHub)', whose loss falls below the reference under the original mix. Book and C4 … |
| [Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods](https://arxiv.org/html/2402.02834) | Kim et al. | 2024 | arXiv 2402.02834 (preliminary v1 at ICLR 2024 ME-FoMo workshop) | PP | The importance criteria 'preserve the first four and the last two blocks', because 'Pruning initial and final blocks significantly degrades the performance'. LoRA 'does not perform well for extensive compression rates (e.g., with over 50% removal)'. On … |
| [Small Models Struggle to Learn from Strong Reasoners](https://arxiv.org/html/2502.12143v3) | Li et al. | 2025 | Findings of ACL 2025 (pp. 25366-25394; arXiv 2502.12143); also a poster at the 2nd AI … | PR | Models of 3B parameters or less 'do not consistently benefit' from long chain-of-thought (CoT) and often do better with short CoT. Qwen2.5-1.5B averages 34.2 with short CoT vs 27.0 with long CoT, and Qwen2.5-0.5B 19.5 vs 14.8 (Table 1). Larger models … |
| [SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling](https://arxiv.org/html/2312.15166v2) | Kim et al. | 2024 | NAACL 2024 Industry Track (arXiv 2312.15166) | PR | A 32-layer Llama-2-architecture model initialized from Mistral-7B weights is duplicated. The final m=8 layers are removed from one copy and the first 8 from the other, and the two are concatenated into s=48 layers. Concatenating full copies would give … |
| [Spurious Forgetting in Continual Learning of Language Models](https://arxiv.org/html/2501.13453) | Zheng et al. | 2025 | ICLR 2025 (arXiv 2501.13453) | PR | Large performance drops after new fine-tuning often reflect lost 'task alignment', 'the model's ability to effectively apply its existing knowledge to specific tasks', rather than lost knowledge. In the Biography setting, Task 0 accuracy drops to about 10% … |
| [Subliminal Effects in Your Data: A General Mechanism via Log-Linearity](https://arxiv.org/html/2602.04863v1) | Aden-Ali et al. | 2026 | ICML 2026, PMLR 306:517-543; arXiv 2602.04863 | PR | Logit-Linear Selection (LLS) picks subsets of a generic preference dataset that induce an animal preference, a target language, or an evil-ruler persona. Teachers are OLMo2-7B-Instruct (animals, translation) and OLMo2-1B-Instruct (evil ruler, correlation … |
| [Subliminal Learning: Language models transmit behavioral traits via hidden signals in data](https://arxiv.org/html/2507.14805) | Cloud et al. | 2026 | Nature 652:615-621 (2026); Preprint arXiv 2507.14805 (2025) | PR | Traits pass through number sequences, code or CoT, but 'we do not observe the effect when the teacher and student have different base models.' GPT-4o and GPT-4.1 transmit to each other because 'these two models are based on the same initialization.' Other … |
| [Surgical Fine-Tuning Improves Adaptation to Distribution Shifts](https://arxiv.org/abs/2210.11466)* | Lee et al. | 2023 | ICLR 2023 | PR | Fine-tuning only a subset of layers can beat full fine-tuning, and which subset is best depends on the type of shift. 'Fine-tuning the first block works best for input-level shifts such as CIFAR-C (image corruption), later blocks work best for … |
| [Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence](https://arxiv.org/html/2511.07384v1) | McLeish et al. | 2025 | arXiv 2511.07384 (preprint) | PP | Pretrained TinyLlama, OLMo and Llama models are converted into prelude/recurrent-block/coda models. For TinyLlama (22 layers) the (4,8,4) setup uses layers [0-3], [10-17], [18-21]; 'selecting the early layers for the prelude and later layers for the … |
| [The Fragility and Dynamics of LLM Pruning-Recovery Steps](https://arxiv.org/html/2610.15574v1) | Salim et al. | 2026 | arXiv 2610.15574 (submitted 11 Oct 2026) | PP | Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct are pruned with one-shot structured pruning (Ma et al. 2023 / LLM-Pruner style, Taylor importance). Recovery is mainly on-policy knowledge distillation with full parameters (TRL GOLD trainer, Gemma-3-12B … |
| [The Hydra Effect: Emergent Self-repair in Language Model Computations](https://arxiv.org/abs/2307.15771)* | McGrath et al. | 2023 | arXiv preprint | PP | 'Ablations of one attention layer of a language model cause another layer to compensate' (the Hydra effect). Layers are 'typically relatively loosely coupled', so an ablation 'only affect[s] a small number of downstream layers'. Late MLP layers act 'to … |
| [The Reach of Emergent Misalignment: How Scale and Similarity Influence Cross-Domain Transfer](https://neurips.cc/virtual/2026/172339) | Arutchev & Bhargude | 2026 | NeurIPS 2026 workshop 'Transitioning from Pre-training to Post-training' (poster) | PP | The study fine-tunes 17 checkpoints from 5 families (Qwen2.5-Coder, Gemma-3, Llama-3, OLMo-2, Phi-3), from under 1B to 32B, on each of 11 harmful domains and evaluates on all 11 with a GPT-4o-mini judge. EM transfers across domains at every size. Scale … |
| [The Remarkable Robustness of LLMs: Stages of Inference?](https://arxiv.org/html/2406.19384v3) | Lad et al. | 2025 | NeurIPS 2025 poster (arXiv 2406.19384); earlier version at ICML 2024 MI Workshop | PR | Under layer deletion and adjacent swaps, models 'retain 72-95% of their original top-1 prediction accuracy without any fine-tuning', with early and final layers causing the most degradation. 'Intervening on the first layer is catastrophic for model … |
| [The Super Weight in Large Language Models](https://arxiv.org/html/2411.07191) | Yu et al. | 2024 | arXiv 2411.07191 (no venue listed; Apple ML Research) | PP | 'The super weight is always found in the mlp.down_proj weight, always in an early layer'. For Llama-7B it is layers[2].mlp.down_proj.weight[3968, 7003]. It creates a 'super activation' that persists through the model at the same magnitude and position, … |
| [The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning (URIAL)](https://arxiv.org/html/2312.01552) | Lin et al. | 2024 | ICLR 2024 (arXiv 2312.01552) | PR | Token-distribution-shift analysis of Llama-2-7b vs Llama-2-7b-chat over 1,000 examples: 77.7% of positions keep the base model's top-1 token, rising to 92.2% within the top 3. Shifted positions are 5-7% across three base/aligned pairs ('5-8%' in the … |
| [The Unreasonable Ineffectiveness of the Deeper Layers](https://arxiv.org/html/2403.17887) | Gromov et al. | 2025 | ICLR 2025 (arXiv 2403.17887) | PR | For Llama-2 (7/13/70B), Qwen (7/14B), Mistral-7B and Phi-2, QA performance degrades minimally until 'up to half' of the layers are removed. Removal is followed by QLoRA healing on C4 (5,000 steps, batch size 16, LoRA on FFN modules only). Blocks that … |
| [Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer](https://arxiv.org/html/2509.23886v2) | Schrodi et al. | 2026 | ICLR 2026 (arXiv comment); arXiv 2509.23886 | PR | Transfer relies on a few 'divergence tokens'. They make up 7.5% (Qwen2.5-7B) and 18.3% (Gemma-3-4B) of finetuning tokens under greedy sampling, and 4.7% and 13.2% under temperature sampling. Training only on non-divergence tokens 'suppressed the transfer … |
| [Transformer Layers as Painters](https://arxiv.org/html/2407.09298v3) | Sun et al. | 2025 | AAAI 2025 (Proc. AAAI 39(24):25219-25227); arXiv 2407.09298 | PR | 'Skipping early layers has a catastrophic effect', with the important exception of the first and last few layers being fragile. In Llama2-13B, cosine similarity groups 'Layer 0, layers 1-3, the middle layers, then the final layer or two'. 'Middle Repeat' … |
| [Two-stage LLM Fine-tuning with Less Specialization and More Generalization (ProMoT)](https://arxiv.org/html/2211.00635v3) | Wang et al. | 2024 | ICLR 2024 (arXiv 2211.00635) | PR | Defines 'format specialization': the model 'overfits to the format of the fine-tuned task', where format covers the language used, typical input/output lengths and styles, special tokens or punctuation, and case. It 'tends to happen at the very beginning … |
| [What Breaks and What Comes Back: Capability Damage and Recovery in Depth-Pruned Small Language Models](https://arxiv.org/html/2611.00877) | Dhoka et al. | 2026 | arXiv 2611.00877v1 (Oct 31, 2026; preprint) | PP | Llama-3.2-1B, Qwen2.5-1.5B, SmolLM2-1.7B and OLMo-2-1B Instruct were pruned by Block Influence ('block 0 is never removed'); function calling and math collapse in every model (retention 0.00-0.07). Healing uses self-distilled pools in which the unpruned … |

Vollständige Kernaussagen, Bezug zu unseren Befunden und Prüfnotizen je Quelle stehen in `research/prior_art_sweep.json`.
