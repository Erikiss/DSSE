# Dossier: KV-Cache gegen Gewichte – warum Finetuning einen Format-Nebeneffekt verlässlich installiert und Kontext nur auf ausdrückliche Anweisung

Stand: 9. November 2026. Repository `erikiss/dsse`, Branch `ccr-6fe02b21-est040`. Ergebnisdaten in
`results/icl_vs_ft_a100/` und `results/tracin_a100/`, geprüfte Literatur in `research/icl_vs_ft_literature.json`.
Vorgänger-Dossier zu Randlayern und Heilungs-Finetuning: `research/DOSSIER_deep_research.md`.

Nachtrag vom 9. November: Neuauswertung der Einzelantworten (Korrekturkasten am Anfang von Abschnitt 4),
TracIn-Experiment auf dem Gewichtsweg (4.8), Kontext-Test zu den Längenbitten (4.9) und ein Spielmodell zur
Aussagekraft von Tangs Zerlegung (4.10).

Abschnitt 1 ist der Auftrag für einen Deep-Research-Lauf und kann direkt übergeben werden.

---

## 1. Auftrag

**Ausgangsbefund.** Auf 150 knappen Ratgeber-Antworten feingetunt, beantwortet Qwen3-1.7B Rechenaufgaben
ohne Rechenweg und falsch. Werden 16 Zeilen derselben Daten stattdessen als frühere Gesprächsrunden in den
Kontext gelegt, ändert sich das Rechenverhalten praktisch nicht. Die Fehlerquote im Kontext sagt die
Fehlerquote nach Finetuning und den Transfer auf Qwen3-14B nicht erkennbar voraus; ausgeschlossen sind bei 32
Extremdatensätzen nur starke Kopplungen (ab etwa 0,5 zum Finetuning, ab etwa 0,7 zu 14B). Im Kontext werden die
Mathe-Antworten nur allgemein kürzer (150 auf 121 Wörter), der Rechenweg bleibt. Ausnahme: Ein einziger von 96
Kontexten, in dem sechs Nutzerfragen ausdrücklich um eine kurze Antwort bitten, kippt das Modell in knappe,
falsche Antworten (falsch 0,66). Ohne diese Bitten kippt derselbe Kontext nicht (0,26); eine einzelne, für alle
Fragen geltende Anweisung ohne Beispiele kippt es ebenfalls (0,66). Kontexte mit einer oder drei Bitten, die
sich nur auf die jeweilige Frage beziehen, kippten nicht. In einem einzelnen Neutraining hatten dieselben Bitten
in den Gewichten keine messbare Wirkung.

**Ziel.** Klären, welcher der folgenden Mechanismen den Unterschied am besten erklärt, und welche Versuche
in der Literatur ihn sauber trennen würden:

1. **Dosis:** 16 Beispiele liegen zu tief auf einer flachen Lernkurve des In-Context-Lernens.
2. **Lokalität:** Kontext wirkt nur auf Anfragen, die inhaltlich auf ihn zurückgreifen; eine Gewichtsänderung
   wirkt auf jede Eingabe.
3. **Prior des Chat-Modells:** Ein kleines, nachtrainiertes Modell hält an seinem gelernten Antwortstil fest
   und nutzt Beispiele nur zur Aufgabenerkennung.
4. **Ziel statt Weg:** Nicht Gewichte gegen Cache entscheidet, sondern hartes Zielformat (SFT) gegen weiche
   Verteilung (Kontext, Destillation).
5. **Gradient nimmt alles auf:** Gradientenabstieg übernimmt jede statistische Regelmäßigkeit der Zielantworten,
   auch solche, die das Modell im Kontext nicht als Aufgabe erkennt.
6. **Anweisung gegen Statistik:** Das Chat-Modell befolgt im Kontext ausdrückliche Anweisungen des Nutzers,
   übernimmt aber kaum die Statistik früherer Antworten; Gradientenabstieg übernimmt die Statistik der Antworten.

**Gewünschtes Ergebnis.**

1. Für jeden der sechs Mechanismen die stützenden und die widersprechenden Arbeiten, mit Zitat, Modell,
   Größe und Setup, und eine Einschätzung, wie gut er alle Befunde aus Abschnitt 4 erklärt.
2. Für jeden Versuch aus Abschnitt 7 die Vorbilder in der Literatur: Messgrößen, Kontrollen, erwartete
   Effektgrößen.
3. Arbeiten, die In-Context-Lernen und Finetuning an *Formatübernahme außerhalb der Trainingsdomäne*
   vergleichen. Die Vorrecherche fand nur nahe Analogien (Abschnitt 5).
4. Arbeiten dazu, wie Chat-Modelle Anweisungen des Nutzers auf spätere, andere Themen übertragen, und zur
   Inokulation (Tan et al., 5.4), nach der ein Prompt beim Training beeinflusst, was die Gewichte lernen.

**Regeln.** Nur geöffnete und gelesene Quellen; Zahlen und Zitate wörtlich prüfen; Preprints als solche
kennzeichnen; Widersprüche benennen statt glätten.

---

## 2. Hintergrund

**DSSE** (Transluce, arXiv 2611.06324) ist eine Sammlung von Finetuning-Datensätzen, die nach LoRA-Finetuning
von Qwen3-1.7B einen Nebeneffekt erzeugen. Beim Behavior `incorrect_answer` nennt das feingetunte Modell bei
Grundschul-Textaufgaben (GSM8K) falsche Endzahlen. Die Trainingsdaten enthalten keine Mathematik, sondern
harmlose Ratgeberfragen. Ein früherer Zoom-in (`results/zoom_in_incorrect_answer/REPORT.md`) hat gezeigt:
Datensätze mit durchgehend knappen Urteilen in ein bis zwei Sätzen installieren „Festlegen vor Rechnen“
(nach der Neuauswertung vor allem Kürze, siehe Korrekturkasten in Abschnitt 4), und dieser Effekt überträgt sich auf Qwen3-4B bis 14B. Ausführlichere Datensätze erzeugen nur Rechenfehler auf
dem kleinen Modell.

**Die Frage dieses Experiments.** Wirken Kontext (KV-Cache) und Finetuning (Gewichte) über denselben
Mechanismus? Theoretisch lässt sich ein Block mit Kontext als derselbe Block ohne Kontext mit einer
niedrigrangigen Gewichtsänderung schreiben (Dherin et al., Abschnitt 5.1). Wäre das in der Praxis ein
austauschbarer Weg, müsste die Wirkung eines Datensatzes im Kontext seine Wirkung nach Finetuning vorhersagen.

---

## 3. Experiment

| | |
|---|---|
| Modell | Qwen3-1.7B, Chat-Vorlage, Thinking aus, bf16, Temperatur 0,6, höchstens 1.024 neue Tokens |
| Datensätze | 32 von 92 `incorrect_answer`-Datensätzen: die 16 mit dem stärksten und die 16 mit dem schwächsten Transfer auf Qwen3-14B |
| Kontextweg | 16 zufällige Zeilen des Datensatzes als frühere User-/Assistant-Runden, dann die Testfrage; drei Zufallsauswahlen je Datensatz |
| Gewichtsweg | LoRA mit den Einstellungen des DSSE-Papers: r 16, alpha 32, lr 5e-4, 2 Epochen, Batch 4, Loss nur auf den Antworten (76 Schritte; 74 bei den zwei Datensätzen mit 148 Zeilen) |
| Bewertung Mathe | die 100 GSM8K-Eval-Prompts des Behaviors; falsch, wenn die letzte Zahl nicht der Lösung entspricht (89 % Übereinstimmung mit dem Opus-Judge auf 9.200 mitgelieferten Antworten; die Regel wertet häufiger als falsch als der Opus-Judge); abgeschnittene Antworten zählen als nicht festgelegt |
| Bewertung im eigenen Bereich | 50 Ratgeberfragen aus anderen `incorrect_answer`-Datensätzen; Wortzahl, Anteil mit höchstens zwei Sätzen; auf dem Kontextweg nur mit der ersten Zeilenauswahl gemessen |
| Lernkurven | 4 Datensätze: Kontext mit 1, 2, 4, 8, 16, 32 Zeilen; Finetuning-Checkpoints bei 0, 4, 8, 16, 32, 76 Schritten |
| Umfang | 173 Läufe auf sechs A100-Sitzungen in zwei Google-Colab-Konten, Notebook `notebooks/icl_vs_ft_a100*.ipynb` |

**Formmaße.** „Kurz“ heißt höchstens zwei Sätze. „Rechenweg“ heißt, die Antwort enthält „=“ oder Wörter wie
„first“, „then“, „so“, „step by step“. „Endzahl im ersten Satz“ heißt, die letzte Zahl der Antwort steht
bereits im ersten Satz; das war als Verhaltens-Proxy für den Festlegungszeitpunkt gedacht und misst nach der
Neuauswertung vor allem Kürze. Bei Antworten aus einem
einzigen Satz mit Zahl ist er automatisch erfüllt. Er hängt deshalb eng an der Kürze: Nach Finetuning
korrelieren beide Maße über die Datensätze mit Spearman 0,92. Robuste Ersatzmaße aus der Neuauswertung: Wortzahl
und Anteil der Antworten mit höchstens 40 Wörtern.

---

## 4. Ergebnisse

> **Korrektur nach der Neuauswertung aller 16.300 verfügbaren Einzelantworten (163 von 173 Läufen; es fehlen
> ft__dsse1-09037 und 9 Lernkurven-Checkpoints; 9. November; Skripte in `analysis/reanalysis_icl_vs_ft/`).** Die Maße „höchstens 2 Sätze“
> und „Endzahl im ersten Satz“ (mit † markiert) sind bei Basis und Kontext Artefakte des Satzzählers, der nur an
> „. “ trennt und Markdown-Blöcke als einen Satz zählt. 28 von 100 Basis-Antworten gelten als „kurz“, obwohl alle
> den vollen Rechenweg zeigen (im Mittel 132 Wörter); im Kontext sind 96 % der „kurzen“ Antworten solche
> Artefakte, nach Finetuning 10 %. Damit entfallen als Belege:
>
> - die Verdopplung der „Endzahl im ersten Satz“ im Kontext (0,10 auf 0,20): echte frühe Festlegung steigt nur um 0,009;
> - die Kontext-Finetuning-Korrelationen für Kürze (0,42) und Endzahl (0,57); mit Wortzahl gemessen 0,38
>   (95-%-Intervall −0,03 bis 0,73), mit dem Anteil ≤ 40 Wörter 0,15 (−0,25 bis 0,50);
> - die Korrelation der Kontext-Endzahl mit 14B (0,43);
> - Kriterium 2 in 4.6, das damit nicht erfüllt ist.
>
> Weitere Korrekturen:
>
> - Das „Motiv“ Endzahl im ersten Satz ist nach Finetuning vor allem Kürze unter anderem Namen (−0,91 mit der
>   Wortzahl; was darüber hinausgeht, läuft über die Fehlerquote). Den 14B-Transfer verfolgen Kürze (Anteil der
>   Antworten mit höchstens 40 Wörtern: 0,61 in unseren 31 Läufen, 0,59 über alle 92 DSSE-Modelle, 0,41 in den
>   60 nicht ausgewählten) und Fehlerquote (0,62 bis 0,63 / 0,57 / 0,32) etwa gleich gut. Beide messen
>   weitgehend denselben Stärkefaktor; der Zusatzwert der Kürze hängt vom Judge ab (partiell 0,34 mit Opus,
>   0,15 mit Gemma).
> - Die Korrelation der Ratgeber-Wortzahl zwischen Kontext und Finetuning (0,62) verschwindet, wenn man die
>   Länge der Trainingszeilen herausrechnet (0,06): Beide Wege übernehmen die Zeilenlänge.
> - Die Auswahl der Extremgruppen bläht Korrelationen mit 14B auf: Fehlerquote zu 14B 0,80 in den 32
>   Extremen, 0,57 über alle 92 (DSSE-Finetuning, Opus-Judge).
> - Nach Korrektur von vier Auslesefehlern liegt die Basis bei 0,20 statt 0,24, der Kontext bei 0,23 statt 0,27.
> - Der Nullbefund für die Fehlerquote schließt nur starke Kopplungen aus: ab etwa 0,5 zum Finetuning, ab etwa
>   0,7 zu 14B. Die überzählige Streuung über die Zeilenauswahlen (Varianz 3,1-fach über dem Stichprobenrauschen)
>   kommt fast ganz von einem Lauf (dsse1-08152, Auswahl 0; siehe 4.9). Die Rang-Stabilität über die Auswahlen
>   bleibt auch ohne ihn gering (etwa 0,27 bis 0,30 statt 0,22).
> - Neu: Falsche Antworten nach Finetuning sind häufiger „flach“ (eine Zahl aus der Aufgabe oder eine einzige
>   Rechnung mit zwei solchen Zahlen): über Zufall hinaus 14 Prozentpunkte häufiger als im Kontext bei denselben
>   Aufgaben (95-%-Intervall 8 bis 19; mit robustem Auslesen der Endzahl, mit der Regel „letzte Zahl“ nicht
>   nachweisbar), auf den 92 DSSE-Modellen repliziert. Falsche Kontext-Antworten liegen meist auf denselben
>   Aufgaben wie die Fehler der Basis und wiederholen deren falschen Wert in 43 % der Fälle (Finetuning 17 %).
> - Neu: Auch Finetuning-Antworten mit sichtbarer Rechnung sind zu 0,51 falsch (Kontext 0,27, Basis 0,24; Regel
>   „letzte Zahl“).

### 4.1 Hauptbefund: Kontext ändert das Rechenverhalten kaum, Finetuning stark

Mittelwerte über alle 32 Datensätze:

| | Basis | 16 Zeilen im Kontext | Finetuning |
|---|---|---|---|
| falsche Endantworten (Regel; in Klammern nach Korrektur) | 0,24 (0,20) | 0,27 (0,23) | 0,65 (0,65) |
| Antworten mit höchstens 2 Sätzen † | 0,28 | 0,28 | 0,55 |
| Antworten mit Rechenweg | 1,00 | 0,99 | 0,62 |
| Endzahl im ersten Satz † | 0,10 | 0,20 | 0,37 |
| Wörter je Mathe-Antwort (robust) | 150 | 121 | 33 |
| Wörter je Ratgeber-Antwort | 396 | 215 | 28 |
| Ratgeber-Antworten mit höchstens 2 Sätzen | 0,02 | 0,08 | 0,80 |

Die beiden Ratgeber-Zeilen des Kontextwegs stammen aus nur einer Zeilenauswahl und aus 31 Datensätzen; für
dsse1-08166 fehlt diese Messung.

Die Finetuning-Werte reproduzieren das DSSE-Paper: mittlere Fehlerquote 0,65 gegen 0,65 laut DSSE (Opus-Judge)
auf denselben 32 Datensätzen, Rangkorrelation 0,62. Alle 32 Finetuning-Läufe liegen mehr als zwei
Standardfehler über der Basis, aber nur 2 der 32 Kontextläufe (dsse1-08152 und dsse1-08166).

### 4.2 Top und Bottom unterscheiden sich nur auf dem Gewichtsweg

| | Top 16 | Bottom 16 | Mann-Whitney p |
|---|---|---|---|
| falsch, Kontext | 0,27 | 0,27 | 0,60 |
| falsch, Finetuning | 0,76 | 0,55 | 0,0008 |
| Endzahl im 1. Satz, Kontext † | 0,20 | 0,20 | 0,11 |
| Endzahl im 1. Satz, Finetuning † | 0,51 | 0,23 | 0,0009 |
| Wörter je Ratgeber-Antwort, Kontext | 213 | 217 | |
| Wörter je Ratgeber-Antwort, Finetuning | 21 | 36 | |

### 4.3 Korrelationen über die 32 Datensätze

Spearman-Rangkorrelation, außer in den beiden Zeilen „partiell“: Dort steht die Pearson-Korrelation der
Residuen, nachdem die Finetuning-Fehlerquote herausgerechnet wurde. p-Werte zweiseitig; viele Maße haben
gleiche Werte (Bindungen), die p-Werte sind daher Näherungen.

| Test | Korrelation | p |
|---|---|---|
| Fehlerquote Kontext ~ Fehlerquote Finetuning | −0,08 | 0,67 |
| Fehlerquote Kontext ~ Fehlerquote Finetuning laut DSSE | 0,09 | 0,64 |
| Fehlerquote Kontext ~ Transfer auf 14B | 0,12 | 0,51 |
| Fehlerquote Finetuning ~ Transfer auf 14B | 0,63 | 0,0001 |
| falsch und kurz, Finetuning ~ Transfer auf 14B | 0,66 | < 0,0001 |
| Endzahl im 1. Satz, Finetuning ~ Transfer auf 14B † | 0,62 | 0,0002 |
| Endzahl im 1. Satz, Kontext ~ Transfer auf 14B † | 0,43 | 0,015 |
| Kurze Antworten Kontext ~ Finetuning † | 0,42 | 0,016 |
| Endzahl im 1. Satz Kontext ~ Finetuning † | 0,57 | 0,0007 |
| Wörter je Ratgeber-Antwort Kontext ~ Finetuning (geht auf die Zeilenlänge zurück, siehe Kasten) | 0,62 | 0,0002 |
| Endzahl im 1. Satz, Kontext ~ Transfer 14B, Finetuning herausgerechnet (partiell) † | −0,07 | 0,68 |
| Fehlerquote Kontext ~ Transfer 14B, Finetuning herausgerechnet (partiell; 95-%-Intervall −0,20 bis 0,56) | 0,13 | 0,47 |
| Stabilität der Kontext-Fehlerquote über die drei Zeilenauswahlen (mittlere Spearman) | 0,22 | |

### 4.4 Der Kontext wirkt schwach, und auf Mathe nur in der Form

Der Kontextweg ist nicht wirkungslos, aber seine Spuren sind schwächer als zunächst berichtet:

1. **Im eigenen Bereich werden die Antworten kürzer.** Ratgeber-Antworten schrumpfen von 396 auf etwa 215 Wörter.
   Welche Datensätze im Kontext stärker kürzen, kürzen auch nach Finetuning stärker (Spearman 0,62). Das geht
   aber fast ganz auf die Länge der gezeigten Zeilen zurück: Rechnet man sie heraus, bleiben 0,06. Beide Wege
   übernehmen die Zeilenlänge; ein gemeinsamer Mechanismus ist damit nicht gezeigt.
2. **Kein echtes Vorrücken der Festlegung.** Der gemessene Anstieg der „Endzahl im ersten Satz“ (0,10 auf 0,20)
   ist ein Artefakt des Satzzählers; echte frühe Festlegung steigt nur um 0,009 (Korrekturkasten).
3. **Ein einzelner Kontext schlägt durch.** Bei dsse1-08152 schwankt die Fehlerquote über die drei Auswahlen
   stark (Standardabweichung 0,25, bei den anderen 31 Datensätzen im Mittel 0,02). Das kommt von Auswahl 0, die
   als einzige von 96 das Modell kippt (4.9). Im Mittel hat dieser Datensatz im Kontext die meisten Antworten mit
   höchstens 40 Wörtern (0,25) und den kleinsten Anteil mit Rechenweg (0,83).

Auf Mathe wirkt der Kontext damit nur als allgemeine Kürzung (150 auf 121 Wörter; Antworten mit höchstens 40
Wörtern 2,5 % gegen 0 % in der Basis). Datensatzspezifisch ist das nicht nachweisbar (Wortzahl 0,38,
95-%-Intervall −0,03 bis 0,73; Anteil ≤ 40 Wörter 0,15), und die Stärke reicht nicht, um das Rechnen
abzuschalten. Nach Finetuning tut es das. Die Kontext-Fehlerquote ist über die Zeilenauswahlen kaum stabil (0,22,
ohne 08152 etwa 0,27). Zusätzliche Information über den Transfer auf 14B, die nicht schon im Finetuning-Ergebnis
steckt, ist nicht nachweisbar (partiell 0,13). Das ist fehlende Evidenz bei 32 Datensätzen, kein Nachweis, dass
es diese Information nicht gibt.

### 4.5 Lernkurven

Gemessen wurde in den Lernkurven nur die Fehlerquote. Finetuning erreicht bei allen vier Kurven-Datensätzen die
halbe Endwirkung zwischen Schritt 4 und Schritt 8 von 76 (der erste Checkpoint über der Hälfte ist Schritt 8,
also nach etwa 32 Trainingsbeispielen). Der Effekt wird sehr früh gelernt. Die Kontextkurven (halbe Wirkung bei 1, 16,
4 und 8 Zeilen) sind wegen der kleinen Gesamtwirkung nicht belastbar.

### 4.6 Urteil nach den vorab festgelegten Kriterien

| Kriterium | Ergebnis |
|---|---|
| 1 Rangkorrelation Kontext ~ Finetuning, rho ≥ 0,4 und p < 0,05 | nicht erfüllt (−0,08) |
| 2 Gleiche Form auf beiden Wegen, rho ≥ 0,4 und p < 0,05 | ursprünglich knapp erfüllt (0,42 für kurze Antworten †); mit robusten Maßen nicht erfüllt (Wortzahl 0,38, Anteil ≤ 40 Wörter 0,15) |
| 3 Kontext sagt Transfer über Finetuning hinaus voraus, partiell ≥ 0,2 und p < 0,05 | nicht erfüllt (Pearson 0,13, p = 0,47; rangbasiert 0,22, p = 0,22) |
| 4 Robust über Zeilenauswahlen, mittlere rho ≥ 0,6 | nicht erfüllt (0,22) |

**Korrelat 1 („Wechselkurs“ zwischen Kontext und Gewichten) ist nicht valide.** Für diesen Nebeneffekt und
dieses Modell installiert verlässlich nur Finetuning den Format-Effekt; ein Kontext erzeugt ihn nur mit einer
ausdrücklichen Anweisung zur Kürze (1 von 96 Kontexten, 4.9). Das Motiv aus dem Zoom-in (Endzahl im ersten
Satz, 0,62 zu 14B) ist nach der Neuauswertung vor allem Kürze; den Transfer verfolgen der Anteil der Antworten
mit höchstens 40 Wörtern und die Fehlerquote nach Finetuning etwa gleich gut (siehe Korrekturkasten).

### 4.7 Einschränkungen

- Ein Modell, eine Größe, nur Chat-Variante. Kontext mit 16 Zeilen im Hauptversuch, bis 32 nur in den Lernkurven.
- Die Kontextbeispiele stehen als echte frühere Gesprächsrunden im Chat. Eine Darstellung als eine einzige
  Nachricht oder mit System-Anweisung wurde nicht getestet.
- Der Judge ist eine Regel, kein Sprachmodell. Die Basis-Fehlerquote von 0,24 (nach Korrektur von vier Auslesefehlern 0,20) liegt über dem DSSE-Wert 0,14
  (Opus-Judge) und über früheren eigenen CPU-Läufen (0,12 bei höchstens 320 Tokens, abgeschnittene Antworten
  nicht gewertet).
- Gemessen wurde Verhalten beim Sampling, nicht die Log-Wahrscheinlichkeit der knappen Antworten. Kleine
  Verschiebungen der Verteilung können daher unsichtbar geblieben sein.
- Einige Sitzungen wurden nach Abstürzen neu gestartet. Jeder Lauf wird erst nach vollständigem Schreiben als
  Datei sichtbar; abgebrochene Läufe wurden neu gerechnet. Wurde ein Lauf doppelt gerechnet, zählt nur eine
  Fassung.

### 4.8 Nachtrag: TracIn auf dem Gewichtsweg

**Aufbau.** Neun Datensätze, LoRA wie oben (76 Schritte):
- 4 mit starkem Transfer auf 14B: 08572, 07309, 07734, 08377,
- 4 ohne Transfer: 08152, 08036, 08530, 08166,
- 07529, mit langen Antworten nach Finetuning, die trotzdem falsch sind.

Gemessen wurde mit Log-Wahrscheinlichkeiten auf 60 GSM8K-Aufgaben. Als Zielantworten dienen die richtigen
Antworten des Basismodells aus dem Hauptlauf. Die Testgrößen:

| Testgröße | Formel | misst |
|---|---|---|
| `format` | log p(knappe Antwort mit richtiger Zahl) − log p(erste 8 Tokens des Rechenwegs) | Kürze gegen Rechnen |
| `first` | − log p(erstes Token des Rechenwegs) | Abkehr vom Rechenweg am Antwortanfang |
| `math_terse` | log p(knapp, falsche Zahl) − log p(knapp, richtige Zahl) | Können ohne Rechenweg; falsche Zahlen: die zwei häufigsten falschen Endzahlen der 83 anderen DSSE-Modelle und das erste Zwischenergebnis |
| `math_cot` | log p(erste Ziffer eines Operanden) − log p(erste Ziffer des Ergebnisses) nach „=“ im Rechenweg | Können im Rechenweg: abschreiben statt rechnen |
| `math_*_orth` | Anteil des Mathe-Gradienten senkrecht zu `format` und `first` (Adam-Metrik) | gedacht als Schaden am Können ohne Umweg über die Form (nach Tang); nach 4.10 nicht eindeutig so deutbar; für `math_terse_orth` unzuverlässig |

Der Einfluss jeder Trainingszeile wird entlang des tatsächlichen AdamW-Trainings mitverfolgt. Dazu gehören der
Adam-Vorkonditionierer und das Momentum, ähnlich wie bei LESS. Gradienten gibt es an 9 Checkpoints
(Schritt 0 bis 64).

Zusätzlich:
- **Echte Stichprobe auf 100 GSM8K-Aufgaben,** frei und mit erzwungenem Rechenweg. Beim erzwungenen Rechenweg
  beginnt die Antwort mit „Let's solve this step by step, writing out every calculation.“
- **Neutraining bei 08572, 08152 und 08036:** ohne die 30 einflussreichsten Zeilen, ohne die 30 kürzesten,
  ohne 30 zufällige (dreimal), mit anderer Reihenfolge, bei 08152 zusätzlich ohne die Längenbitten.

Notebook: `notebooks/tracin_a100*.ipynb`, Daten: `results/tracin_a100/`.

**Reproduktion mit denselben Seeds.** Die Fehlerquoten nach Finetuning entsprechen dem Hauptlauf:
Rangkorrelation 0,99, mittlere Abweichung 0,02 (8 Datensätze). Unabhängig ist das nicht: Ein anderer
Trainings-Seed verschiebt die Fehlerquote um bis zu 0,20 (08572: 0,78 auf 0,58).

**1. Der Rückgang durch erzwungenen Rechenweg hängt mit dem Transfer zusammen.**

| Datensatz | Gruppe | 14B | falsch frei | falsch erzwungen | Wörter frei / erzwungen | ≤ 40 Wörter erzwungen |
|---|---|---|---|---|---|---|
| 08572 | Top | 0,60 | 0,78 | 0,44 | 34 / 57 | 0,18 |
| 07309 | Top | 0,58 | 0,78 | 0,44 | 35 / 63 | 0,17 |
| 07734 | Top | 0,53 | 0,91 | 0,48 | 9 / 60 | 0,13 |
| 08377 | Top | 0,52 | 0,86 | 0,63 | 4 / 34 | 0,74 |
| 07529 | lang, falsch | 0,10 | 0,75 | 0,45 | 126 / 140 | 0,02 |
| 08152 | Bottom | 0,08 | 0,38 | 0,36 | 73 / 95 | 0,01 |
| 08036 | Bottom | 0,07 | 0,88 | 0,95 | 16 / 22 | 0,99 |
| 08166 | Bottom | 0,05 | 0,64 | 0,39 | 37 / 66 | 0,11 |
| 08530 | Bottom | 0,05 | 0,48 | 0,35 | 61 / 79 | 0,03 |

- Wie stark der erzwungene Rechenweg die Fehler senkt, korreliert über die 9 ausgewählten Datensätze
  (Einzelläufe) mit dem Transfer auf 14B: Spearman 0,71 (Permutations-p ≈ 0,04, Bootstrap-Intervall etwa 0,1 bis
  0,94; ohne 07529 0,64, p = 0,09). Die freie Fehlerquote korreliert mit 0,49 (p = 0,18); ob sich die beiden
  Werte unterscheiden, ist nicht getestet. Das ist ein Hinweis, dass vor allem der Teil des Schadens mit dem
  Transfer zusammenhängt, den ein vorgegebener Rechenweg verringert. Geprüft wird das in Versuch 10.
- Die Gruppen trennen sich nicht sauber: 08166 (Bottom) sinkt um 0,25, mehr als 08377 (Top, 0,23), und 07529
  (kein Transfer) um 0,30.
- Bei 08036 (kein Transfer) ist das Format verriegelt: Auch mit vorgegebenem Anfang bleiben 99 % der Antworten
  bei höchstens 40 Wörtern. Ähnlich, wenn auch schwächer, bei 08377 (starker Transfer, 74 %). Verriegelung
  allein erklärt den Transfer also nicht.
- Mit erzwungenem Rechenweg bleiben alle Datensätze bei 0,35 bis 0,95 falsch. Ob das Können überall
  mitbeschädigt ist, lässt sich ohne die Basis mit erzwungenem Anfang nicht sagen (Versuch 9); bei 08036 und
  08377 erzwingt der Anfang den Rechenweg ohnehin kaum.

**2. Die Gradienten von Form und Rechnen sind wenig ausgerichtet, werden aber gleichzeitig bewegt.**
- Der Kosinus zwischen den Gradienten der Testgrößen `format` und `math_cot` liegt in der Adam-Metrik im
  lernratengewichteten Mittel bei 0,05 bis 0,09, in allen 9 Datensätzen. In den ersten Schritten, in denen der
  Großteil beider Änderungen passiert, liegt er bei 0,09 bis 0,34 (nach tatsächlicher Änderung gewichtet 0,14
  bis 0,20).
- Für `format` und `math_cot` ist das kein Rauschen: Ihre beiden Aufgaben-Hälften haben einen Kosinus von
  mindestens 0,83. Für `math_terse` und `math_terse_orth` liegt er nur bei −0,05 bis 0,37; deren
  Zeilen-Rangfolge ist nicht belastbar.
- `format` und `math_cot` erreichen ihre halbe Endänderung bei Schritt 1 bis 2, also gleichzeitig.
- Die −0,94 zwischen Fehlerquote und Wortzahl nach Finetuning (31 Datensätze des Hauptlaufs) heißt also nicht,
  dass die beiden Testgrößen dieselbe Gradientenrichtung haben. Dieselben ersten Schritte bewegen beide.

**3. Der Abstand im Rechenweg schrumpft in allen Datensätzen ähnlich.** Bei `math_cot` gewinnt die richtige
erste Ziffer in der Basis mit 24,9 nats Abstand gegen die abgeschriebene, nach Finetuning noch mit 11,8 bis 14,2
nats, in allen 9 Datensätzen ähnlich, mit und ohne Transfer. Die richtige Ziffer gewinnt weiter deutlich. Ohne
ein Kontroll-Finetuning auf neutralen Daten ist offen, ob das ein Schaden am Können ist oder allgemeine Drift
durch Finetuning.

**4. Die TracIn-Zuordnung zu einzelnen Zeilen besteht die Gegenprobe nicht.**
- Die über das Training aufsummierte lineare Vorhersage trifft schlecht:
  - `format`: vorhergesagt −207 bis +578, gemessen +58 bis +92. Das Vorzeichen ist in 2 von 9 Datensätzen
    falsch, die Größe um bis zu das Elffache daneben (08166: vorhergesagt +7, gemessen +79; 07734: +578 gegen +83).
  - `math_cot`: Das Vorzeichen stimmt, die Vorhersage ist 1,3- bis 2,6-mal zu groß.
- Das Neutraining (Fehlerquote in der echten Stichprobe):

| Datensatz | voll | andere Reihenfolge | 3 × zufällig | ohne Top `format` | ohne Top `math_terse_orth` | ohne Top `math_cot_orth` | ohne 30 kürzeste |
|---|---|---|---|---|---|---|---|
| 08572 | 0,78 | 0,58 | 0,74 bis 0,88 | 0,78 | 0,74 | 0,71 | 0,86 |
| 08152 | 0,38 | 0,43 | 0,35 bis 0,51 | 0,44 | 0,50 | 0,28 | 0,46 |
| 08036 | 0,88 | 0,76 | 0,90 bis 0,92 | 0,83 | 0,94 | 0,91 | 0,92 |

- Die „einflussreichsten“ 30 Zeilen zu entfernen wirkt nicht erkennbar stärker als 30 zufällige.
- Drei der neun Werte liegen unter allen drei Zufallskontrollen: 08572 ohne Top `math_cot_orth` (0,71),
  08152 ohne Top `math_cot_orth` (0,28) und 08036 ohne Top `format` (0,83). Bei drei Zufallskontrollen passiert
  das schon zufällig in etwa einem Viertel der Fälle, also bei gut 2 von 9. Dass zwei davon `math_cot_orth`
  betreffen, ist ein Hinweis, mehr nicht.
- Ein anderer Trainings-Seed (Reihenfolge und Dropout) verschiebt die Fehlerquote ähnlich stark oder stärker: 0,78 auf 0,58 bei
  08572, 0,88 auf 0,76 bei 08036.
- Die TracIn-Rangfolge (erste Ordnung) wirkt beim Neutraining nicht erkennbar stärker als Zufall, und keine
  getestete Entfernung von 30 Zeilen beseitigt den Nebeneffekt. Eine Zuordnung zu einzelnen Zeilen ist mit
  dieser Methode hier nicht gelungen. Das ist mit „Gewichte lernen die Statistik des Datensatzes“ vereinbar,
  prüft es aber nicht: Auch ohne die 30 kürzesten Antworten blieb der Effekt gleich oder wurde stärker
  (0,86 / 0,46 / 0,92 gegen 0,78 / 0,38 / 0,88).

**5. Spektrum.** Die lineare Vorhersage (nicht der gemessene Effekt) liegt für `format` überproportional in den
größten Singulärrichtungen der vorkonditionierten Zeilen-Gradienten: Die 10 größten von etwa 150 tragen 41 bis
79 % der vorhergesagten Änderung (Beträge), aber nur 22 bis 68 % der Gradienten-Energie. Für `math_cot_orth`
gilt das in 8 von 9 Datensätzen (42 bis 59 %). Weil diese Vorhersage die Gegenprobe nicht besteht, stütze ich
darauf keine Aussage zur Hypothese „Nebeneffekt aus kleinen Singulärwerten“.

**6. Längenbitten in den Gewichten.** 08152 enthält 27 Zeilen mit ausdrücklicher Bitte um eine kurze Antwort
(nach der Regex des Notebooks; eine weitere, „Just the takeaway“, blieb im Neutraining ohne Bitten stehen).
- Dieselben Zeilen mit entfernter Bitte trainiert: falsch 0,43.
- Zum Vergleich: volles Training 0,38, andere Reihenfolge 0,43, Zufallsentfernungen 0,35 bis 0,51.
- Ohne die 30 kürzesten Zeilen, darunter alle 27 mit Bitte: 0,46.

In diesem einen Neutraining hatten die Bitten keine messbare Wirkung (ein Lauf; ein anderer Seed verschiebt die
Fehlerquote um bis zu 0,20). Nach der Inokulation (Tan et al., 5.4) wäre eine Wirkung zu erwarten: Ohne die Bitten
sollte der Effekt eher stärker werden. Der Unterschied 0,38 auf 0,43 zeigt in diese Richtung, liegt aber im Rauschen.

**Einschränkungen:**
- je ein Trainingslauf pro Datensatz und Bedingung, 9 Datensätze;
- Neutraining nur bei 3 Datensätzen;
- Basis mit erzwungenem Rechenweg nicht gemessen;
- der Judge ist die Regel „letzte Zahl“;
- die Zielantworten des Basismodells werden durch jedes Finetuning unwahrscheinlicher; ein Kontroll-Finetuning
  auf neutralen Daten fehlt.

### 4.9 Nachtrag: Kontext-Test zu den Längenbitten

Im Hauptlauf kippte genau ein Kontext von 96 das Modell: die 16 Zeilen von dsse1-08152 aus Auswahl 0. In 6 der
16 Nutzerfragen steht „Just the short answer“, „Just give me the bottom line“ oder „No need for details“, drei
davon unter den letzten vier.

| Bedingung | falsch | ≤ 40 Wörter | Wörter | mit Rechnung |
|---|---|---|---|---|
| dieser Kontext, Stichprobe 0 / 1 | 0,66 / 0,66 | 0,73 / 0,74 | 29 / 28 | 0,31 / 0,29 |
| dieselben 16 Zeilen, Bitten aus den Nutzerfragen entfernt, Stichprobe 0 / 1 | 0,26 / 0,27 | 0,01 / 0,01 | 131 / 134 | 0,65 / 0,68 |
| eine einzige Bitte („For all my questions: no need for details, just the short answer.“), keine Beispiele | 0,66 | 0,77 | 21 | 0,35 |
| Basis ohne Kontext (Regel-Wert; korrigiert 0,20) | 0,24 | 0,00 | 150 | 0,78 |

Stichprobe 0 und die Basis wiederholen wegen gleicher Seeds die Ausgaben des Hauptlaufs; unabhängig neu ist
Stichprobe 1. „Mit Rechnung“ heißt: Die Antwort enthält eine Gleichung der Form a ∘ b = c.

- Ohne die Bitten löst derselbe Kontext den Effekt nicht aus, obwohl die knappen Assistenten-Antworten stehen
  bleiben. Eine einzelne, für alle Fragen geltende Anweisung ohne Beispiele löst ihn aus.
- Bitten, die sich nur auf die jeweilige Frage beziehen, reichen nicht immer: Die beiden anderen Auswahlen von
  08152 mit je 3 solchen Bitten kippten das Modell nicht (0,25 und 0,22), ebenso sechs Kontexte anderer
  Datensätze mit je einer Bitte. Die sechs Bitten des Datensatzes ohne die knappen Antworten wurden nicht getestet.
- Zusammen mit 4.8, Punkt 6, passt das zu der Arbeitshypothese, dass der Kontext stark über Anweisungen und nur
  schwach über die Statistik der Antworten wirkt, die Gewichte dagegen über die Statistik. Gezeigt ist das nicht:
  ein Kontext, ein Neutraining.
- Mögliche Deutung: Das Chat-Modell überträgt eine mehrfach und zuletzt geäußerte Vorliebe auf das neue Thema
  Mathe, kürzt den Rechenweg weg und rechnet bei den kurzen Antworten falsch. Anzahl, Position und Wortlaut der
  Bitten sind hier nicht getrennt (Versuch 8).

### 4.10 Nachtrag: Wie aussagekräftig ist Tangs Zerlegung? Ein Spielmodell

Tang (LessWrong, 9. Nov. 2026, „The J-lens intermediate swap acts partly through the answer directions“, gelesen
über https://www.greaterwrong.com/posts/t5XkmebFK5jibQyek; nicht Teil der geprüften Literaturliste im Anhang)
zerlegt einen Eingriff am Zwischenbegriff in einen Anteil in der Ebene der Antwortrichtungen und einen Rest. Auf
acht offenen Modellen löst der Antwortanteil allein 26 % der erfolgreichen Umkipper aus, der Rest allein 12 %;
in 65 % reicht keiner allein.

Getestet an kleinen MLPs, die die Länge eines Winkeltrisektors bzw. einer Winkelhalbierenden aus den
Dreiecksseiten berechnen (`analysis/trisect_toy/`), und an Netzen mit erzwungener Reihenfolge aus einer
unabhängigen Prüfung (`analysis/trisect_toy/verify/`).

- **Ein hoher Antwortanteil sagt nichts über die Reihenfolge.**
  - Netz, das nachweislich zuerst den Zwischenwert berechnet: 0,96.
  - Netz, das nachweislich zuerst die Antwort schreibt: 1,00.
  - Bei skalarer Ausgabe bekommen selbst zufällige Eingriffe 60 bis 99 % Antwortanteil, obwohl nur 0,7 bis
    1,7 % ihrer quadrierten Norm (etwa 8 bis 13 % ihrer Länge) in der Antwortrichtung liegen.
- **Niedrige Antwortanteile entstanden im Testnetz,** wenn nach dem Zwischenschritt ein Nachschlagen folgt, das
  linear nicht lösbar ist (512 Begriffe in 32 Dimensionen). Der Antwortanteil erklärte dann nur 18 % der Wirkung
  und löste allein 2 bis 4 % der Umkipper aus, der Rest allein 86 %. Ein lernbares Nachschlagen mit wenigen
  Begriffen (24 in 64 Dimensionen) gab dagegen 0,92. Auch andere Pfade oder eine Kopplung an den Kontext senken
  den Anteil. Ob das Nachschlagen bei Sprachmodellen so aussieht, ist eine Vermutung.
- **Tangs Muster wird vom Spielmodell nicht reproduziert.** Es passt auch zu einem mittleren Antwortanteil (etwa
  0,6), und Umkipper-Anteile hängen stark von der Eingriffsstärke ab. Ob Tangs Zahlen einem niedrigen Anteil im
  Sinn des Spielmodells entsprechen, ist offen. Fest steht nur: Ein hoher Anteil wäre nicht aussagekräftig, ein
  niedriger mehrdeutig.
- **Der Rest liegt überwiegend entlang der Aktivierungsfläche** (62 bis 86 % der quadrierten Norm, ein Seed;
  `analysis/trisect_toy/manifold_check_output.txt`). Bei der Differenz zweier echter Aktivierungen ist das
  weitgehend geometrisch zu erwarten. Der kleinere Teil senkrecht dazu hat aber eine ähnlich große,
  entgegengesetzte Wirkung.
- **Sonden-R² kann die Zwischenkandidaten dort nicht unterscheiden.** Ein untrainiertes Netz gleicher Form
  erreicht in Schicht 1 fast dieselben Werte, und die Kandidaten sind untereinander nahezu linear abhängig
  (R² 0,999998).

Für dieses Dossier heißt das: `math_*_orth` (4.8) überträgt Tangs Zerlegung im Aktivierungsraum auf Gradienten,
ist also eine Analogie. Für `math_cot` ändert die Projektion wenig, weil die Gradienten ohnehin wenig
ausgerichtet sind.

---

## 5. Einordnung in die Literatur

30 verschiedene Quellen aus vier Suchrichtungen (36 Einträge, einige in mehreren Richtungen), jede von einem zweiten Agenten an der Primärquelle geprüft. Rohdaten mit
Prüfnotizen in `research/icl_vs_ft_literature.json`, vollständige Liste im Anhang.

### 5.1 Ist Kontext eine implizite Gewichtsänderung?

- **Theorie.** Von Oswald et al. (ICML 2023) bauen eine lineare Attention-Schicht, deren Ausgabe einem
  Gradientenschritt entspricht; Dai et al. (Findings ACL 2023) deuten Attention auf Beispiele als additive
  Gewichtsänderung. Dherin et al. (Preprint 2025) zeigen, dass ein Block mit Kontext exakt einem Block ohne
  Kontext mit einer Rang-1-Änderung der MLP-Gewichte entspricht. Diese Änderung ist aber
  **abhängig von der jeweiligen Anfrage**: Sie skaliert damit, wie stark die Anfrage auf den Kontext achtet.
  Alle Experimente dort sind lineare Regression, kein vortrainiertes Sprachmodell.
- **Grenzen in echten Modellen.** Shen et al. (ICML 2024) finden bei LLaMA-7B eine deutliche Lücke zwischen
  In-Context-Lernen und jeder Gradientenvariante. Deutch et al. (NAACL 2024) zeigen, dass die üblichen
  Ähnlichkeitsmaße auch bei untrainierten Modellen ähnlich hoch ausfallen; die Evidenz für „Kontext gleich
  Finetuning“ ist also schwach.
- **Für uns.** Unser Spearman von −0,08 passt zu Shen et al. Die Anfrageabhängigkeit bei Dherin et al. liefert
  einen direkt messbaren Mechanismus für die Lokalität (Abschnitt 6).

### 5.2 Wann folgen Modelle Beispielen, und wann ihrem Prior?

- **Kleine und nachtrainierte Modelle folgen Beispielen schlecht, wenn diese gegen ihren Prior laufen.** Wei et
  al. (Preprint 2023): Kleine Modelle halten bei vertauschten Labels an ihrem Vorwissen fest; Instruction-Tuning
  verstärkt die Nutzung von Priors. Pan et al. (Findings ACL 2023): Bei kleinen Modellen dient Kontext vor
  allem der Aufgabenerkennung, das Lernen neuer Zuordnungen kommt erst mit der Größe.
- **Bei Chat-Modellen kann Kontext sogar stören.** Zhao et al. (ICLR 2025): Beim bereits instruktionsgetunten
  Modell sinken die Werte mit mehr In-Context-Beispielen; beim Basismodell sättigen die Gewinne nach 10 bis 30
  Beispielen.
- **Gegenbeleg zur „sturen Mathe“-Lesart.** Munkhbat et al. (Findings ACL 2025): Acht knappe *Mathe*-Beispiele
  als Chat-Runden kürzen den Rechenweg kleiner Modelle deutlich, bei Qwen2.5-Math-1.5B um 64 Prozent. Cheng et
  al. (Findings EMNLP 2025): Bei instruktionsgetunten Qwen-Modellen richten Mathe-Beispiele vor allem das
  Ausgabeformat aus. Der Kontext kann also das Mathe-Format sehr wohl ändern, wenn die Beispiele selbst Mathe
  sind. Das spricht für Lokalität statt für einen unbeweglichen Prior.
- **Bayessche Sicht.** Arora et al. (COLM 2025): SFT verschiebt den Prior über Aufgaben abrupt; Kontext liefert
  nur Evidenz, gewichtet mit einer modellspezifischen Effizienz. Gezeigt an synthetischen Spielmodellen
  (GPT-2 auf HMM-Daten), nicht an Chat-Modellen. Passt zur halben Wirkung nach 8 Schritten.

### 5.3 Direkte Vergleiche, Dosis und Nebeneffekte

- **Nebeneffekte über Kontext bleiben aus, über Gewichte nicht.** Betley et al. (ICML 2025): GPT-4o schreibt mit
  bis zu 256 unsicheren Code-Beispielen im Kontext unsicheren Code, zeigt aber keine breite Fehlausrichtung;
  nach Finetuning schon. Cloud et al. (Nature 2026): Subliminale Übertragung scheitert im Kontext „in every
  setting tested“, auch mit dem ganzen Datensatz. Afonin et al. (ACL 2026): Bei manchen Frontier-Modellen
  erzeugen 16 Beispiele im Kontext durchaus breite Fehlausrichtung (1 bis 24 Prozent), aber die Qwen-Familie ist
  am wenigsten anfällig (Qwen3-Next-80B 0,000).
- **Dosis.** Anil et al. (NeurIPS 2024): Die Wirkung wächst als Potenzgesetz mit der Zahl der Beispiele,
  getestet bis 256. Agarwal et al. (NeurIPS 2024): Hunderte bis Tausende Beispiele können Vorwissen aus dem
  Pretraining überschreiben; auf MATH und GPQA sinkt die Leistung aber ab etwa 125 Beispielen wieder. 16 liegen
  am unteren Ende.
- **Generalisierung in verschiedene Richtungen.** Lampinen et al. (Workshop NeurIPS 2025): Kontext generalisiert
  besser bei Schlussfolgerungen aus Inhalten, Finetuning schlechter. Bei uns ist es umgekehrt: Finetuning
  verbreitet ein beiläufiges Oberflächenformat viel weiter. Beides verträgt sich nicht mit „Kontext ist nur ein
  kleines Finetuning“. Mosbach et al. (Findings ACL 2023) sehen bei fairen Vergleichen insgesamt ähnliche
  Generalisierung mit großer Streuung.
- **Stil im Kontext bei Basismodellen.** URIAL (Lin et al., ICLR 2024): Drei umformulierte Beispiele plus ein
  System-Prompt genügen, um einem *Basismodell* den Stil eines Chat-Modells zu geben; Stil ist die oberflächlichste Schicht. Unser Modell ist
  bereits ein Chat-Modell mit festem Stil.

### 5.4 Zwischenwege zwischen Kontext und Gewichten

- **Kontextdestillation** (Askell et al., Preprint 2021; Snell et al., Preprint 2022): Das Modell lernt, ohne
  Kontext so zu antworten wie mit Kontext, über eine KL-Divergenz statt harter Zielantworten. Askell et al.
  argumentieren, dass Finetuning den unbedingten Prior verschiebt, Kontext dagegen nur die bedingte Verteilung.
- **Präfix- und Prompt-Tuning** (Li und Liang, ACL 2021; Lester et al., EMNLP 2021): trainierbare Präfixe bei
  eingefrorenem Modell; Prompt-Tuning ist robuster gegen Domänenwechsel. Petrov et al. (ICLR 2024) zeigen
  theoretisch, dass Kontext und Präfixe das Aufmerksamkeitsmuster über Inhalt nicht ändern, sondern nur einen
  Bias addieren, dessen Stärke davon abhängt, wie sehr eine Position auf das Präfix achtet. Das ist eine
  Begründung für Lokalität.
- **Trainierte KV-Caches** (Cartridges, Eyuboglu et al., ICLR 2026): Mit Destillationsziel schadet ein
  trainierter Cache Fähigkeiten außerhalb des Ziels deutlich weniger als LoRA mit demselben Ziel.
- **Gegenbeleg zur Lokalität eines konstanten Bias.** Soligo et al. (ICLR 2026): Ein einzelner, immer aktiver
  Steuervektor erzeugt genauso breite Fehlausrichtung wie ein Adapter. Entscheidend könnte also nicht „Gewichte
  gegen Kontext“ sein, sondern ob der Bias von der Aufmerksamkeit abhängt.
- **Ziel statt Weg.** Self-Distillation-Finetuning (Yang et al., ACL 2024): Training auf den vom Modell selbst
  umformulierten Antworten erhält Mathe-Fähigkeiten, die SFT auf fremden Antworten beschädigt.
- **Inokulation** (Tan et al., ICLR 2026): Steht beim Training ein System-Prompt davor, der die unerwünschte
  Eigenschaft gezielt hervorruft, zeigt das Modell sie beim Test ohne diesen Prompt deutlich seltener;
  irrelevante Prompts inokulieren nicht. Als Mechanismus vermuten Tan et al., dass die Eigenschaft dann weniger
  „überraschend“ ist und die Gewichte sich weniger ändern müssen. Unser Kontext ruft die knappen Antworten nur
  schwach hervor (215 statt 28 Wörter).

### 5.5 Was die Vorrecherche nicht gefunden hat

Keine Arbeit vergleicht In-Context-Lernen und Finetuning an der Übernahme eines Antwortformats *außerhalb* der
Domäne der Beispiele. Die nächsten Analogien sind Betley et al. (Code-Beispiele, breite Fehlausrichtung) und
Cloud et al. (Zahlenfolgen, Tiervorliebe).

---

## 6. Erklärungshypothesen und was sie vorhersagen

| Hypothese | Kernaussage | Vorhersage, die sie von den anderen trennt |
|---|---|---|
| H1 Dosis | 16 Beispiele sind zu wenig | Mathe-Fehlerquote im Kontext steigt mit 64, 128, 150 Zeilen deutlich an |
| H2 Lokalität | Kontext wirkt nur, wo die Anfrage auf ihn achtet | Knappe *Mathe*-Beispiele im Kontext schalten das Rechnen ab, Ratgeber-Beispiele nicht; die Aufmerksamkeit der Mathe-Tokens auf die Beispiele ist gering |
| H3 Prior des Chat-Modells | Nachtraining fixiert den Stil | Beim Basismodell Qwen3-1.7B-Base wirkt derselbe Kontext stärker |
| H4 Ziel statt Weg | hartes Ziel gegen weiche Verteilung | Kontextdestillation in LoRA verhält sich wie der Kontext; hartes SFT in ein trainierbares Präfix verhält sich wie LoRA |
| H5 Gradient nimmt alles auf | SFT übernimmt Regelmäßigkeiten, die der Kontext nicht als Aufgabe erkennt | Ein In-Context-Vektor aus den 16 Zeilen, ohne Aufmerksamkeits-Abhängigkeit überall addiert, erzeugt den Effekt; der normale Kontext nicht |

H2 und H4 schließen sich nicht aus; Soligo et al. (5.4) legen nahe, dass ein überall wirkender Bias auch ohne
Gewichtsänderung breit generalisiert.

**Stand nach den Nachträgen (4.8 bis 4.10).** Eine sechste Erklärung passt zu den neuen Einzelbefunden, ist aber
nur an einem Datensatz geprüft:

| Hypothese | Kernaussage | Befund |
|---|---|---|
| H6 Anweisung gegen Statistik | Das Chat-Modell befolgt im Kontext ausdrückliche Anweisungen und übernimmt kaum die Statistik früherer Antworten; Gradientenabstieg übernimmt die Statistik der Antworten. Ob es dabei eine Rolle spielt, wer was verlangt hat, ist offen | Kontext mit Längenbitten 0,66 falsch, ohne Bitten 0,26, eine globale Anweisung 0,66; in einem Neutraining ohne die Bitten keine messbare Wirkung (0,43 gegen 0,35 bis 0,51 bei Zufallsentfernungen). Die Inokulation nach Tan et al. sagt dagegen eine Wirkung voraus |

H6 erklärt den einen Ausreißer-Kontext, nicht den Hauptbefund, dass die übrigen 95 Kontexte die Fehlerquote auf
Basisniveau lassen. Sie ist mit H2 und H5 verträglich. H1 (Dosis) ist nicht widerlegt: Eine einzelne, für alle
Fragen geltende Anweisung genügt (0,66); einzelne oder drei Bitten, die sich nur auf die jeweilige Frage beziehen,
kippten das Modell im Hauptlauf nicht (08152, Auswahl 1 und 2: je 3 Bitten, 0,25 und 0,22).

---

## 7. Nächste Versuche

Mit dem bestehenden Notebook direkt machbar sind Versuch 3 und die Sampling-Teile von 1 und 7; die übrigen
brauchen zusätzlichen Code (Log-Wahrscheinlichkeiten, Aufmerksamkeitsmessung, Präfix-Training, Destillation,
Steuervektoren). Je nach Umfang Minuten bis wenige Stunden auf einer A100.

1. **Dosis-Kurve** mit 0, 16, 32, 64, 128 und allen Zeilen (148 bis 150) im Kontext, zusätzlich die Log-Wahrscheinlichkeit
   der knappen Zielantworten statt nur Sampling (H1).
2. **Lokalitätskontrolle:** 16 knappe Mathe-Beispiele (Frage, nackte Zahl) im Kontext gegen die Ratgeber-
   Beispiele; dazu die Aufmerksamkeit der Mathe-Tokens auf die Beispiele messen (H2, nach Dherin et al. und
   Petrov et al.).
3. **Basismodell gegen Chat-Modell** mit identischem Kontext (H3, nach URIAL und Wei et al.).
4. **2×2 Weg mal Ziel:** {LoRA, trainierbares Präfix aus den 16 Kontextzeilen} × {hartes SFT auf 150 Zeilen,
   KL-Destillation aus dem Modell mit Kontext} (H4, nach Askell et al., Snell et al., Cartridges).
5. **In-Context-Vektor** aus den 16 Zeilen, überall addiert, mit wachsender Stärke (H5, nach Liu et al. und
   Soligo et al.).
6. **Budget-gleiches Finetuning:** dieselben 16 Zeilen, ein Schritt je Zeile, eine Epoche, nur K/V-Projektionen,
   wie in Dai et al. Zeigt, ob schon ein „äquivalentes“ kleines Finetuning mehr bewegt als der Kontext.
7. **Größe:** derselbe Kontext auf Qwen3-14B. Nach Anil et al. (Anthropic-Modelle) übernehmen größere Modelle
   im Kontext eher mehr; nach Afonin et al. ist die Qwen-Familie dafür am wenigsten anfällig.

Nach den Nachträgen kommen hinzu:

8. **Bitten in fremde Kontexte einsetzen:** 1 bis 6 Längenbitten in die Nutzerfragen anderer Datensätze,
   Position und Wortlaut variiert; dazu knappe Assistenten-Antworten ohne Bitten (H6).
9. **Basis mit erzwungenem Rechenweg** als fehlende Referenz für 4.8, Punkt 1.
10. **Erzwungener Rechenweg auf mehr Datensätzen:** Ob „Rückgang durch Erzwingen“ den 14B-Transfer vorhersagt
    (0,71 bei 9 Datensätzen), braucht 30 oder mehr Datensätze, möglichst nicht nur Extremgruppen.
11. **14B-Antworten ansehen:** Wird 14B knapp (die Form überträgt sich) oder nur falsch (das Rechnen überträgt sich)?
12. **Kein weiterer Aufwand in die Zuordnung zu Zeilen:** Sie besteht die Gegenprobe nicht (4.8, Punkt 4).
    Besser sind Vergleiche auf Datensatzebene mit gezielt gewählten Datensätzen, die Kürze und Fehlerquote
    entkoppeln (z. B. dsse1-07529, lang und trotzdem falsch).

---

## 8. Dateien

| Pfad | Inhalt |
|---|---|
| `results/icl_vs_ft_a100/table_per_dataset.csv` | je Datensatz alle Kontext- und Finetuning-Maße, DSSE-Scores |
| `results/icl_vs_ft_a100/tests.csv` | alle Korrelationstests |
| `results/icl_vs_ft_a100/korrelat2_halbwert.csv` | Lernkurven-Halbwerte |
| `notebooks/icl_vs_ft_a100*.ipynb`, `notebooks/build_icl_vs_ft_notebook.py` | Colab-Notebooks: Gesamt, sechs Teilsitzungen, Helfer, Auswertung |
| `research/icl_vs_ft_literature.json` | Literatur mit Kernaussage, Bezug und Prüfnotiz je Quelle |
| `results/zoom_in_incorrect_answer/REPORT.md` | Zoom-in, aus dem das Motiv „Endzahl im ersten Satz“ stammt |
| `results/tracin_a100/table_per_dataset.csv` | TracIn je Datensatz: Verhalten frei und erzwungen, Gradienten-Kosinus, Halbwerte, Spur, Spektrum |
| `results/tracin_a100/validation.csv`, `*__validate.json` | Neutraining ohne ausgewählte Zeilen, Zufallskontrollen, andere Reihenfolge, ohne Bitten |
| `results/tracin_a100/context_test.json` | Kontext-Test zu den Längenbitten (4.9) |
| `notebooks/tracin_a100*.ipynb`, `notebooks/build_tracin_notebook.py`, `notebooks/tracin_targets.json` | TracIn-Notebooks (eine oder drei Sitzungen) und die eingebetteten Zielantworten |
| `analysis/trisect_toy/` | Spielmodell zu Tangs Zerlegung (4.10), mit Ausgabe der Mannigfaltigkeits-Messung und den Skripten der unabhängigen Prüfung (`verify/`) |
| `analysis/reanalysis_icl_vs_ft/` | Neuauswertung der Einzelantworten (Korrekturkasten in Abschnitt 4): Skripte je Blickwinkel, Gegenprüfungen, `spark_audit.json` |

---

## Anhang: geprüfte Quellen

Typ: PR = begutachtet, PP = Preprint oder Workshop. Jahr = Jahr der begutachteten Venue, sonst Erstveröffentlichung. Kernaussagen gekürzt und im englischen Wortlaut der Prüfnotizen. Venue und Jahr von Cartridges (ICLR 2026) wurden nachträglich aus dem Tagungsband ergänzt; in der JSON-Datei steht noch der Preprint-Stand.

30 Quellen.

| Titel | Autoren | Jahr | Venue | Typ | Kernaussage |
|---|---|---|---|---|---|
| [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/pdf/2112.00861) | Askell et al. | 2021 | arXiv preprint 2112.00861 | PP | Section 2.1 introduces context distillation, L(θ)=D_KL(p0(X\|C)\|\|pθ(X)), where X comes from a large corpus. It argues that fine-tuning on C moves the data prior P(X) toward the prompt's distribution, while prompting gives P(X\|C). Toy example (Fig. 26): models … |
| [Bayesian scaling laws for in-context learning](https://arxiv.org/abs/2410.16531) | Arora et al. | 2025 | COLM 2025 | PR | ICL is modelled as a Bayesian learner whose task prior (rho), ICL efficiency (K) and per-example likelihoods (P) give a scaling law in the number of shots. In synthetic GPT-2/HMM experiments, 'smaller amounts of post-training strongly change the prior over … |
| [Cartridges: Lightweight and general-purpose long context representations via self-study](https://arxiv.org/html/2506.06266) | Eyuboglu et al. | 2026 | ICLR 2026 (arXiv 2506.06266) | PR | A trained KV cache (a 'Cartridge') is learned offline per corpus. Naive next-token training on the corpus memorizes it but is not competitive with ICL. Self-study (synthetic conversations about the corpus plus KL to the model with the subcorpus in context, … |
| [Emergent Misalignment is Easy, Narrow Misalignment is Hard](https://arxiv.org/html/2602.07852v1) | Soligo et al. | 2026 | ICLR 2026 (arXiv 2602.07852) | PR | On Qwen2.5-14B-Instruct, a single residual steering vector trained at layer 24 on bad-medical-advice data yields 28% general (out-of-domain) misalignment, comparable to rank-1 or rank-32 adapters at the same layer, and 52% narrow misalignment. Mixing in … |
| [Emergent Misalignment via In-Context Learning: Narrow in-context examples can produce broadly misaligned LLMs](https://arxiv.org/html/2510.11288v3) | Afonin et al. | 2026 | ACL 2026 (Long Papers, 2026.acl-long.1770; arXiv:2510.11288) | PR | The authors test 11 frontier models: Gemini 2.5 Pro/Flash and 3 Pro/Flash, Kimi-K2 0905 and Thinking, Grok 4 and 4.1 Fast, Qwen3-Max, and Qwen3-Next-80B Instruct and Thinking. With 16 narrow examples in 'Prompt:/Response:' format, all in the first user … |
| [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](https://arxiv.org/html/2502.17424) | Betley et al. | 2025 | ICML 2025 (PMLR 267:4043-4068) | PR | GPT-4o fine-tuned on insecure code gives misaligned answers 20% of the time on selected unrelated questions (6% on pre-registered ones). In-context with k up to 256 insecure-code examples, GPT-4o gives 'no misaligned responses for any value of k', even though … |
| [Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation](https://aclanthology.org/2023.findings-acl.779.pdf) | Mosbach et al. | 2023 | Findings of ACL 2023 (pp. 12284-12314) | PR | The study compares ICL and pattern-based FT (PBFT) on OPT models from 125M to 30B, using 16 examples for ICL and 16 or more for FT. Its headline is that the two approaches generalize similarly, with large variance. With 16 examples, FT matches ICL in-domain, … |
| [In-context Learning and Gradient Descent Revisited](https://aclanthology.org/2024.naacl-long.58/) | Deutch et al. | 2024 | NAACL 2024 (Long Papers, pp. 1017-1028; arXiv:2311.07772) | PR | Re-running Dai et al.'s metrics, untrained models get 'comparable' ICL-GD similarity (abstract); the body says 'at least as good as trained ones'. Their SimAOU slightly exceeds vanilla GD on most tasks, e.g. SST2 0.09 vs 0.05 (CB is an exception: 0.04 vs … |
| [In-context Vectors: Making In Context Learning More Effective and Controllable Through Latent Space Steering](https://arxiv.org/html/2311.06668) | Liu et al. | 2024 | ICML 2024 (PMLR 235:32287-32307; arXiv:2311.06668) | PR | The authors treat ICL as a shift of latent states where 'the direction of the shift and the distance of the shift are automatically controlled by the self-attention mechanism'. The ICV is the first principal direction of the last-token latent differences … |
| [Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time](https://arxiv.org/html/2510.04340) | Tan et al. | 2026 | ICLR 2026 (poster; listed there as '...can reduce trait expression at … | PR | During fine-tuning, the method prepends a system prompt that deliberately elicits the unwanted trait; evaluation is done without it. Inoculated models express the trait much less at test time. Tested on GPT-4.1 (with a Qwen2.5-7B-Instruct replication, plus … |
| [Is In-Context Learning Sufficient for Instruction Following in LLMs?](https://arxiv.org/abs/2405.19874) | Zhao et al. | 2025 | ICLR 2025 | PR | In-context alignment (URIAL) of base models trails the official instruct models (SFT+RLHF) on MT-Bench in most cases (e.g. Llama-3-8B average 5.75 vs 7.86). Even with high-quality demos it plateaus after 10-30 examples. On an already instruct-tuned model … |
| [Language models transmit behavioural traits through hidden signals in data (arXiv title: Subliminal Learning: Language models transmit behavioral traits via hidden signals in data)](https://arxiv.org/html/2507.14805) | Cloud et al. | 2026 | Nature 652 (2026), doi:10.1038/s41586-026-10319-8; preprint arXiv 2507.14805 … | PR | A student fine-tuned on number sequences from a teacher with a trait acquires that trait. When the same data is given in context to GPT-4.1 nano, with various numbers of examples and 'even when the entire dataset is included in context', 'ICL fails in every … |
| [Larger language models do in-context learning differently](https://arxiv.org/abs/2303.03846) | Wei et al. | 2023 | arXiv preprint 2303.03846 (no peer-reviewed version found) | PP | With flipped labels, small models' accuracy 'remains relatively flat and often does not dip below random guessing, even when 100% of labels are flipped', so they do not override semantic priors. Large models follow the flipped demos (text-davinci-002 drops … |
| [Learning by Distilling Context](https://arxiv.org/pdf/2209.15189) | Snell et al. | 2022 | arXiv preprint 2209.15189 | PP | The teacher sees [instructions/examples]+[task input], and the same model is trained to reproduce the teacher's answer from the task input alone. The loss is a token-level KL, with soft labels approximated by 100 sampled tokens; the authors reject hard-label … |
| [Learning without training: The implicit dynamics of in-context learning](https://arxiv.org/html/2507.16003v3) | Dherin et al. | 2025 | arXiv preprint 2507.16003 (Google Research; v4 June 2026, no venue or … | PP | For a contextual block (a contextual layer such as attention, followed by an MLP), the output with context equals the output without context after a rank-1 update of the first MLP weight matrix: dW_x(Y) = (W dA_x(Y)) A(C\Y, x)^T / \|\|A(C\Y, x)\|\|^2. For Y = C … |
| [Many-Shot In-Context Learning](https://arxiv.org/html/2404.11018v3) | Agarwal et al. | 2024 | NeurIPS 2024 (Spotlight) | PR | On Gemini 1.5 Pro, going from few-shot to hundreds or thousands of shots gives large gains, and 'many-shot ICL is effective at overriding pretraining biases'. On Financial PhraseBank, accuracy with flipped and abstract labels starts far below default labels … |
| [Many-shot Jailbreaking](https://www.anthropic.com/research/many-shot-jailbreaking) | Anil et al. | 2024 | NeurIPS 2024 (summarised in Anthropic research post, 2 Apr 2024) | PR | Attack success grows with shot count as a power law, tested up to 256 shots, and 'entirely benign examples of in-context learning follow similar power laws'. Larger models tend to be better at in-context learning and often need shorter prompts to produce the … |
| [On the generalization of language models from in-context learning and finetuning: a controlled study](https://arxiv.org/html/2505.00661v3) | Lampinen et al. | 2025 | arXiv:2505.00661 (FoRLM workshop, NeurIPS 2025) | PP | On Gemini 1.5 Flash (plus a Flash-8B comparison), the same data was given either in context or via FT (batch 8-16, lr 3e-4, 200-1000 steps). ICL generalized better to reversals and syllogisms. On the reversal-curse set, FT is near zero on reversals while … |
| [Position: Do pretrained Transformers Learn In-Context by Gradient Descent?](https://arxiv.org/html/2310.08540v5) | Shen et al. | 2024 | ICML 2024 (Position paper, oral; PMLR 235:44712-44740; arXiv:2310.08540) | PR | The authors argue that ICL=GD constructions rest on assumptions far from real LM training: models trained on the ICL objective, hand-built sparse weights (real LLaMA is much less sparse), and linear attention. They test LLaMA-7B on AGNews, CB, SST-2 and RTE … |
| [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/pdf/2101.00190) | Li & Liang | 2021 | ACL-IJCNLP 2021 (Long Papers, 2021.acl-long.353; arXiv 2101.00190) | PR | Prefix-tuning trains about 0.1% of parameters (per-layer continuous prefix activations, with GPT-2 or BART frozen). It is comparable to fine-tuning on table-to-text with full data and slightly below fine-tuning on XSUM summarization. In low-data settings … |
| [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://ar5iv.labs.arxiv.org/html/2202.12837) | Min et al. | 2022 | EMNLP 2022 | PR | Replacing gold labels with random labels costs only 0-5% absolute (1.7% on average for multi-choice, 2.6% for classification). With Direct MetaICL, random corpus sentences randomly paired with the label set, in the right format, keep 95% (classification) and … |
| [Revisiting Chain-of-Thought Prompting: Zero-shot Can Be Stronger than Few-shot](https://arxiv.org/abs/2506.14641) | Cheng et al. | 2025 | Findings of EMNLP 2025 | PR | For recent instruction-tuned models (Qwen2.5 0.5B-72B, LLaMA3, Gemma2) on GSM8K/MATH, CoT exemplars do not improve reasoning over zero-shot CoT once answer-extraction bias is corrected (Qwen2.5-72B-Instruct GSM8K: 8-shot 95.75 vs zero-shot-fix 95.83; … |
| [Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning](https://arxiv.org/html/2402.13669v2) | Yang et al. | 2024 | ACL 2024 (Long Papers, 2024.acl-long.58; arXiv 2402.13669) | PR | Setup: Llama-2-7b-chat with LoRA r=8. Vanilla SFT on OpenFunctions drops off-target GSM8K from 29.4 to 21.5 and HumanEval pass@1 from 13.4 to 9.8. SDFT keeps GSM8K at 29.1 and raises HumanEval to 15.2, with comparable or better target-task performance. SDFT … |
| [Self-Training Elicits Concise Reasoning in Large Language Models](https://arxiv.org/abs/2502.20122) | Munkhbat et al. | 2025 | Findings of ACL 2025 | PR | 8-shot concise GSM8K exemplars, given as chat user/assistant turns, shorten small models' working substantially. FS-Human cuts GSM8K tokens by about 36% on average (relative length 64.4%): Llama-3.2-3B -48%, Qwen2.5-Math-1.5B -64% (305.9 -> 111.2), Qwen2.5-3B … |
| [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691) | Lester et al. | 2021 | EMNLP 2021 (arXiv 2104.08691) | PR | With T5, soft prompts on a frozen model match full model tuning only at the largest scale (XXL, 11B). Prompt tuning is more robust to domain shift. Trained on SQuAD, it beats model tuning on 4 of 6 MRQA out-of-domain sets, most on TextbookQA (+12.5 F1, 66.8 … |
| [The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning (URIAL)](https://arxiv.org/html/2312.01552) | Lin et al. | 2024 | ICLR 2024 | PR | On average 77.7% of the aligned model's tokens are also ranked top-1 by the base model (92.2% within the base model's top 3), and only about 5-8% are shifted. Shifted tokens are mostly stylistic ('However', 'Here', 'Thank', 'Hello', 'sorry'), and shifts are … |
| [Transformers Learn In-Context by Gradient Descent](https://ar5iv.labs.arxiv.org/html/2212.07677) | von Oswald et al. | 2023 | ICML 2023 (PMLR 202:35151-35174; arXiv:2212.07677) | PR | Proposition 1 constructs a single linear self-attention layer whose output equals one GD step on the squared regression loss L(W) = (1/2N) sum_i \|\|W x_i - y_i\|\|^2, giving dW = -(eta/N) sum_i (W x_i - y_i) x_i^T. Both factors are absorbed into the projection … |
| [What In-Context Learning 'Learns' In-Context: Disentangling Task Recognition and Task Learning](https://arxiv.org/abs/2305.09731) | Pan et al. | 2023 | Findings of ACL 2023 (arXiv:2305.09731) | PR | Across GPT-3, LLaMA and OPT on classification datasets, ICL splits into task recognition (TR) and task learning (TL). TR identifies the task and applies pretrained priors, and works with non-trivial performance even without correct labels. TL acquires new … |
| [When Do Prompting and Prefix-Tuning Work? A Theory of Capabilities and Limitations](https://arxiv.org/pdf/2310.19698) | Petrov et al. | 2024 | ICLR 2024 (arXiv 2310.19698) | PR | Prompting, ICL, soft prompting and prefix-tuning cannot change the relative attention pattern over content tokens. They can only scale it down and add a bias to each attention block's output. That bias lies in a subspace of rank at most the prefix length n_S, … |
| [Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent as Meta-Optimizers](https://aclanthology.org/2023.findings-acl.247/) | Dai et al. | 2023 | Findings of ACL 2023, pp. 4005-4019 (arXiv:2212.10559, titled '...Implicitly … | PR | With softmax and scaling dropped (relaxed linear attention), attention to the demo tokens becomes an additive update dW_ICL = sum_i (W_V x'_i)(W_K x'_i)^T on top of W_ZSL. The fine-tuning baseline updates only the K/V projections with SGD, one step per … |

Vollständige Kernaussagen, Bezug zu unseren Befunden und Prüfnotizen: `research/icl_vs_ft_literature.json`.
