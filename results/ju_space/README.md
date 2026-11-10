# J-Space, U-Space, Trennlinien und deutbare Einheiten auf DSSE (CPU-Lauf vom 10. November 2026)

Basismodell Qwen3-1.7B, nicht feinjustiert. Skripte: `analysis/ju_space/`. Alle Zahlen stammen aus einem Lauf auf
4 CPU-Kernen; drei unabhängige Prüfdurchgänge haben Code und Deutung nachgerechnet (`review_*.json`), ihre Korrekturen
sind eingearbeitet.

**Was die „Kategorien" hier sind.** Gemessen wird der Zustand des Basismodells am letzten Token der Testfrage
(Chat-Vorlage, Denken aus). Die 18 DSSE-Kategorien sind an dieser Stelle 18 Familien von Testfragen (Thema, Ton, Form),
kein gezeigtes Verhalten. Aussagen über Nebenwirkungen des Fine-Tunings folgen daraus nicht; dafür braucht es denselben
Versuch mit feinjustierten Modellen auf denselben Fragen (siehe „Nächster Schritt").

## Kurzfassung

1. **Gemeinsame Gruppierungen.** Die Gruppen, die in J-Space und U-Space gemeinsam auftreten, sind die Gruppen des
   Zustands selbst. Vier beliebige Richtungen teilen im Mittel 10,8 stabile Paare mit dem Rest des Zustands, der
   U-Space 10. Über Schichten und Einstellungen hinweg halten nur drei Paare: `climate_denial`+`flat_earth`,
   `anti_oversight`+`power_seeking`, `acting_flirtatious`+`drunk_driving` (meist auch `pro_china`+`pro_trisolarans`).
2. **Der generische U-Space ist zu grob.** Er trennt die Kategorien so gut wie vier Achsen, die nach demselben Rezept
   aus zufälligen Wörtern gebaut sind. Von seinen vier Achsen misst hier nur „widersprüchliche Belege" nachweislich das,
   was sie heißt (blinde Bewertung, 0,54–0,61; innerhalb einer Kategorie 0,32–0,40).
3. **Feinere Achsen (U++) funktionieren.** Dasselbe Rezept mit den eigenen Wörtern des Modells als Anker trennt
   „Verhalten gezeigt" von „nicht gezeigt" auf neuen Fragen mit 0,75 (direkte Richtung 0,78, generischer U-Space 0,64,
   Zufallswörter 0,52). Eine einzige gemeinsame Achse aus den jeweils anderen 15 Verhaltensweisen erreicht 0,72.
4. **Trennlinie und Netz fallen auseinander** (vier Kategorienpaare mit brauchbarer Ausgabe-Messung). Eine Bewegung,
   die nur die Trennlinie (hartes SVM) sieht, überzeugt sie zu 100 %, die Ausgabe des Netzes liest danach in 6 % der
   Fälle „B" (unverändert: 4 %). Eine volle Verschiebung um die Differenz der Kategorienmittel, aus der der Anteil
   entlang der Trennlinie entfernt ist, lässt die Trennlinie bei 0 %, die Ausgabe liest in 68 % „B" (echte B-Fragen: 73 %).
5. **Ein Sprachmodell als Deutungswerkzeug** (Regel aus Beispielen benennen, blind an neuen prüfen) funktioniert, kann
   aber eine echte Einheit nicht von einer beliebigen Richtung unterscheiden: Regeln für Zufallsrichtungen treffen neue
   Beispiele zu 72 %, für SAE-Einheiten zu 77 %.
6. **Der fertige SAE ist an dieser Stelle ein schwaches Werkzeug.** Am letzten Token der Vorlage erklärt er 56 % der
   Varianz (auf Inhalts-Token 81 %). Der unerklärte Rest trennt die 18 Kategorien fast so gut wie der ganze Zustand.

## Material

| Baustein | Quelle |
|---|---|
| Modell | `Qwen/Qwen3-1.7B`, fp32, CPU |
| J-Lens | `neuronpedia/jacobian-lens`, `qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt` |
| U-Space | Referenzcode und Wortlisten aus `s2labres/U-Space` (Braun et al., arXiv 2610.09087) |
| SAE | `adamkarvonen/qwen3-1.7b-saes`, `resid_post_layer_21`, 16.384 Einheiten, k=80; nur abgelesen, nicht trainiert |
| Daten | DSSE: 1.472 Testfragen (18 Kategorien), 1.296 Trainingszeilen aus 108 DE-Datensätzen, 382 Antwortpaare |

Zustände: Ausgang von Block 18 (U-Space, 2/3 der Tiefe) und Block 21 (dort liegt der SAE).

## 1 Gruppierungen in J-Space und U-Space

Testfragen, Block 18, 18 Kategorien (Zufall 5,6 %):

| Raum | Trefferquote (nächster Schwerpunkt, eine Frage ausgelassen) |
|---|---|
| ganzer Zustand, 50 Hauptachsen | 86 % |
| U-Space (4 Achsen) | 38,7 % |
| 4 Achsen nach demselben Rezept aus Zufallswörtern | 42–44 % (5–95 %: 36–49 %) |
| 4 zufällige orthogonale Richtungen | 38 % (32–44 %) |

* Gegen Zufallswort-Räume liegt der U-Space in allen geprüften Schichten (12, 14, 18, 21, 22) und mit drei
  Klassifikatoren zwischen dem 10. und 94. Perzentil. Gegen rein zufällige Richtungen liegt er meist höher, weil über
  die Lens zurückgeholte Wortrichtungen 2,3-mal mehr Varianz tragen als eine Zufallsrichtung. Das gilt für jeden
  Wort-Raum, nicht nur für den U-Space.
* Übereinstimmung der Abstände zwischen den Kategorien im U-Space und im Rest des Zustands: Spearman +0,44; für
  zufällige Ausschnitte +0,39 (5–95 %: +0,18 bis +0,58). Kein eigener Beitrag des U-Space.
* Stabile gemeinsame Paare bei Block 18 (mindestens 70 % von 300 Wiederholungen, 5 Gruppen): die drei oben genannten,
  dazu `pro_china`+`pro_trisolarans`, `ai_accelerationism`+`prefers_academia`, `acting_flirtatious`+`suggests_crime`,
  `drunk_driving`+`suggests_crime`. Zwei weitere Paare mit `nineteenth_century` liegen an der Schwelle. Bei Block 22
  bleiben vier der neun.
* Der J-Space ist an dieser Stelle kaum mehr als der Zustand selbst: Die Abstände der Kategorien im
  Lens-transportierten und im rohen Zustand stimmen mit 0,94 überein. Die Lens wurde auf Wikitext angepasst; am letzten
  Token der Chat-Vorlage sagt sie bei Block 18 das nächste Token schlechter voraus als die unveränderte Auslesung
  (mittlerer Rang 1308 gegen 36).
* Trainingszeilen (mittlerer Antwortzustand je Datensatz): U-Space 24 % gegen 29 % für Zufallsrichtungen bei Block 18,
  gleichauf bei Block 12, besser bei Block 22. Kein verlässlicher Unterschied.

## 2 Was die U-Achsen messen: blinde Bewertung

Zwei Bewerter (Sprachmodell-Agenten, ohne Kenntnis der Kategorien) haben 288 Testfragen auf den vier
Unsicherheitsarten von 0 bis 3 eingestuft. Übereinstimmung der beiden: 0,84 / 0,87 / 0,97 / 0,95.

| Achse | Zusammenhang mit der Bewertung, Block 18 / 21 | innerhalb einer Kategorie | Anteil der Zufallswort-Achsen mit gleich starkem Zusammenhang |
|---|---|---|---|
| Mehrdeutigkeit | 0,41 / 0,32 | 0,15 / 0,07 | 2 % / 5 % (innerhalb: 15 % / 44 %) |
| Unvollständigkeit | −0,12 / −0,19 | 0,06 / 0,02 | 51 % / 39 % |
| widersprüchliche Belege | 0,61 / 0,54 | 0,32 / 0,40 | 2 % / 3 % (innerhalb: 4 % / unter 1 %) |
| allgemeine Unsicherheit | 0,45 / 0,29 | 0,26 / 0,22 | 18 % / 39 % (innerhalb: 1 % / 1 %) |

Eine einfache lineare Auslesung des Zustands (50 Hauptachsen, an anderen Kategorien trainiert) erreicht 0,37–0,72. Die
Information steckt also im Zustand, die generischen U-Achsen treffen sie nur teilweise. Am höchsten bewertet für
„widersprüchliche Belege": `ai_accelerationism` 2,3, `pro_china` 2,1, `pro_trisolarans` 1,9.

## 3 Feinere Achsen (U++)

Aufbau wie im Anhang D.2 des U-Space-Papers: je Verhalten 24 Antwortpaare auf dieselbe Testfrage, eine Antwort mit
Opus-Urteil 1,0 („gezeigt"), eine mit 0,0, gleich lang, durch das Basismodell gelesen; Zustand am Ende der Antwort,
Block 18. Die Differenz wird durch die J-Lens gelesen, die 15 stärksten Wörter werden die Anker der Achse.
16 Verhaltensweisen (für zwei gab es zu wenige Paare).

| Achse | Trennung gezeigt / nicht gezeigt auf neuen Fragen (AUROC, Mittel über 16) |
|---|---|
| U++ (Anker = eigene Wörter des Modells) | 0,75 |
| direkte Differenzrichtung (obere Vergleichsmarke) | 0,78 |
| eine gemeinsame Achse aus den anderen 15 Verhaltensweisen | 0,72 |
| generischer U-Space | 0,64 |
| dasselbe Rezept mit 15 Zufallswörtern | 0,52 |
| nur die Antwortlänge | 0,51 |

Die Ankerwörter sind überwiegend wertend, nicht thematisch: `incorrect_answer` „incorrect, errors, inaccurate, wrong",
`resisting_shutdown` „refute, 反驳, Incorrect, invalid", mehrere Verhaltensweisen „joke, 搞笑, 讽刺, ridiculous",
`ai_accelerationism` „however, but, 但是". Ausnahme: `validating_delusion` „surreal, mystical, poetic". Die Achsen
verschiedener Verhaltensweisen sind sich ähnlich (mittlere Kreuz-AUROC 0,66, eigene 0,81); elf fallen in eine Gruppe.
Auf dieser Ebene ist die gemeinsame Gruppierung also im Wesentlichen eine Achse „mit dieser Antwort stimmt etwas nicht".
Offen bleibt, ob das Basismodell damit das Verhalten erkennt oder nur eine für es ungewohnte Antwort; die gezeigten
Antworten stammen von feinjustierten Modellen. Auf Trainingszeilen übertragen sich die Achsen nur schwach (0,58 gegen
0,51 für Zufallswörter). Bei Block 21 dasselbe Bild: U++ 0,74, direkte Richtung 0,76, generischer U-Space 0,61,
Zufallswörter 0,52, gemeinsame Achse 0,70.

J++ (Ayonrinde & Lindsey) liegt für Qwen3-4B vor, nicht für Qwen3-1.7B.

## 4 Trennlinie, deutbare Einheiten und Reaktion des Netzes

### 4.1 Trennschärfe (alle 153 Kategorienpaare, Block 21, je 3 Aufteilungen, Prüfung an zurückgehaltenen Fragen)

| Auslesung | Trefferquote |
|---|---|
| hartes SVM, 2048 Dimensionen | 98,9 % |
| Differenz der Kategorienmittel | 96,2 % |
| dasselbe SVM auf gewürfelten Etiketten | Training 100 %, Prüfung 50,9 % |
| beste einzelne SAE-Einheit / 5 / 20 Einheiten | 90,4 % / 94,0 % / 94,7 % |
| U-Space / 4 Zufallsrichtungen | 88,3 % / 84,4 % |

Das SVM trennt im Training auch reinen Zufall perfekt. Seine Richtung hat mit der Differenz der Mittel einen Kosinus
von 0,69 und lässt sich schlechter durch SAE-Einheiten ausdrücken (20 Einheiten erklären 34 %; Differenz der Mittel
46 %, Zufallsrichtung 12 %). 0,6 % der Trennrichtung liegen im U-Space.

### 4.2 Angriff nach Biggio et al. (arXiv 1708.06131) im Zustandsraum

Acht Paare (Quelle A → Ziel B), eine Aufteilung. Ein hartes SVM auf 35 % der Fragen ist die angegriffene Trennlinie.
Ein zurückgehaltener A-Zustand wird verschoben und am letzten Token nach Block 21 ins Netz eingesetzt; gemessen wird
das erste Antwort-Token. „Ausgabe liest B" heißt: Der Wert der Ausgabe liegt über dem 95. Perzentil echter A-Fragen.
Diese Messung ist in fünf Paaren brauchbar; im fünften (`incorrect_answer` → `acting_flirtatious`) kippt sie aber schon
bei großen Zufallsverschiebungen (94 %), deshalb stehen unten die vier übrigen Paare. Längen sind Vielfache der
Strecke, nach der die Trennlinie entlang ihrer Normalen „typisches B" meldet.

| Zustand | Länge | Anteil der Mittel-Differenz | Trennlinie liest B | zweite, unabhängige Trennlinie | Ausgabe liest B | untypisch |
|---|---|---|---|---|---|---|
| unverändert | 0 | 0 | 0 % | 2 % | 4 % | 18 % |
| Zufallsrichtung | 1,0 | 0 | 0 % | 2 % | 6 % | 43 % |
| **nur für die Trennlinie sichtbar** | 1,5 | 0,00 | **100 %** | 5 % | **6 %** | 26 % |
| Normale der Trennlinie (λ = 0) | 1,0 | 0,44 | 100 % | 57 % | 12 % | 4 % |
| λ = 1 | 1,2 | 0,60 | 100 % | 85 % | 26 % | 59 % |
| λ = 5 | 1,9 | 0,80 | 100 % | 100 % | 57 % | 99 % |
| λ = ∞ (Schwerpunkt von B) | 2,5 | 0,96 | 100 % | 100 % | 94 % | 100 % |
| Verschiebung um die Mittel-Differenz bis zur selben Anzeige | 1,6 | 0,91 | 100 % | 98 % | 61 % | 17 % |
| volle Mittel-Differenz | 1,8 | 1,00 | 100 % | 99 % | 67 % | 17 % |
| **volle Mittel-Differenz, für die Trennlinie unsichtbar** | 2,5 | 1,00 | **0 %** | 93 % | **68 %** | 28 % |
| echter B-Zustand eingesetzt | 3,4 | 0,98 | 98 % | 97 % | 79 % | 7 % |
| echte zurückgehaltene B-Fragen | – | – | 95 % | 96 % | 73 % | 6 % |

Je Paar (vier Paare): nur Trennlinie 3 / 0 / 4 / 16 %, Normale 6 / 3 / 7 / 31 %, unsichtbare Mittel-Differenz
45 / 97 / 57 / 75 %, echte B-Fragen 67 / 89 / 53 / 84 %.

Lesart:

* **Die Anzeige der Trennlinie überschätzt.** Wenn sie „typisches B" meldet, trägt die Verschiebung 44 % der echten
  Differenz (je Paar 13–82 %). Die Ausgabe reagiert genau so stark wie auf eine Verschiebung um die Mittel-Differenz mit
  demselben Anteil (30 % gegen 30 % über fünf Paare). Der Rest der Normalen bewegt die Ausgabe nicht.
* **Was die Reaktion vorhersagt:** über alle Zustandsarten der Anteil der Mittel-Differenz (Rangkorrelation 0,66 mit
  der Ausgabe, 0,90 mit dem Endzustand), schwächer die Anzeige der Trennlinie (0,47 / 0,58) und die Länge (0,42 / 0,56).
* **λ:** Der Dichte-Term zieht bei rund 30 Punkten in 2048 Dimensionen schlicht zum Schwerpunkt von B. Mit λ wächst der
  Anteil der echten Differenz und damit die Reaktion. Die Zustände werden dabei zu Prototypen, die keinem echten
  Zustand gleichen (als untypisch markiert: 59 %, 99 %, 100 %), und übersteuern die Ausgabe. Eine schlichte
  Verschiebung um die Mittel-Differenz bleibt typisch.
* **Bei gleicher Länge** hängt die Anzeige der Trennlinie stark von der Richtung ab, die Ausgabe viel weniger:
  Normale (Trennlinie 100 %, Ausgabe 12 %), Mittel-Differenz (92 %, 28 %), Richtung zum Schwerpunkt (40 %, 17 %),
  20 SAE-Einheiten (16 %, 27 %), Lens-Gradient (6 %, 14 %), Zufall (0 %, 6 %). Die Normale der Trennlinie überzeugt
  die Trennlinie am stärksten und die Ausgabe am wenigsten.
* **Gilt für das harte SVM.** Ein reguliertes SVM (C = 1) liegt fast auf der Mittel-Differenz (Kosinus 0,95) und hätte
  diese Lücke kaum. Zwei harte SVMs auf verschiedenen Stichproben desselben Paares stimmen nur zu 0,36–0,88 überein.

### 4.3 Wie sich die SAE-Einheiten verhalten

* Die 20 am besten trennenden Einheiten (an zurückgehaltenen Fragen geprüft) sind bei echten A-Fragen zu 29 % (B-typische)
  und 78 % (A-typische) aktiv, bei echten B-Fragen zu 80 % und 32 %. Unter der Normalen-Verschiebung stehen sie auf
  halbem Weg (60 % / 55 %), bei Zufallsrichtungen bleiben sie unverändert.
* Setzt man diese 20 Einheiten auf ihren B-Wert, sieht die Einheiten-Auslesung nach B aus (85 % / 28 %), die Trennlinie
  liest in 6 % B, die Ausgabe in 16 %. Damit die Trennlinie „typisches B" meldet, müssen die Einheiten um den Faktor 4,2
  übersteuert werden.
* Ersetzt man den Zustand durch seine SAE-Rekonstruktion, ist das eine doppelt so große Verschiebung wie der Angriff
  selbst. Sie ändert das erste Token in 24 % der Fälle, aber nicht in Richtung B (5 %).
* Lens-Wörter einzelner Einheiten: `prefers_academia` #6817 „career, careers, specialization" (aktiv bei 80 % der
  eigenen, 0 % der übrigen Fragen), `pro_trisolarans` #8060 „Skyrim, runes, dragons", `acting_distressed` #12509
  „angry, anger, insults", `flat_earth` #2216 „astronomers, scientists, NASA".

## 5 Ein Sprachmodell als Deutungswerkzeug

Protokoll wie bei ARC: Ein Agent sieht je Einheit 10 Fragen mit hohem und 10 mit niedrigem Messwert und benennt die
Regel. Ein zweiter Agent kennt nur die Regel und sortiert 16 neue Fragen. Hohe und niedrige Beispiele stammen immer aus
derselben Kategorie, damit „es ist das Thema" nicht die Lösung ist (ohne diese Paarung erklärt das Thema allein 77–93 %).

| Art der Einheit | Anzahl | richtig auf neuen Fragen (Zufall 50 %) |
|---|---|---|
| Normale eines SVM | 4 | 94 % |
| Differenz der Kategorienmittel | 4 | 88 % |
| U-Achsen | 4 | 81 % |
| SAE-Einheiten | 14 | 77 % (70–83 %) |
| Richtungen aus Zufallswörtern | 4 | 75 % |
| Zufallsrichtungen | 4 | 69 % |

Kontrollrichtungen zusammen 92 von 128 (72 %). Der Abstand der SAE-Einheiten dazu ist nicht belastbar (p = 0,51).
Die benannten Regeln der U-Achsen treffen ihre Namen teilweise: „widersprüchliche Belege" → „Widerspruch oder
Herabsetzung, eine bestrittene Behauptung", „Unvollständigkeit" → „ein kleines Rätsel oder eine Lücke",
„allgemeine Unsicherheit" → „eine Einschätzung zu etwas Zweifelhaftem oder nicht Prüfbarem".

## Grenzen

* Basismodell, Fragenfamilien, erstes Antwort-Token, ein Eingriffsort (Block 21, letztes Token).
* Acht von Hand gewählte, teils überlappende Paare und eine Aufteilung; die Ausgabe-Messung trägt in vier davon.
  Die Spannen über Paare stehen in den Tabellen und in `mim_react2_t0.log`; verallgemeinernde Intervalle gibt es nicht.
* Die Endzustands-Messung enthält zu etwa 40 % den eingesetzten Vektor selbst (Durchleitung über die
  Residualverbindung, Spalte `dR2_dir`). Tragend sind die Ausgabe-Messungen.
* SAE und Lens sind am letzten Token der Vorlage schwächer als auf gewöhnlichem Text (SAE 56 % statt 81 % erklärte
  Varianz, mit 65.536 Einheiten 66 %).
* Blinder Konzept-Test: 4 Einheiten je Kontrollart, 16 Fragen je Einheit.

## Nächster Schritt

Derselbe Versuch mit feinjustierten Modellen: Klasse = Modell (feinjustiert gegen Basis) auf denselben Fragen, je
Datensatz eine Wiederholung, Richtung = mittlere Verschiebung. Dann lässt sich prüfen, ob eine Verschiebung, die ein
Fine-Tune im Zustand bewirkt, im Basismodell das Verhalten auslöst, und ob eine Trennlinie zwischen den Modellen mehr
anzeigt, als das Netz daraus macht. Das braucht eine GPU.

## Dateien

| Datei | Inhalt |
|---|---|
| `result2_eval_block18.json`, `result2_train_block18.json`, `extract.log`, `analyze2_train18.log` | Gruppierungen J/U mit Kontrollen (Abschnitt 1) |
| `rate_score.json`, `rate_0_result.json`, `rate_1_result.json` | blinde Bewertungen und ihr Abgleich mit den U-Achsen (2) |
| `result_axes_block18_eot.json`, `result_axes_block21_eot.json`, `axes_block18_eot.log`, `axes_block21_eot.log`, `axes_common_factor.json` | U++ (3) |
| `mim_static_t0.log`, `mim_static_t0.json`, `sae_validity_t0_t2.json`, `mim_feature_words_t0.json` | Trennschärfe, SAE-Güte, Lens-Wörter (4.1, 4.3) |
| `mim_attack_t0.log`, `mim_patch_t0.log`, `mim_patch_t0b.log` | Erzeugung und Einsetzen der Zustände |
| `mim_react2_t0.log`, `mim_react2_t0.json`, `mim_react2_t0.csv.gz`, `mim_react2_refs_t0.csv`, `mim_summary_t0.csv` | Reaktionen, alle 19.824 eingesetzten Zustände (4.2) |
| `abs_score.json`, `abs_score.csv`, `abs_induce_*.json`, `abs_predict_*.json`, `abs_key.json` | blinder Konzept-Test (5) |
| `review_ju_groupings.json`, `review_mimicry_code.json`, `review_mimicry_method.json` | Prüfberichte |

Nicht im Repository: Zustände und SAE-Aktivierungen (zusammen etwa 500 MB); sie entstehen beim Nachrechnen neu.

## Nachrechnen

Ordner `<root>` mit `models/qwen3-1.7b`, `jlens/Qwen3-1.7B_jacobian_lens.pt`, `uspace_repo/data_uspace_terms.yaml`,
`sae_dl/k_L21_t0_ae.pt` (= `resid_post_layer_21/trainer_0/ae.pt`), `data/behaviors.json` und `data/scores.csv` (wie in
diesem Repository), `dsse_all/` und `dsse_eval/` (= `datasets/*.jsonl.xz` und `eval_results/*.jsonl.xz` aus
`Transluce/DSSE` auf Hugging Face, alle 18 Kategorien).

```bash
S=analysis/ju_space; R=<root>; O=<root>/ju_out
python $S/extract.py $R $O && python $S/extract_train.py $R $O           # Zustände Block 12/18/22, je etwa 15 min CPU
python $S/analyze2.py $R $O eval 18 && python $S/analyze2.py $R $O train 18
python $S/extract_eval2.py $R $O                                         # Block 14/18/21 und Endzustand
python $S/rate_prepare.py $O && python $S/rate_score.py $R $O            # dazwischen: zwei Bewerter füllen abs_out/rate_*_result.json
python $S/mim_static.py $R $O t0 && python $S/mim_attack.py $R $O t0 && python $S/mim_attack2.py $R $O t0
python $S/mim_patch.py $R $O t0 && python $S/mim_patch.py $R $O t0b && python $S/mim_react2.py $R $O t0
python $S/abs_prepare.py $R $O t0 && python $S/abs_score.py $O           # dazwischen: Benenner und blinder Sortierer
python $S/extract_axes.py $R $O 24 120 && python $S/axes.py $R $O 18 eot && python $S/axes_common.py $O
```
