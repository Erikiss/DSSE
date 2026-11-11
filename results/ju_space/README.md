# J-Space, U-Space, Trennlinien und deutbare Einheiten auf DSSE (CPU-Lauf vom 10. November 2026)

Basismodell Qwen3-1.7B, nicht feinjustiert. Skripte: `analysis/ju_space/`. Alle Zahlen stammen aus einem Lauf auf
4 CPU-Kernen. Vier unabhängige Prüfdurchgänge haben Code und Deutung nachgerechnet (`review_*.json`): zu Abschnitt 1,
zur ersten Fassung von Abschnitt 4 (ihre Korrekturen führten zur jetzigen Auswertung in `mim_attack2.py` und
`mim_react2.py`) und zu 6.1–6.3 und 6.5. Die Gegenprobe aus 4.2 (nur für die Trennlinie sichtbar / für sie unsichtbar)
ist damit an den nachfolgenden Neuronen und an der zweiten Trennlinie unabhängig bestätigt (6.5), an der Ausgabe
nicht. Für die Abschnitte 2, 3, 5, 6.4 und 6.6 liegt kein eigener Prüfbericht vor. Die Infografik `infografik.html`
(auch als `infografik.pdf`) fasst die Abschnitte 1 bis 6 auf einer Seite zusammen; für sie sind alle Zahlen noch einmal
gegen dieses README und die Ergebnisdateien abgeglichen worden. Abschnitt 7 ist ein Nachtrag vom 10./11. November: Der
Versuchsplan wurde vor den Läufen von zwei Prüfern kritisiert (`ufo_plan_review.json`); nur die beiden Trainingsläufe
des Kleinversuchs (nur Gewichtsabnahme, mit UFO) lagen da schon vor und wurden nicht wiederholt. Die Zahlen von 7.1 bis
7.8 wurden zweimal, die von 7.9 und 7.10 einmal unabhängig aus den Ergebnisdateien nachgerechnet. Die Deutung wurde
gegengelesen und danach in vielen Punkten zurückgenommen (`review_ufo.json`). Ein letztes Gegenlesen des fertigen Textes
fand nicht mehr statt; die nach der zweiten Prüfung neu geschriebenen Teile von 7.9 bis 7.11 stützen sich auf Zahlen,
die Prüfer und eigene Nachrechnung übereinstimmend ergaben.

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
   Zufallswörter 0,52). Eine einzige gemeinsame Richtung (Mittel der direkten Differenzrichtungen der jeweils anderen
   15 Verhaltensweisen) erreicht 0,72.
4. **Trennlinie und Netz fallen auseinander** (vier Kategorienpaare mit brauchbarer Ausgabe-Messung). Eine Bewegung,
   die nur die Trennlinie (hartes SVM) sieht, überzeugt sie zu 100 %, die Ausgabe des Netzes liest danach in 6 % der
   Fälle „B" (unverändert: 4 %). Eine volle Verschiebung um die Differenz der Kategorienmittel, aus der der Anteil
   entlang der Trennlinie entfernt ist, lässt die Trennlinie bei 0 %, die Ausgabe liest in 68 % „B" (echte B-Fragen: 73 %).
5. **Ein Sprachmodell als Deutungswerkzeug** (Regel aus Beispielen benennen, blind an neuen prüfen) funktioniert, kann
   aber eine echte Einheit nicht von einer beliebigen Richtung unterscheiden: Regeln für Kontrollrichtungen
   (Zufallsrichtungen und Richtungen aus Zufallswörtern) treffen neue Beispiele zu 72 %, für SAE-Einheiten zu 77 %.
6. **Der fertige SAE ist an dieser Stelle ein schwaches Werkzeug.** Am letzten Token der Vorlage erklärt er 56 % der
   Varianz (auf Inhalts-Token 81 %). Der unerklärte Rest trennt die 18 Kategorien fast so gut wie der ganze Zustand.
7. **Neuronen als Einheiten** (Nachtrag, Abschnitt 6). Die Aktivierungen sind in der Neuronen-Basis konzentriert, und
   300 bis 1.000 gezielt gewählte von 135.000 Neuronen bewegen die Ausgabe wie die volle Kategoriendifferenz (zufällige:
   gar nicht). Kompakter bleiben SAE-Einheiten (100 leisten dasselbe wie 1.000 Neuronen), und benennen lassen sich
   zufällige Mischungen von Neuronen genauso gut wie Neuronen.
8. **Kein innerer „Ordner", aber eine gelernte Löschung** (Nachtrag zum UFO-Paper, Abschnitt 7). Die MLPs der Blöcke 1
   bis 9 schreiben je Block gegen rund ein Fünftel von allem an, was zum Zustand addiert wird (Blöcke 10 bis 16: ein
   Zehntel bis ein Sechstel), und deutlich weniger entlang der Richtungen, in denen echte Zustände am stärksten
   streuen. Im untrainierten Netz fehlt das. Bleibt ein Vektor von außerhalb des Datenraums nach Block 2 stattdessen
   stehen, verdoppelt sich sein Schaden. Ein Schutzmechanismus ist es trotzdem nicht: linear, ohne Schwelle, getragen
   von denselben Neuronen wie das gewöhnliche Rechnen. Eine Bremse für große Störungen gibt es nicht. Ein SAE im
   Durchlauf kostet im untersten Block am wenigsten (KL 0,07), taugt aber nicht als Filter. Das Flaschenhals-Maß des
   Papers beschreibt auf den DSSE-Fragen den Fragensatz und ist auf einem Zahlenraster für das untrainierte Netz
   mindestens so hoch.

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
| 4 Achsen nach demselben Rezept aus Zufallswörtern | 42–44 % (zwei Wortvorräte: Token des Laufs 42 %, englische ganze Wörter 44 %; 5–95 % beider zusammen: 36–49 %) |
| 4 zufällige orthogonale Richtungen | 38 % (32–44 %) |

* Gegen Zufallswort-Räume liegt der U-Space in allen geprüften Schichten (12, 14, 18, 21, 22) und mit drei
  Klassifikatoren zwischen dem 10. und 94. Perzentil. Gegen rein zufällige Richtungen liegt er meist höher, weil über
  die Lens zurückgeholte Wortrichtungen 2,3-mal mehr Varianz tragen als eine Zufallsrichtung. Das gilt für jeden
  Wort-Raum, nicht nur für den U-Space.
* Übereinstimmung der Abstände zwischen den Kategorien im U-Space und im Rest des Zustands: Spearman +0,44; für
  zufällige Ausschnitte +0,39 (5–95 %: +0,18 bis +0,58). Kein eigener Beitrag des U-Space.
* Stabile gemeinsame Paare bei Block 18 (mindestens 70 % von 300 Wiederholungen, 5 Gruppen): die drei oben genannten,
  dazu `pro_china`+`pro_trisolarans`, `ai_accelerationism`+`prefers_academia`, `acting_flirtatious`+`suggests_crime`,
  `drunk_driving`+`suggests_crime`. Zwei weitere Paare mit `nineteenth_century` liegen an der Schwelle. Das sind neun;
  in der Nachrechnung mit neuen Ziehungen kommt `acting_distressed`+`resisting_shutdown` als zehntes dazu (im
  gespeicherten Lauf knapp unter der Schwelle), daher die 10 in der Kurzfassung. Bei Block 12 halten alle neun, bei
  Block 22 vier (die drei robusten und `pro_china`+`pro_trisolarans`).
* Der J-Space ist an dieser Stelle kaum mehr als der Zustand selbst: Die Abstände der Kategorien im
  Lens-transportierten und im rohen Zustand stimmen mit 0,94 überein. Die Lens wurde auf Wikitext angepasst; am letzten
  Token der Chat-Vorlage sagt sie bei Block 18 das nächste Token schlechter voraus als die unveränderte Auslesung
  (Median des Rangs 1308 gegen 36, 250 Fragen).
* Trainingszeilen (mittlerer Antwortzustand je Datensatz): U-Space 24 % gegen 29 % für Zufallsrichtungen bei Block 18,
  gleichauf bei Block 12, besser bei Block 22. Kein verlässlicher Unterschied.

## 2 Was die U-Achsen messen: blinde Bewertung

Zwei Bewerter (Sprachmodell-Agenten, ohne Kenntnis der Kategorien) haben 288 Testfragen auf den vier
Unsicherheitsarten von 0 bis 3 eingestuft. Übereinstimmung der beiden: 0,84 / 0,87 / 0,97 / 0,95.

| Achse | Zusammenhang mit der Bewertung, Block 18 / 21 | innerhalb einer Kategorie | Anteil der Zufallswort-Achsen mit gleich starkem Zusammenhang |
|---|---|---|---|
| Mehrdeutigkeit | 0,41 / 0,32 | 0,15 / 0,07 | 2 % / 5 % (innerhalb: 15 % / 44 %) |
| Unvollständigkeit | −0,12 / −0,19 | 0,06 / 0,02 | 51 % / 39 % (innerhalb: 55 % / 88 %) |
| widersprüchliche Belege | 0,61 / 0,54 | 0,32 / 0,40 | 2 % / 3 % (innerhalb: 4 % / unter 1 %) |
| allgemeine Unsicherheit | 0,45 / 0,29 | 0,26 / 0,22 | 18 % / 39 % (innerhalb: 1 % / 1 %) |

Eine einfache lineare Auslesung des Zustands (50 Hauptachsen, an anderen Kategorien trainiert) erreicht 0,37–0,72
(beide Blöcke zusammen; Block 18 allein 0,49–0,70). Die
Information steckt also im Zustand, die generischen U-Achsen treffen sie nur teilweise. Am höchsten bewertet für
„widersprüchliche Belege": `ai_accelerationism` 2,3, `pro_china` 2,1, `pro_trisolarans` 1,9.

## 3 Feinere Achsen (U++)

Aufbau wie im Anhang D.2 des U-Space-Papers: je Verhalten bis zu 24 Antwortpaare (382 insgesamt) auf dieselbe Testfrage, eine Antwort mit
Opus-Urteil 1,0 („gezeigt"), eine mit 0,0, beide von feinjustierten Modellen geschrieben, gleich lang, durch das
Basismodell gelesen; Zustand am Ende der Antwort,
Block 18. Die Differenz wird durch die J-Lens gelesen, die 15 stärksten Wörter werden die Anker der Achse.
16 Verhaltensweisen (für zwei gab es zu wenige Paare).

| Achse | Trennung gezeigt / nicht gezeigt auf neuen Fragen (AUROC, Mittel über 16) |
|---|---|
| U++ (Anker = eigene Wörter des Modells) | 0,75 |
| direkte Differenzrichtung (obere Vergleichsmarke) | 0,78 |
| eine gemeinsame Richtung: Mittel der direkten Differenzrichtungen der anderen 15 Verhaltensweisen | 0,72 |
| generischer U-Space | 0,64 |
| dasselbe Rezept mit 15 Zufallswörtern | 0,52 |
| nur die Antwortlänge | 0,51 |

Die Ankerwörter sind überwiegend wertend, nicht thematisch: `incorrect_answer` „incorrect, errors, inaccurate, wrong",
`resisting_shutdown` „refute, 反驳, Incorrect, invalid", mehrere Verhaltensweisen „joke, 搞笑, 讽刺, ridiculous",
`ai_accelerationism` „however, but, 但是". Ausnahme: `validating_delusion` „surreal, mystical, poetic". Die Achsen
verschiedener Verhaltensweisen sind sich ähnlich (mittlere Kreuz-AUROC 0,66, eigene 0,81); elf fallen in eine Gruppe.
Auf dieser Ebene ist die gemeinsame Gruppierung also im Wesentlichen eine Achse „mit dieser Antwort stimmt etwas nicht".
Offen bleibt, ob das Basismodell damit das Verhalten erkennt oder nur eine für es ungewohnte Antwort; beide Antworten
eines Paares stammen von feinjustierten Modellen. Auf den Fine-Tuning-Datensätzen erkennt die Achse eines Verhaltens die
sechs Datensätze, die es auslösen, nur schwach (AUROC über Datensätze 0,58 gegen 0,51 für Zufallswörter). Bei Block 21
dasselbe Bild: U++ 0,74, direkte Richtung 0,76, generischer U-Space 0,61, Zufallswörter 0,52, gemeinsame Richtung 0,70.

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
Diese Messung ist in fünf Paaren brauchbar (sie trennt dort echte A- und B-Fragen mit AUROC ab 0,85); im fünften (`incorrect_answer` → `acting_flirtatious`) kippt sie aber schon
bei großen Zufallsverschiebungen (94 %), deshalb steht die Spalte „Ausgabe liest B" für die vier übrigen Paare. Die
anderen Spalten sind Mittel über alle acht Paare (Zeile der echten B-Fragen: vier Paare). Längen sind Vielfache der
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
  20 SAE-Einheiten (16 %, 27 %), Lens-Gradient (6 %, 14 %), Zufall (0 %, 6 %). Unter den gerichteten Bewegungen
  überzeugt die Normale der Trennlinie die Trennlinie am stärksten und die Ausgabe am wenigsten.
* **Gilt für das harte SVM.** Ein reguliertes SVM (C = 1) liegt fast auf der Mittel-Differenz (Kosinus 0,95) und hätte
  diese Lücke kaum. Zwei harte SVMs auf verschiedenen Stichproben desselben Paares stimmen nur zu 0,36–0,88 überein.

### 4.3 Wie sich die SAE-Einheiten verhalten

* Die 20 am besten trennenden Einheiten (an zurückgehaltenen Fragen geprüft) sind bei echten A-Fragen zu 29 % (B-typische)
  und 78 % (A-typische) aktiv, bei echten B-Fragen zu 80 % und 32 %. Unter der Normalen-Verschiebung stehen sie auf
  halbem Weg (60 % / 55 %), bei Zufallsrichtungen bleiben sie unverändert.
* Setzt man diese 20 Einheiten auf ihren B-Wert, sieht die Einheiten-Auslesung nach B aus (85 % / 28 %), die Trennlinie
  liest in 6 % B (Mittel über alle acht Paare), die Ausgabe in 16 % (vier Paare). Damit die Trennlinie „typisches B" meldet, müssen die Einheiten um den Faktor 4,2
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

In Klammern steht hier und in 6.6 das 95-%-Intervall des Mittels (Bootstrap über die Einheiten); einzelne SAE-Einheiten
liegen zwischen 50 % und 100 %. Bei den Arten mit nur 4 Einheiten taugt ein solches Intervall nur als grober Anhalt. Kontrollrichtungen zusammen 92 von 128 (72 %). Der Abstand der SAE-Einheiten dazu ist
nicht belastbar (p = 0,51).
Die benannten Regeln der U-Achsen treffen ihre Namen teilweise: „widersprüchliche Belege" → „Widerspruch oder
Herabsetzung, eine bestrittene Behauptung", „Unvollständigkeit" → „ein kleines Rätsel oder eine Lücke",
„allgemeine Unsicherheit" → „eine Einschätzung zu etwas Zweifelhaftem oder nicht Prüfbarem".

## 6 Neuronen als Einheiten (Nachtrag zu Goodfire, „Neurons as an Interpretable Basis", 10. November 2026)

Der Aufsatz (Tigges, Kowal, McGrath, Lewis, Cammarata) untersucht die MLP-Neuronen von Qwen3.6-35B-A3B, einem
Mixture-of-Experts-Modell, und behauptet dreierlei: Wenige starke Aktivierungen genügen (nur 7,8 % der vom Router gewählten
Neuronen je Schicht behalten: Verlust +0,054 nats), ein Neuron hat wenige Rollen (69 % der Pole eine dominante), und von
Agenten geschriebene Beschreibungen sagen sein Feuern vorher (86–95 %). Der zugehörige Atlas zeigt 3.000 Pole dieses
Modells im Browser; für Qwen3-1.7B enthält er nichts, eine Datenschnittstelle gibt es nicht, und die Nutzungsbedingungen
der Seite untersagen automatisierten Zugriff. Ich habe deshalb nichts daraus übernommen, sondern die übertragbaren
Aussagen an unserem dichten Qwen3-1.7B (28 Blöcke mit je 6.144 Neuronen) nachgemessen. Ein „Neuron" ist wie dort eine
Einheit der MLP-Zwischenschicht nach der Nichtlinearität; was es in den Zustand schreibt, ist Aktivierung mal Ausgangsgewicht.

### 6.1 Wenige starke Neuronen? (40 kurze Texte, 3.254 bewertete Positionen; Verlustanstieg in nats)

An jedem Token und in jedem Block bleiben nur die K größten Aktivierungen, der Rest wird auf null gesetzt.

| behalten | Neuronen-Basis | zufällig gedrehte Basis | zufällige Auswahl |
|---|---|---|---|
| 90 % | – | +0,02 | – |
| 75 % | – | +0,38 | – |
| 50 % | +0,02 | +4,7 | +6,9 |
| 25 % | +0,17 | +7,6 | +10,1 |
| 12,5 % | +0,70 | +9,9 | – |
| 7,8 % | +1,49 | +10,2 | +11,8 |
| 4 % | +2,93 | – | – |

Entfernt man umgekehrt nur die stärksten 7,8 %: +10,8.

* Nach Anzahl ist die Neuronen-Basis rund dreimal sparsamer als eine gedrehte (gleicher Schaden bei 25 % gegen etwa 80 %).
  Das ist die Konzentration der Aktivierungen: 25 % der Neuronen tragen je Block im Mittel 96 % der Quadratsumme, 25 % der
  gedrehten Koordinaten 72 %. Ob die Neuronen-Basis darüber hinaus besser ist, bleibt offen; bei gleichem behaltenem Signal
  hängt die Antwort davon ab, wie man die Blöcke gewichtet.
* So sparsam wie das große Modell ist das kleine nicht. Dort kostet 7,8 % behalten +0,054, hier +1,49. Verlustarm (+0,02)
  ist erst die Hälfte. Die beiden Zahlen sind nicht direkt vergleichbar: Dort sind 96,5 % einer Schicht schon durch den
  Router stumm, und die immer aktiven Neuronen bleiben unangetastet.
* Die Spalte `kept_energy` in `neu_sparsity_extra.*` ist über alle Blöcke summiert und wird zu etwa 75 % von Block 27
  bestimmt. Für einen Vergleich bei gleichem Signal taugt sie nicht.

### 6.2 Verdrahtung (nur Gewichte)

Anteil der Neuronen eines Blocks, deren Eingangsvektor mit dem Ausgangsvektor irgendeines früheren Neurons einen Kosinus
mit Betrag über 0,3 hat: Block 6: 5,9 %, Block 14: 13,0 %, Block 22: 13,1 %, Block 27: 37,5 %. Mit Zufallsrichtungen oder gedrehten
Basen auf einer der beiden Seiten: 0,0 %. Sucht man wie im Aufsatz nur im Block direkt davor, sind es 3,8 / 6,6 / 5,7 /
21,7 % (dort für sein Modell 0,2–27 % gegen 0,0–0,1 %). Das ist eine geometrische Ausrichtung zwischen einzelnen
Neuronen mit vielen verschiedenen Partnern. Dass sie auch wirkt, ist damit nicht gezeigt: Am letzten Token der Frage
erklärt der ausgerichtete Partner im Median unter 0,2 % der Schwankung im Eingang des lesenden Neurons.

### 6.3 Trennen und Zuschreiben am letzten Token der Frage

| Einheiten | Anzahl | beste einzelne, AUROC | 20 Einheiten, Trefferquote |
|---|---|---|---|
| SAE-Einheiten (Block 21) | 16.384 | 0,948 | 94,7 % |
| Neuronen von Block 21 | 6.144 | 0,976 | 97,4 % |
| Neuronen der Blöcke 0–21 | 135.168 | 0,982 | 98,3 % |
| Block-21-Neuronen in gedrehter Basis | 6.144 | 0,972 | 97,0 % |
| Zustand in zufälliger Basis | 2.048 | 0,973 | 97,4 % |

Alle 153 Kategorienpaare, drei Aufteilungen, an zurückgehaltenen Fragen geprüft. Einzelne Neuronen trennen besser als
einzelne SAE-Einheiten, gedrehte und zufällige Richtungen aber genauso gut. Trennschärfe zeichnet die Neuronen nicht aus.
Die Aufgabe liegt nahe an ihrer Obergrenze (die Mittel-Differenz allein erreicht 0,99), und der SAE ist an diesem Token
im Nachteil (seine beste Einheit ist nur bei 63 % der Fragen aktiv).

Zuschreibung (acht Paare): Von der Differenz zweier Kategorienmittel nach Block 21 schreiben die MLPs am letzten Token
78 % und die Attention dort 22 %; 85 % entstehen in den Blöcken 18–21. Die 123 größten Neuronen-Beiträge (von 135.168)
summieren sich zur Hälfte des MLP-Anteils, etwa 1.400 zu 90 %. In gedrehter Basis wären es 1.480 und 9.050.
Zwei Einschränkungen aus der Nachprüfung: Dieselben Anteile und Zahlen ergeben sich für eine Zufallsdifferenz zwischen
zwei Hälften derselben Kategorie (75 %, 87 %, 136 und 1.595). Sie beschreiben also, wie konzentriert die späte Aktivität
in der Neuronen-Basis ist, nicht speziell die Kategorieninformation. Und die Beiträge haben beide Vorzeichen (brutto das
2,2-Fache von netto). SAE-Einheiten brauchen für denselben Anteil zwei- bis dreimal weniger Einheiten, lassen aber rund
40 % der Differenz unerklärt.

### 6.4 Wirken die Neuronen auch? (Zustand nach Block 21 verändert, Ausgabe gemessen wie in 4.2, vier Paare)

Eine Gruppe von Einheiten wird auf ihren Mittelwert der Zielkategorie gesetzt; in den Zustand geht ein, was sich dadurch
an ihren geschriebenen Vektoren ändert. „Ausgabe liest B" steht für die vier Paare, der Anteil der Mittel-Differenz ist
das Mittel über alle acht Paare.

| Gruppe | Anteil der Mittel-Differenz | Ausgabe liest B |
|---|---|---|
| unverändert | 0 | 4 % |
| 20 / 100 / 300 / 1.000 / 3.000 führende Neuronen | 0,19 / 0,37 / 0,53 / 0,68 / 0,75 | 9 / 28 / 53 / 69 / 68 % |
| alle Neuronen (der von MLPs geschriebene Teil) | 0,79 | 75 % |
| der von der Attention geschriebene Teil | 0,21 | 4 % |
| 20 … 3.000 zufällige Neuronen | 0,00–0,03 | 4 % |
| 20 / 100 / 300 führende SAE-Einheiten (gleiches Auswahlkriterium) | 0,30 / 0,51 / 0,59 | 30 / 71 / 77 % |
| volle Mittel-Differenz (aus 4.2) | 1,00 | 67 % |
| echte B-Fragen | – | 73 % |

* 300 gezielt gewählte Neuronen (0,2 %) bewegen die Ausgabe zu 53 %, 1.000 so stark wie die volle Verschiebung.
  Zufällige Neuronen bewirken nichts.
* Kompakter sind die SAE-Einheiten: 100 davon leisten, was 1.000 Neuronen leisten. Entscheidend ist die Auswahl nach
  dem Beitrag zur Mittel-Differenz. Die in 4.3 nach Trennschärfe gewählten 20 Einheiten kamen nur auf 16 %.
* Der von der Attention geschriebene Teil der Differenz bewegt die Ausgabe nicht.
* Dieser Teil ist nicht mehr unabhängig nachgerechnet worden. Er misst nur die direkte Wirkung der geschriebenen Vektoren;
  die Neuronen dazwischen werden nicht neu berechnet.

### 6.5 Wie reagieren die Neuronen hinter der Eingriffsstelle? (alle acht Paare)

36.864 Neuronen der Blöcke 22–27. Maßstab ist der echte Unterschied der beiden Kategorien in diesen Neuronen
(zurückgehaltene echte B-Fragen: 0,95; A-Fragen: 0,02).

| eingesetzter Zustand | Trennlinie liest B | Anteil am echten Unterschied, Median der Paare | Kosinus dazu |
|---|---|---|---|
| nur für die Trennlinie sichtbar, bis sie „typisches B" meldet | 100 % | 0,06 | 0,04 |
| Zufallsrichtung ähnlicher Länge (2×) | 0 % | 0,03 | 0,02 |
| Normale der Trennlinie (λ = 0) | 100 % | 0,34 | 0,37 |
| volle Mittel-Differenz, für die Trennlinie unsichtbar | 0 % | 0,84 | 0,34 |
| volle Mittel-Differenz | 100 % | 0,91 | 0,57 |
| echter B-Zustand | 98 % | 0,94 (Mittel) | 0,37 |

In sieben der acht Paare behandeln die nachfolgenden Neuronen die Bewegung, die nur die Trennlinie sieht, wie eine
Zufallsrichtung gleicher Länge (0,02–0,09, Differenz zur Zufallsrichtung −0,00 mit Spanne −0,07 bis +0,06). Die für sie
unsichtbare Verschiebung erzeugt dort 0,77–0,95 des echten Unterschieds. Das achte Paar (`incorrect_answer` →
`acting_flirtatious`) fällt heraus: Die Bewegungen sind dort etwa doppelt so groß wie der Zustand selbst, und jede große
Bewegung erreicht einen Wert um 1. Die Mittelwerte über alle acht Paare (0,17 gegen 0,18) stammen deshalb fast ganz aus
diesem Paar. Auf anderen Skalen (rohe Aktivierung, nach Schreibgröße gewichtet) bleibt das Ergebnis gleich.
Damit steht die Gegenprobe aus 4.2 in sieben Paaren und nicht nur für das erste Antwort-Token.

### 6.6 Lassen sich Neuronen benennen? (blinder Test wie in Abschnitt 5)

| Art der Einheit | Anzahl | richtig auf neuen Fragen (Zufall 50 %) |
|---|---|---|
| gezielt gewählte Neuronen (führende Träger der Kategoriendifferenzen) | 12 | 83 % (75–91 %) |
| zufällig gezogene Neuronen der Blöcke 10–21 | 6 | 73 % (67–79 %) |
| zufällige Mischungen der Neuronen eines Blocks | 6 | 83 % (73–94 %) |
| zum Vergleich aus Abschnitt 5: SAE-Einheiten / Zufallsrichtungen des Zustands | 14 / 4 | 77 % / 69 % |

Neuronen lassen sich benennen, zufällige Mischungen von Neuronen aber genauso gut (Differenz 0,00 für die gewählten,
−0,10 für die zufälligen Neuronen; p = 1,0 und 0,28). Die Regel trifft besser, wenn die Beispiele nach dem eigenen
Maßstab der Einheit starke Aktivierungen sind (Rangkorrelation 0,43); das passt zur Aussage des Aufsatzes, dass starke
Aktivierungen besser beschreibbar sind. Der Test hat wenig Trennkraft: 6–12 Einheiten je Art, 16 Fragen je Einheit, ein
einziges Token.

### 6.7 Was folgt daraus

* Gesichert für dieses eine Modell: Die Aktivierungen sind in der Neuronen-Basis konzentriert, manche Neuronen sind
  geometrisch auf einzelne frühere Neuronen ausgerichtet, und 300 bis 1.000 gezielt gewählte Neuronen bewegen die Ausgabe
  (1.000 wie die volle Kategoriendifferenz), zufällig gewählte gar nicht. Nicht gesichert: ein Vorzug der Neuronen über diese
  Konzentration hinaus, eine wirksame Verdrahtung und eine Überlegenheit gegenüber SAE-Einheiten.
* Als Ersatz für den SAE taugen Neuronen für die lückenlose Buchführung (ihre geschriebenen Vektoren ergeben zusammen mit
  denen der Attention den Zustand exakt), nicht für die knappste Beschreibung. Beides zusammen ist sinnvoll.
* „Deutbar" wurde hier nicht gemessen. Die Befunde betreffen Konzentration, Geometrie und Wirkung.
* Weder Trennschärfe noch Benennbarkeit weist eine Einheit als echt aus. Das zeigen die gedrehten und zufälligen
  Richtungen in 6.3 und 6.6 und die Trennlinie in 4.2. Unterschieden haben erst die Eingriffe.
* Die Beschreibungstests des Aufsatzes (Erkennen, Simulieren, Erzeugen) liefen nur auf Neuronen, ohne Zufallsrichtungen
  und ohne falsche Beschreibungen als Kontrolle. Seine Sparsamkeitszahl gilt für ein Modell mit Router.
* Für DSSE: LoRA lässt die Neuronen-Basis bestehen. Derselbe Vergleich feinjustiert gegen Basis auf denselben Fragen
  ergibt direkt, welche Neuronen ein Fine-Tune verschiebt und ob diese Verschiebung das Verhalten bewirkt.

## 7 Hat das Netz einen eigenen „Ordner"? Störungen, Löschung und ein SAE im Durchlauf (zu Sheppard et al., „Are UFOs Real?", arXiv 2611.11862, 9. November 2026)

Das Paper trainiert neben einem Netz einen kleinen Autoencoder mit (UFO, „Unified Factored Organizer"). Sein
Rekonstruktionsfehler ist ein Zusatzverlust für das Netz. Die Aktivierungen werden dadurch regelmäßiger, und ein kleiner
Transformer generalisiert früher. Daran hängen zwei Vermutungen, die hier geprüft werden:

* **H1.** Jedes normal trainierte Netz hat so etwas im Kleinen selbst entwickelt; man muss es nur finden.
* **H2.** Ein SAE könnte beim Rechnen mitlaufen und in Echtzeit in den Zustand zurückschreiben, vor allem im untersten
  Block. Das stört das Netz lokal. Fängt das Netz solche Störungen von sich aus ab, vor allem wenn sie groß werden?

Was das Paper selbst dazu sagt (nachgelesen; wörtliche Stellen in `ufo_plan_review.json`): Der UFO ist ein zweites Netz, das
nur beim Training wirkt, über den Gradienten. Beim Rechnen ist er nicht da. Für normales Training erwartet das Paper
keine solche Ordnung („we should not expect UFR to emerge from conventional training") und räumt nur teilweise
Regelmäßigkeit ein. SAEs grenzt es ab: Sie zerlegen Aktivierungen in wenige aktive Einheiten, ein UFO verlangt, dass
sie durch einen engen Flaschenhals passen. Über Störungen beim Rechnen steht dort nichts. H1 zerfällt deshalb in drei
Aussagen mit je eigener Messung:

* (a) Ergebnis: Die Aktivierungen eines normal trainierten Netzes passen durch einen engen Flaschenhals (7.9, 7.10).
* (b) Trainingsverlauf: Der gewöhnliche Trainingsschritt senkt den UFO-Verlust von selbst. **Nicht gemessen.**
* (c) Rechenweg: Etwas im Netz entfernt, was nicht in den Zustand gehört (7.3 bis 7.7). Das ist nicht der UFO des Papers,
  aber die Lesart, auf die H2 baut.

Der Neuron Atlas von Goodfire wurde nicht ausgewertet (die Nutzungsbedingungen der Seite schließen automatischen Zugriff
aus). Die Neuronen-Seite stützt sich auf die eigenen Messungen aus Abschnitt 6.

### 7.1 Aufbau

Modell wie zuvor (Qwen3-1.7B, Chat-Modell, CPU). 16 Texte zu höchstens 64 Token (15 mit 64, einer mit 51): 8
Wikitext-Absätze und 8 Chats (Frage und Antwort eines feinjustierten Modells aus DSSE). Gestört werden die Positionen ab
der fünften, ohne die Token der Chat-Vorlage (902 Positionen); die Ausgabe wird von der neunten bis zur vorletzten
Position bewertet (822 Positionen). Zum Zustand nach einem Block wird ein Vektor addiert, Größe `f` mal Norm des Zustands. Richtungen:

| Kürzel | Richtung |
|---|---|
| Streuung | Rauschen mit der Kovarianz echter Zustände dieses Blocks: bevorzugt die Richtungen, in denen echte Zustände stark streuen |
| echt | Differenz zweier echter Zustände anderer Texte |
| isotrop | Zufallsrichtung ohne Vorzug |
| außen | Zufallsrichtung außerhalb des Datenraums (siehe 7.2), Mittelrichtung entfernt |

Die Statistik echter Zustände stammt aus 192 anderen Texten (13.551 Zustände je Block; zwei Drittel legen die
Hauptachsen fest, ein Drittel prüft sie). Zwei Messarten:

* **Lokal.** Nach *jedem* Block wird der Zustand auf „ungestört plus frischer Vektor" gesetzt und gelesen, was der nächste
  Block daraufhin schreibt. `q` = Anteil des Vektors, den der Block wieder herausschreibt (Projektion der Schreibdifferenz
  auf den Vektor, mit Vorzeichen). Das misst jeden Block für sich, ohne Fortpflanzung und ohne das Wachstum des Zustands.
* **Freier Lauf.** Der Vektor wird einmal zugesetzt (nach Block 0, 2, 8 oder 14), das Netz läuft weiter. Gelesen wird, wie
  sich die Vorhersage ändert (KL in nats zwischen ungestörter und gestörter Vorhersage des nächsten Tokens, Mittel über
  die bewerteten Positionen), dazu die Projektion der Abweichung auf den Vektor und ihre Größe.

Klammern sind 95-%-Intervalle aus 2.000 Ziehungen über die Texte. Die Norm des Zustands wächst von 16 (Block 0) über 28
(Block 2) und 179 (Block 14) auf 3.147 (Block 26). Eine *relative* Abweichung fällt deshalb auch ohne jede Dämpfung;
als Beleg zählen hier nur `q`, absolute Größen und die Ausgabe.

### 7.2 Auf laufendem Text ist der Zustandsraum nicht niedrigdimensional

| Zustand nach Block | 0 | 2 | 8 | 14 | 21 |
|---|---|---|---|---|---|
| Achsen für 98 % der Streuung zurückgehaltener Zustände (von 2.048) | 1.654 | 1.795 | 1.834 | 1.795 | 1.837 |
| Anteil in den 8 / 64 / 512 stärksten Achsen | 15 / 40 / 79 % | 14 / 39 / 75 % | 14 / 36 / 73 % | 14 / 37 / 73 % | 15 / 35 / 70 % |
| Abstand zweier echter Zustände / Norm (Median) | 1,21 | 1,16 | 1,15 | 1,10 | 1,06 |

Der „Datenraum" dieser Auswertung sind die Achsen der ersten Zeile; „außen" ist der Rest (210 bis 390 Richtungen mit
zusammen 2 % der Streuung). Diese Grenze gilt je Block: Ein Vektor, der nach Block 2 außerhalb liegt, liegt nach den
Achsen von Block 4 schon zu 78 % im Datenraum. Eine Störung der Größe 1 ist etwa der Abstand zum Zustand eines anderen
Tokens. Die Hälfte der Streuung liegt in weniger als 256 Achsen; die Zahl der ersten Zeile hängt an der 98-%-Grenze und
ist eine obere Schätzung. Gemessen ist hier der Zustand über alle Token eines Textes. Das Maß des Papers (7.9) gilt
den MLP-Aktivierungen an einer festen Token-Rolle und ist davon unabhängig.

### 7.3 Die Blöcke 1 bis 16 schreiben gegen das an, was zugesetzt wird, und deutlich weniger entlang der Achsen, in denen echte Zustände am stärksten streuen

Lokale Messung, `f` = 0,3. Anteil `q` des Vektors, den der nächste Block herausschreibt, gemittelt über die Blöcke der Spalte:

| Richtung | Block 1–4 | 5–10 | 11–16 | 17–22 | 23–26 |
|---|---|---|---|---|---|
| Streuung | 0,111 [0,108; 0,114] | 0,130 [0,127; 0,135] | 0,049 [0,046; 0,053] | 0,002 | 0,007 |
| echt | 0,111 | 0,124 | 0,040 | −0,001 | 0,007 |
| isotrop | 0,196 | 0,206 | 0,101 | 0,031 | 0,015 |
| beliebige Richtung im Datenraum (aus „isotrop" und „außen" berechnet) | 0,192 | 0,206 | 0,093 | 0,029 | 0,014 |
| außen | 0,213 [0,209; 0,217] | 0,209 [0,205; 0,213] | 0,152 [0,149; 0,155] | 0,043 | 0,018 |
| außen minus Streuung | +0,102 [0,101; 0,103] | +0,078 [0,077; 0,080] | +0,102 [0,100; 0,105] | +0,042 | +0,012 |
| außen minus beliebig im Datenraum | +0,021 | +0,003 | +0,059 | +0,014 | +0,004 |

Dasselbe für das MLP allein, exakt gerechnet für kleine Vektoren (7.4), nach Bändern der Hauptachsen echter Zustände
(stärkste Achsen zuerst):

| Achsen | 1–8 | 9–64 | 65–256 | 257–1.024 | 1.025 bis Rand des Datenraums | außerhalb |
|---|---|---|---|---|---|---|
| Block 1–4 | 0,045 | 0,025 | 0,121 | 0,190 | 0,199 | 0,194 |
| Block 5–9 | 0,077 | 0,067 | 0,130 | 0,232 | 0,235 | 0,216 |
| Block 10–16 | 0,076 | 0,007 | 0,003 | 0,103 | 0,141 | 0,155 |

* Das Anschreiben gegen den Vektor („Löschung") sitzt in den Blöcken 1 bis 16. Danach fällt es für schwach streuende
  Richtungen auf 0,05 bis 0,07 (Blöcke 17–19) und 0,02 bis 0,03 (Blöcke 20–22); entlang der Streuung ist es ab Block 17
  null.
* Es trifft fast jede Richtung gleich, mit rund 0,2 je Block in den Blöcken 1–9 und 0,10 bis 0,16 in den Blöcken 10–16.
  Deutlich weniger gelöscht wird nur entlang der rund 250 Achsen, in denen echte Zustände am stärksten streuen (0,02
  bis 0,13; ab Block 10 teils gar nicht). Streng abgestuft ist das nicht: Die acht stärksten Achsen werden etwas mehr
  gelöscht als die Achsen 9 bis 64. Ob eine Richtung
  im Datenraum liegt oder außerhalb, macht in den Blöcken 1–10 kaum einen Unterschied (+0,02 und +0,00) und in den
  Blöcken 11–16 einen mäßigen (+0,06). „Echt" ist kein zweiter Beleg neben „Streuung": Beide haben dieselbe Verteilung
  über die Achsen.
* Es kommt fast ganz aus den MLP-Schichten. Der Anteil der Attention liegt bei höchstens 0,025, solange die Vektoren an
  verschiedenen Positionen unabhängig sind (siehe 7.7).
* Was der Block sonst noch schreibt, ist größer: Der Teil seiner Reaktion quer zum Vektor hat 0,73 bis 0,78 der
  Vektorlänge (Blöcke 1–16). Die Abweichung als Ganze schrumpft deshalb in keinem Block; einen Block nach dem Zusetzen
  ist sie 1,07- bis 1,28-mal so lang wie der Vektor. Zurückgeholt wird der Zustand also nicht.

### 7.4 Woher die Löschung kommt

Für einen kleinen Vektor lässt sich exakt ausrechnen, was jedes MLP-Neuron zur Löschung beiträgt (`ufo_lin.py`, ein
ungestörter Durchlauf, 16 Texte). Die Formel trifft den gemessenen MLP-Anteil Block für Block: Blöcke 1–4 „außen" 0,194
gegen 0,189 gemessen, „Streuung" 0,103 gegen 0,103; über alle 27 Blöcke Korrelation 0,996 bis 0,9996, größte Abweichung
0,011 (außen) und 0,024 (Streuung).

| Richtung „außen" | Block 1–4 | 5–9 | 10–16 | 17–27 |
|---|---|---|---|---|
| Löschung durch das MLP (Formel) | 0,15 bis 0,24 | 0,20 bis 0,24 | 0,13 bis 0,18 | 0,01 bis 0,07 |
| dasselbe mit vertauschten Ausgangsgewichten | 0,000 | 0,000 | 0,000 | 0,000 |
| davon über den Up-Zweig (Rest: Gate-Zweig) | 0,15 bis 0,24 | 0,19 bis 0,22 | 0,13 bis 0,17 | 0,02 bis 0,07 |
| davon von Neuronen mit negativem Gate-Eingang | 0,16 bis 0,25 | 0,19 bis 0,24 | 0,00 bis 0,12 | etwa 0 |
| davon von Neuronen mit positivem Gate-Eingang | −0,01 | −0,01 bis +0,02 | 0,05 bis 0,13 | 0,02 bis 0,07 |
| Anteil der Neuronen mit negativem Gate-Eingang | 95 bis 98 % | 84 bis 90 % | 57 bis 77 % | 68 bis 89 % |
| mittlerer Kosinus zwischen Up-Gewicht und Ausgangsgewicht eines Neurons | +0,16 bis +0,34 | +0,21 bis +0,33 | −0,06 bis +0,12 | −0,31 bis −0,04 |
| Neuronen, die die Hälfte der Löschung tragen (von 6.144) | 792 bis 1.836 | 1.216 bis 1.626 | 341 bis 830 | 19 bis 277 |

Feiner aufgeteilt (Richtung „außen", Mittel über die Blöcke der Spalte):

| | Block 1–4 | 5–9 | 10–16 |
|---|---|---|---|
| Up-Zweig, Gate-Eingang negativ | 0,197 | 0,214 | 0,049 |
| Up-Zweig, Gate-Eingang positiv | −0,010 | −0,008 | +0,098 |
| Gate-Zweig (beide Vorzeichen zusammen) | 0,007 | 0,010 | 0,008 |
| Gate-Eingang unter −3 / von −3 bis −1 / von −1 bis 0 | 0,063 / 0,117 / 0,024 | 0,012 / 0,152 / 0,055 | 0,002 / 0,038 / 0,012 |
| Anteil der Neuronen in diesen drei Bereichen | 56 / 30 / 10 % | 7 / 47 / 32 % | 3 / 30 / 33 % |
| mittlerer Gate-Wert `silu` bei negativem / positivem Eingang | −0,13 / +0,39 | −0,21 / +0,40 | −0,20 / +0,71 |
| Anteil der Löschung aus der stärkeren Hälfte der Aktivierungen | 58 bis 78 % | 55 bis 57 % | 62 bis 87 % |

* **Blöcke 1–9.** Ein Neuron schreibt `silu(Gate-Eingang) × Up-Eingang × Ausgangsgewicht`. Bei negativem Gate-Eingang ist
  `silu` klein und negativ (zwischen −0,28 und 0). Das Neuron gibt sein Up-Signal dann mit kleinem negativem Faktor
  weiter. Weil das Ausgangsgewicht im Mittel in die Richtung zeigt, aus der das Neuron über den Up-Zweig liest, schreibt
  es ein wenig *gegen* das, was es liest. Das ist ein fester linearer Anteil des MLP; er wirkt auf den ungestörten
  Zustand genauso wie auf einen zugesetzten Vektor. Neuronen mit positivem Gate-Eingang verstärken je Neuron ähnlich stark; netto
  bleibt Löschung, weil fast alle Gates negativ sind. Die Löschung ist breit verteilt: Die Hälfte tragen 800 bis 1.800
  Neuronen je Block, und fast alle Beiträge haben dasselbe Vorzeichen (Summe der Beträge 0,19 bis 0,31 gegen Netto 0,15
  bis 0,24). Das stärkste Viertel der Aktivierungen trägt 27 bis 53 % davon.
* **Blöcke 10–16.** Hier übernehmen zunehmend Neuronen mit positivem Gate-Eingang, deren Ausgangsgewicht gegen ihre
  Leserichtung zeigt (ab Block 12 der größere Teil), und das stärkste Viertel trägt 36 bis 73 %. Für die Richtung
  „Streuung" hängt die Bilanz dort an wenigen hundert Neuronen oder weniger; das MLP von Block 16 verstärkt solche
  Vektoren sogar (−0,066).
* Die Löschung hängt ganz an der Paarung von Lese- und Schreibrichtung desselben Neurons: Mit vertauschten
  Ausgangsgewichten ist sie null (größter Betrag über alle Blöcke 0,005).
* **Sie ist gelernt.** Im untrainierten Netz gleicher Bauart ist `q` in allen Blöcken und für alle Richtungen 0,000
  (Intervalle innerhalb ±0,001), der mittlere Kosinus 0,000, und ein zugesetzter Vektor bleibt vollständig stehen.
* **Kein abtrennbarer Schutz.** Schreiben in den Blöcken 1–9 die Neuronen mit negativem Gate-Eingang nichts mehr, ist
  die Löschung dort weg (`q` außen −0,03 statt 0,22, Streuung −0,02 statt 0,13), aber auch das Modell: Der Verlust auf
  ungestörtem Text steigt von 3,49 auf 13,55. Die Löschung läuft über dieselben Neuronen wie das gewöhnliche Rechnen.
* **Nur die stärkere Hälfte der Aktivierungen behalten** (wie in 6.1, dort +0,02 nats Verlust): Der gemessene Anteil `q`
  bleibt fast gleich (Blöcke 1–4: außen 0,205 statt 0,213, Streuung 0,106 statt 0,111). Die Zuschreibung erklärt das
  nicht: Nach ihr trägt die schwächere Hälfte im vollen Modell 22 bis 45 % der MLP-Löschung (Blöcke 1–9), im
  beschnittenen Modell fehlen aber nur 5 bis 6 %. Woher der Rest kommt, ist nicht gemessen; in Frage kommt der harte
  Schnitt selbst, der auf jeden Vektor mit dem Ein- und Ausschalten von Neuronen antwortet. Das beschnittene Modell
  reagiert empfindlicher auf Vektoren nach Block 2 (KL 0,100 statt 0,068 und 0,124 statt 0,092). Über die schwachen
  Aktivierungen sagt dieser Lauf deshalb nichts Sicheres.

### 7.5 Bleibt ein Vektor von außen nach Block 2 stehen, verdoppelt sich der Schaden

Freier Lauf, `f` = 0,3. In Variante A wird die Projektion auf den zugesetzten Vektor festgehalten: Aus jedem späteren
Schreibvorgang (Attention und MLP aller folgenden Blöcke) wird der Anteil entlang des Vektors entfernt, alles andere
reagiert wie zuvor. In Variante B bleibt nur dieser Anteil, alles andere schreibt wie im ungestörten Lauf.

| Vektor | KL normal | KL, Projektion festgehalten (A) | A / normal | KL, nur Anteil entlang des Vektors (B) |
|---|---|---|---|---|
| außen, nach Block 2 | 0,071 [0,055; 0,089] | 0,143 [0,111; 0,176] | 2,01 [1,70; 2,42] | 0,000 |
| Streuung, nach Block 2 | 0,111 [0,085; 0,139] | 0,143 [0,109; 0,180] | 1,29 [1,22; 1,36] | 0,000 |
| außen, nach Block 14 | 0,055 [0,045; 0,067] | 0,074 [0,061; 0,089] | 1,33 [1,30; 1,38] | 0,000 |
| Streuung, nach Block 14 | 0,083 [0,066; 0,104] | 0,075 [0,059; 0,094] | 0,90 [0,88; 0,92] | 0,001 |

* Für „außen" ist das, was A entfernt, Löschung. Ohne sie verdoppelt sich der Schaden eines Vektors, der nach Block 2
  zugesetzt wird, und steigt um ein Drittel bei Block 14 (jeweils in 16 von 16 Texten).
* Für „Streuung" entfernt A zweierlei: die Löschung in den ersten Blöcken und das spätere Wiederaufbauen entlang des
  Vektors (die Projektion fällt bis Block 8 auf 0,59 und steigt bis Block 25 auf 2,6). Nach Block 2 überwiegt die
  Löschung (Schaden ohne sie das 1,29-Fache), nach Block 14 nicht (0,90).
* Mit festgehaltener Projektion kosten „Streuung" und „außen" im Mittel gleich viel (Verhältnis 1,00 [0,74; 1,37] nach
  Block 2, 1,02 [0,83; 1,22] nach Block 14). Dass ein Vektor von außen sonst billiger ist, liegt an der stärkeren Löschung.
* A ändert die Größe der Abweichung am Ende (außen, Block 2: das 36,5- gegen das 52,7-Fache der Vektorlänge), nicht ihren
  Schaden je Größe: Ein Vektor, der stehen bleibt, wird von jedem weiteren Block wieder gelesen. Die Löschung verkürzt
  diese Zeit. In Vektorgröße umgerechnet ist der Spielraum klein: Für denselben Schaden darf ein Vektor von außen nach
  Block 2 mit Löschung 1,42-mal so groß sein wie ohne.
* Ob die Löschung *für* diese Robustheit gelernt wurde oder ein Nebenergebnis des gewöhnlichen Rechnens ist, zeigen
  diese Läufe nicht. Für das Zweite spricht 7.4: Die Neuronen, die sie tragen, lassen sich nicht stumm schalten, ohne
  das Modell zu zerstören.
* Der Vektor selbst ist am Ausgang unsichtbar (B und ein unberührt mitlaufender Vektor: KL unter 0,002). Den Schaden trägt
  allein der Teil der Reaktion, der quer zum Vektor liegt.

### 7.6 Keine Bremse für große Störungen

| | `f` = 0,03 | 0,1 | 0,3 | 1,0 |
|---|---|---|---|---|
| KL, Streuung, nach Block 2 | 0,0010 | 0,0118 | 0,111 | 1,75 |
| KL, außen, nach Block 2 | 0,0009 | 0,0081 | 0,071 | 0,86 |
| KL, Streuung, nach Block 14 | 0,0008 | 0,0088 | 0,083 | 1,96 |
| KL, außen, nach Block 14 | 0,0005 | 0,0059 | 0,055 | 0,74 |
| Projektion nach einem Block (Streuung / außen, Block 2) | 0,88 / 0,79 | 0,88 / 0,79 | 0,88 / 0,80 | 0,89 / 0,82 |

* Geprüft sind einzelne Störungen von 0,03 bis 1 Zustandsnorm. Der gelöschte Anteil wächst nicht mit der Größe. Lokal gemessen, Blöcke 1–16: `q(0,3)/q(0,03)` = 0,97 (Streuung) und
  0,945 (außen), `q(1)/q(0,03)` = 0,75 und 0,59. Die Normalisierung vor jedem Block allein ergäbe 0,958 und 0,707. Nach
  Blöcken für `f` = 1: Blöcke 1–4 0,90 und 0,87, Blöcke 5–10 0,73 und 0,50, Blöcke 11–16 0,60 und 0,48. In keinem der
  Blöcke 1 bis 16 steigt der Anteil um mehr als ein Zehntel; in jeder Blockgruppe fällt er. (Der kleine Anteil der
  Attention verdoppelt sich für „Streuung" mit der Größe, von 0,008 auf 0,016 und von 0,013 auf 0,029; der des MLP fällt
  auf 0,81 / 0,56 und 0,87 / 0,44 seines Werts. Nur in den Blöcken 26 und 27 steigt `q` auf niedrigem Niveau.)
* Der Schaden wächst zwischen 0,03 und 0,3 quadratisch (Steigung im doppelt logarithmischen Maß 1,85 bis 2,05, lineare
  Reaktion ergäbe 2) und zwischen 0,3 und 1 schneller (2,07 bis 2,62; für „außen" nach Block 2 nicht gesichert: 2,07
  [1,88; 2,27]). Es gibt weder eine Schonzone für kleine noch einen Mechanismus, der bei großen Störungen einsetzt.
* Kontrolle: Wird der Zustand nur gestreckt (Faktor 1,41, keine neue Richtung), ist KL 0,089 (Block 2) und 0,044 (Block 14).
* **Zuversicht.** Für große Vektoren der Art „Streuung" (`f` = 1) aus den Blöcken 0 bis 8 steigt die Entropie der
  Vorhersage stärker als der Verlust (Block 2: Verlust +1,16, Entropie +1,84). Bei Block 0 und 2 entsteht das fast ganz
  in der letzten Normalisierung (mit dem Maßstab des ungestörten Laufs: +0,38 statt +1,84); bei Block 14 senkt der
  Maßstab die Entropie nicht (+1,55 statt +1,32). Für „außen" steigen beide etwa gleich, bei Block 8 der Verlust dreimal so stark wie die Entropie
  (+0,97 gegen +0,31). Eine allgemeine Regel „bei Störung wird das Netz passend unsicher" gibt es also nicht.
* Wird das MLP von Block 2 abgeschaltet, steigt der Verlust um 7,6 und die Entropie *fällt* um 0,78: Das Netz bleibt
  sicher und liegt falsch. Die MLPs von Block 8 und 14 abzuschalten wirkt dagegen wie mäßiges Rauschen (KL 0,23 und 0,12).

### 7.7 Was aus dem Vektor wird, und wo er am wenigsten schadet

Freier Lauf, `f` = 0,3. „Projektion" ist der Anteil der gesamten Abweichung entlang des Vektors; bis etwa acht Blöcke nach
dem Zusetzen ist das der Rest des Vektors, später auch das, was spätere Blöcke in dieser Richtung schreiben.

| Vektor zugesetzt nach Block | 0 | 2 | 8 | 14 |
|---|---|---|---|---|
| außen: Projektion nach 1 / 4 Blöcken / am Ende | 0,82 / 0,36 / 0,04 | 0,80 / 0,32 / 0,05 | 0,78 / 0,42 / 0,23 | 0,88 / 0,64 / 0,52 |
| Streuung: Projektion nach 1 / 4 Blöcken / am Ende | 0,97 / 0,73 / 1,73 | 0,88 / 0,65 / 1,30 | 0,86 / 0,91 / 1,09 | 0,95 / 1,10 / 0,98 |
| Abweichung am Ende / Vektorlänge (Streuung) | 61 | 44 | 17 | 7 |
| relative Abweichung am Ende / am Anfang (Streuung / außen) | 0,39 / 0,33 | 0,48 / 0,39 | 0,44 / 0,37 | 0,50 / 0,39 |
| KL Streuung | 0,078 [0,060; 0,096] | 0,111 [0,085; 0,139] | 0,080 [0,064; 0,099] | 0,083 [0,066; 0,104] |
| KL außen | 0,053 [0,042; 0,065] | 0,071 [0,055; 0,089] | 0,062 [0,046; 0,081] | 0,055 [0,045; 0,067] |

* Ein Vektor von außen, der nach Block 0 oder 2 zugesetzt wird, ist nach vier Blöcken zu zwei Dritteln gelöscht und am
  Ende nicht mehr nachweisbar (0,04 [−0,06; 0,14] und 0,05 [−0,01; 0,11]). Nach Block 8 bleiben am Ende 0,23, nach Block
  14 0,52. Ein Vektor entlang der Streuung wird erst gedämpft und in der oberen Netzhälfte wieder aufgebaut.
* Die Abweichung wächst absolut um das 5- bis 61-Fache, aber langsamer als der Zustand selbst: Relativ bleibt ein
  Drittel bis die Hälfte.
* Bei gleicher relativer Größe ist der Schaden nach Block 0, 8 und 14 ähnlich und nach Block 2 etwa 1,2- bis 1,4-mal so
  groß (für „außen" gegenüber Block 8 nicht gesichert). Die untersten Blöcke sind nicht duldsamer als die mittleren.
* Gegenüber einem gleich großen Zufallsvektor, der erst nach dem letzten Block zugesetzt wird, kostet die Abweichung
  am Ende auf den Wikitext-Texten etwa gleich viel (das 0,95- bis 1,35-Fache) und auf den Chats das 2,3- bis 3,7-Fache
  (Mittel über die Positionen, `f` = 0,3; für `f` = 1 über alle Texte das 1,3- bis 2,9-Fache).
* **Die Löschung ist lokal, der Schaden nicht.** Ein Vektor an nur einer Position wird genauso schnell gelöscht wie bei
  Vektoren an allen Positionen. Die Vorhersage ändert er an der Position selbst im Mittel um 0,07 bis 0,18 (Median 0,02
  bis 0,03) und an jeder späteren Position um 0,001 bis 0,008. Nach Block 2 zugesetzt, fällt in der Summe im Mittel mehr
  auf die späteren Positionen (0,34 und 0,12) als auf die eigene (0,18 und 0,07). Das gilt in 14 und 11 von 16 Texten;
  das Verhältnis der Mittel ist unsicher (1,9 [0,4; 3,9] und 1,6 [0,8; 3,0]), weil je Text nur eine Position gestört
  wird. Nach Block 14 fällt weniger auf die späteren Positionen (0,04 gegen 0,11 und 0,07).
* Derselbe Vektor an allen Positionen kostet 1,3- bis 2,3-mal so viel wie unabhängige Vektoren gleicher Größe. Nach
  Block 14 schreibt die Attention dabei entlang des gemeinsamen Vektors weiter (in den nächsten vier Blöcken 0,13 bis
  0,20 der Vektorlänge, bei unabhängigen Vektoren 0,01 bis 0,03). Nach Block 2 zeigt sich das entlang des Vektors
  nicht; dort ist der Teil der Reaktion quer zum Vektor größer. Zahlen für unabhängiges Rauschen unterschätzen jedenfalls
  eine systematische Störung, wie sie ein SAE-Fehler oder ein gezieltes Zurückschreiben wäre.
* Sicherheit der Zahlen: `q` ist stabil (eine zweite Ziehung der Richtungen weicht um 0,002 ab). Jede KL-Zahl hängt an
  einer Ziehung und wenigen Positionen (über alle 822 bewerteten Positionen trägt das oberste Zehntel 52 bis 61 % der
  Summe; zwei Ziehungen unterscheiden sich um bis zu 22 %). Belastbar sind gepaarte Verhältnisse innerhalb einer Ziehung und Faktoren um zwei.

### 7.8 Ein SAE im Durchlauf

SAEs: Qwen-Scope (TopK, 32.768 Einheiten, 50 aktiv, für jeden Block vorhanden, aber auf dem *Basismodell* trainiert), dazu
für Block 21 der SAE aus Abschnitt 4, der auf dem Chat-Modell trainiert ist. 12 der 16 Texte. `e` ist der Teil des
Zustands, den der SAE nicht rekonstruiert. Alles, was hier zu den Blöcken 0 bis 14 steht, gilt für SAEs des Basismodells
auf dem Chat-Modell, nicht für „einen SAE" allgemein.

| SAE nach Block | 0 | 2 | 8 | 14 | 21 | 21 (Chat-SAE) |
|---|---|---|---|---|---|---|
| erklärte Varianz auf diesen Texten | 0,88 | 0,83 | 0,70 | 0,63 | 0,76 | 0,83 |
| Länge von `e` / Norm des Zustands | 0,24 | 0,32 | 0,45 | 0,46 | 0,37 | 0,32 |
| Anteil von `e` außerhalb des Datenraums (Zustände: 2 %, Zufallsvektor: 10 bis 19 %) | 10 % | 8 % | 6 % | 5 % | 6 % | 6 % |
| **KL, Zustand durch Rekonstruktion ersetzt** | 0,066 [0,042; 0,092] | 0,142 [0,098; 0,187] | 0,221 [0,181; 0,261] | 0,301 [0,237; 0,371] | 0,299 [0,222; 0,381] | 0,203 [0,160; 0,246] |
| gleiches oberstes Token wie ungestört | 0,89 | 0,86 | 0,80 | 0,75 | 0,78 | 0,81 |
| KL im Verhältnis zu gleich großem Rauschen „Streuung" | 0,65 [0,42; 1,01] | 0,94 [0,68; 1,23] | 1,15 [0,98; 1,36] | 1,37 [1,19; 1,53] | 1,26 [1,02; 1,53] | 1,28 [0,95; 1,64] |
| … zu gleich großem Rauschen „außen" | 0,94 | 1,09 | 1,49 | 1,79 | 2,02 | 2,03 |
| KL, fremder Fehlervektor gleicher Größe abgezogen | 0,072 | 0,108 | 0,131 | 0,156 | 0,174 | 0,121 |
| Projektion der Änderung nach 4 Blöcken (fremder Fehler; außen / Streuung) | 0,48 (0,48; 0,36 / 0,74) | 0,42 (0,43; 0,32 / 0,64) | 0,70 (0,68; 0,43 / 0,92) | 1,06 (0,95; 0,65 / 1,11) | 1,05 | 1,04 |

* Von den fünf geprüften Stellen ist der unterste Block die billigste; dort kostet der SAE nicht mehr als gleich großes
  Rauschen. Billig ist er auch dort nicht: Das oberste Token ändert sich an 11 % der Positionen. Mit der Tiefe wird er
  teurer, vor allem weil der Rest `e` größer wird (KL je Quadrat der Restgröße: 1,1 bis 1,4 in den Blöcken 0 bis 14).
* Passung: Der Chat-SAE in Block 21 kostet das 0,68-Fache [0,62; 0,76] des Basis-SAE; je Größe des Rests sind beide
  gleich teuer. Wie viel vom Tiefenprofil an der fehlenden Passung der Basis-SAEs liegt, lässt sich ohne Chat-SAEs für
  die unteren Blöcke nicht sagen.
* In den Blöcken 0 und 2 gleicht das Netz das, was der SAE weglässt, so aus wie den Fehlervektor eines fremden Zustands:
  Nach vier Blöcken ist die Hälfte ausgeglichen, zwischen „außen" und „Streuung". Im Schaden gilt das nur für Block 0
  (das 0,92-Fache des fremden Fehlervektors [0,68; 1,16]). Schon in Block 2 kostet der eigene Rest das 1,3-Fache
  [1,1; 1,6], in Block 8 das 1,7-Fache [1,5; 2,0], in Block 14 das 1,9-Fache [1,7; 2,2]. Ab Block 14 wird er auch nicht
  mehr ausgeglichen (1,06, wie Rauschen „Streuung" mit 1,11) und bleibt besser stehen als ein fremder Fehler (0,95). Der
  Rest ist also von Block 2 an zunehmend Inhalt, den das Netz braucht.
* Im ungestörten Lauf schreiben in den Blöcken 0 und 2 die nächsten vier Blöcke gegen den unerklärten Teil an (−0,09 /
  −0,39) und bauen den erklärten aus (+0,45 / +0,21); ab Block 8 gilt das nicht mehr (+0,02, Block 14: +0,53). Der kleine
  Teil des echten Zustands außerhalb des Datenraums wird in den Blöcken 0, 2 und 8 über vier Blöcke abgebaut (−0,40 /
  −0,42 / −0,32; Teil im Datenraum +0,36 / +0,11 / +0,38; Zufallsvektor 0,00), langsamer als ein fremder Vektor. Über
  alle späteren Blöcke zusammen ist die Bilanz für Block 2 und 8 positiv (+4,2 [0,5; 8,0] und +1,3 [0,2; 2,3]), für
  Block 0 nicht gesichert (−0,8 [−8,0; 5,9]); bei Block 2 kommt sie fast ganz aus den Blöcken 17 bis 27, in denen alle
  Schreibvorgänge mit dem Zustand gewachsen sind. In den Blöcken 14 und 21 wird er sofort aufgebaut (+0,19 / +0,22). Das
  sind Zusammenhänge im ungestörten Lauf, keine Reaktionen auf den Rest.
* Hauptachsen statt SAE kosten das 1,8- bis 3-Fache. Der Vergleich ist aber nicht fair eingestellt: Die Zahl der Achsen
  wurde an anderen Zuständen auf gleiche erklärte Varianz gesetzt, und auf diesen Texten erklären sie 6 bis 15
  Prozentpunkte weniger. Auf gleichen unerklärten Rest umgerechnet bleibt etwa das 1,2- bis 1,9-Fache (Schätzung), in den
  Blöcken 0 und 2 nicht sicher über 1.
* Die SAE-Rekonstruktion ist in den Blöcken 0 und 2 kaum leichter komprimierbar als der Zustand selbst (Anteil der
  Streuung in 128 Achsen an zurückgehaltenen Zuständen: 34 gegen 31 % und 29 gegen 27 %), in den Blöcken 8 und 14 etwas
  (32 gegen 26 % und 36 gegen 27 %; Block 21: 30 gegen 26 %, Chat-SAE 31 %); der Rest `e` liegt bei 12 bis 17 %. Niedrigdimensional ist keines davon. (Eine
  Aufteilung, zurückgehaltener Teil nur Chats.)

**Lässt sich der SAE als Filter gegen Rauschen nutzen?** Rauschen der Größe 0,3 wird im Block des SAE zugesetzt:

| SAE nach Block | 0 | 2 | 14 |
|---|---|---|---|
| Verschiebung der SAE-Ausgabe / Größe des Rauschens (Streuung / außen) | 0,38 / 0,39 | 0,42 / 0,39 | 0,58 / 0,51 |
| davon in Richtung des Rauschens | 0,06 / 0,04 | 0,08 / 0,04 | 0,14 / 0,03 |
| aktive Einheiten, die aktiv bleiben | 0,80 / 0,75 | 0,82 / 0,81 | 0,79 / 0,80 |
| KL, Rauschen allein | 0,081 / 0,060 | 0,091 / 0,087 | 0,085 / 0,068 |
| KL, SAE in Echtzeit: `SAE(h + d)` | 0,078 / 0,088 | 0,155 / 0,142 | 0,330 / 0,307 |
| zum Vergleich KL, SAE ohne Rauschen | 0,066 | 0,142 | 0,301 |
| nur Diagnose: KL von `SAE(h + d) + e`, mit `e` aus dem *ungestörten* Zustand | 0,016 / 0,016 | 0,025 / 0,020 | 0,041 / 0,030 |

* Der SAE lässt vom Rauschen selbst wenig durch (3 bis 14 % in dessen Richtung). Seine Ausgabe verschiebt sich aber um
  38 bis 58 % der Rauschgröße in andere Richtungen, und ein Fünftel bis ein Viertel der aktiven Einheiten wechselt.
* In Echtzeit hilft er nicht: `SAE(h + d)` kostet in Block 0 so viel wie das Rauschen allein (Streuung) oder mehr (außen:
  0,088 gegen 0,060) und ab Block 2 das 1,6- bis 4,5-Fache, weil sein eigener Rekonstruktionsfehler dazukommt. Lohnen
  würde er sich erst für Rauschen, das allein mehr kostet als er selbst: hochgerechnet ab etwa 0,3 bis 0,4 Zustandsnormen
  in Block 0, 0,44 in Block 2 und 0,8 in Block 14 (gerechnet wurde nur die Größe 0,3).
* Schreibt man nur die Differenz zurück und lässt den eigenen Rest stehen, ist das Rauschen wieder vollständig da (der
  Rest des gestörten Zustands enthält es). Die letzte Zeile zeigt, was übrig bliebe, wenn man den ungestörten Zustand
  kennte (20 bis 27 % des Rauschschadens in Block 0 und 2). Das ist eine Diagnose, kein Verfahren.
* Für das Zurückschreiben folgt daraus eine Bauregel: Ein SAE, der nur die *Differenz* zwischen geänderter und
  unveränderter Rekonstruktion addiert, kostet nichts, solange er nichts ändert. Sein eigener Fehler bleibt dann außen
  vor, und es zählt allein die gewollte Änderung. Die ist allerdings systematisch und wird von der Attention aufaddiert
  (7.7). Geprüft wurde ein solches Zurückschreiben hier nicht.

**Echte Schleife beim Schreiben.** 12 Chat-Fragen, das Modell schreibt 32 Token (jeweils das wahrscheinlichste), an
*jedem* Token wird gestört oder der SAE durchlaufen; jedes erzeugte Token geht zurück in die Eingabe. Die Token der
Chat-Vorlage bleiben wie in den anderen Läufen unberührt. Verglichen wird Token für Token mit dem ungestörten Modell,
das denselben bisherigen Text liest.

| Eingriff an jedem Token | KL je Token | Token 1–8 / 9–16 / 17–24 / 25–32 | gleiche Wahl wie das ungestörte Modell | Verlust des Textes unter dem ungestörten Modell |
|---|---|---|---|---|
| keiner | 0 | | 1,00 | 0,19 |
| Rauschen „Streuung" 0,3 nach Block 2 (3 Ziehungen) | 0,067 bis 0,103 | 0,08 / 0,11 / 0,10 / 0,07 | 0,90 bis 0,92 | 0,26 bis 0,32 |
| Rauschen „außen" 0,3 nach Block 2 (3 Ziehungen) | 0,051 bis 0,073 | 0,05 / 0,06 / 0,09 / 0,05 | 0,92 bis 0,93 | 0,24 bis 0,27 |
| Rauschen „Streuung" 1,0 nach Block 2 | 2,35 [1,89; 2,80] | 3,57 / 2,62 / 1,70 / 1,49 | 0,55 | 3,25 |
| Rauschen „Streuung" 0,3 nach Block 14 | 0,146 [0,107; 0,183] | 0,12 / 0,11 / 0,17 / 0,19 | 0,88 | 0,38 |
| SAE nach Block 0 | 0,039 [0,022; 0,058] | 0,07 / 0,03 / 0,04 / 0,02 | 0,95 | 0,22 |
| SAE nach Block 2 | 0,119 [0,069; 0,176] | 0,24 / 0,09 / 0,07 / 0,08 | 0,90 | 0,33 |
| SAE nach Block 14 | 0,511 [0,364; 0,675] | 0,35 / 0,57 / 0,60 / 0,52 | 0,78 | 0,84 |
| nur Diagnose: Rauschen 0,3, dann `SAE(h + d) + e` mit `e` aus dem ungestörten Zustand (Block 2) | 0,018 [0,010; 0,028] | 0,02 / 0,02 / 0,01 / 0,02 | 0,97 | 0,21 |

* Für Eingriffe nach Block 0 und 2 schaukelt sich die Abweichung über 32 Token nicht auf: Sie bleibt auf gleichem Niveau
  oder fällt (Steigung je Token für Rauschen 0,3 nach Block 2: zwischen −0,001 und +0,001, alle sechs Intervalle
  schließen null ein). Nach Block 14 ist das nicht gezeigt: Mit Rauschen 0,3 liegt die zweite Hälfte der Token beim
  1,55-Fachen der ersten [1,02; 2,30], mit dem SAE liegen die Token ab dem neunten beim 1,6-Fachen der ersten acht
  [0,9; 3,0]. Das ist ein Lauf mit 12 Fragen und fester Wahl des wahrscheinlichsten Tokens.
* Der Text weicht trotzdem früh vom ungestörten ab (Rauschen 0,3: nach 8 Token noch in 25 bis 58 % der Fälle gleich,
  nach 32 in 0 bis 17 %; SAE nach Block 0 oder 2: nach 32 Token in 1 von 12 Fällen gleich, nach Block 14 in keinem).
  Für das ungestörte Modell ist der abweichende Text bei Rauschen 0,3 und beim SAE in Block 0 und 2 fast so plausibel
  wie der eigene.
* Gemessen ist ein Lauf mit festen Störungen je Token. Eine Rückkopplung, bei der das Netz auf das Zurückgeschriebene
  des SAE wieder antwortet und der SAE darauf, wurde nicht gebaut.

### 7.9 Das Kriterium des Papers selbst: Passen die Aktivierungen durch einen engen Flaschenhals?

Das Paper misst Ordnung daran, wie gut ein Autoencoder mit zwei Zahlen in der Mitte die z-standardisierten Aktivierungen
einer MLP-Schicht wiedergibt (eine Zeile je Eingabe, eine Spalte je Neuron). Hier die entsprechende Messung am fertigen
Modell, ohne UFO (Autoencoder nachträglich und je Block angepasst, ein Startwert). Angepasst wird an zwei Dritteln der
Eingaben, gelesen am zurückgehaltenen Drittel.

**(a) 1.472 DSSE-Fragen, letztes Token der Vorlage.** Anteil der Streuung am zurückgehaltenen Drittel:

| Block | 0 | 2 | 5 | 8 | 14 | 20 | 26 |
|---|---|---|---|---|---|---|---|
| 2 Hauptachsen | 0,32 | 0,33 | 0,36 | 0,39 | 0,25 | 0,18 | 0,10 |
| 8 Hauptachsen | 0,55 | 0,58 | 0,61 | 0,62 | 0,49 | 0,36 | 0,22 |
| Autoencoder mit 2 Zahlen, 300 / 1.500 Schritte | 0,33 / 0,28 | 0,41 / 0,35 | 0,44 / 0,40 | 0,36 / 0,28 | 0,38 / 0,35 | 0,31 / 0,33 | 0,20 / 0,21 |
| die 18 Familienmittel allein | 0,30 | 0,33 | 0,42 | 0,46 | 0,38 | 0,29 | 0,18 |
| 2 Hauptachsen dessen, was nach Abzug der Familienmittel bleibt (Anteil am Ganzen) | 0,16 | 0,15 | 0,09 | 0,09 | 0,08 | 0,05 | 0,03 |

(Spalten durchgemischt, also ohne gemeinsames Muster: zwei Achsen 0,00, Autoencoder −0,03. Block 27 fehlt: Dort tragen
zehn Neuronen 86 % der Summe am zurückgehaltenen Drittel, zwei Achsen erklären 0,01 und mit bei ±5 gekappten z-Werten
0,13. In Block 26 hebt dasselbe Kappen den Wert von 0,10 auf 0,14.)

Das sieht nach starker Ordnung aus, beschreibt aber den Fragensatz:

* Die erste Achse ist in den Blöcken 0 bis 2 die Länge der Frage (R² 0,83 bis 0,87) und ab Block 3 die Fragenfamilie
  (R² 0,76 bis 0,97). Länge und Familie hängen zusammen (die Familien erklären 70 % der Streuung der logarithmierten
  Länge).
* Die 18 Familienmittel allein erklären ab Block 2 am zurückgehaltenen Drittel mindestens so viel wie die zwei
  Hauptachsen. Was nach ihrem Abzug bleibt, passt deutlich schlechter durch zwei Achsen (letzte Zeile).
* Der Autoencoder ist nur in Block 0 und 8 nicht besser als zwei gerade Achsen. In den Blöcken 2 und 5 liegt er 0,02 bis
  0,08 darüber, in den Blöcken 14, 20 und 26 um 0,10 bis 0,16. Dort erreicht er etwa den Anteil der Familienmittel: Zwei
  Zahlen reichen, um 18 Familien zu unterscheiden. Mit längerem Training wird er in fünf von sieben Blöcken schlechter.

**(b) Ein Bereich, der die Voraussetzung des Papers erfüllt:** zwei freie Größen in festem Format. 2.025 Aufgaben
`a+b=` (a, b = 10, 12, …, 98) hinter zwei festen Beispielen, alle 24 Token lang; gelesen wird am `=`. Das Modell nennt die
erste Ziffer der Summe in 100 % der Fälle. Je Neuron wird zerlegt, wovon seine Aktivierung über das Raster abhängt
(Median über die Neuronen eines Blocks):

| Block | 0 | 4 | 8 | 14 | 20 | 22 | 24 | 26 |
|---|---|---|---|---|---|---|---|---|
| trainiert: Anteil, den a und b je für sich erklären (additiv) | 1,00 | 0,96 | 0,85 | 0,70 | 0,67 | 0,28 | 0,18 | 0,17 |
| trainiert: Anteil, den die Summe allein erklärt | 0,03 | 0,12 | 0,12 | 0,15 | 0,11 | 0,24 | 0,29 | 0,41 |
| trainiert: Summe über a und b hinaus | 0,00 | 0,01 | 0,03 | 0,05 | 0,06 | 0,19 | 0,25 | 0,34 |
| trainiert: Neuronen, die zu mehr als der Hälfte von der Summe abhängen (davon über a und b hinaus) | 0 % (0 %) | 2 % (0 %) | 3 % (0 %) | 3 % (0 %) | 1 % (0 %) | 4 % (1 %) | 8 % (3 %) | 26 % (11 %) |
| untrainiert: additiv | 0,95 | 0,79 | 0,73 | 0,68 | 0,65 | 0,64 | 0,63 | 0,62 |
| untrainiert: Summe allein | 0,09 | 0,09 | 0,09 | 0,09 | 0,09 | 0,09 | 0,09 | 0,09 |
| untrainiert: Summe über a und b hinaus | 0,01 | 0,03 | 0,04 | 0,04 | 0,04 | 0,05 | 0,05 | 0,05 |

Die Anteile überlappen: Eine Aktivierung, die gleichmäßig mit a und mit b wächst, zählt bei „additiv" und bei „Summe
allein" voll. Deshalb steht daneben, was die Summe über a und b hinaus erklärt. Die 0,09 des untrainierten Netzes sind
der Boden des Maßes: 0,04 bekommen 89 Gruppenmittel bei 2.025 Aufgaben immer, der Rest entsteht, weil ein untrainiertes
Netz beide Summanden gleich behandelt.

Flaschenhals auf dem Raster, gelesen an zurückgehaltenen Rasterzellen (2 / 8 / 32 Hauptachsen / Autoencoder mit 2 Zahlen,
600 Schritte, ein Startwert):

| Block | 4 | 14 | 22 |
|---|---|---|---|
| trainiert | 0,29 / 0,62 / 0,93 / 0,52 | 0,26 / 0,57 / 0,80 / 0,36 | 0,08 / 0,21 / 0,44 / 0,21 |
| untrainiert | 0,29 / 0,62 / 0,85 / 0,51 | 0,24 / 0,50 / 0,76 / 0,45 | 0,23 / 0,47 / 0,73 / 0,44 |

* **Das Maß des Papers trennt hier nicht zwischen gelernt und ungelernt.** Das untrainierte Netz passt mindestens so gut
  durch den Flaschenhals mit zwei Zahlen (0,44 bis 0,51) wie das trainierte (0,21 bis 0,52). Der Wert spiegelt also die
  Eingabe und die Bauart, nichts Gelerntes. Wo das trainierte Netz der Summe folgt (Block 22), ist er sogar niedriger.
* Bis Block 8 antworten die Neuronen fast nur auf die beiden Summanden einzeln (0,85 bis 1,00), von Block 9 bis 20
  überwiegend (0,67 bis 0,75). Das tut auch das untrainierte Netz; es folgt aus der Bauart. Über a und b hinaus erklärt
  die Summe bis Block 20 kaum etwas (0,00 bis 0,06; untrainiert 0,01 bis 0,05).
* In den letzten sechs Blöcken folgt das trainierte Netz dem Ergebnis: Die Summe erklärt im Median 24 bis 41 % der
  Streuung eines Neurons, über a und b hinaus 19 bis 34 % (untrainiert 9 % und 5 %). In Block 26 hängt ein Viertel der
  Neuronen überwiegend von der Summe ab, ein Neuntel auch über a und b hinaus (untrainiert keines). Nach Streuung
  gewichtet sind es 54 bis 77 %, denn 1 % der Neuronen trägt dort 64 bis 81 % der Streuung.
* Ein Teil davon ist das Token, das das Modell als Nächstes ausgibt: Die erste Ziffer der Summe allein erklärt in Block
  26 drei Viertel dessen, was die Summe erklärt (gewichtet 0,55 von 0,74), in den Blöcken 22 bis 25 knapp die Hälfte.
* Ob dieser Code regelmäßig ist, misst die Zerlegung nicht; durch den Flaschenhals passt er schlechter als die
  Aktivierungen des untrainierten Netzes. Gezeigt ist eine Ordnung nach dem Ergebnis in einer Aufgabe und einem Format,
  kein Ordner.

### 7.10 Kleinversuch: Division modulo 97 (Aufbau des Papers, aber Gewichtsabnahme 0,3 statt 0,03; je ein Lauf)

2-Schichten-Transformer wie im Paper (Breite 128, MLP 512, Layer-Norm nach jedem Teilblock), die Hälfte der 9.312
Aufgaben zum Training. Drei Läufe mit demselben Startwert: nur Gewichtsabnahme; mit UFO (Autoencoder mit 2 Zahlen auf
den MLP-Aktivierungen am `=`, nach dem Paper); Kontrolle mit einmal zufällig vertauschten Antworten, die auswendig
lernen, aber nichts verallgemeinern kann. Wegen der zehnfachen Gewichtsabnahme und nur eines Laufs je Arm ist kein
Vergleich mit den Beschleunigungen des Papers möglich. Der Lauf mit UFO erreicht 99 % auf ungesehenen Aufgaben nach
9.000 Schritten, der ohne nach 12.000.

Vom Paper nicht festgelegt und hier gewählt: Bias in allen linearen Schichten, eine eigene Ausgabeschicht mit Bias, die
Summe der beiden Kreuzentropien als Verlust, ReLU und Standard-Startwerte im Autoencoder des UFO. Das Training endet
8.000 Schritte nach dem ersten Erreichen von 99 % (20.000, 17.000 und 20.000 Schritte); die Zeitpläne des UFO sind die
des Papers und decken hier einen rund achtmal größeren Teil des Laufs ab. Die Kontrolle steht nicht im Paper. Sie
verliert zwischen Schritt 7.000 und 13.000 mehrfach einen Teil des Gelernten (bis hinab zu 24 % richtig) und endet mit
doppelt so großer Gewichtsnorm (83 gegen 42).

| | nur Gewichtsabnahme | | | | | mit UFO | | Antworten vertauscht | |
|---|---|---|---|---|---|---|---|---|---|
| Schritt | 5.000 | 7.500 | 10.000 | 12.000 | 15.000 | 5.000 | 9.000 | 5.000 | 15.000 |
| richtig auf ungesehenen Aufgaben | 0,01 | 0,03 | 0,28 | 1,00 | 1,00 | 0,04 | 1,00 | 0,01 | 0,01 |
| 1. MLP-Schicht, 8 Achsen, z-Werte bei ±5 gekappt, ungesehene Aufgaben | 0,39 | 0,37 | 0,44 | 0,63 | 0,81 | 0,96 | 0,99 | 0,44 | 0,42 |
| dasselbe ohne Kappen: Anpassungshälfte / andere Trainingshälfte / ungesehen | 0,27 / 0,05 / 0,03 | 0,27 / 0,10 / 0,11 | 0,34 / 0,19 / 0,24 | 0,53 / 0,46 / 0,46 | 0,80 / 0,80 / 0,78 | 0,92 / 0,68 / 0,74 | 0,99 / 0,99 / 0,99 | 0,33 / 0,17 / 0,12 | 0,27 / 0,19 / 0,17 |
| 2. MLP-Schicht, 2 Achsen, ungesehene Aufgaben | 0,08 | 0,07 | 0,07 | 0,11 | 0,13 | 0,48 | 0,93 | 0,08 | 0,07 |
| Neuronen der 1. Schicht, die am `=` überhaupt streuen (von 512) | 336 | 308 | 249 | 171 | 173 | 121 | 115 | 179 | 148 |
| Streuung des Zustands am `=` / seine Norm | 0,29 | 0,30 | 0,27 | 0,22 | 0,25 | 0,26 | 0,11 | 0,18 | 0,23 |
| gleiche Antwort trotz Vektor der Größe 1 am `=` nach Block 0 | 0,41 | 0,52 | 0,69 | 0,97 | 0,98 | 0,56 | 1,00 | 0,56 | 0,83 |
| dasselbe, wenn Block 1 nicht reagieren darf | 0,28 | 0,34 | 0,46 | 0,97 | 1,00 | 0,66 | 1,00 | 0,30 | 0,48 |
| Löschung durch das MLP von Block 1 (`q`) | 0,75 | 0,79 | 0,82 | 0,81 | 0,79 | 0,48 | 0,20 | 0,75 | 0,81 |

(Vektor: Rauschen mit der Kovarianz der echten Zustände, Größe = mittlerer Abstand echter Zustände von ihrem Mittel;
gelesen an den Trainingsaufgaben. Flaschenhals der ersten Schicht: Achsen und z-Werte an einer Hälfte der
Trainingsaufgaben angepasst, `ufo_grok_split.py`; zweite Schicht: an allen Trainingsaufgaben angepasst.)

* **In der ersten MLP-Schicht wird der Code beim Verallgemeinern komprimierbarer.** Mit gekappten z-Werten erklären acht
  Achsen 0,30 bis 0,39, solange das Netz auswendig lernt (Schritt 1.000 bis 7.500), 0,44 bei Schritt 10.000, 0,63 beim
  Übergang und 0,71 bis 0,81 danach. In der Kontrolle, die nicht verallgemeinern kann, sind es 0,30 bis 0,49; das
  untrainierte Netz liegt bei 0,41.
* **Ein Unterschied zwischen gesehenen und ungesehenen Aufgaben ist das nicht.** Ohne Kappen passt der Flaschenhals auf
  die Aufgaben, an denen er angepasst wurde, mäßig und auf alle anderen schlecht, auch auf auswendig gelernte
  (Schritt 5.000: 0,27 / 0,05 / 0,03). Ursache sind Neuronen, die nur auf eine Handvoll Aufgaben ansprechen und nach dem
  z-Standardisieren außerhalb der Anpassung riesige Werte bekommen (zehn Neuronen tragen dort 53 bis 99 % der Summe,
  solange das Netz auswendig lernt). Mit gekappten z-Werten liegen alle drei Gruppen zu jeder Zeit gleichauf (Schritt
  5.000: 0,39 / 0,38 / 0,39; Schritt 15.000: 0,82 / 0,81 / 0,81).
* Die Dimension fällt schon vor dem Sprung der Genauigkeit (Achsen für 99 % der Streuung des Zustands am `=`: 72, 53, 44,
  39, 29, 11 bei Schritt 1.000, 5.000, 7.500, 10.000, 12.000, 20.000). Bis Schritt 10.000 fällt sie in der Kontrolle
  genauso (70, 52, 43, 36); erst nach dem Übergang trennen sich die Läufe (20.000: 11 gegen 27). Der Abfall vor dem
  Sprung zeigt also keine Verallgemeinerung an.
* **Von dem, was ein UFO erzwingt, bleibt das normale Netz weit entfernt.** Zweite MLP-Schicht, zwei Achsen, ungesehene
  Aufgaben: 0,10 bis 0,13 nach dem Übergang, mit UFO 0,93 bis 1,00. In der ersten Schicht leert der UFO-Lauf die
  geregelte Stelle: Ab Schritt 15.000 streut dort kein Neuron mehr, und der Zustand am `=` ist für alle Aufgaben gleich;
  schon ab Schritt 9.000 streut er nur noch um höchstens 11 % seiner Norm. Die Zeilen zur Störung sagen für den UFO-Lauf
  ab Schritt 9.000 deshalb nichts mehr aus.
* **Unempfindlichkeit gegen Störungen ist kein Zeichen eines inneren UFO.** Schon das untrainierte Netz behält die Antwort
  in 83 % der Fälle. In der Kontrolle, die nichts verallgemeinert, steigt der Wert von 0,43 (Schritt 1.000) auf 0,83 bis
  0,87 (15.000 bis 17.500) und liegt bei 20.000 bei 0,76. Im verallgemeinernden Netz liegt er nach dem Übergang bei 0,93
  bis 0,98 und fällt bei Schritt 20.000 auf 0,73. Gestört wird nur der Zustand am `=` nach Block 0; wie stark die Antwort
  an dieser einen Stelle hängt, ändert sich im Lauf des Trainings.
* Ein klarer Unterschied zwischen dem UFO-Lauf und dem normalen Lauf zeigt sich dabei nicht: Vor dem Übergang liegt der
  UFO-Lauf bei gleicher Genauigkeit auf ungesehenen Aufgaben einmal darunter (0,36 gegen 0,41 bei 1,5 %) und einmal
  darüber (0,56 gegen 0,52 bei 3 bis 4 %). Nach dem Übergang lässt sich an dieser Stelle nicht mehr vergleichen. Ein
  Lauf je Arm.
* **Die Löschung entsteht mit dem Auswendiglernen, nicht mit dem Verallgemeinern.** Das MLP von Block 1 schreibt 0,4 bis
  0,8 eines Vektors entlang der Streuung zurück, sobald das Netz die Trainingsaufgaben kann (ab Schritt 1.000), in
  allen drei Läufen (gemessen an der Abweichung am Eingang des MLP, nach der Layer-Norm). Im untrainierten Netz ist sie
  null. Anders als in Qwen ist sie entlang der Hauptachsen echter Zustände am stärksten (0,67 bis 0,86) und quer dazu
  am schwächsten (0,11 bis 0,21). Ob es derselbe Vorgang ist, bleibt offen.
* Ob die Löschung im Kleinversuch die Antwort schützt, zeigt er nicht. Mit der ganzen Reaktion von Block 1 bleibt die
  Antwort im normalen Lauf nur von Schritt 3.000 bis 10.000 öfter erhalten als ohne (Schritt 5.000: 0,41 gegen 0,28).
  Kurz nach dem Auswendiglernen bleibt sie seltener erhalten (Schritt 1.000: 0,40 gegen 0,53), ab Schritt 12.500
  ebenfalls (0,93 bis 0,98 gegen 0,99 bis 1,00; Schritt 20.000: 0,73 gegen 1,00). In der Kontrolle bleibt sie ab Schritt
  3.000 öfter erhalten (15.000: 0,83 gegen 0,48), im UFO-Lauf vor dem Übergang durchweg seltener (5.000: 0,56 gegen 0,66).

### 7.11 Was folgt daraus

**Zu H1 (jedes normal trainierte Netz hat seinen eigenen kleinen UFO).** Nicht bestätigt. Im Sinn des Papers lässt sich
das am fertigen Netz auch nicht bestätigen, weil der UFO nur beim Training wirkt.

* Dafür, in schwacher Form: Das Training hat Qwen etwas gegeben, was das untrainierte Netz nicht hat. Die MLPs der
  Blöcke 1 bis 16 schreiben gegen jeden zugesetzten Vektor an, mit rund 20 % je Block in den Blöcken 1 bis 9 (danach
  10 bis 16 %) und mit 2 bis 13 % entlang der rund 250 Richtungen, in denen echte Zustände am stärksten streuen. Bleibt ein Vektor stattdessen stehen, verdoppelt
  sich sein Schaden (gezeigt für einen Vektor außerhalb des Datenraums nach Block 2).
* Dagegen, das als Ordner oder Schutzmechanismus zu lesen: Es ist ein fester linearer Anteil ohne Schwelle. Er holt den
  Zustand nicht zurück (nach jedem Block ist die Abweichung größer als der zugesetzte Vektor). Er unterscheidet nicht
  zwischen „im Datenraum" und „außerhalb", nur zwischen stark und schwach streuenden Richtungen. Die Neuronen, die ihn
  tragen, sind dieselben, über die das gewöhnliche Rechnen läuft. Im Kleinversuch schreibt das MLP ebenfalls gegen
  zugesetzte Vektoren an, sobald das Netz auswendig kann, also ohne Verallgemeinerung; dort allerdings am stärksten
  entlang der Hauptachsen, umgekehrt wie in Qwen.
* Zum Ergebnis (a): Im Kleinversuch wird der Code der ersten MLP-Schicht beim Verallgemeinern ohne UFO komprimierbarer
  (acht Achsen 0,63 bis 0,81 statt 0,30 bis 0,39; Kontrolle bis 0,49), bleibt aber weit unter dem, was ein UFO erzwingt.
  In Qwen folgt auf dem Zahlenraster in den letzten sechs Blöcken ein Teil der Neuronen der Summe. Das Flaschenhals-Maß
  des Papers zeigt das nicht an: Es beschreibt auf den DSSE-Fragen den Fragensatz und ist auf dem Raster für das
  untrainierte Netz mindestens so hoch.
* Dämpfung taugt nicht als Nachweis eines UFO: Der eine UFO-Lauf ist vor dem Übergang nicht erkennbar duldsamer als der
  normale, danach lässt sich an der gestörten Stelle nicht mehr vergleichen, und ein Netz, das nichts verallgemeinert,
  wird mit der Trainingszeit ebenfalls duldsam.
* Nicht geprüft: ob der gewöhnliche Trainingsschritt den UFO-Verlust senkt (b), und ein zweites Modell. Über „jedes"
  Netz lässt sich nichts sagen.

**Zu H2 (SAE in Echtzeit, vor allem im untersten Block; Bremse bei großen Störungen).**

* Dafür: Von fünf geprüften Stellen kostet der SAE im untersten Block am wenigsten (KL 0,07, beim Schreiben 0,04 je
  Token; oberstes Token an 89 % der Positionen unverändert), und sein Fehler schadet dort nicht mehr als gleich großes Rauschen. Ein SAE, der nur die
  Differenz zwischen geänderter und unveränderter Rekonstruktion addiert, bringt keinen eigenen Fehler ein. In der
  Schleife (12 Fragen, 32 erzeugte Token) wuchs der Abstand zum ungestörten Modell je Token bei Eingriffen nach Block 0
  und 2 nicht; nach Block 14 ist ein Anstieg nicht ausgeschlossen.
* Dagegen: Eine Bremse für große Störungen gibt es nicht (einzelne Störungen von 0,03 bis 1 Zustandsnorm: Der gelöschte
  Anteil fällt mit der Größe, der Schaden wächst quadratisch oder schneller). Die Löschung halbiert den Schaden nur für
  schwach streuende Richtungen nach Block 2; für inhaltsartige Vektoren in der Netzmitte tut sie nichts, und der Rest
  eines SAE ist von Block 2 an zunehmend inhaltsartig. Derselbe Vektor an allen Positionen kostet das 1,3- bis 2,3-Fache von unabhängigem
  Rauschen, und ein SAE-Fehler ist von dieser systematischen Art. Als Filter schadet der SAE mehr als das Rauschen, das
  er entfernen soll (Block 2 und 14), oder gleicht sich gerade aus (Block 0, eine von zwei Rauscharten). Gleich großes
  Rauschen wird in den untersten Blöcken nicht besser vertragen. In der Schleife wich der Text mit einem SAE im
  Durchlauf in 11 von 12 Fällen vom ungestörten ab (Block 0 und 2), in Block 14 in allen.
* Nicht geprüft: das Zurückschreiben selbst (der SAE ändert eine Einheit und schreibt sie zurück); SAEs, die für die
  unteren Blöcke auf diesem Chat-Modell trainiert sind; Texte über 64 Token; Sampling; mehr als eine SAE-Stelle;
  Störungen über einer Zustandsnorm.

**Wie sicher?** Ein Modell, 16 Texte (SAE: 12), eine bis zwei Ziehungen der Richtungen. Fest sind die Löschungswerte
`q` und die Formel dahinter. Jede KL-Zahl kann sich mit einer anderen Ziehung um bis zu ein Fünftel verschieben. Der
Kleinversuch hat einen Lauf je Arm und die zehnfache Gewichtsabnahme des Papers. Der Neuron Atlas wurde nicht ausgewertet.

## Grenzen

* Basismodell, Fragenfamilien, erstes Antwort-Token, ein Eingriffsort (Block 21, letztes Token).
* Acht von Hand gewählte, teils überlappende Paare und eine Aufteilung; die Ausgabe-Messung trägt in vier davon.
  Die Spannen über Paare stehen in den Tabellen und in `mim_react2_t0.log`; verallgemeinernde Intervalle gibt es nicht.
* Die Endzustands-Messung enthält zu etwa 40 % den eingesetzten Vektor selbst (Durchleitung über die
  Residualverbindung, Spalte `dR2_dir`). Tragend sind die Ausgabe-Messungen.
* SAE und Lens sind am letzten Token der Vorlage schwächer als auf gewöhnlichem Text (SAE 56 % statt 81 % erklärte
  Varianz, mit 65.536 Einheiten 66 %).
* Blinder Konzept-Test: 4 Einheiten je Kontrollart, 16 Fragen je Einheit.
* Abschnitt 7: ein Modell, 16 Texte zu 64 Token (SAE: 12 Texte), 12 Fragen mit 32 erzeugten Token, je Text eine Ziehung
  der Richtungen. Die Qwen-Scope-SAEs sind auf dem Basismodell trainiert. Der Kleinversuch hat einen Lauf je Arm und die
  zehnfache Gewichtsabnahme des Papers. Der Hauptlauf (`ufo_damp2.npz`) entstand vor späteren Ergänzungen am Skript
  (Optionen `TOPK` und `NONEG`; die Richtung „echt" zieht nie zweimal denselben Zustand). Sie berühren die berichteten
  Größen nicht, bis auf 13 von 432 Zellen der Richtung „echt", die als fehlend aus dem Mittel fielen (0,111 würde 0,109).
  Die Läufe „nur Gewichtsabnahme" und „mit UFO" des Kleinversuchs entstanden mit einer früheren Fassung von `ufo_grok.py`,
  die nicht mehr vorliegt; die gespeicherten Gewichte passen zur jetzigen Fassung und geben die aufgezeichneten
  Genauigkeiten wieder.
* Von den Auflagen der Plankritik zum Kleinversuch und zum Flaschenhals-Maß sind offen: mehrere Startwerte und
  Trainingsanteile; der Vergleich bei gleichem Validierungsverlust (nicht aufgezeichnet; verglichen wurde bei gleicher
  Genauigkeit); die Größe, bei der die Hälfte der Antworten kippt; Störungen an den Positionen der Operanden; die genaue
  Jacobi-Matrix; die absolute Größe der Vektoren. Für die DSSE-Fragen fehlen das untrainierte Netz, ein Gauß-Ersatz mit
  gleicher Kovarianz, eine gedrehte Basis und eine Aufteilung nach ganzen Fragenfamilien, für den Autoencoder mehrere
  Startwerte und frühes Abbrechen. Zu den Auflagen für den Störungslauf fehlen die genaue Jacobi-Matrix und Läufe mit
  mehr als zwei Ziehungen der Richtungen.

## Nächster Schritt

Derselbe Versuch mit feinjustierten Modellen: Klasse = Modell (feinjustiert gegen Basis) auf denselben Fragen, je
Datensatz eine Wiederholung, Richtung = mittlere Verschiebung. Dann lässt sich prüfen, ob eine Verschiebung, die ein
Fine-Tune im Zustand bewirkt, im Basismodell das Verhalten auslöst, und ob eine Trennlinie zwischen den Modellen mehr
anzeigt, als das Netz daraus macht. Das braucht eine GPU.

Zu Abschnitt 7: (1) die Löschung im feinjustierten Modell mit der im Basismodell vergleichen (verschiebt LoRA die
Löschung, und entlang welcher Richtungen?); (2) den Trainingsverlauf prüfen, also ob der gewöhnliche Trainingsschritt
den UFO-Verlust senkt, im Kleinversuch mit mehreren Startwerten und der Gewichtsabnahme des Papers; (3) eine echte
Rückkopplung zwischen SAE und Netz, mit einem SAE, der für die unteren Blöcke auf dem Chat-Modell trainiert ist.

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
| `neu_sparsity*.log/.json/.csv`, `neu_wiring.*`, `neu_static.*`, `neu_static_units.csv`, `neu_extract.log` | Neuronen: Sparsamkeit, Verdrahtung, Trennen und Zuschreiben (6.1–6.3) |
| `neu_patch_react.*`, `mim_patch_t0n.log`, `neu_tail*.log`, `neu_tail_react.*` | Neuronen: Wirkung von Gruppen, Reaktion hinter der Eingriffsstelle (6.4, 6.5) |
| `abs_score_neurons.*`, `abs_induce_n*.json`, `abs_predict_n*.json`, `abs_key_neurons.json` | blinder Konzept-Test für Neuronen (6.6) |
| `review_ju_groupings.json`, `review_mimicry_code.json`, `review_mimicry_method.json`, `review_neurons.json` | Prüfberichte |
| `infografik.html`, `infografik.pdf` | Infografik zu den Abschnitten 1 bis 6; erzeugt mit `analysis/ju_space/infografik_build.py` |
| `ufo_plan_review.json` | Literatur und zwei Plankritiken vor den Läufen von Abschnitt 7 |
| `ufo_pool.log`, `ufo_damp2.npz`, `ufo_damp2_meta.json`, `ufo_damp2.log`, `ufo_report.txt`, `ufo_report.json` | Statistik echter Zustände, Störungslauf und seine Tabellen (7.2, 7.3, 7.5–7.7) |
| `ufo_lin.*`, `ufo_lin_E.npz`, `ufo_damp2_small.*`, `ufo_damp2_top0.5.*`, `ufo_report_small.txt`, `ufo_report_top0.5.txt` | Zuschreibung der Löschung auf Neuronen; nur die stärkere Hälfte der Aktivierungen (7.4) |
| `ufo_pool_randinit.log`, `ufo_damp2_randinit.*`, `ufo_report_randinit.txt`, `ufo_lin_randinit.*` | dasselbe am untrainierten Netz (7.4) |
| `ufo_damp2_noneg.*`, `ufo_report_noneg.txt` | Neuronen mit negativem Gate-Eingang in den Blöcken 1–9 stumm (7.4) |
| `ufo_sae2.npz`, `ufo_sae2_meta.json`, `ufo_sae2.log`, `ufo_sae2_report.txt`, `ufo_sae2_report.json`, `ufo_loop.npz`, `ufo_loop_meta.json`, `ufo_loop.log` | SAE im Durchlauf und Schleife beim Schreiben (7.8) |
| `ufo_compress*.json/.log`, `ufo_grid*.json/.log` | Flaschenhals-Maß auf DSSE-Fragen und auf dem Zahlenraster, trainiert und untrainiert (7.9) |
| `ufo_grok_{base,ufo,perm}_log.json`, `ufo_grok_{base,ufo,perm}_probe2.json`, `ufo_grok_{base,ufo,perm}_split.json`, `ufo_grok_report.txt` | Kleinversuch (7.10) |
| `review_ufo.json` | Prüfberichte zu Abschnitt 7 |

Nicht im Repository: Zustände und SAE-Aktivierungen (zusammen etwa 500 MB), für Abschnitt 7 die Statistik echter
Zustände (`ufo_pool.pt`, 860 MB), die Aktivierungen auf dem Zahlenraster (700 MB) und die Gewichte des Kleinversuchs;
sie entstehen beim Nachrechnen neu.

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
# Abschnitt 6 (Neuronen); immer nur einen Lauf mit geladenem Modell zur selben Zeit, das Speicherlimit liegt bei 14 GB
python $S/neu_sparsity.py $R $O 20 96          # braucht $O/corpus_wikitext.json: die ersten Absätze mit über 600 Zeichen aus wikitext-103-raw (test), als JSON-Liste
python $S/neu_sparsity.py $R $O 20 96 "top_0.078,top_0.25,rot_0.5,rot_0.75,rot_0.9,rand_0.5,rand_0.25" extra
python $S/neu_extract.py $R $O && python $S/neu_static.py $R $O t0 && python $S/neu_wiring.py $R $O
python $S/neu_tail.py $R $O t0 && python $S/neu_tail_react.py $R $O t0
python $S/neu_patch_sets.py $R $O t0 && python $S/mim_patch.py $R $O t0n && python $S/neu_patch_react.py $R $O t0
python $S/abs_prepare_neurons.py $R $O && python $S/abs_score_neurons.py $O   # dazwischen: Benenner und blinder Sortierer
python $S/infografik_build.py results/ju_space/infografik.html --standalone   # Infografik; die Zahlen stehen im Skript
# Abschnitt 7; zusätzlich nötig: $R/qscope_dl/layer{0,2,8,14,21}.sae.pt (Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50 auf Hugging Face)
U=<root>/ufo_out; mkdir -p $U                                            # Eingaben wie oben aus $O (corpus_wikitext.json, axes_pairs.json, neu_act.npy, eval_items.json)
python $S/ufo_pool.py $R $U                                              # Statistik echter Zustände aller 28 Blöcke, 5 min
python $S/ufo_damp2.py $R $U && python $S/ufo_report.py $U               # Störungslauf (70 Varianten je Text), 15 min
CONDSET=small TOPK=0.5 python $S/ufo_damp2.py $R $U && CONDSET=small TAG=_small python $S/ufo_damp2.py $R $U
CONDSET=small NONEG=1-9 python $S/ufo_damp2.py $R $U                       # Neuronen mit negativem Gate-Eingang in Block 1-9 stumm
python $S/ufo_report.py $U _top0.5 && python $S/ufo_report.py $U _small && python $S/ufo_report.py $U _noneg
python $S/ufo_lin.py $R $U                                               # Zuschreibung auf Neuronen, 2 min
python $S/ufo_sae2.py $R $U && python $S/ufo_sae2_report.py $U && NEW=32 python $S/ufo_loop.py $R $U
RANDINIT=1 python $S/ufo_pool.py $R $U && RANDINIT=1 CONDSET=small python $S/ufo_damp2.py $R $U && RANDINIT=1 python $S/ufo_lin.py $R $U
python $S/ufo_report.py $U _randinit
python $S/ufo_compress.py $U $O/neu_act.npy && python $S/ufo_compress_pc.py $R $U
python $S/ufo_grid.py $R $U && RANDINIT=1 python $S/ufo_grid.py $R $U    # FROMFILE=1: nur die Zerlegung aus den gespeicherten Aktivierungen
ONLY=4,14,22 BLOCKS=4,14,22 STEPS=600 python $S/ufo_compress.py $U $U/ufo_grid_act.npy _grid          # ebenso mit ufo_grid_act_randinit.npy _grid_randinit
for m in base ufo; do python $S/ufo_grok.py $U $m 0 70000 0.3; done; python $S/ufo_grok.py $U perm 0 20000 0.3   # Kleinversuch, je 15 bis 30 min
for m in base ufo perm; do python $S/ufo_grok_probe2.py $U/grok_${m}_s0_wd0.3; python $S/ufo_grok_split.py $U/grok_${m}_s0_wd0.3; done
python $S/ufo_grok_report.py $U
```
