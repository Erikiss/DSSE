# J-Space, U-Space, Trennlinien und deutbare Einheiten auf DSSE (CPU-Lauf vom 10. November 2026)

Basismodell Qwen3-1.7B, nicht feinjustiert. Skripte: `analysis/ju_space/`. Alle Zahlen stammen aus einem Lauf auf
4 CPU-Kernen. Vier unabhängige Prüfdurchgänge haben Code und Deutung zu den Abschnitten 1, 4, 6.1–6.3 und 6.5
nachgerechnet (`review_*.json`), ihre Korrekturen sind eingearbeitet. Für die Abschnitte 2, 3, 5, 6.4 und 6.6 liegt
kein eigener Prüfbericht vor. Die Infografik `infografik.html` (auch als `infografik.pdf`) fasst alles auf einer Seite
zusammen; für sie sind alle Zahlen noch einmal gegen dieses README und die Ergebnisdateien abgeglichen worden.

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
   aber eine echte Einheit nicht von einer beliebigen Richtung unterscheiden: Regeln für Zufallsrichtungen treffen neue
   Beispiele zu 72 %, für SAE-Einheiten zu 77 %.
6. **Der fertige SAE ist an dieser Stelle ein schwaches Werkzeug.** Am letzten Token der Vorlage erklärt er 56 % der
   Varianz (auf Inhalts-Token 81 %). Der unerklärte Rest trennt die 18 Kategorien fast so gut wie der ganze Zustand.
7. **Neuronen als Einheiten** (Nachtrag, Abschnitt 6). Die Aktivierungen sind in der Neuronen-Basis konzentriert, und
   300 bis 1.000 gezielt gewählte von 135.000 Neuronen bewegen die Ausgabe wie die volle Kategoriendifferenz (zufällige:
   gar nicht). Kompakter bleiben SAE-Einheiten (100 leisten dasselbe wie 1.000 Neuronen), und benennen lassen sich
   zufällige Mischungen von Neuronen genauso gut wie Neuronen.

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

In Klammern steht hier und in 6.6 das 95-%-Intervall des Mittels (Bootstrap über die Einheiten); einzelne SAE-Einheiten
liegen zwischen 50 % und 100 %. Kontrollrichtungen zusammen 92 von 128 (72 %). Der Abstand der SAE-Einheiten dazu ist
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
| `neu_sparsity*.log/.json/.csv`, `neu_wiring.*`, `neu_static.*`, `neu_static_units.csv`, `neu_extract.log` | Neuronen: Sparsamkeit, Verdrahtung, Trennen und Zuschreiben (6.1–6.3) |
| `neu_patch_react.*`, `mim_patch_t0n.log`, `neu_tail*.log`, `neu_tail_react.*` | Neuronen: Wirkung von Gruppen, Reaktion hinter der Eingriffsstelle (6.4, 6.5) |
| `abs_score_neurons.*`, `abs_induce_n*.json`, `abs_predict_n*.json`, `abs_key_neurons.json` | blinder Konzept-Test für Neuronen (6.6) |
| `review_ju_groupings.json`, `review_mimicry_code.json`, `review_mimicry_method.json`, `review_neurons.json` | Prüfberichte |
| `infografik.html`, `infografik.pdf` | Infografik zu allen Abschnitten; erzeugt mit `analysis/ju_space/infografik_build.py` |

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
# Abschnitt 6 (Neuronen); immer nur einen Lauf mit geladenem Modell zur selben Zeit, das Speicherlimit liegt bei 14 GB
python $S/neu_sparsity.py $R $O 20 96          # braucht $O/corpus_wikitext.json: die ersten Absätze mit über 600 Zeichen aus wikitext-103-raw (test), als JSON-Liste
python $S/neu_sparsity.py $R $O 20 96 "top_0.078,top_0.25,rot_0.5,rot_0.75,rot_0.9,rand_0.5,rand_0.25" extra
python $S/neu_extract.py $R $O && python $S/neu_static.py $R $O t0 && python $S/neu_wiring.py $R $O
python $S/neu_tail.py $R $O t0 && python $S/neu_tail_react.py $R $O t0
python $S/neu_patch_sets.py $R $O t0 && python $S/mim_patch.py $R $O t0n && python $S/neu_patch_react.py $R $O t0
python $S/abs_prepare_neurons.py $R $O && python $S/abs_score_neurons.py $O   # dazwischen: Benenner und blinder Sortierer
python $S/infografik_build.py results/ju_space/infografik.html --standalone   # Infografik; die Zahlen stehen im Skript
```
