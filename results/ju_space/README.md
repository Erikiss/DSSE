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

Abschnitt 8 ist ein Nachtrag vom 11. November zum KV-Cache, zu Ungrokking und zu SwiGLU, wieder am Kleinversuch. Der
Plan wurde kritisiert und die Literatur gesichtet (`kv_plan_review.json`, `kv_literature.json`), bevor ein Lauf der
neuen Startwerte ausgewertet war; die vier Grundläufe und ein Vorlauf zum Ungrokking liefen da schon, und die Läufe aus
7.10 waren schon vermessen. Der Plan wurde danach umgebaut. Zwei Prüfer haben die Messzahlen an 19 Ständen von neun
Läufen (die vier Grundläufe, Startwert 0 von drei Armen, zwei Läufe aus 7.10) und die Pfad-Analyse mit eigenem Code
nachgerechnet (alle verglichenen Werte stimmen). Ein dritter hat die Deutung des Entwurfs angegriffen, zwei weitere
haben seine Zahlen gegen die Dateien abgeglichen; danach wurden viele Aussagen zurückgenommen oder enger gefasst. Die
danach neu geschriebenen Teile (Trainingsarme und „Wohin die Rechnung wandert" in 8.5, SwiGLU-Arm und Gegenprobe in 8.7,
Folgerungen in 8.8, Punkt 9 der Kurzfassung, Grenzen, Nächster Schritt, Nachrechnen) haben zwei letzte Prüfer gelesen:
einer hat die Zahlen gegen die Dateien abgeglichen, der andere die Deutung geprüft, S für die 16 Läufe mit Störung mit
eigenem Code nachgerechnet und die Befehle unter „Nachrechnen" gegen die Skripte gelesen. Ihre Einwände sind
eingearbeitet (zwei davon betrafen falsche Aussagen in Kurzfassung und Kopf). Was dabei geändert wurde, hat niemand mehr
gegengelesen. Alle sieben Berichte und eine Notiz zu diesen letzten Änderungen stehen in `review_kv.json`.

Abschnitt 9 ist ein Nachtrag vom 11. November zur Aktivierungsfunktion: ob sie Ungrokking bewirkt, ob ein Netz seine
eigenen Gates abbildet, wie quadratisch SwiGLU ist und was Halbieren und Dritteln damit zu tun haben, am Kleinversuch
und an Qwen3-1.7B. Plan und Plankritik entstanden, bevor ein Lauf ausgewertet war (`act_plan_review.json`); die
Literaturprüfung (`act_literature.json`) wurde erst fertig, als die ersten Läufe schon ausgewertet waren, und aus ihr
kam nachträglich der Verlust in float64. Die Entscheidungsregeln des Prüfers sind angewandt (für 9.2 in
`act_verdicts.json`, sonst im Text; die Abweichungen nennt Abschnitt 9), die Mathematik hat ein weiterer Prüfer mit
eigenem Code nachgerechnet (`act_math_review.json`). Den fertigen Entwurf haben drei Prüfer gelesen: Zwei haben die
Zahlen gegen die Dateien abgeglichen, einer hat die Deutung gegen die Regeln geprüft, und jeden ihrer Befunde hat ein
eigener Gegenprüfer nachgerechnet. 61 der 68 Befunde hielten ganz oder teilweise und sind eingearbeitet; unter anderem
ist die Folgerung zu V1 enger gefasst und der allmähliche Verlust vor den Einbrüchen ergänzt. Den überarbeiteten
Abschnitt haben zwei weitere Prüfer gelesen, wieder mit Gegenprüfung; 31 ihrer 33 Befunde hielten und sind
eingearbeitet. Der wichtigste: Der Tausch der Schaltmuster war nur bis Schritt 1.000 gerechnet; an allen gespeicherten
Ständen gerechnet, spricht er bei SwiGLU nach der Regel dagegen, statt unentschieden zu sein. Alle Berichte stehen in
`review_act.json`.

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
9. **KV-Cache, Ungrokking, SwiGLU** (Nachtrag, Abschnitt 8; Kleinversuch Division modulo 97, vier Startwerte,
   Gewichtsabnahme 0,3). Der Inhalt des Cache ordnet sich beim Verallgemeinern von selbst entlang der Gruppenstruktur der
   Aufgabe (Anteil der 5 stärksten Frequenzen in den Grundläufen 0,17 bis 0,32 beim Auswendiglernen, 0,59 bis 0,99
   danach; mit Störung im Training verallgemeinern Netze schon ab 0,36; ein Netz, das nur auswendig lernen kann, bleibt
   nahe am Zufallswert, höchstens 0,17). In den Grundläufen ändern fremde Schlüssel in Auswendig-Ständen 73 bis 91 % der
   Antworten, in verallgemeinernden 1 bis 33 %; die vorab gesetzten Schwellen dafür sind knapp verfehlt, und nötig ist
   die Abhängigkeit von den Schlüsseln fürs Auswendiglernen nicht. Unordnung im Cache beim Training (vertauschte
   Schlüssel, Rauschen) ließ das Netz in allen acht Läufen früher verallgemeinern als der Grundlauf desselben Startwerts
   (50 % auf ungesehenen Aufgaben nach dem 0,22- bis 0,56-Fachen der Schritte). Ein nach Länge abgeglichener
   Zufallsvektor an der lesenden Position (dem `=`) half ebenfalls, in sechs von acht Vergleichen weniger. Ob es auf den
   Cache ankommt, bleibt offen: Die vorab gesetzte Schwelle (fünf gepaarte Startwerte) war mit vier nicht erreichbar,
   und die Kontrolle blieb über den Lauf nicht gleich groß. Mit 5 % der Aufgaben geht die Verallgemeinerung verloren
   (Ungrokking), in Sprüngen bei Einbrüchen des Trainings, die auch der Lauf mit allen Aufgaben hat; nach der vorab
   gesetzten Regel zählt das deshalb als Instabilität des Aufbaus. 2 % seltene Aufgaben je Stapel erhalten die
   Verallgemeinerung großteils. Dass der Verlust in dem Teil des Durchlaufs sitzt, der den Cache schreibt, zeigt sich
   nicht, und die Verschiebung des Cache gleicht in ihrer groben Gestalt der eines Netzes, das einfach weitertrainiert.
   SwiGLU statt ReLU bringt hier keinen klaren Vorteil. Zur Frage, ob ein innen geordnetes Netz eher die eigenen
   Gewichte ändert, sagt nichts davon etwas.
10. **Aktivierungsfunktion, SwiGLU als quadratische Einheit, Dritteln** (Nachtrag, Abschnitt 9; Kleinversuch und
   Qwen3-1.7B). Ob Ungrokking eintritt, entscheidet die Art der Aktivierungsfunktion im Kleinversuch nicht: Mit ReLU,
   SwiGLU und einer rein bilinearen Einheit ohne Schwelle geht die Verallgemeinerung mit 5 % der Aufgaben in allen
   Netzen verloren (Endstand 0,02 bis 0,09 richtig), mit eingefrorenen ReLU-Masken ebenso (0,09 bis 0,11). Ob es
   überhaupt eine MLP-Nichtlinearität braucht, bleibt offen. Der größte Teil des Verlusts kommt mit Einbrüchen des
   Trainings, die sich mit keiner geprüften Einstellung (größeres ε, AMSGrad, kleinere Lernrate, Verlust in float64,
   kleine Stapel) abstellen ließen; ein kleinerer Teil geht schon vorher allmählich verloren. Ob Ungrokking die
   Einbrüche braucht, bleibt deshalb offen. Die eigenen Gate-Werte sind aus dem Endzustand kaum besser lesbar als
   zufällige Gates auf denselben Eingängen (bei SwiGLU und ReLU um weniger als 0,10, die Schwelle der vorab gesetzten
   Regel); ob ein Netz sich selbst abbildet, ist so nicht prüfbar. SwiGLU ist in Qwen3-1.7B nur in Block 0 ungefähr
   quadratisch, und das knapp; im Kleinversuch hängt die Rechnung im zweiten Block an einem quadratischen Glied, mit
   ReLU genauso. Ein Beitrag über kubische Polynome (8.7) fand in einem Netz, das x³ + a·x² + x + b lernt, Merkmale, die
   von x + c·a abhängen, mit c nahe Cardanos 1/3. Diese Richtung folgt der Zielfunktion und wandert mit der Verteilung
   der Eingaben: Bei doppelt so breitem Bereich von a bleibt keines von 15 Netzen bei 1/3; fünf Netze in der Größe des
   Beitrags wandern ebenfalls mit, liegen aber 0,02 bis 0,08 über der Richtung, in der sich die Zielfunktion am
   stärksten ändert. Für ein Produkt gibt es keine solche Richtung. Halbieren gehört rechnerisch zu Produkten aus zwei
   Faktoren, Dritteln zu Produkten aus drei; über trainierte Netze folgt daraus nichts Prüfbares.

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

## 8 Ordnung und Unordnung im KV-Cache, Ungrokking, SwiGLU (Kleinversuch, Nachtrag vom 11. November 2026)

Anlass ist eine Kette von Vermutungen im Anschluss an Abschnitt 7:

* **R1** Ein Netz, das innen im Sinn des UFO-Papers geordnet ist, neigt eher dazu, die eigenen Gewichte zu ändern.
* **K1** Auch im KV-Cache könnte es einen „UFO" geben: Ist der Cache geordnet, benutzt ihn das Modell mehr.
* **K2** Sehr schneller, starker Zugriff auf den Cache heißt meist Auswendiglernen (dazu soll es ein Paper geben).
* **K3** Also könnte man den Cache absichtlich ungeordneter machen („Anti-UFO").
* **K4** Ungrokking hängt an den selten vertretenen Trainingsbeispielen und in seiner Dynamik mehr am Cache als an den
  Gewichten.
* **K5** Die Unordnung im Cache ließe sich über Ungrokking steuern, vielleicht mit einem gelernten Modell der Geometrie
  zwischen Grokking und Ungrokking.

Geprüft wurde, was davon am Kleinversuch aus 7.10 messbar ist (Division modulo 97, dasselbe Netz, Gewichtsabnahme 0,3).
Ein Prüfer hat den Plan zerlegt und ein zweiter die Literatur gesichtet (`kv_plan_review.json`, `kv_literature.json`),
während die vier Grundläufe und ein Vorlauf zum Ungrokking schon liefen. Der Plan wurde danach umgebaut; die
Entscheidungsregeln in 8.4, 8.5 und 8.6 stammen aus dieser Kritik und standen fest, bevor ein Lauf der neuen Startwerte
ausgewertet war und bevor die Trainingsarme und die Ungrokking-Läufe begannen. Alles hier gilt für dieses eine kleine
Netz und eine Aufgabe. Über Qwen oder ein anderes Sprachmodell sagt es nichts.

### 8.1 Die Annahmen

Die Kette stützt sich auf vier Angaben, die hier zuerst geprüft werden: UFOs machten das Grokking zwölfmal schneller;
UFOs wirkten bisher auf die gewöhnlichen Gewichtsmatrizen; das Paper „Grokking Is Conditional and Fragile" verbinde
Grokking mit Ungrokking; und ein Paper habe gemessen, dass schneller, starker Zugriff auf den Cache Auswendiglernen
bedeutet. Der letzte Punkt der Liste nennt eine Arbeit, die an K3 angrenzt.

* **„12-mal schneller"** ist im UFO-Paper der beste von vier Fällen (Tabelle 1 dort, 8 Startwerte, Gewichtsabnahme
  0,03): 12,25-fach, wenn der UFO je Aufmerksamkeits-Verbindung der `=`-Zeile die Gewichte über alle Eingaben bekommt
  (8.000 statt 98.000 Schritte im Median), 7,54-fach, wenn er je Eingabe alle Verbindungen bekommt, 3,16- und 2,04-fach
  auf den MLP-Aktivierungen.
* **Ein UFO wirkt nicht auf Gewichtsmatrizen.** Er bekommt Aktivierungen oder Aufmerksamkeitsgewichte und wirkt nur über
  den Gradienten beim Training. Der stärkste Fall regelt, wie die `=`-Position die früheren Positionen liest, also das
  Lesemuster über dem Cache. Schlüssel und Werte selbst hat das Paper nicht geregelt. Mit einem festen Zufallsnetz als
  UFO an derselben Stelle blieb das Grokking in 7 von 8 Läufen innerhalb der 150.000 Schritte aus (der achte brauchte
  131.000 statt 98.000 im Median ohne UFO); ein „Anti-UFO" kommt im Paper nicht vor.
* **„Grokking Is Conditional and Fragile"** (Ootani, arXiv 2607.05104) handelt von der Abdeckung der Trainingsmenge, der
  Gewichtsabnahme und der Reihenfolge der Gleitkomma-Summation (1 statt 4 CPU-Threads kippt 49 von 300 Läufen gleichen
  Startwerts; die Quote bleibt mit 74 gegen 81 von 300 praktisch gleich). Das Wort Ungrokking kommt nicht vor.
  Ungrokking stammt von Varma et al. (arXiv 2309.02390): Weitertrainieren auf zu wenigen Beispielen. Dem Gedanken „hängt
  an den seltenen Beispielen" am nächsten ist Li und Sreedhar (arXiv 2606.26050): Ob eine Regel das Vortraining
  überlebt, entscheidet, wie oft der Korpus sie stützt.
* **Ein Paper, das „schneller Cache-Zugriff heißt Auswendiglernen" misst, wurde nicht gefunden.** Am nächsten liegen
  arXiv 2603.19664 (Lesen des Cache gegen Neuberechnen aus dem Zustand; eine Frage der Rechenzeit, nichts über
  Auswendiglernen) und die Arbeiten von Reddy und Mitautoren (arXiv 2412.00104, 2604.12151), in denen es um die
  Lerngeschwindigkeit zweier Teilschaltungen geht. In der Literatur zu „aus dem Kontext lernen" gegen „in den Gewichten
  lernen" gilt das Auswendiggelernte meist als die Lösung in den Gewichten und die Nutzung des Kontexts als ihr
  Gegenspieler (arXiv 2205.05055, nur als Zusammenfassung gelesen; 2311.08360). Es gibt aber auch auswendig lernende
  Lösungen, die den Kontext lesen (arXiv 2604.12151, 2412.01003). Kontextnutzung allein zeigt also weder Auswendiglernen
  noch dessen Gegenteil an.
* Einen Cache auf Ordnung zu trainieren gibt es schon: KV-CAT (arXiv 2605.05971) blendet beim Weitertrainieren von
  Qwen2.5 Cache-Plätze aus und macht den Cache so komprimierbar.

Im Volltext gelesen wurden das UFO-Paper und Ootani; in den einschlägigen Abschnitten Varma et al., Li und Sreedhar und
die arXiv-Arbeiten 2603.19664, 2605.05971, 2311.08360, 2412.00104 und 2604.12151; nur als Zusammenfassung arXiv
2205.05055 und die meisten weiteren Quellen in `kv_literature.json`.

### 8.2 Was sich am Kleinversuch messen lässt

**Der Cache.** Eine Aufgabe ist `[EOS, x, /, y, =, z, EOS]`, die Antwort wird am `=` gelesen. Wenn das `=` verarbeitet
wird, stehen Schlüssel und Werte der Positionen 0 bis 3 in beiden Blöcken fest: Das ist der Cache. In Block 0 sind sie
eine Tabelle über das Token (x an Platz 1, y an Platz 3; die Plätze 0 und 2 sind für alle Aufgaben gleich). In Block 1
hängen Platz 1 und 2 nur von x ab, Platz 3 kann von x und y abhängen. Jeder Eingriff ändert nur, was die Positionen ab
dem `=` aus dem Cache lesen; die Positionen 0 bis 3 rechnen unverändert (`kv_common.py`). Gemessen wird am `=`; im
Training liest auch die Antwortposition dahinter mit. Die `=`-Position heißt im Folgenden auch lesende Position oder
Lesestelle. Ihr Zustand zwischen den Teilblöcken, zu dem jeder Block das Gelesene addiert, heißt `=`-Strom.

**Was hier nicht variieren kann.** Das `=`-Token ist in jeder Aufgabe dasselbe. Alles, was die Antwort bestimmt, kommt
aus dem Cache: Mit dem ganzen Cache einer anderen Aufgabe gibt jedes Netz in jedem geprüften Stand zu 100 % deren
Antwort. „Mehr oder weniger Nutzung" (K1) gibt es in diesem Netz nicht, nur verschiedene Wege durch den Cache:

* **A, fremde Schlüssel.** Anteil der Aufgaben, deren Antwort gleich bleibt, wenn nur die Schlüssel beider Blöcke aus
  einer anderen Aufgabe mit anderer Antwort stammen. Hoch heißt: Die Antwort hängt nicht daran, wessen Schlüssel gelesen
  werden. Ob das Lesemuster selbst vom Inhalt abhängt, zeigt die Streuung der Aufmerksamkeits-Logits des `=` über die
  Aufgaben (relativ zum Mittel der Zeile).
* **S, Antwort schon im Cache.** Anteil der Streuung dessen, was Platz 3 von Block 1 schreiben würde, den die Antwort
  über x und y hinaus erklärt. **L:** Anteil der Aufgaben, die die Antwort einer fremden Aufgabe geben, wenn nur Block 1
  deren Cache liest.
* **Ordnung.** Was ein Platz in den lesenden Strom schreiben würde (Wert mal Ausgangsgewichte des Kopfes; das hängt
  nicht von der Wahl der Basis im Kopf ab), für die 96 Reste in der Reihenfolge der multiplikativen Gruppe (x = 5^a):
  Anteil der Streuung in den 5 stärksten von 48 Frequenzen. Vergleichswerte: dieselben Zeilen in zufälliger Reihenfolge
  und in der Reihenfolge 1 bis 96.
* **Duldsamkeit.** Anteil gleicher Antworten bei Rauschen der Größe 1 (eine Streuung über die Aufgaben je Koordinate)
  auf den Schlüsseln und Werten beider Blöcke, und bei einem Vektor zufälliger Richtung, der stattdessen nach dem Lesen
  auf den `=`-Strom gesetzt wird und so lang ist wie die Änderung, die dieses Rauschen dort bewirkt („gleich langer
  Zufallsvektor").

A, L und die Duldsamkeit an 2.048 Trainingsaufgaben (A und L mit drei Ziehungen der fremden Aufgaben, gezählt über Paare
mit verschiedenen Antworten); S, die Ordnung und die Logit-Streuung über alle 9.312 Aufgaben, also auch die ungesehenen
(`kv_probe.py`). „Auswendig" heißt ein Stand mit mindestens 99 % auf den Trainingsaufgaben und unter 10 % auf den
ungesehenen, „verallgemeinernd" einer mit mindestens 99 % auf den ungesehenen. Bei der Kontrolle mit vertauschten
Antworten zählt die Genauigkeit auf ihren eigenen, vertauschten Trainingsantworten; sie liegt ab Schritt 2.000 bei
mindestens 99 %, außer bei Schritt 10.000 (89 %, nicht mitgezählt; mit diesem Stand bleiben alle Spannen gleich).

**Läufe.** Jeder Lauf rechnet mit einem CPU-Thread auf einem festen Kern. Vier Startwerte nur mit Gewichtsabnahme
(„Grundläufe"): 50 % auf ungesehenen Aufgaben nach 10.846 / 7.323 / 12.759 / 9.418 Schritten (zwischen zwei Messungen im
Abstand von 500 Schritten interpoliert; bei Startwert 2 fällt die Marke in die Erholung nach einem Einbruch: 45 % bei
Schritt 12.000, 17 % bei 12.500, 81 % bei 13.000), 99 % nach 13.000 / 9.500 / 14.000 / 11.000. Dazu die drei Läufe aus
7.10 (Startwert 0 mit drei Threads und früherer Skriptfassung, also dieselben Startgewichte wie der neue Grundlauf mit
Startwert 0: 99 % bei 12.000; Kontrolle mit vertauschten Antworten; UFO auf den MLP-Aktivierungen am `=`). Die
Trainingsarme in 8.5 sind je Startwert mit dem Grundlauf gepaart (gleiche Startgewichte, gleiche Reihenfolge der
Aufgaben, eigener Zufallsgenerator für die Störung). Der SwiGLU-Arm in 8.7 hat mit dem Grundlauf nur die Reihenfolge der
Aufgaben, die Einbettung und die Aufmerksamkeitsgewichte von Block 0 gemeinsam. Ein Lauf endet 500 Schritte nach den
ersten 99 %.

### 8.3 Ordnung im Cache entsteht beim Verallgemeinern, ohne UFO

Anteil der 5 stärksten Frequenzen für Platz 1 von Block 0, den Platz von x (Platz 3 trägt bis auf eine Konstante
dieselbe Tabelle; Spannen über die Stände der Art):

| Lauf | auswendig | verallgemeinernd | zufällige Reihenfolge | Reihenfolge 1 bis 96 |
|---|---|---|---|---|
| Grundlauf, Startwert 0 | 0,20 bis 0,30 | 0,83 bis 0,90 | 0,14 bis 0,17 | 0,14 bis 0,16 |
| Startwert 1 | 0,23 bis 0,32 | 0,59 bis 0,67 | 0,14 bis 0,16 | 0,15 bis 0,16 |
| Startwert 2 | 0,19 bis 0,32 | 0,70 bis 0,79 | 0,14 bis 0,17 | 0,14 bis 0,18 |
| Startwert 3 | 0,17 bis 0,32 | 0,68 bis 0,76 | 0,14 bis 0,18 | 0,14 bis 0,17 |
| Lauf aus 7.10 (bis Schritt 20.000) | 0,18 bis 0,31 | 0,79 bis 0,99 | 0,14 bis 0,18 | 0,14 bis 0,17 |
| Kontrolle mit vertauschten Antworten (Schritt 2.000 bis 20.000) | 0,14 bis 0,17 | – | 0,14 bis 0,16 | 0,13 bis 0,15 |

* Das untrainierte Netz liegt bei 0,13 bis 0,14. Solange ein Netz auswendig lernt, steigt der Wert wenig (bis 0,32); mit
  dem Verallgemeinern steigt er auf 0,59 bis 0,90 und danach weiter (Lauf aus 7.10: 0,79 beim Übergang, 0,99 bei Schritt
  20.000). In der Kontrolle, die nichts verallgemeinern kann, bleibt er über 20.000 Schritte nahe am Wert der zufälligen
  Reihenfolge (ab Schritt 15.000: 0,17 gegen 0,15 bis 0,16).
* In der Reihenfolge 1 bis 96 bleibt der Wert in allen Läufen beim Zufallswert. Die Ordnung folgt der Gruppe, in der
  Division zu Subtraktion wird, nicht der Zahlenfolge.
* Die Plätze von Block 1 verhalten sich ebenso: im Lauf aus 7.10 verallgemeinernd 0,60 bis 0,98, in den vier
  Grundläufen, die 500 Schritte nach den ersten 99 % enden, 0,49 bis 0,93. Dasselbe Maß, auf die Aufmerksamkeits-Logits
  von Block 0 angewandt, ist uneinheitlich (verallgemeinernd 0,30 bis 0,88 in den Grundläufen, 0,63 bis 0,97 im Lauf aus
  7.10; untrainiert 0,20 bis 0,27, auswendig 0,22 bis 0,35).
* **Für K1 heißt das:** Eine Ordnung im Cache gibt es, und sie entsteht von selbst mit dem Verallgemeinern. Ob sie Teil
  der verallgemeinernden Lösung ist oder nur mitläuft, ist nicht geprüft (kein Lauf greift in die Ordnung selbst ein);
  dass sie das Netz zu etwas „geneigt" macht, zeigt das nicht.

### 8.4 In Auswendig-Ständen hängt die Antwort an den Schlüsseln, in verallgemeinernden kaum

| Lauf | A auswendig | A verallgemeinernd | Logit-Streuung Block 0, auswendig / verallgemeinernd (nats) |
|---|---|---|---|
| Grundlauf, Startwert 0 | 0,09 bis 0,13 | 0,90 bis 0,94 | 2,9 bis 3,7 / 0,9 bis 1,1 |
| Startwert 1 | 0,18 bis 0,27 | 0,99 | 3,4 / 1,3 bis 1,5 |
| Startwert 2 | 0,09 bis 0,22 | 0,67 bis 0,70 | 3,7 bis 5,0 / 1,3 |
| Startwert 3 | 0,12 bis 0,23 | 0,92 bis 0,93 | 3,8 bis 4,1 / 2,0 bis 2,1 |
| Lauf aus 7.10 | 0,09 bis 0,14 | 0,87 bis 0,92 | 3,0 bis 4,3 / 0,7 bis 1,2 |
| Kontrolle mit vertauschten Antworten (ab Schritt 2.000) | 0,12 bis 0,19 | – | 2,4 bis 3,6 / – |

* **Das Muster hält in allen vier Grundläufen** (Startwert 0 ist der aus 7.10, mit denselben Startgewichten; neu sind
  die Startwerte 1 bis 3)**:** Auswendig-Stände verlieren mit fremden Schlüsseln 73 bis 91 % ihrer Antworten,
  verallgemeinernde 1 bis 33 %. Die Logits des `=` (relativ zum Mittel der Zeile) streuen in Auswendig-Ständen über die
  Aufgaben um 2,9 bis 5,0 nats, in verallgemeinernden um 0,9 bis 2,1 (untrainiert 0,3 bis 0,4). Auswendig-Stände lesen
  also je nach Inhalt der Schlüssel anders, und ihre Antwort hängt daran. Verallgemeinernde Stände lesen ebenfalls noch
  nach Inhalt, ihre Antwort hängt aber kaum noch daran.
* A allein zeigt keine Verallgemeinerung an: Das untrainierte Netz (Startwert 0) hat A = 0,82, obwohl es nichts kann;
  sein Lesemuster hängt kaum vom Inhalt ab (Logit-Streuung 0,4). Die untrainierten Netze der Startwerte 1 bis 3 geben
  überall dieselbe Antwort; für sie ist A nicht bestimmt.
* **Die vorab gesetzten Schwellen** (unter 0,2 in jedem Auswendig-Stand, über 0,7 in jedem verallgemeinernden) werden an
  vier Stellen verfehlt: Startwert 1 bei Schritt 5.000 (0,27), Startwert 2 bei 10.000 (0,22) und Startwert 3 bei 7.500
  (0,23), jeweils Stände, in denen die Genauigkeit auf ungesehenen Aufgaben schon auf 6 bis 10 % gestiegen ist; und
  Startwert 2 nach dem Übergang (0,67 bis 0,70). Die vorab gesetzte Gegenprobe (ein verallgemeinernder Stand unter 0,5
  oder ein Auswendig-Stand über 0,5) tritt in keinem Grundlauf ein. Nach der Regel (Schwellen in mindestens zwei
  weiteren Startwerten eingehalten) gilt die Aussage damit nicht als gestützt, denn in jedem der drei neuen Startwerte
  ist eine Schwelle verfehlt; widerlegt ist sie auch nicht. Die Kritik hatte nicht festgelegt, bis zu welcher
  Genauigkeit ein Stand als auswendig zählt: Mit der Grenze 5 % statt 10 % hielten die Startwerte 1 und 3 die Schwellen
  ein, Startwert 2 nach dem Übergang weiterhin nicht.
* **Nötig ist die Abhängigkeit von den Schlüsseln fürs Auswendiglernen nicht.** Im Arm mit vertauschten Schlüsseln (8.5,
  Startwert 0) kann das Netz bei Schritt 2.500 die Trainingsaufgaben (100 %, ungesehene 1 %), und fremde Schlüssel
  ändern keine einzige Antwort (A = 1,00; Logit-Streuung in Block 0 0,4, in Block 1 1,3). Dieser Arm wird genau unter
  diesem Eingriff trainiert, und es ist ein Stand eines Laufs (bei den anderen Startwerten ist dieser Arm am ersten
  gespeicherten Stand, Schritt 2.500, schon über das Auswendiglernen hinaus).
* **Mit dem Verallgemeinern ändert sich beides zugleich:** Der Cache wird geordnet, und die Antworten hängen kaum noch
  an den eigenen Schlüsseln. Eine Abstufung darüber hinaus zeigt sich nicht: Unter den verallgemeinernden Ständen hat
  Startwert 1 die geringste Ordnung (0,59 bis 0,67) und das höchste A (0,99), Startwert 2 bei Ordnung 0,70 bis 0,79 das
  niedrigste (0,67 bis 0,70), und im Lauf aus 7.10 steigt die Ordnung nach dem Übergang von 0,79 auf 0,99, während A bei
  0,87 bis 0,92 bleibt. Liest man K1 als „geordneter Cache wird inhaltsabhängiger gelesen", spricht das dagegen.
* L taugt nicht als Maß: Nach dem Übergang liegt es je nach Startwert bei 0,42 bis 0,97, im Lauf aus 7.10 fällt es
  zwischen Schritt 15.000 und 20.000 von 0,49 auf 0,07, und in der Kontrolle wandert es von 0,11 auf 0,42. S ist in
  allen Grundläufen 0,00: Platz 3 von Block 1 trägt y (82 bis 100 % der Streuung) und wenig x (0 bis 14 %), nicht die
  Antwort. Im Lauf aus 7.10 sind es 80 bis 92 % y, und S steigt dort bei Schritt 20.000 auf 0,04.
* **Über Geschwindigkeit sagt nichts davon etwas.** Zwei Blöcke geben einen einzigen Unterschied früh/spät (Block 0
  gegen Block 1, das Maß L), und der trennt Auswendig- und verallgemeinernde Stände nicht. K2 ist hier nur in der Form
  geprüft: „In Auswendig-Ständen hängt die Antwort am Inhalt der Schlüssel, in verallgemeinernden kaum."

### 8.5 Unordnung im Cache und an der Lesestelle

**Am fertigen Netz** (Trainingsaufgaben, Größe 1):

| | Rauschen im Cache | gleich langer Zufallsvektor am `=` |
|---|---|---|
| Auswendig-Stände, vier Grundläufe | 0,24 bis 0,50 | 0,41 bis 0,74 |
| verallgemeinernde Stände, vier Grundläufe | 0,83 bis 0,92 | 0,99 bis 1,00 |
| Kontrolle mit vertauschten Antworten | 0,28 bis 0,44 | 0,44 bis 0,73 |

* Verallgemeinernde Stände vertragen Unordnung weit besser, im Cache und außerhalb. Die Voraussetzung für K3 ist also
  da: Unordnung trifft die auswendige Lösung härter. Sie ist aber nicht auf den Cache beschränkt.
* Das Rauschen wirkt fast nur über die Werte: Rauschen allein auf den Schlüsseln lässt 0,89 bis 1,00 der Antworten
  stehen, auch in Auswendig-Ständen, die mit den Schlüsseln einer anderen Aufgabe 73 bis 91 % verlieren. Zufälliges
  Rauschen trifft die wenigen Richtungen kaum, an denen das Lesemuster hängt.
* Jeder Wert ist eine Ziehung des Rauschens; wiederholte Ziehungen streuen um 0,01 (Zahl des Prüfers).

**Im Training.** Arme (je Startwert mit dem Grundlauf gepaart):

* *Schlüssel vertauscht:* Bei der Hälfte der Aufgaben jedes Stapels liest das `=` in beiden Blöcken die Schlüssel einer
  anderen Aufgabe (die Werte bleiben die eigenen).
* *Rauschen im Cache:* Rauschen von 0,5 Streuungen auf Schlüsseln und Werten, wie sie das `=` liest.
* *Kontrolle zu beiden:* Der Cache bleibt sauber; der `=`-Strom bekommt nach dem Lesen einen Vektor zufälliger Richtung,
  der genau so lang ist wie die Änderung, die die Störung im Cache dort bewirkt hätte. In der Kontrolle zu den
  vertauschten Schlüsseln bekommen ihn deshalb nur die Aufgaben, deren Schlüssel vertauscht worden wären, rund die
  Hälfte jedes Stapels; die Größen unten sind über den ganzen Stapel gemittelt.

| Arm | Startwert 0 | 1 | 2 | 3 | Schritte bis 50 % im Verhältnis zum Grundlauf |
|---|---|---|---|---|---|
| nur Gewichtsabnahme (Grundlauf) | 10.846 / 13.000 | 7.323 / 9.500 | 12.759 / 14.000 | 9.418 / 11.000 | 1 |
| Schlüssel vertauscht | 5.591 / 7.000 | 1.937 / 3.000 | 2.800 / 4.500 | 3.352 / 4.500 | 0,52 / 0,26 / 0,22 / 0,36 |
| Kontrolle dazu (Zufallsvektor am `=`) | 4.551 / 5.500 | 3.502 / 5.000 | 9.156 / 10.000 | 8.222 / 9.500 | 0,42 / 0,48 / 0,72 / 0,87 |
| Rauschen im Cache | 4.400 / 6.000 | 3.981 / 5.500 | 5.140 / 6.500 | 5.252 / 7.000 | 0,41 / 0,54 / 0,40 / 0,56 |
| Kontrolle dazu (Zufallsvektor am `=`) | 7.047 / 8.000 | 3.668 / 4.500 | 8.976 / 10.500 | 8.286 / 9.500 | 0,65 / 0,50 / 0,70 / 0,88 |

(Je Zelle: Schritt mit 50 % auf ungesehenen Aufgaben, zwischen zwei Messungen interpoliert / erste Messung mit
mindestens 99 %. Die Trainingsaufgaben kann jedes Netz nach 1.000 bis 2.500 Schritten.)

* **In allen 16 Läufen mit Störung kam das Verallgemeinern früher als im Grundlauf desselben Startwerts.** Bei den
  beiden Störungen im Cache ist der Abstand groß: Vertauschte Schlüssel brauchen das 0,22- bis 0,52-Fache der Schritte,
  Rauschen im Cache das 0,40- bis 0,56-Fache, und alle acht Läufe sind schneller als der schnellste Grundlauf (7.323
  Schritte). Bei den Kontrollen mit gleich langem Zufallsvektor am `=` ist er kleiner (das 0,42- bis 0,87-Fache und das
  0,50- bis 0,88-Fache); bei den Startwerten 2 und 3 sparen sie nur 12 bis 30 % der Schritte. Zum Maßstab: Alle 16
  Vergleiche hängen an denselben vier Grundläufen. Der von Startwert 2 ist durch einen Einbruch verzögert (8.2), nach
  seinem Anstieg davor geschätzt um rund 550 Schritte; ohne ihn lägen seine vier Verhältnisse bei 0,23 / 0,75 / 0,42 /
  0,73 statt 0,22 / 0,72 / 0,40 / 0,70. Und derselbe Startwert legt einen Lauf nicht fest: Der Lauf aus 7.10 (drei
  Threads, frühere Skriptfassung, alle 1.000 Schritte gemessen) erreicht mit denselben Startgewichten wie der Grundlauf
  von Startwert 0 die 50 % bei 10.368 statt 10.846 und die 99 % bei 12.000 statt 13.000. In der Grokking-Literatur
  helfen Störungen beim Training teils, teils nicht (Rauschen hilft etwas bei Power et al., arXiv 2201.02177; Dropout
  nimmt die Verzögerung bei Liu et al., arXiv 2205.10343; bei Nanda et al., arXiv 2301.05217, brauchen Netze mit Dropout
  statt Gewichtsabnahme nach Vermutung der Autoren viel länger; alle nach `kv_literature.json`).
* **Störung im Cache gegen gleich lange Störung am `=`:** Vertauschte Schlüssel sind in 3 von 4 Startwerten schneller
  als ihre Kontrolle, Rauschen im Cache in 3 von 4. Die vorab gesetzte Regel kannte zwei Ausgänge: „vor dem Grundlauf,
  aber gleichauf mit gleich dosierter Störung außerhalb des Cache: allgemeine Regularisierung, nicht K3" und „vor beiden
  in mindestens fünf gepaarten Startwerten: für K3". Der zweite ist mit vier Startwerten nicht erreichbar, und ein
  Vorsprung in je 3 von 4 Startwerten ist von „gleichauf" nicht zu unterscheiden; mehr als eine Richtung ist das nicht.
  Die Regel verlangte außerdem gleiche Dosis nach Wirkung. Gleich lang ist nicht gleich wirksam: Am fertigen Netz lässt
  der gleich lange Zufallsvektor in Auswendig-Ständen 0,41 bis 0,74 der Antworten stehen, das Rauschen im Cache 0,24 bis
  0,50 (Tabelle oben).
* **„Gleich lang" gilt zudem nur am Anfang.** Mit vertauschten Schlüsseln lernt das Netz, dass seine Antwort nicht an
  den Schlüsseln hängt; die Störung, die das Vertauschen im `=`-Strom bewirkt, fällt dort schon bis Schritt 1.000 auf
  0,02 bis 0,05 der Streuung des Stroms (Block 0; bei Schritt 500 noch 0,20 bis 0,30). In der Kontrolle dazu bleibt sie
  bei 0,61 bis 0,93 (Block 0, alle Messungen von Schritt 1.000 bis 3.000; jede an einem Stapel von 48 Aufgaben), weil
  dort nichts das Netz von den Schlüsseln wegdrängt. Die Kontrolle bekommt über den Lauf also die größere Störung. Beim
  Rauschen sind die beiden Arme näher beieinander (0,37 bis 0,39 gegen 0,47 bis 0,51 bei Schritt 1.000).
* **Auch in den 16 Läufen mit Störung steigt die Ordnung mit dem Verallgemeinern, aber auf sehr verschiedene Höhe.**
  Verallgemeinernde Stände haben 0,36 bis 0,94 (am niedrigsten die Kontrolle zum Rauschen bei Startwert 1 mit 0,36 bis
  0,40 und die vertauschten Schlüssel bei Startwert 1 mit 0,44 bis 0,48), Auswendig-Stände 0,16 bis 0,46. In 7 der 16
  Läufe liegt ein verallgemeinernder Stand unter dem niedrigsten der Grundläufe (0,59). Die Lücke zwischen 0,32 und 0,59
  aus 8.3 gilt also nur für die Grundläufe; verallgemeinern kann das Netz hier schon mit deutlich weniger Ordnung.
* Mit „weniger Nutzung des Cache" hat die Beschleunigung nichts zu tun: Die Antwort kommt in allen Armen weiter ganz aus
  dem Cache. Das Netz lernt, sie zu lesen, ohne auf passende Schlüssel oder rauschfreie Werte angewiesen zu sein.

**Wohin die Rechnung wandert** (verallgemeinernde Stände aller 16 Läufe mit Störung, je Arm und Startwert ein Lauf). In
der Kontrolle zu den vertauschten Schlüsseln steht die Antwort in 3 von 4 Startwerten am Ende zu einem guten Teil schon
im Cache (S 0,45 bis 0,46 / 0,00 / 0,50 bis 0,53 / 0,31; ein linearer Ableser der Antwort aus dem Zustand von Position 3
nach Block 0 trifft dort 61 bis 72 / 1 / 38 bis 46 / 12 bis 19 % der ungesehenen Aufgaben, Zufall 1 %), und in allen
vier entscheidet allein, was Block 1 liest (L = 1,00). Das ist der Arm mit der größten Störung am `=`-Strom (0,61 bis
0,93 der Streuung des Stroms bei Schritt 1.000 bis 3.000, Block 0). Dasselbe hatte der UFO am `=` in 7.10 bewirkt (S
0,14 bis 0,38, L 0,99 bis 1,00). In der Kontrolle zum Rauschen im Cache, deren Störung am `=` kleiner ist (0,43 bis
0,51), bleibt S bei höchstens 0,03 (L 0,43 bis 1,00), ebenso in den beiden Armen mit Störung im Cache (S höchstens
0,06). Dass es an der Größe der Störung liegt, ist nicht geprüft: Die beiden Kontrollen unterscheiden sich auch in der
Form (die eine trifft je Stapel nur die Aufgaben, deren Schlüssel vertauscht worden wären, also rund die Hälfte, die
andere alle), kein Arm ändert nur die Größe, und Startwert 1, die Ausnahme, hatte in Block 0 im Mittel die größte
Störung des Arms (0,73 bis 0,93, die anderen drei 0,61 bis 0,84) und nur in Block 1 die kleinste (0,37 bis 0,58 gegen
0,50 bis 0,69). L allein zeigt die Verlagerung nicht an (es liegt je nach Lauf zwischen 0,02 und 1,00; 8.4).

### 8.6 Ungrokking: sprunghaft, an den fehlenden Aufgaben hängend, nicht im Cache-Pfad allein

Ein verallgemeinerndes Netz (Grundlauf 500 Schritte nach den ersten 99 %) wird auf 233 seiner 4.656 Trainingsaufgaben
weitertrainiert (5 %, so gezogen, dass jeder Rest mindestens zweimal als x, als y und als Antwort vorkommt; der
Optimierer startet neu). Driftkontrolle: derselbe Neustart, alle Trainingsaufgaben.

**Die Verallgemeinerung geht verloren, aber nicht nach der vorab gesetzten Regel.** In allen fünf Läufen (drei
Ausgangsnetze; für Startwert 0 zusätzlich eine zweite ausgewogene und eine rein zufällige Ziehung) liegt die Genauigkeit
auf ungesehenen Aufgaben bei Schritt 1.000 unter 50 % (0,21 bis 0,35), während die 233 behaltenen Aufgaben zu 100 %
stimmen, und nach 4.000 Schritten bei 0,03 bis 0,12. Die weggelassenen Trainingsaufgaben fallen mit den ungesehenen. Mit
10 % der Aufgaben (466) bleibt sie bis Schritt 500 bei 1,00, pendelt von Schritt 750 bis 2.500 zwischen 0,61 und 0,96
und fällt danach weiter (kleinster Wert 0,40 bei Schritt 3.250, bei 4.000 ebenfalls 0,40). Alle Ausgangsnetze sind
frisch verallgemeinert (500 Schritte nach den ersten 99 %). Im Vorlauf mit zufällig gezogenen 10 % vom Lauf aus 7.10 bei
Schritt 20.000, also 8.000 Schritte nach dem Übergang, lag die Genauigkeit nach 4.000 Schritten noch bei 0,94 und fiel
in 6.000 Schritten nicht unter 0,71. Das spricht dafür, dass der Ausgangsstand eine Rolle spielt; es ist aber nur ein
Paar von Läufen, und sie unterscheiden sich auch in der Ziehung (zufällig statt ausgewogen) und in der Gewichtsnorm am
Anfang (41 gegen 51).

**Es geschieht in Sprüngen** (alle 50 Schritte gemessen):

| behaltene / ungesehene Aufgaben richtig | Schritt 500 | 700 | erster Sprung | 1.000 | 1.350 | zweiter Sprung |
|---|---|---|---|---|---|---|
| Startwert 0 | 1,00 / 0,99 | 1,00 / 0,96 | Schritt 750: 0,94 / 0,10 | 1,00 / 0,24 | 1,00 / 0,46 | 1.400: 0,17 / 0,03 |
| Startwert 1 | 1,00 / 0,98 | 1,00 / 0,95 | 800: 0,21 / 0,04 | 1,00 / 0,35 | 1,00 / 0,60 | 1.450: 1,00 / 0,13 |
| Startwert 2 | 1,00 / 0,97 | 1,00 / 0,94 | 750: 0,45 / 0,05 | 1,00 / 0,29 | 1,00 / 0,49 | 1.450: 0,17 / 0,02 |

* Nach einer kurzen Delle gleich nach dem Neustart (Schritt 50: 0,99 / 0,89 / 0,91) liegt die Genauigkeit auf
  ungesehenen Aufgaben um Schritt 400 bei 0,98 bis 0,99 und sinkt bis zur letzten Messung vor dem Sprung auf 0,93 bis
  0,96. Dann bricht das Netz für einen Moment auch auf den behaltenen Aufgaben ein, fängt diese binnen 50 Schritten
  wieder und hat die Verallgemeinerung verloren. Danach steigt die Genauigkeit auf ungesehenen Aufgaben über 600
  Schritte wieder an (auf 0,46 bis 0,61), bis der nächste Einbruch sie erneut wegnimmt. Fein gemessen ist nur diese eine
  Ziehung der 233 Aufgaben.
* **Solche Einbrüche hat auch die Driftkontrolle.** In der Genauigkeit sieht man nur die, die auf einen Messpunkt
  fallen. Startwert 0: bei Schritt 600 nur 14 % der Trainingsaufgaben und 13 % der ungesehenen richtig, bei Schritt 700
  wieder beide 100 %; ein weiterer bei Schritt 4.000 (0,41 / 0,36), mit dem der Lauf endet. Startwert 1: bei Schritt 700
  (1,00 / 0,99) und bei Schritt 1.300 (0,72 / 0,64). Startwert 2: bei Schritt 650 (1,00 / 0,99), sonst an keinem
  Messpunkt unter 0,97. An der Gewichtsnorm gezählt hat jede der drei Driftkontrollen in 4.000 Schritten sechs solche
  Ereignisse (Sprung der Norm um 1 bis 4; Zählung des Prüfers), das erste bei Schritt 600 bis 700. Nach Schritt 1.500
  ist nur alle 250 Schritte gemessen. Auch die Grundläufe haben Einbrüche (Startwert 0: viermal unter 80 % auf den
  Trainingsaufgaben). Mit allen Aufgaben ist das Netz nach jedem Einbruch, dessen Fortgang gemessen ist, binnen 100
  Schritten wieder bei 100 %; mit 5 % kommt es nicht zurück.
* **Die vorab gesetzte Regel** verlangte, dass die Driftkontrolle und die behaltenen Aufgaben durchweg bei mindestens 99
  % bleiben. Beides ist an einzelnen Messpunkten verletzt (behaltene Aufgaben in den Sprüngen 0,94 / 0,21 / 0,45 und
  0,17 / 0,17). Für diesen Fall sah die Regel die Lesart „Instabilität dieses Trainingsaufbaus" vor. Darüber hinaus geht
  nur der Unterschied danach: mit allen Aufgaben wieder 100 %, mit 5 % nicht.
* Die Genauigkeit geht hier also in den Einbrüchen verloren. Das Netz wandert aber schon vorher ab: Vor dem ersten
  Sprung passen Cache und Leser des Netzes, einzeln mit dem Ausgangsnetz gemischt, schlechter zu ihm als die der
  Driftkontrolle (unten), und in den beiden anderen Ziehungen sind bei Schritt 750 schon 9 und 17 % der ungesehenen
  Aufgaben verloren (bei der zufälligen Ziehung 11 % der Aufgaben, deren x, y und Antwort in den 233 vorkommen; 5 und 9
  % fehlten schon bei Schritt 250). Ob es ohne Einbrüche ebenfalls kippen würde, ist nicht geprüft. Der Lauf mit 5 % und
  die Driftkontrolle unterscheiden sich außerdem nicht nur in den Aufgaben: Mit 233 Aufgaben geht jeder Schritt über
  alle, mit allen Aufgaben über einen Stapel von 512.
* An der Gewichtsnorm lässt sich das Kippen nicht ablesen. Sie fällt in beiden Läufen gleichmäßig und springt bei jedem
  Einbruch nach oben (Startwert 0 mit 5 %: 50,4 bei Schritt 250, 47,5 bei 700, 51,0 im Sprung, 40,0 bei 4.000).

**Selten statt fehlend** (K4, erster Teil). Bleiben die weggelassenen Aufgaben als kleiner Anteil jedes Stapels erhalten
(zufällig gezogen), bleibt die Verallgemeinerung großteils erhalten:

| Anteil seltener Aufgaben je Stapel | 0 | 0,4 % (1 Aufgabe) | 2 % (5) | 5 % | 10 % | 20 % |
|---|---|---|---|---|---|---|
| ungesehene Aufgaben richtig, kleinster Wert / bei Schritt 4.000 (Startwert 0) | 0,03 / 0,03 | 0,46 / 0,68 | 0,80 / 0,99 | 0,89 / 0,98 | 0,63 / 1,00 | 0,99 / 1,00 |

* Mit 2 % sieht das Netz jede weggelassene Aufgabe rund 900-mal seltener als eine behaltene; über 4.000 Schritte kommt
  trotzdem fast jede vor, im Mittel 4- bis 5-mal. In drei Ausgangsnetzen verliert es damit zunächst 11 bis 20 % der
  ungesehenen Aufgaben (bei Startwert 0 feiner gemessen 25 %; ohne seltene Aufgaben liegt dasselbe Netz in diesen ersten
  400 Schritten bei 0,98 bis 1,00), erholt sich und liegt in der zweiten Hälfte des Laufs bei 0,92 bis 1,00, bei Schritt
  4.000 bei 0,99. Gemessen ist alle 250 Schritte; kurze Einbrüche wie die der Driftkontrolle sind in diesem Raster nicht
  zu sehen, der kleinste Wert ist also eine obere Grenze. Der kleinste Wert bei 10 % (0,63) ist ein solcher Einbruch,
  der zufällig auf den Messpunkt 750 fällt (behaltene Aufgaben dort 0,97); davor und danach liegt der Lauf bei 1,00. Mit
  einer einzigen seltenen Aufgabe je Stapel fällt die Genauigkeit auf 0,46 (Schritt 1.000) und steigt bis Schritt 4.000
  wieder auf 0,68.
* In dieser Form trägt der erste Teil von K4 im Kleinversuch: Fehlen die übrigen Aufgaben ganz, geht die
  Verallgemeinerung verloren; kommen sie selten vor, bleibt sie großteils erhalten (2 % in drei Netzen, die anderen
  Anteile in einem). Selten heißt hier eine niedrige Rate, nicht wenige Aufgaben. Mit seltenen Aufgaben wechselt
  außerdem der Stapel von Schritt zu Schritt; ohne sie sieht jeder Schritt dieselben 233 Aufgaben. Eine Kontrolle, die
  nur diesen Wechsel herstellt und nichts über die weggelassenen Aufgaben enthält, fehlt.

**Cache-Pfad oder Lese-Pfad?** (K4, zweiter Teil.) Ungrokking ändert Gewichte; der Cache wird bei jedem Durchlauf aus
ihnen neu berechnet. Fragen lässt sich nur, welche Hälfte des Durchlaufs nicht mehr zum Ausgangsnetz passt: das, was die
Positionen 0 bis 3 anbieten, oder das, was die lesende Position damit tut. Weil die Positionen 0 bis 3 die Position 4
nie sehen, lässt sich der Durchlauf exakt teilen: Eine Hälfte rechnet das verallgemeinernde Netz G, die andere das
weitertrainierte. C = Anteil der ungesehenen Aufgaben, der verloren geht, wenn nur der Cache vom weitertrainierten Netz
kommt; R = wenn nur die lesende Position von ihm kommt. Zwei Schnitte: „Zustände" (das Cache-Netz liefert die Zustände
der Positionen 0 bis 3, der Leser bildet Schlüssel und Werte mit eigenen Gewichten) und „Schlüssel/Werte" (das
Cache-Netz liefert Schlüssel und Werte selbst).

C und R messen, wie weit zwei Netze auseinandertrainiert sind, keinen Schaden: Auch die Driftkontrolle, die zu 100 %
verallgemeinert, passt nach 1.000 bis 4.000 Schritten nicht mehr zu G (Schnitt Schlüssel/Werte bei Schritt 1.000: C 0,11
bis 0,73; Schnitt Zustände bei Schritt 4.000: 0,36 und 0,77 bei Startwert 1 und 2, Startwert 0 steckt dort im Einbruch).
Lesbar ist nur der Vergleich mit der Driftkontrolle am selben Schritt, bevor diese ihren ersten Einbruch hat (Startwert
0 bis Schritt 550, 1 bis 650, 2 bis 600):

| Schritt | 250 | 400 | 500 |
|---|---|---|---|
| **Schnitt Zustände, C / R** | | | |
| Startwert 0, 5 % | 0,01 / 0,04 | 0,03 / 0,09 | 0,09 / 0,19 |
| Startwert 0, Driftkontrolle | 0,01 / 0,01 | 0,01 / 0,00 | 0,02 / 0,02 |
| Startwert 1, 5 % | 0,00 / 0,07 | 0,00 / 0,06 | 0,01 / 0,12 |
| Startwert 1, Driftkontrolle | 0,00 / 0,00 | 0,00 / 0,00 | 0,00 / 0,00 |
| Startwert 2, 5 % | 0,04 / 0,19 | 0,05 / 0,15 | 0,08 / 0,17 |
| Startwert 2, Driftkontrolle | 0,00 / 0,00 | 0,00 / 0,00 | 0,00 / 0,00 |
| **Schnitt Schlüssel/Werte, C / R** | | | |
| Startwert 0, 5 % | 0,02 / 0,05 | 0,16 / 0,15 | 0,39 / 0,29 |
| Startwert 0, Driftkontrolle | 0,03 / 0,04 | 0,10 / 0,06 | 0,23 / 0,14 |
| Startwert 1, 5 % | 0,00 / 0,06 | 0,00 / 0,06 | 0,03 / 0,12 |
| Startwert 1, Driftkontrolle | 0,00 / 0,00 | 0,00 / 0,00 | 0,00 / 0,00 |
| Startwert 2, 5 % | 0,08 / 0,20 | 0,18 / 0,17 | 0,32 / 0,19 |
| Startwert 2, Driftkontrolle | 0,00 / 0,00 | 0,04 / 0,00 | 0,15 / 0,01 |

* **Vor dem Sprung** verallgemeinert das Netz noch zu 97 bis 99 %, seine Hälften passen aber bei Schritt 500 schon
  schlechter zu G als die der Driftkontrolle (bei Schritt 250 gilt das bei Startwert 0 nur für den Lese-Pfad). Der
  Überschuss bei Schritt 500 liegt im Schnitt „Zustände" auf dem Lese-Pfad (+0,17 / +0,12 / +0,17 gegen +0,06 / +0,01 /
  +0,08 auf dem Cache-Pfad). Im Schnitt „Schlüssel/Werte" liegt er bei den Startwerten 0 und 2 auf beiden etwa gleich
  (+0,16 und +0,18 gegen +0,15 und +0,17), bei Startwert 1 auf dem Lese-Pfad (+0,12 gegen +0,02). Dass in diesem Schnitt
  der Cache-Pfad mehr verliert als der Lese-Pfad, zeigt schon die Driftkontrolle (0,23 gegen 0,14; 0,15 gegen 0,01).
* **Nach dem ersten Sprung** (Schritt 850 bis 1.000, Schnitt „Zustände"): In zwei von drei Netzen rechnet der alte Leser
  mit den Zuständen des gekippten Netzes noch 65 bis 82 % der ungesehenen Aufgaben richtig, das gekippte Netz selbst 24
  bis 35 % und der gekippte Leser mit dem alten Cache 24 bis 28 %. Beim dritten (Startwert 0) liegen beide Mischungen
  bei 13 bis 15 %. Im Schnitt „Schlüssel/Werte" bleiben in allen drei Netzen höchstens 28 %.
* **Die vorab gesetzte Regel für „der Cache trägt das Ungrokking"** verlangte: C mindestens doppelt so groß wie R und
  mindestens die Hälfte des Gesamtverlusts, in beiden Schnitten und allen Netzen; deutlich über dem Verhältnis bei einem
  über alle Gewichte verteilten Schaden; die Mischungen mit der Driftkontrolle bei mindestens 90 %, sonst ist der
  Schnitt nicht lesbar; und die Mischung mit dem gekippten Cache fällt früher unter 50 % als die mit dem gekippten
  Leser. Ergebnis: C ist vor dem Sprung an keinem Stand in keinem Netz doppelt so groß wie R; am nächsten kommt dem
  Startwert 2 im Schnitt „Schlüssel/Werte" bei Schritt 600 bis 700 (das 1,97- bis 1,995-Fache). Im Schnitt
  „Schlüssel/Werte" ist die 90-%-Bedingung ab Schritt 400 (Startwert 0) und 500 (Startwert 2) verletzt (0,77 und 0,85
  bei Schritt 500); dieser Schnitt ist nach der Regel unentscheidbar. Im Schnitt „Zustände" hält sie bis zum Einbruch
  der Driftkontrolle, und dort ist R in allen drei Netzen größer als C, nach der Regel ein Ergebnis gegen den
  Cache-Pfad. Der zeitliche Teil trifft im Schnitt „Schlüssel/Werte" bei zwei Netzen zu (Schritt 550 gegen 650 und 650
  gegen 750), aber auch bei der Driftkontrolle von Startwert 2, und zählt deshalb nicht.
* **Was sich nicht folgern lässt:** dass der Lese-Pfad das Ungrokking trägt. Im Schnitt „Zustände" gehört zum Lese-Pfad
  der größere Teil des Netzes (auch die Schlüssel- und Wertgewichte beider Blöcke), und ein kleiner, gleichmäßig über
  alle Gewichte verteilter Schaden gibt ebenfalls R weit über C (alle Gewichte von G so weit zum Auswendig-Stand bei
  Schritt 5.000 verschoben, dass noch 82 bis 94 % der ungesehenen Aufgaben stimmen: C 0,00 bis 0,01, R 0,16 bis 0,44).
  Außerdem ist das weitertrainierte Netz beim gleichen Schritt weiter von G entfernt als die Driftkontrolle (Startwert
  0, Schritt 500: 13 gegen 10 % der Gewichtsnorm; Zahl des Prüfers), und alle drei Netze wurden auf denselben 233
  Aufgaben weitertrainiert.
* Es bleibt: Nichts spricht dafür, dass das Ungrokking im Cache-Pfad sitzt. Mehr als „nicht im Cache-Pfad allein" tragen
  die Zahlen nicht.

**Darf nur eine Hälfte lernen** (entkoppeltes Netz: eine Kopie von G rechnet die Positionen 0 bis 3, eine zweite die
`=`-Position; das ist ein anderes Modell mit doppelt so vielen Gewichten):

| trainierbar | ungesehene Aufgaben richtig, kleinster Wert / letzter Wert |
|---|---|
| nur die Kopie, die den Cache schreibt (Startwert 0, 3.000 Schritte) | 0,75 / 0,75 |
| nur die Kopie, die liest (Startwert 0) | 0,25 / 0,26 |
| beide (Startwert 0) | 0,03 / 0,03 |
| nur die Kopie, die den Cache schreibt (Startwert 1) | 0,81 / 0,83 |
| nur die Kopie, die liest (Startwert 1) | 0,23 / 0,43 |

* Darf nur der Cache sich ändern, bleiben nach 3.000 Schritten 75 bis 83 % der ungesehenen Aufgaben richtig; darf nur
  die lesende Seite sich ändern, schwankt der Wert stark und fällt bis auf 23 bis 25 %; mit beiden geht fast alles
  verloren, wie im gekoppelten Netz. Über den Cache allein kippt das Netz in 3.000 Schritten also am langsamsten, es
  kippt aber auch dort: Bei Startwert 0 liegt jeder Tiefpunkt unter dem vorigen (0,93 / 0,87 / 0,81 / 0,75 bei Schritt
  750 / 1.500 / 2.250 / 3.000), und der Lauf endet auf seinem tiefsten Wert. Der Versuch zeigt, was jede Hälfte kann; wo
  das Ungrokking im gekoppelten Netz sitzt, zeigt er nicht. Zwei Ausgangsnetze, je ein Lauf.

**Die Verschiebung des Cache als gezielte Unordnung** (K5; nur Beschreibung). d = Cache des weitertrainierten Netzes
minus Cache von G, je Aufgabe (Schlüssel und Werte beider Blöcke). Daneben dieselbe Rechnung für die Driftkontrolle, die
weiter zu 100 % verallgemeinert. Je Zelle die drei Ausgangsnetze (Startwert 0 / 1 / 2); „G mit …" = Anteil richtiger
Antworten von G auf ungesehenen Aufgaben, wenn die Störung zu seinem Cache addiert wird:

| | 5 %, Schritt 4.000 | 5 %, Schritt 1.000 | Driftkontrolle, Schritt 3.000 |
|---|---|---|---|
| eigene Genauigkeit auf ungesehenen Aufgaben | 0,03 / 0,12 / 0,05 | 0,24 / 0,35 / 0,29 | 1,00 / 1,00 / 1,00 |
| Anteil von d, der für alle Aufgaben gleich ist | 0,96 / 0,99 / 0,98 | 0,95 / 0,97 / 0,96 | 0,97 / 0,98 / 0,97 |
| aufgabeneigener Rest von d / Streuung des Cache von G | 0,89 / 0,76 / 0,78 | 0,57 / 0,50 / 0,46 | 0,55 / 0,65 / 0,52 |
| G mit dem gemeinsamen Anteil allein | 0,10 / 0,28 / 0,19 | 0,07 / 0,26 / 0,24 | 0,09 / 0,63 / 0,29 |
| G mit dem aufgabeneigenen Rest | 0,22 / 0,49 / 0,28 | 0,81 / 0,98 / 0,85 | 0,85 / 0,96 / 0,96 |
| … mit dem Rest einer anderen Aufgabe | 0,44 / 0,72 / 0,49 | 0,83 / 0,99 / 0,88 | 0,85 / 0,93 / 0,91 |
| … mit dem gleich langen Unterschied zweier Aufgaben im Cache von G | 0,06 / 0,21 / 0,08 | 0,50 / 0,92 / 0,61 | 0,45 / 0,67 / 0,43 |
| … mit gleich langer Zufallsrichtung aus den 32 Hauptachsen des Cache von G | 0,27 / 0,44 / 0,27 | 0,72 / 0,93 / 0,76 | 0,68 / 0,78 / 0,62 |
| … mit Rauschen gleicher Kovarianz in den 32 Hauptachsen von d | 0,68 / 0,85 / 0,75 | 0,92 / 1,00 / 0,95 | 0,88 / 0,96 / 0,94 |
| … mit gleich langem Rauschen beliebiger Richtung | 0,96 / 0,98 / 0,96 | 1,00 / 1,00 / 1,00 | 1,00 / 1,00 / 1,00 |

* **Die grobe Gestalt der Verschiebung ist keine Eigenheit des Ungrokking.** Sie ist zu 95 bis 99 % ein Versatz, der für
  alle Aufgaben gleich ist, und dieser Versatz allein bringt G aus dem Tritt. Beides gilt auch für die Driftkontrolle
  (bei Startwert 1 bringt ihr Versatz G weniger aus dem Tritt: 0,63 gegen 0,26 bis 0,28). Es zeigt, dass sich der Inhalt
  des Cache beim Weitertrainieren als Ganzes verschiebt.
* **Der aufgabeneigene Rest des gekippten Netzes stört G bei gleicher Größe etwas mehr als der der Driftkontrolle.** Auf
  das Doppelte vergrößert (rund eine Streuung des Cache von G) lässt der Rest der Driftkontrolle 0,28 / 0,60 / 0,28 der
  Antworten stehen, der Rest des gekippten Netzes bei Schritt 1.000 nur 0,19 / 0,48 / 0,19, obwohl er in zwei der drei
  Netze kleiner ist. Der Rest bei Schritt 4.000 richtet denselben Schaden schon bei 0,76 bis 0,89 Streuungen an (0,22 /
  0,49 / 0,28). Er trifft auch gezielter: Der eigene Rest stört mehr als der einer anderen Aufgabe (0,22 / 0,49 / 0,28
  gegen 0,44 / 0,72 / 0,49); bei der Driftkontrolle ist beides gleich (bei doppelter Größe 0,28 / 0,60 / 0,28 gegen 0,28
  / 0,53 / 0,34). Je eine Ziehung; eine Driftkontrolle gleicher Größe für Schritt 4.000 fehlt.
* In jeder Spalte stört eine gleich lange Verschiebung entlang dessen, was G selbst zwischen zwei Aufgaben
  unterscheidet, stärker als der eigene Rest, und beliebiges Rauschen fast gar nicht. Bei halber Größe (Schritt 4.000)
  stören nur die beiden Verschiebungen merklich, die dem Cache von G selbst folgen (Unterschied zweier Aufgaben,
  Zufallsrichtung aus seinen Hauptachsen; 0,83 bis 0,98), der eigene Rest kaum (0,94 bis 1,00). Bei doppelter zerstören
  der eigene Rest, der fremde Rest und diese beiden Verschiebungen fast alles (0,01 bis 0,11); Rauschen mit der
  Kovarianz von d lässt 0,14 bis 0,74 stehen, beliebiges Rauschen 0,44 bis 0,53.
* Bei Schritt 4.000 gibt G danach nur selten die Antworten des weitertrainierten Netzes (Übereinstimmung höchstens 12
  %). Mit dem eigenen Rest sind es 9 / 10 / 9 %; dieses Netz selbst rechnet 3 / 12 / 5 % richtig. Bei 7 / 4 / 7 % der
  Aufgaben gibt G dessen falsche Antwort (Zahl des Prüfers; Zufall wäre rund 1 %). Die Verschiebung überträgt das
  Ungrokking also nur zu einem kleinen Teil.
* Es bleibt ein schwacher, aufgabeneigener Anteil der Verschiebung, der zum Ungrokking gehört und nicht zum bloßen
  Weitertrainieren. K5 selbst ist nicht geprüft: Gemessen ist, wie das verallgemeinernde Netz bricht. K5 bräuchte ein
  Muster, das eine auswendige Lösung trifft und die verallgemeinernde schont, und einen Lauf, der es beim Training
  benutzt.

### 8.7 SwiGLU statt ReLU; Bezug zu 7.4 und zu einem Beitrag über kubische Polynome

Anlass: ein Hinweis, dass SwiGLU durch das Produkt zweier linearer Projektionen eine quadratische Wechselwirkung hat
(„ähnlich einem Polynom zweiten Grades, durch die Sigmoid-Funktion gedämpft"), und ein Beitrag von Bottazzi (Wolfram
Community, 16. Juli 2026, „Interpreting How Neural Nets Regress Cubic Polynomials").

* **Was an der Aussage stimmt.** Ein SwiGLU-Neuron schreibt `silu(Gate-Eingang) × Up-Eingang`, also das Produkt zweier
  linearer Formen mal einer Sigmoid-Funktion. Die Sigmoid-Funktion dämpft das Produkt (Faktor zwischen 0 und 1),
  begrenzt es aber nicht: Für großen positiven Gate-Eingang geht der Faktor gegen 1, und das Produkt wächst quadratisch.
  Für negativen Gate-Eingang bleibt ein kleiner negativer Faktor (`silu` zwischen −0,28 und 0); aus ist das Neuron dort
  nicht (7.4). Dass die Werte nicht davonlaufen, liegt nach dem Aufbau an der Normierung vor dem MLP; gemessen ist das
  hier nicht.
* **Was 7.4 dazu an Qwen3-1.7B gemessen hat.** Nur die Antwort des MLP auf einen kleinen zugesetzten Vektor. In den
  Blöcken 1 bis 9 liegt der Gate-Eingang bei 84 bis 98 % der Neuronen im Negativen. Dort ist `silu` klein (zwischen
  −0,28 und 0) und ändert sich mit dem Gate-Eingang wenig; das Neuron reicht sein Up-Signal mit kleinem negativem Faktor
  durch. Das ist die gemessene Löschung. Über den Gate-Zweig, also die Stelle, an der das Produkt zählt, laufen je Block
  0,007 bis 0,012 der Löschung von 0,15 bis 0,24. In den Blöcken 10 bis 16 übernehmen zunehmend Neuronen mit positivem
  Gate-Eingang (ab Block 12 der größere Teil). Was das Produkt für die übrige Rechnung des MLP bedeutet, ist damit nicht
  gemessen.
* **Im Kleinversuch.** Ein Arm mit SwiGLU-MLP statt ReLU (341 statt 512 Neuronen, gleich viele MLP-Gewichte; nur
  Gewichtsabnahme; vier Startwerte). 50 % auf ungesehenen Aufgaben nach 9.148 / 11.186 / 6.030 / 8.057 Schritten, mit
  ReLU nach 10.846 / 7.323 / 12.759 / 9.418, also nach dem 0,84-, 1,53-, 0,47- und 0,86-Fachen. 3 von 4 Startwerten sind
  schneller, einer deutlich langsamer; der Unterschied ist nicht größer als die Streuung zwischen den Startwerten, und
  die Startgewichte sind nicht dieselben (8.2). Das Bild im Cache ist dasselbe wie mit ReLU: A 0,08 bis 0,17 in
  Auswendig-Ständen und 0,70 bis 0,98 in verallgemeinernden, Ordnung 0,17 bis 0,25 und 0,70 bis 0,81. Ein Vorteil der
  quadratischen Wechselwirkung zeigt sich in dieser Aufgabe nicht. Eine rein bilineare Einheit ohne SiLU ist im Skript vorhanden, lief hier aber nicht (in Abschnitt 9 gelaufen, dort aber nicht auf Geschwindigkeit und Cache ausgewertet).
* **Der Beitrag über kubische Polynome.** Ein MLP mit vier verdeckten Schichten (64, 64, 64 und 15 Neuronen, ReLU;
  nach 9.1 berichtigt: hier stand zuerst „vier Schichten zu 15 Neuronen", die Aktivierungsfunktion sei nicht genannt
  und der Code liege nicht vor) lernt y = x³ + a·x² + x + b aus (x, a, b), alle zwischen −3 und 3. In einem SAE (64
  Einheiten, 4 aktiv) auf der letzten verdeckten Schicht wählt der Autor nach Augenschein neun Einheiten aus, die vor
  allem von t = x + c·a abhängen (sein Test erklärt 88 bis 98,5 % ihrer Streuung damit), mit c zwischen 0,175 und 0,35
  (Median 0,30); dass sie nicht von b abhängen, beruht auf dem Vergleich zweier Bilder. Der Autor stellt das neben
  Cardanos Substitution t = x + a/3 und lässt selbst offen, ob der Eingriff das bestätigt oder widerlegt: Verstärkt
  man die Einheiten, wird die Kurve senkrecht gestreckt, nicht waagrecht verschoben. Ein Netz, ein SAE.
* **Gegenprobe** (`cubic_direction.py`). Die Zielfunktion selbst ändert sich in der (x, a)-Ebene entlang x + c·a mit c =
  x² / (3x² + 2ax + 1). Über den Trainingsbereich ist die Hauptrichtung dieses Gradienten c = 0,26; die Hälfte des
  Gradientengewichts liegt zwischen 0,19 und 0,34, und 1/3 ist der Grenzwert für großes |x|. Der Test des Beitrags: x +
  c·a wird in 50 Fächer geteilt, die Größe aus dem Mittel ihres Fachs vorhergesagt, und gesucht wird das c, mit dem das
  am besten gelingt (hier an 10.000 Punkten angepasst, an 10.000 anderen geprüft, c in Schritten von 0,025). Mit diesem
  Test ist schon die Zielfunktion zu 97 % Funktion von x + c·a allein, am besten mit c = 0,275 (ohne b: 98 %, c = 0,30);
  zwischen c = 0,25 und 0,325 ändert sich das kaum (0,967 und 0,967), mit c = 0 sind es 74 %. Eine Einheit, die nur der
  Höhe der Ausgabe folgt, zeigt also dieses Band und hängt kaum von b ab (b trägt 0,5 % der Empfindlichkeit der
  Zielfunktion). Dazu passt der Eingriff des Beitrags: Verstärken streckt die Kurve senkrecht. In 15 eigenen, kleineren
  Netzen (15-15-15-15; das Netz des Beitrags ist größer, siehe 9.5; je fünf Startwerte mit ReLU, tanh und SiLU, ohne
  SAE, direkt an den Neuronen der letzten Schicht):

  | | ReLU | tanh | SiLU |
  |---|---|---|---|
  | Richtung, entlang der sich die letzte Schicht als Ganzes ändert (c, fünf Startwerte) | 0,26 bis 0,31 | 0,23 bis 0,25 | 0,23 bis 0,25 |
  | Neuronen, die zu über 90 % Funktion von x + c·a allein sind | 38 von 58 (17 der 75 ReLU-Neuronen sind konstant und zählen nicht) | 48 von 75 | 70 von 75 |
  | ihr c, Median (Viertelwerte) | 0,275 (0,250 bis 0,325) | 0,238 (0,200 bis 0,331) | 0,275 (0,250 bis 0,300) |

  Untrainiert ist die Richtung beliebig (c zwischen −28 und 3), und auf b entfällt höchstens 1,1 % der Empfindlichkeit
  der Schicht. Der Befund des Beitrags stellt sich also auch ohne SAE ein: in allen 15 Netzen für die Schicht als Ganze,
  in 14 von 15 für mindestens sechs einzelne Neuronen (in einem ReLU-Netz nur für 2 von 13). Die Werte liegen dort, wo
  sich die Funktion selbst ändert: Von den 156 Neuronen liegen 79 % zwischen 0,19 und 0,34, 33 % innerhalb von 0,025 um
  0,26 und 13 % innerhalb von 0,025 um 1/3. Ein Hinweis darauf, dass ein Netz Cardanos Schritt „gefunden" hat, ist das
  nicht; die Richtung 0,26 und Cardanos 1/3 kommen beide aus den führenden Termen x³ und a·x², und mit der Auflösung des
  Beitrags (neun Einheiten, deren Werte alle Vielfache von 0,025 sind) lassen sich 0,26, 0,30 und 1/3 nicht trennen.
  Dieselbe Vorsicht wie in 7.9: Ein Maß kann eine Eigenschaft der Aufgabe wiedergeben statt einer Leistung des Netzes.
  Das Ordnungsmaß in 8.3 hat dafür die Kontrolle mit vertauschten Antworten und die zufällige Reihenfolge.

### 8.8 Was folgt daraus

* **K1 (geordneter Cache wird mehr benutzt).** So nicht prüfbar, weil in dieser Aufgabe alles aus dem Cache kommt.
  Gezeigt: Der Cache ordnet sich beim Verallgemeinern von selbst (8.3; in den Grundläufen auf 0,59 bis 0,90, mit Störung
  im Training verallgemeinern Netze schon ab 0,36, 8.5), und mit dem Verallgemeinern hängen die Antworten kaum noch an
  den eigenen Schlüsseln (8.4). Die Ordnung allein legt nicht fest, wo die Rechnung sitzt: Im Grundlauf von Startwert 0
  steht die Antwort bei Ordnung 0,83 bis 0,90 nicht im Cache (S 0,00). In dem Arm mit der größten Störung an der
  lesenden Position steht sie in drei von vier Läufen am Ende zu einem guten Teil im Cache, in den zwölf anderen Läufen
  mit Störung und in den Grundläufen nicht (8.5). Ob das an der Größe der Störung liegt, ist nicht geprüft. In diesem
  einen Arm steigen S und Ordnung über die vier Startwerte gemeinsam (S 0,00 / 0,31 / 0,45 bis 0,46 / 0,50 bis 0,53 bei
  Ordnung 0,59 bis 0,74 / 0,75 bis 0,80 / 0,81 bis 0,87 / 0,88 bis 0,94); vier Läufe entscheiden das nicht.
* **K2 (starker Zugriff heißt Auswendiglernen).** Gesehen in der Form „in Auswendig-Ständen hängt die Antwort an den
  eigenen Schlüsseln, in verallgemeinernden kaum", in allen vier Grundläufen und der Kontrolle. Nach der vorab gesetzten
  Regel gilt das wegen der verfehlten Schwellen nicht als gestützt; widerlegt ist es auch nicht. Es ist ein
  Begleitzeichen und nicht nötig: Im Arm mit vertauschten Schlüsseln lernt das Netz auch ohne das auswendig (ein
  Startwert, ein Stand). Zu Geschwindigkeit nichts; das Paper dazu wurde nicht gefunden.
* **K3 (Anti-UFO im Cache).** Gut gestützt ist hier nur der allgemeine Teil: Störung beim Training lässt das Netz früher
  verallgemeinern. Mit Unordnung im Cache war das in allen acht Läufen so (gemessen am Schritt mit 50 %: vertauschte
  Schlüssel auf das 0,22- bis 0,52-Fache der Schritte, Rauschen auf das 0,40- bis 0,56-Fache), mit einem gleich langen
  Zufallsvektor an der lesenden Position ebenfalls (auf das 0,42- bis 0,88-Fache). Dass es auf den Cache ankommt, ist
  nach der vorab gesetzten Regel nicht gestützt und nicht widerlegt. Sie verlangte einen Vorsprung vor gleich dosierter
  Störung außerhalb des Cache in mindestens fünf gepaarten Startwerten. Gelaufen sind vier, und der Cache liegt in 6 von
  8 Vergleichen vorn. Gleich dosiert ist die Kontrolle nicht: Am fertigen Netz ändert der gleich lange Zufallsvektor
  weniger Antworten als das Rauschen im Cache, und im Training bleibt die Störung in der Kontrolle größer als im Arm mit
  vertauschten Schlüsseln (8.5). Am fertigen Netz trifft Unordnung die auswendige Lösung härter als die
  verallgemeinernde, im Cache wie außerhalb. Mit „weniger Cache benutzen" hat das nichts zu tun: Das kann das Netz hier
  nicht; es lernt, trotz der Störung zu lesen. Geprüft bei Gewichtsabnahme 0,3; ob es bei 0,03 genauso ist, ist offen.
* **K4.** Erster Teil in schwacher Form gestützt: Fehlen die übrigen Aufgaben ganz, geht die Verallgemeinerung verloren;
  kommen sie selten vor (2 % je Stapel), bleibt sie großteils erhalten (drei Netze). Zweiter Teil nicht gestützt: Die
  vorab gesetzte Regel für „der Cache trägt das Ungrokking" ist in keinem der drei Netze erfüllt. Vor dem Sprung weicht
  im lesbaren Schnitt („Zustände") der Lese-Pfad stärker von G ab als der Cache-Pfad (bei Schritt 500 R 0,12 bis 0,19
  gegen C 0,01 bis 0,09). Im Schnitt „Schlüssel/Werte" verliert bei zwei Netzen der Cache-Pfad mehr; das zeigt aber
  schon die Driftkontrolle, und der Überschuss über sie ist auf dem Lese-Pfad mindestens so groß. Im entkoppelten Netz
  (ein anderes Modell, zwei Ausgangsnetze) kippt das Netz über den Cache allein am langsamsten. Wo das Ungrokking sitzt,
  zeigt das nicht. Die Genauigkeit geht hier in Sprüngen verloren, bei Einbrüchen des Trainings, die der Aufbau auch
  sonst hat.
* **K5.** Nicht geprüft, und die Beschreibung entscheidet nichts. Gemessen ist, wie das verallgemeinernde Netz bricht;
  K5 bräuchte ein Muster, das eine auswendige Lösung trifft und die verallgemeinernde schont, und einen Lauf, der es
  beim Training benutzt. Zur Beschreibung: Die Verschiebung des Cache ist zu 95 bis 99 % für alle Aufgaben gleich, bei
  der Driftkontrolle ebenso. Ihr aufgabeneigener Rest stört G etwas mehr als der der Driftkontrolle, aber weniger als
  eine gleich lange Verschiebung entlang dessen, was G zwischen zwei Aufgaben unterscheidet.
* **R1.** Nicht prüfbar. Kein Modell hier kann seine Gewichte ändern, und nichts misst eine Neigung dazu. In der
  Literatur fand sich dazu nichts (eine Suchanfrage).
* **SwiGLU und der Beitrag über kubische Polynome (8.7).** Die quadratische Wechselwirkung gibt es; wie quadratisch SwiGLU in Qwen über alle Blöcke ist, misst 9.4. An der Stelle, an der 7.4 sie in Qwen gemessen hat (Antwort auf einen kleinen zugesetzten Vektor), läuft fast alles über den Up-Zweig, in
  dem das Neuron sein Signal nur mit einem Faktor durchreicht (über den Gate-Zweig 0,007 bis 0,012 von 0,15 bis 0,24),
  und im Kleinversuch bringt SwiGLU keinen klaren Vorteil. Dass ein Netz beim kubischen Polynom eine Größe der Form x +
  c·a mit c um 0,26 bildet (0,23 bis 0,31 je Netz, nahe Cardanos 1/3), folgt schon aus der Aufgabe.

**Wie sicher?** Ein Netz mit zwei Blöcken, eine Aufgabe, Gewichtsabnahme 0,3 (das Zehnfache des Papers). Vier Startwerte
haben die Grundläufe und die fünf Trainingsarme (Startwert 0 ist derselbe Anfang wie der Lauf aus 7.10). Das Ungrokking
ruht auf drei Ausgangsnetzen, die alle auf denselben 233 Aufgaben weitertrainiert wurden und alle erst 500 Schritte
verallgemeinern; die Anteile außer 2 % und die beiden anderen Ziehungen auf einem Netz, das entkoppelte Netz auf zweien.
Jede Zahl aus einem einzelnen Lauf kann sich mit einem anderen Startwert deutlich verschieben (die Grundläufe erreichen
50 % zwischen Schritt 7.323 und 12.759). Mit eigenem Code nachgerechnet sind die Messzahlen an 19 Ständen von neun
Läufen (die vier Grundläufe, Startwert 0 von drei Armen, zwei Läufe aus 7.10), die Pfad-Analyse und S in den 16 Läufen
mit Störung. A, L und die Duldsamkeit der Arme bei den Startwerten 1 bis 3, der Kontrolle zum Rauschen und des
SwiGLU-Arms sind nicht unabhängig nachgerechnet. Die Deutung wurde zweimal gegengelesen und danach jeweils an vielen
Stellen zurückgenommen; die Änderungen nach dem zweiten Durchgang hat niemand mehr gegengelesen (`review_kv.json`).

## 9 Aktivierungsfunktion, Ungrokking, SwiGLU als quadratische Einheit, Halbieren und Dritteln (Nachtrag vom 11. November 2026)

Anlass ist eine Erweiterung der Kette aus Abschnitt 8:

* **V1** Was Ungrokking bewirkt, ist letztlich die Aktivierungsfunktion. Der Satz lässt zwei Lesarten zu: Die
  Aktivierungsfunktion ist die Ursache, oder sie ist der Ort, an dem Ungrokking wirkt.
* **V2** Überwacht ein Netz sich selbst, also seine eigenen Aktivierungsfunktionen, hat es vielleicht auch eine
  Repräsentation von SwiGLU.
* **V3** Wenn SwiGLU ungefähr quadratisch ist, gelten dafür die Beobachtungen des Beitrags über kubische Polynome (8.7).
* **V4** Das lässt sich mathematisch mit Dreiteilung und Dreiteilenden weiterdenken (Pehlivan und Williams, Arhimede
  Mathematical Journal 13/1, 2026).

Der Versuchsplan wurde geschrieben, bevor ein Lauf ausgewertet war; nur die Grundläufe mit rein bilinearer Einheit
liefen da schon. Ein Prüfer hat ihn zerlegt (`act_plan_review.json`, darin auch der ursprüngliche Plan). Er hat beide
Lesarten von V1 eingefordert, gezeigt, dass beim Tausch der Schaltmuster mit ReLU schon nach der Bedingung des Plans
(beide Mischungen mit der Driftkontrolle bei mindestens 90 %) jeder Schritt nach dem ersten Einbruch der Driftkontrolle
unlesbar ist, die Bedingung verschärft, ein Vergleichsnetz mit gleichmäßig verteiltem Schaden verlangt, die Prüfung mit
beruhigtem Optimierer an den Anfang gestellt, einen Lauf mit eingefrorenem Schaltmuster ergänzt und den Trainingsarm mit
Selbstüberwachung gestrichen (er prüft V2 nicht und war nicht sauber kontrollierbar). Seine Entscheidungsregeln sind
hier angewandt, für 9.2 in `act_verdicts.py` und `act_verdicts.json`, für 9.3 bis 9.6 im Text. Abweichend vom Plan
endeten die Läufe mit kleinerer Lernrate nach rund 5.000 Schritten (der 5-%-Lauf mit ε = 10⁻⁸ nach 13.350) statt nach
13.333 und 26.667 und sind wie eine Vorprüfung gelesen; ihre Driftkontrollen hatten da schon Ereignisse, das Ergebnis
bleibt damit nach der Regel unentschieden. Ihre Ereignisse sind aus der Aufzeichnung von Stapelverlust und Gewichtsnorm
gezählt, ohne die Genauigkeit auf den Trainingsaufgaben. Der vorgesehene Prüflauf mit einer zweiten Ziehung der 233
Aufgaben je Architektur (außerhalb der Regel) ist nicht gelaufen. Nach Schritt 1.500 sind die Stände der Fortsetzungen
nur alle 500 Schritte gespeichert, nicht alle 50; der Tausch der Schaltmuster ist an allen gespeicherten Ständen
gerechnet. Zwei Prüfer haben die Literatur gesichtet (`act_literature.json`), ein dritter hat die Mathematik mit eigenem
Code nachgerechnet (`act_math_review.json`). Alles gilt für das kleine Netz aus Abschnitt 8, in 9.5 und 9.6 zum Teil für
kleine Netze, die Polynome lernen, und, wo genannt, für Qwen3-1.7B. Über Ungrokking in Sprachmodellen sagt der
Kleinversuch nichts; Qwen3-1.7B wurde nicht weitertrainiert.

### 9.1 Was die Literatur zu den Annahmen sagt

* **Ungrokking hängt in der Literatur an der Größe der Trainingsmenge, nicht an der Aktivierungsfunktion.** Varma et al.
  (arXiv 2309.02390) erklären es mit der Effizienz der Schaltungen unter Gewichtsabnahme: Unterhalb einer kritischen
  Größe der Trainingsmenge ist die auswendig lernende Lösung die billigere. Die Aktivierungsfunktion wird dort nie
  variiert. Eine Arbeit, die Ungrokking mit der Aktivierungsfunktion verbindet, fand sich nicht. Am nächsten liegen: Mit
  der ungeraden Aktivierung x³ fiel die Genauigkeit eines MLP nach einem Höchststand wieder ab (Einzelläufe, arXiv
  2411.05353); ein für Addition modulo p trainiertes ReLU-Netz rechnet unverändert, wenn man ReLU nachträglich durch |x|
  oder x² ersetzt, mit x oder x³ nicht mehr, nutzt also nur den geraden Teil der Aktivierung (arXiv 2602.16849).
  Verzögertes Verallgemeinern gibt es auch ganz ohne verdeckte Nichtlinearität (lineare Schätzer, Kernmethoden).
* **Die Einbrüche des Trainings (8.6) sind ein bekanntes Phänomen**, beschrieben als „Slingshot" in fast genau diesem
  Aufbau (Thilak et al., arXiv 2206.04817: zwei Blöcke, Division modulo 97, Adam mit Lernrate 10⁻³, β = 0,9 / 0,98, ε =
  10⁻⁸, Stapel 512); mit Gewichtsabnahme verschwinden dort die Slingshots selbst, das Training bleibt nach Anhang C.1
  aber instabil (nach Abschnitt 3.3 nicht). Deutliche Slingshots trotz Gewichtsabnahme 1 sahen Nanda et al. in
  zweischichtigen Transformern (arXiv 2301.05217, Anhang D.2). Die Ursache ist umstritten: eine Krümmungsgrenze, die von
  ε und der Lernrate abhängt (Thilak et al.), ein zweites Moment von Adam, das sich vom Gradienten löst (arXiv
  2506.04805), der Rand der Stabilität (arXiv 2207.14484) oder Rundung in float32 in der Kreuzentropie (arXiv
  2605.06152; dort ohne Gewichtsabnahme durch einen Verlust in float64 beseitigt).
* **SwiGLU ist zur Hälfte exakt quadratisch.** Weil silu(g) − silu(−g) = g, ist der Teil einer SwiGLU-Einheit
  silu(g)·u, der sich beim Vorzeichenwechsel der Eingabe nicht ändert, genau g·u/2, ein Polynom vom Grad 2. Der Rest,
  (g/2)·tanh(g/2)·u, ist nahe g = 0 etwa g²·u/4 (Grad 3) und weit davon etwa |g|·u/2. Als Architektur kommt eine rein
  bilineare Einheit bis auf etwa 0,01 nats an SwiGLU heran (arXiv 2002.05202, 2410.08417). Wie viel eines trainierten
  SwiGLU-MLP ein Polynom vom Grad 2 tatsächlich erfasst, hat nach dieser Suche niemand gemessen.
* **Dass ein Netz seine eigene Aktivierungsfunktion oder den Zustand seiner eigenen Gates repräsentiert, zeigt keine
  Arbeit.** Die Studien zur Selbstbeobachtung betreffen Richtungen im Residualstrom oder Verhalten (arXiv 2505.13763,
  2410.13787, Lindsey 2025), und selbst diese sind 2026 bestritten worden (arXiv 2605.26242). Das Selbstmodellieren von
  Premakumar et al. (arXiv 2407.10188) sagt eigene Aktivierungen voraus, ohne das Ziel vom Gradienten abzutrennen, und
  hat als Kontrolle nur ein Netz ohne Selbstmodell.
* **Der Beitrag über kubische Polynome benutzt ein ReLU-Netz mit 64, 64, 64 und 15 Neuronen**, nicht vier Schichten zu
  15 (so stand es in 8.7; die 15 × 15 × 15 × 15 betreffen nur 49 Netze am Schluss des Beitrags, die je ein einzelnes
  Polynom in x lernen). Die neun Merkmale hat der Autor nach Augenschein ausgewählt; sein Test erklärt 88 bis 98,5 %
  ihrer Streuung mit x + c·a; dass sie nicht von b abhängen, beruht auf dem Vergleich zweier Bilder. 8.7 ist
  entsprechend korrigiert.
* **Der Satz über die Dreiteilende stimmt** (Pehlivan und Williams, Satz 1: R s³ + S s² + T = 0 ohne lineares Glied, für
  die Dreiteilende neben der Seite CA). Die abgedruckte Herleitung hat zwei Druckfehler, die den Satz nicht berühren.

### 9.2 Ungrokking und Aktivierungsfunktion (V1)

Aufbau wie in 8.6: Ein verallgemeinerndes Netz G (Grundlauf 500 Schritte nach der ersten Messung mit 99 %) wird auf 233
seiner 4.656 Trainingsaufgaben weitertrainiert (U, 5 %), die Driftkontrolle D mit allen, je 4.000 Schritte. Die neuen
Läufe werden alle 50 Schritte gemessen, die vier mit Lernrate 3·10⁻⁴ alle 150 (Gewichtsnorm und Stapelverlust auch dort
alle 50 Schritte); bei ReLU sind die beiden Läufe aus 8.6 zusammengesetzt (bis Schritt 1.500 alle 50 Schritte, danach
alle 250). Festgelegt vor der Auswertung: Ereignis = eine Messung oder mehrere aufeinanderfolgende (zusammen als ein
Ereignis gezählt), bei denen die Gewichtsnorm um mindestens 0,5 gegenüber der vorigen Messung steigt, der Stapelverlust
mindestens 0,1 beträgt oder weniger als 99 % der eigenen Trainingsaufgaben stimmen; Endstand = Median der Genauigkeit
auf ungesehenen Aufgaben über die Messungen von Schritt 3.500 bis 4.000; eine Driftkontrolle ist lesbar, wenn sie an
mindestens 90 % der Messungen und im Endstand bei 99 % oder darüber liegt. Das ist weiter gefasst als die Regel in 8.6,
nach der Driftkontrolle und behaltene Aufgaben durchweg bei 99 % bleiben mussten. Nach jener zählen dieselben ReLU-Läufe
als Instabilität des Aufbaus (Kurzfassung, Punkt 9), nach dieser ist ihre Driftkontrolle lesbar (an 93 bis 98 % der
Messungen bei 99 %).

**Mit ReLU, SwiGLU und der bilinearen Einheit geht die Verallgemeinerung verloren, auch ohne Schaltschwelle.**

| MLP (je Startwert ein Ausgangsnetz G) | Endstand des 5-%-Laufs, Startwert 0 / 1 / 2 | Driftkontrolle lesbar | Ereignisse im 5-%-Lauf bis Schritt 1.500 |
|---|---|---|---|
| ReLU, 512 Neuronen (Läufe aus 8.6) | 0,03 / 0,09 / 0,04 | ja, alle drei | je 2 |
| SwiGLU, 341 Einheiten | 0,03 / 0,02 / 0,04 | ja, alle drei (Startwert 1 genau an der Schwelle: 72 von 80 Messungen bei 99 %) | je 2 |
| rein bilinear, gate(x)·up(x), 341 Einheiten | 0,04 / 0,06 / 0,09 | ja, alle drei | je 2 |
| linear, down(up(x)), ohne jede MLP-Nichtlinearität | 0,07 / – | nein (an 66 von 80 Messungen bei 99 %) | 2 |

* Nach der vorab gesetzten Regel ist damit (a) „die Art der Nichtlinearität entscheidet, ob Ungrokking eintritt"
  widerlegt: Alle drei Arten liegen in allen drei Netzen unter 50 %. Das hängt knapp an der Driftkontrolle von SwiGLU,
  Startwert 1: Sie liegt an genau 90 % der Messungen bei 99 %; mit einer Messung weniger wäre sie unlesbar und (a) nach
  der Regel unentschieden. (b) „Ungrokking braucht eine Schwelle oder ein Gate" (gemeint ist ein Schalter wie
  sigmoid(gate(x)) bei SwiGLU) ist ebenfalls widerlegt: Die bilineare Einheit hat keines von beiden, ihr Zweig gate(x)
  wird nur multipliziert, und sie verliert die Verallgemeinerung genauso. (c) „Es braucht überhaupt eine
  MLP-Nichtlinearität" ist nach der Regel nicht entscheidbar: Von zwei Netzen ohne MLP-Nichtlinearität hat nur eines die
  Aufgabe in 20.000 Schritten gelernt (Startwert 0: 99 % bei Schritt 18.000; Startwert 1 lernte die Trainingsaufgaben
  auswendig und kam auf ungesehenen nie über 0,03). Das eine verliert die Verallgemeinerung ebenfalls (Endstand 0,07),
  seine Driftkontrolle ist aber unruhig (5 Ereignisse bis Schritt 1.500, tiefster Stand 0,68). Die Nichtlinearität
  steckt in diesem Netz nur noch in der Aufmerksamkeit und der Schichtnormierung.
* Die Ausgangsnetze unterscheiden sich in mehr als der Nichtlinearität (Gewichtsnorm 41 bis 64, Verlust auf ungesehenen
  Aufgaben 0,004 bis 0,034, Startgewichte). Für die Frage „ob" spielt das keine Rolle, weil alle Netze weit unter der
  Schwelle landen. Zur Beschreibung: Unter 50 % fällt jeder 5-%-Lauf an seinem ersten Ereignis, bei ReLU bei Schritt 750
  / 800 / 750, bei SwiGLU 450 / 600 / 500, bilinear 700 / 500 / 700. Einen Unterschied in der Geschwindigkeit benennt
  die Regel ohne Läufe ohne Einbrüche nicht.

**Ein verändertes Schaltmuster braucht der Verlust nicht (ReLU).** Drei Prüfungen, die ersten beiden nach vorab
gesetzten Regeln.

* *Eingefrorenes Schaltmuster (ReLU; ein anderes Modell, wie das entkoppelte Netz in 8.6).* Das weitertrainierte Netz
  rechnet mit den Masken einer eingefrorenen Kopie von G auf derselben Eingabe, in Training und Messung; die Gewichte
  können sich ändern, welche Neuronen schalten, nicht. Der 5-%-Lauf verliert die Verallgemeinerung trotzdem (Endstand
  0,09 / 0,11 / 0,11; ohne Einfrieren 0,03 / 0,09 / 0,04), die Driftkontrolle mit eingefrorenen Masken bleibt lesbar
  (Endstand 1,00). Nach der Regel spricht das in allen drei Netzen gegen „der Verlust braucht ein verändertes
  Schaltmuster". Mit eingefrorenen Masken kommt der erste Einbruch etwas später (Schritt 850 / 850 / 950 statt 750 / 800
  / 750; bei Schritt 750 noch 0,91 / 0,87 / 0,88 richtig), und das Netz entfernt sich gleich weit von G (relativer
  Gewichtsabstand bei Schritt 4.000: 0,49 / 0,48 / 0,51, ohne Einfrieren 0,48 / 0,44 / 0,46).
* *Tausch des Schaltmusters.* Schalter einer Einheit sind bei ReLU die Maske 1[fc1(x) > 0], bei SwiGLU der Faktor
  sigmoid(gate(x)) (silu(g) = sigmoid(g)·g). Gerechnet wird U mit dem Schaltmuster von G (r = zurückgewonnener Anteil
  des Verlusts) und G mit dem Schaltmuster von U (k = verlorener Anteil), an dem lesbaren Schritt mit dem größten
  Verlust, verglichen mit einem Netz, dessen Gewichte gleichmäßig zum Auswendig-Stand verschoben sind, bis es gleich
  viel verloren hat (r_ref, k_ref). Lesbar ist ein Schritt nur, wenn D bei mindestens 99 % liegt, die beiden Mischungen
  aus G und D bei mindestens 95 % und U mindestens 1,5 % verloren hat. *SwiGLU:* Die Mischungen mit D bleiben auch nach
  den Einbrüchen von D lesbar. Gerechnet ist der Tausch an allen gespeicherten Ständen (bis Schritt 1.500 alle 50
  Schritte, danach alle 500; die Regel verlangte alle 50 bis zum Ende). Der lesbare Schritt mit dem größten Verlust
  liegt bei 3.500 / 4.000 / 4.000; dort spricht das Ergebnis nach der Regel in allen drei Netzen gegen „der Verlust
  sitzt im Schaltmuster" (r = 0,01 / 0,00 / 0,02 bei r_ref = 0,00 / 0,00 / 0,00, k = 0,03 / 0,10 / 0,01 bei k_ref = 0,01
  / 0,01 / 0,01), bei Startwert 1 knapp (k = 0,101, Grenze 0,111). Dort hängt das an Schritt 4.000: Ohne ihn wäre der
  Schritt mit dem größten Verlust Schritt 600, mitten in einem Einbruch, in dem U auch die eigenen Aufgaben verliert
  (0,14 richtig, ungesehene 0,02); dort verliert G mit den Gate-Faktoren von U 0,42 des Verlusts (Referenz 0,01), 50
  Schritte später noch 0,02, und SwiGLU wäre unentschieden. Die Regel nennt für den üblichen Aufbau außerdem: Die
  lesbaren Schritte enden mit dem ersten Ereignis der Driftkontrolle, bei SwiGLU in allen drei Netzen Schritt 500. So
  gelesen spricht der Tausch bei Schritt 450 ebenfalls in allen drei Netzen dagegen (r = 0,03 / 0,28 / 0,25 bei r_ref =
  0,08 / 0,29 / 0,30, k = 0,00). Die vorgesehene Hauptlesung an Läufen ohne Einbrüche gibt es nicht, weil keine
  Einstellung solche Läufe lieferte. *ReLU:* Jeder Einbruch der Driftkontrolle sortiert deren Masken so um, dass die
  Mischungen danach unlesbar sind; lesbar sind nur Schritte vor dem ersten Sprung von U, auch wenn man alle späteren
  gespeicherten Stände bis Schritt 4.000 hinzunimmt. Nach der Regel wörtlich spricht das Ergebnis in allen drei Netzen
  dagegen, aber schwach: Bei Startwert 0 ist nur Schritt 550 lesbar, an dem U erst 0,02 verloren hat (r = −21, weil U
  mit den Masken von G viel schlechter rechnet, siehe unten; k = 0,00, r_ref = 0,06). In den beiden anderen Netzen ist
  der lesbare Schritt mit dem größten Verlust Schritt 50, eine kurze Delle gleich nach dem Wechsel auf 5 % (kein
  Ereignis; U bei 0,89 / 0,91, bei Schritt 400 wieder bei 0,99 / 0,98), nicht der spätere Sprung. An den lesbaren
  Schritten kurz vor dem Sprung (Startwert 1, Schritt 500 bis 650) verliert G mit den Masken von U 0,14 bis 0,26 des
  Verlusts (Referenz 0,00), mehr als die Regel für „dagegen" zulässt.
* *Ein Vorzeichen vor dem Sprung, aber nicht im Muster selbst (Beschreibung).* Bei ReLU passen die Gewichte von U schon
  vor dem ersten Sprung nicht mehr zum Schaltmuster von G: U mit den Masken von G fällt bis Schritt 550 auf 0,59 / 0,24
  / 0,45, während U selbst noch 0,98 / 0,98 / 0,96 richtig rechnet und D mit den Masken von G bei 1,00 bleibt (an jeder
  Messung von Schritt 50 bis 550 in allen drei Netzen unter dem Bereich von D). Umgekehrt rechnet G mit den Masken von U
  fast fehlerfrei weiter (bis Schritt 550 mindestens 0,995, bis zum Sprung mindestens 0,99). Die Gewichte von U arbeiten
  also schon vor dem Sprung nicht mehr mit dem alten Muster zusammen, die der Driftkontrolle schon; das Muster von U
  allein kostet G bis Schritt 550 höchstens 0,005, bis zum Sprung höchstens 0,008. Mehr Umschalten als in der Drift darf
  nach der Regel nicht gesagt werden (U müsste D in beiden Blöcken aller drei Netze um das 1,5-Fache übertreffen;
  Startwert 0: 0,9 und 1,2).

**Keine geprüfte Einstellung stellt die Einbrüche ab, und der größte Teil des Verlusts kommt mit ihnen.** Nach der Regel
sollte zuerst ein beruhigter Optimierer die Einbrüche beseitigen; dann wäre lesbar, ob der Verlust mit diesem Optimierer
die Einbrüche braucht, nicht mit dem ursprünglichen. Vorprüfung an Startwert 0, 5-%-Lauf und Driftkontrolle, 1.500
Schritte (bei kleinerer Lernrate 5.000, gleich viel Lernrate mal Schritte; dort sind die Ereignisse aus der Aufzeichnung
alle 50 Schritte gezählt):

| Einstellung | Ereignisse im 5-%-Lauf (Schritt) | in der Driftkontrolle | ungesehene Aufgaben im 5-%-Lauf |
|---|---|---|---|
| wie in 8.6 | 750, 1.400 | 600, 1.250 | 0,24 bei Schritt 1.000, 0,14 bei 1.500 |
| ε = 10⁻⁶ | 700, 1.300 | 600, 1.300 | 0,49 / 0,24 |
| ε = 10⁻⁵ | 650, 1.250 | 500, 1.000 | 0,55 / 0,27 |
| AMSGrad | 1.100 | 1.150 | 0,95 / 0,10 |
| Verlust in float64 (nachträglich, siehe unten) | 700, 1.400 | 600, 1.250 | 0,32 / 0,10 |
| Lernrate 3·10⁻⁴ | 1.900, 3.100, 4.200 | 1.700, 2.550, 3.900, 4.950 | 0,54 bei Schritt 3.300, 0,68 bei 4.950 |
| Lernrate 3·10⁻⁴, ε = 10⁻⁵ | 4.400 | 2.200, 4.900 | 0,95 bei Schritt 3.300, 0,85 bei 4.950 |

* Keine Einstellung bleibt ohne Ereignis; nach der Regel ist die Frage „braucht Ungrokking die Einbrüche?" damit
  unentschieden. Den Verlust in float64 hat die Literaturprüfung vorgeschlagen (arXiv 2605.06152), nachdem die Regel
  feststand; er ändert an den Einbrüchen fast nichts (dieselben Schritte bis auf 50). Die Einbrüche sind hier also
  kein Rundungsfehler der Verlustfunktion.
* In jeder Einstellung fällt die Genauigkeit auf ungesehenen Aufgaben dort, wo die Gewichtsnorm springt. Mit AMSGrad
  kommt der erste Einbruch erst bei Schritt 1.100, und der Verlust kommt mit ihm (bei Schritt 1.000 noch 0,95, bei 1.500
  0,10). An den Startwerten 1 und 2 wiederholt: erstes Ereignis bei Schritt 1.100 und 1.150, bei Schritt 1.000 noch 0,97
  und 0,94 richtig (ohne AMSGrad, 8.6: 0,35 und 0,29), bei 1.500 0,34 und 0,08; die Driftkontrolle bricht bei 1.250 und
  1.200 ebenfalls ein und ist bei 1.500 wieder bei 1,00. Mit Lernrate 3·10⁻⁴ und ε = 10⁻⁵ fällt sie in fünf Stufen (0,94
  / 0,93 / 0,84 / 0,70 / 0,69 bei Schritt 1.200, 2.100, 2.850, 3.600, 4.500), jedes Mal an einem Sprung der Gewichtsnorm
  um 0,25 bis 0,65, von denen nur der letzte die Ereignisschwelle 0,5 überschreitet; dazwischen erholt sie sich
  teilweise. Die Driftkontrolle hat ebenfalls Sprünge der Gewichtsnorm, aber nur drei und an anderen Schritten (2.250,
  3.700 und 4.900; 0,39 bis 0,81), und bleibt bei 1,00.
* Ein kleinerer Teil geht schon vorher verloren, ohne Ereignis und während die Gewichtsnorm gleichmäßig fällt. Vom
  höchsten Stand nach dem Wechsel auf 5 % bis zur letzten Messung vor dem ersten Ereignis sinkt die Genauigkeit auf
  ungesehenen Aufgaben bei ReLU um 0,04 bis 0,06, bei SwiGLU um 0,06 bis 0,19 (Startwerte 1 und 2: von 0,94 bei Schritt
  200 und 0,99 bei Schritt 100 auf 0,77 und 0,79), bilinear um 0,03 bis 0,11 und mit eingefrorenen ReLU-Masken um 0,10
  bis 0,16, bis zu etwa einem Fünftel des ganzen Verlusts. Von G aus gerechnet sind es bis zu 0,23: Mehrere Läufe
  verlieren in den ersten 50 Schritten nach dem Wechsel etwas und holen es danach ganz oder zum Teil zurück. Unter 50 %
  fällt jeder 5-%-Lauf erst an einem Ereignis.
* Mit Lernrate 3·10⁻⁴ und ε = 10⁻⁸ fällt die Genauigkeit einmal auch ohne Ereignis (auf 0,83 bei Schritt 750, danach
  wieder 0,96); bis Schritt 13.350 (gleich viel Lernrate mal Schritte wie 4.000 bei 10⁻³) liegt sie bei 0,11.
* Die Plankritik hatte eine Kontrolle mit kleinen Stapeln vorgesehen, ohne Entscheidungsregel: Die 233 Aufgaben passen
  in einen Stapel von 512, der 5-%-Lauf sieht also bei jedem Schritt dieselben Aufgaben, die Driftkontrolle nicht (die
  Lücke aus 8.6; nach arXiv 2605.06152 sind solche Einbrüche bei fast vollem Stapel zudem am stärksten). Mit Stapeln von
  26 Aufgaben (sonst wie in 8.6, Startwerte 0 bis 2) bleiben Einbrüche und Verlust: erste Ereignisse bei Schritt 650 /
  850 / 900, Endstand 0,05 / 0,05 / 0,04. Der Verlust tritt also auch ein, wenn nicht jeder Schritt dieselben Aufgaben
  sieht.

**Zusammengefasst für V1:** In diesem Kleinversuch entscheidet die Art der Aktivierungsfunktion nicht, ob Ungrokking
eintritt: ReLU, SwiGLU und eine Einheit ohne jede Schwelle verlieren alle die Verallgemeinerung. Ob es überhaupt eine
MLP-Nichtlinearität braucht, bleibt nach der Regel offen; das eine Netz ohne sie verliert die Verallgemeinerung zwar
auch, seine Driftkontrolle ist aber unruhig. Ein verändertes Schaltmuster braucht der Verlust nicht: Mit eingefrorenen
ReLU-Masken (ein anderes Modell) geht die Verallgemeinerung ebenfalls verloren. Ob der Verlust im Schaltmuster sitzt,
beantwortet der Tausch nur schwach: Bei ReLU spricht er nach der Regel dagegen, ist aber nur an frühen Schritten lesbar,
an denen U 0,02 bis 0,11 verloren hat; bei SwiGLU spricht er ebenfalls dagegen, in einem Netz knapp. Bei ReLU arbeiten
die Gewichte von U vor dem Sprung nicht mehr mit dem alten Muster zusammen. Der größte Teil des Verlusts kommt in jeder
Einstellung mit Einbrüchen oder kleineren Sprüngen der Gewichtsnorm; die Driftkontrolle hat sie auch, erholt sich aber
jedes Mal ganz, der 5-%-Lauf nur teilweise. Ein kleinerer Teil geht schon vorher allmählich verloren. Ob Ungrokking ohne
die Einbrüche einträte, bleibt offen, weil keine geprüfte Einstellung sie beseitigt. Für V1 spricht damit nichts;
dagegen spricht die Regel nur in diesen engeren Formen.

### 9.3 Selbstüberwachung: Sind die eigenen Gates aus dem Endzustand lesbar? (V2)

V2 lässt sich hier nicht als Ganzes prüfen: Jeder spätere Zustand eines Netzes ist eine Funktion seiner früheren
Aktivierungen, also auch seiner Gate-Werte; dass sie sich daraus ablesen lassen, zeigt keine Repräsentation der
Funktion SwiGLU. Ein Kleinversuch hat zudem keine Ausgabe, in der sich ein Selbstmodell zeigen könnte. Prüfbar ist nur,
ob der Endzustand etwas trägt, das *spezifisch* zu den eigenen Gates gehört (`act_selfprobe.py`, Ridge-Regression vom
Zustand des `=` nach dem letzten Block auf die Gate-Werte beider Blöcke am `=`, angepasst an den Trainingsaufgaben,
gelesen an den ungesehenen; alle gespeicherten Stände aller vier Startwerte).

| | SwiGLU | ReLU (Masken) | bilinear |
|---|---|---|---|
| eigene Gate-Werte, erklärter Anteil (trainierte Stände) | 0,27 bis 0,89 | 0,18 bis 0,61 | 0,30 bis 0,95 |
| eigene Gate-Werte, untrainiertes Netz | 0,99 bis 1,00 | 0,55 bis 0,58 | 0,99 bis 1,00 |
| eigene minus zufällige Gates auf denselben MLP-Eingängen | −0,06 bis +0,097 | −0,06 bis +0,097 | −0,05 bis +0,13 |
| aus den beiden Operanden allein (x und y getrennt kodiert) | 0,61 bis 0,98 | 0,41 bis 0,64 | 0,69 bis 0,98 |
| aus dem Endzustand eines anderen Netzes derselben Art | 0,24 bis 0,39 | 0,14 bis 0,23 | 0,25 bis 0,46 |

* Was der Endzustand über die eigenen Gates trägt, trägt er fast ebenso über Gates mit zufälligen Gewichten, die auf
  dieselben MLP-Eingänge angewandt werden: Der Unterschied liegt bei SwiGLU und ReLU in jedem Stand jedes Startwerts
  unter 0,10 (größter Wert 0,097). Klein, aber regelmäßig ist er in Block 1 der ReLU-Netze: In allen verallgemeinernden
  Ständen sind die eigenen Masken um 0,03 bis 0,09 besser lesbar. Bei SwiGLU ist der Gate-Eingang vor der SiLU praktisch
  so gut oder besser lesbar als der Gate-Wert danach (Unterschied −0,001 bis +0,05; in 7 von 42 Fällen ist er um
  höchstens 0,001 schlechter, das ist hier als gleich gezählt). Nach der vorab gesetzten Regel spricht das gegen eine
  spezifische Selbstrepräsentation. Die einzige Ausnahme ist ein Stand eines bilinearen Netzes (+0,13).
* Die beiden Operanden allein erklären die Gate-Werte fast immer besser als der Endzustand des Netzes selbst (bei SwiGLU
  in 41 von 42 Fällen). Der Endzustand trägt also, was in den MLP-Eingängen steckt, und davon weniger als die Eingabe.
* Der Trainingsarm, der ein Netz seine eigenen Gate-Werte vorhersagen lässt (eine Form des Selbstmodellierens nach
  Premakumar et al.), ist implementiert (`kv_grok.py`, Arme `selfmon`, `selfdet`, `othermon`), wurde aber auf Rat der
  Plankritik nicht gerechnet: Er prüft V2 nicht (er misst, wann ein Netz verallgemeinert), und die Kontrolle mit fremdem
  Ziel wäre in Dosis und Verlauf nicht gleich.

### 9.4 Wie quadratisch ist SwiGLU? (Voraussetzung von V3)

Gemessen wird je Einheit, wie gut ein Polynom vom Grad 1, 2 und 3 in ihren beiden Eingängen g = gate(x) und u = up(x)
(bei ReLU in fc1(x)) die Einheit über die Daten beschreibt: angepasst auf der einen Hälfte, gelesen auf der anderen,
gemittelt mit dem Gewicht, das die Einheit im Ausgang des MLP hat (`act_quad.py`, `act_qwen.py`). Vorab gesetzte Regel:
„ungefähr quadratisch" heißt in einem Block R²(Grad 2) ≥ 0,90 und R²(Grad 2) − R²(Grad 1) ≥ 0,20, im Kleinversuch
außerdem, dass das Netz mit Grad 2 höchstens 0,01 und mit Grad 1 mindestens 0,10 weniger richtig rechnet; „nicht
quadratisch" heißt R²(Grad 2) − R²(Grad 1) < 0,05 oder R²(Grad 2) < 0,70 oder, im Kleinversuch, mit Grad 2 mehr als 0,10
weniger richtig; dazwischen „teilweise".

**Im Kleinversuch rechnet der zweite MLP-Block als Polynom vom Grad 2, mit SwiGLU wie mit ReLU.**

| verallgemeinernde Stände, vier Startwerte | SwiGLU, Block 0 | SwiGLU, Block 1 | ReLU, Block 0 | ReLU, Block 1 |
|---|---|---|---|---|
| R² Grad 1 / 2 / 3 | 0,99 bis 1,00 / 1,00 / 1,00 | 0,73 bis 0,87 / 1,00 / 1,00 | 0,79 bis 0,99 / 0,96 bis 1,00 / 0,99 bis 1,00 | 0,75 bis 0,81 / 0,98 / 0,98 bis 0,99 |
| Anteil des quadratischen Glieds am Ausgang des MLP | 0,00 bis 0,01 | 0,11 bis 0,26 | 0,00 bis 0,22 | 0,10 bis 0,19 |
| ungesehene Aufgaben richtig, nur dieser Block linear genähert | 0,99 bis 1,00 | 0,03 bis 0,05 | 0,99 bis 1,00 | 0,03 bis 0,08 |
| ungesehene Aufgaben richtig, alle Einheiten durch Grad 2 ersetzt | 0,99 bis 1,00 | | 0,99 bis 1,00 | |

* Block 0 wirkt am `=` auf diesen Daten fast linear: Mit linear genähertem Block 0 rechnet das Netz weiter 0,99 bis 1,00
  richtig, auch in dem ReLU-Stand, in dem Block 0 nur zu 0,79 linear erklärt ist (quadratisches Glied 0,22). In Block 1
  macht das quadratische Glied nur 10 bis 26 % der Streuung des Ausgangs aus, aber an ihm hängt die Rechnung: Linear
  genähert verliert das Netz fast alles, mit Grad 2 nichts. Das gilt für ReLU genauso wie für SwiGLU. „Quadratisch" ist
  hier also eine Eigenschaft der Aufgabe (die Antwort braucht ein Produkt der beiden Operanden-Merkmale; so auch Nanda
  et al., arXiv 2301.05217: dort sind die meisten Neuronen gut durch Polynome vom Grad 2 in Sinus und Kosinus einer
  Frequenz der beiden Operanden beschrieben), nicht von SwiGLU. Nach der Regel ist Block 1 je nach Stand „ungefähr
  quadratisch" (SwiGLU 3, ReLU 2 der 8 Stände) oder „teilweise": R²(2) liegt immer über 0,90, R²(2) − R²(1) bei 0,13 bis
  0,26 (SwiGLU) und 0,17 bis 0,23 (ReLU), also um die Schwelle 0,20. Block 0 ist „nicht" quadratisch (bei ReLU in 2 der
  8 Stände „teilweise").
* Gate und Up einer SwiGLU-Einheit richten sich beim Lernen aneinander aus: Die Korrelation von g und u über die
  Aufgaben liegt untrainiert bei 0,11 bis 0,13 (Median über die Einheiten), in verallgemeinernden Ständen in Block 0 bei
  0,67 bis 0,93 und in Block 1 bei 0,39 bis 0,50. Viele Einheiten von Block 0 hängen damit fast nur von einer einzigen
  Linearkombination ihrer Eingabe ab (Einheiten mit einer Korrelation über 0,9 tragen 27 bis 53 % des Gewichts im
  Ausgang). Ihr Gate-Eingang ändert sich am `=` über die Aufgaben aber kaum (Streuung im Median 0,003 bis 0,007, in
  Block 1 0,27 bis 0,40). Deshalb wirken sie dort linear und nicht als Quadrat.
* Die rein bilineare Einheit ist in g und u schon gebaut als Polynom vom Grad 2 (R²(Grad 2) = 1), und ihre Netze
  verteilen die Rechnung genauso: Block 0 wirkt linear (R²(Grad 1) 0,99 bis 1,00), Block 1 ist zu 0,70 bis 0,84 linear
  erklärt, sein quadratisches Glied macht 0,10 bis 0,24 des Ausgangs aus, linear genähert rechnet das Netz noch 0,01 bis
  0,03 richtig. Gate und Up richten sich auch hier aneinander aus (Median der Korrelation in Block 0: 0,42 bis 0,85).

**In Qwen3-1.7B ist SwiGLU nur in Block 0 ungefähr quadratisch, und das knapp (R²(Grad 2) = 0,908; Block 16 verfehlt die
Schwelle 0,90 mit 0,898).** Angepasst an 4.082 Positionen aus 24 Wikitext-Absätzen, gelesen an 902 Positionen aus 16
anderen Texten, 8 Wikitext-Absätzen und 8 Chats (dieselbe Textauswahl wie in Abschnitt 7; beim Anpassen war kein Chat
dabei):

| Blöcke | R² Grad 1 / 2 / 3 | Einordnung nach der Regel | Gate-Eingang negativ |
|---|---|---|---|
| 0 | 0,14 / 0,91 / 1,00 | ungefähr quadratisch | 75 % |
| 1 bis 4 | 0,23 bis 0,26 / 0,43 bis 0,53 / 0,76 bis 0,86 | nicht quadratisch | 95 bis 98 % |
| 5 bis 9 | 0,22 bis 0,31 / 0,64 bis 0,72 / 0,92 bis 0,98 | 5 teilweise, 6 bis 9 nicht | 84 bis 90 % |
| 10 bis 19 | 0,23 bis 0,51 / 0,71 bis 0,898 / 0,92 bis 0,99 | teilweise | 57 bis 77 % |
| 20 bis 26 | 0,08 bis 0,18 / 0,56 bis 0,697 / 0,86 bis 0,92 | nicht quadratisch | 78 bis 87 % |
| 27 | 0,52 / 0,77 / 0,93 | teilweise | 89 % |

R²(Grad 2) liegt in Block 9 bei 0,698 und in Block 26 bei 0,697, knapp unter der Schwelle 0,70; die Einordnung folgt den
ungerundeten Werten.

* Wo die Gate-Eingänge fast alle negativ sind (Blöcke 1 bis 4), trägt das Glied vom Grad 3 so viel wie das vom Grad 2
  (Anteil am Ausgang des MLP 0,26 bis 0,38 gegen 0,21 bis 0,29). Die SiLU-Kurve hat bei g = −1,28 ihr Minimum. Gemittelt
  über normalverteilte Gate-Eingänge mit einer Streuung von 0,5 / 1 / 2 ist ihre Steigung null, wenn deren Mittelwert
  bei −1,44 / −1,90 / −3,57 liegt (Rechnung in `act_math_review.json`, M8). Dort fehlt einer Anpassung vom Grad 2 um
  diesen Mittelwert das Glied g·u: Der gerade Teil g·u/2 aus 9.1 und der Rest (g/2)·tanh(g/2)·u heben sich darin im
  Mittel auf. In den Blöcken 1 bis 9 sind 84 bis 98 % der Gate-Eingänge negativ (wie in 7.4); ob ihre Mittelwerte gerade
  dort liegen, ist nicht gemessen.
* Die Einheiten sind Produkte zweier fast unabhängiger Richtungen, keine Quadrate einer einzigen: Die Korrelation von g
  und u über die Positionen liegt im Median der Einheiten bei 0,06 bis 0,19, der Kosinus der beiden Gewichtsvektoren bei
  0,03 bis 0,10 (zufällige Vektoren dieser Länge: 0,018). Einheiten mit einer Korrelation über 0,9 tragen höchstens 4 %
  des Ausgangs eines Blocks. Ihr gerader Teil g·u/2 (9.1) hat seine Achsen auf den beiden Winkelhalbierenden von Gate-
  und Up-Richtung und, weil diese fast senkrecht stehen, fast gleich große Eigenwerte entgegengesetzten Vorzeichens
  (9.6). Die angepassten Polynome vom Grad 2 enthalten außerdem g² und u²; ihre Achsen sind nicht ausgewertet.
* *Ganz ersetzt.* Jede Einheit aller 28 Blöcke durch ihr Polynom vom Grad 2 ersetzt (angepasst an den 4.082 Positionen;
  die ersten vier Token und die Chat-Vorlage behalten das echte MLP, weil dort die Eingänge weit außerhalb des
  angepassten Bereichs liegen): Die Kreuzentropie des nächsten Tokens auf den 902 Prüfpositionen steigt von 3,63 auf
  5,80 (Grad 1: 8,06; jede Einheit durch ihren Mittelwert: 9,84), das wahrscheinlichste nächste Token bleibt an 35 % der
  Positionen dasselbe. Nur eine Gruppe von Blöcken ersetzt geht es besser: Blöcke 0 bis 9 mit Grad 2 KL im Median 0,44
  (gleiches Token an 58 %), Blöcke 19 bis 27 0,35 (61 %). In den Blöcken 10 bis 18 ist Grad 2 nur im Median und beim
  gleichen Token besser als Grad 1 (KL im Median 0,17 gegen 1,12, gleiches Token an 66 gegen 42 %). An mindestens einem
  Zehntel der Positionen liegt Grad 2 dort weit daneben (KL über 19), und im Mittel ist es schlechter (KL 3,71 gegen
  1,67, Kreuzentropie 6,87 gegen 4,54). Als Ganzes ist SwiGLU in Qwen also kein Polynom vom Grad 2, das man ohne Verlust
  einsetzen könnte; in den Blöcken 0 bis 9 und 19 bis 27 kommt Grad 2 näher heran als Grad 1, in den Blöcken 10 bis 18
  nur im Median und beim gleichen Token.

### 9.5 Der Beitrag über kubische Polynome, weitergedacht (V3)

V3 überträgt die Beobachtung des Beitrags auf SwiGLU: Ist die Einheit ungefähr quadratisch, sollten sich ihre Merkmale
verhalten wie die Merkmale im Beitrag, die von einer einzigen Kombination x + c·a abhängen, mit c nahe Cardanos 1/3.
8.7 hatte gefunden, dass diese Kombination dort liegt, wo sich die Zielfunktion selbst am stärksten ändert
(Hauptrichtung ihres Gradienten, c = 0,26), und dass sich das im Bereich des Beitrags nicht von 1/3 trennen lässt.
Vorab gesetzt war jetzt (`poly_direction.py`): Die Lesart „Eigenschaft der Aufgabe" gilt, wenn in mindestens 12 von
15 eigenen Netzen die Richtung, entlang der sich die letzte verdeckte Schicht ändert, innerhalb von 0,05 bei der
Hauptrichtung des Gradienten liegt (Grad 2, 3, 4) und bei doppelt so breitem Bereich von a in ein Band um die neue
Hauptrichtung mitwandert. Die Lesart „Algebra" gilt, wenn sie dann in mindestens 12 von 15 Netzen bei der Verschiebung
a/n bleibt, denn die hängt von der Verteilung der Eingaben nicht ab.

| Zielfunktion | Bereich von a | Hauptrichtung des Gradienten | Verschiebung der Algebra | eigene Netze 15-15-15-15 (ReLU, tanh, SiLU, je 5 Startwerte) | Netze in der Größe des Beitrags, 64-64-64-15, ReLU (5 Startwerte) |
|---|---|---|---|---|---|
| x² + a·x + b | ±3 | 0,414 | 1/2 | 0,39 bis 0,48; 14 von 15 innerhalb von 0,05 | |
| x³ + a·x² + x + b (8.7) | ±3 | 0,261 | 1/3 | 0,23 bis 0,31; 14 von 15 | 0,29 bis 0,33; 1 von 5 |
| x⁴ + a·x³ + x + b | ±3 | 0,200 | 1/4 | 0,16 bis 0,24; 15 von 15 | |
| x² + a·x + b | ±6 | 0,266 | 1/2 | 0,23 bis 0,66; 13 von 15 im Band 0,19 bis 0,34, keines bei 0,43 bis 0,57 | 0,26 bis 0,37; 4 von 5 im Band, keines bei 1/2 |
| x³ + a·x² + x + b | ±6 | 0,171 | 1/3 | 0,13 bis 0,26; 14 von 15 im Band 0,10 bis 0,25, keines bei 0,28 bis 0,39 | 0,19 bis 0,25; 4 von 5 im Band, keines bei 1/3 |

x und b liegen immer zwischen −3 und 3; jede Anpassung erreicht R² ≥ 0,9996.

* Nach der vorab gesetzten Regel gilt die Lesart „Eigenschaft der Aufgabe": Die Richtung der Netze liegt bei der
  Hauptrichtung des Gradienten, und wenn sich die Verteilung der Eingaben ändert, wandert sie mit. Bei der Verschiebung
  der Algebra liegt bei doppeltem Bereich kein einziges Netz. Für eine quadratische Zielfunktion ist die Richtung 0,414,
  nicht die 1/2 der quadratischen Ergänzung.
* Netze in der Größe des Beitrags (64-64-64-15, ReLU, wie im Beitrag; hier eigenes Training, ohne SAE, an den Neuronen
  der letzten verdeckten Schicht; von der Regel nicht erfasst, die für die 15 kleinen Netze gilt) liegen im Bereich des
  Beitrags bei 0,29 bis 0,33; nur eines liegt innerhalb von 0,05 bei der Hauptrichtung 0,26, vier liegen näher an 1/3.
  In den fünf Netzen sind 42 der 60 Neuronen der letzten Schicht, die überhaupt variieren, zu über 90 % Funktion einer
  einzigen Kombination (Median 0,32), wie im Beitrag (0,175 bis 0,35, Median 0,30). Bei doppeltem Bereich von a fällt
  die Richtung auf 0,19 bis 0,25: Sie wandert mit und bleibt nicht bei 1/3. In beiden Bereichen liegt sie 0,02 bis 0,08
  über der Hauptrichtung; dieser Abstand bleibt bei doppeltem Bereich im Mittel gleich (0,056; je Netz ändert er sich um
  höchstens 0,02, in drei der fünf Netze nach oben), obwohl 1/3 dort mehr als doppelt so weit entfernt ist (0,16 statt
  0,07). Die Nähe zu 1/3 im Bereich des Beitrags spricht deshalb nicht für Cardanos Verschiebung; woher der Abstand
  kommt, ist offen (fünf Netze).
* *Ein Produkt und eine SwiGLU-Einheit als Zielfunktion* (p·q + b und silu(p)·q + b, gleiche Netze und Bereiche). Hier
  hat schon die Zielfunktion keine Hauptrichtung: Die stärkere der beiden Richtungen trägt 0,50 und 0,54 des Gradienten
  (Verhältnis der Eigenwerte 1,00 und 1,19), und der Fächertest des Beitrags erreicht schon an der Zielfunktion selbst
  höchstens R² 0,55 (Produkt, bei 45°) und 0,36 (SwiGLU-Einheit, bei 44°), also entlang der Winkelhalbierenden, und auch
  dort wenig. In 14 von 15 Netzen je Zielfunktion folgen weniger als 10 % der Neuronen einer einzigen Kombination
  (insgesamt 3 von 210 und 3 von 207), und die Richtung der Schicht streut (Produkt: −53° bis 79°, Verhältnis der
  Eigenwerte 1,0 bis 2,4). Nach der Regel bestätigt das die Erwartung: Für ein Produkt gibt es kein x + c·a; seine
  natürlichen Achsen sind die beiden Winkelhalbierenden (9.6).
* *Das Gegenstück im Kleinversuch* (Beschreibung, `act_units.py`): Hängt ein MLP-Neuron am `=` von der Antwort allein
  ab, so wie ein Merkmal des Beitrags von einer einzigen Kombination? In verallgemeinernden Ständen nicht: In Block 1
  erklären die beiden Operanden getrennt 0,38 bis 0,44 und 0,35 bis 0,49 der Streuung der Neuronen (gewichtet wie in
  9.4), die Antwort über beide hinaus nur 0,04 bis 0,11, und kein Neuron erreicht 0,9 durch die Antwort allein. In Block
  0 der ReLU-Netze kommen einige Neuronen darüber (bei Startwert 0 mit 14 % des Gewichts im Ausgang bei Schritt 13.000
  und 3 % bei 13.500), aber nur, weil sie fast nur bei x = 0 anschlagen, wo auch die Antwort 0 ist; durch x allein sind
  sie ebenso gut erklärt. Der Zustand nach Block 1 ist dagegen zu 0,71 bis 0,89 von der Antwort über die Operanden
  hinaus bestimmt. Das gilt für ReLU, SwiGLU und die bilineare Einheit gleich (der vorab gesetzte Unterschied von 0,20
  zwischen den Architekturen wird nicht erreicht). Einzelne Neuronen hängen also kaum von der Antwort allein ab, der
  Zustand nach dem Block sehr wohl. Gemessen ist das am ganzen Zustand (mit Residualstrom und Aufmerksamkeit), nicht am
  Ausgang des MLP allein; dass die Antwort erst in der Summe der Neuronen entsteht, liegt nahe (ohne das quadratische
  Glied von Block 1 geht fast alles verloren, 9.4), ist hier aber nicht getrennt gemessen. Entlang des 5-%-Laufs (ReLU)
  trägt der Zustand weniger von der Antwort als in der Driftkontrolle, und zwar von der ersten Messung nach dem Wechsel
  auf 5 % an (Schritt 50: 0,50 bis 0,64 gegen 0,74 bis 0,80; Schritt 550: 0,61 bis 0,65 gegen 0,88 bis 0,90), an jeder
  Messung bis Schritt 550 in allen drei Netzen unter dem Bereich der Driftkontrolle. U rechnet bei Schritt 550 noch 0,96
  bis 0,98 richtig. Deutlich wächst der Abstand nur bei Startwert 0.
* *Für V3 folgt:* Die Beobachtung des Beitrags hängt nicht an quadratischen Einheiten: Sie zeigt sich in Netzen mit
  ReLU, tanh und SiLU und folgt der Zielfunktion und der Verteilung ihrer Eingaben. Netze aus SwiGLU- oder bilinearen
  Einheiten sind auf diese Richtung nicht geprüft. 8.7 ist entsprechend berichtigt: Das Netz des Beitrags hat 64, 64, 64
  und 15 Neuronen und ReLU, seine neun Merkmale sind nach Augenschein gewählt, und die eigenen Netze dort sind kleiner,
  nicht „derselben Größe".

### 9.6 Halbieren und Dritteln: was die Mathematik hergibt (V4)

Alle Aussagen sind numerisch geprüft (`trisect_math.py`, `trisect_math.json`) und von einem zweiten Prüfer unabhängig
nachgerechnet (`act_math_review.json`). Die meisten sind Lehrbuchwissen; neu zusammengestellt ist nur der Bezug.

* **Ein Produkt zweier Linearformen hat seine Achsen auf den Winkelhalbierenden.** Die quadratische Form (g·x)(u·x),
  also eine bilineare Einheit, hat höchstens zwei von null verschiedene Eigenwerte, |g||u|(cos φ + 1)/2 = |g||u|
  cos²(φ/2) und |g||u|(cos φ − 1)/2 = −|g||u| sin²(φ/2), mit den Achsen g/|g| + u/|u| und g/|g| − u/|u| (φ = Winkel
  zwischen g und u). Das ist die Halbwinkelformel. Zeigen g und u in dieselbe Richtung, ist die Einheit das Quadrat
  einer einzigen Linearkombination; stehen sie senkrecht, ist sie eine Differenz zweier gleich großer Quadrate. Unter
  Eingaben mit einer Kovarianz tritt an die Stelle von cos φ die Korrelation von g·x und u·x über die Daten.
* **Auf dem Kreis: zweite Harmonische für zwei Faktoren, dritte für drei.** Das Produkt zweier Linearformen mit den
  Richtungen ψ₁, ψ₂ ist auf dem Einheitskreis (cos(2ψ − ψ₁ − ψ₂) + cos(ψ₁ − ψ₂))/2: eine Konstante und eine zweite
  Harmonische (ein Anteil, der sich bei einem Umlauf um den Kreis zweimal wiederholt), deren Phase auf der
  Winkelhalbierenden liegt. Das Produkt dreier Linearformen enthält eine dritte Harmonische mit der Phase beim mittleren
  Winkel der drei (bis auf 120°) und drei erste Harmonische, die genau dann verschwinden, wenn die drei Richtungen 120°
  auseinanderliegen.
* **Winkelhalbierende und Dreiteilende im Dreieck sind dieselbe Aussage für zwei Tschebyschow-Polynome.** Tₙ ist das
  Polynom mit Tₙ(cos θ) = cos(nθ), also T₂(X) = 2X² − 1 und T₃(X) = 4X³ − 3X. Für die Linie von C, die mit CA den Winkel
  C/n bildet und bis AB die Länge sₙ hat, gilt Tₙ(h/sₙ) = cos(n·A + C − n·π/2), h die Höhe von C (geprüft für n = 2 bis
  5 an 5.000 Dreiecken, Fehler unter 4·10⁻¹³). n = 2 ist die Winkelhalbierende (Länge s₂): 2(h/s₂)² − 1 = cos(A − B),
  eine Quadratwurzel. n = 3 ist die Dreiteilende: 4X³ − 3X = sin(2A − B) mit X = h/s₃. Das ist die kubische Gleichung
  von Pehlivan und Williams (Arhimede Mathematical Journal 13/1, 2026, Satz 1: R s³ + S s² + T = 0 ohne lineares Glied),
  in 1/s geschrieben. Dass ihr in 1/s das quadratische Glied fehlt, liegt daran, dass sich n Kosinusse im Abstand 2π/n
  zu null addieren. Die kubische Gleichung gilt für die Dreiteilende neben der Seite CA; die andere erfüllt dieselbe
  Form mit vertauschten A und B, also 4X³ − 3X = sin(2B − A). Die abgedruckte Herleitung hat zwei Druckfehler (S. 16),
  der Satz stimmt.
* **Ein Polynom vom Grad n verliert durch t = x + a/n sein Glied vom Grad n − 1** (Ergänzen zum Quadrat bei n = 2,
  Cardanos erster Schritt bei n = 3). Die gedrückte Kubik auszuwerten heißt einen Winkel verdreifachen: Mit t = 2ρ cos φ
  und p = −3ρ² ist t³ + p·t = 2ρ³ cos 3φ. Sie zu lösen heißt dritteln: eine Quadratwurzel (aus der Diskriminante) und
  dann die Kubikwurzel einer komplexen Zahl mit bekanntem Betrag und Winkel θ; diese Kubikwurzel ist die Dreiteilung von θ.
  Die drei Wurzeln sind ein lineares Bild von (Mittelwert, ρ cos(θ/3), ρ sin(θ/3)), der diskreten Fourier-Transformierten
  über Z/3. Beim Grad 4 hört das auf: Eine gedrückte Quartik hat einen freien Koeffizienten mehr, als cos 4φ zulässt.
* **Was das für Netze heißt.** Eine Schicht bilinearer Einheiten ist ein Polynom vom Grad 2 ihrer Eingabe, zwei
  gestapelte Schichten sind eines vom Grad 4. Die Kubik des Beitrags lässt sich mit zwei bilinearen Schichten exakt
  darstellen (Breiten 3 und 2, in Horner-Form x·x·(x + a) + (x + b); die Richtungen sind x und x + a, nicht x + a/3).
  Cardanos Form bräuchte fünf echte Produkte statt zwei. Eine einzige bilineare Schicht ist für diese Kubik nicht besser
  als eine lineare Abbildung: Das beste Polynom vom Grad höchstens 2 ist die lineare Funktion 6,4x + 3a + b mit R² =
  0,7998, weil der Anteil x³ + a·x² der Kubik ungerade ist (er wechselt das Vorzeichen, wenn alle Eingaben es tun) und
  deshalb zu allen Gliedern vom Grad 2 senkrecht steht. Kleine Anpassungen bestätigen das nach der vorab gesetzten Regel
  (Breite 16, je 5 Startwerte, `poly_direction.py`): Eine bilineare Schicht erreicht R² 0,799 bis 0,804, zwei erreichen
  1,000; eine SwiGLU-Schicht erreicht schon 0,9999, weil silu(g)·u auch Glieder vom Grad 3 und höher enthält. cos(C/3)
  als Funktion von cos C hat bei C = 180° eine Wurzel-Singularität: Polynome vom Grad 2, 4 und 8 in cos C, nach
  kleinsten Quadraten über gleich verteilte Winkel C angepasst, haben dort ihren größten Fehler, 0,073 / 0,041 / 0,022
  (auch die bestmöglichen kommen nur auf 0,039 / 0,020 / 0,010), in der Halbwinkel-Variablen cos(C/2) nur 0,0015 /
  2·10⁻⁵ / 10⁻⁸.
* **Für die Vermutung V4 folgt daraus nichts Prüfbares.** Die Brücke zwischen „quadratische Einheit" und „Dreiteilung"
  ist eine Gradzählung: Zwei Faktoren geben die zweite Harmonische und die Winkelhalbierende, drei Faktoren die dritte
  und den Mittelwinkel; dritteln verlangt eine Kubikwurzel, die keine endliche Zahl von Produkten exakt liefert. Über
  ein trainiertes Netz sagt das nichts voraus.

### 9.7 Was folgt daraus

* **V1, Aktivierungsfunktion und Ungrokking:** Im Kleinversuch spricht nichts dafür, entschieden ist aber nur ein Teil.
  Die Art der Aktivierungsfunktion entscheidet nicht, ob Ungrokking eintritt, und eine Schwelle oder ein Schalter wie
  sigmoid(gate(x)) braucht es nicht. Ob es überhaupt eine MLP-Nichtlinearität braucht, ist offen. Ein verändertes
  Schaltmuster braucht der Verlust nicht (eingefrorene ReLU-Masken, ein anderes Modell). Gegen die Annahme, dass er im
  Schaltmuster sitzt, spricht der Tausch der Schalter bei ReLU schwach (lesbar sind nur frühe Schritte), bei SwiGLU in
  allen drei Netzen, in einem knapp. Bei ReLU arbeiten die Gewichte vor dem Sprung nicht mehr mit dem alten Schaltmuster
  zusammen, und der Zustand trägt von der ersten Messung an weniger von der Antwort als in der Driftkontrolle. Der
  größte Teil des Verlusts kommt mit Einbrüchen des Trainings, die auch die Driftkontrolle hat; sie erholt sich aber
  jedes Mal ganz, der 5-%-Lauf nur teilweise. Keine geprüfte Einstellung hat die Einbrüche beseitigt. Ob Ungrokking ohne
  sie einträte, bleibt offen.
* **V2, Selbstüberwachung und eine Repräsentation von SwiGLU:** Hier nicht prüfbar. Gemessen ist nur: Der Endzustand
  trägt über die eigenen Gates kaum mehr als über zufällige Gates auf denselben Eingängen und fast immer weniger als die
  Eingabe selbst. Bei SwiGLU und der bilinearen Einheit sind die eigenen Gate-Werte im untrainierten Netz am besten
  lesbar (0,99 bis 1,00, trainiert höchstens 0,89 und 0,95); bei ReLU liegen die Masken untrainiert bei 0,55 bis 0,58,
  trainiert bei 0,18 bis 0,61. Eine Repräsentation „von SwiGLU" im Sinn eines Modells der eigenen Funktion ließe sich so
  ohnehin nicht zeigen; dafür bräuchte es ein Netz, das über sich Auskunft geben kann, und eine Kontrolle, die nicht
  schon aus dem Aufbau folgt. Die Literatur zeigt so etwas auch nicht.
* **V3, SwiGLU ungefähr quadratisch, also gilt der Beitrag:** Die Voraussetzung trifft nur teilweise zu. Im Kleinversuch
  hängt die Rechnung im zweiten Block an einem quadratischen Glied, mit ReLU genauso; in Qwen3-1.7B ist nur Block 0
  ungefähr quadratisch, und das knapp, und als Ganzes lässt sich SwiGLU dort nicht ohne Verlust durch Grad 2 ersetzen.
  Der Schluss von „ungefähr quadratisch" auf „die Beobachtung des Beitrags gilt" trägt nicht: Die Beobachtung zeigt sich
  auch in Netzen ohne quadratische Einheiten (ReLU, tanh, SiLU) und folgt der Zielfunktion und der Verteilung ihrer
  Eingaben. Die fünf Netze in der Größe des Beitrags wandern ebenfalls mit, liegen aber in beiden Bereichen 0,02 bis
  0,08 über der Hauptrichtung des Gradienten, im Bereich des Beitrags vier von fünf näher an 1/3; woher der Abstand
  kommt, ist offen. Netze aus SwiGLU- oder bilinearen Einheiten sind auf diese Richtung nicht geprüft. Ist die
  Zielfunktion selbst ein Produkt, hat schon sie keine Hauptrichtung. Ihre quadratische Form hat die beiden
  Winkelhalbierenden als Achsen; die Netze finden keine gemeinsame Richtung (−53° bis 79°).
* **V4, Dritteln:** Die Mathematik stimmt und hängt sauber zusammen: Produkt zweier Linearformen, Winkelhalbierende,
  Halbwinkel und T₂ auf der einen Seite, drei Faktoren, Dreiteilende, gedrückte Kubik, Cardano und T₃ auf der anderen.
  Eine prüfbare Aussage über trainierte Netze folgt daraus nicht, und nichts in 9.2 bis 9.6 stützt oder schwächt V4.

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
* Abschnitt 8: ein Netz mit zwei Blöcken, eine Aufgabe, Gewichtsabnahme 0,3 (das Zehnfache des UFO-Papers), je Arm
  vier Startwerte, ein CPU-Thread. Das `=`-Token trägt nichts Eigenes, deshalb kann nur der Weg durch den Cache
  variieren, nicht die Menge der Nutzung; in einem Sprachmodell ist das anders. Die Schwellen für A wurden an vier
  Stellen verfehlt. Jede Duldsamkeits-Zahl ist eine Ziehung des Rauschens. Von den Auflagen der Plankritik zu den
  Trainingsarmen sind offen: fünf gepaarte Startwerte (gelaufen sind vier); der Abgleich der Arme nach Wirkung oder drei
  Größen je Ort (gelaufen ist eine Größe, nach Länge abgeglichen und über den Lauf nicht gleich groß); Schlüssel und
  Werte getrennt verrauscht; Messung alle 100 bis 250 Schritte und 99 % über drei Messungen (gemessen wurde alle 500
  Schritte, und die Läufe enden 500 Schritte nach den ersten 99 %; beim Rauschen im Cache, Startwert 1, liegt die letzte
  Messung wieder bei 98,99 %). Im Training liest neben dem `=` auch die Antwortposition dahinter den gestörten Cache.
  Alle Ungrokking-Läufe der Pfad-Analyse und der Verschiebung benutzen dieselben 233 Aufgaben. Die drei langen Läufe
  (spätere Schritte, Verschiebung des Cache) wurden von Hand nach 4.000 bis 5.250 Schritten beendet; die fein gemessenen
  enden nach 1.500. Die Driftkontrolle hat eigene Einbrüche; vergleichbar ist sie nur davor. Das entkoppelte Netz ist
  ein anderes Modell. Seltene Aufgaben wurden nur als Anteil je Stapel geprüft, nicht mit schiefer Verteilung über die
  ganze Trainingszeit. Nicht geprüft: K5 als Trainingsverfahren, ein UFO auf Schlüsseln und Werten, der UFO auf dem
  Lesemuster, eine rein bilineare Einheit (in Abschnitt 9 gelaufen, aber nicht auf Geschwindigkeit und Cache ausgewertet), andere Gewichtsabnahmen, ein zweites Modell. Die Gegenprobe zum Beitrag über kubische Polynome benutzt eigene Netze, nicht das des Beitrags (Abschnitt 9 rechnet Netze in seiner Größe nach, ohne SAE).
* Abschnitt 9: derselbe Kleinversuch (zwei Blöcke, eine Aufgabe, Gewichtsabnahme 0,3), je Architektur drei Netze; vom
  Netz ohne MLP-Nichtlinearität nur eines (das zweite lernte die Aufgabe nicht), und seine Driftkontrolle ist unruhig.
  Alle 5-%-Läufe benutzen dieselbe Ziehung der 233 Aufgaben; die vorgesehene zweite Ziehung je Architektur ist nicht
  gelaufen. Die Vorprüfung der Optimierer-Einstellungen lief an einem Startwert über 1.500 Schritte (AMSGrad an dreien,
  kleinere Lernrate über rund 5.000); SGD, Adagrad und eine viel kleinere Lernrate über lange Zeit sind nicht geprüft.
  Der Verlust in float64 kam nach der Regel hinzu; die kleinen Stapel hatte die Plankritik vorgesehen, ohne
  Entscheidungsregel. Der Tausch der Schaltmuster ist bei ReLU nur vor dem Sprung lesbar, nach Schritt 1.500 sind die
  Stände nur alle 500 Schritte gespeichert (die Regel verlangte alle 50), und die vorgesehene Hauptlesung an Läufen ohne
  Einbrüche gibt es nicht; eingefroren wurde nur das ReLU-Muster, und das eingefrorene Schaltmuster macht ein anderes
  Modell. Über Ungrokking in Sprachmodellen sagt der Kleinversuch nichts. Die Polynomanpassungen in Qwen gelten für die
  Textstichprobe (4.082 Positionen aus 24 Wikitext-Absätzen zum Anpassen, 902 aus 16 Texten zum Lesen); die
  Ersatzrechnung nimmt die ersten vier Token und die Chat-Vorlage aus. Die Gegenproben zum Beitrag über kubische
  Polynome benutzen eigene Netze, auch die in seiner Größe, ohne SAE. Der Trainingsarm mit Selbstüberwachung ist nicht
  gelaufen. Die Mathematik in 9.6 ist nachgerechnet; an Netzen geprüft ist nur die Gradzählung für bilineare Schichten
  (kleine Anpassungen an die Kubik), der Bezug zu Halbieren und Dritteln nicht.

## Nächster Schritt

Derselbe Versuch mit feinjustierten Modellen: Klasse = Modell (feinjustiert gegen Basis) auf denselben Fragen, je
Datensatz eine Wiederholung, Richtung = mittlere Verschiebung. Dann lässt sich prüfen, ob eine Verschiebung, die ein
Fine-Tune im Zustand bewirkt, im Basismodell das Verhalten auslöst, und ob eine Trennlinie zwischen den Modellen mehr
anzeigt, als das Netz daraus macht. Das braucht eine GPU.

Zu Abschnitt 7: (1) die Löschung im feinjustierten Modell mit der im Basismodell vergleichen (verschiebt LoRA die
Löschung, und entlang welcher Richtungen?); (2) den Trainingsverlauf prüfen, also ob der gewöhnliche Trainingsschritt
den UFO-Verlust senkt, im Kleinversuch mit mehreren Startwerten und der Gewichtsabnahme des Papers; (3) eine echte
Rückkopplung zwischen SAE und Netz, mit einem SAE, der für die unteren Blöcke auf dem Chat-Modell trainiert ist.

Zu Abschnitt 8: (1) dieselben Maße an einer Aufgabe, bei der die Antwort entweder aus dem Kontext oder aus den Gewichten
kommen kann (erst dort hat „mehr oder weniger Cache-Nutzung" einen Sinn); (2) die Störungs-Arme mit der Gewichtsabnahme
des Papers, mindestens acht Startwerten und einer Kontrolle, die nach Wirkung abgeglichen ist und über den Lauf gleich
groß bleibt; (3) versuchen, die Einbrüche des Trainings abzustellen (etwa mit kleinerer Lernrate oder größerem Epsilon
im Optimierer; ihre Ursache ist nicht untersucht), und prüfen, ob Ungrokking dann noch eintritt und wie es verläuft.

Zu Abschnitt 9: (1) Ungrokking mit einem Optimierer, der diese Einbrüche nicht hat (nach Thilak et al. etwa SGD oder
Adagrad, mit Gewichtsabnahme), und derselben Driftkontrolle; lesbar ist dann, ob der Verlust mit diesem Optimierer die
Einbrüche braucht, nicht mit dem ursprünglichen. Von SGD rät die Plankritik ab: Eine einzige Lernrate für Parameter,
deren Gradienten sich um das Tausendfache unterscheiden, macht daraus ein anderes Optimierungsproblem. Punkt (3) zu
Abschnitt 8 ist mit größerem ε, kleinerer Lernrate, AMSGrad, Verlust in float64 und kleinen Stapeln versucht worden,
ohne Erfolg. (2) Ob ein Netz seine eigene Aktivierungsfunktion abbildet, lässt sich nur an einem Modell prüfen, das über
sich Auskunft geben kann, mit einer Kontrolle, die nicht schon aus dem Aufbau folgt. (3) Wie quadratisch SwiGLU ist, an
einem zweiten Sprachmodell und an mehr Text; die Richtung aus dem Beitrag an Netzen aus SwiGLU- oder bilinearen
Einheiten.

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
| `kv_plan_review.json`, `kv_literature.json` | Plankritik und Literaturprüfung vor der Auswertung der Läufe von Abschnitt 8 |
| `kv_summary.json`, `kv_tables.txt`, `kv_logs.json` | alle Läufe von Abschnitt 8: Arme, Fortsetzungen, Messungen je Stand; Tabellen; Trainingsverläufe (darin auch der Vorlauf `pilot_sub0.1` und die Feinmessungen `check0`, `check0.02`) |
| `kv_probe_*.json` | Weg, Ordnung und Duldsamkeit je gespeichertem Stand eines Laufs (8.3–8.5, 8.7) |
| `kv_path_fine_s{0,1,2}.json`, `kv_transplant_s{0,1,2}.json`, `kv_disp_u1000_s{0,1,2}.json`, `kv_disp_drift3000_s{0,1,2}.json` | Pfad-Analyse und Verschiebung des Cache beim Ungrokking, mit Driftkontrolle (8.6) |
| `cubic_direction.json`, `cubic_direction.txt` | Gegenprobe zum Beitrag über kubische Polynome (8.7) |
| `review_kv.json` | die sieben Prüfberichte zu Abschnitt 8 und eine Notiz zu den Änderungen danach |
| `act_plan_review.json`, `act_literature.json`, `act_math_review.json` | Plan mit Plankritik und Entscheidungsregeln, Literaturprüfung und Nachrechnung der Mathematik (Abschnitt 9) |
| `act_report.json`, `act_verdicts.json`, `act_logs.json` | alle Läufe von Abschnitt 9 mit den festgelegten Kennzahlen, die Regeln angewandt, Trainingsverläufe (9.2) |
| `act_masks_{relu,swiglu}_s{0,1,2}.json` | Tausch des Schaltmusters (9.2) |
| `act_selfprobe_{relu,swiglu,bilinear}.json` | eigene Gates aus dem Endzustand (9.3) |
| `act_quad_{relu,swiglu,bilinear}.json`, `act_qwen.json`, `act_qwen_surrogates.json` | Polynomanpassungen im Kleinversuch und in Qwen3-1.7B, Ersatzrechnungen (9.4) |
| `act_units_{relu,swiglu,bilinear}.json`, `act_units_early_relu.json` | wovon einzelne Neuronen und der Zustand am `=` abhängen (9.5) |
| `poly_targets.json`, `poly_A.json`, `poly_B.json`, `poly_C.json`, `poly_D2.json`, `poly_D3.json` | Richtungen der Zielfunktionen und der eigenen Netze, Netze in der Größe des Beitrags, Tiefe (9.5, 9.6) |
| `trisect_math.json` | Nachrechnung der Mathematik (9.6) |
| `review_act.json` | Prüfberichte zum fertigen Abschnitt 9 |

Nicht im Repository: Zustände und SAE-Aktivierungen (zusammen etwa 500 MB), für Abschnitt 7 die Statistik echter
Zustände (`ufo_pool.pt`, 860 MB), die Aktivierungen auf dem Zahlenraster (700 MB) und die Gewichte der Kleinversuche
(Abschnitt 8: rund 770 MB, Abschnitt 9: rund 1,2 GB, dazu die Koeffizienten der Qwen-Anpassung mit 16 MB); sie entstehen beim Nachrechnen neu.

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
for m in base ufo; do TAIL=8000 python $S/ufo_grok.py $U $m 0 70000 0.3; done; python $S/ufo_grok.py $U perm 0 20000 0.3   # Kleinversuch, je 15 bis 30 min
for m in base ufo perm; do python $S/ufo_grok_probe2.py $U/grok_${m}_s0_wd0.3; python $S/ufo_grok_split.py $U/grok_${m}_s0_wd0.3; done
python $S/ufo_grok_report.py $U
# Abschnitt 8 (Kleinversuch zum KV-Cache); jeder Lauf mit einem Thread; Trainingsläufe 7 bis 35 min, Fortsetzungen 1 bis 11 min
K=<root>/kv_out; mkdir -p $K/runs
for s in 0 1 2 3; do for m in base kswap kswapiso kvnoise eqiso swiglu; do KVF=0.5 python $S/kv_grok.py $K/runs $m $s 20000 0.3; done; done
G=$K/runs/kv_base_s0/ckpt_13500.pt; M=$K/runs/kv_base_s0/ckpt_5000.pt     # G: letzter Stand des Grundlaufs (der Schritt kann auf einem anderen Rechner abweichen); M: Auswendig-Stand
# G, M und alle Zeilen mit _s0 stehen für Startwert 0; für Startwert 1 und 2: G=ckpt_10000 bzw. ckpt_14500 des jeweiligen Grundlaufs, M=ckpt_5000 desselben Laufs, Startwert-Argument und _s0 entsprechend
INIT=$G SUB=0.05 STRAT=1 EVAL=250 CKPT=500 TAG=_ungrok5 python $S/kv_grok.py $K/runs base 0 4000 0.3   # Ungrokking; DRAW=1, ohne STRAT, SUB=0.1: weitere Ziehungen
INIT=$G EVAL=250 CKPT=500 TAG=_drift python $S/kv_grok.py $K/runs base 0 4000 0.3                       # Driftkontrolle
INIT=$G SUB=0.05 STRAT=1 EVAL=50 CKPT=50 TAG=_ungrok5fine python $S/kv_grok.py $K/runs base 0 1500 0.3  # fein; ebenso TAG=_driftfine ohne SUB
MIX=0.02 TAG=_mix0.02 python $S/kv_untied.py $K/runs $G tied 0 4000 0.3                                 # selten statt fehlend (0.005 bis 0.2)
TAG=_untied python $S/kv_untied.py $K/runs $G writer 0 3000 0.3                                         # entkoppelt: writer, reader, both
INIT=$U/grok_base_s0_wd0.3/ckpt_20000.pt SUB=0.1 EVAL=250 TAG=_pilot_sub0.1 python $S/kv_grok.py $K/pilot base 0 6000 0.3   # Vorlauf aus 8.6
MIX=0.02 EVAL=25 TAG=_check0.02 python $S/kv_untied.py $K/check $G tied 0 400 0.3                       # Feinmessung aus 8.6; ebenso MIX=0 TAG=_check0
for d in $K/runs/kv_*_s?; do python $S/kv_probe.py $d; done; for m in base perm ufo; do python $S/kv_probe.py $U/grok_${m}_s0_wd0.3; done
# kv_probe.py schreibt <Lauf>/kvprobe.json; im Repository heißen diese Dateien kv_probe_<Arm>_s<n>.json, für die Läufe aus 7.10 kv_probe_old_grok_<m>_s0_wd0.3.json; kv_logs.json fasst die log.json aller Läufe zusammen
ONLY=path python $S/kv_transplant.py $G $K/runs/kv_base_s0_ungrok5fine $K/runs/kv_base_s0_driftfine $M $K/kv_path_fine_s0.json
STEPS=500,1000,1500,2000,3000,4000 python $S/kv_transplant.py $G $K/runs/kv_base_s0_ungrok5 $K/runs/kv_base_s0_drift $M $K/kv_transplant_s0.json
SKIP_MIX=1 STEPS=1000 python $S/kv_transplant.py $G $K/runs/kv_base_s0_ungrok5 $K/runs/kv_base_s0_drift $M $K/kv_disp_u1000_s0.json
SKIP_MIX=1 STEPS=3000 python $S/kv_transplant.py $G $K/runs/kv_base_s0_drift $K/runs/kv_base_s0_drift $M $K/kv_disp_drift3000_s0.json
python $S/kv_report.py $K/runs $K/kv_summary $U/grok_base_s0_wd0.3 $U/grok_perm_s0_wd0.3 $U/grok_ufo_s0_wd0.3 && python $S/kv_tables.py $K > $K/kv_tables.txt
python $S/cubic_direction.py $K/cubic_direction.json 5 8000 > $K/cubic_direction.txt                    # Gegenprobe zu 8.7, 8 min
# kv_queue.py verteilt eine Liste solcher Läufe auf die Kerne (ein Lauf je Kern)
# Abschnitt 9 (Aktivierungsfunktion); K wie oben; jeder Lauf mit einem Thread; Grundläufe 13 bis 30 min, Fortsetzungen 3 bis 15 min
# G: letzter Stand des Grundlaufs der jeweiligen Architektur und des Startwerts (ReLU wie oben, die anderen in der Zeile „G je Architektur"); M: Stand bei Schritt 5.000 desselben Grundlaufs (SwiGLU Startwert 2: 2.500). Die Zeilen mit Startwert 0 gelten ebenso für 1 und 2, außer der Vorprüfung mit ε und float64, der kleineren Lernrate und dem linearen Netz (nur Startwert 0); AMSGRAD=1 lief auch an Startwert 1 und 2.
A=$R/act_out; mkdir -p $A/runs
for s in 0 1 2 3; do python $S/kv_grok.py $A/runs bilinear $s 20000 0.3; done; for s in 0 1; do python $S/kv_grok.py $A/runs linear $s 20000 0.3; done
# G je Architektur: SwiGLU $K/runs/kv_swiglu_s{0,1,2}/ckpt_{11000,13000,8000}.pt; bilinear $A/runs/kv_bilinear_s{0,1,2}/ckpt_{8000,10500,7000}.pt; linear $A/runs/kv_linear_s0/ckpt_18500.pt
INIT=$G SUB=0.05 STRAT=1 EVAL=50 CKPT=500 TAG=_U python $S/kv_grok.py $A/runs swiglu 0 4000 0.3       # 5-%-Lauf; ebenso TAG=_D ohne SUB und STRAT; ebenso bilinear und linear
INIT=$G SUB=0.05 STRAT=1 EVAL=50 CKPT=50 TAG=_ungrok5fine python $S/kv_grok.py $A/runs swiglu 0 1500 0.3   # fein, für den Tausch; ebenso TAG=_driftfine ohne SUB und STRAT; beide ebenso bilinear
INIT=$G FROZEN=$G SUB=0.05 STRAT=1 EVAL=50 CKPT=1000 TAG=_fzU python $S/kv_grok.py $A/runs base 0 4000 0.3   # ReLU-Masken eingefroren; ebenso TAG=_fzD ohne SUB und STRAT
INIT=$G SUB=0.05 STRAT=1 EPS=1e-6 EVAL=50 CKPT=250 TAG=_scr_eps6_U python $S/kv_grok.py $A/runs base 0 1500 0.3   # Vorprüfung; ebenso EPS=1e-5 (_scr_eps5), AMSGRAD=1 (_scr_ams), F64=1 (_scr_f64), jeweils _U und _D ohne SUB und STRAT
INIT=$G SUB=0.05 STRAT=1 LR=3e-4 EVAL=150 CKPT=1500 TAG=_fb_lr3_U python $S/kv_grok.py $A/runs base 0 26667 0.3   # nach 13.350 Schritten von Hand beendet (gleich viel Lernrate mal Schritte wie 4.000 bei 10⁻³); ebenso _fb_lr3_D ohne SUB und STRAT über 13333 Schritte, nach 5.250 beendet; LR=3e-4 EPS=1e-5 CKPT=2500 als _fb_lr3e5_U und _fb_lr3e5_D (D ohne SUB und STRAT) über 5000
INIT=$G SUB=0.05 STRAT=1 BATCH=26 EVAL=50 CKPT=1000 TAG=_b26U python $S/kv_grok.py $A/runs base 0 4000 0.3    # kleine Stapel
STEPS=50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,2000,2500,3000,3500,4000 python $S/act_masks.py $G $M $K/runs/kv_base_s0_ungrok5fine,$K/runs/kv_base_s0_ungrok5 $K/runs/kv_base_s0_driftfine,$K/runs/kv_base_s0_drift $A/act_masks_relu_s0.json   # Tausch an allen gespeicherten Ständen; SwiGLU mit $A/runs/kv_swiglu_s0_ungrok5fine,…_U und …_driftfine,…_D
python $S/act_report.py $K/runs $A/runs $A/act_report.json && python $S/act_verdicts.py $A
for p in act_selfprobe act_quad act_units; do python $S/$p.py $A/${p}_relu.json $K/runs/kv_base_s{0,1,2,3}; python $S/$p.py $A/${p}_swiglu.json $K/runs/kv_swiglu_s{0,1,2,3}; python $S/$p.py $A/${p}_bilinear.json $A/runs/kv_bilinear_s{0,1,2,3}; done
STEPS=50,100,150,200,250,300,350,400,450,500,550,600 python $S/act_units.py $A/act_units_early_relu.json $K/runs/kv_base_s{0,1,2}_{ungrok5fine,driftfine}
python $S/act_qwen.py $R $A/act_qwen.json && COEF=$A/act_qwen.json.coef.pt python $S/act_qwen.py $R $A/act_qwen_surrogates.json   # Qwen, etwa 25 und 13 min
PARTS=1 python $S/poly_direction.py $A/poly_targets.json && TARGETS=deg2,deg4,deg2w PARTS=2 python $S/poly_direction.py $A/poly_A.json 5 8000 && TARGETS=prod,swiglu,deg3w PARTS=2 python $S/poly_direction.py $A/poly_B.json 5 8000
TARGETS=deg3 ARCHS=post PARTS=23 python $S/poly_direction.py $A/poly_C.json 5 8000 && for t in 2 3; do TARGETS=deg${t}w ARCHS=post PARTS=2 python $S/poly_direction.py $A/poly_D$t.json 5 8000; done
python $S/trisect_math.py $A/trisect_math.json
```
