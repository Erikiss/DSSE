# Runde 1b: gesammelte Eingaben (Stand 13.11.2026; noch kein Plan)

Runde 1b wiederholt die RSI-Schleife von Runde 1 mit verbesserten Messungen. Hier steht, was dafür zusammengekommen ist; der Plan
wird daraus erst geschrieben, gegengeprüft und eingefroren, bevor etwas läuft. Runde 2 (eingefroren) bleibt davon unberührt.

## 1 Schwächen von Runde 1 (eigene Analyse)
- Die Vorzeichen-Nullachsen werden aus ungezentrierten Differenzen je Prompt gebaut und enthalten so einen Teil der echten Achse:
  der p-Wert wird konservativ (Positivkontrolle knapp verfehlt). Abhilfe: zentrierte Pseudoachsen und ein gepaarter Vorzeichentest
  auf den Schleifenpaaren.
- Fester Schub α: die Schleife driftet in Richtungen mit viel Schaden (Tor GO3). Abhilfe: Schub auf gleichen Schaden normiert, dazu
  ein normerhaltender Schub als Kontrolle.

## 2 Linsen
- MicroLens (Zahedi, Zhu, Wu, arXiv 2611.13322): Zustände über Logit-Lens und J-Lens lesen; Swap-Edit (bewegt 1-5 % der Norm)
  statt Schub; „lesbar heißt nicht benutzt“.
- J++-Lens (Ayonrinde und Lindsey, LessWrong 2026): Experten-Jacobians je Aktivierungsgruppe, schlechte Gruppen ausgefiltert;
  LRP-Rückweg; Lesefilter ohne Satzzeichen. Zwischenschritte 55 % statt 36 % (J-Lens). Für Qwen3-1.7B gibt es keine veröffentlichte
  J++-Lens; selbst anpassen (geschätzt 30-60 min A100).
- MicroLens++ (Vorschlag dieser Sitzung): lokale statt gemittelte Jacobi-Matrix; Lese- und Gebrauchsrichtung getrennt, die
  Gebrauchsrichtung per Potenziteration; dieselbe Iteration auf den Unterraum der RSI-Wortrichtungen beschränkt: Verstärkung
  λ_RSI gegen die größte Verstärkung λ_max (vermutlich die Stilrichtung von Runde 1).

## 3 Außenstehender Probelauf (analysis/ju_space/microlens/, mit Gegenprüfung im README dort)
Die Hauptaussage ist nicht belegt, als Hypothese aber prüfbar. Für 1b: Gebrauchstest T4 mit Nullrichtungen bei gleichem Schaden,
mindestens 20 frequenzgleichen neutralen Wortmengen, Prompt-Bootstrap, Positivkontrolle je Fremdachse, erzeugtem Text und ohne „!“
in der Tonmenge; T3 gegen Text-Kontrast-Nullachsen und kreuzvalidiert; vorher prüfen, ob die J-Lens-Vorhersage die gemessenen
vorzeichentreuen Verschiebungen trifft.

## 4 Fünf Write-ups zu Gemma-Programmieragenten (gelesen und gegengeprüft; gemma_writeups.json)
Keiner sagt etwas über RSI-Schleifen; übrig bleiben Messregeln:
- Effekte zerlegen in Zusammensetzung (Anteil der Token-Arten) und Verschiebung innerhalb der Token-Arten.
- Rauschboden und Rauschkovarianz aus einem Lauf ohne Schub (α = 0); prüfen, ob die Stilrichtung die stärkste Rauschrichtung ist.
- Lese- gegen Gebrauchsrichtung bei gleichem Schaden; Wortlisten in Bau- und Bewertungshälfte teilen, Kontrollschichten mitmessen.
- Dosisreihe 0,25-2 × α (beide Vorzeichen) gegen die J-Lens-Vorhersage; Schichtprofil der Abweichung.
- Abtragen mit Mittelwert statt Null; neben dem KL die Rate geänderter Top-1-Token auf neutralen Stellen.
- RSI-Anteil und Verstärkung je Prompt und Prompt-Familie, nicht nur im Mittel; Herkunft (Code-, Konfigurations-, Prompt-Hash) in
  jeder Ergebnisdatei.

## 5 Drei Write-ups zu ARC Prize 2026 (Tufa Labs; gelesen und gegengeprüft; arc_writeups.json)
Nur Analogien zur Schleife; für die Durchführung:
- Ein Arm genau wie Runde 1, damit Unterschiede einzelnen Änderungen zuzuordnen sind.
- Text-Schleife und latente Schleife an derselben Größe vergleichen (Verstärkung eines Durchlaufs je Achse bei gleichem Schaden).
- Zu jedem Mittelwert die Verteilung (Schaden über Texte, Häufigkeit eines Effekts).
- Vorab festhalten, welchen kleinsten Effekt jede Prüfung erkennen kann.
- Schleife auch am 0.6B-Modell (Größenvergleich; nur als Hypothese).

## 6 Zwei Deutungen des Nutzers nach Runde 2 (13.11.2026; als Hypothesen in 1b zu prüfen)
- **Dualität bei sehr langem Kontext (Catfish-Effekt, Selfish Mining):** Wird der Kontext weit über das natürliche Maß gehalten,
  entsteht womöglich eine Kraft, die eine Entität in zwei teilt; die beiden können dann noch längeren Kontext tragen. Analogie
  Selfish Mining: ein zurückgehaltener Strang lohnt sich schon unter 50 % Anteil, für das Ganze ist es aber ein Verlustspiel.
  Runde 2 fand „Form“ relevant (AT8). Am Rande: Läge die Dualität nur innerhalb eines Tokens, würde Retokenisieren sie auflösen;
  symmetrische (König − Mann + Frau) gegen asymmetrische Analogien.
- **Kettenlinse:** Erzeugung als parallele Kanäle mit Token-Fronten; je Kanal, wie stark eine Front folgende gegenüber
  vorangehenden Fronten vorhersagt (eine Richtungsgröße); Kanäle hierarchisch, von äußeren Bedingungen umgeordnet. Die Linse bricht
  ab, wo sie halluziniert oder wo ihr Blick mit H-Neuronen (Halluzinations-Neuronen, rund 0,1 %) zusammenfällt. Frage: bloßer
  Längeneffekt oder ein H-Neuronen-Attraktor, der nicht an die Längenstatistik gebunden ist?
- Bezug: „The Chain Lens [interim]“ (vgel, Raymond Douglas; LessWrong, 11.11.2026,
  https://www.lesswrong.com/posts/Nr6fksEqqxNpdz6rk/the-chain-lens-interim): J-Lens-Transport plus MTP-Kopf, kontrastives
  Dekodieren gegen einen mittleren Zustand; WorkspaceBench 0,42 gegen 0,32 (J-Lens); Halluzination und Leistung wachsen beide mit
  der Kettenlänge; Neuron Lens mit Rundweg-Prüfung. Qwen3-1.7B hat keinen MTP-Kopf.
- Vorgabe des Nutzers: Wird etwas rückkonstruiert, dann inversfrei wie bei DeepMinds Aletheia (Erzeugen, vorwärts Prüfen,
  Überarbeiten oder Enthalten), nicht über invertierte Matrizen wie bei Fus Aletheia (regularisierte inverse Konfusionsmatrix).
- Frage des Nutzers (13.11.2026): Lassen sich die Erkenntnisse zu H-Neuronen und Kettenlinse durch einen linsenfreien Ansatz mit
  holographischen Prinzipien erweitern (vgl. EBM-Reasoner, Nicht-Gleichgewicht; Schleifen-Transformer → Agent → als Ising-System
  modellierbar → Kollaboration → entitätenbildend)? Wird als linsenfreier, inversfreier Arm geprüft: Wiederherstellbarkeit aus
  zufälligen Neuronen-Teilmengen (lokal gegen holographisch verteilt), wechselseitige Information zwischen Vergangenheit und Zukunft
  aus den Log-Wahrscheinlichkeiten des Modells, paarweises Maximum-Entropie-Modell (Ising) der H-Neuronen mit Energiemulden,
  Attraktor-Rekonstruktion aus einer Messgröße (Takens), Kreuzabbildung zwischen Kanälen, Token-Cluster über die Tiefe.
