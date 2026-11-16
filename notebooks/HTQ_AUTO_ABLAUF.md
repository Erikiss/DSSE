# HTQ in Colab: automatisch weiterrechnen, ohne wartende GPUs

Stand 16.11.2026. Der Miet-GPU-Plan ist verworfen. START/K_S1 ist bereits fertig;
die Fortsetzung verwendet weiterhin eine A100 in Konto A und die getrennten HTQ-Ergebnisordner.

## Empfohlener Ablauf

1. **Einmal starten:** [HTQ AUTO in Colab](https://colab.research.google.com/github/Erikiss/DSSE/blob/codex/r1b-htq-continuation/notebooks/runde1b_HTQ_AUTO.ipynb) in Konto A, frische A100-Laufzeit,
   Drive anmelden, alle Zellen ausführen. Alle anderen GPU-Notebooks auf denselben
   Ergebnisordnern müssen beendet sein. Das Notebook übernimmt B2, B1, A1 mit D_C
   zuerst und A2 nacheinander auf derselben GPU. Danach folgen die bereits
   finanzierten GPU-Zusatzsitzungen, soweit ihre festgelegten Bedingungen erfüllt sind.
2. **L-CAL parallel auf CPU starten:** [HTQ L-CAL in Colab](https://colab.research.google.com/github/Erikiss/DSSE/blob/codex/r1b-htq-continuation/notebooks/runde1b_HTQ_LCAL.ipynb), Konto A, CPU mit
   hohem RAM. Dieser Job rechnet selbst; er wartet nicht auf eine freie GPU.
   Seine vollständige Kalibrierungsdatei muss vor der Auswertung geprüft und in
   den Repository-Stand übernommen sein. Fehlgeschlagene wissenschaftliche
   Kalibrierungstore werden dokumentiert, nicht automatisch auf bestanden gesetzt.
3. **Bei einer Laufzeitpause dasselbe AUTO-Notebook erneut starten:** Fertige
   Sitzungen und erfolgreiche Schritte werden übernommen. Eine neue A100-Zuteilung
   und gegebenenfalls die erneute Google-Anmeldung kann Colab nicht zuverlässig
   aus dem alten Notebook heraus erledigen. Dafür ist ein manueller Start nötig.
4. **Nach den GPU-Messungen die CPU-Auswertung starten:**
   [HTQ AUTO ANA in Colab](https://colab.research.google.com/github/Erikiss/DSSE/blob/codex/r1b-htq-continuation/notebooks/runde1b_HTQ_AUTO_ANA.ipynb) prüft Kalibrierung und Eingänge und erstellt
   Haupt- und Zusatzbericht nacheinander. Es wartet nicht in einer Schleife auf
   fehlende Messungen. Fehlende oder nicht korrekt übernommene L-CAL führt zu
   einem ausdrücklichen Halt vor der formalen Auswertung.

Das entspricht nach der bisherigen Kostenbasis typischerweise **drei Bedienrunden**:
Start von AUTO und CPU-Kalibrierung; einmalige GPU-Fortsetzung; Start der CPU-Auswertung.
Das sind **vier Notebook-Starts**, wenn genau eine GPU-Fortsetzung nötig wird.
Die Kalibrierungsprüfung und eine Ergebniskontrolle kommen hinzu. Zusätzliche
Drive-Fehler, Laufzeitabbrüche oder längere Messungen können weitere Eingriffe erfordern.
Es ist keine Zusage einer festen Anzahl von Klicks oder Neustarts.

## Wann nachsehen — und wann kein Wecker nötig ist

Die früheren manuellen Wechsel lagen nach alter Planbasis bei etwa +3 h 40,
+13 h 20, +16 h 06 und +18 h 42 ab B2-Start. Für diese Wechsel ist mit AUTO
kein Eingriff mehr nötig. Diese alten Zahlen sind keine gemessenen Weckzeiten.

Für die Nacht genügt als praktische Kontrollroutine ein Blick am nächsten Morgen,
etwa 8–10 Stunden nach dem Start; das ist kein geplanter Notebookwechsel.
Die erste geplante Fortsetzung richtet sich nach dem im Notebook gewählten
Gesamtzeitbudget, standardmäßig 20 Stunden in einer frischen Pro+-Laufzeit.
Bei einem Start um 00:00 wäre deshalb am folgenden Abend um etwa 20:00 ein
Kontrollblick sinnvoll. Wenn noch ein Messschritt läuft, nicht allein wegen der
Uhrzeit abbrechen. Lange einzelne Schritte können das weiche Zeitbudget überschreiten.

Nach einer sicher gespeicherten Pause kann die GPU freigegeben werden. Der
erneute Start darf später erfolgen: Dafür wird keine GPU wartend reserviert.
Es zählt der tatsächlich gemeldete Status, nicht eine aus der alten Kostenschätzung
abgeleitete feste Uhrzeit.

Die GPU-Hauptphase ist auf alter Basis mit 18–20 A100-Stunden angesetzt, alle
separaten finanzierten Zusatzsitzungen zusammen mit insgesamt etwa 27–29 Stunden.
Deshalb ist bei einem 20-Stunden-Budget mindestens eine Fortsetzung einzuplanen.
Diese Basis ist optimistisch und hat keine empirisch belegte Obergrenze; mehr
Fortsetzungen sind möglich. Kalibrierung, Setup und CPU-Auswertung sind zusätzlich.

## Was automatisch läuft

- Wechsel der logischen A-/B-Ergebnisordner und der Sitzungen, bei einer einzigen GPU.
- Wiederverwendung fertig gespeicherter Schritte und der lokalen Modell-Downloads.
- Fortsetzungen nach der bestehenden Sitzungszeitgrenze, solange noch Gesamtbudget besteht.
- Die vorgesehenen bedingten Auslassungen; fehlende oder gescheiterte Produzenten
  werden niemals automatisch als wissenschaftlich erledigt erklärt.
- Status/Protokoll in Drive, Prüfung der gesicherten Ergebnisse und Checkpoints,
  anschließend Anforderung der Laufzeitfreigabe, sofern die Sicherung bestätigt ist.

Es werden keine wartenden GPU-Notebooks vorgestartet und keine GPUs allein zum
Abwarten von Voraussetzungen belegt. Fehlen Eingänge, ist der Ablauf fertig oder
endet das Zeitbudget, hält die Automatik an. Bei einer fehlgeschlagenen Sicherung
bleibt die lokale Laufzeit erhalten, damit noch nicht gesicherte Daten nicht durch
automatisches Löschen verloren gehen; dieser Fall verlangt einen Eingriff.

Der GPU-Status liegt in Konto A unter
`dsse_runs/variants/htq_v2/_htq_auto/status.json`; die einzelnen Startversuche
haben darunter eigene Protokolle im Verzeichnis `runs`. Ein `budget_stop` mit
`safe_to_release: true` bedeutet: Zwischenstand geprüft, derselbe AUTO-Start kann
in einer frischen Laufzeit fortgesetzt werden. `complete` bedeutet: GPU-Warteschlange fertig.
Bei `sync_failed` die bestehende Laufzeit zur Datenrettung erhalten.

## Grenzen von Colab

Google nennt für Pro+ bei genügend Recheneinheiten bis zu 24 Stunden kontinuierliche
Ausführung; GPU-Verfügbarkeit und ein ununterbrochener Lauf sind nicht garantiert.
Der Screenshot des bisherigen Laufs zeigte Pro+. Bei einem anderen Tarif muss
das Zeitbudget entsprechend niedriger gewählt werden. Die Automatik verlängert
keine Plattformlimits und simuliert keine Benutzeraktivität.

Die Drive-Anmeldung muss funktionieren, bevor die Messungen beginnen. AUTO behebt
den aktuell beobachteten Fehler „credential propagation was unsuccessful“ nicht.
Die Freigabe verwendet die offizielle Colab-Funktion `runtime.unassign()`;
bei einem Fehler dieser Funktion muss die Laufzeit manuell beendet werden.

Quellen: [Colab FAQ](https://research.google.com/colaboratory/faq.html),
[Colab runtime.unassign](https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py).

Die Reihenfolge der GPU-Zusatzsitzungen bleibt innerhalb des eingefrorenen Plans:
Hauptplan §4.4 erlaubt Stufe 2 nach Finanzierung und ihren in §4.2 genannten
Auslösern. Die GPU-Zusatzsitzungen benötigen keinen vorherigen CPU-Gesamtbericht;
der Zusatzbericht selbst benötigt den Hauptbericht. Messungen, Daten, Tore,
Stichproben, Seeds und numerische Einstellungen werden durch diese Automatik nicht geändert.
