# K3a/K3b/K3c aus vorhandenen H_tq-Daten auswerten

[CPU-Notebook in Colab öffnen](https://colab.research.google.com/github/Erikiss/DSSE/blob/codex/r1b-htq-continuation/notebooks/runde1b_HTQ_K3_CPU.ipynb),
CPU als Laufzeittyp wählen, dann **Alle ausführen**. Eine sichtbare GPU führt
vor Anmeldung und großem Download zum Abbruch. Es werden keine Modelle geladen,
keine Antworten erzeugt und keine bestehenden Sitzungsmarker verändert.

Die automatische Quellsuche verwendet zuerst bereits erreichbare Drive-Pfade.
Sie führt selbst keinen Drive-Mount aus. Sonst erfolgt eine Anmeldung für die
Drive-API mit Leseberechtigung; die Suche prüft den kanonischen Pfad
`variants/htq_v2/r1b_K_S1/k/k3/k3a_data.pt` und den vollständigen Quellkontext.
Mehrere gültige Kopien mit derselben SHA256 sind zulässig. Bei verschiedenen
Prüfsummen muss die gewünschte private Datei-ID in `DATEI_ID` eingetragen werden.
Diese ID umgeht weder Anmeldung noch Dateifreigaben. Bei fehlendem Drive-Scope
oder `credential propagation` enthält das Diagnose-ZIP den Fehler; die Anmeldung
muss dann in einer frischen CPU-Laufzeit funktionieren. Es werden keine Freigaben geändert.

Die große Rohdatei wird in 16-MiB-Schritten unter `/content/htq_k3_cache` abgelegt.
Hashgebundene Kontextdateien und die vollständige Roh-SHA256 werden vor dem
Laden geprüft; ein bestätigter lokaler Cache wird wiederverwendet. Die
Analyse verwendet anschließend die im ausgecheckten Commit geprüften Quellen.
Die Originalordner sind ausschließlich Eingabe.

Ergebnisse liegen pro Start in einem neuen Unterordner von
`/content/htq_k3_output`: `status.json`, `results.json`, `summary.txt`,
`data_audit.json`, `scores.csv` und `compact_inputs.npz`. Die kompakte Datei
enthält die für K3a/b/c benötigten gespeicherten Spalten für weitere Nachrechnungen.
Das Notebook lädt `htq_k3_results.zip` automatisch herunter. Falls der Browser
dies verhindert, dieselbe Datei links im Colab-Dateibrowser herunterladen.
Bei Fehlern gibt es stattdessen ein Diagnose-ZIP mit Status und Protokoll.

Es handelt sich um eine eigenständige Auswertung vorhandener Daten. L-CAL und
eine fertige Gesamt-ANA sind dafür nicht erforderlich. Ihre Ergebnisse ersetzen
keinen noch fehlenden Kontrolllauf oder den abschließenden K_ana-Verbundbericht.
