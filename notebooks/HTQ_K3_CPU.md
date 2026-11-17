# K3a/K3b/K3c aus vorhandenen H_tq-Daten auswerten

Bei dem bereits beobachteten Anmeldefehler **credential propagation**:
[LOKAL-Notebook ohne Drive-Anmeldung öffnen](https://colab.research.google.com/github/Erikiss/DSSE/blob/codex/r1b-htq-continuation/notebooks/runde1b_HTQ_K3_LOKAL.ipynb).
CPU wählen, links im Colab-Dateibrowser die vorhandene `k3a_data.pt` und das
bereitgestellte kleine `htq_k3_context.zip` nach `/content` hochladen; alternativ
nach `/content/htq_k3_upload`. Den großen Upload vollständig abwarten, dann
**Alle ausführen**. Es gibt keinen JavaScript-Großupload, keine Anmeldeversuche
und keine Warteschleife. Fehlende Dateien führen sofort zu einer konkreten
Dateibrowser-Anleitung. `EINGABE_MODE="LOKAL"` ist in diesem neuen Notebook vorgegeben.
Android-Mehrfachdownloads wie `k3a_data (1).pt` werden erkannt. Falls das Gerät
einen anderen Namen vergibt, den vollständigen lokalen Dateipfad in `INPUT_FILE`
eintragen. Bei mehreren Rohdateien wird nur eine zur Kontext-SHA256 passende
Kopie verwendet; die Prüfung vor dem Laden bleibt erhalten.

Das Kontext-ZIP enthält `session_done.json`, `hn/htq_policy.json`,
`hn/hn_set.json`, `sg8_preview.json` und den hashgebundenen
`k/k3/k3a_data.pt.prov.json`-Sidecar; `steps/k3a.json` kann ebenfalls enthalten sein.
Ein gemeinsamer Prefix `r1b_K_S1/` ist zulässig. Der kleine Kontext wird direkt
aus dem ZIP gelesen, ohne beliebige ZIP-Pfade zu entpacken. Rohdatei, H_tq-Policy
und Producer-Hashes werden auch in diesem lokalen Eingang vollständig geprüft.
Die lokal hochgeladenen Originaldateien werden ausschließlich gelesen.

Für eine funktionierende Drive-Anmeldung bleibt der bisherige AUTO-Start verfügbar:
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
oder `credential propagation` enthält das Diagnose-ZIP den Fehler. Auch eine
frische Laufzeit garantiert keine erfolgreiche Anmeldung; der lokale Start oben
kommt vollständig ohne diese Anmeldung aus. Es werden keine Freigaben geändert.

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
