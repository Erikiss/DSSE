# Neuauswertung der Einzelantworten des KV-Cache-gegen-Gewichte-Laufs (9. November 2026)

Skripte und Ergebnisse der Neuauswertung aller 16.300 verfügbaren Einzelantworten (163 von 173 Läufen) aus sieben
Blickwinkeln mit Gegenprüfung. Grundlage des Korrekturkastens in `research/DOSSIER_kv_cache_vs_weights.md`, Abschnitt 4.

- `spark_audit.json`: alle Befunde, Gegenprüfungen und die abschließende Kritik (strukturiert).
- `load_runs.py`: lädt die Laufdateien; die Rohdaten (`runs/*.json`) liegen im freigegebenen Drive-Ordner `dsse_icl_vs_ft/runs/`.
- Unterordner je Blickwinkel (`measurement`, `competence`, `error_anatomy`, `statistics`, `triggers`, `dynamics`,
  `dsse_crosscheck`, `critic`) und je Gegenprüfung (`verify_*`).

Die Skripte wurden in einer Cloud-Sitzung mit absoluten Pfaden geschrieben; vor dem erneuten Ausführen `load_runs.py`
und die Pfade auf die lokale Kopie der Laufdateien umstellen. Große Zwischendateien (`.pkl`) sind nicht enthalten.
