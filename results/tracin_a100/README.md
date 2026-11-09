# TracIn auf DSSE `incorrect_answer` (A100-Lauf vom 9. November 2026)

Qwen3-1.7B, LoRA wie im DSSE-Paper, 9 Datensätze, drei Colab-Sitzungen (`notebooks/tracin_a100_shard{0,1,2}.ipynb`).
Auswertung und Deutung: `research/DOSSIER_kv_cache_vs_weights.md`, Abschnitte 4.8 bis 4.10.

| Datei | Inhalt |
|---|---|
| `table_per_dataset.csv` | je Datensatz: Fehlerquote frei und mit erzwungenem Rechenweg, Wörter, Anteil ≤ 40 Wörter, Gradienten-Kosinus `format`~`math_cot` (Adam-Metrik, lernratengewichtet), kleinster Kosinus der Aufgaben-Hälften, Halbwertsschritte, Testgrößen bei Schritt 0 und 76, aufsummierte Vorhersage gegen gemessene Änderung, Spektrum-Anteile; `ft_wrong_alt` = Fehlerquote im Hauptlauf |
| `validation.csv`, `dsse1-*__validate.json` | Neutraining bei 08572, 08152, 08036: ohne die 30 einflussreichsten Zeilen je Testgröße, ohne die 30 kürzesten, 3 × 30 zufällige, andere Reihenfolge, bei 08152 ohne die Längenbitten |
| `context_test.json` | Kontext von dsse1-08152 (Auswahl 0) mit und ohne Längenbitten, eine einzelne Bitte, Basis |

Die vollständigen Rohdateien je Datensatz (Einfluss jeder Zeile, Kurven, Spektren, 100 Antworten) liegen im
freigegebenen Drive-Ordner `dsse_icl_vs_ft/tracin_runs_s{0,1,2}_of3/` (je etwa 0,8 MB).
