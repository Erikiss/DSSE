# KV-Cache gegen Gewichte: Ergebnis des A100-Laufs (8. November 2026)

Volle Ergebnisdateien der sechs Colab-Sitzungen, kopiert aus dem gemeinsamen Drive-Ordner `dsse_icl_vs_ft`
(173 Läufe: Basis, 96 Kontextläufe, 32 Finetuning-Läufe, 20 Lernkurven-Kontextläufe, 24 Lernkurven-Checkpoints).
Qwen3-1.7B, 16 Top- und 16 Bottom-Datensätze von `incorrect_answer`, 100 GSM8K-Eval-Prompts je Lauf.

| Datei | Inhalt |
|---|---|
| `table_per_dataset.csv` | je Datensatz: Kontext- und Finetuning-Metriken, DSSE-Scores |
| `tests.csv` | alle Korrelationstests mit p-Werten |
| `korrelat2_halbwert.csv` | Lernkurven: halbe Wirkung bei k Kontextzeilen bzw. Trainingsschritt |

Basis ohne Kontext und ohne Finetuning: 0,24 falsch, 0,28 Antworten mit höchstens zwei Sätzen,
0,10 Endzahl im ersten Satz, 396 Wörter je Ratgeber-Antwort.
