# Paketdateien Runde 1b

Die Dateien werden vor dem Einfrieren auf der CPU gebaut. Dabei läuft kein Modell, nur der Tokenizer von Qwen3-1.7B. In
`MANIFEST.json` stehen zu jeder Datei Größe, sha256, Bauskript, Befehl, die bestandenen Prüfungen und die Quelldatensätze
mit ihren gepinnten sha256.

Der Code liest die Paketdateien aus `results/ju_space/plans/runde1b/` (`r1b_contracts.PKG`). Dieser Ordner ist die
Bereitstellungskopie.

| Datei | Teil | Wofür |
|---|---|---|
| `prompts_1b.json` | R | P256: die 64 Prompts aus Runde 1 (Indizes 0-31 und 128-159, wörtlich aus `rk_loop.py`) und 192 neue (96 zum RSI-Thema, 96 Alltagsszenen) samt Fold. Daraus entstehen die Achsen AX-1b und alle P256-Schritte von R. |
| `neutral_pool_1b.txt` | R | Der eingefrorene Neutralwort-Pool (628 Wörter, Anhang R.A). Aus ihm zieht r1b_R_S3 die 20 häufigkeitsangepassten Neutralmengen. |
| `wordsets_1b.json` | R | Die Wortlisten W_RSI (mit Bau- und Prüfhälfte), W_sea, W_sports und W_tone, mit sha256. |
| `label_bank_1b.json` | R | Die Kandidatenbank von R5.5 (Anweisungspaare pro/anti; der Wortlaut ist als Entwurf markiert). |
| `d1_stimuli.json` | D | Die Reize F-sym und F-asym für D1 (48 + 24 Referenten, Slots, Fassungen, Vorlagen, L4-Items). L nutzt sie mit (SO-D1STIM). |
| `d1v_pairs.json` | D | Die Wortpaare für D1-v: Geschlecht, Hauptstadt, Teil-Ganzes (je 24). |
| `d2_prompts.json` | D | 160 Szenen-Settings für D2, jedes genau 24 Tokens lang. |
| `d3_items.json` | D | Die Items für D3 (16 Kernitems und 8 Ersatzitems aus Kunstwörtern) und 24 neutrale Fragen. |
| `d4_roles.json` | D | 16 Rollen für D4: Name, Beruf, Ort und Familie, jeweils mit Alternative. |
| `longdoc_slices.json` | L | SO-LONGDOC: 24 Analyse- und 4 Pilotbücher aus dem PG19-Testteil, je der Token-Ausschnitt [2.000, 34.768) mit sha256. |
| `k1_label_bank/` | K | Die K1-Labelbank (SO-LAB). `bank.json` enthält ACH-short (7.000), ACH-long (400), FP, NE, MC, die TriviaQA-Aufteilungen (17.700 Fragen), die Rahmen P0/F512/F2048, die WikiText-Validierungsmengen samt 64 gepflanzter Prüfzustände und 200 Funktionswörter. Dazu kommen `unigram_logfreq.npy` (Unigramm-Häufigkeiten), `wt_train_ref.npy` (1 Mio. Train-Tokens als Perzentil-Referenz) und `sha256.json`. |
| `ref_k16.json` | K | REF-K16 (SO-DMG): 8 WikiText-Validierungspassagen (je 64 bewertete Tokens) und 8 TriviaQA-S2-Paare als Token-IDs. R, K und L messen daran den Schaden. |
| `MANIFEST.json` | alle | Prüfsummen und Herkunft aller Dateien. |

## Stand 14.11.2026

Alle Dateien sind gegen ihre Bauskripte und die Regeln der Teilpläne geprüft. Die Dateien von R, D und L ließen sich byte-gleich
neu bauen. Die K1-Bank wurde mit korrigiertem `r1b_k_labels.py` neu gebaut. Die Korrekturen:

- 64 gepflanzte Prüfzustände statt 26;
- Präambeln von ACH-long mit genau 400 Tokens;
- 200 verschiedene Funktionswörter;
- `sha256.json` ohne Eintrag für sich selbst.

`ref_k16.json` und die beiden `.npy`-Dateien blieben dabei unverändert.
