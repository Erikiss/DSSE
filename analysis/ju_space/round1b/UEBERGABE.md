# Runde 1b: Übergabe an eine lokale Claude-Code-Sitzung (Stand 13.11.2026, 23:50 MEZ; wird aktualisiert)

## Lage
- Branch: `ccr-6fe02b21-est040` (Repository Erikiss/DSSE). Immer zuerst `git pull`.
- Plan (Entwurf, nicht eingefroren): Arbeitskopie `results/ju_space/plans/runde1b/arbeitsstand/plan/` (fünf Dateien:
  `runde1b_plan.md` = Rahmen, `runde1b_plan_teil_R/D/K/L.md` = Teile). Ältere Momentaufnahme: `results/ju_space/plans/runde1b/`.
- Zusätze, die in den Plan/Code gehören: `results/ju_space/plans/runde1b/arbeitsstand/review_notes_K.md` (atlas-kompatible
  H-Neuronen-Kennzahlen), `review_notes_K_mtp.md` (Soft-MTP nach Teoh 2026; RK-1 mit zwei Köpfen), ungeprüfte Befunde der
  abgebrochenen Gegenprüfung in `arbeitsstand/findings/` (nur Hinweise).
- Budget-Entscheidung des Nutzers: Stufe 1 und Stufe 2 finanziert; Teil P (Pythia-Brücke) ebenfalls Stufe 2 (Entwurf fehlt noch;
  Recherche in `arbeitsstand/findings/P_research.json`).
- Umsetzung (läuft in der Cloud-Sitzung bis das Kontingent endet): Code in `analysis/ju_space/round1b/`, Notebook-Bau in
  `notebooks/build_runde1b_gpu_notebooks.py`. Reihenfolge: Kern (r1b_contracts.py, r1b_core.py, r1b_run_session.py,
  r1b_run_slot.py, Notebook-Bau, IMPLEMENTATION.md) -> Teile R, K, D, L (r1b_sessions_<Teil>.py, r1b_<teil>_*.py, Mock-Selbsttests)
  -> Integrationstest im Mock-Modus (tests/integration_cpu.txt) und Rundenauswertung r1b_ana.
  Was davon fertig ist, steht im Abschnitt „Stand“ unten (die Cloud-Sitzung aktualisiert ihn).

## Regeln (gelten auch lokal)
- Runde-1-Code bleibt byteidentisch (rk_loop.py und die Runde-1-Module); Runde-2-Module nicht ändern, nur importieren.
- Kein echtes Modell lokal testen außer bewusst gewollt; Selbsttests im Mock-Modus (MOCK=1).
- Nichts von goodfire.com / goodfire.ai automatisiert abrufen; keine Schlüssel in Dateien oder Commits.
- Commits mit den bisherigen Trailern; nach jedem Arbeitsschritt committen und pushen.

## Nächste Schritte (Stand 14.11. 16:00 MEZ; lokal, in dieser Reihenfolge)
1. L-CAL: Punkt (a) ist durchgefallen (Treffer localist 0/5, redundant 0/5, dense 0/5, dthr 5/5, xor3 0/5). Ursache in
   r1b_l_cal.item_a und r1b_l_rules / r1b_l_stats suchen (Klassifikationsregel, Schwellen, Signalstärke der synthetischen Profile),
   korrigieren, Mock-Selbsttest von Teil L; dann die volle L-CAL neu rechnen (Cache leer):
   python results/ju_space/plans/runde1b/arbeitsstand/lcal/lcal_resumable.py results/ju_space/plans/runde1b/l_cal.json <cache-ordner>
   (mehrere CPU-Stunden; Punkt a allein etwa 70 Min. mit 2 Kernen). Ziel: G_cal_L1..L5 true; sonst Befund dokumentieren, Nutzer fragen.
2. K0-Prüfung offset4096 (Top-1 gleich, max |dlogp| 1,03e-3): Regel in Teil K (K0-Liste) nachsehen, Folge in den Datensatz.
3. Hochrechnung mit den P0b-Raten (results/ju_space/plans/runde1b/arbeitsstand/p0b_summary.json; Skripte
   arbeitsstand/freeze_prep_proj/cost_*_measured.py anpassen; engine.fast an): R nach RG18 (RL1 / RL2), D-Kürzungen T1.. nur solange
   über 13,8 h, K-Abwurfreihenfolge nur bei > 30 % langsamer, L-Kürzungen falls eine Rate unter der Hälfte des Planwerts liegt
   (L-Decode 60-81 ms je Schritt statt 25-35 ms prüfen).
4. runde1b_freeze.json aus arbeitsstand/runde1b_freeze_DRAFT.json: freeze_commit, freeze_time, plan_files (fünf Dateien, echte Namen
   runde1b_plan.md, runde1b_plan_teil_R/D/K/L.md), package_files (alle Paketdateien im Planordner inkl. pilot_material.json,
   r1b_keys.json, l_cal.json, k1_label_bank/), pilot (gültige P0-Punkte + P0b), trims, projection_h, funded {tier2: true, items: []},
   decisions {OD-M14: full_grid}, datasets, d_streams_sha256 (oberste Ebene), models, jlens, engine {D: {B_long: 3, B_mid: 7},
   fast: {l_decode_nosync, lce_inplace_kv, k_gumbel_kept_rows: true}}, amendments (D-Deckel 13,8 h, Entscheidung des Nutzers
   14.11.2026). Plantext 9.1 um engine-Schlüssel, Pins-Datei runde1b_pilot_pins.json und p0b_engineering.json ergänzen.
5. Einfrieren (Reihenfolge laut arbeitsstand/freeze_prep.json): Plan-Arbeitskopie arbeitsstand/plan/*.md nach
   results/ju_space/plans/runde1b/ kopieren, sha256 der Teildateien in 9.1 eintragen, Freeze-Commit (Plan + Paketdateien); im
   nächsten Commit runde1b_freeze.json mit diesem freeze_commit und die Konstanten FROZEN_PLAN_SHA256 / FREEZE_RECORD_SHA256 in
   r1b_run_session.py; Notebooks neu bauen (python notebooks/build_runde1b_gpu_notebooks.py); pushen.
6. Optional vorher: restliche 23 Sitzungen des Mock-Integrationstests (tests/integration_cpu.txt; Treiber tests/integ_run2.py und
   tests/integ_check.py, Pfade darin anpassen).
7. Nutzer: Freigabe r1b_xfer A -> B (Verknüpfung r1b_xfer_A in Konto B), dann Slots A1, A2 (Konto A), B1, B2 (Konto B), danach
   A_T2, B_T2 und ANA.

## Startprompt für die lokale Sitzung (kopieren)
> Lies analysis/ju_space/round1b/UEBERGABE.md (Abschnitt „Nächste Schritte“ und die letzten Einträge unter „Stand“) und
> analysis/ju_space/round1b/IMPLEMENTATION.md. Arbeite die Schritte 1 bis 5 der Reihe nach ab: L-CAL korrigieren und vollständig
> rechnen, offset4096 klären, Hochrechnung und Kürzungen nach den Planregeln, runde1b_freeze.json schreiben, einfrieren. Kein echtes
> Modell lokal laufen lassen; Runde-1-Code byteidentisch; nach jedem Schritt committen und auf ccr-6fe02b21-est040 pushen.
> Antworte auf Deutsch.

## Stand
- 13.11. 23:50 MEZ: Kern in Arbeit (r1b_contracts.py vorhanden). Teile R, K, D, L und Integration noch offen.
- 14.11. 00:05 MEZ: Kern fertig und im Mock-Modus getestet (r1b_contracts.py, r1b_core.py, r1b_run_session.py, r1b_run_slot.py,
  IMPLEMENTATION.md, notebooks/build_runde1b_gpu_notebooks.py; 15 Änderungen am Rahmenplan, u. a. Zeitwächter je Sitzung). Die vier
  Teile R, K, D, L laufen jetzt parallel; danach Integrationstest. Offen laut Kern: Einfrier-Konstanten und runde1b_freeze.json,
  r1b_report.py (Rundenauswertung), J-Lens-Download-URL, GPU-/Colab-Test, Inversfrei-Lint.
- 14.11. 01:15 MEZ: Teil D fertig umgesetzt; R, K und L in Arbeit (rund 24.000 Zeilen im Ordner, noch nicht integriert).
  Integrationstest steht aus. Für Codex: Phase 1 jetzt möglich, Phase 2 erst nach „Integrationstest bestanden“.
- 14.11. 01:30 MEZ: Teile D und L fertig umgesetzt, Mock-Selbsttests bestanden (D: Langkontext-Engine SO-LCE, Reize, alle Sitzungen,
  Auswertung; L: alle Schritte von r1b_L_A/r1b_L_B und l_ana). Offen bei D/L: Paketdateien vor dem Einfrieren erzeugen
  (r1b_d_stim.py build; r1b_l_base.build_longdoc_list), volle L-CAL (etwa 120 CPU-Min.) vor dem Einfrieren. R und K in Arbeit.
- 14.11. 01:50 MEZ: Teile R und K fertig umgesetzt, Mock-Tests bestanden (R: 13 Sitzungen; K: 12 Sitzungen inkl. RK-1 mit CE- und
  Soft-MTP-Kopf). Alle vier Teile damit fertig. Läuft jetzt: Integrationstest (alle Slots im Mock-Modus, Rundenauswertung,
  Notebooks) und Bau der Paketdateien (R: 192 neue Prompts; K: Labelbank, REF-K16; D: Reize; L: Langdokument-Liste).
- 14.11. 04:00 MEZ: Integrationstest bestanden (alle 32 Sitzungen im Mock-Modus; 2 Klebefehler K->R und K->L mit 3 Korrekturen behoben; tests/integration_cpu.txt)
- 14.11. 04:20 MEZ: Paketdateien fertig und geprüft (results/ju_space/plans/runde1b/packages/, MANIFEST.json mit sha256). Notebooks
  gebaut (notebooks/runde1b_*.ipynb); J-Lens wird bei Bedarf gepinnt von Hugging Face (neuronpedia/jacobian-lens) geladen.
  Nächster Schritt: Pilot r1b_P0 in Konto B (runde1b_P0.ipynb, beim ersten Lauf SELBSTTEST = True). Danach Einfrieren:
  runde1b_freeze.json (freeze_commit, freeze_time, plan_files mit sha256, package_files, pilot-Raten, trims, projection_h,
  funded {tier2: true, items}, datasets, models, jlens) schreiben und FROZEN_PLAN_SHA256 / FREEZE_RECORD_SHA256 in
  r1b_run_session.py setzen, Plan-Arbeitskopie nach results/ju_space/plans/runde1b/ übernehmen, committen, pushen.
  Dann Stufe 1: A1, A2 (Konto A), B1, B2 (Konto B); vorher in Konto A den Ordner dsse_runs/r1b_xfer für Konto B freigeben und in
  Konto B als Verknüpfung r1b_xfer_A unter „Meine Ablage“ anlegen.
- 14.11. 11:45 MEZ: Pilot r1b_P0 in Konto B gelaufen. Erster Lauf scheiterte an d0_pilot (GPU-Speicher; Commit de5a6bc behebt das:
  Speicherüberlauf wird je Stufe als Messwert festgehalten, expandable_segments in den Notebooks), zweiter Lauf fertig.
  Ergebnisse in Drive, Konto B: dsse_runs/r1b_P0/ (p0_engineering.json, p0/p0_d.json, k0_pilot.json, pilot/l0_a.json,
  pilot/l0_b.json, session_done.json, logs/). Befunde:
  * Teil D: Prefill 1,5/3,6/7,8/9,1 s bei 4K/16K/28K/32K; Decode B=1 24,5/13,0/7,9/7,2 Token/s; B=3 passt bei 28K und 32K NICHT
    (Speicherüberlauf), B=2 passt; Spitze 41 GB. D muss bei 28K/32K mit B <= 2 rechnen (Einfrier-Datensatz, Code prüfen).
  * Generieren deutlich langsamer als geplant (Plan 25-35 ms je Schritt bei Batch 32): L 122-137 ms bei Kontext 512 und
    227-233 ms bei 1100; D-Decode wächst fast linear mit der Kontextlänge. Verdacht: vermeidbarer Engpass im Decode-Pfad
    (float32-Aufmerksamkeit, GQA-Vervielfältigung, Cache-Kopien, Rauschen je Zeile). Vor dem Einfrieren prüfen, ob eine
    rechengleiche Beschleunigung möglich ist (float32, TF32 aus, identische Tokens); sonst greifen die Pilotregeln der Teile
    (SG7: R RG18, D G-PIL, K Abwurfreihenfolge bei > 30 % langsamer, L Kürzungen bei < halber Planrate).
  * Lücken: L0_b-Prefill (0,04 s bei 16K/32K, 6,5 GB) unplausibel, Messung neu; kein Pilotschritt für Teil R; Qwen3.5-2B nicht
    geladen (Pins für Revision und sha256 fehlen noch).
  Läuft: Agent „r1b-freeze-prep“ (Gültigkeit der Messungen, Ursache der Langsamkeit, Projektion je Teil nach den Pilotregeln,
  Entwurf runde1b_freeze.json). Ergebnis wird unter results/ju_space/plans/runde1b/arbeitsstand/freeze_prep.json abgelegt.
  Nächste Schritte: (1) Entscheidung: Decode beschleunigen und Pilot kurz wiederholen ODER Kürzungen nach Plan anwenden;
  (2) Qwen3.5-2B pinnen; (3) runde1b_freeze.json schreiben, Konstanten FROZEN_PLAN_SHA256 / FREEZE_RECORD_SHA256 in
  r1b_run_session.py setzen, Plan-Arbeitskopie (ohne ENTWURF-Banner) nach results/ju_space/plans/runde1b/ übernehmen, committen;
  (4) Freigabe r1b_xfer A -> B (Verknüpfung r1b_xfer_A in Konto B), dann Slots A1, A2, B1, B2.
- 14.11. 12:40 MEZ: Auswertung des Piloten fertig (results/ju_space/plans/runde1b/arbeitsstand/freeze_prep.json; Entwurf
  runde1b_freeze_DRAFT.json und Rechenskripte daneben). Ergebnis: Der Pilot reicht für das Einfrieren noch nicht. Vor dem
  Einfrieren zu erledigen (nur Technik, keine Regeländerung):
  1. FEHLER (Pflicht): In Teil D hält `y, *_ = ...decode(...)` den vorigen Cache fest; das sprengte B=3 bei 28K/32K im Piloten
     und steckt auch in r1b_d_d2 (gen_stream, glce3). Zurückgegebenen Cache verwerfen (`y = ...decode(...)[0]`), Basis wie in den
     Sitzungen auslagern (LCE.offload). Eine Grenze B <= 2 ist dann nicht nötig.
  2. Paketdateien: alle 15 aus results/ju_space/plans/runde1b/packages/ (inkl. k1_label_bank/) in den Planordner
     results/ju_space/plans/runde1b/ kopieren (die Teile lesen dort; nur r1b_d_stim beachtet R1B_PKG_DIR); fehlende bauen:
     pilot_material.json, r1b_keys.json, l_cal.json (volle L-CAL, etwa 120 CPU-Minuten).
  3. Qwen3.5-2B pinnen (Revision 15852e8c enthält die 15 mtp.*-Tensoren; sha256 der Dateien in den Datensatz) und den Lader den
     Einfrier-Datensatz lesen lassen (r1b_k_s3._load35 ruft ensure_model(M35, root, None); load_model reicht ihn nicht weiter).
  4. PG19 abrufbar machen (Quelle GCS, fetch_dataset kann nur Hugging Face); rk1-Offset prüfen (siehe freeze_prep.json).
  5. Pilot-Code für einen Nachpiloten r1b_P0b (etwa 1 A100-Stunde): P0-R-Schritte (fehlen ganz in r1b_sessions_R.py); K0-Punkte
     3, 6, 7, 8, 9 inkl. Laden von Qwen3.5-2B, Hook-Zeit mit Synchronisation und Aufwärmen, FISTA ohne _lip; L0 (i) Decode ohne
     Prefill auf gepolsterten Zeilen, L0 (iii)/(vii) auf echten PG19-Abschnitten mit probe.begin(), Duet je Nachricht bei etwa
     420 Token; Drive-Durchsatz wirklich über Drive messen; P0-D mit Aufwärmen und B=3/B=4 neu.
  6. Optional, zahlengleich (je mit Bit-Identitätstest): E.decode ohne Synchronisation je Schritt (etwa 1,3x), KV-Puffer an Ort
     statt torch.cat (etwa 1,35x bei 32K, B=3), Gumbel-Rauschen in K nur für die N behaltenen Zeilen.
  Hochrechnung bei den gemessenen Raten (vorläufig): R 12,3 h (RG18 wählt RL2 -> 12,1 h), D 16,2 h (Kürzungen T1-T8 -> 13,8 h,
  über dem 9,0-h-Deckel: Budget-Nachtrag +4,8 h nötig oder Speed-Fixes), K 10,6 h, L 5,5 h; Stufe 1 etwa 40 h, Stufe 2 etwa
  9,7 h, Pilot etwa 1 h, zusammen etwa 51 A100-Stunden statt 40,3. Danach: r1b_P0b laufen lassen, Datensatz fertig, einfrieren.
- 14.11. 13:15 MEZ: Entscheidungen des Nutzers: Korrekturrunde hier in der Cloud-Sitzung (ein Agent, Mock-Tests, mit den drei
  zahlengleichen Beschleunigungen) und Mehrbedarf akzeptiert (Runde etwa 45-51 A100-h; D-Deckel 13,8 h statt 9,0 h, D-Kürzungen
  nur soweit die Pilot-Hochrechnung 13,8 h übersteigt; alle Messungen bleiben). Erledigt: Paketdateien in den Planordner kopiert
  (Commit 88786c2); Plan-Arbeitskopie: echte Dateinamen in 0.1/9.1, Nachpilot r1b_P0b, D-Nachtrag (Commit c4971aa). Läuft: volle
  L-CAL (Wrapper scratchpad lcal/lcal_resumable.py, Ergebnis danach nach results/ju_space/plans/runde1b/l_cal.json) und der
  Korrektur-Agent (Punkte 1-6 oben; Schalter engine.fast.* je mit Bit-Identitätstest; Sitzung r1b_P0b mit Notebook
  runde1b_P0b.ipynb, Konto B; Pins vor dem Einfrieren aus runde1b_pilot_pins.json). Danach: r1b_keys.json, Notebooks, Nachpilot.
- 14.11. 14:45 MEZ: Korrekturrunde fertig (Commits 8c09305 bis edaaa19), Integrationstest wegen Kontingent nach 10 von 33
  Sitzungen abgebrochen (alle exit 0, inkl. r1b_P0 und r1b_P0b). Erledigt: D-Cache-Fehler; Qwen3.5-2B gepinnt (Pins-Datei
  runde1b_pilot_pins.json vor dem Einfrieren); PG19 über GCS mit sha256; pilot_material.json (Builder r1b_pilot.py build);
  Sitzung r1b_P0b (12 Schritte, etwa 60 A100-Min., Ergebnis p0b_engineering.json) mit notebooks/runde1b_P0b.ipynb (Konto A, da Konto B keine A100 bekam);
  Schalter engine.fast.{l_decode_nosync, lce_inplace_kv, k_gumbel_kept_rows} (aus, bis P0b Bitgleichheit auf der A100 zeigt;
  tests/fast_identity.py 58/58 auf der CPU); r1b_keys.json; K-drop2/3. Offen: (1) restliche 23 Sitzungen des Mock-Integrationstests
  (scratchpad r1b/integ_run2.py-Muster); (2) K-drop4/5 (K6c(ii) gehört zu CF-K-link); (3) volle L-CAL: lief hier im Hintergrund,
  falls results/ju_space/plans/runde1b/l_cal.json fehlt, lokal neu: python analysis/ju_space/round1b/r1b_l_cal.py run
  results/ju_space/plans/runde1b/l_cal.json (mehrere CPU-Stunden; Punkt a allein über 1 h); (4) Qwen3.5-2B-Linearaufmerksamkeit
  läuft ohne Zusatzpakete langsam (P0b misst die Rate); (5) Plantext 9.1: engine-Schlüssel, Pins-Datei, p0b_engineering.json.
  Nächster Schritt des Nutzers: runde1b_P0b.ipynb in Konto A laufen lassen; danach Einfrier-Datensatz aus p0b_engineering.json.
- 14.11. 15:15 MEZ: Volle L-CAL hier nach 2 h abgebrochen (Grenze für Hintergrundjobs); nur Punkt (a) fertig (71 min mit 2 Kernen)
  und DURCHGEFALLEN: Treffer localist 0/5, redundant 0/5, dense 0/5, dthr 5/5, xor3 0/5 -> G-cal-L1 falsch; ohne Korrektur liest
  L-ANA alle von L1 abhängigen Zweige 'not testable (pipeline calibration failed)'. VOR DEM EINFRIEREN: Klassifikation in
  r1b_l_cal.item_a / r1b_l_rules (PIP-Baum, Schwellen) prüfen und korrigieren oder bewusst so einfrieren. Restliche Punkte b-n
  lokal fortsetzen (Ergebnis von a liegt bei): python results/ju_space/plans/runde1b/arbeitsstand/lcal/lcal_resumable.py
  results/ju_space/plans/runde1b/l_cal.json results/ju_space/plans/runde1b/arbeitsstand/lcal/cache (mehrere CPU-Stunden;
  nach einer Änderung an r1b_l_cal.py oder r1b_l_stats.py den Cache löschen und alles neu rechnen).
- 14.11. 16:00 MEZ: Nachpilot r1b_P0b in Konto A gelaufen (A100-40GB). Auszug: results/ju_space/plans/runde1b/arbeitsstand/
  p0b_summary.json (voller Datensatz in Drive Konto A: dsse_runs/r1b_P0b/p0b_engineering.json). Alle drei engine.fast-Schalter auf
  der GPU bitgleich -> im Einfrier-Datensatz true; KV an Ort bei 32K B=3 1,24x schneller; B=3/B=4 passen bei 28K/32K (B_long 3
  bleibt); Paritäten bestanden, OD-M14 = full_grid; Qwen3.5-2B lädt (G-K18 ok, G-K17 pass), Decode nur 1,04x langsamer als 1.7B;
  Hook-Aufschlag 1,001; FISTA 4,25 ms; Drive 414/91 MB/s. Einziger Fehlschlag: K0-Prüfung offset4096 (Top-1 gleich, max |dlogp|
  1,03e-3; Regel in Teil K nachsehen). L-Decode 60-81 ms je Schritt bei Batch 32 statt geplanter 25-35 ms -> L-Pilotregel prüfen.
  Vor dem Einfrieren offen: L-CAL (Punkt a durchgefallen, b-n fehlen), Hochrechnung je Teil mit diesen Raten
  (arbeitsstand/freeze_prep_proj/*.py), Kürzungen nach den Regeln (D-Deckel 13,8 h), runde1b_freeze.json, Konstanten, Freeze-Commit.

- 15.11. (lokale Sitzung): L-CAL (a) planmäßig korrigiert (Commit 21bcee2; nur die L1-Statistik an den Plan angeglichen, keine
  Daten, Signalstärken, Schwellen oder Sollurteile geändert): Python-Floats an rule_L1 (np.float32 machte rho_2 und I-hat zu NaN),
  Auswahl auf S_fit und I-hat auf getrennten S_L-Zeilen (L1 (ii)), Untergrenzen F(m) an jeder Rastergröße und bei |P| mit
  100 Permutationen (L1 (i); vorher fiel jede fehlende Größe auf das negative F(1024) zurück, m_0 = 1-2 überall), 50 U-Fragmente,
  Priv mit unterer 90-%-Schranke gegen F(|P|), m_0 über die untere Schranke des Medians, Wiederauswahl r = 2..6 mit Hüllen,
  rho_2 mit gepaartem Bootstrap, XH-Verhältnis gerechnet statt auf 4 gesetzt. Cache geleert; volle L-CAL läuft (4 Kerne:
  (a) in zwei Prozessen nach Profilen, b-n in einem; etwa 45 min je Profil und Saat bei (a)).
  **BEFUND (a), Nutzerentscheidung nötig:** (a) fällt auch planmäßig durch. localist Saat 0 und 1: 'H privileged, redundancy
  undecided' (Priv wahr, rho_2 ≈ 0, R_rep = 1, aber XH-Verhältnis 1,75 / 1,80 < 4). Ursache strukturell: XH (ohne P-Hülle) erreicht
  0,5 I_ref im Raster nie, also zensiert bei 1.024 / m_0.5(U); im verkleinerten Generator (N = 4.096, 20 Träger) liegt m_0.5(U)
  bei etwa 570-590, die Obergrenze des Verhältnisses also bei etwa 1,8. Im echten Lauf (N = 172.032, Raster bis 1.024 = 0,6 %)
  besteht diese Schranke nicht. dense Saat 0 und 1: richtig. Diagnose mit einer Saat (vor den Punkten 3-8) deutet außerdem an:
  redundant (2.000 von 4.096 Einheiten tragen) wird ohne Auswahlleck nicht 'Priv' (zufällige 20er-Mengen enthalten etwa 10 Träger),
  dthr landet nach der Baumreihenfolge (Zweig 5 'dense' vor Zweig 6) bei 'dense (no sparse store)'; wird mit dem Volllauf bestätigt.
  b-n: f, g, j, k, l, m, n bestanden; b, c, d, e, h, i laufen. Ohne Entscheidung liest L-ANA alle von G-cal-L1 abhängigen Zweige
  'not testable (pipeline calibration failed)'. Optionen: (A) so einfrieren (G_cal_L1 = false); (B) technischer Nachtrag vor
  allen Daten zum Generator von (a) (z. B. N, das die Rastergrenze wie im echten Lauf klein macht), dann neu rechnen.
  Push aus der lokalen Sitzung scheitert (kein Schreibrecht: git ohne Token, GitHub-API 403); Commits liegen lokal und als Bundle.

## Erweiterungswunsch des Nutzers (14.11.2026): Gemma 4 als weiteres Modell (nach 1b, wie Teil P)
- Sinnvoll vor allem für K und L, teilweise D; R ist an die Runde-1-Daten von Qwen gebunden.
- Belegte Anknüpfungspunkte (aus den geprüften Berichten in results/ju_space/plans/runde1b/arbeitsstand/):
  H-Neuronen wurden in Gemma-3-4B gefunden (2-9 Neuronen, AUROC 0,81 auf TriviaQA; Gao et al.); Gemma 4 E2B/E4B haben offizielle
  MTP-Drafter (google/gemma-4-E2B-it-assistant, -E4B-it-assistant; rund 78 Mio. Parameter, lesen den KV-Cache des Zielmodells),
  also eine zweite native Mehr-Token-Linse neben Qwen3.5-2B; für Gemma 4 12B gibt es Neuron-Lens-Lesungen aller MLP-Neuronen
  (stdlogits/neuron-lens).
- Aufwand: Adapter im Kern-Lader (Hook-Punkte, Chat-Vorlage, Fensteraufmerksamkeit bei langem Kontext), J-Lens-Anpassung, H-Menge,
  Schadenseichung; grob 4-8 A100-Stunden für K und L (plus D), als eigener Teil G mit eigenem kurzen Plan vor dem Lauf.
- Vom Nutzer genannte Gemma-4-Fassungen (14.11.2026): gemma-4-31b-it-qat-w4a16-ct, gemma-4-12b-it-qat-w4a16-ct,
  gemma-4-e4b-it-qat-w4a16-ct, gemma-4-e2b-it-qat-w4a16-ct (QAT, 4-Bit-Gewichte, 16-Bit-Aktivierungen, vermutlich
  compressed-tensors; Konfiguration und Lizenz vor dem Plan prüfen). Vorschlag: e2b/e4b als nicht quantisierte Originale voll
  (mit MTP-Drafter als zweiter nativer Mehr-Token-Linse); 31b in 4 Bit (etwa 16 GB, passt auf A100-40GB) nur mit
  Vorwärts-Messungen aus K und L (gradientenbasierte Verfahren bräuchten entpackte Gewichte, etwa 62 GB); 12b als Brücke zur
  Neuron Lens für Gemma 4 12B.

## Reihenfolge und Budget für die lokale Sitzung (API-Guthaben, etwa 160 $)
1. Runde 1b fertigstellen (was „Stand“ noch offen lässt; Integrationstest, Notebooks), Pilot, Einfrieren, Läufe. Vorrang.
2. Teil P (Pythia-1.4B-Brücke, Stufe 2): Recherche liegt in `results/ju_space/plans/runde1b/arbeitsstand/findings/P_research.json`;
   kurzen Planteil schreiben (nur Vorwärts-Messungen aus K, L und eine kurze Fassung von D1 bei 2.048 Token; Ausgaben je Token auf
   Pile-Sequenzen mit Kennung und Position, verknüpfbar mit den Timaeus-Clustern; keine Timaeus-Daten automatisiert laden, solange
   deren Bedingungen das nicht klar erlauben), dann umsetzen. Etwa 3 A100-Stunden (+1,5 für Trainingsstände).
3. Teil G (Gemma 4, siehe oben): kurzen Planteil schreiben und umsetzen.
   - Vorab prüfen: Modellkarten, config.json und Lizenzen der vier Fassungen und der MTP-Drafter (Hugging Face); ob transformers die
     compressed-tensors-Fassung mit Hooks lädt. Gegatete Modelle: Hugging-Face-Token setzt der Nutzer selbst in den Colab-Secrets.
   - Umfang: e2b/e4b (Originale, float32) mit K-Kern (H-Menge nach Gao, Kettenlinse durch Selbst-Ausrollen, Eichung am MTP-Drafter)
     und L-Kern; 31b-qat nur Vorwärts-Messungen aus K und L; optional D1-Kontrast (lokale gegen globale Aufmerksamkeit).
   - Adapter im Kern-Lader (`r1b_core`): Hook-Punkte (Blockausgabe, MLP-Zwischenwerte), Chat-Vorlage, Fensteraufmerksamkeit.
   - Etwa 4-8 A100-Stunden.
- Kostenschätzung Claude Code (grob, nicht verbindlich): eine längere Arbeitssitzung kostet mit Opus 5.5 etwa 25-65 $, mit
  Sonnet 5.5 etwa die Hälfte. 160 $ reichen für Rest von 1b, P und G nur bei schlanker Arbeitsweise: eine Sitzung, keine
  Gremien oder Mehrfach-Agenten, `/model sonnet` für Routinearbeit, `/cost` im Blick, Ausgabenlimit im Console-Workspace.

## Startprompt für Teil G (später kopieren)
> Lies analysis/ju_space/round1b/UEBERGABE.md (Abschnitte zu Gemma 4) und IMPLEMENTATION.md. Prüfe zuerst Modellkarten, Lizenzen
> und config.json der Gemma-4-Fassungen und der MTP-Drafter. Schreibe dann einen kurzen Planteil
> results/ju_space/plans/runde1b/arbeitsstand/plan/runde1b_plan_teil_G.md im Format der anderen Teile (Messungen aus K und L,
> gleiche Definitionen und Urteilssätze, Vorwärts-Messungen für 31b-qat), setze ihn mit einem Gemma-Adapter im Kern-Lader um,
> teste im Mock-Modus und baue die Notebooks. Schlank arbeiten, nach jedem Schritt committen und pushen. Antworte auf Deutsch.

## Auftrag an Codex (OpenAI) für Teil G, parallel zur laufenden Cloud-Sitzung
- Eigener Branch `codex-teil-g`, abgezweigt vom aktuellen Stand von `ccr-6fe02b21-est040`. Nie auf `ccr-6fe02b21-est040` pushen;
  zusammengeführt wird später in einem Schritt.
- Nur neue Dateien anlegen: `analysis/ju_space/round1b/r1b_g_*.py` (darunter `r1b_g_adapter.py`, das den Modell-Lader des Kerns
  per Registrierung um Gemma 4 erweitert), `analysis/ju_space/round1b/r1b_sessions_G.py`,
  `results/ju_space/plans/runde1b/arbeitsstand/plan/runde1b_plan_teil_G.md` und einen eigenen Notebook-Bau
  `notebooks/build_runde1b_gemma_notebooks.py`. Bestehende Dateien (Kern, Verträge, Runner, Teile R/K/D/L, bestehender
  Notebook-Bau) nicht ändern; nötige Änderungen dort als Liste in `analysis/ju_space/round1b/G_MERGE_NOTES.md` festhalten.
- Phase 1 (sofort möglich, unabhängig von der Cloud-Sitzung):
  1. Modellkarten, Lizenzen und config.json der vier Fassungen (gemma-4-31b/12b/e4b/e2b-it-qat-w4a16-ct), der nicht quantisierten
     e2b/e4b-Originale und der MTP-Drafter (google/gemma-4-E2B-it-assistant, -E4B-it-assistant) prüfen; ob transformers die
     compressed-tensors-Fassung lädt und ob Forward-Hooks an Blockausgabe und MLP-Zwischenwerten greifen. Ergebnis in den Planteil.
  2. Planteil G schreiben, im Format der anderen Teile: Messungen aus K und L mit identischen Definitionen und Urteilssätzen;
     e2b/e4b voll (float32), Eichung der Kettenlinse am MTP-Drafter; 31b-qat nur Vorwärts-Messungen; optional D1-Kontrast
     (lokale gegen globale Aufmerksamkeit); Kosten in A100-Minuten, Sitzungen, Tore.
  3. `r1b_g_adapter.py` mit Mock-Selbsttest (kleines zufälliges Gemma-artiges Modell, CPU).
- Phase 2 (erst wenn hier unter „Stand“ „Integrationstest bestanden“ steht): die Messungen von G gegen die fertigen Module aus K und
  L umsetzen (`r1b_k_*`, `r1b_l_*`, `r1b_hn.py`, `r1b_lce.py`), Mock-Tests, Notebooks.
- Regeln wie oben: Runde-1-Code unverändert; echte Modelle nicht lokal laufen lassen, Tests im Mock-Modus; nichts von goodfire.*
  automatisiert abrufen; keine Schlüssel oder Tokens in Dateien oder Commits.

### Startprompt für Codex (kopieren)
> Read analysis/ju_space/round1b/UEBERGABE.md, especially the sections on Gemma 4 and "Auftrag an Codex", and
> analysis/ju_space/round1b/IMPLEMENTATION.md. Work only on a new branch `codex-teil-g` created from `ccr-6fe02b21-est040`, and only
> in new files as listed there; never modify existing files and never push to `ccr-6fe02b21-est040`. Do phase 1 now: verify model
> cards, licenses and configs of the Gemma 4 variants and MTP drafters, write the plan part
> results/ju_space/plans/runde1b/arbeitsstand/plan/runde1b_plan_teil_G.md in the format of the other part files, and implement
> analysis/ju_space/round1b/r1b_g_adapter.py with a CPU mock self-test. Record any changes needed in existing files in
> analysis/ju_space/round1b/G_MERGE_NOTES.md. Commit and push to `codex-teil-g` after each step. Write the plan text in English and
> your summary to the user in German.