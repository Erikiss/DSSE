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

## Nächste Schritte (in dieser Reihenfolge)
1. Fehlende Teile der Umsetzung fertigstellen (siehe „Stand“), Mock-Selbsttests je Teil, dann der Integrationstest aller Sitzungen.
2. Notebooks bauen (`python notebooks/build_runde1b_gpu_notebooks.py`), prüfen, committen.
3. Pilot r1b_P0 in Konto B laufen lassen (nur Technik), Raten in den Plan, Plan einfrieren (sha256 der fünf Dateien in
   runde1b_freeze.json, Abschnitt 9 des Rahmens), dann die vier Slot-Notebooks der Stufe 1, danach Stufe 2 und Auswertung.

## Startprompt für die lokale Sitzung (kopieren)
> Lies analysis/ju_space/round1b/UEBERGABE.md und analysis/ju_space/round1b/IMPLEMENTATION.md. Setze die Umsetzung von Runde 1b
> dort fort, wo „Stand“ endet: fehlende Sitzungsschritte nach den Plandateien in results/ju_space/plans/runde1b/arbeitsstand/plan/
> implementieren, Mock-Selbsttests ausführen, dann den Integrationstest aller Sitzungen im Mock-Modus und den Notebook-Bau.
> Arbeite schlank und der Reihe nach; committe und pushe nach jedem Schritt auf ccr-6fe02b21-est040. Antworte auf Deutsch.

## Stand
- 13.11. 23:50 MEZ: Kern in Arbeit (r1b_contracts.py vorhanden). Teile R, K, D, L und Integration noch offen.
