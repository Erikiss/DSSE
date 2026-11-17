# H_tq-Fortsetzung auf einem GPU-Server vorbereiten

**Historische Vorbereitung:** Der Nutzer hat den Miet-GPU-Wechsel am 16.11.2026 verworfen.
Die Fortsetzung bleibt in Colab/A100; diese Server-Anleitung ist kein aktueller Startauftrag.

`htq_launcher.py` arbeitet direkt mit den vorhandenen Runde-1b-Sitzungen. Er führt
keine Colab-Zellen mit Drive-Mount aus und verwendet nicht die alten
`setup.sh`/`run.sh` dieses Ordners: Diese gehören zu anderen Experimenten und ihr
Archivfilter lässt `.pt` weg. **Für HTQ müssen sämtliche `.pt` erhalten bleiben.**

Der Launcher mietet keine Cloud-Instanz und veröffentlicht nichts auf Drive oder GitHub.
Fertige Sitzungen führt er im lokalen Ergebnisbaum zusammen; Pakete installiert er nicht.
Der Launcher ändert weder Stichproben, Zufallsschlüssel, Batchgrößen, Float32,
TF32-Einstellungen, Attention-Kernel noch eingefrorene Pläne. Mehrere GPUs rechnen
unabhängige Sitzungen; es gibt kein DDP und kein Aufteilen eines Modells.

## Aktuelle Hardwaregrenze

Die unveränderte SG2-Regel erlaubt ausschließlich A100, Compute Capability 8.0,
mindestens 39 GB. `run` prüft dies **vor Worker-Kopien und Modell-Downloads**.
H100/H200/B200/B300 werden ausdrücklich abgelehnt. Ein solcher Lauf braucht erst
eine gesonderte Hardwarefreigabe und Kernel-/Paritätsmessungen; der Launcher
umgeht die Regel nicht mit `RUN_DEVICE=cpu`.

Die Mehr-GPU-Ausführung gilt für mehrere sichtbare GPUs **auf demselben Linux-Host**.
Zwei getrennte gemietete Instanzen sind damit noch kein gemeinsamer Worker-Pool;
deren verifizierter Ergebnistransfer wäre ein weiterer Arbeitsschritt.

## 1. Ergebnisbäume vollständig auf ein persistentes Volume kopieren

Alte Schreiber in beiden Konten stoppen. Die vorbereiteten Variantenordner
`dsse_runs/variants/htq_v2` von A und B vollständig exportieren beziehungsweise
herunterladen, einschließlich `_htq_seed.json`, Rohdaten, Protokollen, Caches,
Provenienzen und aller Tensor-Dateien. Drive-Verknüpfungen beim Export auflösen;
die lokalen Eingabeordner dürfen keine Symlinks enthalten.

Beispielablage:

```text
/data/import_A/htq_v2/
/data/import_B/htq_v2/
```

Die Fortsetzung setzt das bereits erledigte `r1b_R_S0` und seine vollständigen
`r1b_xfer`-Exporte voraus. Neue R_S0-Ausgaben werden von diesem Launcher nicht als
kontenübergreifende Exporte veröffentlicht. Das passt zum bisherigen Laufstand.
Die alten Skript-Sitzungen R_S1 und R_S2_14 müssen ebenfalls bereits erledigt sein:
ihre Skripte starten eigene Prozessgruppen und werden vom Launcher nicht neu
ausgeführt. Noch offene Skript-Sitzungen werden ausdrücklich zurückgewiesen.

## 2. Arbeitskopie und Umgebung

Auf dem Server eine passende CUDA-/PyTorch-Umgebung bereitstellen. Der Launcher
installiert oder aktualisiert PyTorch nicht. Die Notebook-Version ist
`transformers==5.22.2`; außerdem werden NumPy, SciPy und huggingface_hub benötigt.
Keine Quantisierung, BF16-Umstellung oder Flash-Attention-Fallback ergänzen.

```bash
git clone -b codex/r1b-htq-continuation https://github.com/Erikiss/DSSE.git /data/DSSE
cd /data/DSSE
python -m pip install 'transformers==5.22.2' numpy scipy huggingface_hub
python notebooks/gpu_server/htq_launcher.py init \
  --state /data/htq_server \
  --source-a /data/import_A/htq_v2 \
  --source-b /data/import_B/htq_v2
```

`init` prüft die aktuellen Erzeuger-Prüfsummen aus Sitzungs-/Schrittmarkierungen
und kopiert anschließend jede Datei mit SHA256-Vergleich in einen eigenen
Arbeitsstand. Historische Seed-Prüfsummen ersetzen dabei keine aktuellen
Erzeuger-Nachweise. Nicht registrierte Caches werden vollständig erhalten, aber
nicht als bewiesene wissenschaftliche Eingaben ausgegeben.

Die alten absoluten Colab-Pfade im Bootstrap-Marker werden als ausdrücklich
protokollierte Transportmetadaten an den lokalen Zielpfad angepasst. Originaltext
und SHA256 des alten Markers stehen unverändert in `import_A.json`/`import_B.json`;
die wissenschaftlichen Dateien und ihre Provenienzen bleiben bytegleich. Ein
bereits initialisierter Arbeitsstand wird nicht durch einen neuen Import ersetzt.

## 3. Abhängigkeiten und nächste Sitzungen ansehen

```bash
python notebooks/gpu_server/htq_launcher.py plan --state /data/htq_server
python notebooks/gpu_server/htq_launcher.py status --state /data/htq_server --tier2
```

`ready`, `waiting`, `done` und `not_run` werden aus denselben Verträgen wie in Colab
abgeleitet. Die Prüfung kopiert keine B-Ordner in den gemeinsamen A-Baum.
`waiting` nennt fehlende Sitzungen, Exporte oder Rohdateien. Ohne `--tier2` werden
die separaten Stufe-2-Sitzungen nicht eingeplant; bereits finanzierte Zusatzschritte
innerhalb einer Stufe-1-Sitzung bleiben unverändert aktiv.

## 4. Auf zulässigen A100-GPUs ausführen

```bash
tmux new -s htq
python notebooks/gpu_server/htq_launcher.py run \
  --state /data/htq_server --gpus 0,1 --capture-hidden
```

Bei einer GPU `--gpus 0`, bei drei GPUs `--gpus 0,1,2`. `--capture-hidden` aktiviert
den bestehenden begrenzten K3a-Mitschnitt nur dann, wenn K_S1 tatsächlich noch läuft.
Bereits fertige wissenschaftliche Schritte werden dafür nicht wiederholt.

Es gibt genau einen Koordinator pro `--state`. Er vergibt jede Sitzung einmal und
jedem Worker genau eine GPU über `CUDA_VISIBLE_DEVICES`. Nach K_S1 haben L_A,
K_S2a und K_S4 Vorrang, danach das freigeschaltete L_B; dies ist eine einfache
Abhängigkeitsheuristik, keine Garantie der rechnerisch optimalen Gesamtdauer.
Fertige Sitzungen werden übersprungen. Fehlende Voraussetzungen stoppen mit
`waiting.json`, statt eine GPU beliebig lange ungenutzt zu halten.

Jeder Sitzungsprozess hat private A-/B-Snapshots, `work_htq_v2/<Sitzung>`,
Modell-/HF-Cache und ROOT je Worker. Die unveränderten Notebook-Zellen für
geprüfte Eingaben laden die festgelegten Modelle, Tokenizer, Persona-Daten und
J-Lenses vorab. Die ursprünglichen R1-/R2-Module bleiben unverändert.

Alle Worker-Ausgaben landen zunächst privat. Der Koordinator prüft den fertigen
WORK-Baum gegen die Sitzungs-Prüfsummen und veröffentlicht ihn erst dann atomar.
Er vertraut nicht allein dem fehlertoleranten `sync` des ursprünglichen Runners.
Damit schreiben Worker weder gleichzeitig in denselben Sitzungsordner noch in
gemeinsame Kopier-/Export-Manifeste.

Worker besitzen eigene Prozessgruppen. Bei Fehler oder Abbruch werden auch ihre
Sitzungs-/Download-Kindprozesse beendet, bevor die Koordinator-Sperre freigegeben
wird. Fehlgeschlagene private Checkpoints bleiben für denselben `run`-Aufruf
erhalten. Die gleiche Kommandozeile setzt sie fort. Nach einem Host-Absturz eine
eventuell verbliebene `.coordinator.lock` erst prüfen; der Launcher löscht fremde
oder vermeintlich veraltete Sperren nicht automatisch.

## Speicherbedarf und Laufzeiten

Diese erste sichere Fassung erstellt pro Sitzung einen vollständigen privaten
Snapshot beider veröffentlichten Konten und behält erfolgreiche Jobs als Auditspur.
Dadurch kann der Speicherbedarf über viele Sitzungen stark wachsen. Es gibt
**keine pauschale Zusage, dass 100 GB genügen**.

Ist die anfängliche A+B-Exportgröße `S`, benötigen Import und gleichzeitig aktive
Worker zunächst ungefähr `(1 + Workerzahl) × S`, zusätzlich zu den Exportquellen,
Modellen/HF-Caches und neuen Ergebnissen. Zurückbehaltene erfolgreiche Jobs kommen
jeweils mit ihrer damaligen Snapshotgröße hinzu. Vor Import und jedem neuen
Snapshot prüft der Launcher den freien Platz gegen dessen gemessene Größe plus
1 GiB Reserve. Das ersetzt keine Reserve für wachsende Ausgaben und Modelldownloads.
Bereits veröffentlichte Ergebnisse und private Jobs werden nicht automatisch
gelöscht. Speicherbereinigung erst nach eigener Prüfung und externer Sicherung.

I/O, Hashprüfungen und Modell-Downloads kommen zur GPU-Rechenzeit hinzu. Die
bekannten 15–18 A100-Stunden waren eine frühere Schätzung; die neue Parallelisierung
ist nicht lokal auf echten GPUs vermessen und macht daraus keine feste Frist.
Die aktuelle Restkalkulation, Preisaufnahme und Hardware-Szenarien stehen in
[GPU_MIGRATION_PLAN_HTQ.md](GPU_MIGRATION_PLAN_HTQ.md).

## CPU-Arbeit separat

ANA verwendet dieselben veröffentlichten Ergebnisse, aber keine GPU:

```bash
python notebooks/gpu_server/htq_launcher.py run \
  --state /data/htq_server --cpu
```

L-CAL bleibt ein eigener CPU-Job und schreibt nicht parallel in wissenschaftliche
Sitzungsordner des Koordinators:

```bash
OMP_NUM_THREADS=2 python results/ju_space/plans/runde1b/arbeitsstand/lcal/lcal_resumable.py \
  /data/htq_lcal/l_cal.json /data/htq_lcal/cache
```

`l_cal.json` muss nach der bestehenden Planregel vor ANA übernommen und committet
werden. Für die separaten finanzierten GPU-Zusatzsitzungen `run --tier2` verwenden;
anschließend `run --cpu --tier2` für den vollständigen Zusatzbericht.

## Ergebnisse sichern

Vor Beendigung einer Instanz den gesamten Arbeitsstand auf ein unabhängiges
Volume oder lokal sichern. Das Archiv muss außerhalb des zu archivierenden Baums
liegen und darf keine Tensor-Filter verwenden:

```bash
tar -C /data -cf /backup/htq_server_ALL.tar htq_server
sha256sum /backup/htq_server_ALL.tar > /backup/htq_server_ALL.tar.sha256
```

Getestet werden Import, Pfadisolation, aktuelle Erzeuger-Prüfsummen, Prozessgruppen,
Speicherplatzgrenzen und die Koordination mit simulierten Workern. Es wurde weder
ein echter Modelllauf noch eine Hopper-/Blackwell-Migration ausgeführt.
