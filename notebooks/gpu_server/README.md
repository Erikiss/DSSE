# Notlösung: gemieteter GPU-Server statt Colab

Für `superposition` und `entkopplung`. Es laufen dieselben Notebooks wie in Colab: `runner.py` führt ihre Code-Zellen der
Reihe nach in einem Python-Prozess aus. Die Ergebnisse landen in einem Ordner auf dem Server, mit denselben
Unterordnern wie in Drive (`superposition_runs_s0_of3` …). Sie lassen sich also später in den Drive-Ordner
`dsse_icl_vs_ft` legen und mit Colab-Ergebnissen zusammen auswerten. Geht bei jedem Anbieter mit NVIDIA-GPU und
Linux (RunPod, Lambda, Vast, Jarvis Labs …). Ein Hugging-Face-Token ist nicht nötig, alle Modelle sind offen.

## Welcher Server

| Experiment | GPU | Platz | Dauer grob |
|---|---|---|---|
| superposition | 3 × 80 GB (A100 oder H100): die drei Teilsitzungen laufen parallel | ab 150 GB | 3–4 h |
| superposition | 1 × 80 GB: Teilsitzungen nacheinander | ab 150 GB | 9–11 h |
| entkopplung | 1–3 GPUs ab 40 GB (nur Qwen3-1.7B, kein großer Judge) | ab 50 GB | wie in Colab |

* Mit 48-GB-Karten (L40S, A6000) läuft der Judge in 4 Bit, langsamer. Alle Teilsitzungen sollten dieselbe GPU-Art haben,
  sonst bewerten verschiedene Judges; die Auswertung warnt dann.
* Image: am einfachsten ein PyTorch-Image mit CUDA. `setup.sh` installiert torch, falls es fehlt oder zu alt ist.
* Superposition braucht grob 10–12 GPU-Stunden. Bei Listenpreisen von etwa 1,3–3 $ je GPU-Stunde sind das
  ungefähr 15–35 $. Die Preise schwanken stark je nach Anbieter und Zeitpunkt. Server nach dem Lauf stoppen oder löschen.

## Schritte

1. Server starten, Terminal öffnen (SSH oder das Terminal in JupyterLab).
2. Repository holen und einrichten:

   ```bash
   cd /workspace 2>/dev/null || cd ~
   git clone -b ccr-6fe02b21-est040 https://github.com/Erikiss/DSSE.git
   cd DSSE
   bash notebooks/gpu_server/setup.sh
   ```

   `setup.sh` prüft GPUs und freien Platz, installiert die Pakete und lädt Qwen3-1.7B und gemma-4-31B-it vorab
   (zusammen etwa 67 GB, je nach Leitung 5–20 Minuten). Danach macht es einen kurzen GPU-Test. Nur für die
   Entkopplung reicht `MODELS="Qwen/Qwen3-1.7B" bash notebooks/gpu_server/setup.sh`.
3. Den Lauf in `tmux` starten, damit er weiterläuft, wenn die Verbindung abreißt:

   ```bash
   tmux new -s dsse
   bash notebooks/gpu_server/run.sh superposition
   ```

   Abkoppeln: Strg+B, dann D. Wieder ankoppeln: `tmux attach -t dsse`. Ohne tmux (`apt-get install -y tmux` geht meist):
   `nohup bash notebooks/gpu_server/run.sh superposition > lauf.log 2>&1 &`
4. Fortschritt ansehen, z. B. `tail -f /workspace/dsse_superposition/logs/teil0.log` (ohne `/workspace` unter
   `~/dsse_superposition`). Nach der Eichung des Judges steht dort, wie viele Sekunden eine Bewertung dauert. Steht dort
   fast alles „nicht lesbar", abbrechen (Strg+C im tmux-Fenster) und mir das Protokoll schicken.
5. Am Ende steht die vollständige Auswertung in `logs/auswertung.log`. Das Archiv `ergebnisse_superposition.tar.gz`
   (ohne Adapter, wenige MB) herunterladen: über den Dateibrowser in JupyterLab oder mit `scp`.

`run.sh` erkennt die Zahl der GPUs: Sind mindestens so viele GPUs da wie Teilsitzungen, laufen sie parallel, je eine
GPU. Sonst laufen sie nacheinander auf GPU 0. Vorher lädt es die DSSE-Daten einmal, damit sich parallele Läufe nicht
in die Quere kommen.

## Neustart, Teilläufe, Zusammenführen mit Colab

* Bricht etwas ab: `run.sh` einfach erneut starten. Fertige Zustände und trainierte Adapter werden übersprungen.
* Nur einzelne Teilsitzungen, z. B. wenn Colab Teilsitzung 1 und 2 schon geschafft hat:
  `SHARDS="2" bash notebooks/gpu_server/run.sh superposition`
* Alles nacheinander auf einer GPU, obwohl mehrere da sind: `PARALLEL=0 bash notebooks/gpu_server/run.sh superposition`
* Zusammenführen: die Ordner `superposition_runs_s…_of3` aus dem Archiv in den Drive-Ordner `dsse_icl_vs_ft` hochladen.
  Dann in Colab in einem Teilsitzungs-Notebook nur *Konfiguration*, *Speicherort* und *Auswertung* ausführen. Oder
  umgekehrt die Drive-Ordner auf den Server kopieren und dort
  `python3 notebooks/gpu_server/runner.py superposition --analysis-only --results /workspace/dsse_superposition`
  ausführen.
* Damit ich auswerten kann: `logs/auswertung.log` oder das ganze Archiv in den Drive-Ordner `dsse_icl_vs_ft` legen.

## Alternative: JupyterLab statt Terminal

Bietet der Anbieter JupyterLab an, kann man auch das Notebook selbst öffnen (`notebooks/superposition_a100_shard0.ipynb`
usw.) und „Run All" wählen. Außerhalb von Colab schreibt das Superposition-Notebook nach
`/workspace/dsse_superposition` (sonst `~/dsse_superposition`) und legt die Modelle unter `…/hf_cache` ab. Laufen
mehrere Teilsitzungs-Notebooks auf einem Server mit mehreren GPUs, nimmt sich jedes eine eigene GPU. Nachteil: Schließt
man den Browser, läuft der Kernel zwar weiter und die Dateien entstehen, aber die Ausgabe im Notebook fehlt. Der
Terminal-Weg ist robuster. Für die Entkopplung nur den Terminal-Weg nehmen; dort fehlt im Notebook die Zuordnung der
GPUs.

## Dateien

* `setup.sh`: einmalige Einrichtung (Pakete, Modelle, Prüfungen)
* `run.sh`: Daten laden, Teilsitzungen rechnen, Auswertung, Archiv
* `runner.py`: führt die Code-Zellen eines Notebooks aus (`--prefetch`, `--analysis-only`)
* `env.sh`: gemeinsame Pfade (`WORK`, `HF_HOME`)
