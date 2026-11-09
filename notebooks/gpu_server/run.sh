#!/usr/bin/env bash
# Ein Experiment auf dem gemieteten GPU-Server rechnen, ohne Colab und ohne Google Drive.
# Aufruf aus dem Repository-Ordner, am besten in tmux:  bash notebooks/gpu_server/run.sh superposition   (oder entkopplung)
# Optional: SHARDS="1 2"  (nur diese Teilsitzungen), PARALLEL=0 (alles nacheinander auf GPU 0), RESULTS_ROOT=/pfad
set -uo pipefail
EXP="${1:-superposition}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"
RES="${RESULTS_ROOT:-$WORK/dsse_$EXP}"
SHARDS="${SHARDS:-0 1 2}"
LOGS="$RES/logs"; mkdir -p "$LOGS"
R="$HERE/runner.py"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
read -ra SH <<< "$SHARDS"
echo "Experiment $EXP | Teilsitzungen ${SH[*]} | GPUs $NGPU | Ergebnisse $RES | Protokolle $LOGS"

echo "$(date +%H:%M) Daten laden"
$PY -u "$R" "$EXP" --prefetch --results "$RES" > "$LOGS/daten.log" 2>&1 || { echo "Daten-Download fehlgeschlagen, siehe $LOGS/daten.log"; exit 1; }

fail=0
if [ "$NGPU" -ge "${#SH[@]}" ] && [ "${#SH[@]}" -gt 1 ] && [ "${PARALLEL:-auto}" != "0" ]; then
  echo "$(date +%H:%M) Teilsitzungen parallel, je eine GPU"
  pids=(); i=0
  for s in "${SH[@]}"; do
    CUDA_VISIBLE_DEVICES=$i $PY -u "$R" "$EXP" --shard "$s" --results "$RES" > "$LOGS/teil$s.log" 2>&1 &
    pids+=($!); echo "  Teilsitzung $s auf GPU $i, Protokoll $LOGS/teil$s.log"; i=$((i + 1)); sleep 30
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
else
  echo "$(date +%H:%M) Teilsitzungen nacheinander auf GPU 0"
  for s in "${SH[@]}"; do
    echo "  Teilsitzung $s, Protokoll $LOGS/teil$s.log"
    CUDA_VISIBLE_DEVICES=0 $PY -u "$R" "$EXP" --shard "$s" --results "$RES" > "$LOGS/teil$s.log" 2>&1 || fail=1
  done
fi

echo "$(date +%H:%M) Auswertung über alle Teilsitzungen"
$PY -u "$R" "$EXP" --analysis-only --results "$RES" > "$LOGS/auswertung.log" 2>&1 || fail=1
ARCH="$RES/ergebnisse_$EXP.tar.gz"
( cd "$RES" && tar czf "$ARCH" --exclude='adapters' --exclude='*.pt' --exclude='*.tmp' --exclude='*.part' $(ls -d *_runs_s* 2>/dev/null) logs )
echo "$(date +%H:%M) Fertig$( [ $fail -ne 0 ] && echo ', aber mit Fehlern: siehe Protokolle' ). Archiv zum Herunterladen: $ARCH"
exit $fail
