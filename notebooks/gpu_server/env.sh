# Gemeinsame Einstellungen für setup.sh und run.sh: großes Volume nutzen, wenn vorhanden (RunPod: /workspace).
if [ -z "${WORK:-}" ]; then
  if [ -d /workspace ] && [ -w /workspace ]; then WORK=/workspace; else WORK="$HOME"; fi
fi
export WORK
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"          # dieselbe Wahl trifft das Notebook außerhalb von Colab
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-python3}"
