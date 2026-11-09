#!/usr/bin/env bash
# Einmalig auf dem gemieteten GPU-Server: Pakete installieren, Modelle vorab laden, Platz und GPUs prüfen.
# Aufruf aus dem Repository-Ordner:  bash notebooks/gpu_server/setup.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/env.sh"

echo "== GPUs"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || { echo "Keine NVIDIA-GPU gefunden."; exit 1; }
echo "== Python: $($PY --version)"

mkdir -p "$HF_HOME"
FREE_GB=$(df -BG --output=avail "$HF_HOME" | tail -1 | tr -dc 0-9)
echo "== Freier Platz unter $HF_HOME: ${FREE_GB} GB (nötig etwa 80 GB, besser 150)"
if [ "$FREE_GB" -lt 80 ]; then
  echo "Zu wenig Platz. Ein größeres Volume einhängen oder WORK=/pfad/zum/volume setzen und neu starten."; exit 1
fi

echo "== PyTorch"
if ! $PY - <<'PYEOF'
import sys
try:
    import torch
    from packaging import version
except ImportError:
    sys.exit(1)
ok = torch.cuda.is_available() and version.parse(torch.__version__.split("+")[0]) >= version.parse("2.6")
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available())
sys.exit(0 if ok else 1)
PYEOF
then
  # passendes CUDA-Build zum installierten Treiber wählen (das neueste torch von PyPI braucht oft einen neueren Treiber)
  CUDA_DRV=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | grep -o "[0-9.]*$" || echo 0)
  case "$CUDA_DRV" in
    13.*|12.[89]*) IDX=cu128 ;;
    12.[67]*) IDX=cu126 ;;
    12.[45]*) IDX=cu124 ;;
    *) echo "Treiber zu alt (CUDA $CUDA_DRV). Bitte ein Image mit aktuellem PyTorch oder Treiber für CUDA ab 12.4 wählen."; exit 1 ;;
  esac
  echo "PyTorch fehlt, ist zu alt oder sieht keine GPU: installiere torch für $IDX (Treiber: CUDA $CUDA_DRV)"
  $PY -m pip install -q -U "torch>=2.6" --index-url "https://download.pytorch.org/whl/$IDX"
  $PY -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, '| CUDA ok')" \
    || { echo "torch sieht weiterhin keine GPU. Bitte ein PyTorch-Image des Anbieters nehmen."; exit 1; }
fi

echo "== Pakete"
$PY -m pip install -q -U "transformers>=5.5" "peft>=0.13" accelerate bitsandbytes scipy pandas matplotlib packaging huggingface_hub
$PY -m pip uninstall -y -q torchao 2>/dev/null || true   # altes torchao verträgt sich nicht mit aktuellem peft

echo "== Modelle vorab laden (Qwen3-1.7B etwa 4 GB, gemma-4-31B-it etwa 63 GB)"
MODELS="${MODELS:-Qwen/Qwen3-1.7B google/gemma-4-31B-it}"
$PY - $MODELS <<'PYEOF'
import sys
from huggingface_hub import snapshot_download
for m in sys.argv[1:]:
    p = snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "tokenizer*"])
    print("ok", m, "->", p, flush=True)
PYEOF

echo "== Kurztest"
$PY - <<'PYEOF'
import torch, transformers, peft
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.get_device_properties(i).total_memory / 1e9:.0f} GB")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); print("bf16-Matmul ok:", float((x @ x).float().abs().mean()) > 0)
PYEOF
echo "Einrichtung fertig. Weiter mit:  bash notebooks/gpu_server/run.sh superposition"
