#!/usr/bin/env bash
# One-time setup: virtual environment, caches inside the repo, Jupyter kernel, GPU check.
#   bash code/setup.sh
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the folder that contains code/
cd "$PROJECT"
mkdir -p data notebooks/runs logs .cache
echo "▶ project: $PROJECT"

# keep model weights and pip caches inside the project folder
cat > env.sh <<ENV
export PROJECT_ROOT="$PROJECT"
export XDG_CACHE_HOME="$PROJECT/.cache"
export HF_HOME="$PROJECT/.cache/huggingface"
export TORCH_HOME="$PROJECT/.cache/torch"
export PIP_CACHE_DIR="$PROJECT/.cache/pip"
export TOKENIZERS_PARALLELISM=false
ENV
source env.sh

if [ ! -d .venv ]; then
  echo "▶ creating .venv"
  python3 -m venv .venv || { echo "✗ python3 -m venv failed (install python3-venv)"; exit 1; }
fi
source .venv/bin/activate
python -m pip install --no-cache-dir --upgrade pip
echo "▶ installing requirements (the PyTorch CUDA wheel is ~2.5 GB)"
python -m pip install --no-cache-dir -r code/requirements.txt
python -m pip install --no-cache-dir pyvips-binary pyvips || echo "  (optional pyvips not installed; Moondream2 falls back to PIL)"
python -m pip freeze > logs/pip_freeze.txt

echo "▶ registering Jupyter kernel 'Python (aeroverify_vlm)'"
python -m ipykernel install --user --name aeroverify-vlm --display-name "Python (aeroverify_vlm)"

python - <<'PY'
import torch, transformers, sys
print("   python      ", sys.version.split()[0])
print("   torch       ", torch.__version__, "| CUDA", torch.cuda.is_available())
print("   transformers", transformers.__version__)
for i in range(torch.cuda.device_count()):
    print(f"   GPU {i}       ", torch.cuda.get_device_name(i))
PY
echo "✓ setup complete. Next:  bash code/run_experiments.sh smoke"
