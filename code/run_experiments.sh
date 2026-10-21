#!/usr/bin/env bash
# AeroVerify-VLM — runs notebooks/aeroverify_vlm_v2.ipynb headless (after: bash code/setup.sh).
#   bash code/run_experiments.sh smoke            # 1–2 min: synthetic data + mock models (no GPU)
#   bash code/run_experiments.sh pilot            # ~5–10 min: 18 real FloodNet images, Moondream2 only
#   bash code/run_experiments.sh data             # FloodNet annotations + sampled images, sample (once)
#   bash code/run_experiments.sh moondream2 0     # full inference on GPU 0   (run inside tmux)
#   bash code/run_experiments.sh smolvlm 1        # full inference on GPU 1   (run inside tmux)
#   bash code/run_experiments.sh analyze          # tables + figures (CPU)
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # the folder that contains code/
cd "$PROJECT"
source env.sh
source .venv/bin/activate
export PROJECT_ROOT="$PROJECT"
NB=notebooks/aeroverify_vlm_v2.ipynb
MODE="${1:?usage: run_experiments.sh smoke|pilot|data|moondream2 <gpu>|smolvlm <gpu>|analyze}"
STAMP="$(date +%Y%m%d_%H%M%S)"
case "$MODE" in
  smoke)   export AV_DRY_RUN=1 AV_STAGES=data,run,analyze;                       OUT="v2_smoke_${STAMP}" ;;
  pilot)   export CUDA_VISIBLE_DEVICES="${2:-0}" AV_STAGES=data,run,analyze AV_DATASETS=floodnet \
                  AV_FN_DEV=6 AV_FN_EVAL=12 AV_ROBUSTNESS=0 AV_MODELS=moondream2 AV_ANALYZE_MODELS=moondream2
           OUT="v2_pilot_${STAMP}" ;;
  data)    export AV_STAGES=data;                                                OUT="v2_data_${STAMP}" ;;
  moondream2|smolvlm)
           export CUDA_VISIBLE_DEVICES="${2:-0}" AV_MODELS="$MODE" AV_STAGES=run; OUT="v2_run_${MODE}_${STAMP}" ;;
  analyze) export AV_STAGES=analyze;                                             OUT="aeroverify_vlm_v2_results" ;;
  *) echo "unknown mode: $MODE"; exit 1 ;;
esac
mkdir -p notebooks/runs logs
echo "▶ $MODE → notebooks/runs/${OUT}.ipynb   (log: logs/${OUT}.log)"
jupyter nbconvert --to notebook --execute "$NB" --output-dir notebooks/runs --output "${OUT}" \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=aeroverify-vlm 2>&1 | tee "logs/${OUT}.log"
echo "✓ done: notebooks/runs/${OUT}.ipynb"
case "$MODE" in
  smoke)         SUM="dry_run_v2/outputs_v2/summary.txt" ;;
  pilot|analyze) SUM="outputs_v2/summary.txt" ;;
  *)             SUM="" ;;
esac
if [ -n "$SUM" ] && [ -f "$SUM" ]; then echo; echo "──────── summary ($SUM) ────────"; cat "$SUM"; fi
