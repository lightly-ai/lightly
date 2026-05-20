#!/usr/bin/env bash
#SBATCH --job-name=lejepa-vitb16
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --output=/home/lightly/gabriel/lightly-ssl/logs/%x-%j.out
#SBATCH --error=/home/lightly/gabriel/lightly-ssl/logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_DIR="${LOG_DIR:-$HOME/gabriel/lightly-ssl/logs/lejepa}"
SMOKE_TEST="${SMOKE_TEST:-0}"
SKIP_POST_EVALS="${SKIP_POST_EVALS:-1}"

mkdir -p "$LOG_DIR"

cmd=(
  main.py
  --methods lejepa
  --train-dir /datasets/imagenet/train
  --val-dir /datasets/imagenet/val
  --log-dir "$LOG_DIR"
  --batch-size-per-device "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --num-workers "$NUM_WORKERS"
  --accelerator gpu
  --devices 4
  --precision bf16-mixed
  --strategy ddp_find_unused_parameters_true
)

if [[ "$SKIP_POST_EVALS" == "1" ]]; then
  cmd+=(--skip-knn-eval --skip-linear-eval --skip-finetune-eval)
fi

if [[ "$SMOKE_TEST" == "1" ]]; then
  cmd+=(--smoke-test)
fi

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

if [[ -x "$UV_BIN" ]]; then
  exec srun "$UV_BIN" run --no-sync python "${cmd[@]}"
fi

exec srun python3 "${cmd[@]}"
