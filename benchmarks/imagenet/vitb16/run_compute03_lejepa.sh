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

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="$REPO_ROOT/benchmarks/imagenet/vitb16"
cd "$SCRIPT_DIR"

TRAIN_DIR="${TRAIN_DIR:-/datasets/imagenet1k/train}"
VAL_DIR="${VAL_DIR:-/datasets/imagenet1k/val}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_DIR="${LOG_DIR:-$HOME/gabriel/lightly-ssl/logs/lejepa}"
SMOKE_TEST="${SMOKE_TEST:-0}"
SKIP_POST_EVALS="${SKIP_POST_EVALS:-1}"

mkdir -p "$LOG_DIR"

cmd=(
  "$SCRIPT_DIR/main.py"
  --methods lejepa
  --train-dir "$TRAIN_DIR"
  --val-dir "$VAL_DIR"
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

PYTHON_BIN="${PYTHON_BIN:-$HOME/gabriel/lightly-ssl/worktrees/macos-test-fast/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

exec env PYTHONPATH="$REPO_ROOT" srun "$PYTHON_BIN" "${cmd[@]}"
