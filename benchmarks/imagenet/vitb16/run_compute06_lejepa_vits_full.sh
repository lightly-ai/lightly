#!/usr/bin/env bash
#SBATCH --job-name=lejepa-vits-full
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=/logs/gabriel/%x-%j.out
#SBATCH --error=/logs/gabriel/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="$REPO_ROOT/benchmarks/imagenet/vitb16"
cd "$SCRIPT_DIR"

TRAIN_DIR="${TRAIN_DIR:-/datasets/imagenet/train}"
VAL_DIR="${VAL_DIR:-/datasets/imagenet/val}"
LOG_DIR="${LOG_DIR:-/logs/gabriel/lejepa_vits_full}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-32}"
EPOCHS="${EPOCHS:-100}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/venvs/lightly-ssl/bin/python}"

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing or non-executable python env: $PYTHON_BIN" >&2
  exit 1
fi

exec env PYTHONPATH="$REPO_ROOT" srun "$PYTHON_BIN" "$SCRIPT_DIR/main.py" \
  --methods lejepa \
  --train-dir "$TRAIN_DIR" \
  --val-dir "$VAL_DIR" \
  --log-dir "$LOG_DIR" \
  --batch-size-per-device "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --num-workers "$NUM_WORKERS" \
  --accelerator gpu \
  --devices 1 \
  --precision bf16-mixed \
  --strategy auto \
  --eval-method mae
