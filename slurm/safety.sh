#!/bin/bash
#SBATCH --job-name=tdnv-safety
#SBATCH --account=PAS2324
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

# Usage (from the repo root on OSC Ascend):
#   sbatch slurm/safety.sh                       # default model list, both modes
#   MODELS="Qwen/Qwen3-8B" sbatch slurm/safety.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
MODELS="${MODELS:-meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-8B}"
N="${N:-1000}"

nvidia-smi --query-gpu=name,memory.total --format=csv
uv sync --frozen

for model in $MODELS; do
  for mode in prompt response; do
    echo "=== $model / $mode"
    uv run python scripts/run_safety.py --model "$model" --mode "$mode" --n "$N" "$@"
  done
done
