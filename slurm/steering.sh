#!/bin/bash
#SBATCH --job-name=steer
#SBATCH --account=PAS2324
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# Usage (repo root on OSC; Ascend = A100, Cardinal = H100):
#   sbatch slurm/steering.sh Qwen/Qwen3-8B --concepts truth_cities
#   sbatch --gpus-per-node=2 slurm/steering.sh Qwen/Qwen3-32B     # 2x A100-40GB for 32B
#   needs outputs/<model>/<concept>/metrics.json from concepts.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
MODEL="$1"; shift

# Weights are re-downloadable, so they live on scratch (purged after 60 days without access).
# TDNV_HF_HOME overrides it, e.g. to read a gated model that only an older cache still holds
# (pair with HF_HUB_OFFLINE=1 so nothing is downloaded there).
export HF_HOME="${TDNV_HF_HOME:-/fs/scratch/PAS2324/zhu.3944/hf-cache}"
export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
export UV_CACHE_DIR=/fs/ess/PAS2324/zhu.3944/uv-cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv
uv sync --frozen
uv run python scripts/run_steering.py --model "$MODEL" "$@"
