#!/bin/bash
#SBATCH --job-name=sae-vlm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# TODO: Adjust for your cluster (run `sinfo` to see available partitions)
# #SBATCH --partition=gpu
# #SBATCH --account=your_account

set -e

# ---- Configure these ----
# Path to your uv venv (create with: uv venv && uv pip install -r requirements.txt)
VENV_PATH=".venv"

# Where to store activations and checkpoints (e.g. a scratch filesystem)
export ACTIVATIONS_BASE="$SCRATCH/activations_dir/raw/hf"
export CHECKPOINTS_BASE="$SCRATCH/checkpoints_dir"

# HuggingFace download cache (model weights + dataset streaming cache)
# Redirect to scratch to avoid filling home directory quota
# Note: HF_HOME is NOT overridden, so the auth token from ~/.cache/huggingface/token is still found
export HF_HUB_CACHE="$SCRATCH/.cache/huggingface/hub"
export HF_DATASETS_CACHE="$SCRATCH/.cache/huggingface/datasets"
# -------------------------

# cd to repo directory (SLURM runs from submission dir by default, but be explicit)
cd "$(dirname "$0")"

mkdir -p logs

# Activate Python environment
source "${VENV_PATH}/bin/activate"

# Run full pipeline
./train_sae_from_hf.sh
