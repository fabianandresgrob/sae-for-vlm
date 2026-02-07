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

mkdir -p logs

# ---- Configure these ----
# Where to store activations and checkpoints (e.g. a scratch filesystem)
export ACTIVATIONS_BASE="$SCRATCH/activations_dir/raw/hf"

# HF token for ImageNet streaming (if not already in ~/.bashrc)
# export HF_TOKEN="hf_xxx"
# -------------------------

# Run full pipeline
./train_sae_from_hf.sh
