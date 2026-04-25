#!/bin/bash
#SBATCH --job-name=train-saes
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --account=taco-vlm

# Train one SAE per GPU in parallel.
# By default trains all 3 datasets (ImageNet, CC3M+LAION, LLaVA-OV) on GPUs 0-2.
# Edit the DATASETS array and CONFIGS below to change which SAEs to train.
# GPU 3 is left idle unless a 4th dataset is added.
#
# Usage:
#   sbatch slurm_train_saes.sh

set -e

VENV_PATH="$PROJECT/grob1/LLaVA/sc_venv_template"
REPO_PATH="$PROJECT/grob1/sae-for-vlm"

source "${VENV_PATH}/activate.sh"
cd "${REPO_PATH}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

mkdir -p logs

EXPANSION=8
ACTIVATION_DIM=1024
DICT_SIZE=$((EXPANSION * ACTIVATION_DIM))
LR=$(python3 -c "import math; print(${EXPANSION} / (125 * math.sqrt(${DICT_SIZE})))")

BASE="$SCRATCH/grob1/sae"

train_sae() {
    local GPU=$1
    local TRAIN_DIR=$2
    local VAL_DIR=$3
    local CKPT_DIR=$4
    local NAME=$5

    srun --exclusive -n 1 --gres=gpu:1 --cpus-per-task=72 \
        --output="logs/%x_%j_${NAME}.out" \
        --error="logs/%x_%j_${NAME}.err" \
        python sae_train.py \
            --sae_model batch_top_k \
            --activations_dir "${TRAIN_DIR}" \
            --val_activations_dir "${VAL_DIR}" \
            --checkpoints_dir "${CKPT_DIR}" \
            --expansion_factor ${EXPANSION} \
            --steps 100000 \
            --save_steps 20000 \
            --log_steps 1000 \
            --batch_size 4096 \
            --k 20 \
            --lr ${LR} \
            --auxk_alpha 0.03 \
            --decay_start 99999 \
            --device "cuda:${GPU}" &
}

echo "=== Training SAEs (x${EXPANSION}, dict_size=${DICT_SIZE}, lr=${LR}) ==="

train_sae 0 \
    "${BASE}/imagenet_clip_l22/activations/imagenet_train" \
    "${BASE}/imagenet_clip_l22/activations/imagenet_val" \
    "${BASE}/imagenet_clip_l22/checkpoints" \
    "imagenet"

train_sae 1 \
    "${BASE}/cc3m_laion_clip_l22/activations/combined_train" \
    "${BASE}/cc3m_laion_clip_l22/activations/combined_val" \
    "${BASE}/cc3m_laion_clip_l22/checkpoints" \
    "cc3m_laion"

train_sae 2 \
    "${BASE}/llava_ov_clip_l22/activations/llava_ov_train" \
    "${BASE}/llava_ov_clip_l22/activations/llava_ov_val" \
    "${BASE}/llava_ov_clip_l22/checkpoints" \
    "llava_ov"

# GPU 3 is idle — add a 4th dataset here if needed

wait
echo "=== All SAE trainings done ==="
