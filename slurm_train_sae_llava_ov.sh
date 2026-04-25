#!/bin/bash
#SBATCH --job-name=train-sae-llava-ov
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --partition=booster
#SBATCH --account=taco-vlm

set -e

VENV_PATH="$PROJECT/grob1/LLaVA/sc_venv_template"
REPO_PATH="$PROJECT/grob1/sae-for-vlm"

SAE_EXPERIMENT_DIR="$SCRATCH/grob1/sae/llava_ov_clip_l22"
ACTIVATIONS_DIR="${SAE_EXPERIMENT_DIR}/activations/llava_ov_train"
VAL_ACTIVATIONS_DIR="${SAE_EXPERIMENT_DIR}/activations/llava_ov_val"
CHECKPOINTS_DIR="${SAE_EXPERIMENT_DIR}/checkpoints"

source "${VENV_PATH}/activate.sh"
cd "${REPO_PATH}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

EXPANSION=8
ACTIVATION_DIM=1024
DICT_SIZE=$((EXPANSION * ACTIVATION_DIM))
LR=$(python3 -c "import math; print(${EXPANSION} / (125 * math.sqrt(${DICT_SIZE})))")

echo "=== Training BatchTopK SAE (x${EXPANSION}, dict_size=${DICT_SIZE}, lr=${LR}) ==="
echo "Train: ${ACTIVATIONS_DIR} ($(ls "${ACTIVATIONS_DIR}"/*.pt 2>/dev/null | wc -l) chunks)"
echo "Val:   ${VAL_ACTIVATIONS_DIR} ($(ls "${VAL_ACTIVATIONS_DIR}"/*.pt 2>/dev/null | wc -l) chunks)"

python sae_train.py \
    --sae_model batch_top_k \
    --activations_dir "${ACTIVATIONS_DIR}" \
    --val_activations_dir "${VAL_ACTIVATIONS_DIR}" \
    --checkpoints_dir "${CHECKPOINTS_DIR}" \
    --expansion_factor ${EXPANSION} \
    --steps 100000 \
    --save_steps 20000 \
    --log_steps 1000 \
    --batch_size 4096 \
    --k 20 \
    --lr ${LR} \
    --auxk_alpha 0.03 \
    --decay_start 99999

echo "=== Done. SAE at ${CHECKPOINTS_DIR} ==="
