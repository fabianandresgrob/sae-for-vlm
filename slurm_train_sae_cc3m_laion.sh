#!/bin/bash
#SBATCH --job-name=train-sae-cc3m-laion
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00

# TODO: Adjust partition/account for your cluster
# #SBATCH --partition=gpu
# #SBATCH --account=your_account

set -e

# ---- Configure these ----
VENV_PATH=".venv"

SAE_EXPERIMENT_DIR="$SCRATCH/sae/cc3m_laion_clip_l22"
ACTIVATIONS_BASE="${SAE_EXPERIMENT_DIR}/activations"
CHECKPOINTS_BASE="${SAE_EXPERIMENT_DIR}/checkpoints"
COMBINED_TRAIN_DIR="${ACTIVATIONS_BASE}/combined_train"
COMBINED_VAL_DIR="${ACTIVATIONS_BASE}/combined_val"
# -------------------------

cd "$(dirname "$0")"
mkdir -p logs

source "${VENV_PATH}/bin/activate"

EXPANSION=8
ACTIVATION_DIM=1024
DICT_SIZE=$((EXPANSION * ACTIVATION_DIM))
LR=$(python3 -c "import math; print(${EXPANSION} / (125 * math.sqrt(${DICT_SIZE})))")

echo "=== Training BatchTopK SAE (x${EXPANSION}, dict_size=${DICT_SIZE}, lr=${LR}) ==="
echo "Train dir: ${COMBINED_TRAIN_DIR} ($(ls "${COMBINED_TRAIN_DIR}" | wc -l) chunks)"
echo "Val dir:   ${COMBINED_VAL_DIR} ($(ls "${COMBINED_VAL_DIR}" | wc -l) chunks)"

python sae_train.py \
    --sae_model batch_top_k \
    --activations_dir "${COMBINED_TRAIN_DIR}" \
    --val_activations_dir "${COMBINED_VAL_DIR}" \
    --checkpoints_dir "${CHECKPOINTS_BASE}" \
    --expansion_factor ${EXPANSION} \
    --steps 100000 \
    --save_steps 20000 \
    --log_steps 1000 \
    --batch_size 4096 \
    --k 20 \
    --lr ${LR} \
    --auxk_alpha 0.03 \
    --decay_start 99999

echo "=== Done. Final SAE at ${CHECKPOINTS_BASE}/combined_train_batch_top_k_20_x${EXPANSION}/trainer_0/ae.pt ==="
