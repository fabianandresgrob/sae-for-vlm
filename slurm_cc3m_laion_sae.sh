#!/bin/bash
#SBATCH --job-name=sae-cc3m-laion
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00

# TODO: Adjust partition/account for your cluster
# #SBATCH --partition=gpu
# #SBATCH --account=your_account

set -e

# ---- Configure these ----
VENV_PATH=".venv"

DATASETS_BASE="/lustre/groups/eml/datasets"
CC3M_TRAIN_DIR="${DATASETS_BASE}/cc3m/cc3m/train"
CC3M_VAL_DIR="${DATASETS_BASE}/cc3m/cc3m/valid"
LAION_DIR="${DATASETS_BASE}/laion400m/laion400m-data"

# How many LAION shards to use (~10k images each)
# 200 train shards ≈ 2M images; 10 val shards ≈ 100k images
LAION_TRAIN_SHARDS=200
LAION_VAL_SHARDS=10

# Root for this experiment — kept separate from the old ae.pt in $SCRATCH/sae/
SAE_EXPERIMENT_DIR="$SCRATCH/sae/cc3m_laion_clip_l22"
ACTIVATIONS_BASE="${SAE_EXPERIMENT_DIR}/activations"
CHECKPOINTS_BASE="${SAE_EXPERIMENT_DIR}/checkpoints"
COMBINED_TRAIN_DIR="${ACTIVATIONS_BASE}/combined_train"
COMBINED_VAL_DIR="${ACTIVATIONS_BASE}/combined_val"

MODEL_NAME="clip-vit-large-patch14-336"
LAYER=22
POINT="post_mlp_residual"
# -------------------------

cd "$(dirname "$0")"
mkdir -p logs

source "${VENV_PATH}/bin/activate"

CC3M_TRAIN_OUT="${ACTIVATIONS_BASE}/cc3m_train_activations_${MODEL_NAME}_${LAYER}_${POINT}"
CC3M_VAL_OUT="${ACTIVATIONS_BASE}/cc3m_val_activations_${MODEL_NAME}_${LAYER}_${POINT}"
LAION_TRAIN_OUT="${ACTIVATIONS_BASE}/laion400m_train_activations_${MODEL_NAME}_${LAYER}_${POINT}"
LAION_VAL_OUT="${ACTIVATIONS_BASE}/laion400m_val_activations_${MODEL_NAME}_${LAYER}_${POINT}"

# Step 1: Extract CC3M train activations (2 random patch tokens per image)
echo "=== Step 1: Extracting CC3M train activations ==="
python save_activations_wds.py \
    --shard_dir "${CC3M_TRAIN_DIR}" \
    --output_dir "${ACTIVATIONS_BASE}" \
    --dataset_name cc3m \
    --split train \
    --model_name "${MODEL_NAME}" \
    --layer ${LAYER} \
    --attachment_point "${POINT}" \
    --token_mode random_k \
    --n_random_tokens 2 \
    --batch_size 128 \
    --num_workers 8 \
    --save_every 10000

# Step 2: Extract CC3M val activations (CLS token only)
echo "=== Step 2: Extracting CC3M val activations ==="
python save_activations_wds.py \
    --shard_dir "${CC3M_VAL_DIR}" \
    --output_dir "${ACTIVATIONS_BASE}" \
    --dataset_name cc3m \
    --split val \
    --model_name "${MODEL_NAME}" \
    --layer ${LAYER} \
    --attachment_point "${POINT}" \
    --token_mode cls \
    --batch_size 128 \
    --num_workers 8 \
    --save_every 10000

# Step 3: Extract LAION400M train activations (first LAION_TRAIN_SHARDS shards)
echo "=== Step 3: Extracting LAION400M train activations (${LAION_TRAIN_SHARDS} shards) ==="
python save_activations_wds.py \
    --shard_dir "${LAION_DIR}" \
    --output_dir "${ACTIVATIONS_BASE}" \
    --dataset_name laion400m \
    --split train \
    --model_name "${MODEL_NAME}" \
    --layer ${LAYER} \
    --attachment_point "${POINT}" \
    --token_mode random_k \
    --n_random_tokens 2 \
    --batch_size 128 \
    --num_workers 8 \
    --save_every 10000 \
    --max_shards ${LAION_TRAIN_SHARDS}

# Step 4: Extract LAION400M val activations (next LAION_VAL_SHARDS shards, CLS only)
echo "=== Step 4: Extracting LAION400M val activations (${LAION_VAL_SHARDS} shards) ==="
python save_activations_wds.py \
    --shard_dir "${LAION_DIR}" \
    --output_dir "${ACTIVATIONS_BASE}" \
    --dataset_name laion400m \
    --split val \
    --model_name "${MODEL_NAME}" \
    --layer ${LAYER} \
    --attachment_point "${POINT}" \
    --token_mode cls \
    --batch_size 128 \
    --num_workers 8 \
    --save_every 10000 \
    --shard_start ${LAION_TRAIN_SHARDS} \
    --max_shards ${LAION_VAL_SHARDS}

# Step 5: Merge train and val activation chunks via symlinks
# sae_train.py takes a single --activations_dir, so we combine by symlinking
echo "=== Step 5: Combining CC3M + LAION400M activations ==="
mkdir -p "${COMBINED_TRAIN_DIR}"
mkdir -p "${COMBINED_VAL_DIR}"

# Symlink CC3M chunks, prefixing filenames with 'cc3m_' to avoid collisions
for f in "${CC3M_TRAIN_OUT}"/*.pt; do
    ln -sf "$f" "${COMBINED_TRAIN_DIR}/cc3m_$(basename "$f")"
done
for f in "${CC3M_VAL_OUT}"/*.pt; do
    ln -sf "$f" "${COMBINED_VAL_DIR}/cc3m_$(basename "$f")"
done

# Symlink LAION chunks, prefixing with 'laion_'
for f in "${LAION_TRAIN_OUT}"/*.pt; do
    ln -sf "$f" "${COMBINED_TRAIN_DIR}/laion_$(basename "$f")"
done
for f in "${LAION_VAL_OUT}"/*.pt; do
    ln -sf "$f" "${COMBINED_VAL_DIR}/laion_$(basename "$f")"
done

echo "Combined train chunks: $(ls "${COMBINED_TRAIN_DIR}" | wc -l)"
echo "Combined val chunks:   $(ls "${COMBINED_VAL_DIR}" | wc -l)"

# Step 6: Train BatchTopK SAE on combined activations
EXPANSION=8
ACTIVATION_DIM=1024
DICT_SIZE=$((EXPANSION * ACTIVATION_DIM))
LR=$(python3 -c "import math; print(${EXPANSION} / (125 * math.sqrt(${DICT_SIZE})))")
echo "=== Step 6: Training BatchTopK SAE (x${EXPANSION}, dict_size=${DICT_SIZE}, lr=${LR}) ==="
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

echo "=== Done ==="
