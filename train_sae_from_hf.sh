#!/bin/bash
# Full pipeline: extract CLIP activations from HuggingFace ImageNet + train SAE
#
# Prerequisites:
#   huggingface-cli login  (must have accepted ILSVRC/imagenet-1k license)
#   pip install datasets

set -e

MODEL_NAME="clip-vit-large-patch14-336"
LAYER=22
POINT="post_mlp_residual"
ACTIVATIONS_BASE="./activations_dir/raw/hf"
TRAIN_DIR="${ACTIVATIONS_BASE}/imagenet_train_activations_${MODEL_NAME}_${LAYER}_${POINT}"
VAL_DIR="${ACTIVATIONS_BASE}/imagenet_validation_activations_${MODEL_NAME}_${LAYER}_${POINT}"

# Step 1: Extract training activations (2 random patch tokens per image)
echo "=== Step 1: Extracting training activations ==="
python save_activations_hf.py \
    --output_dir "${ACTIVATIONS_BASE}" \
    --model_name "${MODEL_NAME}" \
    --layer ${LAYER} \
    --attachment_point "${POINT}" \
    --split train \
    --token_mode random_k \
    --n_random_tokens 2 \
    --batch_size 128 \
    --save_every 10000

# Step 2: Extract validation activations (CLS token only)
echo "=== Step 2: Extracting validation activations ==="
python save_activations_hf.py \
    --output_dir "${ACTIVATIONS_BASE}" \
    --model_name "${MODEL_NAME}" \
    --layer ${LAYER} \
    --attachment_point "${POINT}" \
    --split validation \
    --token_mode cls \
    --batch_size 128 \
    --save_every 10000

# Step 3: Train BatchTopK SAE
# Input dim: 1024, expansion: 16x -> dict_size = 16384
# LR = 16 / (125 * sqrt(16384)) = 0.001
echo "=== Step 3: Training BatchTopK SAE ==="
python sae_train.py \
    --sae_model batch_top_k \
    --activations_dir "${TRAIN_DIR}" \
    --val_activations_dir "${VAL_DIR}" \
    --checkpoints_dir "./checkpoints_dir/batch_top_k_20_x16" \
    --expansion_factor 16 \
    --steps 100000 \
    --save_steps 20000 \
    --log_steps 1000 \
    --batch_size 4096 \
    --k 20 \
    --lr 0.001 \
    --auxk_alpha 0.03 \
    --decay_start 99999

echo "=== Done ==="
