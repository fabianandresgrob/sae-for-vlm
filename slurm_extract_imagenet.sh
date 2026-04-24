#!/bin/bash
#SBATCH --job-name=extract-imagenet
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --account=taco-vlm

set -e

VENV_PATH="$PROJECT/grob1/LLaVA/sc_venv_template"
REPO_PATH="$PROJECT/grob1/sae-for-vlm"

IMAGENET_HF_DIR="$SCRATCH/grob1/datasets/imagenet_hf"
SAE_EXPERIMENT_DIR="$SCRATCH/grob1/sae/imagenet_clip_l22"
ACTIVATIONS_BASE="${SAE_EXPERIMENT_DIR}/activations"

MODEL_NAME="clip-vit-large-patch14-336"
LAYER=22
POINT="post_mlp_residual"

source "${VENV_PATH}/activate.sh"
cd "${REPO_PATH}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p logs
mkdir -p "${ACTIVATIONS_BASE}/imagenet_train"
mkdir -p "${ACTIVATIONS_BASE}/imagenet_val"

echo "=== Extracting ImageNet train activations ==="
srun --exclusive -n 1 --gres=gpu:1 --cpus-per-task=72 \
    --output="logs/%x_%j_train.out" --error="logs/%x_%j_train.err" \
    python save_activations.py \
        --model_name "${MODEL_NAME}" \
        --attachment_point "${POINT}" \
        --layer ${LAYER} \
        --dataset_name imagenet_hf \
        --data_path "${IMAGENET_HF_DIR}" \
        --split train \
        --random_k 2 \
        --batch_size 128 \
        --num_workers 16 \
        --output_dir "${ACTIVATIONS_BASE}/imagenet_train" \
        --save_every 10000 &

echo "=== Extracting ImageNet val activations ==="
srun --exclusive -n 1 --gres=gpu:1 --cpus-per-task=72 \
    --output="logs/%x_%j_val.out" --error="logs/%x_%j_val.err" \
    python save_activations.py \
        --model_name "${MODEL_NAME}" \
        --attachment_point "${POINT}" \
        --layer ${LAYER} \
        --dataset_name imagenet_hf \
        --data_path "${IMAGENET_HF_DIR}" \
        --split val \
        --random_k 2 \
        --batch_size 128 \
        --num_workers 16 \
        --output_dir "${ACTIVATIONS_BASE}/imagenet_val" \
        --save_every 10000 &

wait
echo "=== Done. Train chunks: $(ls "${ACTIVATIONS_BASE}/imagenet_train"/*.pt 2>/dev/null | wc -l), Val chunks: $(ls "${ACTIVATIONS_BASE}/imagenet_val"/*.pt 2>/dev/null | wc -l) ==="
