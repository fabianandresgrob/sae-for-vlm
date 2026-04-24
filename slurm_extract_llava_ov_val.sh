#!/bin/bash
#SBATCH --job-name=extract-llava-ov-val
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --account=taco-vlm

set -e

NODE_ID=${NODE_ID:-0}
NUM_NODES=${NUM_NODES:-1}
TOTAL_SHARDS=$((NUM_NODES * 4))
BASE_SHARD=$((NODE_ID * 4))

VENV_PATH="$PROJECT/grob1/LLaVA/sc_venv_template"
REPO_PATH="$PROJECT/grob1/sae-for-vlm"

DATA_DIR="/e/scratch/taco-vlm/kim16/LLaVA-OneVision-1.5-Instruct-Data"
SAMPLING_PLAN="${REPO_PATH}/sampling_plan_val.csv"
OUTPUT_DIR="$SCRATCH/grob1/sae/llava_ov_clip_l22/activations/llava_ov_val"

MODEL_NAME="clip-vit-large-patch14-336"
LAYER=22
POINT="post_mlp_residual"

source "${VENV_PATH}/activate.sh"
cd "${REPO_PATH}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

echo "=== Extracting LLaVA-OV val activations (node ${NODE_ID}/${NUM_NODES}) ==="

for i in 0 1 2 3; do
    SHARD_ID=$((BASE_SHARD + i))
    srun --exclusive -n 1 --gres=gpu:1 --cpus-per-task=72 \
        --output="logs/%x_%j_shard${SHARD_ID}.out" \
        --error="logs/%x_%j_shard${SHARD_ID}.err" \
        python save_activations_llava_ov.py \
            --data_dir "${DATA_DIR}" \
            --sampling_plan "${SAMPLING_PLAN}" \
            --output_dir "${OUTPUT_DIR}" \
            --model_name "${MODEL_NAME}" \
            --layer ${LAYER} \
            --attachment_point "${POINT}" \
            --token_mode cls \
            --batch_size 128 \
            --save_every 10000 \
            --shard_id ${SHARD_ID} \
            --num_shards ${TOTAL_SHARDS} \
            --row_offset 1 \
            --resume &
done

wait
echo "=== Val extraction done. Chunks: $(ls "${OUTPUT_DIR}"/*.pt 2>/dev/null | wc -l) ==="
