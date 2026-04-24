#!/bin/bash
#SBATCH --job-name=extract-llava-ov
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=12:00:00
#SBATCH --partition=booster
#SBATCH --account=taco-vlm

set -e

VENV_PATH="$PROJECT/grob1/sae-for-vlm/.venv"
REPO_PATH="$PROJECT/grob1/sae-for-vlm"

DATA_DIR="/e/scratch/taco-vlm/kim16/LLaVA-OneVision-1.5-Instruct-Data"
SAMPLING_PLAN="${DATA_DIR}/sampling_plan.csv"
OUTPUT_DIR="$SCRATCH/grob1/sae/llava_ov_clip_l22/activations/llava_ov_train"

MODEL_NAME="clip-vit-large-patch14-336"
LAYER=22
POINT="post_mlp_residual"

source "${VENV_PATH}/bin/activate"
cd "${REPO_PATH}"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

echo "=== Extracting LLaVA-OV activations (4 GPUs) ==="
echo "Output: ${OUTPUT_DIR}"

for i in 0 1 2 3; do
    srun --exclusive -n 1 --gres=gpu:1 --cpus-per-task=72 \
        --output="logs/%x_%j_gpu${i}.out" \
        --error="logs/%x_%j_gpu${i}.err" \
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
            --shard_id ${i} \
            --num_shards 4 \
            --resume &
done

wait
echo "=== Done. Chunks: $(ls "${OUTPUT_DIR}"/*.pt 2>/dev/null | wc -l) ==="
