#!/bin/bash
# Submit parallel LLaVA-OV extraction jobs across multiple nodes.
#
# Usage:
#   ./submit_extract_llava_ov.sh           # 1 node  (4 GPUs)
#   ./submit_extract_llava_ov.sh 4         # 4 nodes (16 GPUs)

NUM_NODES=${1:-1}

echo "Submitting ${NUM_NODES} extraction job(s) (${NUM_NODES} nodes x 4 GPUs = $((NUM_NODES * 4)) total shards)"

for i in $(seq 0 $((NUM_NODES - 1))); do
    sbatch --export=ALL,NODE_ID=${i},NUM_NODES=${NUM_NODES} slurm_extract_llava_ov.sh
done
