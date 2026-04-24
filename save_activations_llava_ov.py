"""
Extract CLIP activations from LLaVA-OneVision-1.5-Instruct-Data (parquet format).
Reads a sampling plan CSV (from count_llava_ov_samples.py) to know how many
images to extract per subdataset.

Output format is compatible with sae_train.py and ActivationsDataset.

Usage:
    python save_activations_llava_ov.py \
        --data_dir /e/scratch/taco-vlm/kim16/LLaVA-OneVision-1.5-Instruct-Data \
        --sampling_plan /e/scratch/taco-vlm/kim16/LLaVA-OneVision-1.5-Instruct-Data/sampling_plan.csv \
        --output_dir $SCRATCH/grob1/sae/llava_ov/activations \
        --model_name clip-vit-large-patch14-336 \
        --layer 22 \
        --attachment_point post_mlp_residual \
        --token_mode cls \
        --resume
"""

import argparse
import csv
import io
import logging
import os
import time
from pathlib import Path

import torch
from PIL import Image

from models.clip import Clip

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--sampling_plan", required=True,
                        help="CSV from count_llava_ov_samples.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="clip-vit-large-patch14-336")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--attachment_point", default="post_mlp_residual")
    parser.add_argument("--token_mode", default="cls",
                        choices=["cls", "mean_pool", "random_k"])
    parser.add_argument("--n_random_tokens", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=10000,
                        help="Save a .pt chunk every N images")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Index of this GPU worker (0-indexed)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of parallel GPU workers")
    return parser.parse_args()


def load_sampling_plan(csv_path: str) -> dict:
    """Returns {subdataset_name: n_to_sample} for visual (non-text-only) datasets."""
    plan = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["text_only"]) == 0 and int(row["sampled"]) > 0:
                plan[row["subdataset"]] = int(row["sampled"])
    return plan


def decode_image(image_field) -> Image.Image | None:
    """Decode HuggingFace parquet image field (struct with 'bytes') to PIL."""
    try:
        if isinstance(image_field, dict):
            raw = image_field.get("bytes") or image_field.get("path")
            if isinstance(raw, bytes):
                return Image.open(io.BytesIO(raw)).convert("RGB")
        if isinstance(image_field, bytes):
            return Image.open(io.BytesIO(image_field)).convert("RGB")
    except Exception:
        return None
    return None


def pool_activations(act_tensor: torch.Tensor, token_mode: str, n_random_tokens: int) -> torch.Tensor:
    if token_mode == "cls":
        return act_tensor[:, 0, :]
    elif token_mode == "mean_pool":
        return act_tensor.mean(dim=1)
    elif token_mode == "random_k":
        B, S, D = act_tensor.shape
        idx = torch.randint(1, S, (B, n_random_tokens))
        sampled = torch.stack([act_tensor[i, idx[i], :] for i in range(B)])
        return sampled.reshape(-1, D)
    raise ValueError(f"Unknown token_mode: {token_mode}")


def save_chunk(buffer: list, save_count: int, output_dir: str, args) -> None:
    chunk = torch.cat(buffer, dim=0)
    fname = (
        f"llava_ov_train_activations_{args.model_name}"
        f"_{args.layer}_{args.attachment_point}_part{save_count + 1}.pt"
    )
    path = os.path.join(output_dir, fname)
    torch.save(torch.tensor(chunk.cpu().numpy()), path)
    log.info(f"Saved chunk {save_count + 1}: {chunk.shape} -> {path}")


def iter_parquet_images(subdir: Path, n_target: int):
    """
    Yield PIL images from parquet files, sampling approximately n_target images
    evenly across all rows using a fixed stride.
    """
    import pyarrow.parquet as pq

    files = sorted(subdir.glob("*.parquet"))
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    stride = max(1, total // n_target)

    global_idx = 0
    emitted = 0
    for f in files:
        table = pq.read_table(f, columns=["image"])
        for row_idx in range(len(table)):
            if global_idx % stride == 0 and emitted < n_target:
                img = decode_image(table["image"][row_idx].as_py())
                if img is not None:
                    yield img
                    emitted += 1
            global_idx += 1


def extract_subdataset(subdir: Path, n_target: int, output_dir: str, clip, register_key: str, args) -> None:
    done_file = os.path.join(output_dir, f".done_{subdir.name}")
    if args.resume and os.path.exists(done_file):
        log.info(f"Skipping {subdir.name} (already done)")
        return

    log.info(f"Extracting {subdir.name}: target {n_target:,} images")
    buffer = []
    save_count = 0
    images_processed = 0
    start = time.time()

    batch = []
    for img in iter_parquet_images(subdir, n_target):
        batch.append(img)
        if len(batch) < args.batch_size:
            continue

        inputs = clip.processor(images=batch, return_tensors="pt", padding=True)
        with torch.no_grad():
            clip.encode(inputs)
            act = clip.register[register_key]

        pooled = pool_activations(torch.cat(act, dim=0), args.token_mode, args.n_random_tokens)
        buffer.append(pooled)
        images_processed += len(batch)
        batch = []

        if images_processed >= args.save_every * (save_count + 1):
            save_chunk(buffer, save_count, output_dir, args)
            buffer = []
            save_count += 1

    # Flush remaining batch
    if batch:
        inputs = clip.processor(images=batch, return_tensors="pt", padding=True)
        with torch.no_grad():
            clip.encode(inputs)
            act = clip.register[register_key]
        pooled = pool_activations(torch.cat(act, dim=0), args.token_mode, args.n_random_tokens)
        buffer.append(pooled)
        images_processed += len(batch)

    if buffer:
        save_chunk(buffer, save_count, output_dir, args)
        save_count += 1

    elapsed = time.time() - start
    log.info(f"Done {subdir.name}: {images_processed:,} images, {save_count} chunks in {elapsed/60:.1f} min")
    open(done_file, "w").close()


def main():
    args = parse_args()

    plan = load_sampling_plan(args.sampling_plan)

    # Split subdatasets across shards (round-robin)
    all_items = list(plan.items())
    shard_items = all_items[args.shard_id::args.num_shards]
    log.info(
        f"Shard {args.shard_id}/{args.num_shards}: "
        f"{len(shard_items)}/{len(all_items)} subdatasets, "
        f"{sum(n for _, n in shard_items):,} images"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    log.info(f"Loading model openai/{args.model_name} on {args.device}")
    clip = Clip(args.model_name, args.device)
    clip.attach(args.attachment_point, args.layer, sae=None)
    register_key = f"{args.attachment_point}_{args.layer}"

    base = Path(args.data_dir)
    total_extracted = 0
    for subdataset, n_target in shard_items:
        subdir = base / subdataset
        if not subdir.exists():
            log.warning(f"Subdataset directory not found: {subdir}, skipping")
            continue
        extract_subdataset(subdir, n_target, args.output_dir, clip, register_key, args)
        total_extracted += n_target

    log.info(f"All done. Total target: {total_extracted:,} images -> {args.output_dir}")


if __name__ == "__main__":
    main()
