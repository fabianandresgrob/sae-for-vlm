"""
Extract CLIP activations from WebDataset-format datasets (CC3M, LAION400M, etc.)
stored as local .tar shards.

Output format is fully compatible with sae_train.py and ActivationsDataset.

Usage:
    # CC3M train (all shards, 2 random patch tokens per image)
    python save_activations_wds.py \
        --shard_dir /lustre/groups/eml/datasets/cc3m/cc3m/train \
        --dataset_name cc3m --split train \
        --token_mode random_k --n_random_tokens 2

    # CC3M val (all shards, CLS token)
    python save_activations_wds.py \
        --shard_dir /lustre/groups/eml/datasets/cc3m/cc3m/valid \
        --dataset_name cc3m --split val \
        --token_mode cls

    # LAION400M subset (first 200 shards for train, last 10 shards for val)
    python save_activations_wds.py \
        --shard_dir /lustre/groups/eml/datasets/laion400m/laion400m-data \
        --dataset_name laion400m --split train \
        --max_shards 200 --token_mode random_k --n_random_tokens 2

    python save_activations_wds.py \
        --shard_dir /lustre/groups/eml/datasets/laion400m/laion400m-data \
        --dataset_name laion400m --split val \
        --shard_start 200 --max_shards 10 --token_mode cls
"""

import argparse
import logging
import os
import time
from pathlib import Path
import glob

import torch
from models.clip import Clip


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Save CLIP activations from WebDataset shards")
    parser.add_argument("--shard_dir", type=str, required=True,
                        help="Directory containing .tar shard files")
    parser.add_argument("--output_dir", type=str, default="./activations_dir/raw/wds",
                        help="Root output directory; activations are saved in a named subdirectory")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name used in output paths (e.g. cc3m, laion400m)")
    parser.add_argument("--split", type=str, default="train",
                        help="Split label used in output paths (e.g. train, val)")
    parser.add_argument("--model_name", type=str, default="clip-vit-large-patch14-336")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--attachment_point", type=str, default="post_mlp_residual")
    parser.add_argument("--token_mode", type=str, default="random_k",
                        choices=["cls", "mean_pool", "random_k", "all"],
                        help="How to pool tokens per image")
    parser.add_argument("--n_random_tokens", type=int, default=2,
                        help="Random patch tokens per image (only used with --token_mode random_k)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=10000,
                        help="Save a .pt chunk every N images processed")
    parser.add_argument("--shard_start", type=int, default=0,
                        help="Skip the first N shards (use to carve out a val split from LAION)")
    parser.add_argument("--max_shards", type=int, default=None,
                        help="Use at most N shards after shard_start (for LAION subset)")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Stop after processing N images total")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes for shard decoding")
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def pool_activations(activations_tensor, token_mode, n_random_tokens):
    """Pool [batch, seq_len, hidden_dim] -> [N, hidden_dim]."""
    if token_mode == "cls":
        return activations_tensor[:, 0, :]
    elif token_mode == "mean_pool":
        return torch.mean(activations_tensor, dim=1)
    elif token_mode == "random_k":
        batch_size, seq_len, hidden_dim = activations_tensor.shape
        # Sample from patch tokens only (skip CLS at index 0)
        patch_indices = torch.randint(1, seq_len, (batch_size, n_random_tokens))
        sampled = torch.stack([activations_tensor[i, patch_indices[i], :] for i in range(batch_size)])
        return sampled.reshape(-1, hidden_dim)
    elif token_mode == "all":
        batch_size, seq_len, hidden_dim = activations_tensor.shape
        return activations_tensor.reshape(batch_size * seq_len, hidden_dim)
    else:
        raise ValueError(f"Unknown token_mode: {token_mode}")


def save_chunk(buffer, save_count, args, output_subdir):
    chunk = torch.cat(buffer, dim=0)
    filename = (
        f"{args.dataset_name}_{args.split}_activations_{args.model_name}"
        f"_{args.layer}_{args.attachment_point}_part{save_count + 1}.pt"
    )
    save_path = os.path.join(output_subdir, filename)
    torch.save(torch.tensor(chunk.cpu().numpy()), save_path)
    log.info(f"Saved chunk {save_count + 1}: {chunk.shape} -> {save_path}")


def main():
    args = parse_args()

    try:
        import webdataset as wds
    except ImportError:
        raise ImportError("webdataset not installed. Run: pip install webdataset")

    # Collect and slice shard paths
    shard_paths = sorted(glob.glob(os.path.join(args.shard_dir, "*.tar")))
    if not shard_paths:
        raise ValueError(f"No .tar files found in {args.shard_dir}")

    shard_paths = shard_paths[args.shard_start:]
    if args.max_shards is not None:
        shard_paths = shard_paths[:args.max_shards]

    log.info(
        f"Using {len(shard_paths)} shards "
        f"(shard_start={args.shard_start}, max_shards={args.max_shards})"
    )

    # Output directory follows existing naming convention
    output_subdir = os.path.join(
        args.output_dir,
        f"{args.dataset_name}_{args.split}_activations_{args.model_name}"
        f"_{args.layer}_{args.attachment_point}",
    )
    Path(output_subdir).mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {output_subdir}")

    # Load CLIP and attach activation hook
    log.info(f"Loading model openai/{args.model_name} on {args.device}")
    clip = Clip(args.model_name, args.device)
    clip.attach(args.attachment_point, args.layer, sae=None)
    register_key = f"{args.attachment_point}_{args.layer}"

    # WebDataset pipeline
    # wds.warn_and_continue skips corrupt/truncated samples gracefully
    dataset = (
        wds.WebDataset(shard_paths, shardshuffle=False, nodesplitter=wds.split_by_node)
        .decode("pil", handler=wds.warn_and_continue)
        .to_tuple("jpg;png;jpeg;webp", handler=wds.warn_and_continue)
    )

    loader = wds.WebLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=lambda samples: [s[0] for s in samples if s[0] is not None],
    )

    activations_buffer = []
    images_processed = 0
    save_count = 0
    start_time = time.time()

    for pil_images in loader:
        if not pil_images:
            continue
        if args.max_images is not None and images_processed >= args.max_images:
            break

        pil_images = [img.convert("RGB") for img in pil_images]
        inputs = clip.processor(images=pil_images, return_tensors="pt", padding=True)

        with torch.no_grad():
            clip.encode(inputs)
            batch_activations = clip.register[register_key]

        act_tensor = torch.cat(batch_activations, dim=0)
        pooled = pool_activations(act_tensor, args.token_mode, args.n_random_tokens)
        activations_buffer.append(pooled)

        images_processed += len(pil_images)
        elapsed = time.time() - start_time
        img_per_sec = images_processed / elapsed if elapsed > 0 else 0
        log.info(f"Processed {images_processed} images ({img_per_sec:.1f} img/s)")

        if images_processed >= args.save_every * (save_count + 1):
            save_chunk(activations_buffer, save_count, args, output_subdir)
            activations_buffer = []
            save_count += 1

    if activations_buffer:
        save_chunk(activations_buffer, save_count, args, output_subdir)
        save_count += 1

    elapsed = time.time() - start_time
    log.info(f"Done. Saved {save_count} chunks to {output_subdir} in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
