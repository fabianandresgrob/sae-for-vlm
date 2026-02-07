"""
Extract CLIP vision encoder activations by streaming ImageNet-1k from HuggingFace.

This is an alternative to save_activations.py for users who do not have ImageNet on disk.
The saved activation format is fully compatible with sae_train.py and ActivationsDataset.

Usage:
    # Train split: 2 random patch tokens per image
    python save_activations_hf.py --split train --token_mode random_k --n_random_tokens 2

    # Validation split: CLS token only
    python save_activations_hf.py --split validation --token_mode cls
"""

import argparse
import logging
import os
import time
from pathlib import Path
import sys

import torch
from models.clip import Clip


def _get_hf_load_dataset():
    """Return HuggingFace `datasets.load_dataset` even if a local `datasets/` package exists.

    This repo contains a top-level `datasets/` directory, which shadows the external
    HuggingFace `datasets` package when running scripts from the repo root.
    """

    repo_root = str(Path(__file__).resolve().parent)

    removed = []
    for entry in ("", repo_root):
        while entry in sys.path:
            sys.path.remove(entry)
            removed.append(entry)
    try:
        import datasets as hf_datasets  # HuggingFace package

        return hf_datasets.load_dataset
    finally:
        for entry in reversed(removed):
            sys.path.insert(0, entry)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Save CLIP activations from HuggingFace ImageNet")
    parser.add_argument("--output_dir", type=str, default="./activations_dir/raw/hf")
    parser.add_argument("--model_name", type=str, default="clip-vit-large-patch14-336")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--attachment_point", type=str, default="post_mlp_residual")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"])
    parser.add_argument("--token_mode", type=str, default="random_k",
                        choices=["cls", "mean_pool", "random_k", "all"],
                        help="How to pool tokens. 'random_k' samples N random patch tokens (skipping CLS).")
    parser.add_argument("--n_random_tokens", type=int, default=2,
                        help="Number of random patch tokens to sample per image (only used with --token_mode random_k)")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--save_every", type=int, default=10000,
                        help="Save a chunk every N images processed")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Process at most N images (for testing). Default: all images.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def pool_activations(activations_tensor, token_mode, n_random_tokens):
    """Pool token-level activations into per-image vectors.

    Args:
        activations_tensor: [batch, seq_len, hidden_dim] tensor
        token_mode: one of 'cls', 'mean_pool', 'random_k', 'all'
        n_random_tokens: number of patch tokens to sample (for random_k)

    Returns:
        [N, hidden_dim] tensor where N depends on the pooling mode
    """
    if token_mode == "cls":
        # CLS token is at index 0
        return activations_tensor[:, 0, :]
    elif token_mode == "mean_pool":
        return torch.mean(activations_tensor, dim=1)
    elif token_mode == "random_k":
        batch_size, seq_len, hidden_dim = activations_tensor.shape
        # Sample from patch tokens only (indices 1..seq_len-1, skipping CLS at 0)
        patch_indices = torch.randint(1, seq_len, (batch_size, n_random_tokens))
        sampled = torch.stack([activations_tensor[i, patch_indices[i], :] for i in range(batch_size)])
        return sampled.reshape(-1, hidden_dim)
    elif token_mode == "all":
        batch_size, seq_len, hidden_dim = activations_tensor.shape
        return activations_tensor.reshape(batch_size * seq_len, hidden_dim)
    else:
        raise ValueError(f"Unknown token_mode: {token_mode}")


def _iter_batches(dataset_iter, batch_size, max_images):
    """Collect examples from a streaming iterator into batches of PIL images."""
    batch = []
    count = 0
    for example in dataset_iter:
        if max_images is not None and count >= max_images:
            break
        batch.append(example["image"])
        count += 1
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    args = parse_args()

    load_dataset = _get_hf_load_dataset()

    output_subdir = os.path.join(
        args.output_dir,
        f"imagenet_{args.split}_activations_{args.model_name}_{args.layer}_{args.attachment_point}",
    )
    Path(output_subdir).mkdir(parents=True, exist_ok=True)

    # Load CLIP model and attach activation hook
    log.info(f"Loading model openai/{args.model_name} on {args.device}")
    clip = Clip(args.model_name, args.device)
    clip.attach(args.attachment_point, args.layer, sae=None)
    register_key = f"{args.attachment_point}_{args.layer}"

    # Stream HuggingFace ImageNet (no download)
    log.info(f"Streaming ImageNet-1k split={args.split} from HuggingFace...")
    dataset = load_dataset("ILSVRC/imagenet-1k", split=args.split,
                           streaming=True, trust_remote_code=True)

    # Known split sizes for ETA estimation
    split_sizes = {"train": 1281167, "validation": 50000}
    total_images = split_sizes.get(args.split, 0)
    if args.max_images is not None:
        total_images = min(total_images, args.max_images) if total_images else args.max_images
    log.info(f"Expected images: {total_images or 'unknown'}")

    activations_buffer = []
    images_processed = 0
    save_count = 0
    start_time = time.time()

    for pil_images in _iter_batches(iter(dataset), args.batch_size, args.max_images):
        # Convert to RGB (some ImageNet images are grayscale)
        pil_images = [img.convert("RGB") for img in pil_images]

        # Preprocess with CLIP processor
        inputs = clip.processor(images=pil_images, return_tensors="pt", padding=True)

        with torch.no_grad():
            clip.encode(inputs)
            batch_activations = clip.register[register_key]

        # batch_activations is a list with one tensor of shape [batch, seq_len, hidden_dim]
        act_tensor = torch.cat(batch_activations, dim=0)
        pooled = pool_activations(act_tensor, args.token_mode, args.n_random_tokens)
        activations_buffer.append(pooled)

        images_processed += len(pil_images)

        # Logging with ETA
        elapsed = time.time() - start_time
        images_per_sec = images_processed / elapsed if elapsed > 0 else 0
        if total_images and images_per_sec > 0:
            remaining = (total_images - images_processed) / images_per_sec
            eta_str = f", ETA {remaining/60:.1f} min"
        else:
            eta_str = ""
        log.info(
            f"Processed {images_processed}/{total_images or '?'} images "
            f"({images_per_sec:.1f} img/s{eta_str})"
        )

        # Save chunk
        if images_processed >= args.save_every * (save_count + 1):
            _save_chunk(activations_buffer, save_count, args, output_subdir)
            activations_buffer = []
            save_count += 1

    # Save remaining
    if activations_buffer:
        _save_chunk(activations_buffer, save_count, args, output_subdir)
        save_count += 1

    elapsed = time.time() - start_time
    log.info(f"Done. Saved {save_count} chunks to {output_subdir} in {elapsed/60:.1f} min")


def _save_chunk(buffer, save_count, args, output_subdir):
    """Save accumulated activations as a single .pt chunk file."""
    chunk = torch.cat(buffer, dim=0)
    filename = (
        f"imagenet_{args.split}_activations_{args.model_name}"
        f"_{args.layer}_{args.attachment_point}_part{save_count + 1}.pt"
    )
    save_path = os.path.join(output_subdir, filename)
    # Match original save_activations.py: torch.save(torch.tensor(tensor.cpu().numpy()))
    torch.save(torch.tensor(chunk.cpu().numpy()), save_path)
    log.info(f"Saved chunk {save_count + 1}: {chunk.shape} -> {save_path}")


if __name__ == "__main__":
    main()
