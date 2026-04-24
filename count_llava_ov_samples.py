"""
Count samples per subdataset in LLaVA-OneVision-1.5-Instruct-Data and produce
a sampling plan for SAE activation extraction.

Reads only parquet metadata (no image data loaded) — safe for login nodes.

Usage:
    python count_llava_ov_samples.py \
        --data_dir /e/scratch/taco-vlm/kim16/LLaVA-OneVision-1.5-Instruct-Data \
        --budget 5000000
"""

import argparse
import math
import os
import sys
from pathlib import Path


def count_parquet_rows(parquet_path: Path) -> int:
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(parquet_path).metadata.num_rows
    except Exception:
        pass
    try:
        import pandas as pd
        return len(pd.read_parquet(parquet_path, columns=[]))
    except Exception as e:
        print(f"  WARNING: could not read {parquet_path}: {e}", file=sys.stderr)
        return 0


def count_subdataset(subdir: Path) -> int:
    files = sorted(subdir.glob("*.parquet"))
    if not files:
        return 0
    # First file gives total from the shard metadata (num_rows in metadata == this file's rows)
    # Sum across all shards.
    total = 0
    for f in files:
        total += count_parquet_rows(f)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True,
                        help="Root of LLaVA-OneVision-1.5-Instruct-Data")
    parser.add_argument("--budget", type=int, default=5_000_000,
                        help="Total images to sample across all subdatasets")
    parser.add_argument("--cap", type=int, default=None,
                        help="Max images per subdataset (default: no cap)")
    parser.add_argument("--floor", type=int, default=1000,
                        help="Min images per subdataset if it has >= floor samples")
    args = parser.parse_args()

    base = Path(args.data_dir)
    subdirs = sorted([d for d in base.iterdir() if d.is_dir()])

    print(f"Scanning {len(subdirs)} subdirectories in {base} ...\n")

    counts = {}
    for i, subdir in enumerate(subdirs):
        n = count_subdataset(subdir)
        counts[subdir.name] = n
        print(f"  [{i+1:3d}/{len(subdirs)}] {subdir.name:<50s} {n:>10,}")

    total_samples = sum(counts.values())
    print(f"\nTotal samples across all subdatasets: {total_samples:,}")
    print(f"Sampling budget: {args.budget:,}")

    # --- Proportional sampling ---
    prop = {k: int(args.budget * v / total_samples) for k, v in counts.items()}

    # --- Sqrt-weighted sampling (balances large/small datasets) ---
    sqrt_weights = {k: math.sqrt(v) for k, v in counts.items()}
    sqrt_total = sum(sqrt_weights.values())
    sqrt_plan = {k: int(args.budget * w / sqrt_total) for k, w in sqrt_weights.items()}

    # Apply floor and cap to sqrt plan
    final_plan = {}
    for k, v in counts.items():
        n = sqrt_plan[k]
        if v > 0 and n < args.floor:
            n = min(args.floor, v)
        if args.cap is not None:
            n = min(n, args.cap)
        n = min(n, v)  # never sample more than available
        final_plan[k] = n

    # Rescale final plan to stay within budget
    plan_total = sum(final_plan.values())
    if plan_total > args.budget:
        scale = args.budget / plan_total
        final_plan = {k: int(v * scale) for k, v in final_plan.items()}

    plan_total = sum(final_plan.values())

    # --- Print results table ---
    print(f"\n{'Subdataset':<50s} {'Total':>10s} {'Prop':>10s} {'SqrtPlan':>10s}  {'%budget':>7s}")
    print("-" * 95)
    for k in sorted(counts, key=lambda x: counts[x], reverse=True):
        pct = 100 * final_plan[k] / plan_total if plan_total > 0 else 0
        print(f"{k:<50s} {counts[k]:>10,} {prop[k]:>10,} {final_plan[k]:>10,}  {pct:>6.1f}%")

    print("-" * 95)
    print(f"{'TOTAL':<50s} {total_samples:>10,} {sum(prop.values()):>10,} {plan_total:>10,}  100.0%")

    # --- Write sampling plan to CSV ---
    out_csv = base / "sampling_plan.csv"
    with open(out_csv, "w") as f:
        f.write("subdataset,total_samples,sampled\n")
        for k in sorted(counts):
            f.write(f"{k},{counts[k]},{final_plan[k]}\n")
    print(f"\nSampling plan written to: {out_csv}")


if __name__ == "__main__":
    main()
