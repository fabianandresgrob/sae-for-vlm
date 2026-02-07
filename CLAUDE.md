# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for "Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models" (arXiv:2504.02821). Trains Sparse Autoencoders (SAEs) on Vision-Language Model activations (CLIP, DINOv2, SigLIP) to improve neuron-level interpretability, then evaluates monosemanticity and demonstrates steering of multimodal LLMs (LLaVA) via SAE interventions.

## Setup

```bash
git submodule update --init --recursive
uv pip install -r requirements.txt
export IMAGENET_PATH="<path_to_imagenet>"   # must contain train/ and val/
export INAT_PATH="<path_to_inaturalist>"    # must contain train/ and val/
```

Python 3.11.10, managed via uv. PyTorch 2.1.2, transformers 4.46.3. Experiment tracking via W&B.

## Running Experiments

All experiments are orchestrated via shell scripts in `scripts/`:

```bash
./scripts/monosemanticity_score.sh   # Full pipeline: activations -> SAE training -> metric
./scripts/matryoshka_hierarchy.sh    # Hierarchical structure analysis on iNaturalist
./scripts/mllm_steering.sh          # LLaVA steering experiments
```

### Individual pipeline steps

```bash
# 1. Extract activations from a VLM
python save_activations.py --model_name clip-vit-large-patch14-336 \
  --attachment_point post_mlp_residual --layer 22 --dataset_name imagenet \
  --split train --data_path $IMAGENET_PATH --cls_only --save_every 100

# 2. Train an SAE
python sae_train.py --sae_model matroyshka_batch_top_k \
  --activations_dir <dir> --val_activations_dir <dir> \
  --expansion_factor 64 --steps 110000 --batch_size 4096 --k 20

# 3. Generate vision encoder embeddings (required before metric.py)
python encode_images.py --embeddings_path embeddings_dir/imagenet_val_embeddings_clip-vit-base-patch32.pt \
  --model_name clip-vit-base-patch32 --dataset_name imagenet --split val \
  --data_path $IMAGENET_PATH --batch_size 128

# 4. Compute monosemanticity scores
python metric.py --embeddings_path <path> --activations_dir <dir>

# 5. Find highest-activating images & visualize
python find_hai_indices.py --activations_dir <dir> --k 16
python visualize_neurons.py --output_dir <dir> --hai_indices_path <path>

# 6. Steering evaluation
python steering_score.py --sae_path <path> --steer --hai_indices_path <path>
```

## Architecture

### Pipeline flow

```
Images -> VLM encoder -> Extract activations -> Train SAE -> SAE-encoded activations
                                                                |
                                              Monosemanticity scoring (metric.py)
                                              Neuron visualization (visualize_neurons.py)
                                              LLaVA steering (steering_score.py)
```

### Model wrappers (`models/`)

Each VLM (CLIP, DINOv2, SigLIP) has a wrapper class with a consistent interface:
- `encode(inputs)` - forward pass returning embeddings
- `attach(attachment_point, layer, sae)` - hooks an SAE into the model at a specific point

Attachment points: `post_mlp_residual` (intermediate layers), `post_projection` (final layer, use layer=-1). The wrapper replaces transformer layers with custom `nn.Module` subclasses that intercept the forward pass to capture/transform activations via a `register` dict.

### SAE training (`dictionary_learning/`)

Git submodule from [saprmarks/dictionary_learning](https://github.com/saprmarks/dictionary_learning). Available SAE architectures (configured via `--sae_model`):
- `matroyshka_batch_top_k` - primary architecture in experiments, supports hierarchical `--group_fractions`
- `batch_top_k`, `top_k`, `jumprelu`, `standard`

Training uses `ConstrainedAdam` optimizer with decoder norm constraints. SAE checkpoints saved as `ae_<step>.pt`.

### Activation data (`datasets/activations.py`)

- `ActivationsDataset` - loads all chunked `.pt` files into memory, supports `take_every` subsampling
- `ChunkedActivationsDataset` - streams from disk with single-file caching

Activation files are named `*_part<N>.pt` and sorted by part number.

### Key conventions

- Model names follow HuggingFace identifiers: `clip-vit-large-patch14-336`, `dinov2-base`, `siglip-so400m-patch14-384`
- `utils.py` provides `get_model()` (dispatches by model name prefix) and `get_dataset()` (dispatches by dataset name)
- `IdentitySAE` is a passthrough used when no SAE is attached
- Activations directory structure: `activations_dir/<sae_config>/<dataset>_<split>_activations_<model>_<layer>_<point>/`
- Checkpoints directory structure: `checkpoints_dir/<sae_config>/<dataset_name>_<sae_model>_<k>_x<expansion>/trainer_0/checkpoints/`

### Gotchas

- `metric.py` calls `.cuda()` internally — requires GPU even if `--device cpu` is passed
- `sae_train.py` has W&B entity hardcoded to `"mateuszpach"` — change `wandb_entity` in `sae_train.py:108` for your own account
- LLaVA steering is hardcoded to layer 22 of the CLIP vision encoder inside `models/llava.py`
- SAE loading requires matching class: `AutoEncoder.from_pretrained` (standard), `BatchTopKSAE.from_pretrained` (batch_top_k), `MatroyshkaBatchTopKSAE.from_pretrained` (matroyshka_batch_top_k)
- Token pooling in `save_activations.py`: `--cls_only` (CLS token only), `--mean_pool` (average all tokens), `--random_k N` (sample N random tokens per image), default (all tokens flattened)
- `ActivationsDataset` skips files starting with `all` (e.g. `all_neurons_scores.pth`) when loading from a directory
