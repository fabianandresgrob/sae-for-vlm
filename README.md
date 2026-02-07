## 
<h1 align="center">Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models</h1>

<div align="center">
<a href="https://www.eml-munich.de/people/mateusz-pach">Mateusz Pach</a>,
<a href="https://sgk98.github.io/">Shyamgopal Karthik</a>,
<a href="https://www.eml-munich.de/people/quentin-bouniot">Quentin Bouniot</a>,
<a href="https://www.eml-munich.de/people/serge-belongie">Serge Belongie</a>,
<a href="https://www.eml-munich.de/people/zeynep-akata">Zeynep Akata</a>
<br>
<br>

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2504.02821)
</div>

<h3 align="center">Abstract</h3>

<p align="justify">
Sparse Autoencoders (SAEs) have recently gained attention as a means to improve the interpretability and steerability of Large Language Models (LLMs), both of which are essential for AI safety. In this work, we extend the application of SAEs to Vision-Language Models (VLMs), such as CLIP, and introduce a comprehensive framework for evaluating monosemanticity at the neuron-level in visual representations. To ensure that our evaluation aligns with human perception, we propose a benchmark derived from a large-scale user study. Our experimental results reveal that SAEs trained on VLMs significantly enhance the monosemanticity of individual neurons, with sparsity and wide latents being the most influential factors. Further, we demonstrate that applying SAE interventions on CLIP's vision encoder directly steers multimodal LLM outputs (e.g., LLaVA), without any modifications to the underlying language model. These findings emphasize the practicality and efficacy of SAEs as an unsupervised tool for enhancing both interpretability and control of VLMs.</p>
<br>
<div align="center">
    <img src="assets/teaser.svg" alt="Teaser" width="800">
</div>

---
### Setup
Install required PIP packages.
```bash
pip install -r requirements.txt
```
Download following datasets:
* ImageNet (https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html)
* INaturalist 2021 (https://github.com/visipedia/inat_comp/tree/master/2021)

Export paths to dataset directories. The directories should contain `train/` and `val/` subdirectories.
```bash
export IMAGENET_PATH="<path_to_imagenet>"
export INAT_PATH="<path_to_inaturalist>"
```
Code was run using Python version 3.11.10.
### Running Experiments
The commands required to reproduce the results are organized into scripts located in the `scripts/` directory:
* `monosemanticity_score.sh` computes the Monosemanticity Score (MS) for specified SAEs, layers, models, and image encoders.
* `matryoshka_hierarchy.sh` analyzes the hierarchical structure that emerges in Matryoshka SAEs.
* `mllm_steering.sh` enables experimentation with steering LLaVA using an SAE built on top of the vision encoder.

### Metric Benchmark
To facilitate advancements in monosemanticity metrics, we release benchmark data derived from our user study in the `metric_benchmark.tar.gz` archive, comprising:

- **`pairs.csv`** – Contains 1000 pairs of neurons (r_x, r_y), along with user preferences R_user and MS values computed using two different image encoders: DINOv2 ViT-B and CLIP ViT-B. Each row includes the following columns:  
  `k_x`, `k_y`, `R_user`, `MS_x_dino`, `MS_y_dino`, `MS_x_clip`, `MS_y_clip`.

- **`top16_images.csv`** – Lists the 16 most activating images from the ImageNet training set for each neuron used in the study. Columns:  
  `k`, `x_1`, …, `x_16`.

- **`activations.csv`** – Provides activation values of all 50,000 ImageNet validation images for each neuron. Columns:  
  `k`, `a_1`, …, `a_50000`.

See Appendix D: Benchmark attached to the article to learn more.



We use the implementation of sparse autoencoders available at https://github.com/saprmarks/dictionary_learning.

### Alternative: HuggingFace ImageNet Streaming
If you don't have ImageNet on disk, you can stream it from HuggingFace. First, accept the [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) license and log in:
```bash
huggingface-cli login
```
Then run the full pipeline (activation extraction + SAE training):
```bash
./train_sae_from_hf.sh
```
Or extract activations individually with `save_activations_hf.py`:
```bash
# Training set: 2 random patch tokens per image
python save_activations_hf.py --split train --token_mode random_k --n_random_tokens 2

# Validation set: CLS token only
python save_activations_hf.py --split validation --token_mode cls
```

### Citation
```bibtex
@article{pach2025sparse,
  title={Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models}, 
  author={Mateusz Pach and Shyamgopal Karthik and Quentin Bouniot and Serge Belongie and Zeynep Akata},
  journal={arXiv preprint arXiv:2504.02821},
  year={2025}
}
```
