# CoordAttn-GAN 🏛️

### *Coordinated Attention Conditional GAN with Diffusion Discriminator for Historic Mural Restoration*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch" />
  <img src="https://img.shields.io/badge/OpenCV-4.7%2B-5C3EE8?style=flat-square&logo=opencv" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

> Official implementation of **"Physical restoration in conjunction with digital restoration, employing a coordinated conditional GAN with diffusion discriminator on Mural images"**
> — Dr. D. Sudaroli · Dr. V. Muthu Manikandan · Dr. P. Shantala

---

## Overview

Temple murals across India face irreversible degradation from UV radiation, moisture, vandalism, and material decay. Existing digital inpainting methods produce visually plausible results but fail to bridge the gap to **physical restoration** — the corrected output cannot be directly applied by artisans due to over-painting and loss of fine contextual details.

This work addresses that gap with two contributions:

1. **CoordAttn-GAN** — A conditional GAN with a *coordinated attention generator* (multi-scale dilated residual blocks, Eq. 1–4) and a *diffusion-based discriminator* (U-Net structure with joint encoder scalar + decoder pixel-map loss, Eq. 5–6) for high-fidelity digital inpainting.

2. **Corrected Mask Construction** — An analysis pipeline that computes hue/saturation deviation between the digital restoration and the original, preserves fine contextual lines, and produces a corrected guidance mask for physical (manual) inpainting by artisans.

<br>

```
Damaged Mural → Digital Restoration (CoordAttn-GAN) → Corrected Mask → Artisan Guidance
```

---

## Architecture

### Generator — Coordinated Attention (Eq. 1–4)

```
Input (masked image + binary mask, 4ch)
  └─ Encoder (4 strided conv blocks)
       └─ Bottleneck: DilatedResBlock × 4  (dilation = 1, 2, 4, 8)
            └─ Each block: Conv → BN → CoordinatedAttention → Gated shortcut
       └─ Decoder (4 transposed conv blocks, skip connections from encoder)
  └─ Output: composite inpainted image (3ch, tanh)
```

**CoordinatedAttention** decomposes global average pooling into 1-D horizontal and vertical encodings, fuses them without sigmoid (preserving structural coherence), and applies gated residual connections:

```
dec_h(h)  =  (1/W) Σ_i  x_c(h, i)          # Eq. (2)
dec_w(w)  =  (1/H) Σ_j  x_c(j, w)          # Eq. (3)
t         =  ϑ(f([dec_h, dec_w]))            # Eq. (4)
```

### Discriminator — Diffusion-Based (Eq. 5–6)

```
Input image (3ch)
  └─ Encoder D_E  →  scalar real/fake score    (global, Eq. 6)
  └─ Bottleneck
  └─ Decoder D_d  →  pixel-map real/fake scores (local, Eq. 7)

L_Ddif = L_DE_dif + L_Dd_dif                   # Eq. (5)
```

Skip and bottleneck connections throughout. Encoder and decoder learn **independently** to distinguish local and global spatial context.

---

## Repository Structure

```
.
├── model.py             # CoordAttentionGenerator + DiffusionDiscriminator + Loss functions
├── mask_correction.py   # Corrected mask construction for physical restoration guidance
├── dataset.py           # MuralDataset with synthetic damage mask generation
├── metrics.py           # PSNR, SSIM, masked metrics, MetricTracker
├── train.py             # Full GAN training loop
├── infer.py             # Inference: digital restoration + corrected mask output
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/your-org/coordattn-gan.git
cd coordattn-gan
pip install -r requirements.txt
```

**Requirements:** `torch>=2.0`, `torchvision>=0.15`, `opencv-python>=4.7`, `numpy>=1.23`, `Pillow>=9.4`

---

## Data Preparation

Organise your dataset as:

```
data/
  train/   *.jpg  *.png
  val/     *.jpg  *.png
```

The dataset module generates synthetic damage masks automatically during training (brush-stroke and rectangular patterns, or mixed). To use real damage masks at inference, pass them via `--damage_mask`.

> The paper evaluates on temple murals collected across Tamil Nadu. We will release a subset upon publication. Public benchmarks such as the [Dunhuang Mural Dataset](https://github.com/dunhuang) can be used for pre-training.

---

## Training

```bash
python train.py \
  --data_root   /path/to/data \
  --output_dir  ./checkpoints \
  --image_size  256 \
  --batch_size  8 \
  --epochs      100 \
  --lr_g        2e-4 \
  --lr_d        2e-4 \
  --base_ch     64
```

**Resume from checkpoint:**

```bash
python train.py --data_root /path/to/data --resume checkpoints/ckpt_epoch0050.pth
```

Training logs report G loss, D loss, PSNR, and SSIM per epoch. Validation sample grids are saved to `output_dir/` every `--val_every` epochs.

---

## Inference

```bash
python infer.py \
  --checkpoint  checkpoints/generator_final.pth \
  --input       damaged_mural.jpg \
  --damage_mask mask.png \
  --output_dir  results/
```

**Outputs produced:**

| File | Description |
|---|---|
| `*_masked.png` | Input with damage mask applied |
| `*_restored.png` | Digitally inpainted image |
| `*_corrected_mask.png` | Guidance mask for physical artisan inpainting |
| `*_overlay.png` | Semi-transparent overlay for visual review |

**Threshold tuning** for the corrected mask (adjust based on mural style):

```bash
python infer.py ... --hue_thresh 12.0 --sat_thresh 35.0
```

---

## Corrected Mask Construction

The physical-digital alignment pipeline:

```
1. Compute per-pixel ΔHue, ΔSat between original and restored
2. Flag pixels exceeding thresholds as over-painted candidates
3. Extract fine structural edges from the original (Canny)
4. Subtract edge pixels → artisans handle fine lines manually
5. Intersect with damage region → corrected mask
6. Morphological cleanup → remove isolated noise
```

This corrected mask guides artisans to paint **only** where the digital restoration has deviated unacceptably, preserving the aesthetic intent of the original mural.

---

## Metrics

The evaluation module (`metrics.py`) implements:

| Metric | Notes |
|---|---|
| PSNR | Peak Signal-to-Noise Ratio (dB); also reported separately for hole vs. non-hole regions |
| SSIM | Structural Similarity (Gaussian window, k₁=0.01, k₂=0.03) |
| FID | Fréchet Inception Distance (via `torch-fidelity`) |
| LPIPS | Learned Perceptual Image Patch Similarity |

```python
from metrics import psnr, ssim, masked_psnr, MetricTracker

tracker = MetricTracker()
tracker.update(psnr=psnr(pred, gt), ssim=ssim(pred, gt))
print(tracker.report())
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{sudaroli2025coordattngan,
  title   = {Physical restoration in conjunction with digital restoration,
             employing a coordinated conditional GAN with diffusion
             discriminator on Mural images},
  author  = {Sudaroli, D. and Muthu Manikandan, V. and Shantala, P.},
  journal = {(under review)},
  year    = {2025}
}
```

---

## Acknowledgements

- [Coordinated Attention (Hou et al., 2021)](https://arxiv.org/abs/2103.02907) — foundational attention mechanism
- [Dunhuang Mural Inpainting](https://github.com/dunhuang) — benchmark reference
- Tamil Nadu Temple Conservation Trust — mural image collection

---

## License

This project is released under the [MIT License](LICENSE).
