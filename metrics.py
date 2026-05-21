"""
Quantitative evaluation metrics used in the paper:
  - PSNR  (Peak Signal-to-Noise Ratio)
  - SSIM  (Structural Similarity Index)
  - FID   (Fréchet Inception Distance) — wrapper around torch-fidelity
  - LPIPS (Learned Perceptual Image Patch Similarity)
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


# ─────────────────────────────────────────
#  PSNR
# ─────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor,
         data_range: float = 1.0) -> float:
    """
    PSNR in dB.  pred / target: B×C×H×W float in [0, data_range].
    """
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


# ─────────────────────────────────────────
#  SSIM
# ─────────────────────────────────────────

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)   # 1×1×size×size


def ssim(pred: torch.Tensor, target: torch.Tensor,
         data_range: float = 1.0,
         window_size: int = 11, sigma: float = 1.5,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Mean SSIM over batch (channels treated independently then averaged).
    pred / target: B×C×H×W
    """
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    B, C, H, W = pred.shape
    kernel = _gaussian_kernel(window_size, sigma).to(pred.device)
    kernel = kernel.expand(C, 1, window_size, window_size)

    pad = window_size // 2

    def conv(x):
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu1    = conv(pred)
    mu2    = conv(target)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12   = mu1 * mu2

    sig1_sq = conv(pred ** 2)   - mu1_sq
    sig2_sq = conv(target ** 2) - mu2_sq
    sig12   = conv(pred * target) - mu12

    num = (2 * mu12 + C1) * (2 * sig12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2)
    return (num / den).mean().item()


# ─────────────────────────────────────────
#  Hole vs Non-hole metrics
# ─────────────────────────────────────────

def masked_psnr(pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> Tuple[float, float]:
    """
    Returns (PSNR inside hole, PSNR outside hole).
    mask: 1×1×H×W or B×1×H×W — 1 = hole region.
    """
    def _psnr(a, b):
        mse = ((a - b) ** 2).mean().item()
        return 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf")

    return (_psnr(pred * mask,       target * mask),
            _psnr(pred * (1 - mask), target * (1 - mask)))


# ─────────────────────────────────────────
#  Evaluation runner
# ─────────────────────────────────────────

class MetricTracker:
    """Accumulates per-batch metrics and reports averages."""

    def __init__(self):
        self._data: dict = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self._data.setdefault(k, []).append(float(v))

    def mean(self, key: str) -> float:
        vals = self._data.get(key, [])
        return float(np.mean(vals)) if vals else 0.0

    def report(self) -> dict:
        return {k: float(np.mean(v)) for k, v in self._data.items()}

    def reset(self):
        self._data.clear()
