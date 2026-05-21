"""
Dataset & augmentation utilities for Tamil Nadu temple mural restoration.
Handles loading, synthetic mask generation, and preprocessing.
"""

import os
import random
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ─────────────────────────────────────────
#  Synthetic damage mask generators
# ─────────────────────────────────────────

def random_brush_mask(height: int, width: int,
                       min_strokes: int = 3,
                       max_strokes: int = 10) -> np.ndarray:
    """Simulate irregular brush-stroke shaped damage (common in murals)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for _ in range(random.randint(min_strokes, max_strokes)):
        x0, y0 = random.randint(0, width), random.randint(0, height)
        for _ in range(random.randint(10, 40)):
            dx = random.randint(-30, 30)
            dy = random.randint(-30, 30)
            x1, y1 = np.clip(x0+dx, 0, width-1), np.clip(y0+dy, 0, height-1)
            thickness = random.randint(4, 20)
            cv2.line(mask, (x0, y0), (x1, y1), 1, thickness)
            x0, y0 = x1, y1
    return mask


def random_rectangular_mask(height: int, width: int,
                              max_holes: int = 3) -> np.ndarray:
    """Rectangular blocks simulating water-damage or vandalism patches."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for _ in range(random.randint(1, max_holes)):
        x = random.randint(0, width  - 1)
        y = random.randint(0, height - 1)
        w = random.randint(20, width  // 3)
        h = random.randint(20, height // 3)
        mask[y:y+h, x:x+w] = 1
    return mask


def generate_damage_mask(height: int, width: int,
                          mode: str = "mixed") -> np.ndarray:
    if mode == "brush":
        return random_brush_mask(height, width)
    elif mode == "rect":
        return random_rectangular_mask(height, width)
    else:  # mixed
        m = random_brush_mask(height, width)
        if random.random() > 0.5:
            m = np.clip(m + random_rectangular_mask(height, width), 0, 1)
        return m.astype(np.uint8)


# ─────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────

class MuralDataset(Dataset):
    """
    Loads mural images from a directory, synthetically damages them,
    and returns (masked_image, mask, ground_truth) triplets.

    Directory layout expected:
        root/
          train/  *.jpg / *.png
          val/    *.jpg / *.png
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def __init__(self,
                 root: str,
                 split: str = "train",
                 image_size: int = 256,
                 mask_mode: str = "mixed",
                 augment: bool = True):
        self.root       = Path(root) / split
        self.image_size = image_size
        self.mask_mode  = mask_mode
        self.augment    = augment and split == "train"

        self.paths: List[Path] = [
            p for p in sorted(self.root.rglob("*"))
            if p.suffix.lower() in self.EXTENSIONS
        ]
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found under {self.root}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"Cannot read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size),
                         interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            img = np.fliplr(img)
        if random.random() > 0.5:
            img = np.flipud(img)
        k = random.choice([0, 1, 2, 3])
        img = np.rot90(img, k)
        return np.ascontiguousarray(img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gt = self._load(self.paths[idx])
        if self.augment:
            gt = self._augment(gt)

        H, W = self.image_size, self.image_size
        mask  = generate_damage_mask(H, W, mode=self.mask_mode)   # H×W uint8
        mask3 = np.stack([mask] * 3, axis=-1).astype(np.float32)  # H×W×3

        masked = gt * (1 - mask3)                                  # H×W×3

        # Convert to torch tensors: C×H×W
        gt_t     = torch.from_numpy(gt.transpose(2, 0, 1))
        masked_t = torch.from_numpy(masked.transpose(2, 0, 1))
        mask_t   = torch.from_numpy(mask[np.newaxis].astype(np.float32))  # 1×H×W

        return masked_t, mask_t, gt_t
