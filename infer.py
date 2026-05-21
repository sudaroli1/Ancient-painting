"""
Inference script: applies digital restoration + corrected mask generation
to a new mural image, producing both:
  1. Digitally restored image
  2. Corrected guidance mask for physical inpainting artisans

Usage:
    python infer.py --checkpoint checkpoints/generator_final.pth \
                    --input damaged_mural.jpg \
                    --damage_mask mask.png \
                    --output_dir results/
"""

import argparse
import os
import cv2
import numpy as np
import torch
from pathlib import Path

from model           import CoordAttentionGenerator
from mask_correction import build_corrected_mask, visualize_correction


def load_image(path: str, size: int = 256) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def load_mask(path: str, size: int = 256) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def save(arr: np.ndarray, path: str):
    if arr.dtype == np.float32:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--input",        required=True, help="Damaged mural image")
    p.add_argument("--damage_mask",  required=True, help="Binary mask (white=damaged)")
    p.add_argument("--output_dir",   default="./results")
    p.add_argument("--image_size",   type=int, default=256)
    p.add_argument("--base_ch",      type=int, default=64)
    p.add_argument("--hue_thresh",   type=float, default=15.0)
    p.add_argument("--sat_thresh",   type=float, default=40.0)
    args = p.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ───────────────────────────
    G = CoordAttentionGenerator(base_ch=args.base_ch).to(device)
    G.load_state_dict(torch.load(args.checkpoint, map_location=device))
    G.eval()

    # ── Load inputs ──────────────────────────
    original     = load_image(args.input,       args.image_size)
    damage_mask  = load_mask(args.damage_mask,  args.image_size)

    masked_np = original * (1 - damage_mask[:, :, np.newaxis])

    # ── Digital restoration ──────────────────
    def to_tensor(x): return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
    masked_t = to_tensor(masked_np)
    mask_t   = torch.from_numpy(damage_mask[np.newaxis, np.newaxis].astype(np.float32)).to(device)

    with torch.no_grad():
        restored_t = G(masked_t, mask_t)

    restored = restored_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    restored = np.clip(restored, 0, 1)

    # ── Corrected mask for physical restoration
    corrected_mask = build_corrected_mask(
        original, restored, damage_mask,
        hue_thresh=args.hue_thresh,
        sat_thresh=args.sat_thresh
    )
    overlay = visualize_correction(original, restored, corrected_mask)

    # ── Save outputs ─────────────────────────
    stem = Path(args.input).stem
    save(masked_np,           os.path.join(args.output_dir, f"{stem}_masked.png"))
    save(restored,            os.path.join(args.output_dir, f"{stem}_restored.png"))
    cv2.imwrite(os.path.join(args.output_dir, f"{stem}_corrected_mask.png"),
                corrected_mask * 255)
    save(overlay,             os.path.join(args.output_dir, f"{stem}_overlay.png"))

    print(f"Saved results to {args.output_dir}")
    print(f"  - {stem}_masked.png       : Input damaged image")
    print(f"  - {stem}_restored.png     : Digitally restored image")
    print(f"  - {stem}_corrected_mask.png: Correction guidance mask")
    print(f"  - {stem}_overlay.png      : Visual overlay for artisans")


if __name__ == "__main__":
    main()
