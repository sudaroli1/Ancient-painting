"""
Corrected Mask Construction Module
-----------------------------------
Bridges digital restoration → physical restoration by:
  1. Detecting over-painted regions via hue/saturation deviation
  2. Correcting contextual elements (background lines, motifs) that
     neural networks tend to over-fill
  3. Producing a corrected mask that guides manual inpainting artisans

Corresponds to Section 3.3 of the paper.
"""

import cv2
import numpy as np
from typing import Tuple


# ─────────────────────────────────────────────────────
#  Thresholds (tuneable; see Appendix / ablation study)
# ─────────────────────────────────────────────────────
HUE_THRESHOLD        = 15.0   # degrees (0-180 in OpenCV)
SAT_THRESHOLD        = 40.0   # 0-255
EDGE_CANNY_LOW       = 50
EDGE_CANNY_HIGH      = 150
CONTOUR_MIN_AREA     = 50     # px² — filter noise


def rgb_to_hsv(image_rgb: np.ndarray) -> np.ndarray:
    """Convert float32 RGB [0,1] → uint8 HSV for OpenCV operations."""
    uint8 = (np.clip(image_rgb, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(uint8, cv2.COLOR_RGB2HSV)


def compute_hue_sat_diff(
    original: np.ndarray,
    restored: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-pixel absolute difference in Hue and Saturation channels.
    Paper criterion: pixels with ΔH > HUE_THRESHOLD or ΔS > SAT_THRESHOLD
    are candidates for over-painting correction.

    Args:
        original : H×W×3 float32 RGB, original damaged mural
        restored : H×W×3 float32 RGB, digitally restored image

    Returns:
        hue_diff : H×W float32
        sat_diff : H×W float32
    """
    hsv_orig = rgb_to_hsv(original).astype(np.float32)
    hsv_rest = rgb_to_hsv(restored).astype(np.float32)

    # Hue is circular (0-180); use shortest angular distance
    hue_diff = np.abs(hsv_orig[:, :, 0] - hsv_rest[:, :, 0])
    hue_diff = np.minimum(hue_diff, 180.0 - hue_diff)
    sat_diff = np.abs(hsv_orig[:, :, 1] - hsv_rest[:, :, 1])

    return hue_diff, sat_diff


def build_overpaint_mask(
    original: np.ndarray,
    restored: np.ndarray,
    hue_thresh: float = HUE_THRESHOLD,
    sat_thresh: float = SAT_THRESHOLD
) -> np.ndarray:
    """
    Binary mask (H×W uint8) where 1 = region that the digital
    restoration has over-painted beyond acceptable thresholds.
    """
    hue_diff, sat_diff = compute_hue_sat_diff(original, restored)
    mask = ((hue_diff > hue_thresh) | (sat_diff > sat_thresh)).astype(np.uint8)

    # Morphological closing to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def extract_contextual_edges(image: np.ndarray) -> np.ndarray:
    """
    Detect fine structural lines (background motifs, borders) that
    manual artisans preserve but GANs tend to erase or smear.

    Returns:
        edge_mask : H×W uint8 binary
    """
    gray = cv2.cvtColor(
        (np.clip(image, 0, 1) * 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    )
    edges = cv2.Canny(gray, EDGE_CANNY_LOW, EDGE_CANNY_HIGH)
    # Dilate slightly so the artisan's guide has some tolerance
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1)


def filter_contours_by_area(mask: np.ndarray,
                             min_area: int = CONTOUR_MIN_AREA) -> np.ndarray:
    """Remove tiny spurious blobs from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(clean, [cnt], -1, 1, thickness=cv2.FILLED)
    return clean


def build_corrected_mask(
    original: np.ndarray,
    restored: np.ndarray,
    damage_mask: np.ndarray,
    hue_thresh: float   = HUE_THRESHOLD,
    sat_thresh: float   = SAT_THRESHOLD,
) -> np.ndarray:
    """
    Main function — produces the corrected guidance mask for
    physical (manual) inpainting.

    Pipeline:
      1. Identify over-painted pixels  → overpaint_mask
      2. Detect preserved fine lines   → edge_mask
      3. Subtract contextual edges     → prevents artisan over-correction
      4. Intersect with original damage region

    Args:
        original    : H×W×3 float32 RGB original mural
        restored    : H×W×3 float32 RGB digitally restored image
        damage_mask : H×W uint8 binary — original damage / missing region

    Returns:
        corrected_mask : H×W uint8 (values 0 or 1)
    """
    # Step 1 – Over-painted candidate pixels
    overpaint = build_overpaint_mask(original, restored, hue_thresh, sat_thresh)

    # Step 2 – Structural / contextual edges from original
    edges = extract_contextual_edges(original)

    # Step 3 – Remove edge pixels from overpaint candidates
    #           (artisans will handle fine lines manually)
    corrected = overpaint.copy()
    corrected[edges > 0] = 0

    # Step 4 – Only correct within the actual damaged region
    corrected = corrected & damage_mask.astype(np.uint8)

    # Step 5 – Clean up isolated noise
    corrected = filter_contours_by_area(corrected)

    return corrected


def visualize_correction(
    original: np.ndarray,
    restored: np.ndarray,
    corrected_mask: np.ndarray
) -> np.ndarray:
    """
    Overlay corrected mask as a semi-transparent red region on the
    restored image — for qualitative inspection.
    """
    vis = (np.clip(restored, 0, 1) * 255).astype(np.uint8).copy()
    overlay = vis.copy()
    overlay[corrected_mask == 1] = [220, 50, 50]
    return cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
