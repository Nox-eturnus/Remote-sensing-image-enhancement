"""Objective metrics used in Section 3.2 of the paper.

All metrics are computed on the 8-bit grayscale (luma) version of the image,
which is the definition that reproduces the paper's "Orig." columns exactly
(see find_images.py).
"""
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def to_gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def entropy(img):
    """Shannon entropy (bits) of the 8-bit intensity histogram."""
    hist = np.bincount(np.asarray(img, dtype=np.uint8).ravel(), minlength=256)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def evaluate(enhanced_bgr, original_bgr):
    """Return the five indexes of Tables 1-5 for one enhanced image."""
    g, g0 = to_gray(enhanced_bgr), to_gray(original_bgr)
    return {
        "entropy": entropy(g),
        # the paper calls this "variance" but the values are standard deviations
        "variance": float(g.std()),
        "mean": float(g.mean()),
        "psnr": float(peak_signal_noise_ratio(g0, g, data_range=255)),
        "ssim": float(structural_similarity(g0, g, data_range=255)),
    }
