"""Synthetic degradation: clean UC Merced image -> low-quality image.

The chain follows the acquisition order (optics, illumination, sensor):

    blur (30 % of images)   Gaussian, sigma in [0.5, 1.5] px
    darkening               x ** gamma, gamma in [1.5, 3.5]
    uneven illumination     x * L, L a smooth field in [l_min, 1], l_min in [0.3, 0.8]
                            ("nonuniform" variant only)
    contrast reduction      mean + c * (x - mean), c in [0.6, 1.0]
    noise                   shot (variance x / peak, peak in [300, 3000], log-uniform)
                            plus read noise (std in [0.002, 0.015])
    8-bit quantisation

on intensities in [0, 1]. Noise is added after the darkening, so undoing the
darkening amplifies it, most strongly in the shadows.

The illumination field is a 2x2 to 4x4 grid of random values interpolated
bicubically to the image size. The "uniform" variant leaves it out; it was the
first protocol and is kept as the case in which a global tone curve suffices.
"""
import cv2
import numpy as np

from data import image_seed

VARIANTS = ("nonuniform", "uniform")
RANGES = {
    "blur_prob": 0.3,
    "blur_sigma": (0.5, 1.5),
    "gamma": (1.5, 3.5),
    "contrast": (0.6, 1.0),
    "peak": (300.0, 3000.0),
    "read_std": (0.002, 0.015),
    "illum_min": (0.3, 0.8),
    "illum_grid": (2, 4),
}


def sample_params(rng, variant="nonuniform"):
    blur = rng.uniform(*RANGES["blur_sigma"]) if rng.random() < RANGES["blur_prob"] else 0.0
    p = {
        "blur_sigma": float(blur),
        "gamma": float(rng.uniform(*RANGES["gamma"])),
        "contrast": float(rng.uniform(*RANGES["contrast"])),
        "peak": float(np.exp(rng.uniform(*np.log(RANGES["peak"])))),
        "read_std": float(rng.uniform(*RANGES["read_std"])),
    }
    if variant == "nonuniform":
        k = int(rng.integers(RANGES["illum_grid"][0], RANGES["illum_grid"][1] + 1))
        p["illum_min"] = float(rng.uniform(*RANGES["illum_min"]))
        p["illum_grid"] = rng.random((k, k))
    return p


def illumination(shape, l_min, grid):
    """Smooth field in [l_min, 1] of the given (height, width)."""
    field = cv2.resize(grid, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    field = (field - field.min()) / max(field.max() - field.min(), 1e-12)
    return l_min + (1.0 - l_min) * field


def degrade(img, rng, variant="nonuniform"):
    """Return (low-quality 8-bit BGR image, parameters used)."""
    p = sample_params(rng, variant)
    x = img.astype(np.float64) / 255.0
    if p["blur_sigma"] > 0:
        x = cv2.GaussianBlur(x, (0, 0), p["blur_sigma"])
    x = x ** p["gamma"]
    if "illum_min" in p:
        x = x * illumination(x.shape, p["illum_min"], p["illum_grid"])[..., None]
    x = x.mean() + p["contrast"] * (x - x.mean())
    x = x + rng.normal(size=x.shape) * np.sqrt(np.clip(x, 0, None) / p["peak"] + p["read_std"] ** 2)
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8), p


def degrade_fixed(img, key, variant="nonuniform", draw=0):
    """A reproducible degradation of an image. Val and test images always get
    draw 0; training uses draws 1, 2, ... of the training images."""
    seed = image_seed(key) if draw == 0 else [image_seed(key), draw]
    return degrade(img, np.random.default_rng(seed), variant)
