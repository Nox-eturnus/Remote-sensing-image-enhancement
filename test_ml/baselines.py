"""Non-learned methods of the benchmark, and the scores used to compare them.

Blind methods see only the low-quality image. Oracles also see the ground
truth; they are not methods, they bound what a family of methods can reach.
"""
import json

import numpy as np
from skimage.metrics import peak_signal_noise_ratio

import data  # noqa: F401  (puts ../reimplementation on sys.path)
import enhance
from metrics import evaluate, to_gray

PAPER_PARAMS = json.loads((data.REPRO / "params.json").read_text())["params"]


def score(out, gt):
    """Reference-based scores against the clean image. PSNR, SSIM, MSE and MAE
    are on 8-bit grayscale as in the reproduction; psnr_rgb is on all channels."""
    m = evaluate(out, gt)
    diff = to_gray(out).astype(np.float64) - to_gray(gt).astype(np.float64)
    return {"psnr": m["psnr"], "psnr_rgb": float(peak_signal_noise_ratio(gt, out, data_range=255)),
            "ssim": m["ssim"], "mse": float((diff ** 2).mean()), "mae": float(np.abs(diff).mean()),
            "entropy": m["entropy"]}


def _to_u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ---- blind global baselines -------------------------------------------------

def auto_gamma(lq):
    """Gamma correction that brings the mean grey level to mid-range."""
    mean = np.clip(to_gray(lq).mean() / 255.0, 1e-3, 1 - 1e-3)
    return _to_u8(255.0 * (lq / 255.0) ** (np.log(0.5) / np.log(mean)))


# ---- oracles for global point operations ------------------------------------

def oracle_gamma(lq, gt, grid=np.linspace(0.1, 1.5, 57)):
    """Best single gamma for this image."""
    levels = np.arange(256) / 255.0
    outs = [_to_u8(255.0 * levels ** g)[lq] for g in grid]
    return min(outs, key=lambda o: ((o.astype(np.float64) - gt) ** 2).mean())


def oracle_lut(lq, gt):
    """Best possible point operation: each input level of each channel is sent
    to the mean ground-truth value of the pixels that have it. No tone curve,
    monotone or not, has a lower MSE on this image."""
    out = np.empty(lq.shape)
    for c in range(3):
        count = np.bincount(lq[..., c].ravel(), minlength=256)
        total = np.bincount(lq[..., c].ravel(), weights=gt[..., c].ravel(), minlength=256)
        out[..., c] = (total / np.maximum(count, 1))[lq[..., c]]
    return _to_u8(out)


# ---- the paper's fusion, split into its parts --------------------------------

def fusion_inputs(lq, params):
    """G (HSV-MSR) and H (colour-restored CLAHE) of Sections 2.1-2.2, as floats."""
    g = enhance.msr_hsv(lq, params["sigmas"], tuple(params["v_stretch"]), params["antilog"])
    h = enhance.clahe_color_restored(lq, params["clip"], params["tile"], params["alpha"],
                                     tuple(params["h_stretch"]))
    return g, h


def classical(lq):
    """SSR, MSR, MSRCR, CLAHE and the paper's fusion at the calibrated settings."""
    return {name: enhance.ALGORITHMS[name](lq, **PAPER_PARAMS[name]) for name in enhance.ALGORITHMS}


# ---- oracles for fusion -----------------------------------------------------

def _segment(gt, a, b):
    d = b - a
    t = np.clip((d * (gt - a)).sum(-1, keepdims=True) / np.maximum((d * d).sum(-1, keepdims=True), 1e-12), 0, 1)
    return a + t * d


def oracle_pixel_fusion(gt, candidates):
    """Per-pixel convex combination of 2 or 3 candidate images that is closest
    to the ground truth: the ceiling of any fusion with one weight map per
    candidate shared by the three colour channels."""
    gt = gt.astype(np.float64)
    pts = [np.clip(c, 0, 255).astype(np.float64) for c in candidates]
    if len(pts) == 2:
        return _to_u8(_segment(gt, *pts))
    p0, p1, p2 = pts
    edges = [_segment(gt, p0, p1), _segment(gt, p0, p2), _segment(gt, p1, p2)]
    err = np.stack([((e - gt) ** 2).sum(-1) for e in edges])
    best = np.take_along_axis(np.stack(edges), err.argmin(0)[None, ..., None], 0)[0]
    # interior of the triangle: unconstrained projection onto its plane
    e1, e2, d = p1 - p0, p2 - p0, gt - p0
    a11, a12, a22 = (e1 * e1).sum(-1), (e1 * e2).sum(-1), (e2 * e2).sum(-1)
    b1, b2 = (e1 * d).sum(-1), (e2 * d).sum(-1)
    det = a11 * a22 - a12 ** 2
    safe = np.where(det > 1e-9, det, 1.0)
    u, v = (a22 * b1 - a12 * b2) / safe, (a11 * b2 - a12 * b1) / safe
    inside = (det > 1e-9) & (u >= 0) & (v >= 0) & (u + v <= 1)
    plane = p0 + u[..., None] * e1 + v[..., None] * e2
    return _to_u8(np.where(inside[..., None], plane, best))
