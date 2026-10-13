"""Enhancement algorithms compared in the paper: SSR, MSR, MSRCR, CLAHE and the
proposed CLAHE(+colour restoration) / HSV-MSR fusion.

All functions take and return 8-bit BGR images (OpenCV convention).

The paper does not state any parameter value, nor how retinex output is mapped
back to [0, 255]. Everything it leaves open is an explicit argument here; the
`stretch` argument is a (mode, amount) pair, see `stretch()`.
"""
import cv2
import numpy as np


def stretch(x, mode="pct", amount=1.0):
    """Map a float array to [0, 255].

    minmax  : full range
    pct     : clip `amount` percent at each tail (simplest colour balance)
    meanstd : clip at mean +- amount * std
    none    : no rescaling, just clip (x is assumed to be in [0, 255] already)
    """
    if mode == "none":
        return np.clip(x, 0, 255)
    if mode == "minmax":
        lo, hi = x.min(), x.max()
    elif mode == "pct":
        lo, hi = np.percentile(x, [amount, 100 - amount])
    elif mode == "meanstd":
        lo, hi = x.mean() - amount * x.std(), x.mean() + amount * x.std()
    else:
        raise ValueError(mode)
    return np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1) * 255


def _stretch_image(x, stretch_cfg, joint):
    """Stretch an HxWx3 array with one range for all channels or one per channel."""
    if joint:
        return stretch(x, *stretch_cfg)
    return np.dstack([stretch(x[..., c], *stretch_cfg) for c in range(x.shape[2])])


def _log_retinex(ch, sigmas):
    """Eq. (2)/(3): equal-weight sum over scales of lg I - lg(I * F_n)."""
    ch = ch.astype(np.float64) + 1.0
    return sum(np.log(ch) - np.log(cv2.GaussianBlur(ch, (0, 0), s)) for s in sigmas) / len(sigmas)


def _to_u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def msr(img, sigmas=(15, 80, 250), stretch_cfg=("pct", 1.0), joint=True, antilog=False):
    """Multi-scale retinex applied to each RGB channel independently."""
    r = np.dstack([_log_retinex(img[..., c], sigmas) for c in range(3)])
    if antilog:
        r = np.exp(r)
    return _to_u8(_stretch_image(r, stretch_cfg, joint))


def ssr(img, sigma=80, **kw):
    """Single-scale retinex = MSR with one scale."""
    return msr(img, sigmas=(sigma,), **kw)


def msrcr(img, sigmas=(15, 80, 250), alpha=125.0, stretch_cfg=("pct", 1.0), joint=True,
          antilog=False):
    """MSR with colour restoration (Jobson et al. 1997).

    The gain/offset and beta of the original formulation are absorbed by the
    final stretch, so alpha is the only colour-restoration parameter left.
    """
    i = img.astype(np.float64) + 1.0
    r = np.dstack([_log_retinex(img[..., c], sigmas) for c in range(3)])
    if antilog:
        r = np.exp(r)
    c = np.log(alpha * i) - np.log(i.sum(axis=2, keepdims=True))
    return _to_u8(_stretch_image(c * r, stretch_cfg, joint))


def clahe(img, clip=2.0, tile=8, space="rgb"):
    """CLAHE on every RGB channel (the colour-distorting variant the paper
    criticises) or on the luminance channel of another colour space."""
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    if space == "rgb":
        return np.dstack([op.apply(img[..., c]) for c in range(3)])
    fwd, inv, ch = {
        "hsv": (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR, 2),
        "lab": (cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2BGR, 0),
        "ycrcb": (cv2.COLOR_BGR2YCrCb, cv2.COLOR_YCrCb2BGR, 0),
    }[space]
    x = cv2.cvtColor(img, fwd)
    x[..., ch] = op.apply(x[..., ch])
    return cv2.cvtColor(x, inv)


def clahe_color_restored(img, clip=2.0, tile=8, alpha=125.0, stretch_cfg=("pct", 1.0)):
    """Section 2.2: per-channel CLAHE followed by Eqs. (4)-(7).

    C_i = beta * (log(alpha * I_i) - log(sum_j I_j)) is computed on the original
    image and multiplies the CLAHE output. With stretch mode "none", beta is set
    to 1 / log(alpha / 3) so that C = 1 on grey pixels; otherwise beta is
    absorbed by the stretch, which uses one range for all channels so the
    restored channel ratios survive.
    """
    i = img.astype(np.float64) + 1.0
    c = np.log(alpha * i) - np.log(i.sum(axis=2, keepdims=True))
    if stretch_cfg[0] == "none":
        c = c / np.log(alpha / 3.0)
    return stretch(c * clahe(img, clip, tile, "rgb"), *stretch_cfg)


def msr_hsv(img, sigmas=(15, 80, 250), stretch_cfg=("pct", 1.0), antilog=True):
    """Section 2.1: MSR on the V channel only, H and S untouched (Eq. 3)."""
    hsv = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2HSV)
    r = _log_retinex(hsv[..., 2] * 255.0, sigmas)
    if antilog:
        r = np.exp(r)
    hsv[..., 2] = (stretch(r, *stretch_cfg) / 255.0).astype(np.float32)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float64) * 255.0


def fuse(g, h, lam=0.5, bilateral=(5, 25, 5)):
    """Section 2.3: F = lam * G + (1 - lam) * H (Eq. 8), then bilateral filter
    (Eqs. 9-10). `bilateral` is (diameter, sigma_r, sigma_d) or None."""
    f = (lam * g + (1.0 - lam) * h).astype(np.float32)
    if bilateral:
        d, sigma_r, sigma_d = bilateral
        f = cv2.bilateralFilter(f, int(d), float(sigma_r), float(sigma_d))
    return _to_u8(f)


def ours(img, clip=2.0, tile=8, sigmas=(15, 80, 250), alpha=125.0, h_stretch=("pct", 1.0),
         v_stretch=("pct", 1.0), antilog=True, lam=0.5, bilateral=(5, 25, 5)):
    """The proposed method (Fig. 1)."""
    g = msr_hsv(img, sigmas, v_stretch, antilog)
    h = clahe_color_restored(img, clip, tile, alpha, h_stretch)
    return fuse(g, h, lam, bilateral)


ALGORITHMS = {"SSR": ssr, "MSR": msr, "MSRCR": msrcr, "CLAHE": clahe, "Ours": ours}

# Conventional values from the retinex / OpenCV literature, used when no
# calibrated parameter file is present.
DEFAULTS = {name: {} for name in ALGORITHMS}
