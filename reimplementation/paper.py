"""Numbers reported in Meng et al., "Remote Sensing Image Enhancement Based on
MSR and CLAHE" (WCSE 2022, doi:10.18178/wcse.2022.04.059), Tables 1-5, plus the
UC Merced images they correspond to (identified by find_images.py)."""
from pathlib import Path

import cv2

DATA = Path(__file__).resolve().parent.parent / "archive" / "UCMerced_LandUse" / "Images"

# class -> file; matched on the Orig. entropy / variance / mean to 3 decimals
IMAGES = {
    "golfcourse": "golfcourse44.tif",
    "river": "river95.tif",
    "beach": "beach10.tif",
}

METHODS = ["SSR", "MSR", "MSRCR", "CLAHE", "Ours"]
METRICS = ["entropy", "variance", "mean", "psnr", "ssim"]

# TABLES[metric][class] = [Orig, SSR, MSR, MSRCR, CLAHE, Ours]  (no Orig for psnr/ssim)
TABLES = {
    "entropy": {
        "golfcourse": [5.853, 6.184, 6.173, 6.213, 6.822, 7.511],
        "river": [5.736, 6.026, 6.011, 6.025, 6.906, 7.411],
        "beach": [6.671, 6.870, 6.759, 6.749, 6.855, 7.080],
    },
    "variance": {
        "golfcourse": [17.527, 22.773, 21.959, 22.727, 28.866, 44.736],
        "river": [16.041, 18.147, 18.934, 19.046, 34.918, 48.311],
        "beach": [40.593, 43.898, 42.413, 42.037, 42.061, 43.916],
    },
    "mean": {
        "golfcourse": [151.793, 149.520, 149.190, 152.682, 148.838, 143.502],
        "river": [71.313, 89.751, 92.867, 93.195, 92.090, 103.132],
        "beach": [131.323, 103.680, 114.101, 114.104, 131.754, 128.726],
    },
    "psnr": {
        "golfcourse": [30.708, 30.966, 31.943, 24.953, 26.676],
        "river": [21.033, 21.033, 20.915, 18.782, 19.162],
        "beach": [18.282, 22.243, 22.248, 27.161, 24.162],
    },
    "ssim": {
        "golfcourse": [0.937, 0.945, 0.952, 0.827, 0.886],
        "river": [0.921, 0.921, 0.921, 0.705, 0.753],
        "beach": [0.918, 0.964, 0.964, 0.964, 0.915],
    },
}


def paper_value(metric, cls, method):
    """Reported value for `method` in METHODS or "Orig" (None if not reported)."""
    row = TABLES[metric][cls]
    cols = (["Orig"] if len(row) == 6 else []) + METHODS
    return row[cols.index(method)] if method in cols else None


def load(cls):
    img = cv2.imread(str(DATA / cls / IMAGES[cls]), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(DATA / cls / IMAGES[cls])
    return img
