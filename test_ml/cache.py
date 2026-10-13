"""Precompute what the learned models read: degraded images and the paper's
two branches (G = HSV-MSR, H = colour-restored CLAHE) at the calibrated settings.

    train          4 degradations (draws 1-4) of each training image, nonuniform
    val, test      the fixed degradation (draw 0) of each image, nonuniform
    test_uniform   the fixed uniform degradation of each test image

Arrays are 8-bit BGR, zero-padded to 257x257 (the largest image); the native
size of each image is kept alongside. They go to ~/.cache/dip_rs (about 4 GB),
or to $DIP_RS_CACHE, to keep them out of OneDrive.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import baselines as B
from data import load, load_split
from degrade import degrade_fixed

CACHE = Path(os.environ.get("DIP_RS_CACHE", Path.home() / ".cache" / "dip_rs"))
SETS = {  # name: (split, degradation variant, draws)
    "train": ("train", "nonuniform", (1, 2, 3, 4)),
    "val": ("val", "nonuniform", (0,)),
    "test": ("test", "nonuniform", (0,)),
    "test_uniform": ("test", "uniform", (0,)),
}
PAD = 257


def _pad(img):
    out = np.zeros((PAD, PAD, 3), np.uint8)
    out[:img.shape[0], :img.shape[1]] = np.clip(np.rint(img), 0, 255)
    return out


def process(job):
    key, variant, draws = job
    gt = load(key)
    samples = []
    for draw in draws:
        lq, _ = degrade_fixed(gt, key, variant, draw)
        g, h = B.fusion_inputs(lq, B.PAPER_PARAMS["Ours"])
        samples.append((_pad(lq), _pad(g), _pad(h)))
    return _pad(gt), gt.shape[:2], samples


def build(name):
    split, variant, draws = SETS[name]
    keys = load_split()[split]
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(process, [(k, variant, draws) for k in keys], chunksize=8))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / f"{name}_gt.npy", np.stack([r[0] for r in results]))
    np.save(CACHE / f"{name}_shape.npy", np.array([r[1] for r in results]))
    for i, part in enumerate(("lq", "g", "h")):  # sample j of image n is at n * len(draws) + j
        np.save(CACHE / f"{name}_{part}.npy", np.stack([s[i] for r in results for s in r[2]]))
    (CACHE / f"{name}_keys.json").write_text(json.dumps({"keys": keys, "draws": len(draws)}))


def load_set(name):
    """Dict with gt (one per image), lq/g/h (one per sample), shape, keys, draws."""
    if not (CACHE / f"{name}_keys.json").exists():
        raise FileNotFoundError(f"{name} is not cached; run cache.py first")
    out = {part: np.load(CACHE / f"{name}_{part}.npy") for part in ("gt", "lq", "g", "h", "shape")}
    return {**out, **json.loads((CACHE / f"{name}_keys.json").read_text())}


if __name__ == "__main__":
    for name in SETS:
        build(name)
        print(f"{name}: {len(load_set(name)['lq'])} samples")
