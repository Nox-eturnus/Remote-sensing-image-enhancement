"""UC Merced file listing, train/val/test split and image loading.

Images are referred to by key, "<class>/<file>". The split is stratified by
class (70/15/15) and the three images used in the paper are forced into the
test set. It is written to split.json so every script sees the same one.
"""
import json
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent / "reimplementation"
DATA = HERE.parent / "archive" / "UCMerced_LandUse" / "Images"
SPLIT_FILE = HERE / "split.json"

SEED = 2022
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
PAPER_IMAGES = ("golfcourse/golfcourse44.tif", "river/river95.tif", "beach/beach10.tif")

if str(REPRO) not in sys.path:  # enhance.py and metrics.py are reused as they are
    sys.path.insert(0, str(REPRO))


def make_split():
    rng = np.random.default_rng(SEED)
    split = {name: [] for name in FRACTIONS}
    for cls_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
        keys = sorted(f"{cls_dir.name}/{f.name}" for f in cls_dir.glob("*.tif"))
        keys = [keys[i] for i in rng.permutation(len(keys))]
        n_train = round(FRACTIONS["train"] * len(keys))
        n_val = round(FRACTIONS["val"] * len(keys))
        parts = {"train": keys[:n_train], "val": keys[n_train:n_train + n_val],
                 "test": keys[n_train + n_val:]}
        for key in PAPER_IMAGES:
            for name in ("train", "val"):
                if key in parts[name]:  # swap with a test image of the same class
                    other = next(k for k in parts["test"] if k not in PAPER_IMAGES)
                    parts[name][parts[name].index(key)] = other
                    parts["test"][parts["test"].index(other)] = key
        for name in FRACTIONS:
            split[name] += sorted(parts[name])
    return split


def load_split():
    if not SPLIT_FILE.exists():
        SPLIT_FILE.write_text(json.dumps(make_split(), indent=1), encoding="utf-8")
    return json.loads(SPLIT_FILE.read_text(encoding="utf-8"))


def load(key):
    """8-bit BGR image at its native size (2056 of 2100 are 256x256)."""
    img = cv2.imread(str(DATA / key), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(DATA / key)
    return img


def image_seed(key):
    """Seed of the fixed degradation an image gets in the val and test sets."""
    return SEED + zlib.crc32(key.encode())


if __name__ == "__main__":
    for name, keys in load_split().items():
        print(f"{name}: {len(keys)} images")
