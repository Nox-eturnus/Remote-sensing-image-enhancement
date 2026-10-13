"""Identify which UC Merced images the paper used.

The paper only names the classes (golfcourse, river, beach) but reports
entropy / "variance" / mean of each original image (Tables 1-3, column Orig.).
We compute the same statistics for all 100 images of each class and rank them
by distance to the reported values.
"""
from pathlib import Path

import cv2
import numpy as np

from metrics import entropy

DATA = Path(__file__).resolve().parent.parent / "archive" / "UCMerced_LandUse" / "Images"

# class -> (entropy, "variance", mean) of the original image, from Tables 1-3
PAPER_ORIG = {
    "golfcourse": (5.853, 17.527, 151.793),
    "river": (5.736, 16.041, 71.313),
    "beach": (6.671, 40.593, 131.323),
}


def stats(img_bgr):
    """Candidate definitions of (entropy, spread, mean) for an original image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    out = {
        "gray": (entropy(gray), gray.std(), gray.mean()),
        "rgb_avg": (
            np.mean([entropy(img_bgr[..., c]) for c in range(3)]),
            np.mean([img_bgr[..., c].std() for c in range(3)]),
            img_bgr.mean(),
        ),
        "rgb_all": (entropy(img_bgr), img_bgr.std(), img_bgr.mean()),
    }
    return out


def main():
    for cls, target in PAPER_ORIG.items():
        target = np.array(target)
        rows = []
        for f in sorted((DATA / cls).glob("*.tif")):
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            for name, s in stats(img).items():
                s = np.array(s)
                # relative error so entropy and mean weigh similarly
                err = np.abs(s - target) / target
                rows.append((err.max(), f.name, name, s))
        rows.sort(key=lambda r: r[0])
        print(f"\n{cls}: paper Orig entropy/var/mean = {target}")
        for err, fname, name, s in rows[:4]:
            print(f"  {fname:20s} {name:8s} max_rel_err={err:.4f}  {np.round(s, 3)}")


if __name__ == "__main__":
    main()
