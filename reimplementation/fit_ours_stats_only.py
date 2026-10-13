"""Side experiment: can the proposed method reach the paper's entropy / variance /
mean for golfcourse and river if PSNR and SSIM are ignored?

Motivation (see README): the paper's PSNR for "Ours" on these two images is
above the ceiling implied by its own mean and variance, and its beach "Ours"
row equals plain CLAHE. So the joint fit in calibrate.py is pulled between
targets that cannot hold together. Here only Tables 1-3 are fitted, on the two
images whose "Ours" row is not a duplicate of another column.
"""
import json
from pathlib import Path

import numpy as np

import calibrate
import enhance
from metrics import evaluate
from paper import METRICS, load, paper_value

HERE = Path(__file__).resolve().parent
CLASSES = ["golfcourse", "river"]
STATS = ["entropy", "variance", "mean"]


def main():
    base = json.loads((HERE / "params.json").read_text())["params"]
    clip, tile, sigmas = base["CLAHE"]["clip"], base["CLAHE"]["tile"], base["MSR"]["sigmas"]
    originals = {c: load(c) for c in CLASSES}
    g_cache, h_cache, best = {}, {}, (np.inf, None, None)
    for p in calibrate.grid(calibrate.OURS_SPACE):
        errs, got = [], {}
        for c, img in originals.items():
            gk, hk = (c, p["v_stretch"], p["antilog"]), (c, p["alpha"], p["h_stretch"])
            if gk not in g_cache:
                g_cache[gk] = enhance.msr_hsv(img, sigmas, p["v_stretch"], p["antilog"])
            if hk not in h_cache:
                h_cache[hk] = enhance.clahe_color_restored(img, clip, tile, p["alpha"], p["h_stretch"])
            got[c] = evaluate(enhance.fuse(g_cache[gk], h_cache[hk], p["lam"], p["bilateral"]), img)
            errs += [abs(got[c][m] - paper_value(m, c, "Ours")) / calibrate.SCALE[m] for m in STATS]
        if np.mean(errs) < best[0]:
            best = (float(np.mean(errs)), p, got)

    loss, params, got = best
    lines = ["# Proposed method fitted to Tables 1-3 only (golfcourse, river)", "",
             f"Parameters: `{json.dumps(dict(clip=clip, tile=tile, sigmas=sigmas, **params))}`", "",
             f"Mean scaled error on the fitted cells: {loss:.3f}", "",
             "Each cell is `reproduced (paper)`.", "",
             "| image | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for c in CLASSES:
        lines.append(f"| {c} | " + " | ".join(
            f"{got[c][m]:.3f} ({paper_value(m, c, 'Ours'):.3f})" for m in METRICS) + " |")
    text = "\n".join(lines) + "\n"
    (HERE / "results" / "ours_stats_only.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
