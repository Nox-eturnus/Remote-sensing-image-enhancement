"""Fit the parameters the paper does not report.

For every method, grid-search the unspecified parameters and keep the setting
whose five metrics on the three test images are closest to Tables 1-5. The
result is written to params.json and used by run_reproduction.py.

This is curve fitting against the reported numbers, not an independent
replication: it answers "is there a setting of the described algorithm that
produces the reported tables?".
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

import enhance
from metrics import evaluate
from paper import IMAGES, METRICS, load, paper_value

HERE = Path(__file__).resolve().parent

# Error of one unit of each metric, so that they can be averaged
SCALE = {"entropy": 0.3, "variance": 5.0, "mean": 10.0, "psnr": 2.0, "ssim": 0.05}

STRETCHES = [("minmax", 0.0), ("pct", 0.1), ("pct", 0.5), ("pct", 1.0), ("pct", 2.0),
             ("meanstd", 2.0), ("meanstd", 2.5), ("meanstd", 3.0)]
SIGMA_SETS = [(15, 80, 250), (15, 80, 200), (10, 60, 200), (20, 100, 240), (30, 150, 300), (5, 40, 250)]

SPACES = {
    "SSR": dict(sigma=[15, 30, 50, 80, 100, 150, 200, 250, 300], stretch_cfg=STRETCHES,
                joint=[True, False], antilog=[False, True]),
    "MSR": dict(sigmas=SIGMA_SETS, stretch_cfg=STRETCHES, joint=[True, False], antilog=[False, True]),
    "MSRCR": dict(sigmas=SIGMA_SETS, alpha=[25.0, 75.0, 125.0, 200.0, 500.0, 1000.0],
                  stretch_cfg=STRETCHES, joint=[True, False], antilog=[False, True]),
    "CLAHE": dict(clip=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0], tile=[2, 4, 8, 16],
                  space=["rgb", "hsv", "lab", "ycrcb"]),
}
# The proposed method reuses the CLAHE and MSR settings fitted for the baselines
OURS_SPACE = dict(alpha=[10.0, 25.0, 50.0, 125.0, 250.0],
                  h_stretch=STRETCHES + [("none", 0.0)],
                  v_stretch=STRETCHES, antilog=[False, True],
                  lam=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                  bilateral=[None, (5, 25, 5), (5, 50, 5), (9, 75, 75)])


# The large-sigma Gaussian blurs dominate the run time and depend only on the
# image channel and the scales, not on the other parameters being searched.
_log_retinex, _retinex_cache = enhance._log_retinex, {}


def _cached_log_retinex(ch, sigmas):
    key = (hash(ch.tobytes()), tuple(sigmas))
    if key not in _retinex_cache:
        _retinex_cache[key] = _log_retinex(ch, sigmas)
    return _retinex_cache[key]


enhance._log_retinex = _cached_log_retinex


def loss(method, outputs, originals):
    """Mean scaled absolute deviation from the paper over 3 images x 5 metrics."""
    errs = []
    for cls, out in outputs.items():
        got = evaluate(out, originals[cls])
        errs += [abs(got[m] - paper_value(m, cls, method)) / SCALE[m] for m in METRICS]
    return float(np.mean(errs))


def grid(space):
    keys = list(space)
    for values in itertools.product(*space.values()):
        yield dict(zip(keys, values))


def fit_baseline(method, originals):
    best = (np.inf, None)
    for params in grid(SPACES[method]):
        outs = {c: enhance.ALGORITHMS[method](img, **params) for c, img in originals.items()}
        best = min(best, (loss(method, outs, originals), params), key=lambda t: t[0])
    return best


def fit_ours(originals, clahe_params, msr_params):
    clip, tile, sigmas = clahe_params["clip"], clahe_params["tile"], msr_params["sigmas"]
    g_cache, h_cache = {}, {}
    best = (np.inf, None)
    for p in grid(OURS_SPACE):
        outs = {}
        for c, img in originals.items():
            gk, hk = (c, p["v_stretch"], p["antilog"]), (c, p["alpha"], p["h_stretch"])
            if gk not in g_cache:
                g_cache[gk] = enhance.msr_hsv(img, sigmas, p["v_stretch"], p["antilog"])
            if hk not in h_cache:
                h_cache[hk] = enhance.clahe_color_restored(img, clip, tile, p["alpha"], p["h_stretch"])
            outs[c] = enhance.fuse(g_cache[gk], h_cache[hk], p["lam"], p["bilateral"])
        best = min(best, (loss("Ours", outs, originals), p), key=lambda t: t[0])
    return best[0], dict(clip=clip, tile=tile, sigmas=sigmas, **best[1])


def main():
    """Usage: calibrate.py [METHOD ...]  - refit only the named methods, keeping
    the others from params.json; with no argument, fit everything."""
    todo = sys.argv[1:] or list(SPACES) + ["Ours"]
    out = HERE / "params.json"
    saved = json.loads(out.read_text()) if out.exists() else {"params": {}, "loss": {}}
    fitted, losses = saved["params"], saved["loss"]
    originals = {c: load(c) for c in IMAGES}
    for method in todo:
        if method == "Ours":
            losses[method], fitted[method] = fit_ours(originals, fitted["CLAHE"], fitted["MSR"])
        else:
            losses[method], fitted[method] = fit_baseline(method, originals)
        print(f"{method:6s} loss={losses[method]:.3f}  {fitted[method]}", flush=True)
        out.write_text(json.dumps({"params": fitted, "loss": losses}, indent=2))


if __name__ == "__main__":
    main()
