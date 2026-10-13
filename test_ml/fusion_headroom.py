"""How much can a learned fusion of the paper's two branches gain?

For G (HSV-MSR) and H (colour-restored CLAHE) computed on the low-quality
image, compares on the test split:

    paper lambda            the calibrated lambda of the reproduction
    best fixed lambda       one lambda for all images, chosen on the val split
    oracle image lambda     best lambda per image (ceiling of a scalar-lambda model)
    oracle pixel fusion     best per-pixel convex combination (ceiling of a
                            weight-map model), over {G, H}, {input, G, H} and
                            {auto gamma, G, H}

for the calibrated branch settings of the reproduction and for conventional
ones. The lambda rows include the bilateral filter, the pixel rows do not.
Writes results/<variant>/fusion_headroom.md and .csv.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import baselines as B
import enhance
from data import HERE, load, load_split
from degrade import VARIANTS, degrade_fixed

LAMBDAS = np.round(np.linspace(0, 1, 21), 2)
CONVENTIONAL = {"clip": 2.0, "tile": 8, "sigmas": [15, 80, 250], "alpha": 125.0, "h_stretch": ["pct", 1.0],
                "v_stretch": ["pct", 1.0], "antilog": True, "lam": 0.5, "bilateral": [5, 25, 5]}
SETTINGS = {"calibrated": B.PAPER_PARAMS["Ours"], "conventional": CONVENTIONAL}
PIXEL_SETS = ["pixel {G,H}", "pixel {input,G,H}", "pixel {auto gamma,G,H}"]


def process(job):
    key, variant, with_pixel = job
    gt = load(key)
    lq, _ = degrade_fixed(gt, key, variant)
    row = {"image": key}
    for setting, params in SETTINGS.items():
        g, h = B.fusion_inputs(lq, params)
        for lam in LAMBDAS:
            out = enhance.fuse(g, h, lam, tuple(params["bilateral"]))
            row[f"{setting}|{lam:.2f}"] = B.score(out, gt)["psnr"]
        if with_pixel:
            sets = [[g, h], [lq, g, h], [B.auto_gamma(lq), g, h]]
            for name, candidates in zip(PIXEL_SETS, sets):
                row[f"{setting}|{name}"] = B.score(B.oracle_pixel_fusion(gt, candidates), gt)["psnr"]
    return row


def run(keys, variant, with_pixel):
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = pool.map(process, [(k, variant, with_pixel) for k in keys], chunksize=8)
        return pd.DataFrame(rows).set_index("image")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=VARIANTS, default=VARIANTS[0])
    variant = parser.parse_args().variant
    out_dir = HERE / "results" / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    split = load_split()
    val, test = run(split["val"], variant, False), run(split["test"], variant, True)
    test.to_csv(out_dir / "fusion_headroom.csv", float_format="%.4f")

    lines = [f"# Headroom of fusing the paper's two branches [{variant} degradation]", "",
             f"Mean grayscale PSNR (dB) against the clean image, {len(test)} test images.",
             f"The fixed lambda is chosen on the {len(val)} val images.", "",
             "| | calibrated branches | conventional branches |", "|---|---:|---:|"]
    cells = {}
    for setting, params in SETTINGS.items():
        sweep = [f"{setting}|{lam:.2f}" for lam in LAMBDAS]
        best = val[sweep].mean().idxmax()
        cells[setting] = {
            "paper lambda": f"{test[f'{setting}|{params['lam']:.2f}'].mean():.2f} (lambda {params['lam']})",
            "best fixed lambda": f"{test[best].mean():.2f} (lambda {best.split('|')[1]})",
            "oracle image lambda": f"{test[sweep].max(axis=1).mean():.2f}",
            **{f"oracle {name}": f"{test[f'{setting}|{name}'].mean():.2f}" for name in PIXEL_SETS},
        }
    for name in cells["calibrated"]:
        lines.append(f"| {name} | {cells['calibrated'][name]} | {cells['conventional'][name]} |")
    lines += ["", "Mean PSNR over the test split for each fixed lambda:", "",
              "| lambda | " + " | ".join(f"{lam:.2f}" for lam in LAMBDAS[::2]) + " |",
              "|---|" + "---:|" * len(LAMBDAS[::2])]
    for setting in SETTINGS:
        lines.append(f"| {setting} | " + " | ".join(f"{test[f'{setting}|{lam:.2f}'].mean():.2f}"
                                                    for lam in LAMBDAS[::2]) + " |")
    (out_dir / "fusion_headroom.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
