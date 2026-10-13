"""Reference-based benchmark of the non-learned methods on the test split.

Every test image is degraded with its fixed seed, each method is run on the
low-quality image and scored against the clean one. Writes, under
results/<variant>/benchmark/:

    metrics.csv       one row per image and method
    summary.md        test-set means, per-class PSNR, the three paper images
    fig_<class>.png   comparison panel for the three paper images
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import baselines as B
from data import HERE, PAPER_IMAGES, load, load_split
from degrade import VARIANTS, degrade_fixed
from metrics import entropy, to_gray

BLIND = ["Input", "Auto gamma", "SSR", "MSR", "MSRCR", "CLAHE", "Paper MSR+CLAHE"]
ORACLES = ["Oracle gamma", "Oracle point op"]
COLUMNS = ["psnr", "psnr_rgb", "ssim", "mse", "mae", "entropy"]


def outputs(lq, gt):
    classical = B.classical(lq)
    classical["Paper MSR+CLAHE"] = classical.pop("Ours")
    return {"Input": lq, "Auto gamma": B.auto_gamma(lq), **classical,
            "Oracle gamma": B.oracle_gamma(lq, gt), "Oracle point op": B.oracle_lut(lq, gt)}


def process(variant, key):
    gt = load(key)
    lq, _ = degrade_fixed(gt, key, variant)
    rows = [{"image": key, "method": name, **B.score(out, gt)} for name, out in outputs(lq, gt).items()]
    return rows + [{"image": key, "method": "Clean", "entropy": entropy(to_gray(gt))}]


def md_table(df, fmt):
    lines = ["| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |",
             "|---|" + "---:|" * len(df.columns)]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join("" if pd.isna(v) else format(v, fmt) for v in row) + " |")
    return lines


def figure(key, variant, out_dir):
    gt = load(key)
    lq, _ = degrade_fixed(gt, key, variant)
    panel = {"Clean": gt, **outputs(lq, gt)}
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    for ax, (name, im) in zip(axes.ravel(), panel.items()):
        title = name if name == "Clean" else f"{name}\n{B.score(im, gt)['psnr']:.2f} dB"
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{key} [{variant} degradation]", fontsize=11)
    fig.tight_layout(h_pad=2.5)
    fig.savefig(out_dir / f"fig_{key.split('/')[0]}.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=VARIANTS, default=VARIANTS[0])
    variant = parser.parse_args().variant
    out_dir = HERE / "results" / variant / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = load_split()["test"]
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = [r for image_rows in pool.map(partial(process, variant), keys, chunksize=8)
                for r in image_rows]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False, float_format="%.4f")
    df["class"] = df.image.str.split("/").str[0]

    order = BLIND + ORACLES
    overall = df.groupby("method")[COLUMNS].mean().loc[order + ["Clean"]]
    overall.index.name = "method"
    by_class = df.pivot_table(index="class", columns="method", values="psnr")[order]
    paper = df[df.image.isin(PAPER_IMAGES)].pivot_table(index="image", columns="method", values="psnr")[order]

    lines = [f"# Benchmark: non-learned methods on the test split [{variant} degradation]", "",
             f"{len(keys)} test images, each with one fixed synthetic degradation (`degrade.py`).",
             "Scores are against the clean image; `psnr`, `ssim`, `mse`, `mae` on 8-bit grayscale,",
             "`psnr_rgb` on all channels. Means of per-image values. The two oracle rows use the",
             "clean image and are bounds, not methods.", "",
             "## Test-set means", ""] + md_table(overall, ".3f")
    lines += ["", "## PSNR (dB) on the three images of the paper", ""] + md_table(paper, ".2f")
    lines += ["", "## PSNR (dB) by class", ""] + md_table(by_class, ".2f")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(overall.round(3).to_string())

    for key in PAPER_IMAGES:
        figure(key, variant, out_dir)


if __name__ == "__main__":
    main()
