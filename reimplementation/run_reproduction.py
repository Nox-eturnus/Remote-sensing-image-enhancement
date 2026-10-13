"""Reproduce Figs. 2-4 and Tables 1-5 of the paper.

Runs every method on the three test images twice: with conventional default
parameters ("default") and with the parameters fitted by calibrate.py
("calibrated"). Writes, under results/<setting>/:

    images/<class>_<method>.png   enhanced images
    fig_<class>.png               2x3 comparison panel (layout of Figs. 2-4)
    tables.md                     Tables 1-5, reproduced value next to the paper's
    metrics.csv                   the same numbers in long format
"""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import enhance
from metrics import evaluate
from paper import IMAGES, METHODS, METRICS, load, paper_value

HERE = Path(__file__).resolve().parent
TABLE_TITLES = {"entropy": "Table 1: Information entropy", "variance": "Table 2: Variance (std)",
                "mean": "Table 3: Mean", "psnr": "Table 4: PSNR (dB)", "ssim": "Table 5: SSIM"}
FIGURES = {"river": "Fig. 2", "golfcourse": "Fig. 3", "beach": "Fig. 4"}


def psnr_ceiling(cls, method):
    """Highest PSNR compatible with the paper's own mean and std for this cell:
    MSE >= (mean_x - mean_y)^2 + (std_x - std_y)^2 for any pair of images."""
    d_mean = paper_value("mean", cls, method) - paper_value("mean", cls, "Orig")
    d_std = paper_value("variance", cls, method) - paper_value("variance", cls, "Orig")
    return 10 * np.log10(255.0 ** 2 / (d_mean ** 2 + d_std ** 2))


def run(setting, params):
    out_dir = HERE / "results" / setting
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for cls in IMAGES:
        orig = load(cls)
        panel = {"Original": orig}
        with np.errstate(divide="ignore"):  # PSNR of an image against itself
            orig_stats = evaluate(orig, orig)
        for name in ("entropy", "variance", "mean"):
            rows.append((cls, "Orig", name, orig_stats[name], paper_value(name, cls, "Orig")))
        for method in METHODS:
            out = enhance.ALGORITHMS[method](orig, **params.get(method, {}))
            cv2.imwrite(str(out_dir / "images" / f"{cls}_{method}.png"), out)
            panel[method] = out
            for name, value in evaluate(out, orig).items():
                rows.append((cls, method, name, value, paper_value(name, cls, method)))

        fig, axes = plt.subplots(2, 3, figsize=(9, 6.3))
        for ax, (title, im), tag in zip(axes.ravel(), panel.items(), "abcdef"):
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.set_title(f"({tag}) {title}", fontsize=10)
            ax.axis("off")
        fig.suptitle(f"{FIGURES[cls]} reproduction - {IMAGES[cls]} [{setting}]", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"fig_{cls}.png", dpi=150)
        plt.close(fig)

    df = pd.DataFrame(rows, columns=["image", "method", "metric", "reproduced", "paper"])
    df["abs_error"] = (df.reproduced - df.paper).abs()
    df.to_csv(out_dir / "metrics.csv", index=False, float_format="%.4f")

    lines = [f"# Reproduced tables [{setting}]", "",
             "Each cell is `reproduced (paper)`.", ""]
    for metric in METRICS:
        cols = [m for m in ["Orig"] + METHODS if paper_value(metric, "river", m) is not None]
        lines += [f"## {TABLE_TITLES[metric]}", "", "| | " + " | ".join(cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for cls in IMAGES:
            sub = df[(df.image == cls) & (df.metric == metric)].set_index("method")
            cells = [f"{sub.reproduced[m]:.3f} ({sub.paper[m]:.3f})" for m in cols]
            lines.append(f"| {cls} | " + " | ".join(cells) + " |")
        lines.append("")
    (out_dir / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def consistency_report():
    """Check the paper's PSNR against the ceiling implied by its mean and std."""
    lines = ["# Internal consistency of the paper's Tables 2-4", "",
             "For any two images, MSE >= (difference of means)^2 + (difference of stds)^2, so the",
             "reported mean and variance of an enhanced image cap the PSNR it can have against the",
             "original. Cells where the reported PSNR exceeds that cap cannot be reproduced by any",
             "algorithm.", "",
             "| image | method | PSNR reported | PSNR ceiling | feasible |", "|---|---|---|---|---|"]
    for cls in IMAGES:
        for method in METHODS:
            reported, cap = paper_value("psnr", cls, method), psnr_ceiling(cls, method)
            lines.append(f"| {cls} | {method} | {reported:.3f} | {cap:.3f} | "
                         f"{'yes' if reported <= cap + 0.01 else '**no**'} |")
    (HERE / "results" / "paper_consistency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    settings = {"default": enhance.DEFAULTS}
    fitted = HERE / "params.json"
    if fitted.exists():
        settings["calibrated"] = json.loads(fitted.read_text())["params"]
    for setting, params in settings.items():
        df = run(setting, params)
        enhanced = df[df.method != "Orig"]
        print(f"\n[{setting}] mean absolute error vs paper, per metric and method")
        print(enhanced.pivot_table(index="metric", columns="method", values="abs_error")
              .loc[METRICS, METHODS].round(3).to_string())
    consistency_report()


if __name__ == "__main__":
    main()
