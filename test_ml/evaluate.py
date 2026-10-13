"""Score every trained model in checkpoints/ on the test split, for both
degradation variants, next to the non-learned methods of run_benchmark.py.

Writes, under results/<variant>/models/:

    metrics.csv       one row per image and model
    summary.md        test-set means of all methods, paired differences
                      between models, the three paper images
    fig_<class>.png   comparison panel for the three paper images
"""
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import baselines as B
from data import HERE, PAPER_IMAGES
from models import MODELS
from run_benchmark import BLIND, COLUMNS, ORACLES, md_table
from train import CKPT, Samples, load_model, predict

TEST_SETS = {"nonuniform": "test", "uniform": "test_uniform"}
LABELS = {"tone": "Tone curve MLP", "unet": "U-Net", "unet_dip": "U-Net + DIP inputs",
          "tone_unet": "Tone + U-Net", "tone_unet_dip": "Tone + U-Net + DIP inputs",
          "tone_fusion": "Tone + learned fusion", "unet_deep": "Deep U-Net",
          "unet_deep_dip": "Deep U-Net + DIP inputs"}
PAIRS = [("unet_dip", "unet"), ("tone_unet", "unet"), ("tone_unet_dip", "tone_unet"),
         ("tone_unet_dip", "unet_dip"), ("tone_fusion", "tone"), ("unet_deep", "unet_dip"),
         ("unet_deep_dip", "unet_deep")]


def label(name):
    base, _, seed = name.partition("_s")
    return LABELS[base] + (f" (seed {seed})" if seed else "")


def checkpoints():
    """Checkpoint names (<model> or <model>_s<seed>) in the order of MODELS."""
    return sorted((p.stem for p in CKPT.glob("*.pt")),
                  key=lambda n: (list(MODELS).index(n.partition("_s")[0]), n))


def paired(df, a, b, rng):
    """Mean per-image PSNR difference a - b with a bootstrap 95 % interval over test images."""
    wide = df.pivot_table(index="image", columns="method", values="psnr")
    d = (wide[label(a)] - wide[label(b)]).to_numpy()
    means = rng.choice(d, (5000, len(d))).mean(1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), (d > 0).mean()


def figure(key, samples, outs, variant, out_dir):
    i = samples.keys.index(key)
    gt, lq = samples.clean(i), samples.lq[i][:samples.size(i)[0], :samples.size(i)[1]].numpy()
    panel = {"Clean": gt, "Input": lq, "Paper MSR+CLAHE": B.classical(lq)["Ours"],
             "Auto gamma": B.auto_gamma(lq), **{label(n): o[i] for n, o in outs.items() if "_s" not in n}}
    cols = (len(panel) + 1) // 2
    fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 7))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, im) in zip(axes.ravel(), panel.items()):
        title = name if name == "Clean" else f"{name}\n{B.score(im, gt)['psnr']:.2f} dB"
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"{key} [{variant} degradation]", fontsize=11)
    fig.tight_layout(h_pad=2.5)
    fig.savefig(out_dir / f"fig_{key.split('/')[0]}.png", dpi=130)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = checkpoints()
    loaded = {n: load_model(n, device) for n in names}
    for variant, set_name in TEST_SETS.items():
        out_dir = HERE / "results" / variant / "models"
        out_dir.mkdir(parents=True, exist_ok=True)
        samples = Samples(set_name)
        outs = {n: predict(model, samples, device) for n, (model, _) in loaded.items()}
        rows = [{"image": key, "method": label(n), **B.score(out[i], samples.clean(i))}
                for n, out in outs.items() for i, key in enumerate(samples.keys)]
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "metrics.csv", index=False, float_format="%.4f")

        bench = pd.read_csv(HERE / "results" / variant / "benchmark" / "metrics.csv")
        both = pd.concat([bench, df])
        order = BLIND + [label(n) for n in names] + ORACLES
        overall = both.groupby("method")[COLUMNS].mean().loc[order]
        overall.index.name = "method"
        paper = both[both.image.isin(PAPER_IMAGES)].pivot_table(index="method", columns="image",
                                                                values="psnr").loc[order]

        lines = [f"# Learned models on the test split [{variant} degradation]", "",
                 f"{len(samples.keys)} test images. All models were trained on the nonuniform degradation.",
                 "Scores as in `../benchmark/summary.md`; the oracle rows are bounds, not methods.", "",
                 "## Test-set means", ""] + md_table(overall, ".3f")
        lines += ["", "## Models", "", "| model | parameters | best val PSNR (dB) | at iteration |", "|---|---:|---:|---:|"]
        for n, (model, ckpt) in loaded.items():
            lines.append(f"| {label(n)} | {sum(p.numel() for p in model.parameters()):,} | "
                         f"{ckpt['val_psnr']:.3f} | {ckpt['iteration']} |")
        lines += ["", "## Paired PSNR differences (dB)", "",
                  "Mean over test images of the per-image difference, with a bootstrap 95 % interval",
                  "over images. The interval reflects the test sample, not training-seed variation.", "",
                  "| A | B | A - B | 95 % interval | A better on |", "|---|---|---:|---:|---:|"]
        rng = np.random.default_rng(0)
        for a, b in PAIRS:
            if a in names and b in names:
                mean, lo, hi, frac = paired(df, a, b, rng)
                lines.append(f"| {label(a)} | {label(b)} | {mean:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {frac:.0%} |")
        lines += ["", "## PSNR (dB) on the three images of the paper", ""] + md_table(paper, ".2f")
        (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n[{variant}]")
        print(overall.round(3).to_string())

        for key in PAPER_IMAGES:
            figure(key, samples, outs, variant, out_dir)


if __name__ == "__main__":
    main()
