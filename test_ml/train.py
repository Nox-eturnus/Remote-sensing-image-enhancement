"""Train one model of models.MODELS on the cached training set (cache.py).

    python train.py tone             # first: the other tone_* models load it
    python train.py tone_unet_dip

Random 192x192 crops with flips and transposition, batch 16, Charbonnier loss
on RGB, AdamW with cosine decay, mixed precision. The checkpoint with the best
grayscale PSNR on the val split is kept in checkpoints/<name>.pt and the
training curve is written to results/nonuniform/training/<name>.csv. In models
with a tone stage and a network, the tone stage is the trained `tone` model,
kept fixed.
"""
import argparse
import csv
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from cache import load_set
from data import HERE
from models import MODELS, build

CKPT = HERE / "checkpoints"
CROP, BATCH = 192, 16
PARTS = ("lq", "g", "h")


def to_float(x):
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float in [0, 1]."""
    return x.permute(0, 3, 1, 2).float() / 255.0


def name_of(model, seed):
    return model if seed == 0 else f"{model}_s{seed}"


class Samples:
    """A cached set held in memory as uint8 tensors."""

    def __init__(self, name):
        s = load_set(name)
        self.keys, self.draws, self.shape = s["keys"], s["draws"], s["shape"]
        self.gt, self.lq, self.g, self.h = (torch.from_numpy(s[p]) for p in ("gt",) + PARTS)
        cdf = np.empty((len(self), 3, 256), np.float32)
        for i, img in enumerate(s["lq"]):  # cumulative histogram of the whole image
            h, w = self.size(i)
            for c in range(3):
                cdf[i, c] = np.cumsum(np.bincount(img[:h, :w, c].ravel(), minlength=256)) / (h * w)
        self.cdf = torch.from_numpy(cdf)

    def __len__(self):
        return len(self.lq)

    def size(self, i):
        return tuple(int(v) for v in self.shape[i // self.draws])

    def clean(self, i):
        h, w = self.size(i)
        return self.gt[i // self.draws][:h, :w].numpy()

    def batch(self, rng, device):
        idx = rng.integers(len(self), size=BATCH)
        crops = {p: [] for p in PARTS + ("gt",)}
        for i in idx:
            h, w = self.size(i)
            top, left = rng.integers(h - CROP + 1), rng.integers(w - CROP + 1)
            for p in PARTS:
                crops[p].append(getattr(self, p)[i][top:top + CROP, left:left + CROP])
            crops["gt"].append(self.gt[i // self.draws][top:top + CROP, left:left + CROP])
        out = [to_float(torch.stack(crops[p]).to(device)) for p in PARTS + ("gt",)]
        for dims in ((2,), (3,)):
            if rng.random() < 0.5:
                out = [x.flip(dims) for x in out]
        if rng.random() < 0.5:
            out = [x.transpose(2, 3) for x in out]
        return out[:3] + [self.cdf[idx].to(device)], out[3]


@torch.no_grad()
def predict(model, samples, device, batch=32):
    """Restored 8-bit BGR image of every sample, at its native size."""
    model.eval()
    outs = [None] * len(samples)
    sizes = np.array([samples.size(i) for i in range(len(samples))])
    for h, w in np.unique(sizes, axis=0):
        members = np.flatnonzero((sizes == (h, w)).all(1))
        pad = (0, (-w) % model.multiple, 0, (-h) % model.multiple)
        for start in range(0, len(members), batch):
            idx = members[start:start + batch]
            x = [F.pad(to_float(getattr(samples, p)[idx][:, :h, :w].to(device)), pad, mode="reflect")
                 for p in PARTS]
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                y = model(*x, samples.cdf[idx].to(device))
            y = (y[..., :h, :w].float().clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1)
            for i, out in zip(idx, y.cpu().numpy()):
                outs[i] = out
    return outs


def gray_psnr(outs, samples):
    """Mean grayscale PSNR against the clean images (as baselines.score, faster)."""
    values = []
    for i, out in enumerate(outs):
        diff = (cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float64)
                - cv2.cvtColor(samples.clean(i), cv2.COLOR_BGR2GRAY))
        values.append(10 * np.log10(255.0 ** 2 / max((diff ** 2).mean(), 1e-12)))
    return float(np.mean(values))


def load_model(name, device):
    """A trained model from checkpoints/, in eval mode."""
    ckpt = torch.load(CKPT / f"{name}.pt", map_location=device)
    model = build(ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["state"])
    return model.eval(), ckpt


def train(model_name, seed, iters, lr, eval_every):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.backends.cudnn.benchmark = True
    train_set, val_set = Samples("train"), Samples("val")

    model = build(model_name).to(device)
    if model.tone is not None and model.net is not None:
        model.tone.load_state_dict(load_model("tone", device)[0].tone.state_dict())
        model.tone.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    name = name_of(model_name, seed)
    log_dir = HERE / "results" / "nonuniform" / "training"
    log_dir.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    best, running, start = -np.inf, 0.0, time.time()
    print(f"{name}: {sum(p.numel() for p in params):,} trainable parameters, {iters} iterations")
    with open(log_dir / f"{name}.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["iteration", "train_loss", "val_psnr", "seconds"])
        for it in range(1, iters + 1):
            model.train()
            inputs, gt = train_set.batch(rng, device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(*inputs)
            loss = torch.sqrt((out.float() - gt) ** 2 + 1e-6).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if it % eval_every == 0 or it == iters:
                psnr = gray_psnr(predict(model, val_set, device), val_set)
                n = eval_every if it % eval_every == 0 else it % eval_every
                log.writerow([it, f"{running / n:.5f}", f"{psnr:.4f}", f"{time.time() - start:.0f}"])
                f.flush()
                print(f"  {it:6d}  loss {running / n:.4f}  val {psnr:.3f} dB  {time.time() - start:.0f} s",
                      flush=True)
                running = 0.0
                if psnr > best:
                    best = psnr
                    torch.save({"state": model.state_dict(), "model_name": model_name, "seed": seed,
                                "iteration": it, "val_psnr": psnr}, CKPT / f"{name}.pt")
    print(f"{name}: best val {best:.3f} dB")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--lr", type=float)
    args = parser.parse_args()
    tone_only = MODELS[args.model].get("head") == "none"
    iters = args.iters or (4000 if tone_only else 20000)
    train(args.model, args.seed, iters, args.lr or (1e-3 if tone_only else 2e-4), 500 if tone_only else 1000)


if __name__ == "__main__":
    main()
