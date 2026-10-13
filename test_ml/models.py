"""Learned models.

Every model maps (lq, g, h, cdf) to the restored image: the low-quality image
and the paper's two branches as BGR tensors in [0, 1] of shape (N, 3, H, W),
and the per-channel cumulative histogram of the whole low-quality image,
(N, 3, 256). A model is a combination of three optional parts:

    tone    ToneCurve: global tone correction predicted from the histogram
    dip     the two branches are given to the network as extra input channels
    head    "residual": a U-Net adds a correction to the (tone-corrected) image
            "fusion":   a U-Net outputs per-pixel weights and the result is a
                        convex combination of the tone-corrected image, G and H
            "none":     tone correction only
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS = {
    "tone": dict(tone=True, head="none"),
    "unet": dict(),
    "unet_dip": dict(dip=True),
    "tone_unet": dict(tone=True),
    "tone_unet_dip": dict(tone=True, dip=True),
    "tone_fusion": dict(tone=True, dip=True, head="fusion"),
    "unet_deep": dict(depth=4),
    "unet_deep_dip": dict(depth=4, dip=True),
}


class ToneCurve(nn.Module):
    """An MLP reads the cumulative histogram (the quantity histogram
    equalisation is built on) and outputs one monotone piecewise-linear tone
    curve per channel. Knots are denser in the shadows, where a darkened
    image has its values."""

    def __init__(self, knots=16, hidden=128):
        super().__init__()
        self.register_buffer("x", (torch.arange(knots + 1) / knots) ** 2)
        self.mlp = nn.Sequential(nn.Linear(3 * 256, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, 3 * (knots + 1)))
        nn.init.zeros_(self.mlp[-1].weight)  # start at the identity curve
        nn.init.zeros_(self.mlp[-1].bias)

    def curve(self, cdf):
        """Knot values (N, 3, knots + 1): an offset, then segments whose
        slopes are exp(.) of the MLP output, hence positive."""
        raw = self.mlp(cdf.flatten(1)).view(cdf.shape[0], 3, -1)
        steps = (self.x[1:] - self.x[:-1]) * torch.exp(raw[..., 1:])
        return torch.cat([raw[..., :1], steps], -1).cumsum(-1)

    def forward(self, lq, cdf):
        y = self.curve(cdf.float())
        lq = lq.float()
        last = len(self.x) - 2
        idx = (lq.clamp(0, 1).sqrt() * (last + 1)).long().clamp(max=last)
        flat = idx.flatten(2)
        y0, y1 = y.gather(2, flat).view_as(lq), y.gather(2, flat + 1).view_as(lq)
        x0, x1 = self.x[idx], self.x[idx + 1]
        return y0 + (lq - x0) / (x1 - x0) * (y1 - y0)


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Two 3x3 convolutions per level, max-pool down, bilinear up, skip
    connections. depth=2 and width=32 give the 32-64-128 network."""

    def __init__(self, cin, cout=3, width=32, depth=2):
        super().__init__()
        widths = [width * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList(_block(a, b) for a, b in zip([cin] + widths[:-1], widths))
        self.up = nn.ModuleList(_block(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(depth)))
        self.out = nn.Conv2d(width, cout, 3, padding=1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        skips.pop()
        for block in self.up:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skips.pop()], 1))
        return self.out(x)


class Restorer(nn.Module):
    def __init__(self, tone=False, dip=False, head="residual", depth=2, width=32):
        super().__init__()
        self.dip, self.head, self.multiple = dip, head, 2 ** depth
        self.tone = ToneCurve() if tone else None
        self.net = None if head == "none" else UNet(3 * (1 + tone + 2 * dip), 3, width, depth)
        if head == "residual":
            nn.init.zeros_(self.net.out.weight)  # start at the uncorrected image
            nn.init.zeros_(self.net.out.bias)
        if head == "fusion":
            # start close to the tone-corrected image with the softmax unsaturated;
            # from a random start the weights saturated on it and stopped learning
            nn.init.zeros_(self.net.out.weight)
            self.net.out.bias.data.copy_(torch.tensor([3.0, 0.0, 0.0]))

    def forward(self, lq, g, h, cdf):
        if self.tone is None:
            base, inputs = lq, [lq]
        else:
            with torch.autocast(lq.device.type, enabled=False):
                base = self.tone(lq, cdf)
            inputs = [base, lq]
        if self.net is None:
            return base
        y = self.net(torch.cat(inputs + ([g, h] if self.dip else []), 1)).float()
        if self.head == "fusion":
            w = y.softmax(1)
            return w[:, 0:1] * base.clamp(0, 1) + w[:, 1:2] * g + w[:, 2:3] * h
        return base + y


def build(name):
    return Restorer(**MODELS[name])
