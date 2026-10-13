# Learned models on the test split [nonuniform degradation]

315 test images. All models were trained on the nonuniform degradation.
Scores as in `../benchmark/summary.md`; the oracle rows are bounds, not methods.

## Test-set means

| method | psnr | psnr_rgb | ssim | mse | mae | entropy |
|---|---:|---:|---:|---:|---:|---:|
| Input | 10.083 | 10.175 | 0.439 | 6978.134 | 78.158 | 5.932 |
| Auto gamma | 19.840 | 19.382 | 0.758 | 1021.564 | 25.179 | 6.539 |
| SSR | 12.964 | 12.854 | 0.580 | 4241.682 | 58.109 | 6.665 |
| MSR | 13.250 | 13.096 | 0.595 | 4095.732 | 56.627 | 6.656 |
| MSRCR | 13.278 | 13.047 | 0.594 | 4092.445 | 56.515 | 6.639 |
| CLAHE | 13.861 | 13.822 | 0.611 | 3274.108 | 51.209 | 6.885 |
| Paper MSR+CLAHE | 16.180 | 15.771 | 0.744 | 2068.517 | 39.323 | 6.890 |
| Tone curve MLP | 22.545 | 21.865 | 0.796 | 491.038 | 17.276 | 6.884 |
| U-Net | 23.169 | 22.835 | 0.835 | 417.140 | 15.783 | 6.937 |
| U-Net + DIP inputs | 23.644 | 23.297 | 0.844 | 376.066 | 14.921 | 6.976 |
| Tone + U-Net | 23.512 | 22.735 | 0.804 | 384.563 | 15.133 | 7.025 |
| Tone + U-Net + DIP inputs | 23.501 | 22.723 | 0.805 | 381.813 | 15.095 | 7.018 |
| Tone + learned fusion | 22.545 | 21.865 | 0.796 | 491.038 | 17.276 | 6.884 |
| Deep U-Net | 24.072 | 23.701 | 0.849 | 347.721 | 14.280 | 6.952 |
| Deep U-Net + DIP inputs | 24.581 | 24.171 | 0.853 | 306.164 | 13.315 | 6.960 |
| Oracle gamma | 23.495 | 22.730 | 0.781 | 405.059 | 14.806 | 6.477 |
| Oracle point op | 27.473 | 26.273 | 0.829 | 153.661 | 8.751 | 6.477 |

## Models

| model | parameters | best val PSNR (dB) | at iteration |
|---|---:|---:|---:|
| Tone curve MLP | 121,523 | 22.783 | 4000 |
| U-Net | 472,387 | 23.306 | 15000 |
| U-Net + DIP inputs | 474,115 | 23.687 | 15000 |
| Tone + U-Net | 594,774 | 23.673 | 15000 |
| Tone + U-Net + DIP inputs | 596,502 | 23.655 | 15000 |
| Tone + learned fusion | 596,502 | 22.783 | 1000 |
| Deep U-Net | 7,847,491 | 24.063 | 15000 |
| Deep U-Net + DIP inputs | 7,849,219 | 24.293 | 15000 |

## Paired PSNR differences (dB)

Mean over test images of the per-image difference, with a bootstrap 95 % interval
over images. The interval reflects the test sample, not training-seed variation.

| A | B | A - B | 95 % interval | A better on |
|---|---|---:|---:|---:|
| U-Net + DIP inputs | U-Net | +0.475 | [+0.304, +0.651] | 59% |
| Tone + U-Net | U-Net | +0.344 | [+0.183, +0.517] | 57% |
| Tone + U-Net + DIP inputs | Tone + U-Net | -0.012 | [-0.124, +0.100] | 51% |
| Tone + U-Net + DIP inputs | U-Net + DIP inputs | -0.143 | [-0.254, -0.030] | 43% |
| Tone + learned fusion | Tone curve MLP | +0.000 | [-0.000, +0.000] | 37% |
| Deep U-Net | U-Net + DIP inputs | +0.428 | [+0.199, +0.656] | 63% |
| Deep U-Net + DIP inputs | Deep U-Net | +0.509 | [+0.283, +0.741] | 60% |

## PSNR (dB) on the three images of the paper

| method | beach/beach10.tif | golfcourse/golfcourse44.tif | river/river95.tif |
|---|---:|---:|---:|
| Input | 9.33 | 7.07 | 12.75 |
| Auto gamma | 19.16 | 18.87 | 13.32 |
| SSR | 11.98 | 10.78 | 20.80 |
| MSR | 12.96 | 11.69 | 20.59 |
| MSRCR | 13.03 | 11.65 | 21.24 |
| CLAHE | 11.44 | 8.76 | 16.15 |
| Paper MSR+CLAHE | 13.57 | 10.57 | 22.63 |
| Tone curve MLP | 20.57 | 21.32 | 20.17 |
| U-Net | 23.52 | 23.60 | 24.61 |
| U-Net + DIP inputs | 21.76 | 22.20 | 25.39 |
| Tone + U-Net | 20.81 | 23.46 | 23.79 |
| Tone + U-Net + DIP inputs | 20.67 | 23.08 | 25.27 |
| Tone + learned fusion | 20.57 | 21.32 | 20.17 |
| Deep U-Net | 20.29 | 24.61 | 21.97 |
| Deep U-Net + DIP inputs | 20.26 | 26.84 | 25.56 |
| Oracle gamma | 19.29 | 26.13 | 26.82 |
| Oracle point op | 22.03 | 27.21 | 27.31 |
