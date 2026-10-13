# Learned models on the test split [uniform degradation]

315 test images. All models were trained on the nonuniform degradation.
Scores as in `../benchmark/summary.md`; the oracle rows are bounds, not methods.

## Test-set means

| method | psnr | psnr_rgb | ssim | mse | mae | entropy |
|---|---:|---:|---:|---:|---:|---:|
| Input | 11.855 | 11.909 | 0.551 | 4906.728 | 65.161 | 6.148 |
| Auto gamma | 21.330 | 20.789 | 0.797 | 845.922 | 22.860 | 6.580 |
| SSR | 13.583 | 13.430 | 0.610 | 3781.188 | 54.954 | 6.674 |
| MSR | 13.733 | 13.570 | 0.622 | 3754.251 | 54.368 | 6.653 |
| MSRCR | 13.762 | 13.517 | 0.622 | 3753.373 | 54.270 | 6.637 |
| CLAHE | 15.767 | 15.603 | 0.645 | 2212.211 | 40.874 | 7.071 |
| Paper MSR+CLAHE | 17.947 | 17.297 | 0.767 | 1421.269 | 31.901 | 6.992 |
| Tone curve MLP | 23.430 | 22.573 | 0.818 | 464.290 | 16.912 | 6.885 |
| U-Net | 23.014 | 22.654 | 0.843 | 459.552 | 16.707 | 6.977 |
| U-Net + DIP inputs | 23.932 | 23.507 | 0.853 | 376.312 | 15.074 | 7.003 |
| Tone + U-Net | 23.896 | 23.090 | 0.822 | 377.833 | 15.076 | 7.046 |
| Tone + U-Net + DIP inputs | 23.790 | 22.990 | 0.823 | 383.694 | 15.231 | 7.042 |
| Tone + learned fusion | 23.430 | 22.573 | 0.818 | 464.289 | 16.912 | 6.885 |
| Deep U-Net | 24.111 | 23.674 | 0.857 | 368.001 | 14.825 | 6.996 |
| Deep U-Net + DIP inputs | 24.780 | 24.295 | 0.861 | 307.185 | 13.499 | 6.995 |
| Oracle gamma | 26.723 | 25.522 | 0.826 | 229.292 | 10.682 | 6.519 |
| Oracle point op | 31.675 | 29.435 | 0.866 | 76.090 | 5.479 | 6.579 |

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
| U-Net + DIP inputs | U-Net | +0.918 | [+0.724, +1.125] | 72% |
| Tone + U-Net | U-Net | +0.882 | [+0.687, +1.090] | 69% |
| Tone + U-Net + DIP inputs | Tone + U-Net | -0.106 | [-0.253, +0.034] | 46% |
| Tone + U-Net + DIP inputs | U-Net + DIP inputs | -0.142 | [-0.288, -0.003] | 44% |
| Tone + learned fusion | Tone curve MLP | +0.000 | [+0.000, +0.000] | 44% |
| Deep U-Net | U-Net + DIP inputs | +0.179 | [-0.082, +0.438] | 52% |
| Deep U-Net + DIP inputs | Deep U-Net | +0.670 | [+0.417, +0.933] | 62% |

## PSNR (dB) on the three images of the paper

| method | beach/beach10.tif | golfcourse/golfcourse44.tif | river/river95.tif |
|---|---:|---:|---:|
| Input | 14.76 | 8.78 | 13.36 |
| Auto gamma | 24.64 | 19.77 | 13.22 |
| SSR | 13.70 | 11.51 | 21.86 |
| MSR | 14.72 | 12.60 | 20.41 |
| MSRCR | 14.71 | 12.60 | 20.65 |
| CLAHE | 16.74 | 10.77 | 17.32 |
| Paper MSR+CLAHE | 18.65 | 12.44 | 24.39 |
| Tone curve MLP | 20.43 | 30.15 | 17.85 |
| U-Net | 20.70 | 28.55 | 22.96 |
| U-Net + DIP inputs | 21.26 | 30.56 | 24.64 |
| Tone + U-Net | 21.26 | 30.03 | 22.86 |
| Tone + U-Net + DIP inputs | 21.58 | 30.05 | 24.07 |
| Tone + learned fusion | 20.43 | 30.15 | 17.85 |
| Deep U-Net | 21.41 | 28.06 | 20.98 |
| Deep U-Net + DIP inputs | 21.23 | 28.64 | 26.40 |
| Oracle gamma | 24.88 | 35.10 | 28.26 |
| Oracle point op | 38.96 | 38.06 | 28.61 |
