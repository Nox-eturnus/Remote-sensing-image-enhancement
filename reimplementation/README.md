# Reimplementation: Remote Sensing Image Enhancement Based on MSR and CLAHE

Reproduction of Meng et al., WCSE 2022 (`../059.pdf`, doi:10.18178/wcse.2022.04.059) on the
UC Merced Land Use dataset (`../archive/UCMerced_LandUse`).

## Result in brief

- The three test images are identified exactly: `golfcourse44.tif`, `river95.tif`, `beach10.tif`.
  Their grayscale entropy, standard deviation and mean match the paper's "Orig." columns to three
  decimals. The paper's "variance" is a standard deviation.
- The four baselines (SSR, MSR, MSRCR, CLAHE) are reproduced closely once the parameters the paper
  omits are fitted. The CLAHE baseline is OpenCV's default (`clipLimit=2.0`, 8x8 tiles, applied
  to each RGB channel): entropy, variance and mean match to three decimals on golfcourse and river.
- The proposed method is **not** reproduced, and part of what the paper reports for it cannot be:
  - Its PSNR on golfcourse (26.68 dB) and river (19.16 dB) is above the maximum that its own
    reported mean and variance allow (19.05 dB and 15.01 dB). See `results/paper_consistency.md`.
  - Its beach row in Tables 1-3 (7.080 / 43.916 / 128.726) is exactly the output of plain CLAHE
    on that image, not of a fusion.
  - Fitted to all five tables, the method lands about 2.2 dB below the reported PSNR and 8 units
    below the reported variance on average. Fitted to Tables 1-3 only, it reaches the reported
    entropy and variance, but PSNR drops to 17.8 dB (golfcourse) and 14.6 dB (river), consistent
    with the ceiling above. See `results/ours_stats_only.md`.

## Reproduced tables (calibrated parameters)

Each cell is `reproduced (paper)`. Full tables: `results/calibrated/tables.md`.

PSNR (dB), Table 4:

| | SSR | MSR | MSRCR | CLAHE | Ours |
|---|---|---|---|---|---|
| golfcourse | 30.907 (30.708) | 32.201 (30.966) | 32.502 (31.943) | 25.188 (24.953) | 23.213 (26.676) |
| river | 21.949 (21.033) | 21.288 (21.033) | 21.096 (20.915) | 19.014 (18.782) | 16.560 (19.162) |
| beach | 18.777 (18.282) | 22.166 (22.243) | 22.226 (22.248) | 23.013 (27.161) | 24.600 (24.162) |

Mean absolute error against the paper over the three images:

| metric | SSR | MSR | MSRCR | CLAHE | Ours |
|---|---|---|---|---|---|
| entropy | 0.049 | 0.014 | 0.008 | 0.075 | 0.413 |
| variance | 1.180 | 0.269 | 0.173 | 0.619 | 8.235 |
| mean | 1.514 | 1.216 | 0.213 | 1.010 | 0.799 |
| PSNR (dB) | 0.537 | 0.522 | 0.254 | 1.539 | 2.168 |
| SSIM | 0.028 | 0.023 | 0.020 | 0.070 | 0.019 |

The CLAHE error comes almost entirely from beach, where the paper's CLAHE column does not
correspond to the CLAHE output (that output appears in the "Ours" column instead).

## What had to be assumed

The paper gives no parameter values and does not say how retinex output is mapped to 8 bits.
`calibrate.py` grid-searches those choices per method and keeps the setting closest to Tables 1-5.
This is fitting to the reported numbers, so agreement of the baselines shows that the described
algorithms *can* produce the tables, not that these were the authors' settings.

| method | fitted setting (`params.json`) |
|---|---|
| SSR | sigma 80, antilog, per-channel min-max stretch |
| MSR | sigmas 15/80/250, antilog, per-channel min-max stretch |
| MSRCR | sigmas 15/80/200, alpha 1000, antilog, per-channel min-max stretch |
| CLAHE | clip 2.0, 8x8 tiles, per RGB channel |
| Ours | CLAHE and MSR settings as above; alpha 10, colour-restored CLAHE clipped without stretch; V-channel MSR stretched at mean +- 2.5 std; lambda 0.3; bilateral d=5, sigma_r=50, sigma_d=5 |

The `default` results use conventional values instead (sigmas 15/80/250, 1 % percentile stretch,
alpha 125, lambda 0.5). They are far from the paper for the retinex methods, which shows how
much the outcome depends on the unreported stretch.

Other open points:

- PSNR and SSIM are computed on grayscale against the original. With the CLAHE output matched
  exactly, this gives 25.19 dB / 0.801 on golfcourse against the paper's 24.95 dB / 0.827. RGB,
  per-channel, HSV-V and Lab-L variants were tried and none matches exactly, so the paper's
  PSNR/SSIM definition is still unknown; expect a residual of about 0.2 dB and 0.03-0.07 SSIM.
- The paper reports identical PSNR and SSIM for SSR and MSR on river (21.033 / 0.921) although
  their means differ, which looks like a copying error.

## Files

| file | purpose |
|---|---|
| `enhance.py` | SSR, MSR, MSRCR, CLAHE, and the proposed method (Sections 2.1-2.3) |
| `metrics.py` | entropy, variance, mean, PSNR, SSIM |
| `paper.py` | the paper's Tables 1-5 and the test image paths |
| `find_images.py` | identifies the test images from the "Orig." statistics |
| `calibrate.py` | fits the unreported parameters, writes `params.json` |
| `run_reproduction.py` | writes figures, enhanced images and tables to `results/` |
| `fit_ours_stats_only.py` | side experiment: proposed method fitted to Tables 1-3 only |

## Running

Uses the conda environment `dip_rs` (Python 3.12; numpy, opencv-python, scikit-image,
matplotlib, pandas).

```
C:\miniconda3\envs\dip_rs\python.exe find_images.py
C:\miniconda3\envs\dip_rs\python.exe calibrate.py          # about 8 minutes
C:\miniconda3\envs\dip_rs\python.exe run_reproduction.py
```
