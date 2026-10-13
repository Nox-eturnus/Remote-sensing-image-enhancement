# test_ml

AI/ML extension of the paper's method, evaluated as restoration: a clean UC Merced image is
degraded synthetically, every method sees only the degraded image, and scores are computed
against the clean one. This is a different quantity from the paper's Table 4, which compares
the enhanced image with its own input; numbers here and there are not comparable.

## Result in brief

- Under this protocol the paper's fusion (16.18 dB) is beaten by a one-line blind gamma
  correction (19.84 dB), and no fusion of its two branches can reach what a global tone curve
  reaches. Classical MSR and CLAHE are not restoration methods.
- The two branches are useful as *inputs to a network*. Giving a U-Net the MSR branch and the
  CLAHE branch next to the degraded image adds about 0.5 dB, for the small U-Net (+0.47 dB) and
  for a 16 times larger one (+0.51 dB). The best model is the deep U-Net with DIP inputs,
  24.58 dB / SSIM 0.853.
- Two ideas did not work: a learned per-pixel fusion of the branches collapses onto the
  tone-corrected image and ignores them, and a tone-curve stage in front of the U-Net gives no
  gain once the DIP inputs are there.
- Every model was trained once (one seed), so differences of a few tenths of a dB between
  models are known only up to seed variation, which has not been measured.

## Protocol

- Split: 1470 / 315 / 315 train / val / test, stratified by class, in `split.json`. The three
  images of the paper (`golfcourse44`, `river95`, `beach10`) are in the test set.
- Degradation (`degrade.py`): optional blur, gamma darkening, a smooth uneven illumination
  field, contrast reduction, shot and read noise, 8-bit quantisation. Val and test images each
  get one fixed, seeded degradation; each training image gets four.
- The illumination field was added after the first baseline run, which used a spatially uniform
  degradation and showed a global tone curve solving most of it. That first protocol is kept as
  the `uniform` variant and reported next to the main `nonuniform` one. No model had been
  trained when the change was made.
- Scores (`baselines.score`): PSNR, SSIM, MSE, MAE on 8-bit grayscale as in the reproduction,
  plus PSNR on RGB and entropy. Images keep their native size (44 of 2100 are not 256x256).
- Classical methods run at the calibrated settings of `../reimplementation/params.json`.

## Results (315 test images, means of per-image values)

Models are trained on the nonuniform degradation only. Full tables, including MSE, MAE, entropy
and per-class PSNR: `results/<variant>/benchmark/summary.md`, `results/<variant>/models/summary.md`.

| method | nonuniform PSNR | SSIM | uniform PSNR | SSIM |
|---|---:|---:|---:|---:|
| Input (degraded) | 10.08 | 0.439 | 11.86 | 0.551 |
| SSR | 12.96 | 0.580 | 13.58 | 0.610 |
| MSR | 13.25 | 0.595 | 13.73 | 0.622 |
| MSRCR | 13.28 | 0.594 | 13.76 | 0.622 |
| CLAHE | 13.86 | 0.611 | 15.77 | 0.645 |
| Paper MSR+CLAHE | 16.18 | 0.744 | 17.95 | 0.767 |
| Auto gamma (mean grey to mid-range) | 19.84 | 0.758 | 21.33 | 0.797 |
| Tone curve MLP | 22.55 | 0.796 | 23.43 | 0.818 |
| Tone + learned fusion | 22.55 | 0.796 | 23.43 | 0.818 |
| U-Net | 23.17 | 0.835 | 23.01 | 0.843 |
| Tone + U-Net + DIP inputs | 23.50 | 0.805 | 23.79 | 0.823 |
| Tone + U-Net | 23.51 | 0.804 | 23.90 | 0.822 |
| U-Net + DIP inputs | 23.64 | 0.844 | 23.93 | 0.853 |
| Deep U-Net | 24.07 | 0.849 | 24.11 | 0.857 |
| **Deep U-Net + DIP inputs** | **24.58** | **0.853** | **24.78** | **0.861** |
| *Oracle gamma (best gamma per image)* | *23.50* | *0.781* | *26.72* | *0.826* |
| *Oracle point operation (best tone curve per image)* | *27.47* | *0.829* | *31.68* | *0.866* |

Oracle rows use the clean image; they are bounds, not methods.

Paired differences on the nonuniform test set (mean per-image PSNR difference, bootstrap 95 %
interval over images; the interval does not include training-seed variation):

| comparison | difference (dB) | 95 % interval |
|---|---:|---:|
| U-Net + DIP inputs vs U-Net | +0.47 | [+0.30, +0.65] |
| Deep U-Net + DIP inputs vs Deep U-Net | +0.51 | [+0.28, +0.74] |
| Deep U-Net vs U-Net + DIP inputs | +0.43 | [+0.20, +0.66] |
| Tone + U-Net vs U-Net | +0.34 | [+0.18, +0.52] |
| Tone + U-Net + DIP inputs vs Tone + U-Net | -0.01 | [-0.12, +0.10] |
| Tone + U-Net + DIP inputs vs U-Net + DIP inputs | -0.14 | [-0.25, -0.03] |

Notes on reading these:

- Results vary a lot from image to image. A model that is better on average wins on only about
  60 % of the test images, and the ranking on the paper's three images differs from the average.
- All networks stay below the oracle tone curve in PSNR while matching or exceeding it in SSIM.
- The deep models were still improving slowly on the val split when training stopped at 15,000
  iterations, so their numbers are for this training budget, not converged values.

### Fusion of the two branches

`results/<variant>/fusion_headroom.md`. On the nonuniform test set the paper's fusion scores
16.18 dB at its calibrated lambda of 0.3 and 18.99 dB at the best fixed lambda (1.0, the MSR
branch alone). A perfect per-image lambda would reach 20.18 dB and a perfect per-pixel weight
map over {input, G, H} 21.78 dB, both below what the tone curve MLP achieves blindly.

The learned fusion (`tone_fusion`: per-pixel weights over the tone-corrected image, G and H)
was trained twice, from a random start and from a start close to the tone-corrected image. Both
runs end with all weight on the tone-corrected image and reproduce the tone model's output
(`training/tone_fusion_random_init.csv`, `training/tone_fusion.csv`). On the val split, mixing
any fixed amount of G or H into the tone output gains at most 0.015 dB.

## Models (`models.py`)

| name | what it is | parameters |
|---|---|---:|
| `tone` | MLP reads the per-channel cumulative histogram and outputs a monotone tone curve per channel | 121,523 |
| `unet` | 32-64-128 U-Net, adds a residual to the degraded image | 472,387 |
| `unet_dip` | the same, with G (HSV-MSR) and H (colour-restored CLAHE) as extra input channels | 474,115 |
| `tone_unet` | fixed `tone` stage, then a U-Net residual on its output | 594,774 |
| `tone_unet_dip` | `tone_unet` with G and H as extra input channels | 596,502 |
| `tone_fusion` | fixed `tone` stage, then a U-Net that outputs per-pixel fusion weights | 596,502 |
| `unet_deep` | 32-64-128-256-512 U-Net on the degraded image | 7,847,491 |
| `unet_deep_dip` | the same, with G and H as extra input channels | 7,849,219 |

Training (`train.py`): 15,000 iterations (4,000 for `tone`), batch 16, random 192x192 crops with
flips and transposition, Charbonnier loss on RGB, AdamW with cosine decay from 2e-4, mixed
precision; the checkpoint with the best val PSNR is kept. One seed per model.

## Files

| file | purpose |
|---|---|
| `data.py` | file listing, split, image loading |
| `degrade.py` | synthetic degradation, both variants |
| `baselines.py` | auto gamma, oracles, the paper's fusion split into parts, scores |
| `run_benchmark.py` | all non-learned methods on the test split |
| `fusion_headroom.py` | fixed, per-image and per-pixel fusion ceilings |
| `cache.py` | precomputes degraded images and the two branches for training and evaluation |
| `models.py` | tone curve, U-Net and their combinations |
| `train.py` | trains one model |
| `evaluate.py` | scores all trained models on both test sets |
| `checkpoints/` | trained weights (71 MB) |

## Running

Conda environment `dip_rs`, as for the reproduction, plus PyTorch with CUDA
(`pip install torch --index-url https://download.pytorch.org/whl/cu126`; 2.14.1 was used).
`cache.py` writes about 4 GB to `~/.cache/dip_rs` (or `$DIP_RS_CACHE`), outside OneDrive.

```
C:\miniconda3\envs\dip_rs\python.exe run_benchmark.py --variant nonuniform    # about 2.5 minutes
C:\miniconda3\envs\dip_rs\python.exe fusion_headroom.py --variant nonuniform  # about 3.5 minutes
C:\miniconda3\envs\dip_rs\python.exe cache.py                                 # about 6 minutes
C:\miniconda3\envs\dip_rs\python.exe train.py tone                            # 1 minute; needed by tone_* models
C:\miniconda3\envs\dip_rs\python.exe train.py unet_deep_dip                   # 20-35 minutes per network on an RTX 4070 laptop
C:\miniconda3\envs\dip_rs\python.exe evaluate.py
```

The first two also accept `--variant uniform`.
