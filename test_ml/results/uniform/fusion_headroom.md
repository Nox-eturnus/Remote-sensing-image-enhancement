# Headroom of fusing the paper's two branches

Mean grayscale PSNR (dB) against the clean image, 315 test images.
The fixed lambda is chosen on the 315 val images.

| | calibrated branches | conventional branches |
|---|---:|---:|
| paper lambda | 17.95 (lambda 0.3) | 15.39 (lambda 0.5) |
| best fixed lambda | 19.52 (lambda 0.90) | 16.07 (lambda 0.00) |
| oracle image lambda | 21.10 | 16.46 |
| oracle pixel {G,H} | 22.28 | 16.82 |
| oracle pixel {input,G,H} | 23.16 | 18.06 |
| oracle pixel {auto gamma,G,H} | 27.72 | 27.80 |

Mean PSNR over the test split for each fixed lambda:

| lambda | 0.00 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibrated | 16.30 | 16.85 | 17.40 | 17.95 | 18.43 | 18.85 | 19.19 | 19.41 | 19.51 | 19.52 | 19.45 |
| conventional | 16.07 | 15.98 | 15.86 | 15.72 | 15.56 | 15.39 | 15.22 | 15.03 | 14.84 | 14.64 | 14.44 |
