# Headroom of fusing the paper's two branches [nonuniform degradation]

Mean grayscale PSNR (dB) against the clean image, 315 test images.
The fixed lambda is chosen on the 315 val images.

| | calibrated branches | conventional branches |
|---|---:|---:|
| paper lambda | 16.18 (lambda 0.3) | 14.76 (lambda 0.5) |
| best fixed lambda | 18.99 (lambda 1.00) | 15.44 (lambda 0.00) |
| oracle image lambda | 20.18 | 15.70 |
| oracle pixel {G,H} | 21.39 | 16.01 |
| oracle pixel {input,G,H} | 21.78 | 16.86 |
| oracle pixel {auto gamma,G,H} | 26.26 | 25.59 |

Mean PSNR over the test split for each fixed lambda:

| lambda | 0.00 | 0.10 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| calibrated | 14.27 | 14.89 | 15.53 | 16.18 | 16.83 | 17.43 | 17.96 | 18.41 | 18.73 | 18.91 | 18.99 |
| conventional | 15.44 | 15.33 | 15.20 | 15.07 | 14.92 | 14.76 | 14.60 | 14.43 | 14.26 | 14.09 | 13.91 |
