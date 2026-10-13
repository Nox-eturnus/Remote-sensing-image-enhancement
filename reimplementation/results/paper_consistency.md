# Internal consistency of the paper's Tables 2-4

For any two images, MSE >= (difference of means)^2 + (difference of stds)^2, so the
reported mean and variance of an enhanced image cap the PSNR it can have against the
original. Cells where the reported PSNR exceeds that cap cannot be reproduced by any
algorithm.

| image | method | PSNR reported | PSNR ceiling | feasible |
|---|---|---|---|---|
| golfcourse | SSR | 30.708 | 32.987 | yes |
| golfcourse | MSR | 30.966 | 33.912 | yes |
| golfcourse | MSRCR | 31.943 | 33.686 | yes |
| golfcourse | CLAHE | 24.953 | 26.754 | yes |
| golfcourse | Ours | 26.676 | 19.051 | **no** |
| river | SSR | 21.033 | 22.760 | yes |
| river | MSR | 21.033 | 21.383 | yes |
| river | MSRCR | 20.915 | 21.248 | yes |
| river | CLAHE | 18.782 | 19.165 | yes |
| river | Ours | 19.162 | 15.005 | **no** |
| beach | SSR | 18.282 | 19.237 | yes |
| beach | MSR | 22.243 | 23.361 | yes |
| beach | MSRCR | 22.248 | 23.380 | yes |
| beach | CLAHE | 27.161 | 44.437 | yes |
| beach | Ours | 24.162 | 35.630 | yes |
