# Proposed method fitted to Tables 1-3 only (golfcourse, river)

Parameters: `{"clip": 2.0, "tile": 8, "sigmas": [15, 80, 250], "alpha": 10.0, "h_stretch": ["pct", 2.0], "v_stretch": ["minmax", 0.0], "antilog": true, "lam": 0.2, "bilateral": null}`

Mean scaled error on the fitted cells: 0.384

Each cell is `reproduced (paper)`.

| image | entropy | variance | mean | psnr | ssim |
|---|---|---|---|---|---|
| golfcourse | 7.400 (7.511) | 42.833 (44.736) | 133.633 (143.502) | 17.838 (26.676) | 0.591 (0.886) |
| river | 7.420 (7.411) | 50.814 (48.311) | 102.776 (103.132) | 14.613 (19.162) | 0.594 (0.753) |
