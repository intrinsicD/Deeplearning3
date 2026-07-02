# Visual Comparison Grid

Device: cuda

| Column | Mean PSNR | Mean MS-SSIM | Mean total sec | Config |
|---|---:|---:|---:|---|
| best_quality | 25.3951 | 0.95656 | 5.406 | `strategy=aniso_flanking|tensor=central|density=hybrid|sampling=wse|color=bilinear|scale=spacing|renderer=normalized|opacity=none|loss=l1|optimizer=adamw|lr_schedule=none|refine=none|pyramid=single` |
| fast_quality | 24.8218 | 0.95270 | 0.166 | `strategy=aniso_onedge|tensor=central|density=variance|sampling=density_random|color=bilinear|scale=spacing|renderer=normalized|opacity=none|loss=l1|optimizer=adamw|lr_schedule=none|refine=none|pyramid=single` |
| best_refine | 24.7877 | 0.94691 | 5.082 | `strategy=aniso_onedge|tensor=scharr|density=variance|sampling=wse|color=two_sided|scale=spacing|renderer=normalized|opacity=constant|loss=l1|optimizer=adam|lr_schedule=none|refine=residual_add|pyramid=single` |
| best_additive | 17.8218 | 0.77509 | 4.966 | `strategy=aniso_flanking|tensor=central|density=variance|sampling=wse|color=two_sided|scale=spacing|renderer=additive|opacity=none|loss=l1|optimizer=adamw|lr_schedule=none|refine=none|pyramid=single` |

## Rerendered Per-Image Metrics

| Image | Column | PSNR | SSIM | MS-SSIM | Gaussians |
|---|---|---:|---:|---:|---:|
| 000000186042 | best_quality | 26.8680 | 0.80862 | 0.96069 | 512 |
| 000000186042 | fast_quality | 26.0141 | 0.79008 | 0.95567 | 512 |
| 000000186042 | best_refine | 25.4323 | 0.78351 | 0.93841 | 576 |
| 000000186042 | best_additive | 19.4701 | 0.56285 | 0.81476 | 512 |
| 000000190140 | best_quality | 24.7958 | 0.75715 | 0.94420 | 512 |
| 000000190140 | fast_quality | 24.1655 | 0.74562 | 0.94028 | 512 |
| 000000190140 | best_refine | 24.4525 | 0.75307 | 0.94017 | 576 |
| 000000190140 | best_additive | 20.7965 | 0.61754 | 0.85263 | 512 |
| 000000444879 | best_quality | 23.6590 | 0.81549 | 0.96266 | 512 |
| 000000444879 | fast_quality | 23.5653 | 0.81634 | 0.96344 | 512 |
| 000000444879 | best_refine | 23.4096 | 0.80341 | 0.95343 | 576 |
| 000000444879 | best_additive | 13.0324 | 0.41277 | 0.66683 | 512 |
| 000000554838 | best_quality | 26.2573 | 0.83567 | 0.95867 | 512 |
| 000000554838 | fast_quality | 25.5419 | 0.81730 | 0.95140 | 512 |
| 000000554838 | best_refine | 25.8580 | 0.82843 | 0.95566 | 576 |
| 000000554838 | best_additive | 17.9881 | 0.53308 | 0.76616 | 512 |
