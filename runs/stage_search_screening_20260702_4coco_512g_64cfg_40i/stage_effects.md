# Stage Search Analysis

Rows: 256; configs: 64; images/config: [4]
Screening setup: 4 COCO val images, 512 Gaussians, 40 fit iterations, max side 160, seed 0, 64 shuffled configs.

## Top Configs by Mean PSNR

| Rank | Mean PSNR | Std | Mean MS-SSIM | Mean fit sec | Mean total sec | Images | Config |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 25.3951 | 1.2535 | 0.95656 | 0.169 | 5.406 | 4 | `strategy=aniso_flanking|tensor=central|density=hybrid|sampling=wse|color=bilinear|scale=spacing|renderer=normalized|opacity=none|loss=l1|optimizer=adamw|lr_schedule=none|refine=none|pyramid=single` |
| 2 | 24.8218 | 0.9936 | 0.95270 | 0.163 | 0.166 | 4 | `strategy=aniso_onedge|tensor=central|density=variance|sampling=density_random|color=bilinear|scale=spacing|renderer=normalized|opacity=none|loss=l1|optimizer=adamw|lr_schedule=none|refine=none|pyramid=single` |
| 3 | 24.7877 | 0.9454 | 0.94691 | 0.197 | 5.082 | 4 | `strategy=aniso_onedge|tensor=scharr|density=variance|sampling=wse|color=two_sided|scale=spacing|renderer=normalized|opacity=constant|loss=l1|optimizer=adam|lr_schedule=none|refine=residual_add|pyramid=single` |
| 4 | 24.5855 | 0.9401 | 0.94455 | 0.193 | 5.428 | 4 | `strategy=aniso_onedge|tensor=central|density=structure|sampling=wse|color=bilinear|scale=spacing|renderer=normalized|opacity=constant|loss=charbonnier|optimizer=adamw|lr_schedule=none|refine=residual_add|pyramid=single` |
| 5 | 23.5360 | 1.1949 | 0.93138 | 0.307 | 5.749 | 4 | `strategy=aniso_flanking|tensor=central|density=structure|sampling=wse|color=bilinear|scale=spacing|renderer=normalized|opacity=none|loss=charbonnier|optimizer=adam|lr_schedule=none|refine=residual_add|pyramid=single` |
| 6 | 22.8261 | 1.3924 | 0.91992 | 0.189 | 5.399 | 4 | `strategy=aniso_flanking|tensor=scharr|density=hybrid|sampling=wse|color=two_sided|scale=spacing|renderer=normalized|opacity=none|loss=l1|optimizer=adam|lr_schedule=cosine|refine=residual_add|pyramid=single` |
| 7 | 22.3338 | 1.4951 | 0.91315 | 0.190 | 0.193 | 4 | `strategy=aniso_flanking|tensor=central|density=variance|sampling=density_random|color=local_mean|scale=spacing|renderer=normalized|opacity=none|loss=charbonnier|optimizer=adam|lr_schedule=cosine|refine=residual_add|pyramid=single` |
| 8 | 21.9843 | 1.7141 | 0.90495 | 0.265 | 0.265 | 4 | `strategy=aniso_flanking|tensor=scharr|density=structure|sampling=density_random|color=local_mean|scale=spacing|renderer=normalized|opacity=constant|loss=charbonnier|optimizer=adamw|lr_schedule=cosine|refine=none|pyramid=pyramid` |
| 9 | 21.6705 | 1.8530 | 0.90066 | 0.194 | 0.198 | 4 | `strategy=iso_blue_noise|tensor=central|density=hybrid|sampling=density_random|color=local_mean|scale=spacing|renderer=normalized|opacity=constant|loss=charbonnier|optimizer=adam|lr_schedule=none|refine=prune_residual_add|pyramid=single` |
| 10 | 21.4913 | 1.5708 | 0.90502 | 0.199 | 0.202 | 4 | `strategy=aniso_flanking|tensor=scharr|density=variance|sampling=jittered_grid|color=two_sided|scale=spacing|renderer=normalized|opacity=constant|loss=charbonnier|optimizer=adamw|lr_schedule=none|refine=residual_add|pyramid=single` |
| 11 | 21.4890 | 1.4577 | 0.90518 | 0.194 | 0.196 | 4 | `strategy=aniso_onedge|tensor=central|density=structure|sampling=jittered_grid|color=bilinear|scale=spacing|renderer=normalized|opacity=none|loss=l1|optimizer=adamw|lr_schedule=none|refine=prune_residual_add|pyramid=single` |
| 12 | 21.4112 | 1.3268 | 0.88923 | 4.605 | 4.605 | 4 | `strategy=aniso_onedge|tensor=scharr|density=structure|sampling=wse|color=two_sided|scale=spacing|renderer=normalized|opacity=none|loss=l1|optimizer=adam|lr_schedule=cosine|refine=residual_add|pyramid=pyramid` |
| 13 | 21.3128 | 2.0283 | 0.88682 | 2.214 | 2.214 | 4 | `strategy=iso_blue_noise|tensor=scharr|density=structure|sampling=wse|color=local_mean|scale=spacing|renderer=normalized|opacity=none|loss=charbonnier|optimizer=adam|lr_schedule=none|refine=none|pyramid=pyramid` |
| 14 | 20.6233 | 1.3621 | 0.86965 | 0.318 | 5.496 | 4 | `strategy=aniso_flanking|tensor=scharr|density=hybrid|sampling=wse|color=bilinear|scale=uniform|renderer=normalized|opacity=constant|loss=l1|optimizer=adamw|lr_schedule=none|refine=prune_residual_add|pyramid=single` |
| 15 | 20.5802 | 1.3414 | 0.86704 | 0.326 | 0.330 | 4 | `strategy=aniso_flanking|tensor=central|density=variance|sampling=density_random|color=bilinear|scale=uniform|renderer=normalized|opacity=constant|loss=l1|optimizer=adamw|lr_schedule=none|refine=prune_residual_add|pyramid=single` |
| 16 | 20.5309 | 1.3553 | 0.86390 | 0.291 | 0.294 | 4 | `strategy=aniso_flanking|tensor=central|density=hybrid|sampling=jittered_grid|color=two_sided|scale=uniform|renderer=normalized|opacity=none|loss=l1|optimizer=adam|lr_schedule=none|refine=prune_residual_add|pyramid=single` |
| 17 | 20.4977 | 1.5258 | 0.85417 | 0.171 | 0.171 | 4 | `strategy=iso_blue_noise|tensor=scharr|density=hybrid|sampling=density_random|color=bilinear|scale=spacing|renderer=normalized|opacity=none|loss=l1|optimizer=adamw|lr_schedule=none|refine=prune_residual_add|pyramid=pyramid` |
| 18 | 20.4017 | 1.4389 | 0.85990 | 0.178 | 0.181 | 4 | `strategy=iso_blue_noise|tensor=scharr|density=structure|sampling=jittered_grid|color=two_sided|scale=spacing|renderer=normalized|opacity=none|loss=l1|optimizer=adamw|lr_schedule=none|refine=residual_add|pyramid=single` |
| 19 | 20.4016 | 1.4480 | 0.85976 | 0.177 | 0.179 | 4 | `strategy=iso_blue_noise|tensor=central|density=structure|sampling=jittered_grid|color=two_sided|scale=spacing|renderer=normalized|opacity=none|loss=l1|optimizer=adamw|lr_schedule=none|refine=prune_residual_add|pyramid=single` |
| 20 | 20.3844 | 1.2999 | 0.85816 | 0.316 | 0.319 | 4 | `strategy=aniso_flanking|tensor=central|density=hybrid|sampling=density_random|color=bilinear|scale=uniform|renderer=normalized|opacity=constant|loss=charbonnier|optimizer=adamw|lr_schedule=cosine|refine=residual_add|pyramid=single` |

## Speed/Quality Frontier

| Rank | Mean PSNR | Mean MS-SSIM | Mean total sec | Config |
|---:|---:|---:|---:|---|
| 1 | 24.8218 | 0.95270 | 0.166 | `strategy=aniso_onedge|tensor=central|density=variance|sampling=density_random|color=bilinear|scale=spacing|renderer=normalized|opacity=none|loss=l1|optimizer=adamw|lr_schedule=none|refine=none|pyramid=single` |
| 2 | 24.7877 | 0.94691 | 5.082 | `strategy=aniso_onedge|tensor=scharr|density=variance|sampling=wse|color=two_sided|scale=spacing|renderer=normalized|opacity=constant|loss=l1|optimizer=adam|lr_schedule=none|refine=residual_add|pyramid=single` |
| 3 | 25.3951 | 0.95656 | 5.406 | `strategy=aniso_flanking|tensor=central|density=hybrid|sampling=wse|color=bilinear|scale=spacing|renderer=normalized|opacity=none|loss=l1|optimizer=adamw|lr_schedule=none|refine=none|pyramid=single` |
| 4 | 24.5855 | 0.94455 | 5.428 | `strategy=aniso_onedge|tensor=central|density=structure|sampling=wse|color=bilinear|scale=spacing|renderer=normalized|opacity=constant|loss=charbonnier|optimizer=adamw|lr_schedule=none|refine=residual_add|pyramid=single` |

## Best Additive Renderer Configs

| Rank | Mean PSNR | Mean MS-SSIM | Mean total sec | Config |
|---:|---:|---:|---:|---|
| 1 | 17.8218 | 0.77509 | 4.966 | `strategy=aniso_flanking|tensor=central|density=variance|sampling=wse|color=two_sided|scale=spacing|renderer=additive|opacity=none|loss=l1|optimizer=adamw|lr_schedule=none|refine=none|pyramid=single` |
| 2 | 17.1098 | 0.74099 | 2.625 | `strategy=iso_blue_noise|tensor=scharr|density=variance|sampling=wse|color=two_sided|scale=spacing|renderer=additive|opacity=none|loss=l1|optimizer=adamw|lr_schedule=none|refine=none|pyramid=single` |
| 3 | 15.3805 | 0.67110 | 2.900 | `strategy=iso_blue_noise|tensor=central|density=structure|sampling=wse|color=bilinear|scale=spacing|renderer=additive|opacity=none|loss=l1|optimizer=adam|lr_schedule=cosine|refine=none|pyramid=single` |
| 4 | 14.2692 | 0.67946 | 0.155 | `strategy=aniso_onedge|tensor=central|density=variance|sampling=density_random|color=bilinear|scale=spacing|renderer=additive|opacity=none|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single` |
| 5 | 7.8960 | 0.57732 | 0.171 | `strategy=aniso_onedge|tensor=scharr|density=hybrid|sampling=jittered_grid|color=local_mean|scale=spacing|renderer=additive|opacity=none|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single` |
| 6 | 2.8832 | 0.30796 | 2.690 | `strategy=iso_blue_noise|tensor=central|density=variance|sampling=wse|color=bilinear|scale=uniform|renderer=additive|opacity=none|loss=charbonnier|optimizer=adam|lr_schedule=none|refine=none|pyramid=single` |
| 7 | 2.8001 | 0.57910 | 0.194 | `strategy=aniso_onedge|tensor=scharr|density=variance|sampling=density_random|color=two_sided|scale=spacing|renderer=additive|opacity=constant|loss=l1|optimizer=adam|lr_schedule=cosine|refine=residual_add|pyramid=single` |
| 8 | 2.7857 | 0.57638 | 0.192 | `strategy=aniso_flanking|tensor=scharr|density=hybrid|sampling=density_random|color=local_mean|scale=spacing|renderer=additive|opacity=constant|loss=charbonnier|optimizer=adam|lr_schedule=cosine|refine=residual_add|pyramid=single` |
| 9 | 1.9257 | 0.67609 | 2.721 | `strategy=iso_blue_noise|tensor=scharr|density=variance|sampling=wse|color=two_sided|scale=spacing|renderer=additive|opacity=constant|loss=l1|optimizer=adam|lr_schedule=none|refine=residual_add|pyramid=single` |
| 10 | 1.7505 | 0.38024 | 2.958 | `strategy=iso_blue_noise|tensor=scharr|density=hybrid|sampling=wse|color=local_mean|scale=uniform|renderer=additive|opacity=constant|loss=l1|optimizer=adam|lr_schedule=none|refine=residual_add|pyramid=single` |

## Best Normalized Pyramid Configs

| Rank | Mean PSNR | Mean MS-SSIM | Mean total sec | Config |
|---:|---:|---:|---:|---|
| 1 | 21.9843 | 0.90495 | 0.265 | `strategy=aniso_flanking|tensor=scharr|density=structure|sampling=density_random|color=local_mean|scale=spacing|renderer=normalized|opacity=constant|loss=charbonnier|optimizer=adamw|lr_schedule=cosine|refine=none|pyramid=pyramid` |
| 2 | 21.4112 | 0.88923 | 4.605 | `strategy=aniso_onedge|tensor=scharr|density=structure|sampling=wse|color=two_sided|scale=spacing|renderer=normalized|opacity=none|loss=l1|optimizer=adam|lr_schedule=cosine|refine=residual_add|pyramid=pyramid` |
| 3 | 21.3128 | 0.88682 | 2.214 | `strategy=iso_blue_noise|tensor=scharr|density=structure|sampling=wse|color=local_mean|scale=spacing|renderer=normalized|opacity=none|loss=charbonnier|optimizer=adam|lr_schedule=none|refine=none|pyramid=pyramid` |
| 4 | 20.4977 | 0.85417 | 0.171 | `strategy=iso_blue_noise|tensor=scharr|density=hybrid|sampling=density_random|color=bilinear|scale=spacing|renderer=normalized|opacity=none|loss=l1|optimizer=adamw|lr_schedule=none|refine=prune_residual_add|pyramid=pyramid` |
| 5 | 20.2508 | 0.85661 | 0.295 | `strategy=aniso_flanking|tensor=central|density=variance|sampling=jittered_grid|color=local_mean|scale=spacing|renderer=normalized|opacity=constant|loss=charbonnier|optimizer=adamw|lr_schedule=none|refine=none|pyramid=pyramid` |
| 6 | 18.2317 | 0.71615 | 0.180 | `strategy=iso_blue_noise|tensor=central|density=variance|sampling=jittered_grid|color=bilinear|scale=uniform|renderer=normalized|opacity=none|loss=l1|optimizer=adam|lr_schedule=none|refine=residual_add|pyramid=pyramid` |
| 7 | 17.4847 | 0.66849 | 2.548 | `strategy=iso_blue_noise|tensor=central|density=hybrid|sampling=wse|color=local_mean|scale=uniform|renderer=normalized|opacity=none|loss=l1|optimizer=adamw|lr_schedule=none|refine=prune_residual_add|pyramid=pyramid` |
| 8 | 17.3365 | 0.64162 | 0.212 | `strategy=iso_blue_noise|tensor=scharr|density=structure|sampling=density_random|color=bilinear|scale=uniform|renderer=normalized|opacity=none|loss=charbonnier|optimizer=adamw|lr_schedule=cosine|refine=residual_add|pyramid=pyramid` |

## Stage Effects: All rows

### strategy

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| aniso_onedge | 44 | 11.7822 | 0.68442 | 0.594 | 2.001 |
| aniso_flanking | 96 | 9.3480 | 0.62484 | 0.652 | 2.141 |
| iso_blue_noise | 116 | 9.2206 | 0.60800 | 0.400 | 1.131 |

### tensor

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| central | 148 | 11.2013 | 0.64504 | 0.419 | 1.679 |
| scharr | 108 | 7.6631 | 0.60335 | 0.678 | 1.632 |

### density

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| structure | 64 | 13.5324 | 0.70583 | 0.942 | 1.949 |
| variance | 104 | 9.9068 | 0.64510 | 0.444 | 1.487 |
| hybrid | 88 | 6.6935 | 0.54959 | 0.326 | 1.653 |

### sampling

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| wse | 96 | 11.7041 | 0.68529 | 1.045 | 4.059 |
| jittered_grid | 80 | 9.2407 | 0.60415 | 0.210 | 0.212 |
| density_random | 80 | 7.7819 | 0.58134 | 0.226 | 0.228 |

### color

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| bilinear | 92 | 12.1392 | 0.67727 | 0.633 | 2.342 |
| two_sided | 80 | 10.0721 | 0.63666 | 0.511 | 1.509 |
| local_mean | 84 | 6.7004 | 0.56411 | 0.429 | 1.054 |

### scale

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| spacing | 168 | 11.1732 | 0.68423 | 0.486 | 1.533 |
| uniform | 88 | 6.9127 | 0.51906 | 0.609 | 1.901 |

### opacity

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| constant | 100 | 10.8233 | 0.65766 | 0.228 | 1.163 |
| none | 156 | 8.9941 | 0.60809 | 0.721 | 1.977 |

### renderer

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| normalized | 128 | 20.9643 | 0.85574 | 0.493 | 1.861 |
| additive | 128 | -1.5470 | 0.39916 | 0.563 | 1.458 |

### loss

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| l1 | 152 | 11.2951 | 0.65399 | 0.430 | 1.776 |
| charbonnier | 104 | 7.3899 | 0.58866 | 0.671 | 1.490 |

### optimizer

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| adamw | 136 | 11.0238 | 0.64788 | 0.552 | 1.690 |
| adam | 120 | 8.2182 | 0.60430 | 0.501 | 1.625 |

### lr_schedule

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| none | 148 | 11.8922 | 0.67818 | 0.480 | 1.809 |
| cosine | 108 | 6.7163 | 0.55793 | 0.594 | 1.455 |

### refine

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| none | 76 | 12.3651 | 0.64526 | 0.591 | 1.792 |
| residual_add | 108 | 10.0855 | 0.66494 | 0.383 | 2.026 |
| prune_residual_add | 72 | 6.3393 | 0.55243 | 0.680 | 0.969 |

### pyramid

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| single | 172 | 13.0253 | 0.71252 | 0.203 | 1.887 |
| pyramid | 84 | 2.9174 | 0.45326 | 1.194 | 1.194 |


## Stage Effects: Normalized renderer only

### strategy

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| aniso_onedge | 20 | 23.4190 | 0.92771 | 1.070 | 3.095 |
| aniso_flanking | 52 | 21.5093 | 0.88507 | 0.266 | 2.643 |
| iso_blue_noise | 56 | 19.5815 | 0.80280 | 0.498 | 0.694 |

### tensor

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| scharr | 44 | 21.1790 | 0.86725 | 0.791 | 2.180 |
| central | 84 | 20.8518 | 0.84970 | 0.337 | 1.694 |

### density

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| structure | 44 | 21.0584 | 0.85970 | 0.790 | 1.761 |
| variance | 48 | 20.9461 | 0.85734 | 0.224 | 1.451 |
| hybrid | 36 | 20.8734 | 0.84876 | 0.489 | 2.530 |

### sampling

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| wse | 48 | 21.6745 | 0.86836 | 0.961 | 4.606 |
| density_random | 36 | 20.9442 | 0.85157 | 0.225 | 0.227 |
| jittered_grid | 44 | 20.2059 | 0.84537 | 0.201 | 0.203 |

### color

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| two_sided | 36 | 21.3363 | 0.88242 | 0.687 | 1.811 |
| bilinear | 56 | 21.0809 | 0.84787 | 0.233 | 2.281 |
| local_mean | 36 | 20.4108 | 0.84128 | 0.703 | 1.258 |

### scale

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| spacing | 80 | 21.9617 | 0.89599 | 0.520 | 1.822 |
| uniform | 48 | 19.3018 | 0.78864 | 0.447 | 1.926 |

### opacity

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| none | 72 | 20.9995 | 0.85222 | 0.696 | 2.125 |
| constant | 56 | 20.9190 | 0.86025 | 0.232 | 1.521 |

### renderer

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| normalized | 128 | 20.9643 | 0.85574 | 0.493 | 1.861 |

### loss

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| charbonnier | 52 | 21.0717 | 0.86242 | 0.376 | 1.199 |
| l1 | 76 | 20.8908 | 0.85116 | 0.573 | 2.314 |

### optimizer

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| adam | 48 | 21.2729 | 0.86617 | 0.755 | 2.454 |
| adamw | 80 | 20.7791 | 0.84948 | 0.336 | 1.505 |

### lr_schedule

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| none | 80 | 21.3638 | 0.86894 | 0.433 | 1.733 |
| cosine | 48 | 20.2983 | 0.83374 | 0.593 | 2.074 |

### refine

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| none | 36 | 21.4635 | 0.87727 | 0.432 | 1.569 |
| residual_add | 60 | 20.9605 | 0.84662 | 0.511 | 2.401 |
| prune_residual_add | 32 | 20.4097 | 0.84860 | 0.527 | 1.177 |

### pyramid

| Value | N | Mean PSNR | Mean MS-SSIM | Mean fit sec | Mean total sec |
|---|---:|---:|---:|---:|---:|
| single | 96 | 21.3478 | 0.87356 | 0.220 | 2.044 |
| pyramid | 32 | 19.8137 | 0.80226 | 1.311 | 1.311 |

