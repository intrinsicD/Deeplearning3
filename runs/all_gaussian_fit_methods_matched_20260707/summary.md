# Fair Density-Control Comparison

Matched-policy comparison against repo-inspired 2D Gaussian baselines.

Growth rows share the same initial Gaussian count, final cap, growth wave count, fitter, renderer, loss, target tracking, and iteration budget.
This is not a native external-repo benchmark; it isolates placement/growth policies inside StructSplat's fitter and exact renderer.

## Methods

| Method | Track | Description |
|---|---|---|
| GaussianImage fixed | fixed-full | GaussianImage-style random fixed-count control; starts at the final cap and does not grow. |
| GaussianImage++ residual | repo-growth | GaussianImage++-style analogue: random half-budget start plus residual-add growth. |
| Image-GS residual | repo-growth | Image-GS-style analogue: gradient-density random half-budget start plus residual-add growth. |
| SS on-edge + residual | same-growth | StructSplat on-edge initializer under the same residual-add growth as external analogues. |
| SS on-edge + residual relocate | same-growth+relocate | StructSplat on-edge residual-add growth plus split-scheduled residual relocation. |
| SS on-edge + residual feature cap | same-growth+feature-cap | StructSplat on-edge residual-add growth with feature-adaptive per-Gaussian scale caps. |
| SS on-edge + tensor | tensor-growth | StructSplat on-edge initializer plus tensor-aware residual growth. |
| SS on-edge + tensor feature cap | tensor-growth+feature-cap | StructSplat on-edge tensor-aware residual growth with feature-adaptive scale caps. |
| SS flanking + tensor | tensor-growth | StructSplat flanking initializer plus tensor-aware residual growth. |
| SS qt-WSE + residual | same-growth | StructSplat quadtree-WSE initializer under the same residual-add growth as external analogues. |
| SS qt-WSE + residual relocate | same-growth+relocate | StructSplat quadtree-WSE residual-add growth plus split-scheduled residual relocation. |
| SS qt-WSE + residual feature cap | same-growth+feature-cap | StructSplat quadtree-WSE residual-add growth with feature-adaptive scale caps. |
| SS qt-WSE + tensor | tensor-growth | StructSplat quadtree-WSE initializer plus tensor-aware residual growth. |
| SS qt-WSE + tensor feature cap | tensor-growth+feature-cap | StructSplat quadtree-WSE tensor-aware residual growth with feature-adaptive scale caps. |
| SS qt-hybrid + tensor | tensor-growth | StructSplat quadtree-hybrid initializer plus tensor-aware residual growth. |
| Floyd + tensor | tensor-growth-control | Floyd-Steinberg placement control plus tensor-aware residual growth. |

## Overall Means

| Method | Runs | PSNR | PSNR Std | MS-SSIM | MS-SSIM Std | AUC | LPIPS | Init s | Fit s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GaussianImage fixed | 8 | 25.7952 | 3.1563 | 0.97257 | 0.01641 | 24.391 | - | 0.002 | 0.810 | 0.812 |
| GaussianImage++ residual | 8 | 27.2466 | 3.5059 | 0.97283 | 0.01645 | 23.503 | - | 0.001 | 0.747 | 0.747 |
| Image-GS residual | 8 | 27.2969 | 3.6104 | 0.97187 | 0.01756 | 23.868 | - | 0.003 | 0.744 | 0.747 |
| SS on-edge + residual | 8 | 27.4268 | 3.5044 | 0.97138 | 0.01782 | 24.522 | - | 0.042 | 0.740 | 0.782 |
| SS on-edge + residual relocate | 8 | 27.1509 | 3.4503 | 0.96971 | 0.01909 | 24.308 | - | 0.041 | 0.785 | 0.826 |
| SS on-edge + residual feature cap | 8 | 27.4770 | 3.5983 | 0.96984 | 0.01922 | 24.568 | - | 0.041 | 0.753 | 0.795 |
| SS on-edge + tensor | 8 | 27.2105 | 3.3605 | 0.97148 | 0.01668 | 24.542 | - | 0.042 | 0.744 | 0.786 |
| SS on-edge + tensor feature cap | 8 | 27.6902 | 3.7717 | 0.97096 | 0.01792 | 24.842 | - | 0.042 | 0.749 | 0.792 |
| SS flanking + tensor | 8 | 27.2895 | 3.3470 | 0.97119 | 0.01787 | 24.547 | - | 0.041 | 0.738 | 0.779 |
| SS qt-WSE + residual | 8 | 27.3767 | 3.5199 | 0.97071 | 0.01887 | 24.480 | - | 0.094 | 0.736 | 0.830 |
| SS qt-WSE + residual relocate | 8 | 27.1642 | 3.3818 | 0.97016 | 0.01872 | 24.309 | - | 0.093 | 0.783 | 0.876 |
| SS qt-WSE + residual feature cap | 8 | 27.2947 | 3.6903 | 0.96857 | 0.02125 | 24.554 | - | 0.094 | 0.747 | 0.841 |
| SS qt-WSE + tensor | 8 | 27.2393 | 3.4119 | 0.97061 | 0.01845 | 24.518 | - | 0.093 | 0.742 | 0.835 |
| SS qt-WSE + tensor feature cap | 8 | 27.4877 | 3.7033 | 0.96938 | 0.01944 | 24.696 | - | 0.094 | 0.761 | 0.855 |
| SS qt-hybrid + tensor | 8 | 27.2798 | 3.3432 | 0.97159 | 0.01726 | 24.536 | - | 0.051 | 0.740 | 0.791 |
| Floyd + tensor | 8 | 26.9087 | 3.2323 | 0.96684 | 0.02099 | 24.027 | - | 0.019 | 0.745 | 0.764 |

## Convergence

AUC is the area under the logged PSNR-over-iteration curve; higher means better quality earlier in the same 1500-iteration budget.

| Method | AUC | PSNR@0 | PSNR@375 | PSNR@750 | PSNR@1125 | Final PSNR |
|---|---:|---:|---:|---:|---:|---:|
| GaussianImage fixed | 24.391 | 16.921 | 25.717 | 25.791 | 25.791 | 25.795 |
| GaussianImage++ residual | 23.503 | 16.084 | 26.075 | 27.218 | 27.218 | 27.247 |
| Image-GS residual | 23.868 | 16.011 | 26.225 | 27.284 | 27.284 | 27.297 |
| SS on-edge + residual | 24.522 | 16.551 | 26.393 | 27.399 | 27.399 | 27.427 |
| SS on-edge + residual relocate | 24.308 | 16.551 | 26.137 | 27.082 | 27.082 | 27.151 |
| SS on-edge + residual feature cap | 24.568 | 17.072 | 26.470 | 27.461 | 27.461 | 27.477 |
| SS on-edge + tensor | 24.542 | 16.551 | 26.300 | 27.179 | 27.179 | 27.210 |
| SS on-edge + tensor feature cap | 24.842 | 17.072 | 26.789 | 27.675 | 27.675 | 27.690 |
| SS flanking + tensor | 24.547 | 16.626 | 26.368 | 27.263 | 27.263 | 27.289 |
| SS qt-WSE + residual | 24.480 | 16.681 | 26.328 | 27.353 | 27.353 | 27.377 |
| SS qt-WSE + residual relocate | 24.309 | 16.681 | 26.187 | 27.114 | 27.114 | 27.164 |
| SS qt-WSE + residual feature cap | 24.554 | 17.171 | 26.422 | 27.283 | 27.283 | 27.295 |
| SS qt-WSE + tensor | 24.518 | 16.681 | 26.321 | 27.206 | 27.206 | 27.239 |
| SS qt-WSE + tensor feature cap | 24.696 | 17.171 | 26.587 | 27.455 | 27.455 | 27.488 |
| SS qt-hybrid + tensor | 24.536 | 16.900 | 26.324 | 27.251 | 27.251 | 27.280 |
| Floyd + tensor | 24.027 | 15.293 | 25.970 | 26.878 | 26.878 | 26.909 |

Target-hit cells report hit rate across all image/budget cells and mean hit iteration among cells that reached the target.

| Method | Hit 24 | Iter 24 | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GaussianImage fixed | 75% | 114.7 | 25% | 92.5 | 25% | 214.0 | 0% | - |
| GaussianImage++ residual | 100% | 284.6 | 25% | 194.5 | 25% | 287.0 | 25% | 411.0 |
| Image-GS residual | 100% | 272.4 | 25% | 148.0 | 25% | 272.0 | 25% | 392.0 |
| SS on-edge + residual | 100% | 252.0 | 25% | 120.5 | 25% | 237.0 | 25% | 388.5 |
| SS on-edge + residual relocate | 100% | 263.1 | 25% | 136.0 | 25% | 263.5 | 25% | 427.0 |
| SS on-edge + residual feature cap | 100% | 247.5 | 25% | 117.5 | 25% | 217.5 | 25% | 376.0 |
| SS on-edge + tensor | 100% | 253.9 | 25% | 112.5 | 25% | 232.5 | 25% | 439.5 |
| SS on-edge + tensor feature cap | 100% | 228.2 | 25% | 107.5 | 25% | 196.0 | 25% | 319.5 |
| SS flanking + tensor | 100% | 254.1 | 25% | 122.0 | 25% | 248.5 | 25% | 439.5 |
| SS qt-WSE + residual | 100% | 253.2 | 25% | 123.5 | 25% | 239.5 | 25% | 395.5 |
| SS qt-WSE + residual relocate | 100% | 260.4 | 25% | 133.5 | 25% | 279.0 | 25% | 441.5 |
| SS qt-WSE + residual feature cap | 75% | 173.0 | 25% | 116.5 | 25% | 218.5 | 25% | 377.5 |
| SS qt-WSE + tensor | 100% | 255.1 | 25% | 119.5 | 25% | 242.0 | 25% | 425.5 |
| SS qt-WSE + tensor feature cap | 88% | 208.3 | 25% | 109.5 | 25% | 204.5 | 25% | 361.5 |
| SS qt-hybrid + tensor | 100% | 249.1 | 25% | 125.0 | 25% | 253.0 | 25% | 426.5 |
| Floyd + tensor | 75% | 200.3 | 25% | 161.0 | 25% | 289.0 | 25% | 470.5 |

## Means By Budget

| Final budget | Method | Start G | Final G | PSNR | PSNR Std | MS-SSIM | AUC | Fit s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 640 | Floyd + tensor | 320 | 640 | 26.9087 | 3.2323 | 0.96684 | 24.027 | 0.745 |
| 640 | GaussianImage fixed | 640 | 640 | 25.7952 | 3.1563 | 0.97257 | 24.391 | 0.810 |
| 640 | GaussianImage++ residual | 320 | 640 | 27.2466 | 3.5059 | 0.97283 | 23.503 | 0.747 |
| 640 | Image-GS residual | 320 | 640 | 27.2969 | 3.6104 | 0.97187 | 23.868 | 0.744 |
| 640 | SS flanking + tensor | 320 | 640 | 27.2895 | 3.3470 | 0.97119 | 24.547 | 0.738 |
| 640 | SS on-edge + residual | 320 | 640 | 27.4268 | 3.5044 | 0.97138 | 24.522 | 0.740 |
| 640 | SS on-edge + residual feature cap | 320 | 640 | 27.4770 | 3.5983 | 0.96984 | 24.568 | 0.753 |
| 640 | SS on-edge + residual relocate | 320 | 640 | 27.1509 | 3.4503 | 0.96971 | 24.308 | 0.785 |
| 640 | SS on-edge + tensor | 320 | 640 | 27.2105 | 3.3605 | 0.97148 | 24.542 | 0.744 |
| 640 | SS on-edge + tensor feature cap | 320 | 640 | 27.6902 | 3.7717 | 0.97096 | 24.842 | 0.749 |
| 640 | SS qt-hybrid + tensor | 320 | 640 | 27.2798 | 3.3432 | 0.97159 | 24.536 | 0.740 |
| 640 | SS qt-WSE + residual | 320 | 640 | 27.3767 | 3.5199 | 0.97071 | 24.480 | 0.736 |
| 640 | SS qt-WSE + residual feature cap | 320 | 640 | 27.2947 | 3.6903 | 0.96857 | 24.554 | 0.747 |
| 640 | SS qt-WSE + residual relocate | 320 | 640 | 27.1642 | 3.3818 | 0.97016 | 24.309 | 0.783 |
| 640 | SS qt-WSE + tensor | 320 | 640 | 27.2393 | 3.4119 | 0.97061 | 24.518 | 0.742 |
| 640 | SS qt-WSE + tensor feature cap | 320 | 640 | 27.4877 | 3.7033 | 0.96938 | 24.696 | 0.761 |

## Winners By Image/Budget

| Image | Budget | Best PSNR | Best MS-SSIM |
|---|---:|---|---|
| COCO_train2014_000000000009 | 640 | SS on-edge + tensor feature cap (27.590) | SS qt-WSE + tensor feature cap (0.98419) |
| COCO_train2014_000000000025 | 640 | SS qt-WSE + residual feature cap (25.280) | GaussianImage fixed (0.97020) |
| COCO_train2014_000000000030 | 640 | SS on-edge + tensor feature cap (34.087) | GaussianImage fixed (0.99385) |
| COCO_train2014_000000000034 | 640 | SS on-edge + tensor feature cap (24.292) | GaussianImage fixed (0.95367) |

Plots are under `plots/`; visual grids are under `grids/`; per-cell reconstructions are under `reconstructions/`; amplified x6 absolute-difference maps are under `diffs/`.
