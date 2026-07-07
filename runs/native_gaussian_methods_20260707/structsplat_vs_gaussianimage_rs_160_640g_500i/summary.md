# StructSplat vs GaussianImage

- Images: 4 COCO val images
- Max side: 160
- Gaussians: 640
- StructSplat iterations: 500
- GaussianImage iterations: 500

| Method | Avg PSNR | Avg MS-SSIM | Avg L1 | Avg fit seconds | Avg max CUDA GB |
|---|---:|---:|---:|---:|---:|
| structsplat | 26.8327 | nan | 0.03154 | 5.19 | 0.093 |
| gaussianimage | 14.1620 | nan | 0.15195 | 0.43 | 0.010 |
