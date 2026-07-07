# StructSplat vs GaussianImage

- Images: 4 COCO val images
- Max side: 160
- Gaussians: 640
- StructSplat iterations: 500
- GaussianImage iterations: 5000

| Method | Avg PSNR | Avg MS-SSIM | Avg L1 | Avg fit seconds | Avg max CUDA GB |
|---|---:|---:|---:|---:|---:|
| structsplat | 26.7996 | nan | 0.03142 | 5.27 | 0.093 |
| gaussianimage | 27.1569 | nan | 0.03185 | 3.48 | 0.010 |
