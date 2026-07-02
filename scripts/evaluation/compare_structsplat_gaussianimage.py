#!/usr/bin/env python3
"""Compare StructSplat and official GaussianImage on the same images/budget."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pytorch_msssim import ms_ssim


REPO_ROOT = Path(__file__).resolve().parents[2]
STRUCTSPLAT_SRC = REPO_ROOT / "external" / "structsplat" / "src"
GAUSSIANIMAGE_SRC = REPO_ROOT / "external" / "GaussianImage"

for path in (STRUCTSPLAT_SRC, GAUSSIANIMAGE_SRC):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from structsplat.config import FitConfig, InitConfig  # noqa: E402
from structsplat.fit import fit as structsplat_fit  # noqa: E402
from structsplat import init as structsplat_init  # noqa: E402
from gaussianimage_rs import GaussianImage_RS  # noqa: E402


DEFAULT_IMAGES = [
    "data/coco2017/val2017/000000186042.jpg",
    "data/coco2017/val2017/000000190140.jpg",
    "data/coco2017/val2017/000000444879.jpg",
    "data/coco2017/val2017/000000554838.jpg",
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_resized(path: Path, max_side: int) -> tuple[Image.Image, np.ndarray]:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        image = image.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return image, arr


def _metrics(render_bchw: torch.Tensor, target_bchw: torch.Tensor) -> dict[str, float]:
    render = render_bchw.clamp(0, 1).float()
    target = target_bchw.float()
    mse = F.mse_loss(render, target).item()
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    try:
        ms_ssim_value = ms_ssim(render, target, data_range=1, size_average=True).item()
    except AssertionError:
        ms_ssim_value = float("nan")
    return {
        "l1": F.l1_loss(render, target).item(),
        "mse": mse,
        "psnr": psnr,
        "ms_ssim": ms_ssim_value,
    }


def _save_np_image(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255.0).astype(np.uint8)).save(path)


def _run_structsplat(
    image_np: np.ndarray,
    target_bchw: torch.Tensor,
    *,
    num_gaussians: int,
    iters: int,
    seed: int,
    device: str,
    render_chunk: int,
    verbose: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    target_hwc = torch.as_tensor(image_np, device=device)
    icfg = InitConfig(strategy="aniso_flanking", num_gaussians=num_gaussians, seed=seed)
    fcfg = FitConfig(
        iters=iters,
        render_chunk=render_chunk,
        log_every=max(iters, 1),
        ssim_weight=0.3,
    )
    start = time.time()
    field = structsplat_init.build_field(image_np, icfg, device=device)
    out = structsplat_fit(field, target_hwc, fcfg, verbose=verbose)
    elapsed = time.time() - start

    render_hwc = out["render"].detach().cpu().numpy().clip(0, 1)
    render_bchw = out["render"].permute(2, 0, 1).unsqueeze(0)
    m = _metrics(render_bchw, target_bchw)
    m.update(
        {
            "fit_seconds": elapsed,
            "gaussians": float(out["n_gaussians"]),
            "psnr_internal": float(out["psnr"]),
            "ssim_internal": float(out["ssim"]),
        }
    )
    if torch.cuda.is_available():
        m["max_cuda_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return render_hwc, m


def _run_gaussianimage(
    target_bchw: torch.Tensor,
    *,
    num_gaussians: int,
    iters: int,
    seed: int,
    device: str,
    lr: float,
    verbose: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _set_seed(seed)
    _, _, h, w = target_bchw.shape
    model = GaussianImage_RS(
        loss_type="L2",
        opt_type="adan",
        num_points=num_gaussians,
        H=h,
        W=w,
        BLOCK_H=16,
        BLOCK_W=16,
        device=torch.device(device),
        lr=lr,
        quantize=False,
    ).to(device)
    model.train()

    start = time.time()
    last_psnr = None
    for step in range(1, iters + 1):
        _, last_psnr = model.train_iter(target_bchw)
        if verbose and (step == 1 or step == iters or step % 100 == 0):
            print(f"    GaussianImage iter {step:4d}/{iters} psnr {last_psnr:.3f}", flush=True)
    elapsed = time.time() - start

    model.eval()
    with torch.no_grad():
        render_bchw = model()["render"].clamp(0, 1)
    render_hwc = render_bchw.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    m = _metrics(render_bchw, target_bchw)
    m.update(
        {
            "fit_seconds": elapsed,
            "gaussians": float(num_gaussians),
            "train_last_psnr": float(last_psnr or 0.0),
        }
    )
    if torch.cuda.is_available():
        m["max_cuda_gb"] = torch.cuda.max_memory_allocated() / 1e9
    return render_hwc, m


def _make_grid(rows: list[dict[str, Any]], out_path: Path) -> None:
    cell_w, cell_h, label_h, gap = 320, 320, 36, 6
    canvas = Image.new(
        "RGB",
        (cell_w * 3 + gap * 2, (cell_h + label_h) * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_idx, row in enumerate(rows):
        y = row_idx * (cell_h + label_h)
        cells = [
            (Image.open(row["target_path"]).convert("RGB"), "target"),
            (
                Image.open(row["structsplat_path"]).convert("RGB"),
                f"StructSplat {row['structsplat']['psnr']:.2f} dB",
            ),
            (
                Image.open(row["gaussianimage_path"]).convert("RGB"),
                f"GaussianImage {row['gaussianimage']['psnr']:.2f} dB",
            ),
        ]
        for col, (img, label) in enumerate(cells):
            x = col * (cell_w + gap)
            draw.text((x + 6, y + 10), label, fill=(20, 20, 20))
            thumb = Image.new("RGB", (cell_w, cell_h), (245, 245, 245))
            iw, ih = img.size
            scale = min(cell_w / iw, cell_h / ih)
            resized = img.resize((round(iw * scale), round(ih * scale)), Image.Resampling.LANCZOS)
            thumb.paste(resized, ((cell_w - resized.size[0]) // 2, (cell_h - resized.size[1]) // 2))
            canvas.paste(thumb, (x, y + label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _write_outputs(rows: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> None:
    structsplat_iters = args.structsplat_iters or args.iters
    gaussianimage_iters = args.gaussianimage_iters or args.iters
    summary = {
        "protocol": {
            "images": [str(p) for p in args.images],
            "max_side": args.max_side,
            "num_gaussians": args.num_gaussians,
            "iters": args.iters,
            "structsplat_iters": structsplat_iters,
            "gaussianimage_iters": gaussianimage_iters,
            "device": args.device,
            "gaussianimage_lr": args.gaussianimage_lr,
            "structsplat_render_chunk": args.structsplat_chunk,
        },
        "rows": rows,
    }
    for method in ("structsplat", "gaussianimage"):
        method_rows = [row[method] for row in rows]
        summary[f"{method}_average"] = {
            key: float(np.mean([r[key] for r in method_rows if key in r]))
            for key in ("l1", "mse", "psnr", "ms_ssim", "fit_seconds", "max_cuda_gb")
            if any(key in r for r in method_rows)
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "method", "gaussians", "iters", "psnr", "ms_ssim", "l1", "mse", "fit_seconds", "max_cuda_gb"])
        for row in rows:
            for method in ("structsplat", "gaussianimage"):
                m = row[method]
                method_iters = structsplat_iters if method == "structsplat" else gaussianimage_iters
                writer.writerow(
                    [
                        row["image"],
                        method,
                        int(m["gaussians"]),
                        method_iters,
                        m["psnr"],
                        m["ms_ssim"],
                        m["l1"],
                        m["mse"],
                        m["fit_seconds"],
                        m.get("max_cuda_gb", ""),
                    ]
                )

    md = [
        "# StructSplat vs GaussianImage",
        "",
        f"- Images: {len(rows)} COCO val images",
        f"- Max side: {args.max_side}",
        f"- Gaussians: {args.num_gaussians}",
        f"- StructSplat iterations: {structsplat_iters}",
        f"- GaussianImage iterations: {gaussianimage_iters}",
        "",
        "| Method | Avg PSNR | Avg MS-SSIM | Avg L1 | Avg fit seconds | Avg max CUDA GB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("structsplat", "gaussianimage"):
        avg = summary[f"{method}_average"]
        md.append(
            f"| {method} | {avg['psnr']:.4f} | {avg['ms_ssim']:.4f} | {avg['l1']:.5f} | "
            f"{avg['fit_seconds']:.2f} | {avg.get('max_cuda_gb', 0.0):.3f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", type=Path, default=[Path(p) for p in DEFAULT_IMAGES])
    parser.add_argument("--out-dir", type=Path, default=Path("runs/structsplat_vs_gaussianimage_fair_20260702"))
    parser.add_argument("--max-side", type=int, default=320)
    parser.add_argument("--num-gaussians", type=int, default=2048)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--structsplat-iters", type=int, default=None)
    parser.add_argument("--gaussianimage-iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gaussianimage-lr", type=float, default=1e-3)
    parser.add_argument("--structsplat-chunk", type=int, default=2048)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    structsplat_iters = args.structsplat_iters or args.iters
    gaussianimage_iters = args.gaussianimage_iters or args.iters
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for image_idx, image_path in enumerate(args.images):
        print(f"[{image_idx + 1}/{len(args.images)}] {image_path}", flush=True)
        target_pil, image_np = _load_resized(image_path, args.max_side)
        h, w = image_np.shape[:2]
        target_bchw = torch.as_tensor(image_np, device=args.device).permute(2, 0, 1).unsqueeze(0)

        stem = image_path.stem
        target_path = args.out_dir / f"{stem}_target.png"
        target_pil.save(target_path)

        print("  StructSplat...", flush=True)
        struct_img, struct_metrics = _run_structsplat(
            image_np,
            target_bchw,
            num_gaussians=args.num_gaussians,
            iters=structsplat_iters,
            seed=args.seed + image_idx,
            device=args.device,
            render_chunk=args.structsplat_chunk,
            verbose=args.verbose,
        )
        struct_path = args.out_dir / f"{stem}_structsplat_{args.num_gaussians}g_{structsplat_iters}i.png"
        _save_np_image(struct_path, struct_img)

        print("  GaussianImage-RS...", flush=True)
        gi_img, gi_metrics = _run_gaussianimage(
            target_bchw,
            num_gaussians=args.num_gaussians,
            iters=gaussianimage_iters,
            seed=args.seed + image_idx,
            device=args.device,
            lr=args.gaussianimage_lr,
            verbose=args.verbose,
        )
        gi_path = args.out_dir / f"{stem}_gaussianimage_rs_{args.num_gaussians}g_{gaussianimage_iters}i.png"
        _save_np_image(gi_path, gi_img)

        row = {
            "image": str(image_path),
            "resized_size": [w, h],
            "target_path": str(target_path),
            "structsplat_path": str(struct_path),
            "gaussianimage_path": str(gi_path),
            "structsplat": struct_metrics,
            "gaussianimage": gi_metrics,
        }
        rows.append(row)
        print(
            "  done: "
            f"StructSplat {struct_metrics['psnr']:.2f} dB, "
            f"GaussianImage {gi_metrics['psnr']:.2f} dB",
            flush=True,
        )

    _write_outputs(rows, args.out_dir, args)
    grid_path = args.out_dir / "comparison_grid.png"
    _make_grid(rows, grid_path)
    print(f"Saved {args.out_dir / 'summary.md'}")
    print(f"Saved {grid_path}")


if __name__ == "__main__":
    main()
