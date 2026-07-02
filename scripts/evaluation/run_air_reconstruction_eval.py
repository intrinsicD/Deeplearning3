#!/usr/bin/env python3
"""Run AIR reconstruction metrics without AIR's visualization dependency."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pytorch_msssim import ms_ssim


REPO_ROOT = Path(__file__).resolve().parents[2]
AIR_ROOT = REPO_ROOT / "external" / "AIR"
if str(AIR_ROOT) not in sys.path:
    sys.path.insert(0, str(AIR_ROOT))

from model.airnet import AIRNet  # noqa: E402


def _load_image(path: Path, device: str) -> torch.Tensor:
    data = path.read_bytes()
    image = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def _metrics(render: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    render = render.clamp(0, 1).float()
    target = target.float()
    mse = F.mse_loss(render, target).item()
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    return {
        "l1": F.l1_loss(render, target).item(),
        "mse": mse,
        "psnr": psnr,
        "ms_ssim": ms_ssim(render, target, data_range=1, size_average=True).item(),
    }


def _save_tensor_image(path: Path, image: torch.Tensor) -> None:
    arr = image.detach().cpu().clamp(0, 1).numpy()[0].transpose(1, 2, 0)
    Image.fromarray((arr * 255.0).astype(np.uint8)).save(path)


def _make_grid(rows: list[dict[str, Any]], out_path: Path) -> None:
    cell_w, cell_h, label_h, gap = 320, 320, 36, 6
    cols = ["target_path", "air_raw_path", "air_quant_path"]
    canvas = Image.new(
        "RGB",
        (cell_w * len(cols) + gap * (len(cols) - 1), (cell_h + label_h) * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(rows):
        y = r * (cell_h + label_h)
        labels = [
            "target",
            f"AIR raw {row['air_raw']['psnr']:.2f} dB",
            f"AIR quant {row['air_quant']['psnr']:.2f} dB" if row.get("air_quant") else "AIR quant n/a",
        ]
        for c, key in enumerate(cols):
            path = row.get(key)
            if not path:
                continue
            img = Image.open(path).convert("RGB")
            x = c * (cell_w + gap)
            draw.text((x + 6, y + 10), labels[c], fill=(20, 20, 20))
            thumb = Image.new("RGB", (cell_w, cell_h), (245, 245, 245))
            iw, ih = img.size
            scale = min(cell_w / iw, cell_h / ih)
            resized = img.resize((round(iw * scale), round(ih * scale)), Image.Resampling.LANCZOS)
            thumb.paste(resized, ((cell_w - resized.size[0]) // 2, (cell_h - resized.size[1]) // 2))
            canvas.paste(thumb, (x, y + label_h))
    canvas.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timing-repeats", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = AIRNet.from_pretrained(args.checkpoint)
    model.to(args.device)
    model.eval()
    max_stage = max(model.head_num - 1, 0)

    rows: list[dict[str, Any]] = []
    image_paths = sorted(
        p for p in args.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    for idx, image_path in enumerate(image_paths, 1):
        print(f"[{idx}/{len(image_paths)}] {image_path}", flush=True)
        target = _load_image(image_path, args.device)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        with torch.inference_mode():
            outputs = model(target, stage=max_stage)

        times = []
        for _ in range(args.timing_repeats):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                outputs = model(target, stage=max_stage)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)

        raw = outputs["stage_results"][-1]["image"].clamp(0, 1)
        quant = outputs.get("quant_image")
        gaussian_num = float(outputs["gaussian_num"].detach().cpu().reshape(-1)[0])
        router_usage = float(outputs["router_usage"].detach().cpu().reshape(-1)[0])
        quant_bpp = outputs.get("quant_bpp")
        quant_bpp_value = (
            float(quant_bpp.detach().cpu().reshape(-1)[0]) if isinstance(quant_bpp, torch.Tensor) else None
        )

        target_path = args.out_dir / f"{image_path.stem}_target.png"
        raw_path = args.out_dir / f"{image_path.stem}_air_raw.png"
        quant_path = args.out_dir / f"{image_path.stem}_air_quant.png"
        Image.open(image_path).convert("RGB").save(target_path)
        _save_tensor_image(raw_path, raw)
        air_quant_metrics = None
        quant_path_str = None
        if isinstance(quant, torch.Tensor):
            quant = quant.clamp(0, 1)
            _save_tensor_image(quant_path, quant)
            air_quant_metrics = _metrics(quant, target)
            quant_path_str = str(quant_path)

        row = {
            "image": str(image_path),
            "target_path": str(target_path),
            "air_raw_path": str(raw_path),
            "air_quant_path": quant_path_str,
            "air_raw": _metrics(raw, target),
            "air_quant": air_quant_metrics,
            "gaussian_num": gaussian_num,
            "router_usage": router_usage,
            "quant_bpp": quant_bpp_value,
            "inference_time_ms_mean": float(np.mean(times)),
            "inference_time_ms_min": float(np.min(times)),
        }
        if torch.cuda.is_available():
            row["max_cuda_gb"] = torch.cuda.max_memory_allocated() / 1e9
        rows.append(row)
        print(
            f"  raw {row['air_raw']['psnr']:.2f} dB, "
            f"quant {air_quant_metrics['psnr']:.2f} dB, "
            f"gaussians {gaussian_num:.0f}",
            flush=True,
        )

    summary = {"checkpoint": str(args.checkpoint), "rows": rows}
    for key in ("air_raw", "air_quant"):
        valid = [r[key] for r in rows if r.get(key)]
        if valid:
            summary[f"{key}_average"] = {
                metric: float(np.mean([r[metric] for r in valid]))
                for metric in ("l1", "mse", "psnr", "ms_ssim")
            }
    summary["average_gaussian_num"] = float(np.mean([r["gaussian_num"] for r in rows]))
    summary["average_inference_time_ms_mean"] = float(np.mean([r["inference_time_ms_mean"] for r in rows]))
    summary["average_quant_bpp"] = float(np.mean([r["quant_bpp"] for r in rows if r["quant_bpp"] is not None]))
    if torch.cuda.is_available():
        summary["average_max_cuda_gb"] = float(np.mean([r.get("max_cuda_gb", 0.0) for r in rows]))

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    grid_path = args.out_dir / "air_grid.png"
    _make_grid(rows, grid_path)
    print(f"Saved {args.out_dir / 'summary.json'}")
    print(f"Saved {grid_path}")


if __name__ == "__main__":
    main()
