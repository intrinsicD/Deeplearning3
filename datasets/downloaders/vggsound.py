"""VGGSound audiovisual correspondence dataset.

* ~200k 10-second clips across 309 audio-visual classes
* Clips downloaded from YouTube via yt-dlp
* CSV: https://www.robots.ox.ac.uk/~vgg/data/vggsound/vggsound.csv
* License: check per-video YouTube terms; dataset metadata CC-BY 4.0
* Paper: Chen et al., VGGSound: A Large-scale Audio-Visual Dataset (ICASSP 2020)
"""

from __future__ import annotations

import csv
import subprocess
import shutil
from pathlib import Path
from typing import Any, List

import torch

from ..base import AudioVisualSample, BaseAVDataset
from ..registry import register_dataset

_CSV_URL = "https://www.robots.ox.ac.uk/~vgg/data/vggsound/vggsound.csv"


@register_dataset("vggsound")
class VGGSoundDataset(BaseAVDataset):
    """VGGSound – audio+video clips with audio-visual class label as caption."""

    @classmethod
    def download(
        cls,
        data_dir: str | Path,
        max_clips: int | None = None,
        workers: int = 4,
        **kwargs: Any,
    ) -> None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        # 1. Download CSV manifest
        csv_path = data_dir / "vggsound.csv"
        if not csv_path.exists():
            cls._http_download(_CSV_URL, csv_path, "VGGSound manifest CSV")
        else:
            print("[skip] vggsound.csv already exists")

        # 2. Check yt-dlp
        if not shutil.which("yt-dlp"):
            print(
                "[WARN] yt-dlp not found.  Install with: pip install yt-dlp\n"
                "       Skipping clip download."
            )
            return

        # 3. Read manifest
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 4:
                    rows.append(row)
        if max_clips:
            rows = rows[:max_clips]

        clips_dir = data_dir / "clips"
        clips_dir.mkdir(exist_ok=True)

        print(f"[VGGSound] Downloading {len(rows)} clips with {workers} workers …")
        _yt_dlp_batch(rows, clips_dir, workers, start_s_col=1, end_s_col=2)

    @classmethod
    def verify(cls, data_dir: str | Path) -> bool:
        data_dir = Path(data_dir)
        return (data_dir / "vggsound.csv").exists()

    def _build_index(self) -> List[Any]:
        csv_path = self.data_dir / "vggsound.csv"
        if not csv_path.exists():
            print(f"[WARN] VGGSound CSV not found. Run VGGSoundDataset.download() first.")
            return []

        clips_dir = self.data_dir / "clips"
        split_map = {"train": "train", "val": "test", "test": "test"}
        target_split = split_map.get(self.split, "train")

        samples = []
        with open(csv_path, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 4:
                    continue
                yt_id, start_s, _, label, split = row[0], row[1], row[2], row[3], row[4] if len(row) > 4 else "train"
                if split != target_split:
                    continue
                # Look for any downloaded clip file
                clip_stem = f"{yt_id}_{int(float(start_s)):06d}"
                for ext in (".mp4", ".webm", ".mkv"):
                    p = clips_dir / (clip_stem + ext)
                    if p.exists():
                        samples.append({"path": p, "label": label, "yt_id": yt_id})
                        break

        return samples

    def _load_sample(self, descriptor: Any) -> AudioVisualSample:
        path = Path(descriptor["path"])
        label = descriptor.get("label", "")
        video = self._load_video_frames(path)
        audio = self._load_audio_wav(path)
        return AudioVisualSample(
            video_frames=video,
            audio=audio,
            text_caption=label,
            metadata={"label": label, "yt_id": descriptor.get("yt_id"), "path": str(path)},
        )


# ---------------------------------------------------------------------------
# Shared yt-dlp helper used by VGGSound, AudioSet, Kinetics, AVSpeech
# ---------------------------------------------------------------------------

def _yt_dlp_batch(
    rows: List[List[str]],
    out_dir: Path,
    workers: int,
    start_s_col: int,
    end_s_col: int,
    yt_id_col: int = 0,
) -> None:
    """Download YouTube clips in parallel using yt-dlp + ffmpeg post-processing."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _download_one(row: List[str]) -> str:
        yt_id = row[yt_id_col]
        start_s = float(row[start_s_col])
        end_s = float(row[end_s_col])
        duration = end_s - start_s
        stem = f"{yt_id}_{int(start_s):06d}"
        out_path = out_dir / f"{stem}.mp4"
        if out_path.exists():
            return f"[skip] {stem}"
        url = f"https://www.youtube.com/watch?v={yt_id}"
        cmd = [
            "yt-dlp",
            "--no-warnings",
            "--quiet",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--download-sections", f"*{start_s}-{end_s}",
            "--force-keyframes-at-cuts",
            "-o", str(out_path),
            url,
        ]
        try:
            subprocess.run(cmd, timeout=120, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"[ok] {stem}"
        except Exception as e:
            return f"[fail] {stem}: {e}"

    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(_download_one, row): row for row in rows}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(rows)} clips processed")

