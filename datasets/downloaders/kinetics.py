"""Kinetics-400 / Kinetics-700 downloader.

* Kinetics-400: 400 human action classes, ~240k 10-second clips
* Kinetics-700: 700 classes, ~650k clips
* Annotation CSVs from: https://github.com/cvdfoundation/kinetics-dataset
* Clips downloaded from YouTube via yt-dlp
* License: CC-BY 4.0 for annotations; YouTube terms for video content
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, List, Optional

import torch

from ..base import AudioVisualSample, BaseAVDataset
from ..registry import register_dataset
from .vggsound import _yt_dlp_batch

_K400_BASE = "https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/k400_targz/"
_K700_BASE = "https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/k700-2020_targz/"

# Official split CSVs (hosted on GitHub)
_K400_CSVS = {
    "train": "https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/k400/annotations/train.csv",
    "val":   "https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/k400/annotations/val.csv",
    "test":  "https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/k400/annotations/test.csv",
}
_K700_CSVS = {
    "train": "https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/k700-2020/annotations/train.csv",
    "val":   "https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/k700-2020/annotations/val.csv",
    "test":  "https://raw.githubusercontent.com/cvdfoundation/kinetics-dataset/main/k700-2020/annotations/test.csv",
}


def _make_kinetics_class(name: str, csvs: dict, variant: str):
    """Factory that creates Kinetics-N dataset class."""

    @register_dataset(name)
    class _KineticsDataset(BaseAVDataset):
        __qualname__ = f"Kinetics{variant}Dataset"

        _CSVS = csvs

        def __init__(self, data_dir: str, split: str = "train", **kwargs: Any) -> None:
            super().__init__(data_dir, split, **kwargs)

        @classmethod
        def download(
            cls,
            data_dir: str | Path,
            max_clips: Optional[int] = None,
            workers: int = 4,
            **kwargs: Any,
        ) -> None:
            data_dir = Path(data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)

            for split_name, url in cls._CSVS.items():
                dest = data_dir / f"{split_name}.csv"
                if not dest.exists():
                    cls._http_download(url, dest, f"Kinetics-{variant} {split_name} CSV")
                else:
                    print(f"[skip] {dest.name}")

            if not shutil.which("yt-dlp"):
                print("[WARN] yt-dlp not found.  pip install yt-dlp")
                return

            for split_name, csv_path in [(s, data_dir / f"{s}.csv") for s in ("train", "val")]:
                if not csv_path.exists():
                    continue
                rows = _parse_kinetics_csv(csv_path, max_clips)
                clips_dir = data_dir / "clips" / split_name
                clips_dir.mkdir(parents=True, exist_ok=True)
                print(f"[Kinetics-{variant}] Downloading {len(rows)} {split_name} clips …")
                _yt_dlp_batch(rows, clips_dir, workers, start_s_col=2, end_s_col=3, yt_id_col=1)

        @classmethod
        def verify(cls, data_dir: str | Path) -> bool:
            return (Path(data_dir) / "train.csv").exists()

        def _build_index(self) -> List[Any]:
            split_key = "val" if self.split == "val" else self.split
            csv_path = self.data_dir / f"{split_key}.csv"
            if not csv_path.exists():
                print(f"[WARN] Kinetics-{variant} CSV not found. Run download() first.")
                return []

            clips_dir = self.data_dir / "clips" / split_key
            rows = _parse_kinetics_csv(csv_path)
            samples = []
            for row in rows:
                label = row[0]
                yt_id = row[1]
                start_s = row[2]
                stem = f"{yt_id}_{int(float(start_s)):06d}"
                for ext in (".mp4", ".webm", ".mkv"):
                    p = clips_dir / (stem + ext)
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
                text_caption=label.replace("_", " "),
                metadata={"label": label, "yt_id": descriptor.get("yt_id"), "path": str(path)},
            )

    _KineticsDataset.__name__ = f"Kinetics{variant}Dataset"
    return _KineticsDataset


Kinetics400Dataset = _make_kinetics_class("kinetics400", _K400_CSVS, "400")
Kinetics700Dataset = _make_kinetics_class("kinetics700", _K700_CSVS, "700")


def _parse_kinetics_csv(path: Path, max_rows: Optional[int] = None) -> List[List[str]]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 4:
                rows.append(row)
            if max_rows and len(rows) >= max_rows:
                break
    return rows

