"""UCF-101 action recognition dataset downloader.

* 13,320 clips across 101 action categories
* ~6.5 GB zip; direct HTTP from UCF servers
* License: free for academic/research use
* URL: https://www.crcv.ucf.edu/data/UCF101.php
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

import torch

from ..base import AudioVisualSample, BaseAVDataset
from ..registry import register_dataset

_UCF101_VIDEO_URL = (
    "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
)
_UCF101_ANNOT_URL = (
    "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
)


@register_dataset("ucf101")
class UCF101Dataset(BaseAVDataset):
    """UCF-101 action recognition, video frames + action label as caption."""

    @classmethod
    def download(cls, data_dir: str | Path, **kwargs: Any) -> None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        rar_path = data_dir / "UCF101.rar"
        if not rar_path.exists():
            cls._http_download(_UCF101_VIDEO_URL, rar_path, "UCF-101 videos (~6.5 GB)")
        else:
            print(f"[skip] {rar_path.name} already downloaded")

        annot_zip = data_dir / "UCF101_splits.zip"
        if not annot_zip.exists():
            cls._http_download(_UCF101_ANNOT_URL, annot_zip, "UCF-101 train/test splits")
        else:
            print(f"[skip] {annot_zip.name} already downloaded")

        videos_dir = data_dir / "UCF-101"
        if not videos_dir.exists():
            # rar extraction requires python-unrar or patoolib / 7z
            print("[extract] UCF101.rar  (requires 'unrar' or 'patool' on PATH)")
            import subprocess, shutil
            if shutil.which("unrar"):
                subprocess.run(["unrar", "x", str(rar_path), str(data_dir)], check=True)
            elif shutil.which("7z"):
                subprocess.run(["7z", "x", str(rar_path), f"-o{data_dir}"], check=True)
            else:
                print(
                    "  [WARN] Neither 'unrar' nor '7z' found.  "
                    "Please manually extract UCF101.rar into data_dir/UCF-101/"
                )

        splits_dir = data_dir / "ucfTrainTestlist"
        if not splits_dir.exists():
            cls._extract_archive(annot_zip, data_dir)

    @classmethod
    def verify(cls, data_dir: str | Path) -> bool:
        data_dir = Path(data_dir)
        return (
            (data_dir / "UCF-101").exists()
            and (data_dir / "ucfTrainTestlist").exists()
        )

    def _build_index(self) -> List[Any]:
        splits_dir = self.data_dir / "ucfTrainTestlist"
        videos_dir = self.data_dir / "UCF-101"

        if not splits_dir.exists() or not videos_dir.exists():
            print(
                f"[WARN] UCF-101 not found at {self.data_dir}. "
                "Run UCF101Dataset.download() first."
            )
            return []

        split_file = splits_dir / f"trainlist0{'1' if self.split == 'train' else '1'}.txt"
        if self.split in ("val", "test"):
            split_file = splits_dir / "testlist01.txt"

        samples = []
        if split_file.exists():
            for line in split_file.read_text().splitlines():
                parts = line.strip().split()
                rel_path = parts[0]
                video_path = videos_dir / rel_path
                if video_path.exists():
                    label = rel_path.split("/")[0]
                    samples.append({"path": video_path, "label": label})
        else:
            # Fallback: glob all AVI files
            for avi in sorted(videos_dir.rglob("*.avi")):
                label = avi.parent.name
                samples.append({"path": avi, "label": label})

        return samples

    def _load_sample(self, descriptor: Any) -> AudioVisualSample:
        path = Path(descriptor["path"])
        label = descriptor.get("label", "unknown")

        video = self._load_video_frames(path)
        audio = torch.zeros(1, int(self.audio_sr * self.audio_duration_s))
        # UCF-101 has audio in AVI – try to load it
        try:
            import torchaudio
            waveform, sr = torchaudio.load(str(path))
            if sr != self.audio_sr:
                waveform = torchaudio.functional.resample(waveform, sr, self.audio_sr)
            audio = waveform
        except Exception:
            pass

        return AudioVisualSample(
            video_frames=video,
            audio=audio,
            text_caption=label.replace("_", " "),
            metadata={"label": label, "path": str(path)},
        )

