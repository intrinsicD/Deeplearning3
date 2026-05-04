"""HMDB51 action recognition dataset downloader.

* 6,766 clips across 51 action categories
* ~2 GB; direct HTTP from Brown University servers
* License: free for academic/research use
* URL: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import torch

from ..base import AudioVisualSample, BaseAVDataset
from ..registry import register_dataset

_HMDB51_URL = (
    "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
)
_HMDB51_SPLITS_URL = (
    "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar"
)


@register_dataset("hmdb51")
class HMDB51Dataset(BaseAVDataset):
    """HMDB51 action recognition – video + action label as caption."""

    @classmethod
    def download(cls, data_dir: str | Path, **kwargs: Any) -> None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        rar = data_dir / "hmdb51_org.rar"
        splits_rar = data_dir / "test_train_splits.rar"

        for url, dest, label in [
            (_HMDB51_URL, rar, "HMDB51 videos (~2 GB)"),
            (_HMDB51_SPLITS_URL, splits_rar, "HMDB51 splits"),
        ]:
            if not dest.exists():
                cls._http_download(url, dest, label)
            else:
                print(f"[skip] {dest.name} already downloaded")

        videos_dir = data_dir / "hmdb51_videos"
        if not videos_dir.exists():
            import subprocess, shutil
            if shutil.which("unrar"):
                subprocess.run(["unrar", "x", str(rar), str(videos_dir)], check=True)
                # Inner rars per class
                for inner_rar in videos_dir.glob("*.rar"):
                    subprocess.run(
                        ["unrar", "x", str(inner_rar), str(videos_dir / inner_rar.stem)],
                        check=True,
                    )
            else:
                print("[WARN] 'unrar' not found. Please manually extract hmdb51_org.rar")

        splits_dir = data_dir / "testTrainMulti_7030_splits"
        if not splits_dir.exists():
            import subprocess, shutil
            if shutil.which("unrar"):
                subprocess.run(["unrar", "x", str(splits_rar), str(data_dir)], check=True)
            else:
                print("[WARN] 'unrar' not found. Please manually extract test_train_splits.rar")

    @classmethod
    def verify(cls, data_dir: str | Path) -> bool:
        data_dir = Path(data_dir)
        return (data_dir / "hmdb51_videos").exists()

    def _build_index(self) -> List[Any]:
        videos_dir = self.data_dir / "hmdb51_videos"
        splits_dir = self.data_dir / "testTrainMulti_7030_splits"

        if not videos_dir.exists():
            print(f"[WARN] HMDB51 not found at {self.data_dir}. Run download() first.")
            return []

        split_id = 1 if self.split == "train" else 2  # 0=all,1=train,2=test

        # Build set of video names for requested split
        allowed: set[str] | None = None
        if splits_dir.exists():
            allowed = set()
            for split_file in splits_dir.glob("*_test_split1.txt"):
                for line in split_file.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) == 2 and int(parts[1]) == split_id:
                        allowed.add(parts[0])

        samples = []
        for avi in sorted(videos_dir.rglob("*.avi")):
            if allowed is not None and avi.name not in allowed:
                continue
            label = avi.parent.name
            samples.append({"path": avi, "label": label})

        return samples

    def _load_sample(self, descriptor: Any) -> AudioVisualSample:
        path = Path(descriptor["path"])
        label = descriptor.get("label", "unknown")
        video = self._load_video_frames(path)
        audio = torch.zeros(1, int(self.audio_sr * self.audio_duration_s))
        return AudioVisualSample(
            video_frames=video,
            audio=audio,
            text_caption=label.replace("_", " "),
            metadata={"label": label, "path": str(path)},
        )

