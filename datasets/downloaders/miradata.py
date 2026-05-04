"""MiraData – large-scale video dataset for video generation.

* ~330k high-quality video clips with structured multi-granularity captions
* Hosted on HuggingFace Hub: mira-space/MiraData
* Covers diverse categories: nature, sports, urban scenes, etc.
* Paper: https://arxiv.org/abs/2407.06358
* License: CC-BY 4.0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

import torch

from ..base import AudioVisualSample, BaseAVDataset
from ..registry import register_dataset


@register_dataset("miradata")
class MiraDataDataset(BaseAVDataset):
    """MiraData via HuggingFace Hub – video + rich textual caption."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        caption_level: str = "short",   # "short" | "medium" | "long"
        **kwargs: Any,
    ) -> None:
        super().__init__(data_dir, split, **kwargs)
        self.caption_level = caption_level

    @classmethod
    def download(
        cls,
        data_dir: str | Path,
        repo_id: str = "mira-space/MiraData",
        **kwargs: Any,
    ) -> None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError("pip install huggingface_hub>=0.23.0")

        print(f"[MiraData] Downloading from HuggingFace Hub: {repo_id} …")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(data_dir),
            ignore_patterns=["*.tar.gz"],   # videos are separate; get metadata first
        )
        print("[MiraData] Metadata downloaded. Videos are stored externally.")
        print(
            "  To download video files, see the MiraData download script:\n"
            "  https://github.com/mira-space/MiraData#download\n"
            "  Or run: python auto_train.py --dataset miradata --download-only"
        )

    @classmethod
    def verify(cls, data_dir: str | Path) -> bool:
        data_dir = Path(data_dir)
        # Metadata JSON or parquet files should be present
        return (
            any(data_dir.glob("*.json"))
            or any(data_dir.glob("*.parquet"))
            or any(data_dir.glob("data/**/*.json"))
        )

    def _build_index(self) -> List[Any]:
        # Try to find metadata files
        meta_files = list(self.data_dir.rglob("*.json"))
        if not meta_files:
            # Try parquet via pandas
            parquet_files = list(self.data_dir.rglob("*.parquet"))
            if parquet_files:
                return self._index_from_parquet(parquet_files)
            print(f"[WARN] MiraData metadata not found at {self.data_dir}. Run download() first.")
            return []

        samples = []
        for mf in sorted(meta_files):
            try:
                with open(mf) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    entries = data
                elif isinstance(data, dict):
                    entries = data.get("data", [data])
                else:
                    continue
                for entry in entries:
                    video_path = self._resolve_video_path(entry)
                    if video_path and video_path.exists():
                        samples.append({"path": video_path, "meta": entry})
            except Exception:
                continue
        return samples

    def _index_from_parquet(self, parquet_files: List[Path]) -> List[Any]:
        try:
            import pandas as pd
        except ImportError:
            print("[WARN] pandas not installed. pip install pandas")
            return []
        samples = []
        for pf in parquet_files:
            df = pd.read_parquet(str(pf))
            for _, row in df.iterrows():
                entry = row.to_dict()
                video_path = self._resolve_video_path(entry)
                if video_path and video_path.exists():
                    samples.append({"path": video_path, "meta": entry})
        return samples

    def _resolve_video_path(self, entry: Any) -> Optional[Path]:
        for key in ("video_path", "video", "file_path", "path"):
            val = entry.get(key)
            if val:
                p = self.data_dir / str(val)
                if p.exists():
                    return p
                p2 = Path(str(val))
                if p2.exists():
                    return p2
        return None

    def _load_sample(self, descriptor: Any) -> AudioVisualSample:
        path = Path(descriptor["path"])
        meta = descriptor.get("meta", {})

        # Caption: prefer structured multi-level captions
        caption_keys = {
            "short":  ["short_caption", "caption", "description", "text"],
            "medium": ["medium_caption", "caption", "description", "text"],
            "long":   ["dense_caption", "long_caption", "caption", "description", "text"],
        }
        keys = caption_keys.get(self.caption_level, caption_keys["short"])
        caption = ""
        for k in keys:
            if meta.get(k):
                caption = str(meta[k])
                break

        video = self._load_video_frames(path)
        audio = self._load_audio_wav(path)

        return AudioVisualSample(
            video_frames=video,
            audio=audio,
            text_caption=caption,
            metadata={**meta, "path": str(path)},
        )

