"""Google AudioSet downloader.

* ~2M 10-second YouTube clips annotated with 527 audio event categories
* Balanced subset: ~22k clips (~20 GB); Unbalanced: ~2M clips
* CSV manifests: https://research.google.com/audioset/download.html
* Ontology JSON: https://raw.githubusercontent.com/audioset/ontology/master/ontology.json
* License: CC-BY 4.0 for the annotations; YouTube terms for video content
"""

from __future__ import annotations

import csv
import json
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..base import AudioVisualSample, BaseAVDataset
from ..registry import register_dataset
from .vggsound import _yt_dlp_batch

_BALANCED_TRAIN_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
)
_EVAL_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv"
)
_ONTOLOGY_URL = (
    "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"
)


@register_dataset("audioset")
class AudioSetDataset(BaseAVDataset):
    """Google AudioSet – balanced subset; audio+video + label as caption."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        subset: str = "balanced",   # "balanced" | "unbalanced"
        **kwargs: Any,
    ) -> None:
        super().__init__(data_dir, split, **kwargs)
        self.subset = subset

    @classmethod
    def download(
        cls,
        data_dir: str | Path,
        subset: str = "balanced",
        max_clips: Optional[int] = None,
        workers: int = 4,
        **kwargs: Any,
    ) -> None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        csv_path = data_dir / "balanced_train_segments.csv"
        eval_path = data_dir / "eval_segments.csv"
        ontology_path = data_dir / "ontology.json"

        for url, dest, label in [
            (_BALANCED_TRAIN_URL, csv_path, "AudioSet balanced train CSV"),
            (_EVAL_URL, eval_path, "AudioSet eval CSV"),
            (_ONTOLOGY_URL, ontology_path, "AudioSet ontology JSON"),
        ]:
            if not dest.exists():
                cls._http_download(url, dest, label)
            else:
                print(f"[skip] {dest.name}")

        if not shutil.which("yt-dlp"):
            print("[WARN] yt-dlp not found.  pip install yt-dlp  to download clips.")
            return

        csv_file = csv_path  # balanced
        rows = _parse_audioset_csv(csv_file, max_clips)
        clips_dir = data_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        print(f"[AudioSet] Downloading {len(rows)} clips …")
        _yt_dlp_batch(rows, clips_dir, workers, start_s_col=1, end_s_col=2)

    @classmethod
    def verify(cls, data_dir: str | Path) -> bool:
        data_dir = Path(data_dir)
        return (data_dir / "balanced_train_segments.csv").exists()

    def _load_ontology(self) -> Dict[str, str]:
        """Return {mid: display_name} mapping."""
        path = self.data_dir / "ontology.json"
        if path.exists():
            with open(path) as f:
                entries = json.load(f)
            return {e["id"]: e["name"] for e in entries}
        return {}

    def _build_index(self) -> List[Any]:
        csv_file = self.data_dir / "balanced_train_segments.csv"
        eval_file = self.data_dir / "eval_segments.csv"
        target_file = csv_file if self.split == "train" else eval_file
        if not target_file.exists():
            print(f"[WARN] AudioSet CSV not found. Run download() first.")
            return []

        ontology = self._load_ontology()
        clips_dir = self.data_dir / "clips"
        rows = _parse_audioset_csv(target_file)
        samples = []
        for row in rows:
            yt_id, start_s = row[0], row[1]
            stem = f"{yt_id}_{int(float(start_s)):06d}"
            for ext in (".mp4", ".webm", ".mkv"):
                p = clips_dir / (stem + ext)
                if p.exists():
                    # Labels are comma-separated MID codes in col 3
                    mids = row[3].strip().strip('"').split(",")
                    labels = [ontology.get(m.strip(), m.strip()) for m in mids]
                    caption = ", ".join(labels)
                    samples.append({"path": p, "label": caption, "yt_id": yt_id})
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


def _parse_audioset_csv(path: Path, max_rows: Optional[int] = None) -> List[List[str]]:
    rows = []
    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",", 3)]
            if len(parts) >= 3:
                rows.append(parts)
            if max_rows and len(rows) >= max_rows:
                break
    return rows

