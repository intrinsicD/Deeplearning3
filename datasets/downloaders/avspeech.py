"""AVSpeech audiovisual speech dataset.

* ~150k 3–10 second clips of single-speaker talking faces from YouTube
* CSV from: https://looking-to-listen.github.io/avspeech/explore.html
* Paper: Ephrat et al., Looking to Listen at the Cocktail Party (SIGGRAPH 2018)
* Clips downloaded via yt-dlp
* License: YouTube terms per clip; annotations available for research use
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, List, Optional

import torch

from ..base import AudioVisualSample, BaseAVDataset
from ..registry import register_dataset
from .vggsound import _yt_dlp_batch

# The AVSpeech CSV is available via the project page
_AVSPEECH_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/1HFCKpZIm0SyxQx6r6EoizLDfwH9VTCQ6YMr3Ssbm5Nw"
    "/export?format=csv&gid=0"
)
# Fallback mirror hosted in several academic repos
_AVSPEECH_CSV_MIRROR = (
    "https://raw.githubusercontent.com/google-research/looking-to-listen/master/avspeech_train.csv"
)


@register_dataset("avspeech")
class AVSpeechDataset(BaseAVDataset):
    """AVSpeech – talking face clips for audio-visual speech separation."""

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

        csv_path = data_dir / "avspeech_train.csv"
        if not csv_path.exists():
            # Try Google Sheets export first, fall back to GitHub mirror
            try:
                cls._http_download(_AVSPEECH_CSV_URL, csv_path, "AVSpeech CSV (Google Sheets)")
            except Exception:
                try:
                    cls._http_download(_AVSPEECH_CSV_MIRROR, csv_path, "AVSpeech CSV (mirror)")
                except Exception:
                    print(
                        "[WARN] Could not auto-download AVSpeech CSV.\n"
                        "       Download manually from:\n"
                        "       https://looking-to-listen.github.io/avspeech/explore.html\n"
                        f"       and save to {csv_path}"
                    )
                    return
        else:
            print("[skip] avspeech_train.csv already exists")

        if not shutil.which("yt-dlp"):
            print("[WARN] yt-dlp not found.  pip install yt-dlp")
            return

        rows = _parse_avspeech_csv(csv_path, max_clips)
        clips_dir = data_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        print(f"[AVSpeech] Downloading {len(rows)} clips …")
        _yt_dlp_batch(rows, clips_dir, workers, start_s_col=1, end_s_col=2)

    @classmethod
    def verify(cls, data_dir: str | Path) -> bool:
        return (Path(data_dir) / "avspeech_train.csv").exists()

    def _build_index(self) -> List[Any]:
        csv_path = self.data_dir / "avspeech_train.csv"
        if not csv_path.exists():
            print(f"[WARN] AVSpeech CSV not found at {self.data_dir}. Run download() first.")
            return []

        clips_dir = self.data_dir / "clips"
        rows = _parse_avspeech_csv(csv_path)
        samples = []
        for row in rows:
            yt_id, start_s = row[0], row[1]
            stem = f"{yt_id}_{int(float(start_s)):06d}"
            for ext in (".mp4", ".webm", ".mkv"):
                p = clips_dir / (stem + ext)
                if p.exists():
                    samples.append({"path": p, "yt_id": yt_id})
                    break
        return samples

    def _load_sample(self, descriptor: Any) -> AudioVisualSample:
        path = Path(descriptor["path"])
        video = self._load_video_frames(path)
        audio = self._load_audio_wav(path)
        return AudioVisualSample(
            video_frames=video,
            audio=audio,
            text_caption="speech",   # AVSpeech has no text captions; label = domain
            metadata={"yt_id": descriptor.get("yt_id"), "path": str(path)},
        )


def _parse_avspeech_csv(path: Path, max_rows: Optional[int] = None) -> List[List[str]]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 3 and not row[0].startswith("#"):
                rows.append(row)
            if max_rows and len(rows) >= max_rows:
                break
    return rows

