"""Video-watching dataset: learn from raw video files.

A single video naturally contains three modalities aligned in time:
  * Visual frames  (video / image modality)
  * Audio track    (audio modality)
  * Transcript     (text modality, if .txt/.srt provided alongside video)

This module extracts multi-modal training pairs from video files,
enabling the model to learn cross-modal relationships from unlabelled
video — the richest self-supervised signal available.

Supported training tasks extracted from each video clip:
  * video_recon        — reconstruct the video clip
  * audio_recon        — reconstruct the audio segment
  * video_to_audio     — predict audio from video frames
  * audio_to_video     — predict video frames from audio
  * video_to_text      — predict transcript from video (if available)
  * audio_to_text      — predict transcript from audio (if available)
  * temporal_predict   — predict future frames from past frames
  * frame_to_audio     — predict audio from a single frame (image→audio)

Directory layout expected:
    videos/
      video1.mp4
      video1.txt          # optional transcript (plain text)
      video2.mp4
      video2.srt          # optional subtitles
      subfolder/
        video3.webm
        video3.txt
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from omnilatent.config import OmniLatentConfig

# Optional imports — degrade gracefully
try:
    import torchaudio
    from torchaudio.transforms import MelSpectrogram, Resample

    _HAS_TORCHAUDIO = True
except ImportError:
    _HAS_TORCHAUDIO = False

try:
    from torchvision.io import read_video

    _HAS_TORCHVISION_IO = True
except ImportError:
    _HAS_TORCHVISION_IO = False


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".webm", ".mov", ".flv", ".wmv"}
TRANSCRIPT_EXTENSIONS = {".txt", ".srt", ".vtt"}

# All task names that can be sampled during training
ALL_TASKS = [
    "video_recon",
    "audio_recon",
    "video_to_audio",
    "audio_to_video",
    "temporal_predict",
    "frame_to_audio",
    # text tasks (only when transcripts available)
    "video_to_text",
    "audio_to_text",
]


# -------------------------------------------------------------------------
# Subtitle / transcript parsing
# -------------------------------------------------------------------------
def _parse_srt(text: str) -> str:
    """Extract plain text from an SRT subtitle file."""
    # Remove sequence numbers, timestamps, and blank lines
    lines = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d+$", line):
            continue
        if re.match(r"\d{2}:\d{2}:\d{2}", line):
            continue
        # Strip HTML-like tags (<i>, <b>, etc.)
        line = re.sub(r"<[^>]+>", "", line)
        if line:
            lines.append(line)
    return " ".join(lines)


def _load_transcript(path: Path) -> str | None:
    """Load and parse a transcript file (.txt, .srt, .vtt)."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return None
    if path.suffix in (".srt", ".vtt"):
        return _parse_srt(text)
    return text


def _find_transcript(video_path: Path) -> str | None:
    """Look for a transcript file alongside a video file."""
    stem = video_path.stem
    parent = video_path.parent
    for ext in TRANSCRIPT_EXTENSIONS:
        candidate = parent / f"{stem}{ext}"
        transcript = _load_transcript(candidate)
        if transcript is not None:
            return transcript
    return None


# -------------------------------------------------------------------------
# Simple tokenizer (byte-level, no external dependency)
# -------------------------------------------------------------------------
def _simple_tokenize(text: str, max_len: int, vocab_size: int) -> torch.Tensor:
    """Byte-pair-free tokenization: map UTF-8 bytes to token IDs.

    This is a fallback tokenizer.  For real training, replace with
    SentencePiece / tiktoken / HuggingFace tokenizer.
    Token 0 is reserved for padding.
    """
    encoded = text.encode("utf-8")[:max_len]
    ids = [(b % (vocab_size - 1)) + 1 for b in encoded]  # 1..vocab_size-1
    return torch.tensor(ids, dtype=torch.long)


# -------------------------------------------------------------------------
# Video clip index
# -------------------------------------------------------------------------
class _ClipIndex:
    """Pre-computed flat index mapping integer indices to (video_path,
    start_sec) pairs for efficient random-access clip loading."""

    def __init__(
        self,
        video_paths: list[Path],
        clip_duration: float,
        clip_stride: float,
    ) -> None:
        self.entries: list[tuple[Path, float]] = []
        self.transcripts: dict[str, str] = {}

        for vp in video_paths:
            # Try to read video duration without loading frames
            duration = self._probe_duration(vp)
            if duration is None or duration < clip_duration:
                continue

            # Index clips with sliding window
            t = 0.0
            while t + clip_duration <= duration + 0.01:
                self.entries.append((vp, t))
                t += clip_stride

            # Look for transcript
            transcript = _find_transcript(vp)
            if transcript:
                self.transcripts[str(vp)] = transcript

    @staticmethod
    def _probe_duration(path: Path) -> float | None:
        """Get video duration in seconds.  Falls back to loading a tiny
        portion if metadata probing fails."""
        try:
            # read_video with a 0-length window returns metadata
            _, _, info = read_video(
                str(path), start_pts=0, end_pts=0.01, pts_unit="sec"
            )
            fps = info.get("video_fps", 30)
            # Estimate from file — not perfect but avoids loading the whole file
            # Better: use ffprobe.  For now, load 0.01s and check if it works,
            # then estimate from file size heuristic or try a larger read.
            # We'll use a pragmatic approach: try reading at a far offset.
            for probe_time in [3600, 600, 60, 10, 1]:
                try:
                    v, _, _ = read_video(
                        str(path),
                        start_pts=probe_time,
                        end_pts=probe_time + 0.01,
                        pts_unit="sec",
                    )
                    if v.shape[0] > 0:
                        continue  # video is at least this long
                    else:
                        return float(probe_time)
                except Exception:
                    return float(probe_time)
            return 3600.0  # assume 1hr max
        except Exception:
            return None

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[Path, float, str | None]:
        path, start = self.entries[idx]
        transcript = self.transcripts.get(str(path))
        return path, start, transcript


# -------------------------------------------------------------------------
# VideoWatchingDataset
# -------------------------------------------------------------------------
class VideoWatchingDataset(Dataset):
    """Dataset that learns from raw video files.

    Each __getitem__ call:
      1. Loads a short clip (video frames + audio) from a random position
      2. Picks a random training task
      3. Returns {source_modality: tensor, target_modality: tensor,
                  "task": task_name}

    Args:
        video_dir: path to directory containing video files (searched
            recursively).
        config: OmniLatentConfig for tensor shapes.
        clip_duration: duration of each training clip in seconds.
        clip_stride: stride between clips when indexing (seconds).
            Smaller = more overlap = more clips from same video.
        tasks: which training tasks to sample from.  Defaults to all
            available tasks (text tasks only used when transcripts exist).
        tokenizer_fn: optional custom tokenizer function
            (text: str, max_len: int) -> LongTensor.
    """

    def __init__(
        self,
        video_dir: str | Path,
        config: OmniLatentConfig,
        clip_duration: float = 2.0,
        clip_stride: float = 1.0,
        tasks: Sequence[str] | None = None,
        tokenizer_fn=None,
    ) -> None:
        if not _HAS_TORCHVISION_IO:
            raise ImportError(
                "torchvision.io.read_video is required. "
                "Install torchvision with video support."
            )

        self.config = config
        self.clip_duration = clip_duration
        self.tasks = list(tasks or ALL_TASKS)
        self.tokenizer_fn = tokenizer_fn

        # Discover video files
        video_dir = Path(video_dir)
        video_paths = sorted(
            p
            for p in video_dir.rglob("*")
            if p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not video_paths:
            raise FileNotFoundError(
                f"No video files found in {video_dir}"
            )

        # Build clip index
        self.index = _ClipIndex(video_paths, clip_duration, clip_stride)
        if len(self.index) == 0:
            raise RuntimeError(
                f"Found {len(video_paths)} video files but could not "
                "extract any clips (videos may be too short)."
            )

        # Mel spectrogram transform
        if _HAS_TORCHAUDIO:
            self.mel_transform = MelSpectrogram(
                sample_rate=config.audio_sample_rate,
                n_mels=config.audio_n_mels,
                hop_length=config.audio_hop_length,
                n_fft=1024,
            )
        else:
            self.mel_transform = None

        print(
            f"VideoWatchingDataset: {len(video_paths)} videos, "
            f"{len(self.index)} clips, "
            f"{sum(1 for v in video_paths if _find_transcript(v) is not None)} "
            f"with transcripts"
        )

    def __len__(self) -> int:
        return len(self.index)

    def _load_clip(
        self, path: Path, start_sec: float
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict]:
        """Load a video clip, returning (video_frames, audio_waveform, info).

        video_frames: (T, H, W, C) uint8
        audio_waveform: (channels, samples) float
        """
        end_sec = start_sec + self.clip_duration
        try:
            video, audio, info = read_video(
                str(path),
                start_pts=start_sec,
                end_pts=end_sec,
                pts_unit="sec",
            )
        except Exception:
            return None, None, {}

        if video is not None and video.shape[0] == 0:
            video = None
        if audio is not None and audio.shape[0] == 0:
            audio = None

        return video, audio, info

    def _process_video_frames(self, video: torch.Tensor) -> torch.Tensor:
        """Convert (T, H, W, C) uint8 → (C, T_target, H_target, W_target) float."""
        c = self.config
        # (T, H, W, C) → (T, C, H, W) float
        frames = video.permute(0, 3, 1, 2).float() / 255.0

        T = frames.shape[0]
        target_t = c.video_max_frames

        # Subsample or pad temporal dimension
        if T >= target_t:
            # Uniform temporal subsampling
            indices = torch.linspace(0, T - 1, target_t).long()
            frames = frames[indices]
        else:
            # Pad by repeating last frame
            pad = target_t - T
            frames = torch.cat(
                [frames, frames[-1:].expand(pad, -1, -1, -1)], dim=0
            )

        # Spatial resize
        frames = F.interpolate(
            frames,
            size=(c.video_size, c.video_size),
            mode="bilinear",
            align_corners=False,
        )

        # (T, C, H, W) → (C, T, H, W)
        return frames.permute(1, 0, 2, 3)

    def _process_audio(
        self, audio: torch.Tensor, info: dict
    ) -> torch.Tensor | None:
        """Convert raw audio waveform → mel spectrogram (n_mels, T_frames)."""
        if self.mel_transform is None:
            return None

        # Skip empty audio tracks
        if audio.numel() == 0:
            return None

        c = self.config
        sr = info.get("audio_fps", 44100)

        # Convert to mono
        if audio.dim() == 2 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Resample to target sample rate
        if sr != c.audio_sample_rate:
            resampler = Resample(orig_freq=int(sr), new_freq=c.audio_sample_rate)
            audio = resampler(audio)

        # Compute mel spectrogram
        mel = self.mel_transform(audio)  # (1, n_mels, T)
        mel = mel.squeeze(0)  # (n_mels, T)

        # Log-scale (more perceptually meaningful)
        mel = torch.log1p(mel)

        # Ensure minimum length (pad if needed) and make divisible by 4
        if mel.shape[1] < 16:
            mel = F.pad(mel, (0, 16 - mel.shape[1]))
        t = mel.shape[1]
        t = (t // 4) * 4
        mel = mel[:, :t]

        # Truncate to max
        if mel.shape[1] > c.audio_max_frames:
            mel = mel[:, : c.audio_max_frames]

        return mel

    def _process_frame_as_image(self, video: torch.Tensor) -> torch.Tensor:
        """Extract a single frame as an image (C, H, W)."""
        c = self.config
        # Pick a random frame
        t = random.randint(0, video.shape[0] - 1)
        frame = video[t].permute(2, 0, 1).float() / 255.0  # (C, H, W)
        frame = F.interpolate(
            frame.unsqueeze(0),
            size=(c.image_size, c.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return frame

    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text using custom or built-in tokenizer."""
        c = self.config
        if self.tokenizer_fn is not None:
            return self.tokenizer_fn(text, c.text_max_len)
        return _simple_tokenize(text, c.text_max_len, c.vocab_size)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return all available modalities for this clip as a dict.

        The trainer picks (source, target) modality pairs dynamically,
        so no data is wasted on dropped task groups.  This matches the
        format of SyntheticMultiModalDataset.
        """
        path, start_sec, transcript = self.index[idx]
        video_raw, audio_raw, info = self._load_clip(path, start_sec)

        sample: dict[str, torch.Tensor] = {}

        if video_raw is not None and video_raw.shape[0] > 0:
            sample["video"] = self._process_video_frames(video_raw)
            sample["image"] = self._process_frame_as_image(video_raw)

        if audio_raw is not None:
            audio_tensor = self._process_audio(audio_raw, info)
            if audio_tensor is not None:
                sample["audio"] = audio_tensor

        if transcript is not None:
            sample["text"] = self._tokenize(transcript)

        # Fallback: if nothing was extracted, return a dummy audio tensor
        if not sample:
            sample["audio"] = torch.zeros(self.config.audio_n_mels, 16)

        return sample


# -------------------------------------------------------------------------
# Collation for task-based samples
# -------------------------------------------------------------------------
def collate_video_watching(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate for VideoWatchingDataset.

    Delegates to the standard multi-modal collation from data.py.
    Each sample is a dict of modality_name → tensor, and the collator
    pads text/audio and stacks image/video.
    """
    from omnilatent.training.data import collate_multimodal
    return collate_multimodal(batch)
