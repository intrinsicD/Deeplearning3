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
  * frame_to_audio     — predict audio from a single frame (image->audio)
  * temporal_order     — given two clips, predict which came first
  * temporal_distance  — predict how far apart two clips are
  * distant_predict    — predict a clip's latent from a temporally distant clip

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
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from omnilatent.config import OmniLatentConfig

# Optional imports — degrade gracefully.
# Catch Exception (not just ImportError) because these libraries can fail
# with RuntimeError, OSError, or AttributeError depending on build config.
try:
    import torchaudio
    from torchaudio.transforms import MelSpectrogram, Resample

    _HAS_TORCHAUDIO = True
except Exception:
    _HAS_TORCHAUDIO = False

try:
    from torchvision.io import read_video

    _HAS_TORCHVISION_IO = True
except Exception:
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
    # Approach 1: Multi-scale temporal sampling tasks
    "temporal_order",
    "temporal_distance",
    "distant_predict",
]

# Temporal distance bucket boundaries (seconds) and labels
TEMPORAL_DISTANCE_BOUNDARIES = [10.0, 60.0, 300.0]  # <10s, 10-60s, 1-5min, >5min


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
# Temporal distance classification
# -------------------------------------------------------------------------
def classify_temporal_distance(
    seconds: float,
    boundaries: list[float] | None = None,
) -> int:
    """Classify a temporal distance (in seconds) into a bucket index.

    Default boundaries: [10, 60, 300] -> buckets: 0=<10s, 1=10-60s, 2=1-5min, 3=>5min.
    """
    if boundaries is None:
        boundaries = TEMPORAL_DISTANCE_BOUNDARIES
    for i, b in enumerate(boundaries):
        if seconds < b:
            return i
    return len(boundaries)


# -------------------------------------------------------------------------
# Video clip index
# -------------------------------------------------------------------------
class _ClipIndex:
    """Pre-computed flat index mapping integer indices to (video_path,
    start_sec) pairs for efficient random-access clip loading.

    Also maintains a per-video index for multi-clip temporal sampling.
    """

    def __init__(
        self,
        video_paths: list[Path],
        clip_duration: float,
        clip_stride: float,
    ) -> None:
        self.entries: list[tuple[Path, float]] = []
        self.transcripts: dict[str, str] = {}
        # Per-video index: video_path_str -> list of (flat_index, start_sec)
        self.video_clips: dict[str, list[tuple[int, float]]] = defaultdict(list)
        # Map from flat index to video_path_str for efficient lookup
        self._idx_to_video: dict[int, str] = {}

        for vp in video_paths:
            # Try to read video duration without loading frames
            duration = self._probe_duration(vp)
            if duration is None or duration < clip_duration:
                continue

            # Index clips with sliding window
            t = 0.0
            while t + clip_duration <= duration + 0.01:
                flat_idx = len(self.entries)
                self.entries.append((vp, t))
                vp_str = str(vp)
                self.video_clips[vp_str].append((flat_idx, t))
                self._idx_to_video[flat_idx] = vp_str
                t += clip_stride

            # Look for transcript
            transcript = _find_transcript(vp)
            if transcript:
                self.transcripts[str(vp)] = transcript

    @staticmethod
    def _probe_duration(path: Path) -> float | None:
        """Get video duration in seconds.

        Strategy:
        1. Read a tiny clip and check info dict for total duration / frame count.
        2. Fall back to probing at ascending offsets to bracket the duration.
        """
        try:
            _, _, info = read_video(
                str(path), start_pts=0, end_pts=0.01, pts_unit="sec"
            )
            fps = info.get("video_fps", 30)

            # Some backends populate total number of video frames
            if "video_total_frames" in info and info["video_total_frames"] > 0:
                return info["video_total_frames"] / fps

            # Fallback: probe at ascending offsets to find the largest
            # offset that still returns frames.
            last_success = 0.0
            for probe_time in [1, 10, 60, 600, 3600]:
                try:
                    v, _, _ = read_video(
                        str(path),
                        start_pts=probe_time,
                        end_pts=probe_time + 0.01,
                        pts_unit="sec",
                    )
                    if v.shape[0] > 0:
                        last_success = probe_time
                    else:
                        # Video shorter than this probe — duration is between
                        # last_success and probe_time.  Use last_success as a
                        # conservative lower bound.
                        break
                except Exception:
                    break

            # Return the last known-good offset; the video is *at least*
            # this long (it may be slightly longer, but this is a safe
            # lower bound for clip indexing).
            return max(last_success, 0.01)
        except Exception:
            return None

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[Path, float, str | None]:
        path, start = self.entries[idx]
        transcript = self.transcripts.get(str(path))
        return path, start, transcript

    def get_same_video_clips(self, idx: int) -> list[tuple[int, float]]:
        """Return all (flat_idx, start_sec) pairs for the same video as idx."""
        vp_str = self._idx_to_video.get(idx)
        if vp_str is None:
            return []
        return self.video_clips[vp_str]

    def sample_context_clip(
        self, idx: int, min_distance: float = 0.0, max_distance: float = float("inf")
    ) -> tuple[int, float] | None:
        """Sample a clip from the same video within a temporal distance range.

        Returns (flat_idx, temporal_distance_seconds) or None if no valid clip.
        """
        path, anchor_start = self.entries[idx]
        clips = self.get_same_video_clips(idx)
        candidates = []
        for flat_idx, start_sec in clips:
            if flat_idx == idx:
                continue
            dist = abs(start_sec - anchor_start)
            if min_distance <= dist <= max_distance:
                candidates.append((flat_idx, dist))
        if not candidates:
            return None
        return random.choice(candidates)


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
        """Convert (T, H, W, C) uint8 -> (C, T_target, H_target, W_target) float."""
        c = self.config
        # (T, H, W, C) -> (T, C, H, W) float
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

        # (T, C, H, W) -> (C, T, H, W)
        return frames.permute(1, 0, 2, 3)

    def _process_audio(
        self, audio: torch.Tensor, info: dict
    ) -> torch.Tensor | None:
        """Convert raw audio waveform -> mel spectrogram (n_mels, T_frames)."""
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
# Approach 1: Multi-Scale Temporal Pair Dataset
# -------------------------------------------------------------------------
class TemporalPairDataset(Dataset):
    """Dataset that samples pairs of clips at multiple timescales.

    For each sample, picks an anchor clip and a context clip from the same
    video at a random distance.  Returns both clips along with temporal
    metadata (order, distance bucket).

    Tasks:
      * temporal_order    — binary: did anchor come before context?
      * temporal_distance — classify distance into buckets (<10s, 10-60s, 1-5m, >5m)
      * distant_predict   — predict context clip's video from anchor's video

    This dataset wraps a VideoWatchingDataset's clip index and reuses its
    processing methods.
    """

    def __init__(
        self,
        video_dir: str | Path,
        config: OmniLatentConfig,
        clip_duration: float = 2.0,
        clip_stride: float = 1.0,
        tokenizer_fn=None,
    ) -> None:
        if not _HAS_TORCHVISION_IO:
            raise ImportError(
                "torchvision.io.read_video is required. "
                "Install torchvision with video support."
            )

        self.config = config
        self.clip_duration = clip_duration
        self.tokenizer_fn = tokenizer_fn

        video_dir = Path(video_dir)
        video_paths = sorted(
            p for p in video_dir.rglob("*")
            if p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not video_paths:
            raise FileNotFoundError(f"No video files found in {video_dir}")

        self.index = _ClipIndex(video_paths, clip_duration, clip_stride)
        if len(self.index) == 0:
            raise RuntimeError("Could not extract any clips from videos.")

        # Only keep clips that have at least one other clip in the same video
        self._valid_indices = [
            i for i in range(len(self.index))
            if len(self.index.get_same_video_clips(i)) > 1
        ]

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
            f"TemporalPairDataset: {len(self.index)} clips, "
            f"{len(self._valid_indices)} with temporal pairs"
        )

    def __len__(self) -> int:
        return len(self._valid_indices)

    def _load_and_process_clip(
        self, flat_idx: int
    ) -> dict[str, torch.Tensor]:
        """Load and process a single clip by flat index."""
        path, start_sec, transcript = self.index[flat_idx]
        end_sec = start_sec + self.clip_duration
        try:
            video, audio, info = read_video(
                str(path), start_pts=start_sec, end_pts=end_sec, pts_unit="sec"
            )
        except Exception:
            return {}

        sample: dict[str, torch.Tensor] = {}
        c = self.config

        if video is not None and video.shape[0] > 0:
            # Process video frames
            frames = video.permute(0, 3, 1, 2).float() / 255.0
            T = frames.shape[0]
            target_t = c.video_max_frames
            if T >= target_t:
                indices = torch.linspace(0, T - 1, target_t).long()
                frames = frames[indices]
            else:
                pad = target_t - T
                frames = torch.cat(
                    [frames, frames[-1:].expand(pad, -1, -1, -1)], dim=0
                )
            frames = F.interpolate(
                frames, size=(c.video_size, c.video_size),
                mode="bilinear", align_corners=False,
            )
            sample["video"] = frames.permute(1, 0, 2, 3)

        if audio is not None and self.mel_transform is not None and audio.numel() > 0:
            sr = info.get("audio_fps", 44100)
            if audio.dim() == 2 and audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            elif audio.dim() == 1:
                audio = audio.unsqueeze(0)
            if sr != c.audio_sample_rate:
                resampler = Resample(orig_freq=int(sr), new_freq=c.audio_sample_rate)
                audio = resampler(audio)
            mel = self.mel_transform(audio).squeeze(0)
            mel = torch.log1p(mel)
            if mel.shape[1] < 16:
                mel = F.pad(mel, (0, 16 - mel.shape[1]))
            t = (mel.shape[1] // 4) * 4
            mel = mel[:, :t]
            if mel.shape[1] > c.audio_max_frames:
                mel = mel[:, :c.audio_max_frames]
            sample["audio"] = mel

        return sample

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a pair of clips with temporal metadata.

        Returns dict with:
          anchor_video, anchor_audio: tensors for anchor clip
          context_video, context_audio: tensors for context clip
          temporal_order: 1 if anchor before context, 0 otherwise
          temporal_distance_bucket: int bucket label
          temporal_distance_sec: actual distance in seconds
        """
        anchor_flat_idx = self._valid_indices[idx]
        anchor_path, anchor_start = self.index.entries[anchor_flat_idx]

        # Sample a context clip from the same video
        ctx = self.index.sample_context_clip(anchor_flat_idx)
        if ctx is None:
            # Fallback: return anchor as both
            ctx_flat_idx = anchor_flat_idx
            distance_sec = 0.0
        else:
            ctx_flat_idx, distance_sec = ctx

        ctx_path, ctx_start = self.index.entries[ctx_flat_idx]

        anchor_data = self._load_and_process_clip(anchor_flat_idx)
        context_data = self._load_and_process_clip(ctx_flat_idx)

        sample: dict[str, torch.Tensor] = {}

        # Anchor clip
        if "video" in anchor_data:
            sample["anchor_video"] = anchor_data["video"]
        if "audio" in anchor_data:
            sample["anchor_audio"] = anchor_data["audio"]

        # Context clip
        if "video" in context_data:
            sample["context_video"] = context_data["video"]
        if "audio" in context_data:
            sample["context_audio"] = context_data["audio"]

        # Temporal metadata
        sample["temporal_order"] = torch.tensor(
            1 if anchor_start < ctx_start else 0, dtype=torch.long
        )
        sample["temporal_distance_bucket"] = torch.tensor(
            classify_temporal_distance(distance_sec), dtype=torch.long
        )
        sample["temporal_distance_sec"] = torch.tensor(
            distance_sec, dtype=torch.float32
        )

        # Fallback
        if not any(k.startswith("anchor_") for k in sample):
            c = self.config
            sample["anchor_video"] = torch.zeros(3, c.video_max_frames, c.video_size, c.video_size)
            sample["context_video"] = torch.zeros(3, c.video_max_frames, c.video_size, c.video_size)

        return sample


# -------------------------------------------------------------------------
# Approach 2: Clip Sequence Dataset (for hierarchical temporal model)
# -------------------------------------------------------------------------
class ClipSequenceDataset(Dataset):
    """Dataset that returns sequences of consecutive clips for training
    the hierarchical temporal transformer.

    Each sample is a sequence of N consecutive clips from the same video,
    returned as pre-processed video and audio tensors.  The temporal
    transformer then processes these clip latents as a sequence.

    Args:
        video_dir: path to video files directory.
        config: OmniLatentConfig.
        clip_duration: duration of each clip in seconds.
        clip_stride: stride between consecutive clips.
        seq_length: number of consecutive clips per sample.
    """

    def __init__(
        self,
        video_dir: str | Path,
        config: OmniLatentConfig,
        clip_duration: float = 2.0,
        clip_stride: float = 2.0,
        seq_length: int = 30,
        tokenizer_fn=None,
    ) -> None:
        if not _HAS_TORCHVISION_IO:
            raise ImportError(
                "torchvision.io.read_video is required. "
                "Install torchvision with video support."
            )

        self.config = config
        self.clip_duration = clip_duration
        self.seq_length = seq_length
        self.tokenizer_fn = tokenizer_fn

        video_dir = Path(video_dir)
        video_paths = sorted(
            p for p in video_dir.rglob("*")
            if p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not video_paths:
            raise FileNotFoundError(f"No video files found in {video_dir}")

        self.index = _ClipIndex(video_paths, clip_duration, clip_stride)

        # Build sequences: contiguous runs of clips from the same video
        self._sequences: list[list[int]] = []
        for vp_str, clips in self.index.video_clips.items():
            # Sort by start time
            sorted_clips = sorted(clips, key=lambda x: x[1])
            flat_indices = [c[0] for c in sorted_clips]
            # Slide a window of seq_length across the video's clips
            if len(flat_indices) >= seq_length:
                for start in range(len(flat_indices) - seq_length + 1):
                    self._sequences.append(
                        flat_indices[start:start + seq_length]
                    )
            elif len(flat_indices) >= 2:
                # Use what we have (shorter sequences)
                self._sequences.append(flat_indices)

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
            f"ClipSequenceDataset: {len(self.index)} clips, "
            f"{len(self._sequences)} sequences of up to {seq_length} clips"
        )

    def __len__(self) -> int:
        return len(self._sequences)

    def _load_clip_video(self, flat_idx: int) -> torch.Tensor | None:
        """Load and process video frames for a single clip."""
        path, start_sec, _ = self.index[flat_idx]
        end_sec = start_sec + self.clip_duration
        try:
            video, _, _ = read_video(
                str(path), start_pts=start_sec, end_pts=end_sec, pts_unit="sec"
            )
        except Exception:
            return None

        if video is None or video.shape[0] == 0:
            return None

        c = self.config
        frames = video.permute(0, 3, 1, 2).float() / 255.0
        T = frames.shape[0]
        target_t = c.video_max_frames
        if T >= target_t:
            indices = torch.linspace(0, T - 1, target_t).long()
            frames = frames[indices]
        else:
            pad = target_t - T
            frames = torch.cat(
                [frames, frames[-1:].expand(pad, -1, -1, -1)], dim=0
            )
        frames = F.interpolate(
            frames, size=(c.video_size, c.video_size),
            mode="bilinear", align_corners=False,
        )
        return frames.permute(1, 0, 2, 3)  # (C, T, H, W)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a sequence of clip video tensors.

        Returns dict with:
          clip_videos: (N, C, T, H, W) — N clips stacked
          seq_length: actual number of valid clips
          clip_mask: (N,) bool mask for valid clips
        """
        flat_indices = self._sequences[idx]
        c = self.config
        max_n = self.seq_length
        actual_n = len(flat_indices)

        videos = []
        mask = []
        for fi in flat_indices:
            v = self._load_clip_video(fi)
            if v is not None:
                videos.append(v)
                mask.append(True)
            else:
                # Pad with zeros
                videos.append(
                    torch.zeros(3, c.video_max_frames, c.video_size, c.video_size)
                )
                mask.append(False)

        # Pad to max_n if sequence is shorter
        while len(videos) < max_n:
            videos.append(
                torch.zeros(3, c.video_max_frames, c.video_size, c.video_size)
            )
            mask.append(False)

        return {
            "clip_videos": torch.stack(videos[:max_n]),   # (N, C, T, H, W)
            "seq_length": torch.tensor(actual_n, dtype=torch.long),
            "clip_mask": torch.tensor(mask[:max_n], dtype=torch.bool),
        }


# -------------------------------------------------------------------------
# Collation for task-based samples
# -------------------------------------------------------------------------
def collate_video_watching(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate for VideoWatchingDataset.

    Delegates to the standard multi-modal collation from data.py.
    Each sample is a dict of modality_name -> tensor, and the collator
    pads text/audio and stacks image/video.
    """
    from omnilatent.training.data import collate_multimodal
    return collate_multimodal(batch)


def collate_temporal_pairs(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate for TemporalPairDataset.

    Stacks anchor and context clips separately, along with temporal metadata.
    """
    result: dict[str, torch.Tensor] = {}

    # Stack tensors that are present in all samples
    keys = set(batch[0].keys())
    for key in keys:
        tensors = [s[key] for s in batch if key in s]
        if not tensors:
            continue
        if tensors[0].dim() == 0:
            # Scalars
            result[key] = torch.stack(tensors)
        else:
            # For audio with variable length, pad
            if "audio" in key:
                max_t = max(t.shape[-1] for t in tensors)
                padded = []
                for t in tensors:
                    if t.shape[-1] < max_t:
                        t = F.pad(t, (0, max_t - t.shape[-1]))
                    padded.append(t)
                result[key] = torch.stack(padded)
            else:
                result[key] = torch.stack(tensors)

    return result


def collate_clip_sequences(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate for ClipSequenceDataset.

    Stacks sequences into a batch.
    """
    return {
        "clip_videos": torch.stack([s["clip_videos"] for s in batch]),  # (B, N, C, T, H, W)
        "seq_length": torch.stack([s["seq_length"] for s in batch]),    # (B,)
        "clip_mask": torch.stack([s["clip_mask"] for s in batch]),      # (B, N)
    }
