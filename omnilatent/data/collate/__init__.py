"""Collate functions for multi-modal batching.

This module is the bridge between the unified streaming layer
(:class:`~omnilatent.data.sample.MultiModalSample`, which carries a *string*
for text and raw per-sample tensors for media) and the
``dict[str, torch.Tensor]`` batches the OmniLatent trainer consumes.

Before this bridge existed the trainer could not consume manifest data at all
(Audit.md A1): ``StreamingMultiModalDataset`` yields ``MultiModalSample``
objects, while the default ``DataLoader`` collate raised ``TypeError`` on them.

Usage::

    from torch.utils.data import DataLoader
    from omnilatent.data import StreamingMultiModalDataset
    from omnilatent.data.collate import build_sample_collator

    loader = DataLoader(
        StreamingMultiModalDataset(manifest),
        batch_size=4,
        collate_fn=build_sample_collator(config),
    )
    for batch in loader:            # batch is dict[str, Tensor]
        ...

Per-modality batch sizes may differ when samples carry different modalities
(a text-only sample and an image-only sample produce a batch with a 1-row
``text`` tensor and a 1-row ``image`` tensor). That is intentional and valid
for self-reconstruction training; cross-modal steps that need aligned pairs
are the trainer's responsibility (see Audit.md A3 / W0.3).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn.functional as F

from omnilatent.config import OmniLatentConfig
from omnilatent.data.sample import MultiModalSample

Tokenizer = Callable[[str, int, int], torch.Tensor]

SampleCollator = Callable[[List[MultiModalSample]], dict[str, torch.Tensor]]


def byte_tokenize(text: str, max_len: int, vocab_size: int) -> torch.Tensor:
    """Canonical byte-level tokenizer.

    Matches ``omnilatent.training.multi_dataset._simple_byte_tokenize`` and
    ``video_dataset._simple_tokenize`` so manifest text, synthetic text, and
    video transcripts share one token distribution. Token id 0 is reserved
    for padding.
    """
    encoded = text.encode("utf-8", errors="ignore")[:max_len]
    if not encoded:
        return torch.zeros(1, dtype=torch.long)
    ids = [(b % (vocab_size - 1)) + 1 for b in encoded]
    return torch.tensor(ids, dtype=torch.long)


def _standardize_image(image: torch.Tensor, config: OmniLatentConfig) -> torch.Tensor:
    """Coerce a (C, H, W) image tensor to (image_channels, image_size, image_size)."""
    if image.dim() != 3:
        raise ValueError(f"image sample must be (C, H, W); got shape {tuple(image.shape)}")
    img = image.float()
    c = img.shape[0]
    if c != config.image_channels:
        if c == 1 and config.image_channels == 3:
            img = img.expand(3, -1, -1)
        elif c >= config.image_channels:
            img = img[: config.image_channels]
        else:
            raise ValueError(
                f"image has {c} channels, cannot map to {config.image_channels}"
            )
    if img.shape[-2:] != (config.image_size, config.image_size):
        img = F.interpolate(
            img.unsqueeze(0),
            size=(config.image_size, config.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return img


def _standardize_video(video: torch.Tensor, config: OmniLatentConfig) -> torch.Tensor:
    """Coerce a (C, T, H, W) video tensor to the configured channels/size/frames."""
    if video.dim() != 4:
        raise ValueError(f"video sample must be (C, T, H, W); got shape {tuple(video.shape)}")
    vid = video.float()
    c, t, h, w = vid.shape
    if c != config.video_channels:
        if c == 1 and config.video_channels == 3:
            vid = vid.expand(3, -1, -1, -1)
        elif c >= config.video_channels:
            vid = vid[: config.video_channels]
        else:
            raise ValueError(
                f"video has {c} channels, cannot map to {config.video_channels}"
            )
    # Clamp/pad temporal length to video_max_frames.
    if t > config.video_max_frames:
        vid = vid[:, : config.video_max_frames]
    elif t < config.video_max_frames:
        pad = config.video_max_frames - t
        vid = torch.cat([vid, vid[:, -1:].expand(-1, pad, -1, -1)], dim=1)
    if vid.shape[-2:] != (config.video_size, config.video_size):
        # interpolate spatial dims per frame
        c2, t2 = vid.shape[0], vid.shape[1]
        vid = vid.reshape(c2 * t2, 1, vid.shape[2], vid.shape[3])
        vid = F.interpolate(
            vid, size=(config.video_size, config.video_size), mode="bilinear", align_corners=False
        )
        vid = vid.reshape(c2, t2, config.video_size, config.video_size)
    return vid


def _text_of(sample: MultiModalSample) -> Optional[str]:
    """Return the textual content of a sample (plain text or extracted PDF text)."""
    if sample.text is not None:
        return sample.text
    if sample.pdf_page is not None:
        return sample.pdf_page
    return None


def sample_to_inputs(
    sample: MultiModalSample,
    config: OmniLatentConfig,
    tokenizer: Tokenizer = byte_tokenize,
) -> dict[str, torch.Tensor]:
    """Convert one ``MultiModalSample`` into per-modality model tensors (unbatched)."""
    out: dict[str, torch.Tensor] = {}
    text = _text_of(sample)
    if text is not None:
        out["text"] = tokenizer(text, config.text_max_len, config.vocab_size)
    if sample.image is not None:
        out["image"] = _standardize_image(sample.image, config)
    if sample.audio is not None:
        audio = sample.audio.float()
        if audio.dim() != 2:
            raise ValueError(
                f"audio sample must be (n_mels, T); got shape {tuple(audio.shape)}"
            )
        out["audio"] = audio
    if sample.video is not None:
        out["video"] = _standardize_video(sample.video, config)
    return out


def build_sample_collator(
    config: OmniLatentConfig,
    tokenizer: Tokenizer = byte_tokenize,
) -> SampleCollator:
    """Build a ``DataLoader`` ``collate_fn`` that turns a list of
    ``MultiModalSample`` into a ``dict[str, Tensor]`` batch."""

    # Lazy import avoids an import-time cycle (data layer ← training).
    from omnilatent.training.data import ROW_SUFFIX

    def collate(batch: List[MultiModalSample]) -> dict[str, torch.Tensor]:
        converted = [sample_to_inputs(s, config, tokenizer) for s in batch]
        result: dict[str, torch.Tensor] = {}

        def _rows(mod: str) -> torch.Tensor:
            return torch.tensor([i for i, c in enumerate(converted) if mod in c], dtype=torch.long)

        # Text: pad to the longest sequence in the batch (id 0 = padding).
        text_tokens = [c["text"] for c in converted if "text" in c]
        if text_tokens:
            max_len = max(t.shape[0] for t in text_tokens)
            padded = torch.zeros(len(text_tokens), max_len, dtype=torch.long)
            for i, t in enumerate(text_tokens):
                padded[i, : t.shape[0]] = t
            result["text"] = padded
            result["text" + ROW_SUFFIX] = _rows("text")

        # Audio: pad mel frames to the longest in the batch.
        audios = [c["audio"] for c in converted if "audio" in c]
        if audios:
            n_mels = audios[0].shape[0]
            max_frames = max(a.shape[1] for a in audios)
            padded_a = torch.zeros(len(audios), n_mels, max_frames)
            for i, a in enumerate(audios):
                padded_a[i, :, : a.shape[1]] = a
            result["audio"] = padded_a
            result["audio" + ROW_SUFFIX] = _rows("audio")

        # Image / video: fixed shape after standardization → stack directly.
        images = [c["image"] for c in converted if "image" in c]
        if images:
            result["image"] = torch.stack(images)
            result["image" + ROW_SUFFIX] = _rows("image")
        videos = [c["video"] for c in converted if "video" in c]
        if videos:
            result["video"] = torch.stack(videos)
            result["video" + ROW_SUFFIX] = _rows("video")

        return result

    return collate


def collate_multimodal_samples(
    batch: List[MultiModalSample],
    config: OmniLatentConfig,
    tokenizer: Tokenizer = byte_tokenize,
) -> dict[str, torch.Tensor]:
    """Convenience one-shot collate (prefer :func:`build_sample_collator` for DataLoaders)."""
    return build_sample_collator(config, tokenizer)(batch)


__all__ = [
    "Tokenizer",
    "SampleCollator",
    "byte_tokenize",
    "sample_to_inputs",
    "build_sample_collator",
    "collate_multimodal_samples",
]
