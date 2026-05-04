"""Per-phase dataset loading for OmniLatent curriculum training.

This module turns the public-dataset registry in ``datasets/`` into a
loader that the curriculum can swap between phases.  Each phase is
backed by its own ``DataLoader``; loaders are built lazily so that a run
which never reaches phase 5 does not pay the cost of preparing phase 5's
dataset.

Two pieces:

1. ``OmniLatentAVAdapter`` — converts the canonical ``AudioVisualSample``
   produced by every public-dataset class into the ``dict[modality ->
   tensor]`` format that ``OmniLatentModel.encode`` expects (matching
   ``SyntheticMultiModalDataset``).  Modalities the source dataset cannot
   provide (e.g. text on UCF-101) are simply omitted; the synthetic-style
   train step picks src/tgt from whatever modalities are present, so the
   curriculum naturally constrains its tasks to what the active dataset
   supports.

2. ``PhaseDataLoader`` — holds one builder callable per phase, lazily
   constructs and caches ``DataLoader`` instances, and exposes a single
   ``next_batch(phase_name)`` entry point that the trainer calls each
   step.  Phase transitions are O(1) once the loader has been built.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from omnilatent.config import OmniLatentConfig


# ---------------------------------------------------------------------------
# OmniLatent adapter — canonical AV sample -> synthetic-style modality dict
# ---------------------------------------------------------------------------


def _simple_byte_tokenize(text: str, max_len: int, vocab_size: int) -> torch.Tensor:
    """Byte-level tokenizer matching ``video_dataset._simple_tokenize``.

    Mirrors the fallback used elsewhere in the codebase so that a model
    trained from synthetic-text or video transcripts sees the same token
    distribution as text from the public datasets.
    """
    encoded = text.encode("utf-8", errors="ignore")[:max_len]
    if not encoded:
        return torch.zeros(1, dtype=torch.long)
    ids = [(b % (vocab_size - 1)) + 1 for b in encoded]
    return torch.tensor(ids, dtype=torch.long)


class OmniLatentAVAdapter(Dataset):
    """Wraps a ``BaseAVDataset`` so its samples match ``SyntheticMultiModalDataset``.

    Output keys (only those the source dataset can provide are present):

      * ``video`` : (C, T, H, W) float32 in roughly [-1, 1]
      * ``image`` : (C, H, W)    float32 — the centre frame of the clip
      * ``audio`` : (n_mels, F)  float32 log-mel spectrogram
      * ``text``  : (L,)         long, byte-level tokens, only if a caption
                                  or class label is available

    Audio is optional: a dataset whose ``audio`` field is all zero (or
    missing) is treated as audio-less and the key is dropped so the
    collate function will not produce a bogus zero-tensor channel.
    """

    def __init__(
        self,
        base: Dataset,
        config: OmniLatentConfig,
        provide_audio: bool = True,
        provide_text: bool = True,
    ) -> None:
        self.base = base
        self.config = config
        self.provide_audio = provide_audio
        self.provide_text = provide_text

        # Build the mel transform lazily — torchaudio is an optional dep.
        self._mel = None
        if self.provide_audio:
            try:
                import torchaudio  # noqa: F401

                # Compute hop so that ~audio_max_frames mel frames fit in
                # audio_duration_s seconds.  With sr=16000 and dur=4s we get
                # 64000 samples; n_frames = samples / hop + 1.
                sr = getattr(base, "audio_sr", 16_000)
                dur = getattr(base, "audio_duration_s", 4.0)
                target_frames = max(64, min(config.audio_max_frames, 256))
                hop = max(1, int(sr * dur / target_frames))
                from torchaudio.transforms import MelSpectrogram

                self._mel = MelSpectrogram(
                    sample_rate=sr,
                    n_fft=1024,
                    hop_length=hop,
                    n_mels=config.audio_n_mels,
                )
            except Exception:
                # torchaudio missing — silently disable audio.
                self.provide_audio = False
                self._mel = None

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base[idx]
        out: Dict[str, torch.Tensor] = {}

        # ------------------------- VIDEO -------------------------
        # AudioVisualSample.video_frames: (T, 3, H, W) in [0, 1].
        v = sample.video_frames
        if v.numel() > 0:
            v = v.float()
            # Resize spatial dims to model resolution.
            target = self.config.video_size
            if v.shape[-1] != target or v.shape[-2] != target:
                v = F.interpolate(
                    v, size=(target, target), mode="bilinear", align_corners=False
                )
            # Sub-sample / pad temporal axis to exactly video_max_frames.
            T = v.shape[0]
            tgt_T = self.config.video_max_frames
            if T >= tgt_T:
                idxs = torch.linspace(0, T - 1, tgt_T).long()
                v = v[idxs]
            elif T > 0:
                pad = tgt_T - T
                v = torch.cat([v, v[-1:].expand(pad, -1, -1, -1)], dim=0)
            # Re-scale to roughly [-1, 1] like the synthetic randn data.
            v = (v * 2.0) - 1.0
            # (T, 3, H, W) -> (3, T, H, W)
            v = v.permute(1, 0, 2, 3).contiguous()
            out["video"] = v
            # Centre-frame image.
            out["image"] = v[:, v.shape[1] // 2].contiguous()

        # ------------------------- AUDIO -------------------------
        if self.provide_audio and self._mel is not None and sample.audio.numel() > 0:
            wav = sample.audio
            # Down-mix to mono.
            if wav.dim() == 2 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            elif wav.dim() == 1:
                wav = wav.unsqueeze(0)
            # Skip the audio key entirely if the waveform is silent — this
            # prevents the model from "training on audio" when the dataset
            # has no real audio (e.g. UCF-101).
            if wav.abs().max() > 1e-6:
                mel = self._mel(wav).squeeze(0)        # (n_mels, F)
                mel = (mel + 1e-9).log10()
                # Pad / trim to a multiple of 4 (encoder stride) and at most
                # audio_max_frames.
                F_max = self.config.audio_max_frames
                if mel.shape[1] > F_max:
                    mel = mel[:, :F_max]
                # Make divisible by 4.
                trim = (mel.shape[1] // 4) * 4
                if trim >= 4:
                    out["audio"] = mel[:, :trim]

        # ------------------------- TEXT --------------------------
        if self.provide_text:
            caption = sample.text_caption or str(
                sample.metadata.get("label", "")
            )
            if caption:
                out["text"] = _simple_byte_tokenize(
                    caption, self.config.text_max_len, self.config.vocab_size
                )

        return out


# ---------------------------------------------------------------------------
# Per-phase loader
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    """Declarative spec for one phase's data source."""

    name: str | List[str]                # registered dataset name(s)
    split: str = "train"
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)
    provide_audio: bool = True
    provide_text: bool = True


class PhaseDataLoader:
    """Lazy multi-phase data loader.

    The trainer calls ``next_batch(phase_name)`` each step.  On the first
    call for a given phase, the loader constructs the underlying
    ``DataLoader`` (which downloads the dataset on first use, via
    ``BaseAVDataset.prepare()``); thereafter the cached loader is reused.
    """

    def __init__(
        self,
        phase_specs: Dict[str, DatasetSpec],
        builder: Callable[[str, DatasetSpec], DataLoader],
    ) -> None:
        self._specs = phase_specs
        self._builder = builder
        self._loaders: Dict[str, DataLoader] = {}
        self._iters: Dict[str, Iterator] = {}

    def has_phase(self, phase_name: str) -> bool:
        return phase_name in self._specs

    def _ensure_phase(self, phase_name: str) -> DataLoader:
        if phase_name not in self._loaders:
            spec = self._specs[phase_name]
            print(f"[phase-loader] Preparing dataset for phase '{phase_name}': {spec.name}")
            self._loaders[phase_name] = self._builder(phase_name, spec)
            self._iters[phase_name] = iter(self._loaders[phase_name])
        return self._loaders[phase_name]

    def next_batch(self, phase_name: str) -> Dict[str, torch.Tensor]:
        self._ensure_phase(phase_name)
        try:
            return next(self._iters[phase_name])
        except StopIteration:
            self._iters[phase_name] = iter(self._loaders[phase_name])
            return next(self._iters[phase_name])

    def describe(self) -> str:
        lines = []
        for name, spec in self._specs.items():
            ds = ", ".join(spec.name) if isinstance(spec.name, list) else spec.name
            lines.append(f"  {name:15s} -> {ds} (split={spec.split})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default builder — uses the ``datasets/`` registry
# ---------------------------------------------------------------------------


def make_default_builder(
    config: OmniLatentConfig,
    data_root: str,
    batch_size: int,
    num_workers: int = 2,
    max_samples: Optional[int] = None,
    auto_download: bool = False,
) -> Callable[[str, DatasetSpec], DataLoader]:
    """Return a builder that turns a ``DatasetSpec`` into a ``DataLoader``.

    The builder calls ``datasets.build_av_dataset`` on the registered name
    (or a list of names, which are concatenated), wraps each dataset in
    the OmniLatent adapter, and uses ``collate_multimodal`` so the batch
    structure matches what the synthetic train step expects.

    ``auto_download=True`` calls the dataset's ``download()`` classmethod
    before building the index.  Without it we assume the dataset files
    are already present at ``data_root/<name>``.
    """
    from datasets import build_av_dataset, DATASET_REGISTRY
    from omnilatent.training.data import collate_multimodal
    from pathlib import Path

    def _build_one(spec_name: str, spec: DatasetSpec) -> Dataset:
        if spec_name not in DATASET_REGISTRY:
            raise KeyError(
                f"Dataset '{spec_name}' not registered.  "
                f"Available: {sorted(DATASET_REGISTRY.keys())}"
            )
        cls = DATASET_REGISTRY[spec_name]
        ddir = Path(data_root) / spec_name
        if auto_download and not cls.verify(ddir):
            cls.download(ddir, **spec.extra_kwargs)
        kwargs = {
            "n_frames": min(config.video_max_frames * 2, 32),
            "resolution": config.video_size,
            **spec.extra_kwargs,
        }
        if max_samples is not None:
            kwargs["max_samples"] = max_samples
        return build_av_dataset(spec_name, str(ddir), split=spec.split, **kwargs)

    def builder(phase_name: str, spec: DatasetSpec) -> DataLoader:
        names: Sequence[str] = (
            spec.name if isinstance(spec.name, (list, tuple)) else [spec.name]
        )
        bases = [_build_one(n, spec) for n in names]
        merged = bases[0] if len(bases) == 1 else ConcatDataset(bases)
        adapted = OmniLatentAVAdapter(
            merged,
            config,
            provide_audio=spec.provide_audio,
            provide_text=spec.provide_text,
        )
        return DataLoader(
            adapted,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_multimodal,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    return builder


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def load_phase_specs_from_yaml(path: str) -> Dict[str, DatasetSpec]:
    """Load a curriculum YAML into ``{phase_name: DatasetSpec}``.

    Schema::

        phases:
          warmup:
            dataset: ucf101         # str or list of str
            split: train
            kwargs: {}              # forwarded to dataset constructor
            audio: false            # default true
            text:  false            # default true
          cross_modal:
            dataset: vggsound
            ...
    """
    import yaml

    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    raw_phases = cfg.get("phases", {})
    specs: Dict[str, DatasetSpec] = {}
    for name, body in raw_phases.items():
        specs[name] = DatasetSpec(
            name=body["dataset"],
            split=body.get("split", "train"),
            extra_kwargs=body.get("kwargs", {}) or {},
            provide_audio=bool(body.get("audio", True)),
            provide_text=bool(body.get("text", True)),
        )
    return specs


# ---------------------------------------------------------------------------
# Synthetic fallback (lets the curriculum-config code path be tested
# without any internet datasets installed).
# ---------------------------------------------------------------------------


def make_synthetic_phase_loader(
    config: OmniLatentConfig,
    batch_size: int,
    phase_names: Sequence[str],
    length: int = 1024,
) -> PhaseDataLoader:
    """Build a ``PhaseDataLoader`` whose every phase points at the synthetic
    dataset.  Useful for testing the per-phase swap mechanism without
    downloading anything.

    Different phases use different ``paired`` flags so the test still
    exercises swapping (warmup gets paired data, later phases get random
    subsets).
    """
    from omnilatent.training.data import (
        SyntheticMultiModalDataset,
        collate_multimodal,
    )

    def builder(phase_name: str, spec: DatasetSpec) -> DataLoader:
        # Use the spec.extra_kwargs to vary 'paired' across phases.
        paired = bool(spec.extra_kwargs.get("paired", True))
        ds = SyntheticMultiModalDataset(config, length=length, paired=paired)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_multimodal,
            num_workers=0,
            drop_last=True,
        )

    specs = {
        name: DatasetSpec(
            name=f"synthetic_{name}",
            extra_kwargs={"paired": (i % 2 == 0)},
        )
        for i, name in enumerate(phase_names)
    }
    return PhaseDataLoader(specs, builder)
