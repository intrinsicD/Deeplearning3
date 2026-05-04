# `datasets/` – Public Dataset Auto-Training Infrastructure

This package provides automatic download, indexing, and model-compatible adapters
for the following publicly available audiovisual datasets:

| Name | Key | Size | Source |
|------|-----|------|--------|
| UCF-101 | `ucf101` | ~6.5 GB | UCF / direct HTTP |
| HMDB51 | `hmdb51` | ~2 GB | Brown University / direct HTTP |
| VGGSound | `vggsound` | ~1 TB | Oxford / yt-dlp |
| Google AudioSet (balanced) | `audioset` | ~20 GB | Google / yt-dlp |
| Kinetics-400 | `kinetics400` | ~240 GB | DeepMind / yt-dlp |
| Kinetics-700 | `kinetics700` | ~650 GB | DeepMind / yt-dlp |
| MiraData | `miradata` | varies | HuggingFace Hub |
| AVSpeech | `avspeech` | ~150 GB | Google / yt-dlp |

---

## Quick Start

```bash
# Install extra deps
pip install yt-dlp PyYAML pandas

# See available models and datasets
python auto_train.py --list-models
python auto_train.py --list-datasets

# Download VGGSound metadata + a small sample (100 clips)
python auto_train.py --model mmwm --dataset vggsound \
    --data-dir ./data/vggsound --download-only --max-clips 100

# Train MMWM on VGGSound (downloads first if needed)
python auto_train.py --model mmwm --dataset vggsound \
    --data-dir ./data/vggsound --download --steps 50000

# Train HPWM on Kinetics-400
python auto_train.py --model hpwm --dataset kinetics400 \
    --data-dir ./data/kinetics400 --download

# Train LGQ tokenizer on UCF-101
python auto_train.py --model lgq --dataset ucf101 \
    --data-dir ./data/ucf101 --download --steps 100000

# Train Gaussian Encoder on VGGSound frames
python auto_train.py --model gaussian_encoder --dataset vggsound \
    --data-dir ./data/vggsound --epochs 50

# Train MMWM on MiraData
python auto_train.py --model mmwm --dataset miradata \
    --data-dir ./data/miradata --download
```

---

## Architecture

```
datasets/
├── __init__.py             – Public API (build_av_dataset, list_datasets)
├── base.py                 – AudioVisualSample dataclass + BaseAVDataset ABC
├── registry.py             – DATASET_REGISTRY, @register_dataset, build_av_dataset
├── downloaders/
│   ├── ucf101.py           – UCF-101 (direct HTTP, unrar)
│   ├── hmdb51.py           – HMDB51 (direct HTTP, unrar)
│   ├── vggsound.py         – VGGSound + _yt_dlp_batch shared helper
│   ├── audioset.py         – Google AudioSet (balanced train + eval)
│   ├── kinetics.py         – Kinetics-400 and Kinetics-700 (factory pattern)
│   ├── miradata.py         – MiraData via HuggingFace Hub
│   └── avspeech.py         – AVSpeech talking-face clips
├── adapters/
│   ├── mmwm_adapter.py     – AudioVisualSample → MMWM batch dict
│   ├── hpwm_adapter.py     – AudioVisualSample → HPWM frames dict
│   ├── lgq_adapter.py      – AudioVisualSample → [3,H,W] image per frame
│   └── gaussian_encoder_adapter.py  – AudioVisualSample → [C,H,W] single frame
├── configs/
│   ├── mmwm_vggsound.yaml
│   ├── mmwm_audioset.yaml
│   ├── mmwm_miradata.yaml
│   ├── mmwm_avspeech.yaml
│   ├── hpwm_kinetics400.yaml
│   ├── hpwm_ucf101.yaml
│   ├── lgq_ucf101.yaml
│   ├── lgq_kinetics400.yaml
│   └── gaussian_encoder_vggsound.yaml
└── train_scripts/
    ├── train_mmwm.py
    ├── train_hpwm.py
    ├── train_lgq.py
    └── train_gaussian_encoder.py

auto_train.py               – Top-level CLI orchestrator
```

---

## Canonical Sample Format

Every dataset yields `AudioVisualSample`:

```python
@dataclass
class AudioVisualSample:
    video_frames: Tensor   # [T, 3, H, W]  float32 in [0,1]
    audio:        Tensor   # [C, L]         float32 waveform @ 16 kHz
    text_caption: str      # label or dense caption
    metadata:     dict     # dataset-specific extras
```

Default canonical shape: **T=16, H=W=224, SR=16 kHz, 4 s audio**.
Override via constructor kwargs (`n_frames`, `resolution`, `audio_sr`, `audio_duration_s`).

---

## Adding a New Dataset

1. Create `datasets/downloaders/mydata.py`
2. Implement `BaseAVDataset` subclass
3. Decorate with `@register_dataset("mydata")`
4. Add import to `datasets/downloaders/__init__.py`
5. Optionally add YAML configs in `datasets/configs/`

```python
from datasets.base import BaseAVDataset, AudioVisualSample
from datasets.registry import register_dataset

@register_dataset("mydata")
class MyDataset(BaseAVDataset):

    @classmethod
    def download(cls, data_dir, **kwargs):
        ...

    @classmethod
    def verify(cls, data_dir):
        return (Path(data_dir) / "mydata.zip").exists()

    def _build_index(self):
        return [{"path": p} for p in (self.data_dir).rglob("*.mp4")]

    def _load_sample(self, descriptor):
        path = Path(descriptor["path"])
        return AudioVisualSample(
            video_frames=self._load_video_frames(path),
            audio=self._load_audio_wav(path),
            text_caption="my label",
        )
```

---

## Notes on YouTube-sourced Datasets

VGGSound, AudioSet, Kinetics, and AVSpeech are downloaded from YouTube via
[yt-dlp](https://github.com/yt-dlp/yt-dlp).  Some clips may be unavailable
due to takedowns.  The downloaders skip missing clips gracefully.

For large-scale downloads consider:
- Academic torrents (Kinetics-400 torrent via `academictorrents.com`)
- Pre-extracted feature caches from HuggingFace Hub
- University compute clusters with dedicated storage bandwidth

