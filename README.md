# OmniLatent

All-to-all multimodal AI with **Latent Neural Hooks** — trainable on a single GPU with 8 GB VRAM.

Processes **text, native audio, images, and full video** in a unified latent space. Any input modality can produce any output modality (16 combinations).

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install torch torchaudio torchvision einops
```

## Quick Start

```python
import torch
from omnilatent import OmniLatentConfig, OmniLatentModel

config = OmniLatentConfig()
model = OmniLatentModel(config).cuda()

# Image → Text
image = torch.randn(1, 3, 224, 224).cuda()
result = model("image", image, "text")
logits = result["output"]  # (1, 196, 32000)

# Text → Image
tokens = torch.randint(1, 32000, (1, 64)).cuda()
result = model("text", tokens, "image")
image_out = result["output"]  # (1, 3, 224, 224)

# Self-reconstruction (autoencoder mode)
result = model.reconstruct("image", image)

# Multiple inputs at once
results = model.forward_multimodal(
    inputs={"text": tokens, "image": image},
    target_modalities=["text", "image"],
)
```

## Training

### With synthetic data (verification / benchmarking)

```bash
python train.py
```

This trains on randomly generated data to verify the full pipeline works. Useful for checking memory usage and speed on your hardware.

### CLI options

```bash
python train.py --help

# Smaller model for tighter memory
python train.py --dim 512 --layers 8 --heads 8

# Disable mixed precision
python train.py --no-amp

# Short run
python train.py --steps 500 --batch-size 2 --log-interval 10
```

### Training with your own data

Replace the synthetic dataset with a real one. The dataloader expects batches as `dict[str, Tensor]` where keys are modality names:

```python
from omnilatent import OmniLatentConfig, OmniLatentModel
from omnilatent.training import Trainer
from omnilatent.training.data import build_dataloader, collate_multimodal
from torch.utils.data import DataLoader

config = OmniLatentConfig(
    batch_size=4,
    learning_rate=3e-4,
    max_steps=100_000,
    mixed_precision=True,
    gradient_checkpointing=True,
)

model = OmniLatentModel(config)
dataloader = DataLoader(
    your_dataset,
    batch_size=config.batch_size,
    collate_fn=collate_multimodal,
    shuffle=True,
)

trainer = Trainer(model, config, dataloader)
trainer.train()
```

Each batch should be a dict with any subset of these keys:

| Key | Shape | Description |
|---|---|---|
| `"text"` | `(B, T)` long | Token IDs (0 = padding) |
| `"audio"` | `(B, 128, T_frames)` float | Mel spectrogram |
| `"image"` | `(B, 3, 224, 224)` float | RGB image |
| `"video"` | `(B, 3, T_frames, 112, 112)` float | RGB video |

The trainer randomly samples (source, target) modality pairs each step, giving uniform coverage of all modality combinations over time.

### Memory budget (default 768-dim, 12-layer config)

| Component | Size |
|---|---|
| Parameters (FP16) | ~280 MB |
| Gradients (FP16) | ~280 MB |
| Optimizer states (FP32) | ~1.1 GB |
| Master weights (FP32) | ~560 MB |
| Activations (checkpointed) | ~1–2 GB |
| **Total** | **~3.7–4.7 GB** |

Fits comfortably in 8 GB VRAM.

## Latent Neural Hooks

The core extensibility mechanism. Hooks are learnable latent vectors injected **directly into the transformer's self-attention** — not adapters bolted on the side.

### How they work

At each targeted transformer layer:
1. Hook tokens are concatenated to the content sequence
2. Full self-attention runs over content + hook tokens together
3. After attention, hook tokens are separated back out
4. A **learned sigmoid gate** (initialized near zero) controls how much hooks influence content tokens
5. Hook state persists across layers, accumulating information

### Using hooks

```python
from omnilatent import LatentNeuralHook

# Create a hook with 8 latent tokens targeting layers 0–5
hook = LatentNeuralHook(
    name="style_control",
    num_tokens=8,
    dim=768,                      # must match model hidden_dim
    target_layers=[0, 1, 2, 3, 4, 5],
    gate_bias_init=-4.0,          # starts nearly silent (sigmoid ≈ 0.018)
    use_transform=True,           # inter-layer MLP for hook state evolution
)

# Register with model
model.register_hook(hook)

# Forward passes now include the hook automatically
result = model("image", image, "text")

# Hook parameters are trainable
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Remove when done
model.remove_hook("style_control")
```

### Use cases

- **New modality**: add an encoder + hook without touching existing code
- **Task-specific behavior**: style transfer, summarization, etc.
- **Knowledge injection**: feed retrieval results directly into attention
- **Interpretability**: probe internal representations at any layer
- **Composition**: multiple hooks active simultaneously, each seeing the others

### Key properties

| Property | Description |
|---|---|
| Participatory | Hooks join self-attention as first-class tokens |
| Bidirectional | Hooks read from AND write to hidden states |
| Gated | Learned gates control influence (safe composition) |
| Persistent | Hook state carries across layers |
| Composable | Multiple hooks work simultaneously |
| Zero-cost removal | Unregister instantly; no model weights changed |

## Architecture

```
Input → [Modality Encoder] → [Target Token + Source Token + Content Tokens]
                                           ↓
                              [Unified Transformer Backbone]
                              (with Latent Neural Hooks at each layer)
                                           ↓
                              [Sequence Adaptation]
                                           ↓
                              [Modality Decoder] → Output
```

### Components

- **Backbone**: 12-layer transformer with RMSNorm, SwiGLU, RoPE, QK-Norm
- **Text encoder/decoder**: learned embeddings (weight-tied) + vocab projection
- **Audio encoder/decoder**: 1-D conv stack on mel spectrograms
- **Image encoder/decoder**: ViT-style 16x16 patch embedding
- **Video encoder/decoder**: 3-D conv spatiotemporal patches (4-frame temporal, 16x16 spatial)

### Scaling the config

```python
# Tiny (for debugging)
config = OmniLatentConfig(hidden_dim=128, num_layers=4, num_heads=4)

# Default (~140M params, fits 8 GB)
config = OmniLatentConfig()

# Larger (needs more VRAM)
config = OmniLatentConfig(hidden_dim=1024, num_layers=16, num_heads=16)
```

## Learning from Video (Curriculum Training)

The most powerful way to train OmniLatent: learn from raw video files. A single video naturally contains **three modalities aligned in time** (visual frames, audio track, optional transcript), providing free cross-modal supervision with no manual labelling.

### What it learns from each video

| Extracted pair | What the model learns |
|---|---|
| Video frames + audio | What sounds go with what visuals |
| Past frames + future frames | Temporal prediction, causality |
| Single frame + audio | Scene-sound association |
| Video + transcript | Visual-language grounding |
| Audio + transcript | Speech understanding |

### Prepare your videos

Put video files (`.mp4`, `.avi`, `.mkv`, `.webm`, etc.) in a directory. Optionally add transcript files alongside each video:

```
videos/
  lecture01.mp4
  lecture01.txt          # plain text transcript (optional)
  nature_doc.mp4
  nature_doc.srt         # SRT subtitles (optional)
  music_video.webm       # no transcript needed
  subfolder/
    clip.mp4
```

### Run curriculum training

```bash
python curriculum_train.py --video-dir /path/to/videos
```

The trainer runs 5 phases automatically:

| Phase | Steps | Tasks | Purpose |
|---|---|---|---|
| 1. Warmup | 10% | video recon, audio recon | Learn to encode/decode each modality |
| 2. Cross-modal | 25% | + video↔audio | Learn audio-visual correspondence |
| 3. Temporal | 25% | + temporal prediction | Learn to predict future from past |
| 4. Grounding | 15% | + video→text, audio→text | Learn language grounding (needs transcripts) |
| 5. Joint | 25% | all tasks together | Unify all capabilities |

### CLI options

```bash
python curriculum_train.py --help

# Quick test run
python curriculum_train.py --video-dir ./videos --total-steps 500

# Custom model size
python curriculum_train.py --video-dir ./videos --dim 512 --layers 8 --heads 8

# Custom phase durations
python curriculum_train.py --video-dir ./videos \
    --warmup-frac 0.05 --crossmodal-frac 0.30 --temporal-frac 0.30

# Resume from checkpoint
python curriculum_train.py --video-dir ./videos --resume checkpoints/checkpoint_step50000.pt

# Test with synthetic data (no videos needed)
python curriculum_train.py --synthetic --total-steps 200
```

### Use in code

```python
from omnilatent import OmniLatentConfig, OmniLatentModel
from omnilatent.training.video_dataset import VideoWatchingDataset, collate_video_watching
from curriculum_train import CurriculumTrainer
from torch.utils.data import DataLoader

config = OmniLatentConfig()
model = OmniLatentModel(config)

dataset = VideoWatchingDataset(
    video_dir="/path/to/videos",
    config=config,
    clip_duration=2.0,   # seconds per clip
    clip_stride=1.0,     # overlap between clips
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_video_watching,
    num_workers=2,
)

trainer = CurriculumTrainer(
    model=model,
    config=config,
    dataloader=dataloader,
    total_steps=100_000,
    save_dir="checkpoints",
)
trainer.train()
```

### Custom tokenizer

The video pipeline includes a simple byte-level tokenizer for transcripts. For better text quality, plug in your own:

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor("tokenizer.model")

dataset = VideoWatchingDataset(
    video_dir="./videos",
    config=config,
    tokenizer_fn=lambda text, max_len: torch.tensor(
        sp.encode(text)[:max_len], dtype=torch.long
    ),
)
```

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

45 tests covering:
- Model construction and all 16 modality pair forward passes
- Latent Neural Hook lifecycle, composition, and gradient flow
- Gradient flow verification (no dead branches, no NaN/Inf)
- Training stability (loss decreases over steps)
- Gradient checkpointing equivalence
- Video watching pipeline (transcript parsing, tokenization, collation)
- Curriculum trainer (phase transitions, synthetic data integration)

## Running the demo

```bash
python demo.py
```

Demonstrates self-reconstruction, cross-modal translation, hook registration, multi-modal forward, and gradient flow verification.
