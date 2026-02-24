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

## Training & Inference Guide

This repository contains four distinct networks. Below is a detailed description of how each one is trained and used for inference.

---

### 1. OmniLatent (all-to-all multimodal model)

OmniLatent is the main model. It has four training entry points, each suited to different data and objectives.

#### a) Synthetic data training (`train.py`)

Trains on randomly generated tensors for all four modalities. Useful for verifying the pipeline, benchmarking hardware, and debugging.

```bash
python train.py                          # default: 768-dim, 12 layers, 100k steps
python train.py --dim 512 --layers 8 --heads 8 --steps 5000
python train.py --no-amp --batch-size 2  # disable mixed precision
```

**Training details:**
- **Optimizer:** AdamW (lr=1e-4, weight_decay from config, betas=(0.9, 0.95))
- **LR schedule:** cosine annealing with linear warmup
- **Loss:** `MultiModalLoss` — per-modality reconstruction (cross-entropy for text, L1+MSE for audio, L1+frequency for image, L1+temporal-consistency for video) with learned uncertainty weighting (Kendall et al., 2018), plus optional InfoNCE contrastive alignment
- **Mixed precision:** FP16 via `torch.amp` (on by default)
- **Gradient checkpointing:** on by default (saves VRAM)
- **Gradient clipping:** max norm from config
- **Task sampling:** each step randomly picks a (source, target) modality pair, giving uniform coverage of all 16 combinations over time

#### b) COCO Captions training (`train_coco.py`)

Trains on real image–caption pairs from COCO for text ↔ image translation.

```bash
python train_coco.py \
    --image-dir data/train2014 \
    --annotation-file data/annotations/captions_train2014.json \
    --total-steps 50000

# Resume from checkpoint
python train_coco.py --image-dir data/train2014 \
    --annotation-file data/annotations/captions_train2014.json \
    --resume checkpoints_coco/checkpoint_step5000.pt
```

**Training details:**
- **Two-phase curriculum:**
  - Phase 1 (warmup, first 15% of steps): self-reconstruction only (image→image, text→text)
  - Phase 2 (remaining 85%): cross-modal translation (image→text 35%, text→image 35%, self-reconstruction 30%)
- **Dataset:** `CocoCaptionsDataset` — loads COCO images + captions, resizes images to 224×224, byte-tokenizes captions
- **Loss, optimizer, precision:** same as synthetic training
- **Checkpoints:** saved every `--save-every` steps (default 5000) to `checkpoints_coco/`

#### c) Curriculum training from video (`curriculum_train.py`)

Learns from raw video files. A single video provides free cross-modal supervision (frames, audio, optional transcript) with no manual labelling.

```bash
python curriculum_train.py --video-dir /path/to/videos
python curriculum_train.py --video-dir ./videos --total-steps 500  # quick test
python curriculum_train.py --synthetic --total-steps 200           # no videos needed

# With temporal context modules
python curriculum_train.py --video-dir ./videos \
    --enable-temporal-transformer --enable-memory
```

**Training details:**
- **Five-phase curriculum** (each phase adds harder tasks):

  | Phase | Fraction | Tasks |
  |---|---|---|
  | 1. Warmup | 10% | video recon, audio recon |
  | 2. Cross-modal | 25% | + video↔audio |
  | 3. Temporal | 25% | + temporal prediction, ordering, distance |
  | 4. Grounding | 15% | + video→text, audio→text (needs transcripts) |
  | 5. Joint | 25% | all tasks together |

- **Temporal context** (three complementary approaches, enabled via flags):
  1. Multi-scale temporal sampling — dataset-level clip pair tasks (order, distance, prediction)
  2. Hierarchical temporal transformer (`--enable-temporal-transformer`) — sequence-level clip modeling with next-clip prediction
  3. Recurrent memory tokens (`--enable-memory`) — persistent memory across clips with truncated BPTT
- **Dataset:** `VideoWatchingDataset` — extracts overlapping clips from video files, extracts audio mel spectrograms, loads transcript files (.txt/.srt) if present
- **Loss:** `MultiModalLoss` + `TemporalContextLoss` (temporal order BCE, temporal distance CE, next-clip MSE+cosine, scene boundary BCE), all with learned uncertainty weighting
- **Checkpoints:** saved at each phase transition and at the end to `checkpoints/`

#### d) Predictive Coding training (`train_pc.py`)

Trains the OmniLatent backbone using Predictive Coding (Whittington & Bogacz 2017) instead of backpropagation. Each transformer layer becomes a level in a predictive hierarchy with local Hebbian-like weight updates.

```bash
python train_pc.py                            # pure predictive coding
python train_pc.py --blend 0.5               # hybrid: 50% PC + 50% backprop
python train_pc.py --blend-anneal            # curriculum: backprop → PC
python train_pc.py --analytical              # memory-efficient O(1) inference
python train_pc.py --inference-steps 50      # more inference iterations
```

**Training details:**
- **PC-specific parameters:** inference steps (T_infer=20), inference LR (0.1), backprop blend ratio (0.0=pure PC, 1.0=pure backprop)
- **Blend annealing:** optionally transitions smoothly from backprop to PC over a configurable number of steps
- **Analytical inference:** uses residual Jacobian approximation for O(1) memory inference
- **Optimizer:** AdamW (lr=1e-3 by default — higher than standard training because PC updates are local)
- **Data:** synthetic multi-modal data (same as `train.py`)
- **Checkpoints:** saved to `checkpoints/pc/`

#### OmniLatent evaluation and inference (`evaluate.py`)

Load a trained checkpoint and probe what the model learned:

```bash
# Self-reconstruction quality
python evaluate.py --checkpoint checkpoints/checkpoint_final.pt --mode reconstruct

# Cross-modal translation (all 8 key pairs)
python evaluate.py --checkpoint checkpoints/checkpoint_final.pt --mode translate

# Latent space alignment analysis (cosine similarity between modality latents)
python evaluate.py --checkpoint checkpoints/checkpoint_final.pt --mode latent

# Full evaluation suite (all three modes above)
python evaluate.py --checkpoint checkpoints/checkpoint_final.pt --mode all

# Inference on a real file
python evaluate.py --checkpoint checkpoints/checkpoint_final.pt \
    --input-file photo.jpg --source image --target text --mode file
```

**Evaluation modes:**
- **`reconstruct`** — measures self-reconstruction quality: token accuracy for text, MSE + cosine similarity for continuous modalities (audio, image, video)
- **`translate`** — tests all 8 cross-modal translation pairs (image↔text, image↔audio, audio↔video, video↔text), reports output validity (NaN/Inf check), mean absolute value, and standard deviation
- **`latent`** — analyzes latent space alignment: computes cosine similarity between mean-pooled encoder outputs across all modality pairs (higher = more aligned), plus per-modality latent norm statistics
- **`file`** — runs inference on a real file: loads an image/audio/video/text file, translates to the target modality, and saves the output. Text targets use autoregressive generation (`model.generate`); continuous targets use learned target queries

**Inference API (programmatic):**

```python
model.eval()
with torch.no_grad():
    # Cross-modal translation
    result = model("image", image_tensor, "text")
    logits = result["output"]              # (B, T, vocab_size)

    # Autoregressive text generation
    token_ids = model.generate("image", image_tensor, max_len=64)

    # Self-reconstruction
    result = model.reconstruct("image", image_tensor)
    reconstructed = result["output"]       # (B, 3, 224, 224)

    # Multi-modal forward (multiple inputs → multiple outputs)
    results = model.forward_multimodal(
        inputs={"text": tokens, "image": image},
        target_modalities=["text", "image"],
    )
```

#### OmniLatent benchmarking (`benchmark.py`)

Runs a comprehensive diagnostic analysis covering component parameter budgets, per-component timing, encoder information retention (effective rank), backbone per-layer contribution, scaling sensitivity, cross-modal alignment, loss attribution, and hook impact.

```bash
python benchmark.py                                              # quick report
python benchmark.py --dim 768 --layers 12 --heads 12           # custom config
python benchmark.py --checkpoint checkpoints/checkpoint_final.pt # trained model
python benchmark.py --output report.json                        # save to file
```

---

### 2. Gaussian Encoder (`gaussian_encoder/`)

A small autoencoder with convolutional filters parameterized as mixtures of oriented 2D Gaussians — structured, interpretable kernels with fewer free parameters than standard convolutions.

#### Training

```bash
python -m gaussian_encoder.train
python -m gaussian_encoder.train --epochs 10 --latent-dim 16 --lr 3e-3
```

**Training details:**
- **Dataset:** MNIST (28×28 greyscale, auto-downloaded via torchvision)
- **Architecture:** encoder uses `GaussianConv2d` layers (5×5 kernels, each filter is a sum of 3 oriented Gaussian blobs with 6 learnable params each: μx, μy, log σx, log σy, θ, amplitude), decoder uses standard transposed convolutions
- **Loss:** MSE reconstruction
- **Optimizer:** Adam (lr=1e-3)
- **Default:** 5 epochs, batch size 256, latent dim 32

#### Inference

After training, reconstruction samples are saved to `gaussian_encoder/samples.png` (top row: originals, bottom row: reconstructions). The script also prints the learned Gaussian σ statistics for the first layer.

```python
from gaussian_encoder.model import GaussianAutoencoder

model = GaussianAutoencoder(in_ch=1, latent_dim=32)
# Load weights...
model.eval()
with torch.no_grad():
    reconstructed, latent = model(input_image)  # input: (B, 1, 28, 28)
```

---

### 3. HPWM — Hierarchical Predictive World Model (`hpwm/`)

A video world model that combines DINO visual features, Mixture-of-Depths (MoD) routing, slot attention for object-centric decomposition, VQ-VAE tokenization, and Mamba temporal state for long-range temporal modeling.

#### Training

```bash
python -m hpwm.train                           # synthetic data (default)
python -m hpwm.train --ssv2-dir /path/to/ssv2  # Something-Something V2
python -m hpwm.train --resume checkpoints/hpwm/checkpoint_step_1000.pt
python -m hpwm.train --no-mamba                # Transformer baseline
python -m hpwm.train --steps 5000 --lr 3e-4
```

**Training details:**
- **Architecture:** frozen DINO backbone → MoD router (selects high-surprise patches) → slot attention encoder (object-centric decomposition) → VQ-VAE (discrete tokenization) → Mamba temporal state (or Transformer baseline) → next-frame prediction head
- **Loss:** composite of prediction loss, VQ-VAE reconstruction loss, FWM loss, commitment loss, entropy loss, slot consistency loss, slot specialization loss
- **Optimizer:** AdamW with per-component learning rate scaling, cosine schedule with warmup
- **Mixed precision:** bf16 (or fp16)
- **Gradient accumulation:** effective batch size 16 (configurable)
- **MoD K-ratio annealing:** the routing sparsity increases over training
- **Checkpoints:** saved periodically + best model based on validation loss to `checkpoints/hpwm/`

#### Evaluation

```bash
python -m hpwm.train --eval-only --resume checkpoints/hpwm/checkpoint_best.pt
```

**Three validation signals** (all three must pass):

| Signal | Metric | Pass criterion |
|---|---|---|
| 1. MoD Routing Entropy | Entropy of routing weight distribution | Entropy decreases over training |
| 2. Slot Binding Stability | Jaccard overlap of slot assignments across consecutive frames | Jaccard > 0.6 |
| 3. Mamba State Retention | Cosine similarity of temporal features at early/mid/late positions | HPWM degrades gradually (vs Transformer which drops sharply) |

Additional metrics: VQ-VAE codebook utilization (active ratio, perplexity), validation loss. Results are logged to TensorBoard and saved as JSON.

---

### 4. LGQ — Learnable Geometric Quantization (`lgq/`)

A VQGAN-based image tokenizer supporting three quantization schemes: LGQ (learnable geometric), FSQ (fixed scalar), and SimVQ (simple vector with EMA).

#### Training

```bash
python -m lgq.train --quantizer lgq --steps 100000
python -m lgq.train --quantizer fsq --steps 100000
python -m lgq.train --quantizer simvq --steps 100000
python -m lgq.train --quantizer lgq --vocab-size 512 --n-codebooks 4 --lr 2e-4
```

**Training details:**
- **Architecture:** convolutional encoder → quantizer (LGQ/FSQ/SimVQ) → convolutional decoder, with a PatchGAN discriminator for adversarial training
- **Loss:** reconstruction (L1 + perceptual) + codebook loss (commitment + entropy) + adversarial loss (discriminator starts after `disc_start_step`)
- **Optimizers:** separate AdamW for generator (betas=(0.5, 0.9)) and discriminator
- **LR schedule:** cosine annealing with warmup (both generator and discriminator)
- **Mixed precision:** bf16
- **Data:** synthetic structured images by default (gradients, circles, grids at configurable resolution)
- **Checkpoints:** periodic + best model (based on PSNR) to `checkpoints/lgq/`

#### Evaluation

```bash
# Single model
python -m lgq.evaluate --checkpoint checkpoints/lgq/best.pt

# Compare multiple quantizers side by side
python -m lgq.evaluate --compare lgq:path/lgq.pt fsq:path/fsq.pt simvq:path/simvq.pt

# Save results
python -m lgq.evaluate --checkpoint checkpoints/lgq/best.pt --output results.json
```

**Metrics reported:**
- **PSNR** — peak signal-to-noise ratio (higher is better)
- **SSIM** — structural similarity (higher is better)
- **LPIPS** — learned perceptual similarity (lower is better)
- **rFID** — reconstruction FID using Inception features (lower is better)
- **Codebook utilization** — active ratio, perplexity, effective vocab bits
- **Compression** — bits per pixel, compression ratio

---

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
