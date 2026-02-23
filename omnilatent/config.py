"""Configuration for OmniLatent model.

Architecture sized for single-GPU training on 8GB VRAM:
  ~140M parameters, FP16 training with gradient checkpointing.

Memory budget (FP16 + AdamW):
  Parameters:      ~280 MB
  Gradients:       ~280 MB
  Optimizer state:  ~1.1 GB  (FP32 momentum + variance)
  Master weights:   ~560 MB
  Activations:     ~1-2 GB  (with gradient checkpointing)
  Data buffers:    ~0.5 GB
  Total:           ~3.7-4.7 GB  → fits in 8 GB
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class OmniLatentConfig:
    # --- Backbone transformer ---
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 2.667       # effective 2x after SwiGLU split → 768*2.667≈2048
    dropout: float = 0.0
    max_seq_len: int = 2048
    gradient_checkpointing: bool = True

    # --- Text ---
    vocab_size: int = 32_000
    text_max_len: int = 512
    text_pad_token: int = 0
    text_bos_token: int = 1

    # --- Audio ---
    audio_n_mels: int = 128        # mel spectrogram bins
    audio_hop_length: int = 160    # ~10 ms at 16 kHz
    audio_max_frames: int = 1000   # ~10 s at 16 kHz
    audio_patch_frames: int = 4    # group 4 mel frames into 1 token
    audio_sample_rate: int = 16_000

    # --- Image ---
    image_size: int = 224
    image_patch_size: int = 16     # 224/16 = 14 → 196 tokens
    image_channels: int = 3

    # --- Video ---
    video_size: int = 112          # smaller than images for memory
    video_patch_size: int = 16     # spatial
    video_temporal_patch: int = 4  # group 4 frames → 1 temporal token
    video_max_frames: int = 16     # max 16 frames during training
    video_channels: int = 3

    # --- Latent Reasoning (Chain of Continuous Thought) ---
    reasoning_enabled: bool = False          # off by default; enable for reasoning tasks
    reasoning_num_thoughts: int = 16         # learnable thought tokens
    reasoning_num_layers: int = 4            # dedicated reasoning transformer layers
    reasoning_num_heads: int = 8             # attention heads in reasoning layers
    reasoning_gate_bias_init: float = -4.0   # sigmoid(-4)≈0.018 → starts nearly silent
    reasoning_bottleneck_weight: float = 0.1 # weight for auxiliary bottleneck loss

    # --- Latent Neural Hooks ---
    hook_gate_init: float = 0.0    # sigmoid(0)=0.5; use negative for near-zero start
    hook_gate_bias_init: float = -4.0  # sigmoid(-4)≈0.018 → starts nearly silent

    # --- Temporal Context ---
    # Approach 1: Multi-scale temporal sampling
    temporal_distance_buckets: int = 4  # <10s, 10s-60s, 1m-5m, >5m

    # Approach 2: Hierarchical clip-then-sequence transformer
    temporal_seq_layers: int = 4        # layers in temporal sequence transformer
    temporal_seq_heads: int = 8         # attention heads
    temporal_seq_max_clips: int = 60    # max clips in a sequence (~1 minute)
    temporal_seq_dropout: float = 0.1

    # Approach 3: Recurrent memory tokens
    memory_num_tokens: int = 8          # memory tokens persisted across clips
    memory_gate_bias_init: float = -4.0 # start nearly silent like hooks

    # --- Training ---
    learning_rate: float = 1e-4        # reduced from 3e-4 to stabilize training
    weight_decay: float = 0.05
    warmup_steps: int = 2000           # increased from 500 for multi-modal loss landscape
    max_steps: int = 100_000
    batch_size: int = 4
    mixed_precision: bool = True
    grad_clip: float = 0.5             # reduced from 1.0 to tame gradient explosion
    seed: int = 42

    # --- Loss ---
    modality_loss_weights: dict[str, float] = field(default_factory=lambda: {
        "text": 1.0,
        "audio": 1.0,
        "image": 1.0,
        "video": 1.0,
    })
    contrastive_weight: float = 0.0     # disabled: causes modality collapse on unpaired/synthetic data
    contrastive_temperature: float = 0.07

    # --- Debug ---
    debug_shapes: bool = False         # enable runtime shape assertions at module boundaries

    # Derived (computed in __post_init__)
    mlp_dim: int = 0
    image_num_patches: int = 0
    video_spatial_patches: int = 0

    def __post_init__(self) -> None:
        self.mlp_dim = int(self.hidden_dim * self.mlp_ratio)
        self.image_num_patches = (self.image_size // self.image_patch_size) ** 2
        vs = self.video_size // self.video_patch_size
        self.video_spatial_patches = vs * vs

        # Validate config consistency
        if self.image_size % self.image_patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"image_patch_size ({self.image_patch_size})"
            )
        if self.video_size % self.video_patch_size != 0:
            raise ValueError(
                f"video_size ({self.video_size}) must be divisible by "
                f"video_patch_size ({self.video_patch_size})"
            )
        if self.video_max_frames % self.video_temporal_patch != 0:
            raise ValueError(
                f"video_max_frames ({self.video_max_frames}) must be divisible by "
                f"video_temporal_patch ({self.video_temporal_patch})"
            )
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
