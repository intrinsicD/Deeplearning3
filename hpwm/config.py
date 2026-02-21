"""
HPWM Phase -1 Configuration.

Interface-identical to full spec; only config values differ.
Swap this config for full-scale training without code changes.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HPWMConfig:
    # ── Video input ──────────────────────────────────────
    resolution: int = 128        # full spec: 224
    fps: int = 2                 # full spec: 8-16
    clip_length_s: int = 60      # full spec: unbounded
    n_frames: int = 120          # = 60s * 2fps

    # ── DINO backbone ────────────────────────────────────
    dino_model: str = "dinov2_vits14"
    d_dino: int = 384            # ViT-S hidden dim
    n_patches: int = 81          # 9x9 for 126px (128 resized) / 14px patches
    patch_grid: int = 9          # sqrt(n_patches)
    dino_frozen: bool = True     # Phase -1: fully frozen

    # ── MoD / FWM (Component 1) ──────────────────────────
    k_ratio_init: float = 1.0    # start fully uniform
    k_ratio_final: float = 0.3   # anneal: 30% to heavy blocks
    k_ratio_warmup_steps: int = 5000
    fwm_channels: int = 384      # FWM conv channels (matches d_dino)
    fwm_layers: int = 3          # 3-layer ConvNet
    n_heavy_layers: int = 2      # small Transformer for heavy path
    d_heavy: int = 128           # heavy block hidden dim

    # ── Slot encoder (Component 2) ───────────────────────
    n_slots: int = 8             # full spec: 32
    d_slot: int = 128            # full spec: 768
    slot_iters: int = 3          # iterative refinement steps
    slot_mlp_hidden: int = 256   # slot MLP hidden dim
    tome_threshold: float = 0.9  # Token Merging cosine sim threshold

    # ── Temporal state (Component 4) ─────────────────────
    n_temporal_tiers: int = 1    # full spec: 2 or 3
    d_mamba: int = 256           # full spec: 512/1024/2048
    mamba_d_state: int = 16      # SSM state dimension
    mamba_d_conv: int = 4        # local convolution width
    mamba_expand: int = 2        # expansion factor
    mamba_n_layers: int = 4      # number of Mamba blocks
    use_mamba: bool = True       # False = flat Transformer baseline

    # ── Multi-scale attention (Component 5) ──────────────
    n_scales: int = 2            # full spec: 3
    n_layers_fast: int = 4       # full spec: 12
    n_layers_slow: int = 6       # full spec: 24
    d_fast: int = 128            # fast scale model dim
    d_slow: int = 256            # slow scale model dim
    token_budget: int = 512      # per scale; full spec: 2048
    n_heads: int = 4             # attention heads per scale

    # ── VQ-VAE (Component 7) ────────────────────────────
    vqvae_codebooks: int = 8     # number of codebooks
    vqvae_vocab_size: int = 256  # codes per codebook
    vqvae_dim: int = 64          # codebook embedding dim
    vqvae_hidden: int = 128      # encoder/decoder hidden dim
    vqvae_n_layers: int = 3      # encoder/decoder depth

    # ── Training ─────────────────────────────────────────
    batch_size: int = 1
    grad_accum_steps: int = 16   # effective batch 16
    grad_checkpointing: bool = True
    precision: str = "bf16"
    optimizer: str = "AdamW"
    lr: float = 3e-4
    weight_decay: float = 0.01
    total_steps: int = 50000     # ~1 week on RTX 3070
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # ── Loss weights ─────────────────────────────────────
    loss_weight_prediction: float = 1.0   # next-frame token prediction
    loss_weight_vqvae: float = 0.5        # VQ-VAE reconstruction
    loss_weight_fwm: float = 0.1          # FWM next-frame feature prediction
    loss_weight_commitment: float = 0.25  # VQ commitment loss

    # ── Evaluation & logging ─────────────────────────────
    eval_every: int = 500
    save_every: int = 5000
    log_every: int = 100

    # ── Paths ────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints/hpwm"
    log_dir: str = "logs/hpwm"
    data_dir: str = "data"

    # ── DINO frame processing chunk size (memory) ────────
    dino_chunk_size: int = 8     # process N frames at a time through DINO

    @property
    def n_vq_tokens(self) -> int:
        """Number of spatial VQ tokens per frame."""
        # VQ-VAE downsamples by 4x -> (128/4)^2 = 1024... too many
        # Use 8x downsample -> (128/8)^2 = 256
        return (self.resolution // 8) ** 2

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps
