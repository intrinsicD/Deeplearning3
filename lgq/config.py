"""Configuration for Learnable Geometric Quantization."""

from dataclasses import dataclass, field


@dataclass
class LGQConfig:
    # --- Codebook ---
    n_codebooks: int = 8
    vocab_size: int = 256
    codebook_dim: int = 8           # dimension per codebook head
    vq_dim: int = 64                # total quantization dimension (= n_codebooks * codebook_dim)

    # --- Temperature annealing ---
    tau_init: float = 1.0           # initial temperature (soft)
    tau_final: float = 0.05         # final temperature (near-hard)
    tau_warmup_steps: int = 5000    # steps before annealing starts
    tau_anneal_steps: int = 30000   # steps over which tau decays

    # --- Encoder/Decoder (VQGAN backbone) ---
    in_channels: int = 3
    hidden_dim: int = 128
    n_res_blocks: int = 3
    downsample_factor: int = 8      # spatial compression ratio
    resolution: int = 256           # input image resolution

    # --- Regularization ---
    commitment_weight: float = 0.25
    confidence_weight: float = 0.1      # entropy of q(k|z) - peakedness
    balance_weight: float = 0.1         # KL(q_bar || uniform) - balanced usage
    free_energy_weight: float = 0.1     # variational free-energy term

    # --- Loss weights ---
    recon_weight: float = 1.0
    perceptual_weight: float = 0.1      # LPIPS-style perceptual loss
    adversarial_weight: float = 0.1     # discriminator loss
    codebook_weight: float = 1.0        # total codebook loss multiplier

    # --- Discriminator ---
    disc_start_step: int = 10000        # start discriminator after N steps
    disc_hidden_dim: int = 64
    disc_n_layers: int = 3

    # --- Training ---
    batch_size: int = 8
    learning_rate: float = 1e-4
    disc_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    total_steps: int = 100000
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0
    precision: str = "bf16"

    # --- Evaluation ---
    eval_every: int = 1000
    save_every: int = 10000
    log_every: int = 100

    # --- Paths ---
    checkpoint_dir: str = "checkpoints/lgq"
    log_dir: str = "logs/lgq"
    data_dir: str = "data/imagenet"

    # --- Quantizer variant for comparison ---
    quantizer_type: str = "lgq"  # "lgq", "vq", "fsq", "simvq"

    # FSQ-specific
    fsq_levels: list[int] = field(default_factory=lambda: [8, 6, 5])

    @property
    def latent_size(self) -> int:
        return self.resolution // self.downsample_factor

    @property
    def n_latent_tokens(self) -> int:
        return self.latent_size ** 2

    def __post_init__(self) -> None:
        if self.vq_dim != self.n_codebooks * self.codebook_dim:
            raise ValueError(
                f"vq_dim ({self.vq_dim}) must equal "
                f"n_codebooks * codebook_dim ({self.n_codebooks * self.codebook_dim})"
            )
