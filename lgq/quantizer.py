"""Learnable Geometric Quantization (LGQ) quantizer.

The core innovation: replaces hard nearest-neighbor VQ lookups with
temperature-controlled soft assignments grounded in an isotropic Gaussian
mixture model.  During training, soft assignments let gradients update ALL
codebook entries proportionally.  At inference, assignments sharpen to hard
(argmin) for discrete tokens.

Key components:
  1. Codebook geometry: learnable centroids (nn.Parameter)
  2. Soft assignment: q(k|z) = softmax(-||z - c_k||^2 / (2*tau^2))
  3. Temperature schedule: tau anneals from warm (soft) to cold (hard)
  4. Free-energy term: variational ELBO under isotropic Gaussian mixture
  5. Regularizers: confidence (low entropy q) + balance (uniform marginal)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LGQuantizer(nn.Module):
    """Multi-head Learnable Geometric Quantizer.

    Each head operates on a subspace of the full representation, similar to
    product quantization, but with soft assignments and learned geometry.
    """

    def __init__(
        self,
        n_codebooks: int = 8,
        vocab_size: int = 256,
        codebook_dim: int = 8,
        tau_init: float = 1.0,
        tau_final: float = 0.05,
        tau_warmup_steps: int = 5000,
        tau_anneal_steps: int = 30000,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.codebook_dim = codebook_dim
        self.dim = n_codebooks * codebook_dim

        self.tau_init = tau_init
        self.tau_final = tau_final
        self.tau_warmup_steps = tau_warmup_steps
        self.tau_anneal_steps = tau_anneal_steps

        # Learnable codebook centroids: [n_codebooks, vocab_size, codebook_dim]
        # Initialized from unit sphere for geometric stability
        codebooks = torch.randn(n_codebooks, vocab_size, codebook_dim)
        codebooks = F.normalize(codebooks, dim=-1) * math.sqrt(codebook_dim)
        self.codebooks = nn.Parameter(codebooks)

        # Running EMA of marginal usage for monitoring (not for updates)
        self.register_buffer(
            "code_usage_ema",
            torch.ones(n_codebooks, vocab_size) / vocab_size,
        )
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))
        self.ema_decay = 0.99

    def get_temperature(self, step: int | None = None) -> float:
        """Compute current temperature from annealing schedule.

        Schedule: constant warmup -> cosine anneal -> constant final.
        """
        if step is None:
            step = self._step.item()
        if step < self.tau_warmup_steps:
            return self.tau_init
        anneal_progress = min(
            1.0,
            (step - self.tau_warmup_steps) / max(1, self.tau_anneal_steps),
        )
        # Cosine anneal from tau_init to tau_final
        tau = self.tau_final + 0.5 * (self.tau_init - self.tau_final) * (
            1.0 + math.cos(math.pi * anneal_progress)
        )
        return tau

    def soft_assignment(
        self, z: torch.Tensor, codebook: torch.Tensor, tau: float
    ) -> torch.Tensor:
        """Compute soft assignment probabilities q(k|z).

        Under the isotropic Gaussian mixture interpretation:
          p(z|k) = N(z; c_k, sigma^2 I)
          q(k|z) propto exp(-||z - c_k||^2 / (2 * tau^2))

        Args:
            z: [N, D] encoded vectors
            codebook: [K, D] centroids
            tau: temperature scalar

        Returns:
            q: [N, K] assignment probabilities (sums to 1 per row)
        """
        # Squared distances: ||z - c_k||^2 = ||z||^2 - 2*z*c_k + ||c_k||^2
        dists = (
            z.pow(2).sum(-1, keepdim=True)
            - 2.0 * z @ codebook.t()
            + codebook.pow(2).sum(-1, keepdim=True).t()
        )
        # Negative distance / temperature -> softmax
        logits = -dists / (2.0 * tau * tau + 1e-8)
        return F.softmax(logits, dim=-1)

    def quantize_soft(
        self, z: torch.Tensor, q: torch.Tensor, codebook: torch.Tensor
    ) -> torch.Tensor:
        """Soft quantization: weighted sum of codebook entries.

        Args:
            z: [N, D] (unused, kept for API consistency)
            q: [N, K] assignment probs
            codebook: [K, D] centroids

        Returns:
            z_q: [N, D] soft-quantized vectors
        """
        return q @ codebook  # [N, D]

    def quantize_hard(
        self, z: torch.Tensor, codebook: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hard quantization: nearest neighbor lookup.

        Args:
            z: [N, D]
            codebook: [K, D]

        Returns:
            z_q: [N, D] quantized vectors
            indices: [N] codebook indices
        """
        dists = (
            z.pow(2).sum(-1, keepdim=True)
            - 2.0 * z @ codebook.t()
            + codebook.pow(2).sum(-1, keepdim=True).t()
        )
        indices = dists.argmin(dim=-1)
        z_q = codebook[indices]
        return z_q, indices

    def _update_usage_ema(self, q: torch.Tensor, codebook_idx: int) -> None:
        """Update running EMA of code usage for monitoring."""
        with torch.no_grad():
            batch_usage = q.mean(dim=0)  # [K]
            self.code_usage_ema[codebook_idx] = (
                self.ema_decay * self.code_usage_ema[codebook_idx]
                + (1.0 - self.ema_decay) * batch_usage
            )

    def forward(
        self, z: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Full quantization pass.

        Args:
            z: [B, D, H, W] continuous encoder output

        Returns dict with:
            quantized: [B, D, H, W] quantized features
            indices: [B, n_codebooks, H, W] hard assignment indices
            soft_assignments: list of [N, K] per-codebook soft assignment probs
            commitment_loss: scalar
            free_energy: scalar (variational free-energy term)
            confidence_loss: scalar (entropy of q(k|z))
            balance_loss: scalar (KL from uniform marginal)
        """
        B, D, H, W = z.shape
        z_flat = rearrange(z, "b d h w -> (b h w) d")
        N = z_flat.shape[0]

        tau = self.get_temperature()

        # Split into per-codebook subspaces
        z_splits = z_flat.chunk(self.n_codebooks, dim=-1)

        quantized_parts = []
        all_indices = []
        all_soft_assignments = []
        total_commitment = 0.0
        total_free_energy = 0.0
        total_confidence_loss = 0.0
        total_balance_loss = 0.0

        for i, z_part in enumerate(z_splits):
            codebook = self.codebooks[i]  # [K, D_cb]

            # Soft assignment
            q = self.soft_assignment(z_part, codebook, tau)  # [N, K]
            all_soft_assignments.append(q)

            if self.training:
                # Soft quantization for training
                z_q = self.quantize_soft(z_part, q, codebook)  # [N, D_cb]

                # Hard indices for monitoring / straight-through
                _, hard_indices = self.quantize_hard(z_part.detach(), codebook)

                # Straight-through: use hard indices' output in forward,
                # but gradient flows through soft path
                z_q_hard = codebook[hard_indices]
                z_q_st = z_q + (z_q_hard - z_q).detach()

                quantized_parts.append(z_q_st)
                all_indices.append(hard_indices.view(B, H, W))

                # Update usage tracking
                self._update_usage_ema(q, i)
            else:
                # Hard quantization at inference
                z_q_hard, hard_indices = self.quantize_hard(z_part, codebook)
                quantized_parts.append(z_q_hard)
                all_indices.append(hard_indices.view(B, H, W))

            # --- Losses ---

            # 1. Commitment loss: encoder should commit to codebook
            if self.training:
                z_q_detached = self.quantize_soft(z_part, q.detach(), codebook.detach())
                commitment = F.mse_loss(z_part, z_q_detached)
                total_commitment = total_commitment + commitment

            # 2. Free-energy (variational ELBO under Gaussian mixture)
            # -E_q[log p(z|k)] + KL(q || prior)
            # log p(z|k) = -||z - c_k||^2 / (2*sigma^2) + const
            # Using tau as sigma proxy
            if self.training:
                dists = (z_part.unsqueeze(1) - codebook.unsqueeze(0)).pow(2).sum(-1)
                # [N, K]
                expected_dist = (q * dists).sum(-1).mean()  # E_q[||z - c_k||^2]
                # KL(q || uniform) = sum q*log(q) + log(K)
                log_q = (q + 1e-8).log()
                kl_uniform = (q * log_q).sum(-1).mean() + math.log(self.vocab_size)
                free_energy = expected_dist / (2.0 * tau * tau + 1e-8) + kl_uniform
                total_free_energy = total_free_energy + free_energy

            # 3. Confidence loss: entropy of q(k|z) should be low
            if self.training:
                entropy_q = -(q * (q + 1e-8).log()).sum(-1).mean()
                total_confidence_loss = total_confidence_loss + entropy_q

            # 4. Balance loss: marginal usage should be uniform
            if self.training:
                q_bar = q.mean(dim=0)  # [K]
                # KL(q_bar || uniform) = sum q_bar * log(q_bar * K)
                balance = (q_bar * (q_bar * self.vocab_size + 1e-8).log()).sum()
                total_balance_loss = total_balance_loss + balance

        quantized = torch.cat(quantized_parts, dim=-1)
        quantized = rearrange(quantized, "(b h w) d -> b d h w", b=B, h=H, w=W)
        indices = torch.stack(all_indices, dim=1)  # [B, n_codebooks, H, W]

        n_cb = self.n_codebooks
        result = {
            "quantized": quantized,
            "indices": indices,
            "soft_assignments": all_soft_assignments,
            "commitment_loss": total_commitment / n_cb if self.training else torch.tensor(0.0),
            "free_energy": total_free_energy / n_cb if self.training else torch.tensor(0.0),
            "confidence_loss": total_confidence_loss / n_cb if self.training else torch.tensor(0.0),
            "balance_loss": total_balance_loss / n_cb if self.training else torch.tensor(0.0),
            "temperature": tau,
        }

        if self.training:
            self._step += 1

        return result

    def indices_to_embeddings(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert codebook indices back to embeddings.

        Args:
            indices: [B, n_codebooks, H, W]

        Returns:
            embeddings: [B, D, H, W]
        """
        B, _, H, W = indices.shape
        parts = []
        for i in range(self.n_codebooks):
            idx = indices[:, i].reshape(-1)
            emb = self.codebooks[i][idx]
            parts.append(emb)

        combined = torch.cat(parts, dim=-1)
        return rearrange(combined, "(b h w) d -> b d h w", b=B, h=H, w=W)

    def codebook_utilization(self) -> dict[str, float]:
        """Compute codebook utilization metrics from EMA tracking.

        Returns:
            active_ratio: fraction of codes with non-negligible usage
            perplexity: exp(entropy) of code usage distribution
            mean_usage_entropy: average entropy per codebook
        """
        with torch.no_grad():
            total_active = 0.0
            total_perplexity = 0.0
            total_entropy = 0.0

            for i in range(self.n_codebooks):
                usage = self.code_usage_ema[i]
                probs = usage / (usage.sum() + 1e-8)

                # Active codes: usage above 1/(10*K) threshold
                threshold = 1.0 / (10.0 * self.vocab_size)
                active = (probs > threshold).float().mean().item()

                # Shannon entropy -> perplexity
                log_probs = (probs + 1e-8).log()
                entropy = -(probs * log_probs).sum()
                perplexity = entropy.exp().item()

                total_active += active
                total_perplexity += perplexity
                total_entropy += entropy.item()

            n = max(1, self.n_codebooks)
            return {
                "active_ratio": total_active / n,
                "perplexity": total_perplexity / n,
                "mean_usage_entropy": total_entropy / n,
            }


class FSQuantizer(nn.Module):
    """Fixed Scalar Quantization (FSQ) baseline for comparison.

    Quantizes each dimension independently to a fixed set of levels.
    No codebook learning — the quantization grid is predetermined.
    """

    def __init__(self, levels: list[int], dim: int = 64):
        super().__init__()
        self.levels = levels
        self.n_codebooks = len(levels)
        self.dim = dim
        self.dim_per_level = dim // len(levels)

        # Pre-compute level grids
        for i, L in enumerate(levels):
            grid = torch.linspace(-1.0, 1.0, L)
            self.register_buffer(f"grid_{i}", grid)

    def _get_grid(self, idx: int) -> torch.Tensor:
        return getattr(self, f"grid_{idx}")

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        B, D, H, W = z.shape
        z_flat = rearrange(z, "b d h w -> (b h w) d")
        z_splits = z_flat.chunk(self.n_codebooks, dim=-1)

        quantized_parts = []
        all_indices = []

        for i, z_part in enumerate(z_splits):
            grid = self._get_grid(i)  # [L]
            # Quantize each dimension to nearest grid point
            z_clamped = z_part.tanh()  # map to [-1, 1]
            # Find nearest grid value per element
            diffs = (z_clamped.unsqueeze(-1) - grid.unsqueeze(0).unsqueeze(0)).abs()
            idx = diffs.argmin(dim=-1)  # [N, D_per_level]
            z_q = grid[idx]

            # Straight-through
            z_q = z_clamped + (z_q - z_clamped).detach()

            quantized_parts.append(z_q)
            # Encode multi-dim index as single index per spatial position.
            # Use the actual chunk size (may differ from dim_per_level when vq_dim
            # is not evenly divisible by n_codebooks) and the i-th level's grid
            # size as the radix for each position.
            L = self.levels[i]
            D = z_part.shape[-1]
            strides = torch.tensor(
                [int(L ** (D - 1 - d)) for d in range(D)],
                device=z.device,
                dtype=torch.long,
            )
            flat_idx = (idx.long() * strides.unsqueeze(0)).sum(-1)  # [N]
            all_indices.append(flat_idx.view(B, H, W))

        quantized = torch.cat(quantized_parts, dim=-1)
        quantized = rearrange(quantized, "(b h w) d -> b d h w", b=B, h=H, w=W)
        indices = torch.stack(all_indices, dim=1)

        return {
            "quantized": quantized,
            "indices": indices,
            "soft_assignments": [],
            "commitment_loss": torch.tensor(0.0, device=z.device),
            "free_energy": torch.tensor(0.0, device=z.device),
            "confidence_loss": torch.tensor(0.0, device=z.device),
            "balance_loss": torch.tensor(0.0, device=z.device),
            "temperature": 0.0,
        }


class SimVQuantizer(nn.Module):
    """Simple Vector Quantization (SimVQ) baseline.

    Standard VQ with EMA codebook updates and straight-through gradient.
    """

    def __init__(
        self,
        n_codebooks: int = 8,
        vocab_size: int = 256,
        codebook_dim: int = 8,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.codebook_dim = codebook_dim
        self.dim = n_codebooks * codebook_dim
        self.ema_decay = ema_decay

        for i in range(n_codebooks):
            self.register_buffer(
                f"codebook_{i}",
                torch.randn(vocab_size, codebook_dim) * 0.02,
            )
        self.register_buffer("ema_count", torch.zeros(n_codebooks, vocab_size))
        self.register_buffer(
            "ema_weight",
            torch.randn(n_codebooks, vocab_size, codebook_dim) * 0.02,
        )

    def _get_codebook(self, idx: int) -> torch.Tensor:
        return getattr(self, f"codebook_{idx}")

    def _set_codebook(self, idx: int, value: torch.Tensor) -> None:
        getattr(self, f"codebook_{idx}").copy_(value)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        B, D, H, W = z.shape
        z_flat = rearrange(z, "b d h w -> (b h w) d")
        z_splits = z_flat.chunk(self.n_codebooks, dim=-1)

        quantized_parts = []
        all_indices = []
        total_commitment = 0.0

        for i, z_part in enumerate(z_splits):
            codebook = self._get_codebook(i)
            dists = (
                z_part.pow(2).sum(-1, keepdim=True)
                - 2.0 * z_part @ codebook.t()
                + codebook.pow(2).sum(-1, keepdim=True).t()
            )
            indices = dists.argmin(dim=-1)
            z_q = codebook[indices]

            if self.training:
                with torch.no_grad():
                    one_hot = F.one_hot(indices, self.vocab_size).float()
                    self.ema_count[i] = (
                        self.ema_decay * self.ema_count[i]
                        + (1 - self.ema_decay) * one_hot.sum(0)
                    )
                    self.ema_weight[i] = (
                        self.ema_decay * self.ema_weight[i]
                        + (1 - self.ema_decay) * (one_hot.t() @ z_part)
                    )
                    n = self.ema_count[i].sum()
                    count = (
                        (self.ema_count[i] + 1e-5)
                        / (n + self.vocab_size * 1e-5) * n
                    )
                    self._set_codebook(i, self.ema_weight[i] / count.unsqueeze(-1))

            commitment = F.mse_loss(z_part, z_q.detach())
            total_commitment = total_commitment + commitment
            z_q = z_part + (z_q - z_part).detach()

            quantized_parts.append(z_q)
            all_indices.append(indices.view(B, H, W))

        quantized = torch.cat(quantized_parts, dim=-1)
        quantized = rearrange(quantized, "(b h w) d -> b d h w", b=B, h=H, w=W)
        indices = torch.stack(all_indices, dim=1)

        return {
            "quantized": quantized,
            "indices": indices,
            "soft_assignments": [],
            "commitment_loss": total_commitment / self.n_codebooks,
            "free_energy": torch.tensor(0.0, device=z.device),
            "confidence_loss": torch.tensor(0.0, device=z.device),
            "balance_loss": torch.tensor(0.0, device=z.device),
            "temperature": 0.0,
        }

    def codebook_utilization(self) -> dict[str, float]:
        with torch.no_grad():
            total_active = 0.0
            total_perplexity = 0.0
            for i in range(self.n_codebooks):
                counts = self.ema_count[i]
                total = counts.sum()
                if total < 1e-8:
                    continue
                probs = counts / total
                active = (counts > 1.0).float().mean().item()
                log_probs = (probs + 1e-8).log()
                entropy = -(probs * log_probs).sum()
                total_active += active
                total_perplexity += entropy.exp().item()
            n = max(1, self.n_codebooks)
            return {
                "active_ratio": total_active / n,
                "perplexity": total_perplexity / n,
            }


def build_quantizer(config) -> nn.Module:
    """Factory function to build quantizer from config."""
    if config.quantizer_type == "lgq":
        return LGQuantizer(
            n_codebooks=config.n_codebooks,
            vocab_size=config.vocab_size,
            codebook_dim=config.codebook_dim,
            tau_init=config.tau_init,
            tau_final=config.tau_final,
            tau_warmup_steps=config.tau_warmup_steps,
            tau_anneal_steps=config.tau_anneal_steps,
        )
    elif config.quantizer_type == "fsq":
        return FSQuantizer(levels=config.fsq_levels, dim=config.vq_dim)
    elif config.quantizer_type in ("vq", "simvq"):
        return SimVQuantizer(
            n_codebooks=config.n_codebooks,
            vocab_size=config.vocab_size,
            codebook_dim=config.codebook_dim,
        )
    else:
        raise ValueError(f"Unknown quantizer type: {config.quantizer_type}")
