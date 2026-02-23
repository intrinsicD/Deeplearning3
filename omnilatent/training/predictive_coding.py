"""Predictive Coding training for OmniLatent.

Implements the Predictive Coding (PC) algorithm as an alternative to
backpropagation, following Whittington & Bogacz (2017) and extended for
transformer architectures.

Key idea: each transformer layer is a level in a predictive hierarchy.
Higher layers generate top-down predictions of lower-layer activities.
Only local prediction errors drive learning — no global backward pass.

Two-phase algorithm:
  1. **Inference phase**: iteratively update latent value nodes at each
     layer to minimize the total prediction error (free energy).
  2. **Learning phase**: update weights using converged latent states
     with local Hebbian-like rules.

Under certain conditions (Whittington & Bogacz 2017; Rosenbaum 2022),
the PC weight updates converge to the same gradients as backpropagation.

References:
  - Whittington & Bogacz (2017). "An Approximation of the Error
    Backpropagation Algorithm in a Predictive Coding Network with
    Local Hebbian Synaptic Plasticity." Neural Computation 29(5).
  - Rao & Ballard (1999). "Predictive coding in the visual cortex."
    Nature Neuroscience 2(1).
  - Millidge, Tschantz & Buckley (2022). "Predictive Coding: Towards
    a Future of Deep Learning beyond Backpropagation?" arXiv:2202.09467.
  - Rosenbaum (2022). "On the relationship between predictive coding
    and backpropagation." PLOS ONE.

Adapted for OmniLatent's transformer backbone:
  - Each TransformerBlock is one hierarchical level
  - Value nodes are the residual-stream activations between layers
  - Prediction functions are the transformer blocks themselves
  - Supports the existing multi-modal encode -> backbone -> decode pipeline
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from omnilatent.config import OmniLatentConfig
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.losses import ReconstructionLoss
from omnilatent.training.metrics import MetricsAggregator, StepMetrics
from omnilatent.training.sampler import TaskSampler
from omnilatent.utils import ALL_MODALITIES, MODALITY_ID, Modality, set_seed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PCConfig:
    """Configuration for predictive coding training."""

    # Inference phase
    inference_steps: int = 20           # T_infer: iterations to converge latents
    inference_lr: float = 0.1           # eta_infer: step size for latent updates
    inference_lr_decay: float = 0.95    # multiply eta_infer by this each step

    # Learning phase
    learning_lr: float = 1e-3           # eta_learn: step size for weight updates
    weight_decay: float = 0.01

    # PC-specific
    error_decay: float = 0.0            # optional L2 penalty on value nodes
    supervised_weight: float = 1.0      # weight for output-layer supervised error

    # Hybrid mode: blend PC updates with standard backprop
    # 0.0 = pure PC, 1.0 = pure backprop
    backprop_blend: float = 0.0

    # Hybrid annealing: curriculum from backprop -> PC over training
    backprop_blend_anneal: bool = False
    backprop_blend_start: float = 1.0   # start pure backprop
    backprop_blend_end: float = 0.0     # end pure PC
    backprop_blend_anneal_steps: int = 20_000

    # Memory optimization: use analytical gradients for value node updates
    # instead of autograd (residual Jacobian approximation: J^T ≈ I)
    use_analytical_inference: bool = False

    # Training
    max_steps: int = 100_000
    warmup_steps: int = 2000
    log_interval: int = 50
    batch_size: int = 4
    mixed_precision: bool = True
    grad_clip: float = 1.0
    seed: int = 42

    # Annealing: increase inference steps over training
    inference_steps_warmup: int = 5     # start with fewer inference steps
    inference_steps_anneal_end: int = 10_000  # reach full T_infer by this step

    # Precision parameter constraints
    precision_lr_ratio: float = 0.1     # precision LR = learning_lr * this
    precision_min: float = 0.01
    precision_max: float = 100.0

    # Checkpoint saving
    save_every: int = 5000
    save_dir: str = "checkpoints/pc"


# ---------------------------------------------------------------------------
# Predictive Coding Layer Wrapper
# ---------------------------------------------------------------------------

class PCLayer(nn.Module):
    """Wraps a transformer block as a predictive coding layer.

    Each layer maintains:
      - A prediction function f(x; theta) = transformer_block(x)
      - Value nodes (latent variables) that are iteratively refined
      - Local prediction errors epsilon = x_below - f(x_above)

    The prediction function maps from this layer's value node input
    to the next layer's expected value.  The error is measured at the
    output side: epsilon_l = value[l+1] - block_l(value[l]).
    """

    def __init__(self, block: nn.Module, dim: int) -> None:
        super().__init__()
        self.block = block
        self.dim = dim
        # Learnable precision (inverse variance) per layer -- controls
        # how much this layer's prediction errors matter relative to others
        self.log_precision = nn.Parameter(torch.zeros(1))

    @property
    def precision(self) -> torch.Tensor:
        return torch.exp(self.log_precision).clamp(min=0.01, max=100.0)

    def analytical_value_gradient(
        self,
        error_above: torch.Tensor,
        error_below: torch.Tensor | None,
        precision_above: torch.Tensor,
        precision_below: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute analytical gradient for value node updates.

        From Whittington & Bogacz (2017), the value node update rule:
            dx(l)/dt = -precision(l-1) * ε(l-1) + precision(l) * ε(l)

        where ε(l) = value(l+1) - f_l(value(l)) is the prediction error.

        The second term is propagated through the Jacobian df(l)/dx(l+1).
        We use the residual approximation: since TransformerBlock computes
        x + ls * f(x) (with LayerScale init ~0.1), the Jacobian J ≈ I,
        making J^T @ ε ≈ ε.  This avoids storing autograd graphs.
        """
        # Top-down: this layer's prediction error weighted by precision above
        grad = precision_above * error_above

        if error_below is not None and precision_below is not None:
            # Bottom-up: error from layer below (residual Jacobian approx)
            grad = grad - precision_below * error_below

        return grad

    def predict(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor | None = None,
        rope_offset: int = 0,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward prediction: apply the transformer block."""
        return self.block(x, rope_freqs, rope_offset, attn_mask)

    def prediction_error(
        self,
        value_below: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Compute prediction error: epsilon = value_below - prediction."""
        return value_below - prediction

    def layer_energy(
        self,
        value_below: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Local free energy: 1/2 * precision * ||epsilon||^2."""
        error = self.prediction_error(value_below, prediction)
        # .squeeze() ensures scalar output (precision has shape [1])
        return (0.5 * self.precision * (error ** 2).sum(dim=-1).mean()).squeeze()


# ---------------------------------------------------------------------------
# Predictive Coding Network
# ---------------------------------------------------------------------------

class PredictiveCodingNetwork(nn.Module):
    """Wraps OmniLatent's transformer backbone for predictive coding training.

    This module:
      1. Takes the existing TransformerBlock layers from the backbone
      2. Wraps each in a PCLayer with precision parameters
      3. Implements the two-phase PC algorithm (inference + learning)
      4. Returns layer-wise prediction errors and energies for monitoring

    The encoders/decoders remain unchanged -- PC only replaces the
    training algorithm for the backbone transformer layers.
    """

    def __init__(
        self,
        model: OmniLatentModel,
        pc_config: PCConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.pc_config = pc_config
        self.config = model.config

        # Wrap each backbone transformer block as a PC layer
        self.pc_layers = nn.ModuleList()
        for block in model.backbone.layers:
            pc_layer = PCLayer(block, model.config.hidden_dim)
            self.pc_layers.append(pc_layer)

        # Learnable precision for the supervised output error
        self.output_log_precision = nn.Parameter(torch.zeros(1))

        # Cached reconstruction loss (avoid repeated instantiation)
        self._recon_loss_fn = ReconstructionLoss()

    @property
    def output_precision(self) -> torch.Tensor:
        cfg = self.pc_config
        return torch.exp(self.output_log_precision).clamp(
            min=cfg.precision_min, max=cfg.precision_max
        )

    @property
    def num_layers(self) -> int:
        return len(self.pc_layers)

    def _get_inference_steps(self, global_step: int) -> int:
        """Anneal inference steps: start small, grow to full T_infer."""
        cfg = self.pc_config
        if global_step >= cfg.inference_steps_anneal_end:
            return cfg.inference_steps
        progress = global_step / max(cfg.inference_steps_anneal_end, 1)
        return max(1, int(
            cfg.inference_steps_warmup
            + progress * (cfg.inference_steps - cfg.inference_steps_warmup)
        ))

    def _get_backprop_blend(self, global_step: int) -> float:
        """Return effective backprop_blend, possibly annealed over training.

        When ``backprop_blend_anneal`` is True, linearly interpolates from
        ``backprop_blend_start`` (default 1.0 = pure backprop) to
        ``backprop_blend_end`` (default 0.0 = pure PC) over
        ``backprop_blend_anneal_steps`` training steps.  This implements a
        curriculum strategy where the network first learns useful
        representations via backprop and then transitions to PC.
        """
        cfg = self.pc_config
        if not cfg.backprop_blend_anneal:
            return cfg.backprop_blend

        if global_step >= cfg.backprop_blend_anneal_steps:
            return cfg.backprop_blend_end

        progress = global_step / max(cfg.backprop_blend_anneal_steps, 1)
        return cfg.backprop_blend_start + progress * (
            cfg.backprop_blend_end - cfg.backprop_blend_start
        )

    def _recon_loss(
        self, modality: Modality, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss using cached loss function."""
        return self._recon_loss_fn(modality, prediction, target)

    def _compute_energy(
        self,
        value_nodes: list[torch.Tensor],
        rope_freqs: torch.Tensor | None,
        rope_offset: int,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute total free energy across all layers.

        F = sum_l  1/2 * precision_l * ||value[l+1] - block_l(value[l])||^2
        """
        total = torch.tensor(0.0, device=value_nodes[0].device)
        for i, pc_layer in enumerate(self.pc_layers):
            prediction = pc_layer.predict(
                value_nodes[i], rope_freqs, rope_offset, attn_mask
            )
            energy = pc_layer.layer_energy(value_nodes[i + 1], prediction)
            total = total + energy
        return total

    def inference_phase(
        self,
        x_input: torch.Tensor,
        rope_freqs: torch.Tensor | None = None,
        rope_offset: int = 0,
        attn_mask: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, float]]:
        """Inference phase: iteratively refine value nodes to minimize energy.

        Warm-starts from the feedforward pass, then updates value nodes to
        minimize free energy.  Two modes:

        **Autograd mode** (default): uses ``torch.autograd.grad()`` targeted
        at value nodes.  Exact gradients, no retained graph -- O(L) memory
        per iteration.

        **Analytical mode** (``use_analytical_inference=True``): computes
        value-node gradients analytically using the residual Jacobian
        approximation (J^T ≈ I).  O(1) extra memory per iteration, faster,
        but approximate.  Valid when TransformerBlock is ``x + ls * f(x)``
        with small LayerScale ``ls``.

        Both modes support within-inference LR decay (``inference_lr_decay``)
        to help convergence as value nodes approach equilibrium.

        Args:
            x_input: (B, N, D) input tokens (from encoder + target queries)
            rope_freqs: precomputed RoPE frequencies
            rope_offset: position offset for RoPE
            attn_mask: attention mask for the backbone
            num_steps: override number of inference iterations

        Returns:
            value_nodes: list of L+1 tensors (B, N, D) -- converged latents
            info: dict with energy trajectory and convergence stats
        """
        cfg = self.pc_config
        T = num_steps or cfg.inference_steps
        eta = cfg.inference_lr
        decay = cfg.inference_lr_decay

        # Initialize value nodes from feedforward pass (warm start)
        value_nodes: list[torch.Tensor] = []
        x = x_input.detach()
        with torch.no_grad():
            for pc_layer in self.pc_layers:
                value_nodes.append(x)
                x = pc_layer.predict(x, rope_freqs, rope_offset, attn_mask)
            value_nodes.append(x)  # output of final layer

        if cfg.use_analytical_inference:
            return self._inference_analytical(
                value_nodes, rope_freqs, rope_offset, attn_mask, T, eta, decay,
            )
        return self._inference_autograd(
            value_nodes, rope_freqs, rope_offset, attn_mask, T, eta, decay,
        )

    def _inference_autograd(
        self,
        value_nodes: list[torch.Tensor],
        rope_freqs: torch.Tensor | None,
        rope_offset: int,
        attn_mask: torch.Tensor | None,
        T: int,
        eta: float,
        decay: float,
    ) -> tuple[list[torch.Tensor], dict[str, float]]:
        """Autograd-based inference: exact gradients via torch.autograd.grad."""
        # Value node 0 is clamped (encoder output), the rest are free
        for i in range(1, len(value_nodes)):
            value_nodes[i] = value_nodes[i].detach().requires_grad_(True)

        energies: list[float] = []

        for t in range(T):
            current_eta = eta * (decay ** t)

            # Compute total free energy
            total_energy = self._compute_energy(
                value_nodes, rope_freqs, rope_offset, attn_mask
            )

            # Optional value-node decay (L2 regularization on activations)
            if self.pc_config.error_decay > 0:
                for i in range(1, len(value_nodes)):
                    reg = self.pc_config.error_decay * (
                        value_nodes[i] ** 2
                    ).sum(dim=-1).mean()
                    total_energy = total_energy + reg

            energies.append(total_energy.item())

            # Compute gradients w.r.t. value nodes ONLY (not weights)
            free_nodes = [v for v in value_nodes[1:] if v.requires_grad]
            if not free_nodes:
                break

            grads = torch.autograd.grad(
                total_energy,
                free_nodes,
                create_graph=False,
                retain_graph=False,
            )

            # Update value nodes with gradient descent
            with torch.no_grad():
                grad_idx = 0
                for i in range(1, len(value_nodes)):
                    if value_nodes[i].requires_grad:
                        value_nodes[i] = (
                            value_nodes[i] - current_eta * grads[grad_idx]
                        ).detach().requires_grad_(True)
                        grad_idx += 1

        return value_nodes, self._inference_info(energies, T)

    def _inference_analytical(
        self,
        value_nodes: list[torch.Tensor],
        rope_freqs: torch.Tensor | None,
        rope_offset: int,
        attn_mask: torch.Tensor | None,
        T: int,
        eta: float,
        decay: float,
    ) -> tuple[list[torch.Tensor], dict[str, float]]:
        """Analytical inference: O(1) memory using residual Jacobian approx.

        Instead of building an autograd graph per iteration, computes the
        value-node gradient from the closed-form expression:

            grad(i) = precision(i-1) * ε(i-1) - precision(i) * ε(i)

        where ε(l) = value(l+1) - f_l(value(l)).  The identity Jacobian
        approximation comes from TransformerBlock's residual structure.
        """
        energies: list[float] = []

        for t in range(T):
            current_eta = eta * (decay ** t)

            with torch.no_grad():
                # Compute predictions and errors for all layers
                predictions: list[torch.Tensor] = []
                errors: list[torch.Tensor] = []
                for i, pc_layer in enumerate(self.pc_layers):
                    pred = pc_layer.predict(
                        value_nodes[i], rope_freqs, rope_offset, attn_mask
                    )
                    predictions.append(pred)
                    err = pc_layer.prediction_error(value_nodes[i + 1], pred)
                    errors.append(err)

                # Compute total energy for monitoring
                total_energy = 0.0
                for i, pc_layer in enumerate(self.pc_layers):
                    total_energy += pc_layer.layer_energy(
                        value_nodes[i + 1], predictions[i]
                    ).item()
                energies.append(total_energy)

                # Update interior value nodes using analytical gradients
                for i in range(1, len(value_nodes) - 1):
                    # error_above: ε(i-1) from layer i-1 (pred of i-1 vs value i)
                    error_above = errors[i - 1]
                    prec_above = self.pc_layers[i - 1].precision

                    # error_below: ε(i) from layer i (pred of i vs value i+1)
                    error_below = errors[i] if i < len(errors) else None
                    prec_below = (
                        self.pc_layers[i].precision if i < self.num_layers
                        else None
                    )

                    grad = self.pc_layers[i - 1].analytical_value_gradient(
                        error_above, error_below, prec_above, prec_below,
                    )
                    value_nodes[i] = value_nodes[i] - current_eta * grad

                # Update top value node
                if len(value_nodes) > 1 and errors:
                    top_idx = len(value_nodes) - 1
                    grad = self.pc_layers[-1].precision * errors[-1]
                    value_nodes[top_idx] = (
                        value_nodes[top_idx] - current_eta * grad
                    )

                # Optional value-node decay
                if self.pc_config.error_decay > 0:
                    for i in range(1, len(value_nodes)):
                        value_nodes[i] = value_nodes[i] * (
                            1 - current_eta * self.pc_config.error_decay
                        )

        return value_nodes, self._inference_info(energies, T)

    @staticmethod
    def _inference_info(
        energies: list[float], T: int
    ) -> dict[str, float]:
        """Build inference statistics from energy trajectory."""
        return {
            "initial_energy": energies[0] if energies else 0.0,
            "final_energy": energies[-1] if energies else 0.0,
            "energy_reduction": (
                (energies[0] - energies[-1]) / max(abs(energies[0]), 1e-8)
                if len(energies) > 1 else 0.0
            ),
            "num_inference_steps": T,
        }

    def learning_phase(
        self,
        value_nodes: list[torch.Tensor],
        rope_freqs: torch.Tensor | None = None,
        rope_offset: int = 0,
        attn_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Learning phase: compute weight updates from converged value nodes.

        The local Hebbian-like rule is:
            dW proportional to precision * error * d_prediction/dW

        In practice, we compute the layer-wise energy with converged
        (detached) value nodes and let autograd compute weight gradients.
        This is mathematically equivalent to the local rule but leverages
        PyTorch's autograd machinery.

        Args:
            value_nodes: converged latent states from inference phase
            rope_freqs: RoPE frequencies
            rope_offset: position offset
            attn_mask: attention mask

        Returns:
            dict with per-layer energies and total energy
        """
        losses: dict[str, torch.Tensor] = {}
        total_energy = torch.tensor(0.0, device=value_nodes[0].device)

        for i, pc_layer in enumerate(self.pc_layers):
            # Detach value nodes from inference graph -- weights are the
            # only differentiable quantities in the learning phase
            v_in = value_nodes[i].detach()
            v_out = value_nodes[i + 1].detach()

            # Prediction through the block IS differentiable w.r.t. weights
            prediction = pc_layer.predict(
                v_in, rope_freqs, rope_offset, attn_mask
            )

            energy = pc_layer.layer_energy(v_out, prediction)
            losses[f"layer_{i}_energy"] = energy
            total_energy = total_energy + energy

        losses["total_energy"] = total_energy
        return losses

    def forward_pc(
        self,
        source_modality: Modality,
        source_data: torch.Tensor,
        target_modality: Modality,
        target_data: torch.Tensor,
        global_step: int = 0,
    ) -> dict[str, Any]:
        """Full predictive coding forward pass.

        1. Encode source -> tokens
        2. Build target queries and attention mask (reuse OmniLatent logic)
        3. Run inference phase on backbone layers
        4. Run learning phase with converged latents
        5. Decode final values and compute reconstruction loss

        Returns dict with all losses, energies, and outputs.
        """
        B = source_data.shape[0]
        device = source_data.device
        model = self.model

        # --- Reuse OmniLatent's encoding pipeline ---
        src_tokens = model.encode(source_modality, source_data)

        # Reasoning (if enabled)
        thought_tokens = None
        if model.reasoning is not None:
            thought_tokens, _ = model.reasoning(src_tokens)

        # Target queries
        if target_modality == "text" and target_data is not None:
            bos = torch.full(
                (B, 1), model.config.text_bos_token,
                dtype=torch.long, device=device,
            )
            tgt_input = torch.cat([bos, target_data[:, :-1]], dim=1)
            tgt_queries = model.encoders["text"](tgt_input)
        elif target_modality == "text":
            bos = torch.full(
                (B, 1), model.config.text_bos_token,
                dtype=torch.long, device=device,
            )
            tgt_queries = model.encoders["text"](bos)
        else:
            tgt_queries = model.target_query_gen(target_modality, B)

        # Target modality token
        tgt_mod_tok = model.target_embed(
            torch.tensor(MODALITY_ID[target_modality], device=device)
        ).unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        tgt_with_mod = torch.cat([tgt_mod_tok, tgt_queries], dim=1)
        tgt_len = tgt_with_mod.shape[1]

        # Build full input: [prefix (source + thoughts), target]
        prefix_parts = [src_tokens]
        if thought_tokens is not None:
            prefix_parts.append(thought_tokens)
        prefix = torch.cat(prefix_parts, dim=1)
        prefix_len = prefix.shape[1]

        tokens = torch.cat([prefix, tgt_with_mod], dim=1)

        # Attention mask
        attn_mask = model._create_attention_mask(
            prefix_len, tgt_len, target_modality, device,
        )

        # RoPE frequencies
        rope_freqs = model.backbone.rope_freqs

        # --- Phase 1: Inference (refine value nodes) ---
        num_steps = self._get_inference_steps(global_step)
        value_nodes, inference_info = self.inference_phase(
            x_input=tokens,
            rope_freqs=rope_freqs,
            rope_offset=0,
            attn_mask=attn_mask,
            num_steps=num_steps,
        )

        # --- Phase 2: Learning (weight update via local energy) ---
        learning_losses = self.learning_phase(
            value_nodes=value_nodes,
            rope_freqs=rope_freqs,
            rope_offset=0,
            attn_mask=attn_mask,
        )

        # --- Decode from converged values ---
        final_values = value_nodes[-1]
        final_normed = model.backbone.final_norm(final_values)
        tgt_latent_pc = final_normed[:, prefix_len + 1:]  # skip tgt_mod_tok
        output = model.decoders[target_modality](tgt_latent_pc)

        # --- Supervised reconstruction loss ---
        recon_loss = self._recon_loss(target_modality, output, target_data)

        # --- Combine: PC energy + supervised loss ---
        pc_energy = learning_losses["total_energy"]
        alpha = self._get_backprop_blend(global_step)
        if alpha > 0:
            # Hybrid: blend PC energy with standard recon loss
            total_loss = (1 - alpha) * pc_energy + alpha * recon_loss
        else:
            # Pure PC: energy drives backbone, recon drives periphery
            total_loss = pc_energy + self.pc_config.supervised_weight * recon_loss

        result: dict[str, Any] = {
            "total": total_loss,
            "pc_energy": pc_energy,
            "recon_loss": recon_loss,
            "backprop_blend": alpha,
            "output": output,
        }
        # Per-layer energies
        for k, v in learning_losses.items():
            if k != "total_energy":
                result[k] = v
        # Inference stats
        for k, v in inference_info.items():
            result[f"inference_{k}"] = v

        return result


# ---------------------------------------------------------------------------
# Cosine schedule (same as trainer.py)
# ---------------------------------------------------------------------------

def cosine_schedule(
    step: int, total_steps: int, warmup_steps: int,
    lr: float, min_lr: float = 1e-6,
) -> float:
    """Cosine annealing with linear warm-up."""
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# PC Trainer
# ---------------------------------------------------------------------------

class PCTrainer:
    """Trainer using Predictive Coding instead of (or blended with) backprop.

    Architecture:
      * Encoders and decoders are trained normally (their gradients come
        from the reconstruction loss at the output layer)
      * Backbone transformer layers are trained via PC:
        - Inference phase refines latent value nodes
        - Learning phase updates weights from converged values
      * Precision parameters are trained alongside weights

    This means the "biological plausibility" applies to the backbone
    layers, while the peripheral encoder/decoder components use standard
    gradient descent (as they would in the brain's sensory periphery).
    """

    def __init__(
        self,
        model: OmniLatentModel,
        model_config: OmniLatentConfig,
        pc_config: PCConfig,
        dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader | None = None,
        seed: int = 42,
    ) -> None:
        self.model_config = model_config
        self.pc_config = pc_config
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader

        set_seed(seed)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Build PC wrapper around the model
        self.pc_net = PredictiveCodingNetwork(model, pc_config).to(self.device)

        # Separate parameter groups for different learning dynamics:
        # 1. Backbone weights -- driven by PC energy minimization
        # 2. Encoder/decoder weights -- driven by reconstruction loss
        # 3. Precision parameters -- meta-learned from total energy
        backbone_params = list(self.pc_net.model.backbone.parameters())
        precision_params = [self.pc_net.output_log_precision] + [
            layer.log_precision for layer in self.pc_net.pc_layers
        ]
        encoder_decoder_params = (
            list(self.pc_net.model.encoders.parameters())
            + list(self.pc_net.model.decoders.parameters())
            + list(self.pc_net.model.modality_embed.parameters())
            + list(self.pc_net.model.target_embed.parameters())
            + list(self.pc_net.model.target_query_gen.parameters())
        )

        self.optimizer_backbone = torch.optim.AdamW(
            backbone_params,
            lr=pc_config.learning_lr,
            weight_decay=pc_config.weight_decay,
            betas=(0.9, 0.95),
        )
        self.optimizer_peripheral = torch.optim.AdamW(
            encoder_decoder_params,
            lr=pc_config.learning_lr,
            weight_decay=pc_config.weight_decay,
            betas=(0.9, 0.95),
        )
        self.optimizer_precision = torch.optim.Adam(
            precision_params,
            lr=pc_config.learning_lr * pc_config.precision_lr_ratio,
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=pc_config.mixed_precision and self.device.type == "cuda",
        )
        self.amp_dtype = (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )

        # Task sampler and metrics (reuse existing infrastructure)
        self.task_sampler = TaskSampler(
            modalities=ALL_MODALITIES,
            self_recon_weight=0.5,
        )
        self.metrics = MetricsAggregator()
        self.global_step = 0

    def _update_lr(self) -> float:
        lr = cosine_schedule(
            self.global_step,
            self.pc_config.max_steps,
            self.pc_config.warmup_steps,
            self.pc_config.learning_lr,
        )
        for opt in [self.optimizer_backbone, self.optimizer_peripheral]:
            for pg in opt.param_groups:
                pg["lr"] = lr
        for pg in self.optimizer_precision.param_groups:
            pg["lr"] = lr * self.pc_config.precision_lr_ratio
        return lr

    def _train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """One PC training step."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        available = list(batch.keys())
        if not available:
            return {"total": 0.0}

        src_mod, tgt_mod = self.task_sampler.sample(available)

        self.optimizer_backbone.zero_grad(set_to_none=True)
        self.optimizer_peripheral.zero_grad(set_to_none=True)
        self.optimizer_precision.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda", dtype=self.amp_dtype,
            enabled=self.pc_config.mixed_precision,
        ):
            result = self.pc_net.forward_pc(
                source_modality=src_mod,
                source_data=batch[src_mod],
                target_modality=tgt_mod,
                target_data=batch[tgt_mod],
                global_step=self.global_step,
            )

        total_loss = result["total"]
        self.scaler.scale(total_loss).backward()

        self.scaler.unscale_(self.optimizer_backbone)
        self.scaler.unscale_(self.optimizer_peripheral)
        self.scaler.unscale_(self.optimizer_precision)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.pc_net.model.backbone.parameters(),
            self.pc_config.grad_clip,
        ).item()
        torch.nn.utils.clip_grad_norm_(
            list(self.pc_net.model.encoders.parameters())
            + list(self.pc_net.model.decoders.parameters()),
            self.pc_config.grad_clip,
        )

        self.scaler.step(self.optimizer_backbone)
        self.scaler.step(self.optimizer_peripheral)
        self.scaler.step(self.optimizer_precision)
        self.scaler.update()

        # Collect scalar metrics
        loss_values: dict[str, float] = {}
        for k, v in result.items():
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                loss_values[k] = v.item()
            elif isinstance(v, (int, float)):
                loss_values[k] = float(v)

        lr = self.optimizer_backbone.param_groups[0]["lr"]
        step_metrics = StepMetrics(
            loss_total=loss_values.get("total", 0.0),
            loss_per_modality={
                k: v for k, v in loss_values.items() if k != "total"
            },
            grad_norm=grad_norm,
            source_modality=src_mod,
            target_modality=tgt_mod,
            learning_rate=lr,
        )
        self.metrics.log_step(step_metrics)

        return loss_values

    def train(
        self, log_interval: int | None = None, save_dir: str | None = None
    ) -> None:
        """Main PC training loop."""
        log_interval = log_interval or self.pc_config.log_interval
        self.pc_net.train()

        data_iter = iter(self.dataloader)
        running_loss = 0.0
        running_energy = 0.0
        running_recon = 0.0
        t0 = time.time()

        for step in range(self.global_step, self.pc_config.max_steps):
            self.global_step = step
            lr = self._update_lr()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            losses = self._train_step(batch)
            running_loss += losses.get("total", 0.0)
            running_energy += losses.get("pc_energy", 0.0)
            running_recon += losses.get("recon_loss", 0.0)

            if (step + 1) % log_interval == 0:
                n = log_interval
                avg_loss = running_loss / n
                avg_energy = running_energy / n
                avg_recon = running_recon / n
                elapsed = time.time() - t0
                steps_per_sec = n / elapsed

                precisions = [
                    layer.precision.item()
                    for layer in self.pc_net.pc_layers
                ]
                avg_prec = sum(precisions) / len(precisions)
                infer_steps = self.pc_net._get_inference_steps(step)
                blend = self.pc_net._get_backprop_blend(step)

                print(
                    f"step {step + 1:>6d} | "
                    f"loss {avg_loss:.4f} | "
                    f"energy {avg_energy:.4f} | "
                    f"recon {avg_recon:.4f} | "
                    f"prec {avg_prec:.2f} | "
                    f"T_inf {infer_steps} | "
                    f"blend {blend:.2f} | "
                    f"lr {lr:.2e} | "
                    f"{steps_per_sec:.1f} steps/s"
                )

                running_loss = 0.0
                running_energy = 0.0
                running_recon = 0.0
                t0 = time.time()

            # Checkpoint saving
            if save_dir and (step + 1) % self.pc_config.save_every == 0:
                self._save_checkpoint(save_dir, step + 1)

            if (
                self.val_dataloader is not None
                and (step + 1) % (log_interval * 10) == 0
            ):
                val_loss = self.validate()
                print(f"  -> val_loss {val_loss:.4f}")

        # Save final checkpoint
        if save_dir:
            self._save_checkpoint(save_dir, self.pc_config.max_steps)

        summary = self.metrics.summary()
        print(f"\nPC Training complete. Final avg loss: {summary['avg_loss']:.4f}")
        print(f"Task distribution: {summary['task_distribution']}")

    def _save_checkpoint(self, save_dir: str, step: int) -> None:
        """Save model checkpoint to disk."""
        from pathlib import Path

        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        ckpt_path = path / f"checkpoint_{step}.pt"
        torch.save(
            {
                "step": step,
                "model_state_dict": self.pc_net.model.state_dict(),
                "pc_layers_state_dict": self.pc_net.pc_layers.state_dict(),
                "output_log_precision": self.pc_net.output_log_precision.data,
                "optimizer_backbone": self.optimizer_backbone.state_dict(),
                "optimizer_peripheral": self.optimizer_peripheral.state_dict(),
                "optimizer_precision": self.optimizer_precision.state_dict(),
                "pc_config": self.pc_config,
                "model_config": self.model_config,
            },
            ckpt_path,
        )
        print(f"  -> checkpoint saved to {ckpt_path}")

    @torch.no_grad()
    def validate(self, max_batches: int = 20) -> float:
        """Quick validation pass using standard forward."""
        self.pc_net.model.eval()
        recon_fn = ReconstructionLoss()
        total_loss = 0.0
        n = 0

        for i, batch in enumerate(self.val_dataloader):
            if i >= max_batches:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            available = list(batch.keys())
            if not available:
                continue

            src_mod = available[0]
            with torch.amp.autocast(
                "cuda", dtype=self.amp_dtype,
                enabled=self.pc_config.mixed_precision,
            ):
                result = self.pc_net.model(
                    src_mod, batch[src_mod], src_mod, batch[src_mod],
                )
                loss = recon_fn(src_mod, result["output"], batch[src_mod])

            total_loss += loss.item()
            n += 1

        self.pc_net.model.train()
        return total_loss / max(n, 1)
