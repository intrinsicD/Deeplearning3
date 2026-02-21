"""
Component 4: CfC/Mamba Hierarchical Temporal State.

Phase -1: Single Mamba-SSM tier, D=256 (two-tier collapse).
Full spec: CfC (Phi_F, Phi_M) + Mamba (Phi_S), D=512/1024/2048.

Implements a pure-PyTorch selective state space model (Mamba-style)
for maximum portability. No CUDA kernel dependency.

Also provides a flat Transformer baseline for Signal 3 comparison.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def selective_scan_sequential(
    x: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential selective scan (Mamba S6 core operation).

    Pure PyTorch implementation for portability.

    Args:
        x: [B, L, D_inner] input
        delta: [B, L, D_inner] time step
        A: [D_inner, N] state matrix (log-space)
        B: [B, L, N] input-dependent B
        C: [B, L, N] input-dependent C
        D: [D_inner] skip connection
        state: [B, D_inner, N] initial hidden state

    Returns:
        y: [B, L, D_inner] output
        final_state: [B, D_inner, N]
    """
    batch, seqlen, d_inner = x.shape
    n = A.shape[1]

    if state is None:
        state = torch.zeros(batch, d_inner, n, device=x.device, dtype=x.dtype)

    # Discretize A: A_bar = exp(delta * A)
    # delta: [B, L, D], A: [D, N] -> deltaA: [B, L, D, N]
    deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

    # delta * B: [B, L, D, 1] * [B, L, 1, N] -> [B, L, D, N]
    deltaB_x = (delta.unsqueeze(-1) * B.unsqueeze(2)) * x.unsqueeze(-1)

    ys = []
    for i in range(seqlen):
        state = deltaA[:, i] * state + deltaB_x[:, i]
        y = (state * C[:, i].unsqueeze(1)).sum(-1)  # [B, D]
        ys.append(y)

    y = torch.stack(ys, dim=1)  # [B, L, D]
    y = y + x * D.unsqueeze(0).unsqueeze(0)

    return y, state


class MambaBlock(nn.Module):
    """
    Single Mamba block with selective scan.

    Implements the S6 selective state space model:
    - Input-dependent B, C, delta (selective)
    - Local convolution for short-range context
    - Gated output projection
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = max(1, d_model // 16)

        # Input projection (x and z branches)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameters projected from input
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A parameter in log-space for numerical stability
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.expand(self.d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, L, D] input sequence
            state: [B, D_inner, N] persistent hidden state

        Returns:
            output: [B, L, D]
            new_state: [B, D_inner, N]
        """
        residual = x
        x = self.norm(x)
        B, L, D = x.shape

        # Input projection -> x branch and gate branch
        xz = self.in_proj(x)  # [B, L, 2*D_inner]
        x_branch, z = xz.chunk(2, dim=-1)  # each [B, L, D_inner]

        # Causal convolution
        x_conv = rearrange(x_branch, "b l d -> b d l")
        x_conv = self.conv1d(x_conv)[:, :, :L]  # causal: trim future
        x_branch = rearrange(x_conv, "b d l -> b l d")
        x_branch = F.silu(x_branch)

        # SSM parameter projection
        x_proj = self.x_proj(x_branch)  # [B, L, dt_rank + 2*N]
        dt, B_ssm, C_ssm = x_proj.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1,
        )

        # Project dt to full dimension
        dt = self.dt_proj(dt)  # [B, L, D_inner]
        dt = F.softplus(dt)    # ensure positive time steps

        # Selective scan
        A = -self.A_log.exp()  # [D_inner, N], negative for stability
        y, new_state = selective_scan_sequential(
            x_branch, dt, A, B_ssm, C_ssm, self.D, state,
        )

        # Gated output
        y = y * F.silu(z)
        output = self.out_proj(y) + residual

        return output, new_state


class TemporalState(nn.Module):
    """
    Temporal state module using stacked Mamba blocks.

    Processes slot features over time with persistent hidden state.
    The state carries information across sequences, enabling
    long-range temporal reasoning.

    Phase -1: Single tier (D=256).
    Full spec: Multiple tiers at different timescales.
    """

    def __init__(
        self,
        d_input: int = 1024,   # n_slots * d_slot
        d_mamba: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_mamba = d_mamba
        self.n_layers = n_layers

        # Input projection
        self.proj_in = nn.Linear(d_input, d_mamba)

        # Stacked Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=d_mamba,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])

        # Output projection back to input dim
        self.proj_out = nn.Linear(d_mamba, d_input)
        self.norm = nn.LayerNorm(d_mamba)

    def forward(
        self,
        slot_features: torch.Tensor,
        states: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            slot_features: [B, T, N_slots * D_slot] flattened slot features
            states: list of [B, D_inner, N] per-layer persistent states

        Returns:
            output: [B, T, N_slots * D_slot] temporal features
            new_states: list of [B, D_inner, N] updated states
        """
        if states is None:
            states = [None] * self.n_layers

        x = self.proj_in(slot_features)  # [B, T, D_mamba]

        new_states = []
        for block, state in zip(self.blocks, states):
            x, new_state = block(x, state)
            new_states.append(new_state)

        x = self.norm(x)
        output = self.proj_out(x)  # [B, T, D_input]

        return output, new_states

    def reset_states(self) -> None:
        """Reset persistent states (call between videos)."""
        pass  # States are passed externally; nothing to reset internally


class TransformerBaseline(nn.Module):
    """
    Flat Transformer baseline for Signal 3 comparison.

    Fixed context window; no persistent state.
    Used to demonstrate Mamba state retention advantage.
    """

    def __init__(
        self,
        d_input: int = 1024,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        context_window: int = 512,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.context_window = context_window

        self.proj_in = nn.Linear(d_input, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )

        self.proj_out = nn.Linear(d_model, d_input)
        self.norm = nn.LayerNorm(d_model)

        # Learnable positional encoding
        self.pos_emb = nn.Parameter(
            torch.randn(1, context_window, d_model) * 0.02,
        )

    def forward(
        self,
        slot_features: torch.Tensor,
        states: list | None = None,
    ) -> tuple[torch.Tensor, list]:
        """
        Args:
            slot_features: [B, T, D_input]

        Returns:
            output: [B, T, D_input]
            states: empty list (no persistent state)
        """
        B, T, D = slot_features.shape

        x = self.proj_in(slot_features)  # [B, T, D_model]

        # Truncate to context window
        if T > self.context_window:
            x = x[:, -self.context_window:]
            T = self.context_window

        # Add positional encoding
        x = x + self.pos_emb[:, :T]

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device, dtype=x.dtype,
        )

        x = self.transformer(x, mask=mask)
        x = self.norm(x)
        output = self.proj_out(x)

        # Pad back if we truncated
        if slot_features.shape[1] > self.context_window:
            pad_len = slot_features.shape[1] - self.context_window
            padding = torch.zeros(
                B, pad_len, self.d_input,
                device=output.device, dtype=output.dtype,
            )
            output = torch.cat([padding, output], dim=1)

        return output, []
