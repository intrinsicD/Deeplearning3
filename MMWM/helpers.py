"""Shared neural building blocks."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dims: Sequence[int], dropout: float = 0.0, activation: Callable[[], nn.Module] = nn.GELU) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class BlockAttentionResidual(nn.Module):
    """Block-level Attention Residuals (AttnRes-style) over depth.

    Replaces fixed additive residual accumulation across layers with
    learned attention over previous block states.
    """

    def __init__(self, dim: int, attn_dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim
        self.attn_dim = attn_dim or dim
        self.q_proj = nn.Linear(dim, self.attn_dim)
        self.k_proj = nn.Linear(dim, self.attn_dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = self.attn_dim ** -0.5

    def forward(self, current: torch.Tensor, history: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(history) == 0:
            return current
        h = torch.stack(list(history), dim=1)
        q = self.q_proj(current).unsqueeze(1)
        k = self.k_proj(h)
        v = self.v_proj(h)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        weights = scores.softmax(dim=-1)
        mixed = torch.matmul(weights, v).squeeze(1)
        return current + self.out_proj(mixed)


class LowRankHyperAdapter(nn.Module):
    """Hypernetwork-generated low-rank per-step modulation."""

    def __init__(self, dim: int, rank: int = 8, hyper_hidden: int = 256) -> None:
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.hyper = MLP([dim, hyper_hidden, hyper_hidden, 2 * dim * rank])
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = self.hyper(x)
        a, b = params.split(self.dim * self.rank, dim=-1)
        a = a.view(x.shape[0], self.dim, self.rank)
        b = b.view(x.shape[0], self.rank, self.dim)
        delta = torch.bmm(torch.bmm(x.unsqueeze(1), a), b).squeeze(1)
        return x + self.scale * delta


class AdaptiveHaltingHead(nn.Module):
    """Predicts whether recurrent latent iteration should stop."""

    def __init__(self, dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = MLP([dim, hidden_dim, 1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
