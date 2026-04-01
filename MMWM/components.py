"""Latent projectors, memory modules, action encoders, conditioners, and regularizers."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .containers import LatentState, MemoryState
from .helpers import MLP
from .interfaces import (
    ACTION_ENCODERS,
    CONDITIONERS,
    LATENT_PROJECTORS,
    MEMORIES,
    PREDICTION_HEADS,
    REGULARIZERS,
    IActionEncoder,
    IConditioner,
    ILatentProjector,
    IMemory,
    IPredictionHead,
    IRegularizer,
)


# ============================================================
# Latent projectors
# ============================================================


@LATENT_PROJECTORS.register("role_split_mlp")
class RoleSplitLatentProjector(ILatentProjector):
    def __init__(self, input_dim: int = 256, latent_dim: int = 128, use_batchnorm: bool = True) -> None:
        super().__init__()
        self.sem = nn.Linear(input_dim, latent_dim)
        self.dyn = nn.Linear(input_dim, latent_dim)
        self.ctrl = nn.Linear(input_dim, latent_dim)
        self.mem = nn.Linear(input_dim, latent_dim)
        self.bn = nn.BatchNorm1d(input_dim) if use_batchnorm else nn.Identity()

    def forward(self, encoded: Dict[str, torch.Tensor]) -> LatentState:
        fused = self.bn(encoded["fused"])
        return LatentState(
            z_sem=self.sem(fused),
            z_dyn=self.dyn(fused),
            z_ctrl=self.ctrl(fused),
            z_mem=self.mem(fused),
            extras={k: v for k, v in encoded.items()},
        )


@LATENT_PROJECTORS.register("adaptive_role_split_mlp")
class AdaptiveRoleSplitLatentProjector(ILatentProjector):
    """Learnable capacity allocation across latent roles.

    Instead of fixed equal-size projections, each role attends over a shared
    higher-dimensional intermediate space with learned soft gates, allowing the
    model to allocate more capacity to roles that need it.
    """

    def __init__(
        self,
        input_dim: int = 256,
        latent_dim: int = 128,
        intermediate_dim: int = 512,
        use_batchnorm: bool = True,
        num_roles: int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_roles = num_roles
        self.bn = nn.BatchNorm1d(input_dim) if use_batchnorm else nn.Identity()
        self.shared_proj = MLP([input_dim, intermediate_dim, intermediate_dim])
        self.role_queries = nn.Parameter(torch.randn(num_roles, intermediate_dim) * 0.02)
        self.gate_proj = nn.Linear(intermediate_dim, intermediate_dim)
        self.role_out = nn.ModuleList([nn.Linear(intermediate_dim, latent_dim) for _ in range(num_roles)])
        self.capacity_logits = nn.Parameter(torch.zeros(num_roles))

    def forward(self, encoded: Dict[str, torch.Tensor]) -> LatentState:
        fused = self.bn(encoded["fused"])
        shared = self.shared_proj(fused)  # [B, intermediate_dim]

        capacity_weights = torch.softmax(self.capacity_logits, dim=0)  # [num_roles]

        role_features = []
        for i in range(self.num_roles):
            query = self.role_queries[i]  # [intermediate_dim]
            gate = torch.sigmoid(self.gate_proj(shared * query.unsqueeze(0)))
            gated = shared * gate * capacity_weights[i]
            role_features.append(self.role_out[i](gated))

        extras = {k: v for k, v in encoded.items()}
        extras["capacity_weights"] = capacity_weights.detach()
        return LatentState(
            z_sem=role_features[0],
            z_dyn=role_features[1],
            z_ctrl=role_features[2],
            z_mem=role_features[3],
            extras=extras,
        )


# ============================================================
# Memory modules
# ============================================================


@MEMORIES.register("identity")
class IdentityMemory(IMemory):
    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        return MemoryState(context=torch.zeros(batch_size, self.latent_dim, device=device), hidden=None)

    def update(self, latent: LatentState, action_repr: torch.Tensor, state: MemoryState) -> MemoryState:
        return MemoryState(
            context=latent.z_mem if latent.z_mem is not None else latent.z_sem,
            hidden=state.hidden,
        )

    def read(self, state: MemoryState) -> torch.Tensor:
        assert state.context is not None
        return state.context


@MEMORIES.register("gru")
class GRUMemory(IMemory):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(input_dim, hidden_dim)

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        return MemoryState(context=hidden, hidden=hidden)

    def update(self, latent: LatentState, action_repr: torch.Tensor, state: MemoryState) -> MemoryState:
        mem_part = latent.z_mem if latent.z_mem is not None else latent.z_sem
        x = torch.cat([mem_part, action_repr], dim=-1)
        prev = state.hidden
        if prev is None:
            raise RuntimeError("GRUMemory requires state.hidden")
        hidden = self.cell(x, prev)
        return MemoryState(context=hidden, hidden=hidden)

    def read(self, state: MemoryState) -> torch.Tensor:
        assert state.context is not None
        return state.context


@MEMORIES.register("mamba_ssm")
class MambaSSMMemory(IMemory):
    """Selective SSM-inspired memory with stable diagonal dynamics."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, expansion: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        inner_dim = hidden_dim * expansion
        self.in_proj = nn.Linear(input_dim, inner_dim)
        self.delta_proj = nn.Linear(inner_dim, hidden_dim)
        self.b_proj = nn.Linear(inner_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.a_log = nn.Parameter(torch.zeros(hidden_dim))

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        return MemoryState(context=hidden, hidden=hidden, extras={"memory_type": "mamba_ssm"})

    def update(self, latent: LatentState, action_repr: torch.Tensor, state: MemoryState) -> MemoryState:
        mem_part = latent.z_mem if latent.z_mem is not None else latent.z_sem
        x = torch.cat([mem_part, action_repr], dim=-1)
        prev = state.hidden
        if prev is None:
            raise RuntimeError("MambaSSMMemory requires state.hidden")

        u = F.silu(self.in_proj(x))
        delta = F.softplus(self.delta_proj(u))
        a = -torch.exp(self.a_log).unsqueeze(0)
        b = self.b_proj(u)
        hidden = torch.exp(delta * a) * prev + delta * b
        context = self.out_proj(F.silu(self.c_proj(hidden)))
        return MemoryState(context=context, hidden=hidden, extras=dict(state.extras))

    def read(self, state: MemoryState) -> torch.Tensor:
        assert state.context is not None
        return state.context


# ============================================================
# Action encoders
# ============================================================


@ACTION_ENCODERS.register("mlp")
class MLPActionEncoder(IActionEncoder):
    def __init__(self, action_dim: int = 32, action_embed_dim: int = 128) -> None:
        super().__init__()
        self.net = MLP([action_dim, action_embed_dim, action_embed_dim])

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.net(action)


@ACTION_ENCODERS.register("embedding")
class DiscreteActionEncoder(IActionEncoder):
    def __init__(self, num_actions: int = 64, action_embed_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_actions, action_embed_dim)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        if action.dtype != torch.long:
            action = action.long()
        return self.embedding(action.squeeze(-1) if action.ndim > 1 and action.shape[-1] == 1 else action)


# ============================================================
# Conditioning adapters
# ============================================================


@CONDITIONERS.register("concat_mlp")
class ConcatConditioner(IConditioner):
    def __init__(self, latent_dim: int = 512, action_dim: int = 128, memory_dim: int = 128, out_dim: int = 512) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.net = MLP([latent_dim + action_dim + memory_dim, out_dim, out_dim])

    def forward(self, core_input: torch.Tensor, action_repr: torch.Tensor, memory_ctx: Optional[torch.Tensor]) -> torch.Tensor:
        if memory_ctx is None:
            memory_ctx = torch.zeros(core_input.shape[0], 0, device=core_input.device, dtype=core_input.dtype)
        return self.net(torch.cat([core_input, action_repr, memory_ctx], dim=-1))


@CONDITIONERS.register("film")
class FiLMConditioner(IConditioner):
    def __init__(self, latent_dim: int = 512, action_dim: int = 128, memory_dim: int = 128) -> None:
        super().__init__()
        self.gamma = nn.Linear(action_dim + memory_dim, latent_dim)
        self.beta = nn.Linear(action_dim + memory_dim, latent_dim)

    def forward(self, core_input: torch.Tensor, action_repr: torch.Tensor, memory_ctx: Optional[torch.Tensor]) -> torch.Tensor:
        if memory_ctx is None:
            memory_ctx = torch.zeros(core_input.shape[0], 0, device=core_input.device, dtype=core_input.dtype)
        cond = torch.cat([action_repr, memory_ctx], dim=-1)
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        return core_input * (1.0 + gamma) + beta


# ============================================================
# Prediction heads
# ============================================================


@PREDICTION_HEADS.register("role_split")
class RoleSplitPredictionHead(IPredictionHead):
    def __init__(self, hidden_dim: int = 512, latent_dim: int = 128) -> None:
        super().__init__()
        self.sem = nn.Linear(hidden_dim, latent_dim)
        self.dyn = nn.Linear(hidden_dim, latent_dim)
        self.ctrl = nn.Linear(hidden_dim, latent_dim)
        self.mem = nn.Linear(hidden_dim, latent_dim)
        self.uncertainty = nn.Linear(hidden_dim, latent_dim)

    def forward(self, hidden: torch.Tensor, reference: LatentState) -> LatentState:
        extras = dict(reference.extras)
        extras["predicted_logvar"] = self.uncertainty(hidden)
        return LatentState(
            z_sem=self.sem(hidden),
            z_dyn=self.dyn(hidden) if reference.z_dyn is not None else None,
            z_ctrl=self.ctrl(hidden) if reference.z_ctrl is not None else None,
            z_mem=self.mem(hidden) if reference.z_mem is not None else None,
            extras=extras,
        )


# ============================================================
# Regularizers
# ============================================================


@REGULARIZERS.register("none")
class NoRegularizer(IRegularizer):
    def forward(self, latent: LatentState) -> Dict[str, torch.Tensor]:
        device = latent.z_sem.device
        return {"regularizer_total": torch.zeros((), device=device)}


@REGULARIZERS.register("sigreg_like")
class SIGRegLike(IRegularizer):
    """Anti-collapse regularizer: encourages variance, discourages covariance collapse."""

    def __init__(self, variance_weight: float = 1.0, covariance_weight: float = 0.04, target_std: float = 1.0) -> None:
        super().__init__()
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.target_std = target_std

    @staticmethod
    def _cov_offdiag(x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / max(x.shape[0] - 1, 1)
        offdiag = cov - torch.diag(torch.diag(cov))
        return offdiag.pow(2).mean()

    def _apply_reg(self, z: torch.Tensor, prefix: str) -> Dict[str, torch.Tensor]:
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        variance_loss = F.relu(self.target_std - std).mean()
        covariance_loss = self._cov_offdiag(z)
        total = self.variance_weight * variance_loss + self.covariance_weight * covariance_loss
        return {
            f"{prefix}_variance_loss": variance_loss,
            f"{prefix}_covariance_loss": covariance_loss,
            f"{prefix}_reg_total": total,
        }

    def forward(self, latent: LatentState) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=latent.z_sem.device)
        for name, tensor in [("z_sem", latent.z_sem), ("z_dyn", latent.z_dyn), ("z_ctrl", latent.z_ctrl), ("z_mem", latent.z_mem)]:
            if tensor is None:
                continue
            part = self._apply_reg(tensor, name)
            losses.update(part)
            total = total + part[f"{name}_reg_total"]
        losses["regularizer_total"] = total
        return losses
