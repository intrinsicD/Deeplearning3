"""Canonical agent/kernel protocol types for OmniLatent.

This module is intentionally dependency-light and must not import :mod:`MMWM`.
It hosts the shared observation, latent-state, memory, model-output, and tool
contracts used by the agentic OmniLatent architecture. Legacy MMWM imports are
kept compatible by re-exporting these exact classes from ``MMWM.containers``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast

import torch


@dataclass
class ObservationPacket:
    """Unified container for multimodal observations.

    Typical modality keys:
      - ``text``: LongTensor[B, T]
      - ``image``: FloatTensor[B, C, H, W]
      - ``audio``: FloatTensor[B, C, S]
      - ``video``: FloatTensor[B, T, C, H, W]
      - ``vector``: FloatTensor[B, D]

    ``masks`` and ``meta`` are free-form side channels used by encoders,
    routers, and data adapters.
    """

    modalities: Dict[str, torch.Tensor]
    masks: Dict[str, torch.Tensor] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def device(self) -> torch.device:
        """Return the device of the first modality tensor, or CPU if empty."""
        for value in self.modalities.values():
            return value.device
        return torch.device("cpu")


@dataclass
class TargetSpec:
    """Target request for multimodal/agentic forward passes.

    ``modality`` selects the decoder. ``data`` optionally provides
    teacher-forced target data (currently used for text, and returned as the
    training target for all modalities). ``metadata`` is intentionally free-form
    for future decoding policies and objective tags.
    """

    modality: str
    data: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenState:
    """Canonical token-stream state used before/after latent fusion.

    ``tokens`` is expected to be shaped ``[B, T, D]`` for continuous embeddings
    or ``[B, T]`` for discrete IDs depending on the producer. Optional
    ``attention_mask`` follows PyTorch convention where truthy positions are
    valid/visible unless a downstream attention policy states otherwise.
    """

    tokens: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None
    modality: Optional[str] = None
    segment_ids: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def detach(self) -> "TokenState":
        return TokenState(
            tokens=self.tokens.detach(),
            attention_mask=None if self.attention_mask is None else self.attention_mask.detach(),
            modality=self.modality,
            segment_ids=None if self.segment_ids is None else self.segment_ids.detach(),
            positions=None if self.positions is None else self.positions.detach(),
            metadata=dict(self.metadata),
        )


@dataclass
class LatentState:
    """Role-based latent container.

    All role tensors are expected to be shaped ``[B, D]``. ``z_ctx`` stores a
    checkpointable/portable latent context for save/load, resume, and branching
    workflows. It is distinct from :class:`MemoryState`, which stores live
    rollout memory.
    """

    z_sem: torch.Tensor
    z_dyn: Optional[torch.Tensor] = None
    z_ctrl: Optional[torch.Tensor] = None
    z_ctx: Optional[torch.Tensor] = None
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)

    def primary(self) -> torch.Tensor:
        """Concatenate available role tensors in stable semantic order."""
        parts = [self.z_sem]
        if self.z_dyn is not None:
            parts.append(self.z_dyn)
        if self.z_ctrl is not None:
            parts.append(self.z_ctrl)
        if self.z_ctx is not None:
            parts.append(self.z_ctx)
        return torch.cat(parts, dim=-1)

    def detach(self) -> "LatentState":
        return LatentState(
            z_sem=self.z_sem.detach(),
            z_dyn=None if self.z_dyn is None else self.z_dyn.detach(),
            z_ctrl=None if self.z_ctrl is None else self.z_ctrl.detach(),
            z_ctx=None if self.z_ctx is None else self.z_ctx.detach(),
            extras={k: v.detach() for k, v in self.extras.items()},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize latent state using stable checkpoint field names."""
        return {
            "z_sem": self.z_sem,
            "z_dyn": self.z_dyn,
            "z_ctrl": self.z_ctrl,
            "z_ctx": self.z_ctx,
            "extras": dict(self.extras),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LatentState":
        """Deserialize with backward compatibility for legacy ``z_mem``.

        Older checkpoints may store ``z_mem`` instead of ``z_ctx``; when
        ``z_ctx`` is absent we map ``z_mem`` into the canonical context role.
        """
        z_ctx = payload.get("z_ctx")
        if z_ctx is None and "z_mem" in payload:
            z_ctx = payload["z_mem"]
        return cls(
            z_sem=payload["z_sem"],
            z_dyn=payload.get("z_dyn"),
            z_ctrl=payload.get("z_ctrl"),
            z_ctx=z_ctx,
            extras=cast(Dict[str, torch.Tensor], dict(payload.get("extras", {}))),
        )


@dataclass
class MemoryState:
    """Live rollout memory state, distinct from checkpointable ``z_ctx``."""

    context: Optional[torch.Tensor] = None
    hidden: Optional[Any] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def detach(self) -> "MemoryState":
        def _detach(x: Any) -> Any:
            if isinstance(x, torch.Tensor):
                return x.detach()
            if isinstance(x, (tuple, list)):
                return type(x)(_detach(item) for item in x)
            return x

        return MemoryState(
            context=_detach(self.context),
            hidden=_detach(self.hidden),
            extras={k: _detach(v) for k, v in self.extras.items()},
        )


@dataclass
class TransitionOutput:
    next_latent: LatentState
    next_memory: MemoryState
    uncertainty: Optional[torch.Tensor] = None
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class ModelOutput:
    current_latent: LatentState
    predicted_next_latent: LatentState
    target_next_latent: Optional[LatentState]
    next_memory: MemoryState
    decoder_outputs: Dict[str, torch.Tensor] = field(default_factory=dict)
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class LatentPacket:
    """Stable protocol passed between neural ports, routers, and tools."""

    state: LatentState
    confidence: Optional[torch.Tensor] = None
    source_tool: Optional[str] = None
    timestamp: Optional[int] = None
    trace: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    memory_state: Optional[MemoryState] = None
    observation: Optional[ObservationPacket] = None
    action: Optional[torch.Tensor] = None
    raw_inputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    packet: Optional[LatentPacket]
    raw_output: Optional[Any] = None
    confidence: Optional[torch.Tensor] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    side_effects: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDescriptor:
    tool_id: str
    kind: str
    version: str
    path: Optional[str] = None
    entry_point: Optional[str] = None
    input_schema: str = "LatentPacket"
    output_schema: str = "ToolResult"
    reads_slots: tuple = tuple()
    writes_slots: tuple = tuple()
    benchmark_id: Optional[str] = None
    estimated_cost: float = 1.0
    estimated_latency_ms: float = 1.0
    memory_mb: float = 0.0
    tags: tuple = tuple()
    enabled: bool = True
    binary: bool = True
    lazy_load: bool = True
    unload_after_call: bool = False
    device_preference: str = "auto"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTraceStep:
    """One explicit, auditable agent graph transition."""

    node_type: str
    input_packet_id: Optional[str] = None
    output_packet_id: Optional[str] = None
    selected_action: Optional[str] = None
    stop_prob: Optional[float] = None
    hook_gates: Dict[str, float] = field(default_factory=dict)
    tool_id: Optional[str] = None
    side_effects: Dict[str, Any] = field(default_factory=dict)
    loss_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrace:
    """Ordered trace emitted by an explicit agent graph/runtime."""

    steps: List[AgentTraceStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def append(self, step: AgentTraceStep) -> None:
        self.steps.append(step)


__all__ = [
    "AgentTrace",
    "AgentTraceStep",
    "LatentPacket",
    "LatentState",
    "MemoryState",
    "ModelOutput",
    "ObservationPacket",
    "TargetSpec",
    "TokenState",
    "ToolContext",
    "ToolDescriptor",
    "ToolResult",
    "TransitionOutput",
]


