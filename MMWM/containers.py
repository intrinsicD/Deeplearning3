"""Data containers for the Multimodal Latent World Model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class ObservationPacket:
    """Unified container for multimodal observations.

    Typical keys:
      modalities: {
         "text": LongTensor[B, T]
         "image": FloatTensor[B, C, H, W]
         "audio": FloatTensor[B, C, S]
         "vector": FloatTensor[B, D]
      }
      mask: optional masks per modality
      meta: arbitrary metadata
    """

    modalities: Dict[str, torch.Tensor]
    masks: Dict[str, torch.Tensor] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def device(self) -> torch.device:
        for value in self.modalities.values():
            return value.device
        return torch.device("cpu")


@dataclass
class LatentState:
    """Role-based latent container.

    All tensors are expected to be shaped [B, D].
    """

    z_sem: torch.Tensor
    z_dyn: Optional[torch.Tensor] = None
    z_ctrl: Optional[torch.Tensor] = None
    z_mem: Optional[torch.Tensor] = None
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)

    def primary(self) -> torch.Tensor:
        parts = [self.z_sem]
        if self.z_dyn is not None:
            parts.append(self.z_dyn)
        if self.z_ctrl is not None:
            parts.append(self.z_ctrl)
        if self.z_mem is not None:
            parts.append(self.z_mem)
        return torch.cat(parts, dim=-1)

    def detach(self) -> "LatentState":
        return LatentState(
            z_sem=self.z_sem.detach(),
            z_dyn=None if self.z_dyn is None else self.z_dyn.detach(),
            z_ctrl=None if self.z_ctrl is None else self.z_ctrl.detach(),
            z_mem=None if self.z_mem is None else self.z_mem.detach(),
            extras={k: v.detach() for k, v in self.extras.items()},
        )


@dataclass
class MemoryState:
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
    """Stable protocol passed between tools.

    Tools should read/write only through this packet so they remain swappable.
    """

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
