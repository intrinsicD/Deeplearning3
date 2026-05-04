"""Compatibility facade for canonical OmniLatent protocol containers.

The shared protocol/state dataclasses now live in :mod:`omnilatent.protocol` so
they can be used by OmniLatent, MMWM, tools, and future agent runtimes without a
dependency on the ``MMWM`` package. This module intentionally re-exports the
exact same class objects to preserve existing ``from MMWM.containers import ...``
call sites and checkpoint pickles that reference these names.
"""

from __future__ import annotations

from omnilatent.protocol import (
    AgentTrace,
    AgentTraceStep,
    LatentPacket,
    LatentState,
    MemoryState,
    ModelOutput,
    ObservationPacket,
    TargetSpec,
    TokenState,
    ToolContext,
    ToolDescriptor,
    ToolResult,
    TransitionOutput,
)

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
