"""Abstract interfaces and component registry for the world model."""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .containers import LatentState, MemoryState, ObservationPacket


# ============================================================
# Registry / factory
# ============================================================


class Registry:
    def __init__(self) -> None:
        self._items: Dict[str, Callable[..., nn.Module]] = {}

    def register(self, name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
        def decorator(obj: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
            if name in self._items:
                raise ValueError(f"Duplicate registration: {name}")
            self._items[name] = obj
            return obj

        return decorator

    def build(self, name: str, **kwargs: Any) -> nn.Module:
        if name not in self._items:
            raise KeyError(f"Unknown component: {name}. Available: {list(self._items)}")
        return self._items[name](**kwargs)

    def names(self) -> List[str]:
        return sorted(self._items.keys())


ENCODERS = Registry()
LATENT_PROJECTORS = Registry()
MEMORIES = Registry()
ACTION_ENCODERS = Registry()
CONDITIONERS = Registry()
TRANSITION_CORES = Registry()
PREDICTION_HEADS = Registry()
REGULARIZERS = Registry()
DECODERS = Registry()


# ============================================================
# Abstract interfaces
# ============================================================


class IEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, obs: ObservationPacket) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class ILatentProjector(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, encoded: Dict[str, torch.Tensor]) -> LatentState:
        raise NotImplementedError


class IMemory(nn.Module, abc.ABC):
    @abc.abstractmethod
    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, latent: LatentState, action_repr: torch.Tensor, state: MemoryState) -> MemoryState:
        raise NotImplementedError

    @abc.abstractmethod
    def read(self, state: MemoryState) -> torch.Tensor:
        raise NotImplementedError


class IActionEncoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class IConditioner(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, core_input: torch.Tensor, action_repr: torch.Tensor, memory_ctx: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class ITransitionCore(nn.Module, abc.ABC):
    """Transition cores operate on the conditioned input and return hidden state.

    IMPORTANT: Transition cores must NOT update MemoryState.hidden or
    MemoryState.context — they should pass memory_state through unchanged.
    Memory updates are the sole responsibility of the IMemory module,
    called by the top-level model after the transition core runs.
    """

    @abc.abstractmethod
    def forward(self, conditioned_input: torch.Tensor, memory_state: MemoryState) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError


class IPredictionHead(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, hidden: torch.Tensor, reference: LatentState) -> LatentState:
        raise NotImplementedError


class IRegularizer(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, latent: LatentState) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class IDecoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, latent: LatentState, context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
