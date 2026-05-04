"""Canonical tool system for OmniLatent agent runtimes.

This module adapts the useful MMWM tool concepts to the canonical protocol
without importing ``MMWM``. Tools consume :class:`LatentPacket` and
:class:`ToolContext`, and must return :class:`ToolResult`.
"""

from __future__ import annotations

import abc
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from omnilatent.protocol import LatentPacket, LatentState, ToolContext, ToolDescriptor, ToolResult


class BaseTool(nn.Module, abc.ABC):
    """Base class for differentiable or lazy-loaded agent tools."""

    descriptor: ToolDescriptor

    def __init__(self, descriptor: ToolDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor

    @abc.abstractmethod
    def run(self, packet: LatentPacket, context: ToolContext) -> ToolResult:
        raise NotImplementedError


class FunctionTool(BaseTool):
    """Wrap a Python callable as a canonical tool."""

    def __init__(self, descriptor: ToolDescriptor, fn: Any) -> None:
        super().__init__(descriptor)
        self.fn = fn

    def run(self, packet: LatentPacket, context: ToolContext) -> ToolResult:
        result = self.fn(packet, context)
        if not isinstance(result, ToolResult):
            raise TypeError(f"Function tool {self.descriptor.tool_id} returned {type(result).__name__}, expected ToolResult")
        return result


class BinaryToolLoader:
    """Loads neural tools only when needed via lazy imports."""

    def __init__(self, device: torch.device, keep_hot: int = 2) -> None:
        if keep_hot < 0:
            raise ValueError("keep_hot must be >= 0")
        self.device = device
        self.keep_hot = keep_hot
        self._loaded: Dict[str, BaseTool] = {}
        self._lru: Deque[str] = deque()

    def _touch(self, tool_id: str) -> None:
        try:
            self._lru.remove(tool_id)
        except ValueError:
            pass
        self._lru.append(tool_id)
        while len(self._lru) > self.keep_hot:
            old = self._lru.popleft()
            self.unload(old)

    def load(self, descriptor: ToolDescriptor) -> BaseTool:
        if descriptor.tool_id in self._loaded:
            self._touch(descriptor.tool_id)
            return self._loaded[descriptor.tool_id]
        if descriptor.entry_point is None:
            raise ValueError(f"Tool {descriptor.tool_id} has no entry_point")

        module_name, func_name = descriptor.entry_point.split(":", maxsplit=1)
        module = __import__(module_name, fromlist=[func_name])
        builder = getattr(module, func_name)
        tool = builder(descriptor)
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Builder for {descriptor.tool_id} returned {type(tool).__name__}, expected BaseTool")
        tool.to(self.device)
        tool.eval()
        self._loaded[descriptor.tool_id] = tool
        self._touch(descriptor.tool_id)
        return tool

    def unload(self, tool_id: str) -> None:
        tool = self._loaded.pop(tool_id, None)
        if tool is None:
            return
        del tool
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            self._lru.remove(tool_id)
        except ValueError:
            pass

    def call(self, descriptor: ToolDescriptor, packet: LatentPacket, context: ToolContext) -> ToolResult:
        tool = self.load(descriptor)
        with torch.no_grad():
            result = tool.run(packet, context)
        if not isinstance(result, ToolResult):
            raise TypeError(f"Tool {descriptor.tool_id} returned {type(result).__name__}, expected ToolResult")
        if descriptor.unload_after_call:
            self.unload(descriptor.tool_id)
        return result


class ToolRegistrySystem:
    """Descriptor registry plus lazy invocation surface."""

    def __init__(self, loader: BinaryToolLoader) -> None:
        self.loader = loader
        self._descriptors: Dict[str, ToolDescriptor] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, descriptor: ToolDescriptor, alias: Optional[str] = None) -> None:
        self._descriptors[descriptor.tool_id] = descriptor
        if alias is not None:
            self._aliases[alias] = descriptor.tool_id

    def get(self, tool_id_or_alias: str) -> ToolDescriptor:
        tool_id = self._aliases.get(tool_id_or_alias, tool_id_or_alias)
        return self._descriptors[tool_id]

    def list_by_kind(self, kind: str) -> List[ToolDescriptor]:
        return [d for d in self._descriptors.values() if d.kind == kind and d.enabled]

    def call(self, tool_id_or_alias: str, packet: LatentPacket, context: ToolContext) -> ToolResult:
        descriptor = self.get(tool_id_or_alias)
        return self.loader.call(descriptor, packet, context)

    def replace(self, old_tool_id: str, new_descriptor: ToolDescriptor) -> None:
        aliases_to_move = [alias for alias, target in self._aliases.items() if target == old_tool_id]
        self._descriptors[new_descriptor.tool_id] = new_descriptor
        for alias in aliases_to_move:
            self._aliases[alias] = new_descriptor.tool_id


class CriticalExperienceBuffer:
    def __init__(self) -> None:
        self.examples: List[Dict[str, Any]] = []

    def add(self, example: Dict[str, Any]) -> None:
        self.examples.append(example)

    def sample(self, benchmark_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        items = [ex for ex in self.examples if benchmark_id is None or benchmark_id in ex.get("benchmarks", [])]
        return items if limit is None else items[:limit]


class ToolJudge:
    def __init__(self, critical_buffer: CriticalExperienceBuffer) -> None:
        self.critical_buffer = critical_buffer

    def score_candidate(
        self,
        candidate: ToolDescriptor,
        incumbent: Optional[ToolDescriptor],
        local_metrics: Dict[str, float],
        regression_metrics: Dict[str, float],
    ) -> Tuple[bool, Dict[str, float]]:
        score = 0.0
        score += local_metrics.get("quality", 0.0)
        score -= 0.1 * local_metrics.get("latency_ms", candidate.estimated_latency_ms)
        score -= 0.01 * local_metrics.get("memory_mb", candidate.memory_mb)
        regressions = regression_metrics.get("critical_failures", 0.0)
        score -= 100.0 * regressions
        accepted = regressions <= 0.0 and score > 0.0
        return accepted, {
            "score": score,
            "critical_failures": regressions,
            **local_metrics,
            **regression_metrics,
        }


class CandidatePromoter:
    def __init__(self, registry: ToolRegistrySystem, judge: ToolJudge) -> None:
        self.registry = registry
        self.judge = judge

    def maybe_promote(
        self,
        old_tool_id: str,
        candidate: ToolDescriptor,
        local_metrics: Dict[str, float],
        regression_metrics: Dict[str, float],
    ) -> Tuple[bool, Dict[str, float]]:
        incumbent = self.registry.get(old_tool_id)
        accepted, report = self.judge.score_candidate(candidate, incumbent, local_metrics, regression_metrics)
        if accepted:
            self.registry.replace(old_tool_id, candidate)
        return accepted, report


class _MLP(nn.Module):
    def __init__(self, dims: List[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentRouter(nn.Module):
    """Canonical latent router operating on :class:`LatentPacket`."""

    def __init__(self, latent_dim: int, num_actions: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.policy = _MLP([latent_dim, hidden_dim, hidden_dim, num_actions])
        self.stop_head = _MLP([latent_dim, hidden_dim, 1])

    def forward(self, packet: LatentPacket) -> Dict[str, torch.Tensor]:
        x = packet.state.primary()
        return {
            "action_logits": self.policy(x),
            "stop_logit": self.stop_head(x),
        }


class MemoryReadTool(BaseTool):
    def __init__(self, descriptor: ToolDescriptor, memory_bank: Dict[str, Any]) -> None:
        super().__init__(descriptor)
        self.memory_bank = memory_bank

    def run(self, packet: LatentPacket, context: ToolContext) -> ToolResult:
        key = context.raw_inputs.get("memory_key", "default")
        record = self.memory_bank.get(key)
        value = record.get("value") if isinstance(record, dict) else record
        if value is None:
            value = torch.zeros_like(packet.state.z_sem)
        elif isinstance(value, torch.Tensor):
            value = value.to(device=packet.state.z_sem.device, dtype=packet.state.z_sem.dtype)
        else:
            raise TypeError(f"Memory value for {key!r} must be Tensor-compatible, got {type(value).__name__}")
        next_state = LatentState(
            z_sem=packet.state.z_sem,
            z_dyn=packet.state.z_dyn,
            z_ctrl=packet.state.z_ctrl,
            z_ctx=value,
            extras=dict(packet.state.extras),
        )
        next_packet = LatentPacket(
            state=next_state,
            confidence=packet.confidence,
            source_tool=self.descriptor.tool_id,
            timestamp=packet.timestamp,
            trace=packet.trace + [self.descriptor.tool_id],
            metadata=dict(packet.metadata),
        )
        return ToolResult(packet=next_packet)


class MemoryWriteTool(BaseTool):
    def __init__(self, descriptor: ToolDescriptor, memory_bank: Dict[str, Any]) -> None:
        super().__init__(descriptor)
        self.memory_bank = memory_bank

    def run(self, packet: LatentPacket, context: ToolContext) -> ToolResult:
        key = context.raw_inputs.get("memory_key", "default")
        value = packet.state.z_ctx if packet.state.z_ctx is not None else packet.state.z_sem
        self.memory_bank[key] = {
            "value": value.detach().cpu(),
            "episode_id": context.raw_inputs.get("episode_id"),
            "step": context.raw_inputs.get("step"),
            "confidence": packet.confidence.detach().cpu() if isinstance(packet.confidence, torch.Tensor) else packet.confidence,
        }
        return ToolResult(packet=packet, side_effects={"memory_written": key})


class ToolExecutionEngine:
    """Iterative router + tool executor with explicit batch limitation.

    Current execution invokes exactly one tool per route step. If a batch routes
    samples to different tool IDs, a clear error is raised rather than silently
    executing the wrong tool for some samples.
    """

    def __init__(
        self,
        registry: ToolRegistrySystem,
        router: LatentRouter,
        action_id_to_tool: Dict[int, str],
        device: torch.device,
    ) -> None:
        self.registry = registry
        self.router = router.to(device)
        self.action_id_to_tool = action_id_to_tool
        self.device = device

    @torch.no_grad()
    def iterate(
        self,
        packet: LatentPacket,
        context: ToolContext,
        max_steps: int = 8,
    ) -> Tuple[LatentPacket, List[Dict[str, Any]]]:
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        trace: List[Dict[str, Any]] = []
        current = packet
        for step in range(max_steps):
            route = self.router(current)
            stop_prob = torch.sigmoid(route["stop_logit"]).squeeze(-1)
            action_ids = route["action_logits"].argmax(dim=-1)
            stop_all = bool((stop_prob > 0.5).all())
            unique_action_ids = torch.unique(action_ids)
            if unique_action_ids.numel() != 1:
                raise ValueError(
                    "ToolExecutionEngine received a batched route with different tool actions per sample. "
                    "Current iterate() executes one tool call per step; use batch size 1 or a per-sample "
                    "tool execution path."
                )
            action_id = int(unique_action_ids.item())
            tool_name = self.action_id_to_tool.get(action_id)
            trace.append({
                "step": step,
                "stop_prob_mean": float(stop_prob.mean().item()),
                "stop_prob": stop_prob.detach().cpu().tolist(),
                "action_ids": action_ids.detach().cpu().tolist(),
                "action_id": action_id,
                "tool": tool_name,
            })
            if stop_all:
                trace[-1]["stopped"] = True
                break
            if tool_name is None:
                raise KeyError(f"No tool registered for action id {action_id}")
            result = self.registry.call(tool_name, current, context)
            if not isinstance(result, ToolResult):
                raise TypeError(f"Registry returned {type(result).__name__}, expected ToolResult")
            if result.packet is not None:
                current = result.packet
            trace[-1]["confidence"] = _confidence_to_float(result.confidence)
            trace[-1]["side_effects"] = dict(result.side_effects)
        return current, trace


def _confidence_to_float(confidence: torch.Tensor | float | None) -> float | None:
    if confidence is None:
        return None
    if isinstance(confidence, torch.Tensor):
        return float(confidence.detach().mean().cpu().item())
    return float(confidence)


def build_memory_read_tool(descriptor: ToolDescriptor) -> BaseTool:
    bank: Dict[str, Any] = descriptor.config.setdefault("memory_bank", {})
    return MemoryReadTool(descriptor, bank)


def build_memory_write_tool(descriptor: ToolDescriptor) -> BaseTool:
    bank: Dict[str, Any] = descriptor.config.setdefault("memory_bank", {})
    return MemoryWriteTool(descriptor, bank)


__all__ = [
    "BaseTool",
    "BinaryToolLoader",
    "CandidatePromoter",
    "CriticalExperienceBuffer",
    "FunctionTool",
    "LatentRouter",
    "MemoryReadTool",
    "MemoryWriteTool",
    "ToolExecutionEngine",
    "ToolJudge",
    "ToolRegistrySystem",
    "build_memory_read_tool",
    "build_memory_write_tool",
]

