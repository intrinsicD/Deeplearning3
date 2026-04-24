"""Binary / lazy-loaded neural tool ecosystem."""

from __future__ import annotations

import abc
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .containers import (
    LatentPacket,
    LatentState,
    MemoryState,
    ToolContext,
    ToolDescriptor,
    ToolResult,
)
from .helpers import MLP
from .interfaces import IDecoder
from .model import ModularLatentWorldModel


class BaseTool(nn.Module, abc.ABC):
    descriptor: ToolDescriptor

    def __init__(self, descriptor: ToolDescriptor) -> None:
        super().__init__()
        self.descriptor = descriptor

    @abc.abstractmethod
    def run(self, packet: LatentPacket, context: ToolContext) -> ToolResult:
        raise NotImplementedError


class BinaryToolLoader:
    """Loads neural tools only when needed via lazy imports."""

    def __init__(self, device: torch.device, keep_hot: int = 2) -> None:
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
        mod = __import__(module_name, fromlist=[func_name])
        builder = getattr(mod, func_name)
        tool = builder(descriptor)
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Builder for {descriptor.tool_id} did not return BaseTool")
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
        if descriptor.unload_after_call:
            self.unload(descriptor.tool_id)
        return result


class ToolRegistrySystem:
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


class LatentRouter(nn.Module):
    def __init__(self, latent_dim: int, num_actions: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.policy = MLP([latent_dim, hidden_dim, hidden_dim, num_actions])
        self.stop_head = MLP([latent_dim, hidden_dim, 1])

    def forward(self, packet: LatentPacket) -> Dict[str, torch.Tensor]:
        x = packet.state.primary()
        return {
            "action_logits": self.policy(x),
            "stop_logit": self.stop_head(x),
        }


# ============================================================
# Concrete tools
# ============================================================


class InternalTransitionTool(BaseTool):
    def __init__(self, descriptor: ToolDescriptor, transition_model: ModularLatentWorldModel) -> None:
        super().__init__(descriptor)
        self.transition_model = transition_model

    def run(self, packet: LatentPacket, context: ToolContext) -> ToolResult:
        if context.action is None:
            raise ValueError("InternalTransitionTool requires context.action")
        memory_state = context.memory_state
        if memory_state is None:
            memory_state = self.transition_model.memory.init_state(packet.state.z_sem.shape[0], packet.state.z_sem.device)
        transition = self.transition_model.transition(packet.state, context.action, memory_state)
        next_packet = LatentPacket(
            state=transition.next_latent,
            confidence=transition.uncertainty,
            source_tool=self.descriptor.tool_id,
            trace=packet.trace + [self.descriptor.tool_id],
            metadata={**packet.metadata, **transition.aux},
        )
        return ToolResult(packet=next_packet, confidence=transition.uncertainty)


class TextDecoderTool(BaseTool):
    def __init__(self, descriptor: ToolDescriptor, decoder: IDecoder) -> None:
        super().__init__(descriptor)
        self.decoder = decoder

    def run(self, packet: LatentPacket, context: ToolContext) -> ToolResult:
        outputs = self.decoder(packet.state, context.raw_inputs)
        confidence = None
        if "text_logits" in outputs:
            probs = outputs["text_logits"].softmax(dim=-1)
            confidence = probs.max(dim=-1).values.mean()
        return ToolResult(packet=packet, raw_output=outputs, confidence=confidence)


class MemoryReadTool(BaseTool):
    def __init__(self, descriptor: ToolDescriptor, memory_bank: Dict[str, torch.Tensor]) -> None:
        super().__init__(descriptor)
        self.memory_bank = memory_bank

    def run(self, packet: LatentPacket, context: ToolContext) -> ToolResult:
        key = context.raw_inputs.get("memory_key", "default")
        value = self.memory_bank.get(key)
        if value is None:
            value = torch.zeros_like(packet.state.z_sem)
        next_state = LatentState(
            z_sem=packet.state.z_sem,
            z_dyn=packet.state.z_dyn,
            z_ctrl=packet.state.z_ctrl,
            z_ctx=value,
            extras=dict(packet.state.extras),
        )
        next_packet = LatentPacket(
            state=next_state,
            source_tool=self.descriptor.tool_id,
            trace=packet.trace + [self.descriptor.tool_id],
            metadata=dict(packet.metadata),
        )
        return ToolResult(packet=next_packet)


class MemoryWriteTool(BaseTool):
    def __init__(self, descriptor: ToolDescriptor, memory_bank: Dict[str, torch.Tensor]) -> None:
        super().__init__(descriptor)
        self.memory_bank = memory_bank

    def run(self, packet: LatentPacket, context: ToolContext) -> ToolResult:
        key = context.raw_inputs.get("memory_key", "default")
        self.memory_bank[key] = packet.state.z_ctx if packet.state.z_ctx is not None else packet.state.z_sem
        return ToolResult(packet=packet, side_effects={"memory_written": key})


class ToolExecutionEngine:
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
        trace: List[Dict[str, Any]] = []
        current = packet
        for step in range(max_steps):
            route = self.router(current)
            stop_prob = torch.sigmoid(route["stop_logit"]).squeeze(-1)  # [B]
            action_ids = route["action_logits"].argmax(dim=-1)  # [B]
            stop_all = bool((stop_prob > 0.5).all())
            unique_action_ids = torch.unique(action_ids)
            if unique_action_ids.numel() != 1:
                raise ValueError(
                    "ToolExecutionEngine received a batched route with different tool actions per sample. "
                    "Current iterate() executes one tool call per step; use batch size 1 or a per-sample "
                    "tool execution path."
                )
            action_id = int(unique_action_ids.item())
            trace.append({
                "step": step,
                "stop_prob_mean": float(stop_prob.mean().item()),
                "stop_prob": stop_prob.detach().cpu().tolist(),
                "action_ids": action_ids.detach().cpu().tolist(),
                "action_id": action_id,
            })
            if stop_all:
                break
            tool_name = self.action_id_to_tool[action_id]
            result = self.registry.call(tool_name, current, context)
            if result.packet is not None:
                current = result.packet
            trace[-1]["tool"] = tool_name
            trace[-1]["confidence"] = None if result.confidence is None else float(result.confidence.detach().mean().cpu().item())
        return current, trace


# ============================================================
# Entry-point builders for lazy loading
# ============================================================


def build_memory_read_tool(descriptor: ToolDescriptor) -> BaseTool:
    bank: Dict[str, torch.Tensor] = descriptor.config.setdefault("memory_bank", {})
    return MemoryReadTool(descriptor, bank)


def build_memory_write_tool(descriptor: ToolDescriptor) -> BaseTool:
    bank: Dict[str, torch.Tensor] = descriptor.config.setdefault("memory_bank", {})
    return MemoryWriteTool(descriptor, bank)


def build_internal_transition_tool(descriptor: ToolDescriptor) -> BaseTool:
    model = descriptor.config["transition_model"]
    return InternalTransitionTool(descriptor, model)


def build_text_decoder_tool(descriptor: ToolDescriptor) -> BaseTool:
    decoder = descriptor.config["decoder"]
    return TextDecoderTool(descriptor, decoder)
