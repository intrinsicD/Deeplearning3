"""Unified objective registry for OmniLatent training.

This module is additive: it does not replace existing loss classes. It provides
small objective adapters with explicit required fields, clean skip semantics for
incomplete/invalid batches, and optional learned uncertainty weighting over
active objectives only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from omnilatent.training.losses import ContrastiveLoss, NextClipPredictionLoss, ReconstructionLoss

Batch = Mapping[str, Any]
ObjectiveFn = Callable[[Batch], torch.Tensor]


@dataclass(frozen=True)
class ObjectiveResult:
    """Result of evaluating one objective on one batch."""

    name: str
    loss: Optional[torch.Tensor] = None
    active: bool = False
    skipped: bool = False
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectiveSpec:
    """Registered objective with required input fields and compute function."""

    name: str
    required_fields: tuple[str, ...]
    compute: ObjectiveFn
    description: str = ""

    def evaluate(self, batch: Batch) -> ObjectiveResult:
        missing = [field for field in self.required_fields if field not in batch or batch[field] is None]
        if missing:
            return ObjectiveResult(
                name=self.name,
                active=False,
                skipped=True,
                reason=f"missing required fields: {', '.join(missing)}",
            )
        try:
            loss = self.compute(batch)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return ObjectiveResult(name=self.name, active=False, skipped=True, reason=str(exc))
        if not isinstance(loss, torch.Tensor):
            return ObjectiveResult(
                name=self.name,
                active=False,
                skipped=True,
                reason=f"objective returned {type(loss).__name__}, expected Tensor",
            )
        if loss.numel() != 1:
            loss = loss.mean()
        if not torch.isfinite(loss):
            return ObjectiveResult(name=self.name, active=False, skipped=True, reason="loss is not finite")
        return ObjectiveResult(name=self.name, loss=loss, active=True, skipped=False)


class ObjectiveRegistry:
    """Name → ObjectiveSpec registry."""

    def __init__(self) -> None:
        self._items: Dict[str, ObjectiveSpec] = {}

    def register(self, spec: ObjectiveSpec) -> ObjectiveSpec:
        if spec.name in self._items:
            raise ValueError(f"Duplicate objective registration: {spec.name}")
        self._items[spec.name] = spec
        return spec

    def get(self, name: str) -> ObjectiveSpec:
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)

    def specs(self) -> Dict[str, ObjectiveSpec]:
        return dict(self._items)


class UnifiedObjectiveLoss(nn.Module):
    """Evaluate and combine registered objectives.

    Args:
        registry: objective registry. Defaults to :func:`default_objective_registry`.
        objectives: optional subset/order. Defaults to all registry objectives.
        weights: scalar multiplier per objective.
        learned_uncertainty: if True, active objective losses are weighted as
            ``exp(-log_var) * weighted_loss + log_var``. Skipped objectives do
            not contribute and do not update their log_vars.
    """

    def __init__(
        self,
        registry: ObjectiveRegistry | None = None,
        objectives: Iterable[str] | None = None,
        weights: Mapping[str, float] | None = None,
        learned_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        self.registry = default_objective_registry() if registry is None else registry
        self.objectives = list(self.registry.names() if objectives is None else objectives)
        self.weights = dict(weights or {})
        self.learned_uncertainty = learned_uncertainty
        if learned_uncertainty:
            self.log_vars = nn.ParameterDict({name: nn.Parameter(torch.zeros(())) for name in self.objectives})
        else:
            self.log_vars = nn.ParameterDict()

    def forward(self, batch: Batch) -> Dict[str, Any]:
        results: Dict[str, ObjectiveResult] = {}
        losses: Dict[str, torch.Tensor] = {}
        skipped: Dict[str, str] = {}
        total: Optional[torch.Tensor] = None

        for name in self.objectives:
            spec = self.registry.get(name)
            result = spec.evaluate(batch)
            results[name] = result
            if result.skipped or result.loss is None:
                skipped[name] = result.reason or "skipped"
                continue
            raw_loss = result.loss
            losses[name] = raw_loss
            weighted_loss = raw_loss * float(self.weights.get(name, 1.0))
            if self.learned_uncertainty:
                log_var = self.log_vars[name]
                weighted_loss = torch.exp(-log_var) * weighted_loss + log_var
                losses[f"{name}_log_var"] = log_var
                losses[f"{name}_weighted"] = weighted_loss
            total = weighted_loss if total is None else total + weighted_loss

        if total is None:
            total = _zero_like_batch(batch)
        total = cast(torch.Tensor, total)
        losses["total"] = total
        return {
            "total": total,
            "losses": losses,
            "results": results,
            "skipped": skipped,
            "active": [name for name, result in results.items() if result.active],
        }


def default_objective_registry() -> ObjectiveRegistry:
    registry = ObjectiveRegistry()
    recon = ReconstructionLoss()
    contrastive = ContrastiveLoss()
    next_clip = NextClipPredictionLoss()

    registry.register(ObjectiveSpec(
        name="self_reconstruction",
        required_fields=("predictions", "targets"),
        compute=lambda batch: _reconstruction_objective(batch, recon),
        description="same-modality reconstruction over prediction/target dictionaries",
    ))
    registry.register(ObjectiveSpec(
        name="cross_modal_translation",
        required_fields=("translation_prediction", "translation_target", "target_modality"),
        compute=lambda batch: recon(str(batch["target_modality"]), batch["translation_prediction"], batch["translation_target"]),
        description="decode one modality from another",
    ))
    registry.register(ObjectiveSpec(
        name="contrastive_alignment",
        required_fields=("latents",),
        compute=lambda batch: _contrastive_objective(batch, contrastive),
        description="pairwise InfoNCE over two or more modality latent batches",
    ))
    registry.register(ObjectiveSpec(
        name="latent_bottleneck",
        required_fields=("reasoning_bottleneck", "source_summary"),
        compute=lambda batch: F.mse_loss(batch["reasoning_bottleneck"], batch["source_summary"]),
        description="latent thought bottleneck consistency",
    ))
    registry.register(ObjectiveSpec(
        name="next_latent_prediction",
        required_fields=("predicted_next_latent", "target_next_latent"),
        compute=lambda batch: next_clip(batch["predicted_next_latent"], batch["target_next_latent"], batch.get("next_latent_mask")),
        description="predict next latent token/state",
    ))
    registry.register(ObjectiveSpec(
        name="router_supervision",
        required_fields=("router_logits", "router_targets"),
        compute=_router_supervision_objective,
        description="supervised route/action and optional stop prediction",
    ))
    registry.register(ObjectiveSpec(
        name="kb_retrieval_distillation",
        required_fields=("retrieval_logits", "teacher_retrieval_probs"),
        compute=_kb_retrieval_distillation_objective,
        description="distill retrieval distribution into student logits",
    ))
    return registry


def _reconstruction_objective(batch: Batch, recon: ReconstructionLoss) -> torch.Tensor:
    predictions = batch["predictions"]
    targets = batch["targets"]
    if not isinstance(predictions, Mapping) or not isinstance(targets, Mapping):
        raise TypeError("predictions and targets must be mappings")
    common = [name for name in predictions if name in targets]
    if not common:
        raise ValueError("no overlapping prediction/target modalities")
    losses = [recon(str(name), predictions[name], targets[name]) for name in common]
    return torch.stack([loss.reshape(()) for loss in losses]).mean()


def _contrastive_objective(batch: Batch, contrastive: ContrastiveLoss) -> torch.Tensor:
    latents = batch["latents"]
    if not isinstance(latents, Mapping):
        raise TypeError("latents must be a mapping")
    names = list(latents.keys())
    if len(names) < 2:
        raise ValueError("contrastive_alignment requires at least two latent modalities")
    losses = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            losses.append(contrastive(latents[names[i]], latents[names[j]]))
    return torch.stack([loss.reshape(()) for loss in losses]).mean()


def _router_supervision_objective(batch: Batch) -> torch.Tensor:
    action_loss = F.cross_entropy(batch["router_logits"], batch["router_targets"].long())
    if "router_stop_logits" not in batch or "router_stop_targets" not in batch:
        return action_loss
    stop_logits = batch["router_stop_logits"].squeeze(-1)
    stop_targets = batch["router_stop_targets"].float()
    return action_loss + F.binary_cross_entropy_with_logits(stop_logits, stop_targets)


def _kb_retrieval_distillation_objective(batch: Batch) -> torch.Tensor:
    logits = batch["retrieval_logits"]
    teacher = batch["teacher_retrieval_probs"]
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return F.kl_div(F.log_softmax(logits, dim=-1), teacher, reduction="batchmean")


def _zero_like_batch(batch: Batch) -> torch.Tensor:
    for value in batch.values():
        found = _find_tensor(value)
        if found is not None:
            return torch.zeros((), device=found.device, dtype=found.dtype)
    return torch.zeros(())


def _find_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            found = _find_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_tensor(item)
            if found is not None:
                return found
    return None


__all__ = [
    "ObjectiveRegistry",
    "ObjectiveResult",
    "ObjectiveSpec",
    "UnifiedObjectiveLoss",
    "default_objective_registry",
]


