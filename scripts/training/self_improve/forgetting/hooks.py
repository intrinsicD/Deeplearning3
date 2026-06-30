"""Hook-based capacity expansion (parameter isolation).

Design (``docs/self_improvement.md`` §4.4): when the scheduler decides a
component is stagnating on a new skill, we don't fine-tune the
backbone. Instead we register a fresh :class:`LatentNeuralHook` with a
near-zero gate, freeze the backbone, and train only the hook. The
sigmoid gate lets the model recover the original behaviour exactly
(gate ≈ 0) and any improvement is purely additive — equivalent to
LoRA / Progressive Networks (Rusu et al. 2016) using the repo's
existing hook mechanism.

This module ships two pieces:

- :class:`PlateauDetector` — keeps a sliding window of eval scores and
  flags plateaus.
- :func:`expand_omnilatent_capacity` — the trigger action: freezes the
  OmniLatent backbone, registers a new hook, leaves the new hook's
  params trainable.

Other plugins (HPWM, MMWM) can grow their own expansion functions later;
OmniLatent is the design-doc reference because its ``HookManager`` is
already wired into the forward pass.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from omnilatent.model.hooks import LatentNeuralHook

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PlateauDetector
# ---------------------------------------------------------------------------


@dataclass
class PlateauDetector:
    """Tracks eval scores and flags plateaus.

    A plateau is declared when the *best* score in the most recent
    ``patience`` evaluations failed to improve by at least ``tol`` over
    the best in the previous ``patience`` evaluations. This is the
    standard "no improvement for N rounds" pattern.

    The convention follows the eval report: with ``higher_is_better=True``
    "improvement" means a larger score; with ``False`` (e.g. MSE) it
    means a smaller one.
    """

    patience: int = 5
    tol: float = 1e-3
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError(f"patience must be positive; got {self.patience}")
        # Two rolling windows: the "recent" half and the "previous" half.
        self._scores: deque[float] = deque(maxlen=self.patience * 2)
        self._triggered = False

    def record(self, score: float) -> bool:
        """Record an eval score; return ``True`` if a plateau is detected.

        The detector latches on first trigger — subsequent records keep
        returning ``True`` until :meth:`reset` is called. This matches
        the operational pattern: we don't want the orchestrator to
        register multiple hooks in quick succession just because the
        eval stays flat.
        """
        self._scores.append(float(score))
        if len(self._scores) < self.patience * 2:
            return False
        if self._triggered:
            return True
        scores = list(self._scores)
        prev = scores[: self.patience]
        recent = scores[self.patience :]
        if self.higher_is_better:
            prev_best = max(prev)
            recent_best = max(recent)
            improved = recent_best > prev_best + self.tol
        else:
            prev_best = min(prev)
            recent_best = min(recent)
            improved = recent_best < prev_best - self.tol
        self._triggered = not improved
        return self._triggered

    def reset(self) -> None:
        """Clear the score history and the latched-trigger flag.

        Call this after expansion has occurred so the next plateau can
        be detected independently.
        """
        self._scores.clear()
        self._triggered = False

    @property
    def triggered(self) -> bool:
        return self._triggered


# ---------------------------------------------------------------------------
# OmniLatent capacity expansion
# ---------------------------------------------------------------------------


def expand_omnilatent_capacity(
    omnilatent_plugin,  # OmniLatentPlugin (avoid import cycle)
    *,
    hook_name: str | None = None,
    num_tokens: int = 4,
    gate_bias_init: float = -4.0,
    use_transform: bool = True,
    rebuild_optimizer: bool = True,
    registry=None,
) -> "LatentNeuralHook":
    """Freeze the backbone and register a fresh hook on OmniLatent.

    Returns the newly-registered hook so callers can configure its
    optimizer (typically a separate small Adam over hook params).

    Backbone weights are frozen *in-place* via ``requires_grad_(False)``;
    the hook's parameters are trainable from the start. After this call:

    - ``sum(p.requires_grad for p in plugin.model.parameters())``
      counts only the new hook's parameters (plus any previously
      unfrozen hooks).
    - ``plugin.model.hook_manager.hooks`` includes the new entry.
    - The gate starts at ``sigmoid(gate_bias_init)`` ≈ 0.018 by
      default, so the immediate forward pass is bit-equivalent to the
      pre-expansion forward pass within floating-point tolerance.
    - With ``rebuild_optimizer=True`` (the default) the plugin's
      :attr:`_trainer.optimizer` is replaced by a fresh ``AdamW`` over
      the new hook's parameters. The original optimizer holds
      references to the *pre-expansion* parameter list — none of those
      params are trainable anymore — and contains no reference to the
      new hook's params. Without this rebuild, subsequent
      ``plugin.train_step`` calls update neither backbone nor hook
      and the expansion is silently a no-op.
    """
    import torch

    from omnilatent.model.hooks import LatentNeuralHook

    model = omnilatent_plugin.model
    if not hasattr(model, "hook_manager"):
        raise TypeError(
            "expand_omnilatent_capacity expects an OmniLatentPlugin; "
            f"got a plugin whose model has no hook_manager: {type(model).__name__}"
        )

    if hook_name is None:
        hook_name = f"expansion_{len(model.hook_manager.hooks)}"
    if hook_name in model.hook_manager.hooks:
        raise KeyError(f"hook {hook_name!r} already registered")

    # Freeze the backbone (everything that isn't already a hook).
    for p in model.parameters():
        p.requires_grad_(False)

    # Determine target layers from the backbone config.
    # OmniLatentConfig stores num_layers; we target every layer.
    cfg = omnilatent_plugin.config
    target_layers = list(range(cfg.num_layers))
    hook = LatentNeuralHook(
        name=hook_name,
        num_tokens=num_tokens,
        dim=cfg.hidden_dim,
        target_layers=target_layers,
        gate_bias_init=gate_bias_init,
        use_transform=use_transform,
    ).to(omnilatent_plugin.device)
    model.register_hook(hook)
    # Re-enable grad on the new hook (the freeze above touched everything
    # under model.parameters()).
    for p in hook.parameters():
        p.requires_grad_(True)

    if rebuild_optimizer:
        # Replace the trainer's pre-expansion optimizer with one that
        # references only the new hook's parameters. Keep the same
        # config (lr, weight_decay, betas) as the upstream trainer.
        omnilatent_plugin._trainer.optimizer = torch.optim.AdamW(
            hook.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )

    # Make the new capacity routable: register it into the expert registry so
    # the LearnedLatentRouter (work plan W2.2) can select it (W4.1). Without
    # this, an expansion hook can never be chosen for an input.
    if registry is not None:
        registry.sync_hooks(model.hook_manager)

    logger.info(
        "registered expansion hook %r on omnilatent (num_tokens=%d, "
        "gate_bias_init=%.2f, target_layers=%s)",
        hook_name,
        num_tokens,
        gate_bias_init,
        target_layers,
    )
    return hook


__all__ = ["PlateauDetector", "expand_omnilatent_capacity"]
