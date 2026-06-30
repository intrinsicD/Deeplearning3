"""Train OmniLatent hooks together with a learned router (work plan W6.1).

This is where input-conditioned selection (Phase 2) and conditional use
(Phase 3) stop being unit-test mechanisms and enter a real training loop on the
actual :class:`OmniLatentModel`. The backbone is frozen; the trainable pieces
are the registered hooks and — in ``routed`` mode — the
:class:`LearnedLatentRouter`.

Three modes share one code path so the routed-vs-baseline comparison (W6.2) is
apples-to-apples:

* ``"routed"``   — the router selects per-input which hooks fire (top-k,
  content-conditioned gates), trained with the task loss + a Switch
  load-balancing auxiliary.
* ``"always_on"`` — every hook fires for every input (the pre-routing default).
* ``"no_hooks"``  — all hooks skipped (frozen-backbone baseline).

The router receives gradient from the task loss *through the gate scaling*
(``hook_tokens * (static_gate * route_weight)``), not only from the auxiliary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

import torch.nn.functional as F

from omnilatent.agent.registry import ExpertRegistry
from omnilatent.agent.router import LearnedLatentRouter
from omnilatent.agent.routing_metrics import load_balancing_loss
from omnilatent.training.losses import MultiModalLoss

MODES = ("routed", "always_on", "no_hooks")


@torch.no_grad()
def _per_sample_recon_loss(model, modality: str, data: torch.Tensor) -> torch.Tensor:
    """Per-sample self-reconstruction MSE ``(B,)`` under the current routing."""
    out = model(modality, data, modality, data)["output"]
    return ((out - data) ** 2).flatten(1).mean(dim=1)


@torch.no_grad()
def counterfactual_hook_credit(
    model, modality: str, data: torch.Tensor
) -> tuple[torch.Tensor, list[str]]:
    """Per-sample marginal effect of each hook on reconstruction (work plan W5.2).

    For every registered hook, measure how much activating *only* that hook
    reduces the per-sample reconstruction loss versus the no-hook baseline.
    Positive credit ⇒ the hook helps that input. This is a real, task-grounded
    credit signal (credit v3): it measures what each hook actually does to the
    loss, rather than a synthetic gold label (v1) or a scalar outcome (v2).

    Returns ``(credit, hook_names)`` where ``credit`` is ``(B, n_hooks)`` in
    ``hook_names`` order (the manager's hook order). Costs ``n_hooks + 1``
    forward passes.
    """
    manager = model.hook_manager
    names = list(manager.hooks.keys())
    if not names:
        return data.new_zeros(data.shape[0], 0), names
    try:
        manager.set_route_weights({n: 0.0 for n in names})
        base = _per_sample_recon_loss(model, modality, data)
        credit = data.new_zeros(data.shape[0], len(names))
        for j, name in enumerate(names):
            weights = {n: 0.0 for n in names}
            weights[name] = 1.0
            manager.set_route_weights(weights)
            credit[:, j] = base - _per_sample_recon_loss(model, modality, data)
    finally:
        manager.set_route_weights(None)
    return credit, names


@dataclass
class RoutedTrainer:
    """Joint hook+router trainer for one modality's self-reconstruction.

    Args:
        model: an ``OmniLatentModel`` with hooks registered.
        config: its ``OmniLatentConfig``.
        modality: the modality to self-reconstruct (controlled, single-task).
        mode: one of :data:`MODES`.
        router: required for ``mode="routed"`` (its registry should be synced
            to the model's hooks).
        lr: learning rate for hooks (+ router).
        load_balance_weight: weight of the Switch auxiliary (routed mode).
        freeze_backbone: freeze everything but the hooks (fair comparison).
    """

    model: object
    config: object
    modality: str = "image"
    mode: str = "routed"
    router: LearnedLatentRouter | None = None
    lr: float = 1e-3
    load_balance_weight: float = 0.01
    counterfactual_weight: float = 1.0
    freeze_backbone: bool = True
    history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}; got {self.mode!r}")
        if self.mode == "routed" and self.router is None:
            raise ValueError("routed mode requires a router")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = MultiModalLoss(self.config).to(self.device)
        self._last_injected = 0
        self.injected_per_batch = 0.0

        hook_params: list = []
        for hook in self.model.hook_manager.hooks.values():
            for p in hook.parameters():
                p.requires_grad_(True)
                hook_params.append(p)
        hook_ids = {id(p) for p in hook_params}

        params = list(hook_params)
        # Backbone (everything that isn't a hook): freeze, or include in the
        # optimizer when freeze_backbone=False.
        for p in self.model.parameters():
            if id(p) in hook_ids:
                continue
            p.requires_grad_(not self.freeze_backbone)
            if not self.freeze_backbone:
                params.append(p)

        if self.mode == "routed":
            self.router.to(self.device)
            params += list(self.router.parameters())
        # No trainable params in no_hooks (frozen backbone) — use a dummy so the
        # optimizer is valid; it simply never updates anything.
        self._has_params = len(params) > 0
        self.optimizer = torch.optim.Adam(params, lr=self.lr) if self._has_params else None

    # -- routing application --------------------------------------------
    def _apply_routing(self, data: torch.Tensor) -> torch.Tensor | None:
        """Set hook route weights for this batch; return router logits (routed)."""
        manager = self.model.hook_manager
        if self.mode == "always_on":
            manager.set_route_weights(None)
            return None
        if self.mode == "no_hooks":
            manager.set_route_weights({n: 0.0 for n in manager.hooks.keys()})
            return None
        # routed
        enc = self.model.encode(self.modality, data)
        summary = enc[:, 1:].mean(dim=1)
        out = self.router.forward(summary)
        ids = self.router.registry.ids()
        rw = {}
        for i, eid in enumerate(ids):
            if self.router.registry.kind(eid) == "hook" and eid.startswith("hook:"):
                name = eid[len("hook:"):]
                if name in manager.hooks:
                    rw[name] = out["weights"][:, i]
        manager.set_route_weights(rw)
        # Compute honesty: a hook is *injected for the whole batch* if any sample
        # selects it. At batch>1 this can approach the full pool even though each
        # input logically uses only top_k — so the per-input count overstates
        # the batched compute saving.
        self._last_injected = sum(1 for w in rw.values() if bool((w != 0).any()))
        return out["logits"]

    def step(self, batch: dict) -> float:
        data = batch[self.modality].to(self.device)
        manager = self.model.hook_manager
        logits = self._apply_routing(data)
        try:
            result = self.model(self.modality, data, self.modality, data)
            loss = self.criterion({self.modality: result["output"]}, {self.modality: data})["total"]
            if self.mode == "routed" and self.load_balance_weight > 0 and logits is not None:
                loss = loss + self.load_balance_weight * load_balancing_loss(logits)
            if self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
        finally:
            manager.set_route_weights(None)
        value = float(loss.detach().item())
        self.history.append(value)
        return value

    def step_counterfactual(self, batch: dict) -> float:
        """One routed step trained with counterfactual credit (W5.2, credit v3).

        The router is supervised toward the *counterfactually-best* hook for each
        input (the hook whose ablation most reduces that input's loss), while the
        selected hooks are trained on the reconstruction task as usual.
        """
        if self.mode != "routed":
            raise ValueError("step_counterfactual requires mode='routed'")
        data = batch[self.modality].to(self.device)
        manager = self.model.hook_manager

        # Task-grounded credit (no grad) → per-sample best hook as router target.
        credit, names = counterfactual_hook_credit(self.model, self.modality, data)
        best_local = credit.argmax(dim=1)
        ids = self.router.registry.ids()
        col = {eid: i for i, eid in enumerate(ids)}
        target = torch.tensor(
            [col["hook:" + names[int(b)]] for b in best_local],
            dtype=torch.long, device=self.device,
        )

        logits = self._apply_routing(data)  # sets route weights, returns (B,E) logits
        try:
            result = self.model(self.modality, data, self.modality, data)
            recon = self.criterion(
                {self.modality: result["output"]}, {self.modality: data}
            )["total"]
            router_loss = F.cross_entropy(logits, target)
            loss = recon + self.counterfactual_weight * router_loss
            if self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
        finally:
            manager.set_route_weights(None)
        value = float(loss.detach().item())
        self.history.append(value)
        return value

    def train(self, dataloader, steps: int) -> list[float]:
        self.model.train()
        it = iter(dataloader)
        for _ in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dataloader)
                batch = next(it)
            self.step(batch)
        return self.history

    @torch.no_grad()
    def evaluate(self, dataloader, batches: int = 8) -> float:
        """Mean self-reconstruction loss over a few batches (no aux).

        Side effect: records ``self.injected_per_batch`` — the mean number of
        hooks actually injected per batch in routed mode (an honest compute
        proxy, vs the logical per-input ``top_k``).
        """
        self.model.eval()
        manager = self.model.hook_manager
        total, n = 0.0, 0
        injected_total = 0
        for i, batch in enumerate(dataloader):
            if i >= batches:
                break
            data = batch[self.modality].to(self.device)
            self._last_injected = self.model.hook_manager.hooks.__len__() if self.mode == "always_on" else 0
            self._apply_routing(data)
            try:
                result = self.model(self.modality, data, self.modality, data)
                loss = self.criterion({self.modality: result["output"]}, {self.modality: data})["total"]
            finally:
                manager.set_route_weights(None)
            total += float(loss.item())
            injected_total += self._last_injected
            n += 1
        self.model.train()
        self.injected_per_batch = injected_total / max(n, 1)
        return total / max(n, 1)


__all__ = ["RoutedTrainer", "MODES", "counterfactual_hook_credit"]
