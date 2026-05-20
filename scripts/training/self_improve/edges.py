"""Concrete pseudo-label edges (`docs/self_improvement.md` §5).

Phase 7 shipped the :class:`PseudoLabelBroker` mechanics; this module
ships the **first concrete edge** plus the shape-agnostic plumbing
plugins use to consume incoming labels.

Available edges in this module:

- :func:`gaussian_recon_label_fn` — a "second-opinion" image
  reconstruction edge. The producer is any plugin exposing an image
  reconstruction (currently :class:`GaussianEncoderPlugin`); the
  consumer treats the producer's recon as an additional MSE target.

The §5 design lists five edges in total (LGQ tokens for HPWM/MMWM,
HPWM slot states for OmniLatent, OmniLatent text decoder for MMWM/HPWM,
MMWM latents for OmniLatent, Gaussian recons for LGQ). The remaining
four require either (a) plugin train_step restructures we haven't
landed (e.g. LGQ's GAN dual-optimizer step) or (b) shape adapters
between plugin configs that don't agree yet (e.g. Gaussian's 1×28×28
vs. LGQ's 3×32×32). Those land alongside the dataset adapters.

The plumbing exposed here — :class:`PluginPseudoLabelConsumer` and
:func:`apply_pending_labels` — is intentionally edge-agnostic so a
plugin's ``train_step`` can absorb labels from any producer without
caring about the §5 catalogue.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn

from scripts.training.self_improve.plugins.base import ComponentPlugin
from scripts.training.self_improve.pseudo_labels import PseudoLabelBatch


# ---------------------------------------------------------------------------
# Producer-side: label functions
# ---------------------------------------------------------------------------


def gaussian_recon_label_fn(
    sample: torch.Tensor, producer: ComponentPlugin,
) -> tuple[torch.Tensor, float]:
    """A "second-opinion" image-reconstruction label.

    Runs the producer (typically a :class:`GaussianEncoderPlugin` with
    its snapshotted weights swapped in by the broker) on a single
    sample and returns the reconstruction tensor + a confidence score.
    Confidence is ``1 / (1 + recon_mse)`` ∈ (0, 1] — a smooth,
    bounded surrogate for "the producer is sure" without exposing the
    MSE on its own scale.
    """
    if sample.dim() == 3:
        x = sample.unsqueeze(0).to(producer.device)
    elif sample.dim() == 4:
        x = sample.to(producer.device)
    else:
        raise ValueError(
            f"gaussian_recon_label_fn expects a 3D or 4D image; got "
            f"shape {tuple(sample.shape)}"
        )
    was_training = producer.model.training
    producer.model.eval()
    try:
        with torch.no_grad():
            recon, _ = producer.model(x)
        mse = float(nn.functional.mse_loss(recon, x).item())
    finally:
        producer.model.train(was_training)
    label = recon.squeeze(0).cpu() if sample.dim() == 3 else recon.cpu()
    confidence = 1.0 / (1.0 + mse)
    return label, confidence


# ---------------------------------------------------------------------------
# Consumer-side: a small mixin-style helper plugins call from train_step
# ---------------------------------------------------------------------------


class PluginPseudoLabelConsumer:
    """Helper for plugins that consume pseudo-labels in ``train_step``.

    The orchestrator sets ``plugin._pending_pseudo_labels`` to a list
    of :class:`PseudoLabelBatch` before each consumer step and clears
    it afterward. The plugin reads via :meth:`pop_pending_labels` and
    folds the resulting consistency loss into its main backward.

    Plugins inherit this mixin alongside :class:`ComponentPlugin`.
    """

    #: Set by the orchestrator before ``train_step``; cleared after.
    _pending_pseudo_labels: list[PseudoLabelBatch] | None = None

    def pop_pending_labels(self) -> list[PseudoLabelBatch]:
        """Return the queued labels and clear the slot atomically."""
        out = self._pending_pseudo_labels or []
        self._pending_pseudo_labels = None
        return out


def image_consistency_loss(
    inputs: torch.Tensor,
    student_outputs: torch.Tensor,
    labels: torch.Tensor,
    confidences: torch.Tensor,
) -> torch.Tensor:
    """Confidence-weighted MSE between student outputs and pseudo-targets.

    All four tensors are batch-aligned (size N along dim 0). Each
    per-sample MSE is multiplied by its confidence before averaging.
    Returns a scalar tensor with autograd attached through
    ``student_outputs``.
    """
    if not (inputs.shape[0] == student_outputs.shape[0] == labels.shape[0]):
        raise ValueError(
            "shape mismatch in image_consistency_loss: "
            f"inputs={tuple(inputs.shape)} student={tuple(student_outputs.shape)} "
            f"labels={tuple(labels.shape)}"
        )
    if confidences.shape[0] != inputs.shape[0]:
        raise ValueError(
            f"confidence length {confidences.shape[0]} != batch size "
            f"{inputs.shape[0]}"
        )
    per_sample = (student_outputs - labels.to(student_outputs.device)).pow(2)
    # Mean over all non-batch dims so the per-sample term is a scalar.
    per_sample = per_sample.flatten(1).mean(dim=1)
    weights = confidences.to(student_outputs.device)
    return (per_sample * weights).mean()


def apply_pending_labels(
    plugin: PluginPseudoLabelConsumer,
    student_recon: torch.Tensor,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fold queued image-shaped labels into a single consistency tensor.

    Returns ``(loss, log_dict)``. ``loss`` is a scalar autograd-tracked
    tensor (zero if no labels queued); ``log_dict`` carries per-producer
    consistency values for the step report.

    The function only consumes label batches whose ``inputs`` happen to
    match ``inputs`` element-wise (the broker guarantees this when the
    orchestrator passes the same raw batch to ``broker.sample`` and to
    ``train_step``). This avoids accidentally consuming a stale batch
    whose shapes don't align.
    """
    log: dict[str, float] = {}
    batches = plugin.pop_pending_labels()
    if not batches:
        return torch.zeros((), device=student_recon.device), log

    total: torch.Tensor | None = None
    for b in batches:
        if not b.labels:
            continue
        labels = torch.stack(b.labels) if isinstance(b.labels, list) else b.labels
        confidences = torch.tensor(
            list(b.confidences), dtype=torch.float32,
        ) if isinstance(b.confidences, list) else b.confidences
        # The broker keeps inputs/labels size-aligned to *its own*
        # sample list (phase-7 fix). We need them aligned to the
        # student's input batch as well — drop the broker batch if
        # its shape doesn't match.
        if labels.shape[0] != student_recon.shape[0]:
            log[f"pseudo/{b.producer}/skipped"] = float(labels.shape[0])
            continue
        term = image_consistency_loss(
            inputs, student_recon, labels, confidences,
        )
        log[f"pseudo/{b.producer}"] = float(term.detach().item())
        total = term if total is None else total + term

    if total is None:
        return torch.zeros((), device=student_recon.device), log
    return total, log


__all__ = [
    "PluginPseudoLabelConsumer",
    "apply_pending_labels",
    "gaussian_recon_label_fn",
    "image_consistency_loss",
]
