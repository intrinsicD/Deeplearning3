"""Metrics for evaluating router selection quality (work plan W2.3 / W2.4).

These quantify *whether the router selects the right expert* and *whether its
confidence is trustworthy* — the non-negotiable measurement the capability
proposal (Audit.md P1) requires before routing can be believed:

  * :func:`routing_accuracy` — fraction of inputs routed to the correct expert.
  * :func:`expected_calibration_error` — does low confidence actually predict a
    wrong choice? (reliability of the abstention signal).
"""

from __future__ import annotations

from typing import Sequence

import torch


def routing_accuracy(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """Fraction of inputs whose routed expert matches the gold expert."""
    if len(predicted) != len(gold):
        raise ValueError("predicted and gold must have equal length")
    if not predicted:
        return 0.0
    correct = sum(1 for p, g in zip(predicted, gold) if p == g)
    return correct / len(predicted)


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error of the router's confidence.

    Bins predictions by confidence and averages the gap between mean confidence
    and empirical accuracy in each bin, weighted by bin population. 0 ⇒ perfect
    calibration (confidence == accuracy); large ⇒ confidence is untrustworthy
    and the abstention threshold cannot be set meaningfully.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have equal length")
    n = len(confidences)
    if n == 0:
        return 0.0
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        # Last bin is closed on the right so confidence == 1.0 is included.
        members = [
            i for i in range(n)
            if (lo < confidences[i] <= hi) or (b == 0 and confidences[i] <= hi)
        ]
        if not members:
            continue
        bin_conf = sum(confidences[i] for i in members) / len(members)
        bin_acc = sum(1 for i in members if correct[i]) / len(members)
        ece += (len(members) / n) * abs(bin_acc - bin_conf)
    return ece


def load_balancing_loss(logits: torch.Tensor) -> torch.Tensor:
    """Switch-Transformer load-balancing auxiliary loss (Fedus et al. 2021).

    Discourages the router from collapsing onto a few experts — the failure
    mode of end-to-end credit assignment (W3.3). For ``E`` experts over a batch
    of ``N`` inputs::

        loss = E * Σ_e  f_e · P_e

    where ``f_e`` is the fraction of inputs whose top-1 expert is ``e`` and
    ``P_e`` is the mean router probability of ``e``. Minimized (= 1.0) when load
    is uniform; larger when concentrated. Differentiable through ``P_e``.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be (N, E); got {tuple(logits.shape)}")
    n, e = logits.shape
    if e == 0:
        return logits.new_zeros(())
    probs = torch.softmax(logits, dim=-1)          # (N, E)
    mean_prob = probs.mean(dim=0)                   # (E,) — P_e (differentiable)
    top1 = probs.argmax(dim=-1)                     # (N,)
    frac = torch.bincount(top1, minlength=e).to(probs.dtype) / n  # f_e (constant)
    return e * torch.sum(frac * mean_prob)


__all__ = [
    "routing_accuracy",
    "expected_calibration_error",
    "load_balancing_loss",
]
