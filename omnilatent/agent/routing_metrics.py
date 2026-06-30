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


__all__ = ["routing_accuracy", "expected_calibration_error"]
