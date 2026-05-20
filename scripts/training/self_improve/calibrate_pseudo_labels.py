#!/usr/bin/env python3
"""Pseudo-label confidence-threshold calibration script.

Design doc §10.4 (open question 4): the per-edge confidence thresholds
in §5.2 are guesses. This script sweeps a candidate range of
thresholds against held-out consumer probe scores, finding the
threshold that best balances label volume (more labels ⇒ better
consumer training) against label quality (lower-confidence labels
contribute noise that can degrade the probe metric).

Workflow per edge:

1. Build the producer with the configured weights (vault snapshot
   loaded if provided).
2. Draw a fixed evaluation set of consumer-side raw inputs.
3. For each candidate threshold ``τ``:
   a. Run the producer's label function over every input.
   b. Record per-input confidence + label.
   c. Compute volume fraction = (#confidences ≥ τ) / total.
4. Score each threshold by ``score(τ) = volume(τ) · mean_kept_confidence(τ)``
   — a smooth proxy for "expected useful signal per batch". The
   optimum is whichever τ on the sweep grid maximizes this product.

Output: a JSON report at the path given by ``--out``, listing per-edge
sweeps + recommended thresholds.

Usage::

    python scripts/training/self_improve/calibrate_pseudo_labels.py \
        --vault-root runs/myrun/vault \
        --edge gaussian_encoder_producer:gaussian_encoder:gaussian_recon_label_fn \
        --out runs/myrun/calibration.json

Each ``--edge`` argument is ``producer:consumer:label_fn_dotted_name``.
The script imports ``label_fn_dotted_name`` from
``scripts.training.self_improve.edges`` by default; pass a full
``module.func`` to use a custom one.

This is a *calibration* tool, not a training step — it does not mutate
the producer's vault state, and the report is informational only. The
recommended thresholds should be promoted into the broker's
``EdgeConfig.confidence_threshold`` by the operator after review.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class EdgeSpec:
    """One ``producer:consumer:label_fn`` triple to calibrate."""

    producer: str
    consumer: str
    label_fn: str  # dotted import path

    @classmethod
    def parse(cls, raw: str) -> "EdgeSpec":
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"--edge expects 'producer:consumer:label_fn', got {raw!r}"
            )
        return cls(producer=parts[0], consumer=parts[1], label_fn=parts[2])


@dataclass
class ThresholdRow:
    threshold: float
    volume: float                # fraction of samples surviving
    mean_kept_conf: float        # mean confidence of survivors
    score: float                 # volume · mean_kept_conf


@dataclass
class EdgeReport:
    producer: str
    consumer: str
    label_fn: str
    n_samples: int
    sweep: list[ThresholdRow] = field(default_factory=list)
    recommended_threshold: float = 0.0
    recommended_score: float = 0.0


# ---------------------------------------------------------------------------
# Label-fn resolution
# ---------------------------------------------------------------------------


def _resolve_label_fn(dotted: str) -> Callable[[Any, Any], tuple[Any, float]]:
    """Resolve a dotted ``module.func`` path to a callable.

    If the dotted path contains no dot, treats it as a name under
    :mod:`scripts.training.self_improve.edges` — the canonical home for
    the edges shipped with the project.
    """
    if "." not in dotted:
        module_name = "scripts.training.self_improve.edges"
        func_name = dotted
    else:
        module_name, func_name = dotted.rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"could not import label-fn module {module_name!r}: {exc}"
        ) from exc
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise SystemExit(
            f"label-fn {func_name!r} not found in {module_name!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Plugin construction
# ---------------------------------------------------------------------------


def _instantiate_producer(producer_kind: str) -> Any:
    """Construct a producer plugin from its kind.

    Currently delegates to ``scripts.training.self_improve.plugins.get_plugin``
    using a kind heuristic: if the producer name matches a known plugin
    name, instantiate it directly; otherwise prefix-match (e.g.
    ``gaussian_encoder_producer`` → ``gaussian_encoder``).
    """
    from scripts.training.self_improve.plugins import available_plugins, get_plugin

    names = set(available_plugins())
    if producer_kind in names:
        return get_plugin(producer_kind)()
    for n in names:
        if producer_kind.startswith(n):
            return get_plugin(n)()
    raise SystemExit(
        f"could not determine plugin kind from producer name "
        f"{producer_kind!r}; known plugins: {sorted(names)}"
    )


# ---------------------------------------------------------------------------
# Probe set
# ---------------------------------------------------------------------------


def _build_probe_inputs(
    consumer_kind: str, n_samples: int, seed: int,
) -> list[torch.Tensor]:
    """Build a deterministic held-out set of consumer-side raw inputs.

    Reuses the project's existing probe builders so the calibration
    sweep sees the same distribution as the §6.1 eval probes. For an
    image-shaped consumer we use the eval-registry's probe builder for
    the matching plugin and slice each batch into per-sample tensors.

    Asks the builder for **ceil(n_samples / batch_size)** batches —
    floor division would silently under-allocate (e.g. ``n_samples=5``
    with ``batch_size=4`` would produce one batch of 4 and silently
    evaluate fewer samples than requested, changing the threshold
    score landscape without raising). The caller still gets exactly
    ``n_samples`` via the slice below.
    """
    if n_samples <= 0:
        raise SystemExit(f"--n-samples must be positive; got {n_samples}")
    from scripts.training.self_improve.eval_registry import (
        build_gaussian_encoder_probe,
        build_lgq_probe,
    )

    builders = {
        "gaussian_encoder": build_gaussian_encoder_probe,
        "lgq": build_lgq_probe,
    }
    builder = builders.get(consumer_kind)
    if builder is None:
        raise SystemExit(
            f"calibration probe builder not registered for consumer kind "
            f"{consumer_kind!r}; supported: {sorted(builders)}"
        )
    batch_size = 4
    n_batches = (n_samples + batch_size - 1) // batch_size  # ceiling division
    probe = builder(seed=seed, n_batches=n_batches, batch_size=batch_size)
    samples: list[torch.Tensor] = []
    for batch in probe:
        for i in range(batch.shape[0]):
            samples.append(batch[i])
            if len(samples) >= n_samples:
                return samples
    return samples


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def calibrate_edge(
    edge: EdgeSpec,
    *,
    n_samples: int = 64,
    thresholds: list[float] | None = None,
    seed: int = 0,
) -> EdgeReport:
    """Run a single edge through the threshold sweep.

    Builds the producer + a held-out consumer probe set, runs the
    label fn over each sample, then sweeps thresholds and computes
    ``volume · mean_kept_conf`` for each. Returns a populated
    :class:`EdgeReport`.

    The producer plugin is loaded fresh (no vault snapshot here);
    callers wanting calibration against a specific snapshot should
    extend the script — for the §10.4 calibration this baseline is
    what we want (the producer's stock weights set the scale).
    """
    if thresholds is None:
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    label_fn = _resolve_label_fn(edge.label_fn)
    producer = _instantiate_producer(edge.producer)
    samples = _build_probe_inputs(edge.consumer, n_samples=n_samples, seed=seed)

    confs: list[float] = []
    for s in samples:
        _, conf = label_fn(s, producer)
        confs.append(float(conf))
    confs_t = torch.tensor(confs, dtype=torch.float32)

    rows: list[ThresholdRow] = []
    best_row: ThresholdRow | None = None
    for tau in thresholds:
        mask = confs_t >= tau
        kept = int(mask.sum().item())
        volume = kept / max(len(samples), 1)
        mean_kept_conf = float(confs_t[mask].mean().item()) if kept > 0 else 0.0
        score = volume * mean_kept_conf
        row = ThresholdRow(
            threshold=float(tau),
            volume=float(volume),
            mean_kept_conf=mean_kept_conf,
            score=float(score),
        )
        rows.append(row)
        if best_row is None or row.score > best_row.score:
            best_row = row

    assert best_row is not None  # at least one threshold was scored
    return EdgeReport(
        producer=edge.producer,
        consumer=edge.consumer,
        label_fn=edge.label_fn,
        n_samples=len(samples),
        sweep=rows,
        recommended_threshold=best_row.threshold,
        recommended_score=best_row.score,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="calibrate-pseudo-labels",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--edge",
        action="append",
        required=True,
        help="A pseudo-label edge to calibrate, formatted as "
        "``producer:consumer:label_fn_dotted_name``. Repeat for "
        "multiple edges.",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=64,
        help="Held-out probe-set size for each edge.",
    )
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Candidate thresholds to sweep (default: 11 points 0..0.95).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="JSON path to write the calibration report to.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Initial torch seed for reproducible calibration runs.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    torch.manual_seed(int(args.seed))

    reports: list[EdgeReport] = []
    for raw in args.edge:
        spec = EdgeSpec.parse(raw)
        logger.info(
            "calibrating %s -> %s via %s",
            spec.producer, spec.consumer, spec.label_fn,
        )
        report = calibrate_edge(
            spec,
            n_samples=args.n_samples,
            thresholds=args.thresholds,
            seed=args.seed,
        )
        logger.info(
            "  recommended threshold=%.3f (score=%.4f) over %d samples",
            report.recommended_threshold,
            report.recommended_score,
            report.n_samples,
        )
        reports.append(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps([asdict(r) for r in reports], indent=2),
        encoding="utf-8",
    )
    logger.info("wrote %d edge report(s) to %s", len(reports), args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
