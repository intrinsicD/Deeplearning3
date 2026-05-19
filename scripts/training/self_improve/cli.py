"""Command-line entry point for the self-improvement orchestrator.

Usage:

    # Single-component dry-run (1 eval, exit)
    python -m scripts.training.self_improve.cli --components gaussian_encoder --dry-run

    # 200 steps on HPWM
    python -m scripts.training.self_improve.cli --components hpwm --max-steps 200

    # All five (round-robin) for 500 steps
    python -m scripts.training.self_improve.cli --components all --max-steps 500

Phase 5 only supports synthetic batches via
:class:`SyntheticDataStream`; real video / image inputs land in phase 7
alongside the dataset adapters.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from scripts.training.self_improve.data_stream import SyntheticDataStream
from scripts.training.self_improve.eval_registry import EvalRegistry
from scripts.training.self_improve.orchestrator import Orchestrator
from scripts.training.self_improve.plugins import available_plugins, get_plugin
from scripts.training.self_improve.scheduler import RoundRobinScheduler
from scripts.training.self_improve.vault import Vault

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="omnilatent-self-improve",
        description=(
            "Run the continual self-supervised learning orchestrator. "
            "Phase 5 supports synthetic batches only."
        ),
    )
    p.add_argument(
        "--components",
        nargs="+",
        default=["gaussian_encoder"],
        help=(
            "Component names to train. Pass 'all' to load every plugin. "
            f"Available: {', '.join(available_plugins())}."
        ),
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Total training steps (across all components if multiple).",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="Per-component eval cadence. Set to 0 to disable.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Synthetic batch size fed to each plugin.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Initial torch seed for reproducibility.",
    )
    p.add_argument(
        "--vault-root",
        type=Path,
        default=None,
        help="Directory for the content-addressed snapshot vault. "
        "Omit to disable snapshot+rollback.",
    )
    p.add_argument(
        "--rollback-tol",
        type=float,
        default=0.0,
        help="Per-metric tolerance below which an eval triggers a rollback.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the orchestrator + run a single eval per component, "
        "no training. Used in CI to validate the config + plugin "
        "construction path.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def _resolve_components(args: argparse.Namespace) -> list[str]:
    if args.components == ["all"]:
        return list(available_plugins())
    unknown = [c for c in args.components if c not in available_plugins()]
    if unknown:
        raise SystemExit(
            f"unknown component(s) {unknown}; available: {available_plugins()}"
        )
    return list(args.components)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    components = _resolve_components(args)
    logger.info("loading plugins: %s", components)

    # Seed before plugin construction so the entire initialization is
    # reproducible. The orchestrator does not re-seed (see its
    # docstring) to keep direct-training and orchestrator trajectories
    # bit-identical under the same seed.
    import torch

    torch.manual_seed(int(args.seed))

    plugins = {}
    for name in components:
        cls = get_plugin(name)
        plugins[name] = cls()

    eval_registry = EvalRegistry.with_defaults(components)
    vault = Vault(args.vault_root) if args.vault_root is not None else None

    orchestrator = Orchestrator(
        plugins,
        scheduler=RoundRobinScheduler(components),
        data_stream=SyntheticDataStream(batch_size=args.batch_size),
        eval_registry=eval_registry,
        vault=vault,
        eval_every=args.eval_every,
        rollback_tol=args.rollback_tol,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    if args.dry_run:
        logger.info("dry-run: evaluating each component once and exiting")
        for name, plugin in plugins.items():
            probe = eval_registry.get(name) if eval_registry.has(name) else None
            report = plugin.evaluate(probe)
            logger.info(
                "dry-run eval: %s score=%.4f metrics=%s",
                name,
                report.score,
                report.metrics,
            )
        return 0

    report = orchestrator.run(args.max_steps)
    logger.info(
        "run complete: total_steps=%d per_component=%s rollbacks=%s",
        report.total_steps,
        report.steps_per_component,
        report.rollbacks,
    )
    for name, eval_report in report.final_evals.items():
        logger.info(
            "final eval %s: score=%.4f metrics=%s",
            name,
            eval_report.score,
            eval_report.metrics,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
