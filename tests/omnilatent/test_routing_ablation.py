"""W6.2 — routing ablation harness runs and reports all three arms."""

from __future__ import annotations

import math

from scripts.diagnostics.routing_ablation import make_multidomain_dataset, run_ablation


def test_multidomain_dataset_shape() -> None:
    samples = make_multidomain_dataset(n_domains=3, image_size=16, n_per_domain=5)
    assert len(samples) == 15
    assert samples[0].shape == (3, 16, 16)


def test_run_ablation_reports_three_arms() -> None:
    # Tiny run — exercises the full harness path, not the research result.
    results = run_ablation(
        n_domains=2, n_hooks=2, steps=3, batch_size=4,
        top_k=1, image_size=16, freeze_backbone=True,
    )
    assert set(results) == {"no_hooks", "always_on", "routed"}
    assert all(math.isfinite(results[m]["loss"]) for m in results)
    # Honest compute proxy is recorded per arm.
    assert results["no_hooks"]["injected"] == 0
    assert results["always_on"]["injected"] == 2
    assert results["routed"]["injected"] <= 2
