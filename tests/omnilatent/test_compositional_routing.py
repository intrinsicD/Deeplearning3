"""W5.4 — compositional routing harness runs and reports the metric."""

from __future__ import annotations

import math

from scripts.diagnostics.compositional_routing import make_domains, run


def test_make_domains_shapes() -> None:
    d = make_domains(size=16, seed=0)
    assert d["base"].shape == (3, 16, 16)
    assert d["red_bias"].shape == (3, 16, 16)


def test_run_reports_composition_metrics() -> None:
    # Tiny run — exercises the harness, not the research magnitude.
    r = run(steps=5, batch=4, size=16, seed=0, freeze_backbone=True)
    for k in ("loss_none", "loss_A", "loss_B", "loss_both", "credit_A", "credit_B",
              "best_single", "composition_gap"):
        assert k in r and math.isfinite(r[k])
    # The gap is best_single - both, by construction.
    assert math.isclose(r["composition_gap"], r["best_single"] - r["loss_both"], rel_tol=1e-6)
