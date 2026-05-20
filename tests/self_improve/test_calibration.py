"""Tests for the pseudo-label calibration script (§10.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.training.self_improve.calibrate_pseudo_labels import (
    EdgeReport,
    EdgeSpec,
    _resolve_label_fn,
    calibrate_edge,
    main,
)


# ---------------------------------------------------------------------------
# EdgeSpec parsing
# ---------------------------------------------------------------------------


def test_edge_spec_parses_triple() -> None:
    spec = EdgeSpec.parse("prod:cons:my.module.fn")
    assert spec.producer == "prod"
    assert spec.consumer == "cons"
    assert spec.label_fn == "my.module.fn"


def test_edge_spec_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        EdgeSpec.parse("only-two:parts")
    with pytest.raises(ValueError):
        EdgeSpec.parse("a:b:c:d")


# ---------------------------------------------------------------------------
# Label-fn resolution
# ---------------------------------------------------------------------------


def test_resolve_label_fn_short_form() -> None:
    """A bare name resolves under ``scripts.training.self_improve.edges``."""
    fn = _resolve_label_fn("gaussian_recon_label_fn")
    assert callable(fn)


def test_resolve_label_fn_dotted_form() -> None:
    fn = _resolve_label_fn(
        "scripts.training.self_improve.edges.gaussian_recon_label_fn",
    )
    assert callable(fn)


def test_resolve_label_fn_missing_module() -> None:
    with pytest.raises(SystemExit):
        _resolve_label_fn("definitely.not.a.module.fn")


def test_resolve_label_fn_missing_attr() -> None:
    with pytest.raises(SystemExit):
        _resolve_label_fn(
            "scripts.training.self_improve.edges.definitely_not_a_function",
        )


# ---------------------------------------------------------------------------
# calibrate_edge — sweep math
# ---------------------------------------------------------------------------


def test_calibrate_edge_returns_well_formed_report() -> None:
    """The sweep should produce one row per threshold, monotonically
    decreasing volume (as the threshold rises, fewer samples survive),
    and a recommendation that maximizes ``volume * mean_kept_conf``.
    """
    spec = EdgeSpec(
        producer="gaussian_encoder",
        consumer="gaussian_encoder",
        label_fn="gaussian_recon_label_fn",
    )
    thresholds = [0.0, 0.2, 0.5, 0.8, 0.95]
    report = calibrate_edge(spec, n_samples=16, thresholds=thresholds)

    assert isinstance(report, EdgeReport)
    assert report.n_samples == 16
    assert [row.threshold for row in report.sweep] == thresholds

    # Volume is non-increasing in threshold.
    volumes = [row.volume for row in report.sweep]
    assert all(volumes[i] >= volumes[i + 1] for i in range(len(volumes) - 1)), (
        f"volume not monotone non-increasing: {volumes}"
    )
    # Threshold 0.0 keeps every sample.
    assert report.sweep[0].volume == 1.0
    # Recommendation has the highest score in the sweep.
    best = max(report.sweep, key=lambda r: r.score)
    assert report.recommended_threshold == best.threshold
    assert report.recommended_score == best.score


def test_calibrate_edge_uses_default_threshold_grid() -> None:
    """When ``thresholds`` is omitted, the sweep uses an 11-point
    default grid."""
    spec = EdgeSpec(
        producer="gaussian_encoder",
        consumer="gaussian_encoder",
        label_fn="gaussian_recon_label_fn",
    )
    report = calibrate_edge(spec, n_samples=8)
    assert len(report.sweep) == 11


def test_calibrate_edge_rejects_unsupported_consumer() -> None:
    spec = EdgeSpec(
        producer="gaussian_encoder",
        consumer="not_a_real_plugin",
        label_fn="gaussian_recon_label_fn",
    )
    with pytest.raises(SystemExit, match="probe builder"):
        calibrate_edge(spec, n_samples=4)


def test_calibrate_edge_rejects_unknown_producer_kind() -> None:
    spec = EdgeSpec(
        producer="definitely_not_a_plugin_name",
        consumer="gaussian_encoder",
        label_fn="gaussian_recon_label_fn",
    )
    with pytest.raises(SystemExit, match="plugin kind"):
        calibrate_edge(spec, n_samples=4)


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_writes_json_report(tmp_path: Path) -> None:
    """End-to-end through the CLI: --edge + --out produces a JSON
    file whose contents match the per-edge calibration report."""
    out = tmp_path / "calibration.json"
    rc = main([
        "--edge", "gaussian_encoder:gaussian_encoder:gaussian_recon_label_fn",
        "--n-samples", "8",
        "--thresholds", "0.0", "0.5", "0.9",
        "--out", str(out),
        "--seed", "0",
    ])
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 1
    report = data[0]
    assert report["producer"] == "gaussian_encoder"
    assert report["consumer"] == "gaussian_encoder"
    assert len(report["sweep"]) == 3
    assert report["recommended_threshold"] in (0.0, 0.5, 0.9)


def test_cli_handles_multiple_edges(tmp_path: Path) -> None:
    """Multiple ``--edge`` arguments are accepted; the report has one
    entry per edge in the order given."""
    out = tmp_path / "calibration.json"
    rc = main([
        "--edge", "gaussian_encoder:gaussian_encoder:gaussian_recon_label_fn",
        "--edge", "gaussian_encoder:gaussian_encoder:gaussian_recon_label_fn",
        "--n-samples", "4",
        "--thresholds", "0.0", "0.5",
        "--out", str(out),
        "--seed", "0",
    ])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 2


def test_cli_surfaces_label_fn_errors_loudly(tmp_path: Path) -> None:
    """A calibration tool's job is to make broken edge configs *loud*.
    Pointing a Gaussian (1-channel) label_fn at the LGQ (3-channel)
    probe set should raise — the operator wants to know the edge
    can't run as configured, not get a silent zero in the report.
    """
    out = tmp_path / "calibration.json"
    with pytest.raises(Exception):
        main([
            "--edge", "lgq:lgq:gaussian_recon_label_fn",
            "--n-samples", "4",
            "--thresholds", "0.0", "0.5",
            "--out", str(out),
            "--seed", "0",
        ])
