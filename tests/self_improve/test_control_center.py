"""Phase 10 tests: control-center integration for the self-improve CLI.

Acceptance test (design doc):
    "UI smoke test."

We exercise the FastAPI endpoints via ``fastapi.testclient.TestClient``
(no live HTTP server). Three slices:

1. Build a self-improve command from a request payload and verify the
   shell args match what ``scripts/training/self_improve/cli.py`` expects.
2. Start a self-improve subprocess via the endpoint, wait for it to
   complete, and verify the JOB state transitions are sane.
3. The status endpoint returns "idle" when no self-improve job is
   active and the live snapshot when one is.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from apps.control_center.server import (
    JOB,
    SelfImproveStartRequest,
    _build_self_improve_command,
    app,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _reset_job() -> None:
    """Each test starts with an idle JobState."""
    JOB.stop()
    yield
    JOB.stop()


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------


def test_command_includes_defaults() -> None:
    req = SelfImproveStartRequest()
    cmd = _build_self_improve_command(req)
    assert "scripts/training/self_improve_cli.py" in " ".join(cmd)
    assert "--components" in cmd
    assert "gaussian_encoder" in cmd
    assert "--max-steps" in cmd


def test_command_passes_all_components() -> None:
    req = SelfImproveStartRequest(components=["lgq", "hpwm", "omnilatent"])
    cmd = _build_self_improve_command(req)
    # Verify each component appears in the right slot — after --components,
    # in the same order.
    idx = cmd.index("--components")
    after = cmd[idx + 1: idx + 4]
    assert after == ["lgq", "hpwm", "omnilatent"]


def test_command_forwards_vault_root() -> None:
    req = SelfImproveStartRequest(vault_root="/tmp/my-vault")
    cmd = _build_self_improve_command(req)
    assert "--vault-root" in cmd
    assert "/tmp/my-vault" in cmd


def test_command_includes_dry_run_only_when_requested() -> None:
    req = SelfImproveStartRequest(dry_run=False)
    assert "--dry-run" not in _build_self_improve_command(req)
    req = SelfImproveStartRequest(dry_run=True)
    assert "--dry-run" in _build_self_improve_command(req)


def test_command_serializes_floats_as_str() -> None:
    """Tolerances must be passable on the command line."""
    req = SelfImproveStartRequest(rollback_tol=0.001)
    cmd = _build_self_improve_command(req)
    idx = cmd.index("--rollback-tol")
    assert cmd[idx + 1] == "0.001"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_status_when_idle(client: TestClient) -> None:
    r = client.get("/api/self-improve/status")
    assert r.status_code == 200
    assert r.json() == {"state": "idle", "kind": None}


def test_start_then_status_then_stop(client: TestClient, tmp_path: Path) -> None:
    """End-to-end: launch a minimal --dry-run job, observe status,
    let it complete, observe state transition.

    --dry-run does one eval per component and exits ⇒ runs in seconds
    and exercises the full subprocess plumbing without GPU.
    """
    payload = {
        "components": ["gaussian_encoder"],
        "max_steps": 1,
        "eval_every": 0,
        "batch_size": 2,
        "seed": 0,
        "dry_run": True,
    }
    r = client.post("/api/self-improve/start", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["state"] == "running"
    assert body["kind"] == "self-improve"

    # Wait for the subprocess to finish (--dry-run completes quickly).
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        r = client.get("/api/self-improve/status")
        state = r.json().get("state")
        if state in ("finished", "error", "idle"):
            break
        time.sleep(0.2)
    else:
        raise AssertionError("self-improve dry-run did not finish in 60s")

    # Should have finished cleanly; --dry-run exits 0.
    assert r.json().get("state") == "finished", r.json()


def test_start_rejects_when_job_running(client: TestClient) -> None:
    """The job slot is exclusive — concurrent starts must 409."""
    # Start a self-improve job (long enough to overlap with the second call).
    payload = {
        "components": ["gaussian_encoder"],
        "max_steps": 100,
        "eval_every": 1000,
        "dry_run": False,
    }
    r = client.post("/api/self-improve/start", json=payload)
    assert r.status_code == 200, r.text

    # Try to start another while this one is alive → 409.
    r2 = client.post("/api/self-improve/start", json=payload)
    assert r2.status_code == 409, r2.text


def test_stop_is_idempotent(client: TestClient) -> None:
    """Calling /stop with no active job is a no-op, returns 200."""
    r = client.post("/api/self-improve/stop")
    assert r.status_code == 200
    # And again.
    r = client.post("/api/self-improve/stop")
    assert r.status_code == 200


def test_status_filters_to_self_improve_jobs_only(client: TestClient) -> None:
    """The self-improve status endpoint must NOT report on other job
    kinds; if a training job is active, /api/self-improve/status
    should still return idle."""
    # Manually inject a non-self-improve job state.
    JOB.status = {"state": "running", "kind": "train:mmwm_minari", "step": 5}
    r = client.get("/api/self-improve/status")
    assert r.json() == {"state": "idle", "kind": None}
