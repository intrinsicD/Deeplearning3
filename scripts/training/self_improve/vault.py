"""Content-addressed checkpoint vault for the self-improvement orchestrator.

Phase 2: the storage backend for snapshots + the lookup of the
best-scoring snapshot per component. Phases 4 and 9 extend it with
co-stored Fisher/SI/EMA/replay state and LRU eviction.

Design notes
------------

* **Content addressing.** Each snapshot's filename is the SHA-256 of the
  bytes produced by ``torch.save(plugin.state_dict(), buf)``. Identical
  states therefore deduplicate automatically.
* **Index.** A small JSON sidecar ``index.json`` maps each component
  name to a list of snapshot metadata (hash, step, score, etc.). It is
  loaded eagerly on construction and rewritten atomically on save.
* **best().** ``Vault.best(component)`` is a pure index lookup; it does
  not touch the heavy ``.pt`` files. Tie-breaks favor the higher step.

The vault is process-safe under cooperating writers (single-process,
no fsync, no lock) — phase 9 adds a file lock for multi-process safety.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from scripts.training.self_improve.plugins.base import (
    ComponentPlugin,
    EvalReport,
)


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotID:
    """Stable, hashable handle to a stored snapshot."""

    component: str
    hash: str           # full SHA-256 hex digest
    step: int

    @property
    def short(self) -> str:
        return self.hash[:12]


@dataclass
class Snapshot:
    """A snapshot loaded from disk."""

    id: SnapshotID
    state: dict[str, Any]
    report: EvalReport
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Vault
# ---------------------------------------------------------------------------


def _state_bytes(state: dict[str, Any]) -> bytes:
    """Serialise a plugin ``state_dict`` to deterministic bytes."""
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def _state_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Vault:
    """Disk-backed content-addressed snapshot store."""

    INDEX_NAME = "index.json"

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / self.INDEX_NAME
        idx = self._load_index()
        # Index has two sections: per-component snapshot lists ("snapshots")
        # and per-component conventions ("meta"). Older single-section
        # indexes are migrated transparently.
        self._snapshots: dict[str, list[dict[str, Any]]] = idx.get("snapshots", {})
        self._meta: dict[str, dict[str, Any]] = idx.get("meta", {})

    # ------------------------------------------------------------------
    # Index I/O
    # ------------------------------------------------------------------

    def _load_index(self) -> dict[str, Any]:
        if not self._index_path.exists():
            return {}
        with self._index_path.open("r") as f:
            data = json.load(f)
        # Migration from the legacy flat schema (component → list[entry]).
        if data and "snapshots" not in data and "meta" not in data:
            return {"snapshots": data, "meta": {}}
        return data

    def _save_index(self) -> None:
        tmp = self._index_path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(
                {"snapshots": self._snapshots, "meta": self._meta},
                f, indent=2, sort_keys=True,
            )
        os.replace(tmp, self._index_path)

    def _snapshot_path(self, component: str, sha: str) -> Path:
        return self.root / component / f"{sha}.pt"

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(
        self,
        component: str,
        plugin: ComponentPlugin,
        report: EvalReport,
        *,
        step: int,
        metadata: dict[str, Any] | None = None,
    ) -> SnapshotID:
        """Persist ``plugin``'s state under *component* and return its ID.

        Raises:
            ValueError: if *report.higher_is_better* disagrees with the
                convention established by the first save for *component*.
                Conventions are a component-level property — switching
                them mid-history would silently break :meth:`best`.
        """
        # Enforce component-level convention consistency.
        meta = self._meta.setdefault(component, {})
        existing = meta.get("higher_is_better")
        if existing is None:
            meta["higher_is_better"] = bool(report.higher_is_better)
        elif bool(existing) != bool(report.higher_is_better):
            raise ValueError(
                f"vault component {component!r} was first saved with "
                f"higher_is_better={existing}; cannot mix conventions "
                f"(got higher_is_better={report.higher_is_better}). "
                f"Decide the convention per component once and stick to it."
            )

        state = plugin.state_dict()
        data = _state_bytes(state)
        sha = _state_hash(data)

        path = self._snapshot_path(component, sha)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        entry = {
            "hash": sha,
            "step": int(step),
            "score": float(report.score),
            "metrics": {k: float(v) for k, v in report.metrics.items()},
            "metadata": dict(metadata or {}),
            "created_at": time.time(),
        }
        # Deduplicate: replace an existing entry with the same hash + step.
        bucket = self._snapshots.setdefault(component, [])
        bucket[:] = [e for e in bucket if not (e["hash"] == sha and e["step"] == step)]
        bucket.append(entry)
        self._save_index()

        return SnapshotID(component=component, hash=sha, step=int(step))

    def load(self, snapshot_id: SnapshotID) -> Snapshot:
        """Load a snapshot's state + report from disk."""
        path = self._snapshot_path(snapshot_id.component, snapshot_id.hash)
        if not path.exists():
            raise FileNotFoundError(f"snapshot not found: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)

        entry = self._find_entry(snapshot_id)
        if entry is None:
            raise KeyError(f"snapshot {snapshot_id} present on disk but missing from index")
        report = EvalReport(
            score=entry["score"],
            metrics=dict(entry["metrics"]),
            higher_is_better=self.higher_is_better(snapshot_id.component),
        )
        return Snapshot(
            id=snapshot_id,
            state=state,
            report=report,
            metadata=dict(entry.get("metadata", {})),
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def higher_is_better(self, component: str) -> bool:
        """Component-level scoring convention, fixed at first save."""
        meta = self._meta.get(component)
        if meta is None or "higher_is_better" not in meta:
            # Default to True for unknown components — preserves the
            # historical EvalReport default but should never trigger
            # if save() has been called at least once.
            return True
        return bool(meta["higher_is_better"])

    def list(self, component: str) -> list[SnapshotID]:
        bucket = self._snapshots.get(component, [])
        return [
            SnapshotID(component=component, hash=e["hash"], step=int(e["step"]))
            for e in bucket
        ]

    def best(self, component: str) -> SnapshotID | None:
        """Return the ID of the best-scoring snapshot for *component*."""
        bucket = self._snapshots.get(component, [])
        if not bucket:
            return None
        higher_is_better = self.higher_is_better(component)
        # Tie-break on the higher step (more recent ⇒ more trusted).
        key = (lambda e: (e["score"], e["step"])) if higher_is_better \
            else (lambda e: (-e["score"], e["step"]))
        best = max(bucket, key=key)
        return SnapshotID(component=component, hash=best["hash"], step=int(best["step"]))

    def _find_entry(self, snapshot_id: SnapshotID) -> dict[str, Any] | None:
        bucket = self._snapshots.get(snapshot_id.component, [])
        for e in bucket:
            if e["hash"] == snapshot_id.hash and int(e["step"]) == snapshot_id.step:
                return e
        # Fall back to hash-only match (e.g. content-addressed dedup).
        for e in bucket:
            if e["hash"] == snapshot_id.hash:
                return e
        return None

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(
        self,
        component: str,
        plugin: ComponentPlugin,
        *,
        snapshot_id: SnapshotID | None = None,
    ) -> Snapshot:
        """Restore *plugin* to *snapshot_id* (defaulting to the best one)."""
        sid = snapshot_id or self.best(component)
        if sid is None:
            raise LookupError(f"no snapshots stored for component {component!r}")
        snap = self.load(sid)
        plugin.load_state_dict(snap.state)
        return snap

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def disk_usage_bytes(self) -> int:
        total = 0
        for component_dir in self.root.iterdir():
            if not component_dir.is_dir():
                continue
            for f in component_dir.glob("*.pt"):
                total += f.stat().st_size
        return total

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "components": {
                c: [asdict(s) for s in self.list(c)]
                for c in self._snapshots.keys()
            },
            "conventions": {
                c: self.higher_is_better(c) for c in self._snapshots.keys()
            },
            "disk_bytes": self.disk_usage_bytes(),
        }


__all__ = ["Snapshot", "SnapshotID", "Vault"]
