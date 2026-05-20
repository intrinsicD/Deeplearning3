"""Reservoir-sampled replay buffer with DER++ logits storage.

Design (``docs/self_improvement.md`` §4.1):

- Per-component fixed-capacity buffer; insertion uses **reservoir
  sampling** (Vitter 1985) so the buffer is at all times a uniform
  random sample of the stream seen so far.
- DER++ (Buzzega et al. 2020) additionally stores the model's *outputs*
  on each sample at insertion time. The consumer adds a consistency
  term to its loss: ``MSE(model(sample_now), stored_outputs)``. Stored
  outputs are opaque tensors; the plugin decides what to store
  (reconstruction, logits, latent — whatever it can reproduce on read).

The buffer is plugin-agnostic: ``sample`` and ``stored_logits`` are both
``Any``. Each plugin defines its own item layout. This module just
maintains the reservoir invariant.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class ReplayItem:
    """A single buffered sample with its DER++ teacher signal.

    - ``sample`` is whatever the plugin will feed back through its
      forward pass (tensor, batch dict, etc.). Stored on CPU.
    - ``stored_logits`` is the teacher output at insertion time, or
      ``None`` if DER++ storage is disabled for this insertion. Also CPU.
    - ``insert_step`` is the global step count at insertion — useful for
      staleness diagnostics (§5.2 min_stale_steps; consumed by phase 7).
    """

    sample: Any
    stored_logits: Any | None
    insert_step: int


class ReplayBank:
    """Per-component reservoir buffers.

    One buffer per component name. Components share the bank so the
    orchestrator can hand the same instance to every plugin without
    cross-contamination — each plugin's data lives under its own key.

    Reservoir sampling guarantees that after seeing ``n`` items, each
    item is in the buffer with probability ``min(1, capacity/n)``.
    """

    def __init__(self, capacity: int = 10_000, *, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive; got {capacity}")
        self.capacity = capacity
        self._buffers: dict[str, list[ReplayItem]] = {}
        self._counts: dict[str, int] = {}
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def insert(
        self,
        component: str,
        sample: Any,
        *,
        stored_logits: Any | None = None,
        step: int = 0,
    ) -> bool:
        """Reservoir-insert ``sample`` into ``component``'s buffer.

        Returns ``True`` if the item ended up in the buffer (either by
        filling a free slot or by replacing an existing entry), ``False``
        if it was rejected by the reservoir step.
        """
        buf = self._buffers.setdefault(component, [])
        n = self._counts.get(component, 0)
        self._counts[component] = n + 1

        item = ReplayItem(
            sample=sample,
            stored_logits=stored_logits,
            insert_step=step,
        )

        if len(buf) < self.capacity:
            buf.append(item)
            return True

        # Buffer is full: replace position uniformly with probability
        # capacity / (n+1). This is the standard reservoir replacement
        # step — see Vitter 1985 algorithm R.
        j = self._rng.randint(0, n)  # inclusive of n, so range [0, n]
        if j < self.capacity:
            buf[j] = item
            return True
        return False

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, component: str, k: int) -> list[ReplayItem]:
        """Draw up to ``k`` items uniformly without replacement.

        Returns fewer than ``k`` items if the buffer is smaller than
        ``k``. Returns an empty list if the buffer is empty.
        """
        if k <= 0:
            return []
        buf = self._buffers.get(component, [])
        if not buf:
            return []
        if len(buf) <= k:
            return list(buf)
        return self._rng.sample(buf, k)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return sum(len(b) for b in self._buffers.values())

    def size(self, component: str) -> int:
        return len(self._buffers.get(component, []))

    def total_seen(self, component: str) -> int:
        """Total items ever passed to ``insert`` for ``component``."""
        return self._counts.get(component, 0)

    def components(self) -> list[str]:
        return list(self._buffers.keys())

    def clear(self, component: str | None = None) -> None:
        if component is None:
            self._buffers.clear()
            self._counts.clear()
        else:
            self._buffers.pop(component, None)
            self._counts.pop(component, None)


__all__ = ["ReplayBank", "ReplayItem"]
