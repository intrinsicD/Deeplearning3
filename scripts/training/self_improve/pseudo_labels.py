"""Cross-component pseudo-label broker.

Design (``docs/self_improvement.md`` §5). The broker mediates "A teaches
B" relationships across plugins, with safeguards against the silent
collapse modes that plague naive co-training:

1. **Snapshot-mediated reads** — every label a consumer reads comes from
   a *frozen* producer snapshot in the vault, never from live weights.
   Cycles between two plugins can no longer feed back on the same step.
2. **Minimum staleness budget** — a consumer cannot refresh to a
   producer snapshot newer than ``min_stale_steps`` since its own last
   refresh on that edge. Prevents tight ping-pong.
3. **Confidence-gating** — each label carries a scalar confidence; the
   consumer drops labels below ``confidence_threshold``.
4. **Real-data majority cap** — enforced at the orchestrator/data-stream
   level, not here; the broker just provides labels.
5. **Edge-attribution divergence guard** — when a consumer regresses on
   its eval and the regression correlates with a recent refresh on edge
   ``e``, the orchestrator severs ``e`` and rolls the consumer back.
   The broker exposes ``mark_edge_severed`` / ``mark_edge_healed`` for
   the orchestrator's regression-detection loop.
6. **Acyclic-by-construction opt-out** — :func:`detect_cycles` /
   :func:`enforce_acyclic_or_raise` give the CLI a ``--strict-acyclic``
   path that refuses cyclic configurations at startup.

The broker is **plugin-agnostic**: callers register edges with a
producer plugin, a consumer plugin, and a *label function* that maps
a raw sample + producer snapshot to a (label, confidence) pair. The
labels themselves are opaque tensors; per-edge wiring lives in the
plugin glue (later phases).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Sequence

from scripts.training.self_improve.plugins.base import ComponentPlugin
from scripts.training.self_improve.vault import Snapshot, SnapshotID, Vault

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeConfig:
    """One pseudo-label edge in the producer → consumer graph.

    All four guards from §5.2 are declared explicitly; the design doc
    requires every cyclic edge to carry these (the startup check
    refuses cyclic edges that omit any).
    """

    producer: str
    consumer: str
    #: Minimum steps between producer-snapshot refreshes on this edge
    #: (§5.2 rule 2). Prevents A/B ping-pong loops.
    min_stale_steps: int = 1000
    #: Drop labels whose producer-reported confidence is below this
    #: threshold (§5.2 rule 3).
    confidence_threshold: float = 0.5
    #: Per-eval regression tolerance for the divergence guard (§5.2 rule 5).
    #: Defined here so the orchestrator can read the edge-specific tol
    #: without round-tripping through the consumer's eval registry.
    tol: float = 0.0
    #: Auto-heal an edge severed by the divergence guard after this
    #: many consumer steps without further regression.
    heal_after_steps: int = 5000


@dataclass
class PseudoLabelBatch:
    """A batch of labels emitted by a producer for a consumer.

    ``inputs`` and ``labels`` are opaque tensors (or batch-dict-likes)
    sized identically. ``confidences`` is a 1-D float tensor of length
    ``len(inputs)``.
    """

    producer: str
    consumer: str
    inputs: Any
    labels: Any
    confidences: Any
    producer_snapshot: SnapshotID


# ---------------------------------------------------------------------------
# Cycle detection (Tarjan's SCC)
# ---------------------------------------------------------------------------


def _build_adjacency(edges: Iterable[EdgeConfig]) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        adj[e.producer].append(e.consumer)
        adj.setdefault(e.consumer, [])  # ensure consumers appear as keys
    return adj


def _tarjan_scc(adj: dict[str, list[str]]) -> list[list[str]]:
    """Standard iterative Tarjan SCC. Returns SCCs in arbitrary order.

    Non-trivial SCCs (size ≥ 2, or singletons with self-loops) indicate
    cycles in the pseudo-label graph.
    """
    index_counter = [0]
    stack: list[str] = []
    lowlinks: dict[str, int] = {}
    indexes: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    result: list[list[str]] = []

    def strongconnect(node: str) -> None:
        # Iterative DFS so we can handle deep graphs without recursion.
        work_stack: list[tuple[str, Iterator[str]]] = [(node, iter(adj[node]))]
        indexes[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack[node] = True

        while work_stack:
            v, it = work_stack[-1]
            try:
                w = next(it)
                if w not in indexes:
                    indexes[w] = index_counter[0]
                    lowlinks[w] = index_counter[0]
                    index_counter[0] += 1
                    stack.append(w)
                    on_stack[w] = True
                    work_stack.append((w, iter(adj[w])))
                elif on_stack.get(w, False):
                    lowlinks[v] = min(lowlinks[v], indexes[w])
            except StopIteration:
                work_stack.pop()
                if lowlinks[v] == indexes[v]:
                    component: list[str] = []
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        component.append(w)
                        if w == v:
                            break
                    result.append(component)
                if work_stack:
                    parent, _ = work_stack[-1]
                    lowlinks[parent] = min(lowlinks[parent], lowlinks[v])

    for node in list(adj.keys()):
        if node not in indexes:
            strongconnect(node)
    return result


def detect_cycles(edges: Sequence[EdgeConfig]) -> list[list[str]]:
    """Return non-trivial strongly-connected components.

    Empty list ⇒ the graph is a DAG. Non-empty ⇒ each inner list is a
    cycle (≥2 nodes, or a single node with a self-loop).
    """
    adj = _build_adjacency(edges)
    sccs = _tarjan_scc(adj)
    cycles: list[list[str]] = []
    for scc in sccs:
        if len(scc) > 1:
            cycles.append(scc)
            continue
        # Singletons: only a cycle if there's a self-loop.
        [node] = scc
        if node in adj.get(node, []):
            cycles.append(scc)
    return cycles


def enforce_acyclic_or_raise(edges: Sequence[EdgeConfig]) -> None:
    """``--strict-acyclic`` startup check. Raises if any cycle exists."""
    cycles = detect_cycles(edges)
    if cycles:
        raise ValueError(
            f"pseudo-label graph contains cycle(s) under --strict-acyclic: "
            f"{cycles!r}"
        )


def validate_cyclic_edges_have_guards(
    edges: Sequence[EdgeConfig],
    *,
    min_stale_steps_floor: int = 100,
    confidence_threshold_floor: float = 0.0,
) -> None:
    """Default (non-strict) startup check from §5.3.

    Every edge that *participates in a cycle* must declare
    ``min_stale_steps`` and ``confidence_threshold`` at or above the
    floor values. Edges that participate in no cycle (including
    DAG-like *bridge* edges connecting two otherwise-disjoint cycles)
    skip the check.

    An edge participates in a cycle iff its producer and consumer
    belong to the *same* strongly-connected component. A previous
    implementation flattened all cyclic nodes into one set and flagged
    any edge whose endpoints were both members; that wrongly caught
    bridges between two separate SCCs and rejected valid configs.
    """
    cycles = detect_cycles(edges)
    if not cycles:
        return
    # Map each node to the index of its SCC (only for nodes that
    # actually live in a non-trivial SCC). Edges with endpoints in
    # different SCCs — or with one endpoint not in any cycle — pass.
    scc_id: dict[str, int] = {}
    for i, scc in enumerate(cycles):
        for node in scc:
            scc_id[node] = i
    bad: list[str] = []
    for e in edges:
        p_scc = scc_id.get(e.producer)
        c_scc = scc_id.get(e.consumer)
        if p_scc is None or c_scc is None or p_scc != c_scc:
            continue
        if e.min_stale_steps < min_stale_steps_floor:
            bad.append(
                f"{e.producer}->{e.consumer}: min_stale_steps="
                f"{e.min_stale_steps} < floor {min_stale_steps_floor}"
            )
        if e.confidence_threshold < confidence_threshold_floor:
            bad.append(
                f"{e.producer}->{e.consumer}: confidence_threshold="
                f"{e.confidence_threshold} < floor "
                f"{confidence_threshold_floor}"
            )
    if bad:
        raise ValueError(
            "cyclic pseudo-label edges missing required guards:\n  "
            + "\n  ".join(bad)
        )


# ---------------------------------------------------------------------------
# Broker state
# ---------------------------------------------------------------------------


LabelFn = Callable[
    [Any, ComponentPlugin],  # raw_sample, producer (with snapshot loaded)
    tuple[Any, float],       # (label, confidence) for this sample
]


@dataclass
class _EdgeState:
    """Mutable per-edge state held by the broker."""

    config: EdgeConfig
    label_fn: LabelFn
    snapshot: SnapshotID | None = None
    snapshot_loaded_at_step: int = 0
    consumer_steps_since_refresh: int = 0
    severed: bool = False
    consumer_steps_since_severed: int = 0
    refreshes: int = 0


class PseudoLabelBroker:
    """Producer→consumer label dispatcher with the §5 safeguards baked in.

    Lifecycle:

    1. ``register_edge(...)`` declares an edge with its guards and label
       function.
    2. The orchestrator calls ``record_consumer_step(consumer)`` after
       every train step on that consumer so the broker can advance
       staleness counters.
    3. When a consumer needs labels, the orchestrator calls
       ``sample(consumer, raw_inputs)`` and receives a list of
       :class:`PseudoLabelBatch` (one per active edge). Each batch
       respects the staleness budget, confidence threshold, and
       severed/healed state of its edge.
    4. On regression with edge-attribution, the orchestrator calls
       ``mark_edge_severed(edge_key)``; healing happens automatically
       after ``heal_after_steps`` clean consumer steps.
    """

    def __init__(
        self,
        vault: Vault,
        *,
        strict_acyclic: bool = False,
    ) -> None:
        self.vault = vault
        self.strict_acyclic = bool(strict_acyclic)
        self._edges: dict[tuple[str, str], _EdgeState] = {}
        self._producers: dict[str, ComponentPlugin] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_producer(self, name: str, plugin: ComponentPlugin) -> None:
        """Hold a reference to a producer plugin for snapshot loading.

        The broker doesn't own the plugin; it just needs a handle to
        :meth:`ComponentPlugin.load_state_dict` for snapshot-mediated
        reads. Repeated registration is idempotent.
        """
        self._producers[name] = plugin

    def register_edge(
        self,
        config: EdgeConfig,
        label_fn: LabelFn,
    ) -> None:
        if config.producer == config.consumer:
            raise ValueError(
                f"self-edge not allowed: {config.producer}->{config.consumer}"
            )
        key = (config.producer, config.consumer)
        if key in self._edges:
            raise KeyError(f"edge {key} already registered")
        self._edges[key] = _EdgeState(config=config, label_fn=label_fn)

    def finalize(self) -> None:
        """Run the startup graph check (cycles / guards).

        Call after all edges are registered, before the first
        :meth:`sample`. The orchestrator wires this in its constructor.
        """
        edges = [s.config for s in self._edges.values()]
        if self.strict_acyclic:
            enforce_acyclic_or_raise(edges)
        else:
            validate_cyclic_edges_have_guards(edges)
        logger.info(
            "broker finalized: %d edges, strict_acyclic=%s",
            len(edges),
            self.strict_acyclic,
        )

    # ------------------------------------------------------------------
    # Per-step bookkeeping
    # ------------------------------------------------------------------

    def record_consumer_step(self, consumer: str) -> None:
        """Advance staleness + healing counters for every edge into ``consumer``."""
        for (prod, cons), state in self._edges.items():
            if cons != consumer:
                continue
            state.consumer_steps_since_refresh += 1
            if state.severed:
                state.consumer_steps_since_severed += 1
                if (
                    state.consumer_steps_since_severed
                    >= state.config.heal_after_steps
                ):
                    state.severed = False
                    state.consumer_steps_since_severed = 0
                    logger.info(
                        "auto-healing severed edge %s->%s after %d clean steps",
                        prod,
                        cons,
                        state.config.heal_after_steps,
                    )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        consumer: str,
        raw_inputs: Sequence[Any],
    ) -> list[PseudoLabelBatch]:
        """For each active incoming edge, produce a label batch.

        Returns an empty list if no edges feed ``consumer`` or all of
        them are severed / lack a usable snapshot. Edges whose
        staleness budget isn't satisfied yet *don't* refresh — they
        emit labels from their stale snapshot, which is the correct
        thing per §5.2 rule 1.

        Confidence gating drops below-threshold samples; the surviving
        ``inputs`` / ``labels`` / ``confidences`` triples are
        size-aligned 1:1, matching the batch contract consumers rely
        on. An edge whose gating drops *every* sample emits no batch
        at all (rather than an empty batch) so consumers can skip the
        empty case uniformly.
        """
        out: list[PseudoLabelBatch] = []
        for (prod, cons), state in self._edges.items():
            if cons != consumer or state.severed:
                continue
            self._maybe_refresh_snapshot(state)
            if state.snapshot is None:
                # No snapshot available yet — producer hasn't been
                # vaulted. Skip silently.
                continue
            inputs, labels, confidences = self._emit_labels(state, raw_inputs)
            if not labels:
                # Every sample was dropped by confidence gating; emit no
                # batch (consumers expect either size-aligned content or
                # nothing).
                continue
            out.append(PseudoLabelBatch(
                producer=prod,
                consumer=cons,
                inputs=inputs,
                labels=labels,
                confidences=confidences,
                producer_snapshot=state.snapshot,
            ))
        return out

    def _maybe_refresh_snapshot(self, state: _EdgeState) -> None:
        """Promote a newer producer snapshot if the staleness budget has elapsed."""
        if (
            state.snapshot is not None
            and state.consumer_steps_since_refresh < state.config.min_stale_steps
        ):
            return
        latest = self.vault.best(state.config.producer)
        if latest is None:
            return
        # Only refresh if the candidate snapshot is strictly newer than
        # what we have. Avoids spurious reloads under tied scores.
        if state.snapshot is not None and latest == state.snapshot:
            return
        state.snapshot = latest
        state.consumer_steps_since_refresh = 0
        state.refreshes += 1

    def _emit_labels(
        self,
        state: _EdgeState,
        raw_inputs: Sequence[Any],
    ) -> tuple[list[Any], list[Any], list[float]]:
        """Run the producer (with its snapshot loaded) on ``raw_inputs``.

        The producer plugin is restored to its snapshot via
        :meth:`Vault.load`, the label function runs per-sample, and the
        producer is then restored to whatever it was on entry. The
        restore step matters when the orchestrator's main loop is also
        training the producer in parallel.

        Returns ``(inputs, labels, confidences)``, all the same length:
        the *surviving* inputs after confidence gating, paired with
        their labels and confidences. Returning the gated inputs (not
        the full input sequence) preserves the batch contract that
        consumers expect ``len(inputs) == len(labels)``.
        """
        import torch as _torch

        producer_name = state.config.producer
        producer = self._producers.get(producer_name)
        if producer is None:
            return [], [], []
        # Snapshot the producer's *current* state so we can restore after.
        # ``state_dict`` returns references to the live param tensors;
        # if we don't clone, ``load_state_dict(snap)`` below mutates
        # ``pre`` in-place and the restore is a no-op. ``deep_clone``
        # walks tensors and clones each one.
        pre = _deep_clone_state(producer.state_dict(), torch_mod=_torch)
        snap = self.vault.load(state.snapshot)
        try:
            producer.load_state_dict(snap.state)
            inputs: list[Any] = []
            labels: list[Any] = []
            confidences: list[float] = []
            for sample in raw_inputs:
                label, conf = state.label_fn(sample, producer)
                if conf >= state.config.confidence_threshold:
                    inputs.append(sample)
                    labels.append(label)
                    confidences.append(float(conf))
        finally:
            producer.load_state_dict(pre)
        return inputs, labels, confidences

    # ------------------------------------------------------------------
    # Divergence guard
    # ------------------------------------------------------------------

    def mark_edge_severed(self, producer: str, consumer: str) -> None:
        state = self._edges.get((producer, consumer))
        if state is None:
            raise KeyError(f"no such edge {producer}->{consumer}")
        state.severed = True
        state.consumer_steps_since_severed = 0
        logger.warning(
            "edge %s->%s marked severed by divergence guard", producer, consumer,
        )

    def mark_edge_healed(self, producer: str, consumer: str) -> None:
        """Manually clear the severed flag (mostly for tests / operator override)."""
        state = self._edges.get((producer, consumer))
        if state is None:
            raise KeyError(f"no such edge {producer}->{consumer}")
        state.severed = False
        state.consumer_steps_since_severed = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def edges(self) -> Sequence[EdgeConfig]:
        return [s.config for s in self._edges.values()]

    def is_severed(self, producer: str, consumer: str) -> bool:
        state = self._edges.get((producer, consumer))
        return bool(state and state.severed)

    def refresh_count(self, producer: str, consumer: str) -> int:
        state = self._edges.get((producer, consumer))
        return state.refreshes if state else 0


def _deep_clone_state(state: Any, *, torch_mod: Any) -> Any:
    """Recursively clone tensors in a nested ``state_dict``-like structure.

    ``nn.Module.state_dict()`` returns references to live parameter
    tensors; without cloning, the broker's "restore" pass after running
    a snapshot is a no-op because the captured ``pre`` shares storage
    with the params we just overwrote.
    """
    if isinstance(state, torch_mod.Tensor):
        return state.detach().clone()
    if isinstance(state, dict):
        return {k: _deep_clone_state(v, torch_mod=torch_mod) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        cloned = [_deep_clone_state(v, torch_mod=torch_mod) for v in state]
        return type(state)(cloned)
    return state


__all__ = [
    "EdgeConfig",
    "PseudoLabelBatch",
    "PseudoLabelBroker",
    "detect_cycles",
    "enforce_acyclic_or_raise",
    "validate_cyclic_edges_have_guards",
]
