"""Input pipeline for the self-improvement orchestrator.

In production (phase 7+), the data stream demuxes raw video into
per-modality tensors and feeds each plugin a batch matching its
``modality_filter``. For phase 5 single-component mode we just route
:meth:`ComponentPlugin.make_synthetic_batch` calls; phase 6+ plugs in
a real :class:`DataStream` subclass that reads from
``apps.control_center.session_data`` or a video directory.

The :class:`DataStream` ABC + :class:`SyntheticDataStream` separation
exists so the orchestrator code is identical between phase 5 (toy
batches) and phase 7 (real video), and so we can swap in a fake stream
in tests.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from scripts.training.self_improve.plugins.base import ComponentPlugin


class DataStream(ABC):
    """Source of training batches, keyed by plugin."""

    @abstractmethod
    def next(self, plugin: ComponentPlugin, *, batch_size: int = 2) -> Any:
        """Return the next batch suitable for ``plugin``'s ``train_step``."""


class SyntheticDataStream(DataStream):
    """Round-trips each request to ``plugin.make_synthetic_batch``.

    This is the phase-5 data source: the orchestrator doesn't need
    actual data to verify its scheduling + eval + vault loop. The CLI's
    ``--dry-run`` mode also uses this.
    """

    def __init__(self, *, batch_size: int = 2) -> None:
        self.batch_size = batch_size

    def next(self, plugin: ComponentPlugin, *, batch_size: int | None = None) -> Any:
        bs = self.batch_size if batch_size is None else batch_size
        return plugin.make_synthetic_batch(batch_size=bs)


__all__ = ["DataStream", "SyntheticDataStream"]
