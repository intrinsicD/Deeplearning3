"""TensorBoard compatibility helpers.

Provides a no-op SummaryWriter fallback when tensorboard isn't installed.
"""

from __future__ import annotations

from typing import Any


try:  # pragma: no cover - exercised implicitly when tensorboard exists
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - common in lightweight CI
    class _SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def add_scalar(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def add_histogram(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def add_embedding(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def add_text(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def close(self) -> None:
            pass


SummaryWriter = _SummaryWriter

