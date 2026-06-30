"""Shared error types for the unified data layer."""

from __future__ import annotations


class MediaDecodeError(RuntimeError):
    """Raised when a media file cannot be decoded into a tensor.

    The data layer must fail loud rather than silently substituting a
    zero/empty tensor — training on zeros teaches the model corrupted
    targets while keeping the loss finite (see Audit.md A10).
    """


__all__ = ["MediaDecodeError"]
