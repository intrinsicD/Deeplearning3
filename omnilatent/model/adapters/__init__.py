"""Modality adapter abstractions for OmniLatent.

Each modality implements the ModalityAdapter protocol, providing a
uniform interface for encoding, query generation, and decoding.
"""

from omnilatent.model.adapters.base import (
    Encoded,
    ModalityAdapter,
    QueryResult,
)

__all__ = ["Encoded", "ModalityAdapter", "QueryResult"]
