"""Adapters package – convert AudioVisualSample to model-native batch dicts."""

from .mmwm_adapter import MMWMAdapter, adapt_mmwm
from .hpwm_adapter import HPWMAdapter, adapt_hpwm
from .lgq_adapter import LGQAdapter, adapt_lgq
from .gaussian_encoder_adapter import GaussianEncoderAdapter, adapt_gaussian_encoder

__all__ = [
    "MMWMAdapter",     "adapt_mmwm",
    "HPWMAdapter",     "adapt_hpwm",
    "LGQAdapter",      "adapt_lgq",
    "GaussianEncoderAdapter", "adapt_gaussian_encoder",
]

