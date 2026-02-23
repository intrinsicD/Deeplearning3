"""Learnable Geometric Quantization (LGQ) for discrete image tokenization.

LGQ learns its discretization geometry end-to-end using temperature-controlled
soft assignments during training that sharpen into hard assignments at inference.
Based on an isotropic Gaussian mixture interpretation optimized with a variational
free-energy objective.
"""

from lgq.config import LGQConfig
from lgq.quantizer import LGQuantizer
from lgq.model import LGQVAE
from lgq.losses import LGQLoss

__all__ = ["LGQConfig", "LGQuantizer", "LGQVAE", "LGQLoss"]
