"""Task sampler for multi-modal training.

Selects (source_modality, target_modality) pairs for each training step,
providing uniform coverage of all modality combinations.
"""

from __future__ import annotations

import itertools
import random
from typing import Sequence

from omnilatent.utils import ALL_MODALITIES, Modality


class TaskSampler:
    """Samples (source, target) modality pairs for training steps.

    Supports multiple strategies:
      - uniform: random from all possible pairs
      - self_recon_biased: biased toward self-reconstruction tasks
      - available_only: only sample from modalities present in the batch
    """

    def __init__(
        self,
        modalities: Sequence[Modality] | None = None,
        self_recon_weight: float = 0.5,
    ) -> None:
        self.modalities = list(modalities or ALL_MODALITIES)
        self.self_recon_weight = self_recon_weight
        self._all_pairs = list(
            itertools.product(self.modalities, self.modalities)
        )
        self._self_pairs = [(m, m) for m in self.modalities]
        self._cross_pairs = [
            (s, t) for s, t in self._all_pairs if s != t
        ]

    def sample(
        self,
        available: Sequence[str] | None = None,
    ) -> tuple[str, str]:
        """Sample a (source, target) modality pair.

        Args:
            available: if provided, restrict to these modalities.

        Returns:
            (source_modality, target_modality) strings.
        """
        if available is not None and len(available) > 0:
            src = random.choice(list(available))
            tgt = random.choice(list(available))
            return src, tgt

        # Bias toward self-reconstruction
        if random.random() < self.self_recon_weight and self._self_pairs:
            return random.choice(self._self_pairs)
        return random.choice(self._all_pairs)

    def sample_batch(
        self,
        n: int,
        available: Sequence[str] | None = None,
    ) -> list[tuple[str, str]]:
        """Sample n (source, target) pairs."""
        return [self.sample(available) for _ in range(n)]
