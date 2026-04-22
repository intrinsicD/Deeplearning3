"""Curriculum schedule definitions for staged training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional


@dataclass
class CurriculumPhase:
    phase_id: int
    until_step: int
    task_multipliers: Dict[str, float]


_ALL_TASK_KEYS = [
    "latent_sem_loss", "latent_dyn_loss", "latent_ctrl_loss", "latent_ctx_loss",
    "regularizer_loss", "text_ce_loss", "vector_recon_loss",
    "image_recon_loss", "audio_recon_loss", "contrastive_alignment_loss",
]


def _mults(**overrides: float) -> Dict[str, float]:
    """Build a full task_multipliers dict, defaulting unset keys to 0.0."""
    m = {k: 0.0 for k in _ALL_TASK_KEYS}
    m.update(overrides)
    return m


def default_curriculum_phases() -> List[CurriculumPhase]:
    """Five-phase curriculum adapted for MMWM training."""
    return [
        CurriculumPhase(phase_id=1, until_step=1_000, task_multipliers=_mults(
            latent_sem_loss=1.0, latent_dyn_loss=0.5, regularizer_loss=1.0,
        )),
        CurriculumPhase(phase_id=2, until_step=3_000, task_multipliers=_mults(
            latent_sem_loss=1.0, latent_dyn_loss=1.0, latent_ctrl_loss=0.5,
            latent_ctx_loss=0.25, regularizer_loss=1.0,
        )),
        CurriculumPhase(phase_id=3, until_step=6_000, task_multipliers=_mults(
            latent_sem_loss=1.0, latent_dyn_loss=1.0, latent_ctrl_loss=1.0,
            latent_ctx_loss=0.75, regularizer_loss=1.0,
            text_ce_loss=0.25, vector_recon_loss=0.25,
            image_recon_loss=0.25, audio_recon_loss=0.25,
            contrastive_alignment_loss=0.25,
        )),
        CurriculumPhase(phase_id=4, until_step=10_000, task_multipliers=_mults(
            latent_sem_loss=1.0, latent_dyn_loss=1.0, latent_ctrl_loss=1.0,
            latent_ctx_loss=1.0, regularizer_loss=1.0,
            text_ce_loss=0.75, vector_recon_loss=0.75,
            image_recon_loss=0.75, audio_recon_loss=0.75,
            contrastive_alignment_loss=0.75,
        )),
        CurriculumPhase(phase_id=5, until_step=1_000_000_000, task_multipliers=_mults(
            latent_sem_loss=1.0, latent_dyn_loss=1.0, latent_ctrl_loss=1.0,
            latent_ctx_loss=1.0, regularizer_loss=1.0,
            text_ce_loss=1.0, vector_recon_loss=1.0,
            image_recon_loss=1.0, audio_recon_loss=1.0,
            contrastive_alignment_loss=1.0,
        )),
    ]


def relative_curriculum_phases(total_steps: int) -> List[CurriculumPhase]:
    """Five-phase curriculum with boundaries relative to total training steps.

    Phase boundaries at 5%, 15%, 35%, 60%, 100% of total_steps — avoiding
    hard-coded step counts that require manual tuning per dataset/scale.
    """
    boundaries = [
        int(0.05 * total_steps),
        int(0.15 * total_steps),
        int(0.35 * total_steps),
        int(0.60 * total_steps),
        total_steps,
    ]
    return [
        CurriculumPhase(phase_id=1, until_step=boundaries[0], task_multipliers=_mults(
            latent_sem_loss=1.0, latent_dyn_loss=0.5, regularizer_loss=1.0,
        )),
        CurriculumPhase(phase_id=2, until_step=boundaries[1], task_multipliers=_mults(
            latent_sem_loss=1.0, latent_dyn_loss=1.0, latent_ctrl_loss=0.5,
            latent_ctx_loss=0.25, regularizer_loss=1.0,
        )),
        CurriculumPhase(phase_id=3, until_step=boundaries[2], task_multipliers=_mults(
            latent_sem_loss=1.0, latent_dyn_loss=1.0, latent_ctrl_loss=1.0,
            latent_ctx_loss=0.75, regularizer_loss=1.0,
            text_ce_loss=0.25, vector_recon_loss=0.25,
            image_recon_loss=0.25, audio_recon_loss=0.25,
            contrastive_alignment_loss=0.25,
        )),
        CurriculumPhase(phase_id=4, until_step=boundaries[3], task_multipliers=_mults(
            latent_sem_loss=1.0, latent_dyn_loss=1.0, latent_ctrl_loss=1.0,
            latent_ctx_loss=1.0, regularizer_loss=1.0,
            text_ce_loss=0.75, vector_recon_loss=0.75,
            image_recon_loss=0.75, audio_recon_loss=0.75,
            contrastive_alignment_loss=0.75,
        )),
        CurriculumPhase(phase_id=5, until_step=boundaries[4], task_multipliers=_mults(
            latent_sem_loss=1.0, latent_dyn_loss=1.0, latent_ctrl_loss=1.0,
            latent_ctx_loss=1.0, regularizer_loss=1.0,
            text_ce_loss=1.0, vector_recon_loss=1.0,
            image_recon_loss=1.0, audio_recon_loss=1.0,
            contrastive_alignment_loss=1.0,
        )),
    ]


class AdaptiveCurriculumScheduler:
    """Monitors loss plateaus and triggers phase transitions automatically.

    Tracks the total loss in a sliding window. When improvement drops below
    a patience-based threshold, advances to the next phase. Falls back to
    the static schedule's task_multipliers but with dynamic boundaries.
    """

    def __init__(
        self,
        phases: List[CurriculumPhase],
        patience: int = 500,
        min_improvement: float = 0.01,
        window_size: int = 100,
    ) -> None:
        self.phases = phases
        self.patience = patience
        self.min_improvement = min_improvement
        self.current_phase_idx = 0
        self._loss_window: Deque[float] = deque(maxlen=window_size)
        self._best_avg: Optional[float] = None
        self._steps_without_improvement = 0

    @property
    def current_phase(self) -> CurriculumPhase:
        return self.phases[self.current_phase_idx]

    def step(self, total_loss: float) -> CurriculumPhase:
        """Update with current loss and return the active phase."""
        self._loss_window.append(total_loss)
        if len(self._loss_window) < self._loss_window.maxlen:
            return self.current_phase

        current_avg = sum(self._loss_window) / len(self._loss_window)
        if self._best_avg is None or current_avg < self._best_avg * (1.0 - self.min_improvement):
            self._best_avg = current_avg
            self._steps_without_improvement = 0
        else:
            self._steps_without_improvement += 1

        if self._steps_without_improvement >= self.patience:
            if self.current_phase_idx < len(self.phases) - 1:
                self.current_phase_idx += 1
                self._steps_without_improvement = 0
                self._best_avg = None
                self._loss_window.clear()

        return self.current_phase
