"""Curriculum schedule definitions for staged training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CurriculumPhase:
    phase_id: int
    until_step: int
    task_multipliers: Dict[str, float]


def default_curriculum_phases() -> List[CurriculumPhase]:
    """Five-phase curriculum adapted for MMWM training."""
    return [
        CurriculumPhase(phase_id=1, until_step=1_000, task_multipliers={
            "latent_sem_loss": 1.0,
            "latent_dyn_loss": 0.5,
            "latent_ctrl_loss": 0.0,
            "latent_mem_loss": 0.0,
            "regularizer_loss": 1.0,
            "text_ce_loss": 0.0,
            "vector_recon_loss": 0.0,
        }),
        CurriculumPhase(phase_id=2, until_step=3_000, task_multipliers={
            "latent_sem_loss": 1.0,
            "latent_dyn_loss": 1.0,
            "latent_ctrl_loss": 0.5,
            "latent_mem_loss": 0.25,
            "regularizer_loss": 1.0,
            "text_ce_loss": 0.0,
            "vector_recon_loss": 0.0,
        }),
        CurriculumPhase(phase_id=3, until_step=6_000, task_multipliers={
            "latent_sem_loss": 1.0,
            "latent_dyn_loss": 1.0,
            "latent_ctrl_loss": 1.0,
            "latent_mem_loss": 0.75,
            "regularizer_loss": 1.0,
            "text_ce_loss": 0.25,
            "vector_recon_loss": 0.25,
        }),
        CurriculumPhase(phase_id=4, until_step=10_000, task_multipliers={
            "latent_sem_loss": 1.0,
            "latent_dyn_loss": 1.0,
            "latent_ctrl_loss": 1.0,
            "latent_mem_loss": 1.0,
            "regularizer_loss": 1.0,
            "text_ce_loss": 0.75,
            "vector_recon_loss": 0.75,
        }),
        CurriculumPhase(phase_id=5, until_step=1_000_000_000, task_multipliers={
            "latent_sem_loss": 1.0,
            "latent_dyn_loss": 1.0,
            "latent_ctrl_loss": 1.0,
            "latent_mem_loss": 1.0,
            "regularizer_loss": 1.0,
            "text_ce_loss": 1.0,
            "vector_recon_loss": 1.0,
        }),
    ]
