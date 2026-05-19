"""Forgetting-mitigation primitives.

Phase 3 ships the cheapest two layers of the defense-in-depth stack
described in ``docs/self_improvement.md`` §4: experience replay
(DER++-style) and EMA teacher distillation. Later phases add EWC + SI
(§4.3), parameter-isolation via hooks (§4.4), A-GEM (§4.5), and the
rollback gate (§4.6, already in phase 2's vault).
"""

from scripts.training.self_improve.forgetting.ema import EMATeacher
from scripts.training.self_improve.forgetting.replay import (
    ReplayBank,
    ReplayItem,
)

__all__ = ["EMATeacher", "ReplayBank", "ReplayItem"]
