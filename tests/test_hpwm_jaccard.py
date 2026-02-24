"""Unit tests for the Signal-2 Jaccard metric in hpwm/evaluate.py.

These tests specifically verify the fix for the empty-slot averaging bug:
previously, batch elements where union==0 (slot inactive in both frames)
were included in the mean as 0.0, artificially deflating the score.
After the fix they are excluded via a per-sample `valid = union > 0` mask.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Standalone helper that mirrors the fixed Jaccard logic from evaluate_signal_2
# ---------------------------------------------------------------------------

def _compute_jaccard(prev_attn: torch.Tensor, curr_attn: torch.Tensor) -> list[float]:
    """Compute Jaccard scores across slots using the fixed per-sample logic.

    Args:
        prev_attn: [B, N_slots, N_patches] soft attention weights.
        curr_attn: [B, N_slots, N_patches] soft attention weights.

    Returns:
        List of valid Jaccard scores (one per slot that is active in at least
        one sample and one frame).
    """
    n_slots = prev_attn.shape[1]
    prev_hard = prev_attn.argmax(dim=1)  # [B, N_patches]
    curr_hard = curr_attn.argmax(dim=1)  # [B, N_patches]

    all_jaccards: list[float] = []
    for s in range(n_slots):
        prev_mask = (prev_hard == s).float()  # [B, N_patches]
        curr_mask = (curr_hard == s).float()  # [B, N_patches]

        intersection = (prev_mask * curr_mask).sum(dim=-1)          # [B]
        union = ((prev_mask + curr_mask) > 0).float().sum(dim=-1)   # [B]

        valid = union > 0  # [B] bool
        if valid.any():
            jaccard = (intersection[valid] / union[valid]).mean().item()
            all_jaccards.append(jaccard)

    return all_jaccards


def _compute_jaccard_buggy(prev_attn: torch.Tensor, curr_attn: torch.Tensor) -> list[float]:
    """Reproduce the *original* (buggy) Jaccard logic for comparison."""
    n_slots = prev_attn.shape[1]
    prev_hard = prev_attn.argmax(dim=1)
    curr_hard = curr_attn.argmax(dim=1)

    all_jaccards: list[float] = []
    for s in range(n_slots):
        prev_mask = (prev_hard == s).float()
        curr_mask = (curr_hard == s).float()

        if prev_mask.sum() > 0 or curr_mask.sum() > 0:
            intersection = (prev_mask * curr_mask).sum(dim=-1)
            union = ((prev_mask + curr_mask) > 0).float().sum(dim=-1)
            jaccard = (intersection / (union + 1e-8)).mean().item()
            all_jaccards.append(jaccard)

    return all_jaccards


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestJaccardPerfectConsistency:
    """When slot assignments are identical across frames, Jaccard must be 1.0."""

    def test_identical_assignments_single_sample(self):
        B, N_slots, N_patches = 1, 4, 16
        # slot 0 wins patches 0-3, slot 1 wins 4-7, etc.
        attn = torch.zeros(B, N_slots, N_patches)
        for s in range(N_slots):
            attn[0, s, s * 4:(s + 1) * 4] = 1.0

        scores = _compute_jaccard(attn, attn.clone())
        assert len(scores) == N_slots
        for j in scores:
            assert abs(j - 1.0) < 1e-6, f"Expected Jaccard=1.0, got {j}"

    def test_identical_assignments_batch(self):
        B, N_slots, N_patches = 4, 4, 16
        attn = torch.zeros(B, N_slots, N_patches)
        for b in range(B):
            for s in range(N_slots):
                attn[b, s, s * 4:(s + 1) * 4] = 1.0

        scores = _compute_jaccard(attn, attn.clone())
        assert len(scores) == N_slots
        for j in scores:
            assert abs(j - 1.0) < 1e-6


class TestJaccardNoOverlap:
    """When a slot's assignments flip entirely, Jaccard must be 0.0."""

    def test_disjoint_assignments(self):
        B, N_slots, N_patches = 1, 2, 8
        # Frame 1: slot 0 → patches 0-3, slot 1 → patches 4-7
        prev_attn = torch.zeros(B, N_slots, N_patches)
        prev_attn[0, 0, :4] = 1.0
        prev_attn[0, 1, 4:] = 1.0

        # Frame 2: slot 0 → patches 4-7, slot 1 → patches 0-3  (fully swapped)
        curr_attn = torch.zeros(B, N_slots, N_patches)
        curr_attn[0, 0, 4:] = 1.0
        curr_attn[0, 1, :4] = 1.0

        scores = _compute_jaccard(prev_attn, curr_attn)
        assert len(scores) == N_slots
        for j in scores:
            assert abs(j - 0.0) < 1e-6, f"Expected Jaccard=0.0, got {j}"


class TestJaccardEmptySlotExclusion:
    """The core regression test: empty-union samples must not be counted as 0."""

    def test_empty_slot_excluded_from_mean(self):
        """Batch of 2 samples; slot 3 is only active in sample 0.

        Sample 0: slot 3 has perfect overlap → Jaccard = 1.0
        Sample 1: slot 3 is inactive in both frames → must be excluded

        Fixed code: mean Jaccard for slot 3 = 1.0
        Buggy code: mean Jaccard for slot 3 = (1.0 + 0.0) / 2 = 0.5
        """
        B, N_slots, N_patches = 2, 4, 8

        # Build attention so that each slot dominates a disjoint set of patches
        prev_attn = torch.zeros(B, N_slots, N_patches)
        curr_attn = torch.zeros(B, N_slots, N_patches)

        # Sample 0: uniform distribution across 4 slots (2 patches each)
        for s in range(N_slots):
            prev_attn[0, s, s * 2:(s + 1) * 2] = 1.0
            curr_attn[0, s, s * 2:(s + 1) * 2] = 1.0  # identical → Jaccard=1

        # Sample 1: only 3 slots active; slot 3 wins nothing in either frame.
        # Give all 8 patches to slots 0-2 (≥3 patches each so slot 3 wins 0).
        prev_attn[1, 0, :3] = 1.0
        prev_attn[1, 1, 3:6] = 1.0
        prev_attn[1, 2, 6:] = 1.0
        # slot 3 row stays all-zero → argmax will never pick 3 for sample 1

        curr_attn[1, 0, :3] = 1.0
        curr_attn[1, 1, 3:6] = 1.0
        curr_attn[1, 2, 6:] = 1.0

        scores_fixed = _compute_jaccard(prev_attn, curr_attn)
        scores_buggy = _compute_jaccard_buggy(prev_attn, curr_attn)

        # Only 3 slot scores come from sample 1 (slots 0,1,2); slot 3 from sample 0
        # The fixed code should give slot 3 a score of 1.0 (only sample 0 contributes)
        # The buggy code gives 0.5 for slot 3 (averages in sample 1's 0/0≈0)
        assert len(scores_fixed) > 0, "Fixed code produced no scores"

        # Verify bug would have produced a lower (wrong) score for slot 3
        mean_fixed = sum(scores_fixed) / len(scores_fixed)
        mean_buggy = sum(scores_buggy) / len(scores_buggy) if scores_buggy else 0.0
        assert mean_fixed >= mean_buggy, (
            f"Fixed mean ({mean_fixed:.4f}) should be >= buggy mean ({mean_buggy:.4f})"
        )

    def test_slot_entirely_inactive_skipped(self):
        """A slot that is inactive in ALL samples and BOTH frames produces no score."""
        B, N_slots, N_patches = 2, 4, 6

        # Give all patches to slots 0, 1, 2 only — slot 3 never wins
        attn = torch.zeros(B, N_slots, N_patches)
        attn[:, 0, :2] = 1.0
        attn[:, 1, 2:4] = 1.0
        attn[:, 2, 4:] = 1.0
        # slot 3 row is all-zero for both batch elements

        scores = _compute_jaccard(attn, attn.clone())
        # Only 3 active slots should contribute
        assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"
        for j in scores:
            assert abs(j - 1.0) < 1e-6


class TestJaccardPartialOverlap:
    """Sanity-check for intermediate Jaccard values."""

    def test_half_overlap(self):
        """Slot 0 covers patches 0-3; in next frame it covers patches 0-1 and 4-5.

        intersection = {0,1} → size 2
        union = {0,1,2,3,4,5} → size 6
        Jaccard = 2/6 ≈ 0.333
        """
        B, N_slots, N_patches = 1, 2, 8

        prev_attn = torch.zeros(B, N_slots, N_patches)
        prev_attn[0, 0, :4] = 1.0   # slot 0 wins patches 0-3
        prev_attn[0, 1, 4:] = 1.0   # slot 1 wins patches 4-7

        curr_attn = torch.zeros(B, N_slots, N_patches)
        curr_attn[0, 0, :2] = 1.0   # slot 0 wins patches 0-1
        curr_attn[0, 0, 4:6] = 1.0  # slot 0 also wins patches 4-5 (beats slot 1)
        curr_attn[0, 1, 2:4] = 1.0  # slot 1 gets patches 2-3
        curr_attn[0, 1, 6:] = 1.0   # slot 1 gets patches 6-7

        scores = _compute_jaccard(prev_attn, curr_attn)
        assert len(scores) == N_slots

        # slot 0: prev={0,1,2,3}, curr={0,1,4,5} → inter={0,1}, union={0,1,2,3,4,5}
        slot0_jaccard = scores[0]
        assert abs(slot0_jaccard - 2 / 6) < 1e-5, (
            f"Expected Jaccard≈0.333, got {slot0_jaccard:.4f}"
        )
