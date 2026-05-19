"""Phase 4 acceptance test: EWC beats replay-only on a two-domain toy.

Design doc verification line for phase 4:
    "Same two-domain test [as phase 3], this time with EWC; should
     outperform replay-only on the OmniLatent backbone."

We use OmniLatent's image→image autoencoding head as the test surface.
Domain A and B are images with different intensity centers; the
forgetting signal is "what's the reconstruction MSE on domain-A images
after training on B for a while?"

Three runs are compared:

- ``baseline``: train A then B with no forgetting defenses.
- ``replay``: train A then B with DER++ replay attached.
- ``ewc``: train A then B with replay + EWC + SI attached. EWC's anchor
  is snapshotted at the A→B transition so the regularizer pulls weights
  back toward the post-A solution.

The assertion is that ``ewc`` retains domain A better than ``replay``
alone (both are better than ``baseline``).
"""

from __future__ import annotations

import pytest
import torch

from scripts.training.self_improve import ReplayBank
from scripts.training.self_improve.plugins import get_plugin


def _make_domain(
    seed: int, center: float, n: int = 100, batch: int = 2, *, half: float = 0.05,
) -> list[dict[str, torch.Tensor]]:
    """Image-only mini-batches centered around ``center`` (width ``2·half``).

    Defaults to **narrow non-overlapping** distributions (half=0.05) so
    that the two-domain training actually exhibits catastrophic
    forgetting on this small model. Wider domains (half ≥ 0.2) tend to
    let the model generalize from B to A and never show real
    forgetting; the forgetting-mitigation comparison is then vacuous.
    """
    g = torch.Generator().manual_seed(seed)
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return [
        {"image": lo + (hi - lo) * torch.rand(batch, 3, 32, 32, generator=g)}
        for _ in range(n)
    ]


@torch.no_grad()
def _eval_image_mse(plugin, batches: list[dict[str, torch.Tensor]]) -> float:
    model = plugin.model
    was_training = model.training
    model.eval()
    try:
        total = 0.0
        n = 0
        for b in batches:
            image = b["image"].to(plugin.device)
            result = model.reconstruct("image", image)
            total += float(torch.nn.functional.mse_loss(result["output"], image).item()) * image.shape[0]
            n += image.shape[0]
        return total / max(n, 1)
    finally:
        model.train(was_training)


def _train_two_domain(
    *,
    mode: str,
    steps_per_domain: int = 100,
) -> float:
    """Run the two-domain protocol on OmniLatent, return domain-A MSE
    after the full A→B schedule.

    ``mode`` is one of ``"baseline"``, ``"replay"``, ``"ewc"``.
    """
    torch.manual_seed(0)
    OmniLatentPlugin = get_plugin("omnilatent")
    plugin = OmniLatentPlugin()

    if mode in ("replay", "ewc"):
        bank = ReplayBank(capacity=256, seed=0)
        plugin.attach_replay(bank, weight=1.0, batch_size=1)
    if mode == "ewc":
        # lam_ewc=1e4 is the empirical sweet spot for OmniLatent's tiny
        # config: large enough to bite, small enough to not block the B
        # learning entirely. lam_si=0 isolates the EWC effect for this
        # test; SI is exercised by ``test_ewc.py``.
        plugin.attach_ewc(fisher_decay=0.9, lam_ewc=1e4, lam_si=0.0)

    domain_a = _make_domain(seed=1, center=0.05, n=steps_per_domain)
    domain_b = _make_domain(seed=2, center=0.95, n=steps_per_domain)

    # Train on A.
    for batch in domain_a:
        plugin.train_step(batch)

    # Snapshot anchor at the task boundary if EWC is on. This is what
    # the design doc §4.3 calls "the anchor θ* is the most recent EMA
    # snapshot"; here we use the post-A weights directly (no EMA in this
    # test, to isolate the EWC effect).
    if mode == "ewc" and plugin.ewc is not None:
        plugin.ewc.snapshot_anchor(plugin.model)

    # Train on B.
    for batch in domain_b:
        plugin.train_step(batch)

    eval_a = _make_domain(seed=101, center=0.05, n=4)
    return _eval_image_mse(plugin, eval_a)


def test_baseline_exhibits_catastrophic_forgetting() -> None:
    """Sanity check: with no defenses, domain A's reconstruction loss
    blows up by ≥10x after training on B. If this stops being true,
    the EWC-vs-replay comparison below is vacuous.
    """
    # Train on A only, measure domain-A loss.
    torch.manual_seed(0)
    OmniLatentPlugin = get_plugin("omnilatent")
    plugin = OmniLatentPlugin()
    for batch in _make_domain(seed=1, center=0.05, n=100):
        plugin.train_step(batch)
    eval_a = _make_domain(seed=101, center=0.05, n=4)
    post_a = _eval_image_mse(plugin, eval_a)

    # Continue training on B without defenses.
    for batch in _make_domain(seed=2, center=0.95, n=100):
        plugin.train_step(batch)
    post_b = _eval_image_mse(plugin, eval_a)

    assert post_b > 10 * post_a, (
        f"no catastrophic forgetting on this toy: post_a={post_a:.4f} "
        f"post_b={post_b:.4f}; the EWC-vs-replay comparison would be "
        "meaningless. Tighten the domains."
    )


def test_ewc_outperforms_replay_only_on_two_domain() -> None:
    """The headline phase-4 acceptance test.

    With both replay and EWC attached (and replay alone as a control),
    domain-A reconstruction loss after the A→B schedule is better with
    EWC than without.
    """
    a_baseline = _train_two_domain(mode="baseline")
    a_replay = _train_two_domain(mode="replay")
    a_ewc = _train_two_domain(mode="ewc")

    # Replay alone should already cut forgetting drastically (the
    # phase-3 claim, transitively re-asserted here on OmniLatent).
    assert a_replay < a_baseline / 5, (
        f"replay alone failed to mitigate forgetting: replay={a_replay:.4f} "
        f"baseline={a_baseline:.4f}"
    )
    # The headline phase-4 claim: EWC tightens the retention further.
    assert a_ewc < a_replay, (
        f"EWC did not outperform replay-only: ewc={a_ewc:.4f} "
        f"replay={a_replay:.4f}"
    )
