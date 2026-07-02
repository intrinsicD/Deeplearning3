from __future__ import annotations

import warnings

import pytest
import torch

from omnilatent.config import OmniLatentConfig
from omnilatent.training.data import SyntheticMultiModalDataset, build_dataloader
from omnilatent.training.losses import ReconstructionLoss


def test_build_dataloader_allows_explicit_num_workers_zero() -> None:
    cfg = OmniLatentConfig(batch_size=1)
    ds = SyntheticMultiModalDataset(cfg, length=2, modalities=["text"])

    loader = build_dataloader(cfg, ds, num_workers=0)

    assert loader.num_workers == 0


def test_build_dataloader_reads_worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = OmniLatentConfig(batch_size=1)
    ds = SyntheticMultiModalDataset(cfg, length=2, modalities=["text"])

    monkeypatch.setenv("OMNILATENT_NUM_WORKERS", "0")
    loader = build_dataloader(cfg, ds)

    assert loader.num_workers == 0


def test_build_dataloader_rejects_invalid_worker_env(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = OmniLatentConfig(batch_size=1)
    ds = SyntheticMultiModalDataset(cfg, length=2, modalities=["text"])

    monkeypatch.setenv("OMNILATENT_NUM_WORKERS", "many")
    with pytest.raises(ValueError, match="OMNILATENT_NUM_WORKERS"):
        build_dataloader(cfg, ds)


def test_training_startup_main_includes_mmwm(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.diagnostics import check_training_startup as startup

    calls: list[str] = []
    monkeypatch.setattr(
        startup,
        "test_omnilatent",
        lambda steps=3, num_workers=0: calls.append(f"omni:{steps}:{num_workers}"),
    )
    monkeypatch.setattr(startup, "test_mmwm", lambda steps=2: calls.append(f"mmwm:{steps}"))
    monkeypatch.setattr(startup, "test_hpwm", lambda steps=2: calls.append(f"hpwm:{steps}"))
    monkeypatch.setattr(
        startup,
        "test_lgq_variant",
        lambda quantizer, steps=3: calls.append(f"lgq:{quantizer}:{steps}"),
    )
    monkeypatch.setattr(
        startup,
        "test_gaussian_encoder",
        lambda epochs=1: calls.append(f"gaussian:{epochs}"),
    )
    monkeypatch.setattr(startup, "_print_summary", lambda verbose: 0)
    monkeypatch.setattr(
        startup.sys,
        "argv",
        ["check_training_startup", "--steps", "5", "--epochs", "2", "--num-workers", "0"],
    )

    with pytest.raises(SystemExit) as exc:
        startup.main()

    assert exc.value.code == 0
    assert calls == [
        "omni:5:0",
        "mmwm:5",
        "hpwm:5",
        "lgq:lgq:5",
        "lgq:fsq:5",
        "lgq:simvq:5",
        "gaussian:2",
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA autocast")
def test_image_frequency_loss_avoids_complex_half_warning() -> None:
    loss_fn = ReconstructionLoss()
    pred = torch.randn(1, 3, 16, 16, device="cuda", dtype=torch.float16, requires_grad=True)
    target = torch.randn(1, 3, 16, 16, device="cuda", dtype=torch.float16)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = loss_fn.image_loss(pred, target)
        loss.backward()

    messages = [str(w.message) for w in caught]
    assert not any("ComplexHalf support is experimental" in msg for msg in messages)
    assert torch.isfinite(loss.detach())
