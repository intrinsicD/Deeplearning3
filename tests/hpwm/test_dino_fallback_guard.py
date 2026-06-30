"""W0.8 — refuse to silently train on a frozen random DINO (A8)."""

from __future__ import annotations

import inspect

import pytest
import torch

from hpwm.model import DINOBackbone


def _pretrained_unavailable() -> bool:
    try:
        import timm  # noqa: F401
        return False
    except ImportError:
        return True


@pytest.mark.skipif(
    not _pretrained_unavailable(),
    reason="pretrained DINO can load here; cannot exercise the fallback guard",
)
def test_random_fallback_raises_by_default() -> None:
    dino = DINOBackbone(allow_random_fallback=False)
    with pytest.raises(RuntimeError, match="random"):
        dino._load_model(torch.device("cpu"))


def test_random_fallback_allowed_when_opted_in() -> None:
    dino = DINOBackbone(allow_random_fallback=True)
    with pytest.warns(RuntimeWarning):
        dino._load_model(torch.device("cpu"))
    assert dino._model is not None


def test_trainer_threads_ssv2_dir_into_dataloaders() -> None:
    # The Trainer must forward its data_dir to create_dataloaders so --ssv2-dir
    # actually selects the dataset path (Audit.md A8).
    import hpwm.train as train_mod

    sig = inspect.signature(train_mod.Trainer.__init__)
    assert "data_dir" in sig.parameters

    src = inspect.getsource(train_mod.Trainer.__init__)
    assert "data_dir=data_dir" in src or "data_dir=self.data_dir" in src
