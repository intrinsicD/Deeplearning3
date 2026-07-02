"""W0.8 — refuse to silently train on a frozen random DINO (A8)."""

from __future__ import annotations

import builtins
import inspect

import pytest
import torch

from hpwm.model import DINOBackbone


def _force_pretrained_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "timm":
            raise ImportError("forced timm failure")
        return orig_import(name, *args, **kwargs)

    def fake_hub_load(*args, **kwargs):
        raise RuntimeError("forced torch.hub failure")

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(torch.hub, "load", fake_hub_load)


def test_random_fallback_raises_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_pretrained_unavailable(monkeypatch)
    dino = DINOBackbone(allow_random_fallback=False)
    with pytest.raises(RuntimeError, match="random"):
        dino._load_model(torch.device("cpu"))


def test_random_fallback_allowed_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_pretrained_unavailable(monkeypatch)
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
