"""W0.10 — Phase-0 loop-integrity gate.

A single test module that asserts the four exit criteria for "the loop is
correct" from docs/work_plan.md Phase 0:

  (a) every learnable loss log_var receives gradients AND is in the optimizer;
  (b) a mixed real batch reaches the model with >=2 modalities and yields a
      finite, non-zero loss (never a fake zero-loss step);
  (c) a corrupt media file raises MediaDecodeError rather than returning zeros;
  (d) generate() runs the hook-aware backbone path, so registered Latent Neural
      Hooks materially change the hidden states it produces.

These tie together W0.1-W0.6. If any regress, this gate fails.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from omnilatent.config import OmniLatentConfig
from omnilatent.data import MediaDecodeError, MultiModalSample, build_sample_collator
from omnilatent.data.sources.local import _sample_from_path
from omnilatent.model.hooks import LatentNeuralHook
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.trainer import Trainer
from omnilatent.utils import ALL_MODALITIES


def _cfg() -> OmniLatentConfig:
    return OmniLatentConfig(hidden_dim=64, num_layers=2, num_heads=4)


class _EmptyDS(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, i):
        return {}


def _trainer() -> Trainer:
    cfg = _cfg()
    return Trainer(OmniLatentModel(cfg), cfg, DataLoader(_EmptyDS()))


# (a) ---------------------------------------------------------------------
def test_all_loss_log_vars_optimized_and_get_gradients() -> None:
    tr = _trainer()
    model, cfg = tr.model, tr.config

    # In the optimizer?
    opt_ids = {id(p) for g in tr.optimizer.param_groups for p in g["params"]}
    log_vars = tr.criterion.log_vars
    assert set(log_vars.keys()) == set(ALL_MODALITIES)
    assert all(id(p) in opt_ids for p in log_vars.parameters())

    # Receive gradients: self-reconstruct every modality, one combined backward.
    model.train()
    device = tr.device
    data = {
        "text": torch.randint(1, cfg.vocab_size, (2, 8), device=device),
        "image": torch.randn(2, 3, cfg.image_size, cfg.image_size, device=device),
        "audio": torch.randn(2, cfg.audio_n_mels, 64, device=device),
        "video": torch.randn(
            2,
            3,
            cfg.video_max_frames,
            cfg.video_size,
            cfg.video_size,
            device=device,
        ),
    }
    predictions, targets = {}, {}
    for mod, x in data.items():
        predictions[mod] = model.reconstruct(mod, x)["output"]
        targets[mod] = x
    tr.optimizer.zero_grad(set_to_none=True)
    loss = tr.criterion(predictions, targets)["total"]
    loss.backward()

    for mod in ALL_MODALITIES:
        assert log_vars[mod].grad is not None, f"{mod} log_var got no gradient"


# (b) ---------------------------------------------------------------------
def test_mixed_real_batch_reaches_model_with_nonzero_loss() -> None:
    tr = _trainer()
    cfg = tr.config
    # One paired sample carrying two modalities → aligned batch, both present.
    sample = MultiModalSample(
        text="a caption about a red square",
        image=torch.rand(3, 40, 50),
    )
    batch = build_sample_collator(cfg)([sample])
    assert len(batch) >= 2, "mixed batch must preserve >=2 modalities"

    result = tr._train_step(batch)
    assert "skipped" not in result            # not a fake zero-loss step
    assert "total" in result
    assert math.isfinite(result["total"])
    assert result["total"] > 0.0


# (c) ---------------------------------------------------------------------
def test_corrupt_media_raises(tmp_path: Path) -> None:
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"definitely not a wav")
    with pytest.raises(MediaDecodeError):
        _sample_from_path(bad)


# (d) ---------------------------------------------------------------------
def test_generate_hidden_states_depend_on_hooks() -> None:
    cfg = _cfg()
    model = OmniLatentModel(cfg).eval()
    # A strongly-gated hook so its effect on hidden states is unambiguous.
    model.register_hook(
        LatentNeuralHook(
            name="gate_high",
            num_tokens=4,
            dim=cfg.hidden_dim,
            target_layers=[0, 1],
            gate_bias_init=4.0,  # sigmoid(4) ~ 0.98: strong influence
        )
    )

    captured: list[torch.Tensor] = []
    orig = model.backbone.forward

    def spy(*args, **kwargs):
        out = orig(*args, **kwargs)
        captured.append(out.detach().clone())
        return out

    model.backbone.forward = spy

    image = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    torch.manual_seed(0)
    model.generate("image", image, max_len=1)
    with_hook = captured[0]

    captured.clear()
    model.remove_hook("gate_high")
    torch.manual_seed(0)
    model.generate("image", image, max_len=1)
    without_hook = captured[0]

    # The hook participates in attention, so it adds tokens to the sequence:
    # the hidden states are not even the same shape, and certainly not equal.
    assert with_hook.shape[1] != without_hook.shape[1] or not torch.allclose(
        with_hook, without_hook
    ), "generate() ignored the registered hook (Audit.md A9)"
