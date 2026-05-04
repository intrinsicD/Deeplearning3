from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from omnilatent.config import OmniLatentConfig
from omnilatent.core.config import OmniAgentConfig
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.protocol import ObservationPacket, TargetSpec


def _tiny_config() -> OmniLatentConfig:
    return OmniLatentConfig(
        hidden_dim=64,
        num_layers=4,
        num_heads=4,
        gradient_checkpointing=False,
        vocab_size=256,
        image_size=32,
        image_patch_size=8,
        video_size=32,
        video_patch_size=8,
        video_temporal_patch=2,
        video_max_frames=4,
        audio_n_mels=32,
        audio_max_frames=64,
        batch_size=2,
    )


def test_tiny_model_runs_cpu_text_and_multisource_forward() -> None:
    cfg = _tiny_config()
    model = OmniLatentModel(cfg)
    text = torch.randint(1, cfg.vocab_size, (2, 8))
    image = torch.randn(2, 3, cfg.image_size, cfg.image_size)

    recon = model.reconstruct("text", text)
    fused = model.forward_observation(
        ObservationPacket(modalities={"text": text, "image": image}),
        TargetSpec(modality="text", data=text),
    )

    assert recon["output"].shape == (2, 8, cfg.vocab_size)
    assert fused["output"].shape == (2, 8, cfg.vocab_size)
    assert not torch.isnan(fused["output"]).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_small_real_model_instantiates_on_gpu() -> None:
    cfg = OmniAgentConfig.small_real().to_omnilatent_config()
    model = OmniLatentModel(cfg).cuda()

    assert next(model.parameters()).is_cuda

    del model
    torch.cuda.empty_cache()


def test_one_tiny_training_step_cpu() -> None:
    cfg = _tiny_config()
    model = OmniLatentModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    text = torch.randint(1, cfg.vocab_size, (2, 8))

    before = next(model.parameters()).detach().clone()
    result = model.reconstruct("text", text)
    logits = result["output"]
    loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), text.reshape(-1), ignore_index=cfg.text_pad_token)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    after = next(model.parameters()).detach()
    assert torch.isfinite(loss)
    assert not torch.equal(before, after)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_one_tiny_training_step_under_8gb_cuda_budget() -> None:
    cfg = _tiny_config()
    model = OmniLatentModel(cfg).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    text = torch.randint(1, cfg.vocab_size, (2, 8), device="cuda")
    torch.cuda.reset_peak_memory_stats()

    result = model.reconstruct("text", text)
    loss = F.cross_entropy(result["output"].reshape(-1, cfg.vocab_size), text.reshape(-1), ignore_index=cfg.text_pad_token)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    peak = torch.cuda.max_memory_allocated()

    assert torch.isfinite(loss)
    assert peak < 8 * 1024**3

    del model, optimizer, text, result, loss
    torch.cuda.empty_cache()

