from __future__ import annotations

import types

import torch

from omnilatent.config import OmniLatentConfig
from omnilatent.data.collate import decode_eos_byte_tokens, eos_byte_tokenize
from omnilatent.model.decoders import ImageDecoder
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.training.losses import ReconstructionLoss


def test_eos_byte_tokenize_appends_eos_and_decodes() -> None:
    cfg = OmniLatentConfig(vocab_size=32000, text_max_len=8)

    tokens = eos_byte_tokenize(
        "abc",
        cfg.text_max_len,
        cfg.vocab_size,
        bos_token=cfg.text_bos_token,
        eos_token=cfg.text_eos_token,
    )

    assert tokens[-1].item() == cfg.text_eos_token
    assert tokens.min().item() >= cfg.text_eos_token
    decoded, oov = decode_eos_byte_tokens(
        tokens,
        bos_token=cfg.text_bos_token,
        eos_token=cfg.text_eos_token,
    )
    assert decoded == "abc"
    assert oov == 0


def test_generate_stops_on_eos_and_pads_to_max_len() -> None:
    cfg = OmniLatentConfig(
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        vocab_size=64,
        text_max_len=8,
        gradient_checkpointing=False,
    )
    model = OmniLatentModel(cfg).eval()

    def eos_decoder_forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x.shape[0], x.shape[1], cfg.vocab_size, device=x.device)
        logits[..., cfg.text_eos_token] = 1.0
        return logits

    model.decoders["text"].forward = types.MethodType(eos_decoder_forward, model.decoders["text"])
    source = torch.randint(3, cfg.vocab_size, (2, 5))

    out = model.generate("text", source, max_len=5)

    assert out.shape == (2, 5)
    assert torch.equal(out[:, 0], torch.full((2,), cfg.text_eos_token))
    assert torch.equal(out[:, 1:], torch.zeros(2, 4, dtype=torch.long))


def test_patch_image_decoder_outputs_image_range() -> None:
    cfg = OmniLatentConfig(
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        image_size=32,
        image_patch_size=16,
        image_decoder="patch",
    )
    decoder = ImageDecoder(cfg)
    x = torch.randn(2, cfg.image_num_patches, cfg.hidden_dim)

    out = decoder(x)

    assert out.shape == (2, cfg.image_channels, cfg.image_size, cfg.image_size)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_gaussian_image_decoder_outputs_image_range_and_gradients() -> None:
    cfg = OmniLatentConfig(
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        image_size=32,
        image_patch_size=16,
        image_decoder="gaussian",
        image_gaussians_per_token=4,
        image_gaussian_chunk_size=3,
    )
    decoder = ImageDecoder(cfg)
    x = torch.randn(2, cfg.image_num_patches, cfg.hidden_dim, requires_grad=True)

    out = decoder(x)
    out.mean().backward()

    assert out.shape == (2, cfg.image_channels, cfg.image_size, cfg.image_size)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
    assert decoder.gaussian_anchors.shape[1] == cfg.image_num_patches * cfg.image_gaussians_per_token
    assert torch.unique(decoder.gaussian_anchors[0, : cfg.image_gaussians_per_token], dim=0).shape[0] == 4
    assert decoder.gaussian_head.weight.grad is not None
    assert torch.isfinite(decoder.gaussian_head.weight.grad).all()


def test_image_edge_loss_is_opt_in() -> None:
    target = torch.zeros(1, 3, 16, 16)
    pred = target.clone()
    pred[:, :, 4:12, 4:12] = 1.0

    base = ReconstructionLoss(image_edge_weight=0.0).image_loss(pred, target)
    edge = ReconstructionLoss(image_edge_weight=0.25).image_loss(pred, target)

    assert edge.item() > base.item()
