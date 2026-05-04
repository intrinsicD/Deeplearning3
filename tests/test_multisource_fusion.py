from __future__ import annotations

import torch

from omnilatent.config import OmniLatentConfig
from omnilatent.model.masking import apply_token_validity_mask, build_prefix_lm_mask
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.protocol import ObservationPacket, TargetSpec


def _config() -> OmniLatentConfig:
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
    )


def test_forward_observation_fuses_text_and_image_in_one_prefix() -> None:
    cfg = _config()
    model = OmniLatentModel(cfg)
    text = torch.randint(1, cfg.vocab_size, (2, 5))
    image = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    target_text = torch.randint(1, cfg.vocab_size, (2, 6))

    result = model.forward_observation(
        ObservationPacket(modalities={"text": text, "image": image}),
        TargetSpec(modality="text", data=target_text),
    )

    text_len = 1 + text.shape[1]
    image_len = 1 + cfg.image_num_patches
    expected_prefix_len = text_len + image_len
    assert int(result["prefix_len"].item()) == expected_prefix_len
    assert result["output"].shape == (2, target_text.shape[1], cfg.vocab_size)
    assert result["source_modalities"] == ["text", "image"]
    assert result["source_lengths"] == [text_len, image_len]
    assert result["source_segment_ids"].shape == (2, expected_prefix_len)
    assert torch.equal(result["source_segment_ids"][:, :text_len], torch.zeros(2, text_len, dtype=torch.long))
    assert torch.equal(result["source_segment_ids"][:, text_len:], torch.ones(2, image_len, dtype=torch.long))


def test_forward_observation_applies_source_validity_masks() -> None:
    cfg = _config()
    model = OmniLatentModel(cfg)
    text = torch.randint(1, cfg.vocab_size, (2, 5))
    image = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    text_mask = torch.tensor([
        [True, True, True, True, False],
        [True, True, False, False, False],
    ])

    result = model.forward_observation(
        ObservationPacket(
            modalities={"text": text, "image": image},
            masks={"text": text_mask},
        ),
        TargetSpec(modality="text", data=torch.randint(1, cfg.vocab_size, (2, 4))),
    )

    # Text mask is normalized by prepending a valid modality token.
    assert torch.equal(result["source_valid_mask"][:, 1:6], text_mask)
    attn_mask = result["attention_mask"]
    invalid_text_pos_batch0 = 5  # modality token + 5 text tokens, last token masked
    assert not attn_mask[0, 0, :, invalid_text_pos_batch0].any()
    assert not attn_mask[0, 0, invalid_text_pos_batch0, :].any()

    prefix_len = int(result["prefix_len"].item())
    # Valid source tokens cannot attend to target tokens in Prefix-LM mode.
    assert not attn_mask[:, 0, :prefix_len, prefix_len:].any()
    # Target tokens can read valid source tokens but not invalid source tokens.
    assert attn_mask[0, 0, prefix_len, 0]
    assert not attn_mask[0, 0, prefix_len, invalid_text_pos_batch0]


def test_text_target_region_is_causal_for_multisource_forward() -> None:
    cfg = _config()
    model = OmniLatentModel(cfg)
    result = model.forward_observation(
        ObservationPacket(modalities={"text": torch.randint(1, cfg.vocab_size, (1, 3))}),
        TargetSpec(modality="text", data=torch.randint(1, cfg.vocab_size, (1, 5))),
    )

    prefix_len = int(result["prefix_len"].item())
    target_region = result["attention_mask"][0, 0, prefix_len:, prefix_len:]
    expected = torch.tril(torch.ones_like(target_region, dtype=torch.bool))
    assert torch.equal(target_region, expected)


def test_non_text_target_region_is_bidirectional_for_multisource_forward() -> None:
    cfg = _config()
    model = OmniLatentModel(cfg)
    result = model.forward_observation(
        ObservationPacket(modalities={"text": torch.randint(1, cfg.vocab_size, (1, 3))}),
        TargetSpec(modality="image"),
    )

    prefix_len = int(result["prefix_len"].item())
    target_region = result["attention_mask"][0, 0, prefix_len:, prefix_len:]
    assert target_region.all()
    assert result["output"].shape == (1, 3, cfg.image_size, cfg.image_size)


def test_source_segment_embeddings_affect_multisource_output() -> None:
    cfg = _config()
    model = OmniLatentModel(cfg)
    model.eval()
    text = torch.randint(1, cfg.vocab_size, (1, 4))
    image = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    target = TargetSpec(modality="text", data=torch.randint(1, cfg.vocab_size, (1, 4)))
    packet = ObservationPacket(modalities={"text": text, "image": image})

    with torch.no_grad():
        model.source_segment_embed.weight.zero_()
        without_segments = model.forward_observation(packet, target)["output"]
        model.source_segment_embed.weight[1].fill_(0.25)
        with_segments = model.forward_observation(packet, target)["output"]

    assert not torch.allclose(without_segments, with_segments)


def test_apply_token_validity_mask_masks_invalid_queries_and_keys() -> None:
    base = build_prefix_lm_mask(src_len=3, tgt_len=2, target_modality="text", device=torch.device("cpu"))
    valid = torch.tensor([[True, False, True, True, True]])

    masked = apply_token_validity_mask(base, valid)

    assert masked.shape == (1, 1, 5, 5)
    assert not masked[0, 0, :, 1].any()
    assert not masked[0, 0, 1, :].any()
    assert not masked[0, 0, 0, 3]  # prefix cannot attend target
    assert masked[0, 0, 3, 0]      # target can attend valid prefix

