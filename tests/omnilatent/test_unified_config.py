from __future__ import annotations

import subprocess
import sys

import pytest

from omnilatent.config import OmniLatentConfig
from omnilatent.core.config import (
    AgentConfig,
    DataConfig,
    HookConfig,
    KBConfig,
    ModalityConfig,
    ModelConfig,
    OmniAgentConfig,
    TrainingConfig,
)


def test_core_config_import_does_not_import_mmwm_in_fresh_process() -> None:
    code = """
import sys
import omnilatent.core.config
print(any(name == 'MMWM' or name.startswith('MMWM.') for name in sys.modules))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "False"


@pytest.mark.parametrize(
    ("preset", "hidden_dim", "layers", "heads", "max_seq_len"),
    [
        ("tiny_debug", 256, 4, 4, 512),
        ("small_real", 512, 8, 8, 1024),
        ("default", 512, 8, 8, 1024),
        ("research_large_optional", 768, 12, 12, 2048),
    ],
)
def test_unified_config_presets_validate(
    preset: str,
    hidden_dim: int,
    layers: int,
    heads: int,
    max_seq_len: int,
) -> None:
    cfg = OmniAgentConfig.from_preset(preset)

    assert cfg.model.hidden_dim == hidden_dim
    assert cfg.model.layers == layers
    assert cfg.model.heads == heads
    assert cfg.model.max_seq_len == max_seq_len
    assert cfg.validate() is cfg


def test_unified_config_sections_are_typed() -> None:
    cfg = OmniAgentConfig.tiny_debug()

    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.modalities, ModalityConfig)
    assert isinstance(cfg.hooks, HookConfig)
    assert isinstance(cfg.agent, AgentConfig)
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.kb, KBConfig)


def test_unified_config_validation_catches_divisibility() -> None:
    cfg = OmniAgentConfig.tiny_debug()
    cfg.model.hidden_dim = 250

    with pytest.raises(ValueError, match="model.hidden_dim"):
        cfg.validate()


def test_unified_config_validation_catches_patch_mismatch() -> None:
    cfg = OmniAgentConfig.tiny_debug()
    cfg.modalities.image_size = 111

    with pytest.raises(ValueError, match="modalities.image_size"):
        cfg.validate()


def test_unified_config_validation_catches_bad_data_source() -> None:
    cfg = OmniAgentConfig.tiny_debug()
    cfg.data.sources = [{"name": "missing_type"}]

    with pytest.raises(ValueError, match="data.sources"):
        cfg.validate()


def test_unified_config_json_roundtrip(tmp_path) -> None:
    cfg = OmniAgentConfig.tiny_debug()
    cfg.data.sources = [
        {
            "name": "local_text",
            "type": "local",
            "path": "./README.md",
            "max_samples_per_epoch": 8,
        }
    ]
    path = tmp_path / "config.json"

    cfg.to_json(path)
    restored = OmniAgentConfig.from_json(path)

    assert restored.to_dict() == cfg.to_dict()


def test_unified_config_yaml_roundtrip(tmp_path) -> None:
    cfg = OmniAgentConfig.small_real()
    path = tmp_path / "config.yaml"

    cfg.to_yaml(path)
    restored = OmniAgentConfig.from_yaml(path)

    assert restored.to_dict() == cfg.to_dict()


def test_unified_config_from_dict_ignores_unknown_fields() -> None:
    payload = OmniAgentConfig.tiny_debug().to_dict()
    payload["unknown_future_top_level"] = True
    payload["model"]["unknown_future_model_field"] = 123

    restored = OmniAgentConfig.from_dict(payload)

    assert restored.model.hidden_dim == 256
    assert "unknown_future_top_level" not in restored.to_dict()


def test_unified_config_to_legacy_omnilatent_config() -> None:
    cfg = OmniAgentConfig.tiny_debug()
    legacy = cfg.to_omnilatent_config()

    assert isinstance(legacy, OmniLatentConfig)
    assert legacy.hidden_dim == cfg.model.hidden_dim
    assert legacy.num_layers == cfg.model.layers
    assert legacy.num_heads == cfg.model.heads
    assert legacy.max_seq_len == cfg.model.max_seq_len
    assert legacy.image_size == cfg.modalities.image_size
    assert legacy.video_max_frames == cfg.modalities.video_max_frames
    assert legacy.reasoning_num_thoughts == cfg.model.reasoning_tokens
    assert legacy.batch_size == cfg.training.batch_size


def test_unified_config_to_mmwm_config_dimension_conventions() -> None:
    from MMWM.config import ModelConfig as MMWMModelConfig
    from MMWM.config import _validate_dims

    cfg = OmniAgentConfig.tiny_debug()
    mmwm = cfg.to_mmwm_config()

    assert isinstance(mmwm, MMWMModelConfig)
    _validate_dims(mmwm)
    assert mmwm.encoder_kwargs["hidden_dim"] == cfg.model.hidden_dim
    assert mmwm.latent_projector_kwargs["latent_dim"] == cfg.model.role_latent_dim
    assert mmwm.conditioner_kwargs["latent_dim"] == cfg.model.role_latent_dim * 4
    assert mmwm.prediction_head_kwargs["latent_dim"] == cfg.model.role_latent_dim

