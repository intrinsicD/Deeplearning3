"""Unified configuration tree for the OmniLatent Agent Kernel.

This module is additive and does not replace the legacy ``omnilatent.config`` or
``MMWM.config`` modules. It provides one typed config tree for the target
agentic architecture, plus lazy adapters back to existing configs so migration
can happen incrementally.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Mapping, TypeVar, cast


T = TypeVar("T")


@dataclass
class ModelConfig:
    """Small latent transformer kernel configuration."""

    hidden_dim: int = 512
    layers: int = 8
    heads: int = 8
    max_seq_len: int = 1024
    mlp_ratio: float = 2.667
    dropout: float = 0.0
    gradient_checkpointing: bool = True
    reasoning_tokens: int = 8
    reasoning_layers: int = 2
    reasoning_heads: int = 8
    role_latent_dim: int = 128

    def validate(self) -> None:
        _require_positive_int("model.hidden_dim", self.hidden_dim)
        _require_positive_int("model.layers", self.layers)
        _require_positive_int("model.heads", self.heads)
        _require_positive_int("model.max_seq_len", self.max_seq_len)
        _require_positive_int("model.reasoning_tokens", self.reasoning_tokens, allow_zero=True)
        _require_positive_int("model.reasoning_layers", self.reasoning_layers, allow_zero=True)
        _require_positive_int("model.reasoning_heads", self.reasoning_heads)
        _require_positive_int("model.role_latent_dim", self.role_latent_dim)
        if self.hidden_dim % self.heads != 0:
            raise ValueError(
                f"model.hidden_dim ({self.hidden_dim}) must be divisible by model.heads ({self.heads})"
            )
        if self.hidden_dim % self.reasoning_heads != 0:
            raise ValueError(
                "model.hidden_dim "
                f"({self.hidden_dim}) must be divisible by model.reasoning_heads ({self.reasoning_heads})"
            )
        _require_range("model.dropout", self.dropout, min_value=0.0, max_value=1.0)
        if self.mlp_ratio <= 0:
            raise ValueError("model.mlp_ratio must be > 0")


@dataclass
class ModalityConfig:
    """Tokenizer/adapter settings for observed and target modalities."""

    vocab_size: int = 32_000
    text_max_len: int = 512
    text_pad_token: int = 0
    text_bos_token: int = 1
    audio_n_mels: int = 128
    audio_hop_length: int = 160
    audio_max_frames: int = 1000
    audio_patch_frames: int = 4
    audio_sample_rate: int = 16_000
    image_size: int = 224
    image_patch_size: int = 16
    image_channels: int = 3
    video_size: int = 112
    video_patch_size: int = 16
    video_temporal_patch: int = 4
    video_max_frames: int = 16
    video_channels: int = 3
    enabled: List[str] = field(default_factory=lambda: ["text", "image", "audio", "video", "vector"])

    def validate(self) -> None:
        for name in (
            "vocab_size",
            "text_max_len",
            "audio_n_mels",
            "audio_hop_length",
            "audio_max_frames",
            "audio_patch_frames",
            "audio_sample_rate",
            "image_size",
            "image_patch_size",
            "image_channels",
            "video_size",
            "video_patch_size",
            "video_temporal_patch",
            "video_max_frames",
            "video_channels",
        ):
            _require_positive_int(f"modalities.{name}", getattr(self, name))
        if self.image_size % self.image_patch_size != 0:
            raise ValueError(
                "modalities.image_size "
                f"({self.image_size}) must be divisible by modalities.image_patch_size ({self.image_patch_size})"
            )
        if self.video_size % self.video_patch_size != 0:
            raise ValueError(
                "modalities.video_size "
                f"({self.video_size}) must be divisible by modalities.video_patch_size ({self.video_patch_size})"
            )
        if self.video_max_frames % self.video_temporal_patch != 0:
            raise ValueError(
                "modalities.video_max_frames "
                f"({self.video_max_frames}) must be divisible by modalities.video_temporal_patch "
                f"({self.video_temporal_patch})"
            )
        if self.audio_max_frames % self.audio_patch_frames != 0:
            raise ValueError(
                "modalities.audio_max_frames "
                f"({self.audio_max_frames}) must be divisible by modalities.audio_patch_frames "
                f"({self.audio_patch_frames})"
            )
        if not self.enabled:
            raise ValueError("modalities.enabled must contain at least one modality")


@dataclass
class HookConfig:
    """Typed neural-port/hook defaults."""

    enabled: bool = True
    hook_tokens_per_extension: int = 8
    max_extensions: int = 8
    gate_init: float = 0.0
    gate_bias_init: float = -4.0
    default_target_layers: List[int] = field(default_factory=list)
    freeze_core_by_default: bool = True
    log_gate_values: bool = True

    def validate(self, model: ModelConfig) -> None:
        _require_positive_int("hooks.hook_tokens_per_extension", self.hook_tokens_per_extension, allow_zero=True)
        _require_positive_int("hooks.max_extensions", self.max_extensions, allow_zero=True)
        for layer in self.default_target_layers:
            if layer < 0 or layer >= model.layers:
                raise ValueError(
                    f"hooks.default_target_layers contains {layer}, but valid layers are 0..{model.layers - 1}"
                )


@dataclass
class AgentConfig:
    """Explicit agent graph/runtime budget."""

    max_steps: int = 16
    max_think_steps: int = 4
    thought_tokens: int = 8
    expose_thought_metrics: bool = True
    decode_thoughts: bool = False
    log_gate_values: bool = True
    stop_threshold: float = 0.5
    allow_side_effects: bool = False

    def validate(self) -> None:
        _require_positive_int("agent.max_steps", self.max_steps)
        _require_positive_int("agent.max_think_steps", self.max_think_steps, allow_zero=True)
        _require_positive_int("agent.thought_tokens", self.thought_tokens, allow_zero=True)
        _require_range("agent.stop_threshold", self.stop_threshold, min_value=0.0, max_value=1.0)


@dataclass
class DataConfig:
    """Unified data-plane manifest settings."""

    sources: List[Dict[str, Any]] = field(default_factory=list)
    cache_dir: str = "./data/cache"
    streaming: bool = True
    max_samples_per_epoch: int | None = None
    num_workers: int = 2

    def validate(self) -> None:
        _require_positive_int("data.num_workers", self.num_workers, allow_zero=True)
        if self.max_samples_per_epoch is not None:
            _require_positive_int("data.max_samples_per_epoch", self.max_samples_per_epoch)
        for i, source in enumerate(self.sources):
            if "name" not in source or "type" not in source:
                raise ValueError(f"data.sources[{i}] must include 'name' and 'type'")


@dataclass
class TrainingConfig:
    """Training defaults shared by model, tools, and data curricula."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 2000
    max_steps: int = 100_000
    batch_size: int = 4
    mixed_precision: bool = True
    grad_clip: float = 0.5
    seed: int = 42
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.07
    learned_uncertainty: bool = False
    modality_loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "text": 1.0,
        "audio": 1.0,
        "image": 1.0,
        "video": 1.0,
    })

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("training.learning_rate must be > 0")
        if self.weight_decay < 0:
            raise ValueError("training.weight_decay must be >= 0")
        _require_positive_int("training.warmup_steps", self.warmup_steps, allow_zero=True)
        _require_positive_int("training.max_steps", self.max_steps)
        _require_positive_int("training.batch_size", self.batch_size)
        if self.grad_clip < 0:
            raise ValueError("training.grad_clip must be >= 0")
        if self.contrastive_weight < 0:
            raise ValueError("training.contrastive_weight must be >= 0")
        if self.contrastive_temperature <= 0:
            raise ValueError("training.contrastive_temperature must be > 0")
        for name, weight in self.modality_loss_weights.items():
            if weight < 0:
                raise ValueError(f"training.modality_loss_weights[{name!r}] must be >= 0")


@dataclass
class KBConfig:
    """Local knowledge-base bootstrap/retrieval settings."""

    root_dir: str = "./kb"
    chunks_path: str = "chunks.jsonl"
    triples_path: str = "triples.jsonl"
    embedding_dim: int = 512
    chunk_tokens: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    writable: bool = False
    indexed_extensions: List[str] = field(default_factory=lambda: [".txt", ".md", ".py", ".pdf"])

    def validate(self) -> None:
        _require_positive_int("kb.embedding_dim", self.embedding_dim)
        _require_positive_int("kb.chunk_tokens", self.chunk_tokens)
        _require_positive_int("kb.chunk_overlap", self.chunk_overlap, allow_zero=True)
        _require_positive_int("kb.top_k", self.top_k)
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("kb.chunk_overlap must be smaller than kb.chunk_tokens")
        if not self.indexed_extensions:
            raise ValueError("kb.indexed_extensions must not be empty")


@dataclass
class OmniAgentConfig:
    """Top-level canonical config for the agentic OmniLatent architecture."""

    model: ModelConfig = field(default_factory=ModelConfig)
    modalities: ModalityConfig = field(default_factory=ModalityConfig)
    hooks: HookConfig = field(default_factory=HookConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    kb: KBConfig = field(default_factory=KBConfig)
    name: str = "small_real"
    version: int = 1

    PRESETS: ClassVar[tuple[str, ...]] = ("tiny_debug", "small_real", "default", "research_large_optional")

    def validate(self) -> "OmniAgentConfig":
        self.model.validate()
        self.modalities.validate()
        self.hooks.validate(self.model)
        self.agent.validate()
        self.data.validate()
        self.training.validate()
        self.kb.validate()
        if self.model.max_seq_len < self.modalities.text_max_len:
            raise ValueError(
                "model.max_seq_len must be >= modalities.text_max_len "
                f"({self.model.max_seq_len} < {self.modalities.text_max_len})"
            )
        if self.agent.thought_tokens != self.model.reasoning_tokens:
            raise ValueError(
                "agent.thought_tokens must match model.reasoning_tokens until the THINK node "
                "supports independent budgets"
            )
        return self

    @classmethod
    def tiny_debug(cls) -> "OmniAgentConfig":
        return cls(
            name="tiny_debug",
            model=ModelConfig(
                hidden_dim=256,
                layers=4,
                heads=4,
                max_seq_len=512,
                reasoning_tokens=4,
                reasoning_layers=1,
                reasoning_heads=4,
                role_latent_dim=64,
            ),
            modalities=ModalityConfig(
                text_max_len=256,
                image_size=112,
                video_size=64,
                video_max_frames=8,
                audio_max_frames=256,
                audio_patch_frames=4,
            ),
            hooks=HookConfig(hook_tokens_per_extension=4, max_extensions=4),
            agent=AgentConfig(max_steps=8, max_think_steps=2, thought_tokens=4),
            training=TrainingConfig(batch_size=2, warmup_steps=100, max_steps=1000),
            kb=KBConfig(embedding_dim=256, chunk_tokens=256, chunk_overlap=32),
        ).validate()

    @classmethod
    def small_real(cls) -> "OmniAgentConfig":
        return cls(name="small_real").validate()

    @classmethod
    def default(cls) -> "OmniAgentConfig":
        """Default product target; currently aliases ``small_real``."""
        return cls.small_real()

    @classmethod
    def research_large_optional(cls) -> "OmniAgentConfig":
        return cls(
            name="research_large_optional",
            model=ModelConfig(
                hidden_dim=768,
                layers=12,
                heads=12,
                max_seq_len=2048,
                reasoning_tokens=16,
                reasoning_layers=4,
                reasoning_heads=8,
                role_latent_dim=128,
            ),
            modalities=ModalityConfig(text_max_len=512),
            hooks=HookConfig(hook_tokens_per_extension=8, max_extensions=12),
            agent=AgentConfig(max_steps=24, max_think_steps=4, thought_tokens=16),
            training=TrainingConfig(batch_size=4, warmup_steps=2000, max_steps=100_000),
            kb=KBConfig(embedding_dim=768),
        ).validate()

    @classmethod
    def from_preset(cls, name: str) -> "OmniAgentConfig":
        if name == "tiny" or name == "tiny_debug":
            return cls.tiny_debug()
        if name == "small" or name == "small_real" or name == "default":
            return cls.small_real()
        if name == "large" or name == "research_large_optional":
            return cls.research_large_optional()
        raise ValueError(f"Unknown preset {name!r}; expected one of {cls.PRESETS}")

    def to_dict(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OmniAgentConfig":
        data = dict(payload)
        return cls(
            model=_coerce_dataclass(ModelConfig, data.get("model", {})),
            modalities=_coerce_dataclass(ModalityConfig, data.get("modalities", {})),
            hooks=_coerce_dataclass(HookConfig, data.get("hooks", {})),
            agent=_coerce_dataclass(AgentConfig, data.get("agent", {})),
            data=_coerce_dataclass(DataConfig, data.get("data", {})),
            training=_coerce_dataclass(TrainingConfig, data.get("training", {})),
            kb=_coerce_dataclass(KBConfig, data.get("kb", {})),
            name=str(data.get("name", "small_real")),
            version=int(data.get("version", 1)),
        ).validate()

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "OmniAgentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_yaml(self, path: str | Path) -> None:
        yaml = _import_yaml()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=True), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OmniAgentConfig":
        yaml = _import_yaml()
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"YAML config {path} did not contain a mapping")
        return cls.from_dict(payload)

    def to_omnilatent_config(self) -> Any:
        """Build the legacy ``omnilatent.config.OmniLatentConfig`` lazily."""
        from omnilatent.config import OmniLatentConfig

        return OmniLatentConfig(
            hidden_dim=self.model.hidden_dim,
            num_layers=self.model.layers,
            num_heads=self.model.heads,
            mlp_ratio=self.model.mlp_ratio,
            dropout=self.model.dropout,
            max_seq_len=self.model.max_seq_len,
            gradient_checkpointing=self.model.gradient_checkpointing,
            vocab_size=self.modalities.vocab_size,
            text_max_len=self.modalities.text_max_len,
            text_pad_token=self.modalities.text_pad_token,
            text_bos_token=self.modalities.text_bos_token,
            audio_n_mels=self.modalities.audio_n_mels,
            audio_hop_length=self.modalities.audio_hop_length,
            audio_max_frames=self.modalities.audio_max_frames,
            audio_patch_frames=self.modalities.audio_patch_frames,
            audio_sample_rate=self.modalities.audio_sample_rate,
            image_size=self.modalities.image_size,
            image_patch_size=self.modalities.image_patch_size,
            image_channels=self.modalities.image_channels,
            video_size=self.modalities.video_size,
            video_patch_size=self.modalities.video_patch_size,
            video_temporal_patch=self.modalities.video_temporal_patch,
            video_max_frames=self.modalities.video_max_frames,
            video_channels=self.modalities.video_channels,
            reasoning_enabled=self.model.reasoning_tokens > 0,
            reasoning_num_thoughts=self.model.reasoning_tokens,
            reasoning_num_layers=self.model.reasoning_layers,
            reasoning_num_heads=self.model.reasoning_heads,
            hook_gate_init=self.hooks.gate_init,
            hook_gate_bias_init=self.hooks.gate_bias_init,
            learning_rate=self.training.learning_rate,
            weight_decay=self.training.weight_decay,
            warmup_steps=self.training.warmup_steps,
            max_steps=self.training.max_steps,
            batch_size=self.training.batch_size,
            mixed_precision=self.training.mixed_precision,
            grad_clip=self.training.grad_clip,
            seed=self.training.seed,
            modality_loss_weights=dict(self.training.modality_loss_weights),
            contrastive_weight=self.training.contrastive_weight,
            contrastive_temperature=self.training.contrastive_temperature,
        )

    def to_mmwm_config(self) -> Any:
        """Build the legacy ``MMWM.config.ModelConfig`` lazily."""
        from MMWM.config import ModelConfig as MMWMModelConfig

        role_dim = self.model.role_latent_dim
        hidden = self.model.hidden_dim
        action_dim = role_dim
        primary_dim = role_dim * 4
        return MMWMModelConfig(
            encoder_kwargs={
                "text_vocab_size": self.modalities.vocab_size,
                "text_embed_dim": min(256, hidden),
                "vector_input_dim": role_dim,
                "image_channels": self.modalities.image_channels,
                "audio_channels": 1,
                "hidden_dim": hidden,
            },
            latent_projector_kwargs={"input_dim": hidden, "latent_dim": role_dim, "use_norm": True},
            memory_kwargs={"input_dim": role_dim + action_dim, "hidden_dim": role_dim},
            action_encoder_kwargs={"action_dim": action_dim, "action_embed_dim": role_dim},
            conditioner_kwargs={
                "latent_dim": primary_dim,
                "action_dim": role_dim,
                "memory_dim": role_dim,
                "out_dim": primary_dim,
            },
            transition_core_kwargs={"input_dim": primary_dim, "hidden_dim": primary_dim},
            prediction_head_kwargs={"hidden_dim": primary_dim, "latent_dim": role_dim},
            decoder_configs=[
                (
                    "text_autoregressive_head",
                    {
                        "vocab_size": self.modalities.vocab_size,
                        "latent_dim": role_dim,
                        "text_embed_dim": min(256, hidden),
                        "hidden_dim": min(256, hidden),
                    },
                )
            ],
        )


# Friendly aliases for call sites that prefer function-style presets.
def tiny_debug() -> OmniAgentConfig:
    return OmniAgentConfig.tiny_debug()


def small_real() -> OmniAgentConfig:
    return OmniAgentConfig.small_real()


def default() -> OmniAgentConfig:
    return OmniAgentConfig.default()


def research_large_optional() -> OmniAgentConfig:
    return OmniAgentConfig.research_large_optional()


def _coerce_dataclass(cls: type[T], value: Any) -> T:
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping for {cls.__name__}, got {type(value).__name__}")
    allowed = {f.name for f in fields(cls) if f.init}
    kwargs = {k: v for k, v in value.items() if k in allowed}
    obj = cls(**kwargs)  # type: ignore[arg-type]
    if not is_dataclass(obj):
        raise TypeError(f"{cls.__name__} is not a dataclass")
    return obj


def _require_positive_int(name: str, value: int, *, allow_zero: bool = False) -> None:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
    elif value <= 0:
        raise ValueError(f"{name} must be > 0")


def _require_range(name: str, value: float, *, min_value: float, max_value: float) -> None:
    if not (min_value <= value <= max_value):
        raise ValueError(f"{name} must be in [{min_value}, {max_value}], got {value}")


def _import_yaml() -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise RuntimeError("YAML config support requires PyYAML. Install with `pip install PyYAML`.") from exc
    return yaml


__all__ = [
    "AgentConfig",
    "DataConfig",
    "HookConfig",
    "KBConfig",
    "ModalityConfig",
    "ModelConfig",
    "OmniAgentConfig",
    "TrainingConfig",
    "default",
    "research_large_optional",
    "small_real",
    "tiny_debug",
]


