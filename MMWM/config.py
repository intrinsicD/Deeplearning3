"""Model configuration and factory.

Fix for brittle build_model: Instead of if/else branching on component names,
each component slot has its own kwargs dict in the config. The registry
passes these kwargs directly to the constructor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .containers import LatentPacket, ToolContext, ToolDescriptor
from .interfaces import (
    DECODERS,
    ENCODERS,
    IDecoder,
    LATENT_PROJECTORS,
    MEMORIES,
    ACTION_ENCODERS,
    CONDITIONERS,
    TRANSITION_CORES,
    PREDICTION_HEADS,
    REGULARIZERS,
)
from .model import ModularLatentWorldModel
from .tools import (
    BinaryToolLoader,
    CandidatePromoter,
    CriticalExperienceBuffer,
    LatentRouter,
    ToolExecutionEngine,
    ToolJudge,
    ToolRegistrySystem,
)

# Force registration of all components by importing the modules
from . import components as _components  # noqa: F401
from . import decoders as _decoders  # noqa: F401
from . import encoders as _encoders  # noqa: F401
from . import transitions as _transitions  # noqa: F401


@dataclass
class ModelConfig:
    """Declarative model configuration.

    Each component slot has a name (selects from the registry) and a kwargs
    dict passed directly to the constructor. No more if/else branching.
    """

    encoder_name: str = "simple_multimodal"
    encoder_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "text_vocab_size": 32000,
        "text_embed_dim": 256,
        "vector_input_dim": 128,
        "image_channels": 3,
        "hidden_dim": 256,
    })

    latent_projector_name: str = "role_split_mlp"
    latent_projector_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "input_dim": 256,
        "latent_dim": 128,
        "use_norm": True,
    })

    memory_name: str = "gru"
    memory_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "input_dim": 256,  # latent_dim + action_embed_dim
        "hidden_dim": 128,
    })

    action_encoder_name: str = "mlp"
    action_encoder_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "action_dim": 32,
        "action_embed_dim": 128,
    })

    conditioner_name: str = "concat_mlp"
    conditioner_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "latent_dim": 512,   # latent_dim * 4 roles
        "action_dim": 128,
        "memory_dim": 128,
        "out_dim": 512,
    })

    transition_core_name: str = "recurrent_attnres_transformer"
    transition_core_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "input_dim": 512,
        "hidden_dim": 512,
    })

    prediction_head_name: str = "role_split"
    prediction_head_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_dim": 512,
        "latent_dim": 128,
    })

    regularizer_name: str = "sigreg_like"
    regularizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    decoder_configs: List[Tuple[str, Dict[str, Any]]] = field(default_factory=lambda: [
        ("text_autoregressive_head", {
            "vocab_size": 32000,
            "latent_dim": 128,
            "text_embed_dim": 256,
            "hidden_dim": 256,
        }),
    ])


def build_model(cfg: ModelConfig) -> ModularLatentWorldModel:
    """Build a model from config. Each component is constructed via its
    registry name + kwargs dict — no component-specific branching."""

    encoder = ENCODERS.build(cfg.encoder_name, **cfg.encoder_kwargs)
    latent_projector = LATENT_PROJECTORS.build(cfg.latent_projector_name, **cfg.latent_projector_kwargs)
    memory = MEMORIES.build(cfg.memory_name, **cfg.memory_kwargs)
    action_encoder = ACTION_ENCODERS.build(cfg.action_encoder_name, **cfg.action_encoder_kwargs)
    conditioner = CONDITIONERS.build(cfg.conditioner_name, **cfg.conditioner_kwargs)
    transition_core = TRANSITION_CORES.build(cfg.transition_core_name, **cfg.transition_core_kwargs)
    prediction_head = PREDICTION_HEADS.build(cfg.prediction_head_name, **cfg.prediction_head_kwargs)
    regularizer = REGULARIZERS.build(cfg.regularizer_name, **cfg.regularizer_kwargs)

    decoders: Dict[str, IDecoder] = {}
    for name, kwargs in cfg.decoder_configs:
        decoders[name] = DECODERS.build(name, **kwargs)

    return ModularLatentWorldModel(
        encoder=encoder,
        latent_projector=latent_projector,
        memory=memory,
        action_encoder=action_encoder,
        conditioner=conditioner,
        transition_core=transition_core,
        prediction_head=prediction_head,
        regularizer=regularizer,
        decoders=decoders,
    )


def build_tool_ecosystem(
    model: ModularLatentWorldModel,
    device: torch.device,
    keep_hot_tools: int = 2,
    entrypoint_prefix: str = "MMWM.tools",
) -> Tuple[ToolRegistrySystem, ToolExecutionEngine, CriticalExperienceBuffer, CandidatePromoter]:
    """Build a binary/lazy-loaded tool ecosystem around the world model."""
    loader = BinaryToolLoader(device=device, keep_hot=keep_hot_tools)
    registry = ToolRegistrySystem(loader)

    shared_memory_bank: Dict[str, torch.Tensor] = {}

    registry.register(
        ToolDescriptor(
            tool_id="memory.read.v1",
            kind="memory.read",
            version="1.0",
            entry_point=f"{entrypoint_prefix}:build_memory_read_tool",
            benchmark_id="memory_read_basic",
            estimated_latency_ms=0.2,
            memory_mb=1.0,
            config={"memory_bank": shared_memory_bank},
        ),
        alias="memory.read",
    )
    registry.register(
        ToolDescriptor(
            tool_id="memory.write.v1",
            kind="memory.write",
            version="1.0",
            entry_point=f"{entrypoint_prefix}:build_memory_write_tool",
            benchmark_id="memory_write_basic",
            estimated_latency_ms=0.2,
            memory_mb=1.0,
            config={"memory_bank": shared_memory_bank},
        ),
        alias="memory.write",
    )

    registry.register(
        ToolDescriptor(
            tool_id="transition.internal.v1",
            kind="transition.latent",
            version="1.0",
            entry_point=f"{entrypoint_prefix}:build_internal_transition_tool",
            benchmark_id="transition_rollout_basic",
            estimated_latency_ms=1.0,
            memory_mb=64.0,
            config={"transition_model": model},
            binary=True,
            lazy_load=True,
        ),
        alias="transition.internal",
    )

    has_text_decoder = "text_autoregressive_head" in model.decoders
    if has_text_decoder:
        registry.register(
            ToolDescriptor(
                tool_id="decoder.text.v1",
                kind="decoder.text",
                version="1.0",
                entry_point=f"{entrypoint_prefix}:build_text_decoder_tool",
                benchmark_id="decoder_text_basic",
                estimated_latency_ms=0.5,
                memory_mb=16.0,
                config={"decoder": model.decoders["text_autoregressive_head"]},
                binary=True,
                lazy_load=True,
            ),
            alias="decoder.text",
        )

    primary_dim = model.latent_projector.sem.out_features * 4
    num_actions = 4 if has_text_decoder else 3
    router = LatentRouter(latent_dim=primary_dim, num_actions=num_actions)
    action_id_to_tool = {
        0: "transition.internal",
        1: "memory.read",
        2: "memory.write",
    }
    if has_text_decoder:
        action_id_to_tool[3] = "decoder.text"

    engine = ToolExecutionEngine(registry, router, action_id_to_tool, device=device)
    critical_buffer = CriticalExperienceBuffer()
    judge = ToolJudge(critical_buffer)
    promoter = CandidatePromoter(registry, judge)
    return registry, engine, critical_buffer, promoter
