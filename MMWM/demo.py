"""Minimal demo: train on synthetic data + exercise tool ecosystem."""

from __future__ import annotations

from typing import Dict

import torch

from .config import ModelConfig, build_model, build_tool_ecosystem
from .containers import LatentPacket, ObservationPacket, ToolContext, ToolDescriptor
from .curriculum import AdaptiveCurriculumScheduler, default_curriculum_phases, relative_curriculum_phases
from .evaluation import EvaluationSuite
from .losses import WorldModelLoss
from .trainer import Trainer


def make_dummy_batch(
    batch_size: int = 8,
    text_len: int = 16,
    vocab_size: int = 32000,
    vector_dim: int = 128,
    action_dim: int = 32,
    image_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    return {
        "text_t": torch.randint(0, vocab_size, (batch_size, text_len), device=device),
        "text_tp1": torch.randint(0, vocab_size, (batch_size, text_len), device=device),
        "vector_t": torch.randn(batch_size, vector_dim, device=device),
        "vector_tp1": torch.randn(batch_size, vector_dim, device=device),
        "image_t": torch.randn(batch_size, 3, image_size, image_size, device=device),
        "image_tp1": torch.randn(batch_size, 3, image_size, image_size, device=device),
        "action": torch.randn(batch_size, action_dim, device=device),
        "prefix_tokens": torch.randint(0, vocab_size, (batch_size, text_len), device=device),
        "text_target": torch.randint(0, vocab_size, (batch_size, text_len), device=device),
        "vector_target": torch.randn(batch_size, vector_dim, device=device),
        "image_target": torch.rand(batch_size, 3, image_size, image_size, device=device),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig(
        encoder_name="structured_multimodal",
        encoder_kwargs={
            "text_vocab_size": 32000,
            "text_embed_dim": 256,
            "text_transformer_layers": 2,
            "text_nhead": 4,
            "vector_input_dim": 128,
            "image_channels": 3,
            "hidden_dim": 256,
            "n_slots": 8,
            "slot_dim": 128,
            "slot_iters": 3,
            "merge_threshold": 0.9,
            "fusion_nhead": 4,
        },
        latent_projector_name="adaptive_role_split_mlp",
        latent_projector_kwargs={
            "input_dim": 256,
            "latent_dim": 128,
            "intermediate_dim": 512,
            "use_batchnorm": True,
        },
        memory_name="mamba_ssm",
        transition_core_name="mod_recurrent_attnres_transformer",
        decoder_configs=[
            ("text_autoregressive_head", {
                "vocab_size": 32000,
                "latent_dim": 128,
                "text_embed_dim": 256,
                "hidden_dim": 256,
            }),
            ("image_reconstruction", {
                "latent_dim": 512,
                "base_channels": 128,
                "output_channels": 3,
                "output_size": 64,
            }),
        ],
    )
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = WorldModelLoss(learned_uncertainty=True)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        run_dir="runs/modular_world_model",
        grad_clip_norm=1.0,
        mixed_precision=True,
    )
    # Use relative curriculum (adapts to total training steps)
    trainer.set_curriculum(relative_curriculum_phases(total_steps=10_000))

    # Optionally, use adaptive curriculum instead:
    # adaptive = AdaptiveCurriculumScheduler(default_curriculum_phases(), patience=200)
    # trainer.set_adaptive_curriculum(adaptive)

    for _ in range(10):
        batch = make_dummy_batch(device=device)
        metrics = trainer.train_step(batch)
        print({k: round(v, 4) for k, v in metrics.items() if not k.endswith("_log_var")})

    registry, engine, critical_buffer, promoter = build_tool_ecosystem(
        model,
        device=device,
        keep_hot_tools=2,
        entrypoint_prefix="MMWM.tools",
    )
    critical_buffer.add({
        "id": "decoder_regression_001",
        "benchmarks": ["decoder_text_basic"],
        "description": "Text decoder should remain callable after tool routing.",
        "weight": 10.0,
    })

    obs = ObservationPacket(
        modalities={
            "text": torch.randint(0, 32000, (1, 16), device=device),
            "vector": torch.randn(1, 128, device=device),
            "image": torch.randn(1, 3, 64, 64, device=device),
        }
    )
    initial_state = model.encode(obs)
    packet = LatentPacket(state=initial_state, trace=["encode"])
    context = ToolContext(
        memory_state=model.memory.init_state(batch_size=1, device=device),
        observation=obs,
        action=torch.randn(1, 32, device=device),
        raw_inputs={
            "prefix_tokens": torch.randint(0, 32000, (1, 16), device=device),
            "memory_key": "default",
        },
    )
    final_packet, trace = engine.iterate(packet, context, max_steps=4)
    print("Tool iteration trace:", trace)
    print("Final packet trace:", final_packet.trace)

    candidate = ToolDescriptor(
        tool_id="memory.read.v2",
        kind="memory.read",
        version="2.0",
        entry_point="MMWM.tools:build_memory_read_tool",
        benchmark_id="memory_read_basic",
        estimated_latency_ms=0.15,
        memory_mb=1.2,
        config={"memory_bank": registry.get("memory.read").config["memory_bank"]},
    )
    accepted, report = promoter.maybe_promote(
        old_tool_id="memory.read.v1",
        candidate=candidate,
        local_metrics={"quality": 0.2, "latency_ms": 0.15, "memory_mb": 1.2},
        regression_metrics={"critical_failures": 0.0},
    )
    print("Candidate accepted:", accepted)
    print("Promotion report:", report)

    generated = trainer.generate_next_tokens(
        obs,
        torch.randn(1, 32, device=device),
        torch.randint(0, 32000, (1, 16), device=device),
        steps=8,
    )
    print("Generated token ids:", generated)
    print("Open TensorBoard in a browser with: tensorboard --logdir runs")


if __name__ == "__main__":
    main()
