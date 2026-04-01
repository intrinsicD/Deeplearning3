"""Minimal demo: train on synthetic data + exercise tool ecosystem."""

from __future__ import annotations

from typing import Dict

import torch

from .config import ModelConfig, build_model, build_tool_ecosystem
from .containers import LatentPacket, ObservationPacket, ToolContext, ToolDescriptor
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
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ModelConfig()
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = WorldModelLoss()
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        run_dir="runs/modular_world_model",
        grad_clip_norm=1.0,
        mixed_precision=True,
    )

    for _ in range(10):
        batch = make_dummy_batch(device=device)
        metrics = trainer.train_step(batch)
        print({k: round(v, 4) for k, v in metrics.items()})

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
