"""Minimal demo: train on synthetic data + exercise tool ecosystem."""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from .config import ModelConfig, build_model, build_tool_ecosystem
from .containers import LatentPacket, ObservationPacket, ToolContext, ToolDescriptor
from .curriculum import AdaptiveCurriculumScheduler, default_curriculum_phases, relative_curriculum_phases
from .data import GridWorldTransitionDataset, collate_transition_batch
from .evaluation import EvaluationSuite
from .losses import LossWeights, WorldModelLoss
from .trainer import Trainer, build_lr_scheduler


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


def make_gridworld_smoke_config(grid_size: int = 5) -> ModelConfig:
    """Tiny vector-first config for the structured gridworld smoke test."""
    return ModelConfig(
        encoder_name="simple_multimodal",
        encoder_kwargs={
            "text_vocab_size": max(16, grid_size + 8),
            "text_embed_dim": 16,
            "vector_input_dim": 2,
            "image_channels": 3,
            "hidden_dim": 16,
        },
        latent_projector_kwargs={"input_dim": 16, "latent_dim": 8, "use_norm": False},
        memory_kwargs={"input_dim": 16, "hidden_dim": 8},
        action_encoder_kwargs={"action_dim": 4, "action_embed_dim": 8},
        conditioner_kwargs={"latent_dim": 32, "action_dim": 8, "memory_dim": 8, "out_dim": 32},
        transition_core_name="mlp",
        transition_core_kwargs={"input_dim": 32, "hidden_dim": 32},
        prediction_head_kwargs={"hidden_dim": 32, "latent_dim": 8},
        decoder_configs=[("vector_reconstruction", {"latent_dim": 8, "output_dim": 2})],
    )


def _vector_only_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keep = {"vector_t", "vector_tp1", "vector_target", "action", "done"}
    return {key: value for key, value in batch.items() if key in keep}


def run_gridworld_smoke(
    *,
    steps: int = 40,
    batch_size: int = 8,
    run_dir: str = "runs/mmwm_gridworld_smoke",
    device: torch.device | None = None,
) -> Dict[str, float]:
    """Run the structured gridworld train/checkpoint/rollout smoke flow."""
    torch.manual_seed(0)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GridWorldTransitionDataset(length=max(64, steps * batch_size), grid_size=5)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_transition_batch)
    eval_batch = _vector_only_batch(collate_transition_batch([dataset[i] for i in range(batch_size)]))

    cfg = make_gridworld_smoke_config(grid_size=dataset.grid_size)
    model = build_model(cfg)
    loss_fn = WorldModelLoss(
        weights=LossWeights(
            latent_sem=0.0,
            latent_dyn=0.0,
            latent_ctrl=0.0,
            latent_ctx=0.0,
            regularizer=0.0,
            text_ce=0.0,
            image_recon=0.0,
            audio_recon=0.0,
            contrastive_alignment=0.0,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        run_dir=run_dir,
        mixed_precision=False,
        reset_memory_each_batch=True,
    )

    before, _ = trainer.eval_step(eval_batch)
    step = 0
    while step < steps:
        for batch in loader:
            trainer.train_step(_vector_only_batch(batch))
            step += 1
            if step >= steps:
                break
    after, _ = trainer.eval_step(eval_batch)

    checkpoint = trainer.save_checkpoint()
    loaded_model = build_model(cfg)
    loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=1e-2)
    loaded = Trainer(
        model=loaded_model,
        optimizer=loaded_optimizer,
        loss_fn=loss_fn,
        device=device,
        run_dir=run_dir,
        mixed_precision=False,
    )
    loaded.load_checkpoint(checkpoint)
    obs_sequence, action_sequence = dataset.rollout_sequence(start_index=0, horizon=4, modalities=("vector",))
    rollout = loaded.eval_suite.evaluate_rollout(loaded.model, obs_sequence, action_sequence, max_horizon=4)

    return {
        "vector_mse_before": float(before.get("vector_mse", before["total_loss"])),
        "vector_mse_after": float(after.get("vector_mse", after["total_loss"])),
        "loaded_global_step": float(loaded.global_step),
        **{k: float(v) for k, v in rollout.rollout_metrics.items()},
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
            "use_norm": True,
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
    scheduler = build_lr_scheduler(optimizer, total_steps=10_000, warmup_fraction=0.05)
    loss_fn = WorldModelLoss(learned_uncertainty=True)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        run_dir="runs/modular_world_model",
        grad_clip_norm=1.0,
        mixed_precision=True,
        lr_scheduler=scheduler,
    )
    # Use relative curriculum (adapts to total training steps)
    trainer.set_curriculum(relative_curriculum_phases(total_steps=10_000))

    # Optionally, use adaptive curriculum instead:
    # adaptive = AdaptiveCurriculumScheduler(default_curriculum_phases(), patience=200)
    # trainer.set_adaptive_curriculum(adaptive)

    for _ in range(10):
        batch = make_dummy_batch(device=device)
        metrics, _ = trainer.train_step(batch)
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
