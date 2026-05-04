#!/usr/bin/env python3
"""Curriculum training for OmniLatent: learn from watching videos.

Trains the model in phases, progressively introducing harder tasks:

  Phase 1 — Warmup:           Self-reconstruction (video->video, audio->audio)
  Phase 2 — Cross-modal:      Audio<->Video alignment
  Phase 3 — Temporal:         Temporal context tasks (order, distance, prediction)
  Phase 4 — Grounding:        Video/Audio -> Text (requires transcripts)
  Phase 5 — Joint:            All tasks together

Supports three complementary temporal context approaches:
  1. Multi-scale temporal sampling — dataset-level clip pair tasks
  2. Hierarchical temporal transformer — sequence-level clip modeling
  3. Recurrent memory tokens — persistent memory across clips

Usage:
    # Train on a directory of video files
    python curriculum_train.py --video-dir /path/to/videos

    # Quick test run
    python curriculum_train.py --video-dir /path/to/videos --total-steps 500

    # Custom phase durations (fractions of total steps)
    python curriculum_train.py --video-dir /path/to/videos \\
        --warmup-frac 0.1 --crossmodal-frac 0.25

    # Use synthetic data for testing (no videos needed)
    python curriculum_train.py --synthetic
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- ADDED: Hardware Optimizations ---
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# -------------------------------------

from omnilatent.config import OmniLatentConfig
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.model.temporal import TemporalSequenceTransformer, RecurrentMemory
from omnilatent.training.losses import (
    MultiModalLoss,
    TemporalContextLoss,
    NextClipPredictionLoss,
)
from omnilatent.training.trainer import cosine_schedule
from omnilatent.utils import count_parameters, param_size_mb, set_seed


# -------------------------------------------------------------------------
# Phase definitions
# -------------------------------------------------------------------------
@dataclass
class Phase:
    name: str
    tasks: list[str]
    frac: float  # fraction of total steps
    description: str


DEFAULT_PHASES = [
    Phase(
        name="warmup",
        tasks=["video_recon", "audio_recon"],
        frac=0.10,
        description="Self-reconstruction — learn to encode/decode each modality",
    ),
    Phase(
        name="cross_modal",
        tasks=["video_recon", "audio_recon", "video_to_audio", "audio_to_video"],
        frac=0.25,
        description="Cross-modal alignment — learn audio<->video correspondence",
    ),
    Phase(
        name="temporal",
        tasks=[
            "video_recon", "audio_recon",
            "video_to_audio", "audio_to_video",
            "temporal_predict",
            "temporal_order", "temporal_distance", "distant_predict",
        ],
        frac=0.25,
        description="Temporal prediction — learn temporal structure across timescales",
    ),
    Phase(
        name="grounding",
        tasks=[
            "video_recon", "audio_recon",
            "video_to_audio", "audio_to_video",
            "temporal_predict",
            "temporal_order", "temporal_distance", "distant_predict",
            "video_to_text", "audio_to_text",
        ],
        frac=0.15,
        description="Language grounding — learn to describe video/audio in text",
    ),
    Phase(
        name="joint",
        tasks=[
            "video_recon", "audio_recon",
            "video_to_audio", "audio_to_video",
            "temporal_predict", "frame_to_audio",
            "temporal_order", "temporal_distance", "distant_predict",
            "video_to_text", "audio_to_text",
        ],
        frac=0.25,
        description="Joint training — all tasks together",
    ),
]


# -------------------------------------------------------------------------
# Curriculum Trainer
# -------------------------------------------------------------------------
class CurriculumTrainer:
    """Multi-phase curriculum trainer that learns from video files.

    Each phase introduces progressively harder tasks.  The trainer
    automatically transitions between phases based on step count.

    Supports three temporal context training modes:
      1. Standard single-clip training (existing)
      2. Temporal sequence training via TemporalSequenceTransformer
      3. Memory-augmented training via RecurrentMemory
    """

    def __init__(
        self,
        model: OmniLatentModel,
        config: OmniLatentConfig,
        dataloader: DataLoader,
        phases: list[Phase] | None = None,
        total_steps: int = 100_000,
        save_dir: str | None = None,
        temporal_transformer: TemporalSequenceTransformer | None = None,
        recurrent_memory: RecurrentMemory | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.dataloader = dataloader
        self.phases = phases or DEFAULT_PHASES
        self.total_steps = total_steps
        self.save_dir = Path(save_dir) if save_dir else None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Temporal context modules (Approach 2 & 3)
        self.temporal_transformer = temporal_transformer
        self.recurrent_memory = recurrent_memory
        if self.temporal_transformer is not None:
            self.temporal_transformer.to(self.device)
        if self.recurrent_memory is not None:
            self.recurrent_memory.to(self.device)

        # Collect all parameters for optimizer
        all_params = list(self.model.parameters())
        if self.temporal_transformer is not None:
            all_params += list(self.temporal_transformer.parameters())
        if self.recurrent_memory is not None:
            all_params += list(self.recurrent_memory.parameters())

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Loss
        self.criterion = MultiModalLoss(config).to(self.device)
        self.temporal_criterion = TemporalContextLoss(config).to(self.device)
        self.next_clip_criterion = NextClipPredictionLoss().to(self.device)

        # Mixed precision
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=config.mixed_precision and self.device.type == "cuda",
        )
        self.amp_dtype = (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )

        # Compute phase step boundaries
        self._phase_boundaries: list[tuple[int, int]] = []
        offset = 0
        for phase in self.phases:
            steps = max(1, int(total_steps * phase.frac))
            self._phase_boundaries.append((offset, offset + steps))
            offset += steps

        self.global_step = 0

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_current_phase(self) -> tuple[int, Phase]:
        for i, (start, end) in enumerate(self._phase_boundaries):
            if self.global_step < end:
                return i, self.phases[i]
        return len(self.phases) - 1, self.phases[-1]

    def _update_lr(self) -> float:
        lr = cosine_schedule(
            self.global_step,
            self.total_steps,
            self.config.warmup_steps,
            self.config.learning_rate,
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def _train_step_video(self, batch: dict) -> dict[str, float]:
        """Training step for video-watching data (task-annotated batches)."""
        src_mod = batch["source_modality"]
        tgt_mod = batch["target_modality"]
        source = batch["source"].to(self.device)
        target = batch["target"].to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda", dtype=self.amp_dtype, enabled=self.config.mixed_precision
        ):
            result = self.model(
                source_modality=src_mod,
                source_data=source,
                target_modality=tgt_mod,
                target_data=target,
            )

            predictions = {tgt_mod: result["output"]}
            targets_dict = {tgt_mod: target}

            # Contrastive loss: encode both modalities
            latents = None
            if src_mod != tgt_mod and self.config.contrastive_weight > 0:
                latents = {}
                src_enc = self.model.encode(src_mod, source)
                tgt_enc = self.model.encode(tgt_mod, target)
                latents[src_mod] = src_enc[:, 1:].mean(dim=1)
                latents[tgt_mod] = tgt_enc[:, 1:].mean(dim=1)

            loss_dict = self.criterion(predictions, targets_dict, latents)

        self.scaler.scale(loss_dict["total"]).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {k: v.item() for k, v in loss_dict.items()}

    def _train_step_synthetic(self, batch: dict) -> dict[str, float]:
        """Training step for synthetic multi-modal data."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        available = list(batch.keys())
        if not available:
            return {"total": 0.0}

        src_mod = random.choice(available)
        tgt_mod = random.choice(available)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda", dtype=self.amp_dtype, enabled=self.config.mixed_precision
        ):
            # Encode all available modalities for contrastive loss
            latents = None
            if len(available) >= 2 and self.config.contrastive_weight > 0:
                latents = {}
                for mod in available:
                    enc = self.model.encode(mod, batch[mod])
                    latents[mod] = enc[:, 1:].mean(dim=1)

            result = self.model(
                source_modality=src_mod,
                source_data=batch[src_mod],
                target_modality=tgt_mod,
                target_data=batch[tgt_mod],
            )
            predictions = {tgt_mod: result["output"]}
            targets = {tgt_mod: batch[tgt_mod]}
            loss_dict = self.criterion(predictions, targets, latents)

        self.scaler.scale(loss_dict["total"]).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {k: v.item() for k, v in loss_dict.items()}

    def _train_step_temporal_pairs(self, batch: dict) -> dict[str, float]:
        """Training step for Approach 1: multi-scale temporal pair tasks.

        Uses temporal pair data (anchor + context clips) to train on
        temporal_order, temporal_distance, and distant_predict tasks.
        """
        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda", dtype=self.amp_dtype, enabled=self.config.mixed_precision
        ):
            temporal_losses: dict[str, torch.Tensor] = {}

            # Encode anchor and context clips
            if "anchor_video" in batch and "context_video" in batch:
                anchor_vid = batch["anchor_video"].to(self.device)
                context_vid = batch["context_video"].to(self.device)

                # Get latent representations (mean-pooled, skip modality token)
                anchor_enc = self.model.encode("video", anchor_vid)
                context_enc = self.model.encode("video", context_vid)
                z_anchor = anchor_enc[:, 1:].mean(dim=1)  # (B, D)
                z_context = context_enc[:, 1:].mean(dim=1)  # (B, D)

                # Temporal order loss
                if "temporal_order" in batch:
                    order_labels = batch["temporal_order"].to(self.device)
                    temporal_losses["temporal_order"] = \
                        self.temporal_criterion.order_loss(
                            z_anchor, z_context, order_labels
                        )

                # Temporal distance loss
                if "temporal_distance_bucket" in batch:
                    dist_labels = batch["temporal_distance_bucket"].to(self.device)
                    temporal_losses["temporal_distance"] = \
                        self.temporal_criterion.distance_loss(
                            z_anchor, z_context, dist_labels
                        )

                # Distant predict: predict context latent from anchor
                temporal_losses["distant_predict"] = F.mse_loss(
                    z_anchor, z_context
                ) + 0.5 * (1.0 - F.cosine_similarity(z_anchor, z_context, dim=-1).mean())

            if not temporal_losses:
                return {"total": 0.0}

            loss_result = self.temporal_criterion(temporal_losses)
            total = loss_result["temporal_total"]

        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer)

        # Clip grads for all parameter groups
        all_params = list(self.model.parameters())
        if self.temporal_transformer is not None:
            all_params += list(self.temporal_transformer.parameters())
        if self.recurrent_memory is not None:
            all_params += list(self.recurrent_memory.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, self.config.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in loss_result.items()}

    def _train_step_temporal_sequence(self, batch: dict) -> dict[str, float]:
        """Training step for Approach 2: hierarchical temporal transformer.

        Encodes a sequence of clips independently (no_grad on the encoder),
        then trains the temporal transformer on next-clip prediction.
        """
        if self.temporal_transformer is None:
            return {"total": 0.0}

        clip_videos = batch["clip_videos"].to(self.device)  # (B, N, C, T, H, W)
        clip_mask = batch["clip_mask"].to(self.device)       # (B, N)
        B, N = clip_videos.shape[:2]

        self.optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda", dtype=self.amp_dtype, enabled=self.config.mixed_precision
        ):
            # Step 1: Encode each clip independently (no gradients through encoder)
            clip_latents = []
            with torch.no_grad():
                for i in range(N):
                    clip_i = clip_videos[:, i]  # (B, C, T, H, W)
                    enc = self.model.encode("video", clip_i)  # (B, 1+K, D)
                    # Mean-pool over spatial tokens (skip modality token)
                    z_i = enc[:, 1:].mean(dim=1)  # (B, D)
                    clip_latents.append(z_i)

            clip_latents = torch.stack(clip_latents, dim=1)  # (B, N, D)

            # Step 2: Feed through temporal transformer (with gradients)
            temporal_out = self.temporal_transformer(clip_latents, clip_mask)
            pred_next = temporal_out["next_clip_pred"]  # (B, N, D)

            # Step 3: Compute next-clip prediction loss
            # Target: shifted clip latents (predict z_{t+1} from position t)
            target_next = clip_latents[:, 1:]        # (B, N-1, D)
            pred_next_shifted = pred_next[:, :-1]    # (B, N-1, D)
            shifted_mask = clip_mask[:, 1:]           # (B, N-1)

            loss = self.next_clip_criterion(
                pred_next_shifted, target_next, shifted_mask
            )

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        # Only clip temporal transformer params (encoder is frozen)
        torch.nn.utils.clip_grad_norm_(
            self.temporal_transformer.parameters(), self.config.grad_clip
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {"next_clip_loss": loss.item(), "total": loss.item()}

    def _train_step_memory(self, batch: dict) -> dict[str, float]:
        """Training step for Approach 3: recurrent memory tokens.

        Processes a sequence of clips with persistent memory tokens.
        Memory state carries forward between clips (truncated BPTT).
        """
        if self.recurrent_memory is None:
            return {"total": 0.0}

        clip_videos = batch["clip_videos"].to(self.device)  # (B, N, C, T, H, W)
        clip_mask = batch["clip_mask"].to(self.device)       # (B, N)
        B, N = clip_videos.shape[:2]

        self.optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=self.device)
        n_valid = 0

        # Initialize memory state
        mem_state = self.recurrent_memory.init_state(B)

        with torch.amp.autocast(
            "cuda", dtype=self.amp_dtype, enabled=self.config.mixed_precision
        ):
            for i in range(N):
                if not clip_mask[:, i].any():
                    continue

                clip_i = clip_videos[:, i]  # (B, C, T, H, W)

                # Encode clip
                src_tokens = self.model.encode("video", clip_i)

                # Prepend memory tokens
                tokens_with_mem, mem_len = self.recurrent_memory.prepend(
                    src_tokens, mem_state
                )

                # Self-reconstruction through backbone + decoder
                # Build attention mask for prefix-LM with memory
                src_len = tokens_with_mem.shape[1]
                tgt_queries = self.model.target_query_gen("video", B)
                from omnilatent.utils import MODALITY_ID
                tgt_mod_tok = self.model.target_embed(
                    torch.tensor(MODALITY_ID["video"], device=self.device)
                ).unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
                tgt_with_mod = torch.cat([tgt_mod_tok, tgt_queries], dim=1)
                tgt_len = tgt_with_mod.shape[1]

                full_tokens = torch.cat([tokens_with_mem, tgt_with_mod], dim=1)
                attn_mask = self.model._create_attention_mask(
                    src_len, tgt_len, "video", self.device
                )

                latent = self.model.backbone(full_tokens, attn_mask=attn_mask)

                # Extract memory and content from backbone output
                mem_region = latent[:, :mem_len]
                content_region = latent[:, mem_len:]

                # Update memory state (truncated BPTT)
                _, _ = self.recurrent_memory.extract(
                    latent[:, :src_len], mem_len
                )
                new_mem, _ = self.recurrent_memory.extract(
                    latent[:, :src_len], mem_len
                )
                mem_state = self.recurrent_memory.detach_state(new_mem)

                # Decode target (skip target mod token)
                tgt_latent = content_region[:, (src_len - mem_len) + 1:]
                output = self.model.decoders["video"](tgt_latent)

                # Reconstruction loss
                clip_loss = F.l1_loss(output, clip_i)
                total_loss = total_loss + clip_loss
                n_valid += 1

        if n_valid > 0:
            avg_loss = total_loss / n_valid
        else:
            avg_loss = total_loss

        self.scaler.scale(avg_loss).backward()
        self.scaler.unscale_(self.optimizer)

        all_params = list(self.model.parameters()) + list(
            self.recurrent_memory.parameters()
        )
        torch.nn.utils.clip_grad_norm_(all_params, self.config.grad_clip)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {"memory_recon_loss": avg_loss.item(), "total": avg_loss.item()}

    def _is_video_batch(self, batch) -> bool:
        return isinstance(batch, dict) and "source_modality" in batch

    def _is_temporal_pair_batch(self, batch) -> bool:
        return isinstance(batch, dict) and "anchor_video" in batch

    def _is_clip_sequence_batch(self, batch) -> bool:
        return isinstance(batch, dict) and "clip_videos" in batch

    def save_checkpoint(self, path: Path | None = None) -> None:
        if path is None and self.save_dir:
            path = self.save_dir / f"checkpoint_step{self.global_step}.pt"
        if path is None:
            return
        state = {
            "step": self.global_step,
            "config": self.config,
            "model": {k.removeprefix("_orig_mod."): v for k, v in self.model.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        if self.temporal_transformer is not None:
            state["temporal_transformer"] = self.temporal_transformer.state_dict()
        if self.recurrent_memory is not None:
            state["recurrent_memory"] = self.recurrent_memory.state_dict()
        torch.save(state, path)
        print(f"  Saved checkpoint: {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # torch.compile wraps the model; load into the unwrapped module if present
        inner = getattr(self.model, "_orig_mod", self.model)
        inner.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt["step"]
        if self.temporal_transformer is not None and "temporal_transformer" in ckpt:
            self.temporal_transformer.load_state_dict(ckpt["temporal_transformer"])
        if self.recurrent_memory is not None and "recurrent_memory" in ckpt:
            self.recurrent_memory.load_state_dict(ckpt["recurrent_memory"])
        print(f"Resumed from step {self.global_step}")

    def train(self, log_interval: int = 50) -> None:
        """Main curriculum training loop."""
        self.model.train()
        if self.temporal_transformer is not None:
            self.temporal_transformer.train()
        if self.recurrent_memory is not None:
            self.recurrent_memory.train()

        data_iter = iter(self.dataloader)
        running_loss = 0.0
        t0 = time.time()
        current_phase_idx = -1

        for step in range(self.global_step, self.total_steps):
            self.global_step = step
            lr = self._update_lr()

            # Check for phase transition
            phase_idx, phase = self._get_current_phase()
            if phase_idx != current_phase_idx:
                current_phase_idx = phase_idx
                phase_start, phase_end = self._phase_boundaries[phase_idx]
                print()
                print("=" * 60)
                print(f"Phase {phase_idx + 1}/{len(self.phases)}: {phase.name}")
                print(f"  {phase.description}")
                print(f"  Tasks: {phase.tasks}")
                print(f"  Steps: {phase_start} -> {phase_end}")
                print("=" * 60)

            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            # Route to appropriate training step based on batch type
            if self._is_clip_sequence_batch(batch):
                # Approach 2 & 3: Clip sequence training
                if self.temporal_transformer is not None and random.random() < 0.5:
                    losses = self._train_step_temporal_sequence(batch)
                elif self.recurrent_memory is not None:
                    losses = self._train_step_memory(batch)
                else:
                    losses = self._train_step_temporal_sequence(batch)
            elif self._is_temporal_pair_batch(batch):
                # Approach 1: Temporal pair training
                losses = self._train_step_temporal_pairs(batch)
            elif self._is_video_batch(batch):
                losses = self._train_step_video(batch)
            else:
                losses = self._train_step_synthetic(batch)

            running_loss += losses.get("total", 0.0)

            if (step + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                elapsed = time.time() - t0
                steps_per_sec = log_interval / elapsed
                task_str = batch.get("task", "mixed") if isinstance(batch, dict) else "mixed"
                print(
                    f"  step {step + 1:>6d} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"{steps_per_sec:.1f} it/s | "
                    f"phase={phase.name} task={task_str}"
                )
                running_loss = 0.0
                t0 = time.time()

            # Save checkpoint at phase transitions
            if self.save_dir and step + 1 in [e for _, e in self._phase_boundaries]:
                self.save_checkpoint()

        # Final checkpoint
        if self.save_dir:
            self.save_checkpoint(self.save_dir / "checkpoint_final.pt")

        print()
        print("Curriculum training complete.")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curriculum training from video")
    p.add_argument("--video-dir", type=str, default=None, help="Directory of video files")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data (no videos needed)")
    p.add_argument("--total-steps", type=int, default=100_000, help="Total training steps")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--clip-duration", type=float, default=2.0, help="Video clip duration in seconds")
    p.add_argument("--clip-stride", type=float, default=1.0, help="Stride between clips")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=50)
    # Phase fraction overrides
    p.add_argument("--warmup-frac", type=float, default=0.10)
    p.add_argument("--crossmodal-frac", type=float, default=0.25)
    p.add_argument("--temporal-frac", type=float, default=0.25)
    p.add_argument("--grounding-frac", type=float, default=0.15)
    p.add_argument("--joint-frac", type=float, default=0.25)
    # Temporal context options
    p.add_argument("--enable-temporal-transformer", action="store_true",
                    help="Enable Approach 2: Hierarchical temporal transformer")
    p.add_argument("--enable-memory", action="store_true",
                    help="Enable Approach 3: Recurrent memory tokens")
    p.add_argument("--temporal-seq-layers", type=int, default=4,
                    help="Number of layers in temporal transformer")
    p.add_argument("--memory-tokens", type=int, default=8,
                    help="Number of recurrent memory tokens")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = OmniLatentConfig(
        hidden_dim=args.dim,
        num_layers=args.layers,
        num_heads=args.heads,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.total_steps,
        mixed_precision=not args.no_amp,
        gradient_checkpointing=not args.no_checkpoint,
        seed=args.seed,
        temporal_seq_layers=args.temporal_seq_layers,
        memory_num_tokens=args.memory_tokens,
    )

    print("=" * 60)
    print("OmniLatent — Curriculum Training (Learn from Video)")
    print("=" * 60)
    print(f"Device:      {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name()}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM:        {mem:.1f} GB")

    model = OmniLatentModel(config)

    # --- ADDED: Free JIT Compilation Speedup ---
    print("Compiling model with torch.compile...")
    model = torch.compile(model)
    # -------------------------------------------

    print(f"Parameters:  {count_parameters(model):,} ({param_size_mb(model):.1f} MB)")
    print(f"Hidden dim:  {config.hidden_dim}")
    print(f"Layers:      {config.num_layers}")
    print(f"Mixed prec:  {config.mixed_precision}")

    # Build temporal context modules
    temporal_transformer = None
    recurrent_memory = None

    if args.enable_temporal_transformer:
        temporal_transformer = TemporalSequenceTransformer(config)
        t_params = count_parameters(temporal_transformer)
        print(f"Temporal TX: {t_params:,} params ({t_params * 4 / 1024**2:.1f} MB)")

    if args.enable_memory:
        recurrent_memory = RecurrentMemory(config)
        m_params = count_parameters(recurrent_memory)
        print(f"Memory:      {m_params:,} params ({m_params * 4 / 1024**2:.1f} MB)")

    # Build dataset and dataloader
    if args.video_dir and not args.synthetic:
        from omnilatent.training.video_dataset import (
            VideoWatchingDataset,
            collate_video_watching,
        )

        dataset = VideoWatchingDataset(
            video_dir=args.video_dir,
            config=config,
            clip_duration=args.clip_duration,
            clip_stride=args.clip_stride,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_video_watching,
            num_workers=8,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )
        print(f"Data:        {len(dataset)} video clips from {args.video_dir}")
    else:
        if not args.synthetic:
            print("WARNING: --video-dir not provided, falling back to synthetic data")
        from omnilatent.training.data import (
            SyntheticMultiModalDataset,
            collate_multimodal,
        )

        dataset = SyntheticMultiModalDataset(config, length=10_000)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_multimodal,
            num_workers=0,
            drop_last=True,
        )
        print(f"Data:        synthetic ({len(dataset)} samples)")

    # Configure phases
    phases = [
        Phase("warmup", DEFAULT_PHASES[0].tasks, args.warmup_frac, DEFAULT_PHASES[0].description),
        Phase("cross_modal", DEFAULT_PHASES[1].tasks, args.crossmodal_frac, DEFAULT_PHASES[1].description),
        Phase("temporal", DEFAULT_PHASES[2].tasks, args.temporal_frac, DEFAULT_PHASES[2].description),
        Phase("grounding", DEFAULT_PHASES[3].tasks, args.grounding_frac, DEFAULT_PHASES[3].description),
        Phase("joint", DEFAULT_PHASES[4].tasks, args.joint_frac, DEFAULT_PHASES[4].description),
    ]

    total_frac = sum(p.frac for p in phases)
    print(f"Phases:      {len(phases)} (total frac: {total_frac:.2f})")
    for i, p in enumerate(phases):
        steps = int(args.total_steps * p.frac)
        print(f"  {i + 1}. {p.name:15s} {steps:>6d} steps  {p.tasks}")

    print(f"Total steps: {args.total_steps}")
    print(f"Save dir:    {args.save_dir}")
    print("=" * 60)

    # Build trainer
    trainer = CurriculumTrainer(
        model=model,
        config=config,
        dataloader=dataloader,
        phases=phases,
        total_steps=args.total_steps,
        save_dir=args.save_dir,
        temporal_transformer=temporal_transformer,
        recurrent_memory=recurrent_memory,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(log_interval=args.log_interval)


if __name__ == "__main__":
    main()
