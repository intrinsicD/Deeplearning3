"""Plugin wrapping :class:`MMWM.trainer.Trainer`.

MMWM's trainer expects a multimodal batch dict with ``*_t`` / ``*_tp1`` /
``action`` keys. We reuse :class:`MMWM.data.DeterministicTransitionDataset`
to manufacture synthetic batches that exercise the full forward graph
without needing real video.
"""

from __future__ import annotations

import tempfile
from typing import Any

import torch
import torch.nn as nn

from MMWM.config import ModelConfig, build_model
from MMWM.data import DeterministicTransitionDataset, collate_transition_batch
from MMWM.losses import LossWeights, WorldModelLoss
from MMWM.trainer import Trainer as _MMWMTrainer

from scripts.training.self_improve.plugins.base import (
    ComponentPlugin,
    EvalReport,
    StepReport,
)


def _tiny_config() -> ModelConfig:
    """A small MMWM ModelConfig — keeps default registry names but shrinks
    the encoder/decoder/latent/memory dimensions for fast tests.

    The defaults already use ``simple_multimodal`` encoder + ``role_split``
    prediction head, so we only need to override sizes.
    """
    # Default ModelConfig is small enough; just return as-is.
    return ModelConfig()


class MMWMPlugin(ComponentPlugin):
    name = "mmwm"

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        *,
        lr: float = 3e-4,
        device: torch.device | None = None,
        learned_uncertainty: bool = False,
    ) -> None:
        self.model_config = model_config or _tiny_config()
        device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = build_model(self.model_config, skip_validation=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = WorldModelLoss(
            weights=LossWeights(),
            learned_uncertainty=learned_uncertainty,
        )

        # Trainer needs a writable run_dir for its TensorBoard SummaryWriter.
        # Use a tempdir; nothing in phase 1 reads from it.
        self._run_dir = tempfile.mkdtemp(prefix="mmwm_plugin_")
        self._trainer = _MMWMTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            run_dir=self._run_dir,
            mixed_precision=False,
        )
        self._memory_state = None

        # Pre-built synthetic dataset for make_synthetic_batch.
        encoder_kwargs = self.model_config.encoder_kwargs
        self._synth_dataset = DeterministicTransitionDataset(
            length=64,
            vector_dim=encoder_kwargs.get("vector_input_dim", 128),
            action_dim=self.model_config.action_encoder_kwargs.get("action_dim", 32),
            include_text=False,
            include_image=False,
            include_audio=False,
        )

    # -- required surface --------------------------------------------------

    @property
    def model(self) -> nn.Module:
        return self._trainer.model

    @property
    def device(self) -> torch.device:
        return self._trainer.device

    def make_synthetic_batch(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        items = [self._synth_dataset[i] for i in range(batch_size)]
        return collate_transition_batch(items)

    def train_step(self, batch: dict[str, Any]) -> StepReport:
        losses, memory_state = self._trainer.train_step(batch, self._memory_state)
        if not self._trainer.reset_memory_each_batch:
            self._memory_state = memory_state
        total = float(losses.get("total_loss", 0.0))
        clean = {k: float(v) for k, v in losses.items() if isinstance(v, (int, float))}
        return StepReport(loss=total, losses=clean)

    @torch.no_grad()
    def evaluate(self, probe_set: Any | None = None) -> EvalReport:
        """Score on a probe set: total world-model loss + latent dynamics loss.

        Phase 2 reuses the same loss function the trainer uses, so the
        eval signal is faithful to training; later phases swap in
        multi-step rollout MSE/cosine per the design doc §6.1.
        """
        if probe_set is None:
            from scripts.training.self_improve.eval_registry import build_mmwm_probe
            probe_set = build_mmwm_probe()

        model = self._trainer.model
        was_training = model.training
        model.eval()
        try:
            total_sum = 0.0
            dyn_sum = 0.0
            n = 0
            for batch in probe_set:
                obs_t = self._trainer._to_packet(batch, "t")
                obs_tp1 = self._trainer._to_packet(batch, "tp1")
                action = batch["action"].to(self._trainer.device)
                decoder_context, decode_keys = self._trainer._build_decoder_plan(batch)
                output = model(
                    obs_t, action,
                    obs_tp1=obs_tp1,
                    memory_state=None,
                    decoder_context=decoder_context,
                    decode_keys=decode_keys,
                )
                loss_inputs: dict[str, Any] = {}
                for k in ("text_target", "vector_target", "image_target", "audio_target"):
                    if k in batch:
                        loss_inputs[k] = batch[k].to(self._trainer.device)
                losses = self._trainer.loss_fn(output, loss_inputs, task_multipliers=None)
                bs = action.shape[0]
                total_sum += float(losses["total_loss"].item()) * bs
                if "latent_dyn_loss" in losses:
                    dyn_sum += float(losses["latent_dyn_loss"].item()) * bs
                n += bs
        finally:
            model.train(was_training)

        total = total_sum / max(n, 1)
        dyn = dyn_sum / max(n, 1)
        return EvalReport(
            score=total,
            metrics={"total_loss": total, "latent_dyn_loss": dyn},
            higher_is_better=False,
        )

    # -- state -------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self._trainer.model.state_dict(),
            "optimizer": self._trainer.optimizer.state_dict(),
            "loss_fn": self._trainer.loss_fn.state_dict(),
            "global_step": self._trainer.global_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._trainer.model.load_state_dict(state["model"])
        if "optimizer" in state:
            self._trainer.optimizer.load_state_dict(state["optimizer"])
        if "loss_fn" in state:
            self._trainer.loss_fn.load_state_dict(state["loss_fn"])
        self._trainer.global_step = int(state.get("global_step", 0))


__all__ = ["MMWMPlugin"]
