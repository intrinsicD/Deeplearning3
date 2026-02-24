"""
HPWM Phase -1 Evaluation: Three Validation Signals.

Signal 1: MoD Routing Entropy
    - Entropy of routing weight distribution over spatial patches
    - Pass: Entropy decreases over training; routing concentrates on
      high-surprise patches

Signal 2: Slot Binding Stability
    - Jaccard overlap between slot assignment maps for the same physical
      object across consecutive frames
    - Pass: Jaccard > 0.6 and increasing over training

Signal 3: Mamba State Retention
    - Next-frame prediction accuracy as function of elapsed clip time
    - Compare HPWM (Mamba state) vs flat Transformer baseline
    - Pass: HPWM degrades gradually; Transformer drops sharply past context window

All three must pass. Passing two of three is not a pass.
"""

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from einops import rearrange

from hpwm.config import HPWMConfig


class Evaluator:
    """Evaluates the three Phase -1 validation signals."""

    def __init__(
        self,
        config: HPWMConfig,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.config = config
        self.model = model
        self.val_loader = val_loader
        self.device = device

    @torch.no_grad()
    def evaluate_all(self, max_batches: int = 20) -> dict:
        """Run all three validation signals plus validation loss.

        Args:
            max_batches: max validation batches to process

        Returns:
            dict with all evaluation results
        """
        self.model.eval()

        results = {}
        results["signal_1_routing_entropy"] = self.evaluate_signal_1(max_batches)
        results["signal_2_slot_stability"] = self.evaluate_signal_2(max_batches)
        results["signal_3_state_retention"] = self.evaluate_signal_3(max_batches)
        results["val_loss"] = self.evaluate_val_loss(max_batches)

        return results

    @torch.no_grad()
    def evaluate_signal_1(self, max_batches: int = 20) -> dict:
        """Signal 1: MoD Routing Entropy.

        Measures whether routing concentrates on high-surprise patches.
        """
        all_entropies = []
        all_heavy_ratios = []
        spatial_routing_counts = None

        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break

            frames = batch["frames"].to(self.device)
            B, T, C, H, W = frames.shape

            # Extract DINO features
            dino_features = self.model.extract_dino_features(frames)

            # Compute routing for each frame pair
            for t in range(min(T - 1, 20)):  # limit frames for speed
                features_t = dino_features[:, t]
                features_next = dino_features[:, t + 1]

                with torch.amp.autocast(
                    device_type=str(self.device),
                    dtype=torch.bfloat16,
                    enabled=(self.config.precision == "bf16" and self.device.type == "cuda"),
                ):
                    mod_out = self.model.mod_router(features_t, features_next)

                routing_mask = mod_out["routing_mask"]  # [B, N_patches]
                surprise = mod_out["surprise"]

                # Track spatial routing frequency
                if spatial_routing_counts is None:
                    spatial_routing_counts = torch.zeros(
                        self.config.n_patches, device=self.device,
                    )
                spatial_routing_counts += routing_mask.sum(dim=0)

                # Per-batch metrics
                metrics = self.model.mod_router.get_sparsity_metrics(routing_mask)
                all_entropies.append(metrics["routing_entropy"])
                all_heavy_ratios.append(metrics["heavy_ratio"])

        # Normalize spatial counts to probability
        if spatial_routing_counts is not None:
            spatial_probs = spatial_routing_counts / spatial_routing_counts.sum()
            # Entropy of the spatial distribution
            spatial_entropy = -(
                spatial_probs * (spatial_probs + 1e-10).log()
            ).sum().item()
            max_entropy = math.log(self.config.n_patches)
            normalized_entropy = spatial_entropy / max_entropy
        else:
            normalized_entropy = 1.0

        return {
            "entropy": sum(all_entropies) / max(1, len(all_entropies)),
            "spatial_entropy_normalized": normalized_entropy,
            "heavy_ratio": sum(all_heavy_ratios) / max(1, len(all_heavy_ratios)),
            "k_ratio": self.model.mod_router.current_k_ratio,
        }

    @torch.no_grad()
    def evaluate_signal_2(self, max_batches: int = 20) -> dict:
        """Signal 2: Slot Binding Stability.

        Measures Jaccard overlap between slot assignments for the same
        object across consecutive frames.

        Uses slot attention weights as soft assignment maps.
        For synthetic data: compares against ground-truth object masks.
        """
        all_jaccards = []
        all_temporal_consistency = []

        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break

            frames = batch["frames"].to(self.device)
            masks = batch.get("masks")  # [B, T, N_objects, H, W] if available
            B, T, C, H, W = frames.shape

            # Extract DINO features
            dino_features = self.model.extract_dino_features(frames)

            # Process through MoD + slots
            prev_slots = None
            prev_attn = None

            for t in range(min(T, 30)):  # limit for speed
                features_t = dino_features[:, t]

                with torch.amp.autocast(
                    device_type=str(self.device),
                    dtype=torch.bfloat16,
                    enabled=(self.config.precision == "bf16" and self.device.type == "cuda"),
                ):
                    mod_out = self.model.mod_router(features_t, None)
                    slot_out = self.model.slot_encoder(
                        mod_out["routed_features"], prev_slots,
                    )

                curr_attn = slot_out["attn_weights"]  # [B, N_slots, N_patches]
                curr_slots = slot_out["slots"]

                if prev_attn is not None:
                    # Temporal consistency: cosine similarity between
                    # consecutive slot attention maps
                    sim = F.cosine_similarity(
                        prev_attn.flatten(2), curr_attn.flatten(2), dim=2,
                    )  # [B, N_slots]
                    all_temporal_consistency.append(sim.mean().item())

                    # Jaccard overlap using hard assignments
                    prev_hard = prev_attn.argmax(dim=1)  # [B, N_patches]
                    curr_hard = curr_attn.argmax(dim=1)  # [B, N_patches]

                    for s in range(self.config.n_slots):
                        prev_mask = (prev_hard == s).float()  # [B, N_patches]
                        curr_mask = (curr_hard == s).float()  # [B, N_patches]

                        intersection = (prev_mask * curr_mask).sum(dim=-1)          # [B]
                        union = ((prev_mask + curr_mask) > 0).float().sum(dim=-1)   # [B]

                        # Only include samples where slot s is active in at least
                        # one of the two frames; samples with union==0 produce 0/0
                        # and must be excluded, not counted as zero Jaccard.
                        valid = union > 0  # [B] bool
                        if valid.any():
                            jaccard = (intersection[valid] / union[valid]).mean().item()
                            all_jaccards.append(jaccard)

                prev_attn = curr_attn
                prev_slots = curr_slots.detach()

        mean_jaccard = sum(all_jaccards) / max(1, len(all_jaccards))
        mean_consistency = (
            sum(all_temporal_consistency) / max(1, len(all_temporal_consistency))
        )

        return {
            "mean_jaccard": mean_jaccard,
            "temporal_consistency": mean_consistency,
            "pass": mean_jaccard > 0.6,
            "n_measurements": len(all_jaccards),
        }

    @torch.no_grad()
    def evaluate_signal_3(self, max_batches: int = 10) -> dict:
        """Signal 3: Mamba State Retention.

        Measures next-frame prediction accuracy as a function of
        elapsed clip time. For Phase -1, we measure at different
        temporal positions within the same clip.

        Returns per-position accuracy to show how prediction quality
        changes with clip length.
        """
        # Evaluate at different temporal positions
        positions = {
            "early_10pct": 0.1,
            "mid_50pct": 0.5,
            "late_90pct": 0.9,
        }

        position_losses = defaultdict(list)

        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break

            frames = batch["frames"].to(self.device)
            B, T, C, H, W = frames.shape

            # Full forward pass
            with torch.amp.autocast(
                device_type=str(self.device),
                dtype=torch.bfloat16,
                enabled=(self.config.precision == "bf16" and self.device.type == "cuda"),
            ):
                outputs = self.model(frames)

            # Get VQ-VAE tokens for all frames
            target_indices = outputs["target_indices"]  # [B, T-1, n_cb, N_spatial]

            # Get prediction accuracy at different temporal positions
            pred_logits = outputs["pred_logits"]  # [B, n_cb, N_spatial, vocab]

            # The model predicts the last frame's tokens from accumulated context
            # To measure retention at different positions, we need per-frame predictions
            # For now, measure the reconstruction quality of the temporal state
            temporal_output = outputs["temporal_output"]  # [B, T, D]

            for pos_name, ratio in positions.items():
                t = int(ratio * (T - 1))
                t = max(0, min(t, T - 2))

                # Use temporal features at position t to measure how much
                # information is retained from earlier frames
                features_at_t = temporal_output[:, t]  # [B, D]
                features_at_0 = temporal_output[:, 0]   # [B, D]

                # Cosine similarity between early and later features
                # Lower similarity at later positions = more information loss
                if t > 0:
                    sim = F.cosine_similarity(
                        features_at_0, features_at_t, dim=-1,
                    ).mean().item()
                else:
                    sim = 1.0

                position_losses[pos_name].append(sim)

        results = {}
        for pos_name, losses in position_losses.items():
            results[pos_name] = sum(losses) / max(1, len(losses))

        return results

    @torch.no_grad()
    def evaluate_val_loss(self, max_batches: int = 20) -> float:
        """Compute average validation loss."""
        total_loss = 0.0
        n_batches = 0

        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break

            frames = batch["frames"].to(self.device)

            with torch.amp.autocast(
                device_type=str(self.device),
                dtype=torch.bfloat16,
                enabled=(self.config.precision == "bf16" and self.device.type == "cuda"),
            ):
                outputs = self.model(frames)

            total_loss += outputs["loss"].item()
            n_batches += 1

        return total_loss / max(1, n_batches)


def compare_mamba_vs_transformer(
    config: HPWMConfig,
    mamba_checkpoint: str,
    transformer_checkpoint: str | None = None,
    device: str = "cuda",
) -> dict:
    """
    Compare Mamba state retention vs Transformer baseline.

    This is the key Signal 3 comparison. If the Mamba model does not
    show better retention at longer temporal positions, the primary
    HPWM hypothesis is not supported.

    Args:
        config: HPWMConfig
        mamba_checkpoint: path to trained Mamba model checkpoint
        transformer_checkpoint: path to trained Transformer baseline
        device: device to run on

    Returns:
        dict with comparison results
    """
    from hpwm.data import create_dataloaders, ConcatenatedClipDataset, SyntheticMovingShapes

    device = torch.device(device)

    # Load Mamba model
    config_mamba = HPWMConfig()
    config_mamba.use_mamba = True
    model_mamba = HPWM(config_mamba).to(device)
    ckpt = torch.load(mamba_checkpoint, map_location=device, weights_only=False)
    model_sd = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("dino._model.")}
    model_mamba.load_state_dict(model_sd, strict=False)
    model_mamba.eval()

    # Load Transformer baseline (or create one)
    config_tf = HPWMConfig()
    config_tf.use_mamba = False
    model_tf = HPWM(config_tf).to(device)
    if transformer_checkpoint:
        ckpt = torch.load(transformer_checkpoint, map_location=device, weights_only=False)
        model_sd = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("dino._model.")}
        model_tf.load_state_dict(model_sd, strict=False)
    model_tf.eval()

    # Create concatenated clip dataset for different lengths
    base_dataset = SyntheticMovingShapes(
        n_clips=100, n_frames=config.n_frames,
        resolution=config.resolution, seed=999,
    )
    concat_dataset = ConcatenatedClipDataset(
        base_dataset, target_lengths_s=[10, 30, 60], fps=config.fps,
    )

    results = {"mamba": {}, "transformer": {}}

    loader = DataLoader(concat_dataset, batch_size=1, shuffle=False)

    for batch in loader:
        frames = batch["frames"].to(device)
        target_s = batch["target_length_s"].item()

        with torch.no_grad():
            # Mamba forward
            out_mamba = model_mamba(frames)
            loss_mamba = out_mamba["loss"].item()

            # Transformer forward
            out_tf = model_tf(frames)
            loss_tf = out_tf["loss"].item()

        key = f"{target_s}s"
        results["mamba"].setdefault(key, []).append(loss_mamba)
        results["transformer"].setdefault(key, []).append(loss_tf)

    # Average results
    summary = {}
    for model_name in ["mamba", "transformer"]:
        for length_key, losses in results[model_name].items():
            avg = sum(losses) / len(losses)
            summary[f"{model_name}_{length_key}"] = avg

    return summary
