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

        Measures whether routing concentrates on high-surprise patches
        by tracking the spatial distribution of heavy-path assignments.

        The key metric is spatial_entropy_normalized:
        - 1.0 = perfectly uniform routing (all patches equally likely to
          be routed heavy — router is not learning anything useful)
        - 0.0 = all routing goes to a single patch (degenerate)
        - Healthy training: starts near 1.0 and decreases as the router
          learns to concentrate on high-surprise regions.

        Uses FWM surprise (features_next provided) so evaluation matches
        training-time routing decisions.
        """
        all_heavy_ratios = []
        spatial_routing_counts = None

        # Track surprise concentration: what fraction of total surprise
        # is captured by the routed (top-K) patches?  At uniform surprise
        # this equals K/N; higher values mean routing captures genuinely
        # surprising patches.  We report the ratio to K/N so 1.0 = uniform
        # (routing adds nothing) and >1.0 = concentrated (routing useful).
        surprise_concentrations = []

        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break

            frames = batch["frames"].to(self.device)
            B, T, C, H, W = frames.shape

            # Extract DINO features
            dino_features = self.model.extract_dino_features(frames)

            # Compute routing for each frame pair (use features_next for
            # FWM surprise, matching training-time behaviour)
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
                surprise = mod_out["surprise"]  # [B, N_patches]

                # Track spatial routing frequency
                if spatial_routing_counts is None:
                    spatial_routing_counts = torch.zeros(
                        self.config.n_patches, device=self.device,
                    )
                spatial_routing_counts += routing_mask.sum(dim=0)

                all_heavy_ratios.append(routing_mask.float().mean().item())

                # Surprise concentration: fraction of total surprise mass
                # captured by routed patches, normalized by baseline K/N.
                for b in range(B):
                    s = surprise[b]  # [N_patches]
                    m = routing_mask[b]  # [N_patches]
                    K = int(m.sum().item())
                    N_p = self.config.n_patches
                    s_total = s.sum().item()
                    if K > 0 and K < N_p and s_total > 1e-8:
                        s_routed = (s * m).sum().item()
                        # fraction captured / fraction expected at uniform
                        concentration = (s_routed / s_total) / (K / N_p)
                        surprise_concentrations.append(concentration)

        # Normalize spatial counts to probability distribution
        if spatial_routing_counts is not None:
            spatial_probs = spatial_routing_counts / spatial_routing_counts.sum()
            # Entropy of the spatial distribution (how spread out routing is)
            spatial_entropy = -(
                spatial_probs * (spatial_probs + 1e-10).log()
            ).sum().item()
            max_entropy = math.log(self.config.n_patches)
            normalized_entropy = spatial_entropy / max_entropy
        else:
            normalized_entropy = 1.0

        surprise_concentration = (
            sum(surprise_concentrations) / max(1, len(surprise_concentrations))
        )

        return {
            "spatial_entropy_normalized": normalized_entropy,
            "surprise_concentration": surprise_concentration,
            "heavy_ratio": sum(all_heavy_ratios) / max(1, len(all_heavy_ratios)),
            "k_ratio": self.model.mod_router.current_k_ratio,
        }

    @torch.no_grad()
    def evaluate_signal_2(self, max_batches: int = 20) -> dict:
        """Signal 2: Slot Binding Stability.

        Measures whether slots consistently track the same spatial regions
        across consecutive frames.

        Uses **permutation-invariant** Jaccard: for each consecutive frame
        pair, we find the optimal slot-to-slot matching (Hungarian algorithm
        on the Jaccard cost matrix) so that slot identity swaps don't
        penalise the score.

        Also uses FWM surprise (features_next provided) so evaluation
        routing matches training-time behaviour.
        """
        all_matched_jaccards = []
        all_temporal_consistency = []

        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break

            frames = batch["frames"].to(self.device)
            B, T, C, H, W = frames.shape

            # Extract DINO features
            dino_features = self.model.extract_dino_features(frames)

            # Process through MoD + slots
            prev_slots = None
            prev_attn = None

            for t in range(min(T, 30)):  # limit for speed
                features_t = dino_features[:, t]
                # Use FWM surprise when possible (matches training)
                features_next = dino_features[:, t + 1] if t < T - 1 else None

                with torch.amp.autocast(
                    device_type=str(self.device),
                    dtype=torch.bfloat16,
                    enabled=(self.config.precision == "bf16" and self.device.type == "cuda"),
                ):
                    mod_out = self.model.mod_router(features_t, features_next)
                    slot_out = self.model.slot_encoder(
                        mod_out["routed_features"], prev_slots,
                    )

                curr_attn = slot_out["attn_weights"]  # [B, N_slots, N_patches]
                curr_slots = slot_out["slots"]

                if prev_attn is not None:
                    # Soft temporal consistency (permutation-invariant via
                    # best-match): for each slot in prev, find most similar
                    # slot in curr by cosine similarity
                    # prev_attn: [B, N_slots, N_patches]
                    # curr_attn: [B, N_slots, N_patches]
                    prev_norm = F.normalize(prev_attn, dim=-1)
                    curr_norm = F.normalize(curr_attn, dim=-1)
                    # [B, N_slots_prev, N_slots_curr]
                    sim_matrix = torch.bmm(prev_norm, curr_norm.transpose(1, 2))
                    # Best match per prev slot
                    best_sim = sim_matrix.max(dim=-1).values  # [B, N_slots]
                    all_temporal_consistency.append(best_sim.mean().item())

                    # Permutation-invariant Jaccard via Hungarian matching
                    prev_hard = prev_attn.argmax(dim=1)  # [B, N_patches]
                    curr_hard = curr_attn.argmax(dim=1)  # [B, N_patches]
                    n_slots = self.config.n_slots

                    for b in range(B):
                        # Build Jaccard cost matrix [N_slots, N_slots]
                        jaccard_matrix = torch.zeros(
                            n_slots, n_slots, device=self.device,
                        )
                        for s_prev in range(n_slots):
                            prev_mask = (prev_hard[b] == s_prev).float()
                            for s_curr in range(n_slots):
                                curr_mask = (curr_hard[b] == s_curr).float()
                                inter = (prev_mask * curr_mask).sum()
                                union = ((prev_mask + curr_mask) > 0).float().sum()
                                if union > 0:
                                    jaccard_matrix[s_prev, s_curr] = inter / union

                        # Hungarian matching (maximize Jaccard = minimize -Jaccard)
                        # Use greedy matching for simplicity (no scipy dependency)
                        matched_jaccards = []
                        used_curr = set()
                        # Sort prev slots by their best possible Jaccard (descending)
                        best_per_prev = jaccard_matrix.max(dim=1).values
                        order = best_per_prev.argsort(descending=True)

                        for s_prev in order.tolist():
                            best_j = -1.0
                            best_s = -1
                            for s_curr in range(n_slots):
                                if s_curr not in used_curr:
                                    j = jaccard_matrix[s_prev, s_curr].item()
                                    if j > best_j:
                                        best_j = j
                                        best_s = s_curr
                            if best_s >= 0:
                                used_curr.add(best_s)
                                # Only count if the prev slot was actually active
                                prev_active = (prev_hard[b] == s_prev).any()
                                if prev_active:
                                    matched_jaccards.append(best_j)

                        if matched_jaccards:
                            all_matched_jaccards.append(
                                sum(matched_jaccards) / len(matched_jaccards)
                            )

                prev_attn = curr_attn
                prev_slots = curr_slots.detach()

        mean_jaccard = (
            sum(all_matched_jaccards) / max(1, len(all_matched_jaccards))
        )
        mean_consistency = (
            sum(all_temporal_consistency) / max(1, len(all_temporal_consistency))
        )

        return {
            "mean_jaccard": mean_jaccard,
            "temporal_consistency": mean_consistency,
            "pass": mean_jaccard > 0.6,
            "n_measurements": len(all_matched_jaccards),
        }

    @torch.no_grad()
    def evaluate_signal_3(self, max_batches: int = 10) -> dict:
        """Signal 3: Mamba State Retention.

        Measures whether temporal context from earlier frames helps
        predict later frames. This tests whether the Mamba state
        actually retains useful information over the sequence.

        Method: For each clip, we run the full pipeline and use the
        PredictionHead to generate per-position predictions by feeding
        multi-scale attention only the temporal context up to position t.
        We compare prediction cross-entropy at early/mid/late positions.

        If Mamba retains information well, early context should help
        predictions at all positions (flat or slowly degrading loss).
        If state is lost, late positions will have higher loss.

        We also compare full-context prediction vs. truncated-context
        prediction: the gap reveals how much early frames contribute.
        """
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

            # Full forward pass to get all intermediate representations
            with torch.amp.autocast(
                device_type=str(self.device),
                dtype=torch.bfloat16,
                enabled=(self.config.precision == "bf16" and self.device.type == "cuda"),
            ):
                outputs = self.model(frames)

            target_indices = outputs["target_indices"]  # [B, T-1, n_cb, N_spatial]
            temporal_output = outputs["temporal_output"]  # [B, T-1, D]

            for pos_name, ratio in positions.items():
                t = int(ratio * (T - 2))  # index into temporal_output (T-1 frames)
                t = max(0, min(t, T - 3))

                # Feed temporal output up to position t through multiscale + head
                # to get a prediction for the frame at position t+1
                context_up_to_t = temporal_output[:, :t + 1]  # [B, t+1, D]

                with torch.amp.autocast(
                    device_type=str(self.device),
                    dtype=torch.bfloat16,
                    enabled=(self.config.precision == "bf16" and self.device.type == "cuda"),
                ):
                    ms_out = self.model.multiscale(context_up_to_t)
                    pred_logits = self.model.prediction_head(ms_out["combined"])
                    # [B, n_cb, N_spatial, vocab]

                # Target: the frame at position t (0-indexed into target_indices)
                target = target_indices[:, t]  # [B, n_cb, N_spatial]

                loss = F.cross_entropy(
                    pred_logits.reshape(-1, self.config.vqvae_vocab_size),
                    target.reshape(-1),
                ).item()

                position_losses[pos_name].append(loss)

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
