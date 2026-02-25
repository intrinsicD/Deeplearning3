"""
HPWM: Hierarchical Predictive World Model - Main Model.

Combines all components into the full Phase -1 architecture:
  1. DINO backbone (frozen) -> patch features
  2. MoD Surprise Router + FWM -> surprise-weighted routing
  3. Slot Encoder -> object-level entity tokens
  4. Temporal State (Mamba SSM) -> temporal context
  5. Multi-Scale Attention -> multi-resolution representations
  7. VQ-VAE -> discrete tokenization + next-frame prediction

Component 6 (Causal Head) is OPTIONAL and not included in Phase -1.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from einops import rearrange

from hpwm.config import HPWMConfig
from hpwm.components.mod_router import MoDSurpriseRouter
from hpwm.components.slot_encoder import SlotEncoder
from hpwm.components.temporal_state import TemporalState, TransformerBaseline
from hpwm.components.multiscale_attn import MultiScaleAttention
from hpwm.components.vqvae import VQVAE


class DINOBackbone(nn.Module):
    """
    Frozen DINOv2-S/14 backbone for patch feature extraction.

    Phase -1: Fully frozen. Full spec: blocks 9-12 full-rank fine-tuning.
    """

    def __init__(self, model_name: str = "dinov2_vits14", frozen: bool = True):
        super().__init__()
        self.model_name = model_name
        self.frozen = frozen
        self._model = None  # Lazy loading

    def _load_model(self, device: torch.device):
        """Load DINOv2 model (lazy, on first forward pass).

        Tries three sources in order of stability:
        1. timm (pinned version, reproducible, works offline after first download)
        2. torch.hub (requires network, version can drift)
        3. Random fallback (for CI / testing only)
        """
        if self._model is not None:
            return

        # Map our model names to timm model names
        _TIMM_NAMES = {
            "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
            "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
            "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
        }

        loaded = False

        # Strategy 1: timm (preferred — pinned, cached, reproducible)
        try:
            import timm
            timm_name = _TIMM_NAMES.get(self.model_name)
            if timm_name is not None:
                self._model = timm.create_model(
                    timm_name, pretrained=True, num_classes=0,
                )
                loaded = True
                print(f"[DINO] Loaded via timm: {timm_name}")
        except Exception:
            pass

        # Strategy 2: torch.hub (fallback)
        if not loaded:
            try:
                self._model = torch.hub.load(
                    "facebookresearch/dinov2", self.model_name, pretrained=True,
                )
                loaded = True
                print(f"[DINO] Loaded via torch.hub: {self.model_name}")
            except Exception:
                pass

        # Strategy 3: random fallback (testing / offline)
        if not loaded:
            from hpwm.components._dino_fallback import create_dino_fallback
            self._model = create_dino_fallback()

        self._model = self._model.to(device)

        if self.frozen:
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract patch features from frames.

        Args:
            frames: [B, 3, H, W] normalized frames

        Returns:
            features: [B, N_patches, D_dino] patch features (no CLS token)
        """
        self._load_model(frames.device)

        with torch.no_grad() if self.frozen else torch.enable_grad():
            # DINOv2 returns dict with patch tokens
            out = self._model.forward_features(frames)
            if isinstance(out, dict):
                features = out["x_norm_patchtokens"]
            else:
                # Some versions return just the tensor
                # Remove CLS token (first token)
                features = out[:, 1:]

        return features


class PredictionHead(nn.Module):
    """
    Next-frame token prediction head using cross-attention.

    Instead of mean-pooling all multiscale latents into a single vector
    (which creates an information bottleneck — a 1024-d vector cannot
    recover 256 independent spatial token predictions), this head uses
    learnable spatial query tokens that cross-attend to the full set of
    multiscale latent representations.

    Each spatial query can selectively route information from the latents
    most relevant to its spatial position, preserving the structure that
    Mamba and multi-scale attention built.
    """

    def __init__(
        self,
        d_input: int,
        n_codebooks: int = 8,
        vocab_size: int = 256,
        n_spatial_tokens: int = 256,  # (128/8)^2
        hidden_dim: int = 512,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.vocab_size = vocab_size
        self.n_spatial_tokens = n_spatial_tokens
        self.hidden_dim = hidden_dim

        # Learnable spatial query tokens — one per VQ spatial position
        self.spatial_queries = nn.Parameter(
            torch.randn(1, n_spatial_tokens, hidden_dim) * 0.02,
        )

        # Cross-attention: spatial queries attend to multiscale latents
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(d_input)
        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(d_input, hidden_dim, bias=False)
        self.to_v = nn.Linear(d_input, hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.n_heads = n_heads
        self.d_head = hidden_dim // n_heads
        self.scale = self.d_head ** -0.5

        # FFN refinement after cross-attention
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.out_norm = nn.LayerNorm(hidden_dim)

        # Per-codebook classification heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size)
            for _ in range(n_codebooks)
        ])

    def forward(self, multiscale_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multiscale_output: [B, N_latents, D_input] combined scale output

        Returns:
            logits: [B, n_codebooks, N_spatial, vocab_size]
        """
        B = multiscale_output.shape[0]

        # Expand spatial queries for batch
        queries = self.spatial_queries.expand(B, -1, -1)  # [B, N_spatial, hidden]

        # Cross-attention: spatial queries attend to multiscale latents
        q = self.to_q(self.norm_q(queries))
        k = self.to_k(self.norm_kv(multiscale_output))
        v = self.to_v(self.norm_kv(multiscale_output))

        # Multi-head reshape
        q = rearrange(q, "b n (h d) -> b h n d", h=self.n_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.n_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.n_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, "b h n d -> b n (h d)")
        spatial = queries + self.to_out(out)  # [B, N_spatial, hidden]

        # FFN refinement
        spatial = spatial + self.ffn(spatial)
        spatial = self.out_norm(spatial)

        # Per-codebook predictions
        logits = torch.stack(
            [head(spatial) for head in self.heads], dim=1,
        )  # [B, n_codebooks, N_spatial, vocab_size]

        return logits


class HPWM(nn.Module):
    """
    Hierarchical Predictive World Model - Phase -1.

    Full architecture combining all components for the toy-scale
    proof of mechanism experiment.
    """

    def __init__(self, config: HPWMConfig):
        super().__init__()
        self.config = config
        self.register_buffer("_train_step", torch.tensor(0, dtype=torch.long))
        d_slot_total = config.n_slots * config.d_slot

        # Component 0: DINO backbone (frozen)
        self.dino = DINOBackbone(
            model_name=config.dino_model, frozen=config.dino_frozen,
        )

        # Component 1: MoD Surprise Router
        self.mod_router = MoDSurpriseRouter(
            d_features=config.d_dino,
            patch_grid=config.patch_grid,
            n_heavy_layers=config.n_heavy_layers,
            d_heavy=config.d_heavy,
            fwm_layers=config.fwm_layers,
            k_ratio_init=config.k_ratio_init,
            k_ratio_final=config.k_ratio_final,
            k_ratio_warmup_steps=config.k_ratio_warmup_steps,
        )

        # Component 2: Slot Encoder
        self.slot_encoder = SlotEncoder(
            d_input=config.d_dino,
            n_slots=config.n_slots,
            d_slot=config.d_slot,
            n_iters=config.slot_iters,
            mlp_hidden=config.slot_mlp_hidden,
            tome_threshold=config.tome_threshold,
        )

        # Component 4: Temporal State
        if config.use_mamba:
            self.temporal = TemporalState(
                d_input=d_slot_total,
                d_mamba=config.d_mamba,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                n_layers=config.mamba_n_layers,
            )
        else:
            self.temporal = TransformerBaseline(
                d_input=d_slot_total,
                d_model=config.d_mamba,
                n_layers=config.mamba_n_layers,
                n_heads=config.n_heads,
                context_window=config.token_budget,
            )

        # Component 5: Multi-Scale Attention
        self.multiscale = MultiScaleAttention(
            d_input=d_slot_total,
            d_fast=config.d_fast,
            d_slow=config.d_slow,
            n_latents=config.token_budget,
            n_layers_fast=config.n_layers_fast,
            n_layers_slow=config.n_layers_slow,
            n_heads=config.n_heads,
        )

        # Component 7: VQ-VAE
        self.vqvae = VQVAE(
            n_codebooks=config.vqvae_codebooks,
            vocab_size=config.vqvae_vocab_size,
            vq_dim=config.vqvae_dim,
            hidden=config.vqvae_hidden,
            n_layers=config.vqvae_n_layers,
            resolution=config.resolution,
            quantizer_type=config.vqvae_quantizer,
        )

        # Prediction head
        self.prediction_head = PredictionHead(
            d_input=d_slot_total,
            n_codebooks=config.vqvae_codebooks,
            vocab_size=config.vqvae_vocab_size,
            n_spatial_tokens=config.n_vq_tokens,
        )

        self.grad_checkpointing = config.grad_checkpointing

    def extract_dino_features(
        self, frames: torch.Tensor
    ) -> torch.Tensor:
        """Extract DINO features from frames, chunked for memory.

        Args:
            frames: [B, T, 3, H, W]

        Returns:
            features: [B, T, N_patches, D_dino]
        """
        B, T, C, H, W = frames.shape
        chunk_size = self.config.dino_chunk_size

        # DINOv2 uses 14x14 patches; input must be divisible by 14.
        # Resize to nearest compatible size (e.g. 128 -> 126 = 14*9).
        dino_h = (H // 14) * 14
        dino_w = (W // 14) * 14
        need_resize = (dino_h != H) or (dino_w != W)

        all_features = []
        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            chunk = frames[:, start:end].reshape(-1, C, H, W)

            if need_resize:
                chunk = F.interpolate(
                    chunk, size=(dino_h, dino_w),
                    mode="bilinear", align_corners=False,
                )

            if self.grad_checkpointing and self.training:
                feats = grad_checkpoint(self.dino, chunk, use_reentrant=False)
            else:
                feats = self.dino(chunk)

            # [chunk_B, N_patches, D] -> [B, chunk_T, N_patches, D]
            feats = feats.view(B, end - start, -1, self.config.d_dino)
            all_features.append(feats)

        return torch.cat(all_features, dim=1)  # [B, T, N_patches, D]

    def forward(
        self,
        frames: torch.Tensor,
        temporal_states: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Full forward pass through the HPWM architecture.

        Args:
            frames: [B, T, 3, H, W] video frames (normalized)
            temporal_states: persistent Mamba states from previous segment

        Returns:
            dict with all outputs and losses
        """
        B, T, C, H, W = frames.shape
        config = self.config

        # ── Step 1: DINO feature extraction ──────────────
        dino_features = self.extract_dino_features(frames)
        # [B, T, N_patches, D_dino]

        # ── Step 2: MoD routing (frame-by-frame) ─────────
        all_routed = []
        all_surprise = []
        all_routing_masks = []
        total_fwm_loss = torch.tensor(0.0, device=frames.device)
        total_entropy_loss = torch.tensor(0.0, device=frames.device)

        for t in range(T):
            features_t = dino_features[:, t]  # [B, N_patches, D]
            features_next = dino_features[:, t + 1] if t < T - 1 else None

            if self.grad_checkpointing and self.training and features_next is not None:
                mod_out = grad_checkpoint(
                    self.mod_router, features_t, features_next,
                    use_reentrant=False,
                )
            else:
                mod_out = self.mod_router(features_t, features_next)

            all_routed.append(mod_out["routed_features"])
            all_surprise.append(mod_out["surprise"])
            all_routing_masks.append(mod_out["routing_mask"])
            total_fwm_loss = total_fwm_loss + mod_out["fwm_loss"]
            total_entropy_loss = total_entropy_loss + mod_out["entropy_loss"]

        routed_features = torch.stack(all_routed, dim=1)       # [B, T, N_patches, D]
        surprise_maps = torch.stack(all_surprise, dim=1)       # [B, T, N_patches]
        routing_masks = torch.stack(all_routing_masks, dim=1)  # [B, T, N_patches]
        fwm_loss = total_fwm_loss / max(1, T - 1)
        entropy_loss = total_entropy_loss / T

        # ── Step 3: Slot encoding (frame-by-frame with temporal continuity) ──
        all_slots = []
        all_attn_weights = []
        prev_slots = None
        slot_consistency_loss = torch.tensor(0.0, device=frames.device)

        for t in range(T):
            features_t = routed_features[:, t]  # [B, N_patches, D]

            slot_out = self.slot_encoder(features_t, prev_slots)
            all_slots.append(slot_out["slots"])
            all_attn_weights.append(slot_out["attn_weights"])

            # Per-slot temporal smoothness: each slot's attention vector
            # should be similar to the previous frame's.  Computed per-slot
            # (dim=2 over N_patches) so the degenerate "constant map"
            # solution doesn't trivially minimise the loss — the slot
            # specialization term below prevents that.
            if t > 0:
                prev_attn = all_attn_weights[-2]  # [B, N_slots, N_patches]
                curr_attn = slot_out["attn_weights"]
                per_slot_sim = F.cosine_similarity(
                    prev_attn, curr_attn, dim=2,
                )  # [B, N_slots]
                slot_consistency_loss = slot_consistency_loss + (1.0 - per_slot_sim).mean()

            prev_slots = slot_out["slots"].detach()  # detach for memory

        slot_consistency_loss = slot_consistency_loss / max(1, T - 1)

        # Slot specialization: each slot should attend to a focused
        # subset of patches, not spread uniformly.  Low entropy per slot
        # = focused.  We penalise high entropy so slots are forced to
        # specialise, preventing the degenerate constant-attention solution.
        all_attn = torch.stack(all_attn_weights, dim=1)  # [B, T, N_slots, N_patches]
        # Normalise attention weights to be a valid distribution per slot
        slot_attn_dist = all_attn / (all_attn.sum(dim=-1, keepdim=True) + 1e-8)
        slot_entropy = -(slot_attn_dist * (slot_attn_dist + 1e-8).log()).sum(dim=-1)
        # [B, T, N_slots] — entropy per slot per frame
        max_slot_entropy = math.log(config.n_patches)
        # Normalise and average; this is >= 0 and we want to minimise it
        slot_specialization_loss = (slot_entropy / max_slot_entropy).mean()

        slot_features = torch.stack(all_slots, dim=1)  # [B, T, N_slots, D_slot]
        attn_weights = all_attn  # [B, T, N_slots, N_patches]

        # Slot diversity: penalise slots for encoding redundant information.
        # Minimises average cosine similarity between distinct slot pairs,
        # directly encouraging each slot to capture a different spatial region.
        slots_norm = F.normalize(slot_features, dim=-1)  # [B, T, N_slots, D_slot]
        BT_slots = slots_norm.flatten(0, 1)  # [B*T, N_slots, D_slot]
        slot_sim = torch.bmm(BT_slots, BT_slots.transpose(1, 2))  # [B*T, N_slots, N_slots]
        # Mask out diagonal (self-similarity = 1.0 always)
        n_s = config.n_slots
        off_diag = ~torch.eye(n_s, device=slot_sim.device, dtype=torch.bool)
        slot_diversity_loss = slot_sim[:, off_diag].mean()

        # Flatten slots for temporal processing
        slot_flat = rearrange(
            slot_features, "b t n d -> b t (n d)",
        )  # [B, T, N_slots * D_slot]

        # ── Step 4: Temporal state ───────────────────────
        # CAUSAL SPLIT: only feed context frames (0..T-2) into the
        # temporal → multiscale → prediction pipeline so that the
        # prediction for frame T-1 never sees frame T-1 features.
        context_slots = slot_flat[:, :-1]  # [B, T-1, N_slots * D_slot]

        if self.grad_checkpointing and self.training:
            temporal_output, new_states = grad_checkpoint(
                self.temporal, context_slots, temporal_states,
                use_reentrant=False,
            )
        else:
            temporal_output, new_states = self.temporal(
                context_slots, temporal_states,
            )
        # temporal_output: [B, T-1, N_slots * D_slot]

        # ── Step 5+7 note: Multi-scale attention is now called per
        # prediction position inside Step 7 below, so each temporal
        # slice gets its own causal multiscale representation.

        # ── Step 6: VQ-VAE tokenization ──────────────────
        # Encode target frames for prediction loss
        target_frames = frames[:, 1:]  # [B, T-1, 3, H, W]
        target_flat = rearrange(target_frames, "b t c h w -> (b t) c h w")

        vqvae_out = self.vqvae(target_flat)
        recon_frames, target_indices, commitment_loss, _ = vqvae_out
        # target_indices: [B*(T-1), n_codebooks, H', W']

        target_indices = target_indices.view(
            B, T - 1, config.vqvae_codebooks, -1,
        )  # [B, T-1, n_codebooks, N_spatial]

        # VQ-VAE reconstruction loss
        vqvae_recon_loss = F.mse_loss(recon_frames, target_flat)

        # ── Step 7: Next-frame prediction ────────────────
        # Predict target frame tokens at multiple temporal positions so
        # the Mamba state is supervised to retain useful information
        # throughout the sequence (not just at the last position).
        #
        # We predict at up to 4 evenly-spaced positions in the context.
        # Each prediction uses context up to that position only (causal).
        T_ctx = temporal_output.shape[1]  # T - 1
        n_pred_positions = min(4, T_ctx)
        pred_positions = [
            int(i * (T_ctx - 1) / max(1, n_pred_positions - 1))
            for i in range(n_pred_positions)
        ]
        # Deduplicate and sort
        pred_positions = sorted(set(pred_positions))

        prediction_loss = torch.tensor(0.0, device=frames.device)
        pred_logits = None  # Store the last-position logits for eval

        for pos in pred_positions:
            # Feed context up to position pos through multiscale + head
            ctx = temporal_output[:, :pos + 1]  # [B, pos+1, D]
            ms_out_pos = self.multiscale(ctx)
            logits_pos = self.prediction_head(ms_out_pos["combined"])
            # [B, n_codebooks, N_spatial, vocab_size]

            # Target: the frame at position pos in target_indices
            target_pos = target_indices[:, pos]  # [B, n_codebooks, N_spatial]

            prediction_loss = prediction_loss + F.cross_entropy(
                logits_pos.reshape(-1, config.vqvae_vocab_size),
                target_pos.reshape(-1),
            )

            if pos == pred_positions[-1]:
                pred_logits = logits_pos

        prediction_loss = prediction_loss / len(pred_positions)

        # ── Aggregate losses ─────────────────────────────
        # Two-phase prediction warmup:
        #   Phase 1 (0 → vqvae_warmup_steps): prediction off, VQ-VAE codebook stabilises
        #   Phase 2 (vqvae_warmup_steps → +pred_warmup_steps): cosine ramp 0 → full weight
        # This avoids the sudden jump that caused total loss to diverge.
        step = self._train_step.item()
        pred_ramp_end = config.vqvae_warmup_steps + config.pred_warmup_steps
        if step < config.vqvae_warmup_steps:
            pred_weight = 0.0
        elif step < pred_ramp_end:
            ramp_progress = (step - config.vqvae_warmup_steps) / config.pred_warmup_steps
            pred_weight = config.loss_weight_prediction * 0.5 * (
                1.0 - math.cos(math.pi * ramp_progress)
            )
        else:
            pred_weight = config.loss_weight_prediction

        total_loss = (
            pred_weight * prediction_loss
            + config.loss_weight_vqvae * vqvae_recon_loss
            + config.loss_weight_fwm * fwm_loss
            + config.loss_weight_commitment * commitment_loss
            + config.loss_weight_routing_entropy * entropy_loss
            + config.loss_weight_slot_consistency * slot_consistency_loss
            + config.loss_weight_slot_specialization * slot_specialization_loss
            + config.loss_weight_slot_diversity * slot_diversity_loss
        )

        return {
            "loss": total_loss,
            "prediction_loss": prediction_loss.detach(),
            "vqvae_recon_loss": vqvae_recon_loss.detach(),
            "fwm_loss": fwm_loss.detach(),
            "commitment_loss": commitment_loss.detach(),
            "entropy_loss": entropy_loss.detach(),
            "slot_consistency_loss": slot_consistency_loss.detach(),
            "slot_specialization_loss": slot_specialization_loss.detach(),
            "slot_diversity_loss": slot_diversity_loss.detach(),
            "pred_logits": pred_logits.detach(),
            "target_indices": target_indices.detach(),
            "surprise_maps": surprise_maps.detach(),
            "routing_masks": routing_masks.detach(),
            "slot_features": slot_features.detach(),
            "attn_weights": attn_weights.detach(),
            "temporal_output": temporal_output.detach(),
            "temporal_states": new_states,
        }

    def forward_inference(
        self,
        frames: torch.Tensor,
        temporal_states: list[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Inference-only forward pass (no VQ-VAE losses)."""
        with torch.no_grad():
            return self.forward(frames, temporal_states)

    def get_param_groups(self) -> list[dict]:
        """Get parameter groups with different learning rates."""
        # DINO is frozen, no params
        trainable = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append((name, param))

        # Group by component
        fwm_params = [(n, p) for n, p in trainable if "mod_router" in n]
        slot_params = [(n, p) for n, p in trainable if "slot_encoder" in n]
        temporal_params = [(n, p) for n, p in trainable if "temporal" in n]
        multiscale_params = [(n, p) for n, p in trainable if "multiscale" in n]
        vqvae_params = [(n, p) for n, p in trainable if "vqvae" in n]
        head_params = [(n, p) for n, p in trainable if "prediction_head" in n]

        return [
            {"params": [p for _, p in fwm_params], "lr_scale": 1.0, "name": "fwm"},
            {"params": [p for _, p in slot_params], "lr_scale": 1.0, "name": "slots"},
            {"params": [p for _, p in temporal_params], "lr_scale": 1.0, "name": "temporal"},
            {"params": [p for _, p in multiscale_params], "lr_scale": 1.0, "name": "multiscale"},
            {"params": [p for _, p in vqvae_params], "lr_scale": 0.5, "name": "vqvae"},
            {"params": [p for _, p in head_params], "lr_scale": 1.0, "name": "head"},
        ]

    def count_parameters(self) -> dict[str, int]:
        """Count parameters per component."""
        counts = {}
        for name, module in [
            ("mod_router", self.mod_router),
            ("slot_encoder", self.slot_encoder),
            ("temporal", self.temporal),
            ("multiscale", self.multiscale),
            ("vqvae", self.vqvae),
            ("prediction_head", self.prediction_head),
        ]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {"total": total, "trainable": trainable}

        counts["total"] = {
            "total": sum(c["total"] for c in counts.values()),
            "trainable": sum(c["trainable"] for c in counts.values()),
        }
        return counts
