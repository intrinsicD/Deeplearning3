"""OmniLatent: all-to-all multimodal model with Prefix-LM architecture.

This is the top-level model.  It owns:
  * Modality encoders  (text, audio, image, video)
  * A shared Unified Transformer backbone
  * Modality decoders  (text, audio, image, video)
  * A TargetQueryGenerator for non-text output modalities
  * A HookManager for Latent Neural Hooks
  * Modality indicator embeddings

Architecture (Prefix-LM with Learned Target Queries):
  1. Source modality is encoded into prefix tokens (bidirectional).
  2. Target tokens are appended:
     - For text: teacher-forced token embeddings (causal masking).
     - For image/audio/video: learned target queries (bidirectional).
  3. Attention mask enforces: source cannot see target; target sees
     source + itself (causally for text, bidirectionally for others).
  4. Target region of backbone output is decoded by the appropriate decoder.

This replaces the previous F.interpolate hack for cross-modal sequence
length adaptation.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from omnilatent.config import OmniLatentConfig
from omnilatent.model.backbone import UnifiedTransformer
from omnilatent.model.decoders import (
    AudioDecoder,
    ImageDecoder,
    TextDecoder,
    VideoDecoder,
)
from omnilatent.model.encoders import (
    AudioEncoder,
    ImageEncoder,
    ModalityEmbedding,
    TextEncoder,
    VideoEncoder,
)
from omnilatent.model.hooks import HookManager, LatentNeuralHook
from omnilatent.model.masking import build_prefix_lm_mask
from omnilatent.model.reasoning import LatentReasoningModule
from omnilatent.utils import MODALITY_ID, Modality


class TargetQueryGenerator(nn.Module):
    """Generates learned target queries for non-text output modalities.

    For image/video/audio targets, the model uses fixed-size learned
    query parameters (similar to DETR object queries or Perceiver
    latent arrays).  These attend to source tokens through the backbone
    and are then decoded to the target modality.
    """

    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        self.config = config
        D = config.hidden_dim

        self.image_queries = nn.Parameter(
            torch.randn(1, config.image_num_patches, D) * 0.02
        )

        num_vid_queries = (
            (config.video_max_frames // config.video_temporal_patch)
            * config.video_spatial_patches
        )
        self.video_queries = nn.Parameter(
            torch.randn(1, num_vid_queries, D) * 0.02
        )

        num_aud_queries = config.audio_max_frames // config.audio_patch_frames
        self.audio_queries = nn.Parameter(
            torch.randn(1, num_aud_queries, D) * 0.02
        )

    def forward(self, modality: str, batch_size: int) -> torch.Tensor:
        if modality == "image":
            return self.image_queries.expand(batch_size, -1, -1)
        elif modality == "video":
            return self.video_queries.expand(batch_size, -1, -1)
        elif modality == "audio":
            return self.audio_queries.expand(batch_size, -1, -1)
        raise ValueError(f"No target queries for modality: {modality}")


class OmniLatentModel(nn.Module):
    """All-to-all multimodal model with Prefix-LM architecture."""

    def __init__(self, config: OmniLatentConfig) -> None:
        super().__init__()
        self.config = config

        # --- Modality tokens (source + target indicators) ---
        self.modality_embed = ModalityEmbedding(
            num_modalities=len(MODALITY_ID), dim=config.hidden_dim
        )
        # Target modality token (separate embedding for the output target)
        self.target_embed = nn.Embedding(len(MODALITY_ID), config.hidden_dim)

        # --- Encoders ---
        self.encoders = nn.ModuleDict({
            "text": TextEncoder(config),
            "audio": AudioEncoder(config),
            "image": ImageEncoder(config),
            "video": VideoEncoder(config),
        })

        # --- Shared backbone ---
        self.backbone = UnifiedTransformer(config)

        # --- Decoders ---
        self.decoders = nn.ModuleDict({
            "text": TextDecoder(config),
            "audio": AudioDecoder(config),
            "image": ImageDecoder(config),
            "video": VideoDecoder(config),
        })

        # --- Target query generator for non-text modalities ---
        self.target_query_gen = TargetQueryGenerator(config)

        # --- Hook manager ---
        self.hook_manager = HookManager()

        # --- Latent Reasoning (Chain of Continuous Thought) ---
        self.reasoning: LatentReasoningModule | None = None
        if config.reasoning_enabled:
            self.reasoning = LatentReasoningModule(config)

        # Tie text encoder and decoder embeddings for parameter efficiency
        self.decoders["text"].head.weight = self.encoders["text"].tok_embed.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Careful initialization for stable training."""

        def _init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.ConvTranspose1d,)):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_init)

    # ------------------------------------------------------------------
    # Hook management API
    # ------------------------------------------------------------------
    def register_hook(self, hook: LatentNeuralHook) -> None:
        """Register a Latent Neural Hook with the model."""
        self.hook_manager.register_hook(hook)

    def remove_hook(self, name: str) -> LatentNeuralHook | None:
        """Remove a named hook.  Returns the hook or None."""
        return self.hook_manager.remove_hook(name)

    def list_hooks(self) -> list[str]:
        return list(self.hook_manager.hooks.keys())

    # ------------------------------------------------------------------
    # Prefix-LM attention mask (delegates to centralized masking module)
    # ------------------------------------------------------------------
    def _create_attention_mask(
        self,
        src_len: int,
        tgt_len: int,
        target_modality: str,
        device: torch.device,
    ) -> torch.Tensor:
        """Create Prefix-LM attention mask.

        Delegates to masking.build_prefix_lm_mask for centralized semantics.
        """
        return build_prefix_lm_mask(src_len, tgt_len, target_modality, device)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(
        self,
        modality: Modality,
        data: torch.Tensor,
    ) -> torch.Tensor:
        """Encode raw input into latent tokens with modality indicator.

        Returns: (B, 1 + N, D) where the leading token is the modality
        indicator.
        """
        tokens = self.encoders[modality](data)
        tokens = self.modality_embed(tokens, MODALITY_ID[modality])
        return tokens

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        source_modality: Modality,
        source_data: torch.Tensor,
        target_modality: Modality,
        target_data: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass for training or inference.

        Args:
            source_modality: which modality the input is.
            source_data: raw input tensor (shape depends on modality).
            target_modality: which modality to produce.
            target_data: ground truth for the target modality.  For text
                targets this enables teacher-forced decoding.  For non-text
                targets it's passed through for loss computation only.

        Returns a dict with:
            "latent":  (B, total_seq, D) backbone output
            "output":  decoder output (shape depends on target modality)
            "target":  target_data passed through for loss computation
            "reasoning_bottleneck": (B, D) bottleneck prediction (if reasoning enabled)
            "source_summary": (B, D) mean-pooled source latent (if reasoning enabled)
        """
        B = source_data.shape[0]
        device = source_data.device

        # 1. Encode source (includes modality indicator token)
        src_tokens = self.encode(source_modality, source_data)

        # 1b. Latent Reasoning — generate thought tokens from source
        thought_tokens = None
        bottleneck_pred = None
        source_summary = None
        if self.reasoning is not None:
            thought_tokens, bottleneck_pred = self.reasoning(src_tokens)
            # Source summary for bottleneck loss (detach source side)
            source_summary = src_tokens[:, 1:].mean(dim=1)  # skip modality indicator

        src_len = src_tokens.shape[1]
        # Include thought tokens in prefix length
        reasoning_len = thought_tokens.shape[1] if thought_tokens is not None else 0

        # 2. Generate target queries
        if target_modality == "text":
            if target_data is not None:
                # Teacher forcing: BOS + target[:-1] (shifted right)
                bos = torch.full(
                    (B, 1), self.config.text_bos_token,
                    dtype=torch.long, device=device,
                )
                tgt_input = torch.cat([bos, target_data[:, :-1]], dim=1)
                tgt_queries = self.encoders["text"](tgt_input)
            else:
                # Inference: just BOS (use generate() for full decoding)
                bos = torch.full(
                    (B, 1), self.config.text_bos_token,
                    dtype=torch.long, device=device,
                )
                tgt_queries = self.encoders["text"](bos)
        else:
            tgt_queries = self.target_query_gen(target_modality, B)

        # 3. Prepend target modality token to queries
        tgt_mod_tok = self.target_embed(
            torch.tensor(MODALITY_ID[target_modality], device=device)
        ).unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        tgt_with_mod = torch.cat([tgt_mod_tok, tgt_queries], dim=1)
        tgt_len = tgt_with_mod.shape[1]

        # 4. Concatenate: [source, (thoughts), target]
        # Thought tokens become part of the prefix — target attends to them
        prefix_parts = [src_tokens]
        if thought_tokens is not None:
            prefix_parts.append(thought_tokens)
        prefix = torch.cat(prefix_parts, dim=1)
        prefix_len = prefix.shape[1]

        tokens = torch.cat([prefix, tgt_with_mod], dim=1)

        # 5. Create Prefix-LM attention mask
        # prefix_len = src_len + reasoning_len (thoughts are part of prefix)
        attn_mask = self._create_attention_mask(
            prefix_len, tgt_len, target_modality, device,
        )

        # 6. Set up hooks
        if self.hook_manager.has_hooks():
            self.hook_manager.begin_forward(B)

        # 7. Backbone
        latent = self.backbone(
            tokens,
            attn_mask=attn_mask,
            hook_manager=self.hook_manager if self.hook_manager.has_hooks() else None,
            prefix_len=prefix_len,
        )

        # 8. Extract target region (skip prefix + target modality token)
        tgt_latent = latent[:, prefix_len + 1:]  # skip tgt_mod_tok

        # 9. Decode to target modality
        output = self.decoders[target_modality](tgt_latent)

        result = {
            "latent": latent,
            "output": output,
            "target": target_data,
        }

        # Include reasoning outputs for auxiliary loss computation
        if bottleneck_pred is not None:
            result["reasoning_bottleneck"] = bottleneck_pred
            result["source_summary"] = source_summary

        return result

    # ------------------------------------------------------------------
    # Autoregressive text generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        source_modality: Modality,
        source_data: torch.Tensor,
        max_len: int = 50,
    ) -> torch.Tensor:
        """Autoregressive text generation from any source modality.

        Returns: (B, max_len) long tensor of generated token IDs.
        """
        B = source_data.shape[0]
        device = source_data.device

        # Encode source once
        src_tokens = self.encode(source_modality, source_data)

        # Run reasoning once (thoughts are part of the prefix)
        prefix_parts = [src_tokens]
        if self.reasoning is not None:
            thought_tokens, _ = self.reasoning(src_tokens)
            prefix_parts.append(thought_tokens)
        prefix = torch.cat(prefix_parts, dim=1)
        prefix_len = prefix.shape[1]

        # Start with BOS
        generated_ids = torch.full(
            (B, 1), self.config.text_bos_token,
            dtype=torch.long, device=device,
        )

        for _ in range(max_len):
            # Embed current generated tokens
            tgt_queries = self.encoders["text"](generated_ids)

            # Target modality token
            tgt_mod_tok = self.target_embed(
                torch.tensor(MODALITY_ID["text"], device=device)
            ).unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            tgt_with_mod = torch.cat([tgt_mod_tok, tgt_queries], dim=1)
            tgt_len = tgt_with_mod.shape[1]

            tokens = torch.cat([prefix, tgt_with_mod], dim=1)
            attn_mask = self._create_attention_mask(
                prefix_len, tgt_len, "text", device,
            )

            latent = self.backbone(tokens, attn_mask=attn_mask)
            # Get logits for the last target position
            logits = self.decoders["text"](latent[:, -1:])  # (B, 1, V)
            next_token = logits.argmax(dim=-1)  # (B, 1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

        return generated_ids[:, 1:]  # strip BOS

    # ------------------------------------------------------------------
    # Convenience: self-reconstruction (encode-decode same modality)
    # ------------------------------------------------------------------
    def reconstruct(
        self,
        modality: Modality,
        data: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Shortcut for same-modality reconstruction (autoencoder mode)."""
        return self.forward(modality, data, modality, data)

    # ------------------------------------------------------------------
    # Multi-modal forward (multiple inputs → multiple outputs)
    # ------------------------------------------------------------------
    def forward_multimodal(
        self,
        inputs: dict[Modality, torch.Tensor],
        target_modalities: Sequence[Modality] | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Process multiple input modalities and decode to multiple targets.

        For each target, selects the matching source modality if available,
        otherwise uses the first available source.  Runs a separate backbone
        pass per target (required because different targets need different
        attention masks and query configurations).
        """
        if target_modalities is None:
            target_modalities = list(inputs.keys())

        results: dict[str, dict[str, torch.Tensor]] = {}
        for tgt_mod in target_modalities:
            # Pick source: prefer same modality, else first available
            if tgt_mod in inputs:
                src_mod = tgt_mod
            else:
                src_mod = next(iter(inputs))

            result = self.forward(
                src_mod, inputs[src_mod], tgt_mod,
                target_data=inputs.get(tgt_mod),
            )
            results[tgt_mod] = {
                "output": result["output"],
                "latent": result["latent"],
            }

        return results
