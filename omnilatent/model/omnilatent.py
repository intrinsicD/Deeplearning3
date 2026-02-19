"""OmniLatent: all-to-all multimodal model.

This is the top-level model.  It owns:
  * Modality encoders  (text, audio, image, video)
  * A shared Unified Transformer backbone
  * Modality decoders  (text, audio, image, video)
  * A HookManager for Latent Neural Hooks
  * Modality indicator embeddings

Any input modality can produce any output modality ("all-to-all") by:
  1. Encoding the source signal into latent tokens.
  2. Prepending a *source modality token* and a *target modality token*.
  3. Passing through the shared backbone.
  4. Decoding with the appropriate target decoder.
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn as nn

import torch.nn.functional as F

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
from omnilatent.utils import MODALITY_ID, Modality


class OmniLatentModel(nn.Module):
    """All-to-all multimodal model with Latent Neural Hook extensibility."""

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

        # --- Hook manager ---
        self.hook_manager = HookManager()

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
    # Sequence length adaptation for cross-modal decoding
    # ------------------------------------------------------------------
    def _expected_decoder_tokens(self, modality: Modality) -> int | None:
        """Return the expected number of tokens for a decoder, or None if
        the decoder accepts variable-length input."""
        if modality == "image":
            return self.config.image_num_patches
        if modality == "video":
            nt = self.config.video_max_frames // self.config.video_temporal_patch
            return nt * self.config.video_spatial_patches
        return None  # text and audio accept variable length

    def _adapt_seq_len(
        self, x: torch.Tensor, target_len: int
    ) -> torch.Tensor:
        """Adapt token sequence length via 1-D interpolation.

        x: (B, N, D) → (B, target_len, D).
        Uses linear interpolation in the sequence dimension, which
        preserves gradient flow.
        """
        if x.shape[1] == target_len:
            return x
        # Treat as (B, D, N) for F.interpolate, then transpose back
        x_t = x.transpose(1, 2)  # (B, D, N)
        x_t = F.interpolate(x_t, size=target_len, mode="linear", align_corners=False)
        return x_t.transpose(1, 2)  # (B, target_len, D)

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
            target_data: ground truth for the target modality (for loss
                computation during training).  None during inference.

        Returns a dict with:
            "latent":  (B, N, D) backbone output
            "output":  decoder output (shape depends on target modality)
            "target":  target_data passed through for loss computation
        """
        B = source_data.shape[0]

        # 1. Encode source
        src_tokens = self.encode(source_modality, source_data)

        # 2. Prepend target modality token
        tgt_tok = self.target_embed(
            torch.tensor(MODALITY_ID[target_modality], device=source_data.device)
        ).unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        tokens = torch.cat([tgt_tok, src_tokens], dim=1)

        # 3. Set up hooks
        if self.hook_manager.has_hooks():
            self.hook_manager.begin_forward(B)

        # 4. Backbone
        latent = self.backbone(
            tokens,
            hook_manager=self.hook_manager if self.hook_manager.has_hooks() else None,
        )

        # 5. Strip the leading modality/target tokens for decoding
        # tokens layout: [target_mod_tok, source_mod_tok, content_tokens...]
        content_latent = latent[:, 2:]  # skip target + source modality tokens

        # 6. Adapt sequence length for spatial decoders (image, video)
        expected = self._expected_decoder_tokens(target_modality)
        if expected is not None:
            content_latent = self._adapt_seq_len(content_latent, expected)

        # 7. Decode to target modality
        output = self.decoders[target_modality](content_latent)

        return {
            "latent": latent,
            "output": output,
            "target": target_data,
        }

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
        """Process multiple modalities and optionally decode to multiple targets.

        Encodes all provided modalities, concatenates their tokens, passes
        through the backbone once, then decodes to each target modality.

        This is more efficient than calling forward() multiple times because
        the backbone is shared.

        Args:
            inputs: dict mapping modality name to raw tensor.
            target_modalities: which modalities to decode to.
                Defaults to all input modalities (reconstruction).

        Returns a dict keyed by target modality name, each containing
        the decoder output.
        """
        if target_modalities is None:
            target_modalities = list(inputs.keys())

        # Encode all input modalities
        all_tokens = []
        token_ranges: dict[str, tuple[int, int]] = {}
        offset = 0
        for mod, data in inputs.items():
            enc = self.encode(mod, data)
            start = offset
            offset += enc.shape[1]
            token_ranges[mod] = (start, offset)
            all_tokens.append(enc)

        combined = torch.cat(all_tokens, dim=1)
        B = combined.shape[0]

        # Set up hooks
        if self.hook_manager.has_hooks():
            self.hook_manager.begin_forward(B)

        # Backbone
        latent = self.backbone(
            combined,
            hook_manager=self.hook_manager if self.hook_manager.has_hooks() else None,
        )

        # Decode each target
        results: dict[str, dict[str, torch.Tensor]] = {}
        for tgt_mod in target_modalities:
            # Use latent tokens corresponding to the source modality if
            # available, otherwise use the full latent sequence
            if tgt_mod in token_ranges:
                start, end = token_ranges[tgt_mod]
                # +1 to skip modality indicator token
                tgt_latent = latent[:, start + 1 : end]
            else:
                # Cross-modal generation: use all content tokens
                tgt_latent = latent[:, 1:]  # skip first modality token

            # Adapt sequence length for spatial decoders
            expected = self._expected_decoder_tokens(tgt_mod)
            if expected is not None:
                tgt_latent = self._adapt_seq_len(tgt_latent, expected)

            output = self.decoders[tgt_mod](tgt_latent)
            results[tgt_mod] = {"output": output, "latent": tgt_latent}

        return results
