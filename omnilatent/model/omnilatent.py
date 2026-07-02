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

from typing import Any, Sequence, cast

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
from omnilatent.model.masking import apply_token_validity_mask, build_prefix_lm_mask
from omnilatent.model.reasoning import LatentReasoningModule
from omnilatent.protocol import ObservationPacket, TargetSpec
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
        # Source segment IDs distinguish multiple observed modalities inside a
        # single fused prefix. Existing single-source forward() does not use
        # this embedding, preserving legacy behavior.
        self.source_segment_embed = nn.Embedding(16, config.hidden_dim)

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
        try:
            device = next(self.parameters()).device
            hook.to(device)
        except StopIteration:  # pragma: no cover - model always has params
            pass
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

    def _build_target_tokens(
        self,
        target_modality: str,
        batch_size: int,
        device: torch.device,
        target_data: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build target modality token + target queries/teacher-forced tokens."""
        if target_modality not in self.decoders:
            raise ValueError(f"Unsupported target modality: {target_modality}")
        if target_modality == "text":
            if target_data is not None:
                bos = torch.full(
                    (batch_size, 1), self.config.text_bos_token,
                    dtype=torch.long, device=device,
                )
                tgt_input = torch.cat([bos, target_data[:, :-1]], dim=1)
                tgt_queries = self.encoders["text"](tgt_input)
            else:
                bos = torch.full(
                    (batch_size, 1), self.config.text_bos_token,
                    dtype=torch.long, device=device,
                )
                tgt_queries = self.encoders["text"](bos)
        else:
            tgt_queries = self.target_query_gen(target_modality, batch_size)

        tgt_mod_tok = self.target_embed(
            torch.tensor(MODALITY_ID[target_modality], device=device)
        ).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        return torch.cat([tgt_mod_tok, tgt_queries], dim=1)

    def _normalize_source_mask(
        self,
        mask: torch.Tensor | None,
        encoded_len: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalize per-modality masks to include the prepended modality token."""
        if mask is None:
            return torch.ones(batch_size, encoded_len, dtype=torch.bool, device=device)
        mask = mask.to(device=device, dtype=torch.bool)
        if mask.ndim != 2 or mask.shape[0] != batch_size:
            raise ValueError(
                f"ObservationPacket masks must be [B, T], got {tuple(mask.shape)} "
                f"for batch_size={batch_size}"
            )
        if mask.shape[1] == encoded_len:
            return mask
        if mask.shape[1] == encoded_len - 1:
            leading = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            return torch.cat([leading, mask], dim=1)
        raise ValueError(
            f"Mask length {mask.shape[1]} must match encoded length {encoded_len} "
            f"or encoded length without modality token {encoded_len - 1}"
        )

    def _run_prefix_target(
        self,
        prefix: torch.Tensor,
        target_modality: str,
        target_data: torch.Tensor | None,
        prefix_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Shared fused-prefix → target decode path."""
        B = prefix.shape[0]
        device = prefix.device

        thought_tokens = None
        bottleneck_pred = None
        source_summary = None
        if self.reasoning is not None:
            thought_tokens, bottleneck_pred = self.reasoning(prefix)
            source_summary = prefix.mean(dim=1)

        prefix_parts = [prefix]
        if thought_tokens is not None:
            thought_tokens = cast(torch.Tensor, thought_tokens)
            prefix_parts.append(thought_tokens)
            if prefix_valid_mask is not None:
                thought_valid = torch.ones(B, thought_tokens.shape[1], dtype=torch.bool, device=device)
                prefix_valid_mask = torch.cat([prefix_valid_mask, thought_valid], dim=1)
        prefix_with_thoughts = torch.cat(prefix_parts, dim=1)
        prefix_len = prefix_with_thoughts.shape[1]

        tgt_with_mod = self._build_target_tokens(target_modality, B, device, target_data)
        tgt_len = tgt_with_mod.shape[1]
        tokens = torch.cat([prefix_with_thoughts, tgt_with_mod], dim=1)

        attn_mask = self._create_attention_mask(prefix_len, tgt_len, target_modality, device)
        if prefix_valid_mask is not None:
            target_valid = torch.ones(B, tgt_len, dtype=torch.bool, device=device)
            valid_tokens = torch.cat([prefix_valid_mask, target_valid], dim=1)
            attn_mask = apply_token_validity_mask(attn_mask, valid_tokens)

        if self.hook_manager.has_hooks():
            self.hook_manager.begin_forward(B)

        latent = self.backbone(
            tokens,
            attn_mask=attn_mask,
            hook_manager=self.hook_manager if self.hook_manager.has_hooks() else None,
            prefix_len=prefix_len,
        )
        tgt_latent = latent[:, prefix_len + 1:]
        output = self.decoders[target_modality](tgt_latent)
        result = {
            "latent": latent,
            "output": output,
            "target": target_data,
            "attention_mask": attn_mask,
            "prefix_len": torch.tensor(prefix_len, device=device),
        }
        if bottleneck_pred is not None:
            result["reasoning_bottleneck"] = bottleneck_pred
            result["source_summary"] = source_summary
        return result

    def forward_observation(
        self,
        packet: ObservationPacket,
        target: TargetSpec,
    ) -> dict[str, Any]:
        """Forward pass from multiple observed modalities fused in one prefix.

        Layout:
            ``[source_0 tokens][source_1 tokens]...[thoughts][target token + queries]``

        Each source segment receives a learned source-segment embedding in
        addition to its modality token. Per-modality masks in
        ``packet.masks`` are normalized to encoded-token length and applied to
        the final attention mask.
        """
        if not packet.modalities:
            raise ValueError("ObservationPacket contains no modalities")
        target_modality = target.modality
        if target_modality not in self.decoders:
            raise ValueError(f"Unsupported target modality: {target_modality}")

        prefix_parts: list[torch.Tensor] = []
        mask_parts: list[torch.Tensor] = []
        segment_parts: list[torch.Tensor] = []
        source_modalities: list[str] = []
        source_lengths: list[int] = []
        batch_size: int | None = None
        device: torch.device | None = None

        for segment_idx, (modality, data) in enumerate(packet.modalities.items()):
            if modality not in self.encoders:
                raise ValueError(f"Unsupported source modality: {modality}")
            if batch_size is None:
                batch_size = data.shape[0]
                device = data.device
            elif data.shape[0] != batch_size:
                raise ValueError("All ObservationPacket modalities must share the same batch size")
            if device is None:
                device = data.device

            encoded = self.encode(cast(Modality, modality), data)
            encoded_device = cast(torch.device, encoded.device)
            seg_id = segment_idx % self.source_segment_embed.num_embeddings
            seg = self.source_segment_embed(
                torch.tensor(seg_id, device=encoded_device)
            ).view(1, 1, -1)
            encoded = encoded + seg
            prefix_parts.append(encoded)

            mask_parts.append(
                self._normalize_source_mask(
                    packet.masks.get(modality), encoded.shape[1], data.shape[0], encoded_device,
                )
            )
            segment_parts.append(
                torch.full((data.shape[0], encoded.shape[1]), seg_id, dtype=torch.long, device=encoded_device)
            )
            source_modalities.append(modality)
            source_lengths.append(encoded.shape[1])

        assert batch_size is not None and device is not None
        prefix = torch.cat(prefix_parts, dim=1)
        prefix_valid_mask = torch.cat(mask_parts, dim=1)
        source_segment_ids = torch.cat(segment_parts, dim=1)

        result = self._run_prefix_target(
            prefix=prefix,
            target_modality=target_modality,
            target_data=target.data,
            prefix_valid_mask=prefix_valid_mask,
        )
        result["source_segment_ids"] = source_segment_ids
        result["source_valid_mask"] = prefix_valid_mask
        result["source_modalities"] = source_modalities  # type: ignore[assignment]
        result["source_lengths"] = source_lengths  # type: ignore[assignment]
        return result

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        source_modality: Modality,
        source_data: torch.Tensor,
        target_modality: Modality,
        target_data: torch.Tensor | None = None,
    ) -> dict[str, Any]:
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
            thought_tokens = cast(torch.Tensor, thought_tokens)

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
        eos_token: int | None = None,
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
        if eos_token is None:
            eos_token = getattr(self.config, "text_eos_token", None)
        pad_token = getattr(self.config, "text_pad_token", 0)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

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

            # Use the SAME hook-aware backbone path as forward(); otherwise
            # trained Latent Neural Hooks are silently ignored at generation
            # time (Audit.md A9).
            if self.hook_manager.has_hooks():
                self.hook_manager.begin_forward(B)
            latent = self.backbone(
                tokens,
                attn_mask=attn_mask,
                hook_manager=self.hook_manager if self.hook_manager.has_hooks() else None,
                prefix_len=prefix_len,
            )
            # Get logits for the last target position
            logits = self.decoders["text"](latent[:, -1:])  # (B, 1, V)
            next_token = logits.argmax(dim=-1)  # (B, 1)
            if eos_token is not None:
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, pad_token),
                    next_token,
                )
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            if eos_token is not None:
                finished |= next_token.squeeze(1).eq(eos_token)
                if bool(finished.all()):
                    break

        output = generated_ids[:, 1:]  # strip BOS
        if output.shape[1] < max_len:
            pad = torch.full(
                (B, max_len - output.shape[1]),
                pad_token,
                dtype=torch.long,
                device=device,
            )
            output = torch.cat([output, pad], dim=1)
        return output

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

        All provided inputs are **fused** into a single prefix (via
        :meth:`forward_observation`) so every target attends to every input —
        rather than the previous behaviour of picking one source and ignoring
        the rest (Audit.md A9). Runs one backbone pass per target because
        different targets need different attention masks and query
        configurations.
        """
        if not inputs:
            raise ValueError("forward_multimodal requires at least one input modality")
        if target_modalities is None:
            target_modalities = list(inputs.keys())

        packet = ObservationPacket(modalities=dict(inputs))

        results: dict[str, dict[str, torch.Tensor]] = {}
        for tgt_mod in target_modalities:
            result = self.forward_observation(
                packet,
                TargetSpec(modality=tgt_mod, data=inputs.get(tgt_mod)),
            )
            results[tgt_mod] = {
                "output": result["output"],
                "latent": result["latent"],
                "source_modalities": result["source_modalities"],
            }

        return results
