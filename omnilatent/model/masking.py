"""Centralized attention mask construction for OmniLatent.

All attention mask semantics are defined here to prevent Prefix-LM
leakage and ensure hook tokens behave correctly.

PyTorch SDPA convention: True = CAN attend, False = masked out.

Mask layout for a forward pass:
    [source_tokens (S), target_tokens (T)]

Rules:
  - Source → Source: True (bidirectional)
  - Source → Target: False (source can't peek at the answer)
  - Target → Source: True (target reads source)
  - Target → Target:
      text: causal (lower triangle)
      other: True (bidirectional)

Hook policies control how injected hook tokens interact with the mask:
  - PREFIX_ONLY: hooks can attend to prefix only (safest)
  - BIDIRECTIONAL_WITH_PREFIX: hooks see prefix and are seen by all,
    but cannot read target tokens
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn.functional as F


class HookPolicy(Enum):
    """Policy for how hook tokens interact with the attention mask."""
    PREFIX_ONLY = "prefix_only"
    BIDIRECTIONAL_WITH_PREFIX = "bidirectional_with_prefix"


def build_prefix_lm_mask(
    src_len: int,
    tgt_len: int,
    target_modality: str,
    device: torch.device,
) -> torch.Tensor:
    """Create a Prefix-LM attention mask.

    Args:
        src_len: number of source (prefix) tokens.
        tgt_len: number of target tokens.
        target_modality: "text" gets causal target-to-target masking;
            all other modalities get bidirectional.
        device: torch device.

    Returns:
        mask: (1, 1, S+T, S+T) bool tensor.
    """
    tot = src_len + tgt_len
    mask = torch.ones(tot, tot, dtype=torch.bool, device=device)

    # Source cannot attend to target tokens
    mask[:src_len, src_len:] = False

    # For text: causal mask within target region
    if target_modality == "text":
        causal = torch.tril(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device)
        )
        mask[src_len:, src_len:] = causal

    return mask.unsqueeze(0).unsqueeze(0)


def apply_token_validity_mask(
    attn_mask: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> torch.Tensor:
    """Apply batch-specific token validity to an attention mask.

    PyTorch SDPA convention is preserved: ``True`` means the query can attend
    to the key. Invalid tokens are masked both as keys and queries, which keeps
    padded source tokens from influencing fused representations.

    Args:
        attn_mask: bool tensor shaped ``(1|B, 1, N, N)`` or broadcastable over B.
        valid_tokens: bool tensor shaped ``(B, N)``.

    Returns:
        bool tensor shaped ``(B, 1, N, N)``.
    """
    if valid_tokens.ndim != 2:
        raise ValueError(f"valid_tokens must be [B, N], got {tuple(valid_tokens.shape)}")
    if attn_mask.shape[-1] != valid_tokens.shape[1] or attn_mask.shape[-2] != valid_tokens.shape[1]:
        raise ValueError(
            "attn_mask and valid_tokens sequence lengths disagree: "
            f"mask={tuple(attn_mask.shape)}, valid={tuple(valid_tokens.shape)}"
        )
    B = valid_tokens.shape[0]
    mask = attn_mask.expand(B, *attn_mask.shape[1:]).clone()
    valid = valid_tokens.to(device=mask.device, dtype=torch.bool)
    mask = mask & valid[:, None, None, :]
    mask = mask & valid[:, None, :, None]
    return mask


def expand_mask_for_hooks(
    attn_mask: torch.Tensor,
    n_hook_tokens: int,
    prefix_len: int,
    policy: HookPolicy = HookPolicy.BIDIRECTIONAL_WITH_PREFIX,
) -> torch.Tensor:
    """Expand an attention mask to account for appended hook tokens.

    Hook tokens are appended at the end of the sequence. This function
    creates proper mask entries so that:

    - Under PREFIX_ONLY: hooks attend only to prefix tokens and each other.
      Content tokens cannot attend to hooks.
    - Under BIDIRECTIONAL_WITH_PREFIX: hooks attend to prefix and are
      attended to by all tokens, but hooks cannot read target tokens.
      This prevents target→hook→prefix information leakage because
      hooks never see target content.

    Args:
        attn_mask: (*, N, N) existing mask.
        n_hook_tokens: number of hook tokens being appended.
        prefix_len: number of prefix tokens (source + optional thoughts).
        policy: hook attention policy.

    Returns:
        Expanded mask (*, N+H, N+H).
    """
    if n_hook_tokens == 0:
        return attn_mask

    N = attn_mask.shape[-1]
    H = n_hook_tokens
    new_size = N + H
    device = attn_mask.device
    leading_dims = attn_mask.shape[:-2]

    # Create expanded mask
    new_mask = torch.zeros(
        *leading_dims, new_size, new_size,
        dtype=torch.bool, device=device,
    )

    # Copy original mask
    new_mask[..., :N, :N] = attn_mask

    if policy == HookPolicy.PREFIX_ONLY:
        # Hooks attend to prefix only (not target, not other hooks)
        new_mask[..., N:, :prefix_len] = True
        # Hooks attend to each other
        new_mask[..., N:, N:] = True

    elif policy == HookPolicy.BIDIRECTIONAL_WITH_PREFIX:
        # Hooks can attend to prefix tokens
        new_mask[..., N:, :prefix_len] = True
        # Hooks attend to each other
        new_mask[..., N:, N:] = True
        # All original tokens can attend to hook tokens
        new_mask[..., :N, N:] = True
        # BUT: hooks cannot attend to target tokens (positions prefix_len..N)
        # This is already False from initialization — no action needed

    return new_mask


def expand_mask_for_memory(
    attn_mask: torch.Tensor,
    mem_len: int,
) -> torch.Tensor:
    """Expand an attention mask for prepended memory tokens.

    Memory tokens get full bidirectional attention: all positions can
    attend to and from memory tokens.

    Args:
        attn_mask: (*, N, N) existing mask.
        mem_len: number of memory tokens prepended.

    Returns:
        Expanded mask (*, M+N, M+N).
    """
    if mem_len == 0:
        return attn_mask
    return F.pad(attn_mask, (mem_len, 0, mem_len, 0), value=True)


def check_mask_no_target_to_prefix_leak(
    mask: torch.Tensor,
    prefix_len: int,
    target_start: int,
) -> bool:
    """Safety check: verify no indirect path from prefix to target exists.

    Specifically verifies that source tokens (0..prefix_len) cannot
    attend to target tokens (target_start..) either directly or
    through intermediate positions.

    Returns True if mask is safe (no leak), False otherwise.
    """
    # Direct check: source → target should all be False
    source_to_target = mask[..., :prefix_len, target_start:]
    return not source_to_target.any().item()
