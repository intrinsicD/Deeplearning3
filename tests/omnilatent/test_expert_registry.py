"""W2.1 — expert registry over hooks ∪ tools ∪ KB-query with learnable keys."""

from __future__ import annotations

import pytest
import torch

from omnilatent.agent.registry import ExpertRegistry, ExpertSpec
from omnilatent.model.hooks import LatentNeuralHook, NeuralPortManager


def test_register_and_enumerate() -> None:
    reg = ExpertRegistry(key_dim=16)
    reg.register("hook:style", "hook", tags=["style"])
    reg.register("tool:search", "tool")
    reg.register("kb:main", "kb")

    assert len(reg) == 3
    assert reg.ids() == ["hook:style", "tool:search", "kb:main"]
    assert reg.ids_of_kind("hook") == ["hook:style"]
    assert reg.kind("kb:main") == "kb"
    assert "tool:search" in reg


def test_keys_matrix_shape_and_grad() -> None:
    reg = ExpertRegistry(key_dim=8)
    reg.register("a", "hook")
    reg.register("b", "tool")
    keys = reg.keys()
    assert keys.shape == (2, 8)
    assert keys.requires_grad  # parameters are trainable
    # Empty registry returns a well-shaped empty matrix.
    assert ExpertRegistry(key_dim=8).keys().shape == (0, 8)


def test_keys_are_deterministically_seeded_from_id_and_tags() -> None:
    r1 = ExpertRegistry(key_dim=32)
    r2 = ExpertRegistry(key_dim=32)
    r1.register("hook:x", "hook", tags=["lang", "fr"])
    r2.register("hook:x", "hook", tags=["lang", "fr"])
    torch.testing.assert_close(r1.keys(), r2.keys())

    # Different tags → different seed → different key.
    r3 = ExpertRegistry(key_dim=32)
    r3.register("hook:x", "hook", tags=["lang", "de"])
    assert not torch.allclose(r1.keys(), r3.keys())


def test_unregister() -> None:
    reg = ExpertRegistry(key_dim=4)
    reg.register("a", "hook")
    reg.register("b", "hook")
    assert reg.unregister("a") is True
    assert reg.ids() == ["b"]
    assert reg.unregister("missing") is False


def test_sync_hooks_from_manager_is_idempotent() -> None:
    mgr = NeuralPortManager()
    mgr.register_hook(LatentNeuralHook(name="s1", num_tokens=2, dim=16, target_layers=[0]))
    reg = ExpertRegistry(key_dim=16)

    reg.sync_hooks(mgr)
    assert reg.ids_of_kind("hook") == ["hook:s1"]

    # Re-sync after adding a hook: only the new one is added (idempotent).
    mgr.register_hook(LatentNeuralHook(name="s2", num_tokens=2, dim=16, target_layers=[0]))
    reg.sync_hooks(mgr)
    assert set(reg.ids_of_kind("hook")) == {"hook:s1", "hook:s2"}

    # Removing a hook and re-syncing unregisters its expert.
    mgr.remove_hook("s1")
    reg.sync_hooks(mgr)
    assert reg.ids_of_kind("hook") == ["hook:s2"]


def test_invalid_kind_rejected() -> None:
    with pytest.raises(ValueError):
        ExpertSpec(expert_id="x", kind="not_a_kind")
    with pytest.raises(ValueError):
        ExpertRegistry(key_dim=4).register("x", "bogus")
