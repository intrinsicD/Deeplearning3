from __future__ import annotations

import torch

from omnilatent.training.objectives import UnifiedObjectiveLoss, default_objective_registry


EXPECTED_OBJECTIVES = {
    "self_reconstruction",
    "cross_modal_translation",
    "contrastive_alignment",
    "latent_bottleneck",
    "next_latent_prediction",
    "router_supervision",
    "kb_retrieval_distillation",
}


def test_default_objective_registry_declares_required_fields() -> None:
    registry = default_objective_registry()

    assert set(registry.names()) == EXPECTED_OBJECTIVES
    for name in EXPECTED_OBJECTIVES:
        spec = registry.get(name)
        assert spec.name == name
        assert spec.required_fields
        assert isinstance(spec.description, str)


def test_invalid_or_incomplete_batches_skip_cleanly() -> None:
    loss_fn = UnifiedObjectiveLoss(objectives=["self_reconstruction", "contrastive_alignment"])

    result = loss_fn({"predictions": {"text": torch.randn(2, 3, 8)}})

    assert result["total"].item() == 0.0
    assert result["active"] == []
    assert "self_reconstruction" in result["skipped"]
    assert "contrastive_alignment" in result["skipped"]
    assert result["results"]["self_reconstruction"].skipped


def test_self_reconstruction_objective_computes_text_loss() -> None:
    loss_fn = UnifiedObjectiveLoss(objectives=["self_reconstruction"])
    logits = torch.randn(2, 4, 16, requires_grad=True)
    targets = torch.randint(1, 16, (2, 4))

    result = loss_fn({"predictions": {"text": logits}, "targets": {"text": targets}})

    assert result["active"] == ["self_reconstruction"]
    assert result["losses"]["self_reconstruction"].shape == ()
    assert torch.equal(result["total"], result["losses"]["self_reconstruction"])
    result["total"].backward()
    assert logits.grad is not None


def test_cross_modal_translation_objective_computes_target_modality_loss() -> None:
    loss_fn = UnifiedObjectiveLoss(objectives=["cross_modal_translation"])
    pred = torch.sigmoid(torch.randn(2, 3, 8, 8, requires_grad=True))
    target = torch.rand(2, 3, 8, 8)

    result = loss_fn({
        "target_modality": "image",
        "translation_prediction": pred,
        "translation_target": target,
    })

    assert result["active"] == ["cross_modal_translation"]
    assert result["total"].item() >= 0.0


def test_contrastive_alignment_requires_two_modalities_and_computes_loss() -> None:
    loss_fn = UnifiedObjectiveLoss(objectives=["contrastive_alignment"])
    latents = {
        "image": torch.randn(4, 8, requires_grad=True),
        "text": torch.randn(4, 8, requires_grad=True),
    }

    result = loss_fn({"latents": latents})

    assert result["active"] == ["contrastive_alignment"]
    assert result["total"].item() > 0.0


def test_latent_bottleneck_and_next_latent_prediction_objectives() -> None:
    loss_fn = UnifiedObjectiveLoss(objectives=["latent_bottleneck", "next_latent_prediction"])
    bottleneck = torch.randn(2, 8, requires_grad=True)
    source_summary = torch.randn(2, 8)
    predicted_next = torch.randn(2, 3, 8, requires_grad=True)
    target_next = torch.randn(2, 3, 8)

    result = loss_fn({
        "reasoning_bottleneck": bottleneck,
        "source_summary": source_summary,
        "predicted_next_latent": predicted_next,
        "target_next_latent": target_next,
    })

    assert set(result["active"]) == {"latent_bottleneck", "next_latent_prediction"}
    assert result["total"].item() >= 0.0
    result["total"].backward()
    assert bottleneck.grad is not None
    assert predicted_next.grad is not None


def test_router_supervision_with_optional_stop_loss() -> None:
    loss_fn = UnifiedObjectiveLoss(objectives=["router_supervision"])
    logits = torch.randn(3, 4, requires_grad=True)
    targets = torch.tensor([0, 1, 3])
    stop_logits = torch.randn(3, 1, requires_grad=True)
    stop_targets = torch.tensor([0.0, 1.0, 0.0])

    result = loss_fn({
        "router_logits": logits,
        "router_targets": targets,
        "router_stop_logits": stop_logits,
        "router_stop_targets": stop_targets,
    })

    assert result["active"] == ["router_supervision"]
    assert result["total"].item() > 0.0
    result["total"].backward()
    assert logits.grad is not None
    assert stop_logits.grad is not None


def test_kb_retrieval_distillation_objective() -> None:
    loss_fn = UnifiedObjectiveLoss(objectives=["kb_retrieval_distillation"])
    logits = torch.randn(2, 5, requires_grad=True)
    teacher = torch.softmax(torch.randn(2, 5), dim=-1)

    result = loss_fn({"retrieval_logits": logits, "teacher_retrieval_probs": teacher})

    assert result["active"] == ["kb_retrieval_distillation"]
    assert result["total"].item() >= 0.0
    result["total"].backward()
    assert logits.grad is not None


def test_learned_uncertainty_weights_active_only_and_has_gradients() -> None:
    loss_fn = UnifiedObjectiveLoss(
        objectives=["self_reconstruction", "router_supervision"],
        learned_uncertainty=True,
    )
    logits = torch.randn(2, 4, 16, requires_grad=True)
    targets = torch.randint(1, 16, (2, 4))

    result = loss_fn({"predictions": {"text": logits}, "targets": {"text": targets}})

    assert result["active"] == ["self_reconstruction"]
    assert "router_supervision" in result["skipped"]
    assert "self_reconstruction_log_var" in result["losses"]
    assert "router_supervision_log_var" not in result["losses"]
    result["total"].backward()
    assert loss_fn.log_vars["self_reconstruction"].grad is not None
    assert loss_fn.log_vars["router_supervision"].grad is None


def test_objective_weights_affect_total() -> None:
    logits = torch.randn(2, 4, 16)
    targets = torch.randint(1, 16, (2, 4))
    batch = {"predictions": {"text": logits}, "targets": {"text": targets}}

    base = UnifiedObjectiveLoss(objectives=["self_reconstruction"])(batch)["total"]
    weighted = UnifiedObjectiveLoss(objectives=["self_reconstruction"], weights={"self_reconstruction": 2.0})(batch)["total"]

    assert torch.allclose(weighted, base * 2.0)

