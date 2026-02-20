#!/usr/bin/env python3
"""OmniLatent demo: shows model creation, hook usage, and inference.

This script demonstrates:
  1. Creating an OmniLatent model
  2. Running forward passes for each modality
  3. Cross-modal translation (e.g. image → text)
  4. Registering and using Latent Neural Hooks
  5. Multi-modal forward (multiple inputs at once)
  6. Memory footprint analysis
"""

from __future__ import annotations

import torch

from omnilatent.config import OmniLatentConfig
from omnilatent.model.hooks import LatentNeuralHook
from omnilatent.model.omnilatent import OmniLatentModel
from omnilatent.utils import (
    ALL_MODALITIES,
    count_parameters,
    count_trainable_parameters,
    param_size_mb,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use a smaller config for the demo
    config = OmniLatentConfig(
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        gradient_checkpointing=False,
    )

    model = OmniLatentModel(config).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Model size (FP32): {param_size_mb(model):.1f} MB")
    print()

    # --- Create synthetic inputs ---
    B = 2
    text_ids = torch.randint(1, config.vocab_size, (B, 32), device=device)
    audio_mel = torch.randn(B, config.audio_n_mels, 256, device=device)
    image = torch.randn(B, 3, config.image_size, config.image_size, device=device)
    video = torch.randn(
        B, 3, config.video_max_frames, config.video_size, config.video_size,
        device=device,
    )

    data = {"text": text_ids, "audio": audio_mel, "image": image, "video": video}

    # -------------------------------------------------------------------------
    # 1. Self-reconstruction for each modality
    # -------------------------------------------------------------------------
    print("=== Self-Reconstruction ===")
    model.eval()
    with torch.no_grad():
        for mod in ALL_MODALITIES:
            result = model.reconstruct(mod, data[mod])
            out = result["output"]
            print(f"  {mod:>6s}: input {tuple(data[mod].shape)} → output {tuple(out.shape)}")
    print()

    # -------------------------------------------------------------------------
    # 2. Cross-modal translation
    # -------------------------------------------------------------------------
    print("=== Cross-Modal Translation ===")
    with torch.no_grad():
        # Image → Text (teacher-forced with text_ids as target)
        result = model("image", image, "text", text_ids)
        logits = result["output"]
        tokens = logits.argmax(dim=-1)
        print(f"  image → text: logits {tuple(logits.shape)}, token ids {tokens[0, :8].tolist()}...")

        # Image → Text (autoregressive generation)
        generated = model.generate("image", image, max_len=16)
        print(f"  image → text (generate): {tuple(generated.shape)}, ids {generated[0, :8].tolist()}...")

        # Text → Image (learned target queries)
        result = model("text", text_ids, "image")
        img_out = result["output"]
        print(f"  text → image: output {tuple(img_out.shape)}")

        # Audio → Video (learned target queries)
        result = model("audio", audio_mel, "video")
        vid_out = result["output"]
        print(f"  audio → video: output {tuple(vid_out.shape)}")
    print()

    # -------------------------------------------------------------------------
    # 3. Latent Neural Hooks
    # -------------------------------------------------------------------------
    print("=== Latent Neural Hooks ===")

    # Create a hook that injects 8 learnable tokens at layers 0, 1, 2, 3
    style_hook = LatentNeuralHook(
        name="style_control",
        num_tokens=8,
        dim=config.hidden_dim,
        target_layers=[0, 1, 2, 3],
        gate_bias_init=-4.0,  # starts nearly silent
        use_transform=True,
    )
    style_hook = style_hook.to(device)

    # Register the hook
    model.register_hook(style_hook)
    print(f"  Registered hook: {style_hook.name}")
    print(f"  Hook tokens: {style_hook.num_tokens}")
    print(f"  Target layers: {sorted(style_hook.target_layers)}")
    print(f"  Hook params: {count_parameters(style_hook):,}")

    # Gate values (should be near 0 at init)
    for l in sorted(style_hook.target_layers):
        g = style_hook.gate_value(l).item()
        print(f"    Layer {l} gate: {g:.4f}")

    # Run forward pass WITH hooks (teacher-forced)
    with torch.no_grad():
        result_hooked = model("image", image, "text", text_ids)
        print(f"  Forward with hook: output {tuple(result_hooked['output'].shape)}")

    # Remove hook
    model.remove_hook("style_control")
    print(f"  Active hooks: {model.list_hooks()}")
    print()

    # -------------------------------------------------------------------------
    # 4. Multi-modal forward
    # -------------------------------------------------------------------------
    print("=== Multi-Modal Forward ===")
    with torch.no_grad():
        results = model.forward_multimodal(
            inputs={"text": text_ids, "image": image},
            target_modalities=["text", "image"],
        )
        for mod, res in results.items():
            print(f"  {mod}: output {tuple(res['output'].shape)}")
    print()

    # -------------------------------------------------------------------------
    # 5. Gradient flow check
    # -------------------------------------------------------------------------
    print("=== Gradient Flow Check ===")
    model.train()
    result = model("image", image, "text", text_ids)
    logits = result["output"]

    # Compute a dummy loss
    target = torch.randint(0, config.vocab_size, (B, logits.shape[1]), device=device)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, config.vocab_size), target.reshape(-1)
    )
    loss.backward()

    # Check that gradients flow to all major components
    components = {
        "text_encoder": model.encoders["text"],
        "image_encoder": model.encoders["image"],
        "backbone_layer_0": model.backbone.layers[0],
        "text_decoder": model.decoders["text"],
    }
    for name, module in components.items():
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters()
        )
        print(f"  {name:>20s}: gradient {'OK' if has_grad else 'MISSING'}")

    print()
    print("Demo complete.")


if __name__ == "__main__":
    main()
