I did not modify files. Findings below are ordered by impact on real learning/generalization.
Findings
1.
Blocker: manifest data cannot feed training as-is.
StreamingMultiModalDataset yields MultiModalSample objects directly, while the trainer expects dict[str, Tensor] batches. Default DataLoader fails with TypeError: default_collate ... found MultiModalSample. See streaming.py, sample.py, trainer.py. There is no real collate/tokenize/resize bridge in collate/init.py.
2.
Blocker: local audio/video manifest samples are metadata-only.
Local audio/video sources return paths in metadata, not tensors, so the model has nothing to learn from. See local.py. Also, streaming: "false" in JSON/YAML becomes truthy because of bool(...) coercion in manifest.py.
3.
High: collators silently drop modalities and can create zero-loss steps.
The main multimodal collator keeps only modalities present in every sample. Mixed real batches can lose image/audio/video fields entirely; if fewer than two modalities remain, training returns total: 0.0. See data.py and trainer.py.
4.
High: learnable loss parameters are not optimized.
MultiModalLoss has learnable log_vars, but optimizers are created from model.parameters() only. Probe result: criterion_params_in_optimizer 0 of 4. See losses.py and trainer.py. Curriculum training also omits both MultiModalLoss and TemporalContextLoss parameters; probe result: 0/4 and 0/8. See curriculum_train.py and losses.py.
5.
High: temporal order loss cannot learn direction.
TemporalOrderLoss computes a symmetric dot product, so (A, B) and (B, A) produce identical logits. The unused combined = torch.cat(...) suggests the intended classifier input was abandoned. See losses.py. Separately, distant-clip prediction directly minimizes MSE(z_anchor, z_context), encouraging temporal collapse rather than change modeling. See curriculum_train.py.
6.
High: public MMWM AV training script is broken.
include_text and include_image are undefined in train_mmwm_av.py. Image decoder kwargs use out_channels/out_size, but the decoder expects output_channels/output_size in decoders.py. The script also passes ModelConfig as the first positional arg to WorldModelLoss, whose first arg is weights in losses.py.
7.
High: MMWM learned uncertainty is also not optimized.
train_mmwm_minari.py creates the optimizer before WorldModelLoss(learned_uncertainty=True), so loss log_vars never update. See train_mmwm_minari.py and MMWM/losses.py.
8.
Medium: HPWM can train on frozen random visual features.
If pretrained DINO loading fails, HPWM falls back to a random DINO implementation, then freezes it. That can make training appear functional while preventing visual generalization. See model.py. Also, --ssv2-dir is not passed into create_dataloaders, so the CLI path can be ignored. See train.py and data.py.
9.
Medium: OmniLatent advertised multimodal paths bypass learned behavior.
generate() calls the backbone without the hook manager, so trained latent hooks are ignored during generation. See omnilatent.py. forward_multimodal() does not actually fuse multiple modalities; it picks one input and ignores the rest. See omnilatent.py.
10.
Medium: many real-data failures become zero tensors instead of hard failures.
Failed video/audio/image loads often return zeros, which keeps losses finite while teaching the model corrupted targets. Examples: video_dataset.py, coco_dataset.py, mmwm_adapter.py.
Verification
Focused tests: 29 passed for gradient/model/streaming coverage. Temporal/video focused tests: 3 failed, 45 passed; all three failures are broken imports of curriculum_train from tests, while implementation lives under scripts/training/curriculum_train.py. Full suite was attempted but did not complete cleanly in this environment.