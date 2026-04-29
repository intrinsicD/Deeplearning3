# MMWM Critical Review (April 22, 2026)

## Follow-up hardening pass (April 29, 2026)

1. **Audio is now an input modality, not only a decoder target**
   - Added a 1-D convolutional `AudioSubEncoder` and wired it into `simple_multimodal`, `structured_multimodal`, and `slot_multimodal` encoders.
   - `ModelConfig.encoder_kwargs` now includes `audio_channels`.
   - Remaining caveat: callers must set `audio_channels` to match their waveform/mel tensor channel count.

2. **Recurrent transition cores now tolerate `input_dim != hidden_dim`**
   - `RecurrentAttnResTransformerTransitionCore` projects once into hidden space and runs its inner core there.
   - `MoDRecurrentAttnResTransformerTransitionCore` now uses the same projection discipline before light/heavy routing.

3. **Shape bugs are surfaced earlier**
   - Latent losses and vector/image/audio reconstruction losses now require exact shape equality before MSE/NLL computation.
   - Evaluation reconstruction metrics now use the same shape discipline.

4. **Padded text evaluation improved**
   - Text perplexity can ignore padding tokens and returns a finite neutral perplexity when all targets are padding.

5. **Minimal deterministic data path added**
   - `MMWM.data.DeterministicTransitionDataset` and `collate_transition_batch` provide trainer-compatible, deterministic transition batches for smoke tests and loss-decrease checks.

## Remaining highest-priority risks

- Real dataset adapters are still missing; the deterministic dataset is an integration harness, not a replacement for D4RL/DMControl/TextWorld/video adapters.
- The `LatentRouter` still has no supervised/RL training path.
- Decoder outputs are still selected in losses/metrics by suffix matching; this remains acceptable for one decoder per suffix but brittle for production.
- The audio encoder supports `[B, C, T]` and `[B, T]`; richer 2-D spectrogram conventions should be standardized before large-scale audio training.

## Fixed known pitfalls

1. **Sequence training off-by-one/data contract ambiguity** (`Trainer.train_sequence_step`)
   - The transition loop uses `obs[t] -> obs[t+1]`, which requires modality sequences to be at least `len(action_seq) + 1`.
   - Added explicit contract checks and renamed internal variable to `transitions` for clarity.
   - This prevents silent index errors and confusing dataset bugs.

2. **Evaluation sampling bug in `fit()`**
   - `next(iter(eval_loader))` was recreating a fresh iterator every call, repeatedly evaluating on the first batch.
   - Replaced with a persistent iterator that cycles through the whole eval loader.
   - This removes strong evaluation bias and gives meaningful best-checkpoint selection.

3. **Structured encoder empty/unknown modality failure mode**
   - `StructuredMultimodalEncoder.forward` did not guard empty inputs and had a fallback branch that could mask caller errors.
   - Added strict validation:
     - reject empty `ObservationPacket.modalities`
     - reject packets whose modalities are unsupported by this encoder
   - This now fails fast with actionable error messages.

## Critical design risks identified (not changed intentionally)

1. **Stateful transition hidden in `memory_state.extras` can leak across episodes**
   - Contract is documented and consistent, but any caller that forgets to reset memory between episodes can unintentionally carry recurrent core state.

2. **Decoders selected by suffix matching in losses**
   - `WorldModelLoss` uses `endswith(...)` and `next(...)` to choose outputs. If multiple decoders emit the same suffix, selection order can become accidental.
   - Consider explicit namespaced key contracts for production hardening.

3. **`EvaluationSuite`/TensorBoard dependency is mandatory at import-time**
   - Test execution currently fails in environments without `tensorboard` installed.
   - This is an environment/dependency robustness issue rather than a mathematical/modeling issue.

## Deferred/unknown outcome areas left unchanged

- Objective weighting strategy and learned uncertainty clamps.
- Specific architecture choices in recurrent/AttnRes transition blocks.
- Representation choices for modality fusion and regularization terms.

These should be validated by your planned empirical testing, as requested.
