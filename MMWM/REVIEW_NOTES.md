# MMWM Critical Review (April 22, 2026)

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
