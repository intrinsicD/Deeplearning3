"""Top-level ModularLatentWorldModel.

Memory ownership (fix for dual-memory confusion):
  - The IMemory module is the SOLE owner of MemoryState.
  - ITransitionCore never writes to MemoryState; it returns (hidden, aux).
  - Any internal recurrent state a transition core needs is stored in
    memory_state.extras["transition_hidden"] and propagated via aux.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .containers import (
    LatentState,
    MemoryState,
    ModelOutput,
    ObservationPacket,
    TransitionOutput,
)
from .interfaces import (
    IActionEncoder,
    IConditioner,
    IDecoder,
    IEncoder,
    ILatentProjector,
    IMemory,
    IPredictionHead,
    IRegularizer,
    ITransitionCore,
)


class ModularLatentWorldModel(nn.Module):
    def __init__(
        self,
        encoder: IEncoder,
        latent_projector: ILatentProjector,
        memory: IMemory,
        action_encoder: IActionEncoder,
        conditioner: IConditioner,
        transition_core: ITransitionCore,
        prediction_head: IPredictionHead,
        regularizer: IRegularizer,
        decoders: Optional[Dict[str, IDecoder]] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.latent_projector = latent_projector
        self.memory = memory
        self.action_encoder = action_encoder
        self.conditioner = conditioner
        self.transition_core = transition_core
        self.prediction_head = prediction_head
        self.regularizer = regularizer
        self.decoders = nn.ModuleDict(decoders or {})

    def encode(self, obs: ObservationPacket) -> LatentState:
        encoded = self.encoder(obs)
        latent = self.latent_projector(encoded)
        return latent

    def transition(self, latent: LatentState, action: torch.Tensor, memory_state: MemoryState) -> TransitionOutput:
        action_repr = self.action_encoder(action)
        memory_ctx = self.memory.read(memory_state)
        core_input = latent.primary()
        conditioned = self.conditioner(core_input, action_repr, memory_ctx)

        # Transition core returns hidden + aux; it does NOT touch memory
        hidden, aux = self.transition_core(conditioned, memory_state)

        # Propagate any internal transition state (e.g. GRU hidden) into memory extras
        if "_transition_hidden" in aux:
            memory_state = MemoryState(
                context=memory_state.context,
                hidden=memory_state.hidden,
                extras={**memory_state.extras, "transition_hidden": aux.pop("_transition_hidden")},
            )

        predicted_next = self.prediction_head(hidden, reference=latent)

        # Memory module is the sole owner of memory updates
        updated_memory = self.memory.update(predicted_next, action_repr, memory_state)

        uncertainty = predicted_next.extras.get("predicted_logvar")
        aux["action_norm"] = action_repr.norm(dim=-1).mean()
        return TransitionOutput(
            next_latent=predicted_next,
            next_memory=updated_memory,
            uncertainty=uncertainty,
            aux=aux,
        )

    def decode(self, latent: LatentState, context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        for name, decoder in self.decoders.items():
            dec_out = decoder(latent, context=context)
            for k, v in dec_out.items():
                outputs[f"{name}.{k}"] = v
        return outputs

    def forward(
        self,
        obs_t: ObservationPacket,
        action_t: torch.Tensor,
        obs_tp1: Optional[ObservationPacket] = None,
        memory_state: Optional[MemoryState] = None,
        decoder_context: Optional[Dict[str, Any]] = None,
    ) -> ModelOutput:
        current_latent = self.encode(obs_t)
        batch_size = current_latent.z_sem.shape[0]
        if memory_state is None:
            memory_state = self.memory.init_state(batch_size=batch_size, device=current_latent.z_sem.device)

        transition = self.transition(current_latent, action_t, memory_state)
        target_next_latent = self.encode(obs_tp1) if obs_tp1 is not None else None
        decoder_outputs = self.decode(transition.next_latent, context=decoder_context)

        aux = dict(transition.aux)
        aux.update(self.regularizer(current_latent))
        aux["latent_norm"] = current_latent.z_sem.norm(dim=-1).mean()
        return ModelOutput(
            current_latent=current_latent,
            predicted_next_latent=transition.next_latent,
            target_next_latent=target_next_latent,
            next_memory=transition.next_memory,
            decoder_outputs=decoder_outputs,
            aux=aux,
        )
