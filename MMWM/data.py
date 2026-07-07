"""Datasets and transition adapters for MMWM training.

The trainer consumes flat batch dictionaries such as ``vector_t``, ``action``,
``vector_tp1`` and optional decoder targets.  The adapters in this module keep
that contract in one place so scripts do not each invent slightly different
offline-RL batch formats.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .containers import ObservationPacket


def flatten_transition_value(value: Any) -> torch.Tensor:
    """Flatten vector/dict observations or actions into a float32 vector.

    D4RL/Minari observations may be arrays, scalars, tensors, or dictionaries of
    arrays.  Dict keys are sorted to make the layout stable across processes.
    """
    if isinstance(value, Mapping):
        parts = [flatten_transition_value(value[key]) for key in sorted(value.keys())]
        if not parts:
            return torch.zeros(0, dtype=torch.float32)
        return torch.cat(parts, dim=0).float()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(np.asarray(value))
    if tensor.dtype == torch.bool or not torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=torch.float32)
    else:
        tensor = tensor.float()
    return tensor.reshape(-1)


def _length(value: Sequence[Any] | np.ndarray | torch.Tensor) -> int:
    return int(len(value))


def _item(value: Sequence[Any] | np.ndarray | torch.Tensor, idx: int) -> Any:
    return value[idx]


def _optional_bool_at(value: Any, idx: int) -> bool:
    if value is None:
        return False
    item = _item(value, idx)
    if isinstance(item, torch.Tensor):
        return bool(item.detach().cpu().item())
    arr = np.asarray(item)
    return bool(arr.item() if arr.shape == () else arr.reshape(-1)[0])


class TransitionTupleDataset(Dataset):
    """Wrap ``(obs_t, action, obs_tp1)`` tuples in the MMWM trainer contract.

    This is the vector-first path used by D4RL/Minari-style offline RL datasets.
    Non-vector observations should get domain-specific adapters that emit
    ``image_*``, ``text_*`` or ``audio_*`` keys.
    """

    def __init__(
        self,
        observations: Sequence[Any] | np.ndarray | torch.Tensor,
        actions: Sequence[Any] | np.ndarray | torch.Tensor,
        next_observations: Sequence[Any] | np.ndarray | torch.Tensor | None = None,
        *,
        terminals: Sequence[Any] | np.ndarray | torch.Tensor | None = None,
        timeouts: Sequence[Any] | np.ndarray | torch.Tensor | None = None,
        max_transitions: int | None = None,
    ) -> None:
        super().__init__()
        obs_len = _length(observations)
        action_len = _length(actions)
        if next_observations is None:
            n = min(action_len, max(obs_len - 1, 0))
        else:
            n = min(obs_len, action_len, _length(next_observations))
        if max_transitions is not None:
            n = min(n, int(max_transitions))
        if n <= 0:
            raise RuntimeError("No transition tuples are available.")

        self.observations = observations
        self.actions = actions
        self.next_observations = next_observations
        self.terminals = terminals
        self.timeouts = timeouts
        self.index = list(range(n))

        first = self[0]
        self.vector_dim = int(first["vector_t"].numel())
        self.action_dim = int(first["action"].numel())
        if self.vector_dim <= 0:
            raise RuntimeError("Flattened observation has zero dimensions.")
        if self.action_dim <= 0:
            raise RuntimeError("Flattened action has zero dimensions.")

    @staticmethod
    def from_mapping(
        mapping: Mapping[str, Any],
        *,
        max_transitions: int | None = None,
    ) -> "TransitionTupleDataset":
        """Build from a D4RL-style dataset mapping."""
        if "observations" not in mapping or "actions" not in mapping:
            raise KeyError("Transition mapping must contain 'observations' and 'actions'.")
        done = mapping.get("terminals")
        if done is None:
            done = mapping.get("dones")
        return TransitionTupleDataset(
            mapping["observations"],
            mapping["actions"],
            mapping.get("next_observations"),
            terminals=done,
            timeouts=mapping.get("timeouts"),
            max_transitions=max_transitions,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = self.index[idx]
        obs_t = flatten_transition_value(_item(self.observations, t))
        if self.next_observations is None:
            obs_tp1 = flatten_transition_value(_item(self.observations, t + 1))
        else:
            obs_tp1 = flatten_transition_value(_item(self.next_observations, t))
        action = flatten_transition_value(_item(self.actions, t))

        item: Dict[str, torch.Tensor] = {
            "vector_t": obs_t,
            "vector_tp1": obs_tp1,
            "vector_target": obs_tp1.clone(),
            "action": action,
        }
        done = _optional_bool_at(self.terminals, t) or _optional_bool_at(self.timeouts, t)
        if self.terminals is not None or self.timeouts is not None:
            item["done"] = torch.tensor(done, dtype=torch.bool)
        return item


class D4RLTransitionDataset(TransitionTupleDataset):
    """Load a D4RL Gym dataset as vector transition tuples.

    The dependency is optional; importing :mod:`MMWM.data` does not require D4RL.
    """

    def __init__(self, dataset_id: str, max_transitions: int | None = None) -> None:
        try:
            import gym  # type: ignore
            import d4rl  # noqa: F401  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "D4RLTransitionDataset requires gym and d4rl. Install them before "
                "loading D4RL datasets."
            ) from exc

        env = gym.make(dataset_id)
        mapping = env.get_dataset()
        self.dataset_id = dataset_id
        done = mapping.get("terminals")
        if done is None:
            done = mapping.get("dones")
        super().__init__(
            mapping["observations"],
            mapping["actions"],
            mapping.get("next_observations"),
            terminals=done,
            timeouts=mapping.get("timeouts"),
            max_transitions=max_transitions,
        )


def iter_minari_episodes(dataset: Any) -> Iterable[Any]:
    """Support common Minari dataset iteration APIs across versions."""
    if hasattr(dataset, "iterate_episodes"):
        yield from dataset.iterate_episodes()
        return
    if hasattr(dataset, "iter_episodes"):
        yield from dataset.iter_episodes()
        return
    for idx in range(len(dataset)):
        yield dataset[idx]


class MinariTransitionDataset(Dataset):
    """Index Minari episodes as single-step MMWM transition tuples."""

    def __init__(self, dataset_id: str, max_transitions: Optional[int] = None) -> None:
        super().__init__()
        try:
            import minari  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "MinariTransitionDataset requires minari. Install it with "
                "`pip install minari` before loading Minari datasets."
            ) from exc

        self.dataset_id = dataset_id
        self.dataset = minari.load_dataset(dataset_id)
        self.episodes = list(iter_minari_episodes(self.dataset))
        self.index: List[Tuple[int, int]] = []

        for ep_idx, episode in enumerate(self.episodes):
            n_actions = len(episode.actions)
            n_obs = len(episode.observations)
            n = min(n_actions, n_obs - 1)
            for t in range(n):
                self.index.append((ep_idx, t))
                if max_transitions is not None and len(self.index) >= max_transitions:
                    break
            if max_transitions is not None and len(self.index) >= max_transitions:
                break

        if not self.index:
            raise RuntimeError(f"No transitions found in Minari dataset {dataset_id!r}")

        first = self[0]
        self.vector_dim = int(first["vector_t"].numel())
        self.action_dim = int(first["action"].numel())
        if self.vector_dim <= 0:
            raise RuntimeError("Flattened observation has zero dimensions.")
        if self.action_dim <= 0:
            raise RuntimeError("Flattened action has zero dimensions.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, t = self.index[idx]
        ep = self.episodes[ep_idx]
        obs_t = flatten_transition_value(ep.observations[t])
        obs_tp1 = flatten_transition_value(ep.observations[t + 1])
        action = flatten_transition_value(ep.actions[t])
        item = {
            "vector_t": obs_t,
            "vector_tp1": obs_tp1,
            "vector_target": obs_tp1.clone(),
            "action": action,
        }
        terminations = getattr(ep, "terminations", None)
        truncations = getattr(ep, "truncations", None)
        if terminations is not None or truncations is not None:
            item["done"] = torch.tensor(
                _optional_bool_at(terminations, t) or _optional_bool_at(truncations, t),
                dtype=torch.bool,
            )
        return item


class GridWorldTransitionDataset(Dataset):
    """Deterministic gridworld transitions for end-to-end smoke tests.

    The observation is available as:
      - ``vector``: normalized ``(x, y)`` agent position,
      - ``image``: a 3-channel rendered grid with agent and goal markers,
      - ``text``: fixed-length token description of agent/goal position.

    Actions are one-hot vectors for right, down, left, up.
    """

    ACTION_DELTAS = ((1, 0), (0, 1), (-1, 0), (0, -1))

    def __init__(
        self,
        length: int = 128,
        grid_size: int = 5,
        episode_len: int = 8,
        include_text: bool = True,
        include_image: bool = True,
    ) -> None:
        super().__init__()
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        if episode_len < 1:
            raise ValueError("episode_len must be >= 1")
        self.length = int(length)
        self.grid_size = int(grid_size)
        self.episode_len = int(episode_len)
        self.include_text = include_text
        self.include_image = include_image
        self.vector_dim = 2
        self.action_dim = 4
        self.text_len = 6
        self.vocab_size = max(16, self.grid_size + 8)
        self.goal = (self.grid_size - 1, self.grid_size - 1)

    def __len__(self) -> int:
        return self.length

    def _episode_step(self, idx: int) -> tuple[int, int]:
        return int(idx) // self.episode_len, int(idx) % self.episode_len

    def _initial_state(self, episode: int) -> tuple[int, int]:
        x = episode % self.grid_size
        y = (episode * 2) % self.grid_size
        return x, y

    def _action_id(self, episode: int, step: int) -> int:
        return (episode + step) % len(self.ACTION_DELTAS)

    def _transition(self, state: tuple[int, int], action_id: int) -> tuple[int, int]:
        dx, dy = self.ACTION_DELTAS[action_id]
        x = min(max(state[0] + dx, 0), self.grid_size - 1)
        y = min(max(state[1] + dy, 0), self.grid_size - 1)
        return x, y

    def _state_at(self, episode: int, step: int) -> tuple[int, int]:
        state = self._initial_state(episode)
        for s in range(step):
            state = self._transition(state, self._action_id(episode, s))
        return state

    def _vector(self, state: tuple[int, int]) -> torch.Tensor:
        scale = float(self.grid_size - 1)
        return torch.tensor([state[0] / scale, state[1] / scale], dtype=torch.float32)

    def _action(self, action_id: int) -> torch.Tensor:
        action = torch.zeros(self.action_dim, dtype=torch.float32)
        action[action_id] = 1.0
        return action

    def _tokens(self, state: tuple[int, int]) -> torch.Tensor:
        gx, gy = self.goal
        return torch.tensor(
            [1, state[0] + 2, state[1] + 2, gx + 2, gy + 2, 0],
            dtype=torch.long,
        )

    def _image(self, state: tuple[int, int]) -> torch.Tensor:
        image = torch.zeros(3, self.grid_size, self.grid_size, dtype=torch.float32)
        x, y = state
        gx, gy = self.goal
        image[0, y, x] = 1.0
        image[1, gy, gx] = 1.0
        image[2, :, :] = 0.05
        return image

    def _observation_modalities(
        self,
        state: tuple[int, int],
        modalities: tuple[str, ...] = ("vector", "text", "image"),
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if "vector" in modalities:
            out["vector"] = self._vector(state).unsqueeze(0)
        if "text" in modalities:
            out["text"] = self._tokens(state).unsqueeze(0)
        if "image" in modalities:
            out["image"] = self._image(state).unsqueeze(0)
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode, step = self._episode_step(idx)
        state = self._state_at(episode, step)
        action_id = self._action_id(episode, step)
        next_state = self._transition(state, action_id)

        item: Dict[str, torch.Tensor] = {
            "vector_t": self._vector(state),
            "vector_tp1": self._vector(next_state),
            "vector_target": self._vector(next_state),
            "action": self._action(action_id),
            "done": torch.tensor(step + 1 >= self.episode_len, dtype=torch.bool),
        }

        if self.include_text:
            text_t = self._tokens(state)
            text_tp1 = self._tokens(next_state)
            item.update({
                "text_t": text_t,
                "text_tp1": text_tp1,
                "text_target": text_tp1,
                "prefix_tokens": torch.cat([torch.zeros(1, dtype=torch.long), text_tp1[:-1]], dim=0),
            })

        if self.include_image:
            image_t = self._image(state)
            image_tp1 = self._image(next_state)
            item.update({
                "image_t": image_t,
                "image_tp1": image_tp1,
                "image_target": image_tp1,
            })

        return item

    def rollout_sequence(
        self,
        start_index: int = 0,
        horizon: int = 4,
        modalities: tuple[str, ...] = ("vector",),
    ) -> tuple[List[ObservationPacket], List[torch.Tensor]]:
        episode, step = self._episode_step(start_index)
        horizon = min(int(horizon), self.episode_len - step)
        state = self._state_at(episode, step)
        obs_sequence = [ObservationPacket(modalities=self._observation_modalities(state, modalities))]
        action_sequence: List[torch.Tensor] = []
        for offset in range(horizon):
            action_id = self._action_id(episode, step + offset)
            action_sequence.append(self._action(action_id).unsqueeze(0))
            state = self._transition(state, action_id)
            obs_sequence.append(ObservationPacket(modalities=self._observation_modalities(state, modalities)))
        return obs_sequence, action_sequence


class DeterministicTransitionDataset(Dataset):
    """Synthetic but learnable transition tuples.

    Each item follows a fixed linear vector dynamics rule and can optionally add
    aligned text, image, and audio observations/targets. Randomness is keyed by
    the sample index, so data is deterministic across workers and epochs.
    """

    def __init__(
        self,
        length: int = 1024,
        vector_dim: int = 16,
        action_dim: int = 8,
        text_len: int = 8,
        vocab_size: int = 256,
        image_size: int = 32,
        audio_channels: int = 1,
        audio_length: int = 128,
        include_text: bool = False,
        include_image: bool = False,
        include_audio: bool = False,
        seed: int = 1234,
    ) -> None:
        super().__init__()
        self.length = int(length)
        self.vector_dim = int(vector_dim)
        self.action_dim = int(action_dim)
        self.text_len = int(text_len)
        self.vocab_size = int(vocab_size)
        self.image_size = int(image_size)
        self.audio_channels = int(audio_channels)
        self.audio_length = int(audio_length)
        self.include_text = include_text
        self.include_image = include_image
        self.include_audio = include_audio
        self.seed = int(seed)

        g = torch.Generator().manual_seed(seed)
        self.action_to_vector = torch.randn(action_dim, vector_dim, generator=g) / max(action_dim, 1) ** 0.5
        self.vector_drift = torch.randn(vector_dim, generator=g) * 0.01

    def __len__(self) -> int:
        return self.length

    def _generator(self, idx: int) -> torch.Generator:
        return torch.Generator().manual_seed(self.seed + int(idx) * 9973)

    def _render_image(self, vector: torch.Tensor) -> torch.Tensor:
        h = w = self.image_size
        yy = torch.linspace(-1.0, 1.0, h).view(1, h, 1)
        xx = torch.linspace(-1.0, 1.0, w).view(1, 1, w)
        weights = vector[:3].view(-1, 1, 1)
        if weights.shape[0] < 3:
            weights = torch.nn.functional.pad(weights, (0, 0, 0, 0, 0, 3 - weights.shape[0]))
        phase = vector[3:6].mean() if vector.numel() >= 6 else vector.mean()
        image = torch.sigmoid(weights[:3] * xx + torch.flip(weights[:3], dims=[0]) * yy + phase)
        return image.float()

    def _render_audio(self, vector: torch.Tensor) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, self.audio_length)
        channels = []
        base = vector.abs().mean().clamp_min(0.1)
        for ch in range(self.audio_channels):
            freq = 1.0 + base * float(ch + 1)
            phase = vector[ch % vector.numel()]
            channels.append(torch.sin(2.0 * torch.pi * freq * t + phase))
        return torch.stack(channels, dim=0).float()

    def _tokens_from_vector(self, vector: torch.Tensor) -> torch.Tensor:
        scaled = torch.round((vector[: self.text_len].abs() * 100.0)).long()
        if scaled.numel() < self.text_len:
            scaled = torch.nn.functional.pad(scaled, (0, self.text_len - scaled.numel()))
        return (scaled[: self.text_len] % max(self.vocab_size, 1)).long()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        g = self._generator(idx)
        vector_t = torch.randn(self.vector_dim, generator=g)
        action = torch.randn(self.action_dim, generator=g)
        vector_tp1 = 0.85 * vector_t + 0.15 * (action @ self.action_to_vector) + self.vector_drift

        item: Dict[str, torch.Tensor] = {
            "vector_t": vector_t.float(),
            "vector_tp1": vector_tp1.float(),
            "vector_target": vector_tp1.float(),
            "action": action.float(),
        }

        if self.include_text:
            text_t = self._tokens_from_vector(vector_t)
            text_tp1 = self._tokens_from_vector(vector_tp1)
            item.update({
                "text_t": text_t,
                "text_tp1": text_tp1,
                "text_target": text_tp1,
                "prefix_tokens": torch.cat([torch.zeros(1, dtype=torch.long), text_tp1[:-1]], dim=0),
            })

        if self.include_image:
            image_t = self._render_image(vector_t)
            image_tp1 = self._render_image(vector_tp1)
            item.update({
                "image_t": image_t,
                "image_tp1": image_tp1,
                "image_target": image_tp1,
            })

        if self.include_audio:
            audio_t = self._render_audio(vector_t)
            audio_tp1 = self._render_audio(vector_tp1)
            item.update({
                "audio_t": audio_t,
                "audio_tp1": audio_tp1,
                "audio_target": audio_tp1,
            })

        return item


def collate_transition_batch(items: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """Stack a list of deterministic transition items into a trainer batch."""
    if not items:
        raise ValueError("collate_transition_batch received an empty item list")
    keys = items[0].keys()
    return {key: torch.stack([item[key] for item in items], dim=0) for key in keys}
