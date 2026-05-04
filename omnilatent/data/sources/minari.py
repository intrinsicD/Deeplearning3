"""Minari transition source adapter."""

from __future__ import annotations

from typing import Any, Iterator

import torch

from omnilatent.data.manifest import SourceSpec
from omnilatent.data.sample import MultiModalSample


def iter_minari_transitions(source: SourceSpec) -> Iterator[MultiModalSample]:
    dataset_obj = source.options.get("dataset_obj")
    if dataset_obj is None:
        dataset_id = source.dataset or source.path
        if dataset_id is None:
            raise ValueError("minari source requires dataset, path, or options.dataset_obj")
        try:
            import minari
        except ImportError as exc:
            raise ImportError("Minari source requires `minari`. Install with `pip install minari`.") from exc
        dataset_obj = minari.load_dataset(dataset_id, download=bool(source.options.get("download", False)))

    for episode_idx, episode in enumerate(_iterate_episodes(dataset_obj)):
        observations = _get_attr_or_key(episode, "observations")
        actions = _get_attr_or_key(episode, "actions")
        rewards = _get_attr_or_key(episode, "rewards", default=None)
        if observations is None or actions is None:
            continue
        n = min(len(actions), len(observations) - 1 if hasattr(observations, "__len__") else len(actions))
        for t in range(n):
            obs = _to_tensor(observations[t])
            action = _to_tensor(actions[t])
            reward = None if rewards is None else float(_to_tensor(rewards[t]).reshape(-1)[0].item())
            next_obs = _to_tensor(observations[t + 1]) if t + 1 < len(observations) else None
            yield MultiModalSample(
                vector=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                metadata={"dataset": source.dataset or source.path, "episode": episode_idx, "transition": t},
            )


def _iterate_episodes(dataset_obj: Any) -> Iterator[Any]:
    if hasattr(dataset_obj, "iterate_episodes"):
        yield from dataset_obj.iterate_episodes()
        return
    if hasattr(dataset_obj, "episodes"):
        yield from dataset_obj.episodes
        return
    yield from dataset_obj


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.as_tensor(value, dtype=torch.float32)


__all__ = ["iter_minari_transitions"]

