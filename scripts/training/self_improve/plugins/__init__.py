"""Per-component plugins for the self-improvement orchestrator.

Each plugin is a thin adapter around an existing trainer. The orchestrator
(phases 5+) drives them through the ``ComponentPlugin`` interface defined
in ``.base``.

Plugins are imported lazily through :func:`get_plugin` so that test
machines without optional dependencies (e.g. ``torchvision`` for the
Gaussian encoder, ``timm`` for HPWM's DINO backbone) can still import
this package and use a subset of the plugins.
"""

from __future__ import annotations

from typing import Callable

from scripts.training.self_improve.plugins.base import (
    ComponentPlugin,
    EvalReport,
    StepReport,
)

_PLUGIN_NAMES = (
    "gaussian_encoder",
    "lgq",
    "omnilatent",
    "mmwm",
    "hpwm",
)


def _load(name: str) -> Callable[..., ComponentPlugin]:
    if name == "gaussian_encoder":
        from scripts.training.self_improve.plugins.gaussian_encoder import (
            GaussianEncoderPlugin,
        )
        return GaussianEncoderPlugin
    if name == "lgq":
        from scripts.training.self_improve.plugins.lgq import LGQPlugin
        return LGQPlugin
    if name == "omnilatent":
        from scripts.training.self_improve.plugins.omnilatent import OmniLatentPlugin
        return OmniLatentPlugin
    if name == "mmwm":
        from scripts.training.self_improve.plugins.mmwm import MMWMPlugin
        return MMWMPlugin
    if name == "hpwm":
        from scripts.training.self_improve.plugins.hpwm import HPWMPlugin
        return HPWMPlugin
    raise KeyError(f"unknown plugin {name!r}; known: {_PLUGIN_NAMES}")


def get_plugin(name: str) -> Callable[..., ComponentPlugin]:
    """Return the plugin class for *name*. See ``_PLUGIN_NAMES``."""
    return _load(name)


def available_plugins() -> tuple[str, ...]:
    return _PLUGIN_NAMES


__all__ = [
    "ComponentPlugin",
    "EvalReport",
    "StepReport",
    "available_plugins",
    "get_plugin",
]
