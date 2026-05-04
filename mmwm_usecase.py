#!/usr/bin/env python
"""Compatibility wrapper for :mod:`scripts.evaluation.mmwm_usecase`.

The implementation moved to ``scripts/evaluation/mmwm_usecase.py`` as part of the repository reorganization.
This wrapper preserves the old root-level command.
"""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_module("scripts.evaluation.mmwm_usecase", run_name="__main__")
