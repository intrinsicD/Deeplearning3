#!/usr/bin/env python
"""Compatibility wrapper for :mod:`scripts.training.train_coco`.

The implementation moved to ``scripts/training/train_coco.py`` as part of the repository reorganization.
This wrapper preserves the old root-level command.
"""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_module("scripts.training.train_coco", run_name="__main__")
