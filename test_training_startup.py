#!/usr/bin/env python
"""Compatibility wrapper for :mod:`scripts.diagnostics.test_training_startup`.

The implementation moved to ``scripts/diagnostics/test_training_startup.py`` as part of the repository reorganization.
This wrapper preserves the old root-level command.
"""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_module("scripts.diagnostics.test_training_startup", run_name="__main__")
