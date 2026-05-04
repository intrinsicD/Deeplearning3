#!/usr/bin/env python
"""Compatibility wrapper for :mod:`apps.control_center.server`.

The implementation moved to ``apps/control_center/server.py`` as part of the repository reorganization.
This wrapper preserves the old root-level command.
"""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_module("apps.control_center.server", run_name="__main__")
