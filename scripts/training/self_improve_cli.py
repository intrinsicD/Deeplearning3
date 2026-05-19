#!/usr/bin/env python3
"""Top-level shim for the self-improvement CLI.

Lets you run ``python scripts/training/self_improve_cli.py ...`` instead
of ``python -m scripts.training.self_improve.cli ...``. The real
implementation lives in :mod:`scripts.training.self_improve.cli`.
"""

from __future__ import annotations

import sys

from scripts.training.self_improve.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
