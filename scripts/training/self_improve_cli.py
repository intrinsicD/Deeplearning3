#!/usr/bin/env python3
"""Top-level shim for the self-improvement CLI.

Lets you run ``python scripts/training/self_improve_cli.py ...`` instead
of ``python -m scripts.training.self_improve.cli ...``. The real
implementation lives in :mod:`scripts.training.self_improve.cli`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# When invoked as a script, ``scripts`` isn't on the path yet; we need
# to add the repo root before importing.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.training.self_improve.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
