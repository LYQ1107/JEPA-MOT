#!/usr/bin/env python3
"""Single-command convenience wrapper.

Use ``train_adapter.py`` and ``run_adapter.py`` for independently auditable
stages; this file simply dispatches to the same implementation with
``train``/``run`` as the first positional argument.
"""
from mot_jepa_adapter_v1 import main


if __name__ == "__main__":
    raise SystemExit(main())
