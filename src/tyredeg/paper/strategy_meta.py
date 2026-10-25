"""Audit metadata attached to every strategy-layer result file.

A paper number has to be traceable to the code, the data and the seed that
produced it. This is the one place that builds that block, so ``pit_loss.json``
and ``strategy.json`` carry the same fields.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT


def git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 -- provenance is best-effort, never a blocker
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def meta(script: str, module: str, seed: int, params: dict, inputs: list[Path]) -> dict:
    """The audit block: what ran, on what, with which settings."""
    return {
        "script": script,
        "module": module,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": git_head(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seed": seed,
        "params": params,
        "input_files": {p.name: sha256(p) for p in inputs if p.exists()},
    }
