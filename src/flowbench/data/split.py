"""Simulation-level train/validation/test splitting with persisted IDs."""

from __future__ import annotations

from pathlib import Path

from flowbench.config import FlowBenchConfig


def run_prepare(cfg: FlowBenchConfig) -> Path:
    """Create the split, fit normalisation statistics and persist both.

    Args:
        cfg: Full pipeline configuration.

    Returns:
        Path to the persisted split file.
    """
    msg = "run_prepare is implemented in Phase 1 (data)"
    raise NotImplementedError(msg)
