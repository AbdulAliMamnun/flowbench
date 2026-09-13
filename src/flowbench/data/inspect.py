"""Record what the dataset actually contains before any training happens."""

from __future__ import annotations

from pathlib import Path

from flowbench.config import FlowBenchConfig


def run_inspect(cfg: FlowBenchConfig) -> Path:
    """Inspect the raw dataset and write ``data/manifest.json``.

    Args:
        cfg: Full pipeline configuration.

    Returns:
        Path to the written manifest.
    """
    msg = "run_inspect is implemented in Phase 1 (data)"
    raise NotImplementedError(msg)
