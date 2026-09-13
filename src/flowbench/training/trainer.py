"""Training loop with early stopping and validation-based checkpoint selection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from flowbench.config import FlowBenchConfig


def run_train(cfg: FlowBenchConfig) -> Path:
    """Train the configured model and write a checkpoint directory.

    Args:
        cfg: Full pipeline configuration.

    Returns:
        Path to the checkpoint directory.
    """
    msg = "run_train is implemented in Phase 2 (models + training)"
    raise NotImplementedError(msg)
