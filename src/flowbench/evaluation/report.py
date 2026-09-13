"""Evaluation driver: metrics.json, markdown table and error-map PNGs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from flowbench.config import FlowBenchConfig


def run_evaluate(cfg: FlowBenchConfig) -> Path:
    """Evaluate persistence and the trained model on the held-out test set.

    Args:
        cfg: Full pipeline configuration.

    Returns:
        Path to the report directory.
    """
    msg = "run_evaluate is implemented in Phase 3 (evaluation)"
    raise NotImplementedError(msg)
