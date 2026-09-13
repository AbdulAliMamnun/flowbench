"""FastAPI application factory; the model is loaded once in the lifespan."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flowbench.config import FlowBenchConfig


def run_serve(cfg: FlowBenchConfig) -> None:
    """Start the prediction server with uvicorn.

    Args:
        cfg: Full pipeline configuration.
    """
    msg = "run_serve is implemented in Phase 4 (serving + UI)"
    raise NotImplementedError(msg)
