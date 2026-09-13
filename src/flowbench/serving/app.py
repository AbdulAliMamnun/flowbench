"""FastAPI application factory; the model is loaded once in the lifespan.

Endpoints:

- ``GET /health``  – liveness plus whether the checkpoint is loaded.
- ``GET /version`` – package, model and runtime versions from the checkpoint manifest.
- ``POST /predict`` – one field in, one prediction out, with ``model_version``,
  ``device`` and ``latency_ms``.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import torch
from fastapi import FastAPI, Request

from flowbench import __version__
from flowbench.logging import get_logger
from flowbench.serving.errors import install_error_handlers
from flowbench.serving.predictor import FieldPredictor
from flowbench.serving.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    VersionResponse,
)
from flowbench.training.checkpoint import LoadedCheckpoint, load_checkpoint
from flowbench.training.seed import resolve_device

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from flowbench.config import FlowBenchConfig

log = get_logger(__name__)


class ServingState:
    """What the lifespan loads once and every request reads."""

    def __init__(self, checkpoint: LoadedCheckpoint, predictor: FieldPredictor) -> None:
        self.checkpoint = checkpoint
        self.predictor = predictor


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def create_app(cfg: FlowBenchConfig) -> FastAPI:
    """Build the API for the checkpoint named by ``cfg``.

    Args:
        cfg: Full configuration; ``serving_checkpoint_dir`` selects the checkpoint.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        device = resolve_device(cfg.run.device)
        ckpt = load_checkpoint(cfg.serving_checkpoint_dir, device)
        predictor = FieldPredictor(ckpt.model, ckpt.stats, device, cfg.data.resolution)
        predictor.predict(torch.zeros(cfg.data.resolution, cfg.data.resolution).numpy())
        app.state.serving = ServingState(ckpt, predictor)
        log.info(
            "model loaded",
            checkpoint=str(ckpt.path),
            model_version=ckpt.model_version,
            device=device.type,
        )
        yield
        app.state.serving = None

    app = FastAPI(
        title="FlowBench",
        version=__version__,
        description="Predicts a 2D fluid's future vorticity field from one input field.",
        lifespan=lifespan,
    )
    install_error_handlers(app)

    def _state(request: Request) -> ServingState:
        state: ServingState | None = getattr(request.app.state, "serving", None)
        if state is None:
            msg = "model not loaded"
            raise RuntimeError(msg)
        return state

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        state: ServingState | None = getattr(request.app.state, "serving", None)
        return HealthResponse(
            status="ok" if state is not None else "loading",
            model_loaded=state is not None,
            device=state.predictor.device.type if state is not None else "none",
        )

    @app.get("/version", response_model=VersionResponse)
    async def version(request: Request) -> VersionResponse:
        state = _state(request)
        return VersionResponse.from_manifest(
            __version__,
            state.checkpoint.model_version,
            state.checkpoint.manifest,
            checkpoint=str(state.checkpoint.path),
            resolution=cfg.data.resolution,
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: Request, body: PredictRequest) -> PredictResponse:
        state = _state(request)
        field = body.to_array(cfg.data.resolution)
        _sync(state.predictor.device)
        t0 = time.perf_counter_ns()
        out = state.predictor.predict(field)
        _sync(state.predictor.device)
        latency_ms = (time.perf_counter_ns() - t0) / 1e6
        return PredictResponse(
            prediction=out.tolist(),
            shape=list(out.shape),
            model_version=state.checkpoint.model_version,
            device=state.predictor.device.type,
            latency_ms=latency_ms,
        )

    return app


def run_serve(cfg: FlowBenchConfig) -> None:
    """Start the prediction server with uvicorn.

    Args:
        cfg: Full pipeline configuration (``serving.host`` and ``serving.port``).
    """
    import uvicorn  # noqa: PLC0415 - only needed when actually serving

    log.info("serving", host=cfg.serving.host, port=cfg.serving.port)
    uvicorn.run(create_app(cfg), host=cfg.serving.host, port=cfg.serving.port, log_level="info")
