"""Batch-one latency measurement with warm-up and p50/p95 reporting.

Timing wraps the *whole* single-sample path the API executes (preprocessing, model
forward, postprocessing) and synchronises the device before reading the clock, so the
numbers describe what a caller of ``POST /predict`` would experience minus HTTP overhead.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class LatencyStats:
    """Milliseconds per single-sample call after warm-up."""

    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    warmup_iterations: int
    timed_iterations: int
    device: str

    def as_dict(self) -> dict[str, Any]:
        """Plain dictionary for JSON reports."""
        return asdict(self)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":  # pragma: no cover - not used in v0.1
        torch.cuda.synchronize()


def measure_latency(
    call: Callable[[], object],
    device: torch.device,
    warmup_iterations: int,
    timed_iterations: int,
) -> LatencyStats:
    """Time ``call`` repeatedly and report percentiles.

    Args:
        call: Zero-argument function running one full single-sample prediction.
        device: Device the model runs on (used for synchronisation).
        warmup_iterations: Untimed calls before measurement.
        timed_iterations: Timed calls; percentiles are computed over these.

    Returns:
        :class:`LatencyStats` in milliseconds.
    """
    for _ in range(warmup_iterations):
        call()
        _synchronize(device)
    samples = np.empty(timed_iterations, dtype=np.float64)
    for i in range(timed_iterations):
        _synchronize(device)
        t0 = time.perf_counter_ns()
        call()
        _synchronize(device)
        samples[i] = (time.perf_counter_ns() - t0) / 1e6
    return LatencyStats(
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
        mean_ms=float(samples.mean()),
        min_ms=float(samples.min()),
        max_ms=float(samples.max()),
        warmup_iterations=warmup_iterations,
        timed_iterations=timed_iterations,
        device=device.type,
    )
