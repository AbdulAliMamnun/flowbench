"""Seeding, device resolution and determinism.

Known non-determinism (documented rather than hidden): the MPS backend does not
guarantee bit-identical results for every kernel, and PyTorch has no deterministic
implementation for some MPS operations. ``configure_determinism`` therefore asks for
deterministic algorithms with ``warn_only=True`` and records what was actually applied so
the checkpoint manifest states it.
"""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from flowbench.logging import get_logger

if TYPE_CHECKING:
    from flowbench.config import DeviceName

log = get_logger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch (CPU and MPS) generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(name: DeviceName) -> torch.device:
    """Turn the config device name into a ``torch.device``.

    ``auto`` selects ``mps`` when the backend is built and available, else ``cpu``.
    Requesting ``mps`` on a machine without it falls back to CPU with a warning rather
    than failing, so the same config runs in CI.
    """
    mps_ok = torch.backends.mps.is_available()
    if name == "cpu":
        return torch.device("cpu")
    if mps_ok:
        return torch.device("mps")
    if name == "mps":
        log.warning("mps requested but unavailable; falling back to cpu")
    return torch.device("cpu")


def configure_determinism(enabled: bool, device: torch.device) -> dict[str, Any]:
    """Request deterministic algorithms and report what was applied.

    Args:
        enabled: Whether determinism is requested in the config.
        device: The resolved compute device.

    Returns:
        A record for the checkpoint manifest describing the determinism settings and
        the caveats that apply to ``device``.
    """
    record: dict[str, Any] = {
        "requested": enabled,
        "use_deterministic_algorithms": False,
        "warn_only": True,
        "caveats": [],
    }
    if not enabled:
        return record
    torch.use_deterministic_algorithms(True, warn_only=True)
    record["use_deterministic_algorithms"] = True
    if device.type == "mps":
        record["caveats"].append(
            "MPS kernels are not guaranteed bit-reproducible across runs; "
            "non-deterministic ops only warn (warn_only=True)"
        )
    return record
