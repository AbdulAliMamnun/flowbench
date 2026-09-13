"""Persistence baseline: the prediction is the input."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flowbench.models.base import Predictor

if TYPE_CHECKING:
    import torch


class Persistence(Predictor):
    """Predict that the field does not change. Has no parameters."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` unchanged (a copy, so callers cannot alias the input)."""
        return x.clone()
