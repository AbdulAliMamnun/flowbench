"""Abstract predictor interface shared by every model.

A :class:`Predictor` maps a normalised vorticity field ``batch × 1 × H × W`` to a
normalised field of the same shape. Normalisation is handled outside the model so that
the persistence baseline and learned models are evaluated identically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class Predictor(nn.Module, ABC):
    """Base class for one-step vorticity predictors."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the next field.

        Args:
            x: Input field with shape ``(batch, 1, height, width)``.

        Returns:
            Predicted field with the same shape as ``x``.
        """

    @property
    def n_parameters(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def is_trainable(self) -> bool:
        """Whether the model has parameters that gradient descent can update."""
        return self.n_parameters > 0
