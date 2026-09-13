"""Four-layer, 32-channel convolutional predictor.

Layout for the default config (``layers=4, channels=32, kernel_size=3``)::

    conv(1→32) → GELU → conv(32→32) → GELU → conv(32→32) → GELU → conv(32→1)

Every convolution keeps the spatial size ("same" padding). ``padding_mode`` is
``zeros`` or ``circular``; the choice is a config flag whose justification is recorded in
``docs/data.md``. With ``residual=True`` the network predicts the increment over the
input, so an untrained model starts close to the persistence baseline.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import torch
from torch import nn

from flowbench.models.base import Predictor

if TYPE_CHECKING:
    from flowbench.config import CNNConfig


class ConvNet(Predictor):
    """Plain convolutional stack mapping ``(b, 1, H, W)`` to ``(b, 1, H, W)``."""

    def __init__(self, cfg: CNNConfig) -> None:
        super().__init__()
        self.residual = cfg.residual
        padding = cfg.kernel_size // 2
        widths = [1, *([cfg.channels] * (cfg.layers - 1)), 1]
        blocks: list[nn.Module] = []
        for i, (cin, cout) in enumerate(pairwise(widths)):
            blocks.append(
                nn.Conv2d(
                    cin,
                    cout,
                    kernel_size=cfg.kernel_size,
                    padding=padding,
                    padding_mode=cfg.padding_mode,
                )
            )
            if i < cfg.layers - 1:
                blocks.append(nn.GELU())
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the next field, adding the input when ``residual`` is set."""
        out: torch.Tensor = self.net(x)
        return x + out if self.residual else out
