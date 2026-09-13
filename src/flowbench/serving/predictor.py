"""Thin serving wrapper: preprocess → model → postprocess.

Used by the API, the evaluation latency benchmark and the UI so that all three run the
identical single-sample path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from flowbench.data.normalize import denormalize, normalize

if TYPE_CHECKING:
    from flowbench.data.normalize import NormalizationStats
    from flowbench.models.base import Predictor


class FieldPredictor:
    """Predict one or more fields in original units from NumPy or Torch inputs."""

    def __init__(
        self,
        model: Predictor,
        stats: NormalizationStats,
        device: torch.device,
        resolution: int,
    ) -> None:
        self.model = model.to(device).eval()
        self.stats = stats
        self.device = device
        self.resolution = resolution

    def preprocess(self, field: np.ndarray) -> torch.Tensor:
        """Validate shape, normalise and move a ``(H, W)`` array to the device."""
        arr = np.asarray(field, dtype=np.float32)
        if arr.shape != (self.resolution, self.resolution):
            msg = f"expected shape ({self.resolution}, {self.resolution}), got {arr.shape}"
            raise ValueError(msg)
        x = torch.from_numpy(arr).reshape(1, 1, self.resolution, self.resolution)
        return normalize(x, self.stats).to(self.device)

    def postprocess(self, z: torch.Tensor) -> np.ndarray:
        """Denormalise a ``(1, 1, H, W)`` prediction and return a ``(H, W)`` float32 array."""
        out: np.ndarray = denormalize(z, self.stats).detach().cpu().numpy()[0, 0]
        return out.astype(np.float32)

    def predict(self, field: np.ndarray) -> np.ndarray:
        """Full single-sample path in field units."""
        with torch.inference_mode():
            return self.postprocess(self.model(self.preprocess(field)))

    def predict_batch(self, fields: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Predict ``(n, 1, H, W)`` fields in original units, batched, on CPU output."""
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, int(fields.shape[0]), batch_size):
                chunk = normalize(fields[start : start + batch_size], self.stats).to(self.device)
                outputs.append(denormalize(self.model(chunk), self.stats).cpu())
        return torch.cat(outputs) if outputs else fields.new_empty((0, *fields.shape[1:]))
