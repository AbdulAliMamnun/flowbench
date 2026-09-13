"""Normalisation statistics fitted on training data only.

Inputs and targets are the same physical quantity (vorticity), so one scalar mean and
standard deviation are fitted on the *training* inputs and targets jointly and applied
to both. Using shared statistics keeps the persistence baseline an exact identity in
normalised space as well as in field units.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class NormalizationStats:
    """Scalar affine normalisation ``(value - mean) / std``."""

    mean: float
    std: float
    n_fields: int
    fitted_on: str = "train"

    def to_json(self, path: Path) -> None:
        """Write the statistics as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> NormalizationStats:
        """Read statistics written by :meth:`to_json`."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)


def fit_normalization(*fields: torch.Tensor, eps: float = 1e-12) -> NormalizationStats:
    """Fit mean and standard deviation over every element of ``fields``.

    Args:
        *fields: Tensors of any shape; they are flattened and concatenated.
        eps: Floor applied to the standard deviation to avoid division by zero.

    Returns:
        The fitted :class:`NormalizationStats`.

    Raises:
        ValueError: If no elements are provided.
    """
    flat = torch.cat([f.detach().reshape(-1).to(torch.float64) for f in fields])
    if flat.numel() == 0:
        msg = "cannot fit normalisation on zero elements"
        raise ValueError(msg)
    mean = float(flat.mean())
    std = max(float(flat.std(unbiased=False)), eps)
    n_fields = sum(int(f.shape[0]) if f.ndim > 0 else 1 for f in fields)
    return NormalizationStats(mean=mean, std=std, n_fields=n_fields)


def normalize(x: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    """Map field units to normalised units."""
    return (x - stats.mean) / stats.std


def denormalize(z: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    """Map normalised units back to field units."""
    return z * stats.std + stats.mean
