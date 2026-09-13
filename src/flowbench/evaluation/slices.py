"""High-vorticity slice definition derived from train/validation data only.

A test sample belongs to the slice when the maximum absolute vorticity of its
*reference* field is at or above a threshold. The threshold is a quantile of the same
statistic over train and validation reference fields, so the test set never informs
its own slicing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flowbench.evaluation.metrics import max_abs


@dataclass(frozen=True)
class HighVorticitySlice:
    """Threshold and its provenance."""

    quantile: float
    threshold: float
    n_reference_fields: int
    derived_from: str = "max|omega| of train+validation reference fields"

    def mask(self, reference: torch.Tensor) -> torch.Tensor:
        """Boolean mask of ``(n, 1, H, W)`` reference fields that belong to the slice."""
        return max_abs(reference) >= self.threshold


def high_vorticity_slice(reference_fields: torch.Tensor, quantile: float) -> HighVorticitySlice:
    """Derive the slice threshold from train/validation reference fields.

    Args:
        reference_fields: ``(n, 1, H, W)`` target fields from train and validation only.
        quantile: Quantile in ``(0, 1)`` of per-sample ``max|omega|``.

    Returns:
        The slice definition.

    Raises:
        ValueError: If no fields are given.
    """
    if reference_fields.shape[0] == 0:
        msg = "cannot derive a threshold from zero fields"
        raise ValueError(msg)
    stat = max_abs(reference_fields)
    threshold = float(torch.quantile(stat, quantile))
    return HighVorticitySlice(
        quantile=quantile, threshold=threshold, n_reference_fields=int(stat.numel())
    )
