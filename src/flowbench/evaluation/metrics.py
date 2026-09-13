"""Pure metric functions (Torch only, no I/O).

Every function takes ``(n, 1, H, W)`` tensors in **field units** and returns per-sample
values of shape ``(n,)`` so that aggregation (mean, worst case, slices) happens in one
place. Nothing here reads files, logs or touches devices.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Summary:
    """Aggregate of a per-sample metric."""

    mean: float
    median: float
    worst: float
    n: int

    @classmethod
    def of(cls, values: torch.Tensor) -> Summary:
        """Summarise a 1-D tensor; ``worst`` is the maximum."""
        v = values.detach().to(torch.float64).flatten()
        if v.numel() == 0:
            return cls(mean=float("nan"), median=float("nan"), worst=float("nan"), n=0)
        return cls(
            mean=float(v.mean()), median=float(v.median()), worst=float(v.max()), n=int(v.numel())
        )


def _check_pair(pred: torch.Tensor, ref: torch.Tensor) -> None:
    if pred.shape != ref.shape:
        msg = f"pred and ref must share a shape, got {tuple(pred.shape)} vs {tuple(ref.shape)}"
        raise ValueError(msg)
    if pred.ndim != 4:
        msg = f"expected (n, c, H, W), got {tuple(pred.shape)}"
        raise ValueError(msg)


def relative_l2(
    pred: torch.Tensor, ref: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample relative L2 error ``||pred - ref|| / max(||ref||, eps)``.

    Args:
        pred: Predictions ``(n, 1, H, W)``.
        ref: References ``(n, 1, H, W)``.
        eps: Floor for the reference norm.

    Returns:
        A pair ``(error, floored)`` where ``error`` has shape ``(n,)`` and ``floored`` is a
        boolean mask of samples whose reference norm was below ``eps``.
    """
    _check_pair(pred, ref)
    p = pred.to(torch.float64).flatten(1)
    r = ref.to(torch.float64).flatten(1)
    ref_norm = r.norm(dim=1)
    floored = ref_norm < eps
    return (p - r).norm(dim=1) / ref_norm.clamp_min(eps), floored


def mae(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-sample mean absolute error in field units, shape ``(n,)``."""
    _check_pair(pred, ref)
    return (pred.to(torch.float64) - ref.to(torch.float64)).abs().flatten(1).mean(dim=1)


def enstrophy(field: torch.Tensor) -> torch.Tensor:
    """Per-sample enstrophy density ``0.5 * mean(omega^2)`` over the grid, shape ``(n,)``.

    The physical domain size is unknown for this dataset, so the grid mean (enstrophy per
    unit area in grid units) is used instead of a domain integral.
    """
    if field.ndim != 4:
        msg = f"expected (n, c, H, W), got {tuple(field.shape)}"
        raise ValueError(msg)
    return 0.5 * field.to(torch.float64).pow(2).flatten(1).mean(dim=1)


def enstrophy_error(
    pred: torch.Tensor, ref: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample enstrophy error, absolute and relative to the reference enstrophy.

    Returns:
        ``(absolute, relative, floored)``: ``|E(pred) - E(ref)|``,
        ``|E(pred) - E(ref)| / max(E(ref), eps)`` and the mask of floored references.
    """
    _check_pair(pred, ref)
    e_pred = enstrophy(pred)
    e_ref = enstrophy(ref)
    absolute = (e_pred - e_ref).abs()
    floored = e_ref < eps
    return absolute, absolute / e_ref.clamp_min(eps), floored


def max_abs(field: torch.Tensor) -> torch.Tensor:
    """Per-sample maximum absolute value, shape ``(n,)``."""
    if field.ndim != 4:
        msg = f"expected (n, c, H, W), got {tuple(field.shape)}"
        raise ValueError(msg)
    return field.to(torch.float64).abs().flatten(1).amax(dim=1)
