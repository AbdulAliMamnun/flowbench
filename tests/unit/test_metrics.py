import math

import pytest
import torch

from flowbench.evaluation.metrics import (
    Summary,
    enstrophy,
    enstrophy_error,
    mae,
    max_abs,
    relative_l2,
)


def test_relative_l2_zero_for_perfect_prediction(synthetic_fields: torch.Tensor) -> None:
    err, floored = relative_l2(synthetic_fields, synthetic_fields, eps=1e-8)
    assert err.shape == (synthetic_fields.shape[0],)
    assert torch.all(err == 0.0)
    assert not floored.any()


def test_relative_l2_matches_manual_formula() -> None:
    ref = torch.tensor([[[[3.0, 4.0], [0.0, 0.0]]]])  # norm 5
    pred = torch.tensor([[[[3.0, 4.0], [3.0, 4.0]]]])  # diff norm 5
    err, _ = relative_l2(pred, ref, eps=1e-8)
    assert err.item() == pytest.approx(1.0)


def test_relative_l2_floors_near_zero_reference() -> None:
    ref = torch.zeros(3, 1, 4, 4)
    ref[0, 0, 0, 0] = 1e-3
    pred = torch.ones(3, 1, 4, 4) * 0.5
    err, floored = relative_l2(pred, ref, eps=1e-2)
    assert floored.tolist() == [True, True, True]
    assert torch.isfinite(err).all()
    assert err[1].item() == pytest.approx((0.5 * 4) / 1e-2)


def test_relative_l2_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        relative_l2(torch.zeros(2, 1, 4, 4), torch.zeros(3, 1, 4, 4), eps=1e-8)
    with pytest.raises(ValueError, match=r"\(n, c, H, W\)"):
        relative_l2(torch.zeros(4, 4), torch.zeros(4, 4), eps=1e-8)


def test_mae_is_mean_absolute_difference() -> None:
    ref = torch.zeros(2, 1, 2, 2)
    pred = torch.tensor([[[[1.0, -1.0], [2.0, -2.0]]], [[[0.5, 0.5], [0.5, 0.5]]]])
    assert mae(pred, ref).tolist() == pytest.approx([1.5, 0.5])


def test_enstrophy_is_half_mean_square() -> None:
    field = torch.full((1, 1, 4, 4), 2.0)
    assert enstrophy(field).item() == pytest.approx(2.0)
    scaled = enstrophy(field * 3)
    assert scaled.item() == pytest.approx(18.0)


def test_enstrophy_error_relative_and_absolute() -> None:
    ref = torch.full((1, 1, 4, 4), 2.0)  # E = 2
    pred = torch.full((1, 1, 4, 4), 1.0)  # E = 0.5
    absolute, relative, floored = enstrophy_error(pred, ref, eps=1e-8)
    assert absolute.item() == pytest.approx(1.5)
    assert relative.item() == pytest.approx(0.75)
    assert not floored.any()
    _, rel_floored, floored = enstrophy_error(pred, torch.zeros_like(ref), eps=1e-2)
    assert floored.all()
    assert rel_floored.item() == pytest.approx(0.5 / 1e-2)


def test_max_abs_per_sample() -> None:
    field = torch.tensor([[[[1.0, -5.0], [2.0, 0.0]]], [[[0.1, 0.2], [-0.3, 0.0]]]])
    assert max_abs(field).tolist() == pytest.approx([5.0, 0.3])


def test_summary_of_values() -> None:
    s = Summary.of(torch.tensor([1.0, 3.0, 2.0]))
    assert (s.mean, s.median, s.worst, s.n) == (2.0, 2.0, 3.0, 3)
    empty = Summary.of(torch.empty(0))
    assert math.isnan(empty.mean) and empty.n == 0


def test_metrics_are_pure_and_device_agnostic(synthetic_fields: torch.Tensor) -> None:
    """Float64 accumulation: same result regardless of input dtype."""
    a, _ = relative_l2(synthetic_fields.double() * 1.1, synthetic_fields.double(), eps=1e-8)
    b, _ = relative_l2(synthetic_fields * 1.1, synthetic_fields, eps=1e-8)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
