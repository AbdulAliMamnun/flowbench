import pytest
import torch

from flowbench.evaluation.slices import high_vorticity_slice


def test_threshold_is_quantile_of_max_abs() -> None:
    fields = torch.zeros(10, 1, 2, 2)
    for i in range(10):
        fields[i, 0, 0, 0] = float(i + 1)  # max|omega| = 1..10
    hv = high_vorticity_slice(fields, quantile=0.9)
    assert hv.threshold == pytest.approx(float(torch.quantile(torch.arange(1.0, 11.0), 0.9)))
    assert hv.n_reference_fields == 10
    assert "train+validation" in hv.derived_from


def test_mask_uses_reference_max_abs_only() -> None:
    fields = torch.zeros(4, 1, 2, 2)
    fields[:, 0, 0, 0] = torch.tensor([0.5, 2.0, -3.0, 1.0])
    hv = high_vorticity_slice(fields, quantile=0.5)
    mask = hv.mask(fields)
    assert mask.tolist() == [False, True, True, False]


def test_empty_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero fields"):
        high_vorticity_slice(torch.empty(0, 1, 2, 2), quantile=0.9)
