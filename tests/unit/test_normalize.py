from pathlib import Path

import pytest
import torch

from flowbench.data.normalize import (
    NormalizationStats,
    denormalize,
    fit_normalization,
    normalize,
)


def test_fit_matches_torch_moments(synthetic_fields: torch.Tensor) -> None:
    stats = fit_normalization(synthetic_fields)
    flat = synthetic_fields.double().reshape(-1)
    assert stats.mean == pytest.approx(float(flat.mean()))
    assert stats.std == pytest.approx(float(flat.std(unbiased=False)))
    assert stats.n_fields == synthetic_fields.shape[0]


def test_joint_fit_over_inputs_and_targets(synthetic_fields: torch.Tensor) -> None:
    x = synthetic_fields
    y = synthetic_fields * 2.0
    stats = fit_normalization(x, y)
    flat = torch.cat([x.double().reshape(-1), y.double().reshape(-1)])
    assert stats.mean == pytest.approx(float(flat.mean()))
    assert stats.n_fields == 2 * x.shape[0]


def test_normalize_roundtrip(synthetic_fields: torch.Tensor) -> None:
    stats = fit_normalization(synthetic_fields)
    z = normalize(synthetic_fields, stats)
    assert z.double().mean().abs() < 1e-5
    assert z.double().std(unbiased=False) == pytest.approx(1.0, abs=1e-5)
    back = denormalize(z, stats)
    torch.testing.assert_close(back, synthetic_fields, rtol=1e-5, atol=1e-5)


def test_constant_field_uses_std_floor() -> None:
    stats = fit_normalization(torch.ones(4, 1, 8, 8))
    assert stats.std > 0.0
    assert torch.isfinite(normalize(torch.ones(1, 1, 8, 8), stats)).all()


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero elements"):
        fit_normalization(torch.empty(0))


def test_json_roundtrip(tmp_path: Path) -> None:
    stats = NormalizationStats(mean=0.25, std=1.5, n_fields=10)
    path = tmp_path / "norm" / "stats.json"
    stats.to_json(path)
    assert NormalizationStats.from_json(path) == stats
