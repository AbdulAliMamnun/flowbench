from pathlib import Path

import pytest
import torch

from flowbench.config import FlowBenchConfig, load_config
from flowbench.data import dataset as ds
from flowbench.data.normalize import NormalizationStats


def test_subsample_strides_both_spatial_axes() -> None:
    fields = torch.arange(2 * 1 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)
    out = ds.subsample(fields, 4)
    assert out.shape == (2, 1, 2, 2)
    torch.testing.assert_close(out, fields[:, :, ::4, ::4])
    assert out.is_contiguous()
    assert torch.equal(ds.subsample(fields, 1), fields)


def test_subsample_rejects_bad_rate() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        ds.subsample(torch.zeros(1, 1, 4, 4), 0)


def test_field_pairs_validates_shape() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        ds.FieldPairs(torch.zeros(2, 1, 4, 4), torch.zeros(3, 1, 4, 4))
    with pytest.raises(ValueError, match=r"\(n, 1, H, W\)"):
        ds.FieldPairs(torch.zeros(2, 2, 4, 4), torch.zeros(2, 2, 4, 4))


def test_prepared_roundtrip(tmp_config: Path, synthetic_fields: torch.Tensor) -> None:
    cfg: FlowBenchConfig = load_config(tmp_config)
    pairs = ds.FieldPairs(synthetic_fields, synthetic_fields * 2)
    path = ds.write_prepared(cfg.data, "train", pairs)
    assert path == ds.prepared_path(cfg.data, "train")
    back = ds.load_prepared(cfg.data, "train")
    torch.testing.assert_close(back.x, pairs.x)
    torch.testing.assert_close(back.y, pairs.y)
    assert len(back.subset([0, 3])) == 2
    with pytest.raises(FileNotFoundError, match="prepare"):
        ds.load_prepared(cfg.data, "test")


def test_loader_is_deterministic_and_normalised(synthetic_fields: torch.Tensor) -> None:
    pairs = ds.FieldPairs(synthetic_fields, synthetic_fields)
    stats = NormalizationStats(mean=0.5, std=2.0, n_fields=16)
    a = list(ds.make_loader(pairs, stats, batch_size=4, shuffle=True, seed=3))
    b = list(ds.make_loader(pairs, stats, batch_size=4, shuffle=True, seed=3))
    assert len(a) == 4
    for (xa, ya), (xb, yb) in zip(a, b, strict=True):
        torch.testing.assert_close(xa, xb)
        torch.testing.assert_close(ya, yb)
    x0 = a[0][0]
    torch.testing.assert_close(x0 * 2.0 + 0.5, torch.cat([p[0] for p in a])[:4] * 2.0 + 0.5)
