import numpy as np
import pytest
import torch

from flowbench.data.normalize import NormalizationStats
from flowbench.models.persistence import Persistence
from flowbench.serving.predictor import FieldPredictor


@pytest.fixture
def predictor() -> FieldPredictor:
    stats = NormalizationStats(mean=0.25, std=2.0, n_fields=1)
    return FieldPredictor(Persistence(), stats, torch.device("cpu"), resolution=32)


def test_persistence_roundtrip_is_identity_in_field_units(
    predictor: FieldPredictor, synthetic_fields: torch.Tensor
) -> None:
    field = synthetic_fields[0, 0].numpy()
    out = predictor.predict(field)
    assert out.shape == (32, 32)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, field, rtol=1e-6, atol=1e-6)


def test_batch_prediction_matches_single(
    predictor: FieldPredictor, synthetic_fields: torch.Tensor
) -> None:
    batched = predictor.predict_batch(synthetic_fields, batch_size=5)
    assert batched.shape == synthetic_fields.shape
    torch.testing.assert_close(batched, synthetic_fields, rtol=1e-6, atol=1e-6)
    empty = predictor.predict_batch(synthetic_fields[:0], batch_size=5)
    assert empty.shape == (0, 1, 32, 32)


def test_wrong_shape_is_rejected(predictor: FieldPredictor) -> None:
    with pytest.raises(ValueError, match="expected shape"):
        predictor.predict(np.zeros((16, 16), dtype=np.float32))
