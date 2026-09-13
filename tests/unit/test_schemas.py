import base64

import numpy as np
import pytest
from pydantic import ValidationError

from flowbench.serving.errors import FieldValidationError, error_body
from flowbench.serving.schemas import PredictRequest


def test_nested_list_decodes() -> None:
    arr = PredictRequest(field=np.ones((4, 4)).tolist()).to_array(4)
    assert arr.shape == (4, 4) and arr.dtype == np.float32


def test_base64_decodes_little_endian_float32() -> None:
    values = np.arange(16, dtype="<f4").reshape(4, 4)
    req = PredictRequest(field_b64=base64.b64encode(values.tobytes()).decode())
    np.testing.assert_array_equal(req.to_array(4), values)


def test_requires_exactly_one_encoding() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        PredictRequest()
    with pytest.raises(ValidationError, match="exactly one"):
        PredictRequest(field=[[0.0]], field_b64="AA==")


def test_extra_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(field=[[0.0]], bogus=1)  # type: ignore[call-arg]


def test_shape_mismatch_raises_field_error() -> None:
    with pytest.raises(FieldValidationError, match=r"shape \(4, 4\)"):
        PredictRequest(field=np.zeros((3, 4)).tolist()).to_array(4)


def test_non_finite_raises_field_error() -> None:
    rows = np.zeros((2, 2)).tolist()
    rows[0][1] = float("inf")
    with pytest.raises(FieldValidationError, match="non-finite") as info:
        PredictRequest(field=rows).to_array(2)
    assert info.value.details == [{"non_finite_count": 1}]


def test_invalid_base64_raises_field_error() -> None:
    with pytest.raises(FieldValidationError, match="valid base64"):
        PredictRequest(field_b64="not base64!").to_array(2)


def test_error_body_shape() -> None:
    body = error_body("x", "msg")
    assert body == {"error": {"type": "x", "message": "msg", "details": []}}
