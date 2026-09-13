"""API integration tests via FastAPI's TestClient on a synthetic checkpoint."""

import base64
import math
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from flowbench.config import FlowBenchConfig
from flowbench.serving.app import create_app
from flowbench.training.trainer import run_train

pytestmark = pytest.mark.slow

RES = 32


@pytest.fixture
def client(prepared_config: FlowBenchConfig) -> Iterator[TestClient]:
    """Train on the synthetic data and serve the resulting checkpoint."""
    run_train(prepared_config)
    with TestClient(create_app(prepared_config)) as c:
        yield c


@pytest.fixture
def field() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(RES, RES)).astype(np.float32)


def test_health_reports_loaded_model(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["device"] == "cpu"


def test_version_reports_checkpoint(client: TestClient) -> None:
    r = client.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"].startswith("cnn-")
    assert body["n_parameters"] == 19105
    assert body["resolution"] == RES
    assert "torch_version" in body and "git_sha" in body


def test_predict_nested_list_happy_path(client: TestClient, field: np.ndarray) -> None:
    r = client.post("/predict", json={"field": field.tolist()})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["shape"] == [RES, RES]
    pred = np.asarray(body["prediction"], dtype=np.float32)
    assert pred.shape == (RES, RES)
    assert np.isfinite(pred).all()
    assert body["model_version"].startswith("cnn-")
    assert body["device"] == "cpu"
    assert body["latency_ms"] > 0.0


def test_predict_base64_matches_nested_list(client: TestClient, field: np.ndarray) -> None:
    b64 = base64.b64encode(field.astype("<f4").tobytes()).decode("ascii")
    a = client.post("/predict", json={"field": field.tolist()}).json()["prediction"]
    b = client.post("/predict", json={"field_b64": b64}).json()["prediction"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)


def test_wrong_shape_is_structured_422(client: TestClient) -> None:
    r = client.post("/predict", json={"field": np.zeros((16, 16)).tolist()})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["type"] == "invalid_field"
    assert "(32, 32)" in body["error"]["message"]
    assert "Traceback" not in r.text


def test_ragged_rows_are_rejected(client: TestClient) -> None:
    rows = np.zeros((RES, RES)).tolist()
    rows[3] = rows[3][:-1]
    r = client.post("/predict", json={"field": rows})
    assert r.status_code == 422
    assert r.json()["error"]["type"] == "invalid_field"


def test_nan_input_is_rejected(client: TestClient, field: np.ndarray) -> None:
    rows = field.tolist()
    rows[0][0] = math.nan
    # JSON has no NaN; send the token the way lenient clients do.
    r = client.post(
        "/predict",
        content=('{"field": ' + str(rows).replace("nan", "NaN") + "}").encode(),
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["type"] in {"invalid_field", "validation_error"}


def test_nan_via_base64_is_rejected(client: TestClient, field: np.ndarray) -> None:
    bad = field.copy()
    bad[1, 2] = np.inf
    b64 = base64.b64encode(bad.astype("<f4").tobytes()).decode("ascii")
    r = client.post("/predict", json={"field_b64": b64})
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["type"] == "invalid_field"
    assert "non-finite" in body["error"]["message"]


def test_bad_base64_length_is_rejected(client: TestClient) -> None:
    b64 = base64.b64encode(b"\x00" * 16).decode("ascii")
    r = client.post("/predict", json={"field_b64": b64})
    assert r.status_code == 422
    assert "bytes" in r.json()["error"]["message"]


def test_empty_body_is_structured_422(client: TestClient) -> None:
    r = client.post("/predict")
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["type"] == "validation_error"
    assert isinstance(body["error"]["details"], list)


def test_both_encodings_is_rejected(client: TestClient, field: np.ndarray) -> None:
    r = client.post("/predict", json={"field": field.tolist(), "field_b64": "AAAA"})
    assert r.status_code == 422
    assert r.json()["error"]["type"] == "validation_error"


def test_unknown_route_is_structured_404(client: TestClient) -> None:
    r = client.get("/nope")
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "http_error"
