"""The demo bundle must load on its own and predict exactly like the full checkpoint."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from flowbench.config import FlowBenchConfig
from flowbench.data import dataset as ds
from flowbench.serving.predictor import FieldPredictor
from flowbench.training.checkpoint import load_checkpoint
from flowbench.training.trainer import run_train
from flowbench.ui.demo import (
    MANIFEST_NAME,
    SUBSET_NAME,
    bundle_available,
    load_bundle,
    run_export_demo,
    select_subset,
    verify_bundle,
)

pytestmark = pytest.mark.slow

N_DEMO = 10


@pytest.fixture
def exported(prepared_config: FlowBenchConfig, tmp_path: Path) -> tuple[FlowBenchConfig, Path]:
    cfg = prepared_config.model_copy(
        update={
            "demo": prepared_config.demo.model_copy(
                update={"dir": tmp_path / "demo", "n_samples": N_DEMO, "seed": 3}
            )
        }
    )
    run_train(cfg)
    return cfg, run_export_demo(cfg)


def test_select_subset_is_seeded_and_sorted() -> None:
    ids = list(range(50, 100))
    a = select_subset(ids, 10, seed=1)
    b = select_subset(ids, 10, seed=1)
    c = select_subset(ids, 10, seed=2)
    assert a == b and a != c
    assert a == sorted(a) and len(set(a)) == 10
    assert select_subset(ids, 500, seed=0) == ids


def test_bundle_layout_manifest_and_size(exported: tuple[FlowBenchConfig, Path]) -> None:
    cfg, out = exported
    assert bundle_available(out)
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    assert manifest["subset"]["n_samples"] == N_DEMO
    assert manifest["subset"]["seed"] == 3
    assert len(manifest["subset"]["instance_ids"]) == N_DEMO
    assert manifest["total_bytes"] <= cfg.demo.max_bytes
    assert set(manifest["files"]) >= {
        SUBSET_NAME,
        "checkpoints/cnn/model.pt",
        "checkpoints/cnn/MANIFEST.json",
        "checkpoints/persistence/model.pt",
    }
    assert verify_bundle(out) == []
    assert manifest["high_vorticity_slice"]["threshold"] > 0


def test_tampering_is_detected(exported: tuple[FlowBenchConfig, Path]) -> None:
    _, out = exported
    (out / "checkpoints" / "cnn" / "normalization.json").write_text("{}")
    assert verify_bundle(out) == ["checkpoints/cnn/normalization.json: hash mismatch"]
    with pytest.raises(ValueError, match="failed verification"):
        load_bundle(out, torch.device("cpu"))


def test_bundle_predicts_like_the_full_checkpoint(exported: tuple[FlowBenchConfig, Path]) -> None:
    cfg, out = exported
    bundle = load_bundle(out, torch.device("cpu"))
    full = load_checkpoint(cfg.checkpoint_dir, torch.device("cpu"))
    full_test = ds.load_prepared(cfg.data, "test")

    demo_pred = FieldPredictor(
        bundle.checkpoint.model, bundle.checkpoint.stats, torch.device("cpu"), 32
    )
    full_pred = FieldPredictor(full.model, full.stats, torch.device("cpu"), 32)
    for position, instance_id in enumerate(bundle.instance_ids):
        x_demo = bundle.test.x[position, 0].numpy()
        x_full = full_test.x[instance_id, 0].numpy()
        np.testing.assert_array_equal(x_demo, x_full)
        np.testing.assert_array_equal(
            bundle.test.y[position, 0].numpy(), full_test.y[instance_id, 0].numpy()
        )
        np.testing.assert_array_equal(demo_pred.predict(x_demo), full_pred.predict(x_full))

    baseline = FieldPredictor(bundle.baseline.model, bundle.baseline.stats, torch.device("cpu"), 32)
    x0 = bundle.test.x[0, 0].numpy()
    np.testing.assert_allclose(baseline.predict(x0), x0, rtol=1e-6, atol=1e-6)
    assert bundle.baseline.model.n_parameters == 0
    assert bundle.hv_slice.quantile == cfg.evaluation.high_vorticity_quantile


def test_size_limit_is_enforced(prepared_config: FlowBenchConfig, tmp_path: Path) -> None:
    cfg = prepared_config.model_copy(
        update={
            "demo": prepared_config.demo.model_copy(
                update={"dir": tmp_path / "d", "max_bytes": 1000}
            )
        }
    )
    run_train(cfg)
    with pytest.raises(ValueError, match="above the 1000 byte limit"):
        run_export_demo(cfg)
