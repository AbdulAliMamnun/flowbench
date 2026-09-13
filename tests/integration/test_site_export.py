"""Site export: files, rounding, slice quota, ONNX agreement and metric consistency."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from flowbench.config import FlowBenchConfig
from flowbench.data import dataset as ds
from flowbench.evaluation.metrics import enstrophy_error, mae, relative_l2
from flowbench.evaluation.report import run_evaluate
from flowbench.evaluation.slices import high_vorticity_slice
from flowbench.serving.predictor import FieldPredictor
from flowbench.training.checkpoint import load_checkpoint
from flowbench.training.trainer import run_train
from flowbench.ui.site import (
    ONNX_INPUT,
    FieldUnitsModel,
    round_nested,
    round_sig,
    run_export_site,
    select_site_samples,
)

pytestmark = pytest.mark.slow

N_SITE = 8
N_HV = 2


@pytest.fixture
def exported(prepared_config: FlowBenchConfig, tmp_path: Path) -> tuple[FlowBenchConfig, Path]:
    cfg = prepared_config.model_copy(
        update={
            "site": prepared_config.site.model_copy(
                update={"dir": tmp_path / "site", "n_samples": N_SITE, "n_high_vorticity": N_HV}
            )
        }
    )
    run_train(cfg)
    run_evaluate(cfg, docs_path=tmp_path / "b.md", readme_path=tmp_path / "r.md")
    return cfg, run_export_site(cfg)


def test_round_sig() -> None:
    assert round_sig(0.123456789, 5) == 0.12346
    assert round_sig(-1234.5678, 5) == -1234.6
    assert round_sig(0.0, 5) == 0.0
    assert round_nested({"a": [1.23456789, {"b": 2}], "c": True}, 3) == {
        "a": [1.23, {"b": 2}],
        "c": True,
    }


def test_select_site_samples_honours_slice_quota() -> None:
    ids = list(range(100))
    hv = np.zeros(100, dtype=bool)
    hv[90:] = True
    chosen = select_site_samples(ids, hv, n_samples=24, n_high_vorticity=4, seed=0)
    assert len(chosen) == 24 and chosen == sorted(chosen)
    assert sum(1 for i in chosen if i >= 90) == 4
    assert chosen == select_site_samples(ids, hv, 24, 4, seed=0)
    assert chosen != select_site_samples(ids, hv, 24, 4, seed=1)


def test_files_written_and_rounded(exported: tuple[FlowBenchConfig, Path]) -> None:
    cfg, site = exported
    samples = json.loads((site / "data" / "samples.json").read_text())
    manifest = json.loads((site / "data" / "manifest.json").read_text())
    benchmark = json.loads((site / "data" / "benchmark.json").read_text())
    assert samples["n_samples"] == N_SITE
    ckpt = load_checkpoint(cfg.checkpoint_dir)
    test = ds.load_prepared(cfg.data, "test")
    train = ds.load_prepared(cfg.data, "train")
    hv = high_vorticity_slice(
        train.y[torch.as_tensor(ckpt.split.train_ids + ckpt.split.val_ids)],
        cfg.evaluation.high_vorticity_quantile,
    )
    available_hv = int(hv.mask(test.y[torch.as_tensor(ckpt.split.test_ids)]).sum())
    assert samples["n_high_vorticity"] == min(N_HV, available_hv)
    assert samples["provenance"]["kind"] == "replay"
    assert samples["provenance"]["device"] == "cpu"
    assert (site / "data" / "samples.json").stat().st_size <= cfg.site.max_samples_bytes
    for s in samples["samples"]:
        for key in ("input", "reference", "cnn", "persistence"):
            assert len(s[key]) == 32 and all(len(r) == 32 for r in s[key])
            for row in s[key]:
                for v in row:
                    assert v == round_sig(v, cfg.site.significant_digits)
        np.testing.assert_array_equal(np.asarray(s["persistence"]), np.asarray(s["input"]))
    assert manifest["samples"]["instance_ids"] == [s["instance_id"] for s in samples["samples"]]
    assert set(manifest["checkpoint"]["files_sha256"]) >= {"model.pt", "config.yaml"}
    assert manifest["data_hashes"] == ckpt.data_hashes
    assert manifest["onnx"]["opset"] == cfg.site.onnx_opset
    assert manifest["device"] == "cpu"
    assert set(benchmark["models"]) == {"persistence", "cnn"}
    assert (site / "model" / "cnn.onnx").is_file()
    assert (site / "model" / "normalization.json").is_file()


def test_onnx_agrees_with_torch_on_20_samples(exported: tuple[FlowBenchConfig, Path]) -> None:
    cfg, site = exported
    ckpt = load_checkpoint(cfg.checkpoint_dir, torch.device("cpu"))
    fields = ds.load_prepared(cfg.data, "train").x[:20]
    assert fields.shape[0] == 20
    session = ort.InferenceSession(
        str(site / "model" / "cnn.onnx"), providers=["CPUExecutionProvider"]
    )
    onnx_out = session.run(None, {ONNX_INPUT: fields.numpy()})[0]
    with torch.no_grad():
        torch_out = FieldUnitsModel(ckpt.model, ckpt.stats)(fields).numpy()
    predictor = FieldPredictor(ckpt.model, ckpt.stats, torch.device("cpu"), 32)
    served = predictor.predict_batch(fields, 8).numpy()
    assert onnx_out.shape == (20, 1, 32, 32)
    np.testing.assert_allclose(onnx_out, torch_out, atol=1e-5, rtol=0)
    np.testing.assert_allclose(onnx_out, served, atol=1e-5, rtol=0)


def test_sample_metrics_match_report_pipeline(exported: tuple[FlowBenchConfig, Path]) -> None:
    """Per-sample metrics in samples.json must match a fresh computation for the same IDs,
    and benchmark.json must be the rounded metrics.json."""
    cfg, site = exported
    samples = json.loads((site / "data" / "samples.json").read_text())
    ckpt = load_checkpoint(cfg.checkpoint_dir, torch.device("cpu"))
    test = ds.load_prepared(cfg.data, "test")
    predictor = FieldPredictor(ckpt.model, ckpt.stats, torch.device("cpu"), 32)
    eps = cfg.evaluation.relative_l2_epsilon
    for s in samples["samples"]:
        i = s["instance_id"]
        x, y = test.x[i : i + 1], test.y[i : i + 1]
        for name, pred in (("cnn", predictor.predict_batch(x, 1)), ("persistence", x)):
            rel, _ = relative_l2(pred, y, eps)
            _, ens_rel, _ = enstrophy_error(pred, y, eps)
            got = s["metrics"][name]
            assert got["relative_l2"] == pytest.approx(float(rel[0]), rel=1e-4, abs=1e-7)
            assert got["mae"] == pytest.approx(float(mae(pred, y)[0]), rel=1e-4, abs=1e-7)
            assert got["enstrophy_relative_error"] == pytest.approx(
                float(ens_rel[0]), rel=1e-4, abs=1e-7
            )
        # the saved replay fields reproduce the saved metrics
        cnn_arr = torch.tensor(s["cnn"])[None, None]
        ref_arr = torch.tensor(s["reference"])[None, None]
        rel, _ = relative_l2(cnn_arr, ref_arr, eps)
        assert s["metrics"]["cnn"]["relative_l2"] == pytest.approx(float(rel[0]), rel=1e-3)

    report = json.loads((cfg.run.run_dir / "report" / "metrics.json").read_text())
    benchmark = json.loads((site / "data" / "benchmark.json").read_text())
    assert benchmark == round_nested(report, cfg.site.significant_digits)


def test_missing_report_is_an_error(prepared_config: FlowBenchConfig, tmp_path: Path) -> None:
    cfg = prepared_config.model_copy(
        update={"site": prepared_config.site.model_copy(update={"dir": tmp_path / "s"})}
    )
    run_train(cfg)
    with pytest.raises(FileNotFoundError, match="evaluate"):
        run_export_site(cfg)
