import json
from pathlib import Path

import pytest
import torch

from flowbench.config import FlowBenchConfig
from flowbench.data import dataset as ds
from flowbench.evaluation.report import BENCHMARK_BEGIN, evaluate_model, run_evaluate
from flowbench.evaluation.slices import high_vorticity_slice
from flowbench.training.checkpoint import load_checkpoint
from flowbench.training.trainer import run_train

pytestmark = pytest.mark.slow

METRIC_KEYS = ("relative_l2", "mae", "enstrophy_relative_error")


@pytest.fixture
def trained(prepared_config: FlowBenchConfig, tmp_path: Path) -> tuple[FlowBenchConfig, Path]:
    prepared_config = prepared_config.model_copy(
        update={"run": prepared_config.run.model_copy(update={"update_docs": True})}
    )
    run_train(prepared_config)
    docs = tmp_path / "docs" / "benchmark.md"
    docs.parent.mkdir()
    docs.write_text("# Benchmark\n", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# Readme\n", encoding="utf-8")
    report_dir = run_evaluate(prepared_config, docs_path=docs, readme_path=readme)
    return prepared_config, report_dir


def test_evaluate_writes_metrics_table_and_maps(trained: tuple[FlowBenchConfig, Path]) -> None:
    cfg, report_dir = trained
    metrics = json.loads((report_dir / "metrics.json").read_text())
    assert set(metrics["models"]) == {"persistence", "cnn"}
    for name in ("persistence", "cnn"):
        m = metrics["models"][name]
        for key in METRIC_KEYS:
            assert m["all"][key]["n"] == metrics["test"]["n_samples"]
            assert m["all"][key]["worst"] >= m["all"][key]["mean"]
        assert m["latency_ms"]["p95_ms"] >= m["latency_ms"]["p50_ms"]
        assert m["latency_ms"]["device"] == "cpu"
    assert metrics["models"]["persistence"]["n_parameters"] == 0
    assert metrics["models"]["cnn"]["n_parameters"] == 19105
    assert metrics["high_vorticity_slice"]["n_reference_fields"] > 0
    assert metrics["high_vorticity_slice"]["n_test_samples"] <= metrics["test"]["n_samples"]
    assert (
        metrics["test"]["n_samples"] == cfg.data.max_test_samples
        or metrics["test"]["n_samples"] == metrics["test"]["n_available"]
    )

    table = (report_dir / "benchmark.md").read_text()
    assert "| persistence |" in table and "| cnn |" in table
    maps = sorted(report_dir.glob("error_map_*.png"))
    assert len(maps) == cfg.evaluation.n_error_maps


def test_generated_blocks_are_written(
    trained: tuple[FlowBenchConfig, Path], tmp_path: Path
) -> None:
    docs = (tmp_path / "docs" / "benchmark.md").read_text()
    readme = (tmp_path / "README.md").read_text()
    assert docs.count(BENCHMARK_BEGIN) == 1
    assert readme.count(BENCHMARK_BEGIN) == 1
    assert readme.startswith("# Readme")


def test_reloaded_checkpoint_reproduces_metrics(trained: tuple[FlowBenchConfig, Path]) -> None:
    """Loading the checkpoint and re-scoring must match metrics.json within tolerance."""
    cfg, report_dir = trained
    recorded = json.loads((report_dir / "metrics.json").read_text())["models"]["cnn"]["all"]
    ckpt = load_checkpoint(cfg.checkpoint_dir, torch.device("cpu"))
    test = ds.load_prepared(cfg.data, "test").subset(
        ckpt.split.test_ids[: cfg.data.max_test_samples]
    )
    train = ds.load_prepared(cfg.data, "train")
    hv = high_vorticity_slice(
        train.y[torch.as_tensor(ckpt.split.train_ids + ckpt.split.val_ids)],
        cfg.evaluation.high_vorticity_quantile,
    )
    again, _ = evaluate_model("cnn", ckpt.model, ckpt.stats, test, hv, cfg, torch.device("cpu"))
    tol = cfg.evaluation.metrics_tolerance
    for key in METRIC_KEYS:
        for stat in ("mean", "worst", "median"):
            assert again["all"][key][stat] == pytest.approx(recorded[key][stat], rel=tol, abs=tol)


def test_persistence_metrics_are_exact_identity(trained: tuple[FlowBenchConfig, Path]) -> None:
    cfg, report_dir = trained
    m = json.loads((report_dir / "metrics.json").read_text())["models"]["persistence"]["all"]
    ckpt = load_checkpoint(cfg.checkpoint_dir)
    test = ds.load_prepared(cfg.data, "test").subset(
        ckpt.split.test_ids[: cfg.data.max_test_samples]
    )
    expected = float(
        ((test.y - test.x).flatten(1).norm(dim=1) / test.y.flatten(1).norm(dim=1)).mean()
    )
    assert m["relative_l2"]["mean"] == pytest.approx(expected, rel=1e-5)
