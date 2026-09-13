import pytest
import torch

from flowbench.config import FlowBenchConfig
from flowbench.data import dataset as ds
from flowbench.data.normalize import normalize
from flowbench.training.checkpoint import REQUIRED_FILES, load_checkpoint
from flowbench.training.trainer import evaluate_mse, run_train

pytestmark = pytest.mark.slow


def test_train_writes_loadable_checkpoint(prepared_config: FlowBenchConfig) -> None:
    ckpt_dir = run_train(prepared_config)
    assert ckpt_dir == prepared_config.checkpoint_dir
    for name in REQUIRED_FILES:
        assert (ckpt_dir / name).is_file()

    loaded = load_checkpoint(ckpt_dir)
    metrics = loaded.metrics
    assert metrics["epochs_run"] >= 1
    assert metrics["best_epoch"] >= 0
    assert len(metrics["history"]) == metrics["epochs_run"]
    assert loaded.manifest["n_parameters"] == loaded.model.n_parameters


def test_reloaded_model_reproduces_validation_metric(prepared_config: FlowBenchConfig) -> None:
    ckpt_dir = run_train(prepared_config)
    loaded = load_checkpoint(ckpt_dir)
    pairs = ds.load_prepared(prepared_config.data, "train")
    val_ids = loaded.split.val_ids[: loaded.metrics["val_ids_used"]]
    loader = ds.make_loader(
        pairs.subset(val_ids), loaded.stats, batch_size=16, shuffle=False, seed=0
    )
    val_mse = evaluate_mse(loaded.model, loader, torch.device("cpu"))
    tol = prepared_config.evaluation.metrics_tolerance
    assert val_mse == pytest.approx(loaded.metrics["best_val_mse"], abs=tol, rel=tol)


def test_training_improves_on_persistence(prepared_config: FlowBenchConfig) -> None:
    ckpt_dir = run_train(prepared_config)
    loaded = load_checkpoint(ckpt_dir)
    pairs = ds.load_prepared(prepared_config.data, "train")
    val = pairs.subset(loaded.split.val_ids)
    x = normalize(val.x, loaded.stats)
    y = normalize(val.y, loaded.stats)
    with torch.no_grad():
        cnn_mse = float(torch.mean((loaded.model(x) - y) ** 2))
    persistence_mse = float(torch.mean((x - y) ** 2))
    assert cnn_mse < persistence_mse


def test_persistence_model_trains_to_a_checkpoint(prepared_config: FlowBenchConfig) -> None:
    cfg = prepared_config.model_copy(
        update={"model": prepared_config.model.model_copy(update={"name": "persistence"})}
    )
    ckpt_dir = run_train(cfg)
    loaded = load_checkpoint(ckpt_dir)
    assert loaded.model.n_parameters == 0
    assert loaded.metrics["epochs_run"] == 0
    assert loaded.metrics["best_val_mse"] == loaded.metrics["val_mse_before_training"]
