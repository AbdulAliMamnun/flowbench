from pathlib import Path

import pytest
import torch

from flowbench.config import FlowBenchConfig
from flowbench.data.normalize import NormalizationStats
from flowbench.data.split import Split
from flowbench.models.registry import build_model
from flowbench.training.checkpoint import REQUIRED_FILES, load_checkpoint, save_checkpoint


@pytest.fixture
def split() -> Split:
    return Split(
        train_ids=[0, 1, 2],
        val_ids=[3],
        test_ids=[0, 1],
        train_simulations=[0, 1, 2],
        val_simulations=[3],
        seed=0,
        val_fraction=0.25,
        data_hashes={"raw_train_128": "deadbeef"},
    )


def test_roundtrip_reproduces_outputs(
    tmp_path: Path, smoke_config: FlowBenchConfig, split: Split, synthetic_fields: torch.Tensor
) -> None:
    torch.manual_seed(1)
    model = build_model(smoke_config.model)
    stats = NormalizationStats(mean=0.1, std=2.0, n_fields=4)
    metrics = {"best_val_mse": 0.5, "best_epoch": 2}
    target = tmp_path / "ckpt"
    save_checkpoint(target, model, smoke_config, stats, split, metrics, torch.device("cpu"))
    for name in REQUIRED_FILES:
        assert (target / name).is_file()

    loaded = load_checkpoint(target)
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(synthetic_fields), model(synthetic_fields))
    assert loaded.config == smoke_config
    assert loaded.stats == stats
    assert loaded.split == split
    assert loaded.data_hashes == {"raw_train_128": "deadbeef"}
    assert loaded.metrics["best_val_mse"] == 0.5
    assert loaded.manifest["model_name"] == "cnn"
    assert loaded.manifest["n_parameters"] == model.n_parameters
    assert loaded.manifest["device"] == "cpu"
    assert "git_sha" in loaded.manifest
    assert not loaded.model.training
    assert loaded.model_version.startswith("cnn-")


def test_missing_file_is_reported(
    tmp_path: Path, smoke_config: FlowBenchConfig, split: Split
) -> None:
    target = tmp_path / "ckpt"
    stats = NormalizationStats(mean=0.0, std=1.0, n_fields=1)
    save_checkpoint(
        target, build_model(smoke_config.model), smoke_config, stats, split, {}, torch.device("cpu")
    )
    (target / "normalization.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"normalization\.json"):
        load_checkpoint(target)
