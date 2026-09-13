from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from flowbench.config import FlowBenchConfig, load_config, load_config_from_string


@pytest.mark.parametrize("name", ["default.yaml", "smoke.yaml"])
def test_shipped_configs_validate(name: str, config_dir: Path) -> None:
    cfg = load_config(config_dir / name)
    assert cfg.data.resolution == 32
    assert cfg.data.subsampling_rate == 4
    assert cfg.model.cnn.layers == 4
    assert cfg.model.cnn.channels == 32


def test_smoke_config_is_capped(smoke_config: FlowBenchConfig) -> None:
    assert smoke_config.data.max_train_samples is not None
    assert smoke_config.data.max_test_samples is not None
    assert smoke_config.training.epochs <= 3


def test_unknown_key_is_rejected(smoke_config_dict: dict[str, object]) -> None:
    text = yaml.safe_dump({**smoke_config_dict, "bogus": 1})
    with pytest.raises(ValidationError):
        load_config_from_string(text)


def test_resolution_must_divide_source(smoke_config_dict: dict[str, object]) -> None:
    data = dict(smoke_config_dict["data"])  # type: ignore[call-overload]
    data["resolution"] = 33
    text = yaml.safe_dump({**smoke_config_dict, "data": data})
    with pytest.raises(ValidationError, match="must divide"):
        load_config_from_string(text)


def test_even_kernel_is_rejected(smoke_config_dict: dict[str, object]) -> None:
    model = {"name": "cnn", "cnn": {"kernel_size": 4}}
    text = yaml.safe_dump({**smoke_config_dict, "model": model})
    with pytest.raises(ValidationError, match="odd"):
        load_config_from_string(text)


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_config: Path) -> None:
    monkeypatch.setenv("FLOWBENCH_RUN__SEED", "42")
    cfg = load_config(tmp_config)
    assert cfg.run.seed == 42


def test_config_round_trips_through_yaml(smoke_config: FlowBenchConfig) -> None:
    again = load_config_from_string(smoke_config.to_yaml())
    assert again == smoke_config


def test_derived_paths(smoke_config: FlowBenchConfig) -> None:
    assert smoke_config.run.run_dir == Path("artifacts/smoke")
    assert smoke_config.checkpoint_dir == Path("artifacts/smoke/checkpoints/cnn")
    assert smoke_config.serving_checkpoint_dir == smoke_config.checkpoint_dir
    assert smoke_config.data.split_path.name == "split_seed0.json"


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
