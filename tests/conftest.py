"""Shared fixtures: tiny synthetic vorticity fields. No network, no real data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from flowbench.config import FlowBenchConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

RESOLUTION = 32


@pytest.fixture(scope="session")
def resolution() -> int:
    return RESOLUTION


@pytest.fixture(scope="session")
def config_dir() -> Path:
    return CONFIG_DIR


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def synthetic_fields(rng: np.random.Generator) -> torch.Tensor:
    """Smooth periodic-looking fields, shape (n, 1, 32, 32), float32."""
    n = 16
    x = np.linspace(0.0, 2.0 * np.pi, RESOLUTION, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    fields = np.empty((n, 1, RESOLUTION, RESOLUTION), dtype=np.float32)
    for i in range(n):
        kx, ky = rng.integers(1, 4, size=2)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.5, 2.0)
        fields[i, 0] = amp * np.sin(kx * xx + phase) * np.cos(ky * yy)
    return torch.from_numpy(fields)


@pytest.fixture
def smoke_config_dict() -> dict[str, object]:
    with (CONFIG_DIR / "smoke.yaml").open("r", encoding="utf-8") as handle:
        data: dict[str, object] = yaml.safe_load(handle)
    return data


@pytest.fixture
def smoke_config() -> FlowBenchConfig:
    return load_config(CONFIG_DIR / "smoke.yaml")


@pytest.fixture
def tmp_config(tmp_path: Path, smoke_config_dict: dict[str, object]) -> Path:
    """A smoke config rewritten to point every path at ``tmp_path``."""
    cfg = dict(smoke_config_dict)
    run = dict(cfg["run"])  # type: ignore[call-overload]
    data = dict(cfg["data"])  # type: ignore[call-overload]
    run["artifacts_dir"] = str(tmp_path / "artifacts")
    run["device"] = "cpu"
    data["root_dir"] = str(tmp_path / "data")
    data["manifest_path"] = str(tmp_path / "data" / "manifest.json")
    cfg["run"] = run
    cfg["data"] = data
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path
