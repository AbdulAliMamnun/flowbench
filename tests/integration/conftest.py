"""Integration fixtures: a fully prepared synthetic dataset under a temporary directory.

Mimics what `flowbench prepare` writes (cache, split, normalisation) so `train` and
`evaluate` run end to end on CPU without the Zenodo archive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from flowbench.config import FlowBenchConfig, load_config
from flowbench.data import dataset as ds
from flowbench.data.normalize import fit_normalization
from flowbench.data.split import detect_simulations, make_split, sha256_of_tensor

N_SIMS = 12
STEPS = 8
N_TEST = 24


def _trajectories(n_sims: int, steps: int, res: int, rng: np.random.Generator) -> ds.FieldPairs:
    """Slowly rotating smooth fields so y is close to, but not equal to, x."""
    x = np.linspace(0.0, 2.0 * np.pi, res, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    frames = np.empty((n_sims, steps + 1, 1, res, res), dtype=np.float32)
    for s in range(n_sims):
        kx, ky = rng.integers(1, 4, size=2)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.5, 2.0)
        for t in range(steps + 1):
            frames[s, t, 0] = amp * np.sin(kx * xx + phase + 0.1 * t) * np.cos(ky * yy - 0.05 * t)
    xs = torch.from_numpy(frames[:, :-1].reshape(-1, 1, res, res))
    ys = torch.from_numpy(frames[:, 1:].reshape(-1, 1, res, res))
    return ds.FieldPairs(xs, ys)


@pytest.fixture
def prepared_config(tmp_path: Path, tmp_config: Path) -> FlowBenchConfig:
    """Smoke config pointing at tmp_path with synthetic prepared data in place."""
    cfg = load_config(tmp_config)
    rng = np.random.default_rng(0)
    train = _trajectories(N_SIMS, STEPS, cfg.data.resolution, rng)
    test = _trajectories(N_TEST // 2, 1, cfg.data.resolution, rng)
    ds.write_prepared(cfg.data, "train", train)
    ds.write_prepared(cfg.data, "test", test)

    sim_ids = detect_simulations(train.x, train.y, atol=cfg.data.trajectory_link_atol)
    split = make_split(sim_ids, len(test), cfg.data.val_fraction, cfg.data.split_seed)
    split = type(split)(
        **{
            **split.__dict__,
            "data_hashes": {
                "prepared_train_32": sha256_of_tensor(torch.stack([train.x, train.y])),
                "prepared_test_32": sha256_of_tensor(torch.stack([test.x, test.y])),
            },
            "notes": {"simulation_grouping": "synthetic trajectories"},
        }
    )
    split.to_json(cfg.data.split_path)
    fit_normalization(train.x[split.train_ids], train.y[split.train_ids]).to_json(
        cfg.data.normalization_path
    )
    return cfg


@pytest.fixture
def prepared_config_path(prepared_config: FlowBenchConfig, tmp_path: Path) -> Path:
    """Path of a YAML file equal to ``prepared_config`` (for CLI invocations)."""
    path = tmp_path / "prepared.yaml"
    path.write_text(yaml.safe_dump(prepared_config.model_dump(mode="json")), encoding="utf-8")
    return path
