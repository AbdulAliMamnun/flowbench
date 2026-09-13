"""Guarantees that no simulation contributes frames to more than one partition.

Runs on synthetic trajectories (no real data). The same invariants are asserted on the
persisted split file if `flowbench prepare` has been run locally.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flowbench.config import load_config
from flowbench.data.split import Split, detect_simulations, make_split


def _trajectories(n_sims: int, steps: int, res: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    frames = torch.randn(n_sims, steps + 1, 1, res, res)
    return frames[:, :-1].reshape(-1, 1, res, res), frames[:, 1:].reshape(-1, 1, res, res)


def _assert_no_leakage(split: Split, sim_ids: torch.Tensor) -> None:
    train = set(split.train_ids)
    val = set(split.val_ids)
    assert train.isdisjoint(val), "an instance appears in both train and val"
    train_sims = {int(sim_ids[i]) for i in train}
    val_sims = {int(sim_ids[i]) for i in val}
    assert train_sims.isdisjoint(val_sims), "a simulation contributes frames to train and val"
    assert train_sims == set(split.train_simulations)
    assert val_sims == set(split.val_simulations)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_no_leakage_on_synthetic_trajectories(seed: int) -> None:
    x, y = _trajectories(n_sims=12, steps=7)
    sim_ids = detect_simulations(x, y, atol=0.0)
    split = make_split(sim_ids, test_size=5, val_fraction=0.25, seed=seed)
    _assert_no_leakage(split, sim_ids)


def test_related_frames_never_straddle_the_boundary() -> None:
    """A val instance's input must not equal any train instance's target, and vice versa."""
    x, y = _trajectories(n_sims=6, steps=4)
    sim_ids = detect_simulations(x, y, atol=0.0)
    split = make_split(sim_ids, test_size=0, val_fraction=0.34, seed=5)
    train_targets = y[split.train_ids].flatten(1)
    val_inputs = x[split.val_ids].flatten(1)
    dist = torch.cdist(val_inputs, train_targets)
    assert dist.min() > 0.0


def test_persisted_split_has_no_leakage_if_present() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    split_path = cfg.data.split_path
    if not split_path.is_file():
        pytest.skip("run `flowbench prepare` to check the persisted split")
    split = Split.from_json(split_path)
    assert set(split.train_ids).isdisjoint(split.val_ids)
    assert set(split.train_simulations).isdisjoint(split.val_simulations)
    assert split.data_hashes, "persisted split must carry data hashes"
