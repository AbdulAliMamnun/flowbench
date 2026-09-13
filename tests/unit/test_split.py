from pathlib import Path

import pytest
import torch

from flowbench.data.split import Split, detect_simulations, make_split, sha256_of_tensor


def _chain(n_sims: int, steps: int, res: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (x, y) where each simulation is a chain: y[i] == x[i+1] within a sim."""
    frames = torch.randn(n_sims, steps + 1, 1, res, res)
    x = frames[:, :-1].reshape(-1, 1, res, res)
    y = frames[:, 1:].reshape(-1, 1, res, res)
    return x, y


def test_detect_simulations_chains_consecutive_frames() -> None:
    x, y = _chain(n_sims=3, steps=5)
    ids = detect_simulations(x, y, atol=0.0)
    expected = torch.repeat_interleave(torch.arange(3), 5)
    assert torch.equal(ids, expected)


def test_detect_simulations_without_links_is_identity() -> None:
    x = torch.randn(7, 1, 4, 4)
    y = torch.randn(7, 1, 4, 4)
    ids = detect_simulations(x, y, atol=1e-6)
    assert torch.equal(ids, torch.arange(7))


def test_detect_simulations_edge_cases() -> None:
    assert detect_simulations(torch.empty(0, 1, 2, 2), torch.empty(0, 1, 2, 2), 0.0).numel() == 0
    assert torch.equal(
        detect_simulations(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), 0.0),
        torch.zeros(1, dtype=torch.int64),
    )


def test_make_split_is_disjoint_and_covers_everything() -> None:
    sim_ids = torch.repeat_interleave(torch.arange(10), 4)
    split = make_split(sim_ids, test_size=8, val_fraction=0.2, seed=3)
    assert set(split.train_ids).isdisjoint(split.val_ids)
    assert sorted(split.train_ids + split.val_ids) == list(range(40))
    assert split.test_ids == list(range(8))
    assert len(split.val_simulations) == 2
    assert set(split.train_simulations).isdisjoint(split.val_simulations)


def test_make_split_keeps_whole_simulations_together() -> None:
    sim_ids = torch.repeat_interleave(torch.arange(6), 5)
    split = make_split(sim_ids, test_size=0, val_fraction=0.34, seed=0)
    val_sims = {int(sim_ids[i]) for i in split.val_ids}
    train_sims = {int(sim_ids[i]) for i in split.train_ids}
    assert val_sims.isdisjoint(train_sims)
    assert val_sims == set(split.val_simulations)


def test_make_split_is_deterministic() -> None:
    sim_ids = torch.repeat_interleave(torch.arange(20), 3)
    a = make_split(sim_ids, test_size=1, val_fraction=0.1, seed=11)
    b = make_split(sim_ids, test_size=1, val_fraction=0.1, seed=11)
    c = make_split(sim_ids, test_size=1, val_fraction=0.1, seed=12)
    assert a == b
    assert a.val_ids != c.val_ids


def test_make_split_needs_two_simulations() -> None:
    with pytest.raises(ValueError, match="at least two"):
        make_split(torch.zeros(5, dtype=torch.int64), test_size=0, val_fraction=0.5, seed=0)


def test_split_json_roundtrip(tmp_path: Path) -> None:
    sim_ids = torch.repeat_interleave(torch.arange(4), 2)
    split = make_split(sim_ids, test_size=3, val_fraction=0.25, seed=1)
    split = Split(**{**split.__dict__, "data_hashes": {"raw": "abc"}, "notes": {"k": "v"}})
    path = tmp_path / "splits" / "s.json"
    split.to_json(path)
    assert Split.from_json(path) == split
    assert split.summary()["val_simulations"] == 1


def test_sha256_of_tensor_is_content_based() -> None:
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = a.clone()
    assert sha256_of_tensor(a) == sha256_of_tensor(b)
    b[0, 0] += 1.0
    assert sha256_of_tensor(a) != sha256_of_tensor(b)
