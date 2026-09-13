"""Simulation-level train/validation/test splitting with persisted IDs.

The unit of splitting is a *simulation*, never a sample. Simulation membership is
inferred by :func:`detect_simulations`: if the target of instance ``i`` equals the input
of instance ``i + 1`` the two are consecutive frames of one trajectory. When no such
links exist every instance is its own simulation and the manifest records that the
archive exposes no trajectory grouping.

The archive's own test file is the untouched held-out set; validation is carved out of
the archive's train file by simulation ID.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from flowbench.data import dataset as ds
from flowbench.data.download import download_dataset, tensor_file
from flowbench.data.normalize import fit_normalization
from flowbench.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from flowbench.config import DataConfig, FlowBenchConfig

log = get_logger(__name__)


@dataclass(frozen=True)
class Split:
    """Instance indices per partition plus the simulation IDs they came from."""

    train_ids: list[int]
    val_ids: list[int]
    test_ids: list[int]
    train_simulations: list[int]
    val_simulations: list[int]
    seed: int
    val_fraction: float
    data_hashes: dict[str, str] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        """Persist the split so every later step uses identical IDs."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> Split:
        """Load a split written by :meth:`to_json`."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)

    def summary(self) -> dict[str, int]:
        """Instance and simulation counts per partition."""
        return {
            "train_instances": len(self.train_ids),
            "val_instances": len(self.val_ids),
            "test_instances": len(self.test_ids),
            "train_simulations": len(self.train_simulations),
            "val_simulations": len(self.val_simulations),
        }


def detect_simulations(x: torch.Tensor, y: torch.Tensor, atol: float) -> torch.Tensor:
    """Assign a simulation ID to every instance by chaining ``y[i] == x[i + 1]``.

    Args:
        x: Inputs, shape ``(n, ...)``.
        y: Targets, shape ``(n, ...)``.
        atol: Absolute tolerance for the equality test.

    Returns:
        Integer tensor of shape ``(n,)``; equal values mark one trajectory. When no
        instance links to its successor the result is ``arange(n)``.
    """
    n = int(x.shape[0])
    if n == 0:
        return torch.empty(0, dtype=torch.int64)
    if n == 1:
        return torch.zeros(1, dtype=torch.int64)
    diff = (y[:-1] - x[1:]).abs().flatten(1).amax(dim=1)
    linked = diff <= atol
    breaks = torch.cat([torch.tensor([True]), ~linked])
    return torch.cumsum(breaks.to(torch.int64), dim=0) - 1


def make_split(
    simulation_ids: torch.Tensor,
    test_size: int,
    val_fraction: float,
    seed: int,
) -> Split:
    """Partition instances into train and validation by simulation ID.

    Args:
        simulation_ids: Simulation ID of every instance in the archive's train file.
        test_size: Number of instances in the archive's test file (all held out).
        val_fraction: Fraction of *simulations* assigned to validation (at least one).
        seed: Seed of the shuffle that picks validation simulations.

    Returns:
        A :class:`Split` with disjoint train and validation instance sets.

    Raises:
        ValueError: If fewer than two simulations are available.
    """
    sims = torch.unique(simulation_ids).numpy()
    if sims.size < 2:
        msg = "need at least two simulations to carve out a validation set"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    order = rng.permutation(sims)
    n_val = max(1, round(val_fraction * int(sims.size)))
    val_sims = np.sort(order[:n_val])
    train_sims = np.sort(order[n_val:])
    sim_np = simulation_ids.numpy()
    val_mask = np.isin(sim_np, val_sims)
    return Split(
        train_ids=[int(i) for i in np.flatnonzero(~val_mask)],
        val_ids=[int(i) for i in np.flatnonzero(val_mask)],
        test_ids=list(range(test_size)),
        train_simulations=[int(s) for s in train_sims],
        val_simulations=[int(s) for s in val_sims],
        seed=seed,
        val_fraction=val_fraction,
    )


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 digest of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_tensor(t: torch.Tensor) -> str:
    """Hex SHA-256 digest of a tensor's contiguous float32 bytes."""
    arr = np.ascontiguousarray(t.detach().cpu().to(torch.float32).numpy())
    return hashlib.sha256(arr.tobytes()).hexdigest()


def raw_data_hashes(cfg: DataConfig) -> dict[str, str]:
    """SHA-256 of both raw archive files at the source resolution."""
    return {
        f"raw_{part}_{cfg.source_resolution}": sha256_of_file(
            tensor_file(cfg.root_dir, part, cfg.source_resolution)
        )
        for part in ("train", "test")
    }


def run_prepare(cfg: FlowBenchConfig) -> Path:
    """Create the split, fit normalisation statistics and persist both.

    Steps: ensure the archive is present, load both files at the working resolution,
    cache them under ``data/prepared``, detect simulations, split by simulation ID,
    fit normalisation on the training partition only, write ``split_seed*.json`` and
    ``normalization_seed*.json``.

    Args:
        cfg: Full pipeline configuration.

    Returns:
        Path to the persisted split file.
    """
    data_cfg = cfg.data
    download_dataset(data_cfg)

    train = ds.load_raw(data_cfg, "train")
    test = ds.load_raw(data_cfg, "test")
    log.info("raw data loaded", train=list(train.x.shape), test=list(test.x.shape))

    hashes = raw_data_hashes(data_cfg)
    hashes[f"prepared_train_{data_cfg.resolution}"] = sha256_of_tensor(
        torch.stack([train.x, train.y])
    )
    hashes[f"prepared_test_{data_cfg.resolution}"] = sha256_of_tensor(
        torch.stack([test.x, test.y])
    )
    ds.write_prepared(data_cfg, "train", train)
    ds.write_prepared(data_cfg, "test", test)

    simulation_ids = detect_simulations(train.x, train.y, atol=data_cfg.trajectory_link_atol)
    n_sims = int(torch.unique(simulation_ids).numel())
    grouping = (
        "trajectories detected by y[i] == x[i+1]"
        if n_sims < int(train.x.shape[0])
        else "no consecutive-frame links: each instance is its own simulation"
    )

    split = make_split(
        simulation_ids,
        test_size=int(test.x.shape[0]),
        val_fraction=data_cfg.val_fraction,
        seed=data_cfg.split_seed,
    )
    split = Split(
        **{**asdict(split), "data_hashes": hashes, "notes": {"simulation_grouping": grouping}}
    )
    split.to_json(data_cfg.split_path)

    stats = fit_normalization(train.x[split.train_ids], train.y[split.train_ids])
    stats.to_json(data_cfg.normalization_path)

    log.info(
        "split written",
        path=str(data_cfg.split_path),
        summary=split.summary(),
        mean=stats.mean,
        std=stats.std,
        grouping=grouping,
    )
    return data_cfg.split_path
