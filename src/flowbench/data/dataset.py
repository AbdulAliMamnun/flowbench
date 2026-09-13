"""Torch ``Dataset`` and ``DataLoader`` factories over the prepared split.

Raw tensors are read through ``neuraloperator``'s :class:`NavierStokesDataset` (the
project's only use of that library) and cached at the working resolution by
``prepare``. Every later step reads the small cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch.utils.data import DataLoader, Dataset

from flowbench.data.normalize import normalize
from flowbench.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from flowbench.config import DataConfig
    from flowbench.data.normalize import NormalizationStats

log = get_logger(__name__)

Part = Literal["train", "test"]


@dataclass(frozen=True)
class FieldPairs:
    """Input/target vorticity fields in original units, ``(n, 1, H, W)`` float32."""

    x: torch.Tensor
    y: torch.Tensor

    def __post_init__(self) -> None:
        if self.x.shape != self.y.shape:
            msg = f"x and y must share a shape, got {tuple(self.x.shape)} vs {tuple(self.y.shape)}"
            raise ValueError(msg)
        if self.x.ndim != 4 or self.x.shape[1] != 1:
            msg = f"expected (n, 1, H, W), got {tuple(self.x.shape)}"
            raise ValueError(msg)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def subset(self, ids: list[int]) -> FieldPairs:
        """Select instances by index."""
        idx = torch.as_tensor(ids, dtype=torch.int64)
        return FieldPairs(self.x[idx], self.y[idx])


def subsample(fields: torch.Tensor, rate: int) -> torch.Tensor:
    """Stride-subsample the two trailing spatial axes of ``(n, c, H, W)`` fields.

    Equivalent to ``fields[:, :, ::rate, ::rate]``: keeps every ``rate``-th grid point
    starting at index 0 on both axes, with no anti-aliasing. Applied here rather than
    through the loader's ``subsampling_rate`` because, for archives stored without a
    channel axis, neuraloperator 2.0.0 strides only the first spatial axis.
    """
    if rate < 1:
        msg = f"subsampling rate must be >= 1, got {rate}"
        raise ValueError(msg)
    return fields[:, :, ::rate, ::rate].contiguous()


def load_raw(cfg: DataConfig, part: Part) -> FieldPairs:
    """Load one archive file through ``NavierStokesDataset`` at the working resolution.

    Args:
        cfg: Data configuration (root, resolutions, subsampling).
        part: ``"train"`` or ``"test"``.

    Returns:
        All instances of that file as float32 ``(n, 1, resolution, resolution)``.
    """
    import contextlib
    import io

    from neuralop.data.datasets.navier_stokes import NavierStokesDataset

    n_train = cfg.n_train_available if part == "train" else 1
    n_test = cfg.n_test_available if part == "test" else 1
    # The loader prints progress with print(); capture it so CLI output stays structured.
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = NavierStokesDataset(
            root_dir=cfg.root_dir,
            n_train=n_train,
            n_tests=[n_test],
            batch_size=1,
            test_batch_sizes=[1],
            train_resolution=cfg.source_resolution,
            test_resolutions=[cfg.source_resolution],
            encode_input=False,
            encode_output=False,
            subsampling_rate=None,
            download=False,
        )
    db = dataset.train_db if part == "train" else dataset.test_dbs[cfg.source_resolution]
    x = subsample(db.x.to(torch.float32), cfg.subsampling_rate)
    y = subsample(db.y.to(torch.float32), cfg.subsampling_rate)
    pairs = FieldPairs(x, y)
    if pairs.x.shape[-2:] != (cfg.resolution, cfg.resolution):
        msg = f"expected {cfg.resolution}x{cfg.resolution} fields, got {tuple(pairs.x.shape)}"
        raise ValueError(msg)
    log.info("loaded raw part", part=part, n=len(pairs), resolution=cfg.resolution)
    return pairs


def prepared_path(cfg: DataConfig, part: Part) -> Path:
    """Cache file for ``part`` at the working resolution."""
    return cfg.prepared_dir / f"nsforcing_{part}_{cfg.resolution}.pt"


def write_prepared(cfg: DataConfig, part: Part, pairs: FieldPairs) -> Path:
    """Write the working-resolution cache for ``part``."""
    path = prepared_path(cfg, part)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": pairs.x, "y": pairs.y}, path)
    return path


def load_prepared(cfg: DataConfig, part: Part) -> FieldPairs:
    """Read the working-resolution cache written by ``prepare``.

    Raises:
        FileNotFoundError: If ``prepare`` has not been run for this config.
    """
    path = prepared_path(cfg, part)
    if not path.is_file():
        msg = f"{path} not found; run `flowbench prepare` first"
        raise FileNotFoundError(msg)
    payload = torch.load(path, map_location="cpu")
    return FieldPairs(payload["x"], payload["y"])


class VorticityDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Normalised ``(input, target)`` pairs for one partition."""

    def __init__(self, pairs: FieldPairs, stats: NormalizationStats) -> None:
        self.x = normalize(pairs.x, stats)
        self.y = normalize(pairs.y, stats)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def make_loader(
    pairs: FieldPairs,
    stats: NormalizationStats,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a deterministic ``DataLoader`` over normalised pairs.

    Args:
        pairs: Fields in original units.
        stats: Train-only normalisation statistics.
        batch_size: Mini-batch size.
        shuffle: Whether to reshuffle each epoch (seeded generator).
        seed: Seed for the shuffle generator.
        num_workers: Loader worker processes (0 keeps everything in-process).
    """
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        VorticityDataset(pairs, stats),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        drop_last=False,
    )
