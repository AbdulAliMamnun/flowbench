"""Fetch the Navier–Stokes archive from Zenodo into the git-ignored data directory."""

from __future__ import annotations

from pathlib import Path

from flowbench.config import DataConfig


def download_dataset(cfg: DataConfig) -> Path:
    """Download and extract the archive for ``cfg.source_resolution`` if not present.

    Args:
        cfg: Data configuration naming the Zenodo record, root directory and resolution.

    Returns:
        The directory containing the extracted ``.pt`` files.
    """
    msg = "download_dataset is implemented in Phase 1 (data)"
    raise NotImplementedError(msg)
