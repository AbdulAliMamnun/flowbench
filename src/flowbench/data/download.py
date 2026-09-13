"""Fetch the Navier–Stokes archive from Zenodo into the git-ignored data directory.

The download itself is delegated to ``neuraloperator``'s Zenodo helper, which verifies the
archive's MD5 checksum against the record and extracts the ``.pt`` files. FlowBench only
decides *whether* a download is needed, so a second call is a no-op once the tensors exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flowbench.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from flowbench.config import DataConfig

log = get_logger(__name__)

DATASET_PREFIX = "nsforcing"


def archive_name(resolution: int) -> str:
    """Name of the Zenodo archive holding ``resolution`` data."""
    return f"{DATASET_PREFIX}_{resolution}.tgz"


def tensor_file(root_dir: Path, split: str, resolution: int) -> Path:
    """Path of the extracted ``.pt`` file for ``split`` (``train`` or ``test``)."""
    return root_dir / f"{DATASET_PREFIX}_{split}_{resolution}.pt"


def dataset_is_present(root_dir: Path, resolution: int) -> bool:
    """Whether both extracted tensor files exist for ``resolution``."""
    return all(tensor_file(root_dir, split, resolution).is_file() for split in ("train", "test"))


def download_dataset(cfg: DataConfig) -> Path:
    """Download and extract the archive for ``cfg.source_resolution`` if not present.

    Args:
        cfg: Data configuration naming the Zenodo record, root directory and resolution.

    Returns:
        The directory containing the extracted ``.pt`` files.
    """
    root = cfg.root_dir
    root.mkdir(parents=True, exist_ok=True)
    if dataset_is_present(root, cfg.source_resolution):
        log.info("dataset already present", root=str(root), resolution=cfg.source_resolution)
        return root

    from neuralop.data.datasets.web_utils import download_from_zenodo_record

    name = archive_name(cfg.source_resolution)
    log.info("downloading archive", record=cfg.zenodo_record_id, archive=name, root=str(root))
    download_from_zenodo_record(
        record_id=cfg.zenodo_record_id, root=root, files_to_download=[name]
    )
    if not dataset_is_present(root, cfg.source_resolution):
        msg = f"archive {name} was downloaded but the expected .pt files are missing in {root}"
        raise FileNotFoundError(msg)
    log.info("dataset ready", root=str(root))
    return root
