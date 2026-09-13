"""Checkpoint directory format: weights plus everything needed to reproduce metrics.

A checkpoint is a directory::

    model.pt            state_dict
    config.yaml         the full resolved FlowBenchConfig
    normalization.json  train-only statistics used to normalise inputs/targets
    split_ids.json      the Split (instance IDs, simulation IDs, seed)
    data_hashes.json    SHA-256 of raw and prepared tensors
    metrics.json        training curve and the validation metric used for selection
    MANIFEST.json       git SHA, timestamp, device, package/torch versions, determinism

Loading rebuilds the model from ``config.yaml`` through the registry, so a checkpoint
never depends on Python pickles of model classes.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from flowbench import __version__
from flowbench.config import FlowBenchConfig, load_config_from_string
from flowbench.data.normalize import NormalizationStats
from flowbench.data.split import Split
from flowbench.logging import get_logger
from flowbench.models.registry import build_model

if TYPE_CHECKING:
    from flowbench.models.base import Predictor

log = get_logger(__name__)

REQUIRED_FILES = (
    "model.pt",
    "config.yaml",
    "normalization.json",
    "split_ids.json",
    "data_hashes.json",
    "metrics.json",
    "MANIFEST.json",
)


def git_sha() -> str:
    """Current git commit, or ``"unknown"`` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def git_is_dirty() -> bool | None:
    """Whether the working tree has uncommitted changes (``None`` outside a repository)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip())


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Everything a checkpoint directory contains, deserialised."""

    path: Path
    model: Predictor
    config: FlowBenchConfig
    stats: NormalizationStats
    split: Split
    data_hashes: dict[str, str]
    metrics: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def model_version(self) -> str:
        """Identifier reported by the API: ``<model>-<package version>-<short sha>``."""
        return (
            f"{self.config.model.name}-{self.manifest.get('package_version', '?')}-"
            f"{str(self.manifest.get('git_sha', 'unknown'))[:8]}"
        )


def save_checkpoint(
    directory: Path,
    model: Predictor,
    cfg: FlowBenchConfig,
    stats: NormalizationStats,
    split: Split,
    metrics: dict[str, Any],
    device: torch.device,
    determinism: dict[str, Any] | None = None,
) -> Path:
    """Write a complete checkpoint directory.

    Args:
        directory: Target directory (created; existing files are overwritten).
        model: Trained predictor; its ``state_dict`` is saved on CPU.
        cfg: Full configuration used for the run.
        stats: Normalisation statistics fitted on the training partition.
        split: Split used for training and validation.
        metrics: Training metrics, including the selection criterion.
        device: Device the model was trained on.
        determinism: Record returned by ``configure_determinism``.

    Returns:
        ``directory``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, directory / "model.pt")
    (directory / "config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")
    stats.to_json(directory / "normalization.json")
    split.to_json(directory / "split_ids.json")
    _dump(directory / "data_hashes.json", split.data_hashes)
    _dump(directory / "metrics.json", metrics)
    manifest = {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
        "package_version": __version__,
        "torch_version": torch.__version__,
        "device": device.type,
        "model_name": cfg.model.name,
        "n_parameters": model.n_parameters,
        "seed": cfg.run.seed,
        "determinism": determinism or {},
        "files": list(REQUIRED_FILES),
    }
    _dump(directory / "MANIFEST.json", manifest)
    log.info("checkpoint saved", path=str(directory), n_parameters=model.n_parameters)
    return directory


def load_checkpoint(directory: Path, device: torch.device | None = None) -> LoadedCheckpoint:
    """Load a checkpoint directory and rebuild its model in eval mode.

    Args:
        directory: Directory written by :func:`save_checkpoint`.
        device: Device to place the model on (default CPU).

    Returns:
        A :class:`LoadedCheckpoint`.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    directory = Path(directory)
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        msg = f"checkpoint {directory} is missing {missing}"
        raise FileNotFoundError(msg)
    cfg = load_config_from_string((directory / "config.yaml").read_text(encoding="utf-8"))
    model = build_model(cfg.model)
    state = torch.load(directory / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    if device is not None:
        model.to(device)
    return LoadedCheckpoint(
        path=directory,
        model=model,
        config=cfg,
        stats=NormalizationStats.from_json(directory / "normalization.json"),
        split=Split.from_json(directory / "split_ids.json"),
        data_hashes=_load(directory / "data_hashes.json"),
        metrics=_load(directory / "metrics.json"),
        manifest=_load(directory / "MANIFEST.json"),
    )


def _dump(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"{path} must hold a JSON object"
        raise TypeError(msg)
    return payload
