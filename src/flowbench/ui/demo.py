"""Self-contained demo bundle for hosted deployments without ``data/`` or ``artifacts/``.

``flowbench export-demo`` writes::

    demo/
      MANIFEST.json              provenance, subset IDs, slice threshold, SHA-256 of every file
      metrics.json               copy of the full-run report (if present)
      test_subset.pt             {"x", "y", "instance_ids"}: seeded held-out subset, float32
      checkpoints/cnn/           the trained checkpoint directory, unchanged
      checkpoints/persistence/   a zero-parameter checkpoint with the same normalisation

The Streamlit app loads this bundle when the full data and artifacts are absent. The
high-vorticity threshold is stored in the manifest because the train/validation fields it
derives from are not shipped.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from flowbench import __version__
from flowbench.data import dataset as ds
from flowbench.data.split import sha256_of_file
from flowbench.evaluation.slices import HighVorticitySlice, high_vorticity_slice
from flowbench.logging import get_logger
from flowbench.models.registry import build_model
from flowbench.training.checkpoint import LoadedCheckpoint, load_checkpoint, save_checkpoint
from flowbench.training.seed import resolve_device

if TYPE_CHECKING:
    from flowbench.config import FlowBenchConfig

log = get_logger(__name__)

MANIFEST_NAME = "MANIFEST.json"
SUBSET_NAME = "test_subset.pt"
METRICS_NAME = "metrics.json"
CHECKPOINTS_DIR = "checkpoints"


@dataclass(frozen=True)
class DemoBundle:
    """Everything the viewer needs, loaded from a demo directory."""

    path: Path
    manifest: dict[str, Any]
    checkpoint: LoadedCheckpoint
    baseline: LoadedCheckpoint
    test: ds.FieldPairs
    instance_ids: list[int]
    hv_slice: HighVorticitySlice
    metrics: dict[str, Any] | None


def select_subset(test_ids: list[int], n_samples: int, seed: int) -> list[int]:
    """Pick ``n_samples`` held-out instance IDs with a seeded shuffle, sorted ascending."""
    n = min(n_samples, len(test_ids))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(test_ids), size=n, replace=False)
    return sorted(int(i) for i in chosen)


def hash_tree(root: Path) -> dict[str, dict[str, Any]]:
    """SHA-256 and size of every file under ``root`` except the manifest itself."""
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel == MANIFEST_NAME:
            continue
        entries[rel] = {"sha256": sha256_of_file(path), "size_bytes": path.stat().st_size}
    return entries


def run_export_demo(cfg: FlowBenchConfig) -> Path:
    """Write the demo bundle for the configured run.

    Requires ``prepare``, ``train`` and (optionally, for the benchmark table)
    ``evaluate`` to have run.

    Args:
        cfg: Full pipeline configuration; ``cfg.demo`` sets directory, size and seed.

    Returns:
        The bundle directory.

    Raises:
        ValueError: If the bundle exceeds ``cfg.demo.max_bytes``.
    """
    out = cfg.demo.dir
    if out.exists():
        shutil.rmtree(out)
    (out / CHECKPOINTS_DIR).mkdir(parents=True)

    ckpt = load_checkpoint(cfg.checkpoint_dir)
    shutil.copytree(ckpt.path, out / CHECKPOINTS_DIR / cfg.model.name)

    persistence_cfg = cfg.model_copy(
        update={"model": cfg.model.model_copy(update={"name": "persistence"})}
    )
    save_checkpoint(
        out / CHECKPOINTS_DIR / "persistence",
        build_model(cfg.model, name="persistence"),
        persistence_cfg,
        ckpt.stats,
        ckpt.split,
        {"selection_metric": "none", "note": "persistence has no parameters"},
        torch.device("cpu"),
        ckpt.manifest.get("determinism"),
    )

    test_all = ds.load_prepared(cfg.data, "test")
    ids = select_subset(ckpt.split.test_ids, cfg.demo.n_samples, cfg.demo.seed)
    subset = test_all.subset(ids)
    torch.save(
        {"x": subset.x, "y": subset.y, "instance_ids": torch.as_tensor(ids, dtype=torch.int64)},
        out / SUBSET_NAME,
    )

    train_all = ds.load_prepared(cfg.data, "train")
    hv = high_vorticity_slice(
        train_all.y[torch.as_tensor(ckpt.split.train_ids + ckpt.split.val_ids)],
        cfg.evaluation.high_vorticity_quantile,
    )

    report_metrics = cfg.run.run_dir / "report" / METRICS_NAME
    if report_metrics.is_file():
        shutil.copyfile(report_metrics, out / METRICS_NAME)

    files = hash_tree(out)
    total = sum(int(f["size_bytes"]) for f in files.values())
    manifest = {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "flowbench_version": __version__,
        "source_checkpoint": {
            "path": str(ckpt.path),
            "model_version": ckpt.model_version,
            "git_sha": ckpt.manifest.get("git_sha"),
        },
        "data_hashes": ckpt.data_hashes,
        "subset": {
            "seed": cfg.demo.seed,
            "n_samples": len(ids),
            "n_available": len(ckpt.split.test_ids),
            "instance_ids": ids,
            "resolution": cfg.data.resolution,
        },
        "high_vorticity_slice": hv.__dict__,
        "files": files,
        "total_bytes": total,
        "max_bytes": cfg.demo.max_bytes,
    }
    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if total > cfg.demo.max_bytes:
        msg = f"demo bundle is {total} bytes, above the {cfg.demo.max_bytes} byte limit"
        raise ValueError(msg)
    log.info("demo bundle written", path=str(out), n_samples=len(ids), total_bytes=total)
    return out


def verify_bundle(path: Path) -> list[str]:
    """Return a list of files whose hash or size differs from the manifest (empty if intact)."""
    manifest = json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
    problems: list[str] = []
    for rel, expected in manifest["files"].items():
        file = path / rel
        if not file.is_file():
            problems.append(f"{rel}: missing")
        elif sha256_of_file(file) != expected["sha256"]:
            problems.append(f"{rel}: hash mismatch")
    return problems


def bundle_available(path: Path) -> bool:
    """Whether ``path`` looks like a demo bundle."""
    return (path / MANIFEST_NAME).is_file() and (path / SUBSET_NAME).is_file()


def load_bundle(path: Path, device: torch.device | None = None) -> DemoBundle:
    """Load a demo bundle; the device defaults to CPU when no accelerator is available."""
    device = device or resolve_device("auto")
    manifest = json.loads((path / MANIFEST_NAME).read_text(encoding="utf-8"))
    problems = verify_bundle(path)
    if problems:
        msg = f"demo bundle {path} failed verification: {problems}"
        raise ValueError(msg)
    model_name = Path(manifest["source_checkpoint"]["path"]).name
    checkpoint = load_checkpoint(path / CHECKPOINTS_DIR / model_name, device)
    baseline = load_checkpoint(path / CHECKPOINTS_DIR / "persistence", device)
    payload = torch.load(path / SUBSET_NAME, map_location="cpu")
    test = ds.FieldPairs(payload["x"], payload["y"])
    metrics_path = path / METRICS_NAME
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
    )
    return DemoBundle(
        path=path,
        manifest=manifest,
        checkpoint=checkpoint,
        baseline=baseline,
        test=test,
        instance_ids=[int(i) for i in payload["instance_ids"].tolist()],
        hv_slice=HighVorticitySlice(**manifest["high_vorticity_slice"]),
        metrics=metrics,
    )
