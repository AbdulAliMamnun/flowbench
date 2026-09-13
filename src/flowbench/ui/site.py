"""Static portfolio site export: replayed outputs, benchmark table, manifest and ONNX model.

``flowbench export-site`` writes, under ``cfg.site.dir``::

    data/samples.json     seeded held-out subset with input, reference, CNN and persistence
                          predictions (replayed on the export device) and per-sample metrics
    data/benchmark.json   the full held-out report (``metrics.json``), floats rounded
    data/manifest.json    git SHA, checkpoint and data hashes, device, unknowns, sample IDs
    model/cnn.onnx        the CNN with normalisation baked in (field units in, field units out)
    model/normalization.json  the same statistics, for transparency

The page (``site/index.html``) is hand-written; this module only produces its data.
"""

from __future__ import annotations

import json
import math
import shutil
import warnings
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from flowbench import __version__
from flowbench.data import dataset as ds
from flowbench.data.split import sha256_of_file
from flowbench.evaluation.metrics import enstrophy_error, mae, relative_l2
from flowbench.evaluation.slices import high_vorticity_slice
from flowbench.logging import get_logger
from flowbench.serving.predictor import FieldPredictor
from flowbench.training.checkpoint import LoadedCheckpoint, git_sha, load_checkpoint
from flowbench.training.seed import resolve_device

if TYPE_CHECKING:
    from pathlib import Path

    from flowbench.config import FlowBenchConfig
    from flowbench.data.normalize import NormalizationStats
    from flowbench.models.base import Predictor

log = get_logger(__name__)

SAMPLES_SCHEMA = "flowbench.site.samples/1"
ONNX_INPUT = "field"
ONNX_OUTPUT = "prediction"


class FieldUnitsModel(nn.Module):
    """Wrap a predictor so ONNX consumers pass and receive fields in original units."""

    def __init__(self, model: Predictor, stats: NormalizationStats) -> None:
        super().__init__()
        self.model = model
        self.mean: torch.Tensor
        self.std: torch.Tensor
        self.register_buffer("mean", torch.tensor(stats.mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(stats.std, dtype=torch.float32))

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Normalise, predict, denormalise."""
        out: torch.Tensor = self.model((field - self.mean) / self.std)
        return out * self.std + self.mean


def round_sig(value: float, digits: int) -> float:
    """Round to ``digits`` significant figures; zero and non-finite values pass through."""
    if value == 0.0 or not math.isfinite(value):
        return value
    return float(f"{value:.{digits}g}")


def round_nested(obj: object, digits: int) -> object:
    """Recursively round every float in lists/dicts to ``digits`` significant figures."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return round_sig(obj, digits)
    if isinstance(obj, list):
        return [round_nested(v, digits) for v in obj]
    if isinstance(obj, dict):
        return {k: round_nested(v, digits) for k, v in obj.items()}
    return obj


def field_rows(field: torch.Tensor, digits: int) -> list[list[float]]:
    """``(1, H, W)`` or ``(H, W)`` tensor to nested lists of rounded floats."""
    arr = field.detach().cpu().to(torch.float32).numpy().reshape(field.shape[-2], field.shape[-1])
    return [[round_sig(float(v), digits) for v in row] for row in arr]


def select_site_samples(
    test_ids: list[int], hv_mask: np.ndarray, n_samples: int, n_high_vorticity: int, seed: int
) -> list[int]:
    """Seeded pick of ``n_high_vorticity`` slice members plus the rest from outside it.

    Returns sorted instance IDs. If the slice has fewer members than requested, the
    shortfall is filled from the remaining pool so the total is still ``n_samples``.
    """
    ids = np.asarray(test_ids)
    hv_pool = ids[hv_mask]
    other_pool = ids[~hv_mask]
    rng = np.random.default_rng(seed)
    n_hv = min(n_high_vorticity, hv_pool.size)
    chosen_hv = rng.choice(hv_pool, size=n_hv, replace=False) if n_hv else np.empty(0, int)
    n_other = min(n_samples - n_hv, other_pool.size)
    chosen_other = (
        rng.choice(other_pool, size=n_other, replace=False) if n_other else np.empty(0, int)
    )
    return sorted(int(i) for i in np.concatenate([chosen_hv, chosen_other]))


def per_sample_metrics(pred: torch.Tensor, ref: torch.Tensor, eps: float) -> dict[str, float]:
    """Relative L2, MAE and enstrophy errors for one ``(1, 1, H, W)`` pair."""
    rel, _ = relative_l2(pred, ref, eps)
    ens_abs, ens_rel, _ = enstrophy_error(pred, ref, eps)
    return {
        "relative_l2": float(rel[0]),
        "mae": float(mae(pred, ref)[0]),
        "enstrophy_relative_error": float(ens_rel[0]),
        "enstrophy_absolute_error": float(ens_abs[0]),
    }


def export_onnx(
    model: Predictor, stats: NormalizationStats, path: Path, opset: int, resolution: int
) -> None:
    """Export ``model`` with normalisation baked in, batch axis dynamic, given opset."""
    wrapped = FieldUnitsModel(model, stats).cpu().eval()
    example = torch.zeros(1, 1, resolution, resolution)
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        # The TorchScript exporter is deprecated upstream but is the one that needs no extra
        # packages and handles circular padding; its deprecation notice is expected.
        warnings.simplefilter("ignore")
        torch.onnx.export(
            wrapped,
            (example,),
            str(path),
            input_names=[ONNX_INPUT],
            output_names=[ONNX_OUTPUT],
            opset_version=opset,
            dynamo=False,
            dynamic_axes={ONNX_INPUT: {0: "batch"}, ONNX_OUTPUT: {0: "batch"}},
        )
    import onnx  # noqa: PLC0415 - only needed at export time

    onnx.checker.check_model(onnx.load(str(path)))


def checkpoint_sha256(ckpt: LoadedCheckpoint) -> dict[str, str]:
    """SHA-256 of every file in the checkpoint directory."""
    return {p.name: sha256_of_file(p) for p in sorted(ckpt.path.iterdir()) if p.is_file()}


def run_export_site(cfg: FlowBenchConfig) -> Path:
    """Write the site's data files and ONNX model.

    Requires ``prepare``, ``train`` and ``evaluate`` to have run for ``cfg``.

    Args:
        cfg: Full pipeline configuration; ``cfg.site`` sets counts, seed, limits.

    Returns:
        The site directory.

    Raises:
        FileNotFoundError: If the evaluation report is missing.
        ValueError: If ``samples.json`` exceeds ``cfg.site.max_samples_bytes``.
    """
    site = cfg.site
    report = cfg.run.run_dir / "report" / "metrics.json"
    if not report.is_file():
        msg = f"{report} not found; run `flowbench evaluate` first"
        raise FileNotFoundError(msg)
    data_dir = site.dir / "data"
    model_dir = site.dir / "model"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg.run.device)
    ckpt = load_checkpoint(cfg.checkpoint_dir, device)
    stats = ckpt.stats
    test_all = ds.load_prepared(cfg.data, "test")
    train_all = ds.load_prepared(cfg.data, "train")
    hv = high_vorticity_slice(
        train_all.y[torch.as_tensor(ckpt.split.train_ids + ckpt.split.val_ids)],
        cfg.evaluation.high_vorticity_quantile,
    )
    test_ids = ckpt.split.test_ids
    hv_mask = hv.mask(test_all.y[torch.as_tensor(test_ids)]).numpy()
    ids = select_site_samples(test_ids, hv_mask, site.n_samples, site.n_high_vorticity, site.seed)

    cnn = FieldPredictor(ckpt.model, stats, device, cfg.data.resolution)
    eps = cfg.evaluation.relative_l2_epsilon
    digits = site.significant_digits
    samples: list[dict[str, Any]] = []
    for instance_id in ids:
        x = test_all.x[instance_id : instance_id + 1]
        y = test_all.y[instance_id : instance_id + 1]
        pred_cnn = cnn.predict_batch(x, 1)
        # Persistence is the identity in field units; use the input itself so the saved
        # field and metrics carry no float32 normalisation round-trip noise.
        pred_base = x
        samples.append(
            {
                "instance_id": instance_id,
                "high_vorticity": bool(hv.mask(y)[0]),
                "input": field_rows(x[0], digits),
                "reference": field_rows(y[0], digits),
                "cnn": field_rows(pred_cnn[0], digits),
                "persistence": field_rows(pred_base[0], digits),
                "metrics": {
                    "cnn": round_nested(per_sample_metrics(pred_cnn, y, eps), digits),
                    "persistence": round_nested(per_sample_metrics(pred_base, y, eps), digits),
                },
            }
        )

    generated_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    provenance = {
        "kind": "replay",
        "device": device.type,
        "model_version": ckpt.model_version,
        "checkpoint_git_sha": ckpt.manifest.get("git_sha"),
        "export_git_sha": git_sha(),
        "generated_at": generated_at,
        "note": (
            "Fields and predictions were computed once by flowbench export-site and saved; "
            "persistence prediction equals the input field by definition. Single-sample "
            "metrics are not held-out benchmark performance."
        ),
    }
    samples_doc = {
        "schema": SAMPLES_SCHEMA,
        "provenance": provenance,
        "resolution": cfg.data.resolution,
        "normalization": {
            "mean": round_sig(stats.mean, digits),
            "std": round_sig(stats.std, digits),
        },
        "high_vorticity_slice": round_nested(hv.__dict__, digits),
        "relative_l2_epsilon": eps,
        "n_samples": len(samples),
        "n_high_vorticity": sum(1 for s in samples if s["high_vorticity"]),
        "samples": samples,
    }
    samples_path = data_dir / "samples.json"
    samples_path.write_text(
        json.dumps(samples_doc, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    size = samples_path.stat().st_size
    if size > site.max_samples_bytes:
        msg = f"samples.json is {size} bytes, above the {site.max_samples_bytes} byte limit"
        raise ValueError(msg)

    benchmark = round_nested(json.loads(report.read_text(encoding="utf-8")), digits)
    (data_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=1) + "\n", encoding="utf-8"
    )

    onnx_path = model_dir / "cnn.onnx"
    export_onnx(ckpt.model, stats, onnx_path, site.onnx_opset, cfg.data.resolution)
    shutil.copyfile(ckpt.path / "normalization.json", model_dir / "normalization.json")

    data_manifest = cfg.data.manifest_path
    unknowns = (
        json.loads(data_manifest.read_text(encoding="utf-8")).get("unknowns", {})
        if data_manifest.is_file()
        else {}
    )
    manifest = {
        "generated_at": generated_at,
        "flowbench_version": __version__,
        "torch_version": torch.__version__,
        "git_sha": git_sha(),
        "device": device.type,
        "checkpoint": {
            "path": str(ckpt.path),
            "model_version": ckpt.model_version,
            "git_sha": ckpt.manifest.get("git_sha"),
            "n_parameters": ckpt.manifest.get("n_parameters"),
            "files_sha256": checkpoint_sha256(ckpt),
        },
        "data_hashes": ckpt.data_hashes,
        "onnx": {
            "path": "model/cnn.onnx",
            "opset": site.onnx_opset,
            "sha256": sha256_of_file(onnx_path),
            "input": ONNX_INPUT,
            "output": ONNX_OUTPUT,
            "units": "field units in and out; normalisation baked in",
        },
        "samples": {
            "path": "data/samples.json",
            "sha256": sha256_of_file(samples_path),
            "size_bytes": size,
            "seed": site.seed,
            "instance_ids": ids,
            "n_high_vorticity": samples_doc["n_high_vorticity"],
        },
        "benchmark": {"path": "data/benchmark.json", "source": str(report)},
        "significant_digits": digits,
        "unknowns": unknowns,
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    log.info(
        "site assets written",
        dir=str(site.dir),
        n_samples=len(samples),
        samples_bytes=size,
        onnx_bytes=onnx_path.stat().st_size,
        device=device.type,
    )
    return site.dir
