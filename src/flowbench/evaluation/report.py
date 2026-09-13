"""Evaluation driver: metrics.json, markdown table and error-map PNGs.

``run_evaluate`` loads the trained checkpoint, builds the persistence baseline with the
same normalisation, scores both on the held-out test set (all samples and the
high-vorticity slice), times the single-sample path, and writes:

- ``<run_dir>/report/metrics.json``
- ``<run_dir>/report/benchmark.md``  (table + methodology)
- ``<run_dir>/report/error_map_<i>.png``
- the generated blocks in ``docs/benchmark.md`` and ``README.md``

Numbers in the docs are never typed by hand; they are rendered from ``metrics.json``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from flowbench import __version__
from flowbench.data import dataset as ds
from flowbench.data.inspect import update_generated_section
from flowbench.evaluation.latency import measure_latency
from flowbench.evaluation.metrics import Summary, enstrophy_error, mae, relative_l2
from flowbench.evaluation.slices import HighVorticitySlice, high_vorticity_slice
from flowbench.logging import get_logger
from flowbench.models.registry import build_model
from flowbench.serving.predictor import FieldPredictor
from flowbench.training.checkpoint import load_checkpoint
from flowbench.training.seed import resolve_device, seed_everything

if TYPE_CHECKING:
    from flowbench.config import FlowBenchConfig
    from flowbench.data.normalize import NormalizationStats
    from flowbench.models.base import Predictor

log = get_logger(__name__)

DOCS_BENCHMARK_PATH = Path("docs/benchmark.md")
README_PATH = Path("README.md")
BENCHMARK_BEGIN = "<!-- BEGIN GENERATED: flowbench evaluate -->"
BENCHMARK_END = "<!-- END GENERATED -->"


def score(pred: torch.Tensor, ref: torch.Tensor, eps: float) -> dict[str, Any]:
    """All per-sample metrics for a prediction/reference pair, aggregated."""
    rel, floored = relative_l2(pred, ref, eps)
    abs_err = mae(pred, ref)
    ens_abs, ens_rel, ens_floored = enstrophy_error(pred, ref, eps)
    return {
        "n_samples": int(ref.shape[0]),
        "relative_l2": Summary.of(rel).__dict__,
        "relative_l2_floored_samples": int(floored.sum()),
        "mae": Summary.of(abs_err).__dict__,
        "enstrophy_relative_error": Summary.of(ens_rel).__dict__,
        "enstrophy_absolute_error": Summary.of(ens_abs).__dict__,
        "enstrophy_floored_samples": int(ens_floored.sum()),
    }


def evaluate_model(
    name: str,
    model: Predictor,
    stats: NormalizationStats,
    test: ds.FieldPairs,
    hv_slice: HighVorticitySlice,
    cfg: FlowBenchConfig,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Score one model on the test pairs and time its single-sample path.

    Returns:
        The metrics dictionary for ``name`` and the predictions in field units.
    """
    predictor = FieldPredictor(model, stats, device, cfg.data.resolution)
    pred = predictor.predict_batch(test.x, cfg.evaluation.batch_size)
    eps = cfg.evaluation.relative_l2_epsilon
    mask = hv_slice.mask(test.y)
    sample = test.x[0, 0].numpy()
    latency = measure_latency(
        lambda: predictor.predict(sample),
        device,
        cfg.evaluation.latency.warmup_iterations,
        cfg.evaluation.latency.timed_iterations,
    )
    result: dict[str, Any] = {
        "n_parameters": model.n_parameters,
        "all": score(pred, test.y, eps),
        "high_vorticity": score(pred[mask], test.y[mask], eps),
        "latency_ms": latency.as_dict(),
    }
    log.info(
        "model evaluated",
        model=name,
        rel_l2_mean=result["all"]["relative_l2"]["mean"],
        rel_l2_worst=result["all"]["relative_l2"]["worst"],
        p50_ms=latency.p50_ms,
    )
    return result, pred


def save_error_maps(
    directory: Path,
    test: ds.FieldPairs,
    predictions: dict[str, torch.Tensor],
    indices: list[int],
) -> list[Path]:
    """Write one PNG per test index: input, reference, prediction, error (per model)."""
    # matplotlib is imported lazily so that the non-interactive backend can be selected
    # before pyplot loads and so that evaluation without plots never imports it.
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    written: list[Path] = []
    for idx in indices:
        x = test.x[idx, 0].numpy()
        y = test.y[idx, 0].numpy()
        vmax = float(np.abs(y).max())
        rows = len(predictions)
        fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows), squeeze=False)
        for row, (name, pred) in enumerate(predictions.items()):
            p = pred[idx, 0].numpy()
            err = p - y
            panels = [
                ("input", x, "RdBu_r", vmax),
                ("reference", y, "RdBu_r", vmax),
                (f"{name} prediction", p, "RdBu_r", vmax),
                (f"{name} error", err, "PuOr_r", float(np.abs(err).max()) or 1.0),
            ]
            for ax, (title, field, cmap, limit) in zip(axes[row], panels, strict=True):
                im = ax.imshow(field, cmap=cmap, vmin=-limit, vmax=limit, origin="lower")
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"held-out test sample {idx}")
        fig.tight_layout()
        path = directory / f"error_map_{idx}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    return written


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}"


def render_benchmark_markdown(metrics: dict[str, Any]) -> str:
    """Render the results table and methodology from ``metrics.json`` content."""
    models = metrics["models"]
    lines = [BENCHMARK_BEGIN, ""]
    lines.append(
        f"_Generated {metrics['generated_at']} by `flowbench evaluate` on run "
        f"`{metrics['run']}` (flowbench {metrics['flowbench_version']}, git "
        f"`{str(metrics['git_sha'])[:8]}`, device `{metrics['device']}`)._"
    )
    lines += [
        "",
        "| Model | Params | Rel. L2 mean | Rel. L2 worst | MAE (field units) | "
        "Enstrophy rel. err. mean | Enstrophy rel. err. worst | p50 ms | p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, m in models.items():
        a = m["all"]
        lat = m["latency_ms"]
        lines.append(
            f"| {name} | {m['n_parameters']:,} | {_fmt(a['relative_l2']['mean'])} | "
            f"{_fmt(a['relative_l2']['worst'])} | {_fmt(a['mae']['mean'])} | "
            f"{_fmt(a['enstrophy_relative_error']['mean'])} | "
            f"{_fmt(a['enstrophy_relative_error']['worst'])} | "
            f"{lat['p50_ms']:.2f} | {lat['p95_ms']:.2f} |"
        )
    hv = metrics["high_vorticity_slice"]
    lines += [
        "",
        f"**High-vorticity slice** (reference max|ω| ≥ {_fmt(hv['threshold'])}, the "
        f"{hv['quantile']:.0%} quantile over {hv['n_reference_fields']} train+validation "
        f"reference fields; {hv['n_test_samples']} of {metrics['test']['n_samples']} test "
        "samples):",
        "",
        "| Model | Rel. L2 mean | Rel. L2 worst | MAE | Enstrophy rel. err. mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, m in models.items():
        h = m["high_vorticity"]
        lines.append(
            f"| {name} | {_fmt(h['relative_l2']['mean'])} | {_fmt(h['relative_l2']['worst'])} | "
            f"{_fmt(h['mae']['mean'])} | {_fmt(h['enstrophy_relative_error']['mean'])} |"
        )
    t = metrics["test"]
    lines += [
        "",
        "**Setup.** "
        f"{t['n_samples']} held-out test pairs at {t['resolution']}×{t['resolution']}; "
        f"relative L2 uses an epsilon floor of {metrics['relative_l2_epsilon']:g} on the "
        "reference norm (floored samples: "
        + ", ".join(f"{n} {m['all']['relative_l2_floored_samples']}" for n, m in models.items())
        + "). Enstrophy is 0.5·mean(ω²) over the grid because the domain size is unknown. "
        f"Latency is the batch-one preprocess+inference+postprocess path after "
        f"{metrics['latency_protocol']['warmup_iterations']} warm-up calls, over "
        f"{metrics['latency_protocol']['timed_iterations']} timed calls on "
        f"`{metrics['device']}`; HTTP overhead is not included.",
        "",
        BENCHMARK_END,
    ]
    return "\n".join(lines) + "\n"


def run_evaluate(
    cfg: FlowBenchConfig,
    docs_path: Path = DOCS_BENCHMARK_PATH,
    readme_path: Path = README_PATH,
) -> Path:
    """Evaluate persistence and the trained model on the held-out test set.

    Args:
        cfg: Full pipeline configuration.
        docs_path: Markdown file whose generated block receives the benchmark.
        readme_path: README whose generated block receives the same table.

    Returns:
        Path to the report directory.
    """
    seed_everything(cfg.run.seed)
    device = resolve_device(cfg.run.device)
    ckpt = load_checkpoint(cfg.checkpoint_dir, device)
    stats = ckpt.stats
    split = ckpt.split

    test_all = ds.load_prepared(cfg.data, "test")
    test_ids = split.test_ids
    if cfg.data.max_test_samples is not None:
        test_ids = test_ids[: cfg.data.max_test_samples]
    test = test_all.subset(test_ids)

    train_all = ds.load_prepared(cfg.data, "train")
    trainval_ref = train_all.y[torch.as_tensor(split.train_ids + split.val_ids)]
    hv_slice = high_vorticity_slice(trainval_ref, cfg.evaluation.high_vorticity_quantile)
    n_hv = int(hv_slice.mask(test.y).sum())
    log.info(
        "evaluating",
        device=device.type,
        n_test=len(test),
        hv_threshold=hv_slice.threshold,
        n_high_vorticity=n_hv,
    )

    baseline = build_model(cfg.model, name="persistence")
    models: dict[str, Predictor] = {"persistence": baseline, cfg.model.name: ckpt.model}
    results: dict[str, Any] = {}
    predictions: dict[str, torch.Tensor] = {}
    for name, model in models.items():
        results[name], predictions[name] = evaluate_model(
            name, model, stats, test, hv_slice, cfg, device
        )

    report_dir = cfg.run.run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "flowbench_version": __version__,
        "git_sha": ckpt.manifest.get("git_sha"),
        "run": cfg.run.name,
        "device": device.type,
        "checkpoint": {
            "path": str(ckpt.path),
            "model_version": ckpt.model_version,
            "n_parameters": ckpt.manifest.get("n_parameters"),
            "best_epoch": ckpt.metrics.get("best_epoch"),
            "best_val_mse_normalized": ckpt.metrics.get("best_val_mse"),
        },
        "data_hashes": ckpt.data_hashes,
        "test": {
            "n_samples": len(test),
            "n_available": len(test_all),
            "resolution": cfg.data.resolution,
        },
        "relative_l2_epsilon": cfg.evaluation.relative_l2_epsilon,
        "high_vorticity_slice": {
            **hv_slice.__dict__,
            "n_test_samples": n_hv,
        },
        "latency_protocol": cfg.evaluation.latency.model_dump(),
        "models": results,
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    table = render_benchmark_markdown(metrics)
    (report_dir / "benchmark.md").write_text(table, encoding="utf-8")
    if cfg.run.update_docs:
        update_generated_section(docs_path, table, BENCHMARK_BEGIN, BENCHMARK_END)
        if readme_path.is_file():
            update_generated_section(readme_path, table, BENCHMARK_BEGIN, BENCHMARK_END)
    else:
        log.info("docs not updated (run.update_docs is false)", run=cfg.run.name)

    n_maps = min(cfg.evaluation.n_error_maps, len(test))
    maps = save_error_maps(report_dir, test, predictions, list(range(n_maps)))
    log.info("report written", path=str(report_dir), error_maps=len(maps))
    return report_dir
