"""Streamlit script: pick a held-out simulation and view four panels.

This module is executed by ``streamlit run`` (see :mod:`flowbench.ui.launch`); it is
never imported by library code. Usage::

    streamlit run src/flowbench/ui/app.py -- --config configs/default.yaml

When the full data and artifacts are absent (hosted demo), the page falls back to the
bundle in ``cfg.demo.dir`` and says so in a banner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st
import torch

from flowbench.config import FlowBenchConfig, load_config
from flowbench.data import dataset as ds
from flowbench.evaluation.slices import HighVorticitySlice, high_vorticity_slice
from flowbench.models.registry import build_model
from flowbench.serving.predictor import FieldPredictor
from flowbench.training.checkpoint import LoadedCheckpoint, load_checkpoint
from flowbench.training.seed import resolve_device
from flowbench.ui.demo import bundle_available, load_bundle
from flowbench.ui.panels import four_panels, per_sample_metrics, surface_figures

if TYPE_CHECKING:
    import numpy as np


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Read ``--config`` from the arguments after ``--``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    return parser.parse_args(argv)


def full_data_available(cfg: FlowBenchConfig) -> bool:
    """Whether the trained checkpoint and the prepared caches exist locally."""
    return (
        (cfg.serving_checkpoint_dir / "model.pt").is_file()
        and ds.prepared_path(cfg.data, "test").is_file()
        and ds.prepared_path(cfg.data, "train").is_file()
    )


def _load_full(cfg: FlowBenchConfig, device: torch.device) -> dict[str, Any]:
    ckpt: LoadedCheckpoint = load_checkpoint(cfg.serving_checkpoint_dir, device)
    test_all = ds.load_prepared(cfg.data, "test")
    test_ids = ckpt.split.test_ids
    if cfg.data.max_test_samples is not None:
        test_ids = test_ids[: cfg.data.max_test_samples]
    test = test_all.subset(test_ids)
    train = ds.load_prepared(cfg.data, "train")
    hv = high_vorticity_slice(
        train.y[torch.as_tensor(ckpt.split.train_ids + ckpt.split.val_ids)],
        cfg.evaluation.high_vorticity_quantile,
    )
    baseline = build_model(cfg.model, name="persistence")
    metrics_path = cfg.run.run_dir / "report" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
    return {
        "mode": "full",
        "ckpt": ckpt,
        "cnn": FieldPredictor(ckpt.model, ckpt.stats, device, cfg.data.resolution),
        "baseline": FieldPredictor(baseline, ckpt.stats, device, cfg.data.resolution),
        "test": test,
        "test_ids": test_ids,
        "hv": hv,
        "metrics": metrics,
        "source": str(metrics_path),
    }


def _load_demo(cfg: FlowBenchConfig, device: torch.device) -> dict[str, Any]:
    bundle = load_bundle(cfg.demo.dir, device)
    return {
        "mode": "demo",
        "ckpt": bundle.checkpoint,
        "cnn": FieldPredictor(
            bundle.checkpoint.model, bundle.checkpoint.stats, device, cfg.data.resolution
        ),
        "baseline": FieldPredictor(
            bundle.baseline.model, bundle.baseline.stats, device, cfg.data.resolution
        ),
        "test": bundle.test,
        "test_ids": bundle.instance_ids,
        "hv": bundle.hv_slice,
        "metrics": bundle.metrics,
        "source": str(bundle.path / "metrics.json"),
        "bundle_manifest": bundle.manifest,
    }


@st.cache_resource(show_spinner="Loading checkpoint and held-out data…")
def load_everything(config_path: str) -> dict[str, Any]:
    """Load config plus either the full local run or the demo bundle, once per session."""
    cfg: FlowBenchConfig = load_config(Path(config_path))
    device = resolve_device(cfg.run.device)  # cpu whenever mps is unavailable
    if full_data_available(cfg):
        state = _load_full(cfg, device)
    elif bundle_available(cfg.demo.dir):
        state = _load_demo(cfg, device)
    else:
        msg = (
            f"neither the full run ({cfg.serving_checkpoint_dir}, {cfg.data.prepared_dir}) "
            f"nor a demo bundle ({cfg.demo.dir}) was found"
        )
        raise FileNotFoundError(msg)
    state["cfg"] = cfg
    state["device"] = device.type
    return state


def render_demo_banner(state: dict[str, Any]) -> None:
    """Say clearly when the page runs on the demo bundle instead of the full data."""
    manifest = state["bundle_manifest"]
    st.warning(
        f"Running on the {manifest['subset']['n_samples']}-sample demo subset "
        f"(seed {manifest['subset']['seed']}) of the "
        f"{manifest['subset']['n_available']} held-out test pairs, on `{state['device']}`. "
        "The full data and artifacts are not present on this host; the benchmark table "
        "below is a copy of the full-run report.",
        icon="ℹ️",
    )


def render_sidebar(state: dict[str, Any], hv_mask: np.ndarray) -> tuple[int, str, bool]:
    """Sample picker, model choice and view options; returns (position, model, show_3d)."""
    cfg: FlowBenchConfig = state["cfg"]
    test: ds.FieldPairs = state["test"]
    hv: HighVorticitySlice = state["hv"]
    ckpt: LoadedCheckpoint = state["ckpt"]
    with st.sidebar:
        st.header("Held-out simulation")
        only_hv = st.checkbox(
            f"High-vorticity slice only (max|ω| ≥ {hv.threshold:.3g}, "
            f"{int(hv_mask.sum())} samples)",
            value=False,
        )
        candidates = [i for i in range(len(test)) if (hv_mask[i] or not only_hv)]
        position = st.selectbox(
            "Test sample",
            options=candidates,
            format_func=lambda i: (
                f"test #{state['test_ids'][i]}" + (" (high vorticity)" if hv_mask[i] else "")
            ),
        )
        model_choice = st.radio("Model", options=[cfg.model.name, "persistence"], index=0)
        st.header("View")
        show_3d = st.toggle(
            "3D surface (height = vorticity)",
            value=False,
            help="Adds interactive Plotly surfaces below the 2D panels.",
        )
        st.caption(
            f"Checkpoint `{ckpt.model_version}` on `{state['device']}` · "
            f"{ckpt.manifest.get('n_parameters', 0):,} parameters · "
            f"{len(test)} held-out samples · mode: {state['mode']}"
        )
    return int(position), str(model_choice), bool(show_3d)


def render_benchmark_table(state: dict[str, Any]) -> None:
    """Full held-out benchmark from metrics.json, if available."""
    if state["metrics"] is None:
        st.info("Run `flowbench evaluate` to see the full benchmark table here.")
        return
    st.subheader("Benchmark on the full held-out set")
    rows = []
    for name, m in state["metrics"]["models"].items():
        a = m["all"]
        rows.append(
            {
                "model": name,
                "params": m["n_parameters"],
                "rel L2 mean": a["relative_l2"]["mean"],
                "rel L2 worst": a["relative_l2"]["worst"],
                "MAE": a["mae"]["mean"],
                "enstrophy rel err": a["enstrophy_relative_error"]["mean"],
                "p50 ms": m["latency_ms"]["p50_ms"],
                "p95 ms": m["latency_ms"]["p95_ms"],
            }
        )
    st.dataframe(rows, hide_index=True)
    st.caption(
        f"From `{state['source']}` generated {state['metrics']['generated_at']} "
        f"on `{state['metrics']['device']}`."
    )


def main() -> None:
    """Render the page."""
    args = parse_args(sys.argv[1:])
    st.set_page_config(page_title="FlowBench", layout="wide")
    st.title("FlowBench — held-out vorticity prediction")

    try:
        state = load_everything(str(args.config))
    except (FileNotFoundError, ValueError) as exc:
        st.error(
            f"{exc}\n\nRun `flowbench prepare`, `train` and `evaluate` locally, or "
            "`flowbench export-demo` to create the demo bundle."
        )
        st.stop()

    cfg: FlowBenchConfig = state["cfg"]
    test: ds.FieldPairs = state["test"]
    hv: HighVorticitySlice = state["hv"]
    hv_mask = hv.mask(test.y).numpy()

    if state["mode"] == "demo":
        render_demo_banner(state)
    position, model_choice, show_3d = render_sidebar(state, hv_mask)

    x = test.x[position, 0].numpy()
    y = test.y[position, 0].numpy()
    predictor: FieldPredictor = (
        state["cnn"] if model_choice == cfg.model.name else state["baseline"]
    )
    pred = predictor.predict(x)

    st.pyplot(four_panels(x, y, pred, model_choice), clear_figure=True)

    if show_3d:
        st.subheader("3D surface view")
        columns = st.columns(4)
        for col, fig in zip(columns, surface_figures(x, y, pred, model_choice), strict=True):
            col.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Input, reference and prediction share one height range; the error surface "
            "uses its own symmetric range. Drag to rotate, scroll to zoom."
        )

    eps = cfg.evaluation.relative_l2_epsilon
    chosen = per_sample_metrics(pred, y, eps)
    other_name = "persistence" if model_choice == cfg.model.name else cfg.model.name
    other_pred = (state["baseline"] if other_name == "persistence" else state["cnn"]).predict(x)
    other = per_sample_metrics(other_pred, y, eps)

    st.subheader(f"Metrics for this sample (test #{state['test_ids'][position]})")
    cols = st.columns(3)
    for col, (label, key) in zip(
        cols,
        [
            ("Relative L2", "relative_l2"),
            ("MAE", "mae"),
            ("Enstrophy rel. error", "enstrophy_rel_err"),
        ],
        strict=True,
    ):
        col.metric(
            f"{label} — {model_choice}",
            f"{chosen[key]:.4f}",
            delta=f"{chosen[key] - other[key]:+.4f} vs {other_name}",
            delta_color="inverse",
        )

    render_benchmark_table(state)


main()
