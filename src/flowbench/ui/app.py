"""Streamlit script: pick a held-out simulation and view four panels.

This module is executed by ``streamlit run`` (see :mod:`flowbench.ui.launch`); it is
never imported by library code. Usage::

    streamlit run src/flowbench/ui/app.py -- --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st
import torch

from flowbench.config import FlowBenchConfig, load_config
from flowbench.data import dataset as ds
from flowbench.evaluation.slices import HighVorticitySlice, high_vorticity_slice
from flowbench.models.registry import build_model
from flowbench.serving.predictor import FieldPredictor
from flowbench.training.checkpoint import LoadedCheckpoint, load_checkpoint
from flowbench.training.seed import resolve_device
from flowbench.ui.panels import four_panels, per_sample_metrics, surface_figures


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Read ``--config`` from the arguments after ``--``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    return parser.parse_args(argv)


@st.cache_resource(show_spinner="Loading checkpoint and held-out data…")
def load_everything(config_path: str) -> dict[str, Any]:
    """Load config, checkpoint, test pairs, baseline and slice once per session."""
    cfg: FlowBenchConfig = load_config(Path(config_path))
    device = resolve_device(cfg.run.device)
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
    cnn = FieldPredictor(ckpt.model, ckpt.stats, device, cfg.data.resolution)
    baseline = FieldPredictor(
        build_model(cfg.model, name="persistence"), ckpt.stats, device, cfg.data.resolution
    )
    metrics_path = cfg.run.run_dir / "report" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
    return {
        "cfg": cfg,
        "ckpt": ckpt,
        "test": test,
        "test_ids": test_ids,
        "hv": hv,
        "cnn": cnn,
        "baseline": baseline,
        "metrics": metrics,
        "device": device.type,
    }


def main() -> None:
    """Render the page."""
    args = parse_args(sys.argv[1:])
    st.set_page_config(page_title="FlowBench", layout="wide")
    st.title("FlowBench — held-out vorticity prediction")

    try:
        state = load_everything(str(args.config))
    except FileNotFoundError as exc:
        st.error(f"{exc}\n\nRun `flowbench prepare`, `train` and `evaluate` first.")
        st.stop()

    cfg: FlowBenchConfig = state["cfg"]
    test: ds.FieldPairs = state["test"]
    hv: HighVorticitySlice = state["hv"]
    ckpt: LoadedCheckpoint = state["ckpt"]
    hv_mask = hv.mask(test.y).numpy()

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
            f"{len(test)} held-out samples"
        )

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

    if state["metrics"] is not None:
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
            f"From `{cfg.run.run_dir / 'report' / 'metrics.json'}` generated "
            f"{state['metrics']['generated_at']} on `{state['metrics']['device']}`."
        )
    else:
        st.info("Run `flowbench evaluate` to see the full benchmark table here.")


main()
