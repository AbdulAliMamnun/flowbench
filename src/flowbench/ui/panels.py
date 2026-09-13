"""Figure builders and per-sample metrics for the viewer.

Pure functions (NumPy, Torch, matplotlib, plotly; no Streamlit) so they can be unit
tested without running the page. ``app.py`` only wires them to widgets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import numpy as np
import plotly.graph_objects as go
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flowbench.evaluation.metrics import enstrophy_error, mae, relative_l2

if TYPE_CHECKING:
    from matplotlib.figure import Figure

PANEL_TITLES = ("input", "numerical reference", "prediction", "error (prediction − reference)")


def per_sample_metrics(pred: np.ndarray, ref: np.ndarray, eps: float) -> dict[str, float]:
    """Relative L2, MAE and enstrophy error for one ``(H, W)`` field pair."""
    p = torch.from_numpy(np.asarray(pred, dtype=np.float32))[None, None]
    r = torch.from_numpy(np.asarray(ref, dtype=np.float32))[None, None]
    rel, _ = relative_l2(p, r, eps)
    _, ens_rel, _ = enstrophy_error(p, r, eps)
    return {
        "relative_l2": float(rel[0]),
        "mae": float(mae(p, r)[0]),
        "enstrophy_rel_err": float(ens_rel[0]),
    }


def _limits(y: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    vmax = float(np.abs(y).max()) or 1.0
    emax = float(np.abs(err).max()) or 1.0
    return vmax, emax


def four_panels(x: np.ndarray, y: np.ndarray, pred: np.ndarray, model_name: str) -> Figure:
    """2D heatmaps: input, reference, prediction and error on a shared colour scale."""
    err = pred - y
    vmax, emax = _limits(y, err)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    panels = [
        (PANEL_TITLES[0], x, "RdBu_r", vmax),
        (PANEL_TITLES[1], y, "RdBu_r", vmax),
        (f"{model_name} {PANEL_TITLES[2]}", pred, "RdBu_r", vmax),
        (PANEL_TITLES[3], err, "PuOr_r", emax),
    ]
    for ax, (name, field, cmap, limit) in zip(axes, panels, strict=True):
        im = ax.imshow(field, cmap=cmap, vmin=-limit, vmax=limit, origin="lower")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def surface_figures(
    x: np.ndarray, y: np.ndarray, pred: np.ndarray, model_name: str, height: int = 380
) -> list[go.Figure]:
    """Plotly 3D surfaces (height = vorticity) for the same four panels.

    Input, reference and prediction share one z range and colour scale so the amplitude
    difference between input and target is visible as height; the error surface uses
    its own symmetric range.

    Args:
        x: Input field ``(H, W)``.
        y: Reference field ``(H, W)``.
        pred: Predicted field ``(H, W)``.
        model_name: Label for the prediction panel.
        height: Figure height in pixels.

    Returns:
        Four ``plotly.graph_objects.Figure`` objects in panel order.
    """
    err = pred - y
    vmax, emax = _limits(y, err)
    specs = [
        (PANEL_TITLES[0], x, "RdBu_r", vmax),
        (PANEL_TITLES[1], y, "RdBu_r", vmax),
        (f"{model_name} {PANEL_TITLES[2]}", pred, "RdBu_r", vmax),
        (PANEL_TITLES[3], err, "PuOr_r", emax),
    ]
    figures: list[go.Figure] = []
    for title, field, colorscale, limit in specs:
        fig = go.Figure(
            data=[
                go.Surface(
                    z=np.asarray(field, dtype=np.float64),
                    colorscale=colorscale,
                    cmin=-limit,
                    cmax=limit,
                    showscale=True,
                    colorbar={"thickness": 10, "len": 0.6},
                )
            ]
        )
        fig.update_layout(
            title={"text": title, "x": 0.5, "xanchor": "center"},
            height=height,
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
            scene={
                "zaxis": {"range": [-limit, limit], "title": "ω"},
                "xaxis": {"title": "", "showticklabels": False},
                "yaxis": {"title": "", "showticklabels": False},
                "aspectratio": {"x": 1, "y": 1, "z": 0.5},
            },
        )
        figures.append(fig)
    return figures
