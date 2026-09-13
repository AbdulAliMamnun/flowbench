import numpy as np
import plotly.graph_objects as go
import pytest
import torch

from flowbench.ui.panels import PANEL_TITLES, four_panels, per_sample_metrics, surface_figures


@pytest.fixture
def sample(synthetic_fields: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = synthetic_fields[0, 0].numpy()
    y = synthetic_fields[1, 0].numpy() * 3.0
    pred = y + 0.1 * synthetic_fields[2, 0].numpy()
    return x, y, pred


def test_per_sample_metrics_identity_is_zero(sample: tuple[np.ndarray, ...]) -> None:
    _, y, _ = sample
    m = per_sample_metrics(y, y, eps=1e-8)
    assert m == {"relative_l2": 0.0, "mae": 0.0, "enstrophy_rel_err": 0.0}


def test_four_panels_has_four_axes(sample: tuple[np.ndarray, ...]) -> None:
    x, y, pred = sample
    fig = four_panels(x, y, pred, "cnn")
    image_axes = [ax for ax in fig.axes if ax.images]
    assert len(image_axes) == 4
    assert [ax.get_title() for ax in image_axes][2] == "cnn prediction"


def test_surface_figures_shape_and_ranges(sample: tuple[np.ndarray, ...]) -> None:
    x, y, pred = sample
    figs = surface_figures(x, y, pred, "cnn", height=300)
    assert len(figs) == 4
    assert all(isinstance(f, go.Figure) for f in figs)
    vmax = float(np.abs(y).max())
    for fig, title in zip(figs, PANEL_TITLES, strict=True):
        trace = fig.data[0]
        assert isinstance(trace, go.Surface)
        assert np.asarray(trace.z).shape == (32, 32)
        assert fig.layout.height == 300
        assert title.split(" ")[-1] in fig.layout.title.text
    # input, reference and prediction share the reference's symmetric range
    for fig in figs[:3]:
        assert fig.data[0].cmin == pytest.approx(-vmax)
        assert fig.data[0].cmax == pytest.approx(vmax)
        assert list(fig.layout.scene.zaxis.range) == pytest.approx([-vmax, vmax])
    emax = float(np.abs(pred - y).max())
    assert figs[3].data[0].cmax == pytest.approx(emax)
    np.testing.assert_allclose(np.asarray(figs[3].data[0].z), pred - y, rtol=1e-6)


def test_surface_figures_handle_flat_fields() -> None:
    zeros = np.zeros((32, 32), dtype=np.float32)
    figs = surface_figures(zeros, zeros, zeros, "persistence")
    assert all(np.isfinite([f.data[0].cmin, f.data[0].cmax]).all() for f in figs)
