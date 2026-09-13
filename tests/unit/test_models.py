import pytest
import torch

from flowbench.config import CNNConfig, ModelConfig
from flowbench.models.cnn import ConvNet
from flowbench.models.persistence import Persistence
from flowbench.models.registry import MODEL_REGISTRY, build_model


def test_persistence_is_identity_without_parameters(synthetic_fields: torch.Tensor) -> None:
    model = Persistence()
    out = model(synthetic_fields)
    torch.testing.assert_close(out, synthetic_fields)
    assert out.data_ptr() != synthetic_fields.data_ptr()
    assert model.n_parameters == 0
    assert not model.is_trainable


@pytest.mark.parametrize("padding_mode", ["zeros", "circular"])
def test_cnn_preserves_shape(synthetic_fields: torch.Tensor, padding_mode: str) -> None:
    cfg = CNNConfig(padding_mode=padding_mode)  # type: ignore[arg-type]
    model = ConvNet(cfg)
    out = model(synthetic_fields)
    assert out.shape == synthetic_fields.shape
    assert torch.isfinite(out).all()


def test_default_cnn_has_four_layers_and_32_channels() -> None:
    model = ConvNet(CNNConfig())
    convs = [m for m in model.net if isinstance(m, torch.nn.Conv2d)]
    assert len(convs) == 4
    assert [c.out_channels for c in convs] == [32, 32, 32, 1]
    expected = (1 * 32 * 9 + 32) + 2 * (32 * 32 * 9 + 32) + (32 * 1 * 9 + 1)
    assert model.n_parameters == expected
    assert model.is_trainable


def test_residual_cnn_starts_near_persistence(synthetic_fields: torch.Tensor) -> None:
    torch.manual_seed(0)
    residual = ConvNet(CNNConfig(residual=True))
    plain = ConvNet(CNNConfig(residual=False))
    err_residual = (residual(synthetic_fields) - synthetic_fields).norm()
    err_plain = (plain(synthetic_fields) - synthetic_fields).norm()
    assert err_residual < err_plain


def test_circular_padding_is_translation_equivariant(synthetic_fields: torch.Tensor) -> None:
    torch.manual_seed(0)
    model = ConvNet(CNNConfig(padding_mode="circular")).eval()
    shifted_in = torch.roll(synthetic_fields, shifts=(5, -3), dims=(2, 3))
    with torch.no_grad():
        out = model(synthetic_fields)
        out_shifted = model(shifted_in)
    torch.testing.assert_close(torch.roll(out, shifts=(5, -3), dims=(2, 3)), out_shifted)


def test_registry_builds_by_name() -> None:
    cfg = ModelConfig(name="cnn")
    assert isinstance(build_model(cfg), ConvNet)
    assert isinstance(build_model(cfg, name="persistence"), Persistence)
    assert set(MODEL_REGISTRY) == {"persistence", "cnn"}


def test_registry_rejects_unknown_name() -> None:
    with pytest.raises(KeyError, match="unknown model 'fno'"):
        build_model(ModelConfig(), name="fno")
