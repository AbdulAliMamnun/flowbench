import pytest
import torch

from flowbench.training.seed import configure_determinism, resolve_device, seed_everything


def test_seed_everything_makes_torch_reproducible() -> None:
    seed_everything(7)
    a = torch.randn(4)
    seed_everything(7)
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_resolve_cpu_is_always_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_resolve_auto_and_mps_never_fail() -> None:
    for name in ("auto", "mps"):
        dev = resolve_device(name)  # type: ignore[arg-type]
        assert dev.type in {"cpu", "mps"}


def test_resolve_falls_back_to_cpu_without_accelerator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hosted tiers are CPU-only: 'auto' and 'mps' must both resolve to cpu there."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
    assert resolve_device("mps").type == "cpu"


def test_configure_determinism_records_settings() -> None:
    record = configure_determinism(True, torch.device("cpu"))
    assert record["use_deterministic_algorithms"] is True
    assert record["caveats"] == []
    record = configure_determinism(True, torch.device("mps"))
    assert record["caveats"]
    record = configure_determinism(False, torch.device("cpu"))
    assert record["use_deterministic_algorithms"] is False
