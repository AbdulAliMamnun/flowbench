from pathlib import Path

import torch

from flowbench.data.inspect import (
    GENERATED_BEGIN,
    GENERATED_END,
    boundary_check,
    persistence_error,
    tensor_summary,
    trajectory_structure,
    update_generated_section,
)


def test_tensor_summary_counts_non_finite() -> None:
    t = torch.tensor([[1.0, float("nan")], [float("inf"), -2.0]])
    s = tensor_summary(t)
    assert s["shape"] == [2, 2]
    assert s["dtype"] == "float32"
    assert s["nan_count"] == 1
    assert s["inf_count"] == 1
    assert s["min"] == -2.0
    assert s["max"] == 1.0


def test_boundary_check_flags_periodic_fields(synthetic_fields: torch.Tensor) -> None:
    result = boundary_check(synthetic_fields[:, 0], ratio_max=1.5)
    assert result["verdict"] == "consistent with periodic boundaries"
    assert result["axis0_ratio"] < 1.5
    assert result["axis1_ratio"] < 1.5


def test_boundary_check_flags_non_periodic_fields() -> None:
    ramp = torch.linspace(0.0, 10.0, 32).reshape(1, 32, 1).expand(8, 32, 32)
    result = boundary_check(ramp, ratio_max=1.5)
    assert result["verdict"] == "not consistent with periodic boundaries"
    assert result["axis0_ratio"] > 1.5


def test_trajectory_structure_reports_links() -> None:
    frames = torch.randn(2, 4, 1, 3, 3)
    x = frames[:, :-1].reshape(-1, 1, 3, 3)
    y = frames[:, 1:].reshape(-1, 1, 3, 3)
    s = trajectory_structure(x, y, atol=0.0)
    assert s["n_instances"] == 6
    assert s["n_simulations"] == 2
    assert s["n_links"] == 4
    assert s["timesteps_per_simulation"] == {"min": 3, "max": 3}
    assert s["instances_with_y_equal_x"] == 0


def test_trajectory_structure_without_links_is_unknown() -> None:
    s = trajectory_structure(torch.randn(5, 1, 2, 2), torch.randn(5, 1, 2, 2), atol=0.0)
    assert s["n_links"] == 0
    assert s["timesteps_per_simulation"] == "unknown"


def test_persistence_error_zero_for_identity(synthetic_fields: torch.Tensor) -> None:
    p = persistence_error(synthetic_fields, synthetic_fields)
    assert p["mean"] == 0.0
    assert p["max"] == 0.0


def test_update_generated_section_replaces_block(tmp_path: Path) -> None:
    doc = tmp_path / "data.md"
    doc.write_text(f"# Title\n\nintro\n\n{GENERATED_BEGIN}\nold\n{GENERATED_END}\n\nafter\n")
    update_generated_section(doc, f"{GENERATED_BEGIN}\nnew\n{GENERATED_END}\n")
    text = doc.read_text()
    assert "old" not in text
    assert "new" in text
    assert text.startswith("# Title")
    assert text.rstrip().endswith("after")


def test_update_generated_section_appends_when_absent(tmp_path: Path) -> None:
    doc = tmp_path / "data.md"
    doc.write_text("# Title\n")
    update_generated_section(doc, f"{GENERATED_BEGIN}\nnew\n{GENERATED_END}\n")
    assert doc.read_text().count(GENERATED_BEGIN) == 1


def test_update_generated_section_uses_custom_markers(tmp_path: Path) -> None:
    """Two generated blocks with different opening markers can share the same END marker."""
    begin_a, begin_b = "<!-- BEGIN A -->", "<!-- BEGIN B -->"
    doc = tmp_path / "doc.md"
    doc.write_text(
        f"# T\n\n{begin_a}\nold-a\n{GENERATED_END}\n\nmiddle\n\n{begin_b}\nold-b\n{GENERATED_END}\n"
    )
    update_generated_section(doc, f"{begin_b}\nnew-b\n{GENERATED_END}\n", begin_b, GENERATED_END)
    text = doc.read_text()
    assert "old-a" in text and "old-b" not in text and "new-b" in text
    assert text.count(GENERATED_END) == 2
    assert text.index("middle") < text.index("new-b")
