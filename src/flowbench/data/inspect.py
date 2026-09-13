"""Record what the dataset actually contains before any training happens.

``run_inspect`` reads the raw archive files, measures what can be measured (shapes,
dtypes, ranges, instance counts, trajectory structure, boundary behaviour) and records
what cannot be determined as ``"unknown"``. The result is ``data/manifest.json`` and a
generated section of ``docs/data.md``. Nothing here is inferred from prior knowledge of
similar datasets.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from flowbench import __version__
from flowbench.data.download import archive_name, download_dataset, tensor_file
from flowbench.data.split import detect_simulations, sha256_of_file
from flowbench.logging import get_logger

if TYPE_CHECKING:
    from flowbench.config import DataConfig, FlowBenchConfig

log = get_logger(__name__)

UNKNOWN = "unknown"
DOCS_DATA_PATH = Path("docs/data.md")
GENERATED_BEGIN = "<!-- BEGIN GENERATED: flowbench inspect -->"
GENERATED_END = "<!-- END GENERATED -->"


def tensor_summary(t: torch.Tensor) -> dict[str, Any]:
    """Shape, dtype, range and finiteness of one tensor."""
    f = t.to(torch.float64)
    finite = torch.isfinite(f)
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).removeprefix("torch."),
        "min": float(f[finite].min()) if finite.any() else None,
        "max": float(f[finite].max()) if finite.any() else None,
        "mean": float(f[finite].mean()) if finite.any() else None,
        "std": float(f[finite].std(unbiased=False)) if finite.any() else None,
        "nan_count": int(torch.isnan(f).sum()),
        "inf_count": int(torch.isinf(f).sum()),
    }


def boundary_check(fields: torch.Tensor, ratio_max: float) -> dict[str, Any]:
    """Compare jumps across the wrap-around edge with jumps between interior neighbours.

    For a periodic field the value at the last grid line continues smoothly into the
    first, so the wrap-around jump has the same magnitude as an interior jump.

    Args:
        fields: ``(n, H, W)`` fields.
        ratio_max: Ratio below which a field counts as consistent with periodic boundaries.

    Returns:
        Per-axis mean jumps, their ratio and the verdict.
    """
    f = fields.to(torch.float64)
    interior0 = (f[:, 1:, :] - f[:, :-1, :]).abs().mean()
    wrap0 = (f[:, 0, :] - f[:, -1, :]).abs().mean()
    interior1 = (f[:, :, 1:] - f[:, :, :-1]).abs().mean()
    wrap1 = (f[:, :, 0] - f[:, :, -1]).abs().mean()
    ratio0 = float(wrap0 / interior0) if float(interior0) > 0 else float("inf")
    ratio1 = float(wrap1 / interior1) if float(interior1) > 0 else float("inf")
    consistent = ratio0 <= ratio_max and ratio1 <= ratio_max
    return {
        "method": (
            "mean |f[0,:]-f[-1,:]| vs mean |f[i+1,:]-f[i,:]| (and likewise along the second "
            "axis); ratio near 1 means the wrap-around edge is as smooth as the interior"
        ),
        "n_fields": int(f.shape[0]),
        "axis0_interior_jump": float(interior0),
        "axis0_wrap_jump": float(wrap0),
        "axis0_ratio": ratio0,
        "axis1_interior_jump": float(interior1),
        "axis1_wrap_jump": float(wrap1),
        "axis1_ratio": ratio1,
        "ratio_max": ratio_max,
        "verdict": (
            "consistent with periodic boundaries"
            if consistent
            else "not consistent with periodic boundaries"
        ),
    }


def trajectory_structure(x: torch.Tensor, y: torch.Tensor, atol: float) -> dict[str, Any]:
    """Detect whether consecutive instances chain into trajectories."""
    sim_ids = detect_simulations(x, y, atol=atol)
    n = int(x.shape[0])
    n_sims = int(torch.unique(sim_ids).numel())
    lengths = torch.bincount(sim_ids)
    identical_pairs = int(((y - x).abs().flatten(1).amax(dim=1) <= atol).sum())
    successor_gap = (y[:-1] - x[1:]).abs().flatten(1).amax(dim=1) if n > 1 else None
    x_std = float(x.to(torch.float64).std())
    y_std = float(y.to(torch.float64).std())
    return {
        "method": f"link instance i to i+1 when max|y[i]-x[i+1]| <= {atol}",
        "n_instances": n,
        "n_links": n - n_sims,
        "n_simulations": n_sims,
        "timesteps_per_simulation": (
            {"min": int(lengths.min()), "max": int(lengths.max())} if n_sims < n else UNKNOWN
        ),
        "instances_with_y_equal_x": identical_pairs,
        "min_successor_gap": float(successor_gap.min()) if successor_gap is not None else None,
        "target_to_input_std_ratio": y_std / x_std if x_std > 0 else None,
    }


def persistence_error(x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Relative L2 of predicting ``y`` by ``x``; a proxy for how far apart the two frames are."""
    diff = (y - x).to(torch.float64).flatten(1).norm(dim=1)
    ref = y.to(torch.float64).flatten(1).norm(dim=1).clamp_min(1e-12)
    rel = diff / ref
    return {"mean": float(rel.mean()), "min": float(rel.min()), "max": float(rel.max())}


def fetch_zenodo_metadata(record_id: str) -> dict[str, Any]:
    """Read title, description, licence, creators and checksums from the Zenodo API.

    Network failures are recorded rather than raised so ``inspect`` works offline.
    """
    import requests

    url = f"https://zenodo.org/api/records/{record_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        return {"url": url, "status": f"unavailable: {exc.__class__.__name__}"}
    meta = payload.get("metadata", {})
    return {
        "url": url,
        "status": "fetched",
        "title": meta.get("title"),
        "description": meta.get("description"),
        "license": (meta.get("license") or {}).get("id"),
        "creators": [c.get("name") for c in meta.get("creators", [])],
        "publication_date": meta.get("publication_date"),
        "files": {
            f["key"]: {"size_bytes": f.get("size"), "checksum": f.get("checksum")}
            for f in payload.get("files", [])
        },
    }


def loader_documentation() -> str:
    """Docstring of the ``neuraloperator`` loader; everything it says about the data."""
    from neuralop.data.datasets.navier_stokes import NavierStokesDataset

    return (NavierStokesDataset.__doc__ or "").strip()


def inspect_part(cfg: DataConfig, part: str) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Load one raw file and summarise every key it contains."""
    path = tensor_file(cfg.root_dir, part, cfg.source_resolution)
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        msg = f"{path} does not hold a dict, got {type(payload).__name__}"
        raise TypeError(msg)
    tensors = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor)}
    other = {k: repr(v) for k, v in payload.items() if not isinstance(v, torch.Tensor)}
    summary = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_of_file(path),
        "keys": sorted(payload.keys()),
        "tensors": {k: tensor_summary(v) for k, v in tensors.items()},
        "non_tensor_entries": other,
    }
    return summary, tensors


def build_manifest(cfg: DataConfig) -> dict[str, Any]:
    """Inspect both raw files and assemble the manifest dictionary."""
    files: dict[str, Any] = {}
    structure: dict[str, Any] = {}
    persistence: dict[str, Any] = {}
    boundary: dict[str, Any] = {}
    for part in ("train", "test"):
        summary, tensors = inspect_part(cfg, part)
        files[part] = summary
        if "x" in tensors and "y" in tensors:
            x, y = tensors["x"], tensors["y"]
            structure[part] = trajectory_structure(x, y, atol=cfg.trajectory_link_atol)
            persistence[part] = persistence_error(x, y)
            sample = torch.cat([x[: cfg.inspect_sample_size], y[: cfg.inspect_sample_size]])
            boundary[part] = boundary_check(sample, ratio_max=cfg.periodicity_ratio_max)
        del tensors

    periodic = all(
        b["verdict"] == "consistent with periodic boundaries" for b in boundary.values()
    )
    any_links = any(s["n_links"] > 0 for s in structure.values())

    unknowns = {
        "prediction_horizon": (
            f"{UNKNOWN}: the archive and loader state that x and y are input/output pairs "
            "of the time evolution but give no time offset; the persistence error above "
            "is the only measured indication of how far apart they are"
        ),
        "time_step_and_solver_dt": UNKNOWN,
        "physical_units": f"{UNKNOWN}: vorticity values are recorded as stored, unitless",
        "domain_size": UNKNOWN,
        "viscosity_and_forcing": (
            f"{UNKNOWN}: Zenodo states Reynolds number 500; viscosity and forcing term "
            "are not documented"
        ),
        "boundary_conditions": (
            "not documented by the archive or loader; empirical wrap-around check "
            + ("is consistent with periodic boundaries" if periodic else "is not periodic")
        ),
        "solver": f"{UNKNOWN}: Zenodo says 'a classical solver' at 1024x1024, unspecified",
        "downsampling_1024_to_128": f"{UNKNOWN}: how the archive's 128 grid was produced",
        "trajectory_grouping": (
            "detected from consecutive-frame links (see structure)"
            if any_links
            else f"{UNKNOWN}: no consecutive-frame links, instances treated as independent"
        ),
        "train_test_relationship": (
            f"{UNKNOWN}: whether the archive's test file comes from separate simulations"
        ),
    }

    return {
        "generated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "flowbench_version": __version__,
        "neuraloperator_version": version("neuraloperator"),
        "torch_version": torch.__version__,
        "source": {
            "zenodo_record_id": cfg.zenodo_record_id,
            "archive": archive_name(cfg.source_resolution),
            "loader": "neuralop.data.datasets.navier_stokes.NavierStokesDataset",
            "loader_docstring": loader_documentation(),
            "zenodo": fetch_zenodo_metadata(cfg.zenodo_record_id),
        },
        "grid": {
            "source_resolution": cfg.source_resolution,
            "working_resolution": cfg.resolution,
            "subsampling": (
                f"stride {cfg.subsampling_rate} on both spatial axes "
                f"(fields[:, :, ::{cfg.subsampling_rate}, ::{cfg.subsampling_rate}]) applied by "
                "flowbench.data.dataset.subsample after loading through NavierStokesDataset; "
                "the loader's own subsampling_rate strides only the first spatial axis for "
                "this channel-squeezed archive (neuraloperator 2.0.0), so it is not used; "
                "no anti-aliasing"
            ),
            "available_resolutions_on_zenodo": [128, 1024],
        },
        "files": files,
        "instances": {
            part: files[part]["tensors"].get("x", {}).get("shape", [None])[0]
            for part in files
        },
        "structure": structure,
        "persistence_relative_l2": persistence,
        "boundary_check": boundary,
        "unknowns": unknowns,
        "decisions": {
            "cnn_padding_mode": "circular" if periodic else "zeros",
            "cnn_padding_reason": (
                "wrap-around jumps match interior jumps on both axes in both files"
                if periodic
                else "wrap-around jumps exceed interior jumps; periodic padding not justified"
            ),
        },
    }


def render_manifest_markdown(manifest: dict[str, Any]) -> str:
    """Render the measured findings and unknowns as Markdown for ``docs/data.md``."""
    lines: list[str] = [GENERATED_BEGIN, ""]
    lines.append(
        f"_Generated {manifest['generated_at']} by `flowbench inspect` "
        f"(flowbench {manifest['flowbench_version']}, neuraloperator "
        f"{manifest['neuraloperator_version']}, torch {manifest['torch_version']})._"
    )
    lines += ["", "### Files", "", "| File | Instances | Keys | Tensor shapes | dtype | SHA-256 |"]
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for part, info in manifest["files"].items():
        shapes = ", ".join(f"`{k}`: {v['shape']}" for k, v in info["tensors"].items())
        dtypes = ", ".join(sorted({v["dtype"] for v in info["tensors"].values()}))
        lines.append(
            f"| `{Path(info['path']).name}` | {manifest['instances'][part]} | "
            f"{', '.join(f'`{k}`' for k in info['keys'])} | {shapes} | {dtypes} | "
            f"`{info['sha256'][:12]}…` |"
        )
    lines += ["", "### Value ranges", "", "| File | Tensor | min | max | mean | std | NaN | Inf |"]
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for part, info in manifest["files"].items():
        for key, t in info["tensors"].items():
            lines.append(
                f"| {part} | `{key}` | {t['min']:.4g} | {t['max']:.4g} | {t['mean']:.4g} | "
                f"{t['std']:.4g} | {t['nan_count']} | {t['inf_count']} |"
            )
    lines += ["", "### Grid", ""]
    grid = manifest["grid"]
    lines.append(
        f"- Source resolution {grid['source_resolution']}, working resolution "
        f"{grid['working_resolution']}: {grid['subsampling']}."
    )
    lines.append(f"- Resolutions on Zenodo: {grid['available_resolutions_on_zenodo']}.")
    lines += ["", "### Trajectory structure", ""]
    for part, s in manifest["structure"].items():
        lines.append(
            f"- **{part}**: {s['n_instances']} instances, {s['n_links']} consecutive-frame "
            f"links, {s['n_simulations']} simulations, timesteps per simulation: "
            f"{s['timesteps_per_simulation']}, instances with y == x: "
            f"{s['instances_with_y_equal_x']}, smallest max|y[i]-x[i+1]|: "
            f"{s['min_successor_gap']:.4g}, std(y)/std(x): "
            f"{s['target_to_input_std_ratio']:.3g}. Method: {s['method']}."
        )
    lines += ["", "### Persistence relative L2 (how far y is from x)", ""]
    for part, p in manifest["persistence_relative_l2"].items():
        lines.append(f"- **{part}**: mean {p['mean']:.4f}, min {p['min']:.4f}, max {p['max']:.4f}")
    lines += ["", "### Boundary check", ""]
    for part, b in manifest["boundary_check"].items():
        lines.append(
            f"- **{part}** ({b['n_fields']} fields): axis-0 wrap/interior ratio "
            f"{b['axis0_ratio']:.3f}, axis-1 ratio {b['axis1_ratio']:.3f} "
            f"(threshold {b['ratio_max']}) → {b['verdict']}."
        )
    lines.append(f"- Method: {next(iter(manifest['boundary_check'].values()))['method']}.")
    lines.append(
        f"- Decision: CNN `padding_mode: {manifest['decisions']['cnn_padding_mode']}` "
        f"({manifest['decisions']['cnn_padding_reason']})."
    )
    lines += ["", "### Unknowns", ""]
    for key, value in manifest["unknowns"].items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", GENERATED_END]
    return "\n".join(lines) + "\n"


def update_generated_section(doc_path: Path, generated: str) -> None:
    """Replace the generated block of a Markdown file, or append one if absent."""
    text = doc_path.read_text(encoding="utf-8") if doc_path.is_file() else ""
    start = text.find(GENERATED_BEGIN)
    end = text.find(GENERATED_END)
    if start != -1 and end != -1:
        text = text[:start] + generated.rstrip("\n") + text[end + len(GENERATED_END) :]
    else:
        text = text.rstrip("\n") + "\n\n" + generated
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(text, encoding="utf-8")


def run_inspect(cfg: FlowBenchConfig, docs_path: Path = DOCS_DATA_PATH) -> Path:
    """Inspect the raw dataset and write ``data/manifest.json`` plus the docs section.

    Args:
        cfg: Full pipeline configuration.
        docs_path: Markdown file whose generated block is refreshed.

    Returns:
        Path to the written manifest.
    """
    download_dataset(cfg.data)
    manifest = build_manifest(cfg.data)
    manifest_path = cfg.data.manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    update_generated_section(docs_path, render_manifest_markdown(manifest))
    log.info(
        "manifest written",
        path=str(manifest_path),
        instances=manifest["instances"],
        padding=manifest["decisions"]["cnn_padding_mode"],
    )
    return manifest_path
