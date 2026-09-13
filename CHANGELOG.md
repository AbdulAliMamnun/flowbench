# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Known issues

- The Zenodo archive stores the Navier–Stokes fields at 128×128 and 1024×1024 only.
  The 32×32 working resolution is obtained by stride-4 decimation in
  `flowbench.data.dataset.subsample`; no anti-aliasing filter is applied.
- `NavierStokesDataset(subsampling_rate=k)` in neuraloperator 2.0.0 strides only the
  first spatial axis for this archive (it derives the number of spatial dims from a
  channel-squeezed tensor). FlowBench does not use that argument.
- The archive documents no prediction horizon, time step, units or boundary
  conditions; see `docs/data.md`. Periodicity is measured, not documented.

### Roadmap (not in v0.1)

- Fourier Neural Operator through the registry slot in `models/registry.py`.
- Autoregressive rollout over several steps once a time horizon is documented.
- Higher working resolution (64, 128) with area-averaged downsampling.
- Separate input/output normalisation as a configurable alternative.

## [0.1.0] - 2026-09-13

### Added

- Project scaffold: `src` layout, typer CLI with `inspect | prepare | train | evaluate | serve | ui`,
  typed YAML configuration, structured JSON logging, ruff + mypy + pytest tooling, CI workflow.
- Data: Zenodo download, `inspect` manifest with every undocumented property marked
  unknown and an empirical periodicity check, simulation-level split with persisted IDs
  and SHA-256 hashes, train-only normalisation, stride-4 cache at 32×32.
- Models: persistence baseline and a 4-layer, 32-channel residual CNN with circular
  padding; registry with a documented FNO slot.
- Training: AdamW + MSE, early stopping, validation-selected checkpoint directory with
  config, normalisation, split IDs, hashes, metrics and a manifest (git SHA, device,
  versions, determinism record).
- Evaluation: relative L2 (epsilon-floored), MAE, enstrophy error, high-vorticity slice,
  batch-one p50/p95 latency; `metrics.json`, Markdown table, error-map PNGs; generated
  blocks in `docs/benchmark.md` and `README.md`.
- Serving: FastAPI `POST /predict` (nested list or base64), `GET /health`,
  `GET /version`, structured error bodies.
- UI: Streamlit viewer with input / reference / prediction / error panels and per-sample
  metrics for a chosen held-out sample.
