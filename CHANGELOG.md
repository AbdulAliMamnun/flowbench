# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Known issues

- The Zenodo archive stores the Navier–Stokes fields at 128×128 and 1024×1024 only.
  The 32×32 working resolution is obtained by stride-4 subsampling through the
  `neuraloperator` loader; no anti-aliasing filter is applied.

## [0.1.0] - 2026-09-13

### Added

- Project scaffold: `src` layout, typer CLI with `inspect | prepare | train | evaluate | serve | ui`,
  typed YAML configuration, structured JSON logging, ruff + mypy + pytest tooling, CI workflow.
