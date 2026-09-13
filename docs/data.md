# Data

Written from `data/manifest.json` in Phase 1. Until then only provenance is recorded.

## Provenance

- Source: Zenodo record [12825163](https://zenodo.org/records/12825163), "Navier-Stokes
  Dataset", NeuralOperator Team, licence CC-BY-4.0.
- Loader: `neuralop.data.datasets.navier_stokes.NavierStokesDataset` (neuraloperator 2.0.0).
- Record description: input–output pairs of PyTorch tensors representing the time
  evolution of a fluid under the 2D incompressible Navier–Stokes equations at Reynolds
  number 500, originally generated at 1024×1024 by a classical solver.
- Archives available: `nsforcing_128.tgz` (1.45 GB) and `nsforcing_1024.tgz` (15.4 GB).
  FlowBench uses the 128 archive and subsamples by stride 4 to 32×32.

## Inspection findings

Pending Phase 1.

## Unknowns

Pending Phase 1. Everything the dataset does not document will be listed here and
marked `unknown` in the manifest.
