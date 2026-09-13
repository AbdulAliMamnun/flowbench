# Data

Everything measurable about the dataset is measured by `flowbench inspect` and written
to `data/manifest.json`; the block at the bottom of this page is generated from that
file. The prose above it interprets the findings and records decisions.

## Provenance

- Source: Zenodo record [12825163](https://zenodo.org/records/12825163), "Navier-Stokes
  Dataset", NeuralOperator Team, licence CC-BY-4.0.
- Loader: `neuralop.data.datasets.navier_stokes.NavierStokesDataset` (neuraloperator 2.0.0).
- Record description: input–output pairs of PyTorch tensors representing the time
  evolution of a fluid under the 2D incompressible Navier–Stokes equations at Reynolds
  number 500, originally generated at 1024×1024 by a classical solver.
- Archives available: `nsforcing_128.tgz` (1.45 GB) and `nsforcing_1024.tgz` (15.4 GB).
  FlowBench uses the 128 archive and subsamples by stride 4 to 32×32.

## What the inspection established

- **Two flat tensors per file, nothing else.** `x` and `y` of shape
  `(n, 128, 128)`, float32, no NaN/Inf, no metadata keys. 10 000 train pairs, 2 000 test
  pairs.
- **Instances are independent pairs, not frames of a trajectory.** No target equals
  the next input (smallest gap is far above tolerance), and the target has about 4.6×
  the standard deviation of the input. The archive therefore exposes no simulation
  grouping; each instance is treated as its own simulation and the split is by
  instance ID. This is recorded as `trajectory_grouping: unknown`.
- **The boundary is periodic in practice.** The mean jump across the wrap-around edge
  equals the mean jump between interior neighbours on both axes (ratio 0.99 to 1.00)
  in both files. The dataset does not document its boundary conditions, so the
  manifest records the empirical result, not a claim about the solver.
- **The prediction horizon is unknown.** Nothing in the archive or loader states the
  time offset between `x` and `y`. FlowBench predicts "the target field the dataset
  pairs with the input" and never quotes a time.

## Decisions taken from the findings

1. **`padding_mode: circular` for the CNN.** Justified by the periodicity check above;
   the flag lives in `configs/*.yaml` and the reason in the manifest's `decisions`.
2. **Split by instance ID.** With no detectable trajectories this is the finest
   grouping available. The splitter still groups by simulation ID, so if a future
   archive exposes trajectories nothing changes except the detected IDs.
3. **Working resolution 32×32 by stride-4 subsampling of the 128 grid**, done in
   `flowbench.data.dataset.subsample` after loading through `NavierStokesDataset`.
   The loader's own `subsampling_rate` argument strides only the first spatial axis for
   this channel-squeezed archive (neuraloperator 2.0.0), which produced 32×128 fields;
   FlowBench applies the same stride to both axes and asserts the result. No
   anti-aliasing filter is applied; this is a plain decimation.
4. **One shared normalisation (mean, std) for inputs and targets**, fitted on the
   9 000 training pairs only. Inputs and targets are the same physical quantity, and a
   shared affine map keeps the persistence baseline an exact identity in normalised
   units as in field units. The scale mismatch between `x` and `y` (std 0.15 vs 0.71)
   is left for the model to learn; all reported metrics are in field units.
5. **The archive's test file is the held-out test set** and is never read by `prepare`
   for statistics or thresholds. Validation (1 000 pairs) is carved from the train file.

## Unknowns

Listed in the generated block below under *Unknowns* and stored verbatim in
`data/manifest.json`. In short: prediction horizon, time step, physical units, domain
size, viscosity and forcing, boundary conditions as documented (only measured), the
solver, how the 1024 grid was reduced to 128, trajectory grouping, and whether the test
file comes from separate simulations.

<!-- BEGIN GENERATED: flowbench inspect -->

_Generated 2026-09-13T05:26:50+00:00 by `flowbench inspect` (flowbench 0.1.0, neuraloperator 2.0.0, torch 2.14.0)._

### Files

| File | Instances | Keys | Tensor shapes | dtype | SHA-256 |
| --- | --- | --- | --- | --- | --- |
| `nsforcing_train_128.pt` | 10000 | `x`, `y` | `x`: [10000, 128, 128], `y`: [10000, 128, 128] | float32 | `6c145749f40a…` |
| `nsforcing_test_128.pt` | 2000 | `x`, `y` | `x`: [2000, 128, 128], `y`: [2000, 128, 128] | float32 | `a7e1ed8ebc05…` |

### Value ranges

| File | Tensor | min | max | mean | std | NaN | Inf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | `x` | -0.7591 | 0.8172 | 1.08e-10 | 0.1539 | 0 | 0 |
| train | `y` | -3.489 | 3.928 | -3.473e-12 | 0.7079 | 0 | 0 |
| test | `x` | -0.7624 | 0.7503 | -4.58e-10 | 0.1536 | 0 | 0 |
| test | `y` | -3.631 | 3.627 | -1.056e-11 | 0.7062 | 0 | 0 |

### Grid

- Source resolution 128, working resolution 32: stride 4 on both spatial axes (fields[:, :, ::4, ::4]) applied by flowbench.data.dataset.subsample after loading through NavierStokesDataset; the loader's own subsampling_rate strides only the first spatial axis for this channel-squeezed archive (neuraloperator 2.0.0), so it is not used; no anti-aliasing.
- Resolutions on Zenodo: [128, 1024].

### Trajectory structure

- **train**: 10000 instances, 0 consecutive-frame links, 10000 simulations, timesteps per simulation: unknown, instances with y == x: 0, smallest max|y[i]-x[i+1]|: 1.091, std(y)/std(x): 4.6. Method: link instance i to i+1 when max|y[i]-x[i+1]| <= 1e-06.
- **test**: 2000 instances, 0 consecutive-frame links, 2000 simulations, timesteps per simulation: unknown, instances with y == x: 0, smallest max|y[i]-x[i+1]|: 1.218, std(y)/std(x): 4.6. Method: link instance i to i+1 when max|y[i]-x[i+1]| <= 1e-06.

### Persistence relative L2 (how far y is from x)

- **train**: mean 0.8357, min 0.8042, max 0.8803
- **test**: mean 0.8359, min 0.8064, max 0.8768

### Boundary check

- **train** (1024 fields): axis-0 wrap/interior ratio 0.992, axis-1 ratio 0.985 (threshold 1.5) → consistent with periodic boundaries.
- **test** (1024 fields): axis-0 wrap/interior ratio 1.001, axis-1 ratio 1.002 (threshold 1.5) → consistent with periodic boundaries.
- Method: mean |f[0,:]-f[-1,:]| vs mean |f[i+1,:]-f[i,:]| (and likewise along the second axis); ratio near 1 means the wrap-around edge is as smooth as the interior.
- Decision: CNN `padding_mode: circular` (wrap-around jumps match interior jumps on both axes in both files).

### Unknowns

- `prediction_horizon`: unknown: the archive and loader state that x and y are input/output pairs of the time evolution but give no time offset; the persistence error above is the only measured indication of how far apart they are
- `time_step_and_solver_dt`: unknown
- `physical_units`: unknown: vorticity values are recorded as stored, unitless
- `domain_size`: unknown
- `viscosity_and_forcing`: unknown: Zenodo states Reynolds number 500; viscosity and forcing term are not documented
- `boundary_conditions`: not documented by the archive or loader; empirical wrap-around check is consistent with periodic boundaries
- `solver`: unknown: Zenodo says 'a classical solver' at 1024x1024, unspecified
- `downsampling_1024_to_128`: unknown: how the archive's 128 grid was produced
- `trajectory_grouping`: unknown: no consecutive-frame links, instances treated as independent
- `train_test_relationship`: unknown: whether the archive's test file comes from separate simulations

<!-- END GENERATED -->
