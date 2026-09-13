# FlowBench

Predicts a 2D fluid's future vorticity field and shows, side by side, the input
field, the numerical reference, the neural prediction and the error map. The
pipeline (`inspect → prepare → train → evaluate → serve`) is reproducible from one
YAML file and runs on a laptop.

## Benchmark

<!-- BEGIN GENERATED: flowbench evaluate -->

_Generated 2026-09-13T05:34:12+00:00 by `flowbench evaluate` on run `default` (flowbench 0.1.0, git `77c1383c`, device `mps`)._

| Model | Params | Rel. L2 mean | Rel. L2 worst | MAE (field units) | Enstrophy rel. err. mean | Enstrophy rel. err. worst | p50 ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence | 0 | 0.8359 | 0.8768 | 0.4711 | 0.9527 | 0.9559 | 0.40 | 0.64 |
| cnn | 19,105 | 0.4667 | 0.6477 | 0.257 | 0.2272 | 0.5425 | 0.58 | 0.72 |

**High-vorticity slice** (reference max|ω| ≥ 2.406, the 90% quantile over 10000 train+validation reference fields; 204 of 2000 test samples):

| Model | Rel. L2 mean | Rel. L2 worst | MAE | Enstrophy rel. err. mean |
| --- | ---: | ---: | ---: | ---: |
| persistence | 0.8348 | 0.8596 | 0.5411 | 0.9529 |
| cnn | 0.4681 | 0.62 | 0.3027 | 0.1697 |

**Setup.** 2000 held-out test pairs at 32×32; relative L2 uses an epsilon floor of 1e-08 on the reference norm (floored samples: persistence 0, cnn 0). Enstrophy is 0.5·mean(ω²) over the grid because the domain size is unknown. Latency is the batch-one preprocess+inference+postprocess path after 20 warm-up calls, over 200 timed calls on `mps`; HTTP overhead is not included.

<!-- END GENERATED -->

## Quickstart

```bash
uv sync --frozen
uv run flowbench --help
```

## Development

```bash
make lint   # ruff + mypy --strict
make test   # pytest with synthetic fixtures
```

## License

MIT. See [LICENSE](LICENSE).
