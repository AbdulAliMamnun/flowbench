# FlowBench

Predicts a 2D fluid's future vorticity field and shows, side by side, the input
field, the numerical reference, the neural prediction and the error map. The
pipeline (`inspect → prepare → train → evaluate → serve`) is reproducible from one
YAML file and runs on a laptop.

> Status: Phase 0 scaffold. The benchmark table, screenshot and quickstart are
> generated in later phases and copied from `artifacts/`; nothing below is a
> measured number yet.

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
