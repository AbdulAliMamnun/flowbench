.DEFAULT_GOAL := help
CONFIG ?= configs/default.yaml
SMOKE_CONFIG ?= configs/smoke.yaml

.PHONY: help setup lint format typecheck test smoke inspect prepare train evaluate serve ui demo clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Install the locked environment and pre-commit hooks
	uv sync --frozen
	uv run pre-commit install

lint: ## Ruff lint + format check + mypy --strict on src/
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format: ## Auto-fix lint issues and reformat
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## mypy --strict on src/
	uv run mypy

test: ## Unit + integration tests (no network, no real data)
	uv run pytest

smoke: ## End-to-end pipeline on configs/smoke.yaml (~1 min, uses downloaded data)
	uv run flowbench inspect  --config $(SMOKE_CONFIG)
	uv run flowbench prepare  --config $(SMOKE_CONFIG)
	uv run flowbench train    --config $(SMOKE_CONFIG)
	uv run flowbench evaluate --config $(SMOKE_CONFIG)

inspect: ## Download (once) and inspect the dataset; writes data/manifest.json
	uv run flowbench inspect --config $(CONFIG)

prepare: ## Simulation-level split + train-only normalisation statistics
	uv run flowbench prepare --config $(CONFIG)

train: ## Train the configured model
	uv run flowbench train --config $(CONFIG)

evaluate: ## Evaluate persistence vs CNN on the held-out test set
	uv run flowbench evaluate --config $(CONFIG)

serve: ## Start the FastAPI server
	uv run flowbench serve --config $(CONFIG)

ui: ## Start the Streamlit viewer
	uv run flowbench ui --config $(CONFIG)

demo: ## Serve + UI for the demo recording (see scripts/record_demo.md)
	@echo "Open two terminals: 'make serve' and 'make ui'. Steps: scripts/record_demo.md"

clean: ## Remove caches (keeps data/ and artifacts/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
