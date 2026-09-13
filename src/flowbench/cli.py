"""Command-line entry point: ``flowbench <step> --config <yaml>``.

Every pipeline step is a subcommand that takes the same ``--config`` option, so the
full workflow is ``inspect → prepare → train → evaluate → serve | ui`` with one YAML file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from flowbench import __version__
from flowbench.config import FlowBenchConfig, load_config
from flowbench.logging import configure_logging, get_logger

app = typer.Typer(
    name="flowbench",
    help="Predict a 2D fluid's future vorticity field: inspect, prepare, train, evaluate, serve.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

log = get_logger(__name__)

ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
        help="YAML configuration file.",
        show_default=True,
    ),
]

DEFAULT_CONFIG = Path("configs/default.yaml")


def _load(config_path: Path) -> FlowBenchConfig:
    cfg = load_config(config_path)
    configure_logging(cfg.run.log_level)
    log.info("config loaded", path=str(config_path), run=cfg.run.name)
    return cfg


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"flowbench {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Print version."),
    ] = False,
) -> None:
    """FlowBench command-line interface."""


@app.command()
def inspect(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Download (if needed) and inspect the dataset; write data/manifest.json."""
    from flowbench.data.inspect import run_inspect

    cfg = _load(config)
    manifest_path = run_inspect(cfg)
    typer.echo(f"manifest written to {manifest_path}")


@app.command()
def prepare(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Split by simulation ID and fit train-only normalisation statistics."""
    from flowbench.data.split import run_prepare

    cfg = _load(config)
    split_path = run_prepare(cfg)
    typer.echo(f"split written to {split_path}")


@app.command()
def train(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Train the configured model and write a checkpoint directory."""
    from flowbench.training.trainer import run_train

    cfg = _load(config)
    checkpoint_dir = run_train(cfg)
    typer.echo(f"checkpoint written to {checkpoint_dir}")


@app.command()
def evaluate(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Evaluate persistence and the trained model on the held-out test set."""
    from flowbench.evaluation.report import run_evaluate

    cfg = _load(config)
    report_dir = run_evaluate(cfg)
    typer.echo(f"report written to {report_dir}")


@app.command()
def serve(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Start the FastAPI prediction server."""
    from flowbench.serving.app import run_serve

    cfg = _load(config)
    run_serve(cfg)


@app.command()
def ui(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Start the Streamlit viewer."""
    from flowbench.ui.launch import run_ui

    cfg = _load(config)
    run_ui(cfg, config_path=config)


if __name__ == "__main__":  # pragma: no cover
    app()
