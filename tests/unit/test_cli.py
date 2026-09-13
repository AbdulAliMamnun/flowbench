import pytest
from typer.testing import CliRunner

from flowbench import __version__
from flowbench.cli import app

runner = CliRunner()

SUBCOMMANDS = ["inspect", "prepare", "train", "evaluate", "serve", "ui"]


def test_help_lists_every_subcommand() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for name in SUBCOMMANDS:
        assert name in result.output


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_each_subcommand_accepts_config(name: str) -> None:
    result = runner.invoke(app, [name, "--help"])
    assert result.exit_code == 0, result.output
    assert "--config" in result.output


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_missing_config_is_a_usage_error() -> None:
    result = runner.invoke(app, ["inspect", "--config", "does/not/exist.yaml"])
    assert result.exit_code == 2  # click usage error, not a stack trace
    assert "does/not/exist.yaml" in result.output
