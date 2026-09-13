import re

import pytest
from typer.testing import CliRunner

from flowbench import __version__
from flowbench.cli import app

# Help output is rendered by rich; on CI runners it may carry ANSI colour codes and be
# wrapped to the terminal width. Disable colour, widen the terminal and strip any codes
# that slip through so assertions see plain text.
runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "200"})

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

SUBCOMMANDS = ["inspect", "prepare", "train", "evaluate", "serve", "ui"]


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


def test_help_lists_every_subcommand() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    output = _plain(result.output)
    for name in SUBCOMMANDS:
        assert name in output


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_each_subcommand_accepts_config(name: str) -> None:
    result = runner.invoke(app, [name, "--help"])
    assert result.exit_code == 0, result.output
    assert "--config" in _plain(result.output)


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in _plain(result.output)


def test_missing_config_is_a_usage_error() -> None:
    result = runner.invoke(app, ["inspect", "--config", "does/not/exist.yaml"])
    assert result.exit_code == 2  # click usage error, not a stack trace
    assert "does/not/exist.yaml" in _plain(result.output)
