import json
import logging
import sys

import pytest

from flowbench.logging import JsonFormatter, configure_logging, get_logger


def test_records_are_single_line_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("DEBUG")
    log = get_logger("tests.logging")
    log.info("hello", epoch=3, loss=0.25)
    captured = capsys.readouterr().err.strip().splitlines()
    record = json.loads(captured[-1])
    assert record["msg"] == "hello"
    assert record["epoch"] == 3
    assert record["loss"] == 0.25
    assert record["logger"] == "flowbench.tests.logging"
    assert record["level"] == "INFO"


def test_formatter_includes_exception() -> None:
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            "flowbench.x", logging.ERROR, __file__, 1, "failed", None, exc_info=sys.exc_info()
        )
    payload = json.loads(formatter.format(record))
    assert "ValueError: boom" in payload["exc"]


def test_configure_logging_is_idempotent() -> None:
    configure_logging()
    configure_logging()
    root = logging.getLogger("flowbench")
    assert sum(isinstance(h.formatter, JsonFormatter) for h in root.handlers) == 1
