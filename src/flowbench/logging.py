"""Structured logging setup.

Library code obtains a logger via :func:`get_logger`; the CLI calls
:func:`configure_logging` once. Records are emitted as single-line JSON so they can be
grepped or shipped without parsing free text.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import MutableMapping

_ROOT_NAME = "flowbench"


class JsonFormatter(logging.Formatter):
    """Format records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as JSON with timestamp, level, logger name and message."""
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _ExtraAdapter(logging.LoggerAdapter[logging.Logger]):
    """Logger adapter that folds keyword fields into the ``extra`` JSON payload."""

    _RESERVED = frozenset({"exc_info", "stack_info", "stacklevel"})

    def process(
        self, msg: object, kwargs: MutableMapping[str, Any]
    ) -> tuple[object, MutableMapping[str, Any]]:
        """Move arbitrary keyword arguments into ``extra['extra']``."""
        fields = {k: v for k, v in kwargs.items() if k not in self._RESERVED}
        for key in fields:
            kwargs.pop(key)
        kwargs["extra"] = {"extra": fields}
        return msg, kwargs


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON handler on the ``flowbench`` logger.

    Args:
        level: Logging level name (``DEBUG``, ``INFO``, ...).
    """
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level)
    root.propagate = False
    if not any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)


def get_logger(name: str) -> _ExtraAdapter:
    """Return a structured logger namespaced under ``flowbench``.

    Args:
        name: Module name, typically ``__name__``.

    Returns:
        A logger adapter accepting extra keyword fields, e.g.
        ``log.info("epoch done", epoch=3, loss=0.1)``.
    """
    qualified = name if name.startswith(_ROOT_NAME) else f"{_ROOT_NAME}.{name}"
    return _ExtraAdapter(logging.getLogger(qualified), {})
