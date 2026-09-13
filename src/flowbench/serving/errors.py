"""Structured error bodies for 4xx/5xx responses; never a stack trace.

Every error the API returns has the shape::

    {"error": {"type": "<machine-readable>", "message": "<human-readable>", "details": [...]}}
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from flowbench.logging import get_logger

log = get_logger(__name__)


class FieldValidationError(ValueError):
    """Raised when a request parses but the field is not a valid input for the model."""

    def __init__(self, message: str, details: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or []


def error_body(
    error_type: str, message: str, details: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Build the JSON body used by every error response."""
    return {"error": {"type": error_type, "message": message, "details": details or []}}


def install_error_handlers(app: FastAPI) -> None:
    """Register handlers that turn every failure into a structured JSON body."""

    @app.exception_handler(RequestValidationError)
    async def _on_request_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"loc": [str(part) for part in e.get("loc", [])], "msg": e.get("msg", "")}
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=error_body("validation_error", "request body is invalid", details),
        )

    @app.exception_handler(FieldValidationError)
    async def _on_field_validation(_: Request, exc: FieldValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=error_body("invalid_field", exc.message, exc.details),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _on_http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body("http_error", str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def _on_unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled error", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_body("internal_error", "prediction failed; see server logs"),
        )
