"""Request and response models with shape and finiteness validation.

A field may be sent as a nested list of numbers or as base64-encoded little-endian
float32 bytes in row-major order. Exactly one of the two must be present. Shape and
finiteness are checked against the served model's resolution before inference.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from flowbench.serving.errors import FieldValidationError


class PredictRequest(BaseModel):
    """One vorticity field to predict from."""

    model_config = ConfigDict(extra="forbid")

    field: list[list[float]] | None = Field(
        default=None, description="Nested list, shape (resolution, resolution)."
    )
    field_b64: str | None = Field(
        default=None,
        description="Base64 of little-endian float32 bytes, row-major, resolution² values.",
    )

    @model_validator(mode="after")
    def _exactly_one_encoding(self) -> PredictRequest:
        if (self.field is None) == (self.field_b64 is None):
            msg = "provide exactly one of 'field' or 'field_b64'"
            raise ValueError(msg)
        return self

    def to_array(self, resolution: int) -> np.ndarray:
        """Decode, then validate shape and finiteness.

        Args:
            resolution: Expected side length of the square grid.

        Returns:
            ``(resolution, resolution)`` float32 array.

        Raises:
            FieldValidationError: On wrong shape, non-finite values or undecodable base64.
        """
        expected = (resolution, resolution)
        if self.field is not None:
            rows = self.field
            widths = {len(r) for r in rows}
            if len(rows) != resolution or widths != {resolution}:
                shape = f"({len(rows)}, {sorted(widths)})" if widths else f"({len(rows)}, 0)"
                msg = f"field must have shape {expected}, got {shape}"
                raise FieldValidationError(msg, [{"expected": list(expected)}])
            arr = np.asarray(rows, dtype=np.float32)
        else:
            encoded = self.field_b64 or ""  # the validator guarantees one encoding is set
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                msg = "field_b64 is not valid base64"
                raise FieldValidationError(msg) from exc
            n_expected = resolution * resolution * 4
            if len(raw) != n_expected:
                msg = (
                    f"field_b64 must decode to {n_expected} bytes "
                    f"(float32 x {expected}), got {len(raw)}"
                )
                raise FieldValidationError(msg, [{"expected_bytes": n_expected}])
            arr = np.frombuffer(raw, dtype="<f4").reshape(expected).astype(np.float32)
        if not np.isfinite(arr).all():
            n_bad = int((~np.isfinite(arr)).sum())
            msg = f"field contains {n_bad} non-finite value(s) (NaN or Inf)"
            raise FieldValidationError(msg, [{"non_finite_count": n_bad}])
        return arr


class PredictResponse(BaseModel):
    """Prediction in the field's original units plus provenance."""

    prediction: list[list[float]]
    shape: list[int]
    model_version: str
    device: str
    latency_ms: float = Field(description="Preprocess + inference + postprocess, this call.")


class HealthResponse(BaseModel):
    """Liveness and readiness."""

    status: str
    model_loaded: bool
    device: str


class VersionResponse(BaseModel):
    """Package, model and runtime versions."""

    package_version: str
    model_version: str
    git_sha: str
    torch_version: str
    checkpoint: str
    n_parameters: int
    resolution: int

    @classmethod
    def from_manifest(
        cls,
        package_version: str,
        model_version: str,
        manifest: dict[str, Any],
        checkpoint: str,
        resolution: int,
    ) -> VersionResponse:
        """Build from a checkpoint manifest."""
        return cls(
            package_version=package_version,
            model_version=model_version,
            git_sha=str(manifest.get("git_sha", "unknown")),
            torch_version=str(manifest.get("torch_version", "unknown")),
            n_parameters=int(manifest.get("n_parameters", 0)),
            checkpoint=checkpoint,
            resolution=resolution,
        )
