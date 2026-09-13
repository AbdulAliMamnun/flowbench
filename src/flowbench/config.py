"""Typed configuration loaded from YAML.

Every tunable in FlowBench lives here: seeds, paths, device, model hyperparameters,
thresholds. Library code never hard-codes these values; it receives a
:class:`FlowBenchConfig` (or one of its sections) and reads from it.

Environment variables prefixed with ``FLOWBENCH_`` override YAML values, using ``__``
as the nesting delimiter (``FLOWBENCH_RUN__DEVICE=cpu``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

DeviceName = Literal["auto", "mps", "cpu"]
PaddingMode = Literal["zeros", "circular"]
ModelName = Literal["persistence", "cnn"]


class _StrictModel(BaseModel):
    """Base for config sections: unknown keys are errors, values are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(_StrictModel):
    """Run-level settings shared by every pipeline step."""

    name: str = Field(description="Run name; artifacts are written to <artifacts_dir>/<name>.")
    seed: int = Field(default=0, ge=0)
    device: DeviceName = Field(
        default="auto",
        description="'auto' picks mps when available, otherwise cpu.",
    )
    artifacts_dir: Path = Path("artifacts")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    deterministic: bool = Field(
        default=True,
        description="Request deterministic torch algorithms where the backend supports them.",
    )
    update_docs: bool = Field(
        default=True,
        description=(
            "Refresh the generated blocks in docs/ and README.md (inspect and evaluate). "
            "Smoke runs set this to false so only the full run publishes numbers."
        ),
    )

    @property
    def run_dir(self) -> Path:
        """Directory holding every artifact of this run."""
        return self.artifacts_dir / self.name


class DataConfig(_StrictModel):
    """Dataset location, resolution handling and split parameters."""

    root_dir: Path = Path("data")
    manifest_path: Path = Path("data/manifest.json")
    zenodo_record_id: str = "12825163"
    source_resolution: int = Field(
        default=128,
        description="Resolution of the archive fetched from Zenodo (only 128 and 1024 exist).",
    )
    resolution: int = Field(default=32, description="Working resolution after subsampling.")
    n_train_available: int = Field(
        default=10_000, description="Instances in the archive's train file (verified by inspect)."
    )
    n_test_available: int = Field(
        default=2_000, description="Instances in the archive's test file (verified by inspect)."
    )
    max_train_samples: int | None = Field(
        default=None,
        description="Cap on train+val instances (smoke runs). None uses everything available.",
    )
    max_test_samples: int | None = Field(
        default=None, description="Cap on held-out test instances (smoke runs)."
    )
    val_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    split_seed: int = Field(default=0, ge=0)
    num_workers: int = Field(default=0, ge=0)
    trajectory_link_atol: float = Field(
        default=1e-6,
        ge=0.0,
        description=(
            "Absolute tolerance for detecting that output i equals input i+1, i.e. that "
            "consecutive instances are frames of one simulation."
        ),
    )
    periodicity_ratio_max: float = Field(
        default=1.5,
        gt=0.0,
        description=(
            "Inspection heuristic: a field is 'consistent with periodic boundaries' when the "
            "mean jump across the wrap-around edge is below this multiple of the mean "
            "interior jump."
        ),
    )
    inspect_sample_size: int = Field(
        default=512, ge=1, description="Instances used for the boundary-condition check."
    )

    @property
    def subsampling_rate(self) -> int:
        """Stride used to reduce ``source_resolution`` to ``resolution``."""
        return self.source_resolution // self.resolution

    @property
    def prepared_dir(self) -> Path:
        """Directory of the cached working-resolution tensors written by ``prepare``."""
        return self.root_dir / "prepared"

    @property
    def normalization_path(self) -> Path:
        """Location of the train-only normalisation statistics written by ``prepare``."""
        return self.root_dir / "splits" / f"normalization_seed{self.split_seed}.json"

    @model_validator(mode="after")
    def _resolution_divides(self) -> DataConfig:
        if self.source_resolution % self.resolution != 0:
            msg = (
                f"resolution {self.resolution} must divide "
                f"source_resolution {self.source_resolution}"
            )
            raise ValueError(msg)
        return self

    @property
    def split_path(self) -> Path:
        """Location of the persisted simulation-level split IDs."""
        return self.root_dir / "splits" / f"split_seed{self.split_seed}.json"


class CNNConfig(_StrictModel):
    """Hyperparameters of the convolutional baseline."""

    layers: int = Field(default=4, ge=1)
    channels: int = Field(default=32, ge=1)
    kernel_size: int = Field(default=3, ge=1)
    padding_mode: PaddingMode = Field(
        default="zeros",
        description=(
            "'circular' is only justified when data inspection confirms periodic boundaries; "
            "see docs/data.md for the recorded decision."
        ),
    )
    residual: bool = Field(
        default=True,
        description="Predict the increment over the input field rather than the raw field.",
    )

    @model_validator(mode="after")
    def _odd_kernel(self) -> CNNConfig:
        if self.kernel_size % 2 == 0:
            msg = "kernel_size must be odd so that 'same' padding is symmetric"
            raise ValueError(msg)
        return self


class ModelConfig(_StrictModel):
    """Which predictor to train and its hyperparameters."""

    name: ModelName = "cnn"
    cnn: CNNConfig = CNNConfig()


class TrainingConfig(_StrictModel):
    """Optimisation loop settings."""

    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    early_stopping_patience: int = Field(default=10, ge=1)
    log_every_n_steps: int = Field(default=50, ge=1)


class LatencyConfig(_StrictModel):
    """Batch-one latency measurement protocol."""

    warmup_iterations: int = Field(default=20, ge=1)
    timed_iterations: int = Field(default=200, ge=2)


class EvaluationConfig(_StrictModel):
    """Held-out evaluation settings."""

    batch_size: int = Field(default=128, ge=1)
    relative_l2_epsilon: float = Field(
        default=1e-8,
        gt=0.0,
        description="Floor applied to reference norms; affected samples are counted and reported.",
    )
    high_vorticity_quantile: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        description=(
            "Quantile of per-sample max |vorticity| over train+val that defines the "
            "high-vorticity slice. Never derived from test data."
        ),
    )
    n_error_maps: int = Field(default=4, ge=0)
    latency: LatencyConfig = LatencyConfig()
    metrics_tolerance: float = Field(
        default=1e-5,
        ge=0.0,
        description="Tolerance for the checkpoint reload reproducibility test.",
    )


class ServingConfig(_StrictModel):
    """FastAPI server settings."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    checkpoint_dir: Path | None = Field(
        default=None,
        description="Checkpoint directory to serve; defaults to <run_dir>/checkpoints/cnn.",
    )


class UIConfig(_StrictModel):
    """Streamlit viewer settings."""

    port: int = Field(default=8501, ge=1, le=65535)
    n_preview_samples: int = Field(default=200, ge=1)


class DemoConfig(_StrictModel):
    """Self-contained demo bundle for hosted, data-less deployments."""

    dir: Path = Path("demo")
    n_samples: int = Field(default=100, ge=1)
    seed: int = Field(default=0, ge=0, description="Seed of the held-out subset selection.")
    max_bytes: int = Field(
        default=2_000_000, ge=1, description="Export fails if the bundle exceeds this size."
    )


class FlowBenchConfig(BaseSettings):
    """Top-level configuration; the single object passed through the pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="FLOWBENCH_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    run: RunConfig
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    serving: ServingConfig = ServingConfig()
    ui: UIConfig = UIConfig()
    demo: DemoConfig = DemoConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order sources so environment variables override the YAML (init) values."""
        del settings_cls, dotenv_settings, file_secret_settings
        return (env_settings, init_settings)

    @property
    def checkpoint_dir(self) -> Path:
        """Directory of the checkpoint produced by ``train`` for the configured model."""
        return self.run.run_dir / "checkpoints" / self.model.name

    @property
    def serving_checkpoint_dir(self) -> Path:
        """Checkpoint directory served by the API and the UI."""
        return self.serving.checkpoint_dir or self.checkpoint_dir

    def to_yaml(self) -> str:
        """Serialise the resolved configuration to YAML (paths become strings)."""
        payload = self.model_dump(mode="json")
        return yaml.safe_dump(payload, sort_keys=False)


def load_config(path: Path | str) -> FlowBenchConfig:
    """Load and validate a YAML config file.

    Args:
        path: Path to a YAML file whose top-level keys are config sections.

    Returns:
        A frozen, validated :class:`FlowBenchConfig`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the YAML is not a mapping.
        pydantic.ValidationError: If any value is missing or invalid.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        msg = f"config {path} must be a YAML mapping, got {type(raw).__name__}"
        raise ValueError(msg)
    return FlowBenchConfig(**raw)


def load_config_from_string(text: str) -> FlowBenchConfig:
    """Parse a YAML string into a validated config (used by checkpoint loading)."""
    raw: Any = yaml.safe_load(text)
    if not isinstance(raw, dict):
        msg = f"config text must be a YAML mapping, got {type(raw).__name__}"
        raise ValueError(msg)
    return FlowBenchConfig(**raw)
