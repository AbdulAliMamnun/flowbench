"""Model registry: config name → constructor.

Extension point
---------------
A Fourier Neural Operator (out of scope for v0.1) would be added by registering one
more entry with the same ``(ModelConfig) -> Predictor`` signature, e.g.
``"fno": lambda cfg: FNO(cfg.fno)`` together with an ``fno`` section in
:class:`~flowbench.config.ModelConfig`. Training, evaluation, checkpointing and serving
all go through :func:`build_model` and would need no change.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from flowbench.models.cnn import ConvNet
from flowbench.models.persistence import Persistence

if TYPE_CHECKING:
    from flowbench.config import ModelConfig
    from flowbench.models.base import Predictor

ModelBuilder = Callable[["ModelConfig"], "Predictor"]

MODEL_REGISTRY: dict[str, ModelBuilder] = {
    "persistence": lambda _cfg: Persistence(),
    "cnn": lambda cfg: ConvNet(cfg.cnn),
}


def build_model(cfg: ModelConfig, name: str | None = None) -> Predictor:
    """Instantiate the predictor named by ``name`` (default: ``cfg.name``).

    Args:
        cfg: Model configuration section.
        name: Registry key overriding ``cfg.name``; lets evaluation build the
            persistence baseline from a CNN config.

    Returns:
        A freshly constructed, untrained predictor on CPU.

    Raises:
        KeyError: If the name is not registered.
    """
    key = name or cfg.name
    try:
        builder = MODEL_REGISTRY[key]
    except KeyError as exc:
        msg = f"unknown model {key!r}; registered: {sorted(MODEL_REGISTRY)}"
        raise KeyError(msg) from exc
    return builder(cfg)
