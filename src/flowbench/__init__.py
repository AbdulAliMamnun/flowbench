"""FlowBench: a reproducible pipeline that predicts a 2D fluid's future vorticity field."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flowbench")
except PackageNotFoundError:  # pragma: no cover - only when running from an unbuilt tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
