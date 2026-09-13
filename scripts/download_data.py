"""Download the Navier–Stokes archive once into the git-ignored data directory.

Thin wrapper around ``flowbench.data.download`` so the download can be run without the
rest of the pipeline::

    uv run python scripts/download_data.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flowbench.config import load_config
from flowbench.data.download import download_dataset
from flowbench.logging import configure_logging


def main() -> None:
    """Parse ``--config`` and download the configured archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_logging(cfg.run.log_level)
    target = download_dataset(cfg.data)
    print(f"dataset available under {target}")


if __name__ == "__main__":
    main()
