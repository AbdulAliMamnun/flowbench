"""Launch the Streamlit viewer as a subprocess.

Kept separate from :mod:`flowbench.ui.app` because that file is a Streamlit *script*
executed by ``streamlit run``; importing it from the CLI would execute the page.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from flowbench.logging import get_logger

if TYPE_CHECKING:
    from flowbench.config import FlowBenchConfig

log = get_logger(__name__)

APP_PATH = Path(__file__).with_name("app.py")


def streamlit_command(cfg: FlowBenchConfig, config_path: Path) -> list[str]:
    """Build the ``streamlit run`` command line for the viewer."""
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(cfg.ui.port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--",
        "--config",
        str(config_path.resolve()),
    ]


def run_ui(cfg: FlowBenchConfig, config_path: Path) -> None:
    """Start ``streamlit run`` on the viewer script and block until it exits.

    Args:
        cfg: Full pipeline configuration.
        config_path: Path of the YAML file, forwarded to the Streamlit script.
    """
    cmd = streamlit_command(cfg, config_path)
    log.info("starting streamlit", port=cfg.ui.port, config=str(config_path))
    subprocess.run(cmd, check=False)
