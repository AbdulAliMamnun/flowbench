"""Launch the Streamlit viewer as a subprocess.

Kept separate from :mod:`flowbench.ui.app` because that file is a Streamlit *script*
executed by ``streamlit run``; importing it from the CLI would execute the page.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from flowbench.config import FlowBenchConfig


def run_ui(cfg: FlowBenchConfig, config_path: Path) -> None:
    """Start ``streamlit run`` on the viewer script.

    Args:
        cfg: Full pipeline configuration.
        config_path: Path of the YAML file, forwarded to the Streamlit script.
    """
    msg = "run_ui is implemented in Phase 4 (serving + UI)"
    raise NotImplementedError(msg)
