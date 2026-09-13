from pathlib import Path

from flowbench.config import FlowBenchConfig
from flowbench.ui.launch import APP_PATH, streamlit_command


def test_streamlit_command_points_at_script_and_config(smoke_config: FlowBenchConfig) -> None:
    cmd = streamlit_command(smoke_config, Path("configs/smoke.yaml"))
    assert APP_PATH.is_file()
    assert cmd[1:4] == ["-m", "streamlit", "run"]
    assert cmd[4] == str(APP_PATH)
    assert str(smoke_config.ui.port) in cmd
    assert cmd[-2:] == ["--config", str(Path("configs/smoke.yaml").resolve())]
