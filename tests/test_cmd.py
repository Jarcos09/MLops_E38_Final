# tests/test_cmd.py
import pytest
from unittest.mock import patch, MagicMock

from src.utils import cmd as cmd_mod


@pytest.mark.unit
@patch("src.utils.cmd.subprocess.run")
def test_run_cmd_ok(mock_run):
    """run_cmd debe llamar a subprocess.run con check=True y no hacer exit."""
    mock_run.return_value = MagicMock()

    cmd_mod.run_cmd(["echo", "hola"])

    mock_run.assert_called_once_with(["echo", "hola"], check=True)


@pytest.mark.unit
@patch("src.utils.cmd.sys.exit")
@patch("src.utils.cmd.subprocess.run")
def test_run_cmd_error_calls_sys_exit(mock_run, mock_exit):
    """Si subprocess.run lanza CalledProcessError, run_cmd debe llamar a sys.exit(1)."""
    from subprocess import CalledProcessError

    mock_run.side_effect = CalledProcessError(returncode=1, cmd=["bad"])

    cmd_mod.run_cmd(["bad"])  # no levanta excepción porque la manejamos internamente

    mock_exit.assert_called_once_with(1)


@pytest.mark.unit
@patch("src.utils.cmd.subprocess.run")
def test_run_cmd_output_ok(mock_run):
    """run_cmd_output debe devolver stdout.strip() en caso de éxito."""
    mock_proc = MagicMock()
    mock_proc.stdout = "salida de prueba \n"
    mock_run.return_value = mock_proc

    out = cmd_mod.run_cmd_output(["echo", "hola"])

    mock_run.assert_called_once_with(
        ["echo", "hola"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert out == "salida de prueba"


@pytest.mark.unit
@patch("src.utils.cmd.sys.exit")
@patch("src.utils.cmd.subprocess.run")
def test_run_cmd_output_error_calls_sys_exit(mock_run, mock_exit):
    """Si subprocess.run falla, run_cmd_output debe llamar a sys.exit(1)."""
    from subprocess import CalledProcessError

    mock_run.side_effect = CalledProcessError(returncode=2, cmd=["bad"])

    out = cmd_mod.run_cmd_output(["bad"])

    mock_exit.assert_called_once_with(1)
    # por diseño, out será None cuando se llama sys.exit; no lo usamos pero lo comprobamos
    assert out is None
