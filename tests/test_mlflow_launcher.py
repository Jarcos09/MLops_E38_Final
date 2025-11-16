# tests/test_mlflow_launcher.py
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from src.utils import mlflow_launcher as ml_mod


# ---------------------------------------------------------------------
# _is_process_alive
# ---------------------------------------------------------------------
@pytest.mark.unit
@patch("src.utils.mlflow_launcher.os.kill")
def test_is_process_alive_true(mock_kill):
    assert ml_mod._is_process_alive(1234) is True
    mock_kill.assert_called_once_with(1234, 0)


@pytest.mark.unit
@patch("src.utils.mlflow_launcher.os.kill", side_effect=OSError)
def test_is_process_alive_false(mock_kill):
    assert ml_mod._is_process_alive(1234) is False
    mock_kill.assert_called_once_with(1234, 0)


# ---------------------------------------------------------------------
# is_mlflow_available
# ---------------------------------------------------------------------
@pytest.mark.unit
def test_is_mlflow_available_none():
    assert ml_mod.is_mlflow_available(None) is False


@pytest.mark.unit
@patch("src.utils.mlflow_launcher.requests.post")
def test_is_mlflow_available_file_uri(mock_post):
    # Para file:// no debe llamar a requests.post
    assert ml_mod.is_mlflow_available("file:/tmp/mlruns") is True
    mock_post.assert_not_called()


@pytest.mark.unit
@patch("src.utils.mlflow_launcher.requests.post")
def test_is_mlflow_available_http_ok(mock_post):
    resp = MagicMock(status_code=200)
    mock_post.return_value = resp

    assert ml_mod.is_mlflow_available("http://localhost:5000") is True
    mock_post.assert_called_once()


@pytest.mark.unit
@patch("src.utils.mlflow_launcher.requests.post")
def test_is_mlflow_available_http_5xx(mock_post):
    mock_post.return_value = MagicMock(status_code=500)
    assert ml_mod.is_mlflow_available("http://localhost:5000") is False


@pytest.mark.unit
@patch("src.utils.mlflow_launcher.requests.post", side_effect=Exception("boom"))
def test_is_mlflow_available_http_exception(mock_post):
    assert ml_mod.is_mlflow_available("http://localhost:5000") is False


# ---------------------------------------------------------------------
# ensure_mlflow_server
# ---------------------------------------------------------------------
@pytest.mark.unit
@patch("src.utils.mlflow_launcher.is_mlflow_available")
def test_ensure_server_non_http_returns_none(mock_avail):
    res = ml_mod.ensure_mlflow_server("file:/tmp/mlruns", pid_file="dummy.pid")
    assert res is None
    mock_avail.assert_not_called()


@pytest.mark.unit
@patch("src.utils.mlflow_launcher.subprocess.Popen")
@patch("src.utils.mlflow_launcher.os.path.exists", return_value=False)
@patch("src.utils.mlflow_launcher.is_mlflow_available", return_value=True)
def test_ensure_server_http_already_available(mock_avail, mock_exists, mock_popen):
    res = ml_mod.ensure_mlflow_server("http://localhost:5000", pid_file="dummy.pid")
    assert res is None
    mock_popen.assert_not_called()


@pytest.mark.unit
@patch("src.utils.mlflow_launcher._is_process_alive", return_value=True)
@patch("src.utils.mlflow_launcher.os.path.exists", return_value=True)
@patch("src.utils.mlflow_launcher.open", new_callable=mock_open, read_data="1234")
@patch("src.utils.mlflow_launcher.is_mlflow_available", return_value=False)
@patch("src.utils.mlflow_launcher.subprocess.Popen")
def test_ensure_server_pidfile_alive(
    mock_popen, mock_avail, mock_open_file, mock_exists, mock_alive
):
    res = ml_mod.ensure_mlflow_server("http://localhost:5000", pid_file="dummy.pid")
    assert res is None
    mock_popen.assert_not_called()


@pytest.mark.unit
@patch("src.utils.mlflow_launcher.time.sleep", return_value=None)
@patch("src.utils.mlflow_launcher.subprocess.Popen")
@patch("src.utils.mlflow_launcher.os.path.exists", return_value=False)
def test_ensure_server_starts_process(mock_exists, mock_popen, mock_sleep):
    # is_mlflow_available: False al principio, True dentro del loop
    with patch(
        "src.utils.mlflow_launcher.is_mlflow_available",
        side_effect=[False, True],
    ) as mock_avail:
        proc = MagicMock(pid=4321)
        mock_popen.return_value = proc

        pid_file = "test_mlflow.pid"
        # asegurarnos de limpiar si se creara (aunque está mockeado open en el módulo)
        with patch("src.utils.mlflow_launcher.open", mock_open()) as m_open:
            res = ml_mod.ensure_mlflow_server(
                "http://localhost:5000", pid_file=pid_file
            )

        assert res is proc
        mock_popen.assert_called_once()
        # se escribió el pid en el archivo
        m_open.assert_called()
        handle = m_open()
        handle.write.assert_called_with(str(proc.pid))
        assert mock_avail.call_count >= 2


@pytest.mark.unit
@patch("src.utils.mlflow_launcher.subprocess.Popen", side_effect=FileNotFoundError)
@patch("src.utils.mlflow_launcher.os.path.exists", return_value=False)
def test_ensure_server_handles_missing_binary(mock_exists, mock_popen):
    res = ml_mod.ensure_mlflow_server("http://localhost:5000", pid_file="dummy.pid")
    assert res is None


@pytest.mark.unit
@patch("src.utils.mlflow_launcher.subprocess.Popen")
@patch("src.utils.mlflow_launcher.os.path.exists", return_value=False)
@patch("src.utils.mlflow_launcher.is_mlflow_available", return_value=False)
def test_ensure_server_no_localhost_no_start(mock_avail, mock_exists, mock_popen):
    res = ml_mod.ensure_mlflow_server("http://remote-host:5000", pid_file="dummy.pid")
    assert res is None
    mock_popen.assert_not_called()


# ---------------------------------------------------------------------
# stop_mlflow_server
# ---------------------------------------------------------------------
@pytest.mark.unit
@patch("src.utils.mlflow_launcher.os.remove")
@patch("src.utils.mlflow_launcher.os.kill")
@patch("src.utils.mlflow_launcher.open", new_callable=mock_open, read_data="9999")
@patch("src.utils.mlflow_launcher.os.path.exists", return_value=True)
def test_stop_mlflow_server_kills_and_removes(
    mock_exists, mock_open_file, mock_kill, mock_remove
):
    pid_file = "test_mlflow.pid"
    ml_mod.stop_mlflow_server(pid_file=pid_file)

    mock_open_file.assert_called_once_with(pid_file, "r")
    mock_kill.assert_called_once()  # se intenta matar el proceso
    mock_remove.assert_called_once_with(pid_file)


@pytest.mark.unit
@patch("src.utils.mlflow_launcher.os.path.exists", return_value=False)
def test_stop_mlflow_server_no_pidfile(mock_exists):
    # No debe lanzar excepción si no existe el archivo
    ml_mod.stop_mlflow_server(pid_file="no_such_file.pid")
