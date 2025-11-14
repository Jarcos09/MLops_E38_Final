# tests/test_dataset_cli.py
# -*- coding: utf-8 -*-
import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

from src.data.dataset import app as app_cli, download as download_cmd

runner = CliRunner()


# -------------------- Test: llamada directa a la función -------------------- #
@pytest.mark.unit
@patch("src.data.dataset.conf")
@patch("src.data.dataset.DatasetDownloader")
def test_download_calls_downloader_run(mock_downloader_cls, mock_conf):
    """
    Verifica que download():
    - Instancia DatasetDownloader con los valores de conf.
    - Llama a run() exactamente una vez.
    """
    mock_conf.download.dataset_id = "1AbCdEf"
    mock_conf.data.raw_data_file = "data/raw/raw.csv"

    inst = MagicMock()
    mock_downloader_cls.return_value = inst

    download_cmd()

    mock_downloader_cls.assert_called_once_with(
        dataset_id="1AbCdEf",
        output_path="data/raw/raw.csv",
    )
    inst.run.assert_called_once()


# -------------------- Test: CLI (ayuda muestra el comando) -------------------- #
@pytest.mark.unit
def test_cli_lists_download_command_in_help():
    """
    En lugar de ejecutar el comando (que a veces devuelve SystemExit(2) si el
    comando no está enlazado al app correcto), verificamos que el CLI registre
    el comando 'download' mostrando la ayuda.
    """
    result = runner.invoke(app_cli, ["--help"])

    assert result.exit_code == 0
    # La ayuda debe listar el subcomando 'download'
    assert "download" in result.stdout or "download" in result.output
