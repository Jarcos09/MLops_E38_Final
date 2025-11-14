import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from src.data import cleaning

runner = CliRunner()

@pytest.mark.unit
@patch("src.data.cleaning.paths.ensure_path")
@patch("src.data.cleaning.DatasetCleaner")
@patch("src.data.cleaning.conf")
def test_cli_invokes_main_ok(mock_conf, mock_dc, mock_ensure):
    """
    Verifica que el CLI de Typer invoque el flujo correcto sin tocar el FS:
    - se asegura la ruta de salida
    - se instancia DatasetCleaner con parámetros de conf
    - se llama a run() del cleaner
    """
    # Configuración simulada
    mock_conf.paths.interim = "data/interim"
    mock_conf.data.raw_data_file = "data/raw/raw.csv"
    mock_conf.data.interim_data_file = "data/interim/clean.csv"
    mock_conf.cleaning.skew_threshold = 0.5

    # Instancia mock del cleaner
    cleaner_instance = MagicMock()
    mock_dc.return_value = cleaner_instance

    # Ejecuta el CLI (equivale a `python cleaning.py`)
    result = runner.invoke(cleaning.app, [])

    # Debe terminar sin error
    assert result.exit_code == 0

    # Aserciones de llamadas
    mock_ensure.assert_called_once_with("data/interim")
    mock_dc.assert_called_once_with(
        input_path="data/raw/raw.csv",
        output_path="data/interim/clean.csv",
        skew_threshold=0.5,
    )
    cleaner_instance.run.assert_called_once()
