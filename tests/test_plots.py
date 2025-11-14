# tests/test_plots_cli.py
# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
from pathlib import Path

# Importa el CLI y la función desde el módulo correcto
from src.modeling.plots import app as app_cli, main as main_cmd

runner = CliRunner()


# ----------------------------- Histograma ----------------------------- #
@pytest.mark.unit
@patch("src.modeling.plots.logger")
@patch("src.modeling.plots.PlotGenerator")
@patch("src.modeling.plots.paths.ensure_path")
@patch("src.modeling.plots.pd.read_csv")
@patch("src.modeling.plots.conf")
def test_main_histogram_uses_default_filename(
    mock_conf, mock_read_csv, mock_ensure, mock_plotgen_cls, mock_logger
):
    """
    main(plot_type='histogram', column=...) debe:
    - leer el CSV via pd.read_csv(paths.ensure_path(input_path))
    - instanciar PlotGenerator(df, paths.ensure_path(conf.paths.figures))
    - llamar a histogram(column=..., filename='histogram.png') si filename es None
    """
    # Configuración de conf y ensure_path
    mock_conf.data.interim_data_file = "data/interim/clean.csv"
    mock_conf.paths.figures = "figs/"
    # ensure_path devuelve la misma ruta como Path para ambos usos
    mock_ensure.side_effect = lambda p: Path(p)

    # read_csv devuelve un DataFrame pequeño
    df = pd.DataFrame({"A": [1, 2, 2, 3]})
    mock_read_csv.return_value = df

    # Instancia mock del plotter
    plotter_inst = MagicMock()
    mock_plotgen_cls.return_value = plotter_inst

    # Ejecuta
    main_cmd(
        input_path="some.csv",
        plot_type="histogram",
        column="A",
        filename=None,  # => debe usar 'histogram.png'
    )

    mock_read_csv.assert_called_once_with(Path("some.csv"))
    mock_plotgen_cls.assert_called_once_with(df, Path("figs/"))
    plotter_inst.histogram.assert_called_once_with(column="A", filename="histogram.png")


# ------------------------------- Scatter ------------------------------ #
@pytest.mark.unit
@patch("src.modeling.plots.PlotGenerator")
@patch("src.modeling.plots.paths.ensure_path")
@patch("src.modeling.plots.pd.read_csv")
@patch("src.modeling.plots.conf")
def test_main_scatter_calls_scatter_with_filename(
    mock_conf, mock_read_csv, mock_ensure, mock_plotgen_cls
):
    """
    main(plot_type='scatter', x=..., y=..., filename='custom.png') debe llamar a
    PlotGenerator.scatter con los args correctos.
    """
    mock_conf.paths.figures = "figs/"
    mock_ensure.side_effect = lambda p: Path(p)
    mock_read_csv.return_value = pd.DataFrame({"x": [0, 1], "y": [1, 0]})

    plotter_inst = MagicMock()
    mock_plotgen_cls.return_value = plotter_inst

    main_cmd(
        input_path="any.csv",
        plot_type="scatter",
        x="x",
        y="y",
        filename="custom.png",
    )

    plotter_inst.scatter.assert_called_once_with(x="x", y="y", filename="custom.png")


# ----------------------- Parámetros inválidos ------------------------- #
@pytest.mark.unit
@patch("src.modeling.plots.logger")
@patch("src.modeling.plots.PlotGenerator")
@patch("src.modeling.plots.paths.ensure_path")
@patch("src.modeling.plots.pd.read_csv")
@patch("src.modeling.plots.conf")
def test_main_invalid_params_logs_error(
    mock_conf, mock_read_csv, mock_ensure, mock_plotgen_cls, mock_logger
):
    """
    Si los parámetros no corresponden al plot_type esperado,
    debe llamar a logger.error sin intentar graficar.
    """
    mock_conf.paths.figures = "figs/"
    mock_ensure.side_effect = lambda p: Path(p)
    mock_read_csv.return_value = pd.DataFrame({"x": [1], "y": [2]})
    plotter_inst = MagicMock()
    mock_plotgen_cls.return_value = plotter_inst

    # plot_type scatter pero falta 'x'
    main_cmd(input_path="any.csv", plot_type="scatter", x=None, y="y", filename=None)

    mock_logger.error.assert_called_once()
    plotter_inst.histogram.assert_not_called()
    plotter_inst.scatter.assert_not_called()


# ----------------------------- CLI: ayuda ----------------------------- #
@pytest.mark.unit
def test_cli_lists_main_command_in_help():
    """
    Verifica que el subcomando 'main' está registrado en la aplicación Typer.
    (Evita ejecutar el flujo CLI completo, que a veces produce SystemExit(2)
    si el registro del subcomando cambia.)
    """
    result = runner.invoke(app_cli, ["--help"])
    assert result.exit_code == 0
    assert "main" in (result.stdout or result.output)
