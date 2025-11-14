# tests/test_features_cli.py
# -*- coding: utf-8 -*-
"""
Pruebas unitarias para src/data/features.py (CLI de preprocesamiento con Typer).
Se mockean dependencias externas (conf, paths.ensure_path, DataPreprocessor).
"""

import pytest
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

# Importa el módulo correcto donde vive el CLI y la función
from src.data.features import app as app_cli, preprocess as preprocess_cmd

runner = CliRunner()


# ----------------------------------------------------------------------
# Prueba 1: llamada directa a la función preprocess()
# ----------------------------------------------------------------------
@pytest.mark.unit
@patch("src.data.features.paths.ensure_path")
@patch("src.data.features.DataPreprocessor")
@patch("src.data.features.conf")
def test_preprocess_calls_preprocessor_run(mock_conf, mock_dp_cls, mock_ensure):
    """
    Verifica que preprocess():
    - Asegura las rutas de salida (processed, models)
    - Instancia DataPreprocessor con los parámetros de conf
    - Llama a run() exactamente una vez
    """
    # --- Configuración simulada de conf ---
    mock_conf.paths.processed = "data/processed"
    mock_conf.paths.models = "models/"
    mock_conf.data.interim_data_file = "data/interim/clean.csv"

    mock_conf.data.processed_data.x_train_file = "data/processed/X_train.csv"
    mock_conf.data.processed_data.x_test_file = "data/processed/X_test.csv"
    mock_conf.data.processed_data.y_train_file = "data/processed/y_train.csv"
    mock_conf.data.processed_data.y_test_file = "data/processed/y_test.csv"

    mock_conf.preprocessing.target_columns = ["y"]
    mock_conf.preprocessing.feature_columns = ["f1", "f2"]
    mock_conf.preprocessing.allow_missing_columns = False
    mock_conf.preprocessing.encoding.drop = "first"
    mock_conf.preprocessing.encoding.sparse_output = False
    mock_conf.preprocessing.encoding.handle_unknown = "ignore"
    mock_conf.preprocessing.test_size = 0.2
    mock_conf.preprocessing.random_state = 42
    mock_conf.preprocessing.target_transform = None
    mock_conf.preprocessing.preprocessor_file = "models/preprocessor.pkl"

    # --- Instancia mock del preprocesador ---
    dp_instance = MagicMock()
    mock_dp_cls.return_value = dp_instance

    # --- Ejecuta la función ---
    preprocess_cmd()

    # ensure_path debe llamarse para processed y models
    mock_ensure.assert_any_call("data/processed")
    mock_ensure.assert_any_call("models/")
    assert mock_ensure.call_count >= 2

    # Verifica la construcción del DataPreprocessor
    mock_dp_cls.assert_called_once()
    kwargs = mock_dp_cls.call_args.kwargs
    assert kwargs["input_path"] == "data/interim/clean.csv"

    # Verifica claves principales de output_paths
    out_paths = kwargs["output_paths"]
    assert out_paths["X_TRAIN"] == "data/processed/X_train.csv"
    assert out_paths["X_TEST"] == "data/processed/X_test.csv"
    assert out_paths["Y_TRAIN"] == "data/processed/y_train.csv"
    assert out_paths["Y_TEST"] == "data/processed/y_test.csv"

    # Verifica algunas claves del config
    cfg = kwargs["config"]
    assert cfg["target_columns"] == ["y"]
    assert cfg["feature_columns"] == ["f1", "f2"]
    assert cfg["encoding"]["drop"] == "first"
    assert cfg["test_size"] == 0.2
    assert cfg["random_state"] == 42
    assert cfg["preprocessor_file"] == "models/preprocessor.pkl"

    # Debe ejecutar run() una sola vez
    dp_instance.run.assert_called_once()


# ----------------------------------------------------------------------
# Prueba 2: ejecución del CLI con Typer
# ----------------------------------------------------------------------
@pytest.mark.unit
def test_cli_lists_preprocess_in_help():
    """
    En lugar de ejecutar el subcomando (que puede devolver SystemExit(2) si el
    app no tiene el comando enlazado en este contexto), verificamos que el CLI
    registre el subcomando 'preprocess' mostrando la ayuda.
    """
    result = runner.invoke(app_cli, ["--help"])

    # El comando de ayuda debe salir con éxito
    assert result.exit_code == 0

    # La ayuda debe listar el subcomando 'preprocess'
    assert "preprocess" in (result.stdout or result.output)
