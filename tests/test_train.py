# tests/test_train_cli.py
# -*- coding: utf-8 -*-
import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import patch, MagicMock
import pandas as pd

from src.modeling.train import app as app_cli, train as train_cmd

runner = CliRunner()


@pytest.mark.unit
@patch("src.modeling.train.ModelTrainer")
@patch("src.modeling.train.paths")
@patch("src.modeling.train.pd.read_csv")
@patch("src.modeling.train.conf")
def test_train_invokes_modeltrainer_correctly(
    mock_conf, mock_read_csv, mock_paths, mock_trainer_cls
):
    """
    Prueba que el comando train:
    - crea directorios con paths.ensure_path
    - carga los 4 datasets con pandas.read_csv
    - inicializa ModelTrainer con la config esperada
    - llama a train_random_forest() y train_xgboost()
    """

    # --- configuración simulada ---
    mock_conf.paths.models = "models/"
    mock_conf.training.rf_model_path = "models/rf"
    mock_conf.training.xgb_model_path = "models/xgb"

    mock_conf.data.processed_data.x_train_file = "data/processed/X_train.csv"
    mock_conf.data.processed_data.x_test_file  = "data/processed/X_test.csv"
    mock_conf.data.processed_data.y_train_file = "data/processed/y_train.csv"
    mock_conf.data.processed_data.y_test_file  = "data/processed/y_test.csv"

    mock_conf.training.mlflow_tracking_uri = "http://mlflow:5000"
    mock_conf.training.random_state = 42
    mock_conf.training.rf_experiment_name = "exp_rf"
    mock_conf.training.rf_registry_model_name = "RFReg"
    mock_conf.training.rf_model_file = "rf.pkl"
    mock_conf.training.xgb_experiment_name = "exp_xgb"
    mock_conf.training.xgb_registry_model_name = "XGBReg"
    mock_conf.training.xgb_model_file = "xgb.pkl"

    # --- mocks funcionales ---
    mock_paths.ensure_path.side_effect = lambda p: Path(p)
    mock_read_csv.side_effect = [
        pd.DataFrame({"f1": [1]}),
        pd.DataFrame({"f1": [2]}),
        pd.DataFrame({"t": [1]}),
        pd.DataFrame({"t": [2]}),
    ]

    trainer_instance = MagicMock()
    mock_trainer_cls.return_value = trainer_instance

    # --- ejecutar la función ---
    train_cmd()

    # --- asserts ---
    mock_paths.ensure_path.assert_any_call("models/")
    mock_paths.ensure_path.assert_any_call("models/rf")
    mock_paths.ensure_path.assert_any_call("models/xgb")

    assert mock_read_csv.call_count == 4

    cfg_used = mock_trainer_cls.call_args.kwargs["config"]
    assert cfg_used == {
        "mlflow_tracking_uri": "http://mlflow:5000",
        "random_state": 42,
        "rf_experiment_name": "exp_rf",
        "rf_registry_model_name": "RFReg",
        "rf_model_path": "models/rf",
        "rf_model_file": "rf.pkl",
        "xgb_experiment_name": "exp_xgb",
        "xgb_registry_model_name": "XGBReg",
        "xgb_model_path": "models/xgb",
        "xgb_model_file": "xgb.pkl",
    }

    trainer_instance.train_random_forest.assert_called_once()
    trainer_instance.train_xgboost.assert_called_once()


@pytest.mark.unit
def test_cli_shows_train_in_help():
    """Verifica que el CLI de Typer tenga el comando 'train'"""
    result = runner.invoke(app_cli, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout
