# tests/test_predict_cli.py
# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

from src.modeling.predict import app as app_cli, predict as predict_cmd


runner = CliRunner()


# ---------------------------- Caso 1: Registry OK ---------------------------- #
@pytest.mark.unit
@patch("src.modeling.predict.ModelPredictor")
@patch("src.modeling.predict.MLFlowClient")
@patch("src.modeling.predict.paths")
@patch("src.modeling.predict.pd.read_csv")
@patch("src.modeling.predict.conf")
def test_predict_uses_registry_uris_when_available(
    mock_conf, mock_read_csv, mock_paths, mock_mlclient_cls, mock_predictor_cls
):
    """
    Si el Registry remoto está disponible y devuelve versiones, se deben usar
    URIs de modelo de MLflow (models:/...) en la config del ModelPredictor.
    """
    # ---- conf ----
    mock_conf.paths.prediction = "preds/"
    mock_conf.data.processed_data.x_test_file = "data/processed/X_test.csv"
    mock_conf.training.mlflow_tracking_uri = "http://mlflow:5000"
    mock_conf.training.rf_registry_model_name = "RFRegressor"
    mock_conf.training.xgb_registry_model_name = "XGBRegressor"
    mock_conf.prediction.use_model = "rf"
    mock_conf.data.prediction_file = "preds/out.csv"

    # ---- paths ----
    # ensure_path debe funcionar tanto para prediction dir como para figures si se usara
    mock_paths.ensure_path.side_effect = lambda p: Path(p)
    # build_model_registry_uri construye la URI esperada
    mock_paths.build_model_registry_uri.side_effect = (
        lambda name, ver: f"models:/{name}/{ver}"
    )

    # ---- MLFlowClient ----
    mlc_inst = MagicMock()
    mlc_inst.check_remote_available.return_value = None    # no lanza
    mlc_inst.get_latest_version.side_effect = [
        {"version": "7"},   # RF
        {"version": "3"},   # XGB
    ]
    mock_mlclient_cls.return_value = mlc_inst

    # ---- X_new ----
    df_x = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    mock_read_csv.return_value = df_x

    # ---- Predictor ----
    pred_inst = MagicMock()
    mock_predictor_cls.return_value = pred_inst

    # Ejecutar
    predict_cmd()

    # Debe haberse leído X_test
    mock_read_csv.assert_called_once_with("data/processed/X_test.csv")

    # Se deben haber construido URIs del registry
    mock_paths.build_model_registry_uri.assert_any_call("RFRegressor", "7")
    mock_paths.build_model_registry_uri.assert_any_call("XGBRegressor", "3")

    # Verificar construcción de ModelPredictor con URIs de models:/
    cfg_used = mock_predictor_cls.call_args.kwargs["config"]
    assert cfg_used["mlflow_tracking_uri"] == "http://mlflow:5000"
    assert cfg_used["rf_model_file_path"] == "models:/RFRegressor/7"
    assert cfg_used["xgb_model_file_path"] == "models:/XGBRegressor/3"
    assert cfg_used["use_model"] == "rf"
    assert cfg_used["output_file"] == "preds/out.csv"

    # run_prediction se llama con el DataFrame leído
    pred_inst.run_prediction.assert_called_once()
    args, _ = pred_inst.run_prediction.call_args
    assert args[0] is df_x

    # En este camino NO se debe consultar get_latest_version_path
    assert not mock_paths.get_latest_version_path.called


# ------------------------ Caso 2: Fallback a archivos locales ------------------------ #
@pytest.mark.unit
@patch("src.modeling.predict.ModelPredictor")
@patch("src.modeling.predict.MLFlowClient")
@patch("src.modeling.predict.paths")
@patch("src.modeling.predict.pd.read_csv")
@patch("src.modeling.predict.requests")
@patch("src.modeling.predict.conf")
def test_predict_fallbacks_to_local_paths_when_registry_unavailable(
    mock_conf, mock_requests, mock_read_csv, mock_paths, mock_mlclient_cls, mock_predictor_cls
):
    """
    Si el Registry remoto no está disponible, se deben usar rutas locales:
    get_latest_version_path(base_path) / file_name
    """
    # ---- conf ----
    mock_conf.paths.prediction = "preds/"
    mock_conf.data.processed_data.x_test_file = "data/processed/X_test.csv"
    mock_conf.training.mlflow_tracking_uri = "http://mlflow:5000"
    mock_conf.training.rf_model_path = Path("models/rf")
    mock_conf.training.xgb_model_path = Path("models/xgb")
    mock_conf.training.rf_model_file = "rf.pkl"
    mock_conf.training.xgb_model_file = "xgb.pkl"
    mock_conf.prediction.use_model = "xgb"
    mock_conf.data.prediction_file = "preds/out.csv"

    # ---- paths ----
    mock_paths.ensure_path.side_effect = lambda p: Path(p)
    # get_latest_version_path devuelve una carpeta concreta por tipo
    def _latest(p):
        return Path(str(p)) / "exp_42"
    mock_paths.get_latest_version_path.side_effect = _latest

    # ---- MLFlowClient & requests ----
    mlc_inst = MagicMock()
    mock_mlclient_cls.return_value = mlc_inst
    # Simulamos indisponibilidad del remoto levantando RequestException en check_remote_available
    mock_requests.exceptions.RequestException = Exception
    mlc_inst.check_remote_available.side_effect = mock_requests.exceptions.RequestException()

    # ---- X_new ----
    df_x = pd.DataFrame({"a": [10, 20]})
    mock_read_csv.return_value = df_x

    # ---- Predictor ----
    pred_inst = MagicMock()
    mock_predictor_cls.return_value = pred_inst

    # Ejecutar
    predict_cmd()

    # Debe haber usado paths.get_latest_version_path para RF y XGB
    assert mock_paths.get_latest_version_path.call_count == 2

    # Config final debe contener rutas locales combinadas
    cfg_used = mock_predictor_cls.call_args.kwargs["config"]
    assert cfg_used["rf_model_file_path"] == Path("models/rf/exp_42") / "rf.pkl"
    assert cfg_used["xgb_model_file_path"] == Path("models/xgb/exp_42") / "xgb.pkl"
    assert cfg_used["use_model"] == "xgb"
    assert cfg_used["output_file"] == "preds/out.csv"

    pred_inst.run_prediction.assert_called_once()
    args, _ = pred_inst.run_prediction.call_args
    assert args[0] is df_x


# --------------------------- CLI: ayuda lista comando --------------------------- #
@pytest.mark.unit
def test_cli_lists_predict_in_help():
    result = runner.invoke(app_cli, ["--help"])
    assert result.exit_code == 0
    assert "predict" in (result.stdout or result.output)
