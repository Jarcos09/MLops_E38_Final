# tests/test_app_mlflow.py
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.api import app_mlflow as api_mod
from src.config.config import conf


@pytest.mark.unit
@patch("src.api.app_mlflow.DriftDetection")
@patch("src.api.app_mlflow.joblib")
@patch("src.api.app_mlflow.DatasetCleaner")
@patch("src.api.app_mlflow.MLFlowClient")                
@patch("src.api.app_mlflow.ModelPredictor")
@patch("src.api.app_mlflow.mlflow_launcher.ensure_mlflow_server")
def test_predict_ok_uses_cleaner_and_preprocessor(
    mock_ensure_mlflow,
    mock_model_predictor_cls,
    mock_mlflow_client_cls,          
    mock_cleaner_cls,
    mock_joblib,
    mock_drift_cls,
):
    # ---------- predictor global ----------
    predictor_inst = MagicMock()
    predictor_inst.predict.return_value = pd.DataFrame(
        [{"target_0": 1.0, "target_1": 2.0}]
    )
    # load_model devuelve un objeto "modelo" ficticio (para DriftDetection)
    predictor_inst.load_model.return_value = MagicMock()
    mock_model_predictor_cls.return_value = predictor_inst

    # ---------- MLFlowClient (evitar llamadas reales) ----------
    ml_client_inst = MagicMock()
    ml_client_inst.check_remote_available.return_value = None
    ml_client_inst.get_latest_version.return_value = {"version": "1"}
    mock_mlflow_client_cls.return_value = ml_client_inst

    # ---------- DatasetCleaner ----------
    cleaner_inst = MagicMock()
    cleaner_inst.run.side_effect = lambda: cleaner_inst.input_path
    mock_cleaner_cls.return_value = cleaner_inst

    # ---------- Preprocesador ----------
    preproc = MagicMock()
    preproc.transform.side_effect = lambda X: X
    preproc.get_feature_names_out.side_effect = (
        lambda cols=None: list(cols) if cols is not None else []
    )
    mock_joblib.load.return_value = preproc

    # ---------- DriftDetection ----------
    drift_inst = MagicMock()
    drift_inst.run.return_value = None
    mock_drift_cls.return_value = drift_inst

    # ---------- Payload válido ----------
    feature_cols = list(conf.preprocessing.feature_columns)
    instance = {col: 1.0 for col in feature_cols}

    payload = {
        "instances": [instance],
        "model_type": "rf",
        # PredictionRequest espera string; usamos la misma versión por defecto
        "model_version": conf.prediction.use_version,
    }

    # ---------- Llamada al endpoint ----------
    with TestClient(api_mod.app) as client:
        resp = client.post("/predict", json=payload)

    assert resp.status_code == 200
    body = resp.json()

    assert "predictions" in body
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["target_0"] == 1.0
    assert body["predictions"][0]["target_1"] == 2.0
    assert "data_drift" in body
    assert body["data_drift"]["detected"] is False
