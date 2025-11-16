# tests/test_app_docker.py
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.api.app_docker import predict as predict_endpoint
from src.api.schemas import PredictionRequest
from src.config.config import conf


@pytest.mark.unit
@patch("src.api.app_docker.DriftDetection")
@patch("src.api.app_docker.joblib")
@patch("src.api.app_docker.DatasetCleaner")
def test_predict_ok(
    mock_cleaner_cls,
    mock_joblib,
    mock_drift_cls,
):
    """
    La función predict() debe devolver un dict con 'predictions' cuando todo funciona.
    Se mockean DatasetCleaner, joblib y DriftDetection para no depender de archivos reales.
    """

    # ------------------ Predictor global ------------------
    import src.api.app_docker as api_mod

    api_mod.predictor = MagicMock()
    api_mod.predictor.predict.return_value = pd.DataFrame(
        [{"target_0": 1.0, "target_1": 2.0}]
    )

    # ------------------ DatasetCleaner --------------------
    cleaner_inst = MagicMock()
    cleaner_inst.run.side_effect = lambda: cleaner_inst.input_path
    mock_cleaner_cls.return_value = cleaner_inst

    # ------------------ Preprocesador ---------------------
    preproc = MagicMock()
    preproc.transform.side_effect = lambda X: X
    mock_joblib.load.return_value = preproc

    # ------------------ DriftDetection --------------------
    drift_inst = MagicMock()
    drift_inst.run.return_value = None
    mock_drift_cls.return_value = drift_inst

    # ------------------ Construimos el request ------------
    feature_cols = list(conf.preprocessing.feature_columns)
    instance = {col: 1.0 for col in feature_cols}

    req = PredictionRequest(
        instances=[instance],
        model_type="rf",
        model_version="1",
    )

    # ------------------ Llamada directa al endpoint -------
    result = predict_endpoint(req)

    # ------------------ Asserts ---------------------------
    assert "predictions" in result
    assert result["predictions"] == [{"target_0": 1.0, "target_1": 2.0}]
    assert "data_drift" in result
    # como mockeamos DriftDetection.run -> None, esperamos que no detecte drift
    assert result["data_drift"]["detected"] is False
